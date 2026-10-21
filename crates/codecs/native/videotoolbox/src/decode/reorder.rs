//! Puts decoded frames back in presentation order. VideoToolbox emits them in
//! decode order.

use cutlass_core::Timestamp;

/// No H.264 or HEVC stream holds back more frames than this (MaxDpbFrames in
/// H.264 A.3.1, MaxDpbSize in HEVC A.4.2).
const MAX_REORDER: usize = 16;

/// Releases a frame once no later packet can show before it.
///
/// Every packet's decode time is at or before its presentation time, and
/// decode times rise. So once a packet with decode time `d` has gone in, any
/// frame showing at or before `d` is safe to release. Without decode times
/// (MKV doesn't store them), frames wait until more than [`MAX_REORDER`] are
/// held.
pub(super) struct Reorder<T> {
    pending: Vec<(Timestamp, T)>,
    latest_dts: Option<Timestamp>,
    missing_dts: bool,
    draining: bool,
}

impl<T> Reorder<T> {
    pub(super) fn new() -> Self {
        Self {
            pending: Vec::with_capacity(MAX_REORDER + 1),
            latest_dts: None,
            missing_dts: false,
            draining: false,
        }
    }

    /// Call before decoding each packet.
    pub(super) fn packet_sent(&mut self, dts: Option<Timestamp>) {
        self.draining = false;
        match dts {
            Some(dts) => self.latest_dts = self.latest_dts.max(Some(dts)),
            None => self.missing_dts = true,
        }
    }

    pub(super) fn push(&mut self, pts: Timestamp, frame: T) {
        self.pending.push((pts, frame));
    }

    /// The earliest held frame, if it is safe to release.
    pub(super) fn pop(&mut self) -> Option<T> {
        let (index, &(pts, _)) = self
            .pending
            .iter()
            .enumerate()
            .min_by_key(|(_, (pts, _))| *pts)?;
        let ready = self.draining
            || self.pending.len() > MAX_REORDER
            || (!self.missing_dts && self.latest_dts.is_some_and(|dts| pts <= dts));
        ready.then(|| self.pending.swap_remove(index).1)
    }

    /// Releases everything held, at the end of the stream.
    pub(super) fn drain(&mut self) {
        self.draining = true;
    }

    pub(super) fn clear(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;

    fn ts(value: i64) -> Timestamp {
        Timestamp::new(value, NonZeroU32::new(1).unwrap())
    }

    /// Feeds `(pts, dts)` packets in decode order, as if each decoded at once,
    /// and records what comes out after each one.
    fn run(packets: &[(i64, Option<i64>)]) -> Vec<Vec<i64>> {
        let mut reorder = Reorder::new();
        let mut released = Vec::new();
        for &(pts, dts) in packets {
            reorder.packet_sent(dts.map(ts));
            reorder.push(ts(pts), pts);
            released.push(std::iter::from_fn(|| reorder.pop()).collect());
        }
        reorder.drain();
        released.push(std::iter::from_fn(|| reorder.pop()).collect());
        released
    }

    #[test]
    fn b_frames_come_out_as_soon_as_decode_time_passes_them() {
        // I P B B with two frames of decode delay, like x264 with B-frames.
        let out = run(&[(0, Some(-2)), (3, Some(-1)), (1, Some(0)), (2, Some(1))]);
        assert_eq!(out, [vec![], vec![], vec![0], vec![1], vec![2, 3]]);
    }

    #[test]
    fn frames_without_reordering_come_out_immediately() {
        let out = run(&[(0, Some(0)), (1, Some(1)), (2, Some(2))]);
        assert_eq!(out, [vec![0], vec![1], vec![2], vec![]]);
    }

    #[test]
    fn without_decode_times_holds_up_to_the_largest_reorder_depth() {
        let packets: Vec<_> = (0..20).map(|pts| (pts, None)).collect();
        let out = run(&packets);
        assert!(out[..MAX_REORDER].iter().all(Vec::is_empty));
        assert_eq!(out[MAX_REORDER], [0]);
        assert_eq!(out.concat(), (0..20).collect::<Vec<_>>());
    }

    #[test]
    fn clear_forgets_held_frames_and_decode_times() {
        let mut reorder = Reorder::new();
        reorder.packet_sent(Some(ts(10)));
        reorder.push(ts(20), 20);
        reorder.clear();
        reorder.packet_sent(Some(ts(0)));
        reorder.push(ts(1), 1);
        assert_eq!(reorder.pop(), None);
    }
}

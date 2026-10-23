//! Puts decoded frames back in presentation order. VideoToolbox emits them in
//! decode order.

use cutlass_core::Timestamp;

/// No H.264 or HEVC stream holds back more frames than this (MaxDpbFrames in
/// H.264 A.3.1, MaxDpbSize in HEVC A.4.2).
const MAX_REORDER: usize = 16;

/// Releases a frame once no later packet can show before it.
///
/// Decode times rise. A frame can still be shown before its own decode time
/// when the file uses B-frames and decode time starts at zero. `offset` is
/// the smallest presentation-minus-decode gap in the track, and it is never
/// positive. Once a packet with decode time `d` has gone in, any frame
/// showing at or before `d + offset` is safe to release. Without decode
/// times, frames wait until more than [`MAX_REORDER`] are held.
pub(super) struct Reorder<T> {
    pending: Vec<(Timestamp, T)>,
    latest_dts: Option<Timestamp>,
    /// Presentation time minus decode time, in the track timescale. `0` when
    /// every frame is shown at or after it is decoded.
    offset: i64,
    missing_dts: bool,
    draining: bool,
}

impl<T> Reorder<T> {
    pub(super) fn new(offset: i64) -> Self {
        Self {
            pending: Vec::with_capacity(MAX_REORDER + 1),
            latest_dts: None,
            offset: offset.min(0),
            missing_dts: false,
            draining: false,
        }
    }

    /// Call before decoding each packet.
    pub(super) fn packet_sent(&mut self, dts: Option<Timestamp>) {
        self.draining = false;
        match dts {
            Some(dts) => {
                if self.latest_dts.is_some_and(|last| dts <= last) {
                    self.missing_dts = true;
                }
                self.latest_dts = self.latest_dts.max(Some(dts));
            }
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
            || (!self.missing_dts
                && self.latest_dts.is_some_and(|dts| {
                    let limit =
                        Timestamp::new(dts.value.saturating_add(self.offset), dts.timescale);
                    pts <= limit
                }));
        ready.then(|| self.pending.swap_remove(index).1)
    }

    /// Releases everything held, at the end of the stream.
    pub(super) fn drain(&mut self) {
        self.draining = true;
    }

    pub(super) fn clear(&mut self) {
        let offset = self.offset;
        *self = Self::new(offset);
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
        let mut reorder = Reorder::new(0);
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
        let mut reorder = Reorder::new(0);
        reorder.packet_sent(Some(ts(10)));
        reorder.push(ts(20), 20);
        reorder.clear();
        reorder.packet_sent(Some(ts(0)));
        reorder.push(ts(1), 1);
        assert_eq!(reorder.pop(), None);
    }

    #[test]
    fn a_negative_composition_offset_holds_a_b_frame() {
        let mut reorder = Reorder::new(-40);
        let mut released = Vec::new();
        for &(pts, dts) in &[(0, 0), (80, 20), (40, 40), (20, 60), (60, 80)] {
            reorder.packet_sent(Some(ts(dts)));
            reorder.push(ts(pts), pts);
            released.extend(std::iter::from_fn(|| reorder.pop()));
        }
        assert_eq!(released, [0, 20, 40]);
        reorder.drain();
        released.extend(std::iter::from_fn(|| reorder.pop()));
        assert_eq!(released, [0, 20, 40, 60, 80]);
    }
}

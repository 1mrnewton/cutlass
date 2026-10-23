//! The packet a seek starts decoding from.

use cutlass_core::{Packet, Timestamp};

/// The keyframe to decode so the sample at `time` can be produced.
///
/// `samples` is in decode order. The result is the keyframe the latest sample
/// at or before `time` depends on. A keyframe's presentation time can be later
/// than that sample. When `time` is before every sample, the result is the
/// first keyframe. `None` when `samples` has no keyframe that can produce a
/// sample at `time`.
pub fn keyframe_before(samples: &[Packet], time: Timestamp) -> Option<usize> {
    let mut target = None;
    for (index, sample) in samples.iter().enumerate() {
        if sample.pts > time {
            continue;
        }
        if target.is_none_or(|(_, pts)| sample.pts >= pts) {
            target = Some((index, sample.pts));
        }
    }
    let Some((index, _)) = target else {
        return samples.iter().position(|sample| sample.keyframe);
    };
    samples[..=index].iter().rposition(|sample| sample.keyframe)
}

/// How a decoder that has already taken `packets[..cursor]` reaches `time`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reach {
    /// `time` is still in the group of pictures the decoder is inside.
    Forward,
    /// Reset and start at this keyframe. Packets before it are already late,
    /// or the playhead moved backward.
    Keyframe(usize),
}

/// The next read for `time`.
///
/// `cursor` is the first packet the decoder has not taken. `shown` is the
/// oldest frame still queued, and `drained` means the decoder has finished
/// the packets it was given. A backward move restarts once that queue already
/// holds a frame later than `time`, or the decoder has finished and the frame
/// is gone. A backward move that is still ahead of the queued frames stays on
/// the open group. A playhead that has reached a later keyframe restarts too.
/// The same group keeps going, including when the playhead is more than a
/// second ahead. `None` when no keyframe can produce `time`.
pub fn reach(
    packets: &[Packet],
    cursor: usize,
    followed: Option<Timestamp>,
    shown: Option<Timestamp>,
    drained: bool,
    time: Timestamp,
) -> Option<Reach> {
    let key = keyframe_before(packets, time)?;
    let cursor = cursor.min(packets.len());
    let backward = followed.is_some_and(|previous| time < previous);
    let passed = shown.is_some_and(|pts| pts > time) || (shown.is_none() && drained);
    if key >= cursor || (backward && passed) {
        Some(Reach::Keyframe(key))
    } else {
        Some(Reach::Forward)
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use cutlass_core::{Packet, Timestamp};

    use super::keyframe_before;

    fn packet(pts: i64, keyframe: bool) -> Packet {
        Packet {
            data: Vec::new(),
            pts: Timestamp::new(pts, NonZeroU32::new(1_000).expect("timescale")),
            dts: None,
            duration: None,
            keyframe,
        }
    }

    fn at(ms: i64) -> Timestamp {
        Timestamp::new(ms, NonZeroU32::new(1_000).expect("timescale"))
    }

    #[test]
    fn a_delta_frame_starts_at_the_earlier_keyframe() {
        let samples = [packet(200, true), packet(100, false), packet(300, false)];
        assert_eq!(keyframe_before(&samples, at(100)), Some(0));
        assert_eq!(keyframe_before(&samples, at(300)), Some(0));
    }

    #[test]
    fn a_later_keyframe_is_preferred() {
        let samples = [
            packet(0, true),
            packet(100, false),
            packet(200, true),
            packet(300, false),
        ];
        assert_eq!(keyframe_before(&samples, at(100)), Some(0));
        assert_eq!(keyframe_before(&samples, at(300)), Some(2));
    }

    #[test]
    fn a_playhead_before_the_first_sample_uses_the_first_keyframe() {
        let samples = [packet(200, true), packet(100, false)];
        assert_eq!(keyframe_before(&samples, at(0)), Some(0));
    }

    #[test]
    fn no_keyframe_has_no_start() {
        let samples = [packet(0, false)];
        assert_eq!(keyframe_before(&samples, at(0)), None);
        assert_eq!(keyframe_before(&[], at(0)), None);
        assert_eq!(super::reach(&samples, 0, None, None, false, at(0)), None);
    }

    #[test]
    fn the_same_group_of_pictures_keeps_going() {
        let samples = [
            packet(0, true),
            packet(10, false),
            packet(20, false),
            packet(1_000, false),
        ];
        assert_eq!(
            super::reach(&samples, 2, Some(at(0)), None, false, at(1_000)),
            Some(super::Reach::Forward)
        );
    }

    #[test]
    fn a_later_keyframe_restarts_even_when_the_gap_is_short() {
        let samples = [
            packet(0, true),
            packet(10, false),
            packet(20, true),
            packet(30, false),
        ];
        assert_eq!(
            super::reach(&samples, 0, None, None, false, at(0)),
            Some(super::Reach::Keyframe(0))
        );
        assert_eq!(
            super::reach(&samples, 2, Some(at(0)), None, false, at(30)),
            Some(super::Reach::Keyframe(2))
        );
    }

    #[test]
    fn a_backward_move_restarts_once_a_later_frame_is_queued() {
        let samples = [packet(0, true), packet(10, false), packet(20, true)];
        assert_eq!(
            super::reach(&samples, 2, Some(at(20)), Some(at(20)), false, at(10)),
            Some(super::Reach::Keyframe(0))
        );
    }

    #[test]
    fn a_backward_move_inside_the_open_group_keeps_going() {
        let samples = [packet(0, true), packet(10, false), packet(20, true)];
        assert_eq!(
            super::reach(&samples, 2, Some(at(20)), None, false, at(10)),
            Some(super::Reach::Forward)
        );
    }

    #[test]
    fn a_finished_decoder_restarts_when_the_frame_is_gone() {
        let samples = [packet(0, true), packet(10, false), packet(20, true)];
        assert_eq!(
            super::reach(&samples, 2, Some(at(20)), None, true, at(10)),
            Some(super::Reach::Keyframe(0))
        );
    }
}

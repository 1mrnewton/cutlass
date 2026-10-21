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
    }
}

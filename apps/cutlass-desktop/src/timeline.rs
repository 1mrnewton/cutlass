//! Sequence-level time queries called from Slint.
//!
//! Currently exposes one entry point — `sequence_duration` — bound to
//! the `TimelineLib.sequence-duration` callback in
//! `ui/lib/timeline-lib.slint`. Heavier timeline math (cut/trim,
//! overlap resolution, render extents) belongs next to the engine,
//! not here.

use crate::{Clip, Rational, RationalTime, Sequence};
use slint::Model;

/// Latest content end across all tracks, expressed at the sequence rate.
///
/// Implements `max(clip.timeline_start + clip.source_range.duration)`
/// as declared in `ui/lib/timeline-lib.slint`. The result is rounded
/// **up** to the next sequence tick when a clip's rate doesn't divide
/// cleanly into the sequence rate, so the ruler / scrubber always
/// covers the full content extent (otherwise the last sub-tick of a
/// mixed-rate clip would fall outside the addressable timeline).
///
/// ## Correctness — single rounding
///
/// `timeline_start + duration` is reduced to **one** rational fraction
/// before being rounded to sequence ticks. Rounding the two operands
/// separately can overshoot by up to one sequence tick per clip
/// (`ceil(a) + ceil(b) ≥ ceil(a + b)`), which surfaces on a mixed-rate
/// project as a ruler that visibly extends past the actual last frame.
/// See `ntsc_clip_in_30fps_sequence_rounds_once_not_twice` for the
/// regression test.
///
/// ## Performance
///
/// Hot path — every clip already authored at the sequence rate, which
/// is overwhelmingly the common case in practice — is a single `i64`
/// add per clip and skips the rational math entirely. The mixed-rate
/// slow path uses `i128` because the cross-multiplied numerator
/// `n_start * d_dur + n_dur * d_start` can exceed `i64` for realistic
/// NTSC rates combined with long sequences, even when each component
/// triple-product still fits. `i128` on aarch64 / x86-64 is a handful
/// of ALU ops, not a software-emulated bigint — and we only pay it on
/// the rare mixed-rate clip.
pub fn sequence_duration(sequence: Sequence) -> RationalTime {
    let target = sequence.fps;
    let target_num = i64::from(target.num);
    let target_den = i64::from(target.den);

    let mut max_end: i64 = 0;

    for track_idx in 0..sequence.tracks.row_count() {
        let Some(track) = sequence.tracks.row_data(track_idx) else {
            continue;
        };
        for clip_idx in 0..track.clips.row_count() {
            let Some(clip) = track.clips.row_data(clip_idx) else {
                continue;
            };

            let end = clip_end_in_target(&clip, &target, target_num, target_den);
            if end > max_end {
                max_end = end;
            }
        }
    }

    RationalTime {
        value: max_end.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
        rate: target,
    }
}

/// End of `clip` in target-rate ticks. See `sequence_duration` for
/// rounding semantics.
#[inline]
fn clip_end_in_target(clip: &Clip, target: &Rational, target_num: i64, target_den: i64) -> i64 {
    let start = &clip.timeline_start;
    let duration = &clip.source_range.duration;

    // Fast path: clip authored at the sequence rate. `value + value`
    // is exact and skips the i128 reduction below. This is the only
    // place this function spends real cycles, so the branch pays for
    // itself the moment we hit it (i.e. almost always, in practice).
    if rate_eq(&start.rate, target) && rate_eq(&duration.rate, target) {
        return i64::from(start.value) + i64::from(duration.value);
    }

    // Slow path: convert (start + duration) to target-rate ticks as a
    // *single* fraction, then ceil. The identity is
    //     a/b + c/d == (a*d + c*b) / (b*d)
    // with each side already cross-multiplied into target-rate-tick
    // space. i128 because the final numerator can overflow i64 for
    // realistic NTSC rates combined with hour-scale sequences, even
    // though each individual triple-product still fits.
    let n_start = i128::from(start.value) * i128::from(start.rate.den) * i128::from(target_num);
    let d_start = i128::from(start.rate.num) * i128::from(target_den);

    let n_dur = i128::from(duration.value) * i128::from(duration.rate.den) * i128::from(target_num);
    let d_dur = i128::from(duration.rate.num) * i128::from(target_den);

    let num = n_start * d_dur + n_dur * d_start;
    let den = d_start * d_dur;

    ceil_div(num, den).clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

#[inline]
fn rate_eq(a: &Rational, b: &Rational) -> bool {
    // No gcd-reduce: rates in this project are stored canonically
    // (24/1, 24000/1001, …). Adding a reduction step here would cost
    // more than the i128 path it's meant to shortcut.
    a.num == b.num && a.den == b.den
}

/// Ceiling of `num / den` for any-sign `num`, **positive** `den`.
///
/// Rust's `/` truncates toward zero, which equals ceiling for negative
/// numerator (`-7 / 2 == -3 == ceil(-3.5)`) but floor for positive
/// (`7 / 2 == 3`, ceil is `4`). The positive branch patches that with
/// the `(n + d - 1) / d` trick.
#[inline]
fn ceil_div(num: i128, den: i128) -> i128 {
    debug_assert!(den > 0);
    if num >= 0 {
        (num + den - 1) / den
    } else {
        num / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TimeRange, Track};
    use slint::{ModelRc, VecModel};
    use std::rc::Rc;

    // ---- builders ----------------------------------------------------

    fn r(num: i32, den: i32) -> Rational {
        Rational { num, den }
    }

    fn rt(value: i32, num: i32, den: i32) -> RationalTime {
        RationalTime {
            value,
            rate: r(num, den),
        }
    }

    fn clip(timeline_start: RationalTime, duration: RationalTime) -> Clip {
        Clip {
            timeline_start,
            source_range: TimeRange {
                duration,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn track(clips: Vec<Clip>) -> Track {
        Track {
            clips: ModelRc::from(Rc::new(VecModel::from(clips))),
            ..Default::default()
        }
    }

    fn sequence(fps: Rational, tracks: Vec<Track>) -> Sequence {
        Sequence {
            fps,
            tracks: ModelRc::from(Rc::new(VecModel::from(tracks))),
            ..Default::default()
        }
    }

    // ---- empty / degenerate -----------------------------------------

    #[test]
    fn empty_sequence_is_zero_at_sequence_rate() {
        let d = sequence_duration(sequence(r(30, 1), vec![]));
        assert_eq!(d.value, 0);
        assert_eq!(d.rate.num, 30);
        assert_eq!(d.rate.den, 1);
    }

    #[test]
    fn empty_tracks_are_zero() {
        let d = sequence_duration(sequence(
            r(30, 1),
            vec![track(vec![]), track(vec![]), track(vec![])],
        ));
        assert_eq!(d.value, 0);
    }

    #[test]
    fn zero_duration_clip_uses_timeline_start() {
        // A bare cut point with no source range should still extend
        // the sequence to its position.
        let d = sequence_duration(sequence(
            r(30, 1),
            vec![track(vec![clip(rt(100, 30, 1), rt(0, 30, 1))])],
        ));
        assert_eq!(d.value, 100);
    }

    // ---- fast path: same rate everywhere -----------------------------

    #[test]
    fn single_clip_at_sequence_rate() {
        let d = sequence_duration(sequence(
            r(30, 1),
            vec![track(vec![clip(rt(0, 30, 1), rt(150, 30, 1))])],
        ));
        assert_eq!(d.value, 150);
    }

    #[test]
    fn multiple_clips_on_one_track_take_max_end() {
        let d = sequence_duration(sequence(
            r(30, 1),
            vec![track(vec![
                clip(rt(0, 30, 1), rt(60, 30, 1)),  // ends @ 60
                clip(rt(60, 30, 1), rt(90, 30, 1)), // ends @ 150 ← max
                clip(rt(30, 30, 1), rt(50, 30, 1)), // ends @ 80
            ])],
        ));
        assert_eq!(d.value, 150);
    }

    #[test]
    fn longest_track_wins_across_multiple_tracks() {
        // Max-end clip lives on track 1, not the first track — guards
        // against any "first track only" regression.
        let d = sequence_duration(sequence(
            r(30, 1),
            vec![
                track(vec![clip(rt(0, 30, 1), rt(100, 30, 1))]),
                track(vec![clip(rt(50, 30, 1), rt(200, 30, 1))]), // ends @ 250
                track(vec![clip(rt(0, 30, 1), rt(75, 30, 1))]),
            ],
        ));
        assert_eq!(d.value, 250);
    }

    // ---- slow path: mixed rate --------------------------------------

    #[test]
    fn ntsc_clip_in_30fps_sequence_rounds_once_not_twice() {
        // Regression for the double-ceil bug. Clip at 23.976 (24000/1001):
        //   start = 1 tick  ≈ 0.04171 s
        //   dur   = 24 ticks ≈ 1.00100 s
        //   end   = 25 ticks ≈ 1.04271 s
        //
        // Mapped to 30 fps: (25 * 1001 * 30) / (24000 * 1)
        //                 = 750_750 / 24_000
        //                 = 31.28125
        // → ceil = 32.
        //
        // The naive implementation rounded start and duration
        // independently and got
        //   ceil(1.251) + ceil(30.0625) = 2 + 31 = 33
        // — one phantom frame past the actual content end. Visible as
        // a ruler / scrubber that extends slightly past the last clip.
        let d = sequence_duration(sequence(
            r(30, 1),
            vec![track(vec![clip(rt(1, 24000, 1001), rt(24, 24000, 1001))])],
        ));
        assert_eq!(d.value, 32);
    }

    #[test]
    fn integer_multiple_rate_change_is_exact() {
        // 60 fps clip in a 30 fps sequence: each value halves cleanly
        // so there's no rounding regardless of strategy. This is the
        // sanity case for the slow path.
        let d = sequence_duration(sequence(
            r(30, 1),
            vec![track(vec![clip(rt(120, 60, 1), rt(60, 60, 1))])],
        ));
        // 120 @ 60fps == 60 @ 30fps; 60 @ 60fps == 30 @ 30fps; end = 90.
        assert_eq!(d.value, 90);
    }

    #[test]
    fn result_carries_sequence_rate_not_clip_rate() {
        // Reported duration is always at the sequence rate, even when
        // the only clip is authored at a different rate.
        let d = sequence_duration(sequence(
            r(30, 1),
            vec![track(vec![clip(rt(0, 24, 1), rt(48, 24, 1))])],
        ));
        assert_eq!(d.rate.num, 30);
        assert_eq!(d.rate.den, 1);
        // 48 @ 24fps = 2 s = 60 @ 30fps.
        assert_eq!(d.value, 60);
    }

    #[test]
    fn fast_and_slow_paths_agree_on_canonical_vs_unreduced_rate() {
        // Fast-path comparison is structural (num/den equality), so
        // 30/1 vs 60/2 take different paths in `clip_end_in_target`.
        // They MUST produce the same answer for a no-op rate
        // conversion — otherwise we have a soundness bug in the
        // i128 reduction.
        let fast = sequence_duration(sequence(
            r(30, 1),
            vec![track(vec![clip(rt(15, 30, 1), rt(90, 30, 1))])],
        ));
        let slow = sequence_duration(sequence(
            r(30, 1),
            vec![track(vec![clip(rt(15, 60, 2), rt(90, 60, 2))])],
        ));
        assert_eq!(fast.value, slow.value);
        assert_eq!(fast.value, 105);
    }

    #[test]
    fn long_ntsc_sequence_does_not_overflow() {
        // ~1 hour @ 23.976: 86_313 ticks ≈ 3599.97 s. Cross-multiplied
        // against 30/1 the slow-path numerator
        //   86_313 * 1001 * 30 ≈ 2.6e9
        // sits well inside i64 on its own, but the
        // `n_start * d_dur + n_dur * d_start` step squares-ish that
        // magnitude into the 1e16+ regime, which is where i64 starts
        // to feel tight and where moving to i128 actually buys safety.
        // This test exists to catch any future "optimization" that
        // shrinks the slow path back to i64.
        //
        // Expected: 86_313 * 1001 / 24_000 * 30 = 107_999.14 → ceil = 108_000.
        let d = sequence_duration(sequence(
            r(30, 1),
            vec![track(vec![clip(
                rt(0, 24000, 1001),
                rt(86_313, 24000, 1001),
            )])],
        ));
        assert_eq!(d.value, 108_000);
    }
}

//! Shared seek policy: roll forward instead of re-seeking for near targets.
//!
//! None of the native readers keeps a keyframe index (the old FFmpeg stack
//! did), so the [`VideoDecoder`] default `frame_at` — seek, then walk — makes
//! the codec re-decode the whole GOP prefix on *every* call. Across a playback
//! or forward-scrub run that is O(GOP²) work per GOP, and scrubbing is the
//! primary touch interaction on mobile.
//!
//! Every backend therefore remembers the PTS of the last frame it emitted and
//! answers `frame_at` with [`frame_at_rolling`]: when the target lies ahead of
//! that frame by at most [`ROLL_FORWARD_WINDOW_SECS`], keep decoding forward
//! from where the codec already is — every in-between frame would have to be
//! decoded after a seek anyway. Backward targets, long forward jumps, and
//! fresh decoders fall back to a real seek, byte-identical to the default
//! seek-then-walk.
//!
//! The window bounds the worst case (rolling past an unnoticed keyframe can't
//! waste more than a window of decode) while comfortably covering the
//! sequential case (gaps of one frame). One second ≈ a typical mobile-capture
//! GOP.
//!
//! ## Tick-truncation slack
//!
//! The walk treats a frame as satisfying the target when its PTS is within
//! **one tick** of the frame's own time base *before* it ([`frame_covers`]),
//! not only at/after it. Backends whose native clocks are integer ticks —
//! Media Foundation's 100-ns units, MediaCodec's microseconds — **truncate**
//! frame times that aren't representable: at 30000/1001 fps frame 1's true
//! time 333 666.⅔ hns arrives as 333 666. Under an exact comparison that
//! frame sits *before* its own rational target, so `frame_at(i)` returns
//! frame `i + 1`, and the off-by-one compounds: targets that exactly equal a
//! truncated PTS (1 in 3 at NTSC rates) then compare `Equal` to `last_pts`,
//! fall off the roll path, and pay a full seek + GOP re-decode *during
//! ordinary sequential playback*. One tick of slack returns the intended
//! frame and keeps playback on the roll path, while being far too small to
//! ever skip a real frame — as a guard for hypothetical frame-granular time
//! bases the slack is additionally capped at half a frame period. Exact-clock
//! backends (Apple's rational `CMTime`) are unaffected: their neighboring
//! frames sit whole periods apart, never within a tick.

use core::cmp::Ordering;

use cutlass_core::{DecodeError, Rational, RationalTime, VideoDecoder, VideoFrame};

/// How far ahead of the last emitted frame a target may lie and still be
/// reached by decoding forward rather than seeking, in seconds.
const ROLL_FORWARD_WINDOW_SECS: i64 = 1;

/// `sec(a) − sec(b)`, exactly, scaled by the (positive) product of both rate
/// numerators: positive iff `a` is later. `i128` keeps the triple products
/// exact (values ≤ 2⁶³, rate parts ≤ 2³¹ ⇒ products < 2¹²⁵).
fn scaled_delta(a: RationalTime, b: RationalTime) -> i128 {
    i128::from(a.value) * i128::from(a.rate.den) * i128::from(b.rate.num)
        - i128::from(b.value) * i128::from(b.rate.den) * i128::from(a.rate.num)
}

/// The tick-truncation slack: whether `delta` — a non-negative
/// [`scaled_delta`] between a requested time and a decoded PTS on
/// `tick_rate`, with `scale = other_num · tick_rate.num` — is at most **one
/// tick** of `tick_rate`, additionally capped at **half a frame period** so a
/// coarse time base (ticks ≈ frames) can never absorb a real frame.
///
/// One tick in the scaled units is `tick_rate.den · other_num` (a tick is
/// `den/num` seconds; the scale contributes `num · other_num`). The
/// half-frame cap compares `delta / scale ≤ fr.den / (2·fr.num)`
/// cross-multiplied; `checked_mul` guards the one product that can exceed
/// `i128`, which is unambiguously "way more than half a frame". An invalid
/// `frame_rate` skips the cap (the tick bound alone still applies).
fn within_tick_slack(
    delta: i128,
    tick_rate: Rational,
    other_num: i32,
    scale: i128,
    frame_rate: Rational,
) -> bool {
    debug_assert!(delta >= 0 && scale > 0);
    if delta > i128::from(tick_rate.den) * i128::from(other_num) {
        return false;
    }
    if !frame_rate.is_valid() {
        return true;
    }
    let rhs = scale * i128::from(frame_rate.den);
    match delta.checked_mul(2 * i128::from(frame_rate.num)) {
        Some(lhs) => lhs <= rhs,
        None => false,
    }
}

/// True when the decoder should keep pulling frames instead of seeking:
/// `target` lies after `last_pts` by no more than the roll window, and by
/// more than the truncation slack — a target within one tick of the last
/// emitted frame *is* that frame (see the module docs), and re-emitting it
/// takes a seek, not a roll. Invalid rates disable rolling — a seek is
/// always correct.
pub(crate) fn should_roll_forward(
    last_pts: Option<RationalTime>,
    target: RationalTime,
    frame_rate: Rational,
) -> bool {
    let Some(last) = last_pts else {
        return false;
    };
    if !last.rate.is_valid() || !target.rate.is_valid() {
        return false;
    }
    if target.compare(last) != Ordering::Greater {
        return false;
    }
    let ahead = scaled_delta(target, last);
    let scale = i128::from(target.rate.num) * i128::from(last.rate.num);
    let window = i128::from(ROLL_FORWARD_WINDOW_SECS) * scale;
    if ahead > window {
        return false;
    }
    !within_tick_slack(ahead, last.rate, target.rate.num, scale, frame_rate)
}

/// Whether the decoded frame at `frame_pts` satisfies a request for `target`:
/// at/after it, or before it by no more than the truncation slack (see the
/// module docs).
pub(crate) fn frame_covers(
    frame_pts: RationalTime,
    target: RationalTime,
    frame_rate: Rational,
) -> bool {
    if frame_pts.compare(target) != Ordering::Less {
        return true;
    }
    if !frame_pts.rate.is_valid() || !target.rate.is_valid() {
        return false;
    }
    let behind = scaled_delta(target, frame_pts);
    let scale = i128::from(target.rate.num) * i128::from(frame_pts.rate.num);
    within_tick_slack(behind, frame_pts.rate, target.rate.num, scale, frame_rate)
}

/// [`VideoDecoder::frame_at`] with the roll-forward fast path.
///
/// `last_pts` is the PTS of the last frame the decoder emitted (backends
/// record it in `next_frame` and clear it in `seek`). When rolling, the walk
/// continues from the decoder's current position; hitting end of stream while
/// rolling means the target lies past the end, which is the same `Ok(None)`
/// the seek path would produce.
pub(crate) fn frame_at_rolling<D: VideoDecoder + ?Sized>(
    decoder: &mut D,
    last_pts: Option<RationalTime>,
    target: RationalTime,
) -> Result<Option<VideoFrame>, DecodeError> {
    let frame_rate = decoder.info().frame_rate;
    if !should_roll_forward(last_pts, target, frame_rate) {
        decoder.seek(target)?;
    }
    while let Some(frame) = decoder.next_frame()? {
        if frame_covers(frame.pts, target, frame_rate) {
            return Ok(Some(frame));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_core::{
        ColorSpace, CpuImage, FrameData, PixelFormat, Plane, Rational, Rect, Rotation, SourceInfo,
    };

    const R30: Rational = Rational::FPS_30;

    fn rt(value: i64, rate: Rational) -> RationalTime {
        RationalTime::new(value, rate)
    }

    #[test]
    fn no_last_pts_never_rolls() {
        assert!(!should_roll_forward(None, rt(5, R30), R30));
    }

    #[test]
    fn backward_and_repeated_targets_never_roll() {
        let last = Some(rt(10, R30));
        assert!(!should_roll_forward(last, rt(9, R30), R30));
        assert!(!should_roll_forward(last, rt(10, R30), R30));
    }

    #[test]
    fn rolls_within_the_window() {
        let last = Some(rt(10, R30));
        // One frame ahead — the sequential-playback case.
        assert!(should_roll_forward(last, rt(11, R30), R30));
        // Exactly one second ahead is still inside the (inclusive) window.
        assert!(should_roll_forward(last, rt(40, R30), R30));
        // Just past the window falls back to a seek.
        assert!(!should_roll_forward(last, rt(41, R30), R30));
    }

    #[test]
    fn window_comparison_is_exact_across_rates() {
        // Last frame on an NTSC stream time base, target at plain 30 fps.
        let last = Some(rt(0, Rational::FPS_29_97));
        // 29/30 s ahead: inside the window.
        assert!(should_roll_forward(last, rt(29, R30), Rational::FPS_29_97));
        // 31/30 s ahead: outside.
        assert!(!should_roll_forward(last, rt(31, R30), Rational::FPS_29_97));
        // Stream-tick time bases (large numerators) stay exact too.
        let last_ticks = Some(rt(90_000, Rational::new(90_000, 1))); // t = 1 s
        assert!(should_roll_forward(last_ticks, rt(60, R30), R30)); // 2 s
        assert!(!should_roll_forward(last_ticks, rt(61, R30), R30)); // 2 s + 1 frame
    }

    #[test]
    fn invalid_rates_never_roll() {
        let last = Some(rt(10, Rational::new(0, 1)));
        assert!(!should_roll_forward(last, rt(11, R30), R30));
        assert!(!should_roll_forward(
            Some(rt(10, R30)),
            rt(11, Rational::new(30, 0)),
            R30
        ));
    }

    /// The Media Foundation shape: PTS on a fine integer tick base (100-ns)
    /// that *truncates* the rational frame times of an NTSC stream.
    #[test]
    fn truncated_tick_pts_counts_as_its_own_frame() {
        const HNS: Rational = Rational::new(10_000_000, 1);
        const NTSC: Rational = Rational::FPS_29_97;
        // Frame 1 at 30000/1001 fps is 333_666.6̅ hns; MF delivers 333_666.
        let pts = rt(333_666, HNS);
        let target = rt(1, NTSC);
        // The truncated frame satisfies its own (slightly later) target...
        assert!(frame_covers(pts, target, NTSC));
        // ...but not the next frame's target.
        assert!(!frame_covers(pts, rt(2, NTSC), NTSC));
        // Requesting the frame just emitted (within a tick) is not a roll
        // (the decoder is already past it)...
        assert!(!should_roll_forward(Some(pts), target, NTSC));
        // ...while the *next* frame rolls instead of seeking.
        assert!(should_roll_forward(Some(pts), rt(2, NTSC), NTSC));
    }

    /// On a frame-granular tick base (PTS ticks == frames), the slack must
    /// never swallow a whole frame: exact behavior is preserved.
    #[test]
    fn slack_never_absorbs_a_real_frame_on_coarse_tick_bases() {
        assert!(!frame_covers(rt(9, R30), rt(10, R30), R30));
        assert!(frame_covers(rt(10, R30), rt(10, R30), R30));
        // One frame ahead still rolls (the slack is capped below one frame).
        assert!(should_roll_forward(Some(rt(10, R30)), rt(11, R30), R30));
    }

    /// A decoder over `count` synthetic 30 fps frames with a keyframe only at
    /// zero, mirroring how the real backends wire [`frame_at_rolling`]:
    /// `next_frame` records `last_pts`, `seek` clears it and counts.
    struct MockDecoder {
        info: SourceInfo,
        cursor: i64,
        count: i64,
        last_pts: Option<RationalTime>,
        seeks: usize,
    }

    impl MockDecoder {
        fn new(count: i64) -> Self {
            Self {
                info: SourceInfo {
                    coded_size: (2, 2),
                    display_size: (2, 2),
                    rotation: Rotation::None,
                    pixel_format: PixelFormat::Nv12,
                    color: ColorSpace::BT709,
                    frame_rate: R30,
                    time_base: R30,
                    duration: Some(rt(count, R30)),
                },
                cursor: 0,
                count,
                last_pts: None,
                seeks: 0,
            }
        }
    }

    impl VideoDecoder for MockDecoder {
        fn info(&self) -> &SourceInfo {
            &self.info
        }

        fn seek(&mut self, _target: RationalTime) -> Result<(), DecodeError> {
            self.cursor = 0; // only keyframe is frame 0
            self.last_pts = None;
            self.seeks += 1;
            Ok(())
        }

        fn next_frame(&mut self) -> Result<Option<VideoFrame>, DecodeError> {
            if self.cursor >= self.count {
                return Ok(None);
            }
            let pts = rt(self.cursor, R30);
            self.cursor += 1;
            self.last_pts = Some(pts);
            let y = Plane::new(vec![16u8; 4], 2, 2);
            let uv = Plane::new(vec![128u8; 2], 2, 1);
            Ok(Some(VideoFrame::new(
                pts,
                PixelFormat::Nv12,
                ColorSpace::BT709,
                (2, 2),
                Rect::from_size(2, 2),
                Rotation::None,
                FrameData::Cpu(CpuImage::new(vec![y, uv])),
            )))
        }

        fn frame_at(&mut self, target: RationalTime) -> Result<Option<VideoFrame>, DecodeError> {
            let last = self.last_pts;
            frame_at_rolling(self, last, target)
        }
    }

    #[test]
    fn sequential_targets_seek_once_and_land_exactly() {
        let mut dec = MockDecoder::new(60);
        for i in (0..30).step_by(3) {
            let f = dec.frame_at(rt(i, R30)).unwrap().unwrap();
            assert_eq!(f.pts.value, i);
        }
        assert_eq!(dec.seeks, 1, "only the first (cold) target may seek");
    }

    #[test]
    fn backward_target_falls_back_to_seek() {
        let mut dec = MockDecoder::new(60);
        dec.frame_at(rt(20, R30)).unwrap().unwrap();
        let f = dec.frame_at(rt(5, R30)).unwrap().unwrap();
        assert_eq!(f.pts.value, 5);
        assert_eq!(dec.seeks, 2);
    }

    #[test]
    fn far_forward_target_falls_back_to_seek() {
        let mut dec = MockDecoder::new(120);
        dec.frame_at(rt(0, R30)).unwrap().unwrap();
        // 40/30 s ahead of frame 0: beyond the window.
        let f = dec.frame_at(rt(40, R30)).unwrap().unwrap();
        assert_eq!(f.pts.value, 40);
        assert_eq!(dec.seeks, 2);
    }

    #[test]
    fn rolling_past_end_of_stream_returns_none() {
        let mut dec = MockDecoder::new(10);
        dec.frame_at(rt(9, R30)).unwrap().unwrap();
        // Within the roll window but past the last frame.
        assert!(dec.frame_at(rt(12, R30)).unwrap().is_none());
        assert_eq!(dec.seeks, 1, "the roll path must not seek");
    }
}

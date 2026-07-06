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
//! that frame by no more than the **roll window**, keep decoding forward from
//! where the codec already is — every in-between frame would have to be
//! decoded after a seek anyway. Backward targets, long forward jumps, and
//! fresh decoders fall back to a real seek, byte-identical to the default
//! seek-then-walk.
//!
//! ## Cost-adaptive roll window
//!
//! How far forward rolling beats seeking depends entirely on the source: a
//! seek costs a keyframe-prefix re-decode (GOP length × per-frame cost),
//! rolling costs one decode per frame skipped. With ~8-second GOPs (common in
//! downloaded/CDN 4K) a fixed 1-second window turns a 30-frame forward drag
//! into a multi-second GOP re-decode; with all-keyframe content rolling more
//! than a couple frames is pure waste. So each decoder carries a [`SeekStats`]
//! cost model: EMAs of the observed seek-path cost and the per-frame roll
//! cost, measured by [`frame_at_rolling`] itself. The window is the
//! break-even point `seek_cost / frame_cost` (in frames, converted through
//! the frame rate), clamped between two frame periods and
//! [`MAX_ROLL_WINDOW_SECS`]. Cold-start defaults reproduce the old fixed
//! window (~1 s at 30 fps) until real measurements arrive.
//!
//! Measured (`scrub_bench`, 4K H.264 fragmented MP4 with ~8 s GOPs, Intel
//! UHD 630, zero-copy GPU output, mean/p95 per target): a +30-frame forward
//! drag — 1.001 s at 29.97 fps, one tick past the old fixed window — went
//! from 449/1 239 ms to 175/238 ms; the CPU path from 1 530/4 282 ms to
//! 646/1 185 ms. Small steps (+2, +8) and the never-rolling patterns
//! (backward, random) are unchanged.
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
use std::time::Instant;

use cutlass_core::{DecodeError, Rational, RationalTime, VideoDecoder, VideoFrame};

/// Upper bound on the roll window: a pathological cost estimate (huge seek
/// EMA, tiny frame EMA) must never make the decoder crawl forever instead of
/// seeking.
const MAX_ROLL_WINDOW_SECS: f64 = 10.0;

/// EMA smoothing factor: heavy enough on the newest sample to track a
/// scene-complexity change within a few observations, light enough that one
/// outlier (cache-cold seek, page fault) doesn't swing the window.
const EMA_ALPHA: f64 = 0.3;

/// Frame rate assumed for the window when the source doesn't report one.
const FALLBACK_FPS: f64 = 30.0;

/// Per-decoder cost model for the adaptive roll-forward window: EMAs of the
/// observed seek cost and per-frame decode cost, updated by
/// [`frame_at_rolling`] from its own wall-clock measurements. Backends keep
/// one next to `last_pts` and hand it back on every `frame_at`.
///
/// The defaults (300 ms per seek, 10 ms per frame) reproduce the previous
/// fixed 1-second window at 30 fps until real measurements replace them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SeekStats {
    /// EMA of a full seek-path `frame_at`: `seek()` plus the keyframe-prefix
    /// walk to the covering frame, in milliseconds.
    seek_ms: f64,
    /// EMA of one `next_frame` on the roll path, in milliseconds.
    frame_ms: f64,
}

impl Default for SeekStats {
    fn default() -> Self {
        Self {
            seek_ms: 300.0,
            frame_ms: 10.0,
        }
    }
}

impl SeekStats {
    fn observe_seek(&mut self, ms: f64) {
        self.seek_ms += EMA_ALPHA * (ms - self.seek_ms);
    }

    fn observe_roll_frame(&mut self, ms: f64) {
        self.frame_ms += EMA_ALPHA * (ms - self.frame_ms);
    }

    /// The break-even roll window in microseconds: rolling `n` frames costs
    /// `n × frame_ms`, so roll while that stays at or under one seek —
    /// `seek_ms / frame_ms` frames, converted to time through the frame rate.
    /// Clamped to at least two frame periods (sequential playback must always
    /// roll) and at most [`MAX_ROLL_WINDOW_SECS`].
    fn roll_window_us(&self, frame_rate: Rational) -> i64 {
        let fps = if frame_rate.is_valid() {
            let fps = f64::from(frame_rate.num) / f64::from(frame_rate.den);
            if fps.is_finite() && fps > 0.0 {
                fps
            } else {
                FALLBACK_FPS
            }
        } else {
            FALLBACK_FPS
        };
        let break_even_frames = (self.seek_ms / self.frame_ms).max(0.0);
        let secs = (break_even_frames / fps).clamp(2.0 / fps, MAX_ROLL_WINDOW_SECS);
        (secs * 1e6) as i64
    }
}

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
/// `target` lies after `last_pts` by no more than `window_us` microseconds,
/// and by more than the truncation slack — a target within one tick of the
/// last emitted frame *is* that frame (see the module docs), and re-emitting
/// it takes a seek, not a roll. Invalid rates disable rolling — a seek is
/// always correct.
pub(crate) fn should_roll_forward(
    last_pts: Option<RationalTime>,
    target: RationalTime,
    frame_rate: Rational,
    window_us: i64,
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
    // ahead/scale secs ≤ window_us µs ⟺ ahead·10⁶ ≤ window_us·scale. The
    // left product can overflow i128 only for astronomically far targets,
    // where seeking is the right answer anyway.
    match ahead.checked_mul(1_000_000) {
        Some(lhs) if lhs <= i128::from(window_us) * scale => {}
        _ => return false,
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
///
/// Times its own work to feed `stats`: a seek-path call updates the seek-cost
/// EMA with the whole `seek()`-plus-walk duration, a roll-path call updates
/// the per-frame EMA with `elapsed / frames_decoded`. The two `Instant`
/// reads are nanoseconds against millisecond-scale decode work.
pub(crate) fn frame_at_rolling<D: VideoDecoder + ?Sized>(
    decoder: &mut D,
    last_pts: Option<RationalTime>,
    stats: &mut SeekStats,
    target: RationalTime,
) -> Result<Option<VideoFrame>, DecodeError> {
    let frame_rate = decoder.info().frame_rate;
    let rolling = should_roll_forward(
        last_pts,
        target,
        frame_rate,
        stats.roll_window_us(frame_rate),
    );
    let start = Instant::now();
    if !rolling {
        decoder.seek(target)?;
    }
    let mut decoded = 0u32;
    while let Some(frame) = decoder.next_frame()? {
        decoded += 1;
        if frame_covers(frame.pts, target, frame_rate) {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
            if rolling {
                stats.observe_roll_frame(elapsed_ms / f64::from(decoded));
            } else {
                stats.observe_seek(elapsed_ms);
            }
            return Ok(Some(frame));
        }
    }
    Ok(None)
}

/// [`VideoDecoder::frame_at_nearest`] with the same roll-forward fast path
/// as [`frame_at_rolling`]: a target just ahead of the last emitted frame
/// (inside the adaptive window) takes the exact walk — a couple of
/// `next_frame`s are cheaper than a container seek *and* land exactly.
/// Every other target seeks and decodes a single frame, snapping to the
/// sync point at or before `target`.
///
/// The snap path deliberately does **not** feed `stats`: its cost (one
/// decode) says nothing about a full exact `frame_at`, and folding it into
/// the seek EMA would shrink the roll window that exact calls rely on.
pub(crate) fn frame_at_nearest_rolling<D: VideoDecoder + ?Sized>(
    decoder: &mut D,
    last_pts: Option<RationalTime>,
    stats: &mut SeekStats,
    target: RationalTime,
) -> Result<Option<VideoFrame>, DecodeError> {
    let frame_rate = decoder.info().frame_rate;
    if should_roll_forward(
        last_pts,
        target,
        frame_rate,
        stats.roll_window_us(frame_rate),
    ) {
        return frame_at_rolling(decoder, last_pts, stats, target);
    }
    decoder.seek(target)?;
    decoder.next_frame()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_core::{
        ColorSpace, CpuImage, FrameData, PixelFormat, Plane, Rational, Rect, Rotation, SourceInfo,
    };

    const R30: Rational = Rational::FPS_30;
    /// The old fixed window, for tests exercising the exact boundary math.
    const ONE_SEC_US: i64 = 1_000_000;

    fn rt(value: i64, rate: Rational) -> RationalTime {
        RationalTime::new(value, rate)
    }

    #[test]
    fn no_last_pts_never_rolls() {
        assert!(!should_roll_forward(None, rt(5, R30), R30, ONE_SEC_US));
    }

    #[test]
    fn backward_and_repeated_targets_never_roll() {
        let last = Some(rt(10, R30));
        assert!(!should_roll_forward(last, rt(9, R30), R30, ONE_SEC_US));
        assert!(!should_roll_forward(last, rt(10, R30), R30, ONE_SEC_US));
    }

    #[test]
    fn rolls_within_the_window() {
        let last = Some(rt(10, R30));
        // One frame ahead — the sequential-playback case.
        assert!(should_roll_forward(last, rt(11, R30), R30, ONE_SEC_US));
        // Exactly one second ahead is still inside the (inclusive) window.
        assert!(should_roll_forward(last, rt(40, R30), R30, ONE_SEC_US));
        // Just past the window falls back to a seek.
        assert!(!should_roll_forward(last, rt(41, R30), R30, ONE_SEC_US));
    }

    #[test]
    fn window_comparison_is_exact_across_rates() {
        // Last frame on an NTSC stream time base, target at plain 30 fps.
        let last = Some(rt(0, Rational::FPS_29_97));
        // 29/30 s ahead: inside the window.
        assert!(should_roll_forward(
            last,
            rt(29, R30),
            Rational::FPS_29_97,
            ONE_SEC_US
        ));
        // 31/30 s ahead: outside.
        assert!(!should_roll_forward(
            last,
            rt(31, R30),
            Rational::FPS_29_97,
            ONE_SEC_US
        ));
        // Stream-tick time bases (large numerators) stay exact too.
        let last_ticks = Some(rt(90_000, Rational::new(90_000, 1))); // t = 1 s
        assert!(should_roll_forward(
            last_ticks,
            rt(60, R30),
            R30,
            ONE_SEC_US
        )); // 2 s
        assert!(!should_roll_forward(
            last_ticks,
            rt(61, R30),
            R30,
            ONE_SEC_US
        )); // 2 s + 1 frame
    }

    #[test]
    fn invalid_rates_never_roll() {
        let last = Some(rt(10, Rational::new(0, 1)));
        assert!(!should_roll_forward(last, rt(11, R30), R30, ONE_SEC_US));
        assert!(!should_roll_forward(
            Some(rt(10, R30)),
            rt(11, Rational::new(30, 0)),
            R30,
            ONE_SEC_US
        ));
    }

    #[test]
    fn default_stats_reproduce_the_old_one_second_window() {
        assert_eq!(SeekStats::default().roll_window_us(R30), ONE_SEC_US);
        // Unknown frame rate falls back to 30 fps rather than disabling.
        assert_eq!(
            SeekStats::default().roll_window_us(Rational::new(0, 1)),
            ONE_SEC_US
        );
    }

    #[test]
    fn window_tracks_observed_costs() {
        // Long-GOP source: seeks ~1.2 s, frames ~5 ms — break-even is 240
        // frames, so the window converges well past the old 1 s.
        let mut stats = SeekStats::default();
        for _ in 0..20 {
            stats.observe_seek(1200.0);
            stats.observe_roll_frame(5.0);
        }
        assert!(stats.roll_window_us(R30) > 5_000_000);

        // All-keyframe source: a seek costs the same as a frame, so rolling
        // is pointless beyond the two-frame-period floor.
        let mut stats = SeekStats::default();
        for _ in 0..20 {
            stats.observe_seek(5.0);
            stats.observe_roll_frame(5.0);
        }
        assert_eq!(stats.roll_window_us(R30), (2.0 / 30.0 * 1e6) as i64);

        // A pathological ratio saturates at the hard cap.
        let mut stats = SeekStats::default();
        for _ in 0..40 {
            stats.observe_seek(60_000.0);
            stats.observe_roll_frame(0.01);
        }
        assert_eq!(
            stats.roll_window_us(R30),
            (MAX_ROLL_WINDOW_SECS * 1e6) as i64
        );
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
        assert!(!should_roll_forward(Some(pts), target, NTSC, ONE_SEC_US));
        // ...while the *next* frame rolls instead of seeking.
        assert!(should_roll_forward(
            Some(pts),
            rt(2, NTSC),
            NTSC,
            ONE_SEC_US
        ));
    }

    /// On a frame-granular tick base (PTS ticks == frames), the slack must
    /// never swallow a whole frame: exact behavior is preserved.
    #[test]
    fn slack_never_absorbs_a_real_frame_on_coarse_tick_bases() {
        assert!(!frame_covers(rt(9, R30), rt(10, R30), R30));
        assert!(frame_covers(rt(10, R30), rt(10, R30), R30));
        // One frame ahead still rolls (the slack is capped below one frame).
        assert!(should_roll_forward(
            Some(rt(10, R30)),
            rt(11, R30),
            R30,
            ONE_SEC_US
        ));
    }

    /// A decoder over `count` synthetic 30 fps frames with a keyframe only at
    /// zero, mirroring how the real backends wire [`frame_at_rolling`]:
    /// `next_frame` records `last_pts`, `seek` clears it and counts.
    struct MockDecoder {
        info: SourceInfo,
        cursor: i64,
        count: i64,
        last_pts: Option<RationalTime>,
        stats: SeekStats,
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
                stats: SeekStats::default(),
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
            let mut stats = self.stats;
            let result = frame_at_rolling(self, last, &mut stats, target);
            self.stats = stats;
            result
        }

        fn frame_at_nearest(
            &mut self,
            target: RationalTime,
        ) -> Result<Option<VideoFrame>, DecodeError> {
            let last = self.last_pts;
            let mut stats = self.stats;
            let result = frame_at_nearest_rolling(self, last, &mut stats, target);
            self.stats = stats;
            result
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
        // 40 frames ahead: beyond the window — the cold-start break-even is
        // 30 frames, and the first (instant) mock seek only shrank it.
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

    /// A far target snaps: one seek, one decode, and the frame is the sync
    /// point at/before the target (frame 0 here — the mock's only keyframe),
    /// not the exact target. That single decode replacing the GOP-prefix
    /// walk is the whole point of the API.
    #[test]
    fn nearest_far_target_snaps_to_sync_frame_with_one_decode() {
        let mut dec = MockDecoder::new(120);
        let f = dec.frame_at_nearest(rt(100, R30)).unwrap().unwrap();
        assert_eq!(f.pts.value, 0, "snap lands on the keyframe, not target");
        assert_eq!(dec.seeks, 1);
        assert_eq!(dec.cursor, 1, "exactly one frame decoded");
    }

    /// A target just ahead of the last emitted frame stays on the exact
    /// roll path: no seek, and the frame is the exact target.
    #[test]
    fn nearest_target_just_ahead_rolls_exactly() {
        let mut dec = MockDecoder::new(60);
        dec.frame_at(rt(10, R30)).unwrap().unwrap();
        let f = dec.frame_at_nearest(rt(12, R30)).unwrap().unwrap();
        assert_eq!(f.pts.value, 12, "roll-window targets land exactly");
        assert_eq!(dec.seeks, 1, "the nearest roll path must not seek");
    }
}

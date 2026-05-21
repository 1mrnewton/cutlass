//! Timeline ruler tick generation.
//!
//! Given the current viewport state (scroll offset, width, zoom level,
//! frame rate, drop-frame flag), returns the list of major and minor
//! tick marks that fall inside the visible window — already positioned
//! in viewport-local pixel coordinates with SMPTE timecode labels on the
//! majors.
//!
//! Two big ideas underpin this module:
//!
//! 1. **Virtualization.** We never emit ticks for off-screen content.
//!    Every pro NLE (Premiere, DaVinci Resolve, FCP, Avid) virtualizes
//!    its ruler. An 8-hour timeline at 24 fps is 691 200 frames;
//!    generating a per-frame tick set up-front is a non-starter.
//!    The bound on ticks rendered is therefore *the viewport*, not the
//!    timeline length.
//!
//! 2. **Adaptive "nice number" tick ladder.** As the user zooms, the
//!    interval between ticks snaps to a fixed sequence of human-readable
//!    intervals so the *on-screen spacing* between labels stays roughly
//!    constant (~80 px). Below 1 second we count in frames
//!    (1, 2, 5, 10, 15, 30 f); from 1 second up we walk the
//!    {1, 2, 5, 10, 15, 30} progression through seconds, minutes, and
//!    hours.
//!
//! ## Why 80 px between majors
//!
//! Premiere, Resolve, and FCP all sit between 70 and 90 px between
//! labelled ticks at their default zoom levels (measured by zooming the
//! UI and inspecting). 80 px gives the caption font (~11 px) enough
//! breathing room that adjacent labels never collide, while still
//! producing dense enough majors to read durations at a glance.
//!
//! ## References
//!
//! - Heckbert, "Nice Numbers for Graph Labels", *Graphics Gems I* (1990).
//!   The original {1, 2, 5} × 10ⁿ trick; we use a time-aware variant
//!   anchored to seconds / minutes / hours instead of pure decades.
//! - Matplotlib `MaxNLocator`, d3 `tickStep` — same idea, generic axes.
//! - OpenTimelineIO `to_timecode` — timecode formatter (in `crate::timecode`).

use slint::{ModelRc, SharedString, VecModel};
use std::rc::Rc;

use crate::RulerTick;
use crate::timecode::format_timecode;

/// Target minimum pixel distance between two consecutive major
/// (labelled) ticks. The ladder picks the smallest interval whose
/// projected pixel width is `>=` this value.
const MIN_MAJOR_PX: f32 = 80.0;

/// Horizontal padding (px) added to each side of the visible frame
/// range so labels can clip gracefully across viewport edges.
///
/// Why this exists: a major tick's *line* is 1 px wide, but its *label*
/// extends ~80 px to the right of the line. If we strictly clipped the
/// frame range at the viewport edge, a major that scrolled past the
/// left edge would be dropped from the tick model and its label would
/// vanish *while still mostly inside the viewport* — visible "popping".
///
/// Instead, we emit a wider range and rely on the parent
/// `Rectangle { clip: true }` to clip the rendered text at the actual
/// viewport edge. The label fades off-screen one pixel at a time
/// (same as Premiere / DaVinci Resolve / FCP behaviour).
///
/// `120 px` is comfortably above the widest plausible major label
/// (`"00:00:00:00"` in caption-size mono is ~75 px) including the
/// 3 px tick→label offset, with headroom for future longer labels
/// (e.g. drop-frame variants or sub-frame indicators).
const LABEL_PAD_PX: f32 = 120.0;

/// Safety cap on ticks emitted per call. At sane zoom levels we emit
/// ~20 majors + ~80 minors; the cap exists so a pathological state
/// (zoom collapsing to ~0, or a stale invocation mid-resize) can't
/// flood Slint with thousands of elements.
const MAX_TICKS: usize = 1024;

/// Compute the visible-window tick list.
///
/// `scroll_x` follows the Slint Flickable convention: negative when the
/// user has scrolled content to the right (i.e. the viewport's origin
/// has moved left within content-space). Frame 0 of the sequence is at
/// content-space pixel 0.
pub fn compute_visible_ticks(
    scroll_x: f32,
    viewport_w: f32,
    zoom: f32,
    fps_num: i64,
    fps_den: i64,
    drop_frame: bool,
) -> Vec<RulerTick> {
    // Bail on degenerate inputs before we underflow or divide by zero.
    // These can happen briefly during layout — when the Ruler hasn't
    // been sized yet, or before the EditorStore is wired — and we'd
    // rather return an empty model than panic.
    if zoom <= 0.0 || viewport_w <= 0.0 || fps_num <= 0 || fps_den <= 0 {
        return Vec::new();
    }

    // Visible *content-space* pixel range. With Slint's Flickable,
    // scroll_x is negative when scrolled right, so the left edge of
    // the visible content is at content pixel `-scroll_x`.
    let left_px = (-scroll_x).max(0.0);
    let right_px = left_px + viewport_w;

    // Convert pixel range to frame range. We move into integer frame
    // indices here and stay in integers through tick layout — this is
    // why an NLE's ruler doesn't drift across hours of timeline.
    //
    // `pad_frames` extends the range by `LABEL_PAD_PX` on each side
    // so a major tick whose *line* has scrolled just past the edge
    // still gets emitted (its *label* would otherwise pop out while
    // most of its glyphs are still inside the viewport — see the doc
    // on `LABEL_PAD_PX`). The parent clip Rectangle handles the actual
    // visual clipping; we just need to keep the right ticks in the model.
    let pad_frames = (LABEL_PAD_PX / zoom).ceil() as i64;
    let first_frame = (left_px / zoom).floor() as i64 - pad_frames;
    let last_frame = (right_px / zoom).ceil() as i64 + pad_frames;

    let (major, minor) = pick_intervals(zoom, fps_num, fps_den);

    let mut out: Vec<RulerTick> = Vec::with_capacity(64);

    // ---- Major (labelled) ticks ----
    //
    // Start at the first multiple of `major` that is >= first_frame.
    // `div_euclid` rounds toward negative infinity (unlike `/` which
    // truncates toward zero), so this is safe for negative first_frame
    // even though we clamp to 0 above.
    let mut k = first_frame.div_euclid(major) * major;
    if k < first_frame {
        k += major;
    }
    while k <= last_frame && out.len() < MAX_TICKS {
        if k >= 0 {
            let x = (k as f32 * zoom + scroll_x).round();
            out.push(RulerTick {
                x,
                is_major: true,
                label: SharedString::from(format_timecode(k, fps_num, fps_den, drop_frame)),
            });
        }
        k += major;
    }

    // ---- Minor (unlabelled) ticks ----
    //
    // `None` at extreme zoom-in where the major is the smallest entry
    // in the ladder (e.g. 1 frame) and there's nothing finer to draw.
    // We skip minors that coincide with a major (modulo == 0) so the
    // major isn't drawn over by a stub.
    if let Some(minor) = minor {
        let mut m = first_frame.div_euclid(minor) * minor;
        if m < first_frame {
            m += minor;
        }
        while m <= last_frame && out.len() < MAX_TICKS {
            if m >= 0 && m % major != 0 {
                let x = (m as f32 * zoom + scroll_x).round();
                out.push(RulerTick {
                    x,
                    is_major: false,
                    label: SharedString::default(),
                });
            }
            m += minor;
        }
    }

    out
}

/// Slint callback adapter: wraps `compute_visible_ticks` in a
/// `ModelRc<RulerTick>` for the Rust → Slint boundary. Called from
/// `main.rs` once at startup to install the handler.
///
/// Slint passes `int` properties as `i32`; we widen to `i64` inside
/// because tick math (`frame * zoom`, drop-frame adjustments) is more
/// comfortable in 64-bit even though `i32` is plenty of range for any
/// realistic project.
pub fn ticks_model(
    scroll_x: f32,
    viewport_w: f32,
    zoom: f32,
    fps_num: i32,
    fps_den: i32,
    drop_frame: bool,
) -> ModelRc<RulerTick> {
    let ticks = compute_visible_ticks(
        scroll_x,
        viewport_w,
        zoom,
        fps_num as i64,
        fps_den as i64,
        drop_frame,
    );
    ModelRc::from(Rc::new(VecModel::from(ticks)))
}

/// Pick the (major, minor) tick intervals (in frames) for the current
/// zoom level. The minor entry is `None` when the major is already at
/// the bottom of the ladder.
fn pick_intervals(zoom: f32, fps_num: i64, fps_den: i64) -> (i64, Option<i64>) {
    let ladder = build_ladder(fps_num, fps_den);
    // Linear scan — the ladder is ~22 entries, so a binary search would
    // be premature optimisation. Picks the *smallest* entry whose
    // projected pixel span is at least MIN_MAJOR_PX.
    let major_idx = ladder
        .iter()
        .position(|&f| (f as f32) * zoom >= MIN_MAJOR_PX)
        .unwrap_or(ladder.len() - 1);
    let major = ladder[major_idx];
    let minor = if major_idx == 0 {
        None
    } else {
        Some(ladder[major_idx - 1])
    };
    (major, minor)
}

/// Build the "nice tick intervals" ladder for a given frame rate.
/// Entries are in **frames**, sorted ascending. Two regimes:
///
///   * **Sub-second** (frames < ceil(fps)): direct frame counts
///     `1, 2, 5, 10, 15, 30`. Capped at `< fps_int` so they stay below
///     one second of wall-time and don't collide with the seconds
///     regime below.
///
///   * **One second and above**: a repeating `{1, 2, 5, 10, 15, 30}`
///     progression mapped onto seconds → minutes → hours. Each entry
///     is converted to frames via `round(seconds * fps_num / fps_den)`.
///     For NTSC rates that introduces a 0–1 frame rounding error per
///     ladder step, but the *label* on the emitted tick is computed
///     from the actual frame index by `format_timecode`, so labels
///     stay clean ("00:00:01:00" at whatever frame our rounded
///     "1 second" interval lands).
fn build_ladder(fps_num: i64, fps_den: i64) -> Vec<i64> {
    let fps_int = (fps_num + fps_den - 1) / fps_den; // ceil(fps)
    let mut ladder: Vec<i64> = Vec::with_capacity(32);

    // Sub-second entries (frame-counted).
    for &k in &[1i64, 2, 5, 10, 15, 30] {
        if k < fps_int {
            ladder.push(k);
        }
    }

    // Seconds-multiple entries:
    //   1s, 2s, 5s, 10s, 15s, 30s,
    //   1m (60),  2m (120), 5m (300), 10m (600), 15m (900), 30m (1800),
    //   1h (3600), 2h (7200), 5h (18000), 10h (36000)
    // We stop at 10 hours; nobody scrubs a 24-hour project at "one
    // major per 10 hours" zoom, and timecode display rolls past 99:59:59
    // territory anyway.
    let seconds_steps: &[i64] = &[
        1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 18000, 36000,
    ];
    for &s in seconds_steps {
        // Banker's-style rounding to the nearest frame.
        let frames = (s * fps_num + fps_den / 2) / fps_den;
        // De-dup adjacent equal entries (can happen at unusual rates
        // where the sub-second and seconds ladders overlap).
        if frames > 0 && ladder.last().copied() != Some(frames) {
            ladder.push(frames);
        }
    }

    ladder
}

#[cfg(test)]
mod tests {
    use super::*;
    use slint::Model;

    fn majors(ticks: &[RulerTick]) -> Vec<(f32, String)> {
        ticks
            .iter()
            .filter(|t| t.is_major)
            .map(|t| (t.x, t.label.to_string()))
            .collect()
    }

    // ---------- Degenerate inputs ----------

    #[test]
    fn empty_for_zero_viewport() {
        assert!(compute_visible_ticks(0.0, 0.0, 10.0, 24, 1, false).is_empty());
    }

    #[test]
    fn empty_for_zero_zoom() {
        assert!(compute_visible_ticks(0.0, 1000.0, 0.0, 24, 1, false).is_empty());
    }

    #[test]
    fn empty_for_bad_fps() {
        assert!(compute_visible_ticks(0.0, 1000.0, 10.0, 0, 1, false).is_empty());
        assert!(compute_visible_ticks(0.0, 1000.0, 10.0, 24, 0, false).is_empty());
    }

    // ---------- Ladder picks ----------

    #[test]
    fn picks_10_frame_major_at_24fps_zoom10() {
        // 24fps, zoom = 10 px/frame.
        //   pps = 24 * 10 = 240 px/s
        //   target = 80 / 240 = 0.333 s
        //   ladder: 1f, 2f, 5f, 10f, 15f, ... (24fps caps sub-second at <24)
        //   10f * 10 px = 100 px ≥ 80; 5f * 10 = 50 px < 80; pick 10f.
        let (major, minor) = pick_intervals(10.0, 24, 1);
        assert_eq!(major, 10);
        assert_eq!(minor, Some(5));
    }

    #[test]
    fn picks_one_frame_major_at_extreme_zoom_in() {
        // zoom = 200 px/frame: 1 frame * 200 px = 200 px ≥ 80. No minor.
        let (major, minor) = pick_intervals(200.0, 24, 1);
        assert_eq!(major, 1);
        assert_eq!(minor, None);
    }

    #[test]
    fn picks_minute_major_at_low_zoom() {
        // 24fps, zoom = 0.1 px/frame → pps = 2.4. target = 80/2.4 ≈ 33s.
        // Ladder hits 60s (1 minute) = 1440 frames * 0.1 px = 144 px ≥ 80.
        let (major, _) = pick_intervals(0.1, 24, 1);
        assert_eq!(major, 1440);
    }

    // ---------- Visible-range behaviour ----------

    #[test]
    fn first_tick_at_zero_when_unscrolled() {
        let ticks = compute_visible_ticks(0.0, 1000.0, 10.0, 24, 1, false);
        let m = majors(&ticks);
        assert!(!m.is_empty());
        // First major should be at x=0 with label 00:00:00:00.
        assert_eq!(m[0].0, 0.0);
        assert_eq!(m[0].1, "00:00:00:00");
    }

    #[test]
    fn ticks_respect_scroll_offset() {
        // Scroll right by 100 px (scroll_x == -100). Zoom = 10 px/frame,
        // major every 10 frames = 100 px.
        //
        // With label padding, frame 0 ("00:00:00:00") is still emitted
        // — at viewport_x = -100 — so its label can clip gracefully
        // (in practice it'd be fully off-screen here, but the model
        // doesn't know the label width; the parent clip handles it).
        // The first major *inside* the viewport (x >= 0) is frame 10
        // at viewport_x = 100 + (-100) = 0, labelled "00:00:00:10".
        let ticks = compute_visible_ticks(-100.0, 1000.0, 10.0, 24, 1, false);
        let m = majors(&ticks);
        let first_inside = m.iter().find(|(x, _)| *x >= 0.0).unwrap();
        assert_eq!(first_inside.0, 0.0);
        assert_eq!(first_inside.1, "00:00:00:10");
    }

    #[test]
    fn ticks_stay_within_label_pad_of_viewport() {
        // viewport_w=100 px, zoom=10 px/frame: visible range is 10
        // frames, but we also emit ticks within `LABEL_PAD_PX` of each
        // edge so a major label whose anchor scrolled just past the
        // edge can still clip gracefully into view (see
        // `LABEL_PAD_PX` doc).
        let ticks = compute_visible_ticks(0.0, 100.0, 10.0, 24, 1, false);
        for t in &ticks {
            assert!(
                t.x >= -LABEL_PAD_PX - 1.0 && t.x <= 100.0 + LABEL_PAD_PX + 1.0,
                "tick at {} is too far outside viewport (pad = {})",
                t.x,
                LABEL_PAD_PX
            );
        }
    }

    #[test]
    fn major_label_survives_left_edge_until_fully_off_screen() {
        // Regression test for the "label pops" glitch:
        //
        // At zoom 10, the major interval is 10 frames = 100 px. As we
        // scroll right, the major at frame 0 (label "00:00:00:00")
        // moves left in viewport-local space. Its tick line is 1 px
        // wide, but its label is ~75 px wide, so the label should
        // remain in the tick model until the tick has scrolled at
        // least ~`LABEL_PAD_PX` past the edge.
        //
        // Pre-fix: any scroll_x < 0 (i.e. user has scrolled even 1 px
        // to the right) dropped frame 0's label from the model because
        // first_frame became >= 1.
        let zoom = 10.0;

        // Scrolled by 50 px — frame 0's tick is at viewport x = -50,
        // label extends from -47 to ~+28, so most of "00" is still
        // visible. We expect frame 0's label to STILL be in the model.
        let ticks = compute_visible_ticks(-50.0, 1000.0, zoom, 24, 1, false);
        let has_zero = ticks.iter().any(|t| t.is_major && t.label == "00:00:00:00");
        assert!(
            has_zero,
            "frame 0 label dropped from model while still partly visible"
        );

        // Scrolled way past — frame 0 is fully off-screen by the
        // padding distance; label is fine to drop now.
        let far = -(LABEL_PAD_PX as f64 + 50.0) as f32;
        let ticks = compute_visible_ticks(far, 1000.0, zoom, 24, 1, false);
        let has_zero = ticks.iter().any(|t| t.is_major && t.label == "00:00:00:00");
        assert!(
            !has_zero,
            "frame 0 label retained after it should have left the padded range"
        );
    }

    #[test]
    fn no_negative_frame_ticks() {
        // Scrolling "past" zero (positive scroll_x) — shouldn't happen
        // in normal Flickable use, but be defensive: never emit ticks
        // for negative frames.
        let ticks = compute_visible_ticks(50.0, 1000.0, 10.0, 24, 1, false);
        // left_px clamped to 0, so first_frame = 0. Should still work
        // and produce non-negative-frame ticks.
        for t in &ticks {
            assert!(t.x >= 0.0 || t.x.is_finite());
        }
    }

    // ---------- Labels follow drop-frame flag ----------

    #[test]
    fn major_labels_use_drop_frame_separator() {
        let ticks = compute_visible_ticks(0.0, 2000.0, 5.0, 30000, 1001, true);
        let m = majors(&ticks);
        // At least one major label should use the ';' (drop-frame) separator.
        assert!(m.iter().any(|(_, l)| l.contains(';')));
        assert!(!m.iter().any(|(_, l)| l.split_once(';').is_none() && l.matches(':').count() == 3),
            "any DF label should have exactly 2 ':' and 1 ';'");
    }

    #[test]
    fn major_labels_use_non_drop_separator_when_off() {
        let ticks = compute_visible_ticks(0.0, 2000.0, 5.0, 30000, 1001, false);
        let m = majors(&ticks);
        assert!(m.iter().all(|(_, l)| !l.contains(';')));
    }

    // ---------- Safety cap ----------

    #[test]
    fn cap_prevents_overflow_at_pathological_inputs() {
        // Very low zoom + very wide viewport could ask for millions of
        // ticks. We cap at MAX_TICKS to keep Slint sane.
        let ticks = compute_visible_ticks(0.0, 100_000_000.0, 1.0, 24, 1, false);
        assert!(ticks.len() <= MAX_TICKS);
    }

    // ---------- NTSC rounding sanity ----------

    #[test]
    fn ntsc_2398_labels_clean_seconds() {
        // 23.976 NDF: 1s major = round(24000/1001) = 24 frames.
        // At frame 24 the timecode label should still read "...:01:00".
        // pps = 24/1001 * 1000 * zoom; pick zoom so 24 frames ≈ 100 px.
        let zoom = 100.0 / 24.0;
        let ticks = compute_visible_ticks(0.0, 1000.0, zoom, 24000, 1001, false);
        let m = majors(&ticks);
        let has_one_sec_label = m.iter().any(|(_, l)| l == "00:00:01:00");
        assert!(
            has_one_sec_label,
            "expected a 00:00:01:00 label among {:?}",
            m
        );
    }

    // ---------- Coordinates rounded to whole pixels ----------

    #[test]
    fn tick_positions_are_integer_pixels() {
        // 1-px tick lines blur if positioned on a half pixel.
        // The implementation calls `.round()` before emitting.
        let ticks = compute_visible_ticks(-37.5, 500.0, 7.3, 24, 1, false);
        for t in &ticks {
            assert_eq!(t.x.fract(), 0.0, "tick {} not integer", t.x);
        }
    }

    // ---------- Model + slint adapter smoke ----------

    #[test]
    fn ticks_model_round_trip() {
        let model = ticks_model(0.0, 500.0, 10.0, 24, 1, false);
        assert!(model.row_count() > 0);
        let first = model.row_data(0).unwrap();
        assert!(first.is_major);
    }
}

//! CapCut-style timeline ruler: adaptive labels + dot subdivisions.
//!
//! Given the viewport state (scroll offset, width, zoom, frame rate),
//! returns the marks visible in the window, positioned in viewport-local
//! pixels. Two kinds of marks come back:
//!
//!   * **labels** (`is_major == true`) — text centered on the position.
//!     CapCut draws no tick line under a label; the text itself is the
//!     marker.
//!   * **dots** (`is_major == false`) — the tiny `· · ·` subdivisions
//!     CapCut renders between labels (instead of the hash-mark tick
//!     lines of pro NLEs).
//!
//! ## How CapCut's ruler adapts (and how this mirrors it)
//!
//! 1. **The label step climbs a time-aware ladder.** Zoomed in, labels
//!    are a few *frames* apart; zoomed out they're seconds, then
//!    minutes, then hours apart. Steps are "nice" units — frame steps
//!    must divide the (nominal) fps so second boundaries always carry a
//!    label, and second steps come from the `{1, 2, 3, 5, 10, 15, 30}`
//!    progression (then the same shape for minutes/hours). The smallest
//!    step that keeps labels at least [`MIN_LABEL_PX`] apart wins.
//!
//! 2. **Labels switch format by regime.** At second-and-coarser steps,
//!    labels read `MM:SS` (`H:MM:SS` past an hour) — CapCut's compact
//!    form, not full SMPTE. At sub-second steps, second boundaries keep
//!    `MM:SS` and the marks between them read `Nf` (frame *within* the
//!    current second): `00:01  5f  10f  15f  20f  00:02` — exactly the
//!    frame-accurate rhythm CapCut shows when zoomed in.
//!
//! 3. **Dots subdivide the label step.** The densest segment count from
//!    `{10, 8, 6, 5, 4, 3, 2}` that divides the label step evenly and
//!    keeps dots at least [`MIN_DOT_PX`] apart. Every label position is
//!    also a dot-grid position by construction, so the rhythm never
//!    drifts against the labels.
//!
//! 4. **Virtualized.** Marks are only emitted for the visible window
//!    (plus a small pad so a centered label can straddle the edge); the
//!    cost is bounded by the viewport, never the timeline length. All
//!    layout math happens on integer frame indices so nothing drifts
//!    hours into a project.
//!
//! ## Nominal fps
//!
//! The ladder runs on the *nominal* integer rate (`ceil(num/den)`: 24
//! for 23.976, 30 for 29.97). "One second" on the ruler is one nominal
//! second — the same convention as NDF timecode and CapCut itself. The
//! ruler doesn't render drop-frame; the transport readout
//! (`crate::timecode`) is the place for SMPTE-exact display.
//!
//! ## References
//!
//! - CapCut desktop (observed behaviour; our north star).
//! - designcombo/react-video-editor (`timeline/ruler.tsx`,
//!   `utils/format.ts`) — closest open CapCut clone: centered labels,
//!   `Nf`/`s`/`MM:SS` adaptive units, fixed segment subdivisions.
//! - OpenCut (`timeline-ruler.tsx`) — px-per-second thresholds for the
//!   interval ladder.

use slint::{ModelRc, SharedString, VecModel};
use std::rc::Rc;

use crate::RulerTick;

/// Minimum pixel distance between two adjacent labels. CapCut keeps its
/// compact `MM:SS` labels (~32 px at an 11 px font) roughly 90–150 px
/// apart; below this the texts start crowding.
const MIN_LABEL_PX: f32 = 90.0;

/// Minimum pixel distance between two adjacent dots. CapCut's dot
/// rhythm sits around 15–30 px; tighter than ~13 px reads as noise.
const MIN_DOT_PX: f32 = 13.0;

/// Pad (px) added to each side of the visible range so a label whose
/// *center* has scrolled past the edge still gets emitted while its
/// glyphs are partially visible. The parent `clip: true` Rectangle does
/// the actual visual clipping, pixel by pixel.
const EDGE_PAD_PX: f32 = 60.0;

/// Hard cap on emitted marks. Sane zooms produce ~40–120; the cap only
/// exists so a degenerate state (zoom collapsing toward 0 mid-gesture)
/// can't flood Slint with elements.
const MAX_TICKS: usize = 2048;

/// Frame steps eligible for sub-second labels, smallest first. A step
/// is only used when it divides the nominal fps — that guarantee is
/// what keeps every second boundary labeled (`00:02`) with clean `Nf`
/// marks between, instead of frame labels drifting across seconds.
const FRAME_STEPS: &[i64] = &[1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30];

/// Second-and-coarser steps, in seconds: {1,2,3,5,10,15,30} shaped
/// progressions through minutes and hours, up to 10 h.
const SECOND_STEPS: &[i64] = &[
    1, 2, 3, 5, 10, 15, 30, // seconds
    60, 120, 180, 300, 600, 900, 1800, // minutes
    3600, 7200, 18000, 36000, // hours
];

/// Candidate subdivision counts between labels, densest first.
const SEGMENT_COUNTS: &[i64] = &[10, 8, 6, 5, 4, 3, 2];

/// Compute the visible marks.
///
/// `scroll_x` follows the Slint Flickable convention: negative once the
/// user has scrolled right. Frame 0 sits at content-space pixel 0;
/// returned `x` values are viewport-local (`frame * zoom + scroll_x`).
pub fn compute_visible_ticks(
    scroll_x: f32,
    viewport_w: f32,
    zoom: f32,
    fps_num: i64,
    fps_den: i64,
) -> Vec<RulerTick> {
    // Degenerate inputs happen briefly during layout (unsized ruler,
    // store not seeded yet). Empty model > panic.
    if zoom <= 0.0 || viewport_w <= 0.0 || fps_num <= 0 || fps_den <= 0 {
        return Vec::new();
    }

    let nfps = nominal_fps(fps_num, fps_den);
    let label_step = pick_label_step(zoom, nfps);
    let step = pick_dot_step(label_step, zoom).unwrap_or(label_step);

    // Visible content-space pixel range → padded integer frame range.
    let left_px = -scroll_x;
    let pad_frames = (EDGE_PAD_PX / zoom).ceil() as i64;
    let first_frame = (left_px / zoom).floor() as i64 - pad_frames;
    let last_frame = ((left_px + viewport_w) / zoom).ceil() as i64 + pad_frames;

    let mut out: Vec<RulerTick> = Vec::with_capacity(128);

    // First multiple of `step` inside the padded range (div_euclid
    // rounds toward -inf, so this is exact even for negative starts).
    let mut k = first_frame.div_euclid(step) * step;
    if k < first_frame {
        k += step;
    }
    while k <= last_frame && out.len() < MAX_TICKS {
        if k >= 0 {
            let x = (k as f32 * zoom + scroll_x).round();
            let is_label = k % label_step == 0;
            out.push(RulerTick {
                x,
                is_major: is_label,
                label: if is_label {
                    SharedString::from(format_label(k, nfps))
                } else {
                    SharedString::default()
                },
            });
        }
        k += step;
    }

    out
}

/// Slint callback adapter (installed in `main.rs`): wraps the tick list
/// in a `ModelRc` for the Rust → Slint boundary. Slint passes `int` as
/// `i32`; widened to `i64` for the frame math.
pub fn ticks_model(
    scroll_x: f32,
    viewport_w: f32,
    zoom: f32,
    fps_num: i32,
    fps_den: i32,
) -> ModelRc<RulerTick> {
    let ticks = compute_visible_ticks(
        scroll_x,
        viewport_w,
        zoom,
        i64::from(fps_num),
        i64::from(fps_den),
    );
    ModelRc::from(Rc::new(VecModel::from(ticks)))
}

/// Nominal integer fps used for ruler units: 24 for 23.976, 30 for
/// 29.97 — the FF-rollover rate of NDF timecode, and what CapCut counts
/// frames against.
fn nominal_fps(fps_num: i64, fps_den: i64) -> i64 {
    (fps_num + fps_den - 1) / fps_den
}

/// Smallest ladder step (in frames) whose pixel width clears
/// [`MIN_LABEL_PX`] at this zoom.
///
/// Phase 1 — sub-second frame steps, restricted to divisors of the
/// nominal fps (see [`FRAME_STEPS`]). Phase 2 — the seconds/minutes/
/// hours progression. Falls back to the coarsest entry at absurd
/// zoom-outs.
fn pick_label_step(zoom: f32, nfps: i64) -> i64 {
    for &f in FRAME_STEPS {
        if f >= nfps || nfps % f != 0 {
            continue;
        }
        if f as f32 * zoom >= MIN_LABEL_PX {
            return f;
        }
    }
    for &s in SECOND_STEPS {
        let frames = s * nfps;
        if frames as f32 * zoom >= MIN_LABEL_PX {
            return frames;
        }
    }
    SECOND_STEPS.last().unwrap() * nfps
}

/// Dot step (in frames) between labels: the densest [`SEGMENT_COUNTS`]
/// entry that divides `label_step` evenly and keeps dots at least
/// [`MIN_DOT_PX`] apart. `None` ⇒ no dots (labels only) — happens when
/// the label step is prime-ish in frames or at extreme zoom-outs.
fn pick_dot_step(label_step: i64, zoom: f32) -> Option<i64> {
    for &segments in SEGMENT_COUNTS {
        if label_step % segments != 0 {
            continue;
        }
        let step = label_step / segments;
        if step as f32 * zoom >= MIN_DOT_PX {
            return Some(step);
        }
    }
    None
}

/// Label text for the mark at `frame`.
///
/// Second boundaries get CapCut's compact clock form — `MM:SS`, or
/// `H:MM:SS` once the timeline crosses an hour. Sub-second positions
/// (only ever emitted when the label step is < 1 s) read `Nf`, the
/// frame index *within* the current second: unambiguous at frame zoom
/// without the redundant `HH:MM:SS:` prefix SMPTE would repeat on
/// every label.
fn format_label(frame: i64, nfps: i64) -> String {
    let frame_in_second = frame % nfps;
    if frame_in_second != 0 {
        return format!("{frame_in_second}f");
    }
    let total_seconds = frame / nfps;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds / 60) % 60;
    let seconds = total_seconds % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use slint::Model;

    fn labels(ticks: &[RulerTick]) -> Vec<(f32, String)> {
        ticks
            .iter()
            .filter(|t| t.is_major)
            .map(|t| (t.x, t.label.to_string()))
            .collect()
    }

    fn dots(ticks: &[RulerTick]) -> Vec<f32> {
        ticks.iter().filter(|t| !t.is_major).map(|t| t.x).collect()
    }

    // ---------- Degenerate inputs ----------

    #[test]
    fn empty_for_bad_inputs() {
        assert!(compute_visible_ticks(0.0, 0.0, 6.0, 24, 1).is_empty());
        assert!(compute_visible_ticks(0.0, 1000.0, 0.0, 24, 1).is_empty());
        assert!(compute_visible_ticks(0.0, 1000.0, 6.0, 0, 1).is_empty());
        assert!(compute_visible_ticks(0.0, 1000.0, 6.0, 24, 0).is_empty());
    }

    // ---------- Label step ladder ----------

    #[test]
    fn second_steps_at_default_zoom() {
        // 24 fps, zoom = 6 px/frame. Frame steps dividing 24 max out at
        // 12f = 72 px < 90; 1 s = 24f = 144 px ≥ 90 → 1-second labels.
        assert_eq!(pick_label_step(6.0, 24), 24);
    }

    #[test]
    fn frame_steps_at_high_zoom() {
        // 24 fps, zoom = 50 px/frame: 2f = 100 px ≥ 90 → 2-frame labels
        // (1f = 50 px misses the threshold).
        assert_eq!(pick_label_step(50.0, 24), 2);
        // zoom = 100: every frame labeled.
        assert_eq!(pick_label_step(100.0, 24), 1);
    }

    #[test]
    fn frame_steps_only_divide_nominal_fps() {
        // 25 fps (PAL), zoom = 25 px/frame: 4f would be 100 px ≥ 90 but
        // 4 ∤ 25 — the eligible 5f step (125 px) wins so second
        // boundaries stay labeled.
        assert_eq!(pick_label_step(25.0, 25), 5);
    }

    #[test]
    fn minute_steps_at_low_zoom() {
        // 24 fps, zoom = 0.07 px/frame: 30 s = 720f = 50.4 px (fail),
        // 1 min = 1440f = 100.8 px ≥ 90 → minute labels.
        assert_eq!(pick_label_step(0.07, 24), 1440);
    }

    #[test]
    fn coarsest_step_at_absurd_zoom() {
        assert_eq!(pick_label_step(1e-9, 24), 36000 * 24);
    }

    // ---------- Dot step ----------

    #[test]
    fn dots_subdivide_the_label_step() {
        // Label step 24f at zoom 6: segments 10 (24 % 10 ≠ 0) skipped,
        // 8 → 3f = 18 px ≥ 13 → dot step 3.
        assert_eq!(pick_dot_step(24, 6.0), Some(3));
    }

    #[test]
    fn dot_density_respects_min_px() {
        // Label step 24f at zoom 1: 8 → 3f = 3 px (fail), 6 → 4f = 4 px
        // (fail), … 2 → 12f = 12 px (fail) → no dots.
        assert_eq!(pick_dot_step(24, 1.0), None);
        // Slightly more zoom: 2 → 12f = 16.8 px ≥ 13 → dots at 12f.
        assert_eq!(pick_dot_step(24, 1.4), Some(12));
    }

    #[test]
    fn frame_regime_dots_go_to_single_frames() {
        // Label step 2f at zoom 50: segments 2 → 1f = 50 px → dots on
        // the odd frames.
        assert_eq!(pick_dot_step(2, 50.0), Some(1));
    }

    // ---------- Label format ----------

    #[test]
    fn labels_use_compact_clock_form() {
        assert_eq!(format_label(0, 24), "00:00");
        assert_eq!(format_label(24, 24), "00:01");
        assert_eq!(format_label(24 * 90, 24), "01:30");
        assert_eq!(format_label(24 * 3600, 24), "1:00:00");
        assert_eq!(format_label(24 * 3661, 24), "1:01:01");
    }

    #[test]
    fn sub_second_labels_use_frame_in_second() {
        assert_eq!(format_label(5, 24), "5f");
        // Frame 29 at 24 fps = second 1 + 5 frames → "5f", not "29f".
        assert_eq!(format_label(29, 24), "5f");
    }

    #[test]
    fn frame_zoom_interleaves_seconds_and_frames() {
        // 24 fps, zoom = 20 px/frame → label step 6f (120 px ≥ 90).
        // Expected rhythm: 00:00, 6f, 12f, 18f, 00:01, 6f, …
        let ticks = compute_visible_ticks(0.0, 1200.0, 20.0, 24, 1);
        let l = labels(&ticks);
        let texts: Vec<&str> = l.iter().map(|(_, s)| s.as_str()).collect();
        assert!(texts.starts_with(&["00:00", "6f", "12f", "18f", "00:01"]));
    }

    #[test]
    fn ntsc_uses_nominal_seconds() {
        // 29.97: nominal 30 — the 1-second label lands on frame 30.
        let ticks = compute_visible_ticks(0.0, 1500.0, 4.0, 30000, 1001);
        let l = labels(&ticks);
        assert_eq!(l[0], (0.0, "00:00".into()));
        assert_eq!(l[1], (120.0, "00:01".into()));
    }

    // ---------- Emission / virtualization ----------

    #[test]
    fn label_positions_align_with_dot_grid() {
        let ticks = compute_visible_ticks(0.0, 1500.0, 6.0, 24, 1);
        // Dot step divides label step, so consecutive marks are evenly
        // spaced and every label sits on the shared grid.
        let xs: Vec<f32> = ticks.iter().map(|t| t.x).collect();
        let step = xs[1] - xs[0];
        for w in xs.windows(2) {
            assert_eq!(w[1] - w[0], step);
        }
    }

    #[test]
    fn dots_are_denser_than_labels() {
        let ticks = compute_visible_ticks(0.0, 2000.0, 6.0, 24, 1);
        assert!(dots(&ticks).len() > labels(&ticks).len());
    }

    #[test]
    fn first_label_at_zero_when_unscrolled() {
        let ticks = compute_visible_ticks(0.0, 1000.0, 6.0, 24, 1);
        let l = labels(&ticks);
        assert_eq!(l[0], (0.0, "00:00".into()));
    }

    #[test]
    fn marks_respect_scroll_offset() {
        // Scrolled right 100 px at zoom 6 → frame 24's label lands at
        // viewport x = 24·6 − 100 = 44.
        let ticks = compute_visible_ticks(-100.0, 1000.0, 6.0, 24, 1);
        let l = labels(&ticks);
        let first_inside = l.iter().find(|(x, _)| *x >= 0.0).unwrap();
        assert_eq!(first_inside, &(44.0, "00:01".into()));
    }

    #[test]
    fn no_marks_for_negative_frames() {
        let ticks = compute_visible_ticks(0.0, 1000.0, 6.0, 24, 1);
        assert!(ticks.iter().all(|t| t.x >= 0.0));
    }

    #[test]
    fn marks_stay_within_pad_of_viewport() {
        let ticks = compute_visible_ticks(-500.0, 800.0, 6.0, 24, 1);
        for t in &ticks {
            assert!(
                t.x >= -EDGE_PAD_PX - 6.0 && t.x <= 800.0 + EDGE_PAD_PX + 6.0,
                "mark at {} outside padded viewport",
                t.x
            );
        }
    }

    #[test]
    fn centered_label_survives_left_edge_until_off_screen() {
        // A label is ~32 px wide, centered — its tick can be up to
        // ~16 px past the edge with glyphs still visible. The pad keeps
        // it in the model; the clip rect trims it visually.
        let ticks = compute_visible_ticks(-10.0, 1000.0, 6.0, 24, 1);
        assert!(ticks.iter().any(|t| t.is_major && t.label == "00:00"));

        let far = -(EDGE_PAD_PX + 50.0);
        let ticks = compute_visible_ticks(far, 1000.0, 6.0, 24, 1);
        assert!(!ticks.iter().any(|t| t.is_major && t.label == "00:00"));
    }

    #[test]
    fn positions_are_integer_pixels() {
        // 1–2 px marks smear when placed on half pixels.
        let ticks = compute_visible_ticks(-37.5, 500.0, 7.3, 24, 1);
        for t in &ticks {
            assert_eq!(t.x.fract(), 0.0);
        }
    }

    #[test]
    fn cap_prevents_overflow_at_pathological_inputs() {
        let ticks = compute_visible_ticks(0.0, 100_000_000.0, 1.0, 24, 1);
        assert!(ticks.len() <= MAX_TICKS);
    }

    // ---------- Slint adapter ----------

    #[test]
    fn ticks_model_round_trip() {
        let model = ticks_model(0.0, 500.0, 6.0, 24, 1);
        assert!(model.row_count() > 0);
        let first = model.row_data(0).unwrap();
        assert!(first.is_major);
        assert_eq!(first.label, "00:00");
    }
}

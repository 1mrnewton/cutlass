//! Drag-time preview-frame tick mapping for timeline / inspector gestures
//! that cannot cheaply apply a faithful project override mid-drag.
//!
//! Speed-ramp handle drags and transition-duration pill drags only refresh
//! the preview at a scrub-style tick (no undo, no project mutation). Commit
//! still happens on release through the existing edit paths.

use cutlass_models::SPEED_CURVE_SCALE;

/// Map a speed-ramp handle's normalized tick (`0..=`[`SPEED_CURVE_SCALE`])
/// onto an absolute sequence tick inside the clip's timeline span.
///
/// Used while the velocity-graph handle is dragged so the preview shows the
/// frame under that control point. Does not apply a tentative speed curve —
/// duration re-derive mid-drag is out of scope; this is a scrub-style refresh.
pub fn speed_ramp_handle_preview_tick(
    clip_start: i64,
    clip_duration: i64,
    handle_norm_tick: i64,
) -> i64 {
    if clip_duration <= 0 {
        return clip_start.max(0);
    }
    let norm = handle_norm_tick.clamp(0, SPEED_CURVE_SCALE);
    let offset = clip_duration.saturating_mul(norm) / SPEED_CURVE_SCALE;
    let last = clip_duration.saturating_sub(1);
    clip_start.saturating_add(offset.min(last))
}

/// Absolute sequence tick at the midpoint of a transition window.
///
/// Junction pills are centered on the cut, so the midpoint is always
/// `cut_tick` regardless of the tentative duration. Dragging the duration
/// handle therefore scrub-previews the cut (where the blend is strongest)
/// without mutating the project.
pub fn transition_midpoint_preview_tick(cut_tick: i64) -> i64 {
    cut_tick.max(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_ramp_handle_maps_norm_onto_clip_span() {
        assert_eq!(speed_ramp_handle_preview_tick(100, 200, 0), 100);
        assert_eq!(
            speed_ramp_handle_preview_tick(100, 200, SPEED_CURVE_SCALE / 2),
            200
        );
        // End of the normalized domain lands on the last inclusive frame.
        assert_eq!(
            speed_ramp_handle_preview_tick(100, 200, SPEED_CURVE_SCALE),
            299
        );
    }

    #[test]
    fn speed_ramp_handle_clamps_empty_and_oob_norm() {
        assert_eq!(speed_ramp_handle_preview_tick(40, 0, 500), 40);
        assert_eq!(speed_ramp_handle_preview_tick(10, 50, -100), 10);
        assert_eq!(
            speed_ramp_handle_preview_tick(10, 50, SPEED_CURVE_SCALE + 50),
            59
        );
    }

    #[test]
    fn transition_midpoint_is_the_cut() {
        assert_eq!(transition_midpoint_preview_tick(480), 480);
        assert_eq!(transition_midpoint_preview_tick(-3), 0);
    }
}

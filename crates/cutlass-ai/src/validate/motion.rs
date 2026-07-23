//! AI-path motion clamps with model-teaching rejection messages.
//!
//! These bounds are intentionally stricter than `cutlass-models` so the
//! desktop UI can stay permissive for manual gestures while the agent gets
//! clear "you meant X" feedback instead of flying rects.

use crate::wire::{WireClipParam, WireScale};

use super::Rejection;

/// Reject a position component outside ±1.5 canvas fractions.
pub(super) fn check_position_component(v: f64) -> Result<(), Rejection> {
    if !v.is_finite() || v.abs() > 1.5 {
        return Err(Rejection::new(format!(
            "position {v} is outside ±1.5: position is the anchor offset from the \
             canvas center in canvas fractions ([0,0] = centered, [0.5,0] = right \
             edge); values beyond ±1.5 are far off-screen"
        )));
    }
    Ok(())
}

pub(super) fn check_position(xy: [f64; 2]) -> Result<(), Rejection> {
    check_position_component(xy[0])?;
    check_position_component(xy[1])
}

/// Reject a scale component that is non-positive or exceeds 10 (1000%).
/// Values that look like CapCut percents get a division hint.
pub(super) fn check_scale_component(v: f64) -> Result<(), Rejection> {
    if !v.is_finite() || v <= 0.0 {
        return Err(Rejection::new(format!(
            "scale {v} must be positive: scale 1.0 = 100% (aspect-fit); never send 0"
        )));
    }
    if v > 10.0 {
        let as_fraction = v / 100.0;
        return Err(Rejection::new(format!(
            "scale {v} exceeds 10: scale 1.0 = 100% (aspect-fit); for {v}% send {as_fraction}"
        )));
    }
    Ok(())
}

pub(super) fn check_wire_scale(scale: WireScale) -> Result<(), Rejection> {
    match scale {
        WireScale::Uniform(s) => check_scale_component(s),
        WireScale::Axes([x, y]) => {
            check_scale_component(x)?;
            check_scale_component(y)
        }
    }
}

/// Reject an anchor component outside `[-1, 2]` (normalized content bounds).
pub(super) fn check_anchor_component(v: f64) -> Result<(), Rejection> {
    if !v.is_finite() || !(-1.0..=2.0).contains(&v) {
        return Err(Rejection::new(format!(
            "anchor {v} is outside [-1, 2]: anchor is normalized within the content \
             bounds ([0.5,0.5] = content center)"
        )));
    }
    Ok(())
}

pub(super) fn check_anchor(xy: [f64; 2]) -> Result<(), Rejection> {
    check_anchor_component(xy[0])?;
    check_anchor_component(xy[1])
}

/// Clamp spatial tangent components to ±2 canvas fractions (AI path only;
/// models stay at ±4 for desktop gestures).
pub(super) fn check_tangent_component(v: f64, axis: &str) -> Result<f32, Rejection> {
    if !v.is_finite() {
        return Err(Rejection::new(format!(
            "spatial tangent {axis} must be finite (got {v})"
        )));
    }
    let f = v as f32;
    if f.abs() > 2.0 {
        return Err(Rejection::new(format!(
            "spatial tangent {axis} = {f} is outside ±2 canvas fractions: tangents are \
             motion-path handle offsets from the keyframe value in the same units as \
             position (anchor offset from canvas center)"
        )));
    }
    Ok(f)
}

/// Run motion clamps on wire args for transform keyframe/constant params.
/// Missing args are left to [`super::param_value`] so messages stay unique.
pub(super) fn check_motion_param_args(
    param: &WireClipParam,
    value: Option<f64>,
    position: Option<[f64; 2]>,
) -> Result<(), Rejection> {
    match param {
        WireClipParam::Position => {
            if let Some(xy) = position {
                check_position(xy)?;
            }
        }
        WireClipParam::AnchorPoint => {
            if let Some(xy) = position {
                check_anchor(xy)?;
            }
        }
        WireClipParam::Scale => {
            if let Some([x, y]) = position {
                check_scale_component(x)?;
                check_scale_component(y)?;
            } else if let Some(v) = value {
                check_scale_component(v)?;
            }
        }
        _ => {}
    }
    Ok(())
}

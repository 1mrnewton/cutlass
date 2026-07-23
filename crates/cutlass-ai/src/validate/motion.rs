//! AI-path motion clamps with model-teaching rejection messages.
//!
//! These bounds are intentionally stricter than `cutlass-models` so the
//! desktop UI can stay permissive for manual gestures while the agent gets
//! clear "you meant X" feedback instead of flying rects.

use cutlass_models::Clip;

use crate::wire::{WireClipParam, WireScale};

use super::Rejection;

/// `speed` is still accepted by wire serde so a mistaken tool call reaches
/// validate, but it is not keyframable — teach the dedicated speed tools.
pub(super) fn reject_speed_keyframe_param(param: &WireClipParam) -> Result<(), Rejection> {
    if matches!(param, WireClipParam::Speed) {
        return Err(Rejection::new(
            "speed is not keyframable — use set_clip_speed for a constant rate \
             or set_speed_curve for a ramp",
        ));
    }
    Ok(())
}

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

/// Smallest scale the AI path accepts. Tiny positives (e.g. `1e-50`) pass an
/// `(0, 10]` f64 clamp but collapse to `0.0f32` in the engine — reject them
/// here with a usable floor. Position / anchor / tangent allow 0, so f64→f32
/// denormal collapse is harmless for those axes.
pub(super) const MIN_USABLE_SCALE: f64 = 0.001;

/// Reject a scale component that is non-positive, below [`MIN_USABLE_SCALE`],
/// or exceeds 10 (1000%). Values that look like CapCut percents get a hint.
pub(super) fn check_scale_component(v: f64) -> Result<(), Rejection> {
    if !v.is_finite() || v <= 0.0 {
        return Err(Rejection::new(format!(
            "scale {v} must be positive: scale 1.0 = 100% (aspect-fit); never send 0"
        )));
    }
    if v < MIN_USABLE_SCALE {
        return Err(Rejection::new(format!(
            "scale {v} is below {MIN_USABLE_SCALE} — the smallest usable scale; 1.0 = 100%"
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

/// `set_clip_transform` lowers with `at: None`, which flattens *every*
/// transform param to a constant. Reject whenever any param is animated so
/// the agent cannot silently destroy keyframe curves.
pub(super) fn check_set_clip_transform_preserves_keyframes(
    clip: &Clip,
    wire_clip: u64,
) -> Result<(), Rejection> {
    let t = &clip.transform;
    let mut animated = Vec::new();
    if t.position.is_animated() {
        animated.push("position");
    }
    if t.anchor_point.is_animated() {
        animated.push("anchor_point");
    }
    if t.scale.is_animated() {
        animated.push("scale");
    }
    if t.rotation.is_animated() {
        animated.push("rotation");
    }
    if t.opacity.is_animated() {
        animated.push("opacity");
    }
    if animated.is_empty() {
        return Ok(());
    }
    let detail = if animated.len() == 1 {
        format!("has keyframes on {}", animated[0])
    } else {
        format!("has keyframed transform params ({})", animated.join(", "))
    };
    Err(Rejection::new(format!(
        "clip {wire_clip} {detail}; set_clip_transform would erase that animation — \
         use set_param_keyframe to edit the curve or set_param_constant to remove it first"
    )))
}

/// `set_clip_crop` lowers with `at: None`, which flattens an animated crop to
/// a constant. Flip flags are separate non-keyframed bools on the clip — this
/// guard is about the crop param only.
pub(super) fn check_set_clip_crop_preserves_keyframes(
    clip: &Clip,
    wire_clip: u64,
) -> Result<(), Rejection> {
    if !clip.crop.is_animated() {
        return Ok(());
    }
    Err(Rejection::new(format!(
        "clip {wire_clip} has keyframes on crop; set_clip_crop would erase that animation — \
         use set_param_keyframe to edit the curve or set_param_constant to remove it first"
    )))
}

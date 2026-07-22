#![allow(unused_imports)]

use std::path::Path;

use crate::clip::{
    Clip, ClipParam, ClipSource, ClipTransform, CropRect, Generator, ParamValue, Replaceable,
    SlotMedia, look_animation_combo_period_ticks, look_animation_window_ticks, split_speed_curve,
};
use crate::effects::EffectInstance;
use crate::error::ModelError;
use crate::ids::{ClipId, MediaId, ProjectId, TrackId};
use crate::look::{
    AnimationRef, AnimationSlot, AudioRole, ChromaKey, ColorAdjustments, Filter, Lut, Mask,
    StabilizeLevel, animation_spec,
};
use crate::media::MediaSource;
use crate::metadata::ProjectMetadata;
use crate::param::{Easing, Param};
use crate::schema::ProjectSchema;
use crate::time::{
    Rational, RationalTime, TimeRange, check_same_rate, resample, time_add, time_sub,
};
use crate::timeline::Timeline;
use crate::track::{Track, TrackKind};
use crate::transition::Transition;

pub(super) fn paths_refer_to_same_file(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(ca), Ok(cb)) => ca == cb,
        _ => false,
    }
}

/// The effect at `index` on a clip's chain, or an out-of-range error.
pub(super) fn effect_mut(clip: &mut Clip, index: u32) -> Result<&mut EffectInstance, ModelError> {
    clip.effects
        .get_mut(index as usize)
        .ok_or_else(|| ModelError::InvalidParam(format!("effect index {index} out of range")))
}

/// A generated clip's generator, or an error for media-backed clips (shape
/// params route here; the generator itself rejects non-shape kinds).
pub(super) fn generator_mut(clip: &mut Clip) -> Result<&mut Generator, ModelError> {
    match &mut clip.content {
        ClipSource::Generated(generator) => Ok(generator),
        ClipSource::Media { .. } => Err(ModelError::InvalidParam(
            "shape parameters apply only to generated clips".into(),
        )),
    }
}

/// Reject splits that the current clip representation cannot express without
/// changing the rendered result. Ordinary keyframes can be rebased, while
/// edge-anchored fades/look animations are partitioned after these checks.
pub(super) fn validate_split_render_continuity(
    clip: &Clip,
    left_duration: i64,
    right_duration: i64,
    rate: Rational,
) -> Result<(), ModelError> {
    if clip.fade_in > left_duration {
        return Err(ModelError::InvalidParam(
            "cannot split inside a clip's fade-in".into(),
        ));
    }
    if clip.fade_out > right_duration {
        return Err(ModelError::InvalidParam(
            "cannot split inside a clip's fade-out".into(),
        ));
    }

    if clip.animation_combo.is_some() {
        let period = look_animation_combo_period_ticks(rate);
        if left_duration % period != 0 {
            return Err(ModelError::InvalidParam(format!(
                "cannot split a combo-animated clip away from its {period}-tick phase boundary"
            )));
        }
    } else {
        let original_window = look_animation_window_ticks(clip.timeline.duration.value, rate);
        if clip.animation_in.is_some()
            && look_animation_window_ticks(left_duration, rate) != original_window
        {
            return Err(ModelError::InvalidParam(
                "split would retime the clip's entrance animation".into(),
            ));
        }
        if clip.animation_out.is_some()
            && look_animation_window_ticks(right_duration, rate) != original_window
        {
            return Err(ModelError::InvalidParam(
                "split would retime the clip's exit animation".into(),
            ));
        }
    }

    match &clip.content {
        ClipSource::Generated(Generator::Lottie { .. }) => Err(ModelError::InvalidParam(
            "cannot split a Lottie clip without an animation-time offset".into(),
        )),
        ClipSource::Generated(Generator::Sticker { asset })
            if crate::sticker::sticker_spec(asset).is_some_and(|spec| spec.animated) =>
        {
            Err(ModelError::InvalidParam(
                "cannot split an animated sticker without an animation-time offset".into(),
            ))
        }
        ClipSource::Media { .. } | ClipSource::Generated(_) => Ok(()),
    }
}

/// Timeline ticks a retimed clip occupies: `source ÷ (base_speed × average
/// ramp)`. A flat ramp keeps the exact integer division M1 used (no f64
/// drift on the common constant-speed path); an active ramp folds in its
/// average multiplier. Always at least one tick.
pub(super) fn retimed_duration(
    src_dur_tl: i64,
    speed: Rational,
    average: f64,
    has_curve: bool,
) -> i64 {
    if !has_curve {
        return (src_dur_tl * i64::from(speed.den) / i64::from(speed.num)).max(1);
    }
    let base = f64::from(speed.num) / f64::from(speed.den);
    let effective = base * average;
    if effective <= 0.0 {
        return src_dur_tl.max(1);
    }
    (src_dur_tl as f64 / effective).round().max(1.0) as i64
}

/// Unwrap a scalar [`ParamValue`].
pub(super) fn scalar_param(value: ParamValue) -> Result<f32, ModelError> {
    match value {
        ParamValue::Scalar(v) => Ok(v),
        ParamValue::Vec2(_) | ParamValue::Color(_) | ParamValue::Rect(_) => {
            Err(ModelError::InvalidParam("expected a scalar value".into()))
        }
    }
}

/// Unwrap a rect [`ParamValue`] as a validated [`CropRect`].
pub(super) fn crop_rect_param(value: ParamValue) -> Result<CropRect, ModelError> {
    let [x, y, w, h] = value.rect()?;
    let crop = CropRect { x, y, w, h };
    crop.validate()?;
    Ok(crop)
}

#![allow(unused_imports)]

use std::path::Path;

use crate::clip::{
    Clip, ClipParam, ClipSource, ClipTransform, CropRect, Generator, LookParam, ParamValue,
    Replaceable, SlotMedia, StyleParam, look_animation_combo_period_ticks,
    look_animation_window_ticks, split_speed_curve,
};
use crate::effects::EffectInstance;
use crate::error::ModelError;
use crate::ids::{ClipId, MediaId, ProjectId, TrackId};
use crate::look::mask::{
    is_mask_param, mask_scalar_mut, mask_vec2_mut, remove_mask_param_keyframe,
    set_mask_param_constant, set_mask_param_keyframe,
};
use crate::look::styles::{
    remove_style_param_keyframe, set_style_param_constant, set_style_param_keyframe,
    style_scalar_mut, style_vec2_mut,
};
use crate::look::{
    AnimationRef, AnimationSlot, AudioRole, ChromaKey, ColorAdjustments, Filter, Lut, Mask,
    StabilizeLevel, animation_spec,
};
use crate::media::MediaSource;
use crate::metadata::ProjectMetadata;
use crate::param::{Easing, Param, PiecewiseEasingPreset, SpatialTangents};
use crate::schema::ProjectSchema;
use crate::time::{
    Rational, RationalTime, TimeRange, check_same_rate, resample, time_add, time_sub,
};
use crate::timeline::Timeline;
use crate::track::{Track, TrackKind};
use crate::transition::Transition;

use super::Project;

fn look_param_mut(clip: &mut Clip, param: LookParam) -> Result<&mut Param<f32>, ModelError> {
    let missing =
        |name: &str| ModelError::InvalidParam(format!("{name} is not enabled on this clip"));
    match param {
        LookParam::FilterIntensity => clip
            .filter
            .as_mut()
            .map(|filter| &mut filter.intensity)
            .ok_or_else(|| missing("filter")),
        LookParam::LutIntensity => clip
            .lut
            .as_mut()
            .map(|lut| &mut lut.intensity)
            .ok_or_else(|| missing("LUT")),
        LookParam::AdjustBrightness => Ok(&mut clip.adjust.brightness),
        LookParam::AdjustContrast => Ok(&mut clip.adjust.contrast),
        LookParam::AdjustSaturation => Ok(&mut clip.adjust.saturation),
        LookParam::AdjustExposure => Ok(&mut clip.adjust.exposure),
        LookParam::AdjustTemperature => Ok(&mut clip.adjust.temperature),
        LookParam::AdjustTint => Ok(&mut clip.adjust.tint),
        LookParam::AdjustHue => Ok(&mut clip.adjust.hue),
        LookParam::AdjustHighlights => Ok(&mut clip.adjust.highlights),
        LookParam::AdjustShadows => Ok(&mut clip.adjust.shadows),
        LookParam::AdjustSharpness => Ok(&mut clip.adjust.sharpness),
        LookParam::AdjustVignette => Ok(&mut clip.adjust.vignette),
        LookParam::ChromaStrength => clip
            .chroma_key
            .as_mut()
            .map(|chroma| &mut chroma.strength)
            .ok_or_else(|| missing("chroma key")),
        LookParam::ChromaShadow => clip
            .chroma_key
            .as_mut()
            .map(|chroma| &mut chroma.shadow)
            .ok_or_else(|| missing("chroma key")),
        LookParam::MaskFeather
        | LookParam::MaskCenter
        | LookParam::MaskSize
        | LookParam::MaskRotation
        | LookParam::MaskRoundness => Err(ModelError::InvalidParam(format!(
            "look parameter {param:?} is routed through the mask param helpers"
        ))),
    }
}

fn validate_look_value(param: LookParam, value: f32) -> Result<(), ModelError> {
    let valid = match param {
        LookParam::FilterIntensity
        | LookParam::LutIntensity
        | LookParam::ChromaStrength
        | LookParam::ChromaShadow => (0.0..=1.0).contains(&value),
        LookParam::AdjustBrightness
        | LookParam::AdjustContrast
        | LookParam::AdjustSaturation
        | LookParam::AdjustExposure
        | LookParam::AdjustTemperature
        | LookParam::AdjustTint
        | LookParam::AdjustHue
        | LookParam::AdjustHighlights
        | LookParam::AdjustShadows => (-1.0..=1.0).contains(&value),
        LookParam::AdjustSharpness | LookParam::AdjustVignette => (0.0..=1.0).contains(&value),
        LookParam::MaskFeather
        | LookParam::MaskCenter
        | LookParam::MaskSize
        | LookParam::MaskRotation
        | LookParam::MaskRoundness => {
            return Err(ModelError::InvalidParam(format!(
                "look parameter {param:?} is routed through the mask param helpers"
            )));
        }
    };
    if value.is_finite() && valid {
        Ok(())
    } else {
        Err(ModelError::InvalidParam(format!(
            "look parameter {param:?} = {value} is out of range"
        )))
    }
}

impl Project {
    /// Set a clip's spatial transform (preview move/scale/rotate, inspector
    /// numerics). Errors if the clip is unknown, sits on an audio track
    /// (nothing to place), or the transform is invalid (non-finite, scale
    /// ≤ 0, opacity outside 0..=1).
    ///
    /// `at` composes the edit with animation CapCut-style: `Some(timeline
    /// tick)` writes a keyframe at that position on properties that already
    /// have keyframes (constants stay constant); `None` flattens every
    /// property to a constant, dropping keyframes. Never-animated clips
    /// behave identically either way.
    pub fn set_transform(
        &mut self,
        clip_id: ClipId,
        transform: ClipTransform,
        at: Option<RationalTime>,
    ) -> Result<(), ModelError> {
        transform.validate()?;
        let track_id = self
            .timeline
            .track_of(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let kind = self
            .timeline
            .track(track_id)
            .ok_or(ModelError::UnknownTrack(track_id))?
            .kind;
        if !kind.is_visual() {
            return Err(ModelError::IncompatibleTrackKind {
                track: track_id,
                kind,
            });
        }
        if let Some(at) = at {
            check_same_rate(at.rate, self.timeline.frame_rate)?;
        }
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        match at {
            Some(at) => {
                let tick = clip.animation_tick(at.value);
                clip.transform.compose_at(transform, tick);
            }
            None => clip.transform.set_constant(transform),
        }
        Ok(())
    }

    /// Shared precondition for parameter edits: the clip exists on a visual
    /// track. Returns the track kind error otherwise (audio has no canvas
    /// placement to animate).
    fn check_param_target(&self, clip_id: ClipId) -> Result<(), ModelError> {
        let track_id = self
            .timeline
            .track_of(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let kind = self
            .timeline
            .track(track_id)
            .ok_or(ModelError::UnknownTrack(track_id))?
            .kind;
        if !kind.is_visual() {
            return Err(ModelError::IncompatibleTrackKind {
                track: track_id,
                kind,
            });
        }
        Ok(())
    }

    /// Precondition for audio-param edits (volume / pan): the clip exists and
    /// is media-backed (generators have nothing to hear). Mirrors
    /// [`Self::set_clip_audio`]'s target rule — audio params ride any media
    /// clip (including video with sound), since linkage lands the audible half
    /// on an audio lane.
    fn check_audio_param_target(&self, clip_id: ClipId) -> Result<(), ModelError> {
        let clip = self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        if clip.is_generated() {
            return Err(ModelError::InvalidParam(
                "audio parameters require a media-backed clip".into(),
            ));
        }
        Ok(())
    }

    /// Convert an absolute timeline position to a clip-relative animation
    /// tick, rejecting positions outside the clip (a keyframe must sit on
    /// the clip it animates).
    fn keyframe_tick(&self, clip_id: ClipId, at: RationalTime) -> Result<i64, ModelError> {
        check_same_rate(at.rate, self.timeline.frame_rate)?;
        let clip = self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        if !clip.timeline.contains(at)? {
            return Err(ModelError::InvalidParam(format!(
                "keyframe position {} is outside clip {clip_id}",
                at.value
            )));
        }
        Ok(at.value - clip.timeline.start.value)
    }

    /// Insert or replace a keyframe on one animatable clip property. `at` is
    /// an absolute timeline position and must fall inside the clip.
    ///
    /// `tangents` shapes a cubic-bezier motion path and is accepted only on
    /// [`ClipParam::Position`] (canvas-fraction handles). Other params reject
    /// a non-`None` value with "spatial tangents are only supported on
    /// position". `None` clears/keeps straight-line motion on position.
    pub fn set_param_keyframe(
        &mut self,
        clip_id: ClipId,
        param: ClipParam,
        at: RationalTime,
        value: ParamValue,
        easing: Easing,
        tangents: Option<SpatialTangents>,
    ) -> Result<(), ModelError> {
        if tangents.is_some() && param != ClipParam::Position {
            return Err(ModelError::InvalidParam(
                "spatial tangents are only supported on position".into(),
            ));
        }
        if let Some(t) = tangents {
            t.validate()?;
        }
        // Volume (M8) is an audio property, not a transform: validate the
        // gain range and an audio-capable target, then write to the envelope.
        if param == ClipParam::Volume {
            easing.validate()?;
            let v = super::helpers::scalar_param(value)?;
            crate::clip::validate_volume(v)?;
            self.check_audio_param_target(clip_id)?;
            let tick = self.keyframe_tick(clip_id, at)?;
            let clip = self
                .timeline
                .clip_mut(clip_id)
                .ok_or(ModelError::UnknownClip(clip_id))?;
            clip.volume.set_keyframe(tick, v, easing);
            return Ok(());
        }
        // Pan mirrors volume: audio property on media-backed clips.
        if param == ClipParam::Pan {
            easing.validate()?;
            let v = super::helpers::scalar_param(value)?;
            crate::clip::validate_pan(v)?;
            self.check_audio_param_target(clip_id)?;
            let tick = self.keyframe_tick(clip_id, at)?;
            let clip = self
                .timeline
                .clip_mut(clip_id)
                .ok_or(ModelError::UnknownClip(clip_id))?;
            clip.pan.set_keyframe(tick, v, easing);
            return Ok(());
        }
        // Crop rides the clip's framing param (validated like set_clip_crop).
        if param == ClipParam::Crop {
            easing.validate()?;
            let crop = super::helpers::crop_rect_param(value)?;
            self.check_param_target(clip_id)?;
            let tick = self.keyframe_tick(clip_id, at)?;
            let clip = self
                .timeline
                .clip_mut(clip_id)
                .ok_or(ModelError::UnknownClip(clip_id))?;
            clip.crop.set_keyframe(tick, crop, easing);
            return Ok(());
        }
        self.check_param_target(clip_id)?;
        let tick = self.keyframe_tick(clip_id, at)?;
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let result = match param {
            ClipParam::Effect { effect, param } => super::helpers::effect_mut(clip, effect)?
                .set_param_value_keyframe(param as usize, tick, value, easing),
            ClipParam::Shape { param } => super::helpers::generator_mut(clip)?
                .set_shape_param_keyframe(param, tick, value, easing),
            ClipParam::Text { param } => super::helpers::generator_mut(clip)?
                .set_text_param_keyframe(param, tick, value, easing),
            ClipParam::Look { param } if is_mask_param(param) => {
                set_mask_param_keyframe(&mut clip.mask, param, tick, value, easing)
            }
            ClipParam::Look { param } => {
                let value = super::helpers::scalar_param(value)?;
                validate_look_value(param, value)?;
                look_param_mut(clip, param)?.set_keyframe(tick, value, easing);
                Ok(())
            }
            ClipParam::Style { param } => {
                set_style_param_keyframe(&mut clip.styles, param, tick, value, easing)
            }
            _ => clip
                .transform
                .set_param_keyframe(param, tick, value, easing),
        };
        result?;
        if param == ClipParam::Position {
            // Always write the tangents slot so a None command clears a
            // previous curve (EditCommand redo / AI wire).
            clip.transform
                .position
                .set_keyframe_tangents(tick, tangents)?;
        }
        Ok(())
    }

    /// Set or clear spatial bezier tangents on a position keyframe at `at`.
    /// Rejected for every param other than [`ClipParam::Position`].
    pub fn set_param_keyframe_tangents(
        &mut self,
        clip_id: ClipId,
        param: ClipParam,
        at: RationalTime,
        tangents: Option<SpatialTangents>,
    ) -> Result<(), ModelError> {
        if param != ClipParam::Position {
            return Err(ModelError::InvalidParam(
                "spatial tangents are only supported on position".into(),
            ));
        }
        if let Some(t) = tangents {
            t.validate()?;
        }
        self.check_param_target(clip_id)?;
        let tick = self.keyframe_tick(clip_id, at)?;
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        clip.transform
            .position
            .set_keyframe_tangents(tick, tangents)
    }

    /// Remove the keyframe at exactly `at` (absolute timeline position) on
    /// one property. Errors when no keyframe sits there.
    pub fn remove_param_keyframe(
        &mut self,
        clip_id: ClipId,
        param: ClipParam,
        at: RationalTime,
    ) -> Result<(), ModelError> {
        if param == ClipParam::Volume {
            self.check_audio_param_target(clip_id)?;
            let tick = self.keyframe_tick(clip_id, at)?;
            let clip = self
                .timeline
                .clip_mut(clip_id)
                .ok_or(ModelError::UnknownClip(clip_id))?;
            return if clip.volume.remove_keyframe(tick) {
                Ok(())
            } else {
                Err(ModelError::InvalidParam(format!(
                    "no volume keyframe at {} to remove",
                    at.value
                )))
            };
        }
        if param == ClipParam::Pan {
            self.check_audio_param_target(clip_id)?;
            let tick = self.keyframe_tick(clip_id, at)?;
            let clip = self
                .timeline
                .clip_mut(clip_id)
                .ok_or(ModelError::UnknownClip(clip_id))?;
            return if clip.pan.remove_keyframe(tick) {
                Ok(())
            } else {
                Err(ModelError::InvalidParam(format!(
                    "no pan keyframe at {} to remove",
                    at.value
                )))
            };
        }
        if param == ClipParam::Crop {
            self.check_param_target(clip_id)?;
            let tick = self.keyframe_tick(clip_id, at)?;
            let clip = self
                .timeline
                .clip_mut(clip_id)
                .ok_or(ModelError::UnknownClip(clip_id))?;
            return if clip.crop.remove_keyframe(tick) {
                Ok(())
            } else {
                Err(ModelError::InvalidParam(format!(
                    "no crop keyframe at {} to remove",
                    at.value
                )))
            };
        }
        self.check_param_target(clip_id)?;
        let tick = self.keyframe_tick(clip_id, at)?;
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        match param {
            ClipParam::Effect { effect, param } => super::helpers::effect_mut(clip, effect)?
                .remove_param_keyframe(param as usize, tick),
            ClipParam::Shape { param } => {
                super::helpers::generator_mut(clip)?.remove_shape_param_keyframe(param, tick)
            }
            ClipParam::Text { param } => {
                super::helpers::generator_mut(clip)?.remove_text_param_keyframe(param, tick)
            }
            ClipParam::Look { param } if is_mask_param(param) => {
                remove_mask_param_keyframe(&mut clip.mask, param, tick)
            }
            ClipParam::Look { param } => {
                if look_param_mut(clip, param)?.remove_keyframe(tick) {
                    Ok(())
                } else {
                    Err(ModelError::InvalidParam(format!(
                        "no {param:?} keyframe at {} to remove",
                        at.value
                    )))
                }
            }
            ClipParam::Style { param } => {
                remove_style_param_keyframe(&mut clip.styles, param, tick)
            }
            _ => clip.transform.remove_param_keyframe(param, tick),
        }
    }

    /// Replace one animatable property with a constant, dropping keyframes.
    pub fn set_param_constant(
        &mut self,
        clip_id: ClipId,
        param: ClipParam,
        value: ParamValue,
    ) -> Result<(), ModelError> {
        if param == ClipParam::Volume {
            let v = super::helpers::scalar_param(value)?;
            crate::clip::validate_volume(v)?;
            self.check_audio_param_target(clip_id)?;
            let clip = self
                .timeline
                .clip_mut(clip_id)
                .ok_or(ModelError::UnknownClip(clip_id))?;
            clip.volume.set_constant(v);
            return Ok(());
        }
        if param == ClipParam::Pan {
            let v = super::helpers::scalar_param(value)?;
            crate::clip::validate_pan(v)?;
            self.check_audio_param_target(clip_id)?;
            let clip = self
                .timeline
                .clip_mut(clip_id)
                .ok_or(ModelError::UnknownClip(clip_id))?;
            clip.pan.set_constant(v);
            return Ok(());
        }
        if param == ClipParam::Crop {
            let crop = super::helpers::crop_rect_param(value)?;
            self.check_param_target(clip_id)?;
            let clip = self
                .timeline
                .clip_mut(clip_id)
                .ok_or(ModelError::UnknownClip(clip_id))?;
            clip.crop.set_constant(crop);
            return Ok(());
        }
        self.check_param_target(clip_id)?;
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        match param {
            ClipParam::Effect { effect, param } => super::helpers::effect_mut(clip, effect)?
                .set_param_value_constant(param as usize, value),
            ClipParam::Shape { param } => {
                super::helpers::generator_mut(clip)?.set_shape_param_constant(param, value)
            }
            ClipParam::Text { param } => {
                super::helpers::generator_mut(clip)?.set_text_param_constant(param, value)
            }
            ClipParam::Look { param } if is_mask_param(param) => {
                set_mask_param_constant(&mut clip.mask, param, value)
            }
            ClipParam::Look { param } => {
                let value = super::helpers::scalar_param(value)?;
                validate_look_value(param, value)?;
                look_param_mut(clip, param)?.set_constant(value);
                Ok(())
            }
            ClipParam::Style { param } => set_style_param_constant(&mut clip.styles, param, value),
            _ => clip.transform.set_param_constant(param, value),
        }
    }

    /// Expand the outgoing keyframe segment at `at` with a piecewise easing
    /// preset (bounce / elastic / back). Scalar and vec2 params only — color
    /// and crop params are rejected (they do not implement [`crate::Extrapolate`]).
    pub fn apply_easing_preset(
        &mut self,
        clip_id: ClipId,
        param: ClipParam,
        at: RationalTime,
        preset: PiecewiseEasingPreset,
    ) -> Result<(), ModelError> {
        if matches!(param, ClipParam::Crop) {
            return Err(ModelError::InvalidParam(
                "easing presets are not supported on crop (no safe extrapolation)".into(),
            ));
        }
        if is_color_clip_param(param) {
            return Err(ModelError::InvalidParam(
                "easing presets are not supported on color parameters".into(),
            ));
        }
        let tick = self.keyframe_tick(clip_id, at)?;
        match param {
            ClipParam::Volume => {
                self.check_audio_param_target(clip_id)?;
                let clip = self
                    .timeline
                    .clip_mut(clip_id)
                    .ok_or(ModelError::UnknownClip(clip_id))?;
                clip.volume.apply_easing_preset(tick, preset)
            }
            ClipParam::Pan => {
                self.check_audio_param_target(clip_id)?;
                let clip = self
                    .timeline
                    .clip_mut(clip_id)
                    .ok_or(ModelError::UnknownClip(clip_id))?;
                clip.pan.apply_easing_preset(tick, preset)
            }
            ClipParam::Position => {
                self.check_param_target(clip_id)?;
                let clip = self
                    .timeline
                    .clip_mut(clip_id)
                    .ok_or(ModelError::UnknownClip(clip_id))?;
                clip.transform.position.apply_easing_preset(tick, preset)
            }
            ClipParam::AnchorPoint => {
                self.check_param_target(clip_id)?;
                let clip = self
                    .timeline
                    .clip_mut(clip_id)
                    .ok_or(ModelError::UnknownClip(clip_id))?;
                clip.transform
                    .anchor_point
                    .apply_easing_preset(tick, preset)
            }
            ClipParam::Scale => {
                self.check_param_target(clip_id)?;
                let clip = self
                    .timeline
                    .clip_mut(clip_id)
                    .ok_or(ModelError::UnknownClip(clip_id))?;
                clip.transform.scale.apply_easing_preset(tick, preset)
            }
            ClipParam::Rotation => {
                self.check_param_target(clip_id)?;
                let clip = self
                    .timeline
                    .clip_mut(clip_id)
                    .ok_or(ModelError::UnknownClip(clip_id))?;
                clip.transform.rotation.apply_easing_preset(tick, preset)
            }
            ClipParam::Opacity => {
                self.check_param_target(clip_id)?;
                let clip = self
                    .timeline
                    .clip_mut(clip_id)
                    .ok_or(ModelError::UnknownClip(clip_id))?;
                clip.transform.opacity.apply_easing_preset(tick, preset)
            }
            ClipParam::Look { param: look } if is_mask_param(look) => {
                self.check_param_target(clip_id)?;
                let clip = self
                    .timeline
                    .clip_mut(clip_id)
                    .ok_or(ModelError::UnknownClip(clip_id))?;
                apply_mask_easing_preset(&mut clip.mask, look, tick, preset)
            }
            ClipParam::Look { param: look } => {
                self.check_param_target(clip_id)?;
                let clip = self
                    .timeline
                    .clip_mut(clip_id)
                    .ok_or(ModelError::UnknownClip(clip_id))?;
                look_param_mut(clip, look)?.apply_easing_preset(tick, preset)
            }
            ClipParam::Style {
                param: StyleParam::ShadowOffset,
            } => {
                self.check_param_target(clip_id)?;
                let clip = self
                    .timeline
                    .clip_mut(clip_id)
                    .ok_or(ModelError::UnknownClip(clip_id))?;
                style_vec2_mut(&mut clip.styles, StyleParam::ShadowOffset)?
                    .apply_easing_preset(tick, preset)
            }
            ClipParam::Style { param: style } => {
                self.check_param_target(clip_id)?;
                let clip = self
                    .timeline
                    .clip_mut(clip_id)
                    .ok_or(ModelError::UnknownClip(clip_id))?;
                style_scalar_mut(&mut clip.styles, style)?.apply_easing_preset(tick, preset)
            }
            ClipParam::Effect { .. }
            | ClipParam::Shape { .. }
            | ClipParam::Text { .. }
            | ClipParam::Speed
            | ClipParam::Crop => Err(ModelError::InvalidParam(
                "easing presets are not supported on this parameter".into(),
            )),
        }
    }

    /// Append an effect (M4) to a visual clip's chain; the id must exist in
    /// the catalog. Returns the new effect's index. Rejected on audio clips.
    pub fn add_effect(&mut self, clip_id: ClipId, effect_id: &str) -> Result<usize, ModelError> {
        let instance = EffectInstance::new(effect_id);
        // Reject unknown ids up front (validate also covers an empty chain).
        instance.validate()?;
        self.check_param_target(clip_id)?;
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        clip.effects.push(instance);
        Ok(clip.effects.len() - 1)
    }

    /// Remove the effect at `index` from a clip's chain.
    pub fn remove_effect(&mut self, clip_id: ClipId, index: usize) -> Result<(), ModelError> {
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        if index >= clip.effects.len() {
            return Err(ModelError::InvalidParam(format!(
                "effect index {index} out of range"
            )));
        }
        clip.effects.remove(index);
        Ok(())
    }

    /// Move one effect within a clip's chain. Both indices address the chain
    /// before the move; `to_index` is the effect's final index after removal
    /// and insertion. Moving an effect to its current index is a valid no-op.
    pub fn move_effect(
        &mut self,
        clip_id: ClipId,
        from_index: usize,
        to_index: usize,
    ) -> Result<(), ModelError> {
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let len = clip.effects.len();
        if from_index >= len {
            return Err(ModelError::InvalidParam(format!(
                "effect from index {from_index} out of range for chain length {len}"
            )));
        }
        if to_index >= len {
            return Err(ModelError::InvalidParam(format!(
                "effect to index {to_index} out of range for chain length {len}"
            )));
        }
        if from_index == to_index {
            return Ok(());
        }

        let effect = clip.effects.remove(from_index);
        clip.effects.insert(to_index, effect);
        Ok(())
    }

    /// Set one effect parameter to a constant (the non-animated quick edit;
    /// keyframes go through [`Self::set_param_keyframe`] with
    /// [`ClipParam::Effect`]).
    pub fn set_effect_param(
        &mut self,
        clip_id: ClipId,
        index: usize,
        param: usize,
        value: f32,
    ) -> Result<(), ModelError> {
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        super::helpers::effect_mut(clip, index as u32)?.set_param_constant(param, value)
    }
}

fn is_color_clip_param(param: ClipParam) -> bool {
    matches!(
        param,
        ClipParam::Style {
            param: StyleParam::ShadowColor
                | StyleParam::GlowColor
                | StyleParam::OutlineColor
                | StyleParam::BackgroundColor,
        } | ClipParam::Shape {
            param: crate::clip::ShapeParam::Fill | crate::clip::ShapeParam::StrokeColor,
        } | ClipParam::Text {
            param: crate::clip::TextParam::Fill
                | crate::clip::TextParam::StrokeColor
                | crate::clip::TextParam::ShadowColor
                | crate::clip::TextParam::BackgroundColor,
        }
    )
}

fn apply_mask_easing_preset(
    mask: &mut Option<Mask>,
    param: LookParam,
    tick: i64,
    preset: PiecewiseEasingPreset,
) -> Result<(), ModelError> {
    match param {
        LookParam::MaskCenter | LookParam::MaskSize => {
            mask_vec2_mut(mask, param)?.apply_easing_preset(tick, preset)
        }
        LookParam::MaskFeather | LookParam::MaskRotation | LookParam::MaskRoundness => {
            mask_scalar_mut(mask, param)?.apply_easing_preset(tick, preset)
        }
        _ => Err(ModelError::InvalidParam(format!(
            "look parameter {param:?} is not a mask geometry param"
        ))),
    }
}

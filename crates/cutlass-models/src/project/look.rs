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
    AnimationRef, AnimationSlot, AudioRole, BlendMode, ChromaKey, ColorAdjustments, Filter, Lut,
    Mask, StabilizeLevel, animation_spec,
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

use super::Project;

impl Project {
    // --- Phase I look properties (persisted + validated, render-neutral) ----

    /// The kind of the track hosting `clip_id`, or the unknown-clip error.
    fn track_kind_of(&self, clip_id: ClipId) -> Result<(TrackId, TrackKind), ModelError> {
        let track_id = self
            .timeline
            .track_of(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let kind = self
            .timeline
            .track(track_id)
            .ok_or(ModelError::UnknownTrack(track_id))?
            .kind;
        Ok((track_id, kind))
    }

    /// Guard for edits that only make sense where pixels show.
    fn require_visual(&self, clip_id: ClipId, what: &str) -> Result<(), ModelError> {
        let (track_id, kind) = self.track_kind_of(clip_id)?;
        if !kind.is_visual() {
            return Err(ModelError::IncompatibleTrackKind {
                track: track_id,
                kind,
            });
        }
        let _ = what;
        Ok(())
    }

    /// Guard for edits that need decoded source pixels (mask, chroma, …):
    /// a media-backed clip on a visual lane.
    fn require_visual_media(&self, clip_id: ClipId, what: &str) -> Result<(), ModelError> {
        self.require_visual(clip_id, what)?;
        let clip = self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        if clip.is_generated() {
            return Err(ModelError::InvalidParam(format!(
                "{what} requires a media-backed clip"
            )));
        }
        Ok(())
    }

    /// Set (or with `None` clear) a clip's shaped alpha mask (CapCut mask,
    /// Phase I — persisted now, rendered later). Media-backed visual clips
    /// only; the mask itself is validated (feather range).
    pub fn set_clip_mask(&mut self, clip_id: ClipId, mask: Option<Mask>) -> Result<(), ModelError> {
        if let Some(mask) = &mask {
            mask.validate()?;
        }
        self.require_visual_media(clip_id, "a mask")?;
        self.timeline
            .clip_mut(clip_id)
            .expect("clip existence checked above")
            .mask = mask;
        Ok(())
    }

    /// Set how the clip composites over the stack below (M-blend). Visual
    /// clips only; audio has nothing to blend.
    pub fn set_blend_mode(&mut self, clip_id: ClipId, mode: BlendMode) -> Result<(), ModelError> {
        self.require_visual(clip_id, "a blend mode")?;
        self.timeline
            .clip_mut(clip_id)
            .expect("clip existence checked above")
            .blend_mode = mode;
        Ok(())
    }

    /// Set (or clear) chroma keying (CapCut chroma key, Phase I —
    /// render-neutral). Media-backed visual clips only; strength/shadow
    /// validated to `0..=1`.
    pub fn set_clip_chroma_key(
        &mut self,
        clip_id: ClipId,
        chroma: Option<ChromaKey>,
    ) -> Result<(), ModelError> {
        if let Some(chroma) = &chroma {
            chroma.validate()?;
        }
        self.require_visual_media(clip_id, "chroma key")?;
        self.timeline
            .clip_mut(clip_id)
            .expect("clip existence checked above")
            .chroma_key = chroma;
        Ok(())
    }

    /// Set (or clear) stabilization (CapCut stabilize, Phase I —
    /// render-neutral). Media-backed *video* clips only: stills have no
    /// motion to smooth.
    pub fn set_clip_stabilize(
        &mut self,
        clip_id: ClipId,
        stabilize: Option<StabilizeLevel>,
    ) -> Result<(), ModelError> {
        self.require_visual_media(clip_id, "stabilization")?;
        let clip = self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        if let ClipSource::Media { media, .. } = &clip.content
            && self.media(*media).is_some_and(|m| m.is_image)
        {
            return Err(ModelError::InvalidParam(
                "stabilization requires video (stills have no motion)".into(),
            ));
        }
        self.timeline
            .clip_mut(clip_id)
            .expect("clip existence checked above")
            .stabilize = stabilize;
        Ok(())
    }

    /// Set (or clear) a color-grade filter preset (CapCut filters, Phase I —
    /// render-neutral). Any visual clip, including `Generator::Filter` lane
    /// bars; the id must exist in the filter catalog.
    pub fn set_clip_filter(
        &mut self,
        clip_id: ClipId,
        filter: Option<Filter>,
    ) -> Result<(), ModelError> {
        if let Some(filter) = &filter {
            filter.validate()?;
        }
        self.require_visual(clip_id, "a filter")?;
        self.timeline
            .clip_mut(clip_id)
            .expect("clip existence checked above")
            .filter = filter;
        Ok(())
    }

    /// Set (or clear) a `.cube` 3D LUT (applied after filter + adjust). Any
    /// visual clip, including `Generator::Filter` lane bars, which apply it
    /// to everything composited beneath them.
    pub fn set_clip_lut(&mut self, clip_id: ClipId, lut: Option<Lut>) -> Result<(), ModelError> {
        if let Some(lut) = &lut {
            lut.validate()?;
        }
        self.require_visual(clip_id, "a LUT")?;
        self.timeline
            .clip_mut(clip_id)
            .expect("clip existence checked above")
            .lut = lut;
        Ok(())
    }

    /// Set a clip's manual color grade (CapCut adjust, Phase I —
    /// render-neutral). Any visual clip, including `Generator::Adjustment`
    /// lane bars; neutral values simply clear the grade.
    pub fn set_clip_adjustments(
        &mut self,
        clip_id: ClipId,
        adjust: ColorAdjustments,
    ) -> Result<(), ModelError> {
        adjust.validate()?;
        self.require_visual(clip_id, "adjustments")?;
        self.timeline
            .clip_mut(clip_id)
            .expect("clip existence checked above")
            .adjust = adjust;
        Ok(())
    }

    /// Set (or clear) one animation slot (CapCut animation In / Out / Combo,
    /// Phase I — render-neutral). Any visual clip. The preset must exist in
    /// the catalog, match the slot, and — for `text_only` presets — sit on a
    /// text clip. CapCut exclusivity is enforced here so every platform
    /// agrees: setting In or Out clears a combo, setting a combo clears both
    /// sides.
    pub fn set_clip_animation(
        &mut self,
        clip_id: ClipId,
        slot: AnimationSlot,
        animation: Option<AnimationRef>,
    ) -> Result<(), ModelError> {
        self.require_visual(clip_id, "animations")?;
        let animation = if let Some(animation) = animation {
            let spec = animation_spec(&animation.id).ok_or_else(|| {
                ModelError::InvalidParam(format!("unknown animation '{}'", animation.id))
            })?;
            if spec.slot != slot {
                return Err(ModelError::InvalidParam(format!(
                    "animation '{}' does not fit that slot",
                    animation.id
                )));
            }
            if spec.text_only {
                let clip = self
                    .timeline
                    .clip(clip_id)
                    .ok_or(ModelError::UnknownClip(clip_id))?;
                if !matches!(clip.content, ClipSource::Generated(Generator::Text { .. })) {
                    return Err(ModelError::InvalidParam(format!(
                        "animation '{}' is a text preset",
                        animation.id
                    )));
                }
            }
            Some(animation.normalized_for(spec)?)
        } else {
            None
        };
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .expect("clip existence checked above");
        match slot {
            AnimationSlot::In => {
                clip.animation_in = animation;
                if clip.animation_in.is_some() {
                    clip.animation_combo = None;
                }
            }
            AnimationSlot::Out => {
                clip.animation_out = animation;
                if clip.animation_out.is_some() {
                    clip.animation_combo = None;
                }
            }
            AnimationSlot::Combo => {
                clip.animation_combo = animation;
                if clip.animation_combo.is_some() {
                    clip.animation_in = None;
                    clip.animation_out = None;
                }
            }
        }
        Ok(())
    }

    /// Tag (or untag) what an audio-lane clip *is* (music / sound FX /
    /// voiceover / extracted, Phase I). Clips on audio tracks only.
    pub fn set_clip_audio_role(
        &mut self,
        clip_id: ClipId,
        role: Option<AudioRole>,
    ) -> Result<(), ModelError> {
        let (track_id, kind) = self.track_kind_of(clip_id)?;
        if kind != TrackKind::Audio {
            return Err(ModelError::IncompatibleTrackKind {
                track: track_id,
                kind,
            });
        }
        self.timeline
            .clip_mut(clip_id)
            .expect("clip existence checked above")
            .audio_role = role;
        Ok(())
    }
}

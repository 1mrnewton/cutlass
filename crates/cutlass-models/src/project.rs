use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::Map;
use crate::clip::{
    Clip, ClipParam, ClipSource, ClipTransform, CropRect, Generator, ParamValue, Replaceable,
    SlotMedia,
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

/// Top-level container: a media pool plus exactly one [`Timeline`].
///
/// `Project` is the aggregate root and the only place that can guarantee
/// referential integrity between clips and media, so clip creation goes through
/// [`add_clip`](Project::add_clip).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    /// Document schema identity (version, kind, extensions).
    #[serde(
        serialize_with = "crate::schema::serialize",
        deserialize_with = "crate::schema::deserialize",
        alias = "schema_version"
    )]
    pub schema: ProjectSchema,
    pub id: ProjectId,
    pub name: String,
    pub metadata: ProjectMetadata,
    #[serde(with = "crate::serde_map")]
    media: Map<MediaId, MediaSource>,
    timeline: Timeline,
}

impl Project {
    /// Create an empty project whose timeline runs at `frame_rate`.
    pub fn new(name: impl Into<String>, frame_rate: Rational) -> Self {
        Self {
            schema: ProjectSchema::current(),
            id: ProjectId::next(),
            name: name.into(),
            metadata: ProjectMetadata::default(),
            media: Map::default(),
            timeline: Timeline::new(frame_rate),
        }
    }

    pub fn schema(&self) -> &ProjectSchema {
        &self.schema
    }

    pub fn metadata(&self) -> &ProjectMetadata {
        &self.metadata
    }

    pub fn metadata_mut(&mut self) -> &mut ProjectMetadata {
        &mut self.metadata
    }

    // --- media pool -------------------------------------------------------

    /// Add a source to the media pool. Returns its [`MediaId`].
    pub fn add_media(&mut self, media: MediaSource) -> MediaId {
        let id = media.id;
        self.media.insert(id, media);
        id
    }

    pub fn media(&self, id: MediaId) -> Option<&MediaSource> {
        self.media.get(&id)
    }

    /// Mutable pool access for state repair (missing-media relink, M0):
    /// re-pointing an entry at a re-probed file in place, keeping its id so
    /// clips stay attached. Editing flows must not mutate sources behind the
    /// timeline's back — placement math reads pool metadata.
    pub fn media_mut(&mut self, id: MediaId) -> Option<&mut MediaSource> {
        self.media.get_mut(&id)
    }

    pub fn media_iter(&self) -> impl Iterator<Item = &MediaSource> {
        self.media.values()
    }

    pub fn media_count(&self) -> usize {
        self.media.len()
    }

    /// Lookup a pool entry by filesystem path (canonical comparison when possible).
    pub fn find_media_by_path(&self, path: &Path) -> Option<MediaId> {
        self.media_iter()
            .find(|m| paths_refer_to_same_file(m.path(), path))
            .map(|m| m.id)
    }

    /// Whether any clip currently references `media_id`.
    pub fn is_media_referenced(&self, media_id: MediaId) -> bool {
        self.timeline
            .tracks_ordered()
            .flat_map(Track::clips)
            .any(|c| c.media() == Some(media_id))
    }

    /// Remove a source from the pool. Fails if any clip still references it.
    pub fn remove_media(&mut self, media_id: MediaId) -> Result<MediaSource, ModelError> {
        if self.is_media_referenced(media_id) {
            return Err(ModelError::MediaReferenced(media_id));
        }
        self.media
            .remove(&media_id)
            .ok_or(ModelError::UnknownMedia(media_id))
    }

    // --- timeline ---------------------------------------------------------

    pub fn timeline(&self) -> &Timeline {
        &self.timeline
    }

    pub fn timeline_mut(&mut self) -> &mut Timeline {
        &mut self.timeline
    }

    /// Convenience: create and append a track, returning its [`TrackId`].
    pub fn add_track(&mut self, kind: TrackKind, name: impl Into<String>) -> TrackId {
        self.timeline.add_track(Track::new(kind, name))
    }

    /// Convenience: create a track at `order_index` in the stack (0 = bottom
    /// layer; clamped), returning its [`TrackId`].
    pub fn insert_track(
        &mut self,
        kind: TrackKind,
        name: impl Into<String>,
        order_index: usize,
    ) -> TrackId {
        self.timeline
            .insert_track(Track::new(kind, name), order_index)
    }

    /// Place a clip referencing `media_id` on `track_id`.
    ///
    /// The clip's timeline duration is resampled from the source rate to the
    /// timeline rate. Validates media/track existence, source bounds, and overlap.
    pub fn add_clip(
        &mut self,
        track_id: TrackId,
        media_id: MediaId,
        source: TimeRange,
        timeline_start: RationalTime,
    ) -> Result<ClipId, ModelError> {
        let media = self
            .media
            .get(&media_id)
            .ok_or(ModelError::UnknownMedia(media_id))?;
        let tl_rate = self.timeline.frame_rate;

        check_same_rate(source.start.rate, media.frame_rate)?;
        check_same_rate(timeline_start.rate, tl_rate)?;

        if source.is_empty() {
            return Err(ModelError::InvalidRange);
        }
        // Stills have no real material bound: one frame repeats for any
        // extent, and the pool duration is only the default placement
        // length — so any window length is legal on image media.
        if source.start.value < 0 || (!media.is_image && source.end_tick() > media.duration.value) {
            return Err(ModelError::SourceOutOfBounds);
        }

        let timeline_duration = resample(source.duration, tl_rate);
        let duration_ticks = timeline_duration.value.max(1);
        let timeline = TimeRange::at_rate(timeline_start.value, duration_ticks, tl_rate);

        let clip = Clip::from_media(media_id, source, timeline);
        self.timeline.add_clip(track_id, clip)
    }

    /// Place a generated clip on `track_id` at the given timeline range.
    pub fn add_generated(
        &mut self,
        track_id: TrackId,
        generator: Generator,
        timeline: TimeRange,
    ) -> Result<ClipId, ModelError> {
        check_same_rate(timeline.start.rate, self.timeline.frame_rate)?;
        if timeline.is_empty() {
            return Err(ModelError::InvalidRange);
        }
        let mut generator = generator;
        generator.resolve_presets()?;
        generator.validate()?;
        let clip = Clip::generated(generator, timeline);
        self.timeline.add_clip(track_id, clip)
    }

    /// Clone `clip_id` onto `to_track` at `start`, preserving every clip
    /// property except identity, placement start, and linkage.
    ///
    /// The destination is explicit: this primitive performs no placement
    /// search or ripple. Placement validation is completed before allocating
    /// the fresh [`ClipId`], avoiding allocator advances on validation errors.
    pub fn duplicate_clip(
        &mut self,
        clip_id: ClipId,
        to_track: TrackId,
        start: RationalTime,
    ) -> Result<ClipId, ModelError> {
        let timeline_rate = self.timeline.frame_rate;
        check_same_rate(start.rate, timeline_rate)?;
        if start.value < 0 {
            return Err(ModelError::InvalidRange);
        }

        let source = self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let destination = self
            .timeline
            .track(to_track)
            .ok_or(ModelError::UnknownTrack(to_track))?;
        if !destination.kind.accepts_content(&source.content) {
            return Err(ModelError::IncompatibleTrackKind {
                track: to_track,
                kind: destination.kind,
            });
        }

        let mut destination_range = source.timeline;
        destination_range.start = start;
        if destination.has_overlap(destination_range, None)? {
            return Err(ModelError::Overlap(to_track));
        }

        // Clone first so every current and future content/property field rides
        // along by default. Duplication deliberately resets only these three
        // identity/placement fields; transitions live on tracks, not clips.
        let mut duplicate = source.clone();
        duplicate.id = ClipId::next();
        duplicate.timeline.start = start;
        duplicate.link = None;

        self.timeline.add_clip(to_track, duplicate)
    }

    /// Replace a generated clip's content (edit a title's text, recolor a
    /// shape, …). Errors if the clip is unknown, is media-backed, or the new
    /// generator isn't accepted by the clip's track.
    pub fn set_generator(
        &mut self,
        clip_id: ClipId,
        generator: Generator,
    ) -> Result<(), ModelError> {
        let track_id = self
            .timeline
            .track_of(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let kind = self
            .timeline
            .track(track_id)
            .ok_or(ModelError::UnknownTrack(track_id))?
            .kind;
        let clip = self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        // Media clips have no generator to replace; reject rather than convert.
        if !clip.is_generated() {
            return Err(ModelError::IncompatibleTrackKind {
                track: track_id,
                kind,
            });
        }
        let mut generator = generator;
        generator.resolve_presets()?;
        generator.validate()?;
        let content = ClipSource::Generated(generator);
        if !kind.accepts_content(&content) {
            return Err(ModelError::IncompatibleTrackKind {
                track: track_id,
                kind,
            });
        }
        self.timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?
            .content = content;
        Ok(())
    }

    /// Replace `clip_id`'s content with a trimmed window of `media_id`, keeping
    /// its timeline placement, speed, transform, and effects. This is the
    /// primitive behind filling a template slot and re-picking a filled slot's
    /// in-point (CapCut "tap clip → adjust which part plays"). Validates media
    /// existence, that the track accepts media content, the source rate, and —
    /// for non-image media — that the source window lies within the source.
    pub fn set_clip_media(
        &mut self,
        clip_id: ClipId,
        media_id: MediaId,
        source: TimeRange,
    ) -> Result<(), ModelError> {
        let track_id = self
            .timeline
            .track_of(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let kind = self
            .timeline
            .track(track_id)
            .ok_or(ModelError::UnknownTrack(track_id))?
            .kind;
        let media = self
            .media
            .get(&media_id)
            .ok_or(ModelError::UnknownMedia(media_id))?;
        check_same_rate(source.start.rate, media.frame_rate)?;
        if source.is_empty() {
            return Err(ModelError::InvalidRange);
        }
        // Stills repeat one frame for any window; real media must contain the
        // requested span.
        if source.start.value < 0 || (!media.is_image && source.end_tick() > media.duration.value) {
            return Err(ModelError::SourceOutOfBounds);
        }
        let content = ClipSource::Media {
            media: media_id,
            source,
        };
        if !kind.accepts_content(&content) {
            return Err(ModelError::IncompatibleTrackKind {
                track: track_id,
                kind,
            });
        }
        self.timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?
            .content = content;
        Ok(())
    }

    /// Mark (or, with `None`, unmark) a clip as a CapCut-style replaceable
    /// template slot. The marker is metadata only — it does not change what
    /// renders — and is what [`Template`](crate::Template) scans to build its
    /// list of fillable slots.
    ///
    /// Marking validates eagerly (mirroring [`set_text_editable`]
    /// (Self::set_text_editable)) so authoring mistakes surface here, not at
    /// apply time with a misleading error: the clip must be a media clip
    /// (generated content cannot be refilled), and the slot's `accepts` must
    /// match the lane — visual restrictions belong on video tracks,
    /// [`SlotMedia::AudioOnly`] on audio tracks. Errors if the clip is
    /// unknown. Unmarking never fails for known clips.
    pub fn set_replaceable(
        &mut self,
        clip_id: ClipId,
        replaceable: Option<Replaceable>,
    ) -> Result<(), ModelError> {
        if let Some(marker) = &replaceable {
            let track_id = self
                .timeline
                .track_of(clip_id)
                .ok_or(ModelError::UnknownClip(clip_id))?;
            let kind = self
                .timeline
                .track(track_id)
                .ok_or(ModelError::UnknownTrack(track_id))?
                .kind;
            let clip = self
                .timeline
                .clip(clip_id)
                .ok_or(ModelError::UnknownClip(clip_id))?;
            if clip.media().is_none() {
                return Err(ModelError::InvalidParam(
                    "replaceable slots apply only to media clips".into(),
                ));
            }
            let want = if marker.accepts == SlotMedia::AudioOnly {
                TrackKind::Audio
            } else {
                TrackKind::Video
            };
            if kind != want {
                return Err(ModelError::InvalidParam(format!(
                    "a {:?} slot cannot sit on a {kind:?} track",
                    marker.accepts
                )));
            }
        }
        self.timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?
            .replaceable = replaceable;
        Ok(())
    }

    /// Mark a text clip's content as user-editable when the project is used as
    /// a template (the text keeps its style and animation). Errors if the clip
    /// is unknown or is not a [`Generator::Text`] clip.
    pub fn set_text_editable(&mut self, clip_id: ClipId, editable: bool) -> Result<(), ModelError> {
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        if !matches!(clip.content, ClipSource::Generated(Generator::Text { .. })) {
            return Err(ModelError::InvalidParam(
                "text editability applies only to text generator clips".into(),
            ));
        }
        clip.text_editable = editable;
        Ok(())
    }

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

    /// Precondition for volume-envelope edits (M8): the clip exists and is
    /// media-backed (generators have nothing to hear). Mirrors
    /// [`Self::set_clip_audio`]'s target rule — volume rides any media clip,
    /// since linkage lands the audible half on an audio lane.
    fn check_audio_param_target(&self, clip_id: ClipId) -> Result<(), ModelError> {
        let clip = self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        if clip.is_generated() {
            return Err(ModelError::InvalidParam(
                "volume requires a media-backed clip".into(),
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
    pub fn set_param_keyframe(
        &mut self,
        clip_id: ClipId,
        param: ClipParam,
        at: RationalTime,
        value: ParamValue,
        easing: Easing,
    ) -> Result<(), ModelError> {
        // Volume (M8) is an audio property, not a transform: validate the
        // gain range and an audio-capable target, then write to the envelope.
        if param == ClipParam::Volume {
            easing.validate()?;
            let v = scalar_param(value)?;
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
        self.check_param_target(clip_id)?;
        let tick = self.keyframe_tick(clip_id, at)?;
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        match param {
            ClipParam::Effect { effect, param } => {
                let v = scalar_param(value)?;
                effect_mut(clip, effect)?.set_param_keyframe(param as usize, tick, v, easing)
            }
            ClipParam::Shape { param } => {
                generator_mut(clip)?.set_shape_param_keyframe(param, tick, value, easing)
            }
            _ => clip
                .transform
                .set_param_keyframe(param, tick, value, easing),
        }
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
        self.check_param_target(clip_id)?;
        let tick = self.keyframe_tick(clip_id, at)?;
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        match param {
            ClipParam::Effect { effect, param } => {
                effect_mut(clip, effect)?.remove_param_keyframe(param as usize, tick)
            }
            ClipParam::Shape { param } => {
                generator_mut(clip)?.remove_shape_param_keyframe(param, tick)
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
            let v = scalar_param(value)?;
            crate::clip::validate_volume(v)?;
            self.check_audio_param_target(clip_id)?;
            let clip = self
                .timeline
                .clip_mut(clip_id)
                .ok_or(ModelError::UnknownClip(clip_id))?;
            clip.volume.set_constant(v);
            return Ok(());
        }
        self.check_param_target(clip_id)?;
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        match param {
            ClipParam::Effect { effect, param } => {
                let v = scalar_param(value)?;
                effect_mut(clip, effect)?.set_param_constant(param as usize, v)
            }
            ClipParam::Shape { param } => {
                generator_mut(clip)?.set_shape_param_constant(param, value)
            }
            _ => clip.transform.set_param_constant(param, value),
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
        effect_mut(clip, index as u32)?.set_param_constant(param, value)
    }

    // --- transitions (M4) -------------------------------------------------

    /// Add (or replace) a transition at the junction where `left` abuts the
    /// next clip on its track. The catalog id must exist and `left` must abut
    /// a following clip. Uses the default window length.
    pub fn add_transition(&mut self, left: ClipId, transition_id: &str) -> Result<(), ModelError> {
        if crate::transition::transition_spec(transition_id).is_none() {
            return Err(ModelError::InvalidParam(format!(
                "unknown transition '{transition_id}'"
            )));
        }
        let track_id = self
            .timeline
            .track_of(left)
            .ok_or(ModelError::UnknownClip(left))?;
        let kind = self
            .timeline
            .track(track_id)
            .ok_or(ModelError::UnknownClip(left))?
            .kind;
        if !kind.supports_transitions() {
            return Err(ModelError::InvalidParam(format!(
                "transitions aren't supported on {kind:?} lanes"
            )));
        }
        let right = self.right_neighbor(track_id, left).ok_or_else(|| {
            ModelError::InvalidParam("clip has no abutting clip to its right".into())
        })?;
        let transition = Transition::new(
            left,
            right,
            transition_id,
            crate::transition::DEFAULT_TRANSITION_TICKS,
        );
        self.timeline
            .track_mut(track_id)
            .ok_or(ModelError::UnknownClip(left))?
            .upsert_transition(transition);
        Ok(())
    }

    /// Remove the transition at the `left` junction. Errors if none exists.
    pub fn remove_transition(&mut self, left: ClipId) -> Result<(), ModelError> {
        let track_id = self
            .timeline
            .track_of(left)
            .ok_or(ModelError::UnknownClip(left))?;
        let removed = self
            .timeline
            .track_mut(track_id)
            .and_then(|t| t.remove_transition(left));
        if removed.is_none() {
            return Err(ModelError::InvalidParam(
                "clip has no transition at its right junction".into(),
            ));
        }
        Ok(())
    }

    /// Set the window length (timeline ticks) of an existing transition.
    pub fn set_transition_duration(
        &mut self,
        left: ClipId,
        duration: i64,
    ) -> Result<(), ModelError> {
        let track_id = self
            .timeline
            .track_of(left)
            .ok_or(ModelError::UnknownClip(left))?;
        let transition = self
            .timeline
            .track_mut(track_id)
            .and_then(|t| t.transition_at_mut(left))
            .ok_or_else(|| {
                ModelError::InvalidParam("clip has no transition at its right junction".into())
            })?;
        transition.duration = duration.max(1);
        Ok(())
    }

    /// The clip on `track` whose start abuts the end of `left`, if any.
    fn right_neighbor(&self, track_id: TrackId, left: ClipId) -> Option<ClipId> {
        let track = self.timeline.track(track_id)?;
        let left_end = track.clip(left)?.timeline.end_tick();
        track
            .clips()
            .find(|c| c.id != left && c.timeline.start.value == left_end)
            .map(|c| c.id)
    }

    /// Whether any track carries a transition (cheap guard for prune).
    pub fn has_transitions(&self) -> bool {
        self.timeline
            .tracks_ordered()
            .any(|t| !t.transitions().is_empty())
    }

    /// Snapshot every track's transitions (for undo of structural edits that
    /// prune dead junctions).
    pub fn transitions_snapshot(&self) -> Vec<(TrackId, Vec<Transition>)> {
        self.timeline
            .tracks_ordered()
            .map(|t| (t.id, t.transitions().to_vec()))
            .collect()
    }

    /// Restore a [`Self::transitions_snapshot`] across tracks.
    pub fn restore_transitions(&mut self, snapshot: Vec<(TrackId, Vec<Transition>)>) {
        for (track_id, transitions) in snapshot {
            if let Some(track) = self.timeline.track_mut(track_id) {
                track.set_transitions(transitions);
            }
        }
    }

    /// Drop transitions whose junction no longer abuts, across all tracks.
    /// Returns whether anything was pruned.
    pub fn prune_dead_transitions(&mut self) -> bool {
        let track_ids: Vec<TrackId> = self.timeline.tracks_ordered().map(|t| t.id).collect();
        let mut pruned = false;
        for id in track_ids {
            if let Some(track) = self.timeline.track_mut(id) {
                pruned |= track.prune_dead_transitions();
            }
        }
        pruned
    }

    /// Find a clip by ID anywhere on the timeline (O(1)).
    pub fn clip(&self, clip_id: ClipId) -> Option<&Clip> {
        self.timeline.clip(clip_id)
    }

    // --- editing primitives ----------------------------------------------

    pub fn remove_clip(&mut self, clip_id: ClipId) -> Option<Clip> {
        self.timeline.remove_clip(clip_id)
    }

    /// Split the clip at timeline position `at` into two abutting clips.
    pub fn split_clip(&mut self, clip_id: ClipId, at: RationalTime) -> Result<ClipId, ModelError> {
        let clip = self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let tl = clip.timeline;
        let tl_rate = self.timeline.frame_rate;
        check_same_rate(at.rate, tl_rate)?;

        if at.value <= tl.start.value || at.value >= tl.end_tick() {
            return Err(ModelError::InvalidRange);
        }

        let left_tl = TimeRange::at_rate(tl.start.value, at.value - tl.start.value, tl_rate);
        let right_tl = TimeRange::at_rate(at.value, tl.end_tick() - at.value, tl_rate);

        let (new_left_source, mut new_clip) = match clip.content.clone() {
            ClipSource::Media { media, source } => {
                let media_fps = self
                    .media
                    .get(&media)
                    .ok_or(ModelError::UnknownMedia(media))?
                    .frame_rate;
                if source.duration.value < 2 {
                    return Err(ModelError::InvalidRange);
                }
                // Source consumed by the left half scales with the clip's
                // speed (1:1 for never-retimed clips).
                let left_src_dur = clip
                    .scale_by_speed(resample(left_tl.duration, media_fps).value)
                    .clamp(1, source.duration.value - 1);
                // A reversed clip plays its window backward: the timeline's
                // left half shows the source window's TOP, so the split
                // hands the window bottom to the right clip.
                let (left_src_start, right_src_start) = if clip.reversed {
                    (
                        source.start.value + source.duration.value - left_src_dur,
                        source.start.value,
                    )
                } else {
                    (source.start.value, source.start.value + left_src_dur)
                };
                let left_source = TimeRange::at_rate(left_src_start, left_src_dur, media_fps);
                let right_source = TimeRange::at_rate(
                    right_src_start,
                    source.duration.value - left_src_dur,
                    media_fps,
                );
                let mut right = Clip::from_media(media, right_source, right_tl);
                // The retiming rides along on both halves.
                right.speed = clip.speed;
                right.reversed = clip.reversed;
                // Audio mix splits CapCut-style: the volume envelope rides on
                // both halves, the fade-in stays with the head, the fade-out
                // with the tail.
                right.volume = clip.volume.clone();
                right.fade_out = clip.fade_out;
                (Some(left_source), right)
            }
            ClipSource::Generated(generator) => (None, Clip::generated(generator, right_tl)),
        };
        // Framing is identical on both halves: crop and flips ride along.
        new_clip.crop = clip.crop;
        new_clip.flip_h = clip.flip_h;
        new_clip.flip_v = clip.flip_v;
        // The effect chain copies to both halves (same as crop): each half is
        // an independent clip that keeps the full chain.
        new_clip.effects = clip.effects.clone();
        // Beat markers are stored in source time, so both halves keep the full
        // list — each clip's `beat_timeline_ticks` shows only the beats inside
        // its own (now-divided) source window.
        new_clip.beats = clip.beats.clone();

        let track_id = self
            .timeline
            .track_of(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;

        {
            let left = self
                .timeline
                .clip_mut(clip_id)
                .expect("clip existence checked above");
            left.timeline = left_tl;
            // The tail's fade-out moved to the right half.
            left.fade_out = 0;
            if let (Some(src), ClipSource::Media { source, .. }) =
                (new_left_source, &mut left.content)
            {
                *source = src;
            }
        }
        self.timeline.add_clip(track_id, new_clip)
    }

    /// Set the clip's timeline placement to `new_timeline` (trim/extend).
    pub fn trim_clip(
        &mut self,
        clip_id: ClipId,
        new_timeline: TimeRange,
    ) -> Result<(), ModelError> {
        if new_timeline.is_empty() {
            return Err(ModelError::InvalidRange);
        }
        let track_id = self
            .timeline
            .track_of(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let clip = self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let old_tl = clip.timeline;
        let tl_rate = self.timeline.frame_rate;
        check_same_rate(new_timeline.start.rate, tl_rate)?;

        if self
            .timeline
            .track(track_id)
            .expect("clip is on a track")
            .has_overlap(new_timeline, Some(clip_id))?
        {
            return Err(ModelError::Overlap(track_id));
        }

        let new_source = match clip.content.clone() {
            ClipSource::Media { media, source } => {
                let media = self
                    .media
                    .get(&media)
                    .ok_or(ModelError::UnknownMedia(media))?;
                // Source ticks consumed per timeline tick scale with the
                // clip's speed (1:1 for never-retimed clips).
                let head_delta = clip.scale_by_speed(
                    resample(
                        RationalTime::new(new_timeline.start.value - old_tl.start.value, tl_rate),
                        media.frame_rate,
                    )
                    .value,
                );
                let new_src_dur = clip
                    .scale_by_speed(resample(new_timeline.duration, media.frame_rate).value)
                    .max(1);
                // A reversed clip plays its window backward, so the
                // timeline head shows the window's END: a head trim drops
                // source from the top, a tail trim from the bottom —
                // mirror-image of the forward case.
                let new_src_start = if clip.reversed {
                    source.start.value + source.duration.value - new_src_dur - head_delta
                } else {
                    source.start.value + head_delta
                };
                // Stills extend freely past the pool's default 5s window —
                // the one frame repeats and decode ignores the window, so
                // the source range is duration bookkeeping only. Clamp the
                // start to 0 so extensions stay canonical.
                if media.is_image {
                    Some(TimeRange::at_rate(
                        new_src_start.max(0),
                        new_src_dur,
                        media.frame_rate,
                    ))
                } else {
                    if new_src_start < 0 || new_src_start + new_src_dur > media.duration.value {
                        return Err(ModelError::SourceOutOfBounds);
                    }
                    Some(TimeRange::at_rate(
                        new_src_start,
                        new_src_dur,
                        media.frame_rate,
                    ))
                }
            }
            ClipSource::Generated(_) => None,
        };

        let clip = self
            .timeline
            .clip_mut(clip_id)
            .expect("clip existence checked above");
        clip.timeline = new_timeline;
        if let (Some(src), ClipSource::Media { source, .. }) = (new_source, &mut clip.content) {
            *source = src;
        }
        Ok(())
    }

    /// Retime a media clip (CapCut speed, M1): keep its timeline start and
    /// source window, set `speed`/`reversed`, and re-derive the timeline
    /// duration (source duration ÷ speed — faster clips occupy less
    /// timeline). Rejected on generated clips (no source to retime), on
    /// non-positive speeds, and when the retimed extent would overlap a
    /// neighbor.
    pub fn set_clip_speed(
        &mut self,
        clip_id: ClipId,
        speed: Rational,
        reversed: bool,
    ) -> Result<(), ModelError> {
        if speed.num <= 0 || speed.den <= 0 {
            return Err(ModelError::InvalidParam("speed must be positive".into()));
        }
        let track_id = self
            .timeline
            .track_of(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let clip = self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let Some(source) = clip.source_range() else {
            return Err(ModelError::InvalidParam(
                "speed requires a media-backed clip".into(),
            ));
        };
        let tl_rate = self.timeline.frame_rate;
        let src_dur_tl = resample(source.duration, tl_rate).value;
        // Faster average ⇒ less timeline. A flat ramp keeps the exact integer
        // path (no f64 drift); any active ramp folds in its average.
        let new_dur = retimed_duration(
            src_dur_tl,
            speed,
            clip.speed_curve_average(),
            clip.has_speed_curve(),
        );
        let new_timeline = TimeRange::at_rate(clip.timeline.start.value, new_dur, tl_rate);

        if self
            .timeline
            .track(track_id)
            .expect("clip is on a track")
            .has_overlap(new_timeline, Some(clip_id))?
        {
            return Err(ModelError::Overlap(track_id));
        }

        let clip = self
            .timeline
            .clip_mut(clip_id)
            .expect("clip existence checked above");
        clip.speed = speed;
        clip.reversed = reversed;
        clip.timeline = new_timeline;
        Ok(())
    }

    /// Set (or clear) a media clip's playback-rate ramp (CapCut speed curves,
    /// M2): keep its timeline start, base `speed`, and source window; store
    /// the normalized `curve` (`None` clears it to a flat unit ramp); and
    /// re-derive the timeline duration from `source ÷ (base_speed ×
    /// average_curve)`. Rejected on generated clips, malformed curves, and
    /// when the retimed extent would overlap a neighbor.
    pub fn set_clip_speed_curve(
        &mut self,
        clip_id: ClipId,
        curve: Option<Param<f32>>,
    ) -> Result<(), ModelError> {
        if let Some(curve) = &curve {
            crate::clip::validate_speed_curve(curve)?;
        }
        let track_id = self
            .timeline
            .track_of(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let clip = self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let Some(source) = clip.source_range() else {
            return Err(ModelError::InvalidParam(
                "speed ramps require a media-backed clip".into(),
            ));
        };
        let new_curve = curve.unwrap_or(Param::Constant(1.0));
        let has_curve = !matches!(&new_curve, Param::Constant(v) if *v == 1.0);
        let average = match &new_curve {
            Param::Constant(v) => f64::from(*v),
            Param::Keyframed { .. } => {
                // Reuse the clip's integral over the candidate curve.
                let mut probe = clip.clone();
                probe.speed_curve = new_curve.clone();
                probe.speed_curve_average()
            }
        };

        let tl_rate = self.timeline.frame_rate;
        let src_dur_tl = resample(source.duration, tl_rate).value;
        let new_dur = retimed_duration(src_dur_tl, clip.speed, average, has_curve);
        let new_timeline = TimeRange::at_rate(clip.timeline.start.value, new_dur, tl_rate);

        if self
            .timeline
            .track(track_id)
            .expect("clip is on a track")
            .has_overlap(new_timeline, Some(clip_id))?
        {
            return Err(ModelError::Overlap(track_id));
        }

        let clip = self
            .timeline
            .clip_mut(clip_id)
            .expect("clip existence checked above");
        clip.speed_curve = new_curve;
        clip.timeline = new_timeline;
        Ok(())
    }

    /// Toggle whether a retimed media clip preserves pitch while it plays
    /// (CapCut's "pitch" switch, M8 Phase 3): `true` time-stretches so the
    /// audio keeps its pitch, `false` lets pitch ride the speed ("chipmunk").
    /// Pure audio property — it changes no duration, so there is no overlap
    /// check. Rejected on generated clips (nothing to hear).
    pub fn set_clip_pitch(
        &mut self,
        clip_id: ClipId,
        preserve_pitch: bool,
    ) -> Result<(), ModelError> {
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        if clip.source_range().is_none() {
            return Err(ModelError::InvalidParam(
                "pitch lock requires a media-backed clip".into(),
            ));
        }
        clip.preserve_pitch = preserve_pitch;
        Ok(())
    }

    /// Toggle noise reduction on a media clip (CapCut "Reduce noise", M8
    /// Phase 5), returning the previous flag for the inverse. The mixers run
    /// the clip's audio through RNNoise when set. Rejected on generated clips
    /// (no source audio to clean).
    pub fn set_clip_denoise(&mut self, clip_id: ClipId, denoise: bool) -> Result<bool, ModelError> {
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        if clip.source_range().is_none() {
            return Err(ModelError::InvalidParam(
                "noise reduction requires a media-backed clip".into(),
            ));
        }
        Ok(std::mem::replace(&mut clip.denoise, denoise))
    }

    /// Set a media clip's audio mix (CapCut volume + fades): `volume` is
    /// `Some` to set a flat gain (`0` mutes, `1` unchanged, up to
    /// [`crate::MAX_CLIP_VOLUME`]× boost), overwriting any M8 envelope
    /// (CapCut's basic slider), or `None` to keep the current gain and only
    /// update the fades — so a fade edit never flattens an envelope. Fades
    /// are linear in/out durations at the timeline rate. Rejected on
    /// generated clips (nothing to hear), out-of-range volume, negative
    /// fades, and fades longer than the clip.
    pub fn set_clip_audio(
        &mut self,
        clip_id: ClipId,
        volume: Option<f32>,
        fade_in: RationalTime,
        fade_out: RationalTime,
    ) -> Result<(), ModelError> {
        if let Some(volume) = volume {
            crate::clip::validate_volume(volume)?;
        }
        let tl_rate = self.timeline.frame_rate;
        check_same_rate(fade_in.rate, tl_rate)?;
        check_same_rate(fade_out.rate, tl_rate)?;

        let clip = self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        if clip.is_generated() {
            return Err(ModelError::InvalidParam(
                "volume requires a media-backed clip".into(),
            ));
        }
        let duration = clip.timeline.duration.value;
        for (name, fade) in [("fade_in", fade_in.value), ("fade_out", fade_out.value)] {
            if fade < 0 {
                return Err(ModelError::InvalidParam(format!("{name} must be ≥ 0")));
            }
            if fade > duration {
                return Err(ModelError::InvalidParam(format!(
                    "{name} ({fade} ticks) is longer than the clip ({duration} ticks)"
                )));
            }
        }

        let clip = self
            .timeline
            .clip_mut(clip_id)
            .expect("clip existence checked above");
        // `Some` is the basic volume slider: a flat level that flattens any
        // envelope (CapCut). `None` keeps the gain (constant or envelope) and
        // touches only the fades, so a fade edit never destroys automation;
        // envelopes are otherwise drawn through the volume keyframe commands
        // (`ClipParam::Volume`).
        if let Some(volume) = volume {
            clip.volume = Param::Constant(volume);
        }
        clip.fade_in = fade_in.value;
        clip.fade_out = fade_out.value;
        Ok(())
    }

    /// Set a clip's framing (CapCut crop, M1): the normalized kept region
    /// plus horizontal/vertical mirroring. Visual clips only — audio has no
    /// frame to crop. Rejected on a degenerate or out-of-frame crop rect.
    pub fn set_clip_crop(
        &mut self,
        clip_id: ClipId,
        crop: CropRect,
        flip_h: bool,
        flip_v: bool,
    ) -> Result<(), ModelError> {
        crop.validate()?;
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
        let clip = self
            .timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        clip.crop = crop;
        clip.flip_h = flip_h;
        clip.flip_v = flip_v;
        Ok(())
    }

    /// Replace a clip's detected beat markers (M8 Phase 6), returning the
    /// previous list for the inverse. Beats are source ticks at the media
    /// frame rate; this only stores what detection (or a clear) produced —
    /// the engine owns the analysis. Media clips only (generated content has
    /// no audio to analyze). Stored sorted + de-duplicated and clamped to the
    /// source window so a stale list can't snap to phantom positions.
    pub fn set_clip_beats(
        &mut self,
        clip_id: ClipId,
        beats: Vec<i64>,
    ) -> Result<Vec<i64>, ModelError> {
        let clip = self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let source = clip
            .source_range()
            .ok_or_else(|| ModelError::InvalidParam("beats require a media-backed clip".into()))?;
        let (lo, hi) = (
            source.start.value,
            source.start.value + source.duration.value,
        );
        let mut beats: Vec<i64> = beats
            .into_iter()
            .filter(|&b| (lo..hi).contains(&b))
            .collect();
        beats.sort_unstable();
        beats.dedup();

        let clip = self
            .timeline
            .clip_mut(clip_id)
            .expect("clip existence checked above");
        Ok(std::mem::replace(&mut clip.beats, beats))
    }

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
        if let Some(animation) = &animation {
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
        }
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

    /// Move a clip to `to_track` at `new_start`, preserving duration and source.
    pub fn move_clip(
        &mut self,
        clip_id: ClipId,
        to_track: TrackId,
        new_start: RationalTime,
    ) -> Result<(), ModelError> {
        let tl_rate = self.timeline.frame_rate;
        check_same_rate(new_start.rate, tl_rate)?;
        if new_start.value < 0 {
            return Err(ModelError::InvalidRange);
        }

        let from_track = self
            .timeline
            .track_of(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let clip_content = &self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?
            .content;

        let dest = self
            .timeline
            .track(to_track)
            .ok_or(ModelError::UnknownTrack(to_track))?;

        if !dest.kind.accepts_content(clip_content) {
            return Err(ModelError::IncompatibleTrackKind {
                track: to_track,
                kind: dest.kind,
            });
        }

        let duration = self
            .timeline
            .clip(clip_id)
            .expect("clip is on a track")
            .timeline
            .duration
            .value;
        let new_tl = TimeRange::at_rate(new_start.value, duration, tl_rate);

        let ignore = (from_track == to_track).then_some(clip_id);
        if dest.has_overlap(new_tl, ignore)? {
            return Err(ModelError::Overlap(to_track));
        }

        if from_track == to_track {
            self.timeline
                .clip_mut(clip_id)
                .expect("clip is on a track")
                .timeline = new_tl;
        } else {
            let mut clip = self
                .timeline
                .remove_clip(clip_id)
                .expect("clip is on a track");
            clip.timeline = new_tl;
            self.timeline.add_clip(to_track, clip)?;
        }
        Ok(())
    }

    /// Shift every clip on `track_id` whose start is at or after `from` by
    /// `delta` ticks (ripple primitive: opens a hole for an insert when
    /// positive, closes a gap when negative).
    ///
    /// Validated atomically: shifting left must not collide with the nearest
    /// unshifted clip or push the first shifted clip below tick 0. Relative
    /// spacing among the shifted clips is preserved, so no other overlap can
    /// arise. Returns the new start of the first shifted clip, or `None` when
    /// no clip starts at/after `from` (a no-op).
    pub fn shift_clips(
        &mut self,
        track_id: TrackId,
        from: RationalTime,
        delta: RationalTime,
    ) -> Result<Option<RationalTime>, ModelError> {
        let tl_rate = self.timeline.frame_rate;
        check_same_rate(from.rate, tl_rate)?;
        check_same_rate(delta.rate, tl_rate)?;
        if delta.value == 0 {
            return Err(ModelError::InvalidRange);
        }

        let track = self
            .timeline
            .track(track_id)
            .ok_or(ModelError::UnknownTrack(track_id))?;

        // Clips never overlap, so the shifted set is a contiguous suffix in
        // start order; only its first member can collide when moving left.
        let mut first_shifted: Option<i64> = None;
        let mut prev_end: i64 = 0;
        for clip in track.clips() {
            let start = clip.timeline.start.value;
            if start >= from.value {
                first_shifted = Some(first_shifted.map_or(start, |s| s.min(start)));
            } else {
                prev_end = prev_end.max(clip.timeline.end_tick());
            }
        }
        let Some(first) = first_shifted else {
            return Ok(None);
        };

        let new_first = first + delta.value;
        if new_first < 0 {
            return Err(ModelError::InvalidRange);
        }
        if delta.value < 0 && new_first < prev_end {
            return Err(ModelError::Overlap(track_id));
        }

        let track = self
            .timeline
            .track_mut(track_id)
            .expect("track existence checked above");
        for clip in track.clips_mut() {
            if clip.timeline.start.value >= from.value {
                clip.timeline.start = time_add(&clip.timeline.start, &delta)?;
            }
        }
        Ok(Some(RationalTime::new(new_first, tl_rate)))
    }

    /// Delete a clip and slide later clips on its track left to close the gap.
    pub fn ripple_delete(&mut self, clip_id: ClipId) -> Result<Clip, ModelError> {
        let track_id = self
            .timeline
            .track_of(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let removed = self
            .timeline
            .remove_clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?;
        let gap_start = removed.timeline.start;
        let gap = removed.timeline.duration;

        let track = self
            .timeline
            .track_mut(track_id)
            .expect("track existence checked above");
        for clip in track.clips_mut() {
            if clip.timeline.start.value >= gap_start.value {
                clip.timeline.start = time_sub(&clip.timeline.start, &gap)?;
            }
        }
        Ok(removed)
    }
}

fn paths_refer_to_same_file(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(ca), Ok(cb)) => ca == cb,
        _ => false,
    }
}

/// The effect at `index` on a clip's chain, or an out-of-range error.
fn effect_mut(clip: &mut Clip, index: u32) -> Result<&mut EffectInstance, ModelError> {
    clip.effects
        .get_mut(index as usize)
        .ok_or_else(|| ModelError::InvalidParam(format!("effect index {index} out of range")))
}

/// A generated clip's generator, or an error for media-backed clips (shape
/// params route here; the generator itself rejects non-shape kinds).
fn generator_mut(clip: &mut Clip) -> Result<&mut Generator, ModelError> {
    match &mut clip.content {
        ClipSource::Generated(generator) => Ok(generator),
        ClipSource::Media { .. } => Err(ModelError::InvalidParam(
            "shape parameters apply only to generated clips".into(),
        )),
    }
}

/// Timeline ticks a retimed clip occupies: `source ÷ (base_speed × average
/// ramp)`. A flat ramp keeps the exact integer division M1 used (no f64
/// drift on the common constant-speed path); an active ramp folds in its
/// average multiplier. Always at least one tick.
fn retimed_duration(src_dur_tl: i64, speed: Rational, average: f64, has_curve: bool) -> i64 {
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

/// Unwrap a scalar [`ParamValue`] (effect params are always scalar).
fn scalar_param(value: ParamValue) -> Result<f32, ModelError> {
    match value {
        ParamValue::Scalar(v) => Ok(v),
        ParamValue::Vec2(_) | ParamValue::Color(_) => Err(ModelError::InvalidParam(
            "effect parameters take a scalar value".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    use crate::clip::Shape;

    const R24: Rational = Rational::FPS_24;
    const R30: Rational = Rational::FPS_30;

    fn rt(value: i64) -> RationalTime {
        RationalTime::new(value, R24)
    }

    fn tr(start: i64, duration: i64) -> TimeRange {
        TimeRange::at_rate(start, duration, R24)
    }

    fn tr_at(start: i64, duration: i64, rate: Rational) -> TimeRange {
        TimeRange::at_rate(start, duration, rate)
    }

    fn sample_media(fps: Rational, duration: i64) -> MediaSource {
        MediaSource::new("/tmp/sample.mp4", 1920, 1080, fps, duration, true)
    }

    fn project_with_media(duration: i64) -> (Project, MediaId, TrackId) {
        let mut project = Project::new("test", R24);
        let media_id = project.add_media(sample_media(R24, duration));
        let track = project.add_track(TrackKind::Video, "V1");
        (project, media_id, track)
    }

    // --- shape params -------------------------------------------------------

    #[test]
    fn shape_params_keyframe_through_project_api() {
        let mut project = Project::new("p", R24);
        let track = project.add_track(TrackKind::Sticker, "S1");
        let clip = project
            .add_generated(
                track,
                Generator::shape(Shape::Ellipse, [255, 255, 255, 255]),
                tr(100, 48),
            )
            .unwrap();

        // Keyframe ticks are clip-relative: timeline 100/140 → ticks 0/40.
        let width = ClipParam::Shape {
            param: crate::ShapeParam::Width,
        };
        project
            .set_param_keyframe(
                clip,
                width,
                rt(100),
                ParamValue::Scalar(100.0),
                Easing::Linear,
            )
            .unwrap();
        project
            .set_param_keyframe(
                clip,
                width,
                rt(140),
                ParamValue::Scalar(500.0),
                Easing::Linear,
            )
            .unwrap();
        let ClipSource::Generated(Generator::Shape { width: w, .. }) =
            &project.clip(clip).unwrap().content
        else {
            panic!("expected shape");
        };
        assert_eq!(w.sample(20), 300.0);

        // Constants flatten; out-of-range and wrong-kind values are rejected
        // with the clip unchanged.
        project
            .set_param_constant(clip, width, ParamValue::Scalar(250.0))
            .unwrap();
        assert!(
            project
                .set_param_keyframe(
                    clip,
                    width,
                    rt(100),
                    ParamValue::Color([0; 4]),
                    Easing::Linear
                )
                .is_err()
        );
        assert!(
            project
                .set_param_constant(clip, width, ParamValue::Scalar(f32::NAN))
                .is_err()
        );

        // Media clips reject shape params outright.
        let media = project.add_media(sample_media(R24, 500));
        let vtrack = project.add_track(TrackKind::Video, "V1");
        let mclip = project.add_clip(vtrack, media, tr(0, 100), rt(0)).unwrap();
        assert!(
            project
                .set_param_constant(mclip, width, ParamValue::Scalar(10.0))
                .is_err()
        );
    }

    #[test]
    fn add_generated_rejects_invalid_shape() {
        let mut project = Project::new("p", R24);
        let track = project.add_track(TrackKind::Sticker, "S1");
        let bad = Generator::shape(Shape::Polygon { sides: 2 }, [255, 255, 255, 255]);
        assert!(matches!(
            project.add_generated(track, bad, tr(0, 48)),
            Err(ModelError::InvalidParam(_))
        ));
    }

    // --- transitions (M4) -------------------------------------------------

    /// Two abutting solid clips on one sticker track (a lane kind that
    /// supports transitions); returns `(project, left, right, track)`.
    fn project_with_abutting_pair() -> (Project, ClipId, ClipId, TrackId) {
        let mut project = Project::new("test", R24);
        let track = project.add_track(TrackKind::Sticker, "S1");
        let left = project
            .add_generated(
                track,
                Generator::SolidColor {
                    rgba: [255, 0, 0, 255],
                },
                tr(0, 24),
            )
            .unwrap();
        let right = project
            .add_generated(
                track,
                Generator::SolidColor {
                    rgba: [0, 0, 255, 255],
                },
                tr(24, 24),
            )
            .unwrap();
        (project, left, right, track)
    }

    #[test]
    fn add_transition_links_abutting_pair() {
        let (mut project, left, right, track) = project_with_abutting_pair();
        project.add_transition(left, "crossfade").unwrap();
        let t = project
            .timeline()
            .track(track)
            .unwrap()
            .transition_at(left)
            .unwrap();
        assert_eq!(t.right, right);
        assert_eq!(t.transition_id, "crossfade");
        assert_eq!(t.duration, crate::transition::DEFAULT_TRANSITION_TICKS);
    }

    #[test]
    fn add_transition_rejects_unknown_id_and_non_abutting() {
        let (mut project, left, _right, _track) = project_with_abutting_pair();
        assert!(matches!(
            project.add_transition(left, "warp_speed"),
            Err(ModelError::InvalidParam(_))
        ));
        // A lone clip with no right neighbor cannot take a transition.
        let track = project.add_track(TrackKind::Sticker, "S2");
        let lone = project
            .add_generated(
                track,
                Generator::SolidColor {
                    rgba: [0, 255, 0, 255],
                },
                tr(0, 24),
            )
            .unwrap();
        assert!(matches!(
            project.add_transition(lone, "crossfade"),
            Err(ModelError::InvalidParam(_))
        ));
    }

    #[test]
    fn add_transition_rejects_canvas_pass_lanes() {
        // Effect/filter/adjustment segments resolve to canvas-wide passes,
        // which the renderer can't nest inside a transition — the model
        // refuses the junction outright.
        let mut project = Project::new("test", R24);
        let track = project.add_track(TrackKind::Adjustment, "FX");
        let left = project
            .add_generated(track, Generator::Adjustment, tr(0, 24))
            .unwrap();
        project
            .add_generated(track, Generator::Adjustment, tr(24, 24))
            .unwrap();
        assert!(matches!(
            project.add_transition(left, "crossfade"),
            Err(ModelError::InvalidParam(_))
        ));
        assert!(!project.has_transitions());
    }

    #[test]
    fn set_and_remove_transition_duration() {
        let (mut project, left, _right, track) = project_with_abutting_pair();
        project.add_transition(left, "wipe_left").unwrap();
        project.set_transition_duration(left, 12).unwrap();
        assert_eq!(
            project
                .timeline()
                .track(track)
                .unwrap()
                .transition_at(left)
                .unwrap()
                .duration,
            12
        );
        project.remove_transition(left).unwrap();
        assert!(
            project
                .timeline()
                .track(track)
                .unwrap()
                .transition_at(left)
                .is_none()
        );
        assert!(matches!(
            project.remove_transition(left),
            Err(ModelError::InvalidParam(_))
        ));
    }

    #[test]
    fn prune_drops_transition_when_junction_breaks() {
        let (mut project, left, _right, track) = project_with_abutting_pair();
        project.add_transition(left, "slide").unwrap();
        assert!(project.has_transitions());

        // Move the left clip so the pair no longer abuts.
        project.move_clip(left, track, rt(100)).unwrap();
        assert!(project.prune_dead_transitions());
        assert!(!project.has_transitions());
        // Idempotent: a second prune finds nothing to do.
        assert!(!project.prune_dead_transitions());
    }

    #[test]
    fn transitions_snapshot_round_trips() {
        let (mut project, left, _right, _track) = project_with_abutting_pair();
        project.add_transition(left, "dip_to_black").unwrap();
        let snapshot = project.transitions_snapshot();

        project.remove_transition(left).unwrap();
        assert!(!project.has_transitions());

        project.restore_transitions(snapshot);
        assert!(project.has_transitions());
    }

    // --- Project::new -----------------------------------------------------

    #[test]
    fn new_creates_empty_project_at_frame_rate() {
        let project = Project::new("my edit", R24);
        assert_eq!(project.schema, ProjectSchema::current());
        assert_eq!(project.metadata, ProjectMetadata::default());
        assert_eq!(project.name, "my edit");
        assert_eq!(project.timeline().frame_rate, R24);
        assert_eq!(project.media_count(), 0);
        assert_eq!(project.timeline().track_count(), 0);
        assert_eq!(project.timeline().clip_count(), 0);
        assert!(project.id.raw() >= 1);
    }

    #[test]
    fn new_accepts_string_name() {
        let owned = Project::new(String::from("owned"), R24);
        assert_eq!(owned.name, "owned");
    }

    // --- media pool -------------------------------------------------------

    #[test]
    fn add_media_returns_id_and_lookup_works() {
        let mut project = Project::new("test", R24);
        let media = sample_media(R24, 500);
        let id = project.add_media(media.clone());

        assert_eq!(project.media_count(), 1);
        assert_eq!(project.media(id).unwrap().path(), media.path());
    }

    #[test]
    fn media_iter_visits_all_sources() {
        let mut project = Project::new("test", R24);
        project.add_media(sample_media(R24, 10));
        project.add_media(sample_media(R24, 20));
        let durations: Vec<i64> = project.media_iter().map(|m| m.duration.value).collect();
        assert_eq!(durations.len(), 2);
        assert!(durations.contains(&10));
        assert!(durations.contains(&20));
    }

    #[test]
    fn find_media_by_path_matches_same_file() {
        let mut project = Project::new("test", R24);
        let id = project.add_media(sample_media(R24, 10));
        assert_eq!(
            project.find_media_by_path(Path::new("/tmp/sample.mp4")),
            Some(id)
        );
        assert_eq!(
            project.find_media_by_path(Path::new("/tmp/other.mp4")),
            None
        );
    }

    #[test]
    fn remove_media_unknown_errors() {
        let mut project = Project::new("test", R24);
        let missing = MediaId::from_raw(99);
        assert_eq!(
            project.remove_media(missing),
            Err(ModelError::UnknownMedia(missing))
        );
    }

    #[test]
    fn is_media_referenced_reflects_clip_usage() {
        let (mut project, media_id, track) = project_with_media(100);
        assert!(!project.is_media_referenced(media_id));

        let clip = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();
        assert!(project.is_media_referenced(media_id));

        project.remove_clip(clip);
        assert!(!project.is_media_referenced(media_id));
    }

    #[test]
    fn remove_media_succeeds_when_unreferenced() {
        let mut project = Project::new("test", R24);
        let id = project.add_media(sample_media(R24, 10));
        let removed = project.remove_media(id).unwrap();
        assert_eq!(removed.duration.value, 10);
        assert_eq!(project.media_count(), 0);
    }

    // --- add_clip ---------------------------------------------------------

    #[test]
    fn add_clip_places_media_with_rate_conform() {
        let mut project = Project::new("test", R24);
        let media_id = project.add_media(sample_media(R30, 1000));
        let track = project.add_track(TrackKind::Video, "V1");

        let clip = project
            .add_clip(track, media_id, tr_at(0, 120, R30), rt(10))
            .unwrap();

        let placed = project.clip(clip).unwrap();
        assert_eq!(placed.start().value, 10);
        assert_eq!(placed.timeline.duration.value, 96);
        assert_eq!(placed.source_range(), Some(tr_at(0, 120, R30)));
    }

    #[test]
    fn add_clip_rejects_unknown_track_and_media() {
        let (mut project, media_id, _) = project_with_media(100);
        let missing_track = TrackId::from_raw(999);
        let missing_media = MediaId::from_raw(999);

        assert_eq!(
            project.add_clip(missing_track, media_id, tr(0, 10), rt(0)),
            Err(ModelError::UnknownTrack(missing_track))
        );
        let track = project.add_track(TrackKind::Video, "V2");
        assert_eq!(
            project.add_clip(track, missing_media, tr(0, 10), rt(0)),
            Err(ModelError::UnknownMedia(missing_media))
        );
    }

    #[test]
    fn add_clip_rejects_empty_source_and_out_of_bounds() {
        let (mut project, media_id, track) = project_with_media(100);

        assert_eq!(
            project.add_clip(track, media_id, tr(0, 0), rt(0)),
            Err(ModelError::InvalidRange)
        );
        assert_eq!(
            project.add_clip(track, media_id, tr(-1, 10), rt(0)),
            Err(ModelError::SourceOutOfBounds)
        );
        assert_eq!(
            project.add_clip(track, media_id, tr(95, 10), rt(0)),
            Err(ModelError::SourceOutOfBounds)
        );
    }

    #[test]
    fn add_clip_rejects_rate_mismatches() {
        let (mut project, media_id, track) = project_with_media(100);

        assert_eq!(
            project.add_clip(track, media_id, tr_at(0, 10, R30), rt(0)),
            Err(ModelError::RateMismatch {
                expected: R24,
                got: R30,
            })
        );
        let bad_start = RationalTime::new(0, R30);
        assert_eq!(
            project.add_clip(track, media_id, tr(0, 10), bad_start),
            Err(ModelError::RateMismatch {
                expected: R24,
                got: R30,
            })
        );
    }

    #[test]
    fn add_clip_rejects_overlap() {
        let (mut project, media_id, track) = project_with_media(1000);
        project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();
        assert_eq!(
            project.add_clip(track, media_id, tr(0, 50), rt(50)),
            Err(ModelError::Overlap(track))
        );
    }

    // --- add_generated ----------------------------------------------------

    #[test]
    fn add_generated_without_media() {
        let mut project = Project::new("test", R24);
        let track = project.add_track(TrackKind::Text, "Titles");
        let clip = project
            .add_generated(track, Generator::text("Hi"), tr(0, 24))
            .unwrap();

        assert!(project.clip(clip).unwrap().is_generated());
        assert_eq!(project.media_count(), 0);
    }

    #[test]
    fn add_generated_rejects_empty_and_wrong_rate() {
        let mut project = Project::new("test", R24);
        let track = project.add_track(TrackKind::Video, "V1");

        assert_eq!(
            project.add_generated(track, Generator::Adjustment, tr(0, 0)),
            Err(ModelError::InvalidRange)
        );
        assert_eq!(
            project.add_generated(
                track,
                Generator::shape(Shape::Rectangle, [255, 255, 255, 255]),
                tr_at(0, 10, R30),
            ),
            Err(ModelError::RateMismatch {
                expected: R24,
                got: R30,
            })
        );
    }

    // --- duplicate_clip ---------------------------------------------------

    #[test]
    fn duplicate_clip_preserves_full_property_snapshot() {
        let (mut project, media_id, source_track) = project_with_media(1_000);
        let destination_track = project.add_track(TrackKind::Video, "V2");
        let source_id = project
            .add_clip(source_track, media_id, tr(17, 80), rt(10))
            .unwrap();
        let source_link = crate::LinkId::next();

        {
            let clip = project.timeline_mut().clip_mut(source_id).unwrap();
            clip.link = Some(source_link);

            clip.transform.position.set_keyframe(
                0,
                [0.1, -0.2],
                Easing::Bezier {
                    points: [0.2, 0.8, 0.7, 0.1],
                },
            );
            clip.transform
                .position
                .set_keyframe(60, [0.7, 0.4], Easing::EaseOut);
            clip.transform.anchor_point = Param::Constant([0.25, 0.75]);
            clip.transform.scale.set_keyframe(0, 0.8, Easing::EaseIn);
            clip.transform
                .scale
                .set_keyframe(60, 1.6, Easing::EaseInOut);
            clip.transform.rotation = Param::Constant(27.5);
            clip.transform.opacity = Param::Constant(0.65);

            clip.speed = Rational::new(3, 2);
            clip.reversed = true;
            clip.speed_curve.set_keyframe(0, 0.75, Easing::EaseIn);
            clip.speed_curve
                .set_keyframe(crate::SPEED_CURVE_SCALE, 1.5, Easing::EaseOut);
            clip.preserve_pitch = false;

            clip.volume.set_keyframe(0, 0.2, Easing::EaseIn);
            clip.volume.set_keyframe(60, 1.4, Easing::EaseOut);
            clip.fade_in = 9;
            clip.fade_out = 11;
            clip.denoise = true;

            clip.crop = CropRect {
                x: 0.1,
                y: 0.2,
                w: 0.7,
                h: 0.6,
            };
            clip.flip_h = true;
            clip.flip_v = true;

            let mut effect = EffectInstance::new("gaussian_blur");
            effect
                .set_param_keyframe(0, 0, 2.0, Easing::EaseIn)
                .unwrap();
            effect
                .set_param_keyframe(0, 60, 12.0, Easing::EaseOut)
                .unwrap();
            clip.effects = vec![effect];
            clip.beats = vec![19, 31, 47];

            clip.mask = Some(Mask {
                kind: MaskKind::Heart,
                feather: 0.35,
                invert: true,
            });
            clip.chroma_key = Some(ChromaKey {
                rgb: [3, 240, 17],
                strength: 0.72,
                shadow: 0.18,
            });
            clip.stabilize = Some(StabilizeLevel::MaxSmooth);
            clip.filter = Some(Filter {
                id: "vivid".into(),
                intensity: 0.63,
            });
            clip.lut = Some(crate::look::Lut {
                path: "/tmp/cinematic.cube".into(),
                intensity: 0.71,
            });
            clip.adjust = ColorAdjustments {
                brightness: 0.1,
                contrast: -0.2,
                saturation: 0.3,
                exposure: -0.4,
                temperature: 0.5,
            };
            clip.animation_in = Some(AnimationRef::new("zoom_in"));
            clip.animation_out = Some(AnimationRef::new("drop"));
            clip.animation_combo = Some(AnimationRef::new("pulse"));
            clip.audio_role = Some(AudioRole::Voiceover);
            clip.replaceable = Some(
                Replaceable::new(7)
                    .with_accepts(SlotMedia::VideoOnly)
                    .with_label("Hero"),
            );
            clip.text_editable = true;
        }

        let source_before = project.clip(source_id).unwrap().clone();
        let destination_start = rt(240);
        let duplicate_id = project
            .duplicate_clip(source_id, destination_track, destination_start)
            .unwrap();
        let duplicate = project.clip(duplicate_id).unwrap();

        let mut expected = source_before.clone();
        expected.id = duplicate_id;
        expected.timeline.start = destination_start;
        expected.link = None;
        assert_eq!(
            duplicate, &expected,
            "normalizing only id/start/link makes the full snapshots equal"
        );
        assert_eq!(
            project.clip(source_id).unwrap(),
            &source_before,
            "duplication never mutates the source"
        );
        assert_ne!(duplicate_id, source_id);
        assert_eq!(
            project.timeline().track_of(duplicate_id),
            Some(destination_track)
        );
    }

    #[test]
    fn duplicate_clip_clones_generated_content_and_template_flag() {
        let mut project = Project::new("test", R24);
        let source_track = project.add_track(TrackKind::Text, "Titles");
        let destination_track = project.add_track(TrackKind::Text, "Titles 2");
        let source_id = project
            .add_generated(source_track, Generator::text("Animated title"), tr(5, 40))
            .unwrap();
        project.set_text_editable(source_id, true).unwrap();
        project
            .set_param_keyframe(
                source_id,
                ClipParam::Scale,
                rt(5),
                ParamValue::Scalar(0.75),
                Easing::EaseInOut,
            )
            .unwrap();

        let source_before = project.clip(source_id).unwrap().clone();
        let start = rt(100);
        let duplicate_id = project
            .duplicate_clip(source_id, destination_track, start)
            .unwrap();

        let mut expected = source_before.clone();
        expected.id = duplicate_id;
        expected.timeline.start = start;
        expected.link = None;
        assert_eq!(project.clip(duplicate_id).unwrap(), &expected);
        assert!(project.clip(duplicate_id).unwrap().text_editable);
        assert_eq!(project.clip(source_id).unwrap(), &source_before);
    }

    #[test]
    fn duplicate_clip_unlinks_copy_without_changing_source_group() {
        let (mut project, media_id, source_track) = project_with_media(300);
        let destination_track = project.add_track(TrackKind::Video, "V2");
        let first = project
            .add_clip(source_track, media_id, tr(0, 20), rt(0))
            .unwrap();
        let second = project
            .add_clip(source_track, media_id, tr(20, 20), rt(20))
            .unwrap();
        let group = crate::LinkId::next();
        for id in [first, second] {
            project.timeline_mut().clip_mut(id).unwrap().link = Some(group);
        }

        let duplicate = project
            .duplicate_clip(first, destination_track, rt(100))
            .unwrap();

        assert_eq!(project.clip(first).unwrap().link, Some(group));
        assert_eq!(project.clip(second).unwrap().link, Some(group));
        assert_eq!(project.clip(duplicate).unwrap().link, None);
    }

    #[test]
    fn duplicate_clip_rejections_are_atomic() {
        let (mut project, media_id, source_track) = project_with_media(500);
        let destination_track = project.add_track(TrackKind::Video, "V2");
        let incompatible_track = project.add_track(TrackKind::Text, "Titles");
        let source = project
            .add_clip(source_track, media_id, tr(0, 40), rt(0))
            .unwrap();
        let blocker = project
            .add_clip(destination_track, media_id, tr(40, 40), rt(100))
            .unwrap();
        let source_before = project.clip(source).unwrap().clone();
        let blocker_before = project.clip(blocker).unwrap().clone();
        let clip_count_before = project.timeline().clip_count();
        let project_before = serde_json::to_value(&project).unwrap();

        macro_rules! assert_atomic_rejection {
            ($result:expr, $expected:expr) => {{
                assert_eq!($result, Err($expected));
                assert_eq!(project.timeline().clip_count(), clip_count_before);
                assert_eq!(project.clip(source).unwrap(), &source_before);
                assert_eq!(project.clip(blocker).unwrap(), &blocker_before);
                assert_eq!(
                    serde_json::to_value(&project).unwrap(),
                    project_before,
                    "the complete serialized project remains unchanged"
                );
            }};
        }

        let equivalent_but_noncanonical_rate = Rational::new(48, 2);
        assert_atomic_rejection!(
            project.duplicate_clip(
                source,
                destination_track,
                RationalTime::new(200, equivalent_but_noncanonical_rate)
            ),
            ModelError::RateMismatch {
                expected: R24,
                got: equivalent_but_noncanonical_rate,
            }
        );
        assert_atomic_rejection!(
            project.duplicate_clip(source, destination_track, rt(-1)),
            ModelError::InvalidRange
        );

        let missing_clip = ClipId::from_raw(u64::MAX - 1);
        assert_atomic_rejection!(
            project.duplicate_clip(missing_clip, destination_track, rt(200)),
            ModelError::UnknownClip(missing_clip)
        );

        let missing_track = TrackId::from_raw(u64::MAX - 1);
        assert_atomic_rejection!(
            project.duplicate_clip(source, missing_track, rt(200)),
            ModelError::UnknownTrack(missing_track)
        );
        assert_atomic_rejection!(
            project.duplicate_clip(source, incompatible_track, rt(200)),
            ModelError::IncompatibleTrackKind {
                track: incompatible_track,
                kind: TrackKind::Text,
            }
        );
        assert_atomic_rejection!(
            project.duplicate_clip(source, destination_track, rt(110)),
            ModelError::Overlap(destination_track)
        );
    }

    // --- remove_clip ------------------------------------------------------

    #[test]
    fn remove_clip_returns_clip_and_leaves_gap() {
        let (mut project, media_id, track) = project_with_media(200);
        let a = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();
        let b = project
            .add_clip(track, media_id, tr(50, 50), rt(100))
            .unwrap();

        let removed = project.remove_clip(a).unwrap();
        assert_eq!(removed.id, a);
        assert!(project.clip(a).is_none());
        assert_eq!(project.clip(b).unwrap().start().value, 100);
        assert_eq!(project.timeline().clip_count(), 1);
    }

    #[test]
    fn remove_clip_unknown_returns_none() {
        let (mut project, _, _) = project_with_media(10);
        assert!(project.remove_clip(ClipId::from_raw(404)).is_none());
    }

    // --- split_clip -------------------------------------------------------

    #[test]
    fn split_clip_divides_media_and_generated() {
        let (mut project, media_id, track) = project_with_media(500);
        let media_clip = project
            .add_clip(track, media_id, tr(100, 100), rt(0))
            .unwrap();
        let right = project.split_clip(media_clip, rt(40)).unwrap();
        assert_eq!(project.clip(media_clip).unwrap().timeline, tr(0, 40));
        assert_eq!(project.clip(right).unwrap().timeline, tr(40, 60));

        let fx = project.add_track(TrackKind::Adjustment, "FX");
        let generated = project
            .add_generated(fx, Generator::Adjustment, tr(200, 100))
            .unwrap();
        let generated_right = project.split_clip(generated, rt(250)).unwrap();
        assert_eq!(project.clip(generated).unwrap().timeline, tr(200, 50));
        assert_eq!(project.clip(generated_right).unwrap().timeline, tr(250, 50));
    }

    #[test]
    fn split_after_reload_allocates_a_fresh_tail_id() {
        // Regression for the cross-session id-collision bug: a clip persisted
        // in one session must not have its id re-handed to a clip created in
        // the next. Round-tripping through JSON (as loading a project does)
        // reseeds the id allocator, so the split's tail id is distinct from
        // every id already in the project.
        let (mut project, media_id, track) = project_with_media(500);
        let persisted = project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();

        let json = serde_json::to_string(&project).unwrap();
        let mut reloaded: Project = serde_json::from_str(&json).unwrap();

        let tail = reloaded.split_clip(persisted, rt(40)).unwrap();
        assert_ne!(tail, persisted, "tail must not reuse the persisted clip id");
        assert_eq!(reloaded.timeline().clip_count(), 2);
        // Both halves are addressable and distinct — the overwrite that used
        // to drop the left half can no longer happen.
        assert_eq!(reloaded.clip(persisted).unwrap().timeline, tr(0, 40));
        assert_eq!(reloaded.clip(tail).unwrap().timeline, tr(40, 60));
    }

    #[test]
    fn split_clip_rejects_at_boundaries() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(10))
            .unwrap();
        assert_eq!(
            project.split_clip(clip, rt(10)),
            Err(ModelError::InvalidRange)
        );
        assert_eq!(
            project.split_clip(clip, rt(110)),
            Err(ModelError::InvalidRange)
        );
    }

    #[test]
    fn split_clip_rejects_unknown_clip() {
        let (mut project, _, _) = project_with_media(10);
        let unknown = ClipId::from_raw(999);
        assert_eq!(
            project.split_clip(unknown, rt(5)),
            Err(ModelError::UnknownClip(unknown))
        );
    }

    #[test]
    fn split_clip_rejects_wrong_timeline_rate() {
        let (mut project, media_id, track) = project_with_media(100);
        let clip = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();
        assert_eq!(
            project.split_clip(clip, RationalTime::new(25, R30)),
            Err(ModelError::RateMismatch {
                expected: R24,
                got: R30,
            })
        );
    }

    // --- trim_clip --------------------------------------------------------

    #[test]
    fn trim_clip_tail_shortens_media_source() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();

        project.trim_clip(clip, tr(0, 60)).unwrap();
        let trimmed = project.clip(clip).unwrap();
        assert_eq!(trimmed.timeline, tr(0, 60));
        assert_eq!(trimmed.source_range(), Some(tr(0, 60)));
    }

    #[test]
    fn trim_extends_image_clips_past_the_default_window() {
        let mut project = Project::new("stills", R24);
        let media_id = project.add_media(MediaSource::image("/tmp/a.png", 800, 600));
        let track = project.add_track(TrackKind::Video, "V1");
        let full = project.media(media_id).unwrap().full_range();
        let clip = project.add_clip(track, media_id, full, rt(0)).unwrap();
        // The 5s default placement at 24 fps.
        assert_eq!(project.clip(clip).unwrap().timeline, tr(0, 120));

        // A still stretches to any length: 10s here, double its pool entry.
        project.trim_clip(clip, tr(0, 240)).unwrap();
        let stretched = project.clip(clip).unwrap();
        assert_eq!(stretched.timeline, tr(0, 240));
        let source = stretched.source_range().unwrap();
        assert_eq!(source.start.value, 0);
        assert!(source.duration.value > crate::media::STILL_DEFAULT_DURATION_TICKS);
    }

    #[test]
    fn add_clip_allows_oversized_image_windows() {
        let mut project = Project::new("stills", R24);
        let media_id = project.add_media(MediaSource::image("/tmp/a.png", 800, 600));
        let track = project.add_track(TrackKind::Video, "V1");
        // Place a 20s clip from the 5s pool entry directly (agent add_clip).
        let window = TimeRange::at_rate(0, 20_000, crate::media::STILL_TICK_RATE);
        let clip = project.add_clip(track, media_id, window, rt(0)).unwrap();
        assert_eq!(project.clip(clip).unwrap().timeline, tr(0, 480));

        // Video media keeps its real material bound.
        let video = project.add_media(sample_media(R24, 100));
        assert!(matches!(
            project.add_clip(track, video, tr(0, 101), rt(600)),
            Err(ModelError::SourceOutOfBounds)
        ));
    }

    #[test]
    fn trim_clip_generated_only_moves_timeline() {
        let mut project = Project::new("test", R24);
        let track = project.add_track(TrackKind::Adjustment, "FX");
        let clip = project
            .add_generated(track, Generator::Adjustment, tr(0, 100))
            .unwrap();

        project.trim_clip(clip, tr(20, 50)).unwrap();
        let trimmed = project.clip(clip).unwrap();
        assert_eq!(trimmed.timeline, tr(20, 50));
        assert_eq!(trimmed.source_range(), None);
    }

    // --- set_clip_speed (M1) -----------------------------------------------

    #[test]
    fn set_clip_speed_rescales_timeline_duration() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();

        // 2× halves the footprint; the source window is untouched.
        project
            .set_clip_speed(clip, Rational::new(2, 1), false)
            .unwrap();
        let c = project.clip(clip).unwrap();
        assert_eq!(c.timeline, tr(0, 50));
        assert_eq!(c.source_range(), Some(tr(0, 100)));
        assert_eq!(c.speed, Rational::new(2, 1));

        // Back to 1× restores the original footprint.
        project
            .set_clip_speed(clip, Rational::new(1, 1), false)
            .unwrap();
        assert_eq!(project.clip(clip).unwrap().timeline, tr(0, 100));

        // Slow motion stretches it.
        project
            .set_clip_speed(clip, Rational::new(1, 2), false)
            .unwrap();
        assert_eq!(project.clip(clip).unwrap().timeline, tr(0, 200));
    }

    #[test]
    fn set_clip_pitch_toggles_the_flag_without_moving_the_clip() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();
        project
            .set_clip_speed(clip, Rational::new(2, 1), false)
            .unwrap();
        assert!(
            project.clip(clip).unwrap().preserve_pitch,
            "locked by default"
        );

        // Unlocking pitch keeps the retimed footprint (pure audio property).
        project.set_clip_pitch(clip, false).unwrap();
        let c = project.clip(clip).unwrap();
        assert!(!c.preserve_pitch);
        assert_eq!(c.timeline, tr(0, 50));

        project.set_clip_pitch(clip, true).unwrap();
        assert!(project.clip(clip).unwrap().preserve_pitch);
    }

    #[test]
    fn set_clip_pitch_rejects_generated_clips() {
        let mut project = Project::new("p", R24);
        let track = project.add_track(TrackKind::Sticker, "T1");
        let clip = project
            .add_generated(
                track,
                Generator::SolidColor {
                    rgba: [0, 0, 0, 255],
                },
                tr(0, 10),
            )
            .unwrap();
        assert!(
            project.set_clip_pitch(clip, false).is_err(),
            "no audio to retime"
        );
    }

    #[test]
    fn set_clip_denoise_toggles_and_returns_the_previous_flag() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();
        assert!(!project.clip(clip).unwrap().denoise, "off by default");

        // Turning it on returns the old value (false); turning it off again
        // returns true — the engine uses this for the undo inverse.
        assert!(!project.set_clip_denoise(clip, true).unwrap());
        assert!(project.clip(clip).unwrap().denoise);
        assert!(project.set_clip_denoise(clip, false).unwrap());
        assert!(!project.clip(clip).unwrap().denoise);
    }

    #[test]
    fn set_clip_denoise_rejects_generated_clips() {
        let mut project = Project::new("p", R24);
        let track = project.add_track(TrackKind::Sticker, "T1");
        let clip = project
            .add_generated(
                track,
                Generator::SolidColor {
                    rgba: [0, 0, 0, 255],
                },
                tr(0, 10),
            )
            .unwrap();
        assert!(
            project.set_clip_denoise(clip, true).is_err(),
            "no audio to clean"
        );
    }

    #[test]
    fn set_clip_speed_curve_rederives_duration_from_average() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();

        // Average 2× ramp halves the footprint (source ÷ avg), like constant 2×.
        let ramp = crate::clip::speed_preset("montage").unwrap();
        let avg = {
            let mut probe = project.clip(clip).unwrap().clone();
            probe.speed_curve = ramp.clone();
            probe.speed_curve_average()
        };
        project.set_clip_speed_curve(clip, Some(ramp)).unwrap();
        let expected = (100.0 / avg).round() as i64;
        assert_eq!(project.clip(clip).unwrap().timeline, tr(0, expected));
        assert!(project.clip(clip).unwrap().has_speed_curve());

        // Clearing the ramp restores the original footprint exactly.
        project.set_clip_speed_curve(clip, None).unwrap();
        assert_eq!(project.clip(clip).unwrap().timeline, tr(0, 100));
        assert!(!project.clip(clip).unwrap().has_speed_curve());
    }

    #[test]
    fn set_clip_speed_curve_rejects_bad_targets() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();
        // Out-of-range ramp value.
        let bad = Param::Keyframed {
            keyframes: vec![
                crate::param::Keyframe {
                    tick: 0,
                    value: 0.0,
                    easing: Easing::Linear,
                },
                crate::param::Keyframe {
                    tick: 1000,
                    value: 1.0,
                    easing: Easing::Linear,
                },
            ],
        };
        assert!(project.set_clip_speed_curve(clip, Some(bad)).is_err());
        // Generated clips cannot be retimed.
        let sticker = project.add_track(TrackKind::Sticker, "S");
        let generated = project
            .add_generated(
                sticker,
                Generator::SolidColor {
                    rgba: [0, 0, 0, 255],
                },
                tr(200, 10),
            )
            .unwrap();
        assert!(matches!(
            project
                .set_clip_speed_curve(generated, Some(crate::clip::speed_preset("hero").unwrap())),
            Err(ModelError::InvalidParam(_))
        ));
    }

    #[test]
    fn set_clip_speed_reverse_keeps_duration() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();

        project
            .set_clip_speed(clip, Rational::new(1, 1), true)
            .unwrap();
        let c = project.clip(clip).unwrap();
        assert_eq!(c.timeline, tr(0, 100));
        assert!(c.reversed && c.is_retimed());
    }

    #[test]
    fn set_clip_speed_rejects_invalid_targets() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();

        assert!(matches!(
            project.set_clip_speed(clip, Rational::new(0, 1), false),
            Err(ModelError::InvalidParam(_))
        ));
        assert!(matches!(
            project.set_clip_speed(clip, Rational::new(-2, 1), false),
            Err(ModelError::InvalidParam(_))
        ));

        let fx = project.add_track(TrackKind::Adjustment, "FX");
        let generated = project
            .add_generated(fx, Generator::Adjustment, tr(0, 100))
            .unwrap();
        assert!(matches!(
            project.set_clip_speed(generated, Rational::new(2, 1), false),
            Err(ModelError::InvalidParam(_))
        ));

        assert_eq!(
            project.set_clip_speed(ClipId::from_raw(404), Rational::new(2, 1), false),
            Err(ModelError::UnknownClip(ClipId::from_raw(404)))
        );
    }

    #[test]
    fn set_clip_speed_rejects_overlap_with_neighbor() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();
        // Neighbor right behind the clip: slowing to ½× (200 ticks) collides.
        project
            .add_clip(track, media_id, tr(0, 50), rt(100))
            .unwrap();

        assert_eq!(
            project.set_clip_speed(clip, Rational::new(1, 2), false),
            Err(ModelError::Overlap(track))
        );
        // The clip is untouched after the rejection.
        let c = project.clip(clip).unwrap();
        assert_eq!(c.timeline, tr(0, 100));
        assert_eq!(c.speed, Rational::new(1, 1));
    }

    // --- set_clip_beats (M8 Phase 6) -----------------------------------------

    #[test]
    fn set_clip_beats_sorts_dedups_clamps_and_returns_previous() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();

        // Out-of-window (>=100), duplicate, and unsorted input is normalized.
        let prev = project
            .set_clip_beats(clip, vec![60, 10, 10, 200, 30])
            .unwrap();
        assert!(prev.is_empty(), "no beats before the first detect");
        assert_eq!(project.clip(clip).unwrap().beats, vec![10, 30, 60]);

        // Replacing returns the old list (the inverse snapshot).
        let prev = project.set_clip_beats(clip, vec![5]).unwrap();
        assert_eq!(prev, vec![10, 30, 60]);
        assert_eq!(project.clip(clip).unwrap().beats, vec![5]);
    }

    #[test]
    fn set_clip_beats_rejects_generated_clips() {
        let mut project = Project::new("test", R24);
        let track = project.add_track(TrackKind::Text, "T");
        let clip = project
            .add_generated(track, Generator::text("hi"), tr(0, 48))
            .unwrap();
        assert!(matches!(
            project.set_clip_beats(clip, vec![1, 2]),
            Err(ModelError::InvalidParam(_))
        ));
    }

    #[test]
    fn split_keeps_beats_on_both_halves() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();
        project.set_clip_beats(clip, vec![10, 60, 90]).unwrap();

        let right = project.split_clip(clip, rt(50)).unwrap();
        // Both halves keep the full source-tick list; each shows only the
        // beats inside its own window via `beat_timeline_ticks`.
        assert_eq!(project.clip(clip).unwrap().beats, vec![10, 60, 90]);
        assert_eq!(project.clip(right).unwrap().beats, vec![10, 60, 90]);
        // Left window [0,50): only the beat at 10 maps in.
        assert_eq!(project.clip(clip).unwrap().beat_timeline_ticks(), vec![10]);
        // Right window [50,100) at timeline [50,100): beats 60, 90 map in.
        assert_eq!(
            project.clip(right).unwrap().beat_timeline_ticks(),
            vec![60, 90]
        );
    }

    // --- set_clip_audio (M1) -------------------------------------------------

    #[test]
    fn set_clip_audio_sets_volume_and_fades() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();

        project
            .set_clip_audio(clip, Some(0.5), rt(10), rt(20))
            .unwrap();
        let c = project.clip(clip).unwrap();
        assert_eq!(c.volume.constant(), Some(0.5));
        assert_eq!((c.fade_in, c.fade_out), (10, 20));
        assert!(c.has_custom_audio());

        // A `None` volume keeps the gain and only moves the fades.
        project.set_clip_audio(clip, None, rt(5), rt(0)).unwrap();
        let c = project.clip(clip).unwrap();
        assert_eq!(c.volume.constant(), Some(0.5));
        assert_eq!((c.fade_in, c.fade_out), (5, 0));

        // Back to defaults clears the custom-audio state.
        project
            .set_clip_audio(clip, Some(1.0), rt(0), rt(0))
            .unwrap();
        assert!(!project.clip(clip).unwrap().has_custom_audio());
    }

    #[test]
    fn set_clip_audio_none_volume_preserves_envelope() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();
        // A keyframed gain envelope (dips to 0.2 mid-clip).
        project
            .set_param_keyframe(
                clip,
                ClipParam::Volume,
                rt(0),
                ParamValue::Scalar(1.0),
                Easing::Linear,
            )
            .unwrap();
        project
            .set_param_keyframe(
                clip,
                ClipParam::Volume,
                rt(50),
                ParamValue::Scalar(0.2),
                Easing::Linear,
            )
            .unwrap();
        assert!(project.clip(clip).unwrap().has_volume_envelope());

        // Setting only fades (None volume) must not flatten the envelope.
        project.set_clip_audio(clip, None, rt(10), rt(20)).unwrap();
        let c = project.clip(clip).unwrap();
        assert!(c.has_volume_envelope());
        assert_eq!(c.volume.sample(50), 0.2);
        assert_eq!((c.fade_in, c.fade_out), (10, 20));
    }

    #[test]
    fn set_clip_audio_rejects_invalid_inputs() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();

        for volume in [-0.1, 11.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                project.set_clip_audio(clip, Some(volume), rt(0), rt(0)),
                Err(ModelError::InvalidParam(_))
            ));
        }
        // Negative or longer-than-the-clip fades.
        assert!(matches!(
            project.set_clip_audio(clip, Some(1.0), rt(-1), rt(0)),
            Err(ModelError::InvalidParam(_))
        ));
        assert!(matches!(
            project.set_clip_audio(clip, Some(1.0), rt(0), rt(101)),
            Err(ModelError::InvalidParam(_))
        ));

        let fx = project.add_track(TrackKind::Adjustment, "FX");
        let generated = project
            .add_generated(fx, Generator::Adjustment, tr(0, 100))
            .unwrap();
        assert!(matches!(
            project.set_clip_audio(generated, Some(0.5), rt(0), rt(0)),
            Err(ModelError::InvalidParam(_))
        ));

        assert_eq!(
            project.set_clip_audio(ClipId::from_raw(404), Some(0.5), rt(0), rt(0)),
            Err(ModelError::UnknownClip(ClipId::from_raw(404)))
        );
    }

    #[test]
    fn split_keeps_volume_and_partitions_fades() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();
        project
            .set_clip_audio(clip, Some(0.5), rt(10), rt(20))
            .unwrap();

        let right = project.split_clip(clip, rt(60)).unwrap();
        let left = project.clip(clip).unwrap();
        let right = project.clip(right).unwrap();
        // Volume rides both halves; the fade-in stays with the head, the
        // fade-out moves to the tail.
        assert_eq!(
            (left.volume.constant(), right.volume.constant()),
            (Some(0.5), Some(0.5))
        );
        assert_eq!((left.fade_in, left.fade_out), (10, 0));
        assert_eq!((right.fade_in, right.fade_out), (0, 20));
    }

    // --- set_clip_crop (M1) ---------------------------------------------------

    #[test]
    fn set_clip_crop_sets_framing() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();

        let crop = CropRect {
            x: 0.25,
            y: 0.0,
            w: 0.5,
            h: 1.0,
        };
        project.set_clip_crop(clip, crop, true, false).unwrap();
        let c = project.clip(clip).unwrap();
        assert_eq!(c.crop, crop);
        assert!(c.flip_h && !c.flip_v);
        assert!(c.has_custom_crop());

        // Back to the full frame clears the custom-framing state.
        project
            .set_clip_crop(clip, CropRect::FULL, false, false)
            .unwrap();
        assert!(!project.clip(clip).unwrap().has_custom_crop());
    }

    #[test]
    fn set_clip_crop_rejects_invalid_targets() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();

        // Invalid rects bounce.
        assert!(matches!(
            project.set_clip_crop(
                clip,
                CropRect {
                    x: 0.8,
                    y: 0.0,
                    w: 0.5,
                    h: 1.0
                },
                false,
                false
            ),
            Err(ModelError::InvalidParam(_))
        ));

        // Audio lanes have no frame to crop.
        let audio = project.add_track(TrackKind::Audio, "A1");
        let audio_clip = project.add_clip(audio, media_id, tr(0, 50), rt(0)).unwrap();
        assert!(matches!(
            project.set_clip_crop(audio_clip, CropRect::FULL, true, false),
            Err(ModelError::IncompatibleTrackKind { .. })
        ));

        assert_eq!(
            project.set_clip_crop(ClipId::from_raw(404), CropRect::FULL, false, false),
            Err(ModelError::UnknownClip(ClipId::from_raw(404)))
        );
    }

    #[test]
    fn split_keeps_crop_and_flips_on_both_halves() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();
        let crop = CropRect {
            x: 0.1,
            y: 0.1,
            w: 0.8,
            h: 0.8,
        };
        project.set_clip_crop(clip, crop, true, true).unwrap();

        let right = project.split_clip(clip, rt(60)).unwrap();
        for id in [clip, right] {
            let c = project.clip(id).unwrap();
            assert_eq!(c.crop, crop);
            assert!(c.flip_h && c.flip_v);
        }
    }

    #[test]
    fn trim_scales_source_consumption_by_speed() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(100, 100), rt(0))
            .unwrap();
        project
            .set_clip_speed(clip, Rational::new(2, 1), false)
            .unwrap();
        // Now timeline [0, 50) over source [100, 200).

        // Head trim by 10 timeline ticks eats 20 source ticks.
        project.trim_clip(clip, tr(10, 40)).unwrap();
        let c = project.clip(clip).unwrap();
        assert_eq!(c.source_range(), Some(tr(120, 80)));

        // Tail trim to 20 timeline ticks keeps 40 source ticks.
        project.trim_clip(clip, tr(10, 20)).unwrap();
        assert_eq!(
            project.clip(clip).unwrap().source_range(),
            Some(tr(120, 40))
        );
    }

    #[test]
    fn trim_reversed_clip_mirrors_source_window() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(100, 100), rt(0))
            .unwrap();
        project
            .set_clip_speed(clip, Rational::new(1, 1), true)
            .unwrap();

        // The timeline head shows the source END: a head trim by 10 drops
        // the top 10 source ticks, keeping the bottom.
        project.trim_clip(clip, tr(10, 90)).unwrap();
        assert_eq!(
            project.clip(clip).unwrap().source_range(),
            Some(tr(100, 90))
        );

        // A tail trim drops the source BOTTOM.
        project.trim_clip(clip, tr(10, 80)).unwrap();
        assert_eq!(
            project.clip(clip).unwrap().source_range(),
            Some(tr(110, 80))
        );
    }

    #[test]
    fn split_retimed_clip_inherits_speed_and_partitions_source() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(0))
            .unwrap();
        project
            .set_clip_speed(clip, Rational::new(2, 1), false)
            .unwrap();
        // Timeline [0, 50) over source [0, 100).

        let right = project.split_clip(clip, rt(20)).unwrap();
        let left = project.clip(clip).unwrap();
        let right = project.clip(right).unwrap();
        assert_eq!(left.timeline, tr(0, 20));
        assert_eq!(left.source_range(), Some(tr(0, 40)));
        assert_eq!(right.timeline, tr(20, 30));
        assert_eq!(right.source_range(), Some(tr(40, 60)));
        assert_eq!(right.speed, Rational::new(2, 1));
    }

    #[test]
    fn split_reversed_clip_hands_window_top_to_the_left() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(100, 100), rt(0))
            .unwrap();
        project
            .set_clip_speed(clip, Rational::new(1, 1), true)
            .unwrap();

        let right = project.split_clip(clip, rt(30)).unwrap();
        let left = project.clip(clip).unwrap();
        let right = project.clip(right).unwrap();
        // Left timeline half plays the source top backward; right half the
        // bottom.
        assert_eq!(left.source_range(), Some(tr(170, 30)));
        assert_eq!(right.source_range(), Some(tr(100, 70)));
        assert!(right.reversed);
    }

    // --- move_clip --------------------------------------------------------

    #[test]
    fn move_clip_repositions_on_same_track() {
        let (mut project, media_id, track) = project_with_media(200);
        let clip = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();

        project.move_clip(clip, track, rt(80)).unwrap();
        assert_eq!(project.clip(clip).unwrap().timeline, tr(80, 50));
        assert_eq!(project.timeline().track_of(clip), Some(track));
    }

    #[test]
    fn move_clip_rejects_negative_start_and_unknown_track() {
        let (mut project, media_id, track) = project_with_media(200);
        let clip = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();
        let missing = TrackId::from_raw(77);

        assert_eq!(
            project.move_clip(clip, track, rt(-1)),
            Err(ModelError::InvalidRange)
        );
        assert_eq!(
            project.move_clip(clip, missing, rt(0)),
            Err(ModelError::UnknownTrack(missing))
        );
    }

    // --- shift_clips ------------------------------------------------------

    fn packed_track() -> (Project, MediaId, TrackId, [ClipId; 3]) {
        let (mut project, media_id, track) = project_with_media(1000);
        let a = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();
        let b = project
            .add_clip(track, media_id, tr(50, 50), rt(50))
            .unwrap();
        let c = project
            .add_clip(track, media_id, tr(100, 50), rt(100))
            .unwrap();
        (project, media_id, track, [a, b, c])
    }

    #[test]
    fn shift_clips_right_moves_suffix_only() {
        let (mut project, _, track, [a, b, c]) = packed_track();
        let first = project.shift_clips(track, rt(50), rt(30)).unwrap();
        assert_eq!(first, Some(rt(80)));
        assert_eq!(project.clip(a).unwrap().start().value, 0);
        assert_eq!(project.clip(b).unwrap().start().value, 80);
        assert_eq!(project.clip(c).unwrap().start().value, 130);
    }

    #[test]
    fn shift_clips_left_closes_gap() {
        let (mut project, _, track, [a, b, c]) = packed_track();
        project.shift_clips(track, rt(50), rt(30)).unwrap();
        let first = project.shift_clips(track, rt(80), rt(-30)).unwrap();
        assert_eq!(first, Some(rt(50)));
        assert_eq!(project.clip(a).unwrap().start().value, 0);
        assert_eq!(project.clip(b).unwrap().start().value, 50);
        assert_eq!(project.clip(c).unwrap().start().value, 100);
    }

    #[test]
    fn shift_clips_left_rejects_collision_and_negative() {
        let (mut project, _, track, [_, b, _]) = packed_track();
        assert_eq!(
            project.shift_clips(track, rt(50), rt(-10)),
            Err(ModelError::Overlap(track))
        );
        assert_eq!(
            project.shift_clips(track, rt(0), rt(-1)),
            Err(ModelError::InvalidRange)
        );
        // Validation failures must not mutate anything.
        assert_eq!(project.clip(b).unwrap().start().value, 50);
    }

    #[test]
    fn shift_clips_past_content_is_noop() {
        let (mut project, _, track, [a, b, c]) = packed_track();
        assert_eq!(project.shift_clips(track, rt(999), rt(40)).unwrap(), None);
        for (clip, start) in [(a, 0), (b, 50), (c, 100)] {
            assert_eq!(project.clip(clip).unwrap().start().value, start);
        }
    }

    #[test]
    fn shift_clips_rejects_zero_delta_and_unknown_track() {
        let (mut project, _, track, _) = packed_track();
        assert_eq!(
            project.shift_clips(track, rt(0), rt(0)),
            Err(ModelError::InvalidRange)
        );
        let missing = TrackId::from_raw(404);
        assert_eq!(
            project.shift_clips(missing, rt(0), rt(10)),
            Err(ModelError::UnknownTrack(missing))
        );
    }

    // --- ripple_delete ----------------------------------------------------

    #[test]
    fn ripple_delete_shifts_later_clips() {
        let (mut project, media_id, track) = project_with_media(300);
        let a = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();
        let b = project
            .add_clip(track, media_id, tr(50, 50), rt(50))
            .unwrap();
        let c = project
            .add_clip(track, media_id, tr(100, 50), rt(150))
            .unwrap();

        project.ripple_delete(b).unwrap();
        assert!(project.clip(b).is_none());
        assert_eq!(project.clip(a).unwrap().start().value, 0);
        assert_eq!(project.clip(c).unwrap().start().value, 100);
    }

    // --- timeline accessors -----------------------------------------------

    #[test]
    fn timeline_mut_allows_direct_timeline_edits() {
        let mut project = Project::new("test", R24);
        let track = project.add_track(TrackKind::Audio, "A1");
        assert_eq!(project.timeline_mut().track_count(), 1);
        assert_eq!(
            project.timeline().track(track).unwrap().kind,
            TrackKind::Audio
        );
    }

    // --- Clone ------------------------------------------------------------

    #[test]
    fn project_clone_is_independent_snapshot() {
        let (mut project, media_id, track) = project_with_media(100);
        let clip = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();

        let mut cloned = project.clone();
        assert_eq!(cloned.clip(clip).unwrap().timeline, tr(0, 50));

        cloned.remove_clip(clip);
        assert!(cloned.clip(clip).is_none());
        assert!(project.clip(clip).is_some());
    }

    // --- Phase I look properties --------------------------------------------

    use crate::look::{
        AnimationRef, AnimationSlot, AudioRole, ChromaKey, ColorAdjustments, Filter, Mask,
        MaskKind, StabilizeLevel,
    };

    #[test]
    fn look_setters_persist_on_a_media_clip() {
        let (mut project, media_id, track) = project_with_media(100);
        let clip = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();

        project
            .set_clip_mask(clip, Some(Mask::new(MaskKind::Circle)))
            .unwrap();
        project
            .set_clip_chroma_key(
                clip,
                Some(ChromaKey {
                    rgb: [0, 255, 0],
                    strength: 0.7,
                    shadow: 0.2,
                }),
            )
            .unwrap();
        project
            .set_clip_stabilize(clip, Some(StabilizeLevel::Smooth))
            .unwrap();
        project
            .set_clip_filter(clip, Some(Filter::new("vivid")))
            .unwrap();
        project
            .set_clip_adjustments(
                clip,
                ColorAdjustments {
                    brightness: 0.25,
                    ..Default::default()
                },
            )
            .unwrap();

        let c = project.clip(clip).unwrap();
        assert_eq!(c.mask.unwrap().kind, MaskKind::Circle);
        assert_eq!(c.chroma_key.unwrap().strength, 0.7);
        assert_eq!(c.stabilize, Some(StabilizeLevel::Smooth));
        assert_eq!(c.filter.as_ref().unwrap().id, "vivid");
        assert_eq!(c.adjust.brightness, 0.25);

        // Clearing works and the fields vanish from saves again.
        project.set_clip_mask(clip, None).unwrap();
        assert!(project.clip(clip).unwrap().mask.is_none());
    }

    #[test]
    fn look_setters_reject_wrong_content() {
        let (mut project, media_id, _) = project_with_media(100);

        // Generated clips have no source pixels to mask/key/stabilize.
        let fx = project.add_track(TrackKind::Adjustment, "FX");
        let bar = project
            .add_generated(fx, Generator::Adjustment, tr(0, 24))
            .unwrap();
        assert!(
            project
                .set_clip_mask(bar, Some(Mask::new(MaskKind::Linear)))
                .is_err()
        );
        assert!(
            project
                .set_clip_stabilize(bar, Some(StabilizeLevel::Recommended))
                .is_err()
        );
        // …but filters and adjustments are exactly what lane bars persist.
        assert!(
            project
                .set_clip_filter(bar, Some(Filter::new("noir")))
                .is_ok()
        );
        assert!(
            project
                .set_clip_adjustments(
                    bar,
                    ColorAdjustments {
                        contrast: -0.5,
                        ..Default::default()
                    }
                )
                .is_ok()
        );

        // Audio-lane clips show no pixels.
        let audio = project.add_track(TrackKind::Audio, "A1");
        let aclip = project.add_clip(audio, media_id, tr(0, 50), rt(0)).unwrap();
        assert!(
            project
                .set_clip_filter(aclip, Some(Filter::new("warm")))
                .is_err()
        );

        // Stills have no motion to stabilize.
        let image = project.add_media(MediaSource::image("/tmp/photo.png", 800, 600));
        let video = project.add_track(TrackKind::Video, "V2");
        let still = project
            .add_clip(
                video,
                image,
                tr_at(0, 100, crate::media::STILL_TICK_RATE),
                rt(0),
            )
            .unwrap();
        assert!(
            project
                .set_clip_stabilize(still, Some(StabilizeLevel::Smooth))
                .is_err()
        );
        assert!(
            project
                .set_clip_mask(still, Some(Mask::new(MaskKind::Heart)))
                .is_ok(),
            "masks apply to stills"
        );

        // Unknown ids and out-of-range values are rejected.
        let (mut p2, m2, t2) = project_with_media(100);
        let clip = p2.add_clip(t2, m2, tr(0, 50), rt(0)).unwrap();
        assert!(p2.set_clip_filter(clip, Some(Filter::new("nope"))).is_err());
        let mut hot = Filter::new("vivid");
        hot.intensity = 1.5;
        assert!(p2.set_clip_filter(clip, Some(hot)).is_err());
    }

    #[test]
    fn animation_slots_enforce_capcut_exclusivity() {
        let (mut project, media_id, track) = project_with_media(100);
        let clip = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();

        project
            .set_clip_animation(clip, AnimationSlot::In, Some(AnimationRef::new("fade_in")))
            .unwrap();
        project
            .set_clip_animation(clip, AnimationSlot::Out, Some(AnimationRef::new("drop")))
            .unwrap();
        let c = project.clip(clip).unwrap();
        assert_eq!(c.animation_in.as_ref().unwrap().id, "fade_in");
        assert_eq!(c.animation_out.as_ref().unwrap().id, "drop");

        // A combo replaces both sides…
        project
            .set_clip_animation(clip, AnimationSlot::Combo, Some(AnimationRef::new("pulse")))
            .unwrap();
        let c = project.clip(clip).unwrap();
        assert!(c.animation_in.is_none() && c.animation_out.is_none());
        assert_eq!(c.animation_combo.as_ref().unwrap().id, "pulse");

        // …and an entrance evicts the combo again.
        project
            .set_clip_animation(clip, AnimationSlot::In, Some(AnimationRef::new("zoom_in")))
            .unwrap();
        let c = project.clip(clip).unwrap();
        assert!(c.animation_combo.is_none());

        // Wrong slot, unknown id, and text-only presets on media are rejected.
        assert!(
            project
                .set_clip_animation(clip, AnimationSlot::Out, Some(AnimationRef::new("fade_in")))
                .is_err()
        );
        assert!(
            project
                .set_clip_animation(clip, AnimationSlot::In, Some(AnimationRef::new("nope")))
                .is_err()
        );
        assert!(
            project
                .set_clip_animation(
                    clip,
                    AnimationSlot::Combo,
                    Some(AnimationRef::new("typewriter"))
                )
                .is_err()
        );

        // Text clips accept the text-only presets.
        let text = project.add_track(TrackKind::Text, "T1");
        let title = project
            .add_generated(text, Generator::text("Hi"), tr(0, 24))
            .unwrap();
        assert!(
            project
                .set_clip_animation(
                    title,
                    AnimationSlot::Combo,
                    Some(AnimationRef::new("typewriter"))
                )
                .is_ok()
        );
    }

    #[test]
    fn audio_role_tags_audio_lane_clips_only() {
        let (mut project, media_id, video) = project_with_media(100);
        let vclip = project.add_clip(video, media_id, tr(0, 50), rt(0)).unwrap();
        assert!(
            project
                .set_clip_audio_role(vclip, Some(AudioRole::Music))
                .is_err()
        );

        let audio = project.add_track(TrackKind::Audio, "A1");
        let aclip = project.add_clip(audio, media_id, tr(0, 50), rt(0)).unwrap();
        project
            .set_clip_audio_role(aclip, Some(AudioRole::Voiceover))
            .unwrap();
        assert_eq!(
            project.clip(aclip).unwrap().audio_role,
            Some(AudioRole::Voiceover)
        );
        project.set_clip_audio_role(aclip, None).unwrap();
        assert!(project.clip(aclip).unwrap().audio_role.is_none());
    }

    #[test]
    fn text_effect_presets_bake_onto_the_style() {
        let mut project = Project::new("t", R24);
        let track = project.add_track(TrackKind::Text, "T1");

        let style = crate::TextStyle {
            effect_preset: Some("neon".into()),
            ..Default::default()
        };
        let clip = project
            .add_generated(
                track,
                Generator::Text {
                    content: "Glow".into(),
                    style,
                },
                tr(0, 24),
            )
            .unwrap();

        let ClipSource::Generated(Generator::Text { style, .. }) =
            &project.clip(clip).unwrap().content
        else {
            panic!("expected text");
        };
        assert_eq!(style.effect_preset.as_deref(), Some("neon"));
        let spec = crate::look::text_effect_spec("neon").unwrap();
        assert_eq!(style.stroke, spec.stroke);
        assert_eq!(style.shadow, spec.shadow);
        assert_eq!(style.background, spec.background);

        // Unknown presets are rejected at the door.
        let bad = crate::TextStyle {
            effect_preset: Some("nope".into()),
            ..Default::default()
        };
        assert!(
            project
                .set_generator(
                    clip,
                    Generator::Text {
                        content: "Glow".into(),
                        style: bad,
                    }
                )
                .is_err()
        );

        // Clearing the preset keeps whatever treatments the style carries.
        let manual = crate::TextStyle {
            effect_preset: None,
            stroke: Some(crate::TextStroke {
                rgba: [1, 2, 3, 255],
                width: 2.0,
            }),
            ..Default::default()
        };
        project
            .set_generator(
                clip,
                Generator::Text {
                    content: "Glow".into(),
                    style: manual.clone(),
                },
            )
            .unwrap();
        let ClipSource::Generated(Generator::Text { style, .. }) =
            &project.clip(clip).unwrap().content
        else {
            panic!("expected text");
        };
        assert_eq!(style.stroke, manual.stroke);
    }

    #[test]
    fn look_fields_roundtrip_and_stay_off_the_wire_when_unset() {
        let (mut project, media_id, track) = project_with_media(100);
        let plain = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();
        let styled = project
            .add_clip(track, media_id, tr(50, 50), rt(50))
            .unwrap();
        project
            .set_clip_mask(styled, Some(Mask::new(MaskKind::Star)))
            .unwrap();
        project
            .set_clip_filter(styled, Some(Filter::new("sunset")))
            .unwrap();

        let plain_json = serde_json::to_value(project.clip(plain).unwrap()).unwrap();
        for key in [
            "mask",
            "chroma_key",
            "stabilize",
            "filter",
            "adjust",
            "audio_role",
        ] {
            assert!(
                plain_json.get(key).is_none(),
                "untouched clip should not serialize `{key}`"
            );
        }
        let styled_json = serde_json::to_value(project.clip(styled).unwrap()).unwrap();
        assert_eq!(styled_json["mask"]["kind"], "star");
        assert_eq!(styled_json["filter"]["id"], "sunset");

        // The whole project (with looks) survives a save/load roundtrip.
        let json = serde_json::to_value(&project).unwrap();
        let back: Project = serde_json::from_value(json).unwrap();
        assert_eq!(
            back.clip(styled).unwrap().mask,
            project.clip(styled).unwrap().mask
        );
        assert_eq!(
            back.clip(styled).unwrap().filter,
            project.clip(styled).unwrap().filter
        );
        assert!(back.clip(plain).unwrap().mask.is_none());
    }

    #[test]
    fn speed_preset_catalog_ids_resolve_and_match_back() {
        for spec in crate::speed_preset_catalog() {
            let curve = crate::speed_preset(spec.id)
                .unwrap_or_else(|| panic!("catalog id '{}' has no curve", spec.id));
            crate::validate_speed_curve(&curve)
                .unwrap_or_else(|e| panic!("preset '{}' curve invalid: {e}", spec.id));
            assert_eq!(crate::speed_preset_id(&curve), Some(spec.id));
        }
        assert_eq!(crate::speed_preset_id(&Param::Constant(1.0)), None);
    }
}

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::Map;
use crate::clip::{Clip, ClipParam, ClipSource, ClipTransform, CropRect, Generator, ParamValue};
use crate::effects::EffectInstance;
use crate::error::ModelError;
use crate::ids::{ClipId, MediaId, ProjectId, TrackId};
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
        self.timeline.insert_track(Track::new(kind, name), order_index)
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
        if source.start.value < 0
            || (!media.is_image && source.end_tick() > media.duration.value)
        {
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
        let clip = Clip::generated(generator, timeline);
        self.timeline.add_clip(track_id, clip)
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
            _ => clip.transform.set_param_keyframe(param, tick, value, easing),
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
        let right = self
            .right_neighbor(track_id, left)
            .ok_or_else(|| ModelError::InvalidParam("clip has no abutting clip to its right".into()))?;
        let transition =
            Transition::new(left, right, transition_id, crate::transition::DEFAULT_TRANSITION_TICKS);
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
                let media = self.media.get(&media).ok_or(ModelError::UnknownMedia(media))?;
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
        let new_dur = retimed_duration(src_dur_tl, speed, clip.speed_curve_average(), clip.has_speed_curve());
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
        ParamValue::Vec2(_) => Err(ModelError::InvalidParam(
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

    // --- transitions (M4) -------------------------------------------------

    /// Two abutting adjustment clips on one track; returns `(project, left,
    /// right, track)`.
    fn project_with_abutting_pair() -> (Project, ClipId, ClipId, TrackId) {
        let mut project = Project::new("test", R24);
        let track = project.add_track(TrackKind::Adjustment, "FX");
        let left = project
            .add_generated(track, Generator::Adjustment, tr(0, 24))
            .unwrap();
        let right = project
            .add_generated(track, Generator::Adjustment, tr(24, 24))
            .unwrap();
        (project, left, right, track)
    }

    #[test]
    fn add_transition_links_abutting_pair() {
        let (mut project, left, right, track) = project_with_abutting_pair();
        project.add_transition(left, "crossfade").unwrap();
        let t = project.timeline().track(track).unwrap().transition_at(left).unwrap();
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
        let track = project.add_track(TrackKind::Adjustment, "FX2");
        let lone = project
            .add_generated(track, Generator::Adjustment, tr(0, 24))
            .unwrap();
        assert!(matches!(
            project.add_transition(lone, "crossfade"),
            Err(ModelError::InvalidParam(_))
        ));
    }

    #[test]
    fn set_and_remove_transition_duration() {
        let (mut project, left, _right, track) = project_with_abutting_pair();
        project.add_transition(left, "wipe_left").unwrap();
        project.set_transition_duration(left, 12).unwrap();
        assert_eq!(
            project.timeline().track(track).unwrap().transition_at(left).unwrap().duration,
            12
        );
        project.remove_transition(left).unwrap();
        assert!(project.timeline().track(track).unwrap().transition_at(left).is_none());
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
        assert_eq!(project.find_media_by_path(Path::new("/tmp/other.mp4")), None);
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
        project.add_clip(track, media_id, tr(0, 100), rt(0)).unwrap();
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

    // --- remove_clip ------------------------------------------------------

    #[test]
    fn remove_clip_returns_clip_and_leaves_gap() {
        let (mut project, media_id, track) = project_with_media(200);
        let a = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();
        let b = project.add_clip(track, media_id, tr(50, 50), rt(100)).unwrap();

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
        assert_eq!(
            project.clip(generated_right).unwrap().timeline,
            tr(250, 50)
        );
    }

    #[test]
    fn split_clip_rejects_at_boundaries() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project
            .add_clip(track, media_id, tr(0, 100), rt(10))
            .unwrap();
        assert_eq!(project.split_clip(clip, rt(10)), Err(ModelError::InvalidRange));
        assert_eq!(project.split_clip(clip, rt(110)), Err(ModelError::InvalidRange));
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
        let clip = project
            .add_clip(track, media_id, tr(0, 50), rt(0))
            .unwrap();
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
        let clip = project.add_clip(track, media_id, tr(0, 100), rt(0)).unwrap();

        // 2× halves the footprint; the source window is untouched.
        project.set_clip_speed(clip, Rational::new(2, 1), false).unwrap();
        let c = project.clip(clip).unwrap();
        assert_eq!(c.timeline, tr(0, 50));
        assert_eq!(c.source_range(), Some(tr(0, 100)));
        assert_eq!(c.speed, Rational::new(2, 1));

        // Back to 1× restores the original footprint.
        project.set_clip_speed(clip, Rational::new(1, 1), false).unwrap();
        assert_eq!(project.clip(clip).unwrap().timeline, tr(0, 100));

        // Slow motion stretches it.
        project.set_clip_speed(clip, Rational::new(1, 2), false).unwrap();
        assert_eq!(project.clip(clip).unwrap().timeline, tr(0, 200));
    }

    #[test]
    fn set_clip_pitch_toggles_the_flag_without_moving_the_clip() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project.add_clip(track, media_id, tr(0, 100), rt(0)).unwrap();
        project.set_clip_speed(clip, Rational::new(2, 1), false).unwrap();
        assert!(project.clip(clip).unwrap().preserve_pitch, "locked by default");

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
            .add_generated(track, Generator::SolidColor { rgba: [0, 0, 0, 255] }, tr(0, 10))
            .unwrap();
        assert!(project.set_clip_pitch(clip, false).is_err(), "no audio to retime");
    }

    #[test]
    fn set_clip_speed_curve_rederives_duration_from_average() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project.add_clip(track, media_id, tr(0, 100), rt(0)).unwrap();

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
        let clip = project.add_clip(track, media_id, tr(0, 100), rt(0)).unwrap();
        // Out-of-range ramp value.
        let bad = Param::Keyframed {
            keyframes: vec![
                crate::param::Keyframe { tick: 0, value: 0.0, easing: Easing::Linear },
                crate::param::Keyframe { tick: 1000, value: 1.0, easing: Easing::Linear },
            ],
        };
        assert!(project.set_clip_speed_curve(clip, Some(bad)).is_err());
        // Generated clips cannot be retimed.
        let sticker = project.add_track(TrackKind::Sticker, "S");
        let generated = project
            .add_generated(sticker, Generator::SolidColor { rgba: [0, 0, 0, 255] }, tr(200, 10))
            .unwrap();
        assert!(matches!(
            project.set_clip_speed_curve(generated, Some(crate::clip::speed_preset("hero").unwrap())),
            Err(ModelError::InvalidParam(_))
        ));
    }

    #[test]
    fn set_clip_speed_reverse_keeps_duration() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project.add_clip(track, media_id, tr(0, 100), rt(0)).unwrap();

        project.set_clip_speed(clip, Rational::new(1, 1), true).unwrap();
        let c = project.clip(clip).unwrap();
        assert_eq!(c.timeline, tr(0, 100));
        assert!(c.reversed && c.is_retimed());
    }

    #[test]
    fn set_clip_speed_rejects_invalid_targets() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project.add_clip(track, media_id, tr(0, 100), rt(0)).unwrap();

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
        let clip = project.add_clip(track, media_id, tr(0, 100), rt(0)).unwrap();
        // Neighbor right behind the clip: slowing to ½× (200 ticks) collides.
        project.add_clip(track, media_id, tr(0, 50), rt(100)).unwrap();

        assert_eq!(
            project.set_clip_speed(clip, Rational::new(1, 2), false),
            Err(ModelError::Overlap(track))
        );
        // The clip is untouched after the rejection.
        let c = project.clip(clip).unwrap();
        assert_eq!(c.timeline, tr(0, 100));
        assert_eq!(c.speed, Rational::new(1, 1));
    }

    // --- set_clip_audio (M1) -------------------------------------------------

    #[test]
    fn set_clip_audio_sets_volume_and_fades() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project.add_clip(track, media_id, tr(0, 100), rt(0)).unwrap();

        project.set_clip_audio(clip, Some(0.5), rt(10), rt(20)).unwrap();
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
        project.set_clip_audio(clip, Some(1.0), rt(0), rt(0)).unwrap();
        assert!(!project.clip(clip).unwrap().has_custom_audio());
    }

    #[test]
    fn set_clip_audio_none_volume_preserves_envelope() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project.add_clip(track, media_id, tr(0, 100), rt(0)).unwrap();
        // A keyframed gain envelope (dips to 0.2 mid-clip).
        project
            .set_param_keyframe(clip, ClipParam::Volume, rt(0), ParamValue::Scalar(1.0), Easing::Linear)
            .unwrap();
        project
            .set_param_keyframe(clip, ClipParam::Volume, rt(50), ParamValue::Scalar(0.2), Easing::Linear)
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
        let clip = project.add_clip(track, media_id, tr(0, 100), rt(0)).unwrap();

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
        let clip = project.add_clip(track, media_id, tr(0, 100), rt(0)).unwrap();
        project.set_clip_audio(clip, Some(0.5), rt(10), rt(20)).unwrap();

        let right = project.split_clip(clip, rt(60)).unwrap();
        let left = project.clip(clip).unwrap();
        let right = project.clip(right).unwrap();
        // Volume rides both halves; the fade-in stays with the head, the
        // fade-out moves to the tail.
        assert_eq!((left.volume.constant(), right.volume.constant()), (Some(0.5), Some(0.5)));
        assert_eq!((left.fade_in, left.fade_out), (10, 0));
        assert_eq!((right.fade_in, right.fade_out), (0, 20));
    }

    // --- set_clip_crop (M1) ---------------------------------------------------

    #[test]
    fn set_clip_crop_sets_framing() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project.add_clip(track, media_id, tr(0, 100), rt(0)).unwrap();

        let crop = CropRect { x: 0.25, y: 0.0, w: 0.5, h: 1.0 };
        project.set_clip_crop(clip, crop, true, false).unwrap();
        let c = project.clip(clip).unwrap();
        assert_eq!(c.crop, crop);
        assert!(c.flip_h && !c.flip_v);
        assert!(c.has_custom_crop());

        // Back to the full frame clears the custom-framing state.
        project.set_clip_crop(clip, CropRect::FULL, false, false).unwrap();
        assert!(!project.clip(clip).unwrap().has_custom_crop());
    }

    #[test]
    fn set_clip_crop_rejects_invalid_targets() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project.add_clip(track, media_id, tr(0, 100), rt(0)).unwrap();

        // Invalid rects bounce.
        assert!(matches!(
            project.set_clip_crop(clip, CropRect { x: 0.8, y: 0.0, w: 0.5, h: 1.0 }, false, false),
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
        let clip = project.add_clip(track, media_id, tr(0, 100), rt(0)).unwrap();
        let crop = CropRect { x: 0.1, y: 0.1, w: 0.8, h: 0.8 };
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
        let clip = project.add_clip(track, media_id, tr(100, 100), rt(0)).unwrap();
        project.set_clip_speed(clip, Rational::new(2, 1), false).unwrap();
        // Now timeline [0, 50) over source [100, 200).

        // Head trim by 10 timeline ticks eats 20 source ticks.
        project.trim_clip(clip, tr(10, 40)).unwrap();
        let c = project.clip(clip).unwrap();
        assert_eq!(c.source_range(), Some(tr(120, 80)));

        // Tail trim to 20 timeline ticks keeps 40 source ticks.
        project.trim_clip(clip, tr(10, 20)).unwrap();
        assert_eq!(project.clip(clip).unwrap().source_range(), Some(tr(120, 40)));
    }

    #[test]
    fn trim_reversed_clip_mirrors_source_window() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project.add_clip(track, media_id, tr(100, 100), rt(0)).unwrap();
        project.set_clip_speed(clip, Rational::new(1, 1), true).unwrap();

        // The timeline head shows the source END: a head trim by 10 drops
        // the top 10 source ticks, keeping the bottom.
        project.trim_clip(clip, tr(10, 90)).unwrap();
        assert_eq!(project.clip(clip).unwrap().source_range(), Some(tr(100, 90)));

        // A tail trim drops the source BOTTOM.
        project.trim_clip(clip, tr(10, 80)).unwrap();
        assert_eq!(project.clip(clip).unwrap().source_range(), Some(tr(110, 80)));
    }

    #[test]
    fn split_retimed_clip_inherits_speed_and_partitions_source() {
        let (mut project, media_id, track) = project_with_media(500);
        let clip = project.add_clip(track, media_id, tr(0, 100), rt(0)).unwrap();
        project.set_clip_speed(clip, Rational::new(2, 1), false).unwrap();
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
        let clip = project.add_clip(track, media_id, tr(100, 100), rt(0)).unwrap();
        project.set_clip_speed(clip, Rational::new(1, 1), true).unwrap();

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
        let clip = project
            .add_clip(track, media_id, tr(0, 50), rt(0))
            .unwrap();

        project.move_clip(clip, track, rt(80)).unwrap();
        assert_eq!(project.clip(clip).unwrap().timeline, tr(80, 50));
        assert_eq!(project.timeline().track_of(clip), Some(track));
    }

    #[test]
    fn move_clip_rejects_negative_start_and_unknown_track() {
        let (mut project, media_id, track) = project_with_media(200);
        let clip = project
            .add_clip(track, media_id, tr(0, 50), rt(0))
            .unwrap();
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
        let b = project.add_clip(track, media_id, tr(50, 50), rt(50)).unwrap();
        let c = project.add_clip(track, media_id, tr(100, 50), rt(100)).unwrap();
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
        let b = project.add_clip(track, media_id, tr(50, 50), rt(50)).unwrap();
        let c = project.add_clip(track, media_id, tr(100, 50), rt(150)).unwrap();

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
        assert_eq!(project.timeline().track(track).unwrap().kind, TrackKind::Audio);
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
}

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

use super::Project;

impl Project {
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
        if self
            .timeline
            .clip(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?
            .freeze_frame
        {
            return Err(ModelError::InvalidParam(
                "cannot replace the media of a freeze-frame clip".into(),
            ));
        }
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
}

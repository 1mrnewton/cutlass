use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::Map;
use crate::clip::{Clip, ClipParam, ClipSource, ClipTransform, Generator, ParamValue};
use crate::error::ModelError;
use crate::ids::{ClipId, MediaId, ProjectId, TrackId};
use crate::media::MediaSource;
use crate::metadata::ProjectMetadata;
use crate::param::Easing;
use crate::schema::ProjectSchema;
use crate::time::{
    Rational, RationalTime, TimeRange, check_same_rate, resample, time_add, time_sub,
};
use crate::timeline::Timeline;
use crate::track::{Track, TrackKind};

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
        if source.start.value < 0 || source.end_tick() > media.duration.value {
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
        self.check_param_target(clip_id)?;
        let tick = self.keyframe_tick(clip_id, at)?;
        self.timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?
            .transform
            .set_param_keyframe(param, tick, value, easing)
    }

    /// Remove the keyframe at exactly `at` (absolute timeline position) on
    /// one property. Errors when no keyframe sits there.
    pub fn remove_param_keyframe(
        &mut self,
        clip_id: ClipId,
        param: ClipParam,
        at: RationalTime,
    ) -> Result<(), ModelError> {
        self.check_param_target(clip_id)?;
        let tick = self.keyframe_tick(clip_id, at)?;
        self.timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?
            .transform
            .remove_param_keyframe(param, tick)
    }

    /// Replace one animatable property with a constant, dropping keyframes.
    pub fn set_param_constant(
        &mut self,
        clip_id: ClipId,
        param: ClipParam,
        value: ParamValue,
    ) -> Result<(), ModelError> {
        self.check_param_target(clip_id)?;
        self.timeline
            .clip_mut(clip_id)
            .ok_or(ModelError::UnknownClip(clip_id))?
            .transform
            .set_param_constant(param, value)
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

        let (new_left_source, new_clip) = match clip.content.clone() {
            ClipSource::Media { media, source } => {
                let media_fps = self
                    .media
                    .get(&media)
                    .ok_or(ModelError::UnknownMedia(media))?
                    .frame_rate;
                if source.duration.value < 2 {
                    return Err(ModelError::InvalidRange);
                }
                let left_src_dur = resample(left_tl.duration, media_fps)
                    .value
                    .clamp(1, source.duration.value - 1);
                let left_source = TimeRange::at_rate(source.start.value, left_src_dur, media_fps);
                let right_source = TimeRange::at_rate(
                    source.start.value + left_src_dur,
                    source.duration.value - left_src_dur,
                    media_fps,
                );
                (Some(left_source), Clip::from_media(media, right_source, right_tl))
            }
            ClipSource::Generated(generator) => (None, Clip::generated(generator, right_tl)),
        };

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
                let head_delta =
                    resample(
                        RationalTime::new(new_timeline.start.value - old_tl.start.value, tl_rate),
                        media.frame_rate,
                    )
                    .value;
                let new_src_start = source.start.value + head_delta;
                let new_src_dur = resample(new_timeline.duration, media.frame_rate)
                    .value
                    .max(1);
                if new_src_start < 0 || new_src_start + new_src_dur > media.duration.value {
                    return Err(ModelError::SourceOutOfBounds);
                }
                Some(TimeRange::at_rate(
                    new_src_start,
                    new_src_dur,
                    media.frame_rate,
                ))
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
                Generator::Shape {
                    shape: Shape::Rectangle,
                    rgba: [255, 255, 255, 255],
                },
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

use crate::Map;
use crate::error::ModelError;
use crate::ids::{ClipId, MediaId, ProjectId, TrackId};
use crate::media::MediaSource;
use crate::time::{Rational, TimeRange, convert_frames};
use crate::timeline::Timeline;
use crate::track::{Track, TrackKind};
use crate::clip::{Clip, ClipSource, Generator};

/// Top-level container: a media pool plus exactly one [`Timeline`].
///
/// `Project` is the aggregate root and the only place that can guarantee
/// referential integrity between clips and media, so clip creation goes through
/// [`add_clip`](Project::add_clip).
#[derive(Debug, Clone)]
pub struct Project {
    pub id: ProjectId,
    pub name: String,
    media: Map<MediaId, MediaSource>,
    timeline: Timeline,
}

impl Project {
    /// Create an empty project whose timeline runs at `frame_rate`.
    pub fn new(name: impl Into<String>, frame_rate: Rational) -> Self {
        Self {
            id: ProjectId::next(),
            name: name.into(),
            media: Map::default(),
            timeline: Timeline::new(frame_rate),
        }
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

    /// Place a clip referencing `media_id` on `track_id`.
    ///
    /// The clip's timeline duration is conformed from the source's frame rate to
    /// the timeline's frame rate, so a 30fps source on a 24fps timeline occupies
    /// the right number of timeline frames. Validates media/track existence,
    /// that `source` is within the media bounds, and that the placement does not
    /// overlap an existing clip on the track.
    pub fn add_clip(
        &mut self,
        track_id: TrackId,
        media_id: MediaId,
        source: TimeRange,
        timeline_start: i64,
    ) -> Result<ClipId, ModelError> {
        let media = self
            .media
            .get(&media_id)
            .ok_or(ModelError::UnknownMedia(media_id))?;

        if source.is_empty() {
            return Err(ModelError::InvalidRange);
        }
        if source.start < 0 || source.end() > media.duration {
            return Err(ModelError::SourceOutOfBounds);
        }

        let timeline_duration =
            convert_frames(source.duration, media.frame_rate, self.timeline.frame_rate);
        let timeline = TimeRange::new(timeline_start, timeline_duration.max(1));

        let clip = Clip::from_media(media_id, source, timeline);
        self.timeline.add_clip(track_id, clip)
    }

    /// Place a generated clip (text, shape, solid, ...) on `track_id`.
    ///
    /// Generated content has no source media, so the caller specifies the
    /// timeline placement directly (in timeline frames).
    pub fn add_generated(
        &mut self,
        track_id: TrackId,
        generator: Generator,
        timeline: TimeRange,
    ) -> Result<ClipId, ModelError> {
        if timeline.is_empty() {
            return Err(ModelError::InvalidRange);
        }
        let clip = Clip::generated(generator, timeline);
        self.timeline.add_clip(track_id, clip)
    }

    /// Find a clip by ID anywhere on the timeline (O(1)).
    pub fn clip(&self, clip_id: ClipId) -> Option<&Clip> {
        self.timeline.clip(clip_id)
    }
}

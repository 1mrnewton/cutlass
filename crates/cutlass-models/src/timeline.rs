use crate::Map;
use crate::clip::Clip;
use crate::error::ModelError;
use crate::ids::{ClipId, TrackId};
use crate::time::Rational;
use crate::track::Track;

/// The single sequence of a [`Project`](crate::Project): an ordered stack of
/// tracks plus a clip-location index.
///
/// - `tracks` is keyed by [`TrackId`] for O(1) lookup.
/// - `order` is the z-stack from bottom (index 0) to top; the topmost enabled
///   video track wins when compositing.
/// - `clip_index` maps every [`ClipId`] to the track containing it, so a clip
///   can be found across the whole timeline in O(1) without scanning tracks.
#[derive(Debug, Clone)]
pub struct Timeline {
    /// Editing/playback frame rate. Clip `timeline` ranges are in these frames.
    pub frame_rate: Rational,
    tracks: Map<TrackId, Track>,
    order: Vec<TrackId>,
    clip_index: Map<ClipId, TrackId>,
}

impl Timeline {
    pub fn new(frame_rate: Rational) -> Self {
        Self {
            frame_rate,
            tracks: Map::default(),
            order: Vec::new(),
            clip_index: Map::default(),
        }
    }

    // --- tracks -----------------------------------------------------------

    /// Append a track to the top of the stack. Returns its [`TrackId`].
    pub fn add_track(&mut self, track: Track) -> TrackId {
        let id = track.id;
        self.tracks.insert(id, track);
        self.order.push(id);
        id
    }

    pub fn track(&self, id: TrackId) -> Option<&Track> {
        self.tracks.get(&id)
    }

    pub fn track_mut(&mut self, id: TrackId) -> Option<&mut Track> {
        self.tracks.get_mut(&id)
    }

    /// Track IDs from bottom to top of the stack.
    pub fn order(&self) -> &[TrackId] {
        &self.order
    }

    /// Tracks in stacking order (bottom to top).
    pub fn tracks_ordered(&self) -> impl Iterator<Item = &Track> {
        self.order.iter().filter_map(move |id| self.tracks.get(id))
    }

    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Remove a track and all its clips (also purging the clip index).
    pub fn remove_track(&mut self, id: TrackId) -> Option<Track> {
        let track = self.tracks.remove(&id)?;
        self.order.retain(|t| *t != id);
        for clip in track.clips() {
            self.clip_index.remove(&clip.id);
        }
        Some(track)
    }

    // --- clips ------------------------------------------------------------

    /// Place `clip` on `track_id`, rejecting unknown tracks and overlaps.
    pub fn add_clip(&mut self, track_id: TrackId, clip: Clip) -> Result<ClipId, ModelError> {
        let track = self
            .tracks
            .get_mut(&track_id)
            .ok_or(ModelError::UnknownTrack(track_id))?;

        if track.has_overlap(clip.timeline, None) {
            return Err(ModelError::Overlap(track_id));
        }

        let clip_id = clip.id;
        track.insert_clip(clip);
        self.clip_index.insert(clip_id, track_id);
        Ok(clip_id)
    }

    /// Remove a clip by ID from wherever it lives.
    pub fn remove_clip(&mut self, clip_id: ClipId) -> Option<Clip> {
        let track_id = self.clip_index.remove(&clip_id)?;
        self.tracks.get_mut(&track_id)?.remove_clip(clip_id)
    }

    /// Find a clip by ID across all tracks in O(1).
    pub fn clip(&self, clip_id: ClipId) -> Option<&Clip> {
        let track_id = *self.clip_index.get(&clip_id)?;
        self.tracks.get(&track_id)?.clip(clip_id)
    }

    pub fn clip_mut(&mut self, clip_id: ClipId) -> Option<&mut Clip> {
        let track_id = *self.clip_index.get(&clip_id)?;
        self.tracks.get_mut(&track_id)?.clip_mut(clip_id)
    }

    /// The track that contains `clip_id`, if any.
    pub fn track_of(&self, clip_id: ClipId) -> Option<TrackId> {
        self.clip_index.get(&clip_id).copied()
    }

    pub fn clip_count(&self) -> usize {
        self.clip_index.len()
    }

    /// Total timeline length in frames: the end of the last-ending clip.
    pub fn duration(&self) -> i64 {
        self.tracks
            .values()
            .map(Track::content_end)
            .max()
            .unwrap_or(0)
    }
}

use serde::{Deserialize, Serialize};

use crate::Map;
use crate::clip::Clip;
use crate::error::ModelError;
use crate::ids::{ClipId, TrackId};
use crate::time::{Rational, RationalTime};
use crate::track::Track;

/// The single sequence of a [`Project`](crate::Project): an ordered stack of
/// tracks plus a clip-location index.
///
/// - `tracks` is keyed by [`TrackId`] for O(1) lookup.
/// - `order` is the z-stack from bottom (index 0) to top; the topmost enabled
///   video track wins when compositing.
/// - `clip_index` maps every [`ClipId`] to the track containing it, so a clip
///   can be found across the whole timeline in O(1) without scanning tracks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timeline {
    /// Editing/playback frame rate. Clip `timeline` ranges are in these frames.
    pub frame_rate: Rational,
    #[serde(with = "crate::serde_map")]
    tracks: Map<TrackId, Track>,
    order: Vec<TrackId>,
    #[serde(with = "crate::serde_map")]
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

    /// Re-insert a removed track at its prior stack position (undo of [`remove_track`]).
    pub fn restore_track(
        &mut self,
        track: Track,
        order_index: usize,
    ) -> Result<TrackId, ModelError> {
        let id = track.id;
        if self.tracks.contains_key(&id) {
            return Err(ModelError::InvalidRange);
        }
        for clip in track.clips() {
            if self.clip_index.contains_key(&clip.id) {
                return Err(ModelError::InvalidRange);
            }
        }
        let clip_ids: Vec<ClipId> = track.clips().map(|c| c.id).collect();
        let idx = order_index.min(self.order.len());
        self.order.insert(idx, id);
        self.tracks.insert(id, track);
        for clip_id in clip_ids {
            self.clip_index.insert(clip_id, id);
        }
        Ok(id)
    }

    // --- clips ------------------------------------------------------------

    /// Place `clip` on `track_id`, rejecting unknown tracks and overlaps.
    pub fn add_clip(&mut self, track_id: TrackId, clip: Clip) -> Result<ClipId, ModelError> {
        let track = self
            .tracks
            .get_mut(&track_id)
            .ok_or(ModelError::UnknownTrack(track_id))?;

        if track.has_overlap(clip.timeline, None)? {
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

    /// Total timeline length: the end of the last-ending clip at [`frame_rate`](Self::frame_rate).
    pub fn duration(&self) -> RationalTime {
        let tick = self
            .tracks
            .values()
            .map(Track::content_end)
            .max()
            .unwrap_or(0);
        RationalTime::new(tick, self.frame_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{Clip, Generator};
    use crate::time::TimeRange;
    use crate::track::{Track, TrackKind};

    const R24: Rational = Rational::FPS_24;

    fn rt(value: i64) -> RationalTime {
        RationalTime::new(value, R24)
    }

    fn tr(start: i64, duration: i64) -> TimeRange {
        TimeRange::at_rate(start, duration, R24)
    }

    fn generated_clip(start: i64, duration: i64) -> Clip {
        Clip::generated(Generator::Adjustment, tr(start, duration))
    }

    fn timeline_with_track() -> (Timeline, TrackId) {
        let mut timeline = Timeline::new(R24);
        let track = timeline.add_track(Track::new(TrackKind::Video, "V1"));
        (timeline, track)
    }

    // --- Timeline::new ----------------------------------------------------

    #[test]
    fn new_starts_empty_at_frame_rate() {
        let timeline = Timeline::new(R24);
        assert_eq!(timeline.frame_rate, R24);
        assert_eq!(timeline.track_count(), 0);
        assert_eq!(timeline.clip_count(), 0);
        assert!(timeline.order().is_empty());
        assert_eq!(timeline.duration(), rt(0));
    }

    // --- tracks -----------------------------------------------------------

    #[test]
    fn add_track_appends_to_stack_order() {
        let mut timeline = Timeline::new(R24);
        let v1 = timeline.add_track(Track::new(TrackKind::Video, "V1"));
        let v2 = timeline.add_track(Track::new(TrackKind::Video, "V2"));
        let a1 = timeline.add_track(Track::new(TrackKind::Audio, "A1"));

        assert_eq!(timeline.order(), &[v1, v2, a1]);
        assert_eq!(timeline.track_count(), 3);
        assert_eq!(timeline.track(v1).unwrap().name, "V1");
        assert_eq!(timeline.track(a1).unwrap().kind, TrackKind::Audio);
    }

    #[test]
    fn tracks_ordered_yields_bottom_to_top() {
        let mut timeline = Timeline::new(R24);
        timeline.add_track(Track::new(TrackKind::Video, "bottom"));
        timeline.add_track(Track::new(TrackKind::Video, "top"));
        let names: Vec<&str> = timeline
            .tracks_ordered()
            .map(|t| t.name.as_str())
            .collect();
        assert_eq!(names, ["bottom", "top"]);
    }

    #[test]
    fn track_mut_can_toggle_enabled() {
        let mut timeline = Timeline::new(R24);
        let id = timeline.add_track(Track::new(TrackKind::Video, "V1"));
        timeline.track_mut(id).unwrap().enabled = false;
        assert!(!timeline.track(id).unwrap().enabled);
    }

    #[test]
    fn restore_track_reinserts_stack_position_and_clip_index() {
        let (mut timeline, track_id) = timeline_with_track();
        let clip_id = timeline
            .add_clip(track_id, generated_clip(0, 10))
            .expect("clip");
        let track = timeline.remove_track(track_id).expect("remove");
        assert_eq!(timeline.track_count(), 0);
        assert!(timeline.clip(clip_id).is_none());

        timeline
            .restore_track(track, 0)
            .expect("restore");
        assert_eq!(timeline.track_count(), 1);
        assert_eq!(timeline.track_of(clip_id), Some(track_id));
    }

    #[test]
    fn remove_track_purges_clips_from_index() {
        let (mut timeline, track) = timeline_with_track();
        let clip = timeline
            .add_clip(track, generated_clip(0, 50))
            .unwrap();

        let removed = timeline.remove_track(track).unwrap();
        assert_eq!(removed.len(), 1);
        assert_eq!(timeline.track_count(), 0);
        assert_eq!(timeline.clip_count(), 0);
        assert!(timeline.clip(clip).is_none());
        assert!(timeline.track_of(clip).is_none());
    }

    #[test]
    fn remove_unknown_track_returns_none() {
        let mut timeline = Timeline::new(R24);
        assert!(timeline.remove_track(TrackId::from_raw(99)).is_none());
    }

    // --- add_clip / clip index --------------------------------------------

    #[test]
    fn add_clip_registers_in_track_and_index() {
        let (mut timeline, track) = timeline_with_track();
        let clip = generated_clip(10, 40);
        let clip_id = clip.id;

        let returned = timeline.add_clip(track, clip).unwrap();
        assert_eq!(returned, clip_id);
        assert_eq!(timeline.clip_count(), 1);
        assert_eq!(timeline.track_of(clip_id), Some(track));
        assert_eq!(timeline.clip(clip_id).unwrap().timeline, tr(10, 40));
        assert_eq!(timeline.track(track).unwrap().len(), 1);
    }

    #[test]
    fn add_clip_unknown_track_errors() {
        let (mut timeline, _) = timeline_with_track();
        let missing = TrackId::from_raw(404);
        assert_eq!(
            timeline.add_clip(missing, generated_clip(0, 10)),
            Err(ModelError::UnknownTrack(missing))
        );
    }

    #[test]
    fn add_clip_rejects_overlap_on_same_track() {
        let (mut timeline, track) = timeline_with_track();
        timeline.add_clip(track, generated_clip(0, 50)).unwrap();
        assert_eq!(
            timeline.add_clip(track, generated_clip(25, 50)),
            Err(ModelError::Overlap(track))
        );
    }

    #[test]
    fn add_clip_allows_same_range_on_different_tracks() {
        let mut timeline = Timeline::new(R24);
        let v1 = timeline.add_track(Track::new(TrackKind::Video, "V1"));
        let v2 = timeline.add_track(Track::new(TrackKind::Video, "V2"));

        let c1 = timeline.add_clip(v1, generated_clip(0, 50)).unwrap();
        let c2 = timeline.add_clip(v2, generated_clip(0, 50)).unwrap();
        assert_ne!(c1, c2);
        assert_eq!(timeline.clip_count(), 2);
    }

    #[test]
    fn add_clip_allows_adjacent_non_overlapping_clips() {
        let (mut timeline, track) = timeline_with_track();
        timeline.add_clip(track, generated_clip(0, 50)).unwrap();
        let second = timeline.add_clip(track, generated_clip(50, 50)).unwrap();
        assert_eq!(timeline.clip_count(), 2);
        assert_eq!(timeline.clip(second).unwrap().start().value, 50);
    }

    // --- remove_clip / lookup ---------------------------------------------

    #[test]
    fn remove_clip_returns_clip_and_clears_index() {
        let (mut timeline, track) = timeline_with_track();
        let id = timeline.add_clip(track, generated_clip(0, 30)).unwrap();

        let removed = timeline.remove_clip(id).unwrap();
        assert_eq!(removed.id, id);
        assert_eq!(timeline.clip_count(), 0);
        assert!(timeline.clip(id).is_none());
        assert!(timeline.track_of(id).is_none());
        assert!(timeline.track(track).unwrap().is_empty());
    }

    #[test]
    fn remove_clip_unknown_returns_none() {
        let (mut timeline, _) = timeline_with_track();
        assert!(timeline.remove_clip(ClipId::from_raw(77)).is_none());
    }

    #[test]
    fn clip_mut_updates_timeline_range() {
        let (mut timeline, track) = timeline_with_track();
        let id = timeline.add_clip(track, generated_clip(0, 50)).unwrap();

        timeline.clip_mut(id).unwrap().timeline = tr(10, 40);
        assert_eq!(timeline.clip(id).unwrap().timeline, tr(10, 40));
    }

    #[test]
    fn clip_lookup_finds_across_tracks() {
        let mut timeline = Timeline::new(R24);
        let v1 = timeline.add_track(Track::new(TrackKind::Video, "V1"));
        let v2 = timeline.add_track(Track::new(TrackKind::Video, "V2"));
        let on_v2 = timeline.add_clip(v2, generated_clip(100, 20)).unwrap();
        timeline.add_clip(v1, generated_clip(0, 10)).unwrap();

        assert_eq!(timeline.track_of(on_v2), Some(v2));
        assert_eq!(timeline.clip(on_v2).unwrap().start().value, 100);
    }

    // --- duration ---------------------------------------------------------

    #[test]
    fn duration_empty_timeline_is_zero() {
        let timeline = Timeline::new(R24);
        assert_eq!(timeline.duration(), rt(0));
    }

    #[test]
    fn duration_is_max_end_across_tracks() {
        let mut timeline = Timeline::new(R24);
        let v1 = timeline.add_track(Track::new(TrackKind::Video, "V1"));
        let v2 = timeline.add_track(Track::new(TrackKind::Video, "V2"));
        timeline.add_clip(v1, generated_clip(0, 100)).unwrap();
        timeline.add_clip(v2, generated_clip(50, 200)).unwrap(); // ends at 250

        assert_eq!(timeline.duration().value, 250);
        assert_eq!(timeline.duration().rate, R24);
    }

    #[test]
    fn duration_ignores_gap_between_clips_on_same_track() {
        let (mut timeline, track) = timeline_with_track();
        timeline.add_clip(track, generated_clip(0, 50)).unwrap();
        timeline.add_clip(track, generated_clip(100, 30)).unwrap(); // ends 130

        assert_eq!(timeline.duration().value, 130);
    }

    // --- Clone ------------------------------------------------------------

    #[test]
    fn clone_is_independent_snapshot() {
        let (mut timeline, track) = timeline_with_track();
        let clip = timeline.add_clip(track, generated_clip(0, 50)).unwrap();

        let mut cloned = timeline.clone();
        cloned.remove_clip(clip);
        assert_eq!(cloned.clip_count(), 0);
        assert_eq!(timeline.clip_count(), 1);
    }
}

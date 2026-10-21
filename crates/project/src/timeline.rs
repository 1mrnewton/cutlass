//! Ordered tracks. The list runs from the bottom of the timeline upward.

use cutlass_core::Timestamp;

use crate::Error;
use crate::id::{SegmentId, TimelineId, TrackId};
use crate::segment::Segment;
use crate::time::SEQUENCE_TIMESCALE;
use crate::track::Track;

/// One timeline. Nested timelines wait.
#[derive(Debug, Clone, PartialEq)]
pub struct Timeline {
    id: TimelineId,
    tracks: Vec<Track>,
}

impl Timeline {
    pub(crate) fn new(id: TimelineId, main_track: TrackId) -> Self {
        Self {
            id,
            tracks: vec![Track::main(main_track)],
        }
    }

    pub const fn id(&self) -> TimelineId {
        self.id
    }

    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    pub fn main_track(&self) -> &Track {
        self.tracks
            .iter()
            .find(|track| track.is_main())
            .expect("a timeline always has a main track")
    }

    /// The end of the last segment, or zero when every track is empty.
    pub fn duration(&self) -> Timestamp {
        let end = self
            .tracks
            .iter()
            .flat_map(Track::segments)
            .map(|segment| segment.timeline().end().value)
            .max()
            .unwrap_or(0);
        Timestamp::new(end, SEQUENCE_TIMESCALE)
    }

    pub fn track(&self, id: TrackId) -> Option<&Track> {
        self.tracks.iter().find(|track| track.id() == id)
    }

    pub fn segment(&self, id: SegmentId) -> Option<&Segment> {
        self.tracks
            .iter()
            .find_map(|track| track.segments().iter().find(|segment| segment.id() == id))
    }

    pub(crate) fn main_mut(&mut self) -> &mut Track {
        self.tracks
            .iter_mut()
            .find(|track| track.is_main())
            .expect("a timeline always has a main track")
    }

    pub(crate) fn track_mut(&mut self, id: TrackId) -> Result<&mut Track, Error> {
        self.tracks
            .iter_mut()
            .find(|track| track.id() == id)
            .ok_or(Error::UnknownTrack(id))
    }

    pub(crate) fn segment_mut(&mut self, id: SegmentId) -> Result<&mut Segment, Error> {
        for track in &mut self.tracks {
            if let Some(segment) = track.segment_mut(id) {
                return Ok(segment);
            }
        }
        Err(Error::UnknownSegment(id))
    }

    pub(crate) fn insert(&mut self, track: Track) -> Result<(), Error> {
        if self
            .tracks
            .iter()
            .any(|existing| existing.id() == track.id())
        {
            return Err(Error::DuplicateTrack(track.id()));
        }
        if track.is_main() {
            return Err(Error::SecondMainTrack);
        }
        if track.segments().is_empty() {
            return Err(Error::EmptyTrack);
        }
        let index = self
            .tracks
            .iter()
            .position(|existing| existing.band() > track.band())
            .unwrap_or(self.tracks.len());
        self.tracks.insert(index, track);
        Ok(())
    }

    pub(crate) fn remove_segment(&mut self, id: SegmentId) -> Result<(), Error> {
        for track in &mut self.tracks {
            if track.remove(id) {
                return Ok(());
            }
        }
        Err(Error::UnknownSegment(id))
    }

    pub(crate) fn drop_empty(&mut self) {
        self.tracks
            .retain(|track| track.is_main() || !track.segments().is_empty());
    }

    pub(crate) fn track_has_overlap(&self, segment: SegmentId) -> bool {
        self.tracks
            .iter()
            .find(|track| track.segments().iter().any(|item| item.id() == segment))
            .is_some_and(Track::has_overlap)
    }

    /// Packs the main track from zero and returns each segment's start delta.
    pub(crate) fn pack_main(&mut self) -> Result<Vec<(SegmentId, i64)>, Error> {
        let index = self
            .tracks
            .iter()
            .position(|track| track.is_main())
            .expect("a timeline always has a main track");
        let mut cursor = 0_i64;
        let mut plan = Vec::new();
        for segment in self.tracks[index].segments() {
            let old = segment.timeline().start().value;
            let duration = segment.timeline().duration().value;
            let delta = cursor.checked_sub(old).ok_or(Error::DurationOverflow)?;
            plan.push((segment.id(), cursor, delta));
            cursor = cursor
                .checked_add(duration)
                .ok_or(Error::DurationOverflow)?;
        }
        for (id, start, _) in &plan {
            self.tracks[index].set_start(*id, *start)?;
        }
        self.tracks[index].sort_by_start();
        Ok(plan.into_iter().map(|(id, _, delta)| (id, delta)).collect())
    }

    pub(crate) fn assign_starts(&mut self, starts: &[(SegmentId, i64)]) -> Result<(), Error> {
        for track in &mut self.tracks {
            let mut changed = false;
            for (id, start) in starts {
                if track.segment_mut(*id).is_some() {
                    track.set_start(*id, *start)?;
                    changed = true;
                }
            }
            if changed {
                track.sort_by_start();
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::Asset;
    use crate::test_support::{asset_id, segment_id, timeline_id, track_id};
    use crate::track::TrackKind;

    fn still(byte: u64, start: i64, duration: i64) -> Segment {
        let mut asset = Asset::image("file:///still.png", 8, 8).unwrap();
        asset.assign(asset_id(100));
        let mut segment = Segment::image(&asset, start, duration).unwrap();
        segment.assign(segment_id(byte));
        segment
    }

    fn occupied(byte: u64, kind: TrackKind) -> Track {
        Track::occupied(track_id(byte), kind, still(byte, 0, 1_000)).unwrap()
    }

    #[test]
    fn a_new_timeline_has_an_empty_main_track_and_a_zero_duration() {
        let timeline = Timeline::new(timeline_id(1), track_id(2));
        assert_eq!(timeline.tracks().len(), 1);
        assert!(timeline.main_track().is_main());
        assert!(timeline.main_track().segments().is_empty());
        assert_eq!(timeline.duration().value, 0);
        assert_eq!(timeline.duration().timescale.get(), 1_000_000);
    }

    #[test]
    fn duration_is_the_end_of_the_latest_segment() {
        let mut timeline = Timeline::new(timeline_id(1), track_id(2));
        timeline.main_mut().insert(still(3, 0, 1_000_000)).unwrap();
        timeline
            .insert(
                Track::occupied(
                    track_id(4),
                    TrackKind::Video,
                    still(5, 3_000_000, 1_000_000),
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(timeline.duration().value, 4_000_000);
        assert_eq!(timeline.duration().timescale.get(), 1_000_000);
    }

    #[test]
    fn an_overlay_can_sit_above_text_and_audio_stays_under_main() {
        let mut timeline = Timeline::new(timeline_id(1), track_id(2));
        timeline.insert(occupied(3, TrackKind::Text)).unwrap();
        timeline.insert(occupied(4, TrackKind::Video)).unwrap();
        timeline.insert(occupied(5, TrackKind::Effect)).unwrap();
        timeline.insert(occupied(6, TrackKind::Audio)).unwrap();
        let kinds: Vec<_> = timeline.tracks().iter().map(Track::kind).collect();
        assert_eq!(
            kinds,
            vec![
                TrackKind::Audio,
                TrackKind::Video,
                TrackKind::Text,
                TrackKind::Video,
                TrackKind::Effect,
            ]
        );
        assert!(timeline.tracks()[1].is_main());
        assert!(!timeline.tracks()[3].is_main());
    }

    #[test]
    fn a_new_track_sits_above_the_other_tracks_of_its_block() {
        let mut timeline = Timeline::new(timeline_id(1), track_id(2));
        timeline.insert(occupied(3, TrackKind::Audio)).unwrap();
        timeline.insert(occupied(4, TrackKind::Audio)).unwrap();
        timeline.insert(occupied(5, TrackKind::Video)).unwrap();
        timeline.insert(occupied(6, TrackKind::Video)).unwrap();
        let ids: Vec<u64> = timeline
            .tracks()
            .iter()
            .map(|track| track.id().get())
            .collect();
        assert_eq!(ids, vec![3, 4, 2, 5, 6]);
    }

    #[test]
    fn picture_kinds_interleave_above_the_main_track() {
        let mut effect_first = Timeline::new(timeline_id(1), track_id(2));
        effect_first.insert(occupied(3, TrackKind::Effect)).unwrap();
        effect_first.insert(occupied(4, TrackKind::Text)).unwrap();
        let effect_then_text: Vec<_> = effect_first.tracks().iter().map(Track::kind).collect();
        assert_eq!(
            effect_then_text,
            vec![TrackKind::Video, TrackKind::Effect, TrackKind::Text]
        );

        let mut text_first = Timeline::new(timeline_id(1), track_id(2));
        text_first.insert(occupied(3, TrackKind::Text)).unwrap();
        text_first.insert(occupied(4, TrackKind::Effect)).unwrap();
        let text_then_effect: Vec<_> = text_first.tracks().iter().map(Track::kind).collect();
        assert_eq!(
            text_then_effect,
            vec![TrackKind::Video, TrackKind::Text, TrackKind::Effect]
        );
    }

    #[test]
    fn a_second_main_track_is_rejected() {
        let mut timeline = Timeline::new(timeline_id(1), track_id(2));
        let error = timeline.insert(Track::main(track_id(3))).unwrap_err();
        assert_eq!(error, Error::SecondMainTrack);
        assert_eq!(timeline.tracks().len(), 1);
    }

    #[test]
    fn an_empty_non_main_track_is_rejected() {
        let mut timeline = Timeline::new(timeline_id(1), track_id(2));
        let error = timeline
            .insert(Track::vacant(track_id(3), TrackKind::Audio))
            .unwrap_err();
        assert_eq!(error, Error::EmptyTrack);
        assert_eq!(timeline.tracks().len(), 1);
    }

    #[test]
    fn removing_the_last_segment_drops_every_track_but_main() {
        let mut timeline = Timeline::new(timeline_id(1), track_id(2));
        timeline.insert(occupied(3, TrackKind::Audio)).unwrap();
        timeline.main_mut().insert(still(4, 0, 1_000)).unwrap();
        timeline.remove_segment(segment_id(3)).unwrap();
        timeline.remove_segment(segment_id(4)).unwrap();
        timeline.drop_empty();
        assert_eq!(timeline.tracks().len(), 1);
        assert!(timeline.main_track().is_main());
        assert!(timeline.main_track().segments().is_empty());
        assert_eq!(timeline.duration().value, 0);
    }
}

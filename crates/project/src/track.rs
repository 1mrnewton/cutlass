//! A track kind, the main track, and the band order.

use crate::Error;
use crate::id::{SegmentId, TrackId};
use crate::segment::Segment;
use crate::time::TimeRange;
use crate::transition::Transition;

/// One kind per track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackKind {
    Video,
    Audio,
    Adjust,
    Filter,
    Effect,
    Text,
    Sticker,
    PluginEffect,
}

/// Row order, bottom to top. Audio is under the main track. Every other
/// track, including an overlay, shares the block above main.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Band {
    Audio,
    Main,
    Above,
}

/// Segments on one lane. They do not overlap.
#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    id: TrackId,
    kind: TrackKind,
    main: bool,
    locked: bool,
    hidden: bool,
    muted: bool,
    segments: Vec<Segment>,
    transitions: Vec<Transition>,
}

impl Track {
    pub(crate) fn main(id: TrackId) -> Self {
        Self {
            id,
            kind: TrackKind::Video,
            main: true,
            locked: false,
            hidden: false,
            muted: false,
            segments: Vec::new(),
            transitions: Vec::new(),
        }
    }

    pub(crate) fn occupied(id: TrackId, kind: TrackKind, segment: Segment) -> Result<Self, Error> {
        let mut track = Self {
            id,
            kind,
            main: false,
            locked: false,
            hidden: false,
            muted: false,
            segments: Vec::new(),
            transitions: Vec::new(),
        };
        track.insert(segment)?;
        Ok(track)
    }

    #[cfg(test)]
    pub(crate) fn vacant(id: TrackId, kind: TrackKind) -> Self {
        Self {
            id,
            kind,
            main: false,
            locked: false,
            hidden: false,
            muted: false,
            segments: Vec::new(),
            transitions: Vec::new(),
        }
    }

    pub(crate) fn band(&self) -> Band {
        match (self.kind, self.main) {
            (TrackKind::Audio, _) => Band::Audio,
            (TrackKind::Video, true) => Band::Main,
            (
                TrackKind::Video
                | TrackKind::Adjust
                | TrackKind::Filter
                | TrackKind::Effect
                | TrackKind::Text
                | TrackKind::Sticker
                | TrackKind::PluginEffect,
                _,
            ) => Band::Above,
        }
    }

    pub(crate) fn insert(&mut self, segment: Segment) -> Result<(), Error> {
        if self
            .segments
            .iter()
            .any(|existing| existing.id() == segment.id())
        {
            return Err(Error::DuplicateSegment(segment.id()));
        }
        if self.overlaps_range(segment.timeline()) {
            return Err(Error::Overlap);
        }
        let start = segment.timeline().start().value;
        let index = self
            .segments
            .iter()
            .position(|existing| existing.timeline().start().value > start)
            .unwrap_or(self.segments.len());
        self.segments.insert(index, segment);
        Ok(())
    }

    pub(crate) fn insert_at(&mut self, index: usize, segment: Segment) {
        self.segments.insert(index, segment);
    }

    pub(crate) fn remove(&mut self, id: SegmentId) -> bool {
        self.take(id).is_some()
    }

    pub(crate) fn take(&mut self, id: SegmentId) -> Option<Segment> {
        let index = self
            .segments
            .iter()
            .position(|segment| segment.id() == id)?;
        Some(self.segments.remove(index))
    }

    pub(crate) fn set_start(&mut self, id: SegmentId, start: i64) -> Result<(), Error> {
        let segment = self
            .segments
            .iter_mut()
            .find(|segment| segment.id() == id)
            .ok_or(Error::UnknownSegment(id))?;
        segment.set_start(start)
    }

    pub(crate) fn sort_by_start(&mut self) {
        self.segments
            .sort_by_key(|segment| segment.timeline().start().value);
    }

    pub(crate) fn segment_mut(&mut self, id: SegmentId) -> Option<&mut Segment> {
        self.segments.iter_mut().find(|segment| segment.id() == id)
    }

    pub(crate) fn overlaps_range(&self, range: TimeRange) -> bool {
        self.segments
            .iter()
            .any(|segment| segment.timeline().overlaps(range))
    }

    pub(crate) fn has_overlap(&self) -> bool {
        let mut order: Vec<&Segment> = self.segments.iter().collect();
        order.sort_by_key(|segment| segment.timeline().start().value);
        order
            .windows(2)
            .any(|pair| pair[1].timeline().start().value < pair[0].timeline().end().value)
    }

    pub(crate) fn ranges(&self) -> Vec<(SegmentId, TimeRange)> {
        self.segments
            .iter()
            .map(|segment| (segment.id(), segment.timeline()))
            .collect()
    }

    pub const fn id(&self) -> TrackId {
        self.id
    }

    pub const fn kind(&self) -> TrackKind {
        self.kind
    }

    pub const fn is_main(&self) -> bool {
        self.main
    }

    /// A locked track refuses an edit.
    pub const fn locked(&self) -> bool {
        self.locked
    }

    /// The preview skips a hidden track.
    pub const fn hidden(&self) -> bool {
        self.hidden
    }

    /// Stored for the mix. Nothing plays it yet.
    pub const fn muted(&self) -> bool {
        self.muted
    }

    pub(crate) fn set_locked(&mut self, on: bool) {
        self.locked = on;
    }

    pub(crate) fn set_hidden(&mut self, on: bool) {
        self.hidden = on;
    }

    pub(crate) fn set_muted(&mut self, on: bool) {
        self.muted = on;
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// Transitions on this track. Only the main track holds any.
    pub fn transitions(&self) -> &[Transition] {
        &self.transitions
    }

    pub(crate) fn set_transitions(&mut self, transitions: Vec<Transition>) {
        self.transitions = transitions;
    }

    pub(crate) fn upsert_transition(&mut self, transition: Transition) {
        if let Some(found) = self
            .transitions
            .iter_mut()
            .find(|existing| existing.left() == transition.left())
        {
            *found = transition;
            return;
        }
        self.transitions.push(transition);
    }

    pub(crate) fn clear_transition(&mut self, left: SegmentId) {
        self.transitions
            .retain(|transition| transition.left() != left);
    }

    pub(crate) fn retarget_left(&mut self, from: SegmentId, to: SegmentId) {
        for transition in &mut self.transitions {
            if transition.left() == from {
                transition.retarget_left(to);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::Asset;
    use crate::test_support::{asset_id, segment_id, track_id};

    fn still(byte: u64, start: i64, duration: i64) -> Segment {
        let mut asset = Asset::image("file:///still.png", 8, 8).unwrap();
        asset.assign(asset_id(100));
        let mut segment = Segment::image(&asset, start, duration).unwrap();
        segment.assign(segment_id(byte));
        segment
    }

    #[test]
    fn the_main_track_is_a_video_track() {
        let track = Track::main(track_id(1));
        assert!(track.is_main());
        assert_eq!(track.kind(), TrackKind::Video);
        assert!(track.segments().is_empty());
        assert_eq!(track.band(), Band::Main);
    }

    #[test]
    fn an_audio_track_is_not_main() {
        let track = Track::occupied(track_id(1), TrackKind::Audio, still(2, 0, 10)).unwrap();
        assert!(!track.is_main());
        assert_eq!(track.band(), Band::Audio);
    }

    #[test]
    fn overlapping_segments_are_rejected_and_the_track_stays() {
        let mut track = Track::occupied(track_id(1), TrackKind::Video, still(2, 0, 100)).unwrap();
        let error = track.insert(still(3, 50, 100)).unwrap_err();
        assert_eq!(error, Error::Overlap);
        assert_eq!(track.segments().len(), 1);
        let inside = track.insert(still(4, 10, 10)).unwrap_err();
        assert_eq!(inside, Error::Overlap);
    }

    #[test]
    fn touching_endpoints_stay_on_one_track_in_start_order() {
        let mut track = Track::occupied(track_id(1), TrackKind::Video, still(2, 100, 50)).unwrap();
        track.insert(still(3, 0, 100)).unwrap();
        track.insert(still(4, 150, 25)).unwrap();
        let starts: Vec<i64> = track
            .segments()
            .iter()
            .map(|segment| segment.timeline().start().value)
            .collect();
        assert_eq!(starts, vec![0, 100, 150]);
    }

    #[test]
    fn a_duplicate_segment_id_is_rejected() {
        let mut track = Track::occupied(track_id(1), TrackKind::Video, still(2, 0, 10)).unwrap();
        let error = track.insert(still(2, 20, 10)).unwrap_err();
        assert_eq!(error, Error::DuplicateSegment(segment_id(2)));
    }
}

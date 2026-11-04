//! Reorder a clip on a packed row by its place in the order.
//!
//! A packed row has no gaps, so a drop names a place, not a time. The place
//! becomes the start a move request takes: the start of the clip that will
//! follow it, or the end of the row for the last place. A move to that start
//! on a packed row goes before the clip already there.

use cutlass_model::{Micros, SegmentId, Track};

/// The start that puts `id` at `place` among the other clips of `track`,
/// counted from zero in start order. A place past the end is the end. An
/// unknown id has no start.
#[must_use]
pub fn reorder_start(track: &Track, id: &SegmentId, place: usize) -> Option<Micros> {
    let mut others: Vec<_> = track
        .segments
        .iter()
        .filter(|segment| &segment.id != id)
        .collect();
    if others.len() == track.segments.len() {
        return None;
    }
    others.sort_by_key(|segment| segment.target.start.0);
    match others.get(place) {
        Some(next) => Some(next.target.start),
        None => Some(others.last().map_or(Micros(0), |last| {
            Micros(last.target.start.0.saturating_add(last.target.duration.0))
        })),
    }
}

/// The place of `id` among the clips of `track` in start order.
#[must_use]
pub fn place_of(track: &Track, id: &SegmentId) -> Option<usize> {
    let mut ordered: Vec<_> = track.segments.iter().collect();
    ordered.sort_by_key(|segment| segment.target.start.0);
    ordered.iter().position(|segment| &segment.id == id)
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Segment, TimeRange, Track};

    use super::*;

    fn row() -> Track {
        let mut track = cutlass_model::Draft::empty("d").tracks[0].clone();
        track.segments = vec![
            Segment::new("a", "m", TimeRange::new(0, 1_000)),
            Segment::new("b", "m", TimeRange::new(1_000, 2_000)),
            Segment::new("c", "m", TimeRange::new(3_000, 500)),
        ];
        track
    }

    #[test]
    fn a_place_is_the_start_of_the_clip_that_follows() {
        let track = row();
        let a = SegmentId::new("a");
        assert_eq!(reorder_start(&track, &a, 0), Some(Micros(1_000)));
        assert_eq!(reorder_start(&track, &a, 1), Some(Micros(3_000)));
        assert_eq!(reorder_start(&track, &a, 2), Some(Micros(3_500)));
        assert_eq!(reorder_start(&track, &a, 9), Some(Micros(3_500)));
        assert_eq!(
            reorder_start(&track, &SegmentId::new("c"), 0),
            Some(Micros(0))
        );
    }

    #[test]
    fn an_unknown_clip_has_no_place() {
        let track = row();
        let missing = SegmentId::new("z");
        assert_eq!(reorder_start(&track, &missing, 0), None);
        assert_eq!(place_of(&track, &missing), None);
        assert_eq!(place_of(&track, &SegmentId::new("c")), Some(2));
    }

    #[test]
    fn the_only_clip_goes_to_zero() {
        let mut track = row();
        track.segments.truncate(1);
        assert_eq!(
            reorder_start(&track, &SegmentId::new("a"), 3),
            Some(Micros(0))
        );
    }
}

//! Placing a library file on the timeline: the requests for one clip that
//! covers the whole file, a lane it opens, and the main-track clip it
//! follows when linkage is on.

use cutlass_model::{
    Draft, Micros, Segment, SegmentId, TimeRange, TrackAttribute, TrackId, TrackType,
};
use cutlass_session::Request;

/// Where a library file goes and what its new ids are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaPlace {
    pub track_id: TrackId,
    pub segment_id: SegmentId,
    pub material_id: String,
    pub start: Micros,
    /// The whole file. The source range is `0..duration`.
    pub duration: Micros,
    /// Open a lane of this kind as `track_id` first.
    pub new_lane: Option<TrackType>,
}

/// The requests that place a library file, as one undo step.
///
/// With `linkage` on, a clip placed off the main track over exactly one
/// main-track clip is linked to it, so it follows that clip.
#[must_use]
pub fn media_requests(draft: &Draft, place: &MediaPlace, linkage: bool) -> Vec<Request> {
    let mut requests = Vec::new();
    if let Some(kind) = &place.new_lane {
        requests.push(Request::InsertTrack {
            track_id: place.track_id.clone(),
            kind: kind.clone(),
        });
    }
    let mut segment = Segment::new(
        place.segment_id.clone(),
        place.material_id.as_str(),
        TimeRange::new(place.start.0, place.duration.0),
    );
    segment.source_range = Some(TimeRange::new(0, place.duration.0));
    requests.push(Request::Place {
        track_id: place.track_id.clone(),
        segment: Box::new(segment),
    });
    let on_main = draft
        .tracks
        .iter()
        .any(|track| track.id == place.track_id && track.is_main());
    if linkage
        && !on_main
        && let Some(owner) = exclusive_main_owner(draft, place.start, place.duration)
    {
        requests.push(Request::Link {
            owner,
            dependent: place.segment_id.clone(),
        });
    }
    requests
}

/// The one main-track clip the span sits on, when it sits on no other.
/// A locked main lane is not an owner. Touching at an endpoint is not an overlap.
fn exclusive_main_owner(draft: &Draft, start: Micros, duration: Micros) -> Option<SegmentId> {
    let end = start.0.checked_add(duration.0)?;
    if duration.0 <= 0 {
        return None;
    }
    let mut owner = None;
    for track in &draft.tracks {
        if !track.is_main() || track.attribute.contains(TrackAttribute::LOCK) {
            continue;
        }
        for segment in &track.segments {
            let segment_end = segment
                .target
                .start
                .0
                .checked_add(segment.target.duration.0)?;
            if !(start.0 < segment_end && segment.target.start.0 < end) {
                continue;
            }
            if owner.is_some() {
                return None;
            }
            owner = Some(segment.id.clone());
        }
    }
    owner
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft() -> Draft {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        for (id, start) in [("a", 0), ("b", 1_000_000)] {
            draft.tracks[0]
                .segments
                .push(Segment::new(id, "m", TimeRange::new(start, 1_000_000)));
        }
        draft
    }

    fn place(track: &str, start: i64, duration: i64) -> MediaPlace {
        MediaPlace {
            track_id: TrackId::new(track),
            segment_id: SegmentId::new("new"),
            material_id: "file".to_owned(),
            start: Micros(start),
            duration: Micros(duration),
            new_lane: Some(TrackType::Audio),
        }
    }

    fn link_owner(requests: &[Request]) -> Option<&SegmentId> {
        requests.iter().find_map(|request| match request {
            Request::Link { owner, .. } => Some(owner),
            _ => None,
        })
    }

    #[test]
    fn a_new_lane_comes_first_and_the_clip_covers_the_whole_file() {
        let requests = media_requests(&draft(), &place("t", 0, 500_000), false);
        assert!(matches!(requests[0], Request::InsertTrack { .. }));
        let Request::Place { segment, .. } = &requests[1] else {
            panic!("the clip is placed");
        };
        assert_eq!(segment.target, TimeRange::new(0, 500_000));
        assert_eq!(segment.source_range, Some(TimeRange::new(0, 500_000)));
        assert_eq!(requests.len(), 2);
    }

    #[test]
    fn a_clip_over_one_main_clip_follows_it() {
        let requests = media_requests(&draft(), &place("t", 200_000, 500_000), true);
        assert_eq!(link_owner(&requests), Some(&SegmentId::new("a")));
    }

    #[test]
    fn two_main_clips_a_main_lane_or_linkage_off_link_nothing() {
        let draft = draft();
        assert_eq!(
            link_owner(&media_requests(&draft, &place("t", 800_000, 500_000), true)),
            None
        );
        assert_eq!(
            link_owner(&media_requests(&draft, &place("main", 0, 500_000), true)),
            None
        );
        assert_eq!(
            link_owner(&media_requests(&draft, &place("t", 0, 500_000), false)),
            None
        );
        // Touching b's start is not sitting on it.
        assert_eq!(
            link_owner(&media_requests(&draft, &place("t", 500_000, 500_000), true)),
            Some(&SegmentId::new("a"))
        );
    }

    #[test]
    fn a_locked_main_lane_owns_nothing() {
        let mut draft = draft();
        draft.tracks[0].attribute = TrackAttribute(TrackAttribute::LOCK);
        assert_eq!(
            link_owner(&media_requests(&draft, &place("t", 0, 500_000), true)),
            None
        );
    }
}

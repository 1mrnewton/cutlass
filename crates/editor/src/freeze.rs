//! Freeze: a still of one frame put into the timeline at the playhead.
//!
//! The clip is split at the playhead and the still goes between the two
//! pieces, so the clip keeps all of its frames and the timeline grows by the
//! still's length. At a clip's edge the still goes before or after it
//! instead, with no split. Which frame and where come from the evaluator.

use cutlass_model::{Draft, FreezeInfo, MaterialRef, Micros, SegmentId, TimeRange, TrackAttribute};
use cutlass_session::Request;

/// The requests that put a still of `freeze` at `at` for `duration`, as one
/// undo step. `None` when the clip is not on the timeline, `at` is not on
/// it, or `duration` is not positive.
///
/// Later clips on the clip's row move right by `duration`, taking what is
/// linked to them; a packed main row makes that room by itself. The still
/// keeps the clip's look (position, scale, crop, filters) but plays at
/// normal speed with no transition, animation, fade or keyframes. Its
/// video is its own copy, so the clip it came from keeps playing.
/// `fresh` makes an unused id from a prefix.
pub fn freeze_requests(
    draft: &Draft,
    segment_id: &SegmentId,
    at: Micros,
    freeze: FreezeInfo,
    duration: Micros,
    fresh: &mut dyn FnMut(&str) -> String,
) -> Option<Vec<Request>> {
    if duration.0 <= 0 {
        return None;
    }
    let track = draft.tracks.iter().find(|track| {
        track
            .segments
            .iter()
            .any(|segment| &segment.id == segment_id)
    })?;
    let segment = track
        .segments
        .iter()
        .find(|segment| &segment.id == segment_id)?;
    let start = segment.target.start.0;
    let end = segment.target.end().0;
    if at.0 < start || at.0 > end {
        return None;
    }
    let mut requests = Vec::new();
    let split = start < at.0 && at.0 < end;
    let right_id = split.then(|| SegmentId::new(fresh("clip")));
    if let Some(right_id) = &right_id {
        requests.push(Request::Split {
            segment_id: segment_id.clone(),
            at,
            right_id: right_id.clone(),
        });
    }
    let packed = draft.config.maintrack_adsorb
        && track.is_main()
        && !track.attribute.contains(TrackAttribute::LOCK);
    if !packed {
        // Last first, so no clip moves onto one that has not moved yet.
        let mut later: Vec<(SegmentId, i64)> = track
            .segments
            .iter()
            .filter(|other| other.target.start.0 >= at.0 && &other.id != segment_id)
            .map(|other| (other.id.clone(), other.target.start.0))
            .collect();
        if at.0 == start {
            later.push((segment_id.clone(), start));
        }
        if let Some(right_id) = &right_id {
            later.push((right_id.clone(), at.0));
        }
        later.sort_by_key(|(_, start)| std::cmp::Reverse(*start));
        for (id, start) in later {
            requests.push(Request::Move {
                segment_id: id,
                track_id: track.id.clone(),
                start: Micros(start.checked_add(duration.0)?),
            });
        }
    }
    let still_id = SegmentId::new(fresh("freeze"));
    let mut still = segment.clone();
    still.id = still_id.clone();
    still.target = TimeRange::new(at.0, duration.0);
    // The held frame must lie in the still's source range, or the frame
    // lookup refuses it and the still draws nothing.
    still.source_range = Some(TimeRange::new(freeze.timestamp.0, duration.0));
    still.speed = 1.0;
    still.reverse = false;
    still.group_id.clear();
    still.common_keyframes.clear();
    still.keyframe_refs.clear();
    still.extra_material_refs.retain(|id| {
        !matches!(
            draft.materials.get(id.as_str()),
            Some(
                MaterialRef::Speed(_)
                    | MaterialRef::Fade(_)
                    | MaterialRef::Transition(_)
                    | MaterialRef::Animations(_)
            )
        )
    });
    requests.push(Request::Place {
        track_id: track.id.clone(),
        segment: Box::new(still),
    });
    requests.push(Request::SetFreeze {
        segment_id: still_id,
        freeze: Some(freeze),
    });
    Some(requests)
}

#[cfg(test)]
mod tests {
    use cutlass_model::{MaterialSpeed, MaterialVideo, Segment, TrackId};
    use cutlass_session::Session;

    use super::*;

    /// Main clips "a" 0..2 s and "b" 2..3 s on video "v", magnet as given.
    fn session(magnet: bool) -> Session {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        draft.config.maintrack_adsorb = magnet;
        draft.materials.videos.push(MaterialVideo {
            id: "v".into(),
            path: "/clip.mov".to_owned(),
            duration: Micros(10_000_000),
            ..MaterialVideo::default()
        });
        draft.materials.speeds.push(MaterialSpeed {
            id: "s".into(),
            speed: 1.0,
            ..MaterialSpeed::default()
        });
        for (id, start, duration) in [("a", 0, 2_000_000), ("b", 2_000_000, 1_000_000)] {
            let mut clip = Segment::new(id, "v", TimeRange::new(start, duration));
            clip.source_range = Some(TimeRange::new(0, duration));
            clip.extra_material_refs = vec!["s".into()];
            draft.tracks[0].segments.push(clip);
        }
        Session::open(draft)
    }

    fn held(timestamp: i64) -> FreezeInfo {
        FreezeInfo {
            source_material_id: "v".to_owned(),
            source_material_path: "/clip.mov".to_owned(),
            timestamp: Micros(timestamp),
            ..FreezeInfo::default()
        }
    }

    fn ids() -> impl FnMut(&str) -> String {
        let mut n = 0;
        move |prefix: &str| {
            n += 1;
            format!("{prefix}{n}")
        }
    }

    /// Each clip on the main row as (id, start, duration, held).
    fn row(session: &Session) -> Vec<(String, i64, i64, bool)> {
        let draft = session.draft();
        draft.tracks[0]
            .segments
            .iter()
            .map(|segment| {
                let held = matches!(
                    draft.materials.get(segment.material_id.as_str()),
                    Some(MaterialRef::Video(video)) if video.freeze.is_some()
                );
                (
                    segment.id.as_str().to_owned(),
                    segment.target.start.0,
                    segment.target.duration.0,
                    held,
                )
            })
            .collect()
    }

    #[test]
    fn a_still_splits_the_clip_and_pushes_later_clips_right() {
        for magnet in [false, true] {
            let mut session = session(magnet);
            let requests = freeze_requests(
                session.draft(),
                &"a".into(),
                Micros(500_000),
                held(500_000),
                Micros(3_000_000),
                &mut ids(),
            )
            .unwrap();
            session.apply(0, &requests).unwrap();
            assert_eq!(
                row(&session),
                vec![
                    ("a".to_owned(), 0, 500_000, false),
                    ("freeze2".to_owned(), 500_000, 3_000_000, true),
                    ("clip1".to_owned(), 3_500_000, 1_500_000, false),
                    ("b".to_owned(), 5_000_000, 1_000_000, false),
                ],
                "magnet {magnet}"
            );
            let still = &session.draft().tracks[0].segments[1];
            assert!(still.extra_material_refs.is_empty());
            assert_eq!(still.source_range, Some(TimeRange::new(500_000, 3_000_000)));
            assert_eq!(still.speed, 1.0);
            session.undo().unwrap();
            assert_eq!(session.draft().tracks[0].segments.len(), 2);
        }
    }

    #[test]
    fn at_an_edge_the_still_goes_beside_the_clip_unsplit() {
        let mut session = session(false);
        let requests = freeze_requests(
            session.draft(),
            &"a".into(),
            Micros(2_000_000),
            held(1_999_999),
            Micros(3_000_000),
            &mut ids(),
        )
        .unwrap();
        session.apply(0, &requests).unwrap();
        assert_eq!(
            row(&session),
            vec![
                ("a".to_owned(), 0, 2_000_000, false),
                ("freeze1".to_owned(), 2_000_000, 3_000_000, true),
                ("b".to_owned(), 5_000_000, 1_000_000, false),
            ]
        );

        let mut session = self::session(false);
        let requests = freeze_requests(
            session.draft(),
            &"a".into(),
            Micros(0),
            held(0),
            Micros(1_000_000),
            &mut ids(),
        )
        .unwrap();
        session.apply(0, &requests).unwrap();
        assert_eq!(
            row(&session),
            vec![
                ("freeze1".to_owned(), 0, 1_000_000, true),
                ("a".to_owned(), 1_000_000, 2_000_000, false),
                ("b".to_owned(), 3_000_000, 1_000_000, false),
            ]
        );
    }

    #[test]
    fn a_time_off_the_clip_or_no_length_makes_no_requests() {
        let session = session(false);
        let draft = session.draft();
        for (at, duration) in [(2_500_000, 3_000_000), (500_000, 0)] {
            assert!(
                freeze_requests(
                    draft,
                    &"a".into(),
                    Micros(at),
                    held(0),
                    Micros(duration),
                    &mut ids()
                )
                .is_none()
            );
        }
        assert!(
            freeze_requests(
                draft,
                &"z".into(),
                Micros(0),
                held(0),
                Micros(1),
                &mut ids()
            )
            .is_none()
        );
    }
}

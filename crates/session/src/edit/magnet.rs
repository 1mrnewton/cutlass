use cutlass_model::{Draft, LinkageSettings, Micros, Segment, SegmentId, Track, TrackAttribute};

use super::{EditError, ensure_unlocked};

/// The main lane packs from zero when the magnet is on and the lane is unlocked.
#[must_use]
pub(super) fn packs(adsorb: bool, track: &Track) -> bool {
    adsorb && track.is_main() && !track.attribute.contains(TrackAttribute::LOCK)
}

/// Where a segment is inserted. On a packed lane, the same start goes before
/// the clip already there, so a drop on that start takes its place in the order.
#[must_use]
pub(super) fn insert_at(segments: &[Segment], start: i64, packed: bool) -> usize {
    segments
        .iter()
        .position(|existing| {
            if packed {
                existing.target.start.0 >= start
            } else {
                existing.target.start.0 > start
            }
        })
        .unwrap_or(segments.len())
}

/// Store the switch. A locked main lane refuses either direction.
pub fn set_magnet(draft: &mut Draft, on: bool) -> Result<(), EditError> {
    if let Some(track) = draft.tracks.iter().find(|track| track.is_main()) {
        ensure_unlocked(track.attribute)?;
    }
    draft.config.maintrack_adsorb = on;
    Ok(())
}

/// Close gaps on the main lane. The order is the current start, and a tie
/// keeps the order already in the lane. Followers of a clip that moved
/// take the same extra shift.
pub(super) fn pack(draft: &mut Draft, linkage: &LinkageSettings) -> Result<(), EditError> {
    if !draft.config.maintrack_adsorb {
        return Ok(());
    }
    let Some(index) = draft.tracks.iter().position(|track| track.is_main()) else {
        return Ok(());
    };
    if draft.tracks[index].attribute.contains(TrackAttribute::LOCK) {
        return Ok(());
    }
    let before: Vec<(SegmentId, i64)> = draft.tracks[index]
        .segments
        .iter()
        .map(|segment| (segment.id.clone(), segment.target.start.0))
        .collect();
    let mut ordered: Vec<Segment> = draft.tracks[index].segments.clone();
    ordered.sort_by_key(|segment| segment.target.start.0);
    let mut cursor = 0_i64;
    for segment in &mut ordered {
        segment.target.start = Micros(cursor);
        cursor = cursor
            .checked_add(segment.target.duration.0)
            .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
    }
    draft.tracks[index].segments = ordered;
    let shifts: Vec<(SegmentId, i64)> = draft.tracks[index]
        .segments
        .iter()
        .map(|segment| (segment.id.clone(), segment.target.start.0))
        .collect();
    for (id, start) in shifts {
        let Some((_, old)) = before.iter().find(|(stored, _)| stored == &id) else {
            continue;
        };
        let delta = start
            .checked_sub(*old)
            .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
        super::linkage::follow_pack_shift(draft, &id, delta, linkage)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, Micros, Segment, SegmentId, TimeRange, TrackAttribute, TrackId, TrackType,
    };

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    fn place(id: &str, start: i64, duration: i64) -> Request {
        Request::Place {
            track_id: TrackId::new(""),
            segment: Box::new(Segment::new(id, "media", TimeRange::new(start, duration))),
        }
    }

    #[test]
    fn the_main_lane_packs_from_zero_and_another_lane_keeps_a_gap() {
        let mut session = Session::open(Draft::empty("D"));
        session.apply(0, &[place("a", 500_000, 100_000)]).unwrap();
        assert_eq!(
            session.draft().tracks[0].segments[0].target,
            TimeRange::new(0, 100_000)
        );
        session.apply(1, &[place("b", 800_000, 100_000)]).unwrap();
        let main = &session.draft().tracks[0].segments;
        assert_eq!(main[0].id, SegmentId::new("a"));
        assert_eq!(main[0].target, TimeRange::new(0, 100_000));
        assert_eq!(main[1].id, SegmentId::new("b"));
        assert_eq!(main[1].target, TimeRange::new(100_000, 100_000));

        session
            .apply(2, &[Request::SetMagnet { on: false }])
            .unwrap();
        session.apply(3, &[place("c", 900_000, 50_000)]).unwrap();
        let main = &session.draft().tracks[0].segments;
        assert_eq!(main[2].target.start, Micros(900_000));

        session
            .apply(
                4,
                &[Request::InsertTrack {
                    track_id: TrackId::new("audio"),
                    kind: TrackType::Audio,
                }],
            )
            .unwrap();
        session
            .apply(5, &[Request::SetMagnet { on: true }])
            .unwrap();
        assert_eq!(
            session.draft().tracks[0].segments[2].target.start,
            Micros(200_000)
        );
        session
            .apply(
                6,
                &[Request::Place {
                    track_id: TrackId::new("audio"),
                    segment: Box::new(Segment::new(
                        "tone",
                        "tone",
                        TimeRange::new(400_000, 50_000),
                    )),
                }],
            )
            .unwrap();
        let audio = session
            .draft()
            .tracks
            .iter()
            .find(|track| track.id == TrackId::new("audio"))
            .unwrap();
        assert_eq!(audio.segments[0].target.start, Micros(400_000));
    }

    #[test]
    fn a_move_onto_a_main_clip_reorders_and_a_delete_closes_the_gap() {
        let mut session = Session::open(Draft::empty("D"));
        session.apply(0, &[place("a", 0, 100_000)]).unwrap();
        session.apply(1, &[place("b", 100_000, 100_000)]).unwrap();
        session
            .apply(
                2,
                &[Request::Move {
                    segment_id: SegmentId::new("b"),
                    track_id: TrackId::new(""),
                    start: Micros(0),
                }],
            )
            .unwrap();
        let main = &session.draft().tracks[0].segments;
        assert_eq!(main[0].id, SegmentId::new("b"));
        assert_eq!(main[0].target.start, Micros(0));
        assert_eq!(main[1].id, SegmentId::new("a"));
        assert_eq!(main[1].target.start, Micros(100_000));

        session
            .apply(
                3,
                &[Request::Remove {
                    segment_id: SegmentId::new("b"),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().tracks[0].segments.len(), 1);
        assert_eq!(
            session.draft().tracks[0].segments[0].target,
            TimeRange::new(0, 100_000)
        );
    }

    #[test]
    fn a_locked_main_lane_refuses_the_magnet_switch() {
        let mut session = Session::open(Draft::empty("D"));
        session
            .apply(
                0,
                &[Request::SetTrackAttribute {
                    track_id: TrackId::new(""),
                    bit: TrackAttribute::LOCK,
                    on: true,
                }],
            )
            .unwrap();
        let error = session
            .apply(1, &[Request::SetMagnet { on: false }])
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert!(session.draft().config.maintrack_adsorb);
    }

    #[test]
    fn a_linked_clip_follows_the_pack() {
        let mut session = Session::open(Draft::empty("D"));
        session.apply(0, &[place("main", 0, 200_000)]).unwrap();
        session
            .apply(
                1,
                &[Request::InsertTrack {
                    track_id: TrackId::new("audio"),
                    kind: TrackType::Audio,
                }],
            )
            .unwrap();
        session
            .apply(
                2,
                &[Request::Place {
                    track_id: TrackId::new("audio"),
                    segment: Box::new(Segment::new("tone", "tone", TimeRange::new(0, 200_000))),
                }],
            )
            .unwrap();
        session
            .apply(
                3,
                &[Request::Link {
                    owner: SegmentId::new("main"),
                    dependent: SegmentId::new("tone"),
                }],
            )
            .unwrap();
        session
            .apply(4, &[Request::SetMagnet { on: false }])
            .unwrap();
        session
            .apply(
                5,
                &[Request::Move {
                    segment_id: SegmentId::new("main"),
                    track_id: TrackId::new(""),
                    start: Micros(300_000),
                }],
            )
            .unwrap();
        session
            .apply(6, &[Request::SetMagnet { on: true }])
            .unwrap();
        assert_eq!(
            session.draft().tracks[0].segments[0].target.start,
            Micros(0)
        );
        let tone = session
            .draft()
            .tracks
            .iter()
            .find(|track| track.kind == TrackType::Audio)
            .unwrap()
            .segments[0]
            .target
            .start;
        assert_eq!(tone, Micros(0));
    }
}

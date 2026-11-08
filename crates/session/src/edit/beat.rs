use cutlass_model::{
    Draft, MaterialId, Materials, Micros, Segment, SegmentId, beat_timeline_time, new_beats,
    segment_beats, set_user_beats, track_holds_beats, user_beats,
};

use super::locate;
use super::{EditError, ensure_unlocked};

/// Store a beat on an audio segment. `source` is the source time.
pub fn add_beat(
    draft: &mut Draft,
    segment_id: &SegmentId,
    source: Micros,
    material_id: &MaterialId,
) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    if !track_holds_beats(&draft.tracks[track_index].kind) {
        return Err(EditError::NotAudio);
    }
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let segment = &draft.tracks[track_index].segments[segment_index];
    if beat_timeline_time(segment, source.0).is_none() {
        return Err(EditError::Range(
            cutlass_model::RangeError::NonPositiveDuration,
        ));
    }
    if let Some(existing) = beat_id(&draft.materials, segment) {
        return push_beat(draft, &existing, source.0);
    }
    if material_id.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    if draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    draft
        .materials
        .beats
        .push(new_beats(material_id.clone(), source.0));
    draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .push(material_id.clone());
    Ok(())
}

/// Remove one stored beat. `source` is the source time.
pub fn remove_beat(
    draft: &mut Draft,
    segment_id: &SegmentId,
    source: Micros,
) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let Some(material_id) = beat_id(
        &draft.materials,
        &draft.tracks[track_index].segments[segment_index],
    ) else {
        return Err(EditError::MissingMaterial);
    };
    remove_source(draft, &material_id, source.0)?;
    let empty = draft
        .materials
        .beats
        .iter()
        .find(|item| item.id == material_id)
        .is_none_or(|item| user_beats(item).is_empty());
    if empty {
        draft.materials.beats.retain(|item| item.id != material_id);
        draft.tracks[track_index].segments[segment_index]
            .extra_material_refs
            .retain(|id| id != &material_id);
    }
    Ok(())
}

/// Give the right piece the beats that fall on its side of the cut.
///
/// A source range keeps the stored source time. Without one, a beat is an
/// offset from zero, and the right piece measures it from its new start.
/// A material shared with another clip stays on the left piece.
pub(super) fn split_beats(
    draft: &mut Draft,
    left: &mut Segment,
    right: &mut Segment,
    right_id: &SegmentId,
) -> Result<(), EditError> {
    let Some(material_id) = beat_id(&draft.materials, left) else {
        return Ok(());
    };
    if ref_count(draft, &material_id) != 1 {
        right.extra_material_refs.retain(|id| id != &material_id);
        return Ok(());
    }
    let Some(index) = draft
        .materials
        .beats
        .iter()
        .position(|item| item.id == material_id)
    else {
        left.extra_material_refs.retain(|id| id != &material_id);
        right.extra_material_refs.retain(|id| id != &material_id);
        return Ok(());
    };
    let beats = user_beats(&draft.materials.beats[index]);
    let mut stay = Vec::new();
    let mut moved = Vec::new();
    let cut = right
        .source_range
        .as_ref()
        .map(|range| range.start.0)
        .filter(|_| left.source_range.is_some());
    for beat in beats {
        if let Some(cut) = cut {
            if beat < cut {
                stay.push(beat);
            } else {
                moved.push(beat);
            }
            continue;
        }
        if beat < left.target.duration.0 {
            stay.push(beat);
            continue;
        }
        match beat.checked_sub(left.target.duration.0) {
            Some(next) => moved.push(next),
            None => stay.push(beat),
        }
    }
    right.extra_material_refs.retain(|id| id != &material_id);
    if stay.is_empty() {
        draft.materials.beats.remove(index);
        left.extra_material_refs.retain(|id| id != &material_id);
    } else {
        set_user_beats(&mut draft.materials.beats[index], &stay);
    }
    if moved.is_empty() {
        return Ok(());
    }
    let id = MaterialId::new(format!("{}:beats", right_id.as_str()));
    if draft.materials.get(id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    draft.materials.beats.push(new_beats(id.clone(), moved[0]));
    let slot = draft
        .materials
        .beats
        .last_mut()
        .ok_or(EditError::MissingMaterial)?;
    set_user_beats(slot, &moved);
    right.extra_material_refs.push(id);
    Ok(())
}

/// Drop beats materials that belonged to a clip which is now gone.
pub(super) fn drop_unreferenced_beats(draft: &mut Draft, ids: &[MaterialId]) {
    for id in ids {
        if ref_count(draft, id) == 0 {
            draft.materials.beats.retain(|item| &item.id != id);
        }
    }
}

pub(super) fn beat_ids(materials: &Materials, segment: &Segment) -> Vec<MaterialId> {
    segment
        .extra_material_refs
        .iter()
        .filter(|id| materials.beats.iter().any(|item| &item.id == *id))
        .cloned()
        .collect()
}

fn push_beat(draft: &mut Draft, material_id: &MaterialId, source: i64) -> Result<(), EditError> {
    let Some(slot) = draft
        .materials
        .beats
        .iter_mut()
        .find(|item| &item.id == material_id)
    else {
        return Err(EditError::MissingMaterial);
    };
    let mut beats = user_beats(slot);
    if beats.contains(&source) {
        return Err(EditError::DuplicateBeat);
    }
    beats.push(source);
    set_user_beats(slot, &beats);
    Ok(())
}

fn remove_source(
    draft: &mut Draft,
    material_id: &MaterialId,
    source: i64,
) -> Result<(), EditError> {
    let Some(slot) = draft
        .materials
        .beats
        .iter_mut()
        .find(|item| &item.id == material_id)
    else {
        return Err(EditError::MissingMaterial);
    };
    let mut beats = user_beats(slot);
    let before = beats.len();
    beats.retain(|beat| *beat != source);
    if beats.len() == before {
        return Err(EditError::MissingMaterial);
    }
    set_user_beats(slot, &beats);
    Ok(())
}

fn beat_id(materials: &Materials, segment: &Segment) -> Option<MaterialId> {
    segment_beats(materials, segment).map(|raw| raw.id.clone())
}

fn ref_count(draft: &Draft, id: &MaterialId) -> usize {
    draft
        .tracks
        .iter()
        .flat_map(|track| track.segments.iter())
        .filter(|segment| {
            segment
                .extra_material_refs
                .iter()
                .any(|ref_id| ref_id == id)
        })
        .count()
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        MaterialId, Micros, Segment, SegmentId, TimeRange, TrackAttribute, TrackId, TrackType,
        segment_beats, user_beats,
    };

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    fn audio_clip() -> Request {
        Request::Place {
            track_id: TrackId::new("audio"),
            segment: Box::new(Segment::new("clip", "tone", TimeRange::new(0, 1_000_000))),
        }
    }

    fn beat(at: i64, material_id: &str) -> Request {
        Request::AddBeat {
            segment_id: SegmentId::new("clip"),
            source: Micros(at),
            material_id: MaterialId::new(material_id),
        }
    }

    fn open() -> Session {
        let mut session = Session::open(crate::edit::test_draft("D"));
        session
            .apply(
                0,
                &[Request::InsertTrack {
                    track_id: TrackId::new("audio"),
                    kind: TrackType::Audio,
                }],
            )
            .unwrap();
        session.apply(1, &[audio_clip()]).unwrap();
        session
    }

    #[test]
    fn a_beat_on_audio_is_a_source_time_and_a_video_clip_refuses_one() {
        let mut session = open();
        session.apply(2, &[beat(400_000, "beats")]).unwrap();
        let draft = session.draft();
        assert_eq!(user_beats(&draft.materials.beats[0]), vec![400_000]);
        assert_eq!(draft.tracks[1].segments[0].extra_material_refs.len(), 1);
        let shown = cutlass_model::visible_beats(draft);
        assert_eq!(shown[0].time, Micros(400_000));

        let error = session.apply(3, &[beat(400_000, "again")]).unwrap_err();
        assert_eq!(error, EditError::DuplicateBeat);
        assert_eq!(user_beats(&session.draft().materials.beats[0]).len(), 1);

        session
            .apply(
                3,
                &[Request::Move {
                    segment_id: SegmentId::new("clip"),
                    track_id: TrackId::new("audio"),
                    start: Micros(1_000_000),
                }],
            )
            .unwrap();
        let shown = cutlass_model::visible_beats(session.draft());
        assert_eq!(shown[0].source, 400_000);
        assert_eq!(shown[0].time, Micros(1_400_000));
        session.undo().unwrap();

        session
            .apply(
                session.revision(),
                &[Request::Remove {
                    segment_id: SegmentId::new("clip"),
                }],
            )
            .unwrap();
        assert!(session.draft().materials.beats.is_empty());

        let mut video = Session::open(crate::edit::test_draft("D"));
        video
            .apply(
                0,
                &[Request::Place {
                    track_id: TrackId::new("main"),
                    segment: Box::new(Segment::new("clip", "media", TimeRange::new(0, 1_000_000))),
                }],
            )
            .unwrap();
        let error = video.apply(1, &[beat(100_000, "beats")]).unwrap_err();
        assert_eq!(error, EditError::NotAudio);
        assert!(video.draft().materials.beats.is_empty());
    }

    #[test]
    fn removing_a_beat_drops_it_and_its_empty_material_in_one_undo_step() {
        let mut session = open();
        session
            .apply(2, &[beat(200_000, "beats"), beat(600_000, "beats")])
            .unwrap();
        let remove = |source: i64| Request::RemoveBeat {
            segment_id: SegmentId::new("clip"),
            source: Micros(source),
        };
        let revision = session.revision();
        assert_eq!(
            session.apply(revision, &[remove(300_000)]),
            Err(EditError::MissingMaterial),
            "no beat there"
        );
        assert_eq!(
            session.apply(
                revision,
                &[Request::RemoveBeat {
                    segment_id: SegmentId::new("nowhere"),
                    source: Micros(200_000),
                }]
            ),
            Err(EditError::MissingSegment)
        );
        session.apply(revision, &[remove(200_000)]).unwrap();
        assert_eq!(
            user_beats(&session.draft().materials.beats[0]),
            vec![600_000]
        );
        session
            .apply(session.revision(), &[remove(600_000)])
            .unwrap();
        let draft = session.draft();
        assert!(draft.materials.beats.is_empty(), "the empty material goes");
        assert!(draft.tracks[1].segments[0].extra_material_refs.is_empty());
        assert!(segment_beats(&draft.materials, &draft.tracks[1].segments[0]).is_none());
        assert_eq!(
            session.apply(session.revision(), &[remove(600_000)]),
            Err(EditError::MissingMaterial),
            "a clip with no beats"
        );
        session.undo().unwrap();
        assert_eq!(
            user_beats(&session.draft().materials.beats[0]),
            vec![600_000]
        );
        session.undo().unwrap();
        assert_eq!(
            user_beats(&session.draft().materials.beats[0]),
            vec![200_000, 600_000]
        );
        session
            .apply(
                session.revision(),
                &[Request::SetTrackAttribute {
                    track_id: TrackId::new("audio"),
                    bit: TrackAttribute::LOCK,
                    on: true,
                }],
            )
            .unwrap();
        assert_eq!(
            session.apply(session.revision(), &[remove(200_000)]),
            Err(EditError::LockedTrack)
        );
    }

    #[test]
    fn a_split_gives_each_piece_the_beats_on_its_side() {
        let mut session = open();
        session.apply(2, &[beat(200_000, "beats")]).unwrap();
        session.apply(3, &[beat(700_000, "unused")]).unwrap();
        session
            .apply(
                4,
                &[Request::Split {
                    segment_id: SegmentId::new("clip"),
                    at: Micros(500_000),
                    right_id: SegmentId::new("right"),
                }],
            )
            .unwrap();
        let draft = session.draft();
        let left = &draft.tracks[1].segments[0];
        let right = &draft.tracks[1].segments[1];
        assert_ne!(left.extra_material_refs[0], right.extra_material_refs[0]);
        let left_beats = user_beats(segment_beats(&draft.materials, left).unwrap());
        let right_beats = user_beats(segment_beats(&draft.materials, right).unwrap());
        assert_eq!(left_beats, vec![200_000]);
        assert_eq!(right_beats, vec![200_000]);
        let shown = cutlass_model::visible_beats(draft);
        assert!(shown.iter().any(|beat| beat.time == Micros(200_000)));
        assert!(shown.iter().any(|beat| beat.time == Micros(700_000)));
    }

    #[test]
    fn a_source_range_keeps_the_stored_time_across_a_split() {
        let mut session = Session::open(crate::edit::test_draft("D"));
        session
            .apply(
                0,
                &[Request::InsertTrack {
                    track_id: TrackId::new("audio"),
                    kind: TrackType::Audio,
                }],
            )
            .unwrap();
        let mut clip = Segment::new("clip", "tone", TimeRange::new(0, 1_000_000));
        clip.source_range = Some(TimeRange::new(0, 1_000_000));
        session
            .apply(
                1,
                &[Request::Place {
                    track_id: TrackId::new("audio"),
                    segment: Box::new(clip),
                }],
            )
            .unwrap();
        session.apply(2, &[beat(200_000, "beats")]).unwrap();
        session.apply(3, &[beat(700_000, "unused")]).unwrap();
        session
            .apply(
                4,
                &[Request::Split {
                    segment_id: SegmentId::new("clip"),
                    at: Micros(500_000),
                    right_id: SegmentId::new("right"),
                }],
            )
            .unwrap();
        let draft = session.draft();
        let right = &draft.tracks[1].segments[1];
        assert_eq!(
            user_beats(segment_beats(&draft.materials, right).unwrap()),
            vec![700_000]
        );
    }

    #[test]
    fn a_locked_audio_clip_refuses_a_beat() {
        let mut session = open();
        session
            .apply(
                2,
                &[Request::SetTrackAttribute {
                    track_id: TrackId::new("audio"),
                    bit: TrackAttribute::LOCK,
                    on: true,
                }],
            )
            .unwrap();
        let error = session.apply(3, &[beat(100_000, "beats")]).unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert!(session.draft().materials.beats.is_empty());
    }
}

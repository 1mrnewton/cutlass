use cutlass_model::{
    Draft, MaterialId, Materials, Micros, Segment, SegmentId, TimeMarkItem, new_time_mark,
    segment_time_mark, set_time_mark_items, time_mark_items,
};

use super::speed::locate;
use super::{EditError, ensure_unlocked};

/// Store a marker. `None` is the ruler. `Some` is a clip, and the time is
/// stored as an offset from that clip's target start.
pub fn add_time_mark(
    draft: &mut Draft,
    segment_id: Option<&SegmentId>,
    at: Micros,
    color: &str,
    material_id: &MaterialId,
) -> Result<(), EditError> {
    if let Some(segment_id) = segment_id {
        add_clip_mark(draft, segment_id, at, color, material_id)
    } else {
        add_ruler_mark(draft, at, color, material_id)
    }
}

/// Remove one stored mark. `start` is the timeline time on the ruler, or the
/// offset on a clip.
pub fn remove_time_mark(
    draft: &mut Draft,
    segment_id: Option<&SegmentId>,
    start: Micros,
) -> Result<(), EditError> {
    if let Some(segment_id) = segment_id {
        let (track_index, segment_index) = locate(draft, segment_id)?;
        ensure_unlocked(draft.tracks[track_index].attribute)?;
        let Some(material_id) = mark_id(
            &draft.materials,
            &draft.tracks[track_index].segments[segment_index],
        ) else {
            return Err(EditError::MissingMaterial);
        };
        remove_item(draft, &material_id, start.0)?;
        let empty = draft
            .materials
            .time_marks
            .iter()
            .find(|item| item.id == material_id)
            .is_none_or(|item| time_mark_items(item).is_empty());
        if empty {
            draft
                .materials
                .time_marks
                .retain(|item| item.id != material_id);
            draft.tracks[track_index].segments[segment_index]
                .extra_material_refs
                .retain(|id| id != &material_id);
        }
        return Ok(());
    }
    let Some(material_id) = ruler_material(draft, Some(start.0)) else {
        return Err(EditError::MissingMaterial);
    };
    remove_item(draft, &material_id, start.0)?;
    let empty = draft
        .materials
        .time_marks
        .iter()
        .find(|item| item.id == material_id)
        .is_none_or(|item| time_mark_items(item).is_empty());
    if empty {
        draft
            .materials
            .time_marks
            .retain(|item| item.id != material_id);
    }
    Ok(())
}

/// Give the right piece the marks that fall on its side of the cut.
///
/// Offsets are from the clip start. A material shared with another clip is
/// left on the left piece.
pub(super) fn split_time_marks(
    draft: &mut Draft,
    left: &mut Segment,
    right: &mut Segment,
    left_duration: i64,
    right_id: &SegmentId,
) -> Result<(), EditError> {
    let Some(material_id) = mark_id(&draft.materials, left) else {
        return Ok(());
    };
    let owners = ref_count(draft, &material_id);
    if owners != 1 {
        right.extra_material_refs.retain(|id| id != &material_id);
        return Ok(());
    }
    let Some(index) = draft
        .materials
        .time_marks
        .iter()
        .position(|item| item.id == material_id)
    else {
        left.extra_material_refs.retain(|id| id != &material_id);
        right.extra_material_refs.retain(|id| id != &material_id);
        return Ok(());
    };
    let items = time_mark_items(&draft.materials.time_marks[index]);
    let mut stay = Vec::new();
    let mut moved = Vec::new();
    for mut item in items {
        if item.start.0 < left_duration {
            stay.push(item);
            continue;
        }
        match item.start.0.checked_sub(left_duration) {
            Some(next) => {
                item.start = Micros(next);
                moved.push(item);
            }
            None => stay.push(item),
        }
    }
    right.extra_material_refs.retain(|id| id != &material_id);
    if stay.is_empty() {
        draft.materials.time_marks.remove(index);
        left.extra_material_refs.retain(|id| id != &material_id);
    } else {
        set_time_mark_items(&mut draft.materials.time_marks[index], &stay);
    }
    if moved.is_empty() {
        return Ok(());
    }
    let id = MaterialId::new(format!("{}:time-mark", right_id.as_str()));
    if draft.materials.get(id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    draft
        .materials
        .time_marks
        .push(new_time_mark(id.clone(), moved[0].clone()));
    let slot = draft
        .materials
        .time_marks
        .last_mut()
        .ok_or(EditError::MissingMaterial)?;
    set_time_mark_items(slot, &moved);
    right.extra_material_refs.push(id);
    Ok(())
}

/// Drop time-mark materials that belonged to a clip which is now gone.
pub(super) fn drop_unreferenced_time_marks(draft: &mut Draft, ids: &[MaterialId]) {
    for id in ids {
        if ref_count(draft, id) == 0 {
            draft.materials.time_marks.retain(|item| &item.id != id);
        }
    }
}

pub(super) fn time_mark_ids(materials: &Materials, segment: &Segment) -> Vec<MaterialId> {
    segment
        .extra_material_refs
        .iter()
        .filter(|id| materials.time_marks.iter().any(|item| &item.id == *id))
        .cloned()
        .collect()
}

fn add_ruler_mark(
    draft: &mut Draft,
    at: Micros,
    color: &str,
    material_id: &MaterialId,
) -> Result<(), EditError> {
    if at.0 < 0 {
        return Err(EditError::Range(
            cutlass_model::RangeError::NonPositiveDuration,
        ));
    }
    if let Some(existing) = ruler_material(draft, None) {
        return push_item(draft, &existing, at, color);
    }
    if material_id.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    if draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    draft.materials.time_marks.push(new_time_mark(
        material_id.clone(),
        TimeMarkItem::point(at, color),
    ));
    Ok(())
}

fn add_clip_mark(
    draft: &mut Draft,
    segment_id: &SegmentId,
    at: Micros,
    color: &str,
    material_id: &MaterialId,
) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let segment = &draft.tracks[track_index].segments[segment_index];
    let start = segment.target.start.0;
    let end = start
        .checked_add(segment.target.duration.0)
        .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
    if at.0 < start || at.0 >= end {
        return Err(EditError::Range(
            cutlass_model::RangeError::NonPositiveDuration,
        ));
    }
    let offset =
        at.0.checked_sub(start)
            .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
    if let Some(existing) = mark_id(&draft.materials, segment) {
        return push_item(draft, &existing, Micros(offset), color);
    }
    if material_id.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    if draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    draft.materials.time_marks.push(new_time_mark(
        material_id.clone(),
        TimeMarkItem::point(Micros(offset), color),
    ));
    draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .push(material_id.clone());
    Ok(())
}

fn push_item(
    draft: &mut Draft,
    material_id: &MaterialId,
    start: Micros,
    color: &str,
) -> Result<(), EditError> {
    let Some(slot) = draft
        .materials
        .time_marks
        .iter_mut()
        .find(|item| &item.id == material_id)
    else {
        return Err(EditError::MissingMaterial);
    };
    let mut items = time_mark_items(slot);
    if items.iter().any(|item| item.start == start) {
        return Err(EditError::DuplicateMark);
    }
    items.push(TimeMarkItem::point(start, color));
    set_time_mark_items(slot, &items);
    Ok(())
}

fn remove_item(draft: &mut Draft, material_id: &MaterialId, start: i64) -> Result<(), EditError> {
    let Some(slot) = draft
        .materials
        .time_marks
        .iter_mut()
        .find(|item| &item.id == material_id)
    else {
        return Err(EditError::MissingMaterial);
    };
    let mut items = time_mark_items(slot);
    let before = items.len();
    items.retain(|item| item.start.0 != start);
    if items.len() == before {
        return Err(EditError::MissingMaterial);
    }
    set_time_mark_items(slot, &items);
    Ok(())
}

fn mark_id(materials: &Materials, segment: &Segment) -> Option<MaterialId> {
    segment_time_mark(materials, segment).map(|raw| raw.id.clone())
}

/// `start` narrows the search to the material that holds that time.
fn ruler_material(draft: &Draft, start: Option<i64>) -> Option<MaterialId> {
    let candidates: Vec<MaterialId> = draft
        .materials
        .time_marks
        .iter()
        .map(|item| item.id.clone())
        .collect();
    candidates.into_iter().find(|id| {
        ref_count(draft, id) == 0
            && start.is_none_or(|start| {
                draft
                    .materials
                    .time_marks
                    .iter()
                    .find(|item| &item.id == id)
                    .is_some_and(|item| {
                        time_mark_items(item)
                            .iter()
                            .any(|mark| mark.start.0 == start)
                    })
            })
    })
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
    use cutlass_model::{Draft, MaterialId, Micros, Segment, SegmentId, TimeRange, TrackAttribute};

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    fn place_clip() -> Request {
        Request::Place {
            track_id: "main".into(),
            segment: Box::new(Segment::new("clip", "media", TimeRange::new(0, 1_000_000))),
        }
    }

    fn mark(segment_id: Option<&str>, at: i64, material_id: &str) -> Request {
        Request::AddTimeMark {
            segment_id: segment_id.map(SegmentId::new),
            at: Micros(at),
            color: "#E8B931".to_owned(),
            material_id: MaterialId::new(material_id),
        }
    }

    #[test]
    fn a_ruler_mark_stays_a_time_and_a_clip_mark_is_an_offset() {
        let mut draft = Draft::empty("D");
        draft.config.maintrack_adsorb = false;
        let mut session = Session::open(draft);
        session.apply(0, &[place_clip()]).unwrap();
        session.apply(1, &[mark(None, 250_000, "ruler")]).unwrap();
        session
            .apply(2, &[mark(Some("clip"), 400_000, "clip-mark")])
            .unwrap();
        let draft = session.draft();
        assert_eq!(draft.materials.time_marks.len(), 2);
        assert_eq!(
            cutlass_model::time_mark_items(&draft.materials.time_marks[0])[0].start,
            Micros(250_000)
        );
        assert!(draft.tracks[0].segments[0].extra_material_refs.len() == 1);
        assert_eq!(
            cutlass_model::time_mark_items(&draft.materials.time_marks[1])[0].start,
            Micros(400_000)
        );
        assert_eq!(
            draft.tracks[0].segments[0].target,
            TimeRange::new(0, 1_000_000)
        );

        let error = session
            .apply(3, &[mark(Some("clip"), 400_000, "again")])
            .unwrap_err();
        assert_eq!(error, EditError::DuplicateMark);
        assert_eq!(session.draft().materials.time_marks.len(), 2);

        session
            .apply(
                3,
                &[Request::Move {
                    segment_id: SegmentId::new("clip"),
                    track_id: "main".into(),
                    start: Micros(1_000_000),
                }],
            )
            .unwrap();
        let marks = cutlass_model::visible_marks(session.draft());
        let clip_mark = marks.iter().find(|item| item.segment_id.is_some()).unwrap();
        assert_eq!(clip_mark.stored_start, 400_000);
        assert_eq!(clip_mark.time, Micros(1_400_000));

        session.undo().unwrap();
        assert_eq!(
            session.draft().tracks[0].segments[0].target.start,
            Micros(0)
        );
        session
            .apply(
                session.revision(),
                &[Request::Remove {
                    segment_id: SegmentId::new("clip"),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().materials.time_marks.len(), 1);
        assert!(session.draft().tracks[0].segments.is_empty());
    }

    #[test]
    fn removing_a_mark_drops_it_and_its_empty_material_in_one_undo_step() {
        let mut draft = Draft::empty("D");
        draft.config.maintrack_adsorb = false;
        let mut session = Session::open(draft);
        session.apply(0, &[place_clip()]).unwrap();
        session
            .apply(
                1,
                &[
                    mark(None, 250_000, "ruler"),
                    mark(None, 500_000, "ruler"),
                    mark(Some("clip"), 400_000, "clip-mark"),
                ],
            )
            .unwrap();
        let remove = |segment_id: Option<&str>, start: i64| Request::RemoveTimeMark {
            segment_id: segment_id.map(SegmentId::new),
            start: Micros(start),
        };
        // A ruler mark that is not there, or a clip with no marks, refuses.
        let revision = session.revision();
        assert_eq!(
            session.apply(revision, &[remove(None, 300_000)]),
            Err(EditError::MissingMaterial)
        );
        assert_eq!(
            session.apply(revision, &[remove(Some("clip"), 100_000)]),
            Err(EditError::MissingMaterial)
        );
        assert_eq!(
            session.apply(revision, &[remove(Some("nowhere"), 400_000)]),
            Err(EditError::MissingSegment)
        );
        // One of two ruler marks goes; the material stays for the other.
        session.apply(revision, &[remove(None, 250_000)]).unwrap();
        let marks = cutlass_model::visible_marks(session.draft());
        let ruler: Vec<i64> = marks
            .iter()
            .filter(|item| item.segment_id.is_none())
            .map(|item| item.time.0)
            .collect();
        assert_eq!(ruler, [500_000]);
        // The last mark on the clip takes its material and its reference.
        session
            .apply(session.revision(), &[remove(Some("clip"), 400_000)])
            .unwrap();
        let draft = session.draft();
        assert!(draft.tracks[0].segments[0].extra_material_refs.is_empty());
        assert_eq!(draft.materials.time_marks.len(), 1);
        // The last ruler mark takes its material too.
        session
            .apply(session.revision(), &[remove(None, 500_000)])
            .unwrap();
        assert!(session.draft().materials.time_marks.is_empty());
        // Each removal is one undo step.
        session.undo().unwrap();
        assert_eq!(cutlass_model::visible_marks(session.draft()).len(), 1);
        session.undo().unwrap();
        assert_eq!(cutlass_model::visible_marks(session.draft()).len(), 2);
        // A locked clip keeps its marks.
        let locked = session.draft().tracks[0].id.clone();
        session
            .apply(
                session.revision(),
                &[Request::SetTrackAttribute {
                    track_id: locked,
                    bit: TrackAttribute::LOCK,
                    on: true,
                }],
            )
            .unwrap();
        assert_eq!(
            session.apply(session.revision(), &[remove(Some("clip"), 400_000)]),
            Err(EditError::LockedTrack)
        );
    }

    #[test]
    fn a_split_gives_each_piece_the_marks_on_its_side() {
        let mut session = Session::open(Draft::empty("D"));
        session.apply(0, &[place_clip()]).unwrap();
        session
            .apply(1, &[mark(Some("clip"), 200_000, "marks")])
            .unwrap();
        session
            .apply(2, &[mark(Some("clip"), 700_000, "unused")])
            .unwrap();
        session
            .apply(
                3,
                &[Request::Split {
                    segment_id: SegmentId::new("clip"),
                    at: Micros(500_000),
                    right_id: SegmentId::new("right"),
                }],
            )
            .unwrap();
        let draft = session.draft();
        assert_eq!(draft.tracks[0].segments.len(), 2);
        let left = &draft.tracks[0].segments[0];
        let right = &draft.tracks[0].segments[1];
        assert_eq!(left.extra_material_refs.len(), 1);
        assert_eq!(right.extra_material_refs.len(), 1);
        assert_ne!(left.extra_material_refs[0], right.extra_material_refs[0]);
        let left_items = cutlass_model::time_mark_items(
            cutlass_model::segment_time_mark(&draft.materials, left).unwrap(),
        );
        let right_items = cutlass_model::time_mark_items(
            cutlass_model::segment_time_mark(&draft.materials, right).unwrap(),
        );
        assert_eq!(left_items[0].start, Micros(200_000));
        assert_eq!(right_items[0].start, Micros(200_000));
    }

    #[test]
    fn a_locked_clip_refuses_a_marker() {
        let mut session = Session::open(Draft::empty("D"));
        session.apply(0, &[place_clip()]).unwrap();
        session
            .apply(
                1,
                &[Request::SetTrackAttribute {
                    track_id: "main".into(),
                    bit: TrackAttribute::LOCK,
                    on: true,
                }],
            )
            .unwrap();
        let error = session
            .apply(2, &[mark(Some("clip"), 100_000, "marks")])
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert!(session.draft().materials.time_marks.is_empty());
    }
}

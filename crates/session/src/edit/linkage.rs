use std::collections::BTreeSet;

use cutlass_model::{
    Draft, LinkPair, LinkageSettings, Micros, Relationship, Segment, SegmentId, TimeRange,
};

use super::speed::locate;
use super::{EditError, ensure_unlocked};

/// Record a pair. The request names the owner. A dependent has one owner.
pub fn link(draft: &mut Draft, owner: &SegmentId, dependent: &SegmentId) -> Result<(), EditError> {
    if owner == dependent {
        return Err(EditError::AlreadyLinked);
    }
    let (owner_track, _) = locate(draft, owner)?;
    let (dependent_track, _) = locate(draft, dependent)?;
    ensure_unlocked(draft.tracks[owner_track].attribute)?;
    ensure_unlocked(draft.tracks[dependent_track].attribute)?;
    let pairs = pairs(draft);
    if pairs.iter().any(|pair| {
        (&pair.dependent == dependent && &pair.owner != owner)
            || &pair.owner == dependent
            || &pair.dependent == owner
    }) {
        return Err(EditError::AlreadyLinked);
    }
    if pairs
        .iter()
        .any(|pair| &pair.owner == owner && &pair.dependent == dependent)
    {
        return Ok(());
    }
    push_pair(
        draft,
        LinkPair {
            owner: owner.clone(),
            dependent: dependent.clone(),
        },
    );
    Ok(())
}

/// Remove one segment. Linkage then removes dependents in the same combo.
pub fn remove(draft: &mut Draft, segment_id: &SegmentId) -> Result<(), EditError> {
    remove_segment(draft, segment_id)
}

/// Apply follow, cut, and group movement for the edit that just landed.
pub fn after_request(
    draft: &mut Draft,
    before: &Draft,
    settings: &LinkageSettings,
) -> Result<(), EditError> {
    let before_segments = index(before);
    let after_segments = index(draft);
    let mut removed = Vec::new();
    let mut edits = Vec::new();
    for (id, old) in &before_segments {
        match after_segments.get(id) {
            None => removed.push(id.clone()),
            Some(new) => edits.push((old.clone(), new.clone())),
        }
    }
    for id in &removed {
        on_removed(draft, id, settings)?;
    }
    for (old, new) in &edits {
        if segment(draft, &old.id).is_none() {
            continue;
        }
        on_edit(draft, before, old, new, settings)?;
    }
    Ok(())
}

fn on_removed(
    draft: &mut Draft,
    removed_id: &SegmentId,
    settings: &LinkageSettings,
) -> Result<(), EditError> {
    if !settings.enabled {
        return Ok(());
    }
    let dependents = dependent_ids(draft, removed_id);
    for dependent_id in dependents {
        if !follow_allows(draft, &dependent_id, settings) {
            continue;
        }
        remove_segment(draft, &dependent_id)?;
        drop_pair(draft, removed_id, &dependent_id);
    }
    drop_pairs_where(draft, |pair| &pair.dependent == removed_id);
    Ok(())
}

fn on_edit(
    draft: &mut Draft,
    before: &Draft,
    old: &Segment,
    new: &Segment,
    settings: &LinkageSettings,
) -> Result<(), EditError> {
    if let Some((cut, right_owner)) = split_pieces(before, draft, old, new) {
        return on_split(draft, &old.id, cut, &right_owner, settings);
    }
    let start_delta = new
        .target
        .start
        .0
        .checked_sub(old.target.start.0)
        .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
    let duration_changed = new.target.duration != old.target.duration;
    if start_delta != 0 && !duration_changed {
        on_move(draft, old, start_delta, settings)?;
    } else if start_delta != 0 {
        on_start_trim(draft, old, new, start_delta, settings)?;
    } else if duration_changed {
        on_duration(draft, old, new, settings)?;
    }
    Ok(())
}

fn on_move(
    draft: &mut Draft,
    old: &Segment,
    delta: i64,
    settings: &LinkageSettings,
) -> Result<(), EditError> {
    let mut ids = group_mates(draft, old);
    if settings.enabled {
        for dependent_id in dependent_ids(draft, &old.id) {
            if follow_allows(draft, &dependent_id, settings) {
                ids.insert(dependent_id);
            }
        }
    }
    for id in ids {
        shift_start(draft, &id, delta)?;
    }
    reap(draft, &old.id, settings)
}

fn on_start_trim(
    draft: &mut Draft,
    old: &Segment,
    new: &Segment,
    offset: i64,
    settings: &LinkageSettings,
) -> Result<(), EditError> {
    let dependents = following(draft, &old.id, settings);
    for dependent_id in &dependents {
        let duration = segment(draft, dependent_id)
            .map(|item| item.target.duration.0)
            .ok_or(EditError::MissingSegment)?;
        shift_start(draft, dependent_id, offset)?;
        let scaled = scale_duration(duration, old.target.duration.0, new.target.duration.0)?;
        set_duration(draft, dependent_id, scaled)?;
    }
    reap(draft, &old.id, settings)
}

fn on_duration(
    draft: &mut Draft,
    old: &Segment,
    new: &Segment,
    settings: &LinkageSettings,
) -> Result<(), EditError> {
    let dependents = following(draft, &old.id, settings);
    for dependent_id in &dependents {
        let duration = segment(draft, dependent_id)
            .map(|item| item.target.duration.0)
            .ok_or(EditError::MissingSegment)?;
        let scaled = scale_duration(duration, old.target.duration.0, new.target.duration.0)?;
        set_duration(draft, dependent_id, scaled)?;
    }
    reap(draft, &old.id, settings)
}

fn on_split(
    draft: &mut Draft,
    owner_id: &SegmentId,
    cut: Micros,
    right_owner: &SegmentId,
    settings: &LinkageSettings,
) -> Result<(), EditError> {
    if !settings.enabled {
        return Ok(());
    }
    let dependents = dependent_ids(draft, owner_id);
    for dependent_id in dependents {
        let Some(track) = track_of(draft, &dependent_id) else {
            continue;
        };
        if !settings.cut.allows(&draft.tracks[track]) {
            continue;
        }
        let Some(dependent) = segment(draft, &dependent_id).cloned() else {
            continue;
        };
        if time_inside(&dependent.target, cut)? {
            let right_id = SegmentId::new(format!(
                "{}+{}",
                dependent_id.as_str(),
                right_owner.as_str()
            ));
            super::split::split(draft, &dependent_id, cut, &right_id)?;
            link(draft, right_owner, &right_id)?;
        } else if settings.follow.allows(&draft.tracks[track])
            && owner_excludes(draft, owner_id, &dependent_id)?
        {
            remove_segment(draft, &dependent_id)?;
            drop_pair(draft, owner_id, &dependent_id);
        }
    }
    Ok(())
}

fn following(draft: &Draft, owner: &SegmentId, settings: &LinkageSettings) -> Vec<SegmentId> {
    if !settings.enabled {
        return Vec::new();
    }
    dependent_ids(draft, owner)
        .into_iter()
        .filter(|dependent_id| follow_allows(draft, dependent_id, settings))
        .collect()
}

fn follow_allows(draft: &Draft, dependent_id: &SegmentId, settings: &LinkageSettings) -> bool {
    track_of(draft, dependent_id).is_some_and(|index| settings.follow.allows(&draft.tracks[index]))
}

fn group_mates(draft: &Draft, segment: &Segment) -> BTreeSet<SegmentId> {
    let mut ids = BTreeSet::new();
    if segment.group_id.is_empty() {
        return ids;
    }
    for track in &draft.tracks {
        for other in &track.segments {
            if other.id != segment.id && other.group_id == segment.group_id {
                ids.insert(other.id.clone());
            }
        }
    }
    ids
}

fn split_pieces(
    before: &Draft,
    after: &Draft,
    old: &Segment,
    new: &Segment,
) -> Option<(Micros, SegmentId)> {
    if new.target.start != old.target.start || new.target.duration.0 >= old.target.duration.0 {
        return None;
    }
    let cut = new.target.start.0.checked_add(new.target.duration.0)?;
    let old_end = old.target.start.0.checked_add(old.target.duration.0)?;
    let before_ids = index(before);
    for track in &after.tracks {
        for segment in &track.segments {
            if before_ids.contains_key(&segment.id) {
                continue;
            }
            let Some(end) = segment
                .target
                .start
                .0
                .checked_add(segment.target.duration.0)
            else {
                continue;
            };
            if segment.target.start.0 == cut && end == old_end {
                return Some((Micros(cut), segment.id.clone()));
            }
        }
    }
    None
}

fn reap(
    draft: &mut Draft,
    owner_id: &SegmentId,
    settings: &LinkageSettings,
) -> Result<(), EditError> {
    if !settings.enabled {
        return Ok(());
    }
    let dependents = following(draft, owner_id, settings);
    for dependent_id in dependents {
        if owner_excludes(draft, owner_id, &dependent_id)? {
            remove_segment(draft, &dependent_id)?;
            drop_pair(draft, owner_id, &dependent_id);
        }
    }
    Ok(())
}

fn owner_excludes(
    draft: &Draft,
    owner_id: &SegmentId,
    dependent_id: &SegmentId,
) -> Result<bool, EditError> {
    let Some(owner) = segment(draft, owner_id) else {
        return Ok(false);
    };
    let Some(dependent) = segment(draft, dependent_id) else {
        return Ok(false);
    };
    entirely_outside(&dependent.target, &owner.target)
}

fn entirely_outside(dependent: &TimeRange, owner: &TimeRange) -> Result<bool, EditError> {
    let dependent_end = dependent
        .start
        .0
        .checked_add(dependent.duration.0)
        .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
    let owner_end = owner
        .start
        .0
        .checked_add(owner.duration.0)
        .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
    Ok(dependent_end <= owner.start.0 || dependent.start.0 >= owner_end)
}

fn time_inside(range: &TimeRange, time: Micros) -> Result<bool, EditError> {
    let end = range
        .start
        .0
        .checked_add(range.duration.0)
        .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
    Ok(time.0 > range.start.0 && time.0 < end)
}

fn scale_duration(duration: i64, old_owner: i64, new_owner: i64) -> Result<i64, EditError> {
    if duration <= 0 || old_owner <= 0 || new_owner <= 0 {
        return Err(EditError::Range(
            cutlass_model::RangeError::NonPositiveDuration,
        ));
    }
    let scaled = (duration as i128) * (new_owner as i128);
    let divisor = old_owner as i128;
    let rounded = (scaled + divisor / 2) / divisor;
    if rounded <= 0 || rounded > i64::MAX as i128 {
        return Err(EditError::Range(
            cutlass_model::RangeError::NonPositiveDuration,
        ));
    }
    Ok(rounded as i64)
}

fn shift_start(draft: &mut Draft, segment_id: &SegmentId, delta: i64) -> Result<(), EditError> {
    if delta == 0 {
        return Ok(());
    }
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let mut moved = draft.tracks[track_index].segments[segment_index].clone();
    let start = moved
        .target
        .start
        .0
        .checked_add(delta)
        .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
    moved.target.start = Micros(start);
    if overlaps(draft, track_index, &moved, segment_id)? {
        return Err(EditError::Overlap);
    }
    draft.tracks[track_index].segments.remove(segment_index);
    let index = draft.tracks[track_index]
        .segments
        .iter()
        .position(|existing| existing.target.start > moved.target.start)
        .unwrap_or(draft.tracks[track_index].segments.len());
    draft.tracks[track_index].segments.insert(index, moved);
    Ok(())
}

fn set_duration(draft: &mut Draft, segment_id: &SegmentId, duration: i64) -> Result<(), EditError> {
    if duration <= 0 {
        return Err(EditError::Range(
            cutlass_model::RangeError::NonPositiveDuration,
        ));
    }
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let mut next = draft.tracks[track_index].segments[segment_index].clone();
    next.target.duration = Micros(duration);
    if overlaps(draft, track_index, &next, segment_id)? {
        return Err(EditError::Overlap);
    }
    draft.tracks[track_index].segments[segment_index] = next;
    Ok(())
}

fn remove_segment(draft: &mut Draft, segment_id: &SegmentId) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    draft.tracks[track_index].segments.remove(segment_index);
    Ok(())
}

fn overlaps(
    draft: &Draft,
    track_index: usize,
    candidate: &Segment,
    ignore: &SegmentId,
) -> Result<bool, EditError> {
    if candidate.target.duration.0 <= 0 {
        return Err(EditError::Range(
            cutlass_model::RangeError::NonPositiveDuration,
        ));
    }
    for segment in &draft.tracks[track_index].segments {
        if &segment.id == ignore {
            continue;
        }
        if candidate
            .target
            .overlaps(&segment.target)
            .map_err(EditError::Range)?
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn index(draft: &Draft) -> std::collections::BTreeMap<SegmentId, Segment> {
    let mut segments = std::collections::BTreeMap::new();
    for track in &draft.tracks {
        for segment in &track.segments {
            segments.insert(segment.id.clone(), segment.clone());
        }
    }
    segments
}

fn segment<'a>(draft: &'a Draft, segment_id: &SegmentId) -> Option<&'a Segment> {
    draft.tracks.iter().find_map(|track| {
        track
            .segments
            .iter()
            .find(|segment| &segment.id == segment_id)
    })
}

fn track_of(draft: &Draft, segment_id: &SegmentId) -> Option<usize> {
    draft.tracks.iter().position(|track| {
        track
            .segments
            .iter()
            .any(|segment| &segment.id == segment_id)
    })
}

fn pairs(draft: &Draft) -> Vec<LinkPair> {
    draft
        .relationships
        .iter()
        .filter(|relationship| relationship.kind == "linkage")
        .flat_map(|relationship| relationship.id_to_id.clone())
        .collect()
}

fn dependent_ids(draft: &Draft, owner: &SegmentId) -> Vec<SegmentId> {
    pairs(draft)
        .into_iter()
        .filter(|pair| &pair.owner == owner)
        .map(|pair| pair.dependent)
        .collect()
}

fn push_pair(draft: &mut Draft, pair: LinkPair) {
    if let Some(index) = draft
        .relationships
        .iter()
        .position(|relationship| relationship.kind == "linkage")
    {
        draft.relationships[index].id_to_id.push(pair);
        return;
    }
    draft.relationships.push(Relationship {
        kind: "linkage".to_owned(),
        id_to_id: vec![pair],
        ..Relationship::default()
    });
}

fn drop_pair(draft: &mut Draft, owner: &SegmentId, dependent: &SegmentId) {
    drop_pairs_where(draft, |pair| {
        &pair.owner == owner && &pair.dependent == dependent
    });
}

fn drop_pairs_where(draft: &mut Draft, mut drop_this: impl FnMut(&LinkPair) -> bool) {
    for relationship in &mut draft.relationships {
        if relationship.kind == "linkage" {
            relationship.id_to_id.retain(|pair| !drop_this(pair));
        }
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, KindSwitches, LinkPair, MaterialDraft, MaterialId, Micros, Segment, TimeRange,
        Track, TrackFlag, TrackId,
    };

    use crate::edit::EditError;
    use crate::request::{Request, TrimEdge};
    use crate::session::Session;

    fn open() -> (Session, TrackId, TrackId) {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        let mut overlay = Track::main("over");
        overlay.flag = TrackFlag::SubVideo;
        draft.tracks.push(overlay);
        (
            Session::open(draft),
            TrackId::new("main"),
            TrackId::new("over"),
        )
    }

    fn place(track: &TrackId, id: &str, start: i64, duration: i64) -> Request {
        placed(track, id, start, duration, "")
    }

    fn placed(track: &TrackId, id: &str, start: i64, duration: i64, group: &str) -> Request {
        let mut segment = Segment::new(id, "media", TimeRange::new(start, duration));
        segment.source_range = Some(TimeRange::new(0, duration));
        segment.volume = 0.4;
        segment.group_id = group.to_owned();
        Request::Place {
            track_id: track.clone(),
            segment: Box::new(segment),
        }
    }

    fn pair(owner: &str, dependent: &str) -> LinkPair {
        LinkPair {
            owner: owner.into(),
            dependent: dependent.into(),
        }
    }

    #[test]
    fn a_link_names_the_owner_and_a_second_owner_leaves_the_draft() {
        let (mut session, main, over) = open();
        session
            .apply(
                0,
                &[
                    place(&main, "a", 0, 1_000_000),
                    place(&over, "b", 0, 500_000),
                    place(&main, "c", 1_000_000, 500_000),
                ],
            )
            .unwrap();
        session
            .apply(
                1,
                &[Request::Link {
                    owner: "a".into(),
                    dependent: "b".into(),
                }],
            )
            .unwrap();
        assert_eq!(
            session.draft().relationships[0].id_to_id,
            vec![pair("a", "b")]
        );
        assert_eq!(session.draft().relationships[0].kind, "linkage");
        let before = session.draft().clone();
        let error = session
            .apply(
                2,
                &[Request::Link {
                    owner: "c".into(),
                    dependent: "b".into(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::AlreadyLinked);
        assert_eq!(session.draft(), &before);
        let error = session
            .apply(
                2,
                &[Request::Link {
                    owner: "b".into(),
                    dependent: "c".into(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::AlreadyLinked);
        assert_eq!(session.draft(), &before);
    }

    #[test]
    fn a_disabled_switch_keeps_the_pair_and_leaves_the_dependent() {
        let (mut session, main, over) = open();
        session
            .apply(
                0,
                &[
                    place(&main, "a", 0, 1_000_000),
                    place(&over, "b", 100_000, 200_000),
                    Request::Link {
                        owner: "a".into(),
                        dependent: "b".into(),
                    },
                ],
            )
            .unwrap();
        let mut settings = session.linkage();
        settings.enabled = false;
        session.set_linkage(settings);
        session
            .apply(
                1,
                &[Request::Move {
                    segment_id: "a".into(),
                    track_id: main,
                    start: Micros(50_000),
                }],
            )
            .unwrap();
        assert_eq!(
            session.draft().tracks[1].segments[0].target,
            TimeRange::new(100_000, 200_000)
        );
        assert_eq!(
            session.draft().relationships[0].id_to_id,
            vec![pair("a", "b")]
        );
    }

    #[test]
    fn a_move_shifts_dependents_and_a_shared_group_once() {
        let (mut session, main, over) = open();
        session
            .apply(
                0,
                &[
                    placed(&main, "a", 0, 1_000_000, "g"),
                    placed(&over, "b", 100_000, 200_000, "g"),
                    place(&over, "c", 500_000, 200_000),
                ],
            )
            .unwrap();
        session
            .apply(
                1,
                &[Request::Link {
                    owner: "a".into(),
                    dependent: "b".into(),
                }],
            )
            .unwrap();
        let before = session.draft().clone();
        let error = session
            .apply(
                2,
                &[Request::Move {
                    segment_id: "a".into(),
                    track_id: main.clone(),
                    start: Micros(400_000),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::Overlap);
        assert_eq!(session.draft(), &before);
        session
            .apply(
                2,
                &[Request::Move {
                    segment_id: "a".into(),
                    track_id: main,
                    start: Micros(40_000),
                }],
            )
            .unwrap();
        assert_eq!(
            session.draft().tracks[0].segments[0].target,
            TimeRange::new(40_000, 1_000_000)
        );
        assert_eq!(
            session.draft().tracks[1].segments[0].target,
            TimeRange::new(140_000, 200_000)
        );
        assert_eq!(session.draft().tracks[1].segments[0].volume, 0.4);
        assert_eq!(
            session.draft().tracks[1].segments[0].source_range,
            Some(TimeRange::new(0, 200_000))
        );
    }

    #[test]
    fn a_group_moves_together_while_linkage_is_off() {
        let (mut session, main, _) = open();
        let mut settings = session.linkage();
        settings.enabled = false;
        session.set_linkage(settings);
        session
            .apply(
                0,
                &[
                    placed(&main, "a", 0, 100_000, "g"),
                    placed(&main, "b", 300_000, 100_000, "g"),
                ],
            )
            .unwrap();
        session
            .apply(
                1,
                &[Request::Move {
                    segment_id: "a".into(),
                    track_id: main,
                    start: Micros(50_000),
                }],
            )
            .unwrap();
        assert_eq!(
            session.draft().tracks[0].segments[0].target,
            TimeRange::new(50_000, 100_000)
        );
        assert_eq!(
            session.draft().tracks[0].segments[1].target,
            TimeRange::new(350_000, 100_000)
        );
        assert!(session.draft().relationships.is_empty());
    }

    #[test]
    fn trim_and_duration_scale_the_dependent_and_leave_the_child_draft() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        let mut overlay = Track::main("over");
        overlay.flag = TrackFlag::SubVideo;
        draft.tracks.push(overlay);
        draft.materials.drafts.push(MaterialDraft {
            id: MaterialId::new("nest"),
            draft: Some(Box::new(Draft::empty("child"))),
            ..MaterialDraft::default()
        });
        let mut session = Session::open(draft);
        let main = TrackId::new("main");
        let over = TrackId::new("over");
        let mut dependent = Segment::new("b", "media", TimeRange::new(0, 500_000));
        dependent.source_range = Some(TimeRange::new(0, 500_000));
        dependent.volume = 0.4;
        dependent.extra_material_refs.push(MaterialId::new("nest"));
        session
            .apply(
                0,
                &[
                    place(&main, "a", 0, 1_000_000),
                    Request::Place {
                        track_id: over,
                        segment: Box::new(dependent),
                    },
                ],
            )
            .unwrap();
        session
            .apply(
                1,
                &[Request::Link {
                    owner: "a".into(),
                    dependent: "b".into(),
                }],
            )
            .unwrap();
        let nested = session.draft().materials.drafts.clone();
        session
            .apply(
                2,
                &[Request::Trim {
                    segment_id: "a".into(),
                    edge: TrimEdge::End,
                    time: Micros(500_000),
                }],
            )
            .unwrap();
        let dependent = &session.draft().tracks[1].segments[0];
        assert_eq!(dependent.target, TimeRange::new(0, 250_000));
        assert_eq!(dependent.source_range, Some(TimeRange::new(0, 500_000)));
        assert_eq!(dependent.volume, 0.4);
        assert_eq!(session.draft().materials.drafts, nested);
        session
            .apply(
                3,
                &[Request::Trim {
                    segment_id: "a".into(),
                    edge: TrimEdge::Start,
                    time: Micros(100_000),
                }],
            )
            .unwrap();
        assert_eq!(
            session.draft().tracks[1].segments[0].target,
            TimeRange::new(100_000, 200_000)
        );
        assert_eq!(session.draft().materials.drafts, nested);
    }

    #[test]
    fn removing_the_owner_removes_a_dependent_that_falls_outside() {
        let (mut session, main, over) = open();
        session
            .apply(
                0,
                &[
                    place(&main, "a", 0, 200_000),
                    place(&over, "b", 50_000, 100_000),
                ],
            )
            .unwrap();
        session
            .apply(
                1,
                &[Request::Link {
                    owner: "a".into(),
                    dependent: "b".into(),
                }],
            )
            .unwrap();
        session
            .apply(
                2,
                &[Request::Trim {
                    segment_id: "a".into(),
                    edge: TrimEdge::End,
                    time: Micros(50_000),
                }],
            )
            .unwrap();
        assert!(session.draft().tracks[1].segments.is_empty());
        assert!(session.draft().relationships[0].id_to_id.is_empty());

        let (mut session, main, over) = open();
        session
            .apply(
                0,
                &[
                    place(&main, "a", 0, 1_000_000),
                    place(&over, "b", 0, 200_000),
                    Request::Link {
                        owner: "a".into(),
                        dependent: "b".into(),
                    },
                ],
            )
            .unwrap();
        session
            .apply(
                1,
                &[Request::Remove {
                    segment_id: "a".into(),
                }],
            )
            .unwrap();
        assert!(session.draft().tracks[0].segments.is_empty());
        assert!(session.draft().tracks[1].segments.is_empty());
        assert!(session.draft().relationships[0].id_to_id.is_empty());
    }

    #[test]
    fn the_cut_switch_splits_a_dependent_at_the_owners_time() {
        let (mut session, main, over) = open();
        session
            .apply(
                0,
                &[
                    place(&main, "a", 0, 1_000_000),
                    place(&over, "b", 0, 1_000_000),
                    Request::Link {
                        owner: "a".into(),
                        dependent: "b".into(),
                    },
                ],
            )
            .unwrap();
        session
            .apply(
                1,
                &[Request::Split {
                    segment_id: "a".into(),
                    at: Micros(400_000),
                    right_id: "a-right".into(),
                }],
            )
            .unwrap();
        assert_eq!(
            session.draft().tracks[1].segments[0].target,
            TimeRange::new(0, 400_000)
        );
        assert_eq!(
            session.draft().tracks[1].segments[1].target,
            TimeRange::new(400_000, 600_000)
        );
        assert_eq!(
            session.draft().relationships[0].id_to_id,
            vec![pair("a", "b"), pair("a-right", "b+a-right")]
        );

        let (mut session, main, over) = open();
        session
            .apply(
                0,
                &[
                    place(&main, "a", 0, 1_000_000),
                    place(&over, "b", 0, 1_000_000),
                    Request::Link {
                        owner: "a".into(),
                        dependent: "b".into(),
                    },
                ],
            )
            .unwrap();
        let mut settings = session.linkage();
        settings.cut = KindSwitches {
            overlay: false,
            ..KindSwitches::ALL
        };
        session.set_linkage(settings);
        let before = session.draft().clone();
        session
            .apply(
                1,
                &[Request::Split {
                    segment_id: "a".into(),
                    at: Micros(400_000),
                    right_id: "a-right".into(),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().tracks[1].segments.len(), 1);
        assert_eq!(
            session.draft().tracks[1].segments[0].target,
            before.tracks[1].segments[0].target
        );
        assert_eq!(
            session.draft().relationships[0].id_to_id,
            vec![pair("a", "b")]
        );
    }

    #[test]
    fn a_duration_that_scales_to_zero_leaves_the_draft() {
        let (mut session, main, over) = open();
        let mut owner = Segment::new("a", "media", TimeRange::new(0, 1_000_000));
        owner.source_range = Some(TimeRange::new(0, 1_000_000));
        session
            .apply(
                0,
                &[
                    Request::Place {
                        track_id: main,
                        segment: Box::new(owner),
                    },
                    place(&over, "b", 0, 1),
                    Request::Link {
                        owner: "a".into(),
                        dependent: "b".into(),
                    },
                ],
            )
            .unwrap();
        let before = session.draft().clone();
        let error = session
            .apply(
                1,
                &[Request::SetConstantSpeed {
                    segment_id: "a".into(),
                    speed: 1_000_000.0,
                }],
            )
            .unwrap_err();
        assert_eq!(
            error,
            EditError::Range(cutlass_model::RangeError::NonPositiveDuration)
        );
        assert_eq!(session.draft(), &before);
    }
}

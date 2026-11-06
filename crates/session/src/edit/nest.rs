//! Compound clips: several clips become one clip that plays a draft of
//! their own, and a compound clip can be ungrouped back onto the timeline.
//!
//! The inside draft keeps each clip's row, its gap to the first clip, and
//! the materials the clips use. Ungrouping puts the part the compound clip
//! plays back where it shows, keeping ids that are free and renaming those
//! that are taken.

use std::collections::BTreeSet;

use cutlass_model::{
    Draft, MaterialDraft, MaterialId, MaterialRef, Materials, Micros, Segment, SegmentId,
    TimeRange, Track, TrackId,
};

use super::linkage::drop_pairs_where;
use super::locate;
use super::speed::curve_points;
use super::{EditError, ensure_unlocked};

/// What a new compound clip is called.
pub const COMPOUND_NAME: &str = "Compound clip";

/// The ids of the materials `segments` use.
fn used(segments: &[&Segment]) -> BTreeSet<String> {
    let mut ids = BTreeSet::new();
    for segment in segments {
        ids.insert(segment.material_id.as_str().to_owned());
        for id in &segment.extra_material_refs {
            ids.insert(id.as_str().to_owned());
        }
    }
    ids
}

fn end(range: &TimeRange) -> i64 {
    range.start.0.saturating_add(range.duration.0)
}

/// Clips on one row may touch but not overlap.
fn ensure_apart(track: &Track) -> Result<(), EditError> {
    for pair in track.segments.windows(2) {
        if end(&pair[0].target) > pair[1].target.start.0 {
            return Err(EditError::Overlap);
        }
    }
    Ok(())
}

/// Make the clips `segment_ids` one compound clip `segment_id`, which
/// plays the draft material `material_id`. It sits on the lowest row of
/// the clips, from the first one's start to the last one's end.
pub fn create_compound(
    draft: &mut Draft,
    segment_ids: &[SegmentId],
    segment_id: &SegmentId,
    material_id: &MaterialId,
) -> Result<(), EditError> {
    if segment_ids.is_empty() {
        return Err(EditError::MissingSegment);
    }
    if segment_id.is_empty() || locate(draft, segment_id).is_ok() {
        return Err(EditError::DuplicateSegment);
    }
    if material_id.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    if draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    let mut host = None;
    for id in segment_ids {
        let (track_index, _) = locate(draft, id)?;
        ensure_unlocked(draft.tracks[track_index].attribute)?;
        host = Some(host.map_or(track_index, |host: usize| host.min(track_index)));
    }
    let host = host.ok_or(EditError::MissingSegment)?;
    let picked: Vec<&Segment> = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .filter(|segment| segment_ids.contains(&segment.id))
        .collect();
    let start = picked
        .iter()
        .map(|segment| segment.target.start.0)
        .min()
        .ok_or(EditError::MissingSegment)?;
    let finish = picked
        .iter()
        .map(|segment| end(&segment.target))
        .max()
        .ok_or(EditError::MissingSegment)?;
    let render_index = draft.tracks[host]
        .segments
        .iter()
        .filter(|segment| segment_ids.contains(&segment.id))
        .map(|segment| segment.render_index)
        .min()
        .unwrap_or_default();
    let keep = used(&picked);

    let mut inside = draft.clone();
    inside.id = material_id.as_str().into();
    inside.name = COMPOUND_NAME.to_owned();
    inside.relationships.clear();
    // The clips keep their gaps inside, even on the main row.
    inside.config.maintrack_adsorb = false;
    inside.duration = Micros(finish - start);
    inside.tracks = draft
        .tracks
        .iter()
        .enumerate()
        .filter(|(_, track)| track.segments.iter().any(|s| segment_ids.contains(&s.id)))
        .map(|(_, track)| {
            let mut row = track.clone();
            row.segments
                .retain(|segment| segment_ids.contains(&segment.id));
            for segment in &mut row.segments {
                segment.target.start = Micros(segment.target.start.0 - start);
            }
            row
        })
        .collect();
    // The host row goes first, so ungrouping puts it back on the host.
    if let Some(at) = inside
        .tracks
        .iter()
        .position(|track| track.id == draft.tracks[host].id)
    {
        let row = inside.tracks.remove(at);
        inside.tracks.insert(0, row);
    }
    inside.materials.retain(|id| keep.contains(id));

    drop_pairs_where(draft, |pair| {
        segment_ids.contains(&pair.owner) || segment_ids.contains(&pair.dependent)
    });
    for track in &mut draft.tracks {
        track
            .segments
            .retain(|segment| !segment_ids.contains(&segment.id));
    }
    let mut compound = Segment::new(
        segment_id.clone(),
        material_id.clone(),
        TimeRange::new(start, finish - start),
    );
    compound.source_range = Some(TimeRange::new(0, finish - start));
    compound.render_index = render_index;
    let row = &mut draft.tracks[host];
    row.segments.push(compound);
    row.segments.sort_by_key(|segment| segment.target.start.0);
    ensure_apart(row)?;
    draft.materials.drafts.push(MaterialDraft {
        id: material_id.clone(),
        name: COMPOUND_NAME.to_owned(),
        combination_type: "compound".to_owned(),
        draft: Some(Box::new(inside)),
        ..MaterialDraft::default()
    });
    Ok(())
}

/// The draft a compound clip plays.
fn inside_of(draft: &Draft, segment: &Segment) -> Option<Draft> {
    match draft.materials.get(segment.material_id.as_str())? {
        MaterialRef::Draft(material) => material.draft.as_deref().cloned(),
        _ => None,
    }
}

/// `child` cut to the window `from..to` of the inside timeline and moved
/// by `shift`. `None` when it is outside the window.
fn cut_to(
    draft: &Draft,
    child: &Segment,
    from: i64,
    to: i64,
    shift: i64,
) -> Result<Option<Segment>, EditError> {
    let (start, finish) = (child.target.start.0, end(&child.target));
    let (left, right) = (start.max(from), finish.min(to));
    if right <= left {
        return Ok(None);
    }
    let mut piece = child.clone();
    if (left, right) != (start, finish) {
        if curve_points(draft, child).is_some() {
            return Err(EditError::Compound);
        }
        if let Some(source) = piece.source_range.as_mut() {
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
            let scaled = |micros: i64| (micros as f64 * child.speed).round() as i64;
            source.start = Micros(source.start.0 + scaled(left - start));
            source.duration = Micros(scaled(right - left));
        }
    }
    piece.target = TimeRange::new(left + shift, right - left);
    Ok(Some(piece))
}

/// Put the clips a compound clip `segment_id` shows back on the timeline
/// in its place. Its inside main row goes on its row; another row goes
/// back on the row of its id when that is still here and has room, and on
/// a new row just above otherwise. A compound clip played at another speed is refused.
pub fn ungroup(draft: &mut Draft, segment_id: &SegmentId) -> Result<(), EditError> {
    let (host, index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[host].attribute)?;
    let compound = draft.tracks[host].segments[index].clone();
    let inside = inside_of(draft, &compound).ok_or(EditError::Compound)?;
    if (compound.speed - 1.0).abs() > 1e-9 || curve_points(draft, &compound).is_some() {
        return Err(EditError::Compound);
    }
    let window = compound
        .source_range
        .clone()
        .unwrap_or_else(|| TimeRange::new(0, compound.target.duration.0));
    let (from, to) = (window.start.0, end(&window));
    let shift = compound.target.start.0 - from;

    draft.tracks[host].segments.remove(index);
    let taken = |draft: &Draft, id: &SegmentId| locate(draft, id).is_ok();
    let mut rows: Vec<(usize, Vec<Segment>)> = Vec::new();
    let mut kept: Vec<Segment> = Vec::new();
    for (row_index, row) in inside.tracks.iter().enumerate() {
        let mut pieces = Vec::new();
        for child in &row.segments {
            if let Some(mut piece) = cut_to(&inside, child, from, to, shift)? {
                if taken(draft, &piece.id) {
                    piece.id = SegmentId::new(format!("{segment_id}-{}", piece.id));
                    if taken(draft, &piece.id) {
                        return Err(EditError::DuplicateSegment);
                    }
                }
                pieces.push(piece);
            }
        }
        kept.extend(pieces.iter().cloned());
        rows.push((row_index, pieces));
    }

    // Each material the clips use comes along; one whose id is taken here
    // by another material is renamed.
    let keep = used(&kept.iter().collect::<Vec<_>>());
    let mut materials = inside.materials.clone();
    materials.retain(|id| keep.contains(id));
    let mut renamed: Vec<(String, MaterialId)> = Vec::new();
    for id in &keep {
        if draft.materials.get(id).is_none() {
            continue;
        }
        if one_material(&draft.materials, id) == one_material(&materials, id) {
            materials.remove(id);
            continue;
        }
        let new_id = MaterialId::new(format!("{segment_id}-{id}"));
        if draft.materials.get(new_id.as_str()).is_some() || !materials.copy_as(id, &new_id) {
            return Err(EditError::DuplicateMaterial);
        }
        materials.remove(id);
        renamed.push((id.clone(), new_id));
    }
    let rename = |segment: &mut Segment| {
        for (old, new) in &renamed {
            if segment.material_id.as_str() == old {
                segment.material_id = new.clone();
            }
            for id in &mut segment.extra_material_refs {
                if id.as_str() == old {
                    *id = new.clone();
                }
            }
        }
    };
    draft.materials.absorb(materials);

    let mut above = host + 1;
    for (row_index, mut pieces) in rows {
        pieces.iter_mut().for_each(rename);
        let row = &inside.tracks[row_index];
        if row_index == 0 && row.kind == draft.tracks[host].kind {
            let target = &mut draft.tracks[host];
            target.segments.extend(pieces);
            target
                .segments
                .sort_by_key(|segment| segment.target.start.0);
            ensure_apart(target)?;
            continue;
        }
        if pieces.is_empty() {
            continue;
        }
        // Back on its own row when that row is still here and has room.
        if let Some(own) = draft
            .tracks
            .iter_mut()
            .find(|track| track.id == row.id && track.kind == row.kind)
        {
            let mut tried = own.segments.clone();
            tried.extend(pieces.iter().cloned());
            tried.sort_by_key(|segment| segment.target.start.0);
            let fits = tried
                .windows(2)
                .all(|pair| end(&pair[0].target) <= pair[1].target.start.0);
            if fits && own.attribute.contains(cutlass_model::TrackAttribute::LOCK) {
                return Err(EditError::LockedTrack);
            }
            if fits {
                own.segments = tried;
                continue;
            }
        }
        let id = TrackId::new(format!("{segment_id}-{}", row.id));
        if draft.tracks.iter().any(|track| track.id == id) {
            return Err(EditError::TrackKind);
        }
        let mut track = Track {
            id,
            segments: pieces,
            ..row.clone()
        };
        track.segments.sort_by_key(|segment| segment.target.start.0);
        ensure_apart(&track)?;
        draft.tracks.insert(above, track);
        above += 1;
    }
    Ok(())
}

/// Where the last clip of `draft` ends.
fn length_of(draft: &Draft) -> i64 {
    draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .map(|segment| end(&segment.target))
        .max()
        .unwrap_or_default()
}

/// Store `inside` as the draft the compound material `material_id` plays.
/// A compound clip that showed the whole inside follows its new length
/// where its row has room; one cut shorter keeps its cut, ending no later
/// than the inside does.
pub fn set_inside(
    draft: &mut Draft,
    material_id: &MaterialId,
    inside: &Draft,
) -> Result<(), EditError> {
    let Some(material) = draft
        .materials
        .drafts
        .iter_mut()
        .find(|material| &material.id == material_id)
    else {
        return Err(EditError::Compound);
    };
    let old = material.draft.as_deref().map_or(0, length_of);
    let mut inside = inside.clone();
    let new = length_of(&inside);
    inside.duration = Micros(new);
    material.draft = Some(Box::new(inside));
    for track_index in 0..draft.tracks.len() {
        let mut row = draft.tracks[track_index].clone();
        let mut changed = false;
        for segment in &mut row.segments {
            if &segment.material_id != material_id {
                continue;
            }
            let window = segment
                .source_range
                .clone()
                .unwrap_or_else(|| TimeRange::new(0, segment.target.duration.0));
            let whole = window.start.0 == 0 && window.duration.0 == old;
            let length = if whole {
                new
            } else {
                window.duration.0.min(new - window.start.0)
            };
            if length <= 0 {
                return Err(EditError::SourceBounds);
            }
            if length == window.duration.0 {
                continue;
            }
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
            let shown = (length as f64 / segment.speed).round() as i64;
            segment.source_range = Some(TimeRange::new(window.start.0, length));
            segment.target.duration = Micros(shown.max(1));
            changed = true;
        }
        if !changed {
            continue;
        }
        row.segments.sort_by_key(|segment| segment.target.start.0);
        if ensure_apart(&row).is_ok() {
            draft.tracks[track_index] = row;
        }
    }
    Ok(())
}

/// The material `id` alone, written out, to compare two pools' copies.
fn one_material(materials: &Materials, id: &str) -> Option<serde_json::Value> {
    let mut one = materials.clone();
    one.retain(|other| other == id);
    serde_json::to_value(one).ok()
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, MaterialAudio, MaterialRef, MaterialVideo, Micros, Segment, SegmentId, TimeRange,
        Track, TrackType,
    };

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    fn clip(id: &str, material: &str, start: i64, length: i64) -> Segment {
        let mut segment = Segment::new(id, material, TimeRange::new(start, length));
        segment.source_range = Some(TimeRange::new(1_000_000, length));
        segment
    }

    /// Two clips on the main row, a sound under the second, and a clip
    /// left out.
    fn session() -> Session {
        let mut draft = Draft::empty("D");
        for id in ["v1", "v2"] {
            let mut video = MaterialVideo::new(id);
            video.kind = "video".to_owned();
            video.duration = Micros(20_000_000);
            draft.materials.videos.push(video);
        }
        draft.materials.audios.push(MaterialAudio {
            id: "song".into(),
            kind: "music".to_owned(),
            duration: Micros(20_000_000),
            ..MaterialAudio::default()
        });
        draft.tracks[0].segments = vec![
            clip("a", "v1", 0, 2_000_000),
            clip("b", "v2", 2_000_000, 3_000_000),
            clip("c", "v1", 5_000_000, 1_000_000),
        ];
        let mut sound = Track::main("music");
        sound.kind = TrackType::Audio;
        sound.flag = cutlass_model::TrackFlag::default();
        sound.segments = vec![clip("s", "song", 3_000_000, 1_000_000)];
        draft.tracks.push(sound);
        Session::open(draft)
    }

    fn create() -> Request {
        Request::CreateCompound {
            segment_ids: vec!["a".into(), "b".into(), "s".into()],
            segment_id: "nest".into(),
            material_id: "inside".into(),
        }
    }

    fn ids(track: &Track) -> Vec<(String, i64, i64)> {
        track
            .segments
            .iter()
            .map(|s| {
                (
                    s.id.as_str().to_owned(),
                    s.target.start.0,
                    s.target.duration.0,
                )
            })
            .collect()
    }

    #[test]
    fn clips_become_one_compound_clip_and_ungroup_puts_them_back() {
        let mut session = session();
        let before = session.draft().clone();
        session.apply(0, &[create()]).unwrap();
        let draft = session.draft();
        assert_eq!(
            ids(&draft.tracks[0]),
            [
                ("nest".into(), 0, 5_000_000),
                ("c".into(), 5_000_000, 1_000_000)
            ]
        );
        // The sound row the compound emptied goes in the same step.
        assert_eq!(draft.tracks.len(), 1);
        let Some(MaterialRef::Draft(material)) = draft.materials.get("inside") else {
            panic!("no compound material");
        };
        let inside = material.draft.as_deref().unwrap();
        assert_eq!(inside.duration, Micros(5_000_000));
        assert_eq!(inside.tracks.len(), 2);
        assert_eq!(
            ids(&inside.tracks[0]),
            [
                ("a".into(), 0, 2_000_000),
                ("b".into(), 2_000_000, 3_000_000)
            ]
        );
        assert_eq!(ids(&inside.tracks[1]), [("s".into(), 3_000_000, 1_000_000)]);
        assert_eq!(inside.materials.videos.len(), 2);
        assert_eq!(inside.materials.audios.len(), 1);
        assert!(inside.materials.drafts.is_empty());
        // One undo takes it back.
        session.undo().unwrap();
        assert_eq!(session.draft(), &before);
        session.redo().unwrap();

        session
            .apply(
                session.revision(),
                &[Request::Ungroup {
                    segment_id: "nest".into(),
                }],
            )
            .unwrap();
        let draft = session.draft();
        assert_eq!(ids(&draft.tracks[0]), ids(&before.tracks[0]));
        assert_eq!(ids(&draft.tracks[1]), ids(&before.tracks[1]));
        assert_eq!(draft.tracks.len(), 2);
        // The same materials come back without copies.
        assert_eq!(draft.materials.videos.len(), 2);
        assert_eq!(draft.materials.audios.len(), 1);
    }

    #[test]
    fn a_trimmed_compound_clip_ungroups_only_the_part_it_shows() {
        let mut session = session();
        session.apply(0, &[create()]).unwrap();
        // Show 1s to 4s of the inside, from 0 on the timeline.
        let mut draft = session.draft().clone();
        let nest = &mut draft.tracks[0].segments[0];
        nest.source_range = Some(TimeRange::new(1_000_000, 3_000_000));
        nest.target = TimeRange::new(0, 3_000_000);
        draft.tracks[0].segments[1].target.start = Micros(3_000_000);
        let mut session = Session::open(draft);
        session
            .apply(
                0,
                &[Request::Ungroup {
                    segment_id: "nest".into(),
                }],
            )
            .unwrap();
        let draft = session.draft();
        assert_eq!(
            ids(&draft.tracks[0]),
            [
                ("a".into(), 0, 1_000_000),
                ("b".into(), 1_000_000, 2_000_000),
                ("c".into(), 3_000_000, 1_000_000)
            ]
        );
        // Each piece starts later in its source by what was cut.
        let a = &draft.tracks[0].segments[0];
        assert_eq!(a.source_range, Some(TimeRange::new(2_000_000, 1_000_000)));
        assert_eq!(ids(&draft.tracks[1]), [("s".into(), 2_000_000, 1_000_000)]);
    }

    #[test]
    fn ids_taken_meanwhile_are_renamed_and_other_rows_get_their_own() {
        let mut session = session();
        session
            .apply(
                0,
                &[Request::CreateCompound {
                    segment_ids: vec!["b".into(), "s".into()],
                    segment_id: "nest".into(),
                    material_id: "inside".into(),
                }],
            )
            .unwrap();
        // A new clip named b goes on another row. The sound row the compound
        // emptied is already gone.
        let mut draft = session.draft().clone();
        assert_eq!(draft.tracks.len(), 1);
        let mut other = Track::main("top");
        other.flag = cutlass_model::TrackFlag::default();
        other.segments = vec![clip("b", "v1", 9_000_000, 1_000_000)];
        draft.tracks.push(other);
        let mut session = Session::open(draft);
        session
            .apply(
                0,
                &[Request::Ungroup {
                    segment_id: "nest".into(),
                }],
            )
            .unwrap();
        let draft = session.draft();
        let names: Vec<String> = draft.tracks[0]
            .segments
            .iter()
            .map(|s| s.id.as_str().to_owned())
            .collect();
        assert_eq!(names, ["a", "nest-b", "c"]);
        assert_eq!(draft.tracks[1].id.as_str(), "nest-music");
        assert_eq!(ids(&draft.tracks[1]), [("s".into(), 3_000_000, 1_000_000)]);
        assert_eq!(draft.tracks[1].kind, TrackType::Audio);
    }

    #[test]
    fn an_edited_inside_is_stored_and_the_whole_clip_follows_its_length() {
        let mut session = session();
        session.apply(0, &[create()]).unwrap();
        let inside = |session: &Session| {
            let Some(MaterialRef::Draft(material)) = session.draft().materials.get("inside") else {
                panic!("no compound material");
            };
            material.draft.as_deref().unwrap().clone()
        };
        let mut shorter = inside(&session);
        shorter.tracks[0].segments.truncate(1);
        shorter.tracks[1].segments.clear();
        session
            .apply(
                1,
                &[Request::SetInside {
                    material_id: "inside".into(),
                    inside: Box::new(shorter.clone()),
                }],
            )
            .unwrap();
        assert_eq!(inside(&session).tracks[0].segments.len(), 1);
        assert_eq!(inside(&session).duration, Micros(2_000_000));
        let nest = &session.draft().tracks[0].segments[0];
        assert_eq!(nest.target.duration, Micros(2_000_000));
        assert_eq!(nest.source_range, Some(TimeRange::new(0, 2_000_000)));
        session.undo().unwrap();
        assert_eq!(
            session.draft().tracks[0].segments[0].target.duration,
            Micros(5_000_000)
        );
        assert_eq!(
            session
                .apply(
                    session.revision(),
                    &[Request::SetInside {
                        material_id: "v1".into(),
                        inside: Box::new(shorter),
                    }],
                )
                .unwrap_err(),
            EditError::Compound
        );
    }

    #[test]
    fn plain_clips_fast_compound_clips_and_missing_clips_are_refused() {
        let mut session = session();
        let ungroup = |id: &str| Request::Ungroup {
            segment_id: SegmentId::new(id),
        };
        assert_eq!(
            session.apply(0, &[ungroup("a")]).unwrap_err(),
            EditError::Compound
        );
        assert_eq!(
            session
                .apply(
                    0,
                    &[Request::CreateCompound {
                        segment_ids: vec!["a".into(), "gone".into()],
                        segment_id: "nest".into(),
                        material_id: "inside".into(),
                    }],
                )
                .unwrap_err(),
            EditError::MissingSegment
        );
        session.apply(0, &[create()]).unwrap();
        session
            .apply(
                1,
                &[Request::SetConstantSpeed {
                    segment_id: "nest".into(),
                    speed: 2.0,
                }],
            )
            .unwrap();
        assert_eq!(
            session.apply(2, &[ungroup("nest")]).unwrap_err(),
            EditError::Compound
        );
        // The id is taken by the compound clip itself.
        assert_eq!(
            session.apply(2, &[create()]).unwrap_err(),
            EditError::DuplicateSegment
        );
    }
}

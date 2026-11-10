use cutlass_model::{Draft, MaterialId, MaterialRef, MaterialTransition, SegmentId};

use super::locate;
use super::{EditError, ensure_unlocked};

/// Store a blend on the outgoing segment. Target ranges are not moved.
pub fn set_transition(
    draft: &mut Draft,
    segment_id: &SegmentId,
    transition: MaterialTransition,
) -> Result<(), EditError> {
    if transition.id.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let existing = transition_on_segment(draft, track_index, segment_index);
    if let Some(existing_id) = existing {
        if existing_id != transition.id && material_taken(draft, &transition.id) {
            return Err(EditError::DuplicateMaterial);
        }
        let Some(slot) = draft
            .materials
            .transitions
            .iter_mut()
            .find(|item| item.id == existing_id)
        else {
            return Err(EditError::MissingMaterial);
        };
        *slot = transition.clone();
        if existing_id != transition.id {
            let refs = &mut draft.tracks[track_index].segments[segment_index].extra_material_refs;
            if let Some(reference) = refs.iter_mut().find(|id| *id == &existing_id) {
                *reference = transition.id;
            }
        }
        return Ok(());
    }
    if material_taken(draft, &transition.id) {
        return Err(EditError::DuplicateMaterial);
    }
    let id = transition.id.clone();
    draft.materials.transitions.push(transition);
    draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .push(id);
    Ok(())
}

/// Take the blend off the outgoing segment, and its material off the
/// draft. A segment with no blend is refused.
pub fn remove_transition(draft: &mut Draft, segment_id: &SegmentId) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let Some(id) = transition_on_segment(draft, track_index, segment_index) else {
        return Err(EditError::MissingMaterial);
    };
    lift(draft, track_index, segment_index, &id);
    Ok(())
}

fn lift(draft: &mut Draft, track_index: usize, segment_index: usize, id: &MaterialId) {
    draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .retain(|reference| reference != id);
    draft.materials.transitions.retain(|item| &item.id != id);
}

/// Whether a clip on `track` starts where the clip at `segment_index`
/// ends, so the two meet at a cut.
fn meets_next(track: &cutlass_model::Track, segment_index: usize) -> bool {
    let end = track.segments[segment_index].target.end();
    track
        .segments
        .iter()
        .enumerate()
        .any(|(index, other)| index != segment_index && other.target.start == end)
}

/// The outgoing segments whose blend sits on a cut now: each has a blend
/// and a clip on its track starts where it ends. Empty at no cost when the
/// draft has no blend.
pub(super) fn on_cuts(draft: &Draft) -> Vec<SegmentId> {
    if draft.materials.transitions.is_empty() {
        return Vec::new();
    }
    let mut found = Vec::new();
    for (track_index, track) in draft.tracks.iter().enumerate() {
        for segment_index in 0..track.segments.len() {
            if transition_on_segment(draft, track_index, segment_index).is_some()
                && meets_next(track, segment_index)
            {
                found.push(track.segments[segment_index].id.clone());
            }
        }
    }
    found
}

/// Lift the blend of each segment of `were` that no longer meets a clip
/// at its end: the two clips were pulled apart, or the incoming one went.
/// A blend that sat on no cut before the edit is left as it was stored.
pub(super) fn drop_parted(draft: &mut Draft, were: &[SegmentId]) {
    for segment_id in were {
        let Ok((track_index, segment_index)) = locate(draft, segment_id) else {
            continue;
        };
        if meets_next(&draft.tracks[track_index], segment_index) {
            continue;
        }
        if let Some(id) = transition_on_segment(draft, track_index, segment_index) {
            lift(draft, track_index, segment_index, &id);
        }
    }
}

fn transition_on_segment(
    draft: &Draft,
    track_index: usize,
    segment_index: usize,
) -> Option<MaterialId> {
    draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .iter()
        .find(|material_id| {
            matches!(
                draft.materials.get(material_id.as_str()),
                Some(MaterialRef::Transition(_))
            )
        })
        .cloned()
}

fn material_taken(draft: &Draft, material_id: &MaterialId) -> bool {
    draft.materials.get(material_id.as_str()).is_some()
}

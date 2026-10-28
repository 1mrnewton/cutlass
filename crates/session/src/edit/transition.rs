use cutlass_model::{Draft, MaterialId, MaterialRef, MaterialTransition, SegmentId};

use super::speed::locate;
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

use cutlass_model::{Draft, MaterialAudioFade, MaterialId, MaterialRef, Micros, SegmentId};

use super::speed::locate;
use super::{EditError, ensure_unlocked};

/// Store fade durations. The segment volume stays as it was.
pub fn set_fade(
    draft: &mut Draft,
    segment_id: &SegmentId,
    fade_id: &MaterialId,
    fade_in: Micros,
    fade_out: Micros,
) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let existing = draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .iter()
        .find(|material_id| {
            matches!(
                draft.materials.get(material_id.as_str()),
                Some(MaterialRef::Fade(_))
            )
        })
        .cloned();
    if let Some(existing_id) = existing {
        let Some(fade) = draft
            .materials
            .audio_fades
            .iter_mut()
            .find(|fade| fade.id == existing_id)
        else {
            return Err(EditError::MissingMaterial);
        };
        fade.fade_in_duration = fade_in;
        fade.fade_out_duration = fade_out;
        return Ok(());
    }
    if fade_id.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    if draft.materials.get(fade_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    draft.materials.audio_fades.push(MaterialAudioFade {
        id: fade_id.clone(),
        fade_in_duration: fade_in,
        fade_out_duration: fade_out,
        ..MaterialAudioFade::default()
    });
    draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .push(fade_id.clone());
    Ok(())
}

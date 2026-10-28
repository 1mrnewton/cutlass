use cutlass_model::{Draft, MaterialText, SegmentId};

use super::speed::locate;
use super::{EditError, ensure_unlocked};

/// Replace the text material's string. Word columns stay as stored.
pub fn set_text(draft: &mut Draft, segment_id: &SegmentId, content: &str) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let material_id = draft.tracks[track_index].segments[segment_index]
        .material_id
        .clone();
    if material_id.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    if let Some(text) = draft
        .materials
        .texts
        .iter_mut()
        .find(|text| text.id == material_id)
    {
        text.content = content.to_owned();
        return Ok(());
    }
    if draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    draft.materials.texts.push(MaterialText {
        id: material_id,
        content: content.to_owned(),
        ..MaterialText::default()
    });
    Ok(())
}

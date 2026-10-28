use cutlass_model::{Crop, Draft, MaterialVideo, SegmentId};

use super::speed::locate;
use super::{EditError, ensure_unlocked};

/// Store the crop quad. Ratio and scale stay as they were.
pub fn set_crop(draft: &mut Draft, segment_id: &SegmentId, crop: Crop) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let material_id = draft.tracks[track_index].segments[segment_index]
        .material_id
        .clone();
    if material_id.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    if let Some(video) = draft
        .materials
        .videos
        .iter_mut()
        .find(|video| video.id == material_id)
    {
        let ratio = video.crop_ratio.clone();
        let scale = video.crop_scale;
        video.crop = crop;
        video.crop_ratio = ratio;
        video.crop_scale = scale;
        return Ok(());
    }
    if draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    draft.materials.videos.push(MaterialVideo {
        id: material_id,
        crop,
        ..MaterialVideo::default()
    });
    Ok(())
}

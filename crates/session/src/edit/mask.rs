//! A clip's mask (see `cutlass_effects::mask`).

use cutlass_effects::{MASK, Mask};
use cutlass_model::{Draft, MaterialId, MaterialRef, RawMaterial, SegmentId, TrackType};

use super::{EditError, ensure_unlocked};

/// Mask `segment_id` with `mask`, or take its mask off with `None`. A mask
/// another clip shares is left to that clip; this clip gets its own, named
/// `material_id`.
pub fn set_mask(
    draft: &mut Draft,
    segment_id: &SegmentId,
    material_id: &MaterialId,
    mask: Option<&Mask>,
) -> Result<(), EditError> {
    if mask.is_some_and(|mask| !mask.is_valid()) {
        return Err(EditError::Mask);
    }
    let (track_index, segment_index) = super::speed::locate(draft, segment_id)?;
    let track = &draft.tracks[track_index];
    ensure_unlocked(track.attribute)?;
    if !matches!(track.kind, TrackType::Video | TrackType::Sticker) {
        return Err(EditError::EffectTarget);
    }
    let current = draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .iter()
        .find(|id| {
            matches!(
                draft.materials.get(id.as_str()),
                Some(MaterialRef::Other(raw)) if raw.kind == MASK
            )
        })
        .cloned();
    if let Some(id) = current {
        let shared = draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .filter(|segment| segment.extra_material_refs.contains(&id))
            .count()
            > 1;
        if !shared {
            match mask {
                None => {
                    draft.tracks[track_index].segments[segment_index]
                        .extra_material_refs
                        .retain(|other| other != &id);
                    draft.materials.common_mask.retain(|raw| raw.id != id);
                }
                Some(mask) => {
                    if let Some(raw) = draft
                        .materials
                        .common_mask
                        .iter_mut()
                        .find(|raw| raw.id == id)
                    {
                        raw.fields = mask.to_fields();
                    }
                }
            }
            return Ok(());
        }
        draft.tracks[track_index].segments[segment_index]
            .extra_material_refs
            .retain(|other| other != &id);
    }
    let Some(mask) = mask else {
        return Ok(());
    };
    if material_id.is_empty() || draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    draft.materials.common_mask.push(RawMaterial {
        id: material_id.clone(),
        kind: MASK.to_owned(),
        fields: mask.to_fields(),
    });
    draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .push(material_id.clone());
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_effects::{Mask, MaskShape, segment_mask};
    use cutlass_model::{Draft, MaterialId, Segment, SegmentId, TimeRange};

    use super::set_mask;
    use crate::edit::EditError;

    #[test]
    fn a_mask_is_set_changed_and_taken_off() {
        let mut draft = Draft::empty("D");
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "m", TimeRange::new(0, 1_000_000)));
        let a = SegmentId::new("a");
        let circle = Mask::new(MaskShape::Circle);
        set_mask(&mut draft, &a, &MaterialId::new("k"), Some(&circle)).unwrap();
        assert_eq!(
            segment_mask(&draft, &draft.tracks[0].segments[0]),
            Some(circle)
        );
        let star = Mask {
            rotation: 30.0,
            ..Mask::new(MaskShape::Star)
        };
        set_mask(&mut draft, &a, &MaterialId::new("k2"), Some(&star)).unwrap();
        assert_eq!(
            segment_mask(&draft, &draft.tracks[0].segments[0]),
            Some(star)
        );
        assert_eq!(draft.materials.common_mask.len(), 1);
        let bad = Mask {
            feather: 2.0,
            ..star
        };
        assert_eq!(
            set_mask(&mut draft, &a, &MaterialId::new("k3"), Some(&bad)),
            Err(EditError::Mask)
        );
        set_mask(&mut draft, &a, &MaterialId::new("k3"), None).unwrap();
        assert!(draft.materials.common_mask.is_empty());
        assert!(draft.tracks[0].segments[0].extra_material_refs.is_empty());
    }
}

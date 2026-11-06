//! A clip's mask (see `cutlass_effects::mask`).

use cutlass_effects::{COLOR_MASK, MASK, Mask, SELECTION, Selection};
use cutlass_model::{Draft, MaterialId, MaterialRef, RawMaterial, SegmentId, TrackType};
use serde_json::{Map, Value};

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
    set_listed(
        draft,
        segment_id,
        material_id,
        shape_fields(mask)?,
        MASK,
        &[TrackType::Video, TrackType::Sticker],
    )
}

/// Limit `segment_id`'s colour work to `mask`, or take that limit off with
/// `None`. On an adjustment layer the mask is where on the canvas the
/// layer acts.
pub fn set_color_mask(
    draft: &mut Draft,
    segment_id: &SegmentId,
    material_id: &MaterialId,
    mask: Option<&Mask>,
) -> Result<(), EditError> {
    set_listed(
        draft,
        segment_id,
        material_id,
        shape_fields(mask)?,
        COLOR_MASK,
        &[TrackType::Video, TrackType::Sticker, TrackType::Adjust],
    )
}

/// Cut `segment_id`, a video clip, down to `selection`, or take its
/// selection off with `None`. The selection sits beside any shape mask.
pub fn set_selection(
    draft: &mut Draft,
    segment_id: &SegmentId,
    material_id: &MaterialId,
    selection: Option<&Selection>,
) -> Result<(), EditError> {
    if selection.is_some_and(|selection| !selection.is_valid()) {
        return Err(EditError::Mask);
    }
    set_listed(
        draft,
        segment_id,
        material_id,
        selection.map(Selection::to_fields),
        SELECTION,
        &[TrackType::Video],
    )
}

fn shape_fields(mask: Option<&Mask>) -> Result<Option<Map<String, Value>>, EditError> {
    match mask {
        Some(mask) if !mask.is_valid() => Err(EditError::Mask),
        mask => Ok(mask.map(Mask::to_fields)),
    }
}

fn set_listed(
    draft: &mut Draft,
    segment_id: &SegmentId,
    material_id: &MaterialId,
    fields: Option<Map<String, Value>>,
    kind: &str,
    tracks: &[TrackType],
) -> Result<(), EditError> {
    let (track_index, segment_index) = super::locate(draft, segment_id)?;
    let track = &draft.tracks[track_index];
    ensure_unlocked(track.attribute)?;
    if !tracks.contains(&track.kind) {
        return Err(EditError::EffectTarget);
    }
    let current = draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .iter()
        .find(|id| {
            matches!(
                draft.materials.get(id.as_str()),
                Some(MaterialRef::Other(raw)) if raw.kind == kind
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
            match fields {
                None => {
                    draft.tracks[track_index].segments[segment_index]
                        .extra_material_refs
                        .retain(|other| other != &id);
                    draft.materials.common_mask.retain(|raw| raw.id != id);
                }
                Some(fields) => {
                    if let Some(raw) = draft
                        .materials
                        .common_mask
                        .iter_mut()
                        .find(|raw| raw.id == id)
                    {
                        raw.fields = fields;
                    }
                }
            }
            return Ok(());
        }
        draft.tracks[track_index].segments[segment_index]
            .extra_material_refs
            .retain(|other| other != &id);
    }
    let Some(fields) = fields else {
        return Ok(());
    };
    if material_id.is_empty() || draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    draft.materials.common_mask.push(RawMaterial {
        id: material_id.clone(),
        kind: kind.to_owned(),
        fields,
    });
    draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .push(material_id.clone());
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_effects::{Mask, MaskShape, segment_color_mask, segment_mask};
    use cutlass_model::{Draft, MaterialId, Segment, SegmentId, TimeRange};

    use super::{set_color_mask, set_mask};
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
        // A colour mask sits beside the clip's mask and goes on its own.
        set_color_mask(&mut draft, &a, &MaterialId::new("c"), Some(&circle)).unwrap();
        let segment = &draft.tracks[0].segments[0];
        assert_eq!(segment_color_mask(&draft, segment), Some(circle));
        assert_eq!(segment_mask(&draft, segment), Some(star));
        set_mask(&mut draft, &a, &MaterialId::new("k3"), None).unwrap();
        assert_eq!(draft.materials.common_mask.len(), 1);
        set_color_mask(&mut draft, &a, &MaterialId::new("c2"), None).unwrap();
        assert!(draft.materials.common_mask.is_empty());
        assert!(draft.tracks[0].segments[0].extra_material_refs.is_empty());
    }

    fn selection() -> cutlass_effects::Selection {
        cutlass_effects::Selection::from_alpha(
            4,
            2,
            &[0, 255, 255, 0, 0, 255, 255, 0],
            cutlass_model::Micros(40_000),
        )
        .unwrap()
    }

    fn one_clip() -> Draft {
        let mut draft = Draft::empty("D");
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "m", TimeRange::new(0, 1_000_000)));
        draft
    }

    #[test]
    fn a_selection_is_set_changed_and_taken_off_beside_a_shape_mask() {
        use cutlass_effects::segment_selection;

        let mut draft = one_clip();
        let a = SegmentId::new("a");
        let circle = Mask::new(MaskShape::Circle);
        set_mask(&mut draft, &a, &MaterialId::new("k"), Some(&circle)).unwrap();
        super::set_selection(&mut draft, &a, &MaterialId::new("s"), Some(&selection())).unwrap();
        let segment = &draft.tracks[0].segments[0];
        assert_eq!(segment_selection(&draft, segment), Some(selection()));
        assert_eq!(segment_mask(&draft, segment), Some(circle));

        let inverted = cutlass_effects::Selection {
            invert: true,
            ..selection()
        };
        super::set_selection(&mut draft, &a, &MaterialId::new("s2"), Some(&inverted)).unwrap();
        assert_eq!(
            segment_selection(&draft, &draft.tracks[0].segments[0]),
            Some(inverted)
        );
        assert_eq!(draft.materials.common_mask.len(), 2);

        super::set_selection(&mut draft, &a, &MaterialId::new("s3"), None).unwrap();
        let segment = &draft.tracks[0].segments[0];
        assert_eq!(segment_selection(&draft, segment), None);
        assert_eq!(segment_mask(&draft, segment), Some(circle));
        assert_eq!(draft.materials.common_mask.len(), 1);
    }

    #[test]
    fn a_bad_selection_or_target_leaves_the_draft() {
        let mut draft = one_clip();
        let a = SegmentId::new("a");
        let before = draft.clone();
        let short = cutlass_effects::Selection {
            runs: vec![1, 2],
            ..selection()
        };
        assert_eq!(
            super::set_selection(&mut draft, &a, &MaterialId::new("s"), Some(&short)),
            Err(EditError::Mask)
        );
        assert!(
            super::set_selection(
                &mut draft,
                &SegmentId::new("missing"),
                &MaterialId::new("s"),
                Some(&selection())
            )
            .is_err()
        );
        // The material id is already a material.
        assert_eq!(
            super::set_selection(&mut draft, &a, &MaterialId::new("m"), Some(&selection())),
            Ok(())
        );
        super::set_selection(&mut draft, &a, &MaterialId::new("x"), None).unwrap();
        assert_eq!(draft, before);
        assert_eq!(
            super::set_selection(&mut draft, &a, &MaterialId::new(""), Some(&selection())),
            Err(EditError::DuplicateMaterial)
        );
        assert_eq!(draft, before);
        draft.tracks[0].attribute =
            cutlass_model::TrackAttribute(cutlass_model::TrackAttribute::LOCK);
        let locked = draft.clone();
        assert_eq!(
            super::set_selection(&mut draft, &a, &MaterialId::new("s"), Some(&selection())),
            Err(EditError::LockedTrack)
        );
        assert_eq!(draft, locked);
    }

    #[test]
    fn a_selection_goes_on_video_clips_only() {
        let mut draft = one_clip();
        draft.tracks[0].kind = cutlass_model::TrackType::Audio;
        let before = draft.clone();
        assert_eq!(
            super::set_selection(
                &mut draft,
                &SegmentId::new("a"),
                &MaterialId::new("s"),
                Some(&selection())
            ),
            Err(EditError::EffectTarget)
        );
        assert_eq!(draft, before);
    }

    #[test]
    fn the_request_reads_from_json() {
        let request: crate::Request = serde_json::from_value(serde_json::json!({
            "op": "set_selection",
            "segment_id": "a",
            "material_id": "s",
            "selection": {"width": 2, "height": 1, "runs": [1, 1], "frame": 0, "invert": false}
        }))
        .unwrap();
        let mut draft = one_clip();
        crate::edit::apply_request(&mut draft, &request).unwrap();
        assert!(cutlass_effects::segment_has_selection(
            &draft,
            &draft.tracks[0].segments[0]
        ));
    }
}

//! A clip's LUT: a `.cube` table kept on an effect material of type `lut`,
//! with a strength (see `cutlass_effects::lut`).

use cutlass_effects::{LUT, Lut};
use cutlass_model::{Draft, MaterialEffect, MaterialId, MaterialRef, SegmentId, TrackType};

use super::{EditError, ensure_unlocked};

/// Put a LUT on `segment_id`. `cube` is the text of a `.cube` file, read
/// and kept on the clip under `name`; `None` keeps the table the clip has
/// and changes its strength, from 0 to 1; an empty text takes it off.
pub fn set_lut(
    draft: &mut Draft,
    segment_id: &SegmentId,
    material_id: &MaterialId,
    name: &str,
    cube: Option<&str>,
    strength: f64,
) -> Result<(), EditError> {
    if !(0.0..=1.0).contains(&strength) {
        return Err(EditError::EffectStrength);
    }
    let table = match cube {
        Some("") | None => None,
        Some(text) => Some(Lut::parse_cube(text).map_err(|_| EditError::Lut)?),
    };
    let (track_index, segment_index) = super::locate(draft, segment_id)?;
    let track = &draft.tracks[track_index];
    ensure_unlocked(track.attribute)?;
    if matches!(track.kind, TrackType::Audio | TrackType::Effect) {
        return Err(EditError::EffectTarget);
    }
    let current = draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .iter()
        .find(|id| {
            matches!(
                draft.materials.get(id.as_str()),
                Some(MaterialRef::Effect(effect)) if effect.kind == LUT
            )
        })
        .cloned();
    let kept = current.as_ref().and_then(|id| {
        draft
            .materials
            .effects
            .iter()
            .find(|effect| &effect.id == id)
            .cloned()
    });
    if cube.is_none() && kept.is_none() {
        return Err(EditError::Lut);
    }
    // Take the clip's LUT off this clip; a material no one else lists goes.
    if let Some(id) = &current {
        draft.tracks[track_index].segments[segment_index]
            .extra_material_refs
            .retain(|other| other != id);
        let listed = draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .any(|segment| segment.extra_material_refs.contains(id));
        if !listed {
            draft.materials.effects.retain(|effect| &effect.id != id);
        }
    }
    if cube == Some("") {
        return Ok(());
    }
    if material_id.is_empty() || draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    let effect = match (table, kept) {
        (Some(table), _) => MaterialEffect {
            kind: LUT.to_owned(),
            name: name.to_owned(),
            value: Some(strength),
            adjust_params: Some(table.to_params()),
            ..MaterialEffect::new(material_id.clone())
        },
        (None, Some(kept)) => MaterialEffect {
            id: material_id.clone(),
            value: Some(strength),
            ..kept
        },
        (None, None) => return Err(EditError::Lut),
    };
    draft.materials.effects.push(effect);
    draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .push(material_id.clone());
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_effects::segment_lut;
    use cutlass_model::{Draft, MaterialId, Segment, SegmentId, TimeRange};

    use super::set_lut;
    use crate::edit::EditError;

    const SWAP: &str = "LUT_3D_SIZE 2\n0 0 0\n0 1 0\n1 0 0\n1 1 0\n0 0 1\n0 1 1\n1 0 1\n1 1 1\n";

    fn draft() -> Draft {
        let mut draft = Draft::empty("D");
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "m", TimeRange::new(0, 1_000_000)));
        draft
    }

    #[test]
    fn a_lut_is_loaded_its_strength_changed_and_taken_off() {
        let mut draft = draft();
        let a = SegmentId::new("a");
        set_lut(
            &mut draft,
            &a,
            &MaterialId::new("l"),
            "Swap",
            Some(SWAP),
            1.0,
        )
        .unwrap();
        let lut = segment_lut(&draft, &draft.tracks[0].segments[0]).unwrap();
        assert_eq!(lut.name, "Swap");
        let out = lut.lut.sample([1.0, 0.0, 0.0]);
        assert!(out[1] > 0.99 && out[0] < 0.01, "{out:?}");
        set_lut(&mut draft, &a, &MaterialId::new("l2"), "", None, 0.25).unwrap();
        let lut = segment_lut(&draft, &draft.tracks[0].segments[0]).unwrap();
        assert_eq!(lut.strength, 0.25);
        assert_eq!(lut.name, "Swap");
        assert_eq!(draft.materials.effects.len(), 1);
        set_lut(&mut draft, &a, &MaterialId::new("l3"), "", Some(""), 1.0).unwrap();
        assert!(segment_lut(&draft, &draft.tracks[0].segments[0]).is_none());
        assert!(draft.materials.effects.is_empty());
    }

    #[test]
    fn a_bad_file_or_a_strength_with_no_table_is_refused() {
        let mut draft = draft();
        let a = SegmentId::new("a");
        let id = MaterialId::new("l");
        assert_eq!(
            set_lut(
                &mut draft,
                &a,
                &id,
                "x",
                Some("LUT_3D_SIZE 2\n0 0 0\n"),
                1.0
            ),
            Err(EditError::Lut)
        );
        assert_eq!(
            set_lut(&mut draft, &a, &id, "x", None, 0.5),
            Err(EditError::Lut)
        );
        assert_eq!(
            set_lut(&mut draft, &a, &id, "x", Some(SWAP), 2.0),
            Err(EditError::EffectStrength)
        );
        assert!(draft.materials.effects.is_empty());
    }
}

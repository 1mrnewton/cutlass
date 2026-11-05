//! A clip's chroma key (see `cutlass_effects::chroma`).

use cutlass_effects::{CHROMA, parse_hex};
use cutlass_model::{Draft, MaterialId, MaterialRef, RawMaterial, SegmentId, TrackType};
use serde_json::json;

use super::{EditError, ensure_unlocked};

/// Key `color` (`#rrggbb`) out of `segment_id` with `strength` and
/// `shadow`, each from 0 to 1. An empty colour takes the key off. A key
/// another clip shares is left to that clip; this clip gets its own,
/// named `material_id`.
pub fn set_chroma(
    draft: &mut Draft,
    segment_id: &SegmentId,
    material_id: &MaterialId,
    color: &str,
    strength: f64,
    shadow: f64,
) -> Result<(), EditError> {
    if !color.is_empty() && parse_hex(color).is_none() {
        return Err(EditError::CanvasColor);
    }
    if !(0.0..=1.0).contains(&strength) || !(0.0..=1.0).contains(&shadow) {
        return Err(EditError::EffectStrength);
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
                Some(MaterialRef::Other(raw)) if raw.kind == CHROMA
            )
        })
        .cloned();
    let fields = |raw: &mut RawMaterial| {
        raw.fields
            .insert("color".to_owned(), json!(color.to_ascii_lowercase()));
        raw.fields.insert("strength".to_owned(), json!(strength));
        raw.fields.insert("shadow".to_owned(), json!(shadow));
    };
    if let Some(id) = current {
        let shared = draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .filter(|segment| segment.extra_material_refs.contains(&id))
            .count()
            > 1;
        if !shared {
            if color.is_empty() {
                draft.tracks[track_index].segments[segment_index]
                    .extra_material_refs
                    .retain(|other| other != &id);
                draft.materials.chromas.retain(|raw| raw.id != id);
            } else if let Some(raw) = draft.materials.chromas.iter_mut().find(|raw| raw.id == id) {
                fields(raw);
            }
            return Ok(());
        }
        draft.tracks[track_index].segments[segment_index]
            .extra_material_refs
            .retain(|other| other != &id);
    }
    if color.is_empty() {
        return Ok(());
    }
    if material_id.is_empty() || draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    let mut raw = RawMaterial {
        id: material_id.clone(),
        kind: CHROMA.to_owned(),
        ..RawMaterial::default()
    };
    fields(&mut raw);
    draft.materials.chromas.push(raw);
    draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .push(material_id.clone());
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_effects::segment_chroma;
    use cutlass_model::{Draft, MaterialId, Segment, SegmentId, TimeRange, Track, TrackType};

    use super::set_chroma;
    use crate::edit::EditError;

    fn draft() -> Draft {
        let mut draft = Draft::empty("D");
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "m", TimeRange::new(0, 1_000_000)));
        draft
    }

    #[test]
    fn a_key_is_set_changed_and_taken_off() {
        let mut draft = draft();
        let a = SegmentId::new("a");
        set_chroma(&mut draft, &a, &MaterialId::new("k"), "#00FF00", 0.5, 0.2).unwrap();
        let key = segment_chroma(&draft, &draft.tracks[0].segments[0]).unwrap();
        assert_eq!(key.hex(), "#00ff00");
        set_chroma(&mut draft, &a, &MaterialId::new("k2"), "#0000ff", 0.7, 0.0).unwrap();
        let key = segment_chroma(&draft, &draft.tracks[0].segments[0]).unwrap();
        assert_eq!(key.hex(), "#0000ff");
        assert!((key.strength - 0.7).abs() < 1e-6);
        assert_eq!(draft.materials.chromas.len(), 1);
        set_chroma(&mut draft, &a, &MaterialId::new("k3"), "", 0.0, 0.0).unwrap();
        assert!(segment_chroma(&draft, &draft.tracks[0].segments[0]).is_none());
        assert!(draft.materials.chromas.is_empty());
    }

    #[test]
    fn bad_colours_values_and_rows_are_refused() {
        let mut draft = draft();
        let a = SegmentId::new("a");
        let id = MaterialId::new("k");
        assert_eq!(
            set_chroma(&mut draft, &a, &id, "green", 0.5, 0.0),
            Err(EditError::CanvasColor)
        );
        assert_eq!(
            set_chroma(&mut draft, &a, &id, "#00ff00", 1.5, 0.0),
            Err(EditError::EffectStrength)
        );
        let mut words = Track::main("t");
        words.kind = TrackType::Text;
        words
            .segments
            .push(Segment::new("t", "m", TimeRange::new(0, 1)));
        draft.tracks.push(words);
        assert_eq!(
            set_chroma(&mut draft, &SegmentId::new("t"), &id, "#00ff00", 0.5, 0.0),
            Err(EditError::EffectTarget)
        );
    }
}

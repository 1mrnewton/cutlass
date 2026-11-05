//! A clip's adjust sliders: exposure, contrast, temperature and the rest.
//!
//! Each slider is an effect material the clip lists among its extra
//! materials, with the slider's reference name as its `type` and the setting
//! in `value`. A slider at zero has no material.

use cutlass_effects::AdjustKey;
use cutlass_model::{Draft, MaterialEffect, MaterialId, MaterialRef, SegmentId, TrackType};

use super::{EditError, ensure_unlocked};

/// Set the slider named `key` (such as `temperature` or `light_sensation`)
/// on `segment_id` to `value`, from -1 to 1 (0 to 1 for fade, sharpen and
/// grain). Zero takes the slider off. A new material is named
/// `material_id`; a material another clip shares is left to that clip.
pub fn set_adjust(
    draft: &mut Draft,
    segment_id: &SegmentId,
    material_id: &MaterialId,
    key: &str,
    value: f64,
) -> Result<(), EditError> {
    let key = AdjustKey::from_kind(key).ok_or(EditError::UnknownEffect)?;
    if !value.is_finite() || value < f64::from(key.min()) || value > 1.0 {
        return Err(EditError::EffectStrength);
    }
    let (track_index, segment_index) = super::speed::locate(draft, segment_id)?;
    let track = &draft.tracks[track_index];
    ensure_unlocked(track.attribute)?;
    if matches!(
        track.kind,
        TrackType::Audio | TrackType::Effect | TrackType::Filter
    ) {
        return Err(EditError::EffectTarget);
    }
    let current = draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .iter()
        .find(|id| {
            matches!(
                draft.materials.get(id.as_str()),
                Some(MaterialRef::Effect(effect)) if effect.kind == key.kind()
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
            if value == 0.0 {
                draft.tracks[track_index].segments[segment_index]
                    .extra_material_refs
                    .retain(|other| other != &id);
                draft.materials.effects.retain(|effect| effect.id != id);
            } else if let Some(effect) = draft
                .materials
                .effects
                .iter_mut()
                .find(|effect| effect.id == id)
            {
                effect.value = Some(value);
            }
            return Ok(());
        }
        draft.tracks[track_index].segments[segment_index]
            .extra_material_refs
            .retain(|other| other != &id);
    }
    if value == 0.0 {
        return Ok(());
    }
    if material_id.is_empty() || draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    draft.materials.effects.push(MaterialEffect {
        kind: key.kind().to_owned(),
        name: key.label().to_owned(),
        value: Some(value),
        ..MaterialEffect::new(material_id.clone())
    });
    draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .push(material_id.clone());
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_effects::{AdjustKey, segment_adjust};
    use cutlass_model::{
        Draft, MaterialId, Segment, SegmentId, TimeRange, Track, TrackAttribute, TrackType,
    };

    use super::set_adjust;
    use crate::edit::EditError;

    fn draft() -> Draft {
        let mut draft = Draft::empty("D");
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "m", TimeRange::new(0, 1_000_000)));
        draft
    }

    fn value(draft: &Draft, key: AdjustKey) -> f32 {
        segment_adjust(draft, &draft.tracks[0].segments[0]).get(key)
    }

    #[test]
    fn a_slider_is_added_changed_and_taken_off() {
        let mut draft = draft();
        let a = SegmentId::new("a");
        set_adjust(&mut draft, &a, &MaterialId::new("t"), "temperature", 0.5).unwrap();
        assert_eq!(value(&draft, AdjustKey::Temperature), 0.5);
        set_adjust(&mut draft, &a, &MaterialId::new("u"), "temperature", -0.25).unwrap();
        assert_eq!(value(&draft, AdjustKey::Temperature), -0.25);
        assert_eq!(draft.materials.effects.len(), 1);
        set_adjust(&mut draft, &a, &MaterialId::new("c"), "contrast", 0.1).unwrap();
        set_adjust(&mut draft, &a, &MaterialId::new("v"), "temperature", 0.0).unwrap();
        assert_eq!(value(&draft, AdjustKey::Temperature), 0.0);
        assert_eq!(draft.materials.effects.len(), 1);
        assert_eq!(draft.tracks[0].segments[0].extra_material_refs.len(), 1);
        set_adjust(&mut draft, &a, &MaterialId::new("w"), "exposure", 0.0).unwrap();
        assert_eq!(draft.materials.effects.len(), 1);
    }

    #[test]
    fn a_shared_slider_is_copied_for_this_clip() {
        let mut draft = draft();
        let mut b = Segment::new("b", "m", TimeRange::new(1_000_000, 1_000_000));
        b.extra_material_refs.push(MaterialId::new("t"));
        draft.tracks[0].segments.push(b);
        let a = SegmentId::new("a");
        set_adjust(&mut draft, &a, &MaterialId::new("t"), "fade", 0.5).unwrap();
        set_adjust(&mut draft, &a, &MaterialId::new("t2"), "fade", 0.75).unwrap();
        assert_eq!(value(&draft, AdjustKey::Fade), 0.75);
        let other = segment_adjust(&draft, &draft.tracks[0].segments[1]);
        assert_eq!(other.get(AdjustKey::Fade), 0.5);
    }

    #[test]
    fn bad_sliders_values_and_clips_are_refused() {
        let mut draft = draft();
        let a = SegmentId::new("a");
        let id = MaterialId::new("x");
        assert!(matches!(
            set_adjust(&mut draft, &a, &id, "glow", 0.5),
            Err(EditError::UnknownEffect)
        ));
        assert!(matches!(
            set_adjust(&mut draft, &a, &id, "fade", -0.5),
            Err(EditError::EffectStrength)
        ));
        assert!(matches!(
            set_adjust(&mut draft, &a, &id, "hue", f64::NAN),
            Err(EditError::EffectStrength)
        ));
        assert!(matches!(
            set_adjust(&mut draft, &a, &MaterialId::new(""), "hue", 0.5),
            Err(EditError::DuplicateMaterial)
        ));
        let mut sound = Track::main("s");
        sound.kind = TrackType::Audio;
        sound
            .segments
            .push(Segment::new("s", "m", TimeRange::new(0, 1)));
        draft.tracks.push(sound);
        assert!(matches!(
            set_adjust(&mut draft, &SegmentId::new("s"), &id, "hue", 0.5),
            Err(EditError::EffectTarget)
        ));
        draft.tracks[0].attribute = TrackAttribute(TrackAttribute::LOCK);
        assert!(set_adjust(&mut draft, &a, &id, "hue", 0.5).is_err());
        assert!(draft.materials.effects.is_empty());
    }
}

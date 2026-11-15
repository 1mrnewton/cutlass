//! A clip's blend mode: how its picture mixes with the picture below.
//!
//! The mode is a `mix_mode` effect material the clip lists among its extra
//! materials, naming a blend package. Normal is no material at all.

use cutlass_effects::BlendMode;
use cutlass_model::{Draft, MaterialEffect, MaterialId, MaterialRef, SegmentId, TrackType};

use super::{EditError, ensure_unlocked};

/// Blend `segment_id` with `mode`, a blend package id such as
/// `blend-multiply`. Normal (`blend-source-over`) or an empty mode takes the
/// blend off. `material_id` names the new `mix_mode` material.
pub fn set_blend(
    draft: &mut Draft,
    segment_id: &SegmentId,
    material_id: &MaterialId,
    mode: &str,
) -> Result<(), EditError> {
    let mode = if mode.is_empty() {
        BlendMode::SourceOver
    } else {
        BlendMode::from_package(mode).ok_or(EditError::UnknownEffect)?
    };
    let (track_index, segment_index) = super::locate(draft, segment_id)?;
    let track = &draft.tracks[track_index];
    ensure_unlocked(track.attribute)?;
    if matches!(
        track.kind,
        TrackType::Audio | TrackType::Effect | TrackType::Filter
    ) {
        return Err(EditError::EffectTarget);
    }
    if mode != BlendMode::SourceOver
        && (material_id.is_empty() || draft.materials.get(material_id.as_str()).is_some())
    {
        return Err(EditError::DuplicateMaterial);
    }
    let old: Vec<MaterialId> = draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .iter()
        .filter(|id| {
            matches!(
                draft.materials.get(id.as_str()),
                Some(MaterialRef::Effect(effect)) if effect.kind == MaterialEffect::MIX_MODE
            )
        })
        .cloned()
        .collect();
    let segment = &mut draft.tracks[track_index].segments[segment_index];
    segment.extra_material_refs.retain(|id| !old.contains(id));
    for id in &old {
        let shared = draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .any(|segment| segment.extra_material_refs.contains(id));
        if !shared {
            draft.materials.effects.retain(|effect| &effect.id != id);
        }
    }
    if mode == BlendMode::SourceOver {
        return Ok(());
    }
    draft.materials.effects.push(MaterialEffect {
        kind: MaterialEffect::MIX_MODE.to_owned(),
        effect_id: mode.package_id().to_owned(),
        name: mode.label().to_owned(),
        value: Some(1.0),
        ..MaterialEffect::new(material_id.clone())
    });
    draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .push(material_id.clone());
    Ok(())
}

/// Turn the clip's Blend switch on or off. Off, it is drawn solid and mixed
/// normally; its opacity and mode stay for when it is on again.
pub fn set_blend_on(draft: &mut Draft, segment_id: &SegmentId, on: bool) -> Result<(), EditError> {
    let (track_index, segment_index) = super::locate(draft, segment_id)?;
    let track = &draft.tracks[track_index];
    ensure_unlocked(track.attribute)?;
    if matches!(
        track.kind,
        TrackType::Audio | TrackType::Effect | TrackType::Filter
    ) {
        return Err(EditError::EffectTarget);
    }
    draft.tracks[track_index].segments[segment_index].blend_off = !on;
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_effects::{BlendMode, segment_blend, stored_blend};
    use cutlass_model::{Draft, MaterialId, Micros, Segment, SegmentId, TimeRange, TrackAttribute};

    use super::{set_blend, set_blend_on};
    use crate::edit::EditError;

    fn draft() -> Draft {
        let mut draft = Draft::empty("D");
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "m", TimeRange::new(0, 1_000_000)));
        draft
    }

    fn blend(draft: &Draft) -> Option<BlendMode> {
        segment_blend(draft, &draft.tracks[0].segments[0])
    }

    #[test]
    fn a_mode_is_stored_replaced_and_taken_off() {
        let mut draft = draft();
        let a = SegmentId::new("a");
        set_blend(&mut draft, &a, &MaterialId::new("mix-1"), "blend-multiply").unwrap();
        assert_eq!(blend(&draft), Some(BlendMode::Multiply));
        set_blend(&mut draft, &a, &MaterialId::new("mix-2"), "blend-screen").unwrap();
        assert_eq!(blend(&draft), Some(BlendMode::Screen));
        assert_eq!(draft.materials.effects.len(), 1);
        assert_eq!(draft.tracks[0].segments[0].extra_material_refs.len(), 1);
        set_blend(&mut draft, &a, &MaterialId::new(""), "blend-source-over").unwrap();
        assert_eq!(blend(&draft), None);
        assert!(draft.materials.effects.is_empty());
        assert!(draft.tracks[0].segments[0].extra_material_refs.is_empty());
    }

    #[test]
    fn the_switch_off_draws_solid_and_normal_and_keeps_both() {
        let mut draft = draft();
        let a = SegmentId::new("a");
        set_blend(&mut draft, &a, &MaterialId::new("mix"), "blend-screen").unwrap();
        draft.tracks[0].segments[0].clip.alpha = 0.4;
        set_blend_on(&mut draft, &a, false).unwrap();
        let segment = &draft.tracks[0].segments[0];
        assert_eq!(blend(&draft), None);
        assert_eq!(stored_blend(&draft, segment), Some(BlendMode::Screen));
        assert_eq!(segment.clip.alpha, 0.4);
        let (clip, _) = cutlass_model::animated_at(&draft, segment, Micros(0)).unwrap();
        assert_eq!(clip.alpha, 1.0);
        // The switch is the app's own key, written only while it is off.
        let json = serde_json::to_value(segment).unwrap();
        assert_eq!(json["cutlass_blend_off"], true);
        set_blend_on(&mut draft, &a, true).unwrap();
        let segment = &draft.tracks[0].segments[0];
        assert_eq!(blend(&draft), Some(BlendMode::Screen));
        let (clip, _) = cutlass_model::animated_at(&draft, segment, Micros(0)).unwrap();
        assert_eq!(clip.alpha, 0.4);
        let json = serde_json::to_value(segment).unwrap();
        assert!(json.get("cutlass_blend_off").is_none());
        draft.tracks[0].attribute = TrackAttribute(TrackAttribute::LOCK);
        assert_eq!(
            set_blend_on(&mut draft, &a, false),
            Err(EditError::LockedTrack)
        );
        assert_eq!(
            set_blend_on(&mut draft, &SegmentId::new("z"), false),
            Err(EditError::MissingSegment)
        );
    }

    #[test]
    fn unknown_modes_taken_ids_and_locked_lanes_are_refused() {
        let mut draft = draft();
        let a = SegmentId::new("a");
        assert_eq!(
            set_blend(&mut draft, &a, &MaterialId::new("x"), "distort-invert"),
            Err(EditError::UnknownEffect)
        );
        set_blend(&mut draft, &a, &MaterialId::new("mix"), "blend-multiply").unwrap();
        assert_eq!(
            set_blend(&mut draft, &a, &MaterialId::new("mix"), "blend-screen"),
            Err(EditError::DuplicateMaterial)
        );
        assert_eq!(
            set_blend(&mut draft, &SegmentId::new("z"), &MaterialId::new("x"), ""),
            Err(EditError::MissingSegment)
        );
        draft.tracks[0].attribute = TrackAttribute(TrackAttribute::LOCK);
        assert_eq!(
            set_blend(&mut draft, &a, &MaterialId::new("x"), "blend-multiply"),
            Err(EditError::LockedTrack)
        );
    }
}

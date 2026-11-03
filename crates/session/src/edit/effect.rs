//! Place a bundled effect on an effect row, and set an effect's strength.

use cutlass_effects::bundled_recipe;
use cutlass_model::{
    Draft, MaterialEffect, MaterialId, MaterialRef, MaterialVideoEffect, Segment, SegmentId,
    TimeRange, TrackId, TrackType,
};

use super::{EditError, ensure_unlocked};

/// Put the bundled effect `effect_id` (such as `recipe/shake`) over
/// `target` on `track_id`, an effect row made when it is missing. The
/// effect runs over every clip drawn below it.
pub fn add_effect(
    draft: &mut Draft,
    track_id: &TrackId,
    segment_id: &SegmentId,
    material_id: &MaterialId,
    target: TimeRange,
    effect_id: &str,
) -> Result<(), EditError> {
    let Some(recipe) = bundled_recipe(effect_id) else {
        return Err(EditError::UnknownEffect);
    };
    if draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    if let Some(track) = draft.tracks.iter().find(|track| &track.id == track_id) {
        if !matches!(track.kind, TrackType::Effect) {
            return Err(EditError::TrackKind);
        }
        ensure_unlocked(track.attribute)?;
    } else {
        super::track::insert_track(draft, track_id, &TrackType::Effect)?;
    }
    let mut segment = Segment::new(segment_id.clone(), material_id.clone(), target);
    segment.render_index = TrackType::Effect.render_floor().unwrap_or(0);
    super::place::place(draft, track_id, segment)?;
    draft.materials.video_effects.push(MaterialVideoEffect {
        id: material_id.clone(),
        kind: "video_effect".to_owned(),
        effect_id: recipe.effect_id(),
        name: recipe.name.clone(),
        value: 1.0,
        ..MaterialVideoEffect::default()
    });
    Ok(())
}

/// Store `value`, from 0 to 1, as the strength of the effect, filter, or
/// adjustment `segment_id` uses. A material other clips also use is copied
/// to `material_id` first, so only this clip changes. A glow is a title's,
/// not a clip of its own, and is refused.
pub fn set_effect_strength(
    draft: &mut Draft,
    segment_id: &SegmentId,
    material_id: &MaterialId,
    value: f64,
) -> Result<(), EditError> {
    if !(0.0..=1.0).contains(&value) {
        return Err(EditError::EffectStrength);
    }
    let (track_index, segment_index) = super::speed::locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let current = draft.tracks[track_index].segments[segment_index]
        .material_id
        .clone();
    let video = match draft.materials.get(current.as_str()) {
        Some(MaterialRef::VideoEffect(_)) => true,
        Some(MaterialRef::Effect(effect)) if effect.kind != MaterialEffect::BLOOM => false,
        _ => return Err(EditError::NotEffect),
    };
    let users = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .filter(|segment| segment.material_id == current)
        .count();
    let target = if users > 1 {
        if material_id.is_empty() {
            return Err(EditError::MissingMaterial);
        }
        if draft.materials.get(material_id.as_str()).is_some() {
            return Err(EditError::DuplicateMaterial);
        }
        if video {
            let mut copy = draft
                .materials
                .video_effects
                .iter()
                .find(|effect| effect.id == current)
                .cloned()
                .ok_or(EditError::MissingMaterial)?;
            copy.id = material_id.clone();
            draft.materials.video_effects.push(copy);
        } else {
            let mut copy = draft
                .materials
                .effects
                .iter()
                .find(|effect| effect.id == current)
                .cloned()
                .ok_or(EditError::MissingMaterial)?;
            copy.id = material_id.clone();
            draft.materials.effects.push(copy);
        }
        draft.tracks[track_index].segments[segment_index].material_id = material_id.clone();
        material_id.clone()
    } else {
        current
    };
    if video {
        if let Some(effect) = draft
            .materials
            .video_effects
            .iter_mut()
            .find(|effect| effect.id == target)
        {
            effect.value = value;
        }
    } else if let Some(effect) = draft
        .materials
        .effects
        .iter_mut()
        .find(|effect| effect.id == target)
    {
        effect.value = Some(value);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, MaterialId, MaterialRef, Micros, SegmentId, TimeRange, TrackAttribute, TrackId,
        TrackType,
    };

    use crate::{EditError, Request, Session};

    fn open() -> Session {
        let mut draft = Draft::empty("draft");
        draft.tracks[0].id = TrackId::new("main");
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "a",
            "m",
            TimeRange::new(0, 4_000_000),
        ));
        Session::open(draft)
    }

    fn add(track: &str, segment: &str, start: i64, effect: &str) -> Request {
        Request::AddEffect {
            track_id: TrackId::new(track),
            segment_id: SegmentId::new(segment),
            material_id: MaterialId::new(format!("m-{segment}")),
            start: Micros(start),
            duration: Micros(2_000_000),
            effect_id: effect.to_owned(),
        }
    }

    #[test]
    fn a_new_effect_gets_its_row_and_material_in_one_undo_step() {
        let mut session = open();
        session
            .apply(0, &[add("fx", "shake", 500_000, "recipe/shake")])
            .unwrap();
        let draft = session.draft();
        let row = draft
            .tracks
            .iter()
            .find(|track| track.id.as_str() == "fx")
            .unwrap();
        assert_eq!(row.kind, TrackType::Effect);
        assert_eq!(row.segments[0].target, TimeRange::new(500_000, 2_000_000));
        assert_eq!(row.segments[0].render_index, 11_000);
        let Some(MaterialRef::VideoEffect(effect)) = draft.materials.get("m-shake") else {
            panic!("no effect material");
        };
        assert_eq!(effect.effect_id, "recipe/shake");
        assert_eq!(effect.name, "Shake");

        session.undo().unwrap();
        assert!(
            session
                .draft()
                .tracks
                .iter()
                .all(|track| track.id.as_str() != "fx")
        );
        assert!(session.draft().materials.video_effects.is_empty());
    }

    #[test]
    fn a_bad_effect_changes_nothing() {
        let mut session = open();
        let before = session.draft().clone();
        for (request, expected) in [
            (add("fx", "x", 0, "recipe/none"), EditError::UnknownEffect),
            (add("fx", "x", 0, "shake"), EditError::UnknownEffect),
            (
                add("fx", "x", 0, "../recipe/shake"),
                EditError::UnknownEffect,
            ),
            (add("main", "x", 0, "recipe/shake"), EditError::TrackKind),
            (
                add("fx", "a", 0, "recipe/shake"),
                EditError::DuplicateSegment,
            ),
        ] {
            assert_eq!(session.apply(0, &[request]).unwrap_err(), expected);
            assert_eq!(session.draft(), &before);
        }
        session
            .apply(0, &[add("fx", "x", 0, "recipe/vhs")])
            .unwrap();
        let after = session.draft().clone();
        assert_eq!(
            session
                .apply(1, &[add("fx", "y", 1_000_000, "recipe/vhs")])
                .unwrap_err(),
            EditError::Overlap
        );
        assert_eq!(session.draft(), &after);
    }

    #[test]
    fn a_locked_effect_row_is_refused() {
        let mut session = open();
        session
            .apply(0, &[add("fx", "x", 0, "recipe/vhs")])
            .unwrap();
        let mut draft = session.draft().clone();
        if let Some(row) = draft
            .tracks
            .iter_mut()
            .find(|track| track.id.as_str() == "fx")
        {
            row.attribute = TrackAttribute(TrackAttribute::LOCK);
        }
        let mut session = Session::open(draft);
        assert_eq!(
            session
                .apply(0, &[add("fx", "y", 3_000_000, "recipe/vhs")])
                .unwrap_err(),
            EditError::LockedTrack
        );
    }

    fn strength(segment: &str, material: &str, value: f64) -> Request {
        Request::SetEffectStrength {
            segment_id: SegmentId::new(segment),
            material_id: MaterialId::new(material),
            value,
        }
    }

    fn filter_session() -> Session {
        let mut draft = Draft::empty("draft");
        let mut filter = cutlass_model::MaterialEffect::new("warm");
        filter.kind = "filter".to_owned();
        draft.materials.effects.push(filter);
        let mut glow = cutlass_model::MaterialEffect::new("glow");
        glow.kind = cutlass_model::MaterialEffect::BLOOM.to_owned();
        draft.materials.effects.push(glow);
        let mut row = cutlass_model::Track::main("looks");
        row.kind = TrackType::Filter;
        for (id, material, start) in [
            ("left", "warm", 0),
            ("right", "warm", 1_000_000),
            ("bloom", "glow", 2_000_000),
        ] {
            row.segments.push(cutlass_model::Segment::new(
                id,
                material,
                TimeRange::new(start, 1_000_000),
            ));
        }
        draft.tracks.push(row);
        Session::open(draft)
    }

    #[test]
    fn a_strength_is_stored_on_the_clip_effect_in_one_undo_step() {
        let mut session = open();
        session
            .apply(0, &[add("fx", "shake", 0, "recipe/shake")])
            .unwrap();
        let before = session.draft().clone();
        session
            .apply(1, &[strength("shake", "unused", 0.25)])
            .unwrap();
        let Some(MaterialRef::VideoEffect(effect)) = session.draft().materials.get("m-shake")
        else {
            panic!("no effect material");
        };
        assert_eq!(effect.value, 0.25);
        assert!(session.draft().materials.get("unused").is_none());
        session.undo().unwrap();
        assert_eq!(session.draft(), &before);
    }

    #[test]
    fn a_shared_filter_is_copied_so_only_one_half_changes() {
        let mut session = filter_session();
        session
            .apply(0, &[strength("right", "warm-2", 0.8)])
            .unwrap();
        let draft = session.draft();
        let row = draft
            .tracks
            .iter()
            .find(|row| row.id.as_str() == "looks")
            .unwrap();
        assert_eq!(row.segments[0].material_id.as_str(), "warm");
        assert_eq!(row.segments[1].material_id.as_str(), "warm-2");
        let Some(MaterialRef::Effect(left)) = draft.materials.get("warm") else {
            panic!("no left filter");
        };
        let Some(MaterialRef::Effect(right)) = draft.materials.get("warm-2") else {
            panic!("no right filter");
        };
        assert_eq!(left.value, None);
        assert_eq!(right.value, Some(0.8));
        assert_eq!(right.kind, "filter");
        session
            .apply(1, &[strength("right", "warm-3", 0.4)])
            .unwrap();
        assert!(session.draft().materials.get("warm-3").is_none());
        session.undo().unwrap();
        session.undo().unwrap();
        assert!(session.draft().materials.get("warm-2").is_none());
    }

    #[test]
    fn a_bad_strength_changes_nothing() {
        let mut session = filter_session();
        let before = session.draft().clone();
        for (request, expected) in [
            (strength("left", "x", -0.1), EditError::EffectStrength),
            (strength("left", "x", 1.5), EditError::EffectStrength),
            (strength("left", "x", f64::NAN), EditError::EffectStrength),
            (strength("bloom", "x", 0.5), EditError::NotEffect),
            (strength("nowhere", "x", 0.5), EditError::MissingSegment),
            (strength("left", "", 0.5), EditError::MissingMaterial),
            (strength("left", "glow", 0.5), EditError::DuplicateMaterial),
        ] {
            assert_eq!(session.apply(0, &[request]).unwrap_err(), expected);
            assert_eq!(session.draft(), &before);
        }
        let mut draft = before.clone();
        if let Some(row) = draft
            .tracks
            .iter_mut()
            .find(|row| row.id.as_str() == "looks")
        {
            row.attribute = TrackAttribute(TrackAttribute::LOCK);
        }
        let mut session = Session::open(draft);
        assert_eq!(
            session.apply(0, &[strength("left", "x", 0.5)]).unwrap_err(),
            EditError::LockedTrack
        );
    }
}

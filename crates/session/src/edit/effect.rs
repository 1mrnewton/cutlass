//! Place a bundled effect on an effect row.

use cutlass_effects::bundled_recipe;
use cutlass_model::{
    Draft, MaterialId, MaterialVideoEffect, Segment, SegmentId, TimeRange, TrackId, TrackType,
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
}

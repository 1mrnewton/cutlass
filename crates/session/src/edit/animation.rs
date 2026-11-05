use cutlass_model::{
    ANIMATION_PRESETS, AnimType, Draft, MaterialAnimations, MaterialId, MaterialRef, Micros,
    Motion, SegmentId, StickerAnimation,
};

use super::locate;
use super::{EditError, ensure_unlocked};

/// The shortest animation the duration slider reaches.
pub const SHORTEST: Micros = Micros(100_000);

/// Set the clip's in or out animation to the preset with `resource_id`, or
/// clear it with `None`, lasting `duration`. The length is held between a
/// tenth of a second and what the clip has left beside its other
/// animation. The row is the reference editor's: one `sticker_animation`
/// in `material_animations` that the segment's `extra_material_refs` name,
/// dropped when it empties. An out animation starts where it must to end
/// with the clip. A letter animation goes only on a title. A loop plays
/// the whole clip, once each `duration`, beside the in and out.
pub fn set_animation(
    draft: &mut Draft,
    segment_id: &SegmentId,
    kind: AnimType,
    resource_id: Option<&str>,
    duration: Micros,
) -> Result<(), EditError> {
    if !matches!(kind, AnimType::In | AnimType::Out | AnimType::Loop) {
        return Err(EditError::Animation);
    }
    let preset = match resource_id {
        Some(id) => Some(
            ANIMATION_PRESETS
                .iter()
                .find(|preset| preset.resource_id == id && preset.kind == kind)
                .ok_or(EditError::Animation)?,
        ),
        None => None,
    };
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    if preset.is_some_and(|preset| matches!(preset.motion, Motion::Letters(_))) {
        let material_id = &draft.tracks[track_index].segments[segment_index].material_id;
        if !matches!(
            draft.materials.get(material_id.as_str()),
            Some(MaterialRef::Text(_))
        ) {
            return Err(EditError::Animation);
        }
    }
    let length = draft.tracks[track_index].segments[segment_index]
        .target
        .duration;
    let row_id = own_row(draft, track_index, segment_index)?;
    let row = draft
        .materials
        .material_animations
        .iter_mut()
        .find(|row| row.id == row_id)
        .ok_or(EditError::MissingMaterial)?;
    let name = match kind {
        AnimType::In => "in",
        AnimType::Out => "out",
        _ => "loop",
    };
    row.animations.retain(|animation| animation.kind != name);
    if let Some(preset) = preset {
        let other: i64 = row
            .animations
            .iter()
            .filter(|animation| matches!(animation.anim_type(), Some(AnimType::In | AnimType::Out)))
            .filter_map(|animation| animation.duration.map(|d| d.0))
            .sum();
        // A loop's length is one round, and it plays the whole clip.
        let room = if kind == AnimType::Loop {
            length.0.max(SHORTEST.0)
        } else {
            (length.0 - other).max(SHORTEST.0)
        };
        let duration = duration.0.clamp(SHORTEST.0, room).min(length.0.max(0));
        let start = if kind == AnimType::Out {
            length.0 - duration
        } else {
            0
        };
        row.animations.push(StickerAnimation {
            id: preset.resource_id.to_owned(),
            kind: name.to_owned(),
            start: Micros(start),
            duration: Some(Micros(duration)),
            name: preset.name.to_owned(),
            resource_id: preset.resource_id.to_owned(),
            platform: StickerAnimation::MAPPING_PLATFORM.to_owned(),
            source_platform: StickerAnimation::MAPPING_SOURCE_PLATFORM,
            category_name: "Basic".to_owned(),
            material_type: "video".to_owned(),
            panel: "video".to_owned(),
            ..StickerAnimation::default()
        });
        // In, out, then loop, as the reference editor writes them.
        row.animations
            .sort_by_key(|animation| match animation.kind.as_str() {
                "in" => 0,
                "out" => 1,
                _ => 2,
            });
    }
    if row.animations.is_empty() {
        draft
            .materials
            .material_animations
            .retain(|row| row.id != row_id);
        draft.tracks[track_index].segments[segment_index]
            .extra_material_refs
            .retain(|id| id != &row_id);
    }
    Ok(())
}

/// The clip's own animation row, made when it has none and copied when
/// another clip shares it, so an edit here leaves that clip alone.
fn own_row(
    draft: &mut Draft,
    track_index: usize,
    segment_index: usize,
) -> Result<MaterialId, EditError> {
    let segment = &draft.tracks[track_index].segments[segment_index];
    let current = segment.extra_material_refs.iter().find(|id| {
        draft
            .materials
            .material_animations
            .iter()
            .any(|row| &row.id == *id)
    });
    let shared = current.is_some_and(|id| uses(draft, id) > 1);
    if let Some(id) = current
        && !shared
    {
        return Ok(id.clone());
    }
    let id = MaterialId::new(format!("{}:animations", segment.id.as_str()));
    if draft.materials.get(id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    let row = match current {
        Some(old) => {
            let old = old.clone();
            let copy = draft
                .materials
                .material_animations
                .iter()
                .find(|row| row.id == old)
                .cloned()
                .ok_or(EditError::MissingMaterial)?;
            draft.tracks[track_index].segments[segment_index]
                .extra_material_refs
                .retain(|ref_id| ref_id != &old);
            MaterialAnimations {
                id: id.clone(),
                ..copy
            }
        }
        None => MaterialAnimations {
            id: id.clone(),
            kind: "sticker_animation".to_owned(),
            multi_language_current: "none".to_owned(),
            ..MaterialAnimations::default()
        },
    };
    draft.materials.material_animations.push(row);
    draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .push(id.clone());
    Ok(id)
}

fn uses(draft: &Draft, id: &MaterialId) -> usize {
    draft
        .tracks
        .iter()
        .flat_map(|track| track.segments.iter())
        .filter(|segment| segment.extra_material_refs.contains(id))
        .count()
}

#[cfg(test)]
mod tests {
    use cutlass_model::{AnimType, Draft, Micros, Segment, TimeRange, animated_at};

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    const FADE_IN: &str = "7649736866270350613";
    const FADE_OUT: &str = "6798320902548230669";

    fn open() -> Session {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = "main".into();
        let mut session = Session::open(draft);
        let segment = Segment::new("a", "media", TimeRange::new(0, 1_000_000));
        session
            .apply(
                0,
                &[Request::Place {
                    track_id: "main".into(),
                    segment: Box::new(segment),
                }],
            )
            .unwrap();
        session
    }

    fn set(kind: AnimType, animation: Option<&str>, duration: i64) -> Request {
        Request::SetAnimation {
            segment_id: "a".into(),
            kind,
            animation: animation.map(str::to_owned),
            duration: Micros(duration),
        }
    }

    fn alpha(session: &Session, time: i64) -> f64 {
        let draft = session.draft();
        animated_at(draft, &draft.tracks[0].segments[0], Micros(time))
            .unwrap()
            .0
            .alpha
    }

    #[test]
    fn an_in_and_an_out_share_one_row_that_goes_when_both_clear() {
        let mut session = open();
        let revision = session
            .apply(1, &[set(AnimType::In, Some(FADE_IN), 400_000)])
            .unwrap();
        let revision = session
            .apply(revision, &[set(AnimType::Out, Some(FADE_OUT), 900_000)])
            .unwrap();
        let draft = session.draft();
        let row = &draft.materials.material_animations[0];
        assert_eq!(row.kind, "sticker_animation");
        assert_eq!(
            draft.tracks[0].segments[0].extra_material_refs,
            std::slice::from_ref(&row.id)
        );
        let sides: Vec<(&str, i64, Option<Micros>)> = row
            .animations
            .iter()
            .map(|a| (a.kind.as_str(), a.start.0, a.duration))
            .collect();
        // The out gets what the in leaves, and ends with the clip.
        assert_eq!(
            sides,
            [
                ("in", 0, Some(Micros(400_000))),
                ("out", 400_000, Some(Micros(600_000)))
            ]
        );
        assert_eq!(alpha(&session, 0), 0.0);
        assert_eq!(alpha(&session, 400_000), 1.0);
        assert!(alpha(&session, 999_999) < 0.01);

        let revision = session
            .apply(revision, &[set(AnimType::In, None, 0)])
            .unwrap();
        session
            .apply(revision, &[set(AnimType::Out, None, 0)])
            .unwrap();
        let draft = session.draft();
        assert!(draft.materials.material_animations.is_empty());
        assert!(draft.tracks[0].segments[0].extra_material_refs.is_empty());
        session.undo().unwrap();
        assert_eq!(session.draft().materials.material_animations.len(), 1);
    }

    #[test]
    fn an_animation_on_the_wrong_side_or_unknown_is_refused() {
        let mut session = open();
        assert_eq!(
            session.apply(1, &[set(AnimType::Out, Some(FADE_IN), 500_000)]),
            Err(EditError::Animation)
        );
        assert_eq!(
            session.apply(1, &[set(AnimType::Caption, None, 500_000)]),
            Err(EditError::Animation)
        );
        // A letter animation is a title's; "media" is not one.
        assert_eq!(
            session.apply(
                1,
                &[set(
                    AnimType::In,
                    Some("cutlass-anim-in-typewriter"),
                    500_000
                )]
            ),
            Err(EditError::Animation)
        );
    }

    #[test]
    fn a_loop_plays_beside_the_in_and_out_over_the_whole_clip() {
        let mut session = open();
        session
            .apply(
                session.revision(),
                &[set(AnimType::In, Some(FADE_IN), 600_000)],
            )
            .unwrap();
        session
            .apply(
                session.revision(),
                &[set(
                    AnimType::Loop,
                    Some("cutlass-anim-loop-pulse"),
                    400_000,
                )],
            )
            .unwrap();
        session
            .apply(
                session.revision(),
                &[set(AnimType::Out, Some(FADE_OUT), 400_000)],
            )
            .unwrap();
        let draft = session.draft();
        let kinds: Vec<_> = draft.materials.material_animations[0]
            .animations
            .iter()
            .map(|animation| (animation.kind.as_str(), animation.duration))
            .collect();
        // The loop takes no room from the in and out.
        assert_eq!(
            kinds,
            [
                ("in", Some(Micros(600_000))),
                ("out", Some(Micros(400_000))),
                ("loop", Some(Micros(400_000))),
            ]
        );
        let segment = &draft.tracks[0].segments[0];
        let (clip, uniform) = animated_at(draft, segment, Micros(200_000)).unwrap();
        assert!(clip.scale.x > 1.0 || uniform.value > 1.0);
        session
            .apply(session.revision(), &[set(AnimType::Loop, None, 0)])
            .unwrap();
        assert_eq!(
            session.draft().materials.material_animations[0]
                .animations
                .len(),
            2
        );
    }

    #[test]
    fn a_title_takes_a_letter_animation_that_moves_its_letters_only() {
        let mut draft = open().draft().clone();
        draft
            .materials
            .texts
            .push(cutlass_model::MaterialText::new("media"));
        let mut session = Session::open(draft);
        session
            .apply(
                0,
                &[set(
                    AnimType::In,
                    Some("cutlass-anim-in-typewriter"),
                    500_000,
                )],
            )
            .unwrap();
        let draft = session.draft();
        let segment = &draft.tracks[0].segments[0];
        let steps = cutlass_model::letters_at(draft, segment, Micros(250_000));
        assert_eq!(steps.len(), 1);
        assert!((steps[0].along - 0.5).abs() < 1e-9);
        // The clip itself stays whole while its letters type in.
        assert_eq!(alpha(&session, 0), 1.0);
    }
}

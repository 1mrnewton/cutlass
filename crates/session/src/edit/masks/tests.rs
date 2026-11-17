use cutlass_effects::{
    COLOR_MASK, MASK, Mask, MaskMode, MaskShape, segment_color_masks, segment_masks,
};
use cutlass_model::{
    CommonKeyframe, CommonKeyframeGroup, Draft, MASK_KEYS, MaterialId, Micros, Segment, SegmentId,
    TimeRange, TrackAttribute, TrackType, mask_key,
};

use super::{MAX_MASKS, add_mask, edit_mask, move_mask, remove_mask};
use crate::Request;
use crate::edit::{EditError, apply_request, mask::set_mask};

fn one_clip() -> Draft {
    let mut draft = Draft::empty("D");
    draft.tracks[0]
        .segments
        .push(Segment::new("a", "m", TimeRange::new(0, 1_000_000)));
    draft
}

fn a() -> SegmentId {
    SegmentId::new("a")
}

fn id(name: &str) -> MaterialId {
    MaterialId::new(name)
}

fn shapes(draft: &Draft) -> Vec<(MaskShape, MaskMode)> {
    segment_masks(draft, &draft.tracks[0].segments[0])
        .iter()
        .map(|listed| (listed.mask.shape, listed.mask.mode))
        .collect()
}

fn ids(draft: &Draft) -> Vec<String> {
    segment_masks(draft, &draft.tracks[0].segments[0])
        .iter()
        .map(|listed| listed.id.as_str().to_owned())
        .collect()
}

fn mask(shape: MaskShape, mode: MaskMode) -> Mask {
    Mask {
        mode,
        ..Mask::new(shape)
    }
}

/// A key on `name` at the clip's start.
fn key(draft: &mut Draft, name: &str, value: f64) {
    draft.tracks[0].segments[0]
        .common_keyframes
        .push(CommonKeyframeGroup {
            property_type: name.to_owned(),
            keyframe_list: vec![CommonKeyframe {
                time_offset: Micros(0),
                values: vec![value],
                ..CommonKeyframe::default()
            }],
            ..CommonKeyframeGroup::default()
        });
}

/// The key names on the clip, sorted.
fn key_names(draft: &Draft) -> Vec<String> {
    let mut names: Vec<String> = draft.tracks[0].segments[0]
        .common_keyframes
        .iter()
        .map(|group| group.property_type.clone())
        .collect();
    names.sort();
    names
}

/// The first key value on `name`.
fn value(draft: &Draft, name: &str) -> Option<f64> {
    draft.tracks[0].segments[0]
        .common_keyframes
        .iter()
        .find(|group| group.property_type == name)
        .and_then(|group| group.keyframe_list.first())
        .and_then(|key| key.values.first().copied())
}

#[test]
fn masks_are_added_changed_moved_and_taken_off_in_order() {
    let mut draft = one_clip();
    let circle = mask(MaskShape::Circle, MaskMode::Add);
    let star = mask(MaskShape::Star, MaskMode::Subtract);
    let heart = mask(MaskShape::Heart, MaskMode::Intersect);
    add_mask(&mut draft, &a(), &id("m1"), &circle, None, false).unwrap();
    add_mask(&mut draft, &a(), &id("m2"), &star, None, false).unwrap();
    add_mask(&mut draft, &a(), &id("m3"), &heart, None, false).unwrap();
    assert_eq!(
        shapes(&draft),
        [
            (MaskShape::Circle, MaskMode::Add),
            (MaskShape::Star, MaskMode::Subtract),
            (MaskShape::Heart, MaskMode::Intersect),
        ]
    );
    // A colour mask is a list of its own beside them.
    add_mask(&mut draft, &a(), &id("c1"), &star, None, true).unwrap();
    assert_eq!(
        segment_color_masks(&draft, &draft.tracks[0].segments[0]).len(),
        1
    );
    assert_eq!(ids(&draft), ["m1", "m2", "m3"]);

    edit_mask(
        &mut draft,
        &a(),
        1,
        &id("unused"),
        &mask(MaskShape::Rectangle, MaskMode::Add),
        None,
        false,
    )
    .unwrap();
    assert_eq!(shapes(&draft)[1], (MaskShape::Rectangle, MaskMode::Add));
    assert_eq!(ids(&draft), ["m1", "m2", "m3"], "an own record is changed");

    move_mask(&mut draft, &a(), 2, 0, false).unwrap();
    assert_eq!(ids(&draft), ["m3", "m1", "m2"]);
    // The colour mask kept its place among the clip's records.
    let refs = &draft.tracks[0].segments[0].extra_material_refs;
    assert_eq!(refs[3], id("c1"));

    remove_mask(&mut draft, &a(), 1, false).unwrap();
    assert_eq!(ids(&draft), ["m3", "m2"]);
    assert!(draft.materials.get("m1").is_none(), "its record goes too");
    remove_mask(&mut draft, &a(), 0, true).unwrap();
    assert!(segment_color_masks(&draft, &draft.tracks[0].segments[0]).is_empty());
}

#[test]
fn keys_follow_their_mask_as_it_moves_comes_first_or_goes() {
    let mut draft = one_clip();
    for name in ["m1", "m2", "m3"] {
        add_mask(
            &mut draft,
            &a(),
            &id(name),
            &Mask::new(MaskShape::Circle),
            None,
            false,
        )
        .unwrap();
    }
    let x = MASK_KEYS[0];
    key(&mut draft, x, 1.0);
    key(&mut draft, &mask_key(x, Some("m2")), 2.0);
    key(&mut draft, &mask_key(x, Some("m3")), 3.0);
    // Keys on something else stay as they are throughout.
    key(&mut draft, "KFTypeVolume", 0.5);

    // m3 to the front: it takes the plain names, m1 its own.
    move_mask(&mut draft, &a(), 2, 0, false).unwrap();
    assert_eq!(value(&draft, x), Some(3.0));
    assert_eq!(value(&draft, &mask_key(x, Some("m1"))), Some(1.0));
    assert_eq!(value(&draft, &mask_key(x, Some("m2"))), Some(2.0));
    assert_eq!(value(&draft, &mask_key(x, Some("m3"))), None);
    let moved = cutlass_effects::masks_at(&draft, &draft.tracks[0].segments[0], MASK, Micros(0));
    assert_eq!(
        moved.iter().map(|mask| mask.x).collect::<Vec<_>>(),
        [3.0, 1.0, 2.0]
    );

    // The first goes: its keys with it, and m1 comes first.
    remove_mask(&mut draft, &a(), 0, false).unwrap();
    assert_eq!(value(&draft, x), Some(1.0));
    assert_eq!(value(&draft, &mask_key(x, Some("m1"))), None);
    // A later one goes with its own keys only.
    remove_mask(&mut draft, &a(), 1, false).unwrap();
    assert_eq!(key_names(&draft), [x, "KFTypeVolume"]);
}

#[test]
fn a_shared_mask_is_copied_for_the_clip_that_changes_it_and_its_keys_come_too() {
    let mut draft = one_clip();
    add_mask(
        &mut draft,
        &a(),
        &id("m1"),
        &Mask::new(MaskShape::Circle),
        None,
        false,
    )
    .unwrap();
    add_mask(
        &mut draft,
        &a(),
        &id("m2"),
        &Mask::new(MaskShape::Star),
        None,
        false,
    )
    .unwrap();
    let x = MASK_KEYS[0];
    key(&mut draft, &mask_key(x, Some("m2")), 0.5);
    // A second clip shares both records, as the halves of a split do.
    let mut b = draft.tracks[0].segments[0].clone();
    b.id = SegmentId::new("b");
    b.target = TimeRange::new(1_000_000, 1_000_000);
    draft.tracks[0].segments.push(b);

    let wide = Mask {
        width: 0.9,
        ..Mask::new(MaskShape::Star)
    };
    edit_mask(&mut draft, &a(), 1, &id("m2-a"), &wide, None, false).unwrap();
    assert_eq!(ids(&draft), ["m1", "m2-a"]);
    assert_eq!(value(&draft, &mask_key(x, Some("m2-a"))), Some(0.5));
    let other = &draft.tracks[0].segments[1];
    let theirs = segment_masks(&draft, other);
    assert_eq!(theirs[1].id, id("m2"));
    assert_eq!(theirs[1].mask, Mask::new(MaskShape::Star));
    // Taking a shared mask off one clip leaves the record to the other.
    remove_mask(&mut draft, &a(), 0, false).unwrap();
    assert!(draft.materials.get("m1").is_some());
}

#[test]
fn a_bad_place_mask_target_or_id_leaves_the_draft() {
    let mut draft = one_clip();
    add_mask(
        &mut draft,
        &a(),
        &id("m1"),
        &Mask::new(MaskShape::Circle),
        None,
        false,
    )
    .unwrap();
    let before = draft.clone();
    let circle = Mask::new(MaskShape::Circle);
    let bad = Mask {
        feather: 2.0,
        ..circle
    };
    let tries: Vec<(Result<(), EditError>, EditError)> = vec![
        (
            edit_mask(&mut draft, &a(), 1, &id("x"), &circle, None, false),
            EditError::MaskList,
        ),
        (
            edit_mask(&mut draft, &a(), 0, &id("x"), &bad, None, false),
            EditError::Mask,
        ),
        (
            add_mask(&mut draft, &a(), &id("y"), &bad, None, false),
            EditError::Mask,
        ),
        (
            add_mask(&mut draft, &a(), &id("m1"), &circle, None, false),
            EditError::DuplicateMaterial,
        ),
        (
            add_mask(&mut draft, &a(), &id(""), &circle, None, false),
            EditError::DuplicateMaterial,
        ),
        (remove_mask(&mut draft, &a(), 3, false), EditError::MaskList),
        (remove_mask(&mut draft, &a(), 0, true), EditError::MaskList),
        (
            move_mask(&mut draft, &a(), 0, 1, false),
            EditError::MaskList,
        ),
    ];
    for (result, wanted) in tries {
        assert_eq!(result, Err(wanted));
    }
    assert_eq!(draft, before);

    // An audio clip takes no mask, and a locked row takes no change.
    draft.tracks[0].kind = TrackType::Audio;
    assert_eq!(
        add_mask(&mut draft, &a(), &id("y"), &circle, None, false),
        Err(EditError::EffectTarget)
    );
    draft.tracks[0].kind = TrackType::Video;
    draft.tracks[0].attribute = TrackAttribute(TrackAttribute::LOCK);
    assert_eq!(
        remove_mask(&mut draft, &a(), 0, false),
        Err(EditError::LockedTrack)
    );
    draft.tracks[0].attribute = TrackAttribute::default();
    assert_eq!(draft, before);
}

#[test]
fn a_clip_takes_eight_masks_of_each_kind() {
    let mut draft = one_clip();
    let circle = Mask::new(MaskShape::Circle);
    for index in 0..MAX_MASKS {
        add_mask(
            &mut draft,
            &a(),
            &id(&format!("m{index}")),
            &circle,
            None,
            false,
        )
        .unwrap();
    }
    assert_eq!(
        add_mask(&mut draft, &a(), &id("one-more"), &circle, None, false),
        Err(EditError::MaskList)
    );
    // The colour masks count on their own.
    add_mask(&mut draft, &a(), &id("c"), &circle, None, true).unwrap();
    assert_eq!(
        draft
            .materials
            .common_mask
            .iter()
            .filter(|raw| raw.kind == COLOR_MASK)
            .count(),
        1
    );
}

#[test]
fn taking_off_the_first_mask_by_the_one_mask_request_hands_its_names_on() {
    let mut draft = one_clip();
    add_mask(
        &mut draft,
        &a(),
        &id("m1"),
        &Mask::new(MaskShape::Circle),
        None,
        false,
    )
    .unwrap();
    add_mask(
        &mut draft,
        &a(),
        &id("m2"),
        &Mask::new(MaskShape::Star),
        None,
        false,
    )
    .unwrap();
    let x = MASK_KEYS[0];
    key(&mut draft, x, 1.0);
    key(&mut draft, &mask_key(x, Some("m2")), 2.0);
    set_mask(&mut draft, &a(), &id("unused"), None).unwrap();
    assert_eq!(ids(&draft), ["m2"]);
    assert_eq!(value(&draft, x), Some(2.0));
    assert_eq!(key_names(&draft), [x]);
}

#[test]
fn the_requests_read_from_json() {
    let mut draft = one_clip();
    let circle = serde_json::json!({
        "shape": "circle", "x": 0.0, "y": 0.0, "width": 0.5, "height": 0.5,
        "rotation": 0.0, "feather": 0.0, "round": 0.0, "invert": false
    });
    let mut star = circle.clone();
    star["shape"] = "star".into();
    star["mode"] = "subtract".into();
    for value in [
        serde_json::json!({"op": "add_mask", "segment_id": "a", "material_id": "m1", "mask": circle}),
        serde_json::json!({"op": "add_mask", "segment_id": "a", "material_id": "m2", "mask": circle}),
        serde_json::json!({"op": "edit_mask", "segment_id": "a", "index": 1, "material_id": "m3", "mask": star}),
        serde_json::json!({"op": "move_mask", "segment_id": "a", "from": 1, "to": 0}),
        serde_json::json!({"op": "remove_mask", "segment_id": "a", "index": 1}),
        serde_json::json!({"op": "add_mask", "segment_id": "a", "material_id": "c", "mask": circle, "color": true}),
    ] {
        let request: Request = serde_json::from_value(value).unwrap();
        apply_request(&mut draft, &request).unwrap();
    }
    assert_eq!(shapes(&draft), [(MaskShape::Star, MaskMode::Subtract)]);
    assert_eq!(
        segment_color_masks(&draft, &draft.tracks[0].segments[0]).len(),
        1
    );
}

#[test]
fn a_drawn_mask_keeps_its_drawing_until_it_is_given_another() {
    use cutlass_effects::{BrushStroke, MaskArt};
    let mut draft = one_clip();
    let stroke = |x: f32| MaskArt::Brush {
        strokes: vec![BrushStroke {
            points: vec![[x, 0.0], [x + 0.25, 0.5]],
            size: 0.1,
            erase: false,
        }],
    };
    let brush = Mask::new(MaskShape::Brush);
    add_mask(&mut draft, &a(), &id("b"), &brush, Some(&stroke(0.0)), false).unwrap();
    let art = |draft: &Draft| {
        segment_masks(draft, &draft.tracks[0].segments[0])
            .first()
            .and_then(|listed| listed.art.clone())
    };
    assert_eq!(art(&draft), Some(stroke(0.0)));
    // Moving the box keeps the strokes.
    let moved = Mask { x: 0.3, ..brush };
    edit_mask(&mut draft, &a(), 0, &id("b2"), &moved, None, false).unwrap();
    assert_eq!(art(&draft), Some(stroke(0.0)));
    edit_mask(&mut draft, &a(), 0, &id("b2"), &moved, Some(&stroke(0.5)), false).unwrap();
    assert_eq!(art(&draft), Some(stroke(0.5)));
    // Another shape's drawing, or a drawing on an analytic shape, is
    // refused and leaves the draft alone.
    let before = draft.clone();
    let pen = Mask::new(MaskShape::Pen);
    assert_eq!(
        edit_mask(&mut draft, &a(), 0, &id("x"), &pen, Some(&stroke(0.0)), false),
        Err(EditError::Mask)
    );
    let circle = Mask::new(MaskShape::Circle);
    assert_eq!(
        add_mask(&mut draft, &a(), &id("y"), &circle, Some(&stroke(0.0)), false),
        Err(EditError::Mask)
    );
    assert_eq!(draft, before);
    // Turned into a circle, the drawing goes.
    edit_mask(&mut draft, &a(), 0, &id("b3"), &circle, None, false).unwrap();
    assert_eq!(art(&draft), None);
    let raw = &draft.materials.common_mask[0];
    assert!(!raw.fields.contains_key("art"));
}

use cutlass_effects::{Mask, MaskMode, MaskShape};
use cutlass_model::{MASK_KEYS, Micros, SegmentId, mask_key};

use super::super::inspector::tests::every_kind;
use super::super::tests::Temp;
use super::*;

fn picked(temp: &Temp) -> Shell {
    let mut desktop = every_kind(temp);
    desktop.select(Some(SegmentId::new("main")));
    desktop
}

fn key_names(desktop: &Shell) -> Vec<String> {
    let mut names: Vec<String> = desktop
        .selected_segment()
        .unwrap()
        .common_keyframes
        .iter()
        .map(|group| group.property_type.clone())
        .collect();
    names.sort();
    names
}

#[test]
fn masks_are_added_picked_moved_and_taken_off_in_hand() {
    let temp = Temp::new();
    let mut desktop = picked(&temp);
    let circle = Mask::new(MaskShape::Circle);
    desktop.set_mask(Some(circle)).unwrap();
    desktop
        .add_mask(false, Mask::new(MaskShape::Star), None)
        .unwrap();
    // The new mask is in hand.
    assert_eq!(
        desktop.mask_focus(),
        MaskFocus {
            color: false,
            index: 1
        }
    );
    desktop.set_mask_mode(MaskMode::Subtract).unwrap();
    let shapes = |desktop: &Shell| -> Vec<(MaskShape, MaskMode)> {
        desktop
            .mask_rows(false)
            .iter()
            .map(|row| (row.mask.shape, row.mask.mode))
            .collect()
    };
    assert_eq!(
        shapes(&desktop),
        vec![
            (MaskShape::Circle, MaskMode::Add),
            (MaskShape::Star, MaskMode::Subtract)
        ]
    );
    desktop.move_focused_mask(true).unwrap();
    assert_eq!(desktop.mask_focus().index, 0);
    assert_eq!(shapes(&desktop)[0].0, MaskShape::Star);
    // Moving past the top does nothing.
    desktop.move_focused_mask(true).unwrap();
    assert_eq!(shapes(&desktop)[0].0, MaskShape::Star);
    desktop.set_mask(None).unwrap();
    assert_eq!(shapes(&desktop), vec![(MaskShape::Circle, MaskMode::Add)]);
    // Undo brings it back as one step.
    desktop.undo().unwrap();
    assert_eq!(shapes(&desktop).len(), 2);
}

#[test]
fn a_key_on_the_mask_in_hand_goes_under_its_own_names() {
    let temp = Temp::new();
    let mut desktop = picked(&temp);
    desktop
        .set_mask(Some(Mask::new(MaskShape::Circle)))
        .unwrap();
    desktop
        .add_mask(false, Mask::new(MaskShape::Star), None)
        .unwrap();
    let second = desktop.mask_rows(false).len() - 1;
    desktop.focus_mask(false, second);
    desktop.seek(Micros(1_000_000));
    desktop
        .toggle_setting_key(KeyedSetting::MaskPosition)
        .unwrap();
    desktop.seek(Micros(2_000_000));
    let star = desktop.focused_mask().unwrap();
    desktop
        .set_focused_mask(Some(Mask { x: 0.5, ..star }))
        .unwrap();
    let names = key_names(&desktop);
    assert!(names.iter().all(|name| name.contains('@')), "{names:?}");
    assert!(desktop.setting_keyed(KeyedSetting::MaskPosition));
    // The first mask has no keys of its own and stays put.
    desktop.focus_mask(false, 0);
    assert!(!desktop.setting_keyed(KeyedSetting::MaskPosition));
    desktop.seek(Micros(1_500_000));
    assert_eq!(desktop.focused_mask().unwrap().x, 0.0);
    desktop.focus_mask(false, second);
    assert!((desktop.focused_mask().unwrap().x - 0.25).abs() < 1e-6);
}

#[test]
fn a_colour_mask_keys_apart_from_the_clip_mask() {
    let temp = Temp::new();
    let mut desktop = picked(&temp);
    desktop
        .set_mask(Some(Mask::new(MaskShape::Circle)))
        .unwrap();
    desktop
        .set_color_mask(Some(Mask::new(MaskShape::Rectangle)))
        .unwrap();
    desktop.focus_mask(true, 0);
    desktop.seek(Micros(1_000_000));
    desktop.toggle_setting_key(KeyedSetting::MaskSize).unwrap();
    let names = key_names(&desktop);
    let record = desktop
        .selected_segment()
        .unwrap()
        .extra_material_refs
        .iter()
        .find(|id| id.as_str().starts_with("color-mask"))
        .unwrap()
        .as_str()
        .to_owned();
    assert_eq!(
        names,
        vec![
            mask_key(MASK_KEYS[3], Some(&record)),
            mask_key(MASK_KEYS[2], Some(&record))
        ]
    );
    desktop.focus_mask(false, 0);
    assert!(!desktop.setting_keyed(KeyedSetting::MaskSize));
}

#[test]
fn a_brush_mask_takes_its_strokes() {
    let temp = Temp::new();
    let mut desktop = picked(&temp);
    desktop
        .add_mask(false, Mask::new(MaskShape::Brush), None)
        .unwrap();
    let stroke = cutlass_effects::BrushStroke {
        points: vec![[0.0, 0.0], [0.5, 0.5]],
        size: 0.2,
        erase: false,
    };
    let art = cutlass_editor::brush_art(None, stroke);
    desktop.set_mask_art(art.clone()).unwrap();
    assert_eq!(desktop.mask_rows(false)[0].art, Some(art));
    // Moving it keeps the strokes.
    let brush = desktop.focused_mask().unwrap();
    desktop
        .set_focused_mask(Some(Mask { x: 0.2, ..brush }))
        .unwrap();
    assert!(desktop.mask_rows(false)[0].art.is_some());
}

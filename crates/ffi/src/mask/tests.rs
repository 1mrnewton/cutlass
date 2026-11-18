use cutlass_session::Micros;

use crate::tests::{Roots, Temp, fixture, still};

#[test]
fn a_phone_adds_edits_moves_and_takes_off_masks() {
    let temp = Temp::new();
    let roots = Roots::new(&temp);
    let mut host = roots.create();
    let id = host
        .import_probed(&fixture("pattern.png"), still(1_000_000))
        .unwrap();
    let clip = host.append(&id).unwrap();
    let at = Micros(0);
    assert!(host.masks(&clip, false, at).is_empty());
    assert!(host.add_mask(&clip, false, "blob", at).is_err());
    assert_eq!(host.add_mask(&clip, false, "circle", at).unwrap(), 0);
    assert_eq!(host.add_mask(&clip, false, "star", at).unwrap(), 1);
    assert_eq!(host.add_mask(&clip, true, "rectangle", at).unwrap(), 0);
    let masks = host.masks(&clip, false, at);
    assert_eq!(
        masks
            .iter()
            .map(|mask| mask.shape.as_str())
            .collect::<Vec<_>>(),
        ["circle", "star"]
    );
    assert_eq!(host.masks(&clip, true, at).len(), 1);

    // A slider drag is one undo step.
    let mut star = masks[1].clone();
    star.mode = "subtract".to_owned();
    host.edit_mask(&clip, false, 1, at, &star, true).unwrap();
    star.feather = 0.5;
    host.edit_mask(&clip, false, 1, at, &star, false).unwrap();
    let star = host.masks(&clip, false, at)[1].clone();
    assert_eq!((star.mode.as_str(), star.feather), ("subtract", 0.5));
    host.undo().unwrap();
    assert_eq!(host.masks(&clip, false, at)[1].mode, "add");
    host.redo().unwrap();
    let mut bad = star.clone();
    bad.mode = "both".to_owned();
    assert!(host.edit_mask(&clip, false, 1, at, &bad, false).is_err());

    host.move_mask(&clip, false, 1, 0).unwrap();
    assert_eq!(host.masks(&clip, false, at)[0].shape, "star");
    host.remove_mask(&clip, false, 0).unwrap();
    assert_eq!(host.masks(&clip, false, at)[0].shape, "circle");
    host.undo().unwrap();
    assert_eq!(host.masks(&clip, false, at).len(), 2);
    assert_eq!(crate::mask_shapes().len(), 9);
}

#[test]
fn a_drag_on_a_handle_moves_a_mask_as_one_undo_step() {
    let temp = Temp::new();
    let roots = Roots::new(&temp);
    let mut host = roots.create();
    let id = host
        .import_probed(&fixture("pattern.png"), still(1_000_000))
        .unwrap();
    let clip = host.append(&id).unwrap();
    let at = Micros(0);
    host.add_mask(&clip, false, "rectangle", at).unwrap();
    let handles = host.mask_handles(&clip, false, 0, at).unwrap();
    assert_eq!(handles.corners.len(), 8);
    assert!(handles.line.is_empty());
    let (x, y) = (handles.middle_x, handles.middle_y);
    // Far outside the box nothing is taken hold of.
    assert!(!host.mask_begin(&clip, false, 0, at, -1e4, -1e4).unwrap());
    assert!(host.mask_drag(x, y).is_err());

    let steps = host.revision();
    assert!(host.mask_begin(&clip, false, 0, at, x, y).unwrap());
    host.mask_drag(x, y).unwrap();
    assert_eq!(host.revision(), steps);
    host.mask_drag(x + 10.0, y).unwrap();
    let moved = host.mask_drag(x + handles.canvas_width / 4.0, y).unwrap();
    host.mask_end();
    assert!(host.revision() > steps);
    assert!(moved.x > 0.4, "{}", moved.x);
    assert!((host.masks(&clip, false, at)[0].x - moved.x).abs() < 1e-6);
    host.undo().unwrap();
    assert_eq!(host.masks(&clip, false, at)[0].x, 0.0);
}

#[test]
fn a_phone_draws_text_brush_and_pen_masks() {
    let temp = Temp::new();
    let roots = Roots::new(&temp);
    let mut host = roots.create();
    let id = host
        .import_probed(&fixture("pattern.png"), still(1_000_000))
        .unwrap();
    let clip = host.append(&id).unwrap();
    let at = Micros(0);
    host.add_mask(&clip, false, "text", at).unwrap();
    assert_eq!(
        host.masks(&clip, false, at)[0].text.as_deref(),
        Some("Text")
    );
    host.set_mask_text(&clip, false, 0, "Hi").unwrap();
    assert_eq!(host.masks(&clip, false, at)[0].text.as_deref(), Some("Hi"));

    host.add_mask(&clip, false, "brush", at).unwrap();
    let handles = host.mask_handles(&clip, false, 1, at).unwrap();
    let (x, y) = (handles.middle_x, handles.middle_y);
    let step = handles.canvas_width / 10.0;
    host.paint_mask(
        &clip,
        false,
        1,
        at,
        &[x, y, x + step, y, x + 2.0 * step, y],
        crate::PhoneBrush {
            size: step / 4.0,
            erase: false,
        },
    )
    .unwrap();
    assert_eq!(host.masks(&clip, false, at)[1].shape, "brush");

    host.add_mask(&clip, false, "pen", at).unwrap();
    assert!(
        host.draw_mask_outline(&clip, false, 2, at, &[x, y, x + step, y])
            .is_err()
    );
    host.draw_mask_outline(&clip, false, 2, at, &[x, y, x + step, y, x, y + step])
        .unwrap();
    assert_eq!(host.masks(&clip, false, at)[2].shape, "pen");
}

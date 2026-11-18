use cutlass_effects::{MaskArt, MaskShape, segment_masks};
use cutlass_model::{Draft, Segment, ShapePoint, TimeRange};
use cutlass_session::Session;

use super::*;

/// A 200 by 100 clip filling a 200 by 100 canvas: its middle at (100, 50),
/// one clip part 100 pixels across and 50 up.
fn frame() -> ClipFrame {
    ClipFrame::from_canvas((100.0, 50.0), (200.0, 50.0), (100.0, 0.0)).unwrap()
}

fn draft_with(mask: Mask) -> Draft {
    let mut draft = Draft::empty("D");
    draft.canvas_config.width = 200;
    draft.canvas_config.height = 100;
    draft.tracks[0]
        .segments
        .push(Segment::new("a", "m", TimeRange::new(0, 1_000_000)));
    let mut session = Session::open_project(draft);
    apply(
        &mut session,
        vec![Request::AddMask {
            segment_id: "a".into(),
            material_id: "m1".into(),
            mask,
            art: None,
            color: false,
        }],
    );
    session.draft().clone()
}

fn apply(session: &mut Session, requests: Vec<Request>) {
    let revision = session.revision();
    session.apply(revision, &requests).unwrap();
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-3
}

#[test]
fn a_clip_frame_goes_both_ways() {
    let frame = frame();
    assert_eq!(frame.clip(100.0, 50.0), (0.0, 0.0));
    let (u, v) = frame.clip(150.0, 25.0);
    assert!(close(u, 0.5) && close(v, 0.5));
    let (x, y) = frame.canvas(0.5, 0.5).unwrap();
    assert!(close(x, 150.0) && close(y, 25.0));
    assert_eq!(
        ClipFrame::from_canvas((0.0, 0.0), (1.0, 1.0), (2.0, 2.0)),
        None
    );
}

#[test]
fn handles_sit_on_the_box_and_a_press_finds_them() {
    let mask = Mask {
        x: 0.2,
        width: 0.3,
        height: 0.4,
        ..Mask::new(MaskShape::Rectangle)
    };
    let frame = frame();
    let handles = mask_handles(&mask, &frame).unwrap();
    assert!(close(handles.middle.0, 120.0) && close(handles.middle.1, 50.0));
    // Top left: x 0.2 - 0.3, y 0.4 up.
    assert!(close(handles.corners[0].0, 90.0) && close(handles.corners[0].1, 30.0));
    assert!(handles.turn.1 < handles.corners[0].1);
    assert!(handles.feather.1 > handles.corners[3].1);
    let pick = |x, y| mask_pick(&mask, &frame, x, y, 6.0);
    assert_eq!(pick(handles.turn.0, handles.turn.1), Some(MaskHandle::Turn));
    assert_eq!(
        pick(handles.feather.0, handles.feather.1),
        Some(MaskHandle::Feather)
    );
    assert_eq!(
        pick(152.0, 71.0),
        Some(MaskHandle::Corner {
            right: true,
            up: false
        })
    );
    assert_eq!(pick(125.0, 45.0), Some(MaskHandle::Move));
    assert_eq!(pick(10.0, 10.0), None);
}

#[test]
fn a_drag_moves_sizes_turns_and_softens_the_mask() {
    let mask = Mask {
        width: 0.2,
        height: 0.2,
        ..Mask::new(MaskShape::Circle)
    };
    let draft = draft_with(mask);
    let frame = frame();
    let begin = |handle, at| {
        MaskGesture::begin(
            &draft,
            &"a".into(),
            (0, false),
            frame,
            Micros(0),
            handle,
            at,
        )
        .unwrap()
    };
    let moved = begin(MaskHandle::Move, (100.0, 50.0)).step(130.0, 40.0);
    assert!(close(f64::from(moved.x), 0.3) && close(f64::from(moved.y), 0.2));
    let sized = begin(
        MaskHandle::Corner {
            right: true,
            up: true,
        },
        (120.0, 40.0),
    )
    .step(150.0, 25.0);
    assert!(close(f64::from(sized.width), 0.5) && close(f64::from(sized.height), 0.5));
    // From straight up to straight left is a quarter turn the other way.
    let turned = begin(MaskHandle::Turn, (100.0, 30.0)).step(80.0, 50.0);
    assert!(
        close(f64::from(turned.rotation), 90.0),
        "{}",
        turned.rotation
    );
    let soft = begin(MaskHandle::Feather, (100.0, 70.0)).step(100.0, 80.0);
    assert!(close(f64::from(soft.feather), 0.4), "{}", soft.feather);
    let hard = begin(MaskHandle::Feather, (100.0, 70.0)).step(100.0, 0.0);
    assert_eq!(hard.feather, 0.0);
}

#[test]
fn a_drag_on_an_unkeyed_mask_changes_it_and_on_a_keyed_one_keys_it() {
    let draft = draft_with(Mask::new(MaskShape::Circle));
    let frame = frame();
    let gesture = MaskGesture::begin(
        &draft,
        &"a".into(),
        (0, false),
        frame,
        Micros(500_000),
        MaskHandle::Move,
        (100.0, 50.0),
    )
    .unwrap();
    let moved = gesture.step(110.0, 50.0);
    let requests = gesture.requests(&moved, &"copy".into());
    assert!(matches!(
        &requests[..],
        [Request::EditMask { index: 0, .. }]
    ));
    let mut session = Session::open_project(draft);
    apply(&mut session, requests);
    apply(
        &mut session,
        vec![Request::KeyValues {
            segment_id: "a".into(),
            offset: Micros(0),
            values: vec![(MASK_KEYS[0].to_owned(), 0.0)],
        }],
    );
    let draft = session.draft().clone();
    let gesture = MaskGesture::begin(
        &draft,
        &"a".into(),
        (0, false),
        frame,
        Micros(500_000),
        MaskHandle::Move,
        (100.0, 50.0),
    )
    .unwrap();
    let requests = gesture.requests(&gesture.step(150.0, 50.0), &"copy".into());
    let [Request::KeyValues { offset, values, .. }] = &requests[..] else {
        panic!("{requests:?}");
    };
    assert_eq!(*offset, Micros(500_000));
    assert_eq!(values[0], (MASK_KEYS[0].to_owned(), 0.5));
    assert_eq!(values.len(), MASK_KEYS.len());
    // No mask at that place: no gesture.
    assert!(
        MaskGesture::begin(
            &draft,
            &"a".into(),
            (3, false),
            frame,
            Micros(0),
            MaskHandle::Move,
            (0.0, 0.0)
        )
        .is_none()
    );
}

#[test]
fn a_finger_draws_brush_strokes_and_a_pen_outline_in_the_box() {
    let mask = Mask::new(MaskShape::Brush);
    let frame = frame();
    let trace = [
        (100.0, 50.0),
        (100.5, 50.0),
        (150.0, 50.0),
        (150.0, f64::NAN),
    ];
    let stroke = brush_stroke(&mask, &frame, &trace, 10.0, false).unwrap();
    assert_eq!(stroke.points, vec![[0.0, 0.0], [0.5, 0.0]]);
    assert!((stroke.size - 0.1).abs() < 1e-4, "{}", stroke.size);
    let art = brush_art(None, stroke.clone());
    let art = brush_art(
        Some(&art),
        BrushStroke {
            erase: true,
            ..stroke
        },
    );
    let MaskArt::Brush { strokes } = &art else {
        panic!()
    };
    assert_eq!(strokes.len(), 2);
    assert!(art.is_valid());
    assert_eq!(brush_stroke(&mask, &frame, &[], 10.0, false), None);
    let corner = |x: f64, y: f64| ShapePoint {
        at: [x, y],
        ..ShapePoint::default()
    };
    let pen = Mask::new(MaskShape::Pen);
    let outline = [
        corner(100.0, 0.0),
        corner(200.0, 50.0),
        corner(100.0, 100.0),
    ];
    let MaskArt::Pen { points } = pen_art(&pen, &frame, &outline).unwrap() else {
        panic!()
    };
    assert_eq!(points[0].at, [0.0, 1.0]);
    assert_eq!(points[1].at, [1.0, 0.0]);
    assert_eq!(pen_art(&pen, &frame, &outline[..2]), None);
}

#[test]
fn a_text_box_keeps_the_letters_shape() {
    // Words four times as wide as tall, half a 1920 by 1080 clip across.
    let (width, height) = text_box(4.0, (1920.0, 1080.0), 0.5);
    assert_eq!(width, 0.5);
    // 960 pixels across, 240 tall: 240 of 1080.
    assert!((height - 240.0 / 1080.0).abs() < 1e-5, "{height}");
}

#[test]
fn a_gesture_reads_the_mask_at_its_place() {
    let mut draft = draft_with(Mask::new(MaskShape::Circle));
    let mut session = Session::open_project(draft.clone());
    apply(
        &mut session,
        vec![Request::AddMask {
            segment_id: "a".into(),
            material_id: "m2".into(),
            mask: Mask {
                x: -0.5,
                ..Mask::new(MaskShape::Star)
            },
            art: None,
            color: false,
        }],
    );
    draft = session.draft().clone();
    let gesture = MaskGesture::begin(
        &draft,
        &"a".into(),
        (1, false),
        frame(),
        Micros(0),
        MaskHandle::Move,
        (50.0, 50.0),
    )
    .unwrap();
    assert_eq!(gesture.start().shape, MaskShape::Star);
    let requests = gesture.requests(&gesture.step(60.0, 50.0), &"copy".into());
    apply(&mut session, requests);
    let masks = segment_masks(session.draft(), &session.draft().tracks[0].segments[0]);
    assert!(close(f64::from(masks[1].mask.x), -0.4));
}

#[test]
fn a_keyed_mask_keys_its_numbers_and_stores_what_keys_leave_out() {
    let draft = draft_with(Mask::new(MaskShape::Circle));
    let mut session = Session::open_project(draft);
    apply(
        &mut session,
        vec![Request::KeyValues {
            segment_id: "a".into(),
            offset: Micros(0),
            values: vec![(MASK_KEYS[0].to_owned(), 0.0)],
        }],
    );
    let draft = session.draft().clone();
    let edit = |mask: &Mask| {
        mask_edit_requests(
            &draft,
            &"a".into(),
            (0, false),
            Micros(0),
            mask,
            &"copy".into(),
        )
        .unwrap()
    };
    let moved = Mask {
        x: 0.25,
        ..Mask::new(MaskShape::Circle)
    };
    assert!(matches!(&edit(&moved)[..], [Request::KeyValues { .. }]));
    let flipped = Mask {
        invert: true,
        mode: cutlass_effects::MaskMode::Subtract,
        ..moved
    };
    let requests = edit(&flipped);
    assert!(matches!(
        &requests[..],
        [Request::EditMask { .. }, Request::KeyValues { .. }]
    ));
    apply(&mut session, requests);
    let masks = segment_masks(session.draft(), &session.draft().tracks[0].segments[0]);
    assert!(masks[0].mask.invert);
    assert_eq!(masks[0].mask.mode, cutlass_effects::MaskMode::Subtract);
    assert_eq!(
        mask_edit_requests(
            &draft,
            &"a".into(),
            (4, false),
            Micros(0),
            &moved,
            &"copy".into()
        ),
        None
    );
}

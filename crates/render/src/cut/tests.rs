use cutlass_effects::{CanvasMask, Mask, MaskShape, Selection};
use cutlass_gpu::Placement;
use cutlass_model::Micros;

use crate::{Layer, RenderError, Renderer, Rgba, Selected};

fn renderer() -> Option<Renderer> {
    let renderer = Renderer::for_test().expect("a GPU is required");
    if renderer.is_none() {
        eprintln!("skipped: no GPU");
    }
    renderer
}

const SIZE: (u32, u32) = (40, 28);

/// Canvas pixels to the frame of a clip that fills the canvas: -1 to 1
/// across, 1 at the top to -1 at the bottom.
fn full_frame(size: (u32, u32)) -> [f32; 6] {
    [
        2.0 / size.0 as f32,
        0.0,
        -1.0,
        0.0,
        -2.0 / size.1 as f32,
        1.0,
    ]
}

/// A clip that fills the canvas, half of it half clear.
fn scene() -> Vec<u8> {
    (0..SIZE.0 * SIZE.1)
        .flat_map(|index| {
            let x = index % SIZE.0;
            if x.is_multiple_of(2) {
                [200, 120, 40, 255]
            } else {
                [60, 50, 100, 128]
            }
        })
        .collect()
}

fn draw(renderer: &Renderer, layer: Layer<'_>) -> Result<Vec<u8>, RenderError> {
    let canvas = renderer.canvas(SIZE.0, SIZE.1)?;
    let bytes = scene();
    let picture = Rgba {
        width: SIZE.0,
        height: SIZE.1,
        bytes: &bytes,
    };
    renderer.draw(
        &canvas,
        &[
            Layer::Rgba(picture, Placement::canvas(SIZE.0, SIZE.1)),
            layer,
        ],
    )?;
    renderer.read(&canvas)
}

fn within(label: &str, got: &[u8], expected: &[u8]) {
    assert_eq!(got.len(), expected.len());
    for (index, (a, b)) in got.iter().zip(expected).enumerate() {
        assert!(
            a.abs_diff(*b) <= 1,
            "{label}: byte {index} is {a}, the CPU has {b}"
        );
    }
}

fn to_clip(frame: [f32; 6]) -> impl Fn(f32, f32) -> Option<(f32, f32)> {
    move |x, y| {
        let [a, b, c, d, e, f] = frame;
        Some((a * x + b * y + c, d * x + e * y + f))
    }
}

#[test]
fn a_clip_mask_cuts_as_the_cpu_does() {
    let Some(renderer) = renderer() else { return };
    let frame = full_frame(SIZE);
    for (index, shape) in MaskShape::ALL.into_iter().enumerate() {
        let mask = Mask {
            rotation: 15.0,
            feather: 0.3,
            round: 0.5,
            invert: index % 2 == 0,
            ..Mask::new(shape)
        };
        let mut expected = scene();
        mask.apply(&mut expected, SIZE.0, to_clip(frame));
        assert_ne!(expected, scene(), "{shape:?} cuts something");
        let got = draw(&renderer, Layer::Cut(&CanvasMask { mask, frame })).expect("draw");
        within(&format!("{shape:?}"), &got, &expected);
    }
}

#[test]
fn a_selection_cuts_as_the_cpu_does_at_any_size() {
    let Some(renderer) = renderer() else { return };
    // A diagonal object on a frame of another size than the canvas, so
    // its edge is read between pixels.
    let (w, h) = (17u32, 9u32);
    let alpha: Vec<u8> = (0..w * h)
        .map(|index| {
            let (x, y) = (index % w, index / w);
            if x + y > 8 && x < 14 { 255 } else { 0 }
        })
        .collect();
    let frame = full_frame(SIZE);
    for invert in [false, true] {
        let selection = Selection {
            invert,
            ..Selection::from_alpha(w, h, &alpha, Micros(0)).expect("selection")
        };
        let mut expected = scene();
        selection.apply(&mut expected, SIZE.0, to_clip(frame));
        let decoded = selection.alpha().expect("alpha");
        let got = draw(
            &renderer,
            Layer::Select(Selected {
                width: w,
                height: h,
                alpha: &decoded,
                invert,
                frame,
            }),
        )
        .expect("draw");
        within(&format!("invert {invert}"), &got, &expected);
    }
}

#[test]
fn a_selection_off_the_clip_clears_it() {
    let Some(renderer) = renderer() else { return };
    // The clip sits far to the right of the canvas.
    let mut frame = full_frame(SIZE);
    frame[2] -= 5.0;
    let alpha = [255u8; 4];
    let got = draw(
        &renderer,
        Layer::Select(Selected {
            width: 2,
            height: 2,
            alpha: &alpha,
            invert: false,
            frame,
        }),
    )
    .expect("draw");
    assert!(got.iter().all(|byte| *byte == 0));
}

#[test]
fn a_selection_whose_bytes_do_not_match_its_size_is_refused() {
    let Some(renderer) = renderer() else { return };
    let alpha = [255u8; 3];
    let result = draw(
        &renderer,
        Layer::Select(Selected {
            width: 2,
            height: 2,
            alpha: &alpha,
            invert: false,
            frame: full_frame(SIZE),
        }),
    );
    assert!(matches!(
        result,
        Err(RenderError::Bytes {
            expected: 4,
            actual: 3
        })
    ));
    let empty = draw(
        &renderer,
        Layer::Select(Selected {
            width: 0,
            height: 2,
            alpha: &[],
            invert: false,
            frame: full_frame(SIZE),
        }),
    );
    assert!(matches!(empty, Err(RenderError::Size { .. })));
}

#[test]
fn a_cut_on_a_shrunk_canvas_lands_where_it_does_on_the_full_one() {
    let Some(renderer) = renderer() else { return };
    // Keep the left half of the clip only.
    let mask = CanvasMask {
        mask: Mask {
            rotation: 90.0,
            ..Mask::new(MaskShape::Linear)
        },
        frame: full_frame(SIZE),
    };
    let half = crate::preview_size(SIZE, 0.5);
    let canvas = renderer.canvas(half.0, half.1).expect("canvas");
    let bytes = scene();
    let picture = Rgba {
        width: SIZE.0,
        height: SIZE.1,
        bytes: &bytes,
    };
    renderer
        .draw_scaled(
            &canvas,
            &[
                Layer::Rgba(picture, Placement::canvas(SIZE.0, SIZE.1)),
                Layer::Cut(&mask),
            ],
            SIZE,
        )
        .expect("draw");
    let got = renderer.read(&canvas).expect("read");
    let full = draw(&renderer, Layer::Cut(&mask)).expect("draw");
    for row in 0..half.1 {
        for column in [1, half.0 - 2] {
            let small = got[((row * half.0 + column) * 4 + 3) as usize];
            let big = full[(((row * 2) * SIZE.0 + column * 2) * 4 + 3) as usize];
            assert_eq!(small == 0, big == 0, "row {row} column {column}");
        }
    }
    // One side is kept and the other cleared.
    assert_ne!(got[3] > 0, got[((half.0 - 1) * 4 + 3) as usize] > 0);
}

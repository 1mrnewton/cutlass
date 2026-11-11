use cutlass_effects::{Lighting, RELIGHT_PRESETS, lighting, relight_rgba, stretch_axis};
use cutlass_gpu::Placement;
use cutlass_model::Relight;

use crate::{Layer, Relit, RenderError, Renderer, Rgba};

fn renderer() -> Option<Renderer> {
    let renderer = Renderer::for_test().expect("a GPU is required");
    if renderer.is_none() {
        eprintln!("skipped: no GPU");
    }
    renderer
}

/// A picture with colour in every channel and some half clear.
fn picture(width: u32, height: u32) -> Vec<u8> {
    (0..width * height)
        .flat_map(|index| {
            let (x, y) = (index % width, index / width);
            let alpha = if (x + y) % 7 == 0 { 128 } else { 255 };
            let scale = |c: u32| (c.min(255) * alpha / 255) as u8;
            [
                scale(30 + x * 200 / width),
                scale(40 + y * 180 / height),
                scale(200 - x * 90 / width),
                alpha as u8,
            ]
        })
        .collect()
}

/// A ball of near depth on a far floor, as RGBA grey.
fn depth(width: u32, height: u32) -> Vec<u8> {
    let (cx, cy) = (width as f32 * 0.45, height as f32 * 0.55);
    let radius = width.min(height) as f32 * 0.4;
    (0..width * height)
        .flat_map(|index| {
            let (x, y) = ((index % width) as f32, (index / width) as f32);
            let d = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() / radius;
            let near = if d < 1.0 {
                (60.0 + 190.0 * (1.0 - d * d).sqrt()) as u8
            } else {
                (20.0 + y * 30.0 / height as f32) as u8
            };
            [near, near, near, 255]
        })
        .collect()
}

/// The red of `depth`, `from` in size, stretched to `to` as eval
/// stretches a depth picture: bilinear, centres matched, rounded.
fn stretched(depth: &[u8], from: (u32, u32), to: (u32, u32)) -> Vec<u8> {
    let xs = stretch_axis(to.0 as usize, from.0 as usize);
    let ys = stretch_axis(to.1 as usize, from.1 as usize);
    let red = |x: usize, y: usize| f32::from(depth[(y * from.0 as usize + x) * 4]);
    let mut out = Vec::new();
    for &(y0, y1, fy) in &ys {
        for &(x0, x1, fx) in &xs {
            let top = red(x0, y0) + (red(x1, y0) - red(x0, y0)) * fx;
            let bottom = red(x0, y1) + (red(x1, y1) - red(x0, y1)) * fx;
            out.push((top + (bottom - top) * fy).round().clamp(0.0, 255.0) as u8);
        }
    }
    out
}

fn draw(
    renderer: &Renderer,
    size: (u32, u32),
    depth: Rgba<'_>,
    lighting: &Lighting,
) -> Result<Vec<u8>, RenderError> {
    let canvas = renderer.canvas(size.0, size.1)?;
    let bytes = picture(size.0, size.1);
    let rgba = Rgba {
        width: size.0,
        height: size.1,
        bytes: &bytes,
    };
    renderer.draw(
        &canvas,
        &[
            Layer::Rgba(rgba, Placement::canvas(size.0, size.1)),
            Layer::Relight(Relit { depth, lighting }),
        ],
    )?;
    renderer.read(&canvas)
}

fn preset(id: &str, strength: f64, seconds: f64) -> Lighting {
    lighting(
        &Relight {
            preset: id.to_owned(),
            strength,
            color: String::new(),
            angle: Relight::DEFAULT_ANGLE,
        },
        seconds,
    )
    .expect("lighting")
}

/// Every byte within two of the CPU's and nearly all within one.
fn agree(label: &str, got: &[u8], expected: &[u8]) {
    assert_eq!(got.len(), expected.len(), "{label}");
    let mut near = 0;
    for (index, (a, b)) in got.iter().zip(expected).enumerate() {
        let off = a.abs_diff(*b);
        assert!(off <= 2, "{label}: byte {index} is {a}, the CPU has {b}");
        near += usize::from(off <= 1);
    }
    assert!(near * 100 >= got.len() * 98, "{label}: {near} within one");
}

fn check(renderer: &Renderer, label: &str, size: (u32, u32), from: (u32, u32), lit: &Lighting) {
    let grey = depth(from.0, from.1);
    let got = draw(
        renderer,
        size,
        Rgba {
            width: from.0,
            height: from.1,
            bytes: &grey,
        },
        lit,
    )
    .expect("draw");
    let mut expected = picture(size.0, size.1);
    relight_rgba(
        &mut expected,
        size.0,
        size.1,
        &stretched(&grey, from, size),
        lit,
    );
    assert_ne!(expected, picture(size.0, size.1), "{label} lights it");
    agree(label, &got, &expected);
}

#[test]
fn every_preset_lights_as_the_cpu_does() {
    let Some(renderer) = renderer() else { return };
    for preset_of in &RELIGHT_PRESETS {
        let lit = preset(preset_of.id, 0.8, 1.3);
        check(&renderer, preset_of.id, (48, 36), (48, 36), &lit);
    }
    let custom = lighting(
        &Relight {
            preset: Relight::CUSTOM.to_owned(),
            strength: 1.0,
            color: "#ff8040".to_owned(),
            angle: 40.0,
        },
        0.0,
    )
    .expect("custom");
    check(&renderer, "custom", (48, 36), (48, 36), &custom);
}

#[test]
fn a_depth_of_another_size_is_stretched_as_eval_stretches_it() {
    let Some(renderer) = renderer() else { return };
    let lit = preset(RELIGHT_PRESETS[0].id, 1.0, 0.0);
    check(&renderer, "smaller depth", (50, 34), (19, 13), &lit);
    check(&renderer, "larger depth", (30, 20), (64, 47), &lit);
    // A picture large enough that the normal steps over more than a pixel.
    check(&renderer, "wide step", (340, 330), (90, 88), &lit);
}

#[test]
fn a_bad_depth_or_too_many_lights_is_refused() {
    let Some(renderer) = renderer() else { return };
    let lit = preset(RELIGHT_PRESETS[0].id, 1.0, 0.0);
    let short = [0u8; 7];
    let bad = Rgba {
        width: 2,
        height: 2,
        bytes: &short,
    };
    assert!(matches!(
        draw(&renderer, (8, 8), bad, &lit),
        Err(RenderError::Bytes { .. })
    ));
    let grey = depth(8, 8);
    let good = Rgba {
        width: 8,
        height: 8,
        bytes: &grey,
    };
    let mut crowded = lit.clone();
    crowded.lights = vec![lit.lights[0]; cutlass_effects::MAX_LIGHTS + 1];
    assert!(matches!(
        draw(&renderer, (8, 8), good, &crowded),
        Err(RenderError::Bytes { .. })
    ));
}

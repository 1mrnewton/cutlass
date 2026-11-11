use cutlass_effects::{
    AdjustKey, CanvasMask, Chroma, ClipLut, Colour, Lut, Mask, MaskShape, WheelSetting,
};
use cutlass_gpu::Placement;

use crate::{Layer, Renderer, Rgba};

fn renderer() -> Option<Renderer> {
    let renderer = Renderer::for_test().expect("a GPU is required");
    if renderer.is_none() {
        eprintln!("skipped: no GPU");
    }
    renderer
}

/// A picture with every hue, light and dark, some of it partly clear and a
/// column in seven fully clear, premultiplied.
fn scene(width: u32, height: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let hue = x as f32 / width as f32 * 6.0;
            let light = 0.1 + 0.85 * y as f32 / height as f32;
            let part = |offset: f32| {
                let d = ((hue + offset) % 6.0 - 3.0).abs();
                ((d - 1.0).clamp(0.0, 1.0) * 0.8 + 0.2) * light
            };
            let colour = [part(0.0), part(2.0), part(4.0)];
            let alpha: f32 = match x % 7 {
                0 => 0.0,
                3 => 0.5,
                5 => 0.8,
                _ => 1.0,
            };
            let byte = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
            let a = byte(alpha);
            for channel in colour {
                let straight = byte(channel);
                bytes.push((f32::from(straight) * f32::from(a) / 255.0).round() as u8);
            }
            bytes.push(a);
        }
    }
    bytes
}

fn on_gpu(
    renderer: &Renderer,
    bytes: &[u8],
    size: (u32, u32),
    colour: &Colour,
    seed: u32,
    mask: Option<&CanvasMask>,
) -> Vec<u8> {
    let canvas = renderer.canvas(size.0, size.1).expect("canvas");
    let picture = Rgba {
        width: size.0,
        height: size.1,
        bytes,
    };
    renderer
        .draw(
            &canvas,
            &[
                Layer::Rgba(picture, Placement::canvas(size.0, size.1)),
                Layer::Colour { colour, seed, mask },
            ],
        )
        .expect("draw");
    renderer.read(&canvas).expect("read")
}

fn on_cpu(bytes: &[u8], size: (u32, u32), colour: &Colour, seed: u32) -> Vec<u8> {
    let mut out = bytes.to_vec();
    colour.apply(&mut out, size.0, size.1, seed);
    out
}

/// Every byte within `reach` of the CPU's, and nearly all of them exact or
/// one off, so a pass that drifts everywhere by a little still fails.
fn matches(label: &str, got: &[u8], expected: &[u8], reach: u8) {
    assert_eq!(got.len(), expected.len(), "{label}");
    let mut near = 0;
    for (index, (a, b)) in got.iter().zip(expected).enumerate() {
        let off = a.abs_diff(*b);
        assert!(
            off <= reach,
            "{label}: byte {index} (pixel {}, channel {}) is {a}, the CPU has {b}",
            index / 4,
            index % 4
        );
        near += usize::from(off <= 1);
    }
    assert!(
        near * 100 >= got.len() * 98,
        "{label}: only {near} of {} bytes within one",
        got.len()
    );
}

fn check(renderer: &Renderer, label: &str, colour: &Colour, reach: u8) {
    let size = (42, 30);
    let bytes = scene(size.0, size.1);
    let expected = on_cpu(&bytes, size, colour, 9);
    assert_ne!(expected, bytes, "{label} changes the scene");
    let got = on_gpu(renderer, &bytes, size, colour, 9, None);
    matches(label, &got, &expected, reach);
}

#[test]
fn every_adjust_slider_matches_the_cpu() {
    let Some(renderer) = renderer() else { return };
    for key in AdjustKey::ALL {
        for value in [0.6, -0.6] {
            if value < key.min() {
                continue;
            }
            let mut colour = Colour::default();
            colour.adjust.set(key, value);
            check(&renderer, &format!("{key:?} {value}"), &colour, 2);
        }
    }
}

#[test]
fn sliders_together_with_clarity_and_sharpen_match_the_cpu() {
    let Some(renderer) = renderer() else { return };
    let mut colour = Colour::default();
    for (key, value) in [
        (AdjustKey::Exposure, 0.3),
        (AdjustKey::Contrast, 0.4),
        (AdjustKey::Saturation, -0.3),
        (AdjustKey::Clarity, 0.8),
        (AdjustKey::Sharpen, 0.7),
        (AdjustKey::Vignette, 0.5),
        (AdjustKey::Grain, 0.4),
    ] {
        colour.adjust.set(key, value);
    }
    check(&renderer, "together", &colour, 2);
    // A picture big enough that clarity reaches past two pixels.
    let size = (320, 240);
    let bytes = scene(size.0, size.1);
    let expected = on_cpu(&bytes, size, &colour, 3);
    let got = on_gpu(&renderer, &bytes, size, &colour, 3, None);
    matches("wide clarity", &got, &expected, 2);
}

#[test]
fn the_grain_follows_its_seed() {
    let Some(renderer) = renderer() else { return };
    let mut colour = Colour::default();
    colour.adjust.set(AdjustKey::Grain, 1.0);
    let size = (42, 30);
    let bytes = scene(size.0, size.1);
    let first = on_gpu(&renderer, &bytes, size, &colour, 1, None);
    assert_eq!(first, on_gpu(&renderer, &bytes, size, &colour, 1, None));
    assert_ne!(first, on_gpu(&renderer, &bytes, size, &colour, 2, None));
}

#[test]
fn curves_wheels_and_hsl_match_the_cpu() {
    let Some(renderer) = renderer() else { return };
    let mut curves = Colour::default();
    curves.grade.curves[0] = vec![(0.0, 0.05), (0.4, 0.3), (1.0, 0.95)];
    curves.grade.curves[1] = vec![(0.2, 0.0), (0.7, 0.9)];
    curves.grade.curves[3] = vec![(0.0, 0.3), (0.5, 0.5), (1.0, 0.7)];
    check(&renderer, "curves", &curves, 2);
    let mut wheels = Colour::default();
    wheels.grade.wheels = [
        WheelSetting {
            x: 0.4,
            y: -0.2,
            luma: 0.1,
        },
        WheelSetting {
            x: -0.3,
            y: 0.5,
            luma: -0.2,
        },
        WheelSetting {
            x: 0.1,
            y: 0.3,
            luma: 0.3,
        },
    ];
    check(&renderer, "wheels", &wheels, 2);
    for index in 0..8 {
        let mut hsl = Colour::default();
        hsl.grade.hsl[index] = [0.7, -0.5, 0.6];
        check(&renderer, &format!("hsl {index}"), &hsl, 2);
    }
    let mut look = Colour::default();
    look.look.hsl[4] = [-0.6, 0.8, -0.4];
    look.grade.curves[2] = vec![(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)];
    check(&renderer, "look then grade", &look, 2);
}

#[test]
fn a_chroma_key_matches_the_cpu() {
    let Some(renderer) = renderer() else { return };
    for (color, shadow) in [([0.0, 1.0, 0.0], 0.0), ([0.1, 0.3, 0.9], 0.6)] {
        let colour = Colour {
            chroma: Some(Chroma {
                color,
                strength: 0.5,
                shadow,
            }),
            ..Colour::default()
        };
        check(&renderer, &format!("key {color:?}"), &colour, 2);
    }
}

#[test]
fn a_lut_matches_the_cpu_at_any_strength() {
    let Some(renderer) = renderer() else { return };
    // Red and blue swapped and the shadows lifted, five points a side.
    let side = 5;
    let mut text = format!("LUT_3D_SIZE {side}\n");
    let at = |index: usize| index as f32 / (side - 1) as f32;
    let lift = |value: f32| 0.1 + value * 0.85;
    for b in 0..side {
        for g in 0..side {
            for r in 0..side {
                text.push_str(&format!(
                    "{} {} {}\n",
                    lift(at(b)),
                    lift(at(g)).powf(1.3),
                    lift(at(r))
                ));
            }
        }
    }
    let lut = Lut::parse_cube(&text).expect("cube");
    for strength in [1.0, 0.4] {
        let colour = Colour {
            lut: Some(ClipLut {
                name: "swap".into(),
                lut: lut.clone(),
                strength,
            }),
            ..Colour::default()
        };
        check(&renderer, &format!("lut {strength}"), &colour, 2);
    }
}

#[test]
fn all_of_it_at_once_matches_the_cpu() {
    let Some(renderer) = renderer() else { return };
    let mut colour = Colour {
        chroma: Some(Chroma {
            color: [0.2, 0.8, 0.2],
            strength: 0.3,
            shadow: 0.2,
        }),
        lut: Some(ClipLut {
            name: "id".into(),
            lut: Lut::identity(9),
            strength: 0.7,
        }),
        ..Colour::default()
    };
    colour.look.hsl[0] = [0.3, 0.2, 0.0];
    colour.grade.curves[0] = vec![(0.0, 0.1), (1.0, 0.9)];
    colour.grade.wheels[2] = WheelSetting {
        x: 0.2,
        y: 0.2,
        luma: 0.1,
    };
    colour.adjust.set(AdjustKey::Temperature, 0.5);
    colour.adjust.set(AdjustKey::Clarity, 0.5);
    colour.adjust.set(AdjustKey::Fade, 0.3);
    check(&renderer, "everything", &colour, 2);
}

#[test]
fn colour_inside_a_mask_matches_the_cpu_limit() {
    let Some(renderer) = renderer() else { return };
    let size = (42u32, 30u32);
    let bytes = scene(size.0, size.1);
    let mut colour = Colour::default();
    colour.adjust.set(AdjustKey::Exposure, 0.8);
    let frame = [
        2.0 / size.0 as f32,
        0.0,
        -1.0,
        0.0,
        -2.0 / size.1 as f32,
        1.0,
    ];
    let mask = CanvasMask {
        mask: Mask {
            feather: 0.4,
            ..Mask::new(MaskShape::Circle)
        },
        frame,
    };
    let mut expected = on_cpu(&bytes, size, &colour, 0);
    mask.mask.limit(&bytes, &mut expected, size.0, |x, y| {
        let [a, b, c, d, e, f] = frame;
        Some((a * x + b * y + c, d * x + e * y + f))
    });
    let got = on_gpu(&renderer, &bytes, size, &colour, 0, Some(&mask));
    matches("masked", &got, &expected, 2);
}

#[test]
fn colour_that_does_nothing_leaves_the_picture() {
    let Some(renderer) = renderer() else { return };
    let size = (42, 30);
    let bytes = scene(size.0, size.1);
    let got = on_gpu(&renderer, &bytes, size, &Colour::default(), 0, None);
    assert_eq!(got, bytes);
}

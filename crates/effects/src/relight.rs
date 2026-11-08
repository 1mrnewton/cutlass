//! Relight: a clip lit again by lights that are not in the shot, from the
//! depth of its frames.
//!
//! The depth (near is light, far is dark, one byte a pixel) comes from a
//! model, made once per clip. Here it becomes the way each surface faces:
//! where depth changes fast across the picture, the surface turns away.
//! Each light then brightens and tints the surfaces that face it, as
//! diffuse light does, wrapped a little around their edges; a light from
//! behind outlines the edges that turn away from the camera. A light can
//! reach the whole scene, or mostly what is near, so a face lit from the
//! side keeps the room behind it as it was.
//!
//! The light only ever multiplies the clip's own colour: black stays
//! black, and a light cannot invent detail the clip does not have. The work
//! is done in linear light and the clip keeps its alpha.
//!
//! Presets come in three groups, as the inspector shows them: facial
//! lights for a person, ambient colour lights for a scene, and creative
//! ones that move over the clip's time. The Custom light is one light of a
//! chosen colour from a chosen side.

use cutlass_model::Relight;

use crate::chroma::parse_hex;

/// How much a change of the whole depth range across the picture's short
/// side tilts a surface. Larger shapes the light more strongly.
const RELIEF: f32 = 1.6;

/// How far diffuse light wraps around an edge: 0 stops at the edge.
const WRAP: f32 = 0.35;

/// The most lights one preset uses.
pub const MAX_LIGHTS: usize = 4;

/// The inspector's groups of presets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelightGroup {
    /// Lights for a face or a person.
    Facial,
    /// Coloured light over the whole scene.
    Ambient,
    /// Lights that move or change over time.
    Creative,
}

impl RelightGroup {
    pub const ALL: [Self; 3] = [Self::Facial, Self::Ambient, Self::Creative];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Facial => "Facial",
            Self::Ambient => "Ambient",
            Self::Creative => "Creative",
        }
    }
}

/// One light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light {
    /// Linear colour, 1 is white.
    pub color: [f32; 3],
    /// Towards the light: x to the right, y up, z towards the camera. Any
    /// length; it is made unit when used.
    pub toward: [f32; 3],
    /// How bright, 1 about doubling a surface that faces it.
    pub power: f32,
    /// 0 lights every depth alike; 1 lights mostly what is near.
    pub reach: f32,
    /// 1 is soft diffuse light; larger narrows it to the surfaces facing
    /// it most squarely, as a beam.
    pub focus: f32,
}

/// The light over a clip at one moment.
#[derive(Clone, Debug, PartialEq)]
pub struct Lighting {
    /// What the clip keeps of its own light, per channel.
    pub ambient: [f32; 3],
    pub lights: Vec<Light>,
    /// How much of the lit picture replaces the clip's own, 0 to 1.
    pub strength: f32,
}

/// A preset: its stored id, the name shown, its group and its light over
/// time in seconds from the clip's start.
pub struct RelightPreset {
    pub id: &'static str,
    pub name: &'static str,
    pub group: RelightGroup,
    light: fn(f64) -> (Vec<Light>, [f32; 3]),
}

impl RelightPreset {
    /// Whether the light changes over time.
    #[must_use]
    pub fn moves(&self) -> bool {
        self.group == RelightGroup::Creative
    }
}

fn light(color: [f32; 3], toward: [f32; 3], power: f32, reach: f32) -> Light {
    Light {
        color,
        toward,
        power,
        reach,
        focus: 1.0,
    }
}

/// A fully saturated colour at `hue` degrees, linear.
fn hue(degrees: f64) -> [f32; 3] {
    let h = degrees.rem_euclid(360.0) / 60.0;
    let x = (1.0 - (h % 2.0 - 1.0).abs()) as f32;
    let [r, g, b] = match h as u32 {
        0 => [1.0, x, 0.0],
        1 => [x, 1.0, 0.0],
        2 => [0.0, 1.0, x],
        3 => [0.0, x, 1.0],
        4 => [x, 0.0, 1.0],
        _ => [1.0, 0.0, x],
    };
    [r, g, b]
}

/// A number from 0 to 1 that changes every step of `t` but not within it.
fn flicker(step: i64) -> f32 {
    let mixed = (step as u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .rotate_left(17)
        .wrapping_mul(0xbf58_476d_1ce4_e5b9);
    (mixed >> 40) as f32 / (1_u64 << 24) as f32
}

const GREY: [f32; 3] = [0.55, 0.55, 0.55];

/// Every preset, in the order the inspector shows them.
pub const RELIGHT_PRESETS: [RelightPreset; 19] = [
    RelightPreset {
        id: "warm_key",
        name: "Warm key",
        group: RelightGroup::Facial,
        light: |_| {
            (
                vec![
                    light([1.0, 0.72, 0.45], [-0.55, 0.45, 0.7], 0.95, 0.7),
                    light([0.55, 0.65, 0.9], [0.8, 0.1, 0.6], 0.2, 0.7),
                ],
                [0.6, 0.57, 0.55],
            )
        },
    },
    RelightPreset {
        id: "soft_front",
        name: "Soft front",
        group: RelightGroup::Facial,
        light: |_| {
            (
                vec![light([1.0, 0.96, 0.92], [0.0, 0.25, 1.0], 0.6, 0.6)],
                [0.7, 0.7, 0.7],
            )
        },
    },
    RelightPreset {
        id: "window_side",
        name: "Window side",
        group: RelightGroup::Facial,
        light: |_| {
            (
                vec![light([0.82, 0.9, 1.0], [0.9, 0.25, 0.4], 1.2, 0.7)],
                [0.38, 0.38, 0.4],
            )
        },
    },
    RelightPreset {
        id: "morning_sun",
        name: "Morning sun",
        group: RelightGroup::Facial,
        light: |_| {
            (
                vec![
                    light([1.0, 0.75, 0.4], [-0.85, 0.2, 0.5], 1.0, 0.6),
                    light([1.0, 0.7, 0.4], [-0.6, 0.3, -0.75], 0.6, 0.6),
                ],
                [0.5, 0.48, 0.47],
            )
        },
    },
    RelightPreset {
        id: "rim_glow",
        name: "Rim glow",
        group: RelightGroup::Facial,
        light: |_| {
            (
                vec![
                    light([1.0, 0.97, 0.92], [0.35, 0.45, -0.85], 1.3, 0.8),
                    light([1.0, 1.0, 1.0], [0.0, 0.1, 1.0], 0.25, 0.5),
                ],
                [0.55, 0.55, 0.57],
            )
        },
    },
    RelightPreset {
        id: "neon_two_tone",
        name: "Neon two-tone",
        group: RelightGroup::Ambient,
        light: |_| {
            (
                vec![
                    light([1.0, 0.15, 0.75], [-0.85, 0.2, 0.45], 1.1, 0.3),
                    light([0.15, 0.75, 1.0], [0.85, 0.2, 0.45], 1.1, 0.3),
                ],
                [0.32, 0.3, 0.38],
            )
        },
    },
    RelightPreset {
        id: "amber_and_blue",
        name: "Amber and blue",
        group: RelightGroup::Ambient,
        light: |_| {
            (
                vec![
                    light([1.0, 0.6, 0.15], [-0.8, 0.3, 0.5], 1.0, 0.3),
                    light([0.25, 0.45, 1.0], [0.8, 0.2, 0.5], 0.9, 0.3),
                ],
                [0.4, 0.38, 0.4],
            )
        },
    },
    RelightPreset {
        id: "teal_back",
        name: "Teal back",
        group: RelightGroup::Ambient,
        light: |_| {
            (
                vec![
                    light([0.1, 0.9, 0.75], [0.4, 0.3, -0.85], 1.4, 0.5),
                    light([1.0, 0.35, 0.2], [-0.6, 0.2, 0.75], 0.7, 0.3),
                ],
                [0.38, 0.4, 0.4],
            )
        },
    },
    RelightPreset {
        id: "blue_back",
        name: "Blue back",
        group: RelightGroup::Ambient,
        light: |_| {
            (
                vec![
                    light([0.2, 0.5, 1.0], [-0.3, 0.35, -0.85], 1.5, 0.5),
                    light([0.6, 0.75, 1.0], [0.0, 0.2, 1.0], 0.35, 0.3),
                ],
                [0.36, 0.4, 0.5],
            )
        },
    },
    RelightPreset {
        id: "green_edge",
        name: "Green edge",
        group: RelightGroup::Ambient,
        light: |_| {
            (
                vec![
                    light([0.3, 1.0, 0.25], [0.9, 0.15, -0.4], 1.3, 0.5),
                    light([1.0, 0.95, 0.9], [-0.3, 0.3, 0.9], 0.35, 0.3),
                ],
                [0.5, 0.5, 0.48],
            )
        },
    },
    RelightPreset {
        id: "red_and_pink",
        name: "Red and pink",
        group: RelightGroup::Ambient,
        light: |_| {
            (
                vec![
                    light([1.0, 0.12, 0.1], [-0.85, 0.15, 0.4], 1.1, 0.3),
                    light([1.0, 0.45, 0.8], [0.7, 0.4, 0.5], 0.8, 0.3),
                ],
                [0.42, 0.36, 0.38],
            )
        },
    },
    RelightPreset {
        id: "violet_haze",
        name: "Violet haze",
        group: RelightGroup::Ambient,
        light: |_| {
            (
                vec![light([0.7, 0.45, 1.0], [0.1, 0.4, 0.9], 0.8, 0.2)],
                [0.5, 0.4, 0.62],
            )
        },
    },
    RelightPreset {
        id: "night_green",
        name: "Night green",
        group: RelightGroup::Ambient,
        light: |_| {
            (
                vec![light([0.35, 1.0, 0.7], [-0.5, 0.5, 0.7], 0.9, 0.4)],
                [0.22, 0.32, 0.3],
            )
        },
    },
    RelightPreset {
        id: "rainbow_spots",
        name: "Rainbow spots",
        group: RelightGroup::Creative,
        light: |t| {
            let lights = (0..3)
                .map(|i| {
                    let turn = t * 1.2 + f64::from(i) * std::f64::consts::TAU / 3.0;
                    light(
                        hue(t * 60.0 + f64::from(i) * 120.0),
                        [turn.cos() as f32, (turn.sin() * 0.6) as f32, 0.6],
                        0.8,
                        0.4,
                    )
                })
                .collect();
            (lights, [0.35, 0.35, 0.38])
        },
    },
    RelightPreset {
        id: "color_sweep",
        name: "Colour sweep",
        group: RelightGroup::Creative,
        light: |t| {
            let across = (t * 2.0).sin() as f32;
            (
                vec![light(hue(t * 90.0), [across, 0.2, 0.55], 1.3, 0.4)],
                [0.4, 0.4, 0.42],
            )
        },
    },
    RelightPreset {
        id: "color_switch",
        name: "Colour switch",
        group: RelightGroup::Creative,
        light: |t| {
            const PALETTE: [[f32; 3]; 4] = [
                [1.0, 0.2, 0.6],
                [0.2, 0.6, 1.0],
                [1.0, 0.75, 0.15],
                [0.4, 1.0, 0.4],
            ];
            let beat = (t * 2.0).floor() as usize;
            (
                vec![
                    light(PALETTE[beat % 4], [-0.8, 0.25, 0.5], 1.0, 0.35),
                    light(PALETTE[(beat + 2) % 4], [0.8, 0.25, 0.5], 1.0, 0.35),
                ],
                [0.35, 0.35, 0.38],
            )
        },
    },
    RelightPreset {
        id: "laser",
        name: "Laser",
        group: RelightGroup::Creative,
        light: |t| {
            let tilt = (t * 1.5).sin() as f32;
            (
                vec![
                    Light {
                        focus: 12.0,
                        ..light([0.75, 0.95, 1.0], [0.3, tilt, 0.7], 2.2, 0.5)
                    },
                    light([0.9, 0.4, 1.0], [-0.6, 0.2, 0.6], 0.4, 0.3),
                ],
                [0.4, 0.38, 0.45],
            )
        },
    },
    RelightPreset {
        id: "color_orbit",
        name: "Colour orbit",
        group: RelightGroup::Creative,
        light: |t| {
            let turn = t * std::f64::consts::TAU / 3.0;
            (
                vec![light(
                    hue(turn.to_degrees()),
                    [turn.cos() as f32, 0.3, turn.sin() as f32],
                    1.3,
                    0.5,
                )],
                [0.42, 0.42, 0.44],
            )
        },
    },
    RelightPreset {
        id: "flicker",
        name: "Flicker",
        group: RelightGroup::Creative,
        light: |t| {
            let step = (t * 12.0).floor() as i64;
            let power = 0.3 + 1.2 * flicker(step);
            let color = if step % 6 < 3 {
                [1.0, 0.3, 0.55]
            } else {
                [0.3, 0.7, 1.0]
            };
            (
                vec![light(color, [-0.4, 0.4, 0.8], power, 0.4)],
                [0.38, 0.38, 0.4],
            )
        },
    },
];

/// The preset stored as `id`, when there is one. Custom is not a preset.
#[must_use]
pub fn relight_preset(id: &str) -> Option<&'static RelightPreset> {
    RELIGHT_PRESETS.iter().find(|preset| preset.id == id)
}

/// The Custom light's colour when none is chosen: a soft warm white.
pub const CUSTOM_COLOR: [f32; 3] = [1.0, 0.85, 0.7];

/// `relight`'s light `seconds` after the clip starts, or `None` for a
/// preset this build does not know.
#[must_use]
pub fn lighting(relight: &Relight, seconds: f64) -> Option<Lighting> {
    let seconds = if seconds.is_finite() {
        seconds.max(0.0)
    } else {
        0.0
    };
    let (lights, ambient) = if relight.preset == Relight::CUSTOM {
        let color = parse_hex(&relight.color).map_or(CUSTOM_COLOR, |srgb| srgb.map(to_linear));
        let angle = relight.angle.to_radians();
        // A third of the way up from level, in front.
        let toward = [(angle.cos() * 0.8) as f32, (angle.sin() * 0.8) as f32, 0.6];
        (vec![light(color, toward, 1.1, 0.5)], GREY)
    } else {
        (relight_preset(&relight.preset)?.light)(seconds)
    };
    Some(Lighting {
        ambient,
        lights: lights.into_iter().take(MAX_LIGHTS).collect(),
        strength: relight.strength.clamp(0.0, 1.0) as f32,
    })
}

/// An sRGB value from 0 to 1 in linear light.
fn to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// A linear value back in sRGB from 0 to 1.
fn to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Each byte in linear light.
fn linear_table() -> [f32; 256] {
    std::array::from_fn(|byte| to_linear(byte as f32 / 255.0))
}

/// Steps of linear light back to bytes: fine near black, where sRGB is.
const SRGB_STEPS: usize = 4096;

fn srgb_table() -> Vec<u8> {
    (0..=SRGB_STEPS)
        .map(|step| {
            let linear = (step as f32 / SRGB_STEPS as f32).powi(2);
            (to_srgb(linear) * 255.0).round() as u8
        })
        .collect()
}

/// A linear value as a byte through `table`, which is indexed by its
/// square root so dark values keep their steps.
fn to_byte(table: &[u8], linear: f32) -> u8 {
    let at = (linear.clamp(0.0, 1.0).sqrt() * SRGB_STEPS as f32).round() as usize;
    table.get(at).copied().unwrap_or(255)
}

fn unit(v: [f32; 3]) -> [f32; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if length > 1e-6 {
        v.map(|c| c / length)
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// How much of `light` reaches a surface facing `normal` at depth `near`
/// (0 far to 1 near).
fn reached(light: &Light, toward: [f32; 3], normal: [f32; 3], near: f32) -> f32 {
    let facing = toward[0] * normal[0] + toward[1] * normal[1] + toward[2] * normal[2];
    let mut diffuse = ((facing + WRAP) / (1.0 + WRAP)).max(0.0);
    if light.focus > 1.0 {
        diffuse = diffuse.powf(light.focus);
    }
    // A light from behind outlines the edges that turn from the camera,
    // on the side it comes from.
    let rim = if toward[2] < 0.0 {
        let side = (normal[0] * toward[0] + normal[1] * toward[1]).max(0.0);
        let across = (toward[0] * toward[0] + toward[1] * toward[1])
            .sqrt()
            .max(0.2);
        (1.0 - normal[2]).max(0.0).powf(1.5) * side / across * 2.0
    } else {
        0.0
    };
    let closeness = near * near * (3.0 - 2.0 * near);
    let depth_weight = 1.0 - light.reach * (1.0 - closeness);
    light.power * (diffuse + rim) * depth_weight
}

/// The surface direction at each pixel of a `width` by `height` depth
/// map, from how fast depth changes around it.
fn normal_at(
    depth: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    step: usize,
    tilt: f32,
) -> [f32; 3] {
    let at = |x: usize, y: usize| f32::from(depth[y * width + x]) / 255.0;
    let (left, right) = (x.saturating_sub(step), (x + step).min(width - 1));
    let (up, down) = (y.saturating_sub(step), (y + step).min(height - 1));
    let across = (at(right, y) - at(left, y)) / (right - left).max(1) as f32;
    let downward = (at(x, down) - at(x, up)) / (down - up).max(1) as f32;
    // Near is high: a surface tilts away from where depth falls. Picture
    // rows run down, the light's y up.
    unit([-across * tilt, downward * tilt, 1.0])
}

/// Light `rgba`, `width` by `height`, again with `lighting`, from `depth`:
/// one byte a pixel at the same size, near light. Alpha stays.
pub fn relight_rgba(rgba: &mut [u8], width: u32, height: u32, depth: &[u8], lighting: &Lighting) {
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 || rgba.len() != w * h * 4 || depth.len() != w * h {
        return;
    }
    let strength = lighting.strength.clamp(0.0, 1.0);
    if strength <= 0.0 {
        return;
    }
    let short = w.min(h);
    let step = (short / 160).max(1);
    let tilt = RELIEF * short as f32;
    let lights: Vec<(Light, [f32; 3])> = lighting
        .lights
        .iter()
        .map(|light| (*light, unit(light.toward)))
        .collect();
    let linear = linear_table();
    let srgb = srgb_table();
    for y in 0..h {
        for x in 0..w {
            let index = y * w + x;
            let normal = normal_at(depth, w, h, x, y, step, tilt);
            let near = f32::from(depth[index]) / 255.0;
            let mut gain = lighting.ambient;
            for (light, toward) in &lights {
                let amount = reached(light, *toward, normal, near);
                for (g, c) in gain.iter_mut().zip(light.color) {
                    *g += amount * c;
                }
            }
            let pixel = &mut rgba[index * 4..index * 4 + 3];
            for (byte, g) in pixel.iter_mut().zip(gain) {
                let own = linear[usize::from(*byte)];
                let lit = own * g;
                *byte = to_byte(&srgb, own + (lit - own) * strength);
            }
        }
    }
}

/// A `size` by `size` RGBA picture of a grey ball on a dark floor lit by
/// `lighting`, as a preset's tile.
#[must_use]
pub fn relight_tile(lighting: &Lighting, size: u32) -> Vec<u8> {
    let n = size.max(2) as usize;
    let mut rgba = Vec::with_capacity(n * n * 4);
    let mut depth = Vec::with_capacity(n * n);
    let radius = n as f32 * 0.34;
    let (cx, cy) = (n as f32 * 0.5, n as f32 * 0.48);
    for y in 0..n {
        for x in 0..n {
            let dx = (x as f32 + 0.5 - cx) / radius;
            let dy = (y as f32 + 0.5 - cy) / radius;
            let d2 = dx * dx + dy * dy;
            if d2 < 1.0 {
                rgba.extend_from_slice(&[200, 196, 190, 255]);
                // A ball rises to the camera at its middle.
                depth.push((150.0 + 105.0 * (1.0 - d2).sqrt()) as u8);
            } else {
                let floor = (40.0 + 30.0 * y as f32 / n as f32) as u8;
                rgba.extend_from_slice(&[floor + 40, floor + 40, floor + 44, 255]);
                depth.push((30.0 + 90.0 * y as f32 / n as f32) as u8);
            }
        }
    }
    let shown = Lighting {
        strength: 1.0,
        ..lighting.clone()
    };
    relight_rgba(&mut rgba, size.max(2), size.max(2), &depth, &shown);
    rgba
}

#[cfg(test)]
mod tests;

//! A clip's colour work as GPU passes: the numbers each pass reads and the
//! WGSL that runs it.
//!
//! The passes come in [`Colour::apply`]'s order: the chroma key, the look's
//! grade, the clip's grade, the adjust sliders, then the LUT. Each pass
//! reads premultiplied RGBA and writes it again, rounded to bytes as the
//! CPU rounds between its steps. The arithmetic follows [`Chroma::apply`],
//! [`Grade::apply`], [`adjust_rgba`] and [`ClipLut::apply`] line for line,
//! so the two change together; `cutlass-render`'s tests hold each pass to
//! the CPU result.
//!
//! Every pass shares one uniform block, [`ColourPass::uniform`]: four whole
//! numbers the renderer fills (the target's width and height, the grain
//! seed and a blur radius), then [`COLOUR_VALUES`] vectors whose meaning
//! the stage names, the eight HSL colours, and the four curve tables.
//!
//! [`Chroma::apply`]: crate::Chroma::apply
//! [`adjust_rgba`]: crate::adjust_rgba
//! [`ClipLut::apply`]: crate::ClipLut::apply

use crate::adjust::{Adjust, AdjustKey};
use crate::chroma::Chroma;
use crate::grade::{Colour, Grade, curve_table, straight};
use crate::lut::Lut;

/// Vectors of stage values in the uniform block.
pub const COLOUR_VALUES: usize = 8;

/// Steps a curve table has.
const CURVE_STEPS: usize = 256;

/// Bytes in the uniform block: four whole numbers, the stage values, the
/// HSL colours and the curve tables, sixteen bytes a vector.
pub const COLOUR_UNIFORM_BYTES: usize = (1 + COLOUR_VALUES + 8 + CURVE_STEPS) * 16;

/// What one colour pass does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColourStage {
    Chroma,
    Grade,
    Adjust,
    Lut,
}

/// One pass of a clip's colour work.
#[derive(Clone, Debug, PartialEq)]
pub struct ColourPass<'a> {
    pub stage: ColourStage,
    /// The stage's values; see each stage's WGSL for what they hold.
    pub values: [[f32; 4]; COLOUR_VALUES],
    /// A grade's hue, saturation and lightness for each HSL colour.
    pub hsl: [[f32; 4]; 8],
    /// A grade's curves, one column a curve: all four, red, green, blue.
    pub curves: Option<Box<[[f32; 4]; CURVE_STEPS]>>,
    /// The table a LUT pass reads.
    pub lut: Option<&'a Lut>,
}

impl<'a> ColourPass<'a> {
    fn new(stage: ColourStage) -> Self {
        Self {
            stage,
            values: [[0.0; 4]; COLOUR_VALUES],
            hsl: [[0.0; 4]; 8],
            curves: None,
            lut: None,
        }
    }

    /// The uniform block as little-endian bytes, with `frame` (width,
    /// height, seed, radius) first.
    #[must_use]
    pub fn uniform(&self, frame: [u32; 4]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(COLOUR_UNIFORM_BYTES);
        for value in frame {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let curves = self.curves.as_deref();
        let vectors = self
            .values
            .iter()
            .chain(&self.hsl)
            .chain(curves.into_iter().flatten());
        for value in vectors.flatten() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.resize(COLOUR_UNIFORM_BYTES, 0);
        bytes
    }

    /// An adjust pass whose sliders look at neighbours: clarity or
    /// sharpen. It runs in several steps on unrounded colour.
    #[must_use]
    pub fn reads_neighbours(&self) -> bool {
        self.stage == ColourStage::Adjust && (self.values[3][1] != 0.0 || self.values[3][2] != 0.0)
    }

    /// Clarity, for an adjust pass.
    #[must_use]
    pub fn clarity(&self) -> f32 {
        self.values[3][1]
    }

    /// Sharpen, for an adjust pass.
    #[must_use]
    pub fn sharpen(&self) -> f32 {
        self.values[3][2]
    }
}

impl Colour {
    /// The passes that run this colour work, in order. Work that leaves the
    /// picture as it is has no pass.
    #[must_use]
    pub fn passes(&self) -> Vec<ColourPass<'_>> {
        let mut passes = Vec::new();
        if let Some(chroma) = self.chroma.filter(|chroma| chroma.strength > 0.0) {
            passes.push(chroma_pass(&chroma));
        }
        for grade in [&self.look, &self.grade] {
            if !grade.is_identity() {
                passes.push(grade_pass(grade));
            }
        }
        if !self.adjust.is_identity() {
            passes.push(adjust_pass(&self.adjust));
        }
        if let Some(lut) = &self.lut {
            let strength = lut.strength.clamp(0.0, 1.0);
            if strength > 0.0 {
                let mut pass = ColourPass::new(ColourStage::Lut);
                #[allow(clippy::cast_precision_loss)]
                let size = lut.lut.size() as f32;
                pass.values[0] = [strength, size, 0.0, 0.0];
                pass.lut = Some(&lut.lut);
                passes.push(pass);
            }
        }
        passes
    }
}

fn chroma_pass(chroma: &Chroma) -> ColourPass<'static> {
    let mut pass = ColourPass::new(ColourStage::Chroma);
    let [r, g, b] = chroma.color;
    pass.values[0] = [r, g, b, chroma.strength];
    let reach = 0.05 + chroma.strength.clamp(0.0, 1.0) * 0.6;
    let key_luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    pass.values[1] = [chroma.shadow, reach, 0.15, key_luma];
    pass
}

fn grade_pass(grade: &Grade) -> ColourPass<'static> {
    let mut pass = ColourPass::new(ColourStage::Grade);
    let mut on = [0.0; 4];
    let mut curves = Box::new([[0.0; 4]; CURVE_STEPS]);
    for (column, points) in grade.curves.iter().enumerate() {
        if straight(points) {
            continue;
        }
        on[column] = 1.0;
        for (step, value) in curve_table(points).into_iter().enumerate() {
            curves[step][column] = value;
        }
    }
    pass.values[0] = on;
    for (slot, wheel) in pass.values[1..4].iter_mut().zip(grade.wheels) {
        let [r, g, b] = wheel.offset();
        *slot = [r, g, b, 0.0];
    }
    let hsl_on = grade.hsl.iter().flatten().any(|value| *value != 0.0);
    pass.values[4] = [if hsl_on { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0];
    for (slot, [h, s, l]) in pass.hsl.iter_mut().zip(grade.hsl) {
        *slot = [h, s, l, 0.0];
    }
    pass.curves = Some(curves);
    pass
}

fn adjust_pass(adjust: &Adjust) -> ColourPass<'static> {
    let mut pass = ColourPass::new(ColourStage::Adjust);
    let get = |key| adjust.get(key);
    let exposure = get(AdjustKey::Exposure);
    let gain = if exposure == 0.0 {
        1.0
    } else {
        (exposure * 2.0).exp2()
    };
    pass.values[0] = [
        gain,
        get(AdjustKey::Temperature) * 0.2,
        get(AdjustKey::Tint) * 0.2,
        get(AdjustKey::Brightness) * 0.25,
    ];
    pass.values[1] = [
        1.0 + get(AdjustKey::Contrast),
        get(AdjustKey::Highlights),
        get(AdjustKey::Shadows),
        get(AdjustKey::Whites),
    ];
    pass.values[2] = [
        get(AdjustKey::Blacks),
        get(AdjustKey::Brilliance),
        1.0 + get(AdjustKey::Saturation),
        get(AdjustKey::Hue) * std::f32::consts::PI,
    ];
    pass.values[3] = [
        get(AdjustKey::Fade),
        get(AdjustKey::Clarity),
        get(AdjustKey::Sharpen),
        get(AdjustKey::Vignette),
    ];
    pass.values[4] = [get(AdjustKey::Grain), 0.0, 0.0, 0.0];
    pass
}

/// The box blur radius clarity uses on a `width` by `height` picture.
#[must_use]
pub fn clarity_radius(width: u32, height: u32) -> u32 {
    (width.min(height) / 100).max(2)
}

/// The colour passes' shared WGSL: the uniform block as `colour_params`
/// and one function a stage. A stage function takes the premultiplied
/// pixel; `adjust_finish` also takes the pixel's position. The renderer
/// binds `colour_params` and the LUT texture and writes the entry points.
pub const COLOUR_WGSL: &str = r"
struct ColourParams {
    // Width, height, grain seed, blur radius.
    frame: vec4<u32>,
    v: array<vec4<f32>, 8>,
    hsl: array<vec4<f32>, 8>,
    // At each of 256 steps: the all-channel curve, then red, green, blue.
    curves: array<vec4<f32>, 256>,
};

fn colour_luma(c: vec3<f32>) -> f32 {
    return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

fn colour_smooth(low: f32, high: f32, value: f32) -> f32 {
    let t = clamp((value - low) / (high - low), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn colour_midtones(level: f32) -> f32 {
    let l = clamp(level, 0.0, 1.0);
    return 4.0 * l * (1.0 - l);
}

// Straight colour, each channel at most 1; black where nothing is drawn.
fn colour_straight(c: vec4<f32>) -> vec3<f32> {
    if (c.a <= 0.0) {
        return vec3<f32>(0.0);
    }
    return min(c.rgb / c.a, vec3<f32>(1.0));
}

fn colour_out(rgb: vec3<f32>, alpha: f32) -> vec4<f32> {
    return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)) * alpha, alpha);
}

// The value `x` wrapped into 0 up to `m`, as Rust's rem_euclid.
fn colour_wrap(x: f32, m: f32) -> f32 {
    return x - m * floor(x / m);
}

// Chroma key. v[0]: key colour, strength. v[1]: shadow, reach, soft, key luma.
fn chroma_of(c: vec3<f32>) -> vec2<f32> {
    let y = colour_luma(c);
    let scale = max(y, 0.1);
    return vec2<f32>((c.b - y) * 0.539 / scale, (c.r - y) * 0.635 / scale);
}

fn chroma_key(c: vec4<f32>) -> vec4<f32> {
    if (c.a <= 0.0) {
        return c;
    }
    let key = colour_params.v[0].xyz;
    let shadow = colour_params.v[1].x;
    let reach = colour_params.v[1].y;
    let soft = colour_params.v[1].z;
    let key_luma = colour_params.v[1].w;
    let colour = colour_straight(c);
    let d = chroma_of(colour) - chroma_of(key);
    let distance = sqrt(d.x * d.x + d.y * d.y);
    var keep = colour_smooth(reach, reach + soft, distance);
    if (shadow > 0.0 && keep < 1.0) {
        let darker = clamp((key_luma - colour_luma(colour)) / max(key_luma, 1e-3), 0.0, 1.0);
        keep = max(keep, darker * shadow);
    }
    if (keep >= 1.0) {
        return c;
    }
    let spill = 1.0 - keep;
    let grey = vec3<f32>(colour_luma(colour));
    let alpha = c.a * keep;
    let excess = (colour - grey) * (key - vec3<f32>(key_luma));
    let pulled = grey + (colour - grey) * (1.0 - spill);
    let cleaned = select(colour, pulled, excess > vec3<f32>(0.0));
    return vec4<f32>(clamp(cleaned, vec3<f32>(0.0), vec3<f32>(1.0)) * alpha, alpha);
}

// Grade. v[0]: which curves are on. v[1], v[2], v[3]: the lift, gamma and
// gain offsets. v[4].x: 1 when any HSL colour is set.
fn grade_look_up(column: u32, value: f32) -> f32 {
    let position = clamp(value, 0.0, 1.0) * 255.0;
    let low = u32(floor(position));
    let high = min(low + 1u, 255u);
    let t = position - f32(low);
    return colour_params.curves[low][column] * (1.0 - t) + colour_params.curves[high][column] * t;
}

fn grade_to_hsl(c: vec3<f32>) -> vec3<f32> {
    let high = max(c.r, max(c.g, c.b));
    let low = min(c.r, min(c.g, c.b));
    let light = (high + low) / 2.0;
    let range = high - low;
    if (range <= 1e-6) {
        return vec3<f32>(0.0, 0.0, light);
    }
    let saturation = range / max(1.0 - abs(2.0 * light - 1.0), 1e-6);
    var hue = 0.0;
    if (high == c.r) {
        hue = 60.0 * colour_wrap((c.g - c.b) / range, 6.0);
    } else if (high == c.g) {
        hue = 60.0 * ((c.b - c.r) / range + 2.0);
    } else {
        hue = 60.0 * ((c.r - c.g) / range + 4.0);
    }
    return vec3<f32>(hue, min(saturation, 1.0), light);
}

fn grade_from_hsl(hue: f32, saturation: f32, light: f32) -> vec3<f32> {
    let chroma = (1.0 - abs(2.0 * light - 1.0)) * saturation;
    let sector = colour_wrap(hue, 360.0) / 60.0;
    let second = chroma * (1.0 - abs(colour_wrap(sector, 2.0) - 1.0));
    var rgb = vec3<f32>(chroma, 0.0, second);
    switch u32(sector) {
        case 0u: { rgb = vec3<f32>(chroma, second, 0.0); }
        case 1u: { rgb = vec3<f32>(second, chroma, 0.0); }
        case 2u: { rgb = vec3<f32>(0.0, chroma, second); }
        case 3u: { rgb = vec3<f32>(0.0, second, chroma); }
        case 4u: { rgb = vec3<f32>(second, 0.0, chroma); }
        default: {}
    }
    return rgb + vec3<f32>(light - chroma / 2.0);
}

fn grade_weight(index: u32, hue: f32) -> f32 {
    var centres = array<f32, 8>(0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 275.0, 310.0);
    let centre = centres[index];
    var next = centres[(index + 1u) % 8u];
    if (index == 7u) {
        next = next + 360.0;
    }
    var previous = centres[(index + 7u) % 8u];
    if (index == 0u) {
        previous = previous - 360.0;
    }
    var delta = colour_wrap(hue - centre, 360.0);
    if (delta > 180.0) {
        delta = delta - 360.0;
    }
    var reach = centre - previous;
    if (delta >= 0.0) {
        reach = next - centre;
    }
    return max(1.0 - abs(delta) / reach, 0.0);
}

fn grade_hsl(colour: vec3<f32>) -> vec3<f32> {
    let hsl = grade_to_hsl(colour);
    if (hsl.y <= 1e-4) {
        return colour;
    }
    var turn = 0.0;
    var scale = 0.0;
    var lift = 0.0;
    for (var index = 0u; index < 8u; index = index + 1u) {
        let weight = grade_weight(index, hsl.x);
        if (weight > 0.0) {
            let setting = colour_params.hsl[index];
            turn = turn + setting.x * weight;
            scale = scale + setting.y * weight;
            lift = lift + setting.z * weight;
        }
    }
    let saturation = clamp(hsl.y * (1.0 + scale), 0.0, 1.0);
    let light = clamp(hsl.z + lift * 0.3 * saturation, 0.0, 1.0);
    return grade_from_hsl(hsl.x + turn * 30.0, saturation, light);
}

fn grade_pixel(c: vec4<f32>) -> vec4<f32> {
    if (c.a <= 0.0) {
        return c;
    }
    var colour = colour_straight(c);
    let on = colour_params.v[0];
    if (on.x > 0.5) {
        colour = vec3<f32>(
            grade_look_up(0u, colour.r),
            grade_look_up(0u, colour.g),
            grade_look_up(0u, colour.b),
        );
    }
    if (on.y > 0.5) {
        colour.r = grade_look_up(1u, colour.r);
    }
    if (on.z > 0.5) {
        colour.g = grade_look_up(2u, colour.g);
    }
    if (on.w > 0.5) {
        colour.b = grade_look_up(3u, colour.b);
    }
    let lift = colour_params.v[1].xyz;
    let gamma = colour_params.v[2].xyz;
    let gain = colour_params.v[3].xyz;
    let lifted = colour + lift * (vec3<f32>(1.0) - colour);
    let gained = max(lifted * (vec3<f32>(1.0) + gain), vec3<f32>(0.0));
    let power = vec3<f32>(1.0) / max(vec3<f32>(1.0) + gamma, vec3<f32>(0.1));
    colour = select(pow(gained, power), vec3<f32>(0.0), gained <= vec3<f32>(0.0));
    if (colour_params.v[4].x > 0.5) {
        colour = grade_hsl(colour);
    }
    return colour_out(colour, c.a);
}

// Adjust. v[0]: exposure gain, warmth, tint, lift. v[1]: contrast,
// highlights, shadows, whites. v[2]: blacks, brilliance, saturation, hue
// turn in radians. v[3]: fade, clarity, sharpen, vignette. v[4].x: grain.
fn adjust_tone(colour: vec3<f32>) -> vec3<f32> {
    let p0 = colour_params.v[0];
    let p1 = colour_params.v[1];
    let p2 = colour_params.v[2];
    var c = colour * p0.x;
    c.r = c.r * (1.0 + p0.y);
    c.b = c.b * (1.0 - p0.y);
    c.g = c.g * (1.0 - p0.z);
    c = c + vec3<f32>(p0.w);
    if (p1.x != 1.0) {
        c = (c - vec3<f32>(0.5)) * p1.x + vec3<f32>(0.5);
    }
    let level = clamp(colour_luma(c), 0.0, 1.0);
    let shift = 0.25
        * (p1.y * colour_smooth(0.5, 1.0, level)
            + p1.z * (1.0 - colour_smooth(0.0, 0.5, level))
            + p1.w * colour_smooth(0.75, 1.0, level)
            + p2.x * (1.0 - colour_smooth(0.0, 0.25, level)))
        + 0.2 * p2.y * colour_midtones(level);
    c = c + vec3<f32>(shift);
    if (p2.z != 1.0) {
        let grey = vec3<f32>(colour_luma(c));
        c = grey + (c - grey) * p2.z;
    }
    if (p2.w != 0.0) {
        let s = sin(p2.w);
        let cs = cos(p2.w);
        let third = (1.0 - cs) / 3.0;
        let root = sqrt(1.0 / 3.0) * s;
        let a = cs + third;
        let b = third - root;
        let k = third + root;
        c = vec3<f32>(
            c.r * a + c.g * b + c.b * k,
            c.r * k + c.g * a + c.b * b,
            c.r * b + c.g * k + c.b * a,
        );
    }
    let fade = colour_params.v[3].x;
    if (fade != 0.0) {
        c = c * (1.0 - 0.35 * fade) + vec3<f32>(0.15 * fade);
    }
    return c;
}

fn adjust_hash(x: u32, y: u32, seed: u32) -> f32 {
    var h = (x * 0x8da6b343u) ^ (y * 0xd8163841u) ^ (seed * 0xcb1ab31fu);
    h = h ^ (h >> 13u);
    h = h * 0x5bd1e995u;
    h = h ^ (h >> 15u);
    return f32(h & 0x00ffffffu) / 16777215.0;
}

// Straight `colour` with the vignette and grain at `pixel`, premultiplied
// by `alpha`.
fn adjust_finish(colour: vec3<f32>, alpha: f32, pixel: vec2<u32>) -> vec4<f32> {
    var c = colour;
    let vignette = colour_params.v[3].w;
    if (vignette != 0.0) {
        let size = vec2<f32>(f32(colour_params.frame.x), f32(colour_params.frame.y));
        let d = (vec2<f32>(pixel) + vec2<f32>(0.5)) / size * 2.0 - vec2<f32>(1.0);
        let distance = sqrt(d.x * d.x + d.y * d.y) / 1.4142135;
        c = c * (1.0 - vignette * colour_smooth(0.35, 1.0, distance));
    }
    let grain = colour_params.v[4].x;
    if (grain != 0.0) {
        let noise = adjust_hash(pixel.x, pixel.y, colour_params.frame.z) - 0.5;
        c = c + vec3<f32>(noise * grain * 0.25);
    }
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)) * alpha, alpha);
}

// The work that does not look at neighbours, as unrounded straight colour;
// nothing where nothing is drawn.
fn adjust_toned(c: vec4<f32>) -> vec4<f32> {
    if (c.a <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, c.a);
    }
    return vec4<f32>(adjust_tone(colour_straight(c)), c.a);
}

// LUT. v[0]: strength, points a side.
fn lut_pixel(c: vec4<f32>) -> vec4<f32> {
    if (c.a <= 0.0) {
        return c;
    }
    let colour = colour_straight(c);
    let strength = colour_params.v[0].x;
    let size = i32(colour_params.v[0].y + 0.5);
    let position = clamp(colour, vec3<f32>(0.0), vec3<f32>(1.0)) * f32(size - 1);
    let low = min(vec3<i32>(floor(position)), vec3<i32>(size - 2));
    let part = position - vec3<f32>(low);
    var mapped = vec3<f32>(0.0);
    for (var corner = 0; corner < 8; corner = corner + 1) {
        let d = vec3<i32>(corner & 1, (corner >> 1u) & 1, (corner >> 2u) & 1);
        let w = select(vec3<f32>(1.0) - part, part, d == vec3<i32>(1));
        let weight = w.x * w.y * w.z;
        if (weight != 0.0) {
            mapped = mapped + textureLoad(colour_lut, low + d, 0).rgb * weight;
        }
    }
    return colour_out(colour + (mapped - colour) * strength, c.a);
}
";

#[cfg(test)]
mod tests {
    use super::{COLOUR_UNIFORM_BYTES, ColourStage, clarity_radius};
    use crate::adjust::AdjustKey;
    use crate::chroma::Chroma;
    use crate::grade::{Colour, Grade, WheelSetting};
    use crate::lut::{ClipLut, Lut};

    #[test]
    fn colour_that_does_nothing_has_no_pass() {
        let mut colour = Colour::default();
        assert!(colour.passes().is_empty());
        colour.chroma = Some(Chroma {
            color: [0.0, 1.0, 0.0],
            strength: 0.0,
            shadow: 0.5,
        });
        colour.lut = Some(ClipLut {
            name: "none".into(),
            lut: Lut::identity(5),
            strength: 0.0,
        });
        assert!(colour.passes().is_empty());
    }

    #[test]
    fn passes_run_in_the_cpu_order() {
        let mut colour = Colour {
            chroma: Some(Chroma {
                color: [0.0, 1.0, 0.0],
                strength: 0.4,
                shadow: 0.0,
            }),
            lut: Some(ClipLut {
                name: "id".into(),
                lut: Lut::identity(5),
                strength: 0.5,
            }),
            ..Colour::default()
        };
        colour.look.hsl[2] = [0.5, 0.0, 0.0];
        colour.grade.wheels[0] = WheelSetting {
            x: 0.2,
            y: 0.0,
            luma: 0.0,
        };
        colour.adjust.set(AdjustKey::Exposure, 0.5);
        let stages: Vec<ColourStage> = colour.passes().iter().map(|pass| pass.stage).collect();
        assert_eq!(
            stages,
            [
                ColourStage::Chroma,
                ColourStage::Grade,
                ColourStage::Grade,
                ColourStage::Adjust,
                ColourStage::Lut,
            ]
        );
        let passes = colour.passes();
        assert_eq!(passes[4].values[0], [0.5, 5.0, 0.0, 0.0]);
        assert!(passes[4].lut.is_some());
        // Exposure 0.5 doubles the light.
        assert!((passes[3].values[0][0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn a_grade_carries_only_its_bent_curves() {
        let mut grade = Grade::default();
        grade.curves[2] = vec![(0.0, 0.2), (1.0, 0.8)];
        let colour = Colour {
            grade,
            ..Colour::default()
        };
        let passes = colour.passes();
        let curves = passes[0].curves.as_ref().expect("curves");
        assert_eq!(passes[0].values[0], [0.0, 0.0, 1.0, 0.0]);
        assert!((curves[0][2] - 0.2).abs() < 1e-6);
        assert!((curves[255][2] - 0.8).abs() < 1e-6);
        assert_eq!(curves[128][0], 0.0);
    }

    #[test]
    fn the_uniform_is_one_size_for_every_pass() {
        let mut colour = Colour::default();
        colour.adjust.set(AdjustKey::Clarity, 0.3);
        colour.grade.curves[0] = vec![(0.0, 0.1), (1.0, 1.0)];
        for pass in colour.passes() {
            let bytes = pass.uniform([8, 4, 7, 2]);
            assert_eq!(bytes.len(), COLOUR_UNIFORM_BYTES);
            assert_eq!(bytes[..4], 8u32.to_le_bytes());
            assert_eq!(bytes[8..12], 7u32.to_le_bytes());
        }
        let passes = colour.passes();
        assert!(passes[1].reads_neighbours());
        assert!(!passes[0].reads_neighbours());
    }

    #[test]
    fn clarity_reaches_a_hundredth_of_the_short_side_and_at_least_two() {
        assert_eq!(clarity_radius(3840, 2160), 21);
        assert_eq!(clarity_radius(64, 4000), 2);
        assert_eq!(clarity_radius(1, 1), 2);
    }
}

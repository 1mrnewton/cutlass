//! A clip's adjust sliders: exposure, contrast, temperature and the rest.
//!
//! Each value is an effect material the clip lists among its extra
//! materials. The material's `type` is the reference editor's name for the
//! slider (`brightness`, `light_sensation`, `vignetting`, ...), and `value`
//! holds the setting. A slider at zero has no material.
//!
//! The pixels are premultiplied RGBA. The work runs on straight colour in
//! a fixed order: exposure, white balance, the tone sliders, saturation and
//! hue, fade, then clarity and sharpen (which look at neighbours), then
//! vignette and grain (which look at the position).

use cutlass_model::{Draft, MaterialRef, Segment};

/// The material `type` of an adjustment layer: a clip on an adjust row
/// whose sliders run over every picture below it.
pub const ADJUSTMENT: &str = "adjust";

/// One adjust slider.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AdjustKey {
    Exposure,
    Brightness,
    Contrast,
    Saturation,
    Brilliance,
    Highlights,
    Shadows,
    Whites,
    Blacks,
    Temperature,
    Tint,
    Hue,
    Fade,
    Clarity,
    Sharpen,
    Vignette,
    Grain,
}

const COUNT: usize = 17;

impl AdjustKey {
    /// Every slider, in the order the inspector shows them.
    pub const ALL: [Self; COUNT] = [
        Self::Exposure,
        Self::Brightness,
        Self::Contrast,
        Self::Saturation,
        Self::Brilliance,
        Self::Highlights,
        Self::Shadows,
        Self::Whites,
        Self::Blacks,
        Self::Temperature,
        Self::Tint,
        Self::Hue,
        Self::Fade,
        Self::Clarity,
        Self::Sharpen,
        Self::Vignette,
        Self::Grain,
    ];

    /// The material `type` the value is stored under.
    #[must_use]
    pub fn kind(self) -> &'static str {
        match self {
            Self::Exposure => "exposure",
            Self::Brightness => "brightness",
            Self::Contrast => "contrast",
            Self::Saturation => "saturation",
            Self::Brilliance => "light_sensation",
            Self::Highlights => "highlight",
            Self::Shadows => "shadow",
            Self::Whites => "white",
            Self::Blacks => "black",
            Self::Temperature => "temperature",
            Self::Tint => "tone",
            Self::Hue => "hue",
            Self::Fade => "fade",
            Self::Clarity => "clear",
            Self::Sharpen => "sharpen",
            Self::Vignette => "vignetting",
            Self::Grain => "particle",
        }
    }

    /// The slider's name.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Exposure => "Exposure",
            Self::Brightness => "Brightness",
            Self::Contrast => "Contrast",
            Self::Saturation => "Saturation",
            Self::Brilliance => "Brilliance",
            Self::Highlights => "Highlights",
            Self::Shadows => "Shadows",
            Self::Whites => "Whites",
            Self::Blacks => "Blacks",
            Self::Temperature => "Temp",
            Self::Tint => "Tint",
            Self::Hue => "Hue",
            Self::Fade => "Fade",
            Self::Clarity => "Clarity",
            Self::Sharpen => "Sharpen",
            Self::Vignette => "Vignette",
            Self::Grain => "Grain",
        }
    }

    /// The slider for a material `type`.
    #[must_use]
    pub fn from_kind(kind: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|key| key.kind() == kind)
    }

    /// The lowest value: 0 for a slider that only adds, else -1.
    #[must_use]
    pub fn min(self) -> f32 {
        match self {
            Self::Fade | Self::Sharpen | Self::Grain => 0.0,
            _ => -1.0,
        }
    }

    fn index(self) -> usize {
        Self::ALL
            .iter()
            .position(|key| *key == self)
            .unwrap_or_default()
    }
}

/// Every slider's value; zero leaves the picture as it is.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Adjust {
    values: [f32; COUNT],
}

impl Adjust {
    /// The value of `key`.
    #[must_use]
    pub fn get(&self, key: AdjustKey) -> f32 {
        self.values[key.index()]
    }

    /// Set `key`, held to its range. A value that is not a number is zero.
    pub fn set(&mut self, key: AdjustKey, value: f32) {
        let value = if value.is_finite() { value } else { 0.0 };
        self.values[key.index()] = value.clamp(key.min(), 1.0);
    }

    /// Every slider is at zero.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.values.iter().all(|value| *value == 0.0)
    }
}

/// The sliders `segment` sets through its extra materials.
#[must_use]
pub fn segment_adjust(draft: &Draft, segment: &Segment) -> Adjust {
    let mut adjust = Adjust::default();
    for id in &segment.extra_material_refs {
        if let Some(MaterialRef::Effect(effect)) = draft.materials.get(id.as_str())
            && let Some(key) = AdjustKey::from_kind(&effect.kind)
        {
            #[allow(clippy::cast_possible_truncation)]
            adjust.set(key, effect.value.unwrap_or_default() as f32);
        }
    }
    adjust
}

/// Run `adjust` over premultiplied RGBA `rgba`, `width` × `height`.
/// `seed` moves the grain; the same seed gives the same grain.
pub fn adjust_rgba(adjust: &Adjust, rgba: &mut [u8], width: u32, height: u32, seed: u32) {
    let (width, height) = (width as usize, height as usize);
    if adjust.is_identity() || width == 0 || height == 0 || rgba.len() < width * height * 4 {
        return;
    }
    let mut colour: Vec<[f32; 3]> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .take(width * height)
        .map(|pixel| {
            let alpha = f32::from(pixel[3]) / 255.0;
            if alpha <= 0.0 {
                return [0.0; 3];
            }
            let straight = |channel: u8| (f32::from(channel) / 255.0 / alpha).min(1.0);
            tone(
                adjust,
                [straight(pixel[0]), straight(pixel[1]), straight(pixel[2])],
            )
        })
        .collect();
    let clarity = adjust.get(AdjustKey::Clarity);
    let sharpen = adjust.get(AdjustKey::Sharpen);
    if clarity != 0.0 {
        let radius = (width.min(height) / 100).max(2);
        let blurred = box_blur(&colour, width, height, radius);
        for (pixel, soft) in colour.iter_mut().zip(&blurred) {
            let weight = midtones(luma(*pixel));
            for channel in 0..3 {
                pixel[channel] += clarity * weight * (pixel[channel] - soft[channel]);
            }
        }
    }
    if sharpen != 0.0 {
        let blurred = box_blur(&colour, width, height, 1);
        for (pixel, soft) in colour.iter_mut().zip(&blurred) {
            for channel in 0..3 {
                pixel[channel] += 1.5 * sharpen * (pixel[channel] - soft[channel]);
            }
        }
    }
    let vignette = adjust.get(AdjustKey::Vignette);
    let grain = adjust.get(AdjustKey::Grain);
    for (index, (pixel, out)) in colour
        .iter()
        .zip(rgba.as_chunks_mut::<4>().0.iter_mut())
        .enumerate()
    {
        let (x, y) = (index % width, index / width);
        let mut colour = *pixel;
        if vignette != 0.0 {
            #[allow(clippy::cast_precision_loss)]
            let (dx, dy) = (
                (x as f32 + 0.5) / width as f32 * 2.0 - 1.0,
                (y as f32 + 0.5) / height as f32 * 2.0 - 1.0,
            );
            let distance = (dx * dx + dy * dy).sqrt() / std::f32::consts::SQRT_2;
            let weight = smoothstep(0.35, 1.0, distance);
            for channel in &mut colour {
                *channel *= 1.0 - vignette * weight;
            }
        }
        if grain != 0.0 {
            #[allow(clippy::cast_possible_truncation)]
            let noise = hash(x as u32, y as u32, seed) - 0.5;
            for channel in &mut colour {
                *channel += noise * grain * 0.25;
            }
        }
        let alpha = f32::from(out[3]) / 255.0;
        for channel in 0..3 {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let value = (colour[channel].clamp(0.0, 1.0) * alpha * 255.0).round() as u8;
            out[channel] = value;
        }
    }
}

/// The work on one straight colour that does not look at neighbours.
fn tone(adjust: &Adjust, colour: [f32; 3]) -> [f32; 3] {
    let [mut r, mut g, mut b] = colour;
    let exposure = adjust.get(AdjustKey::Exposure);
    if exposure != 0.0 {
        let gain = (exposure * 2.0).exp2();
        r *= gain;
        g *= gain;
        b *= gain;
    }
    let warm = adjust.get(AdjustKey::Temperature) * 0.2;
    let tint = adjust.get(AdjustKey::Tint) * 0.2;
    r *= 1.0 + warm;
    b *= 1.0 - warm;
    g *= 1.0 - tint;
    let lift = adjust.get(AdjustKey::Brightness) * 0.25;
    r += lift;
    g += lift;
    b += lift;
    let contrast = 1.0 + adjust.get(AdjustKey::Contrast);
    if contrast != 1.0 {
        r = (r - 0.5) * contrast + 0.5;
        g = (g - 0.5) * contrast + 0.5;
        b = (b - 0.5) * contrast + 0.5;
    }
    let level = luma([r, g, b]).clamp(0.0, 1.0);
    let shift = 0.25
        * (adjust.get(AdjustKey::Highlights) * smoothstep(0.5, 1.0, level)
            + adjust.get(AdjustKey::Shadows) * (1.0 - smoothstep(0.0, 0.5, level))
            + adjust.get(AdjustKey::Whites) * smoothstep(0.75, 1.0, level)
            + adjust.get(AdjustKey::Blacks) * (1.0 - smoothstep(0.0, 0.25, level)))
        + 0.2 * adjust.get(AdjustKey::Brilliance) * midtones(level);
    r += shift;
    g += shift;
    b += shift;
    let saturation = 1.0 + adjust.get(AdjustKey::Saturation);
    if saturation != 1.0 {
        let grey = luma([r, g, b]);
        r = grey + (r - grey) * saturation;
        g = grey + (g - grey) * saturation;
        b = grey + (b - grey) * saturation;
    }
    let hue = adjust.get(AdjustKey::Hue);
    if hue != 0.0 {
        [r, g, b] = rotate_hue([r, g, b], hue * std::f32::consts::PI);
    }
    let fade = adjust.get(AdjustKey::Fade);
    if fade != 0.0 {
        r = r * (1.0 - 0.35 * fade) + 0.15 * fade;
        g = g * (1.0 - 0.35 * fade) + 0.15 * fade;
        b = b * (1.0 - 0.35 * fade) + 0.15 * fade;
    }
    [r, g, b]
}

fn luma([r, g, b]: [f32; 3]) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// 1 at mid grey, 0 at black and white.
fn midtones(level: f32) -> f32 {
    let level = level.clamp(0.0, 1.0);
    4.0 * level * (1.0 - level)
}

fn smoothstep(low: f32, high: f32, value: f32) -> f32 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Turn the colour about the grey axis by `angle` radians.
fn rotate_hue([r, g, b]: [f32; 3], angle: f32) -> [f32; 3] {
    let (sin, cos) = angle.sin_cos();
    let third = (1.0 - cos) / 3.0;
    let root = (1.0f32 / 3.0).sqrt() * sin;
    let (a, b2, c) = (cos + third, third - root, third + root);
    [
        r * a + g * b2 + b * c,
        r * c + g * a + b * b2,
        r * b2 + g * c + b * a,
    ]
}

/// A value from 0 to 1 for one pixel and seed.
fn hash(x: u32, y: u32, seed: u32) -> f32 {
    let mut h =
        x.wrapping_mul(0x8da6_b343) ^ y.wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    #[allow(clippy::cast_precision_loss)]
    let unit = (h & 0x00ff_ffff) as f32 / 16_777_215.0;
    unit
}

/// A box blur `radius` pixels wide each way, edges held.
fn box_blur(colour: &[[f32; 3]], width: usize, height: usize, radius: usize) -> Vec<[f32; 3]> {
    let pass = |input: &[[f32; 3]], horizontal: bool| -> Vec<[f32; 3]> {
        let mut output = vec![[0.0; 3]; input.len()];
        let (lines, length) = if horizontal {
            (height, width)
        } else {
            (width, height)
        };
        let at = |line: usize, step: usize| {
            if horizontal {
                line * width + step
            } else {
                step * width + line
            }
        };
        #[allow(clippy::cast_precision_loss)]
        let count = (2 * radius + 1) as f32;
        for line in 0..lines {
            for step in 0..length {
                let mut sum = [0.0f32; 3];
                for offset in 0..=2 * radius {
                    let near = (step + offset).saturating_sub(radius).min(length - 1);
                    let pixel = input[at(line, near)];
                    for channel in 0..3 {
                        sum[channel] += pixel[channel];
                    }
                }
                output[at(line, step)] = sum.map(|value| value / count);
            }
        }
        output
    };
    pass(&pass(colour, true), false)
}

/// The sliders auto adjust sets; it leaves the others as they are.
pub const AUTO_KEYS: [AdjustKey; 5] = [
    AdjustKey::Exposure,
    AdjustKey::Contrast,
    AdjustKey::Temperature,
    AdjustKey::Tint,
    AdjustKey::Saturation,
];

/// What auto adjust and colour match read from a picture: brightness at
/// the darkest, middle and brightest, the mean colour, and how colourful
/// it is. Clear pixels do not count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PictureStats {
    pub low: f32,
    pub middle: f32,
    pub high: f32,
    pub mean: [f32; 3],
    pub chroma: f32,
}

impl PictureStats {
    /// The stats of premultiplied RGBA, or `None` when nothing is drawn.
    #[must_use]
    pub fn of(rgba: &[u8]) -> Option<Self> {
        let mut levels = [0_u32; 256];
        let mut sums = [0.0_f64; 3];
        let mut chroma = 0.0_f64;
        let mut count = 0_u32;
        for pixel in rgba.as_chunks::<4>().0 {
            if pixel[3] < 128 {
                continue;
            }
            let alpha = f32::from(pixel[3]) / 255.0;
            let colour = [0, 1, 2].map(|index| (f32::from(pixel[index]) / 255.0 / alpha).min(1.0));
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let bin = (luma(colour) * 255.0).round().clamp(0.0, 255.0) as usize;
            levels[bin] += 1;
            for (sum, value) in sums.iter_mut().zip(colour) {
                *sum += f64::from(value);
            }
            let high = colour[0].max(colour[1]).max(colour[2]);
            let low = colour[0].min(colour[1]).min(colour[2]);
            chroma += f64::from(high - low);
            count += 1;
        }
        if count == 0 {
            return None;
        }
        let at = |share: f64| {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let wanted = (f64::from(count) * share).ceil().max(1.0) as u32;
            let mut seen = 0;
            for (bin, &many) in levels.iter().enumerate() {
                seen += many;
                if seen >= wanted {
                    #[allow(clippy::cast_precision_loss)]
                    return bin as f32 / 255.0;
                }
            }
            1.0
        };
        #[allow(clippy::cast_possible_truncation)]
        let mean = sums.map(|sum| (sum / f64::from(count)) as f32);
        #[allow(clippy::cast_possible_truncation)]
        let chroma = (chroma / f64::from(count)) as f32;
        Some(Self {
            low: at(0.01),
            middle: at(0.5),
            high: at(0.99),
            mean,
            chroma,
        })
    }

    /// Warmth as temperature reads it: blue over red, -1 to 1.
    fn cool(&self) -> f32 {
        let [r, _, b] = self.mean;
        if r + b > 0.02 { (b - r) / (r + b) } else { 0.0 }
    }

    /// Green over the red and blue mean, as tint reads it.
    fn green(&self) -> f32 {
        let [r, g, b] = self.mean;
        if g > 0.02 {
            1.0 - (r + b) / 2.0 / g
        } else {
            0.0
        }
    }
}

/// Held to `reach`, rounded to hundredths, and zero when too small to see.
fn tidy(value: f32, reach: f32) -> f32 {
    let value = (value.clamp(-reach, reach) * 100.0).round() / 100.0;
    if value.abs() < 0.05 { 0.0 } else { value }
}

/// Slider values that even out a picture, premultiplied RGBA: its middle
/// brightness toward mid grey, a flat range stretched, a colour cast
/// toward neutral, and dull colour lifted a little. Each value is gentle
/// and rounded to hundredths; a picture that needs nothing gets zeros.
#[must_use]
pub fn auto_adjust(rgba: &[u8]) -> Adjust {
    let mut adjust = Adjust::default();
    let Some(stats) = PictureStats::of(rgba) else {
        return adjust;
    };
    // Exposure doubles the light at 0.5.
    let gain = 0.47 / stats.middle.max(0.02);
    adjust.set(AdjustKey::Exposure, tidy(gain.log2() / 2.0, 0.6));
    let range = (stats.high - stats.low).max(0.02);
    if range < 0.8 {
        adjust.set(AdjustKey::Contrast, tidy(0.9 / range - 1.0, 0.4));
    }
    // Temperature scales red up and blue down by a fifth at 1; tint
    // scales green down.
    adjust.set(AdjustKey::Temperature, tidy(stats.cool() / 0.2, 1.0));
    adjust.set(AdjustKey::Tint, tidy(stats.green() / 0.2, 0.5));
    if stats.chroma < 0.2 && stats.chroma > 0.02 {
        adjust.set(
            AdjustKey::Saturation,
            tidy((0.2 - stats.chroma) * 1.5, 0.25),
        );
    }
    adjust
}

/// Slider values that bring a picture's look toward `reference`'s: its
/// middle brightness, range, warmth, tint and colourfulness.
#[must_use]
pub fn match_adjust(rgba: &[u8], reference: &[u8]) -> Adjust {
    let mut adjust = Adjust::default();
    let (Some(own), Some(goal)) = (PictureStats::of(rgba), PictureStats::of(reference)) else {
        return adjust;
    };
    let gain = goal.middle.max(0.02) / own.middle.max(0.02);
    adjust.set(AdjustKey::Exposure, tidy(gain.log2() / 2.0, 1.0));
    let range = (goal.high - goal.low).max(0.02) / (own.high - own.low).max(0.02);
    adjust.set(AdjustKey::Contrast, tidy(range - 1.0, 0.6));
    adjust.set(
        AdjustKey::Temperature,
        tidy((own.cool() - goal.cool()) / 0.2, 1.0),
    );
    adjust.set(
        AdjustKey::Tint,
        tidy((own.green() - goal.green()) / 0.2, 1.0),
    );
    if own.chroma > 0.02 {
        adjust.set(
            AdjustKey::Saturation,
            tidy(goal.chroma / own.chroma - 1.0, 1.0),
        );
    }
    adjust
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, MaterialEffect, MaterialId, Segment, TimeRange};

    use super::{
        AUTO_KEYS, Adjust, AdjustKey, adjust_rgba, auto_adjust, match_adjust, segment_adjust,
    };

    fn grey(level: u8, alpha: u8) -> Vec<u8> {
        let mut bytes = Vec::new();
        for _ in 0..16 {
            #[allow(clippy::cast_possible_truncation)]
            let premultiplied = (u32::from(level) * u32::from(alpha) / 255) as u8;
            bytes.extend([premultiplied, premultiplied, premultiplied, alpha]);
        }
        bytes
    }

    fn run(key: AdjustKey, value: f32, bytes: &mut [u8]) {
        let mut adjust = Adjust::default();
        adjust.set(key, value);
        adjust_rgba(&adjust, bytes, 4, 4, 0);
    }

    #[test]
    fn zero_leaves_the_picture_alone() {
        let mut bytes = grey(100, 200);
        let before = bytes.clone();
        adjust_rgba(&Adjust::default(), &mut bytes, 4, 4, 0);
        assert_eq!(bytes, before);
    }

    #[test]
    fn each_slider_moves_the_picture_the_way_its_name_says() {
        let mut bright = grey(100, 255);
        run(AdjustKey::Exposure, 0.5, &mut bright);
        assert_eq!(bright[0], 200);
        let mut dark = grey(100, 255);
        run(AdjustKey::Brightness, -0.4, &mut dark);
        assert!(dark[0] < 100);
        let mut flat = grey(40, 255);
        run(AdjustKey::Contrast, -1.0, &mut flat);
        assert_eq!(flat[0], 128);
        let mut warm = grey(100, 255);
        run(AdjustKey::Temperature, 1.0, &mut warm);
        assert!(warm[0] > 100 && warm[2] < 100 && warm[1] == 100);
        let mut faded = grey(0, 255);
        run(AdjustKey::Fade, 1.0, &mut faded);
        assert_eq!(faded[0], 38);
    }

    #[test]
    fn saturation_and_hue_leave_grey_and_turn_colour() {
        let mut bytes = grey(90, 255);
        run(AdjustKey::Saturation, 1.0, &mut bytes);
        run(AdjustKey::Hue, 0.5, &mut bytes);
        assert_eq!(&bytes[..3], &[90, 90, 90]);
        let mut red = [200, 0, 0, 255].repeat(16);
        run(AdjustKey::Saturation, -1.0, &mut red);
        assert_eq!(red[0], red[1]);
        let mut red = [200, 0, 0, 255].repeat(16);
        run(AdjustKey::Hue, 2.0 / 3.0, &mut red);
        assert!(red[1] > 150 && red[0] < 20, "{red:?}");
    }

    #[test]
    fn alpha_stays_and_colour_stays_premultiplied() {
        let mut bytes = grey(100, 128);
        run(AdjustKey::Exposure, 1.0, &mut bytes);
        assert_eq!(bytes[3], 128);
        assert_eq!(bytes[0], 128);
        let mut clear = grey(0, 0);
        run(AdjustKey::Fade, 1.0, &mut clear);
        assert!(clear.iter().all(|byte| *byte == 0));
    }

    #[test]
    fn vignette_darkens_corners_and_grain_follows_its_seed() {
        let mut bytes = [128, 128, 128, 255].repeat(64);
        let mut adjust = Adjust::default();
        adjust.set(AdjustKey::Vignette, 1.0);
        adjust_rgba(&adjust, &mut bytes, 8, 8, 0);
        let centre = bytes[(3 * 8 + 3) * 4];
        assert!(bytes[0] < centre, "{} {centre}", bytes[0]);
        let mut adjust = Adjust::default();
        adjust.set(AdjustKey::Grain, 1.0);
        let mut first = [128, 128, 128, 255].repeat(64);
        let mut again = first.clone();
        let mut other = first.clone();
        adjust_rgba(&adjust, &mut first, 8, 8, 1);
        adjust_rgba(&adjust, &mut again, 8, 8, 1);
        adjust_rgba(&adjust, &mut other, 8, 8, 2);
        assert_eq!(first, again);
        assert_ne!(first, other);
    }

    #[test]
    fn sharpen_and_clarity_raise_an_edge() {
        let mut edge = Vec::new();
        for _ in 0..8 {
            for x in 0..8 {
                let level = if x < 4 { 80 } else { 160 };
                edge.extend([level, level, level, 255]);
            }
        }
        let mut sharp = edge.clone();
        let mut adjust = Adjust::default();
        adjust.set(AdjustKey::Sharpen, 1.0);
        adjust_rgba(&adjust, &mut sharp, 8, 8, 0);
        assert!(sharp[4 * 4] > 160 && sharp[3 * 4] < 80);
        let mut clear = edge;
        let mut adjust = Adjust::default();
        adjust.set(AdjustKey::Clarity, 1.0);
        adjust_rgba(&adjust, &mut clear, 8, 8, 0);
        assert!(clear[4 * 4] > 160);
    }

    #[test]
    fn values_hold_their_range_and_read_from_the_draft() {
        let mut adjust = Adjust::default();
        adjust.set(AdjustKey::Grain, -1.0);
        adjust.set(AdjustKey::Contrast, 9.0);
        adjust.set(AdjustKey::Hue, f32::NAN);
        assert_eq!(adjust.get(AdjustKey::Grain), 0.0);
        assert_eq!(adjust.get(AdjustKey::Contrast), 1.0);
        assert!(!adjust.is_identity());
        for key in AdjustKey::ALL {
            assert_eq!(AdjustKey::from_kind(key.kind()), Some(key));
        }

        let mut draft = Draft::empty("D");
        draft.materials.effects.push(MaterialEffect {
            kind: "temperature".to_owned(),
            value: Some(0.4),
            ..MaterialEffect::new(MaterialId::new("warm"))
        });
        let mut segment = Segment::new("a", "m", TimeRange::new(0, 1));
        segment.extra_material_refs.push(MaterialId::new("warm"));
        let read = segment_adjust(&draft, &segment);
        assert!((read.get(AdjustKey::Temperature) - 0.4).abs() < 1e-6);
        assert_eq!(read.get(AdjustKey::Exposure), 0.0);
    }

    fn mean(rgba: &[u8]) -> [f32; 3] {
        let pixels = rgba.as_chunks::<4>().0;
        #[allow(clippy::cast_precision_loss)]
        let many = pixels.len() as f32;
        [0, 1, 2].map(|index| pixels.iter().map(|p| f32::from(p[index])).sum::<f32>() / many)
    }

    #[test]
    fn auto_adjust_evens_out_a_dark_picture_and_a_blue_cast() {
        // A full ramp of greys needs nothing.
        let ramp: Vec<u8> = (0..=255_u8).flat_map(|v| [v, v, v, 255]).collect();
        let none = auto_adjust(&ramp);
        for key in AUTO_KEYS {
            assert_eq!(none.get(key), 0.0, "{key:?}");
        }
        assert!(auto_adjust(&[0, 0, 0, 0].repeat(4)).is_identity());

        // A dark, flat picture is brightened and stretched.
        let dark: Vec<u8> = (20..60_u8).flat_map(|v| [v, v, v, 255]).collect();
        let lift = auto_adjust(&dark);
        assert!(lift.get(AdjustKey::Exposure) > 0.3, "{lift:?}");
        assert!(lift.get(AdjustKey::Contrast) > 0.0, "{lift:?}");
        let mut fixed = dark.clone();
        adjust_rgba(&lift, &mut fixed, 40, 1, 0);
        assert!(
            mean(&fixed)[0] > mean(&dark)[0] + 30.0,
            "{:?}",
            mean(&fixed)
        );

        // A blue cast is warmed toward grey.
        let blue: Vec<u8> = (0..64_u8)
            .flat_map(|v| [80 + v, 100 + v, 150 + v, 255])
            .collect();
        let warm = auto_adjust(&blue);
        assert!(warm.get(AdjustKey::Temperature) > 0.2, "{warm:?}");
        let mut fixed = blue.clone();
        adjust_rgba(&warm, &mut fixed, 64, 1, 0);
        let [r, _, b] = mean(&fixed);
        let [r0, _, b0] = mean(&blue);
        assert!((b - r).abs() < (b0 - r0).abs() / 2.0, "{r} {b}");
    }

    #[test]
    fn colour_match_moves_a_picture_toward_the_reference() {
        let cool: Vec<u8> = (0..64_u8)
            .flat_map(|v| [60 + v, 80 + v, 140 + v, 255])
            .collect();
        let warm: Vec<u8> = (0..64_u8)
            .flat_map(|v| [150 + v, 110 + v, 70 + v, 255])
            .collect();
        let toward = match_adjust(&cool, &warm);
        assert!(toward.get(AdjustKey::Temperature) > 0.5, "{toward:?}");
        let mut fixed = cool.clone();
        adjust_rgba(&toward, &mut fixed, 64, 1, 0);
        let [r, _, b] = mean(&fixed);
        let [r0, _, b0] = mean(&cool);
        assert!(r - b > r0 - b0 + 40.0, "{r} {b}");
        // A picture matched to itself needs nothing.
        assert!(match_adjust(&warm, &warm).is_identity());
        assert!(match_adjust(&warm, &[0, 0, 0, 0]).is_identity());
    }
}

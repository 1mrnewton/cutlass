//! Blend modes: how a clip's picture mixes with the picture below it.
//!
//! Pixels are premultiplied RGBA. Every mode but source over follows the
//! separable compositing formula:
//! `co = cs·(1 − αb) + cb·(1 − αs) + αs·αb·B(Cb, Cs)` and
//! `αo = αs + αb·(1 − αs)`, where `cs` and `cb` are premultiplied and `Cs`
//! and `Cb` are the straight colours. Source over keeps the integer
//! formula the compositor uses, so a normal clip reads the same either way.
//!
//! [`BLEND_WGSL`] is the same arithmetic for a GPU pass; the two are kept
//! side by side so they change together.

use crate::catalog::BlendMode;

impl BlendMode {
    /// Every mode, in the order a picker lists them.
    pub const ALL: [Self; 13] = [
        Self::SourceOver,
        Self::Darken,
        Self::Multiply,
        Self::ColorBurn,
        Self::Lighten,
        Self::Screen,
        Self::ColorDodge,
        Self::Add,
        Self::Overlay,
        Self::SoftLight,
        Self::HardLight,
        Self::Difference,
        Self::Exclusion,
    ];

    /// The name a person reads.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SourceOver => "Normal",
            Self::Darken => "Darken",
            Self::Multiply => "Multiply",
            Self::ColorBurn => "Color burn",
            Self::Lighten => "Lighten",
            Self::Screen => "Screen",
            Self::ColorDodge => "Color dodge",
            Self::Add => "Add",
            Self::Overlay => "Overlay",
            Self::SoftLight => "Soft light",
            Self::HardLight => "Hard light",
            Self::Difference => "Difference",
            Self::Exclusion => "Exclusion",
        }
    }

    /// The catalog package id that stores this mode on a clip.
    #[must_use]
    pub const fn package_id(self) -> &'static str {
        match self {
            Self::SourceOver => crate::catalog::SOURCE_OVER,
            Self::Darken => crate::catalog::DARKEN,
            Self::Multiply => crate::catalog::MULTIPLY,
            Self::ColorBurn => crate::catalog::COLOR_BURN,
            Self::Lighten => crate::catalog::LIGHTEN,
            Self::Screen => crate::catalog::SCREEN,
            Self::ColorDodge => crate::catalog::COLOR_DODGE,
            Self::Add => crate::catalog::ADD,
            Self::Overlay => crate::catalog::OVERLAY,
            Self::SoftLight => crate::catalog::SOFT_LIGHT,
            Self::HardLight => crate::catalog::HARD_LIGHT,
            Self::Difference => crate::catalog::DIFFERENCE,
            Self::Exclusion => crate::catalog::EXCLUSION,
        }
    }

    /// The mode a package id stores, if it is a blend.
    #[must_use]
    pub fn from_package(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|mode| mode.package_id() == id)
    }

    /// The number [`BLEND_WGSL`]'s `blend` takes for this mode.
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            Self::SourceOver => 0,
            Self::Darken => 1,
            Self::Multiply => 2,
            Self::ColorBurn => 3,
            Self::Lighten => 4,
            Self::Screen => 5,
            Self::ColorDodge => 6,
            Self::Add => 7,
            Self::Overlay => 8,
            Self::SoftLight => 9,
            Self::HardLight => 10,
            Self::Difference => 11,
            Self::Exclusion => 12,
        }
    }
}

/// Mix one premultiplied `source` pixel over one premultiplied
/// `destination` pixel.
#[must_use]
pub fn blend_pixel(mode: BlendMode, source: [u8; 4], destination: [u8; 4]) -> [u8; 4] {
    if mode == BlendMode::SourceOver {
        let [sr, sg, sb, sa] = source;
        let [dr, dg, db, da] = destination;
        return [
            over_channel(sr, dr, sa),
            over_channel(sg, dg, sa),
            over_channel(sb, db, sa),
            over_channel(sa, da, sa),
        ];
    }
    let unit = |value: u8| f32::from(value) / 255.0;
    let source = source.map(unit);
    let destination = destination.map(unit);
    let mixed = blend_unit(mode, source, destination);
    mixed.map(|value| {
        let scaled = (value.clamp(0.0, 1.0) * 255.0).round();
        // In 0..=255 after the clamp.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let byte = scaled as u8;
        byte
    })
}

/// [`blend_pixel`] on premultiplied colours in `0..=1`.
#[must_use]
pub fn blend_unit(mode: BlendMode, source: [f32; 4], destination: [f32; 4]) -> [f32; 4] {
    let (alpha_s, alpha_b) = (source[3], destination[3]);
    let alpha = alpha_s + alpha_b * (1.0 - alpha_s);
    let mut out = [0.0, 0.0, 0.0, alpha];
    for channel in 0..3 {
        let (cs, cb) = (source[channel], destination[channel]);
        let straight_s = straight(cs, alpha_s);
        let straight_b = straight(cb, alpha_b);
        let mixed = channel_blend(mode, straight_b, straight_s);
        let value = cs * (1.0 - alpha_b) + cb * (1.0 - alpha_s) + alpha_s * alpha_b * mixed;
        out[channel] = value.clamp(0.0, alpha);
    }
    out
}

fn straight(color: f32, alpha: f32) -> f32 {
    if alpha <= 0.0 {
        0.0
    } else {
        (color / alpha).clamp(0.0, 1.0)
    }
}

/// `B(Cb, Cs)` for one straight channel.
fn channel_blend(mode: BlendMode, backdrop: f32, source: f32) -> f32 {
    match mode {
        BlendMode::SourceOver => source,
        BlendMode::Darken => backdrop.min(source),
        BlendMode::Multiply => backdrop * source,
        BlendMode::ColorBurn => {
            if backdrop >= 1.0 {
                1.0
            } else if source <= 0.0 {
                0.0
            } else {
                1.0 - ((1.0 - backdrop) / source).min(1.0)
            }
        }
        BlendMode::Lighten => backdrop.max(source),
        BlendMode::Screen => screen(backdrop, source),
        BlendMode::ColorDodge => {
            if backdrop <= 0.0 {
                0.0
            } else if source >= 1.0 {
                1.0
            } else {
                (backdrop / (1.0 - source)).min(1.0)
            }
        }
        BlendMode::Add => (backdrop + source).min(1.0),
        BlendMode::Overlay => hard_light(source, backdrop),
        BlendMode::SoftLight => soft_light(backdrop, source),
        BlendMode::HardLight => hard_light(backdrop, source),
        BlendMode::Difference => (backdrop - source).abs(),
        BlendMode::Exclusion => backdrop + source - 2.0 * backdrop * source,
    }
}

fn screen(backdrop: f32, source: f32) -> f32 {
    backdrop + source - backdrop * source
}

fn hard_light(backdrop: f32, source: f32) -> f32 {
    if source <= 0.5 {
        backdrop * 2.0 * source
    } else {
        screen(backdrop, 2.0 * source - 1.0)
    }
}

fn soft_light(backdrop: f32, source: f32) -> f32 {
    if source <= 0.5 {
        backdrop - (1.0 - 2.0 * source) * backdrop * (1.0 - backdrop)
    } else {
        let lifted = if backdrop <= 0.25 {
            ((16.0 * backdrop - 12.0) * backdrop + 4.0) * backdrop
        } else {
            backdrop.sqrt()
        };
        backdrop + (2.0 * source - 1.0) * (lifted - backdrop)
    }
}

/// `output = source + destination * (1 - source.alpha)`.
///
/// The 8-bit alpha is divided by 255 and the remainder is truncated.
/// A channel above 255 is clamped. Bytes are not premultiplied first.
fn over_channel(source: u8, destination: u8, source_alpha: u8) -> u8 {
    let inverse = u16::from(255_u8.saturating_sub(source_alpha));
    let mixed = u16::from(source) + u16::from(destination) * inverse / 255;
    u8::try_from(mixed.min(255)).unwrap_or(255)
}

/// `fn blend(mode: u32, source: vec4<f32>, backdrop: vec4<f32>) -> vec4<f32>`
/// for a WGSL shader, on premultiplied colours, as [`blend_unit`] does.
pub const BLEND_WGSL: &str = r"
fn blend_straight(color: f32, alpha: f32) -> f32 {
    if (alpha <= 0.0) {
        return 0.0;
    }
    return clamp(color / alpha, 0.0, 1.0);
}

fn blend_screen(b: f32, s: f32) -> f32 {
    return b + s - b * s;
}

fn blend_hard_light(b: f32, s: f32) -> f32 {
    if (s <= 0.5) {
        return b * 2.0 * s;
    }
    return blend_screen(b, 2.0 * s - 1.0);
}

fn blend_soft_light(b: f32, s: f32) -> f32 {
    if (s <= 0.5) {
        return b - (1.0 - 2.0 * s) * b * (1.0 - b);
    }
    var lifted = sqrt(b);
    if (b <= 0.25) {
        lifted = ((16.0 * b - 12.0) * b + 4.0) * b;
    }
    return b + (2.0 * s - 1.0) * (lifted - b);
}

fn blend_channel(mode: u32, b: f32, s: f32) -> f32 {
    switch mode {
        case 1u: { return min(b, s); }
        case 2u: { return b * s; }
        case 3u: {
            if (b >= 1.0) { return 1.0; }
            if (s <= 0.0) { return 0.0; }
            return 1.0 - min((1.0 - b) / s, 1.0);
        }
        case 4u: { return max(b, s); }
        case 5u: { return blend_screen(b, s); }
        case 6u: {
            if (b <= 0.0) { return 0.0; }
            if (s >= 1.0) { return 1.0; }
            return min(b / (1.0 - s), 1.0);
        }
        case 7u: { return min(b + s, 1.0); }
        case 8u: { return blend_hard_light(s, b); }
        case 9u: { return blend_soft_light(b, s); }
        case 10u: { return blend_hard_light(b, s); }
        case 11u: { return abs(b - s); }
        case 12u: { return b + s - 2.0 * b * s; }
        default: { return s; }
    }
}

fn blend(mode: u32, source: vec4<f32>, backdrop: vec4<f32>) -> vec4<f32> {
    let alpha_s = source.a;
    let alpha_b = backdrop.a;
    let alpha = alpha_s + alpha_b * (1.0 - alpha_s);
    if (mode == 0u) {
        return vec4<f32>(source.rgb + backdrop.rgb * (1.0 - alpha_s), alpha);
    }
    var out = vec4<f32>(0.0, 0.0, 0.0, alpha);
    for (var channel = 0; channel < 3; channel = channel + 1) {
        let cs = source[channel];
        let cb = backdrop[channel];
        let mixed = blend_channel(mode, blend_straight(cb, alpha_b), blend_straight(cs, alpha_s));
        let value = cs * (1.0 - alpha_b) + cb * (1.0 - alpha_s) + alpha_s * alpha_b * mixed;
        out[channel] = clamp(value, 0.0, alpha);
    }
    return out;
}
";

#[cfg(test)]
mod tests {
    use super::{BlendMode, blend_pixel};

    const GREY: [u8; 4] = [128, 128, 128, 255];

    fn on_grey(mode: BlendMode, source: [u8; 4]) -> [u8; 4] {
        blend_pixel(mode, source, GREY)
    }

    #[test]
    fn every_mode_has_a_package_a_label_and_its_own_code() {
        let mut codes = Vec::new();
        for mode in BlendMode::ALL {
            assert_eq!(BlendMode::from_package(mode.package_id()), Some(mode));
            assert!(!mode.label().is_empty());
            assert!(!codes.contains(&mode.code()));
            codes.push(mode.code());
        }
        assert_eq!(BlendMode::from_package("distort-invert"), None);
    }

    #[test]
    fn opaque_modes_follow_their_formulas() {
        let white = [255, 255, 255, 255];
        let black = [0, 0, 0, 255];
        assert_eq!(on_grey(BlendMode::Multiply, white), GREY);
        assert_eq!(on_grey(BlendMode::Multiply, black), black);
        assert_eq!(on_grey(BlendMode::Screen, black), GREY);
        assert_eq!(on_grey(BlendMode::Screen, white), white);
        assert_eq!(on_grey(BlendMode::Darken, white), GREY);
        assert_eq!(on_grey(BlendMode::Lighten, black), GREY);
        assert_eq!(on_grey(BlendMode::Difference, white), [127, 127, 127, 255]);
        assert_eq!(on_grey(BlendMode::Exclusion, black), GREY);
        assert_eq!(on_grey(BlendMode::Add, GREY), [255, 255, 255, 255]);
        assert_eq!(on_grey(BlendMode::ColorDodge, black), GREY);
        assert_eq!(on_grey(BlendMode::ColorBurn, white), GREY);
        // A mid grey leaves overlay, hard light and soft light as they were.
        assert_eq!(on_grey(BlendMode::HardLight, GREY), [128, 128, 128, 255]);
        assert_eq!(on_grey(BlendMode::SoftLight, GREY), [128, 128, 128, 255]);
        // The grey is a hair over half, so overlay screens a little.
        assert_eq!(on_grey(BlendMode::Overlay, black), [1, 1, 1, 255]);
        assert_eq!(on_grey(BlendMode::Overlay, white), white);
        assert_eq!(on_grey(BlendMode::HardLight, black), black);
    }

    #[test]
    fn a_clear_source_leaves_the_backdrop_and_a_clear_backdrop_shows_the_source() {
        let clear = [0, 0, 0, 0];
        let half_red = [128, 0, 0, 128];
        for mode in BlendMode::ALL {
            assert_eq!(blend_pixel(mode, clear, GREY), GREY, "{mode:?}");
            assert_eq!(blend_pixel(mode, half_red, clear), half_red, "{mode:?}");
        }
    }

    #[test]
    fn a_half_clear_source_mixes_half_way() {
        // Multiply by black at half alpha darkens a grey by half.
        let mixed = blend_pixel(BlendMode::Multiply, [0, 0, 0, 128], GREY);
        assert_eq!(mixed, [64, 64, 64, 255]);
    }
}

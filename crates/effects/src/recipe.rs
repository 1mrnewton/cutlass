//! Effect recipes: an effect as data, built from blocks the app ships.
//!
//! A recipe is JSON. It names up to [`MAX_PASSES`] passes; each pass is one
//! [`Block`] and values for that block's parameters. A value is a number or
//! a curve over the effect's own time: keyframes, a wave, or a held random
//! step. The GPU runs the blocks; a recipe never carries shader code, so a
//! recipe from anywhere can only ask for what the blocks already do.
//!
//! Every value a curve can reach is checked against the parameter's range
//! when the recipe is read, so [`Recipe::steps_at`] has nothing to refuse.
//! Positions and sizes are fractions of the frame, so an effect looks the
//! same on a phone preview and in a 4K export.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::OnceLock;

use cutlass_model::Micros;
use serde::{Deserialize, Serialize};

/// The recipe format this app reads. A recipe with a higher version needs
/// blocks or fields this app does not have. Version 2 added eased keys.
pub const RECIPE_VERSION: u32 = 2;
/// The first version whose keys may name an [`Ease`].
pub const EASE_VERSION: u32 = 2;
/// The most passes one recipe may run.
pub const MAX_PASSES: usize = 8;
/// The most keyframes one curve may hold.
pub const MAX_KEYS: usize = 64;
/// The largest recipe file read.
pub const MAX_RECIPE_BYTES: usize = 64 * 1024;
/// The most values one block takes.
pub const MAX_VALUES: usize = 12;
/// The fastest a held random value may change, in steps per second.
pub const MAX_JITTER_RATE: f32 = 120.0;
/// The effect id a recipe is stored under on a draft's effect material.
pub const RECIPE_PREFIX: &str = "recipe/";
/// The most controls one recipe may show.
pub const MAX_CONTROLS: usize = 6;

/// One building block the GPU runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Block {
    /// Scale, move and turn the frame about its centre.
    Transform,
    /// Pull red one way and blue the other.
    RgbSplit,
    /// Shift horizontal bands sideways by a random amount each.
    Slices,
    /// A box blur, as wide as a fraction of the shorter side.
    Blur,
    /// Brightness, contrast and saturation.
    Grade,
    /// Mix toward a solid colour.
    Flash,
    /// Darken toward the corners.
    Vignette,
    /// Add grain.
    Noise,
    /// Darken every other line.
    Scanlines,
    /// Reflect one half onto the other.
    Mirror,
    /// Square blocks of one colour.
    Pixelate,
    /// Turn the colours inside out.
    Invert,
    /// Smear toward a point, as if zooming during the exposure.
    ZoomBlur,
    /// Smear along one direction, as if the camera moved.
    MotionBlur,
    /// Twist the picture around a point, most at the middle.
    Swirl,
    /// Waves running across, down or out from the middle.
    Ripple,
    /// Swell the picture out from a point, or pinch it in.
    Bulge,
    /// Let the bright parts bleed light around them.
    Glow,
    /// A soft patch of coloured light laid over the picture.
    LightLeak,
    /// Turn every colour around the colour wheel.
    Hue,
    /// Repeat the whole picture in a grid.
    Tile,
    /// Fewer shades of each colour, in flat bands.
    Posterize,
    /// Draw the outlines: darken them, or show them alone.
    Edges,
    /// Colour each pixel by its brightness from a palette: two colours of
    /// its own, heat, or a rainbow.
    GradientMap,
    /// Repeat the picture in a grid, each copy cropped to fill its cell.
    Grid,
    /// Darken all but a soft circle of light.
    Spotlight,
    /// Show the picture only on one side of a moving edge: a wipe, an iris
    /// or blinds. The rest is black.
    Reveal,
    /// Scale across and down apart, optionally black past the picture.
    Stretch,
    /// Specks of dust and scratches, as on old film.
    Dust,
    /// Repeat one wedge around a point, as a kaleidoscope does.
    Kaleido,
    /// Pull red out and blue in from a point, as a cheap lens does.
    Fringe,
    /// Print the picture as dots, big where it is dark.
    Halftone,
    /// Lay the picture out as a stack of tilted photo prints with white
    /// borders and soft shadows, over a soft, pale copy of itself.
    Prints,
    /// Knock random blocks of the frame out of place, some with their
    /// colours swapped, as a broken video stream does.
    Blocks,
    /// Streaks of light from the bright parts out from a point.
    Rays,
    /// Blur by place: all but a sharp band, all but a sharp middle, or
    /// spun around the middle.
    Focus,
    /// Scale, move, stretch, turn and fade the picture as a card: clear
    /// where it no longer reaches, so what is behind it shows. A
    /// transition slides or flips one picture off the other with it. The
    /// card can be a white-edged photo print with a shadow, over a soft,
    /// dim copy of the picture.
    Place,
}

/// One parameter of a block: its name, its value when a recipe leaves it
/// out, and the range a value must stay in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Param {
    pub name: &'static str,
    pub default: f32,
    pub min: f32,
    pub max: f32,
}

const fn param(name: &'static str, default: f32, min: f32, max: f32) -> Param {
    Param {
        name,
        default,
        min,
        max,
    }
}

const TRANSFORM: &[Param] = &[
    param("scale", 1.0, 0.1, 4.0),
    param("x", 0.0, -1.0, 1.0),
    param("y", 0.0, -1.0, 1.0),
    param("rotate", 0.0, -180.0, 180.0),
];
const RGB_SPLIT: &[Param] = &[param("x", 0.0, -0.2, 0.2), param("y", 0.0, -0.2, 0.2)];
const SLICES: &[Param] = &[
    param("count", 8.0, 1.0, 64.0),
    param("amount", 0.0, 0.0, 0.5),
    param("seed", 0.0, 0.0, 1000.0),
];
const BLUR: &[Param] = &[param("radius", 0.0, 0.0, 0.1)];
const GRADE: &[Param] = &[
    param("brightness", 0.0, -1.0, 1.0),
    param("contrast", 1.0, 0.0, 3.0),
    param("saturation", 1.0, 0.0, 3.0),
];
const FLASH: &[Param] = &[
    param("amount", 0.0, 0.0, 1.0),
    param("red", 1.0, 0.0, 1.0),
    param("green", 1.0, 0.0, 1.0),
    param("blue", 1.0, 0.0, 1.0),
];
const VIGNETTE: &[Param] = &[param("amount", 0.0, 0.0, 1.0), param("size", 0.6, 0.0, 1.5)];
const NOISE: &[Param] = &[
    param("amount", 0.0, 0.0, 1.0),
    param("seed", 0.0, 0.0, 1000.0),
];
const SCANLINES: &[Param] = &[
    param("amount", 0.0, 0.0, 1.0),
    param("count", 120.0, 2.0, 1000.0),
];
/// Side 0 keeps the left half, 1 the right, 2 the top, 3 the bottom.
const MIRROR: &[Param] = &[param("side", 0.0, 0.0, 3.0)];
const PIXELATE: &[Param] = &[param("size", 0.0, 0.0, 0.25)];
const INVERT: &[Param] = &[param("amount", 1.0, 0.0, 1.0)];
/// `amount` is how far toward the point a pixel is smeared; `x` and `y`
/// move the point from the middle.
const ZOOM_BLUR: &[Param] = &[
    param("amount", 0.0, 0.0, 0.5),
    param("x", 0.0, -0.5, 0.5),
    param("y", 0.0, -0.5, 0.5),
];
/// `length` is a fraction of the shorter side; `angle` 0 is across.
const MOTION_BLUR: &[Param] = &[
    param("length", 0.0, 0.0, 0.2),
    param("angle", 0.0, -180.0, 180.0),
];
/// `radius` is a fraction of the shorter side.
const SWIRL: &[Param] = &[
    param("angle", 0.0, -720.0, 720.0),
    param("radius", 0.5, 0.0, 1.5),
    param("x", 0.0, -0.5, 0.5),
    param("y", 0.0, -0.5, 0.5),
];
/// `amount` and `length` are fractions of the shorter side. A `phase`
/// going 0 to 1 moves the waves on by one wave. `kind` 0 shifts sideways
/// in bands down the frame, 1 shifts up and down in bands across it, 2
/// runs rings out from the middle.
const RIPPLE: &[Param] = &[
    param("amount", 0.0, 0.0, 0.1),
    param("length", 0.1, 0.01, 1.0),
    param("phase", 0.0, -100.0, 100.0),
    param("kind", 0.0, 0.0, 2.0),
];
/// A positive `amount` swells, a negative one pinches; `radius` is a
/// fraction of the shorter side.
const BULGE: &[Param] = &[
    param("amount", 0.0, -1.0, 0.9),
    param("radius", 0.5, 0.01, 1.5),
    param("x", 0.0, -0.5, 0.5),
    param("y", 0.0, -0.5, 0.5),
];
/// Light above `threshold` spreads as far as `radius`, a fraction of the
/// shorter side.
const GLOW: &[Param] = &[
    param("amount", 0.0, 0.0, 3.0),
    param("radius", 0.03, 0.0, 0.2),
    param("threshold", 0.6, 0.0, 1.0),
];
/// `x` and `y` place the light from the middle, in frames; `size` is a
/// fraction of the shorter side.
const LIGHT_LEAK: &[Param] = &[
    param("amount", 0.0, 0.0, 1.0),
    param("x", 0.0, -1.5, 1.5),
    param("y", 0.0, -1.5, 1.5),
    param("size", 0.5, 0.01, 3.0),
    param("red", 1.0, 0.0, 1.0),
    param("green", 0.5, 0.0, 1.0),
    param("blue", 0.2, 0.0, 1.0),
];
const HUE: &[Param] = &[param("degrees", 0.0, -180.0, 180.0)];
/// Whole copies across and down.
const TILE: &[Param] = &[param("x", 1.0, 1.0, 8.0), param("y", 1.0, 1.0, 8.0)];
const POSTERIZE: &[Param] = &[param("levels", 256.0, 2.0, 256.0)];
/// `amount` darkens the outlines over the picture; `only` fades the
/// picture away to leave light outlines on black.
const EDGES: &[Param] = &[param("amount", 0.0, 0.0, 1.0), param("only", 0.0, 0.0, 1.0)];
/// `palette` 0 runs from the low colour to the high one, 1 is heat from
/// black through blue, magenta and orange to white, 2 is a rainbow.
const GRADIENT_MAP: &[Param] = &[
    param("amount", 0.0, 0.0, 1.0),
    param("palette", 0.0, 0.0, 2.0),
    param("low_red", 0.0, 0.0, 1.0),
    param("low_green", 0.0, 0.0, 1.0),
    param("low_blue", 0.0, 0.0, 1.0),
    param("high_red", 1.0, 0.0, 1.0),
    param("high_green", 1.0, 0.0, 1.0),
    param("high_blue", 1.0, 0.0, 1.0),
];
/// Whole cells across and down; `zoom` above 1 crops each copy closer.
const GRID: &[Param] = &[
    param("x", 1.0, 1.0, 8.0),
    param("y", 1.0, 1.0, 8.0),
    param("zoom", 1.0, 1.0, 4.0),
];
/// `x` and `y` move the light from the middle, in frames; `size` and
/// `soft` are fractions of the shorter side.
const SPOTLIGHT: &[Param] = &[
    param("amount", 0.0, 0.0, 1.0),
    param("x", 0.0, -1.0, 1.0),
    param("y", 0.0, -1.0, 1.0),
    param("size", 0.3, 0.0, 2.0),
    param("soft", 0.15, 0.0, 1.0),
];
/// `progress` 0 shows nothing and 1 all of it. `kind` 0 wipes from one
/// side, 1 opens a circle from the middle, 2 opens `count` blinds, 3 opens
/// two doors from the middle line, 4 sweeps round like a clock hand from
/// `angle`, 5 opens a diamond, 6 fills a checkerboard of `count` squares
/// across, one colour then the other, and 7 dissolves in fine grains.
/// `angle` 0 runs left to right; `soft` is the edge, as a fraction of the
/// run.
const REVEAL: &[Param] = &[
    param("progress", 1.0, 0.0, 1.0),
    param("kind", 0.0, 0.0, 7.0),
    param("angle", 0.0, -180.0, 180.0),
    param("soft", 0.02, 0.0, 0.5),
    param("count", 8.0, 1.0, 64.0),
];
/// `black` 1 leaves black where the shrunk picture no longer reaches;
/// 0 reflects it, as the other moving blocks do.
const STRETCH: &[Param] = &[
    param("x", 1.0, 0.0, 4.0),
    param("y", 1.0, 0.0, 4.0),
    param("black", 0.0, 0.0, 1.0),
];
/// `amount` is how much of the frame is specked, `scratches` how likely
/// each of three scratches is, `size` a speck's room as a fraction of the
/// shorter side.
const DUST: &[Param] = &[
    param("amount", 0.0, 0.0, 1.0),
    param("scratches", 0.0, 0.0, 1.0),
    param("seed", 0.0, 0.0, 1000.0),
    param("size", 0.03, 0.005, 0.2),
];
/// `segments` below 2 leaves the picture as it is.
const KALEIDO: &[Param] = &[
    param("segments", 1.0, 1.0, 32.0),
    param("angle", 0.0, -360.0, 360.0),
    param("x", 0.0, -0.5, 0.5),
    param("y", 0.0, -0.5, 0.5),
];
const FRINGE: &[Param] = &[
    param("amount", 0.0, 0.0, 0.2),
    param("x", 0.0, -0.5, 0.5),
    param("y", 0.0, -0.5, 0.5),
];
/// `size` is a dot's cell, as a fraction of the shorter side; `color` 0
/// prints black dots and 1 dots of the picture's own colour, on white.
/// `count` prints from 0 (none) to 4, the back one the largest. `scale`
/// is the back print's size against the frame, `spread` how far the prints
/// sit from the middle, `turn` turns the whole stack and `tilt` turns each
/// print its own way, in degrees. `border`, the white edge, and `blur`, the
/// copy behind, are fractions of the shorter side; `wash` pales that copy.
/// `arrive` is the clock the prints fly in by, the back one first: each
/// takes one unit to land and the next sets off 0.6 after it, so a print
/// is still settling when the next comes. At 8 every print has landed.
const PRINTS: &[Param] = &[
    param("count", 0.0, 0.0, 4.0),
    param("scale", 0.7, 0.1, 1.5),
    param("spread", 0.5, 0.0, 1.0),
    param("turn", 0.0, -180.0, 180.0),
    param("tilt", 8.0, 0.0, 45.0),
    param("border", 0.025, 0.0, 0.1),
    param("blur", 0.03, 0.0, 0.1),
    param("wash", 0.5, 0.0, 1.0),
    param("arrive", 8.0, 0.0, 8.0),
];
/// `amount` is the share of blocks knocked out, `size` a block's height as
/// a fraction of the shorter side (blocks are wider than tall), `shift`
/// how far one moves, in frames, and `color` the share of those whose
/// colours swap.
const BLOCKS: &[Param] = &[
    param("amount", 0.0, 0.0, 1.0),
    param("size", 0.06, 0.01, 0.5),
    param("shift", 0.05, 0.0, 0.3),
    param("seed", 0.0, 0.0, 1000.0),
    param("color", 0.3, 0.0, 1.0),
];
/// `x` and `y` move the point from the middle, in frames; `length` is how
/// far toward it a streak reaches, `threshold` the brightness that shines.
const RAYS: &[Param] = &[
    param("amount", 0.0, 0.0, 3.0),
    param("x", 0.0, -1.0, 1.0),
    param("y", 0.0, -1.0, 1.0),
    param("length", 0.4, 0.0, 1.0),
    param("threshold", 0.6, 0.0, 1.0),
];
/// `amount` is the blur at its widest, a fraction of the shorter side.
/// `kind` 0 keeps a band across sharp, as a tilt-shift lens does, 1 keeps
/// the middle sharp, 2 spins the picture around the middle by up to
/// `angle` degrees. `size` is the sharp part, `y` moves the band.
const FOCUS: &[Param] = &[
    param("amount", 0.0, 0.0, 0.1),
    param("kind", 0.0, 0.0, 2.0),
    param("size", 0.3, 0.0, 1.0),
    param("y", 0.0, -0.5, 0.5),
    param("angle", 10.0, 0.0, 90.0),
];
/// `x` and `y` move the picture right and down, in frames; `stretch_x`
/// and `stretch_y` scale it across and down on top of `scale`.
///
/// `border` puts the picture on a white photo print, its edge that wide,
/// and `foot` makes the print's bottom edge that much deeper; both are
/// fractions of the shorter side and shrink with `scale`. `shadow` is how
/// soft the print's shadow is, none at 0. `back` fills the rest of the
/// frame with a copy of the picture blurred that far, as a fraction of the
/// shorter side, and darkened by `dim`; at 0 the rest is clear.
const PLACE: &[Param] = &[
    param("scale", 1.0, 0.0, 4.0),
    param("x", 0.0, -2.0, 2.0),
    param("y", 0.0, -2.0, 2.0),
    param("rotate", 0.0, -360.0, 360.0),
    param("stretch_x", 1.0, 0.0, 4.0),
    param("stretch_y", 1.0, 0.0, 4.0),
    param("opacity", 1.0, 0.0, 1.0),
    param("border", 0.0, 0.0, 0.2),
    param("foot", 0.0, 0.0, 0.3),
    param("back", 0.0, 0.0, 0.1),
    param("dim", 0.0, 0.0, 1.0),
    param("shadow", 0.0, 0.0, 0.1),
];
const HALFTONE: &[Param] = &[
    param("size", 0.0, 0.0, 0.1),
    param("amount", 1.0, 0.0, 1.0),
    param("color", 0.0, 0.0, 1.0),
];

impl Block {
    /// Every block, in the order of their codes.
    pub const ALL: [Self; 37] = [
        Self::Transform,
        Self::RgbSplit,
        Self::Slices,
        Self::Blur,
        Self::Grade,
        Self::Flash,
        Self::Vignette,
        Self::Noise,
        Self::Scanlines,
        Self::Mirror,
        Self::Pixelate,
        Self::Invert,
        Self::ZoomBlur,
        Self::MotionBlur,
        Self::Swirl,
        Self::Ripple,
        Self::Bulge,
        Self::Glow,
        Self::LightLeak,
        Self::Hue,
        Self::Tile,
        Self::Posterize,
        Self::Edges,
        Self::GradientMap,
        Self::Grid,
        Self::Spotlight,
        Self::Reveal,
        Self::Stretch,
        Self::Dust,
        Self::Kaleido,
        Self::Fringe,
        Self::Halftone,
        Self::Prints,
        Self::Blocks,
        Self::Rays,
        Self::Focus,
        Self::Place,
    ];

    /// The parameters in the order the shader reads them.
    #[must_use]
    pub const fn params(self) -> &'static [Param] {
        match self {
            Self::Transform => TRANSFORM,
            Self::RgbSplit => RGB_SPLIT,
            Self::Slices => SLICES,
            Self::Blur => BLUR,
            Self::Grade => GRADE,
            Self::Flash => FLASH,
            Self::Vignette => VIGNETTE,
            Self::Noise => NOISE,
            Self::Scanlines => SCANLINES,
            Self::Mirror => MIRROR,
            Self::Pixelate => PIXELATE,
            Self::Invert => INVERT,
            Self::ZoomBlur => ZOOM_BLUR,
            Self::MotionBlur => MOTION_BLUR,
            Self::Swirl => SWIRL,
            Self::Ripple => RIPPLE,
            Self::Bulge => BULGE,
            Self::Glow => GLOW,
            Self::LightLeak => LIGHT_LEAK,
            Self::Hue => HUE,
            Self::Tile => TILE,
            Self::Posterize => POSTERIZE,
            Self::Edges => EDGES,
            Self::GradientMap => GRADIENT_MAP,
            Self::Grid => GRID,
            Self::Spotlight => SPOTLIGHT,
            Self::Reveal => REVEAL,
            Self::Stretch => STRETCH,
            Self::Dust => DUST,
            Self::Kaleido => KALEIDO,
            Self::Fringe => FRINGE,
            Self::Halftone => HALFTONE,
            Self::Prints => PRINTS,
            Self::Blocks => BLOCKS,
            Self::Rays => RAYS,
            Self::Focus => FOCUS,
            Self::Place => PLACE,
        }
    }

    /// The number the shader switches on.
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            Self::Transform => 0,
            Self::RgbSplit => 1,
            Self::Slices => 2,
            Self::Blur => 3,
            Self::Grade => 4,
            Self::Flash => 5,
            Self::Vignette => 6,
            Self::Noise => 7,
            Self::Scanlines => 8,
            Self::Mirror => 9,
            Self::Pixelate => 10,
            Self::Invert => 11,
            Self::ZoomBlur => 12,
            Self::MotionBlur => 13,
            Self::Swirl => 14,
            Self::Ripple => 15,
            Self::Bulge => 16,
            Self::Glow => 17,
            Self::LightLeak => 18,
            Self::Hue => 19,
            Self::Tile => 20,
            Self::Posterize => 21,
            Self::Edges => 22,
            Self::GradientMap => 23,
            Self::Grid => 24,
            Self::Spotlight => 25,
            Self::Reveal => 26,
            Self::Stretch => 27,
            Self::Dust => 28,
            Self::Kaleido => 29,
            Self::Fringe => 30,
            Self::Halftone => 31,
            Self::Prints => 32,
            Self::Blocks => 33,
            Self::Rays => 34,
            Self::Focus => 35,
            Self::Place => 36,
        }
    }
}

/// A value over the effect's time, in seconds from the effect's start.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum Curve {
    Constant(f32),
    /// `[seconds, value]` keys, held before the first and after the last.
    /// With `repeat`, time wraps every that many seconds. `ease` names how
    /// the value moves across each gap between two keys, the first gap
    /// first; a gap past the end of the list moves as the last one named,
    /// and with none named every gap is a straight line.
    Keys {
        keys: Vec<[f32; 2]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        repeat: Option<f32>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        ease: Vec<Ease>,
    },
    /// A sine from `from` to `to` and back every `period` seconds.
    Wave {
        wave: Wave,
    },
    /// A random value between `from` and `to`, held for `1 / rate` seconds.
    Jitter {
        jitter: Jitter,
    },
}

/// How a value moves from one key to the next.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Ease {
    /// A straight line.
    #[default]
    Linear,
    /// Slow at the start.
    In,
    /// Slow at the end.
    Out,
    /// Slow at both ends.
    InOut,
    /// The first key's value until the second key.
    Hold,
    /// Past the second key's value, then back to it.
    Overshoot,
    /// To the second key's value, then three smaller bounces off it.
    Bounce,
    /// Swings past the second key's value both ways and settles on it.
    Spring,
}

impl Ease {
    /// Every ease, in the order a picker lists them.
    pub const ALL: [Self; 8] = [
        Self::Linear,
        Self::In,
        Self::Out,
        Self::InOut,
        Self::Hold,
        Self::Overshoot,
        Self::Bounce,
        Self::Spring,
    ];

    /// The name a recipe writes.
    #[must_use]
    pub fn key(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::In => "in",
            Self::Out => "out",
            Self::InOut => "in_out",
            Self::Hold => "hold",
            Self::Overshoot => "overshoot",
            Self::Bounce => "bounce",
            Self::Spring => "spring",
        }
    }

    /// How far along the value is, 0 at the first key and 1 at the second,
    /// when the time is `t` of the way across the gap. An overshoot or a
    /// spring passes 1 on the way, and a spring dips under it again.
    #[must_use]
    pub fn at(self, t: f32) -> f32 {
        let t = if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.0
        };
        match self {
            Self::Linear => t,
            Self::In => t * t * t,
            Self::Out => 1.0 - (1.0 - t).powi(3),
            Self::InOut => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    1.0 - (2.0 - 2.0 * t).powi(3) / 2.0
                }
            }
            Self::Hold => {
                if t < 1.0 {
                    0.0
                } else {
                    1.0
                }
            }
            Self::Overshoot => {
                const BACK: f32 = 1.701_58;
                let back = t - 1.0;
                1.0 + (BACK + 1.0) * back.powi(3) + BACK * back.powi(2)
            }
            Self::Bounce => bounce(t),
            Self::Spring => {
                if t <= 0.0 {
                    0.0
                } else if t >= 1.0 {
                    1.0
                } else {
                    let swing = (t * 10.0 - 0.75) * (std::f32::consts::TAU / 3.0);
                    2.0_f32.powf(-10.0 * t) * swing.sin() + 1.0
                }
            }
        }
    }

    /// The least and the most [`Self::at`] reaches. A value between two
    /// keys can leave the span between them by this much.
    #[must_use]
    pub fn reach(self) -> (f32, f32) {
        match self {
            Self::Overshoot | Self::Spring => {
                let mut reach = (0.0_f32, 1.0_f32);
                for step in 0..=REACH_STEPS {
                    #[allow(clippy::cast_precision_loss)]
                    let at = self.at(step as f32 / REACH_STEPS as f32);
                    reach = (reach.0.min(at), reach.1.max(at));
                }
                // Between two samples the curve can pass a little further.
                (reach.0 - REACH_SLACK, reach.1 + REACH_SLACK)
            }
            _ => (0.0, 1.0),
        }
    }
}

/// How finely [`Ease::reach`] samples an ease that leaves 0 to 1.
const REACH_STEPS: u32 = 512;
/// What [`Ease::reach`] adds for the part of a peak between two samples.
const REACH_SLACK: f32 = 0.001;

/// A ball dropped onto the second key's value: there at 4/11 of the way,
/// then three bounces, each a quarter as high as the one before.
fn bounce(t: f32) -> f32 {
    const RATE: f32 = 7.5625;
    const SPAN: f32 = 2.75;
    if t < 1.0 / SPAN {
        RATE * t * t
    } else if t < 2.0 / SPAN {
        let t = t - 1.5 / SPAN;
        RATE * t * t + 0.75
    } else if t < 2.5 / SPAN {
        let t = t - 2.25 / SPAN;
        RATE * t * t + 0.9375
    } else {
        let t = t - 2.625 / SPAN;
        RATE * t * t + 0.984_375
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Wave {
    pub period: f32,
    pub from: f32,
    pub to: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Jitter {
    pub rate: f32,
    pub from: f32,
    pub to: f32,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub seed: u32,
}

/// How a control's value reads in the inspector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlUnit {
    /// A plain number.
    #[default]
    Plain,
    /// A rate, such as 1.5x.
    Times,
    /// A part from 0 to 1 shown as a percent.
    Percent,
    /// An angle in degrees.
    Degrees,
}

/// A pass's parameter a control acts on: the pass's place in the recipe
/// and the parameter's name.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Target(pub usize, pub String);

/// One setting a recipe shows in the inspector, such as Speed or
/// Intensity. Its value plays the effect faster (`speed`), multiplies
/// parameters as the recipe writes them (`scale`), or replaces them
/// (`set`). A value is kept to the parameter's range.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Control {
    /// Lowercase letters, digits and underscores; the name it is stored by.
    pub id: String,
    /// What the inspector calls it.
    pub name: String,
    pub default: f32,
    pub min: f32,
    pub max: f32,
    #[serde(default, skip_serializing_if = "is_plain")]
    pub unit: ControlUnit,
    /// The value multiplies the effect's own time.
    #[serde(default, skip_serializing_if = "is_false")]
    pub speed: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scale: Vec<Target>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub set: Vec<Target>,
    /// The keyframe property it keys under, named when the recipe is read.
    #[serde(skip)]
    pub keyframe: String,
}

impl Control {
    /// Whether the control can be keyed. A speed control cannot: it
    /// multiplies the effect's own time, so a key on it would jump the
    /// effect back and forth rather than speed it up.
    #[must_use]
    pub fn keyable(&self) -> bool {
        !self.speed
    }

    /// `value` kept inside this control's range; the default when it is
    /// not a number.
    #[must_use]
    pub fn clamp(&self, value: f32) -> f32 {
        if value.is_finite() {
            value.clamp(self.min, self.max)
        } else {
            self.default
        }
    }
}

/// One pass of a recipe as written.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Pass {
    pub block: Block,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub params: BTreeMap<String, Curve>,
}

// What a written recipe leaves out: a field at the value a reader gives it.
fn is_zero(value: &u32) -> bool {
    *value == 0
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn is_plain(unit: &ControlUnit) -> bool {
    *unit == ControlUnit::Plain
}

/// An effect as data.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    /// The format version, at most [`RECIPE_VERSION`].
    pub version: u32,
    /// Lowercase letters, digits and dashes.
    pub id: String,
    /// What the picker shows.
    pub name: String,
    /// The shelf category it is listed under, one of
    /// [`EFFECT_CATEGORIES`]; a recipe without one is listed under All only.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub category: String,
    /// The scene its card shows it over, one of
    /// [`crate::preview_scenes`]; empty is [`crate::DEFAULT_PREVIEW_SCENE`].
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub preview: String,
    /// The second of the effect its card's still is taken from; none is
    /// [`crate::DEFAULT_POSTER`]. A flash names the moment it is brightest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub poster: Option<f32>,
    /// The settings the inspector shows for it, at most [`MAX_CONTROLS`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub controls: Vec<Control>,
    pub passes: Vec<Pass>,
}

/// The effects shelf's categories, in the order the shelf lists them.
pub const EFFECT_CATEGORIES: &[&str] = &[
    "Basic", "Motion", "Reveal", "Blur", "Glitch", "Retro", "Light", "Distort",
];

/// One block with every value worked out for one moment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Step {
    pub block: Block,
    /// In the order of [`Block::params`]; the rest are zero.
    pub values: [f32; MAX_VALUES],
}

/// Why a recipe was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecipeError {
    /// Larger than [`MAX_RECIPE_BYTES`].
    TooLarge,
    /// Written for a newer app: update to use it.
    Newer(u32),
    /// Not JSON, or not a recipe's shape.
    Shape(String),
    /// The id or name is empty, too long, or has other characters.
    Name,
    /// No passes, or more than [`MAX_PASSES`].
    Passes,
    /// A pass names a parameter its block does not have.
    UnknownParam { block: Block, name: String },
    /// A value, or a value a curve reaches, is outside the parameter's range.
    Range { block: Block, name: String },
    /// A curve's own numbers are unusable: no keys, too many, keys out of
    /// order, or a period or rate that is not positive.
    Curve { block: Block, name: String },
    /// A control is unusable: a bad or repeated id, a range that does not
    /// hold its default, a speed that can reach zero, a target that names
    /// no pass or parameter, or more than [`MAX_CONTROLS`] of them.
    Control(String),
}

impl fmt::Display for RecipeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge => write!(
                formatter,
                "the recipe is larger than {MAX_RECIPE_BYTES} bytes"
            ),
            Self::Newer(version) => {
                write!(
                    formatter,
                    "the recipe needs format {version}; update the app"
                )
            }
            Self::Shape(reason) => write!(formatter, "not a recipe: {reason}"),
            Self::Name => write!(formatter, "the recipe's id or name is not usable"),
            Self::Passes => write!(formatter, "a recipe has 1 to {MAX_PASSES} passes"),
            Self::UnknownParam { block, name } => {
                write!(formatter, "{block:?} has no parameter {name}")
            }
            Self::Range { block, name } => {
                write!(formatter, "{block:?} {name} goes outside its range")
            }
            Self::Curve { block, name } => write!(formatter, "{block:?} {name} has a broken curve"),
            Self::Control(id) => write!(formatter, "the control {id} is not usable"),
        }
    }
}

impl std::error::Error for RecipeError {}

impl Recipe {
    /// Read and check a recipe.
    pub fn read(bytes: &[u8]) -> Result<Self, RecipeError> {
        if bytes.len() > MAX_RECIPE_BYTES {
            return Err(RecipeError::TooLarge);
        }
        // The version first: a newer recipe may use blocks this app cannot
        // name, and that should read as "update", not as a broken file.
        #[derive(Deserialize)]
        struct Version {
            version: u32,
        }
        let Version { version } =
            serde_json::from_slice(bytes).map_err(|error| RecipeError::Shape(error.to_string()))?;
        if version > RECIPE_VERSION {
            return Err(RecipeError::Newer(version));
        }
        let recipe: Self =
            serde_json::from_slice(bytes).map_err(|error| RecipeError::Shape(error.to_string()))?;
        recipe.checked()
    }

    /// A recipe built in memory, checked as [`Self::read`] checks a file,
    /// with each control's keyframe property named.
    pub fn checked(mut self) -> Result<Self, RecipeError> {
        self.check()?;
        for control in &mut self.controls {
            control.keyframe = cutlass_model::effect_control_key(&control.id);
        }
        Ok(self)
    }

    /// The recipe as the JSON [`Self::read`] reads back to the same recipe.
    /// It names the lowest version that can hold it, and leaves out every
    /// field at the value a reader gives it. A recipe that would not read
    /// back is refused and nothing is written.
    pub fn to_json(&self) -> Result<String, RecipeError> {
        let mut written = self.clone();
        written.version = self.least_version();
        written.check()?;
        to_json(&written)
    }

    fn check(&self) -> Result<(), RecipeError> {
        let id_ok = !self.id.is_empty()
            && self.id.len() <= 64
            && self
                .id
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-');
        let name_ok = !self.name.trim().is_empty() && self.name.chars().count() <= 64;
        if self.version == 0 {
            return Err(RecipeError::Shape("version 0".to_owned()));
        }
        if !id_ok || !name_ok {
            return Err(RecipeError::Name);
        }
        if !self.preview.is_empty() && crate::preview_scene(&self.preview).is_none() {
            return Err(RecipeError::Shape(format!(
                "no preview scene {}",
                self.preview
            )));
        }
        if self
            .poster
            .is_some_and(|poster| !poster.is_finite() || !(0.0..=60.0).contains(&poster))
        {
            return Err(RecipeError::Shape(
                "a poster time from 0 to 60 s".to_owned(),
            ));
        }
        if self.passes.is_empty() || self.passes.len() > MAX_PASSES {
            return Err(RecipeError::Passes);
        }
        self.check_controls()?;
        check_passes(&self.passes)?;
        check_version(self.version, self.least_version())
    }

    /// The lowest format version that can hold this recipe: what a file of
    /// it names, so an app from before a field reads "update" and not a
    /// broken file.
    #[must_use]
    pub fn least_version(&self) -> u32 {
        least_version(self.passes.iter().flat_map(|pass| pass.params.values()))
    }

    fn check_controls(&self) -> Result<(), RecipeError> {
        if self.controls.len() > MAX_CONTROLS {
            return Err(RecipeError::Control(String::new()));
        }
        for (index, control) in self.controls.iter().enumerate() {
            let refuse = || Err(RecipeError::Control(control.id.clone()));
            let id_ok = !control.id.is_empty()
                && control.id.len() <= 32
                && control
                    .id
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_');
            let repeated = self.controls[..index]
                .iter()
                .any(|other| other.id == control.id);
            let numbers = [control.default, control.min, control.max];
            let range_ok = numbers.iter().all(|value| value.is_finite())
                && control.min <= control.default
                && control.default <= control.max;
            let name_ok = !control.name.trim().is_empty() && control.name.chars().count() <= 32;
            if !id_ok || repeated || !range_ok || !name_ok || (control.speed && control.min <= 0.0)
            {
                return refuse();
            }
            for Target(pass, name) in control.scale.iter().chain(&control.set) {
                let known = self
                    .passes
                    .get(*pass)
                    .is_some_and(|pass| pass.block.params().iter().any(|param| param.name == name));
                if !known {
                    return refuse();
                }
            }
        }
        Ok(())
    }

    /// Every control at its default.
    #[must_use]
    pub fn default_controls(&self) -> BTreeMap<String, f32> {
        self.controls
            .iter()
            .map(|control| (control.id.clone(), control.default))
            .collect()
    }

    /// The id stored on a draft's effect material.
    #[must_use]
    pub fn effect_id(&self) -> String {
        format!("{RECIPE_PREFIX}{}", self.id)
    }

    /// The scene its card shows it over.
    #[must_use]
    pub fn preview_scene(&self) -> &str {
        if self.preview.is_empty() {
            crate::DEFAULT_PREVIEW_SCENE
        } else {
            &self.preview
        }
    }

    /// The moment of the effect its card's still shows.
    #[must_use]
    pub fn poster_time(&self) -> Micros {
        let seconds = self.poster.unwrap_or(crate::DEFAULT_POSTER);
        Micros((f64::from(seconds) * 1_000_000.0).round() as i64)
    }

    /// Every pass with its values at `time` from the effect's start. A time
    /// before the start is the start.
    #[must_use]
    pub fn steps_at(&self, time: Micros) -> Vec<Step> {
        self.steps_with(time, &BTreeMap::new())
    }

    /// Every pass at `time` with the controls set to `controls`, by id. A
    /// control left out is at its default, and a value past its range is
    /// kept to it; each parameter stays in its own range.
    #[must_use]
    pub fn steps_with(&self, time: Micros, controls: &BTreeMap<String, f32>) -> Vec<Step> {
        let value_of = |control: &Control| {
            control.clamp(
                controls
                    .get(&control.id)
                    .copied()
                    .unwrap_or(control.default),
            )
        };
        let speed: f64 = self
            .controls
            .iter()
            .filter(|control| control.speed)
            .map(|control| f64::from(value_of(control)))
            .product();
        // Seconds as f64 first: a long timeline keeps its fraction.
        let seconds = (time.0.max(0) as f64 / 1_000_000.0 * speed) as f32;
        self.passes
            .iter()
            .enumerate()
            .map(|(index, pass)| {
                let mut values = [0.0; MAX_VALUES];
                for (slot, param) in values.iter_mut().zip(pass.block.params()) {
                    let mut value = pass
                        .params
                        .get(param.name)
                        .map_or(param.default, |curve| curve.at(seconds));
                    let aims = |targets: &[Target]| {
                        targets
                            .iter()
                            .any(|Target(pass, name)| *pass == index && name == param.name)
                    };
                    for control in &self.controls {
                        if aims(&control.scale) {
                            value *= value_of(control);
                        }
                        if aims(&control.set) {
                            value = value_of(control);
                        }
                    }
                    *slot = value.clamp(param.min, param.max);
                }
                Step {
                    block: pass.block,
                    values,
                }
            })
            .collect()
    }
}

pub(crate) fn check_curve(block: Block, param: &Param, curve: &Curve) -> Result<(), RecipeError> {
    let range = |value: f32| {
        if value.is_finite() && value >= param.min && value <= param.max {
            Ok(())
        } else {
            Err(RecipeError::Range {
                block,
                name: param.name.to_owned(),
            })
        }
    };
    let broken = || RecipeError::Curve {
        block,
        name: param.name.to_owned(),
    };
    let positive = |value: f32| value.is_finite() && value > 0.0;
    match curve {
        Curve::Constant(value) => range(*value),
        Curve::Keys { keys, repeat, ease } => {
            if keys.is_empty() || keys.len() > MAX_KEYS {
                return Err(broken());
            }
            if repeat.is_some_and(|repeat| !positive(repeat)) {
                return Err(broken());
            }
            // One ease for each gap at most; a longer list names an ease
            // for a gap that is not there.
            if ease.len() > keys.len().saturating_sub(1) {
                return Err(broken());
            }
            let mut last = 0.0;
            for [time, value] in keys {
                if !time.is_finite() || *time < last {
                    return Err(broken());
                }
                last = *time;
                range(*value)?;
            }
            // An overshoot or a spring leaves the span between its two
            // keys, so the furthest it goes is checked as well.
            for (gap, pair) in keys.windows(2).enumerate() {
                let (from, to) = (pair[0][1], pair[1][1]);
                let (least, most) = gap_ease(ease, gap).reach();
                range(from + (to - from) * least)?;
                range(from + (to - from) * most)?;
            }
            Ok(())
        }
        Curve::Wave { wave } => {
            if !positive(wave.period) {
                return Err(broken());
            }
            range(wave.from)?;
            range(wave.to)
        }
        Curve::Jitter { jitter } => {
            if !positive(jitter.rate) || jitter.rate > MAX_JITTER_RATE {
                return Err(broken());
            }
            range(jitter.from)?;
            range(jitter.to)
        }
    }
}

/// `recipe` as indented JSON with a newline at the end, no larger than a
/// reader takes.
pub(crate) fn to_json<T: Serialize>(recipe: &T) -> Result<String, RecipeError> {
    let mut json = serde_json::to_string_pretty(recipe)
        .map_err(|error| RecipeError::Shape(error.to_string()))?;
    json.push('\n');
    if json.len() > MAX_RECIPE_BYTES {
        return Err(RecipeError::TooLarge);
    }
    Ok(json)
}

/// The lowest format version that can hold `curves`.
pub(crate) fn least_version<'a>(curves: impl IntoIterator<Item = &'a Curve>) -> u32 {
    let eased = curves
        .into_iter()
        .any(|curve| matches!(curve, Curve::Keys { ease, .. } if !ease.is_empty()));
    if eased { EASE_VERSION } else { 1 }
}

/// A recipe that names `version` may use what `least` first had.
pub(crate) fn check_version(version: u32, least: u32) -> Result<(), RecipeError> {
    if version < least {
        return Err(RecipeError::Shape(format!(
            "version {version} names what version {least} added"
        )));
    }
    Ok(())
}

/// Every parameter of `passes` is one its block has, inside its range.
pub(crate) fn check_passes(passes: &[Pass]) -> Result<(), RecipeError> {
    for pass in passes {
        for (name, curve) in &pass.params {
            let Some(param) = pass.block.params().iter().find(|p| p.name == name) else {
                return Err(RecipeError::UnknownParam {
                    block: pass.block,
                    name: name.clone(),
                });
            };
            check_curve(pass.block, param, curve)?;
        }
    }
    Ok(())
}

impl Curve {
    /// The value at `seconds`, never negative.
    pub(crate) fn at(&self, seconds: f32) -> f32 {
        match self {
            Self::Constant(value) => *value,
            Self::Keys { keys, repeat, ease } => {
                let time = repeat.map_or(seconds, |repeat| seconds.rem_euclid(repeat));
                along(keys, ease, time)
            }
            Self::Wave { wave } => {
                let turn = (seconds / wave.period).fract();
                let rise = 0.5 - 0.5 * (turn * std::f32::consts::TAU).cos();
                wave.from + (wave.to - wave.from) * rise
            }
            Self::Jitter { jitter } => {
                let step = (seconds * jitter.rate).floor() as u64;
                let unit = unit_hash(u64::from(jitter.seed), step);
                jitter.from + (jitter.to - jitter.from) * unit
            }
        }
    }
}

/// The ease of the gap after key `gap`: the one named for it, or the last
/// one named, or a straight line when none is.
#[must_use]
pub fn gap_ease(ease: &[Ease], gap: usize) -> Ease {
    ease.get(gap)
        .or_else(|| ease.last())
        .copied()
        .unwrap_or_default()
}

fn along(keys: &[[f32; 2]], ease: &[Ease], time: f32) -> f32 {
    let Some(first) = keys.first() else {
        return 0.0;
    };
    if time <= first[0] {
        return first[1];
    }
    for (gap, pair) in keys.windows(2).enumerate() {
        let ([t0, v0], [t1, v1]) = (pair[0], pair[1]);
        if time <= t1 {
            if t1 <= t0 {
                return v1;
            }
            let across = gap_ease(ease, gap).at((time - t0) / (t1 - t0));
            return v0 + (v1 - v0) * across;
        }
    }
    keys.last().map_or(0.0, |last| last[1])
}

/// A number in `[0, 1)` from a seed and a step, the same on every machine.
fn unit_hash(seed: u64, step: u64) -> f32 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(step);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    // The top 24 bits fit an f32 exactly.
    (x >> 40) as f32 / (1u64 << 24) as f32
}

const BUNDLED_FILES: [&str; 200] = [
    include_str!("../recipes/vignette.json"),
    include_str!("../recipes/black-white.json"),
    include_str!("../recipes/vivid.json"),
    include_str!("../recipes/fade-in.json"),
    include_str!("../recipes/invert.json"),
    include_str!("../recipes/duotone.json"),
    include_str!("../recipes/golden-duotone.json"),
    include_str!("../recipes/ocean-duotone.json"),
    include_str!("../recipes/warm.json"),
    include_str!("../recipes/cool.json"),
    include_str!("../recipes/noir.json"),
    include_str!("../recipes/matte.json"),
    include_str!("../recipes/teal-orange.json"),
    include_str!("../recipes/pastel.json"),
    include_str!("../recipes/cinematic.json"),
    include_str!("../recipes/bright.json"),
    include_str!("../recipes/moody.json"),
    include_str!("../recipes/white-in.json"),
    include_str!("../recipes/pink-duotone.json"),
    include_str!("../recipes/lime-duotone.json"),
    include_str!("../recipes/muted.json"),
    include_str!("../recipes/punchy.json"),
    include_str!("../recipes/shake.json"),
    include_str!("../recipes/zoom-pulse.json"),
    include_str!("../recipes/earthquake.json"),
    include_str!("../recipes/heartbeat.json"),
    include_str!("../recipes/slow-zoom.json"),
    include_str!("../recipes/sway.json"),
    include_str!("../recipes/float.json"),
    include_str!("../recipes/spin-in.json"),
    include_str!("../recipes/bounce.json"),
    include_str!("../recipes/whip-in.json"),
    include_str!("../recipes/whip-pan.json"),
    include_str!("../recipes/impact.json"),
    include_str!("../recipes/shockwave.json"),
    include_str!("../recipes/squeeze.json"),
    include_str!("../recipes/zoom-in.json"),
    include_str!("../recipes/zoom-out.json"),
    include_str!("../recipes/pan-left.json"),
    include_str!("../recipes/pan-right.json"),
    include_str!("../recipes/tilt-up.json"),
    include_str!("../recipes/rock.json"),
    include_str!("../recipes/jello.json"),
    include_str!("../recipes/drop-in.json"),
    include_str!("../recipes/slide-up.json"),
    include_str!("../recipes/pop-in.json"),
    include_str!("../recipes/spin.json"),
    include_str!("../recipes/wobble.json"),
    include_str!("../recipes/handheld.json"),
    include_str!("../recipes/flash-zoom.json"),
    include_str!("../recipes/tumble-in.json"),
    include_str!("../recipes/hit.json"),
    include_str!("../recipes/iris-in.json"),
    include_str!("../recipes/wipe-in.json"),
    include_str!("../recipes/diagonal-wipe.json"),
    include_str!("../recipes/blinds.json"),
    include_str!("../recipes/barn-doors.json"),
    include_str!("../recipes/split-open.json"),
    include_str!("../recipes/clock-wipe.json"),
    include_str!("../recipes/diamond.json"),
    include_str!("../recipes/checkerboard.json"),
    include_str!("../recipes/dissolve.json"),
    include_str!("../recipes/wipe-down.json"),
    include_str!("../recipes/wipe-left.json"),
    include_str!("../recipes/soft-wipe.json"),
    include_str!("../recipes/shutters.json"),
    include_str!("../recipes/zoom-reveal.json"),
    include_str!("../recipes/spin-reveal.json"),
    include_str!("../recipes/pixel-reveal.json"),
    include_str!("../recipes/glitch-reveal.json"),
    include_str!("../recipes/blur-in.json"),
    include_str!("../recipes/blur.json"),
    include_str!("../recipes/rack-focus.json"),
    include_str!("../recipes/dreamy.json"),
    include_str!("../recipes/zoom-blur.json"),
    include_str!("../recipes/zoom-burst.json"),
    include_str!("../recipes/motion-blur.json"),
    include_str!("../recipes/speed-lines.json"),
    include_str!("../recipes/tilt-shift.json"),
    include_str!("../recipes/lens-blur.json"),
    include_str!("../recipes/spin-blur.json"),
    include_str!("../recipes/blur-out.json"),
    include_str!("../recipes/focus-pulse.json"),
    include_str!("../recipes/dream-zoom.json"),
    include_str!("../recipes/shake-blur.json"),
    include_str!("../recipes/spin-out.json"),
    include_str!("../recipes/soft-focus.json"),
    include_str!("../recipes/blur-flash.json"),
    include_str!("../recipes/vertical-blur.json"),
    include_str!("../recipes/frosted.json"),
    include_str!("../recipes/rgb-split.json"),
    include_str!("../recipes/glitch.json"),
    include_str!("../recipes/digital-glitch.json"),
    include_str!("../recipes/signal-loss.json"),
    include_str!("../recipes/rgb-shift.json"),
    include_str!("../recipes/invert-flicker.json"),
    include_str!("../recipes/stutter.json"),
    include_str!("../recipes/neon-edges.json"),
    include_str!("../recipes/color-glitch.json"),
    include_str!("../recipes/data-blocks.json"),
    include_str!("../recipes/datamosh.json"),
    include_str!("../recipes/corrupt.json"),
    include_str!("../recipes/bad-signal.json"),
    include_str!("../recipes/chroma-shake.json"),
    include_str!("../recipes/static.json"),
    include_str!("../recipes/interlace.json"),
    include_str!("../recipes/glitch-flash.json"),
    include_str!("../recipes/hologram.json"),
    include_str!("../recipes/pixel-glitch.json"),
    include_str!("../recipes/scan-glitch.json"),
    include_str!("../recipes/error.json"),
    include_str!("../recipes/retro-glitch.json"),
    include_str!("../recipes/wave-glitch.json"),
    include_str!("../recipes/tear.json"),
    include_str!("../recipes/vhs.json"),
    include_str!("../recipes/film-grain.json"),
    include_str!("../recipes/old-film.json"),
    include_str!("../recipes/sepia.json"),
    include_str!("../recipes/8-bit.json"),
    include_str!("../recipes/crt.json"),
    include_str!("../recipes/pop-art.json"),
    include_str!("../recipes/comic.json"),
    include_str!("../recipes/sketch.json"),
    include_str!("../recipes/tv-off.json"),
    include_str!("../recipes/tv-on.json"),
    include_str!("../recipes/thermal.json"),
    include_str!("../recipes/night-vision.json"),
    include_str!("../recipes/halftone.json"),
    include_str!("../recipes/comic-dots.json"),
    include_str!("../recipes/dusty-film.json"),
    include_str!("../recipes/film-scratches.json"),
    include_str!("../recipes/faded-film.json"),
    include_str!("../recipes/super-8.json"),
    include_str!("../recipes/newsprint.json"),
    include_str!("../recipes/pocket-game.json"),
    include_str!("../recipes/vhs-rewind.json"),
    include_str!("../recipes/retro-tv.json"),
    include_str!("../recipes/cross-process.json"),
    include_str!("../recipes/silent-film.json"),
    include_str!("../recipes/print-dots.json"),
    include_str!("../recipes/white-flash.json"),
    include_str!("../recipes/black-flash.json"),
    include_str!("../recipes/strobe.json"),
    include_str!("../recipes/flicker.json"),
    include_str!("../recipes/red-pulse.json"),
    include_str!("../recipes/exposure-flash.json"),
    include_str!("../recipes/golden-hour.json"),
    include_str!("../recipes/moonlight.json"),
    include_str!("../recipes/glow.json"),
    include_str!("../recipes/soft-glow.json"),
    include_str!("../recipes/light-leak.json"),
    include_str!("../recipes/rose-leak.json"),
    include_str!("../recipes/sun-flare.json"),
    include_str!("../recipes/color-cycle.json"),
    include_str!("../recipes/spotlight.json"),
    include_str!("../recipes/searchlight.json"),
    include_str!("../recipes/film-burn.json"),
    include_str!("../recipes/rainbow-map.json"),
    include_str!("../recipes/god-rays.json"),
    include_str!("../recipes/light-burst.json"),
    include_str!("../recipes/star-glow.json"),
    include_str!("../recipes/neon-glow.json"),
    include_str!("../recipes/flare-sweep.json"),
    include_str!("../recipes/amber-leak.json"),
    include_str!("../recipes/blue-leak.json"),
    include_str!("../recipes/dual-leak.json"),
    include_str!("../recipes/disco.json"),
    include_str!("../recipes/twinkle.json"),
    include_str!("../recipes/radiance.json"),
    include_str!("../recipes/blink.json"),
    include_str!("../recipes/mirror.json"),
    include_str!("../recipes/mirror-vertical.json"),
    include_str!("../recipes/kaleidoscope.json"),
    include_str!("../recipes/pixelate.json"),
    include_str!("../recipes/depixelate.json"),
    include_str!("../recipes/dutch-angle.json"),
    include_str!("../recipes/swirl.json"),
    include_str!("../recipes/vortex.json"),
    include_str!("../recipes/water-ripple.json"),
    include_str!("../recipes/wavy.json"),
    include_str!("../recipes/heat-haze.json"),
    include_str!("../recipes/fisheye.json"),
    include_str!("../recipes/pinch.json"),
    include_str!("../recipes/bulge-pulse.json"),
    include_str!("../recipes/split-screen.json"),
    include_str!("../recipes/psychedelic.json"),
    include_str!("../recipes/triple-stack.json"),
    include_str!("../recipes/side-by-side.json"),
    include_str!("../recipes/six-grid.json"),
    include_str!("../recipes/nine-grid.json"),
    include_str!("../recipes/prism.json"),
    include_str!("../recipes/mandala.json"),
    include_str!("../recipes/lens-fringe.json"),
    include_str!("../recipes/photo-stack.json"),
    include_str!("../recipes/photo-drop.json"),
    include_str!("../recipes/funhouse.json"),
    include_str!("../recipes/twist.json"),
    include_str!("../recipes/liquid.json"),
    include_str!("../recipes/melt.json"),
    include_str!("../recipes/bubble.json"),
];

/// The recipes that ship in the app, in picker order.
#[must_use]
pub fn bundled() -> &'static [Recipe] {
    static RECIPES: OnceLock<Vec<Recipe>> = OnceLock::new();
    RECIPES.get_or_init(|| {
        // A test reads every file, so none is dropped here in practice.
        BUNDLED_FILES
            .iter()
            .filter_map(|file| Recipe::read(file.as_bytes()).ok())
            .collect()
    })
}

/// The bundled recipe stored under `effect_id`, such as `recipe/shake`.
#[must_use]
pub fn bundled_recipe(effect_id: &str) -> Option<&'static Recipe> {
    let id = effect_id.strip_prefix(RECIPE_PREFIX)?;
    bundled().iter().find(|recipe| recipe.id == id)
}

#[cfg(test)]
mod tests {
    use super::{
        BUNDLED_FILES, Block, EFFECT_CATEGORIES, MAX_PASSES, MAX_RECIPE_BYTES, Recipe, RecipeError,
        bundled, bundled_recipe, unit_hash,
    };
    use cutlass_model::Micros;

    fn read(text: &str) -> Result<Recipe, RecipeError> {
        Recipe::read(text.as_bytes())
    }

    fn at(recipe: &Recipe, seconds: f64) -> Vec<f32> {
        let time = Micros((seconds * 1_000_000.0) as i64);
        recipe.steps_at(time)[0].values.to_vec()
    }

    #[test]
    fn every_bundled_recipe_reads_and_ids_are_unique() {
        assert_eq!(bundled().len(), BUNDLED_FILES.len());
        let mut ids: Vec<&str> = bundled().iter().map(|recipe| recipe.id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), BUNDLED_FILES.len());
        assert_eq!(
            bundled_recipe("recipe/shake").map(|recipe| recipe.name.as_str()),
            Some("Shake")
        );
        assert!(bundled_recipe("shake").is_none());
        assert!(bundled_recipe("recipe/none").is_none());
        // Every bundled effect is filed under a shelf category.
        for recipe in bundled() {
            assert!(
                EFFECT_CATEGORIES.contains(&recipe.category.as_str()),
                "{} has category {:?}",
                recipe.id,
                recipe.category
            );
        }
        // And names the scene its card shows it over.
        for recipe in bundled() {
            assert!(!recipe.preview.is_empty(), "{} names no scene", recipe.id);
        }
        // One from elsewhere may leave it out.
        let plain =
            read(r#"{"version":1,"id":"a","name":"A","passes":[{"block":"grade","params":{}}]}"#);
        assert!(plain.is_ok_and(|recipe| recipe.category.is_empty()));
    }

    #[test]
    fn every_block_fits_the_values_a_step_carries_and_its_code_is_its_place() {
        for (index, block) in Block::ALL.iter().enumerate() {
            assert!(block.params().len() <= super::MAX_VALUES, "{block:?}");
            assert_eq!(block.code() as usize, index, "{block:?}");
            for param in block.params() {
                assert!(
                    param.min <= param.default && param.default <= param.max,
                    "{block:?} {}",
                    param.name
                );
            }
        }
    }

    #[test]
    fn a_left_out_value_is_the_default_and_order_follows_the_block() {
        let recipe =
            read(r#"{"version":1,"id":"a","name":"A","passes":[{"block":"grade","params":{"saturation":0}}]}"#)
                .unwrap();
        assert_eq!(at(&recipe, 0.0)[..4], [0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn keys_interpolate_hold_and_repeat() {
        let recipe = read(
            r#"{"version":1,"id":"a","name":"A","passes":[{"block":"flash","params":{
                "amount":{"keys":[[0.5,0],[1.5,1]],"repeat":2}}}]}"#,
        )
        .unwrap();
        assert_eq!(at(&recipe, 0.0)[0], 0.0);
        assert!((at(&recipe, 1.0)[0] - 0.5).abs() < 1e-5);
        assert_eq!(at(&recipe, 1.9)[0], 1.0);
        assert!((at(&recipe, 3.0)[0] - 0.5).abs() < 1e-5);
        assert_eq!(at(&recipe, -4.0)[0], 0.0);
    }

    #[test]
    fn a_wave_starts_at_from_and_peaks_at_half_its_period() {
        let recipe = read(
            r#"{"version":1,"id":"a","name":"A","passes":[{"block":"transform","params":{
                "scale":{"wave":{"period":1,"from":1,"to":1.5}}}}]}"#,
        )
        .unwrap();
        assert!((at(&recipe, 0.0)[0] - 1.0).abs() < 1e-5);
        assert!((at(&recipe, 0.5)[0] - 1.5).abs() < 1e-5);
        assert!((at(&recipe, 1.0)[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn jitter_holds_within_a_step_stays_in_range_and_repeats_exactly() {
        let recipe = read(
            r#"{"version":1,"id":"a","name":"A","passes":[{"block":"transform","params":{
                "x":{"jitter":{"rate":10,"from":-0.1,"to":0.1,"seed":7}}}}]}"#,
        )
        .unwrap();
        assert_eq!(at(&recipe, 0.01)[1], at(&recipe, 0.09)[1]);
        let values: Vec<f32> = (0..200)
            .map(|step| at(&recipe, f64::from(step) * 0.1)[1])
            .collect();
        assert!(values.iter().all(|value| (-0.1..=0.1).contains(value)));
        assert!(values.windows(2).any(|pair| pair[0] != pair[1]));
        assert_eq!(at(&recipe, 3.3), at(&recipe, 3.3));
        assert!((0..10_000).all(|step| (0.0..1.0).contains(&unit_hash(3, step))));
    }

    #[test]
    fn values_outside_a_range_are_refused_even_inside_a_curve() {
        for params in [
            r#"{"scale":9}"#,
            r#"{"scale":{"keys":[[0,1],[1,5]]}}"#,
            r#"{"x":{"wave":{"period":1,"from":0,"to":2}}}"#,
            r#"{"rotate":{"jitter":{"rate":5,"from":-200,"to":0}}}"#,
        ] {
            let text = format!(
                r#"{{"version":1,"id":"a","name":"A","passes":[{{"block":"transform","params":{params}}}]}}"#
            );
            assert!(
                matches!(
                    read(&text),
                    Err(RecipeError::Range {
                        block: Block::Transform,
                        ..
                    })
                ),
                "{params}"
            );
        }
    }

    #[test]
    fn broken_curves_are_refused() {
        for params in [
            r#"{"x":{"keys":[]}}"#,
            r#"{"x":{"keys":[[1,0],[0,0]]}}"#,
            r#"{"x":{"keys":[[0,0]],"repeat":0}}"#,
            r#"{"x":{"wave":{"period":0,"from":0,"to":0}}}"#,
            r#"{"x":{"jitter":{"rate":500,"from":0,"to":0}}}"#,
            r#"{"x":{"jitter":{"rate":-1,"from":0,"to":0}}}"#,
        ] {
            let text = format!(
                r#"{{"version":1,"id":"a","name":"A","passes":[{{"block":"transform","params":{params}}}]}}"#
            );
            assert!(
                matches!(read(&text), Err(RecipeError::Curve { .. })),
                "{params}"
            );
        }
        let many: Vec<String> = (0..65).map(|key| format!("[{key},0]")).collect();
        let text = format!(
            r#"{{"version":1,"id":"a","name":"A","passes":[{{"block":"transform","params":{{"x":{{"keys":[{}]}}}}}}]}}"#,
            many.join(",")
        );
        assert!(matches!(read(&text), Err(RecipeError::Curve { .. })));
    }

    #[test]
    fn unknown_blocks_params_and_fields_are_refused() {
        assert!(matches!(
            read(r#"{"version":1,"id":"a","name":"A","passes":[{"block":"shader"}]}"#),
            Err(RecipeError::Shape(_))
        ));
        assert!(matches!(
            read(
                r#"{"version":1,"id":"a","name":"A","passes":[{"block":"blur","params":{"code":1}}]}"#
            ),
            Err(RecipeError::UnknownParam {
                block: Block::Blur,
                ..
            })
        ));
        assert!(matches!(
            read(r#"{"version":1,"id":"a","name":"A","wgsl":"x","passes":[{"block":"blur"}]}"#),
            Err(RecipeError::Shape(_))
        ));
        assert!(matches!(read("not json"), Err(RecipeError::Shape(_))));
    }

    #[test]
    fn names_pass_counts_and_size_are_checked() {
        assert_eq!(
            read(r#"{"version":1,"id":"A b","name":"A","passes":[{"block":"blur"}]}"#),
            Err(RecipeError::Name)
        );
        assert_eq!(
            read(r#"{"version":1,"id":"a","name":" ","passes":[{"block":"blur"}]}"#),
            Err(RecipeError::Name)
        );
        assert_eq!(
            read(r#"{"version":1,"id":"a","name":"A","passes":[]}"#),
            Err(RecipeError::Passes)
        );
        let passes = [r#"{"block":"blur"}"#; MAX_PASSES + 1].join(",");
        assert_eq!(
            read(&format!(
                r#"{{"version":1,"id":"a","name":"A","passes":[{passes}]}}"#
            )),
            Err(RecipeError::Passes)
        );
        let padded = format!("{}{}", " ".repeat(MAX_RECIPE_BYTES), "{}");
        assert_eq!(read(&padded), Err(RecipeError::TooLarge));
    }

    fn eased(version: u32, keys: &str, ease: &str) -> Result<Recipe, RecipeError> {
        read(&format!(
            r#"{{"version":{version},"id":"a","name":"A","passes":[{{"block":"transform","params":{{"scale":{{"keys":{keys},"ease":{ease}}}}}}}]}}"#
        ))
    }

    fn scale_at(recipe: &Recipe, seconds: f64) -> f32 {
        #[allow(clippy::cast_possible_truncation)]
        let time = Micros((seconds * 1_000_000.0).round() as i64);
        recipe.steps_at(time)[0].values[0]
    }

    #[test]
    fn an_ease_bends_the_gap_it_is_named_for_and_the_gaps_after_it() {
        let plain = eased(1, "[[0,1],[1,2],[2,1]]", "[]").unwrap();
        assert!((scale_at(&plain, 0.25) - 1.25).abs() < 1e-6);
        // Slow at the end: further along than a straight line early on.
        let out = eased(2, "[[0,1],[1,2],[2,1]]", r#"["out"]"#).unwrap();
        assert!(scale_at(&out, 0.25) > 1.5);
        // The second gap has no ease of its own, so it moves as the first.
        assert!(scale_at(&out, 1.25) < 1.5);
        // Each gap takes the ease named for it.
        let mixed = eased(2, "[[0,1],[1,2],[2,1]]", r#"["hold","linear"]"#).unwrap();
        assert!((scale_at(&mixed, 0.99) - 1.0).abs() < 1e-6);
        assert!((scale_at(&mixed, 1.5) - 1.5).abs() < 1e-6);
        // Every ease starts on the first key and ends on the second.
        for ease in super::Ease::ALL {
            assert!(ease.at(0.0).abs() < 1e-6, "{ease:?}");
            assert!((ease.at(1.0) - 1.0).abs() < 1e-6, "{ease:?}");
            assert!((ease.at(f32::NAN)).abs() < 1e-6, "{ease:?}");
            let recipe = eased(2, "[[0,1],[1,1.5]]", &format!(r#"["{}"]"#, ease.key()));
            assert!(recipe.is_ok(), "{ease:?} reads by its key");
        }
    }

    #[test]
    fn an_ease_that_passes_its_key_is_checked_against_the_range() {
        // Scale tops out at 4: a spring to 4 would pass it, so it is refused.
        assert!(matches!(
            eased(2, "[[0,1],[1,4]]", r#"["spring"]"#),
            Err(RecipeError::Range { .. })
        ));
        assert!(matches!(
            eased(2, "[[0,1],[1,4]]", r#"["overshoot"]"#),
            Err(RecipeError::Range { .. })
        ));
        // The same keys in a straight line stay inside it.
        assert!(eased(2, "[[0,1],[1,4]]", r#"["out"]"#).is_ok());
        // With room above the second key, the overshoot is drawn.
        let over = eased(2, "[[0,1],[1,2]]", r#"["overshoot"]"#).unwrap();
        let peak = (0..=100)
            .map(|step| scale_at(&over, f64::from(step) / 100.0))
            .fold(0.0_f32, f32::max);
        assert!(peak > 2.05 && peak < 2.2, "{peak}");
        let (least, most) = super::Ease::Spring.reach();
        assert!(least < 1.0 && most > 1.3, "{least} {most}");
        assert_eq!(super::Ease::Bounce.reach(), (0.0, 1.0));
    }

    #[test]
    fn an_ease_needs_version_two_and_a_gap_to_bend() {
        assert!(matches!(
            eased(1, "[[0,1],[1,2]]", r#"["out"]"#),
            Err(RecipeError::Shape(_))
        ));
        assert_eq!(eased(1, "[[0,1],[1,2]]", "[]").unwrap().least_version(), 1);
        assert_eq!(
            eased(2, "[[0,1],[1,2]]", r#"["out"]"#)
                .unwrap()
                .least_version(),
            2
        );
        // More eases than gaps, or a name the app does not have.
        assert!(matches!(
            eased(2, "[[0,1],[1,2]]", r#"["out","in"]"#),
            Err(RecipeError::Curve { .. })
        ));
        assert!(matches!(
            eased(2, "[[0,1],[1,2]]", r#"["wobble"]"#),
            Err(RecipeError::Shape(_))
        ));
    }

    #[test]
    fn every_bundled_recipe_writes_and_reads_back_the_same() {
        for recipe in bundled() {
            let json = recipe.to_json().expect("writes");
            assert!(json.len() <= MAX_RECIPE_BYTES);
            assert_eq!(&read(&json).expect("reads back"), recipe, "{}", recipe.id);
        }
    }

    #[test]
    fn a_written_recipe_names_its_least_version_and_skips_defaults() {
        let plain = read(
            r#"{"version":2,"id":"a","name":"A","passes":[{"block":"blur","params":{"radius":{"jitter":{"rate":4,"from":0,"to":0.05}}}}]}"#,
        )
        .unwrap();
        let json = plain.to_json().unwrap();
        // No ease, so an app from before eases can read it.
        assert!(json.contains("\"version\": 1"), "{json}");
        for absent in ["category", "preview", "poster", "controls", "seed", "ease"] {
            assert!(!json.contains(absent), "{absent} in {json}");
        }
        let eased = eased(2, "[[0,1],[1,2]]", r#"["out"]"#).unwrap();
        let json = eased.to_json().unwrap();
        assert!(json.contains("\"version\": 2"), "{json}");
        assert_eq!(read(&json).unwrap(), eased);
    }

    #[test]
    fn a_recipe_that_would_not_read_back_is_not_written() {
        let mut recipe =
            read(r#"{"version":1,"id":"a","name":"A","passes":[{"block":"blur"}]}"#).unwrap();
        recipe.passes.clear();
        assert_eq!(recipe.to_json(), Err(RecipeError::Passes));
        assert_eq!(recipe.clone().checked(), Err(RecipeError::Passes));
        let mut recipe =
            read(r#"{"version":1,"id":"a","name":"A","passes":[{"block":"blur"}]}"#).unwrap();
        recipe.passes[0]
            .params
            .insert("radius".to_owned(), super::Curve::Constant(9.0));
        assert!(matches!(recipe.to_json(), Err(RecipeError::Range { .. })));
        recipe.id = "Not An Id".to_owned();
        assert_eq!(recipe.to_json(), Err(RecipeError::Name));
    }

    #[test]
    fn a_newer_recipe_asks_for_an_update_even_with_blocks_this_app_lacks() {
        assert_eq!(
            read(r#"{"version":3,"id":"a","name":"A","passes":[{"block":"bloom"}]}"#),
            Err(RecipeError::Newer(3))
        );
        assert!(matches!(
            read(r#"{"id":"a","name":"A","passes":[{"block":"blur"}]}"#),
            Err(RecipeError::Shape(_))
        ));
        assert!(matches!(
            read(r#"{"version":0,"id":"a","name":"A","passes":[{"block":"blur"}]}"#),
            Err(RecipeError::Shape(_))
        ));
    }

    #[test]
    fn every_bundled_control_changes_its_effect() {
        let times =
            [0.04, 0.1, 0.2, 0.5, 0.9, 1.4, 2.1].map(|seconds| Micros((seconds * 1e6) as i64));
        for recipe in bundled() {
            for control in &recipe.controls {
                let moved = if control.default < control.max {
                    control.max
                } else {
                    control.min
                };
                let set = [(control.id.clone(), moved)].into_iter().collect();
                let changes = times
                    .iter()
                    .any(|time| recipe.steps_with(*time, &set) != recipe.steps_at(*time));
                assert!(changes, "{} {} does nothing", recipe.id, control.id);
            }
        }
    }

    #[test]
    fn controls_speed_scale_and_set_the_steps_within_their_ranges() {
        let recipe = read(
            r#"{"version":1,"id":"a","name":"A","controls":[
                {"id":"speed","name":"Speed","default":1,"min":0.5,"max":3,"unit":"times","speed":true},
                {"id":"strength","name":"Strength","default":1,"min":0,"max":4,"scale":[[0,"amount"]]},
                {"id":"green","name":"Green","default":0.5,"min":0,"max":1,"set":[[0,"green"]]}],
                "passes":[{"block":"flash","params":{"amount":{"keys":[[0,0],[1,0.5]]}}}]}"#,
        )
        .unwrap();
        let at = |seconds: f64, controls: &[(&str, f32)]| {
            let controls = controls
                .iter()
                .map(|(id, value)| ((*id).to_owned(), *value))
                .collect();
            recipe.steps_with(Micros((seconds * 1_000_000.0) as i64), &controls)[0].values
        };
        // Defaults: the curve as written, green at its default.
        assert!((at(0.5, &[])[0] - 0.25).abs() < 1e-5);
        assert!((at(0.5, &[])[2] - 0.5).abs() < 1e-5);
        assert_eq!(
            recipe.steps_at(Micros(500_000)),
            recipe.steps_with(Micros(500_000), &recipe.default_controls())
        );
        // Twice as fast reaches the same value in half the time.
        assert!((at(0.25, &[("speed", 2.0)])[0] - 0.25).abs() < 1e-5);
        // A speed past its range is kept to it.
        assert_eq!(at(0.1, &[("speed", 100.0)]), at(0.1, &[("speed", 3.0)]));
        // Scaled, and kept to the parameter's own range.
        assert!((at(0.5, &[("strength", 2.0)])[0] - 0.5).abs() < 1e-5);
        assert!((at(1.0, &[("strength", 4.0)])[0] - 1.0).abs() < 1e-5);
        assert!((at(0.5, &[("green", 0.1)])[2] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn broken_controls_are_refused() {
        for control in [
            r#"{"id":"Speed","name":"S","default":1,"min":0.5,"max":2,"speed":true}"#,
            r#"{"id":"s","name":"S","default":3,"min":0.5,"max":2}"#,
            r#"{"id":"s","name":"S","default":1,"min":0,"max":2,"speed":true}"#,
            r#"{"id":"s","name":"S","default":1,"min":0,"max":2,"scale":[[1,"amount"]]}"#,
            r#"{"id":"s","name":"S","default":1,"min":0,"max":2,"set":[[0,"nothing"]]}"#,
            r#"{"id":"s","name":" ","default":1,"min":0,"max":2}"#,
        ] {
            let text = format!(
                r#"{{"version":1,"id":"a","name":"A","controls":[{control}],"passes":[{{"block":"flash"}}]}}"#
            );
            assert!(
                matches!(read(&text), Err(RecipeError::Control(_))),
                "{control}"
            );
        }
        let twice = r#"{"id":"s","name":"S","default":1,"min":0,"max":2}"#;
        let text = format!(
            r#"{{"version":1,"id":"a","name":"A","controls":[{twice},{twice}],"passes":[{{"block":"flash"}}]}}"#
        );
        assert!(matches!(read(&text), Err(RecipeError::Control(_))));
    }

    #[test]
    fn the_effect_id_carries_the_prefix() {
        let recipe =
            read(r#"{"version":1,"id":"zoom-pulse","name":"Zoom","passes":[{"block":"blur"}]}"#)
                .unwrap();
        assert_eq!(recipe.effect_id(), "recipe/zoom-pulse");
    }
}

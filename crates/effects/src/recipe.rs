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
use serde::Deserialize;

/// The recipe format this app reads. A recipe with a higher version needs
/// blocks or fields this app does not have.
pub const RECIPE_VERSION: u32 = 1;
/// The most passes one recipe may run.
pub const MAX_PASSES: usize = 8;
/// The most keyframes one curve may hold.
pub const MAX_KEYS: usize = 64;
/// The largest recipe file read.
pub const MAX_RECIPE_BYTES: usize = 64 * 1024;
/// The most values one block takes.
pub const MAX_VALUES: usize = 8;
/// The fastest a held random value may change, in steps per second.
pub const MAX_JITTER_RATE: f32 = 120.0;
/// The effect id a recipe is stored under on a draft's effect material.
pub const RECIPE_PREFIX: &str = "recipe/";

/// One building block the GPU runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize)]
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
/// side, 1 opens a circle from the middle, 2 opens `count` blinds. `angle`
/// 0 runs left to right; `soft` is the edge, as a fraction of the run.
const REVEAL: &[Param] = &[
    param("progress", 1.0, 0.0, 1.0),
    param("kind", 0.0, 0.0, 2.0),
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
const PRINTS: &[Param] = &[
    param("count", 0.0, 0.0, 4.0),
    param("scale", 0.7, 0.1, 1.5),
    param("spread", 0.5, 0.0, 1.0),
    param("turn", 0.0, -180.0, 180.0),
    param("tilt", 8.0, 0.0, 45.0),
    param("border", 0.025, 0.0, 0.1),
    param("blur", 0.03, 0.0, 0.1),
    param("wash", 0.5, 0.0, 1.0),
];
const HALFTONE: &[Param] = &[
    param("size", 0.0, 0.0, 0.1),
    param("amount", 1.0, 0.0, 1.0),
    param("color", 0.0, 0.0, 1.0),
];

impl Block {
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
        }
    }
}

/// A value over the effect's time, in seconds from the effect's start.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum Curve {
    Constant(f32),
    /// Straight lines between `[seconds, value]` keys, held before the first
    /// and after the last. With `repeat`, time wraps every that many seconds.
    Keys {
        keys: Vec<[f32; 2]>,
        #[serde(default)]
        repeat: Option<f32>,
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

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Wave {
    pub period: f32,
    pub from: f32,
    pub to: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Jitter {
    pub rate: f32,
    pub from: f32,
    pub to: f32,
    #[serde(default)]
    pub seed: u32,
}

/// One pass of a recipe as written.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pass {
    pub block: Block,
    #[serde(default)]
    pub params: BTreeMap<String, Curve>,
}

/// An effect as data.
#[derive(Clone, Debug, PartialEq, Deserialize)]
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
    #[serde(default)]
    pub category: String,
    /// The scene its card shows it over, one of
    /// [`crate::preview_scenes`]; empty is [`crate::DEFAULT_PREVIEW_SCENE`].
    #[serde(default)]
    pub preview: String,
    /// The second of the effect its card's still is taken from; none is
    /// [`crate::DEFAULT_POSTER`]. A flash names the moment it is brightest.
    #[serde(default)]
    pub poster: Option<f32>,
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
        recipe.check()?;
        Ok(recipe)
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
        for pass in &self.passes {
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
        // Seconds as f64 first: a long timeline keeps its fraction.
        let seconds = (time.0.max(0) as f64 / 1_000_000.0) as f32;
        self.passes
            .iter()
            .map(|pass| {
                let mut values = [0.0; MAX_VALUES];
                for (slot, param) in values.iter_mut().zip(pass.block.params()) {
                    *slot = pass
                        .params
                        .get(param.name)
                        .map_or(param.default, |curve| curve.at(seconds));
                }
                Step {
                    block: pass.block,
                    values,
                }
            })
            .collect()
    }
}

fn check_curve(block: Block, param: &Param, curve: &Curve) -> Result<(), RecipeError> {
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
        Curve::Keys { keys, repeat } => {
            if keys.is_empty() || keys.len() > MAX_KEYS {
                return Err(broken());
            }
            if repeat.is_some_and(|repeat| !positive(repeat)) {
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

impl Curve {
    /// The value at `seconds`, never negative.
    fn at(&self, seconds: f32) -> f32 {
        match self {
            Self::Constant(value) => *value,
            Self::Keys { keys, repeat } => {
                let time = repeat.map_or(seconds, |repeat| seconds.rem_euclid(repeat));
                along(keys, time)
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

fn along(keys: &[[f32; 2]], time: f32) -> f32 {
    let Some(first) = keys.first() else {
        return 0.0;
    };
    if time <= first[0] {
        return first[1];
    }
    for pair in keys.windows(2) {
        let ([t0, v0], [t1, v1]) = (pair[0], pair[1]);
        if time <= t1 {
            if t1 <= t0 {
                return v1;
            }
            return v0 + (v1 - v0) * (time - t0) / (t1 - t0);
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

const BUNDLED_FILES: [&str; 103] = [
    include_str!("../recipes/vignette.json"),
    include_str!("../recipes/black-white.json"),
    include_str!("../recipes/vivid.json"),
    include_str!("../recipes/fade-in.json"),
    include_str!("../recipes/invert.json"),
    include_str!("../recipes/duotone.json"),
    include_str!("../recipes/golden-duotone.json"),
    include_str!("../recipes/ocean-duotone.json"),
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
    include_str!("../recipes/iris-in.json"),
    include_str!("../recipes/wipe-in.json"),
    include_str!("../recipes/diagonal-wipe.json"),
    include_str!("../recipes/blinds.json"),
    include_str!("../recipes/blur-in.json"),
    include_str!("../recipes/blur.json"),
    include_str!("../recipes/rack-focus.json"),
    include_str!("../recipes/dreamy.json"),
    include_str!("../recipes/zoom-blur.json"),
    include_str!("../recipes/zoom-burst.json"),
    include_str!("../recipes/motion-blur.json"),
    include_str!("../recipes/speed-lines.json"),
    include_str!("../recipes/rgb-split.json"),
    include_str!("../recipes/glitch.json"),
    include_str!("../recipes/digital-glitch.json"),
    include_str!("../recipes/signal-loss.json"),
    include_str!("../recipes/rgb-shift.json"),
    include_str!("../recipes/invert-flicker.json"),
    include_str!("../recipes/stutter.json"),
    include_str!("../recipes/neon-edges.json"),
    include_str!("../recipes/color-glitch.json"),
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

    #[test]
    fn a_newer_recipe_asks_for_an_update_even_with_blocks_this_app_lacks() {
        assert_eq!(
            read(r#"{"version":2,"id":"a","name":"A","passes":[{"block":"bloom"}]}"#),
            Err(RecipeError::Newer(2))
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
    fn the_effect_id_carries_the_prefix() {
        let recipe =
            read(r#"{"version":1,"id":"zoom-pulse","name":"Zoom","passes":[{"block":"blur"}]}"#)
                .unwrap();
        assert_eq!(recipe.effect_id(), "recipe/zoom-pulse");
    }
}

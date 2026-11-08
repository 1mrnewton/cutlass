//! Transitions as data: what happens to the outgoing picture as it leaves,
//! what happens to the incoming one as it arrives, and how the two are
//! swapped.
//!
//! A transition recipe is JSON, as an effect recipe is. `outgoing` and
//! `incoming` are passes of the same GPU blocks effects use, worked out at
//! the transition's progress `t`, 0 at the start of the window and 1 at its
//! end, in place of seconds: a curve's key at 0.5 is the middle of the
//! transition however long it is. `cross` decides, pixel by pixel, which of
//! the two changed pictures shows: a wipe, an iris, a shape opening, a
//! fade, by brightness, or one picture laid over the other.
//!
//! The GPU draws them. The CPU path, which a frame takes for a few kinds of
//! clip, draws a recipe transition as a dissolve.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use cutlass_model::Micros;
use serde::Deserialize;

use crate::recipe::{Curve, MAX_PASSES, Param, Pass, Recipe, RecipeError, Step, check_passes};

/// The transitions shelf's categories, in the order the shelf lists them.
pub const TRANSITION_CATEGORIES: &[&str] = &[
    "Basic", "Wipe", "Slide", "Zoom", "Spin", "Blur", "Light", "Glitch", "Shape", "Distort",
];

/// The length a recipe that names none is dropped with, in seconds.
pub const DEFAULT_CROSSING_SECONDS: f32 = 0.5;

/// How `cross` picks between the two pictures.
///
/// For the shapes, every pixel has a place along the swap, 0 to 1, and the
/// incoming picture shows where that place is behind `progress`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossKind {
    /// From one side; `angle` 0 runs left to right.
    Wipe,
    /// A circle out from `x`, `y`.
    Circle,
    /// `count` blinds along `angle`.
    Blinds,
    /// Two doors out from the middle line.
    Doors,
    /// A clock hand round from `angle`, clockwise from the top.
    Clock,
    /// A diamond out from `x`, `y`.
    Diamond,
    /// A checkerboard of `count` squares across, one colour then the other.
    Checker,
    /// Fine grains in a random order.
    Grains,
    /// The whole picture at once.
    Fade,
    /// The outgoing picture's dark parts first, its bright parts last.
    Luma,
    /// The incoming picture laid over the outgoing one, by its own
    /// clearness; `progress` plays no part.
    Over,
    /// The outgoing picture laid over the incoming one.
    Under,
    /// A five-pointed star out from `x`, `y`.
    Star,
    /// A hexagon out from `x`, `y`.
    Hexagon,
    /// A heart out from `x`, `y`.
    Heart,
    /// `count` squares across, each at its own random moment.
    Squares,
    /// `count` stripes along `angle`, each wiping the other way from the
    /// last.
    Zip,
}

impl CrossKind {
    /// Every kind, in the order of its number.
    pub const ALL: [Self; 17] = [
        Self::Wipe,
        Self::Circle,
        Self::Blinds,
        Self::Doors,
        Self::Clock,
        Self::Diamond,
        Self::Checker,
        Self::Grains,
        Self::Fade,
        Self::Luma,
        Self::Over,
        Self::Under,
        Self::Star,
        Self::Hexagon,
        Self::Heart,
        Self::Squares,
        Self::Zip,
    ];
}

const fn param(name: &'static str, default: f32, min: f32, max: f32) -> Param {
    Param {
        name,
        default,
        min,
        max,
    }
}

/// `cross`'s parameters, in the order a [`Crossing`] carries them. A
/// recipe that leaves `progress` out swaps at the transition's own pace:
/// `progress` is `t`. `edge` lays a band of the `red`, `green`, `blue`
/// colour along the swapping line, as wide as that part of the run.
pub const CROSS: &[Param] = &[
    param("progress", 0.0, 0.0, 1.0),
    param("kind", 8.0, 0.0, 16.0),
    param("angle", 0.0, -360.0, 360.0),
    param("soft", 0.02, 0.0, 1.0),
    param("count", 8.0, 1.0, 64.0),
    param("x", 0.0, -0.5, 0.5),
    param("y", 0.0, -0.5, 0.5),
    param("edge", 0.0, 0.0, 0.3),
    param("red", 1.0, 0.0, 1.0),
    param("green", 1.0, 0.0, 1.0),
    param("blue", 1.0, 0.0, 1.0),
];

/// A transition as data.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransitionRecipe {
    pub version: u32,
    /// Lowercase letters, digits and dashes; never a built-in key.
    pub id: String,
    pub name: String,
    /// One of [`TRANSITION_CATEGORIES`].
    #[serde(default)]
    pub category: String,
    /// The two scenes its card plays between; none is the default pair.
    #[serde(default)]
    pub preview: Vec<String>,
    /// Seconds it is dropped with; none is [`DEFAULT_CROSSING_SECONDS`].
    #[serde(default)]
    pub duration: Option<f32>,
    #[serde(default)]
    pub outgoing: Vec<Pass>,
    #[serde(default)]
    pub incoming: Vec<Pass>,
    #[serde(default)]
    pub cross: BTreeMap<String, Curve>,
}

/// One moment of a recipe transition: the steps for each picture, and the
/// values of `cross` in the order of [`CROSS`].
#[derive(Clone, Debug, PartialEq)]
pub struct Crossing {
    pub outgoing: Vec<Step>,
    pub incoming: Vec<Step>,
    pub cross: [f32; 12],
}

impl Crossing {
    /// What `cross` does.
    #[must_use]
    pub fn kind(&self) -> CrossKind {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let index = self.cross[1].round().max(0.0) as usize;
        CrossKind::ALL
            .get(index)
            .copied()
            .unwrap_or(CrossKind::Fade)
    }
}

impl TransitionRecipe {
    /// Read and check a transition recipe.
    pub fn read(bytes: &[u8]) -> Result<Self, RecipeError> {
        if bytes.len() > crate::recipe::MAX_RECIPE_BYTES {
            return Err(RecipeError::TooLarge);
        }
        #[derive(Deserialize)]
        struct Version {
            version: u32,
        }
        let Version { version } =
            serde_json::from_slice(bytes).map_err(|error| RecipeError::Shape(error.to_string()))?;
        if version > crate::RECIPE_VERSION {
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
        let scenes_ok = self.preview.is_empty()
            || (self.preview.len() == 2
                && self
                    .preview
                    .iter()
                    .all(|scene| crate::preview_scene(scene).is_some()));
        if !scenes_ok {
            return Err(RecipeError::Shape(
                "a preview names two scenes the app ships".to_owned(),
            ));
        }
        if self
            .duration
            .is_some_and(|seconds| !seconds.is_finite() || !(0.1..=5.0).contains(&seconds))
        {
            return Err(RecipeError::Shape("a duration from 0.1 to 5 s".to_owned()));
        }
        if self.outgoing.len() > MAX_PASSES || self.incoming.len() > MAX_PASSES {
            return Err(RecipeError::Passes);
        }
        check_passes(&self.outgoing)?;
        check_passes(&self.incoming)?;
        for (name, curve) in &self.cross {
            let Some(param) = CROSS.iter().find(|param| param.name == name) else {
                return Err(RecipeError::Shape(format!("cross has no {name}")));
            };
            crate::recipe::check_curve(crate::Block::Reveal, param, curve)?;
        }
        Ok(())
    }

    /// The `effect_id` a material stores for it.
    #[must_use]
    pub fn effect_id(&self) -> String {
        format!("{}{}", crate::BUILT_IN_PREFIX, self.id)
    }

    /// The length it is dropped with.
    #[must_use]
    pub fn duration(&self) -> Micros {
        let seconds = self.duration.unwrap_or(DEFAULT_CROSSING_SECONDS);
        #[allow(clippy::cast_possible_truncation)]
        let micros = (f64::from(seconds) * 1_000_000.0).round() as i64;
        Micros(micros)
    }

    /// Moment `t`, 0 to 1, of the transition.
    #[must_use]
    pub fn at(&self, t: f32) -> Crossing {
        let t = if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.0
        };
        #[allow(clippy::cast_possible_truncation)]
        let time = Micros((f64::from(t) * 1_000_000.0).round() as i64);
        let steps = |passes: &[Pass]| {
            if passes.is_empty() {
                return Vec::new();
            }
            Recipe {
                version: self.version,
                id: self.id.clone(),
                name: self.name.clone(),
                category: String::new(),
                preview: String::new(),
                poster: None,
                controls: Vec::new(),
                passes: passes.to_vec(),
            }
            .steps_at(time)
        };
        let mut cross = [0.0; 12];
        for (slot, param) in cross.iter_mut().zip(CROSS) {
            *slot = self.cross.get(param.name).map_or(
                if param.name == "progress" {
                    t
                } else {
                    param.default
                },
                |curve| curve.at(t),
            );
            *slot = slot.clamp(param.min, param.max);
        }
        Crossing {
            outgoing: steps(&self.outgoing),
            incoming: steps(&self.incoming),
            cross,
        }
    }
}

/// The scene pairs a card plays between when its recipe names none, spread
/// over the shelf so neighbours differ.
pub const PREVIEW_PAIRS: [(&str, &str); 8] = [
    ("portrait", "beach"),
    ("city", "lake"),
    ("skater", "flowers"),
    ("cafe", "road"),
    ("concert", "dancer"),
    ("lake", "portrait"),
    ("beach", "city"),
    ("flowers", "skater"),
];

/// The two scenes `kind`'s card plays between: its recipe's own, or a pair
/// picked by its key.
#[must_use]
pub fn transition_scenes(kind: crate::TransitionKind) -> (&'static str, &'static str) {
    if let Some([from, to]) = kind.recipe().map(|recipe| recipe.preview.as_slice())
        && let (Some(from), Some(to)) = (scene_name(from), scene_name(to))
    {
        return (from, to);
    }
    let sum = kind.key().bytes().fold(0usize, |sum, byte| {
        sum.wrapping_mul(31).wrapping_add(usize::from(byte))
    });
    PREVIEW_PAIRS[sum % PREVIEW_PAIRS.len()]
}

fn scene_name(name: &str) -> Option<&'static str> {
    crate::preview_scenes()
        .into_iter()
        .find(|scene| *scene == name)
}

/// How long a playing transition card takes before it starts again: the
/// outgoing scene held, the transition, then the incoming scene held.
pub const TRANSITION_PREVIEW_LOOP: Micros = Micros(1_600_000);
const PREVIEW_HOLD: f32 = 0.3;

/// The transition's `t` a card playing for `elapsed` shows, stepped at
/// [`crate::PREVIEW_RATE`].
#[must_use]
pub fn transition_preview_t(elapsed: Micros) -> f32 {
    let step = 1_000_000 / i64::from(crate::PREVIEW_RATE);
    let looped = elapsed.0.max(0) % TRANSITION_PREVIEW_LOOP.0;
    #[allow(clippy::cast_precision_loss)]
    let seconds = (looped - looped % step) as f32 / 1_000_000.0;
    #[allow(clippy::cast_precision_loss)]
    let length = TRANSITION_PREVIEW_LOOP.0 as f32 / 1_000_000.0;
    ((seconds - PREVIEW_HOLD) / (length - 2.0 * PREVIEW_HOLD)).clamp(0.0, 1.0)
}

/// The transition recipes the app ships, in shelf order.
const BUNDLED_FILES: [&str; 101] = [
    include_str!("../transitions/flash.json"),
    include_str!("../transitions/quick-flash.json"),
    include_str!("../transitions/red-flash.json"),
    include_str!("../transitions/luma-fade.json"),
    include_str!("../transitions/grain-dissolve.json"),
    include_str!("../transitions/pixel-dissolve.json"),
    include_str!("../transitions/overexpose.json"),
    include_str!("../transitions/underexpose.json"),
    include_str!("../transitions/mono-fade.json"),
    include_str!("../transitions/tint-fade.json"),
    include_str!("../transitions/slow-fade.json"),
    include_str!("../transitions/film-fade.json"),
    include_str!("../transitions/burn-fade.json"),
    include_str!("../transitions/soft-wipe-left.json"),
    include_str!("../transitions/soft-wipe-right.json"),
    include_str!("../transitions/wipe-up-right.json"),
    include_str!("../transitions/wipe-up-left.json"),
    include_str!("../transitions/wipe-down-left.json"),
    include_str!("../transitions/wipe-down-right.json"),
    include_str!("../transitions/split-across.json"),
    include_str!("../transitions/split-down.json"),
    include_str!("../transitions/clock-wipe.json"),
    include_str!("../transitions/clock-from-below.json"),
    include_str!("../transitions/blinds-across.json"),
    include_str!("../transitions/blinds-down.json"),
    include_str!("../transitions/zip-across.json"),
    include_str!("../transitions/zip-down.json"),
    include_str!("../transitions/checker-wipe.json"),
    include_str!("../transitions/line-wipe.json"),
    include_str!("../transitions/wipe-slide.json"),
    include_str!("../transitions/wipe-blur.json"),
    include_str!("../transitions/band-wipe.json"),
    include_str!("../transitions/venetian.json"),
    include_str!("../transitions/reveal-left.json"),
    include_str!("../transitions/reveal-right.json"),
    include_str!("../transitions/reveal-up.json"),
    include_str!("../transitions/reveal-down.json"),
    include_str!("../transitions/card-stack.json"),
    include_str!("../transitions/swipe-left.json"),
    include_str!("../transitions/swipe-right.json"),
    include_str!("../transitions/swipe-up.json"),
    include_str!("../transitions/swipe-down.json"),
    include_str!("../transitions/bounce-in.json"),
    include_str!("../transitions/slide-fade.json"),
    include_str!("../transitions/elastic-push.json"),
    include_str!("../transitions/drop-down.json"),
    include_str!("../transitions/rise-up.json"),
    include_str!("../transitions/diagonal-slide.json"),
    include_str!("../transitions/diagonal-slide-down.json"),
    include_str!("../transitions/zoom-blur-in.json"),
    include_str!("../transitions/shrink-away.json"),
    include_str!("../transitions/zoom-through.json"),
    include_str!("../transitions/punch-in.json"),
    include_str!("../transitions/crash-zoom.json"),
    include_str!("../transitions/zoom-rotate-in.json"),
    include_str!("../transitions/iris-zoom.json"),
    include_str!("../transitions/grow-in.json"),
    include_str!("../transitions/zoom-fade.json"),
    include_str!("../transitions/zoom-pulse.json"),
    include_str!("../transitions/lens-zoom.json"),
    include_str!("../transitions/warp-zoom.json"),
    include_str!("../transitions/zoom-left.json"),
    include_str!("../transitions/zoom-right.json"),
    include_str!("../transitions/dolly-fade.json"),
    include_str!("../transitions/snap-zoom.json"),
    include_str!("../transitions/pop-in.json"),
    include_str!("../transitions/pop-out.json"),
    include_str!("../transitions/tunnel-zoom.json"),
    include_str!("../transitions/zoom-out-fade.json"),
    include_str!("../transitions/spin-in.json"),
    include_str!("../transitions/spin-out.json"),
    include_str!("../transitions/spin-left.json"),
    include_str!("../transitions/spin-right.json"),
    include_str!("../transitions/roll-left.json"),
    include_str!("../transitions/roll-right.json"),
    include_str!("../transitions/twist.json"),
    include_str!("../transitions/tornado.json"),
    include_str!("../transitions/vortex.json"),
    include_str!("../transitions/spin-zoom.json"),
    include_str!("../transitions/tilt-in.json"),
    include_str!("../transitions/swing-in.json"),
    include_str!("../transitions/flip-horizontal.json"),
    include_str!("../transitions/flip-vertical.json"),
    include_str!("../transitions/barrel-roll.json"),
    include_str!("../transitions/spiral-in.json"),
    include_str!("../transitions/coin-flip.json"),
    include_str!("../transitions/soft-blur.json"),
    include_str!("../transitions/heavy-blur.json"),
    include_str!("../transitions/motion-blur-left.json"),
    include_str!("../transitions/motion-blur-right.json"),
    include_str!("../transitions/motion-blur-up.json"),
    include_str!("../transitions/motion-blur-down.json"),
    include_str!("../transitions/focus-pull.json"),
    include_str!("../transitions/tilt-shift.json"),
    include_str!("../transitions/lens-blur.json"),
    include_str!("../transitions/radial-blur.json"),
    include_str!("../transitions/blur-circle.json"),
    include_str!("../transitions/dream-fade.json"),
    include_str!("../transitions/frost-fade.json"),
    include_str!("../transitions/blur-dip.json"),
    include_str!("../transitions/defocus-zoom.json"),
];

/// Every bundled transition recipe, read once.
#[must_use]
pub fn bundled_transitions() -> &'static [TransitionRecipe] {
    static RECIPES: OnceLock<Vec<TransitionRecipe>> = OnceLock::new();
    RECIPES.get_or_init(|| {
        BUNDLED_FILES
            .iter()
            .filter_map(|file| TransitionRecipe::read(file.as_bytes()).ok())
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::{BUNDLED_FILES, CROSS, CrossKind, TRANSITION_CATEGORIES, TransitionRecipe};
    use crate::{Block, TransitionKind};

    fn read(text: &str) -> TransitionRecipe {
        TransitionRecipe::read(text.as_bytes()).expect("reads")
    }

    #[test]
    fn every_bundled_transition_reads_and_names_its_category() {
        let mut ids = std::collections::HashSet::new();
        for file in BUNDLED_FILES {
            let recipe = TransitionRecipe::read(file.as_bytes())
                .unwrap_or_else(|error| panic!("{error:?} in {}", &file[..file.len().min(80)]));
            assert!(
                TRANSITION_CATEGORIES.contains(&recipe.category.as_str()),
                "{} has category {}",
                recipe.id,
                recipe.category
            );
            assert!(ids.insert(recipe.id.clone()), "{} twice", recipe.id);
            assert!(
                TransitionKind::ALL
                    .iter()
                    .all(|kind| kind.key() != recipe.id),
                "{} is a built-in key",
                recipe.id
            );
        }
        assert_eq!(super::bundled_transitions().len(), BUNDLED_FILES.len());
    }

    #[test]
    fn progress_follows_t_unless_a_recipe_curves_it() {
        let plain = read(r#"{"version":1,"id":"a","name":"A","cross":{"kind":0}}"#);
        let half = plain.at(0.5);
        assert_eq!(half.cross[0], 0.5);
        assert_eq!(half.kind(), CrossKind::Wipe);
        assert!(half.outgoing.is_empty() && half.incoming.is_empty());
        let late = read(
            r#"{"version":1,"id":"a","name":"A","cross":{"progress":{"keys":[[0.5,0],[1,1]]}},
            "outgoing":[{"block":"blur","params":{"radius":{"keys":[[0,0],[1,0.1]]}}}]}"#,
        );
        let at = late.at(0.5);
        assert_eq!(at.cross[0], 0.0);
        assert_eq!(at.kind(), CrossKind::Fade);
        assert_eq!(at.outgoing[0].block, Block::Blur);
        assert!((at.outgoing[0].values[0] - 0.05).abs() < 1e-6);
        assert_eq!(late.at(f32::NAN).cross[0], 0.0);
        assert_eq!(CROSS.len(), 11);
    }

    #[test]
    fn broken_transitions_are_refused() {
        for text in [
            r#"{"version":1,"id":"A","name":"A"}"#,
            r#"{"version":1,"id":"a","name":"A","cross":{"wobble":1}}"#,
            r#"{"version":1,"id":"a","name":"A","cross":{"kind":40}}"#,
            r#"{"version":1,"id":"a","name":"A","duration":9}"#,
            r#"{"version":1,"id":"a","name":"A","preview":["portrait"]}"#,
            r#"{"version":1,"id":"a","name":"A","outgoing":[{"block":"blur","params":{"size":1}}]}"#,
        ] {
            assert!(TransitionRecipe::read(text.as_bytes()).is_err(), "{text}");
        }
    }
}

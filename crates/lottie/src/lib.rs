//! Lottie animations: parse a Bodymovin JSON file, evaluate one frame as
//! flat vector geometry, and draw it on the CPU or hand it to the GPU.
//!
//! [`Animation::parse`] reads the file and refuses one past the limits in
//! [`limits`]. [`Animation::scene`] evaluates a frame into a [`Scene`]:
//! fills in draw order, each already transformed into output pixels and cut
//! into line edges, with groups for layer opacity, masks, mattes and
//! precomposition bounds. [`rasterize`] draws a scene into premultiplied
//! RGBA bytes. [`pack`] lays the same scene out for [`WGSL`], which draws it
//! with the same samples and arithmetic; `cutlass-render` runs that shader.
//!
//! Drawn: shape, solid, null and precomposition layers; parenting; anchor,
//! position (split or along its spatial curve), scale, rotation, skew and
//! opacity; linear, eased and hold keys; paths, rectangles (rounded),
//! ellipses, stars and polygons (rounded); fills (non-zero and even-odd),
//! strokes (width, caps, joins, miter limit, dashes); linear and radial
//! gradients with a highlight; groups; trims (each path, or all in turn);
//! repeaters; masks (add, subtract, intersect, lighten, darken, difference,
//! inverted, opacity, expansion); alpha and luma mattes, inverted or not;
//! precomposition time remap, stretch, start and bounds; in and out points.
//!
//! Not drawn: image and text layers, effects, layer blend modes, merge
//! paths, rounded corners, twist, offset paths, expressions, 3D and
//! auto-orient.

#![forbid(unsafe_code)]

mod anim;
mod eval;
mod geom;
pub mod limits;
mod pack;
mod prop;
mod raster;
mod scene;
mod shape;
mod stroke;
mod transform;

use std::fmt;

pub use anim::Animation;
pub use pack::{DATA_WIDTH, Packed, PackedItem, pack};
pub use raster::{rasterize, shade};
pub use scene::{
    Bounds, Edge, Expansion, Fill, FillRule, Gradient, GradientKind, Group, Item, Mask, MaskMode,
    Matte, MatteMode, Paint, Scene, Stop,
};

/// The shader that draws a [`Packed`] scene: `vs` with `fs_fill` for
/// fills, `vs` with `fs_group` for groups.
pub const WGSL: &str = include_str!("vector.wgsl");

/// Why a file could not be read or a frame not drawn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LottieError {
    /// The bytes are not JSON.
    Json(String),
    /// The JSON is not an animation this crate can play.
    Invalid(&'static str),
    /// The file or the frame is past a limit in [`limits`].
    Limit(&'static str),
}

impl fmt::Display for LottieError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(reason) => write!(formatter, "the animation is not JSON: {reason}"),
            Self::Invalid(reason) => write!(formatter, "not a Lottie animation: {reason}"),
            Self::Limit(what) => write!(formatter, "the animation has too many {what}"),
        }
    }
}

impl std::error::Error for LottieError {}

/// Whether `path` names a Lottie file: its extension is `json`.
#[must_use]
pub fn is_lottie_path(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
}

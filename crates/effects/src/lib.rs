//! Effect packages and the pictures they sample.
//!
//! One runtime loads a package by the id on the effect material and samples
//! it through [`cutlass_picture::PictureSource`]. Blend modes are packages in
//! [`catalog`], not a shader-host enum. The compositor is not in this crate.
//!
//! [`recipe`] holds effects as data: passes over blocks the GPU runs, with
//! values that move over time. Ten recipes ship in the app.
//!
//! There is no script host and no shader dialect. A checksum proves that
//! published bytes arrived intact. It does not authorize running those bytes.
//! The software stand-in is the picture a test draws. Shaders and the rest of
//! a catalog of about 200 packages are unsupported. The supported rows are
//! listed on [`catalog`].

#![forbid(unsafe_code)]

pub mod catalog;
pub mod package;
pub mod recipe;
pub mod source;

pub use catalog::{
    ADD, ANIM_IN, ANIM_LOOP, ANIM_OUT, BlendMode, CONTRAST, CatalogEntry, DIFFERENCE, Distort,
    INVERT, MULTIPLY, PIXELATE, PIXELATE_BLOCK, PackageKind, SCREEN, SOURCE_OVER, TRANSITION_FADE,
    Transition, entries, entry,
};
pub use package::{
    Located, MAX_PACKAGE_BYTES, MissingPackages, Package, PackageError, PackageStore,
    animation_package_id, load, load_segment, load_transition, map_storage_error, package_id,
    plugin_package_id, segment_package_id, transition_package_id, video_package_id,
};
pub use recipe::{
    Block, Curve, MAX_PASSES, MAX_VALUES, Param, Pass, RECIPE_PREFIX, RECIPE_VERSION, Recipe,
    RecipeError, Step, bundled, bundled_recipe,
};
pub use source::{AnimSpan, EffectSource, Progress, progress};

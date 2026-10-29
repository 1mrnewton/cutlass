//! Effect package ids, still posters, and preview URLs.
//!
//! Blend modes, distortions, and text animations are packages in this
//! catalog, not a shader-host enum. There is no script host and no shader
//! dialect. This crate does not start a transfer. Shaders and the rest of a
//! catalog of about 200 packages are unsupported.

#![forbid(unsafe_code)]

mod catalog;

pub use catalog::{
    ADD, ANIM_IN, ANIM_LOOP, ANIM_OUT, BlendMode, CONTRAST, CatalogEntry, DIFFERENCE, Distort,
    INVERT, MULTIPLY, PIXELATE, PIXELATE_BLOCK, PackageKind, SCREEN, SOURCE_OVER, entries, entry,
};

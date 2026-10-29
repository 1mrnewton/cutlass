//! Effect packages and the ids a draft stores.
//!
//! One runtime loads a package by the id on the effect material. Blend modes
//! are packages in [`catalog`], not a shader-host enum. A missing package is
//! a named error. The draft is not deleted or rewritten.
//!
//! There is no script host and no shader dialect. A checksum proves that
//! published bytes arrived intact. It does not authorize running those bytes.
//! Shaders and the rest of a catalog of about 200 packages are unsupported.
//! This crate does not start a transfer.

#![forbid(unsafe_code)]

mod catalog;
mod package;

pub use catalog::{
    ADD, ANIM_IN, ANIM_LOOP, ANIM_OUT, BlendMode, CONTRAST, CatalogEntry, DIFFERENCE, Distort,
    INVERT, MULTIPLY, PIXELATE, PIXELATE_BLOCK, PackageKind, SCREEN, SOURCE_OVER, entries, entry,
};
pub use package::{
    Located, MAX_PACKAGE_BYTES, Package, PackageError, PackageStore, animation_package_id, load,
    load_segment, map_storage_error, package_id, plugin_package_id, segment_package_id,
    video_package_id,
};

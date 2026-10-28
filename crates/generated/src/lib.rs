//! Solid frames.
//!
//! The host path fills each pixel from a packed `0xAARRGGBB` color. The bytes
//! are RGBA: bits 16–23, 8–15, 0–7, then 24–31. A gradient, a contour, and a
//! template are not produced here. An edge past the device limit is refused.

#![forbid(unsafe_code)]

mod color;
mod source;

pub use color::{FillError, fill, parse_packed, rgba_from_packed};
pub use cutlass_picture::PictureSource;
pub use source::SolidColor;

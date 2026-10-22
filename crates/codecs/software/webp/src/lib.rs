//! WebP decoder.
//!
//! The crate takes the file bytes and returns one still. It does no I/O.
//! `docs/codecs.md` lists the picture.

#![deny(unsafe_code)]

mod bits;
mod decode;
mod error;
mod huff;
mod lossless;

/// Each edge stops here. `docs/codecs.md`.
pub(crate) const MAX_EDGE: u32 = 32_768;
/// One pixel buffer, or the prefix-code tables together, stop here.
pub(crate) const MAX_BYTES: usize = 256 * 1024 * 1024;

pub use decode::decode;
pub use error::Error;

//! JPEG decoder.
//!
//! The crate takes the file bytes and returns one still. It does no I/O.
//! `docs/codecs.md` lists the picture.

#![deny(unsafe_code)]

mod bits;
mod color;
mod decode;
mod error;
mod huff;
mod idct;

pub use decode::decode;
pub use error::Error;

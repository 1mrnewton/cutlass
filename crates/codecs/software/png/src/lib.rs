//! PNG decoder.
//!
//! The crate takes the file bytes and returns one still. It does no I/O.
//! `docs/codecs.md` lists the picture.

#![deny(unsafe_code)]

mod crc;
mod decode;
mod error;
mod expand;
mod image;

pub use decode::decode;
pub use error::{ChunkKind, Error};

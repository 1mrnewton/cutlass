//! Ogg reader.
//!
//! The crate takes the file bytes and returns Vorbis and Opus tracks. It does
//! no I/O. `docs/codecs.md` lists the containers.

#![deny(unsafe_code)]

mod error;
mod ogg;
mod page;

pub use error::Error;
pub use ogg::{Movie, Track};

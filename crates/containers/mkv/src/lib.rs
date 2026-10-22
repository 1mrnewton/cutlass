//! MKV and WebM reader.
//!
//! The crate takes the file bytes and returns tracks. It does no I/O. Sample
//! bytes are not read yet. `docs/codecs.md` lists the containers.

#![deny(unsafe_code)]

mod ebml;
mod error;
mod movie;

pub use error::Error;
pub use movie::{DocType, Kind, Movie, Track};

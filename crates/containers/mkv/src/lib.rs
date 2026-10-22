//! MKV and WebM reader.
//!
//! The crate takes the file bytes and returns tracks and samples. It does no
//! I/O. `docs/codecs.md` lists the containers.

#![deny(unsafe_code)]

mod block;
mod ebml;
mod error;
mod movie;

pub use error::Error;
pub use movie::{DocType, Kind, Movie, Track};

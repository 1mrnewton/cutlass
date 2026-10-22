//! Native FLAC reader.
//!
//! The crate takes the file bytes and returns one track and its samples. It
//! does no I/O. `docs/codecs.md` lists the containers.

#![deny(unsafe_code)]

mod error;
mod flac;

pub use error::Error;
pub use flac::{Movie, Track};

//! AVI reader.
//!
//! The crate takes the file bytes and returns tracks and samples. It does no
//! I/O. `docs/codecs.md` lists the containers.

#![deny(unsafe_code)]

mod error;
mod movie;
mod riff;

pub use error::Error;
pub use movie::{Kind, Movie, Track};

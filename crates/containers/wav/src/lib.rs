//! WAV reader.
//!
//! The crate takes the file bytes and returns one track and its samples. It
//! does no I/O. `docs/codecs.md` lists the containers.

#![deny(unsafe_code)]

mod error;
mod riff;
mod wave;

pub use error::Error;
pub use wave::{Kind, Movie, Track};

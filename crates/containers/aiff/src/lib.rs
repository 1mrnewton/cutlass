//! AIFF reader.
//!
//! The crate takes the file bytes and returns one track and its samples. It
//! does no I/O. `docs/codecs.md` lists the containers.

#![deny(unsafe_code)]

mod aiff;
mod error;
mod iff;

pub use aiff::{Kind, Movie, Track};
pub use error::Error;

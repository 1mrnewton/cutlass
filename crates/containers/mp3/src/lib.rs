//! MP3 reader.
//!
//! The crate takes the file bytes and returns one track and its frames. It
//! does no I/O. `docs/codecs.md` lists the containers.

#![deny(unsafe_code)]

mod error;
mod frame;
mod mpeg;

pub use error::Error;
pub use mpeg::{Movie, Track};

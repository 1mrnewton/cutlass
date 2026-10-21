//! MP4 and MOV reader.
//!
//! The crate takes the file bytes and returns tracks and samples. It does no
//! I/O. Sample timestamps are media time. An edit list is kept on the track
//! and is not applied. `docs/codecs.md` lists the containers.

#![deny(unsafe_code)]

mod entry;
mod error;
mod fourcc;
mod movie;
mod read;
mod tables;

pub use cutlass_core::AudioCodec;
pub use entry::Audio;
pub use error::Error;
pub use fourcc::Fourcc;
pub use movie::{Edit, Kind, Movie, Rotation, Track};

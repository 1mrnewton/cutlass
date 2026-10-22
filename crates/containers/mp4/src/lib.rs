//! MP4 and MOV reader and writer.
//!
//! The crate takes the file bytes and returns tracks and samples. It does no
//! I/O. Sample timestamps are media time when the track has no edit list. An
//! edit list moves them onto the movie timeline. It also writes an MP4 file
//! from H.264, HEVC, and AAC samples, and a QuickTime file from ProRes
//! samples. The index follows the samples, or it comes first. `docs/codecs.md`
//! lists the containers.

#![deny(unsafe_code)]

mod entry;
mod error;
mod fourcc;
mod movie;
mod read;
mod tables;
mod write;

pub use cutlass_core::{AudioCodec, Rotation};
pub use entry::Audio;
pub use error::Error;
pub use fourcc::Fourcc;
pub use movie::{Edit, Kind, Movie, Track};
pub use write::{MovieFile, ProRes};

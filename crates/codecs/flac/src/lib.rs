//! FLAC, the Free Lossless Audio Codec (RFC 9639).
//!
//! One packet is one FLAC frame, as the native `.flac` container, Ogg FLAC,
//! and MP4 `fLaC` sample entries all frame it. This crate does not read the
//! `fLaC` marker or metadata blocks; that is `cutlass-flac-container`.

#![forbid(unsafe_code)]

mod bits;
pub mod crc;
mod decode;
mod header;
pub mod md5;
mod pcm;
mod streaminfo;

pub use decode::{FlacDecoder, Frame, MAX_FRAME_BYTES, Md5Check};
pub use header::{ChannelAssignment, FrameHeader, MAX_HEADER_LEN};
pub use pcm::layout;
pub use streaminfo::{STREAMINFO_LEN, StreamInfo};

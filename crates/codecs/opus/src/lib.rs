//! Opus decoder and encoder (RFC 6716, updated by RFC 8251).
//!
//! The support record is `STATUS.md` next to this crate. Ogg framing lives
//! in `cutlass-ogg`, and MP4 framing in `cutlass-isobmff`.

#![forbid(unsafe_code)]

mod celt;
mod header;
mod packet;
mod range;
mod silk;

pub use header::{MAX_CHANNELS, OpusHead, vorbis_mapping};
pub use packet::{Bandwidth, Frames, Mode, Toc, parse as parse_packet};

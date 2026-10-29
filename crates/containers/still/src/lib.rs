//! Still-image container.
//!
//! Identifies a JPEG, PNG, WebP, or GIF, splits it into packets, and writes
//! those packets back. Pixel decoding stays in the codec crates. Huffman,
//! inflate, and LZW are not implemented here. The support record is
//! `SUPPORT.md` next to this crate.

#![forbid(unsafe_code)]

mod bytes;
mod demux;
mod fail;
mod frame;
mod gif;
mod jpeg;
mod limits;
mod mux;
mod png;
mod probe;
mod webp;

pub use demux::StillDemux;
pub use frame::{FrameControl, StillKind};
pub use limits::{
    GIF_DEFAULT_DELAY, GIF_TIMESCALE, MAX_CHUNK_DEPTH, MAX_FILE_BYTES, MAX_FRAMES, UNSUPPORTED,
    WEBP_DEFAULT_DELAY_MS, WEBP_TIMESCALE, file_cap, still_limits,
};
pub use mux::StillMux;
pub use probe::{PROBE_BYTES, StillProbe, probe};

#[cfg(test)]
mod tests;

//! Ogg framing (RFC 3533) with the Opus mapping (RFC 7845).
#![forbid(unsafe_code)]

pub mod crc;
mod demux;
mod error;
mod limits;
mod logical;
mod mux;
mod opus;
pub mod page;
mod probe;
mod source;
mod vorbis;

pub use demux::{OggDemux, OpusInfo, SeekCost, VorbisInfo};
pub use error::OggError;
pub use limits::{
    MAX_COMMENTS, MAX_LINK_PROBES, MAX_OPUS_PACKET_PER_STREAM, MAX_PAGE_BYTES, MAX_RESYNC_BYTES,
    MAX_SEEK_PROBES, MAX_START_SCAN_BYTES, MAX_TAIL_SCAN_BYTES, MUX_PAGE_BYTES, MUX_PAGE_SAMPLES,
    OPUS_PREROLL, OPUS_RATE, PAGE_HEADER_BYTES, PROBE_BYTES,
};
pub use mux::{FIRST_SERIAL, OggMux, VENDOR};
pub use opus::{OpusHead, OpusTags, packet_samples};
pub use probe::{OggFamily, probe};
pub use vorbis::{VorbisComment, VorbisId};

#[cfg(test)]
mod behavior;
#[cfg(test)]
mod vorbis_behavior;

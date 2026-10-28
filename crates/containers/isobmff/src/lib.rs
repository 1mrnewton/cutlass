//! ISO base media file format: MP4, QuickTime, M4A, M4V, 3GP, fragmented MP4,
//! and the box layout of HEIC and AVIF.
//!
//! Timestamps are ticks of the track timescale. This crate does not convert
//! them to microseconds and it does not decode the payload.

#![forbid(unsafe_code)]

mod brand;
mod demux;
mod fragment;
mod heif;
mod index;
mod limits;
mod movie;
mod mux;
mod parse;
mod stsd;

pub use brand::probe;
pub use demux::IsoBmff;
pub use limits::{MAX_METADATA_BYTES, MAX_PACKET_BYTES, MAX_SAMPLES, PROBE_BYTES};
pub use mux::{FileBrand, IsoBmffMux, MuxOptions};

pub use cutlass_container::{
    AudioParams, CodecId, ColorTags, Demux, Edit, Error, FileSource, MediaKind, Mux, Packet,
    SampleFormat, SeekAt, SliceSource, Source, Stream, StreamParams, Timescale, VideoParams,
};

#[cfg(test)]
mod behavior;
#[cfg(test)]
mod oracle;

//! ISO base media file format: MP4, QuickTime, M4A, M4V, 3GP, fragmented MP4,
//! and the box layout of HEIC and AVIF.
//!
//! Timestamps are ticks of the track timescale. This crate does not convert
//! them to microseconds and it does not decode the payload.

#![forbid(unsafe_code)]

mod alac;
mod brand;
mod demux;
mod file_mux;
mod fragment;
mod heif;
mod index;
mod legacy;
mod limits;
mod movie;
mod mux;
mod opus;
mod parse;
mod shared;
mod stsd;

pub use brand::probe;
pub use demux::{IsoBmff, IsoBmffFile};
pub use file_mux::{MovieFile, Mp4Mux, Mp4Options};
pub use limits::{MAX_METADATA_BYTES, MAX_PACKET_BYTES, MAX_SAMPLES, PROBE_BYTES};
pub use mux::{
    DEFAULT_FRAGMENT_BYTES, DEFAULT_FRAGMENT_MILLIS, FileBrand, IsoBmffMux, MAX_MOVIE_TIMESCALE,
    MuxOptions,
};

pub use legacy::{
    AudioParams, CodecId, ColorTags, Demux, Edit, Error, FileSource, MediaKind, Mux, Packet,
    SampleFormat, SeekAt, SliceSource, Source, Stream, StreamParams, Timescale, VideoParams,
};

#[cfg(test)]
mod behavior;
#[cfg(test)]
mod mux_tests;
#[cfg(test)]
mod oracle;

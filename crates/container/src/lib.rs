//! Demux and mux traits.
//!
//! A container crate turns a file into packets and packets back into a file.
//! Timestamps stay in the file timescale. Only the engine time map converts
//! those ticks into microseconds.

#![forbid(unsafe_code)]

mod demux;
mod error;
mod mux;
mod source;
mod stream;

pub use demux::{Demux, SeekAt};
pub use error::Error;
pub use mux::Mux;
pub use source::{FileSource, SliceSource, Source};
pub use stream::{Edit, Stream};

pub use cutlass_codec::{
    AudioParams, CodecId, ColorTags, MediaKind, Packet, SampleFormat, StreamParams, Timescale,
    VideoParams,
};

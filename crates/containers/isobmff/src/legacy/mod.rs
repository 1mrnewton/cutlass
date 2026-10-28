//! Framing types this ISOBMFF parser was written against.
//!
//! The workspace demux and mux traits live in `cutlass_container`. This
//! module keeps the box parser on the types it already tests.

mod demux;
mod error;
mod id;
mod mux;
mod packet;
mod params;
mod pixel;
mod sample;
mod source;
mod stream;

pub use demux::{Demux, SeekAt};
pub use error::Error;
pub use id::CodecId;
pub use mux::Mux;
pub use packet::{Packet, Timescale};
pub use params::{AudioParams, MediaKind, StreamParams, VideoParams};
pub use pixel::ColorTags;
pub use sample::SampleFormat;
pub use source::{FileSource, SliceSource, Source};
pub use stream::{Edit, Stream};

//! Codec ids, packets, and parameters.
//!
//! This crate names a payload. It does not open a file and it does not decode.
//! Timestamps on a [`Packet`] are ticks of a file timescale, not microseconds.

#![forbid(unsafe_code)]

mod id;
mod packet;
mod params;
mod pixel;
mod sample;

pub use id::{CodecId, ContainerId};
pub use packet::{Packet, Timescale};
pub use params::{AudioParams, MediaKind, StreamParams, VideoParams};
pub use pixel::{ColorTags, PixelFormat};
pub use sample::SampleFormat;

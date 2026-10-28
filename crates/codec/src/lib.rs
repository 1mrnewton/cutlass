//! Codec ids, packets, parameters, and PCM.
//!
//! This crate does not open files and it does not decode. A container crate
//! turns a file into [`Packet`] values. A codec crate turns packets into
//! [`Pcm`] or the reverse.

#![forbid(unsafe_code)]

mod error;
mod id;
mod packet;
mod params;
mod pixel;
mod sample;

pub use error::CodecError;
pub use id::{CodecId, ContainerId};
pub use packet::{Packet, SideData};
pub use params::CodecParams;
pub use pixel::{ColorPrimaries, PixelFormat, Transfer};
pub use sample::{Channel, ChannelLayout, Pcm, SampleFormat};

//! Codec ids, packets, and parameters.
//!
//! This crate does not open a file and it does not decode. A container crate
//! turns a file into [`Packet`] values. A codec crate turns packets into
//! pictures or PCM.

#![forbid(unsafe_code)]

mod id;
mod packet;
mod params;

pub use id::{CodecId, ContainerId};
pub use packet::Packet;
pub use params::CodecParams;

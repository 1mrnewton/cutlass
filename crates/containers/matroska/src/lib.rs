//! Matroska and WebM framing.

#![forbid(unsafe_code)]

pub mod block;
pub mod codec;
mod demux;
pub mod ebml;
mod error;
pub mod ids;
pub mod limits;
pub mod meta;
mod mux;
mod probe;
mod seek;
mod source;

pub use demux::{Clock, MatroskaDemux};
pub use error::Error;
pub use mux::MatroskaMux;
pub use probe::{MatroskaFamily, probe};

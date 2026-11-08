//! The Vorbis I audio codec: decoder and encoder, native Rust.
//!
//! [`VorbisDecoder`] and [`VorbisEncoder`] speak raw Vorbis packets through
//! [`cutlass_decode::Decoder`] and [`cutlass_encode::Encoder`]. The codec
//! configuration is the three header packets joined with Xiph lacing
//! ([`Headers::join`]). Ogg pages, granule positions, and chaining stay in
//! the Ogg container.
#![forbid(unsafe_code)]

mod bits;
mod codebook;
mod header;
mod setup;

pub use header::{Comments, Headers, Identification, MAX_CHANNELS, MAX_CONFIG_BYTES};

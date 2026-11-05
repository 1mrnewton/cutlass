//! VP8 decoder and encoder (RFC 6386).
//!
//! The decoder accepts every VP8 bitstream version. The encoder emits
//! version 0. A WebM container is outside this crate.

#![forbid(unsafe_code)]

mod bool_coder;
mod decode;
mod filter;
mod header;
mod inter;
mod modes;
mod predict;
mod tables;
mod tokens;
mod transform;

#[cfg(test)]
mod prove;

pub use decode::{Decoder, Picture, Vp8Decoder};

use cutlass_decode::video::VideoDecoder;
use cutlass_decode::{CodecError, CodecId, CodecParams, VideoDecodeFactory};

/// Registers the VP8 decoder on a video decode factory.
pub fn register_decoders(factory: &mut VideoDecodeFactory) {
    factory.register(CodecId::Vp8, open_decoder);
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError> {
    Ok(Box::new(Vp8Decoder::open(params)?))
}

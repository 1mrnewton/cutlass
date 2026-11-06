//! VP8 decoder and encoder (RFC 6386).
//!
//! The decoder accepts every VP8 bitstream version, and the encoder emits
//! every version. [`register_engine_decoders`] and
//! [`register_engine_encoders`] put them on the engine's factories for
//! playback and export. A WebM container is outside this crate.

#![forbid(unsafe_code)]

mod bool_coder;
mod decode;
mod encode;
mod engine;
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
#[cfg(test)]
mod prove_encode;
#[cfg(test)]
mod prove_lifecycle;

pub use decode::{Decoder, Picture, Vp8Decoder};
pub use encode::{Vp8Encoder, Vp8Options};
pub use engine::{
    EngineDecoder, EngineEncoder, register_engine_decoders, register_engine_encoders,
};

use cutlass_decode::video::VideoDecoder;
use cutlass_decode::{CodecError, CodecId, CodecParams, VideoDecodeFactory};
use cutlass_encode::VideoEncodeFactory;
use cutlass_encode::video::VideoEncoder;

/// Registers the VP8 decoder on a video decode factory.
pub fn register_decoders(factory: &mut VideoDecodeFactory) {
    factory.register(CodecId::Vp8, open_decoder);
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError> {
    Ok(Box::new(Vp8Decoder::open(params)?))
}

/// Registers the VP8 encoder on a video encode factory.
pub fn register_encoders(factory: &mut VideoEncodeFactory) {
    factory.register(CodecId::Vp8, open_encoder);
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn VideoEncoder>, CodecError> {
    Ok(Box::new(Vp8Encoder::open(params)?))
}

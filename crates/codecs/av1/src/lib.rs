//! AV1 encoder and decoder.
//!
//! The support record is `STATUS.md` next to this crate. Film grain, super
//! resolution, loop restoration, CDEF, and bit depths other than 8 are
//! rejected. Inter frames are rejected by the tile decoder.

#![forbid(unsafe_code)]

mod bits;
mod deblock;
mod decode;
mod dims;
mod encode;
mod header;
mod idct;
mod predict;
mod symbol;
mod tables;
mod tile;

pub use decode::Av1Decoder;
pub use encode::Av1Encoder;

use cutlass_decode::{CodecError, CodecId, CodecParams, DecodeFactory, Decoder};
use cutlass_encode::{EncodeFactory, Encoder};

pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Av1, open_decoder);
}

pub fn register_encoders(factory: &mut EncodeFactory) {
    factory.register(CodecId::Av1, open_encoder);
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(Av1Decoder::open(params)?))
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    Ok(Box::new(Av1Encoder::open(params)?))
}

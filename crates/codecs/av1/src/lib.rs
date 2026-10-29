//! AV1 encoder and decoder.
//!
//! The support record is `STATUS.md` next to this crate. Loop restoration and
//! film grain are applied for 8-bit 4:2:0. A single-reference frame may carry
//! an order hint. A frame may use a uniform tile grid and a quantizer matrix.
//! Lossless and bit depths other than 8 are rejected. Compound, warped, and
//! overlapped motion are rejected.

#![forbid(unsafe_code)]

mod bits;
mod cdef;
mod deblock;
mod decode;
mod dims;
mod encode;
mod grain;
mod header;
mod idct;
mod inter;
mod predict;
mod qm;
mod restore;
mod superres;
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

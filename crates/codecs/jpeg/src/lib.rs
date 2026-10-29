//! JPEG decoder and encoder.
//!
//! A still is one packet and one picture. Baseline SOF0 and progressive
//! SOF2 decode to 8-bit grey or RGB. The encoder writes baseline SOF0.
//! Arithmetic coding is refused by name. The support record is `STATUS.md`.

#![forbid(unsafe_code)]

mod bits;
mod color;
mod dct;
mod decode;
mod encode;
mod huffman;
mod limits;
mod tables;

#[cfg(test)]
mod tests;

pub use decode::{JpegDecoder, JpegInfo, read_jpeg_info};
pub use encode::JpegEncoder;

use cutlass_decode::{CodecError, CodecId, CodecParams, DecodeFactory, Decoder};
use cutlass_encode::{EncodeFactory, Encoder};

/// Register this crate as the software JPEG decoder.
pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Jpeg, open_decoder);
}

/// Register this crate as the software JPEG encoder.
pub fn register_encoders(factory: &mut EncodeFactory) {
    factory.register(CodecId::Jpeg, open_encoder);
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(JpegDecoder::open(params)?))
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    Ok(Box::new(JpegEncoder::open(params)?))
}

//! JPEG decoder and encoder.
//!
//! A still is one packet and one picture. Baseline SOF0, extended SOF1, and
//! progressive SOF2 decode to 8-bit grey or RGB. The encoder writes baseline
//! SOF0. Arithmetic coding is refused by name. The support record is
//! `STATUS.md`.
//!
//! The planar core is public for codecs built on JPEG images, such as Motion
//! JPEG: [`decode_image`] returns component planes and carries [`Tables`]
//! between images, and [`Coefficients`] encodes planes at any quality.

#![forbid(unsafe_code)]

mod bits;
mod color;
mod compress;
mod dct;
mod decode;
mod encode;
mod huffman;
mod image;
mod limits;
mod tables;
mod upsample;

#[cfg(test)]
mod tests;

pub use compress::{Coefficients, PlaneRef, Planes, Sampling, Source, WriteSettings};
pub use decode::{JpegDecoder, JpegInfo, read_jpeg_info};
pub use encode::JpegEncoder;
pub use image::{ColorModel, Component, Decoded, Image, Tables, decode_image};
pub use limits::{MAX_EDGE, MAX_INPUT, MAX_PIXELS};

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

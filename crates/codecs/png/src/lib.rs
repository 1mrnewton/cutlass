//! PNG decoder and encoder.
//!
//! A still is one packet and one picture. Samples stay in the color type,
//! bit depth, and alpha domain declared by the datastream. Color conversion
//! is left to the color crate. `CodecParams.extra` of `[1]` asks the encoder
//! for Adam7. An empty extra, or `[0]`, writes a non-interlaced image.
//!
//! The support record is `STATUS.md` next to this crate.

#![forbid(unsafe_code)]

mod crc;
mod decode;
mod encode;
mod limits;
mod raster;
mod read;
mod write;
mod zlib;

#[cfg(test)]
mod tests;

pub use decode::PngDecoder;
pub use encode::PngEncoder;

use cutlass_decode::{CodecError, CodecId, CodecParams, DecodeFactory, Decoder};
use cutlass_encode::{EncodeFactory, Encoder};

/// Register this crate as the software PNG decoder.
pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Png, open_decoder);
}

/// Register this crate as the software PNG encoder.
pub fn register_encoders(factory: &mut EncodeFactory) {
    factory.register(CodecId::Png, open_encoder);
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(PngDecoder::open(params)?))
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    Ok(Box::new(PngEncoder::open(params)?))
}

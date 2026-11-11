//! WebP decoder and encoder.
//!
//! A still is one packet and one picture. An animation is one packet per
//! frame, as the `still` container splits it, and each frame decodes to the
//! composed canvas. Pictures come out as straight 8-bit RGB, or RGBA when
//! the file carries alpha. An `ICCP` profile is passed through on the
//! picture and not applied.
//!
//! The support record is `README.md` next to this crate.

#![forbid(unsafe_code)]

mod alpha;
mod animation;
mod decoder;
mod encoder;
mod limits;
mod lossless;
mod lossy;
mod picture;
mod riff;

#[cfg(test)]
mod tests;

pub use decoder::WebpDecoder;
pub use encoder::{WebpEncoder, WebpOptions};

use cutlass_decode::{CodecError, CodecId, CodecParams, DecodeFactory, Decoder};
use cutlass_encode::{EncodeFactory, Encoder};

/// Register this crate as the software WebP decoder.
pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Webp, open_decoder);
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(WebpDecoder::open(params)?))
}

/// Register this crate as the software WebP encoder.
pub fn register_encoders(factory: &mut EncodeFactory) {
    factory.register(CodecId::Webp, open_encoder);
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    Ok(Box::new(WebpEncoder::open(params)?))
}

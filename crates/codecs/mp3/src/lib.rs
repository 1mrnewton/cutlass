//! MPEG-1, MPEG-2 LSF and MPEG-2.5 Audio Layer III.
//!
//! A packet is one whole frame, header included, as `cutlass-mp3-container`
//! returns it. Layers I and II are refused. The support record is
//! `STATUS.md` next to this crate.

#![forbid(unsafe_code)]

mod analysis;
mod bits;
mod decode;
mod encode;
pub mod header;
mod huffenc;
mod huffman;
mod layer3;
mod psy;
mod quantize;
mod sideinfo;
mod synth;
#[cfg(test)]
mod synthetic;
mod tables;

pub use decode::{DECODER_DELAY, Mp3Decoder};
pub use encode::{ENCODER_DELAY, Mp3Encoder, Mp3EncoderConfig, Mp3Rate};

use cutlass_decode::{CodecError, CodecId, CodecParams, DecodeFactory, Decoder};

/// Register this crate as the software Layer III decoder.
pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Mp3, open_decoder);
}

/// Register this crate as the software Layer III encoder.
pub fn register_encoders(factory: &mut cutlass_encode::EncodeFactory) {
    factory.register(CodecId::Mp3, open_encoder);
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn cutlass_encode::Encoder>, CodecError> {
    Ok(Box::new(Mp3Encoder::open(params)?))
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(Mp3Decoder::open(params)?))
}

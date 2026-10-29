//! Integer and float PCM.
//!
//! Packet bytes are the samples a container already framed. This crate does
//! not parse WAV, AIFF, or CAF. The support record is `STATUS.md`.

#![forbid(unsafe_code)]

mod convert;
mod decode;
mod encode;
mod layout;
mod limit;

pub use decode::PcmDecoder;
pub use encode::PcmEncoder;
pub use layout::{Arrangement, Coding, Endian, Layout};

#[cfg(test)]
mod tests;

use cutlass_decode::{CodecError, CodecId, CodecParams, DecodeFactory, Decoder};
use cutlass_encode::{EncodeFactory, Encoder};

/// Register this crate as the software PCM decoder.
pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Pcm, open_decoder);
}

/// Register this crate as the software PCM encoder.
pub fn register_encoders(factory: &mut EncodeFactory) {
    factory.register(CodecId::Pcm, open_encoder);
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(PcmDecoder::open(params)?))
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    Ok(Box::new(PcmEncoder::open(params)?))
}

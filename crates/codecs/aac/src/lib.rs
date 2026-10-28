//! AAC-LC encoder and decoder.
//!
//! The support record is `STATUS.md` next to this crate. Spectral band
//! replication, parametric stereo, Main, SSR, LTP, and xHE are unsupported.
//! Transport framing lives in `cutlass-adts` and `cutlass-isobmff`.

#![forbid(unsafe_code)]

mod bands;
mod bits;
mod config;
mod decode;
mod encode;
mod filterbank;
mod huff_tables;
mod huffman;
mod quant;
mod tns;

pub use config::{AudioConfig, EncoderConfig, Profile, RateControl};
pub use decode::AacDecoder;
pub use encode::AacEncoder;

use cutlass_decode::{CodecError, CodecId, CodecParams, DecodeFactory, Decoder};
use cutlass_encode::{EncodeFactory, Encoder};

pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Aac, open_decoder);
}

pub fn register_encoders(factory: &mut EncodeFactory) {
    factory.register(CodecId::Aac, open_encoder);
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(AacDecoder::open(params)?))
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    Ok(Box::new(AacEncoder::open(params)?))
}

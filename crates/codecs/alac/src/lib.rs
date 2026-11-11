//! Apple Lossless (ALAC) decoder and encoder.
//!
//! One packet is one ALAC frame. `CodecParams::extra` is the magic cookie
//! (`ALACSpecificConfig`, then an optional `chan` atom); [`Cookie::parse`]
//! also reads the MP4 box and legacy `frma` forms. Containers frame the
//! packets; this crate does not read MP4 or CAF. The support record is
//! `README.md` next to this crate.

#![forbid(unsafe_code)]

mod bits;
mod config;
mod decode;
mod encode;
mod frame;
mod golomb;
mod predictor;

#[cfg(test)]
mod tests;

pub use config::{
    AlacConfig, CHANNEL_ATOM_BYTES, CONFIG_BYTES, Cookie, DEFAULT_FRAME_LENGTH, LAYOUT_TAGS,
    MAX_CHANNELS, MAX_FRAME_LENGTH, default_layout,
};
pub use decode::{AlacDecoder, DecodedFrame, MAX_PACKET_BYTES};
pub use encode::{AlacEncoder, EncodeMode, EncoderConfig};

use cutlass_decode::{CodecError, CodecId, CodecParams, DecodeFactory, Decoder};
use cutlass_encode::{EncodeFactory, Encoder};

/// Register this crate as the software ALAC decoder.
pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Alac, open_decoder);
}

/// Register this crate as the software ALAC encoder.
pub fn register_encoders(factory: &mut EncodeFactory) {
    factory.register(CodecId::Alac, open_encoder);
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(AlacDecoder::open(params)?))
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    Ok(Box::new(AlacEncoder::open(params)?))
}

//! AC-3 and E-AC-3 (ATSC A/52:2018, including Annex E).
//!
//! The support record is `README.md` next to this crate.

#![forbid(unsafe_code)]

mod aht;
mod bitalloc;
mod bits;
mod crc;
mod decode;
mod ecpl;
mod encode;
mod exponents;
mod frame;
mod header;
mod mdct;
mod spx;
mod tables;
mod tpnp;

pub use decode::{Ac3Decoder, Downmix, Extension, Options};
pub use encode::{
    Ac3Encoder, ENCODER_DELAY, EncoderConfig, EncoderMetadata, Format, MixCodes, default_bit_rate,
};
pub use frame::{DynamicRange, Tools, Transient};
pub use header::{
    Dac3, Dec3, Dec3Substream, MAX_FRAME_BYTES, Metadata, MixLevels, PROBE_BYTES, SYNCWORD,
    StreamType, SyncInfo, Syntax, ac3_bit_rate, compr_db, compr_gain, dynrng_gain, probe,
    program_gain,
};

use cutlass_decode::{CodecError, CodecId, CodecParams, DecodeFactory, Decoder};

fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(Ac3Decoder::open(params)?))
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn cutlass_encode::Encoder>, CodecError> {
    Ok(Box::new(Ac3Encoder::open(params)?))
}

/// Register the AC-3 and E-AC-3 decoder.
pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Ac3, open_decoder);
    factory.register(CodecId::Eac3, open_decoder);
}

/// Register the AC-3 and E-AC-3 encoder.
pub fn register_encoders(factory: &mut cutlass_encode::EncodeFactory) {
    factory.register(CodecId::Ac3, open_encoder);
    factory.register(CodecId::Eac3, open_encoder);
}

//! The Vorbis I audio codec, native Rust.
//!
//! [`VorbisDecoder`] and [`VorbisEncoder`] speak raw Vorbis packets. The
//! codec configuration is the three header packets joined with Xiph lacing
//! ([`Headers::join`]). Ogg pages, granule positions, and chaining stay in
//! `cutlass-ogg`. The support record is `README.md` next to this crate.

#![forbid(unsafe_code)]

mod bits;
mod codebook;
mod decode;
mod encoder;
mod floor;
mod header;
mod mdct;
mod residue;
mod setup;
mod tables;
mod window;

pub use decode::VorbisDecoder;
pub use encoder::{Options, VorbisEncoder};
pub use header::{Comments, Headers, Identification, MAX_CHANNELS, MAX_CONFIG_BYTES};

use cutlass_decode::{CodecError, CodecId, CodecParams, DecodeFactory, Decoder};
use cutlass_encode::{EncodeFactory, Encoder};

pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Vorbis, open_decoder);
}

pub fn register_encoders(factory: &mut EncodeFactory) {
    factory.register(CodecId::Vorbis, open_encoder);
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(VorbisDecoder::open(params)?))
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    Ok(Box::new(VorbisEncoder::open(params)?))
}

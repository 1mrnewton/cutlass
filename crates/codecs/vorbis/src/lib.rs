//! The Vorbis I audio codec, native Rust.
//!
//! [`VorbisDecoder`] speaks raw Vorbis packets through
//! [`cutlass_decode::Decoder`]. The codec configuration is the three header
//! packets joined with Xiph lacing ([`Headers::join`]). Ogg pages, granule
//! positions, and chaining stay in `cutlass-ogg`. The support record is
//! `STATUS.md` next to this crate.

#![forbid(unsafe_code)]

mod bits;
mod codebook;
mod decode;
mod floor;
mod header;
mod mdct;
mod residue;
mod setup;
mod tables;
mod window;

pub use decode::VorbisDecoder;
pub use header::{Comments, Headers, Identification, MAX_CHANNELS, MAX_CONFIG_BYTES};

use cutlass_decode::{CodecError, CodecId, CodecParams, DecodeFactory, Decoder};

pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Vorbis, open_decoder);
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(VorbisDecoder::open(params)?))
}

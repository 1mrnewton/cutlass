//! Opus decoder and encoder (RFC 6716, updated by RFC 8251).
//!
//! The support record is `STATUS.md` next to this crate. Ogg framing lives
//! in `cutlass-ogg`, and MP4 framing in `cutlass-isobmff`.

#![forbid(unsafe_code)]

mod celt;
mod decode;
mod encode;
mod header;
mod packet;
mod range;
mod silk;

pub use decode::OpusDecoder;
pub use encode::OpusEncoder;
pub use header::{MAX_CHANNELS, OpusHead, vorbis_mapping};
pub use packet::{Bandwidth, Frames, Mode, Toc, parse as parse_packet};

use cutlass_decode::{CodecError, CodecId, CodecParams, DecodeFactory, Decoder};
use cutlass_encode::{EncodeFactory, Encoder};

pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Opus, open_decoder);
}

pub fn register_encoders(factory: &mut EncodeFactory) {
    factory.register(CodecId::Opus, open_encoder);
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(OpusDecoder::open(params)?))
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    Ok(Box::new(OpusEncoder::open(params)?))
}

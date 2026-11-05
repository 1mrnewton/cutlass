//! FLAC, the Free Lossless Audio Codec (RFC 9639).
//!
//! One packet is one FLAC frame, as the native `.flac` container, Ogg FLAC,
//! and MP4 `fLaC` sample entries all frame it. This crate does not read the
//! `fLaC` marker or metadata blocks; that is `cutlass-flac-container`.

#![forbid(unsafe_code)]

mod bits;
pub mod crc;
mod decode;
mod encode;
mod header;
pub mod md5;
mod pcm;
mod streaminfo;

pub use decode::{FlacDecoder, Frame, MAX_FRAME_BYTES, Md5Check};
pub use encode::{Apodization, DEFAULT_LEVEL, EncodedFrame, FlacEncoder, Settings, Stereo};
pub use header::{ChannelAssignment, FrameHeader, MAX_HEADER_LEN};
pub use pcm::layout;
pub use streaminfo::{STREAMINFO_LEN, StreamInfo};

use cutlass_decode::{CodecError, CodecParams, DecodeFactory, Decoder};
use cutlass_encode::{CodecId, EncodeFactory, Encoder};

/// Register this crate as the software FLAC decoder.
pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Flac, open_decoder);
}

/// Register this crate as the software FLAC encoder, at [`DEFAULT_LEVEL`].
pub fn register_encoders(factory: &mut EncodeFactory) {
    factory.register(CodecId::Flac, open_encoder);
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(FlacDecoder::open(params)?))
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    Ok(Box::new(FlacEncoder::open(params)?))
}

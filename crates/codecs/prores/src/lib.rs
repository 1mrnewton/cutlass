//! Apple ProRes, decode and encode, per SMPTE RDD 36:2022.
//!
//! Every profile decodes: 422 Proxy, LT, 422, HQ at 10-bit 4:2:2, and 4444
//! and 4444 XQ at 12-bit 4:4:4 with 8- or 16-bit alpha. Progressive and
//! interlaced frames, custom matrices, and odd sizes are covered. Slices
//! decode in parallel on [`cutlass_pool`]; one worker gives the same samples.
//!
//! The encoder writes every profile, progressive, at Apple's target data
//! rates: one quantizer for the frame, searched to fill the budget, then
//! slices moved one step finer where the budget still allows.
//!
//! [`ProresDecoder`] is the video decoder: wide 4:2:2 or 4:4:4 frames with
//! alpha on [`cutlass_decode::VideoFrame::alpha`]. [`ProresPictureDecoder`] is
//! the app's picture decoder, which hands on 8-bit 4:2:0.

#![forbid(unsafe_code)]

mod bits;
mod color;
mod convert;
mod dct;
mod decode;
mod decoder;
mod encode;
mod header;
mod profile;
mod tables;

#[cfg(test)]
mod tests;

pub use convert::to_yuv420p;
pub use decode::{DecodedFrame, decode_frame};
pub use decoder::{ProresDecoder, ProresPictureDecoder};
pub use encode::{ProresEncoder, target_frame_bits, target_mbps};
pub use header::{AlphaBits, Chroma, FrameHeader, Interlace, parse_frame_header};
pub use profile::Profile;

use cutlass_decode::video::VideoDecoder;
use cutlass_decode::{
    CodecError, CodecId, CodecParams, DecodeFactory, Decoder, VideoDecodeFactory,
};
use cutlass_encode::video::VideoEncoder;
use cutlass_encode::{EncodeFactory, Encoder, VideoEncodeFactory};

/// Registers the picture decoder the app reads through.
pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Prores, open_picture_decoder);
}

/// Registers the wide video decoder.
pub fn register_video_decoders(factory: &mut VideoDecodeFactory) {
    factory.register(CodecId::Prores, open_video_decoder);
}

/// Registers the encoder the app exports through: RGBA or 8-bit 4:2:0 in.
pub fn register_encoders(factory: &mut EncodeFactory) {
    factory.register(CodecId::Prores, open_encoder);
}

/// Registers the planar video encoder.
pub fn register_video_encoders(factory: &mut VideoEncodeFactory) {
    factory.register(CodecId::Prores, open_video_encoder);
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    Ok(Box::new(ProresEncoder::open(params)?))
}

fn open_video_encoder(params: &CodecParams) -> Result<Box<dyn VideoEncoder>, CodecError> {
    Ok(Box::new(ProresEncoder::open(params)?))
}

fn open_picture_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(ProresPictureDecoder::open(params)?))
}

fn open_video_decoder(params: &CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError> {
    Ok(Box::new(ProresDecoder::open(params)?))
}

//! Apple ProRes, decode and encode, per SMPTE RDD 36:2022.
//!
//! Every profile decodes: 422 Proxy, LT, 422, HQ at 10-bit 4:2:2, and 4444
//! and 4444 XQ at 12-bit 4:4:4 with 8- or 16-bit alpha. Progressive and
//! interlaced frames, custom matrices, and odd sizes are covered. Slices
//! decode in parallel on [`cutlass_pool`]; one worker gives the same samples.
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
mod header;
mod profile;
mod tables;

pub use convert::to_yuv420p;
pub use decode::{DecodedFrame, decode_frame};
pub use decoder::{ProresDecoder, ProresPictureDecoder};
pub use header::{AlphaBits, Chroma, FrameHeader, Interlace, parse_frame_header};
pub use profile::Profile;

use cutlass_decode::video::VideoDecoder;
use cutlass_decode::{
    CodecError, CodecId, CodecParams, DecodeFactory, Decoder, VideoDecodeFactory,
};

/// Registers the picture decoder the app reads through.
pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Prores, open_picture_decoder);
}

/// Registers the wide video decoder.
pub fn register_video_decoders(factory: &mut VideoDecodeFactory) {
    factory.register(CodecId::Prores, open_video_decoder);
}

fn open_picture_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(ProresPictureDecoder::open(params)?))
}

fn open_video_decoder(params: &CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError> {
    Ok(Box::new(ProresDecoder::open(params)?))
}

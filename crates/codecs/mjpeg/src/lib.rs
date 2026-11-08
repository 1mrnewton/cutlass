//! Motion JPEG decoding.
//!
//! A packet is one JPEG image, or two fields of an interlaced frame. The
//! JPEG coding itself is `cutlass-jpeg`'s; this crate owns the Motion JPEG
//! conventions around it: the Annex K.3 Huffman tables when a frame has no
//! DHT, tables carried from earlier packets and from the extradata, field
//! pairs woven into one frame, format A headers, and a refusal of format B.
//!
//! [`MjpegDecoder`] is the video decoder: planar frames at their coded
//! sampling. [`MjpegPictureDecoder`] is the app's picture decoder: RGBA.

#![forbid(unsafe_code)]

mod config;
mod decoder;
mod picture;

#[cfg(test)]
mod tests;

pub use config::{FieldOrder, Flavour};
pub use decoder::{MjpegDecoder, MjpegPictureDecoder};
pub use picture::{MAX_PACKET, Picture};

use cutlass_decode::video::VideoDecoder;
use cutlass_decode::{
    CodecError, CodecId, CodecParams, DecodeFactory, Decoder, VideoDecodeFactory,
};

/// Registers the picture decoder the app reads through.
pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Mjpeg, open_picture_decoder);
}

/// Registers the planar video decoder.
pub fn register_video_decoders(factory: &mut VideoDecodeFactory) {
    factory.register(CodecId::Mjpeg, open_video_decoder);
}

fn open_picture_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(MjpegPictureDecoder::open(params)?))
}

fn open_video_decoder(params: &CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError> {
    Ok(Box::new(MjpegDecoder::open(params)?))
}

//! Motion JPEG, decode and encode.
//!
//! A packet is one JPEG image, or two fields of an interlaced frame. The
//! JPEG coding itself is `cutlass-jpeg`'s; this crate owns the Motion JPEG
//! conventions around it: the Annex K.3 Huffman tables when a frame has no
//! DHT, tables carried from earlier packets and from the extradata, field
//! pairs woven into one frame, format A headers, and a refusal of format B.
//!
//! [`MjpegDecoder`] is the video decoder: planar frames at their coded
//! sampling. [`MjpegPictureDecoder`] is the app's picture decoder: RGBA.
//! [`MjpegEncoder`] writes baseline frames at a fixed quality or to a
//! bitrate, from RGBA or planar input.

#![forbid(unsafe_code)]

mod config;
mod decoder;
mod encoder;
mod picture;

#[cfg(test)]
mod fuzz;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod tests;

pub use config::{FieldOrder, Flavour};
pub use cutlass_jpeg::Sampling;
pub use decoder::{MjpegDecoder, MjpegPictureDecoder};
pub use encoder::{DEFAULT_QUALITY, MjpegEncoder, RateMode};
pub use picture::{MAX_PACKET, Picture};

use cutlass_decode::video::VideoDecoder;
use cutlass_decode::{
    CodecError, CodecId, CodecParams, DecodeFactory, Decoder, VideoDecodeFactory,
};
use cutlass_encode::video::VideoEncoder;
use cutlass_encode::{EncodeFactory, Encoder, VideoEncodeFactory};

/// Registers the picture decoder the app reads through.
pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Mjpeg, open_picture_decoder);
}

/// Registers the planar video decoder.
pub fn register_video_decoders(factory: &mut VideoDecodeFactory) {
    factory.register(CodecId::Mjpeg, open_video_decoder);
}

/// Registers the encoder the app exports through: RGBA in.
pub fn register_encoders(factory: &mut EncodeFactory) {
    factory.register(CodecId::Mjpeg, open_encoder);
}

/// Registers the planar video encoder.
pub fn register_video_encoders(factory: &mut VideoEncodeFactory) {
    factory.register(CodecId::Mjpeg, open_video_encoder);
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    Ok(Box::new(MjpegEncoder::open(params)?))
}

fn open_video_encoder(params: &CodecParams) -> Result<Box<dyn VideoEncoder>, CodecError> {
    Ok(Box::new(MjpegEncoder::open(params)?))
}

fn open_picture_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(MjpegPictureDecoder::open(params)?))
}

fn open_video_decoder(params: &CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError> {
    Ok(Box::new(MjpegDecoder::open(params)?))
}

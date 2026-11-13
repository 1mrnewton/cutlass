//! MediaCodec decoders and encoders, and the AAudio preview queue.
//!
//! H.264, HEVC, VP9 and AV1 decode, and H.264 and HEVC encode, go through
//! MediaCodec behind [`cutlass_decode::Decoder`] and
//! [`cutlass_encode::Encoder`], the phone's hardware codec first and
//! Android's software codec after it. The phone registers them ahead of the
//! Rust video codecs. An opener that this phone cannot run returns
//! [`cutlass_decode::CodecError::Unsupported`], and the factory asks the
//! next opener. Off Android every opener declines.
//!
//! [`OutputQueue`] is a shared-mode AAudio stream. Playback calls that safe
//! queue. This crate does not mix samples.

mod media;
mod planes;
mod slice;

#[cfg(target_os = "android")]
mod aaudio;
#[cfg(not(target_os = "android"))]
mod off;

#[cfg(target_os = "android")]
pub use aaudio::OutputQueue;
#[cfg(not(target_os = "android"))]
pub use off::OutputQueue;

pub use slice::QueueError;

pub use media::{
    register_decoders, register_decoders_with, register_encoders, register_encoders_with,
};

#[cfg(target_os = "android")]
pub use media::{VideoDecoder, VideoEncoder, open_decoder_on, open_encoder_on};

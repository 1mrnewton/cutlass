//! VideoToolbox, AudioToolbox, and ImageIO behind [`cutlass_decode::Decoder`]
//! and [`cutlass_encode::Encoder`]. The preview output queue is Core Audio.
//! Playback calls that safe queue. This crate does not mix samples.
//!
//! The desktop registers these openers first. An opener that this framework
//! cannot run returns [`cutlass_decode::CodecError::Unsupported`], and the
//! factory asks the next opener.

mod apple;
mod limits;
mod nal;
mod orient;
mod status;

pub use apple::{register_decoders, register_encoders};

#[cfg(target_vendor = "apple")]
pub use apple::{
    OutputQueue, QueueError, decode_file, encode_file, release_buffer, to_cpu, with_pixel_buffer,
};

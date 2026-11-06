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
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
mod reorder;
mod status;

pub use apple::{
    register_decoders, register_decoders_with, register_encoders, register_encoders_with,
    register_surface_decoders_with,
};

#[cfg(target_vendor = "apple")]
pub use apple::{
    OutputQueue, QueueError, StreamSink, decode_file, encode_file, import_surface, release_buffer,
    run_at_interface_priority, to_cpu, with_pixel_buffer,
};
#[cfg(target_os = "macos")]
pub use apple::{listen_for_opened_files, take_opened_files, wake_on_opened_files};

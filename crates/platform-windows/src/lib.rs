//! Media Foundation and WIC decoders and encoders, and the WASAPI preview
//! queue.
//!
//! H.264, HEVC, and AAC go through Media Foundation, on the GPU first, and
//! JPEG and PNG through the Windows Imaging Component, all behind
//! [`cutlass_decode::Decoder`] and [`cutlass_encode::Encoder`]. The desktop
//! registers them first. An opener that this Windows cannot run returns
//! [`cutlass_decode::CodecError::Unsupported`], and the factory asks the next
//! opener. Off Windows every opener declines.
//!
//! [`OutputQueue`] is a shared-mode WASAPI stream. Playback calls that safe
//! queue. This crate does not mix samples.

mod format;
mod limits;
mod mf;
mod nal;
mod nv12;
mod slice;

#[cfg(not(target_os = "windows"))]
mod off;
#[cfg(target_os = "windows")]
mod wasapi;

pub use mf::{
    register_decoders, register_decoders_with, register_encoders, register_encoders_with,
};

#[cfg(not(target_os = "windows"))]
pub use off::OutputQueue;
#[cfg(target_os = "windows")]
pub use wasapi::OutputQueue;

pub use slice::QueueError;

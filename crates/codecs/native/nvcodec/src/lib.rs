//! NVDEC and NVENC video decode and encode for NVIDIA GPUs.
//!
//! The startup query lists decode and encode codecs. It does not decode or
//! encode a frame. Windows has no query here yet.

#![cfg(any(target_os = "linux", target_os = "windows"))]

mod caps;
#[cfg(target_os = "linux")]
mod nvidia;

pub use caps::Capabilities;

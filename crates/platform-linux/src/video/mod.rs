//! Hardware video openers, or a decline when the platform is not Linux.
//!
//! On Linux the openers are NVDEC and NVENC first, then VA-API. Each one
//! declines with [`cutlass_decode::CodecError::Unsupported`] when its
//! library, device, profile, or picture size is missing, and the factory
//! moves on to the next opener, and in the end to the Rust codec.
//!
//! The pieces here are plain Rust and build everywhere, so their tests run
//! on any host. The calls into the system libraries are in `imp`.

#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

mod backend;
mod colour;
mod fallback;
mod kind;
mod limits;
mod packets;
mod rate;

#[cfg(not(target_os = "linux"))]
mod off;
#[cfg(not(target_os = "linux"))]
pub use off::{
    probe, register_decoders, register_decoders_with, register_encoders, register_encoders_with,
};

#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
mod imp;
#[cfg(target_os = "linux")]
pub use imp::{
    probe, register_decoders, register_decoders_with, register_encoders, register_encoders_with,
};

pub use backend::BACKEND_VAR;

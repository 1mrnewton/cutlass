//! The Metal device behind [`cutlass_gpu::Device`].
//!
//! This path owns framebuffer bytes and blends RGBA on the CPU. A platform
//! buffer is refused. A clip does not choose a backend.

#![forbid(unsafe_code)]

mod blend;
mod error;
mod framebuffer;

pub use error::DeviceError;
pub use framebuffer::MetalDevice;

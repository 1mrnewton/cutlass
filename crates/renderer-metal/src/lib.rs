//! The Metal device behind [`cutlass_gpu::Device`].
//!
//! The portable path owns framebuffer bytes and blends RGBA on the CPU.
//! A platform buffer is refused there. On macOS and iOS, the same device
//! can submit those bytes through a Metal command buffer. A clip does not
//! choose a backend.

mod blend;
mod error;
mod framebuffer;

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[allow(unsafe_code)]
pub mod metal;

pub use error::DeviceError;
pub use framebuffer::MetalDevice;

//! The CPU device behind [`cutlass_gpu::Device`].
//!
//! [`WindowsDevice`] blends RGBA on the CPU and refuses a platform buffer.
//! [`open_direct3d`] asks for the Direct3D 11 backend. On this host it
//! returns [`WindowsError::GpuUnsupported`].

#![forbid(unsafe_code)]

mod blend;
mod device;
mod error;
mod off;

pub use device::WindowsDevice;
pub use error::WindowsError;
pub use off::{Direct3DDevice, open_direct3d};

#[cfg(test)]
mod pixels;

//! The Direct3D device behind [`cutlass_gpu::Device`].
//!
//! [`WindowsDevice`] is the portable CPU path. It is always compiled.
//! It blends RGBA on the CPU and refuses a platform buffer.
//! [`open_direct3d`] is the Direct3D 11 backend on Windows. Every other host
//! returns [`WindowsError::GpuUnsupported`].

mod blend;
mod device;
mod error;

pub use device::WindowsDevice;
pub use error::WindowsError;

#[cfg(target_os = "windows")]
mod direct3d;
#[cfg(not(target_os = "windows"))]
mod off;

#[cfg(target_os = "windows")]
pub use direct3d::{Direct3DDevice, open_direct3d};
#[cfg(not(target_os = "windows"))]
pub use off::{Direct3DDevice, open_direct3d};

#[cfg(test)]
mod pixels;

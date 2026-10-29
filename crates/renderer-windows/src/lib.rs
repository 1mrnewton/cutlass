//! The CPU device behind [`cutlass_gpu::Device`].
//!
//! [`WindowsDevice`] blends RGBA on the CPU and refuses a platform buffer.

#![forbid(unsafe_code)]

mod blend;
mod device;
mod error;

pub use device::WindowsDevice;
pub use error::WindowsError;

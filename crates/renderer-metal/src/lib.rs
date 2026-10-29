//! The Metal device behind [`cutlass_gpu::Device`].
//!
//! A clip does not choose a backend.

#![forbid(unsafe_code)]

mod error;

pub use error::DeviceError;

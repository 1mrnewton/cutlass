//! Upload a decoded frame and turn it into RGB.
//!
//! CPU planes are copied into textures. On Apple, an IOSurface is wrapped and
//! sampled in place. The colour weights are the ones from [`cutlass_color`].

#![deny(unsafe_code)]

mod error;
mod gpu;
#[cfg(target_vendor = "apple")]
mod import;

pub use error::Error;
pub use gpu::{Gpu, RgbImage};

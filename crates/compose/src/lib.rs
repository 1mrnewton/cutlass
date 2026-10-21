//! Upload a decoded frame and turn it into RGB.
//!
//! CPU planes are copied into textures. An IOSurface import, which does not
//! copy, comes later. The colour weights are the ones from [`cutlass_color`].

#![deny(unsafe_code)]

mod error;
mod gpu;

pub use error::Error;
pub use gpu::{Gpu, RgbImage};

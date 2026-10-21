//! YUV to RGB for the frames Cutlass decodes.
//!
//! The result is R'G'B' in the source transfer. It is not linear, and it is
//! not tone mapped. A later pass does that.

#![deny(unsafe_code)]

mod convert;
mod error;
mod matrix;

pub use convert::{Coefficients, YuvPlanes, to_rgba};
pub use error::Error;
pub use matrix::Matrix;

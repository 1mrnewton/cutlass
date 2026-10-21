//! AudioToolbox AAC decode and encode for macOS and iOS.
//!
//! Takes compressed packets plus the container's codec config record.
//! Container parsing does not belong here. Empty on non-Apple targets.
//! Decoded samples are the planar `f32` layout from `cutlass-core`.

#![cfg(target_vendor = "apple")]

mod convert;
mod decode;
mod encode;
mod error;
mod format;

pub use decode::{Decoder, decodes};
pub use encode::{Encoder, Packet};
pub use error::Error;

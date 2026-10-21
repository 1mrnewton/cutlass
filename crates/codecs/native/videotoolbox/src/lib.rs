//! VideoToolbox video decode and encode for macOS and iOS.
//!
//! Takes compressed frames plus the container's codec config record. Container
//! parsing does not belong here. Empty on non-Apple targets.

#![cfg(target_vendor = "apple")]

mod caps;
mod cf;
mod decode;
mod encode;
mod error;
mod pixel_buffer;
mod queue;
mod time;

pub use caps::{Capabilities, EncoderInfo};
pub use decode::{Decoder, DecoderOptions, decodes};
pub use encode::{Encoder, EncoderConfig};
pub use error::Error;
pub use pixel_buffer::{PixelBuffer, Plane, PlaneMut};

//! Types shared across Cutlass crates: timestamps, codecs, packets, frames, and stills.

#![deny(unsafe_code)]

mod audio;
mod codec;
mod color;
mod frame;
mod packet;
mod pixel;
mod rotation;
mod still;
mod time;

pub use audio::{AudioCodec, AudioFrame, AudioSamples, AudioSamplesError, AudioStreamConfig};
pub use codec::{VideoCodec, VideoStreamConfig};
pub use color::{ColorTags, ContentLight, MasteringDisplay};
pub use frame::VideoFrame;
pub use packet::Packet;
pub use pixel::{ColorRange, PixelFormat};
pub use rotation::Rotation;
pub use still::{Still, StillError};
pub use time::Timestamp;

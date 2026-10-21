//! Types shared across Cutlass crates: timestamps, codecs, packets, and frames.

#![deny(unsafe_code)]

mod audio;
mod codec;
mod color;
mod frame;
mod packet;
mod pixel;
mod time;

pub use audio::{AudioCodec, AudioFrame, AudioSamples, AudioSamplesError, AudioStreamConfig};
pub use codec::{VideoCodec, VideoStreamConfig};
pub use color::ColorTags;
pub use frame::VideoFrame;
pub use packet::Packet;
pub use pixel::{ColorRange, PixelFormat};
pub use time::Timestamp;

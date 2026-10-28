//! Access units, PCM, and the errors a codec returns.
//!
//! These types are what an encoder and a decoder exchange. A container
//! packet in this crate is a different value: file timescale, no side data,
//! no PCM.

mod error;
mod packet;
mod params;
mod pixel;
mod plane;
mod sample;

pub use error::CodecError;
pub use packet::{Packet, SideData};
pub use params::CodecParams;
pub use pixel::{ColorPrimaries, ColorRange, MatrixCoefficients, PixelFormat, Transfer};
pub use plane::{Plane, Yuv420};
pub use sample::{Channel, ChannelLayout, Pcm, SampleFormat};

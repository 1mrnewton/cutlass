//! Access units, PCM, stills, and the errors a codec returns.
//!
//! These types are what an encoder and a decoder exchange. A container
//! packet in this crate is a different value: file timescale, no side data,
//! no PCM. A still is coded samples. Color conversion is not done here.

mod error;
mod packet;
mod params;
mod pixel;
mod sample;
mod still;

pub use error::CodecError;
pub use packet::{Packet, SideData};
pub use params::CodecParams;
pub use pixel::{ColorPrimaries, PixelFormat, Transfer};
pub use sample::{Channel, ChannelLayout, Pcm, SampleFormat};
pub use still::{
    BitDepth, ColorType, ImageError, PhysicalPixel, Rgb8, StillImage, TextItem, Transparency,
};

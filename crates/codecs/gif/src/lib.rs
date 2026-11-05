//! GIF89a pictures from RGBA frames.
//!
//! Each frame gets its own palette of at most 256 colors, cut from the
//! frame's colors by median cut, and its own local color table. The
//! packets follow the still container's GIF split: the first carries the
//! header and the logical screen, every packet carries one graphic-control
//! extension and one image, and the last also carries the trailer, so
//! joining the packets in order is the file. The encoder holds one frame
//! back so that the trailer lands on the last one when it is drained.
//! Alpha is not kept; every frame is opaque.

#![forbid(unsafe_code)]

mod encode;
mod lzw;
mod quant;

use cutlass_decode::{CodecError, CodecId, CodecParams};
use cutlass_encode::{EncodeFactory, Encoder};

pub use encode::GifEncoder;

/// The GIF clock: a frame delay is stored in hundredths of a second.
pub const GIF_TICKS_PER_SECOND: i64 = 100;

pub fn register_encoders(factory: &mut EncodeFactory) {
    factory.register(CodecId::Gif, open_encoder);
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    Ok(Box::new(GifEncoder::open(params)?))
}

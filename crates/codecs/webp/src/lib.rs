//! WebP pictures to RGBA frames: lossy, lossless, alpha, and animation.
//!
//! [`WebpDecoder`] takes the still container's packets: a still file as one
//! packet, an animation as one packet per frame. It decodes the lossy
//! `VP8 ` bitstream with `cutlass-vp8`, the lossless `VP8L` bitstream and
//! the `ALPH` plane here, and paints animation frames onto a canvas it
//! keeps between packets. Every output is straight sRGB RGBA, the whole
//! canvas.
//!
//! Specifications: RFC 9649 (WebP container and lossless bitstream) and
//! RFC 6386 (VP8). There is no encoder in this crate.

#![forbid(unsafe_code)]

mod alpha;
mod bits;
mod decode;
mod huffman;
mod lossless;
mod picture;
mod yuv;

use cutlass_decode::{CodecError, CodecId, CodecParams, DecodeFactory, Decoder};

pub use decode::{MAX_PIXELS, WebpDecoder};

/// The animation clock: a frame duration is stored in milliseconds.
pub const WEBP_TICKS_PER_SECOND: i64 = 1000;

/// Register this crate as the software WebP decoder.
pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Webp, open_decoder);
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(WebpDecoder::open(params)?))
}

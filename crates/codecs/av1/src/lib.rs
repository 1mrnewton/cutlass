//! AV1 encoder and decoder in safe Rust.
//!
//! The decoder ([`dec`]) implements the whole AV1 decoding process of the
//! specification for every profile: 8, 10 and 12 bits, 4:2:0, 4:2:2, 4:4:4
//! and monochrome, all intra, inter, compound, warped and screen-content
//! tools, the in-loop filters, super resolution, scaled references and film
//! grain. The encoder ([`enc`]) codes those bitstreams through the decoder's
//! own syntax code. `STATUS.md` next to this crate is the support record.

#![forbid(unsafe_code)]
// Kernels index fixed-size tables and take many scalar parameters, as the
// specification's processes do.
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::cast_lossless,
    clippy::identity_op,
    clippy::erasing_op
)]

pub mod dec;
mod decode;
pub mod dsp;
pub mod enc;
mod encode;
pub mod spec;
mod yuv;

pub use decode::Av1Decoder;
pub use encode::Av1Encoder;
pub use yuv::{AsView, View};

use cutlass_decode::{CodecError, CodecId, CodecParams, DecodeFactory, Decoder};
use cutlass_encode::{EncodeFactory, Encoder};

pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Av1, open_decoder);
}

pub fn register_encoders(factory: &mut EncodeFactory) {
    factory.register(CodecId::Av1, open_encoder);
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(Av1Decoder::open(params)?))
}

fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    Ok(Box::new(Av1Encoder::open(params)?))
}

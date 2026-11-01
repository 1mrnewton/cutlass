//! AV1 encoder and decoder.
//!
//! The support record is `STATUS.md` next to this crate. Loop restoration and
//! film grain are applied for 8-bit 4:2:0. A single-reference frame may carry
//! an order hint. A frame may use a uniform tile grid, a quantizer matrix,
//! and a superblock quantizer delta. DC luma may use the recursive intra
//! filter.
//! Bit depths other than 8 are rejected. A frame with `base_q_idx` 0 uses a
//! 4×4 Walsh-Hadamard transform. A palette block and an intra block copy are
//! reconstructed, and an inter frame may be forced to integer motion vectors.
//! Two references may be
//! averaged, weighted by order-hint distance, or blended with a wedge or a
//! difference mask. A skip-mode block uses the nearest compound pair and
//! writes no residual. A local warp fits the neighboring motion, and overlapped
//! motion blends those neighboring predictors. A block may take a motion
//! candidate from the field projected out of an earlier frame. An inter frame
//! may sample a reference of a different size, and it may copy its upscaled
//! size from a reference. A warp of a scaled reference is rejected. A frame
//! may replace a loop-filter reference or mode delta, and a later frame keeps it.
//! A superblock may add its own loop-filter delta. A sequence may use a
//! 128×128 superblock. A switch frame is predicted as an inter frame and
//! refreshes every reference slot. A segment may change the quantizer, and a
//! later frame may keep that segment map.

#![forbid(unsafe_code)]

mod bits;
mod cdef;
mod deblock;
mod decode;
mod dims;
mod encode;
mod grain;
mod header;
mod idct;
mod inter;
mod palette;
mod predict;
mod qm;
mod restore;
mod superres;
mod symbol;
mod tables;
mod tile;
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

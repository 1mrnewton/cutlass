//! H.264 / AVC, progressive 4:2:0 at 8 through 14 bits.
//!
//! The decoder follows Recommendation ITU-T H.264 (08/2021) clauses 7, 8, and 9.
//! The encoder chooses prediction and rate control; the bitstream it emits is
//! the same syntax, so this decoder reconstructs it.

#![forbid(unsafe_code)]

mod bits;
mod cabac;
mod cavlc;
mod dec;
mod enc;
mod nal;
mod recon;
mod set;
mod tables;

pub use dec::{H264Decoder, register_decoders};
pub use enc::{H264Encoder, register_encoders};

#[cfg(test)]
mod depth;
#[cfg(test)]
mod high10;
#[cfg(test)]
mod uhd420;

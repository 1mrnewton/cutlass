//! H.264 / AVC, 4:2:0, 4:2:2, and 4:4:4 at 8 through 14 bits, including
//! separate colour planes, field pictures, macroblock-adaptive frames, slice groups,
//! data partitions, SP and SI slices, picture timing, film grain, and Annex E
//! sample aspect, timing, colour tags, and Annex A picture sizes.
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
mod fmo;
mod grain;
mod nal;
mod recon;
mod sei;
mod set;
mod sp;
mod tables;

pub use dec::{H264Decoder, register_decoders};
pub use enc::{H264Encoder, register_encoders};

#[cfg(test)]
mod annex_a;
#[cfg(test)]
mod annex_e;
#[cfg(test)]
mod chroma;
#[cfg(test)]
mod depth;
#[cfg(test)]
mod fields;
#[cfg(test)]
mod high10;
#[cfg(test)]
mod lossless;
#[cfg(test)]
mod part;
#[cfg(test)]
mod planes;
#[cfg(test)]
mod sei_pic;
#[cfg(test)]
mod uhd420;

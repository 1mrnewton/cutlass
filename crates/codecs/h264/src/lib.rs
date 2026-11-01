//! H.264 / AVC, 4:2:0, 4:2:2, and 4:4:4 at 8 through 14 bits, including
//! separate colour planes, field pictures, macroblock-adaptive frames, slice groups,
//! data partitions, SP and SI slices, picture timing, film grain, Annex E
//! sample aspect, timing, colour tags, and Annex A picture sizes.
//! A slice extension with no inter-layer prediction is reconstructed.
//! Same-size inter-layer intra prediction copies the reference layer and adds a coded residual.
//! Same-size inter-layer motion prediction inherits list-0 motion.
//! The reference layer's list-0 residual, including a nonzero vector, is added to an enhancement that codes its own motion.
//! An MVC non-base view that is intra is reconstructed.
//! An Annex I depth view that is intra is reconstructed.
//! Base mode together with residual prediction, resampling, a list other than
//! list 0 at reference index 0, inter-view prediction, and a 3D-AVC extension
//! header are refused.
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
mod annex_g;
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

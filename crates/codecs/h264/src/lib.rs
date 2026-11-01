//! H.264 / AVC, 4:2:0, 4:2:2, and 4:4:4 at 8 through 14 bits, including
//! separate colour planes, field pictures, macroblock-adaptive frames, slice groups,
//! data partitions, SP and SI slices, picture timing, film grain, Annex E
//! sample aspect, timing, colour tags, and Annex A picture sizes.
//! A slice extension with no inter-layer prediction is reconstructed.
//! Same-size inter-layer intra prediction copies the reference layer and adds a coded residual.
//! Same-size inter-layer motion prediction inherits list-0 motion.
//! The reference layer's list-0 residual, including a nonzero vector, is added to an enhancement that codes its own motion.
//! Base mode and that residual can both apply, and the enhancement can add a coded residual of its own.
//! List 0 can use a reference index other than 0, and a reference macroblock can use list 1 or both lists.
//! An MVC non-base view that is intra is reconstructed.
//! A non-base view can predict from the base view at the same order count.
//! An Annex I depth view that is intra is reconstructed.
//! An inter-layer macroblock, intra or inter, can add an 8×8 residual.
//! `adaptive_base_mode_flag` selects base mode per macroblock.
//! `adaptive_motion_prediction_flag` uses the reference layer's vector as the predictor.
//! `adaptive_residual_prediction_flag` selects residual prediction per macroblock.
//! Picture-level intra resampling upsamples the reference layer with the 16-phase intra filter.
//! A zero-offset inter base layer scales its motion onto the enhancement.
//! Its residual is resampled with the transform-block residual filter.
//! A non-negative crop insets that window, and motion inside it uses the window size.
//! Residual resampling uses that window. A negative offset shifts it.
//! A crop that differs on the reference picture adds that difference to the scaled vector.
//! Macroblock-level spatial scalability reads that crop from the slice.
//! An intra 3D-AVC picture is reconstructed. A predicted depth picture copies
//! its reference when `depth_weighted_pred_flag` is 0. A different depth range
//! scales that luma. An intra depth slice can copy its header from the texture
//! slice. A one-macroblock texture P picture can use the base view's motion,
//! including an 8×8, 8×4, 4×8, or 4×4 partition, and a coded difference is added to that predictor.
//! The other 3D tools stay refused.
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

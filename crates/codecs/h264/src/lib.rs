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
//! slice. A one-macroblock texture P or B picture can use the base view's motion,
//! including an 8×8, 8×4, 4×8, or 4×4 partition, and a coded difference is added to that predictor.
//! A one-macroblock texture P picture copies the base texture at a disparity of 32 when
//! `slice_vsp_flag` and `bvsp_flag` are 1. The scale is 0 and the offset is 2016, so that
//! picture is luma 180 and chroma 90. A missing base depth picture is an error.
//! Scale 3968 and offset 0 read the depth sample 64 at the far corner, so the
//! picture shifts four pixels: four columns stay 100 and the rest are 180.
//! A B_L0_16×16 picture with the same flags and the offset disparity copies
//! that texture as luma 180 and chroma 90.
//! A P_L0_L0_16×8 picture samples each partition: the near half stays the
//! split and the far half shifts four pixels.
//! A P_L0_L0_8×16 picture reads a depth sample the 16×16 corners miss, so the
//! left half shifts four pixels and the right half stays 180.
//! A P_8×8 picture reads a depth sample only its top-left block includes, so
//! that block shifts four pixels and the other three stay still.
//! An 8×4 sub-partition of that block uses the 8×8 corners, so both halves
//! shift. A 4×8 sub-partition uses that same 8×8, so both columns shift.
//! A 4×4 sub-partition uses that same 8×8, so all four blocks shift.
//! A skipped P macroblock with `mb_skip_type_flag` 1 is four 8×8 disparities,
//! so that same top-left block shifts and the other three stay still.
//! A skipped B macroblock with that flag uses the same four disparities on
//! list 0.
//! A 32×16 picture samples each macroblock: the left block shifts four pixels
//! and the right block stays the split.
//! A depth parameter set deviation of 15 maps depth 64 to 81, so that scale
//! shifts five pixels: three columns stay 100 and the rest are 180.
//! A 32×16 skipped texture picture copies each base macroblock: the left
//! vector is zero, so that half stays 40 and 220, and the right vector of
//! eight pixels makes that half 220.
//! A 32×16 coded texture picture adds a zero difference to each of those
//! vectors, so the left half stays 40 and 220 and the right half is 220.
//! A 32×32 texture picture reads a 16×16 depth picture. Depth 64 is only at
//! sample (7, 7). The multipliers are 1 and the shifts are 1, so the top-left
//! macroblock shifts four pixels: four columns stay 100, eight are 180, and
//! four are 100. The other three macroblocks stay eight columns of 100 and
//! eight of 180. Chroma stays 90. The depth picture is chroma 40. A depth
//! width that is not half the texture is an error.
//! A 48×32 texture picture scales one macroblock from its neighbors. The
//! reference is luma 100 and the neighbors are luma 80, so the top 16×8
//! becomes 80. The bottom 16×8 meets the picture edge, so that half is luma 0.
//! Chroma stays 90.
//! A 48×16 B picture copies its reference from one skip bin. That reference is
//! luma 40, then 180, then 40, and chroma 90. A skip flag on every macroblock
//! runs out of bins.
//! A same-size enhancement that codes no residual keeps the base QP. The
//! rescaled DC level stays 1, so the top-left 4×4 stays 103, the rest of the
//! luma stays 100 and 180, and chroma stays 90. The slice QP is 22. An 8×8
//! enhancement at QP 22 rescales the base's 8×8 DC level of 1. The base is flat
//! luma 150 and chroma 90, so its top-left 8×8 is 152. The enhancement's
//! bottom-right 8×8 codes a DC level of 1 and is 151. The rescaled top-left
//! 8×8 is 151. The other two blocks stay 150. Chroma stays 90. A base chroma
//! DC level of 1 at QP 26 makes Cb 92 and leaves Cr 90. An enhancement that
//! codes an empty luma residual at QP 22 rescales that chroma level. Cb becomes
//! 91 and the top-left 4×4 becomes 102. FFmpeg emits the base and does not
//! decode the enhancement.
//! A depth P slice copies a later texture header. That source is intra, so the
//! slice uses the picture's one reference. Motion of eight pixels makes that
//! picture luma 220 and chroma 50. The texture picture between them is luma 10
//! and chroma 20.
//! Sources 1 and 3, and a texture inter slice, stay refused.
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
mod subpel;
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
mod subpart;
#[cfg(test)]
mod uhd420;

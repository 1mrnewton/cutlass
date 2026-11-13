//! Kernels of the HEVC decoder (`cutlass-hevc`), for 8-bit samples on
//! x86-64 with AVX2:
//!
//! - [`mc`]: fractional sample interpolation and default and explicit
//!   weighted prediction (clauses 8.5.3.3.3 and 8.5.3.3.4).
//! - [`mc16`]: the same interpolation and default weighting for samples
//!   of 9 to 12 bits.
//! - [`idct`]: the inverse DCT of 4×4 to 32×32 blocks and the 4×4 DST
//!   (clause 8.6.4.2), alone or added to the prediction.
//! - [`deblock`]: the luma deblocking filter of two edge segments
//!   (clause 8.7.2.5.3 and on).
//! - [`sao`]: edge and band offsets of a row (clause 8.7.3), of 8 bits
//!   and of 9 to 12.
//!
//! Each entry returns `false`, having written nothing, when it does not
//! run on this CPU or does not take its input; the decoder then does the
//! work itself.

pub mod deblock;
pub mod idct;
pub mod mc;
pub mod mc16;
pub mod sao;

//! Native HEVC (H.265) video decoder.
//!
//! It takes what an MP4 or QuickTime track holds (the `hvcC` payload as
//! extradata and length-prefixed samples) or Annex B byte streams, and
//! returns pictures in output order, cropped by the conformance window.
//!
//! Support:
//!
//! | | |
//! |---|---|
//! | Profiles | Main, Main 10, Main Still Picture; 4:2:0 at 8 to 12 bits with equal luma and chroma depth |
//! | Coding tools | every version 1 tool: AMP, TMVP, weighted prediction, PCM, transform skip, scaling lists, transquant bypass, sign hiding, constrained intra |
//! | Loop filters | deblocking with slice and PPS overrides, SAO |
//! | Picture partitions | slices, dependent slice segments, tiles, wavefronts, entry points |
//! | Reference handling | RPS with inter-RPS prediction, long-term pictures, list modification, missing-reference generation, RASL skipping, C.5.2 bumping |
//! | Not supported | 4:0:0, 4:2:2, 4:4:4, range-extension tools, multilayer, 3D, screen content: refused as `Unsupported` |
//!
//! Limits: at most 16888 samples on an edge and 35 651 584 luma samples
//! (level 6.2), 16 pictures in the buffer, 4096 NAL units per access unit.
//! Larger pictures return `CodecError::Limit`.
//!
//! A high-bit-depth stream returns the 8-bit picture (each sample shifted
//! down) plus `yuv420p10` for 10 bits or `yuv420_high` for 9, 11 and 12.

#![forbid(unsafe_code)]

mod bits;
mod cabac;
mod ctu;
mod dec;
mod filter;
mod inter;
mod intra;
mod mc;
mod nal;
mod picture;
mod ps;
mod slice;
mod transform;

pub use dec::{HevcDecoder, register_decoders};

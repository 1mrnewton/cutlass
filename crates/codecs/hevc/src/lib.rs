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
//! | Profiles | Main, Main 10, Main Still Picture, and the format range extensions profiles with none of their tools on: 4:2:0, 4:2:2 and 4:4:4 at 8 to 12 bits, and 4:0:0, with equal luma and chroma depth |
//! | Coding tools | every version 1 tool: AMP, TMVP, weighted prediction, PCM, transform skip, scaling lists, transquant bypass, sign hiding, constrained intra |
//! | Loop filters | deblocking with slice and PPS overrides, SAO |
//! | Picture partitions | slices, dependent slice segments, tiles, wavefronts, entry points |
//! | Reference handling | RPS with inter-RPS prediction, long-term pictures, list modification, missing-reference generation, RASL skipping, C.5.2 bumping |
//! | Threads | several pictures at once, each waiting only for the reference rows it reads; the CTB rows of a wavefront slice segment on threads of their own, each two CTBs behind the row above; deblocking and SAO a CTB row behind reconstruction; output identical for any thread count |
//! | Not supported | separate colour planes, odd-sized 4:0:0, range-extension tools, multilayer, 3D, screen content: refused as `Unsupported` |
//!
//! Limits: at most 16888 samples on an edge and 35 651 584 luma samples
//! (level 6.2), 16 pictures in the buffer, 4096 NAL units per access unit.
//! Larger pictures return `CodecError::Limit`.
//!
//! [`HevcDecoder::push_pictures`], and the decoder [`register_decoders`]
//! gives the engine, return each picture as one buffer at the depth and in
//! the chroma format of the stream (`Yuv420p`, `Yuv420p10`, `Yuv420p12`,
//! and the `Yuv422p` and `Yuv444p` formats of the same depths; 4:0:0 as
//! 4:2:0 with mid-grey chroma) with the range, colour and
//! timing of the VUI and the mastering display, light level, preferred
//! transfer and T.35 messages (HDR10+) of its SEI.
//!
//! As a [`cutlass_decode::VideoDecoder`], a high-bit-depth stream returns
//! the 8-bit picture (each sample shifted down) plus `yuv420p10` for 10
//! bits or `yuv420_high` for 9, 11 and 12. A 4:2:2 or 4:4:4 stream fills
//! `yuv422`/`yuv444` (8 bits) or `yuv422_high`/`yuv444_high` (deeper), and
//! its 8-bit `picture` is a 4:2:0 view: luma cut to an even size, chroma
//! the top-left sample of each pair.
//!
//! The vector kernels of 8-bit pictures (interpolation, inverse transforms,
//! deblocking, SAO) are in `cutlass-simd`; this crate has no `unsafe`.

#![forbid(unsafe_code)]

mod bits;
mod cabac;
mod ctu;
mod dec;
mod filter;
mod frame;
mod inter;
mod intra;
mod job;
mod mc;
mod nal;
mod picture;
mod pixel;
mod ps;
mod sei;
mod slice;
mod transform;

pub use dec::{HevcDecoder, HevcPicture, register_decoders};
pub use sei::PicTiming;

//! AV1 encoder and decoder.
//!
//! The support record is `STATUS.md` next to this crate. A still picture may use the full sequence header. A frame with loop-filter sharpness 7 matches dav1d. Loop restoration and
//! film grain are applied for 8-bit 4:2:0. Self-guided restoration also matches on 8-bit 4:4:4 and 4:2:2 when a chroma plane has more than one unit. A 4:2:0 chroma restoration stripe is 32 samples tall. 10-bit restoration matches on 4:4:4, including switchable chroma, and on 4:2:2. A 512×64 10-bit frame has a switchable luma plane. 12-bit self-guided luma restoration matches on 4:2:0. A 512×128 12-bit 4:2:0 frame, a 512×128 12-bit 4:4:4 frame, and a 256×512 12-bit 4:2:2 frame are Wiener on every plane. A restoration frame keeps the updated type CDF for the next unit, including a Wiener frame with four units, and the next frame loads that saved Wiener CDF. A block wider than 64 copies its CDEF index across each 64×64 unit. A 128×128 frame uses CDEF damping 6. 8-bit and 10-bit film grain is also applied for 4:4:4 and 4:2:2. A 128×128 frame uses a lag-3 grain filter and does not clip to the restricted range. Another 128×128 frame uses a grain sample shift of 1. Another 128×128 frame uses a grain scaling shift of 10. A 128×128 4:4:4 frame with an identity matrix clips restricted grain chroma to the luma limit. A 128×128 frame leaves the intra edge filter off. A single-reference frame may carry
//! an order hint. A frame may use a uniform or non-uniform tile grid, a quantizer matrix,
//! and a superblock quantizer delta. A 128×128 keyframe and inter frame add 2 to the chroma DC and AC quantizers. DC luma may use the recursive intra
//! filter. Filter intra reconstructs a luma block no larger than 32.
//! A profile 1 sequence is published as 4:4:4 at 8-bit and at 10-bit. An 8-bit
//! profile 2 sequence is published as 4:2:2 at 8-bit and at 10-bit. An 8-bit
//! or 10-bit monochrome sequence is published as one gray
//! plane. A
//! 10-bit 4:2:0 sequence is published as little-endian samples, including a
//! frame that is upscaled after reconstruction, a keyframe restored by Wiener
//! or self-guided filtering, and film grain on the shown picture. A 12-bit
//! profile 2 picture is little-endian samples for 4:2:0, 4:2:2, and 4:4:4.
//! Film grain at 12-bit is refused, and so is 12-bit monochrome. A frame with `base_q_idx` 0 uses a
//! 4×4 Walsh-Hadamard transform. A 256×256 keyframe and inter frame use the reduced transform set. A lossless segment inside a lossy frame uses that transform. An inter frame may signal LAST and GOLDEN and derive the other five reference slots from the stored order hints. A palette block and an intra block copy are
//! reconstructed, and an inter frame may be forced to integer motion vectors.
//! Two references may be
//! averaged, weighted by order-hint distance, or blended with a wedge or a
//! difference mask. A skip-mode block uses the nearest compound pair and
//! writes no residual. A local warp fits the neighboring motion, and overlapped
//! motion blends those neighboring predictors. A block may take a motion
//! candidate from the field projected out of an earlier frame. A 128×128 inter frame leaves reference motion vectors off while order hint stays on. An inter frame
//! may sample a reference of a different size, and it may copy its upscaled
//! size from a reference. A global-motion block on a scaled reference uses
//! the scaled motion filter. A 3840×2160 inter frame uses the eight-tap sharp filter. A 64×64 frame samples a 32×32 reference with the eight-tap smooth filter, and the next frame uses eight-tap sharp. A
//! single-reference block may blend intra through a luma wedge. A frame
//! may replace a loop-filter reference or mode delta, and a later frame keeps it.
//! A superblock may add its own loop-filter delta. A sequence may use a
//! 128×128 superblock. A 3840×2160 inter frame on that superblock reads a
//! quantizer delta for a skipped block smaller than the superblock, and a
//! motion candidate or a warp sample in another tile is unavailable. A switch
//! frame is predicted as an inter frame and refreshes every reference slot.
//! A reserved OBU is skipped. An OBU with no
//! size field runs to the end of the packet. A frame header may be followed by
//! tile groups. A large-scale tile list copies tiles from the eight stored
//! references and does not refresh them. Film grain may be copied from a
//! reference, and the stored picture is the one from before grain. Every shown
//! frame of one real 8-bit 4:2:0 file matches an independent decoder. A packet
//! may be one Annex B temporal unit. A segment may change the quantizer, and a
//! later frame may keep that segment map. Colour primaries, transfer, matrix,
//! chroma sample position, and the display size are published on the picture. A 128×128 frame publishes a vertical chroma position.
//! The sample buffer stays the frame size. Timing info is kept. Decoder-model
//! info is parsed and the samples still decode. The schedule is not run.
//! `seq_level_idx` is stored. A picture inside the largest defined level
//! decodes, including an edge above 4096. A picture past every defined level
//! is refused before the frame is allocated.

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
mod sample;
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

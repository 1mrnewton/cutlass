//! HEVC for decoders that reconstruct pictures but do not read bitstreams:
//! VA-API, Vulkan Video and the like.
//!
//! [`StatelessDecoder`] does everything but reconstruction, with the code
//! [`crate::HevcDecoder`] runs: it splits access units into NAL units,
//! keeps the parameter sets, parses slice segment headers, derives picture
//! order counts, applies reference picture sets (generating missing
//! references), builds the reference picture lists with their
//! modification, skips RASL pictures that cannot be decoded, and keeps the
//! decoded picture buffer, bumping pictures out in display order (clause
//! C.5.2). For each picture it hands an [`Accelerator`] a [`Picture`]: the
//! parameter sets, the order count, the references and every slice segment
//! with its parsed header and coded bytes, which is what VA-API's picture,
//! slice and matrix buffers and Vulkan Video's picture information hold.
//!
//! It refuses with [`CodecError::Unsupported`] exactly the streams the
//! software decoder refuses, and the accelerator may refuse more in
//! [`Accelerator::sequence`].
//!
//! The parameter set and header types name their fields after the syntax
//! elements of H.265 clause 7.3, holding the coded value or, when not
//! coded, the inferred one (clause 7.4); derived variables are methods.

use std::ops::Range;
use std::sync::Arc;

use cutlass_decode::{CodecError, CodecId, CodecParams, Packet};

pub use crate::front::RpsSet;
use crate::front::{Backend, Bumped, Front, SliceIn, Start};
pub use crate::ps::{Pcm, Pps, ProfileTierLevel, ScalingList, Sps, StRps, Vui};
pub use crate::slice::{LongTerm, PredWeight, PredWeightTable, SliceHeader, SliceType};

/// What reconstructs the pictures: a hardware decoder that takes parsed
/// headers.
pub trait Accelerator {
    /// What a decoded picture lives in, such as an `Arc` around a GPU
    /// surface. The decoded picture buffer holds clones of it while the
    /// picture may be referenced or waits for output.
    type Surface: Clone;

    /// The SPS the pictures use from here on has changed: the first
    /// picture, or a new active SPS (one whose content differs). Return
    /// [`CodecError::Unsupported`] to refuse the stream, for example a
    /// profile, bit depth or size the hardware lacks.
    fn sequence(&mut self, sps: &Sps) -> Result<(), CodecError>;

    /// A surface standing in for a reference picture the stream lacks
    /// (clause 8.3.3), of order count `poc`. It is never output; mid-grey
    /// is what the software decoder uses.
    fn missing(&mut self, sps: &Sps, poc: i32) -> Result<Self::Surface, CodecError>;

    /// Decode one whole picture, all its slice segments, and return the
    /// surface it is in.
    fn decode(&mut self, picture: &Picture<'_, Self::Surface>)
    -> Result<Self::Surface, CodecError>;
}

/// One picture to decode.
pub struct Picture<'a, S> {
    pub sps: &'a Sps,
    pub pps: &'a Pps,
    /// `PicOrderCntVal`.
    pub poc: i32,
    /// `nal_unit_type` of the picture's slice segments: IDR, CRA, BLA (all
    /// IRAP, 16 to 23), RASL, RADL, and so on.
    pub nal_unit_type: u8,
    /// `colWidth[i]`: the width of each tile column in CTBs, derived when
    /// `uniform_spacing_flag` is 1 too. One column without tiles.
    pub column_widths: &'a [u32],
    /// `rowHeight[j]`: the height of each tile row in CTBs.
    pub row_heights: &'a [u32],
    /// Every picture of the decoded picture buffer marked as used for
    /// reference, short or long term, once the current picture's reference
    /// picture set is applied: those of the current picture's lists and
    /// those kept for later pictures. At most 16 (15 in a conforming
    /// stream).
    pub refs: &'a [Reference<'a, S>],
    /// The slice segments, in decoding order.
    pub slices: &'a [Slice<'a>],
}

/// A reference picture of the current one.
pub struct Reference<'a, S> {
    pub surface: &'a S,
    /// `PicOrderCntVal` of the reference.
    pub poc: i32,
    /// Marked as used for long-term reference.
    pub long_term: bool,
    /// Which list of the current picture's reference picture set holds it.
    pub set: RpsSet,
}

/// One slice segment of a picture.
pub struct Slice<'a> {
    /// The header. A dependent slice segment's holds the values it
    /// inherits from the independent one before it, with its own
    /// `dependent_slice_segment_flag`, `slice_segment_address` and entry
    /// points.
    pub header: &'a SliceHeader,
    /// The whole NAL unit as the stream carries it: the two-byte NAL unit
    /// header and emulation prevention bytes included, no start code or
    /// length prefix.
    pub nal: &'a [u8],
    /// Bits from the first bit of the NAL unit header to the end of
    /// `slice_segment_header()`, `byte_alignment()` included, counted with
    /// emulation prevention bytes removed: where `slice_segment_data()`
    /// starts in the RBSP. Always a multiple of 8.
    pub header_bits: u32,
    /// Byte offset in [`Self::nal`] of the first byte of
    /// `slice_segment_data()`, emulation prevention bytes counted (VA-API's
    /// `slice_data_byte_offset`). Less `header_bits / 8`, it is the number
    /// of emulation prevention bytes in the header.
    pub data_offset: usize,
    /// Bits of the `short_term_ref_pic_set()` coded in the slice header (of
    /// the independent slice segment, for a dependent one); 0 when
    /// `short_term_ref_pic_set_sps_flag` is 1 or the picture is IDR.
    pub st_rps_bits: u32,
    /// `RefPicList0` and `RefPicList1` after initialisation and
    /// modification (clause 8.3.4), as indices into [`Picture::refs`]:
    /// `num_ref_idx_lX_active_minus1 + 1` entries for each list the slice
    /// type uses, none for the others.
    pub ref_lists: [Vec<usize>; 2],
}

/// A decoded picture, in display order.
pub struct Output<S> {
    pub surface: S,
    /// The timing and random-access flag of the packet that held the
    /// picture.
    pub pts: i64,
    pub duration: i64,
    pub timescale: u32,
    /// The picture is IRAP.
    pub keyframe: bool,
    /// The picture's SPS: [`Sps::crop`] is the conformance window to cut
    /// the surface to, and [`Sps::vui`] the colour description.
    pub sps: Arc<Sps>,
}

/// HEVC with reconstruction done by an [`Accelerator`].
///
/// Pushes, drains and resets behave as [`crate::HevcDecoder`]'s do, and its
/// pictures come out in the same order with the same timing.
pub struct StatelessDecoder<A: Accelerator> {
    length_size: u8,
    extra: Vec<u8>,
    front: Front<A::Surface>,
    accelerator: A,
    drained: bool,
}

impl<A: Accelerator> StatelessDecoder<A> {
    /// A decoder of `params`: an `hvcC` record (whose NAL unit length size
    /// replaces `params.nal_length_size`) or Annex B parameter sets as
    /// extradata, and length-prefixed (`nal_length_size` 1 to 4) or Annex
    /// B (0) packets.
    pub fn open(params: &CodecParams, accelerator: A) -> Result<Self, CodecError> {
        if params.codec != CodecId::Hevc {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        let mut dec = Self {
            length_size: params.nal_length_size,
            extra: params.extra.clone(),
            front: Front::new(),
            accelerator,
            drained: false,
        };
        dec.front.apply_extra(&dec.extra, &mut dec.length_size)?;
        Ok(dec)
    }

    /// Decode one access unit and return the pictures it lets out, in
    /// display order. A push that fails leaves the decoder as it was before
    /// it, though the accelerator may have decoded pictures of it.
    pub fn push(&mut self, packet: &Packet) -> Result<Vec<Output<A::Surface>>, CodecError> {
        if packet.data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        let mut front = self.front.clone();
        let mut out = Vec::new();
        let mut backend = Hw {
            accelerator: &mut self.accelerator,
        };
        front.access_unit(&mut backend, packet, self.length_size, &mut out)?;
        self.front = front;
        self.drained = false;
        Ok(out.into_iter().map(output).collect())
    }

    /// Every picture still waiting for output, in display order. A second
    /// drain with no push between returns nothing.
    pub fn drain(&mut self) -> Result<Vec<Output<A::Surface>>, CodecError> {
        if self.drained {
            return Ok(Vec::new());
        }
        self.drained = true;
        let mut out = Vec::new();
        self.front.flush(&mut out);
        Ok(out.into_iter().map(output).collect())
    }

    /// Forget every picture and parameter set but those of the extradata,
    /// to start again at a random access point. The accelerator is told
    /// the SPS of the next picture again.
    pub fn reset(&mut self) {
        self.front = Front::new();
        self.drained = false;
        // The extradata parsed when the decoder opened, so it parses again.
        let _ = self.front.apply_extra(&self.extra, &mut self.length_size);
    }

    /// The accelerator the pictures go to.
    pub fn accelerator(&self) -> &A {
        &self.accelerator
    }

    /// The accelerator, to change. Surfaces the decoded picture buffer
    /// holds stay referenced and may still be output.
    pub fn accelerator_mut(&mut self) -> &mut A {
        &mut self.accelerator
    }
}

fn output<S>((surface, meta): Bumped<S>) -> Output<S> {
    Output {
        surface,
        pts: meta.pts,
        duration: meta.duration,
        timescale: meta.timescale,
        keyframe: meta.keyframe,
        sps: meta.sps.clone(),
    }
}

/// An accelerator as the front end's backend.
struct Hw<'a, A> {
    accelerator: &'a mut A,
}

/// A picture whose slice segments are being read.
struct Building<S> {
    sps: Arc<Sps>,
    pps: Arc<Pps>,
    poc: i32,
    nal_unit_type: u8,
    column_widths: Vec<u32>,
    row_heights: Vec<u32>,
    refs: Vec<(S, i32, bool, RpsSet)>,
    slices: Vec<SliceRecord>,
}

struct SliceRecord {
    header: SliceHeader,
    /// Where the NAL unit is in the packet.
    span: Range<usize>,
    header_bits: u32,
    data_offset: usize,
    st_rps_bits: u32,
    ref_lists: [Vec<usize>; 2],
}

impl<A: Accelerator> Backend for Hw<'_, A> {
    type Pic = A::Surface;
    type Building = Building<A::Surface>;

    fn sequence(&mut self, sps: &Arc<Sps>) -> Result<(), CodecError> {
        self.accelerator.sequence(sps)
    }

    fn missing(&mut self, sps: &Arc<Sps>, poc: i32) -> Result<A::Surface, CodecError> {
        self.accelerator.missing(sps, poc)
    }

    fn start(&mut self, start: Start<'_, A::Surface>) -> Result<Self::Building, CodecError> {
        let Start {
            sps,
            pps,
            nal,
            poc,
            refs,
            ..
        } = start;
        Ok(Building {
            sps: sps.clone(),
            pps: pps.clone(),
            poc,
            nal_unit_type: nal.kind,
            column_widths: pps.column_widths(sps)?,
            row_heights: pps.row_heights(sps)?,
            refs: refs
                .iter()
                .map(|r| (r.pic.clone(), r.poc, r.long_term, r.set))
                .collect(),
            slices: Vec::new(),
        })
    }

    fn slice(&mut self, building: &mut Self::Building, slice: SliceIn) -> Result<(), CodecError> {
        let SliceIn {
            nal,
            header,
            data_offset,
            st_rps_bits,
            ref_lists,
            ..
        } = slice;
        let Some(header_bits) = data_offset
            .checked_add(2)
            .and_then(|bytes| u32::try_from(bytes).ok())
            .and_then(|bytes| bytes.checked_mul(8))
        else {
            return Err(CodecError::Limit {
                what: "slice segment header",
            });
        };
        building.slices.push(SliceRecord {
            header,
            span: nal.span.clone(),
            header_bits,
            data_offset: 2 + nal.raw_of(data_offset),
            st_rps_bits,
            ref_lists,
        });
        Ok(())
    }

    fn finish(&mut self, building: Self::Building, data: &[u8]) -> Result<A::Surface, CodecError> {
        let refs: Vec<Reference<'_, A::Surface>> = building
            .refs
            .iter()
            .map(|(surface, poc, long_term, set)| Reference {
                surface,
                poc: *poc,
                long_term: *long_term,
                set: *set,
            })
            .collect();
        let mut slices = Vec::with_capacity(building.slices.len());
        for record in &building.slices {
            let Some(nal) = data.get(record.span.clone()) else {
                return Err(CodecError::InvalidApi {
                    what: "slice segment outside its packet",
                });
            };
            slices.push(Slice {
                header: &record.header,
                nal,
                header_bits: record.header_bits,
                data_offset: record.data_offset,
                st_rps_bits: record.st_rps_bits,
                ref_lists: record.ref_lists.clone(),
            });
        }
        self.accelerator.decode(&Picture {
            sps: &building.sps,
            pps: &building.pps,
            poc: building.poc,
            nal_unit_type: building.nal_unit_type,
            column_widths: &building.column_widths,
            row_heights: &building.row_heights,
            refs: &refs,
            slices: &slices,
        })
    }
}

#[cfg(test)]
mod tests;

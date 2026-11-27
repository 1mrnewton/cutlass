//! A front end for decoders that reconstruct H.264 pictures elsewhere, such
//! as VA-API, which take a picture's parameter sets, slice headers and
//! reference pictures and leave everything else to the caller.
//!
//! [`StatelessDecoder`] splits each packet into NAL units, keeps the
//! parameter sets, reads every slice header, and does what the software
//! decoder does around reconstruction, with the same code: picture order
//! counts of all three types (clause 8.2.1), `frame_num` gaps (8.2.5.2),
//! reference marking by sliding window and by
//! `memory_management_control_operation` (8.2.5), reference list
//! initialisation and modification (8.2.4), and a decoded picture buffer
//! that hands pictures out in display order. What it cannot do, turning a
//! picture's slices into samples, it asks of an [`Accelerator`], whose
//! surfaces it keeps in the buffer and hands back as [`Output`]s.
//!
//! Its scope is narrower than the software decoder's: progressive 8-bit
//! 4:2:0 Baseline (constrained or not), Main and High pictures without slice
//! groups. Any other sequence or picture parameter set is refused with
//! [`CodecError::Unsupported`]: when [`StatelessDecoder::open`] reads it from
//! an `avcC` record, or else when it arrives in a packet. NAL units of the
//! scalable, multiview and 3D extensions (types 14, 15, 16, 20 and 21) are
//! skipped, so their base layer or base view is decoded. Redundant slices
//! are skipped, as the software decoder skips them. Each packet is one
//! access unit, as for [`crate::H264Decoder`].
//!
//! Two choices of the software decoder hold here too, as the code is the
//! same: a frame inferred for a `frame_num` gap is a short-term reference
//! that slides older ones out but is never put in a reference list, and a
//! picture with `memory_management_control_operation` 5 keeps its order
//! counts.

use std::sync::Arc;

use cutlass_decode::{CodecError, CodecId, CodecParams, Packet};

use crate::bits::Bits;
use crate::dpb::{self, NO_PICTURE, Stored};
use crate::header::{self as syntax, Context};
use crate::nal::{self, Nal};
use crate::poc::{self, Poc};
use crate::sei;
use crate::set::{self, MAX_PPS, MAX_SPS};

pub use crate::header::{
    DecRefPicMarking, MemoryManagementControlOperation, PredWeight, PredWeightTable,
    RefPicListModification, SliceHeader,
};
pub use crate::set::{Pps, Sps, Vui};

/// What reconstructs pictures for a [`StatelessDecoder`].
///
/// Calls come in decoding order. An error from any of them fails the
/// [`StatelessDecoder::push`] that made the call, and the decoder is left as
/// it was before that push: surfaces made during it are then dropped by the
/// decoder, and the calls are made again when the packet is pushed again.
pub trait Accelerator {
    /// A decoded picture. The decoder keeps a clone for as long as the
    /// picture is a reference or waits to be output.
    type Surface: Clone;

    /// A coded video sequence begins with `sps`. Called before the first
    /// picture after [`StatelessDecoder::open`] or
    /// [`StatelessDecoder::reset`], and before each picture whose sequence
    /// parameter set differs from the previous picture's. Pictures made for
    /// the earlier sequence may still be output after it.
    fn sequence(&mut self, sps: &Sps) -> Result<(), CodecError>;

    /// A frame for the missing `frame_num` of a gap the sequence allows
    /// (clause 8.2.5.2). It is a short-term reference that is never output
    /// and never in a reference list, so its samples matter only to a
    /// stream that is broken; a grey or repeated picture serves.
    fn missing(&mut self, sps: &Sps, frame_num: u32) -> Result<Self::Surface, CodecError>;

    /// Reconstructs `picture` and returns the surface that holds it.
    fn decode(&mut self, picture: &Picture<'_, Self::Surface>)
    -> Result<Self::Surface, CodecError>;
}

/// One picture to reconstruct: a progressive frame.
#[derive(Debug)]
pub struct Picture<'a, S> {
    /// The active sequence parameter set.
    pub sps: &'a Sps,
    /// The picture parameter set every slice of the picture names.
    pub pps: &'a Pps,
    /// `frame_num`.
    pub frame_num: u32,
    /// `TopFieldOrderCnt`.
    pub top_field_order_cnt: i32,
    /// `BottomFieldOrderCnt`.
    pub bottom_field_order_cnt: i32,
    /// `IdrPicFlag`.
    pub idr: bool,
    /// `nal_ref_idc` of the picture's slices.
    pub nal_ref_idc: u8,
    /// Every picture of the buffer marked as a reference before this one is
    /// decoded, short-term ones first as they were stored, then long-term
    /// ones. Frames inferred for a `frame_num` gap are among them, marked
    /// `non_existing`.
    pub refs: &'a [Reference<S>],
    /// The picture's slices in decoding order. Redundant slices are not here.
    pub slices: &'a [Slice<'a>],
}

/// A reference picture of a [`Picture`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reference<S> {
    pub surface: S,
    /// Marked "used for long-term reference".
    pub long_term: bool,
    /// `FrameNum` of a short-term reference, `LongTermFrameIdx` of a
    /// long-term one.
    pub frame_num: u32,
    /// `TopFieldOrderCnt`.
    pub top_field_order_cnt: i32,
    /// `BottomFieldOrderCnt`.
    pub bottom_field_order_cnt: i32,
    /// The frame was inferred for a `frame_num` gap and its surface came from
    /// [`Accelerator::missing`].
    pub non_existing: bool,
}

/// One slice of a [`Picture`].
#[derive(Clone, Debug)]
pub struct Slice<'a> {
    pub header: &'a SliceHeader,
    /// The whole NAL unit: its header byte and payload, emulation
    /// prevention bytes included, without the start code or length prefix.
    pub nal: &'a [u8],
    /// The bits from the start of the NAL unit header to the end of
    /// `slice_header()`, counted in the RBSP (with the emulation prevention
    /// bytes removed), before any `cabac_alignment_one_bit`.
    pub header_bits: u32,
    /// `RefPicList0` and `RefPicList1` after initialisation and
    /// modification, as indices into [`Picture::refs`]. Empty for an I
    /// slice, and list 1 for a P slice. A list is shorter than
    /// `num_ref_idx_lX_active_minus1 + 1` when the buffer holds too few
    /// pictures: the entries past its end are "no reference picture".
    pub ref_lists: [Vec<usize>; 2],
}

/// A decoded picture whose turn it is to be displayed.
#[derive(Clone, Debug)]
pub struct Output<S> {
    pub surface: S,
    /// Display time in ticks of `timescale`: the packet's, moved by a
    /// picture timing message as the software decoder moves it.
    pub pts: i64,
    pub duration: i64,
    pub timescale: u32,
    /// An IDR picture, or one whose first slice is I or SI.
    pub keyframe: bool,
    /// The sequence parameter set the picture was decoded with: its crop,
    /// colour description and sample aspect.
    pub sps: Arc<Sps>,
}

/// The H.264 decoding process around an [`Accelerator`]. See the module
/// documentation.
pub struct StatelessDecoder<A: Accelerator> {
    accelerator: A,
    state: State<A::Surface>,
    /// 0 for Annex B, else the bytes of each NAL unit's length.
    length_size: u8,
    extra: Vec<u8>,
}

#[derive(Clone)]
struct State<S> {
    sps: Vec<Option<Arc<Sps>>>,
    pps: Vec<Option<Arc<Pps>>>,
    /// The sequence parameter set the last picture was decoded with.
    active: Option<Arc<Sps>>,
    poc: Poc,
    dpb: Vec<Entry<S>>,
    drained: bool,
}

impl<S> State<S> {
    fn new() -> Self {
        Self {
            sps: vec![None; MAX_SPS],
            pps: vec![None; MAX_PPS],
            active: None,
            poc: Poc::new(),
            dpb: Vec::new(),
            drained: false,
        }
    }
}

/// A picture in the buffer.
#[derive(Clone, Debug)]
struct Entry<S> {
    surface: S,
    frame_num: u32,
    poc: i32,
    fields: [i32; 2],
    referenced: bool,
    long_term: bool,
    long_idx: i32,
    output: bool,
    nonexisting: bool,
    max_fn: u32,
    max_ref: usize,
    size: (usize, usize),
    pts: i64,
    duration: i64,
    timescale: u32,
    keyframe: bool,
    sps: Arc<Sps>,
}

impl<S> Entry<S> {
    /// The frame clause 8.2.5.2 infers for a missing `frame_num`.
    fn absent(surface: S, sps: &Arc<Sps>, frame_num: u32, poc: i32) -> Self {
        Self {
            surface,
            frame_num,
            poc,
            fields: [poc; 2],
            referenced: true,
            long_term: false,
            long_idx: -1,
            output: false,
            nonexisting: true,
            max_fn: sps.max_frame_num(),
            max_ref: usize::from(sps.max_num_ref_frames).clamp(1, 16),
            size: (sps.width as usize, sps.height as usize),
            pts: 0,
            duration: 0,
            timescale: 0,
            keyframe: false,
            sps: sps.clone(),
        }
    }

    fn shown(self) -> Output<S> {
        Output {
            surface: self.surface,
            pts: self.pts,
            duration: self.duration,
            timescale: self.timescale,
            keyframe: self.keyframe,
            sps: self.sps,
        }
    }
}

impl<S: Clone> Stored for Entry<S> {
    fn frame_num(&self) -> u32 {
        self.frame_num
    }

    fn set_frame_num(&mut self, frame_num: u32) {
        self.frame_num = frame_num;
    }

    fn poc(&self) -> i32 {
        self.poc
    }

    fn size(&self) -> (usize, usize) {
        self.size
    }

    fn nonexisting(&self) -> bool {
        self.nonexisting
    }

    fn referenced(&self) -> bool {
        self.referenced
    }

    fn set_referenced(&mut self, referenced: bool) {
        self.referenced = referenced;
    }

    fn long_term(&self) -> bool {
        self.long_term
    }

    fn long_idx(&self) -> i32 {
        self.long_idx
    }

    fn set_long(&mut self, idx: i32) {
        self.long_term = true;
        self.long_idx = idx;
    }

    fn set_short(&mut self) {
        self.long_term = false;
    }

    fn output(&self) -> bool {
        self.output
    }

    fn set_output(&mut self, output: bool) {
        self.output = output;
    }

    fn max_fn(&self) -> u32 {
        self.max_fn
    }

    fn max_ref(&self) -> usize {
        self.max_ref
    }
}

/// A slice read from the access unit, with what it was read against.
struct Read<'p> {
    header: SliceHeader,
    nal: &'p [u8],
    header_bits: u32,
    sps: Arc<Sps>,
    pps: Arc<Pps>,
    /// Indices into the buffer.
    lists: [Vec<usize>; 2],
}

/// The picture the access unit's slices make.
struct Begun {
    sps: Arc<Sps>,
    pps: Arc<Pps>,
    counts: poc::Counts,
    size: (usize, usize),
}

impl<A: Accelerator> StatelessDecoder<A> {
    /// A decoder for `params`, whose `extra` may be an `avcC` record; other
    /// extra data is ignored, as [`crate::H264Decoder`] ignores it.
    /// `nal_length_size` 0 is Annex B; `avcC` sets its own.
    pub fn open(params: &CodecParams, accelerator: A) -> Result<Self, CodecError> {
        if params.codec != CodecId::H264 {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        let mut state = State::new();
        let length_size = configure(&mut state, &params.extra)?.unwrap_or(params.nal_length_size);
        Ok(Self {
            accelerator,
            state,
            length_size,
            extra: params.extra.clone(),
        })
    }

    /// Decodes one access unit and returns the pictures whose turn it is to
    /// be displayed, in display order. On an error nothing changes: the
    /// decoder is as it was before the call.
    pub fn push(&mut self, packet: &Packet) -> Result<Vec<Output<A::Surface>>, CodecError> {
        if packet.data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        let units = nal::access_unit_with(&packet.data, self.length_size, |raw| {
            Ok((raw, nal::nal_from_ebsp(raw)?))
        })?;
        let mut state = self.state.clone();
        let out = decode_unit(&mut state, &mut self.accelerator, &units, packet)?;
        state.drained = false;
        self.state = state;
        Ok(out.into_iter().map(Entry::shown).collect())
    }

    /// Every picture still waiting, in display order. A second call without
    /// a push between returns none. Decoding may go on after it with a push.
    pub fn drain(&mut self) -> Result<Vec<Output<A::Surface>>, CodecError> {
        if self.state.drained {
            return Ok(Vec::new());
        }
        let out = dpb::take_output(&mut self.state.dpb, 0);
        self.state.drained = true;
        Ok(out.into_iter().map(Entry::shown).collect())
    }

    /// Forgets every picture and parameter set, and reads the `avcC` record
    /// `open` was given again. The next picture begins a sequence.
    pub fn reset(&mut self) {
        let mut state = State::new();
        if let Ok(Some(length_size)) = configure(&mut state, &self.extra) {
            self.length_size = length_size;
        }
        self.state = state;
    }

    pub fn accelerator(&self) -> &A {
        &self.accelerator
    }

    pub fn accelerator_mut(&mut self) -> &mut A {
        &mut self.accelerator
    }
}

/// The parameter sets of an `avcC` record, and its length size. Other
/// extra data is not read.
fn configure<S>(state: &mut State<S>, extra: &[u8]) -> Result<Option<u8>, CodecError> {
    if extra.first() != Some(&1) {
        return Ok(None);
    }
    let (length_size, nals) = nal::avcc(extra)?;
    for nal in &nals {
        match nal.kind {
            7 => store_sps(state, &nal.rbsp)?,
            8 => store_pps(state, &nal.rbsp)?,
            _ => {}
        }
    }
    Ok(Some(length_size))
}

fn store_sps<S>(state: &mut State<S>, rbsp: &[u8]) -> Result<(), CodecError> {
    let sps = set::parse_sps(rbsp)?;
    supported_sequence(&sps)?;
    let id = usize::from(sps.seq_parameter_set_id);
    state.sps[id] = Some(Arc::new(sps));
    Ok(())
}

fn store_pps<S>(state: &mut State<S>, rbsp: &[u8]) -> Result<(), CodecError> {
    let pps = set::parse_pps_with(rbsp, |id| state.sps.get(id).and_then(|sps| sps.as_deref()))?;
    if pps.num_slice_groups_minus1() != 0 {
        return Err(CodecError::Unsupported {
            what: "slice groups",
        });
    }
    let id = usize::from(pps.pic_parameter_set_id);
    state.pps[id] = Some(Arc::new(pps));
    Ok(())
}

/// Progressive 8-bit 4:2:0 Baseline, Main or High.
fn supported_sequence(sps: &Sps) -> Result<(), CodecError> {
    if !matches!(sps.profile_idc, 66 | 77 | 100) {
        return Err(CodecError::Unsupported { what: "profile" });
    }
    if sps.chroma_format_idc != 1 {
        return Err(CodecError::Unsupported {
            what: "chroma format",
        });
    }
    if sps.bit_depth_luma_minus8 != 0 || sps.bit_depth_chroma_minus8 != 0 {
        return Err(CodecError::Unsupported { what: "bit depth" });
    }
    if !sps.frame_mbs_only_flag {
        return Err(CodecError::Unsupported {
            what: "field pictures",
        });
    }
    Ok(())
}

fn decode_unit<A: Accelerator>(
    state: &mut State<A::Surface>,
    accelerator: &mut A,
    units: &[(&[u8], Nal)],
    packet: &Packet,
) -> Result<Vec<Entry<A::Surface>>, CodecError> {
    let mut begun: Option<Begun> = None;
    let mut slices: Vec<Read<'_>> = Vec::new();
    let mut prior = Vec::new();
    let mut messages = Vec::new();
    for (raw, nal) in units {
        match nal.kind {
            7 => store_sps(state, &nal.rbsp)?,
            8 => store_pps(state, &nal.rbsp)?,
            1 | 5 => {
                let mut slice = read_slice(state, raw, nal)?;
                // A redundant picture repeats one that was already decoded.
                if slice.header.redundant_pic_cnt > 0 {
                    continue;
                }
                let picture = match begun.take() {
                    Some(picture) => picture,
                    None => begin(state, accelerator, &slice, &mut prior)?,
                };
                if *picture.pps != *slice.pps {
                    return Err(CodecError::Unsupported {
                        what: "picture parameter sets within a picture",
                    });
                }
                let header = &slice.header;
                let (list0, list1) = dpb::reference_lists(
                    &state.dpb,
                    &dpb::ListSyntax {
                        kind: header.slice_kind(),
                        frame_num: header.frame_num,
                        view_id: 0,
                        active: header.active(),
                        modifications: [
                            &header.ref_pic_list_modification_l0,
                            &header.ref_pic_list_modification_l1,
                        ],
                        interview: [&[], &[]],
                    },
                    picture.counts.poc,
                    picture.sps.max_frame_num(),
                    picture.size,
                )?;
                slice.lists = [list0, list1].map(|list| {
                    list.into_iter()
                        .take_while(|&slot| slot != NO_PICTURE)
                        .collect()
                });
                slices.push(slice);
                begun = Some(picture);
            }
            2..=4 => {
                return Err(CodecError::Unsupported {
                    what: "data partitioning",
                });
            }
            6 => messages.extend(sei::payloads(&nal.rbsp)?),
            9..=12 | 14..=16 | 20 | 21 => {}
            _ => {
                return Err(CodecError::Unsupported {
                    what: "nal unit type",
                });
            }
        }
    }
    let (Some(picture), Some(first)) = (begun, slices.first()) else {
        return Ok(Vec::new());
    };
    let header = &first.header;
    let sps = &picture.sps;
    // The buffer's references in its order, and where each is in `refs`.
    let mut place = vec![usize::MAX; state.dpb.len()];
    let mut refs = Vec::new();
    for long_term in [false, true] {
        for (index, entry) in state.dpb.iter().enumerate() {
            if entry.referenced && entry.long_term == long_term {
                place[index] = refs.len();
                refs.push(Reference {
                    surface: entry.surface.clone(),
                    long_term: entry.long_term,
                    frame_num: if entry.long_term {
                        u32::try_from(entry.long_idx).unwrap_or(0)
                    } else {
                        entry.frame_num
                    },
                    top_field_order_cnt: entry.fields[0],
                    bottom_field_order_cnt: entry.fields[1],
                    non_existing: entry.nonexisting,
                });
            }
        }
    }
    let views: Vec<Slice<'_>> = slices
        .iter()
        .map(|slice| Slice {
            header: &slice.header,
            nal: slice.nal,
            header_bits: slice.header_bits,
            ref_lists: slice
                .lists
                .clone()
                .map(|list| list.into_iter().map(|slot| place[slot]).collect()),
        })
        .collect();
    let surface = accelerator.decode(&Picture {
        sps,
        pps: &picture.pps,
        frame_num: header.frame_num,
        top_field_order_cnt: picture.counts.fields[0],
        bottom_field_order_cnt: picture.counts.fields[1],
        idr: header.idr_pic_flag,
        nal_ref_idc: header.nal_ref_idc,
        refs: &refs,
        slices: &views,
    })?;
    let (pts, duration) = sei::shown_time(
        &messages,
        &sei::TimingClock::of(sps),
        packet.pts,
        packet.duration,
        packet.timescale,
    )?;
    let marking = &header.dec_ref_pic_marking;
    let long_term = marking.long_term_reference_flag;
    let entry = Entry {
        surface,
        frame_num: header.frame_num,
        poc: picture.counts.poc,
        fields: picture.counts.fields,
        referenced: header.nal_ref_idc != 0,
        long_term,
        long_idx: if long_term { 0 } else { -1 },
        output: true,
        nonexisting: false,
        max_fn: sps.max_frame_num(),
        max_ref: usize::from(sps.max_num_ref_frames).clamp(1, 16),
        size: picture.size,
        pts,
        duration,
        timescale: packet.timescale,
        keyframe: header.idr_pic_flag || syntax::intra_slice(header.slice_kind()),
        sps: sps.clone(),
    };
    let operations = marking
        .adaptive_ref_pic_marking_mode_flag
        .then_some(marking.operations.as_slice());
    prior.extend(dpb::store(
        &mut state.dpb,
        entry,
        operations,
        &mut state.poc,
        sps.output_delay(),
    )?);
    Ok(prior)
}

/// The header of a slice NAL unit, read against the parameter sets it names.
fn read_slice<'p, S>(state: &State<S>, raw: &'p [u8], nal: &Nal) -> Result<Read<'p>, CodecError> {
    let mut bits = Bits::new(&nal.rbsp);
    let start = syntax::read_start(&mut bits)?;
    let Some(pps) = state.pps[usize::from(start.pic_parameter_set_id)].clone() else {
        return Err(bits.malformed("missing pps"));
    };
    let Some(sps) = state
        .sps
        .get(usize::from(pps.seq_parameter_set_id))
        .cloned()
        .flatten()
    else {
        return Err(bits.malformed("missing sps"));
    };
    let header = syntax::read_rest(
        &mut bits,
        start,
        &sps,
        &pps,
        Context::annex_a(nal.kind == 5, nal.ref_idc),
    )?;
    // The NAL unit header is one byte for types 1 and 5.
    let header_bits = u32::try_from(8 + bits.pos()).map_err(|_| CodecError::Limit {
        what: "slice header",
    })?;
    Ok(Read {
        header,
        nal: raw,
        header_bits,
        sps,
        pps,
        lists: [Vec::new(), Vec::new()],
    })
}

/// The first slice of a picture: a new sequence, a `frame_num` gap, the
/// order counts, and for an IDR picture the end of every earlier reference.
fn begin<A: Accelerator>(
    state: &mut State<A::Surface>,
    accelerator: &mut A,
    slice: &Read<'_>,
    prior: &mut Vec<Entry<A::Surface>>,
) -> Result<Begun, CodecError> {
    let header = &slice.header;
    let sps = &slice.sps;
    if state.active.as_deref() != Some(&**sps) {
        accelerator.sequence(sps)?;
        state.active = Some(sps.clone());
    }
    let idr = header.idr_pic_flag;
    if !idr {
        dpb::fill_frame_num_gap(
            sps,
            &mut state.poc,
            &mut state.dpb,
            header.frame_num,
            |frame_num, poc| {
                let surface = accelerator.missing(sps, frame_num)?;
                Ok(Entry::absent(surface, sps, frame_num, poc))
            },
        )?;
    }
    let counts = poc::counts(&header.order(sps), sps, &mut state.poc, idr)?;
    if idr {
        let marking = &header.dec_ref_pic_marking;
        prior.extend(dpb::begin_idr(
            &mut state.dpb,
            &mut state.poc,
            marking.long_term_reference_flag,
            marking.no_output_of_prior_pics_flag,
        ));
    }
    if header.nal_ref_idc != 0 {
        state.poc.follow(header.frame_num, &counts);
    }
    Ok(Begun {
        sps: sps.clone(),
        pps: slice.pps.clone(),
        counts,
        size: (sps.width as usize, sps.height as usize),
    })
}

#[cfg(test)]
#[path = "stateless_tests.rs"]
mod tests;

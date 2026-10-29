//! Access-unit decoder. One `push` is one picture: Annex B or length-prefixed NAL units.

use cutlass_decode::{
    CodecError, CodecId, CodecParams, ColorMatrix, ColorPrimaries, ColorRange, DecodeFactory,
    Packet, Transfer, VideoDecoder, VideoOutput, Yuv420p,
};

use crate::bits::Bits;
use crate::cabac::Cabac;
use crate::nal::{self, Nal};
use crate::recon;
use crate::set::{self, Pps, Sps, ZIGZAG4, ZIGZAG8};

#[path = "mb.rs"]
mod mb;

pub(crate) const CBP_INTRA: [u8; 48] = [
    47, 31, 15, 0, 23, 27, 29, 30, 7, 11, 13, 14, 39, 43, 45, 46, 16, 3, 5, 10, 12, 19, 21, 26, 28,
    35, 37, 42, 44, 1, 2, 4, 8, 17, 18, 20, 24, 6, 9, 22, 25, 32, 33, 34, 36, 40, 38, 41,
];
pub(crate) const CBP_INTER: [u8; 48] = [
    0, 16, 1, 2, 4, 8, 32, 3, 5, 10, 12, 15, 47, 7, 11, 13, 14, 6, 9, 31, 35, 37, 42, 44, 33, 34,
    36, 40, 39, 43, 45, 46, 17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41,
];

#[derive(Clone)]
pub(crate) struct Mb {
    slice: u16,
    intra: bool,
    pcm: bool,
    qp: i8,
    qp_c: i8,
    qp_cr: i8,
    cbp: u8,
    transform8: bool,
    /// Every inter partition is at least 8×8, so an 8×8 transform flag may be present.
    partition8: bool,
    skip: bool,
    /// B_Direct_16x16. The CABAC mb_type context treats this the same as a skip.
    direct16: bool,
    intra16: bool,
    nnz_y: [u8; 16],
    nnz_c: [u8; 8],
    mode4: [u8; 16],
    chroma_mode: u8,
    ref0: [i8; 4],
    ref1: [i8; 4],
    /// Decoded-picture-buffer index of each list, or −1 when that list is unused.
    slot0: [i8; 4],
    slot1: [i8; 4],
    mv0: [[i16; 2]; 16],
    mv1: [[i16; 2]; 16],
    /// Picture order count of the reference each 8×8 partition used. Kept so a later
    /// temporal-direct picture can find that reference after the buffer moves.
    ref_poc0: [i32; 4],
    ref_poc1: [i32; 4],
    mvd0: [[i16; 2]; 16],
    mvd1: [[i16; 2]; 16],
    /// 4×4 blocks whose motion vector is already decoded. Same-macroblock predictors ignore the rest.
    ready: u16,
    off_a: i8,
    off_b: i8,
    deblock: u8,
    /// mb_qp_delta that was coded for this macroblock. Zero when the delta is inferred.
    qpd: i8,
    /// coded_block_flag of the Intra_16x16 luma DC block.
    dc_coded: bool,
    /// coded_block_flag of the Cb and Cr DC blocks.
    dc_c: [bool; 2],
}

impl Mb {
    pub(crate) fn new(slice: u16) -> Self {
        Self {
            slice,
            intra: false,
            pcm: false,
            qp: 0,
            qp_c: 0,
            qp_cr: 0,
            cbp: 0,
            transform8: false,
            partition8: true,
            skip: false,
            direct16: false,
            intra16: false,
            nnz_y: [0; 16],
            nnz_c: [0; 8],
            mode4: [2; 16],
            chroma_mode: 0,
            ref0: [-1; 4],
            ref1: [-1; 4],
            slot0: [-1; 4],
            slot1: [-1; 4],
            mv0: [[0; 2]; 16],
            mv1: [[0; 2]; 16],
            ref_poc0: [0; 4],
            ref_poc1: [0; 4],
            mvd0: [[0; 2]; 16],
            mvd1: [[0; 2]; 16],
            ready: 0,
            off_a: 0,
            off_b: 0,
            deblock: 0,
            qpd: 0,
            dc_coded: false,
            dc_c: [false; 2],
        }
    }
}

#[derive(Clone)]
pub(crate) struct Frame {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    mb: Vec<Mb>,
    w: usize,
    h: usize,
    mb_w: usize,
    poc: i32,
    frame_num: u32,
    referenced: bool,
    long_term: bool,
    long_idx: i32,
    output: bool,
    key: bool,
    pts: i64,
    duration: i64,
    timescale: u32,
    crop_x: u32,
    crop_y: u32,
    vis_w: u32,
    vis_h: u32,
    range: ColorRange,
    primaries: ColorPrimaries,
    transfer: Transfer,
    matrix: ColorMatrix,
    nonexisting: bool,
    max_ref: usize,
    reorder: usize,
    adaptive: bool,
    mmco: Vec<Mark>,
    max_fn: u32,
}

#[derive(Clone)]
struct Sets {
    sps: Vec<Option<Sps>>,
    pps: Vec<Option<Pps>>,
}

impl Sets {
    fn new() -> Self {
        Self {
            sps: vec![None; set::MAX_SPS],
            pps: vec![None; set::MAX_PPS],
        }
    }
}

#[derive(Clone)]
struct Poc {
    prev_frame_num: u32,
    prev_frame_num_offset: u32,
    prev_poc_msb: i32,
    prev_poc_lsb: i32,
    have_ref: bool,
    max_long: Option<u8>,
}

impl Poc {
    fn new() -> Self {
        Self {
            prev_frame_num: 0,
            prev_frame_num_offset: 0,
            prev_poc_msb: 0,
            prev_poc_lsb: 0,
            have_ref: false,
            max_long: None,
        }
    }
}

pub struct H264Decoder {
    length_size: u8,
    sets: Sets,
    dpb: Vec<Frame>,
    poc: Poc,
    drained: bool,
    extra: Vec<u8>,
}

pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register_video(CodecId::H264, open_decoder);
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError> {
    Ok(Box::new(H264Decoder::open(params)?))
}

impl H264Decoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::H264 {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        let mut dec = Self {
            length_size: if params.nal_length_size == 0 {
                0
            } else {
                params.nal_length_size
            },
            sets: Sets::new(),
            dpb: Vec::new(),
            poc: Poc::new(),
            drained: false,
            extra: params.extra.clone(),
        };
        if !params.extra.is_empty() {
            dec.apply_extra(&params.extra)?;
        }
        Ok(dec)
    }

    fn apply_extra(&mut self, extra: &[u8]) -> Result<(), CodecError> {
        if extra.first() == Some(&1) {
            let (size, nals) = nal::avcc(extra)?;
            self.length_size = size;
            for nal in nals {
                self.store_param(&nal)?;
            }
        }
        Ok(())
    }

    fn store_param(&mut self, nal: &Nal) -> Result<(), CodecError> {
        match nal.kind {
            7 => {
                let sps = set::parse_sps(&nal.rbsp)?;
                let id = usize::from(sps.id);
                self.sets.sps[id] = Some(sps);
            }
            8 => {
                let pps = set::parse_pps(&nal.rbsp, &self.sets.sps)?;
                let id = usize::from(pps.id);
                self.sets.pps[id] = Some(pps);
            }
            _ => {}
        }
        Ok(())
    }

    /// The reconstructed picture for `pts`, including one still held for reorder.
    pub(crate) fn picture_at(&self, pts: i64) -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let frame = self
            .dpb
            .iter()
            .rev()
            .find(|frame| frame.pts == pts && !frame.nonexisting)?;
        let output = emit(frame)?;
        Some((
            output.picture.y().to_vec(),
            output.picture.u().to_vec(),
            output.picture.v().to_vec(),
        ))
    }
}

impl VideoDecoder for H264Decoder {
    fn codec(&self) -> CodecId {
        CodecId::H264
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<VideoOutput>, CodecError> {
        if packet.data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        let mut sets = self.sets.clone();
        let mut poc = self.poc.clone();
        let mut dpb = self.dpb.clone();
        let out = decode_au(
            &mut sets,
            &mut poc,
            &mut dpb,
            &packet.data,
            self.length_size,
            packet,
        )?;
        self.sets = sets;
        self.poc = poc;
        self.dpb = dpb;
        self.drained = false;
        Ok(out)
    }

    fn drain(&mut self) -> Result<Vec<VideoOutput>, CodecError> {
        if self.drained {
            return Ok(Vec::new());
        }
        self.drained = true;
        Ok(take_output(&mut self.dpb, 0))
    }

    fn reset(&mut self) {
        self.dpb.clear();
        self.poc = Poc::new();
        self.drained = false;
        self.sets = Sets::new();
        if !self.extra.is_empty() {
            let extra = self.extra.clone();
            let _ = self.apply_extra(&extra);
        }
    }
}

fn decode_au(
    sets: &mut Sets,
    poc: &mut Poc,
    dpb: &mut Vec<Frame>,
    data: &[u8],
    length_size: u8,
    packet: &Packet,
) -> Result<Vec<VideoOutput>, CodecError> {
    let nals = nal::access_unit(data, length_size)?;
    let mut picture: Option<Frame> = None;
    let mut prior = Vec::new();
    let mut slice_id = 0u16;
    let mut next_mb = 0usize;
    for nal in &nals {
        match nal.kind {
            7 | 8 => {
                if nal.kind == 7 {
                    let sps = set::parse_sps(&nal.rbsp)?;
                    let id = usize::from(sps.id);
                    sets.sps[id] = Some(sps);
                } else {
                    let pps = set::parse_pps(&nal.rbsp, &sets.sps)?;
                    let id = usize::from(pps.id);
                    sets.pps[id] = Some(pps);
                }
            }
            1 | 5 => {
                let idr = nal.kind == 5;
                let built = decode_slice(
                    sets,
                    poc,
                    dpb,
                    SliceCursor {
                        nal,
                        idr,
                        slice_id,
                        packet,
                    },
                    &mut picture,
                    &mut next_mb,
                    &mut prior,
                )?;
                if built {
                    slice_id = slice_id.saturating_add(1);
                }
            }
            6 | 9 | 10 | 11 | 12 => {}
            _ => {
                return Err(CodecError::Unsupported {
                    what: "nal unit type",
                });
            }
        }
    }
    let Some(pic) = picture else {
        return Ok(Vec::new());
    };
    if next_mb != pic.mb_w * (pic.h / 16) {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "incomplete picture",
        });
    }
    let mut pic = pic;
    let _ = mb::deblock;
    mb::deblock(&mut pic);
    let mut pictures = prior;
    pictures.extend(finish_picture(poc, dpb, pic)?);
    Ok(pictures)
}

fn finish_picture(
    _poc: &mut Poc,
    dpb: &mut Vec<Frame>,
    mut pic: Frame,
) -> Result<Vec<VideoOutput>, CodecError> {
    let max_ref = dpb_max(dpb, &pic);
    if pic.referenced {
        if pic.adaptive {
            apply_mmco(dpb, &mut pic, _poc)?;
        } else {
            mark_ref(dpb, &pic, max_ref);
        }
    }
    let reorder = pic_reorder(&pic);
    dpb.push(pic);
    dpb.retain(|frame| frame.referenced || frame.output);
    Ok(take_output(dpb, reorder))
}

fn dpb_max(dpb: &[Frame], pic: &Frame) -> usize {
    let _ = dpb;
    pic.max_ref.clamp(1, 16)
}

fn pic_reorder(pic: &Frame) -> usize {
    pic.reorder
}

fn mark_ref(dpb: &mut [Frame], pic: &Frame, max_ref: usize) {
    let mut short = dpb.iter().filter(|f| f.referenced && !f.long_term).count();
    let long = dpb.iter().filter(|f| f.referenced && f.long_term).count();
    short += 1;
    while short + long > max_ref && short > 0 {
        if let Some(pos) = oldest_short(dpb, pic.frame_num, pic.max_fn) {
            dpb[pos].referenced = false;
            short -= 1;
        } else {
            break;
        }
    }
}

/// Clause 8.2.5. Adaptive marking replaces the sliding window for this picture.
fn apply_mmco(dpb: &mut [Frame], pic: &mut Frame, poc: &mut Poc) -> Result<(), CodecError> {
    let mut current_long = false;
    for mark in pic.mmco.clone() {
        match mark.op {
            1 | 3 => {
                let target = i32::try_from(pic.frame_num).unwrap_or(0)
                    - i32::try_from(mark.arg).unwrap_or(0)
                    - 1;
                let Some(pos) = short_at(dpb, target, pic.frame_num, pic.max_fn, mark.op == 3)
                else {
                    return Err(CodecError::Malformed {
                        at_bit: 0,
                        what: "mmco",
                    });
                };
                if mark.op == 1 {
                    dpb[pos].referenced = false;
                } else {
                    let idx = i32::try_from(mark.arg2).unwrap_or(i32::MAX);
                    assign_long(dpb, pos, idx, poc.max_long)?;
                }
            }
            2 => {
                let idx = i32::try_from(mark.arg).unwrap_or(i32::MAX);
                if let Some(frame) = dpb
                    .iter_mut()
                    .find(|frame| frame.referenced && frame.long_term && frame.long_idx == idx)
                {
                    frame.referenced = false;
                }
            }
            4 => {
                if mark.arg > 16 {
                    return Err(CodecError::Malformed {
                        at_bit: 0,
                        what: "mmco",
                    });
                }
                if mark.arg == 0 {
                    poc.max_long = None;
                    for frame in dpb.iter_mut().filter(|frame| frame.long_term) {
                        frame.referenced = false;
                    }
                } else {
                    let max = mark.arg - 1;
                    poc.max_long = Some(u8::try_from(max).unwrap_or(0));
                    for frame in dpb.iter_mut() {
                        if frame.long_term && frame.long_idx > i32::try_from(max).unwrap_or(0) {
                            frame.referenced = false;
                        }
                    }
                }
            }
            5 => {
                for frame in dpb.iter_mut() {
                    frame.referenced = false;
                }
                pic.frame_num = 0;
                poc.prev_frame_num = 0;
                poc.prev_frame_num_offset = 0;
                poc.max_long = None;
            }
            6 => {
                let idx = i32::try_from(mark.arg).unwrap_or(i32::MAX);
                let allowed = poc
                    .max_long
                    .is_some_and(|max| idx >= 0 && idx <= i32::from(max));
                if !allowed {
                    return Err(CodecError::Malformed {
                        at_bit: 0,
                        what: "mmco",
                    });
                }
                for frame in dpb.iter_mut() {
                    if frame.long_term && frame.long_idx == idx {
                        frame.referenced = false;
                    }
                }
                pic.long_term = true;
                pic.long_idx = idx;
                current_long = true;
            }
            _ => {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "mmco",
                });
            }
        }
    }
    if !current_long {
        pic.long_term = false;
    }
    pic.mmco.clear();
    let refs = dpb.iter().filter(|frame| frame.referenced).count() + 1;
    if refs > pic.max_ref {
        return Err(CodecError::Limit {
            what: "reference frames",
        });
    }
    Ok(())
}

fn short_at(
    dpb: &[Frame],
    pic_num: i32,
    current: u32,
    max_fn: u32,
    coded_only: bool,
) -> Option<usize> {
    dpb.iter().enumerate().find_map(|(index, frame)| {
        if frame.referenced
            && !frame.long_term
            && !(coded_only && frame.nonexisting)
            && wrapped(frame, current, max_fn) == pic_num
        {
            Some(index)
        } else {
            None
        }
    })
}

fn wrapped(frame: &Frame, current: u32, max_fn: u32) -> i32 {
    let mut number = i32::try_from(frame.frame_num).unwrap_or(0);
    if frame.frame_num > current {
        number -= i32::try_from(max_fn).unwrap_or(0);
    }
    number
}

fn assign_long(
    dpb: &mut [Frame],
    pos: usize,
    idx: i32,
    max_long: Option<u8>,
) -> Result<(), CodecError> {
    let allowed = max_long.is_some_and(|max| idx >= 0 && idx <= i32::from(max));
    if !allowed {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "mmco",
        });
    }
    for (index, frame) in dpb.iter_mut().enumerate() {
        if index != pos && frame.long_term && frame.long_idx == idx {
            frame.referenced = false;
        }
    }
    dpb[pos].long_term = true;
    dpb[pos].long_idx = idx;
    Ok(())
}

fn oldest_short(dpb: &[Frame], current: u32, max_fn: u32) -> Option<usize> {
    dpb.iter()
        .enumerate()
        .filter(|(_, frame)| frame.referenced && !frame.long_term)
        .min_by_key(|(_, frame)| wrapped(frame, current, max_fn))
        .map(|(index, _)| index)
}

/// Clause 8.2.5.2. Each missing frame_num becomes a non-existing short-term reference
/// and can slide an older reference out. Those pictures are not displayed and are not
/// eligible as inter references.
fn fill_frame_num_gap(
    sps: &Sps,
    state: &mut Poc,
    dpb: &mut Vec<Frame>,
    frame_num: u32,
) -> Result<(), CodecError> {
    if !state.have_ref {
        return Ok(());
    }
    let max_fn = 1u32 << sps.log2_max_frame_num;
    if max_fn == 0 {
        return Ok(());
    }
    let mut unused = state.prev_frame_num.wrapping_add(1) % max_fn;
    if frame_num == state.prev_frame_num || frame_num == unused {
        return Ok(());
    }
    if !sps.gaps_allowed {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "frame_num gap",
        });
    }
    let mut steps = 0u32;
    while unused != frame_num {
        if steps >= max_fn {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "frame_num gap",
            });
        }
        steps += 1;
        let (poc, offset) = ghost_order(sps, state, unused)?;
        let ghost = absent_frame(sps, unused, poc);
        mark_ref(dpb, &ghost, ghost.max_ref);
        state.prev_frame_num = unused;
        state.prev_frame_num_offset = offset;
        dpb.push(ghost);
        dpb.retain(|frame| frame.referenced || frame.output);
        unused = unused.wrapping_add(1) % max_fn;
    }
    Ok(())
}

fn ghost_order(sps: &Sps, state: &Poc, frame_num: u32) -> Result<(i32, u32), CodecError> {
    let max_fn = 1u32 << sps.log2_max_frame_num;
    let offset = if frame_num < state.prev_frame_num {
        state.prev_frame_num_offset.wrapping_add(max_fn)
    } else {
        state.prev_frame_num_offset
    };
    let poc = if sps.poc_type == 1 {
        poc_type1(sps, offset, frame_num, true, [0, 0])?
    } else if sps.poc_type == 2 {
        let abs = i32::try_from(offset.saturating_add(frame_num)).unwrap_or(0);
        2 * abs
    } else {
        0
    };
    Ok((poc, offset))
}

fn absent_frame(sps: &Sps, frame_num: u32, poc: i32) -> Frame {
    Frame {
        y: Vec::new(),
        u: Vec::new(),
        v: Vec::new(),
        mb: Vec::new(),
        w: sps.width as usize,
        h: sps.height as usize,
        mb_w: sps.mb_width,
        poc,
        frame_num,
        referenced: true,
        long_term: false,
        long_idx: -1,
        output: false,
        key: false,
        pts: 0,
        duration: 0,
        timescale: 0,
        crop_x: 0,
        crop_y: 0,
        vis_w: 0,
        vis_h: 0,
        range: sps.range,
        primaries: sps.primaries,
        transfer: sps.transfer,
        matrix: sps.matrix,
        nonexisting: true,
        max_ref: usize::from(sps.max_num_ref_frames).clamp(1, 16),
        reorder: 0,
        adaptive: false,
        mmco: Vec::new(),
        max_fn: 1u32 << sps.log2_max_frame_num,
    }
}

fn take_output(dpb: &mut Vec<Frame>, keep: usize) -> Vec<VideoOutput> {
    let mut out = Vec::new();
    loop {
        let pending: Vec<usize> = dpb
            .iter()
            .enumerate()
            .filter(|(_, f)| f.output)
            .map(|(i, _)| i)
            .collect();
        if pending.len() <= keep {
            break;
        }
        let pos = pending.into_iter().min_by_key(|&i| dpb[i].poc).unwrap_or(0);
        dpb[pos].output = false;
        if let Some(picture) = emit(&dpb[pos]) {
            out.push(picture);
        }
    }
    dpb.retain(|frame| frame.referenced || frame.output);
    out
}

fn emit(frame: &Frame) -> Option<VideoOutput> {
    if frame.nonexisting {
        return None;
    }
    let x0 = frame.crop_x as usize;
    let y0 = frame.crop_y as usize;
    let w = frame.vis_w as usize;
    let h = frame.vis_h as usize;
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; (w / 2) * (h / 2)];
    let mut v = vec![0u8; (w / 2) * (h / 2)];
    for row in 0..h {
        let src = (y0 + row) * frame.w + x0;
        y[row * w..row * w + w].copy_from_slice(&frame.y[src..src + w]);
    }
    for row in 0..h / 2 {
        let src = (y0 / 2 + row) * (frame.w / 2) + x0 / 2;
        let n = w / 2;
        u[row * n..row * n + n].copy_from_slice(&frame.u[src..src + n]);
        v[row * n..row * n + n].copy_from_slice(&frame.v[src..src + n]);
    }
    let picture = Yuv420p::try_new(frame.vis_w, frame.vis_h, y, u, v).ok()?;
    Some(VideoOutput {
        picture,
        pts: frame.pts,
        duration: frame.duration,
        timescale: frame.timescale,
        keyframe: frame.key,
        range: frame.range,
        primaries: frame.primaries,
        transfer: frame.transfer,
        matrix: frame.matrix,
    })
}

pub(crate) struct Slice<'a> {
    sps: &'a Sps,
    pps: &'a Pps,
    kind: u8,
    poc: i32,
    qp: i32,
    cabac: bool,
    deblock: u8,
    off_a: i32,
    off_b: i32,
    ref0: Vec<usize>,
    ref1: Vec<usize>,
    poc0: Vec<i32>,
    poc1: Vec<i32>,
    long0: Vec<bool>,
    long1: Vec<bool>,
    weighted: bool,
    luma_denom: i32,
    chroma_denom: i32,
    yw: [[i32; 32]; 2],
    yo: [[i32; 32]; 2],
    cw: [[[i32; 2]; 32]; 2],
    co: [[[i32; 2]; 32]; 2],
    num_ref_l0: usize,
    num_ref_l1: usize,
    spatial_direct: bool,
}

struct SliceCursor<'a> {
    nal: &'a Nal,
    idr: bool,
    slice_id: u16,
    packet: &'a Packet,
}

fn decode_slice(
    sets: &mut Sets,
    poc_state: &mut Poc,
    dpb: &mut Vec<Frame>,
    cursor: SliceCursor<'_>,
    picture: &mut Option<Frame>,
    next_mb: &mut usize,
    prior: &mut Vec<VideoOutput>,
) -> Result<bool, CodecError> {
    let SliceCursor {
        nal,
        idr,
        slice_id,
        packet,
    } = cursor;
    let mut bits = Bits::new(&nal.rbsp);
    let header = read_header(&mut bits, sets, idr, nal.ref_idc != 0)?;
    // A redundant picture repeats one that was already decoded. It is not displayed.
    if header.redundant > 0 {
        return Ok(false);
    }
    let sps_id = usize::from(header.sps_id);
    let Some(sps) = sets.sps[sps_id].clone() else {
        return Err(bits.malformed("missing sps"));
    };
    if picture.is_none() && !idr {
        fill_frame_num_gap(&sps, poc_state, dpb, header.frame_num)?;
    }
    let (poc, frame_num_offset) = picture_order(&header, &sps, poc_state, idr)?;
    let _ = frame_num_offset;
    if picture.is_none() {
        // An IDR marks every earlier picture unused for reference. The flag
        // drops pictures that were still waiting; otherwise they are returned
        // before this picture. The flag does not hide the IDR itself.
        if idr {
            poc_state.max_long = if header.long_term_ref { Some(0) } else { None };
            for frame in dpb.iter_mut() {
                frame.referenced = false;
                if header.no_output_prior {
                    frame.output = false;
                }
            }
            prior.extend(take_output(dpb, 0));
        }
        *picture = Some(new_frame(&sps, &header, poc, packet, idr)?);
        *next_mb = 0;
        if header.nal_ref {
            poc_state.prev_frame_num = header.frame_num;
            poc_state.prev_frame_num_offset = frame_num_offset;
            poc_state.have_ref = true;
        }
    }
    if header.first_mb != *next_mb {
        return Err(bits.malformed("first macroblock"));
    }
    let slice = make_slice(sets, dpb, &header, poc)?;
    let Some(pic) = picture.as_mut() else {
        return Err(bits.malformed("picture"));
    };
    if slice.cabac {
        let mut cab = Cabac::start(&mut bits, slice.qp, cabac_init(&header))?;
        mb::decode_mbs(
            pic,
            dpb,
            &slice,
            slice_id,
            next_mb,
            Entropy::Cabac(&mut cab),
        )?;
    } else {
        mb::decode_mbs(
            pic,
            dpb,
            &slice,
            slice_id,
            next_mb,
            Entropy::Cavlc(&mut bits),
        )?;
    }
    Ok(true)
}

pub(crate) enum Entropy<'a, 'b> {
    Cavlc(&'a mut Bits<'b>),
    Cabac(&'a mut Cabac<'b>),
}

#[derive(Clone)]
struct Mark {
    op: u8,
    arg: u32,
    arg2: u32,
}

struct Header {
    first_mb: usize,
    pps_id: u8,
    sps_id: u8,
    kind: u8,
    frame_num: u32,
    poc_lsb: i32,
    delta_poc: [i32; 2],
    qp: i32,
    deblock: u8,
    off_a: i32,
    off_b: i32,
    cabac_init: u8,
    spatial_direct: bool,
    num_ref_l0: usize,
    num_ref_l1: usize,
    nal_ref: bool,
    redundant: u32,
    no_output_prior: bool,
    long_term_ref: bool,
    adaptive: bool,
    mmco: Vec<Mark>,
    ref_mod_l0: Vec<Mod>,
    ref_mod_l1: Vec<Mod>,
    weights: bool,
    luma_denom: i32,
    chroma_denom: i32,
    yw: [[i32; 32]; 2],
    yo: [[i32; 32]; 2],
    cw: [[[i32; 2]; 32]; 2],
    co: [[[i32; 2]; 32]; 2],
}

struct Mod {
    idc: u32,
    arg: u32,
}

fn read_header(
    bits: &mut Bits<'_>,
    sets: &Sets,
    idr: bool,
    nal_ref: bool,
) -> Result<Header, CodecError> {
    let first_mb = bits.ue()? as usize;
    let slice_type = bits.ue()?;
    let kind = match slice_type % 5 {
        0 => 0,
        1 => 1,
        2 => 2,
        _ => {
            return Err(CodecError::Unsupported { what: "slice type" });
        }
    };
    let pps_id = bits.ue()?;
    if pps_id >= set::MAX_PPS as u32 {
        return Err(bits.malformed("pps id"));
    }
    let Some(pps) = sets.pps[pps_id as usize].clone() else {
        return Err(bits.malformed("missing pps"));
    };
    let Some(sps) = sets.sps[usize::from(pps.sps_id)].as_ref() else {
        return Err(bits.malformed("missing sps"));
    };
    let frame_num = bits.u(u32::from(sps.log2_max_frame_num))?;
    if idr {
        let _ = bits.ue()?;
    }
    let mut poc_lsb = 0i32;
    let mut delta_poc = [0i32; 2];
    if sps.poc_type == 0 {
        poc_lsb = bits.u(u32::from(sps.log2_max_poc_lsb))? as i32;
        if pps.bottom_field_pic_order {
            let _ = bits.se()?;
        }
    } else if sps.poc_type == 1 && !sps.delta_pic_order_always_zero {
        delta_poc[0] = bits.se()?;
        if pps.bottom_field_pic_order {
            delta_poc[1] = bits.se()?;
        }
    }
    let mut redundant = 0u32;
    if pps.redundant_pic_cnt_present {
        redundant = bits.ue()?;
        if redundant > 127 {
            return Err(bits.malformed("redundant_pic_cnt"));
        }
    }
    let mut spatial_direct = true;
    if kind == 1 {
        spatial_direct = bits.bit()?;
    }
    let mut num_ref_l0 = usize::from(pps.num_ref_idx_l0);
    let mut num_ref_l1 = usize::from(pps.num_ref_idx_l1);
    if kind != 2 {
        if bits.bit()? {
            num_ref_l0 = bits.ue()? as usize + 1;
            if kind == 1 {
                num_ref_l1 = bits.ue()? as usize + 1;
            }
        }
        if num_ref_l0 > 16 || num_ref_l1 > 16 {
            return Err(CodecError::Limit {
                what: "reference index",
            });
        }
    } else {
        num_ref_l0 = 0;
        num_ref_l1 = 0;
    }
    let (ref_mod_l0, ref_mod_l1) = if kind != 2 {
        (
            read_list_mod(bits)?,
            if kind == 1 {
                read_list_mod(bits)?
            } else {
                Vec::new()
            },
        )
    } else {
        (Vec::new(), Vec::new())
    };
    let mut weights = false;
    let mut luma_denom = 0;
    let mut chroma_denom = 0;
    let mut yw = [[0i32; 32]; 2];
    let mut yo = [[0i32; 32]; 2];
    let mut cw = [[[0i32; 2]; 32]; 2];
    let mut co = [[[0i32; 2]; 32]; 2];
    if (pps.weighted_pred && kind == 0) || (pps.weighted_bipred == 1 && kind == 1) {
        weights = true;
        luma_denom = bits.ue()? as i32;
        chroma_denom = bits.ue()? as i32;
        if luma_denom > 7 || chroma_denom > 7 {
            return Err(bits.malformed("weight denom"));
        }
        read_weights(
            bits,
            0,
            num_ref_l0,
            (luma_denom, chroma_denom),
            WeightTables {
                yw: &mut yw,
                yo: &mut yo,
                cw: &mut cw,
                co: &mut co,
            },
        )?;
        if kind == 1 {
            read_weights(
                bits,
                1,
                num_ref_l1,
                (luma_denom, chroma_denom),
                WeightTables {
                    yw: &mut yw,
                    yo: &mut yo,
                    cw: &mut cw,
                    co: &mut co,
                },
            )?;
        }
    }
    let mut no_output_prior = false;
    let mut long_term_ref = false;
    let mut adaptive = false;
    let mut mmco = Vec::new();
    if nal_ref {
        if idr {
            no_output_prior = bits.bit()?;
            long_term_ref = bits.bit()?;
        } else if bits.bit()? {
            adaptive = true;
            loop {
                let op = bits.ue()?;
                if op == 0 {
                    break;
                }
                if op > 6 {
                    return Err(bits.malformed("mmco"));
                }
                let mut arg = 0u32;
                let mut arg2 = 0u32;
                if matches!(op, 1..=4) {
                    arg = bits.ue()?;
                }
                if matches!(op, 3 | 6) {
                    arg2 = bits.ue()?;
                }
                if op == 6 {
                    arg = arg2;
                    arg2 = 0;
                }
                mmco.push(Mark {
                    op: u8::try_from(op).unwrap_or(0),
                    arg,
                    arg2,
                });
            }
        }
    }
    let mut cabac_init = 0u8;
    if pps.cabac && kind != 2 {
        cabac_init = bits.ue()? as u8;
        if cabac_init > 2 {
            return Err(bits.malformed("cabac_init_idc"));
        }
    }
    let qp = pps.init_qp + bits.se()?;
    if !(0..=51).contains(&qp) {
        return Err(bits.malformed("slice qp"));
    }
    let mut deblock = 0u8;
    let mut off_a = 0i32;
    let mut off_b = 0i32;
    if pps.deblocking_control {
        deblock = bits.ue()? as u8;
        if deblock > 2 {
            return Err(bits.malformed("deblock idc"));
        }
        if deblock != 1 {
            off_a = bits.se()? * 2;
            off_b = bits.se()? * 2;
            if !(-12..=12).contains(&off_a) || !(-12..=12).contains(&off_b) {
                return Err(bits.malformed("deblock offset"));
            }
        }
    }
    Ok(Header {
        first_mb,
        pps_id: pps_id as u8,
        sps_id: pps.sps_id,
        kind: kind as u8,
        frame_num,
        poc_lsb,
        delta_poc,
        qp,
        deblock,
        off_a,
        off_b,
        cabac_init,
        spatial_direct,
        num_ref_l0,
        num_ref_l1,
        nal_ref,
        redundant,
        no_output_prior,
        long_term_ref,
        adaptive,
        mmco,
        ref_mod_l0,
        ref_mod_l1,
        weights,
        luma_denom,
        chroma_denom,
        yw,
        yo,
        cw,
        co,
    })
}

fn read_list_mod(bits: &mut Bits<'_>) -> Result<Vec<Mod>, CodecError> {
    let mut mods = Vec::new();
    if !bits.bit()? {
        return Ok(mods);
    }
    loop {
        let idc = bits.ue()?;
        if idc == 3 {
            break;
        }
        if idc > 3 {
            return Err(bits.malformed("ref list modification"));
        }
        let arg = bits.ue()?;
        mods.push(Mod { idc, arg });
        if mods.len() > 32 {
            return Err(CodecError::Limit {
                what: "ref list modification",
            });
        }
    }
    Ok(mods)
}

struct WeightTables<'a> {
    yw: &'a mut [[i32; 32]; 2],
    yo: &'a mut [[i32; 32]; 2],
    cw: &'a mut [[[i32; 2]; 32]; 2],
    co: &'a mut [[[i32; 2]; 32]; 2],
}

fn read_weights(
    bits: &mut Bits<'_>,
    list: usize,
    n: usize,
    denom: (i32, i32),
    tables: WeightTables<'_>,
) -> Result<(), CodecError> {
    let (luma_denom, chroma_denom) = denom;
    let WeightTables { yw, yo, cw, co } = tables;
    for i in 0..n {
        yw[list][i] = 1 << luma_denom;
        yo[list][i] = 0;
        cw[list][i] = [1 << chroma_denom, 1 << chroma_denom];
        co[list][i] = [0, 0];
        if bits.bit()? {
            yw[list][i] = bits.se()?;
            yo[list][i] = bits.se()?;
        }
        if bits.bit()? {
            for c in 0..2 {
                cw[list][i][c] = bits.se()?;
                co[list][i][c] = bits.se()?;
            }
        }
    }
    Ok(())
}

fn picture_order(
    header: &Header,
    sps: &Sps,
    state: &mut Poc,
    idr: bool,
) -> Result<(i32, u32), CodecError> {
    let max_fn = 1u32 << sps.log2_max_frame_num;
    let offset = if idr || !state.have_ref {
        0
    } else if header.frame_num < state.prev_frame_num {
        state.prev_frame_num_offset.wrapping_add(max_fn)
    } else {
        state.prev_frame_num_offset
    };
    if idr && header.nal_ref {
        state.prev_poc_msb = 0;
        state.prev_poc_lsb = if sps.poc_type == 0 { header.poc_lsb } else { 0 };
    }
    let poc = if sps.poc_type == 1 {
        poc_type1(
            sps,
            offset,
            header.frame_num,
            header.nal_ref,
            header.delta_poc,
        )?
    } else if sps.poc_type == 2 {
        let abs = i32::try_from(offset.saturating_add(header.frame_num)).unwrap_or(0);
        if header.nal_ref { 2 * abs } else { 2 * abs - 1 }
    } else if idr {
        header.poc_lsb
    } else {
        let max_lsb = 1i32 << sps.log2_max_poc_lsb;
        let prev_msb = if state.have_ref {
            state.prev_poc_msb
        } else {
            0
        };
        let prev_lsb = if state.have_ref {
            state.prev_poc_lsb
        } else {
            0
        };
        let msb = if header.poc_lsb < prev_lsb && prev_lsb - header.poc_lsb >= max_lsb / 2 {
            prev_msb + max_lsb
        } else if header.poc_lsb > prev_lsb && header.poc_lsb - prev_lsb > max_lsb / 2 {
            prev_msb - max_lsb
        } else {
            prev_msb
        };
        if header.nal_ref {
            state.prev_poc_msb = msb;
            state.prev_poc_lsb = header.poc_lsb;
        }
        msb + header.poc_lsb
    };
    Ok((poc, offset))
}

/// Clause 8.2.1.2. A frame's order count is the smaller of the two fields.
fn poc_type1(
    sps: &Sps,
    frame_num_offset: u32,
    frame_num: u32,
    nal_ref: bool,
    delta: [i32; 2],
) -> Result<i32, CodecError> {
    let mut abs_frame_num = if sps.offset_for_ref.is_empty() {
        0
    } else {
        i64::from(frame_num_offset) + i64::from(frame_num)
    };
    if !nal_ref && abs_frame_num > 0 {
        abs_frame_num -= 1;
    }
    let mut expected = 0i64;
    if abs_frame_num > 0 {
        let cycle = sps.offset_for_ref.len() as i64;
        let per_cycle: i64 = sps
            .offset_for_ref
            .iter()
            .map(|offset| i64::from(*offset))
            .sum();
        let cycle_count = (abs_frame_num - 1) / cycle;
        let in_cycle = usize::try_from((abs_frame_num - 1) % cycle).unwrap_or(0);
        expected = cycle_count * per_cycle;
        for offset in sps.offset_for_ref.iter().take(in_cycle + 1) {
            expected += i64::from(*offset);
        }
    }
    if !nal_ref {
        expected += i64::from(sps.offset_non_ref);
    }
    let top = expected + i64::from(delta[0]);
    let bottom = top + i64::from(sps.offset_top_bottom) + i64::from(delta[1]);
    i32::try_from(top.min(bottom)).map_err(|_| CodecError::Malformed {
        at_bit: 0,
        what: "picture order count",
    })
}

fn new_frame(
    sps: &Sps,
    header: &Header,
    poc: i32,
    packet: &Packet,
    idr: bool,
) -> Result<Frame, CodecError> {
    let (vis_w, vis_h) = sps.visible()?;
    let w = sps.width as usize;
    let h = sps.height as usize;
    let mb_w = sps.mb_width;
    let mb_h = sps.mb_height;
    let chroma = (w / 2).saturating_mul(h / 2);
    let reorder = if sps.max_reorder > 0 {
        sps.max_reorder as usize
    } else if sps.max_num_ref_frames > 1 {
        1
    } else {
        0
    };
    Ok(Frame {
        y: vec![0; w.saturating_mul(h)],
        u: vec![128; chroma],
        v: vec![128; chroma],
        mb: vec![Mb::new(u16::MAX); mb_w.saturating_mul(mb_h)],
        w,
        h,
        mb_w,
        poc,
        frame_num: header.frame_num,
        referenced: header.nal_ref,
        long_term: header.long_term_ref,
        long_idx: if header.long_term_ref { 0 } else { -1 },
        output: true,
        key: idr || header.kind == 2,
        pts: packet.pts,
        duration: packet.duration,
        timescale: packet.timescale,
        crop_x: sps.crop_left.saturating_mul(2),
        crop_y: sps.crop_top.saturating_mul(2),
        vis_w,
        vis_h,
        range: sps.range,
        primaries: sps.primaries,
        transfer: sps.transfer,
        matrix: sps.matrix,
        nonexisting: false,
        max_ref: usize::from(sps.max_num_ref_frames).clamp(1, 16),
        reorder,
        adaptive: header.adaptive,
        mmco: header.mmco.clone(),
        max_fn: 1u32 << sps.log2_max_frame_num,
    })
}

fn make_slice<'a>(
    sets: &'a Sets,
    dpb: &[Frame],
    header: &Header,
    poc: i32,
) -> Result<Slice<'a>, CodecError> {
    let Some(sps) = sets.sps[usize::from(header.sps_id)].as_ref() else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "missing sps",
        });
    };
    let Some(pps) = sets.pps[usize::from(header.pps_id)].as_ref() else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "missing pps",
        });
    };
    let max_fn = 1u32 << sps.log2_max_frame_num;
    let (ref0, ref1) = mb::reference_lists(dpb, header, poc, max_fn);
    let (poc0, long0) = ref_marks(dpb, &ref0);
    let (poc1, long1) = ref_marks(dpb, &ref1);
    Ok(Slice {
        sps,
        pps,
        kind: header.kind,
        poc,
        qp: header.qp,
        cabac: pps.cabac,
        deblock: header.deblock,
        off_a: header.off_a,
        off_b: header.off_b,
        ref0,
        ref1,
        poc0,
        poc1,
        long0,
        long1,
        weighted: header.weights,
        luma_denom: header.luma_denom,
        chroma_denom: header.chroma_denom,
        yw: header.yw,
        yo: header.yo,
        cw: header.cw,
        co: header.co,
        num_ref_l0: header.num_ref_l0,
        num_ref_l1: header.num_ref_l1,
        spatial_direct: header.spatial_direct,
    })
}

fn ref_marks(dpb: &[Frame], list: &[usize]) -> (Vec<i32>, Vec<bool>) {
    let mut pocs = Vec::with_capacity(list.len());
    let mut longs = Vec::with_capacity(list.len());
    for &index in list {
        let Some(frame) = dpb.get(index) else {
            pocs.push(0);
            longs.push(true);
            continue;
        };
        pocs.push(frame.poc);
        longs.push(frame.long_term);
    }
    (pocs, longs)
}

fn cabac_init(header: &Header) -> usize {
    if header.kind == 2 {
        0
    } else {
        usize::from(header.cabac_init) + 1
    }
}

fn clip_ref(value: i32) -> i16 {
    value.clamp(-32768, 32767) as i16
}

fn blk_x(i: usize) -> usize {
    (i & 1) | ((i >> 1) & 2)
}

fn blk_y(i: usize) -> usize {
    ((i >> 1) & 1) | ((i >> 2) & 2)
}

fn put_block(plane: &mut [u8], stride: usize, x: usize, y: usize, src: &[u8], n: usize) {
    for row in 0..n {
        for col in 0..n {
            plane[(y + row) * stride + x + col] = src[row * n + col];
        }
    }
}

fn add_scaled(
    plane: &mut [u8],
    stride: usize,
    origin: (usize, usize),
    coeffs: &[i32],
    qp: i32,
    weight: &[u8],
    n: usize,
) {
    let (x, y) = origin;
    if n == 4 {
        let mut block = [0i32; 16];
        for (i, &pos) in ZIGZAG4.iter().enumerate() {
            let (ry, rx) = (pos / 4, pos % 4);
            block[ry * 4 + rx] = recon::scale4(
                coeffs.get(i).copied().unwrap_or(0),
                qp,
                rx,
                ry,
                i32::from(weight[ry * 4 + rx]),
            );
        }
        recon::idct4(&mut block);
        recon::add_block(plane, stride, x, y, &block, 4);
    } else {
        let mut block = [0i32; 64];
        for (i, &pos) in ZIGZAG8.iter().enumerate() {
            let (ry, rx) = (pos / 8, pos % 8);
            block[ry * 8 + rx] = recon::scale8(
                coeffs.get(i).copied().unwrap_or(0),
                qp,
                rx,
                ry,
                i32::from(weight[pos]),
            );
        }
        recon::idct8(&mut block);
        recon::add_block(plane, stride, x, y, &block, 8);
    }
}

#[cfg(test)]
mod tests {
    use super::H264Decoder;
    use crate::H264Encoder;
    use cutlass_decode::{CodecError, CodecId, CodecParams, Packet, VideoDecoder, Yuv420p};
    use cutlass_encode::{VideoEncoder, VideoInput};
    use std::process::Command;

    #[test]
    fn picture_order_type1_follows_the_reference_cycle() {
        let sps = type1_sps(&[2, 4], -1, 1);
        let order = |frame_num, nal_ref, delta| {
            super::poc_type1(&sps, 0, frame_num, nal_ref, delta).expect("poc")
        };
        assert_eq!(order(0, true, [0, 0]), 0);
        assert_eq!(order(1, true, [0, 0]), 2);
        assert_eq!(order(2, true, [0, 0]), 6);
        assert_eq!(order(3, true, [0, 0]), 8);
        assert_eq!(order(1, false, [0, 0]), -1);
        assert_eq!(order(2, false, [3, -8]), -3);
        let wide = type1_sps(&[2_000_000_000], 0, 0);
        assert!(super::poc_type1(&wide, 0, 2, true, [0, 0]).is_err());
    }

    #[test]
    fn frame_num_gap_slides_real_references_out() {
        let mut sps = type1_sps(&[], 0, 0);
        sps.poc_type = 2;
        sps.gaps_allowed = true;
        sps.max_num_ref_frames = 2;
        sps.log2_max_frame_num = 4;
        let mut dpb = vec![held(0), held(1)];
        let mut poc = super::Poc::new();
        poc.prev_frame_num = 1;
        poc.have_ref = true;
        super::fill_frame_num_gap(&sps, &mut poc, &mut dpb, 4).unwrap();
        let live: Vec<_> = dpb
            .iter()
            .filter(|frame| frame.referenced)
            .map(|frame| (frame.frame_num, frame.nonexisting, frame.output))
            .collect();
        assert_eq!(live, vec![(2, true, false), (3, true, false)]);
        assert_eq!(poc.prev_frame_num, 3);
        sps.gaps_allowed = false;
        poc.prev_frame_num = 1;
        assert!(super::fill_frame_num_gap(&sps, &mut poc, &mut dpb, 4).is_err());
    }

    #[test]
    fn memory_management_commands_change_which_pictures_stay_referenced() {
        let mut dpb = vec![held(0), held(1)];
        let mut current = held(2);
        current.mmco = vec![super::Mark {
            op: 1,
            arg: 0,
            arg2: 0,
        }];
        let mut poc = super::Poc::new();
        super::apply_mmco(&mut dpb, &mut current, &mut poc).unwrap();
        assert!(dpb[0].referenced);
        assert!(!dpb[1].referenced);

        let mut dpb = vec![held(0), held(1)];
        let mut current = held(2);
        current.mmco = vec![
            super::Mark {
                op: 4,
                arg: 1,
                arg2: 0,
            },
            super::Mark {
                op: 3,
                arg: 1,
                arg2: 0,
            },
        ];
        super::apply_mmco(&mut dpb, &mut current, &mut poc).unwrap();
        assert!(dpb[0].long_term && dpb[0].long_idx == 0);
        assert!(dpb[1].referenced && !dpb[1].long_term);

        current.mmco = vec![super::Mark {
            op: 2,
            arg: 0,
            arg2: 0,
        }];
        super::apply_mmco(&mut dpb, &mut current, &mut poc).unwrap();
        assert!(!dpb[0].referenced);

        current.mmco = vec![super::Mark {
            op: 6,
            arg: 0,
            arg2: 0,
        }];
        super::apply_mmco(&mut dpb, &mut current, &mut poc).unwrap();
        assert!(current.long_term && current.long_idx == 0);

        current.mmco = vec![super::Mark {
            op: 5,
            arg: 0,
            arg2: 0,
        }];
        super::apply_mmco(&mut dpb, &mut current, &mut poc).unwrap();
        assert!(dpb.iter().all(|frame| !frame.referenced));
        assert_eq!(current.frame_num, 0);
        assert_eq!(poc.prev_frame_num, 0);
        assert!(poc.max_long.is_none());
    }

    fn held(frame_num: u32) -> super::Frame {
        use cutlass_decode::{ColorMatrix, ColorPrimaries, ColorRange, Transfer};
        super::Frame {
            y: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
            mb: Vec::new(),
            w: 16,
            h: 16,
            mb_w: 1,
            poc: i32::try_from(frame_num).unwrap_or(0),
            frame_num,
            referenced: true,
            long_term: false,
            long_idx: -1,
            output: false,
            key: false,
            pts: 0,
            duration: 1,
            timescale: 30,
            crop_x: 0,
            crop_y: 0,
            vis_w: 16,
            vis_h: 16,
            range: ColorRange::Limited,
            primaries: ColorPrimaries::Unspecified,
            transfer: Transfer::Unspecified,
            matrix: ColorMatrix::Unspecified,
            nonexisting: false,
            max_ref: 4,
            reorder: 0,
            adaptive: true,
            mmco: Vec::new(),
            max_fn: 16,
        }
    }

    fn type1_sps(offsets: &[i32], non_ref: i32, top_bottom: i32) -> super::Sps {
        use cutlass_decode::{ColorMatrix, ColorPrimaries, ColorRange, Transfer};
        super::Sps {
            id: 0,
            log2_max_frame_num: 4,
            poc_type: 1,
            log2_max_poc_lsb: 0,
            delta_pic_order_always_zero: false,
            offset_non_ref: non_ref,
            offset_top_bottom: top_bottom,
            offset_for_ref: offsets.to_vec(),
            max_num_ref_frames: 1,
            gaps_allowed: false,
            width: 16,
            height: 16,
            crop_left: 0,
            crop_right: 0,
            crop_top: 0,
            crop_bottom: 0,
            direct_8x8_inference: true,
            mb_width: 1,
            mb_height: 1,
            weight4: [[16; 16]; 6],
            weight8: [[16; 64]; 6],
            range: ColorRange::Limited,
            primaries: ColorPrimaries::Unspecified,
            transfer: Transfer::Unspecified,
            matrix: ColorMatrix::Unspecified,
            max_reorder: 0,
        }
    }

    fn same(name: &str, got: &[u8], want: &[u8]) {
        assert_eq!(got.len(), want.len(), "{name} length");
        if let Some(index) = got.iter().zip(want).position(|(a, b)| a != b) {
            panic!(
                "{name}[{index}] = {} want {} (got {:?} want {:?})",
                got[index],
                want[index],
                &got[index..index.saturating_add(8).min(got.len())],
                &want[index..index.saturating_add(8).min(want.len())]
            );
        }
    }

    fn ffmpeg() -> Option<std::path::PathBuf> {
        for candidate in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"] {
            let path = std::path::PathBuf::from(candidate);
            if path.is_file() {
                return Some(path);
            }
        }
        None
    }

    #[test]
    fn decodes_one_independent_intra_picture() {
        let Some(ff) = ffmpeg() else {
            return;
        };
        let dir = std::env::temp_dir();
        let h264 = dir.join("cutlass-h264-intra.h264");
        let yuv = dir.join("cutlass-h264-intra.yuv");
        let encoded = Command::new(&ff)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:size=176x144:rate=30:duration=1",
                "-frames:v",
                "1",
                "-c:v",
                "libx264",
                "-profile:v",
                "baseline",
                "-bf",
                "0",
                "-g",
                "1",
                "-x264-params",
                "cabac=0",
                "-f",
                "h264",
            ])
            .arg(&h264)
            .status();
        let Ok(status) = encoded else {
            return;
        };
        if !status.success() {
            return;
        }
        let decoded = Command::new(&ff)
            .args(["-y", "-i"])
            .arg(&h264)
            .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status();
        let Ok(status) = decoded else {
            return;
        };
        if !status.success() {
            return;
        }
        let bytes = std::fs::read(&h264).expect("annex b");
        let reference = std::fs::read(&yuv).expect("yuv");
        let mut params = CodecParams::video(CodecId::H264, 176, 144);
        params.nal_length_size = 0;
        let mut decoder = H264Decoder::open(&params).expect("open");
        let pictures = decoder
            .push(&Packet::video(bytes, 0, 1, 30, true))
            .expect("decode");
        let Some(picture) = pictures.first() else {
            panic!("no picture");
        };
        assert_eq!(picture.picture.width(), 176);
        assert_eq!(picture.picture.height(), 144);
        let y_len = 176 * 144;
        let c_len = 88 * 72;
        assert_eq!(reference.len(), y_len + c_len * 2);
        same("y", picture.picture.y(), &reference[..y_len]);
        same("u", picture.picture.u(), &reference[y_len..y_len + c_len]);
        same("v", picture.picture.v(), &reference[y_len + c_len..]);
    }

    /// A later IDR must return pictures the reorder buffer was still holding.
    /// x264 leaves `no_output_of_prior_pics_flag` clear, which is the case
    /// that used to drop those pictures.
    #[test]
    fn idr_returns_pictures_still_waiting() {
        let Some(ff) = ffmpeg() else {
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-idr-wait");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let h264 = dir.join("gop.h264");
        let yuv = dir.join("gop.yuv");
        let encoded = Command::new(&ff)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30:duration=1",
                "-frames:v",
                "8",
                "-c:v",
                "libx264",
                "-profile:v",
                "main",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=1:bframes=1:keyint=4:min-keyint=4:scenecut=0:weightp=0:8x8dct=0",
                "-f",
                "h264",
            ])
            .arg(&h264)
            .status();
        let Ok(status) = encoded else {
            return;
        };
        if !status.success() {
            return;
        }
        let decoded = Command::new(&ff)
            .args(["-y", "-i"])
            .arg(&h264)
            .args(["-frames:v", "8", "-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status();
        let Ok(status) = decoded else {
            return;
        };
        if !status.success() {
            return;
        }
        let bytes = std::fs::read(&h264).expect("annex b");
        let reference = std::fs::read(&yuv).expect("yuv");
        let mut params = CodecParams::video(CodecId::H264, 64, 64);
        params.nal_length_size = 0;
        let mut decoder = H264Decoder::open(&params).expect("open");
        let mut got = Vec::new();
        for (n, unit) in access_units(&bytes).iter().enumerate() {
            let pictures = decoder
                .push(&Packet::video(unit.clone(), n as i64, 1, 30, n == 0))
                .expect("decode");
            got.extend(pictures);
        }
        got.extend(decoder.drain().expect("drain"));
        let frame = 64 * 64 * 3 / 2;
        assert_eq!(reference.len() / frame, 8, "ffmpeg picture count");
        assert_eq!(
            got.len(),
            reference.len() / frame,
            "dropped a waiting picture"
        );
        for (index, picture) in got.iter().enumerate() {
            let start = index * frame;
            let y_len = 64 * 64;
            let c_len = 32 * 32;
            same("y", picture.picture.y(), &reference[start..start + y_len]);
            same(
                "u",
                picture.picture.u(),
                &reference[start + y_len..start + y_len + c_len],
            );
            same(
                "v",
                picture.picture.v(),
                &reference[start + y_len + c_len..start + frame],
            );
        }
    }

    #[test]
    fn one_and_two_byte_lengths_match_annex_b() {
        let mut params = CodecParams::video(CodecId::H264, 16, 16);
        params.nal_length_size = 0;
        let mut encoder = H264Encoder::open(&params).unwrap();
        let packets = encoder
            .push(&VideoInput {
                picture: Yuv420p::solid(16, 16, 40, 80, 160).unwrap(),
                pts: 0,
                duration: 1,
                timescale: 30,
                force_keyframe: true,
            })
            .unwrap();
        let annex = &packets[0].data;
        let mut annex_decoder = H264Decoder::open(&params).unwrap();
        let mut want = annex_decoder
            .push(&Packet::video(annex.clone(), 0, 1, 30, true))
            .unwrap();
        want.extend(annex_decoder.drain().unwrap());
        assert_eq!(want.len(), 1);
        for size in [1u8, 2] {
            let prefixed = length_prefix(annex, size);
            let mut sized = params.clone();
            sized.nal_length_size = size;
            let mut decoder = H264Decoder::open(&sized).unwrap();
            let mut got = decoder
                .push(&Packet::video(prefixed, 0, 1, 30, true))
                .unwrap();
            got.extend(decoder.drain().unwrap());
            assert_eq!(got.len(), 1, "length size {size}");
            assert_eq!(got[0].picture.y(), want[0].picture.y());
            assert_eq!(got[0].picture.u(), want[0].picture.u());
            assert_eq!(got[0].picture.v(), want[0].picture.v());
        }
    }

    fn length_prefix(annex_b: &[u8], length_size: u8) -> Vec<u8> {
        let mut payloads = Vec::new();
        let mut i = 0;
        while i + 3 < annex_b.len() {
            if annex_b[i] == 0 && annex_b[i + 1] == 0 && annex_b[i + 2] == 1 {
                let start = i + 3;
                let mut j = start;
                while j + 3 < annex_b.len()
                    && !(annex_b[j] == 0 && annex_b[j + 1] == 0 && annex_b[j + 2] == 1)
                {
                    j += 1;
                }
                let end = if j + 3 < annex_b.len() {
                    if j > start && annex_b[j - 1] == 0 {
                        j - 1
                    } else {
                        j
                    }
                } else {
                    annex_b.len()
                };
                payloads.push(&annex_b[start..end]);
                i = start;
            } else {
                i += 1;
            }
        }
        let mut out = Vec::new();
        for payload in payloads {
            assert!(
                payload.len() < 1usize << (8 * usize::from(length_size)),
                "nal does not fit in a {length_size}-byte length"
            );
            for shift in (0..length_size).rev() {
                out.push((payload.len() >> (8 * usize::from(shift))) as u8);
            }
            out.extend_from_slice(payload);
        }
        out
    }

    fn access_units(data: &[u8]) -> Vec<Vec<u8>> {
        let mut starts = Vec::new();
        let mut i = 0;
        while i + 3 < data.len() {
            if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
                let header = if i > 0 && data[i - 1] == 0 { i - 1 } else { i };
                starts.push((header, i + 3));
                i += 3;
            } else {
                i += 1;
            }
        }
        let mut units = Vec::new();
        let mut current = Vec::new();
        let mut saw_vcl = false;
        for (n, &(header, body)) in starts.iter().enumerate() {
            let end = starts
                .get(n + 1)
                .map(|(next, _)| *next)
                .unwrap_or(data.len());
            let kind = data[body] & 31;
            let vcl = kind == 1 || kind == 5;
            if (kind == 9 || (vcl && saw_vcl)) && !current.is_empty() {
                units.push(std::mem::take(&mut current));
                saw_vcl = false;
            }
            current.extend_from_slice(&data[header..end]);
            saw_vcl |= vcl;
        }
        if !current.is_empty() {
            units.push(current);
        }
        units
    }

    #[test]
    fn hostile_framing_is_rejected_and_the_decoder_stays_usable() {
        let good = idr_bytes();
        let mut huge = vec![0x00, 0x80, 0x00, 0x01];
        huge.push(0x67);
        let mut cut = good.clone();
        cut.truncate(cut.len() / 2);
        let cases = [
            (4u8, Vec::new()),
            (4, vec![0, 0, 1, 0x80]),
            (4, vec![0, 0, 1, 0x02]),
            (4, vec![0, 0, 1, 0x0d]),
            (4, huge),
            (4, vec![0, 0, 0, 10, 0x67]),
            (0, cut),
            (0, vec![1]),
        ];
        for (length_size, data) in cases {
            rejects_then_decodes(length_size, data, &good);
        }
        let mut broken = CodecParams::video(CodecId::H264, 16, 16);
        broken.extra = vec![1, 0, 0, 0, 0];
        assert!(matches!(
            H264Decoder::open(&broken),
            Err(CodecError::Malformed { what: "avcC", .. })
        ));
    }

    #[test]
    fn hostile_parameter_sets_are_rejected() {
        let good = idr_bytes();
        let cases = [
            (
                sps_packet(66, SpsTail::Interlaced),
                CodecError::Unsupported {
                    what: "interlaced picture",
                },
            ),
            (
                sps_packet(100, SpsTail::Deep),
                CodecError::Unsupported { what: "bit depth" },
            ),
            (
                sps_packet(66, SpsTail::Wide),
                CodecError::Limit {
                    what: "picture size",
                },
            ),
            (
                sps_packet(66, SpsTail::Refs),
                CodecError::Limit {
                    what: "reference frames",
                },
            ),
        ];
        for (data, want) in cases {
            let mut params = CodecParams::video(CodecId::H264, 16, 16);
            params.nal_length_size = 0;
            let mut decoder = H264Decoder::open(&params).unwrap();
            let err = decoder
                .push(&Packet::video(data, 0, 1, 30, false))
                .unwrap_err();
            assert_eq!(err, want);
            assert_eq!(luma_of(&mut decoder, &good), reference_luma(&good));
        }
    }

    #[test]
    fn random_packets_fail_closed() {
        let good = idr_bytes();
        let want = reference_luma(&good);
        let mut params = CodecParams::video(CodecId::H264, 16, 16);
        params.nal_length_size = 0;
        let mut state = 0xA5A5_1357u32;
        let mut failures = 0u32;
        for _ in 0..64 {
            let mut decoder = H264Decoder::open(&params).unwrap();
            let len = usize::try_from(next_u32(&mut state) % 48).unwrap_or(1) + 1;
            let mut data = Vec::with_capacity(len);
            for _ in 0..len {
                data.push(u8::try_from(next_u32(&mut state) & 0xFF).unwrap_or(0));
            }
            if decoder.push(&Packet::video(data, 0, 1, 30, false)).is_err() {
                failures += 1;
            }
            assert_eq!(luma_of(&mut decoder, &good), want);
        }
        assert!(failures > 50, "{failures}");
    }

    /// Thirty minutes of deterministic garbage and one-bit flips of a real
    /// access unit. `cargo test -p cutlass-h264 --release --lib -- --ignored
    /// thirty_minute_decoder_campaign --nocapture` records the run.
    #[test]
    #[ignore]
    fn thirty_minute_decoder_campaign() {
        let good = idr_bytes();
        let want = reference_luma(&good);
        let mut params = CodecParams::video(CodecId::H264, 16, 16);
        params.nal_length_size = 0;
        let mut decoder = H264Decoder::open(&params).unwrap();
        let started = std::time::Instant::now();
        let mut state = 0xC0FF_EE01u32;
        let mut packets = 0u64;
        let mut failures = 0u64;
        while started.elapsed() < std::time::Duration::from_secs(30 * 60) {
            let data = if packets.is_multiple_of(2) {
                let len = usize::try_from(next_u32(&mut state) % 256).unwrap_or(1) + 1;
                let mut raw = Vec::with_capacity(len);
                for _ in 0..len {
                    raw.push(u8::try_from(next_u32(&mut state) & 0xFF).unwrap_or(0));
                }
                raw
            } else {
                let mut raw = good.clone();
                let at = usize::try_from(next_u32(&mut state)).unwrap_or(0) % raw.len();
                let bit = u8::try_from(next_u32(&mut state) % 8).unwrap_or(0);
                raw[at] ^= 1u8 << bit;
                raw
            };
            if decoder.push(&Packet::video(data, 0, 1, 30, false)).is_err() {
                failures += 1;
            }
            packets += 1;
            if packets.is_multiple_of(2_000) {
                assert_eq!(luma_of(&mut decoder, &good), want);
                decoder = H264Decoder::open(&params).unwrap();
            }
        }
        assert_eq!(luma_of(&mut decoder, &good), want);
        eprintln!("h264 decoder campaign packets={packets} failures={failures}");
        assert!(packets > 1_000, "{packets}");
    }

    /// Release High decode of 1280×720. `cargo test -p cutlass-h264 --release
    /// --lib -- --ignored high_720p_decode_meets_the_release_gates --nocapture`
    /// records frames per second and the resident growth.
    #[test]
    #[ignore]
    fn high_720p_decode_meets_the_release_gates() {
        let Some(ffmpeg) = ffmpeg_bin() else {
            eprintln!("720p gate not run: ffmpeg is absent");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-720p");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("high.h264");
        let made = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=1280x720:rate=30",
                "-frames:v",
                "30",
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=1:bframes=2",
                "-f",
                "h264",
            ])
            .arg(&path)
            .output()
            .unwrap();
        assert!(made.status.success(), "ffmpeg did not make the 720p stream");
        let bytes = std::fs::read(&path).unwrap();
        let units = access_units(&bytes);
        assert!(units.len() >= 30, "access units {}", units.len());
        let mut params = CodecParams::video(CodecId::H264, 1280, 720);
        params.nal_length_size = 0;
        let mut decoder = H264Decoder::open(&params).unwrap();
        let baseline = resident_kib();
        let started = std::time::Instant::now();
        let mut pictures = 0usize;
        let mut peak = baseline;
        for (index, unit) in units.iter().enumerate() {
            let out = decoder
                .push(&Packet::video(
                    unit.clone(),
                    i64::try_from(index).unwrap_or(0),
                    1,
                    30,
                    false,
                ))
                .unwrap();
            pictures += out.len();
            peak = peak.max(resident_kib());
        }
        pictures += decoder.drain().unwrap().len();
        peak = peak.max(resident_kib());
        let elapsed = started.elapsed().as_secs_f64();
        let extra_kib = peak.saturating_sub(baseline);
        eprintln!("720p pictures={pictures} seconds={elapsed:.2} extra_kib={extra_kib}");
        assert_eq!(pictures, 30, "{pictures}");
        assert!(
            elapsed <= 1.0,
            "720p decode {pictures} pictures in {elapsed:.2}s"
        );
        assert!(extra_kib <= 64 * 1024, "720p decode grew {extra_kib} KiB");
    }

    fn rejects_then_decodes(length_size: u8, data: Vec<u8>, good: &[u8]) {
        let mut params = CodecParams::video(CodecId::H264, 16, 16);
        params.nal_length_size = length_size;
        let mut decoder = H264Decoder::open(&params).unwrap();
        assert!(decoder.push(&Packet::video(data, 0, 1, 30, false)).is_err());
        assert_eq!(luma_of(&mut decoder, good), reference_luma(good));
    }

    fn reference_luma(good: &[u8]) -> Vec<u8> {
        let mut params = CodecParams::video(CodecId::H264, 16, 16);
        params.nal_length_size = 0;
        let mut decoder = H264Decoder::open(&params).unwrap();
        luma_of(&mut decoder, good)
    }

    fn luma_of(decoder: &mut H264Decoder, data: &[u8]) -> Vec<u8> {
        let mut frames = decoder
            .push(&Packet::video(data.to_vec(), 0, 1, 30, true))
            .unwrap();
        frames.extend(decoder.drain().unwrap());
        frames
            .iter()
            .find(|frame| frame.picture.width() == 16)
            .expect("idr")
            .picture
            .y()
            .to_vec()
    }

    fn idr_bytes() -> Vec<u8> {
        let mut params = CodecParams::video(CodecId::H264, 16, 16);
        params.nal_length_size = 0;
        let mut encoder = H264Encoder::open(&params).unwrap();
        encoder
            .push(&VideoInput {
                picture: Yuv420p::solid(16, 16, 40, 80, 160).unwrap(),
                pts: 0,
                duration: 1,
                timescale: 30,
                force_keyframe: true,
            })
            .unwrap()
            .into_iter()
            .next()
            .expect("idr")
            .data
    }

    enum SpsTail {
        Interlaced,
        Deep,
        Wide,
        Refs,
    }

    fn sps_packet(profile: u32, tail: SpsTail) -> Vec<u8> {
        let mut bits = Pack::new();
        bits.u(8, profile);
        bits.u(8, 0);
        bits.u(8, 30);
        bits.ue(0);
        if profile == 100 {
            bits.ue(1);
            bits.ue(2);
            bits.ue(0);
        } else {
            bits.ue(0);
            bits.ue(0);
            bits.ue(0);
            match tail {
                SpsTail::Refs => bits.ue(17),
                SpsTail::Wide => {
                    bits.ue(1);
                    bits.bit(false);
                    bits.ue(256);
                    bits.ue(0);
                }
                SpsTail::Interlaced | SpsTail::Deep => {
                    bits.ue(1);
                    bits.bit(false);
                    bits.ue(0);
                    bits.ue(0);
                    bits.bit(false);
                }
            }
        }
        crate::nal::annex_b_nal(3, 7, &bits.finish())
    }

    struct Pack {
        bytes: Vec<u8>,
        acc: u32,
        n: u32,
    }

    impl Pack {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                acc: 0,
                n: 0,
            }
        }

        fn bit(&mut self, one: bool) {
            self.acc = (self.acc << 1) | u32::from(one);
            self.n += 1;
            if self.n == 8 {
                self.bytes.push(u8::try_from(self.acc).unwrap_or(0));
                self.acc = 0;
                self.n = 0;
            }
        }

        fn u(&mut self, n: u32, value: u32) {
            for shift in (0..n).rev() {
                self.bit(((value >> shift) & 1) == 1);
            }
        }

        fn ue(&mut self, value: u32) {
            let code = value.saturating_add(1);
            let width = 31 - code.leading_zeros();
            for _ in 0..width {
                self.bit(false);
            }
            self.u(width + 1, code);
        }

        fn finish(mut self) -> Vec<u8> {
            if self.n > 0 {
                self.acc <<= 8 - self.n;
                self.bytes.push(u8::try_from(self.acc).unwrap_or(0));
            }
            self.bytes
        }
    }

    fn next_u32(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *state
    }

    fn ffmpeg_bin() -> Option<String> {
        [
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "ffmpeg",
        ]
        .into_iter()
        .find(|path| Command::new(path).arg("-version").output().is_ok())
        .map(str::to_string)
    }

    fn resident_kib() -> u64 {
        let out = Command::new("ps")
            .args(["-o", "rss=", "-p"])
            .arg(std::process::id().to_string())
            .output()
            .unwrap();
        assert!(out.status.success(), "ps did not report rss");
        let text = String::from_utf8_lossy(&out.stdout);
        text.split_whitespace()
            .next()
            .and_then(|token| token.parse().ok())
            .expect("rss")
    }
}

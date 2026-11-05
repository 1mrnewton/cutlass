//! The decoder: parameter sets, picture order counts, reference picture
//! sets and lists (clause 8.3), the decoded picture buffer and output order
//! (clause C.5.2), and the engine's decoder interfaces.

use std::sync::Arc;

use cutlass_decode::{
    CodecError, CodecId, CodecParams, DecodeFactory, Decoder, Output, Packet, PictureInfo,
    PictureOutput, PixelFormat, SequenceTiming, VideoDecoder, VideoOutput, Yuv420High, Yuv420p,
    Yuv420p10,
};

use crate::cabac::{CONTEXTS, Cabac};
use crate::ctu::{Cu, SliceDecoder};
use crate::nal::{self, Nal};
use crate::picture::{ColMv, Decoded, Picture, RefPic, SliceCtx};
use crate::ps::{self, Layout, Pps, Sps};
use crate::slice::{self, SliceHeader, SliceType};

const MAX_SPS: usize = 16;
const MAX_PPS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Marking {
    Unused,
    Short,
    Long,
}

/// What an output picture carries besides its samples.
#[derive(Clone, Debug)]
struct Meta {
    pts: i64,
    duration: i64,
    timescale: u32,
    keyframe: bool,
    sps: Arc<Sps>,
}

#[derive(Clone)]
struct DpbPic {
    pic: Arc<Decoded>,
    poc: i32,
    marking: Marking,
    /// Waiting for output.
    output: bool,
    uid: u64,
    meta: Option<Arc<Meta>>,
}

/// Everything a push may change. A failed push leaves the last good copy.
#[derive(Clone)]
struct State {
    sps: Vec<Option<Arc<Sps>>>,
    pps: Vec<Option<Arc<Pps>>>,
    dpb: Vec<DpbPic>,
    /// `PicOrderCntVal` of the previous `TemporalId` 0 picture.
    prev_tid0_poc: i32,
    /// An IRAP picture has started a coded video sequence.
    started: bool,
    /// `NoRaslOutputFlag` of the last IRAP picture.
    no_rasl_output: bool,
    /// An end of sequence came before the next picture.
    after_eos: bool,
    next_uid: u64,
}

impl State {
    fn new() -> Self {
        Self {
            sps: vec![None; MAX_SPS],
            pps: vec![None; MAX_PPS],
            dpb: Vec::new(),
            prev_tid0_poc: 0,
            started: false,
            no_rasl_output: true,
            after_eos: false,
            next_uid: 1,
        }
    }

    fn uid(&mut self) -> u64 {
        let uid = self.next_uid;
        self.next_uid += 1;
        uid
    }

    fn store_param(&mut self, nal: &Nal) -> Result<(), CodecError> {
        match nal.kind {
            nal::SPS => {
                let sps = ps::parse_sps(&nal.rbsp)?;
                let id = usize::from(sps.id);
                self.sps[id] = Some(Arc::new(sps));
            }
            nal::PPS => {
                let pps = ps::parse_pps(&nal.rbsp)?;
                let id = usize::from(pps.id);
                self.pps[id] = Some(Arc::new(pps));
            }
            _ => {}
        }
        Ok(())
    }

    fn sets(&self, pps_id: u8) -> Option<(Arc<Sps>, Arc<Pps>)> {
        let pps = self.pps.get(usize::from(pps_id))?.clone()?;
        let sps = self.sps.get(usize::from(pps.sps_id))?.clone()?;
        Some((sps, pps))
    }

    /// The bumping process: output the picture with the smallest order
    /// count while too many wait (clause C.5.2.4, as the reference decoder
    /// counts them).
    fn bump(&mut self, max_output: usize, max_dpb: usize, out: &mut Vec<VideoOutput>) {
        loop {
            let waiting = self.dpb.iter().filter(|p| p.output).count();
            if waiting == 0 || (waiting <= max_output && self.dpb.len() <= max_dpb) {
                break;
            }
            let Some(index) = self
                .dpb
                .iter()
                .enumerate()
                .filter(|(_, p)| p.output)
                .min_by_key(|(_, p)| p.poc)
                .map(|(i, _)| i)
            else {
                break;
            };
            self.dpb[index].output = false;
            if let Some(frame) = emit(&self.dpb[index]) {
                out.push(frame);
            }
            self.dpb
                .retain(|p| p.output || p.marking != Marking::Unused);
        }
    }

    fn flush(&mut self, out: &mut Vec<VideoOutput>) {
        self.bump(0, usize::MAX, out);
        self.dpb
            .retain(|p| p.output || p.marking != Marking::Unused);
    }
}

/// The picture being decoded, with its reference picture set.
struct Pending {
    pic: Picture,
    output: bool,
    meta: Arc<Meta>,
    /// The last independent slice segment header, for dependent ones.
    independent: Option<SliceHeader>,
    slice_addr: u32,
    st_before: Vec<RefPic>,
    st_after: Vec<RefPic>,
    lt_curr: Vec<RefPic>,
    uid: u64,
}

/// Native HEVC (H.265) decoder: Main and Main 10, 4:2:0.
pub struct HevcDecoder {
    length_size: u8,
    extra: Vec<u8>,
    state: State,
    drained: bool,
}

impl HevcDecoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Hevc {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        let mut dec = Self {
            length_size: params.nal_length_size,
            extra: params.extra.clone(),
            state: State::new(),
            drained: false,
        };
        dec.apply_extra()?;
        Ok(dec)
    }

    fn apply_extra(&mut self) -> Result<(), CodecError> {
        if self.extra.is_empty() {
            return Ok(());
        }
        let nals = if nal::is_hvcc(&self.extra) {
            let (size, nals) = nal::hvcc(&self.extra)?;
            self.length_size = size;
            nals
        } else {
            nal::access_unit(&self.extra, 0)?
        };
        for nal in &nals {
            self.state.store_param(nal)?;
        }
        Ok(())
    }

    fn decode(&self, state: &mut State, packet: &Packet) -> Result<Vec<VideoOutput>, CodecError> {
        let nals = nal::access_unit(&packet.data, self.length_size)?;
        let mut out = Vec::new();
        let mut current: Option<Pending> = None;
        let mut skipping = false;
        for nal in &nals {
            if nal.layer != 0 {
                continue;
            }
            match nal.kind {
                nal::SPS | nal::PPS => state.store_param(nal)?,
                nal::EOS | nal::EOB => {
                    if let Some(done) = current.take() {
                        finish(state, done, &mut out);
                    }
                    state.after_eos = true;
                }
                _ if nal.is_slice() => {
                    let first = nal.rbsp.first().is_some_and(|b| b & 0x80 != 0);
                    if first {
                        if let Some(done) = current.take() {
                            finish(state, done, &mut out);
                        }
                        let (header, _, _) = parse(state, nal, None)?;
                        current = start_picture(state, nal, &header, packet, &mut out)?;
                        skipping = current.is_none();
                        if let Some(pending) = current.as_mut() {
                            decode_slice(pending, nal, header)?;
                        }
                    } else if !skipping && let Some(pending) = current.as_mut() {
                        let (header, _, _) = parse(state, nal, pending.independent.as_ref())?;
                        decode_slice(pending, nal, header)?;
                    }
                }
                _ => {}
            }
        }
        if let Some(done) = current.take() {
            finish(state, done, &mut out);
        }
        Ok(out)
    }
}

fn parse(
    state: &State,
    nal: &Nal,
    prev: Option<&SliceHeader>,
) -> Result<(SliceHeader, Sps, Pps), CodecError> {
    let lookup = |id: u8| state.sets(id).map(|(s, p)| ((*s).clone(), (*p).clone()));
    let prev = prev.cloned();
    slice::parse_header(nal, &lookup, prev.as_ref())
}

/// Start a picture at its first slice segment: order count, reference
/// picture set, missing references, and the output of prior pictures.
/// `None` skips the picture.
fn start_picture(
    state: &mut State,
    nal: &Nal,
    header: &SliceHeader,
    packet: &Packet,
    out: &mut Vec<VideoOutput>,
) -> Result<Option<Pending>, CodecError> {
    let Some((sps, pps)) = state.sets(header.pps_id) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "slice references a missing parameter set",
        });
    };
    let irap = nal.is_irap();
    if irap {
        state.no_rasl_output = nal.is_idr() || nal.is_bla() || !state.started || state.after_eos;
        state.started = true;
    } else if !state.started {
        return Ok(None);
    }
    if nal.is_rasl() && state.no_rasl_output {
        return Ok(None);
    }
    let new_sequence = irap && state.no_rasl_output;
    let flushed_by_eos = state.after_eos;
    state.after_eos = false;

    // Picture order count (clause 8.3.1).
    let max_lsb = 1i32 << sps.log2_max_poc_lsb;
    let lsb = header.poc_lsb as i32;
    let msb = if new_sequence {
        0
    } else {
        let prev_lsb = state.prev_tid0_poc & (max_lsb - 1);
        let prev_msb = state.prev_tid0_poc - prev_lsb;
        if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
            prev_msb + max_lsb
        } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
            prev_msb - max_lsb
        } else {
            prev_msb
        }
    };
    let poc = msb + lsb;
    if nal.temporal_id == 0 && !nal.is_rasl() && !nal.is_radl() && !nal.is_sub_layer_non_ref() {
        state.prev_tid0_poc = poc;
    }

    // Output of prior pictures at a new sequence (clause C.5.2.2).
    if new_sequence {
        let discard = header.no_output_of_prior_pics || (nal.kind == nal::CRA && flushed_by_eos);
        if discard {
            state.dpb.clear();
        } else {
            for p in &mut state.dpb {
                p.marking = Marking::Unused;
            }
            state.flush(out);
        }
        state.dpb.clear();
    }

    // Reference picture set (clause 8.3.2).
    let mut keep = vec![Marking::Unused; state.dpb.len()];
    let mut lt_curr = Vec::new();
    for lt in &header.long_term {
        let (value, exact) = if lt.msb {
            (poc + lt.poc, true)
        } else {
            (lt.poc, false)
        };
        let found = state.dpb.iter().position(|p| {
            p.marking != Marking::Unused
                && if exact {
                    p.poc == value
                } else {
                    p.poc & (max_lsb - 1) == value
                }
        });
        if let Some(i) = found {
            keep[i] = Marking::Long;
        }
        if lt.used {
            lt_curr.push((value, found));
        }
    }
    let mut st_lists = [Vec::new(), Vec::new()];
    let rps = &header.st_rps;
    let deltas = [
        (&rps.negative, &rps.negative_used, 0usize),
        (&rps.positive, &rps.positive_used, 1usize),
    ];
    for (values, used, list) in deltas {
        for (&delta, &used) in values.iter().zip(used.iter()) {
            let target = poc + delta;
            let found = state.dpb.iter().enumerate().position(|(i, p)| {
                p.marking == Marking::Short && keep[i] != Marking::Long && p.poc == target
            });
            if let Some(i) = found {
                keep[i] = Marking::Short;
            }
            if used {
                st_lists[list].push((target, found));
            }
        }
    }
    for (p, mark) in state.dpb.iter_mut().zip(&keep) {
        p.marking = *mark;
    }
    // Generate what is missing (clause 8.3.3), mid-grey and not output.
    let (width, height) = (sps.width as usize, sps.height as usize);
    let resolve =
        |state: &mut State, wanted: Vec<(i32, Option<usize>)>, long: bool| -> Vec<RefPic> {
            wanted
                .into_iter()
                .map(|(value, found)| {
                    let index = found.unwrap_or_else(|| {
                        let uid = state.uid();
                        state.dpb.push(DpbPic {
                            pic: Arc::new(Decoded::missing(width, height, sps.bit_depth, value)),
                            poc: value,
                            marking: if long { Marking::Long } else { Marking::Short },
                            output: false,
                            uid,
                            meta: None,
                        });
                        state.dpb.len() - 1
                    });
                    let p = &state.dpb[index];
                    RefPic {
                        pic: p.pic.clone(),
                        poc: p.poc,
                        long_term: long,
                        uid: p.uid,
                    }
                })
                .collect()
        };
    let [before, after] = st_lists;
    let st_before = resolve(state, before, false);
    let st_after = resolve(state, after, false);
    let lt_curr = resolve(state, lt_curr, true);
    state
        .dpb
        .retain(|p| p.output || p.marking != Marking::Unused);

    let layout = Arc::new(Layout::new(&sps, &pps)?);
    let pic = Picture::new(sps.clone(), pps.clone(), layout, poc)?;
    let output = header.pic_output && !(nal.is_rasl() && state.no_rasl_output);
    let uid = state.uid();
    Ok(Some(Pending {
        pic,
        output,
        meta: Arc::new(Meta {
            pts: packet.pts,
            duration: packet.duration,
            timescale: packet.timescale,
            keyframe: irap,
            sps,
        }),
        independent: None,
        slice_addr: 0,
        st_before,
        st_after,
        lt_curr,
        uid,
    }))
}

/// Reference picture lists of one slice (clause 8.3.4).
fn ref_lists(pending: &Pending, header: &SliceHeader) -> Result<[Vec<RefPic>; 2], CodecError> {
    if header.slice_type == SliceType::I {
        return Ok([Vec::new(), Vec::new()]);
    }
    let total = pending.st_before.len() + pending.st_after.len() + pending.lt_curr.len();
    if total == 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "inter slice without references",
        });
    }
    let mut lists = [Vec::new(), Vec::new()];
    let count = if header.slice_type == SliceType::B {
        2
    } else {
        1
    };
    for (list, out) in lists.iter_mut().enumerate().take(count) {
        let wanted = header.num_ref_idx[list];
        let size = wanted.max(total);
        let order: [&Vec<RefPic>; 3] = if list == 0 {
            [&pending.st_before, &pending.st_after, &pending.lt_curr]
        } else {
            [&pending.st_after, &pending.st_before, &pending.lt_curr]
        };
        let mut temp = Vec::with_capacity(size);
        while temp.len() < size {
            for part in order {
                for r in part {
                    if temp.len() < size {
                        temp.push(r.clone());
                    }
                }
            }
        }
        for i in 0..wanted {
            let index = match &header.list_entry[list] {
                Some(entries) => *entries.get(i).ok_or(CodecError::Malformed {
                    at_bit: 0,
                    what: "list_entry",
                })? as usize,
                None => i,
            };
            let Some(r) = temp.get(index) else {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "list_entry",
                });
            };
            out.push(r.clone());
        }
    }
    Ok(lists)
}

fn decode_slice(pending: &mut Pending, nal: &Nal, header: SliceHeader) -> Result<(), CodecError> {
    if !header.dependent {
        pending.slice_addr = header.segment_address;
        pending.independent = Some(header.clone());
    } else if pending.independent.is_none() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "dependent slice without a slice",
        });
    }
    if pending.pic.pps.id != header.pps_id {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "pps changed inside a picture",
        });
    }
    let refs = ref_lists(pending, &header)?;
    let poc = pending.pic.poc;
    let no_backward_pred = refs.iter().flatten().all(|r| r.poc <= poc);
    let qp = pending.pic.pps.init_qp + header.qp_delta;
    let slice_type = header.slice_type;
    // Substream starts, from coded offsets to RBSP offsets in the data.
    let start = header.data_offset;
    let Some(data) = nal.rbsp.get(start..) else {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "slice data",
        });
    };
    let raw_start = nal.raw_of(start);
    let mut entries = Vec::with_capacity(header.entry_points.len());
    let mut raw = raw_start;
    for &offset in &header.entry_points {
        raw = raw.saturating_add(offset as usize);
        let at = nal.rbsp_of(raw).saturating_sub(start);
        if at > data.len() {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "entry point",
            });
        }
        entries.push(at);
    }
    pending.pic.slices.push(SliceCtx {
        header,
        refs,
        slice_addr: pending.slice_addr,
        qp,
        no_backward_pred,
    });
    let index = pending.pic.slices.len() - 1;
    if index >= usize::from(u16::MAX) {
        return Err(CodecError::Limit { what: "slices" });
    }
    let sps = pending.pic.sps.clone();
    let pps = pending.pic.pps.clone();
    let mut decoder = SliceDecoder {
        pic: &mut pending.pic,
        slice: index,
        sps: &sps,
        pps: &pps,
        cabac: Cabac::new(data, [0; CONTEXTS]),
        data,
        entries,
        next_entry: 0,
        qp_y: qp,
        qp_pred: qp,
        last_qp: qp,
        first_qg: true,
        qp_delta_coded: false,
        qp_delta: 0,
        cu: Cu::default(),
        coeffs: [0; 32 * 32],
        pred: [[0; 64 * 64]; 2],
        ctb_x: 0,
        ctb_y: 0,
        ctb_rs: 0,
        merge_2nx2n: false,
        slice_type,
        mc: crate::mc::Scratch::default(),
    };
    decoder.decode()
}

/// Loop filters, the motion kept for later pictures, and the decoded
/// picture buffer (clause C.5.2.3).
fn finish(state: &mut State, pending: Pending, out: &mut Vec<VideoOutput>) {
    let Pending {
        mut pic,
        output,
        meta,
        uid,
        ..
    } = pending;
    pic.deblock();
    pic.sao();
    let col_stride = pic.width.div_ceil(16);
    let col_rows = pic.height.div_ceil(16);
    let mut col = vec![ColMv::default(); col_stride * col_rows];
    for by in 0..col_rows {
        for bx in 0..col_stride {
            let (x, y) = (bx * 16, by * 16);
            let field = pic.mvf[pic.at4(x, y)];
            if field.pred == 0 {
                continue;
            }
            let Some(slice) = pic.slice_of(x, y) else {
                continue;
            };
            let mut cell = ColMv {
                pred: field.pred,
                ..ColMv::default()
            };
            for list in 0..2 {
                if !field.uses(list) {
                    continue;
                }
                let Some(r) = usize::try_from(field.ref_idx[list])
                    .ok()
                    .and_then(|i| slice.refs[list].get(i))
                else {
                    continue;
                };
                cell.mv[list] = field.mv[list];
                cell.poc[list] = r.poc;
                cell.long_term[list] = r.long_term;
            }
            col[by * col_stride + bx] = cell;
        }
    }
    let sps = pic.sps.clone();
    let poc = pic.poc;
    let decoded = Decoded {
        width: pic.width,
        height: pic.height,
        planes: std::mem::take(&mut pic.planes),
        col,
        col_stride,
        poc,
    };
    state.dpb.push(DpbPic {
        pic: Arc::new(decoded),
        poc,
        marking: Marking::Short,
        output,
        uid,
        meta: Some(meta),
    });
    let top = usize::from(sps.max_sub_layers.saturating_sub(1)).min(7);
    let reorder = usize::from(sps.max_num_reorder[top]);
    let max_dpb = usize::from(sps.max_dec_pic_buffering[top]);
    state.bump(reorder, max_dpb, out);
}

/// The cropped output picture, 8-bit and, above 8 bits, the full samples.
fn emit(p: &DpbPic) -> Option<VideoOutput> {
    let meta = p.meta.as_ref()?;
    let sps = &meta.sps;
    let pic = &p.pic;
    let [left, right, top, bottom] = sps.crop.map(|v| v as usize);
    let w = pic.width.checked_sub(left + right)?;
    let h = pic.height.checked_sub(top + bottom)?;
    let bits = u32::from(sps.bit_depth);
    let crop =
        |plane: &[u16], stride: usize, x0: usize, y0: usize, w: usize, h: usize| -> Vec<u16> {
            let mut outp = Vec::with_capacity(w * h);
            for row in 0..h {
                let start = (y0 + row) * stride + x0;
                outp.extend_from_slice(&plane[start..start + w]);
            }
            outp
        };
    let y = crop(&pic.planes[0], pic.width, left, top, w, h);
    let u = crop(
        &pic.planes[1],
        pic.width / 2,
        left / 2,
        top / 2,
        w / 2,
        h / 2,
    );
    let v = crop(
        &pic.planes[2],
        pic.width / 2,
        left / 2,
        top / 2,
        w / 2,
        h / 2,
    );
    let shift = bits - 8;
    let eight = |plane: &[u16]| -> Vec<u8> { plane.iter().map(|&s| (s >> shift) as u8).collect() };
    let picture = Yuv420p::try_new(w as u32, h as u32, eight(&y), eight(&u), eight(&v)).ok()?;
    let (yuv420p10, yuv420_high) = match bits {
        8 => (None, None),
        10 => (Yuv420p10::try_new(w as u32, h as u32, y, u, v).ok(), None),
        _ => (
            None,
            Yuv420High::try_new(bits as u8, w as u32, h as u32, y, u, v).ok(),
        ),
    };
    let vui = &sps.vui;
    Some(VideoOutput {
        picture,
        yuv420p10,
        yuv420_high,
        yuv422: None,
        yuv422_high: None,
        yuv444: None,
        yuv444_high: None,
        pts: meta.pts,
        duration: meta.duration,
        timescale: meta.timescale,
        keyframe: meta.keyframe,
        range: vui.range,
        primaries: vui.primaries,
        transfer: vui.transfer,
        matrix: vui.matrix,
        sample_aspect: vui.sample_aspect,
        timing: SequenceTiming {
            num_units: vui.num_units,
            time_scale: vui.time_scale,
            fixed: vui.fixed_rate,
        },
    })
}

impl VideoDecoder for HevcDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Hevc
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<VideoOutput>, CodecError> {
        if packet.data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        let mut state = self.state.clone();
        let out = self.decode(&mut state, packet)?;
        self.state = state;
        self.drained = false;
        Ok(out)
    }

    fn drain(&mut self) -> Result<Vec<VideoOutput>, CodecError> {
        if self.drained {
            return Ok(Vec::new());
        }
        self.drained = true;
        let mut out = Vec::new();
        self.state.flush(&mut out);
        Ok(out)
    }

    fn reset(&mut self) {
        self.state = State::new();
        self.drained = false;
        // The extradata parsed when the decoder opened, so it parses again.
        let _ = self.apply_extra();
    }
}

pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Hevc, open_registered);
    factory.register_video(CodecId::Hevc, open_decoder);
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError> {
    Ok(Box::new(HevcDecoder::open(params)?))
}

fn open_registered(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(AccessDecoder {
        inner: HevcDecoder::open(params)?,
    }))
}

/// The engine's decoder. It returns one packed picture per displayed frame.
struct AccessDecoder {
    inner: HevcDecoder,
}

impl Decoder for AccessDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Hevc
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        self.inner
            .push(packet)?
            .into_iter()
            .map(picture_output)
            .collect()
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        self.inner
            .drain()?
            .into_iter()
            .map(picture_output)
            .collect()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

fn picture_output(frame: VideoOutput) -> Result<Output, CodecError> {
    let picture = &frame.picture;
    let mut bytes = Vec::with_capacity(picture.y().len() + picture.u().len() + picture.v().len());
    bytes.extend_from_slice(picture.y());
    bytes.extend_from_slice(picture.u());
    bytes.extend_from_slice(picture.v());
    let picture = PictureOutput::cpu(
        PictureInfo {
            width: picture.width(),
            height: picture.height(),
            pixel: PixelFormat::Yuv420p,
            primaries: frame.primaries,
            transfer: frame.transfer,
        },
        bytes,
        frame.pts,
        frame.timescale,
    )?;
    Ok(Output::Picture(picture))
}

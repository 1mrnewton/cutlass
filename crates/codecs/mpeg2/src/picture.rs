//! Encoding of one progressive frame picture: motion and mode decisions,
//! then the picture header, its coding extension, and one
//! slice per macroblock row (H.262 6.2.3 to 6.2.6).
//!
//! The writer keeps the same predictor state a decoder keeps (7.2.1,
//! 7.6.3.3) so every skip, vector difference, and DC difference it emits
//! decodes to the samples it predicted from.

use cutlass_decode::CodecError;

use crate::bits::BitWriter;
use crate::dct::fdct;
use crate::frame::{Frame, Geometry, View};
use crate::headers::{
    Matrix, PICTURE_B, PICTURE_I, PICTURE_P, PictureCodingExtension, PictureHeader, STRUCTURE_FRAME,
};
use crate::mc::{Target, predict};
use crate::motion::{Found, Search, sad_bi};
use crate::rate::PictureRate;
use crate::tables::{
    CBP_ENC, COEF_ENC_ONE, COEF_ENC_ZERO, CoefEncode, DC_CHROMA_ENC, DC_LUMA_ENC, EOB_ONE,
    EOB_ZERO, ESCAPE, MB_ADDR_ENC, MB_BACKWARD, MB_FORWARD, MB_INTRA, MB_PATTERN, MB_QUANT,
    MB_TYPE_B_ENC, MB_TYPE_I_ENC, MB_TYPE_P_ENC, MOTION_ENC, SCAN,
};

/// Added to the intra cost before it competes with the inter SAD: an intra
/// macroblock sends every DC and loses the prediction.
const INTRA_BIAS: u32 = 512;

/// How a macroblock is predicted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Intra,
    Forward,
    Backward,
    Both,
}

impl Mode {
    /// Forward, backward.
    pub(crate) fn dir(self) -> [bool; 2] {
        match self {
            Mode::Intra => [false, false],
            Mode::Forward => [true, false],
            Mode::Backward => [false, true],
            Mode::Both => [true, true],
        }
    }
}

/// The decision for one macroblock. `mv[s]` is the frame vector in half
/// samples for direction `s`, zero when that direction is unused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Plan {
    pub(crate) mode: Mode,
    pub(crate) mv: [[i32; 2]; 2],
}

const INTRA_PLAN: Plan = Plan {
    mode: Mode::Intra,
    mv: [[0; 2]; 2],
};

/// Decisions for a whole picture.
#[derive(Clone, Debug)]
pub(crate) struct Analysis {
    pub(crate) plans: Vec<Plan>,
    /// `f_code[s][t]`, 15 for a direction the picture does not use.
    pub(crate) f_code: [[u8; 2]; 2],
}

/// Sum of absolute deviations from the mean of one luma macroblock: what an
/// intra macroblock has to code.
fn intra_cost(geo: &Geometry, src: &Frame, mbx: usize, mby: usize) -> u32 {
    let (stride, _) = geo.plane(0);
    let rows = |row: usize| &src.planes[0][(mby * 16 + row) * stride + mbx * 16..][..16];
    let sum: u32 = (0..16)
        .map(|row| rows(row).iter().map(|&v| u32::from(v)).sum::<u32>())
        .sum();
    let mean = (sum + 128) / 256;
    (0..16)
        .map(|row| {
            rows(row)
                .iter()
                .map(|&v| u32::from(v).abs_diff(mean))
                .sum::<u32>()
        })
        .sum()
}

/// Picks a mode and vectors for every macroblock. `refs` are the forward
/// and backward reference reconstructions the picture type uses; `lambda`
/// weighs vector bits against SAD.
pub(crate) fn analyse(
    geo: &Geometry,
    kind: u8,
    src: &Frame,
    refs: [Option<&Frame>; 2],
    lambda: u32,
) -> Analysis {
    let (mb_w, mb_h) = (geo.mb_width, geo.mb_height);
    let mut plans = vec![INTRA_PLAN; mb_w * mb_h];
    let searches = refs.map(|r| r.map(|r| Search::new(geo, src, r, lambda)));
    if kind != PICTURE_I {
        for mby in 0..mb_h {
            // The bitstream predictors restart with every slice, one per row.
            let mut pmv = [[0i32; 2]; 2];
            let mut prev: Option<Plan> = None;
            for mbx in 0..mb_w {
                let addr = mby * mb_w + mbx;
                let mut found: [Option<Found>; 2] = [None, None];
                for (s, search) in searches.iter().enumerate() {
                    let Some(search) = search else {
                        continue;
                    };
                    let mut candidates = [[0i32; 2]; 2];
                    let mut count = 0;
                    if mby > 0 {
                        for above in [Some(addr - mb_w), (mbx + 1 < mb_w).then(|| addr - mb_w + 1)]
                            .into_iter()
                            .flatten()
                        {
                            let plan = plans[above];
                            if plan.mode.dir()[s] {
                                candidates[count] = plan.mv[s];
                                count += 1;
                            }
                        }
                    }
                    let mut best = search.block(mbx, mby, pmv[s], &candidates[..count]);
                    if kind == PICTURE_P && best.mv != [0, 0] {
                        // A zero vector can be skipped or sent without one.
                        let zero = search.evaluate(mbx, mby, pmv[s], [0, 0]);
                        if zero.sad <= best.cost {
                            best = zero;
                        }
                    }
                    found[s] = Some(best);
                }
                let (mut plan, inter_cost, inter_sad) = match found {
                    [Some(f), None] => (
                        Plan {
                            mode: Mode::Forward,
                            mv: [f.mv, [0, 0]],
                        },
                        f.cost,
                        f.sad,
                    ),
                    [None, Some(b)] => (
                        Plan {
                            mode: Mode::Backward,
                            mv: [[0, 0], b.mv],
                        },
                        b.cost,
                        b.sad,
                    ),
                    [Some(f), Some(b)] => {
                        let both = [f.mv, b.mv];
                        let mut options = [
                            (Mode::Forward, [f.mv, [0, 0]], f.cost, f.sad),
                            (Mode::Backward, [[0, 0], b.mv], b.cost, b.sad),
                            (Mode::Both, both, 0, 0),
                        ];
                        if let [Some(fr), Some(br)] = refs {
                            let sad = sad_bi(geo, src, [fr, br], mbx, mby, both);
                            options[2].2 = sad + (f.cost - f.sad) + (b.cost - b.sad);
                            options[2].3 = sad;
                        }
                        let mut best = options[0];
                        for option in &options[1..] {
                            if option.2 < best.2 {
                                best = *option;
                            }
                        }
                        (
                            Plan {
                                mode: best.0,
                                mv: best.1,
                            },
                            best.2,
                            best.3,
                        )
                    }
                    [None, None] => (INTRA_PLAN, u32::MAX, u32::MAX),
                };
                // Repeating the previous macroblock's prediction lets a B
                // macroblock be skipped and costs no vector bits.
                if kind == PICTURE_B
                    && let Some(previous) = prev
                    && previous.mode != Mode::Intra
                    && previous != plan
                    && let Some(sad) = repeat_sad(geo, src, &searches, refs, mbx, mby, previous)
                    && sad <= inter_cost
                {
                    plan = previous;
                }
                if intra_cost(geo, src, mbx, mby).saturating_add(INTRA_BIAS) < inter_sad {
                    plan = INTRA_PLAN;
                }
                match plan.mode {
                    Mode::Intra => pmv = [[0; 2]; 2],
                    mode => {
                        for (s, used) in mode.dir().into_iter().enumerate() {
                            if used {
                                pmv[s] = plan.mv[s];
                            }
                        }
                    }
                }
                prev = Some(plan);
                plans[addr] = plan;
            }
        }
    }
    let mut f_code = [[15u8; 2]; 2];
    let used = match kind {
        PICTURE_P => 1,
        PICTURE_B => 2,
        _ => 0,
    };
    for (s, codes) in f_code.iter_mut().enumerate().take(used) {
        for (t, code) in codes.iter_mut().enumerate() {
            let (low, high) = plans
                .iter()
                .filter(|plan| plan.mode.dir()[s])
                .fold((0, 0), |(low, high), plan| {
                    (low.min(plan.mv[s][t]), high.max(plan.mv[s][t]))
                });
            *code = (1u8..=9)
                .find(|&f| {
                    let range = 16i32 << (f - 1);
                    low >= -range && high < range
                })
                .unwrap_or(9);
        }
    }
    Analysis { plans, f_code }
}

/// SAD of predicting `mbx, mby` with exactly the vectors of `plan`, or
/// `None` when they leave the picture there.
fn repeat_sad(
    geo: &Geometry,
    src: &Frame,
    searches: &[Option<Search<'_>>; 2],
    refs: [Option<&Frame>; 2],
    mbx: usize,
    mby: usize,
    plan: Plan,
) -> Option<u32> {
    let (w, h) = geo.plane(0);
    for (s, used) in plan.mode.dir().into_iter().enumerate() {
        let x2 = (mbx * 32) as i32 + plan.mv[s][0];
        let y2 = (mby * 32) as i32 + plan.mv[s][1];
        if used && (x2 < 0 || y2 < 0 || x2 > 2 * (w as i32 - 16) || y2 > 2 * (h as i32 - 16)) {
            return None;
        }
    }
    match (plan.mode, searches, refs) {
        (Mode::Forward, [Some(search), _], _) => {
            Some(search.evaluate(mbx, mby, plan.mv[0], plan.mv[0]).sad)
        }
        (Mode::Backward, [_, Some(search)], _) => {
            Some(search.evaluate(mbx, mby, plan.mv[1], plan.mv[1]).sad)
        }
        (Mode::Both, _, [Some(f), Some(b)]) => Some(sad_bi(geo, src, [f, b], mbx, mby, plan.mv)),
        _ => None,
    }
}

/// One picture to write.
pub(crate) struct Job<'a> {
    pub(crate) geo: Geometry,
    pub(crate) kind: u8,
    pub(crate) temporal_reference: u16,
    pub(crate) vbv_delay: u16,
    pub(crate) src: &'a Frame,
    /// Forward and backward reconstructions.
    pub(crate) refs: [Option<&'a Frame>; 2],
    pub(crate) quant: &'a Quantisers,
}

/// A written picture and where each slice starts.
pub(crate) struct Coded {
    pub(crate) data: Vec<u8>,
    /// `(slice_vertical_position, byte offset after the start code)`.
    pub(crate) slices: Vec<(u8, usize)>,
}

/// Reciprocal quantiser steps for each `quantiser_scale_code`, with
/// `q_scale_type` 0: `16 / (W · 2 · code)`.
pub(crate) struct Quantisers {
    intra: [[f32; 64]; 32],
    non_intra: [[f32; 64]; 32],
}

impl Quantisers {
    pub(crate) fn new(intra: &Matrix, non_intra: &Matrix) -> Box<Self> {
        let mut out = Box::new(Self {
            intra: [[0.0; 64]; 32],
            non_intra: [[0.0; 64]; 32],
        });
        for code in 1..32 {
            let scale = 2.0 * code as f32;
            for k in 0..64 {
                out.intra[code][k] = 16.0 / (f32::from(intra[k]) * scale);
                out.non_intra[code][k] = 16.0 / (f32::from(non_intra[k]) * scale);
            }
        }
        out
    }
}

/// Picture coding extension fields the encoder always uses.
fn coding_extension(f_code: [[u8; 2]; 2]) -> PictureCodingExtension {
    PictureCodingExtension {
        f_code,
        intra_dc_precision: 0,
        picture_structure: STRUCTURE_FRAME,
        top_field_first: false,
        frame_pred_frame_dct: true,
        concealment_motion_vectors: false,
        q_scale_type: false,
        intra_vlc_format: true,
        alternate_scan: false,
        repeat_first_field: false,
        chroma_420_type: true,
        progressive_frame: true,
    }
}

/// The parameters a decoder reads from [`coding_extension`], for rebuilding
/// the reference from the written bits.
pub(crate) fn decode_params(
    kind: u8,
    f_code: [[u8; 2]; 2],
    vertical_size: usize,
) -> crate::slice::PictureParams {
    let ext = coding_extension(f_code);
    crate::slice::PictureParams {
        mpeg1: false,
        coding_type: kind,
        field: None,
        second_field: false,
        f_code,
        full_pel: [false; 2],
        intra_dc_precision: ext.intra_dc_precision,
        frame_pred_frame_dct: ext.frame_pred_frame_dct,
        concealment: ext.concealment_motion_vectors,
        q_scale_type: ext.q_scale_type,
        intra_vlc: ext.intra_vlc_format,
        alternate_scan: ext.alternate_scan,
        top_field_first: ext.top_field_first,
        vertical_size,
    }
}

/// Writes the picture header, its coding extension, and every slice.
pub(crate) fn write_picture(
    job: &Job<'_>,
    analysis: &Analysis,
    rate: &mut PictureRate,
) -> Result<Coded, CodecError> {
    let geo = &job.geo;
    let mut w = BitWriter::with_capacity(geo.frame_bytes() / 4);
    PictureHeader {
        temporal_reference: job.temporal_reference,
        picture_coding_type: job.kind,
        vbv_delay: job.vbv_delay,
        full_pel_forward: false,
        forward_f_code: 7,
        full_pel_backward: false,
        backward_f_code: 7,
    }
    .write(&mut w);
    coding_extension(analysis.f_code).write(&mut w);
    let mut mb = MacroblockWriter {
        job,
        f_code: analysis.f_code,
        w,
        q: 0,
        dc_pred: [0; 3],
        pmv: [[0; 2]; 2],
        prev_dir: None,
        prev_addr: 0,
        scratch: Box::new(Scratch {
            input: [0; 64],
            coef: [0.0; 64],
            levels: [[0; 64]; 6],
            pred: [[0; 256]; 3],
        }),
    };
    let mut slices = Vec::with_capacity(geo.mb_height);
    for row in 0..geo.mb_height {
        let position = u8::try_from(row + 1).map_err(|_| CodecError::Unsupported {
            what: "more than 175 macroblock rows",
        })?;
        mb.w.start_code(position);
        slices.push((position, (mb.w.bit_len() / 8) as usize));
        let first = row * geo.mb_width;
        let spent = mb.w.bit_len() as f64;
        let q = rate.quant(first, spent);
        mb.w.put(5, q);
        // extra_bit_slice
        mb.w.put_bit(false);
        mb.q = q;
        mb.dc_pred = [128; 3];
        mb.pmv = [[0; 2]; 2];
        mb.prev_dir = None;
        for mbx in 0..geo.mb_width {
            let addr = first + mbx;
            let q = if mbx == 0 {
                q
            } else {
                let spent = mb.w.bit_len() as f64;
                rate.quant(addr, spent)
            };
            let edge = mbx == 0 || mbx + 1 == geo.mb_width;
            match analysis.plans[addr] {
                plan if plan.mode == Mode::Intra => mb.intra(addr, q)?,
                plan => mb.inter(addr, plan, q, edge)?,
            }
        }
    }
    Ok(Coded {
        data: mb.w.finish(),
        slices,
    })
}

struct Scratch {
    input: [i32; 64],
    coef: [f32; 64],
    /// Quantised levels of the six blocks, raster order.
    levels: [[i32; 64]; 6],
    pred: [[u8; 256]; 3],
}

struct MacroblockWriter<'a> {
    job: &'a Job<'a>,
    f_code: [[u8; 2]; 2],
    w: BitWriter,
    /// `quantiser_scale_code` in force.
    q: u32,
    dc_pred: [i32; 3],
    /// `PMV[0][s]`; with frame vectors `PMV[1][s]` always equals it.
    pmv: [[i32; 2]; 2],
    /// Directions of the previous macroblock, `None` after intra or at a
    /// slice start.
    prev_dir: Option<[bool; 2]>,
    /// Address of the last coded macroblock.
    prev_addr: usize,
    scratch: Box<Scratch>,
}

impl MacroblockWriter<'_> {
    /// Component, x, and y of block `i` inside the plane, 4:2:0 order.
    fn block_origin(mbx: usize, mby: usize, i: usize) -> (usize, usize, usize) {
        if i < 4 {
            (0, mbx * 16 + (i & 1) * 8, mby * 16 + (i >> 1) * 8)
        } else {
            (i - 3, mbx * 8, mby * 8)
        }
    }

    fn intra(&mut self, addr: usize, q: u32) -> Result<(), CodecError> {
        let geo = &self.job.geo;
        let (mbx, mby) = (addr % geo.mb_width, addr / geo.mb_width);
        let inv = &self.job.quant.intra[q as usize];
        let s = &mut *self.scratch;
        for i in 0..6 {
            let (c, x0, y0) = Self::block_origin(mbx, mby, i);
            let (stride, _) = geo.plane(c);
            let plane = &self.job.src.planes[c];
            for y in 0..8 {
                let row = &plane[(y0 + y) * stride + x0..][..8];
                for (dst, &v) in s.input[y * 8..y * 8 + 8].iter_mut().zip(row) {
                    *dst = i32::from(v);
                }
            }
            fdct(&s.input, &mut s.coef);
            let levels = &mut s.levels[i];
            // intra_dc_precision 0: the DC step is 8.
            levels[0] = ((s.coef[0] / 8.0).round() as i32).clamp(0, 255);
            for k in 1..64 {
                let scaled = s.coef[k].abs() * inv[k];
                let level = ((scaled + 0.375) as i32).min(2047);
                levels[k] = if s.coef[k] < 0.0 { -level } else { level };
            }
        }
        let flags = MB_INTRA | if q != self.q { MB_QUANT } else { 0 };
        self.header(addr, flags, q)?;
        for i in 0..6usize {
            let c = i.saturating_sub(3);
            let levels = &self.scratch.levels[i];
            let diff = levels[0] - self.dc_pred[c];
            self.dc_pred[c] = levels[0];
            write_dc(&mut self.w, diff, c > 0);
            write_coefs(&mut self.w, levels, 1, &COEF_ENC_ONE, EOB_ONE, false);
        }
        self.pmv = [[0; 2]; 2];
        self.prev_dir = None;
        Ok(())
    }

    fn inter(&mut self, addr: usize, plan: Plan, q: u32, edge: bool) -> Result<(), CodecError> {
        let job = self.job;
        let geo = &job.geo;
        let (mbx, mby) = (addr % geo.mb_width, addr / geo.mb_width);
        let dir = plan.mode.dir();
        let s = &mut *self.scratch;
        // Prediction, as a decoder forms it (7.6.4, 7.6.7).
        for (c, pred) in s.pred.iter_mut().enumerate() {
            let (bw, bh) = geo.mb_size(c);
            let mut first = true;
            for ((reference, used), mv) in job.refs.iter().zip(dir).zip(plan.mv) {
                let Some(reference) = reference.filter(|_| used) else {
                    continue;
                };
                let mv = if c == 0 { mv } else { [mv[0] / 2, mv[1] / 2] };
                let mut dst = Target {
                    buf: &mut pred[..],
                    offset: 0,
                    stride: bw,
                };
                predict(
                    &mut dst,
                    View::frame(reference, geo, c),
                    [(mbx * bw) as i32, (mby * bh) as i32],
                    mv,
                    [bw, bh],
                    !first,
                );
                first = false;
            }
            if first {
                return Err(CodecError::Unsupported {
                    what: "inter macroblock without a reference",
                });
            }
        }
        let inv = &job.quant.non_intra[q as usize];
        let mut cbp = 0u32;
        for i in 0..6 {
            let (c, x0, y0) = Self::block_origin(mbx, mby, i);
            let (stride, _) = geo.plane(c);
            let (bw, _) = geo.mb_size(c);
            let (px, py) = if c == 0 {
                ((i & 1) * 8, (i >> 1) * 8)
            } else {
                (0, 0)
            };
            let plane = &job.src.planes[c];
            for y in 0..8 {
                let row = &plane[(y0 + y) * stride + x0..][..8];
                let pred = &s.pred[c][(py + y) * bw + px..][..8];
                for ((dst, &v), &p) in s.input[y * 8..y * 8 + 8].iter_mut().zip(row).zip(pred) {
                    *dst = i32::from(v) - i32::from(p);
                }
            }
            fdct(&s.input, &mut s.coef);
            let levels = &mut s.levels[i];
            let mut any = false;
            for k in 0..64 {
                let level = ((s.coef[k].abs() * inv[k]) as i32).min(2047);
                any |= level != 0;
                levels[k] = if s.coef[k] < 0.0 { -level } else { level };
            }
            if any {
                cbp |= 1 << (5 - i);
            }
        }
        let p_picture = job.kind == PICTURE_P;
        if cbp == 0 && !edge {
            let skip = if p_picture {
                plan.mv[0] == [0, 0]
            } else {
                self.prev_dir == Some(dir) && (0..2).all(|d| !dir[d] || plan.mv[d] == self.pmv[d])
            };
            if skip {
                if p_picture {
                    self.pmv = [[0; 2]; 2];
                }
                self.dc_pred = [128; 3];
                return Ok(());
            }
        }
        // P "No MC": a zero vector sent as a pattern only, which resets PMV.
        let no_mc = p_picture && cbp != 0 && plan.mv[0] == [0, 0];
        let mut flags = 0;
        if dir[0] && !no_mc {
            flags |= MB_FORWARD;
        }
        if dir[1] {
            flags |= MB_BACKWARD;
        }
        if cbp != 0 {
            flags |= MB_PATTERN;
            if q != self.q {
                flags |= MB_QUANT;
            }
        }
        self.header(addr, flags, q)?;
        for d in 0..2 {
            if flags & [MB_FORWARD, MB_BACKWARD][d] != 0 {
                for t in 0..2 {
                    write_vector(
                        &mut self.w,
                        plan.mv[d][t] - self.pmv[d][t],
                        self.f_code[d][t],
                    );
                }
                self.pmv[d] = plan.mv[d];
            }
        }
        if cbp != 0 {
            let (code, len) = CBP_ENC[cbp as usize];
            self.w.put(u32::from(len), code);
            for i in 0..6 {
                if cbp & (1 << (5 - i)) != 0 {
                    write_coefs(
                        &mut self.w,
                        &self.scratch.levels[i],
                        0,
                        &COEF_ENC_ZERO,
                        EOB_ZERO,
                        true,
                    );
                }
            }
        }
        if no_mc {
            self.pmv = [[0; 2]; 2];
        }
        self.dc_pred = [128; 3];
        self.prev_dir = Some(dir);
        Ok(())
    }

    /// Address increment, macroblock type, and quantiser code.
    fn header(&mut self, addr: usize, flags: i16, q: u32) -> Result<(), CodecError> {
        let geo = &self.job.geo;
        let slice_start = addr.is_multiple_of(geo.mb_width);
        let mut increment = if slice_start {
            1
        } else {
            addr - self.prev_addr
        };
        while increment > 33 {
            let (code, len) = MB_ADDR_ENC[0];
            self.w.put(u32::from(len), code);
            increment -= 33;
        }
        let (code, len) = MB_ADDR_ENC[increment];
        self.w.put(u32::from(len), code);
        let table = match self.job.kind {
            PICTURE_P => &MB_TYPE_P_ENC,
            PICTURE_B => &MB_TYPE_B_ENC,
            _ => &MB_TYPE_I_ENC,
        };
        let (code, len) = table[flags as usize];
        if len == 0 {
            return Err(CodecError::Unsupported {
                what: "macroblock type for this picture type",
            });
        }
        self.w.put(u32::from(len), code);
        if flags & MB_QUANT != 0 {
            self.w.put(5, q);
            self.q = q;
        }
        self.prev_addr = addr;
        Ok(())
    }
}

/// `dct_dc_size` and `dct_dc_differential` (7.2.1).
fn write_dc(w: &mut BitWriter, diff: i32, chroma: bool) {
    let size = 32 - diff.unsigned_abs().leading_zeros();
    let (code, len) = if chroma {
        DC_CHROMA_ENC[size as usize]
    } else {
        DC_LUMA_ENC[size as usize]
    };
    w.put(u32::from(len), code);
    if size > 0 {
        let value = if diff > 0 {
            diff
        } else {
            diff + (1 << size) - 1
        };
        w.put(size, value as u32);
    }
}

/// Run and level pairs from scan index `start`, then End of Block. A
/// non-intra block codes a first `(0, ±1)` as `1s` (Table B.14 note).
fn write_coefs(
    w: &mut BitWriter,
    levels: &[i32; 64],
    start: usize,
    table: &CoefEncode,
    eob: (u32, u32),
    first_short: bool,
) {
    let mut run = 0u32;
    let mut first = first_short;
    for &pos in &SCAN[0][start..] {
        let level = levels[usize::from(pos)];
        if level == 0 {
            run += 1;
            continue;
        }
        let abs = level.unsigned_abs();
        let sign = u32::from(level < 0);
        if first && run == 0 && abs == 1 {
            w.put(2, 0b10 | sign);
        } else {
            let (code, len) = if run < 32 && abs <= 40 {
                table[run as usize][abs as usize]
            } else {
                (0, 0)
            };
            if len > 0 {
                w.put(u32::from(len) + 1, (u32::from(code) << 1) | sign);
            } else {
                w.put(ESCAPE.1, ESCAPE.0);
                w.put(6, run);
                w.put(12, level as u32);
            }
        }
        first = false;
        run = 0;
    }
    w.put(eob.1, eob.0);
}

/// `motion_code` and `motion_residual` for one vector difference (7.6.3.1).
fn write_vector(w: &mut BitWriter, delta: i32, f_code: u8) {
    let r_size = u32::from(f_code) - 1;
    let f = 1i32 << r_size;
    let (low, high) = (-16 * f, 16 * f - 1);
    let mut delta = delta;
    if delta < low {
        delta += 32 * f;
    } else if delta > high {
        delta -= 32 * f;
    }
    if delta == 0 {
        let (code, len) = MOTION_ENC[16];
        w.put(u32::from(len), code);
        return;
    }
    let abs = delta.abs();
    let motion_code = ((abs - 1) >> r_size) + 1;
    let residual = (abs - 1) & (f - 1);
    let index = if delta < 0 {
        16 - motion_code
    } else {
        16 + motion_code
    };
    let (code, len) = MOTION_ENC[index as usize];
    w.put(u32::from(len), code);
    w.put(r_size, residual as u32);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitReader;
    use crate::headers::CHROMA_420;
    use crate::tables::{MOTION, MOTION_BITS};

    #[test]
    fn vectors_decode_back_after_wrapping() {
        for f_code in 1..=4u8 {
            let f = 1i32 << (f_code - 1);
            for delta in -32 * f..32 * f {
                let mut w = BitWriter::new();
                write_vector(&mut w, delta, f_code);
                let bytes = w.finish();
                let mut r = BitReader::new(&bytes);
                let entry = MOTION[r.peek(MOTION_BITS) as usize];
                r.skip(u32::from(entry.len));
                let code = i32::from(entry.value);
                let r_size = u32::from(f_code) - 1;
                let decoded = if code == 0 || r_size == 0 {
                    code
                } else {
                    let residual = r.read(r_size) as i32;
                    let magnitude = ((code.abs() - 1) << r_size) + residual + 1;
                    magnitude * code.signum()
                };
                let wrapped = (decoded - delta).rem_euclid(32 * f);
                assert_eq!(
                    wrapped, 0,
                    "f_code {f_code} delta {delta} decoded {decoded}"
                );
            }
        }
    }

    #[test]
    fn dc_differences_use_the_size_and_offset() {
        let mut w = BitWriter::new();
        write_dc(&mut w, -3, false);
        // Size 2 is `01`, then -3 + 3 = 0 in two bits.
        assert_eq!(w.finish(), vec![0b0100_0000]);
        let mut w = BitWriter::new();
        write_dc(&mut w, 0, true);
        // Chroma size 0 is `00`.
        assert_eq!(w.bit_len(), 2);
    }

    #[test]
    fn first_non_intra_coefficient_uses_the_short_code() {
        let mut levels = [0i32; 64];
        levels[0] = -1;
        let mut w = BitWriter::new();
        write_coefs(&mut w, &levels, 0, &COEF_ENC_ZERO, EOB_ZERO, true);
        // `1 1` (first coefficient, negative), then End of Block `10`.
        assert_eq!(w.finish(), vec![0b1110_0000]);
    }

    #[test]
    fn large_levels_escape() {
        let mut levels = [0i32; 64];
        levels[1] = -300;
        let mut w = BitWriter::new();
        write_coefs(&mut w, &levels, 1, &COEF_ENC_ONE, EOB_ONE, false);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read(6), 1);
        assert_eq!(r.read(6), 0);
        assert_eq!(r.read(12), (-300i32 as u32) & 0xfff);
    }

    #[test]
    fn intra_cost_is_the_deviation_from_the_mean() {
        let geo = Geometry {
            mb_width: 2,
            mb_height: 1,
            chroma_format: CHROMA_420,
        };
        let mut frame = Frame::new(&geo, 77);
        assert_eq!(intra_cost(&geo, &frame, 1, 0), 0);
        // Half the macroblock at 77, half at 87: mean 82, every sample 5 off.
        for row in 8..16 {
            frame.planes[0][row * 32 + 16..row * 32 + 32].fill(87);
        }
        assert_eq!(intra_cost(&geo, &frame, 1, 0), 256 * 5);
    }

    #[test]
    fn i_pictures_plan_intra_everywhere() {
        let geo = Geometry {
            mb_width: 3,
            mb_height: 2,
            chroma_format: CHROMA_420,
        };
        let frame = Frame::new(&geo, 90);
        let analysis = analyse(&geo, PICTURE_I, &frame, [None, None], 4);
        assert!(analysis.plans.iter().all(|plan| plan.mode == Mode::Intra));
        assert_eq!(analysis.f_code, [[15; 2]; 2]);
        let analysis = analyse(&geo, PICTURE_P, &frame, [Some(&frame), None], 4);
        assert!(analysis.plans.iter().all(|plan| *plan
            == Plan {
                mode: Mode::Forward,
                mv: [[0; 2]; 2]
            }));
        assert_eq!(analysis.f_code, [[1, 1], [15, 15]]);
    }
}

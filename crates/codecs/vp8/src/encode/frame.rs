//! Choosing each macroblock's modes, coding its residual, and
//! reconstructing it exactly as the decoder will.

use crate::bool_coder::cost;
use crate::decode::{Frame, chroma_mvs};
use crate::inter::{self, Filter};
use crate::modes::{
    Bounds, GOLDEN, INTRA, LAST, MbInfo, Mv, NEARESTMV, NEARMV, NEWMV, SubInfo, ZEROMV, find_near,
    implied_b_mode,
};
use crate::predict::{self, B_PRED, CHROMA_BUF, CHROMA_STRIDE, LUMA_BUF, LUMA_STRIDE};
use crate::tokens::{
    Context, Dequant, TYPE_CHROMA, TYPE_Y_AFTER_Y2, TYPE_Y_WITH_DC, TYPE_Y2, skip_mb,
};
use crate::transform::{fdct, fwht, idct_add, iwht};

use super::cost::{MAX_MV_DELTA, ModeCosts, MvCosts, inter_mode_costs};
use super::residual::{Costs, QBlock, Rounding, quantize};
use super::search::{self, Search, Window};

/// A picture to code, padded to whole macroblocks by repeating its edges.
pub struct Source {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub mbw: usize,
    pub mbh: usize,
}

impl Source {
    /// Pads planes of `width`×`height` with strides to whole macroblocks.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn padded(
        y: &[u8],
        y_stride: usize,
        u: &[u8],
        u_stride: usize,
        v: &[u8],
        v_stride: usize,
        width: usize,
        height: usize,
    ) -> Self {
        let mbw = width.div_ceil(16);
        let mbh = height.div_ceil(16);
        let pad = |src: &[u8], stride: usize, w: usize, h: usize, pw: usize, ph: usize| {
            let mut out = vec![0u8; pw * ph];
            for r in 0..ph {
                let sr = r.min(h - 1);
                let row = &src[sr * stride..sr * stride + w];
                out[r * pw..r * pw + w].copy_from_slice(row);
                let last = row[w - 1];
                out[r * pw + w..r * pw + pw].fill(last);
            }
            out
        };
        let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
        Self {
            y: pad(y, y_stride, width, height, mbw * 16, mbh * 16),
            u: pad(u, u_stride, cw, ch, mbw * 8, mbh * 8),
            v: pad(v, v_stride, cw, ch, mbw * 8, mbh * 8),
            mbw,
            mbh,
        }
    }
}

/// Settings for one frame.
pub struct Setup<'a> {
    pub key: bool,
    pub dq: Dequant,
    /// Lagrange multiplier: squared error per bit, times 16.
    pub lambda16: u64,
    pub coeff_costs: &'a Costs,
    pub mode_costs: &'a ModeCosts,
    pub mv_costs: &'a MvCosts,
    pub prob_intra: u8,
    pub prob_last: u8,
    pub prob_golden: u8,
    pub last: Option<&'a Frame>,
    pub golden: Option<&'a Frame>,
    /// 0 is the most thorough.
    pub speed: u8,
}

/// The coded frame before entropy coding.
pub struct Coded {
    pub mbs: Vec<MbInfo>,
    pub subs: Vec<SubInfo>,
    /// Per macroblock: 16 Y, 4 U, 4 V, Y2.
    pub blocks: Vec<[QBlock; 25]>,
    /// Whether any block of the macroblock has tokens.
    pub any: Vec<bool>,
    /// Unfiltered reconstruction.
    pub recon: Frame,
}

#[inline]
fn rd(sse: u32, rate: u32, lambda16: u64) -> u64 {
    u64::from(sse) * 4096 + lambda16 * u64::from(rate)
}

fn sse(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = i32::from(x) - i32::from(y);
            (d * d) as u32
        })
        .sum()
}

/// Rounding for intra and inter residual.
fn rounding(intra: bool) -> Rounding {
    if intra {
        Rounding { dc: 64, ac: 50 }
    } else {
        Rounding { dc: 56, ac: 42 }
    }
}

/// A macroblock's luma residual coded with a Y2 block.
#[derive(Clone)]
struct LumaY2 {
    q: [QBlock; 16],
    y2: QBlock,
    recon: [u8; 256],
    sse: u32,
    rate: u32,
}

fn code_luma_y2(
    src: &[u8; 256],
    pred: &[u8; 256],
    s: &Setup<'_>,
    left: &Context,
    above: &Context,
    round: Rounding,
) -> LumaY2 {
    let mut coeffs = [[0i16; 16]; 16];
    let mut dc = [0i16; 16];
    for b in 0..16 {
        let (r, c) = (b >> 2, b & 3);
        let mut res = [0i16; 16];
        for y in 0..4 {
            for x in 0..4 {
                let i = (4 * r + y) * 16 + 4 * c + x;
                res[4 * y + x] = i16::from(src[i]) - i16::from(pred[i]);
            }
        }
        coeffs[b] = fdct(&res);
        dc[b] = coeffs[b][0];
    }
    let mut deq_y2 = [0i16; 16];
    let y2 = quantize(&fwht(&dc), s.dq.y2, round, 0, &mut deq_y2);
    let dcs = if y2.end > 0 { iwht(&deq_y2) } else { [0; 16] };
    let mut out = LumaY2 {
        q: [QBlock::default(); 16],
        y2,
        recon: *pred,
        sse: 0,
        rate: 0,
    };
    let (mut l, mut a) = (*left, *above);
    let ctx = usize::from(l[8] + a[8]);
    out.rate += s.coeff_costs.block(TYPE_Y2, ctx, 0, &y2);
    for b in 0..16 {
        let (r, c) = (b >> 2, b & 3);
        let mut deq = [0i16; 16];
        let q = quantize(&coeffs[b], s.dq.y1, round, 1, &mut deq);
        deq[0] = dcs[b];
        if q.end > 1 || deq[0] != 0 {
            idct_add(&deq, &mut out.recon, 4 * r * 16 + 4 * c, 16);
        }
        let ctx = usize::from(l[r] + a[c]);
        out.rate += s.coeff_costs.block(TYPE_Y_AFTER_Y2, ctx, 1, &q);
        let has = u8::from(q.end > 1);
        l[r] = has;
        a[c] = has;
        out.q[b] = q;
    }
    out.sse = sse(&out.recon, src);
    out
}

/// A macroblock's chroma residual.
#[derive(Clone)]
struct Chroma {
    q: [QBlock; 8],
    recon: [[u8; 64]; 2],
    sse: u32,
    rate: u32,
}

fn code_chroma(
    src: &[[u8; 64]; 2],
    pred: &[[u8; 64]; 2],
    s: &Setup<'_>,
    left: &Context,
    above: &Context,
    round: Rounding,
) -> Chroma {
    let mut out = Chroma {
        q: [QBlock::default(); 8],
        recon: *pred,
        sse: 0,
        rate: 0,
    };
    let (mut l, mut a) = (*left, *above);
    for p in 0..2 {
        for k in 0..4 {
            let (r, c) = (k >> 1, k & 1);
            let mut res = [0i16; 16];
            for y in 0..4 {
                for x in 0..4 {
                    let i = (4 * r + y) * 8 + 4 * c + x;
                    res[4 * y + x] = i16::from(src[p][i]) - i16::from(pred[p][i]);
                }
            }
            let mut deq = [0i16; 16];
            let q = quantize(&fdct(&res), s.dq.uv, round, 0, &mut deq);
            if q.end > 0 {
                idct_add(&deq, &mut out.recon[p], 4 * r * 8 + 4 * c, 8);
            }
            let (sl, sa) = (4 + 2 * p + r, 4 + 2 * p + c);
            let ctx = usize::from(l[sl] + a[sa]);
            out.rate += s.coeff_costs.block(TYPE_CHROMA, ctx, 0, &q);
            let has = u8::from(q.end > 0);
            l[sl] = has;
            a[sa] = has;
            out.q[4 * p + k] = q;
        }
        out.sse += sse(&out.recon[p], &src[p]);
    }
    out
}

/// B_PRED luma: sub-block modes and residual.
#[derive(Clone)]
struct BPred {
    q: [QBlock; 16],
    modes: [u8; 16],
    buf: [u8; LUMA_BUF],
    sse: u32,
    rate: u32,
}

#[allow(clippy::too_many_arguments)]
fn code_bpred(
    src: &[u8; 256],
    edges: &[u8; LUMA_BUF],
    s: &Setup<'_>,
    left: &Context,
    above: &Context,
    above_modes: Option<&[u8; 16]>,
    left_modes: Option<&[u8; 16]>,
    limit: u64,
) -> Option<BPred> {
    let round = rounding(true);
    let mut out = BPred {
        q: [QBlock::default(); 16],
        modes: [0; 16],
        buf: *edges,
        sse: 0,
        rate: 0,
    };
    predict::copy_above_right(&mut out.buf);
    let (mut l, mut a) = (*left, *above);
    let mut total = 0u64;
    for b in 0..16 {
        let (r, c) = (b >> 2, b & 3);
        let mode_cost = |m: usize| -> u32 {
            if s.key {
                let am = if r > 0 {
                    out.modes[b - 4]
                } else {
                    above_modes.map_or(0, |m| m[b + 12])
                };
                let lm = if c > 0 {
                    out.modes[b - 1]
                } else {
                    left_modes.map_or(0, |m| m[b + 3])
                };
                s.mode_costs.kf_b[usize::from(am)][usize::from(lm)][m]
            } else {
                s.mode_costs.b[m]
            }
        };
        let mut src4 = [0u8; 16];
        for y in 0..4 {
            for x in 0..4 {
                src4[4 * y + x] = src[(4 * r + y) * 16 + 4 * c + x];
            }
        }
        // Rank the ten predictions by error plus mode cost, then code the best few.
        let mut ranked: Vec<(u64, u8, [u8; 16])> = Vec::with_capacity(10);
        for m in 0..10u8 {
            let mut trial = out.buf;
            predict::predict_sub(&mut trial, r, c, m);
            let mut p = [0u8; 16];
            for y in 0..4 {
                let at = (4 * r + y + 1) * LUMA_STRIDE + 4 * c + 1;
                p[4 * y..4 * y + 4].copy_from_slice(&trial[at..at + 4]);
            }
            let score = rd(sse(&p, &src4), mode_cost(usize::from(m)), s.lambda16);
            ranked.push((score, m, p));
        }
        ranked.sort_by_key(|e| e.0);
        let keep = if s.speed == 0 { 4 } else { 2 };
        let ctx = usize::from(l[r] + a[c]);
        let mut best: Option<(u64, u8, QBlock, [u8; 16], u32, u32)> = None;
        for &(_, m, p) in ranked.iter().take(keep) {
            let mut res = [0i16; 16];
            for i in 0..16 {
                res[i] = i16::from(src4[i]) - i16::from(p[i]);
            }
            let mut deq = [0i16; 16];
            let q = quantize(&fdct(&res), s.dq.y1, round, 0, &mut deq);
            let mut rec = p;
            if q.end > 0 {
                idct_add(&deq, &mut rec, 0, 4);
            }
            let e = sse(&rec, &src4);
            let bits = s.coeff_costs.block(TYPE_Y_WITH_DC, ctx, 0, &q) + mode_cost(usize::from(m));
            let j = rd(e, bits, s.lambda16);
            if best.as_ref().is_none_or(|b| j < b.0) {
                best = Some((j, m, q, rec, e, bits));
            }
        }
        let (j, m, q, rec, e, bits) = best?;
        total += j;
        if total > limit {
            return None;
        }
        for y in 0..4 {
            let at = (4 * r + y + 1) * LUMA_STRIDE + 4 * c + 1;
            out.buf[at..at + 4].copy_from_slice(&rec[4 * y..4 * y + 4]);
        }
        out.modes[b] = m;
        out.q[b] = q;
        out.sse += e;
        out.rate += bits;
        let has = u8::from(q.end > 0);
        l[r] = has;
        a[c] = has;
    }
    Some(out)
}

/// The decision for one macroblock.
struct Decision {
    mb: MbInfo,
    sub: SubInfo,
    luma: LumaKind,
    chroma: Chroma,
    j: u64,
}

enum LumaKind {
    Y2(Box<LumaY2>),
    B(Box<BPred>),
}

impl Decision {
    fn has_y2(&self) -> bool {
        matches!(self.luma, LumaKind::Y2(_))
    }
}

fn intra_chroma(
    recon: &Frame,
    mbx: usize,
    mby: usize,
    src: &[[u8; 64]; 2],
    s: &Setup<'_>,
    left: &Context,
    above: &Context,
) -> (u8, Chroma, u64) {
    let cs = recon.mbw * 8;
    let mut bufs = [[0u8; CHROMA_BUF]; 2];
    predict::load_edges(&mut bufs[0], &recon.u, cs, mbx, mby, 8, 0);
    predict::load_edges(&mut bufs[1], &recon.v, cs, mbx, mby, 8, 0);
    let mut best: Option<(u8, Chroma, u64)> = None;
    for mode in 0..4u8 {
        let mut pred = [[0u8; 64]; 2];
        for p in 0..2 {
            let mut buf = bufs[p];
            predict::predict_block(&mut buf, 8, CHROMA_STRIDE, mode, mby > 0, mbx > 0);
            for r in 0..8 {
                let at = (r + 1) * CHROMA_STRIDE + 1;
                pred[p][r * 8..r * 8 + 8].copy_from_slice(&buf[at..at + 8]);
            }
        }
        let c = code_chroma(src, &pred, s, left, above, rounding(true));
        let j = rd(
            c.sse,
            c.rate + s.mode_costs.uv[usize::from(mode)],
            s.lambda16,
        );
        if best.as_ref().is_none_or(|b| j < b.2) {
            best = Some((mode, c, j));
        }
    }
    best.unwrap_or_else(|| {
        (
            0,
            code_chroma(src, &[[128; 64]; 2], s, left, above, rounding(true)),
            u64::MAX,
        )
    })
}

/// Everything a macroblock decision looks at.
struct MbCtx<'a> {
    mbx: usize,
    mby: usize,
    src_y: [u8; 256],
    src_c: [[u8; 64]; 2],
    left: Context,
    above: Context,
    above_mb: Option<&'a MbInfo>,
    left_mb: Option<&'a MbInfo>,
    above_left_mb: Option<&'a MbInfo>,
    above_sub: Option<&'a SubInfo>,
    left_sub: Option<&'a SubInfo>,
}

fn decide_intra(recon: &Frame, m: &MbCtx<'_>, s: &Setup<'_>, limit: u64) -> Option<Decision> {
    let ys = recon.mbw * 16;
    let mut edges = [0u8; LUMA_BUF];
    predict::load_edges(&mut edges, &recon.y, ys, m.mbx, m.mby, 16, 4);
    let intra_flag = if s.key { 0 } else { cost(s.prob_intra, false) };
    let round = rounding(true);
    // 16×16 modes: rank by prediction error, code the best.
    let mut ranked: Vec<(u64, u8, [u8; 256])> = Vec::with_capacity(4);
    for mode in 0..4u8 {
        let mut buf = edges;
        predict::predict_block(&mut buf, 16, LUMA_STRIDE, mode, m.mby > 0, m.mbx > 0);
        let mut pred = [0u8; 256];
        for r in 0..16 {
            let at = (r + 1) * LUMA_STRIDE + 1;
            pred[r * 16..r * 16 + 16].copy_from_slice(&buf[at..at + 16]);
        }
        let score = rd(
            sse(&pred, &m.src_y),
            s.mode_costs.y[usize::from(mode)],
            s.lambda16,
        );
        ranked.push((score, mode, pred));
    }
    ranked.sort_by_key(|e| e.0);
    let keep = if s.speed == 0 { 2 } else { 1 };
    let mut best: Option<(u64, u8, LumaKind)> = None;
    for &(_, mode, ref pred) in ranked.iter().take(keep) {
        let l = code_luma_y2(&m.src_y, pred, s, &m.left, &m.above, round);
        let j = rd(
            l.sse,
            l.rate + s.mode_costs.y[usize::from(mode)] + intra_flag,
            s.lambda16,
        );
        if best.as_ref().is_none_or(|b| j < b.0) {
            best = Some((j, mode, LumaKind::Y2(Box::new(l))));
        }
    }
    let try_b = s.key || s.speed <= 1 || best.as_ref().is_some_and(|b| b.0 < limit);
    if try_b {
        let bound = best.as_ref().map_or(u64::MAX, |b| b.0);
        let b_cost = s.mode_costs.y[usize::from(B_PRED)] + intra_flag;
        let base = rd(0, b_cost, s.lambda16);
        if base < bound {
            let am = m.above_sub.map(|x| &x.b_modes);
            let lm = m.left_sub.map(|x| &x.b_modes);
            if let Some(bp) =
                code_bpred(&m.src_y, &edges, s, &m.left, &m.above, am, lm, bound - base)
            {
                let j = rd(bp.sse, bp.rate + b_cost, s.lambda16);
                if best.as_ref().is_none_or(|b| j < b.0) {
                    best = Some((j, B_PRED, LumaKind::B(Box::new(bp))));
                }
            }
        }
    }
    let (j, y_mode, luma) = best?;
    let (uv_mode, chroma, cj) = intra_chroma(recon, m.mbx, m.mby, &m.src_c, s, &m.left, &m.above);
    let mut sub = SubInfo::default();
    match &luma {
        LumaKind::B(bp) => sub.b_modes = bp.modes,
        LumaKind::Y2(_) => sub.b_modes = [implied_b_mode(y_mode); 16],
    }
    let mb = MbInfo {
        y_mode,
        uv_mode,
        ref_frame: INTRA,
        ..MbInfo::default()
    };
    Some(Decision {
        mb,
        sub,
        luma,
        chroma,
        j: j.saturating_add(cj),
    })
}

/// One inter candidate before its residual is coded.
struct Candidate {
    ref_frame: u8,
    mode: u8,
    mv: Mv,
    rate: u32,
    score: u64,
    pred: [u8; 256],
}

fn inter_candidates(m: &MbCtx<'_>, s: &Setup<'_>, out: &mut Vec<Candidate>) {
    let mbw = s.last.map_or(1, |f| f.mbw);
    let mbh = s.last.map_or(1, |f| f.mbh);
    let bounds = Bounds::new(m.mbx, m.mby, mbw, mbh);
    let sign_bias = [false; 4];
    let inter_flag = cost(s.prob_intra, true);
    let refs: [(u8, Option<&Frame>); 2] = [(LAST, s.last), (GOLDEN, s.golden)];
    for (ref_frame, frame) in refs {
        let Some(frame) = frame else { continue };
        let near = find_near(
            m.above_mb,
            m.left_mb,
            m.above_left_mb,
            &sign_bias,
            ref_frame,
        );
        let modes = inter_mode_costs(&near.probs);
        let ref_cost = inter_flag
            + if ref_frame == LAST {
                cost(s.prob_last, false)
            } else {
                cost(s.prob_last, true) + cost(s.prob_golden, false)
            };
        let best = near.best.clamp(&bounds);
        let nearest = near.nearest.clamp(&bounds);
        let nearv = near.near.clamp(&bounds);
        let plane = frame.luma();
        let push = |mode: u8, mv: Mv, rate: u32, out: &mut Vec<Candidate>| {
            if out
                .iter()
                .any(|c| c.ref_frame == ref_frame && c.mv == mv && c.rate <= rate)
            {
                return;
            }
            let mut pred = [0u8; 256];
            search::predict16(plane, m.mbx, m.mby, mv, Filter::SixTap, &mut pred);
            let score = rd(search::sse16(&pred, &m.src_y), rate, s.lambda16);
            out.push(Candidate {
                ref_frame,
                mode,
                mv,
                rate,
                score,
                pred,
            });
        };
        push(
            ZEROMV,
            Mv::ZERO,
            ref_cost + modes[usize::from(ZEROMV - NEARESTMV)],
            out,
        );
        push(NEARESTMV, nearest, ref_cost + modes[0], out);
        push(NEARMV, nearv, ref_cost + modes[1], out);
        // NEWMV: search around the predictors.
        let new_base = ref_cost + modes[usize::from(NEWMV - NEARESTMV)];
        let lim = |v: i32, lo: i32, hi: i32| v.clamp(lo, hi);
        let window = Window {
            min: Mv::new(
                lim(best.row - MAX_MV_DELTA, bounds.top, bounds.bottom),
                lim(best.col - MAX_MV_DELTA, bounds.left, bounds.right),
            ),
            max: Mv::new(
                lim(best.row + MAX_MV_DELTA, bounds.top, bounds.bottom),
                lim(best.col + MAX_MV_DELTA, bounds.left, bounds.right),
            ),
        };
        // SAD per bit is about the square root of the squared-error lambda.
        let sad_per_bit = ((s.lambda16 as f64 / 16.0).sqrt()).max(1.0);
        let mv_costs = s.mv_costs;
        let penalty = move |mv: Mv| -> u32 {
            let bits = mv_costs.cost(mv.row - best.row, mv.col - best.col);
            (sad_per_bit * f64::from(bits) / 256.0) as u32
        };
        let golden_extra = ref_frame == GOLDEN;
        let search = Search {
            plane,
            src: &m.src_y,
            mbx: m.mbx,
            mby: m.mby,
            window,
            penalty: &penalty,
            filter: Filter::SixTap,
            range: if golden_extra {
                2
            } else if s.speed == 0 {
                16
            } else {
                8
            },
            subpel: true,
        };
        let (mv, _) = search::search(&search, &[best, nearest, nearv, Mv::ZERO]);
        let dr = mv.row - best.row;
        let dc = mv.col - best.col;
        if dr.abs() <= MAX_MV_DELTA && dc.abs() <= MAX_MV_DELTA {
            push(NEWMV, mv, new_base + mv_costs.cost(dr, dc), out);
        }
    }
}

fn inter_chroma_pred(frame: &Frame, mbx: usize, mby: usize, mv: Mv) -> [[u8; 64]; 2] {
    let mut out = [[0u8; 64]; 2];
    for (p, plane) in out.iter_mut().enumerate() {
        inter::predict(
            frame.chroma(p == 1),
            (mbx * 8) as i32 + (mv.col >> 3),
            (mby * 8) as i32 + (mv.row >> 3),
            (mv.col & 7) as usize,
            (mv.row & 7) as usize,
            8,
            8,
            Filter::SixTap,
            plane,
            8,
        );
    }
    out
}

fn decide_inter(m: &MbCtx<'_>, s: &Setup<'_>) -> Option<Decision> {
    let mut cands = Vec::with_capacity(8);
    inter_candidates(m, s, &mut cands);
    cands.sort_by_key(|c| c.score);
    let keep = if s.speed == 0 { 3 } else { 2 };
    let round = rounding(false);
    let mut best: Option<Decision> = None;
    for cand in cands.iter().take(keep) {
        let luma = code_luma_y2(&m.src_y, &cand.pred, s, &m.left, &m.above, round);
        let frame = if cand.ref_frame == GOLDEN {
            s.golden
        } else {
            s.last
        };
        let Some(frame) = frame else { continue };
        let mb = MbInfo {
            y_mode: cand.mode,
            uv_mode: 0,
            ref_frame: cand.ref_frame,
            mv: cand.mv,
            ..MbInfo::default()
        };
        let sub = SubInfo {
            b_modes: [0; 16],
            mvs: [cand.mv; 16],
        };
        let cmv = chroma_mvs(&mb, &sub, false)[0];
        let cpred = inter_chroma_pred(frame, m.mbx, m.mby, cmv);
        let chroma = code_chroma(&m.src_c, &cpred, s, &m.left, &m.above, round);
        let j = rd(
            luma.sse + chroma.sse,
            luma.rate + chroma.rate + cand.rate,
            s.lambda16,
        );
        if best.as_ref().is_none_or(|b| j < b.j) {
            best = Some(Decision {
                mb,
                sub,
                luma: LumaKind::Y2(Box::new(luma)),
                chroma,
                j,
            });
        }
    }
    best
}

/// Codes a whole frame's macroblocks.
pub fn code_frame(src: &Source, s: &Setup<'_>) -> Coded {
    let (mbw, mbh) = (src.mbw, src.mbh);
    let count = mbw * mbh;
    let mut coded = Coded {
        mbs: vec![MbInfo::default(); count],
        subs: vec![SubInfo::default(); count],
        blocks: vec![[QBlock::default(); 25]; count],
        any: vec![false; count],
        recon: Frame::new(mbw, mbh),
    };
    let ys = mbw * 16;
    let cs = mbw * 8;
    let mut above_ctx: Vec<Context> = vec![[0; 9]; mbw];
    for mby in 0..mbh {
        let mut left_ctx: Context = [0; 9];
        for (mbx, above) in above_ctx.iter_mut().enumerate() {
            let i = mby * mbw + mbx;
            let mut src_y = [0u8; 256];
            for r in 0..16 {
                let at = (mby * 16 + r) * ys + mbx * 16;
                src_y[r * 16..r * 16 + 16].copy_from_slice(&src.y[at..at + 16]);
            }
            let mut src_c = [[0u8; 64]; 2];
            for (p, plane) in [&src.u, &src.v].iter().enumerate() {
                for r in 0..8 {
                    let at = (mby * 8 + r) * cs + mbx * 8;
                    src_c[p][r * 8..r * 8 + 8].copy_from_slice(&plane[at..at + 8]);
                }
            }
            let decision = {
                let m = MbCtx {
                    mbx,
                    mby,
                    src_y,
                    src_c,
                    left: left_ctx,
                    above: *above,
                    above_mb: (mby > 0).then(|| &coded.mbs[i - mbw]),
                    left_mb: (mbx > 0).then(|| &coded.mbs[i - 1]),
                    above_left_mb: (mby > 0 && mbx > 0).then(|| &coded.mbs[i - mbw - 1]),
                    above_sub: (mby > 0).then(|| &coded.subs[i - mbw]),
                    left_sub: (mbx > 0).then(|| &coded.subs[i - 1]),
                };
                let inter = if s.key { None } else { decide_inter(&m, s) };
                let limit = inter.as_ref().map_or(u64::MAX, |d| d.j);
                let intra = decide_intra(&coded.recon, &m, s, limit);
                match (inter, intra) {
                    (Some(a), Some(b)) => {
                        if b.j < a.j {
                            b
                        } else {
                            a
                        }
                    }
                    (Some(a), None) => a,
                    (None, Some(b)) => b,
                    (None, None) => continue,
                }
            };
            commit(&mut coded, i, mbx, mby, decision, &mut left_ctx, above, s);
        }
    }
    coded
}

#[allow(clippy::too_many_arguments)]
fn commit(
    coded: &mut Coded,
    i: usize,
    mbx: usize,
    mby: usize,
    d: Decision,
    left: &mut Context,
    above: &mut Context,
    s: &Setup<'_>,
) {
    let has_y2 = d.has_y2();
    let mut blocks = [QBlock::default(); 25];
    let ys = coded.recon.mbw * 16;
    let cs = coded.recon.mbw * 8;
    match &d.luma {
        LumaKind::Y2(l) => {
            blocks[..16].copy_from_slice(&l.q);
            blocks[24] = l.y2;
            for r in 0..16 {
                let at = (mby * 16 + r) * ys + mbx * 16;
                coded.recon.y[at..at + 16].copy_from_slice(&l.recon[r * 16..r * 16 + 16]);
            }
        }
        LumaKind::B(bp) => {
            blocks[..16].copy_from_slice(&bp.q);
            predict::store(&bp.buf, &mut coded.recon.y, ys, mbx, mby, 16, LUMA_STRIDE);
        }
    }
    blocks[16..24].copy_from_slice(&d.chroma.q);
    for (p, plane) in [&mut coded.recon.u, &mut coded.recon.v]
        .into_iter()
        .enumerate()
    {
        for r in 0..8 {
            let at = (mby * 8 + r) * cs + mbx * 8;
            plane[at..at + 8].copy_from_slice(&d.chroma.recon[p][r * 8..r * 8 + 8]);
        }
    }
    let first = |b: usize| -> u8 { u8::from(b < 16 && has_y2) };
    let any = blocks
        .iter()
        .enumerate()
        .any(|(b, q)| (b < 24 || has_y2) && q.end > first(b));
    let mut mb = d.mb;
    mb.skip = !any;
    let _ = s;
    if any {
        if has_y2 {
            let has = u8::from(blocks[24].end > 0);
            left[8] = has;
            above[8] = has;
        }
        for b in 0..16 {
            let has = u8::from(blocks[b].end > first(b));
            left[b >> 2] = has;
            above[b & 3] = has;
        }
        for k in 0..8 {
            let (p, r, c) = (k >> 2, (k & 3) >> 1, k & 1);
            let has = u8::from(blocks[16 + k].end > 0);
            left[4 + 2 * p + r] = has;
            above[4 + 2 * p + c] = has;
        }
    } else {
        skip_mb(left, above, has_y2);
    }
    coded.mbs[i] = mb;
    coded.subs[i] = d.sub;
    coded.blocks[i] = blocks;
    coded.any[i] = any;
}

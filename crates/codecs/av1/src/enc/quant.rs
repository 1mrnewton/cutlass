//! Quantization at the moment a transform block is written.
//!
//! The tile code calls [`TileDec::enc_quantize`] from `coeffs()` when it runs
//! with a writing coder. The prediction for the transform block is already in
//! the current frame, so the residual here is exactly what the decoder will
//! add its reconstruction to.
//!
//! Coefficient decisions are rate-distortion optimized:
//! - the rate of a candidate set of levels is the exact cost of the symbols
//!   `coeffs()` will write (all_zero, luma tx type, eob, coeff_base_eob,
//!   coeff_base, coeff_br, signs and Golomb), from the current CDFs and the
//!   current neighbor contexts;
//! - RDOQ (in the spirit of libaom's `av1_optimize_txb`) walks the block from
//!   the last coefficient backward and keeps, for each coefficient, the best
//!   of `level`, `level - 1` and `0` with context-aware costs, then picks the
//!   end of block by RD cost;
//! - the transform type and the all-zero choice are decided by the true RD
//!   cost, with the distortion of each candidate measured on an actual
//!   reconstruction.

use crate::dec::cdf::CdfContext;
use crate::dec::coder::Coder;
use crate::dec::tile::{TileDec, get_scan, get_tx_class};
use crate::dec::types::{
    block_height, block_width, plane_residual_size, tx_h, tx_h_log2, tx_w, tx_w_log2,
};
use crate::dsp::Pixel;
use crate::dsp::itx::inv_txfm_add;
use crate::spec::tables::{
    ADJUSTED_TX_SIZE, COEFF_BASE_CTX_OFFSET, COEFF_BASE_POS_CTX_OFFSET,
    FILTER_INTRA_MODE_TO_INTRA_DIR, MAG_REF_OFFSET_WITH_TX_CLASS, SIG_REF_DIFF_OFFSET,
    TRANSFORM_ROW_SHIFT, TX_SIZE_SQR, TX_SIZE_SQR_UP, TX_TYPE_INTER_INV_SET1,
    TX_TYPE_INTER_INV_SET2, TX_TYPE_INTER_INV_SET3, TX_TYPE_INTRA_INV_SET1, TX_TYPE_INTRA_INV_SET2,
};

use std::cell::Cell;

use super::fwd_txfm::{FwdCols, fwd_txfm_cols, fwd_wht4x4};
use super::msac_enc::{COST_SHIFT, bool_cost, symbol_cost};

/// Encoder tuning for coefficient decisions.
#[derive(Clone, Copy, Debug)]
pub struct QuantParams {
    /// Lagrange multiplier for D + lambda * R, with D in squared sample
    /// errors and R in bits.
    pub lambda: f64,
    /// Rounding offset in 1/128 of the quantizer step for intra and inter
    /// blocks (dead-zone quantization when RDOQ is off).
    pub round_intra: i32,
    pub round_inter: i32,
    /// Luma transform types tried besides DCT_DCT: 0 none, 1 the ADST
    /// combinations and IDTX, 2 also V_DCT and H_DCT, 3 every type.
    pub tx_type_search: u8,
    /// With more transform types than this, a quick estimate picks DCT_DCT
    /// and the best others up to this many for the full search (0: all).
    pub tx_type_keep: u8,
    /// Rate-distortion optimized quantization: 0 off (dead-zone rounding),
    /// 1 one greedy pass plus end-of-block search, 2 a second greedy pass
    /// after the end of block moved.
    pub rdoq: u8,
}

impl Default for QuantParams {
    fn default() -> Self {
        Self {
            lambda: 8.0,
            round_intra: 48,
            round_inter: 36,
            tx_type_search: 1,
            tx_type_keep: 0,
            rdoq: 2,
        }
    }
}

const ADST_ADST: u8 = 3;
const ADST_DCT: u8 = 1;
const DCT_ADST: u8 = 2;
const IDTX: u8 = 9;
const V_DCT: u8 = 10;
const H_DCT: u8 = 11;

/// Largest level the bitstream carries (`coeffs()` masks with 0xFFFFF).
const MAX_LEVEL: u32 = (1 << 20) - 1;

/// One bit in cost units.
const BIT: u32 = 1 << COST_SHIFT;

/// Golomb code length of `v` in cost units.
#[inline]
fn golomb_cost(v: u32) -> u32 {
    let x = u64::from(v) + 1;
    let len = 63 - x.leading_zeros();
    (2 * len + 1) << COST_SHIFT
}

/// Contexts of one transform block that do not depend on its transform type.
struct Env {
    tx_sz_ctx: usize,
    ptype: usize,
    skip_ctx: usize,
    dc_ctx: usize,
}

/// Scan geometry and context neighborhoods for one transform type.
struct Layout {
    scan: &'static [u16],
    class: usize,
    bwl: u32,
    /// Adjusted transform area (for the coeff_base_eob context).
    n: usize,
    tx: usize,
    offs: [usize; 5],
    br_offs: [usize; 3],
}

impl Layout {
    fn new(tx: u8, tx_type: u8) -> Self {
        let adj = ADJUSTED_TX_SIZE[tx as usize];
        let bwl = tx_w_log2(adj);
        let ls = (1usize << bwl) + 4;
        let class = get_tx_class(tx_type);
        Self {
            scan: get_scan(tx, tx_type),
            class,
            bwl,
            n: tx_h(adj) << bwl,
            tx: tx as usize,
            offs: std::array::from_fn(|i| {
                SIG_REF_DIFF_OFFSET[class][i][0] as usize * ls
                    + SIG_REF_DIFF_OFFSET[class][i][1] as usize
            }),
            br_offs: std::array::from_fn(|i| {
                MAG_REF_OFFSET_WITH_TX_CLASS[class][i][0] as usize * ls
                    + MAG_REF_OFFSET_WITH_TX_CLASS[class][i][1] as usize
            }),
        }
    }

    /// coeff_base context of a non-eob position (as in `read_levels`).
    #[inline]
    fn base_ctx(&self, lev: &[u8], pos: usize, row: usize, col: usize, li: usize) -> usize {
        let mut mag = 0u32;
        for o in self.offs {
            mag += u32::from(lev[li + o]).min(3);
        }
        let m = ((mag + 1) >> 1).min(4) as usize;
        match self.class {
            0 => {
                if pos == 0 {
                    0
                } else {
                    m + COEFF_BASE_CTX_OFFSET[self.tx][row.min(4)][col.min(4)] as usize
                }
            }
            1 => m + COEFF_BASE_POS_CTX_OFFSET[col.min(2)] as usize,
            _ => m + COEFF_BASE_POS_CTX_OFFSET[row.min(2)] as usize,
        }
    }

    /// coeff_br context (as in `read_levels`).
    #[inline]
    fn br_ctx(&self, lev: &[u8], pos: usize, row: usize, col: usize, li: usize) -> usize {
        let mut mag = 0u32;
        for o in self.br_offs {
            mag += u32::from(lev[li + o]);
        }
        let mag = ((mag + 1) >> 1).min(6) as usize;
        let near = match self.class {
            0 => row < 2 && col < 2,
            1 => col == 0,
            _ => row == 0,
        };
        if pos == 0 {
            mag
        } else if near {
            mag + 7
        } else {
            mag + 14
        }
    }

    /// coeff_base_eob context of scan index `last`.
    #[inline]
    fn eob_ctx(&self, last: usize) -> usize {
        if last == 0 {
            0
        } else if last <= self.n / 8 {
            1
        } else if last <= self.n / 4 {
            2
        } else {
            3
        }
    }
}

/// The CDFs the coefficient symbols of one block use.
struct Costs<'a> {
    base: &'a [u64; 42],
    br: &'a [u64; 21],
    base_eob: &'a [[u16; 3]; 4],
    /// dc_sign cost for the sign of the DC coefficient of the candidate.
    dc_sign: u32,
    /// Symbol costs computed so far (`u16::MAX`: not yet): coeff_base,
    /// then coeff_br, then coeff_base_eob.
    memo: [Cell<u16>; MEMO_LEN],
}

const MEMO_BR: usize = 42 * 4;
const MEMO_EOB: usize = MEMO_BR + 21 * 4;
const MEMO_LEN: usize = MEMO_EOB + 4 * 3;

impl<'a> Costs<'a> {
    fn new(
        base: &'a [u64; 42],
        br: &'a [u64; 21],
        base_eob: &'a [[u16; 3]; 4],
        dc_sign: u32,
    ) -> Self {
        Self {
            base,
            br,
            base_eob,
            dc_sign,
            memo: std::array::from_fn(|_| Cell::new(u16::MAX)),
        }
    }

    /// `symbol_cost(cdf, s)`, remembered at `memo[i]`.
    #[inline(always)]
    fn cached(&self, i: usize, cdf: &[u16], s: usize) -> u32 {
        let m = &self.memo[i];
        let v = m.get();
        if v != u16::MAX {
            return u32::from(v);
        }
        let c = symbol_cost(cdf, s);
        m.set(c as u16);
        c
    }

    /// Cost of the coeff_br symbols and the Golomb remainder of level `l` (> 2).
    #[inline]
    fn tail(&self, br_ctx: usize, l: u32) -> u32 {
        let cdf = &crate::dec::cdf::unpack4(self.br[br_ctx]);
        let mut rem = l - 3;
        let mut cost = 0;
        for _ in 0..4 {
            let wb = rem.min(3);
            rem -= wb;
            cost += self.cached(MEMO_BR + br_ctx * 4 + wb as usize, cdf, wb as usize);
            if wb < 3 {
                break;
            }
        }
        if l >= 15 {
            cost += golomb_cost(l - 15);
        }
        cost
    }

    #[inline]
    fn sign(&self, c: usize) -> u32 {
        if c == 0 { self.dc_sign } else { BIT }
    }

    /// Cost of level `l` at a non-eob scan index `c`.
    #[inline]
    fn level(&self, c: usize, base_ctx: usize, br_ctx: usize, l: u32) -> u32 {
        let s = l.min(3) as usize;
        let mut cost = self.cached(
            base_ctx * 4 + s,
            &crate::dec::cdf::unpack4(self.base[base_ctx]),
            s,
        );
        if l > 0 {
            cost += self.sign(c);
        }
        if l > 2 {
            cost += self.tail(br_ctx, l);
        }
        cost
    }

    /// Cost of level `l` (>= 1) as the eob coefficient at scan index `c`.
    #[inline]
    fn last(&self, c: usize, eob_ctx: usize, br_ctx: usize, l: u32) -> u32 {
        let s = (l.min(3) - 1) as usize;
        let mut cost =
            self.cached(MEMO_EOB + eob_ctx * 3 + s, &self.base_eob[eob_ctx], s) + self.sign(c);
        if l > 2 {
            cost += self.tail(br_ctx, l);
        }
        cost
    }
}

/// Transform-domain distortion of the levels of one block.
/// Work buffers of [`TileDec::enc_quantize`], kept between calls.
pub struct QuantScratch {
    res: [i32; 64 * 64],
    best: [i32; 32 * 32],
    coefs: [i32; 32 * 32],
    levels: [i32; 32 * 32],
    lv: [u32; 32 * 32],
    rec: [Rec; 32 * 32],
    /// Dequantized levels of the candidate being reconstructed.
    dqbuf: [i32; 32 * 32],
    dist: Dist,
    /// Coefficients of each transform type the quick estimate ranked (up
    /// to 16x16, which every type with more than DCT_DCT allows).
    ranked: [i32; 15 * 16 * 16],
    /// The column passes of the current residual block.
    cols: FwdCols,
}

impl QuantScratch {
    fn new() -> Box<Self> {
        Box::new(Self {
            res: [0; 64 * 64],
            best: [0; 32 * 32],
            coefs: [0; 32 * 32],
            levels: [0; 32 * 32],
            lv: [0; 32 * 32],
            rec: [Rec::default(); 32 * 32],
            dqbuf: [0; 32 * 32],
            ranked: [0; 15 * 16 * 16],
            cols: FwdCols::default(),
            dist: Dist {
                ac: [0; 32 * 32],
                q2: [0; 32 * 32],
                denom_log2: 0,
                lim: 1,
                g2: 1.0,
            },
        })
    }
}

struct Dist {
    /// |coefficient| per position.
    ac: [i64; 32 * 32],
    /// Dequantization step per position (quantizer matrix applied).
    q2: [i64; 32 * 32],
    /// log2 of the dequantization divisor (1, 2 or 4).
    denom_log2: u32,
    lim: i64,
    /// Squared gain from the dequantized domain to samples.
    g2: f64,
}

impl Dist {
    /// Dequantized magnitude of level `l` at `pos`, as `coeffs()` computes it.
    #[inline]
    fn dq(&self, pos: usize, l: u32) -> i64 {
        (((i64::from(l) * self.q2[pos]) & 0xFF_FFFF) >> self.denom_log2).min(self.lim - 1)
    }

    /// Distortion (in squared sample errors) of level `l` at `pos`.
    #[inline]
    fn err(&self, pos: usize, l: u32) -> f64 {
        let e = (self.ac[pos] - if l == 0 { 0 } else { self.dq(pos, l) }) as f64;
        e * e * self.g2
    }
}

/// Per scan index bookkeeping of an RDOQ pass.
#[derive(Clone, Copy, Default)]
struct Rec {
    /// Rate of the chosen level as coded (non-eob form unless last).
    rate: u32,
    /// Rate of the chosen level if it were the eob coefficient.
    last_rate: u32,
    d_chosen: f64,
    d_zero: f64,
}

/// Codes (or, with `opt`, optimizes then codes) the levels `lv` (magnitudes
/// by position) for an end of block `eob`, walking the scan backward exactly
/// as `read_levels` does. Returns the rate of the coeff_base_eob, coeff_base,
/// coeff_br, sign and Golomb symbols. With `opt`, each coefficient keeps the
/// best of its level, level - 1 and 0 (the eob coefficient stays nonzero) and
/// `rec` receives what the end-of-block search needs.
fn levels_pass(
    lay: &Layout,
    costs: &Costs<'_>,
    lv: &mut [u32],
    eob: usize,
    opt: Option<(&Dist, f64)>,
    rec: &mut [Rec],
) -> u32 {
    let mut lev = [0u8; 36 * 36];
    let mask = (1usize << lay.bwl) - 1;
    let mut total = 0u32;
    for c in (0..eob).rev() {
        let pos = lay.scan[c] as usize;
        let row = pos >> lay.bwl;
        let col = pos & mask;
        let li = pos + (row << 2);
        let l = lv[pos];
        let bctx = lay.br_ctx(&lev, pos, row, col, li);
        let ectx = lay.eob_ctx(c);
        let (k, rate) = if c == eob - 1 {
            let mut k = l.max(1);
            let mut rate = costs.last(c, ectx, bctx, k);
            if let Some((d, lam)) = opt
                && k > 1
            {
                let r1 = costs.last(c, ectx, bctx, k - 1);
                if d.err(pos, k - 1) + lam * f64::from(r1) < d.err(pos, k) + lam * f64::from(rate) {
                    k -= 1;
                    rate = r1;
                }
            }
            (k, rate)
        } else {
            let ctx = lay.base_ctx(&lev, pos, row, col, li);
            let mut k = l;
            let mut rate = costs.level(c, ctx, bctx, k);
            if let Some((d, lam)) = opt
                && k > 0
            {
                let mut best = d.err(pos, k) + lam * f64::from(rate);
                // level - 1, and 0 for small levels.
                let cands = [k - 1, 0];
                let nc = if k == 2 { 2 } else { 1 };
                for &cand in &cands[..nc] {
                    let r = costs.level(c, ctx, bctx, cand);
                    let j = d.err(pos, cand) + lam * f64::from(r);
                    if j < best {
                        best = j;
                        k = cand;
                        rate = r;
                    }
                }
            }
            (k, rate)
        };
        if let Some((d, _)) = opt {
            let dz = d.err(pos, 0);
            rec[c] = Rec {
                rate,
                last_rate: if k > 0 {
                    costs.last(c, ectx, bctx, k)
                } else {
                    0
                },
                d_chosen: if k > 0 { d.err(pos, k) } else { dz },
                d_zero: dz,
            };
        }
        lv[pos] = k;
        lev[li] = k.min(15) as u8;
        total += rate;
    }
    total
}

impl<P: Pixel, C: Coder> TileDec<'_, P, C> {
    /// Contexts of all_zero and dc_sign, as computed in `coeffs()`.
    fn txb_env(&self, plane: usize, x4: usize, y4: usize, tx: u8) -> Env {
        let w4 = tx_w(tx) >> 2;
        let h4 = tx_h(tx) >> 2;
        let tx_sz_ctx =
            ((TX_SIZE_SQR[tx as usize] + TX_SIZE_SQR_UP[tx as usize] + 1) >> 1) as usize;
        let (sx, sy) = self.plane_sub(plane);
        let max_x4 = self.fs.mi_cols >> sx;
        let max_y4 = self.fs.mi_rows >> sy;
        let sb_mask = (self.sb_size4 - 1) >> sy;
        let bsize = plane_residual_size(self.b.bsize, sx != 0, sy != 0);
        let bw = block_width(bsize);
        let bh = block_height(bsize);
        let w = tx_w(tx);
        let h = tx_h(tx);
        let ly0 = y4 & sb_mask;
        let xs = (0..w4).filter(|k| x4 + k < max_x4).map(|k| x4 + k);
        let ys = || {
            (0..h4)
                .filter(move |k| y4 + k < max_y4)
                .map(move |k| (ly0 + k) & 31)
        };
        let skip_ctx = if plane == 0 {
            let top = xs
                .clone()
                .map(|i| self.above_level[plane][i])
                .max()
                .unwrap_or(0);
            let left = ys().map(|i| self.left_level[plane][i]).max().unwrap_or(0);
            if bw == w && bh == h {
                0
            } else if top == 0 && left == 0 {
                1
            } else if top == 0 || left == 0 {
                2 + usize::from(top.max(left) > 3)
            } else if top.max(left) <= 3 {
                4
            } else if top.min(left) <= 3 {
                5
            } else {
                6
            }
        } else {
            let above = xs.clone().fold(0u8, |a, i| {
                a | self.above_level[plane][i] | self.above_dc[plane][i]
            });
            let left = ys().fold(0u8, |a, i| {
                a | self.left_level[plane][i] | self.left_dc[plane][i]
            });
            let mut c = usize::from(above != 0) + usize::from(left != 0) + 7;
            if bw * bh > w * h {
                c += 3;
            }
            c
        };
        let dc_of = |v: u8| match v {
            1 => -1,
            2 => 1,
            _ => 0,
        };
        let dc_sign: i32 = xs.map(|i| dc_of(self.above_dc[plane][i])).sum::<i32>()
            + ys().map(|i| dc_of(self.left_dc[plane][i])).sum::<i32>();
        let dc_ctx = match dc_sign.cmp(&0) {
            std::cmp::Ordering::Less => 1,
            std::cmp::Ordering::Greater => 2,
            std::cmp::Ordering::Equal => 0,
        };
        Env {
            tx_sz_ctx,
            ptype: usize::from(plane > 0),
            skip_ctx,
            dc_ctx,
        }
    }

    /// Cost of the luma tx type symbol (as written by `read_tx_type`).
    fn tx_type_cost(&self, tx: u8, t: u8) -> u32 {
        let set = self.tx_set_for(tx);
        let q = if self.hdr.segmentation.enabled {
            i32::from(self.hdr.qindex(self.b.segment_id as usize))
        } else {
            i32::from(self.hdr.quant.base_q_idx)
        };
        if set == 0 || q == 0 {
            return 0;
        }
        let pos = |inv: &[u8]| inv.iter().position(|&v| v == t).unwrap_or(0);
        let sqr = TX_SIZE_SQR[tx as usize] as usize;
        let cdf = &*self.cdf;
        if self.b.is_inter {
            match set {
                1 => symbol_cost(&cdf.inter_tx_type_set1[sqr], pos(&TX_TYPE_INTER_INV_SET1)),
                2 => symbol_cost(&cdf.inter_tx_type_set2, pos(&TX_TYPE_INTER_INV_SET2)),
                _ => bool_cost(
                    cdf.inter_tx_type_set3[sqr][0],
                    pos(&TX_TYPE_INTER_INV_SET3) == 1,
                ),
            }
        } else {
            let dir = if self.b.use_filter_intra {
                FILTER_INTRA_MODE_TO_INTRA_DIR[self.b.filter_intra_mode as usize] as usize
            } else {
                self.b.y_mode as usize
            };
            if set == 1 {
                symbol_cost(
                    &cdf.intra_tx_type_set1[sqr][dir],
                    pos(&TX_TYPE_INTRA_INV_SET1),
                )
            } else {
                symbol_cost(
                    &cdf.intra_tx_type_set2[sqr][dir],
                    pos(&TX_TYPE_INTRA_INV_SET2),
                )
            }
        }
    }

    /// Distortion of the levels `lv` (transform type `t`) measured on an
    /// actual reconstruction over the prediction.
    #[allow(clippy::too_many_arguments)]
    fn recon_dist(
        &self,
        plane: usize,
        x: usize,
        y: usize,
        tx: u8,
        t: u8,
        lv: &[i32],
        res_sse: u64,
        rec: &mut [P],
        dq: &mut [i32],
    ) -> u64 {
        let w = tx_w(tx);
        let h = tx_h(tx);
        let tw = w.min(32);
        let th = h.min(32);
        let bd = self.bd;
        let qi = self.dequant_info(plane, t, tx);
        let dq = &mut dq[..tw * th];
        dq.fill(0);
        let mut nz_rows = 0;
        let mut nz_cols = 0;
        for i in 0..th {
            for j in 0..tw {
                let pos = i * tw + j;
                let l = lv[pos];
                if l == 0 {
                    continue;
                }
                nz_rows = nz_rows.max(i + 1);
                nz_cols = nz_cols.max(j + 1);
                let qv = if pos == 0 { qi.dc } else { qi.ac };
                let q2 = match qi.qm {
                    Some(m) => crate::dsp::round2(qv * i32::from(m[pos]), 5),
                    None => qv,
                };
                let mut v = (i64::from(l.unsigned_abs()) * i64::from(q2)) & 0xFF_FFFF;
                v /= i64::from(qi.denom);
                let lim = 1i64 << (7 + bd);
                let v = if l < 0 { -v } else { v };
                dq[pos] = v.clamp(-lim, lim - 1) as i32;
            }
        }
        if nz_rows == 0 {
            return res_sse;
        }
        let Some(src) = self.src else {
            return res_sse;
        };
        let sp = &src.planes[plane];
        let cp = &self.cur.planes[plane];
        for i in 0..h {
            rec[i * w..i * w + w].copy_from_slice(&cp.data[(y + i) * cp.stride + x..][..w]);
        }
        inv_txfm_add(
            rec,
            w,
            dq,
            tx as usize,
            t as usize,
            nz_rows,
            nz_cols,
            self.b.lossless,
            bd,
        );
        let mut d = 0u64;
        for i in 0..h {
            let s = &sp.data[(y + i) * sp.stride + x..][..w];
            for j in 0..w {
                let e = s[j].to_i32() - rec[i * w + j].to_i32();
                d += (e * e) as u64;
            }
        }
        d
    }

    /// Cost of the eob syntax (eob_pt, eob_extra and the extra bits).
    fn eob_cost(&self, env: &Env, tx: u8, class: usize, eob: usize) -> u32 {
        let cdf: &CdfContext = &*self.cdf;
        let ptype = env.ptype;
        let ectx = usize::from(class != 0);
        let multisize = tx_w_log2(tx).min(5) + tx_h_log2(tx).min(5) - 4;
        let eob_pt = if eob <= 2 {
            eob
        } else {
            (31 - ((eob - 1) as u32).leading_zeros()) as usize + 2
        };
        let extra = eob.saturating_sub(if eob_pt < 2 {
            eob_pt
        } else {
            (1 << (eob_pt - 2)) + 1
        });
        let s = eob_pt - 1;
        let mut cost = match multisize {
            0 => symbol_cost(&cdf.eob_pt_16[ptype][ectx], s),
            1 => symbol_cost(&cdf.eob_pt_32[ptype][ectx], s),
            2 => symbol_cost(&cdf.eob_pt_64[ptype][ectx], s),
            3 => symbol_cost(&cdf.eob_pt_128[ptype][ectx], s),
            4 => symbol_cost(&cdf.eob_pt_256[ptype][ectx], s),
            5 => symbol_cost(&cdf.eob_pt_512[ptype], s),
            _ => symbol_cost(&cdf.eob_pt_1024[ptype], s),
        };
        if eob_pt >= 3 {
            let shift = eob_pt - 3;
            cost += bool_cost(
                cdf.eob_extra[env.tx_sz_ctx][ptype][eob_pt - 3][0],
                (extra >> shift) & 1 != 0,
            );
            cost += (eob_pt as u32 - 3) * BIT;
        }
        cost
    }

    /// Quantizes the residual of one transform block into `scratch.want`
    /// (signed levels, decoder `Quant` layout) and returns the transform type
    /// the levels use.
    pub fn enc_quantize(&mut self, plane: usize, x: usize, y: usize, tx: u8) -> u8 {
        let w = tx_w(tx);
        let h = tx_h(tx);
        let tw = w.min(32);
        let th = h.min(32);
        let bd = self.bd;
        let lossless = self.b.lossless;
        // Residual = source - prediction.
        let Some(src) = self.src else {
            for v in self.scratch.want.iter_mut() {
                *v = 0;
            }
            return 0;
        };
        let mut qs = self.scratch.enc_q.take().unwrap_or_else(QuantScratch::new);
        let mut rbuf = std::mem::take(&mut self.scratch.enc_px[3]);
        if rbuf.len() < 64 * 64 {
            rbuf.resize(64 * 64, P::default());
        }
        let QuantScratch {
            res,
            best,
            coefs,
            levels,
            lv,
            rec,
            dqbuf,
            dist,
            ranked,
            cols,
        } = &mut *qs;
        cols.clear();
        let sp = &src.planes[plane];
        let cp = &self.cur.planes[plane];
        for i in 0..h {
            let s = &sp.data[(y + i) * sp.stride + x..][..w];
            let p = &cp.data[(y + i) * cp.stride + x..][..w];
            for j in 0..w {
                res[i * w + j] = s[j].to_i32() - p[j].to_i32();
            }
        }
        best[..tw * th].fill(0);
        let res_sse = res[..w * h].iter().map(|&v| (v * v) as u64).sum::<u64>();
        let x4 = x >> 2;
        let y4 = y >> 2;
        // Candidate transform types.
        let mut cands = [0u8; 16];
        let mut nc = 1;
        if plane == 0 && !lossless {
            let set = self.tx_set_for(tx);
            let q_nonzero = i32::from(self.hdr.quant.base_q_idx) > 0;
            if set > 0 && q_nonzero && self.enc.tx_type_search > 0 {
                let list: &[u8] = match self.enc.tx_type_search {
                    1 => &[ADST_DCT, DCT_ADST, ADST_ADST, IDTX],
                    2 => &[ADST_DCT, DCT_ADST, ADST_ADST, IDTX, V_DCT, H_DCT],
                    _ => &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                };
                for &t in list {
                    if self.tx_type_allowed(set, t) {
                        cands[nc] = t;
                        nc += 1;
                    }
                }
            }
        } else if plane > 0 {
            cands[0] = self.compute_tx_type(plane, tx, x4, y4);
        }
        let env = self.txb_env(plane, x4, y4, tx);
        let lam = self.enc.lambda / f64::from(BIT);
        let skip_cdf = self.cdf.txb_skip[env.tx_sz_ctx][env.skip_ctx][0];
        let rdoq = if lossless { 0 } else { self.enc.rdoq };
        // Squared gain from the dequantized coefficient domain to samples:
        // the inverse transform scales by 2^-(rowShift + 4) * sqrt(w h) / 2
        // (times 1 / sqrt(2) on 2:1 rectangles), a power of two.
        let gain_log2 = ((tx_w_log2(tx) + tx_h_log2(tx)) / 2) as i32
            - 1
            - i32::from(TRANSFORM_ROW_SHIFT[tx as usize])
            - 4;
        let g2 = 2f64.powi(2 * gain_log2);
        let round = if self.b.is_inter {
            self.enc.round_inter
        } else {
            self.enc.round_intra
        };

        let mut best_type = cands[0];
        // All zero: the residual stays, only all_zero = 1 is coded. Without
        // RDOQ the dead-zone levels are kept whenever one is nonzero (the
        // RD all-zero choice only pays off on RD-optimized levels).
        let mut best_cost = if rdoq > 0 {
            res_sse as f64 + lam * f64::from(bool_cost(skip_cdf, true))
        } else {
            f64::MAX
        };
        dist.denom_log2 = 0;
        dist.lim = 1i64 << (7 + bd);
        dist.g2 = g2;
        let keep = self.enc.tx_type_keep as usize;
        // Where the full search finds each candidate's coefficients
        // computed already (index into `ranked`).
        let mut done = [None::<usize>; 16];
        let n = tw * th;
        if keep > 0 && nc > keep && !lossless && n <= 16 * 16 {
            // Rank the other types by a quick estimate (rounded levels, a
            // rate proxy) and run the full search on the best few only.
            let mut est = [(0f64, 0u8, 0usize); 16];
            for (k, &t) in cands[1..nc].iter().enumerate() {
                let coefs = &mut ranked[k * n..(k + 1) * n];
                fwd_txfm_cols(cols, &res[..], w, coefs, tx as usize, t as usize, bd);
                let qi = self.dequant_info(plane, t, tx);
                let den = i64::from(qi.denom);
                let dl = qi.denom.trailing_zeros();
                // Rounded levels and the squared error they leave (integers:
                // the sum is exact, as the f64 sum of the squares was).
                // Coefficients are clamped to 2^(7 + bd), so the quotients
                // have operands below 2^26 and are exact in f64.
                let level = |a: i64, q2: i64| -> (u32, u64) {
                    let l = ((a * den + q2 / 2) as f64 / q2 as f64) as i64;
                    let e = a - ((l * q2) >> dl);
                    (l as u32, (e * e) as u64)
                };
                let mut d2 = 0u64;
                match qi.qm {
                    None => {
                        let (dc, ac) = (i64::from(qi.dc.max(1)), i64::from(qi.ac.max(1)));
                        let (l, e2) = level(i64::from(coefs[0].unsigned_abs()), dc);
                        lv[0] = l;
                        d2 += e2;
                        for (o, &c) in lv[1..tw * th].iter_mut().zip(&coefs[1..tw * th]) {
                            let (l, e2) = level(i64::from(c.unsigned_abs()), ac);
                            *o = l;
                            d2 += e2;
                        }
                    }
                    Some(m) => {
                        for pos in 0..tw * th {
                            let qv = if pos == 0 { qi.dc } else { qi.ac };
                            let q2 = crate::dsp::round2(qv * i32::from(m[pos]), 5).max(1);
                            let (l, e2) =
                                level(i64::from(coefs[pos].unsigned_abs()), i64::from(q2));
                            lv[pos] = l;
                            d2 += e2;
                        }
                    }
                }
                let d = d2 as f64;
                let mut bits = 0f64;
                for &l in &lv[..tw * th] {
                    if l > 0 {
                        bits += 2.0 + 1.5 * f64::from(l).log2();
                    }
                }
                // Zeros before the end of block cost too.
                let lay = Layout::new(tx, t);
                let eob = lay
                    .scan
                    .iter()
                    .rposition(|&p| lv[p as usize] != 0)
                    .map_or(0, |e| e + 1);
                bits += 0.5 * eob as f64;
                let rate = bits * f64::from(BIT) + f64::from(self.tx_type_cost(tx, t));
                est[k] = (d * g2 + lam * rate, t, k);
            }
            let est = &mut est[..nc - 1];
            est.sort_by(|a, b| a.0.total_cmp(&b.0));
            for k in 0..keep - 1 {
                cands[1 + k] = est[k].1;
                done[1 + k] = Some(est[k].2);
            }
            nc = keep;
        }
        for (i, &t) in cands[..nc].iter().enumerate() {
            if lossless {
                fwd_wht4x4(&res[..], w, &mut coefs[..]);
            } else if let Some(k) = done[i] {
                coefs[..n].copy_from_slice(&ranked[k * n..(k + 1) * n]);
            } else {
                fwd_txfm_cols(
                    cols,
                    &res[..],
                    w,
                    &mut coefs[..],
                    tx as usize,
                    t as usize,
                    bd,
                );
            }
            let qi = self.dequant_info(plane, t, tx);
            dist.denom_log2 = qi.denom.trailing_zeros();
            let lay = Layout::new(tx, t);
            // Initial levels: dead-zone rounding, or rounding to nearest as
            // the starting point of RDOQ.
            let rnd = if rdoq > 0 { 64 } else { round };
            for pos in 0..tw * th {
                let c = coefs[pos];
                let qv = if pos == 0 { qi.dc } else { qi.ac };
                let q2 = match qi.qm {
                    Some(m) => crate::dsp::round2(qv * i32::from(m[pos]), 5),
                    None => qv,
                }
                .max(1);
                let a = i64::from(c.unsigned_abs()) * i64::from(qi.denom);
                // An exact quotient: both operands are well below 2^26.
                let n = a + i64::from(q2) * i64::from(rnd) / 128;
                let l = (n as f64 / f64::from(q2)) as i64;
                lv[pos] = (l as u32).min(MAX_LEVEL);
                dist.ac[pos] = i64::from(c.unsigned_abs());
                dist.q2[pos] = i64::from(q2);
            }
            let mut eob = 0;
            for (c, &p) in lay.scan.iter().enumerate() {
                if lv[p as usize] != 0 {
                    eob = c + 1;
                }
            }
            if eob == 0 {
                continue;
            }
            let ptype = env.ptype;
            let costs = Costs::new(
                &self.cdf.coeff_base[env.tx_sz_ctx][ptype],
                &self.cdf.coeff_br[env.tx_sz_ctx.min(3)][ptype],
                &self.cdf.coeff_base_eob[env.tx_sz_ctx][ptype],
                bool_cost(self.cdf.dc_sign[ptype][env.dc_ctx][0], coefs[0] < 0),
            );
            // The rate of the last RDOQ pass, when it is the rate of the final
            // levels: each level's contexts come from positions later in the
            // scan, which that pass has already settled.
            let mut opt_rate = None;
            if rdoq > 0 {
                opt_rate = Some(levels_pass(
                    &lay,
                    &costs,
                    &mut lv[..],
                    eob,
                    Some((&*dist, lam)),
                    &mut rec[..],
                ));
                // End of block: keep the prefix ending at scan index e - 1
                // and zero the rest, for the best total cost.
                let total_zero: f64 = rec[..eob].iter().map(|r| r.d_zero).sum();
                let mut pre_rate = 0u64;
                let mut pre_d = 0f64;
                let mut pre_z = 0f64;
                let mut best_e = eob;
                let mut best_j = f64::MAX;
                for e in 1..=eob {
                    let r = rec[e - 1];
                    if lv[lay.scan[e - 1] as usize] != 0 {
                        let rate = pre_rate
                            + u64::from(r.last_rate)
                            + u64::from(self.eob_cost(&env, tx, lay.class, e));
                        let j = lam * rate as f64
                            + pre_d
                            + r.d_chosen
                            + (total_zero - pre_z - r.d_zero);
                        if j < best_j {
                            best_j = j;
                            best_e = e;
                        }
                    }
                    pre_rate += u64::from(r.rate);
                    pre_d += r.d_chosen;
                    pre_z += r.d_zero;
                }
                if best_e < eob {
                    for &p in &lay.scan[best_e..eob] {
                        lv[p as usize] = 0;
                    }
                    eob = best_e;
                    opt_rate = (rdoq > 1).then(|| {
                        levels_pass(
                            &lay,
                            &costs,
                            &mut lv[..],
                            eob,
                            Some((&*dist, lam)),
                            &mut rec[..],
                        )
                    });
                }
            }
            // Exact rate of what will be written.
            let coef_rate = match opt_rate {
                Some(r) => r,
                None => levels_pass(&lay, &costs, &mut lv[..], eob, None, &mut rec[..]),
            };
            let mut rate =
                bool_cost(skip_cdf, false) + self.eob_cost(&env, tx, lay.class, eob) + coef_rate;
            if plane == 0 {
                rate += self.tx_type_cost(tx, t);
            }
            for pos in 0..tw * th {
                let l = lv[pos] as i32;
                levels[pos] = if coefs[pos] < 0 { -l } else { l };
            }
            let d = self.recon_dist(
                plane,
                x,
                y,
                tx,
                t,
                &levels[..tw * th],
                res_sse,
                &mut rbuf,
                &mut dqbuf[..],
            );
            let cost = d as f64 + lam * f64::from(rate);
            // Lossless must stay exact: never trade levels for rate.
            if lossless || cost < best_cost {
                best_cost = cost;
                best_type = t;
                best[..tw * th].copy_from_slice(&levels[..tw * th]);
            }
        }
        // The coefficient coder reads scan positions below tw * th only.
        self.scratch.want[..tw * th].copy_from_slice(&best[..tw * th]);
        self.scratch.enc_q = Some(qs);
        self.scratch.enc_px[3] = rbuf;
        best_type
    }
}

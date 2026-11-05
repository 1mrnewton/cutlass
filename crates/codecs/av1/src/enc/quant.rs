//! Quantization at the moment a transform block is written.
//!
//! The tile code calls [`TileDec::enc_quantize`] from `coeffs()` when it runs
//! with a writing coder. The prediction for the transform block is already in
//! the current frame, so the residual here is exactly what the decoder will
//! add its reconstruction to.

use crate::dec::coder::Coder;
use crate::dec::tile::TileDec;
use crate::dec::types::{tx_h, tx_w};
use crate::dsp::Pixel;
use crate::dsp::itx::inv_txfm_add;

use super::fwd_txfm::{fwd_txfm, fwd_wht4x4};

/// Encoder tuning for coefficient decisions.
#[derive(Clone, Copy, Debug)]
pub struct QuantParams {
    /// Lagrange multiplier for D + lambda * R, with D in squared sample
    /// errors and R in 1/8 bits.
    pub lambda: f64,
    /// Rounding offset in 1/128 of the quantizer step for intra and inter blocks.
    pub round_intra: i32,
    pub round_inter: i32,
    /// Try more than DCT_DCT for luma.
    pub tx_type_search: u8,
}

impl Default for QuantParams {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            round_intra: 48,
            round_inter: 36,
            tx_type_search: 1,
        }
    }
}

const ADST_ADST: u8 = 3;
const ADST_DCT: u8 = 1;
const DCT_ADST: u8 = 2;
const IDTX: u8 = 9;

impl<P: Pixel, C: Coder> TileDec<'_, P, C> {
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
        let mut res = [0i32; 64 * 64];
        let Some(src) = self.src else {
            for v in self.scratch.want.iter_mut() {
                *v = 0;
            }
            return 0;
        };
        let sp = &src.planes[plane];
        let cp = &self.cur.planes[plane];
        for i in 0..h {
            let s = &sp.data[(y + i) * sp.stride + x..][..w];
            let p = &cp.data[(y + i) * cp.stride + x..][..w];
            for j in 0..w {
                res[i * w + j] = s[j].to_i32() - p[j].to_i32();
            }
        }
        let x4 = x >> 2;
        let y4 = y >> 2;
        // Candidate transform types.
        let mut cands = [0u8; 5];
        let mut nc = 1;
        if plane == 0 && !lossless {
            let set = self.tx_set_for(tx);
            let q_nonzero = i32::from(self.hdr.quant.base_q_idx) > 0;
            if set > 0 && q_nonzero && self.enc.tx_type_search > 0 {
                for t in [ADST_DCT, DCT_ADST, ADST_ADST, IDTX] {
                    if self.tx_type_allowed(set, t) {
                        cands[nc] = t;
                        nc += 1;
                    }
                }
            }
        } else if plane > 0 {
            cands[0] = self.compute_tx_type(plane, tx, x4, y4);
        }
        let mut best_type = cands[0];
        let mut best_cost = f64::MAX;
        let mut best = [0i32; 32 * 32];
        let mut coefs = [0i32; 32 * 32];
        let mut levels = [0i32; 32 * 32];
        let round = if self.b.is_inter {
            self.enc.round_inter
        } else {
            self.enc.round_intra
        };
        for &t in &cands[..nc] {
            if lossless {
                fwd_wht4x4(&res, w, &mut coefs);
            } else {
                fwd_txfm(&res, w, &mut coefs, tx as usize, t as usize, bd);
            }
            let qi = self.dequant_info(plane, t, tx);
            let mut nz_rows = 0;
            let mut nz_cols = 0;
            let mut rate = 0u32;
            for i in 0..th {
                for j in 0..tw {
                    let pos = i * tw + j;
                    let c = coefs[pos];
                    let qv = if pos == 0 { qi.dc } else { qi.ac };
                    let q2 = match qi.qm {
                        Some(m) => crate::dsp::round2(qv * i32::from(m[pos]), 5),
                        None => qv,
                    }
                    .max(1);
                    let a = i64::from(c.unsigned_abs()) * i64::from(qi.denom);
                    let mut l =
                        ((a + i64::from(q2) * i64::from(round) / 128) / i64::from(q2)) as i32;
                    l = l.min((1 << 20) - 1);
                    levels[pos] = if c < 0 { -l } else { l };
                    if l != 0 {
                        nz_rows = nz_rows.max(i + 1);
                        nz_cols = nz_cols.max(j + 1);
                        rate += 8 * (2 + 2 * (32 - (l as u32).leading_zeros()));
                    }
                }
            }
            // Distortion from an actual reconstruction.
            let dist = if nz_rows == 0 {
                res[..w * h].iter().map(|&v| (v * v) as u64).sum::<u64>()
            } else {
                let mut rec = [P::default(); 64 * 64];
                for i in 0..h {
                    rec[i * w..i * w + w].copy_from_slice(&cp.data[(y + i) * cp.stride + x..][..w]);
                }
                let mut dq = [0i32; 32 * 32];
                for pos in 0..tw * th {
                    let l = levels[pos];
                    if l == 0 {
                        continue;
                    }
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
                inv_txfm_add(
                    &mut rec,
                    w,
                    &mut dq,
                    tx as usize,
                    t as usize,
                    nz_rows,
                    nz_cols,
                    lossless,
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
            };
            let type_rate = if t == 0 { 0 } else { 16 };
            let cost = dist as f64 + self.enc.lambda * f64::from(rate + type_rate);
            if cost < best_cost {
                best_cost = cost;
                best_type = t;
                best[..tw * th].copy_from_slice(&levels[..tw * th]);
            }
        }
        let want = &mut self.scratch.want;
        for v in want.iter_mut() {
            *v = 0;
        }
        want[..tw * th].copy_from_slice(&best[..tw * th]);
        best_type
    }
}

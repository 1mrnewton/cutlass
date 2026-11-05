//! Mode decision. The tile code calls these hooks when it runs with a writing
//! coder: [`TileDec::enc_partition`] when it reaches a partition node and
//! [`TileDec::enc_block`] when it starts a block. Decisions made in
//! `enc_block` see the reconstructed neighbors and the real motion vector stack,
//! exactly as the decoder will.

use crate::dec::coder::Coder;
use crate::dec::tile::{Block, TileDec};
use crate::dec::types::{
    BLOCK_8X8, DC_PRED, GLOBALMV, INTRA_FRAME, LAST_FRAME, Mv, NEARESTMV, NEARMV, NEWMV, NONE,
    PAETH_PRED, SMOOTH_PRED, bh4, block_height, block_width, bw4, mi_height_log2, mi_width_log2,
    plane_residual_size, tx_h, tx_w,
};
use crate::dsp::Pixel;
use crate::spec::tables::{MAX_TX_SIZE_RECT, NUM_4X4_BLOCKS_WIDE, PARTITION_SUBSIZE};

use super::me::{MeParams, motion_search, satd, sse};

/// Per-frame encoder settings the hooks read.
#[derive(Clone, Copy, Debug)]
pub struct ModeParams {
    /// Multiplier from bits to SATD units.
    pub lambda_satd: f64,
    /// Smallest and largest block edge the partition search uses (4..=64).
    pub min_block: usize,
    pub max_block: usize,
    /// Residual SSE per sample below which a block is coded as skip.
    pub skip_sse_per_px: f64,
    pub inter_frame: bool,
    pub me_range: i32,
    /// 0 = full pel only, 2 = quarter pel, 3 = eighth pel.
    pub subpel: u8,
    pub intra_modes: u8,
}

impl Default for ModeParams {
    fn default() -> Self {
        Self {
            lambda_satd: 4.0,
            min_block: 8,
            max_block: 64,
            skip_sse_per_px: 1.0,
            inter_frame: false,
            me_range: 32,
            subpel: 2,
            intra_modes: 13,
        }
    }
}

const PARTITION_NONE: usize = 0;
const PARTITION_HORZ: usize = 1;
const PARTITION_VERT: usize = 2;
const PARTITION_SPLIT: usize = 3;

impl<P: Pixel, C: Coder> TileDec<'_, P, C> {
    /// SATD of a DC prediction formed from the source neighbors (open loop).
    fn est_intra(&self, x: usize, y: usize, w: usize, h: usize) -> f64 {
        let Some(src) = self.src else { return 0.0 };
        let p = &src.planes[0];
        let mut sum = 0i64;
        let mut n = 0i64;
        if y > 0 {
            for j in 0..w {
                sum += i64::from(p.data[(y - 1) * p.stride + x + j].to_i32());
            }
            n += w as i64;
        }
        if x > 0 {
            for i in 0..h {
                sum += i64::from(p.data[(y + i) * p.stride + x - 1].to_i32());
            }
            n += h as i64;
        }
        let dc = if n > 0 {
            (sum + n / 2) / n
        } else {
            1 << (self.bd - 1)
        } as i32;
        let mut pred = [P::default(); 64 * 64];
        let v = P::from_i32(dc);
        for s in pred[..w * h].iter_mut() {
            *s = v;
        }
        f64::from(satd(&p.data[y * p.stride + x..], p.stride, &pred, w, w, h))
    }

    /// SATD of the best full-pel motion from LAST (open loop, small window).
    fn est_inter(&mut self, x: usize, y: usize, w: usize, h: usize) -> f64 {
        let Some(src) = self.src else { return f64::MAX };
        let Some(rv) = self.refs[LAST_FRAME as usize].as_ref() else {
            return f64::MAX;
        };
        let rp = &rv.pic.planes[0];
        let sp = &src.planes[0];
        let params = MeParams {
            max_range: self.mode.me_range / 2,
            subpel: 0,
            ..MeParams::default()
        };
        let lam = self.mode.lambda_satd;
        let cost = |r: i32, c: i32| -> u32 { (lam * (2.0 + bits(r) + bits(c))) as u32 };
        let res = motion_search(
            &mut self.me_scratch,
            &sp.data,
            sp.stride,
            x,
            y,
            w,
            h,
            &rp.data,
            rp.stride,
            rp.w,
            rp.h,
            &[(0, 0)],
            &params,
            &cost,
            self.bd,
        );
        f64::from(res.cost)
    }

    fn est_block(&mut self, x: usize, y: usize, w: usize, h: usize) -> f64 {
        let fw = self.fs.mi_cols * 4;
        let fh = self.fs.mi_rows * 4;
        if x >= fw || y >= fh {
            return 0.0;
        }
        let w = w.min(fw - x).min(64);
        let h = h.min(fh - y).min(64);
        let intra = self.est_intra(x, y, w, h) + self.mode.lambda_satd * 6.0;
        if self.mode.inter_frame {
            intra.min(self.est_inter(x, y, w, h))
        } else {
            intra
        }
    }

    /// Chooses the partition at a node (open-loop estimate of NONE vs SPLIT).
    pub fn enc_partition(
        &mut self,
        r: usize,
        c: usize,
        bsize: u8,
        has_rows: bool,
        has_cols: bool,
    ) -> usize {
        let n = bw4(bsize) * 4;
        let half = n / 2;
        let x = c * 4;
        let y = r * 4;
        let force_split = n > self.mode.max_block;
        if !has_rows && !has_cols {
            return PARTITION_SPLIT;
        }
        if !has_rows || !has_cols {
            // Only one of the two halves exists: HORZ (or VERT) codes one block.
            if force_split || half < self.mode.min_block {
                return PARTITION_SPLIT;
            }
            let whole = if !has_rows {
                self.est_block(x, y, n, half)
            } else {
                self.est_block(x, y, half, n)
            };
            let quads = self.est_block(x, y, half, half)
                + if !has_rows {
                    self.est_block(x + half, y, half, half)
                } else {
                    self.est_block(x, y + half, half, half)
                };
            if quads + self.mode.lambda_satd * 4.0 < whole {
                return PARTITION_SPLIT;
            }
            return if !has_rows {
                PARTITION_HORZ
            } else {
                PARTITION_VERT
            };
        }
        if force_split {
            return PARTITION_SPLIT;
        }
        if half < self.mode.min_block || bsize < BLOCK_8X8 {
            return PARTITION_NONE;
        }
        let none = self.est_block(x, y, n, n);
        let split = self.est_block(x, y, half, half)
            + self.est_block(x + half, y, half, half)
            + self.est_block(x, y + half, half, half)
            + self.est_block(x + half, y + half, half, half)
            + self.mode.lambda_satd * 8.0;
        if split < none {
            PARTITION_SPLIT
        } else {
            PARTITION_NONE
        }
    }

    /// Decides one block's modes. `self.b` already holds the block position,
    /// size and availability; the result goes to `self.pb`.
    pub fn enc_block(&mut self) {
        let mut pb = Block {
            mi_row: self.b.mi_row,
            mi_col: self.b.mi_col,
            bsize: self.b.bsize,
            has_chroma: self.b.has_chroma,
            avail_u: self.b.avail_u,
            avail_l: self.b.avail_l,
            avail_u_chroma: self.b.avail_u_chroma,
            avail_l_chroma: self.b.avail_l_chroma,
            ref_frame: [INTRA_FRAME, NONE],
            compound_idx: 1,
            ..Block::default()
        };
        let bs = self.b.bsize;
        let x = self.b.mi_col * 4;
        let y = self.b.mi_row * 4;
        let tx = MAX_TX_SIZE_RECT[bs as usize];
        // Evaluate on the first transform block's area (all of a block up to 64x64).
        let w = tx_w(tx).min(block_width(bs));
        let h = tx_h(tx).min(block_height(bs));
        let lam = self.mode.lambda_satd;

        // Intra luma.
        let saved = self.b.clone();
        let (best_mode, intra_cost) = self.pick_intra_luma(x, y, w, h, tx);
        pb.y_mode = best_mode;
        pb.tx_size = tx;
        let mut best_cost = intra_cost
            + if self.mode.inter_frame {
                lam * 6.0
            } else {
                0.0
            };
        let mut inter_choice: Option<(u8, Mv, usize)> = None;
        if self.mode.inter_frame {
            if let Some((mode, mv, idx, cost)) =
                self.pick_inter(x, y, block_width(bs), block_height(bs))
            {
                if cost < best_cost {
                    best_cost = cost;
                    inter_choice = Some((mode, mv, idx));
                }
            }
        }
        self.b = saved;
        let _ = best_cost;
        if let Some((mode, mv, idx)) = inter_choice {
            pb.is_inter = true;
            pb.ref_frame = [LAST_FRAME, NONE];
            pb.y_mode = mode;
            pb.mv = [mv, Mv::ZERO];
            pb.ref_mv_idx = idx;
            pb.interp_filter = [0, 0];
            pb.uv_mode = DC_PRED;
        } else {
            pb.uv_mode = if pb.has_chroma {
                self.pick_intra_chroma(&pb)
            } else {
                DC_PRED
            };
        }
        // Skip when the prediction is already close to the source.
        pb.skip = self.block_is_skip(&pb);
        self.pb = pb;
        self.plan_q = self.current_q_index;
        self.plan_delta_lf = self.delta_lf;
    }

    fn pick_intra_luma(&mut self, x: usize, y: usize, w: usize, h: usize, tx: u8) -> (u8, f64) {
        let Some(src) = self.src else {
            return (DC_PRED, 0.0);
        };
        let sb_mask = self.sb_size4 - 1;
        let sub_row = self.b.mi_row & sb_mask;
        let sub_col = self.b.mi_col & sb_mask;
        let step_x = tx_w(tx) >> 2;
        let step_y = tx_h(tx) >> 2;
        let bdf = &self.block_decoded[0];
        let have_ar = bdf[sub_row][sub_col + step_x + 1];
        let have_bl = bdf[sub_row + step_y + 1][sub_col];
        let (al, au) = (self.b.avail_l, self.b.avail_u);
        let lw = tx_w(tx).trailing_zeros();
        let lh = tx_h(tx).trailing_zeros();
        self.b.is_inter = false;
        self.b.use_filter_intra = false;
        self.b.palette_size_y = 0;
        self.b.angle_delta_y = 0;
        let lam = self.mode.lambda_satd;
        let mut best = (DC_PRED, f64::MAX);
        let n_modes = self.mode.intra_modes.min(13);
        for mode in 0..n_modes {
            self.predict_intra(0, x, y, al, au, have_ar, have_bl, mode, lw, lh);
            let cp = &self.cur.planes[0];
            let sp = &src.planes[0];
            let d = satd(
                &sp.data[y * sp.stride + x..],
                sp.stride,
                &cp.data[y * cp.stride + x..],
                cp.stride,
                w,
                h,
            );
            let rate = match mode {
                DC_PRED => 1.0,
                SMOOTH_PRED | PAETH_PRED => 2.5,
                _ => 3.5,
            };
            let cost = f64::from(d) + lam * rate;
            if cost < best.1 {
                best = (mode, cost);
            }
        }
        best
    }

    fn pick_intra_chroma(&mut self, pb: &Block) -> u8 {
        let Some(src) = self.src else { return DC_PRED };
        let (sx, sy) = (u32::from(self.sub_x), u32::from(self.sub_y));
        let psz = plane_residual_size(pb.bsize, sx != 0, sy != 0);
        let uv_tx = MAX_TX_SIZE_RECT[psz as usize];
        let uv_tx = self.get_tx_size_for(1, uv_tx, pb.bsize);
        let w = tx_w(uv_tx);
        let h = tx_h(uv_tx);
        let x = (pb.mi_col >> sx) * 4;
        let y = (pb.mi_row >> sy) * 4;
        let saved = self.b.clone();
        self.b.y_mode = pb.y_mode;
        self.b.angle_delta_uv = 0;
        let (al, au) = (self.b.avail_l_chroma, self.b.avail_u_chroma);
        let lam = self.mode.lambda_satd;
        let mut best = (DC_PRED, f64::MAX);
        for mode in [DC_PRED, 1, 2, SMOOTH_PRED, PAETH_PRED] {
            let mut d = 0u32;
            for plane in 1..3 {
                self.predict_intra(
                    plane,
                    x,
                    y,
                    al,
                    au,
                    false,
                    false,
                    mode,
                    w.trailing_zeros(),
                    h.trailing_zeros(),
                );
                let cp = &self.cur.planes[plane];
                let sp = &src.planes[plane];
                d += satd(
                    &sp.data[y * sp.stride + x..],
                    sp.stride,
                    &cp.data[y * cp.stride + x..],
                    cp.stride,
                    w,
                    h,
                );
            }
            let cost = f64::from(d) + lam * if mode == DC_PRED { 1.0 } else { 3.0 };
            if cost < best.1 {
                best = (mode, cost);
            }
        }
        self.b = saved;
        best.0
    }

    fn get_tx_size_for(&self, plane: usize, tx: u8, bsize: u8) -> u8 {
        let saved = self.b.bsize;
        let _ = saved;
        let _ = plane;
        // Chroma transform size as get_tx_size computes it for `bsize`.
        let mut b = self.b.clone();
        b.bsize = bsize;
        let uv = MAX_TX_SIZE_RECT[plane_residual_size(bsize, self.sub_x, self.sub_y) as usize];
        if tx_w(uv) == 64 || tx_h(uv) == 64 {
            if tx_w(uv) == 16 {
                return 9;
            }
            if tx_h(uv) == 16 {
                return 10;
            }
            return 3;
        }
        let _ = tx;
        uv
    }

    /// Returns (mode, mv, ref_mv_idx, cost) of the best single-reference mode from LAST.
    fn pick_inter(
        &mut self,
        x: usize,
        y: usize,
        bw: usize,
        bh: usize,
    ) -> Option<(u8, Mv, usize, f64)> {
        let src = self.src?;
        self.refs[LAST_FRAME as usize].as_ref()?;
        let fw = self.fs.mi_cols * 4;
        let fh = self.fs.mi_rows * 4;
        let w = bw.min(fw - x);
        let h = bh.min(fh - y);
        self.b.ref_frame = [LAST_FRAME, NONE];
        self.b.is_inter = true;
        self.b.use_intrabc = false;
        self.find_mv_stack(false);
        let stack: Vec<Mv> = (0..self.b.num_mv_found.max(2))
            .map(|i| self.b.ref_stack_mv[i][0])
            .collect();
        let global = self.b.global_mvs[0];
        let lam = self.mode.lambda_satd;
        let hp = self.hdr.allow_high_precision_mv;
        // Motion search seeded with the stack.
        let cands: Vec<(i32, i32)> = stack
            .iter()
            .map(|m| (i32::from(m.row), i32::from(m.col)))
            .collect();
        let pred0 = stack[0];
        let params = MeParams {
            max_range: self.mode.me_range,
            subpel: if hp {
                self.mode.subpel
            } else {
                self.mode.subpel.min(2)
            },
            ..MeParams::default()
        };
        let cost_fn = |r: i32, c: i32| -> u32 {
            let dr = r - i32::from(pred0.row);
            let dc = c - i32::from(pred0.col);
            (lam * (3.0 + bits(dr) + bits(dc))) as u32
        };
        let me = {
            let rv = self.refs[LAST_FRAME as usize].as_ref()?;
            let rp = &rv.pic.planes[0];
            let sp = &src.planes[0];
            motion_search(
                &mut self.me_scratch,
                &sp.data,
                sp.stride,
                x,
                y,
                w,
                h,
                &rp.data,
                rp.stride,
                rp.w,
                rp.h,
                &cands,
                &params,
                &cost_fn,
                self.bd,
            )
        };
        let mut mv_new = Mv::new(me.mv.0, me.mv.1);
        if !hp {
            mv_new = Mv::new(me.mv.0 & !1, me.mv.1 & !1);
        }
        let mut options: Vec<(u8, Mv, usize, f64)> = Vec::with_capacity(4);
        options.push((NEARESTMV, stack[0], 0, 1.0));
        if self.b.num_mv_found > 1 {
            options.push((NEARMV, stack[1], 1, 2.5));
        }
        options.push((GLOBALMV, global, 0, 2.0));
        let dr = i32::from(mv_new.row) - i32::from(pred0.row);
        let dc = i32::from(mv_new.col) - i32::from(pred0.col);
        options.push((NEWMV, mv_new, 0, 3.0 + bits(dr) + bits(dc)));
        let mut best: Option<(u8, Mv, usize, f64)> = None;
        for (mode, mv, idx, rate) in options {
            if best.is_some_and(|b| b.1 == mv && b.0 != NEWMV) {
                continue;
            }
            if self
                .predict_into_scratch(0, LAST_FRAME, x, y, w, h, mv, [0, 0])
                .is_err()
            {
                continue;
            }
            let sp = &src.planes[0];
            let d = satd(
                &sp.data[y * sp.stride + x..],
                sp.stride,
                &self.scratch.obmc,
                w,
                w,
                h,
            );
            let cost = f64::from(d) + lam * rate;
            if best.is_none_or(|b| cost < b.3) {
                best = Some((mode, mv, idx, cost));
            }
        }
        best
    }

    fn block_is_skip(&mut self, pb: &Block) -> bool {
        let Some(src) = self.src else { return false };
        if !pb.is_inter {
            return false;
        }
        let x = pb.mi_col * 4;
        let y = pb.mi_row * 4;
        let fw = self.fs.mi_cols * 4;
        let fh = self.fs.mi_rows * 4;
        let w = block_width(pb.bsize).min(fw - x);
        let h = block_height(pb.bsize).min(fh - y);
        if self
            .predict_into_scratch(0, LAST_FRAME, x, y, w, h, pb.mv[0], [0, 0])
            .is_err()
        {
            return false;
        }
        let sp = &src.planes[0];
        let e = sse(
            &sp.data[y * sp.stride + x..],
            sp.stride,
            &self.scratch.obmc,
            w,
            w,
            h,
        );
        (e as f64) < self.mode.skip_sse_per_px * (w * h) as f64
    }
}

/// Rough bit count of a motion vector component difference (1/8 units).
fn bits(v: i32) -> f64 {
    let a = v.unsigned_abs();
    if a == 0 {
        0.5
    } else {
        2.0 + 2.0 * f64::from(32 - a.leading_zeros())
    }
}

// Silence helpers kept for later search stages.
const _: (usize, usize) = (
    PARTITION_SUBSIZE[0][0] as usize,
    NUM_4X4_BLOCKS_WIDE[0] as usize,
);
const _: fn(u8) -> u32 = mi_width_log2;
const _: fn(u8) -> u32 = mi_height_log2;
const _: fn(u8) -> usize = bh4;

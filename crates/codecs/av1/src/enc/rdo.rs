//! Mode decision. The tile code calls these hooks when it runs with a writing
//! coder: [`TileDec::enc_partition`] when it reaches a partition node and
//! [`TileDec::enc_block`] when it starts a block. Decisions made in
//! `enc_block` see the reconstructed neighbors and the real motion vector stack,
//! exactly as the decoder will.

use crate::dec::coder::Coder;
use crate::dec::tile::{Block, TileDec};
use crate::dec::types::{
    ALTREF_FRAME, BLOCK_8X8, BWDREF_FRAME, DC_PRED, GLOBALMV, GOLDEN_FRAME, INTRA_FRAME,
    LAST_FRAME, Mv, NEAREST_NEARESTMV, NEARESTMV, NEARMV, NEW_NEWMV, NEWMV, NONE, PAETH_PRED,
    SMOOTH_PRED, UV_CFL_PRED, bh4, block_height, block_width, bw4, mi_height_log2, mi_width_log2,
    plane_residual_size, tx_h, tx_w,
};
use crate::dsp::Pixel;
use crate::dsp::ipred::{EDGE_LEN, EDGE_OFF, cfl_ac, cfl_pred, predict_clipped};
use crate::dsp::itx::inv_txfm_add;
use crate::spec::tables::{
    MAX_TX_SIZE_RECT, NUM_4X4_BLOCKS_WIDE, PARTITION_SUBSIZE, SPLIT_TX_SIZE,
};

use super::fwd_txfm::fwd_txfm;

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
    /// How many transform splits the size search tries (0 keeps the largest).
    pub tx_depth: u8,
    /// Try chroma from luma.
    pub cfl: bool,
    pub angle_deltas: bool,
    pub filter_intra: bool,
    /// Allow 64-point transforms (chosen against their split by RD cost).
    pub tx64: bool,
    /// Partition by transform-coded RD estimates instead of SATD.
    pub rd_partition: bool,
    /// Reference frame types (index 1..=7) the inter search may use.
    pub ref_mask: [bool; 8],
    /// Decide intra against inter by transform-coded RD instead of SATD.
    pub rd_inter_intra: bool,
    /// Partition early exit: keep a block whole when its estimate is below
    /// this many bits' worth of cost.
    pub early_none: f64,
    /// Intra luma modes re-ranked by transform-coded RD (1 = SATD only).
    pub intra_rd_cands: u8,
    /// Scale of the signalling cost charged to a split in partition search.
    pub split_bits_scale: f64,
    /// Choose the interpolation filter per block (frame filter switchable).
    pub filter_search: bool,
    /// Interpolation filter the sub-sample motion search evaluates with.
    pub me_filter: u8,
    /// Try local warped motion for single-reference inter blocks.
    pub local_warp: bool,
    /// Try overlapped block motion compensation.
    pub obmc: bool,
    pub skip_mode: bool,
    /// Mode bits a skip-mode block saves over a skipped regular inter block.
    pub skip_mode_sse: f64,
    /// Estimate transform distortion in the transform domain (no inverse).
    pub fast_tx_dist: bool,
    /// Also try horizontal and vertical halves in partition search.
    pub rect_partitions: bool,
    /// Sub-sample motion search tries 5 instead of 8 neighbours per step.
    pub subpel_pruned: bool,
    /// Inter frames: intra angle deltas and filter intra are tried only when
    /// the best intra SATD cost is below this multiple of the inter one (0: always).
    pub intra_refine_ratio: f64,
    /// Inter frames: the oblique intra directions are tried only when the best
    /// of DC, V, H, SMOOTH and PAETH is below this multiple of the inter cost (0: always).
    pub intra_sweep_ratio: f64,
    /// Inter frames: inter is taken without the transform-coded comparison
    /// when the intra SATD cost exceeds this multiple of the inter one (0: never).
    pub inter_clear_ratio: f64,
    /// Partition estimates transform-code an intra or inter candidate only
    /// when its SATD cost is within this factor of the other one (0: always).
    pub est_skip_ratio: f64,
    /// Inter mode decisions charge the signalling cost from the CDFs.
    pub exact_rates: bool,
    /// Inter candidates (best per reference and compound) compared by
    /// transform-coded RD, and how far behind the best SATD cost they may be.
    pub inter_rd_cands: usize,
    pub inter_rd_ratio: f64,
    /// Modes per single reference offered to that comparison.
    pub inter_rd_modes: usize,
    /// The intra transform split is judged with each smaller transform
    /// block predicted on its own (as coded), not cut from the whole one.
    pub intra_split_pred: bool,
    /// The intra/inter comparison charges the is_inter flag from its CDF.
    pub inter_flag_bits: bool,
    /// Intra mode decisions charge mode, angle, filter intra and CFL bits
    /// from the CDFs.
    pub exact_intra_rates: bool,
    /// Screen content intra frames: partition estimates include a palette
    /// and the chroma.
    pub screen_est: bool,
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
            tx_depth: 1,
            cfl: true,
            angle_deltas: true,
            filter_intra: true,
            tx64: true,
            rd_partition: true,
            ref_mask: [false, true, true, true, true, true, true, true],
            rd_inter_intra: true,
            early_none: 16.0,
            intra_rd_cands: 3,
            split_bits_scale: 1.0,
            filter_search: true,
            me_filter: 0,
            local_warp: true,
            obmc: true,
            skip_mode: true,
            skip_mode_sse: 4.0,
            fast_tx_dist: true,
            rect_partitions: false,
            subpel_pruned: false,
            intra_refine_ratio: 0.0,
            intra_sweep_ratio: 0.0,
            inter_clear_ratio: 0.0,
            est_skip_ratio: 0.0,
            exact_rates: true,
            inter_rd_cands: 1,
            inter_rd_ratio: 1.1,
            inter_rd_modes: 1,
            intra_split_pred: true,
            inter_flag_bits: true,
            exact_intra_rates: true,
            screen_est: true,
        }
    }
}

/// An inter mode decision.
#[derive(Clone, Copy)]
struct InterChoice {
    refs: [i8; 2],
    mode: u8,
    mv: [Mv; 2],
    idx: usize,
    cost: f64,
    /// Signalling bits counted in `cost`.
    bits: f64,
}

/// A single-reference mode option: mode, vector, `ref_mv_idx`, SATD cost
/// (with its signalling) and signalling bits.
type ModeOpt = (u8, Mv, usize, f64, f64);

/// What a translational prediction in the mode search depends on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PredKey {
    plane: usize,
    refs: [i8; 2],
    mv: [Mv; 2],
    filters: [u8; 2],
    area: [usize; 4],
    intrabc: bool,
}

/// Predictions kept per block by the mode search.
const PRED_CACHE: usize = 6;

const PARTITION_NONE: usize = 0;
const PARTITION_HORZ: usize = 1;
const PARTITION_VERT: usize = 2;
const PARTITION_SPLIT: usize = 3;

impl<P: Pixel, C: Coder> TileDec<'_, P, C> {
    /// The cost of the extra signalling a split brings, in the units `est_block` returns.
    fn split_penalty(&self, bits: f64) -> f64 {
        if self.mode.rd_partition {
            self.enc.lambda * bits * self.mode.split_bits_scale
        } else {
            self.mode.lambda_satd * bits
        }
    }

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

    /// Rate-distortion estimate (squared error + lambda * bits) of coding
    /// a `w`x`h` luma area with the best of a few intra predictions formed
    /// from the source neighbors, and, in inter frames, with full-sample
    /// motion from LAST.
    fn est_block_rd(&mut self, x: usize, y: usize, w: usize, h: usize) -> f64 {
        let Some(src) = self.src else { return 0.0 };
        let fw = self.fs.mi_cols * 4;
        let fh = self.fs.mi_rows * 4;
        if x >= fw || y >= fh {
            return 0.0;
        }
        let cw = w.min(fw - x);
        let ch = h.min(fh - y);
        // Work on a power-of-two area and scale the cost to the clipped one.
        let pw = 1usize << (usize::BITS - 1 - cw.leading_zeros());
        let ph = 1usize << (usize::BITS - 1 - ch.leading_zeros());
        let (pw, ph) = (pw.min(64).min(4 * ph), ph.min(64).min(4 * pw));
        if pw < 4 || ph < 4 {
            return 0.0;
        }
        let scale = (cw * ch) as f64 / (pw * ph) as f64;
        let cap = if self.mode.tx64 { 64 } else { 32 };
        let (tw, th) = (pw.min(cap), ph.min(cap));
        let tx = (0..19u8)
            .find(|&t| tx_w(t) == tw && tx_h(t) == th)
            .unwrap_or(0);
        let lam_bits = self.enc.lambda;
        let sp = &src.planes[0];
        let bd = self.bd;
        let base = 1i32 << (bd - 1);
        let have_above = y > 0;
        let have_left = x > 0;
        let mut above = [P::default(); EDGE_LEN];
        let mut left = [P::default(); EDGE_LEN];
        let n = pw + ph;
        for i in 0..n {
            above[EDGE_OFF + i] = if have_above {
                sp.data[(y - 1) * sp.stride + (x + i).min(fw - 1)]
            } else if have_left {
                sp.data[y * sp.stride + x - 1]
            } else {
                P::from_i32(base - 1)
            };
            left[EDGE_OFF + i] = if have_left {
                sp.data[(y + i).min(fh - 1) * sp.stride + x - 1]
            } else if have_above {
                sp.data[(y - 1) * sp.stride + x]
            } else {
                P::from_i32(base + 1)
            };
        }
        let corner = match (have_above, have_left) {
            (true, true) => sp.data[(y - 1) * sp.stride + x - 1],
            (true, false) => sp.data[(y - 1) * sp.stride + x],
            (false, true) => sp.data[y * sp.stride + x - 1],
            _ => P::from_i32(base),
        };
        above[EDGE_OFF - 1] = corner;
        left[EDGE_OFF - 1] = corner;
        let mut pred = take_buf(&mut self.scratch.enc_px[1], 64 * 64, P::default());
        let mut best_pred = take_buf(&mut self.scratch.enc_px[2], 64 * 64, P::default());
        let mut best = (f64::MAX, 0.0);
        let lam = self.mode.lambda_satd;
        for (mode, bits) in [
            (DC_PRED, 1.5),
            (1, 3.0),
            (2, 3.0),
            (SMOOTH_PRED, 3.0),
            (PAETH_PRED, 3.0),
        ] {
            let (mut a, mut l) = (above, left);
            predict_clipped(
                &mut pred, pw, pw, ph, mode, 0, &mut a, &mut l, have_left, have_above, false,
                false, pw, ph, bd,
            );
            let d = f64::from(satd(
                &sp.data[y * sp.stride + x..],
                sp.stride,
                &pred,
                pw,
                pw,
                ph,
            ));
            let c = d + lam * bits;
            if c < best.0 {
                best = (c, bits);
                best_pred[..pw * ph].copy_from_slice(&pred[..pw * ph]);
            }
        }
        let mut res = take_buf(&mut self.scratch.enc_i32[2], 64 * 64, 0);
        let mut inter = None;
        let mut est_mv = None;
        if self.mode.inter_frame
            && let Some(rv) = self.refs[LAST_FRAME as usize].as_ref()
        {
            let rp = &rv.pic.planes[0];
            let params = MeParams {
                max_range: self.mode.me_range / 4,
                subpel: 0,
                ..MeParams::default()
            };
            let cost_fn = |r: i32, c: i32| -> u32 { (lam * (2.0 + bits(r) + bits(c))) as u32 };
            // Also start from the motion of the enclosing area and of the
            // left and above neighbours of the same size.
            let mut cands = [(0, 0), self.est_hint, (0, 0), (0, 0), (0, 0)];
            let memo = &self.me_scratch.est_memo;
            if let Some(e) = memo.iter().rev().find(|e| {
                let [ex, ey, ew, eh] = e.0;
                ex <= x && ey <= y && ex + ew >= x + w && ey + eh >= y + h && ew * eh > w * h
            }) {
                cands[2] = e.2;
            }
            for (i, key) in [[x.wrapping_sub(w), y, w, h], [x, y.wrapping_sub(h), w, h]]
                .iter()
                .enumerate()
            {
                if let Some(e) = memo.iter().find(|e| e.0 == *key) {
                    cands[3 + i] = e.2;
                }
            }
            let me = motion_search(
                &mut self.me_scratch,
                &sp.data,
                sp.stride,
                x,
                y,
                pw,
                ph,
                &rp.data,
                rp.stride,
                rp.w,
                rp.h,
                &cands,
                &params,
                &cost_fn,
                bd,
            );
            let (dy, dx) = (me.mv.0 >> 3, me.mv.1 >> 3);
            self.est_hint = me.mv;
            est_mv = Some((dy, dx));
            for i in 0..ph {
                let ry = (y as i32 + i as i32 + dy).clamp(0, rp.h as i32 - 1) as usize;
                for j in 0..pw {
                    let rx = (x as i32 + j as i32 + dx).clamp(0, rp.w as i32 - 1) as usize;
                    pred[i * pw + j] = rp.data[ry * rp.stride + rx];
                }
            }
            let mvb = 2.0 + bits(me.mv.0) + bits(me.mv.1);
            let d = f64::from(satd(
                &sp.data[y * sp.stride + x..],
                sp.stride,
                &pred,
                pw,
                pw,
                ph,
            ));
            inter = Some((d + lam * mvb, mvb));
        }
        // Screen content intra frames: a palette, by transform-coded RD.
        // (In inter frames it would favour intra blocks whose chroma the
        // block decision then serves badly.)
        let mut cost = f64::MAX;
        if self.hdr.allow_screen_content_tools
            && !self.mode.inter_frame
            && pw == cw
            && ph == ch
            && pw.min(ph) >= 8
            && self.mode.screen_est
            && let Some((n, colors, _, pbits)) = self.pick_palette_y(x, y, pw, ph)
        {
            for i in 0..ph {
                for j in 0..pw {
                    let k = usize::from(self.color_map_y[i * pw + j]).min(usize::from(n) - 1);
                    res[i * pw + j] =
                        sp.data[(y + i) * sp.stride + x + j].to_i32() - i32::from(colors[k]);
                }
            }
            cost = self.tx_cost(&res[..pw * ph], pw, ph, tx) + lam_bits * pbits;
        }
        let chroma = if self.hdr.allow_screen_content_tools
            && self.mode.screen_est
            && pw == cw
            && ph == ch
        {
            self.est_chroma_sc(x, y, pw, ph, est_mv)
        } else {
            0.0
        };
        // Transform-code only the candidates whose SATD cost is not far
        // behind the other's.
        let t = self.mode.est_skip_ratio;
        if inter.is_none_or(|(s, _)| t <= 0.0 || s * t >= best.0) {
            self.residual_of(&best_pred, pw, x, y, pw, ph, &mut res);
            cost = cost.min(self.tx_cost(&res[..pw * ph], pw, ph, tx) + lam_bits * best.1);
        }
        if let Some((s, mvb)) = inter
            && (t <= 0.0 || s <= best.0 * t)
        {
            self.residual_of(&pred, pw, x, y, pw, ph, &mut res);
            cost = cost.min(self.tx_cost(&res[..pw * ph], pw, ph, tx) + lam_bits * mvb);
        }
        self.scratch.enc_px[1] = pred;
        self.scratch.enc_px[2] = best_pred;
        self.scratch.enc_i32[2] = res;
        (cost + chroma) * scale
    }

    /// The partition estimate of an area, computed once per superblock.
    fn est_block(&mut self, x: usize, y: usize, w: usize, h: usize) -> f64 {
        let key = [x, y, w, h];
        if let Some(&(_, v, _)) = self.me_scratch.est_memo.iter().find(|e| e.0 == key) {
            return v;
        }
        let v = self.est_block_uncached(x, y, w, h);
        self.me_scratch.est_memo.push((key, v, self.est_hint));
        v
    }

    fn est_block_uncached(&mut self, x: usize, y: usize, w: usize, h: usize) -> f64 {
        if self.mode.rd_partition {
            return self.est_block_rd(x, y, w, h);
        }
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
        let p = self.enc_partition_pick(r, c, bsize, has_rows, has_cols);
        // A partition whose blocks have no chroma size in this subsampling is not allowed.
        let sub = PARTITION_SUBSIZE[p][bsize as usize];
        if self.num_planes > 1
            && plane_residual_size(sub, self.sub_x, self.sub_y) == crate::dec::types::BLOCK_INVALID
        {
            return PARTITION_SPLIT;
        }
        p
    }

    fn enc_partition_pick(
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
        if n >= self.sb_size4 * 4 {
            self.me_scratch.est_memo.clear();
        }
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
            if quads + self.split_penalty(4.0) < whole {
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
        // A block that is already cheap is not worth splitting.
        if none < self.split_penalty(self.mode.early_none) {
            return PARTITION_NONE;
        }
        // The quarters' estimates are not negative: stop once their partial
        // sum already loses to NONE (same decision, fewer estimates).
        let pen = self.split_penalty(8.0);
        let mut split = 0.0;
        for (i, (qx, qy)) in [(x, y), (x + half, y), (x, y + half), (x + half, y + half)]
            .into_iter()
            .enumerate()
        {
            let e = self.est_block(qx, qy, half, half);
            split = if i == 0 { e } else { split + e };
            if !self.mode.rect_partitions && split + pen >= none {
                return PARTITION_NONE;
            }
        }
        let split = split + pen;
        let mut best = (PARTITION_NONE, none);
        if split < best.1 {
            best = (PARTITION_SPLIT, split);
        }
        if self.mode.rect_partitions {
            let horz = self.est_block(x, y, n, half)
                + self.est_block(x, y + half, n, half)
                + self.split_penalty(4.0);
            if horz < best.1 {
                best = (PARTITION_HORZ, horz);
            }
            let vert = self.est_block(x, y, half, n)
                + self.est_block(x + half, y, half, n)
                + self.split_penalty(4.0);
            if vert < best.1 {
                best = (PARTITION_VERT, vert);
            }
        }
        best.0
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
        let max_tx = MAX_TX_SIZE_RECT[bs as usize];
        // 64-point transforms drop all but the lowest 32x32 frequencies; start below them.
        let mut tx = max_tx;
        while !self.mode.tx64 && (tx_w(tx) > 32 || tx_h(tx) > 32) {
            tx = SPLIT_TX_SIZE[tx as usize];
        }
        // Evaluate on the first transform block's area (all of a block up to 64x64).
        let w = tx_w(tx).min(block_width(bs));
        let h = tx_h(tx).min(block_height(bs));
        let lam = self.mode.lambda_satd;

        self.me_scratch.tx_memo = None;

        // Intra luma.
        self.cur_filters = [0, 0];
        let saved = self.b.clone();
        // Inter first: an intra search far behind it skips its refinements.
        let (picked_inter, picked_rd) = if self.mode.inter_frame {
            let list = self.pick_inter(x, y, block_width(bs), block_height(bs));
            self.b = saved.clone();
            self.rerank_inter(list, &pb, x, y, w, h, tx)
        } else {
            (None, None)
        };
        let limit = |t: f64| match &picked_inter {
            Some(c) if t > 0.0 => c.cost * t,
            _ => f64::MAX,
        };
        let refine_limit = limit(self.mode.intra_refine_ratio);
        let sweep_limit = limit(self.mode.intra_sweep_ratio);
        let (best_mode, intra_cost) =
            self.pick_intra_luma(x, y, w, h, tx, refine_limit, sweep_limit);
        pb.y_mode = best_mode;
        let (angle, fi) = self.intra_choice;
        pb.angle_delta_y = angle;
        pb.use_filter_intra = fi.is_some();
        pb.filter_intra_mode = fi.unwrap_or(0);
        let mut intra_cost = intra_cost;
        let mut pal_bits = 0.0;
        let (bw, bh) = (block_width(bs), block_height(bs));
        if self.hdr.allow_screen_content_tools && bw.min(bh) >= 8 && bw.max(bh) <= 64 {
            // The mode cost was measured on the first transform block only.
            let whole = intra_cost * ((bw * bh) as f64 / (w * h) as f64);
            if let Some((n, colors, cost, pbits)) = self.pick_palette_y(x, y, bw, bh)
                && cost < whole
            {
                pal_bits = pbits;
                pb.y_mode = DC_PRED;
                pb.angle_delta_y = 0;
                pb.use_filter_intra = false;
                pb.palette_size_y = n;
                pb.palette_y = colors;
                intra_cost = cost * ((w * h) as f64 / (bw * bh) as f64);
            }
        }
        pb.tx_size = tx;
        let mut best_cost = intra_cost
            + if self.mode.inter_frame {
                lam * 6.0
            } else {
                0.0
            };
        let mut inter_choice: Option<InterChoice> = None;
        if let Some(c) = picked_inter {
            let clear = self.mode.inter_clear_ratio;
            if clear > 0.0 && best_cost > c.cost * clear {
                // Intra is far behind: no transform-coded comparison.
                inter_choice = Some(c);
            } else if self.mode.rd_inter_intra && (pb.palette_size_y == 0 || self.mode.screen_est) {
                // Transform-coded RD on the first transform block's area.
                let fw = self.fs.mi_cols * 4;
                let fh = self.fs.mi_rows * 4;
                if x + w <= fw && y + h <= fh {
                    let lam_b = self.enc.lambda;
                    let mut res = vec![0i32; w * h];
                    let pred: Vec<P> = if pb.palette_size_y > 0 {
                        let n = usize::from(pb.palette_size_y);
                        (0..h)
                            .flat_map(|i| (0..w).map(move |j| (i, j)))
                            .map(|(i, j)| {
                                let k = usize::from(self.color_map_y[i * bw + j]).min(n - 1);
                                P::from_i32(i32::from(pb.palette_y[k]))
                            })
                            .collect()
                    } else {
                        self.b.is_inter = false;
                        self.b.use_filter_intra = pb.use_filter_intra;
                        self.b.filter_intra_mode = pb.filter_intra_mode;
                        self.b.angle_delta_y = pb.angle_delta_y;
                        self.b.palette_size_y = 0;
                        let sb_mask = self.sb_size4 - 1;
                        let (sr, sc) = (self.b.mi_row & sb_mask, self.b.mi_col & sb_mask);
                        let bdf = &self.block_decoded[0];
                        let ar = bdf[sr][sc + (w >> 2) + 1];
                        let bl = bdf[sr + (h >> 2) + 1][sc];
                        let (al, au) = (self.b.avail_l, self.b.avail_u);
                        self.predict_intra(
                            0,
                            x,
                            y,
                            al,
                            au,
                            ar,
                            bl,
                            pb.y_mode,
                            w.trailing_zeros(),
                            h.trailing_zeros(),
                        );
                        let cp = &self.cur.planes[0];
                        (0..h)
                            .flat_map(|i| cp.data[(y + i) * cp.stride + x..][..w].to_vec())
                            .collect()
                    };
                    self.residual_of(&pred, w, x, y, w, h, &mut res);
                    let ib = self.mode.inter_flag_bits;
                    let mut intra_rd = self.tx_cost(&res, w, h, tx)
                        + lam_b
                            * (pal_bits
                                + if ib {
                                    self.is_inter_bits(false) + 4.0
                                } else {
                                    5.0
                                });
                    // Screen content: the chroma decides too.
                    let chroma = self.hdr.allow_screen_content_tools
                        && self.mode.screen_est
                        && pb.has_chroma
                        && w == bw
                        && h == bh;
                    let mut inter_chroma = 0.0;
                    if chroma {
                        intra_rd += self.est_chroma_sc(x, y, bw, bh, None);
                        inter_chroma = self.inter_chroma_cost(&pb, c.refs, c.mv);
                    }
                    let mut inter_rd = f64::MAX;
                    if let Some(rd) = picked_rd {
                        inter_rd = rd
                            + if ib {
                                lam_b * self.is_inter_bits(true)
                            } else {
                                0.0
                            };
                    } else if self.pred_luma(c.refs, c.mv, x, y, w, h) {
                        let pred = std::mem::take(&mut self.scratch.obmc);
                        self.residual_of(&pred, w, x, y, w, h, &mut res);
                        self.scratch.obmc = pred;
                        let new = c.mode == NEWMV || c.mode == NEW_NEWMV;
                        let bits = if new { 10.0 } else { 3.0 }
                            + if c.refs[1] > INTRA_FRAME {
                                3.0
                            } else if c.refs[0] == LAST_FRAME {
                                1.0
                            } else {
                                2.5
                            };
                        inter_rd = self.tx_cost(&res, w, h, tx) + lam_b * bits;
                    }
                    if inter_rd + inter_chroma < intra_rd {
                        inter_choice = Some(c);
                    }
                } else if c.cost < best_cost {
                    inter_choice = Some(c);
                }
            } else if c.cost < best_cost {
                best_cost = c.cost;
                inter_choice = Some(c);
            }
        }
        self.b = saved;
        let _ = best_cost;
        if let Some(c) = inter_choice {
            pb.is_inter = true;
            pb.palette_size_y = 0;
            pb.use_filter_intra = false;
            pb.ref_frame = c.refs;
            pb.y_mode = c.mode;
            pb.mv = c.mv;
            pb.ref_mv_idx = c.idx;
            pb.interp_filter = [0, 0];
            if self.hdr.interpolation_filter == 4 && self.mode.filter_search {
                pb.interp_filter = self.pick_filter(&pb);
            }
            self.cur_filters = pb.interp_filter;
            if (self.mode.local_warp || self.mode.obmc) && c.refs[1] <= INTRA_FRAME {
                pb.motion_mode = self.pick_motion_mode(&pb);
            }
            pb.uv_mode = DC_PRED;
            pb.tx_size = self.pick_inter_tx(&pb, tx);
        } else {
            if pb.palette_size_y == 0 {
                pb.tx_size = self.pick_intra_tx(&pb, tx, max_tx);
            }
            let (uv_mode, uv_cost) = if pb.has_chroma {
                self.pick_intra_chroma(&pb)
            } else {
                (DC_PRED, f64::MAX)
            };
            pb.uv_mode = uv_mode;
            if pb.uv_mode == UV_CFL_PRED {
                (pb.cfl_alpha_u, pb.cfl_alpha_v) = self.cfl_choice;
            }
            if self.hdr.allow_screen_content_tools
                && pb.has_chroma
                && bw.min(bh) >= 8
                && bw.max(bh) <= 64
                && let Some((n, u, v, cost)) = self.pick_palette_uv(&pb)
                && cost < uv_cost
            {
                pb.uv_mode = DC_PRED;
                pb.palette_size_uv = n;
                pb.palette_u = u;
                pb.palette_v = v;
            }
        }
        // Skip when the prediction is already close to the source.
        pb.skip = self.block_is_skip(&pb);
        if self.hdr.skip_mode_present && self.mode.skip_mode {
            self.try_skip_mode(&mut pb);
        }
        self.pb = pb;
        self.plan_q = self.plan.sb_target.unwrap_or(self.current_q_index);
        self.plan_delta_lf = self.delta_lf;
    }

    /// Transform-coded cost of the chroma residual of the block `pb` predicted
    /// by `refs` / `mv` (0 when a prediction is not available).
    fn inter_chroma_cost(&mut self, pb: &Block, refs: [i8; 2], mv: [Mv; 2]) -> f64 {
        let Some(src) = self.src else { return 0.0 };
        let (sx, sy) = (usize::from(self.sub_x), usize::from(self.sub_y));
        let cx = (pb.mi_col * 4) >> sx;
        let cy = (pb.mi_row * 4) >> sy;
        let cw = (block_width(pb.bsize) >> sx).max(4);
        let ch = (block_height(pb.bsize) >> sy).max(4);
        let uv_tx = self.get_tx_size_for(1, 0, pb.bsize);
        if !cw.is_multiple_of(tx_w(uv_tx)) || !ch.is_multiple_of(tx_h(uv_tx)) {
            return 0.0;
        }
        let mut res = vec![0i32; cw * ch];
        let mut cost = 0.0;
        for plane in 1..3 {
            let sp = &src.planes[plane];
            if cx + cw > sp.w || cy + ch > sp.h {
                return 0.0;
            }
            if !self.pred_plane_f(plane, refs, mv, cx, cy, cw, ch, [0, 0]) {
                return 0.0;
            }
            for i in 0..ch {
                for j in 0..cw {
                    res[i * cw + j] = sp.data[(cy + i) * sp.stride + cx + j].to_i32()
                        - self.scratch.obmc[i * cw + j].to_i32();
                }
            }
            cost += self.tx_cost(&res, cw, ch, uv_tx);
        }
        cost
    }

    /// [`Self::tx_cost`] for the other mode decision modules.
    pub(crate) fn tx_cost_any(&mut self, res: &[i32], w: usize, h: usize, tx: u8) -> f64 {
        self.tx_cost(res, w, h, tx)
    }

    /// Estimated D + lambda * R of coding residual `res` (w x h, stride w) with
    /// transform blocks of size `tx` (DCT, dead-zone quantization).
    fn tx_cost(&mut self, res: &[i32], w: usize, h: usize, tx: u8) -> f64 {
        let tw = tx_w(tx);
        let th = tx_h(tx);
        let qi = self.dequant_info(0, 0, tx);
        let bd = self.bd;
        let lam = self.enc.lambda;
        let mut d = 0f64;
        let mut bits = 0f64;
        let mid = P::from_i32(1 << (bd - 1));
        let mut coefs = take_buf(&mut self.scratch.enc_i32[1], 64 * 64, 0);
        let mut rec = take_buf(&mut self.scratch.enc_px[0], 64 * 64, mid);
        let fast = self.mode.fast_tx_dist;
        // Squared gain from the transform domain to samples (see quant.rs).
        let gain_log2 = ((crate::dec::types::tx_w_log2(tx) + crate::dec::types::tx_h_log2(tx)) / 2)
            as i32
            - 1
            - i32::from(crate::spec::tables::TRANSFORM_ROW_SHIFT[tx as usize])
            - 4;
        let g2 = 2f64.powi(2 * gain_log2);
        let qdc = qi.dc.max(1);
        let qac = qi.ac.max(1);
        let dlog = qi.denom.trailing_zeros();
        let mut y0 = 0;
        while y0 < h {
            let mut x0 = 0;
            while x0 < w {
                let rb = &res[y0 * w + x0..];
                fwd_txfm(rb, w, &mut coefs, tx as usize, 0, bd);
                let cw = tw.min(32);
                let ch = th.min(32);
                if fast {
                    let dc = quant_est(&coefs[..1], qdc, dlog);
                    let ac = quant_est(&coefs[1..cw * ch], qac, dlog);
                    bits += f64::from(dc.bits + ac.bits) + 1.0;
                    // Transform-domain distortion; what a 64-point transform
                    // drops is the residual energy it does not keep.
                    let mut e = 0u64;
                    for i in 0..th {
                        for &r in &rb[i * w..i * w + tw] {
                            e += u64::from(r.unsigned_abs() * r.unsigned_abs());
                        }
                    }
                    let kept = (dc.kept + ac.kept) as f64;
                    let qerr = (dc.qerr + ac.qerr) as f64;
                    d += qerr * g2 + (e as f64 - kept * g2).max(0.0);
                    x0 += tw;
                    continue;
                }
                let mut nz = (0, 0);
                for i in 0..ch {
                    for j in 0..cw {
                        let pos = i * cw + j;
                        let q = if pos == 0 { qdc } else { qac };
                        let c = coefs[pos];
                        let l = quant_level(c, q, dlog);
                        if l != 0 {
                            bits += 2.0 + 2.0 * f64::from(32 - l.leading_zeros());
                            nz = (nz.0.max(i + 1), nz.1.max(j + 1));
                        }
                        let v = ((l * q as u32) >> dlog) as i32;
                        coefs[pos] = if c < 0 { -v } else { v };
                    }
                }
                bits += 1.0;
                for v in rec[..tw * th].iter_mut() {
                    *v = mid;
                }
                if nz.0 > 0 {
                    inv_txfm_add(
                        &mut rec,
                        tw,
                        &mut coefs,
                        tx as usize,
                        0,
                        nz.0,
                        nz.1,
                        false,
                        bd,
                    );
                } else {
                    for v in coefs[..cw * ch].iter_mut() {
                        *v = 0;
                    }
                }
                for i in 0..th {
                    for j in 0..tw {
                        let e = res[(y0 + i) * w + x0 + j]
                            - (rec[i * tw + j].to_i32() - (1 << (bd - 1)));
                        d += f64::from(e * e);
                    }
                }
                x0 += tw;
            }
            y0 += th;
        }
        self.scratch.enc_i32[1] = coefs;
        self.scratch.enc_px[0] = rec;
        d + lam * bits
    }

    fn residual_of(
        &self,
        pred: &[P],
        pred_stride: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        out: &mut [i32],
    ) {
        let Some(src) = self.src else { return };
        let sp = &src.planes[0];
        for i in 0..h {
            for j in 0..w {
                out[i * w + j] = sp.data[(y + i) * sp.stride + x + j].to_i32()
                    - pred[i * pred_stride + j].to_i32();
            }
        }
    }

    fn pick_inter_tx(&mut self, pb: &Block, tx: u8) -> u8 {
        let x = pb.mi_col * 4;
        let y = pb.mi_row * 4;
        let w = block_width(pb.bsize).min(self.fs.mi_cols * 4 - x);
        let h = block_height(pb.bsize).min(self.fs.mi_rows * 4 - y);
        let (w, h) = (w & !(tx_w(tx) - 1), h & !(tx_h(tx) - 1));
        if w == 0 || h == 0 || !self.pred_luma(pb.ref_frame, pb.mv, x, y, w, h) {
            return tx;
        }
        let mut res = vec![0i32; w * h];
        let pred = std::mem::take(&mut self.scratch.obmc);
        self.residual_of(&pred, w, x, y, w, h, &mut res);
        self.scratch.obmc = pred;
        let key = self.pred_key(0, pb.ref_frame, pb.mv, [x, y, w, h]);
        let whole = match self.me_scratch.tx_memo {
            Some((k, t, c)) if k == key && t == tx => c,
            _ => self.tx_cost(&res, w, h, tx),
        };
        let mut best = (tx, whole);
        let mut t = tx;
        for _ in 0..self.mode.tx_depth {
            let s = SPLIT_TX_SIZE[t as usize];
            if s == t {
                break;
            }
            t = s;
            let c = self.tx_cost(&res, w, h, t);
            if c < best.1 {
                best = (t, c);
            }
        }
        self.me_scratch.tx_memo = Some((key, best.0, best.1));
        best.0
    }

    /// The cache key of a prediction with the current filters.
    fn pred_key(&self, plane: usize, refs: [i8; 2], mv: [Mv; 2], area: [usize; 4]) -> PredKey {
        PredKey {
            plane,
            refs,
            mv,
            filters: self.cur_filters,
            area,
            intrabc: self.b.use_intrabc,
        }
    }

    fn pick_intra_tx(&mut self, pb: &Block, tx: u8, max_tx: u8) -> u8 {
        // The transform size must be reachable from the largest one in two splits.
        let mut depth0 = 0;
        let mut t = max_tx;
        while t != tx {
            t = SPLIT_TX_SIZE[t as usize];
            depth0 += 1;
        }
        if self.mode.tx_depth == 0 || depth0 >= 2 || pb.bsize == crate::dec::types::BLOCK_4X4 {
            return tx;
        }
        let x = pb.mi_col * 4;
        let y = pb.mi_row * 4;
        let w = tx_w(tx);
        let h = tx_h(tx);
        if x + w > self.fs.mi_cols * 4 || y + h > self.fs.mi_rows * 4 {
            return tx;
        }
        // Prediction of the first transform block with the chosen mode.
        let sb_mask = self.sb_size4 - 1;
        let bdf = &self.block_decoded[0];
        let (sr, sc) = (pb.mi_row & sb_mask, pb.mi_col & sb_mask);
        let ar = bdf[sr][sc + (w >> 2) + 1];
        let bl = bdf[sr + (h >> 2) + 1][sc];
        let saved = self.b.clone();
        self.b.use_filter_intra = pb.use_filter_intra;
        self.b.filter_intra_mode = pb.filter_intra_mode;
        self.b.angle_delta_y = pb.angle_delta_y;
        self.predict_intra(
            0,
            x,
            y,
            pb.avail_l,
            pb.avail_u,
            ar,
            bl,
            pb.y_mode,
            w.trailing_zeros(),
            h.trailing_zeros(),
        );
        self.b = saved;
        let cp = &self.cur.planes[0];
        let mut res = vec![0i32; w * h];
        let pred: Vec<P> = (0..h)
            .flat_map(|i| cp.data[(y + i) * cp.stride + x..][..w].to_vec())
            .collect();
        self.residual_of(&pred, w, x, y, w, h, &mut res);
        let whole = self.tx_cost(&res, w, h, tx);
        let s = SPLIT_TX_SIZE[tx as usize];
        if s == tx {
            return tx;
        }
        let split = if self.mode.intra_split_pred {
            self.intra_split_cost(pb, x, y, w, h, s)
        } else {
            self.tx_cost(&res, w, h, s)
        };
        if split < whole { s } else { tx }
    }

    /// Cost of the `w` x `h` area at (`x`, `y`) coded as transform blocks of
    /// size `s`, each predicted on its own with the block's intra mode. The
    /// earlier transform blocks stand in with their source samples.
    fn intra_split_cost(
        &mut self,
        pb: &Block,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        s: u8,
    ) -> f64 {
        let Some(src) = self.src else { return f64::MAX };
        let (sw, sh) = (tx_w(s), tx_h(s));
        let sb_mask = self.sb_size4 - 1;
        let (sr, sc) = (pb.mi_row & sb_mask, pb.mi_col & sb_mask);
        let (nr, nc) = (h / sh, w / sw);
        let saved = self.b.clone();
        self.b.use_filter_intra = pb.use_filter_intra;
        self.b.filter_intra_mode = pb.filter_intra_mode;
        self.b.angle_delta_y = pb.angle_delta_y;
        let mut res = vec![0i32; sw * sh];
        let mut total = 0.0;
        for i in 0..nr {
            for j in 0..nc {
                let (px, py) = (x + j * sw, y + i * sh);
                // Decoded state of the 4x4 at (r4, c4) relative to the superblock.
                let decoded = |r4: isize, c4: isize, bdf: &[[bool; 35]; 35]| -> bool {
                    let (ra, ca) = (sr as isize, sc as isize);
                    let inside = r4 >= ra
                        && r4 < ra + (h / 4) as isize
                        && c4 >= ca
                        && c4 < ca + (w / 4) as isize;
                    if inside {
                        let ti = ((r4 - ra) as usize * 4) / sh;
                        let tj = ((c4 - ca) as usize * 4) / sw;
                        ti < i || (ti == i && tj < j)
                    } else if r4 + 1 < 0 || c4 + 1 < 0 || r4 + 1 >= 35 || c4 + 1 >= 35 {
                        false
                    } else {
                        bdf[(r4 + 1) as usize][(c4 + 1) as usize]
                    }
                };
                let bdf = &self.block_decoded[0];
                let r0 = (sr + i * sh / 4) as isize;
                let c0 = (sc + j * sw / 4) as isize;
                let ar = decoded(r0 - 1, c0 + (sw / 4) as isize, bdf);
                let bl = decoded(r0 + (sh / 4) as isize, c0 - 1, bdf);
                let al = j > 0 || pb.avail_l;
                let au = i > 0 || pb.avail_u;
                self.predict_intra(
                    0,
                    px,
                    py,
                    al,
                    au,
                    ar,
                    bl,
                    pb.y_mode,
                    sw.trailing_zeros(),
                    sh.trailing_zeros(),
                );
                let sp = &src.planes[0];
                let cp = &mut self.cur.planes[0];
                for r in 0..sh {
                    for c in 0..sw {
                        let o = (py + r) * cp.stride + px + c;
                        let sv = sp.data[(py + r) * sp.stride + px + c];
                        res[r * sw + c] = sv.to_i32() - cp.data[o].to_i32();
                        cp.data[o] = sv;
                    }
                }
                total += self.tx_cost(&res, sw, sh, s);
            }
        }
        self.b = saved;
        total
    }

    /// Best luma intra mode by SATD (refined by transform RD, angle deltas and
    /// filter intra; the last two only while the cost is below `refine_limit`).
    fn pick_intra_luma(
        &mut self,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        tx: u8,
        refine_limit: f64,
        sweep_limit: f64,
    ) -> (u8, f64) {
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
        let mut cands: Vec<(f64, u8, f64)> = Vec::with_capacity(13);
        let n_modes = self.mode.intra_modes.min(13);
        // With a sweep limit the non-directional modes, V and H go first and
        // the other directions only when one of them is below the limit.
        let order: [u8; 13] = if sweep_limit < f64::MAX {
            [0, 1, 2, 9, 12, 3, 4, 5, 6, 7, 8, 10, 11]
        } else {
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        };
        for (i, &mode) in order.iter().enumerate() {
            if mode >= n_modes {
                continue;
            }
            if i == 5 && best.1 > sweep_limit {
                break;
            }
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
            let rate = if self.mode.exact_intra_rates {
                self.y_mode_bits(mode)
            } else {
                match mode {
                    DC_PRED => 1.0,
                    SMOOTH_PRED | PAETH_PRED => 2.5,
                    _ => 3.5,
                }
            };
            let cost = f64::from(d) + lam * rate;
            cands.push((cost, mode, rate));
            if cost < best.1 {
                best = (mode, cost);
            }
        }
        // Transform-coded RD among the best few SATD candidates.
        let k = usize::from(self.mode.intra_rd_cands);
        if k > 1 && w <= 64 && h <= 64 {
            cands.sort_by(|a, b| a.0.total_cmp(&b.0));
            let lam_b = self.enc.lambda;
            let mut best_rd = f64::MAX;
            let mut res = vec![0i32; w * h];
            for &(sc, mode, rate) in cands.iter().take(k) {
                self.predict_intra(0, x, y, al, au, have_ar, have_bl, mode, lw, lh);
                let cp = &self.cur.planes[0];
                let sp = &src.planes[0];
                for i in 0..h {
                    for j in 0..w {
                        res[i * w + j] = sp.data[(y + i) * sp.stride + x + j].to_i32()
                            - cp.data[(y + i) * cp.stride + x + j].to_i32();
                    }
                }
                let rd = self.tx_cost(&res, w, h, tx) + lam_b * rate;
                if rd < best_rd {
                    best_rd = rd;
                    best = (mode, sc);
                }
            }
        }
        let sat = |s: &Self| -> f64 {
            let cp = &s.cur.planes[0];
            let sp = &src.planes[0];
            f64::from(satd(
                &sp.data[y * sp.stride + x..],
                sp.stride,
                &cp.data[y * cp.stride + x..],
                cp.stride,
                w,
                h,
            ))
        };
        let bs = self.b.bsize;
        let mut angle = 0i8;
        let mut fi: Option<u8> = None;
        // Angle refinements of a directional winner.
        let refine = best.1 <= refine_limit;
        if refine && self.mode.angle_deltas && (1..=8).contains(&best.0) && bs >= BLOCK_8X8 {
            let base = best.0;
            let base_bits = self.y_mode_bits(base) - self.angle_bits(base, 0);
            for d in [-3i8, -2, -1, 1, 2, 3] {
                self.b.angle_delta_y = d;
                self.predict_intra(0, x, y, al, au, have_ar, have_bl, base, lw, lh);
                let r = if self.mode.exact_intra_rates {
                    base_bits + self.angle_bits(base, d)
                } else {
                    3.5 + 1.0 + 0.5 * f64::from(d.unsigned_abs())
                };
                let cost = sat(self) + lam * r;
                if cost < best.1 {
                    best = (base, cost);
                    angle = d;
                }
            }
            self.b.angle_delta_y = 0;
        }
        // Recursive filter intra (signalled as DC_PRED).
        if refine
            && self.mode.filter_intra
            && self.seq.enable_filter_intra
            && block_width(bs).max(block_height(bs)) <= 32
        {
            self.b.use_filter_intra = true;
            let dc_bits = self.y_mode_bits(DC_PRED) - self.filter_intra_bits(None);
            for m in 0..5u8 {
                self.b.filter_intra_mode = m;
                self.predict_intra(0, x, y, al, au, have_ar, have_bl, DC_PRED, lw, lh);
                let r = if self.mode.exact_intra_rates {
                    dc_bits + self.filter_intra_bits(Some(m))
                } else {
                    4.5
                };
                let cost = sat(self) + lam * r;
                if cost < best.1 {
                    best = (DC_PRED, cost);
                    angle = 0;
                    fi = Some(m);
                }
            }
            self.b.use_filter_intra = false;
        }
        self.intra_choice = (angle, fi);
        best
    }

    fn pick_intra_chroma(&mut self, pb: &Block) -> (u8, f64) {
        let Some(src) = self.src else {
            return (DC_PRED, f64::MAX);
        };
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
            let r = if self.mode.exact_intra_rates {
                self.uv_mode_bits(pb.y_mode, mode)
            } else if mode == DC_PRED {
                1.0
            } else {
                3.0
            };
            let cost = f64::from(d) + lam * r;
            if cost < best.1 {
                best = (mode, cost);
            }
        }
        // Chroma from luma, with the scaling fitted against the source luma.
        let bs = pb.bsize;
        let cfl_allowed = !self.b.lossless && block_width(bs).max(block_height(bs)) <= 32;
        let mut cfl = (0i8, 0i8);
        if cfl_allowed && self.mode.cfl {
            let sp0 = &src.planes[0];
            let lx = x << sx;
            let ly = y << sy;
            let fw = self.fs.mi_cols * 4;
            let fh = self.fs.mi_rows * 4;
            let vw = ((fw.saturating_sub(lx) + (1 << sx) - 1) >> sx).clamp(1, w);
            let vh = ((fh.saturating_sub(ly) + (1 << sy) - 1) >> sy).clamp(1, h);
            let mut ac = [0i32; 32 * 32];
            cfl_ac(
                &mut ac,
                &sp0.data[ly * sp0.stride + lx..],
                sp0.stride,
                w,
                h,
                sx != 0,
                sy != 0,
                vw,
                vh,
            );
            let mut d = 0u32;
            let mut alphas = [0i8; 2];
            for plane in 1..3 {
                self.predict_intra(
                    plane,
                    x,
                    y,
                    al,
                    au,
                    false,
                    false,
                    DC_PRED,
                    w.trailing_zeros(),
                    h.trailing_zeros(),
                );
                let sp = &src.planes[plane];
                let cp = &self.cur.planes[plane];
                let mut dc = [P::default(); 32 * 32];
                let mut num = 0f64;
                let mut den = 0f64;
                for i in 0..h {
                    for j in 0..w {
                        let p = cp.data[(y + i) * cp.stride + x + j];
                        dc[i * w + j] = p;
                        let r = sp.data[(y + i) * sp.stride + x + j].to_i32() - p.to_i32();
                        let a = ac[i * w + j];
                        num += f64::from(a) * f64::from(r);
                        den += f64::from(a) * f64::from(a);
                    }
                }
                let opt = if den > 0.0 {
                    (64.0 * num / den).round()
                } else {
                    0.0
                };
                let a0 = opt.clamp(-16.0, 16.0) as i32;
                let mut bp = (u32::MAX, 0i32);
                for a in [a0 - 1, a0, a0 + 1] {
                    if !(-16..=16).contains(&a) {
                        continue;
                    }
                    let mut pr = dc;
                    cfl_pred(&mut pr, w, w, h, &ac, a, self.bd);
                    let dd = satd(&sp.data[y * sp.stride + x..], sp.stride, &pr, w, w, h)
                        + (lam
                            * if a == 0 {
                                0.5
                            } else {
                                2.0 + 0.3 * f64::from(a.abs())
                            }) as u32;
                    if dd < bp.0 {
                        bp = (dd, a);
                    }
                }
                d = d.saturating_add(bp.0);
                alphas[plane - 1] = bp.1 as i8;
            }
            if alphas != [0, 0] {
                let cost = if self.mode.exact_intra_rates {
                    // Replace the per-plane alpha guesses with the exact cost.
                    let guess: f64 = alphas
                        .iter()
                        .map(|&a| {
                            if a == 0 {
                                0.5
                            } else {
                                2.0 + 0.3 * f64::from(a.unsigned_abs())
                            }
                        })
                        .sum();
                    f64::from(d)
                        + lam
                            * (self.uv_mode_bits(pb.y_mode, UV_CFL_PRED)
                                + self.cfl_bits(alphas[0], alphas[1])
                                - guess)
                } else {
                    f64::from(d) + lam * 3.0
                };
                if cost < best.1 {
                    best = (UV_CFL_PRED, cost);
                    cfl = (alphas[0], alphas[1]);
                }
            }
        }
        self.b = saved;
        self.cfl_choice = cfl;
        best
    }

    fn get_tx_size_for(&self, plane: usize, tx: u8, bsize: u8) -> u8 {
        let _ = plane;
        // Chroma transform size as get_tx_size computes it for `bsize`.
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

    /// Returns (mode, mv, ref_mv_idx, cost) of the best single-reference mode
    /// from `rf`, and the motion search result.
    fn pick_inter_ref(
        &mut self,
        rf: i8,
        x: usize,
        y: usize,
        bw: usize,
        bh: usize,
        out: &mut Vec<ModeOpt>,
    ) -> Option<(ModeOpt, Mv)> {
        let src = self.src?;
        self.refs[rf as usize].as_ref()?;
        let fw = self.fs.mi_cols * 4;
        let fh = self.fs.mi_rows * 4;
        let w = bw.min(fw - x);
        let h = bh.min(fh - y);
        self.b.ref_frame = [rf, NONE];
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
        let mut cands: Vec<(i32, i32)> = stack
            .iter()
            .map(|m| (i32::from(m.row), i32::from(m.col)))
            .collect();
        // The partition estimate searched LAST for this area.
        if rf == LAST_FRAME
            && let Some(e) = self
                .me_scratch
                .est_memo
                .iter()
                .find(|e| e.0 == [x, y, bw, bh])
        {
            cands.push(e.2);
        }
        let pred0 = stack[0];
        let params = MeParams {
            // Secondary references only refine around their predicted vectors.
            max_range: if rf == LAST_FRAME || rf == BWDREF_FRAME {
                self.mode.me_range
            } else {
                self.mode.me_range / 4
            },
            subpel: if hp {
                self.mode.subpel
            } else {
                self.mode.subpel.min(2)
            },
            filter: self.mode.me_filter,
            subpel_pruned: self.mode.subpel_pruned,
            ..MeParams::default()
        };
        let cost_fn = |r: i32, c: i32| -> u32 {
            let dr = r - i32::from(pred0.row);
            let dc = c - i32::from(pred0.col);
            (lam * (3.0 + bits(dr) + bits(dc))) as u32
        };
        let me = {
            let rv = self.refs[rf as usize].as_ref()?;
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
        let dr = i32::from(mv_new.row) - i32::from(pred0.row);
        let dc = i32::from(mv_new.col) - i32::from(pred0.col);
        if self.mode.exact_rates {
            options.push((NEARESTMV, stack[0], 0, self.mode_bits(false, NEARESTMV, 0)));
            if self.b.num_mv_found > 1 {
                options.push((NEARMV, stack[1], 1, self.mode_bits(false, NEARMV, 1)));
            }
            options.push((GLOBALMV, global, 0, self.mode_bits(false, GLOBALMV, 0)));
            let r = self.mode_bits(false, NEWMV, 0) + self.mv_bits((dr, dc));
            options.push((NEWMV, mv_new, 0, r));
        } else {
            options.push((NEARESTMV, stack[0], 0, 1.0));
            if self.b.num_mv_found > 1 {
                options.push((NEARMV, stack[1], 1, 2.5));
            }
            options.push((GLOBALMV, global, 0, 2.0));
            options.push((NEWMV, mv_new, 0, 3.0 + bits(dr) + bits(dc)));
        }
        let mut best: Option<(u8, Mv, usize, f64)> = None;
        let mut best_rate = 0.0;
        for (mode, mv, idx, rate) in options {
            if best.is_some_and(|b| b.1 == mv && b.0 != NEWMV) {
                continue;
            }
            let warped = mode == GLOBALMV && self.predict_global_warp_into_scratch(rf, x, y, w, h);
            if !warped && !self.pred_plane_f(0, [rf, NONE], [mv, Mv::ZERO], x, y, w, h, [0, 0]) {
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
                best_rate = rate;
            }
            out.push((mode, mv, idx, cost, rate));
        }
        out.sort_by(|a, b| a.3.total_cmp(&b.3));
        best.map(|b| ((b.0, b.1, b.2, b.3, best_rate), mv_new))
    }

    /// Luma prediction of `refs`/`mv` into `scratch.obmc` (stride `w`);
    /// compound predictions are averaged.
    fn pred_luma(
        &mut self,
        refs: [i8; 2],
        mv: [Mv; 2],
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    ) -> bool {
        self.pred_plane(0, refs, mv, x, y, w, h)
    }

    /// Prediction of one plane (position and size in that plane's samples).
    fn pred_plane(
        &mut self,
        plane: usize,
        refs: [i8; 2],
        mv: [Mv; 2],
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    ) -> bool {
        let f = self.cur_filters;
        self.pred_plane_f(plane, refs, mv, x, y, w, h, f)
    }

    fn pred_plane_f(
        &mut self,
        plane: usize,
        refs: [i8; 2],
        mv: [Mv; 2],
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        filters: [u8; 2],
    ) -> bool {
        let n = w * h;
        let key = PredKey {
            plane,
            refs,
            mv,
            filters,
            area: [x, y, w, h],
            intrabc: self.b.use_intrabc,
        };
        let cache = &mut self.me_scratch.pred_cache;
        if let Some(e) = cache.iter().find(|e| e.0 == key) {
            self.scratch.obmc[..n].copy_from_slice(&e.1[..n]);
            return true;
        }
        if self
            .predict_into_scratch(plane, refs[0], x, y, w, h, mv[0], filters)
            .is_err()
        {
            return false;
        }
        if refs[1] > INTRA_FRAME {
            let first: Vec<P> = self.scratch.obmc[..n].to_vec();
            if self
                .predict_into_scratch(plane, refs[1], x, y, w, h, mv[1], filters)
                .is_err()
            {
                return false;
            }
            for (d, s) in self.scratch.obmc[..n].iter_mut().zip(&first) {
                *d = P::from_i32((d.to_i32() + s.to_i32() + 1) >> 1);
            }
        }
        // Keep the last few predictions; the oldest buffer is reused.
        let cache = &mut self.me_scratch.pred_cache;
        let mut buf = if cache.len() >= PRED_CACHE {
            cache.remove(0).1
        } else {
            Vec::new()
        };
        buf.clear();
        buf.extend_from_slice(&self.scratch.obmc[..n]);
        cache.push((key, buf));
        true
    }

    /// Skip mode: compound NEAREST from the two skip-mode references with no
    /// residual, signalled by one flag. Taken when its prediction is about as
    /// good as the chosen mode's (when that is skipped too) or simply close
    /// enough to the source.
    fn try_skip_mode(&mut self, pb: &mut Block) {
        let Some(src) = self.src else { return };
        let bs = pb.bsize;
        let (bw, bh) = (block_width(bs), block_height(bs));
        let x = pb.mi_col * 4;
        let y = pb.mi_row * 4;
        if bw < 8 || bh < 8 || x + bw > self.fs.mi_cols * 4 || y + bh > self.fs.mi_rows * 4 {
            return;
        }
        let refs = [
            self.hdr.skip_mode_frame[0] as i8,
            self.hdr.skip_mode_frame[1] as i8,
        ];
        let saved = self.b.clone();
        self.b.ref_frame = refs;
        self.b.is_inter = true;
        self.b.use_intrabc = false;
        self.find_mv_stack(true);
        let mv = self.b.ref_stack_mv[0];
        self.b = saved;
        // Only a block already coded without residual is a candidate: skip
        // mode then just trades its mode bits for a different prediction.
        if !(pb.is_inter && pb.skip) || pb.motion_mode != 0 {
            return;
        }
        let planes = if self.num_planes > 1 && pb.has_chroma {
            3
        } else {
            1
        };
        let mut sse_sm = 0f64;
        let mut sse_cur = 0f64;
        let (cur_refs, cur_mv, cur_f) = (pb.ref_frame, pb.mv, pb.interp_filter);
        for plane in 0..planes {
            let (sx, sy) = if plane == 0 {
                (0, 0)
            } else {
                (usize::from(self.sub_x), usize::from(self.sub_y))
            };
            let (px, py, pw, ph) = (
                (pb.mi_col >> sx) * 4,
                (pb.mi_row >> sy) * 4,
                bw >> sx,
                bh >> sy,
            );
            let sp = &src.planes[plane];
            if !self.pred_plane_f(plane, refs, mv, px, py, pw, ph, [0, 0]) {
                return;
            }
            sse_sm += sse(
                &sp.data[py * sp.stride + px..],
                sp.stride,
                &self.scratch.obmc,
                pw,
                pw,
                ph,
            ) as f64;
            if !self.pred_plane_f(plane, cur_refs, cur_mv, px, py, pw, ph, cur_f) {
                return;
            }
            sse_cur += sse(
                &sp.data[py * sp.stride + px..],
                sp.stride,
                &self.scratch.obmc,
                pw,
                pw,
                ph,
            ) as f64;
        }
        // Bits a skipped inter block spends on its mode, references and vectors.
        let saved_bits = self.mode.skip_mode_sse;
        if sse_sm <= sse_cur + self.enc.lambda * saved_bits {
            pb.skip_mode = true;
            pb.skip = true;
            pb.is_inter = true;
            pb.ref_frame = refs;
            pb.y_mode = NEAREST_NEARESTMV;
            pb.mv = mv;
            pb.ref_mv_idx = 0;
            pb.interp_filter = [0, 0];
            pb.motion_mode = 0;
            pb.compound_idx = 1;
            pb.palette_size_y = 0;
            pb.use_filter_intra = false;
            pb.uv_mode = DC_PRED;
        }
    }

    /// SIMPLE or LOCALWARP, whichever predicts the luma with less SATD.
    fn pick_motion_mode(&mut self, pb: &Block) -> u8 {
        let Some(src) = self.src else { return 0 };
        let x = pb.mi_col * 4;
        let y = pb.mi_row * 4;
        let w = block_width(pb.bsize);
        let h = block_height(pb.bsize);
        if x + w > self.fs.mi_cols * 4 || y + h > self.fs.mi_rows * 4 {
            return 0;
        }
        if !self.pred_plane(0, pb.ref_frame, pb.mv, x, y, w, h) {
            return 0;
        }
        let sp = &src.planes[0];
        let simple = f64::from(satd(
            &sp.data[y * sp.stride + x..],
            sp.stride,
            &self.scratch.obmc,
            w,
            w,
            h,
        ));
        let saved = self.b.clone();
        self.b.is_inter = true;
        self.b.ref_frame = pb.ref_frame;
        self.b.mv = pb.mv;
        self.b.y_mode = pb.y_mode;
        self.b.skip_mode = false;
        self.b.interp_filter = pb.interp_filter;
        let lam = self.mode.lambda_satd;
        let mut best = (0u8, simple);
        if self.mode.local_warp && self.predict_local_warp_into_scratch(x, y, w, h) {
            let c = f64::from(satd(
                &sp.data[y * sp.stride + x..],
                sp.stride,
                &self.scratch.obmc,
                w,
                w,
                h,
            )) + lam * 1.5;
            if c < best.1 {
                best = (crate::dec::types::LOCALWARP, c);
            }
        }
        if self.mode.obmc && self.predict_obmc_into_scratch(x, y, w, h) {
            let c = f64::from(satd(
                &sp.data[y * sp.stride + x..],
                sp.stride,
                &self.scratch.obmc,
                w,
                w,
                h,
            )) + lam * 1.5;
            if c < best.1 {
                best = (crate::dec::types::OBMC, c);
            }
        }
        self.b = saved;
        best.0
    }

    /// Interpolation filter (same both directions) with the least SATD.
    fn pick_filter(&mut self, pb: &Block) -> [u8; 2] {
        let Some(src) = self.src else { return [0, 0] };
        let x = pb.mi_col * 4;
        let y = pb.mi_row * 4;
        let w = block_width(pb.bsize).min(self.fs.mi_cols * 4 - x);
        let h = block_height(pb.bsize).min(self.fs.mi_rows * 4 - y);
        let (w, h) = (w & !3, h & !3);
        if w == 0 || h == 0 {
            return [0, 0];
        }
        let lam = self.mode.lambda_satd;
        let mut best = ([0u8, 0u8], f64::MAX);
        let dual = self.seq.enable_dual_filter;
        let mut combos: Vec<([u8; 2], f64)> = Vec::with_capacity(9);
        for fy in 0..3u8 {
            for fx in 0..3u8 {
                if fy != fx && !dual {
                    continue;
                }
                let bits = if fy == 0 && fx == 0 {
                    0.6
                } else if fy == fx {
                    2.0
                } else {
                    3.5
                };
                combos.push(([fy, fx], bits));
            }
        }
        for (f, bits) in combos {
            if !self.pred_plane_f(0, pb.ref_frame, pb.mv, x, y, w, h, f) {
                continue;
            }
            let sp = &src.planes[0];
            let d = f64::from(satd(
                &sp.data[y * sp.stride + x..],
                sp.stride,
                &self.scratch.obmc,
                w,
                w,
                h,
            ));
            let c = d + lam * bits;
            if c < best.1 {
                best = (f, c);
            }
        }
        best.0
    }

    /// The best inter choice over every distinct reference and, when the
    /// frame allows it, forward/backward compound pairs.
    /// The best of the inter candidates (sorted by SATD cost) after a
    /// transform-coded RD comparison of the first few on the first
    /// transform block's area, and its RD cost when it was measured.
    fn rerank_inter(
        &mut self,
        list: Vec<InterChoice>,
        pb: &Block,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        tx: u8,
    ) -> (Option<InterChoice>, Option<f64>) {
        let k = self.mode.inter_rd_cands;
        let first = list.first().copied();
        let fw = self.fs.mi_cols * 4;
        let fh = self.fs.mi_rows * 4;
        if k < 2 || list.len() < 2 || x + w > fw || y + h > fh {
            return (first, None);
        }
        let lam_b = self.enc.lambda;
        let mut res = vec![0i32; w * h];
        let mut best: Option<(InterChoice, f64, f64, f64)> = None;
        let lim = first.map_or(f64::MAX, |c| c.cost * self.mode.inter_rd_ratio);
        let chroma = self.hdr.allow_screen_content_tools && self.mode.screen_est && pb.has_chroma;
        for c in list.iter().take(k) {
            if c.cost > lim {
                break;
            }
            if !self.pred_luma(c.refs, c.mv, x, y, w, h) {
                continue;
            }
            let pred = std::mem::take(&mut self.scratch.obmc);
            self.residual_of(&pred, w, x, y, w, h, &mut res);
            self.scratch.obmc = pred;
            let e: f64 = res.iter().map(|&v| f64::from(v * v)).sum();
            let coded = self.tx_cost(&res, w, h, tx);
            let rd = coded.min(e + lam_b) + lam_b * c.bits;
            let rank = if chroma {
                rd + self.inter_chroma_cost(pb, c.refs, c.mv)
            } else {
                rd
            };
            if best.as_ref().is_none_or(|b| rank < b.3) {
                best = Some((*c, rd, coded, rank));
            }
        }
        match best {
            Some((c, rd, coded, _)) => {
                // The transform size search starts from this cost.
                let key = self.pred_key(0, c.refs, c.mv, [x, y, w, h]);
                self.me_scratch.tx_memo = Some((key, tx, coded));
                (Some(c), Some(rd))
            }
            None => (first, None),
        }
    }

    fn pick_inter(&mut self, x: usize, y: usize, bw: usize, bh: usize) -> Vec<InterChoice> {
        let mut all = Vec::with_capacity(8);
        self.pick_inter_into(x, y, bw, bh, &mut all);
        all.sort_by(|a, b| a.cost.total_cmp(&b.cost));
        all
    }

    fn pick_inter_into(
        &mut self,
        x: usize,
        y: usize,
        bw: usize,
        bh: usize,
        all: &mut Vec<InterChoice>,
    ) -> Option<()> {
        let src = self.src?;
        let lam = self.mode.lambda_satd;
        let mut seen = [false; 8];
        let mut opts = Vec::with_capacity(4);
        // Best single result per reference: (cost, motion search vector).
        let mut single: [Option<(f64, Mv)>; 8] = [None; 8];
        for rf in LAST_FRAME..=ALTREF_FRAME {
            if self.refs[rf as usize].is_none()
                || !self
                    .mode
                    .ref_mask
                    .get(rf as usize)
                    .copied()
                    .unwrap_or(false)
            {
                continue;
            }
            let slot = self.hdr.ref_frame_idx[rf as usize - 1] as usize & 7;
            if seen[slot] {
                continue;
            }
            seen[slot] = true;
            opts.clear();
            let Some(((mode, mv, idx, cost, mbits), me)) =
                self.pick_inter_ref(rf, x, y, bw, bh, &mut opts)
            else {
                continue;
            };
            let rbits = if self.mode.exact_rates {
                self.ref_bits([rf, NONE])
            } else if rf == LAST_FRAME {
                1.0
            } else {
                2.5
            };
            let cost = cost + lam * rbits;
            single[rf as usize] = Some((cost, me));
            all.push(InterChoice {
                refs: [rf, NONE],
                mode,
                mv: [mv, Mv::ZERO],
                idx,
                cost,
                bits: mbits + rbits,
            });
            // The other modes of this reference, as RD candidates.
            for &(m2, mv2, idx2, c2, r2) in opts.iter().take(self.mode.inter_rd_modes) {
                if m2 != mode || mv2 != mv {
                    all.push(InterChoice {
                        refs: [rf, NONE],
                        mode: m2,
                        mv: [mv2, Mv::ZERO],
                        idx: idx2,
                        cost: c2 + lam * rbits,
                        bits: r2 + rbits,
                    });
                }
            }
        }
        if !self.hdr.reference_select || bw.min(bh) < 8 {
            return Some(());
        }
        let pick = |range: std::ops::RangeInclusive<i8>| -> Option<(i8, Mv)> {
            range
                .filter_map(|r| single[r as usize].map(|(c, m)| (r, c, m)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(r, _, m)| (r, m))
        };
        let (Some((f, fmv)), Some((bk, bmv))) = (
            pick(LAST_FRAME..=GOLDEN_FRAME),
            pick(BWDREF_FRAME..=ALTREF_FRAME),
        ) else {
            return Some(());
        };
        let fw = self.fs.mi_cols * 4;
        let fh = self.fs.mi_rows * 4;
        let w = bw.min(fw - x);
        let h = bh.min(fh - y);
        self.b.ref_frame = [f, bk];
        self.b.is_inter = true;
        self.find_mv_stack(true);
        let s0 = self.b.ref_stack_mv[0];
        let hp = self.hdr.allow_high_precision_mv;
        let lower = |m: Mv| {
            if hp {
                m
            } else {
                Mv::new(i32::from(m.row) & !1, i32::from(m.col) & !1)
            }
        };
        let (fmv, bmv) = (lower(fmv), lower(bmv));
        let d = |a: Mv, b: Mv| {
            bits(i32::from(a.row) - i32::from(b.row)) + bits(i32::from(a.col) - i32::from(b.col))
        };
        let dm = |a: Mv, b: Mv| {
            (
                i32::from(a.row) - i32::from(b.row),
                i32::from(a.col) - i32::from(b.col),
            )
        };
        let opts = if self.mode.exact_rates {
            let rb = self.ref_bits([f, bk]) - 3.0;
            [
                (
                    NEAREST_NEARESTMV,
                    [s0[0], s0[1]],
                    self.mode_bits(true, NEAREST_NEARESTMV, 0) + rb,
                ),
                (
                    NEW_NEWMV,
                    [fmv, bmv],
                    self.mode_bits(true, NEW_NEWMV, 0)
                        + self.mv_bits(dm(fmv, s0[0]))
                        + self.mv_bits(dm(bmv, s0[1]))
                        + rb,
                ),
            ]
        } else {
            [
                (NEAREST_NEARESTMV, [s0[0], s0[1]], 2.0),
                (NEW_NEWMV, [fmv, bmv], 4.0 + d(fmv, s0[0]) + d(bmv, s0[1])),
            ]
        };
        for (mode, mv, rate) in opts {
            if !self.pred_luma([f, bk], mv, x, y, w, h) {
                continue;
            }
            let sp = &src.planes[0];
            let dist = satd(
                &sp.data[y * sp.stride + x..],
                sp.stride,
                &self.scratch.obmc,
                w,
                w,
                h,
            );
            let cost = f64::from(dist) + lam * (rate + 3.0);
            all.push(InterChoice {
                refs: [f, bk],
                mode,
                mv,
                idx: 0,
                cost,
                bits: rate + 3.0,
            });
        }
        Some(())
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
        if !self.pred_luma(pb.ref_frame, pb.mv, x, y, w, h) {
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
        if (e as f64) >= self.mode.skip_sse_per_px * (w * h) as f64 {
            return false;
        }
        // Skip only when dropping the residual beats coding it.
        let tx = pb.tx_size;
        let (cw, ch) = (w & !(tx_w(tx) - 1), h & !(tx_h(tx) - 1));
        if cw == 0 || ch == 0 || cw != w || ch != h {
            return false;
        }
        let key = self.pred_key(0, pb.ref_frame, pb.mv, [x, y, w, h]);
        let coded = match self.me_scratch.tx_memo {
            Some((k, t, c)) if k == key && t == tx => c,
            _ => {
                let mut res = vec![0i32; w * h];
                let pred = std::mem::take(&mut self.scratch.obmc);
                self.residual_of(&pred, w, x, y, w, h, &mut res);
                self.scratch.obmc = pred;
                self.tx_cost(&res, w, h, tx)
            }
        };
        if (e as f64) >= coded {
            return false;
        }
        // Chroma: the skipped residual must not cost more than coding it.
        if self.num_planes > 1 && pb.has_chroma {
            let (sx, sy) = (usize::from(self.sub_x), usize::from(self.sub_y));
            let cx = (pb.mi_col >> sx) * 4;
            let cy = (pb.mi_row >> sy) * 4;
            let cw = (w >> sx).max(4);
            let chh = (h >> sy).max(4);
            let uv_tx = self.get_tx_size_for(1, 0, pb.bsize);
            let (tw, th) = (tx_w(uv_tx), tx_h(uv_tx));
            if !cw.is_multiple_of(tw) || !chh.is_multiple_of(th) {
                return true;
            }
            for plane in 1..3 {
                if !self.pred_plane(plane, pb.ref_frame, pb.mv, cx, cy, cw, chh) {
                    return false;
                }
                let sp = &src.planes[plane];
                if cx + cw > sp.w.max(sp.stride) || cy + chh > sp.alloc_h {
                    return false;
                }
                let mut res = vec![0i32; cw * chh];
                for i in 0..chh {
                    for j in 0..cw {
                        res[i * cw + j] = sp.data[(cy + i) * sp.stride + cx + j].to_i32()
                            - self.scratch.obmc[i * cw + j].to_i32();
                    }
                }
                let ec: f64 = res.iter().map(|&v| f64::from(v * v)).sum();
                if ec >= self.tx_cost(&res, cw, chh, uv_tx) {
                    return false;
                }
            }
        }
        true
    }
}

/// Dead-zone quantizer level of a transform coefficient for `tx_cost`:
/// `(|c| * denom + q * 48 / 128) / q`, at most `1 << 15` (`denom = 1 << dlog`).
/// The quotient of these small integers is exact in `f64`, which vectorises.
#[inline(always)]
fn quant_level(c: i32, q: i32, dlog: u32) -> u32 {
    let a = (c.unsigned_abs() << dlog) + (q * 48 / 128) as u32;
    ((f64::from(a) / f64::from(q)) as u32).min(1 << 15)
}

/// Rate and distortion statistics of quantizing coefficients with one step.
#[derive(Default)]
struct QuantEst {
    /// Sum of `2 + 2 * bit_length(level)` over the non-zero levels.
    bits: u32,
    /// Sum of squared coefficients.
    kept: u64,
    /// Sum of squared quantization errors.
    qerr: u64,
}

fn quant_est(coefs: &[i32], q: i32, dlog: u32) -> QuantEst {
    let mut s = QuantEst::default();
    for &c in coefs {
        let l = quant_level(c, q, dlog);
        let a = u64::from(c.unsigned_abs());
        let v = u64::from((l * q as u32) >> dlog);
        s.kept += a * a;
        s.qerr += a.abs_diff(v) * a.abs_diff(v);
        if l != 0 {
            s.bits += 2 + 2 * (32 - l.leading_zeros());
        }
    }
    s
}

/// Takes a work buffer out of the scratch, sized to at least `n` (it is
/// only zeroed when first allocated; callers overwrite what they read).
fn take_buf<T: Copy>(v: &mut Vec<T>, n: usize, fill: T) -> Vec<T> {
    let mut b = std::mem::take(v);
    if b.len() < n {
        b.resize(n, fill);
    }
    b
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

//! Inter prediction for one block (spec 5.11.33 and 7.11.3).

use crate::dsp::Pixel;

use super::coder::Coder;
use crate::dsp::mc;
use crate::spec::tables::{QUANT_DIST_LOOKUP, QUANT_DIST_WEIGHT};

use super::obu::TRANSLATION;
use super::tile::{R, TileDec, ref_rows};
use super::types::{
    COMPOUND_AVERAGE, COMPOUND_DIFFWTD, COMPOUND_DISTANCE, COMPOUND_INTRA, COMPOUND_WEDGE, DC_PRED,
    GLOBAL_GLOBALMV, GLOBALMV, H_PRED, INTRA_FRAME, LOCALWARP, Mv, OBMC, SIMPLE, SMOOTH_PRED,
    V_PRED, bh4, block_height, block_width, bw4, mi_height_log2, mi_width_log2,
    plane_residual_size,
};

/// Where a block prediction goes.
enum Dst {
    /// Final pixels into the current frame at (x, y) of the plane.
    Frame,
    /// Final pixels into `scratch.obmc` with stride w.
    Obmc,
    /// Compound intermediate into tmp0 / tmp1.
    Tmp(usize),
}

impl<P: Pixel, C: Coder> TileDec<'_, P, C> {
    /// compute_prediction() (5.11.33).
    pub fn compute_prediction(&mut self) -> R<()> {
        if !self.b.is_inter {
            return Ok(());
        }
        let sb_mask = self.sb_size4 - 1;
        let sub_row = self.b.mi_row & sb_mask;
        let sub_col = self.b.mi_col & sb_mask;
        let planes = 1 + 2 * usize::from(self.b.has_chroma);
        for plane in 0..planes {
            let (sx, sy) = self.plane_sub(plane);
            let psz = plane_residual_size(self.b.bsize, sx != 0, sy != 0);
            let n4w = bw4(psz);
            let n4h = bh4(psz);
            let log2w = 2 + mi_width_log2(psz);
            let log2h = 2 + mi_height_log2(psz);
            let base_x = (self.b.mi_col >> sx) * 4;
            let base_y = (self.b.mi_row >> sy) * 4;
            let mut cand_row = (self.b.mi_row >> sy) << sy;
            let mut cand_col = (self.b.mi_col >> sx) << sx;
            let is_ii = self.b.ref_frame[1] == INTRA_FRAME;
            if is_ii {
                let mode = match self.b.interintra_mode {
                    0 => DC_PRED,
                    1 => V_PRED,
                    2 => H_PRED,
                    _ => SMOOTH_PRED,
                };
                let bdf = &self.block_decoded;
                let ar = bdf.get(plane, sub_row >> sy, (sub_col >> sx) + n4w + 1);
                let bl = bdf.get(plane, (sub_row >> sy) + n4h + 1, sub_col >> sx);
                let (al, au) = if plane == 0 {
                    (self.b.avail_l, self.b.avail_u)
                } else {
                    (self.b.avail_l_chroma, self.b.avail_u_chroma)
                };
                self.predict_intra(plane, base_x, base_y, al, au, ar, bl, mode, log2w, log2h);
            }
            let mut pred_w = block_width(self.b.bsize) >> sx;
            let mut pred_h = block_height(self.b.bsize) >> sy;
            let mut some_use_intra = false;
            // Unless the candidates start before the block (chroma of
            // blocks narrower or shorter than 8), they are the block's own
            // 4x4s, inter when its first reference is.
            let own = cand_row == self.b.mi_row
                && cand_col == self.b.mi_col
                && self.b.ref_frame[0] > INTRA_FRAME;
            for r in 0..if own { 0 } else { n4h << sy } {
                for c in 0..(n4w << sx) {
                    if self.mi(cand_row + r, cand_col + c).ref_frame[0] == INTRA_FRAME {
                        some_use_intra = true;
                    }
                }
            }
            let mut rows = n4h * 4;
            if some_use_intra {
                pred_w = n4w * 4;
                pred_h = n4h * 4;
                cand_row = self.b.mi_row;
                cand_col = self.b.mi_col;
            } else if base_y + pred_h > (self.fs.mi_rows * 4) >> sy && self.height_free() {
                // Nothing reads the rows below the decoded area: predict
                // the block down to it, in whole 8-row groups, where the
                // height changes nothing else of the prediction.
                let end = (self.fs.mi_rows * 4) >> sy;
                let h = end.saturating_sub(base_y).next_multiple_of(8).max(8);
                if h < pred_h && pred_h == rows {
                    (pred_h, rows) = (h, h);
                }
            }
            let mut r = 0;
            let mut y = 0;
            while y < rows {
                let mut c = 0;
                let mut x = 0;
                while x < n4w * 4 {
                    self.predict_inter(
                        plane,
                        base_x + x,
                        base_y + y,
                        pred_w,
                        pred_h,
                        cand_row + r,
                        cand_col + c,
                    )?;
                    x += pred_w;
                    c += 1;
                }
                y += pred_h;
                r += 1;
            }
        }
        Ok(())
    }

    /// Whether the rows of the block's inter prediction are independent
    /// of its height: no overlapped or warped motion, no inter-intra, and
    /// not a copy of the frame itself.
    fn height_free(&self) -> bool {
        let b = &self.b;
        let global = (b.y_mode == GLOBALMV || b.y_mode == GLOBAL_GLOBALMV)
            && b.ref_frame
                .iter()
                .any(|&rf| rf > INTRA_FRAME && self.hdr.gm_type[rf as usize] > TRANSLATION);
        b.motion_mode == SIMPLE && !global && !b.use_intrabc && b.ref_frame[1] != INTRA_FRAME
    }

    fn predict_inter(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        cand_row: usize,
        cand_col: usize,
    ) -> R<()> {
        let cand = *self.mi(cand_row, cand_col);
        let is_compound = cand.ref_frame[1] > INTRA_FRAME;
        let bd = self.bd;
        if plane == 0 && self.b.motion_mode == LOCALWARP {
            self.warp_estimation();
            if self.b.local_valid {
                self.b.local_valid = mc::setup_shear(&self.b.local_warp).is_some();
            }
        }
        if is_compound
            && self.b.compound_type == COMPOUND_AVERAGE
            && self.whole_sample_average(plane, x, y, w, h, &cand)
        {
            if self.b.motion_mode == OBMC {
                self.overlapped_mc(plane, w, h)?;
            }
            return Ok(());
        }
        let is_ii = self.b.is_inter && self.b.ref_frame[1] == INTRA_FRAME;
        if is_ii {
            // Save the intra prediction; the inter prediction goes to the frame.
            let cy = y - self.row0(plane);
            let pl = &self.cur.planes[plane];
            let st = pl.stride;
            for i in 0..h {
                self.scratch.obmc[i * w..i * w + w]
                    .copy_from_slice(&pl.data[(cy + i) * st + x..][..w]);
            }
        }
        for ref_list in 0..1 + usize::from(is_compound) {
            let ref_frame = cand.ref_frame[ref_list];
            let ym = self.b.y_mode;
            let gm_warp = (ym == GLOBALMV || ym == GLOBAL_GLOBALMV)
                && !self.b.use_intrabc
                && self.hdr.gm_type[ref_frame.max(0) as usize] > TRANSLATION;
            let global_valid =
                gm_warp && mc::setup_shear(&self.hdr.gm_params[ref_frame as usize]).is_some();
            let use_warp = if w < 8 || h < 8 || self.hdr.force_integer_mv {
                0
            } else if self.b.motion_mode == LOCALWARP && self.b.local_valid {
                1
            } else if gm_warp && !self.is_scaled(ref_frame) && global_valid {
                2
            } else {
                0
            };
            let dst = if is_compound {
                Dst::Tmp(ref_list)
            } else {
                Dst::Frame
            };
            if use_warp != 0 {
                let mat = if use_warp == 1 {
                    self.b.local_warp
                } else {
                    self.hdr.gm_params[ref_frame as usize]
                };
                self.warp_block(plane, ref_frame, &mat, x, y, w, h, &dst)?;
            } else {
                let mv = cand.mv[ref_list];
                self.block_pred(plane, ref_frame, x, y, w, h, mv, cand.interp_filter(), &dst)?;
            }
        }
        // Combine.
        let (sx, sy) = self.plane_sub(plane);
        let ct = self.b.compound_type;
        let (fwd, bck) = if is_compound && ct == COMPOUND_DISTANCE {
            self.distance_weights(cand.ref_frame)
        } else {
            (0, 0)
        };
        let cy = y - self.row0(plane);
        let pl = &mut self.cur.planes[plane];
        let st = pl.stride;
        let dstp = &mut pl.data[cy * st + x..];
        if is_compound {
            let t0 = &self.scratch.tmp0[..w * h];
            let t1 = &self.scratch.tmp1[..w * h];
            match ct {
                COMPOUND_WEDGE => {
                    let bs = self.b.bsize as usize;
                    let mstride = block_width(self.b.bsize);
                    let Some(m) =
                        mc::wedge_mask(bs, self.b.wedge_sign != 0, self.b.wedge_index as usize)
                    else {
                        return Err(super::tile::malformed("wedge mask"));
                    };
                    mc::mask_blend(dstp, st, t0, t1, w, h, m, mstride, sx != 0, sy != 0, bd);
                }
                COMPOUND_DIFFWTD => {
                    if plane == 0 {
                        mc::diffwtd_mask(
                            &mut self.scratch.mask,
                            t0,
                            t1,
                            w,
                            h,
                            self.b.mask_type != 0,
                            bd,
                        );
                    }
                    let mstride = block_width(self.b.bsize);
                    mc::mask_blend(
                        dstp,
                        st,
                        t0,
                        t1,
                        w,
                        h,
                        &self.scratch.mask,
                        mstride,
                        sx != 0,
                        sy != 0,
                        bd,
                    );
                }
                COMPOUND_DISTANCE => {
                    mc::w_avg(dstp, st, t0, t1, w, h, fwd, bck, bd);
                }
                _ => {
                    debug_assert_eq!(ct, COMPOUND_AVERAGE);
                    mc::avg(dstp, st, t0, t1, w, h, bd);
                }
            }
        } else if is_ii {
            let intra = &self.scratch.obmc[..w * h];
            if ct == COMPOUND_INTRA {
                mc::ii_smooth_mask(&mut self.scratch.mask, w, w, h, self.b.interintra_mode);
                mc::interintra_blend(
                    dstp,
                    st,
                    intra,
                    w,
                    w,
                    h,
                    &self.scratch.mask,
                    w,
                    false,
                    false,
                );
            } else {
                debug_assert_eq!(ct, COMPOUND_WEDGE);
                let bs = self.b.bsize as usize;
                let Some(m) = mc::wedge_mask(bs, false, self.b.wedge_index as usize) else {
                    return Err(super::tile::malformed("wedge mask"));
                };
                let mstride = block_width(self.b.bsize);
                mc::interintra_blend(dstp, st, intra, w, w, h, m, mstride, sx != 0, sy != 0);
            }
        }
        if self.b.motion_mode == OBMC {
            self.overlapped_mc(plane, w, h)?;
        }
        Ok(())
    }

    /// The position `(x, y)` in plane `plane` of the reference
    /// `ref_frame` of the block at `(x, y)` its motion vector `mv` points
    /// to, when the reference is unscaled, the position whole samples and
    /// the block inside the reference (the prediction is then a plain copy).
    fn whole_sample_offset(
        &self,
        plane: usize,
        ref_frame: i8,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        mv: Mv,
    ) -> Option<(usize, usize)> {
        let (pw, ph, up_w, rh) = self.ref_geometry(plane, ref_frame);
        let (sx0, sy0, step_x, step_y) = self.scale_mv(plane, up_w, rh, x, y, mv);
        if step_x != 1024 || step_y != 1024 || (sx0 | sy0) & (15 << 6) != 0 {
            return None;
        }
        let (ix, iy) = (sx0 >> 10, sy0 >> 10);
        if ix < 0 || iy < 0 || ix + w as i64 > pw as i64 || iy + h as i64 > ph as i64 {
            return None;
        }
        Some((ix as usize, iy as usize))
    }

    /// `COMPOUND_AVERAGE` of two predictions that are both plain copies
    /// (see [`Self::whole_sample_offset`]): the compound rounding then
    /// reduces to `(a + b + 1) >> 1` at every bit depth. Returns false,
    /// predicting nothing, unless both are.
    fn whole_sample_average(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        cand: &super::types::MiInfo,
    ) -> bool {
        if self.b.use_intrabc {
            return false;
        }
        let (sx, sy) = self.plane_sub(plane);
        let ym = self.b.y_mode;
        let global = ym == GLOBALMV || ym == GLOBAL_GLOBALMV;
        let mut offs = [(0usize, 0usize); 2];
        for l in 0..2 {
            let (rf, mv) = (cand.ref_frame[l], cand.mv[l]);
            // Whole luma (1/8) or chroma (1/16) samples.
            if (i32::from(mv.row) & ((8 << sy) - 1)) | (i32::from(mv.col) & ((8 << sx) - 1)) != 0
                || rf <= INTRA_FRAME
                || (global && self.hdr.gm_type[rf as usize] > TRANSLATION)
            {
                return false;
            }
            match self.whole_sample_offset(plane, rf, x, y, w, h, mv) {
                Some(o) => offs[l] = o,
                None => return false,
            }
        }
        let refs = &self.refs;
        let mirrors = &self.mirrors;
        let (Some(r0), Some(r1)) = (
            refs[cand.ref_frame[0] as usize].as_ref(),
            refs[cand.ref_frame[1] as usize].as_ref(),
        ) else {
            return false;
        };
        let (Some((s0, st0)), Some((s1, st1))) = (
            ref_rows(r0, mirrors, plane, offs[0].1, offs[0].1 + h),
            ref_rows(r1, mirrors, plane, offs[1].1, offs[1].1 + h),
        ) else {
            return false;
        };
        let cy = y - self.row0(plane);
        let pl = &mut self.cur.planes[plane];
        let st = pl.stride;
        mc::avg_copy(
            &mut pl.data[cy * st + x..],
            st,
            (&s0[offs[0].0..], st0),
            (&s1[offs[1].0..], st1),
            w,
            h,
        );
        true
    }

    /// Predicts a single-reference block into `scratch.obmc` (stride `w`)
    /// for mode decision.
    pub fn predict_into_scratch(
        &mut self,
        plane: usize,
        ref_frame: i8,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        mv: Mv,
        filters: [u8; 2],
    ) -> R<()> {
        self.block_pred(plane, ref_frame, x, y, w, h, mv, filters, &Dst::Obmc)
    }

    pub(crate) fn distance_weights(&self, refs: [i8; 2]) -> (i32, i32) {
        let mut dist = [0i32; 2];
        for l in 0..2 {
            let hnt = self.hdr.order_hints[refs[l] as usize];
            dist[l] = self
                .hdr
                .get_relative_dist(self.seq, hnt, self.hdr.order_hint)
                .abs()
                .clamp(0, 31);
        }
        let d0 = dist[1];
        let d1 = dist[0];
        let order = usize::from(d0 <= d1);
        if d0 == 0 || d1 == 0 {
            return (
                i32::from(QUANT_DIST_LOOKUP[3][order]),
                i32::from(QUANT_DIST_LOOKUP[3][1 - order]),
            );
        }
        let mut i = 0;
        while i < 3 {
            let c0 = i32::from(QUANT_DIST_WEIGHT[i][order]);
            let c1 = i32::from(QUANT_DIST_WEIGHT[i][1 - order]);
            if order == 1 {
                if d0 * c0 > d1 * c1 {
                    break;
                }
            } else if d0 * c0 < d1 * c1 {
                break;
            }
            i += 1;
        }
        (
            i32::from(QUANT_DIST_LOOKUP[i][order]),
            i32::from(QUANT_DIST_LOOKUP[i][1 - order]),
        )
    }

    /// Reference plane dimensions (lastX + 1, lastY + 1) and scale.
    fn ref_geometry(&self, plane: usize, ref_frame: i8) -> (usize, usize, usize, usize) {
        let (sx, sy) = self.plane_sub(plane);
        if self.b.use_intrabc {
            let fw = self.hdr.frame_width as usize;
            let fh = self.hdr.frame_height as usize;
            // Scale from FrameWidth (no effect); clamp region is MiCols*4.
            let _ = (fw, fh);
            return (
                (self.fs.mi_cols * 4) >> sx,
                (self.fs.mi_rows * 4) >> sy,
                fw,
                fh,
            );
        }
        let rv = self.refs[ref_frame as usize].as_ref();
        let (uw, rh) = rv.map_or((0, 0), |r| (r.upscaled_width, r.frame_height));
        (
            (uw + (1 << sx) - 1) >> sx,
            (rh + (1 << sy) - 1) >> sy,
            uw,
            rh,
        )
    }

    /// Motion vector scaling (7.11.3.3). Returns startX, startY, stepX, stepY.
    fn scale_mv(
        &self,
        plane: usize,
        ref_up_w: usize,
        ref_h: usize,
        x: usize,
        y: usize,
        mv: Mv,
    ) -> (i64, i64, i64, i64) {
        let (sx, sy) = self.plane_sub(plane);
        let fw = i64::from(self.hdr.frame_width);
        let fh = i64::from(self.hdr.frame_height);
        if ref_up_w as i64 == fw && ref_h as i64 == fh {
            // Unscaled (the common case): the scale is 1 << 14, the base
            // `(orig - 8) << 14`, and the rounding shift by 8 exact.
            let ox = ((x as i64) << 4) + ((2 * i64::from(mv.col)) >> sx);
            let oy = ((y as i64) << 4) + ((2 * i64::from(mv.row)) >> sy);
            return ((ox << 6) + 32, (oy << 6) + 32, 1024, 1024);
        }
        // Scaled ones need a division.
        let scale = |r: i64, f: i64| {
            if r == f {
                1 << 14
            } else {
                ((r << 14) + f / 2) / f
            }
        };
        let x_scale = scale(ref_up_w as i64, fw);
        let y_scale = scale(ref_h as i64, fh);
        let orig_x = ((x as i64) << 4) + ((2 * i64::from(mv.col)) >> sx) + 8;
        let orig_y = ((y as i64) << 4) + ((2 * i64::from(mv.row)) >> sy) + 8;
        let base_x = orig_x * x_scale - (8 << 14);
        let base_y = orig_y * y_scale - (8 << 14);
        let r2s = |v: i64, n: u32| -> i64 {
            if v >= 0 {
                (v + (1 << (n - 1))) >> n
            } else {
                -((-v + (1 << (n - 1))) >> n)
            }
        };
        let off = 32;
        (
            r2s(base_x, 8) + off,
            r2s(base_y, 8) + off,
            r2s(x_scale, 4),
            r2s(y_scale, 4),
        )
    }

    fn block_pred(
        &mut self,
        plane: usize,
        ref_frame: i8,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        mv: Mv,
        filters: [u8; 2],
        dst: &Dst,
    ) -> R<()> {
        let bd = self.bd;
        let (pw, ph, up_w, rh) = self.ref_geometry(plane, ref_frame);
        let (sx0, sy0, step_x, step_y) = if self.b.use_intrabc {
            let (sx, sy) = self.plane_sub(plane);
            let s = self.scale_mv(
                plane,
                self.hdr.frame_width as usize,
                self.hdr.frame_height as usize,
                x,
                y,
                mv,
            );
            let _ = (sx, sy);
            s
        } else {
            self.scale_mv(plane, up_w, rh, x, y, mv)
        };
        let filt = |f: u8, n: usize| -> usize {
            let f = usize::from(f);
            if n <= 4 {
                match f {
                    0 | 2 => 4,
                    1 => 5,
                    _ => f,
                }
            } else {
                f
            }
        };
        let fx = filt(filters[1], w);
        let fy = filt(filters[0], h);
        let ix = sx0 >> 10;
        let iy = sy0 >> 10;
        let x_frac = (sx0 & 1023) as usize;
        let y_frac = (sy0 & 1023) as usize;
        let scaled = step_x != 1024 || step_y != 1024;
        let (fw, fh) = if scaled {
            mc::scaled_footprint(w, h, x_frac, y_frac, step_x as usize, step_y as usize)
        } else {
            (w + 7, h + 7)
        };
        // Source: the reference plane, or an edge-emulated copy. Unscaled,
        // a direction without a fraction reads no samples beside the block
        // (the filters copy), so only the samples read need to be inside.
        let (left, right) = if scaled || (sx0 >> 6) & 15 != 0 {
            (3, fw as i64 - 3)
        } else {
            (0, w as i64)
        };
        let (top, bottom) = if scaled || (sy0 >> 6) & 15 != 0 {
            (3, fh as i64 - 3)
        } else {
            (0, h as i64)
        };
        let inside =
            ix - left >= 0 && iy - top >= 0 && ix + right <= pw as i64 && iy + bottom <= ph as i64;
        let refs = &self.refs;
        let mirrors = &self.mirrors;
        let rv = if self.b.use_intrabc {
            None
        } else {
            match refs[ref_frame as usize].as_ref() {
                Some(rv) => Some(rv),
                None => return Err(super::tile::malformed("missing reference")),
            }
        };
        // The rows read, in place (in one band of a reference still being
        // decoded).
        let direct = match rv {
            Some(rv) if inside => {
                let (y0, y1) = ((iy - top) as usize, (iy + bottom) as usize);
                // Without a vertical fraction the filters read no rows
                // above the block, but the vector kernels address their
                // source from 3 rows above it: the rows start there when
                // the picture (or the band) has them.
                let up = if top == 0 { y0.min(3) } else { 0 };
                ref_rows(rv, mirrors, plane, y0 - up, y1)
                    .map(|(s, st)| (s, st, up))
                    .or_else(|| ref_rows(rv, mirrors, plane, y0, y1).map(|(s, st)| (s, st, 0)))
            }
            _ => None,
        };
        let (src, sstride, soff): (&[P], usize, usize) = match direct {
            Some((s, st, up)) => (s, st, (top as usize + up) * st + ix as usize),
            None => {
                if self.scratch.emu.len() < fw * fh {
                    self.scratch.emu.resize(fw * fh, P::default());
                }
                let emu = &mut self.scratch.emu;
                let (ex, ey) = ((ix - 3) as isize, (iy - 3) as isize);
                let src = match rv {
                    None => Some(&self.cur.planes[plane]),
                    Some(rv) if rv.mirror.is_none() => rv.pic.planes.get(plane),
                    Some(_) => None,
                };
                let ok = match (src, rv.and_then(|rv| rv.mirror)) {
                    (Some(p), _) => {
                        mc::emu_edge(emu, fw, &p.data, p.stride, pw, ph, ex, ey, fw, fh);
                        true
                    }
                    (None, Some(k)) => mirrors.get(k).is_some_and(|m| {
                        // The band of the last row read, while it holds
                        // the next: its first row, the row past its last,
                        // the samples from its first and their stride.
                        let at = core::cell::Cell::new((0, 0, &[][..], 0));
                        mc::emu_edge_rows(
                            emu,
                            fw,
                            |r| {
                                let (a, b, s, st) = at.get();
                                if (a..b).contains(&r) {
                                    return s.get((r - a) * st..);
                                }
                                let (s, st, end) = m.rows_from(plane, r)?;
                                at.set((r, end, s, st));
                                Some(s)
                            },
                            pw,
                            ph,
                            ex,
                            ey,
                            fw,
                            fh,
                        )
                    }),
                    (None, None) => false,
                };
                if !ok {
                    // Rows of a reference still being decoded that are
                    // not published yet: the caller waits for them and
                    // predicts the block again.
                    if let Some(k) = rv.and_then(|rv| rv.mirror) {
                        let last = (ey + fh as isize - 1).clamp(0, ph as isize - 1) as usize;
                        let sy = self.plane_sub(plane).1;
                        self.scratch.want_rows = Some((k, (last + 1) << sy));
                    }
                    return Err(super::tile::malformed("reference rows"));
                }
                (&self.scratch.emu[..], fw, 3 * fw + 3)
            }
        };
        let mx = (sx0 >> 6) as usize & 15;
        let my = (sy0 >> 6) as usize & 15;
        match dst {
            Dst::Frame | Dst::Obmc => {
                let (out, ostride): (&mut [P], usize) = if matches!(dst, Dst::Frame) {
                    let cy = y - self.row0(plane);
                    let pl = &mut self.cur.planes[plane];
                    let st = pl.stride;
                    (&mut pl.data[cy * st + x..], st)
                } else {
                    (&mut self.scratch.obmc[..], w)
                };
                if scaled {
                    mc::put_scaled(
                        out,
                        ostride,
                        src,
                        soff,
                        sstride,
                        w,
                        h,
                        x_frac,
                        y_frac,
                        step_x as usize,
                        step_y as usize,
                        fx,
                        fy,
                        bd,
                    );
                } else {
                    mc::put(out, ostride, src, soff, sstride, w, h, mx, my, fx, fy, bd);
                }
            }
            Dst::Tmp(l) => {
                let tmp = if *l == 0 {
                    &mut self.scratch.tmp0[..]
                } else {
                    &mut self.scratch.tmp1[..]
                };
                if scaled {
                    mc::prep_scaled(
                        tmp,
                        src,
                        soff,
                        sstride,
                        w,
                        h,
                        x_frac,
                        y_frac,
                        step_x as usize,
                        step_y as usize,
                        fx,
                        fy,
                        bd,
                    );
                } else {
                    mc::prep(tmp, src, soff, sstride, w, h, mx, my, fx, fy, bd);
                }
            }
        }
        Ok(())
    }

    fn warp_block(
        &mut self,
        plane: usize,
        ref_frame: i8,
        mat: &[i32; 6],
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        dst: &Dst,
    ) -> R<()> {
        let bd = self.bd;
        let (sx, sy) = self.plane_sub(plane);
        let (pw, ph, _, _) = self.ref_geometry(plane, ref_frame);
        let Some(shear) = mc::setup_shear(mat) else {
            return Err(super::tile::malformed("warp"));
        };
        let Some(rv) = self.refs[ref_frame as usize].as_ref() else {
            return Err(super::tile::malformed("missing reference"));
        };
        for i8 in 0..h.div_ceil(8) {
            for j8 in 0..w.div_ceil(8) {
                let bx = x + j8 * 8;
                let by = y + i8 * 8;
                // The whole plane of a reference that is done, or the
                // rows this 8x8 reads of one still being decoded, from the
                // band that holds them.
                let (refp, ref_stride, row0): (&[P], usize, usize) = match rv.mirror {
                    None => {
                        let rp = &rv.pic.planes[plane];
                        (&rp.data, rp.stride, 0)
                    }
                    Some(k) => {
                        let (first, last) = mc::warp_rows(mat, bx, by, sx != 0, sy != 0, ph);
                        let rows = self
                            .mirrors
                            .get(k)
                            .and_then(|m| m.rows(plane, first, last + 1));
                        let Some((rows, stride)) = rows else {
                            // Not published yet: the caller waits for
                            // them and predicts the block again.
                            self.scratch.want_rows = Some((k, (last + 1) << sy));
                            return Err(super::tile::malformed("reference rows"));
                        };
                        (rows, stride, first)
                    }
                };
                match dst {
                    Dst::Tmp(l) => {
                        let tmp = if *l == 0 {
                            &mut self.scratch.tmp0[..]
                        } else {
                            &mut self.scratch.tmp1[..]
                        };
                        mc::warp_8x8_prep_rows(
                            &mut tmp[i8 * 8 * w + j8 * 8..],
                            w,
                            refp,
                            ref_stride,
                            row0,
                            pw,
                            ph,
                            mat,
                            &shear,
                            bx,
                            by,
                            sx != 0,
                            sy != 0,
                            bd,
                        );
                    }
                    _ => {
                        let cy = by - self.row0(plane);
                        let pl = &mut self.cur.planes[plane];
                        let st = pl.stride;
                        mc::warp_8x8_put_rows(
                            &mut pl.data[cy * st + bx..],
                            st,
                            refp,
                            ref_stride,
                            row0,
                            pw,
                            ph,
                            mat,
                            &shear,
                            bx,
                            by,
                            sx != 0,
                            sy != 0,
                            bd,
                        );
                    }
                }
            }
        }
        Ok(())
    }

    fn overlapped_mc(&mut self, plane: usize, w: usize, h: usize) -> R<()> {
        let (sx, sy) = self.plane_sub(plane);
        let b = self.b.clone();
        if b.avail_u && plane_residual_size(b.bsize, sx != 0, sy != 0) >= super::types::BLOCK_8X8 {
            let w4 = bw4(b.bsize);
            let mut x4 = b.mi_col;
            let mut count = 0;
            let limit = 4.min(mi_width_log2(b.bsize));
            while count < limit && x4 < self.fs.mi_cols.min(b.mi_col + w4) {
                let cr = b.mi_row - 1;
                let cc = (x4 | 1).min(self.fs.mi_cols - 1);
                let cand = *self.mi(cr, cc);
                let step4 = bw4(cand.bsize).clamp(2, 16);
                if cand.ref_frame[0] > INTRA_FRAME {
                    count += 1;
                    let pred_w = w.min((step4 * 4) >> sx);
                    let pred_h = (h >> 1).min(32 >> sy);
                    let px = (x4 * 4) >> sx;
                    let py = (b.mi_row * 4) >> sy;
                    self.block_pred(
                        plane,
                        cand.ref_frame[0],
                        px,
                        py,
                        pred_w,
                        pred_h,
                        cand.mv[0],
                        cand.interp_filter(),
                        &Dst::Obmc,
                    )?;
                    let cy = py - self.row0(plane);
                    let pl = &mut self.cur.planes[plane];
                    let st = pl.stride;
                    mc::obmc_blend_above(
                        &mut pl.data[cy * st + px..],
                        st,
                        &self.scratch.obmc,
                        pred_w,
                        pred_w,
                        pred_h,
                    );
                }
                x4 += step4;
            }
        }
        if b.avail_l {
            let h4 = bh4(b.bsize);
            let mut y4 = b.mi_row;
            let mut count = 0;
            let limit = 4.min(mi_height_log2(b.bsize));
            while count < limit && y4 < self.fs.mi_rows.min(b.mi_row + h4) {
                let cc = b.mi_col - 1;
                let cr = (y4 | 1).min(self.fs.mi_rows - 1);
                let cand = *self.mi(cr, cc);
                let step4 = bh4(cand.bsize).clamp(2, 16);
                if cand.ref_frame[0] > INTRA_FRAME {
                    count += 1;
                    let pred_w = (w >> 1).min(32 >> sx);
                    let pred_h = h.min((step4 * 4) >> sy);
                    let px = (b.mi_col * 4) >> sx;
                    let py = (y4 * 4) >> sy;
                    self.block_pred(
                        plane,
                        cand.ref_frame[0],
                        px,
                        py,
                        pred_w,
                        pred_h,
                        cand.mv[0],
                        cand.interp_filter(),
                        &Dst::Obmc,
                    )?;
                    let cy = py - self.row0(plane);
                    let pl = &mut self.cur.planes[plane];
                    let st = pl.stride;
                    mc::obmc_blend_left(
                        &mut pl.data[cy * st + px..],
                        st,
                        &self.scratch.obmc,
                        pred_w,
                        pred_w,
                        pred_h,
                    );
                }
                y4 += step4;
            }
        }
        Ok(())
    }

    /// Warp estimation (7.11.3.8).
    /// Encoder: when `self.b` (a single-reference inter block with its
    /// motion vector) may use local warped motion, writes that prediction of
    /// the luma block at (x, y) into `scratch.obmc` (stride w) and returns
    /// true. Leaves `self.b.motion_mode` as SIMPLE.
    pub fn predict_local_warp_into_scratch(
        &mut self,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    ) -> bool {
        let bs = self.b.bsize;
        let rf = self.b.ref_frame[0];
        if !self.hdr.is_motion_mode_switchable
            || !self.hdr.allow_warped_motion
            || self.hdr.force_integer_mv
            || self.b.ref_frame[1] > INTRA_FRAME
            || block_width(bs).min(block_height(bs)) < 8
            || w < 8
            || h < 8
            || ((self.b.y_mode == GLOBALMV || self.b.y_mode == GLOBAL_GLOBALMV)
                && self.hdr.gm_type[rf.max(0) as usize] > TRANSLATION)
            || self.is_scaled(rf)
            || !self.has_overlappable_candidates()
        {
            return false;
        }
        self.find_warp_samples();
        if self.b.num_samples == 0 {
            return false;
        }
        self.b.motion_mode = LOCALWARP;
        self.warp_estimation();
        self.b.motion_mode = SIMPLE;
        if !self.b.local_valid || mc::setup_shear(&self.b.local_warp).is_none() {
            return false;
        }
        let mat = self.b.local_warp;
        // The block is not coded yet, so its frame area is free to predict into.
        if self
            .warp_block(0, rf, &mat, x, y, w, h, &Dst::Frame)
            .is_err()
        {
            return false;
        }
        let pl = &self.cur.planes[0];
        for i in 0..h {
            self.scratch.obmc[i * w..i * w + w]
                .copy_from_slice(&pl.data[(y + i) * pl.stride + x..][..w]);
        }
        true
    }

    /// Encoder: the overlapped (OBMC) luma prediction of the single-reference
    /// block `self.b` at (x, y) into `scratch.obmc` (stride w), when the
    /// block may signal OBMC.
    pub fn predict_obmc_into_scratch(&mut self, x: usize, y: usize, w: usize, h: usize) -> bool {
        let bs = self.b.bsize;
        let rf = self.b.ref_frame[0];
        if !self.hdr.is_motion_mode_switchable
            || self.b.ref_frame[1] > INTRA_FRAME
            || block_width(bs).min(block_height(bs)) < 8
            || ((self.b.y_mode == GLOBALMV || self.b.y_mode == GLOBAL_GLOBALMV)
                && self.hdr.gm_type[rf.max(0) as usize] > TRANSLATION)
            || !self.has_overlappable_candidates()
        {
            return false;
        }
        let mv = self.b.mv[0];
        let f = self.b.interp_filter;
        if self
            .block_pred(0, rf, x, y, w, h, mv, f, &Dst::Frame)
            .is_err()
        {
            return false;
        }
        self.b.motion_mode = OBMC;
        let ok = self.overlapped_mc(0, w, h).is_ok();
        self.b.motion_mode = SIMPLE;
        if !ok {
            return false;
        }
        let pl = &self.cur.planes[0];
        for i in 0..h {
            self.scratch.obmc[i * w..i * w + w]
                .copy_from_slice(&pl.data[(y + i) * pl.stride + x..][..w]);
        }
        true
    }

    /// Encoder: the global-motion warp prediction from `rf` of the luma
    /// block at (x, y) into `scratch.obmc` (stride w), when GLOBALMV would
    /// warp there.
    pub fn predict_global_warp_into_scratch(
        &mut self,
        rf: i8,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    ) -> bool {
        if w < 8
            || h < 8
            || self.hdr.force_integer_mv
            || self.hdr.gm_type[rf.max(0) as usize] <= TRANSLATION
            || self.is_scaled(rf)
        {
            return false;
        }
        let mat = self.hdr.gm_params[rf as usize];
        if mc::setup_shear(&mat).is_none()
            || self
                .warp_block(0, rf, &mat, x, y, w, h, &Dst::Frame)
                .is_err()
        {
            return false;
        }
        let pl = &self.cur.planes[0];
        for i in 0..h {
            self.scratch.obmc[i * w..i * w + w]
                .copy_from_slice(&pl.data[(y + i) * pl.stride + x..][..w]);
        }
        true
    }

    fn warp_estimation(&mut self) {
        let b = &self.b;
        let mut a = [[0i64; 2]; 2];
        let mut bx = [0i64; 2];
        let mut by = [0i64; 2];
        let w4 = bw4(b.bsize) as i64;
        let h4 = bh4(b.bsize) as i64;
        let mid_y = b.mi_row as i64 * 4 + h4 * 2 - 1;
        let mid_x = b.mi_col as i64 * 4 + w4 * 2 - 1;
        let suy = mid_y * 8;
        let sux = mid_x * 8;
        let duy = suy + i64::from(b.mv[0].row);
        let dux = sux + i64::from(b.mv[0].col);
        let ls = |a: i64, b: i64| ((a * b) >> 2) + (a + b);
        for i in 0..b.num_samples {
            let c = b.cand_list[i];
            let sy = i64::from(c[0]) - suy;
            let sx = i64::from(c[1]) - sux;
            let dy = i64::from(c[2]) - duy;
            let dx = i64::from(c[3]) - dux;
            if (sx - dx).abs() < 256 && (sy - dy).abs() < 256 {
                a[0][0] += ls(sx, sx) + 8;
                a[0][1] += ls(sx, sy) + 4;
                a[1][1] += ls(sy, sy) + 8;
                bx[0] += ls(sx, dx) + 8;
                bx[1] += ls(sy, dx) + 4;
                by[0] += ls(sx, dy) + 4;
                by[1] += ls(sy, dy) + 8;
            }
        }
        let det = a[0][0] * a[1][1] - a[0][1] * a[0][1];
        let Some((div_shift, div_factor)) = mc::resolve_divisor(det) else {
            self.b.local_valid = false;
            return;
        };
        let mut div_shift = div_shift as i64 - 16;
        let mut div_factor = i64::from(div_factor);
        if div_shift < 0 {
            div_factor <<= -div_shift;
            div_shift = 0;
        }
        let r2s = |v: i128, n: i64| -> i64 {
            if n == 0 {
                return v as i64;
            }
            let r = if v >= 0 {
                (v + (1i128 << (n - 1))) >> n
            } else {
                -((-v + (1i128 << (n - 1))) >> n)
            };
            r as i64
        };
        let clamp_nd = 1i64 << 13;
        let nondiag = |v: i64| {
            r2s(i128::from(v) * i128::from(div_factor), div_shift)
                .clamp(-clamp_nd + 1, clamp_nd - 1)
        };
        let diag = |v: i64| {
            r2s(i128::from(v) * i128::from(div_factor), div_shift)
                .clamp((1 << 16) - clamp_nd + 1, (1 << 16) + clamp_nd - 1)
        };
        let mut p = [0i64; 6];
        p[2] = diag(a[1][1] * bx[0] - a[0][1] * bx[1]);
        p[3] = nondiag(-a[0][1] * bx[0] + a[0][0] * bx[1]);
        p[4] = nondiag(a[1][1] * by[0] - a[0][1] * by[1]);
        p[5] = diag(-a[0][1] * by[0] + a[0][0] * by[1]);
        let mvx = i64::from(b.mv[0].col);
        let mvy = i64::from(b.mv[0].row);
        let vx = mvx * (1 << 13) - (mid_x * (p[2] - (1 << 16)) + mid_y * p[3]);
        let vy = mvy * (1 << 13) - (mid_x * p[4] + mid_y * (p[5] - (1 << 16)));
        let tc = 1i64 << 23;
        p[0] = vx.clamp(-tc, tc - 1);
        p[1] = vy.clamp(-tc, tc - 1);
        self.b.local_warp = std::array::from_fn(|i| p[i] as i32);
        self.b.local_valid = true;
    }
}

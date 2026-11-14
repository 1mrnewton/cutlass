//! Reconstruction of one superblock row from the parsed syntax: intra and
//! inter prediction and the residual of each transform block, into the
//! rows of the picture its writer has not published yet.

use std::sync::Arc;

use cutlass_simd::vp9::frame::{SharedFrame, Tail};

use crate::dsp::ipred::{self, Edges};
use crate::dsp::itx::{self, Kind};
use crate::dsp::mc;
use crate::dsp::pixel::Pixel;
use crate::geom::{self, BLOCK_8X8, INTRA_FRAME};
use crate::tables::SUBPEL_FILTERS;

use super::frame::Dims;
use super::syntax::{BlockRec, TileSyntax};

/// Width and height of each block size in 8×8 units (at least 1).
const MI_WH: [(usize, usize); 13] = [
    (1, 1),
    (1, 1),
    (1, 1),
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (2, 4),
    (4, 2),
    (4, 4),
    (4, 8),
    (8, 4),
    (8, 8),
];

/// Fixed-point scale of a reference of another size (libvpx
/// `scale_factors`).
#[derive(Clone, Copy, Debug)]
pub struct Scale {
    pub x_fp: i32,
    pub y_fp: i32,
    pub x_step: i32,
    pub y_step: i32,
}

impl Scale {
    /// `None` for the same size; `Err` for a ratio VP9 does not allow.
    pub fn between(rw: usize, rh: usize, w: usize, h: usize) -> Result<Option<Scale>, ()> {
        if rw == 0 || rh == 0 || w == 0 || h == 0 {
            return Err(());
        }
        if !(2 * w >= rw && 2 * h >= rh && w <= 16 * rw && h <= 16 * rh) {
            return Err(());
        }
        let x_fp = ((rw << 14) / w) as i32;
        let y_fp = ((rh << 14) / h) as i32;
        if x_fp == 1 << 14 && y_fp == 1 << 14 {
            return Ok(None);
        }
        Ok(Some(Scale {
            x_fp,
            y_fp,
            x_step: sc(16, x_fp),
            y_step: sc(16, y_fp),
        }))
    }
}

#[inline]
fn sc(v: i32, fp: i32) -> i32 {
    ((i64::from(v) * i64::from(fp)) >> 14) as i32
}

/// A reference picture with its scale relative to the frame.
pub struct RefPic<P> {
    pub buf: Arc<SharedFrame<P>>,
    pub dims: Dims,
    pub scale: Option<Scale>,
}

/// What reconstruction reads: the frame's shape and its references.
pub struct Ctx<P> {
    pub dims: Dims,
    pub refs: [Option<RefPic<P>>; 3],
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub lossless: bool,
}

/// One plane of the picture being made: the rows from `first_row` on.
pub struct View<'a, P> {
    pub data: &'a mut [P],
    /// Index of the first element of `data` in the whole plane.
    pub base: usize,
    pub stride: usize,
}

impl<'a, P> View<'a, P> {
    pub fn of(t: Tail<'a, P>) -> Self {
        Self {
            base: t.first_row * t.stride,
            stride: t.stride,
            data: t.data,
        }
    }

    /// The samples from row `y`, column `x` on.
    #[inline(always)]
    pub fn at(&mut self, y: usize, x: usize) -> &mut [P] {
        &mut self.data[y * self.stride + x - self.base..]
    }
}

/// Buffers reconstruction keeps from row to row.
pub struct Scratch<P> {
    /// Coefficients of the transform block being added; the transforms
    /// clear what they read.
    pub coef: Vec<i32>,
    pub emu: Vec<P>,
    pub mc: Vec<u8>,
}

impl<P: Pixel> Scratch<P> {
    pub fn new() -> Self {
        Self {
            coef: vec![0; 1024],
            emu: vec![P::default(); 80 * 80],
            mc: vec![0; cutlass_simd::vp9::mc::TMP],
        }
    }
}

/// The reconstruction of a frame, row by row.
pub struct Recon<'c, 'v, P> {
    c: &'c Ctx<P>,
    views: &'c mut [View<'v, P>; 3],
    s: &'c mut Scratch<P>,
    bd: u32,
    max: i32,
}

impl<'c, 'v, P: Pixel> Recon<'c, 'v, P> {
    pub fn new(c: &'c Ctx<P>, views: &'c mut [View<'v, P>; 3], s: &'c mut Scratch<P>) -> Self {
        let bd = u32::from(c.dims.bit_depth);
        Self {
            c,
            views,
            s,
            bd,
            max: (1i32 << bd) - 1,
        }
    }

    /// Reconstructs superblock row `sb_row` of every tile column.
    pub fn sb_row(&mut self, tiles: &[TileSyntax], sb_row: usize) {
        for t in tiles {
            let start = t.row_start(sb_row);
            let end = t.rows.get(sb_row).copied().unwrap_or(start);
            let mut tx = start.txs;
            let mut coef = start.coefs;
            for b in &t.blocks[start.blocks..end.blocks] {
                self.block(t, b, &mut tx, &mut coef);
            }
        }
    }

    fn plane_ss(&self, plane: usize) -> (usize, usize) {
        if plane == 0 {
            (0, 0)
        } else {
            (self.c.dims.ss_x, self.c.dims.ss_y)
        }
    }

    /// The tx blocks of a plane inside the frame: (blocks wide, high).
    fn visible(&self, b: &BlockRec, plane: usize) -> (usize, usize, usize, usize) {
        let (ssx, ssy) = self.plane_ss(plane);
        let (bw, bh) = MI_WH[usize::from(b.bsize)];
        let n4w = (bw * 2) >> ssx;
        let n4h = (bh * 2) >> ssy;
        let right = (self.c.mi_cols as i32 - bw as i32 - i32::from(b.col)) * 64;
        let bottom = (self.c.mi_rows as i32 - bh as i32 - i32::from(b.row)) * 64;
        let vw = if right >= 0 { n4w } else { (n4w as i32 + (right >> (5 + ssx))).max(0) as usize };
        let vh = if bottom >= 0 { n4h } else { (n4h as i32 + (bottom >> (5 + ssy))).max(0) as usize };
        (n4w, n4h, vw, vh)
    }

    fn block(&mut self, t: &TileSyntax, b: &BlockRec, tx_at: &mut usize, coef_at: &mut usize) {
        let intra = b.refs[0] == INTRA_FRAME;
        if !intra {
            self.predict_inter(b);
            if !b.coded {
                return;
            }
        }
        let lossless = self.c.lossless;
        for plane in 0..3 {
            let tx = if plane == 0 { b.tx } else { b.uv_tx };
            let (ssx, ssy) = self.plane_ss(plane);
            let (n4w, _, vw, vh) = self.visible(b, plane);
            let step = 1usize << tx;
            let x0 = (usize::from(b.col) * 8) >> ssx;
            let y0 = (usize::from(b.row) * 8) >> ssy;
            let mut r = 0;
            while r < vh {
                let mut c = 0;
                while c < vw {
                    let (x, y) = (x0 + c * 4, y0 + r * 4);
                    let mut kind = 0;
                    if intra {
                        let mode = if plane == 0 {
                            if b.bsize < BLOCK_8X8 { b.bmode[(r << 1) + c] } else { b.mode }
                        } else {
                            b.uv
                        };
                        let have_above = r > 0 || b.row > 0;
                        let have_left = c > 0 || usize::from(b.col) > t.col0;
                        let have_right = c + step < n4w;
                        self.predict_intra(plane, mode, tx, x, y, have_left, have_above, have_right);
                        if plane == 0 && !lossless && tx != geom::TX_32X32 {
                            kind = geom::intra_tx_type(mode);
                        }
                    }
                    if b.coded {
                        let rec = t.txs[*tx_at];
                        *tx_at += 1;
                        let n = usize::from(rec.n);
                        if rec.eob > 0 {
                            for k in *coef_at..*coef_at + n {
                                self.s.coef[usize::from(t.pos[k])] = t.val[k];
                            }
                            self.add_residual(plane, x, y, tx, kind, usize::from(rec.eob), (usize::from(rec.nzr), usize::from(rec.nzc)));
                        }
                        *coef_at += n;
                    }
                    c += step;
                }
                r += step;
            }
        }
    }

    /// Adds the residual of one tx block at sample `(x, y)` of a plane.
    #[allow(clippy::too_many_arguments)]
    fn add_residual(&mut self, plane: usize, x: usize, y: usize, tx: u8, kind: u8, eob: usize, nz: (usize, usize)) {
        let max = self.max;
        let v = &mut self.views[plane];
        let stride = v.stride;
        let dst = v.at(y, x);
        if self.c.lossless {
            itx::wht_add(dst, stride, &mut self.s.coef, max);
            return;
        }
        let (col, row) = match kind {
            1 => (Kind::Adst, Kind::Dct),
            2 => (Kind::Dct, Kind::Adst),
            3 => (Kind::Adst, Kind::Adst),
            _ => (Kind::Dct, Kind::Dct),
        };
        itx::add(dst, stride, &mut self.s.coef, usize::from(tx), col, row, eob, nz, max);
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_intra(
        &mut self,
        plane: usize,
        mode: u8,
        tx: u8,
        x: usize,
        y: usize,
        have_left: bool,
        have_above: bool,
        have_right: bool,
    ) {
        let bd = self.bd;
        let base = 1i32 << (bd - 1);
        let n = 4usize << tx;
        let (fw, fh) = (self.c.dims.aw[plane], self.c.dims.ah[plane]);
        let v = &mut self.views[plane];
        let stride = v.stride;
        let mut e = Edges {
            above: [P::new(base - 1); 64],
            left: [P::new(base + 1); 32],
            tl: P::new(base - 1),
            have_above,
            have_left,
        };
        let off = v.base;
        let data = &v.data;
        let px = |i: usize| data[i - off];
        if have_left {
            let count = n.min(fh.saturating_sub(y)).max(1);
            for i in 0..n {
                e.left[i] = px((y + i.min(count - 1)) * stride + x - 1);
            }
        }
        if have_above {
            let avail = if tx == 0 && have_right { 2 * n } else { n };
            let count = avail.min(fw.saturating_sub(x)).max(1);
            let row = (y - 1) * stride + x;
            for i in 0..2 * n {
                e.above[i] = px(row + i.min(count - 1));
            }
            e.tl = if have_left { px(row - 1) } else { P::new(base + 1) };
        }
        ipred::predict(mode, n, v.at(y, x), stride, &e, bd);
    }

    fn predict_inter(&mut self, b: &BlockRec) {
        let nref = 1 + usize::from(b.refs[1] > INTRA_FRAME);
        let filter = &SUBPEL_FILTERS[usize::from(b.interp).min(3)];
        let (bw, bh) = MI_WH[usize::from(b.bsize)];
        let (row, col) = (usize::from(b.row), usize::from(b.col));
        for plane in 0..3 {
            let (ssx, ssy) = self.plane_ss(plane);
            let x0 = (col * 8) >> ssx;
            let y0 = (row * 8) >> ssy;
            for r in 0..nref {
                let slot = usize::from(b.refs[r]).saturating_sub(1).min(2);
                if b.bsize < BLOCK_8X8 {
                    let n4w = (bw * 2) >> ssx;
                    let n4h = (bh * 2) >> ssy;
                    let mut i = 0;
                    for y in 0..n4h {
                        for x in 0..n4w {
                            let mv = split_mv(b, plane, i, r, ssx, ssy);
                            i += 1;
                            self.mc(slot, plane, (x0 + 4 * x, y0 + 4 * y), (4, 4), mv, (col * 8 + 4 * x, row * 8 + 4 * y), filter, r == 1);
                        }
                    }
                } else {
                    let mv = b.bmv[0][r];
                    let q = (i32::from(mv.row) * (2 >> ssy), i32::from(mv.col) * (2 >> ssx));
                    let w = (bw * 8) >> ssx;
                    let h = (bh * 8) >> ssy;
                    self.mc(slot, plane, (x0, y0), (w, h), q, (col * 8, row * 8), filter, r == 1);
                }
            }
        }
    }

    /// One prediction of a `w`×`h` block at `(x, y)` of a plane, with the
    /// vector `(row, col)` in sixteenths of a sample of that plane. `luma`
    /// is the luma position libvpx scales the vector's offset at.
    #[allow(clippy::too_many_arguments)]
    fn mc(
        &mut self,
        slot: usize,
        plane: usize,
        (x, y): (usize, usize),
        (w, h): (usize, usize),
        (mv_row, mv_col): (i32, i32),
        (lx, ly): (usize, usize),
        filter: &mc::Filter,
        avg: bool,
    ) {
        let Some(rp) = self.c.refs[slot].as_ref() else {
            return;
        };
        let max = self.max;
        let shift = if plane == 0 { 0 } else { rp.dims.ss_y };
        let (rw, rh) = (rp.dims.w[plane] as i32, rp.dims.h[plane] as i32);
        let v = &mut self.views[plane];
        let ds = v.stride;
        let dst = v.at(y, x);
        if let Some(sc_) = rp.scale {
            // libvpx `dec_build_inter_predictors` for a scaled reference.
            rp.buf.wait(usize::MAX);
            let (data, stride) = rp.buf.plane(plane);
            let col = sc(mv_col, sc_.x_fp) + (((i64::from(lx as i32) << 4) * i64::from(sc_.x_fp) >> 14) as i32 & 15);
            let row = sc(mv_row, sc_.y_fp) + (((i64::from(ly as i32) << 4) * i64::from(sc_.y_fp) >> 14) as i32 & 15);
            let x0 = sc(x as i32, sc_.x_fp) + (col >> 4);
            let y0 = sc(y as i32, sc_.y_fp) + (row >> 4);
            let get = |xx: i32, yy: i32| -> P {
                let xx = xx.clamp(0, rw - 1) as usize;
                let yy = yy.clamp(0, rh - 1) as usize;
                data[yy * stride + xx]
            };
            mc::predict_scaled(dst, ds, get, (w, h), (x0, y0), (col & 15, row & 15), (sc_.x_step, sc_.y_step), filter, max, avg);
            return;
        }
        let mx = (mv_col & 15) as usize;
        let my = (mv_row & 15) as usize;
        let bx = x as i32 + (mv_col >> 4);
        let by = y as i32 + (mv_row >> 4);
        let (l, r) = if mx != 0 { (3, 4) } else { (0, 0) };
        let (t, bo) = if my != 0 { (3, 4) } else { (0, 0) };
        // The reference rows this block reads (the edge copy below reads 4
        // past the block whatever the fraction), and one more for the
        // kernels' loads past the end of a row, must be final.
        let last = (by + h as i32 + 4).clamp(1, rh) as usize;
        rp.buf.wait((last + 1) << shift);
        let (data, stride) = rp.buf.plane(plane);
        if bx - l >= 0 && by - t >= 0 && bx + w as i32 + r <= rw && by + h as i32 + bo <= rh {
            let at = by as usize * stride + bx as usize;
            mc::predict(dst, ds, data, at, stride, (w, h), (mx, my), filter, max, avg, &mut self.s.mc);
        } else {
            // Edge emulation: the window held to the reference.
            let ew = w + 7;
            let eh = h + 7;
            let emu = &mut self.s.emu;
            if emu.len() < ew * eh + 32 {
                emu.resize(ew * eh + 32, P::default());
            }
            // Columns of the window inside the picture, copied whole; the
            // rest repeat the edge samples.
            let x0 = bx - 3;
            let lo = (-x0).clamp(0, ew as i32) as usize;
            let hi = (rw - x0).clamp(lo as i32, ew as i32) as usize;
            for j in 0..eh {
                let yy = (by - 3 + j as i32).clamp(0, rh - 1) as usize;
                let row = &data[yy * stride..yy * stride + rw as usize];
                let out = &mut emu[j * ew..j * ew + ew];
                if hi > lo {
                    let from = (x0 + lo as i32) as usize;
                    out[lo..hi].copy_from_slice(&row[from..from + hi - lo]);
                }
                out[..lo].fill(row[0]);
                out[hi..].fill(row[rw as usize - 1]);
            }
            mc::predict(dst, ds, emu, 3 * ew + 3, ew, (w, h), (mx, my), filter, max, avg, &mut self.s.mc);
        }
    }
}

/// The vector of sub-block `i` of a plane in sixteenths, libvpx
/// `average_split_mvs` (with its sequential sub-block index for 4:2:2).
fn split_mv(b: &BlockRec, plane: usize, i: usize, r: usize, ssx: usize, ssy: usize) -> (i32, i32) {
    let q2 = |v: i32| (if v < 0 { v - 1 } else { v + 1 }) / 2;
    let q4 = |v: i32| (if v < 0 { v - 2 } else { v + 2 }) / 4;
    let get = |k: usize| b.bmv[k.min(3)][r];
    let (row, col) = if plane == 0 || (ssx == 0 && ssy == 0) {
        let m = get(i);
        (i32::from(m.row), i32::from(m.col))
    } else if ssx == 1 && ssy == 1 {
        let (mut row, mut col) = (0, 0);
        for k in 0..4 {
            row += i32::from(b.bmv[k][r].row);
            col += i32::from(b.bmv[k][r].col);
        }
        (q4(row), q4(col))
    } else {
        let other = if ssy == 1 { i + 2 } else { i + 1 };
        let (a, c) = (get(i), get(other));
        (
            q2(i32::from(a.row) + i32::from(c.row)),
            q2(i32::from(a.col) + i32::from(c.col)),
        )
    };
    let (ssx, ssy) = if plane == 0 { (0, 0) } else { (ssx, ssy) };
    (row * (2 >> ssy), col * (2 >> ssx))
}

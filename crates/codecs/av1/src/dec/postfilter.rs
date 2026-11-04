//! Frame-level in-loop filters: deblocking (7.14), CDEF (7.15), super
//! resolution (7.16) and loop restoration (7.17).

use crate::dsp::Pixel;
use crate::dsp::cdef::{
    CDEF_BORDER, adjust_luma_pri_strength, cdef_damping, cdef_direction, cdef_filter,
    cdef_pad_block, cdef_uv_dir,
};
use crate::dsp::lf::{filter_horizontal_edge, filter_vertical_edge};
use crate::dsp::lr::{LrScratch, LrUnit, lr_plane};
use crate::dsp::superres::upscale_plane;

use super::frame::{FrameState, Picture};
use super::obu::{FrameHeader, SequenceHeader};
use super::types::{
    GLOBAL_GLOBALMV, GLOBALMV, INTRA_FRAME, MiInfo, NEARESTMV, NONE, block_height, block_width,
    plane_residual_size, tx_h, tx_w,
};

/// Filter level (7.14.5) of the block `m` for loop filter index `i`
/// (0, 1: luma vertical / horizontal edges, 2, 3: U, V), with the block's
/// `delta_lf` value `delta_lf`.
fn filter_level_of(hdr: &FrameHeader, m: &MiInfo, i: usize, delta_lf: i32) -> u8 {
    level_for(
        hdr,
        i,
        delta_lf,
        m.segment_id as usize,
        m.ref_frame[0],
        mode_type(m),
    )
}

#[inline]
fn mode_type(m: &MiInfo) -> usize {
    usize::from(m.y_mode >= NEARESTMV && m.y_mode != GLOBALMV && m.y_mode != GLOBAL_GLOBALMV)
}

fn level_for(
    hdr: &FrameHeader,
    i: usize,
    delta_lf: i32,
    seg: usize,
    rf: i8,
    mode_type: usize,
) -> u8 {
    let base = (delta_lf + i32::from(hdr.loop_filter.level[i])).clamp(0, 63);
    let mut lvl = base;
    let feature = 1 + i;
    if hdr.seg_feature_active(seg, feature) {
        lvl = (lvl + i32::from(hdr.segmentation.feature_data[seg][feature])).clamp(0, 63);
    }
    if hdr.loop_filter.delta_enabled {
        let n_shift = lvl >> 5;
        if rf == INTRA_FRAME {
            lvl += i32::from(hdr.loop_filter.ref_deltas[0]) << n_shift;
        } else if rf > INTRA_FRAME {
            lvl += (i32::from(hdr.loop_filter.ref_deltas[rf as usize]) << n_shift)
                + (i32::from(hdr.loop_filter.mode_deltas[mode_type]) << n_shift);
        }
        lvl = lvl.clamp(0, 63);
    }
    lvl as u8
}

/// Number of `ref_frame[0]` classes in [`LfPlane::lut`]: 0..=7, and one for
/// every negative value.
const LF_REFS: usize = 9;

/// Edge decisions for one plane and pass of the loop filter (7.14.2 -
/// 7.14.5).
struct LfPlane<'a> {
    hdr: &'a FrameHeader,
    fs: &'a FrameState,
    tx: &'a [u8],
    plane: usize,
    pass: usize,
    sx: usize,
    sy: usize,
    /// Loop filter index `i` of 7.14.5.
    li: usize,
    /// Levels for `delta_lf == 0`, by `[segment][ref class][mode type]`.
    lut: [[[u8; 2]; LF_REFS]; 8],
}

impl<'a> LfPlane<'a> {
    fn new(
        hdr: &'a FrameHeader,
        fs: &'a FrameState,
        plane: usize,
        pass: usize,
        sx: usize,
        sy: usize,
    ) -> Self {
        let li = if plane == 0 { pass } else { plane + 1 };
        let mut lut = [[[0u8; 2]; LF_REFS]; 8];
        for (seg, t) in lut.iter_mut().enumerate() {
            for (rc, t) in t.iter_mut().enumerate() {
                let rf = if rc < 8 { rc as i8 } else { NONE };
                for (mt, t) in t.iter_mut().enumerate() {
                    *t = level_for(hdr, li, 0, seg, rf, mt);
                }
            }
        }
        Self {
            hdr,
            fs,
            tx: &fs.lf_tx[plane],
            plane,
            pass,
            sx,
            sy,
            li,
            lut,
        }
    }

    #[inline]
    fn level(&self, m: &MiInfo) -> u8 {
        let d = if self.hdr.delta_lf_multi {
            m.delta_lf[self.li]
        } else {
            m.delta_lf[0]
        };
        if d == 0 {
            let rf = m.ref_frame[0];
            let rc = if rf >= 0 { rf as usize } else { 8 };
            self.lut[m.segment_id as usize & 7][rc.min(8)][mode_type(m)]
        } else {
            filter_level_of(self.hdr, m, self.li, i32::from(d))
        }
    }

    /// `(filter_size, lvl)` of the edge at the 4x4 unit `(row, col)` (luma
    /// 4x4 units, multiples of the subsampling step); `lvl == 0` when the
    /// edge is not filtered. `filter_size` follows [`crate::dsp::lf`].
    #[inline]
    fn edge(&self, row: usize, col: usize) -> (u8, u8) {
        let x = col * 4;
        let y = row * 4;
        let hdr = self.hdr;
        if x >= hdr.frame_width as usize
            || y >= hdr.frame_height as usize
            || (self.pass == 0 && x == 0)
            || (self.pass == 1 && y == 0)
        {
            return (0, 0);
        }
        let (sx, sy) = (self.sx, self.sy);
        let fs = self.fs;
        let st = fs.mi_stride;
        let r = row | sy;
        let c = col | sx;
        let xp = x >> sx;
        let yp = y >> sy;
        let (prev_r, prev_c) = if self.pass == 0 {
            (r, c - (1 << sx))
        } else {
            (r - (1 << sy), c)
        };
        let m = fs.mi_at(r, c);
        let tx = self.tx[(r >> sy) * st + (c >> sx)];
        let psz = plane_residual_size(m.bsize, sx != 0, sy != 0);
        let (is_block_edge, is_tx_edge) = if self.pass == 0 {
            (
                xp.is_multiple_of(block_width(psz)),
                xp.is_multiple_of(tx_w(tx)),
            )
        } else {
            (
                yp.is_multiple_of(block_height(psz)),
                yp.is_multiple_of(tx_h(tx)),
            )
        };
        let is_intra = m.ref_frame[0] <= INTRA_FRAME;
        if !(is_tx_edge && (is_block_edge || !m.skip || is_intra)) {
            return (0, 0);
        }
        let prev_tx = self.tx[(prev_r >> sy) * st + (prev_c >> sx)];
        let base = if self.pass == 0 {
            tx_w(prev_tx).min(tx_w(tx))
        } else {
            tx_h(prev_tx).min(tx_h(tx))
        };
        let filter_size = if self.plane == 0 {
            base.min(16)
        } else if base.min(8) == 8 {
            6
        } else {
            4
        };
        let mut lvl = self.level(m);
        if lvl == 0 {
            lvl = self.level(fs.mi_at(prev_r, prev_c));
        }
        (filter_size as u8, lvl)
    }
}

/// Plane rows filtered by one vertical-edge pass of [`loop_filter`] before
/// moving to the next strip (keeps the strip in cache).
const LF_STRIP_UNITS: usize = 16;

/// Loop filter process (7.14).
///
/// Edges of one pass never overlap (an edge of size `s` has blocks at least
/// `s` samples wide / tall on both sides), so the edges of a pass may be
/// filtered in any order. Runs of adjacent 4-sample edges with the same
/// filter size and level go to the kernel in one call.
pub fn loop_filter<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    pic: &mut Picture<P>,
) {
    if hdr.loop_filter.level[0] == 0 && hdr.loop_filter.level[1] == 0 {
        return;
    }
    let bd = pic.bd;
    let sharp = hdr.loop_filter.sharpness;
    for plane in 0..seq.num_planes() {
        if plane > 0 && hdr.loop_filter.level[1 + plane] == 0 {
            continue;
        }
        let (sx, sy) = if plane > 0 {
            (
                usize::from(seq.subsampling_x),
                usize::from(seq.subsampling_y),
            )
        } else {
            (0, 0)
        };
        // Units (4x4 in the plane) per row / column.
        let nc = fs.mi_cols.div_ceil(1 << sx);
        let nr = fs.mi_rows.div_ceil(1 << sy);
        let p = &mut pic.planes[plane];
        let stride = p.stride;
        // Vertical edges, column by column within each strip of rows.
        let lp = LfPlane::new(hdr, fs, plane, 0, sx, sy);
        for strip in (0..nr).step_by(LF_STRIP_UNITS) {
            let end = (strip + LF_STRIP_UNITS).min(nr);
            for uc in 1..nc {
                let mut run = (0u8, 0u8);
                let mut start = strip;
                for ur in strip..=end {
                    let e = if ur < end {
                        lp.edge(ur << sy, uc << sx)
                    } else {
                        (0, 0)
                    };
                    if e != run {
                        if run.1 != 0 {
                            filter_vertical_edge(
                                &mut p.data,
                                stride,
                                uc * 4,
                                start * 4,
                                (ur - start) * 4,
                                usize::from(run.0),
                                run.1,
                                sharp,
                                bd,
                            );
                        }
                        run = e;
                        start = ur;
                    }
                }
            }
        }
        // Horizontal edges, row by row.
        let lp = LfPlane::new(hdr, fs, plane, 1, sx, sy);
        for ur in 1..nr {
            let mut run = (0u8, 0u8);
            let mut start = 0;
            for uc in 0..=nc {
                let e = if uc < nc {
                    lp.edge(ur << sy, uc << sx)
                } else {
                    (0, 0)
                };
                if e != run {
                    if run.1 != 0 {
                        filter_horizontal_edge(
                            &mut p.data,
                            stride,
                            start * 4,
                            ur * 4,
                            (uc - start) * 4,
                            usize::from(run.0),
                            run.1,
                            sharp,
                            bd,
                        );
                    }
                    run = e;
                    start = uc;
                }
            }
        }
    }
}

/// Side, in samples, of the padded copy of one 64x64 luma area in [`cdef`].
const CDEF_PAD_STRIDE: usize = 64 + 2 * CDEF_BORDER;

/// CDEF process (7.15). `src` is the deblocked frame; `dst` starts as a copy of it.
pub fn cdef<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    src: &Picture<P>,
    dst: &mut Picture<P>,
) {
    let bd = src.bd;
    let coeff_shift = bd - 8;
    let num_planes = seq.num_planes();
    let (ssx, ssy) = (seq.subsampling_x, seq.subsampling_y);
    let y_damping = cdef_damping(i32::from(hdr.cdef.damping), bd, false);
    let uv_damping = cdef_damping(i32::from(hdr.cdef.damping), bd, true);
    // One padded copy per plane of the current 64x64 area, built on first use.
    let mut pads: [Vec<u16>; 3] =
        core::array::from_fn(|_| vec![0u16; CDEF_PAD_STRIDE * CDEF_PAD_STRIDE]);
    let lp = &src.planes[0];
    for sbr in (0..fs.mi_rows).step_by(16) {
        for sbc in (0..fs.mi_cols).step_by(16) {
            let idx = fs.cdef_idx[(sbr >> 4) * fs.cdef_stride + (sbc >> 4)];
            if idx < 0 {
                continue;
            }
            let idx = idx as usize;
            let y_pri0 = i32::from(hdr.cdef.y_pri[idx]) << coeff_shift;
            let y_sec = i32::from(hdr.cdef.y_sec[idx]) << coeff_shift;
            let uv_pri = i32::from(hdr.cdef.uv_pri[idx]) << coeff_shift;
            let uv_sec = i32::from(hdr.cdef.uv_sec[idx]) << coeff_shift;
            let luma_on = y_pri0 != 0 || y_sec != 0;
            let chroma_on = num_planes > 1 && (uv_pri != 0 || uv_sec != 0);
            if !luma_on && !chroma_on {
                continue;
            }
            let need_dir = y_pri0 != 0 || (chroma_on && uv_pri != 0);
            let mut padded = [false; 3];
            let r_end = (sbr + 16).min(fs.mi_rows);
            let c_end = (sbc + 16).min(fs.mi_cols);
            for r in (sbr..r_end).step_by(2) {
                for c in (sbc..c_end).step_by(2) {
                    let rr = (r + 1).min(fs.mi_rows - 1);
                    let cc = (c + 1).min(fs.mi_cols - 1);
                    let skip = fs.mi_at(r, c).skip
                        && fs.mi_at(rr, c).skip
                        && fs.mi_at(r, cc).skip
                        && fs.mi_at(rr, cc).skip;
                    if skip {
                        continue;
                    }
                    let (ydir, var) = if need_dir {
                        cdef_direction(&lp.data[r * 4 * lp.stride + c * 4..], lp.stride, bd)
                    } else {
                        (0, 0)
                    };
                    let blk = CdefBlock { sbr, sbc, r, c, bd };
                    if luma_on {
                        let pri = adjust_luma_pri_strength(y_pri0, var);
                        let dir = if y_pri0 == 0 { 0 } else { ydir };
                        if pri != 0 || y_sec != 0 {
                            cdef_plane_block(
                                src,
                                dst,
                                0,
                                &blk,
                                (0, 0),
                                &mut pads[0],
                                &mut padded[0],
                                fs,
                                (pri, y_sec, dir, y_damping),
                            );
                        }
                    }
                    if chroma_on {
                        let dir = if uv_pri == 0 {
                            0
                        } else {
                            cdef_uv_dir(ssx, ssy, ydir)
                        };
                        for plane in 1..3 {
                            cdef_plane_block(
                                src,
                                dst,
                                plane,
                                &blk,
                                (u32::from(ssx), u32::from(ssy)),
                                &mut pads[plane],
                                &mut padded[plane],
                                fs,
                                (uv_pri, uv_sec, dir, uv_damping),
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Position of one 8x8 CDEF block: its 64x64 area `(sbr, sbc)` and its own
/// `(r, c)`, both in luma 4x4 units.
struct CdefBlock {
    sbr: usize,
    sbc: usize,
    r: usize,
    c: usize,
    bd: u32,
}

/// Filters one 8x8 block (subsampled for chroma) of `plane`. `strength` is
/// `(pri, sec, dir, damping)`. Pads the 64x64 area holding the block into
/// `pad` first, unless `padded`.
fn cdef_plane_block<P: Pixel>(
    src: &Picture<P>,
    dst: &mut Picture<P>,
    plane: usize,
    blk: &CdefBlock,
    (sx, sy): (u32, u32),
    pad: &mut [u16],
    padded: &mut bool,
    fs: &FrameState,
    (pri, sec, dir, damping): (i32, i32, u8, i32),
) {
    let ps = CDEF_PAD_STRIDE;
    let sp = &src.planes[plane];
    let sb_x = (blk.sbc * 4) >> sx;
    let sb_y = (blk.sbr * 4) >> sy;
    if !*padded {
        let avail_w = (fs.mi_cols * 4) >> sx;
        let avail_h = (fs.mi_rows * 4) >> sy;
        cdef_pad_block(
            pad,
            ps,
            &sp.data,
            sp.stride,
            sb_x,
            sb_y,
            64 >> sx,
            64 >> sy,
            avail_w,
            avail_h,
        );
        *padded = true;
    }
    let w = 8 >> sx;
    let h = 8 >> sy;
    let x0 = (blk.c * 4) >> sx;
    let y0 = (blk.r * 4) >> sy;
    let dp = &mut dst.planes[plane];
    let ds = dp.stride;
    cdef_filter(
        &mut dp.data[y0 * ds + x0..],
        ds,
        &pad[(y0 - sb_y) * ps + (x0 - sb_x)..],
        ps,
        w,
        h,
        pri,
        sec,
        dir,
        damping,
        blk.bd,
    );
}

/// Upscaling process (7.16) of every plane.
pub fn superres<P: Pixel>(seq: &SequenceHeader, hdr: &FrameHeader, src: &Picture<P>) -> Picture<P> {
    let mut out = Picture::new(
        hdr.upscaled_width as usize,
        hdr.frame_height as usize,
        seq.num_planes(),
        seq.subsampling_x,
        seq.subsampling_y,
        src.bd,
    );
    for p in 0..seq.num_planes() {
        let sx = p > 0 && seq.subsampling_x;
        let sp = &src.planes[p];
        let rows = sp.alloc_h.min(out.planes[p].alloc_h);
        let dp = &mut out.planes[p];
        let ds = dp.stride;
        upscale_plane(
            &mut dp.data,
            ds,
            &sp.data,
            sp.stride,
            hdr.frame_width as usize,
            hdr.upscaled_width as usize,
            rows,
            sx,
            src.bd,
        );
    }
    out
}

/// Loop restoration (7.17). `cdef` is UpscaledCdefFrame, `deblocked` is
/// UpscaledCurrFrame; the result replaces `out` (initially a copy of `cdef`).
pub fn loop_restoration<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    cdef: &Picture<P>,
    deblocked: &Picture<P>,
    out: &mut Picture<P>,
    scratch: &mut LrScratch,
) {
    for p in 0..seq.num_planes() {
        if hdr.lr.frame_type[p] == 0 {
            continue;
        }
        let (sx, sy) = if p > 0 {
            (seq.subsampling_x, seq.subsampling_y)
        } else {
            (false, false)
        };
        let pw = (hdr.upscaled_width as usize + usize::from(sx)) >> u32::from(sx);
        let ph = (hdr.frame_height as usize + usize::from(sy)) >> u32::from(sy);
        let units: Vec<LrUnit> = fs.lr[p]
            .iter()
            .map(|u| LrUnit {
                kind: u.kind,
                wiener: [
                    [
                        i32::from(u.wiener[0][0]),
                        i32::from(u.wiener[0][1]),
                        i32::from(u.wiener[0][2]),
                    ],
                    [
                        i32::from(u.wiener[1][0]),
                        i32::from(u.wiener[1][1]),
                        i32::from(u.wiener[1][2]),
                    ],
                ],
                sgr_set: u.sgr_set,
                sgr_xqd: [i32::from(u.sgr_xqd[0]), i32::from(u.sgr_xqd[1])],
            })
            .collect();
        let stride = cdef.planes[p].stride;
        lr_plane(
            &mut out.planes[p].data,
            &cdef.planes[p].data,
            &deblocked.planes[p].data,
            stride,
            pw,
            ph,
            hdr.lr.unit_size[p] as usize,
            fs.lr_cols[p],
            fs.lr_rows[p],
            &units,
            sx,
            sy,
            cdef.bd,
            scratch,
        );
    }
}

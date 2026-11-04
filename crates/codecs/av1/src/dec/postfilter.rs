//! Frame-level in-loop filters: deblocking (7.14), CDEF (7.15), super
//! resolution (7.16) and loop restoration (7.17).

use crate::dsp::cdef::{
    CDEF_BORDER, adjust_luma_pri_strength, cdef_damping, cdef_direction, cdef_filter,
    cdef_pad_block, cdef_uv_dir,
};
use crate::dsp::lf::{filter_horizontal_edge, filter_vertical_edge};
use crate::dsp::lr::{LrScratch, LrUnit, lr_plane};
use crate::dsp::superres::upscale_plane;
use crate::dsp::Pixel;

use super::frame::{FrameState, Picture};
use super::obu::{FrameHeader, SequenceHeader};
use super::types::{
    GLOBAL_GLOBALMV, GLOBALMV, INTRA_FRAME, NEARESTMV, block_height, block_width,
    plane_residual_size, tx_h, tx_w,
};

fn filter_level(hdr: &FrameHeader, fs: &FrameState, row: usize, col: usize, plane: usize, pass: usize) -> u8 {
    let m = fs.mi_at(row, col);
    let i = if plane == 0 { pass } else { plane + 1 };
    let delta_lf = if hdr.delta_lf_multi {
        i32::from(m.delta_lf[i])
    } else {
        i32::from(m.delta_lf[0])
    };
    let base = (delta_lf + i32::from(hdr.loop_filter.level[i])).clamp(0, 63);
    let mut lvl = base;
    let seg = m.segment_id as usize;
    let feature = 1 + i;
    if hdr.seg_feature_active(seg, feature) {
        lvl = (lvl + i32::from(hdr.segmentation.feature_data[seg][feature])).clamp(0, 63);
    }
    if hdr.loop_filter.delta_enabled {
        let n_shift = lvl >> 5;
        let rf = m.ref_frame[0];
        if rf == INTRA_FRAME {
            lvl += i32::from(hdr.loop_filter.ref_deltas[0]) << n_shift;
        } else if rf > INTRA_FRAME {
            let mode_type = usize::from(
                m.y_mode >= NEARESTMV && m.y_mode != GLOBALMV && m.y_mode != GLOBAL_GLOBALMV,
            );
            lvl += (i32::from(hdr.loop_filter.ref_deltas[rf as usize]) << n_shift)
                + (i32::from(hdr.loop_filter.mode_deltas[mode_type]) << n_shift);
        }
        lvl = lvl.clamp(0, 63);
    }
    lvl as u8
}

/// Loop filter process (7.14).
pub fn loop_filter<P: Pixel>(seq: &SequenceHeader, hdr: &FrameHeader, fs: &FrameState, pic: &mut Picture<P>) {
    if hdr.loop_filter.level[0] == 0 && hdr.loop_filter.level[1] == 0 {
        return;
    }
    let bd = pic.bd;
    let sharp = hdr.loop_filter.sharpness;
    let st = fs.mi_stride;
    for plane in 0..seq.num_planes() {
        if plane > 0 && hdr.loop_filter.level[1 + plane] == 0 {
            continue;
        }
        let (sx, sy) = if plane > 0 {
            (usize::from(seq.subsampling_x), usize::from(seq.subsampling_y))
        } else {
            (0, 0)
        };
        let p = &mut pic.planes[plane];
        let stride = p.stride;
        for pass in 0..2 {
            let row_step = 1 << sy;
            let col_step = 1 << sx;
            let mut row = 0;
            while row < fs.mi_rows {
                let mut col = 0;
                while col < fs.mi_cols {
                    let x = col * 4;
                    let y = row * 4;
                    let on_screen = !(x >= hdr.frame_width as usize
                        || y >= hdr.frame_height as usize
                        || (pass == 0 && x == 0)
                        || (pass == 1 && y == 0));
                    if on_screen {
                        let r = row | sy;
                        let c = col | sx;
                        let xp = x >> sx;
                        let yp = y >> sy;
                        let (prev_r, prev_c) = if pass == 0 {
                            (r, c - (1 << sx))
                        } else {
                            (r - (1 << sy), c)
                        };
                        let m = fs.mi_at(r, c);
                        let tx = fs.lf_tx[plane][(r >> sy) * st + (c >> sx)];
                        let psz = plane_residual_size(m.bsize, sx != 0, sy != 0);
                        let is_intra = m.ref_frame[0] <= INTRA_FRAME;
                        let prev_tx = fs.lf_tx[plane][(prev_r >> sy) * st + (prev_c >> sx)];
                        let (is_block_edge, is_tx_edge, base) = if pass == 0 {
                            (
                                xp % block_width(psz) == 0,
                                xp % tx_w(tx) == 0,
                                tx_w(prev_tx).min(tx_w(tx)),
                            )
                        } else {
                            (
                                yp % block_height(psz) == 0,
                                yp % tx_h(tx) == 0,
                                tx_h(prev_tx).min(tx_h(tx)),
                            )
                        };
                        let apply = is_tx_edge && (is_block_edge || !m.skip || is_intra);
                        if apply {
                            let filter_size = if plane == 0 {
                                base.min(16)
                            } else if base.min(8) == 8 {
                                6
                            } else {
                                4
                            };
                            let mut lvl = filter_level(hdr, fs, r, c, plane, pass);
                            if lvl == 0 {
                                lvl = filter_level(hdr, fs, prev_r, prev_c, plane, pass);
                            }
                            if lvl > 0 {
                                if pass == 0 {
                                    filter_vertical_edge(&mut p.data, stride, xp, yp, 4, filter_size, lvl, sharp, bd);
                                } else {
                                    filter_horizontal_edge(&mut p.data, stride, xp, yp, 4, filter_size, lvl, sharp, bd);
                                }
                            }
                        }
                    }
                    col += col_step;
                }
                row += row_step;
            }
        }
    }
}

/// CDEF process (7.15). `src` is the deblocked frame; `dst` starts as a copy of it.
pub fn cdef<P: Pixel>(seq: &SequenceHeader, hdr: &FrameHeader, fs: &FrameState, src: &Picture<P>, dst: &mut Picture<P>) {
    let bd = src.bd;
    let coeff_shift = bd - 8;
    let num_planes = seq.num_planes();
    let (ssx, ssy) = (seq.subsampling_x, seq.subsampling_y);
    let mut pad = vec![0u16; 12 * 12];
    let pstride = 8 + 2 * CDEF_BORDER;
    let mut r = 0;
    while r < fs.mi_rows {
        let mut c = 0;
        while c < fs.mi_cols {
            let idx = fs.cdef_idx[(r >> 4) * fs.cdef_stride + (c >> 4)];
            if idx >= 0 {
                let rr = (r + 1).min(fs.mi_rows - 1);
                let cc = (c + 1).min(fs.mi_cols - 1);
                let skip = fs.mi_at(r, c).skip
                    && fs.mi_at(rr, c).skip
                    && fs.mi_at(r, cc).skip
                    && fs.mi_at(rr, cc).skip;
                if !skip {
                    let idx = idx as usize;
                    let lp = &src.planes[0];
                    let (ydir, var) =
                        cdef_direction(&lp.data[r * 4 * lp.stride + c * 4..], lp.stride, bd);
                    let pri0 = i32::from(hdr.cdef.y_pri[idx]) << coeff_shift;
                    let sec = i32::from(hdr.cdef.y_sec[idx]) << coeff_shift;
                    let dir = if pri0 == 0 { 0 } else { ydir };
                    let pri = adjust_luma_pri_strength(pri0, var);
                    let damping = cdef_damping(i32::from(hdr.cdef.damping), bd, false);
                    if pri != 0 || sec != 0 {
                        cdef_plane_block(src, dst, 0, r, c, 0, 0, &mut pad, pstride, fs, pri, sec, dir, damping, bd);
                    }
                    if num_planes > 1 {
                        let pri = i32::from(hdr.cdef.uv_pri[idx]) << coeff_shift;
                        let sec = i32::from(hdr.cdef.uv_sec[idx]) << coeff_shift;
                        let dir = if pri == 0 { 0 } else { cdef_uv_dir(ssx, ssy, ydir) };
                        let damping = cdef_damping(i32::from(hdr.cdef.damping), bd, true);
                        if pri != 0 || sec != 0 {
                            for plane in 1..3 {
                                cdef_plane_block(
                                    src, dst, plane, r, c, u32::from(ssx), u32::from(ssy), &mut pad, pstride, fs,
                                    pri, sec, dir, damping, bd,
                                );
                            }
                        }
                    }
                }
            }
            c += 2;
        }
        r += 2;
    }
}

fn cdef_plane_block<P: Pixel>(
    src: &Picture<P>,
    dst: &mut Picture<P>,
    plane: usize,
    r: usize,
    c: usize,
    sx: u32,
    sy: u32,
    pad: &mut [u16],
    pstride: usize,
    fs: &FrameState,
    pri: i32,
    sec: i32,
    dir: u8,
    damping: i32,
    bd: u32,
) {
    let w = 8 >> sx;
    let h = 8 >> sy;
    let x0 = (c * 4) >> sx;
    let y0 = (r * 4) >> sy;
    let sp = &src.planes[plane];
    let avail_w = (fs.mi_cols * 4) >> sx;
    let avail_h = (fs.mi_rows * 4) >> sy;
    cdef_pad_block(pad, pstride, &sp.data, sp.stride, x0, y0, w, h, avail_w, avail_h);
    let dp = &mut dst.planes[plane];
    let ds = dp.stride;
    cdef_filter(&mut dp.data[y0 * ds + x0..], ds, pad, pstride, w, h, pri, sec, dir, damping, bd);
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
                    [i32::from(u.wiener[0][0]), i32::from(u.wiener[0][1]), i32::from(u.wiener[0][2])],
                    [i32::from(u.wiener[1][0]), i32::from(u.wiener[1][1]), i32::from(u.wiener[1][2])],
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

//! Frame-level in-loop filters: deblocking (7.14), CDEF (7.15), super
//! resolution (7.16) and loop restoration (7.17).

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::dsp::Pixel;
use crate::dsp::cdef::{
    CDEF_PAD_STRIDE, adjust_luma_pri_strength, cdef_damping, cdef_direction, cdef_filter,
    cdef_pad_block, cdef_uv_dir,
};
use crate::dsp::lf::{filter_horizontal_edge, filter_vertical_edge};
use crate::dsp::lr::{LrPlane, LrScratch, LrUnit, lr_planes, lr_planes_stripes};
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
    /// Units (4x4 in the plane) per row of the plane.
    nc: usize,
    /// Units whose top-left luma sample is inside the frame (`onScreen`).
    nc_vis: usize,
    nr_vis: usize,
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
        let nc = fs.mi_cols.div_ceil(1 << sx);
        let nr = fs.mi_rows.div_ceil(1 << sy);
        Self {
            hdr,
            fs,
            tx: &fs.lf_tx[plane],
            plane,
            pass,
            sx,
            sy,
            nc,
            nc_vis: (hdr.frame_width as usize).div_ceil(4 << sx).min(nc),
            nr_vis: (hdr.frame_height as usize).div_ceil(4 << sy).min(nr),
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

    /// `(filter_size, lvl)` of the edges of the unit row `ur` (plane 4x4
    /// units) into `out[..nc]`; `lvl == 0` where the edge is not filtered.
    /// `filter_size` follows [`crate::dsp::lf`].
    fn row_edges(&self, ur: usize, out: &mut [(u8, u8)]) {
        let out = &mut out[..self.nc];
        out.fill((0, 0));
        let pass0 = self.pass == 0;
        if ur >= self.nr_vis || (!pass0 && ur == 0) {
            return;
        }
        let (sx, sy) = (self.sx, self.sy);
        let fs = self.fs;
        let st = fs.mi_stride;
        let tx_row = &self.tx[ur * st..ur * st + self.nc];
        let prev_row = if pass0 {
            tx_row
        } else {
            &self.tx[(ur - 1) * st..(ur - 1) * st + self.nc]
        };
        // MiInfo row holding the unit's bottom-right 4x4 for chroma, and the
        // row above it for the horizontal edges.
        let r = (ur << sy) | sy;
        let mi_row = &fs.mi[r * st..(r + 1) * st];
        let mi_prev = if pass0 {
            mi_row
        } else {
            let pr = r - (1 << sy);
            &fs.mi[pr * st..(pr + 1) * st]
        };
        let yp = ur * 4;
        for uc in usize::from(pass0)..self.nc_vis {
            let tx = tx_row[uc];
            let (t, pos) = if pass0 {
                (tx_w(tx), uc * 4)
            } else {
                (tx_h(tx), yp)
            };
            if pos & (t - 1) != 0 {
                continue;
            }
            let c = (uc << sx) | sx;
            let m = &mi_row[c];
            if m.skip && m.ref_frame[0] > INTRA_FRAME {
                let psz = plane_residual_size(m.bsize, sx != 0, sy != 0);
                let bs = if pass0 {
                    block_width(psz)
                } else {
                    block_height(psz)
                };
                if pos & (bs - 1) != 0 {
                    continue;
                }
            }
            let (base, pm) = if pass0 {
                (tx_w(tx_row[uc - 1]).min(t), &mi_row[c - (1 << sx)])
            } else {
                (tx_h(prev_row[uc]).min(t), &mi_prev[c])
            };
            let filter_size = if self.plane == 0 {
                base.min(16)
            } else if base >= 8 {
                6
            } else {
                4
            };
            let mut lvl = self.level(m);
            if lvl == 0 {
                lvl = self.level(pm);
            }
            out[uc] = (filter_size as u8, lvl);
        }
    }
}

/// Calls `f(start, len, filter_size, lvl)` for every maximal run of equal
/// filtered edges in `edges`.
#[inline]
fn edge_runs(edges: &[(u8, u8)], mut f: impl FnMut(usize, usize, u8, u8)) {
    let mut run = (0u8, 0u8);
    let mut start = 0;
    for (i, &e) in edges.iter().chain(core::iter::once(&(0, 0))).enumerate() {
        if e != run {
            if run.1 != 0 {
                f(start, i - start, run.0, run.1);
            }
            run = e;
            start = i;
        }
    }
}

/// Plane rows filtered by one vertical-edge pass of [`loop_filter`] before
/// moving to the next strip (keeps the strip in cache); also the alignment
/// of the per-thread row ranges.
const LF_STRIP_UNITS: usize = 16;

/// Splits `0..n` into `t` consecutive ranges with boundaries at multiples of
/// `align` (some may be empty). Returns the `t + 1` boundaries.
fn split_rows(n: usize, t: usize, align: usize) -> Vec<usize> {
    (0..=t)
        .map(|g| {
            if g == t {
                n
            } else {
                (n * g / t) / align * align
            }
        })
        .collect()
}

/// Splits each plane in `planes` (with its stride) into row chunks at the
/// sample rows `bounds[p][1..t]`; returns, per chunk index, the chunks of
/// every plane. The last chunk holds the rest of the plane.
fn split_planes<'a, P>(
    planes: Vec<(&'a mut [P], usize, &[usize])>,
    t: usize,
) -> Vec<Vec<&'a mut [P]>> {
    let mut out: Vec<Vec<&'a mut [P]>> = (0..t).map(|_| Vec::new()).collect();
    for (data, stride, bounds) in planes {
        let mut rest = data;
        let mut row = 0;
        for (g, o) in out.iter_mut().enumerate() {
            if g + 1 == t {
                o.push(core::mem::take(&mut rest));
            } else {
                let n = ((bounds[g + 1] - row) * stride).min(rest.len());
                let (a, b) = core::mem::take(&mut rest).split_at_mut(n);
                o.push(a);
                rest = b;
                row = bounds[g + 1];
            }
        }
    }
    out
}

/// One plane's share of [`loop_filter`]: the thread's chunk of the plane
/// starts at unit row `u0` and holds unit rows `u0..u1` (and all rows after
/// them for the last chunk).
struct LfJob<'a, 'b> {
    vert: &'b LfPlane<'a>,
    horz: &'b LfPlane<'a>,
    stride: usize,
    u0: usize,
    u1: usize,
    first: bool,
    last: bool,
}

/// Unit rows of horizontal edges a chunk filters itself: the filters reach
/// at most 7 rows from the edge, so the edges within two unit rows of an
/// inner chunk boundary are left to [`loop_filter`]'s final step.
fn lf_own_rows(u0: usize, u1: usize, first: bool, last: bool) -> core::ops::Range<usize> {
    let lo = if first { u0.max(1) } else { u0 + 2 };
    let hi = if last { u1 } else { u1.saturating_sub(1) };
    lo..hi.max(lo)
}

/// Filters `job`'s rows; `data` starts at plane sample row `row0`.
fn lf_chunk<P: Pixel>(job: &LfJob<'_, '_>, data: &mut [P], row0: usize, sharp: u8, bd: u32) {
    let (u0, u1, stride) = (job.u0, job.u1, job.stride);
    let nc = job.vert.nc;
    // Vertical edges: decisions for a strip of unit rows, then runs down
    // each column.
    let mut grid = vec![(0u8, 0u8); LF_STRIP_UNITS * nc];
    let mut col = [(0u8, 0u8); LF_STRIP_UNITS];
    for strip in (u0..u1).step_by(LF_STRIP_UNITS) {
        let rows = (u1 - strip).min(LF_STRIP_UNITS);
        for i in 0..rows {
            job.vert
                .row_edges(strip + i, &mut grid[i * nc..(i + 1) * nc]);
        }
        for uc in 1..nc {
            for i in 0..rows {
                col[i] = grid[i * nc + uc];
            }
            edge_runs(&col[..rows], |start, len, size, lvl| {
                filter_vertical_edge(
                    data,
                    stride,
                    uc * 4,
                    (strip + start) * 4 - row0,
                    len * 4,
                    usize::from(size),
                    lvl,
                    sharp,
                    bd,
                );
            });
        }
    }
    // Horizontal edges whose samples stay inside the chunk.
    let mut row = vec![(0u8, 0u8); nc];
    for ur in lf_own_rows(u0, u1, job.first, job.last) {
        lf_horz_row(job.horz, ur, &mut row, data, stride, row0, sharp, bd);
    }
}

fn lf_horz_row<P: Pixel>(
    lp: &LfPlane<'_>,
    ur: usize,
    row: &mut [(u8, u8)],
    data: &mut [P],
    stride: usize,
    row0: usize,
    sharp: u8,
    bd: u32,
) {
    lp.row_edges(ur, row);
    edge_runs(row, |start, len, size, lvl| {
        filter_horizontal_edge(
            data,
            stride,
            start * 4,
            ur * 4 - row0,
            len * 4,
            usize::from(size),
            lvl,
            sharp,
            bd,
        );
    });
}

/// Loop filter process (7.14), on up to `threads` threads.
///
/// Edges of one pass never interact: an edge of size `s` has transform
/// blocks at least `s` samples wide / tall on both sides, so no edge reads
/// or writes samples another edge of the same pass writes. The edges of a
/// pass may therefore be filtered in any order. Runs of adjacent 4-sample
/// edges with the same filter size and level go to the kernel in one call.
///
/// Each plane is split into row chunks, one per thread. A thread filters the
/// vertical edges of its rows (they only touch their own row), then the
/// horizontal edges whose samples lie within its chunk. The horizontal
/// edges near the chunk boundaries are filtered last, on the whole plane.
pub fn loop_filter<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    pic: &mut Picture<P>,
    threads: usize,
) {
    if hdr.loop_filter.level[0] == 0 && hdr.loop_filter.level[1] == 0 {
        return;
    }
    let bd = pic.bd;
    let sharp = hdr.loop_filter.sharpness;
    let mut lps = Vec::new();
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
        lps.push((
            plane,
            LfPlane::new(hdr, fs, plane, 0, sx, sy),
            LfPlane::new(hdr, fs, plane, 1, sx, sy),
            fs.mi_rows.div_ceil(1 << sy),
        ));
    }
    let max_nr = lps.iter().map(|l| l.3).max().unwrap_or(0);
    let t = threads.clamp(1, max_nr.div_ceil(LF_STRIP_UNITS).max(1));
    let bounds: Vec<Vec<usize>> = lps
        .iter()
        .map(|l| split_rows(l.3, t, LF_STRIP_UNITS))
        .collect();
    let row_bounds: Vec<Vec<usize>> = bounds
        .iter()
        .map(|b| b.iter().map(|&u| u * 4).collect())
        .collect();
    let strides: Vec<usize> = lps.iter().map(|l| pic.planes[l.0].stride).collect();
    {
        let mut planes = Vec::new();
        for (p, pl) in pic.planes.iter_mut().enumerate() {
            if let Some(i) = lps.iter().position(|l| l.0 == p) {
                planes.push((&mut pl.data[..], strides[i], &row_bounds[i][..]));
            }
        }
        let chunks = split_planes(planes, t);
        let run = |g: usize, chunks: Vec<&mut [P]>| {
            for (i, data) in chunks.into_iter().enumerate() {
                let job = LfJob {
                    vert: &lps[i].1,
                    horz: &lps[i].2,
                    stride: strides[i],
                    u0: bounds[i][g],
                    u1: bounds[i][g + 1],
                    first: g == 0,
                    last: g + 1 == t,
                };
                lf_chunk(&job, data, job.u0 * 4, sharp, bd);
            }
        };
        std::thread::scope(|s| {
            let mut it = chunks.into_iter().enumerate();
            let first = it.next();
            for (g, c) in it {
                s.spawn(move || run(g, c));
            }
            if let Some((g, c)) = first {
                run(g, c);
            }
        });
    }
    // Horizontal edges next to the chunk boundaries.
    if t > 1 {
        for (i, l) in lps.iter().enumerate() {
            let nr = l.3;
            let mut own = vec![false; nr];
            for g in 0..t {
                for ur in lf_own_rows(bounds[i][g], bounds[i][g + 1], g == 0, g + 1 == t) {
                    own[ur] = true;
                }
            }
            let mut row = vec![(0u8, 0u8); l.2.nc];
            let data = &mut pic.planes[l.0].data[..];
            for ur in 1..nr {
                if !own[ur] {
                    lf_horz_row(&l.2, ur, &mut row, data, strides[i], 0, sharp, bd);
                }
            }
        }
    }
}

/// Loop filter (7.14) of the superblock rows holding the luma 4x4 rows
/// `mi_r0..mi_r1` (`last`: up to the bottom of the frame), on the calling
/// thread. Rows above `mi_r0` must be filtered already and the rows below
/// not yet: the edges of these rows only reach samples of these rows and
/// the 7 rows above them, so filtering superblock rows in order gives the
/// result of [`loop_filter`].
pub fn loop_filter_rows<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    pic: &mut Picture<P>,
    mi_r0: usize,
    mi_r1: usize,
    last: bool,
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
        let vert = LfPlane::new(hdr, fs, plane, 0, sx, sy);
        let horz = LfPlane::new(hdr, fs, plane, 1, sx, sy);
        let nr = fs.mi_rows.div_ceil(1 << sy);
        let u0 = mi_r0 >> sy;
        let u1 = if last { nr } else { (mi_r1 >> sy).min(nr) };
        if u0 >= u1 {
            continue;
        }
        let pl = &mut pic.planes[plane];
        let job = LfJob {
            vert: &vert,
            horz: &horz,
            stride: pl.stride,
            u0,
            u1,
            first: true,
            last: true,
        };
        lf_chunk(&job, &mut pl.data, 0, sharp, bd);
    }
}

/// CDEF (7.15) of the 64x64 areas whose first luma 4x4 row is in
/// `mi_r0..mi_r1`, from `src` into `dst` (whose rows there start as a copy
/// of `src`), on the calling thread. `pads` is scratch from [`cdef_pads`].
pub fn cdef_sb_rows<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    src: &Picture<P>,
    dst: &mut Picture<P>,
    mi_r0: usize,
    mi_r1: usize,
    pads: &mut [Vec<u16>; 3],
) {
    let num_planes = seq.num_planes();
    let out = CdefOut {
        row0: [0; 3],
        stride: core::array::from_fn(|p| dst.planes.get(p).map_or(0, |pl| pl.stride)),
    };
    let mut chunks: Vec<&mut [P]> = dst
        .planes
        .iter_mut()
        .take(num_planes)
        .map(|pl| &mut pl.data[..])
        .collect();
    cdef_rows(seq, hdr, fs, src, &mut chunks, &out, mi_r0..mi_r1, pads);
}

/// Scratch for [`cdef_sb_rows`].
pub fn cdef_pads() -> [Vec<u16>; 3] {
    core::array::from_fn(|_| vec![0u16; CDEF_PAD_STRIDE * CDEF_PAD_STRIDE])
}

/// CDEF process (7.15). `src` is the deblocked frame; `dst` starts as a copy of it.
///
/// Runs on up to `threads` threads, each filtering a range of 64-row areas
/// into its own rows of `dst` (every block reads only `src`).
pub fn cdef<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    src: &Picture<P>,
    dst: &mut Picture<P>,
    threads: usize,
) {
    let num_planes = seq.num_planes();
    let sb_rows = fs.mi_rows.div_ceil(16);
    let t = threads.clamp(1, sb_rows.max(1));
    // One set of plane chunks per 64-row area; threads take areas in turn.
    let bounds: Vec<usize> = (0..=sb_rows).collect();
    let mut planes = Vec::new();
    let mut strides = Vec::new();
    let mut row_bounds = Vec::new();
    for (p, pl) in dst.planes.iter().enumerate().take(num_planes) {
        let sy = if p > 0 {
            usize::from(seq.subsampling_y)
        } else {
            0
        };
        row_bounds.push(bounds.iter().map(|&b| (b * 64) >> sy).collect::<Vec<_>>());
        strides.push(pl.stride);
    }
    for ((pl, &stride), rb) in dst.planes.iter_mut().zip(&strides).zip(&row_bounds) {
        planes.push((&mut pl.data[..], stride, &rb[..]));
    }
    let slots: Vec<Mutex<Option<Vec<&mut [P]>>>> = split_planes(planes, sb_rows.max(1))
        .into_iter()
        .map(|c| Mutex::new(Some(c)))
        .collect();
    let next = AtomicUsize::new(0);
    let (slots, next, row_bounds, strides) = (&slots, &next, &row_bounds, &strides);
    let run = move || {
        let mut pads: [Vec<u16>; 3] =
            core::array::from_fn(|_| vec![0u16; CDEF_PAD_STRIDE * CDEF_PAD_STRIDE]);
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= sb_rows {
                break;
            }
            let Some(mut chunks) = slots[i].lock().ok().and_then(|mut c| c.take()) else {
                continue;
            };
            let out = CdefOut {
                row0: core::array::from_fn(|p| row_bounds.get(p).map_or(0, |b| b[i])),
                stride: core::array::from_fn(|p| strides.get(p).copied().unwrap_or(0)),
            };
            cdef_rows(
                seq,
                hdr,
                fs,
                src,
                &mut chunks,
                &out,
                i * 16..(i + 1) * 16,
                &mut pads,
            );
        }
    };
    std::thread::scope(|s| {
        for _ in 1..t {
            s.spawn(run);
        }
        run();
    });
}

/// Where [`cdef_rows`] writes: plane `p`'s chunk starts at sample row
/// `row0[p]` and has stride `stride[p]`.
struct CdefOut {
    row0: [usize; 3],
    stride: [usize; 3],
}

/// CDEF of the 64x64 areas whose first luma 4x4 row is in `mi_rows`.
fn cdef_rows<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    src: &Picture<P>,
    dst: &mut [&mut [P]],
    out: &CdefOut,
    mi_rows: core::ops::Range<usize>,
    pads: &mut [Vec<u16>; 3],
) {
    let bd = src.bd;
    let coeff_shift = bd - 8;
    let num_planes = seq.num_planes();
    let (ssx, ssy) = (seq.subsampling_x, seq.subsampling_y);
    let y_damping = cdef_damping(i32::from(hdr.cdef.damping), bd, false);
    let uv_damping = cdef_damping(i32::from(hdr.cdef.damping), bd, true);
    // `pads`: one padded copy per plane of the current 64x64 area, built on
    // first use.
    let lp = &src.planes[0];
    for sbr in mi_rows.step_by(16) {
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
                                out,
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
                                out,
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
    dst: &mut [&mut [P]],
    out: &CdefOut,
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
    let ds = out.stride[plane];
    cdef_filter(
        &mut dst[plane][(y0 - out.row0[plane]) * ds + x0..],
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

/// The loop restoration units of every plane in the form [`LrPlane`] reads.
pub fn lr_units(seq: &SequenceHeader, fs: &FrameState) -> Vec<Vec<LrUnit>> {
    (0..seq.num_planes())
        .map(|p| {
            fs.lr[p]
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
                .collect()
        })
        .collect()
}

/// The [`LrPlane`]s of the planes loop restoration filters.
fn lr_plane_args<'a, P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    cdef: &'a Picture<P>,
    deblocked: &'a Picture<P>,
    out: &'a mut Picture<P>,
    units: &'a [Vec<LrUnit>],
) -> Vec<LrPlane<'a, P>> {
    let num_planes = seq.num_planes();
    let mut planes = Vec::new();
    for (p, dst) in out.planes.iter_mut().enumerate().take(num_planes) {
        if hdr.lr.frame_type[p] == 0 {
            continue;
        }
        let (sx, sy) = if p > 0 {
            (seq.subsampling_x, seq.subsampling_y)
        } else {
            (false, false)
        };
        planes.push(LrPlane {
            dst: &mut dst.data[..],
            cdef: &cdef.planes[p].data,
            deblocked: &deblocked.planes[p].data,
            stride: cdef.planes[p].stride,
            plane_w: (hdr.upscaled_width as usize + usize::from(sx)) >> u32::from(sx),
            plane_h: (hdr.frame_height as usize + usize::from(sy)) >> u32::from(sy),
            unit_size: hdr.lr.unit_size[p] as usize,
            units_per_row: fs.lr_cols[p],
            units_rows: fs.lr_rows[p],
            units: &units[p],
            sub_y: sy,
            bd: cdef.bd,
        });
    }
    planes
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
    threads: usize,
) {
    let units = lr_units(seq, fs);
    let planes = lr_plane_args(seq, hdr, fs, cdef, deblocked, out, &units);
    lr_planes(planes, scratch, threads);
}

/// [`loop_restoration`] of the stripes `stripes` only (stripe `k` starts
/// at luma row `64 k - 8`), on the calling thread, with `units` from
/// [`lr_units`]. Stripe `k` reads CDEF output up to the end of superblock
/// row `k` and deblocked samples a few rows into it.
pub fn loop_restoration_stripes<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    cdef: &Picture<P>,
    deblocked: &Picture<P>,
    out: &mut Picture<P>,
    units: &[Vec<LrUnit>],
    scratch: &mut LrScratch,
    stripes: core::ops::Range<usize>,
) {
    let planes = lr_plane_args(seq, hdr, fs, cdef, deblocked, out, units);
    lr_planes_stripes(planes, scratch, stripes);
}

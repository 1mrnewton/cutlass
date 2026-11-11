//! Frame-level in-loop filters: deblocking (7.14), CDEF (7.15), super
//! resolution (7.16) and loop restoration (7.17).

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::dsp::Pixel;
use crate::dsp::cdef::{
    CDEF_BORDER, CDEF_PAD_STRIDE, CDEF_UNAVAILABLE, CdefTaps, adjust_luma_pri_strength,
    cdef_damping, cdef_direction, cdef_filter, cdef_filter_inner, cdef_filter_taps, cdef_uv_dir,
};
use crate::dsp::lf::{filter_horizontal_edge, filter_vertical_edge};
use crate::dsp::lr::{LrPlane, LrScratch, LrUnit, lr_planes, lr_planes_stripes, lr_save_bounds};
use crate::dsp::superres::upscale_plane;

use super::frame::{FrameState, Picture, lf_h4, lf_w4};
use super::obu::{FrameHeader, SequenceHeader};
use super::types::{
    GLOBAL_GLOBALMV, GLOBALMV, INTRA_FRAME, NEARESTMV, NONE, bh4, bw4,
};

#[inline]
fn mode_type(y_mode: u8) -> usize {
    usize::from(y_mode >= NEARESTMV && y_mode != GLOBALMV && y_mode != GLOBAL_GLOBALMV)
}

/// Filter level (7.14.5) for loop filter index `i` (0, 1: luma vertical /
/// horizontal edges, 2, 3: U, V) of a block with the `delta_lf` value
/// `delta_lf`, the segment `seg`, the first reference frame `rf` and the
/// mode type `mode_type`.
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

/// Number of `ref_frame[0]` classes in [`LfLevels`]: 0..=7, and one for
/// every negative value.
const LF_REFS: usize = 9;

/// [`LfInfo::flags`]: the block is skipped (CDEF leaves an 8x8 of
/// skipped blocks alone).
pub const LF_SKIP: u8 = 1;
/// [`LfInfo::flags`]: the block is skipped and inter, so only its own
/// edges are filtered, not those of its transform blocks (7.14.2).
pub const LF_BLOCK_EDGES: u8 = 2;

/// What the loop filter and CDEF read of a block, kept for each of its
/// 4x4 units: its filter level for each loop filter index (7.14.5), its
/// size and its flags. Under a third the size of the mode info, and the
/// levels are worked out once for a block, not for each of its units in
/// each pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LfInfo {
    pub lvl: [u8; 4],
    pub flags: u8,
    pub bsize: u8,
    /// Makes the info 8 bytes, which a run of units is filled with as
    /// whole words.
    pub pad: [u8; 2],
}

/// The filter levels of a frame's blocks.
pub struct LfLevels {
    multi: bool,
    /// Levels for `delta_lf == 0`, by `[index][segment][ref class][mode type]`.
    lut: [[[[u8; 2]; LF_REFS]; 8]; 4],
}

impl LfLevels {
    pub fn new(hdr: &FrameHeader) -> Self {
        let mut lut = [[[[0u8; 2]; LF_REFS]; 8]; 4];
        for (li, t) in lut.iter_mut().enumerate() {
            for (seg, t) in t.iter_mut().enumerate() {
                for (rc, t) in t.iter_mut().enumerate() {
                    let rf = if rc < 8 { rc as i8 } else { NONE };
                    for (mt, t) in t.iter_mut().enumerate() {
                        *t = level_for(hdr, li, 0, seg, rf, mt);
                    }
                }
            }
        }
        Self {
            multi: hdr.delta_lf_multi,
            lut,
        }
    }

    /// What the filters read of a block with these fields.
    #[inline]
    pub fn info(
        &self,
        hdr: &FrameHeader,
        ref_frame: i8,
        segment_id: u8,
        y_mode: u8,
        skip: bool,
        bsize: u8,
        delta_lf: [i8; 4],
    ) -> LfInfo {
        let seg = usize::from(segment_id & 7);
        let rc = if ref_frame >= 0 {
            (ref_frame as usize).min(8)
        } else {
            8
        };
        let mt = mode_type(y_mode);
        let lvl = core::array::from_fn(|li| {
            let d = delta_lf[if self.multi { li } else { 0 }];
            if d == 0 {
                self.lut[li][seg][rc][mt]
            } else {
                level_for(hdr, li, i32::from(d), seg, ref_frame, mt)
            }
        });
        LfInfo {
            lvl,
            flags: u8::from(skip) * LF_SKIP
                + u8::from(skip && ref_frame > INTRA_FRAME) * LF_BLOCK_EDGES,
            bsize,
            pad: [0; 2],
        }
    }
}

/// The [`LfInfo`] of every 4x4 unit of a frame, laid out like
/// [`FrameState::mi`].
#[derive(Default)]
pub struct FilterInfo {
    pub units: Vec<LfInfo>,
    pub stride: usize,
}

impl FilterInfo {
    /// Sizes the array for the frame of `fs`. Units keep stale values
    /// until their blocks are stored.
    pub fn size_for(&mut self, fs: &FrameState) {
        self.stride = fs.mi_stride;
        self.units.resize(fs.lf_tx[0].len(), LfInfo::default());
    }

    /// Stores the info of a block at luma 4x4 position `(row, col)`, in
    /// its units inside the `mi_rows` x `mi_cols` frame.
    #[inline]
    pub fn store(
        &mut self,
        (row, col): (usize, usize),
        (mi_rows, mi_cols): (usize, usize),
        info: LfInfo,
    ) {
        let w4 = bw4(info.bsize).min(mi_cols.saturating_sub(col));
        let h4 = bh4(info.bsize).min(mi_rows.saturating_sub(row));
        for y in 0..h4 {
            let at = (row + y) * self.stride + col;
            if let Some(d) = self.units.get_mut(at..at + w4) {
                d.fill(info);
            }
        }
    }

    /// Fills luma 4x4 rows `rows` from the mode info of `fs`.
    pub fn fill_rows(
        &mut self,
        hdr: &FrameHeader,
        levels: &LfLevels,
        fs: &FrameState,
        rows: core::ops::Range<usize>,
    ) {
        let st = fs.mi_stride;
        let cols = fs.mi_cols;
        let rows = rows.start..rows.end.min(fs.mi_rows);
        // For the column a block starts at: the row it ends at and its
        // width. A row then steps over the blocks that began above it
        // without reading their mode info again (a block starts at a
        // multiple of its width and of its height).
        let mut ends = vec![(0u16, 0u8); cols];
        for r in rows.clone() {
            let Some(mi) = fs.mi.get(r * st..r * st + cols) else {
                return;
            };
            let mut c = 0;
            while c < cols {
                let (until, width) = ends[c];
                if usize::from(until) > r {
                    c += usize::from(width);
                    continue;
                }
                let m = &mi[c];
                let (w, h) = (bw4(m.bsize).max(1), bh4(m.bsize).max(1));
                let end = ((c / w + 1) * w).min(cols);
                let bottom = (r - r % h + h).min(rows.end);
                let info = levels.info(
                    hdr,
                    m.ref_frame[0],
                    m.segment_id,
                    m.y_mode,
                    m.skip,
                    m.bsize,
                    if hdr.delta_lf_present {
                        fs.delta_lf.get(r * st + c).copied().unwrap_or_default()
                    } else {
                        [0; 4]
                    },
                );
                for rr in r..bottom {
                    if let Some(out) = self.units.get_mut(rr * st + c..rr * st + end) {
                        out.fill(info);
                    }
                }
                ends[c] = (bottom as u16, (end - c) as u8);
                c = end;
            }
        }
    }

    /// Whether the block at luma 4x4 position `(row, col)` is skipped.
    #[inline(always)]
    pub fn skip(&self, row: usize, col: usize) -> bool {
        self.units[row * self.stride + col].flags & LF_SKIP != 0
    }
}

/// Edge decisions for one plane and pass of the loop filter (7.14.2 -
/// 7.14.5).
struct LfPlane<'a> {
    fs: &'a FrameState,
    info: &'a FilterInfo,
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
}

impl<'a> LfPlane<'a> {
    fn new(
        hdr: &'a FrameHeader,
        fs: &'a FrameState,
        info: &'a FilterInfo,
        plane: usize,
        pass: usize,
        sx: usize,
        sy: usize,
    ) -> Self {
        let li = if plane == 0 { pass } else { plane + 1 };
        let nc = fs.mi_cols.div_ceil(1 << sx);
        let nr = fs.mi_rows.div_ceil(1 << sy);
        Self {
            fs,
            info,
            tx: &fs.lf_tx[plane],
            plane,
            pass,
            sx,
            sy,
            nc,
            nc_vis: (hdr.frame_width as usize).div_ceil(4 << sx).min(nc),
            nr_vis: (hdr.frame_height as usize).div_ceil(4 << sy).min(nr),
            li,
        }
    }

    /// Calls `emit(unit, filter_size, lvl)` for the filtered edges of the
    /// unit row `ur` (plane 4x4 units), from left to right. `filter_size`
    /// follows [`crate::dsp::lf`], and `lvl` is not 0.
    #[inline]
    fn row_edges(&self, ur: usize, mut emit: impl FnMut(usize, u8, u8)) {
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
        // The row of block info holding the unit's bottom-right 4x4 for
        // chroma, and the row above it for the horizontal edges.
        let r = (ur << sy) | sy;
        let units = &self.info.units;
        let mi_row = &units[r * st..(r + 1) * st];
        let mi_prev = if pass0 {
            mi_row
        } else {
            let pr = r - (1 << sy);
            &units[pr * st..(pr + 1) * st]
        };
        let li = self.li;
        let mut uc = usize::from(pass0);
        while uc < self.nc_vis {
            let tx = tx_row[uc];
            // The units of this transform block left in the row: none of
            // them starts a vertical edge, and they all lie on a
            // horizontal edge or none does. A transform block starts at
            // a multiple of its width.
            let at = uc;
            let rest = (at | (lf_w4(tx) - 1)) + 1;
            let (t, pos) = if pass0 {
                uc = rest;
                (lf_w4(tx), at)
            } else {
                uc += 1;
                (lf_h4(tx), ur)
            };
            if pos & (t - 1) != 0 {
                uc = rest;
                continue;
            }
            let uc = at;
            let c = (uc << sx) | sx;
            let m = &mi_row[c];
            // In samples.
            let (base, pm) = if pass0 {
                (4 * lf_w4(tx_row[uc - 1]).min(t), &mi_row[c - (1 << sx)])
            } else {
                (4 * lf_h4(prev_row[uc]).min(t), &mi_prev[c])
            };
            let filter_size = if self.plane == 0 {
                base.min(16)
            } else if base >= 8 {
                6
            } else {
                4
            };
            let mut lvl = m.lvl[li];
            if lvl == 0 {
                lvl = pm.lvl[li];
            }
            if lvl != 0 {
                emit(uc, filter_size as u8, lvl);
            }
        }
    }
}

/// A run of edges with one filter size and level: `len` units from
/// `start`, along a row or down a column. `lvl == 0`: no run.
#[derive(Clone, Copy, Default)]
struct EdgeRun {
    start: u32,
    len: u32,
    size: u8,
    lvl: u8,
}

impl EdgeRun {
    /// Takes the edge at unit `at`: the run grows when the edge follows
    /// it with the same filter. Otherwise the edge starts a run, and the
    /// run it ends is returned (it may be no run).
    #[inline(always)]
    fn take(&mut self, at: usize, size: u8, lvl: u8) -> Option<EdgeRun> {
        if self.lvl == lvl && self.size == size && (self.start + self.len) as usize == at {
            self.len += 1;
            return None;
        }
        let ended = *self;
        *self = EdgeRun {
            start: at as u32,
            len: 1,
            size,
            lvl,
        };
        Some(ended)
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
    // Vertical edges: the edges of a strip of unit rows, gathered into
    // runs down each column that has any. A column's run is filtered when
    // an edge does not continue it, and at the end of the strip.
    let mut cols = vec![EdgeRun::default(); nc];
    let mut open: Vec<u32> = Vec::new();
    for strip in (u0..u1).step_by(LF_STRIP_UNITS) {
        let rows = (u1 - strip).min(LF_STRIP_UNITS);
        let down = |uc: usize, run: EdgeRun, data: &mut [P]| {
            filter_vertical_edge(
                data,
                stride,
                uc * 4,
                (strip + run.start as usize) * 4 - row0,
                run.len as usize * 4,
                usize::from(run.size),
                run.lvl,
                sharp,
                bd,
            );
        };
        for i in 0..rows {
            job.vert.row_edges(strip + i, |uc, size, lvl| {
                let Some(col) = cols.get_mut(uc) else {
                    return;
                };
                match col.take(i, size, lvl) {
                    Some(ended) if ended.lvl != 0 => down(uc, ended, data),
                    Some(_) => open.push(uc as u32),
                    None => {}
                }
            });
        }
        for uc in open.drain(..) {
            let run = std::mem::take(&mut cols[uc as usize]);
            down(uc as usize, run, data);
        }
    }
    // Horizontal edges whose samples stay inside the chunk.
    for ur in lf_own_rows(u0, u1, job.first, job.last) {
        lf_horz_row(job.horz, ur, data, stride, row0, sharp, bd);
    }
}

fn lf_horz_row<P: Pixel>(
    lp: &LfPlane<'_>,
    ur: usize,
    data: &mut [P],
    stride: usize,
    row0: usize,
    sharp: u8,
    bd: u32,
) {
    let across = |run: EdgeRun, data: &mut [P]| {
        if run.lvl != 0 {
            filter_horizontal_edge(
                data,
                stride,
                run.start as usize * 4,
                ur * 4 - row0,
                run.len as usize * 4,
                usize::from(run.size),
                run.lvl,
                sharp,
                bd,
            );
        }
    };
    let mut run = EdgeRun::default();
    lp.row_edges(ur, |uc, size, lvl| {
        if let Some(ended) = run.take(uc, size, lvl) {
            across(ended, data);
        }
    });
    across(run, data);
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
    loop_filter_scratch(seq, hdr, fs, pic, threads, &mut FilterInfo::default());
}

/// [`loop_filter`] with the block info worked out in `info`, scratch a
/// caller keeps from frame to frame.
pub fn loop_filter_scratch<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    pic: &mut Picture<P>,
    threads: usize,
    info: &mut FilterInfo,
) {
    if hdr.loop_filter.level[0] == 0 && hdr.loop_filter.level[1] == 0 {
        return;
    }
    let bd = pic.bd;
    let sharp = hdr.loop_filter.sharpness;
    info.size_for(fs);
    info.fill_rows(hdr, &LfLevels::new(hdr), fs, 0..fs.mi_rows);
    let info = &*info;
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
            LfPlane::new(hdr, fs, info, plane, 0, sx, sy),
            LfPlane::new(hdr, fs, info, plane, 1, sx, sy),
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
            let data = &mut pic.planes[l.0].data[..];
            for ur in 1..nr {
                if !own[ur] {
                    lf_horz_row(&l.2, ur, data, strides[i], 0, sharp, bd);
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
    // The horizontal edges at the top of these rows read the row above.
    let mut info = FilterInfo::default();
    info.size_for(fs);
    let end = if last { fs.mi_rows } else { mi_r1 };
    info.fill_rows(hdr, &LfLevels::new(hdr), fs, mi_r0.saturating_sub(2)..end);
    loop_filter_rows_with(seq, hdr, fs, &info, pic, mi_r0, mi_r1, last);
}

/// [`loop_filter_rows`] with the block info of the rows (and of the two
/// rows of 4x4 units above them) in `info`: `fs` gives the frame's size
/// and its transform sizes, not its mode info.
pub fn loop_filter_rows_with<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    info: &FilterInfo,
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
        let vert = LfPlane::new(hdr, fs, info, plane, 0, sx, sy);
        let horz = LfPlane::new(hdr, fs, info, plane, 1, sx, sy);
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

/// The pre-CDEF samples CDEF reads across the boundaries between rows of
/// 64x64 areas, saved before the areas are filtered in place: boundary
/// `k` (the top of area row `k`, at plane row `y = (64 k) >> subY`) holds
/// rows `y - 2 ..= y + 1`, the `(4 MiCols) >> subX` columns CDEF reads.
#[derive(Default)]
pub struct CdefLines<P> {
    rows: [Vec<P>; 3],
    w: [usize; 3],
}

/// Saves the boundaries `ks` of `src` (see [`CdefLines`]); boundary 0,
/// above the frame, is never read.
pub fn cdef_save_lines<P: Pixel>(
    seq: &SequenceHeader,
    fs: &FrameState,
    src: &Picture<P>,
    lines: &mut CdefLines<P>,
    ks: core::ops::Range<usize>,
) {
    let n = fs.mi_rows.div_ceil(16) + 1;
    for p in 0..seq.num_planes() {
        let (sx, sy) = plane_sub(seq, p);
        let pl = &src.planes[p];
        let w = ((fs.mi_cols * 4) >> sx).min(pl.stride);
        let rows = &mut lines.rows[p];
        lines.w[p] = w;
        if rows.len() < 4 * n * w {
            rows.resize(4 * n * w, P::default());
        }
        for k in ks.start.max(1)..ks.end.min(n) {
            let y = (64 * k) >> sy;
            for i in 0..4 {
                let yy = (y + i - 2).min(pl.alloc_h - 1);
                let o = (4 * k + i) * w;
                rows[o..o + w].copy_from_slice(&pl.data[yy * pl.stride..yy * pl.stride + w]);
            }
        }
    }
}

/// Subsampling shifts of plane `p`.
fn plane_sub(seq: &SequenceHeader, p: usize) -> (u32, u32) {
    if p > 0 {
        (u32::from(seq.subsampling_x), u32::from(seq.subsampling_y))
    } else {
        (0, 0)
    }
}

/// Working buffers of the CDEF of a run of areas: per plane, the padded
/// copy of the current 64x64 area and that of the last area filtered (its
/// columns are the pre-CDEF left columns of the next area).
struct CdefWork<P> {
    pads: [[Vec<u16>; 2]; 3],
    /// The taps of the 8-bit blocks filtered last, by [`taps_slot`]:
    /// blocks share a few strengths and eight directions.
    taps: Vec<Option<CdefTaps>>,
    _p: core::marker::PhantomData<P>,
}

/// Where [`CdefWork::taps`] keeps the taps for a plane's block with these
/// strengths and this direction.
#[inline(always)]
fn taps_slot(chroma: bool, pri: i32, dir: u8) -> usize {
    usize::from(dir & 7) | ((pri as usize & 15) << 3) | (usize::from(chroma) << 7)
}

impl<P: Pixel> CdefWork<P> {
    fn new() -> Self {
        Self {
            pads: core::array::from_fn(|_| {
                core::array::from_fn(|_| vec![0u16; CDEF_PAD_STRIDE * CDEF_PAD_STRIDE])
            }),
            taps: vec![None; 256],
            _p: core::marker::PhantomData,
        }
    }
}

/// Scratch of [`cdef_sb_rows`] and [`cdef_sb_rows_in_place`], which keep
/// the saved boundaries of a frame in it from call to call.
pub struct CdefScratch<P> {
    work: CdefWork<P>,
    lines: CdefLines<P>,
}

/// Scratch for [`cdef_sb_rows`].
pub fn cdef_pads<P: Pixel>() -> CdefScratch<P> {
    CdefScratch {
        work: CdefWork::new(),
        lines: CdefLines::default(),
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
    pads: &mut CdefScratch<P>,
) {
    let ks = mi_r0 / 16..mi_r1.div_ceil(16) + 1;
    cdef_save_lines(seq, fs, src, &mut pads.lines, ks);
    cdef_areas(seq, hdr, fs, None, dst, mi_r0, mi_r1, pads);
}

/// CDEF (7.15) in place of the deblocked 64x64 areas whose first luma 4x4
/// row is in `mi_r0..mi_r1`, on the calling thread. Calls go down the
/// frame in order, each with the same `sc`, and the rows below these areas
/// that CDEF reads must be deblocked: their pre-CDEF rows next to the
/// boundaries are saved in `sc` for the next call.
///
/// With `info`, the skipped blocks are read from it and `fs` gives the
/// frame's size and its CDEF indices, not its mode info.
pub fn cdef_sb_rows_in_place<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    info: Option<&FilterInfo>,
    pic: &mut Picture<P>,
    mi_r0: usize,
    mi_r1: usize,
    sc: &mut CdefScratch<P>,
) {
    let ks = mi_r0 / 16 + 1..mi_r1.div_ceil(16) + 1;
    cdef_save_lines(seq, fs, pic, &mut sc.lines, ks);
    cdef_areas(seq, hdr, fs, info, pic, mi_r0, mi_r1, sc);
}

/// CDEF in place of the areas of `mi_r0..mi_r1` with the boundaries in `sc`.
fn cdef_areas<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    info: Option<&FilterInfo>,
    pic: &mut Picture<P>,
    mi_r0: usize,
    mi_r1: usize,
    sc: &mut CdefScratch<P>,
) {
    let num_planes = seq.num_planes();
    let bd = pic.bd;
    let out = CdefOut {
        row0: [0; 3],
        stride: core::array::from_fn(|p| pic.planes.get(p).map_or(0, |pl| pl.stride)),
    };
    let mut chunks: Vec<&mut [P]> = pic
        .planes
        .iter_mut()
        .take(num_planes)
        .map(|pl| &mut pl.data[..])
        .collect();
    cdef_rows(
        seq,
        hdr,
        fs,
        info,
        bd,
        &mut chunks,
        &out,
        &sc.lines,
        mi_r0..mi_r1,
        &mut sc.work,
    );
}

/// CDEF process (7.15). `src` is the deblocked frame; `dst` ends as the
/// CDEF output (whatever it held before).
///
/// Runs on up to `threads` threads, each filtering a range of 64-row areas
/// in its own rows of `dst`.
pub fn cdef<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    src: &Picture<P>,
    dst: &mut Picture<P>,
    threads: usize,
) {
    for (d, s) in dst.planes.iter_mut().zip(&src.planes) {
        if d.data.len() == s.data.len() {
            d.data.copy_from_slice(&s.data);
        } else {
            d.data.clone_from(&s.data);
        }
    }
    let mut lines = CdefLines::default();
    cdef_save_lines(seq, fs, src, &mut lines, 0..usize::MAX);
    cdef_threads(seq, hdr, fs, dst, &lines, threads);
}

/// CDEF process (7.15) of the deblocked frame `pic`, in place, on up to
/// `threads` threads.
pub fn cdef_in_place<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    pic: &mut Picture<P>,
    threads: usize,
    sc: &mut CdefScratch<P>,
) {
    cdef_save_lines(seq, fs, pic, &mut sc.lines, 0..usize::MAX);
    cdef_threads(seq, hdr, fs, pic, &sc.lines, threads);
}

/// [`cdef_in_place`] once the boundaries are saved.
fn cdef_threads<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    dst: &mut Picture<P>,
    lines: &CdefLines<P>,
    threads: usize,
) {
    let num_planes = seq.num_planes();
    let bd = dst.bd;
    let sb_rows = fs.mi_rows.div_ceil(16);
    let t = threads.clamp(1, sb_rows.max(1));
    // One set of plane chunks per 64-row area; threads take areas in turn.
    let bounds: Vec<usize> = (0..=sb_rows).collect();
    let mut planes = Vec::new();
    let mut strides = Vec::new();
    let mut row_bounds = Vec::new();
    for (p, pl) in dst.planes.iter().enumerate().take(num_planes) {
        let sy = plane_sub(seq, p).1;
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
        let mut sc = CdefWork::<P>::new();
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
                None,
                bd,
                &mut chunks,
                &out,
                lines,
                i * 16..(i + 1) * 16,
                &mut sc,
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

/// Where [`cdef_rows`] works: plane `p`'s chunk starts at sample row
/// `row0[p]` and has stride `stride[p]`.
struct CdefOut {
    row0: [usize; 3],
    stride: [usize; 3],
}

/// CDEF in place of the 64x64 areas whose first luma 4x4 row is in
/// `mi_rows`, in the plane chunks `dst`.
fn cdef_rows<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    info: Option<&FilterInfo>,
    bd: u32,
    dst: &mut [&mut [P]],
    out: &CdefOut,
    lines: &CdefLines<P>,
    mi_rows: core::ops::Range<usize>,
    sc: &mut CdefWork<P>,
) {
    let coeff_shift = bd - 8;
    let num_planes = seq.num_planes();
    let (ssx, ssy) = (seq.subsampling_x, seq.subsampling_y);
    let y_damping = cdef_damping(i32::from(hdr.cdef.damping), bd, false);
    let uv_damping = cdef_damping(i32::from(hdr.cdef.damping), bd, true);
    let ls = out.stride[0];
    for sbr in mi_rows.step_by(16) {
        // Whether the area to the left was filtered (its padded copy, with
        // the pre-CDEF columns, is then in `sc.pads[p][1]`).
        let mut left_saved = [false; 3];
        for sbc in (0..fs.mi_cols).step_by(16) {
            let mut padded = [false; 3];
            let idx = fs.cdef_idx[(sbr >> 4) * fs.cdef_stride + (sbc >> 4)];
            let strengths = (idx >= 0).then(|| {
                let idx = idx as usize;
                (
                    i32::from(hdr.cdef.y_pri[idx]) << coeff_shift,
                    i32::from(hdr.cdef.y_sec[idx]) << coeff_shift,
                    i32::from(hdr.cdef.uv_pri[idx]) << coeff_shift,
                    i32::from(hdr.cdef.uv_sec[idx]) << coeff_shift,
                )
            });
            if let Some((y_pri0, y_sec, uv_pri, uv_sec)) = strengths {
                let luma_on = y_pri0 != 0 || y_sec != 0;
                let chroma_on = num_planes > 1 && (uv_pri != 0 || uv_sec != 0);
                let need_dir = y_pri0 != 0 || (chroma_on && uv_pri != 0);
                let r_end = (sbr + 16).min(fs.mi_rows);
                let c_end = (sbc + 16).min(fs.mi_cols);
                let area = CdefArea {
                    sbr,
                    sbc,
                    bd,
                    left_saved,
                };
                for r in (sbr..r_end).step_by(2) {
                    if !luma_on && !chroma_on {
                        break;
                    }
                    for c in (sbc..c_end).step_by(2) {
                        let rr = (r + 1).min(fs.mi_rows - 1);
                        let cc = (c + 1).min(fs.mi_cols - 1);
                        let skip = match info {
                            Some(i) => {
                                i.skip(r, c) && i.skip(rr, c) && i.skip(r, cc) && i.skip(rr, cc)
                            }
                            None => {
                                fs.mi_at(r, c).skip
                                    && fs.mi_at(rr, c).skip
                                    && fs.mi_at(r, cc).skip
                                    && fs.mi_at(rr, cc).skip
                            }
                        };
                        if skip {
                            continue;
                        }
                        let (ydir, var) = if need_dir {
                            let o = (r * 4 - out.row0[0]) * ls + c * 4;
                            cdef_direction(&dst[0][o..], ls, bd)
                        } else {
                            (0, 0)
                        };
                        if luma_on {
                            let pri = adjust_luma_pri_strength(y_pri0, var);
                            let dir = if y_pri0 == 0 { 0 } else { ydir };
                            if pri != 0 || y_sec != 0 {
                                cdef_plane_block(
                                    dst,
                                    out,
                                    lines,
                                    0,
                                    &area,
                                    (r, c),
                                    (0, 0),
                                    sc,
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
                                    dst,
                                    out,
                                    lines,
                                    plane,
                                    &area,
                                    (r, c),
                                    (u32::from(ssx), u32::from(ssy)),
                                    sc,
                                    &mut padded[plane],
                                    fs,
                                    (uv_pri, uv_sec, dir, uv_damping),
                                );
                            }
                        }
                    }
                }
            }
            // The padded copy of a filtered plane holds the pre-CDEF left
            // columns of the next area.
            for p in 0..3 {
                if padded[p] {
                    sc.pads[p].swap(0, 1);
                }
                left_saved[p] = padded[p];
            }
        }
    }
}

/// One 64x64 area of [`cdef_rows`]: its position in luma 4x4 units, and
/// per plane whether the area to its left was filtered (see [`CdefWork`]).
struct CdefArea {
    sbr: usize,
    sbc: usize,
    bd: u32,
    left_saved: [bool; 3],
}

/// Filters the 8x8 block (subsampled for chroma) at luma 4x4 position
/// `(r, c)` of `plane` in place. `strength` is `(pri, sec, dir, damping)`.
/// Pads the area holding the block first, unless `padded`.
#[inline(always)]
fn cdef_plane_block<P: Pixel>(
    dst: &mut [&mut [P]],
    out: &CdefOut,
    lines: &CdefLines<P>,
    plane: usize,
    area: &CdefArea,
    (r, c): (usize, usize),
    (sx, sy): (u32, u32),
    sc: &mut CdefWork<P>,
    padded: &mut bool,
    fs: &FrameState,
    (pri, sec, dir, damping): (i32, i32, u8, i32),
) {
    let ps = CDEF_PAD_STRIDE;
    let ds = out.stride[plane];
    let row0 = out.row0[plane];
    let sb_x = (area.sbc * 4) >> sx;
    let sb_y = (area.sbr * 4) >> sy;
    let (aw, ah) = (64 >> sx, 64 >> sy);
    let CdefWork { pads, taps, .. } = sc;
    let [pad, prev] = &mut pads[plane];
    if !*padded {
        let avail_w = (fs.mi_cols * 4) >> sx;
        let avail_h = (fs.mi_rows * 4) >> sy;
        let chunk = &dst[plane][..];
        let k = area.sbr >> 4;
        let lw = lines.w[plane];
        let lr = &lines.rows[plane];
        let left = (area.left_saved[plane] && sb_x > 0).then_some(&prev[..]);
        pad_area(
            pad,
            chunk,
            ds,
            row0,
            |i| &lr[(4 * k + i) * lw..(4 * k + i + 1) * lw],
            |i| &lr[(4 * k + 6 + i) * lw..(4 * k + 7 + i) * lw],
            left,
            (sb_x, sb_y, aw, ah),
            (avail_w, avail_h),
        );
        *padded = true;
    }
    let w = 8 >> sx;
    let h = 8 >> sy;
    let x0 = (c * 4) >> sx;
    let y0 = (r * 4) >> sy;
    // A block with the border CDEF reads inside the picture has every
    // tap available.
    let b = CDEF_BORDER;
    let avail_w = (fs.mi_cols * 4) >> sx;
    let avail_h = (fs.mi_rows * 4) >> sy;
    let inner = x0 >= b && y0 >= b && x0 + w + b <= avail_w && y0 + h + b <= avail_h;
    let at = (y0 - row0) * ds + x0;
    let src = &pad[(y0 - sb_y) * ps + (x0 - sb_x)..];
    if inner
        && area.bd == 8
        && let Some(bytes) = P::bytes_mut(&mut dst[plane][at..])
    {
        // 8-bit and inside the picture: the byte kernel, with the taps
        // of the last block that had these strengths and direction.
        let slot = &mut taps[taps_slot(plane > 0, pri, dir)];
        let ready = match slot {
            Some(t) if t.is_for(pri, sec, dir, damping) => *t,
            _ => *slot.insert(CdefTaps::new(pri, sec, dir, damping)),
        };
        if cdef_filter_taps(bytes, ds, src, w, h, &ready) {
            return;
        }
    }
    let filter = if inner {
        cdef_filter_inner::<P>
    } else {
        cdef_filter::<P>
    };
    filter(
        &mut dst[plane][at..],
        ds,
        src,
        ps,
        w,
        h,
        pri,
        sec,
        dir,
        damping,
        area.bd,
    );
}

/// Builds the padded source of [`cdef_filter`] for the `w x h` area at
/// `(x0, y0)` of a plane being filtered in place: rows `y0 - 2 ..= y0 + h + 1`
/// and columns `x0 - 2 ..= x0 + w + 1`, [`CDEF_UNAVAILABLE`] outside
/// `avail_w x avail_h`. The two rows above come from `top(i)`, the two
/// below from `bottom(i)` (whole saved rows), the others from `chunk`
/// (stride `stride`, first row `row0`). When the area to the left was
/// filtered, its padded copy `left` gives the two columns left of the area.
#[inline(never)]
fn pad_area<'a, P: Pixel>(
    pad: &mut [u16],
    chunk: &'a [P],
    stride: usize,
    row0: usize,
    top: impl Fn(usize) -> &'a [P],
    bottom: impl Fn(usize) -> &'a [P],
    left: Option<&[u16]>,
    (x0, y0, w, h): (usize, usize, usize, usize),
    (avail_w, avail_h): (usize, usize),
) {
    let b = CDEF_BORDER;
    let ps = CDEF_PAD_STRIDE;
    let xs = x0.saturating_sub(b);
    let xe = (x0 + w + b).min(avail_w).max(xs);
    let lead = xs + b - x0;
    let n = xe - xs;
    for r in 0..h + 2 * b {
        let out = &mut pad[r * ps..r * ps + w + 2 * b];
        let y = (y0 + r).wrapping_sub(b);
        if y0 + r < b || y >= avail_h {
            out.fill(CDEF_UNAVAILABLE);
            continue;
        }
        let row = if r < b {
            top(r)
        } else if r >= h + b {
            bottom(r - h - b)
        } else {
            let o = (y - row0) * stride;
            &chunk[o..o + xe]
        };
        let (lpad, rest) = out.split_at_mut(lead);
        lpad.fill(CDEF_UNAVAILABLE);
        let (mid, right) = rest.split_at_mut(n);
        let (mid, rx) = match left {
            Some(l) if xs < x0 => {
                // Columns w, w + 1 of the left area's copy.
                let (a, m) = mid.split_at_mut(b);
                a.copy_from_slice(&l[r * ps + w..r * ps + w + b]);
                (m, x0)
            }
            _ => (mid, xs),
        };
        for (o, s) in mid.iter_mut().zip(&row[rx..xe]) {
            *o = s.to_i32() as u16;
        }
        right.fill(CDEF_UNAVAILABLE);
    }
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

/// The deblocked rows loop restoration reads across the stripe boundaries
/// of every plane, saved before CDEF and loop restoration change them in
/// place (see [`lr_save_bounds`]).
pub struct LrBounds<P> {
    rows: [Vec<P>; 3],
}

impl<P> Default for LrBounds<P> {
    fn default() -> Self {
        Self {
            rows: Default::default(),
        }
    }
}

/// Plane size in the upscaled domain of plane `p`.
fn lr_plane_size(seq: &SequenceHeader, hdr: &FrameHeader, p: usize) -> (usize, usize) {
    let (sx, sy) = plane_sub(seq, p);
    (
        (hdr.upscaled_width as usize + (1 << sx) - 1) >> sx,
        (hdr.frame_height as usize + (1 << sy) - 1) >> sy,
    )
}

/// Saves the rows of the deblocked (upscaled) frame `pic` next to the
/// bottom boundaries of the loop restoration stripes `stripes` of every
/// plane loop restoration filters. Stripe `k` ends above luma row
/// `64 k + 56`; the rows saved are final once the frame is deblocked down
/// to luma row `64 k + 64`.
pub fn lr_save_frame_bounds<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    pic: &Picture<P>,
    bounds: &mut LrBounds<P>,
    stripes: core::ops::Range<usize>,
) {
    for p in 0..seq.num_planes() {
        if hdr.lr.frame_type[p] == 0 {
            continue;
        }
        let (w, h) = lr_plane_size(seq, hdr, p);
        let pl = &pic.planes[p];
        lr_save_bounds(
            &mut bounds.rows[p],
            &pl.data,
            pl.stride,
            w,
            h,
            plane_sub(seq, p).1,
            stripes.clone(),
        );
    }
}

/// The [`LrPlane`]s of the planes loop restoration filters.
fn lr_plane_args<'a, P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    pic: &'a mut Picture<P>,
    bounds: &'a LrBounds<P>,
    units: &'a [Vec<LrUnit>],
) -> Vec<LrPlane<'a, P>> {
    let num_planes = seq.num_planes();
    let bd = pic.bd;
    let mut planes = Vec::new();
    for (p, dst) in pic.planes.iter_mut().enumerate().take(num_planes) {
        if hdr.lr.frame_type[p] == 0 {
            continue;
        }
        let (plane_w, plane_h) = lr_plane_size(seq, hdr, p);
        planes.push(LrPlane {
            stride: dst.stride,
            dst: &mut dst.data[..],
            bounds: &bounds.rows[p],
            plane_w,
            plane_h,
            unit_size: hdr.lr.unit_size[p] as usize,
            units_per_row: fs.lr_cols[p],
            units_rows: fs.lr_rows[p],
            units: &units[p],
            sub_y: p > 0 && seq.subsampling_y,
            bd,
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
    let _ = cdef;
    let mut bounds = LrBounds::default();
    lr_save_frame_bounds(seq, hdr, deblocked, &mut bounds, 0..usize::MAX);
    loop_restoration_in_place(seq, hdr, fs, out, &bounds, scratch, threads);
}

/// Loop restoration (7.17) in place of UpscaledCdefFrame `pic`, with the
/// rows of UpscaledCurrFrame saved in `bounds` (all stripes), on up to
/// `threads` threads.
pub fn loop_restoration_in_place<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    pic: &mut Picture<P>,
    bounds: &LrBounds<P>,
    scratch: &mut LrScratch,
    threads: usize,
) {
    let units = lr_units(seq, fs);
    let planes = lr_plane_args(seq, hdr, fs, pic, bounds, &units);
    lr_planes(planes, scratch, threads);
}

/// [`loop_restoration_in_place`] of the stripes `stripes` only (stripe `k`
/// starts at luma row `64 k - 8`), on the calling thread, with `units` from
/// [`lr_units`]. Stripe `k` reads CDEF output up to the end of superblock
/// row `k`, and `bounds` of stripes `k - 1` and `k`.
pub fn loop_restoration_stripes<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    pic: &mut Picture<P>,
    bounds: &LrBounds<P>,
    units: &[Vec<LrUnit>],
    scratch: &mut LrScratch,
    stripes: core::ops::Range<usize>,
) {
    let planes = lr_plane_args(seq, hdr, fs, pic, bounds, units);
    lr_planes_stripes(planes, scratch, stripes);
}

#[cfg(test)]
mod tests;

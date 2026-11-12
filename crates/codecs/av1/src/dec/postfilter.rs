//! Frame-level in-loop filters: deblocking (7.14), CDEF (7.15), super
//! resolution (7.16) and loop restoration (7.17).

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::dsp::Pixel;
use crate::dsp::cdef::{
    CDEF_BORDER, CDEF_PAD_STRIDE, CDEF_UNAVAILABLE, CdefTaps, adjust_luma_pri_strength,
    cdef_damping, cdef_direction, cdef_filter, cdef_filter_inner, cdef_filter_taps,
    cdef_filter_taps_bytes, cdef_uv_dir,
};
use crate::dsp::lf::{filter_horizontal_edge, filter_vertical_edge};
use crate::dsp::lr::{LrPlane, LrScratch, LrUnit, lr_planes, lr_planes_stripes, lr_save_bounds};
use crate::dsp::superres::upscale_plane;

use super::frame::{FrameState, Picture};
use super::obu::{FrameHeader, SequenceHeader};
use super::types::{GLOBAL_GLOBALMV, GLOBALMV, INTRA_FRAME, NEARESTMV, NONE, bh4, bw4};
use cutlass_simd::av1::lfmask::{self, EdgeUnits, Edges, edge_codes};

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
/// The bit of the first byte of [`LfInfo::bytes`] that says the block
/// is skipped, above the 6 bits of the level.
pub const UNIT_SKIP: u8 = 0x80;

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
        }
    }
}

impl LfInfo {
    /// What [`FilterInfo`] keeps of the info: the levels (6 bits each),
    /// with [`UNIT_SKIP`] in the first byte for a skipped block. A
    /// word, so a frame's units take half the memory of eight bytes.
    #[inline(always)]
    pub fn bytes(self) -> [u8; 4] {
        let [a, b, c, d] = self.lvl;
        let skip = if self.flags & LF_SKIP != 0 {
            UNIT_SKIP
        } else {
            0
        };
        [a | skip, b, c, d]
    }
}

/// The [`LfInfo`] of every 4x4 unit of a frame as its
/// [`bytes`](LfInfo::bytes), laid out like [`FrameState::mi`].
#[derive(Default)]
pub struct FilterInfo {
    pub units: Vec<[u8; 4]>,
    pub stride: usize,
}

impl FilterInfo {
    /// Sizes the array for the frame of `fs`. Units keep stale values
    /// until their blocks are stored.
    pub fn size_for(&mut self, fs: &FrameState) {
        self.stride = fs.mi_stride;
        self.units.resize(fs.lf_tx[0].len(), [0; 4]);
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
                d.fill(info.bytes());
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
                let mut m = &mi[c];
                // The first row may cross a block whose even rows hold
                // only its edges (parsing's mode info, see
                // `MiFill::OddRows`): the odd row below is in the block
                // too when it is taller than a unit, and otherwise this
                // row is the block's last.
                if r == rows.start
                    && r % 2 == 0
                    && let Some(below) = fs.mi.get((r + 1) * st + c)
                    && r + 1 < fs.mi_rows
                    && bh4(below.bsize) > 1
                {
                    m = below;
                }
                let (w, h) = (bw4(m.bsize).max(1), bh4(m.bsize).max(1));
                let end = ((c / w + 1) * w).min(cols);
                let bottom = (r - r % h + h).min(rows.end);
                let info = levels.info(
                    hdr,
                    m.ref_frame[0],
                    m.segment_id(),
                    m.y_mode,
                    m.skip(),
                    m.bsize,
                    if hdr.delta_lf_present {
                        fs.delta_lf.get(r * st + c).copied().unwrap_or_default()
                    } else {
                        [0; 4]
                    },
                );
                for rr in r..bottom {
                    if let Some(out) = self.units.get_mut(rr * st + c..rr * st + end) {
                        out.fill(info.bytes());
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
        self.units[row * self.stride + col][0] & UNIT_SKIP != 0
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

    /// Writes the edges of the unit row `ur` (plane 4x4 units) to
    /// `row.codes`: for a unit whose left side (pass 0) or top side
    /// (pass 1) is an edge with a level, its filter size (following
    /// [`crate::dsp::lf`]) times 256 plus the level; 0 for the other
    /// units. Returns how many units the row has, 0 for a row without
    /// edges.
    ///
    /// Branch free, so the compiler works on many units at once: a unit
    /// starts a vertical edge when its column is a multiple of its
    /// transform's width, and a horizontal one when its row is a
    /// multiple of the transform's height. The filter size follows from
    /// the smaller of the transform sizes on the two sides.
    fn row_codes(&self, ur: usize, row: &mut LfRow) -> usize {
        let pass0 = self.pass == 0;
        let n = self.nc_vis;
        if ur >= self.nr_vis || (!pass0 && ur == 0) || n == 0 {
            return 0;
        }
        let sy = self.sy;
        let st = self.fs.mi_stride;
        let units = &self.info.units;
        // The row of block info holding the unit's bottom-right 4x4 for
        // chroma, and the row above it for the horizontal edges.
        let r = (ur << sy) | sy;
        let cur = &self.tx[ur * st..ur * st + n];
        let key = (self.plane, self.pass, ur);
        let (across, edges) = if pass0 {
            (cur, None)
        } else {
            // The row above's levels, if they are the last ones made.
            if row.lvl_of == Some((self.plane, 1, ur - 1)) {
                core::mem::swap(&mut row.lvl, &mut row.lvl_prev);
            } else {
                let pr = r - (1 << sy);
                self.levels(&units[pr * st..(pr + 1) * st], &mut row.lvl_prev[..n]);
            }
            let zeros = ur.trailing_zeros().min(7) as u8;
            (
                &self.tx[(ur - 1) * st..(ur - 1) * st + n],
                Some(Edges::Top { zeros }),
            )
        };
        self.levels(&units[r * st..(r + 1) * st], &mut row.lvl[..n]);
        row.lvl_of = Some(key);
        let luma = self.plane == 0;
        row.n = n;
        let codes = &mut row.codes[..n];
        let masks = &mut row.masks[..n.div_ceil(16)];
        let (out, _) = codes.as_chunks_mut::<16>();
        let whole = out.len() * 16;
        let dims = cur.as_chunks::<16>().0;
        let lvl = row.lvl.as_chunks::<16>().0;
        // For vertical edges, the sixteen units before: none at first.
        let before = |c: &[[u8; 16]], k: usize| if k == 0 { [0; 16] } else { c[k - 1] };
        let (across_c, lvl_prev) = (across.as_chunks::<16>().0, row.lvl_prev.as_chunks::<16>().0);
        for (k, (o, mask)) in out.iter_mut().zip(masks.iter_mut()).enumerate() {
            let u = if pass0 {
                EdgeUnits {
                    dims: dims[k],
                    across: before(dims, k),
                    lvl: lvl[k],
                    lvl_across: before(lvl, k),
                }
            } else {
                EdgeUnits {
                    dims: dims[k],
                    across: across_c[k],
                    lvl: lvl[k],
                    lvl_across: lvl_prev[k],
                }
            };
            (*o, *mask) = edge_codes(
                &u,
                edges.unwrap_or(Edges::Left {
                    first: (16 * k) as u8,
                }),
                luma,
            );
        }
        if whole < n {
            // The last units, with idle lanes after them.
            let rest = n - whole;
            let pad = |s: &[u8]| {
                let mut a = [0u8; 16];
                a[..rest].copy_from_slice(&s[whole..n]);
                a
            };
            let k = whole / 16;
            let u = if pass0 {
                EdgeUnits {
                    dims: pad(cur),
                    across: before(dims, k),
                    lvl: pad(&row.lvl),
                    lvl_across: before(lvl, k),
                }
            } else {
                EdgeUnits {
                    dims: pad(cur),
                    across: pad(across),
                    lvl: pad(&row.lvl),
                    lvl_across: pad(&row.lvl_prev),
                }
            };
            let first = Edges::Left { first: whole as u8 };
            let (c, mask) = edge_codes(&u, edges.unwrap_or(first), luma);
            codes[whole..].copy_from_slice(&c[..rest]);
            masks[k] = mask & ((1 << rest) - 1);
        }
        if pass0 {
            // Nothing to the left of the frame.
            codes[0] = 0;
            masks[0] &= !1;
        }
        n
    }

    /// The level (of this plane and pass) of each unit of a row whose
    /// blocks' info is `mi`.
    fn levels(&self, mi: &[[u8; 4]], out: &mut [u8]) {
        let li = self.li;
        let mut done = 0;
        if self.sx == 0 {
            for (o, m) in out
                .as_chunks_mut::<16>()
                .0
                .iter_mut()
                .zip(mi.as_chunks::<16>().0)
            {
                *o = lfmask::levels(m, li);
                done += 16;
            }
        } else {
            for (o, m) in out
                .as_chunks_mut::<16>()
                .0
                .iter_mut()
                .zip(mi.as_chunks::<32>().0)
            {
                *o = lfmask::levels_odd(m, li);
                done += 16;
            }
        }
        let sx = self.sx;
        for (i, o) in out.iter_mut().enumerate().skip(done) {
            *o = mi[(i << sx) | sx][li] & 63;
        }
    }
}

/// Scratch for [`LfPlane::row_codes`], as long as a plane row of units.
struct LfRow {
    codes: Vec<u16>,
    /// The units whose code is not 0, sixteen to an entry.
    masks: Vec<u16>,
    /// The units of the last row.
    n: usize,
    /// The plane, pass and row `lvl` holds the levels of.
    lvl_of: Option<(usize, usize, usize)>,
    lvl: Vec<u8>,
    lvl_prev: Vec<u8>,
}

impl LfRow {
    fn new(nc: usize) -> Self {
        Self {
            codes: vec![0; nc.div_ceil(16) * 16],
            masks: vec![0; nc.div_ceil(16)],
            n: 0,
            lvl_of: None,
            lvl: vec![0; nc],
            lvl_prev: vec![0; nc],
        }
    }
}

/// For sixteen units, a bit for each whose code differs from `before`'s
/// at the same place.
#[inline(always)]
fn changes(now: &[u16; 16], before: &[u16; 16]) -> u32 {
    let mut m = 0;
    for l in 0..16 {
        m |= u32::from(now[l] != before[l]) << l;
    }
    m
}

impl LfRow {
    /// Calls `run(start, len, code)` for the runs of units of one code
    /// (not 0), from left to right. Only the units where the code changes
    /// are visited one by one.
    #[inline]
    fn runs(&self, mut run: impl FnMut(usize, usize, u16)) {
        let chunks = self.n.div_ceil(16);
        let (whole, _) = self.codes.as_chunks::<16>();
        // The code of the last unit seen, and where its run began.
        let (mut code, mut start) = (0, 0);
        for (k, (now, &m)) in whole.iter().zip(&self.masks).take(chunks).enumerate() {
            if m == 0 && code == 0 {
                continue;
            }
            let before: [u16; 16] =
                core::array::from_fn(|l| if l == 0 { code } else { now[l - 1] });
            let mut c = changes(now, &before);
            while c != 0 {
                let l = c.trailing_zeros() as usize;
                c &= c - 1;
                let i = 16 * k + l;
                if code != 0 {
                    run(start, i - start, code);
                }
                (code, start) = (now[l], i);
            }
        }
        // The units past the row are 0: a run still open ends with the
        // last whole chunk.
        if code != 0 {
            run(start, 16 * chunks - start, code);
        }
    }
}

/// The runs of vertical edges down each column of a strip of unit rows
/// (see [`lf_chunk`]): the code of each column's open run (0: none),
/// sixteen columns to an entry, and the row it began at.
struct ColumnRuns {
    code: Vec<[u16; 16]>,
    start: Vec<u16>,
}

impl ColumnRuns {
    fn new(nc: usize) -> Self {
        Self {
            code: vec![[0; 16]; nc.div_ceil(16)],
            start: vec![0; nc.div_ceil(16) * 16],
        }
    }

    /// Takes row `i`'s codes (`None`: no edges), calling
    /// `ended(column, start, len, code)` for each run it ends.
    #[inline]
    fn row(
        &mut self,
        i: usize,
        codes: Option<&[[u16; 16]]>,
        mut ended: impl FnMut(usize, usize, usize, u16),
    ) {
        const NONE: [u16; 16] = [0; 16];
        for (k, cur) in self.code.iter_mut().enumerate() {
            let now = codes.and_then(|c| c.get(k)).unwrap_or(&NONE);
            let mut c = changes(now, cur);
            while c != 0 {
                let l = c.trailing_zeros() as usize;
                c &= c - 1;
                let uc = 16 * k + l;
                if cur[l] != 0 {
                    let s = usize::from(self.start[uc]);
                    ended(uc, s, i - s, cur[l]);
                }
                cur[l] = now[l];
                self.start[uc] = i as u16;
            }
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
    let mut row = LfRow::new(nc);
    // Vertical edges: the edges of a strip of unit rows, gathered into
    // runs down each column. A column's run is filtered when an edge
    // does not continue it, and at the end of the strip.
    let mut cols = ColumnRuns::new(nc);
    for strip in (u0..u1).step_by(LF_STRIP_UNITS) {
        let rows = (u1 - strip).min(LF_STRIP_UNITS);
        let down = |uc: usize, start: usize, len: usize, code: u16, data: &mut [P]| {
            filter_vertical_edge(
                data,
                stride,
                uc * 4,
                (strip + start) * 4 - row0,
                len * 4,
                usize::from(code >> 8),
                code as u8,
                sharp,
                bd,
            );
        };
        for i in 0..=rows {
            // One row past the strip ends every run.
            let has = i < rows && job.vert.row_codes(strip + i, &mut row) != 0;
            let codes = has.then(|| row.codes.as_chunks::<16>().0);
            cols.row(i, codes, |uc, start, len, code| {
                down(uc, start, len, code, data)
            });
        }
    }
    // Horizontal edges whose samples stay inside the chunk.
    for ur in lf_own_rows(u0, u1, job.first, job.last) {
        lf_horz_row(job.horz, ur, data, stride, row0, sharp, bd, &mut row);
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
    row: &mut LfRow,
) {
    if lp.row_codes(ur, row) == 0 {
        return;
    }
    row.runs(|start, len, code| {
        filter_horizontal_edge(
            data,
            stride,
            start * 4,
            ur * 4 - row0,
            len * 4,
            usize::from(code >> 8),
            code as u8,
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
            let mut row = LfRow::new(l.2.nc);
            for ur in 1..nr {
                if !own[ur] {
                    lf_horz_row(&l.2, ur, data, strides[i], 0, sharp, bd, &mut row);
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

/// How far, in luma 4x4 rows, [`cdef_sb_rows_in_place`] moves its 64-row
/// areas up: deblocking the superblock row below changes none of the rows
/// these areas and the 2 rows past them hold, so a superblock row's areas
/// are filtered as soon as it is deblocked (the loop restoration stripes
/// lie the same way).
pub const CDEF_ROW_OFFSET: usize = 2;

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
    off: usize,
) {
    let n = (fs.mi_rows + off).div_ceil(16) + 1;
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
            let y = (64 * k - 4 * off) >> sy;
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
    /// The same for 8-bit pictures, as bytes: only blocks whose taps all
    /// lie in the picture read them as they are; the others take a copy
    /// marking the samples outside it (see [`cdef_plane_block`]).
    pads8: [[Vec<u8>; 2]; 3],
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
                core::array::from_fn(|_| {
                    if P::HBD {
                        vec![0u16; CDEF_PAD_STRIDE * CDEF_PAD_STRIDE]
                    } else {
                        Vec::new()
                    }
                })
            }),
            pads8: core::array::from_fn(|_| {
                core::array::from_fn(|_| {
                    if P::HBD {
                        Vec::new()
                    } else {
                        vec![0u8; CDEF_PAD_STRIDE * CDEF_PAD_STRIDE]
                    }
                })
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
    cdef_save_lines(seq, fs, src, &mut pads.lines, ks, 0);
    cdef_areas(seq, hdr, fs, None, dst, mi_r0, mi_r1, 0, pads);
}

/// CDEF (7.15) in place of the deblocked luma 4x4 rows `mi_r0..mi_r1`, on
/// the calling thread, in areas of 64 rows moved up [`CDEF_ROW_OFFSET`]
/// rows (both ends lie on area boundaries, or end the frame). Calls go
/// down the frame in order, each with the same `sc`, and the rows below
/// these areas that CDEF reads must be deblocked: their pre-CDEF rows next
/// to the boundaries are saved in `sc` for the next call.
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
    let off = CDEF_ROW_OFFSET;
    let ks = (mi_r0 + off) / 16 + 1..(mi_r1 + off).div_ceil(16) + 1;
    cdef_save_lines(seq, fs, pic, &mut sc.lines, ks, off);
    cdef_areas(seq, hdr, fs, info, pic, mi_r0, mi_r1, off, sc);
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
    off: usize,
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
        off,
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
    cdef_save_lines(seq, fs, src, &mut lines, 0..usize::MAX, 0);
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
    cdef_save_lines(seq, fs, pic, &mut sc.lines, 0..usize::MAX, 0);
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
                0,
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
    off: usize,
    sc: &mut CdefWork<P>,
) {
    let coeff_shift = bd - 8;
    let num_planes = seq.num_planes();
    let (ssx, ssy) = (seq.subsampling_x, seq.subsampling_y);
    let y_damping = cdef_damping(i32::from(hdr.cdef.damping), bd, false);
    let uv_damping = cdef_damping(i32::from(hdr.cdef.damping), bd, true);
    let ls = out.stride[0];
    for seg in (mi_rows.start + off) / 16..(mi_rows.end + off).div_ceil(16) {
        // The area's rows: those of 64x64 area `seg` moved up `off` 4x4
        // rows, which take the strengths of the area above.
        let top = (16 * seg).saturating_sub(off);
        let end = (16 * seg + 16 - off).min(fs.mi_rows);
        // Whether the area to the left was filtered (its padded copy, with
        // the pre-CDEF columns, is then in `sc.pads[p][1]`).
        let mut left_saved = [false; 3];
        for sbc in (0..fs.mi_cols).step_by(16) {
            let mut padded = [false; 3];
            let area = CdefArea {
                seg,
                top,
                rows: 16 * seg + 16 - off - top,
                sbc,
                bd,
                left_saved,
            };
            for (sbr, r0, r1) in [
                (16 * seg).checked_sub(16).map(|a| (a, top, 16 * seg)),
                Some((16 * seg, 16 * seg, end)),
            ]
            .into_iter()
            .flatten()
            {
                if r0 >= r1 {
                    continue;
                }
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
                    let c_end = (sbc + 16).min(fs.mi_cols);
                    if !luma_on && !chroma_on {
                        continue;
                    }
                    // The 8x8 blocks to filter and their luma direction and
                    // variance, then each plane's blocks in a loop of their
                    // own: one loop for all of it kept too much live and
                    // spilled.
                    let mut todo = [(0u16, 0u16, 0u8, 0i32); 64];
                    let mut n = 0;
                    for r in (r0..r1).step_by(2) {
                        for c in (sbc..c_end).step_by(2) {
                            let rr = (r + 1).min(fs.mi_rows - 1);
                            let cc = (c + 1).min(fs.mi_cols - 1);
                            let skip = match info {
                                Some(i) => {
                                    i.skip(r, c) && i.skip(rr, c) && i.skip(r, cc) && i.skip(rr, cc)
                                }
                                None => {
                                    fs.mi_at(r, c).skip()
                                        && fs.mi_at(rr, c).skip()
                                        && fs.mi_at(r, cc).skip()
                                        && fs.mi_at(rr, cc).skip()
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
                            if let Some(t) = todo.get_mut(n) {
                                *t = (r as u16, c as u16, ydir, var);
                                n += 1;
                            }
                        }
                    }
                    let todo = &todo[..n];
                    if luma_on {
                        for &(r, c, ydir, var) in todo {
                            let pri = adjust_luma_pri_strength(y_pri0, var);
                            let dir = if y_pri0 == 0 { 0 } else { ydir };
                            if pri != 0 || y_sec != 0 {
                                cdef_plane_block(
                                    dst,
                                    out,
                                    lines,
                                    0,
                                    &area,
                                    (usize::from(r), usize::from(c)),
                                    (0, 0),
                                    sc,
                                    &mut padded[0],
                                    fs,
                                    (pri, y_sec, dir, y_damping),
                                );
                            }
                        }
                    }
                    if chroma_on {
                        for plane in 1..3 {
                            for &(r, c, ydir, _) in todo {
                                let dir = if uv_pri == 0 {
                                    0
                                } else {
                                    cdef_uv_dir(ssx, ssy, ydir)
                                };
                                cdef_plane_block(
                                    dst,
                                    out,
                                    lines,
                                    plane,
                                    &area,
                                    (usize::from(r), usize::from(c)),
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
                    sc.pads8[p].swap(0, 1);
                }
                left_saved[p] = padded[p];
            }
        }
    }
}

/// One 64x64 area of [`cdef_rows`]: its position in luma 4x4 units, and
/// per plane whether the area to its left was filtered (see [`CdefWork`]).
struct CdefArea {
    /// The index of the area (and of its upper boundary in [`CdefLines`]).
    seg: usize,
    /// The first luma 4x4 row, and the luma 4x4 rows the area spans when
    /// inside the picture.
    top: usize,
    rows: usize,
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
    let sb_y = (area.top * 4) >> sy;
    let (aw, ah) = (64 >> sx, (4 * area.rows) >> sy);
    if !P::HBD {
        return cdef_plane_block_8(
            dst,
            out,
            lines,
            plane,
            area,
            (r, c),
            (sx, sy),
            sc,
            padded,
            fs,
            (pri, sec, dir, damping),
        );
    }
    let CdefWork { pads, taps, .. } = sc;
    let [pad, prev] = &mut pads[plane];
    if !*padded {
        let avail_w = (fs.mi_cols * 4) >> sx;
        let avail_h = (fs.mi_rows * 4) >> sy;
        let chunk = &dst[plane][..];
        let k = area.seg;
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
        // The taps in place, not a copy of them.
        if !slot
            .as_ref()
            .is_some_and(|t| t.is_for(pri, sec, dir, damping))
        {
            *slot = Some(CdefTaps::new(pri, sec, dir, damping));
        }
        if let Some(ready) = slot.as_ref()
            && cdef_filter_taps(bytes, ds, src, w, h, ready)
        {
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

/// [`cdef_plane_block`] for 8-bit pictures: the area is padded as bytes.
/// A block whose taps all lie in the picture is filtered from them; any
/// other gets a 16-bit copy of its window with the samples outside the
/// picture marked [`CDEF_UNAVAILABLE`].
#[inline(always)]
fn cdef_plane_block_8<P: Pixel>(
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
    let sb_y = (area.top * 4) >> sy;
    let (aw, ah) = (64 >> sx, (4 * area.rows) >> sy);
    let avail_w = (fs.mi_cols * 4) >> sx;
    let avail_h = (fs.mi_rows * 4) >> sy;
    let CdefWork { pads8, taps, .. } = sc;
    let [pad, prev] = &mut pads8[plane];
    if !*padded {
        let chunk = &dst[plane][..];
        let k = area.seg;
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
    let b = CDEF_BORDER;
    let inner = x0 >= b && y0 >= b && x0 + w + b <= avail_w && y0 + h + b <= avail_h;
    let at = (y0 - row0) * ds + x0;
    let src = &pad[(y0 - sb_y) * ps + (x0 - sb_x)..];
    if inner && let Some(bytes) = P::bytes_mut(&mut dst[plane][at..]) {
        let slot = &mut taps[taps_slot(plane > 0, pri, dir)];
        // The taps in place, not a copy of them.
        if !slot
            .as_ref()
            .is_some_and(|t| t.is_for(pri, sec, dir, damping))
        {
            *slot = Some(CdefTaps::new(pri, sec, dir, damping));
        }
        if let Some(ready) = slot.as_ref()
            && cdef_filter_taps_bytes(bytes, ds, src, w, h, ready)
        {
            return;
        }
    }
    // The window of the block, rows y0 - 2 .. y0 + h + 2 and columns
    // x0 - 2 .. x0 + w + 2, with what lies outside the picture marked.
    let mut win = [CDEF_UNAVAILABLE; CDEF_PAD_STRIDE * 12];
    for i in 0..h + 2 * b {
        let y = (y0 + i).wrapping_sub(b);
        if y >= avail_h {
            continue;
        }
        for j in 0..w + 2 * b {
            let x = (x0 + j).wrapping_sub(b);
            if x < avail_w
                && let Some(&v) = src.get(i * ps + j)
            {
                win[i * ps + j] = u16::from(v);
            }
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
        &win,
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

/// A sample of a padded CDEF source: 16-bit, or a byte for 8-bit pictures
/// (whose samples outside the picture are then not marked).
trait PadSample: Copy {
    /// Whether samples outside the picture are marked.
    const MARKS: bool;
    const UNAVAILABLE: Self;
    fn of<P: Pixel>(p: P) -> Self;
}

impl PadSample for u16 {
    const MARKS: bool = true;
    const UNAVAILABLE: Self = CDEF_UNAVAILABLE;
    #[inline(always)]
    fn of<P: Pixel>(p: P) -> Self {
        p.to_i32() as u16
    }
}

impl PadSample for u8 {
    const MARKS: bool = false;
    const UNAVAILABLE: Self = 0;
    #[inline(always)]
    fn of<P: Pixel>(p: P) -> Self {
        p.to_i32() as u8
    }
}

/// Converts `src` into `out` (as long as the shorter): the widths of a
/// padded row inside the picture take fixed-size loops, with no call.
#[inline(always)]
fn convert_row<P: Pixel, T: PadSample>(out: &mut [T], src: &[P]) {
    macro_rules! fixed {
        ($($n:literal)*) => {
            match out.len().min(src.len()) {
                $($n => {
                    if let (Some(o), Some(s)) = (out.first_chunk_mut::<$n>(), src.first_chunk::<$n>()) {
                        for (o, s) in o.iter_mut().zip(s) {
                            *o = T::of(*s);
                        }
                    }
                })*
                _ => {
                    for (o, s) in out.iter_mut().zip(src) {
                        *o = T::of(*s);
                    }
                }
            }
        };
    }
    fixed!(68 66 64 36 34 32);
}

/// Builds the padded source of [`cdef_filter`] for the `w x h` area at
/// `(x0, y0)` of a plane being filtered in place: rows `y0 - 2 ..= y0 + h + 1`
/// and columns `x0 - 2 ..= x0 + w + 1`, [`CDEF_UNAVAILABLE`] outside
/// `avail_w x avail_h`. The two rows above come from `top(i)`, the two
/// below from `bottom(i)` (whole saved rows), the others from `chunk`
/// (stride `stride`, first row `row0`). When the area to the left was
/// filtered, its padded copy `left` gives the two columns left of the area.
#[inline(never)]
fn pad_area<'a, P: Pixel, T: PadSample>(
    pad: &mut [T],
    chunk: &'a [P],
    stride: usize,
    row0: usize,
    top: impl Fn(usize) -> &'a [P],
    bottom: impl Fn(usize) -> &'a [P],
    left: Option<&[T]>,
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
            if T::MARKS {
                out.fill(T::UNAVAILABLE);
            }
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
        if T::MARKS {
            lpad.fill(T::UNAVAILABLE);
        }
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
        convert_row(mid, &row[rx..xe]);
        if T::MARKS {
            right.fill(T::UNAVAILABLE);
        }
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

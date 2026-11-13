//! The picture under reconstruction and the decoded pictures it references.

use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64};
use std::sync::{Arc, Mutex, MutexGuard};

use cutlass_decode::CodecError;

use crate::cabac::Contexts;
use crate::frame::Frame;
use crate::pixel::Pixel;
use crate::ps::{Layout, Pps, Sps};
use crate::slice::SliceHeader;

/// A motion vector in quarter luma samples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mv {
    pub x: i16,
    pub y: i16,
}

/// Motion of one 4×4 luma block. `pred` bit 0 is list 0, bit 1 list 1;
/// zero is intra or not yet decoded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MvField {
    pub mv: [Mv; 2],
    pub ref_idx: [i8; 2],
    pub pred: u8,
}

impl MvField {
    pub fn uses(&self, list: usize) -> bool {
        self.pred & (1 << list) != 0
    }
}

/// Motion of a 16×16 block of a decoded picture, kept for temporal motion
/// vector prediction (clause 8.5.3.2.8). The reference is named by its
/// order count and whether it was a long-term reference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColMv {
    pub mv: [Mv; 2],
    pub poc: [i32; 2],
    pub long_term: [bool; 2],
    pub pred: u8,
}

#[derive(Clone)]
pub struct RefPic {
    pub frame: Arc<Frame>,
    pub poc: i32,
    pub long_term: bool,
    /// Identity of the DPB picture, for deblocking's same-picture test.
    pub uid: u64,
}

/// One independent or dependent slice segment of the picture.
#[derive(Clone)]
pub struct SliceCtx {
    pub header: SliceHeader,
    pub refs: [Vec<RefPic>; 2],
    /// `SliceAddrRs`: the first CTB of the independent slice.
    pub slice_addr: u32,
    pub qp: i32,
    /// Every reference precedes the current picture (`NoBackwardPredFlag`).
    pub no_backward_pred: bool,
}

/// SAO parameters of one CTB and component (clause 7.4.9.3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SaoParams {
    /// 0 off, 1 band, 2 edge.
    pub kind: u8,
    /// Band position or edge class.
    pub class: u8,
    pub offsets: [i16; 4],
}

pub const MODE_INTRA: u8 = 1;
pub const MODE_INTER: u8 = 2;
pub const FLAG_SKIP: u8 = 4;
/// PCM with the loop filter off, or transquant bypass: no deblocking or SAO.
pub const FLAG_KEEP: u8 = 8;
pub const FLAG_PCM: u8 = 16;
pub const FLAG_CBF: u8 = 32;

/// Reconstruction state of the picture being decoded, except its samples.
///
/// Row workers share it while they decode one slice segment, so every map a
/// coding tree unit writes is atomic. A worker only reads what the CTB row
/// gate says another worker has finished, so relaxed loads and stores carry
/// no ordering of their own.
pub struct Picture {
    pub sps: Arc<Sps>,
    pub pps: Arc<Pps>,
    pub layout: Arc<Layout>,
    pub width: usize,
    pub height: usize,
    /// Width of the 4×4 grid.
    pub w4: usize,
    /// The 4×4 maps (`mvf`, `info`, `intra_mode`, `qp`, `depth`) hold a
    /// ring of rows of 4×4 blocks: row `r` is at `r & mask4`. Nothing reads
    /// them more than a few CTB rows behind the rows being reconstructed,
    /// so a few CTB rows serve the picture and stay in cache. `usize::MAX`
    /// holds the whole picture (tiles reorder the rows).
    pub mask4: usize,
    pub mvf: Motion,
    /// `MODE_*` and `FLAG_*` of each 4×4.
    pub info: Bytes,
    pub intra_mode: Bytes,
    /// `QpY` of each 4×4, stored as the bits of an `i8`.
    pub qp: Bytes,
    pub depth: Bytes,
    /// Boundary strength of vertical edges: x on the 8 grid, 4 rows each.
    pub bs_ver: Bytes,
    /// Boundary strength of horizontal edges: y on the 8 grid, 4 columns each.
    pub bs_hor: Bytes,
    /// Slice segment index of each CTB, raster order.
    pub ctb_slice: Vec<AtomicU16>,
    /// `SliceAddrRs` of each decoded CTB.
    pub ctb_slice_addr: Vec<AtomicU32>,
    pub sao: Sao,
    pub slices: Vec<SliceCtx>,
    pub poc: i32,
    /// Contexts and QP at the end of the last slice segment, for a
    /// dependent slice segment.
    pub dependent: Mutex<(Contexts, i32)>,
    /// Contexts stored after the second CTB of each row, by CTB.
    pub wpp_ctx: Mutex<Vec<Option<Box<Contexts>>>>,
}

/// One byte for each cell of a map.
pub struct Bytes(Vec<AtomicU8>);

impl Bytes {
    pub fn new(len: usize, value: u8) -> Self {
        Self((0..len).map(|_| AtomicU8::new(value)).collect())
    }

    #[inline]
    pub fn get(&self, at: usize) -> u8 {
        self.0[at].load(Relaxed)
    }

    #[inline]
    pub fn set(&self, at: usize, value: u8) {
        self.0[at].store(value, Relaxed);
    }

    /// Every cell to `value`.
    pub fn fill_all(&self, value: u8) {
        for cell in &self.0 {
            cell.store(value, Relaxed);
        }
    }

    /// Fill the `w`×`h` luma area at (`x`, `y`) of a map on the 4×4 grid.
    pub fn fill4(
        &self,
        (w4, mask4): (usize, usize),
        (x, y): (usize, usize),
        (w, h): (usize, usize),
        value: u8,
    ) {
        for row in (y >> 2)..((y + h) >> 2) {
            let start = (row & mask4) * w4 + (x >> 2);
            for cell in &self.0[start..start + (w >> 2)] {
                cell.store(value, Relaxed);
            }
        }
    }
}

/// The motion of each 4×4: both vectors in one word, the reference indices
/// and the prediction flags in another.
pub struct Motion {
    mv: Vec<AtomicU64>,
    refs: Vec<AtomicU32>,
}

impl Motion {
    fn new(len: usize) -> Self {
        Self {
            mv: (0..len).map(|_| AtomicU64::new(0)).collect(),
            refs: (0..len).map(|_| AtomicU32::new(0)).collect(),
        }
    }

    #[inline]
    pub fn get(&self, at: usize) -> MvField {
        let mv = self.mv[at].load(Relaxed);
        let refs = self.refs[at].load(Relaxed);
        let half = |shift: u32| (mv >> shift) as u16 as i16;
        MvField {
            mv: [
                Mv {
                    x: half(0),
                    y: half(16),
                },
                Mv {
                    x: half(32),
                    y: half(48),
                },
            ],
            ref_idx: [refs as u8 as i8, (refs >> 8) as u8 as i8],
            pred: (refs >> 16) as u8,
        }
    }

    /// Store `field` in `len` cells from `start`.
    pub fn fill(&self, start: usize, len: usize, field: MvField) {
        let bits = |v: i16| u64::from(v as u16);
        let mv = bits(field.mv[0].x)
            | bits(field.mv[0].y) << 16
            | bits(field.mv[1].x) << 32
            | bits(field.mv[1].y) << 48;
        let refs = u32::from(field.ref_idx[0] as u8)
            | u32::from(field.ref_idx[1] as u8) << 8
            | u32::from(field.pred) << 16;
        for (a, b) in self.mv[start..start + len]
            .iter()
            .zip(&self.refs[start..start + len])
        {
            a.store(mv, Relaxed);
            b.store(refs, Relaxed);
        }
    }
}

/// SAO parameters of each CTB and component.
pub struct Sao {
    offsets: Vec<AtomicU64>,
    kind: Vec<AtomicU16>,
}

impl Sao {
    fn new(ctbs: usize) -> Self {
        Self {
            offsets: (0..ctbs * 3).map(|_| AtomicU64::new(0)).collect(),
            kind: (0..ctbs * 3).map(|_| AtomicU16::new(0)).collect(),
        }
    }

    fn clear(&self) {
        for (a, b) in self.offsets.iter().zip(&self.kind) {
            a.store(0, Relaxed);
            b.store(0, Relaxed);
        }
    }

    pub fn get(&self, rs: usize) -> [SaoParams; 3] {
        let mut out = [SaoParams::default(); 3];
        for (c, p) in out.iter_mut().enumerate() {
            let offsets = self.offsets[rs * 3 + c].load(Relaxed);
            let kind = self.kind[rs * 3 + c].load(Relaxed);
            p.kind = kind as u8;
            p.class = (kind >> 8) as u8;
            for (k, o) in p.offsets.iter_mut().enumerate() {
                *o = (offsets >> (16 * k)) as u16 as i16;
            }
        }
        out
    }

    pub fn set(&self, rs: usize, params: [SaoParams; 3]) {
        for (c, p) in params.iter().enumerate() {
            let mut offsets = 0u64;
            for (k, &o) in p.offsets.iter().enumerate() {
                offsets |= u64::from(o as u16) << (16 * k);
            }
            self.offsets[rs * 3 + c].store(offsets, Relaxed);
            self.kind[rs * 3 + c].store(u16::from(p.kind) | u16::from(p.class) << 8, Relaxed);
        }
    }
}

/// The picture under reconstruction, one buffer for each CTB row so that
/// row workers and the loop filters each lock only the rows they touch,
/// and the bottom line of every CTB row as reconstruction left it, before
/// the loop filters, for intra prediction of the row below.
///
/// Unless tiles reorder the CTB rows, a few buffers serve every row in
/// turn ([`RING`] more than the rows reconstructed at once): a row's buffer
/// is free again once its final samples are out, and the few in use stay
/// in cache.
pub struct Work<P> {
    pub rows: Vec<Mutex<[Vec<P>; 3]>>,
    pub lines: Vec<Mutex<[Vec<P>; 3]>>,
    pub width: usize,
    pub height: usize,
    pub log2_ctb: u32,
    /// CTB rows of the picture.
    pub count: usize,
}

/// Row buffers beyond the rows reconstructed at once. A row's buffer is
/// read until the final samples of the row below are made, which follows
/// that row's deblocking ([`Picture::sao_row`]); the spare buffers let
/// reconstruction run ahead of the filters without waiting (two measured
/// slower than four).
pub const RING: usize = 4;

impl<P: Pixel> Work<P> {
    /// Buffers for a picture of `width`×`height` with CTBs of `1 <<
    /// log2_ctb`: `slots` of them, at most one for each CTB row.
    pub fn new(width: usize, height: usize, log2_ctb: u32, slots: usize) -> Self {
        let ctb = 1usize << log2_ctb;
        let count = height.div_ceil(ctb);
        let slots = slots.clamp(1, count.max(1));
        let rows = (0..slots)
            .map(|r| {
                // Without the ring, the last row holds only its lines.
                let lines = if slots < count {
                    ctb
                } else {
                    ctb.min(height - r * ctb)
                };
                Mutex::new([
                    vec![P::default(); lines * width],
                    vec![P::default(); lines / 2 * (width / 2)],
                    vec![P::default(); lines / 2 * (width / 2)],
                ])
            })
            .collect();
        let lines = (0..slots)
            .map(|_| {
                Mutex::new([
                    vec![P::default(); width],
                    vec![P::default(); width / 2],
                    vec![P::default(); width / 2],
                ])
            })
            .collect();
        Self {
            rows,
            lines,
            width,
            height,
            log2_ctb,
            count,
        }
    }

    /// Whether these buffers fit a picture of this size and CTB with
    /// `slots` buffers.
    pub fn fits(&self, width: usize, height: usize, log2_ctb: u32, slots: usize) -> bool {
        self.width == width
            && self.height == height
            && self.log2_ctb == log2_ctb
            && self.rows.len() == slots.clamp(1, self.count.max(1))
    }

    /// The buffer of CTB row `row`.
    #[inline]
    pub fn slot(&self, row: usize) -> usize {
        row % self.rows.len().max(1)
    }

    /// The samples of CTB row `row`, locked.
    pub fn band(&self, row: usize) -> Samples<'_, P> {
        Samples {
            band: lock(&self.rows[self.slot(row)]),
            top: row << self.log2_ctb,
            width: self.width,
            work: self,
        }
    }
}

/// The samples of one CTB row under reconstruction, read and written by
/// luma or chroma position. The line above the row comes from what the
/// row above saved.
pub struct Samples<'p, P> {
    band: MutexGuard<'p, [Vec<P>; 3]>,
    /// First luma row held.
    top: usize,
    width: usize,
    work: &'p Work<P>,
}

impl<P: Pixel> Samples<'_, P> {
    #[inline]
    fn stride(&self, c: usize) -> usize {
        if c == 0 { self.width } else { self.width / 2 }
    }

    /// The CTB row held.
    pub fn row(&self) -> usize {
        self.top >> self.work.log2_ctb
    }

    /// Index of component sample (`x`, `y`). The row must be held.
    #[inline]
    pub fn index(&self, c: usize, x: usize, y: usize) -> usize {
        let top = if c == 0 { self.top } else { self.top / 2 };
        (y - top) * self.stride(c) + x
    }

    /// The plane of component `c` from sample (`x`, `y`) on, and its stride.
    #[inline]
    pub fn from(&mut self, c: usize, x: usize, y: usize) -> (&mut [P], usize) {
        let at = self.index(c, x, y);
        let stride = self.stride(c);
        (&mut self.band[c][at..], stride)
    }

    /// Whether component row `y` is held here (else it is the line above).
    #[inline]
    pub fn holds(&self, c: usize, y: usize) -> bool {
        y >= if c == 0 { self.top } else { self.top / 2 }
    }

    /// Component samples (`x`, `y`) downwards, one per entry of `out`. The
    /// rows must be held.
    #[inline]
    pub fn column(&self, c: usize, x: usize, y: usize, out: &mut [i32]) {
        let stride = self.stride(c);
        let plane = &self.band[c][self.index(c, x, y)..];
        for (o, s) in out.iter_mut().zip(plane.iter().step_by(stride)) {
            *o = s.i();
        }
    }

    /// Component samples `x..x + out.len()` of the line just above the
    /// row, as reconstruction left it.
    pub fn above(&self, c: usize, x: usize, out: &mut [P]) {
        match self
            .row()
            .checked_sub(1)
            .and_then(|r| self.work.lines.get(self.work.slot(r)))
        {
            Some(line) => {
                let line = lock(line);
                out.copy_from_slice(&line[c][x..x + out.len()]);
            }
            None => out.fill(P::default()),
        }
    }

    /// Save the bottom line of the CTB at luma (`x`, `y`), `size` wide, for
    /// the row below.
    pub fn publish(&self, x: usize, y: usize, size: usize, height: usize) {
        let Some(line) = self.work.lines.get(self.work.slot(y >> self.work.log2_ctb)) else {
            return;
        };
        let bottom = (y + size).min(height) - 1;
        let w = size.min(self.width - x);
        let mut line = lock(line);
        for c in 0..3 {
            let (sx, sy, sw) = if c == 0 {
                (x, bottom, w)
            } else {
                (x / 2, bottom / 2, w / 2)
            };
            let at = self.index(c, sx, sy);
            line[c][sx..sx + sw].copy_from_slice(&self.band[c][at..at + sw]);
        }
    }
}

pub fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

impl Picture {
    /// The row mask of the 4×4 maps when `ring` CTB rows are
    /// reconstructed at once ([`Self::mask4`]), `None` for the whole
    /// picture: rows of [`RING`] + 2 more CTB rows, as a power of two of
    /// 4×4 rows.
    pub fn map_mask(sps: &Sps, ring: Option<usize>) -> usize {
        let h4 = sps.height as usize / 4;
        let ctb4 = (1usize << sps.log2_ctb) / 4;
        match ring {
            Some(rows) if ((rows + RING + 2) * ctb4).next_power_of_two() < h4 => {
                ((rows + RING + 2) * ctb4).next_power_of_two() - 1
            }
            _ => usize::MAX,
        }
    }

    pub fn new(
        sps: Arc<Sps>,
        pps: Arc<Pps>,
        layout: Arc<Layout>,
        (poc, mask4): (i32, usize),
    ) -> Result<Self, CodecError> {
        let width = sps.width as usize;
        let height = sps.height as usize;
        width.checked_mul(height).ok_or(CodecError::Limit {
            what: "picture size",
        })?;
        let w4 = width / 4;
        let h4 = height / 4;
        // Rows of the 4×4 maps: the ring, or every row.
        let r4 = if mask4 == usize::MAX { h4 } else { mask4 + 1 };
        let ctbs = layout.ctb_count() as usize;
        Ok(Self {
            width,
            height,
            w4,
            mask4,
            mvf: Motion::new(w4 * r4),
            info: Bytes::new(w4 * r4, 0),
            intra_mode: Bytes::new(w4 * r4, 1),
            qp: Bytes::new(w4 * r4, 0),
            depth: Bytes::new(w4 * r4, 0),
            bs_ver: Bytes::new((width / 8 + 1) * h4, 0),
            bs_hor: Bytes::new((height / 8 + 1) * w4, 0),
            ctb_slice: (0..ctbs).map(|_| AtomicU16::new(u16::MAX)).collect(),
            ctb_slice_addr: (0..ctbs).map(|_| AtomicU32::new(u32::MAX)).collect(),
            sao: Sao::new(ctbs),
            slices: Vec::new(),
            poc,
            dependent: Mutex::new(([0; 256], 0)),
            wpp_ctx: Mutex::new(if pps.entropy_sync {
                vec![None; ctbs]
            } else {
                Vec::new()
            }),
            sps,
            pps,
            layout,
        })
    }

    /// This picture's maps made ready for another picture of the same
    /// size and layout, or new ones.
    pub fn reuse(
        old: Option<Self>,
        sps: Arc<Sps>,
        pps: Arc<Pps>,
        layout: Arc<Layout>,
        (poc, mask4): (i32, usize),
    ) -> Result<Self, CodecError> {
        let Some(mut pic) = old.filter(|p| {
            p.width == sps.width as usize
                && p.height == sps.height as usize
                && p.ctb_slice.len() == layout.ctb_count() as usize
                && p.mask4 == mask4
        }) else {
            return Self::new(sps, pps, layout, (poc, mask4));
        };
        // Every map but the boundary strengths is written for each block
        // before it is read; edges no block names must read as 0.
        pic.bs_ver.fill_all(0);
        pic.bs_hor.fill_all(0);
        for (a, b) in pic.ctb_slice.iter().zip(&pic.ctb_slice_addr) {
            a.store(u16::MAX, Relaxed);
            b.store(u32::MAX, Relaxed);
        }
        pic.sao.clear();
        pic.slices.clear();
        pic.poc = poc;
        *lock(&pic.dependent) = ([0; 256], 0);
        {
            let ctbs = layout.ctb_count() as usize;
            let mut wpp = lock(&pic.wpp_ctx);
            wpp.clear();
            if pps.entropy_sync {
                wpp.resize(ctbs, None);
            }
        }
        pic.sps = sps;
        pic.pps = pps;
        pic.layout = layout;
        Ok(pic)
    }

    /// The motion of the 4×4 holding luma (`x`, `y`).
    #[inline]
    pub fn mv_at(&self, x: usize, y: usize) -> MvField {
        self.mvf.get(self.at4(x, y))
    }

    /// The `SliceAddrRs` of CTB `rs`.
    #[inline]
    pub fn slice_addr(&self, rs: usize) -> u32 {
        self.ctb_slice_addr[rs].load(Relaxed)
    }

    #[inline]
    pub fn at4(&self, x: usize, y: usize) -> usize {
        ((y >> 2) & self.mask4) * self.w4 + (x >> 2)
    }

    /// Width and row mask of the 4×4 maps, for [`Bytes::fill4`].
    #[inline]
    pub fn geom4(&self) -> (usize, usize) {
        (self.w4, self.mask4)
    }

    /// Where row `row` of 4×4 blocks starts in the 4×4 maps.
    #[inline]
    pub fn row4(&self, row: usize) -> usize {
        (row & self.mask4) * self.w4
    }

    #[inline]
    pub fn ctb_rs(&self, x: usize, y: usize) -> usize {
        let log2 = self.layout.log2_ctb;
        (y >> log2) * self.layout.ctb_width as usize + (x >> log2)
    }

    /// Z-scan order of a luma position inside its CTB (clause 6.5.2).
    #[inline]
    pub fn z_in_ctb(&self, x: usize, y: usize) -> u16 {
        let layout = &*self.layout;
        let mask = (1usize << layout.log2_ctb) - 1;
        let shift = layout.log2_min_tb;
        let at = ((y & mask) >> shift) * layout.zorder_edge + ((x & mask) >> shift);
        layout.zorder.get(at).copied().unwrap_or(0)
    }

    /// Z-scan availability (clause 6.4.1).
    #[inline]
    pub fn available(&self, x: usize, y: usize, nx: isize, ny: isize) -> bool {
        if nx < 0 || ny < 0 || nx as usize >= self.width || ny as usize >= self.height {
            return false;
        }
        let (nx, ny) = (nx as usize, ny as usize);
        // Most neighbours lie in the same CTB, which slices and tiles never
        // split: only the decoding order inside it counts.
        let log2 = self.layout.log2_ctb;
        if (nx >> log2, ny >> log2) == (x >> log2, y >> log2) {
            return self.z_in_ctb(nx, ny) <= self.z_in_ctb(x, y);
        }
        let nb_ctb = self.ctb_rs(nx, ny);
        let cur_ctb = self.ctb_rs(x, y);
        if self.slice_addr(nb_ctb) != self.slice_addr(cur_ctb) {
            return false;
        }
        // Without tiles, tile scan is raster scan.
        if !self.pps.tiles {
            return nb_ctb < cur_ctb;
        }
        self.layout.tile_of[nb_ctb] == self.layout.tile_of[cur_ctb]
            && self.layout.rs_to_ts[nb_ctb] < self.layout.rs_to_ts[cur_ctb]
    }

    pub fn slice_of(&self, x: usize, y: usize) -> Option<&SliceCtx> {
        let index = self.ctb_slice.get(self.ctb_rs(x, y))?.load(Relaxed);
        self.slices.get(usize::from(index))
    }
}

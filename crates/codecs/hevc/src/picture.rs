//! The picture under reconstruction and the decoded pictures it references.

use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64};
use std::sync::{Arc, Mutex, MutexGuard};

use cutlass_decode::CodecError;

use crate::cabac::Contexts;
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

/// A decoded picture, immutable once it enters the DPB.
#[derive(Debug)]
pub struct Decoded {
    pub width: usize,
    pub height: usize,
    pub planes: [Vec<u16>; 3],
    pub col: Vec<ColMv>,
    pub col_stride: usize,
    pub poc: i32,
}

impl Decoded {
    /// A picture that the bitstream named and did not send (clause 8.3.3):
    /// mid-grey, with no motion.
    pub fn missing(width: usize, height: usize, bit_depth: u8, poc: i32) -> Self {
        let grey = 1u16 << (bit_depth - 1);
        Self {
            width,
            height,
            planes: [
                vec![grey; width * height],
                vec![grey; (width / 2) * (height / 2)],
                vec![grey; (width / 2) * (height / 2)],
            ],
            col: Vec::new(),
            col_stride: 0,
            poc,
        }
    }

    pub fn col_at(&self, x: usize, y: usize) -> ColMv {
        if self.col_stride == 0 {
            return ColMv::default();
        }
        self.col
            .get((y >> 4) * self.col_stride + (x >> 4))
            .copied()
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug)]
pub struct RefPic {
    pub pic: Arc<Decoded>,
    pub poc: i32,
    pub long_term: bool,
    /// Identity of the DPB picture, for deblocking's same-picture test.
    pub uid: u64,
}

/// One independent or dependent slice segment of the picture.
#[derive(Clone, Debug)]
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

    /// Fill the `w`×`h` luma area at (`x`, `y`) of a map on the 4×4 grid.
    pub fn fill4(&self, w4: usize, (x, y): (usize, usize), (w, h): (usize, usize), value: u8) {
        for row in (y >> 2)..((y + h) >> 2) {
            let start = row * w4 + (x >> 2);
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

/// Samples of the picture under reconstruction, read and written by luma
/// or chroma position. A row worker holds only its own band of CTB rows;
/// the line above the band comes from what the worker above published.
pub struct Samples<'p> {
    planes: [&'p mut [u16]; 3],
    /// First luma row held.
    top: usize,
    width: usize,
    lines: Option<&'p Lines>,
}

impl<'p> Samples<'p> {
    /// Every row of the picture.
    pub fn whole(planes: &'p mut [Vec<u16>; 3], width: usize) -> Self {
        let [y, u, v] = planes;
        Self {
            planes: [y.as_mut_slice(), u.as_mut_slice(), v.as_mut_slice()],
            top: 0,
            width,
            lines: None,
        }
    }

    /// Luma rows from `top` on, held in `planes`.
    pub fn band(planes: [&'p mut [u16]; 3], top: usize, width: usize, lines: &'p Lines) -> Self {
        Self {
            planes,
            top,
            width,
            lines: Some(lines),
        }
    }

    #[inline]
    fn stride(&self, c: usize) -> usize {
        if c == 0 { self.width } else { self.width / 2 }
    }

    /// Index of component sample (`x`, `y`). The row must be held.
    #[inline]
    pub fn index(&self, c: usize, x: usize, y: usize) -> usize {
        let top = if c == 0 { self.top } else { self.top / 2 };
        (y - top) * self.stride(c) + x
    }

    /// The plane of component `c` from sample (`x`, `y`) on, and its stride.
    #[inline]
    pub fn from(&mut self, c: usize, x: usize, y: usize) -> (&mut [u16], usize) {
        let at = self.index(c, x, y);
        let stride = self.stride(c);
        (&mut self.planes[c][at..], stride)
    }

    /// Whether component row `y` is held here (else it is the line above).
    #[inline]
    pub fn holds(&self, c: usize, y: usize) -> bool {
        y >= if c == 0 { self.top } else { self.top / 2 }
    }

    #[inline]
    pub fn get(&self, c: usize, x: usize, y: usize) -> u16 {
        self.planes[c][self.index(c, x, y)]
    }

    /// Component samples `x..x + out.len()` of the line just above the
    /// band. Zeros when no line was published.
    pub fn above(&self, c: usize, x: usize, out: &mut [u16]) {
        let row = (self.top >> self.lines.map_or(0, |l| l.log2_ctb)).wrapping_sub(1);
        match self.lines.and_then(|l| l.rows.get(row)) {
            Some(line) => {
                let line = lock(line);
                out.copy_from_slice(&line[c][x..x + out.len()]);
            }
            None => out.fill(0),
        }
    }

    /// Publish the bottom line of the CTB at luma (`x`, `y`), `size` wide,
    /// for the worker below. Nothing happens on whole planes.
    pub fn publish(&self, x: usize, y: usize, size: usize, height: usize) {
        let Some(lines) = self.lines else {
            return;
        };
        let Some(line) = lines.rows.get(y >> lines.log2_ctb) else {
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
            line[c][sx..sx + sw].copy_from_slice(&self.planes[c][at..at + sw]);
        }
    }
}

/// The bottom sample line of every CTB row, for the row below when another
/// worker decodes it.
pub struct Lines {
    log2_ctb: u32,
    rows: Vec<Mutex<[Vec<u16>; 3]>>,
}

impl Lines {
    pub fn new(width: usize, ctb_rows: usize, log2_ctb: u32) -> Self {
        Self {
            log2_ctb,
            rows: (0..ctb_rows)
                .map(|_| Mutex::new([vec![0; width], vec![0; width / 2], vec![0; width / 2]]))
                .collect(),
        }
    }
}

/// Luma, Cb and Cr rows of one band.
pub type Band<'a> = [&'a mut [u16]; 3];

/// Split the planes into bands of luma rows starting at `tops` (the first
/// is 0, each even), the last ending at `height`.
pub fn split_bands<'a>(
    planes: &'a mut [Vec<u16>; 3],
    width: usize,
    tops: &[usize],
    height: usize,
) -> Vec<Mutex<Band<'a>>> {
    let [y, u, v] = planes;
    let (mut y, mut u, mut v) = (y.as_mut_slice(), u.as_mut_slice(), v.as_mut_slice());
    let mut out = Vec::with_capacity(tops.len());
    for (i, &top) in tops.iter().enumerate() {
        let bottom = tops.get(i + 1).map_or(height, |&t| t);
        let rows = bottom - top;
        let luma = (rows * width).min(y.len());
        let chroma = (rows / 2 * (width / 2)).min(u.len()).min(v.len());
        let (band_y, rest_y) = std::mem::take(&mut y).split_at_mut(luma);
        let (band_u, rest_u) = std::mem::take(&mut u).split_at_mut(chroma);
        let (band_v, rest_v) = std::mem::take(&mut v).split_at_mut(chroma);
        (y, u, v) = (rest_y, rest_u, rest_v);
        out.push(Mutex::new([band_y, band_u, band_v]));
    }
    out
}

pub fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Fresh planes for a picture of `width`×`height` luma samples.
pub fn planes(width: usize, height: usize) -> Result<[Vec<u16>; 3], CodecError> {
    let luma = width.checked_mul(height).ok_or(CodecError::Limit {
        what: "picture size",
    })?;
    Ok([vec![0; luma], vec![0; luma / 4], vec![0; luma / 4]])
}

impl Picture {
    pub fn new(
        sps: Arc<Sps>,
        pps: Arc<Pps>,
        layout: Arc<Layout>,
        poc: i32,
    ) -> Result<Self, CodecError> {
        let width = sps.width as usize;
        let height = sps.height as usize;
        width.checked_mul(height).ok_or(CodecError::Limit {
            what: "picture size",
        })?;
        let w4 = width / 4;
        let h4 = height / 4;
        let ctbs = layout.ctb_count() as usize;
        Ok(Self {
            width,
            height,
            w4,
            mvf: Motion::new(w4 * h4),
            info: Bytes::new(w4 * h4, 0),
            intra_mode: Bytes::new(w4 * h4, 1),
            qp: Bytes::new(w4 * h4, 0),
            depth: Bytes::new(w4 * h4, 0),
            bs_ver: Bytes::new((width / 8 + 1) * h4, 0),
            bs_hor: Bytes::new((height / 8 + 1) * w4, 0),
            ctb_slice: (0..ctbs).map(|_| AtomicU16::new(u16::MAX)).collect(),
            ctb_slice_addr: (0..ctbs).map(|_| AtomicU32::new(u32::MAX)).collect(),
            sao: Sao::new(ctbs),
            slices: Vec::new(),
            poc,
            dependent: Mutex::new(([0; crate::cabac::CONTEXTS], 0)),
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
        (y >> 2) * self.w4 + (x >> 2)
    }

    #[inline]
    pub fn ctb_rs(&self, x: usize, y: usize) -> usize {
        let log2 = self.layout.log2_ctb;
        (y >> log2) * self.layout.ctb_width as usize + (x >> log2)
    }

    /// `MinTbAddrZs` of a luma position (clause 6.5.2), computed from the
    /// tile scan and the z-order inside the CTB.
    #[inline]
    pub fn zs(&self, x: usize, y: usize) -> u64 {
        let log2_ctb = u32::from(self.layout.log2_ctb);
        let log2_tb = u32::from(self.sps.log2_min_tb);
        let rs = self.ctb_rs(x, y);
        let ts = u64::from(self.layout.rs_to_ts[rs]);
        let mask = (1usize << log2_ctb) - 1;
        let tx = ((x & mask) >> log2_tb) as u64;
        let ty = ((y & mask) >> log2_tb) as u64;
        let mut morton = 0u64;
        for bit in 0..(log2_ctb - log2_tb) {
            morton |= ((tx >> bit) & 1) << (2 * bit);
            morton |= ((ty >> bit) & 1) << (2 * bit + 1);
        }
        (ts << (2 * (log2_ctb - log2_tb))) | morton
    }

    /// Z-scan availability (clause 6.4.1).
    #[inline]
    pub fn available(&self, x: usize, y: usize, nx: isize, ny: isize) -> bool {
        if nx < 0 || ny < 0 || nx as usize >= self.width || ny as usize >= self.height {
            return false;
        }
        let (nx, ny) = (nx as usize, ny as usize);
        let nb_ctb = self.ctb_rs(nx, ny);
        let cur_ctb = self.ctb_rs(x, y);
        if self.slice_addr(nb_ctb) != self.slice_addr(cur_ctb)
            || self.layout.tile_of[nb_ctb] != self.layout.tile_of[cur_ctb]
        {
            return false;
        }
        if nb_ctb != cur_ctb {
            return self.layout.rs_to_ts[nb_ctb] < self.layout.rs_to_ts[cur_ctb];
        }
        self.zs(nx, ny) <= self.zs(x, y)
    }

    pub fn slice_of(&self, x: usize, y: usize) -> Option<&SliceCtx> {
        let index = self.ctb_slice.get(self.ctb_rs(x, y))?.load(Relaxed);
        self.slices.get(usize::from(index))
    }
}

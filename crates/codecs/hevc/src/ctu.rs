//! Slice segment data (clause 7.3.8): coding tree units, coding units,
//! prediction and transform units, residual coding, SAO syntax, and the
//! reconstruction each of them drives.

use std::sync::atomic::Ordering::Relaxed;

use cutlass_decode::CodecError;

use crate::cabac::{self, Cabac};
use crate::intra::{self, Refs};
use cutlass_pool::Rows;

use crate::job::Pipeline;
use crate::picture::{
    FLAG_CBF, FLAG_KEEP, FLAG_PCM, FLAG_SKIP, MODE_INTER, MODE_INTRA, MvField, Picture, Samples,
    SaoParams, Work, lock,
};
use crate::pixel::Pixel;
use crate::ps::{Pps, Sps};
use crate::slice::SliceType;
use crate::transform;

pub const PART_2NX2N: u8 = 0;
pub const PART_2NXN: u8 = 1;
pub const PART_NX2N: u8 = 2;
pub const PART_NXN: u8 = 3;
pub const PART_2NXNU: u8 = 4;
pub const PART_2NXND: u8 = 5;
pub const PART_NLX2N: u8 = 6;
pub const PART_NRX2N: u8 = 7;

/// A transform block: its position, the position of its parent (where a
/// 4×4 luma split puts the chroma block), its size, and its index.
#[derive(Clone, Copy)]
struct Tb {
    x: usize,
    y: usize,
    base: (usize, usize),
    log2: u32,
    blk: usize,
}

/// Coding unit state the prediction and transform units read.
#[derive(Clone, Copy, Default)]
pub struct Cu {
    pub x: usize,
    pub y: usize,
    pub log2: u32,
    pub bypass: bool,
    pub intra: bool,
    pub part: u8,
    pub luma_modes: [u8; 4],
    pub chroma_mode: u8,
    pub max_trafo_depth: u32,
    pub intra_split: bool,
}

/// One CTB row of a wavefront slice segment, decoded beside the rows above
/// and below it.
#[derive(Clone, Copy)]
pub struct Wave<'p> {
    /// The CTB row this worker starts at.
    pub row: usize,
    /// Progress of the row above, by CTB column. `None` when that row is done.
    pub above: Option<&'p Rows>,
    /// Progress of this row.
    pub mine: &'p Rows,
    /// Stop at the end of the row: a later worker decodes the next one.
    pub row_only: bool,
}

pub struct SliceDecoder<'a, 'p, P: Pixel> {
    pub pic: &'p Picture,
    pub work: &'p Work<P>,
    /// The CTB row being reconstructed.
    pub out: Samples<'p, P>,
    /// Told of every CTB reconstructed, to run the loop filters behind.
    pub pipe: &'p Pipeline<'p, P>,
    /// The first CTB, raster order, when it is not the segment's first.
    pub start_rs: Option<usize>,
    pub wave: Option<Wave<'p>>,
    pub slice: usize,
    pub sps: &'a Sps,
    pub pps: &'a Pps,
    pub cabac: Cabac<'a>,
    /// Data of the whole slice segment and where each substream starts.
    pub data: &'a [u8],
    pub entries: Vec<usize>,
    pub next_entry: usize,
    pub qp_y: i32,
    pub qp_pred: i32,
    pub last_qp: i32,
    pub first_qg: bool,
    pub qp_delta_coded: bool,
    pub qp_delta: i32,
    pub cu: Cu,
    /// Scaled coefficients: zero outside the block under way.
    pub coeffs: [i32; 32 * 32],
    /// The residual of the block under way.
    pub res: [i32; 32 * 32],
    pub pred: [[i16; 64 * 64]; 2],
    pub ctb_x: usize,
    pub ctb_y: usize,
    pub ctb_rs: usize,
    pub merge_2nx2n: bool,
    pub slice_type: SliceType,
    pub mc: crate::mc::Scratch<P>,
}

/// The scans of clause 6.5.3 to 6.5.5 as (x, y): up-right diagonal,
/// horizontal, vertical.
const fn scan(size: usize, kind: usize) -> [(u8, u8); 64] {
    let mut out = [(0u8, 0u8); 64];
    let mut i = 0;
    if kind == 0 {
        let (mut x, mut y) = (0i32, 0i32);
        while i < size * size {
            while y >= 0 {
                if (x as usize) < size && (y as usize) < size {
                    out[i] = (x as u8, y as u8);
                    i += 1;
                }
                y -= 1;
                x += 1;
            }
            y = x;
            x = 0;
        }
    } else {
        while i < size * size {
            let (a, b) = ((i % size) as u8, (i / size) as u8);
            out[i] = if kind == 1 { (a, b) } else { (b, a) };
            i += 1;
        }
    }
    out
}

/// The scans of a 4×4 block, by scan kind.
const SCAN4: [[(u8, u8); 16]; 3] = {
    let mut out = [[(0u8, 0u8); 16]; 3];
    let mut k = 0;
    while k < 3 {
        let s = scan(4, k);
        let mut i = 0;
        while i < 16 {
            out[k][i] = s[i];
            i += 1;
        }
        k += 1;
    }
    out
};

/// The sub-block scans of blocks with 1, 2, 4 and 8 sub-blocks per edge,
/// by scan kind (the horizontal and vertical scans occur for 4×4 and 8×8
/// blocks only, so larger ones take the diagonal).
const SUB_SCANS: [[[(u8, u8); 64]; 3]; 4] = {
    let mut out = [[[(0u8, 0u8); 64]; 3]; 4];
    let mut l = 0;
    while l < 4 {
        let mut k = 0;
        while k < 3 {
            out[l][k] = scan(1 << l, if l > 1 { 0 } else { k });
            k += 1;
        }
        l += 1;
    }
    out
};

/// Each position's place in its scan: `INV4[kind][y * 4 + x]`.
const INV4: [[u8; 16]; 3] = {
    let mut out = [[0u8; 16]; 3];
    let mut k = 0;
    while k < 3 {
        let mut i = 0;
        while i < 16 {
            let (x, y) = SCAN4[k][i];
            out[k][y as usize * 4 + x as usize] = i as u8;
            i += 1;
        }
        k += 1;
    }
    out
};

/// Each sub-block's place in its scan: `INV_SUB[log2 - 2][kind][ys * 8 + xs]`.
const INV_SUB: [[[u8; 64]; 3]; 4] = {
    let mut out = [[[0u8; 64]; 3]; 4];
    let mut l = 0;
    while l < 4 {
        let mut k = 0;
        while k < 3 {
            let mut i = 0;
            while i < (1 << (2 * l)) {
                let (x, y) = SUB_SCANS[l][k][i];
                out[l][k][y as usize * 8 + x as usize] = i as u8;
                i += 1;
            }
            k += 1;
        }
        l += 1;
    }
    out
};

fn sub_scan(log2s: usize, kind: usize) -> &'static [(u8, u8); 64] {
    &SUB_SCANS[log2s.min(3)][kind.min(2)]
}

/// `ctxIdxMap` of equation 9-50 for 4×4 blocks.
const CTX_IDX_MAP: [u8; 16] = [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 8];

/// `sigCtx` of a 4×4 block by scan position: `SIG4[kind][p]`.
const SIG4: [[u8; 16]; 3] = {
    let mut out = [[0u8; 16]; 3];
    let mut k = 0;
    while k < 3 {
        let mut p = 0;
        while p < 16 {
            let (x, y) = SCAN4[k][p];
            out[k][p] = CTX_IDX_MAP[(y as usize) * 4 + x as usize];
            p += 1;
        }
        k += 1;
    }
    out
};

/// The part of `sigCtx` of larger blocks that depends on the coded flags
/// of the sub-blocks to the right and below (`prev`) and the position in
/// the sub-block, by scan position: `SIG_PATTERN[kind][prev][p]`.
const SIG_PATTERN: [[[u8; 16]; 4]; 3] = {
    let mut out = [[[0u8; 16]; 4]; 3];
    let mut k = 0;
    while k < 3 {
        let mut prev = 0;
        while prev < 4 {
            let mut p = 0;
            while p < 16 {
                let (xp, yp) = SCAN4[k][p];
                let (xp, yp) = (xp as usize, yp as usize);
                out[k][prev][p] = match prev {
                    0 => {
                        if xp + yp == 0 {
                            2
                        } else if xp + yp < 3 {
                            1
                        } else {
                            0
                        }
                    }
                    1 => {
                        if yp == 0 {
                            2
                        } else if yp == 1 {
                            1
                        } else {
                            0
                        }
                    }
                    2 => {
                        if xp == 0 {
                            2
                        } else if xp == 1 {
                            1
                        } else {
                            0
                        }
                    }
                    _ => 2,
                };
                p += 1;
            }
            prev += 1;
        }
        k += 1;
    }
    out
};

/// How the coefficients of one block are scaled.
#[derive(Clone, Copy)]
enum Scale<'a> {
    /// Transquant bypass: as decoded.
    Bypass,
    /// One factor for every position.
    Flat { factor: i64, add: i64, shift: u32 },
    /// A scaling list: its 8×8 base upsampled by `1 << ratio_log2` (`None`:
    /// a 4×4 list), and its DC value.
    List {
        base: &'a [u8; 64],
        dc: Option<u8>,
        ratio_log2: Option<u32>,
        factor: i64,
        add: i64,
        shift: u32,
    },
}

impl Scale<'_> {
    #[inline(always)]
    fn apply(&self, value: i64, x: usize, y: usize) -> i32 {
        match *self {
            Self::Bypass => value as i32,
            Self::Flat { factor, add, shift } => {
                ((value * factor + add) >> shift).clamp(-32768, 32767) as i32
            }
            Self::List {
                base,
                dc,
                ratio_log2,
                factor,
                add,
                shift,
            } => {
                let m = match (dc, ratio_log2) {
                    (Some(dc), _) if x == 0 && y == 0 => dc,
                    // A 4×4 list is 4×4, in rows of 4.
                    (_, None) => base[(y & 3) * 4 + (x & 3)],
                    (_, Some(r)) => base[((y >> r) & 7) * 8 + ((x >> r) & 7)],
                };
                ((value * i64::from(m) * factor + add) >> shift).clamp(-32768, 32767) as i32
            }
        }
    }
}

/// Chroma QP of 4:2:0 (Table 8-10).
pub fn chroma_qp(qpi: i32) -> i32 {
    if qpi < 30 {
        qpi
    } else if qpi >= 43 {
        qpi - 6
    } else {
        const TABLE: [i32; 13] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37];
        TABLE[(qpi - 30) as usize]
    }
}

fn truncated(what: &'static str) -> CodecError {
    CodecError::Truncated { at_bit: 0, what }
}

fn malformed(what: &'static str) -> CodecError {
    CodecError::Malformed { at_bit: 0, what }
}

impl<'a, 'p, P: Pixel> SliceDecoder<'a, 'p, P> {
    /// Decode every CTU of the slice segment.
    pub fn decode(&mut self) -> Result<(), CodecError> {
        let header = self.pic.slices[self.slice].header.clone();
        let layout = self.pic.layout.clone();
        let ctb_size = 1usize << self.sps.log2_ctb;
        let ctb_w = layout.ctb_width as usize;
        let total = layout.ctb_count() as usize;
        let start_rs = self.start_rs.unwrap_or(header.segment_address as usize);
        let Some(&start_ts) = layout.rs_to_ts.get(start_rs) else {
            return Err(malformed("slice runs past the picture"));
        };
        let mut ts = start_ts as usize;
        let slice_addr = self.pic.slices[self.slice].slice_addr;
        let mut first = self.start_rs.is_none();
        let height = self.pic.height;
        loop {
            let Some(&rs) = layout.ts_to_rs.get(ts) else {
                return Err(malformed("slice runs past the picture"));
            };
            let rs = rs as usize;
            let col = rs % ctb_w;
            let in_wave = self.wave.filter(|w| w.row == rs / ctb_w);
            if let Some(above) = in_wave.and_then(|w| w.above) {
                above.wait(col as u32 + 1);
            }
            if self.pic.ctb_slice[rs].load(Relaxed) != u16::MAX {
                return Err(malformed("a CTB decoded twice"));
            }
            self.ctb_rs = rs;
            self.ctb_x = col * ctb_size;
            self.ctb_y = (rs / ctb_w) * ctb_size;
            if self.out.row() != rs / ctb_w {
                self.out = self.work.band(rs / ctb_w);
                self.pipe.nudge(Some(rs / ctb_w));
            }
            self.pic.ctb_slice[rs].store(self.slice as u16, Relaxed);
            self.pic.ctb_slice_addr[rs].store(slice_addr, Relaxed);
            let tile_start =
                ts == 0 || layout.tile_of[rs] != layout.tile_of[layout.ts_to_rs[ts - 1] as usize];
            let row_start =
                self.pps.entropy_sync && col as u32 == layout.tile_col_start(col as u32);
            if first {
                if !header.dependent || tile_start {
                    self.reset_contexts();
                    self.first_qg = true;
                } else if row_start {
                    self.sync_wpp();
                    self.first_qg = true;
                } else {
                    (self.cabac.ctx, self.last_qp) = *lock(&self.pic.dependent);
                    self.first_qg = false;
                }
            } else if tile_start || row_start {
                self.next_substream()?;
                if tile_start {
                    self.reset_contexts();
                } else {
                    self.sync_wpp();
                }
                self.first_qg = true;
            }
            first = false;
            self.coding_tree_unit()?;
            if self.pps.entropy_sync
                && col as u32 == layout.tile_col_start(col as u32) + 1
                && let Some(slot) = lock(&self.pic.wpp_ctx).get_mut(rs)
            {
                *slot = Some(Box::new(self.cabac.ctx));
            }
            self.out.publish(self.ctb_x, self.ctb_y, ctb_size, height);
            if let Some(wave) = in_wave {
                wave.mine.report_through(col as u32);
            }
            self.pipe.ctb_done(rs);
            let end = self.cabac.terminate();
            if self.cabac.overrun() {
                return Err(truncated("slice data"));
            }
            ts += 1;
            if end {
                if self.next_entry < self.entries.len() {
                    return Err(malformed("slice ends before its last entry point"));
                }
                if self.pps.dependent_slices {
                    *lock(&self.pic.dependent) = (self.cabac.ctx, self.last_qp);
                }
                return Ok(());
            }
            if ts >= total {
                return Err(malformed("slice runs past the picture"));
            }
            let next_rs = layout.ts_to_rs[ts] as usize;
            let next_col = next_rs % ctb_w;
            let next_tile = layout.tile_of[next_rs] != layout.tile_of[rs];
            let next_row =
                self.pps.entropy_sync && next_col as u32 == layout.tile_col_start(next_col as u32);
            if (next_tile || next_row) && !self.cabac.terminate() {
                return Err(malformed("end_of_subset_one_bit"));
            }
            if next_row && in_wave.is_some_and(|w| w.row_only) {
                return Ok(());
            }
        }
    }

    fn next_substream(&mut self) -> Result<(), CodecError> {
        let start = if let Some(&at) = self.entries.get(self.next_entry) {
            self.next_entry += 1;
            at
        } else {
            self.data.len() - self.cabac.remaining().len()
        };
        let Some(rest) = self.data.get(start..) else {
            return Err(truncated("entry point"));
        };
        if rest.is_empty() {
            return Err(truncated("entry point"));
        }
        self.cabac.restart(rest);
        Ok(())
    }

    fn reset_contexts(&mut self) {
        let header = &self.pic.slices[self.slice].header;
        let init_type = match header.slice_type {
            SliceType::I => 0,
            SliceType::P => {
                if header.cabac_init {
                    2
                } else {
                    1
                }
            }
            SliceType::B => {
                if header.cabac_init {
                    1
                } else {
                    2
                }
            }
        };
        self.cabac.ctx = cabac::init_contexts(init_type, self.pic.slices[self.slice].qp);
    }

    /// The first CTB of a row with wavefronts takes the contexts stored
    /// after the CTB above and to the right when that CTB is available
    /// (clause 9.3.1); otherwise it initializes them.
    fn sync_wpp(&mut self) {
        let ctb = 1usize << self.sps.log2_ctb;
        let (x, y) = (self.ctb_x, self.ctb_y);
        if y >= ctb
            && self
                .pic
                .available(x, y, (x + ctb) as isize, (y - ctb) as isize)
        {
            let ctb_w = self.pic.layout.ctb_width as usize;
            let above_right = self.ctb_rs - ctb_w + 1;
            if let Some(Some(ctx)) = lock(&self.pic.wpp_ctx).get(above_right) {
                self.cabac.ctx = **ctx;
                return;
            }
        }
        self.reset_contexts();
    }

    fn coding_tree_unit(&mut self) -> Result<(), CodecError> {
        let header = &self.pic.slices[self.slice].header;
        if header.sao_luma || header.sao_chroma {
            self.sao_syntax()?;
        }
        self.coding_quadtree(self.ctb_x, self.ctb_y, u32::from(self.sps.log2_ctb), 0)
    }

    fn sao_syntax(&mut self) -> Result<(), CodecError> {
        let layout = self.pic.layout.clone();
        let rs = self.ctb_rs;
        let ctb_w = layout.ctb_width as usize;
        let slice_addr = self.pic.slices[self.slice].slice_addr as usize;
        let (luma_on, chroma_on) = {
            let h = &self.pic.slices[self.slice].header;
            (h.sao_luma, h.sao_chroma)
        };
        let mut merge_left = false;
        if !rs.is_multiple_of(ctb_w)
            && rs > slice_addr
            && layout.tile_of[rs] == layout.tile_of[rs - 1]
        {
            merge_left = self.cabac.decide(cabac::SAO_MERGE);
        }
        if merge_left {
            self.pic.sao.set(rs, self.pic.sao.get(rs - 1));
            return Ok(());
        }
        if rs >= ctb_w
            && rs - ctb_w >= slice_addr
            && layout.tile_of[rs] == layout.tile_of[rs - ctb_w]
            && self.cabac.decide(cabac::SAO_MERGE)
        {
            self.pic.sao.set(rs, self.pic.sao.get(rs - ctb_w));
            return Ok(());
        }
        let bit_depth = u32::from(self.sps.bit_depth);
        let max_abs = (1u32 << (bit_depth.min(10) - 5)) - 1;
        let mut params = [SaoParams::default(); 3];
        for c in 0..3 {
            let on = if c == 0 { luma_on } else { chroma_on };
            if !on {
                continue;
            }
            if c == 2 {
                params[2].kind = params[1].kind;
            } else {
                params[c].kind = if !self.cabac.decide(cabac::SAO_TYPE) {
                    0
                } else if self.cabac.bypass() {
                    2
                } else {
                    1
                };
            }
            if params[c].kind == 0 {
                continue;
            }
            let mut abs = [0i16; 4];
            for value in &mut abs {
                let mut v = 0;
                while v < max_abs && self.cabac.bypass() {
                    v += 1;
                }
                *value = v as i16;
            }
            if params[c].kind == 1 {
                for value in &mut abs {
                    if *value != 0 && self.cabac.bypass() {
                        *value = -*value;
                    }
                }
                params[c].class = self.cabac.bypass_bits(5) as u8;
            } else {
                abs[2] = -abs[2];
                abs[3] = -abs[3];
                if c == 0 {
                    params[0].class = self.cabac.bypass_bits(2) as u8;
                } else if c == 1 {
                    params[1].class = self.cabac.bypass_bits(2) as u8;
                } else {
                    params[2].class = params[1].class;
                }
            }
            for (out, value) in params[c].offsets.iter_mut().zip(abs) {
                *out = value;
            }
        }
        self.pic.sao.set(rs, params);
        Ok(())
    }

    fn coding_quadtree(
        &mut self,
        x0: usize,
        y0: usize,
        log2: u32,
        depth: u32,
    ) -> Result<(), CodecError> {
        let size = 1usize << log2;
        let min_cb = u32::from(self.sps.log2_min_cb);
        let split = if x0 + size <= self.pic.width && y0 + size <= self.pic.height && log2 > min_cb
        {
            let mut inc = 0;
            if self.pic.available(x0, y0, x0 as isize - 1, y0 as isize)
                && u32::from(self.pic.depth.get(self.pic.at4(x0 - 1, y0))) > depth
            {
                inc += 1;
            }
            if self.pic.available(x0, y0, x0 as isize, y0 as isize - 1)
                && u32::from(self.pic.depth.get(self.pic.at4(x0, y0 - 1))) > depth
            {
                inc += 1;
            }
            self.cabac.decide(cabac::SPLIT_CU + inc)
        } else {
            log2 > min_cb
        };
        let log2_qg = u32::from(self.sps.log2_ctb) - u32::from(self.pps.diff_cu_qp_delta_depth);
        if log2 >= log2_qg {
            self.start_qg(x0, y0);
        }
        if split {
            let half = size / 2;
            for (dx, dy) in [(0, 0), (half, 0), (0, half), (half, half)] {
                let (x, y) = (x0 + dx, y0 + dy);
                if x < self.pic.width && y < self.pic.height {
                    self.coding_quadtree(x, y, log2 - 1, depth + 1)?;
                }
            }
            Ok(())
        } else {
            self.coding_unit(x0, y0, log2, depth)
        }
    }

    /// Start of a quantization group: `qPY_PRED` (clause 8.6.1).
    fn start_qg(&mut self, x: usize, y: usize) {
        self.qp_delta_coded = false;
        self.qp_delta = 0;
        // The flag stays up until a coding unit ends: a 64×64 CTB with
        // 32×32 groups starts a group at both depths.
        let prev = if self.first_qg {
            self.pic.slices[self.slice].qp
        } else {
            self.last_qp
        };
        let ctb = self.ctb_rs;
        let neighbour = |pic: &Picture, nx: isize, ny: isize| -> i32 {
            if pic.available(x, y, nx, ny) && pic.ctb_rs(nx as usize, ny as usize) == ctb {
                i32::from(pic.qp.get(pic.at4(nx as usize, ny as usize)) as i8)
            } else {
                prev
            }
        };
        let a = neighbour(self.pic, x as isize - 1, y as isize);
        let b = neighbour(self.pic, x as isize, y as isize - 1);
        self.qp_pred = (a + b + 1) >> 1;
        self.qp_y = self.qp_pred;
    }

    fn set_qp(&mut self) {
        let offset = self.sps.qp_bd_offset();
        self.qp_y =
            (self.qp_pred + self.qp_delta + 52 + 2 * offset).rem_euclid(52 + offset) - offset;
    }

    fn coding_unit(
        &mut self,
        x0: usize,
        y0: usize,
        log2: u32,
        depth: u32,
    ) -> Result<(), CodecError> {
        let size = 1usize << log2;
        let w4 = self.pic.w4;
        self.cu = Cu {
            x: x0,
            y: y0,
            log2,
            ..Cu::default()
        };
        if self.pps.transquant_bypass {
            self.cu.bypass = self.cabac.decide(cabac::TRANSQUANT_BYPASS);
        }
        self.pic
            .depth
            .fill4(w4, (x0, y0), (size, size), depth as u8);
        let mut skip = false;
        if self.slice_type != SliceType::I {
            let mut inc = 0;
            if self.pic.available(x0, y0, x0 as isize - 1, y0 as isize)
                && self.pic.info.get(self.pic.at4(x0 - 1, y0)) & FLAG_SKIP != 0
            {
                inc += 1;
            }
            if self.pic.available(x0, y0, x0 as isize, y0 as isize - 1)
                && self.pic.info.get(self.pic.at4(x0, y0 - 1)) & FLAG_SKIP != 0
            {
                inc += 1;
            }
            skip = self.cabac.decide(cabac::SKIP + inc);
        }
        let keep = if self.cu.bypass { FLAG_KEEP } else { 0 };
        if skip {
            self.pic
                .info
                .fill4(w4, (x0, y0), (size, size), MODE_INTER | FLAG_SKIP | keep);
            self.cu.part = PART_2NX2N;
            self.prediction_unit(x0, y0, (size, size), depth, 0, true)?;
            self.merge_2nx2n = true;
            self.deblock_cu(false)?;
            self.finish_cu();
            return Ok(());
        }
        let intra = if self.slice_type != SliceType::I {
            self.cabac.decide(cabac::PRED_MODE)
        } else {
            true
        };
        self.cu.intra = intra;
        let min_cb = u32::from(self.sps.log2_min_cb);
        let part = if !intra || log2 == min_cb {
            self.part_mode(log2)
        } else {
            PART_2NX2N
        };
        self.cu.part = part;
        let mode = if intra { MODE_INTRA } else { MODE_INTER };
        self.pic.info.fill4(w4, (x0, y0), (size, size), mode | keep);
        if intra {
            // No motion: the picture's maps are not cleared between pictures.
            for row in (y0 >> 2)..((y0 + size).min(self.pic.height) >> 2) {
                let start = row * w4 + (x0 >> 2);
                self.pic.mvf.fill(start, size >> 2, MvField::default());
            }
        }
        let mut pcm = false;
        if intra {
            self.cu.intra_split = part == PART_NXN;
            if let Some(pcm_params) = self.sps.pcm
                && part == PART_2NX2N
                && log2 >= u32::from(pcm_params.log2_min)
                && log2 <= u32::from(pcm_params.log2_max)
            {
                pcm = self.cabac.terminate();
            }
            if pcm {
                self.pcm_sample(x0, y0, log2)?;
                let flags = MODE_INTRA
                    | FLAG_PCM
                    | if self.sps.pcm.is_some_and(|p| p.loop_filter_disabled) || self.cu.bypass {
                        FLAG_KEEP
                    } else {
                        0
                    };
                self.pic.info.fill4(w4, (x0, y0), (size, size), flags);
                self.pic.intra_mode.fill4(w4, (x0, y0), (size, size), 1);
            } else {
                self.intra_modes(x0, y0, log2)?;
            }
        } else {
            self.merge_2nx2n = false;
            let n = size;
            let q = size / 4;
            let parts: &[(usize, usize, usize, usize)] = match part {
                PART_2NX2N => &[(0, 0, n, n)],
                PART_2NXN => &[(0, 0, n, n / 2), (0, n / 2, n, n / 2)],
                PART_NX2N => &[(0, 0, n / 2, n), (n / 2, 0, n / 2, n)],
                PART_2NXNU => &[(0, 0, n, q), (0, q, n, n - q)],
                PART_2NXND => &[(0, 0, n, n - q), (0, n - q, n, q)],
                PART_NLX2N => &[(0, 0, q, n), (q, 0, n - q, n)],
                PART_NRX2N => &[(0, 0, n - q, n), (n - q, 0, q, n)],
                _ => &[
                    (0, 0, n / 2, n / 2),
                    (n / 2, 0, n / 2, n / 2),
                    (0, n / 2, n / 2, n / 2),
                    (n / 2, n / 2, n / 2, n / 2),
                ],
            };
            for (idx, &(dx, dy, w, h)) in parts.iter().enumerate() {
                self.prediction_unit(x0 + dx, y0 + dy, (w, h), depth, idx, false)?;
            }
        }
        if !pcm {
            let root = if !intra && !(part == PART_2NX2N && self.merge_2nx2n) {
                self.cabac.decide(cabac::RQT_ROOT_CBF)
            } else {
                true
            };
            if root {
                self.cu.max_trafo_depth = if intra {
                    u32::from(self.sps.max_th_depth_intra) + u32::from(self.cu.intra_split)
                } else {
                    u32::from(self.sps.max_th_depth_inter)
                };
                let root = Tb {
                    x: x0,
                    y: y0,
                    base: (x0, y0),
                    log2,
                    blk: 0,
                };
                self.transform_tree(root, 0, [false; 2])?;
            } else {
                self.deblock_cu(false)?;
            }
        } else {
            self.deblock_cu(false)?;
        }
        if !intra {
            self.deblock_pu_edges();
        }
        self.finish_cu();
        Ok(())
    }

    fn finish_cu(&mut self) {
        let size = 1usize << self.cu.log2;
        let w4 = self.pic.w4;
        let qp = self.qp_y as i8 as u8;
        self.pic
            .qp
            .fill4(w4, (self.cu.x, self.cu.y), (size, size), qp);
        self.last_qp = self.qp_y;
        self.first_qg = false;
    }

    fn part_mode(&mut self, log2: u32) -> u8 {
        if self.cabac.decide(cabac::PART_MODE) {
            return PART_2NX2N;
        }
        let min_cb = u32::from(self.sps.log2_min_cb);
        if log2 == min_cb {
            if self.cu.intra {
                return PART_NXN;
            }
            if self.cabac.decide(cabac::PART_MODE + 1) {
                return PART_2NXN;
            }
            if log2 == 3 {
                return PART_NX2N;
            }
            if self.cabac.decide(cabac::PART_MODE + 2) {
                return PART_NX2N;
            }
            return PART_NXN;
        }
        if !self.sps.amp {
            if self.cabac.decide(cabac::PART_MODE + 1) {
                return PART_2NXN;
            }
            return PART_NX2N;
        }
        if self.cabac.decide(cabac::PART_MODE + 1) {
            if self.cabac.decide(cabac::PART_MODE + 3) {
                return PART_2NXN;
            }
            if self.cabac.bypass() {
                return PART_2NXND;
            }
            return PART_2NXNU;
        }
        if self.cabac.decide(cabac::PART_MODE + 3) {
            return PART_NX2N;
        }
        if self.cabac.bypass() {
            return PART_NRX2N;
        }
        PART_NLX2N
    }

    fn pcm_sample(&mut self, x0: usize, y0: usize, log2: u32) -> Result<(), CodecError> {
        let Some(pcm) = self.sps.pcm else {
            return Err(malformed("pcm without pcm parameters"));
        };
        let rest = self.cabac.remaining();
        let size = 1usize << log2;
        let luma_bits = u32::from(pcm.bit_depth);
        let chroma_bits = u32::from(pcm.bit_depth_chroma);
        let total_bits =
            size * size * luma_bits as usize + 2 * (size / 2) * (size / 2) * chroma_bits as usize;
        let bytes = total_bits.div_ceil(8);
        if rest.len() < bytes {
            return Err(truncated("pcm samples"));
        }
        let mut reader = crate::bits::Bits::new(&rest[..bytes]);
        let bit_depth = u32::from(self.sps.bit_depth);
        let (plane, stride) = self.out.from(0, x0, y0);
        for y in 0..size {
            for x in 0..size {
                let v = reader.u(luma_bits)?;
                plane[y * stride + x] = P::of((v << (bit_depth - luma_bits)) as i32);
            }
        }
        for c in 1..3 {
            let (plane, stride) = self.out.from(c, x0 / 2, y0 / 2);
            for y in 0..size / 2 {
                for x in 0..size / 2 {
                    let v = reader.u(chroma_bits)?;
                    plane[y * stride + x] = P::of((v << (bit_depth - chroma_bits)) as i32);
                }
            }
        }
        self.cabac.restart(&rest[bytes..]);
        Ok(())
    }

    fn intra_modes(&mut self, x0: usize, y0: usize, log2: u32) -> Result<(), CodecError> {
        let parts = if self.cu.part == PART_NXN { 2 } else { 1 };
        let pb = (1usize << log2) / parts;
        let mut prev = [false; 4];
        for flag in prev.iter_mut().take(parts * parts) {
            *flag = self.cabac.decide(cabac::PREV_INTRA);
        }
        let w4 = self.pic.w4;
        for j in 0..parts {
            for i in 0..parts {
                let idx = j * parts + i;
                let (x, y) = (x0 + i * pb, y0 + j * pb);
                let cand_a = self.candidate_mode(x, y, x as isize - 1, y as isize, false);
                let cand_b = self.candidate_mode(x, y, x as isize, y as isize - 1, true);
                let list = if cand_a == cand_b {
                    if cand_a < 2 {
                        [0, 1, 26]
                    } else {
                        [
                            cand_a,
                            2 + ((cand_a + 29) % 32),
                            2 + ((cand_a - 2 + 1) % 32),
                        ]
                    }
                } else {
                    let c = if cand_a != 0 && cand_b != 0 {
                        0
                    } else if cand_a != 1 && cand_b != 1 {
                        1
                    } else {
                        26
                    };
                    [cand_a, cand_b, c]
                };
                let mode = if prev[idx] {
                    let mut mpm = 0;
                    while mpm < 2 && self.cabac.bypass() {
                        mpm += 1;
                    }
                    list[mpm]
                } else {
                    let mut mode = self.cabac.bypass_bits(5) as u8;
                    let mut sorted = list;
                    sorted.sort_unstable();
                    for &candidate in &sorted {
                        if mode >= candidate {
                            mode += 1;
                        }
                    }
                    mode
                };
                self.cu.luma_modes[idx] = mode;
                self.pic
                    .intra_mode
                    .fill4(w4, (x, y), (pb.max(4), pb.max(4)), mode);
            }
        }
        // intra_chroma_pred_mode
        let chroma = if !self.cabac.decide(cabac::CHROMA_PRED) {
            4
        } else {
            self.cabac.bypass_bits(2) as u8
        };
        let luma = self.cu.luma_modes[0];
        self.cu.chroma_mode = if chroma == 4 {
            luma
        } else {
            let mode = [0u8, 26, 10, 1][usize::from(chroma)];
            if mode == luma { 34 } else { mode }
        };
        Ok(())
    }

    fn candidate_mode(&self, x: usize, y: usize, nx: isize, ny: isize, above: bool) -> u8 {
        if !self.pic.available(x, y, nx, ny) {
            return 1;
        }
        if above {
            let log2 = self.sps.log2_ctb;
            if (ny as usize) < ((y >> log2) << log2) {
                return 1;
            }
        }
        let at = self.pic.at4(nx as usize, ny as usize);
        let info = self.pic.info.get(at);
        if info & 3 != MODE_INTRA || info & FLAG_PCM != 0 {
            return 1;
        }
        self.pic.intra_mode.get(at)
    }

    fn transform_tree(
        &mut self,
        tb: Tb,
        depth: u32,
        parent_cbf: [bool; 2],
    ) -> Result<(), CodecError> {
        let Tb {
            x: x0, y: y0, log2, ..
        } = tb;
        let max_tb = u32::from(self.sps.log2_max_tb);
        let min_tb = u32::from(self.sps.log2_min_tb);
        let inter_split = self.sps.max_th_depth_inter == 0
            && !self.cu.intra
            && self.cu.part != PART_2NX2N
            && depth == 0;
        let split = if log2 <= max_tb
            && log2 > min_tb
            && depth < self.cu.max_trafo_depth
            && !(self.cu.intra_split && depth == 0)
        {
            self.cabac
                .decide(cabac::SPLIT_TRANSFORM + 5 - log2 as usize)
        } else {
            log2 > max_tb || (self.cu.intra_split && depth == 0) || inter_split
        };
        let mut cbf = [false; 2];
        if log2 > 2 {
            for (c, flag) in cbf.iter_mut().enumerate() {
                if depth == 0 || parent_cbf[c] {
                    *flag = self.cabac.decide(cabac::CBF_CHROMA + depth as usize);
                }
            }
        } else {
            cbf = parent_cbf;
        }
        if split {
            let half = 1usize << (log2 - 1);
            for (i, (dx, dy)) in [(0, 0), (half, 0), (0, half), (half, half)]
                .into_iter()
                .enumerate()
            {
                let child = Tb {
                    x: x0 + dx,
                    y: y0 + dy,
                    base: (x0, y0),
                    log2: log2 - 1,
                    blk: i,
                };
                self.transform_tree(child, depth + 1, cbf)?;
            }
            return Ok(());
        }
        let cbf_luma = if self.cu.intra || depth != 0 || cbf[0] || cbf[1] {
            self.cabac.decide(cabac::CBF_LUMA + usize::from(depth == 0))
        } else {
            true
        };
        self.transform_unit(tb, cbf_luma, cbf)
    }

    fn transform_unit(&mut self, tb: Tb, cbf_luma: bool, cbf: [bool; 2]) -> Result<(), CodecError> {
        let Tb {
            x: x0,
            y: y0,
            base: (x_base, y_base),
            log2,
            blk,
        } = tb;
        let chroma_here = log2 > 2 || blk == 3;
        let any_chroma = cbf[0] || cbf[1];
        if (cbf_luma || any_chroma) && self.pps.cu_qp_delta && !self.qp_delta_coded {
            let mut abs = 0i32;
            let mut ctx = 0;
            while abs < 5 && self.cabac.decide(cabac::CU_QP_DELTA + ctx) {
                abs += 1;
                ctx = 1;
            }
            if abs == 5 {
                let mut k = 0;
                while self.cabac.bypass() {
                    k += 1;
                    if k > 30 {
                        return Err(malformed("cu_qp_delta_abs"));
                    }
                }
                abs += ((1 << k) - 1) + self.cabac.bypass_bits(k) as i32;
            }
            let delta = if abs > 0 && self.cabac.bypass() {
                -abs
            } else {
                abs
            };
            let offset = self.sps.qp_bd_offset();
            if delta < -(26 + offset / 2) || delta > 25 + offset / 2 {
                return Err(malformed("cu_qp_delta"));
            }
            self.qp_delta = delta;
            self.qp_delta_coded = true;
            self.set_qp();
        }
        let size = 1usize << log2;
        // Luma.
        if self.cu.intra {
            let mode = self.luma_mode_at(x0, y0);
            self.intra_predict(x0, y0, log2, 0, mode);
        }
        if cbf_luma {
            self.residual(x0, y0, log2, 0)?;
        }
        let w4 = self.pic.w4;
        let flag_cbf = if cbf_luma { FLAG_CBF } else { 0 };
        for row in (y0 >> 2)..((y0 + size) >> 2) {
            for col in (x0 >> 2)..((x0 + size) >> 2) {
                let at = row * w4 + col;
                self.pic
                    .info
                    .set(at, (self.pic.info.get(at) & !FLAG_CBF) | flag_cbf);
            }
        }
        self.deblock_tu(x0, y0, size);
        // Chroma.
        if chroma_here {
            let (xc, yc, log2c) = if log2 > 2 {
                (x0, y0, log2 - 1)
            } else {
                (x_base, y_base, 2)
            };
            for c in 1..3 {
                if self.cu.intra {
                    let mode = self.cu.chroma_mode;
                    self.intra_predict(xc, yc, log2c, c, mode);
                }
                if cbf[c - 1] {
                    self.residual(xc, yc, log2c, c)?;
                }
            }
        }
        Ok(())
    }

    fn luma_mode_at(&self, x: usize, y: usize) -> u8 {
        if self.cu.part == PART_NXN {
            let half = 1usize << (self.cu.log2 - 1);
            let i = usize::from(x >= self.cu.x + half);
            let j = usize::from(y >= self.cu.y + half);
            self.cu.luma_modes[j * 2 + i]
        } else {
            self.cu.luma_modes[0]
        }
    }

    /// Intra prediction of one transform block. `x`, `y` are luma positions.
    fn intra_predict(&mut self, x: usize, y: usize, log2: u32, c: usize, mode: u8) {
        let n = 1usize << log2;
        let (scale, plane_w) = if c == 0 {
            (1usize, self.pic.width)
        } else {
            (2usize, self.pic.width / 2)
        };
        let (px, py) = (x / scale, y / scale);
        let mut refs = Refs::new(n);
        let constrained = self.pps.constrained_intra_pred;
        let pic: &Picture = self.pic;
        // Availability is checked per 4 luma samples (2 chroma samples).
        let unit = 4 / scale;
        // The row above a band belongs to another worker: copy what this
        // block may read of it once.
        let mut line = [P::default(); 129];
        let line_x = px.saturating_sub(1);
        let from_line = py > 0 && !self.out.holds(c, py - 1);
        if from_line {
            let end = (px + 2 * n).min(plane_w);
            self.out.above(c, line_x, &mut line[..end - line_x]);
        }
        let out = &self.out;
        let above = |sx: usize| -> i32 {
            if from_line {
                line[sx - line_x].i()
            } else {
                out.get(c, sx, py - 1).i()
            }
        };
        let check = |pic: &Picture, sx: isize, sy: isize| -> bool {
            // sx, sy in component samples.
            let lx = sx * scale as isize;
            let ly = sy * scale as isize;
            if !pic.available(x, y, lx, ly) {
                return false;
            }
            !constrained || pic.info.get(pic.at4(lx as usize, ly as usize)) & 3 == MODE_INTRA
        };
        // Left column, bottom to top, and the corner.
        let ni = n as isize;
        let mut yy = 2 * ni - 1;
        while yy >= 0 {
            let base = yy - (yy % unit as isize);
            let ok = check(pic, px as isize - 1, py as isize + base);
            for k in 0..unit as isize {
                let sy = base + k;
                let idx = refs.left(sy as i32);
                if ok {
                    refs.samples[idx] = out.get(c, px - 1, py + sy as usize).i();
                    refs.available[idx] = true;
                }
            }
            yy = base - 1;
        }
        if check(pic, px as isize - 1, py as isize - 1) {
            let idx = refs.left(-1);
            refs.samples[idx] = above(px - 1);
            refs.available[idx] = true;
        }
        let mut xx = 0isize;
        while xx < 2 * ni {
            let ok = check(pic, px as isize + xx, py as isize - 1);
            for k in 0..unit as isize {
                let sx = xx + k;
                let idx = refs.top(sx as i32);
                if ok {
                    refs.samples[idx] = above(px + sx as usize);
                    refs.available[idx] = true;
                }
            }
            xx += unit as isize;
        }
        let bit_depth = u32::from(self.sps.bit_depth);
        refs.substitute(bit_depth);
        if c == 0 {
            intra::filter(&mut refs, mode, self.sps.strong_intra_smoothing, bit_depth);
        }
        let (dst, stride) = self.out.from(c, px, py);
        intra::predict(&refs, mode, c == 0, bit_depth, dst, stride);
    }

    /// `residual_coding` (clause 7.3.8.11) and the reconstruction of its
    /// residual into the picture. The coefficients are scaled (clause
    /// 8.6.3) as they are read, into `coeffs`, which is zero outside the
    /// rectangle of the coefficients of the block under way.
    fn residual(&mut self, x0: usize, y0: usize, log2: u32, c: usize) -> Result<(), CodecError> {
        let n = 1usize << log2;
        let mut transform_skip = false;
        if self.pps.transform_skip && !self.cu.bypass && log2 == 2 {
            transform_skip = self
                .cabac
                .decide(cabac::TRANSFORM_SKIP + usize::from(c > 0));
        }
        // last_sig_coeff prefixes.
        let (offset, shift) = if c == 0 {
            (
                3 * (log2 as usize - 2) + ((log2 as usize - 1) >> 2),
                (log2 as usize + 1) >> 2,
            )
        } else {
            (15, log2 as usize - 2)
        };
        let max = (log2 as usize) * 2 - 1;
        let mut prefix_x = 0;
        while prefix_x < max
            && self
                .cabac
                .decide(cabac::LAST_X + offset + (prefix_x >> shift))
        {
            prefix_x += 1;
        }
        let mut prefix_y = 0;
        while prefix_y < max
            && self
                .cabac
                .decide(cabac::LAST_Y + offset + (prefix_y >> shift))
        {
            prefix_y += 1;
        }
        let mut last_x = prefix_x;
        if prefix_x > 3 {
            let bits = (prefix_x >> 1) - 1;
            let suffix = self.cabac.bypass_bits(bits as u32) as usize;
            last_x = (1 << bits) * (2 + (prefix_x & 1)) + suffix;
        }
        let mut last_y = prefix_y;
        if prefix_y > 3 {
            let bits = (prefix_y >> 1) - 1;
            let suffix = self.cabac.bypass_bits(bits as u32) as usize;
            last_y = (1 << bits) * (2 + (prefix_y & 1)) + suffix;
        }
        // scanIdx (clause 7.4.9.11).
        let mut scan = 0usize;
        if self.cu.intra && (log2 == 2 || (log2 == 3 && c == 0)) {
            let mode = if c == 0 {
                self.luma_mode_at(x0, y0)
            } else {
                self.cu.chroma_mode
            };
            if (6..=14).contains(&mode) {
                scan = 2;
            } else if (22..=30).contains(&mode) {
                scan = 1;
            }
        }
        if scan == 2 {
            std::mem::swap(&mut last_x, &mut last_y);
        }
        if last_x >= n || last_y >= n {
            return Err(malformed("last significant coefficient"));
        }
        let log2s = log2 as usize - 2;
        let sub = 1usize << log2s;
        let subs = sub_scan(log2s, scan);
        // The scan positions of the last sub-block and of the last
        // coefficient inside it.
        let last_sub = usize::from(INV_SUB[log2s][scan][(last_y >> 2) * 8 + (last_x >> 2)]);
        let last_pos = usize::from(INV4[scan][(last_y & 3) * 4 + (last_x & 3)]);
        let scan4 = &SCAN4[scan];
        let scale = self.scaling(log2, c, transform_skip);
        // Coded sub-block flags, bit `ys * 8 + xs`.
        let mut csbf = 0u64;
        let sign_hiding = self.pps.sign_data_hiding && !self.cu.bypass;
        let chroma = c > 0;
        let mut greater1_state = 1u32;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        for i in (0..=last_sub).rev() {
            let (xs, ys) = subs[i];
            let (xs, ys) = (usize::from(xs), usize::from(ys));
            let right = xs + 1 < sub && csbf >> (ys * 8 + xs + 1) & 1 != 0;
            let below = ys + 1 < sub && csbf >> ((ys + 1) * 8 + xs) & 1 != 0;
            let mut infer_dc = false;
            let coded = if i < last_sub && i > 0 {
                let inc = usize::from(right || below) + if chroma { 2 } else { 0 };
                let flag = self.cabac.decide(cabac::CODED_SUB_BLOCK + inc);
                infer_dc = true;
                flag
            } else {
                true
            };
            if !coded {
                continue;
            }
            csbf |= 1 << (ys * 8 + xs);
            // Significant coefficients, high to low scan position.
            let prev = usize::from(right) | (usize::from(below) << 1);
            let (table, base): (&[u8; 16], usize) = if log2 == 2 {
                (&SIG4[scan], cabac::SIG_COEFF + if chroma { 27 } else { 0 })
            } else {
                let base = if chroma {
                    27 + if log2 == 3 { 9 } else { 12 }
                } else {
                    (if xs + ys > 0 { 3 } else { 0 })
                        + if log2 == 3 {
                            if scan == 0 { 9 } else { 15 }
                        } else {
                            21
                        }
                };
                (&SIG_PATTERN[scan][prev], cabac::SIG_COEFF + base)
            };
            let dc_ctx = cabac::SIG_COEFF + if chroma { 27 } else { 0 };
            let mut sig = [0u8; 16];
            let mut count = 0usize;
            let start = if i == last_sub {
                sig[0] = last_pos as u8;
                count = 1;
                last_pos as isize - 1
            } else {
                15
            };
            let mut p = start;
            while p > 0 {
                let pu = p as usize;
                if self.cabac.decide(base + usize::from(table[pu])) {
                    sig[count] = pu as u8;
                    count += 1;
                    infer_dc = false;
                }
                p -= 1;
            }
            if p == 0 {
                // Position 0: inferred when the sub-block is coded and no
                // other coefficient was; the DC of a large block has its
                // own context.
                let significant = if infer_dc {
                    true
                } else {
                    let ctx = if log2 > 2 && i == 0 {
                        dc_ctx
                    } else {
                        base + usize::from(table[0])
                    };
                    self.cabac.decide(ctx)
                };
                if significant {
                    sig[count] = 0;
                    count += 1;
                }
            }
            if count == 0 {
                continue;
            }
            // Greater-than-one flags.
            let mut ctx_set = if i == 0 || chroma { 0 } else { 2 };
            if greater1_state == 0 {
                ctx_set += 1;
            }
            greater1_state = 1;
            let mut greater1 = [false; 16];
            let mut first_g1 = None;
            let g1_base = cabac::GREATER1 + ctx_set * 4 + if chroma { 16 } else { 0 };
            for (k, flag) in greater1.iter_mut().enumerate().take(count.min(8)) {
                *flag = self.cabac.decide(g1_base + greater1_state.min(3) as usize);
                if *flag {
                    greater1_state = 0;
                    if first_g1.is_none() {
                        first_g1 = Some(k);
                    }
                } else if greater1_state > 0 && greater1_state < 3 {
                    greater1_state += 1;
                }
            }
            let mut greater2 = false;
            if first_g1.is_some() {
                greater2 = self
                    .cabac
                    .decide(cabac::GREATER2 + ctx_set + if chroma { 4 } else { 0 });
            }
            let last_sig_pos = usize::from(sig[0]);
            let first_sig_pos = usize::from(sig[count - 1]);
            let hidden = sign_hiding && last_sig_pos - first_sig_pos > 3;
            let coded_signs = if hidden { count - 1 } else { count };
            let signs = self.cabac.bypass_bits(coded_signs as u32) << (32 - coded_signs as u32);
            // Remaining levels.
            let mut rice = 0u32;
            let mut sum = 0i64;
            for k in 0..count {
                let mut base = 1 + i32::from(greater1[k]);
                if Some(k) == first_g1 {
                    base += i32::from(greater2);
                }
                let needs = if k < 8 {
                    if Some(k) == first_g1 { 3 } else { 2 }
                } else {
                    1
                };
                let mut level = i64::from(base);
                if base == needs {
                    let rem = self.coeff_abs_level_remaining(rice)?;
                    level += i64::from(rem);
                    if i64::from(base) + i64::from(rem) > 3 * (1i64 << rice) {
                        rice = (rice + 1).min(4);
                    }
                }
                let (xp, yp) = scan4[usize::from(sig[k])];
                let (xc, yc) = ((xs << 2) + usize::from(xp), (ys << 2) + usize::from(yp));
                let negative = k < coded_signs && (signs << k) & 0x8000_0000 != 0;
                sum += level;
                let mut value = if negative { -level } else { level };
                if hidden && k == count - 1 && sum % 2 == 1 {
                    value = -value;
                }
                let value = value.clamp(-32768, 32767);
                self.coeffs[yc * n + xc] = scale.apply(value, xc, yc);
                max_x = max_x.max(xc);
                max_y = max_y.max(yc);
            }
        }
        if self.cabac.overrun() {
            self.clear_coeffs(n, max_x, max_y);
            return Err(truncated("residual"));
        }
        self.reconstruct(x0, y0, log2, c, transform_skip, (max_x, max_y));
        Ok(())
    }

    /// Zero the coefficients of the block just used.
    fn clear_coeffs(&mut self, n: usize, max_x: usize, max_y: usize) {
        for row in self.coeffs.chunks_exact_mut(n).take(max_y + 1) {
            row[..=max_x.min(n - 1)].fill(0);
        }
    }

    /// How the coefficients of a block are scaled (clause 8.6.3).
    fn scaling(&self, log2: u32, c: usize, transform_skip: bool) -> Scale<'a> {
        let n = 1usize << log2;
        let bit_depth = u32::from(self.sps.bit_depth);
        if self.cu.bypass {
            return Scale::Bypass;
        }
        let qp = if c == 0 {
            self.qp_y + self.sps.qp_bd_offset()
        } else {
            let slice = &self.pic.slices[self.slice].header;
            let offset = if c == 1 {
                self.pps.cb_qp_offset + slice.cb_qp_offset
            } else {
                self.pps.cr_qp_offset + slice.cr_qp_offset
            };
            let qpi = (self.qp_y + offset).clamp(-self.sps.qp_bd_offset_c(), 57);
            chroma_qp(qpi) + self.sps.qp_bd_offset_c()
        };
        let shift = bit_depth + log2 - 5;
        let add = 1i64 << (shift - 1);
        let factor = transform::LEVEL_SCALE[(qp % 6) as usize] << (qp / 6);
        let list = self
            .pps
            .scaling_list
            .as_ref()
            .or(self.sps.scaling_list.as_ref());
        match list {
            Some(list) if !(transform_skip && n > 4) => {
                let size_id = log2 as usize - 2;
                let matrix = if size_id == 3 {
                    if self.cu.intra { 0 } else { 3 }
                } else {
                    c + if self.cu.intra { 0 } else { 3 }
                };
                Scale::List {
                    base: &list.lists[size_id][matrix],
                    dc: (size_id > 1).then(|| list.dc[size_id - 2][matrix]),
                    ratio_log2: log2.checked_sub(3),
                    factor,
                    add,
                    shift,
                }
            }
            _ => Scale::Flat {
                factor: factor * 16,
                add,
                shift,
            },
        }
    }

    fn coeff_abs_level_remaining(&mut self, rice: u32) -> Result<u32, CodecError> {
        let mut prefix = 0u32;
        while self.cabac.bypass() {
            prefix += 1;
            if prefix > 32 {
                return Err(malformed("coeff_abs_level_remaining"));
            }
        }
        if prefix <= 3 {
            Ok((prefix << rice) + self.cabac.bypass_bits(rice))
        } else {
            let extra = prefix - 3 + rice;
            if extra > 31 {
                return Err(malformed("coeff_abs_level_remaining"));
            }
            let base = ((1u64 << (prefix - 3)) + 2) << rice;
            let value = base + u64::from(self.cabac.bypass_bits(extra));
            u32::try_from(value.min(u64::from(u32::MAX >> 1)))
                .map_err(|_| malformed("coeff_abs_level_remaining"))
        }
    }

    /// Transform, and add a residual block to the prediction. `max` is the
    /// largest column and row holding a coefficient.
    fn reconstruct(
        &mut self,
        x0: usize,
        y0: usize,
        log2: u32,
        c: usize,
        transform_skip: bool,
        max: (usize, usize),
    ) {
        let (max_x, max_y) = max;
        let n = 1usize << log2;
        let bit_depth = u32::from(self.sps.bit_depth);
        let scale = if c == 0 { 1 } else { 2 };
        let maxv = (1i32 << bit_depth) - 1;
        let coeffs = &self.coeffs[..n * n];
        let res = &mut self.res[..n * n];
        let flat = if self.cu.bypass {
            res.copy_from_slice(coeffs);
            None
        } else if transform_skip {
            transform::skip(coeffs, res, bit_depth);
            None
        } else {
            let dst = c == 0 && log2 == 2 && self.cu.intra;
            transform::inverse(coeffs, res, log2, dst, bit_depth, max_x, max_y)
        };
        self.clear_coeffs(n, max_x, max_y);
        let (plane, stride) = self.out.from(c, x0 / scale, y0 / scale);
        let res = &self.res[..n * n];
        cutlass_simd::wide::wide(
            #[inline(always)]
            || match flat {
                Some(v) => {
                    for row in plane.chunks_mut(stride).take(n) {
                        for s in &mut row[..n] {
                            *s = P::of((s.i() + v).clamp(0, maxv));
                        }
                    }
                }
                None => {
                    for (row, r) in plane.chunks_mut(stride).zip(res.chunks_exact(n)) {
                        for (s, &r) in row[..n].iter_mut().zip(r) {
                            *s = P::of((s.i() + r).clamp(0, maxv));
                        }
                    }
                }
            },
        );
    }

    /// Boundary strengths of the left and top edges of a transform unit
    /// (clause 8.7.2.4). `x0`, `y0` is the q side.
    fn deblock_tu(&mut self, x0: usize, y0: usize, size: usize) {
        self.edges(x0, y0, size, size, true);
    }

    /// The left and top edges of a coding unit with no transform tree.
    fn deblock_cu(&mut self, _tu: bool) -> Result<(), CodecError> {
        let size = 1usize << self.cu.log2;
        let w4 = self.pic.w4;
        for row in (self.cu.y >> 2)..((self.cu.y + size) >> 2) {
            for col in (self.cu.x >> 2)..((self.cu.x + size) >> 2) {
                let at = row * w4 + col;
                self.pic.info.set(at, self.pic.info.get(at) & !FLAG_CBF);
            }
        }
        self.edges(self.cu.x, self.cu.y, size, size, true);
        Ok(())
    }

    /// Edges between prediction units inside an inter coding unit.
    fn deblock_pu_edges(&mut self) {
        let n = 1usize << self.cu.log2;
        let (x0, y0) = (self.cu.x, self.cu.y);
        let q = n / 4;
        let (vertical, horizontal): (Option<usize>, Option<usize>) = match self.cu.part {
            PART_2NXN => (None, Some(n / 2)),
            PART_NX2N => (Some(n / 2), None),
            PART_NXN => (Some(n / 2), Some(n / 2)),
            PART_2NXNU => (None, Some(q)),
            PART_2NXND => (None, Some(n - q)),
            PART_NLX2N => (Some(q), None),
            PART_NRX2N => (Some(n - q), None),
            _ => (None, None),
        };
        if let Some(dx) = vertical {
            self.edge_ver(x0 + dx, y0, n, false);
        }
        if let Some(dy) = horizontal {
            self.edge_hor(x0, y0 + dy, n, false);
        }
    }

    fn edges(&mut self, x0: usize, y0: usize, w: usize, h: usize, tu: bool) {
        self.edge_ver(x0, y0, h, tu);
        self.edge_hor(x0, y0, w, tu);
    }

    /// Whether an edge at q position (x, y) with p at (px, py) is filtered at
    /// all (`filterEdgeFlag` of clause 8.7.2.3).
    fn edge_allowed(&self, x: usize, y: usize, px: isize, py: isize) -> bool {
        if px < 0 || py < 0 {
            return false;
        }
        let slice = &self.pic.slices[self.slice];
        if slice.header.deblocking_disabled {
            return false;
        }
        let (px, py) = (px as usize, py as usize);
        let p_ctb = self.pic.ctb_rs(px, py);
        let q_ctb = self.pic.ctb_rs(x, y);
        if p_ctb == q_ctb {
            return true;
        }
        if self.pic.slice_addr(p_ctb) != slice.slice_addr && !slice.header.loop_filter_across_slices
        {
            return false;
        }
        if self.pic.layout.tile_of[p_ctb] != self.pic.layout.tile_of[q_ctb]
            && !self.pps.loop_filter_across_tiles
        {
            return false;
        }
        true
    }

    fn edge_ver(&mut self, x: usize, y0: usize, len: usize, tu: bool) {
        if !x.is_multiple_of(8) || x >= self.pic.width {
            return;
        }
        let stride = self.pic.width / 8 + 1;
        let mut y = y0;
        while y < y0 + len && y < self.pic.height {
            if self.edge_allowed(x, y, x as isize - 1, y as isize) {
                let bs = self.strength(x - 1, y, x, y, tu);
                let at = (y >> 2) * stride + (x >> 3);
                let bs = if tu {
                    bs
                } else {
                    self.pic.bs_ver.get(at).max(bs)
                };
                self.pic.bs_ver.set(at, bs);
            }
            y += 4;
        }
    }

    fn edge_hor(&mut self, x0: usize, y: usize, len: usize, tu: bool) {
        if !y.is_multiple_of(8) || y >= self.pic.height {
            return;
        }
        let w4 = self.pic.w4;
        let mut x = x0;
        while x < x0 + len && x < self.pic.width {
            if self.edge_allowed(x, y, x as isize, y as isize - 1) {
                let bs = self.strength(x, y - 1, x, y, tu);
                let at = (y >> 3) * w4 + (x >> 2);
                let bs = if tu {
                    bs
                } else {
                    self.pic.bs_hor.get(at).max(bs)
                };
                self.pic.bs_hor.set(at, bs);
            }
            x += 4;
        }
    }

    /// `bS` between p at (px, py) and q at (qx, qy).
    fn strength(&self, px: usize, py: usize, qx: usize, qy: usize, tu: bool) -> u8 {
        let p = self.pic.at4(px, py);
        let q = self.pic.at4(qx, qy);
        let pi = self.pic.info.get(p);
        let qi = self.pic.info.get(q);
        if pi & 3 == MODE_INTRA || qi & 3 == MODE_INTRA {
            return 2;
        }
        if tu && (pi & FLAG_CBF != 0 || qi & FLAG_CBF != 0) {
            return 1;
        }
        let pm = self.pic.mvf.get(p);
        let qm = self.pic.mvf.get(q);
        let p_slice = self.pic.slice_of(px, py);
        let q_slice = &self.pic.slices[self.slice];
        let uid =
            |slice: Option<&crate::picture::SliceCtx>, m: &MvField, list: usize| -> Option<u64> {
                if !m.uses(list) {
                    return None;
                }
                slice
                    .and_then(|s| s.refs[list].get(m.ref_idx[list] as usize))
                    .map(|r| r.uid)
            };
        let p_refs = [uid(p_slice, &pm, 0), uid(p_slice, &pm, 1)];
        let q_refs = [uid(Some(q_slice), &qm, 0), uid(Some(q_slice), &qm, 1)];
        let far = |a: crate::picture::Mv, b: crate::picture::Mv| -> bool {
            (i32::from(a.x) - i32::from(b.x)).abs() >= 4
                || (i32::from(a.y) - i32::from(b.y)).abs() >= 4
        };
        let p_count = p_refs.iter().filter(|r| r.is_some()).count();
        let q_count = q_refs.iter().filter(|r| r.is_some()).count();
        if p_count != q_count {
            return 1;
        }
        if p_count == 1 {
            let (pr, pmv) = if pm.uses(0) {
                (p_refs[0], pm.mv[0])
            } else {
                (p_refs[1], pm.mv[1])
            };
            let (qr, qmv) = if qm.uses(0) {
                (q_refs[0], qm.mv[0])
            } else {
                (q_refs[1], qm.mv[1])
            };
            if pr != qr {
                return 1;
            }
            return u8::from(far(pmv, qmv));
        }
        // Two vectors each.
        let same_set = (p_refs[0] == q_refs[0] && p_refs[1] == q_refs[1])
            || (p_refs[0] == q_refs[1] && p_refs[1] == q_refs[0]);
        if !same_set {
            return 1;
        }
        if p_refs[0] != p_refs[1] {
            if p_refs[0] == q_refs[0] {
                u8::from(far(pm.mv[0], qm.mv[0]) || far(pm.mv[1], qm.mv[1]))
            } else {
                u8::from(far(pm.mv[0], qm.mv[1]) || far(pm.mv[1], qm.mv[0]))
            }
        } else {
            u8::from(
                (far(pm.mv[0], qm.mv[0]) || far(pm.mv[1], qm.mv[1]))
                    && (far(pm.mv[0], qm.mv[1]) || far(pm.mv[1], qm.mv[0])),
            )
        }
    }
}

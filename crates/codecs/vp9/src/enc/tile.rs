//! The encoder of one tile: partition and mode decisions, motion search,
//! the residual of each transform block, the syntax written as the
//! decoder reads it (libvpx `vp9_bitstream.c` order), and the
//! reconstruction, through the decoder's own reconstructor so that the
//! encoder's picture is the decoder's.
//!
//! Every context the syntax uses is derived as `dec::parse` derives it,
//! and every symbol is counted as it counts it, so that the backward
//! probability adaptation after the frame gives both the same
//! probabilities.

use crate::bits::BoolWriter;
use crate::dec::ctx::{self, Nb};
use crate::dec::frame::{Mv, MvRef};
use crate::dec::parse::{LfMi, State};
use crate::dec::recon::Recon;
use crate::dec::syntax::BlockRec;
use crate::dsp::{ipred, sad};
use crate::geom::{
    self, BLOCK_8X8, BLOCK_64X64, INTRA_FRAME, LAST_FRAME, NEAR, NEAREST, NEW, PARTITION_HORZ,
    PARTITION_NONE, PARTITION_SPLIT, PARTITION_VERT, TX_32X32, ZERO,
};
use crate::header::Header;
use crate::lpf;
use crate::prob::{self, FrameContext};
use crate::tables::{BAND_4X4, BAND_8X8_PLUS, CAT6_PROB, KF_UV_MODE_PROB, KF_Y_MODE_PROB, PARETO8};

use super::tx::{forward, fwht};

const MODE_COUNTER: [i32; 14] = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 3, 1];
const COUNTER_CTX: [u8; 19] = [2, 3, 4, 1, 3, 9, 0, 9, 9, 5, 5, 9, 5, 9, 9, 9, 9, 9, 6];
const MV_REF: [[[i8; 2]; 8]; 13] = {
    const N8: [[i8; 2]; 8] = [[-1, 0], [0, -1], [-1, -1], [-2, 0], [0, -2], [-2, -1], [-1, -2], [-2, -2]];
    [
        N8,
        N8,
        N8,
        N8,
        [[0, -1], [-1, 0], [1, -1], [-1, -1], [0, -2], [-2, 0], [-2, -1], [-1, -2]],
        [[-1, 0], [0, -1], [-1, 1], [-1, -1], [-2, 0], [0, -2], [-1, -2], [-2, -1]],
        [[-1, 0], [0, -1], [-1, 1], [1, -1], [-1, -1], [-3, 0], [0, -3], [-3, -3]],
        [[0, -1], [-1, 0], [2, -1], [-1, -1], [-1, 1], [0, -3], [-3, 0], [-3, -3]],
        [[-1, 0], [0, -1], [-1, 2], [-1, -1], [1, -1], [-3, 0], [0, -3], [-3, -3]],
        [[-1, 1], [1, -1], [-1, 2], [2, -1], [-1, -1], [-3, 0], [0, -3], [-3, -3]],
        [[0, -1], [-1, 0], [4, -1], [-1, 2], [-1, -1], [0, -3], [-3, 0], [2, -1]],
        [[-1, 0], [0, -1], [-1, 4], [2, -1], [-1, -1], [-3, 0], [0, -3], [-1, 2]],
        [[-1, 3], [3, -1], [-1, 4], [4, -1], [-1, -1], [-1, 0], [0, -1], [-1, 6]],
    ]
};
const MI_WH: [(usize, usize); 13] = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 2), (2, 1), (2, 2), (2, 4), (4, 2), (4, 4), (4, 8), (8, 4), (8, 8)];

/// Square block sizes by log2 of their width in 8×8 units.
const SQUARE: [u8; 4] = [BLOCK_8X8, geom::BLOCK_16X16, geom::BLOCK_32X32, BLOCK_64X64];

/// The cost of a bit, in 1/256 bits: `COST[p]` for a 0 of probability
/// `p / 256`, `COST[256 - p]` for a 1.
static COST: std::sync::LazyLock<[u16; 257]> = std::sync::LazyLock::new(|| {
    let mut t = [0u16; 257];
    for (p, v) in t.iter_mut().enumerate().skip(1) {
        *v = (-(p as f64 / 256.0).log2() * 256.0).round() as u16;
    }
    t[0] = t[1];
    t
});

#[inline]
pub fn bit_cost(bit: bool, p: u8) -> u32 {
    let c = &*COST;
    u32::from(if bit { c[256 - usize::from(p)] } else { c[usize::from(p)] })
}

/// A source picture: three planes padded to whole superblocks.
pub struct Source<'a> {
    pub planes: [&'a [u8]; 3],
    pub strides: [usize; 3],
}

/// What a frame's tiles read.
pub struct Shared<'a> {
    pub h: &'a Header,
    pub fc: &'a FrameContext,
    pub prev_mvs: Option<&'a [MvRef]>,
    pub levels: &'a lpf::Levels,
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub intra_frame: bool,
    pub part_probs: [[u8; 3]; 16],
    pub filter_on: bool,
    /// The SAD a bit of rate is worth.
    pub lambda: u32,
    pub src: Source<'a>,
    /// The quantizer steps of luma and chroma, DC and AC.
    pub dq: [[i32; 2]; 2],
    /// Skip motion search where it cannot gain (realtime speed).
    pub speed: u8,
}

/// The block being encoded.
#[derive(Clone, Copy)]
struct Blk {
    row: usize,
    col: usize,
    bsize: u8,
    bw: usize,
    bh: usize,
    x_mis: usize,
    y_mis: usize,
    skip: bool,
    tx: u8,
    mode: u8,
    uv: u8,
    refs: [u8; 2],
    mv: Mv,
}

/// The encoder of one tile.
pub struct Tile<'s, 't, 'r, 'v> {
    s: &'s Shared<'s>,
    f: &'t mut State,
    w: &'t mut BoolWriter,
    rec: &'t mut Recon<'r, 'v, u8>,
    col0: usize,
    col1: usize,
    left: [Nb; 8],
    left_part: [u8; 8],
    left_nz: [[u8; 16]; 3],
    cache: [u8; 1024],
    pred: Vec<u8>,
    res: Vec<i32>,
    fcoef: Vec<f32>,
    /// Best vectors found per 8×8 position this superblock, for the
    /// next searches' starting points.
    seeds: Vec<Mv>,
    /// The whole-sample vector and SAD the partition search found for each
    /// square block of the superblock (see [`Tile::slot`]).
    found: [(Mv, u64); 85],
}

impl<'s, 't, 'r, 'v> Tile<'s, 't, 'r, 'v> {
    /// `seeds`: the vectors found so far (one per 8×8 block of the frame),
    /// back from [`Tile::finish`] of the row above.
    pub fn new(
        s: &'s Shared<'s>,
        f: &'t mut State,
        w: &'t mut BoolWriter,
        rec: &'t mut Recon<'r, 'v, u8>,
        (col0, col1): (usize, usize),
        mut seeds: Vec<Mv>,
    ) -> Self {
        seeds.resize(s.mi_cols * s.mi_rows, Mv::ZERO);
        Self {
            s,
            f,
            w,
            rec,
            col0,
            col1,
            left: [Nb::default(); 8],
            left_part: [0; 8],
            left_nz: [[0; 16]; 3],
            cache: [0; 1024],
            pred: vec![0; 64 * 64],
            res: vec![0; 64 * 64],
            fcoef: vec![0.0; 32 * 32],
            seeds,
            found: [(Mv::ZERO, u64::MAX); 85],
        }
    }

    /// The vectors found, for the next row's tile.
    pub fn finish(self) -> Vec<Mv> {
        self.seeds
    }

    /// Encodes the superblocks of one row of this tile.
    pub fn sb_row(&mut self, mi_row: usize) {
        self.left = [Nb::default(); 8];
        self.left_part = [0; 8];
        self.left_nz = [[0; 16]; 3];
        let mut mi_col = self.col0;
        while mi_col < self.col1 {
            let split = self.choose(mi_row, mi_col, 3);
            self.partition(mi_row, mi_col, BLOCK_64X64, 4, &split);
            mi_col += 8;
        }
    }

    // Partition decision --------------------------------------------------

    /// Where the search result of the square block `bsize` at `(row, col)`
    /// is kept, in its superblock.
    fn slot(row: usize, col: usize, bsize: u8) -> usize {
        let level = SQUARE.iter().position(|&b| b == bsize).unwrap_or(0);
        let n = 8 >> level;
        let (r, c) = ((row & 7) >> level, (col & 7) >> level);
        [21, 5, 1, 0][level] + r * n + c
    }

    /// The flag index of the square node of `level` (3: 64×64 down to
    /// 1: 16×16) at `(row, col)` in its superblock.
    fn node(row: usize, col: usize, level: usize) -> usize {
        let (r, c) = ((row & 7) >> level, (col & 7) >> level);
        match level {
            3 => 0,
            2 => 1 + r * 2 + c,
            _ => 5 + r * 4 + c,
        }
    }

    /// The split flags of the superblock at `(row, col)`. Each block's
    /// cost is its best prediction's SAD plus the rate of its vector, in
    /// SAD units; a block splits when its four quarters cost less.
    fn choose(&mut self, row: usize, col: usize, level: usize) -> [bool; 21] {
        let mut split = [false; 21];
        self.choose_into(row, col, level, &mut split);
        split
    }

    fn choose_into(&mut self, row: usize, col: usize, level: usize, split: &mut [bool; 21]) -> u64 {
        let s = self.s;
        if row >= s.mi_rows || col >= s.mi_cols {
            return 0;
        }
        if level == 0 {
            return self.estimate(row, col, SQUARE[0]);
        }
        let n = 1usize << level;
        let hbs = n / 2;
        let whole = row + n <= s.mi_rows && col + n <= s.mi_cols;
        let own = if whole { self.estimate(row, col, SQUARE[level]) } else { u64::MAX };
        // Splitting costs the partition symbols and three more blocks'
        // mode info.
        let mut quarters = u64::from(s.lambda) * 12;
        for (dr, dc) in [(0, 0), (0, hbs), (hbs, 0), (hbs, hbs)] {
            quarters = quarters.saturating_add(self.choose_into(row + dr, col + dc, level - 1, split));
            if quarters >= own {
                break;
            }
        }
        if quarters < own {
            split[Self::node(row, col, level)] = true;
            quarters
        } else {
            own
        }
    }

    /// The cost of the best prediction of a square block, from the
    /// source, in SAD units: distortion plus `lambda` a bit.
    fn estimate(&mut self, row: usize, col: usize, bsize: u8) -> u64 {
        let s = self.s;
        let (bw, _) = MI_WH[usize::from(bsize)];
        let n = bw * 8;
        let (x, y) = (col * 8, row * 8);
        let lam = u64::from(s.lambda);
        if s.intra_frame {
            // DC prediction from the source's own edges.
            let (src, ss) = (s.src.planes[0], s.src.strides[0]);
            let mut sum = 0u32;
            let mut cnt = 0u32;
            if y > 0 {
                for i in 0..n {
                    sum += u32::from(src[(y - 1) * ss + x + i]);
                }
                cnt += n as u32;
            }
            if x > 0 {
                for i in 0..n {
                    sum += u32::from(src[(y + i) * ss + x - 1]);
                }
                cnt += n as u32;
            }
            let dc = if cnt > 0 { ((sum + cnt / 2) / cnt) as i32 } else { 128 };
            let mut cost = 0u64;
            for r in 0..n {
                for c in 0..n {
                    cost += u64::from((i32::from(src[(y + r) * ss + x + c]) - dc).unsigned_abs());
                }
            }
            return cost + lam * 4;
        }
        let (mv, sad) = self.search(row, col, bsize);
        self.found[Self::slot(row, col, bsize)] = (mv, sad);
        for r in 0..bw.min(s.mi_rows - row) {
            for c in 0..bw.min(s.mi_cols - col) {
                self.seeds[(row + r) * s.mi_cols + col + c] = mv;
            }
        }
        sad + lam * u64::from(mv_bits(mv)) / 256 + lam * 3
    }

    // Motion search -------------------------------------------------------

    /// The sum of absolute differences of the luma block at `(x, y)`, `n`
    /// wide and high, against the reference at whole-sample offset
    /// `(dr, dc)`, or `None` when the window leaves the reference.
    fn sad_int(&self, x: usize, y: usize, w: usize, h: usize, dr: i32, dc: i32) -> Option<u64> {
        let s = self.s;
        let rp = self.rec_ref()?;
        let (data, stride) = rp.0;
        let (rw, rh) = rp.1;
        let (bx, by) = (x as i32 + dc, y as i32 + dr);
        if bx < 0 || by < 0 || bx + w as i32 > rw || by + h as i32 > rh {
            return None;
        }
        let (src, ss) = (s.src.planes[0], s.src.strides[0]);
        let a = &src[y * ss + x..];
        let b = &data[by as usize * stride + bx as usize..];
        Some(u64::from(sad::sad(a, ss, b, stride, w, h)))
    }

    fn rec_ref(&self) -> Option<((&[u8], usize), (i32, i32))> {
        let c = self.rec.ctx();
        let rp = c.refs[0].as_ref()?;
        Some((rp.buf.plane(0), (rp.dims.w[0] as i32, rp.dims.h[0] as i32)))
    }

    /// The SAD of a block against the reference moved by a sub-sample
    /// vector (eighths), with bilinear interpolation (the search's proxy
    /// for the 8-tap filters), or `None` when the window leaves the
    /// reference.
    fn sad_sub(&self, x: usize, y: usize, w: usize, h: usize, mv: Mv) -> Option<u64> {
        let s = self.s;
        let ((data, stride), (rw, rh)) = self.rec_ref()?;
        let (r, c) = (i32::from(mv.row), i32::from(mv.col));
        let (bx, by) = (x as i32 + (c >> 3), y as i32 + (r >> 3));
        if bx < 0 || by < 0 || bx + w as i32 + 1 > rw || by + h as i32 + 1 > rh {
            return None;
        }
        let (src, ss) = (s.src.planes[0], s.src.strides[0]);
        let a = &src[y * ss + x..];
        let b = &data[by as usize * stride + bx as usize..];
        Some(u64::from(sad::sad_bilinear(a, ss, b, stride, w, h, ((r & 7) * 2) as u32, ((c & 7) * 2) as u32)))
    }

    /// Searches LAST for the block's vector: the best of the neighbours'
    /// vectors and zero, a diamond search in whole samples, then halves
    /// and quarters. Returns the vector (eighths) and its SAD.
    fn search(&mut self, row: usize, col: usize, bsize: u8) -> (Mv, u64) {
        let s = self.s;
        let (bw, bh) = MI_WH[usize::from(bsize)];
        let (w, h) = (bw * 8, bh * 8);
        let (x, y) = (col * 8, row * 8);
        let w = w.min(s.mi_cols * 8 - x).max(8);
        let h = h.min(s.mi_rows * 8 - y).max(8);
        let w = w.min((s.mi_cols - col) * 8);
        let h = h.min((s.mi_rows - row) * 8);
        // Starting points.
        let mut cands = [Mv::ZERO; 6];
        let mut nc = 1;
        let seed = |r: isize, c: isize| -> Option<Mv> {
            if r < 0 || c < 0 || r as usize >= s.mi_rows || c as usize >= s.mi_cols {
                None
            } else {
                Some(self.seeds[r as usize * s.mi_cols + c as usize])
            }
        };
        for (dr, dc) in [(-1isize, 0isize), (0, -1), (-1, bw as isize), (0, 0)] {
            if let Some(m) = seed(row as isize + dr, col as isize + dc) {
                cands[nc] = m;
                nc += 1;
            }
        }
        if let Some(p) = s.prev_mvs
            && let Some(m) = p.get(row * s.mi_cols + col)
            && m.refs[0] == LAST_FRAME
        {
            cands[nc] = m.mv[0];
            nc += 1;
        }
        let mut best = (0i32, 0i32);
        let mut best_sad = u64::MAX;
        for m in &cands[..nc] {
            let (r, c) = ((i32::from(m.row) + 4) >> 3, (i32::from(m.col) + 4) >> 3);
            if let Some(v) = self.sad_int(x, y, w, h, r, c) {
                let v = v + mvcost_px(r, c, best);
                if v < best_sad {
                    best_sad = v;
                    best = (r, c);
                }
            }
        }
        if best_sad == u64::MAX {
            return (Mv::ZERO, u64::MAX / 4);
        }
        // Diamond search in whole samples.
        let range = 64;
        let mut step = 8;
        while step >= 1 {
            let mut moved = true;
            let mut iters = 0;
            while moved && iters < 16 {
                moved = false;
                iters += 1;
                let center = best;
                for (dr, dc) in [(-step, 0), (step, 0), (0, -step), (0, step)] {
                    let (r, c) = (center.0 + dr, center.1 + dc);
                    if r.abs() > range || c.abs() > range {
                        continue;
                    }
                    if let Some(v) = self.sad_int(x, y, w, h, r, c)
                        && v < best_sad
                    {
                        best_sad = v;
                        best = (r, c);
                        moved = true;
                    }
                }
            }
            step /= 2;
        }
        let mv = Mv {
            row: (best.0 * 8) as i16,
            col: (best.1 * 8) as i16,
        };
        (mv, best_sad)
    }

    /// Refines a whole-sample vector of a block to halves and quarters (and
    /// eighths where they may be coded).
    fn refine(&mut self, row: usize, col: usize, bsize: u8, mut mv: Mv, mut best_sad: u64) -> (Mv, u64) {
        let s = self.s;
        let (bw, bh) = MI_WH[usize::from(bsize)];
        let (x, y) = (col * 8, row * 8);
        let w = (bw * 8).min((s.mi_cols - col) * 8);
        let h = (bh * 8).min((s.mi_rows - row) * 8);
        if s.speed < 9 {
            let hp = s.h.allow_hp && mv.row.unsigned_abs() < 64 && mv.col.unsigned_abs() < 64;
            let steps: &[i16] = if hp && s.speed < 4 { &[4, 2, 1] } else { &[4, 2] };
            for &step in steps {
                let center = mv;
                for (dr, dc) in [(-step, 0), (step, 0), (0, -step), (0, step), (-step, -step), (-step, step), (step, -step), (step, step)] {
                    let m = Mv {
                        row: center.row + dr,
                        col: center.col + dc,
                    };
                    let Some(v) = self.sad_sub(x, y, w, h, m) else { continue };
                    if v < best_sad {
                        best_sad = v;
                        mv = m;
                    }
                }
            }
        }
        (mv, best_sad)
    }

    // Partitions -----------------------------------------------------------

    fn partition(&mut self, row: usize, col: usize, bsize: u8, n4: u32, split: &[bool; 21]) {
        let s = self.s;
        if row >= s.mi_rows || col >= s.mi_cols {
            return;
        }
        let n8 = n4 - 1;
        let hbs = (1usize << n8) >> 1;
        let has_rows = row + hbs < s.mi_rows;
        let has_cols = col + hbs < s.mi_cols;
        // A block that does not fit the frame splits.
        let p = if hbs == 0 {
            PARTITION_NONE
        } else if !(has_rows && has_cols) || split[Self::node(row, col, n8 as usize)] {
            PARTITION_SPLIT
        } else {
            PARTITION_NONE
        };
        let ctx = {
            let above = usize::from(self.f.above_part[col - self.col0] >> n8) & 1;
            let left = usize::from(self.left_part[row & 7] >> n8) & 1;
            left * 2 + above + n8 as usize * 4
        };
        let probs = s.part_probs[ctx];
        if has_rows && has_cols {
            prob::write_tree(self.w, &prob::PARTITION_TREE, &probs, p);
        } else if !has_rows && has_cols {
            self.w.write(u32::from(p == PARTITION_SPLIT), u32::from(probs[1]));
        } else if has_rows && !has_cols {
            self.w.write(u32::from(p == PARTITION_SPLIT), u32::from(probs[2]));
        }
        if self.collect() {
            self.f.counts.partition[ctx][usize::from(p)] += 1;
        }
        let sub = geom::subsize(p, bsize);
        if p == PARTITION_NONE {
            self.block(row, col, sub);
        } else {
            for (dr, dc) in [(0, 0), (0, hbs), (hbs, 0), (hbs, hbs)] {
                self.partition(row + dr, col + dc, sub, n8, split);
            }
        }
        if bsize == BLOCK_8X8 || p != PARTITION_SPLIT {
            let n = 1usize << geom::mi_width_log2(bsize);
            let (above, left) = geom::partition_context_byte(sub);
            let a = col - self.col0;
            let end = (a + n).min(self.f.above_part.len());
            self.f.above_part[a..end].fill(above);
            let l = row & 7;
            self.left_part[l..(l + n).min(8)].fill(left);
        }
        let _ = (PARTITION_HORZ, PARTITION_VERT);
    }

    fn above_nb(&self, b: &Blk) -> Option<&Nb> {
        (b.row > 0).then(|| &self.f.above[b.col - self.col0])
    }

    fn left_nb(&self, b: &Blk) -> Option<&Nb> {
        (b.col > self.col0).then(|| &self.left[b.row & 7])
    }

    fn collect(&self) -> bool {
        !self.s.h.error_resilient && !self.s.h.frame_parallel
    }

    // Blocks ----------------------------------------------------------------

    fn block(&mut self, row: usize, col: usize, bsize: u8) {
        let s = self.s;
        let (bw, bh) = MI_WH[usize::from(bsize)];
        let mut b = Blk {
            row,
            col,
            bsize,
            bw,
            bh,
            x_mis: bw.min(s.mi_cols - col),
            y_mis: bh.min(s.mi_rows - row),
            skip: false,
            tx: geom::max_tx(bsize).min(geom::tx_mode_biggest(s.h.tx_mode)),
            mode: geom::DC,
            uv: geom::DC,
            refs: [INTRA_FRAME, INTRA_FRAME],
            mv: Mv::ZERO,
        };
        if s.intra_frame {
            self.pick_intra(&mut b);
        } else {
            self.pick_inter(&mut b);
        }
        // The residual decides skip: code it, and see.
        let rec = BlockRec {
            row: row as u16,
            col: col as u16,
            bsize,
            coded: true,
            tx: b.tx,
            uv_tx: self.uv_tx(&b),
            mode: b.mode,
            bmode: [b.mode; 4],
            uv: b.uv,
            interp: geom::EIGHTTAP,
            refs: b.refs,
            bmv: [[b.mv, Mv::ZERO]; 4],
        };
        if b.refs[0] != INTRA_FRAME {
            self.rec.predict_inter(&rec);
        }
        // Coefficients of every tx block, kept until the mode info is
        // written (intra predicts tx block by tx block from the ones
        // before, so it is reconstructed as it goes).
        let mut toks: Vec<(usize, u8, usize, usize, Vec<i32>, usize)> = Vec::new();
        let mut eobs = 0;
        self.residual(&b, &rec, &mut toks, &mut eobs);
        b.skip = eobs == 0;
        // Mode info.
        if s.intra_frame {
            self.write_intra_frame_info(&b);
        } else {
            self.write_inter_frame_info(&b);
        }
        {
            let r = MvRef {
                mv: [b.mv, Mv::ZERO],
                refs: b.refs,
            };
            let tw = self.f.width;
            for y in 0..b.y_mis {
                let at = (row + y) * tw + col - self.col0;
                self.f.mvs[at..at + b.x_mis].fill(r);
            }
        }
        if b.skip {
            self.reset_nz(&b);
        } else {
            for (plane, tx, r4, c4, levels, eob) in toks {
                self.write_coefs(&b, plane, tx, r4, c4, &levels, eob);
            }
        }
        let nb = Nb {
            bsize,
            mode: b.mode,
            bmode: [b.mode; 4],
            skip: b.skip,
            tx: b.tx,
            seg_pred: false,
            interp: if b.refs[0] == INTRA_FRAME { 3 } else { geom::EIGHTTAP },
            refs: b.refs,
            bmv: [[b.mv, Mv::ZERO]; 4],
        };
        let a = col - self.col0;
        let end = (a + bw).min(self.f.above.len());
        self.f.above[a..end].fill(nb);
        let l = row & 7;
        self.left[l..(l + bh).min(8)].fill(nb);
        if s.filter_on {
            let index = (row >> 3) * self.f.width.div_ceil(8) + ((col - self.col0) >> 3);
            lpf::build(
                &mut self.f.masks[index],
                s.levels,
                &lpf::Block {
                    bsize,
                    tx: b.tx,
                    skip: b.skip,
                    inter: b.refs[0] != INTRA_FRAME,
                    seg: 0,
                    mode: b.mode,
                    reference: b.refs[0],
                    bw,
                    bh,
                },
                row,
                col,
            );
            if !self.f.grid.is_empty() {
                let g = LfMi {
                    bsize,
                    tx: b.tx,
                    skip: b.skip,
                    ref0: b.refs[0],
                    seg: 0,
                    mode: b.mode,
                };
                for y in 0..b.y_mis {
                    let at = (row + y) * self.f.width + col - self.col0;
                    self.f.grid[at..at + b.x_mis].fill(g);
                }
            }
        }
    }

    fn uv_tx(&self, b: &Blk) -> u8 {
        geom::uv_tx_420(b.bsize, b.tx)
    }

    // Mode decisions ---------------------------------------------------------

    /// Intra modes of a key frame: the luma mode with the lowest SATD
    /// over the block's first tx block (from the reconstruction), and DC
    /// for chroma.
    fn pick_intra(&mut self, b: &mut Blk) {
        let tx = b.tx;
        let n = 4usize << tx;
        let (x, y) = (b.col * 8, b.row * 8);
        let have_above = b.row > 0;
        let have_left = b.col > self.col0;
        let e = self.rec.intra_edges(0, tx, x, y, have_left, have_above, false);
        let s = self.s;
        let (src, ss) = (s.src.planes[0], s.src.strides[0]);
        let mut best = (u64::MAX, geom::DC);
        let modes: &[u8] = if s.speed >= 8 {
            &[geom::DC, geom::V, geom::H, geom::TM]
        } else {
            &[geom::DC, geom::V, geom::H, geom::TM, geom::D45, geom::D135, geom::D117, geom::D153, geom::D207, geom::D63]
        };
        for &m in modes {
            ipred::predict(m, n, &mut self.pred, n, &e, 8);
            let mut cost = 0u64;
            for r in 0..n {
                cost += sad_row(&src[(y + r) * ss + x..(y + r) * ss + x + n], &self.pred[r * n..r * n + n]);
            }
            let cost = cost + u64::from(s.lambda) / 64 * u64::from(m != geom::DC);
            if cost < best.0 {
                best = (cost, m);
            }
        }
        b.mode = best.1;
        b.uv = geom::DC;
        b.refs = [INTRA_FRAME; 2];
    }

    /// The vector of an inter block: the searched one, or a candidate of
    /// the reference list when it is as good.
    fn pick_inter(&mut self, b: &mut Blk) {
        let (mv, sad) = self.found[Self::slot(b.row, b.col, b.bsize)];
        let (mv, _) = self.refine(b.row, b.col, b.bsize, mv, sad);
        b.refs = [LAST_FRAME, INTRA_FRAME];
        b.mv = mv;
        // NEAREST / NEAR / ZERO when one of them is the vector.
        let nearest = self.find_mv(b, NEAREST);
        let near = self.find_mv(b, NEAR);
        b.mode = if mv == Mv::ZERO {
            ZERO
        } else if mv == nearest {
            NEAREST
        } else if mv == near {
            NEAR
        } else {
            NEW
        };
        b.uv = geom::DC;
    }

    // Residual -----------------------------------------------------------

    /// Every tx block's prediction (intra), residual, quantized levels and
    /// reconstruction, in the decoder's order.
    fn residual(&mut self, b: &Blk, rec: &BlockRec, toks: &mut Vec<(usize, u8, usize, usize, Vec<i32>, usize)>, eobs: &mut usize) {
        let s = self.s;
        let intra = b.refs[0] == INTRA_FRAME;
        let lossless = s.h.lossless;
        for plane in 0..3 {
            let tx = if plane == 0 { b.tx } else { rec.uv_tx };
            let (ssx, ssy) = if plane == 0 { (0, 0) } else { (1, 1) };
            let (n4w, _, vw, vh) = self.visible(b, plane);
            let step = 1usize << tx;
            let x0 = (b.col * 8) >> ssx;
            let y0 = (b.row * 8) >> ssy;
            let mut r = 0;
            while r < vh {
                let mut c = 0;
                while c < vw {
                    let (x, y) = (x0 + c * 4, y0 + r * 4);
                    let mut kind = 0u8;
                    if intra {
                        let mode = if plane == 0 { b.mode } else { b.uv };
                        let have_above = r > 0 || b.row > 0;
                        let have_left = c > 0 || b.col > self.col0;
                        let have_right = c + step < n4w;
                        self.rec.predict_intra(plane, mode, tx, x, y, have_left, have_above, have_right);
                        if plane == 0 && !lossless && tx != TX_32X32 {
                            kind = geom::intra_tx_type(mode);
                        }
                    }
                    let (levels, eob) = self.quantize(plane, tx, kind, x, y, intra);
                    *eobs += eob;
                    toks.push((plane, tx, r, c, levels, eob));
                    c += step;
                }
                r += step;
            }
        }
    }

    fn visible(&self, b: &Blk, plane: usize) -> (usize, usize, usize, usize) {
        let ss = usize::from(plane > 0);
        let n4w = (b.bw * 2) >> ss;
        let n4h = (b.bh * 2) >> ss;
        let right = (self.s.mi_cols as i32 - b.bw as i32 - b.col as i32) * 64;
        let bottom = (self.s.mi_rows as i32 - b.bh as i32 - b.row as i32) * 64;
        let vw = if right >= 0 { n4w } else { (n4w as i32 + (right >> (5 + ss))).max(0) as usize };
        let vh = if bottom >= 0 { n4h } else { (n4h as i32 + (bottom >> (5 + ss))).max(0) as usize };
        (n4w, n4h, vw, vh)
    }

    /// The levels of one tx block (transposed positions, the scan's order
    /// of the decoder), and its end of block; the block is reconstructed.
    fn quantize(&mut self, plane: usize, tx: u8, kind: u8, x: usize, y: usize, intra: bool) -> (Vec<i32>, usize) {
        let s = self.s;
        let n = 4usize << tx;
        let log2n = 2 + usize::from(tx);
        let (src, ss) = (s.src.planes[plane], s.src.strides[plane]);
        {
            let v = &mut self.rec.views()[plane];
            let stride = v.stride;
            let pred = v.at(y, x);
            for r in 0..n {
                for c in 0..n {
                    self.res[r * n + c] = i32::from(src[(y + r) * ss + x + c]) - i32::from(pred[r * stride + c]);
                }
            }
        }
        let mut levels = vec![0i32; n * n];
        let dqi = usize::from(plane > 0);
        let (dc_q, ac_q) = (s.dq[dqi][0], s.dq[dqi][1]);
        let shift = u32::from(tx == 3);
        if s.h.lossless {
            let mut c = [0i32; 16];
            fwht(&self.res, n, &mut c);
            for (i, v) in c.iter().enumerate() {
                let q = if i == 0 { dc_q } else { ac_q };
                levels[i] = *v / q.max(1);
            }
        } else {
            forward(&self.res, n, usize::from(tx), kind & 1 == 1, kind & 2 == 2, &mut self.fcoef);
            // A dead zone: less rounding up between, more at DC.
            let round_ac = if intra { 0.42 } else { 0.30 };
            for i in 0..n * n {
                let q = if i == 0 { dc_q } else { ac_q };
                let step = q as f32 / (1 << shift) as f32;
                let v = self.fcoef[i];
                let a = v.abs() / step;
                let rnd = if i == 0 { 0.5 } else { round_ac };
                let l = (a + rnd) as i32;
                levels[i] = if v < 0.0 { -l } else { l };
            }
        }
        // End of block in the decoder's scan; drop what lies past it.
        let (scan, _) = scan_pair(tx, kind);
        let mut eob = 0;
        for (k, &rc) in scan.iter().enumerate() {
            let rc = usize::from(rc);
            let t = ((rc & (n - 1)) << log2n) | (rc >> log2n);
            if levels[t] != 0 {
                eob = k + 1;
            }
        }
        // Dequantize as the decoder does, and reconstruct.
        if eob > 0 {
            let mut max_r = 0;
            let mut max_c = 0;
            {
                let coef = self.rec.coef();
                for (k, &rc) in scan.iter().enumerate().take(eob) {
                    let rc = usize::from(rc);
                    let t = ((rc & (n - 1)) << log2n) | (rc >> log2n);
                    let l = levels[t];
                    if l != 0 {
                        let q = if k == 0 { dc_q } else { ac_q };
                        let v = ((i64::from(l.unsigned_abs()) * i64::from(q)) >> shift) as i32;
                        coef[t] = if l < 0 { -v } else { v };
                        max_r = max_r.max(rc >> log2n);
                        max_c = max_c.max(rc & (n - 1));
                    }
                }
            }
            self.rec.add_residual(plane, x, y, tx, kind, eob, (max_r + 1, max_c + 1));
        }
        (levels, eob)
    }

    // Syntax ---------------------------------------------------------------

    fn write_skip(&mut self, b: &Blk) {
        let ctx = usize::from(self.above_nb(b).is_some_and(|n| n.skip)) + usize::from(self.left_nb(b).is_some_and(|n| n.skip));
        self.w.write(u32::from(b.skip), u32::from(self.s.fc.skip[ctx]));
        if self.collect() {
            self.f.counts.skip[ctx][usize::from(b.skip)] += 1;
        }
    }

    fn write_intra_frame_info(&mut self, b: &Blk) {
        self.write_skip(b);
        // tx_mode is not TX_MODE_SELECT: no tx size.
        let above = self.above_nb(b).copied();
        let left = self.left_nb(b).copied();
        let a = match above {
            Some(a) if a.refs[0] == INTRA_FRAME => a.bmode[2],
            _ => geom::DC,
        };
        let l = match left {
            Some(l) if l.refs[0] == INTRA_FRAME => l.bmode[1],
            _ => geom::DC,
        };
        prob::write_tree(self.w, &prob::INTRA_TREE, &KF_Y_MODE_PROB[usize::from(a)][usize::from(l)], b.mode);
        prob::write_tree(self.w, &prob::INTRA_TREE, &KF_UV_MODE_PROB[usize::from(b.mode)], b.uv);
    }

    fn write_inter_frame_info(&mut self, b: &Blk) {
        let s = self.s;
        self.write_skip(b);
        let inter = b.refs[0] != INTRA_FRAME;
        let ctx = ctx::intra_inter_ctx(self.above_nb(b), self.left_nb(b));
        self.w.write(u32::from(inter), u32::from(s.fc.intra_inter[ctx]));
        if self.collect() {
            self.f.counts.intra_inter[ctx][usize::from(inter)] += 1;
        }
        if !inter {
            let group = geom::size_group(b.bsize);
            prob::write_tree(self.w, &prob::INTRA_TREE, &s.fc.y_mode[group], b.mode);
            if self.collect() {
                self.f.counts.y_mode[group][usize::from(b.mode)] += 1;
            }
            prob::write_tree(self.w, &prob::INTRA_TREE, &s.fc.uv_mode[usize::from(b.mode)], b.uv);
            if self.collect() {
                self.f.counts.uv_mode[usize::from(b.mode)][usize::from(b.uv)] += 1;
            }
            return;
        }
        // Single reference, LAST: bit 0 of the first reference symbol.
        let c0 = ctx::single_p1(self.above_nb(b), self.left_nb(b));
        self.w.write(0, u32::from(s.fc.single_ref[c0][0]));
        if self.collect() {
            self.f.counts.single_ref[c0][0][0] += 1;
        }
        let search = &MV_REF[usize::from(b.bsize)];
        let mut counter = 0i32;
        for pos in &search[..2] {
            if let Some(n) = self.near(b, *pos) {
                counter += MODE_COUNTER[usize::from(n.mode)];
            }
        }
        let mctx = usize::from(COUNTER_CTX[counter as usize]).min(6);
        let k = (b.mode - NEAREST) as usize;
        let p = s.fc.inter_mode[mctx];
        match k {
            2 => self.w.write(0, u32::from(p[0])),
            0 => {
                self.w.write(1, u32::from(p[0]));
                self.w.write(0, u32::from(p[1]));
            }
            1 => {
                self.w.write(1, u32::from(p[0]));
                self.w.write(1, u32::from(p[1]));
                self.w.write(0, u32::from(p[2]));
            }
            _ => {
                self.w.write(1, u32::from(p[0]));
                self.w.write(1, u32::from(p[1]));
                self.w.write(1, u32::from(p[2]));
            }
        }
        if self.collect() {
            self.f.counts.inter_mode[mctx][k] += 1;
        }
        // The frame's filter is not switchable.
        if b.mode == NEW {
            let best = self.find_mv(b, NEW);
            self.write_mv(b.mv, best);
        }
    }

    fn write_mv(&mut self, mv: Mv, best: Mv) {
        let s = self.s;
        let d = Mv {
            row: mv.row - best.row,
            col: mv.col - best.col,
        };
        let joint = if d.row == 0 {
            u8::from(d.col != 0)
        } else if d.col == 0 {
            2
        } else {
            3
        };
        prob::write_tree(self.w, &prob::JOINT_TREE, &s.fc.mv.joints, joint);
        let hp = s.h.allow_hp && best.row.unsigned_abs() < 64 && best.col.unsigned_abs() < 64;
        if joint == 2 || joint == 3 {
            self.write_mv_comp(0, i32::from(d.row), hp);
        }
        if joint == 1 || joint == 3 {
            self.write_mv_comp(1, i32::from(d.col), hp);
        }
        if self.collect() {
            count_mv(&mut self.f.counts, d);
        }
    }

    fn write_mv_comp(&mut self, which: usize, v: i32, use_hp: bool) {
        let c = &self.s.fc.mv.comp[which];
        let sign = v < 0;
        let z = v.unsigned_abs() - 1;
        let class = mv_class(z);
        let base = if class == 0 { 0 } else { 2u32 << (class + 2) };
        let o = z - base;
        let d = o >> 3;
        let f = (o >> 1) & 3;
        let e = o & 1;
        self.w.write(u32::from(sign), u32::from(c.sign));
        prob::write_tree(self.w, &prob::CLASS_TREE, &c.classes, class as u8);
        if class == 0 {
            self.w.write(d, u32::from(c.class0));
            prob::write_tree(self.w, &prob::FP_TREE, &c.class0_fp[d as usize], f as u8);
            if use_hp {
                self.w.write(e, u32::from(c.class0_hp));
            }
        } else {
            for i in 0..class {
                self.w.write((d >> i) & 1, u32::from(c.bits[i]));
            }
            prob::write_tree(self.w, &prob::FP_TREE, &c.fp, f as u8);
            if use_hp {
                self.w.write(e, u32::from(c.hp));
            }
        }
    }

    fn near(&self, b: &Blk, pos: [i8; 2]) -> Option<&Nb> {
        let r = b.row as isize + isize::from(pos[0]);
        let c = b.col as isize + isize::from(pos[1]);
        if r < 0 || c < self.col0 as isize || r >= self.s.mi_rows as isize || c >= self.col1 as isize {
            return None;
        }
        if pos[0] == -1 {
            Some(&self.f.above[c as usize - self.col0])
        } else {
            Some(&self.left[(r as usize) & 7])
        }
    }

    fn cell(&self, b: &Blk, pos: [i8; 2]) -> Option<MvRef> {
        let r = b.row as isize + isize::from(pos[0]);
        let c = b.col as isize + isize::from(pos[1]);
        if r < 0 || c < self.col0 as isize || r >= self.s.mi_rows as isize || c >= self.col1 as isize {
            return None;
        }
        Some(self.f.mvs[r as usize * self.f.width + c as usize - self.col0])
    }

    /// The vector `mode` takes (NEAREST, NEAR) or codes from (NEW): the
    /// list of `dec::parse::find_mv` for blocks of 8×8 and up, reference
    /// LAST, its last entry, with the precision lowered.
    fn find_mv(&self, b: &Blk, mode: u8) -> Mv {
        let s = self.s;
        let frame = LAST_FRAME;
        let bias = &s.h.sign_bias;
        let early = mode != NEAR;
        let mut list = [Mv::ZERO; 2];
        let mut count = 0usize;
        let mut different = false;
        let search = &MV_REF[usize::from(b.bsize)];
        let push = |list: &mut [Mv; 2], count: &mut usize, mv: Mv| -> bool {
            if *count > 0 {
                if list[0] != mv {
                    list[*count] = mv;
                    *count += 1;
                    true
                } else {
                    false
                }
            } else {
                list[0] = mv;
                *count = 1;
                early
            }
        };
        let prev = if s.h.use_prev_mvs {
            s.prev_mvs.and_then(|m| m.get(b.row * s.mi_cols + b.col).copied())
        } else {
            None
        };
        'search: {
            for pos in search {
                if let Some(c) = self.cell(b, *pos) {
                    different = true;
                    let mv = if c.refs[0] == frame {
                        Some(c.mv[0])
                    } else if c.refs[1] == frame {
                        Some(c.mv[1])
                    } else {
                        None
                    };
                    if let Some(mv) = mv
                        && push(&mut list, &mut count, mv)
                    {
                        break 'search;
                    }
                }
            }
            if let Some(p) = prev {
                let mv = if p.refs[0] == frame {
                    Some(p.mv[0])
                } else if p.refs[1] == frame {
                    Some(p.mv[1])
                } else {
                    None
                };
                if let Some(mv) = mv
                    && push(&mut list, &mut count, mv)
                {
                    break 'search;
                }
            }
            if different {
                for pos in search {
                    let Some(c) = self.cell(b, *pos) else { continue };
                    if c.refs[0] == INTRA_FRAME {
                        continue;
                    }
                    if c.refs[0] != frame && push(&mut list, &mut count, flip(c.mv[0], c.refs[0], frame, bias)) {
                        break 'search;
                    }
                    if c.refs[1] > INTRA_FRAME && c.refs[1] != frame && c.mv[1] != c.mv[0] && push(&mut list, &mut count, flip(c.mv[1], c.refs[1], frame, bias)) {
                        break 'search;
                    }
                }
            }
            if let Some(p) = prev {
                if p.refs[0] != frame && p.refs[0] > INTRA_FRAME && push(&mut list, &mut count, flip(p.mv[0], p.refs[0], frame, bias)) {
                    break 'search;
                }
                if p.refs[1] > INTRA_FRAME && p.refs[1] != frame && p.mv[1] != p.mv[0] && push(&mut list, &mut count, flip(p.mv[1], p.refs[1], frame, bias)) {
                    break 'search;
                }
            }
            count = if mode == NEAR { 2 } else { 1 };
        }
        let (bw, bh) = (b.bw as i32, b.bh as i32);
        let (row, col) = (b.row as i32, b.col as i32);
        let top = -(row * 64) - 128;
        let bottom = (s.mi_rows as i32 - bh - row) * 64 + 128;
        let left = -(col * 64) - 128;
        let right = (s.mi_cols as i32 - bw - col) * 64 + 128;
        for mv in list.iter_mut().take(count) {
            mv.row = i32::from(mv.row).clamp(top, bottom.max(top)) as i16;
            mv.col = i32::from(mv.col).clamp(left, right.max(left)) as i16;
        }
        lower_precision(list[count.max(1) - 1], s.h.allow_hp)
    }

    fn reset_nz(&mut self, b: &Blk) {
        for plane in 0..3 {
            let ss = usize::from(plane > 0);
            let w = (b.bw * 2) >> ss;
            let hh = (b.bh * 2) >> ss;
            let a = ((b.col - self.col0) * 2) >> ss;
            let l = ((b.row * 2) & 15) >> ss;
            let above = &mut self.f.above_nz[plane];
            let end = (a + w).min(above.len());
            above[a..end].fill(0);
            let left = &mut self.left_nz[plane];
            left[l..(l + hh).min(16)].fill(0);
        }
    }

    /// Writes the tokens of one tx block, as `dec::parse::decode_coefs`
    /// reads them, and sets its contexts.
    #[allow(clippy::too_many_arguments)]
    fn write_coefs(&mut self, b: &Blk, plane: usize, tx: u8, r4: usize, c4: usize, levels: &[i32], eob: usize) {
        let s = self.s;
        let ss = usize::from(plane > 0);
        let n = 1usize << tx;
        let a = (((b.col - self.col0) * 2) >> ss) + c4;
        let l = (((b.row * 2) & 15) >> ss) + r4;
        let (_, _, vw, vh) = self.visible(b, plane);
        let anz = self.f.above_nz[plane][a..(a + n).min(self.f.above_nz[plane].len())].iter().any(|&v| v != 0);
        let lnz = self.left_nz[plane][l..(l + n).min(16)].iter().any(|&v| v != 0);
        let mut ctx = usize::from(anz) + usize::from(lnz);
        let intra = b.refs[0] == INTRA_FRAME;
        let kind = if intra && plane == 0 && !s.h.lossless && tx != TX_32X32 { geom::intra_tx_type(b.mode) } else { 0 };
        let (scan, nb) = scan_pair(tx, kind);
        let log2n = 2 + usize::from(tx);
        let nn = 4usize << tx;
        let probs = &s.fc.coef[usize::from(tx)][usize::from(plane > 0)][usize::from(!intra)];
        let bands: &[u8] = if tx == 0 { &BAND_4X4 } else { &BAND_8X8_PLUS };
        let collect = self.collect();
        let (ti, pi, ri) = (usize::from(tx), usize::from(plane > 0), usize::from(!intra));
        let max_eob = 16usize << (2 * tx);
        let mut c = 0usize;
        let mut after_zero = false;
        while c < eob {
            let band = usize::from(bands[c]);
            let p = probs[band][ctx];
            let rc = usize::from(scan[c]);
            let t = ((rc & (nn - 1)) << log2n) | (rc >> log2n);
            let v = levels[t];
            if !after_zero {
                if collect {
                    self.f.counts.eob[ti][pi][ri][band][ctx] += 1;
                }
                self.w.write(1, u32::from(p[0]));
            }
            if v == 0 {
                self.w.write(0, u32::from(p[1]));
                if collect {
                    self.f.counts.coef[ti][pi][ri][band][ctx][0] += 1;
                }
                self.cache[rc] = 0;
                after_zero = true;
            } else {
                self.w.write(1, u32::from(p[1]));
                let a = v.unsigned_abs();
                if a == 1 {
                    self.w.write(0, u32::from(p[2]));
                    if collect {
                        self.f.counts.coef[ti][pi][ri][band][ctx][1] += 1;
                    }
                    self.cache[rc] = 1;
                } else {
                    self.w.write(1, u32::from(p[2]));
                    if collect {
                        self.f.counts.coef[ti][pi][ri][band][ctx][2] += 1;
                    }
                    let pp = &PARETO8[usize::from(p[2]) - 1];
                    self.cache[rc] = write_big(self.w, pp, a);
                }
                self.w.write(u32::from(v < 0), 128);
                after_zero = false;
            }
            c += 1;
            if c < max_eob {
                ctx = (1 + usize::from(self.cache[usize::from(nb[2 * c])]) + usize::from(self.cache[usize::from(nb[2 * c + 1])])) >> 1;
            }
        }
        if eob < max_eob {
            // The end of block (always after a nonzero token).
            let band = usize::from(bands[eob]);
            if collect {
                self.f.counts.eob[ti][pi][ri][band][ctx] += 1;
                self.f.counts.coef[ti][pi][ri][band][ctx][3] += 1;
            }
            self.w.write(0, u32::from(probs[band][ctx][0]));
        }
        let on = u8::from(eob > 0);
        let above = &mut self.f.above_nz[plane];
        for i in 0..n {
            if let Some(v) = above.get_mut(a + i) {
                *v = if c4 + i < vw { on } else { 0 };
            }
        }
        let left = &mut self.left_nz[plane];
        for i in 0..n {
            if let Some(v) = left.get_mut(l + i) {
                *v = if r4 + i < vh { on } else { 0 };
            }
        }
    }
}

/// Writes a token of magnitude `a` of 2 and more; returns its energy.
fn write_big(w: &mut BoolWriter, pp: &[u8; 8], a: u32) -> u8 {
    let bits = |w: &mut BoolWriter, v: u32, probs: &[u8]| {
        for (i, &p) in probs.iter().enumerate() {
            w.write((v >> (probs.len() - 1 - i)) & 1, u32::from(p));
        }
    };
    let p = |k: usize| u32::from(pp[k]);
    if a <= 4 {
        w.write(0, p(0));
        if a == 2 {
            w.write(0, p(1));
            2
        } else {
            w.write(1, p(1));
            w.write(a - 3, p(2));
            3
        }
    } else {
        w.write(1, p(0));
        if a <= 10 {
            w.write(0, p(3));
            if a <= 6 {
                w.write(0, p(4));
                bits(w, a - 5, &[159]);
            } else {
                w.write(1, p(4));
                bits(w, a - 7, &[165, 145]);
            }
            4
        } else {
            w.write(1, p(3));
            if a <= 34 {
                w.write(0, p(5));
                if a <= 18 {
                    w.write(0, p(6));
                    bits(w, a - 11, &[173, 148, 140]);
                } else {
                    w.write(1, p(6));
                    bits(w, a - 19, &[176, 155, 140, 135]);
                }
            } else {
                w.write(1, p(5));
                if a <= 66 {
                    w.write(0, p(7));
                    bits(w, a - 35, &[180, 157, 141, 134, 130]);
                } else {
                    w.write(1, p(7));
                    bits(w, (a - 67).min((1 << 14) - 1), &CAT6_PROB);
                }
            }
            5
        }
    }
}

#[inline]
fn sad_row(a: &[u8], b: &[u8]) -> u64 {
    a.iter().zip(b).map(|(&x, &y)| u64::from(x.abs_diff(y))).sum()
}

/// A rough rate of a vector, in 1/256 bits.
fn mv_bits(mv: Mv) -> u32 {
    let comp = |v: i16| -> u32 {
        let a = u32::from(v.unsigned_abs());
        if a == 0 { 256 } else { 256 * (2 + 2 * (32 - (a + 1).leading_zeros())) }
    };
    comp(mv.row) + comp(mv.col)
}

fn mvcost_px(r: i32, c: i32, from: (i32, i32)) -> u64 {
    ((r - from.0).unsigned_abs() + (c - from.1).unsigned_abs()) as u64 * 4
}

fn mv_class(z: u32) -> usize {
    if z >= 8192 {
        10
    } else if z < 16 {
        0
    } else {
        (z >> 3).ilog2() as usize
    }
}

fn flip(mv: Mv, from: u8, to: u8, bias: &[u8; 4]) -> Mv {
    if bias[usize::from(from)] == bias[usize::from(to)] {
        mv
    } else {
        Mv {
            row: mv.row.wrapping_neg(),
            col: mv.col.wrapping_neg(),
        }
    }
}

fn lower_precision(mv: Mv, allow_hp: bool) -> Mv {
    if allow_hp && mv.row.unsigned_abs() < 64 && mv.col.unsigned_abs() < 64 {
        return mv;
    }
    let d = |v: i16| if v & 1 == 0 { v } else if v > 0 { v - 1 } else { v + 1 };
    Mv { row: d(mv.row), col: d(mv.col) }
}

fn count_mv(counts: &mut prob::Counts, d: Mv) {
    let joint = if d.row == 0 {
        usize::from(d.col != 0)
    } else if d.col == 0 {
        2
    } else {
        3
    };
    counts.joints[joint] += 1;
    let mut comp = |which: usize, value: i32| {
        counts.mv_sign[which][usize::from(value < 0)] += 1;
        let z = value.unsigned_abs() - 1;
        let class = mv_class(z);
        let base = if class == 0 { 0 } else { 2u32 << (class + 2) };
        let o = z - base;
        counts.mv_class[which][class] += 1;
        let d = (o >> 3) as usize;
        let f = ((o >> 1) & 3) as usize;
        let e = (o & 1) as usize;
        if class == 0 {
            counts.mv_class0[which][d] += 1;
            counts.mv_class0_fp[which][d][f] += 1;
            counts.mv_class0_hp[which][e] += 1;
        } else {
            for i in 0..class {
                counts.mv_bits[which][i][(d >> i) & 1] += 1;
            }
            counts.mv_fp[which][f] += 1;
            counts.mv_hp[which][e] += 1;
        }
    };
    if joint == 2 || joint == 3 {
        comp(0, i32::from(d.row));
    }
    if joint == 1 || joint == 3 {
        comp(1, i32::from(d.col));
    }
}

use crate::tables::{
    NEIGHBORS_COL_4X4, NEIGHBORS_COL_8X8, NEIGHBORS_COL_16X16, NEIGHBORS_DEFAULT_4X4, NEIGHBORS_DEFAULT_8X8,
    NEIGHBORS_DEFAULT_16X16, NEIGHBORS_DEFAULT_32X32, NEIGHBORS_ROW_4X4, NEIGHBORS_ROW_8X8, NEIGHBORS_ROW_16X16,
    SCAN_COL_4X4, SCAN_COL_8X8, SCAN_COL_16X16, SCAN_DEFAULT_4X4, SCAN_DEFAULT_8X8, SCAN_DEFAULT_16X16,
    SCAN_DEFAULT_32X32, SCAN_ROW_4X4, SCAN_ROW_8X8, SCAN_ROW_16X16,
};

fn scan_pair(tx: u8, kind: u8) -> (&'static [u16], &'static [u16]) {
    match (tx, kind) {
        (0, 1) => (&SCAN_ROW_4X4, &NEIGHBORS_ROW_4X4),
        (0, 2) => (&SCAN_COL_4X4, &NEIGHBORS_COL_4X4),
        (0, _) => (&SCAN_DEFAULT_4X4, &NEIGHBORS_DEFAULT_4X4),
        (1, 1) => (&SCAN_ROW_8X8, &NEIGHBORS_ROW_8X8),
        (1, 2) => (&SCAN_COL_8X8, &NEIGHBORS_COL_8X8),
        (1, _) => (&SCAN_DEFAULT_8X8, &NEIGHBORS_DEFAULT_8X8),
        (2, 1) => (&SCAN_ROW_16X16, &NEIGHBORS_ROW_16X16),
        (2, 2) => (&SCAN_COL_16X16, &NEIGHBORS_COL_16X16),
        (2, _) => (&SCAN_DEFAULT_16X16, &NEIGHBORS_DEFAULT_16X16),
        _ => (&SCAN_DEFAULT_32X32, &NEIGHBORS_DEFAULT_32X32),
    }
}

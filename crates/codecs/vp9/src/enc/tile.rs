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
    let p = p.max(1);
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
    /// The sample SSE a bit of rate is worth.
    pub rd_lambda: f32,
    pub src: Source<'a>,
    /// The quantizer steps of luma and chroma, DC and AC.
    pub dq: [[i32; 2]; 2],
    /// Search effort: higher is faster.
    pub speed: u8,
    /// Which references (LAST, GOLDEN, ALTREF) are worth searching: those
    /// that hold a picture none before them holds.
    pub refs_live: [bool; 3],
}

/// The rate-distortion costs of one tx block (sample SSE plus lambda
/// times bits): as coded, and dropped (no coefficients).
#[derive(Clone, Copy, Default)]
struct TxRd {
    coded: f32,
    dropped: f32,
}

/// A tx block's coefficients: plane, size, position (4×4 units in the
/// block), levels, end of block.
type Tok = (usize, u8, usize, usize, Vec<i32>, usize);

/// What coding a square changes, kept to try another way.
struct Snap {
    at: (usize, usize, usize),
    cost: u64,
    above: [Nb; 8],
    above_part: [u8; 8],
    above_nz: [[u8; 16]; 3],
    left: [Nb; 8],
    left_part: [u8; 8],
    left_nz: [[u8; 16]; 3],
    mvs: [MvRef; 64],
    pix: [Vec<u8>; 3],
}

impl Default for Snap {
    fn default() -> Self {
        Self {
            at: (0, 0, 0),
            cost: 0,
            above: [Nb::default(); 8],
            above_part: [0; 8],
            above_nz: [[0; 16]; 3],
            left: [Nb::default(); 8],
            left_part: [0; 8],
            left_nz: [[0; 16]; 3],
            mvs: [MvRef::default(); 64],
            pix: [vec![0; 64 * 64], vec![0; 64 * 64], vec![0; 64 * 64]],
        }
    }
}

/// The block being encoded.
#[derive(Clone, Copy, Default)]
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
    /// The decision for each square block of the superblock (see
    /// [`Tile::slot`]) and which squares split.
    choice: [Blk; 85],
    split: [bool; 21],
    /// Snapshots: 0 the superblock, then four per level.
    snaps: Vec<Snap>,
    toks: Vec<Tok>,
    /// The cost of the best inter candidate last ranked (SAD units).
    last_sad: u64,
    /// The fast path's search result for each square of the superblock.
    found: [(Mv, u64); 85],
    /// The reference the searches read (0 LAST, 1 GOLDEN, 2 ALTREF).
    cur_ref: usize,
    /// Whether the block coded last skipped its residual; for the best
    /// candidate so far and the block `best_block` chose last, that and
    /// its SSE.
    coded_skip: bool,
    choice_skip: (bool, f64),
    last_skip: (bool, f64),
    /// The costs of the tx block quantized last.
    rd: TxRd,
    /// In a trial nothing is written or counted: bits add to `cost`
    /// (1/256 bits).
    trial: bool,
    cost: u64,
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
            choice: [Blk::default(); 85],
            split: [false; 21],
            snaps: (0..17).map(|_| Snap::default()).collect(),
            toks: Vec::new(),
            last_sad: 0,
            found: [(Mv::ZERO, u64::MAX); 85],
            cur_ref: 0,
            coded_skip: false,
            choice_skip: (false, 0.0),
            last_skip: (false, 0.0),
            rd: TxRd::default(),
            trial: false,
            cost: 0,
        }
    }

    /// The vectors found, for the next row's tile.
    pub fn finish(self) -> Vec<Mv> {
        self.seeds
    }

    /// Encodes the superblocks of one row of this tile: each superblock's
    /// partition and modes are searched by trial coding (nothing written,
    /// rates costed), then it is coded for real with what won.
    pub fn sb_row(&mut self, mi_row: usize) {
        self.left = [Nb::default(); 8];
        self.left_part = [0; 8];
        self.left_nz = [[0; 16]; 3];
        let mut mi_col = self.col0;
        let fast = self.fast();
        while mi_col < self.col1 {
            if fast {
                // Partition from the motion search's SAD, one pass.
                let mut split = [false; 21];
                self.choose_into(mi_row, mi_col, 3, &mut split);
                self.partition(mi_row, mi_col, BLOCK_64X64, 4, &split);
                mi_col += 8;
                continue;
            }
            self.save(mi_row, mi_col, 8, 0);
            self.trial = true;
            self.split = [false; 21];
            self.rd_node(mi_row, mi_col, 3);
            self.restore(0);
            self.trial = false;
            let split = self.split;
            self.partition(mi_row, mi_col, BLOCK_64X64, 4, &split);
            mi_col += 8;
        }
    }

    // Partition decision --------------------------------------------------

    /// Whether this frame takes the one-pass path: inter frames at speed
    /// 7 and up.
    fn fast(&self) -> bool {
        self.s.speed >= 7 && !self.s.intra_frame
    }

    /// Fast partition: each square's cost is its motion search's SAD plus
    /// the rate of its vector (SAD units); it splits when its quarters
    /// cost less.
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

    /// The SAD of a square's best whole-sample vector on LAST plus the
    /// vector's rate; the vector is kept for the block's coding.
    fn estimate(&mut self, row: usize, col: usize, bsize: u8) -> u64 {
        let s = self.s;
        let (bw, _) = MI_WH[usize::from(bsize)];
        let lam = u64::from(s.lambda);
        self.cur_ref = 0;
        let (mv, sad) = self.search(row, col, bsize);
        self.found[Self::slot(row, col, bsize)] = (mv, sad);
        for r in 0..bw.min(s.mi_rows - row) {
            for c in 0..bw.min(s.mi_cols - col) {
                self.seeds[(row + r) * s.mi_cols + col + c] = mv;
            }
        }
        sad + lam * u64::from(mv_bits(mv)) / 256 + lam * 3
    }

    /// Codes a block on the fast path: the best inter candidate, or, when
    /// its prediction is poor, the better of it and the best intra mode.
    fn fast_block(&mut self, row: usize, col: usize, bsize: u8) {
        let base = self.blk(row, col, bsize);
        let mut cands = Vec::with_capacity(2);
        self.inter_cands(&base, &mut cands);
        if self.intra_worth(&base, &cands) {
            self.intra_cands(&base, &mut cands);
        }
        let pick = if cands.len() <= 1 {
            0
        } else {
            let lam = f64::from(self.s.rd_lambda) / 256.0;
            let n = MI_WH[usize::from(bsize)].0;
            self.trial = true;
            self.save(row, col, n, 1);
            let mut best = (f64::INFINITY, 0);
            for (i, b) in cands.iter().enumerate() {
                if i > 0 {
                    self.restore(1);
                }
                let c0 = self.cost;
                let j = self.code_block(b) + (self.cost - c0) as f64 * lam;
                if j < best.0 {
                    best = (j, i);
                }
            }
            self.restore(1);
            self.trial = false;
            best.1
        };
        if let Some(b) = cands.get(pick).copied() {
            self.code_block(&b);
        }
    }

    /// Where the decision for the square block `bsize` at `(row, col)` is
    /// kept, in its superblock.
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

    /// The least cost (SSE + lambda · bits) of the square node of `level`
    /// at `(row, col)`: coded whole, or split (recursively). Leaves the
    /// state as the winner left it, and its decisions in `split` and
    /// `choice`.
    fn rd_node(&mut self, row: usize, col: usize, level: usize) -> f64 {
        let s = self.s;
        if row >= s.mi_rows || col >= s.mi_cols {
            return 0.0;
        }
        let lam = f64::from(s.rd_lambda) / 256.0;
        let n = 1usize << level;
        let hbs = n / 2;
        let bsize = SQUARE[level];
        let whole = level == 0 || (row + hbs < s.mi_rows && col + hbs < s.mi_cols);
        let before = 1 + 4 * (3 - level);
        self.save(row, col, n, before);
        let mut j_none = f64::INFINITY;
        if whole {
            let c0 = self.cost;
            self.part_symbol(row, col, bsize, PARTITION_NONE);
            let sym = (self.cost - c0) as f64 * lam;
            j_none = sym + self.best_block(row, col, bsize, before + 2);
            self.part_after(row, col, bsize, PARTITION_NONE);
        }
        if level == 0 {
            return j_none;
        }
        // No split when the whole block needs no residual and its error is
        // under the quantizer's own noise.
        if whole && self.last_skip.0 {
            let pixels = (n * 8 * n * 8) as f64;
            let step = f64::from(s.dq[0][1]) / 8.0;
            if self.last_skip.1 < pixels * step * step / 24.0 {
                return j_none;
            }
        }
        if whole {
            self.save(row, col, n, before + 1);
            self.restore(before);
        }
        let c0 = self.cost;
        self.part_symbol(row, col, bsize, PARTITION_SPLIT);
        let mut j = (self.cost - c0) as f64 * lam;
        for (dr, dc) in [(0, 0), (0, hbs), (hbs, 0), (hbs, hbs)] {
            if j >= j_none {
                break;
            }
            j += self.rd_node(row + dr, col + dc, level - 1);
        }
        if j < j_none {
            self.split[Self::node(row, col, level)] = true;
            j
        } else {
            self.restore(before + 1);
            j_none
        }
    }

    /// Codes the best of the block's candidate modes (leaving the state
    /// as it leaves it) and returns its cost. `slot` holds the state
    /// before it.
    fn best_block(&mut self, row: usize, col: usize, bsize: u8, slot: usize) -> f64 {
        let s = self.s;
        let lam = f64::from(s.rd_lambda) / 256.0;
        let n = MI_WH[usize::from(bsize)].0;
        self.save(row, col, n, slot);
        let mut cands: Vec<Blk> = Vec::with_capacity(8);
        let base = self.blk(row, col, bsize);
        if !s.intra_frame {
            self.inter_cands(&base, &mut cands);
        }
        let intra_first = cands.len();
        if s.intra_frame || self.intra_worth(&base, &cands) {
            self.intra_cands(&base, &mut cands);
        }
        let mut best = (f64::INFINITY, 0usize);
        let mut last = usize::MAX;
        // The state after the best so far is kept in `slot + 1`.
        let mut try_one = |t: &mut Self, i: usize, b: &Blk, best: &mut (f64, usize), last: &mut usize| {
            if *last != usize::MAX {
                t.restore(slot);
            }
            let c0 = t.cost;
            let d = t.code_block(b);
            let j = d + (t.cost - c0) as f64 * lam;
            *last = i;
            if j < best.0 {
                *best = (j, i);
                t.choice_skip = (t.coded_skip, d);
                t.save(row, col, n, slot + 1);
            }
        };
        for (i, b) in cands.clone().iter().enumerate() {
            try_one(self, i, b, &mut best, &mut last);
        }
        // Chroma and transform size for the best intra mode.
        if best.1 >= intra_first && cands.len() > intra_first && s.speed < 8 {
            let b0 = cands[best.1];
            let mut more = Vec::new();
            for uv in [b0.mode, geom::TM, geom::V, geom::H] {
                if uv != b0.uv && !more.iter().any(|m: &Blk| m.uv == uv) && more.len() < if s.speed < 4 { 3 } else { 1 } {
                    more.push(Blk { uv, ..b0 });
                }
            }
            for b in more {
                let i = cands.len();
                cands.push(b);
                try_one(self, i, &b, &mut best, &mut last);
            }
        }
        if s.speed < 8 {
            let b0 = cands[best.1];
            let tries = if s.speed < 4 { 2 } else { 1 };
            let mut tx = b0.tx;
            for _ in 0..tries {
                if tx == 0 || b0.bsize < BLOCK_8X8 {
                    break;
                }
                tx -= 1;
                let i = cands.len();
                let b = Blk { tx, ..b0 };
                cands.push(b);
                try_one(self, i, &b, &mut best, &mut last);
            }
        }
        if last != best.1 {
            self.restore(slot + 1);
        }
        self.last_skip = self.choice_skip;
        self.choice[Self::slot(row, col, bsize)] = cands[best.1];
        best.0
    }

    /// A block at `(row, col)` with the defaults.
    fn blk(&self, row: usize, col: usize, bsize: u8) -> Blk {
        let s = self.s;
        let (bw, bh) = MI_WH[usize::from(bsize)];
        Blk {
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
        }
    }

    /// Intra candidates: the luma modes whose predictions are nearest the
    /// source (by the sum of absolute Hadamard coefficients), chroma DC.
    fn intra_cands(&mut self, base: &Blk, out: &mut Vec<Blk>) {
        let s = self.s;
        let tx = base.tx;
        let n = 4usize << tx;
        let (x, y) = (base.col * 8, base.row * 8);
        let have_above = base.row > 0;
        let have_left = base.col > self.col0;
        let e = self.rec.intra_edges(0, tx, x, y, have_left, have_above, false);
        let (src, ss) = (s.src.planes[0], s.src.strides[0]);
        let modes: &[u8] = if s.speed >= 8 {
            &[geom::DC, geom::V, geom::H, geom::TM]
        } else {
            &[geom::DC, geom::V, geom::H, geom::TM, geom::D45, geom::D135, geom::D117, geom::D153, geom::D207, geom::D63]
        };
        let mut scored: Vec<(u64, u8)> = Vec::with_capacity(10);
        let lam = u64::from(s.lambda);
        for &m in modes {
            ipred::predict(m, n, &mut self.pred, n, &e, 8);
            let cost = satd(&src[y * ss + x..], ss, &self.pred, n, n) + lam * u64::from(m != geom::DC) * 2;
            scored.push((cost, m));
        }
        scored.sort_unstable();
        let k = match s.speed {
            0..=3 => 4,
            4..=6 => 3,
            _ => 1,
        };
        for &(_, m) in scored.iter().take(k) {
            out.push(Blk {
                mode: m,
                uv: geom::DC,
                refs: [INTRA_FRAME; 2],
                ..*base
            });
        }
    }

    /// Whether intra is worth trying in an inter frame: when the best
    /// inter prediction is poor.
    fn intra_worth(&self, base: &Blk, cands: &[Blk]) -> bool {
        if cands.is_empty() {
            return true;
        }
        let pixels = (base.x_mis * base.y_mis * 64) as u64;
        // A mean error of more than a few levels a sample.
        self.last_sad > pixels * if self.s.speed >= 8 { 12 } else { 5 }
    }

    /// Inter candidates: the searched vector and the reference list's,
    /// the best two by SAD and vector rate.
    fn inter_cands(&mut self, base: &Blk, out: &mut Vec<Blk>) {
        let s = self.s;
        let (row, col, bsize) = (base.row, base.col, base.bsize);
        let (x, y) = (col * 8, row * 8);
        let w = (base.bw * 8).min((s.mi_cols - col) * 8);
        let h = (base.bh * 8).min((s.mi_rows - row) * 8);
        let lam = u64::from(s.lambda);
        let mut scored: Vec<(u64, Blk)> = Vec::with_capacity(8);
        for r in 0..3 {
            if !s.refs_live[r] {
                continue;
            }
            self.cur_ref = r;
            if self.rec_ref().is_none() {
                continue;
            }
            let (mv, sad) = if r == 0 && self.fast() { self.found[Self::slot(row, col, bsize)] } else { self.search(row, col, bsize) };
            if r == 0 {
                let (bw, _) = MI_WH[usize::from(bsize)];
                for rr in 0..bw.min(s.mi_rows - row) {
                    for c in 0..bw.min(s.mi_cols - col) {
                        self.seeds[(row + rr) * s.mi_cols + col + c] = mv;
                    }
                }
            }
            let (mv, sad) = self.refine(row, col, bsize, mv, sad);
            let mut b = Blk {
                refs: [r as u8 + 1, INTRA_FRAME],
                ..*base
            };
            let nearest = self.find_mv(&b, NEAREST);
            let near = self.find_mv(&b, NEAR);
            // Other references cost their symbol: about a bit or two more.
            let ref_bits = lam * 2 * r as u64;
            let mode_of = |v: Mv, m: u8| {
                if v == nearest {
                    NEAREST
                } else if v == near {
                    NEAR
                } else if v == Mv::ZERO {
                    ZERO
                } else {
                    m
                }
            };
            b.mv = mv;
            b.mode = mode_of(mv, NEW);
            let new_bits = if b.mode == NEW {
                lam * u64::from(mv_bits(Mv { row: mv.row - nearest.row, col: mv.col - nearest.col })) / 256 + lam * 2
            } else {
                lam
            };
            scored.push((sad + new_bits + ref_bits, b));
            for v in [nearest, near, Mv::ZERO] {
                if v == mv || scored.iter().any(|(_, o)| o.refs == b.refs && o.mv == v) {
                    continue;
                }
                if let Some(d) = self.sad_sub(x, y, w, h, v) {
                    scored.push((d + lam + ref_bits, Blk { mv: v, mode: mode_of(v, ZERO), ..b }));
                }
            }
        }
        self.cur_ref = 0;
        scored.sort_by_key(|t| t.0);
        self.last_sad = scored.first().map_or(u64::MAX, |t| t.0);
        let k = match s.speed {
            0..=3 => 3,
            4..=7 => 2,
            _ => 1,
        };
        out.extend(scored.iter().take(k).map(|t| t.1));
    }

    // State snapshots ------------------------------------------------------

    /// Keeps in snapshot `k` everything coding the square of `n` 8×8
    /// blocks at `(row, col)` changes: its contexts, motion and samples.
    fn save(&mut self, row: usize, col: usize, n: usize, k: usize) {
        let mut sn = std::mem::take(&mut self.snaps[k]);
        sn.at = (row, col, n);
        sn.cost = self.cost;
        let a = col - self.col0;
        let aw = n.min(self.f.above.len() - a);
        sn.above[..aw].copy_from_slice(&self.f.above[a..a + aw]);
        sn.above_part[..aw].copy_from_slice(&self.f.above_part[a..a + aw]);
        let l = row & 7;
        sn.left[..n].copy_from_slice(&self.left[l..l + n]);
        sn.left_part[..n].copy_from_slice(&self.left_part[l..l + n]);
        for p in 0..3 {
            let ss = usize::from(p > 0);
            let (a4, n4) = ((a * 2) >> ss, (n * 2) >> ss);
            let w4 = n4.min(self.f.above_nz[p].len() - a4);
            sn.above_nz[p][..w4].copy_from_slice(&self.f.above_nz[p][a4..a4 + w4]);
            let l4 = ((row * 2) & 15) >> ss;
            sn.left_nz[p][..n4].copy_from_slice(&self.left_nz[p][l4..l4 + n4]);
        }
        let rows = n.min(self.s.mi_rows - row);
        let cols = n.min(self.col1 - col);
        for r in 0..rows {
            let at = (row + r) * self.f.width + a;
            sn.mvs[r * 8..r * 8 + cols].copy_from_slice(&self.f.mvs[at..at + cols]);
        }
        let views = self.rec.views();
        for p in 0..3 {
            let ss = usize::from(p > 0);
            let (x, y, m) = ((col * 8) >> ss, (row * 8) >> ss, (n * 8) >> ss);
            let v = &mut views[p];
            let stride = v.stride;
            let src = v.at(y, x);
            for r in 0..m {
                sn.pix[p][r * 64..r * 64 + m].copy_from_slice(&src[r * stride..r * stride + m]);
            }
        }
        self.snaps[k] = sn;
    }

    /// Puts back what snapshot `k` kept.
    fn restore(&mut self, k: usize) {
        let sn = std::mem::take(&mut self.snaps[k]);
        let (row, col, n) = sn.at;
        self.cost = sn.cost;
        let a = col - self.col0;
        let aw = n.min(self.f.above.len() - a);
        self.f.above[a..a + aw].copy_from_slice(&sn.above[..aw]);
        self.f.above_part[a..a + aw].copy_from_slice(&sn.above_part[..aw]);
        let l = row & 7;
        self.left[l..l + n].copy_from_slice(&sn.left[..n]);
        self.left_part[l..l + n].copy_from_slice(&sn.left_part[..n]);
        for p in 0..3 {
            let ss = usize::from(p > 0);
            let (a4, n4) = ((a * 2) >> ss, (n * 2) >> ss);
            let w4 = n4.min(self.f.above_nz[p].len() - a4);
            self.f.above_nz[p][a4..a4 + w4].copy_from_slice(&sn.above_nz[p][..w4]);
            let l4 = ((row * 2) & 15) >> ss;
            self.left_nz[p][l4..l4 + n4].copy_from_slice(&sn.left_nz[p][..n4]);
        }
        let rows = n.min(self.s.mi_rows - row);
        let cols = n.min(self.col1 - col);
        for r in 0..rows {
            let at = (row + r) * self.f.width + a;
            self.f.mvs[at..at + cols].copy_from_slice(&sn.mvs[r * 8..r * 8 + cols]);
        }
        let views = self.rec.views();
        for p in 0..3 {
            let ss = usize::from(p > 0);
            let (x, y, m) = ((col * 8) >> ss, (row * 8) >> ss, (n * 8) >> ss);
            let v = &mut views[p];
            let stride = v.stride;
            let dst = v.at(y, x);
            for r in 0..m {
                dst[r * stride..r * stride + m].copy_from_slice(&sn.pix[p][r * 64..r * 64 + m]);
            }
        }
        self.snaps[k] = sn;
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
        let rp = c.refs[self.cur_ref].as_ref()?;
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
            && usize::from(m.refs[0]) == self.cur_ref + 1
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
        // Diamond search in whole samples; below 64×64 the search starts
        // from the square around it, whose vector is a seed, with a smaller
        // step.
        let range = 64;
        let mut step = match (bsize, s.speed) {
            (BLOCK_64X64, _) | (_, 0..=2) => 8,
            (_, 3..=6) => 4,
            _ => 2,
        };
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

    /// Codes the partition of the square at `(row, col)` as `split` and
    /// `choice` decided.
    fn partition(&mut self, row: usize, col: usize, bsize: u8, n4: u32, split: &[bool; 21]) {
        let s = self.s;
        if row >= s.mi_rows || col >= s.mi_cols {
            return;
        }
        let n8 = n4 - 1;
        let hbs = (1usize << n8) >> 1;
        let fits = row + hbs < s.mi_rows && col + hbs < s.mi_cols;
        let p = if hbs == 0 {
            PARTITION_NONE
        } else if !fits || split[Self::node(row, col, n8 as usize)] {
            PARTITION_SPLIT
        } else {
            PARTITION_NONE
        };
        self.part_symbol(row, col, bsize, p);
        let sub = geom::subsize(p, bsize);
        if p == PARTITION_NONE {
            if self.fast() {
                self.fast_block(row, col, bsize);
            } else {
                let b = self.choice[Self::slot(row, col, bsize)];
                self.code_block(&b);
            }
        } else {
            for (dr, dc) in [(0, 0), (0, hbs), (hbs, 0), (hbs, hbs)] {
                self.partition(row + dr, col + dc, sub, n8, split);
            }
        }
        self.part_after(row, col, bsize, p);
        let _ = (PARTITION_HORZ, PARTITION_VERT);
    }

    /// The partition symbol of the square at `(row, col)`.
    fn part_symbol(&mut self, row: usize, col: usize, bsize: u8, p: u8) {
        let s = self.s;
        let n8 = geom::mi_width_log2(bsize) as usize;
        let hbs = (1usize << n8) >> 1;
        let has_rows = row + hbs < s.mi_rows;
        let has_cols = col + hbs < s.mi_cols;
        let ctx = {
            let above = usize::from(self.f.above_part[col - self.col0] >> n8) & 1;
            let left = usize::from(self.left_part[row & 7] >> n8) & 1;
            left * 2 + above + n8 * 4
        };
        let probs = s.part_probs[ctx];
        if has_rows && has_cols {
            self.put_tree(&prob::PARTITION_TREE, &probs, p);
        } else if !has_rows && has_cols {
            self.put(u32::from(p == PARTITION_SPLIT), u32::from(probs[1]));
        } else if has_rows && !has_cols {
            self.put(u32::from(p == PARTITION_SPLIT), u32::from(probs[2]));
        }
        if self.collect() {
            self.f.counts.partition[ctx][usize::from(p)] += 1;
        }
    }

    /// The partition contexts a coded square leaves.
    fn part_after(&mut self, row: usize, col: usize, bsize: u8, p: u8) {
        if bsize == BLOCK_8X8 || p != PARTITION_SPLIT {
            let n = 1usize << geom::mi_width_log2(bsize);
            let sub = geom::subsize(p, bsize);
            let (above, left) = geom::partition_context_byte(sub);
            let a = col - self.col0;
            let end = (a + n).min(self.f.above_part.len());
            self.f.above_part[a..end].fill(above);
            let l = row & 7;
            self.left_part[l..(l + n).min(8)].fill(left);
        }
    }

    fn above_nb(&self, b: &Blk) -> Option<&Nb> {
        (b.row > 0).then(|| &self.f.above[b.col - self.col0])
    }

    fn left_nb(&self, b: &Blk) -> Option<&Nb> {
        (b.col > self.col0).then(|| &self.left[b.row & 7])
    }

    fn collect(&self) -> bool {
        !self.trial && !self.s.h.error_resilient && !self.s.h.frame_parallel
    }

    /// Writes a bit, or in a trial adds its cost.
    #[inline]
    fn put(&mut self, bit: u32, prob: u32) {
        if self.trial {
            self.cost += u64::from(bit_cost(bit != 0, prob as u8));
        } else {
            self.w.write(bit, prob);
        }
    }

    /// Writes a symbol of a tree, or in a trial adds its cost.
    fn put_tree(&mut self, tree: &[i8], probs: &[u8], symbol: u8) {
        if self.trial {
            let mut c = Cost(0);
            tree_to(&mut c, tree, probs, symbol);
            self.cost += u64::from(c.0);
        } else {
            prob::write_tree(self.w, tree, probs, symbol);
        }
    }

    // Blocks ----------------------------------------------------------------

    /// Codes one block as `b0` says (in a trial, costed): prediction,
    /// residual, the skip decision, the syntax, the contexts. Returns the
    /// SSE of its reconstruction.
    fn code_block(&mut self, b0: &Blk) -> f64 {
        let s = self.s;
        let mut b = *b0;
        let (row, col, bsize) = (b.row, b.col, b.bsize);
        let (bw, bh) = (b.bw, b.bh);
        let inter = b.refs[0] != INTRA_FRAME;
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
        if inter {
            self.rec.predict_inter(&rec);
        }
        // Coefficients of every tx block, kept until the mode info is
        // written (intra predicts tx block by tx block from the ones
        // before, so it is reconstructed as it goes).
        let mut toks = std::mem::take(&mut self.toks);
        toks.clear();
        let mut eobs = 0;
        let (coded, dropped) = self.residual(&b, &rec, &mut toks, &mut eobs);
        if inter && eobs > 0 {
            // Skip when the residual is not worth its bits.
            let ctx = usize::from(self.above_nb(&b).is_some_and(|n| n.skip)) + usize::from(self.left_nb(&b).is_some_and(|n| n.skip));
            let p = s.fc.skip[ctx];
            let lam = s.rd_lambda / 256.0;
            let j_skip = dropped + lam * bit_cost(true, p) as f32;
            let j_coded = coded + lam * bit_cost(false, p) as f32;
            if j_skip <= j_coded {
                eobs = 0;
                self.rec.predict_inter(&rec);
            }
        }
        b.skip = eobs == 0;
        if inter && b.skip {
            // A skipped inter block takes the largest transform.
            b.tx = geom::max_tx(bsize).min(geom::tx_mode_biggest(s.h.tx_mode));
        }
        // Mode info.
        let pos0 = self.w.bits_written();
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
        let pos1 = self.w.bits_written();
        if b.skip {
            self.reset_nz(&b);
        } else {
            for (plane, tx, r4, c4, levels, eob) in toks.iter() {
                self.write_coefs(&b, *plane, *tx, *r4, *c4, levels, *eob);
            }
        }
        self.toks = toks;
        if !self.trial {
            let pos2 = self.w.bits_written();
            stats::add(&[(0, pos1 - pos0), (1, pos2 - pos1), (2 + usize::from(b.mode).min(13), 1), (16 + usize::from(b.skip), 1), (18 + usize::from(bsize), 1)]);
        }
        let nb = Nb {
            bsize,
            mode: b.mode,
            bmode: [b.mode; 4],
            skip: b.skip,
            tx: b.tx,
            seg_pred: false,
            interp: if inter { geom::EIGHTTAP } else { 3 },
            refs: b.refs,
            bmv: [[b.mv, Mv::ZERO]; 4],
        };
        let a = col - self.col0;
        let end = (a + bw).min(self.f.above.len());
        self.f.above[a..end].fill(nb);
        let l = row & 7;
        self.left[l..(l + bh).min(8)].fill(nb);
        if s.filter_on && !self.trial {
            let index = (row >> 3) * self.f.width.div_ceil(8) + ((col - self.col0) >> 3);
            lpf::build(
                &mut self.f.masks[index],
                s.levels,
                &lpf::Block {
                    bsize,
                    tx: b.tx,
                    skip: b.skip,
                    inter,
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
        self.coded_skip = b.skip;
        if self.trial { self.block_sse(&b) } else { 0.0 }
    }

    /// The SSE of the block's reconstruction against the source, inside
    /// the frame.
    fn block_sse(&mut self, b: &Blk) -> f64 {
        let s = self.s;
        let mut total = 0u64;
        for p in 0..3 {
            let ss = usize::from(p > 0);
            let (_, _, vw, vh) = self.visible(b, p);
            let (x, y) = ((b.col * 8) >> ss, (b.row * 8) >> ss);
            let (src, sst) = (s.src.planes[p], s.src.strides[p]);
            let v = &mut self.rec.views()[p];
            let stride = v.stride;
            total += sad::sse(&src[y * sst + x..], sst, v.at(y, x), stride, vw * 4, vh * 4);
        }
        total as f64
    }

    fn uv_tx(&self, b: &Blk) -> u8 {
        geom::uv_tx_420(b.bsize, b.tx)
    }

    // Residual -----------------------------------------------------------

    /// Every tx block's prediction (intra), residual, quantized levels and
    /// reconstruction, in the decoder's order.
    /// Returns the summed costs of the tx blocks as coded and dropped.
    fn residual(&mut self, b: &Blk, rec: &BlockRec, toks: &mut Vec<Tok>, eobs: &mut usize) -> (f32, f32) {
        let s = self.s;
        let mut sums = (0.0f32, 0.0f32);
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
                    sums.0 += self.rd.coded;
                    sums.1 += self.rd.dropped;
                    toks.push((plane, tx, r, c, levels, eob));
                    c += step;
                }
                r += step;
            }
        }
        sums
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
        self.rd = TxRd::default();
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
            let (eob, rd) = self.rd_levels(plane, tx, kind, x, y, intra, &mut levels);
            self.rd = rd;
            let _ = eob;
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

    /// Rate-distortion quantization of `self.fcoef` into `levels`: each
    /// coefficient takes the level, of the two nearest its value (or
    /// zero), with the least distortion plus `lambda` times its token's
    /// bits, in scan order (so each token's context is known); then the
    /// end of block goes where the cost is least. Returns the end of
    /// block and the costs of coding and of dropping the block.
    #[allow(clippy::too_many_arguments)]
    fn rd_levels(&mut self, plane: usize, tx: u8, kind: u8, x: usize, y: usize, intra: bool, levels: &mut [i32]) -> (usize, TxRd) {
        let s = self.s;
        let n = 4usize << tx;
        let log2n = 2 + usize::from(tx);
        let shift = u32::from(tx == 3);
        let dqi = usize::from(plane > 0);
        let (dc_q, ac_q) = (s.dq[dqi][0], s.dq[dqi][1]);
        let g = super::tx::gain(usize::from(tx));
        let lam = s.rd_lambda * g / 256.0;
        let (scan, nb) = scan_pair(tx, kind);
        let probs = &s.fc.coef[usize::from(tx)][dqi][usize::from(!intra)];
        let bands: &[u8] = if tx == 0 { &BAND_4X4 } else { &BAND_8X8_PLUS };
        let max = n * n;
        // The context of the first token, from the neighbours so far.
        let ctx0 = {
            let ss = usize::from(plane > 0);
            let n4 = 1usize << tx;
            let a = (x >> 2) - ((self.col0 * 8 >> ss) >> 2);
            let l = (y >> 2) & (15 >> ss);
            let anz = self.f.above_nz[plane][a..(a + n4).min(self.f.above_nz[plane].len())].iter().any(|&v| v != 0);
            let lnz = self.left_nz[plane][l..(l + n4).min(16)].iter().any(|&v| v != 0);
            usize::from(anz) + usize::from(lnz)
        };
        // Values in steps, in scan order; nothing past the last that could
        // round to a level is worth looking at.
        let mut av = [0.0f32; 1024];
        let mut steps = [0.0f32; 2];
        steps[0] = dc_q as f32 / (1 << shift) as f32;
        steps[1] = ac_q as f32 / (1 << shift) as f32;
        let mut last_sig = 0usize;
        let mut d_all = 0.0f32;
        for k in 0..max {
            let rc = usize::from(scan[k]);
            let t = ((rc & (n - 1)) << log2n) | (rc >> log2n);
            let v = self.fcoef[t];
            let st = steps[usize::from(k > 0)];
            av[k] = v.abs() / st;
            d_all += v * v;
            if av[k] >= 0.3 {
                last_sig = k + 1;
            }
        }
        if last_sig == 0 {
            let e0 = lam * bit_cost(false, probs[0][ctx0][0]) as f32;
            levels[..max].fill(0);
            let d = (d_all + e0) / g;
            return (0, TxRd { coded: d, dropped: d });
        }
        // Token costs by band and context.
        let cost = &*COST;
        let bc = |bit: bool, p: u8| u32::from(if bit { cost[256 - usize::from(p.max(1))] } else { cost[usize::from(p.max(1))] });
        let mut tc = [[[0u32; 5]; 6]; 6];
        for (band, row) in probs.iter().enumerate() {
            for (c, p) in row.iter().enumerate() {
                // more, eob, zero, one, two-or-more (signs added below)
                tc[band][c] = [bc(true, p[0]), bc(false, p[0]), bc(false, p[1]), bc(true, p[1]) + bc(false, p[2]) + 256, bc(true, p[1]) + bc(true, p[2]) + 256];
            }
        }
        let mut energy = [0u8; 1024];
        let mut ctx = ctx0;
        let mut prev_zero = false;
        // Per position: the chosen cost (distortion + lambda·rate) so far,
        // the distortion if it were zero, and the end-of-block cost there.
        let mut j_pref = [0.0f32; 1025];
        let mut d_tail = [0.0f32; 1025];
        let mut eob_cost = [0.0f32; 1025];
        let mut last_nz = 0usize;
        let mut d_rest = d_all;
        for k in 0..last_sig {
            let rc = usize::from(scan[k]);
            let t = ((rc & (n - 1)) << log2n) | (rc >> log2n);
            let st = steps[usize::from(k > 0)];
            let a = av[k];
            let band = usize::from(bands[k]);
            let tcs = &tc[band][ctx];
            eob_cost[k] = lam * tcs[1] as f32;
            let lo = a as u32;
            let more = if prev_zero { 0 } else { tcs[0] };
            let d_of = |l: u32| {
                let e = (a - l as f32) * st;
                e * e
            };
            let r_of = |l: u32| -> u32 {
                match l {
                    0 => more + tcs[2],
                    1 => more + tcs[3],
                    _ => {
                        let mut c = Cost(0);
                        write_big(&mut c, &PARETO8[usize::from(probs[band][ctx][2].max(1)) - 1], l);
                        more + tcs[4] + c.0
                    }
                }
            };
            let mut best = (f32::MAX, 0u32);
            let cands = if lo == 0 { [0, 1, 1] } else if lo == 1 { [0, 1, 2] } else { [lo, lo + 1, lo + 1] };
            for &l in &cands {
                let j = d_of(l) + lam * r_of(l) as f32;
                if j < best.0 {
                    best = (j, l);
                }
            }
            let l = best.1;
            let v = self.fcoef[t];
            levels[t] = if v < 0.0 { -(l as i32) } else { l as i32 };
            j_pref[k + 1] = j_pref[k] + best.0;
            d_tail[k] = d_rest;
            d_rest -= d_of(0);
            if l != 0 {
                last_nz = k + 1;
            }
            energy[rc] = match l {
                0 => 0,
                1 => 1,
                2 => 2,
                3 | 4 => 3,
                5..=10 => 4,
                _ => 5,
            };
            prev_zero = l == 0;
            if k + 1 < max {
                ctx = (1 + usize::from(energy[usize::from(nb[2 * (k + 1)])]) + usize::from(energy[usize::from(nb[2 * (k + 1) + 1])])) >> 1;
            }
        }
        d_tail[last_sig] = d_rest.max(0.0);
        if last_sig < max {
            eob_cost[last_sig] = lam * tc[usize::from(bands[last_sig])][ctx][1] as f32;
        }
        for k in last_sig..max {
            let rc = usize::from(scan[k]);
            let t = ((rc & (n - 1)) << log2n) | (rc >> log2n);
            levels[t] = 0;
        }
        // The best end: after a nonzero level, or none.
        let mut best = (d_tail[0] + eob_cost[0], 0usize);
        for e in 1..=last_nz {
            let rc = usize::from(scan[e - 1]);
            let t = ((rc & (n - 1)) << log2n) | (rc >> log2n);
            if levels[t] == 0 {
                continue;
            }
            let end = if e < max { eob_cost[e] } else { 0.0 };
            let j = j_pref[e] + d_tail[e] + end;
            if j < best.0 {
                best = (j, e);
            }
        }
        let eob = best.1;
        for k in eob..last_sig {
            let rc = usize::from(scan[k]);
            let t = ((rc & (n - 1)) << log2n) | (rc >> log2n);
            levels[t] = 0;
        }
        let zero = TxRd {
            coded: best.0 / g,
            dropped: (d_tail[0] + eob_cost[0]) / g,
        };
        (eob, zero)
    }

    // Syntax ---------------------------------------------------------------

    fn write_skip(&mut self, b: &Blk) {
        let ctx = usize::from(self.above_nb(b).is_some_and(|n| n.skip)) + usize::from(self.left_nb(b).is_some_and(|n| n.skip));
        self.put(u32::from(b.skip), u32::from(self.s.fc.skip[ctx]));
        if self.collect() {
            self.f.counts.skip[ctx][usize::from(b.skip)] += 1;
        }
    }

    /// The tx size, when the frame selects it per block (parse.rs
    /// `read_tx`).
    fn write_tx(&mut self, b: &Blk, allow: bool) {
        let s = self.s;
        let limit = geom::max_tx(b.bsize);
        if !(allow && s.h.tx_mode == geom::TX_MODE_SELECT && b.bsize >= BLOCK_8X8) {
            return;
        }
        let above = self.above_nb(b);
        let left = self.left_nb(b);
        let ntx = |n: Option<&Nb>| n.map_or(limit, |n| if n.skip { limit } else { n.tx });
        let mut a = ntx(above);
        let mut l = ntx(left);
        if left.is_none() {
            l = a;
        }
        if above.is_none() {
            a = l;
        }
        let ctx = usize::from(a + l > limit);
        let fc = s.fc;
        let probs: [u8; 3] = match limit {
            geom::TX_8X8 => [fc.tx8[ctx][0], 0, 0],
            geom::TX_16X16 => [fc.tx16[ctx][0], fc.tx16[ctx][1], 0],
            _ => fc.tx32[ctx],
        };
        let tx = b.tx;
        self.put(u32::from(tx != 0), u32::from(probs[0]));
        if tx != 0 && limit >= geom::TX_16X16 {
            self.put(u32::from(tx > 1), u32::from(probs[1]));
            if tx != 1 && limit >= TX_32X32 {
                self.put(u32::from(tx > 2), u32::from(probs[2]));
            }
        }
        if self.collect() {
            let t = usize::from(tx);
            match limit {
                geom::TX_8X8 => self.f.counts.tx8[ctx][t] += 1,
                geom::TX_16X16 => self.f.counts.tx16[ctx][t] += 1,
                _ => self.f.counts.tx32[ctx][t] += 1,
            }
        }
    }

    fn write_intra_frame_info(&mut self, b: &Blk) {
        self.write_skip(b);
        self.write_tx(b, true);
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
        self.put_tree(&prob::INTRA_TREE, &KF_Y_MODE_PROB[usize::from(a)][usize::from(l)], b.mode);
        self.put_tree(&prob::INTRA_TREE, &KF_UV_MODE_PROB[usize::from(b.mode)], b.uv);
    }

    fn write_inter_frame_info(&mut self, b: &Blk) {
        let s = self.s;
        self.write_skip(b);
        let inter = b.refs[0] != INTRA_FRAME;
        let ctx = ctx::intra_inter_ctx(self.above_nb(b), self.left_nb(b));
        self.put(u32::from(inter), u32::from(s.fc.intra_inter[ctx]));
        if self.collect() {
            self.f.counts.intra_inter[ctx][usize::from(inter)] += 1;
        }
        self.write_tx(b, !b.skip || !inter);
        if !inter {
            let group = geom::size_group(b.bsize);
            self.put_tree(&prob::INTRA_TREE, &s.fc.y_mode[group], b.mode);
            if self.collect() {
                self.f.counts.y_mode[group][usize::from(b.mode)] += 1;
            }
            self.put_tree(&prob::INTRA_TREE, &s.fc.uv_mode[usize::from(b.mode)], b.uv);
            if self.collect() {
                self.f.counts.uv_mode[usize::from(b.mode)][usize::from(b.uv)] += 1;
            }
            return;
        }
        // Single reference: LAST, or GOLDEN / ALTREF.
        let c0 = ctx::single_p1(self.above_nb(b), self.left_nb(b));
        let bit0 = u32::from(b.refs[0] != LAST_FRAME);
        self.put(bit0, u32::from(s.fc.single_ref[c0][0]));
        if self.collect() {
            self.f.counts.single_ref[c0][0][bit0 as usize] += 1;
        }
        if bit0 == 1 {
            let c1 = ctx::single_p2(self.above_nb(b), self.left_nb(b));
            let bit1 = u32::from(b.refs[0] == geom::ALTREF_FRAME);
            self.put(bit1, u32::from(s.fc.single_ref[c1][1]));
            if self.collect() {
                self.f.counts.single_ref[c1][1][bit1 as usize] += 1;
            }
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
            2 => self.put(0, u32::from(p[0])),
            0 => {
                self.put(1, u32::from(p[0]));
                self.put(0, u32::from(p[1]));
            }
            1 => {
                self.put(1, u32::from(p[0]));
                self.put(1, u32::from(p[1]));
                self.put(0, u32::from(p[2]));
            }
            _ => {
                self.put(1, u32::from(p[0]));
                self.put(1, u32::from(p[1]));
                self.put(1, u32::from(p[2]));
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
        self.put_tree(&prob::JOINT_TREE, &s.fc.mv.joints, joint);
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
        self.put(u32::from(sign), u32::from(c.sign));
        self.put_tree(&prob::CLASS_TREE, &c.classes, class as u8);
        if class == 0 {
            self.put(d, u32::from(c.class0));
            self.put_tree(&prob::FP_TREE, &c.class0_fp[d as usize], f as u8);
            if use_hp {
                self.put(e, u32::from(c.class0_hp));
            }
        } else {
            for i in 0..class {
                self.put((d >> i) & 1, u32::from(c.bits[i]));
            }
            self.put_tree(&prob::FP_TREE, &c.fp, f as u8);
            if use_hp {
                self.put(e, u32::from(c.hp));
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
        let frame = b.refs[0];
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
                self.put(1, u32::from(p[0]));
            }
            if v == 0 {
                self.put(0, u32::from(p[1]));
                if collect {
                    self.f.counts.coef[ti][pi][ri][band][ctx][0] += 1;
                }
                self.cache[rc] = 0;
                after_zero = true;
            } else {
                self.put(1, u32::from(p[1]));
                let a = v.unsigned_abs();
                if a == 1 {
                    self.put(0, u32::from(p[2]));
                    if collect {
                        self.f.counts.coef[ti][pi][ri][band][ctx][1] += 1;
                    }
                    self.cache[rc] = 1;
                } else {
                    self.put(1, u32::from(p[2]));
                    if collect {
                        self.f.counts.coef[ti][pi][ri][band][ctx][2] += 1;
                    }
                    let pp = &PARETO8[usize::from(p[2]) - 1];
                    self.cache[rc] = if self.trial {
                        let mut c = Cost(0);
                        let e = write_big(&mut c, pp, a);
                        self.cost += u64::from(c.0);
                        e
                    } else {
                        write_big(self.w, pp, a)
                    };
                }
                self.put(u32::from(v < 0), 128);
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
            self.put(0, u32::from(probs[band][ctx][0]));
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

/// Where token bits go: the bool coder, or a running cost.
trait Sink {
    fn write(&mut self, bit: u32, prob: u32);
}

impl Sink for BoolWriter {
    #[inline]
    fn write(&mut self, bit: u32, prob: u32) {
        BoolWriter::write(self, bit, prob);
    }
}

/// A cost in 1/256 bits.
struct Cost(u32);

impl Sink for Cost {
    #[inline]
    fn write(&mut self, bit: u32, prob: u32) {
        self.0 += bit_cost(bit != 0, prob as u8);
    }
}

/// A symbol of a tree, as `prob::write_tree` writes it.
fn tree_to<S: Sink>(w: &mut S, tree: &[i8], probs: &[u8], symbol: u8) {
    let reaches = |mut node: i16| -> bool {
        // Depth-first: whether the subtree at `node` holds `symbol`.
        let mut stack = [0i16; 16];
        let mut n = 0;
        loop {
            if node <= 0 {
                if -node == i16::from(symbol) {
                    return true;
                }
            } else {
                let at = node as usize;
                stack[n] = i16::from(tree[at + 1]);
                n += 1;
                node = i16::from(tree[at]);
                continue;
            }
            if n == 0 {
                return false;
            }
            n -= 1;
            node = stack[n];
        }
    };
    let mut index = 0i16;
    loop {
        let at = index as usize;
        let left = i16::from(tree[at]);
        let right = i16::from(tree[at + 1]);
        let bit = u32::from(!reaches(left));
        w.write(bit, u32::from(probs[at >> 1]));
        let next = if bit == 0 { left } else { right };
        if next <= 0 {
            return;
        }
        index = next;
    }
}

/// Writes a token of magnitude `a` of 2 and more; returns its energy.
fn write_big<S: Sink>(w: &mut S, pp: &[u8; 8], a: u32) -> u8 {
    let bits = |w: &mut S, v: u32, probs: &[u8]| {
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

pub mod stats {
    use std::sync::atomic::{AtomicU64, Ordering};
    static C: [AtomicU64; 32] = [const { AtomicU64::new(0) }; 32];
    pub fn add(v: &[(usize, u64)]) {
        for &(i, n) in v {
            C[i].fetch_add(n, Ordering::Relaxed);
        }
    }
    pub fn dump() {
        let g = |i: usize| C[i].load(Ordering::Relaxed);
        eprintln!("bits mode {} coef {}; modes {:?}; skip {}/{}; sizes {:?}", g(0), g(1), (2..16).map(g).collect::<Vec<_>>(), g(17), g(16) + g(17), (18..31).map(g).collect::<Vec<_>>());
    }
}

/// The sum of absolute 4×4 Hadamard coefficients of `src - pred` over an
/// `w`×`h` block (`pred` rows `w` apart).
fn satd(src: &[u8], ss: usize, pred: &[u8], w: usize, h: usize) -> u64 {
    let mut total = 0u64;
    for by in (0..h).step_by(4) {
        for bx in (0..w).step_by(4) {
            let mut d = [0i32; 16];
            for r in 0..4 {
                for c in 0..4 {
                    d[r * 4 + c] = i32::from(src[(by + r) * ss + bx + c]) - i32::from(pred[(by + r) * w + bx + c]);
                }
            }
            for r in 0..4 {
                let (a, b, c, e) = (d[r * 4], d[r * 4 + 1], d[r * 4 + 2], d[r * 4 + 3]);
                let (s0, s1, d0, d1) = (a + b, c + e, a - b, c - e);
                d[r * 4] = s0 + s1;
                d[r * 4 + 1] = s0 - s1;
                d[r * 4 + 2] = d0 + d1;
                d[r * 4 + 3] = d0 - d1;
            }
            for c in 0..4 {
                let (a, b, cc, e) = (d[c], d[4 + c], d[8 + c], d[12 + c]);
                let (s0, s1, d0, d1) = (a + b, cc + e, a - b, cc - e);
                total += u64::from((s0 + s1).unsigned_abs() + (s0 - s1).unsigned_abs() + (d0 + d1).unsigned_abs() + (d0 - d1).unsigned_abs());
            }
        }
    }
    total / 2
}

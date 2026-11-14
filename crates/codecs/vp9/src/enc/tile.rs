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

use super::probs;
use crate::dec::ctx::{self, Nb};
use crate::dec::frame::{Mv, MvRef};
use crate::dec::parse::{LfMi, State};
use crate::dec::recon::Recon;
use crate::dec::syntax::BlockRec;
use crate::dsp::{ipred, sad};
use crate::geom::{
    self, BLOCK_4X4, BLOCK_8X8, BLOCK_64X64, INTRA_FRAME, LAST_FRAME, NEAR, NEAREST, NEW,
    PARTITION_HORZ, PARTITION_NONE, PARTITION_SPLIT, PARTITION_VERT, TX_32X32, ZERO,
};
use crate::header::Header;
use crate::lpf;
use crate::prob::{self, FrameContext};
use crate::tables::{BAND_4X4, BAND_8X8_PLUS, CAT6_PROB, KF_UV_MODE_PROB, KF_Y_MODE_PROB, PARETO8};

use super::tx::{forward, fwht};

const MODE_COUNTER: [i32; 14] = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 3, 1];
const COUNTER_CTX: [u8; 19] = [2, 3, 4, 1, 3, 9, 0, 9, 9, 5, 5, 9, 5, 9, 9, 9, 9, 9, 6];
const MV_REF: [[[i8; 2]; 8]; 13] = {
    const N8: [[i8; 2]; 8] = [
        [-1, 0],
        [0, -1],
        [-1, -1],
        [-2, 0],
        [0, -2],
        [-2, -1],
        [-1, -2],
        [-2, -2],
    ];
    [
        N8,
        N8,
        N8,
        N8,
        [
            [0, -1],
            [-1, 0],
            [1, -1],
            [-1, -1],
            [0, -2],
            [-2, 0],
            [-2, -1],
            [-1, -2],
        ],
        [
            [-1, 0],
            [0, -1],
            [-1, 1],
            [-1, -1],
            [-2, 0],
            [0, -2],
            [-1, -2],
            [-2, -1],
        ],
        [
            [-1, 0],
            [0, -1],
            [-1, 1],
            [1, -1],
            [-1, -1],
            [-3, 0],
            [0, -3],
            [-3, -3],
        ],
        [
            [0, -1],
            [-1, 0],
            [2, -1],
            [-1, -1],
            [-1, 1],
            [0, -3],
            [-3, 0],
            [-3, -3],
        ],
        [
            [-1, 0],
            [0, -1],
            [-1, 2],
            [-1, -1],
            [1, -1],
            [-3, 0],
            [0, -3],
            [-3, -3],
        ],
        [
            [-1, 1],
            [1, -1],
            [-1, 2],
            [2, -1],
            [-1, -1],
            [-3, 0],
            [0, -3],
            [-3, -3],
        ],
        [
            [0, -1],
            [-1, 0],
            [4, -1],
            [-1, 2],
            [-1, -1],
            [0, -3],
            [-3, 0],
            [2, -1],
        ],
        [
            [-1, 0],
            [0, -1],
            [-1, 4],
            [2, -1],
            [-1, -1],
            [-3, 0],
            [0, -3],
            [-1, 2],
        ],
        [
            [-1, 3],
            [3, -1],
            [-1, 4],
            [4, -1],
            [-1, -1],
            [-1, 0],
            [0, -1],
            [-1, 6],
        ],
    ]
};
const MI_WH: [(usize, usize); 13] = [
    (1, 1),
    (1, 1),
    (1, 1),
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (2, 4),
    (4, 2),
    (4, 4),
    (4, 8),
    (8, 4),
    (8, 8),
];

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
    u32::from(if bit {
        c[256 - usize::from(p)]
    } else {
        c[usize::from(p)]
    })
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
    /// The quantizer steps of luma and chroma, DC and AC, by segment.
    pub dq: [[[i32; 2]; 2]; 8],
    /// The segment of every 8×8 block of the frame (empty: segmentation
    /// off).
    pub seg_map: &'a [u8],
    /// Search effort: higher is faster.
    pub speed: u8,
    /// Whether each inter block codes its filter.
    pub switchable: bool,
    /// The fast path's search, made ahead for the whole frame (inter
    /// frames at speed 7 and up), by superblock in raster order.
    pub prepass: Option<&'a [Pre]>,
    /// Which references (LAST, GOLDEN, ALTREF) are worth searching: those
    /// that hold a picture none before them holds.
    pub refs_live: [bool; 3],
    /// Token costs (1/256 bit) by tx size, plane type, inter, band and
    /// context, from the frame's probabilities: see [`token_costs`].
    pub tok: &'a TokCosts,
}

/// The context energy of a level (libvpx `vp9_pt_energy_class`): 0, 1, 2,
/// 3 for 3 and 4, 4 for 5 to 10, and 5 past.
const ENERGY: [u8; 12] = [0, 1, 2, 3, 3, 4, 4, 4, 4, 4, 4, 5];

/// [`Shared::tok`]: more, end of block, zero, one, two or more (signs
/// added for the last two).
pub type TokCosts = [[[[[[u32; 5]; 6]; 6]; 2]; 2]; 4];

/// The token costs of a frame's coefficient probabilities, made once a
/// frame rather than once a transform block.
pub fn token_costs(fc: &FrameContext) -> Box<TokCosts> {
    let cost = &*COST;
    let bc = |bit: bool, p: u8| {
        u32::from(if bit {
            cost[256 - usize::from(p.max(1))]
        } else {
            cost[usize::from(p.max(1))]
        })
    };
    let mut t: Box<TokCosts> = Box::new([[[[[[0; 5]; 6]; 6]; 2]; 2]; 4]);
    for (tx, a) in t.iter_mut().enumerate() {
        for (pl, b) in a.iter_mut().enumerate() {
            for (inter, c) in b.iter_mut().enumerate() {
                for (band, d) in c.iter_mut().enumerate() {
                    for (ctx, e) in d.iter_mut().enumerate() {
                        let p = &fc.coef[tx][pl][inter][band][ctx];
                        *e = [
                            bc(true, p[0]),
                            bc(false, p[0]),
                            bc(false, p[1]),
                            bc(true, p[1]) + bc(false, p[2]) + 256,
                            bc(true, p[1]) + bc(true, p[2]) + 256,
                        ];
                    }
                }
            }
        }
    }
    t
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
    at: (usize, usize, usize, usize),
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
            at: (0, 0, 0, 0),
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
    /// The vector of the second reference, in compound prediction.
    mv2: Mv,
    /// The segment.
    seg: u8,
    /// The vectors of the sub-blocks of an inter block under 8×8, in
    /// raster order (a sub-block of two 4×4 cells repeats its vector).
    smv: [Mv; 4],
    /// The inter prediction filter (EIGHTTAP, SMOOTH, SHARP).
    interp: u8,
    /// The four 4×4 luma modes of a block under 8×8 (`AUTO`: chosen as the
    /// block is coded).
    bmode: [u8; 4],
}

/// A sub-block mode to choose while coding.
const AUTO: u8 = 255;

/// Which sub-block of a neighbour under 8×8 gives sub-block `i`'s
/// candidate: `[i][from above]` (libvpx `idx_n_column_to_subblock`).
const SUBBLOCK: [[u8; 2]; 4] = [[1, 2], [1, 3], [3, 2], [3, 3]];

/// Speeds at and below which inter frames try blocks under 8×8.
const SUB8_SPEED: u8 = 1;

/// From this speed the search's trials take each coefficient's dead-zone
/// level (costed as RD quantization costs it, with its end of block
/// chosen); the blocks the search chooses are coded with RD quantization.
const DZ_TRIAL_SPEED: u8 = 6;

/// The 4×4 steps of a block under 8×8's sub-blocks across and down.
fn sub_steps(bsize: u8) -> (usize, usize) {
    match bsize {
        geom::BLOCK_4X8 => (1, 2),
        geom::BLOCK_8X4 => (2, 1),
        _ => (1, 1),
    }
}

/// Whether the 4×4 column `c` of a block has samples on its left.
fn have_left_of(c: usize, b: &Blk, col0: usize) -> bool {
    c > 0 || b.col > col0
}

/// The scratch a tile's rows lend one another: made once, not for every
/// superblock row.
#[derive(Default)]
pub struct TileBufs {
    pred: Vec<u8>,
    res: Vec<i32>,
    fcoef: Vec<f32>,
    snaps: Vec<Snap>,
    toks: Vec<Tok>,
    pool: Vec<Vec<i32>>,
    mctmp: Vec<u8>,
    rdq: Rdq,
}

/// The per-position scratch of [`Tile::rd_levels`], kept between calls:
/// a call writes every entry it reads, so it is never cleared.
#[derive(Default)]
pub struct Rdq {
    av: Vec<f32>,
    a_nat: Vec<f32>,
    j_pref: Vec<f32>,
    d_tail: Vec<f32>,
    eob_cost: Vec<f32>,
    energy: Vec<u8>,
}

/// The encoder of one tile.
pub struct Tile<'s, 't, 'r, 'v> {
    s: &'s Shared<'s>,
    f: &'t mut State,
    w: &'t mut Out,
    rec: &'t mut Recon<'r, 'v, u8>,
    col0: usize,
    col1: usize,
    left: [Nb; 8],
    left_part: [u8; 8],
    left_nz: [[u8; 16]; 3],
    cache: [u8; 1024],
    /// The segment of the block being coded (its quantizer).
    seg: usize,
    pred: Vec<u8>,
    res: Vec<i32>,
    fcoef: Vec<f32>,
    /// Best vectors found per 8×8 position this superblock, for the
    /// next searches' starting points.
    seeds: Vec<Mv>,
    /// The decision for each square block of the superblock (see
    /// [`Tile::slot`]) and which squares split.
    choice: [Blk; 85],
    /// Each square's partition (by [`Tile::node`]), and the halves of
    /// those coded across (0) or down (1).
    part: [u8; 21],
    rect: [[[Blk; 2]; 21]; 2],
    /// Snapshots: 0 the superblock, then four per level.
    snaps: Vec<Snap>,
    toks: Vec<Tok>,
    /// The cost of the best inter candidate last ranked (SAD units).
    last_sad: u64,
    /// The SATD (and mode rate) of the best intra mode last ranked.
    last_satd: u64,
    /// The fast path's search result for each square of the superblock.
    found: [(Mv, u64); 85],
    /// Spent level buffers, to use again.
    pool: Vec<Vec<i32>>,
    /// The 4×4 modes the block coded last chose.
    sub: [u8; 4],
    /// How each 8×8 square of the superblock is divided: `PARTITION_NONE`,
    /// or the partition of its block under 8×8 (four 4×4, two 8×4 or two
    /// 4×8).
    sub8: [u8; 64],
    /// Scratch of the 8-tap filters across.
    mctmp: Vec<u8>,
    rdq: Rdq,
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
        w: &'t mut Out,
        rec: &'t mut Recon<'r, 'v, u8>,
        (col0, col1): (usize, usize),
        mut seeds: Vec<Mv>,
        bufs: TileBufs,
    ) -> Self {
        seeds.resize(s.mi_cols * s.mi_rows, Mv::ZERO);
        let TileBufs {
            mut pred,
            mut res,
            mut fcoef,
            mut snaps,
            toks,
            pool,
            mut mctmp,
            mut rdq,
        } = bufs;
        pred.resize(64 * 64, 0);
        res.resize(64 * 64, 0);
        fcoef.resize(32 * 32, 0.0);
        snaps.resize_with(17, Snap::default);
        mctmp.resize(cutlass_simd::vp9::mc::TMP, 0);
        for v in [
            &mut rdq.av,
            &mut rdq.a_nat,
            &mut rdq.j_pref,
            &mut rdq.d_tail,
            &mut rdq.eob_cost,
        ] {
            v.resize(1025, 0.0);
        }
        rdq.energy.resize(1024, 0);
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
            seg: 0,
            pred,
            res,
            fcoef,
            seeds,
            choice: [Blk::default(); 85],
            part: [PARTITION_NONE; 21],
            rect: [[[Blk::default(); 2]; 21]; 2],
            snaps,
            toks,
            last_sad: 0,
            last_satd: u64::MAX,
            found: [(Mv::ZERO, u64::MAX); 85],
            pool,
            sub: [geom::DC; 4],
            sub8: [PARTITION_NONE; 64],
            mctmp,
            rdq,
            cur_ref: 0,
            coded_skip: false,
            choice_skip: (false, 0.0),
            last_skip: (false, 0.0),
            rd: TxRd::default(),
            trial: false,
            cost: 0,
        }
    }

    /// The vectors found, for the next row's tile, and the buffers to
    /// lend it.
    pub fn finish(self) -> (Vec<Mv>, TileBufs) {
        (
            self.seeds,
            TileBufs {
                pred: self.pred,
                res: self.res,
                fcoef: self.fcoef,
                snaps: self.snaps,
                toks: self.toks,
                pool: self.pool,
                mctmp: self.mctmp,
                rdq: self.rdq,
            },
        )
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
                match self.s.prepass {
                    Some(pre) => {
                        let p = pre[(mi_row / 8) * self.s.mi_cols.div_ceil(8) + mi_col / 8];
                        split = p.split;
                        self.found = p.found;
                    }
                    None => {
                        self.choose_into(mi_row, mi_col, 3, &mut split);
                    }
                }
                let part = split.map(|s| if s { PARTITION_SPLIT } else { PARTITION_NONE });
                self.partition(mi_row, mi_col, BLOCK_64X64, 4, &part);
                mi_col += 8;
                continue;
            }
            self.save(mi_row, mi_col, (8, 8), 0);
            self.trial = true;
            self.part = [PARTITION_NONE; 21];
            self.rd_node(mi_row, mi_col, 3);
            self.restore(0);
            self.trial = false;
            let part = self.part;
            self.partition(mi_row, mi_col, BLOCK_64X64, 4, &part);
            mi_col += 8;
        }
    }

    // Partition decision --------------------------------------------------

    /// Whether this frame takes the one-pass path: inter frames at speed
    /// 7 and up, every frame at 9.
    fn fast(&self) -> bool {
        self.s.speed >= 7 && (!self.s.intra_frame || self.s.speed >= 9)
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
        let own = if whole {
            self.estimate(row, col, SQUARE[level])
        } else {
            u64::MAX
        };
        // Splitting costs the partition symbols and three more blocks'
        // mode info.
        let mut quarters = u64::from(s.lambda) * 12;
        for (dr, dc) in [(0, 0), (0, hbs), (hbs, 0), (hbs, hbs)] {
            quarters =
                quarters.saturating_add(self.choose_into(row + dr, col + dc, level - 1, split));
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
        if s.intra_frame {
            // Key frames: the texture the block's mean leaves, by SATD.
            let n = bw * 8;
            let (x, y) = (col * 8, row * 8);
            let (src, ss) = (s.src.planes[0], s.src.strides[0]);
            let mut sum = 0u32;
            for r in 0..n {
                for c in 0..n {
                    sum += u32::from(src[(y + r) * ss + x + c]);
                }
            }
            let dc = ((sum + (n * n / 2) as u32) / (n * n) as u32) as u8;
            self.pred[..n * n].fill(dc);
            return satd(&src[y * ss + x..], ss, &self.pred, n, n) + lam * 4;
        }
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
        } else if self.s.speed >= 10 {
            // Fastest: intra only when its SATD is clearly under the inter
            // prediction's SAD.
            usize::from(self.last_satd.saturating_mul(5) < self.last_sad.saturating_mul(4))
        } else {
            let lam = f64::from(self.s.rd_lambda) / 256.0;
            let n = MI_WH[usize::from(bsize)].0;
            self.trial = true;
            self.save(row, col, (n, n), 1);
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
        // Slots: the state before, after the best whole shape so far, and
        // two for the blocks' own trials.
        let before = 1 + 4 * (3 - level);
        self.save(row, col, (n, n), before);
        let mut best = (f64::INFINITY, PARTITION_NONE);
        if whole {
            let c0 = self.cost;
            self.part_symbol(row, col, bsize, PARTITION_NONE);
            let sym = (self.cost - c0) as f64 * lam;
            let (j, b) = self.best_block(row, col, bsize, before + 2);
            self.choice[Self::slot(row, col, bsize)] = b;
            self.part_after(row, col, bsize, PARTITION_NONE);
            best = (sym + j, PARTITION_NONE);
        }
        if level == 0 {
            let i8 = (row & 7) * 8 + (col & 7);
            self.sub8[i8] = PARTITION_NONE;
            // Key frames: four 4×4 blocks instead.
            if s.intra_frame && s.speed <= 6 && whole {
                self.save(row, col, (1, 1), before + 1);
                self.restore(before);
                let c0 = self.cost;
                self.part_symbol(row, col, bsize, PARTITION_SPLIT);
                let base = self.blk(row, col, BLOCK_4X4);
                let b = Blk {
                    tx: 0,
                    bmode: [AUTO; 4],
                    uv: self.choice[Self::slot(row, col, bsize)].uv,
                    ..base
                };
                let d = self.code_block(&b);
                self.part_after(row, col, bsize, PARTITION_SPLIT);
                let j = d + (self.cost - c0) as f64 * lam;
                if j < best.0 {
                    let mut chosen = b;
                    chosen.bmode = self.sub;
                    self.choice[Self::slot(row, col, bsize)] = chosen;
                    self.sub8[i8] = PARTITION_SPLIT;
                    return j;
                }
                self.restore(before + 1);
            }
            // Inter frames: the best whole block's reference over 4×4,
            // 8×4 or 4×8 sub-blocks, each with its own vector.
            let whole_b = self.choice[Self::slot(row, col, bsize)];
            if !s.intra_frame
                && s.speed <= SUB8_SPEED
                && whole
                && whole_b.refs[0] != INTRA_FRAME
                && whole_b.refs[1] == INTRA_FRAME
            {
                self.save(row, col, (1, 1), before + 1);
                let mut won = None;
                for (p, sbs) in [
                    (PARTITION_SPLIT, BLOCK_4X4),
                    (PARTITION_HORZ, geom::BLOCK_8X4),
                    (PARTITION_VERT, geom::BLOCK_4X8),
                ] {
                    self.restore(before);
                    let b = self.sub_inter(&whole_b, sbs);
                    let c0 = self.cost;
                    self.part_symbol(row, col, bsize, p);
                    let d = self.code_block(&b);
                    self.part_after(row, col, bsize, p);
                    let j = d + (self.cost - c0) as f64 * lam;
                    if j < best.0 {
                        best = (j, p);
                        won = Some((p, b));
                        self.save(row, col, (1, 1), before + 1);
                    }
                }
                self.restore(before + 1);
                if let Some((p, b)) = won {
                    self.choice[Self::slot(row, col, bsize)] = b;
                    self.sub8[i8] = p;
                }
            }
            return best.0;
        }
        let none_skip = self.last_skip;
        if whole {
            self.save(row, col, (n, n), before + 1);
        }
        // Halves, across and down.
        if whole && s.speed < 4 {
            let node = Self::node(row, col, level);
            for (p, k, offs) in [
                (PARTITION_HORZ, 0usize, [(0, 0), (hbs, 0)]),
                (PARTITION_VERT, 1, [(0, 0), (0, hbs)]),
            ] {
                self.restore(before);
                let sub = geom::subsize(p, bsize);
                let c0 = self.cost;
                self.part_symbol(row, col, bsize, p);
                let mut j = (self.cost - c0) as f64 * lam;
                let mut picks = [Blk::default(); 2];
                for (i, (dr, dc)) in offs.into_iter().enumerate() {
                    if j >= best.0 {
                        break;
                    }
                    let (jb, b) = self.best_block(row + dr, col + dc, sub, before + 2);
                    j += jb;
                    picks[i] = b;
                }
                if j < best.0 {
                    self.part_after(row, col, bsize, p);
                    best = (j, p);
                    self.rect[k][node] = picks;
                    self.save(row, col, (n, n), before + 1);
                }
            }
        }
        // No split when the whole block needs no residual and its error is
        // under the quantizer's own noise.
        if whole && best.1 == PARTITION_NONE && none_skip.0 {
            let pixels = (n * 8 * n * 8) as f64;
            let step = f64::from(s.dq[0][0][1]) / 8.0;
            if none_skip.1 < pixels * step * step / 24.0 {
                self.restore(before + 1);
                self.part[Self::node(row, col, level)] = PARTITION_NONE;
                return best.0;
            }
        }
        self.restore(before);
        let c0 = self.cost;
        self.part_symbol(row, col, bsize, PARTITION_SPLIT);
        let mut j = (self.cost - c0) as f64 * lam;
        for (dr, dc) in [(0, 0), (0, hbs), (hbs, 0), (hbs, hbs)] {
            if j >= best.0 {
                break;
            }
            j += self.rd_node(row + dr, col + dc, level - 1);
        }
        let node = Self::node(row, col, level);
        if j < best.0 {
            self.part[node] = PARTITION_SPLIT;
            j
        } else {
            self.restore(before + 1);
            self.part[node] = best.1;
            best.0
        }
    }

    /// Codes the best of the block's candidate modes (leaving the state
    /// as it leaves it) and returns its cost. `slot` holds the state
    /// before it.
    fn best_block(&mut self, row: usize, col: usize, bsize: u8, slot: usize) -> (f64, Blk) {
        let s = self.s;
        let lam = f64::from(s.rd_lambda) / 256.0;
        let (bw_mi, bh_mi) = MI_WH[usize::from(bsize)];
        self.save(row, col, (bw_mi, bh_mi), slot);
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
        let try_one =
            |t: &mut Self, i: usize, b: &Blk, best: &mut (f64, usize), last: &mut usize| {
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
                    t.save(row, col, (bw_mi, bh_mi), slot + 1);
                }
            };
        for (i, b) in cands.clone().iter().enumerate() {
            try_one(self, i, b, &mut best, &mut last);
        }
        // Chroma and transform size for the best intra mode.
        if best.1 >= intra_first
            && cands.len() > intra_first
            && (s.speed < 6 || (s.intra_frame && s.speed < 9))
        {
            let b0 = cands[best.1];
            let mut more = Vec::new();
            for uv in [b0.mode, geom::TM, geom::V, geom::H] {
                if uv != b0.uv
                    && !more.iter().any(|m: &Blk| m.uv == uv)
                    && more.len() < if s.speed < 4 { 3 } else { 1 }
                {
                    more.push(Blk { uv, ..b0 });
                }
            }
            for b in more {
                let i = cands.len();
                cands.push(b);
                try_one(self, i, &b, &mut best, &mut last);
            }
        }
        if s.speed < 6 || (s.intra_frame && s.speed < 9) {
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
        // The other filters for the best inter candidate.
        if s.switchable && cands[best.1].refs[0] != INTRA_FRAME {
            let b0 = cands[best.1];
            for f in [geom::SMOOTH, geom::SHARP] {
                let i = cands.len();
                let b = Blk { interp: f, ..b0 };
                cands.push(b);
                try_one(self, i, &b, &mut best, &mut last);
            }
        }
        if last != best.1 {
            self.restore(slot + 1);
        }
        self.last_skip = self.choice_skip;
        (best.0, cands[best.1])
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
            tx: geom::max_tx(bsize)
                .min(geom::tx_mode_biggest(s.h.tx_mode))
                .min(if s.speed >= 10 {
                    geom::TX_16X16
                } else {
                    TX_32X32
                }),
            mode: geom::DC,
            uv: geom::DC,
            refs: [INTRA_FRAME, INTRA_FRAME],
            mv: Mv::ZERO,
            mv2: Mv::ZERO,
            seg: s.seg_map.get(row * s.mi_cols + col).copied().unwrap_or(0),
            smv: [Mv::ZERO; 4],
            interp: geom::EIGHTTAP,
            bmode: [geom::DC; 4],
        }
    }

    /// The 8×8 block `whole` as an inter block of size `bsize` under 8×8
    /// on the same reference and filter: each sub-block's vector by a
    /// small search from the whole block's, coded new (or zero).
    fn sub_inter(&mut self, whole: &Blk, bsize: u8) -> Blk {
        let s = self.s;
        let mut b = Blk {
            bsize,
            tx: 0,
            refs: whole.refs,
            interp: whole.interp,
            ..self.blk(whole.row, whole.col, bsize)
        };
        let best = self.find_mv(&b, NEW);
        let hp_ok = s.h.allow_hp && best.row.unsigned_abs() < 64 && best.col.unsigned_abs() < 64;
        let (sw, sh) = sub_steps(bsize);
        let (x0, y0) = (whole.col * 8, whole.row * 8);
        self.cur_ref = usize::from(whole.refs[0]) - 1;
        for idy in (0..2).step_by(sh) {
            for idx in (0..2).step_by(sw) {
                let j = idy * 2 + idx;
                let (x, y) = (x0 + 4 * idx, y0 + 4 * idy);
                let (bw4, bh4) = (4 * sw, 4 * sh);
                let new = self.sub_search(x, y, (bw4, bh4), whole.mv, best, hp_ok);
                // The sub-block's own candidates, from the ones before it;
                // a mode costs about a bit, a new vector its rate too.
                let lam = u64::from(s.lambda);
                let nearest = self.sub_candidate(&b, NEAREST, j);
                let near = self.sub_candidate(&b, NEAR, j);
                let new_rate = u64::from(mv_bits(Mv {
                    row: new.row - best.row,
                    col: new.col - best.col,
                })) + 512;
                let mut pick = (u64::MAX, ZERO, Mv::ZERO);
                for (m, mv, rate) in [
                    (NEAREST, nearest, 256),
                    (NEAR, near, 384),
                    (ZERO, Mv::ZERO, 256),
                    (NEW, new, new_rate),
                ] {
                    if let Some(d) = self.sad_sub(x, y, bw4, bh4, mv) {
                        let c = d + lam * rate / 256;
                        if c < pick.0 {
                            pick = (c, m, mv);
                        }
                    }
                }
                let (_, mode, mv) = pick;
                b.smv[j] = mv;
                b.bmode[j] = mode;
                if sw == 2 {
                    b.smv[j + 1] = mv;
                    b.bmode[j + 1] = b.bmode[j];
                }
                if sh == 2 {
                    b.smv[j + 2] = mv;
                    b.bmode[j + 2] = b.bmode[j];
                }
            }
        }
        self.cur_ref = 0;
        b.mv = b.smv[3];
        b.mode = b.bmode[3];
        b
    }

    /// A sub-block's vector: from `seed`, whole samples two around, then
    /// halves, quarters and (where `hp`) eighths, by SAD and the vector's
    /// rate against `best`.
    fn sub_search(
        &self,
        x: usize,
        y: usize,
        (w, h): (usize, usize),
        seed: Mv,
        best: Mv,
        hp: bool,
    ) -> Mv {
        let lam = u64::from(self.s.lambda);
        let cost = |mv: Mv| -> Option<u64> {
            let d = self.sad_sub(x, y, w, h, mv)?;
            let rate = if mv == Mv::ZERO {
                0
            } else {
                u64::from(mv_bits(Mv {
                    row: mv.row - best.row,
                    col: mv.col - best.col,
                })) + 512
            };
            Some(d + lam * rate / 256)
        };
        let even = |v: i16| if v & 1 != 0 { v - v.signum() } else { v };
        let fix = |m: Mv| {
            if hp {
                m
            } else {
                Mv {
                    row: even(m.row),
                    col: even(m.col),
                }
            }
        };
        let mut at = fix(seed);
        let mut bc = cost(at).unwrap_or(u64::MAX);
        for c in [Mv::ZERO, fix(best)] {
            if let Some(v) = cost(c)
                && v < bc
            {
                (at, bc) = (c, v);
            }
        }
        // Steps in eighths: whole samples two around, then one half, one
        // quarter and (where coded) one eighth.
        let steps: &[(i16, i16)] = if hp {
            &[(8, 2), (4, 1), (2, 1), (1, 1)]
        } else {
            &[(8, 2), (4, 1), (2, 1)]
        };
        for &(st, reach) in steps {
            let center = at;
            for dr in -reach..=reach {
                for dc in -reach..=reach {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let m = Mv {
                        row: center.row + dr * st,
                        col: center.col + dc * st,
                    };
                    if let Some(v) = cost(m)
                        && v < bc
                    {
                        (at, bc) = (m, v);
                    }
                }
            }
        }
        at
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
        let e = self
            .rec
            .intra_edges(0, tx, x, y, have_left, have_above, false);
        let (src, ss) = (s.src.planes[0], s.src.strides[0]);
        let modes: &[u8] = if s.speed >= 8 {
            &[geom::DC, geom::V, geom::H, geom::TM]
        } else {
            &[
                geom::DC,
                geom::V,
                geom::H,
                geom::TM,
                geom::D45,
                geom::D135,
                geom::D117,
                geom::D153,
                geom::D207,
                geom::D63,
            ]
        };
        let mut scored: Vec<(u64, u8)> = Vec::with_capacity(10);
        let lam = u64::from(s.lambda);
        for &m in modes {
            ipred::predict(m, n, &mut self.pred, n, &e, 8);
            let cost =
                satd(&src[y * ss + x..], ss, &self.pred, n, n) + lam * u64::from(m != geom::DC) * 2;
            scored.push((cost, m));
        }
        scored.sort_unstable();
        self.last_satd = scored.first().map_or(u64::MAX, |t| t.0);
        let k = match s.speed {
            0..=3 => 4,
            4..=5 => 3,
            6 => 2,
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
        let mut searched = [None; 3];
        // `r` also names the reference the search reads.
        #[allow(clippy::needless_range_loop)]
        for r in 0..3 {
            if !s.refs_live[r] {
                continue;
            }
            self.cur_ref = r;
            if self.rec_ref().is_none() {
                continue;
            }
            let (mv, sad) = if r == 0 && self.fast() {
                self.found[Self::slot(row, col, bsize)]
            } else if r > 0 && s.speed >= 9 {
                // Other references: only their candidate vectors, below.
                (Mv::ZERO, u64::MAX / 4)
            } else {
                self.search(row, col, bsize)
            };
            if r == 0 {
                let (bw, _) = MI_WH[usize::from(bsize)];
                for rr in 0..bw.min(s.mi_rows - row) {
                    for c in 0..bw.min(s.mi_cols - col) {
                        self.seeds[(row + rr) * s.mi_cols + col + c] = mv;
                    }
                }
            }
            let (mv, sad) = if sad < u64::MAX / 4 {
                self.refine(row, col, bsize, mv, sad)
            } else {
                (mv, sad)
            };
            let mut b = Blk {
                refs: [r as u8 + 1, INTRA_FRAME],
                ..*base
            };
            let nearest = self.find_mv(&b, NEAREST);
            let near = self.find_mv(&b, NEAR);
            // Eighths code only when the vector a NEW one is coded against
            // allows them.
            let hp_ok =
                s.h.allow_hp && nearest.row.unsigned_abs() < 64 && nearest.col.unsigned_abs() < 64;
            let even = |v: i16| if v & 1 != 0 { v - v.signum() } else { v };
            let mv = if hp_ok {
                mv
            } else {
                Mv {
                    row: even(mv.row),
                    col: even(mv.col),
                }
            };
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
            searched[r] = Some(mv);
            let new_bits = if b.mode == NEW {
                lam * u64::from(mv_bits(Mv {
                    row: mv.row - nearest.row,
                    col: mv.col - nearest.col,
                })) / 256
                    + lam * 2
            } else {
                lam
            };
            scored.push((sad + new_bits + ref_bits, b));
            for v in [nearest, near, Mv::ZERO] {
                if v == mv || scored.iter().any(|(_, o)| o.refs == b.refs && o.mv == v) {
                    continue;
                }
                if let Some(d) = if s.speed <= 5 {
                    self.sad_sub_exact(x, y, w, h, v)
                } else {
                    self.sad_sub(x, y, w, h, v)
                } {
                    scored.push((
                        d + lam + ref_bits,
                        Blk {
                            mv: v,
                            mode: mode_of(v, ZERO),
                            ..b
                        },
                    ));
                }
            }
        }
        self.cur_ref = 0;
        if s.h.reference_mode == geom::REFERENCE_SELECT {
            self.compound_cands(base, (x, y, w, h), &searched, &mut scored);
        }
        scored.sort_by_key(|t| t.0);
        self.last_sad = scored.first().map_or(u64::MAX, |t| t.0);
        let k = match s.speed {
            0..=3 => 3,
            4..=7 => 2,
            _ => 1,
        };
        out.extend(scored.iter().take(k).map(|t| t.1));
    }

    /// Compound candidates: the fixed reference with each variable one,
    /// both at their NEAREST, NEAR or zero vectors, or at the vectors
    /// their own searches found (NEW), by the SAD of the averaged
    /// prediction and their vectors' rate.
    fn compound_cands(
        &mut self,
        base: &Blk,
        (x, y, w, h): (usize, usize, usize, usize),
        searched: &[Option<Mv>; 3],
        scored: &mut Vec<(u64, Blk)>,
    ) {
        let s = self.s;
        let lam = u64::from(s.lambda);
        let fixed = s.h.comp_fixed;
        let at = usize::from(s.h.sign_bias[usize::from(fixed)]);
        for var in s.h.comp_var {
            let mut refs = [INTRA_FRAME; 2];
            refs[at] = fixed;
            refs[at ^ 1] = var;
            if refs.iter().any(|&r| {
                !s.refs_live
                    .get(usize::from(r) - 1)
                    .copied()
                    .unwrap_or(false)
            }) {
                continue;
            }
            let b = Blk { refs, ..*base };
            let of = |t: &Self, m: u8| [t.find_mv_of(&b, m, refs[0]), t.find_mv_of(&b, m, refs[1])];
            let (nearest, near, best) = (of(self, NEAREST), of(self, NEAR), of(self, NEW));
            let found = [
                searched[usize::from(refs[0]) - 1],
                searched[usize::from(refs[1]) - 1],
            ];
            let mut cands = vec![(NEAREST, nearest), (NEAR, near), (ZERO, [Mv::ZERO; 2])];
            if let [Some(a), Some(c)] = found {
                cands.push((NEW, [a, c]));
            }
            for (m, mvs) in cands {
                let mode = if mvs == nearest {
                    NEAREST
                } else if mvs == near {
                    NEAR
                } else if mvs == [Mv::ZERO; 2] {
                    ZERO
                } else {
                    m
                };
                if scored
                    .iter()
                    .any(|(_, o)| o.refs == refs && o.mv == mvs[0] && o.mv2 == mvs[1])
                {
                    continue;
                }
                let Some(d) = self.comp_sad(x, y, w, h, refs, mvs) else {
                    continue;
                };
                let bits = if mode == NEW {
                    let rate = |v: Mv, b: Mv| {
                        u64::from(mv_bits(Mv {
                            row: v.row - b.row,
                            col: v.col - b.col,
                        }))
                    };
                    lam * (rate(mvs[0], best[0]) + rate(mvs[1], best[1])) / 256 + lam * 4
                } else {
                    lam * 2
                };
                scored.push((
                    d + bits + lam * 3,
                    Blk {
                        mode,
                        mv: mvs[0],
                        mv2: mvs[1],
                        ..b
                    },
                ));
            }
        }
    }

    /// The luma SAD of the compound prediction from `refs` at `mvs`
    /// (eighths), with the real filters: the first reference's prediction
    /// averaged with the second's, as the decoder makes it. `None` when a
    /// window leaves its reference or the block is not one the kernels
    /// take.
    fn comp_sad(
        &mut self,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        refs: [u8; 2],
        mvs: [Mv; 2],
    ) -> Option<u64> {
        let s = self.s;
        if !matches!(w, 8 | 16 | 32 | 64) {
            return None;
        }
        let mut pred = std::mem::take(&mut self.pred);
        let mut tmp = std::mem::take(&mut self.mctmp);
        let filter = &crate::dsp::mc::FILTERS[usize::from(geom::EIGHTTAP)];
        let mut ok = true;
        for (k, (&r, mv)) in refs.iter().zip(mvs).enumerate() {
            let c = self.rec.ctx();
            let Some(rp) = c.refs[usize::from(r) - 1].as_ref() else {
                ok = false;
                break;
            };
            let ((data, stride), (rw, rh)) =
                (rp.buf.plane(0), (rp.dims.w[0] as i32, rp.dims.h[0] as i32));
            let (mr, mc) = (i32::from(mv.row), i32::from(mv.col));
            let (bx, by) = (x as i32 + (mc >> 3), y as i32 + (mr >> 3));
            if bx < 3 || by < 3 || bx + w as i32 + 4 > rw || by + h as i32 + 4 > rh {
                ok = false;
                break;
            }
            crate::dsp::mc::predict(
                &mut pred,
                w,
                data,
                by as usize * stride + bx as usize,
                stride,
                (w, h),
                (((mc & 7) * 2) as usize, ((mr & 7) * 2) as usize),
                filter,
                255,
                k == 1,
                &mut tmp,
            );
        }
        let out = ok.then(|| {
            let (src, ss) = (s.src.planes[0], s.src.strides[0]);
            u64::from(sad::sad(&src[y * ss + x..], ss, &pred, w, w, h))
        });
        self.pred = pred;
        self.mctmp = tmp;
        out
    }

    // State snapshots ------------------------------------------------------

    /// Keeps in snapshot `k` everything coding the square of `n` 8×8
    /// blocks at `(row, col)` changes: its contexts, motion and samples.
    fn save(&mut self, row: usize, col: usize, (nw, nh): (usize, usize), k: usize) {
        // In place: taking it out left a new snapshot behind, three
        // cleared buffers allocated and freed on every call.
        let sn = &mut self.snaps[k];
        sn.at = (row, col, nw, nh);
        sn.cost = self.cost;
        let a = col - self.col0;
        let aw = nw.min(self.f.above.len() - a);
        sn.above[..aw].copy_from_slice(&self.f.above[a..a + aw]);
        sn.above_part[..aw].copy_from_slice(&self.f.above_part[a..a + aw]);
        let l = row & 7;
        sn.left[..nh].copy_from_slice(&self.left[l..l + nh]);
        sn.left_part[..nh].copy_from_slice(&self.left_part[l..l + nh]);
        for p in 0..3 {
            let ss = usize::from(p > 0);
            let (a4, n4, h4) = ((a * 2) >> ss, (nw * 2) >> ss, (nh * 2) >> ss);
            let w4 = n4.min(self.f.above_nz[p].len() - a4);
            sn.above_nz[p][..w4].copy_from_slice(&self.f.above_nz[p][a4..a4 + w4]);
            let l4 = ((row * 2) & 15) >> ss;
            sn.left_nz[p][..h4].copy_from_slice(&self.left_nz[p][l4..l4 + h4]);
        }
        let rows = nh.min(self.s.mi_rows - row);
        let cols = nw.min(self.col1 - col);
        for r in 0..rows {
            let at = (row + r) * self.f.width + a;
            sn.mvs[r * 8..r * 8 + cols].copy_from_slice(&self.f.mvs[at..at + cols]);
        }
        let views = self.rec.views();
        #[allow(clippy::needless_range_loop)]
        for p in 0..3 {
            let ss = usize::from(p > 0);
            let (x, y, mw, mh) = (
                (col * 8) >> ss,
                (row * 8) >> ss,
                (nw * 8) >> ss,
                (nh * 8) >> ss,
            );
            let v = &mut views[p];
            let stride = v.stride;
            let src = v.at(y, x);
            for r in 0..mh {
                sn.pix[p][r * 64..r * 64 + mw].copy_from_slice(&src[r * stride..r * stride + mw]);
            }
        }
    }

    /// Puts back what snapshot `k` kept.
    fn restore(&mut self, k: usize) {
        let sn = &self.snaps[k];
        let (row, col, nw, nh) = sn.at;
        self.cost = sn.cost;
        let a = col - self.col0;
        let aw = nw.min(self.f.above.len() - a);
        self.f.above[a..a + aw].copy_from_slice(&sn.above[..aw]);
        self.f.above_part[a..a + aw].copy_from_slice(&sn.above_part[..aw]);
        let l = row & 7;
        self.left[l..l + nh].copy_from_slice(&sn.left[..nh]);
        self.left_part[l..l + nh].copy_from_slice(&sn.left_part[..nh]);
        for p in 0..3 {
            let ss = usize::from(p > 0);
            let (a4, n4, h4) = ((a * 2) >> ss, (nw * 2) >> ss, (nh * 2) >> ss);
            let w4 = n4.min(self.f.above_nz[p].len() - a4);
            self.f.above_nz[p][a4..a4 + w4].copy_from_slice(&sn.above_nz[p][..w4]);
            let l4 = ((row * 2) & 15) >> ss;
            self.left_nz[p][l4..l4 + h4].copy_from_slice(&sn.left_nz[p][..h4]);
        }
        let rows = nh.min(self.s.mi_rows - row);
        let cols = nw.min(self.col1 - col);
        for r in 0..rows {
            let at = (row + r) * self.f.width + a;
            self.f.mvs[at..at + cols].copy_from_slice(&sn.mvs[r * 8..r * 8 + cols]);
        }
        let views = self.rec.views();
        #[allow(clippy::needless_range_loop)]
        for p in 0..3 {
            let ss = usize::from(p > 0);
            let (x, y, mw, mh) = (
                (col * 8) >> ss,
                (row * 8) >> ss,
                (nw * 8) >> ss,
                (nh * 8) >> ss,
            );
            let v = &mut views[p];
            let stride = v.stride;
            let dst = v.at(y, x);
            for r in 0..mh {
                dst[r * stride..r * stride + mw].copy_from_slice(&sn.pix[p][r * 64..r * 64 + mw]);
            }
        }
    }

    // Motion search -------------------------------------------------------

    #[allow(clippy::type_complexity)]
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
        Some(u64::from(sad::sad_bilinear(
            a,
            ss,
            b,
            stride,
            w,
            h,
            ((r & 7) * 2) as u32,
            ((c & 7) * 2) as u32,
        )))
    }

    /// As [`Tile::sad_sub`], with the real 8-tap prediction (needs the
    /// filter's reach inside the reference; else the bilinear proxy).
    fn sad_sub_exact(&mut self, x: usize, y: usize, w: usize, h: usize, mv: Mv) -> Option<u64> {
        let s = self.s;
        let (r, c) = (i32::from(mv.row), i32::from(mv.col));
        let (bx, by) = (x as i32 + (c >> 3), y as i32 + (r >> 3));
        let mut pred = std::mem::take(&mut self.pred);
        let mut tmp = std::mem::take(&mut self.mctmp);
        let out = (|| {
            let ((data, stride), (rw, rh)) = self.rec_ref()?;
            if bx < 3
                || by < 3
                || bx + w as i32 + 4 > rw
                || by + h as i32 + 4 > rh
                || !matches!(w, 8 | 16 | 32 | 64)
            {
                return None;
            }
            let filter = &crate::dsp::mc::FILTERS[usize::from(geom::EIGHTTAP)];
            let at = by as usize * stride + bx as usize;
            crate::dsp::mc::predict(
                &mut pred,
                w,
                data,
                at,
                stride,
                (w, h),
                (((c & 7) * 2) as usize, ((r & 7) * 2) as usize),
                filter,
                255,
                false,
                &mut tmp,
            );
            let (src, ss) = (s.src.planes[0], s.src.strides[0]);
            Some(u64::from(sad::sad(&src[y * ss + x..], ss, &pred, w, w, h)))
        })();
        self.pred = pred;
        self.mctmp = tmp;
        out.or_else(|| self.sad_sub(x, y, w, h, mv))
    }

    /// Searches LAST for the block's vector: the best of the neighbours'
    /// vectors and zero, a diamond search in whole samples, then halves
    /// and quarters. Returns the vector (eighths) and its SAD.
    fn search(&mut self, row: usize, col: usize, bsize: u8) -> (Mv, u64) {
        let Some(rp) = self.rec_ref() else {
            return (Mv::ZERO, u64::MAX / 4);
        };
        search_in(self.s, rp, &self.seeds, self.cur_ref, row, col, bsize)
    }

    /// Refines a whole-sample vector of a block to halves and quarters (and
    /// eighths where they may be coded).
    fn refine(
        &mut self,
        row: usize,
        col: usize,
        bsize: u8,
        mut mv: Mv,
        mut best_sad: u64,
    ) -> (Mv, u64) {
        let s = self.s;
        let (bw, bh) = MI_WH[usize::from(bsize)];
        let (x, y) = (col * 8, row * 8);
        let w = (bw * 8).min((s.mi_cols - col) * 8);
        let h = (bh * 8).min((s.mi_rows - row) * 8);
        if s.speed < 9 {
            let hp = s.h.allow_hp && mv.row.unsigned_abs() < 64 && mv.col.unsigned_abs() < 64;
            let steps: &[i16] = if hp && s.speed < 4 {
                &[4, 2, 1]
            } else {
                &[4, 2]
            };
            for &step in steps {
                let center = mv;
                for (dr, dc) in [
                    (-step, 0),
                    (step, 0),
                    (0, -step),
                    (0, step),
                    (-step, -step),
                    (-step, step),
                    (step, -step),
                    (step, step),
                ] {
                    let m = Mv {
                        row: center.row + dr,
                        col: center.col + dc,
                    };
                    let exact = s.speed <= 5;
                    let v = if exact {
                        self.sad_sub_exact(x, y, w, h, m)
                    } else {
                        self.sad_sub(x, y, w, h, m)
                    };
                    let Some(v) = v else {
                        continue;
                    };
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
    fn partition(&mut self, row: usize, col: usize, bsize: u8, n4: u32, part: &[u8; 21]) {
        let s = self.s;
        if row >= s.mi_rows || col >= s.mi_cols {
            return;
        }
        let n8 = n4 - 1;
        let hbs = (1usize << n8) >> 1;
        let fits = row + hbs < s.mi_rows && col + hbs < s.mi_cols;
        let node = if hbs == 0 {
            0
        } else {
            Self::node(row, col, n8 as usize)
        };
        let p = if hbs == 0 {
            if self.fast() {
                PARTITION_NONE
            } else {
                self.sub8[(row & 7) * 8 + (col & 7)]
            }
        } else if !fits {
            PARTITION_SPLIT
        } else {
            part[node]
        };
        self.part_symbol(row, col, bsize, p);
        let sub = geom::subsize(p, bsize);
        match p {
            PARTITION_NONE => {
                if self.fast() {
                    self.fast_block(row, col, bsize);
                } else {
                    let b = self.choice[Self::slot(row, col, bsize)];
                    self.code_block(&b);
                }
            }
            PARTITION_HORZ | PARTITION_VERT if hbs > 0 => {
                let [a, b] = self.rect[usize::from(p == PARTITION_VERT)][node];
                self.code_block(&a);
                self.code_block(&b);
            }
            _ if hbs == 0 => {
                // A block under 8×8, coded as one.
                let b = self.choice[Self::slot(row, col, bsize)];
                self.code_block(&b);
            }
            _ => {
                for (dr, dc) in [(0, 0), (0, hbs), (hbs, 0), (hbs, hbs)] {
                    self.partition(row + dr, col + dc, sub, n8, part);
                }
            }
        }
        self.part_after(row, col, bsize, p);
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
        let split = u32::from(p == PARTITION_SPLIT);
        if s.intra_frame {
            // A key frame's partition probabilities are fixed.
            if has_rows && has_cols {
                self.put_tree(&prob::PARTITION_TREE, &probs, p);
            } else if !has_rows && has_cols {
                self.put(split, u32::from(probs[1]));
            } else if has_rows && !has_cols {
                self.put(split, u32::from(probs[2]));
            }
        } else {
            let at = probs::PARTITION + 3 * ctx;
            if has_rows && has_cols {
                self.put_tree_nc(&prob::PARTITION_TREE, &probs, p, at);
            } else if !has_rows && has_cols {
                self.put_nc(split, probs[1], at + 1);
            } else if has_rows && !has_cols {
                self.put_nc(split, probs[2], at + 2);
            }
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
            self.w.push((bit as u8, prob as u16));
        }
    }

    /// A coefficient token bit, at the probability `id` names (which the
    /// frame's update may change).
    #[inline]
    fn put_coef(&mut self, bit: u32, prob: u8, id: u16) {
        if self.trial {
            self.cost += u64::from(bit_cost(bit != 0, prob));
        } else {
            self.w.push((bit as u8, COEF_ID + id));
        }
    }

    /// A bit at the updatable probability `at` of [`probs`](super::probs)
    /// (which holds `prob` now).
    #[inline]
    fn put_nc(&mut self, bit: u32, prob: u8, at: usize) {
        if self.trial {
            self.cost += u64::from(bit_cost(bit != 0, prob));
        } else {
            self.w.push((bit as u8, NC_ID + at as u16));
        }
    }

    /// A symbol of a tree whose probabilities `probs` are the updatable
    /// ones from `at` on.
    fn put_tree_nc(&mut self, tree: &[i8], probs: &[u8], symbol: u8, at: usize) {
        if self.trial {
            self.put_tree(tree, probs, symbol);
        } else {
            let w = &mut *self.w;
            probs::walk(tree, symbol, |bit, node| {
                w.push((bit as u8, NC_ID + (at + node) as u16));
            });
        }
    }

    /// Writes a symbol of a tree, or in a trial adds its cost.
    fn put_tree(&mut self, tree: &[i8], probs: &[u8], symbol: u8) {
        if self.trial {
            let mut c = Cost(0);
            tree_to(&mut c, tree, probs, symbol);
            self.cost += u64::from(c.0);
        } else {
            tree_to(self.w, tree, probs, symbol);
        }
    }

    // Blocks ----------------------------------------------------------------

    /// Codes one block as `b0` says (in a trial, costed): prediction,
    /// residual, the skip decision, the syntax, the contexts. Returns the
    /// SSE of its reconstruction.
    fn code_block(&mut self, b0: &Blk) -> f64 {
        let s = self.s;
        let mut b = *b0;
        self.seg = usize::from(b.seg);
        let (row, col, bsize) = (b.row, b.col, b.bsize);
        let (bw, bh) = (b.bw, b.bh);
        let inter = b.refs[0] != INTRA_FRAME;
        let sub_inter = inter && bsize < BLOCK_8X8;
        let rec = BlockRec {
            row: row as u16,
            col: col as u16,
            bsize,
            coded: true,
            tx: b.tx,
            uv_tx: self.uv_tx(&b),
            mode: b.mode,
            bmode: if sub_inter { b.bmode } else { [b.mode; 4] },
            uv: b.uv,
            interp: b.interp,
            refs: b.refs,
            bmv: if sub_inter {
                b.smv.map(|m| [m, Mv::ZERO])
            } else {
                [[b.mv, b.mv2]; 4]
            },
        };
        if inter {
            self.rec.predict_inter(&rec);
        }
        // Coefficients of every tx block, kept until the mode info is
        // written (intra predicts tx block by tx block from the ones
        // before, so it is reconstructed as it goes).
        let mut toks = std::mem::take(&mut self.toks);
        for t in toks.drain(..) {
            self.pool.push(t.4);
        }
        let mut eobs = 0;
        let (coded, dropped) = self.residual(&b, &rec, &mut toks, &mut eobs);
        if bsize < BLOCK_8X8 && !inter {
            b.bmode = self.sub;
            b.mode = b.bmode[3];
        }
        if inter && eobs > 0 {
            // Skip when the residual is not worth its bits.
            let ctx = usize::from(self.above_nb(&b).is_some_and(|n| n.skip))
                + usize::from(self.left_nb(&b).is_some_and(|n| n.skip));
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
        if s.intra_frame {
            self.write_intra_frame_info(&b);
        } else {
            self.write_inter_frame_info(&b);
        }
        {
            let r = MvRef {
                mv: [b.mv, b.mv2],
                refs: b.refs,
            };
            let tw = self.f.width;
            for y in 0..b.y_mis {
                let at = (row + y) * tw + col - self.col0;
                self.f.mvs[at..at + b.x_mis].fill(r);
                if let Some(cells) = self.f.segmap.get_mut(at..at + b.x_mis) {
                    cells.fill(b.seg);
                }
            }
        }
        if b.skip {
            self.reset_nz(&b);
        } else {
            for (plane, tx, r4, c4, levels, eob) in toks.iter() {
                self.write_coefs(&b, *plane, *tx, *r4, *c4, levels, *eob);
            }
        }
        self.toks = toks;
        let nb = Nb {
            bsize,
            mode: b.mode,
            // A block under 8×8 shows its 4×4 modes when intra; when inter,
            // its last sub-block's mode (as the decoder keeps it).
            bmode: if bsize < BLOCK_8X8 && !inter {
                b.bmode
            } else {
                [b.mode; 4]
            },
            skip: b.skip,
            tx: b.tx,
            seg_pred: false,
            interp: if inter { b.interp } else { 3 },
            refs: b.refs,
            bmv: if sub_inter {
                b.smv.map(|m| [m, Mv::ZERO])
            } else {
                [[b.mv, b.mv2]; 4]
            },
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
                    seg: b.seg,
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
                    seg: b.seg,
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

    /// The intra mode of the 4×4 luma block at `(x, y)` whose prediction
    /// is nearest the source (SATD, with a little for leaving DC).
    fn pick_4x4(
        &mut self,
        x: usize,
        y: usize,
        have_left: bool,
        have_above: bool,
        have_right: bool,
    ) -> u8 {
        let s = self.s;
        let e = self
            .rec
            .intra_edges(0, 0, x, y, have_left, have_above, have_right);
        let (src, ss) = (s.src.planes[0], s.src.strides[0]);
        let lam = u64::from(s.lambda);
        let mut best = (u64::MAX, geom::DC);
        for m in [
            geom::DC,
            geom::V,
            geom::H,
            geom::TM,
            geom::D45,
            geom::D135,
            geom::D117,
            geom::D153,
            geom::D207,
            geom::D63,
        ] {
            ipred::predict(m, 4, &mut self.pred, 4, &e, 8);
            let c = satd(&src[y * ss + x..], ss, &self.pred, 4, 4) + lam * u64::from(m != geom::DC);
            if c < best.0 {
                best = (c, m);
            }
        }
        best.1
    }

    fn uv_tx(&self, b: &Blk) -> u8 {
        geom::uv_tx_420(b.bsize, b.tx)
    }

    // Residual -----------------------------------------------------------

    /// Every tx block's prediction (intra), residual, quantized levels and
    /// reconstruction, in the decoder's order.
    /// Returns the summed costs of the tx blocks as coded and dropped.
    fn residual(
        &mut self,
        b: &Blk,
        rec: &BlockRec,
        toks: &mut Vec<Tok>,
        eobs: &mut usize,
    ) -> (f32, f32) {
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
                        let mode = if plane > 0 {
                            b.uv
                        } else if b.bsize < BLOCK_8X8 {
                            let k = r * 2 + c;
                            if b.bmode[k] == AUTO {
                                // The 4×4's mode from its own edges, by SATD.
                                self.sub[k] = self.pick_4x4(
                                    x,
                                    y,
                                    have_left_of(c, b, self.col0),
                                    r > 0 || b.row > 0,
                                    c + step < n4w,
                                );
                            } else {
                                self.sub[k] = b.bmode[k];
                            }
                            self.sub[k]
                        } else {
                            b.mode
                        };
                        let have_above = r > 0 || b.row > 0;
                        let have_left = c > 0 || b.col > self.col0;
                        let have_right = c + step < n4w;
                        self.rec.predict_intra(
                            plane, mode, tx, x, y, have_left, have_above, have_right,
                        );
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
        let vw = if right >= 0 {
            n4w
        } else {
            (n4w as i32 + (right >> (5 + ss))).max(0) as usize
        };
        let vh = if bottom >= 0 {
            n4h
        } else {
            (n4h as i32 + (bottom >> (5 + ss))).max(0) as usize
        };
        (n4w, n4h, vw, vh)
    }

    /// The levels of one tx block (transposed positions, the scan's order
    /// of the decoder), and its end of block; the block is reconstructed.
    fn quantize(
        &mut self,
        plane: usize,
        tx: u8,
        kind: u8,
        x: usize,
        y: usize,
        intra: bool,
    ) -> (Vec<i32>, usize) {
        let s = self.s;
        let n = 4usize << tx;
        let log2n = 2 + usize::from(tx);
        let (src, ss) = (s.src.planes[plane], s.src.strides[plane]);
        {
            let v = &mut self.rec.views()[plane];
            let stride = v.stride;
            let pred = v.at(y, x);
            // Row slices, so that each row's differences vectorize.
            for r in 0..n {
                let sr = &src[(y + r) * ss + x..][..n];
                let pr = &pred[r * stride..][..n];
                for ((d, &a), &b) in self.res[r * n..][..n].iter_mut().zip(sr).zip(pr) {
                    *d = i32::from(a) - i32::from(b);
                }
            }
        }
        let mut levels = self.pool.pop().unwrap_or_default();
        levels.clear();
        levels.resize(n * n, 0);
        self.rd = TxRd::default();
        let mut rd_eob = None;
        let dqi = usize::from(plane > 0);
        let (dc_q, ac_q) = (s.dq[self.seg][dqi][0], s.dq[self.seg][dqi][1]);
        let shift = u32::from(tx == 3);
        let quiet = {
            // No coefficient can reach 0.3 of a step when the residual's
            // energy (by Parseval, through the transform's gain) is below
            // that: every level is zero, as quantizing would find.
            let ss: i64 = self.res[..n * n]
                .iter()
                .map(|&r| i64::from(r) * i64::from(r))
                .sum();
            let step = (dc_q.min(ac_q) as f32) / (1 << shift) as f32;
            let g = super::tx::gain(usize::from(tx));
            let floor = 0.3 * step;
            (ss as f32) * g < floor * floor
        };
        if s.h.lossless {
            let mut c = [0i32; 16];
            fwht(&self.res, n, &mut c);
            for (i, v) in c.iter().enumerate() {
                let q = if i == 0 { dc_q } else { ac_q };
                levels[i] = *v / q.max(1);
            }
        } else if quiet {
            // Dropped and coded cost the same: the samples' error and an end
            // of block.
            let ss: i64 = self.res[..n * n]
                .iter()
                .map(|&r| i64::from(r) * i64::from(r))
                .sum();
            self.rd = TxRd {
                coded: ss as f32,
                dropped: ss as f32,
            };
            rd_eob = Some(0);
        } else {
            forward(
                &self.res,
                n,
                usize::from(tx),
                kind & 1 == 1,
                kind & 2 == 2,
                &mut self.fcoef,
            );
            if s.speed >= 9 {
                // Fastest: a dead zone, and costs from a bit model.
                let (rd, any) = self.dz_levels(tx, dc_q, ac_q, intra, &mut levels);
                self.rd = rd;
                // The end of block: none, or from the end of the scan back.
                let (scan, _) = scan_pair(tx, kind);
                let mut e = if any { scan.len() } else { 0 };
                while e > 0 {
                    let rc = usize::from(scan[e - 1]);
                    if levels[((rc & (n - 1)) << log2n) | (rc >> log2n)] != 0 {
                        break;
                    }
                    e -= 1;
                }
                rd_eob = Some(e);
            } else {
                let one = self.trial && s.speed >= DZ_TRIAL_SPEED;
                let (eob, rd) = self.rd_levels(plane, tx, kind, x, y, intra, one, &mut levels);
                self.rd = rd;
                rd_eob = Some(eob);
            }
        }
        // End of block in the decoder's scan; drop what lies past it.
        let (scan, _) = scan_pair(tx, kind);
        let eob = rd_eob.unwrap_or_else(|| {
            let mut eob = 0;
            for (k, &rc) in scan.iter().enumerate() {
                let rc = usize::from(rc);
                let t = ((rc & (n - 1)) << log2n) | (rc >> log2n);
                if levels[t] != 0 {
                    eob = k + 1;
                }
            }
            eob
        });
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
            self.rec
                .add_residual(plane, x, y, tx, kind, eob, (max_r + 1, max_c + 1));
        }
        (levels, eob)
    }

    /// Dead-zone quantization of `self.fcoef` into `levels`, with the
    /// costs of coding and dropping the block from a rough bit model.
    fn dz_levels(
        &self,
        tx: u8,
        dc_q: i32,
        ac_q: i32,
        intra: bool,
        levels: &mut [i32],
    ) -> (TxRd, bool) {
        let s = self.s;
        let n = 4usize << tx;
        let max = n * n;
        let shift = u32::from(tx == 3);
        let g = super::tx::gain(usize::from(tx));
        let inv = [
            (1 << shift) as f32 / dc_q as f32,
            (1 << shift) as f32 / ac_q as f32,
        ];
        let steps = [
            dc_q as f32 / (1 << shift) as f32,
            ac_q as f32 / (1 << shift) as f32,
        ];
        let round = if intra { 0.42 } else { 0.33 };
        // The DC at its own step, rounded to nearest; then the rest.
        use cutlass_simd::vp9::quant;
        let dc = quant::dz(&self.fcoef[..1], &mut levels[..1], inv[0], steps[0], 0.5);
        let ac = quant::dz(
            &self.fcoef[1..max],
            &mut levels[1..max],
            inv[1],
            steps[1],
            round,
        );
        let d_coded = dc.err + ac.err;
        let d_all = dc.energy + ac.energy;
        let bits = (dc.bits + ac.bits) as f32;
        let any = dc.nonzero || ac.nonzero;
        let lam = s.rd_lambda;
        (
            TxRd {
                coded: (d_coded + lam * (bits + 1.0) * g) / g,
                dropped: (d_all + lam * g) / g,
            },
            any,
        )
    }

    /// Rate-distortion quantization of `self.fcoef` into `levels`: each
    /// coefficient takes the level, of the two nearest its value (or
    /// zero), with the least distortion plus `lambda` times its token's
    /// bits, in scan order (so each token's context is known); then the
    /// end of block goes where the cost is least. Returns the end of
    /// block and the costs of coding and of dropping the block. With
    /// `one`, each coefficient takes its dead-zone level instead (the
    /// search's trials, from `DZ_TRIAL_SPEED`).
    #[allow(clippy::too_many_arguments)]
    fn rd_levels(
        &mut self,
        plane: usize,
        tx: u8,
        kind: u8,
        x: usize,
        y: usize,
        intra: bool,
        one: bool,
        levels: &mut [i32],
    ) -> (usize, TxRd) {
        let mut q = std::mem::take(&mut self.rdq);
        let out = self.rd_levels_in(&mut q, plane, tx, kind, x, y, intra, one, levels);
        self.rdq = q;
        out
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn rd_levels_in(
        &mut self,
        q: &mut Rdq,
        plane: usize,
        tx: u8,
        kind: u8,
        x: usize,
        y: usize,
        intra: bool,
        one: bool,
        levels: &mut [i32],
    ) -> (usize, TxRd) {
        let s = self.s;
        // With `one`, each coefficient takes its dead-zone level alone.
        let rnd = if intra { 0.42f32 } else { 0.33 };
        let n = 4usize << tx;
        let shift = u32::from(tx == 3);
        let dqi = usize::from(plane > 0);
        let (dc_q, ac_q) = (s.dq[self.seg][dqi][0], s.dq[self.seg][dqi][1]);
        let g = super::tx::gain(usize::from(tx));
        let lam = s.rd_lambda * g / 256.0;
        let (scan, nb) = scan_pair(tx, kind);
        // Each scan position's place in the transposed block.
        let tpos = crate::dec::parse::scan_pair(tx, kind).2;
        let probs = &s.fc.coef[usize::from(tx)][dqi][usize::from(!intra)];
        let bands: &[u8] = if tx == 0 { &BAND_4X4 } else { &BAND_8X8_PLUS };
        let max = n * n;
        // The context of the first token, from the neighbours so far.
        let ctx0 = {
            let ss = usize::from(plane > 0);
            let n4 = 1usize << tx;
            let a = (x >> 2) - (((self.col0 * 8) >> ss) >> 2);
            let l = (y >> 2) & (15 >> ss);
            let anz = self.f.above_nz[plane][a..(a + n4).min(self.f.above_nz[plane].len())]
                .iter()
                .any(|&v| v != 0);
            let lnz = self.left_nz[plane][l..(l + n4).min(16)]
                .iter()
                .any(|&v| v != 0);
            usize::from(anz) + usize::from(lnz)
        };
        // Values in steps, in scan order; nothing past the last that could
        // round to a level is worth looking at.
        let av = &mut q.av[..1024];
        let mut steps = [0.0f32; 2];
        steps[0] = dc_q as f32 / (1 << shift) as f32;
        steps[1] = ac_q as f32 / (1 << shift) as f32;
        // In steps, in natural order (one pass the compiler vectorizes),
        // then the last that could round to a level, from the end of the
        // scan back.
        let inv_ac = 1.0 / steps[1];
        let a_nat = &mut q.a_nat[..1024];
        // Eight running sums and peaks, so that the pass vectorizes (a
        // single float sum must keep its order; every block has a
        // multiple of 16 coefficients).
        // And the scan's last that could round to a level (0.3 of a step
        // or more; with `one`, what the dead zone rounds up), as one past
        // the largest scan index among them.
        let iscan = crate::dec::parse::iscan(tx, kind);
        let (th, th_dc) = if one { (1.0 - rnd, 0.5) } else { (0.3, 0.3) };
        let mut sums = [0.0f32; 8];
        let mut lasts = [0u32; 8];
        for ((a8, v8), i8) in a_nat[..max]
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(self.fcoef[..max].as_chunks::<8>().0)
            .zip(iscan[..max].as_chunks::<8>().0)
        {
            for i in 0..8 {
                let v = v8[i];
                let a = v.abs() * inv_ac;
                a8[i] = a;
                sums[i] += v * v;
                let e = if a >= th { u32::from(i8[i]) + 1 } else { 0 };
                lasts[i] = lasts[i].max(e);
            }
        }
        let d_all: f32 = sums.iter().sum();
        a_nat[0] = self.fcoef[0].abs() / steps[0];
        let tr = |k: usize| usize::from(tpos[k]);
        // Only the DC is at scan index 0, and it has its own step.
        let mut last_sig = lasts.iter().copied().max().unwrap_or(0) as usize;
        if last_sig <= 1 {
            last_sig = usize::from(a_nat[0] >= th_dc);
        }
        if !one {
            for (k, a) in av[..last_sig].iter_mut().enumerate() {
                *a = a_nat[tr(k)];
            }
        }
        if last_sig == 0 {
            let e0 = lam * bit_cost(false, probs[0][ctx0][0]) as f32;
            levels[..max].fill(0);
            let d = (d_all + e0) / g;
            return (
                0,
                TxRd {
                    coded: d,
                    dropped: d,
                },
            );
        }
        // Token costs by band and context.
        let tc = &s.tok[usize::from(tx)][dqi][usize::from(!intra)];
        let energy = &mut q.energy[..1024];
        let mut ctx = ctx0;
        let mut prev_zero = false;
        // Per position: the chosen cost (distortion + lambda·rate) so far,
        // the distortion if it were zero, and the end-of-block cost there.
        let j_pref = &mut q.j_pref[..1025];
        let d_tail = &mut q.d_tail[..1025];
        let eob_cost = &mut q.eob_cost[..1025];
        j_pref[0] = 0.0;
        let mut last_nz = 0usize;
        let mut d_rest = d_all;
        // A trial's walk: each level the dead zone's, costed as below.
        if one {
            for k in 0..last_sig {
                let rc = usize::from(scan[k]);
                let t = usize::from(tpos[k]);
                let (st, r0) = if k == 0 {
                    (steps[0], 0.5)
                } else {
                    (steps[1], rnd)
                };
                let a = a_nat[t];
                let band = usize::from(bands[k]);
                let tcs = &tc[band][ctx];
                eob_cost[k] = lam * tcs[1] as f32;
                let l = (a + r0) as u32;
                let more = if prev_zero { 0 } else { tcs[0] };
                let r = more
                    + match l {
                        0 => tcs[2],
                        1 => tcs[3],
                        _ => {
                            let mut c = Cost(0);
                            write_big(
                                &mut c,
                                &PARETO8[usize::from(probs[band][ctx][2].max(1)) - 1],
                                l,
                            );
                            tcs[4] + c.0
                        }
                    };
                let e = (a - l as f32) * st;
                let z = a * st;
                j_pref[k + 1] = j_pref[k] + (e * e + lam * r as f32);
                d_tail[k] = d_rest;
                d_rest -= z * z;
                levels[t] = if l == 0 {
                    0
                } else {
                    last_nz = k + 1;
                    if self.fcoef[t] < 0.0 {
                        -(l as i32)
                    } else {
                        l as i32
                    }
                };
                energy[rc] = ENERGY[l.min(11) as usize];
                prev_zero = l == 0;
                if k + 1 < max {
                    ctx = (1
                        + usize::from(energy[usize::from(nb[2 * (k + 1)])])
                        + usize::from(energy[usize::from(nb[2 * (k + 1) + 1])]))
                        >> 1;
                }
            }
        } else {
            for k in 0..last_sig {
                let rc = usize::from(scan[k]);
                let t = usize::from(tpos[k]);
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
                            write_big(
                                &mut c,
                                &PARETO8[usize::from(probs[band][ctx][2].max(1)) - 1],
                                l,
                            );
                            more + tcs[4] + c.0
                        }
                    }
                };
                let mut best = (f32::MAX, 0u32);
                let only;
                let cands: &[u32] = if one {
                    only = [(a + if k == 0 { 0.5 } else { rnd }) as u32];
                    &only
                } else if lo == 0 {
                    &[0, 1]
                } else if lo == 1 {
                    &[0, 1, 2]
                } else {
                    &[lo, lo + 1]
                };
                for &l in cands {
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
                energy[rc] = ENERGY[l.min(11) as usize];
                prev_zero = l == 0;
                if k + 1 < max {
                    ctx = (1
                        + usize::from(energy[usize::from(nb[2 * (k + 1)])])
                        + usize::from(energy[usize::from(nb[2 * (k + 1) + 1])]))
                        >> 1;
                }
            }
        }
        d_tail[last_sig] = d_rest.max(0.0);
        if last_sig < max {
            eob_cost[last_sig] = lam * tc[usize::from(bands[last_sig])][ctx][1] as f32;
        }
        for &t in &tpos[last_sig..max] {
            levels[usize::from(t)] = 0;
        }
        // The best end: after a nonzero level, or none.
        let mut best = (d_tail[0] + eob_cost[0], 0usize);
        for e in 1..=last_nz {
            if levels[usize::from(tpos[e - 1])] == 0 {
                continue;
            }
            let end = if e < max { eob_cost[e] } else { 0.0 };
            let j = j_pref[e] + d_tail[e] + end;
            if j < best.0 {
                best = (j, e);
            }
        }
        let eob = best.1;
        for &t in &tpos[eob..last_sig] {
            levels[usize::from(t)] = 0;
        }
        let zero = TxRd {
            coded: best.0 / g,
            dropped: (d_tail[0] + eob_cost[0]) / g,
        };
        (eob, zero)
    }

    // Syntax ---------------------------------------------------------------

    fn write_skip(&mut self, b: &Blk) {
        let ctx = usize::from(self.above_nb(b).is_some_and(|n| n.skip))
            + usize::from(self.left_nb(b).is_some_and(|n| n.skip));
        self.put_nc(u32::from(b.skip), self.s.fc.skip[ctx], probs::SKIP + ctx);
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
        let at = match limit {
            geom::TX_8X8 => probs::TX8 + ctx,
            geom::TX_16X16 => probs::TX16 + 2 * ctx,
            _ => probs::TX32 + 3 * ctx,
        };
        let tx = b.tx;
        self.put_nc(u32::from(tx != 0), probs[0], at);
        if tx != 0 && limit >= geom::TX_16X16 {
            self.put_nc(u32::from(tx > 1), probs[1], at + 1);
            if tx != 1 && limit >= TX_32X32 {
                self.put_nc(u32::from(tx > 2), probs[2], at + 2);
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

    /// The block's segment, when the frame codes a map (never predicted
    /// from the last frame's).
    fn write_seg(&mut self, b: &Blk) {
        if !self.s.seg_map.is_empty() {
            let probs = self.s.h.seg.tree_probs;
            self.put_tree(&prob::SEG_TREE, &probs, b.seg);
        }
    }

    fn write_intra_frame_info(&mut self, b: &Blk) {
        self.write_seg(b);
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
        if b.bsize < BLOCK_8X8 {
            // Four modes, each in the context of the ones above and left
            // of it (parse.rs `intra_frame_info`).
            let ab = |k: usize| match above {
                Some(a) if a.refs[0] == INTRA_FRAME => a.bmode[k + 2],
                _ => geom::DC,
            };
            let lb = |k: usize| match left {
                Some(l) if l.refs[0] == INTRA_FRAME => l.bmode[k + 1],
                _ => geom::DC,
            };
            for k in 0..4 {
                let a = if k < 2 { ab(k) } else { b.bmode[k - 2] };
                let l = if k % 2 == 0 { lb(k) } else { b.bmode[k - 1] };
                self.put_tree(
                    &prob::INTRA_TREE,
                    &KF_Y_MODE_PROB[usize::from(a)][usize::from(l)],
                    b.bmode[k],
                );
            }
        } else {
            self.put_tree(
                &prob::INTRA_TREE,
                &KF_Y_MODE_PROB[usize::from(a)][usize::from(l)],
                b.mode,
            );
        }
        self.put_tree(
            &prob::INTRA_TREE,
            &KF_UV_MODE_PROB[usize::from(b.mode)],
            b.uv,
        );
    }

    fn write_inter_frame_info(&mut self, b: &Blk) {
        let s = self.s;
        self.write_seg(b);
        self.write_skip(b);
        let inter = b.refs[0] != INTRA_FRAME;
        let ctx = ctx::intra_inter_ctx(self.above_nb(b), self.left_nb(b));
        self.put_nc(
            u32::from(inter),
            s.fc.intra_inter[ctx],
            probs::INTRA_INTER + ctx,
        );
        if self.collect() {
            self.f.counts.intra_inter[ctx][usize::from(inter)] += 1;
        }
        self.write_tx(b, !b.skip || !inter);
        if !inter {
            let group = geom::size_group(b.bsize);
            self.put_tree_nc(
                &prob::INTRA_TREE,
                &s.fc.y_mode[group],
                b.mode,
                probs::Y_MODE + 9 * group,
            );
            if self.collect() {
                self.f.counts.y_mode[group][usize::from(b.mode)] += 1;
            }
            self.put_tree(&prob::INTRA_TREE, &s.fc.uv_mode[usize::from(b.mode)], b.uv);
            if self.collect() {
                self.f.counts.uv_mode[usize::from(b.mode)][usize::from(b.uv)] += 1;
            }
            return;
        }
        let compound = b.refs[1] > INTRA_FRAME;
        if s.h.reference_mode == geom::REFERENCE_SELECT {
            let ctx = ctx::reference_mode_ctx(self.above_nb(b), self.left_nb(b), s.h.comp_fixed);
            self.put_nc(
                u32::from(compound),
                s.fc.comp_inter[ctx],
                probs::COMP_INTER + ctx,
            );
            if self.collect() {
                self.f.counts.comp_inter[ctx][usize::from(compound)] += 1;
            }
        }
        if compound {
            // The fixed reference takes its sign bias's place; the other
            // is one of the two variable ones.
            let ctx = ctx::comp_ref_ctx(self.above_nb(b), self.left_nb(b), s.h);
            let fixed = usize::from(s.h.sign_bias[usize::from(s.h.comp_fixed)]);
            let bit = u32::from(b.refs[fixed ^ 1] == s.h.comp_var[1]);
            self.put_nc(bit, s.fc.comp_ref[ctx], probs::COMP_REF + ctx);
            if self.collect() {
                self.f.counts.comp_ref[ctx][bit as usize] += 1;
            }
        }
        // Single reference: LAST, or GOLDEN / ALTREF.
        let c0 = ctx::single_p1(self.above_nb(b), self.left_nb(b));
        let bit0 = u32::from(b.refs[0] != LAST_FRAME);
        if !compound {
            self.put_nc(bit0, s.fc.single_ref[c0][0], probs::SINGLE_REF + 2 * c0);
            if self.collect() {
                self.f.counts.single_ref[c0][0][bit0 as usize] += 1;
            }
        }
        if !compound && bit0 == 1 {
            let c1 = ctx::single_p2(self.above_nb(b), self.left_nb(b));
            let bit1 = u32::from(b.refs[0] == geom::ALTREF_FRAME);
            self.put_nc(bit1, s.fc.single_ref[c1][1], probs::SINGLE_REF + 2 * c1 + 1);
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
        let p = s.fc.inter_mode[mctx];
        if b.bsize >= BLOCK_8X8 {
            self.write_inter_mode(b.mode, &p, mctx);
        }
        if s.switchable {
            let l = self.left_nb(b).map(|n| n.interp);
            let a = self.above_nb(b).map(|n| n.interp);
            let ctx = ctx::interp_ctx(l, a);
            self.put_tree_nc(
                &prob::INTERP_TREE,
                &s.fc.interp[ctx],
                b.interp,
                probs::INTERP + 2 * ctx,
            );
            if self.collect() {
                self.f.counts.interp[ctx][usize::from(b.interp)] += 1;
            }
        }
        if b.bsize < BLOCK_8X8 {
            // Each sub-block's mode, and its vector when new, against the
            // block's list.
            let best = self.find_mv(b, NEW);
            let (sw, sh) = sub_steps(b.bsize);
            for idy in (0..2).step_by(sh) {
                for idx in (0..2).step_by(sw) {
                    let j = idy * 2 + idx;
                    self.write_inter_mode(b.bmode[j], &p, mctx);
                    if b.bmode[j] == NEW {
                        self.write_mv(b.smv[j], best);
                    }
                }
            }
        } else if b.mode == NEW {
            let best = self.find_mv(b, NEW);
            self.write_mv(b.mv, best);
            if compound {
                let best = self.find_mv_of(b, NEW, b.refs[1]);
                self.write_mv(b.mv2, best);
            }
        }
    }

    /// An inter mode, in context `mctx` (probabilities `p`).
    fn write_inter_mode(&mut self, mode: u8, p: &[u8; 3], mctx: usize) {
        let k = mode - NEAREST;
        self.put_tree_nc(&prob::INTER_TREE, p, k, probs::INTER_MODE + 3 * mctx);
        if self.collect() {
            self.f.counts.inter_mode[mctx][usize::from(k)] += 1;
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
        self.put_tree_nc(&prob::JOINT_TREE, &s.fc.mv.joints, joint, probs::MV_JOINTS);
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
        let at = probs::MV_COMP + which * probs::MV_COMP_LEN;
        let fp = probs::MV_FP + which * probs::MV_FP_LEN;
        let hp = probs::MV_HP + 2 * which;
        self.put_nc(u32::from(sign), c.sign, at);
        self.put_tree_nc(&prob::CLASS_TREE, &c.classes, class as u8, at + 1);
        if class == 0 {
            self.put_nc(d, c.class0, at + 11);
            self.put_tree_nc(
                &prob::FP_TREE,
                &c.class0_fp[d as usize],
                f as u8,
                fp + 3 * d as usize,
            );
            if use_hp {
                self.put_nc(e, c.class0_hp, hp);
            }
        } else {
            for (i, &p) in c.bits.iter().enumerate().take(class) {
                self.put_nc((d >> i) & 1, p, at + 12 + i);
            }
            self.put_tree_nc(&prob::FP_TREE, &c.fp, f as u8, fp + 6);
            if use_hp {
                self.put_nc(e, c.hp, hp + 1);
            }
        }
    }

    fn near(&self, b: &Blk, pos: [i8; 2]) -> Option<&Nb> {
        let r = b.row as isize + isize::from(pos[0]);
        let c = b.col as isize + isize::from(pos[1]);
        if r < 0
            || c < self.col0 as isize
            || r >= self.s.mi_rows as isize
            || c >= self.col1 as isize
        {
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
        if r < 0
            || c < self.col0 as isize
            || r >= self.s.mi_rows as isize
            || c >= self.col1 as isize
        {
            return None;
        }
        Some(self.f.mvs[r as usize * self.f.width + c as usize - self.col0])
    }

    /// The vector `mode` takes (NEAREST, NEAR) or codes from (NEW): the
    /// list of `dec::parse::find_mv` for blocks of 8×8 and up, reference
    /// LAST, its last entry, with the precision lowered.
    fn find_mv(&self, b: &Blk, mode: u8) -> Mv {
        self.find_mv_of(b, mode, b.refs[0])
    }

    /// [`Tile::find_mv`] for reference `frame` of the block.
    fn find_mv_of(&self, b: &Blk, mode: u8, frame: u8) -> Mv {
        let (list, count) = self.find_list(b, mode, frame, -1);
        lower_precision(list[count - 1], self.s.h.allow_hp)
    }

    /// The reference list of `dec::parse::find_mv`: for sub-block `block`
    /// of a block under 8×8 (`-1` the whole block), clamped.
    fn find_list(&self, b: &Blk, mode: u8, frame: u8, block: i32) -> ([Mv; 2], usize) {
        let s = self.s;
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
            s.prev_mvs
                .and_then(|m| m.get(b.row * s.mi_cols + b.col).copied())
        } else {
            None
        };
        'search: {
            let mut i = 0;
            if block >= 0 {
                // The two nearest neighbours by their sub-blocks' vectors.
                while i < 2 {
                    if let Some(n) = self.near(b, search[i]) {
                        different = true;
                        let which = if n.refs[0] == frame {
                            Some(0)
                        } else if n.refs[1] == frame {
                            Some(1)
                        } else {
                            None
                        };
                        if let Some(w) = which {
                            let mv = if n.bsize < BLOCK_8X8 {
                                let k = SUBBLOCK[block as usize][usize::from(search[i][1] == 0)];
                                n.bmv[usize::from(k)][w]
                            } else {
                                n.bmv[3][w]
                            };
                            if push(&mut list, &mut count, mv) {
                                break 'search;
                            }
                        }
                    }
                    i += 1;
                }
            }
            for pos in &search[i..] {
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
                    let Some(c) = self.cell(b, *pos) else {
                        continue;
                    };
                    if c.refs[0] == INTRA_FRAME {
                        continue;
                    }
                    if c.refs[0] != frame
                        && push(&mut list, &mut count, flip(c.mv[0], c.refs[0], frame, bias))
                    {
                        break 'search;
                    }
                    if c.refs[1] > INTRA_FRAME
                        && c.refs[1] != frame
                        && c.mv[1] != c.mv[0]
                        && push(&mut list, &mut count, flip(c.mv[1], c.refs[1], frame, bias))
                    {
                        break 'search;
                    }
                }
            }
            if let Some(p) = prev {
                if p.refs[0] != frame
                    && p.refs[0] > INTRA_FRAME
                    && push(&mut list, &mut count, flip(p.mv[0], p.refs[0], frame, bias))
                {
                    break 'search;
                }
                if p.refs[1] > INTRA_FRAME
                    && p.refs[1] != frame
                    && p.mv[1] != p.mv[0]
                    && push(&mut list, &mut count, flip(p.mv[1], p.refs[1], frame, bias))
                {
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
        (list, count.max(1))
    }

    /// The NEAREST or NEAR vector of sub-block `j` of an inter block under
    /// 8×8, from the sub-blocks before it (`dec::parse::sub_candidate`).
    fn sub_candidate(&self, b: &Blk, bm: u8, j: usize) -> Mv {
        let frame = b.refs[0];
        if j == 0 {
            let (list, count) = self.find_list(b, bm, frame, 0);
            return list[count - 1];
        }
        if bm == NEAREST {
            return b.smv[if j == 3 { 2 } else { 0 }];
        }
        let (list, _) = self.find_list(b, bm, frame, j as i32);
        if j == 3 {
            [b.smv[1], b.smv[0], list[0], list[1]]
                .into_iter()
                .find(|&c| c != b.smv[2])
                .unwrap_or(Mv::ZERO)
        } else {
            list.into_iter()
                .find(|&c| c != b.smv[0])
                .unwrap_or(Mv::ZERO)
        }
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
    fn write_coefs(
        &mut self,
        b: &Blk,
        plane: usize,
        tx: u8,
        r4: usize,
        c4: usize,
        levels: &[i32],
        eob: usize,
    ) {
        let s = self.s;
        let ss = usize::from(plane > 0);
        let n = 1usize << tx;
        let a = (((b.col - self.col0) * 2) >> ss) + c4;
        let l = (((b.row * 2) & 15) >> ss) + r4;
        let (_, _, vw, vh) = self.visible(b, plane);
        let anz = self.f.above_nz[plane][a..(a + n).min(self.f.above_nz[plane].len())]
            .iter()
            .any(|&v| v != 0);
        let lnz = self.left_nz[plane][l..(l + n).min(16)]
            .iter()
            .any(|&v| v != 0);
        let mut ctx = usize::from(anz) + usize::from(lnz);
        let intra = b.refs[0] == INTRA_FRAME;
        let kind = if intra && plane == 0 && !s.h.lossless && tx != TX_32X32 {
            geom::intra_tx_type(if b.bsize < BLOCK_8X8 {
                b.bmode[(r4 * 2 + c4) & 3]
            } else {
                b.mode
            })
        } else {
            0
        };
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
                self.put_coef(1, p[0], coef_flat(ti, pi, ri, band, ctx, 0));
            }
            if v == 0 {
                self.put_coef(0, p[1], coef_flat(ti, pi, ri, band, ctx, 1));
                if collect {
                    self.f.counts.coef[ti][pi][ri][band][ctx][0] += 1;
                }
                self.cache[rc] = 0;
                after_zero = true;
            } else {
                self.put_coef(1, p[1], coef_flat(ti, pi, ri, band, ctx, 1));
                let a = v.unsigned_abs();
                if a == 1 {
                    self.put_coef(0, p[2], coef_flat(ti, pi, ri, band, ctx, 2));
                    if collect {
                        self.f.counts.coef[ti][pi][ri][band][ctx][1] += 1;
                    }
                    self.cache[rc] = 1;
                } else {
                    self.put_coef(1, p[2], coef_flat(ti, pi, ri, band, ctx, 2));
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
                        let p2 = coef_flat(ti, pi, ri, band, ctx, 2);
                        write_big(&mut BigOut { out: self.w, p2 }, pp, a)
                    };
                }
                self.put(u32::from(v < 0), 128);
                after_zero = false;
            }
            c += 1;
            if c < max_eob {
                ctx = (1
                    + usize::from(self.cache[usize::from(nb[2 * c])])
                    + usize::from(self.cache[usize::from(nb[2 * c + 1])]))
                    >> 1;
            }
        }
        if eob < max_eob {
            // The end of block (always after a nonzero token).
            let band = usize::from(bands[eob]);
            if collect {
                self.f.counts.eob[ti][pi][ri][band][ctx] += 1;
                self.f.counts.coef[ti][pi][ri][band][ctx][3] += 1;
            }
            self.put_coef(0, probs[band][ctx][0], coef_flat(ti, pi, ri, band, ctx, 0));
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

    /// A bit at probability `k` of a Pareto row.
    #[inline]
    fn pareto(&mut self, bit: u32, k: usize, pp: &[u8; 8]) {
        self.write(bit, u32::from(pp[k]));
    }
}

/// The final pass's symbols, kept to be written once the frame's
/// probability updates are known: a bit and what gives its probability
/// (see [`prob_of`]).
pub type Out = Vec<(u8, u16)>;

/// Symbol ids below 256 are the probability itself.
const COEF_ID: u16 = 0x1000;
/// The updatable probabilities other than the coefficients'
/// ([`probs`](super::probs)).
const NC_ID: u16 = 0x2000;
const PARETO_ID: u16 = 0x4000;

/// The index of a coefficient probability in the flattened table.
#[inline]
fn coef_flat(tx: usize, plane: usize, r: usize, band: usize, ctx: usize, node: usize) -> u16 {
    (((((tx * 2 + plane) * 2 + r) * 6 + band) * 6 + ctx) * 3 + node) as u16
}

/// The probability symbol `id` is coded at, given the frame's
/// coefficient and other updatable probabilities flattened.
#[inline]
pub fn prob_of(id: u16, coef: &[u8; 1728], nc: &probs::Flat) -> u32 {
    if id < 256 {
        u32::from(id)
    } else if (NC_ID..PARETO_ID).contains(&id) {
        u32::from(nc[usize::from(id - NC_ID)])
    } else if id < PARETO_ID {
        u32::from(coef[usize::from(id - COEF_ID)])
    } else {
        let f = usize::from((id & 0x3fff) >> 3);
        u32::from(PARETO8[usize::from(coef[f].max(1)) - 1][usize::from(id & 7)])
    }
}

impl Sink for Out {
    #[inline]
    fn write(&mut self, bit: u32, prob: u32) {
        self.push((bit as u8, prob as u16));
    }
}

/// The big-token bits of a token whose node-2 probability is `p2`.
struct BigOut<'a> {
    out: &'a mut Out,
    p2: u16,
}

impl Sink for BigOut<'_> {
    #[inline]
    fn write(&mut self, bit: u32, prob: u32) {
        self.out.push((bit as u8, prob as u16));
    }

    #[inline]
    fn pareto(&mut self, bit: u32, k: usize, _pp: &[u8; 8]) {
        self.out
            .push((bit as u8, PARETO_ID | (self.p2 << 3) | k as u16));
    }
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
    if a <= 4 {
        w.pareto(0, 0, pp);
        if a == 2 {
            w.pareto(0, 1, pp);
            2
        } else {
            w.pareto(1, 1, pp);
            w.pareto(a - 3, 2, pp);
            3
        }
    } else {
        w.pareto(1, 0, pp);
        if a <= 10 {
            w.pareto(0, 3, pp);
            if a <= 6 {
                w.pareto(0, 4, pp);
                bits(w, a - 5, &[159]);
            } else {
                w.pareto(1, 4, pp);
                bits(w, a - 7, &[165, 145]);
            }
            4
        } else {
            w.pareto(1, 3, pp);
            if a <= 34 {
                w.pareto(0, 5, pp);
                if a <= 18 {
                    w.pareto(0, 6, pp);
                    bits(w, a - 11, &[173, 148, 140]);
                } else {
                    w.pareto(1, 6, pp);
                    bits(w, a - 19, &[176, 155, 140, 135]);
                }
            } else {
                w.pareto(1, 5, pp);
                if a <= 66 {
                    w.pareto(0, 7, pp);
                    bits(w, a - 35, &[180, 157, 141, 134, 130]);
                } else {
                    w.pareto(1, 7, pp);
                    bits(w, (a - 67).min((1 << 14) - 1), &CAT6_PROB);
                }
            }
            5
        }
    }
}

/// A rough rate of a vector, in 1/256 bits.
fn mv_bits(mv: Mv) -> u32 {
    let comp = |v: i16| -> u32 {
        let a = u32::from(v.unsigned_abs());
        if a == 0 {
            256
        } else {
            256 * (2 + 2 * (32 - (a + 1).leading_zeros()))
        }
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
    let d = |v: i16| {
        if v & 1 == 0 {
            v
        } else if v > 0 {
            v - 1
        } else {
            v + 1
        }
    };
    Mv {
        row: d(mv.row),
        col: d(mv.col),
    }
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
    NEIGHBORS_COL_4X4, NEIGHBORS_COL_8X8, NEIGHBORS_COL_16X16, NEIGHBORS_DEFAULT_4X4,
    NEIGHBORS_DEFAULT_8X8, NEIGHBORS_DEFAULT_16X16, NEIGHBORS_DEFAULT_32X32, NEIGHBORS_ROW_4X4,
    NEIGHBORS_ROW_8X8, NEIGHBORS_ROW_16X16, SCAN_COL_4X4, SCAN_COL_8X8, SCAN_COL_16X16,
    SCAN_DEFAULT_4X4, SCAN_DEFAULT_8X8, SCAN_DEFAULT_16X16, SCAN_DEFAULT_32X32, SCAN_ROW_4X4,
    SCAN_ROW_8X8, SCAN_ROW_16X16,
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

/// The sum of absolute 4×4 Hadamard coefficients of `src - pred` over an
/// `w`×`h` block (`pred` rows `w` apart).
fn satd(src: &[u8], ss: usize, pred: &[u8], w: usize, h: usize) -> u64 {
    let mut total = 0u64;
    for by in (0..h).step_by(4) {
        for bx in (0..w).step_by(4) {
            let mut d = [0i32; 16];
            for r in 0..4 {
                for c in 0..4 {
                    d[r * 4 + c] = i32::from(src[(by + r) * ss + bx + c])
                        - i32::from(pred[(by + r) * w + bx + c]);
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
                total += u64::from(
                    (s0 + s1).unsigned_abs()
                        + (s0 - s1).unsigned_abs()
                        + (d0 + d1).unsigned_abs()
                        + (d0 - d1).unsigned_abs(),
                );
            }
        }
    }
    total / 2
}

/// The SAD of the source block at `(x, y)`, `w`×`h`, against the reference
/// at whole-sample offset `(dr, dc)`, or `None` outside the reference.
#[inline]
#[allow(clippy::type_complexity)]
fn sad_ref(
    (src, ss): (&[u8], usize),
    ((data, stride), (rw, rh)): ((&[u8], usize), (i32, i32)),
    (x, y, w, h): (usize, usize, usize, usize),
    dr: i32,
    dc: i32,
) -> Option<u64> {
    let (bx, by) = (x as i32 + dc, y as i32 + dr);
    if bx < 0 || by < 0 || bx + w as i32 > rw || by + h as i32 > rh {
        return None;
    }
    let a = &src[y * ss + x..];
    let b = &data[by as usize * stride + bx as usize..];
    Some(u64::from(sad::sad(a, ss, b, stride, w, h)))
}

/// A superblock's whole-sample search on LAST, made ahead of coding.
#[allow(clippy::type_complexity)]
fn search_in(
    s: &Shared<'_>,
    rp: ((&[u8], usize), (i32, i32)),
    seeds: &[Mv],
    cur_ref: usize,
    row: usize,
    col: usize,
    bsize: u8,
) -> (Mv, u64) {
    let src = (s.src.planes[0], s.src.strides[0]);
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
            seeds.get(r as usize * s.mi_cols + c as usize).copied()
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
        && usize::from(m.refs[0]) == cur_ref + 1
    {
        cands[nc] = m.mv[0];
        nc += 1;
    }
    let mut best = (0i32, 0i32);
    let mut best_sad = u64::MAX;
    for m in &cands[..nc] {
        let (r, c) = ((i32::from(m.row) + 4) >> 3, (i32::from(m.col) + 4) >> 3);
        if let Some(v) = sad_ref(src, rp, (x, y, w, h), r, c) {
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
        (BLOCK_64X64, 9..) => 4,
        (geom::BLOCK_32X32, 9..) => 1,
        // Speed 10 checks only the candidates below 32×32 (its square's
        // vector among them).
        (_, 10..) => 0,
        (_, 9) => 1,
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
                if let Some(v) = sad_ref(src, rp, (x, y, w, h), r, c)
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
    // Every whole sample within two of the diamond's choice.
    if s.speed <= 5 {
        let c = best;
        for dr in -2..=2 {
            for dc in -2..=2 {
                if (dr, dc) == (0, 0) {
                    continue;
                }
                let (r, cc) = (c.0 + dr, c.1 + dc);
                if let Some(v) = sad_ref(src, rp, (x, y, w, h), r, cc)
                    && v < best_sad
                {
                    best_sad = v;
                    best = (r, cc);
                }
            }
        }
    }
    let mv = Mv {
        row: (best.0 * 8) as i16,
        col: (best.1 * 8) as i16,
    };
    (mv, best_sad)
}

/// The fast path's partition and vectors of one superblock, searched
/// ahead of coding (frame-wide, on threads, by superblock rows).
#[derive(Clone, Copy)]
pub struct Pre {
    pub split: [bool; 21],
    pub found: [(Mv, u64); 85],
}

impl Default for Pre {
    fn default() -> Self {
        Self {
            split: [false; 21],
            found: [(Mv::ZERO, u64::MAX); 85],
        }
    }
}

/// [`Tile::choose_into`] for an inter frame, outside a tile: the split
/// flags and each square's best whole-sample vector on LAST.
#[allow(clippy::type_complexity)]
pub fn prepass_sb(
    s: &Shared<'_>,
    rp: ((&[u8], usize), (i32, i32)),
    seeds: &mut [Mv],
    mi_row: usize,
    mi_col: usize,
) -> Pre {
    let mut pre = Pre::default();
    pre_choose(s, rp, seeds, &mut pre, mi_row, mi_col, 3);
    pre
}

#[allow(clippy::type_complexity)]
fn pre_choose(
    s: &Shared<'_>,
    rp: ((&[u8], usize), (i32, i32)),
    seeds: &mut [Mv],
    pre: &mut Pre,
    row: usize,
    col: usize,
    level: usize,
) -> u64 {
    if row >= s.mi_rows || col >= s.mi_cols {
        return 0;
    }
    let lam = u64::from(s.lambda);
    let estimate = |seeds: &mut [Mv], pre: &mut Pre, bsize: u8| -> u64 {
        let (mv, sad) = search_in(s, rp, seeds, 0, row, col, bsize);
        pre.found[Tile::slot(row, col, bsize)] = (mv, sad);
        let bw = MI_WH[usize::from(bsize)].0;
        for r in 0..bw.min(s.mi_rows - row) {
            for c in 0..bw.min(s.mi_cols - col) {
                if let Some(v) = seeds.get_mut((row + r) * s.mi_cols + col + c) {
                    *v = mv;
                }
            }
        }
        sad.saturating_add(lam * u64::from(mv_bits(mv)) / 256 + lam * 3)
    };
    if level == 0 {
        return estimate(seeds, pre, SQUARE[0]);
    }
    let n = 1usize << level;
    let hbs = n / 2;
    let whole = row + n <= s.mi_rows && col + n <= s.mi_cols;
    let own = if whole {
        estimate(seeds, pre, SQUARE[level])
    } else {
        u64::MAX
    };
    let mut quarters = lam * 12;
    for (dr, dc) in [(0, 0), (0, hbs), (hbs, 0), (hbs, hbs)] {
        quarters =
            quarters.saturating_add(pre_choose(s, rp, seeds, pre, row + dr, col + dc, level - 1));
        if quarters >= own {
            break;
        }
    }
    if quarters < own {
        pre.split[Tile::node(row, col, level)] = true;
        quarters
    } else {
        own
    }
}

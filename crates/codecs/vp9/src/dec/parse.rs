//! The syntax of one tile: the partition walk, mode info, motion vectors
//! and coefficient tokens of each block, into a [`TileSyntax`] that
//! [`super::recon`] turns into samples.
//!
//! The order of the syntax follows libvpx `decode_partition`,
//! `read_intra_frame_mode_info`, `read_inter_frame_mode_info`,
//! `dec_find_mv_refs` and `decode_coefs`.

use crate::geom::{
    self, BLOCK_4X4, BLOCK_4X8, BLOCK_8X4, BLOCK_8X8, BLOCK_64X64, BLOCK_INVALID,
    COMPOUND_REFERENCE, INTRA_FRAME, NEAR, NEAREST, NEW, PARTITION_HORZ, PARTITION_NONE,
    PARTITION_SPLIT, PARTITION_VERT, REFERENCE_SELECT, TX_4X4, TX_8X8, TX_16X16, TX_32X32,
    TX_MODE_SELECT, ZERO,
};
use crate::header::Header;
use crate::lpf;
use crate::prob::{self, Counts, FrameContext};
use crate::tables::{
    BAND_4X4, BAND_8X8_PLUS, CAT6_PROB, KF_UV_MODE_PROB, KF_Y_MODE_PROB, NEIGHBORS_COL_4X4,
    NEIGHBORS_COL_8X8, NEIGHBORS_COL_16X16, NEIGHBORS_DEFAULT_4X4, NEIGHBORS_DEFAULT_8X8,
    NEIGHBORS_DEFAULT_16X16, NEIGHBORS_DEFAULT_32X32, NEIGHBORS_ROW_4X4, NEIGHBORS_ROW_8X8,
    NEIGHBORS_ROW_16X16, PARETO8, SCAN_COL_4X4, SCAN_COL_8X8, SCAN_COL_16X16, SCAN_DEFAULT_4X4,
    SCAN_DEFAULT_8X8, SCAN_DEFAULT_16X16, SCAN_DEFAULT_32X32, SCAN_ROW_4X4, SCAN_ROW_8X8,
    SCAN_ROW_16X16,
};

use super::bool::Bool;
use super::ctx::{self, Nb};
use super::frame::{Mv, MvRef};
use super::syntax::{BlockRec, TileSyntax, TxRec};

const MV_LOW: i32 = -16_384;
const MV_UPP: i32 = 16_383;
const MODE_COUNTER: [i32; 14] = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 3, 1];
const COUNTER_CTX: [u8; 19] = [2, 3, 4, 1, 3, 9, 0, 9, 9, 5, 5, 9, 5, 9, 9, 9, 9, 9, 6];
const SUBBLOCK: [[u8; 2]; 4] = [[1, 2], [1, 3], [3, 2], [3, 3]];

/// Neighbor offsets `[row, col]` in mode-info units, libvpx `mv_ref_blocks`.
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

/// Width and height of each block size in 8×8 units (at least 1).
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

/// What every tile of a frame reads and never changes.
pub struct Shared<'a> {
    pub h: &'a Header,
    pub fc: &'a FrameContext,
    pub prev_mvs: Option<&'a [MvRef]>,
    pub last_seg: &'a [u8],
    pub levels: &'a lpf::Levels,
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub sb_cols: usize,
    pub collect: bool,
    pub intra_frame: bool,
    pub part_probs: [[u8; 3]; 16],
    pub bd: u32,
    pub filter_on: bool,
    pub needs_grid: bool,
    /// LAST, GOLDEN and ALTREF: 0 usable, 1 missing, 2 of a size the
    /// frame cannot predict from.
    pub ref_status: [u8; 3],
}

/// The loop filter's view of one 8×8 block, for chroma that is not 4:2:0.
#[derive(Clone, Copy, Default)]
pub struct LfMi {
    pub bsize: u8,
    pub tx: u8,
    pub skip: bool,
    pub ref0: u8,
    pub seg: u8,
    pub mode: u8,
}

/// What one tile column writes besides its syntax, in the tile's own
/// columns (`col0` on): the motion and segmentation maps, the above
/// contexts, the loop filter masks and the symbol counts.
pub struct State {
    /// The tile's first column and its width, in 8×8 units.
    pub col0: usize,
    pub width: usize,
    pub mvs: Vec<MvRef>,
    pub segmap: Vec<u8>,
    pub above: Vec<Nb>,
    pub above_part: Vec<u8>,
    pub above_nz: [Vec<u8>; 3],
    pub masks: Vec<lpf::Mask>,
    pub grid: Vec<LfMi>,
    pub counts: Counts,
}

impl State {
    pub fn empty() -> Self {
        Self {
            col0: 0,
            width: 0,
            mvs: Vec::new(),
            segmap: Vec::new(),
            above: Vec::new(),
            above_part: Vec::new(),
            above_nz: [Vec::new(), Vec::new(), Vec::new()],
            masks: Vec::new(),
            grid: Vec::new(),
            counts: Counts::new(),
        }
    }

    /// Clears the state for a tile column `width` wide from `col0`, of a
    /// frame `mi_rows` high, with `masks` superblock masks and, with
    /// `grid`, the per-sample filter grid.
    pub fn reset(&mut self, col0: usize, width: usize, mi_rows: usize, masks: usize, grid: bool) {
        let cells = width * mi_rows;
        let a8 = width.div_ceil(8) * 8;
        self.col0 = col0;
        self.width = width;
        self.mvs.clear();
        self.mvs.resize(cells, MvRef::default());
        self.segmap.clear();
        self.segmap.resize(cells, 0);
        self.above.clear();
        self.above.resize(a8, Nb::default());
        self.above_part.clear();
        self.above_part.resize(a8, 0);
        for nz in &mut self.above_nz {
            nz.clear();
            nz.resize(a8 * 2, 0);
        }
        self.masks.clear();
        self.masks.resize(masks, lpf::Mask::default());
        self.grid.clear();
        self.grid
            .resize(if grid { cells } else { 0 }, LfMi::default());
        self.counts = Counts::new();
    }
}

/// The parser of one superblock row of one tile.
pub struct Tile<'s, 't, 'd> {
    s: &'s Shared<'s>,
    f: &'t mut State,
    out: &'t mut TileSyntax,
    b: &'t mut Bool<'d>,
    col0: usize,
    col1: usize,
    left: [Nb; 8],
    left_part: [u8; 8],
    left_nz: [[u8; 16]; 3],
    /// Token energies by position; a position is always written before
    /// a later one reads it, so it is never cleared.
    cache: [u8; 1024],
    /// Whether a block read past its data or held a bad vector.
    pub corrupt: bool,
    /// Whether a block used a reference of a size it cannot predict from.
    pub unsupported: bool,
}

/// The state of the block being decoded.
#[derive(Clone, Copy)]
struct Blk {
    row: usize,
    col: usize,
    bsize: u8,
    bw: usize,
    bh: usize,
    x_mis: usize,
    y_mis: usize,
    seg: u8,
    skip: bool,
    tx: u8,
    mode: u8,
    bmode: [u8; 4],
    uv: u8,
    interp: u8,
    refs: [u8; 2],
    mv: [Mv; 2],
    bmv: [[Mv; 2]; 4],
    seg_pred: bool,
}

impl<'s, 't, 'd> Tile<'s, 't, 'd> {
    /// The parser of one superblock row of a tile whose columns are
    /// `col0..col1`, reading from `b` into `out`.
    pub fn new(
        s: &'s Shared<'s>,
        f: &'t mut State,
        out: &'t mut TileSyntax,
        b: &'t mut Bool<'d>,
        col0: usize,
        col1: usize,
    ) -> Self {
        Self {
            s,
            f,
            out,
            b,
            col0,
            col1,
            left: [Nb::default(); 8],
            left_part: [0; 8],
            left_nz: [[0; 16]; 3],
            cache: [0; 1024],
            corrupt: false,
            unsupported: false,
        }
    }

    /// Decodes the superblocks of one row of this tile.
    pub fn sb_row(&mut self, mi_row: usize) {
        self.left = [Nb::default(); 8];
        self.left_part = [0; 8];
        self.left_nz = [[0; 16]; 3];
        let mut mi_col = self.col0;
        while mi_col < self.col1 {
            self.partition(mi_row, mi_col, BLOCK_64X64, 4);
            mi_col += 8;
        }
        if self.b.overrun() {
            self.corrupt = true;
        }
        self.out.end_row(mi_row >> 3);
    }

    // Partitions --------------------------------------------------------

    fn partition(&mut self, row: usize, col: usize, bsize: u8, n4: u32) {
        let s = self.s;
        if row >= s.mi_rows || col >= s.mi_cols {
            return;
        }
        let n8 = n4 - 1;
        let hbs = (1usize << n8) >> 1;
        let has_rows = row + hbs < s.mi_rows;
        let has_cols = col + hbs < s.mi_cols;
        let ctx = {
            let above = usize::from(self.f.above_part[col - self.col0] >> n8) & 1;
            let left = usize::from(self.left_part[row & 7] >> n8) & 1;
            left * 2 + above + n8 as usize * 4
        };
        let probs = &s.part_probs[ctx];
        let p = if has_rows && has_cols {
            if !self.b.read(probs[0]) {
                PARTITION_NONE
            } else if !self.b.read(probs[1]) {
                PARTITION_HORZ
            } else if !self.b.read(probs[2]) {
                PARTITION_VERT
            } else {
                PARTITION_SPLIT
            }
        } else if !has_rows && has_cols {
            if self.b.read(probs[1]) {
                PARTITION_SPLIT
            } else {
                PARTITION_HORZ
            }
        } else if has_rows && !has_cols {
            if self.b.read(probs[2]) {
                PARTITION_SPLIT
            } else {
                PARTITION_VERT
            }
        } else {
            PARTITION_SPLIT
        };
        if s.collect {
            self.f.counts.partition[ctx][usize::from(p)] += 1;
        }
        let sub = geom::subsize(p, bsize);
        if sub == BLOCK_INVALID {
            self.corrupt = true;
            return;
        }
        if hbs == 0 {
            self.block(row, col, sub);
        } else {
            match p {
                PARTITION_NONE => self.block(row, col, sub),
                PARTITION_HORZ => {
                    self.block(row, col, sub);
                    if has_rows {
                        self.block(row + hbs, col, sub);
                    }
                }
                PARTITION_VERT => {
                    self.block(row, col, sub);
                    if has_cols {
                        self.block(row, col + hbs, sub);
                    }
                }
                _ => {
                    self.partition(row, col, sub, n8);
                    self.partition(row, col + hbs, sub, n8);
                    self.partition(row + hbs, col, sub, n8);
                    self.partition(row + hbs, col + hbs, sub, n8);
                }
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
    }

    // Blocks ------------------------------------------------------------

    fn above_nb(&self, b: &Blk) -> Option<&Nb> {
        (b.row > 0).then(|| &self.f.above[b.col - self.col0])
    }

    fn left_nb(&self, b: &Blk) -> Option<&Nb> {
        (b.col > self.col0).then(|| &self.left[b.row & 7])
    }

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
            seg: 0,
            skip: false,
            tx: 0,
            mode: 0,
            bmode: [0; 4],
            uv: 0,
            interp: 3,
            refs: [0; 2],
            mv: [Mv::ZERO; 2],
            bmv: [[Mv::ZERO; 2]; 4],
            seg_pred: false,
        };
        if s.intra_frame {
            self.intra_frame_info(&mut b);
        } else {
            self.inter_frame_info(&mut b);
        }
        // Motion of the frame, for the vectors of later blocks and frames.
        {
            let r = MvRef {
                mv: b.mv,
                refs: b.refs,
            };
            let tw = self.f.width;
            for y in 0..b.y_mis {
                let at = (row + y) * tw + col - self.col0;
                self.f.mvs[at..at + b.x_mis].fill(r);
            }
        }
        let coded = !b.skip;
        if b.skip {
            self.reset_nz(&b);
        } else {
            let eobs = self.residual(&b);
            if b.refs[0] != INTRA_FRAME && eobs == 0 && bsize >= BLOCK_8X8 {
                b.skip = true;
            }
        }
        self.out.blocks.push(BlockRec {
            row: row as u16,
            col: col as u16,
            bsize,
            coded,
            tx: b.tx,
            uv_tx: self.uv_tx(&b),
            mode: b.mode,
            bmode: b.bmode,
            uv: b.uv,
            interp: b.interp,
            refs: b.refs,
            bmv: b.bmv,
        });
        // Leave the block for the contexts of the blocks after it.
        let nb = Nb {
            bsize,
            mode: b.mode,
            bmode: b.bmode,
            skip: b.skip,
            tx: b.tx,
            seg_pred: b.seg_pred,
            interp: b.interp,
            refs: b.refs,
            bmv: b.bmv,
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
                    seg: b.seg,
                    mode: b.mode,
                    reference: b.refs[0],
                    bw,
                    bh,
                },
                row,
                col,
            );
            if s.needs_grid {
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
    }

    // Mode info -----------------------------------------------------------

    fn intra_frame_info(&mut self, b: &mut Blk) {
        let h = self.s.h;
        if h.seg.enabled {
            if h.seg.update_map {
                b.seg = self.read_seg_id();
                self.write_seg(b, b.seg);
            } else {
                self.copy_seg(b);
            }
        }
        b.skip = self.read_skip(b);
        b.tx = self.read_tx(b, true);
        let above = self.above_nb(b).copied();
        let left = self.left_nb(b).copied();
        let amode = |k: usize, cur: &[u8; 4]| -> u8 {
            if k < 2 {
                match above {
                    Some(a) if a.refs[0] == INTRA_FRAME => a.bmode[k + 2],
                    _ => geom::DC,
                }
            } else {
                cur[k - 2]
            }
        };
        let lmode = |k: usize, cur: &[u8; 4]| -> u8 {
            if k == 0 || k == 2 {
                match left {
                    Some(l) if l.refs[0] == INTRA_FRAME => l.bmode[k + 1],
                    _ => geom::DC,
                }
            } else {
                cur[k - 1]
            }
        };
        let mut m = [0u8; 4];
        let (step_w, step_h) = sub_steps(b.bsize);
        for idy in (0..2).step_by(step_h) {
            for idx in (0..2).step_by(step_w) {
                let k = idy * 2 + idx;
                let a = usize::from(amode(k, &m));
                let l = usize::from(lmode(k, &m));
                let mode = self.b.tree(&prob::INTRA_TREE, &KF_Y_MODE_PROB[a][l]);
                m[k] = mode;
                if step_h == 2 {
                    m[k + 2] = mode;
                }
                if step_w == 2 {
                    m[k + 1] = mode;
                }
            }
        }
        if b.bsize >= BLOCK_8X8 {
            m = [m[0]; 4];
        }
        b.bmode = m;
        b.mode = m[3];
        b.uv = self
            .b
            .tree(&prob::INTRA_TREE, &KF_UV_MODE_PROB[usize::from(b.mode)]);
        b.refs = [INTRA_FRAME; 2];
        b.interp = 3;
    }

    fn inter_frame_info(&mut self, b: &mut Blk) {
        let h = self.s.h;
        if h.seg.enabled {
            let predicted = self.predicted_seg(b);
            if !h.seg.update_map {
                self.copy_seg(b);
                b.seg = predicted;
            } else if !h.seg.temporal {
                b.seg = self.read_seg_id();
                self.write_seg(b, b.seg);
            } else {
                let ctx = usize::from(self.above_nb(b).is_some_and(|n| n.seg_pred))
                    + usize::from(self.left_nb(b).is_some_and(|n| n.seg_pred));
                b.seg_pred = self.b.read(h.seg.pred_probs[ctx]);
                b.seg = if b.seg_pred {
                    predicted
                } else {
                    self.read_seg_id()
                };
                self.write_seg(b, b.seg);
            }
        }
        b.skip = self.read_skip(b);
        let seg = usize::from(b.seg);
        let inter = if h.seg.enabled && h.seg.feature_on[seg][2] {
            h.seg.feature_data[seg][2] != 0
        } else {
            let ctx = ctx::intra_inter_ctx(self.above_nb(b), self.left_nb(b));
            let v = self.b.read(self.s.fc.intra_inter[ctx]);
            if self.s.collect {
                self.f.counts.intra_inter[ctx][usize::from(v)] += 1;
            }
            v
        };
        b.tx = self.read_tx(b, !b.skip || !inter);
        if inter {
            self.inter_block_info(b);
        } else {
            self.intra_block_info(b);
        }
    }

    fn intra_block_info(&mut self, b: &mut Blk) {
        let fc = self.s.fc;
        let collect = self.s.collect;
        let y_mode = |t: &mut Self, group: usize| -> u8 {
            let m = t.b.tree(&prob::INTRA_TREE, &fc.y_mode[group]);
            if collect {
                t.f.counts.y_mode[group][usize::from(m)] += 1;
            }
            m
        };
        if b.bsize >= BLOCK_8X8 {
            let m = y_mode(self, geom::size_group(b.bsize));
            b.mode = m;
            b.bmode = [m; 4];
        } else {
            let (step_w, step_h) = sub_steps(b.bsize);
            let mut m = [0u8; 4];
            for idy in (0..2).step_by(step_h) {
                for idx in (0..2).step_by(step_w) {
                    let k = idy * 2 + idx;
                    let mode = y_mode(self, 0);
                    m[k] = mode;
                    if step_h == 2 {
                        m[k + 2] = mode;
                    }
                    if step_w == 2 {
                        m[k + 1] = mode;
                    }
                }
            }
            b.bmode = m;
            b.mode = m[3];
        }
        let i = usize::from(b.mode);
        b.uv = self.b.tree(&prob::INTRA_TREE, &fc.uv_mode[i]);
        if collect {
            self.f.counts.uv_mode[i][usize::from(b.uv)] += 1;
        }
        b.refs = [INTRA_FRAME; 2];
        b.interp = 3;
    }

    fn read_seg_id(&mut self) -> u8 {
        self.b
            .tree(&prob::SEG_TREE, &self.s.h.seg.tree_probs)
            .min(7)
    }

    fn predicted_seg(&self, b: &Blk) -> u8 {
        let s = self.s;
        let mut id = 8u8;
        for y in 0..b.y_mis {
            let at = (b.row + y) * s.mi_cols + b.col;
            if let Some(cells) = s.last_seg.get(at..at + b.x_mis) {
                for &v in cells {
                    id = id.min(v);
                }
            }
        }
        if id > 7 { 0 } else { id }
    }

    fn copy_seg(&mut self, b: &Blk) {
        let s = self.s;
        for y in 0..b.y_mis {
            let at = (b.row + y) * s.mi_cols + b.col;
            let local = (b.row + y) * self.f.width + b.col - self.col0;
            for x in 0..b.x_mis {
                self.f.segmap[local + x] = s.last_seg.get(at + x).copied().unwrap_or(0);
            }
        }
    }

    fn write_seg(&mut self, b: &Blk, id: u8) {
        for y in 0..b.y_mis {
            let at = (b.row + y) * self.f.width + b.col - self.col0;
            self.f.segmap[at..at + b.x_mis].fill(id);
        }
    }

    fn read_skip(&mut self, b: &Blk) -> bool {
        let h = self.s.h;
        if h.seg.enabled && h.seg.feature_on[usize::from(b.seg)][3] {
            return true;
        }
        let ctx = usize::from(self.above_nb(b).is_some_and(|n| n.skip))
            + usize::from(self.left_nb(b).is_some_and(|n| n.skip));
        let v = self.b.read(self.s.fc.skip[ctx]);
        if self.s.collect {
            self.f.counts.skip[ctx][usize::from(v)] += 1;
        }
        v
    }

    fn read_tx(&mut self, b: &Blk, allow: bool) -> u8 {
        let h = self.s.h;
        let limit = geom::max_tx(b.bsize);
        if !(allow && h.tx_mode == TX_MODE_SELECT && b.bsize >= BLOCK_8X8) {
            return limit.min(geom::tx_mode_biggest(h.tx_mode));
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
        let fc = self.s.fc;
        let probs: &[u8] = match limit {
            TX_8X8 => &fc.tx8[ctx],
            TX_16X16 => &fc.tx16[ctx],
            _ => &fc.tx32[ctx],
        };
        let mut tx = u8::from(self.b.read(probs[0]));
        if tx != TX_4X4 && limit >= TX_16X16 {
            tx += u8::from(self.b.read(probs[1]));
            if tx != TX_8X8 && limit >= TX_32X32 {
                tx += u8::from(self.b.read(probs[2]));
            }
        }
        if self.s.collect {
            let t = usize::from(tx);
            match limit {
                TX_8X8 => self.f.counts.tx8[ctx][t] += 1,
                TX_16X16 => self.f.counts.tx16[ctx][t] += 1,
                _ => self.f.counts.tx32[ctx][t] += 1,
            }
        }
        tx
    }

    // Inter mode info -----------------------------------------------------

    fn inter_block_info(&mut self, b: &mut Blk) {
        let s = self.s;
        let h = s.h;
        let seg = usize::from(b.seg);
        // References.
        if h.seg.enabled && h.seg.feature_on[seg][2] {
            b.refs = [h.seg.feature_data[seg][2] as u8, INTRA_FRAME];
        } else {
            let mode = if h.reference_mode == REFERENCE_SELECT {
                let ctx = ctx::reference_mode_ctx(self.above_nb(b), self.left_nb(b), h.comp_fixed);
                let v = self.b.read(s.fc.comp_inter[ctx]);
                if s.collect {
                    self.f.counts.comp_inter[ctx][usize::from(v)] += 1;
                }
                u8::from(v)
            } else {
                h.reference_mode
            };
            if mode == COMPOUND_REFERENCE {
                let ctx = ctx::comp_ref_ctx(self.above_nb(b), self.left_nb(b), h);
                let bit = self.b.read(s.fc.comp_ref[ctx]);
                if s.collect {
                    self.f.counts.comp_ref[ctx][usize::from(bit)] += 1;
                }
                let fixed = usize::from(h.sign_bias[usize::from(h.comp_fixed)]);
                b.refs[fixed] = h.comp_fixed;
                b.refs[fixed ^ 1] = h.comp_var[usize::from(bit)];
            } else {
                let c0 = ctx::single_p1(self.above_nb(b), self.left_nb(b));
                let bit0 = self.b.read(s.fc.single_ref[c0][0]);
                if s.collect {
                    self.f.counts.single_ref[c0][0][usize::from(bit0)] += 1;
                }
                b.refs[0] = if bit0 {
                    let c1 = ctx::single_p2(self.above_nb(b), self.left_nb(b));
                    let bit1 = self.b.read(s.fc.single_ref[c1][1]);
                    if s.collect {
                        self.f.counts.single_ref[c1][1][usize::from(bit1)] += 1;
                    }
                    if bit1 {
                        geom::ALTREF_FRAME
                    } else {
                        geom::GOLDEN_FRAME
                    }
                } else {
                    geom::LAST_FRAME
                };
                b.refs[1] = INTRA_FRAME;
            }
        }
        let compound = b.refs[1] > INTRA_FRAME;
        for r in &b.refs[..1 + usize::from(compound)] {
            match s.ref_status.get(usize::from(*r).wrapping_sub(1)).copied() {
                Some(0) => {}
                Some(2) => self.unsupported = true,
                _ => self.corrupt = true,
            }
        }
        // The mode context from the first two candidates.
        let search = &MV_REF[usize::from(b.bsize)];
        let mut counter = 0i32;
        for pos in &search[..2] {
            if let Some(n) = self.near(b, *pos) {
                counter += MODE_COUNTER[usize::from(n.mode)];
            }
        }
        let mctx = usize::from(COUNTER_CTX[counter as usize]).min(6);
        let seg_skip = h.seg.enabled && h.seg.feature_on[seg][3];
        if seg_skip {
            b.mode = ZERO;
            if b.bsize < BLOCK_8X8 {
                self.corrupt = true;
            }
        } else if b.bsize >= BLOCK_8X8 {
            b.mode = self.read_inter_mode(mctx);
        }
        b.interp = if h.interp == geom::SWITCHABLE {
            let l = self.left_nb(b).map(|n| n.interp);
            let a = self.above_nb(b).map(|n| n.interp);
            let ctx = ctx::interp_ctx(l, a);
            let k = self.b.tree(&prob::INTERP_TREE, &s.fc.interp[ctx]);
            if s.collect {
                self.f.counts.interp[ctx][usize::from(k)] += 1;
            }
            k
        } else {
            h.interp
        };
        let nref = 1 + usize::from(compound);
        if b.bsize < BLOCK_8X8 {
            let (step_w, step_h) = sub_steps(b.bsize);
            let mut best_ref = [Mv::ZERO; 2];
            let mut got_new = false;
            let mut last = ZERO;
            for idy in (0..2).step_by(step_h) {
                for idx in (0..2).step_by(step_w) {
                    let j = idy * 2 + idx;
                    let bm = if seg_skip {
                        ZERO
                    } else {
                        self.read_inter_mode(mctx)
                    };
                    let mut near = [Mv::ZERO; 2];
                    if bm == NEAREST || bm == NEAR {
                        for (r, slot) in near.iter_mut().enumerate().take(nref) {
                            *slot = self.sub_candidate(b, bm, j, r);
                        }
                    } else if bm == NEW && !got_new {
                        for (r, slot) in best_ref.iter_mut().enumerate().take(nref) {
                            let (list, _) = self.find_mv(b, NEW, b.refs[r], -1);
                            *slot = lower_precision(list[0], h.allow_hp);
                        }
                        got_new = true;
                    }
                    let mut mv = [Mv::ZERO; 2];
                    self.assign(bm, &mut mv, &best_ref, &near, nref);
                    b.bmv[j] = mv;
                    if step_h == 2 {
                        b.bmv[j + 2] = mv;
                    }
                    if step_w == 2 {
                        b.bmv[j + 1] = mv;
                    }
                    last = bm;
                }
            }
            b.mode = last;
            b.mv = b.bmv[3];
            b.bmode = [last; 4];
        } else {
            let mut best = [Mv::ZERO; 2];
            if b.mode != ZERO {
                for (r, slot) in best.iter_mut().enumerate().take(nref) {
                    let (list, count) = self.find_mv(b, b.mode, b.refs[r], -1);
                    *slot = lower_precision(list[count - 1], h.allow_hp);
                }
            }
            let mut mv = [Mv::ZERO; 2];
            self.assign(b.mode, &mut mv, &best, &best, nref);
            b.mv = mv;
            b.bmv = [mv; 4];
            b.bmode = [b.mode; 4];
        }
    }

    fn read_inter_mode(&mut self, ctx: usize) -> u8 {
        let p = &self.s.fc.inter_mode[ctx];
        // INTER_TREE: ZERO first, then NEAREST, then NEAR / NEW.
        let k = if !self.b.read(p[0]) {
            2
        } else if !self.b.read(p[1]) {
            0
        } else if !self.b.read(p[2]) {
            1
        } else {
            3
        };
        if self.s.collect {
            self.f.counts.inter_mode[ctx][k] += 1;
        }
        NEAREST + k as u8
    }

    fn assign(&mut self, mode: u8, mv: &mut [Mv; 2], best: &[Mv; 2], near: &[Mv; 2], nref: usize) {
        match mode {
            NEW => {
                for r in 0..nref {
                    let d = self.read_mv(best[r]);
                    let row = i32::from(best[r].row) + i32::from(d.row);
                    let col = i32::from(best[r].col) + i32::from(d.col);
                    if !(MV_LOW < row && row < MV_UPP && MV_LOW < col && col < MV_UPP) {
                        self.corrupt = true;
                    }
                    mv[r] = Mv {
                        row: row as i16,
                        col: col as i16,
                    };
                }
            }
            NEAR | NEAREST => {
                mv[..nref].copy_from_slice(&near[..nref]);
            }
            _ => *mv = [Mv::ZERO; 2],
        }
    }

    fn read_mv(&mut self, reference: Mv) -> Mv {
        let fc = self.s.fc;
        let joint = self.b.tree(&prob::JOINT_TREE, &fc.mv.joints);
        let hp = self.s.h.allow_hp
            && reference.row.unsigned_abs() < 64
            && reference.col.unsigned_abs() < 64;
        let mut d = Mv::ZERO;
        if joint == 2 || joint == 3 {
            d.row = self.read_mv_comp(0, hp) as i16;
        }
        if joint == 1 || joint == 3 {
            d.col = self.read_mv_comp(1, hp) as i16;
        }
        if self.s.collect {
            count_mv(&mut self.f.counts, d);
        }
        d
    }

    fn read_mv_comp(&mut self, which: usize, use_hp: bool) -> i32 {
        let c = &self.s.fc.mv.comp[which];
        let sign = self.b.read(c.sign);
        let class = usize::from(self.b.tree(&prob::CLASS_TREE, &c.classes));
        let (base, bits) = if class == 0 {
            (0, i32::from(self.b.read(c.class0)))
        } else {
            let mut v = 0i32;
            for i in 0..class {
                v |= i32::from(self.b.read(c.bits[i])) << i;
            }
            (2i32 << (class + 2), v)
        };
        let fp = if class == 0 {
            &c.class0_fp[bits as usize]
        } else {
            &c.fp
        };
        let frac = i32::from(self.b.tree(&prob::FP_TREE, fp));
        let hp = if use_hp {
            i32::from(self.b.read(if class == 0 { c.class0_hp } else { c.hp }))
        } else {
            1
        };
        let mag = base + ((bits << 3) | (frac << 1) | hp) + 1;
        if sign { -mag } else { mag }
    }

    /// The block at `pos` from `b`, if inside the tile and the frame, when
    /// it is in the row just above or the column just left.
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

    /// libvpx `dec_find_mv_refs`: up to two candidate vectors for `frame`,
    /// stopping at the first for any mode but NEAR. `block` is the
    /// sub-block for blocks under 8×8, or -1.
    fn find_mv(&self, b: &Blk, mode: u8, frame: u8, block: i32) -> ([Mv; 2], usize) {
        let s = self.s;
        let bias = &s.h.sign_bias;
        let early = mode != NEAR;
        let mut list = [Mv::ZERO; 2];
        let mut count = 0usize;
        let mut different = false;
        let search = &MV_REF[usize::from(b.bsize)];
        // Returns true when the search is done.
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
            while i < 8 {
                if let Some(c) = self.cell(b, search[i]) {
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
                i += 1;
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
        // Clamp to 16 samples (MV_BORDER) beyond the frame.
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

    /// libvpx `append_sub8x8_mvs_for_idx`.
    fn sub_candidate(&self, b: &Blk, bm: u8, j: usize, r: usize) -> Mv {
        let frame = b.refs[r];
        if j == 0 {
            let (list, count) = self.find_mv(b, bm, frame, 0);
            return list[count - 1];
        }
        if bm == NEAREST {
            return b.bmv[if j == 3 { 2 } else { 0 }][r];
        }
        let (list, _) = self.find_mv(b, bm, frame, j as i32);
        if j == 3 {
            let cands = [b.bmv[1][r], b.bmv[0][r], list[0], list[1]];
            for c in cands {
                if b.bmv[2][r] != c {
                    return c;
                }
            }
            Mv::ZERO
        } else {
            for c in list {
                if b.bmv[0][r] != c {
                    return c;
                }
            }
            Mv::ZERO
        }
    }

    // Residual ------------------------------------------------------------

    fn plane_ss(&self, plane: usize) -> (usize, usize) {
        if plane == 0 {
            (0, 0)
        } else {
            (self.s.h.ss_x as usize, self.s.h.ss_y as usize)
        }
    }

    fn uv_tx(&self, b: &Blk) -> u8 {
        let h = self.s.h;
        if h.ss_x == 1 && h.ss_y == 1 {
            geom::uv_tx_420(b.bsize, b.tx)
        } else {
            geom::uv_tx(b.bsize, b.tx, h.ss_x, h.ss_y)
        }
    }

    fn reset_nz(&mut self, b: &Blk) {
        for plane in 0..3 {
            let (ssx, ssy) = self.plane_ss(plane);
            let w = (b.bw * 2) >> ssx;
            let hh = (b.bh * 2) >> ssy;
            let a = ((b.col - self.col0) * 2) >> ssx;
            let l = ((b.row * 2) & 15) >> ssy;
            let above = &mut self.f.above_nz[plane];
            let end = (a + w).min(above.len());
            above[a..end].fill(0);
            let left = &mut self.left_nz[plane];
            left[l..(l + hh).min(16)].fill(0);
        }
    }

    /// Reads the coefficients of every tx block of a block, in the order
    /// libvpx reads them; returns the sum of their ends of block.
    fn residual(&mut self, b: &Blk) -> usize {
        let intra = b.refs[0] == INTRA_FRAME;
        let lossless = self.s.h.lossless;
        let mut total = 0;
        for plane in 0..3 {
            let tx = if plane == 0 { b.tx } else { self.uv_tx(b) };
            let (_, _, vw, vh) = self.visible(b, plane);
            let step = 1usize << tx;
            let mut r = 0;
            while r < vh {
                let mut c = 0;
                while c < vw {
                    let kind = if intra && plane == 0 && !lossless && tx != TX_32X32 {
                        let mode = if b.bsize < BLOCK_8X8 {
                            b.bmode[(r << 1) + c]
                        } else {
                            b.mode
                        };
                        geom::intra_tx_type(mode)
                    } else {
                        0
                    };
                    let (eob, _) = self.tx_coefs(b, plane, tx, kind, r, c, (vw, vh));
                    total += eob;
                    c += step;
                }
                r += step;
            }
        }
        total
    }

    /// The tx blocks of a plane inside the frame: (blocks wide, high).
    fn visible(&self, b: &Blk, plane: usize) -> (usize, usize, usize, usize) {
        let (ssx, ssy) = self.plane_ss(plane);
        let n4w = (b.bw * 2) >> ssx;
        let n4h = (b.bh * 2) >> ssy;
        let right = (self.s.mi_cols as i32 - b.bw as i32 - b.col as i32) * 64;
        let bottom = (self.s.mi_rows as i32 - b.bh as i32 - b.row as i32) * 64;
        let vw = if right >= 0 {
            n4w
        } else {
            (n4w as i32 + (right >> (5 + ssx))).max(0) as usize
        };
        let vh = if bottom >= 0 {
            n4h
        } else {
            (n4h as i32 + (bottom >> (5 + ssy))).max(0) as usize
        };
        (n4w, n4h, vw, vh)
    }

    /// Reads the coefficients of one tx block and sets its contexts.
    /// Returns the end of block and the extent of the nonzero rows and
    /// columns.
    #[allow(clippy::too_many_arguments)]
    fn tx_coefs(
        &mut self,
        b: &Blk,
        plane: usize,
        tx: u8,
        kind: u8,
        r4: usize,
        c4: usize,
        vis: (usize, usize),
    ) -> (usize, (usize, usize)) {
        let (ssx, ssy) = self.plane_ss(plane);
        let n = 1usize << tx;
        let a = (((b.col - self.col0) * 2) >> ssx) + c4;
        let l = (((b.row * 2) & 15) >> ssy) + r4;
        let above = &self.f.above_nz[plane];
        let left = &self.left_nz[plane];
        let anz = above[a..(a + n).min(above.len())].iter().any(|&v| v != 0);
        let lnz = left[l..(l + n).min(16)].iter().any(|&v| v != 0);
        let ctx = usize::from(anz) + usize::from(lnz);
        let h = self.s.h;
        let seg = usize::from(b.seg);
        let dq = if plane == 0 {
            h.y_dq[seg]
        } else {
            h.uv_dq[seg]
        };
        let inter = b.refs[0] != INTRA_FRAME;
        let n_before = self.out.pos.len();
        let (eob, nz) = self.decode_coefs(tx, plane > 0, inter, dq, ctx, kind);
        self.out.txs.push(TxRec {
            eob: eob as u16,
            n: (self.out.pos.len() - n_before) as u16,
            nzr: nz.0 as u8,
            nzc: nz.1 as u8,
        });
        // Contexts, cleared past the edge of the frame.
        let on = u8::from(eob > 0);
        let (vw, vh) = vis;
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
        (eob, nz)
    }

    /// libvpx `decode_coefs`, into the syntax's coefficients: positions
    /// transposed and values dequantized.
    fn decode_coefs(
        &mut self,
        tx: u8,
        uv: bool,
        inter: bool,
        dq: [i16; 2],
        mut ctx: usize,
        kind: u8,
    ) -> (usize, (usize, usize)) {
        let (scan, nb) = scan_pair(tx, kind);
        let log2n = 2 + usize::from(tx);
        let mask = (1usize << log2n) - 1;
        let max_eob = 16usize << (2 * tx);
        let probs = &self.s.fc.coef[usize::from(tx)][usize::from(uv)][usize::from(inter)];
        let bands: &[u8] = if tx == 0 { &BAND_4X4 } else { &BAND_8X8_PLUS };
        let shift = u32::from(tx == 3);
        let bd = self.s.bd;
        let collect = self.s.collect;
        let (ti, pi, ri) = (usize::from(tx), usize::from(uv), usize::from(inter));
        let cache = &mut self.cache;
        let mut dqv = i32::from(dq[0]);
        let mut c = 0usize;
        let (mut max_r, mut max_c) = (0usize, 0usize);
        // The decoder works on a copy in registers, written back at the end.
        let mut local = *self.b;
        let b = &mut local;
        let out_pos = &mut self.out.pos;
        let out_val = &mut self.out.val;
        let counts = &mut self.f.counts;
        while c < max_eob {
            let mut band = usize::from(bands[c]);
            let mut p = &probs[band][ctx];
            if collect {
                counts.eob[ti][pi][ri][band][ctx] += 1;
            }
            if !b.read(p[0]) {
                if collect {
                    counts.coef[ti][pi][ri][band][ctx][3] += 1;
                }
                break;
            }
            while !b.read(p[1]) {
                if collect {
                    counts.coef[ti][pi][ri][band][ctx][0] += 1;
                }
                dqv = i32::from(dq[1]);
                cache[usize::from(scan[c])] = 0;
                c += 1;
                if c >= max_eob {
                    *self.b = local;
                    return (c, (max_r + 1, max_c + 1));
                }
                ctx = (1
                    + usize::from(cache[usize::from(nb[2 * c])])
                    + usize::from(cache[usize::from(nb[2 * c + 1])]))
                    >> 1;
                band = usize::from(bands[c]);
                p = &probs[band][ctx];
            }
            let v;
            if !b.read(p[2]) {
                if collect {
                    counts.coef[ti][pi][ri][band][ctx][1] += 1;
                }
                cache[usize::from(scan[c])] = 1;
                v = dqv >> shift;
            } else {
                if collect {
                    counts.coef[ti][pi][ri][band][ctx][2] += 1;
                }
                let pp = &PARETO8[usize::from(p[2]) - 1];
                let val: i32;
                if b.read(pp[0]) {
                    if b.read(pp[3]) {
                        cache[usize::from(scan[c])] = 5;
                        if b.read(pp[5]) {
                            if b.read(pp[7]) {
                                let skip = match bd {
                                    12 => 0,
                                    10 => 2,
                                    _ => 4,
                                };
                                let mut x = 0i32;
                                for &q in &CAT6_HIGH[skip..] {
                                    x = (x << 1) | i32::from(b.read(q));
                                }
                                val = 67 + x;
                            } else {
                                val = 35 + read_bits(b, &[180, 157, 141, 134, 130]);
                            }
                        } else if b.read(pp[6]) {
                            val = 19 + read_bits(b, &[176, 155, 140, 135]);
                        } else {
                            val = 11 + read_bits(b, &[173, 148, 140]);
                        }
                    } else {
                        cache[usize::from(scan[c])] = 4;
                        if b.read(pp[4]) {
                            val = 7 + read_bits(b, &[165, 145]);
                        } else {
                            val = 5 + read_bits(b, &[159]);
                        }
                    }
                } else if b.read(pp[1]) {
                    cache[usize::from(scan[c])] = 3;
                    val = 3 + i32::from(b.read(pp[2]));
                } else {
                    cache[usize::from(scan[c])] = 2;
                    val = 2;
                }
                v = ((i64::from(val) * i64::from(dqv)) >> shift) as i32;
            }
            let rc = usize::from(scan[c]);
            let (r, col) = (rc >> log2n, rc & mask);
            out_pos.push(((col << log2n) | r) as u16);
            out_val.push(if b.bit() { -v } else { v });
            max_r = max_r.max(r);
            max_c = max_c.max(col);
            c += 1;
            if c < max_eob {
                ctx = (1
                    + usize::from(cache[usize::from(nb[2 * c])])
                    + usize::from(cache[usize::from(nb[2 * c + 1])]))
                    >> 1;
            }
            dqv = i32::from(dq[1]);
        }
        *self.b = local;
        (c, (max_r + 1, max_c + 1))
    }
}

const CAT6_HIGH: [u8; 18] = {
    let mut t = [255u8; 18];
    let mut i = 0;
    while i < 14 {
        t[4 + i] = CAT6_PROB[i];
        i += 1;
    }
    t
};

#[inline(always)]
fn read_bits(b: &mut Bool<'_>, probs: &[u8]) -> i32 {
    let mut v = 0i32;
    for &p in probs {
        v = (v << 1) | i32::from(b.read(p));
    }
    v
}

fn sub_steps(bsize: u8) -> (usize, usize) {
    match bsize {
        BLOCK_4X8 => (1, 2),
        BLOCK_8X4 => (2, 1),
        BLOCK_4X4 => (1, 1),
        _ => (2, 2),
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

fn count_mv(counts: &mut Counts, d: Mv) {
    let joint = if d.row == 0 {
        usize::from(d.col != 0)
    } else if d.col == 0 {
        2
    } else {
        3
    };
    counts.joints[joint] += 1;
    if joint == 2 || joint == 3 {
        count_comp(counts, 0, i32::from(d.row));
    }
    if joint == 1 || joint == 3 {
        count_comp(counts, 1, i32::from(d.col));
    }
}

fn count_comp(counts: &mut Counts, comp: usize, value: i32) {
    counts.mv_sign[comp][usize::from(value < 0)] += 1;
    let z = value.unsigned_abs() - 1;
    let class = if z >= 8192 {
        10
    } else if z < 16 {
        0
    } else {
        (z >> 3).ilog2() as usize
    };
    let base = if class == 0 { 0 } else { 2u32 << (class + 2) };
    let o = z - base;
    counts.mv_class[comp][class] += 1;
    let d = (o >> 3) as usize;
    let f = ((o >> 1) & 3) as usize;
    let e = (o & 1) as usize;
    if class == 0 {
        counts.mv_class0[comp][d] += 1;
        counts.mv_class0_fp[comp][d][f] += 1;
        counts.mv_class0_hp[comp][e] += 1;
    } else {
        for i in 0..class {
            counts.mv_bits[comp][i][(d >> i) & 1] += 1;
        }
        counts.mv_fp[comp][f] += 1;
        counts.mv_hp[comp][e] += 1;
    }
}

fn scan_pair(tx: u8, kind: u8) -> (&'static [u16], &'static [u16]) {
    match (tx, kind) {
        (TX_4X4, 1) => (&SCAN_ROW_4X4, &NEIGHBORS_ROW_4X4),
        (TX_4X4, 2) => (&SCAN_COL_4X4, &NEIGHBORS_COL_4X4),
        (TX_4X4, _) => (&SCAN_DEFAULT_4X4, &NEIGHBORS_DEFAULT_4X4),
        (TX_8X8, 1) => (&SCAN_ROW_8X8, &NEIGHBORS_ROW_8X8),
        (TX_8X8, 2) => (&SCAN_COL_8X8, &NEIGHBORS_COL_8X8),
        (TX_8X8, _) => (&SCAN_DEFAULT_8X8, &NEIGHBORS_DEFAULT_8X8),
        (TX_16X16, 1) => (&SCAN_ROW_16X16, &NEIGHBORS_ROW_16X16),
        (TX_16X16, 2) => (&SCAN_COL_16X16, &NEIGHBORS_COL_16X16),
        (TX_16X16, _) => (&SCAN_DEFAULT_16X16, &NEIGHBORS_DEFAULT_16X16),
        _ => (&SCAN_DEFAULT_32X32, &NEIGHBORS_DEFAULT_32X32),
    }
}

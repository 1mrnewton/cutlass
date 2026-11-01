//! Inter prediction.
//!
//! Motion-vector candidates follow the find-MV-stack process. The sample
//! predictor is the two-tap separable subpel filter from the decoding process.
//! A global model that is not identity supplies the `GLOBALMV` vector.
//! A rotation-zoom or affine model warps that block when the shear is valid.
//! Interintra blends a single reference with an intra predictor.
//! Compound prediction averages two references, weights them by order-hint
//! distance, or blends them through a wedge or difference mask. A local warp
//! fits an affine model to neighboring motion, and overlapped motion blends
//! the neighboring predictors into the block. A frame that asks for reference
//! motion vectors also searches the field projected from earlier frames.

use cutlass_decode::CodecError;

use crate::sample::{Pix, clip_sample, depth_shift};

use crate::dims::{
    ADST_ADST, ADST_DCT, ADST_FLIPADST, DCT_ADST, DCT_DCT, DCT_FLIPADST, FLIPADST_ADST,
    FLIPADST_DCT, FLIPADST_FLIPADST, H_ADST, H_DCT, H_FLIPADST, IDTX, NUM_4X4_HIGH, NUM_4X4_WIDE,
    TX_16X16, TX_32X32, TX_SIZE_SQR, TX_SIZE_SQR_UP, V_ADST, V_DCT, V_FLIPADST, block_height,
    block_width,
};
use crate::tables::{DIV_LUT, SUBPEL_FILTERS, WARPED_FILTERS};

pub const NEARESTMV: u8 = 14;
pub const NEARMV: u8 = 15;
pub const GLOBALMV: u8 = 16;
pub const NEWMV: u8 = 17;
pub const NEAREST_NEARESTMV: u8 = 18;
pub const NEAR_NEARMV: u8 = 19;
pub const NEAREST_NEWMV: u8 = 20;
pub const NEW_NEARESTMV: u8 = 21;
pub const NEAR_NEWMV: u8 = 22;
pub const NEW_NEARMV: u8 = 23;
pub const GLOBAL_GLOBALMV: u8 = 24;
pub const NEW_NEWMV: u8 = 25;
pub const LAST_FRAME: i8 = 1;
pub const LAST2_FRAME: i8 = 2;
pub const LAST3_FRAME: i8 = 3;
pub const GOLDEN_FRAME: i8 = 4;
pub const BWDREF_FRAME: i8 = 5;
pub const ALTREF2_FRAME: i8 = 6;
pub const ALTREF_FRAME: i8 = 7;
pub const NONE_FRAME: i8 = -1;
pub const INTRA_FRAME: i8 = 0;
pub const GM_IDENTITY: u8 = 0;
pub const GM_TRANSLATION: u8 = 1;
pub const MOTION_SIMPLE: u8 = 0;
pub const MOTION_OBMC: u8 = 1;
pub const MOTION_LOCAL_WARP: u8 = 2;
const REF_CAT_LEVEL: i32 = 640;
const MV_BORDER: i32 = 128;
const STACK: usize = 8;
const NO_SCALE: i64 = 1 << 14;

pub const SIZE_GROUP: [usize; 22] = [
    0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 2, 2,
];

const INTER_INV1: [u8; 16] = [
    IDTX,
    V_DCT,
    H_DCT,
    V_ADST,
    H_ADST,
    V_FLIPADST,
    H_FLIPADST,
    DCT_DCT,
    ADST_DCT,
    DCT_ADST,
    FLIPADST_DCT,
    DCT_FLIPADST,
    ADST_ADST,
    FLIPADST_FLIPADST,
    ADST_FLIPADST,
    FLIPADST_ADST,
];
const INTER_INV2: [u8; 12] = [
    IDTX,
    V_DCT,
    H_DCT,
    DCT_DCT,
    ADST_DCT,
    DCT_ADST,
    FLIPADST_DCT,
    DCT_FLIPADST,
    ADST_ADST,
    FLIPADST_FLIPADST,
    ADST_FLIPADST,
    FLIPADST_ADST,
];
const INTER_INV3: [u8; 2] = [IDTX, DCT_DCT];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Motion {
    pub row: i32,
    pub col: i32,
}

pub struct RefView<'a, S: Pix = u8> {
    pub y: &'a [S],
    pub u: &'a [S],
    pub v: &'a [S],
    pub width: usize,
    pub height: usize,
    pub frame_type: u8,
    pub mi_cols: usize,
    pub mi_rows: usize,
    /// Order hint this reference saw for each frame type. Index 0 is unused.
    pub saved_hints: [u32; 8],
    /// Reference named by each 4×4 of the saved motion field, or `NONE_FRAME`.
    pub mf_refs: &'a [i8],
    pub mf_mvs: &'a [Motion],
}

pub struct MotionGrid<'a> {
    pub cols: usize,
    pub rows: usize,
    /// Spatial candidates stay inside the tile. A neighbor in another tile is unavailable.
    pub tile_row: usize,
    pub tile_col: usize,
    pub tile_row_end: usize,
    pub tile_col_end: usize,
    pub inter: &'a [u8],
    pub refs: &'a [i8],
    pub refs1: &'a [i8],
    pub mvs: &'a [Motion],
    pub mvs1: &'a [Motion],
    pub modes: &'a [u8],
    pub sizes: &'a [u8],
}

pub struct BlockPlace {
    pub row: usize,
    pub col: usize,
    pub bw4: usize,
    pub bh4: usize,
    pub bsize: usize,
    pub reference: i8,
    pub reference1: i8,
    pub compound: bool,
    pub global1: Motion,
    pub affine1: bool,
    /// `RefFrameSignBias`, indexed by reference frame.
    pub bias: [bool; 8],
}

pub struct Candidates {
    pub mv: [Motion; STACK],
    pub mv1: [Motion; STACK],
    pub found: usize,
    pub new_ctx: usize,
    pub ref_ctx: usize,
    /// Context for the single-reference global-motion symbol.
    pub zero_ctx: usize,
    pub drl: [usize; STACK],
}

/// The mode one compound list uses when the block mode names both lists.
#[must_use]
pub fn list_mode(y_mode: u8, list: usize) -> u8 {
    if list == 0 {
        if y_mode < NEAREST_NEARESTMV {
            return y_mode;
        }
        if matches!(y_mode, NEW_NEWMV | NEW_NEARESTMV | NEW_NEARMV) {
            return NEWMV;
        }
        if matches!(y_mode, NEAREST_NEARESTMV | NEAREST_NEWMV) {
            return NEARESTMV;
        }
        if matches!(y_mode, NEAR_NEARMV | NEAR_NEWMV) {
            return NEARMV;
        }
        return GLOBALMV;
    }
    if matches!(y_mode, NEW_NEWMV | NEAREST_NEWMV | NEAR_NEWMV) {
        return NEWMV;
    }
    if matches!(y_mode, NEAREST_NEARESTMV | NEW_NEARESTMV) {
        return NEARESTMV;
    }
    if matches!(y_mode, NEAR_NEARMV | NEW_NEARMV) {
        return NEARMV;
    }
    GLOBALMV
}

#[must_use]
pub fn has_new_mv(mode: u8) -> bool {
    matches!(
        mode,
        NEWMV | NEW_NEWMV | NEAR_NEWMV | NEW_NEARMV | NEAREST_NEWMV | NEW_NEARESTMV
    )
}

#[must_use]
pub fn has_near_mv(mode: u8) -> bool {
    matches!(mode, NEARMV | NEAR_NEARMV | NEAR_NEWMV | NEW_NEARMV)
}

struct Line {
    delta_row: i32,
    delta_col: i32,
    end4: usize,
    step16: bool,
    along_row: bool,
}

struct Search {
    mv: [Motion; STACK],
    mv1: [Motion; STACK],
    weight: [i32; STACK],
    found: usize,
    new_count: usize,
    found_match: bool,
    global: Motion,
    global1: Motion,
    reference: i8,
    reference1: i8,
    hp: bool,
    integer: bool,
    /// The current reference uses a rotation-zoom or affine model.
    affine: bool,
    affine1: bool,
    compound: bool,
    bias: [bool; 8],
    zero_ctx: usize,
}

/// Transform set for an inter block. `0` is DCT only.
#[must_use]
pub fn inter_tx_set(tx: usize, reduced: bool) -> usize {
    let up = TX_SIZE_SQR_UP[tx];
    if up > TX_32X32 {
        return 0;
    }
    if reduced || up == TX_32X32 {
        return 3;
    }
    if TX_SIZE_SQR[tx] == TX_16X16 {
        return 2;
    }
    1
}

#[must_use]
pub fn inter_tx_type(set: usize, symbol: u32) -> Option<u8> {
    let table: &[u8] = match set {
        1 => &INTER_INV1,
        2 => &INTER_INV2,
        3 => &INTER_INV3,
        _ => return Some(DCT_DCT),
    };
    table.get(symbol as usize).copied()
}

#[must_use]
pub fn inter_set_allows(set: usize, kind: u8) -> bool {
    const TABLE: [[u8; 16]; 4] = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ];
    TABLE
        .get(set)
        .and_then(|row| row.get(kind as usize))
        .copied()
        == Some(1)
}

#[must_use]
pub fn find_stack(
    grid: &MotionGrid<'_>,
    at: &BlockPlace,
    global: Motion,
    hp: bool,
    integer: bool,
    affine: bool,
    field: Option<&MotionField>,
) -> Candidates {
    let mut search = Search {
        mv: [Motion::default(); STACK],
        mv1: [Motion::default(); STACK],
        weight: [0; STACK],
        found: 0,
        new_count: 0,
        found_match: false,
        global,
        global1: at.global1,
        reference: at.reference,
        reference1: at.reference1,
        hp,
        integer,
        affine,
        affine1: at.affine1,
        compound: at.compound,
        bias: at.bias,
        zero_ctx: 0,
    };
    search.scan_row(grid, at, -1);
    let mut above = search.found_match;
    search.found_match = false;
    search.scan_col(grid, at, -1);
    let mut left = search.found_match;
    search.found_match = false;
    if at.bw4.max(at.bh4) <= 16 {
        search.scan_point(grid, at, -1, at.bw4 as i32);
        if search.found_match {
            above = true;
        }
    }
    let close = usize::from(above) + usize::from(left);
    let nearest = search.found;
    let num_new = search.new_count;
    for weight in search.weight.iter_mut().take(nearest) {
        *weight += REF_CAT_LEVEL;
    }
    search.scan_far(grid, at, &mut above, &mut left, field);
    let total = usize::from(above) + usize::from(left);
    search.sort(0, nearest);
    search.sort(nearest, search.found);
    if search.found < 2 {
        search.extra(grid, at);
    }
    search.clamp(grid, at);
    let (new_ctx, ref_ctx) = contexts(close, total, num_new);
    let mut drl = [0usize; STACK];
    for (idx, slot) in drl.iter_mut().enumerate().take(search.found) {
        *slot = drl_context(&search.weight, search.found, idx);
    }
    Candidates {
        mv: search.mv,
        mv1: search.mv1,
        found: search.found,
        new_ctx,
        ref_ctx,
        zero_ctx: search.zero_ctx,
        drl,
    }
}

fn contexts(close: usize, total: usize, num_new: usize) -> (usize, usize) {
    if close == 0 {
        (total.min(1), total)
    } else if close == 1 {
        (3 - num_new.min(1), 2 + total)
    } else {
        (5 - num_new.min(1), 5)
    }
}

fn drl_context(weight: &[i32], found: usize, idx: usize) -> usize {
    if idx + 1 >= found {
        return 0;
    }
    let w0 = weight[idx];
    let w1 = weight[idx + 1];
    if w0 >= REF_CAT_LEVEL {
        usize::from(w1 < REF_CAT_LEVEL)
    } else {
        2
    }
}

struct CompoundScratch<'a> {
    same: &'a mut [[Motion; 2]; 2],
    same_n: &'a mut [usize; 2],
    other: &'a mut [[Motion; 2]; 2],
    other_n: &'a mut [usize; 2],
}

impl Search {
    fn scan_far(
        &mut self,
        grid: &MotionGrid<'_>,
        at: &BlockPlace,
        above: &mut bool,
        left: &mut bool,
        field: Option<&MotionField>,
    ) {
        if let Some(field) = field {
            self.temporal(grid, at, field);
        }
        self.found_match = false;
        self.scan_point(grid, at, -1, -1);
        if self.found_match {
            *above = true;
        }
        self.found_match = false;
        self.scan_row(grid, at, -3);
        if self.found_match {
            *above = true;
        }
        self.found_match = false;
        self.scan_col(grid, at, -3);
        if self.found_match {
            *left = true;
        }
        self.found_match = false;
        if at.bh4 > 1 {
            self.scan_row(grid, at, -5);
            if self.found_match {
                *above = true;
            }
            self.found_match = false;
        }
        if at.bw4 > 1 {
            self.scan_col(grid, at, -5);
            if self.found_match {
                *left = true;
            }
        }
    }

    fn scan_row(&mut self, grid: &MotionGrid<'_>, at: &BlockPlace, mut delta_row: i32) {
        let mut delta_col = 0i32;
        let end4 = at.bw4.min(grid.cols.saturating_sub(at.col)).min(16);
        let step16 = at.bw4 >= 16;
        if delta_row.abs() > 1 {
            delta_row += (at.row & 1) as i32;
            delta_col = 1 - (at.col & 1) as i32;
        }
        self.scan_line(
            grid,
            at,
            Line {
                delta_row,
                delta_col,
                end4,
                step16,
                along_row: true,
            },
        );
    }

    fn scan_col(&mut self, grid: &MotionGrid<'_>, at: &BlockPlace, mut delta_col: i32) {
        let mut delta_row = 0i32;
        let end4 = at.bh4.min(grid.rows.saturating_sub(at.row)).min(16);
        let step16 = at.bh4 >= 16;
        if delta_col.abs() > 1 {
            delta_row = 1 - (at.row & 1) as i32;
            delta_col += (at.col & 1) as i32;
        }
        self.scan_line(
            grid,
            at,
            Line {
                delta_row,
                delta_col,
                end4,
                step16,
                along_row: false,
            },
        );
    }

    fn scan_line(&mut self, grid: &MotionGrid<'_>, at: &BlockPlace, line: Line) {
        let mut i = 0usize;
        while i < line.end4 {
            let mv_row = at.row as i32 + line.delta_row + if line.along_row { 0 } else { i as i32 };
            let mv_col = at.col as i32 + line.delta_col + if line.along_row { i as i32 } else { 0 };
            if !grid.inside(mv_row, mv_col) {
                break;
            }
            let size = grid.size(mv_row, mv_col);
            let span = if line.along_row {
                NUM_4X4_WIDE[size]
            } else {
                NUM_4X4_HIGH[size]
            };
            let limit = if line.along_row { at.bw4 } else { at.bh4 };
            let mut len = limit.min(span);
            let delta = if line.along_row {
                line.delta_row
            } else {
                line.delta_col
            };
            if delta.abs() > 1 {
                len = len.max(2);
            }
            if line.step16 {
                len = len.max(4);
            }
            if len == 0 {
                break;
            }
            self.add_ref(grid, mv_row, mv_col, (len as i32) * 2);
            i += len;
        }
    }

    fn scan_point(
        &mut self,
        grid: &MotionGrid<'_>,
        at: &BlockPlace,
        delta_row: i32,
        delta_col: i32,
    ) {
        let row = at.row as i32 + delta_row;
        let col = at.col as i32 + delta_col;
        if grid.inside(row, col) {
            self.add_ref(grid, row, col, 4);
        }
    }

    fn add_ref(&mut self, grid: &MotionGrid<'_>, row: i32, col: i32, weight: i32) {
        if grid.inter_at(row, col) == 0 {
            return;
        }
        if self.compound {
            if grid.ref_at(row, col) == self.reference && grid.ref1_at(row, col) == self.reference1
            {
                self.compound_search(grid, row, col, weight);
            }
            return;
        }
        let refs = [grid.ref_at(row, col), grid.ref1_at(row, col)];
        for (list, cand_ref) in refs.into_iter().enumerate() {
            if cand_ref == self.reference {
                self.search(grid, row, col, list, weight);
            }
        }
    }

    fn search(&mut self, grid: &MotionGrid<'_>, row: i32, col: i32, list: usize, weight: i32) {
        let mode = grid.mode(row, col);
        let size = grid.size(row, col);
        let large = block_width(size).min(block_height(size)) >= 8;
        // A large global-motion neighbor contributes this block's model, not the vector it stored.
        let mv = if self.affine && large && (mode == GLOBALMV || mode == GLOBAL_GLOBALMV) {
            self.global
        } else if list == 0 {
            grid.mv(row, col)
        } else {
            grid.mv1(row, col)
        };
        let mv = lower(mv, self.hp, self.integer);
        if has_new_mv(mode) {
            self.new_count += 1;
        }
        self.found_match = true;
        if let Some(idx) = self.mv.iter().take(self.found).position(|have| *have == mv) {
            self.weight[idx] += weight;
            return;
        }
        if self.found < STACK {
            self.mv[self.found] = mv;
            self.weight[self.found] = weight;
            self.found += 1;
        }
    }

    fn compound_search(&mut self, grid: &MotionGrid<'_>, row: i32, col: i32, weight: i32) {
        let mode = grid.mode(row, col);
        let mut mv0 = grid.mv(row, col);
        let mut mv1 = grid.mv1(row, col);
        if mode == GLOBAL_GLOBALMV {
            if self.affine {
                mv0 = self.global;
            }
            if self.affine1 {
                mv1 = self.global1;
            }
        }
        let mv0 = lower(mv0, self.hp, self.integer);
        let mv1 = lower(mv1, self.hp, self.integer);
        self.found_match = true;
        if let Some(idx) = (0..self.found).find(|idx| self.mv[*idx] == mv0 && self.mv1[*idx] == mv1)
        {
            self.weight[idx] += weight;
        } else if self.found < STACK {
            self.mv[self.found] = mv0;
            self.mv1[self.found] = mv1;
            self.weight[self.found] = weight;
            self.found += 1;
        }
        if has_new_mv(mode) {
            self.new_count += 1;
        }
    }

    /// Partial neighbors, then this block's two global vectors. Section 7.10.2.12.
    fn compound_extra(&mut self, grid: &MotionGrid<'_>, at: &BlockPlace) {
        let w4 = 16.min(at.bw4).min(grid.cols.saturating_sub(at.col));
        let h4 = 16.min(at.bh4).min(grid.rows.saturating_sub(at.row));
        let num = w4.min(h4);
        let mut same = [[Motion::default(); 2]; 2];
        let mut other = [[Motion::default(); 2]; 2];
        let mut same_n = [0usize; 2];
        let mut other_n = [0usize; 2];
        for pass in 0..2 {
            let mut idx = 0usize;
            while idx < num {
                let (row, col) = if pass == 0 {
                    (at.row as i32 - 1, at.col as i32 + idx as i32)
                } else {
                    (at.row as i32 + idx as i32, at.col as i32 - 1)
                };
                if !grid.inside(row, col) {
                    break;
                }
                self.collect_compound_extra(
                    grid,
                    row,
                    col,
                    &mut CompoundScratch {
                        same: &mut same,
                        same_n: &mut same_n,
                        other: &mut other,
                        other_n: &mut other_n,
                    },
                );
                let size = grid.size(row, col);
                idx += if pass == 0 {
                    NUM_4X4_WIDE[size]
                } else {
                    NUM_4X4_HIGH[size]
                };
            }
        }
        let mut combined = [[Motion::default(); 2]; 2];
        for list in 0..2 {
            let mut count = 0usize;
            for motion in same[list].iter().take(same_n[list]) {
                combined[count][list] = *motion;
                count += 1;
            }
            for motion in other[list].iter().take(other_n[list]) {
                if count >= 2 {
                    break;
                }
                combined[count][list] = *motion;
                count += 1;
            }
            let global = if list == 0 { self.global } else { self.global1 };
            while count < 2 {
                combined[count][list] = global;
                count += 1;
            }
        }
        if self.found == 1 {
            let use_second = self.mv[0] == combined[0][0] && self.mv1[0] == combined[0][1];
            let pick = usize::from(use_second);
            self.mv[self.found] = combined[pick][0];
            self.mv1[self.found] = combined[pick][1];
            self.weight[self.found] = 2;
            self.found += 1;
        } else {
            for pair in &combined {
                self.mv[self.found] = pair[0];
                self.mv1[self.found] = pair[1];
                self.weight[self.found] = 2;
                self.found += 1;
            }
        }
    }

    fn collect_compound_extra(
        &self,
        grid: &MotionGrid<'_>,
        row: i32,
        col: i32,
        scratch: &mut CompoundScratch<'_>,
    ) {
        let refs = [self.reference, self.reference1];
        let stored = [grid.ref_at(row, col), grid.ref1_at(row, col)];
        let vectors = [grid.mv(row, col), grid.mv1(row, col)];
        for (cand_list, cand_ref) in stored.into_iter().enumerate() {
            if cand_ref <= INTRA_FRAME {
                continue;
            }
            for (list, list_ref) in refs.iter().enumerate() {
                let mut cand = vectors[cand_list];
                if cand_ref == *list_ref && scratch.same_n[list] < 2 {
                    scratch.same[list][scratch.same_n[list]] = cand;
                    scratch.same_n[list] += 1;
                } else if scratch.other_n[list] < 2 {
                    if self.bias_differs(cand_ref, *list_ref) {
                        cand = negate_mv(cand);
                    }
                    scratch.other[list][scratch.other_n[list]] = cand;
                    scratch.other_n[list] += 1;
                }
            }
        }
    }

    fn bias_differs(&self, candidate: i8, list_ref: i8) -> bool {
        let cand = self
            .bias
            .get(usize::try_from(candidate).unwrap_or(0))
            .copied()
            .unwrap_or(false);
        let list = self
            .bias
            .get(usize::try_from(list_ref).unwrap_or(0))
            .copied()
            .unwrap_or(false);
        cand != list
    }

    fn extra(&mut self, grid: &MotionGrid<'_>, at: &BlockPlace) {
        if self.compound {
            self.compound_extra(grid, at);
            return;
        }
        let w4 = 16.min(at.bw4).min(grid.cols.saturating_sub(at.col));
        let h4 = 16.min(at.bh4).min(grid.rows.saturating_sub(at.row));
        let num = w4.min(h4);
        for pass in 0..2 {
            let mut idx = 0usize;
            while idx < num && self.found < 2 {
                let (row, col) = if pass == 0 {
                    (at.row as i32 - 1, at.col as i32 + idx as i32)
                } else {
                    (at.row as i32 + idx as i32, at.col as i32 - 1)
                };
                if !grid.inside(row, col) {
                    break;
                }
                self.add_extra(grid, row, col);
                let size = grid.size(row, col);
                idx += if pass == 0 {
                    NUM_4X4_WIDE[size]
                } else {
                    NUM_4X4_HIGH[size]
                };
            }
        }
        for slot in self.found..2 {
            self.mv[slot] = self.global;
        }
    }

    fn add_extra(&mut self, grid: &MotionGrid<'_>, row: i32, col: i32) {
        let refs = [grid.ref_at(row, col), grid.ref1_at(row, col)];
        for (list, cand_ref) in refs.into_iter().enumerate() {
            if cand_ref <= 0 || self.found >= STACK {
                continue;
            }
            let mut mv = if list == 0 {
                grid.mv(row, col)
            } else {
                grid.mv1(row, col)
            };
            if self.bias_differs(cand_ref, self.reference) {
                mv = negate_mv(mv);
            }
            if self.mv.iter().take(self.found).any(|have| *have == mv) {
                continue;
            }
            self.mv[self.found] = mv;
            self.weight[self.found] = 2;
            self.found += 1;
        }
    }

    fn sort(&mut self, start: usize, mut end: usize) {
        while end > start {
            let mut new_end = start;
            for idx in (start + 1)..end {
                if self.weight[idx - 1] < self.weight[idx] {
                    self.weight.swap(idx - 1, idx);
                    self.mv.swap(idx - 1, idx);
                    self.mv1.swap(idx - 1, idx);
                    new_end = idx;
                }
            }
            end = new_end;
        }
    }

    fn clamp(&mut self, grid: &MotionGrid<'_>, at: &BlockPlace) {
        let bh = block_height(at.bsize) as i32;
        let bw = block_width(at.bsize) as i32;
        for mv in self.mv.iter_mut().take(self.found) {
            mv.row = clamp_edge(mv.row, at.row, at.bh4, grid.rows, MV_BORDER + bh * 8);
            mv.col = clamp_edge(mv.col, at.col, at.bw4, grid.cols, MV_BORDER + bw * 8);
        }
        for mv in self.mv1.iter_mut().take(self.found) {
            mv.row = clamp_edge(mv.row, at.row, at.bh4, grid.rows, MV_BORDER + bh * 8);
            mv.col = clamp_edge(mv.col, at.col, at.bw4, grid.cols, MV_BORDER + bw * 8);
        }
    }

    /// Temporal scan. Samples inside the block, then three positions that stay
    /// inside the same 64×64 superblock. A missing field entry is `-32768`.
    fn temporal(&mut self, grid: &MotionGrid<'_>, at: &BlockPlace, field: &MotionField) {
        let step_w = if at.bw4 >= 16 { 4 } else { 2 };
        let step_h = if at.bh4 >= 16 { 4 } else { 2 };
        let rows = i32::try_from(at.bh4.min(16)).unwrap_or(0);
        let cols = i32::try_from(at.bw4.min(16)).unwrap_or(0);
        let mut delta_row = 0;
        while delta_row < rows {
            let mut delta_col = 0;
            while delta_col < cols {
                self.add_temporal(grid, at, field, delta_row, delta_col);
                delta_col += step_w;
            }
            delta_row += step_h;
        }
        if !(2..16).contains(&at.bh4) || !(2..16).contains(&at.bw4) {
            return;
        }
        let bh4 = i32::try_from(at.bh4).unwrap_or(0);
        let bw4 = i32::try_from(at.bw4).unwrap_or(0);
        for (row_off, col_off) in [(bh4, -2), (bh4, bw4), (bh4 - 2, bw4)] {
            if superblock_contains(at.row, at.col, row_off, col_off) {
                self.add_temporal(grid, at, field, row_off, col_off);
            }
        }
    }

    fn add_temporal(
        &mut self,
        grid: &MotionGrid<'_>,
        at: &BlockPlace,
        field: &MotionField,
        delta_row: i32,
        delta_col: i32,
    ) {
        let mv_row = (at.row as i32).wrapping_add(delta_row) | 1;
        let mv_col = (at.col as i32).wrapping_add(delta_col) | 1;
        if !grid.inside(mv_row, mv_col) {
            return;
        }
        let y8 = (mv_row as usize) >> 1;
        let x8 = (mv_col as usize) >> 1;
        if delta_row == 0 && delta_col == 0 {
            self.zero_ctx = 1;
        }
        if self.compound {
            self.add_temporal_pair(field, y8, x8, delta_row, delta_col);
        } else {
            self.add_temporal_single(field, y8, x8, delta_row, delta_col);
        }
    }

    fn add_temporal_single(
        &mut self,
        field: &MotionField,
        y8: usize,
        x8: usize,
        delta_row: i32,
        delta_col: i32,
    ) {
        let cand = field.get(self.reference, y8, x8);
        if cand.row == INVALID_MV {
            return;
        }
        let cand = lower(cand, self.hp, self.integer);
        if delta_row == 0 && delta_col == 0 {
            self.zero_ctx = usize::from(vector_is_far(cand, self.global));
        }
        if let Some(idx) = (0..self.found).find(|&idx| self.mv[idx] == cand) {
            self.weight[idx] += 2;
        } else if self.found < STACK {
            self.mv[self.found] = cand;
            self.weight[self.found] = 2;
            self.found += 1;
        }
    }

    fn add_temporal_pair(
        &mut self,
        field: &MotionField,
        y8: usize,
        x8: usize,
        delta_row: i32,
        delta_col: i32,
    ) {
        let cand0 = field.get(self.reference, y8, x8);
        if cand0.row == INVALID_MV {
            return;
        }
        let cand1 = field.get(self.reference1, y8, x8);
        if cand1.row == INVALID_MV {
            return;
        }
        let cand0 = lower(cand0, self.hp, self.integer);
        let cand1 = lower(cand1, self.hp, self.integer);
        if delta_row == 0 && delta_col == 0 {
            let far = vector_is_far(cand0, self.global) || vector_is_far(cand1, self.global1);
            self.zero_ctx = usize::from(far);
        }
        if let Some(idx) =
            (0..self.found).find(|&idx| self.mv[idx] == cand0 && self.mv1[idx] == cand1)
        {
            self.weight[idx] += 2;
        } else if self.found < STACK {
            self.mv[self.found] = cand0;
            self.mv1[self.found] = cand1;
            self.weight[self.found] = 2;
            self.found += 1;
        }
    }
}

impl MotionGrid<'_> {
    fn inside(&self, row: i32, col: i32) -> bool {
        row >= self.tile_row as i32
            && col >= self.tile_col as i32
            && (row as usize) < self.tile_row_end
            && (col as usize) < self.tile_col_end
            && (row as usize) < self.rows
            && (col as usize) < self.cols
    }

    fn index(&self, row: i32, col: i32) -> usize {
        (row as usize) * self.cols + col as usize
    }

    fn inter_at(&self, row: i32, col: i32) -> u8 {
        self.inter[self.index(row, col)]
    }

    fn ref_at(&self, row: i32, col: i32) -> i8 {
        self.refs[self.index(row, col)]
    }

    fn ref1_at(&self, row: i32, col: i32) -> i8 {
        self.refs1[self.index(row, col)]
    }

    fn mv(&self, row: i32, col: i32) -> Motion {
        self.mvs[self.index(row, col)]
    }

    fn mv1(&self, row: i32, col: i32) -> Motion {
        self.mvs1[self.index(row, col)]
    }

    fn mode(&self, row: i32, col: i32) -> u8 {
        self.modes[self.index(row, col)]
    }

    fn size(&self, row: i32, col: i32) -> usize {
        usize::from(self.sizes[self.index(row, col)])
    }
}

fn clamp_edge(mvec: i32, mi: usize, blocks: usize, limit: usize, border: i32) -> i32 {
    let to_start = -((mi as i32) * 32);
    let to_end = (limit as i32 - blocks as i32 - mi as i32) * 32;
    clip3(to_start - border, to_end + border, mvec)
}

fn clip3(min: i32, max: i32, value: i32) -> i32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

fn superblock_contains(mi_row: usize, mi_col: usize, delta_row: i32, delta_col: i32) -> bool {
    let row = (mi_row as i32 & 15).wrapping_add(delta_row);
    let col = (mi_col as i32 & 15).wrapping_add(delta_col);
    (0..16).contains(&row) && (0..16).contains(&col)
}

fn vector_is_far(cand: Motion, global: Motion) -> bool {
    cand.row.abs_diff(global.row) >= 16 || cand.col.abs_diff(global.col) >= 16
}

const INVALID_MV: i32 = -32_768;
const MAX_FRAME_DISTANCE: i32 = 31;
const MFMV_STACK_SIZE: i32 = 3;
const REFMVS_LIMIT: u32 = (1 << 12) - 1;
const MV_CLAMP: i32 = (1 << 14) - 1;
const DIV_MULT: [i32; 32] = [
    0, 16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638, 1489, 1365, 1260, 1170, 1092,
    1024, 963, 910, 862, 819, 780, 744, 712, 682, 655, 630, 606, 585, 564, 546, 528,
];

/// Motion vectors projected onto the current frame, one 8×8 grid per reference.
pub struct MotionField {
    cols8: usize,
    rows8: usize,
    mvs: Vec<Motion>,
}

impl MotionField {
    pub fn empty() -> Self {
        Self {
            cols8: 0,
            rows8: 0,
            mvs: Vec::new(),
        }
    }

    fn filled(mi_cols: usize, mi_rows: usize) -> Self {
        let cols8 = mi_cols >> 1;
        let rows8 = mi_rows >> 1;
        let count = cols8.saturating_mul(rows8).saturating_mul(7);
        Self {
            cols8,
            rows8,
            mvs: vec![
                Motion {
                    row: INVALID_MV,
                    col: INVALID_MV,
                };
                count
            ],
        }
    }

    fn get(&self, reference: i8, y8: usize, x8: usize) -> Motion {
        let invalid = Motion {
            row: INVALID_MV,
            col: INVALID_MV,
        };
        let Some(index) = self.index(reference, y8, x8) else {
            return invalid;
        };
        self.mvs.get(index).copied().unwrap_or(invalid)
    }

    fn set(&mut self, reference: i8, y8: usize, x8: usize, mv: Motion) {
        let Some(index) = self.index(reference, y8, x8) else {
            return;
        };
        if let Some(slot) = self.mvs.get_mut(index) {
            *slot = mv;
        }
    }

    fn index(&self, reference: i8, y8: usize, x8: usize) -> Option<usize> {
        if !(LAST_FRAME..=ALTREF_FRAME).contains(&reference)
            || self.cols8 == 0
            || y8 >= self.rows8
            || x8 >= self.cols8
        {
            return None;
        }
        let plane = usize::from(u8::try_from(reference - LAST_FRAME).unwrap_or(0));
        Some(
            plane
                .saturating_mul(self.rows8)
                .saturating_add(y8)
                .saturating_mul(self.cols8)
                .saturating_add(x8),
        )
    }
}

/// One reference slot's saved motion field. Only the odd 4×4 of each 8×8 is read.
pub struct SavedMotion {
    pub frame_type: u8,
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub saved_hints: [u32; 8],
    pub refs: Vec<i8>,
    pub mvs: Vec<Motion>,
}

impl SavedMotion {
    fn unused() -> Self {
        Self {
            frame_type: crate::header::KEY_FRAME,
            mi_cols: 0,
            mi_rows: 0,
            saved_hints: [0; 8],
            refs: Vec::new(),
            mvs: Vec::new(),
        }
    }
}

/// The motion field a reference stored, borrowed for projection.
pub struct SavedView<'a> {
    pub frame_type: u8,
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub saved_hints: [u32; 8],
    pub refs: &'a [i8],
    pub mvs: &'a [Motion],
}

/// Inputs to motion-field estimation. `slot_hints` are the current frame's eight slots.
pub struct FieldSource<'a> {
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub order_hint: u32,
    pub hint_bits: u8,
    pub ref_slots: [u8; 7],
    pub slot_hints: &'a [u32; 8],
    pub saved: &'a [Option<SavedView<'a>>; 8],
}

pub fn saved_views<'a, S: Pix>(refs: &[Option<RefView<'a, S>>]) -> [Option<SavedView<'a>>; 8] {
    let mut saved = std::array::from_fn(|_| None);
    for (slot, view) in refs.iter().enumerate().take(8) {
        if let Some(view) = view {
            saved[slot] = Some(SavedView {
                frame_type: view.frame_type,
                mi_cols: view.mi_cols,
                mi_rows: view.mi_rows,
                saved_hints: view.saved_hints,
                refs: view.mf_refs,
                mvs: view.mf_mvs,
            });
        }
    }
    saved
}

/// Project earlier motion onto this frame. Section 7.9. An invalid entry stays `-32768`.
pub fn project_motion_field(source: &FieldSource<'_>) -> MotionField {
    let mut field = MotionField::filled(source.mi_cols, source.mi_rows);
    let last_slot = usize::from(source.ref_slots[0]);
    let last_alt = source
        .saved
        .get(last_slot)
        .and_then(|slot| slot.as_ref())
        .map(|saved| saved.saved_hints[7])
        .unwrap_or(0);
    if last_alt != order_hint_of(source, GOLDEN_FRAME) {
        let _ = project_reference(&mut field, source, LAST_FRAME, -1);
    }
    let mut stamp = MFMV_STACK_SIZE - 2;
    if forward_of(source, BWDREF_FRAME) && project_reference(&mut field, source, BWDREF_FRAME, 1) {
        stamp -= 1;
    }
    if forward_of(source, ALTREF2_FRAME) && project_reference(&mut field, source, ALTREF2_FRAME, 1)
    {
        stamp -= 1;
    }
    if stamp >= 0
        && forward_of(source, ALTREF_FRAME)
        && project_reference(&mut field, source, ALTREF_FRAME, 1)
    {
        stamp -= 1;
    }
    if stamp >= 0 {
        let _ = project_reference(&mut field, source, LAST2_FRAME, -1);
    }
    field
}

fn forward_of(source: &FieldSource<'_>, reference: i8) -> bool {
    crate::header::relative_dist(
        order_hint_of(source, reference),
        source.order_hint,
        source.hint_bits,
    ) > 0
}

fn order_hint_of(source: &FieldSource<'_>, reference: i8) -> u32 {
    let index = usize::try_from(reference - LAST_FRAME).unwrap_or(0);
    let slot = usize::from(*source.ref_slots.get(index).unwrap_or(&0));
    source.slot_hints.get(slot).copied().unwrap_or(0)
}

/// One source reference. Returns whether that reference was a same-size inter frame.
fn project_reference(
    field: &mut MotionField,
    source: &FieldSource<'_>,
    src: i8,
    dst_sign: i32,
) -> bool {
    let slot = usize::from(
        *source
            .ref_slots
            .get(usize::try_from(src - LAST_FRAME).unwrap_or(0))
            .unwrap_or(&0),
    );
    let Some(saved) = source.saved.get(slot).and_then(|item| item.as_ref()) else {
        return false;
    };
    if saved.mi_rows != source.mi_rows
        || saved.mi_cols != source.mi_cols
        || saved.frame_type == crate::header::KEY_FRAME
        || saved.frame_type == crate::header::INTRA_ONLY
    {
        return false;
    }
    let cells = source.mi_rows.saturating_mul(source.mi_cols);
    if saved.refs.len() < cells || saved.mvs.len() < cells {
        return false;
    }
    let src_hint = source.slot_hints.get(slot).copied().unwrap_or(0);
    let width8 = source.mi_cols >> 1;
    let height8 = source.mi_rows >> 1;
    for y8 in 0..height8 {
        for x8 in 0..width8 {
            let row = 2 * y8 + 1;
            let col = 2 * x8 + 1;
            let cell = row.saturating_mul(saved.mi_cols).saturating_add(col);
            let Some(&src_ref) = saved.refs.get(cell) else {
                continue;
            };
            if src_ref <= INTRA_FRAME {
                continue;
            }
            let ref_to_cur =
                crate::header::relative_dist(src_hint, source.order_hint, source.hint_bits);
            let past = saved
                .saved_hints
                .get(usize::try_from(src_ref).unwrap_or(0))
                .copied()
                .unwrap_or(0);
            let ref_offset = crate::header::relative_dist(src_hint, past, source.hint_bits);
            if ref_to_cur.unsigned_abs() > u32::try_from(MAX_FRAME_DISTANCE).unwrap_or(31)
                || ref_offset.unsigned_abs() > u32::try_from(MAX_FRAME_DISTANCE).unwrap_or(31)
                || ref_offset <= 0
            {
                continue;
            }
            let Some(&mv) = saved.mvs.get(cell) else {
                continue;
            };
            let placed = project_mv(mv, ref_to_cur.wrapping_mul(dst_sign), ref_offset);
            let (pos_y, pos_x, valid) = block_position(x8, y8, dst_sign, placed, height8, width8);
            if !valid {
                continue;
            }
            for dst in LAST_FRAME..=ALTREF_FRAME {
                let ref_to_dst = crate::header::relative_dist(
                    source.order_hint,
                    order_hint_of(source, dst),
                    source.hint_bits,
                );
                field.set(dst, pos_y, pos_x, project_mv(mv, ref_to_dst, ref_offset));
            }
        }
    }
    true
}

fn project_mv(mv: Motion, numerator: i32, denominator: i32) -> Motion {
    let clipped_den = if denominator < 0 {
        0
    } else {
        denominator.min(MAX_FRAME_DISTANCE)
    };
    let clipped_num = numerator.clamp(-MAX_FRAME_DISTANCE, MAX_FRAME_DISTANCE);
    let factor = DIV_MULT[usize::try_from(clipped_den).unwrap_or(0)];
    Motion {
        row: scale_mv_component(mv.row, clipped_num, factor),
        col: scale_mv_component(mv.col, clipped_num, factor),
    }
}

fn scale_mv_component(value: i32, numerator: i32, factor: i32) -> i32 {
    let product = i64::from(value)
        .wrapping_mul(i64::from(numerator))
        .wrapping_mul(i64::from(factor));
    round2_signed_i64(product, 14).clamp(-MV_CLAMP, MV_CLAMP)
}

fn block_position(
    x8: usize,
    y8: usize,
    dst_sign: i32,
    mv: Motion,
    max_y: usize,
    max_x: usize,
) -> (usize, usize, bool) {
    let (y, y_ok) = project_coord(i32::try_from(y8).unwrap_or(0), mv.row, dst_sign, max_y, 0);
    let (x, x_ok) = project_coord(i32::try_from(x8).unwrap_or(0), mv.col, dst_sign, max_x, 8);
    (
        usize::try_from(y).unwrap_or(0),
        usize::try_from(x).unwrap_or(0),
        y_ok && x_ok,
    )
}

fn project_coord(v8: i32, delta: i32, dst_sign: i32, max8: usize, max_off: i32) -> (i32, bool) {
    let base8 = v8 >> 3 << 3;
    let offset8 = if delta >= 0 {
        delta >> 6
    } else {
        -(delta.wrapping_neg() >> 6)
    };
    let next = v8.wrapping_add(dst_sign.wrapping_mul(offset8));
    let max8 = i32::try_from(max8).unwrap_or(i32::MAX);
    let ok = next >= 0 && next < max8 && next >= base8 - max_off && next < base8 + 8 + max_off;
    (next, ok)
}

/// Inputs to the motion-field storage process. Backward vectors within the limit are kept.
pub struct SaveQuery<'a> {
    pub frame_type: u8,
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub order_hint: u32,
    pub hint_bits: u8,
    pub ref_slots: [u8; 7],
    pub slot_hints: &'a [u32; 8],
    pub block_ref: &'a [i8],
    pub block_ref1: &'a [i8],
    pub mvs: &'a [Motion],
    pub mvs1: &'a [Motion],
}

pub fn save_motion_field(query: &SaveQuery<'_>) -> SavedMotion {
    if query.frame_type == crate::header::KEY_FRAME || query.frame_type == crate::header::INTRA_ONLY
    {
        return SavedMotion {
            frame_type: query.frame_type,
            mi_cols: query.mi_cols,
            mi_rows: query.mi_rows,
            saved_hints: [0; 8],
            refs: Vec::new(),
            mvs: Vec::new(),
        };
    }
    let cells = query.mi_rows.saturating_mul(query.mi_cols);
    let mut refs = vec![NONE_FRAME; cells];
    let mut mvs = vec![Motion::default(); cells];
    let mut saved_hints = [0u32; 8];
    for (index, slot) in query.ref_slots.iter().enumerate() {
        let slot = usize::from(*slot);
        saved_hints[index + 1] = query.slot_hints.get(slot).copied().unwrap_or(0);
    }
    for cell in 0..cells {
        if let Some((reference, mv)) = kept_mv(
            query,
            query.block_ref.get(cell).copied().unwrap_or(INTRA_FRAME),
            query.mvs.get(cell).copied().unwrap_or_default(),
        ) {
            refs[cell] = reference;
            mvs[cell] = mv;
        }
        if let Some((reference, mv)) = kept_mv(
            query,
            query.block_ref1.get(cell).copied().unwrap_or(NONE_FRAME),
            query.mvs1.get(cell).copied().unwrap_or_default(),
        ) {
            refs[cell] = reference;
            mvs[cell] = mv;
        }
    }
    SavedMotion {
        frame_type: query.frame_type,
        mi_cols: query.mi_cols,
        mi_rows: query.mi_rows,
        saved_hints,
        refs,
        mvs,
    }
}

fn kept_mv(query: &SaveQuery<'_>, reference: i8, mv: Motion) -> Option<(i8, Motion)> {
    if reference <= INTRA_FRAME {
        return None;
    }
    let index = usize::try_from(reference - LAST_FRAME).unwrap_or(0);
    let slot = usize::from(*query.ref_slots.get(index).unwrap_or(&0));
    let hint = query.slot_hints.get(slot).copied().unwrap_or(0);
    if crate::header::relative_dist(hint, query.order_hint, query.hint_bits) >= 0 {
        return None;
    }
    if mv.row.unsigned_abs() > REFMVS_LIMIT || mv.col.unsigned_abs() > REFMVS_LIMIT {
        return None;
    }
    Some((reference, mv))
}

/// An empty motion field. A sequence that leaves reference motion off stores this.
pub fn motion_unused() -> SavedMotion {
    SavedMotion::unused()
}

/// Project one block's global motion vector. Section 7.10.2.1.
pub fn project_global(
    kind: u8,
    params: [i32; 6],
    row: usize,
    col: usize,
    bsize: usize,
    hp: bool,
    integer: bool,
) -> Motion {
    let mv = if kind == GM_IDENTITY {
        Motion::default()
    } else if kind == GM_TRANSLATION {
        Motion {
            row: params[0] >> (WARPED_PREC_BITS - 3),
            col: params[1] >> (WARPED_PREC_BITS - 3),
        }
    } else {
        let bw = block_width(bsize) as i64;
        let bh = block_height(bsize) as i64;
        let x = col as i64 * 4 + bw / 2 - 1;
        let y = row as i64 * 4 + bh / 2 - 1;
        let one = 1i64 << WARPED_PREC_BITS;
        let xc = (i64::from(params[2]) - one) * x + i64::from(params[3]) * y + i64::from(params[0]);
        let yc = i64::from(params[4]) * x + (i64::from(params[5]) - one) * y + i64::from(params[1]);
        if hp {
            Motion {
                row: round2_signed_i64(yc, WARPED_PREC_BITS - 3),
                col: round2_signed_i64(xc, WARPED_PREC_BITS - 3),
            }
        } else {
            Motion {
                row: round2_signed_i64(yc, WARPED_PREC_BITS - 2) * 2,
                col: round2_signed_i64(xc, WARPED_PREC_BITS - 2) * 2,
            }
        }
    };
    lower(mv, hp, integer)
}

const WARPED_PREC_BITS: u32 = 16;

fn round2_signed_i64(value: i64, bits: u32) -> i32 {
    let bias = 1i64 << (bits - 1);
    let rounded = if value >= 0 {
        (value + bias) >> bits
    } else {
        -((value.saturating_neg() + bias) >> bits)
    };
    i32::try_from(rounded).unwrap_or(if rounded > 0 { i32::MAX } else { i32::MIN })
}

fn lower(mv: Motion, hp: bool, integer: bool) -> Motion {
    if hp {
        return mv;
    }
    Motion {
        row: lower_comp(mv.row, integer),
        col: lower_comp(mv.col, integer),
    }
}

fn lower_comp(value: i32, integer: bool) -> i32 {
    if integer {
        let rounded = (value.unsigned_abs() + 3) >> 3;
        let magnitude = i32::try_from(rounded << 3).unwrap_or(0);
        if value > 0 { magnitude } else { -magnitude }
    } else if value & 1 != 0 {
        if value > 0 { value - 1 } else { value + 1 }
    } else {
        value
    }
}

pub struct InterSample<'a, S: Pix = u8> {
    pub dst: &'a mut [S],
    pub stride: usize,
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    pub sub_x: bool,
    pub sub_y: bool,
    pub mv: Motion,
    pub filter: [u8; 2],
    pub reference: &'a [S],
    pub ref_stride: usize,
    pub plane_w: usize,
    pub plane_h: usize,
    pub frame_w: usize,
    pub frame_h: usize,
    pub ref_w: usize,
    pub ref_h: usize,
    /// Where the predicted samples are stored. The motion vector still uses `x` and `y`.
    pub write_x: usize,
    pub write_y: usize,
    pub bit_depth: u8,
}

/// Shear of a rotation-zoom or affine model. Section 7.11.3.6.
#[must_use]
pub fn global_shear_valid(params: [i32; 6]) -> bool {
    shear(params).is_some()
}

/// Least-squares affine model for one block. Section 7.11.3.8.
///
/// `samples` are source and destination points in 1/8-pel luma units:
/// `[sy, sx, dy, dx]`. `mid_x` and `mid_y` are the block center in luma samples.
/// `None` means the fit is singular.
#[must_use]
pub fn local_warp_params(
    samples: &[[i32; 4]],
    mv: Motion,
    mid_x: i32,
    mid_y: i32,
) -> Option<[i32; 6]> {
    let mut a00 = 0i64;
    let mut a01 = 0i64;
    let mut a11 = 0i64;
    let mut bx0 = 0i64;
    let mut bx1 = 0i64;
    let mut by0 = 0i64;
    let mut by1 = 0i64;
    let suy = i64::from(mid_y) * 8;
    let sux = i64::from(mid_x) * 8;
    let duy = suy + i64::from(mv.row);
    let dux = sux + i64::from(mv.col);
    for cand in samples {
        let sy = i64::from(cand[0]) - suy;
        let sx = i64::from(cand[1]) - sux;
        let dy = i64::from(cand[2]) - duy;
        let dx = i64::from(cand[3]) - dux;
        if (sx - dx).abs() < 256 && (sy - dy).abs() < 256 {
            a00 += ls_product(sx, sx) + 8;
            a01 += ls_product(sx, sy) + 4;
            a11 += ls_product(sy, sy) + 8;
            bx0 += ls_product(sx, dx) + 8;
            bx1 += ls_product(sy, dx) + 4;
            by0 += ls_product(sx, dy) + 4;
            by1 += ls_product(sy, dy) + 8;
        }
    }
    let det = i128::from(a00) * i128::from(a11) - i128::from(a01) * i128::from(a01);
    let det = i64::try_from(det).ok()?;
    let (shift, mut factor) = resolve_divisor(det)?;
    let mut shift_i = i32::try_from(shift).ok()? - 16;
    if shift_i < 0 {
        let bits = u32::try_from(-shift_i).ok()?;
        if bits >= 63 {
            return None;
        }
        factor = factor.wrapping_shl(bits);
        shift_i = 0;
    }
    let shift_u = u32::try_from(shift_i).ok()?;
    let p2 = warp_diag(
        i128::from(a11) * i128::from(bx0) - i128::from(a01) * i128::from(bx1),
        factor,
        shift_u,
    )?;
    let p3 = warp_nondiag(
        i128::from(a00) * i128::from(bx1) - i128::from(a01) * i128::from(bx0),
        factor,
        shift_u,
    )?;
    let p4 = warp_nondiag(
        i128::from(a11) * i128::from(by0) - i128::from(a01) * i128::from(by1),
        factor,
        shift_u,
    )?;
    let p5 = warp_diag(
        i128::from(a00) * i128::from(by1) - i128::from(a01) * i128::from(by0),
        factor,
        shift_u,
    )?;
    let one = 1i64 << 16;
    let vx = i64::from(mv.col) * (1 << 13)
        - (i64::from(mid_x) * (i64::from(p2) - one) + i64::from(mid_y) * i64::from(p3));
    let vy = i64::from(mv.row) * (1 << 13)
        - (i64::from(mid_x) * i64::from(p4) + i64::from(mid_y) * (i64::from(p5) - one));
    Some([
        i32::try_from(vx.clamp(-8_388_608, 8_388_607)).unwrap_or(0),
        i32::try_from(vy.clamp(-8_388_608, 8_388_607)).unwrap_or(0),
        p2,
        p3,
        p4,
        p5,
    ])
}

fn ls_product(a: i64, b: i64) -> i64 {
    ((a * b) >> 2) + a + b
}

fn warp_diag(value: i128, factor: i64, shift: u32) -> Option<i32> {
    let rounded = round2_signed_i128(value * i128::from(factor), shift);
    i32::try_from(rounded.clamp(57_345, 73_727)).ok()
}

fn warp_nondiag(value: i128, factor: i64, shift: u32) -> Option<i32> {
    let rounded = round2_signed_i128(value * i128::from(factor), shift);
    i32::try_from(rounded.clamp(-8_191, 8_191)).ok()
}

fn shear(params: [i32; 6]) -> Option<(i32, i32, i32, i32)> {
    let prec = 1i64 << WARPED_PREC_BITS;
    let alpha0 = (i64::from(params[2]) - prec).clamp(-32768, 32767);
    let beta0 = i64::from(params[3]).clamp(-32768, 32767);
    let (div_shift, div_factor) = resolve_divisor(i64::from(params[2]))?;
    let v = i128::from(params[4]) << WARPED_PREC_BITS;
    let gamma0 = round2_signed_i128(v * i128::from(div_factor), div_shift).clamp(-32768, 32767);
    let w = i128::from(params[3]) * i128::from(params[4]);
    let delta0 =
        (i64::from(params[5]) - round2_signed_i128(w * i128::from(div_factor), div_shift) - prec)
            .clamp(-32768, 32767);
    let alpha = reduce_warp(alpha0)?;
    let beta = reduce_warp(beta0)?;
    let gamma = reduce_warp(gamma0)?;
    let delta = reduce_warp(delta0)?;
    let valid = 4 * alpha.abs() + 7 * beta.abs() < (1 << WARPED_PREC_BITS)
        && 4 * gamma.abs() + 4 * delta.abs() < (1 << WARPED_PREC_BITS);
    valid.then_some((alpha, beta, gamma, delta))
}

fn reduce_warp(value: i64) -> Option<i32> {
    let reduced =
        round2_signed_i128(i128::from(value), WARP_PARAM_REDUCE_BITS) << WARP_PARAM_REDUCE_BITS;
    i32::try_from(reduced).ok()
}

fn resolve_divisor(value: i64) -> Option<(u32, i64)> {
    if value == 0 {
        return None;
    }
    let magnitude = value.unsigned_abs();
    let n = 63 - magnitude.leading_zeros();
    let e = magnitude - (1u64 << n);
    let f = if n > DIV_LUT_BITS {
        let shift = n - DIV_LUT_BITS;
        (e + (1u64 << (shift - 1))) >> shift
    } else {
        e << (DIV_LUT_BITS - n)
    };
    let factor = i64::from(*DIV_LUT.get(usize::try_from(f).ok()?)?);
    let shift = n + DIV_LUT_PREC_BITS;
    if shift >= 63 {
        return None;
    }
    Some((shift, if value < 0 { -factor } else { factor }))
}

fn round2_signed_i128(value: i128, bits: u32) -> i64 {
    if bits == 0 {
        return i64::try_from(value).unwrap_or(if value > 0 { i64::MAX } else { i64::MIN });
    }
    let bias = 1i128 << (bits - 1);
    let rounded = if value >= 0 {
        (value + bias) >> bits
    } else {
        -((value.saturating_neg() + bias) >> bits)
    };
    i64::try_from(rounded).unwrap_or(if rounded > 0 { i64::MAX } else { i64::MIN })
}

/// Warp one plane from a global rotation-zoom or affine model. Section 7.11.3.5.
pub fn predict_warp<S: Pix>(
    sample: &mut InterSample<'_, S>,
    params: [i32; 6],
) -> Result<(), CodecError> {
    let x_scale = scale(sample.ref_w, sample.frame_w);
    let y_scale = scale(sample.ref_h, sample.frame_h);
    if x_scale != NO_SCALE || y_scale != NO_SCALE {
        return Err(CodecError::Unsupported {
            what: "scaled reference",
        });
    }
    let Some((alpha, beta, gamma, delta)) = shear(params) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "global warp",
        });
    };
    let sub_x = i32::from(sample.sub_x);
    let sub_y = i32::from(sample.sub_y);
    let last_x = plane_last(sample.ref_w, sample.sub_x, sample.plane_w);
    let last_y = plane_last(sample.ref_h, sample.sub_y, sample.plane_h);
    let mut i8 = 0i32;
    while i8 * 8 < sample.h as i32 {
        let mut j8 = 0i32;
        while j8 * 8 < sample.w as i32 {
            warp_section(
                sample,
                params,
                (alpha, beta, gamma, delta),
                (sub_x, sub_y),
                (last_x, last_y),
                (i8, j8),
            )?;
            j8 += 1;
        }
        i8 += 1;
    }
    Ok(())
}

fn warp_section<S: Pix>(
    sample: &mut InterSample<'_, S>,
    params: [i32; 6],
    shear: (i32, i32, i32, i32),
    sub: (i32, i32),
    last: (i32, i32),
    section: (i32, i32),
) -> Result<(), CodecError> {
    let (alpha, beta, gamma, delta) = shear;
    let (sub_x, sub_y) = sub;
    let (last_x, last_y) = last;
    let (i8, j8) = section;
    let src_x = (sample.x as i32 + j8 * 8 + 4) << sub_x;
    let src_y = (sample.y as i32 + i8 * 8 + 4) << sub_y;
    let dst_x = i64::from(params[2]) * i64::from(src_x)
        + i64::from(params[3]) * i64::from(src_y)
        + i64::from(params[0]);
    let dst_y = i64::from(params[4]) * i64::from(src_x)
        + i64::from(params[5]) * i64::from(src_y)
        + i64::from(params[1]);
    let x4 = dst_x >> sub_x;
    let y4 = dst_y >> sub_y;
    let ix4 = (x4 >> WARPED_PREC_BITS) as i32;
    let sx4 = (x4 & ((1i64 << WARPED_PREC_BITS) - 1)) as i32;
    let iy4 = (y4 >> WARPED_PREC_BITS) as i32;
    let sy4 = (y4 & ((1i64 << WARPED_PREC_BITS) - 1)) as i32;
    let (round0, round1) = single_rounds(sample.bit_depth);
    let mut intermediate = [[0i32; 8]; 15];
    for i1 in -7..8 {
        for i2 in -4..4 {
            let sx = sx4 + alpha * i2 + beta * i1;
            let offs = warp_offset(sx)?;
            let mut sum = 0i32;
            let taps = warp_taps(offs)?;
            for (i3, coeff) in taps.iter().enumerate() {
                let sx = ix4 + i2 - 3 + i3 as i32;
                sum += i32::from(*coeff) * ref_sample(sample, last_x, last_y, sx, iy4 + i1);
            }
            intermediate[(i1 + 7) as usize][(i2 + 4) as usize] = round2(sum, round0);
        }
    }
    let row_limit = 4.min(sample.h as i32 - i8 * 8 - 4);
    let col_limit = 4.min(sample.w as i32 - j8 * 8 - 4);
    for i1 in -4..row_limit {
        for i2 in -4..col_limit {
            let sy = sy4 + gamma * i2 + delta * i1;
            let offs = warp_offset(sy)?;
            let mut sum = 0i32;
            let taps = warp_taps(offs)?;
            for (i3, coeff) in taps.iter().enumerate() {
                let row = (i1 + i3 as i32 + 4) as usize;
                let col = (i2 + 4) as usize;
                sum += i32::from(*coeff) * intermediate[row][col];
            }
            let pred = S::from_i32(clip_sample(round2(sum, round1), sample.bit_depth));
            let y = sample.write_y + (i8 * 8 + i1 + 4) as usize;
            let x = sample.write_x + (j8 * 8 + i2 + 4) as usize;
            if x < sample.stride {
                let at = y * sample.stride + x;
                if at < sample.dst.len() {
                    sample.dst[at] = pred;
                }
            }
        }
    }
    Ok(())
}

fn warp_offset(value: i32) -> Result<usize, CodecError> {
    let offs = round2(value, WARPEDDIFF_PREC_BITS) + WARPEDPIXEL_PREC_SHIFTS;
    usize::try_from(offs).map_err(|_| CodecError::Malformed {
        at_bit: 0,
        what: "global warp",
    })
}

fn warp_taps(offset: usize) -> Result<&'static [i8], CodecError> {
    let start = offset * 8;
    WARPED_FILTERS
        .get(start..start + 8)
        .ok_or(CodecError::Malformed {
            at_bit: 0,
            what: "global warp",
        })
}

const WARPEDDIFF_PREC_BITS: u32 = 10;
const WARPEDPIXEL_PREC_SHIFTS: i32 = 64;
const WARP_PARAM_REDUCE_BITS: u32 = 6;
const DIV_LUT_BITS: u32 = 8;
const DIV_LUT_PREC_BITS: u32 = 14;

pub fn predict<S: Pix>(sample: &mut InterSample<'_, S>) -> Result<(), CodecError> {
    let Some(pass) = prepare(sample)? else {
        return Ok(());
    };
    vertical_pixels(sample, &pass);
    Ok(())
}

/// One compound reference, still shifted by `InterPostRound` (4 for 8-bit).
pub fn predict_precise<S: Pix>(
    sample: &InterSample<'_, S>,
    out: &mut [i32],
) -> Result<(), CodecError> {
    let Some(pass) = prepare(sample)? else {
        return Ok(());
    };
    vertical_precise(sample, &pass, 7, out);
    Ok(())
}

struct VerticalPass {
    start_y: i64,
    step_y: i64,
    filter: u8,
    rows: Vec<i32>,
}

/// `COMPOUND_AVERAGE`: `Clip1(Round2(a + b, 1 + InterPostRound))`.
/// `InterPostRound` is 4 at 8-bit and 10-bit, and 2 at 12-bit.
pub fn average_compound(a: i32, b: i32, depth: u8) -> i32 {
    clip_sample(round2(a.saturating_add(b), compound_shift(1, depth)), depth)
}

/// `COMPOUND_DISTANCE`: `Clip1(Round2(fwd * a + bck * b, 4 + InterPostRound))`.
pub fn distance_compound(a: i32, b: i32, fwd: i32, bck: i32, depth: u8) -> i32 {
    let sum = a.saturating_mul(fwd).saturating_add(b.saturating_mul(bck));
    clip_sample(round2(sum, compound_shift(4, depth)), depth)
}

/// `COMPOUND_WEDGE` and `COMPOUND_DIFFWTD`: `Clip1(Round2(m * a + (64 - m) * b, 6 + InterPostRound))`.
pub fn mask_compound(a: i32, b: i32, weight: i32, depth: u8) -> i32 {
    let shift = compound_shift(6, depth);
    let bias = 1i64 << (shift - 1);
    let sum = i64::from(weight) * i64::from(a) + i64::from(64 - weight) * i64::from(b);
    let mixed = i32::try_from((sum + bias) >> shift).unwrap_or(i32::MAX);
    clip_sample(mixed, depth)
}

/// `Wedge_Bits`. Sizes with a zero have no wedge index; the mask is difference-weighted.
pub fn wedge_bits(bsize: usize) -> u8 {
    const BITS: [u8; 22] = [
        0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0,
    ];
    BITS.get(bsize).copied().unwrap_or(0)
}

/// Luma wedge mask for one block. `sign` selects the complement of the master ramp.
pub fn wedge_mask(w: usize, h: usize, index: usize, sign: u8, out: &mut [u8]) {
    let n = w.saturating_mul(h);
    if w == 0 || h == 0 || out.len() < n {
        return;
    }
    let shape = if h > w {
        0
    } else if h < w {
        1
    } else {
        2
    };
    let code = WEDGE_CODEBOOK[shape][index.min(15)];
    let xoff = 32 - ((i32::from(code[1]) * i32::try_from(w).unwrap_or(0)) >> 3);
    let yoff = 32 - ((i32::from(code[2]) * i32::try_from(h).unwrap_or(0)) >> 3);
    let mut sum = 0i32;
    for col in 0..w {
        sum += i32::from(master(
            code[0],
            yoff,
            xoff + i32::try_from(col).unwrap_or(0),
        ));
    }
    for row in 1..h {
        sum += i32::from(master(
            code[0],
            yoff + i32::try_from(row).unwrap_or(0),
            xoff,
        ));
    }
    let denom = i32::try_from(w + h - 1).unwrap_or(1).max(1);
    let flip = (sum + denom / 2) / denom < 32;
    let keep = (sign != 0) == flip;
    for row in 0..h {
        for col in 0..w {
            let sample = master(
                code[0],
                yoff + i32::try_from(row).unwrap_or(0),
                xoff + i32::try_from(col).unwrap_or(0),
            );
            out[row * w + col] = if keep { sample } else { 64 - sample };
        }
    }
}

/// Difference-weight mask from the two luma predictors, still at `InterPostRound`.
pub fn difference_mask(
    a: &[i32],
    b: &[i32],
    w: usize,
    h: usize,
    mask_type: u8,
    depth: u8,
    out: &mut [u8],
) {
    let n = w.saturating_mul(h);
    let shift = depth_shift(depth) + 4;
    for index in 0..n.min(out.len()).min(a.len()).min(b.len()) {
        let diff = (i64::from(a[index]) - i64::from(b[index])).unsigned_abs();
        let rounded = i32::try_from(diff).unwrap_or(i32::MAX);
        let scaled = round2(rounded, shift);
        let weight = (38 + scaled / 16).clamp(0, 64) as u8;
        out[index] = if mask_type == 0 { weight } else { 64 - weight };
    }
}

/// Chroma takes the rounded average of the luma mask samples that cover it.
pub fn compound_mask_weight(
    mask: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    sub_x: bool,
    sub_y: bool,
) -> i32 {
    if stride == 0 {
        return 0;
    }
    if !sub_x && !sub_y {
        return mask_at(mask, stride, y, x);
    }
    if sub_x && !sub_y {
        return (mask_at(mask, stride, y, 2 * x) + mask_at(mask, stride, y, 2 * x + 1) + 1) >> 1;
    }
    let sum = mask_at(mask, stride, 2 * y, 2 * x)
        + mask_at(mask, stride, 2 * y, 2 * x + 1)
        + mask_at(mask, stride, 2 * y + 1, 2 * x)
        + mask_at(mask, stride, 2 * y + 1, 2 * x + 1);
    (sum + 2) >> 2
}

fn mask_at(mask: &[u8], stride: usize, y: usize, x: usize) -> i32 {
    i32::from(
        mask.get(y.saturating_mul(stride).saturating_add(x))
            .copied()
            .unwrap_or(0),
    )
}

fn master(dir: u8, y: i32, x: i32) -> u8 {
    match dir {
        0 => ramp(y),
        1 => ramp(x),
        2 => oblique63(x, y),
        3 => oblique63(y, x),
        4 => 64 - oblique63(y, 63 - x),
        _ => 64 - oblique63(x, 63 - y),
    }
}

fn ramp(index: i32) -> u8 {
    WEDGE_VERTICAL[index.clamp(0, 63) as usize]
}

fn oblique63(y: i32, x: i32) -> u8 {
    let row = y.clamp(0, 63) as usize;
    let shift = 16 - ((row + (row & 1)) / 2) as i32;
    let index = (x - shift).clamp(0, 63) as usize;
    if row.is_multiple_of(2) {
        WEDGE_EVEN[index]
    } else {
        WEDGE_ODD[index]
    }
}

/// `Wedge_Codebook`. Direction, horizontal offset, vertical offset, for tall, wide, and square blocks.
const WEDGE_CODEBOOK: [[[u8; 3]; 16]; 3] = [
    [
        [2, 4, 4],
        [3, 4, 4],
        [4, 4, 4],
        [5, 4, 4],
        [0, 4, 2],
        [0, 4, 4],
        [0, 4, 6],
        [1, 4, 4],
        [2, 4, 2],
        [2, 4, 6],
        [5, 4, 2],
        [5, 4, 6],
        [3, 2, 4],
        [3, 6, 4],
        [4, 2, 4],
        [4, 6, 4],
    ],
    [
        [2, 4, 4],
        [3, 4, 4],
        [4, 4, 4],
        [5, 4, 4],
        [1, 2, 4],
        [1, 4, 4],
        [1, 6, 4],
        [0, 4, 4],
        [2, 4, 2],
        [2, 4, 6],
        [5, 4, 2],
        [5, 4, 6],
        [3, 2, 4],
        [3, 6, 4],
        [4, 2, 4],
        [4, 6, 4],
    ],
    [
        [2, 4, 4],
        [3, 4, 4],
        [4, 4, 4],
        [5, 4, 4],
        [0, 4, 2],
        [0, 4, 6],
        [1, 2, 4],
        [1, 6, 4],
        [2, 4, 2],
        [2, 4, 6],
        [5, 4, 2],
        [5, 4, 6],
        [3, 2, 4],
        [3, 6, 4],
        [4, 2, 4],
        [4, 6, 4],
    ],
];

const WEDGE_ODD: [u8; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 6,
    18, 37, 53, 60, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64,
];

const WEDGE_EVEN: [u8; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 11,
    27, 46, 58, 62, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64,
];

const WEDGE_VERTICAL: [u8; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 7,
    21, 43, 57, 62, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64,
];

fn negate_mv(mv: Motion) -> Motion {
    Motion {
        row: mv.row.wrapping_neg(),
        col: mv.col.wrapping_neg(),
    }
}

fn scale(reference: usize, frame: usize) -> i64 {
    (((reference as i64) << 14) + (frame as i64 / 2)) / frame as i64
}

/// `2 * frame >= reference` and `frame <= 16 * reference`.
fn scale_ratio_ok(reference: usize, frame: usize) -> bool {
    reference > 0
        && frame > 0
        && reference <= frame.saturating_mul(2)
        && frame <= reference.saturating_mul(16)
}

fn start_and_step(origin: usize, mv: i32, sub: bool, scale_value: i64) -> (i64, i64) {
    let half = 8i64;
    let shifted = (i64::from(mv) * 2) >> usize::from(sub);
    let orig = (origin as i64) * 16 + shifted + half;
    let base = orig * scale_value - (half << 14);
    let start = round2_signed(base, 8) + 32;
    let step = round2_signed(scale_value, 4);
    (start, step)
}

fn intermediate_rows(height: usize, step: i64) -> usize {
    let span = (height as i64 - 1) * step + ((1 << 10) - 1);
    (span >> 10) as usize + 8
}

fn small_filter(filter: u8, edge: usize) -> u8 {
    if edge > 4 {
        return filter;
    }
    match filter {
        0 | 2 => 4,
        1 => 5,
        other => other,
    }
}

fn horizontal<S: Pix>(
    sample: &InterSample<'_, S>,
    intermediate: &mut [i32],
    rows: usize,
    start_x: i64,
    start_y: i64,
    step_x: i64,
    filter: u8,
) {
    let last_x = plane_last(sample.ref_w, sample.sub_x, sample.plane_w);
    let last_y = plane_last(sample.ref_h, sample.sub_y, sample.plane_h);
    let (round0, _) = single_rounds(sample.bit_depth);
    let base_row = (start_y >> 10) as i32;
    for row in 0..rows {
        for col in 0..sample.w {
            let p = start_x + step_x * col as i64;
            let mut sum = 0i32;
            let taps = filter_taps(filter, phase(p));
            for (tap, coeff) in taps.iter().enumerate() {
                let sy = base_row + row as i32 - 3;
                let sx = (p >> 10) as i32 + tap as i32 - 3;
                sum += i32::from(*coeff) * ref_sample(sample, last_x, last_y, sx, sy);
            }
            intermediate[row * sample.w + col] = round2(sum, round0);
        }
    }
}

fn prepare<S: Pix>(sample: &InterSample<'_, S>) -> Result<Option<VerticalPass>, CodecError> {
    if sample.w == 0 || sample.h == 0 || sample.frame_w == 0 || sample.frame_h == 0 {
        return Ok(None);
    }
    if !scale_ratio_ok(sample.ref_w, sample.frame_w)
        || !scale_ratio_ok(sample.ref_h, sample.frame_h)
    {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "scaled reference",
        });
    }
    let x_scale = scale(sample.ref_w, sample.frame_w);
    let y_scale = scale(sample.ref_h, sample.frame_h);
    let (start_x, step_x) = start_and_step(sample.x, sample.mv.col, sample.sub_x, x_scale);
    let (start_y, step_y) = start_and_step(sample.y, sample.mv.row, sample.sub_y, y_scale);
    let rows = intermediate_rows(sample.h, step_y);
    // A legal reference is at most twice as tall, which adds about two rows per output row.
    let row_limit = sample.h.saturating_mul(2).saturating_add(16);
    if rows > row_limit || sample.w > 128 {
        return Err(CodecError::Limit {
            what: "inter block",
        });
    }
    let mut intermediate = vec![0i32; rows * sample.w];
    let h_filter = small_filter(sample.filter[1], sample.w);
    horizontal(
        sample,
        &mut intermediate,
        rows,
        start_x,
        start_y,
        step_x,
        h_filter,
    );
    let filter = small_filter(sample.filter[0], sample.h);
    Ok(Some(VerticalPass {
        start_y,
        step_y,
        filter,
        rows: intermediate,
    }))
}

fn vertical_pixels<S: Pix>(sample: &mut InterSample<'_, S>, pass: &VerticalPass) {
    let frac = pass.start_y & 1023;
    let (_, round1) = single_rounds(sample.bit_depth);
    for row in 0..sample.h {
        for col in 0..sample.w {
            let sum = vertical_sum(sample, pass, frac, row, col);
            let pred = S::from_i32(clip_sample(round2(sum, round1), sample.bit_depth));
            let y = sample.write_y + row;
            let x = sample.write_x + col;
            if x < sample.stride {
                let at = y * sample.stride + x;
                if at < sample.dst.len() {
                    sample.dst[at] = pred;
                }
            }
        }
    }
}

fn vertical_precise<S: Pix>(
    sample: &InterSample<'_, S>,
    pass: &VerticalPass,
    round1: u32,
    out: &mut [i32],
) {
    let frac = pass.start_y & 1023;
    for row in 0..sample.h {
        for col in 0..sample.w {
            let sum = vertical_sum(sample, pass, frac, row, col);
            let at = row * sample.w + col;
            if at < out.len() {
                out[at] = round2(sum, round1);
            }
        }
    }
}

fn vertical_sum<S: Pix>(
    sample: &InterSample<'_, S>,
    pass: &VerticalPass,
    frac: i64,
    row: usize,
    col: usize,
) -> i32 {
    let p = frac + pass.step_y * row as i64;
    let mut sum = 0i32;
    let taps = filter_taps(pass.filter, phase(p));
    for (tap, coeff) in taps.iter().enumerate() {
        let sy = (p >> 10) as usize + tap;
        let at = sy * sample.w + col;
        sum += i32::from(*coeff) * pass.rows.get(at).copied().unwrap_or(0);
    }
    sum
}

fn plane_last(luma: usize, sub: bool, plane: usize) -> i32 {
    let from_luma = (luma + usize::from(sub)) >> usize::from(sub);
    let last = from_luma.saturating_sub(1).min(plane.saturating_sub(1));
    i32::try_from(last).unwrap_or(0)
}

fn ref_sample<S: Pix>(
    sample: &InterSample<'_, S>,
    last_x: i32,
    last_y: i32,
    x: i32,
    y: i32,
) -> i32 {
    if sample.ref_stride == 0 || sample.reference.is_empty() {
        return 0;
    }
    let yy = clip3(0, last_y, y) as usize;
    let xx = clip3(0, last_x, x) as usize;
    let at = yy * sample.ref_stride + xx;
    sample
        .reference
        .get(at)
        .copied()
        .map(Pix::to_i32)
        .unwrap_or(0)
}

fn filter_taps(filter: u8, phase: usize) -> &'static [i16] {
    let which = usize::from(filter).min(5);
    let phase = phase.min(15);
    let start = (which * 16 + phase) * 8;
    &SUBPEL_FILTERS[start..start + 8]
}

fn phase(position: i64) -> usize {
    ((position >> 6) & 15) as usize
}

/// Section 7.11.3.2. `InterRound0` then `InterRound1` for one predictor.
/// Eight-bit and ten-bit stay `(3, 11)`. Twelve-bit is `(5, 9)`.
fn single_rounds(bit_depth: u8) -> (u32, u32) {
    if bit_depth == 12 { (5, 9) } else { (3, 11) }
}

/// `extra + InterPostRound`. The post round is 4 at 8-bit and 10-bit, and 2 at 12-bit.
fn compound_shift(extra: u32, bit_depth: u8) -> u32 {
    extra + if bit_depth == 12 { 2 } else { 4 }
}

fn round2(value: i32, n: u32) -> i32 {
    if n == 0 {
        value
    } else {
        (value + (1 << (n - 1))) >> n
    }
}

fn round2_signed(value: i64, n: u32) -> i64 {
    if n == 0 {
        return value;
    }
    let bias = 1i64 << (n - 1);
    if value >= 0 {
        (value + bias) >> n
    } else {
        -(((-value) + bias) >> n)
    }
}

/// `Ii_Weights_1d`. The intra predictor is weighted by this value, the inter by its complement.
pub const II_WEIGHTS: [u8; 128] = [
    60, 58, 56, 54, 52, 50, 48, 47, 45, 44, 42, 41, 39, 38, 37, 35, 34, 33, 32, 31, 30, 29, 28, 27,
    26, 25, 24, 23, 22, 22, 21, 20, 19, 19, 18, 18, 17, 16, 16, 15, 15, 14, 14, 13, 13, 12, 12, 12,
    11, 11, 10, 10, 10, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4,
    4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

/// Interintra mask sample. Mode 0 is DC, 1 vertical, 2 horizontal, 3 smooth.
pub fn interintra_weight(mode: u8, y: usize, x: usize, scale: usize) -> i32 {
    let index = match mode {
        1 => y * scale,
        2 => x * scale,
        3 => y.min(x) * scale,
        _ => return 32,
    };
    i32::from(II_WEIGHTS[index.min(II_WEIGHTS.len() - 1)])
}

#[cfg(test)]
mod tests {
    use super::{difference_mask, mask_compound, wedge_mask};

    #[test]
    fn wedge_signs_are_complements() {
        let mut mask = [0u8; 64];
        let mut flipped = [0u8; 64];
        wedge_mask(8, 8, 0, 0, &mut mask);
        wedge_mask(8, 8, 0, 1, &mut flipped);
        assert!(mask.iter().all(|sample| *sample <= 64));
        assert!(mask.iter().any(|sample| *sample < 32));
        assert!(mask.iter().any(|sample| *sample > 32));
        for (sample, other) in mask.iter().zip(flipped) {
            assert_eq!(u16::from(*sample) + u16::from(other), 64);
        }
    }

    #[test]
    fn difference_mask_uses_the_predictor_gap() {
        let mut out = [0u8; 4];
        difference_mask(&[0; 4], &[0; 4], 2, 2, 0, 8, &mut out);
        assert_eq!(out, [38, 38, 38, 38]);
        difference_mask(&[0; 4], &[0; 4], 2, 2, 1, 8, &mut out);
        assert_eq!(out, [26, 26, 26, 26]);
        assert_eq!(mask_compound(16, 0, 64, 8), 1);
        assert_eq!(mask_compound(16, 0, 0, 8), 0);
    }
}

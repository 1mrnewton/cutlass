//! Single-reference inter prediction.
//!
//! Motion-vector candidates follow the find-MV-stack process. The sample
//! predictor is the two-tap separable subpel filter from the decoding process.
//! Compound, warped, and overlapped motion stay outside this path.

use cutlass_decode::CodecError;

use crate::dims::{
    ADST_ADST, ADST_DCT, ADST_FLIPADST, DCT_ADST, DCT_DCT, DCT_FLIPADST, FLIPADST_ADST,
    FLIPADST_DCT, FLIPADST_FLIPADST, H_ADST, H_DCT, H_FLIPADST, IDTX, NUM_4X4_HIGH, NUM_4X4_WIDE,
    TX_16X16, TX_32X32, TX_SIZE_SQR, TX_SIZE_SQR_UP, V_ADST, V_DCT, V_FLIPADST, block_height,
    block_width,
};
use crate::tables::SUBPEL_FILTERS;

pub const NEARESTMV: u8 = 14;
pub const NEARMV: u8 = 15;
pub const GLOBALMV: u8 = 16;
pub const NEWMV: u8 = 17;
pub const LAST_FRAME: i8 = 1;
pub const NONE_FRAME: i8 = -1;
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

pub struct RefView<'a> {
    pub y: &'a [u8],
    pub u: &'a [u8],
    pub v: &'a [u8],
    pub width: usize,
    pub height: usize,
}

pub struct MotionGrid<'a> {
    pub cols: usize,
    pub rows: usize,
    pub inter: &'a [u8],
    pub refs: &'a [i8],
    pub refs1: &'a [i8],
    pub mvs: &'a [Motion],
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
}

pub struct Candidates {
    pub mv: [Motion; STACK],
    pub found: usize,
    pub new_ctx: usize,
    pub ref_ctx: usize,
    pub drl: [usize; STACK],
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
    weight: [i32; STACK],
    found: usize,
    new_count: usize,
    found_match: bool,
    global: Motion,
    reference: i8,
    hp: bool,
    integer: bool,
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
pub fn find_stack(grid: &MotionGrid<'_>, at: &BlockPlace, hp: bool, integer: bool) -> Candidates {
    let mut search = Search {
        mv: [Motion::default(); STACK],
        weight: [0; STACK],
        found: 0,
        new_count: 0,
        found_match: false,
        global: lower(Motion::default(), hp, integer),
        reference: at.reference,
        hp,
        integer,
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
    search.scan_far(grid, at, &mut above, &mut left);
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
        found: search.found,
        new_ctx,
        ref_ctx,
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

impl Search {
    fn scan_far(
        &mut self,
        grid: &MotionGrid<'_>,
        at: &BlockPlace,
        above: &mut bool,
        left: &mut bool,
    ) {
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
        let refs = [grid.ref_at(row, col), grid.ref1_at(row, col)];
        for (list, cand_ref) in refs.into_iter().enumerate() {
            if cand_ref == self.reference {
                self.search(grid, row, col, list, weight);
            }
        }
    }

    fn search(&mut self, grid: &MotionGrid<'_>, row: i32, col: i32, list: usize, weight: i32) {
        let mode = grid.mode(row, col);
        let mv = if list == 0 {
            grid.mv(row, col)
        } else {
            Motion::default()
        };
        let mv = lower(mv, self.hp, self.integer);
        if mode == NEWMV {
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

    fn extra(&mut self, grid: &MotionGrid<'_>, at: &BlockPlace) {
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
            let mv = if list == 0 {
                grid.mv(row, col)
            } else {
                Motion::default()
            };
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
    }
}

impl MotionGrid<'_> {
    fn inside(&self, row: i32, col: i32) -> bool {
        row >= 0 && col >= 0 && (row as usize) < self.rows && (col as usize) < self.cols
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

pub struct InterSample<'a> {
    pub dst: &'a mut [u8],
    pub stride: usize,
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    pub sub_x: bool,
    pub sub_y: bool,
    pub mv: Motion,
    pub filter: [u8; 2],
    pub reference: &'a [u8],
    pub ref_stride: usize,
    pub plane_w: usize,
    pub plane_h: usize,
    pub frame_w: usize,
    pub frame_h: usize,
    pub ref_w: usize,
    pub ref_h: usize,
}

pub fn predict(sample: &mut InterSample<'_>) -> Result<(), CodecError> {
    if sample.w == 0 || sample.h == 0 || sample.frame_w == 0 || sample.frame_h == 0 {
        return Ok(());
    }
    let x_scale = scale(sample.ref_w, sample.frame_w);
    let y_scale = scale(sample.ref_h, sample.frame_h);
    if x_scale != NO_SCALE || y_scale != NO_SCALE {
        return Err(CodecError::Unsupported {
            what: "scaled reference",
        });
    }
    let (start_x, step_x) = start_and_step(sample.x, sample.mv.col, sample.sub_x, x_scale);
    let (start_y, step_y) = start_and_step(sample.y, sample.mv.row, sample.sub_y, y_scale);
    let rows = intermediate_rows(sample.h, step_y);
    if rows > 128 || sample.w > 128 {
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
    let v_filter = small_filter(sample.filter[0], sample.h);
    vertical(sample, &intermediate, start_y, step_y, v_filter);
    Ok(())
}

fn scale(reference: usize, frame: usize) -> i64 {
    (((reference as i64) << 14) + (frame as i64 / 2)) / frame as i64
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

fn horizontal(
    sample: &InterSample<'_>,
    intermediate: &mut [i32],
    rows: usize,
    start_x: i64,
    start_y: i64,
    step_x: i64,
    filter: u8,
) {
    let last_x = plane_last(sample.ref_w, sample.sub_x, sample.plane_w);
    let last_y = plane_last(sample.ref_h, sample.sub_y, sample.plane_h);
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
            intermediate[row * sample.w + col] = round2(sum, 3);
        }
    }
}

fn vertical(
    sample: &mut InterSample<'_>,
    intermediate: &[i32],
    start_y: i64,
    step_y: i64,
    filter: u8,
) {
    let frac = start_y & 1023;
    for row in 0..sample.h {
        for col in 0..sample.w {
            let p = frac + step_y * row as i64;
            let mut sum = 0i32;
            let taps = filter_taps(filter, phase(p));
            for (tap, coeff) in taps.iter().enumerate() {
                let sy = (p >> 10) as usize + tap;
                let at = sy * sample.w + col;
                sum += i32::from(*coeff) * intermediate.get(at).copied().unwrap_or(0);
            }
            let pred = round2(sum, 11).clamp(0, 255) as u8;
            let y = sample.y + row;
            let x = sample.x + col;
            if x < sample.stride {
                let at = y * sample.stride + x;
                if at < sample.dst.len() {
                    sample.dst[at] = pred;
                }
            }
        }
    }
}

fn plane_last(luma: usize, sub: bool, plane: usize) -> i32 {
    let from_luma = (luma + usize::from(sub)) >> usize::from(sub);
    let last = from_luma.saturating_sub(1).min(plane.saturating_sub(1));
    i32::try_from(last).unwrap_or(0)
}

fn ref_sample(sample: &InterSample<'_>, last_x: i32, last_y: i32, x: i32, y: i32) -> i32 {
    if sample.ref_stride == 0 || sample.reference.is_empty() {
        return 0;
    }
    let yy = clip3(0, last_y, y) as usize;
    let xx = clip3(0, last_x, x) as usize;
    let at = yy * sample.ref_stride + xx;
    i32::from(sample.reference.get(at).copied().unwrap_or(0))
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

//! The loop filter over one superblock row, in libvpx's order: for each
//! superblock, every vertical edge of luma two 8-row strips at a time,
//! then every horizontal edge, then the same for each chroma plane.
//! Edge selection is `lpf::Mask` (libvpx `vp9_setup_mask`); 4:2:2 and
//! 4:4:0 chroma follow the specification's per-sample process.

use crate::dsp::lf::{Thresh, Wd, filter};
use crate::dsp::pixel::Pixel;
use crate::geom::{self, INTRA_FRAME, TX_4X4, TX_8X8, TX_16X16};
use crate::lpf::{Levels, Mask};

use super::frame::Plane;
use super::tile::LfMi;

/// The geometry of the frame the filter runs on.
pub struct Geo {
    pub mi_rows: usize,
    pub mi_cols: usize,
    pub sb_cols: usize,
    pub ss_x: usize,
    pub ss_y: usize,
    pub bd: u32,
}

/// Filters superblock row `sb_row` of all three planes.
pub fn sb_row<P: Pixel>(
    planes: &mut [Plane<P>; 3],
    masks: &[Mask],
    grid: &[LfMi],
    levels: &Levels,
    g: &Geo,
    sb_row: usize,
) {
    let [y, u, v] = planes;
    for sb_col in 0..g.sb_cols {
        let mut m = masks[sb_row * g.sb_cols + sb_col];
        let (mi_row, mi_col) = (sb_row * 8, sb_col * 8);
        m.adjust(mi_row, mi_col, g.mi_rows, g.mi_cols);
        luma(&mut y.data, y.stride, &m, levels, g, mi_row, mi_col);
        match (g.ss_x, g.ss_y) {
            (1, 1) => {
                chroma420(&mut u.data, u.stride, &m, levels, g, mi_row, mi_col);
                chroma420(&mut v.data, v.stride, &m, levels, g, mi_row, mi_col);
            }
            (0, 0) => {
                luma(&mut u.data, u.stride, &m, levels, g, mi_row, mi_col);
                luma(&mut v.data, v.stride, &m, levels, g, mi_row, mi_col);
            }
            _ => {
                for pl in [&mut *u, &mut *v] {
                    general(&mut pl.data, pl.stride, grid, levels, g, mi_row, mi_col);
                }
            }
        }
    }
}

#[inline(always)]
fn byte(mask: u64, row: usize) -> u8 {
    (mask >> (row * 8)) as u8
}

#[inline(always)]
fn nibble(mask: u16, row: usize) -> u8 {
    ((mask >> (row * 4)) & 15) as u8
}

/// The edges of one 8-sample strip: bit `c` for column `c`.
#[derive(Clone, Copy)]
struct Run {
    wide: u8,
    mid: u8,
    narrow: u8,
    inner: u8,
    lfl: [u8; 8],
}

/// One edge of `n` lines (8 or 16) with thresholds for each 8.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn edge<P: Pixel>(buf: &mut [P], stride: usize, x: usize, y: usize, vertical: bool, n: usize, wd: Wd, thr: [Thresh; 2], bd: u32) {
    let at = y * stride + x;
    if P::lf_fast(buf, at, stride, vertical, n, wd, thr) {
        return;
    }
    let (step, line) = if vertical { (1, stride) } else { (stride, 1) };
    filter(buf, at, step, line, n.min(8), wd, thr[0], bd);
    if n > 8 {
        filter(buf, at + 8 * line, step, line, n - 8, wd, thr[1], bd);
    }
}

#[inline(always)]
fn vert<P: Pixel>(buf: &mut [P], stride: usize, x: usize, y: usize, n: usize, wd: Wd, t: Thresh, bd: u32) {
    edge(buf, stride, x, y, true, n, wd, [t, t], bd);
}

#[inline(always)]
fn horiz<P: Pixel>(buf: &mut [P], stride: usize, x: usize, y: usize, n: usize, wd: Wd, t: Thresh, bd: u32) {
    edge(buf, stride, x, y, false, n, wd, [t, t], bd);
}

/// The filter of an edge from its strip's bits, if any.
#[inline(always)]
fn kind(r: &Run, bit: u8, skip_wide: bool) -> Option<Wd> {
    if !skip_wide && r.wide & bit != 0 {
        Some(Wd::W16)
    } else if r.mid & bit != 0 {
        Some(Wd::W8)
    } else if r.narrow & bit != 0 {
        Some(Wd::W4)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn column<P: Pixel>(buf: &mut [P], stride: usize, x: usize, y: usize, col: usize, r: &Run, levels: &Levels, bd: u32, skip_wide: bool) {
    let bit = 1u8 << col;
    let t = levels.dsp(r.lfl[col]);
    if let Some(k) = kind(r, bit, skip_wide) {
        vert(buf, stride, x, y, 8, k, t, bd);
    }
    if r.inner & bit != 0 {
        vert(buf, stride, x + 4, y, 8, Wd::W4, t, bd);
    }
}

/// Two strips, 8 rows apart, of vertical edges, each pair of edges of
/// the same kind filtered at once.
#[allow(clippy::too_many_arguments)]
fn vertical_pair<P: Pixel>(buf: &mut [P], stride: usize, x0: usize, y: usize, cols: usize, a: &Run, b: Option<&Run>, levels: &Levels, bd: u32) {
    for col in 0..cols {
        let x = x0 + col * 8;
        let bit = 1u8 << col;
        let ta = levels.dsp(a.lfl[col]);
        let Some(b) = b else {
            column(buf, stride, x, y, col, a, levels, bd, false);
            continue;
        };
        let tb = levels.dsp(b.lfl[col]);
        let both_wide = a.wide & bit != 0 && b.wide & bit != 0;
        if both_wide {
            // libvpx filters both strips with the first strip's level.
            vert(buf, stride, x, y, 16, Wd::W16, ta, bd);
        }
        let ka = kind(a, bit, both_wide);
        let kb = kind(b, bit, both_wide);
        match (ka, kb) {
            (Some(p), Some(q)) if p == q => edge(buf, stride, x, y, true, 16, p, [ta, tb], bd),
            _ => {
                if let Some(p) = ka {
                    vert(buf, stride, x, y, 8, p, ta, bd);
                }
                if let Some(q) = kb {
                    vert(buf, stride, x, y + 8, 8, q, tb, bd);
                }
            }
        }
        let (ia, ib) = (a.inner & bit != 0, b.inner & bit != 0);
        if ia && ib {
            edge(buf, stride, x + 4, y, true, 16, Wd::W4, [ta, tb], bd);
        } else if ia {
            vert(buf, stride, x + 4, y, 8, Wd::W4, ta, bd);
        } else if ib {
            vert(buf, stride, x + 4, y + 8, 8, Wd::W4, tb, bd);
        }
    }
}

/// Horizontal edges of one strip, each two neighbouring columns of the
/// same kind filtered at once.
#[allow(clippy::too_many_arguments)]
fn horizontal<P: Pixel>(buf: &mut [P], stride: usize, x0: usize, y: usize, cols: usize, r: &Run, levels: &Levels, bd: u32) {
    let mut col = 0;
    while col < cols {
        let bit = 1u8 << col;
        let x = x0 + col * 8;
        let t = levels.dsp(r.lfl[col]);
        let two = col + 1 < cols;
        if two && r.wide & bit != 0 && r.wide & (bit << 1) != 0 {
            horiz(buf, stride, x, y, 16, Wd::W16, t, bd);
            col += 2;
            continue;
        }
        let k = kind(r, bit, false);
        let next = if two { kind(r, bit << 1, false) } else { None };
        if two && k.is_some() && k == next && k != Some(Wd::W16) {
            let tn = levels.dsp(r.lfl[col + 1]);
            if let Some(k) = k {
                edge(buf, stride, x, y, false, 16, k, [t, tn], bd);
            }
            let (i0, i1) = (r.inner & bit != 0, r.inner & (bit << 1) != 0);
            if i0 && i1 {
                edge(buf, stride, x, y + 4, false, 16, Wd::W4, [t, tn], bd);
            } else if i0 {
                horiz(buf, stride, x, y + 4, 8, Wd::W4, t, bd);
            } else if i1 {
                horiz(buf, stride, x + 8, y + 4, 8, Wd::W4, tn, bd);
            }
            col += 2;
            continue;
        }
        if let Some(k) = k {
            horiz(buf, stride, x, y, 8, k, t, bd);
        }
        if r.inner & bit != 0 {
            horiz(buf, stride, x, y + 4, 8, Wd::W4, t, bd);
        }
        col += 1;
    }
}
fn luma<P: Pixel>(buf: &mut [P], stride: usize, m: &Mask, levels: &Levels, g: &Geo, mi_row: usize, mi_col: usize) {
    let cols = g.mi_cols.saturating_sub(mi_col).min(8);
    let lfl = |row: usize| -> [u8; 8] { core::array::from_fn(|c| m.lfl_y[row * 8 + c]) };
    let vrun = |row: usize| Run {
        wide: byte(m.left_y[2], row),
        mid: byte(m.left_y[1], row),
        narrow: byte(m.left_y[0], row),
        inner: byte(m.int_y, row),
        lfl: lfl(row),
    };
    let mut row = 0;
    while row < 8 && mi_row + row < g.mi_rows {
        let y = (mi_row + row) * 8;
        let a = vrun(row);
        let b = (row + 1 < 8 && mi_row + row + 1 < g.mi_rows).then(|| vrun(row + 1));
        vertical_pair(buf, stride, mi_col * 8, y, cols, &a, b.as_ref(), levels, g.bd);
        row += 2;
    }
    for row in 0..8 {
        if mi_row + row >= g.mi_rows {
            break;
        }
        let edge = mi_row + row != 0;
        let pick = |mask: u64| if edge { byte(mask, row) } else { 0 };
        let r = Run {
            wide: pick(m.above_y[2]),
            mid: pick(m.above_y[1]),
            narrow: pick(m.above_y[0]),
            inner: byte(m.int_y, row),
            lfl: lfl(row),
        };
        horizontal(buf, stride, mi_col * 8, (mi_row + row) * 8, cols, &r, levels, g.bd);
    }
}

fn chroma420<P: Pixel>(buf: &mut [P], stride: usize, m: &Mask, levels: &Levels, g: &Geo, mi_row: usize, mi_col: usize) {
    let cols = g.mi_cols.saturating_sub(mi_col).div_ceil(2).min(4);
    let lfl = |row: usize| -> [u8; 8] { core::array::from_fn(|c| if c < 4 { m.lfl_y[row * 16 + c * 2] } else { 0 }) };
    let vrun = |row: usize| Run {
        wide: nibble(m.left_uv[2], row),
        mid: nibble(m.left_uv[1], row),
        narrow: nibble(m.left_uv[0], row),
        inner: nibble(m.int_uv, row),
        lfl: lfl(row),
    };
    let mut row = 0;
    while row < 4 && mi_row + row * 2 < g.mi_rows {
        let y = mi_row * 4 + row * 8;
        let a = vrun(row);
        let b = (row + 1 < 4 && mi_row + (row + 1) * 2 < g.mi_rows).then(|| vrun(row + 1));
        vertical_pair(buf, stride, mi_col * 4, y, cols, &a, b.as_ref(), levels, g.bd);
        row += 2;
    }
    for row in 0..4 {
        let mi = mi_row + row * 2;
        if mi >= g.mi_rows {
            break;
        }
        let edge = mi != 0;
        let pick = |mask: u16| if edge { nibble(mask, row) } else { 0 };
        let r = Run {
            wide: pick(m.above_uv[2]),
            mid: pick(m.above_uv[1]),
            narrow: pick(m.above_uv[0]),
            inner: if mi + 1 == g.mi_rows { 0 } else { nibble(m.int_uv, row) },
            lfl: lfl(row),
        };
        horizontal(buf, stride, mi_col * 4, mi_row * 4 + row * 8, cols, &r, levels, g.bd);
    }
}

/// 4:2:2 and 4:4:0 chroma, one sample of an edge at a time (clause
/// 8.8.2's edge loop).
fn general<P: Pixel>(buf: &mut [P], stride: usize, grid: &[LfMi], levels: &Levels, g: &Geo, mi_row: usize, mi_col: usize) {
    let (ssx, ssy) = (g.ss_x, g.ss_y);
    for pass in 0..2 {
        let (sub, edge_len) = if pass == 0 { (ssx, 64 >> ssy) } else { (ssy, 64 >> ssx) };
        let edges = 16 >> sub;
        for edge in 0..edges {
            for along in 0..edge_len {
                let (x, y) = if pass == 0 {
                    (mi_col * 8 + edge * (4 << ssx), mi_row * 8 + (along << ssy))
                } else {
                    (mi_col * 8 + (along << ssx), mi_row * 8 + edge * (4 << ssy))
                };
                if x >= g.mi_cols * 8 || y >= g.mi_rows * 8 {
                    continue;
                }
                if (pass == 0 && x == 0) || (pass == 1 && y == 0) {
                    continue;
                }
                let loop_col = ((x >> 3) >> ssx) << ssx;
                let loop_row = ((y >> 3) >> ssy) << ssy;
                let mi = grid[loop_row * g.mi_cols + loop_col];
                let tx = geom::uv_tx(mi.bsize, mi.tx, ssx as u32, ssy as u32);
                let sb_size = if sub == 0 { mi.bsize } else { mi.bsize.max(geom::BLOCK_16X16) };
                let (wide, high) = geom::mi_wh(sb_size);
                let is_block = if pass == 0 {
                    wide > 0 && x % (8 * wide) == 0
                } else {
                    high > 0 && y % (8 * high) == 0
                };
                let step = 1usize << tx;
                let is_tx = if pass == 1 && ssx == 1 && g.mi_cols % 2 == 1 && edge % 2 == 1 && x + 8 >= g.mi_cols * 8 {
                    false
                } else {
                    edge % step == 0
                };
                let intra = mi.ref0 == INTRA_FRAME;
                if !is_block && !(is_tx && (intra || !mi.skip)) {
                    continue;
                }
                let Some(level) = levels.level(mi.seg, mi.ref0, mi.mode) else {
                    continue;
                };
                if level == 0 {
                    continue;
                }
                let is32 = edge % 8 == 0;
                let base = if tx == TX_4X4 && is32 { TX_8X8 } else { tx.min(TX_16X16) };
                let clipped = base == TX_16X16
                    && ((pass == 0 && ssx == 1 && (x >> 3) + 1 == g.mi_cols)
                        || (pass == 1 && ssy == 1 && (y >> 3) + 1 == g.mi_rows));
                let size = if clipped { TX_8X8 } else { base };
                let wd = match size {
                    TX_4X4 => Wd::W4,
                    TX_8X8 => Wd::W8,
                    _ => Wd::W16,
                };
                let t = levels.dsp(level);
                let (px, py) = (x >> ssx, y >> ssy);
                if pass == 0 {
                    vert(buf, stride, px, py, 1, wd, t, g.bd);
                } else {
                    horiz(buf, stride, px, py, 1, wd, t, g.bd);
                }
            }
        }
    }
}

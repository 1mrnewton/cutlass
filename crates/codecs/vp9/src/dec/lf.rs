//! The loop filter over one superblock row, in libvpx's order: for each
//! superblock, every vertical edge of luma two 8-row strips at a time,
//! then every horizontal edge, then the same for each chroma plane.
//! Edge selection is `lpf::Mask` (libvpx `vp9_setup_mask`); 4:2:2 and
//! 4:4:0 chroma follow the specification's per-sample process.

use crate::dsp::lf::{Thresh, Wd, filter};
use crate::dsp::pixel::Pixel;
use crate::geom::{self, INTRA_FRAME, TX_4X4, TX_8X8, TX_16X16};
use crate::lpf::{Levels, Mask};

use super::parse::LfMi;
use super::recon::View;

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
    planes: &mut [View<'_, P>; 3],
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
        luma(y.data, y.base, y.stride, &m, levels, g, mi_row, mi_col);
        match (g.ss_x, g.ss_y) {
            (1, 1) => {
                chroma420(u.data, u.base, u.stride, &m, levels, g, mi_row, mi_col);
                chroma420(v.data, v.base, v.stride, &m, levels, g, mi_row, mi_col);
            }
            (0, 0) => {
                luma(u.data, u.base, u.stride, &m, levels, g, mi_row, mi_col);
                luma(v.data, v.base, v.stride, &m, levels, g, mi_row, mi_col);
            }
            _ => {
                for pl in [&mut *u, &mut *v] {
                    general(pl.data, pl.base, pl.stride, grid, levels, g, mi_row, mi_col);
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

/// Two halves of 8 lines across one edge: a vertical edge at column `x`
/// over rows `y..y + 16`, or a horizontal one at row `y` over columns
/// `x..x + 16`. Each half has its filter (`None`: left alone) and level.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn pair<P: Pixel>(buf: &mut [P], base: usize, stride: usize, x: usize, y: usize, vertical: bool, wd: [Option<Wd>; 2], thr: [Thresh; 2], bd: u32) {
    if wd[0].is_none() && wd[1].is_none() {
        return;
    }
    let at = y * stride + x - base;
    if P::lf_fast(buf, at, stride, vertical, wd, thr) {
        return;
    }
    let (step, line) = if vertical { (1, stride) } else { (stride, 1) };
    for (h, w) in wd.iter().enumerate() {
        if let Some(w) = *w {
            filter(buf, at + h * 8 * line, step, line, 8, w, thr[h], bd);
        }
    }
}

/// One line across an edge, for the per-sample chroma path.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn one<P: Pixel>(buf: &mut [P], base: usize, stride: usize, x: usize, y: usize, vertical: bool, wd: Wd, t: Thresh, bd: u32) {
    let at = y * stride + x - base;
    let (step, line) = if vertical { (1, stride) } else { (stride, 1) };
    filter(buf, at, step, line, 1, wd, t, bd);
}

/// The filter of an edge from its strip's bits, if any.
#[inline(always)]
fn kind(r: &Run, bit: u8) -> Option<Wd> {
    if r.wide & bit != 0 {
        Some(Wd::W16)
    } else if r.mid & bit != 0 {
        Some(Wd::W8)
    } else if r.narrow & bit != 0 {
        Some(Wd::W4)
    } else {
        None
    }
}

#[inline(always)]
fn any(r: &Run) -> u8 {
    r.wide | r.mid | r.narrow | r.inner
}

/// Two strips, 8 rows apart, of vertical edges: each column's edges of
/// both strips at once, then its inner 4×4 edges.
#[allow(clippy::too_many_arguments)]
fn vertical_pair<P: Pixel>(buf: &mut [P], base: usize, stride: usize, x0: usize, y: usize, cols: usize, a: &Run, b: Option<&Run>, levels: &Levels, bd: u32) {
    let limit = if cols >= 8 { 0xff } else { (1u8 << cols) - 1 };
    let mut bits = (any(a) | b.map_or(0, any)) & limit;
    while bits != 0 {
        let col = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        let bit = 1u8 << col;
        let x = x0 + col * 8;
        let ta = levels.dsp(a.lfl[col]);
        let tb = b.map_or(ta, |b| levels.dsp(b.lfl[col]));
        let ka = kind(a, bit);
        let kb = b.and_then(|b| kind(b, bit));
        // libvpx filters two 16-wide strips with the first strip's level.
        let thr = if ka == Some(Wd::W16) && kb == Some(Wd::W16) { [ta, ta] } else { [ta, tb] };
        pair(buf, base, stride, x, y, true, [ka, kb], thr, bd);
        let ia = (a.inner & bit != 0).then_some(Wd::W4);
        let ib = b.and_then(|b| (b.inner & bit != 0).then_some(Wd::W4));
        pair(buf, base, stride, x + 4, y, true, [ia, ib], [ta, tb], bd);
    }
}

/// Horizontal edges of one strip, two neighbouring columns at once. Two
/// 16-wide columns pair as libvpx pairs them, scanning from the left, and
/// share the first's level.
#[allow(clippy::too_many_arguments)]
fn horizontal<P: Pixel>(buf: &mut [P], base: usize, stride: usize, x0: usize, y: usize, cols: usize, r: &Run, levels: &Levels, bd: u32) {
    let limit = if cols >= 8 { 0xff } else { (1u8 << cols) - 1 };
    let bits = any(r) & limit;
    let wide = r.wide & limit;
    let mut col = 0;
    while col < cols {
        if bits >> col == 0 {
            break;
        }
        let bit = 1u8 << col;
        if bits & bit == 0 {
            col += 1;
            continue;
        }
        let x = x0 + col * 8;
        let t = levels.dsp(r.lfl[col]);
        if wide & bit != 0 && wide & (bit << 1) != 0 && col + 1 < cols {
            pair(buf, base, stride, x, y, false, [Some(Wd::W16); 2], [t, t], bd);
            col += 2;
            continue;
        }
        // The next column joins unless it starts a 16-wide pair.
        let next = bit << 1;
        let merge = col + 1 < cols && !(wide & next != 0 && wide & (next << 1) != 0 && col + 2 < cols);
        let tn = if merge { levels.dsp(r.lfl[col + 1]) } else { t };
        let kn = if merge { kind(r, next) } else { None };
        let i0 = (r.inner & bit != 0).then_some(Wd::W4);
        let i1 = (merge && r.inner & next != 0).then_some(Wd::W4);
        pair(buf, base, stride, x, y, false, [kind(r, bit), kn], [t, tn], bd);
        pair(buf, base, stride, x, y + 4, false, [i0, i1], [t, tn], bd);
        col += if merge { 2 } else { 1 };
    }
}

fn luma<P: Pixel>(buf: &mut [P], base: usize, stride: usize, m: &Mask, levels: &Levels, g: &Geo, mi_row: usize, mi_col: usize) {
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
        vertical_pair(buf, base, stride, mi_col * 8, y, cols, &a, b.as_ref(), levels, g.bd);
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
        horizontal(buf, base, stride, mi_col * 8, (mi_row + row) * 8, cols, &r, levels, g.bd);
    }
}

fn chroma420<P: Pixel>(buf: &mut [P], base: usize, stride: usize, m: &Mask, levels: &Levels, g: &Geo, mi_row: usize, mi_col: usize) {
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
        vertical_pair(buf, base, stride, mi_col * 4, y, cols, &a, b.as_ref(), levels, g.bd);
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
        horizontal(buf, base, stride, mi_col * 4, mi_row * 4 + row * 8, cols, &r, levels, g.bd);
    }
}

/// 4:2:2 and 4:4:0 chroma, one sample of an edge at a time (clause
/// 8.8.2's edge loop).
fn general<P: Pixel>(buf: &mut [P], base: usize, stride: usize, grid: &[LfMi], levels: &Levels, g: &Geo, mi_row: usize, mi_col: usize) {
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
                let tbase = if tx == TX_4X4 && is32 { TX_8X8 } else { tx.min(TX_16X16) };
                let clipped = tbase == TX_16X16
                    && ((pass == 0 && ssx == 1 && (x >> 3) + 1 == g.mi_cols)
                        || (pass == 1 && ssy == 1 && (y >> 3) + 1 == g.mi_rows));
                let size = if clipped { TX_8X8 } else { tbase };
                let wd = match size {
                    TX_4X4 => Wd::W4,
                    TX_8X8 => Wd::W8,
                    _ => Wd::W16,
                };
                let t = levels.dsp(level);
                let (px, py) = (x >> ssx, y >> ssy);
                if pass == 0 {
                    one(buf, base, stride, px, py, true, wd, t, g.bd);
                } else {
                    one(buf, base, stride, px, py, false, wd, t, g.bd);
                }
            }
        }
    }
}

#[cfg(test)]
mod order_tests {
    use super::{Run, Wd, horizontal, kind, pair};

    /// The driver against libvpx's `filter_selectively_horiz` order, one
    /// column at a time.
    #[test]
    fn horizontal_matches_one_column_at_a_time() {
        let mut h = crate::header::tests_header();
        h.filter_level = 30;
        let levels = crate::lpf::levels(&h);
        let stride = 96;
        let mut seed = 3u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            seed >> 8
        };
        for case in 0..5000 {
            let buf: Vec<u8> = (0..stride * 40).map(|i| (100 + (i % 7) as u32 + rnd() % 3) as u8).collect();
            let wide = (rnd() & 0xff) as u8;
            let mid = (rnd() & 0xff) as u8 & !wide;
            let narrow = (rnd() & 0xff) as u8 & !wide & !mid;
            let r = Run {
                wide,
                mid,
                narrow,
                inner: (rnd() & 0xff) as u8 & !wide,
                lfl: core::array::from_fn(|_| (rnd() % 63 + 1) as u8),
            };
            let cols = (rnd() % 8 + 1) as usize;
            let mut a = buf.clone();
            horizontal(&mut a, 0, stride, 8, 16, cols, &r, &levels, 8);
            let mut b = buf.clone();
            let mut col = 0;
            while col < cols {
                let bit = 1u8 << col;
                let x = 8 + 8 * col;
                let t = levels.dsp(r.lfl[col]);
                if col + 1 < cols && r.wide & bit != 0 && r.wide & (bit << 1) != 0 {
                    pair(&mut b, 0, stride, x, 16, false, [Some(Wd::W16); 2], [t, t], 8);
                    col += 2;
                    continue;
                }
                pair(&mut b, 0, stride, x, 16, false, [kind(&r, bit), None], [t, t], 8);
                let i = (r.inner & bit != 0).then_some(Wd::W4);
                pair(&mut b, 0, stride, x, 20, false, [i, None], [t, t], 8);
                col += 1;
            }
            assert_eq!(a, b, "case {case} cols {cols} {wide:08b} {mid:08b} {narrow:08b}");
        }
    }
}

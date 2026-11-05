//! Deblocking sample filter (spec 7.14.4 limits and 7.14.6 sample filtering).
//!
//! The caller discovers edges (7.14.2), picks the filter size (7.14.3) and the
//! level (7.14.5). These kernels apply the filter mask and the narrow or wide
//! filter to a run of `n` sample lines across one edge.
//!
//! `filter_size` is the spec's filterSize with the plane folded in:
//!
//! * `4`: any plane, filterSize 4.
//! * `6`: chroma with filterSize 8 (filterLen 6, 5-tap wide filter).
//! * `8`: luma with filterSize 8.
//! * `16`: luma with filterSize 16 (13-tap wide filter).
//!
//! Sample-access requirements: the filter reads (and may write) `filter_size / 2`
//! samples on either side of the edge for sizes 4, 6 and 8 (two, three and four
//! samples), and seven before / seven after the edge for size 16 (writing six).

use super::Pixel;

/// Filter limits derived from a level and sharpness (7.14.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LfLimits {
    pub limit: i32,
    pub blimit: i32,
    pub thresh: i32,
}

/// Derives `limit`, `blimit` and `thresh` from `lvl` (0..=63) and
/// `loop_filter_sharpness` (0..=7) per 7.14.4.
#[inline]
pub fn lf_limits(lvl: u8, sharpness: u8) -> LfLimits {
    let lvl = lvl as i32;
    let sharp = sharpness as i32;
    let shift = if sharp > 4 {
        2
    } else if sharp > 0 {
        1
    } else {
        0
    };
    let limit = if sharp > 0 {
        (lvl >> shift).clamp(1, 9 - sharp)
    } else {
        (lvl >> shift).max(1)
    };
    LfLimits {
        limit,
        blimit: 2 * (lvl + 2) + limit,
        thresh: lvl >> 4,
    }
}

/// Filters a vertical edge between columns `x - 1` and `x` for rows
/// `y..y + n`. `filter_size` is 4, 6, 8 or 16 (see the module docs).
/// `lvl == 0` filters nothing.
pub fn filter_vertical_edge<P: Pixel>(
    buf: &mut [P],
    stride: usize,
    x: usize,
    y: usize,
    n: usize,
    filter_size: usize,
    lvl: u8,
    sharpness: u8,
    bd: u32,
) {
    if lvl == 0 || n == 0 {
        return;
    }
    let lim = lf_limits(lvl, sharpness);
    match filter_size {
        4 => edge::<P, 4, false>(buf, stride, x, y, n, lim, bd),
        6 => edge::<P, 6, false>(buf, stride, x, y, n, lim, bd),
        8 => edge::<P, 8, false>(buf, stride, x, y, n, lim, bd),
        16 => edge::<P, 16, false>(buf, stride, x, y, n, lim, bd),
        _ => {}
    }
}

/// Filters a horizontal edge between rows `y - 1` and `y` for columns
/// `x..x + n`. `filter_size` is 4, 6, 8 or 16 (see the module docs).
/// `lvl == 0` filters nothing.
pub fn filter_horizontal_edge<P: Pixel>(
    buf: &mut [P],
    stride: usize,
    x: usize,
    y: usize,
    n: usize,
    filter_size: usize,
    lvl: u8,
    sharpness: u8,
    bd: u32,
) {
    if lvl == 0 || n == 0 {
        return;
    }
    let lim = lf_limits(lvl, sharpness);
    match filter_size {
        4 => edge::<P, 4, true>(buf, stride, x, y, n, lim, bd),
        6 => edge::<P, 6, true>(buf, stride, x, y, n, lim, bd),
        8 => edge::<P, 8, true>(buf, stride, x, y, n, lim, bd),
        16 => edge::<P, 16, true>(buf, stride, x, y, n, lim, bd),
        _ => {}
    }
}

/// First and one-past-last tap index (in the 14-sample line p6..q6, where
/// index 7 is q0) that a filter of size `FS` reads.
const fn tap_range(fs: usize) -> (usize, usize) {
    match fs {
        4 => (5, 9),
        6 => (4, 10),
        8 => (3, 11),
        _ => (0, 14),
    }
}

#[inline(always)]
fn edge<P: Pixel, const FS: usize, const HORZ: bool>(
    buf: &mut [P],
    stride: usize,
    x: usize,
    y: usize,
    n: usize,
    lim: LfLimits,
    bd: u32,
) {
    let mut i = 0;
    while i + 8 <= n {
        lanes::<P, FS, HORZ, 8>(buf, stride, x, y, i, lim, bd);
        i += 8;
    }
    while i + 4 <= n {
        lanes::<P, FS, HORZ, 4>(buf, stride, x, y, i, lim, bd);
        i += 4;
    }
    while i < n {
        lanes::<P, FS, HORZ, 1>(buf, stride, x, y, i, lim, bd);
        i += 1;
    }
}

/// Loads `L` lines starting at line `first` (lane `l` holds line
/// `first + l`; a vertical edge's lines are rows, transposed into lanes),
/// filters them, stores them back.
#[inline(always)]
fn lanes<P: Pixel, const FS: usize, const HORZ: bool, const L: usize>(
    buf: &mut [P],
    stride: usize,
    x: usize,
    y: usize,
    first: usize,
    lim: LfLimits,
    bd: u32,
) {
    let (lo, hi) = tap_range(FS);
    // Samples have at most 12 bits.
    let mut v = [[0i16; L]; 14];
    if HORZ {
        for k in lo..hi {
            let row = (y + k - 7) * stride + x + first;
            let src = &buf[row..row + L];
            for l in 0..L {
                v[k][l] = src[l].to_i32() as i16;
            }
        }
    } else {
        for l in 0..L {
            let base = (y + first + l) * stride + x - 7;
            let src = &buf[base + lo..base + hi];
            for k in lo..hi {
                v[k][l] = src[k - lo].to_i32() as i16;
            }
        }
    }
    let changed = if L > 1 {
        filter_vec::<FS, L>(&mut v, lim, bd)
    } else {
        filter_lanes::<FS, L>(&mut v, lim, bd)
    };
    if !changed {
        return;
    }
    // Only the modified range needs storing: one less than the read range on
    // each side, except for size 4, which reads and writes p1..q1.
    let (wlo, whi) = if FS == 4 { (lo, hi) } else { (lo + 1, hi - 1) };
    if HORZ {
        for k in wlo..whi {
            let row = (y + k - 7) * stride + x + first;
            let dst = &mut buf[row..row + L];
            for l in 0..L {
                dst[l] = P::from_i32(i32::from(v[k][l]));
            }
        }
    } else {
        for l in 0..L {
            let base = (y + first + l) * stride + x - 7;
            let dst = &mut buf[base + wlo..base + whi];
            for k in wlo..whi {
                dst[k - wlo] = P::from_i32(i32::from(v[k][l]));
            }
        }
    }
}

/// Wide filter (7.14.6.4) output for taps `-N..N` around the edge (index 7 is
/// q0), reading `px` only within `7 - (N + 1) ..= 7 + N`.
#[inline(always)]
fn wide<const N: i32, const N2: i32, const LOG2: u32>(px: &[i32; 14], out: &mut [i32; 14]) {
    let mut i = -N;
    while i < N {
        let mut t = 0;
        let mut j = -N;
        while j <= N {
            let p = (i + j).clamp(-(N + 1), N);
            let tap = if j.abs() <= N2 { 2 } else { 1 };
            t += px[(7 + p) as usize] * tap;
            j += 1;
        }
        out[(7 + i) as usize] = (t + (1 << (LOG2 - 1))) >> LOG2;
        i += 1;
    }
}

/// Applies the filter mask and filters to `L` lines, one at a time.
/// Returns whether any sample may have changed.
#[inline(always)]
fn filter_lanes<const FS: usize, const L: usize>(
    v: &mut [[i16; L]; 14],
    lim: LfLimits,
    bd: u32,
) -> bool {
    let sh = bd - 8;
    let limit_bd = lim.limit << sh;
    let blimit_bd = lim.blimit << sh;
    let thresh_bd = lim.thresh << sh;
    let one = 1i32 << sh;
    let half = 0x80i32 << sh;
    let lo4 = -(1i32 << (bd - 1));
    let hi4 = (1i32 << (bd - 1)) - 1;
    let (lo, hi) = tap_range(FS);
    let mut any = false;
    for l in 0..L {
        let mut px = [0i32; 14];
        for k in lo..hi {
            px[k] = i32::from(v[k][l]);
        }
        let (p1, p0, q0, q1) = (px[5], px[6], px[7], px[8]);
        let hev = (p1 - p0).abs() > thresh_bd || (q1 - q0).abs() > thresh_bd;
        let mut mask = (p1 - p0).abs() > limit_bd
            || (q1 - q0).abs() > limit_bd
            || (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 > blimit_bd;
        if FS >= 6 {
            mask |= (px[4] - p1).abs() > limit_bd || (px[9] - q1).abs() > limit_bd;
        }
        if FS >= 8 {
            mask |= (px[3] - px[4]).abs() > limit_bd || (px[10] - px[9]).abs() > limit_bd;
        }
        if mask {
            continue;
        }
        any = true;
        let mut flat = false;
        if FS >= 6 {
            flat = (p1 - p0).abs() <= one
                && (q1 - q0).abs() <= one
                && (px[4] - p0).abs() <= one
                && (px[9] - q0).abs() <= one;
            if FS >= 8 {
                flat &= (px[3] - p0).abs() <= one && (px[10] - q0).abs() <= one;
            }
        }
        let mut flat2 = false;
        if FS == 16 {
            flat2 = (px[0] - p0).abs() <= one
                && (px[13] - q0).abs() <= one
                && (px[1] - p0).abs() <= one
                && (px[12] - q0).abs() <= one
                && (px[2] - p0).abs() <= one
                && (px[11] - q0).abs() <= one;
        }
        let mut out = px;
        if FS == 4 || !flat {
            let c4 = |x: i32| x.clamp(lo4, hi4);
            let ps1 = p1 - half;
            let ps0 = p0 - half;
            let qs0 = q0 - half;
            let qs1 = q1 - half;
            let f = if hev { c4(ps1 - qs1) } else { 0 };
            let f = c4(f + 3 * (qs0 - ps0));
            let f1 = c4(f + 4) >> 3;
            let f2 = c4(f + 3) >> 3;
            out[7] = c4(qs0 - f1) + half;
            out[6] = c4(ps0 + f2) + half;
            if !hev {
                let f = (f1 + 1) >> 1;
                out[8] = c4(qs1 - f) + half;
                out[5] = c4(ps1 + f) + half;
            }
        } else if FS == 6 {
            wide::<2, 1, 3>(&px, &mut out);
        } else if FS == 8 || !flat2 {
            wide::<3, 0, 3>(&px, &mut out);
        } else {
            wide::<6, 1, 4>(&px, &mut out);
        }
        for k in lo..hi {
            v[k][l] = out[k] as i16;
        }
    }
    any
}

/// Wide filter on `L` lanes at once; see [`wide`]. The sums of at most 16
/// samples of 12 bits fit in `u16`.
#[inline(always)]
fn wide_vec<const N: i32, const N2: i32, const LOG2: u32, const L: usize>(
    v: &[[i16; L]; 14],
    out: &mut [[i16; L]; 14],
) {
    let mut i = -N;
    while i < N {
        let mut t = [1u16 << (LOG2 - 1); L];
        let mut j = -N;
        while j <= N {
            let p = (i + j).clamp(-(N + 1), N);
            let tap = if j.abs() <= N2 { 2 } else { 1 };
            let row = &v[(7 + p) as usize];
            for l in 0..L {
                t[l] = t[l].wrapping_add((row[l] as u16).wrapping_mul(tap));
            }
            j += 1;
        }
        let o = &mut out[(7 + i) as usize];
        for l in 0..L {
            o[l] = (t[l] >> LOG2) as i16;
        }
        i += 1;
    }
}

/// Branch-free version of [`filter_lanes`] for `L` lines: every per-sample
/// step is a straight loop over the lanes, on 16 bits (samples of 12 bits,
/// their differences and the narrow filter's terms all fit). Conditions
/// are lane masks (0 or -1).
#[inline(always)]
fn filter_vec<const FS: usize, const L: usize>(
    v: &mut [[i16; L]; 14],
    lim: LfLimits,
    bd: u32,
) -> bool {
    let sh = bd - 8;
    let limit_bd = (lim.limit << sh) as i16;
    let blimit_bd = (lim.blimit << sh) as i16;
    let thresh_bd = (lim.thresh << sh) as i16;
    let one = 1i16 << sh;
    let half = 0x80i16 << sh;
    let lo4 = -(1i16 << (bd - 1));
    let hi4 = ((1i32 << (bd - 1)) - 1) as i16;
    let ad = |a: i16, b: i16| (a - b).abs();
    let mask = |c: bool| i16::from(c).wrapping_neg();
    let sel = |m: i16, a: i16, b: i16| (a & m) | (b & !m);
    let mut fm = [0i16; L];
    let mut hev = [0i16; L];
    let mut flat = [0i16; L];
    let mut flat2 = [0i16; L];
    for l in 0..L {
        let (p1, p0, q0, q1) = (v[5][l], v[6][l], v[7][l], v[8][l]);
        let d10 = ad(p1, p0);
        let e10 = ad(q1, q0);
        hev[l] = mask(d10 > thresh_bd) | mask(e10 > thresh_bd);
        let mut m = mask(d10 > limit_bd)
            | mask(e10 > limit_bd)
            | mask(ad(p0, q0) * 2 + ad(p1, q1) / 2 > blimit_bd);
        if FS >= 6 {
            m |= mask(ad(v[4][l], p1) > limit_bd) | mask(ad(v[9][l], q1) > limit_bd);
        }
        if FS >= 8 {
            m |= mask(ad(v[3][l], v[4][l]) > limit_bd) | mask(ad(v[10][l], v[9][l]) > limit_bd);
        }
        fm[l] = !m;
        if FS >= 6 {
            let mut f = mask(d10 <= one)
                & mask(e10 <= one)
                & mask(ad(v[4][l], p0) <= one)
                & mask(ad(v[9][l], q0) <= one);
            if FS >= 8 {
                f &= mask(ad(v[3][l], p0) <= one) & mask(ad(v[10][l], q0) <= one);
            }
            flat[l] = f & !m;
        }
        if FS == 16 {
            flat2[l] = flat[l]
                & mask(ad(v[0][l], p0) <= one)
                & mask(ad(v[13][l], q0) <= one)
                & mask(ad(v[1][l], p0) <= one)
                & mask(ad(v[12][l], q0) <= one)
                & mask(ad(v[2][l], p0) <= one)
                & mask(ad(v[11][l], q0) <= one);
        }
    }
    if fm.iter().fold(0, |a, &b| a | b) == 0 {
        return false;
    }
    // Narrow filter on every lane; selected below.
    let mut nar = [[0i16; L]; 4];
    for l in 0..L {
        let c4 = |x: i16| x.clamp(lo4, hi4);
        let ps1 = v[5][l] - half;
        let ps0 = v[6][l] - half;
        let qs0 = v[7][l] - half;
        let qs1 = v[8][l] - half;
        let h = hev[l];
        let f = c4(ps1 - qs1) & h;
        let f = c4(f + 3 * (qs0 - ps0));
        let f1 = c4(f + 4) >> 3;
        let f2 = c4(f + 3) >> 3;
        let f3 = (f1 + 1) >> 1;
        let on = fm[l];
        let outer = on & !h;
        nar[0][l] = sel(outer, c4(ps1 + f3) + half, v[5][l]);
        nar[1][l] = sel(on, c4(ps0 + f2) + half, v[6][l]);
        nar[2][l] = sel(on, c4(qs0 - f1) + half, v[7][l]);
        nar[3][l] = sel(outer, c4(qs1 - f3) + half, v[8][l]);
    }
    let any_flat = FS >= 6 && flat.iter().fold(0, |a, &b| a | b) != 0;
    let any_flat2 = FS == 16 && flat2.iter().fold(0, |a, &b| a | b) != 0;
    let mut w8 = [[0i16; L]; 14];
    let mut w16 = [[0i16; L]; 14];
    if any_flat {
        if FS == 6 {
            wide_vec::<2, 1, 3, L>(v, &mut w8);
        } else {
            wide_vec::<3, 0, 3, L>(v, &mut w8);
        }
    }
    if any_flat2 {
        wide_vec::<6, 1, 4, L>(v, &mut w16);
    }
    for (k, nk) in (5..9).zip(0..4) {
        v[k] = nar[nk];
    }
    if any_flat {
        let (a, b) = if FS == 6 { (5, 9) } else { (4, 10) };
        for k in a..b {
            for l in 0..L {
                v[k][l] = sel(flat[l], w8[k][l], v[k][l]);
            }
        }
    }
    if any_flat2 {
        for k in 1..13 {
            for l in 0..L {
                v[k][l] = sel(flat2[l], w16[k][l], v[k][l]);
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{filter_horizontal_edge, filter_vertical_edge};
    use crate::dsp::Pixel;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 16) as u32
        }
        fn range(&mut self, n: u32) -> i32 {
            (self.next() % n) as i32
        }
    }

    /// Literal transliteration of 7.14.4 + 7.14.6 for one sample position.
    #[allow(clippy::too_many_lines)]
    fn ref_sample(
        f: &mut [i32],
        stride: usize,
        x: i32,
        y: i32,
        plane: usize,
        lvl: i32,
        sharp: i32,
        dx: i32,
        dy: i32,
        filter_size: i32,
        bd: u32,
    ) {
        let shift = if sharp > 4 {
            2
        } else if sharp > 0 {
            1
        } else {
            0
        };
        let limit = if sharp > 0 {
            (lvl >> shift).clamp(1, 9 - sharp)
        } else {
            (lvl >> shift).max(1)
        };
        let blimit = 2 * (lvl + 2) + limit;
        let thresh = lvl >> 4;
        let at = |f: &[i32], k: i32| f[((y + dy * k) as usize) * stride + (x + dx * k) as usize];
        let g = |k: i32| at(f, k);
        let (q0, q1, q2, q3) = (g(0), g(1), g(2), g(3));
        let (p0, p1, p2, p3) = (g(-1), g(-2), g(-3), g(-4));
        let mut hev = 0;
        let thresh_bd = thresh << (bd - 8);
        hev |= ((p1 - p0).abs() > thresh_bd) as i32;
        hev |= ((q1 - q0).abs() > thresh_bd) as i32;
        let filter_len = if filter_size == 4 {
            4
        } else if plane != 0 {
            6
        } else if filter_size == 8 {
            8
        } else {
            16
        };
        let limit_bd = limit << (bd - 8);
        let blimit_bd = blimit << (bd - 8);
        let mut mask = 0;
        mask |= ((p1 - p0).abs() > limit_bd) as i32;
        mask |= ((q1 - q0).abs() > limit_bd) as i32;
        mask |= ((p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 > blimit_bd) as i32;
        if filter_len >= 6 {
            mask |= ((p2 - p1).abs() > limit_bd) as i32;
            mask |= ((q2 - q1).abs() > limit_bd) as i32;
        }
        if filter_len >= 8 {
            mask |= ((p3 - p2).abs() > limit_bd) as i32;
            mask |= ((q3 - q2).abs() > limit_bd) as i32;
        }
        let filter_mask = mask == 0;
        let tbd = 1 << (bd - 8);
        let mut flat = 0;
        if filter_size >= 8 {
            let mut m = 0;
            m |= ((p1 - p0).abs() > tbd) as i32;
            m |= ((q1 - q0).abs() > tbd) as i32;
            m |= ((p2 - p0).abs() > tbd) as i32;
            m |= ((q2 - q0).abs() > tbd) as i32;
            if filter_len >= 8 {
                m |= ((p3 - p0).abs() > tbd) as i32;
                m |= ((q3 - q0).abs() > tbd) as i32;
            }
            flat = (m == 0) as i32;
        }
        let mut flat2 = 0;
        if filter_size >= 16 {
            let (q4, q5, q6, p4, p5, p6) = (g(4), g(5), g(6), g(-5), g(-6), g(-7));
            let mut m = 0;
            m |= ((p6 - p0).abs() > tbd) as i32;
            m |= ((q6 - q0).abs() > tbd) as i32;
            m |= ((p5 - p0).abs() > tbd) as i32;
            m |= ((q5 - q0).abs() > tbd) as i32;
            m |= ((p4 - p0).abs() > tbd) as i32;
            m |= ((q4 - q0).abs() > tbd) as i32;
            flat2 = (m == 0) as i32;
        }
        let idx = |k: i32| ((y + dy * k) as usize) * stride + (x + dx * k) as usize;
        if !filter_mask {
        } else if filter_size == 4 || flat == 0 {
            let c4 = |v: i32| v.clamp(-(1 << (bd - 1)), (1 << (bd - 1)) - 1);
            let o = 0x80 << (bd - 8);
            let (ps1, ps0, qs0, qs1) = (p1 - o, p0 - o, q0 - o, q1 - o);
            let mut filter = if hev != 0 { c4(ps1 - qs1) } else { 0 };
            filter = c4(filter + 3 * (qs0 - ps0));
            let filter1 = c4(filter + 4) >> 3;
            let filter2 = c4(filter + 3) >> 3;
            f[idx(0)] = c4(qs0 - filter1) + o;
            f[idx(-1)] = c4(ps0 + filter2) + o;
            if hev == 0 {
                let filter = (filter1 + 1) >> 1;
                f[idx(1)] = c4(qs1 - filter) + o;
                f[idx(-2)] = c4(ps1 + filter) + o;
            }
        } else {
            let log2 = if filter_size == 8 || flat2 == 0 { 3 } else { 4 };
            let n: i32 = if log2 == 4 {
                6
            } else if plane == 0 {
                3
            } else {
                2
            };
            let n2 = if log2 == 3 && plane == 0 { 0 } else { 1 };
            let mut fo = [0i32; 16];
            for i in -n..n {
                let mut t = 0;
                for j in -n..=n {
                    let p = (i + j).clamp(-(n + 1), n);
                    let tap = if j.abs() <= n2 { 2 } else { 1 };
                    t += at(f, p) * tap;
                }
                fo[(i + 8) as usize] = (t + (1 << (log2 - 1))) >> log2;
            }
            for i in -n..n {
                f[idx(i)] = fo[(i + 8) as usize];
            }
        }
    }

    fn run<P: Pixel>(bd: u32, seed: u64) {
        let mut rng = Rng(seed);
        let (w, h) = (40usize, 40usize);
        let max = (1 << bd) - 1;
        let mut seen = [[0u32; 8]; 4];
        for iter in 0..4000 {
            let fs_idx = rng.range(4);
            let (filter_size, kfs, plane) = match fs_idx {
                0 => (4usize, 4, rng.range(2) as usize),
                1 => (6, 8, 1),
                2 => (8, 8, 0),
                _ => (16, 16, 0),
            };
            let lvl = if iter % 7 == 0 { 0 } else { rng.range(64) };
            let sharp = rng.range(8);
            // Smooth-ish data with a step at the edge, so flat paths happen.
            let base = rng.range(max as u32 + 1);
            let noise = [0, 1, 2, 4, 16, 64, 1024][rng.range(7) as usize] << (bd - 8) >> 2;
            let step = [0, 1, 3, 8, 30, 200][rng.range(6) as usize] << (bd - 8) >> 1;
            let vert = rng.range(2) == 0;
            let n = [1usize, 2, 3, 4, 4, 4, 8, 13][rng.range(8) as usize];
            let (ex, ey) = (16usize, 16usize);
            let mut frame = vec![0i32; w * h];
            for yy in 0..h {
                for xx in 0..w {
                    let side = if vert { xx >= ex } else { yy >= ey };
                    let nz = if noise > 0 {
                        rng.range(2 * noise as u32 + 1) - noise
                    } else {
                        0
                    };
                    frame[yy * w + xx] = (base + nz + if side { step } else { 0 }).clamp(0, max);
                }
            }
            let mut buf: Vec<P> = frame.iter().map(|&v| P::from_i32(v)).collect();
            let mut refr = frame.clone();
            if lvl > 0 {
                for i in 0..n {
                    let (x, y, dx, dy) = if vert {
                        (ex as i32, (ey + i) as i32, 1, 0)
                    } else {
                        ((ex + i) as i32, ey as i32, 0, 1)
                    };
                    ref_sample(&mut refr, w, x, y, plane, lvl, sharp, dx, dy, kfs, bd);
                }
            }
            if vert {
                filter_vertical_edge(
                    &mut buf,
                    w,
                    ex,
                    ey,
                    n,
                    filter_size,
                    lvl as u8,
                    sharp as u8,
                    bd,
                );
            } else {
                filter_horizontal_edge(
                    &mut buf,
                    w,
                    ex,
                    ey,
                    n,
                    filter_size,
                    lvl as u8,
                    sharp as u8,
                    bd,
                );
            }
            let got: Vec<i32> = buf.iter().map(|v| v.to_i32()).collect();
            let mut far = 0;
            for (i, (a, b)) in frame.iter().zip(&refr).enumerate() {
                if a != b {
                    let pos = if vert { i % w } else { i / w };
                    let d = if pos < ex { ex - pos } else { pos - ex + 1 };
                    far = far.max(d);
                }
            }
            seen[fs_idx as usize][far] += 1;
            let bad = got.iter().zip(&refr).position(|(a, b)| a != b);
            assert_eq!(
                bad, None,
                "iter {iter} bd {bd} fs {filter_size} vert {vert} n {n} lvl {lvl}"
            );
        }
        // Every filter variant ran: narrow (1 or 2 samples), wide 8 (3) and
        // wide 16 (6).
        for (fs_idx, far) in [
            (0, 1),
            (0, 2),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 3),
            (3, 6),
        ] {
            assert!(seen[fs_idx][far] > 10, "{seen:?}");
        }
    }

    #[test]
    fn lf_matches_spec_8bit() {
        run::<u8>(8, 0x1234_5678_9abc_def1);
    }

    #[test]
    fn lf_matches_spec_10bit() {
        run::<u16>(10, 0x2234_5678_9abc_def1);
    }

    #[test]
    fn lf_matches_spec_12bit() {
        run::<u16>(12, 0x3234_5678_9abc_def1);
    }
}

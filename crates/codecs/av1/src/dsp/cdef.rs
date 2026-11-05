//! Constrained directional enhancement filter (spec 7.15).
//!
//! The caller walks the 8x8 blocks (7.15 / 7.15.1): it copies skipped blocks
//! and blocks with `cdef_idx == -1` unchanged, runs [`cdef_direction`] on the
//! pre-CDEF luma, derives the per-plane strengths and direction, builds a
//! padded copy of the pre-CDEF samples with [`cdef_pad_block`] and calls
//! [`cdef_filter`].
//!
//! Per 7.15.1, for a non-skipped block with `idx = cdef_idx`:
//!
//! * luma: `pri = adjust_luma_pri_strength(cdef_y_pri_strength[idx] << (bd - 8), var)`,
//!   `sec = cdef_y_sec_strength[idx] << (bd - 8)`,
//!   `dir = if unadjusted pri == 0 { 0 } else { yDir }`,
//!   `damping = cdef_damping(CdefDamping, bd, false)`.
//! * chroma: `pri = cdef_uv_pri_strength[idx] << (bd - 8)` (no adjustment),
//!   `sec = cdef_uv_sec_strength[idx] << (bd - 8)`,
//!   `dir = if pri == 0 { 0 } else { cdef_uv_dir(subsampling_x, subsampling_y, yDir) }`,
//!   `damping = cdef_damping(CdefDamping, bd, true)`.
//!
//! The secondary strength is the value after the spec's `cdef_*_sec_strength`
//! mapping (3 becomes 4), shifted by `bd - 8`.

use super::Pixel;
use crate::spec::tables::{CDEF_DIRECTIONS, CDEF_PRI_TAPS, CDEF_SEC_TAPS, CDEF_UV_DIR, DIV_TABLE};

/// Value in the padded source of a sample that is not CdefAvailable.
pub const CDEF_UNAVAILABLE: u16 = i16::MAX as u16;

/// Border, in samples, around a block in the padded source of [`cdef_filter`].
pub const CDEF_BORDER: usize = 2;

/// Weights of the diagonal partial sums in the direction cost: an entry
/// summing `n` samples gets `Div_Table[n]` (the spec's cost loops written as
/// one weighted sum of squares, exact since no cost exceeds `i32`).
const CDEF_W_DIAG: [i32; 16] = [
    840, 420, 280, 210, 168, 140, 120, 105, 120, 140, 168, 210, 280, 420, 840, 0,
];
/// Weights of the `partial[1, 3, 5, 7]` sums, as [`CDEF_W_DIAG`].
const CDEF_W_ALT: [i32; 12] = [420, 210, 140, 105, 105, 105, 105, 105, 140, 210, 420, 0];

/// Direction search on the 8x8 luma block at `src` (7.15.2). Returns
/// `(yDir, var)`.
pub fn cdef_direction<P: Pixel>(src: &[P], stride: usize, bd: u32) -> (u8, i32) {
    let sh = bd - 8;
    let mut img = [[0i32; 8]; 8];
    for (i, r) in img.iter_mut().enumerate() {
        let row = &src[i * stride..i * stride + 8];
        for (d, s) in r.iter_mut().zip(row) {
            *d = (s.to_i32() >> sh) - 128;
        }
    }
    // partial[k] of the spec: hv[0] = partial[2], hv[1] = partial[6],
    // diag[0] = partial[0], diag[1] = partial[4], alt = partial[1, 3, 5, 7].
    let mut hv = [[0i32; 8]; 2];
    let mut diag = [[0i32; 16]; 2];
    let mut alt = [[0i32; 12]; 4];
    for (i, r) in img.iter().enumerate() {
        hv[0][i] = r.iter().sum();
        for j in 0..8 {
            hv[1][j] += r[j];
            diag[0][i + j] += r[j];
            diag[1][7 + i - j] += r[j];
        }
        for k in 0..4 {
            let pair = r[2 * k] + r[2 * k + 1];
            alt[0][i + k] += pair;
            alt[1][3 + i - k] += pair;
        }
    }
    for m in 0..4 {
        for j in 0..8 {
            let pair = img[2 * m][j] + img[2 * m + 1][j];
            alt[2][3 - m + j] += pair;
            alt[3][m + j] += pair;
        }
    }
    let div8 = DIV_TABLE[8] as i32;
    let sq = |v: &[i32], w: &[i32]| -> i32 { v.iter().zip(w).map(|(&a, &b)| a * a * b).sum() };
    let sq8 = |v: &[i32; 8]| -> i32 { v.iter().map(|&a| a * a).sum::<i32>() * div8 };
    let cost = [
        sq(&diag[0], &CDEF_W_DIAG),
        sq(&alt[0], &CDEF_W_ALT),
        sq8(&hv[0]),
        sq(&alt[1], &CDEF_W_ALT),
        sq(&diag[1], &CDEF_W_DIAG),
        sq(&alt[2], &CDEF_W_ALT),
        sq8(&hv[1]),
        sq(&alt[3], &CDEF_W_ALT),
    ];
    let mut best_cost = 0;
    let mut y_dir = 0usize;
    for (i, &c) in cost.iter().enumerate() {
        if c > best_cost {
            best_cost = c;
            y_dir = i;
        }
    }
    let var = (best_cost - cost[(y_dir + 4) & 7]) >> 10;
    (y_dir as u8, var)
}

/// Luma primary strength adjustment of 7.15.1 steps 4 and 5. `pri` is
/// `cdef_y_pri_strength[idx] << (bd - 8)`; `var` comes from
/// [`cdef_direction`].
#[inline]
pub fn adjust_luma_pri_strength(pri: i32, var: i32) -> i32 {
    if var == 0 {
        return 0;
    }
    let v = var >> 6;
    let var_str = if v != 0 {
        (31 - v.leading_zeros() as i32).min(12)
    } else {
        0
    };
    (pri * (4 + var_str) + 8) >> 4
}

/// Damping for one plane (7.15.1 steps 6 and 11): `CdefDamping + bd - 8`,
/// minus one for chroma. Pass the result to [`cdef_filter`].
#[inline]
pub fn cdef_damping(cdef_damping: i32, bd: u32, chroma: bool) -> i32 {
    cdef_damping + (bd as i32 - 8) - chroma as i32
}

/// Chroma direction from the luma direction (`Cdef_Uv_Dir`).
#[inline]
pub fn cdef_uv_dir(sub_x: bool, sub_y: bool, y_dir: u8) -> u8 {
    CDEF_UV_DIR[sub_x as usize][sub_y as usize][(y_dir & 7) as usize]
}

/// Builds the padded source of [`cdef_filter`] for the `w x h` block at
/// `(x0, y0)` of `plane` (pre-CDEF samples, i.e. the spec's CurrFrame):
/// `(h + 4)` rows of `(w + 4)` samples written to `dst` with stride
/// `dst_stride`, the block's top-left sample at `dst[2 * dst_stride + 2]`.
///
/// Samples outside `0..avail_w` x `0..avail_h` get [`CDEF_UNAVAILABLE`]. Per
/// `is_inside_filter_region` (which tests MiRows / MiCols), pass
/// `avail_w = (MiCols * 4) >> subX` and `avail_h = (MiRows * 4) >> subY`; the
/// plane must hold decoded samples throughout that area (which reaches past
/// the frame width / height up to the next multiple of 8 luma samples).
///
/// The block may be larger than one CDEF block (e.g. a whole 64x64), so that
/// several blocks share one padded copy.
pub fn cdef_pad_block<P: Pixel>(
    dst: &mut [u16],
    dst_stride: usize,
    plane: &[P],
    plane_stride: usize,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    avail_w: usize,
    avail_h: usize,
) {
    let b = CDEF_BORDER;
    // Columns `x0 - 2 .. x0 + w + 2` clipped to `0..avail_w`, as offsets in
    // the padded row.
    let xs = x0.saturating_sub(b);
    let xe = (x0 + w + b).min(avail_w).max(xs);
    let lead = xs + b - x0;
    let n = xe - xs;
    for r in 0..h + 4 {
        let out = &mut dst[r * dst_stride..r * dst_stride + w + 4];
        let y = (y0 + r).wrapping_sub(b);
        if y0 + r < b || y >= avail_h {
            out.fill(CDEF_UNAVAILABLE);
            continue;
        }
        let (left, rest) = out.split_at_mut(lead);
        left.fill(CDEF_UNAVAILABLE);
        let (mid, right) = rest.split_at_mut(n);
        let row = &plane[y * plane_stride + xs..y * plane_stride + xe];
        for (o, &s) in mid.iter_mut().zip(row) {
            *o = s.to_i32() as u16;
        }
        right.fill(CDEF_UNAVAILABLE);
    }
}

#[inline(always)]
fn floor_log2(v: i32) -> i32 {
    31 - v.leading_zeros() as i32
}

#[inline(always)]
fn constrain(diff: i32, threshold: i32, shift: u32) -> i32 {
    // threshold == 0 gives 0 here without a special case.
    let ad = diff.abs();
    let v = ad.min((threshold - (ad >> shift)).max(0));
    if diff < 0 { -v } else { v }
}

/// Filters one `w x h` block (7.15.3) into `dst`.
///
/// `src` is the padded pre-CDEF block from [`cdef_pad_block`] (block sample
/// `(i, j)` at `src[(i + 2) * src_stride + j + 2]`, unavailable samples equal
/// to [`CDEF_UNAVAILABLE`]). `pri`, `sec` are the final strengths (already
/// shifted by `bd - 8`, luma primary adjusted), `dir` the direction, and
/// `damping` the per-plane damping from [`cdef_damping`] (at most 10, as in
/// any conformant stream).
pub fn cdef_filter<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    src: &[u16],
    src_stride: usize,
    w: usize,
    h: usize,
    pri: i32,
    sec: i32,
    dir: u8,
    damping: i32,
    bd: u32,
) {
    if pri == 0 && sec == 0 {
        // Every tap constrains to 0: the block is unchanged.
        for i in 0..h {
            let s = &src[(i + CDEF_BORDER) * src_stride + CDEF_BORDER..][..w];
            for (d, &v) in dst[i * dst_stride..][..w].iter_mut().zip(s) {
                *d = P::from_i32(i32::from(v));
            }
        }
        return;
    }
    let p = Params::new(src_stride, pri, sec, dir, damping, bd);
    match (w, h % 4) {
        (8, 0) => dispatch::<P, 8, 4, 32>(dst, dst_stride, src, src_stride, h, &p),
        (4, 0) => dispatch::<P, 4, 4, 16>(dst, dst_stride, src, src_stride, h, &p),
        _ => filter_scalar(dst, dst_stride, src, src_stride, w, h, &p),
    }
}

/// Per-block constants of [`cdef_filter`].
struct Params {
    pri: i32,
    sec: i32,
    pri_shift: u32,
    sec_shift: u32,
    /// Primary taps for k = 0, 1.
    pt: [i32; 2],
    /// Offsets in the padded source of the primary taps (k = 0, 1) and the
    /// secondary taps (k = 0: dir + 2, dir - 2; k = 1: dir + 2, dir - 2).
    pri_off: [isize; 2],
    sec_off: [isize; 4],
}

impl Params {
    #[inline(always)]
    fn new(stride: usize, pri: i32, sec: i32, dir: u8, damping: i32, bd: u32) -> Self {
        let s = stride as isize;
        let dir = (dir & 7) as usize;
        let off = |d: usize, k: usize| -> isize {
            let t = CDEF_DIRECTIONS[d][k];
            t[0] as isize * s + t[1] as isize
        };
        let tap_set = ((pri >> (bd - 8)) & 1) as usize;
        let shift = |v: i32| {
            if v != 0 {
                (damping - floor_log2(v)).max(0) as u32
            } else {
                0
            }
        };
        Self {
            pri,
            sec,
            pri_shift: shift(pri),
            sec_shift: shift(sec),
            pt: [
                CDEF_PRI_TAPS[tap_set][0] as i32,
                CDEF_PRI_TAPS[tap_set][1] as i32,
            ],
            pri_off: [off(dir, 0), off(dir, 1)],
            sec_off: [
                off((dir + 2) & 7, 0),
                off((dir + 6) & 7, 0),
                off((dir + 2) & 7, 1),
                off((dir + 6) & 7, 1),
            ],
        }
    }
}

#[inline(always)]
fn dispatch<P: Pixel, const W: usize, const R: usize, const L: usize>(
    dst: &mut [P],
    dst_stride: usize,
    src: &[u16],
    src_stride: usize,
    h: usize,
    p: &Params,
) {
    match (p.pri != 0, p.sec != 0) {
        (true, true) => block::<P, W, R, L, true, true>(dst, dst_stride, src, src_stride, h, p),
        (true, false) => block::<P, W, R, L, true, false>(dst, dst_stride, src, src_stride, h, p),
        _ => block::<P, W, R, L, false, true>(dst, dst_stride, src, src_stride, h, p),
    }
}

/// Loads `R` rows of `W` samples starting at `start` into `L = W * R` lanes.
#[inline(always)]
fn load<const W: usize, const R: usize, const L: usize>(
    src: &[u16],
    start: usize,
    stride: usize,
) -> [i16; L] {
    let mut v = [0i16; L];
    for r in 0..R {
        let row = &src[start + r * stride..start + r * stride + W];
        for (d, &s) in v[r * W..r * W + W].iter_mut().zip(row) {
            *d = s as i16;
        }
    }
    v
}

/// Adds `tap * constrain(p - x)` for both signs of `off` to `sum`, and
/// (when `MINMAX`) tracks the minimum / maximum of the available taps in
/// `mn` / `mx` (the latter as `p + 1` wrapping, so unavailable samples, which
/// are `i16::MAX`, become `i16::MIN` and never win; they never win the
/// minimum either).
///
/// Everything runs on signed 16-bit lanes. For an unavailable `p`,
/// `|p - x| >= 32767 - 4095`, whose shift by at most
/// `damping - floor_log2(thr)` still exceeds `thr` (damping is at most 10):
/// `thr - (|d| >> shift)` is then negative and the tap 0. Every constrained
/// value is at most 240 (12-bit primary strength), so the weighted sums stay
/// far inside `i16`.
#[inline(always)]
fn taps<const W: usize, const R: usize, const L: usize, const MINMAX: bool>(
    src: &[u16],
    c: usize,
    stride: usize,
    off: isize,
    x: &[i16; L],
    thr: i16,
    shift: u32,
    tap: i16,
    sum: &mut [i16; L],
    mn: &mut [i16; L],
    mx: &mut [i16; L],
) {
    for o in [off, -off] {
        let p = load::<W, R, L>(src, (c as isize + o) as usize, stride);
        for l in 0..L {
            let d = p[l].wrapping_sub(x[l]);
            let ad = d.max(d.wrapping_neg());
            let t = thr.wrapping_sub(ad >> shift).max(0);
            let v = ad.min(t);
            let m = d >> 15;
            let v = (v ^ m).wrapping_sub(m);
            sum[l] = sum[l].wrapping_add(tap.wrapping_mul(v));
        }
        if MINMAX {
            for l in 0..L {
                mn[l] = mn[l].min(p[l]);
                mx[l] = mx[l].max(p[l].wrapping_add(1));
            }
        }
    }
}

/// [`cdef_filter`] for blocks `W` samples wide, `R` rows at a time
/// (`L = W * R` lanes). With only one of the primary / secondary filters
/// the result never leaves the range of the taps (their weights add up to
/// 12 < 16), so the clamp is only needed when both run.
#[inline(always)]
fn block<
    P: Pixel,
    const W: usize,
    const R: usize,
    const L: usize,
    const PRI: bool,
    const SEC: bool,
>(
    dst: &mut [P],
    dst_stride: usize,
    src: &[u16],
    stride: usize,
    h: usize,
    p: &Params,
) {
    let b = CDEF_BORDER;
    let (pri, sec) = (p.pri as i16, p.sec as i16);
    let pt = [p.pt[0] as i16, p.pt[1] as i16];
    let st = CDEF_SEC_TAPS[0].map(i16::from);
    let mut i = 0;
    while i < h {
        let c = (i + b) * stride + b;
        let x = load::<W, R, L>(src, c, stride);
        let mut sum = [0i16; L];
        let mut mn = x;
        let mut mx = x;
        for v in &mut mx {
            *v += 1;
        }
        if PRI {
            for k in 0..2 {
                taps::<W, R, L, SEC>(
                    src,
                    c,
                    stride,
                    p.pri_off[k],
                    &x,
                    pri,
                    p.pri_shift,
                    pt[k],
                    &mut sum,
                    &mut mn,
                    &mut mx,
                );
            }
        }
        if SEC {
            for (k, &off) in p.sec_off.iter().enumerate() {
                taps::<W, R, L, PRI>(
                    src,
                    c,
                    stride,
                    off,
                    &x,
                    sec,
                    p.sec_shift,
                    st[k >> 1],
                    &mut sum,
                    &mut mn,
                    &mut mx,
                );
            }
        }
        let mut y = [0i16; L];
        for l in 0..L {
            let s = sum[l];
            let s = (s + 8 + (s >> 15)) >> 4;
            y[l] = x[l] + s;
            if PRI && SEC {
                y[l] = y[l].max(mn[l]).min(mx[l] - 1);
            }
        }
        for r in 0..R {
            let out = &mut dst[(i + r) * dst_stride..(i + r) * dst_stride + W];
            for j in 0..W {
                out[j] = P::from_i32(i32::from(y[r * W + j]));
            }
        }
        i += R;
    }
}

/// Any-size fallback of [`cdef_filter`].
fn filter_scalar<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    src: &[u16],
    src_stride: usize,
    w: usize,
    h: usize,
    p: &Params,
) {
    let st = CDEF_SEC_TAPS[0].map(i32::from);
    let b = CDEF_BORDER;
    for i in 0..h {
        let base = (i + b) * src_stride + b;
        let out = &mut dst[i * dst_stride..i * dst_stride + w];
        for j in 0..w {
            let c = base + j;
            let x = src[c] as i32;
            let mut sum = 0i32;
            let mut mx = (x as i16).wrapping_add(1);
            let mut mn = x as i16;
            for k in 0..2 {
                for sign in [-1isize, 1] {
                    let v = src[(c as isize + sign * p.pri_off[k]) as usize];
                    sum += p.pt[k] * constrain(v as i32 - x, p.pri, p.pri_shift);
                    mx = mx.max((v as i16).wrapping_add(1));
                    mn = mn.min(v as i16);
                    for so in [p.sec_off[2 * k], p.sec_off[2 * k + 1]] {
                        let q = src[(c as isize + sign * so) as usize];
                        sum += st[k] * constrain(q as i32 - x, p.sec, p.sec_shift);
                        mx = mx.max((q as i16).wrapping_add(1));
                        mn = mn.min(q as i16);
                    }
                }
            }
            let mx = mx as i32 - 1;
            let mn = mn as i32;
            let y = x + ((8 + sum - (sum < 0) as i32) >> 4);
            out[j] = P::from_i32(y.clamp(mn, mx));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CDEF_UNAVAILABLE, adjust_luma_pri_strength, cdef_damping, cdef_direction, cdef_filter,
        cdef_pad_block, cdef_uv_dir,
    };
    use crate::dsp::Pixel;
    use crate::spec::tables::{CDEF_DIRECTIONS, CDEF_PRI_TAPS, CDEF_SEC_TAPS, DIV_TABLE};

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

    fn floor_log2(v: i32) -> i32 {
        let mut s = 0;
        let mut x = v;
        while x > 1 {
            x >>= 1;
            s += 1;
        }
        s
    }

    /// 7.15.2 transliterated.
    fn ref_direction(f: &[i32], stride: usize, x0: usize, y0: usize, bd: u32) -> (i32, i32) {
        let mut cost = [0i64; 8];
        let mut partial = [[0i64; 15]; 8];
        let mut best_cost = 0;
        let mut y_dir = 0;
        for i in 0..8 {
            for j in 0..8 {
                let x = ((f[(y0 + i) * stride + x0 + j] >> (bd - 8)) - 128) as i64;
                partial[0][i + j] += x;
                partial[1][i + j / 2] += x;
                partial[2][i] += x;
                partial[3][3 + i - j / 2] += x;
                partial[4][7 + i - j] += x;
                partial[5][3 - i / 2 + j] += x;
                partial[6][j] += x;
                partial[7][i / 2 + j] += x;
            }
        }
        let d = |k: usize| DIV_TABLE[k] as i64;
        for i in 0..8 {
            cost[2] += partial[2][i] * partial[2][i];
            cost[6] += partial[6][i] * partial[6][i];
        }
        cost[2] *= d(8);
        cost[6] *= d(8);
        for i in 0..7 {
            cost[0] += (partial[0][i] * partial[0][i] + partial[0][14 - i] * partial[0][14 - i])
                * d(i + 1);
            cost[4] += (partial[4][i] * partial[4][i] + partial[4][14 - i] * partial[4][14 - i])
                * d(i + 1);
        }
        cost[0] += partial[0][7] * partial[0][7] * d(8);
        cost[4] += partial[4][7] * partial[4][7] * d(8);
        let mut i = 1;
        while i < 8 {
            for j in 0..5 {
                cost[i] += partial[i][3 + j] * partial[i][3 + j];
            }
            cost[i] *= d(8);
            for j in 0..3 {
                cost[i] += (partial[i][j] * partial[i][j]
                    + partial[i][10 - j] * partial[i][10 - j])
                    * d(2 * j + 2);
            }
            i += 2;
        }
        for i in 0..8 {
            if cost[i] > best_cost {
                best_cost = cost[i];
                y_dir = i;
            }
        }
        let var = (best_cost - cost[(y_dir + 4) & 7]) >> 10;
        (y_dir as i32, var as i32)
    }

    fn constrain(diff: i32, threshold: i32, damping: i32) -> i32 {
        if threshold == 0 {
            return 0;
        }
        let adj = (damping - floor_log2(threshold)).max(0);
        let sign = if diff < 0 { -1 } else { 1 };
        sign * (threshold - (diff.abs() >> adj)).clamp(0, diff.abs())
    }

    /// 7.15.3 transliterated on a whole plane with the MiRows / MiCols region.
    fn ref_filter(
        f: &[i32],
        stride: usize,
        mi_rows: i32,
        mi_cols: i32,
        sub_x: u32,
        sub_y: u32,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        pri: i32,
        sec: i32,
        damping: i32,
        dir: usize,
        bd: u32,
    ) -> Vec<i32> {
        let cs = bd - 8;
        let mut out = vec![0; w * h];
        for i in 0..h {
            for j in 0..w {
                let mut sum = 0;
                let x = f[(y0 + i) * stride + x0 + j];
                let mut max = x;
                let mut min = x;
                let get = |d: usize, k: usize, sign: i32| -> Option<i32> {
                    let y = (y0 + i) as i32 + sign * CDEF_DIRECTIONS[d][k][0] as i32;
                    let xx = (x0 + j) as i32 + sign * CDEF_DIRECTIONS[d][k][1] as i32;
                    let cr = (y << sub_y) >> 2;
                    let cc = (xx << sub_x) >> 2;
                    if cc >= 0 && cc < mi_cols && cr >= 0 && cr < mi_rows {
                        Some(f[y as usize * stride + xx as usize])
                    } else {
                        None
                    }
                };
                for k in 0..2 {
                    for sign in [-1, 1] {
                        if let Some(p) = get(dir, k, sign) {
                            sum += CDEF_PRI_TAPS[((pri >> cs) & 1) as usize][k] as i32
                                * constrain(p - x, pri, damping);
                            max = max.max(p);
                            min = min.min(p);
                        }
                        for dir_off in [-2i32, 2] {
                            if let Some(s) = get(((dir as i32 + dir_off) & 7) as usize, k, sign) {
                                sum += CDEF_SEC_TAPS[((pri >> cs) & 1) as usize][k] as i32
                                    * constrain(s - x, sec, damping);
                                max = max.max(s);
                                min = min.min(s);
                            }
                        }
                    }
                }
                out[i * w + j] = (x + ((8 + sum - (sum < 0) as i32) >> 4)).clamp(min, max);
            }
        }
        out
    }

    fn run<P: Pixel>(bd: u32, seed: u64) {
        let mut rng = Rng(seed);
        let max = (1 << bd) - 1;
        let mut dirs_seen = [0u32; 8];
        for iter in 0..3000 {
            let mi_cols = 2 + 2 * rng.range(4);
            let mi_rows = 2 + 2 * rng.range(4);
            let (sub_x, sub_y) = match rng.range(4) {
                0 => (0u32, 0u32),
                1 => (1, 1),
                2 => (1, 0),
                _ => (0, 0),
            };
            let chroma = rng.range(2) == 1;
            let (sx, sy) = if chroma { (sub_x, sub_y) } else { (0, 0) };
            let pw = ((mi_cols * 4) >> sx) as usize;
            let ph = ((mi_rows * 4) >> sy) as usize;
            // Plane with a garbage margin past the available region.
            let stride = pw + 9;
            let rows = ph + 9;
            let mut f = vec![0i32; stride * rows];
            let style = rng.range(3);
            let (gx, gy) = (rng.range(64) - 32, rng.range(64) - 32);
            let base = rng.range(max as u32 + 1);
            let noise = [1, 4, 16, 256, 4096][rng.range(5) as usize] << (bd - 8) >> 2;
            for y in 0..rows {
                for x in 0..stride {
                    let v = match style {
                        0 => rng.range(max as u32 + 1),
                        1 => base + ((gx * x as i32 + gy * y as i32) << (bd - 8) >> 2),
                        _ => base + rng.range(2 * noise as u32 + 1) - noise,
                    };
                    f[y * stride + x] = v.clamp(0, max);
                }
            }
            let plane: Vec<P> = f.iter().map(|&v| P::from_i32(v)).collect();
            let (bw, bh) = (8 >> sx, 8 >> sy);
            let bx = bw * rng.range((mi_cols / 2) as u32) as usize;
            let by = bh * rng.range((mi_rows / 2) as u32) as usize;
            // Direction on a luma-sized 8x8 window (any position works).
            let (dx0, dy0) = (
                rng.range((stride - 8) as u32) as usize,
                rng.range((rows - 8) as u32) as usize,
            );
            let (rd, rv) = ref_direction(&f, stride, dx0, dy0, bd);
            let (gd, gv) = cdef_direction(&plane[dy0 * stride + dx0..], stride, bd);
            assert_eq!((gd as i32, gv), (rd, rv), "direction iter {iter}");
            dirs_seen[gd as usize] += 1;

            let pri_raw = rng.range(16) << (bd - 8);
            let sec_raw = [0, 1, 2, 4][rng.range(4) as usize] << (bd - 8);
            let var = [0, 1, 63, 64, 200, 5000, 1 << 20][rng.range(7) as usize];
            let pri = if chroma {
                pri_raw
            } else {
                adjust_luma_pri_strength(pri_raw, var)
            };
            let dir = rng.range(8) as u8;
            let dir = if chroma {
                cdef_uv_dir(sub_x == 1, sub_y == 1, dir)
            } else {
                dir
            };
            let damping = cdef_damping(3 + rng.range(4), bd, chroma);

            let expect = ref_filter(
                &f,
                stride,
                mi_rows,
                mi_cols,
                sx,
                sy,
                bx,
                by,
                bw,
                bh,
                pri,
                sec_raw,
                damping,
                dir as usize,
                bd,
            );
            let pstride = bw + 4 + rng.range(3) as usize;
            let mut pad = vec![0u16; pstride * (bh + 4)];
            cdef_pad_block(&mut pad, pstride, &plane, stride, bx, by, bw, bh, pw, ph);
            let mut out = vec![P::default(); bw * bh];
            cdef_filter(
                &mut out, bw, &pad, pstride, bw, bh, pri, sec_raw, dir, damping, bd,
            );
            let got: Vec<i32> = out.iter().map(|v| v.to_i32()).collect();
            assert_eq!(got, expect, "filter iter {iter} bd {bd}");
            assert!(pad.contains(&CDEF_UNAVAILABLE) || (bx > 0 && by > 0));
        }
        assert!(dirs_seen.iter().all(|&c| c > 0), "{dirs_seen:?}");
    }

    #[test]
    fn adjust_matches_spec() {
        for pri in 0..64 {
            for var in [
                0,
                1,
                63,
                64,
                127,
                128,
                4095,
                4096,
                1 << 18,
                1 << 19,
                1 << 25,
            ] {
                let v = var >> 6;
                let var_str = if v != 0 { floor_log2(v).min(12) } else { 0 };
                let e = if var != 0 {
                    (pri * (4 + var_str) + 8) >> 4
                } else {
                    0
                };
                assert_eq!(adjust_luma_pri_strength(pri, var), e);
            }
        }
    }

    #[test]
    fn cdef_matches_spec_8bit() {
        run::<u8>(8, 0x9e37_79b9_7f4a_7c15);
    }

    #[test]
    fn cdef_matches_spec_10bit() {
        run::<u16>(10, 0x1e37_79b9_7f4a_7c15);
    }

    #[test]
    fn cdef_matches_spec_12bit() {
        run::<u16>(12, 0x2e37_79b9_7f4a_7c15);
    }
}

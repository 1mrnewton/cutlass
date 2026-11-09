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
pub const CDEF_UNAVAILABLE: u16 = 0x7000;

/// Border, in samples, around a block in the padded source of [`cdef_filter`].
pub const CDEF_BORDER: usize = 2;

/// Stride of the padded copy of one 64x64 area (luma) that [`cdef_filter`]
/// runs its vector kernel on.
pub const CDEF_PAD_STRIDE: usize = 64 + 2 * CDEF_BORDER;

/// Weights of the diagonal partial sums in the direction cost: an entry
/// summing `n` samples gets `Div_Table[n]` (the spec's cost loops written as
/// one weighted sum of squares, exact since no cost exceeds `i32`).
const CDEF_W_DIAG: [i32; 16] = [
    840, 420, 280, 210, 168, 140, 120, 105, 120, 140, 168, 210, 280, 420, 840, 0,
];
/// Weights of the `partial[1, 3, 5, 7]` sums, as [`CDEF_W_DIAG`].
const CDEF_W_ALT: [i32; 16] = [
    420, 210, 140, 105, 105, 105, 105, 105, 140, 210, 420, 0, 0, 0, 0, 0,
];

/// Eight 16-bit lanes.
type Lanes = [i16; 8];

/// Adds `r` to lanes `i..i + 8` of the 16 lanes `acc`.
#[inline(always)]
fn add_at(acc: &mut [Lanes; 2], r: &Lanes, i: usize) {
    let mut lo = [0i16; 8];
    let mut hi = [0i16; 8];
    lo[i..].copy_from_slice(&r[..8 - i]);
    hi[..i].copy_from_slice(&r[8 - i..]);
    for l in 0..8 {
        acc[0][l] = acc[0][l].wrapping_add(lo[l]);
        acc[1][l] = acc[1][l].wrapping_add(hi[l]);
    }
}

/// Sum of `v[l]^2 * w[l]` over the 16 lanes.
#[inline(always)]
fn weighted_squares(v: &[Lanes; 2], w: &[i32; 16]) -> i32 {
    let mut f = [0i32; 16];
    for l in 0..8 {
        f[l] = i32::from(v[0][l]);
        f[l + 8] = i32::from(v[1][l]);
    }
    let mut s = 0i32;
    for l in 0..16 {
        s = s.wrapping_add(f[l].wrapping_mul(f[l]).wrapping_mul(w[l]));
    }
    s
}

/// Direction search on the 8x8 luma block at `src` (7.15.2). Returns
/// `(yDir, var)`.
///
/// The partial sums run on 16-bit lanes (a sum covers at most 16 samples
/// of `-128..=127`): every row is added at its own lane offset, so the
/// sums along the diagonals become sums of whole lane vectors.
pub fn cdef_direction<P: Pixel>(src: &[P], stride: usize, bd: u32) -> (u8, i32) {
    let sh = bd - 8;
    let mut img = [[0i16; 8]; 8];
    for (i, r) in img.iter_mut().enumerate() {
        let row = &src[i * stride..i * stride + 8];
        for (d, s) in r.iter_mut().zip(row) {
            *d = ((s.to_i32() >> sh) - 128) as i16;
        }
    }
    // partial[k] of the spec: hv0 = partial[2], hv1 = partial[6],
    // d0 = partial[0], d1 = partial[4], a0..a3 = partial[1, 3, 5, 7].
    let z = [[0i16; 8]; 2];
    let (mut d0, mut d1, mut a0, mut a1, mut a2, mut a3) = (z, z, z, z, z, z);
    let mut hv0 = [0i32; 8];
    let mut hv1 = [0i16; 8];
    macro_rules! row {
        ($i:literal) => {{
            let r = &img[$i];
            for l in 0..8 {
                hv1[l] = hv1[l].wrapping_add(r[l]);
            }
            add_at(&mut d0, r, $i);
            add_at(&mut d1, &std::array::from_fn(|l| r[7 - l]), $i);
            // Sums of column pairs, in lanes 0..4.
            let mut p = [0i16; 8];
            for k in 0..4 {
                p[k] = r[2 * k].wrapping_add(r[2 * k + 1]);
            }
            hv0[$i] = i32::from(p[0]) + i32::from(p[1]) + i32::from(p[2]) + i32::from(p[3]);
            add_at(&mut a0, &p, $i);
            let mut q = [0i16; 8];
            for k in 0..4 {
                q[k] = p[3 - k];
            }
            add_at(&mut a1, &q, $i);
        }};
    }
    row!(0);
    row!(1);
    row!(2);
    row!(3);
    row!(4);
    row!(5);
    row!(6);
    row!(7);
    // Sums of row pairs.
    macro_rules! pair {
        ($m:literal) => {{
            let mut p = [0i16; 8];
            for l in 0..8 {
                p[l] = img[2 * $m][l].wrapping_add(img[2 * $m + 1][l]);
            }
            add_at(&mut a2, &p, 3 - $m);
            add_at(&mut a3, &p, $m);
        }};
    }
    pair!(0);
    pair!(1);
    pair!(2);
    pair!(3);
    let div8 = DIV_TABLE[8] as i32;
    let sq8 = |v: [i32; 8]| -> i32 { v.iter().map(|&a| a * a).sum::<i32>() * div8 };
    let cost = [
        weighted_squares(&d0, &CDEF_W_DIAG),
        weighted_squares(&a0, &CDEF_W_ALT),
        sq8(hv0),
        weighted_squares(&a1, &CDEF_W_ALT),
        weighted_squares(&d1, &CDEF_W_DIAG),
        weighted_squares(&a2, &CDEF_W_ALT),
        sq8(hv1.map(i32::from)),
        weighted_squares(&a3, &CDEF_W_ALT),
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
///
/// 8x8, 4x4 and 4x8 blocks with `src_stride == CDEF_PAD_STRIDE` take the
/// vector kernel; anything else the scalar one.
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
    if src_stride == CDEF_PAD_STRIDE {
        let k = Kernel::new(&p);
        match (w, h) {
            (8, 8) => return dispatch::<P, 8, 8, 1, 8>(dst, dst_stride, src, &k),
            (4, 4) => return dispatch::<P, 4, 4, 2, 8>(dst, dst_stride, src, &k),
            (4, 8) => return dispatch::<P, 4, 8, 2, 8>(dst, dst_stride, src, &k),
            _ => {}
        }
    }
    filter_scalar(dst, dst_stride, src, src_stride, w, h, &p);
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

/// Largest distance, in the padded source with stride [`CDEF_PAD_STRIDE`],
/// between a sample and one of its taps.
const TAP_SPAN: usize = 2 * CDEF_PAD_STRIDE + 2;

/// [`Params`] for the vector kernel: tap offsets relative to the padded
/// block's first sample, `o` for the tap at `+off` and `2 * TAP_SPAN - o` for
/// the one at `-off` (so both stay inside the padded block).
struct Kernel {
    pri_thr: u16,
    sec_thr: u16,
    pri_shift: u32,
    sec_shift: u32,
    pt: [i16; 2],
    pri_o: [usize; 2],
    sec_o: [usize; 4],
}

impl Kernel {
    #[inline(always)]
    fn new(p: &Params) -> Self {
        let o = |off: isize| (off + TAP_SPAN as isize).clamp(0, 2 * TAP_SPAN as isize) as usize;
        Self {
            pri_thr: p.pri as u16,
            sec_thr: p.sec as u16,
            pri_shift: p.pri_shift.min(15),
            sec_shift: p.sec_shift.min(15),
            pt: [p.pt[0] as i16, p.pt[1] as i16],
            pri_o: p.pri_off.map(o),
            sec_o: p.sec_off.map(o),
        }
    }
}

#[inline(always)]
fn dispatch<P: Pixel, const W: usize, const H: usize, const R: usize, const L: usize>(
    dst: &mut [P],
    dst_stride: usize,
    src: &[u16],
    k: &Kernel,
) {
    match (k.pri_thr != 0, k.sec_thr != 0) {
        (true, true) => block::<P, W, H, R, L, true, true>(dst, dst_stride, src, k),
        (true, false) => block::<P, W, H, R, L, true, false>(dst, dst_stride, src, k),
        _ => block::<P, W, H, R, L, false, true>(dst, dst_stride, src, k),
    }
}

/// `R` rows of `W` samples of [`CDEF_PAD_STRIDE`]-strided `win` from
/// column `o` (at most `2 * TAP_SPAN`), as `L = W * R` lanes.
#[inline(always)]
fn gather<const W: usize, const R: usize, const L: usize>(win: &[u16], o: usize) -> [i16; L] {
    let o = o.min(2 * TAP_SPAN);
    let mut v = [0i16; L];
    for r in 0..R {
        let s = r * CDEF_PAD_STRIDE + o;
        let src: &[u16; W] = win[s..s + W].try_into().unwrap_or(&[0; W]);
        for j in 0..W {
            v[r * W + j] = src[j] as i16;
        }
    }
    v
}

/// Adds `tap * constrain(p - x)` for the taps at `o` and its mirror image,
/// and (when `MINMAX`) folds them into `mn` / `mx`.
///
/// Everything runs on 16-bit lanes. `constrain(d)` is `d` clamped to
/// `+-max(0, thr - (|d| >> shift))`, the lower bound a saturating unsigned
/// subtraction. An unavailable tap ([`CDEF_UNAVAILABLE`]) has
/// `|d| >= 0x7000 - 4095`, which shifted by at most 6 (the damping minus the
/// strength's log2, both scaled by `bd - 8`) still exceeds any threshold
/// (at most 240): it adds 0. It never wins the minimum, and `p & 0xfff`,
/// which keeps every real sample (at most 12 bits) and clears the marker,
/// leaves it out of the maximum.
#[inline(always)]
fn taps<const W: usize, const R: usize, const L: usize, const MINMAX: bool, const MUL: bool>(
    win: &[u16],
    o: usize,
    x: &[i16; L],
    thr: u16,
    shift: u32,
    tap: i16,
    sum: &mut [i16; L],
    mn: &mut [i16; L],
    mx: &mut [i16; L],
) {
    let o = o.min(2 * TAP_SPAN);
    for o in [o, 2 * TAP_SPAN - o] {
        let p = gather::<W, R, L>(win, o);
        for l in 0..L {
            let d = p[l].wrapping_sub(x[l]);
            let ad = d.max(0i16.wrapping_sub(d)) as u16;
            let t = thr.saturating_sub(ad.wrapping_shr(shift)) as i16;
            let v = d.max(0i16.wrapping_sub(t)).min(t);
            let v = if MUL { tap.wrapping_mul(v) } else { v };
            sum[l] = sum[l].wrapping_add(v);
        }
        if MINMAX {
            for l in 0..L {
                mn[l] = mn[l].min(p[l]);
                mx[l] = mx[l].max(p[l] & 0xfff);
            }
        }
    }
}

/// [`cdef_filter`] for `W x H` blocks in a padded source of stride
/// [`CDEF_PAD_STRIDE`], `R` rows (`L = W * R` samples) at a time. With only one of
/// the primary / secondary filters the result never leaves the range of the
/// taps (their weights add up to 12 < 16), so the clamp is only needed when
/// both run.
#[inline(always)]
fn block<
    P: Pixel,
    const W: usize,
    const H: usize,
    const R: usize,
    const L: usize,
    const PRI: bool,
    const SEC: bool,
>(
    dst: &mut [P],
    dst_stride: usize,
    src: &[u16],
    k: &Kernel,
) {
    for i in 0..H / R {
        let row = i * R;
        // The padded rows `row .. row + R + 4`.
        let s = row * CDEF_PAD_STRIDE;
        let Some(win) = src.get(s..s + (R - 1) * CDEF_PAD_STRIDE + 2 * TAP_SPAN + W) else {
            return;
        };
        let x = gather::<W, R, L>(win, TAP_SPAN);
        let mut sum = [0i16; L];
        let mut mn = x;
        let mut mx = x;
        if PRI {
            for t in 0..2 {
                taps::<W, R, L, SEC, true>(
                    win,
                    k.pri_o[t],
                    &x,
                    k.pri_thr,
                    k.pri_shift,
                    k.pt[t],
                    &mut sum,
                    &mut mn,
                    &mut mx,
                );
            }
        }
        if SEC {
            // Weight 2 for the inner taps, 1 for the outer ones.
            let mut s2 = [0i16; L];
            for t in 0..2 {
                taps::<W, R, L, PRI, false>(
                    win,
                    k.sec_o[t],
                    &x,
                    k.sec_thr,
                    k.sec_shift,
                    1,
                    &mut s2,
                    &mut mn,
                    &mut mx,
                );
            }
            for l in 0..L {
                sum[l] = sum[l].wrapping_add(s2[l].wrapping_add(s2[l]));
            }
            for t in 2..4 {
                taps::<W, R, L, PRI, false>(
                    win,
                    k.sec_o[t],
                    &x,
                    k.sec_thr,
                    k.sec_shift,
                    1,
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
                y[l] = y[l].max(mn[l]).min(mx[l]);
            }
        }
        for r in 0..R {
            let Some(out) = dst.get_mut((row + r) * dst_stride..(row + r) * dst_stride + W) else {
                return;
            };
            for j in 0..W {
                out[j] = P::from_i32(i32::from(y[r * W + j]));
            }
        }
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
            let mut mx = x as i16;
            let mut mn = x as i16;
            for k in 0..2 {
                for sign in [-1isize, 1] {
                    let v = src[(c as isize + sign * p.pri_off[k]) as usize];
                    sum += p.pt[k] * constrain(v as i32 - x, p.pri, p.pri_shift);
                    mx = mx.max((v & 0xfff) as i16);
                    mn = mn.min(v as i16);
                    for so in [p.sec_off[2 * k], p.sec_off[2 * k + 1]] {
                        let q = src[(c as isize + sign * so) as usize];
                        sum += st[k] * constrain(q as i32 - x, p.sec, p.sec_shift);
                        mx = mx.max((q & 0xfff) as i16);
                        mn = mn.min(q as i16);
                    }
                }
            }
            let mx = mx as i32;
            let mn = mn as i32;
            let y = x + ((8 + sum - (sum < 0) as i32) >> 4);
            out[j] = P::from_i32(y.clamp(mn, mx));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CDEF_PAD_STRIDE, CDEF_UNAVAILABLE, adjust_luma_pri_strength, cdef_damping, cdef_direction,
        cdef_filter, cdef_pad_block, cdef_uv_dir,
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
            let pstride = if rng.range(2) == 0 {
                CDEF_PAD_STRIDE
            } else {
                bw + 4 + rng.range(3) as usize
            };
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

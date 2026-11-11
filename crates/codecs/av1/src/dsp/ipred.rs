//! Intra prediction (specification section 7.11.2) and chroma from luma
//! (section 7.11.5).
//!
//! The caller builds the `AboveRow` and `LeftCol` arrays of section 7.11.2.1
//! (including the replication rules and the corner sample at index -1). The
//! kernels here run everything after that: the DC, smooth, Paeth, and
//! directional processes, the intra edge filter and upsampler, the recursive
//! filter intra process, and CFL.
//!
//! Spec index `i` of `AboveRow`/`LeftCol` lives at buffer index
//! `EDGE_OFF + i`.

use super::{Pixel, clip_px, round2, round2_signed};
use crate::spec::tables::{
    DR_INTRA_DERIVATIVE, INTRA_EDGE_KERNEL, INTRA_FILTER_SCALE_BITS, INTRA_FILTER_TAPS,
    MODE_TO_ANGLE, SM_WEIGHTS_TX_4X4, SM_WEIGHTS_TX_8X8, SM_WEIGHTS_TX_16X16, SM_WEIGHTS_TX_32X32,
    SM_WEIGHTS_TX_64X64,
};

/// Buffer index of spec index 0 of `AboveRow`/`LeftCol`. Index -1 (the corner)
/// is at `EDGE_OFF - 1`; upsampling writes down to index -2.
pub const EDGE_OFF: usize = 16;
/// Length of an edge buffer: room for spec indices -16..=2*128+31.
pub const EDGE_LEN: usize = 16 + 2 * 128 + 32;

const DC_PRED: u8 = 0;
const V_PRED: u8 = 1;
const D67_PRED: u8 = 8;
const SMOOTH_PRED: u8 = 9;
const SMOOTH_V_PRED: u8 = 10;
const SMOOTH_H_PRED: u8 = 11;

/// The two edge arrays of one transform block.
#[derive(Clone)]
pub struct IntraEdge<P: Pixel> {
    pub above: [P; EDGE_LEN],
    pub left: [P; EDGE_LEN],
}

impl<P: Pixel> IntraEdge<P> {
    pub fn new() -> Self {
        Self {
            above: [P::default(); EDGE_LEN],
            left: [P::default(); EDGE_LEN],
        }
    }
}

impl<P: Pixel> Default for IntraEdge<P> {
    fn default() -> Self {
        Self::new()
    }
}

#[inline(always)]
fn row_mut<P: Pixel>(dst: &mut [P], stride: usize, i: usize, w: usize) -> &mut [P] {
    &mut dst[i * stride..i * stride + w]
}

/// Two-tap directional interpolation with a 5-bit phase.
#[inline(always)]
fn interp<P: Pixel>(a: P, b: P, shift: i32) -> P {
    P::from_i32((a.to_i32() * (32 - shift) + b.to_i32() * shift + 16) >> 5)
}

fn sm_weights(n: usize) -> &'static [u8] {
    match n {
        4 => &SM_WEIGHTS_TX_4X4,
        8 => &SM_WEIGHTS_TX_8X8,
        16 => &SM_WEIGHTS_TX_16X16,
        32 => &SM_WEIGHTS_TX_32X32,
        _ => &SM_WEIGHTS_TX_64X64,
    }
}

/// Intra prediction for a transform block that lies wholly inside the frame
/// (`x + w - 1 <= maxX` and `y + h - 1 <= maxY`). See [`predict_clipped`].
pub fn predict<P: Pixel>(
    dst: &mut [P],
    stride: usize,
    w: usize,
    h: usize,
    mode: u8,
    angle_delta: i8,
    above: &mut [P; EDGE_LEN],
    left: &mut [P; EDGE_LEN],
    have_left: bool,
    have_above: bool,
    smooth_filter: bool,
    enable_intra_edge_filter: bool,
    bd: u32,
) {
    predict_clipped(
        dst,
        stride,
        w,
        h,
        mode,
        angle_delta,
        above,
        left,
        have_left,
        have_above,
        smooth_filter,
        enable_intra_edge_filter,
        w,
        h,
        bd,
    );
}

/// Intra prediction (spec 7.11.2 after the edge arrays are built).
///
/// Modes: DC_PRED=0, V_PRED=1, H_PRED=2, D45..D67 = 3..8, SMOOTH=9,
/// SMOOTH_V=10, SMOOTH_H=11, PAETH=12. `angle_delta` in -3..=3 is used by the
/// directional modes 1..=8 only. `w`, `h` are powers of two in 4..=64.
/// `smooth_filter` is filterType from get_filter_type.
///
/// `edge_w` = Min(w, maxX - x + 1) and `edge_h` = Min(h, maxY - y + 1): the
/// number of above / left samples inside the frame. They only change the
/// length of the intra edge filter.
///
/// The edges are modified in place (corner filter, edge filter, upsampling),
/// as in the specification.
pub fn predict_clipped<P: Pixel>(
    dst: &mut [P],
    stride: usize,
    w: usize,
    h: usize,
    mode: u8,
    angle_delta: i8,
    above: &mut [P; EDGE_LEN],
    left: &mut [P; EDGE_LEN],
    have_left: bool,
    have_above: bool,
    smooth_filter: bool,
    enable_intra_edge_filter: bool,
    edge_w: usize,
    edge_h: usize,
    bd: u32,
) {
    match mode {
        DC_PRED => pred_dc(dst, stride, w, h, above, left, have_left, have_above, bd),
        V_PRED..=D67_PRED => {
            let p_angle = MODE_TO_ANGLE[mode as usize] as i32 + angle_delta as i32 * 3;
            pred_directional(
                dst,
                stride,
                w,
                h,
                p_angle,
                above,
                left,
                have_left,
                have_above,
                smooth_filter,
                enable_intra_edge_filter,
                edge_w,
                edge_h,
                bd,
            );
        }
        SMOOTH_PRED => pred_smooth(dst, stride, w, h, above, left),
        SMOOTH_V_PRED => pred_smooth_v(dst, stride, w, h, above, left),
        SMOOTH_H_PRED => pred_smooth_h(dst, stride, w, h, above, left),
        _ => pred_paeth(dst, stride, w, h, above, left),
    }
}

fn fill<P: Pixel>(dst: &mut [P], stride: usize, w: usize, h: usize, v: P) {
    for i in 0..h {
        fill_row(row_mut(dst, stride, i, w), v);
    }
}

/// Sets every sample of `row` to `v`: a block's row is 4 to 64 samples,
/// which take fixed-size stores in place of a call to `memset`.
#[inline(always)]
fn fill_row<P: Pixel>(row: &mut [P], v: P) {
    macro_rules! put {
        ($($n:literal)*) => {
            match row.len() {
                $($n => row.copy_from_slice(&[v; $n]),)*
                _ => row.fill(v),
            }
        };
    }
    put!(4 8 16 32 64);
}

/// Copies `src` into `row`, as [`fill_row`] for a block's row.
#[inline(always)]
fn copy_row<P: Pixel>(row: &mut [P], src: &[P]) {
    macro_rules! put {
        ($($n:literal)*) => {
            match row.len() {
                $($n => row.copy_from_slice(&src[..$n]),)*
                _ => row.copy_from_slice(src),
            }
        };
    }
    put!(4 8 16 32 64);
}

fn sum_px<P: Pixel>(s: &[P]) -> u32 {
    s.iter().map(|p| p.to_i32() as u32).sum()
}

fn pred_dc<P: Pixel>(
    dst: &mut [P],
    stride: usize,
    w: usize,
    h: usize,
    above: &[P; EDGE_LEN],
    left: &[P; EDGE_LEN],
    have_left: bool,
    have_above: bool,
    bd: u32,
) {
    let v = match (have_left, have_above) {
        (true, true) => {
            let n = (w + h) as u32;
            let sum =
                sum_px(&left[EDGE_OFF..EDGE_OFF + h]) + sum_px(&above[EDGE_OFF..EDGE_OFF + w]);
            ((sum + (n >> 1)) / n) as i32
        }
        (true, false) => {
            let sum = sum_px(&left[EDGE_OFF..EDGE_OFF + h]);
            ((sum + (h as u32 >> 1)) >> h.trailing_zeros()) as i32
        }
        (false, true) => {
            let sum = sum_px(&above[EDGE_OFF..EDGE_OFF + w]);
            ((sum + (w as u32 >> 1)) >> w.trailing_zeros()) as i32
        }
        (false, false) => 1 << (bd - 1),
    };
    fill(dst, stride, w, h, P::from_i32(v));
}

fn pred_paeth<P: Pixel>(
    dst: &mut [P],
    stride: usize,
    w: usize,
    h: usize,
    above: &[P; EDGE_LEN],
    left: &[P; EDGE_LEN],
) {
    let tl_p = above[EDGE_OFF - 1];
    let tl = tl_p.to_i32();
    let top = &above[EDGE_OFF..EDGE_OFF + w];
    for i in 0..h {
        let l_p = left[EDGE_OFF + i];
        let l = l_p.to_i32();
        let p_top = (l - tl).abs();
        let row = row_mut(dst, stride, i, w);
        for (d, &t_p) in row.iter_mut().zip(top) {
            let t = t_p.to_i32();
            // base = t + l - tl; pLeft = |base - l|, pTop = |base - t|.
            let p_left = (t - tl).abs();
            let p_tl = (t + l - 2 * tl).abs();
            *d = if p_left <= p_top && p_left <= p_tl {
                l_p
            } else if p_top <= p_tl {
                t_p
            } else {
                tl_p
            };
        }
    }
}

fn pred_smooth<P: Pixel>(
    dst: &mut [P],
    stride: usize,
    w: usize,
    h: usize,
    above: &[P; EDGE_LEN],
    left: &[P; EDGE_LEN],
) {
    let wx = sm_weights(w);
    let wy = sm_weights(h);
    let bl = left[EDGE_OFF + h - 1].to_i32();
    let tr = above[EDGE_OFF + w - 1].to_i32();
    let mut a = [0i32; 64];
    let mut wxa = [0i32; 64];
    let mut colc = [0i32; 64];
    for j in 0..w {
        a[j] = above[EDGE_OFF + j].to_i32();
        wxa[j] = wx[j] as i32;
        colc[j] = (256 - wx[j] as i32) * tr + 256;
    }
    let (a, wxa, colc) = (&a[..w], &wxa[..w], &colc[..w]);
    for i in 0..h {
        let wyi = wy[i] as i32;
        let l = left[EDGE_OFF + i].to_i32();
        let rowc = (256 - wyi) * bl;
        let row = row_mut(dst, stride, i, w);
        for j in 0..w {
            row[j] = P::from_i32((wyi * a[j] + rowc + wxa[j] * l + colc[j]) >> 9);
        }
    }
}

fn pred_smooth_v<P: Pixel>(
    dst: &mut [P],
    stride: usize,
    w: usize,
    h: usize,
    above: &[P; EDGE_LEN],
    left: &[P; EDGE_LEN],
) {
    let wy = sm_weights(h);
    let bl = left[EDGE_OFF + h - 1].to_i32();
    let top = &above[EDGE_OFF..EDGE_OFF + w];
    for i in 0..h {
        let wyi = wy[i] as i32;
        let rowc = (256 - wyi) * bl + 128;
        let row = row_mut(dst, stride, i, w);
        for (d, &t) in row.iter_mut().zip(top) {
            *d = P::from_i32((wyi * t.to_i32() + rowc) >> 8);
        }
    }
}

fn pred_smooth_h<P: Pixel>(
    dst: &mut [P],
    stride: usize,
    w: usize,
    h: usize,
    above: &[P; EDGE_LEN],
    left: &[P; EDGE_LEN],
) {
    let wx = sm_weights(w);
    let tr = above[EDGE_OFF + w - 1].to_i32();
    let mut wxa = [0i32; 64];
    let mut colc = [0i32; 64];
    for j in 0..w {
        wxa[j] = wx[j] as i32;
        colc[j] = (256 - wx[j] as i32) * tr + 128;
    }
    let (wxa, colc) = (&wxa[..w], &colc[..w]);
    for i in 0..h {
        let l = left[EDGE_OFF + i].to_i32();
        let row = row_mut(dst, stride, i, w);
        for j in 0..w {
            row[j] = P::from_i32((wxa[j] * l + colc[j]) >> 8);
        }
    }
}

/// Intra edge filter strength selection (spec 7.11.2.9).
fn edge_filter_strength(w: usize, h: usize, smooth: bool, delta: i32) -> usize {
    let d = delta.abs();
    let blk_wh = w + h;
    if !smooth {
        if blk_wh <= 8 {
            (d >= 56) as usize
        } else if blk_wh <= 16 {
            (d >= 40) as usize
        } else if blk_wh <= 24 {
            if d >= 32 {
                3
            } else if d >= 16 {
                2
            } else if d >= 8 {
                1
            } else {
                0
            }
        } else if blk_wh <= 32 {
            if d >= 32 {
                3
            } else if d >= 4 {
                2
            } else {
                1
            }
        } else {
            3
        }
    } else if blk_wh <= 8 {
        if d >= 64 {
            2
        } else if d >= 40 {
            1
        } else {
            0
        }
    } else if blk_wh <= 16 {
        if d >= 48 {
            2
        } else if d >= 20 {
            1
        } else {
            0
        }
    } else if blk_wh <= 24 {
        if d >= 4 { 3 } else { 0 }
    } else {
        3
    }
}

/// Intra edge upsample selection (spec 7.11.2.10).
fn use_upsample(w: usize, h: usize, smooth: bool, delta: i32) -> bool {
    let d = delta.abs();
    if d <= 0 || d >= 40 {
        false
    } else if !smooth {
        w + h <= 16
    } else {
        w + h <= 8
    }
}

/// Intra edge filter (spec 7.11.2.12) on spec indices -1..sz-2 of `buf`.
fn edge_filter<P: Pixel>(buf: &mut [P; EDGE_LEN], sz: usize, strength: usize) {
    if strength == 0 || sz == 0 {
        return;
    }
    // e[k + 2] = edge[Clip3(0, sz - 1, k)] for k in -2..sz+2.
    let mut e = [0i32; 129 + 4];
    let src = &buf[EDGE_OFF - 1..EDGE_OFF - 1 + sz];
    for (d, s) in e[2..2 + sz].iter_mut().zip(src) {
        *d = s.to_i32();
    }
    e[0] = e[2];
    e[1] = e[2];
    e[sz + 2] = e[sz + 1];
    e[sz + 3] = e[sz + 1];
    let k = &INTRA_EDGE_KERNEL[strength - 1];
    let (k0, k1, k2, k3, k4) = (
        k[0] as i32,
        k[1] as i32,
        k[2] as i32,
        k[3] as i32,
        k[4] as i32,
    );
    let out = &mut buf[EDGE_OFF..EDGE_OFF + sz - 1];
    for (i, d) in out.iter_mut().enumerate() {
        // Filters edge index i + 1, whose taps are e[i + 1..i + 6].
        let s = k0 * e[i + 1] + k1 * e[i + 2] + k2 * e[i + 3] + k3 * e[i + 4] + k4 * e[i + 5];
        *d = P::from_i32((s + 8) >> 4);
    }
}

/// Intra edge upsample (spec 7.11.2.11). Valid input at -1..num_px-1, output
/// at -2..2*num_px-2.
fn edge_upsample<P: Pixel>(buf: &mut [P; EDGE_LEN], num_px: usize, bd: u32) {
    let mut dup = [0i32; 16 + 3];
    dup[0] = buf[EDGE_OFF - 1].to_i32();
    for i in 0..=num_px {
        dup[i + 1] = buf[EDGE_OFF - 1 + i].to_i32();
    }
    dup[num_px + 2] = buf[EDGE_OFF + num_px - 1].to_i32();
    buf[EDGE_OFF - 2] = P::from_i32(dup[0]);
    for i in 0..num_px {
        let s = -dup[i] + 9 * dup[i + 1] + 9 * dup[i + 2] - dup[i + 3];
        buf[EDGE_OFF + 2 * i - 1] = clip_px(round2(s, 4), bd);
        buf[EDGE_OFF + 2 * i] = P::from_i32(dup[i + 2]);
    }
}

#[inline(always)]
fn at(i: isize) -> usize {
    (EDGE_OFF as isize + i) as usize
}

fn pred_directional<P: Pixel>(
    dst: &mut [P],
    stride: usize,
    w: usize,
    h: usize,
    p_angle: i32,
    above: &mut [P; EDGE_LEN],
    left: &mut [P; EDGE_LEN],
    have_left: bool,
    have_above: bool,
    smooth: bool,
    enable_edge_filter: bool,
    edge_w: usize,
    edge_h: usize,
    bd: u32,
) {
    let mut up_a = 0usize;
    let mut up_l = 0usize;
    if enable_edge_filter {
        if p_angle != 90 && p_angle != 180 {
            if p_angle > 90 && p_angle < 180 && w + h >= 24 {
                let s = left[EDGE_OFF].to_i32() * 5
                    + above[EDGE_OFF - 1].to_i32() * 6
                    + above[EDGE_OFF].to_i32() * 5;
                let c = P::from_i32(round2(s, 4));
                above[EDGE_OFF - 1] = c;
                left[EDGE_OFF - 1] = c;
            }
            if have_above {
                let strength = edge_filter_strength(w, h, smooth, p_angle - 90);
                let n = w.min(edge_w) + if p_angle < 90 { h } else { 0 } + 1;
                edge_filter(above, n, strength);
            }
            if have_left {
                let strength = edge_filter_strength(w, h, smooth, p_angle - 180);
                let n = h.min(edge_h) + if p_angle > 180 { w } else { 0 } + 1;
                edge_filter(left, n, strength);
            }
        }
        if use_upsample(w, h, smooth, p_angle - 90) {
            up_a = 1;
            edge_upsample(above, w + if p_angle < 90 { h } else { 0 }, bd);
        }
        if use_upsample(w, h, smooth, p_angle - 180) {
            up_l = 1;
            edge_upsample(left, h + if p_angle > 180 { w } else { 0 }, bd);
        }
    }

    if p_angle == 90 {
        let top = &above[EDGE_OFF..EDGE_OFF + w];
        for i in 0..h {
            copy_row(row_mut(dst, stride, i, w), top);
        }
    } else if p_angle == 180 {
        for i in 0..h {
            fill_row(row_mut(dst, stride, i, w), left[EDGE_OFF + i]);
        }
    } else if p_angle < 90 {
        let dx = DR_INTRA_DERIVATIVE[p_angle as usize] as usize;
        z1(dst, stride, w, h, dx, above, up_a);
    } else if p_angle < 180 {
        let dx = DR_INTRA_DERIVATIVE[(180 - p_angle) as usize] as isize;
        let dy = DR_INTRA_DERIVATIVE[(p_angle - 90) as usize] as isize;
        if up_a == 0 && up_l == 0 {
            z2(dst, stride, w, h, dx, dy, above, left);
        } else {
            z2_generic(dst, stride, w, h, dx, dy, above, left, up_a, up_l);
        }
    } else {
        let dy = DR_INTRA_DERIVATIVE[(270 - p_angle) as usize] as usize;
        z3(dst, stride, w, h, dy, left, up_l);
    }
}

/// Zone 1: pAngle < 90, predicts from the above row only.
fn z1<P: Pixel>(
    dst: &mut [P],
    stride: usize,
    w: usize,
    h: usize,
    dx: usize,
    above: &[P; EDGE_LEN],
    up: usize,
) {
    let max_base = (w + h - 1) << up;
    let fill_v = above[EDGE_OFF + max_base];
    if up == 0 {
        for i in 0..h {
            let idx = (i + 1) * dx;
            let base0 = idx >> 6;
            let shift = ((idx >> 1) & 31) as i32;
            let n = max_base.saturating_sub(base0).min(w);
            let row = row_mut(dst, stride, i, w);
            let (pred, rest) = row.split_at_mut(n);
            let a0 = &above[EDGE_OFF + base0..EDGE_OFF + base0 + n];
            let a1 = &above[EDGE_OFF + base0 + 1..EDGE_OFF + base0 + 1 + n];
            for ((d, &p0), &p1) in pred.iter_mut().zip(a0).zip(a1) {
                *d = interp(p0, p1, shift);
            }
            rest.fill(fill_v);
        }
    } else {
        for i in 0..h {
            let idx = (i + 1) * dx;
            let base0 = idx >> 5;
            let shift = (idx & 31) as i32;
            let row = row_mut(dst, stride, i, w);
            for (j, d) in row.iter_mut().enumerate() {
                let base = base0 + (j << 1);
                *d = if base < max_base {
                    interp(above[EDGE_OFF + base], above[EDGE_OFF + base + 1], shift)
                } else {
                    fill_v
                };
            }
        }
    }
}

/// Zone 3: pAngle > 180, predicts from the left column only.
fn z3<P: Pixel>(
    dst: &mut [P],
    stride: usize,
    w: usize,
    h: usize,
    dy: usize,
    left: &[P; EDGE_LEN],
    up: usize,
) {
    for j in 0..w {
        let idx = (j + 1) * dy;
        let base0 = idx >> (6 - up);
        let shift = (((idx << up) >> 1) & 31) as i32;
        let col = &left[EDGE_OFF + base0..EDGE_OFF + base0 + (h << up) + 1];
        if up == 0 {
            for i in 0..h {
                dst[i * stride + j] = interp(col[i], col[i + 1], shift);
            }
        } else {
            for i in 0..h {
                dst[i * stride + j] = interp(col[2 * i], col[2 * i + 1], shift);
            }
        }
    }
}

/// Zone 2 (90 < pAngle < 180) without upsampling.
fn z2<P: Pixel>(
    dst: &mut [P],
    stride: usize,
    w: usize,
    h: usize,
    dx: isize,
    dy: isize,
    above: &[P; EDGE_LEN],
    left: &[P; EDGE_LEN],
) {
    for i in 0..h {
        let off = -((i as isize + 1) * dx);
        let b = off >> 6;
        let shift = ((off >> 1) & 31) as i32;
        // Above is used where j + b >= -1.
        let j0 = (-1 - b).clamp(0, w as isize) as usize;
        let row = row_mut(dst, stride, i, w);
        let (lpart, apart) = row.split_at_mut(j0);
        let n = w - j0;
        if n > 0 {
            let s = at(j0 as isize + b);
            let a0 = &above[s..s + n];
            let a1 = &above[s + 1..s + 1 + n];
            for ((d, &p0), &p1) in apart.iter_mut().zip(a0).zip(a1) {
                *d = interp(p0, p1, shift);
            }
        }
        let iy = (i as isize) << 6;
        for (j, d) in lpart.iter_mut().enumerate() {
            let idx = iy - (j as isize + 1) * dy;
            let base = idx >> 6;
            let sh = ((idx >> 1) & 31) as i32;
            let k = at(base);
            *d = interp(left[k], left[k + 1], sh);
        }
    }
}

/// Zone 2 with upsampling of either edge, literal per-sample form.
fn z2_generic<P: Pixel>(
    dst: &mut [P],
    stride: usize,
    w: usize,
    h: usize,
    dx: isize,
    dy: isize,
    above: &[P; EDGE_LEN],
    left: &[P; EDGE_LEN],
    up_a: usize,
    up_l: usize,
) {
    let min_base = -(1isize << up_a);
    for i in 0..h {
        let row = row_mut(dst, stride, i, w);
        for (j, d) in row.iter_mut().enumerate() {
            let idx = ((j as isize) << 6) - (i as isize + 1) * dx;
            let base = idx >> (6 - up_a);
            *d = if base >= min_base {
                let shift = (((idx << up_a) >> 1) & 31) as i32;
                interp(above[at(base)], above[at(base + 1)], shift)
            } else {
                let idx = ((i as isize) << 6) - (j as isize + 1) * dy;
                let base = idx >> (6 - up_l);
                let shift = (((idx << up_l) >> 1) & 31) as i32;
                interp(left[at(base)], left[at(base + 1)], shift)
            };
        }
    }
}

/// Recursive intra prediction process (spec 7.11.2.3), `mode` is
/// filter_intra_mode 0..=4. Reads `AboveRow[-1..w-1]` and `LeftCol[-1..h-1]`.
pub fn filter_intra<P: Pixel>(
    dst: &mut [P],
    stride: usize,
    w: usize,
    h: usize,
    mode: u8,
    above: &[P; EDGE_LEN],
    left: &[P; EDGE_LEN],
    bd: u32,
) {
    let taps = &INTRA_FILTER_TAPS[mode as usize];
    let mut t = [[0i32; 7]; 8];
    for k in 0..8 {
        for m in 0..7 {
            t[k][m] = taps[k][m] as i32;
        }
    }
    let max = (1i32 << bd) - 1;
    let sb = INTRA_FILTER_SCALE_BITS as u32;
    for i2 in 0..h >> 1 {
        let r = i2 << 1;
        for j4 in 0..w >> 2 {
            let c = j4 << 2;
            let mut p = [0i32; 7];
            if i2 == 0 {
                for i in 0..5 {
                    p[i] = above[EDGE_OFF + c + i - 1].to_i32();
                }
            } else {
                let prev = (r - 1) * stride;
                p[0] = if j4 == 0 {
                    left[EDGE_OFF + r - 1].to_i32()
                } else {
                    dst[prev + c - 1].to_i32()
                };
                for i in 1..5 {
                    p[i] = dst[prev + c + i - 1].to_i32();
                }
            }
            if j4 == 0 {
                p[5] = left[EDGE_OFF + r].to_i32();
                p[6] = left[EDGE_OFF + r + 1].to_i32();
            } else {
                p[5] = dst[r * stride + c - 1].to_i32();
                p[6] = dst[(r + 1) * stride + c - 1].to_i32();
            }
            for k in 0..8 {
                let tk = &t[k];
                let mut pr = 0;
                for m in 0..7 {
                    pr += tk[m] * p[m];
                }
                let v = round2_signed(pr, sb).clamp(0, max);
                dst[(r + (k >> 2)) * stride + c + (k & 3)] = P::from_i32(v);
            }
        }
    }
}

/// CFL step 1 (spec 7.11.5): writes `L[i][j] - lumaAvg` into `ac` (row major,
/// stride `w`) for a `w`x`h` chroma transform block. `luma` starts at the luma
/// sample co-located with the block's top-left sample. Columns `>= valid_w`
/// and rows `>= valid_h` replicate the last valid column / row, as the
/// specification's clamp of lumaX / lumaY does.
pub fn cfl_ac<P: Pixel>(
    ac: &mut [i32],
    luma: &[P],
    luma_stride: usize,
    w: usize,
    h: usize,
    sub_x: bool,
    sub_y: bool,
    valid_w: usize,
    valid_h: usize,
) {
    let vw = valid_w.clamp(1, w);
    let vh = valid_h.clamp(1, h);
    for i in 0..vh {
        let ly = i << sub_y as usize;
        let r0 = &luma[ly * luma_stride..];
        let out = &mut ac[i * w..i * w + w];
        match (sub_x, sub_y) {
            (true, true) => {
                let r1 = &luma[(ly + 1) * luma_stride..];
                for j in 0..vw {
                    let t = r0[2 * j].to_i32()
                        + r0[2 * j + 1].to_i32()
                        + r1[2 * j].to_i32()
                        + r1[2 * j + 1].to_i32();
                    out[j] = t << 1;
                }
            }
            (true, false) => {
                for j in 0..vw {
                    out[j] = (r0[2 * j].to_i32() + r0[2 * j + 1].to_i32()) << 2;
                }
            }
            (false, true) => {
                let r1 = &luma[(ly + 1) * luma_stride..];
                for j in 0..vw {
                    out[j] = (r0[j].to_i32() + r1[j].to_i32()) << 2;
                }
            }
            (false, false) => {
                for j in 0..vw {
                    out[j] = r0[j].to_i32() << 3;
                }
            }
        }
        let last = out[vw - 1];
        out[vw..].fill(last);
    }
    for i in vh..h {
        ac.copy_within((vh - 1) * w..vh * w, i * w);
    }
    let ac = &mut ac[..w * h];
    let sum: i32 = ac.iter().sum();
    let avg = round2(sum, w.trailing_zeros() + h.trailing_zeros());
    for v in ac.iter_mut() {
        *v -= avg;
    }
}

/// CFL step 2 (spec 7.11.5): `dst` holds the DC prediction; adds
/// `Round2Signed(alpha * ac, 6)` and clips. `alpha` is CflAlphaU/V.
pub fn cfl_pred<P: Pixel>(
    dst: &mut [P],
    stride: usize,
    w: usize,
    h: usize,
    ac: &[i32],
    alpha: i32,
    bd: u32,
) {
    let max = (1i32 << bd) - 1;
    for i in 0..h {
        let row = row_mut(dst, stride, i, w);
        let a = &ac[i * w..i * w + w];
        for (d, &v) in row.iter_mut().zip(a) {
            let m = alpha * v;
            let r = (m.abs() + 32) >> 6;
            let s = if m < 0 { -r } else { r };
            *d = P::from_i32((d.to_i32() + s).clamp(0, max));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EDGE_LEN, EDGE_OFF, IntraEdge, cfl_ac, cfl_pred, filter_intra, predict, predict_clipped,
    };
    use crate::dsp::Pixel;
    use crate::spec::tables::{
        DR_INTRA_DERIVATIVE, INTRA_EDGE_KERNEL, INTRA_FILTER_TAPS, MODE_TO_ANGLE,
        SM_WEIGHTS_TX_4X4, SM_WEIGHTS_TX_8X8, SM_WEIGHTS_TX_16X16, SM_WEIGHTS_TX_32X32,
        SM_WEIGHTS_TX_64X64,
    };

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
        fn below(&mut self, n: u32) -> u32 {
            self.next() % n
        }
    }

    const SIZES: [(usize, usize); 19] = [
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (4, 8),
        (8, 4),
        (4, 16),
        (16, 4),
        (8, 16),
        (16, 8),
        (8, 32),
        (32, 8),
        (16, 32),
        (32, 16),
        (16, 64),
        (64, 16),
        (32, 64),
        (64, 32),
    ];

    fn round2(x: i32, n: u32) -> i32 {
        if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
    }

    /// Edge arrays of the reference, indexed by spec index.
    struct Edges {
        a: Vec<i32>,
        l: Vec<i32>,
    }
    impl Edges {
        fn a(&self, i: i32) -> i32 {
            self.a[(EDGE_OFF as i32 + i) as usize]
        }
        fn l(&self, i: i32) -> i32 {
            self.l[(EDGE_OFF as i32 + i) as usize]
        }
        fn buf(&mut self, left: bool) -> &mut Vec<i32> {
            if left { &mut self.l } else { &mut self.a }
        }
    }

    fn sm(log2: u32) -> &'static [u8] {
        match log2 {
            2 => &SM_WEIGHTS_TX_4X4,
            3 => &SM_WEIGHTS_TX_8X8,
            4 => &SM_WEIGHTS_TX_16X16,
            5 => &SM_WEIGHTS_TX_32X32,
            _ => &SM_WEIGHTS_TX_64X64,
        }
    }

    // Literal transliteration, including the spec's duplicated branch.
    #[allow(clippy::if_same_then_else)]
    fn strength(w: i32, h: i32, ft: i32, delta: i32) -> i32 {
        let d = delta.abs();
        let blk_wh = w + h;
        let mut strength = 0;
        if ft == 0 {
            if blk_wh <= 8 {
                if d >= 56 {
                    strength = 1
                }
            } else if blk_wh <= 12 {
                if d >= 40 {
                    strength = 1
                }
            } else if blk_wh <= 16 {
                if d >= 40 {
                    strength = 1
                }
            } else if blk_wh <= 24 {
                if d >= 8 {
                    strength = 1
                }
                if d >= 16 {
                    strength = 2
                }
                if d >= 32 {
                    strength = 3
                }
            } else if blk_wh <= 32 {
                strength = 1;
                if d >= 4 {
                    strength = 2
                }
                if d >= 32 {
                    strength = 3
                }
            } else {
                strength = 3
            }
        } else if blk_wh <= 8 {
            if d >= 40 {
                strength = 1
            }
            if d >= 64 {
                strength = 2
            }
        } else if blk_wh <= 16 {
            if d >= 20 {
                strength = 1
            }
            if d >= 48 {
                strength = 2
            }
        } else if blk_wh <= 24 {
            if d >= 4 {
                strength = 3
            }
        } else {
            strength = 3
        }
        strength
    }

    fn upsample_sel(w: i32, h: i32, ft: i32, delta: i32) -> bool {
        let d = delta.abs();
        if d <= 0 || d >= 40 {
            false
        } else if ft == 0 {
            w + h <= 16
        } else {
            w + h <= 8
        }
    }

    fn ref_edge_filter(e: &mut Edges, sz: i32, strength: i32, left: bool) {
        if strength == 0 {
            return;
        }
        let edge: Vec<i32> = (0..sz)
            .map(|i| if left { e.l(i - 1) } else { e.a(i - 1) })
            .collect();
        for i in 1..sz {
            let mut s = 0;
            for j in 0..5 {
                let k = (i - 2 + j).clamp(0, sz - 1);
                s += INTRA_EDGE_KERNEL[(strength - 1) as usize][j as usize] as i32
                    * edge[k as usize];
            }
            e.buf(left)[(EDGE_OFF as i32 + i - 1) as usize] = (s + 8) >> 4;
        }
    }

    fn ref_upsample(e: &mut Edges, num_px: i32, left: bool, bd: u32) {
        let buf = e.buf(left);
        let o = EDGE_OFF as i32;
        let g = |b: &Vec<i32>, i: i32| b[(o + i) as usize];
        let mut dup = vec![0; (num_px + 3) as usize];
        dup[0] = g(buf, -1);
        for i in -1..num_px {
            dup[(i + 2) as usize] = g(buf, i);
        }
        dup[(num_px + 2) as usize] = g(buf, num_px - 1);
        buf[(o - 2) as usize] = dup[0];
        for i in 0..num_px as usize {
            let s = -dup[i] + 9 * dup[i + 1] + 9 * dup[i + 2] - dup[i + 3];
            let s = round2(s, 4).clamp(0, (1 << bd) - 1);
            buf[(o + 2 * i as i32 - 1) as usize] = s;
            buf[(o + 2 * i as i32) as usize] = dup[i + 2];
        }
    }

    fn ref_predict(
        e: &mut Edges,
        w: i32,
        h: i32,
        mode: i32,
        angle_delta: i32,
        have_left: bool,
        have_above: bool,
        ft: i32,
        edge_filter: bool,
        edge_w: i32,
        edge_h: i32,
        bd: u32,
    ) -> Vec<i32> {
        let mut pred = vec![0; (w * h) as usize];
        let log2w = w.trailing_zeros();
        let log2h = h.trailing_zeros();
        let mut set = |i: i32, j: i32, v: i32| pred[(i * w + j) as usize] = v;
        if (1..=8).contains(&mode) {
            let p_angle = MODE_TO_ANGLE[mode as usize] as i32 + angle_delta * 3;
            let mut up_a = 0;
            let mut up_l = 0;
            if edge_filter {
                if p_angle != 90 && p_angle != 180 {
                    if p_angle > 90 && p_angle < 180 && (w + h) >= 24 {
                        let s = e.l(0) * 5 + e.a(-1) * 6 + e.a(0) * 5;
                        let c = round2(s, 4);
                        e.a[EDGE_OFF - 1] = c;
                        e.l[EDGE_OFF - 1] = c;
                    }
                    if have_above {
                        let st = strength(w, h, ft, p_angle - 90);
                        let n = w.min(edge_w) + if p_angle < 90 { h } else { 0 } + 1;
                        ref_edge_filter(e, n, st, false);
                    }
                    if have_left {
                        let st = strength(w, h, ft, p_angle - 180);
                        let n = h.min(edge_h) + if p_angle > 180 { w } else { 0 } + 1;
                        ref_edge_filter(e, n, st, true);
                    }
                }
                if upsample_sel(w, h, ft, p_angle - 90) {
                    up_a = 1;
                    ref_upsample(e, w + if p_angle < 90 { h } else { 0 }, false, bd);
                }
                if upsample_sel(w, h, ft, p_angle - 180) {
                    up_l = 1;
                    ref_upsample(e, h + if p_angle > 180 { w } else { 0 }, true, bd);
                }
            }
            let dx = if p_angle < 90 {
                DR_INTRA_DERIVATIVE[p_angle as usize] as i32
            } else if p_angle > 90 && p_angle < 180 {
                DR_INTRA_DERIVATIVE[(180 - p_angle) as usize] as i32
            } else {
                0
            };
            let dy = if p_angle > 90 && p_angle < 180 {
                DR_INTRA_DERIVATIVE[(p_angle - 90) as usize] as i32
            } else if p_angle > 180 {
                DR_INTRA_DERIVATIVE[(270 - p_angle) as usize] as i32
            } else {
                0
            };
            for i in 0..h {
                for j in 0..w {
                    let v = if p_angle < 90 {
                        let idx = (i + 1) * dx;
                        let base = (idx >> (6 - up_a)) + (j << up_a);
                        let shift = ((idx << up_a) >> 1) & 0x1f;
                        let max_base_x = (w + h - 1) << up_a;
                        if base < max_base_x {
                            round2(e.a(base) * (32 - shift) + e.a(base + 1) * shift, 5)
                        } else {
                            e.a(max_base_x)
                        }
                    } else if p_angle > 90 && p_angle < 180 {
                        let idx = (j << 6) - (i + 1) * dx;
                        let base = idx >> (6 - up_a);
                        if base >= -(1 << up_a) {
                            let shift = ((idx << up_a) >> 1) & 0x1f;
                            round2(e.a(base) * (32 - shift) + e.a(base + 1) * shift, 5)
                        } else {
                            let idx = (i << 6) - (j + 1) * dy;
                            let base = idx >> (6 - up_l);
                            assert!(base >= -2, "left base {base}");
                            let shift = ((idx << up_l) >> 1) & 0x1f;
                            round2(e.l(base) * (32 - shift) + e.l(base + 1) * shift, 5)
                        }
                    } else if p_angle > 180 {
                        let idx = (j + 1) * dy;
                        let base = (idx >> (6 - up_l)) + (i << up_l);
                        let shift = ((idx << up_l) >> 1) & 0x1f;
                        round2(e.l(base) * (32 - shift) + e.l(base + 1) * shift, 5)
                    } else if p_angle == 90 {
                        e.a(j)
                    } else {
                        e.l(i)
                    };
                    set(i, j, v);
                }
            }
        } else if (9..=11).contains(&mode) {
            for i in 0..h {
                for j in 0..w {
                    let v = if mode == 9 {
                        let wx = sm(log2w)[j as usize] as i32;
                        let wy = sm(log2h)[i as usize] as i32;
                        let s = wy * e.a(j)
                            + (256 - wy) * e.l(h - 1)
                            + wx * e.l(i)
                            + (256 - wx) * e.a(w - 1);
                        round2(s, 9)
                    } else if mode == 10 {
                        let wy = sm(log2h)[i as usize] as i32;
                        round2(wy * e.a(j) + (256 - wy) * e.l(h - 1), 8)
                    } else {
                        let wx = sm(log2w)[j as usize] as i32;
                        round2(wx * e.l(i) + (256 - wx) * e.a(w - 1), 8)
                    };
                    set(i, j, v);
                }
            }
        } else if mode == 0 {
            let v = if have_left && have_above {
                let mut sum = 0;
                for k in 0..h {
                    sum += e.l(k);
                }
                for k in 0..w {
                    sum += e.a(k);
                }
                sum += (w + h) >> 1;
                sum / (w + h)
            } else if have_left {
                let mut sum = 0;
                for k in 0..h {
                    sum += e.l(k);
                }
                ((sum + (h >> 1)) >> log2h).clamp(0, (1 << bd) - 1)
            } else if have_above {
                let mut sum = 0;
                for k in 0..w {
                    sum += e.a(k);
                }
                ((sum + (w >> 1)) >> log2w).clamp(0, (1 << bd) - 1)
            } else {
                1 << (bd - 1)
            };
            for i in 0..h {
                for j in 0..w {
                    set(i, j, v);
                }
            }
        } else {
            for i in 0..h {
                for j in 0..w {
                    let base = e.a(j) + e.l(i) - e.a(-1);
                    let p_left = (base - e.l(i)).abs();
                    let p_top = (base - e.a(j)).abs();
                    let p_top_left = (base - e.a(-1)).abs();
                    let v = if p_left <= p_top && p_left <= p_top_left {
                        e.l(i)
                    } else if p_top <= p_top_left {
                        e.a(j)
                    } else {
                        e.a(-1)
                    };
                    set(i, j, v);
                }
            }
        }
        pred
    }

    fn ref_filter_intra(e: &Edges, w: i32, h: i32, mode: usize, bd: u32) -> Vec<i32> {
        let mut pred = vec![0i32; (w * h) as usize];
        for i2 in 0..h >> 1 {
            for j4 in 0..w >> 2 {
                let mut p = [0i32; 7];
                for i in 0..7 {
                    p[i as usize] = if i < 5 {
                        if i2 == 0 {
                            e.a((j4 << 2) + i - 1)
                        } else if j4 == 0 && i == 0 {
                            e.l((i2 << 1) - 1)
                        } else {
                            pred[(((i2 << 1) - 1) * w + (j4 << 2) + i - 1) as usize]
                        }
                    } else if j4 == 0 {
                        e.l((i2 << 1) + i - 5)
                    } else {
                        pred[(((i2 << 1) + i - 5) * w + (j4 << 2) - 1) as usize]
                    };
                }
                for i1 in 0..2 {
                    for j1 in 0..4 {
                        let mut pr = 0;
                        for i in 0..7 {
                            pr +=
                                INTRA_FILTER_TAPS[mode][((i1 << 2) + j1) as usize][i] as i32 * p[i];
                        }
                        let r = if pr >= 0 {
                            round2(pr, 4)
                        } else {
                            -round2(-pr, 4)
                        };
                        pred[(((i2 << 1) + i1) * w + (j4 << 2) + j1) as usize] =
                            r.clamp(0, (1 << bd) - 1);
                    }
                }
            }
        }
        pred
    }

    fn rand_edges<P: Pixel>(rng: &mut Rng, bd: u32, smooth: bool) -> IntraEdge<P> {
        let mut e = IntraEdge::<P>::new();
        let max = 1u32 << bd;
        // Smooth-ish edges hit Paeth ties and all interpolation phases;
        // random edges hit clipping.
        let mut va = rng.below(max) as i32;
        let mut vl = rng.below(max) as i32;
        for k in 0..EDGE_LEN {
            if smooth {
                va = (va + rng.below(9) as i32 - 4).clamp(0, max as i32 - 1);
                vl = (vl + rng.below(9) as i32 - 4).clamp(0, max as i32 - 1);
                e.above[k] = P::from_i32(va);
                e.left[k] = P::from_i32(vl);
            } else {
                e.above[k] = P::from_i32(rng.below(max) as i32);
                e.left[k] = P::from_i32(rng.below(max) as i32);
            }
        }
        e
    }

    fn to_edges<P: Pixel>(e: &IntraEdge<P>) -> Edges {
        Edges {
            a: e.above.iter().map(|p| p.to_i32()).collect(),
            l: e.left.iter().map(|p| p.to_i32()).collect(),
        }
    }

    fn run_predict<P: Pixel>(bd: u32, seed: u64) {
        let mut rng = Rng(seed);
        let stride = 80;
        let mut cases = 0;
        for &(w, h) in &SIZES {
            for mode in 0u8..=12 {
                let deltas: &[i8] = if (1..=8).contains(&mode) {
                    &[-3, -2, -1, 0, 1, 2, 3]
                } else {
                    &[0]
                };
                for &delta in deltas {
                    for flags in 0..16u32 {
                        let have_left = flags & 1 != 0;
                        let have_above = flags & 2 != 0;
                        let smooth = flags & 4 != 0;
                        let ef = flags & 8 != 0;
                        for rep in 0..3 {
                            let (ew, eh) = if rep == 2 {
                                (
                                    1 + rng.below(w as u32) as usize,
                                    1 + rng.below(h as u32) as usize,
                                )
                            } else {
                                (w, h)
                            };
                            let mut e = rand_edges::<P>(&mut rng, bd, rep == 1);
                            let mut re = to_edges(&e);
                            let want = ref_predict(
                                &mut re,
                                w as i32,
                                h as i32,
                                mode as i32,
                                delta as i32,
                                have_left,
                                have_above,
                                smooth as i32,
                                ef,
                                ew as i32,
                                eh as i32,
                                bd,
                            );
                            let mut dst = vec![P::from_i32(0); stride * h];
                            if ew == w && eh == h && rng.below(2) == 0 {
                                predict(
                                    &mut dst,
                                    stride,
                                    w,
                                    h,
                                    mode,
                                    delta,
                                    &mut e.above,
                                    &mut e.left,
                                    have_left,
                                    have_above,
                                    smooth,
                                    ef,
                                    bd,
                                );
                            } else {
                                predict_clipped(
                                    &mut dst,
                                    stride,
                                    w,
                                    h,
                                    mode,
                                    delta,
                                    &mut e.above,
                                    &mut e.left,
                                    have_left,
                                    have_above,
                                    smooth,
                                    ef,
                                    ew,
                                    eh,
                                    bd,
                                );
                            }
                            for i in 0..h {
                                for j in 0..w {
                                    assert_eq!(
                                        dst[i * stride + j].to_i32(),
                                        want[i * w + j],
                                        "bd {bd} {w}x{h} mode {mode} delta {delta} \
                                         flags {flags} edge {ew}x{eh} at ({i},{j})"
                                    );
                                }
                            }
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert!(cases > 10000);
    }

    #[test]
    fn predict_8bit() {
        run_predict::<u8>(8, 0x1234_5678_9abc_def1);
    }

    #[test]
    fn predict_10bit() {
        run_predict::<u16>(10, 0x2234_5678_9abc_def1);
    }

    #[test]
    fn predict_12bit() {
        run_predict::<u16>(12, 0x3234_5678_9abc_def1);
    }

    fn run_filter_intra<P: Pixel>(bd: u32, seed: u64) {
        let mut rng = Rng(seed);
        let stride = 72;
        for &(w, h) in &SIZES {
            for mode in 0..5u8 {
                for rep in 0..20 {
                    let e = rand_edges::<P>(&mut rng, bd, rep & 1 == 1);
                    let want =
                        ref_filter_intra(&to_edges(&e), w as i32, h as i32, mode as usize, bd);
                    let mut dst = vec![P::from_i32(0); stride * h];
                    filter_intra(&mut dst, stride, w, h, mode, &e.above, &e.left, bd);
                    for i in 0..h {
                        for j in 0..w {
                            assert_eq!(
                                dst[i * stride + j].to_i32(),
                                want[i * w + j],
                                "bd {bd} {w}x{h} mode {mode} at ({i},{j})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn filter_intra_all() {
        run_filter_intra::<u8>(8, 7);
        run_filter_intra::<u16>(10, 8);
        run_filter_intra::<u16>(12, 9);
    }

    fn run_cfl<P: Pixel>(bd: u32, seed: u64) {
        let mut rng = Rng(seed);
        let max = 1u32 << bd;
        let luma_stride = 136;
        for &(w, h) in &SIZES {
            if w > 32 || h > 32 {
                continue;
            }
            for sub in 0..4 {
                let (sx, sy) = (sub & 1 != 0, sub & 2 != 0);
                for rep in 0..20 {
                    let (vw, vh) = if rep % 2 == 0 {
                        (w, h)
                    } else {
                        (
                            1 + rng.below(w as u32) as usize,
                            1 + rng.below(h as u32) as usize,
                        )
                    };
                    let luma: Vec<P> = (0..luma_stride * 2 * h + 2 * w)
                        .map(|_| P::from_i32(rng.below(max) as i32))
                        .collect();
                    let alpha = rng.below(33) as i32 - 16;
                    let dc: Vec<P> = (0..w * h)
                        .map(|_| P::from_i32(rng.below(max) as i32))
                        .collect();
                    // Reference: spec with lumaX/lumaY clamped at the last
                    // valid position.
                    let (sxu, syu) = (sx as usize, sy as usize);
                    let mut l = vec![0i32; w * h];
                    let mut avg = 0i32;
                    for i in 0..h {
                        let ly = (i << syu).min((vh - 1) << syu);
                        for j in 0..w {
                            let lx = (j << sxu).min((vw - 1) << sxu);
                            let mut t = 0;
                            for dy in 0..=syu {
                                for dx in 0..=sxu {
                                    t += luma[(ly + dy) * luma_stride + lx + dx].to_i32();
                                }
                            }
                            let v = t << (3 - sxu - syu);
                            l[i * w + j] = v;
                            avg += v;
                        }
                    }
                    let avg = round2(avg, w.trailing_zeros() + h.trailing_zeros());
                    let mut ac = vec![0i32; w * h];
                    cfl_ac(&mut ac, &luma, luma_stride, w, h, sx, sy, vw, vh);
                    let mut dst = dc.clone();
                    cfl_pred(&mut dst, w, w, h, &ac, alpha, bd);
                    for k in 0..w * h {
                        assert_eq!(ac[k], l[k] - avg);
                        let m = alpha * (l[k] - avg);
                        let s = if m >= 0 { round2(m, 6) } else { -round2(-m, 6) };
                        let want = (dc[k].to_i32() + s).clamp(0, max as i32 - 1);
                        assert_eq!(dst[k].to_i32(), want, "bd {bd} {w}x{h} sub {sub}");
                    }
                }
            }
        }
    }

    #[test]
    fn cfl_all() {
        run_cfl::<u8>(8, 11);
        run_cfl::<u16>(10, 12);
        run_cfl::<u16>(12, 13);
    }
}

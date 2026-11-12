//! The deblocking sample filter as the specification writes it, one line
//! at a time.

use super::{Limits, Size};

/// The wide filter (7.14.6.4) for taps `-n .. n` around the edge: `log2`
/// is 3 or 4, and taps within `n2` of the centre count twice.
fn wide(px: &[i32; 14], out: &mut [i32; 14], n: i32, n2: i32, log2: u32) {
    for i in -n..n {
        let mut t = 0;
        for j in -n..=n {
            let p = (i + j).clamp(-(n + 1), n);
            let tap = if j.abs() <= n2 { 2 } else { 1 };
            t += px[(7 + p) as usize] * tap;
        }
        out[(7 + i) as usize] = (t + (1 << (log2 - 1))) >> log2;
    }
}

/// Filters one line of 8-bit samples (index 7 is `q0`). Returns whether
/// the line passed the filter mask (7.14.6.2).
pub(super) fn line(px: &mut [i32; 14], size: Size, lim: Limits) -> bool {
    let limit = i32::from(lim.limit);
    let blimit = i32::from(lim.blimit);
    let thresh = i32::from(lim.thresh);
    let six = size != Size::Four;
    let eight = matches!(size, Size::Eight | Size::Sixteen);
    let (p1, p0, q0, q1) = (px[5], px[6], px[7], px[8]);
    let hev = (p1 - p0).abs() > thresh || (q1 - q0).abs() > thresh;
    let mut mask = (p1 - p0).abs() > limit
        || (q1 - q0).abs() > limit
        || (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 > blimit;
    if six {
        mask |= (px[4] - p1).abs() > limit || (px[9] - q1).abs() > limit;
    }
    if eight {
        mask |= (px[3] - px[4]).abs() > limit || (px[10] - px[9]).abs() > limit;
    }
    if mask {
        return false;
    }
    let near = |a: i32, b: i32| (a - b).abs() <= 1;
    let mut flat = false;
    if six {
        flat = near(p1, p0) && near(q1, q0) && near(px[4], p0) && near(px[9], q0);
        if eight {
            flat &= near(px[3], p0) && near(px[10], q0);
        }
    }
    let flat2 = size == Size::Sixteen
        && near(px[0], p0)
        && near(px[13], q0)
        && near(px[1], p0)
        && near(px[12], q0)
        && near(px[2], p0)
        && near(px[11], q0);
    let mut out = *px;
    if !flat {
        // The narrow filter (7.14.6.3).
        let c4 = |x: i32| x.clamp(-128, 127);
        let (ps1, ps0, qs0, qs1) = (p1 - 128, p0 - 128, q0 - 128, q1 - 128);
        let f = if hev { c4(ps1 - qs1) } else { 0 };
        let f = c4(f + 3 * (qs0 - ps0));
        let f1 = c4(f + 4) >> 3;
        let f2 = c4(f + 3) >> 3;
        out[7] = c4(qs0 - f1) + 128;
        out[6] = c4(ps0 + f2) + 128;
        if !hev {
            let f = (f1 + 1) >> 1;
            out[8] = c4(qs1 - f) + 128;
            out[5] = c4(ps1 + f) + 128;
        }
    } else if size == Size::Six {
        wide(px, &mut out, 2, 1, 3);
    } else if size == Size::Eight || !flat2 {
        wide(px, &mut out, 3, 0, 3);
    } else {
        wide(px, &mut out, 6, 1, 4);
    }
    *px = out;
    true
}

/// [`super::by_sample`], one line at a time.
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon"),
    allow(dead_code)
)]
pub(super) fn by_sample(taps: &mut [[u8; 16]; 14], size: Size, lim: Limits) -> bool {
    let mut any = false;
    for l in 0..16 {
        let mut px: [i32; 14] = core::array::from_fn(|k| i32::from(taps[k][l]));
        if line(&mut px, size, lim) {
            any = true;
            for (row, &v) in taps.iter_mut().zip(&px) {
                row[l] = v as u8;
            }
        }
    }
    any
}

/// [`super::by_line`], one line at a time.
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon"),
    allow(dead_code)
)]
pub(super) fn by_line(lines: &mut [[u8; 8]; 16], size: Size, lim: Limits) -> bool {
    let mut any = false;
    for l in lines {
        let mut px = [0i32; 14];
        for (k, &v) in l.iter().enumerate() {
            px[3 + k] = i32::from(v);
        }
        if line(&mut px, size, lim) {
            any = true;
            for (k, v) in l.iter_mut().enumerate() {
                *v = px[3 + k] as u8;
            }
        }
    }
    any
}

/// [`super::by_line_16`], one line at a time.
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon"),
    allow(dead_code)
)]
pub(super) fn by_line_16(lines: &mut [[u8; 16]; 16], lim: Limits) -> bool {
    let mut any = false;
    for l in lines {
        let mut px = [0i32; 14];
        for (k, v) in px.iter_mut().enumerate() {
            *v = i32::from(l[1 + k]);
        }
        if line(&mut px, Size::Sixteen, lim) {
            any = true;
            for (k, &v) in px.iter().enumerate() {
                l[1 + k] = v as u8;
            }
        }
    }
    any
}

/// [`super::rows_in_place`] through [`by_sample`].
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon"),
    allow(dead_code)
)]
pub(super) fn rows_in_place(
    buf: &mut [u8],
    stride: usize,
    at: usize,
    size: Size,
    lim: Limits,
) -> bool {
    let (lo, hi) = size.reads();
    let mut taps = [[0u8; 16]; 14];
    for (k, t) in taps.iter_mut().enumerate().take(hi).skip(lo) {
        let o = at + k * stride - 7 * stride;
        t.copy_from_slice(&buf[o..o + 16]);
    }
    if !by_sample(&mut taps, size, lim) {
        return false;
    }
    let (lo, hi) = size.writes();
    for (k, t) in taps.iter().enumerate().take(hi).skip(lo) {
        let o = at + k * stride - 7 * stride;
        buf[o..o + 16].copy_from_slice(t);
    }
    true
}

/// [`super::cols_in_place`] through [`by_line`].
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon"),
    allow(dead_code)
)]
pub(super) fn cols_in_place(
    buf: &mut [u8],
    stride: usize,
    at: usize,
    size: Size,
    lim: Limits,
) -> bool {
    let mut lines = [[0u8; 8]; 16];
    for (l, line) in lines.iter_mut().enumerate() {
        let o = at + l * stride - 4;
        line.copy_from_slice(&buf[o..o + 8]);
    }
    if !by_line(&mut lines, size, lim) {
        return false;
    }
    for (l, line) in lines.iter().enumerate() {
        let o = at + l * stride - 4;
        buf[o..o + 8].copy_from_slice(line);
    }
    true
}

/// [`super::cols_16_in_place`] through [`by_line_16`].
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon"),
    allow(dead_code)
)]
pub(super) fn cols_16_in_place(buf: &mut [u8], stride: usize, at: usize, lim: Limits) -> bool {
    let mut lines = [[0u8; 16]; 16];
    for (l, line) in lines.iter_mut().enumerate() {
        let o = at + l * stride - 8;
        line.copy_from_slice(&buf[o..o + 16]);
    }
    if !by_line_16(&mut lines, lim) {
        return false;
    }
    for (l, line) in lines.iter().enumerate() {
        let o = at + l * stride - 8;
        buf[o..o + 16].copy_from_slice(line);
    }
    true
}

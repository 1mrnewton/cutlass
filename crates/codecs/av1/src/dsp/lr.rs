//! Loop restoration (spec 7.17) for one plane.
//!
//! [`lr_plane`] runs the whole of 7.17 for a plane whose
//! `FrameRestorationType` is not `RESTORE_NONE`: it walks the 64-luma-row
//! stripes (offset up by 8 luma rows), maps each stripe and column range to
//! its restoration unit exactly as 7.17.1 does, fetches samples per 7.17.6
//! (UpscaledCdefFrame inside the stripe, UpscaledCurrFrame for the two rows
//! above and below it, clamped to the plane) and applies the Wiener (7.17.4)
//! or self-guided (7.17.2 / 7.17.3) filter.

use super::{Pixel, round2};
use crate::spec::tables::{
    RESTORE_SGRPROJ, RESTORE_WIENER, SGR_PARAMS, SGRPROJ_MTABLE_BITS, SGRPROJ_PRJ_BITS,
    SGRPROJ_RECIP_BITS, SGRPROJ_RST_BITS, SGRPROJ_SGR_BITS,
};

/// One loop restoration unit's parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LrUnit {
    /// `LrType`: `RESTORE_NONE` (0), `RESTORE_WIENER` (1) or
    /// `RESTORE_SGRPROJ` (2), as in `spec::tables`.
    pub kind: u8,
    /// `LrWiener[plane][row][col]`: `[0]` is the vertical filter's three
    /// coefficients, `[1]` the horizontal filter's.
    pub wiener: [[i32; 3]; 2],
    /// `LrSgrSet`.
    pub sgr_set: u8,
    /// `LrSgrXqd`.
    pub sgr_xqd: [i32; 2],
}

/// Reusable buffers for [`lr_plane`]. Keep one per decoder (or per thread);
/// it grows to the largest plane it has seen.
#[derive(Clone, Debug, Default)]
pub struct LrScratch {
    src: Vec<i32>,
    inter: Vec<i32>,
    a: Vec<i32>,
    b: Vec<i32>,
    flt0: Vec<i32>,
    flt1: Vec<i32>,
    colsum: Vec<i32>,
    colsq: Vec<i32>,
}

impl LrScratch {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Columns of horizontal padding in the stripe source (filters reach 3).
const PAD: usize = 4;
/// Rows of context above and below a stripe in the stripe source.
const CTX: usize = 3;

fn grow(v: &mut Vec<i32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0);
    }
}

/// `count_units_in_frame` from the spec.
#[inline]
pub fn count_units_in_frame(unit_size: usize, frame_size: usize) -> usize {
    ((frame_size + (unit_size >> 1)) / unit_size).max(1)
}

/// Applies 7.17 to one plane.
///
/// * `cdef` is the plane of UpscaledCdefFrame, `deblocked` the plane of
///   UpscaledCurrFrame (deblocked, not CDEF filtered, upscaled when superres
///   is on). Both have stride `stride`.
/// * `plane_w x plane_h` is the plane size in the upscaled domain:
///   `Round2(UpscaledWidth, subX)` by `Round2(FrameHeight, subY)` (the spec's
///   PlaneEndX + 1 and PlaneEndY + 1). Only samples inside it are read.
/// * `unit_size` is `LoopRestorationSize[plane]`; `units` is
///   `units_rows x units_per_row`, row-major, where
///   `units_rows = count_units_in_frame(unit_size, plane_h)` and
///   `units_per_row = count_units_in_frame(unit_size, plane_w)` (the same
///   counts the syntax uses to read the units, with superres too). The last
///   unit in a row / column covers the remainder of the plane.
/// * `dst` (the LrFrame plane, same stride) must already hold a copy of
///   `cdef`; only samples of units whose `kind` is not `RESTORE_NONE` are
///   written.
///
/// The caller skips the call when `FrameRestorationType[plane]` is
/// `RESTORE_NONE`.
pub fn lr_plane<P: Pixel>(
    dst: &mut [P],
    cdef: &[P],
    deblocked: &[P],
    stride: usize,
    plane_w: usize,
    plane_h: usize,
    unit_size: usize,
    units_per_row: usize,
    units_rows: usize,
    units: &[LrUnit],
    sub_x: bool,
    sub_y: bool,
    bd: u32,
    scratch: &mut LrScratch,
) {
    let _ = sub_x;
    if plane_w == 0 || plane_h == 0 || unit_size == 0 || units_per_row == 0 || units_rows == 0 {
        return;
    }
    let sy = sub_y as u32;
    let off = 8isize >> sy;
    let stripe_h = 64isize >> sy;
    let end_y = plane_h as isize - 1;
    let sw = plane_w + 2 * PAD;
    let max_rows = stripe_h as usize + 2 * CTX;
    grow(&mut scratch.src, sw * max_rows);

    let mut stripe = 0isize;
    loop {
        let start = (stripe * 64 - 8) >> sy;
        if start > end_y {
            break;
        }
        stripe += 1;
        let stripe_end = start + stripe_h - 1;
        let y0 = start.max(0);
        let y1 = stripe_end.min(end_y);
        let unit_row = (((y0 + off) as usize) / unit_size).min(units_rows - 1);
        let row_units = &units[unit_row * units_per_row..(unit_row + 1) * units_per_row];
        if row_units
            .iter()
            .all(|u| u.kind as i32 != RESTORE_WIENER && u.kind as i32 != RESTORE_SGRPROJ)
        {
            continue;
        }
        let h = (y1 - y0 + 1) as usize;
        // Stripe source rows y0 - 3 ..= y1 + 3 per the get source sample
        // process, padded by edge replication horizontally.
        for r in 0..h + 2 * CTX {
            let mut y = (y0 - CTX as isize + r as isize).clamp(0, end_y);
            let plane = if y < start {
                y = y.max(start - 2);
                deblocked
            } else if y > stripe_end {
                y = y.min(stripe_end + 2);
                deblocked
            } else {
                cdef
            };
            let row = &plane[y as usize * stride..y as usize * stride + plane_w];
            let out = &mut scratch.src[r * sw..(r + 1) * sw];
            let (first, last) = (row[0].to_i32(), row[plane_w - 1].to_i32());
            out[..PAD].fill(first);
            for (o, s) in out[PAD..PAD + plane_w].iter_mut().zip(row) {
                *o = s.to_i32();
            }
            out[PAD + plane_w..].fill(last);
        }
        for (uc, unit) in row_units.iter().enumerate() {
            let x0 = uc * unit_size;
            if x0 >= plane_w {
                break;
            }
            let x1 = if uc + 1 == units_per_row {
                plane_w
            } else {
                (x0 + unit_size).min(plane_w)
            };
            let rect = Rect {
                x0,
                w: x1 - x0,
                y0: y0 as usize,
                h,
            };
            match unit.kind as i32 {
                RESTORE_WIENER => wiener(dst, stride, &mut *scratch, sw, rect, unit, bd),
                RESTORE_SGRPROJ => sgr(dst, cdef, stride, &mut *scratch, sw, rect, unit, bd),
                _ => {}
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Rect {
    x0: usize,
    w: usize,
    y0: usize,
    h: usize,
}

/// Wiener coefficient process (7.17.5).
fn wiener_taps(c: [i32; 3]) -> [i32; 7] {
    [
        c[0],
        c[1],
        c[2],
        128 - 2 * (c[0] + c[1] + c[2]),
        c[2],
        c[1],
        c[0],
    ]
}

/// Wiener filter process (7.17.4) on one rectangle of a stripe.
fn wiener<P: Pixel>(
    dst: &mut [P],
    stride: usize,
    s: &mut LrScratch,
    sw: usize,
    rc: Rect,
    unit: &LrUnit,
    bd: u32,
) {
    let (round0, round1) = if bd == 12 { (5u32, 9u32) } else { (3, 11) };
    let vf = wiener_taps(unit.wiener[0]);
    let hf = wiener_taps(unit.wiener[1]);
    let offset = 1i32 << (bd + 7 - round0 - 1);
    let limit = (1i32 << (bd + 1 + 7 - round0)) - 1;
    let (w, h) = (rc.w, rc.h);
    grow(&mut s.inter, w * (h + 6));
    let rnd0 = 1i32 << (round0 - 1);
    for r in 0..h + 6 {
        let base = r * sw + PAD + rc.x0 - 3;
        let row = &s.src[base..base + w + 6];
        let out = &mut s.inter[r * w..(r + 1) * w];
        for c in 0..w {
            let mut acc = 0;
            for t in 0..7 {
                acc += hf[t] * row[c + t];
            }
            out[c] = ((acc + rnd0) >> round0).clamp(-offset, limit - offset);
        }
    }
    let rnd1 = 1i32 << (round1 - 1);
    let maxv = (1i32 << bd) - 1;
    for r in 0..h {
        let rows: [&[i32]; 7] = core::array::from_fn(|t| &s.inter[(r + t) * w..(r + t + 1) * w]);
        let o = (rc.y0 + r) * stride + rc.x0;
        let out = &mut dst[o..o + w];
        for c in 0..w {
            let mut acc = 0;
            for t in 0..7 {
                acc += vf[t] * rows[t][c];
            }
            out[c] = P::from_i32(((acc + rnd1) >> round1).clamp(0, maxv));
        }
    }
}

/// `((z << SGRPROJ_SGR_BITS) + z / 2) / (z + 1)` with the spec's end cases,
/// indexed by `Min(z, 255)`.
const A2_TABLE: [i32; 256] = {
    let mut t = [0i32; 256];
    let mut z = 0;
    while z < 256 {
        t[z] = if z >= 255 {
            256
        } else if z == 0 {
            1
        } else {
            (((z << SGRPROJ_SGR_BITS) + z / 2) / (z + 1)) as i32
        };
        z += 1;
    }
    t
};

/// Box filter process (7.17.3) for one pass on one rectangle; writes
/// `w x h` values to `flt`.
fn box_filter<P: Pixel>(
    cdef: &[P],
    stride: usize,
    src: &[i32],
    sw: usize,
    a_buf: &mut Vec<i32>,
    b_buf: &mut Vec<i32>,
    colsum: &mut Vec<i32>,
    colsq: &mut Vec<i32>,
    flt: &mut Vec<i32>,
    rc: Rect,
    r: usize,
    eps: i32,
    pass: usize,
    bd: u32,
) {
    let (w, h) = (rc.w, rc.h);
    let gw = w + 2;
    grow(a_buf, gw * (h + 2));
    grow(b_buf, gw * (h + 2));
    grow(colsum, gw + 2 * r);
    grow(colsq, gw + 2 * r);
    grow(flt, w * h);
    let n = ((2 * r + 1) * (2 * r + 1)) as i32;
    let n2e = n * n * eps;
    let s = ((1 << SGRPROJ_MTABLE_BITS) + n2e / 2) / n2e;
    let one_over_n = (((1 << SGRPROJ_RECIP_BITS) + n / 2) / n) as u32;
    let sh_a = 2 * (bd - 8);
    let sh_b = bd - 8;
    let ncol = gw + 2 * r;
    // A / B rows i = -1 ..= h (grid row i + 1).
    for gi in 0..h + 2 {
        // Pass 0 only uses odd rows i (gi even).
        if pass == 0 && gi & 1 == 1 {
            continue;
        }
        // Source rows i - r ..= i + r: stripe-source row index i + dy + 3.
        let top = gi + CTX - 1 - r;
        let col0 = PAD + rc.x0 - 1 - r;
        {
            let first = &src[top * sw + col0..top * sw + col0 + ncol];
            for k in 0..ncol {
                colsum[k] = first[k];
                colsq[k] = first[k] * first[k];
            }
        }
        for dy in 1..=2 * r {
            let row = &src[(top + dy) * sw + col0..(top + dy) * sw + col0 + ncol];
            for k in 0..ncol {
                colsum[k] += row[k];
                colsq[k] += row[k] * row[k];
            }
        }
        let arow = &mut a_buf[gi * gw..(gi + 1) * gw];
        let brow = &mut b_buf[gi * gw..(gi + 1) * gw];
        for j in 0..gw {
            let mut a = 0;
            let mut b = 0;
            for dx in 0..=2 * r {
                a += colsq[j + dx];
                b += colsum[j + dx];
            }
            let a = round2(a, sh_a);
            let d = round2(b, sh_b);
            let p = (a * n - d * d).max(0);
            let z = ((p as u64 * s as u64 + (1 << (SGRPROJ_MTABLE_BITS - 1)))
                >> SGRPROJ_MTABLE_BITS)
                .min(255) as usize;
            let a2 = A2_TABLE[z];
            let b2 = (256 - a2) as u32 * b as u32 * one_over_n;
            arow[j] = a2;
            brow[j] = ((b2 + (1 << (SGRPROJ_RECIP_BITS - 1))) >> SGRPROJ_RECIP_BITS) as i32;
        }
    }
    let rst = SGRPROJ_RST_BITS as u32;
    let sgr = SGRPROJ_SGR_BITS as u32;
    for i in 0..h {
        let gi = i + 1;
        let crow = &cdef[(rc.y0 + i) * stride + rc.x0..(rc.y0 + i) * stride + rc.x0 + w];
        let out = &mut flt[i * w..(i + 1) * w];
        if pass == 0 {
            if i & 1 == 1 {
                let ar = &a_buf[gi * gw..(gi + 1) * gw];
                let br = &b_buf[gi * gw..(gi + 1) * gw];
                let sh = sgr + 4 - rst;
                for j in 0..w {
                    let a = 6 * ar[j + 1] + 5 * (ar[j] + ar[j + 2]);
                    let b = 6 * br[j + 1] + 5 * (br[j] + br[j + 2]);
                    out[j] = round2(a * crow[j].to_i32() + b, sh);
                }
            } else {
                let au = &a_buf[(gi - 1) * gw..gi * gw];
                let bu = &b_buf[(gi - 1) * gw..gi * gw];
                let ad = &a_buf[(gi + 1) * gw..(gi + 2) * gw];
                let bd_ = &b_buf[(gi + 1) * gw..(gi + 2) * gw];
                let sh = sgr + 5 - rst;
                for j in 0..w {
                    let a =
                        6 * (au[j + 1] + ad[j + 1]) + 5 * (au[j] + au[j + 2] + ad[j] + ad[j + 2]);
                    let b = 6 * (bu[j + 1] + bd_[j + 1])
                        + 5 * (bu[j] + bu[j + 2] + bd_[j] + bd_[j + 2]);
                    out[j] = round2(a * crow[j].to_i32() + b, sh);
                }
            }
        } else {
            let au = &a_buf[(gi - 1) * gw..gi * gw];
            let am = &a_buf[gi * gw..(gi + 1) * gw];
            let ad = &a_buf[(gi + 1) * gw..(gi + 2) * gw];
            let bu = &b_buf[(gi - 1) * gw..gi * gw];
            let bm = &b_buf[gi * gw..(gi + 1) * gw];
            let bd_ = &b_buf[(gi + 1) * gw..(gi + 2) * gw];
            let sh = sgr + 5 - rst;
            for j in 0..w {
                let a = 4 * (am[j] + am[j + 1] + am[j + 2] + au[j + 1] + ad[j + 1])
                    + 3 * (au[j] + au[j + 2] + ad[j] + ad[j + 2]);
                let b = 4 * (bm[j] + bm[j + 1] + bm[j + 2] + bu[j + 1] + bd_[j + 1])
                    + 3 * (bu[j] + bu[j + 2] + bd_[j] + bd_[j + 2]);
                out[j] = round2(a * crow[j].to_i32() + b, sh);
            }
        }
    }
}

/// Self guided filter process (7.17.2) on one rectangle of a stripe.
fn sgr<P: Pixel>(
    dst: &mut [P],
    cdef: &[P],
    stride: usize,
    s: &mut LrScratch,
    sw: usize,
    rc: Rect,
    unit: &LrUnit,
    bd: u32,
) {
    let params = SGR_PARAMS[(unit.sgr_set & 15) as usize];
    let (r0, r1) = (params[0] as usize, params[2] as usize);
    grow(&mut s.flt0, rc.w * rc.h);
    grow(&mut s.flt1, rc.w * rc.h);
    if r0 != 0 {
        box_filter(
            cdef,
            stride,
            &s.src,
            sw,
            &mut s.a,
            &mut s.b,
            &mut s.colsum,
            &mut s.colsq,
            &mut s.flt0,
            rc,
            r0,
            params[1] as i32,
            0,
            bd,
        );
    }
    if r1 != 0 {
        box_filter(
            cdef,
            stride,
            &s.src,
            sw,
            &mut s.a,
            &mut s.b,
            &mut s.colsum,
            &mut s.colsq,
            &mut s.flt1,
            rc,
            r1,
            params[3] as i32,
            1,
            bd,
        );
    }
    let w0 = unit.sgr_xqd[0];
    let w1 = unit.sgr_xqd[1];
    let w2 = (1 << SGRPROJ_PRJ_BITS) - w0 - w1;
    let sh = (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS) as u32;
    let maxv = (1i32 << bd) - 1;
    let (w, h) = (rc.w, rc.h);
    for i in 0..h {
        let o = (rc.y0 + i) * stride + rc.x0;
        let crow = &cdef[o..o + w];
        let out = &mut dst[o..o + w];
        let f0 = &s.flt0[i * w..];
        let f1 = &s.flt1[i * w..];
        for j in 0..w {
            let u = crow[j].to_i32() << SGRPROJ_RST_BITS;
            let mut v = w1 * u;
            v += w0 * if r0 != 0 { f0[j] } else { u };
            v += w2 * if r1 != 0 { f1[j] } else { u };
            out[j] = P::from_i32(round2(v, sh).clamp(0, maxv));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{LrScratch, LrUnit, count_units_in_frame, lr_plane};
    use crate::dsp::Pixel;
    use crate::spec::tables::{SGR_PARAMS, WIENER_TAPS_MAX, WIENER_TAPS_MIN};

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

    fn r2(x: i64, n: i64) -> i64 {
        if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
    }

    struct Ref<'a> {
        cdef: &'a [i32],
        curr: &'a [i32],
        stride: usize,
        plane_end_x: i64,
        plane_end_y: i64,
        stripe_start: i64,
        stripe_end: i64,
        bd: u32,
    }

    impl Ref<'_> {
        /// 7.17.6.
        fn get(&self, x: i64, y: i64) -> i64 {
            let x = x.min(self.plane_end_x).max(0);
            let y = y.min(self.plane_end_y).max(0);
            if y < self.stripe_start {
                let y = (self.stripe_start - 2).max(y);
                self.curr[y as usize * self.stride + x as usize] as i64
            } else if y > self.stripe_end {
                let y = (self.stripe_end + 2).min(y);
                self.curr[y as usize * self.stride + x as usize] as i64
            } else {
                self.cdef[y as usize * self.stride + x as usize] as i64
            }
        }
        fn cdef_at(&self, x: i64, y: i64) -> i64 {
            self.cdef[y as usize * self.stride + x as usize] as i64
        }
        fn clip1(&self, v: i64) -> i32 {
            v.clamp(0, (1 << self.bd) - 1) as i32
        }

        /// 7.17.3.
        fn box_filter(&self, x: i64, y: i64, w: i64, h: i64, set: usize, pass: usize) -> Vec<i64> {
            let bd = self.bd as i64;
            let r = SGR_PARAMS[set][pass * 2] as i64;
            if r == 0 {
                return Vec::new();
            }
            let eps = SGR_PARAMS[set][pass * 2 + 1] as i64;
            let gw = (w + 2) as usize;
            let mut aa = vec![0i64; gw * (h + 2) as usize];
            let mut bb = vec![0i64; gw * (h + 2) as usize];
            let n = (2 * r + 1) * (2 * r + 1);
            let n2e = n * n * eps;
            let s = ((1 << 20) + n2e / 2) / n2e;
            for i in -1..h + 1 {
                for j in -1..w + 1 {
                    let mut a = 0;
                    let mut b = 0;
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let c = self.get(x + j + dx, y + i + dy);
                            a += c * c;
                            b += c;
                        }
                    }
                    a = r2(a, 2 * (bd - 8));
                    let d = r2(b, bd - 8);
                    let p = (a * n - d * d).max(0);
                    let z = r2(p * s, 20);
                    let a2 = if z >= 255 {
                        256
                    } else if z == 0 {
                        1
                    } else {
                        ((z << 8) + z / 2) / (z + 1)
                    };
                    let one_over_n = ((1 << 12) + n / 2) / n;
                    let b2 = ((1 << 8) - a2) * b * one_over_n;
                    let idx = (i + 1) as usize * gw + (j + 1) as usize;
                    aa[idx] = a2;
                    bb[idx] = r2(b2, 12);
                }
            }
            let mut f = vec![0i64; (w * h) as usize];
            for i in 0..h {
                let shift = if pass == 0 && (i & 1) == 1 { 4 } else { 5 };
                for j in 0..w {
                    let mut a = 0;
                    let mut b = 0;
                    for dy in -1..=1i64 {
                        for dx in -1..=1i64 {
                            let weight = if pass == 0 {
                                if ((i + dy) & 1) != 0 {
                                    if dx == 0 { 6 } else { 5 }
                                } else {
                                    0
                                }
                            } else if dx == 0 || dy == 0 {
                                4
                            } else {
                                3
                            };
                            let idx = (i + dy + 1) as usize * gw + (j + dx + 1) as usize;
                            a += weight * aa[idx];
                            b += weight * bb[idx];
                        }
                    }
                    let v = a * self.cdef_at(x + j, y + i) + b;
                    f[(i * w + j) as usize] = r2(v, 8 + shift - 4);
                }
            }
            f
        }
    }

    /// 7.17 for one plane, transliterated (loop over 4x4 luma blocks).
    fn ref_lr(
        cdef: &[i32],
        curr: &[i32],
        stride: usize,
        frame_h: i64,
        up_w: i64,
        sub_x: i64,
        sub_y: i64,
        unit_size: i64,
        units: &[LrUnit],
        bd: u32,
    ) -> Vec<i32> {
        let mut lr = cdef.to_vec();
        let unit_rows =
            count_units_in_frame(unit_size as usize, r2(frame_h, sub_y) as usize) as i64;
        let unit_cols = count_units_in_frame(unit_size as usize, r2(up_w, sub_x) as usize) as i64;
        let mut yy = 0;
        while yy < frame_h {
            let mut xx = 0;
            while xx < up_w {
                let row = yy >> 2;
                let col = xx >> 2;
                let luma_y = row * 4;
                let stripe_num = (luma_y + 8) / 64;
                let stripe_start = (-8 + stripe_num * 64) >> sub_y;
                let stripe_end = stripe_start + (64 >> sub_y) - 1;
                let unit_row = (unit_rows - 1).min(((row * 4 + 8) >> sub_y) / unit_size);
                let unit_col = (unit_cols - 1).min(((col * 4) >> sub_x) / unit_size);
                let plane_end_x = r2(up_w, sub_x) - 1;
                let plane_end_y = r2(frame_h, sub_y) - 1;
                let x = (col * 4) >> sub_x;
                let y = (row * 4) >> sub_y;
                let w = (4 >> sub_x).min(plane_end_x - x + 1);
                let h = (4 >> sub_y).min(plane_end_y - y + 1);
                let rf = Ref {
                    cdef,
                    curr,
                    stride,
                    plane_end_x,
                    plane_end_y,
                    stripe_start,
                    stripe_end,
                    bd,
                };
                let u = units[(unit_row * unit_cols + unit_col) as usize];
                if u.kind == 1 {
                    let (r0, r1) = if bd == 12 { (5, 9) } else { (3, 11) };
                    let coef = |c: [i32; 3]| {
                        let mut f = [0i64; 7];
                        f[3] = 128;
                        for i in 0..3 {
                            f[i] = c[i] as i64;
                            f[6 - i] = c[i] as i64;
                            f[3] -= 2 * c[i] as i64;
                        }
                        f
                    };
                    let vf = coef(u.wiener[0]);
                    let hf = coef(u.wiener[1]);
                    let bdi = bd as i64;
                    let offset = 1i64 << (bdi + 7 - r0 - 1);
                    let limit = (1i64 << (bdi + 1 + 7 - r0)) - 1;
                    let mut inter = vec![0i64; ((h + 6) * w) as usize];
                    for r in 0..h + 6 {
                        for c in 0..w {
                            let mut s = 0;
                            for t in 0..7 {
                                s += hf[t as usize] * rf.get(x + c + t - 3, y + r - 3);
                            }
                            let v = r2(s, r0);
                            inter[(r * w + c) as usize] = v.clamp(-offset, limit - offset);
                        }
                    }
                    for r in 0..h {
                        for c in 0..w {
                            let mut s = 0;
                            for t in 0..7 {
                                s += vf[t as usize] * inter[((r + t) * w + c) as usize];
                            }
                            let v = r2(s, r1);
                            lr[(y + r) as usize * stride + (x + c) as usize] = rf.clip1(v);
                        }
                    }
                } else if u.kind == 2 {
                    let set = u.sgr_set as usize;
                    let flt0 = rf.box_filter(x, y, w, h, set, 0);
                    let flt1 = rf.box_filter(x, y, w, h, set, 1);
                    let w0 = u.sgr_xqd[0] as i64;
                    let w1 = u.sgr_xqd[1] as i64;
                    let w2 = (1 << 7) - w0 - w1;
                    let rr0 = SGR_PARAMS[set][0];
                    let rr1 = SGR_PARAMS[set][2];
                    for i in 0..h {
                        for j in 0..w {
                            let uu = rf.cdef_at(x + j, y + i) << 4;
                            let mut v = w1 * uu;
                            v += if rr0 != 0 {
                                w0 * flt0[(i * w + j) as usize]
                            } else {
                                w0 * uu
                            };
                            v += if rr1 != 0 {
                                w2 * flt1[(i * w + j) as usize]
                            } else {
                                w2 * uu
                            };
                            let s = r2(v, 4 + 7);
                            lr[(y + i) as usize * stride + (x + j) as usize] = rf.clip1(s);
                        }
                    }
                }
                xx += 4;
            }
            yy += 4;
        }
        lr
    }

    fn run<P: Pixel>(bd: u32, seed: u64, iters: usize) {
        let mut rng = Rng(seed);
        let max = (1 << bd) - 1;
        for iter in 0..iters {
            let up_w = 1 + rng.range(330) as i64;
            let frame_h = 1 + rng.range(200) as i64;
            let (sub_x, sub_y) = match rng.range(4) {
                0 => (0i64, 0i64),
                1 => (1, 1),
                2 => (1, 0),
                _ => (0, 0),
            };
            let chroma = sub_x != 0 || sub_y != 0 || rng.range(2) == 0;
            let unit_size = if sub_y == 1 {
                [32i64, 64, 128][rng.range(3) as usize]
            } else {
                [64i64, 128, 256][rng.range(3) as usize]
            };
            let pw = r2(up_w, sub_x) as usize;
            let ph = r2(frame_h, sub_y) as usize;
            let stride = pw + rng.range(5) as usize;
            let rows = ph;
            let base = rng.range(max as u32 + 1);
            let noise = [2, 16, 256, 1 << 12][rng.range(4) as usize] << (bd - 8) >> 1;
            let mut cdef = vec![0i32; stride * rows];
            let mut curr = vec![0i32; stride * rows];
            for i in 0..stride * rows {
                cdef[i] = (base + rng.range(2 * noise as u32 + 1) - noise).clamp(0, max);
                curr[i] = (cdef[i] + rng.range(9) - 4).clamp(0, max);
            }
            let ucols = count_units_in_frame(unit_size as usize, pw);
            let urows = count_units_in_frame(unit_size as usize, ph);
            let units: Vec<LrUnit> = (0..ucols * urows)
                .map(|_| {
                    let mut wiener = [[0i32; 3]; 2];
                    for p in 0..2 {
                        for i in 0..3 {
                            let lo = WIENER_TAPS_MIN[i] as i32;
                            let hi = WIENER_TAPS_MAX[i] as i32;
                            wiener[p][i] = lo + rng.range((hi - lo + 1) as u32);
                        }
                        if chroma {
                            wiener[p][0] = 0;
                        }
                    }
                    LrUnit {
                        kind: rng.range(3) as u8,
                        wiener,
                        sgr_set: rng.range(16) as u8,
                        sgr_xqd: [rng.range(128) - 96, rng.range(128) - 32],
                    }
                })
                .collect();
            let expect = ref_lr(
                &cdef, &curr, stride, frame_h, up_w, sub_x, sub_y, unit_size, &units, bd,
            );
            let cdef_p: Vec<P> = cdef.iter().map(|&v| P::from_i32(v)).collect();
            let curr_p: Vec<P> = curr.iter().map(|&v| P::from_i32(v)).collect();
            let mut dst = cdef_p.clone();
            let mut scratch = LrScratch::new();
            lr_plane(
                &mut dst,
                &cdef_p,
                &curr_p,
                stride,
                pw,
                ph,
                unit_size as usize,
                ucols,
                urows,
                &units,
                sub_x == 1,
                sub_y == 1,
                bd,
                &mut scratch,
            );
            let got: Vec<i32> = dst.iter().map(|v| v.to_i32()).collect();
            let bad = got.iter().zip(&expect).position(|(a, b)| a != b);
            assert_eq!(
                bad, None,
                "iter {iter} bd {bd} {pw}x{ph} sub {sub_x}{sub_y} unit {unit_size}"
            );
        }
    }

    #[test]
    fn lr_matches_spec_8bit() {
        run::<u8>(8, 0x5555_1234_8765_4321, 60);
    }

    #[test]
    fn lr_matches_spec_10bit() {
        run::<u16>(10, 0x6555_1234_8765_4321, 60);
    }

    #[test]
    fn lr_matches_spec_12bit() {
        run::<u16>(12, 0x7555_1234_8765_4321, 60);
    }
}

//! The VP9 loop filter across one edge for 10- and 12-bit samples
//! (libvpx `vpx_highbd_lpf_*`): 16 lines at once, 4, 8 or 16 samples
//! wide, with the 8-bit thresholds and sample offsets shifted by
//! `bit_depth - 8`.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(test)]
mod tests;

pub use super::lf::{Thresh, Wd};

/// One edge to filter, as [`super::lf::Edge`] but on `u16` samples of
/// `bd` bits (10 or 12).
pub struct Edge<'a> {
    pub buf: &'a mut [u16],
    pub at: usize,
    pub stride: usize,
    pub vertical: bool,
    pub wd: [Option<Wd>; 2],
    pub thr: [Thresh; 2],
    pub bd: u32,
}

/// The widest filter of two halves.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[inline(always)]
fn widest(wd: [Option<Wd>; 2]) -> Option<Wd> {
    let r = |w: Option<Wd>| match w {
        None => 0,
        Some(Wd::W4) => 1,
        Some(Wd::W8) => 2,
        Some(Wd::W16) => 3,
    };
    if r(wd[0]) >= r(wd[1]) { wd[0] } else { wd[1] }
}

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
impl Edge<'_> {
    #[inline(always)]
    fn fits(&self) -> bool {
        let Some(wd) = widest(self.wd) else {
            return false;
        };
        if !matches!(self.bd, 10 | 12) {
            return false;
        }
        let taps = if wd == Wd::W16 { 8 } else { 4 };
        if self.vertical {
            // Each line reads 16 samples from 8 before the edge, or 8
            // from 4 before it.
            self.at >= taps && self.at + 15 * self.stride + taps <= self.buf.len()
        } else {
            self.at >= taps * self.stride
                && self.at + (taps - 1) * self.stride + 16 <= self.buf.len()
        }
    }
}

/// Filters `e` with vector instructions; `false`, having written nothing,
/// when the CPU or the edge is not one the kernels take. The samples must
/// lie in `0..1 << bd`.
#[must_use]
#[inline(always)]
pub fn filter(e: Edge<'_>) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if super::avx2() && e.fits() {
            // SAFETY: `avx2` checked at run time that the CPU has AVX2,
            // the only feature `avx2::filter` enables.
            #[allow(unsafe_code)]
            unsafe {
                avx2::filter(e.buf, e.at, e.stride, e.vertical, e.wd, e.thr, e.bd);
            }
            return true;
        }
    }
    let _ = e;
    false
}

/// The scalar twin of [`filter`], from libvpx's C.
pub fn filter_scalar(e: Edge<'_>) {
    let Edge {
        buf,
        at,
        stride,
        vertical,
        wd: wds,
        thr,
        bd,
    } = e;
    let sh = bd - 8;
    let one = 1i32 << sh;
    let half = 0x80i32 << sh;
    let (lo, hi) = (-(128i32 << sh), (128i32 << sh) - 1);
    let c = |v: i32| v.clamp(lo, hi);
    let (step, line) = if vertical { (1, stride) } else { (stride, 1) };
    for i in 0..16 {
        let t = thr[i / 8];
        let Some(wd) = wds[i / 8] else { continue };
        let (blimit, limit, hev_t) = (
            i32::from(t.blimit) << sh,
            i32::from(t.limit) << sh,
            i32::from(t.hev) << sh,
        );
        let q0i = at + i * line;
        let px = |buf: &[u16], k: isize| -> i32 {
            i32::from(buf[(q0i as isize + k * step as isize) as usize])
        };
        let set = |buf: &mut [u16], k: isize, v: i32| {
            buf[(q0i as isize + k * step as isize) as usize] = v as u16
        };
        let (p3, p2, p1, p0) = (px(buf, -4), px(buf, -3), px(buf, -2), px(buf, -1));
        let (q0, q1, q2, q3) = (px(buf, 0), px(buf, 1), px(buf, 2), px(buf, 3));
        let mask = (p3 - p2).abs() <= limit
            && (p2 - p1).abs() <= limit
            && (p1 - p0).abs() <= limit
            && (q1 - q0).abs() <= limit
            && (q2 - q1).abs() <= limit
            && (q3 - q2).abs() <= limit
            && (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 <= blimit;
        if !mask {
            continue;
        }
        let flat = wd != Wd::W4
            && [p1, q1, p2, q2, p3, q3]
                .iter()
                .zip([p0, q0, p0, q0, p0, q0])
                .all(|(a, b)| (a - b).abs() <= one);
        if flat && wd == Wd::W16 {
            let s: [i32; 16] = core::array::from_fn(|k| px(buf, k as isize - 8));
            let flat2 = (1..5)
                .all(|k| (s[7 - 3 - k] - p0).abs() <= one && (s[8 + 3 + k] - q0).abs() <= one);
            if flat2 {
                let mut out = [0i32; 14];
                for (j, o) in out.iter_mut().enumerate() {
                    // Output j is sample j + 1 of s: a 15-tap window
                    // centred there, the ends repeated, centre doubled.
                    let c = j as isize + 1;
                    let mut sum = 0;
                    for k in -7..=7isize {
                        sum += s[(c + k).clamp(0, 15) as usize];
                    }
                    sum += s[c as usize];
                    *o = (sum + 8) >> 4;
                }
                for (j, v) in out.iter().enumerate() {
                    set(buf, j as isize - 7, *v);
                }
                continue;
            }
        }
        if flat {
            set(buf, -3, (p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3);
            set(buf, -2, (p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3);
            set(buf, -1, (p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3);
            set(buf, 0, (p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3);
            set(buf, 1, (p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3 + 4) >> 3);
            set(buf, 2, (p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3 + 4) >> 3);
            continue;
        }
        let (ps1, ps0, qs0, qs1) = (p1 - half, p0 - half, q0 - half, q1 - half);
        let hev = (p1 - p0).abs() > hev_t || (q1 - q0).abs() > hev_t;
        let mut f = if hev { c(ps1 - qs1) } else { 0 };
        f = c(f + 3 * (qs0 - ps0));
        let f1 = c(f + 4) >> 3;
        let f2 = c(f + 3) >> 3;
        set(buf, 0, c(qs0 - f1) + half);
        set(buf, -1, c(ps0 + f2) + half);
        if !hev {
            let f = (f1 + 1) >> 1;
            set(buf, 1, c(qs1 - f) + half);
            set(buf, -2, c(ps1 + f) + half);
        }
    }
}

//! The VP9 loop filter across one edge for 8-bit samples (libvpx
//! `vpx_lpf_*`): 8 or 16 lines at once, 4, 8 or 16 samples wide.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(test)]
mod tests;

/// The thresholds of one filter level.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Thresh {
    pub blimit: u8,
    pub limit: u8,
    pub hev: u8,
}

/// How wide the filter is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Wd {
    W4,
    W8,
    W16,
}

/// One edge to filter: 16 lines of `buf`, as two halves of 8. Line `i`
/// has its first sample past the edge at `at + i * stride` for a vertical
/// edge (`vertical`, samples along the row), or at `at + i` for a
/// horizontal one (samples down the column, `stride` apart). Lines 0 to 7
/// take `wd[0]` and `thr[0]`, lines 8 to 15 `wd[1]` and `thr[1]`; a half
/// whose filter is `None` is left as it is.
pub struct Edge<'a> {
    pub buf: &'a mut [u8],
    pub at: usize,
    pub stride: usize,
    pub vertical: bool,
    pub wd: [Option<Wd>; 2],
    pub thr: [Thresh; 2],
}

impl Edge<'_> {
    /// The widest filter of the two halves.
    fn widest(&self) -> Option<Wd> {
        let r = |w: Option<Wd>| match w {
            None => 0,
            Some(Wd::W4) => 1,
            Some(Wd::W8) => 2,
            Some(Wd::W16) => 3,
        };
        if r(self.wd[0]) >= r(self.wd[1]) { self.wd[0] } else { self.wd[1] }
    }

    fn fits(&self) -> bool {
        let Some(wd) = self.widest() else {
            return false;
        };
        let taps = if wd == Wd::W16 { 8 } else { 4 };
        if self.vertical {
            // Each line reads 16 samples from 8 before the edge, or 8
            // from 4 before it.
            self.at >= taps && self.at + 15 * self.stride + taps <= self.buf.len()
        } else {
            self.at >= taps * self.stride && self.at + (taps - 1) * self.stride + 16 <= self.buf.len()
        }
    }
}

/// Filters `e` with vector instructions; `false`, having written nothing,
/// when the CPU or the edge is not one the kernels take.
#[must_use]
pub fn filter(e: Edge<'_>) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if super::avx2() && e.fits() {
            // SAFETY: `avx2` checked at run time that the CPU has AVX2,
            // the only feature `avx2::filter` enables.
            #[allow(unsafe_code)]
            unsafe {
                avx2::filter(e);
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
    } = e;
    let (step, line) = if vertical { (1, stride) } else { (stride, 1) };
    for i in 0..16 {
        let t = thr[i / 8];
        let Some(wd) = wds[i / 8] else { continue };
        let q0i = at + i * line;
        let px = |buf: &[u8], k: isize| -> i32 { i32::from(buf[(q0i as isize + k * step as isize) as usize]) };
        let set = |buf: &mut [u8], k: isize, v: i32| buf[(q0i as isize + k * step as isize) as usize] = v as u8;
        let (p3, p2, p1, p0) = (px(buf, -4), px(buf, -3), px(buf, -2), px(buf, -1));
        let (q0, q1, q2, q3) = (px(buf, 0), px(buf, 1), px(buf, 2), px(buf, 3));
        let limit = i32::from(t.limit);
        let mask = (p3 - p2).abs() <= limit
            && (p2 - p1).abs() <= limit
            && (p1 - p0).abs() <= limit
            && (q1 - q0).abs() <= limit
            && (q2 - q1).abs() <= limit
            && (q3 - q2).abs() <= limit
            && (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 <= i32::from(t.blimit);
        if !mask {
            continue;
        }
        let flat = wd != Wd::W4 && [p1, q1, p2, q2, p3, q3].iter().zip([p0, q0, p0, q0, p0, q0]).all(|(a, b)| (a - b).abs() <= 1);
        if flat && wd == Wd::W16 {
            let s: [i32; 16] = core::array::from_fn(|k| px(buf, k as isize - 8));
            let flat2 = (1..5).all(|k| (s[7 - 3 - k] - p0).abs() <= 1 && (s[8 + 3 + k] - q0).abs() <= 1);
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
        let c = |v: i32| v.clamp(-128, 127);
        let (ps1, ps0, qs0, qs1) = (p1 - 128, p0 - 128, q0 - 128, q1 - 128);
        let hev = (p1 - p0).abs() > i32::from(t.hev) || (q1 - q0).abs() > i32::from(t.hev);
        let mut f = if hev { c(ps1 - qs1) } else { 0 };
        f = c(f + 3 * (qs0 - ps0));
        let f1 = c(f + 4) >> 3;
        let f2 = c(f + 3) >> 3;
        set(buf, 0, c(qs0 - f1) + 128);
        set(buf, -1, c(ps0 + f2) + 128);
        if !hev {
            let f = (f1 + 1) >> 1;
            set(buf, 1, c(qs1 - f) + 128);
            set(buf, -2, c(ps1 + f) + 128);
        }
    }
}

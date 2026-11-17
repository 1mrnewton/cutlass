//! Block distortion of the VP9 encoder's searches for 8-bit samples: the
//! sum of absolute differences of one block against one or four others,
//! and the sum of squared differences. Each entry returns `None` when it
//! does not run on this CPU or does not take the block.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(test)]
mod tests;

/// Whether a `w`×`h` block, rows `stride` apart, lies in `s` and the
/// kernels take its size.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[inline(always)]
fn fits(s: &[u8], stride: usize, w: usize, h: usize) -> bool {
    matches!(w, 8 | 16 | 32 | 64)
        && h > 0
        && h <= 64
        && stride >= w
        && (h - 1) * stride + w <= s.len()
}

/// The SAD of the `w`×`h` blocks at the start of `a` and `b` (rows `sa`
/// and `sb` apart).
#[must_use]
#[inline(always)]
pub fn sad(a: &[u8], sa: usize, b: &[u8], sb: usize, w: usize, h: usize) -> Option<u32> {
    #[cfg(target_arch = "x86_64")]
    {
        if super::avx2() && fits(a, sa, w, h) && fits(b, sb, w, h) {
            // SAFETY: `avx2` checked at run time that the CPU has AVX2,
            // the only feature `avx2::sad` enables.
            #[allow(unsafe_code)]
            return Some(unsafe { avx2::sad(a, sa, b, sb, w, h) });
        }
    }
    let _ = (a, sa, b, sb, w, h);
    None
}

/// [`sad`] of one block against four, the source rows read once.
#[must_use]
#[inline(always)]
pub fn sad4(a: &[u8], sa: usize, b: [&[u8]; 4], sb: usize, w: usize, h: usize) -> Option<[u32; 4]> {
    #[cfg(target_arch = "x86_64")]
    {
        if super::avx2() && fits(a, sa, w, h) && b.iter().all(|b| fits(b, sb, w, h)) {
            // SAFETY: `avx2` checked at run time that the CPU has AVX2,
            // the only feature `avx2::sad4` enables.
            #[allow(unsafe_code)]
            return Some(unsafe { avx2::sad4(a, sa, b, sb, w, h) });
        }
    }
    let _ = (a, sa, b, sb, w, h);
    None
}

/// The SADs of the left and right halves (8 samples each) of a block 16
/// wide and `h` rows (rows `sa` apart) against four others (rows `sb`
/// apart), for each of the four.
#[must_use]
#[inline(always)]
pub fn halves4(a: &[u8], sa: usize, b: [&[u8]; 4], sb: usize, h: usize) -> Option<[[u32; 2]; 4]> {
    #[cfg(target_arch = "x86_64")]
    {
        if super::avx2() && fits(a, sa, 16, h) && b.iter().all(|b| fits(b, sb, 16, h)) {
            // SAFETY: `avx2` checked at run time that the CPU has AVX2,
            // the only feature `avx2::halves4` enables.
            #[allow(unsafe_code)]
            return Some(unsafe { avx2::halves4(a, sa, b, sb, h) });
        }
    }
    let _ = (a, sa, b, sb, h);
    None
}

/// The scalar twin of [`halves4`].
#[must_use]
pub fn halves4_scalar(a: &[u8], sa: usize, b: [&[u8]; 4], sb: usize, h: usize) -> [[u32; 2]; 4] {
    b.map(|b| {
        [
            sad_scalar(a, sa, b, sb, 8, h),
            sad_scalar(&a[8..], sa, &b[8..], sb, 8, h),
        ]
    })
}

/// The sum of squared differences of two `w`×`h` blocks.
#[must_use]
#[inline(always)]
pub fn sse(a: &[u8], sa: usize, b: &[u8], sb: usize, w: usize, h: usize) -> Option<u64> {
    #[cfg(target_arch = "x86_64")]
    {
        if super::avx2() && fits(a, sa, w, h) && fits(b, sb, w, h) {
            // SAFETY: `avx2` checked at run time that the CPU has AVX2,
            // the only feature `avx2::sse` enables.
            #[allow(unsafe_code)]
            return Some(unsafe { avx2::sse(a, sa, b, sb, w, h) });
        }
    }
    let _ = (a, sa, b, sb, w, h);
    None
}

/// The SATD of the `n`×`n` blocks (`n` 4, 8, 16, 32 or 64) at the start
/// of `a` and `b` (rows `sa` and `sb` apart): the sum of the absolute
/// values of each 4×4's Walsh-Hadamard transform of the differences, over
/// two (as [`satd_scalar`]).
#[must_use]
#[inline(always)]
pub fn satd(a: &[u8], sa: usize, b: &[u8], sb: usize, n: usize) -> Option<u64> {
    #[cfg(target_arch = "x86_64")]
    {
        let fits = |s: &[u8], st: usize| {
            matches!(n, 4 | 8 | 16 | 32 | 64) && st >= n && (n - 1) * st + n <= s.len()
        };
        if super::avx2() && fits(a, sa) && fits(b, sb) {
            // SAFETY: `avx2` checked at run time that the CPU has AVX2,
            // the only feature `avx2::satd` enables.
            #[allow(unsafe_code)]
            return Some(unsafe { avx2::satd(a, sa, b, sb, n) });
        }
    }
    let _ = (a, sa, b, sb, n);
    None
}

/// The scalar twin of [`satd`], for any `w`×`h` of whole 4×4s.
#[must_use]
pub fn satd_scalar(a: &[u8], sa: usize, b: &[u8], sb: usize, w: usize, h: usize) -> u64 {
    let mut total = 0u64;
    for by in (0..h).step_by(4) {
        for bx in (0..w).step_by(4) {
            let mut d = [0i32; 16];
            for r in 0..4 {
                for c in 0..4 {
                    d[r * 4 + c] =
                        i32::from(a[(by + r) * sa + bx + c]) - i32::from(b[(by + r) * sb + bx + c]);
                }
            }
            for r in 0..4 {
                let (p, q, u, v) = (d[r * 4], d[r * 4 + 1], d[r * 4 + 2], d[r * 4 + 3]);
                let (s0, s1, d0, d1) = (p + q, u + v, p - q, u - v);
                d[r * 4] = s0 + s1;
                d[r * 4 + 1] = s0 - s1;
                d[r * 4 + 2] = d0 + d1;
                d[r * 4 + 3] = d0 - d1;
            }
            for c in 0..4 {
                let (p, q, u, v) = (d[c], d[4 + c], d[8 + c], d[12 + c]);
                let (s0, s1, d0, d1) = (p + q, u + v, p - q, u - v);
                total += u64::from(
                    (s0 + s1).unsigned_abs()
                        + (s0 - s1).unsigned_abs()
                        + (d0 + d1).unsigned_abs()
                        + (d0 - d1).unsigned_abs(),
                );
            }
        }
    }
    total / 2
}

/// The scalar twin of [`sad`].
#[must_use]
pub fn sad_scalar(a: &[u8], sa: usize, b: &[u8], sb: usize, w: usize, h: usize) -> u32 {
    let mut acc = 0;
    for r in 0..h {
        for c in 0..w {
            acc += u32::from(a[r * sa + c].abs_diff(b[r * sb + c]));
        }
    }
    acc
}

/// The scalar twin of [`sse`].
#[must_use]
pub fn sse_scalar(a: &[u8], sa: usize, b: &[u8], sb: usize, w: usize, h: usize) -> u64 {
    let mut acc = 0u64;
    for r in 0..h {
        for c in 0..w {
            let d = u64::from(a[r * sa + c].abs_diff(b[r * sb + c]));
            acc += d * d;
        }
    }
    acc
}

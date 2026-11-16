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

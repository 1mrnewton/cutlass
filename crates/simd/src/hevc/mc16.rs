//! Fractional sample interpolation of HEVC pictures of 9 to 12 bits
//! (clause 8.5.3.3.3 of ITU-T H.265) into the 14-bit intermediates of the
//! weighted sample prediction, as [`super::mc::put`] does for 8 bits.
//!
//! The entry returns `false`, having written nothing, for a block it does
//! not take (its width, its bit depth, or a source too short for the
//! loads): the caller then predicts the block its own way. The test
//! compares it with a scalar version written from the clause.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(test)]
mod tests;

pub use super::mc::Taps;

/// The reference samples of one block, as [`super::mc::Src`] for 8 bits:
/// `src[at]` is the first sample of the window the filters read, `T / 2 -
/// 1` rows above and columns left of the block's first sample.
#[derive(Clone, Copy, Debug)]
pub struct Src<'a> {
    pub src: &'a [u16],
    pub at: usize,
    pub stride: usize,
}

impl Src<'_> {
    /// Where row `j` of the window starts.
    #[inline]
    #[must_use]
    pub fn row(&self, j: usize) -> usize {
        self.at + j * self.stride
    }

    /// Whether every load of a `w`×`h` block stays inside `src`: the
    /// kernel reads `w + 24` samples of each of `h + taps - 1` rows.
    #[inline]
    fn holds(&self, w: usize, h: usize, taps: usize) -> bool {
        (h + taps - 2)
            .checked_mul(self.stride)
            .and_then(|r| r.checked_add(self.at))
            .and_then(|r| r.checked_add(w + 24))
            .is_some_and(|end| end <= self.src.len())
    }
}

/// Scratch rows the kernel keeps between blocks, made on first use.
#[derive(Default)]
pub struct Tmp {
    #[cfg(target_arch = "x86_64")]
    rows: Vec<avx2::Strips>,
}

/// The intermediates of a `w`×`h` block of samples of `bit_depth` bits
/// (9 to 12) into `dst` (rows `dst_stride` apart): `x << (14 - bit_depth)`
/// with no fraction, the sums of the taps shifted right by `bit_depth - 8`
/// with one, and with both the vertical sums of those shifted right by 6.
#[inline]
pub fn put(
    dst: &mut [i16],
    dst_stride: usize,
    src: &Src<'_>,
    (w, h): (usize, usize),
    taps: &Taps,
    bit_depth: u32,
    tmp: &mut Tmp,
) -> bool {
    let fits = matches!(w, 4 | 8 | 12 | 16 | 24 | 32 | 48 | 64)
        && (1..=64).contains(&h)
        && (9..=12).contains(&bit_depth)
        && matches!(taps.len, 4 | 8)
        && dst_stride >= w
        && (h - 1) * dst_stride + w <= dst.len()
        && src.holds(w, h, taps.len);
    if !fits || !super::mc::accelerated() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `accelerated` checked at run time that the CPU has AVX2.
        #[allow(unsafe_code)]
        unsafe {
            avx2::put(dst, dst_stride, src, (w, h), taps, bit_depth, &mut tmp.rows);
        }
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = tmp;
        false
    }
}

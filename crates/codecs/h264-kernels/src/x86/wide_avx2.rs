//! Sixteen-wide luma prediction for the AVX2 tier: a 256-bit register
//! holds a row's sixteen 16-bit sums, so each step of the six-tap filter
//! is one instruction for the whole row.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_avg_epu8, _mm_packus_epi16, _mm256_add_epi16, _mm256_adds_epi16,
    _mm256_castsi256_si128, _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_set1_epi16,
    _mm256_slli_epi16, _mm256_srai_epi16, _mm256_sub_epi16,
};

use super::kernels::{ld16, st16};

feat! {
    /// Sixteen samples from `p` as 16-bit lanes.
    ///
    /// # Safety
    ///
    /// Sixteen bytes from `p` must be readable.
    #[inline]
    unsafe fn wide16(p: *const u8) -> __m256i {
        // SAFETY: the caller's promise.
        _mm256_cvtepu8_epi16(unsafe { ld16(p) })
    }

    /// The six-tap filter's sum in each lane: see the 128-bit `six`.
    #[inline]
    fn six(t: [__m256i; 6]) -> __m256i {
        let outer = _mm256_add_epi16(t[0], t[5]);
        let near = _mm256_add_epi16(t[1], t[4]);
        let mid = _mm256_add_epi16(t[2], t[3]);
        let inner = _mm256_sub_epi16(_mm256_slli_epi16::<2>(mid), near);
        _mm256_add_epi16(
            _mm256_add_epi16(_mm256_slli_epi16::<2>(inner), inner),
            outer,
        )
    }

    /// Sixteen 16-bit lanes as bytes, with saturation.
    #[inline]
    fn narrow(lanes: __m256i) -> __m128i {
        _mm_packus_epi16(
            _mm256_castsi256_si128(lanes),
            _mm256_extracti128_si256::<1>(lanes),
        )
    }

    /// A row of sixteen full samples.
    ///
    /// # Safety
    ///
    /// Sixteen bytes from `p` must be readable.
    #[inline]
    unsafe fn full16(p: *const u8) -> __m128i {
        // SAFETY: the caller's promise.
        unsafe { ld16(p) }
    }

    /// The six-tap sums along the row at `p` for sixteen samples.
    ///
    /// # Safety
    ///
    /// Twenty-one bytes from two before `p` must be readable.
    #[inline]
    unsafe fn rawh16(p: *const u8) -> __m256i {
        // SAFETY: the last load ends 21 bytes after `p - 2`.
        unsafe {
            let p = p.sub(2);
            six([
                wide16(p),
                wide16(p.add(1)),
                wide16(p.add(2)),
                wide16(p.add(3)),
                wide16(p.add(4)),
                wide16(p.add(5)),
            ])
        }
    }

    /// The six-tap sums down the columns at `p` for sixteen samples.
    ///
    /// # Safety
    ///
    /// Sixteen bytes must be readable from `p` in each of the two rows
    /// before it, its own and the three after, `s` apart.
    #[inline]
    unsafe fn rawv16(p: *const u8, s: usize) -> __m256i {
        // SAFETY: the six rows the caller promises.
        unsafe {
            let p = p.sub(2 * s);
            six([
                wide16(p),
                wide16(p.add(s)),
                wide16(p.add(2 * s)),
                wide16(p.add(3 * s)),
                wide16(p.add(4 * s)),
                wide16(p.add(5 * s)),
            ])
        }
    }

    /// Sixteen half samples from their sums: (sum + 16) >> 5, clipped.
    #[inline]
    fn round16(sums: __m256i) -> __m128i {
        narrow(_mm256_srai_epi16::<5>(_mm256_add_epi16(
            sums,
            _mm256_set1_epi16(16),
        )))
    }

    /// Sixteen centre samples from six rows of sums: see the 128-bit
    /// `centre_lanes` for why 16 bits and one saturating sum are enough.
    #[inline]
    fn centre16v(s: &[__m256i; 6]) -> __m128i {
        let a = _mm256_add_epi16(s[0], s[5]);
        let b = _mm256_add_epi16(s[1], s[4]);
        let c = _mm256_add_epi16(s[2], s[3]);
        let t = _mm256_srai_epi16::<2>(_mm256_sub_epi16(a, b));
        let t = _mm256_srai_epi16::<2>(_mm256_adds_epi16(_mm256_sub_epi16(t, b), c));
        let t = _mm256_add_epi16(_mm256_add_epi16(t, c), _mm256_set1_epi16(32));
        narrow(_mm256_srai_epi16::<6>(t))
    }

    /// The sixteen bytes of `row` to `dst`, or their rounded average with
    /// what is there when `AVG`.
    ///
    /// # Safety
    ///
    /// Sixteen bytes from `dst` must be readable and writable.
    #[inline]
    unsafe fn put16<const AVG: bool>(dst: *mut u8, row: __m128i) {
        // SAFETY: the caller's promise.
        unsafe {
            let row = if AVG {
                _mm_avg_epu8(row, ld16(dst))
            } else {
                row
            };
            st16(dst, row);
        }
    }
}

x86_luma!(
    luma16, __m256i, full16, rawh16, rawv16, round16, centre16v, put16
);

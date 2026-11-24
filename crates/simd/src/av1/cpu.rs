//! Which of the AV1 kernels this CPU can run, decided at run time where
//! the build does not already say.
//!
//! On x86-64 the kernels need AVX2, which the baseline build does not
//! have: [`avx2`] asks the CPU once (through [`crate::wide::has_avx2`]).
//! On aarch64 the motion compensation kernels need the dot product
//! extension, which only Apple targets enable at build time: [`dotprod`]
//! is the build's answer there and the CPU's elsewhere.
//!
//! [`force_scalar`] turns the run-time choices off, so that tests and
//! measurements can run the paths a CPU without the features takes.

use std::sync::atomic::{AtomicBool, Ordering};

/// Set by [`force_scalar`].
static SCALAR: AtomicBool = AtomicBool::new(false);

/// With `on`, every run-time check of this module answers no from now on:
/// the decoder runs as it does on a CPU without AVX2 (x86-64) or without
/// the dot product extension (aarch64 targets that do not enable it at
/// build time). For tests and measurements; a kernel already running
/// finishes its block.
pub fn force_scalar(on: bool) {
    SCALAR.store(on, Ordering::Relaxed);
}

/// Whether the AVX2 kernels run: an x86-64 CPU with AVX2 (and BMI1, BMI2,
/// LZCNT and POPCNT, as every CPU with AVX2 has), and no
/// [`force_scalar`].
#[must_use]
#[inline]
pub fn avx2() -> bool {
    cfg!(target_arch = "x86_64") && !SCALAR.load(Ordering::Relaxed) && crate::wide::has_avx2()
}

/// Whether the NEON kernels that use the dot product instructions run:
/// always where the build enables the extension, and elsewhere on aarch64
/// when the CPU reports it and there is no [`force_scalar`].
#[must_use]
#[inline]
pub fn dotprod() -> bool {
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "dotprod"
    ))]
    {
        true
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        not(target_feature = "dotprod")
    ))]
    {
        !SCALAR.load(Ordering::Relaxed) && std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        false
    }
}

/// Whether the kernels that only need the build's own vector
/// instructions (NEON on aarch64) or AVX2 run.
#[must_use]
#[inline]
pub fn vector() -> bool {
    cfg!(all(target_arch = "aarch64", target_feature = "neon")) || avx2()
}

/// Loads and stores for the AVX2 kernels, without a pointer: a vector is
/// read from and written to a fixed-size array, which compiles to one
/// unaligned move.
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86 {
    use core::arch::x86_64::{
        __m128i, __m256i, _mm_cvtsi128_si64, _mm_extract_epi64, _mm_set_epi64x,
        _mm256_extract_epi64, _mm256_set_m128i,
    };

    /// Sixteen bytes as one vector.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn ld128(a: &[u8; 16]) -> __m128i {
        let v = u128::from_le_bytes(*a);
        _mm_set_epi64x((v >> 64) as i64, v as i64)
    }

    /// Eight bytes in the low half of a vector, zeros above.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn ld64(a: &[u8; 8]) -> __m128i {
        _mm_set_epi64x(0, i64::from_le_bytes(*a))
    }

    /// Thirty-two bytes as one vector.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn ld256(a: &[u8; 32]) -> __m256i {
        let (lo, hi) = a.split_at(16);
        let mut l = [0u8; 16];
        let mut h = [0u8; 16];
        l.copy_from_slice(lo);
        h.copy_from_slice(hi);
        _mm256_set_m128i(ld128(&h), ld128(&l))
    }

    /// The low eight bytes of a vector.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn st64(d: &mut [u8; 8], v: __m128i) {
        *d = _mm_cvtsi128_si64(v).to_le_bytes();
    }

    /// The sixteen bytes of a vector.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn st128(d: &mut [u8; 16], v: __m128i) {
        let q = [_mm_cvtsi128_si64(v), _mm_extract_epi64::<1>(v)];
        for (d, q) in d.as_chunks_mut::<8>().0.iter_mut().zip(q) {
            *d = q.to_le_bytes();
        }
    }

    /// The thirty-two bytes of a vector.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn st256(d: &mut [u8; 32], v: __m256i) {
        let q = [
            _mm256_extract_epi64::<0>(v),
            _mm256_extract_epi64::<1>(v),
            _mm256_extract_epi64::<2>(v),
            _mm256_extract_epi64::<3>(v),
        ];
        for (d, q) in d.as_chunks_mut::<8>().0.iter_mut().zip(q) {
            *d = q.to_le_bytes();
        }
    }

    /// The four 16-bit values of a 64-bit lane.
    #[inline]
    fn words(q: i64) -> [i16; 4] {
        let q = q as u64;
        [
            q as i16,
            (q >> 16) as i16,
            (q >> 32) as i16,
            (q >> 48) as i16,
        ]
    }

    /// The eight 16-bit values of a vector.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn st128_i16(d: &mut [i16; 8], v: __m128i) {
        let q = [_mm_cvtsi128_si64(v), _mm_extract_epi64::<1>(v)];
        for (d, q) in d.as_chunks_mut::<4>().0.iter_mut().zip(q) {
            *d = words(q);
        }
    }

    /// The sixteen 16-bit values of a vector.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn st256_i16(d: &mut [i16; 16], v: __m256i) {
        let q = [
            _mm256_extract_epi64::<0>(v),
            _mm256_extract_epi64::<1>(v),
            _mm256_extract_epi64::<2>(v),
            _mm256_extract_epi64::<3>(v),
        ];
        for (d, q) in d.as_chunks_mut::<4>().0.iter_mut().zip(q) {
            *d = words(q);
        }
    }
}

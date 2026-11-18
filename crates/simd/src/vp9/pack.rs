//! `packuswb` as one instruction, for the 8-bit kernels.
//!
//! The intrinsics are written over generic vector operations, a clamp to
//! 0..=255 and a truncation, and LLVM keeps the clamp in front of the
//! `packuswb` it emits for them: two `pminsw` and two `pmaxsw` (or two
//! `pminuw` when it knows the lanes are not negative) more for every 32
//! bytes packed. These take the same arguments as `_mm256_packus_epi16`
//! and `_mm_packus_epi16` and give the same result.

use core::arch::x86_64::{__m128i, __m256i};

/// `_mm256_packus_epi16(a, b)`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
pub(crate) fn packus256(a: __m256i, b: __m256i) -> __m256i {
    let r: __m256i;
    // SAFETY: one AVX2 instruction on registers, which this function's
    // target feature enables; it reads and writes no memory.
    unsafe {
        core::arch::asm!(
            "vpackuswb {r}, {a}, {b}",
            r = lateout(ymm_reg) r,
            a = in(ymm_reg) a,
            b = in(ymm_reg) b,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    r
}

/// `_mm_packus_epi16(a, b)`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
pub(crate) fn packus128(a: __m128i, b: __m128i) -> __m128i {
    let r: __m128i;
    // SAFETY: as for `packus256`.
    unsafe {
        core::arch::asm!(
            "vpackuswb {r}, {a}, {b}",
            r = lateout(xmm_reg) r,
            a = in(xmm_reg) a,
            b = in(xmm_reg) b,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    r
}

#[cfg(test)]
mod tests {
    use core::arch::x86_64::{
        __m128i, __m256i, _mm_packus_epi16, _mm256_packus_epi16, _mm256_setr_epi16,
    };

    #[test]
    fn packs_as_the_intrinsics_do() {
        if !super::super::avx2() {
            return;
        }
        let mut x = 0x9e37_79b9_u32;
        for _ in 0..1000 {
            let mut lanes = [0i16; 32];
            for v in &mut lanes {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                // Mostly near the byte range, sometimes anywhere.
                *v = if x & 7 == 0 {
                    x as i16
                } else {
                    ((x >> 8) % 600) as i16 - 150
                };
            }
            let l = |o: usize| -> __m256i {
                let v = &lanes[o..o + 16];
                // SAFETY: AVX2 checked above.
                #[allow(unsafe_code)]
                unsafe {
                    _mm256_setr_epi16(
                        v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11],
                        v[12], v[13], v[14], v[15],
                    )
                }
            };
            let (a, b) = (l(0), l(16));
            // SAFETY: AVX2 checked above.
            #[allow(unsafe_code)]
            let (want, got, want1, got1) = unsafe {
                let a1: __m128i = core::mem::transmute::<__m256i, [__m128i; 2]>(a)[0];
                let b1: __m128i = core::mem::transmute::<__m256i, [__m128i; 2]>(b)[1];
                (
                    core::mem::transmute::<__m256i, [u8; 32]>(_mm256_packus_epi16(a, b)),
                    core::mem::transmute::<__m256i, [u8; 32]>(super::packus256(a, b)),
                    core::mem::transmute::<__m128i, [u8; 16]>(_mm_packus_epi16(a1, b1)),
                    core::mem::transmute::<__m128i, [u8; 16]>(super::packus128(a1, b1)),
                )
            };
            assert_eq!(want, got);
            assert_eq!(want1, got1);
        }
    }
}

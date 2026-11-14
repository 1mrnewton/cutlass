//! The compound mask blend of 8-bit predictions (AV1 specification
//! 7.11.3.14, `interintra == 0`): eight samples at a time, and the
//! weights of a subsampled plane.

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
mod scalar;
#[cfg(test)]
mod tests;

/// Blends eight samples of two compound intermediates of 8-bit
/// predictions with their weights `m` (0 to 64):
/// `Clip1(Round2(m a + (64 - m) b, 10))`, worked out as
/// `Clip1(Round2(b + ((m (a - b)) >> 6), 4))`, the same value.
///
/// The intermediates of 8-bit predictions lie within -5131 and 9212, so
/// `a - b` fits 16 bits. Other values wrap where 16 bits do not hold
/// them: `a - b`, `(m (a - b)) >> 6` and the sum with `b`.
#[inline]
#[must_use]
pub fn mask_blend8(a: &[i16; 8], b: &[i16; 8], m: &[u8; 8]) -> [u8; 8] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: the `cfg` above proves NEON is enabled for this build,
        // and the kernel names AES only where the build has it too.
        #[allow(unsafe_code)]
        unsafe {
            neon::mask_blend8(a, b, m)
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    scalar::mask_blend8(a, b, m)
}

/// The weights of eight samples of a plane subsampled across only, from
/// sixteen of a row of the luma mask: `Round2(m[2 i] + m[2 i + 1], 1)`.
#[inline]
#[must_use]
pub fn weights_pairs8(m: &[u8; 16]) -> [u8; 8] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: as in `mask_blend8`.
        #[allow(unsafe_code)]
        unsafe {
            neon::weights_pairs8(m)
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    scalar::weights_pairs8(m)
}

/// The weights of eight samples of a plane subsampled down, from sixteen
/// of two rows of the luma mask: the rounded mean of each 2x2 quad,
/// `Round2(a[2 i] + a[2 i + 1] + b[2 i] + b[2 i + 1], 2)`.
#[inline]
#[must_use]
pub fn weights_quads8(a: &[u8; 16], b: &[u8; 16]) -> [u8; 8] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: as in `mask_blend8`.
        #[allow(unsafe_code)]
        unsafe {
            neon::weights_quads8(a, b)
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    scalar::weights_quads8(a, b)
}

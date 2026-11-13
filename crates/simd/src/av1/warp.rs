//! The block warp process (AV1 specification 7.11.3.5) for one 8x8 block
//! of 8-bit samples: the horizontal filter over fifteen rows, then the
//! vertical one, each output sample with a filter of its own.

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
mod scalar;
#[cfg(test)]
mod tests;

/// The filters of the warp, `Warped_Filters` of the specification, and
/// where an 8x8 block reads them. The horizontal filter of row `r` (0 to
/// 14) and column `c` (0 to 7) is entry
/// `Round2(h[0] + h[1] (c - 4) + h[2] (r - 7), 10) + 64` of `table`, the
/// vertical one of output row `r` (0 to 7) and column `c` is
/// `Round2(v[0] + v[1] (c - 4) + v[2] (r - 4), 10) + 64`: `h` is `sx4`,
/// `alpha` and `beta`, `v` is `sy4`, `gamma` and `delta`.
///
/// Every filter of `table` sums to 128 and the absolute values of its
/// taps to at most 255, as those of the specification do (222 at most):
/// the kernel sums a row in 16 bits.
#[derive(Clone, Copy, Debug)]
pub struct Filters<'a> {
    pub table: &'a [[i8; 8]; 193],
    pub h: [i32; 3],
    pub v: [i32; 3],
}

impl Filters<'_> {
    /// The filter at `Round2(at, 10) + 64`. Panics outside the table,
    /// which the specification's parameters never reach.
    #[inline]
    fn at(&self, at: i32) -> &[i8; 8] {
        &self.table[(((at + 512) >> 10) + 64) as usize]
    }
}

/// The block's samples for a frame that is not compound:
/// `Clip1(Round2(sum, 11))` of the vertical sums. `rows` are the fifteen
/// rows of the reference the filters read, from seven above the block's
/// centre row to seven below, each from seven columns left of its centre
/// column on (the sixteenth sample is not read).
#[inline]
#[must_use]
pub fn put8x8(rows: [&[u8; 16]; 15], f: &Filters<'_>) -> [[u8; 8]; 8] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: the `cfg` above proves NEON is enabled for this build,
        // and the kernel names AES only where the build has it too.
        #[allow(unsafe_code)]
        unsafe {
            neon::put8x8(rows, f)
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    scalar::put8x8(rows, f)
}

/// [`put8x8`] for a compound prediction: the intermediate
/// `Round2(sum, 7)` of each sample.
#[inline]
#[must_use]
pub fn prep8x8(rows: [&[u8; 16]; 15], f: &Filters<'_>) -> [[i16; 8]; 8] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: as in `put8x8`.
        #[allow(unsafe_code)]
        unsafe {
            neon::prep8x8(rows, f)
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    scalar::prep8x8(rows, f)
}

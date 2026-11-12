//! Sub-sample motion compensation (AV1 specification 7.11.3.4) for 8-bit
//! samples and unscaled references: the block predicted through the
//! horizontal filter and then the vertical one, or through one of them
//! when the other fraction is 0.
//!
//! An entry returns `false`, having written nothing, for a block it does
//! not take: the caller then filters the block its own way.

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "dotprod"
))]
mod neon;
mod scalar;
#[cfg(test)]
mod tests;

/// Whether this build has the kernels of this module.
pub const ACCELERATED: bool = cfg!(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "dotprod"
));

/// A block to predict and what it is predicted from.
#[derive(Clone, Copy, Debug)]
pub struct Block<'a> {
    /// The reference samples. `src[start]` is the sample three rows
    /// above and three columns left of the block's first: the filters
    /// read `w + 7` samples of each of `h + 7` rows, `src_stride` apart.
    /// The kernel loads sixteen samples at a time, so it takes a block
    /// only when one more sample follows the last one read.
    pub src: &'a [u8],
    pub start: usize,
    pub src_stride: usize,
    pub w: usize,
    pub h: usize,
    /// The rows of the subpel filter table for the horizontal and the
    /// vertical fraction.
    pub fh: &'a [i8; 8],
    pub fv: &'a [i8; 8],
}

impl Block<'_> {
    /// Whether the entries take the block: its size, the source it
    /// reads, and room in a destination of `dst_len` values with rows
    /// `dst_stride` apart.
    #[inline]
    fn taken(&self, dst_len: usize, dst_stride: usize) -> bool {
        let (w, h) = (self.w, self.h);
        matches!(w, 8 | 16 | 32 | 64 | 128)
            && (1..=128).contains(&h)
            && (h - 1) * dst_stride + w <= dst_len
            && self
                .start
                .checked_add((h + 6) * self.src_stride + w + 8)
                .is_some_and(|end| end <= self.src.len())
    }
}

/// The compound intermediate of a block (`isCompound = 1`): `Round2` of
/// the horizontal sums by 3, then of the vertical sums by 7, rows of `w`
/// values `dst_stride` apart.
#[inline]
pub fn prep_hv_u8(dst: &mut [i16], dst_stride: usize, b: &Block<'_>) -> bool {
    if !b.taken(dst.len(), dst_stride) {
        return false;
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "dotprod"
    ))]
    {
        // SAFETY: the `cfg` above proves NEON and the dot product
        // extension are enabled for this build, and the kernel names
        // AES only where the build has it too.
        #[allow(unsafe_code)]
        unsafe {
            neon::hv(neon::Out::Prep(dst), dst_stride, b);
        }
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "dotprod"
    )))]
    scalar::prep_hv(dst, dst_stride, b);
    true
}

/// The final samples of a block (`isCompound = 0`): `Round2` of the
/// horizontal sums by 3, then of the vertical sums by 11, clipped to 8
/// bits.
#[inline]
pub fn put_hv_u8(dst: &mut [u8], dst_stride: usize, b: &Block<'_>) -> bool {
    if !b.taken(dst.len(), dst_stride) {
        return false;
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "dotprod"
    ))]
    {
        // SAFETY: as in `prep_hv_u8`.
        #[allow(unsafe_code)]
        unsafe {
            neon::hv(neon::Out::Put(dst), dst_stride, b);
        }
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "dotprod"
    )))]
    scalar::put_hv(dst, dst_stride, b);
    true
}

/// The compound intermediate of a block whose vertical fraction is 0:
/// `Round2` of the horizontal sums by 3. `b.fv` is not read.
#[inline]
pub fn prep_h_u8(dst: &mut [i16], dst_stride: usize, b: &Block<'_>) -> bool {
    if !b.taken(dst.len(), dst_stride) {
        return false;
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "dotprod"
    ))]
    {
        // SAFETY: as in `prep_hv_u8`.
        #[allow(unsafe_code)]
        unsafe {
            neon::h(neon::Out::Prep(dst), dst_stride, b);
        }
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "dotprod"
    )))]
    scalar::prep_h(dst, dst_stride, b);
    true
}

/// The final samples of a block whose vertical fraction is 0: `Round2`
/// of the horizontal sums by 7, clipped to 8 bits. `b.fv` is not read.
#[inline]
pub fn put_h_u8(dst: &mut [u8], dst_stride: usize, b: &Block<'_>) -> bool {
    if !b.taken(dst.len(), dst_stride) {
        return false;
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "dotprod"
    ))]
    {
        // SAFETY: as in `prep_hv_u8`.
        #[allow(unsafe_code)]
        unsafe {
            neon::h(neon::Out::Put(dst), dst_stride, b);
        }
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "dotprod"
    )))]
    scalar::put_h(dst, dst_stride, b);
    true
}

/// The compound intermediate of a block whose horizontal fraction is 0:
/// `Round2` of the vertical sums of the samples by 3. `b.fh` is not read,
/// and the taps of `b.fv` must be even (as every AV1 filter's are).
#[inline]
pub fn prep_v_u8(dst: &mut [i16], dst_stride: usize, b: &Block<'_>) -> bool {
    if !b.taken(dst.len(), dst_stride) || b.fv.iter().any(|&f| f & 1 != 0) {
        return false;
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "dotprod"
    ))]
    {
        // SAFETY: as in `prep_hv_u8`.
        #[allow(unsafe_code)]
        unsafe {
            neon::v(neon::Out::Prep(dst), dst_stride, b);
        }
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "dotprod"
    )))]
    scalar::prep_v(dst, dst_stride, b);
    true
}

/// The final samples of a block whose horizontal fraction is 0: `Round2`
/// of the vertical sums of the samples by 7, clipped to 8 bits. `b.fh` is
/// not read, and the taps of `b.fv` must be even.
#[inline]
pub fn put_v_u8(dst: &mut [u8], dst_stride: usize, b: &Block<'_>) -> bool {
    if !b.taken(dst.len(), dst_stride) || b.fv.iter().any(|&f| f & 1 != 0) {
        return false;
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "dotprod"
    ))]
    {
        // SAFETY: as in `prep_hv_u8`.
        #[allow(unsafe_code)]
        unsafe {
            neon::v(neon::Out::Put(dst), dst_stride, b);
        }
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "dotprod"
    )))]
    scalar::put_v(dst, dst_stride, b);
    true
}

//! Sub-sample motion compensation (AV1 specification 7.11.3.4) for
//! unscaled references: the block predicted through the horizontal filter
//! and then the vertical one, or through one of them when the other
//! fraction is 0. The `_u8` entries take 8-bit samples and the `_u16`
//! entries 10-bit and 12-bit ones.
//!
//! An entry returns `false`, having written nothing, for a block it does
//! not take: the caller then filters the block its own way.

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "dotprod"
))]
#[macro_use]
mod neon;
#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "dotprod"
))]
mod neon16;
mod scalar;
mod scalar16;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests16;

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

/// A block of 10-bit or 12-bit samples to predict and what it is
/// predicted from.
#[derive(Clone, Copy, Debug)]
pub struct Block16<'a> {
    /// The reference samples, laid out as [`Block::src`] is: `src[start]`
    /// is three rows above and three columns left of the block's first,
    /// and one more sample follows the last one read.
    pub src: &'a [u16],
    pub start: usize,
    pub src_stride: usize,
    pub w: usize,
    pub h: usize,
    /// The rows of the subpel filter table for the horizontal and the
    /// vertical fraction.
    pub fh: &'a [i8; 8],
    pub fv: &'a [i8; 8],
    /// The bit depth: 10 or 12.
    pub bd: u32,
}

impl Block16<'_> {
    /// Whether the entries take the block: its depth and size, the source
    /// it reads, and room in a destination of `dst_len` values with rows
    /// `dst_stride` apart.
    #[inline]
    fn taken(&self, dst_len: usize, dst_stride: usize) -> bool {
        let (w, h) = (self.w, self.h);
        matches!(self.bd, 10 | 12)
            && matches!(w, 8 | 16 | 32 | 64 | 128)
            && (1..=128).contains(&h)
            && (h - 1) * dst_stride + w <= dst_len
            && self
                .start
                .checked_add((h + 6) * self.src_stride + w + 8)
                .is_some_and(|end| end <= self.src.len())
    }

    /// `InterRound0` (7.11.3.2).
    #[inline]
    fn round0(&self) -> i32 {
        if self.bd == 12 { 5 } else { 3 }
    }

    /// `InterRound1` of a prediction that is not compound.
    #[inline]
    fn round1(&self) -> i32 {
        if self.bd == 12 { 9 } else { 11 }
    }

    /// The largest sample.
    #[inline]
    fn max(&self) -> u16 {
        if self.bd == 12 { 4095 } else { 1023 }
    }
}

/// One entry for samples past 8 bits: the NEON kernel where the build has
/// it, the scalar twin elsewhere.
macro_rules! entry16 {
    ($(#[$doc:meta])* $name:ident, $dst:ty, $out:ident, $pass:ident, $twin:ident) => {
        $(#[$doc])*
        #[inline]
        pub fn $name(dst: &mut [$dst], dst_stride: usize, b: &Block16<'_>) -> bool {
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
                // extension are enabled for this build, and the kernel
                // names AES only where the build has it too.
                #[allow(unsafe_code)]
                unsafe {
                    neon16::$pass(neon16::Out::$out(dst), dst_stride, b);
                }
            }
            #[cfg(not(all(
                target_arch = "aarch64",
                target_feature = "neon",
                target_feature = "dotprod"
            )))]
            scalar16::$twin(dst, dst_stride, b);
            true
        }
    };
}

entry16!(
    /// The compound intermediate of a block of 10-bit or 12-bit samples:
    /// `Round2` of the horizontal sums by `InterRound0`, then of the
    /// vertical sums by 7, less 8192, rows of `w` values `dst_stride`
    /// apart.
    prep_hv_u16, i16, Prep, hv, prep_hv
);
entry16!(
    /// The final samples of a block: `Round2` of the horizontal sums by
    /// `InterRound0`, then of the vertical sums by `InterRound1`, clipped
    /// to the bit depth.
    put_hv_u16, u16, Put, hv, put_hv
);
entry16!(
    /// The compound intermediate of a block whose vertical fraction is 0:
    /// `Round2` of the horizontal sums by `InterRound0`, less 8192. `b.fv`
    /// is not read.
    prep_h_u16, i16, Prep, h, prep_h
);
entry16!(
    /// The final samples of a block whose vertical fraction is 0, clipped
    /// to the bit depth. `b.fv` is not read.
    put_h_u16, u16, Put, h, put_h
);
entry16!(
    /// The compound intermediate of a block whose horizontal fraction is
    /// 0: `Round2` of the vertical sums of the samples by `InterRound0`,
    /// less 8192. `b.fh` is not read.
    prep_v_u16, i16, Prep, v, prep_v
);
entry16!(
    /// The final samples of a block whose horizontal fraction is 0:
    /// `Round2` of the vertical sums of the samples by 7, clipped to the
    /// bit depth. `b.fh` is not read.
    put_v_u16, u16, Put, v, put_v
);

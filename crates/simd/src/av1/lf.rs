//! The deblocking sample filter (AV1 specification 7.14.6) on 8-bit
//! samples, sixteen lines of one edge at a time.
//!
//! A line is the run of samples across the edge: `p6 .. p0`, then
//! `q0 .. q6`. Every line of a call shares one filter size and one set of
//! limits. A caller with fewer than sixteen lines fills the rest with
//! [`IDLE`] samples, which no filter changes, and ignores them afterwards.

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
mod scalar;
#[cfg(test)]
mod tests;

/// The limits of 7.14.4 for 8-bit samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub limit: u8,
    pub blimit: u8,
    pub thresh: u8,
}

/// The filter of an edge: the spec's `filterSize` with the plane folded in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Size {
    /// Any plane, `filterSize` 4: reads and writes `p1 .. q1`.
    Four,
    /// Chroma, `filterSize` 8: reads `p2 .. q2`, writes `p1 .. q1`.
    Six,
    /// Luma, `filterSize` 8: reads `p3 .. q3`, writes `p2 .. q2`.
    Eight,
    /// Luma, `filterSize` 16: reads `p6 .. q6`, writes `p5 .. q5`.
    Sixteen,
}

impl Size {
    /// First and one-past-last sample a filter of this size reads, in a
    /// line of 14 where index 7 is `q0`.
    #[must_use]
    pub const fn reads(self) -> (usize, usize) {
        match self {
            Self::Four => (5, 9),
            Self::Six => (4, 10),
            Self::Eight => (3, 11),
            Self::Sixteen => (0, 14),
        }
    }

    /// First and one-past-last sample a filter of this size may write.
    #[must_use]
    pub const fn writes(self) -> (usize, usize) {
        match self {
            Self::Four | Self::Six => (5, 9),
            Self::Eight => (4, 10),
            Self::Sixteen => (1, 13),
        }
    }
}

/// Samples of a line no filter changes: `p` side then `q` side. The step
/// across the edge is above every `blimit`.
pub const IDLE: (u8, u8) = (0, 255);

/// Filters sixteen lines given by sample: `taps[k][l]` is sample `k` of
/// line `l`, where `taps[7]` holds every line's `q0`. This is how the
/// rows around a horizontal edge lie in a picture. Only the samples
/// [`Size::reads`] names are read, and only those [`Size::writes`] names
/// change. Returns whether any line passed the filter mask.
pub fn by_sample(taps: &mut [[u8; 16]; 14], size: Size, lim: Limits) -> bool {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: the `cfg` above proves NEON is enabled for this build.
        #[allow(unsafe_code)]
        unsafe {
            neon::by_sample(taps, size, lim)
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    scalar::by_sample(taps, size, lim)
}

/// Filters sixteen lines given by line, eight samples each: `p3 .. p0`,
/// then `q0 .. q3`. This is how the columns around a vertical edge lie in
/// a picture. `size` is not [`Size::Sixteen`], which filters nothing here.
/// Returns whether any line passed the filter mask.
pub fn by_line(lines: &mut [[u8; 8]; 16], size: Size, lim: Limits) -> bool {
    if size == Size::Sixteen {
        return false;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: the `cfg` above proves NEON is enabled for this build.
        #[allow(unsafe_code)]
        unsafe {
            neon::by_line(lines, size, lim)
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    scalar::by_line(lines, size, lim)
}

/// [`by_line`] for [`Size::Sixteen`], sixteen samples a line: one unused
/// sample, `p6 .. p0`, `q0 .. q6`, one unused sample.
pub fn by_line_16(lines: &mut [[u8; 16]; 16], lim: Limits) -> bool {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: the `cfg` above proves NEON is enabled for this build.
        #[allow(unsafe_code)]
        unsafe {
            neon::by_line_16(lines, lim)
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    scalar::by_line_16(lines, lim)
}

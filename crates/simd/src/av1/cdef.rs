//! The constrained directional enhancement filter (AV1 specification 7.15).

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
mod scalar;
#[cfg(test)]
mod tests;

/// Direction search on an 8x8 block of 8-bit luma (7.15.2). Returns
/// `(yDir, var)`.
#[must_use]
pub fn direction_u8(px: &[[u8; 8]; 8]) -> (u8, i32) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: the `cfg` above proves NEON is enabled for this build.
        #[allow(unsafe_code)]
        unsafe {
            neon::direction_u8(px)
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    scalar::direction(&px.map(|r| r.map(u16::from)), 8)
}

/// Direction search on an 8x8 block of luma of `bd` bits, 8 to 12
/// (7.15.2). Returns `(yDir, var)`.
#[must_use]
pub fn direction_u16(px: &[[u16; 8]; 8], bd: u32) -> (u8, i32) {
    let bd = bd.clamp(8, 12);
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: the `cfg` above proves NEON is enabled for this build.
        #[allow(unsafe_code)]
        unsafe {
            neon::direction_u16(px, bd)
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    scalar::direction(px, bd)
}

/// Stride of the padded source [`filter_u8`] and [`filter_u16`] read.
pub const STRIDE: usize = 68;

/// Value in the padded source of a sample that is not `CdefAvailable`.
pub const UNAVAILABLE: u16 = 0x7000;

/// Distance in the padded source from the first sample of a window to the
/// sample being filtered: two rows and two columns of border.
pub const CENTRE: usize = 2 * STRIDE + 2;

/// Samples of padded source one block's filter may read: the eight rows
/// of the block, each with its taps up to [`CENTRE`] on either side.
pub const WINDOW: usize = 7 * STRIDE + 2 * CENTRE + 8;

/// One block's filter (7.15.3), with the strengths already scaled to the
/// bit depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Filter {
    /// Primary and secondary strength. Not both zero.
    pub pri: u16,
    pub sec: u16,
    /// `max(0, damping - FloorLog2(strength))` for each.
    pub pri_shift: u32,
    pub sec_shift: u32,
    /// `Cdef_Pri_Taps` for the two primary taps.
    pub pri_taps: [i16; 2],
    /// Where the taps on one side sit, from the window's first sample of
    /// a row, at most `2 * CENTRE`: the two primary taps, then the
    /// secondary taps at `dir + 2` and `dir - 2`, near pair first. The
    /// tap on the other side is at `2 * CENTRE` minus this.
    pub pri_at: [usize; 2],
    pub sec_at: [usize; 4],
}

/// Filters the first `rows` rows (at most 8) of an 8-wide block of 8-bit
/// samples. Sample `(i, j)` of the block is `win[i * STRIDE + CENTRE + j]`,
/// and samples outside the picture hold [`UNAVAILABLE`]. A narrower block
/// ignores the columns it does not have. Rows past `rows` come back zero.
#[must_use]
pub fn filter_u8(win: &[u16; WINDOW], f: &Filter, rows: usize) -> [[u8; 8]; 8] {
    filter_u16(win, f, rows).map(|r| r.map(|v| v as u8))
}

/// [`filter_u8`] for samples of up to 12 bits.
#[must_use]
pub fn filter_u16(win: &[u16; WINDOW], f: &Filter, rows: usize) -> [[u16; 8]; 8] {
    let rows = rows.min(8);
    // Whatever the caller derived, no tap leaves the window and no shift
    // passes the width of a sample.
    let f = &Filter {
        pri_shift: f.pri_shift.min(15),
        sec_shift: f.sec_shift.min(15),
        pri_at: f.pri_at.map(|at| at.min(2 * CENTRE)),
        sec_at: f.sec_at.map(|at| at.min(2 * CENTRE)),
        ..*f
    };
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: the `cfg` above proves NEON is enabled for this build.
        #[allow(unsafe_code)]
        unsafe {
            neon::filter(win, f, rows)
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    scalar::filter(win, f, rows)
}

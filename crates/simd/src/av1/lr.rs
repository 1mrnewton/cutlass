//! The Wiener filter of loop restoration (AV1 specification 7.17.4) for
//! 8-bit samples, a pass at a time.
//!
//! The filter is separable and symmetric: seven taps `f0 f1 f2 f3 f2 f1
//! f0` across a row, rounded and clamped into 16 bits, then seven down the
//! rows of that intermediate. A caller filters `rows + 6` rows across and
//! then each output row from seven of them.
//!
//! Samples of 10 and 12 bits stay on the decoder's own loops: measured on
//! 1080p, this kernel's 32-bit sums were no faster there than the lanes
//! the compiler makes of those loops.
//!
//! An entry returns `false`, having written nothing, for a row it does
//! not take: the caller then filters the row its own way.

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
mod scalar;
#[cfg(test)]
mod tests;

/// The rounding and the clamp of one pass at a bit depth (7.17.4):
/// `InterRound0` is 3, or 5 at 12 bits, and `InterRound1` is 11, or 9.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rounding {
    /// The bits `Round2` takes off the horizontal sums.
    pub round0: u32,
    /// The bits it takes off the vertical sums.
    pub round1: u32,
    /// The clamp of the horizontal pass: `-offset ..= limit - offset`.
    pub low: i16,
    pub high: i16,
    /// The largest sample.
    pub max: u16,
}

impl Rounding {
    /// The values for samples of `bd` bits, 8 to 12.
    #[must_use]
    pub fn of(bd: u32) -> Self {
        let bd = bd.clamp(8, 12);
        let (round0, round1) = if bd == 12 { (5, 9) } else { (3, 11) };
        let offset = 1i32 << (bd + 7 - round0 - 1);
        let limit = (1i32 << (bd + 1 + 7 - round0)) - 1;
        Self {
            round0,
            round1,
            low: (-offset) as i16,
            high: (limit - offset) as i16,
            max: ((1u32 << bd) - 1) as u16,
        }
    }
}

/// Whether a pass over `w` outputs is one the kernels take: at least
/// eight columns, so that the last eight can be filtered again in place of
/// a narrower tail.
#[inline]
fn taken(w: usize) -> bool {
    w >= 8
}

/// The horizontal pass of one row: `dst[c]` is `Round2` of the seven
/// samples from `src[c]` times the taps, clamped. `src` holds the row from
/// three samples left of the first output, and one sample past the last
/// one read (`dst.len() + 7` samples in all), which a load of eight takes
/// in.
///
/// `taps` is `[f0, f1, f2, f3]`, the row's first four.
#[inline]
pub fn wiener_h(dst: &mut [i16], src: &[i16], taps: &[i16; 4], r: &Rounding) -> bool {
    if !taken(dst.len()) || src.len() < dst.len() + 7 {
        return false;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: the `cfg` above proves NEON is enabled for this build,
        // and the kernel names AES only where the build has it too.
        #[allow(unsafe_code)]
        unsafe {
            neon::horizontal(dst, src, taps, r);
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    scalar::horizontal(dst, src, taps, r);
    true
}

/// The vertical pass of one row of 8-bit samples: `dst[c]` is `Round2` of
/// the seven intermediates `rows[t][c]` times the taps, clipped to 8 bits.
/// Each of `rows` holds at least `dst.len()` values.
#[inline]
pub fn wiener_v_u8(dst: &mut [u8], rows: &[&[i16]; 7], taps: &[i16; 4], r: &Rounding) -> bool {
    if !taken(dst.len()) || rows.iter().any(|row| row.len() < dst.len()) {
        return false;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: as in `wiener_h`.
        #[allow(unsafe_code)]
        unsafe {
            neon::vertical(dst, rows, taps, r);
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    for (c, d) in dst.iter_mut().enumerate() {
        *d = scalar::vertical(rows, c, taps, r) as u8;
    }
    true
}

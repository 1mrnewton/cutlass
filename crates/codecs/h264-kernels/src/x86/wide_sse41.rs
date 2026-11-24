//! Sixteen-wide luma prediction for the SSE4.1 tier: a 128-bit register
//! holds eight 16-bit sums, so a block sixteen wide is its two halves.

#![allow(unsafe_code)]

use super::kernels::luma8;

feat! {
    /// A luma block sixteen wide: see `luma8`.
    ///
    /// # Safety
    ///
    /// As for `luma8`, for a block sixteen wide: what `luma_reach`
    /// checks.
    pub(crate) unsafe fn luma16<const AVG: bool>(
        dst: *mut u8,
        ds: usize,
        src: *const u8,
        s: usize,
        h: usize,
        frac: (u8, u8),
    ) {
        // SAFETY: each half reads and writes inside what the whole block
        // does: the right one reaches eleven bytes past its eighth
        // column, which is within the 32 from two before the first.
        unsafe {
            luma8::<AVG>(dst, ds, src, s, h, frac);
            luma8::<AVG>(dst.add(8), ds, src.add(8), s, h, frac);
        }
    }
}

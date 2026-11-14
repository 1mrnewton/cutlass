//! The luma deblocking filter of 8-bit samples, and of 9 to 11 bits (clauses 8.7.2.5.3 and
//! 8.7.2.5.7 of ITU-T H.265), eight lines at a time: two segments of four
//! lines along one edge, each with its own decisions.
//!
//! An entry returns `false`, having changed nothing, when it does not run
//! on this CPU; the caller then filters the lines its own way.

#[cfg(target_arch = "x86_64")]
mod sse;
#[cfg(test)]
mod tests;

/// Whether [`luma8`] runs on this CPU.
#[must_use]
#[inline]
pub fn accelerated() -> bool {
    crate::wide::has_avx2()
}

/// The filter parameters of one segment of four lines.
#[derive(Clone, Copy, Debug, Default)]
pub struct Segment {
    /// `β` and `tC` (both 0 when the segment is not filtered).
    pub beta: i16,
    pub tc: i16,
    /// The samples on the `p` (`q`) side stay as they are (PCM or
    /// transquant bypass).
    pub no_p: bool,
    pub no_q: bool,
}

/// Filter eight lines across an edge: `rows[i]` holds sample `i` across
/// the edge (`p3 p2 p1 p0 q0 q1 q2 q3`) of each line; lines 0 to 3 make
/// segment `seg[0]`, lines 4 to 7 segment `seg[1]`. Every decision is the
/// one of the clauses, taken on lines 0 and 3 of each segment.
#[inline]
pub fn luma8(rows: &mut [[u8; 8]; 8], seg: [Segment; 2]) -> bool {
    if !crate::wide::has_avx2() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `has_avx2` checked at run time that the CPU has AVX2,
        // and with it the SSE4.1 the kernel names.
        #[allow(unsafe_code)]
        unsafe {
            sse::luma8(rows, seg);
        }
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (rows, seg);
        false
    }
}

/// [`luma8`] on eight lines along a vertical edge: `lines[l]` holds the
/// samples `p3` to `q3` of line `l`.
#[inline]
pub fn luma8_lines(lines: &mut [[u8; 8]; 8], seg: [Segment; 2]) -> bool {
    if !crate::wide::has_avx2() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `has_avx2` checked at run time that the CPU has AVX2.
        #[allow(unsafe_code)]
        unsafe {
            sse::luma8_lines(lines, seg);
        }
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (lines, seg);
        false
    }
}

/// [`luma8`] for samples of `bit_depth` bits, 9 to 11 (the sums of the
/// weak filter fit 16 bits up to 11), with `β` and `tC` already scaled to
/// the depth. Samples are clipped to the depth's range.
#[inline]
pub fn luma16(rows: &mut [[u16; 8]; 8], seg: [Segment; 2], bit_depth: u32) -> bool {
    if !(9..=11).contains(&bit_depth) || !crate::wide::has_avx2() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `has_avx2` checked at run time that the CPU has AVX2.
        #[allow(unsafe_code)]
        unsafe {
            sse::luma16(rows, seg, ((1i32 << bit_depth) - 1) as i16)
        };
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (rows, seg);
        false
    }
}

/// [`luma16`] on eight lines along a vertical edge: `lines[l]` holds the
/// samples `p3` to `q3` of line `l`.
#[inline]
pub fn luma16_lines(lines: &mut [[u16; 8]; 8], seg: [Segment; 2], bit_depth: u32) -> bool {
    if !(9..=11).contains(&bit_depth) || !crate::wide::has_avx2() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `has_avx2` checked at run time that the CPU has AVX2.
        #[allow(unsafe_code)]
        unsafe {
            sse::luma16_lines(lines, seg, ((1i32 << bit_depth) - 1) as i16)
        };
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (lines, seg);
        false
    }
}

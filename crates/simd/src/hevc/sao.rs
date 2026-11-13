//! Sample adaptive offset of 8-bit rows (clause 8.7.3 of ITU-T H.265).
//!
//! An entry returns `false`, having written nothing, when it does not run
//! on this CPU; the caller then applies the offsets its own way.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(test)]
mod tests;

/// Edge offset of one row: `dst[x] = Clip1(cur[x] + o)`, where `o` is
/// `offsets[k]` for `edgeIdx` 0, 1, 3, 4 (`k` 0 to 3) and 0 for `edgeIdx`
/// 2, `edgeIdx = 2 + Sign(cur[x] - a[x]) + Sign(cur[x] - b[x])`: `a` and
/// `b` are the two neighbours of each sample, already aligned with `cur`.
/// The offsets of 8-bit samples lie within -7 and 7.
#[inline]
pub fn edge_row(dst: &mut [u8], cur: &[u8], (a, b): (&[u8], &[u8]), offsets: [i8; 4]) -> bool {
    let n = dst.len();
    if cur.len() < n || a.len() < n || b.len() < n || !crate::wide::has_avx2() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `has_avx2` checked at run time that the CPU has AVX2.
        #[allow(unsafe_code)]
        unsafe {
            avx2::edge_row(dst, cur, (a, b), offsets);
        }
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = offsets;
        false
    }
}

/// Band offset of one row: `dst[x] = Clip1(cur[x] + o)`, where `o` is
/// `offsets[k]` when `k = ((cur[x] >> 3) - position) & 31` is below 4,
/// else 0.
#[inline]
pub fn band_row(dst: &mut [u8], cur: &[u8], position: u8, offsets: [i8; 4]) -> bool {
    if cur.len() < dst.len() || !crate::wide::has_avx2() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `has_avx2` checked at run time that the CPU has AVX2.
        #[allow(unsafe_code)]
        unsafe {
            avx2::band_row(dst, cur, position, offsets);
        }
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (position, offsets);
        false
    }
}

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

/// [`edge_row`] over `rows` rows of `n` samples of one buffer `src`
/// (rows `stride` apart): row `j` starts at `first + j * stride`, and its
/// neighbours `off_a` and `off_b` samples from there; row `j` goes to
/// `dst[j * dst_stride..j * dst_stride + n]`. `false`, with nothing
/// written, when a row or a neighbour lies outside `src`.
#[inline]
pub fn edge_strided(
    (dst, dst_stride): (&mut [u8], usize),
    (src, stride, first): (&[u8], usize, usize),
    (rows, n): (usize, usize),
    (off_a, off_b): (isize, isize),
    offsets: [i8; 4],
) -> bool {
    if rows == 0 {
        return true;
    }
    let last = first + (rows - 1) * stride;
    let reach = |off: isize| {
        first.checked_add_signed(off).is_some()
            && last
                .checked_add_signed(off)
                .is_some_and(|e| e + n <= src.len())
    };
    let fits = n > 0
        && last + n <= src.len()
        && reach(off_a)
        && reach(off_b)
        && dst_stride >= n
        && (rows - 1) * dst_stride + n <= dst.len();
    if !fits || !crate::wide::has_avx2() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `has_avx2` checked at run time that the CPU has AVX2.
        #[allow(unsafe_code)]
        unsafe {
            avx2::edge_strided(
                (dst, dst_stride),
                (src, stride, first),
                (rows, n),
                (off_a, off_b),
                offsets,
            );
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

/// [`edge_row`] for samples of 9 to 12 bits (`bit_depth`), whose offsets
/// are scaled up and may be as large as 7 << 5.
#[inline]
pub fn edge_row16(
    dst: &mut [u16],
    cur: &[u16],
    (a, b): (&[u16], &[u16]),
    offsets: [i16; 4],
    bit_depth: u32,
) -> bool {
    let n = dst.len();
    if cur.len() < n
        || a.len() < n
        || b.len() < n
        || !(9..=12).contains(&bit_depth)
        || !crate::wide::has_avx2()
    {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `has_avx2` checked at run time that the CPU has AVX2.
        #[allow(unsafe_code)]
        unsafe {
            avx2::edge_row16(dst, cur, (a, b), offsets, bit_depth);
        }
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = offsets;
        false
    }
}

/// [`band_row`] for samples of 9 to 12 bits (`bit_depth`): the band of a
/// sample is its top five bits, `cur[x] >> (bit_depth - 5)`.
#[inline]
pub fn band_row16(
    dst: &mut [u16],
    cur: &[u16],
    position: u8,
    offsets: [i16; 4],
    bit_depth: u32,
) -> bool {
    if cur.len() < dst.len() || !(9..=12).contains(&bit_depth) || !crate::wide::has_avx2() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `has_avx2` checked at run time that the CPU has AVX2.
        #[allow(unsafe_code)]
        unsafe {
            avx2::band_row16(dst, cur, position, offsets, bit_depth);
        }
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (position, offsets);
        false
    }
}

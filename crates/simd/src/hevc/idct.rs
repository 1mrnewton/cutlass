//! The inverse DCT of HEVC (clause 8.6.4.2 of ITU-T H.265) for blocks of
//! 8×8 to 32×32, as the two matrix products of the clause with 16-bit
//! operands: the scaled coefficients and the clipped intermediate values
//! both fit 16 bits, so pairs of them are multiplied by pairs of matrix
//! entries into 32-bit sums (`pmaddwd`).
//!
//! The 4×4 blocks take the DCT or, for intra luma, the DST of equation
//! 8-316, by the same two products.
//!
//! An entry returns `false`, having written nothing, when it does not run
//! on this CPU or does not take the block.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(test)]
mod tests;

/// `transMatrix` of equation 8-317: entry `[k][n]` of the 32-point
/// transform; an `N`-point transform takes rows `k · 32 / N`.
pub const DCT32: [[i8; 32]; 32] = {
    const COS: [i8; 33] = [
        64, 90, 90, 90, 89, 88, 87, 85, 83, 82, 80, 78, 75, 73, 70, 67, 64, 61, 57, 54, 50, 46, 43,
        38, 36, 31, 25, 22, 18, 13, 9, 4, 0,
    ];
    let mut out = [[0i8; 32]; 32];
    let mut k = 0;
    while k < 32 {
        let mut n = 0;
        while n < 32 {
            out[k][n] = if k == 0 {
                64
            } else {
                let mut a = ((2 * n + 1) * k) % 128;
                let mut sign = 1;
                if a > 64 {
                    a = 128 - a;
                }
                if a > 32 {
                    a = 64 - a;
                    sign = -1;
                }
                sign * COS[a]
            };
            n += 1;
        }
        k += 1;
    }
    out
};

/// `transMatrix` of the 4×4 DST (equation 8-316): entry `[k][n]`.
pub const DST4: [[i8; 4]; 4] = [
    [29, 55, 74, 84],
    [74, 74, 0, -74],
    [84, -29, -74, 55],
    [55, -84, 74, -29],
];

/// The residual of a 4×4 block of scaled coefficients `coeffs` (rows of
/// 4, each within 16 bits) into `out`: the DCT, or the DST when `dst`.
/// The first pass rounds by 7 bits and clips to 16 bits, the second
/// rounds by `20 - bit_depth` bits.
#[inline]
pub fn inverse4(coeffs: &[i32], out: &mut [i32], dst: bool, bit_depth: u32) -> bool {
    let (Some(c), Some(o)) = (coeffs.first_chunk::<16>(), out.first_chunk_mut::<16>()) else {
        return false;
    };
    if !(8..=12).contains(&bit_depth) || !crate::wide::has_avx2() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `has_avx2` checked at run time that the CPU has AVX2.
        #[allow(unsafe_code)]
        unsafe {
            avx2::inverse4(c, o, dst, bit_depth);
        }
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (c, o, dst);
        false
    }
}

/// The residual of an `n`×`n` block (`n` 8, 16 or 32) of scaled
/// coefficients `coeffs` (rows of `n`, each within 16 bits) into `out`
/// (rows of `n`): coefficients past column `cols` or row `rows` (each at
/// least 1) are zero. The first pass rounds by 7 bits and clips to 16
/// bits, the second rounds by `20 - bit_depth` bits.
#[inline]
pub fn inverse(
    coeffs: &[i32],
    out: &mut [i32],
    n: usize,
    (cols, rows): (usize, usize),
    bit_depth: u32,
) -> bool {
    let fits = matches!(n, 8 | 16 | 32)
        && (1..=n).contains(&cols)
        && (1..=n).contains(&rows)
        && coeffs.len() >= n * n
        && out.len() >= n * n
        && (8..=12).contains(&bit_depth);
    if !fits || !crate::wide::has_avx2() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `has_avx2` checked at run time that the CPU has AVX2.
        #[allow(unsafe_code)]
        unsafe {
            match n {
                8 => avx2::inverse::<8>(coeffs, out, (cols, rows), bit_depth),
                16 => avx2::inverse::<16>(coeffs, out, (cols, rows), bit_depth),
                _ => avx2::inverse::<32>(coeffs, out, (cols, rows), bit_depth),
            }
        }
        true
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

//! The fixed-point primitives the SILK reference decoder is written in
//! (`silk/macros.h`, `silk/SigProc_FIX.h`, `silk/Inlines.h`,
//! `silk/log2lin.c`, `silk/lin2log.c`, `silk/sum_sqr_shift.c`).
//!
//! The reference computes in 32-bit integers and relies on two's-complement
//! wraparound in a few places. Every helper here wraps instead of
//! overflowing, so a hostile bitstream can drive values anywhere and the
//! decoder still never panics; on valid streams the results are the
//! reference's bit for bit.

/// `silk_SMULWB`: `(a * (i16)b) >> 16`.
#[inline(always)]
pub(crate) fn smulwb(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b as i16)) >> 16) as i32
}

/// `silk_SMLAWB`: `a + ((b * (i16)c) >> 16)`.
#[inline(always)]
pub(crate) fn smlawb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(smulwb(b, c))
}

/// `silk_SMULWW`: `(a * b) >> 16`.
#[inline(always)]
pub(crate) fn smulww(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 16) as i32
}

/// `silk_SMLAWW`: `a + ((b * c) >> 16)`.
#[inline(always)]
pub(crate) fn smlaww(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(smulww(b, c))
}

/// `silk_SMULBB`: `(i16)a * (i16)b`.
#[inline(always)]
pub(crate) fn smulbb(a: i32, b: i32) -> i32 {
    i32::from(a as i16) * i32::from(b as i16)
}

/// `silk_SMLABB`: `a + (i16)b * (i16)c`, wrapping.
#[inline(always)]
pub(crate) fn smlabb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(smulbb(b, c))
}

/// `silk_SMMUL`: `(a * b) >> 32`.
#[inline(always)]
pub(crate) fn smmul(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 32) as i32
}

/// `silk_RSHIFT_ROUND` for `shift` in `1..32`.
#[inline(always)]
pub(crate) fn rshift_round(a: i32, shift: u32) -> i32 {
    if shift == 1 {
        (a >> 1).wrapping_add(a & 1)
    } else {
        ((a >> (shift - 1)).wrapping_add(1)) >> 1
    }
}

/// `silk_RSHIFT_ROUND64` for `shift` in `1..64`.
#[inline(always)]
pub(crate) fn rshift_round64(a: i64, shift: u32) -> i64 {
    if shift == 1 {
        (a >> 1).wrapping_add(a & 1)
    } else {
        ((a >> (shift - 1)).wrapping_add(1)) >> 1
    }
}

/// `silk_LSHIFT32`: a left shift that drops the bits shifted out.
#[inline(always)]
pub(crate) fn lshift(a: i32, shift: u32) -> i32 {
    ((a as u32) << (shift & 31)) as i32
}

/// `silk_SAT16`.
#[inline(always)]
pub(crate) fn sat16(a: i32) -> i16 {
    a.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

/// `silk_LSHIFT_SAT32`: shifts left, saturating to the 32-bit range.
#[inline]
pub(crate) fn lshift_sat32(a: i32, shift: u32) -> i32 {
    let shift = shift.min(31);
    let lo = i32::MIN >> shift;
    let hi = i32::MAX >> shift;
    lshift(a.clamp(lo, hi), shift)
}

/// `silk_CLZ32`: leading zeros of the 32-bit pattern, 32 for zero.
#[inline(always)]
pub(crate) fn clz32(a: i32) -> i32 {
    (a as u32).leading_zeros() as i32
}

/// `silk_DIV32`, returning 0 for a zero divisor and wrapping `MIN / -1`.
#[inline]
pub(crate) fn div32(a: i32, b: i32) -> i32 {
    if b == 0 { 0 } else { a.wrapping_div(b) }
}

/// Shifts `a` by `lshift` bits: right for a positive count, left with
/// saturation for a negative one, and 0 past 31 bits. The tail of
/// `silk_DIV32_varQ` and `silk_INVERSE32_varQ`.
#[inline]
fn shift_to_q(result: i32, lshift: i32) -> i32 {
    if lshift < 0 {
        lshift_sat32(result, lshift.unsigned_abs())
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

/// Normalization headroom of `a`: `CLZ(|a|) - 1`, kept in `0..=31` so a
/// hostile `i32::MIN` or zero never shifts by a negative count.
#[inline]
fn headroom(a: i32) -> i32 {
    (clz32(a.wrapping_abs()) - 1).clamp(0, 31)
}

/// `silk_DIV32_varQ`: approximately `(a << q_res) / b`.
pub(crate) fn div32_varq(a: i32, b: i32, q_res: i32) -> i32 {
    let a_headrm = headroom(a);
    let a_nrm = lshift(a, a_headrm as u32);
    let b_headrm = headroom(b);
    let b_nrm = lshift(b, b_headrm as u32);
    // Inverse of b with 14 bits of precision, Q(29 + 16 - b_headrm).
    let b_inv = div32(i32::MAX >> 2, b_nrm >> 16);
    // First approximation, Q(29 + a_headrm - b_headrm).
    let mut result = smulwb(a_nrm, b_inv);
    // Residual; the reference lets this wrap.
    let a_nrm = a_nrm.wrapping_sub(lshift(smmul(b_nrm, result), 3));
    result = smlawb(result, a_nrm, b_inv);
    shift_to_q(result, 29 + a_headrm - b_headrm - q_res)
}

/// `silk_INVERSE32_varQ`: approximately `(1 << q_res) / b`.
pub(crate) fn inverse32_varq(b: i32, q_res: i32) -> i32 {
    let b_headrm = headroom(b);
    let b_nrm = lshift(b, b_headrm as u32);
    let b_inv = div32(i32::MAX >> 2, b_nrm >> 16);
    let mut result = lshift(b_inv, 16);
    let err_q32 = lshift((1i32 << 29).wrapping_sub(smulwb(b_nrm, b_inv)), 3);
    result = smlaww(result, err_q32, b_inv);
    shift_to_q(result, 61 - b_headrm - q_res)
}

/// `silk_CLZ_FRAC`: leading zeros and the 7 bits after the leading one.
#[inline]
fn clz_frac(a: i32) -> (i32, i32) {
    let lz = clz32(a);
    let rot = 24 - lz;
    let x = a as u32;
    let rotated = if rot >= 0 {
        x.rotate_right(rot as u32)
    } else {
        x.rotate_left(rot.unsigned_abs())
    };
    (lz, (rotated & 0x7f) as i32)
}

/// `silk_SQRT_APPROX`: square root within a few percent.
pub(crate) fn sqrt_approx(x: i32) -> i32 {
    if x <= 0 {
        return 0;
    }
    let (lz, frac_q7) = clz_frac(x);
    let mut y: i32 = if lz & 1 != 0 { 32768 } else { 46214 };
    y >>= lz >> 1;
    smlawb(y, y, smulbb(213, frac_q7))
}

/// `silk_log2lin`: approximately `2^(in / 128)`.
pub(crate) fn log2lin(in_log_q7: i32) -> i32 {
    if in_log_q7 < 0 {
        return 0;
    }
    if in_log_q7 >> 7 >= 31 {
        // 2^31 does not fit; the reference never asks for more than 3967.
        return i32::MAX;
    }
    let out = 1i32 << (in_log_q7 >> 7);
    let frac_q7 = in_log_q7 & 0x7f;
    let poly = smlawb(frac_q7, smulbb(frac_q7, 128 - frac_q7), -174);
    if in_log_q7 < 2048 {
        out.wrapping_add(out.wrapping_mul(poly) >> 7)
    } else {
        out.wrapping_add((out >> 7).wrapping_mul(poly))
    }
}

/// `silk_RAND`: the linear congruential generator.
#[inline(always)]
pub(crate) fn rand(seed: i32) -> i32 {
    907_633_515i32.wrapping_add(seed.wrapping_mul(196_314_165))
}

/// `silk_sum_sqr_shift`: the energy of `x` and the right shift that keeps
/// it within 30 bits.
pub(crate) fn sum_sqr_shift(x: &[i16]) -> (i32, i32) {
    let sq = |v: i16| i32::from(v) * i32::from(v);
    let len = x.len().saturating_sub(1);
    let mut nrg: i32 = 0;
    let mut shft: i32 = 0;
    let mut i = 0;
    while i < len {
        nrg = nrg.wrapping_add(sq(x[i]));
        nrg = nrg.wrapping_add(sq(x[i + 1]));
        if nrg < 0 {
            nrg = ((nrg as u32) >> 2) as i32;
            shft = 2;
            // The reference breaks before its loop increment, so the second
            // loop adds this pair again at the new shift.
            break;
        }
        i += 2;
    }
    while i < len {
        let tmp = sq(x[i]).wrapping_add(sq(x[i + 1]));
        nrg = (nrg as u32).wrapping_add((tmp as u32) >> shft) as i32;
        if nrg < 0 {
            nrg = ((nrg as u32) >> 2) as i32;
            shft += 2;
        }
        i += 2;
    }
    if i == len
        && let Some(&last) = x.get(i)
    {
        nrg = (nrg as u32).wrapping_add((sq(last) as u32) >> shft) as i32;
    }
    if nrg as u32 & 0xC000_0000 != 0 {
        nrg = ((nrg as u32) >> 2) as i32;
        shft += 2;
    }
    (nrg, shft)
}

#[cfg(test)]
mod tests {
    use super::{
        div32_varq, inverse32_varq, log2lin, rshift_round, smulwb, smulww, sqrt_approx,
        sum_sqr_shift,
    };

    #[test]
    fn multiply_helpers_match_the_split_reference_form() {
        // The reference computes SMULWB as hi * b + ((lo * b) >> 16).
        for &(a, b) in &[
            (123_456_789, -12_345),
            (-7, 3),
            (i32::MAX, i32::MAX),
            (i32::MIN, 1),
        ] {
            let b16 = i32::from(b as i16);
            let split = (a >> 16)
                .wrapping_mul(b16)
                .wrapping_add(((a & 0xffff) * b16) >> 16);
            assert_eq!(smulwb(a, b), split);
        }
        assert_eq!(smulww(1 << 20, 3 << 16), 3 << 20);
        assert_eq!(rshift_round(5, 1), 3);
        assert_eq!(rshift_round(-5, 2), -1);
    }

    #[test]
    fn division_approximations_are_close() {
        for &(a, b, q) in &[(1000, 7, 16), (65536, 70000, 16), (-5000, 33, 10)] {
            let exact = ((f64::from(a) * f64::from(1u32 << q)) / f64::from(b)) as i32;
            let got = div32_varq(a, b, q);
            assert!(
                (got - exact).abs() <= exact.abs() / 1000 + 2,
                "{a}/{b}: {got} vs {exact}"
            );
        }
        for &(b, q) in &[(65536, 47), (1 << 20, 46), (12345, 30)] {
            let exact = (2f64.powi(q) / f64::from(b)) as i64;
            let got = i64::from(inverse32_varq(b, q));
            assert!(
                (got - exact).abs() <= exact / 1000 + 2,
                "1/{b}: {got} vs {exact}"
            );
        }
        // Hostile operands do not panic.
        let _ = div32_varq(i32::MIN, 0, 16);
        let _ = inverse32_varq(0, 47);
        let _ = inverse32_varq(i32::MIN, 1);
    }

    #[test]
    fn log_and_square_root_approximations() {
        assert_eq!(log2lin(0), 1);
        assert_eq!(log2lin(128 * 16), 1 << 16);
        assert_eq!(log2lin(-1), 0);
        for x in [100, 10_000, 1 << 20, i32::MAX] {
            let got = f64::from(sqrt_approx(x));
            let exact = f64::from(x).sqrt();
            assert!((got - exact).abs() <= exact * 0.1, "{x}");
        }
    }

    #[test]
    fn energy_shift_keeps_two_headroom_bits() {
        assert_eq!(sum_sqr_shift(&[3, 4, 12]), (169, 0));
        let loud = [i16::MAX; 320];
        let (nrg, shift) = sum_sqr_shift(&loud);
        assert!(nrg > 0 && nrg < 1 << 30);
        let exact = 320.0 * f64::from(i16::MAX).powi(2);
        assert!((f64::from(nrg) * 2f64.powi(shift) - exact).abs() / exact < 0.01);
        assert_eq!(sum_sqr_shift(&[]), (0, 0));
    }
}

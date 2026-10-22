//! Integer helpers matching the specification's arithmetic conventions.
//!
//! Right shifts on signed values are arithmetic. Division truncates toward
//! zero. `Round2` uses the integer form in section 4.

/// `Clip3(x, y, z)` with `x <= y`.
#[inline]
pub fn clip3(lo: i32, hi: i32, z: i32) -> i32 {
    z.clamp(lo, hi)
}

/// `Clip1(x)` into `0 .. 2^bit_depth - 1`.
#[inline]
pub fn clip1(bit_depth: u8, x: i32) -> i32 {
    let max = (1i32 << bit_depth) - 1;
    clip3(0, max, x)
}

/// `Round2(x, n)` for a non-negative shift. `x` is widened before the add.
#[inline]
#[allow(dead_code)]
pub fn round2(x: i32, n: u32) -> i32 {
    if n == 0 {
        return x;
    }
    let x = i64::from(x);
    ((x + (1i64 << (n - 1))) >> n) as i32
}

/// `Round2` on a 64-bit accumulator.
#[inline]
#[allow(dead_code)]
pub fn round2_i64(x: i64, n: u32) -> i64 {
    if n == 0 {
        return x;
    }
    (x + (1i64 << (n - 1))) >> n
}

/// `Round2Signed`.
#[inline]
#[allow(dead_code)]
pub fn round2_signed(x: i32, n: u32) -> i32 {
    if x >= 0 { round2(x, n) } else { -round2(-x, n) }
}

/// `FloorLog2`. `x` must be >= 1.
#[inline]
pub fn floor_log2(x: u32) -> u32 {
    debug_assert!(x >= 1);
    31 - x.leading_zeros()
}

/// `CeilLog2`. Returns 0 when `x < 2`.
#[inline]
pub fn ceil_log2(x: u32) -> u32 {
    if x < 2 {
        0
    } else {
        32 - (x - 1).leading_zeros()
    }
}

/// Sign-extended order-hint distance (`get_relative_dist`).
#[allow(dead_code)]
pub fn relative_dist(a: u32, b: u32, bits: u8) -> i32 {
    if bits == 0 {
        return 0;
    }
    let diff = a.wrapping_sub(b);
    let m = 1u32 << (bits - 1);
    (diff & (m - 1)) as i32 - (diff & m) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round2_matches_spec_examples() {
        assert_eq!(round2(7, 0), 7);
        assert_eq!(round2(8, 1), 4);
        assert_eq!(round2(7, 1), 4);
        assert_eq!(round2_signed(-3, 1), -2);
    }

    #[test]
    fn logs() {
        assert_eq!(floor_log2(1), 0);
        assert_eq!(floor_log2(8), 3);
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(5), 3);
    }

    #[test]
    fn order_hint_wrap() {
        // 3-bit hints: 1 is ahead of 7 by 2 (mod 8, sign extended).
        assert_eq!(relative_dist(1, 7, 3), 2);
        assert_eq!(relative_dist(7, 1, 3), -2);
    }
}

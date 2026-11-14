//! Block distortion for the encoder's searches: sums of absolute
//! differences, plain and against a bilinear sub-sample position, and
//! sums of squared differences. Written over fixed-width rows so that,
//! compiled for AVX2 by [`wide`], each row is a few `vpsadbw`.

use cutlass_simd::vp9::wide;

#[inline(always)]
fn row<const W: usize>(a: &[u8], b: &[u8]) -> u32 {
    let (Some(a), Some(b)) = (a.first_chunk::<W>(), b.first_chunk::<W>()) else {
        return 0;
    };
    let mut s = 0u32;
    for i in 0..W {
        s += u32::from(a[i].abs_diff(b[i]));
    }
    s
}

#[inline(always)]
fn sad_w<const W: usize>(a: &[u8], sa: usize, b: &[u8], sb: usize, h: usize) -> u32 {
    let mut acc = 0;
    for r in 0..h {
        acc += row::<W>(&a[r * sa..], &b[r * sb..]);
    }
    acc
}

/// The SAD of the `w`×`h` blocks at the start of `a` and `b` (rows `sa`
/// and `sb` apart).
#[inline]
pub fn sad(a: &[u8], sa: usize, b: &[u8], sb: usize, w: usize, h: usize) -> u32 {
    wide(|| match w {
        64 => sad_w::<64>(a, sa, b, sb, h),
        32 => sad_w::<32>(a, sa, b, sb, h),
        16 => sad_w::<16>(a, sa, b, sb, h),
        8 => sad_w::<8>(a, sa, b, sb, h),
        _ => {
            let mut acc = 0;
            for r in 0..h {
                for c in 0..w {
                    acc += u32::from(a[r * sa + c].abs_diff(b[r * sb + c]));
                }
            }
            acc
        }
    })
}

#[inline(always)]
fn sse_w<const W: usize>(a: &[u8], sa: usize, b: &[u8], sb: usize, h: usize) -> u64 {
    let mut acc = 0u64;
    for r in 0..h {
        let (Some(x), Some(y)) = (
            a[r * sa..].first_chunk::<W>(),
            b[r * sb..].first_chunk::<W>(),
        ) else {
            continue;
        };
        let mut s = 0u32;
        for i in 0..W {
            let d = i32::from(x[i]) - i32::from(y[i]);
            s += (d * d) as u32;
        }
        acc += u64::from(s);
    }
    acc
}

/// The sum of squared differences of two `w`×`h` blocks.
#[inline]
pub fn sse(a: &[u8], sa: usize, b: &[u8], sb: usize, w: usize, h: usize) -> u64 {
    wide(|| match w {
        64 => sse_w::<64>(a, sa, b, sb, h),
        32 => sse_w::<32>(a, sa, b, sb, h),
        16 => sse_w::<16>(a, sa, b, sb, h),
        8 => sse_w::<8>(a, sa, b, sb, h),
        4 => sse_w::<4>(a, sa, b, sb, h),
        _ => {
            let mut acc = 0u64;
            for r in 0..h {
                for c in 0..w {
                    let d = i64::from(a[r * sa + c]) - i64::from(b[r * sb + c]);
                    acc += (d * d) as u64;
                }
            }
            acc
        }
    })
}

#[inline(always)]
fn across<const W: usize>(b: &[u8], fx: u16, out: &mut [u16; W]) {
    let (Some(p), Some(q)) = (
        b.first_chunk::<W>(),
        b.get(1..).and_then(|t| t.first_chunk::<W>()),
    ) else {
        return;
    };
    for i in 0..W {
        out[i] = (u16::from(p[i]) * (16 - fx) + u16::from(q[i]) * fx + 8) >> 4;
    }
}

#[inline(always)]
fn bil_w<const W: usize>(
    a: &[u8],
    sa: usize,
    b: &[u8],
    sb: usize,
    h: usize,
    fx: u32,
    fy: u32,
) -> u32 {
    let (fx, fy) = (fx as u16, fy as u16);
    let mut acc = 0;
    let mut top = [0u16; W];
    let mut bot = [0u16; W];
    across::<W>(b, fx, &mut top);
    for r in 0..h {
        across::<W>(&b[(r + 1) * sb..], fx, &mut bot);
        let mut v = [0u8; W];
        for i in 0..W {
            v[i] = ((top[i] * (16 - fy) + bot[i] * fy + 8) >> 4) as u8;
        }
        acc += row::<W>(&a[r * sa..], &v);
        top = bot;
    }
    acc
}

/// The SAD of the block at the start of `a` against the reference at the
/// start of `b` moved by `(fy, fx)` sixteenths down and across, bilinear
/// (libvpx's sub-sample search proxy). `b` must hold a row and a column
/// past the block.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn sad_bilinear(
    a: &[u8],
    sa: usize,
    b: &[u8],
    sb: usize,
    w: usize,
    h: usize,
    fy: u32,
    fx: u32,
) -> u32 {
    wide(|| match w {
        64 => bil_w::<64>(a, sa, b, sb, h, fx, fy),
        32 => bil_w::<32>(a, sa, b, sb, h, fx, fy),
        16 => bil_w::<16>(a, sa, b, sb, h, fx, fy),
        8 => bil_w::<8>(a, sa, b, sb, h, fx, fy),
        _ => {
            let mut acc = 0;
            for r in 0..h {
                for c in 0..w {
                    let p = |rr: usize, cc: usize| u32::from(b[rr * sb + cc]);
                    let t = (p(r, c) * (16 - fx) + p(r, c + 1) * fx + 8) >> 4;
                    let u = (p(r + 1, c) * (16 - fx) + p(r + 1, c + 1) * fx + 8) >> 4;
                    let v = (t * (16 - fy) + u * fy + 8) >> 4;
                    acc += v.abs_diff(u32::from(a[r * sa + c]));
                }
            }
            acc
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, seed: u32) -> Vec<u8> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (x >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn widths_agree_with_the_general_loop() {
        let a = data(80 * 80, 1);
        let b = data(80 * 80, 2);
        for w in [8, 16, 32, 64] {
            for h in [8, 16, 32, 64] {
                let mut want = 0u32;
                let mut want_sse = 0u64;
                for r in 0..h {
                    for c in 0..w {
                        let d = i32::from(a[r * 80 + c]) - i32::from(b[r * 79 + c]);
                        want += d.unsigned_abs();
                        want_sse += (d * d) as u64;
                    }
                }
                assert_eq!(sad(&a, 80, &b, 79, w, h), want);
                assert_eq!(sse(&a, 80, &b, 79, w, h), want_sse);
                for (fy, fx) in [(0, 0), (8, 0), (0, 4), (12, 8)] {
                    let mut want = 0;
                    for r in 0..h {
                        for c in 0..w {
                            let p = |rr: usize, cc: usize| u32::from(b[rr * 79 + cc]);
                            let t = (p(r, c) * (16 - fx) + p(r, c + 1) * fx + 8) >> 4;
                            let u = (p(r + 1, c) * (16 - fx) + p(r + 1, c + 1) * fx + 8) >> 4;
                            let v = (t * (16 - fy) + u * fy + 8) >> 4;
                            want += v.abs_diff(u32::from(a[r * 80 + c]));
                        }
                    }
                    assert_eq!(
                        sad_bilinear(&a, 80, &b, 79, w, h, fy, fx),
                        want,
                        "{w}x{h} {fy} {fx}"
                    );
                }
            }
        }
    }
}

//! The subpel filters for samples past 8 bits, as the specification
//! writes them (7.11.3.4). `InterRound0` is 3, or 5 at 12 bits;
//! `InterRound1` is 7 for a compound intermediate and 11, or 9 at 12 bits,
//! for final samples. With a fraction of 0 the filter is the identity (one
//! tap of 128). A compound intermediate is stored less 8192, which keeps
//! it inside 16 bits.

use super::Block16;

const BIAS: i32 = 8192;

/// `Round2(x, n)`.
fn round2(x: i32, n: i32) -> i32 {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

/// The filter row a pass uses: the identity when `plain`.
fn taps(f: &[i8; 8], plain: bool) -> [i32; 8] {
    if plain {
        [0, 0, 0, 128, 0, 0, 0, 0]
    } else {
        f.map(i32::from)
    }
}

/// The horizontal pass: `mid[r * w + c]` for the `h + 7` rows.
fn horizontal(b: &Block16<'_>, plain: bool) -> Vec<i32> {
    let f = taps(b.fh, plain);
    let mut mid = vec![0i32; (b.h + 7) * b.w];
    for r in 0..b.h + 7 {
        for c in 0..b.w {
            let at = b.start + r * b.src_stride + c;
            let sum: i32 = (0..8).map(|t| f[t] * i32::from(b.src[at + t])).sum();
            mid[r * b.w + c] = round2(sum, b.round0());
        }
    }
    mid
}

/// Both passes, `plain` naming the pass whose fraction is 0, rounded by
/// `round1`.
fn predict(b: &Block16<'_>, plain: (bool, bool), round1: i32) -> Vec<i32> {
    let mid = horizontal(b, plain.0);
    let f = taps(b.fv, plain.1);
    let mut out = vec![0i32; b.w * b.h];
    for r in 0..b.h {
        for c in 0..b.w {
            let sum: i32 = (0..8).map(|t| f[t] * mid[(r + t) * b.w + c]).sum();
            out[r * b.w + c] = round2(sum, round1);
        }
    }
    out
}

fn prep(dst: &mut [i16], dst_stride: usize, b: &Block16<'_>, plain: (bool, bool)) {
    let out = predict(b, plain, 7);
    for r in 0..b.h {
        for c in 0..b.w {
            dst[r * dst_stride + c] = (out[r * b.w + c] - BIAS) as i16;
        }
    }
}

fn put(dst: &mut [u16], dst_stride: usize, b: &Block16<'_>, plain: (bool, bool)) {
    let out = predict(b, plain, b.round1());
    for r in 0..b.h {
        for c in 0..b.w {
            dst[r * dst_stride + c] = out[r * b.w + c].clamp(0, i32::from(b.max())) as u16;
        }
    }
}

macro_rules! twin {
    ($name:ident, $dst:ty, $f:ident, $plain:expr) => {
        #[cfg_attr(
            all(
                target_arch = "aarch64",
                target_feature = "neon",
                target_feature = "dotprod",
                not(test)
            ),
            allow(dead_code)
        )]
        pub(super) fn $name(dst: &mut [$dst], dst_stride: usize, b: &Block16<'_>) {
            $f(dst, dst_stride, b, $plain);
        }
    };
}

twin!(prep_hv, i16, prep, (false, false));
twin!(put_hv, u16, put, (false, false));
twin!(prep_h, i16, prep, (false, true));
twin!(put_h, u16, put, (false, true));
twin!(prep_v, i16, prep, (true, false));
twin!(put_v, u16, put, (true, false));

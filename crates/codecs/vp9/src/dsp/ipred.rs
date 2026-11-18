//! Intra prediction of one transform block (clause 8.6.1), from edges
//! the caller has built: `above[0..2n]` (the row above and, for 4×4, the
//! row above-right), `left[0..n]` and the corner `tl`.

use super::pixel::Pixel;

/// Mode numbers of the bitstream.
pub const V: u8 = 1;
pub const H: u8 = 2;
pub const D45: u8 = 3;
pub const D135: u8 = 4;
pub const D117: u8 = 5;
pub const D153: u8 = 6;
pub const D207: u8 = 7;
pub const D63: u8 = 8;
pub const TM: u8 = 9;

/// The edges of one block.
pub struct Edges<P> {
    pub above: [P; 64],
    pub left: [P; 32],
    pub tl: P,
    pub have_above: bool,
    pub have_left: bool,
}

#[inline(always)]
fn r1<P: Pixel>(a: P, b: P) -> P {
    P::new((a.get() + b.get() + 1) >> 1)
}

#[inline(always)]
fn r2<P: Pixel>(a: P, b: P, c: P) -> P {
    P::new((a.get() + 2 * b.get() + c.get() + 2) >> 2)
}

/// Predicts an `n`×`n` block into `dst`.
#[inline]
pub fn predict<P: Pixel>(mode: u8, n: usize, dst: &mut [P], stride: usize, e: &Edges<P>, bd: u32) {
    let max = (1i32 << bd) - 1;
    let a = &e.above;
    let l = &e.left;
    let tl = e.tl;
    macro_rules! at {
        ($r:expr, $c:expr) => {
            dst[$r * stride + $c]
        };
    }
    match mode {
        V => {
            for r in 0..n {
                dst[r * stride..r * stride + n].copy_from_slice(&a[..n]);
            }
        }
        H => {
            for r in 0..n {
                dst[r * stride..r * stride + n].fill(l[r]);
            }
        }
        D45 => {
            for r in 0..n {
                for c in 0..n {
                    at!(r, c) = if r + c + 2 < 2 * n {
                        r2(a[r + c], a[r + c + 1], a[r + c + 2])
                    } else {
                        a[2 * n - 1]
                    };
                }
            }
        }
        D63 => {
            for r in 0..n {
                for c in 0..n {
                    let b = r / 2 + c;
                    at!(r, c) = if r & 1 == 1 {
                        r2(a[b], a[b + 1], a[b + 2])
                    } else {
                        r1(a[b], a[b + 1])
                    };
                }
            }
        }
        D207 => {
            // Work column by column in a small buffer, then copy.
            let mut p = [P::default(); 32 * 32];
            for c in 0..n {
                p[(n - 1) * n + c] = l[n - 1];
            }
            for r in 0..n - 1 {
                p[r * n] = r1(l[r], l[r + 1]);
            }
            for r in 0..n - 2 {
                p[r * n + 1] = r2(l[r], l[r + 1], l[r + 2]);
            }
            p[(n - 2) * n + 1] = r2(l[n - 2], l[n - 1], l[n - 1]);
            for r in (0..n - 1).rev() {
                for c in 2..n {
                    p[r * n + c] = p[(r + 1) * n + c - 2];
                }
            }
            for r in 0..n {
                dst[r * stride..r * stride + n].copy_from_slice(&p[r * n..r * n + n]);
            }
        }
        D117 => {
            let mut p = [P::default(); 32 * 32];
            for c in 0..n {
                let prev = if c == 0 { tl } else { a[c - 1] };
                p[c] = r1(prev, a[c]);
            }
            p[n] = r2(l[0], tl, a[0]);
            for c in 1..n {
                let back = if c == 1 { tl } else { a[c - 2] };
                p[n + c] = r2(back, a[c - 1], a[c]);
            }
            p[2 * n] = r2(tl, l[0], l[1]);
            for r in 3..n {
                p[r * n] = r2(l[r - 3], l[r - 2], l[r - 1]);
            }
            for r in 2..n {
                for c in 1..n {
                    p[r * n + c] = p[(r - 2) * n + c - 1];
                }
            }
            for r in 0..n {
                dst[r * stride..r * stride + n].copy_from_slice(&p[r * n..r * n + n]);
            }
        }
        D135 => {
            // The diagonal of edge samples: d[k] for k = 0 .. 2n - 1,
            // bottom-left to top-right; row r column c takes d[n - 1 - r + c].
            let mut d = [P::default(); 64];
            // left part, from the bottom up
            #[allow(clippy::needless_range_loop)]
            for k in 0..n {
                // sample at row n - 1 - k of the first column
                let r = n - 1 - k;
                d[k] = if r == 0 {
                    r2(l[0], tl, a[0])
                } else if r == 1 {
                    r2(tl, l[0], l[1])
                } else {
                    r2(l[r - 2], l[r - 1], l[r])
                };
            }
            for c in 1..n {
                let back = if c == 1 { tl } else { a[c - 2] };
                d[n - 1 + c] = r2(back, a[c - 1], a[c]);
            }
            for r in 0..n {
                let s = n - 1 - r;
                dst[r * stride..r * stride + n].copy_from_slice(&d[s..s + n]);
            }
        }
        D153 => {
            let mut p = [P::default(); 32 * 32];
            p[0] = r1(l[0], tl);
            for r in 1..n {
                p[r * n] = r1(l[r - 1], l[r]);
            }
            p[1] = r2(l[0], tl, a[0]);
            p[n + 1] = r2(tl, l[0], l[1]);
            for r in 2..n {
                p[r * n + 1] = r2(l[r - 2], l[r - 1], l[r]);
            }
            for c in 2..n {
                let x = if c == 2 { tl } else { a[c - 3] };
                let y = if c == 2 { a[0] } else { a[c - 2] };
                p[c] = r2(x, y, a[c - 1]);
            }
            for r in 1..n {
                for c in 2..n {
                    p[r * n + c] = p[(r - 1) * n + c - 2];
                }
            }
            for r in 0..n {
                dst[r * stride..r * stride + n].copy_from_slice(&p[r * n..r * n + n]);
            }
        }
        TM => {
            let t = tl.get();
            for r in 0..n {
                let lr = l[r].get() - t;
                let row = &mut dst[r * stride..r * stride + n];
                for (c, p) in row.iter_mut().enumerate() {
                    *p = P::clip(a[c].get() + lr, max);
                }
            }
        }
        _ => {
            let log2 = n.trailing_zeros();
            let sum = |s: &[P]| s[..n].iter().map(|p| p.get()).sum::<i32>();
            let v = match (e.have_left, e.have_above) {
                (true, true) => (sum(l) + sum(a) + n as i32) >> (log2 + 1),
                (true, false) => (sum(l) + (n as i32 >> 1)) >> log2,
                (false, true) => (sum(a) + (n as i32 >> 1)) >> log2,
                (false, false) => 1 << (bd - 1),
            };
            let v = P::new(v);
            for r in 0..n {
                dst[r * stride..r * stride + n].fill(v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Edges, predict};

    /// Against the predictors of `intra.rs`, which match libvpx.
    #[test]
    fn matches_the_scalar_predictors() {
        let mut seed = 99u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            ((seed >> 8) & 255) as u8
        };
        for tx in 0..4u8 {
            let n = 4usize << tx;
            for mode in 0..10u8 {
                for avail in 0..4 {
                    // A picture with the block at (n, n), edges around it.
                    let stride = 4 * n;
                    let mut pic: Vec<u8> = (0..stride * 4 * n).map(|_| rnd()).collect();
                    let (have_left, have_above) = (avail & 1 == 1, avail & 2 == 2);
                    let have_right = tx == 0;
                    let site = crate::intra::Site {
                        tx,
                        stride,
                        crop_w: stride,
                        crop_h: 4 * n,
                        x: n,
                        y: n,
                        have_left,
                        have_above,
                        have_right,
                    };
                    let mut want = pic.clone();
                    crate::intra::predict(mode, &mut want, site);
                    let mut e = Edges {
                        above: [127u8; 64],
                        left: [129u8; 32],
                        tl: 127,
                        have_above,
                        have_left,
                    };
                    if have_left {
                        for i in 0..n {
                            e.left[i] = pic[(n + i) * stride + n - 1];
                        }
                    }
                    if have_above {
                        let k = if have_right { 2 * n } else { n };
                        for i in 0..2 * n {
                            e.above[i] = pic[(n - 1) * stride + n + i.min(k - 1)];
                        }
                        e.tl = if have_left {
                            pic[(n - 1) * stride + n - 1]
                        } else {
                            129
                        };
                    }
                    let at = n * stride + n;
                    predict(mode, n, &mut pic[at..], stride, &e, 8);
                    assert_eq!(pic, want, "tx {tx} mode {mode} avail {avail}");
                }
            }
        }
    }
}

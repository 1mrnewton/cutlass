//! The kernel against the interpolation of clause 8.5.3.3.3, sample by
//! sample, on random samples of 10 and 12 bits.

use super::{Src, Taps, Tmp, put};

/// The luma and chroma filters (Tables 8-11 and 8-12).
const LUMA: [[i8; 8]; 4] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
];
const CHROMA: [[i8; 4]; 8] = [
    [0, 64, 0, 0],
    [-2, 58, 10, -2],
    [-4, 54, 16, -2],
    [-6, 46, 28, -4],
    [-4, 36, 36, -4],
    [-4, 28, 46, -6],
    [-2, 16, 54, -4],
    [-2, 10, 58, -2],
];

fn reference(
    src: &[u16],
    (at, stride): (usize, usize),
    (w, h): (usize, usize),
    (across, down): (Option<&[i8]>, Option<&[i8]>),
    t: usize,
    bit_depth: u32,
) -> Vec<i16> {
    let half = t / 2 - 1;
    let s = |x: usize, y: usize| i32::from(src[at + y * stride + x]);
    let shift1 = bit_depth - 8;
    let mut out = vec![0i16; w * h];
    for y in 0..h {
        for x in 0..w {
            let across_at = |yy: usize| -> i32 {
                match across {
                    Some(f) => {
                        (0..t).map(|k| i32::from(f[k]) * s(x + k, yy)).sum::<i32>() >> shift1
                    }
                    None => s(x + half, yy),
                }
            };
            let v = match (across, down) {
                (None, None) => s(x + half, y + half) << (14 - bit_depth),
                (Some(_), None) => across_at(y + half),
                (None, Some(f)) => {
                    (0..t)
                        .map(|k| i32::from(f[k]) * s(x + half, y + k))
                        .sum::<i32>()
                        >> shift1
                }
                (Some(_), Some(f)) => {
                    (0..t)
                        .map(|k| i32::from(f[k]) * across_at(y + k))
                        .sum::<i32>()
                        >> 6
                }
            };
            out[y * w + x] = v as i16;
        }
    }
    out
}

#[test]
fn put_matches_the_specification() {
    if !crate::hevc::mc::accelerated() {
        return;
    }
    let mut seed = 0x1016_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let (sw, sh) = (120usize, 80usize);
    let mut tmp = Tmp::default();
    for bit_depth in [10u32, 12] {
        let max = (1u64 << bit_depth) - 1;
        let noise: Vec<u16> = (0..sw * sh).map(|_| (next() % (max + 1)) as u16).collect();
        let edges: Vec<u16> = (0..sw * sh)
            .map(|_| if next() & 1 == 0 { 0 } else { max as u16 })
            .collect();
        for plane in [&noise, &edges] {
            for &(w, h) in &[
                (4, 4),
                (8, 8),
                (12, 16),
                (16, 4),
                (24, 32),
                (32, 8),
                (48, 64),
                (64, 64),
                (8, 1),
            ] {
                for t in [8usize, 4] {
                    for fx in 0..if t == 8 { 4 } else { 8 } {
                        for fy in 0..if t == 8 { 4 } else { 8 } {
                            let pick = |f: usize| -> [i8; 8] {
                                let mut a = [0i8; 8];
                                if t == 8 {
                                    a = LUMA[f];
                                } else {
                                    a[..4].copy_from_slice(&CHROMA[f]);
                                }
                                a
                            };
                            let (ax, ay) = (pick(fx), pick(fy));
                            let taps = Taps {
                                across: (fx > 0).then_some(ax),
                                down: (fy > 0).then_some(ay),
                                len: t,
                            };
                            let at = 3 * sw + 5;
                            let src = Src {
                                src: plane,
                                at,
                                stride: sw,
                            };
                            let stride = w + 3;
                            let mut got = vec![7i16; (h - 1) * stride + w];
                            assert!(
                                put(&mut got, stride, &src, (w, h), &taps, bit_depth, &mut tmp),
                                "{w}x{h} refused"
                            );
                            let want = reference(
                                plane,
                                (at, sw),
                                (w, h),
                                (
                                    taps.across.as_ref().map(|a| &a[..t]),
                                    taps.down.as_ref().map(|a| &a[..t]),
                                ),
                                t,
                                bit_depth,
                            );
                            for y in 0..h {
                                assert_eq!(
                                    &got[y * stride..y * stride + w],
                                    &want[y * w..y * w + w],
                                    "{bit_depth} bits {w}x{h} taps {t} fractions {fx},{fy} row {y}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn store_matches_the_default_weighting() {
    use super::store;
    if !crate::hevc::mc::accelerated() {
        return;
    }
    let mut seed = 7u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for bd in [9u32, 10, 12] {
        let max = (1i32 << bd) - 1;
        // Intermediates over their whole range, extremes included.
        let mut inter = || -> i16 {
            match next() % 8 {
                0 => i16::MIN,
                1 => i16::MAX,
                _ => (next() as i32 % 26000 - 10000) as i16,
            }
        };
        let a: Vec<i16> = (0..64 * 70).map(|_| inter()).collect();
        let b: Vec<i16> = (0..64 * 70).map(|_| inter()).collect();
        for w in (4..=64).step_by(4) {
            for h in [1usize, 2, 7, 64] {
                for two in [false, true] {
                    let (stride, ab) = (w + 3, 70);
                    let mut got = vec![0u16; (h - 1) * stride + w];
                    assert!(store(
                        &mut got,
                        stride,
                        (&a, two.then_some(&b[..])),
                        ab,
                        (w, h),
                        bd
                    ));
                    for y in 0..h {
                        for x in 0..w {
                            let (p, q) = (i32::from(a[y * ab + x]), i32::from(b[y * ab + x]));
                            let want = if two {
                                (p + q + (1 << (14 - bd))) >> (15 - bd)
                            } else {
                                (p + (1 << (13 - bd))) >> (14 - bd)
                            };
                            assert_eq!(
                                i32::from(got[y * stride + x]),
                                want.clamp(0, max),
                                "bd {bd} w {w} h {h} two {two} at {x},{y}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn predict_to_samples_matches_put_then_store() {
    use super::{Dst, predict, store};
    if !crate::hevc::mc::accelerated() {
        return;
    }
    let mut seed = 0xfeed_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut tmp = Tmp::default();
    for bd in [10u32, 12] {
        let max = (1u64 << bd) - 1;
        let (sw, sh) = (64 + 7 + 24, 64 + 7);
        let src: Vec<u16> = (0..sw * sh).map(|_| (next() & max) as u16).collect();
        let s = Src {
            src: &src,
            at: 0,
            stride: sw,
        };
        for t in [8usize, 4] {
            for w in [4usize, 8, 12, 16, 24, 32, 48, 64] {
                for h in [1usize, 4, 7, 64] {
                    for (fx, fy) in [(0, 0), (1, 0), (0, 3), (2, 1)] {
                        let tap = |f: usize| -> [i8; 8] {
                            if t == 8 {
                                LUMA[f]
                            } else {
                                let mut c = [0i8; 8];
                                c[..4].copy_from_slice(&CHROMA[2 * f + 1]);
                                c
                            }
                        };
                        let taps = Taps {
                            across: (fx != 0).then(|| tap(fx)),
                            down: (fy != 0).then(|| tap(fy)),
                            len: t,
                        };
                        let other: Vec<i16> =
                            (0..w * h).map(|_| (next() % 20000) as i16 - 4000).collect();
                        let mut mid = vec![0i16; w * h];
                        assert!(put(&mut mid, w, &s, (w, h), &taps, bd, &mut tmp));
                        let stride = w + 5;
                        for two in [false, true] {
                            let mut want = vec![0u16; (h - 1) * stride + w];
                            let pair = if two {
                                (&other[..], Some(&mid[..]))
                            } else {
                                (&mid[..], None)
                            };
                            assert!(store(&mut want, stride, pair, w, (w, h), bd));
                            let mut got = vec![0u16; (h - 1) * stride + w];
                            let dst = if two {
                                Dst::Bi(&mut got, &other)
                            } else {
                                Dst::Uni(&mut got)
                            };
                            assert!(predict(dst, stride, &s, (w, h), &taps, bd, &mut tmp));
                            assert_eq!(
                                got, want,
                                "bd {bd} t {t} w {w} h {h} f {fx},{fy} two {two}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn explicit_weighting_matches_the_clause() {
    use super::{Dst, predict};
    use crate::hevc::mc::Weights;
    if !crate::hevc::mc::accelerated() {
        return;
    }
    let mut seed = 0x77ee_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut tmp = Tmp::default();
    for bd in [10u32, 12] {
        let top = (1i32 << bd) - 1;
        let (sw, sh) = (64 + 7 + 24, 64 + 7);
        let src: Vec<u16> = (0..sw * sh).map(|_| (next() as i32 & top) as u16).collect();
        let s = Src {
            src: &src,
            at: 0,
            stride: sw,
        };
        for (w, h) in [
            (4usize, 4usize),
            (8, 2),
            (12, 16),
            (16, 8),
            (24, 4),
            (64, 64),
        ] {
            for (fx, fy) in [(0usize, 0usize), (1, 0), (0, 2), (3, 1)] {
                let taps = Taps {
                    across: (fx != 0).then(|| LUMA[fx]),
                    down: (fy != 0).then(|| LUMA[fy]),
                    len: 8,
                };
                let mut mid = vec![0i16; w * h];
                assert!(put(&mut mid, w, &s, (w, h), &taps, bd, &mut tmp));
                let other: Vec<i16> = (0..w * h).map(|_| (next() % 20000) as i16 - 4000).collect();
                for denom in [0u32, 3, 7] {
                    let scale = 1 << (bd.min(10) - 8);
                    let wt = Weights {
                        w0: (next() % 256) as i16 - 128,
                        w1: (next() % 256) as i16 - 128,
                        o0: ((next() % 256) as i32 - 128) * scale,
                        o1: ((next() % 256) as i32 - 128) * scale,
                        log2wd: denom + 14 - bd,
                    };
                    let l = wt.log2wd;
                    for two in [false, true] {
                        let mut got = vec![0u16; w * h];
                        let dst = if two {
                            Dst::BiW(&mut got, &other, wt)
                        } else {
                            Dst::UniW(&mut got, wt)
                        };
                        assert!(predict(dst, w, &s, (w, h), &taps, bd, &mut tmp));
                        for i in 0..w * h {
                            let p = i32::from(mid[i]);
                            let want = if two {
                                let q = i32::from(other[i]);
                                (q * i32::from(wt.w0)
                                    + p * i32::from(wt.w1)
                                    + ((wt.o0 + wt.o1 + 1) << l))
                                    >> (l + 1)
                            } else {
                                ((p * i32::from(wt.w0) + (1 << (l - 1))) >> l) + wt.o0
                            };
                            assert_eq!(
                                i32::from(got[i]),
                                want.clamp(0, top),
                                "bd {bd} {w}x{h} f {fx},{fy} denom {denom} two {two} at {i}"
                            );
                        }
                    }
                }
            }
        }
    }
}

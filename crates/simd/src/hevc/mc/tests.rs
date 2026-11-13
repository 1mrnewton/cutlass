//! The kernels against the filters as clause 8.5.3.3.3 writes them.

use super::{Dst, Src, Taps, accelerated, bi, predict, put, uni};

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

fn noise(n: usize, mut seed: u64) -> Vec<u8> {
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as u8
        })
        .collect()
}

/// The intermediates as the specification computes them.
fn reference(src: &Src<'_>, (w, h): (usize, usize), taps: &Taps) -> Vec<i16> {
    let t = taps.len;
    let h0 = t / 2 - 1;
    let at = |r: usize, c: usize| i32::from(src.src[src.at + r * src.stride + c]);
    let sum = |f: &[i8; 8], v: &dyn Fn(usize) -> i32| -> i32 {
        (0..t).map(|k| i32::from(f[k]) * v(k)).sum()
    };
    let mut out = vec![0i16; w * h];
    for y in 0..h {
        for x in 0..w {
            let v = match (taps.across, taps.down) {
                (None, None) => at(y + h0, x + h0) << 6,
                (Some(fh), None) => sum(&fh, &|k| at(y + h0, x + k)),
                (None, Some(fv)) => sum(&fv, &|k| at(y + k, x + h0)),
                (Some(fh), Some(fv)) => sum(&fv, &|k| sum(&fh, &|m| at(y + k, x + m))) >> 6,
            };
            out[y * w + x] = v as i16;
        }
    }
    out
}

#[test]
fn put_matches_the_specification() {
    if !accelerated() {
        return;
    }
    let (sw, sh) = (96usize, 80usize);
    let samples = noise(sw * sh, 0x1234_5678);
    let edges = noise(sw * sh, 1)
        .iter()
        .map(|&v| if v & 1 == 0 { 0 } else { 255 })
        .collect::<Vec<u8>>();
    let mut tmp = vec![0i16; 71 * 64];
    for plane in [&samples, &edges] {
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
            for len in [8usize, 4] {
                for f in 0..if len == 8 { 4 } else { 8 } {
                    for g in 0..if len == 8 { 4 } else { 8 } {
                        let pad = |c: [i8; 4]| [c[0], c[1], c[2], c[3], 0, 0, 0, 0];
                        let tap = |i: usize| if len == 8 { LUMA[i] } else { pad(CHROMA[i]) };
                        let taps = Taps {
                            across: (f > 0).then(|| tap(f)),
                            down: (g > 0).then(|| tap(g)),
                            len,
                        };
                        let src = Src {
                            src: plane,
                            at: 3 * sw + 5,
                            stride: sw,
                        };
                        let want = reference(&src, (w, h), &taps);
                        let stride = w + 3;
                        let mut got = vec![0i16; h * stride];
                        assert!(put(&mut got, stride, &src, (w, h), &taps, &mut tmp));
                        for y in 0..h {
                            assert_eq!(
                                &got[y * stride..y * stride + w],
                                &want[y * w..y * w + w],
                                "{w}x{h} taps {len} f {f} g {g} row {y}"
                            );
                        }
                        // Straight to samples, alone and averaged.
                        let other: Vec<i16> = want.iter().rev().map(|&v| v / 3 + 200).collect();
                        let mut px = vec![0u8; h * stride];
                        assert!(predict(
                            Dst::Uni(&mut px),
                            stride,
                            &src,
                            (w, h),
                            &taps,
                            &mut tmp
                        ));
                        for y in 0..h {
                            for x in 0..w {
                                let v = ((i32::from(want[y * w + x]) + 32) >> 6).clamp(0, 255);
                                assert_eq!(i32::from(px[y * stride + x]), v, "uni {w}x{h}");
                            }
                        }
                        assert!(predict(
                            Dst::Bi(&mut px, &other),
                            stride,
                            &src,
                            (w, h),
                            &taps,
                            &mut tmp
                        ));
                        for y in 0..h {
                            for x in 0..w {
                                let s = i32::from(want[y * w + x]) + i32::from(other[y * w + x]);
                                let v = ((s + 64) >> 7).clamp(0, 255);
                                assert_eq!(i32::from(px[y * stride + x]), v, "bi {w}x{h}");
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn weighting_matches_the_specification() {
    if !accelerated() {
        return;
    }
    let raw = noise(64 * 64 * 4, 7);
    let a: Vec<i16> = raw
        .chunks(2)
        .map(|p| i16::from_le_bytes([p[0], p[1]]) >> 1)
        .collect();
    let b: Vec<i16> = a.iter().rev().map(|&v| v.wrapping_mul(3) >> 2).collect();
    for &(w, h) in &[(4, 4), (8, 2), (12, 3), (16, 16), (24, 5), (64, 64)] {
        let stride = w + 5;
        let mut got = vec![0u8; h * stride];
        assert!(uni(&mut got, stride, &a, w, (w, h)));
        for y in 0..h {
            for x in 0..w {
                let want = ((i32::from(a[y * w + x]) + 32) >> 6).clamp(0, 255);
                assert_eq!(i32::from(got[y * stride + x]), want, "uni {w}x{h}");
            }
        }
        assert!(bi(&mut got, stride, (&a, &b), w, (w, h)));
        for y in 0..h {
            for x in 0..w {
                let s = i32::from(a[y * w + x]) + i32::from(b[y * w + x]);
                let want = ((s + 64) >> 7).clamp(0, 255);
                assert_eq!(i32::from(got[y * stride + x]), want, "bi {w}x{h}");
            }
        }
    }
}

#[test]
fn blocks_too_close_to_the_end_are_refused() {
    let samples = vec![0u8; 64];
    let src = Src {
        src: &samples,
        at: 0,
        stride: 8,
    };
    let taps = Taps {
        across: None,
        down: None,
        len: 8,
    };
    let mut out = vec![0i16; 64];
    let mut tmp = vec![0i16; 15 * 8];
    assert!(!put(&mut out, 8, &src, (8, 8), &taps, &mut tmp));
}

//! The kernels against the filters as clause 8.5.3.3.3 writes them.

use super::{Dst, Src, Taps, Weights, accelerated, bi, predict, put, uni};

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
    let at = |r: usize, c: usize| i32::from(src.src[src.row(r) + c]);
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
    let mut tmp = super::Tmp::default();
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
                        // Rows held within the plane, the window
                        // reaching above it or below it in turn.
                        let clamp = match (f + g) % 3 {
                            0 => None,
                            1 => Some((-5, sh)),
                            _ => Some((sh as isize - h as isize - 3, sh)),
                        };
                        let src = Src {
                            clamp,
                            src: plane,
                            at: if clamp.is_some() { 5 } else { 3 * sw + 5 },
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
                        // Explicit weights.
                        let wt = Weights {
                            w0: 77,
                            w1: -13,
                            o0: -40,
                            o1: 21,
                            log2wd: 6 + (f % 3) as u32,
                        };
                        let dst = Dst::UniW(&mut px, wt);
                        assert!(predict(dst, stride, &src, (w, h), &taps, &mut tmp));
                        for y in 0..h {
                            for x in 0..w {
                                let p = i32::from(want[y * w + x]);
                                let r = 1 << (wt.log2wd - 1);
                                let v = (((p * i32::from(wt.w0) + r) >> wt.log2wd) + wt.o0)
                                    .clamp(0, 255);
                                assert_eq!(i32::from(px[y * stride + x]), v, "uniw {w}x{h}");
                            }
                        }
                        let dst = Dst::BiW(&mut px, &other, wt);
                        assert!(predict(dst, stride, &src, (w, h), &taps, &mut tmp));
                        for y in 0..h {
                            for x in 0..w {
                                let a = i32::from(other[y * w + x]);
                                let b = i32::from(want[y * w + x]);
                                let s = a * i32::from(wt.w0)
                                    + b * i32::from(wt.w1)
                                    + ((wt.o0 + wt.o1 + 1) << wt.log2wd);
                                let v = (s >> (wt.log2wd + 1)).clamp(0, 255);
                                assert_eq!(i32::from(px[y * stride + x]), v, "biw {w}x{h}");
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
        clamp: None,
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
    let mut tmp = super::Tmp::default();
    assert!(!put(&mut out, 8, &src, (8, 8), &taps, &mut tmp));
}

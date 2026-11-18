use super::{halves4, halves4_scalar, sad, sad_scalar, sad4, sse, sse_scalar};

#[test]
fn matches_the_scalar_twins() {
    let mut seed = 7u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        seed >> 8
    };
    let (sa, sb) = (80, 77);
    let mut ran = 0;
    for case in 0..2000 {
        // Extremes first (one block all 0, the other all 255, or a
        // checkerboard of both), then random samples.
        let mut gen_ = |n: usize, k: u8| -> Vec<u8> {
            (0..n)
                .map(|i| match case % 5 {
                    0 => [0, 255][k as usize],
                    1 => [0, 255][(i + k as usize) % 2],
                    _ => rnd() as u8,
                })
                .collect()
        };
        // Each block ends exactly where its last row does.
        let w = [8, 16, 32, 64][(case / 5) % 4];
        let h = [8, 16, 32, 64, 1, 2, 3, 5, 7][(case / 20) % 9];
        let a = gen_((h - 1) * sa + w, 0);
        let bs: Vec<Vec<u8>> = (0..4)
            .map(|k| gen_((h - 1) * sb + w, (k % 2) as u8 ^ 1))
            .collect();
        let want: Vec<u32> = bs.iter().map(|b| sad_scalar(&a, sa, b, sb, w, h)).collect();
        let want_sse = sse_scalar(&a, sa, &bs[0], sb, w, h);
        if let Some(got) = sad(&a, sa, &bs[0], sb, w, h) {
            ran += 1;
            assert_eq!(got, want[0], "sad case {case} {w}x{h}");
        }
        if let Some(got) = sse(&a, sa, &bs[0], sb, w, h) {
            assert_eq!(got, want_sse, "sse case {case} {w}x{h}");
        }
        if let Some(got) = sad4(&a, sa, [&bs[0], &bs[1], &bs[2], &bs[3]], sb, w, h) {
            assert_eq!(got.to_vec(), want, "sad4 case {case} {w}x{h}");
        }
    }
    if super::super::avx2() {
        assert_eq!(ran, 2000);
    }
}

#[test]
fn halves_match_the_scalar_twin() {
    let mut seed = 11u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        (seed >> 8) as u8
    };
    let (sa, sb) = (48, 90);
    let mut ran = 0;
    for case in 0..200 {
        let h = 1 + case % 16;
        let mut gen_ = |n: usize, k: usize| -> Vec<u8> {
            (0..n)
                .map(|i| match case % 4 {
                    0 => [0, 255][k % 2],
                    1 => [0, 255][(i + k) % 2],
                    _ => rnd(),
                })
                .collect()
        };
        // Each block ends exactly where its last row does.
        let a = gen_((h - 1) * sa + 16, 0);
        let bs: Vec<Vec<u8>> = (0..4).map(|k| gen_((h - 1) * sb + 16, k + 1)).collect();
        let b = [&bs[0][..], &bs[1][..], &bs[2][..], &bs[3][..]];
        let want = halves4_scalar(&a, sa, b, sb, h);
        if let Some(got) = halves4(&a, sa, b, sb, h) {
            ran += 1;
            assert_eq!(got, want, "case {case} h {h}");
        }
    }
    if super::super::avx2() {
        assert_eq!(ran, 200);
    }
}

#[test]
fn refuses_what_does_not_fit() {
    let a = vec![0u8; 64 * 63 + 63];
    assert!(sad(&a, 64, &a, 64, 64, 64).is_none());
    assert!(sse(&a, 64, &a, 64, 12, 4).is_none());
    assert!(sad(&a, 64, &a, 64, 4, 4).is_none());
}

#[test]
fn satd_matches_the_scalar_twin() {
    let mut seed = 5u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        (seed >> 8) as u8
    };
    let (sa, sb) = (72, 64);
    let mut ran = 0;
    for case in 0..500 {
        let n = [4, 8, 16, 32, 64][case % 5];
        // Extremes first (all 0 against all 255, a checkerboard of both),
        // then random samples; each block ends where its last row does.
        let mut gen_ = |len: usize, k: usize| -> Vec<u8> {
            (0..len)
                .map(|i| match (case / 5) % 4 {
                    0 => [0, 255][k],
                    1 => [0, 255][(i + k) % 2],
                    _ => rnd(),
                })
                .collect()
        };
        let a = gen_((n - 1) * sa + n, 0);
        let b = gen_((n - 1) * sb + n, 1);
        let want = super::satd_scalar(&a, sa, &b, sb, n, n);
        if let Some(got) = super::satd(&a, sa, &b, sb, n) {
            ran += 1;
            assert_eq!(got, want, "satd case {case} {n}x{n}");
        }
    }
    if super::super::avx2() {
        assert_eq!(ran, 500);
    }
}

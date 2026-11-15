use super::{sad, sad_scalar, sad4, sse, sse_scalar};

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
fn refuses_what_does_not_fit() {
    let a = vec![0u8; 64 * 63 + 63];
    assert!(sad(&a, 64, &a, 64, 64, 64).is_none());
    assert!(sse(&a, 64, &a, 64, 12, 4).is_none());
    assert!(sad(&a, 64, &a, 64, 4, 4).is_none());
}

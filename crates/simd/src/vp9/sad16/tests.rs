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
    for case in 0..1500 {
        let bd = if case % 2 == 0 { 10 } else { 12 };
        let max = (1u16 << bd) - 1;
        // Extremes first (one block all 0, the other all max, or a
        // checkerboard of both), then random samples.
        let mut gen_ = |n: usize, k: u16| -> Vec<u16> {
            (0..n)
                .map(|i| match case % 5 {
                    0 => [0, max][k as usize],
                    1 => [0, max][(i + k as usize) % 2],
                    _ => (rnd() as u16) & max,
                })
                .collect()
        };
        let a = gen_(sa * 64 + 64, 0);
        let bs: Vec<Vec<u16>> = (0..4)
            .map(|k| gen_(sb * 64 + 64, (k % 2) as u16 ^ 1))
            .collect();
        let w = [4, 8, 16, 32, 64][(rnd() % 5) as usize];
        let h = [4, 8, 16, 32, 64, 1, 3][(rnd() % 7) as usize];
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
        assert_eq!(ran, 1500);
    }
}

#[test]
fn refuses_what_does_not_fit() {
    let a = vec![0u16; 64 * 63 + 63];
    assert!(sad(&a, 64, &a, 64, 64, 64).is_none());
    assert!(sse(&a, 64, &a, 64, 12, 4).is_none());
}

use super::{Block, Filter, predict, predict_scalar, predict_with};

const FILTERS: [[i16; 8]; 4] = [
    [-1, 6, -19, 78, 78, -19, 6, -1],
    [-3, -1, 32, 64, 38, 1, -3, 0],
    [-3, 7, -17, 119, 28, -11, 5, -1],
    [0, 0, 0, 120, 8, 0, 0, 0],
];

#[test]
fn matches_the_scalar_twin() {
    let mut seed = 1u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        seed >> 8
    };
    let stride = 100;
    let mut ran_count = 0;
    let cases = 4000;
    for case in 0..cases {
        let bd = if case % 2 == 0 { 10 } else { 12 };
        let max = (1u16 << bd) - 1;
        // Extremes (alternating 0 and max, all max), then random.
        let src: Vec<u16> = (0..stride * 90)
            .map(|i| match case % 4 {
                0 => [0, max][i % 2],
                1 => [max, 0, 0, max][(i / 3) % 4],
                _ => (rnd() as u16) & max,
            })
            .collect();
        let w = [4, 8, 16, 32, 64][case % 5];
        let h = [4, 8, 16, 32, 64, 2, 1, 3, 5][(rnd() % 9) as usize];
        let pick = |r: u32| (!r.is_multiple_of(5)).then(|| Filter::new(FILTERS[(r % 4) as usize]));
        let fx = pick(rnd());
        let fy = pick(rnd());
        let avg = rnd() % 2 == 0;
        let at = 4 * stride + 4;
        let init: Vec<u16> = (0..64 * 64).map(|_| (rnd() as u16) & max).collect();
        let mut want = init.clone();
        predict_scalar(Block {
            dst: &mut want,
            dst_stride: 64,
            src: &src,
            at,
            src_stride: stride,
            w,
            h,
            fx: fx.as_ref(),
            fy: fy.as_ref(),
            avg,
            max,
        });
        // The AVX2 kernels, and the AVX-VNNI ones where the CPU has them.
        for vnni in [false, true] {
            let mut got = init.clone();
            let ran = predict_with(
                Block {
                    dst: &mut got,
                    dst_stride: 64,
                    src: &src,
                    at,
                    src_stride: stride,
                    w,
                    h,
                    fx: fx.as_ref(),
                    fy: fy.as_ref(),
                    avg,
                    max,
                },
                vnni,
            );
            if !ran {
                continue;
            }
            ran_count += usize::from(!vnni);
            assert_eq!(
                got, want,
                "case {case} w {w} h {h} {fx:?} {fy:?} avg {avg} vnni {vnni}"
            );
        }
    }
    if super::super::avx2() {
        assert_eq!(ran_count, cases);
    }
}

/// A block whose reads would pass the end of `src` is left to the caller.
#[test]
fn refuses_what_does_not_fit() {
    let src = vec![0u16; 64 * 20];
    let mut dst = vec![0u16; 64 * 16];
    let f = Filter::new(FILTERS[0]);
    let b = Block {
        dst: &mut dst,
        dst_stride: 64,
        src: &src,
        at: 64 * 10,
        src_stride: 64,
        w: 16,
        h: 16,
        fx: Some(&f),
        fy: Some(&f),
        avg: false,
        max: 1023,
    };
    assert!(!predict(b));
}

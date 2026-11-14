use super::{Block, predict, predict_scalar};

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
    for case in 0..2000 {
        let src: Vec<u8> = (0..stride * 90)
            .map(|i| if case % 3 == 0 { [0u8, 255][i % 2] } else { (rnd() & 255) as u8 })
            .collect();
        let w = [4, 8, 16, 32, 64][case % 5];
        let h = [4, 8, 16, 32, 64, 2, 1][(rnd() % 7) as usize].min(64);
        let pick = |r: u32| (r % 5 != 0).then(|| FILTERS[(r % 4) as usize]);
        let fx = pick(rnd());
        let fy = pick(rnd());
        let avg = rnd() % 2 == 0;
        let at = 4 * stride + 4;
        let init: Vec<u8> = (0..64 * 64).map(|_| (rnd() & 255) as u8).collect();
        let mut want = init.clone();
        predict_scalar(Block { dst: &mut want, dst_stride: 64, src: &src, at, src_stride: stride, w, h, fx, fy, avg });
        let mut got = init.clone();
        let ran = predict(Block { dst: &mut got, dst_stride: 64, src: &src, at, src_stride: stride, w, h, fx, fy, avg });
        if !ran {
            continue;
        }
        ran_count += 1;
        assert_eq!(got, want, "case {case} w {w} h {h} {fx:?} {fy:?} avg {avg}");
    }
    if super::super::avx2() {
        assert_eq!(ran_count, 2000);
    }
}

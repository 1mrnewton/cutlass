use super::{Rounding, scalar, wiener_h, wiener_v_u8};

/// A small fixed-seed generator: the same rows on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    fn between(&mut self, low: i32, high: i32) -> i32 {
        low + (self.next() % (high - low + 1) as u32) as i32
    }
}

/// The first four taps of a Wiener row from three coefficients inside the
/// ranges the bitstream codes (5.11.57): the centre makes the row add up
/// to 128.
fn taps(rng: &mut Rng) -> [i16; 4] {
    let c = [
        rng.between(-5, 10),
        rng.between(-23, 8),
        rng.between(-17, 46),
    ];
    [
        c[0] as i16,
        c[1] as i16,
        c[2] as i16,
        (128 - 2 * (c[0] + c[1] + c[2])) as i16,
    ]
}

/// The rows of taps a test runs: the identity, the largest and smallest
/// centre, and random ones.
fn tap_rows(rng: &mut Rng) -> Vec<[i16; 4]> {
    let mut rows = vec![[0, 0, 0, 128], [10, 8, 46, 0], [-5, -23, -17, 218]];
    rows.extend((0..9).map(|_| taps(rng)));
    rows
}

#[test]
fn the_horizontal_pass_matches_the_specification_at_every_depth() {
    for bd in [8u32, 10, 12] {
        let r = Rounding::of(bd);
        let mut rng = Rng(0x57_1E4E + u64::from(bd));
        let max = i32::from(r.max);
        for w in [8usize, 9, 15, 16, 17, 31, 64, 100, 256] {
            for (round, f) in tap_rows(&mut rng).iter().enumerate() {
                // Flat, extreme and noisy rows; one sample past the last
                // one read.
                let src: Vec<i16> = (0..w + 7)
                    .map(|i| match round % 4 {
                        0 => max as i16,
                        1 => 0,
                        2 => rng.between(0, max) as i16,
                        _ => {
                            if (i / 5) % 2 == 0 {
                                max as i16
                            } else {
                                0
                            }
                        }
                    })
                    .collect();
                let mut got = vec![7i16; w];
                let mut want = vec![7i16; w];
                assert!(wiener_h(&mut got, &src, f, &r));
                scalar::horizontal(&mut want, &src, f, &r);
                assert_eq!(got, want, "{bd}-bit width {w} taps {f:?}");
            }
        }
    }
}

#[test]
fn the_vertical_pass_matches_the_specification() {
    {
        let bd = 8u32;
        let r = Rounding::of(bd);
        let mut rng = Rng(0x0BAD_5EED + u64::from(bd));
        for w in [8usize, 9, 15, 16, 17, 31, 64, 100, 256] {
            for (round, f) in tap_rows(&mut rng).iter().enumerate() {
                // Intermediates over the whole range the horizontal pass
                // clamps to, flat at each end and noisy.
                let rows: Vec<Vec<i16>> = (0..7)
                    .map(|t| {
                        (0..w)
                            .map(|_| match (round + t) % 4 {
                                0 => r.high,
                                1 => r.low,
                                _ => rng.between(i32::from(r.low), i32::from(r.high)) as i16,
                            })
                            .collect()
                    })
                    .collect();
                let rows: [&[i16]; 7] = core::array::from_fn(|t| rows[t].as_slice());
                let want: Vec<u16> = (0..w).map(|c| scalar::vertical(&rows, c, f, &r)).collect();
                let mut got = vec![9u8; w];
                assert!(wiener_v_u8(&mut got, &rows, f, &r));
                let got: Vec<u16> = got.iter().map(|&v| u16::from(v)).collect();
                assert_eq!(got, want, "width {w} taps {f:?}");
                assert!(want.iter().all(|&v| v <= r.max));
            }
        }
    }
}

#[test]
fn the_identity_row_returns_the_sample() {
    // One tap of 128: the horizontal pass scales a sample by
    // 128 >> InterRound0 and the vertical pass scales it back.
    let r = Rounding::of(8);
    let src = vec![200i16; 16 + 7];
    let mut mid = vec![0i16; 16];
    assert!(wiener_h(&mut mid, &src, &[0, 0, 0, 128], &r));
    assert!(mid.iter().all(|&v| v == 200 << 4));
    let rows: [&[i16]; 7] = [&mid; 7];
    let mut out = vec![0u8; 16];
    assert!(wiener_v_u8(&mut out, &rows, &[0, 0, 0, 128], &r));
    assert!(out.iter().all(|&v| v == 200));
}

#[test]
fn the_rounding_is_the_one_of_the_specification() {
    let eight = Rounding::of(8);
    assert_eq!((eight.round0, eight.round1), (3, 11));
    assert_eq!((eight.low, eight.high, eight.max), (-2048, 6143, 255));
    let ten = Rounding::of(10);
    assert_eq!((ten.round0, ten.round1), (3, 11));
    assert_eq!((ten.low, ten.high, ten.max), (-8192, 24575, 1023));
    let twelve = Rounding::of(12);
    assert_eq!((twelve.round0, twelve.round1), (5, 9));
    assert_eq!((twelve.low, twelve.high, twelve.max), (-8192, 24575, 4095));
}

#[test]
fn a_row_the_kernel_does_not_take_is_left_alone() {
    let r = Rounding::of(8);
    let f = [1, 2, 3, 116];
    let src = vec![100i16; 64];
    // Fewer than eight outputs, and a source that ends before the sample
    // past the last one read.
    let mut short = vec![7i16; 7];
    assert!(!wiener_h(&mut short, &src, &f, &r));
    let mut out = vec![7i16; 16];
    assert!(!wiener_h(&mut out, &src[..22], &f, &r));
    assert!(out.iter().all(|&v| v == 7));
    assert!(wiener_h(&mut out, &src[..23], &f, &r));
    // A row of intermediates shorter than the output.
    let mid = vec![50i16; 16];
    let cut = vec![50i16; 15];
    let mut rows: [&[i16]; 7] = [&mid; 7];
    rows[4] = &cut;
    let mut bytes = vec![7u8; 16];
    assert!(!wiener_v_u8(&mut bytes, &rows, &f, &r));
    assert!(bytes.iter().all(|&v| v == 7));
    let mut narrow = vec![7u8; 4];
    assert!(!wiener_v_u8(&mut narrow, &[&mid; 7], &f, &r));
}

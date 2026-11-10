//! Inter prediction samples, clause 8.4.2.2.
//!
//! A luma half sample is the six-tap filter (1, −5, 20, 20, −5, 1) over six
//! full samples in a line. Along a row the six inputs are six windows of
//! that row, each one sample further right; down a column they are six rows.

pub use crate::mc_block::{BiWeight, Block};

/// The rounded average of two predictions of `size` (width, height), whose
/// rows are `size.0` apart, into `dst` with rows `dst_stride` apart.
/// `false` means a block does not fit its slice and nothing was done.
pub fn avg_block(
    dst: &mut [u8],
    dst_stride: usize,
    a: &[u8],
    b: &[u8],
    size: (usize, usize),
) -> bool {
    kernel!(avg_block(dst, dst_stride, a, b, size))
}

/// Two predictions mixed by `weight`, laid out as for [`avg_block`].
/// `false` also when the weights are outside the kernels' range.
pub fn weight_block(
    dst: &mut [u8],
    dst_stride: usize,
    a: &[u8],
    b: &[u8],
    size: (usize, usize),
    weight: BiWeight,
) -> bool {
    kernel!(weight_block(dst, dst_stride, a, b, size, weight))
}

/// A luma block at quarter-sample offsets `frac` (each 0 to 3), written to
/// `dst`, whose rows are `dst_stride` apart. Covers blocks 8 or 16 wide and
/// at most 16 high; `false` means nothing was done. The source must hold
/// the two samples before the block and the three after it, in both
/// directions.
pub fn luma_block(dst: &mut [u8], dst_stride: usize, block: Block<'_>, frac: (u8, u8)) -> bool {
    kernel!(luma_block(dst, dst_stride, block, frac))
}

/// A chroma block at eighth-sample offsets `frac` (each 0 to 7). Covers
/// blocks 4 or 8 wide; `false` means nothing was done. The source must
/// hold one more sample after the block's last column and row.
pub fn chroma_block(dst: &mut [u8], dst_stride: usize, block: Block<'_>, frac: (u8, u8)) -> bool {
    kernel!(chroma_block(dst, dst_stride, block, frac))
}

/// The rounded average of `a` and `b` into `dst`. `false` means the inputs
/// are shorter than `dst` and nothing was done.
pub fn avg_row(dst: &mut [u8], a: &[u8], b: &[u8]) -> bool {
    kernel!(avg_row(dst, a, b))
}

/// Half samples from six lines of full samples: the filter, rounded and
/// clipped to a sample.
pub fn half16(taps: [&[u8; 16]; 6]) -> [u8; 16] {
    kernel!(half16(taps))
}

/// [`half16`] for eight samples.
pub fn half8(taps: [&[u8; 8]; 6]) -> [u8; 8] {
    kernel!(half8(taps))
}

/// The filter's sum with no rounding, the first pass of a centre sample.
pub fn raw16(taps: [&[u8; 16]; 6]) -> [i16; 16] {
    kernel!(raw16(taps))
}

/// [`raw16`] for eight samples.
pub fn raw8(taps: [&[u8; 8]; 6]) -> [i16; 8] {
    kernel!(raw8(taps))
}

/// Centre samples (the half sample in both directions) from six lines of
/// [`raw16`] sums: the filter again, rounded and clipped.
pub fn centre16(taps: [&[i16; 16]; 6]) -> [u8; 16] {
    kernel!(centre16(taps))
}

/// [`centre16`] for eight samples.
pub fn centre8(taps: [&[i16; 8]; 6]) -> [u8; 8] {
    kernel!(centre8(taps))
}

/// The rounded average of two lines, which is a quarter sample and also a
/// default-weight prediction from two references.
pub fn avg16(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    kernel!(avg16(a, b))
}

/// [`avg16`] for eight samples.
pub fn avg8(a: &[u8; 8], b: &[u8; 8]) -> [u8; 8] {
    kernel!(avg8(a, b))
}

/// Eight chroma samples, clause 8.4.2.2.2, from the nine samples above and
/// the nine below. `xf` and `yf` are the eighth-sample offsets, 0 to 7.
pub fn chroma8(top: &[u8; 9], bottom: &[u8; 9], xf: u8, yf: u8) -> [u8; 8] {
    kernel!(chroma8(top, bottom, xf, yf))
}

#[cfg(test)]
mod tests {
    use crate::scalar;

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as u32
        }

        /// Mostly any value, sometimes one of the two ends so sums reach
        /// their limits.
        fn sample(&mut self) -> u8 {
            match self.next() % 8 {
                0 => 0,
                1 => 255,
                _ => self.next() as u8,
            }
        }

        fn line<const N: usize>(&mut self) -> [u8; N] {
            let mut line = [0u8; N];
            for sample in &mut line {
                *sample = self.sample();
            }
            line
        }
    }

    fn refs<T, const N: usize>(lines: &[[T; N]; 6]) -> [&[T; N]; 6] {
        [
            &lines[0], &lines[1], &lines[2], &lines[3], &lines[4], &lines[5],
        ]
    }

    #[test]
    fn half_samples_match_the_plain_filter() {
        let mut rng = Lcg(11);
        for _ in 0..50_000 {
            let wide: [[u8; 16]; 6] = core::array::from_fn(|_| rng.line());
            assert_eq!(super::half16(refs(&wide)), scalar::half16(refs(&wide)));
            assert_eq!(super::raw16(refs(&wide)), scalar::raw16(refs(&wide)));
            let narrow: [[u8; 8]; 6] = core::array::from_fn(|_| rng.line());
            assert_eq!(super::half8(refs(&narrow)), scalar::half8(refs(&narrow)));
            assert_eq!(super::raw8(refs(&narrow)), scalar::raw8(refs(&narrow)));
        }
    }

    #[test]
    fn half_samples_match_at_the_extremes() {
        // Every tap at one end or the other: 64 patterns reach the largest
        // and smallest sums.
        for pattern in 0..64u32 {
            let wide: [[u8; 16]; 6] =
                core::array::from_fn(|tap| [if pattern >> tap & 1 == 1 { 255 } else { 0 }; 16]);
            assert_eq!(super::half16(refs(&wide)), scalar::half16(refs(&wide)));
            assert_eq!(super::raw16(refs(&wide)), scalar::raw16(refs(&wide)));
            let narrow: [[u8; 8]; 6] =
                core::array::from_fn(|tap| [if pattern >> tap & 1 == 1 { 255 } else { 0 }; 8]);
            assert_eq!(super::half8(refs(&narrow)), scalar::half8(refs(&narrow)));
            assert_eq!(super::raw8(refs(&narrow)), scalar::raw8(refs(&narrow)));
        }
    }

    #[test]
    fn centre_samples_match_the_plain_filter() {
        let mut rng = Lcg(12);
        // A first-pass sum lies between −2550 and 10710.
        let sum = |rng: &mut Lcg| match rng.next() % 6 {
            0 => -2550,
            1 => 10710,
            _ => (rng.next() % 13_261) as i16 - 2550,
        };
        for _ in 0..50_000 {
            let wide: [[i16; 16]; 6] =
                core::array::from_fn(|_| core::array::from_fn(|_| sum(&mut rng)));
            assert_eq!(super::centre16(refs(&wide)), scalar::centre16(refs(&wide)));
            let narrow: [[i16; 8]; 6] =
                core::array::from_fn(|_| core::array::from_fn(|_| sum(&mut rng)));
            assert_eq!(
                super::centre8(refs(&narrow)),
                scalar::centre8(refs(&narrow))
            );
        }
        for pattern in 0..64u32 {
            let wide: [[i16; 16]; 6] = core::array::from_fn(|tap| {
                [if pattern >> tap & 1 == 1 {
                    10710
                } else {
                    -2550
                }; 16]
            });
            assert_eq!(super::centre16(refs(&wide)), scalar::centre16(refs(&wide)));
        }
    }

    #[test]
    fn whole_blocks_match_the_plain_ones() {
        let mut rng = Lcg(15);
        let (stride, rows) = (40usize, 40usize);
        let plane: Vec<u8> = (0..stride * rows).map(|_| rng.sample()).collect();
        for round in 0..3_000u32 {
            let w = [8, 16][(round % 2) as usize];
            let h = [4, 8, 16][(round / 2 % 3) as usize];
            let at = (
                2 + (rng.next() % 16) as usize,
                2 + (rng.next() % 16) as usize,
            );
            let block = super::Block {
                src: &plane,
                stride,
                at,
                size: (w, h),
            };
            let frac = ((round / 6 % 4) as u8, (round / 24 % 4) as u8);
            let dst_stride = 20;
            let mut fast = vec![7u8; dst_stride * 16];
            let mut plain = fast.clone();
            assert!(super::luma_block(&mut fast, dst_stride, block, frac));
            assert!(scalar::luma_block(&mut plain, dst_stride, block, frac));
            assert_eq!(fast, plain, "luma {w}x{h} at {at:?} frac {frac:?}");
            // Samples outside the block stay as they were.
            assert!(
                fast.chunks(dst_stride)
                    .all(|row| row[w..] == [7u8; 20][w..])
            );

            let block = super::Block {
                size: (w / 2, h / 2),
                ..block
            };
            let frac = ((round / 6 % 8) as u8, (round / 48 % 8) as u8);
            let mut fast = vec![7u8; dst_stride * 16];
            let mut plain = fast.clone();
            assert!(super::chroma_block(&mut fast, dst_stride, block, frac));
            assert!(scalar::chroma_block(&mut plain, dst_stride, block, frac));
            assert_eq!(fast, plain, "chroma {w}x{h} at {at:?} frac {frac:?}");

            let n = 1 + (rng.next() % 40) as usize;
            let mut fast = vec![0u8; n];
            let mut plain = vec![0u8; n];
            assert!(super::avg_row(&mut fast, &plane[..n], &plane[100..100 + n]));
            assert!(scalar::avg_row(
                &mut plain,
                &plane[..n],
                &plane[100..100 + n]
            ));
            assert_eq!(fast, plain, "average of {n}");
        }
    }

    #[test]
    fn a_block_is_the_row_kernels_applied_to_its_rows() {
        // Ties the block functions to the single-row ones the other tests
        // check against the clause's formulas.
        let mut rng = Lcg(16);
        let stride = 32usize;
        let plane: Vec<u8> = (0..stride * 32).map(|_| rng.sample()).collect();
        let block = super::Block {
            src: &plane,
            stride,
            at: (5, 6),
            size: (16, 4),
        };
        let line = |start: usize| -> &[u8; 16] { plane[start..start + 16].try_into().unwrap() };
        let mut out = vec![0u8; 16 * 4];
        assert!(scalar::luma_block(&mut out, 16, block, (2, 0)));
        for row in 0..4 {
            let start = (6 + row) * stride + 5 - 2;
            let taps = core::array::from_fn(|tap| line(start + tap));
            assert_eq!(out[row * 16..row * 16 + 16], scalar::half16(taps));
        }
        assert!(scalar::luma_block(&mut out, 16, block, (0, 2)));
        for row in 0..4 {
            let start = (6 + row - 2) * stride + 5;
            let taps = core::array::from_fn(|tap| line(start + tap * stride));
            assert_eq!(out[row * 16..row * 16 + 16], scalar::half16(taps));
        }
        assert!(scalar::luma_block(&mut out, 16, block, (0, 0)));
        for row in 0..4 {
            assert_eq!(out[row * 16..row * 16 + 16], *line((6 + row) * stride + 5));
        }
    }

    #[test]
    fn mixed_blocks_match_the_plain_ones() {
        let mut rng = Lcg(17);
        for round in 0..30_000u32 {
            let w = [4usize, 8, 12, 16][(round % 4) as usize];
            let h = [4usize, 8, 16][(round / 4 % 3) as usize];
            let a: Vec<u8> = (0..w * h).map(|_| rng.sample()).collect();
            let b: Vec<u8> = (0..w * h).map(|_| rng.sample()).collect();
            let stride = 20;
            let (mut fast, mut plain) = (vec![9u8; stride * 16], vec![9u8; stride * 16]);
            assert!(super::avg_block(&mut fast, stride, &a, &b, (w, h)));
            assert!(scalar::avg_block(&mut plain, stride, &a, &b, (w, h)));
            assert_eq!(fast, plain, "average {w}x{h}");
            // Weights as a stream can give them: implicit ones sum to 64,
            // explicit ones are any pair, with an offset.
            let weight = super::BiWeight {
                w0: (rng.next() % 257) as i32 - 128,
                w1: (rng.next() % 257) as i32 - 128,
                shift: 1 + (rng.next() % 8) as i32,
                offset: (rng.next() % 257) as i32 - 128,
            };
            let (mut fast, mut plain) = (vec![9u8; stride * 16], vec![9u8; stride * 16]);
            assert!(super::weight_block(
                &mut fast,
                stride,
                &a,
                &b,
                (w, h),
                weight
            ));
            assert!(scalar::weight_block(
                &mut plain,
                stride,
                &a,
                &b,
                (w, h),
                weight
            ));
            assert_eq!(fast, plain, "{weight:?} {w}x{h}");
            // And against the clause's formula, written out here.
            let at = (rng.next() as usize) % (w * h);
            let sum = i32::from(a[at]) * weight.w0 + i32::from(b[at]) * weight.w1;
            let want =
                (((sum + (1 << (weight.shift - 1))) >> weight.shift) + weight.offset).clamp(0, 255);
            assert_eq!(i32::from(plain[at / w * stride + at % w]), want);
        }
        let weight = super::BiWeight {
            w0: 1 << 20,
            w1: 0,
            shift: 6,
            offset: 0,
        };
        let mut out = vec![0u8; 16];
        assert!(!super::weight_block(
            &mut out,
            4,
            &[0; 16],
            &[0; 16],
            (4, 4),
            weight
        ));
    }

    #[test]
    fn averages_match_the_plain_ones() {
        let mut rng = Lcg(13);
        for _ in 0..20_000 {
            let (a, b) = (rng.line(), rng.line());
            assert_eq!(super::avg16(&a, &b), scalar::avg16(&a, &b));
            let (a, b) = (rng.line(), rng.line());
            assert_eq!(super::avg8(&a, &b), scalar::avg8(&a, &b));
        }
    }

    #[test]
    fn chroma_samples_match_the_plain_ones() {
        let mut rng = Lcg(14);
        for round in 0..40_000u32 {
            let (top, bottom) = (rng.line(), rng.line());
            let (xf, yf) = ((round % 8) as u8, (round / 8 % 8) as u8);
            assert_eq!(
                super::chroma8(&top, &bottom, xf, yf),
                scalar::chroma8(&top, &bottom, xf, yf),
                "xf {xf} yf {yf}"
            );
        }
    }
}

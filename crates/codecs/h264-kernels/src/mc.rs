//! Inter prediction samples, clause 8.4.2.2.
//!
//! A luma half sample is the six-tap filter (1, −5, 20, 20, −5, 1) over six
//! full samples in a line. Along a row the six inputs are six windows of
//! that row, each one sample further right; down a column they are six rows.

pub use crate::mc_block::{BiWeight, Block, ChromaSource, UniWeight};

/// The rounded average of two predictions of `size` (width, height), whose
/// rows are `size.0` apart, into `dst` with rows `dst_stride` apart.
/// `false` means a block does not fit its slice and nothing was done.
#[inline]
pub fn avg_block(
    dst: &mut [u8],
    dst_stride: usize,
    a: &[u8],
    b: &[u8],
    size: (usize, usize),
) -> bool {
    kernel!(avg_block(dst, dst_stride, a, b, size))
}

/// The block of `size` at `dst`, rows `dst_stride` apart, weighted in
/// place: one prediction under explicit weights. Widths a multiple of 4 up
/// to 16. `false` means nothing was done: another width, a block that does
/// not fit `dst`, or weights outside [`UniWeight`]'s range.
#[inline]
pub fn scale_block(
    dst: &mut [u8],
    dst_stride: usize,
    size: (usize, usize),
    weight: UniWeight,
) -> bool {
    kernel!(scale_block(dst, dst_stride, size, weight))
}

/// The prediction at `dst`, rows `dst_stride` apart, mixed by `weight`
/// with the one in `b`, rows `size.0` apart, in place: the first
/// prediction made straight into the picture. Widths a multiple of 4 up to
/// 16. `false` means nothing was done: another width, a block that does
/// not fit, or weights outside the kernels' range.
#[inline]
pub fn weight_onto(
    dst: &mut [u8],
    dst_stride: usize,
    b: &[u8],
    size: (usize, usize),
    weight: BiWeight,
) -> bool {
    kernel!(weight_onto(dst, dst_stride, b, size, weight))
}

/// Two predictions mixed by `weight`, laid out as for [`avg_block`].
/// `false` also when the weights are outside the kernels' range.
#[inline]
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
#[inline]
pub fn luma_block(dst: &mut [u8], dst_stride: usize, block: Block<'_>, frac: (u8, u8)) -> bool {
    kernel!(luma_block(dst, dst_stride, block, frac))
}

/// Asks for `rows` rows of `plane`, `stride` apart from sample `at`, to be
/// brought into the cache before they are read, as FFmpeg's
/// `prefetch_motion` does for the reference a macroblock further along is
/// likely to read. Rows outside `plane` are left out. Nothing changes that
/// a caller could see.
#[inline]
pub fn prefetch(plane: &[u8], at: usize, stride: usize, rows: usize) {
    kernel!(prefetch(plane, at, stride, rows));
}

/// A chroma block at eighth-sample offsets `frac` (each 0 to 7). Covers
/// blocks 4 or 8 wide; `false` means nothing was done. The source must
/// hold one more sample after the block's last column and row.
#[inline]
pub fn chroma_block(dst: &mut [u8], dst_stride: usize, block: Block<'_>, frac: (u8, u8)) -> bool {
    kernel!(chroma_block(dst, dst_stride, block, frac))
}

/// The rounded average of `a` and `b` into `dst`. `false` means the inputs
/// are shorter than `dst` and nothing was done.
/// A luma block predicted from `first` into `dst` and, when `second` is
/// there, averaged with the block predicted from it, both at their quarter
/// offsets: default bi-prediction, clause 8.4.2.3.1. Both blocks have one
/// size. `false` means nothing was written: a size `luma_block` does not
/// take, or a block whose samples are not all in its source.
#[inline]
pub fn luma_pair(
    dst: &mut [u8],
    dst_stride: usize,
    first: (Block<'_>, (u8, u8)),
    second: Option<(Block<'_>, (u8, u8))>,
) -> bool {
    kernel!(luma_pair(dst, dst_stride, first, second))
}

/// [`chroma_pair`] for both planes at once, as one call: the Cb block
/// into `dst.0` and the Cr block into `dst.1`, rows `dst_stride` apart,
/// each predicted from `first` or, when there is a `second`, the rounded
/// average of the two. `size` is the block's, 4 or 8 wide and an even
/// height up to 8. `false` means nothing was written: another size, or a
/// source without one more sample after the block's last column and row.
#[inline]
pub fn chroma_planes(
    dst: (&mut [u8], &mut [u8]),
    dst_stride: usize,
    size: (usize, usize),
    first: ChromaSource<'_>,
    second: Option<ChromaSource<'_>>,
) -> bool {
    kernel!(chroma_planes(dst, dst_stride, size, first, second))
}

/// [`luma_pair`] for a chroma block at eighth offsets.
#[inline]
pub fn chroma_pair(
    dst: &mut [u8],
    dst_stride: usize,
    first: (Block<'_>, (u8, u8)),
    second: Option<(Block<'_>, (u8, u8))>,
) -> bool {
    kernel!(chroma_pair(dst, dst_stride, first, second))
}

#[inline]
pub fn avg_row(dst: &mut [u8], a: &[u8], b: &[u8]) -> bool {
    kernel!(avg_row(dst, a, b))
}

/// Half samples from six lines of full samples: the filter, rounded and
/// clipped to a sample.
#[inline]
pub fn half16(taps: [&[u8; 16]; 6]) -> [u8; 16] {
    kernel!(half16(taps))
}

/// [`half16`] for eight samples.
#[inline]
pub fn half8(taps: [&[u8; 8]; 6]) -> [u8; 8] {
    kernel!(half8(taps))
}

/// The filter's sum with no rounding, the first pass of a centre sample.
#[inline]
pub fn raw16(taps: [&[u8; 16]; 6]) -> [i16; 16] {
    kernel!(raw16(taps))
}

/// [`raw16`] for eight samples.
#[inline]
pub fn raw8(taps: [&[u8; 8]; 6]) -> [i16; 8] {
    kernel!(raw8(taps))
}

/// Centre samples (the half sample in both directions) from six lines of
/// [`raw16`] sums: the filter again, rounded and clipped.
#[inline]
pub fn centre16(taps: [&[i16; 16]; 6]) -> [u8; 16] {
    kernel!(centre16(taps))
}

/// [`centre16`] for eight samples.
#[inline]
pub fn centre8(taps: [&[i16; 8]; 6]) -> [u8; 8] {
    kernel!(centre8(taps))
}

/// The rounded average of two lines, which is a quarter sample and also a
/// default-weight prediction from two references.
#[inline]
pub fn avg16(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    kernel!(avg16(a, b))
}

/// [`avg16`] for eight samples.
#[inline]
pub fn avg8(a: &[u8; 8], b: &[u8; 8]) -> [u8; 8] {
    kernel!(avg8(a, b))
}

/// Eight chroma samples, clause 8.4.2.2.2, from the nine samples above and
/// the nine below. `xf` and `yf` are the eighth-sample offsets, 0 to 7.
#[inline]
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
    fn pairs_match_the_plain_ones() {
        let mut rng = Lcg(18);
        let (stride, rows) = (48usize, 48usize);
        let plane: Vec<u8> = (0..stride * rows).map(|_| rng.sample()).collect();
        let place = |rng: &mut Lcg| {
            (
                2 + (rng.next() % 20) as usize,
                2 + (rng.next() % 20) as usize,
            )
        };
        for round in 0..6_000u32 {
            let w = [8, 16][(round % 2) as usize];
            let h = [4, 8, 16][(round / 2 % 3) as usize];
            let block = |at| super::Block {
                src: &plane,
                stride,
                at,
                size: (w, h),
            };
            let quarter = |rng: &mut Lcg| ((rng.next() % 4) as u8, (rng.next() % 4) as u8);
            let first = (block(place(&mut rng)), quarter(&mut rng));
            let second = (round % 5 != 0).then(|| (block(place(&mut rng)), quarter(&mut rng)));
            let dst_stride = 20;
            let mut fast = vec![7u8; dst_stride * 16];
            let mut plain = fast.clone();
            assert!(super::luma_pair(&mut fast, dst_stride, first, second));
            assert!(scalar::luma_pair(&mut plain, dst_stride, first, second));
            assert_eq!(fast, plain, "luma pair {w}x{h}");
            assert!(
                fast.chunks(dst_stride)
                    .all(|row| row[w..] == [7u8; 20][w..])
            );

            let half = |at| super::Block {
                size: (w / 2, h / 2),
                ..block(at)
            };
            let eighth = |rng: &mut Lcg| ((rng.next() % 8) as u8, (rng.next() % 8) as u8);
            let first = (half(place(&mut rng)), eighth(&mut rng));
            let second = (round % 5 != 0).then(|| (half(place(&mut rng)), eighth(&mut rng)));
            let mut fast = vec![7u8; dst_stride * 16];
            let mut plain = fast.clone();
            assert!(super::chroma_pair(&mut fast, dst_stride, first, second));
            assert!(scalar::chroma_pair(&mut plain, dst_stride, first, second));
            assert_eq!(fast, plain, "chroma pair {}x{}", w / 2, h / 2);
        }
    }

    #[test]
    fn both_chroma_planes_match_the_plain_ones() {
        let mut rng = Lcg(23);
        let (stride, rows) = (40usize, 30usize);
        let cb: Vec<u8> = (0..stride * rows).map(|_| rng.sample()).collect();
        let cr: Vec<u8> = (0..stride * rows).map(|_| rng.sample()).collect();
        let mut written = 0;
        for round in 0..20_000u32 {
            let w = [4, 8][(round % 2) as usize];
            let h = [2, 4, 8, 3][(round / 2 % 4) as usize];
            // Places up to the last one whose reach fits.
            let source = |rng: &mut Lcg| super::ChromaSource {
                cb: &cb,
                cr: &cr,
                stride,
                at: (
                    (rng.next() % 32) as usize,
                    (rng.next() % (rows as u32 - 8)) as usize,
                ),
                frac: ((rng.next() % 8) as u8, (rng.next() % 8) as u8),
            };
            let first = source(&mut rng);
            let second = (round % 3 != 0).then(|| source(&mut rng));
            let dst_stride = 12;
            let fresh = || -> Vec<u8> { (0..dst_stride * 8).map(|i| (i * 7) as u8).collect() };
            let (mut fast_cb, mut fast_cr) = (fresh(), fresh());
            let (mut plain_cb, mut plain_cr) = (fresh(), fresh());
            let fast = super::chroma_planes(
                (&mut fast_cb, &mut fast_cr),
                dst_stride,
                (w, h),
                first,
                second,
            );
            let plain = scalar::chroma_planes(
                (&mut plain_cb, &mut plain_cr),
                dst_stride,
                (w, h),
                first,
                second,
            );
            assert_eq!(fast, plain);
            if !fast {
                // An odd height: nothing written.
                assert_eq!(h, 3);
                assert_eq!((&fast_cb, &fast_cr), (&fresh(), &fresh()));
                continue;
            }
            written += 1;
            assert_eq!(
                (&fast_cb, &fast_cr),
                (&plain_cb, &plain_cr),
                "{w}x{h} {round}"
            );
            // Each plane is what `chroma_pair` makes of it.
            for (cb, made) in [(true, &fast_cb), (false, &fast_cr)] {
                let mut one = fresh();
                let pair = (first.plane(cb, (w, h)), first.frac);
                let other = second.map(|s| (s.plane(cb, (w, h)), s.frac));
                assert!(super::chroma_pair(&mut one, dst_stride, pair, other));
                assert_eq!(&one, made);
            }
        }
        assert!(written > 10_000);
    }

    #[test]
    fn scaled_blocks_match_the_plain_ones() {
        let mut rng = Lcg(19);
        for round in 0..20_000u32 {
            let w = 4 * (1 + (round % 4) as usize);
            let h = 1 + (rng.next() % 16) as usize;
            // The ends of each range, and values between.
            let weight = super::UniWeight {
                weight: [-126, 128, 2, 3, 127][(round / 4 % 5) as usize] - (rng.next() % 3) as i32,
                offset: (rng.next() % 511) as i32 - 255,
                shift: (rng.next() % 8) as i32,
            };
            let stride = 20;
            let mut fast: Vec<u8> = (0..stride * 16).map(|_| rng.sample()).collect();
            let mut plain = fast.clone();
            let before = fast.clone();
            assert!(super::scale_block(&mut fast, stride, (w, h), weight));
            assert!(scalar::scale_block(&mut plain, stride, (w, h), weight));
            assert_eq!(fast, plain, "{w}x{h} {weight:?}");
            assert!(
                fast.chunks(stride)
                    .zip(before.chunks(stride))
                    .all(|(row, old)| row[w..] == old[w..])
            );
        }
    }

    #[test]
    fn weights_onto_a_block_match_the_plain_ones() {
        let mut rng = Lcg(23);
        for round in 0..20_000u32 {
            let w = 4 * (1 + (round % 4) as usize);
            let h = 1 + (rng.next() % 16) as usize;
            let pick = |rng: &mut Lcg| {
                [-255, 256, 1, 2, 64, 255][(rng.next() % 6) as usize] - (rng.next() % 2) as i32
            };
            let weight = super::BiWeight {
                w0: pick(&mut rng),
                w1: pick(&mut rng),
                shift: 1 + (rng.next() % 15) as i32,
                offset: (rng.next() % 511) as i32 - 255,
            };
            let stride = 20;
            let mut fast: Vec<u8> = (0..stride * 16).map(|_| rng.sample()).collect();
            let b: Vec<u8> = (0..w * h).map(|_| rng.sample()).collect();
            let mut plain = fast.clone();
            let before = fast.clone();
            assert!(super::weight_onto(&mut fast, stride, &b, (w, h), weight));
            assert!(scalar::weight_onto(&mut plain, stride, &b, (w, h), weight));
            assert_eq!(fast, plain, "{w}x{h} {weight:?}");
            assert!(
                fast.chunks(stride)
                    .zip(before.chunks(stride))
                    .all(|(row, old)| row[w..] == old[w..])
            );
        }
    }

    #[test]
    fn weights_held_to_16_bits_match_the_plain_ones() {
        // The weights a stream uses: implicit ones, whose sum is 64 at a
        // shift of 6, and explicit ones with small shifts and offsets,
        // which the vector kernels mix in 16 bits.
        let mut rng = Lcg(29);
        for round in 0..40_000u32 {
            let w = 4 * (1 + (round % 4) as usize);
            let h = 1 + (rng.next() % 16) as usize;
            let weight = if round % 2 == 0 {
                let w1 = (rng.next() % 193) as i32 - 64;
                super::BiWeight {
                    w0: 64 - w1,
                    w1,
                    shift: 6,
                    offset: 0,
                }
            } else {
                let shift = 1 + (rng.next() % 8) as i32;
                let limit: i32 = 1 << (shift - 1).min(7);
                super::BiWeight {
                    w0: (rng.next() % (2 * limit + 1) as u32) as i32 - limit,
                    w1: (rng.next() % (2 * limit + 1) as u32) as i32 - limit,
                    shift,
                    offset: (rng.next() % 61) as i32 - 30,
                }
            };
            let stride = 20;
            let a: Vec<u8> = (0..w * h).map(|_| rng.sample()).collect();
            let b: Vec<u8> = (0..w * h).map(|_| rng.sample()).collect();
            let start: Vec<u8> = (0..stride * 16).map(|_| rng.sample()).collect();
            let (mut fast, mut plain) = (start.clone(), start.clone());
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
            assert_eq!(fast, plain, "block {w}x{h} {weight:?}");
            let (mut fast, mut plain) = (start.clone(), start);
            assert!(super::weight_onto(&mut fast, stride, &b, (w, h), weight));
            assert!(scalar::weight_onto(&mut plain, stride, &b, (w, h), weight));
            assert_eq!(fast, plain, "onto {w}x{h} {weight:?}");
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

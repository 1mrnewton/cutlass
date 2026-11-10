//! Inter prediction samples, clause 8.4.2.2.
//!
//! A luma half sample is the six-tap filter (1, −5, 20, 20, −5, 1) over six
//! full samples in a line. Along a row the six inputs are six windows of
//! that row, each one sample further right; down a column they are six rows.

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

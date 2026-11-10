//! The deblocking filter across one 16-sample luma edge, clause 8.7.2.
//!
//! The samples on the two sides of an edge are named p and q, counted away
//! from it: p3 p2 p1 p0 | q0 q1 q2 q3.

/// The eight rows around an edge that runs along a row: p3, p2, p1, p0,
/// then q0, q1, q2, q3, each 16 samples along the edge.
pub type Rows = [[u8; 16]; 8];

/// The sixteen lines across an edge that runs down the picture. Each line
/// is p3, p2, p1, p0, q0, q1, q2, q3.
pub type Lines = [[u8; 8]; 16];

/// The four rows around an 8-sample chroma edge: p1, p0, then q0, q1.
/// Chroma filters change only p0 and q0.
pub type ChromaRows = [[u8; 8]; 4];

/// The chroma filter for a strength below 4. `tc0` is per group of two
/// samples along the edge; a negative one leaves its group alone.
pub fn chroma_normal(rows: &mut ChromaRows, alpha: i32, beta: i32, tc0: [i8; 4]) {
    kernel!(chroma_normal(rows, alpha, beta, tc0));
}

/// The chroma filter for strength 4.
pub fn chroma_strong(rows: &mut ChromaRows, alpha: i32, beta: i32) {
    kernel!(chroma_strong(rows, alpha, beta));
}

/// The filter for a strength below 4 (8.7.2.3), on rows.
///
/// `alpha` and `beta` are the edge's thresholds. `tc0` is the clipping
/// value of each group of four samples along the edge; a negative one
/// leaves its group alone.
pub fn luma_normal_rows(rows: &mut Rows, alpha: i32, beta: i32, tc0: [i8; 4]) {
    kernel!(luma_normal_rows(rows, alpha, beta, tc0));
}

/// The filter for strength 4 (8.7.2.4), on rows.
pub fn luma_strong_rows(rows: &mut Rows, alpha: i32, beta: i32) {
    kernel!(luma_strong_rows(rows, alpha, beta));
}

/// [`luma_normal_rows`] for an edge that runs down the picture. `tc0` is
/// per group of four lines.
pub fn luma_normal_lines(lines: &mut Lines, alpha: i32, beta: i32, tc0: [i8; 4]) {
    kernel!(luma_normal_lines(lines, alpha, beta, tc0));
}

/// [`luma_strong_rows`] for an edge that runs down the picture.
pub fn luma_strong_lines(lines: &mut Lines, alpha: i32, beta: i32) {
    kernel!(luma_strong_lines(lines, alpha, beta));
}

#[cfg(test)]
mod tests {
    use super::{Lines, Rows};
    use crate::scalar;

    /// A small generator, so the samples are the same on every run.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as u32
        }

        /// A sample. `spread` 0 is any value; otherwise it stays within
        /// `spread` of `base`, like the two sides of a real edge.
        fn sample(&mut self, base: u8, spread: u32) -> u8 {
            if spread == 0 {
                return self.next() as u8;
            }
            let offset = (self.next() % (2 * spread + 1)) as i32 - spread as i32;
            (i32::from(base) + offset).clamp(0, 255) as u8
        }
    }

    fn rows(rng: &mut Lcg, spread: u32) -> Rows {
        let mut rows = [[0u8; 16]; 8];
        for column in 0..16 {
            let base = rng.next() as u8;
            for row in &mut rows {
                row[column] = rng.sample(base, spread);
            }
        }
        rows
    }

    fn lines(rng: &mut Lcg, spread: u32) -> Lines {
        let mut lines = [[0u8; 8]; 16];
        for line in &mut lines {
            let base = rng.next() as u8;
            for sample in line {
                *sample = rng.sample(base, spread);
            }
        }
        lines
    }

    /// Thresholds as the tables give them, and a few past their ends.
    fn thresholds(rng: &mut Lcg) -> (i32, i32, [i8; 4]) {
        let alpha = [0, 4, 15, 40, 113, 255][(rng.next() % 6) as usize];
        let beta = [0, 2, 6, 11, 18][(rng.next() % 5) as usize];
        let mut tc0 = [0i8; 4];
        for group in &mut tc0 {
            *group = [-1, 0, 1, 4, 13, 25][(rng.next() % 6) as usize];
        }
        (alpha, beta, tc0)
    }

    #[test]
    fn the_normal_filter_matches_the_plain_one() {
        let mut rng = Lcg(1);
        let mut changed = 0;
        for round in 0..40_000 {
            let spread = [0, 1, 3, 8, 30][round % 5];
            let (alpha, beta, tc0) = thresholds(&mut rng);
            let mut fast = rows(&mut rng, spread);
            let mut plain = fast;
            let before = fast;
            super::luma_normal_rows(&mut fast, alpha, beta, tc0);
            scalar::luma_normal_rows(&mut plain, alpha, beta, tc0);
            assert_eq!(fast, plain, "rows, alpha {alpha} beta {beta} tc0 {tc0:?}");
            changed += usize::from(fast != before);
            let mut fast = lines(&mut rng, spread);
            let mut plain = fast;
            super::luma_normal_lines(&mut fast, alpha, beta, tc0);
            scalar::luma_normal_lines(&mut plain, alpha, beta, tc0);
            assert_eq!(fast, plain, "lines, alpha {alpha} beta {beta} tc0 {tc0:?}");
        }
        // The samples have to be ones the filter acts on, or the match
        // above proves nothing.
        assert!(changed > 5_000, "only {changed} blocks were filtered");
    }

    #[test]
    fn the_strong_filter_matches_the_plain_one() {
        let mut rng = Lcg(2);
        for round in 0..40_000 {
            let spread = [0, 1, 3, 8, 30][round % 5];
            let (alpha, beta, _) = thresholds(&mut rng);
            let mut fast = rows(&mut rng, spread);
            let mut plain = fast;
            super::luma_strong_rows(&mut fast, alpha, beta);
            scalar::luma_strong_rows(&mut plain, alpha, beta);
            assert_eq!(fast, plain, "rows, alpha {alpha} beta {beta}");
            let mut fast = lines(&mut rng, spread);
            let mut plain = fast;
            super::luma_strong_lines(&mut fast, alpha, beta);
            scalar::luma_strong_lines(&mut plain, alpha, beta);
            assert_eq!(fast, plain, "lines, alpha {alpha} beta {beta}");
        }
    }

    #[test]
    fn the_chroma_filters_match_the_plain_ones() {
        let mut rng = Lcg(4);
        let mut changed = 0;
        for round in 0..60_000 {
            let spread = [0, 1, 3, 8, 30][round % 5];
            let (alpha, beta, tc0) = thresholds(&mut rng);
            let mut rows = [[0u8; 8]; 4];
            for column in 0..8 {
                let base = match rng.next() % 6 {
                    0 => 0,
                    1 => 255,
                    _ => rng.next() as u8,
                };
                for row in &mut rows {
                    row[column] = rng.sample(base, spread);
                }
            }
            let (mut fast, mut plain) = (rows, rows);
            super::chroma_normal(&mut fast, alpha, beta, tc0);
            scalar::chroma_normal(&mut plain, alpha, beta, tc0);
            assert_eq!(fast, plain, "normal, alpha {alpha} beta {beta} tc0 {tc0:?}");
            changed += usize::from(fast != rows);
            let (mut fast, mut plain) = (rows, rows);
            super::chroma_strong(&mut fast, alpha, beta);
            scalar::chroma_strong(&mut plain, alpha, beta);
            assert_eq!(fast, plain, "strong, alpha {alpha} beta {beta}");
            assert_eq!((fast[0], fast[3]), (rows[0], rows[3]));
        }
        assert!(changed > 5_000, "only {changed} blocks were filtered");
    }

    #[test]
    fn the_extremes_match_the_plain_filters() {
        for fill in [0u8, 1, 127, 128, 254, 255] {
            for other in [0u8, 255, 128] {
                let mut rows = [[fill; 16]; 8];
                for row in rows.iter_mut().skip(4) {
                    *row = [other; 16];
                }
                for (alpha, beta) in [(255, 18), (4, 2), (0, 0)] {
                    let mut fast = rows;
                    let mut plain = rows;
                    super::luma_normal_rows(&mut fast, alpha, beta, [25, 0, 13, -1]);
                    scalar::luma_normal_rows(&mut plain, alpha, beta, [25, 0, 13, -1]);
                    assert_eq!(fast, plain);
                    let mut fast = rows;
                    let mut plain = rows;
                    super::luma_strong_rows(&mut fast, alpha, beta);
                    scalar::luma_strong_rows(&mut plain, alpha, beta);
                    assert_eq!(fast, plain);
                }
            }
        }
    }

    #[test]
    fn a_filter_leaves_the_outer_samples_alone() {
        let mut rng = Lcg(3);
        for _ in 0..2_000 {
            let before = rows(&mut rng, 3);
            let mut after = before;
            super::luma_normal_rows(&mut after, 255, 18, [25; 4]);
            assert_eq!((after[0], after[1]), (before[0], before[1]));
            assert_eq!((after[6], after[7]), (before[6], before[7]));
            let mut after = before;
            super::luma_strong_rows(&mut after, 255, 18);
            assert_eq!((after[0], after[7]), (before[0], before[7]));
        }
    }
}

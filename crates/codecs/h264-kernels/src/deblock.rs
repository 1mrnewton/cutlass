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

pub use crate::edge::{Edge, Filter, MbEdges};

/// Filters one 16-sample luma edge of `plane` in place. `false` means the
/// edge's samples are not all in the plane and nothing was done.
#[inline]
pub fn luma_edge(plane: &mut [u8], edge: Edge) -> bool {
    kernel!(luma_edge(plane, edge))
}

/// Filters one 8-sample chroma edge of `plane` in place; `edge.tc0` is per
/// two samples. `false` as for [`luma_edge`].
#[inline]
pub fn chroma_edge(plane: &mut [u8], edge: Edge) -> bool {
    kernel!(chroma_edge(plane, edge))
}

/// Filters every edge of one 16×16 luma macroblock of `plane` in place,
/// as [`luma_edge`] on each in turn would. `false` means the macroblock's
/// samples are not all in the plane and nothing was done.
#[inline]
pub fn luma_mb(plane: &mut [u8], edges: &MbEdges) -> bool {
    kernel!(luma_mb(plane, edges))
}

/// [`luma_mb`] for one 8×8 chroma macroblock, as [`chroma_edge`] on each
/// edge in turn would.
#[inline]
pub fn chroma_mb(plane: &mut [u8], edges: &MbEdges) -> bool {
    kernel!(chroma_mb(plane, edges))
}

/// [`luma_mb`] on the luma plane and [`chroma_mb`] on each 4:2:0 chroma
/// plane of one macroblock, each plane only when its `some` is set. Every
/// plane's samples are checked before any is filtered, and only then:
/// `false` means one plane's are not all there and nothing was done.
#[inline]
pub fn mb(
    planes: (&mut [u8], &mut [u8], &mut [u8]),
    edges: &[MbEdges; 3],
    some: [bool; 3],
) -> bool {
    kernel!(mb_planes(planes, edges, some))
}

/// [`mb`] when both chroma planes are filtered at the same edges,
/// `chroma`, as they are when Cb and Cr share their QPs: `some` is for
/// luma and for both chroma planes. The two are filtered side by side.
#[inline]
pub fn mb_twin(
    planes: (&mut [u8], &mut [u8], &mut [u8]),
    luma: &MbEdges,
    chroma: &MbEdges,
    some: [bool; 2],
) -> bool {
    kernel!(mb_twin(planes, luma, chroma, some))
}

/// The four rows around an 8-sample chroma edge: p1, p0, then q0, q1.
/// Chroma filters change only p0 and q0.
pub type ChromaRows = [[u8; 8]; 4];

/// The chroma filter for a strength below 4. `tc0` is per group of two
/// samples along the edge; a negative one leaves its group alone.
#[inline]
pub fn chroma_normal(rows: &mut ChromaRows, alpha: i32, beta: i32, tc0: [i8; 4]) {
    kernel!(chroma_normal(rows, alpha, beta, tc0));
}

/// The chroma filter for strength 4.
#[inline]
pub fn chroma_strong(rows: &mut ChromaRows, alpha: i32, beta: i32) {
    kernel!(chroma_strong(rows, alpha, beta));
}

/// The filter for a strength below 4 (8.7.2.3), on rows.
///
/// `alpha` and `beta` are the edge's thresholds. `tc0` is the clipping
/// value of each group of four samples along the edge; a negative one
/// leaves its group alone.
#[inline]
pub fn luma_normal_rows(rows: &mut Rows, alpha: i32, beta: i32, tc0: [i8; 4]) {
    kernel!(luma_normal_rows(rows, alpha, beta, tc0));
}

/// The filter for strength 4 (8.7.2.4), on rows.
#[inline]
pub fn luma_strong_rows(rows: &mut Rows, alpha: i32, beta: i32) {
    kernel!(luma_strong_rows(rows, alpha, beta));
}

/// [`luma_normal_rows`] for an edge that runs down the picture. `tc0` is
/// per group of four lines.
#[inline]
pub fn luma_normal_lines(lines: &mut Lines, alpha: i32, beta: i32, tc0: [i8; 4]) {
    kernel!(luma_normal_lines(lines, alpha, beta, tc0));
}

/// [`luma_strong_rows`] for an edge that runs down the picture.
#[inline]
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
    fn whole_edges_match_the_plain_ones() {
        let mut rng = Lcg(5);
        let stride = 24usize;
        for round in 0..20_000 {
            let spread = [0, 1, 3, 8, 30][round % 5];
            let base = rng.next() as u8;
            let plane: Vec<u8> = (0..stride * 24).map(|_| rng.sample(base, spread)).collect();
            let (alpha, beta, tc0) = thresholds(&mut rng);
            let edge = super::Edge {
                at: (4 + (rng.next() % 4) as usize) * stride + 4 + (rng.next() % 4) as usize,
                down: stride,
                vertical: round % 2 == 0,
                alpha,
                beta,
                tc0: (round % 3 != 0).then_some(tc0),
            };
            let (mut fast, mut plain) = (plane.clone(), plane.clone());
            assert!(super::luma_edge(&mut fast, edge));
            assert!(scalar::luma_edge(&mut plain, edge));
            assert_eq!(fast, plain, "luma {round}");
            let (mut fast, mut plain) = (plane.clone(), plane.clone());
            assert!(super::chroma_edge(&mut fast, edge));
            assert!(scalar::chroma_edge(&mut plain, edge));
            assert_eq!(fast, plain, "chroma {round}");
        }
        // An edge too near the plane's start or end is refused untouched.
        let mut plane = vec![9u8; 64];
        let edge = super::Edge {
            at: 1,
            down: 8,
            vertical: true,
            alpha: 255,
            beta: 18,
            tc0: None,
        };
        assert!(!super::luma_edge(&mut plane, edge));
        assert!(!super::chroma_edge(
            &mut plane,
            super::Edge { at: 60, ..edge }
        ));
        assert_eq!(plane, vec![9u8; 64]);
    }

    /// Every edge of a macroblock: left alone, or filtered at some
    /// strength, at random.
    fn mb_edges(rng: &mut Lcg, at: usize, down: usize, odd: bool) -> super::MbEdges {
        let mut pick = |allowed: bool| {
            let (alpha, beta, tc0) = thresholds(rng);
            let alpha = u8::try_from(alpha).expect("a table value");
            let beta = u8::try_from(beta).expect("a table value");
            match rng.next() % 4 {
                _ if !allowed => None,
                0 => None,
                1 => Some(super::Filter {
                    alpha,
                    beta,
                    tc0: None,
                }),
                _ => Some(super::Filter {
                    alpha,
                    beta,
                    tc0: Some(tc0),
                }),
            }
        };
        // The first edge of each kind needs samples before the macroblock.
        // `odd` is whether the second and fourth edges may be filtered:
        // 4:2:0 chroma has none there.
        let vertical = [pick(at % down >= 4), pick(odd), pick(true), pick(odd)];
        let horizontal = [pick(at >= 4 * down), pick(odd), pick(true), pick(odd)];
        super::MbEdges {
            at,
            down,
            vertical,
            horizontal,
        }
    }

    #[test]
    fn boundaries_alone_match_the_plain_ones() {
        // Most macroblocks filter only their left and top edges, which the
        // vector version does on its own path.
        let mut rng = Lcg(8);
        let mut changed = 0;
        for round in 0..20_000 {
            let spread = [0, 2, 6, 30][round % 4];
            let down = 40 + (rng.next() % 9) as usize;
            let base = rng.next() as u8;
            let before: Vec<u8> = (0..down * 44).map(|_| rng.sample(base, spread)).collect();
            let x = [0, 4, 8, 20][(rng.next() % 4) as usize];
            let y = [0, 3, 4, 16, 24][(rng.next() % 5) as usize];
            let mut edges = mb_edges(&mut rng, y * down + x, down, true);
            edges.vertical[1..].fill(None);
            edges.horizontal[1..].fill(None);
            let mut fast = before.clone();
            let mut plain = before.clone();
            assert_eq!(
                super::luma_mb(&mut fast, &edges),
                scalar::luma_mb(&mut plain, &edges),
                "round {round}: {edges:?}"
            );
            assert_eq!(fast, plain, "round {round}: {edges:?}");
            changed += usize::from(fast != before);
        }
        assert!(changed > 5_000, "only {changed} macroblocks were filtered");
    }

    #[test]
    fn chroma_boundaries_alone_match_the_plain_ones() {
        let mut rng = Lcg(9);
        let mut changed = 0;
        for round in 0..20_000 {
            let spread = [0, 2, 6, 30][round % 4];
            let down = 40 + (rng.next() % 9) as usize;
            let base = rng.next() as u8;
            let before: Vec<u8> = (0..down * 44).map(|_| rng.sample(base, spread)).collect();
            let x = [0, 2, 4, 8, 20][(rng.next() % 5) as usize];
            let y = [0, 2, 3, 16, 24][(rng.next() % 5) as usize];
            let mut edges = mb_edges(&mut rng, y * down + x, down, false);
            edges.vertical[0] = edges.vertical[2].filter(|_| x >= 2);
            edges.horizontal[0] = edges.horizontal[2].filter(|_| y >= 2);
            edges.vertical[1..].fill(None);
            edges.horizontal[1..].fill(None);
            let mut fast = before.clone();
            let mut plain = before.clone();
            assert_eq!(
                super::chroma_mb(&mut fast, &edges),
                scalar::chroma_mb(&mut plain, &edges),
                "round {round}: {edges:?}"
            );
            assert_eq!(fast, plain, "round {round}: {edges:?}");
            changed += usize::from(fast != before);
        }
        assert!(changed > 5_000, "only {changed} macroblocks were filtered");
    }

    #[test]
    fn whole_macroblocks_match_the_plain_ones() {
        let mut rng = Lcg(7);
        let mut changed = 0;
        for round in 0..6_000 {
            let spread = [0, 2, 6, 30][round % 4];
            let down = 40 + (rng.next() % 9) as usize;
            let base = rng.next() as u8;
            let before: Vec<u8> = (0..down * 44).map(|_| rng.sample(base, spread)).collect();
            // Anywhere, the top row and the first column included.
            let x = [0, 4, 8, 20][(rng.next() % 4) as usize];
            let y = [0, 3, 4, 16, 24][(rng.next() % 5) as usize];
            for luma in [true, false] {
                // Luma leaves out edges 1 and 3 now and then, as an 8×8
                // transform does, which the vector version does on its own path.
                let odd = if luma { round % 8 != 1 } else { round % 8 == 0 };
                let edges = mb_edges(&mut rng, y * down + x, down, odd);
                let mut fast = before.clone();
                let mut plain = before.clone();
                let (done, wanted) = if luma {
                    (
                        super::luma_mb(&mut fast, &edges),
                        scalar::luma_mb(&mut plain, &edges),
                    )
                } else {
                    (
                        super::chroma_mb(&mut fast, &edges),
                        scalar::chroma_mb(&mut plain, &edges),
                    )
                };
                assert_eq!(done, wanted, "round {round}, luma {luma}: {edges:?}");
                assert_eq!(fast, plain, "round {round}, luma {luma}: {edges:?}");
                changed += usize::from(fast != before);
            }
        }
        assert!(changed > 4_000, "only {changed} macroblocks were filtered");
    }

    #[test]
    fn three_planes_match_the_plain_ones_and_all_or_none() {
        let mut rng = Lcg(12);
        let mut changed = 0;
        let mut refused = 0;
        for round in 0..6_000 {
            let spread = [0, 2, 6, 30][round % 4];
            let down = 40 + (rng.next() % 9) as usize;
            let base = rng.next() as u8;
            let mut plane =
                |len: usize| -> Vec<u8> { (0..len).map(|_| rng.sample(base, spread)).collect() };
            let before = [plane(down * 44), plane(down * 44), plane(down * 44)];
            let x = [0, 4, 8, 20][(rng.next() % 4) as usize];
            let y = [0, 3, 4, 16, 24][(rng.next() % 5) as usize];
            // Now and then one chroma macroblock lies past its plane's end.
            let far = if round % 7 == 3 { down * 40 } else { 0 };
            let edges = [
                mb_edges(&mut rng, y * down + x, down, round % 8 != 1),
                mb_edges(&mut rng, y * down + x + far, down, false),
                mb_edges(&mut rng, y * down + x, down, false),
            ];
            let some = [0, 1, 2].map(|_| !rng.next().is_multiple_of(5));
            let [mut fast_y, mut fast_u, mut fast_v] = before.clone();
            let done = super::mb((&mut fast_y, &mut fast_u, &mut fast_v), &edges, some);
            let [mut plain_y, mut plain_u, mut plain_v] = before.clone();
            let [some_y, some_u, some_v] = some;
            let fits = (!some_y || edges[0].fits(plain_y.len(), 16))
                && (!some_u || edges[1].fits(plain_u.len(), 8))
                && (!some_v || edges[2].fits(plain_v.len(), 8));
            let wanted = fits
                && (!some_y || scalar::luma_mb(&mut plain_y, &edges[0]))
                && (!some_u || scalar::chroma_mb(&mut plain_u, &edges[1]))
                && (!some_v || scalar::chroma_mb(&mut plain_v, &edges[2]));
            assert_eq!(done, wanted, "round {round}");
            assert_eq!(
                [&fast_y, &fast_u, &fast_v],
                [&plain_y, &plain_u, &plain_v],
                "round {round}"
            );
            if !done {
                assert_eq!([fast_y, fast_u, fast_v], before, "round {round}");
                refused += 1;
            } else {
                changed += usize::from(
                    [&fast_y, &fast_u, &fast_v] != [&before[0], &before[1], &before[2]],
                );
            }
        }
        assert!(changed > 3_000, "only {changed} macroblocks were filtered");
        assert!(refused > 300, "only {refused} were refused");
    }

    #[test]
    fn twin_chroma_planes_match_the_plain_ones() {
        let mut rng = Lcg(13);
        let mut changed = 0;
        let mut refused = 0;
        for round in 0..20_000 {
            let spread = [0, 2, 6, 30][round % 4];
            let down = 40 + (rng.next() % 9) as usize;
            let base = rng.next() as u8;
            let mut plane =
                |len: usize| -> Vec<u8> { (0..len).map(|_| rng.sample(base, spread)).collect() };
            // Now and then Cr is shorter than Cb and the macroblock lies
            // past its end.
            let short = if round % 11 == 5 {
                down * 10
            } else {
                down * 44
            };
            let before = [plane(down * 44), plane(down * 44), plane(short)];
            let x = [0, 2, 4, 8, 20][(rng.next() % 5) as usize];
            let y = [0, 2, 3, 4, 16, 24][(rng.next() % 6) as usize];
            let luma = mb_edges(&mut rng, y * down + x, down, round % 8 != 1);
            let mut chroma = mb_edges(&mut rng, y * down + x, down, round % 16 == 0);
            // Half the time the boundary alone, which has its own path.
            if round % 2 == 0 {
                chroma.vertical[1..].fill(None);
                chroma.horizontal[1..].fill(None);
                if x < 2 {
                    chroma.vertical[0] = None;
                }
                if y < 2 {
                    chroma.horizontal[0] = None;
                }
            }
            let some = [0, 1].map(|_| !rng.next().is_multiple_of(5));
            let [mut fast_y, mut fast_u, mut fast_v] = before.clone();
            let done = super::mb_twin(
                (&mut fast_y, &mut fast_u, &mut fast_v),
                &luma,
                &chroma,
                some,
            );
            let [mut plain_y, mut plain_u, mut plain_v] = before.clone();
            let wanted = scalar::mb_twin(
                (&mut plain_y, &mut plain_u, &mut plain_v),
                &luma,
                &chroma,
                some,
            );
            assert_eq!(done, wanted, "round {round}: {chroma:?}");
            assert_eq!(
                [&fast_y, &fast_u, &fast_v],
                [&plain_y, &plain_u, &plain_v],
                "round {round}: {chroma:?}"
            );
            if !done {
                assert_eq!([fast_y, fast_u, fast_v], before, "round {round}");
                refused += 1;
            } else {
                changed += usize::from([&fast_u, &fast_v] != [&before[1], &before[2]]);
            }
        }
        assert!(changed > 5_000, "only {changed} macroblocks were filtered");
        assert!(refused > 500, "only {refused} were refused");
    }

    #[test]
    fn a_macroblock_outside_the_plane_is_left_alone() {
        let down = 32;
        let before = vec![90u8; down * 20];
        let strong = Some(super::Filter {
            alpha: 40,
            beta: 10,
            tc0: None,
        });
        for at in [down * 5, down * 4 + 20, down * 30] {
            let edges = super::MbEdges {
                at,
                down,
                vertical: [None, strong, strong, strong],
                horizontal: [None, strong, strong, strong],
            };
            let mut plane = before.clone();
            assert!(!super::luma_mb(&mut plane, &edges));
            assert_eq!(plane, before);
        }
    }

    #[test]
    fn a_whole_edge_is_the_row_filter_on_its_rows() {
        // Ties the plane functions to the row filters the other tests check.
        let mut rng = Lcg(6);
        let stride = 16usize;
        let plane: Vec<u8> = (0..stride * 8).map(|_| rng.sample(100, 3)).collect();
        let edge = super::Edge {
            at: 4 * stride,
            down: stride,
            vertical: false,
            alpha: 255,
            beta: 18,
            tc0: Some([3; 4]),
        };
        let mut filtered = plane.clone();
        assert!(scalar::luma_edge(&mut filtered, edge));
        let mut rows = [[0u8; 16]; 8];
        for (row, samples) in rows.iter_mut().zip(plane.chunks(stride)) {
            row.copy_from_slice(samples);
        }
        scalar::luma_normal_rows(&mut rows, 255, 18, [3; 4]);
        assert_eq!(filtered, rows.concat());
        assert_ne!(filtered, plane);
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

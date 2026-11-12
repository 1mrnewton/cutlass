//! Intra prediction of a whole 16×16 luma block and a whole 8×8 chroma
//! block of a 4:2:0 picture, clauses 8.3.3 and 8.3.4, made straight in the
//! picture from the samples beside the block there.

/// Predicts the 16×16 block of `plane` whose first sample is at `at`, with
/// rows `down` apart, in Intra_16x16 `mode` (0 vertical, 1 horizontal, 2
/// DC, 3 plane), from the row above it when `above` holds, the column to
/// its left when `left` holds, and the sample at their corner. A mode
/// whose samples are not there, or any other mode, fills the block with
/// 128. `false` means the block with the row above and the column to the
/// left is not all in the plane, or the rows overlap, and nothing was
/// done.
#[inline]
pub fn pred16(plane: &mut [u8], at: usize, down: usize, mode: u8, near: (bool, bool)) -> bool {
    kernel!(pred16(plane, at, down, mode, near))
}

/// [`pred16`] for an 8×8 chroma block of a 4:2:0 picture, clause 8.3.4,
/// in `mode` 0 DC, 1 horizontal, 2 vertical or 3 plane. A mode whose
/// samples are not there, or any other mode, is DC.
#[inline]
pub fn pred_chroma(plane: &mut [u8], at: usize, down: usize, mode: u8, near: (bool, bool)) -> bool {
    kernel!(pred_chroma(plane, at, down, mode, near))
}

/// Which samples beside an 8×8 luma block its prediction may use: the
/// eight above, the eight to their right, the eight to the left, and the
/// one at the corner of the row above and the column to the left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Near8 {
    pub above: bool,
    pub above_right: bool,
    pub left: bool,
    pub corner: bool,
}

/// Predicts the 8×8 luma block of `plane` whose first sample is at `at`,
/// with rows `down` apart, in Intra_8x8 `mode` (clause 8.3.2), from the
/// samples beside it that `near` allows, after the filter of clause
/// 8.3.2.2.1. Missing samples above and right are the last one above
/// repeated; a mode whose samples are not there, or any other mode, is DC,
/// and DC with neither edge is 128. `false` means the block, the sixteen
/// samples above it with their corner, and the column to its left are not
/// all in the plane, or the rows overlap, and nothing was done.
#[inline]
pub fn pred8(plane: &mut [u8], at: usize, down: usize, mode: u8, near: Near8) -> bool {
    kernel!(pred8(plane, at, down, mode, near))
}

/// Where the corner above and left of an 8×8 block is, when the block,
/// the column to its left, and the corner with the sixteen samples after
/// it are all in a plane of `len` samples with rows `down` apart.
#[inline(always)]
pub(crate) fn corner8_at(len: usize, at: usize, down: usize) -> Option<usize> {
    let start = at.checked_sub(down.checked_add(1)?)?;
    let top_end = start.checked_add(17)?;
    let end = 7usize.checked_mul(down)?.checked_add(at)?.checked_add(8)?;
    (down > 8 && top_end <= len && end <= len).then_some(start)
}

/// The edges an 8×8 prediction works from, after the filter: sixteen
/// samples above, eight to the left, top first, and the corner.
pub(crate) struct Edges8 {
    pub(crate) above: [u8; 16],
    pub(crate) left: [u8; 8],
    pub(crate) corner: u8,
}

/// The (a + 2b + c + 2) / 4 of three samples.
#[inline(always)]
pub(crate) fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((u16::from(a) + 2 * u16::from(b) + u16::from(c) + 2) >> 2) as u8
}

/// The (a + b + 1) / 2 of two samples.
#[inline(always)]
pub(crate) fn avg2(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b) + 1) >> 1) as u8
}

/// Sample (`x`, `y`) of an 8×8 block in Intra_8x8 vertical right (5),
/// horizontal down (6) or horizontal up (8) from its filtered `edges`,
/// equations 8-91 to 8-95: the modes the vector version also works out a
/// sample at a time.
#[inline(always)]
pub(crate) fn sample8(mode: u8, x: usize, y: usize, edges: &Edges8) -> u8 {
    let Edges8 {
        above,
        left,
        corner,
    } = edges;
    let up = |index: isize| -> u8 { usize::try_from(index).map_or(*corner, |index| above[index]) };
    let side = |index: isize| -> u8 { usize::try_from(index).map_or(*corner, |index| left[index]) };
    let (x, y) = (x as isize, y as isize);
    match mode {
        5 => {
            let z = 2 * x - y;
            let index = x - (y >> 1);
            if z >= 0 && z & 1 == 0 {
                avg2(up(index - 1), up(index))
            } else if z > 0 {
                avg3(up(index - 2), up(index - 1), up(index))
            } else if z == -1 {
                avg3(left[0], *corner, above[0])
            } else {
                let index = y - 2 * x;
                avg3(side(index - 1), side(index - 2), side(index - 3))
            }
        }
        6 => {
            let z = 2 * y - x;
            let index = y - (x >> 1);
            if z >= 0 && z & 1 == 0 {
                avg2(side(index - 1), side(index))
            } else if z > 0 {
                avg3(side(index - 2), side(index - 1), side(index))
            } else if z == -1 {
                avg3(left[0], *corner, above[0])
            } else {
                let index = x - 2 * y;
                avg3(up(index - 1), up(index - 2), up(index - 3))
            }
        }
        _ => {
            let z = x + 2 * y;
            let base = (y + (x >> 1)) as usize;
            match z {
                0..=12 if z & 1 == 0 => avg2(left[base], left[base + 1]),
                1..=11 => avg3(left[base], left[base + 1], left[base + 2]),
                13 => avg3(left[6], left[7], left[7]),
                _ => left[7],
            }
        }
    }
}

/// Where the corner sample above and left of a block of `N` samples a
/// side is, when the block, the row above and the column to the left are
/// all in a plane of `len` samples with rows `down` apart.
#[inline(always)]
pub(crate) fn corner_at<const N: usize>(len: usize, at: usize, down: usize) -> Option<usize> {
    let start = at.checked_sub(down.checked_add(1)?)?;
    let end = N
        .checked_mul(down)?
        .checked_add(start)?
        .checked_add(N + 1)?;
    (down > N && end <= len).then_some(start)
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
    }

    /// Planes of random, extreme and smooth samples, so the plane mode
    /// reaches past both ends of a sample.
    fn plane(rng: &mut Lcg, len: usize) -> Vec<u8> {
        let kind = rng.next() % 4;
        (0..len)
            .map(|i| match kind {
                0 => rng.next() as u8,
                1 => [0, 255][(rng.next() % 2) as usize],
                2 => (i % 251) as u8,
                _ => 255 - (i % 239) as u8,
            })
            .collect()
    }

    #[test]
    fn the_16x16_predictions_match_the_plain_ones() {
        let mut rng = Lcg(41);
        let mut done = 0;
        for round in 0..40_000u32 {
            let down = 15 + (rng.next() % 12) as usize;
            let before = plane(&mut rng, down * 19);
            let at = (rng.next() as usize) % (down * 3);
            let mode = (rng.next() % 6) as u8;
            let near = (rng.next().is_multiple_of(2), rng.next().is_multiple_of(2));
            let (mut fast, mut plain) = (before.clone(), before.clone());
            let ok = super::pred16(&mut fast, at, down, mode, near);
            assert_eq!(
                ok,
                scalar::pred16(&mut plain, at, down, mode, near),
                "round {round}"
            );
            assert_eq!(fast, plain, "round {round}: mode {mode} near {near:?}");
            if ok {
                done += 1;
            } else {
                assert_eq!(fast, before);
            }
        }
        assert!(done > 10_000, "{done}");
    }

    #[test]
    fn the_chroma_predictions_match_the_plain_ones() {
        let mut rng = Lcg(43);
        let mut done = 0;
        for round in 0..40_000u32 {
            let down = 7 + (rng.next() % 12) as usize;
            let before = plane(&mut rng, down * 11);
            let at = (rng.next() as usize) % (down * 3);
            let mode = (rng.next() % 6) as u8;
            let near = (rng.next().is_multiple_of(2), rng.next().is_multiple_of(2));
            let (mut fast, mut plain) = (before.clone(), before.clone());
            let ok = super::pred_chroma(&mut fast, at, down, mode, near);
            assert_eq!(
                ok,
                scalar::pred_chroma(&mut plain, at, down, mode, near),
                "round {round}"
            );
            assert_eq!(fast, plain, "round {round}: mode {mode} near {near:?}");
            if ok {
                done += 1;
            } else {
                assert_eq!(fast, before);
            }
        }
        assert!(done > 10_000, "{done}");
    }

    #[test]
    fn the_8x8_predictions_match_the_plain_ones() {
        let mut rng = Lcg(47);
        let mut done = [0u32; 10];
        for round in 0..100_000u32 {
            let down = 8 + (rng.next() % 20) as usize;
            let before = plane(&mut rng, down * 11);
            let at = (rng.next() as usize) % (down * 3);
            let mode = (rng.next() % 10) as u8;
            let mut flag = || !rng.next().is_multiple_of(4);
            let near = super::Near8 {
                above: flag(),
                above_right: flag(),
                left: flag(),
                corner: flag(),
            };
            let (mut fast, mut plain) = (before.clone(), before.clone());
            let ok = super::pred8(&mut fast, at, down, mode, near);
            assert_eq!(
                ok,
                scalar::pred8(&mut plain, at, down, mode, near),
                "round {round}"
            );
            assert_eq!(fast, plain, "round {round}: mode {mode} near {near:?}");
            if ok {
                done[usize::from(mode)] += 1;
            } else {
                assert_eq!(fast, before);
            }
        }
        assert!(done.iter().all(|&count| count > 2_000), "{done:?}");
    }

    #[test]
    fn a_flat_edge_predicts_a_flat_block() {
        let down = 20;
        for mode in 0..4 {
            let mut plane = vec![77u8; down * 18];
            assert!(super::pred16(
                &mut plane,
                down + 1,
                down,
                mode,
                (true, true)
            ));
            assert!(plane.iter().all(|&s| s == 77), "mode {mode}");
            let mut plane = vec![33u8; down * 10];
            assert!(super::pred_chroma(
                &mut plane,
                down + 1,
                down,
                mode,
                (true, true)
            ));
            assert!(plane.iter().all(|&s| s == 33), "mode {mode}");
        }
        // The first row and column of a plane have no corner beside them.
        let mut plane = vec![0u8; down * 18];
        assert!(!super::pred16(&mut plane, down, down, 2, (false, false)));
    }
}

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

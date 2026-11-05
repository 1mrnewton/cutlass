//! Scan orders and entropy codebooks from RDD 36.

/// Progressive scan: position `i` in coded order is raster index `PROGRESSIVE[i]`.
pub(crate) const PROGRESSIVE_SCAN: [u8; 64] = [
    0, 1, 8, 9, 2, 3, 10, 11, 16, 17, 24, 25, 18, 19, 26, 27, 4, 5, 12, 20, 13, 6, 7, 14, 21, 28,
    29, 22, 15, 23, 30, 31, 32, 33, 40, 48, 41, 34, 35, 42, 49, 56, 57, 50, 43, 36, 37, 44, 51, 58,
    59, 52, 45, 38, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Interlaced scan, tuned for the taller vertical spectrum of a field.
pub(crate) const INTERLACED_SCAN: [u8; 64] = [
    0, 8, 1, 9, 16, 24, 17, 25, 2, 10, 3, 11, 18, 26, 19, 27, 32, 40, 33, 34, 41, 48, 56, 49, 42,
    35, 43, 50, 57, 58, 51, 59, 4, 12, 5, 6, 13, 20, 28, 21, 14, 7, 15, 22, 29, 36, 44, 37, 30, 23,
    31, 38, 45, 52, 60, 53, 46, 39, 47, 54, 61, 62, 55, 63,
];

/// Codebook of a slice's first DC coefficient: exp-Golomb order 5.
pub(crate) const FIRST_DC_CODEBOOK: u8 = 0xB8;

/// DC difference codebook, by the previous difference symbol capped at 6.
pub(crate) const DC_CODEBOOK: [u8; 7] = [0x04, 0x28, 0x28, 0x4D, 0x4D, 0x70, 0x70];

/// The previous DC symbol a slice starts from.
pub(crate) const FIRST_DC_SYMBOL: u32 = 5;

/// Run codebook, by the previous run capped at 15.
pub(crate) const RUN_CODEBOOK: [u8; 16] = [
    0x06, 0x06, 0x05, 0x05, 0x04, 0x29, 0x29, 0x29, 0x29, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x4C,
];

/// The previous run a component starts from.
pub(crate) const FIRST_RUN: u32 = 4;

/// Level codebook, by the previous level magnitude less one, capped at 8.
pub(crate) const LEVEL_CODEBOOK: [u8; 9] = [0x0A, 0x05, 0x06, 0x04, 0x28, 0x28, 0x28, 0x28, 0x4C];

/// The previous level magnitude a component starts from.
pub(crate) const FIRST_LEVEL: u32 = 2;

/// Run codebook for a previous run.
#[inline]
pub(crate) fn run_codebook(previous: u32) -> u8 {
    RUN_CODEBOOK[previous.min(15) as usize]
}

/// Level codebook for a previous level magnitude, which is at least 1.
#[inline]
pub(crate) fn level_codebook(previous: u32) -> u8 {
    LEVEL_CODEBOOK[previous.clamp(1, 9) as usize - 1]
}

/// DC difference codebook for a previous symbol.
#[inline]
pub(crate) fn dc_codebook(previous: u32) -> u8 {
    DC_CODEBOOK[previous.min(6) as usize]
}

/// qScale for a quantization index of 1 to 224.
#[inline]
pub(crate) fn qscale(index: u8) -> u32 {
    let index = u32::from(index);
    if index <= 128 {
        index
    } else {
        128 + 4 * (index - 128)
    }
}

#[cfg(test)]
mod tests {
    use super::{INTERLACED_SCAN, PROGRESSIVE_SCAN, level_codebook, qscale, run_codebook};

    fn is_permutation(scan: &[u8; 64]) -> bool {
        let mut seen = [false; 64];
        for &index in scan {
            seen[usize::from(index)] = true;
        }
        seen.iter().all(|&s| s)
    }

    #[test]
    fn both_scans_visit_every_coefficient_once() {
        assert!(is_permutation(&PROGRESSIVE_SCAN));
        assert!(is_permutation(&INTERLACED_SCAN));
        assert_eq!(PROGRESSIVE_SCAN[0], 0);
        assert_eq!(INTERLACED_SCAN[0], 0);
    }

    #[test]
    fn qscale_follows_table_15() {
        assert_eq!(qscale(1), 1);
        assert_eq!(qscale(128), 128);
        assert_eq!(qscale(129), 132);
        assert_eq!(qscale(130), 136);
        assert_eq!(qscale(223), 508);
        assert_eq!(qscale(224), 512);
    }

    #[test]
    fn adaptive_codebooks_cap_their_history() {
        assert_eq!(run_codebook(0), 0x06);
        assert_eq!(run_codebook(15), 0x4C);
        assert_eq!(run_codebook(400), 0x4C);
        assert_eq!(level_codebook(1), 0x0A);
        assert_eq!(level_codebook(2), 0x05);
        assert_eq!(level_codebook(9), 0x4C);
        assert_eq!(level_codebook(70_000), 0x4C);
    }
}

//! Quantizer matrix from the specification table `Quantizer_Matrix`.
//!
//! The bytes are that table, flattened as level, then plane (luma then chroma).
//! Level 15 means the matrix is not applied. A transform type at or above
//! identity also leaves the quantizer unchanged.

const TOTAL: usize = 3344;

/// `Qm_Offset` for the nineteen transform sizes.
const OFFSET: [usize; 19] = [
    0, 16, 80, 336, 336, 1360, 1392, 1424, 1552, 1680, 2192, 336, 336, 2704, 2768, 2832, 3088,
    1680, 2192,
];

static MATRIX: &[u8] = include_bytes!("quantizer_matrix.bin");

/// `Round2(q * Quantizer_Matrix[level][chroma][Qm_Offset[tx] + row * tw + col], 5)`.
pub fn scale(q: i32, level: u8, chroma: bool, tx: usize, row: usize, col: usize, tw: usize) -> i32 {
    if level >= 15 || tx >= OFFSET.len() {
        return q;
    }
    let at = OFFSET[tx] + row.saturating_mul(tw) + col;
    let index = (usize::from(level) * 2 + usize::from(chroma)) * TOTAL + at;
    let Some(weight) = MATRIX.get(index) else {
        return q;
    };
    (q.saturating_mul(i32::from(*weight)) + 16) >> 5
}

#[cfg(test)]
mod tests {
    #[test]
    fn level_zero_luma_4x4_matches_the_specification() {
        assert_eq!(super::MATRIX.len(), 15 * 2 * 3344);
        assert_eq!(
            &super::MATRIX[..16],
            &[
                32, 43, 73, 97, 43, 67, 94, 110, 73, 94, 137, 150, 97, 110, 150, 200
            ]
        );
        assert_eq!(super::scale(32, 0, false, 0, 0, 1, 4), (32 * 43 + 16) >> 5);
        assert_eq!(super::scale(32, 15, false, 0, 0, 1, 4), 32);
    }
}

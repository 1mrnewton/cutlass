//! Intra prediction, RFC 6386 section 12.
//!
//! Every predictor reads only the edges it is given, so the caller decides
//! what lies outside the frame: 127 above the top row (the corner too),
//! 129 left of the left column.

use crate::transform::clamp255;

/// Whole-block modes, numbered as the key frame trees leave them.
pub(crate) const DC_PRED: u8 = 0;
pub(crate) const V_PRED: u8 = 1;
pub(crate) const H_PRED: u8 = 2;
pub(crate) const TM_PRED: u8 = 3;
pub(crate) const B_PRED: u8 = 4;

/// Subblock modes, section 11.2.
pub(crate) const B_DC_PRED: u8 = 0;
pub(crate) const B_TM_PRED: u8 = 1;
pub(crate) const B_VE_PRED: u8 = 2;
pub(crate) const B_HE_PRED: u8 = 3;
pub(crate) const B_LD_PRED: u8 = 4;
pub(crate) const B_RD_PRED: u8 = 5;
pub(crate) const B_VR_PRED: u8 = 6;
pub(crate) const B_VL_PRED: u8 = 7;
pub(crate) const B_HD_PRED: u8 = 8;
pub(crate) const B_HU_PRED: u8 = 9;

/// The edges around one square block.
pub(crate) struct Edges<'a> {
    pub above: &'a [u8],
    pub left: &'a [u8],
    pub corner: u8,
    /// Whether `above` and `left` are real pixels, for DC only.
    pub has_above: bool,
    pub has_left: bool,
}

/// Fill the `size`-square `out` (row stride `size`) with a whole-block
/// prediction (sections 12.2 and 12.3).
pub(crate) fn predict_block(mode: u8, edges: &Edges<'_>, size: usize, out: &mut [u8]) {
    match mode {
        V_PRED => {
            for row in out.chunks_exact_mut(size) {
                row.copy_from_slice(&edges.above[..size]);
            }
        }
        H_PRED => {
            for (row, &left) in out.chunks_exact_mut(size).zip(edges.left) {
                row.fill(left);
            }
        }
        TM_PRED => {
            for (row, &left) in out.chunks_exact_mut(size).zip(edges.left) {
                for (pixel, &above) in row.iter_mut().zip(edges.above) {
                    *pixel = clamp255(i32::from(left) + i32::from(above) - i32::from(edges.corner));
                }
            }
        }
        _ => {
            let shift = size.trailing_zeros();
            let sum = |values: &[u8]| values[..size].iter().map(|&v| u32::from(v)).sum::<u32>();
            let value = match (edges.has_above, edges.has_left) {
                (true, true) => (sum(edges.above) + sum(edges.left) + (1 << shift)) >> (shift + 1),
                (true, false) => (sum(edges.above) + (1 << (shift - 1))) >> shift,
                (false, true) => (sum(edges.left) + (1 << (shift - 1))) >> shift,
                (false, false) => 128,
            };
            out.fill(u8::try_from(value).unwrap_or(u8::MAX));
        }
    }
}

fn avg2(x: u8, y: u8) -> u8 {
    u8::try_from((u16::from(x) + u16::from(y) + 1) >> 1).unwrap_or(u8::MAX)
}

fn avg3(x: u8, y: u8, z: u8) -> u8 {
    u8::try_from((u16::from(x) + 2 * u16::from(y) + u16::from(z) + 2) >> 2).unwrap_or(u8::MAX)
}

/// Predict one 4x4 luma subblock (section 12.3).
///
/// `above` is the corner pixel then eight above (four over the block, four
/// over and to the right); `left` is the four to the left.
pub(crate) fn predict_subblock(mode: u8, above: &[u8; 9], left: &[u8; 4]) -> [[u8; 4]; 4] {
    let p = above[0];
    let a = |i: usize| above[i + 1];
    let l = |i: usize| left[i];
    // E: the left column bottom-up, the corner, then the row above.
    let e = [l(3), l(2), l(1), l(0), p, a(0), a(1), a(2), a(3)];
    let mut b = [[0_u8; 4]; 4];
    match mode {
        B_TM_PRED => {
            for (r, row) in b.iter_mut().enumerate() {
                for (c, pixel) in row.iter_mut().enumerate() {
                    *pixel = clamp255(i32::from(l(r)) + i32::from(a(c)) - i32::from(p));
                }
            }
        }
        B_VE_PRED => {
            for c in 0..4 {
                let before = if c == 0 { p } else { a(c - 1) };
                let value = avg3(before, a(c), a(c + 1));
                for row in &mut b {
                    row[c] = value;
                }
            }
        }
        B_HE_PRED => {
            let rows = [
                avg3(p, l(0), l(1)),
                avg3(l(0), l(1), l(2)),
                avg3(l(1), l(2), l(3)),
                avg3(l(2), l(3), l(3)),
            ];
            for (row, value) in b.iter_mut().zip(rows) {
                row.fill(value);
            }
        }
        B_LD_PRED => {
            for (r, row) in b.iter_mut().enumerate() {
                for (c, pixel) in row.iter_mut().enumerate() {
                    let i = r + c;
                    *pixel = if i == 6 {
                        avg3(a(6), a(7), a(7))
                    } else {
                        avg3(a(i), a(i + 1), a(i + 2))
                    };
                }
            }
        }
        B_RD_PRED => {
            // Down-right diagonals: index 4 runs through the corner.
            for (r, row) in b.iter_mut().enumerate() {
                for (c, pixel) in row.iter_mut().enumerate() {
                    let i = 4 + c - r;
                    *pixel = avg3(e[i - 1], e[i], e[i + 1]);
                }
            }
        }
        B_VR_PRED => {
            b[3][0] = avg3(e[1], e[2], e[3]);
            b[2][0] = avg3(e[2], e[3], e[4]);
            b[3][1] = avg3(e[3], e[4], e[5]);
            b[1][0] = b[3][1];
            b[2][1] = avg2(e[4], e[5]);
            b[0][0] = b[2][1];
            b[3][2] = avg3(e[4], e[5], e[6]);
            b[1][1] = b[3][2];
            b[2][2] = avg2(e[5], e[6]);
            b[0][1] = b[2][2];
            b[3][3] = avg3(e[5], e[6], e[7]);
            b[1][2] = b[3][3];
            b[2][3] = avg2(e[6], e[7]);
            b[0][2] = b[2][3];
            b[1][3] = avg3(e[6], e[7], e[8]);
            b[0][3] = avg2(e[7], e[8]);
        }
        B_VL_PRED => {
            b[0][0] = avg2(a(0), a(1));
            b[1][0] = avg3(a(0), a(1), a(2));
            b[2][0] = avg2(a(1), a(2));
            b[0][1] = b[2][0];
            b[1][1] = avg3(a(1), a(2), a(3));
            b[3][0] = b[1][1];
            b[2][1] = avg2(a(2), a(3));
            b[0][2] = b[2][1];
            b[3][1] = avg3(a(2), a(3), a(4));
            b[1][2] = b[3][1];
            b[2][2] = avg2(a(3), a(4));
            b[0][3] = b[2][2];
            b[3][2] = avg3(a(3), a(4), a(5));
            b[1][3] = b[3][2];
            b[2][3] = avg3(a(4), a(5), a(6));
            b[3][3] = avg3(a(5), a(6), a(7));
        }
        B_HD_PRED => {
            b[3][0] = avg2(e[0], e[1]);
            b[3][1] = avg3(e[0], e[1], e[2]);
            b[2][0] = avg2(e[1], e[2]);
            b[3][2] = b[2][0];
            b[2][1] = avg3(e[1], e[2], e[3]);
            b[3][3] = b[2][1];
            b[2][2] = avg2(e[2], e[3]);
            b[1][0] = b[2][2];
            b[2][3] = avg3(e[2], e[3], e[4]);
            b[1][1] = b[2][3];
            b[1][2] = avg2(e[3], e[4]);
            b[0][0] = b[1][2];
            b[1][3] = avg3(e[3], e[4], e[5]);
            b[0][1] = b[1][3];
            b[0][2] = avg3(e[4], e[5], e[6]);
            b[0][3] = avg3(e[5], e[6], e[7]);
        }
        B_HU_PRED => {
            b[0][0] = avg2(l(0), l(1));
            b[0][1] = avg3(l(0), l(1), l(2));
            b[0][2] = avg2(l(1), l(2));
            b[1][0] = b[0][2];
            b[0][3] = avg3(l(1), l(2), l(3));
            b[1][1] = b[0][3];
            b[1][2] = avg2(l(2), l(3));
            b[2][0] = b[1][2];
            b[1][3] = avg3(l(2), l(3), l(3));
            b[2][1] = b[1][3];
            b[2][2] = l(3);
            b[2][3] = l(3);
            b[3] = [l(3); 4];
        }
        // B_DC_PRED, and any other value the tree cannot produce.
        _ => {
            let sum: u32 = (0..4).map(|i| u32::from(a(i)) + u32::from(l(i))).sum();
            let value = u8::try_from((sum + 4) >> 3).unwrap_or(u8::MAX);
            b = [[value; 4]; 4];
        }
    }
    b
}

/// The subblock mode a whole-block luma mode stands for when the next
/// subblock's context is read (section 11.3, item 4).
pub(crate) fn implied_subblock_mode(mode: u8) -> u8 {
    match mode {
        V_PRED => B_VE_PRED,
        H_PRED => B_HE_PRED,
        TM_PRED => B_TM_PRED,
        DC_PRED => B_DC_PRED,
        _ => B_DC_PRED,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edges<'a>(above: &'a [u8], left: &'a [u8], has: (bool, bool)) -> Edges<'a> {
        Edges {
            above,
            left,
            corner: 100,
            has_above: has.0,
            has_left: has.1,
        }
    }

    #[test]
    fn whole_block_modes_fill_from_their_edges() {
        let above: Vec<u8> = (0..16).map(|v| v * 10).collect();
        let left: Vec<u8> = (0..16).map(|v| 200 - v * 5).collect();
        let mut out = [0_u8; 256];
        predict_block(V_PRED, &edges(&above, &left, (true, true)), 16, &mut out);
        assert_eq!(&out[16 * 7..16 * 8], &above[..]);
        predict_block(H_PRED, &edges(&above, &left, (true, true)), 16, &mut out);
        assert_eq!(out[16 * 3 + 9], left[3]);
        predict_block(TM_PRED, &edges(&above, &left, (true, true)), 16, &mut out);
        // left + above - corner.
        assert_eq!(out[16 * 2 + 15], 240);
        assert_eq!(out[16 * 15], 25);
        // DC: both edges, one edge, none.
        let sum: u32 = above.iter().chain(&left).map(|&v| u32::from(v)).sum();
        predict_block(DC_PRED, &edges(&above, &left, (true, true)), 16, &mut out);
        assert_eq!(u32::from(out[77]), (sum + 16) >> 5);
        predict_block(DC_PRED, &edges(&above, &left, (false, true)), 16, &mut out);
        let left_sum: u32 = left.iter().map(|&v| u32::from(v)).sum();
        assert_eq!(u32::from(out[0]), (left_sum + 8) >> 4);
        let mut chroma = [0_u8; 64];
        predict_block(
            DC_PRED,
            &edges(&above, &left, (false, false)),
            8,
            &mut chroma,
        );
        assert_eq!(chroma, [128; 64]);
    }

    #[test]
    fn subblock_modes_follow_the_section_12_3_formulas() {
        let above = [50, 10, 20, 30, 40, 50, 60, 70, 80];
        let left = [15, 25, 35, 45];
        let dc = predict_subblock(B_DC_PRED, &above, &left);
        assert_eq!(
            dc[2][2],
            ((10 + 20 + 30 + 40 + 15 + 25 + 35 + 45 + 4) >> 3) as u8
        );
        let ve = predict_subblock(B_VE_PRED, &above, &left);
        assert_eq!(ve[3][0], avg3(50, 10, 20));
        assert_eq!(ve[0][3], avg3(30, 40, 50));
        let he = predict_subblock(B_HE_PRED, &above, &left);
        assert_eq!(he[3], [avg3(35, 45, 45); 4]);
        let ld = predict_subblock(B_LD_PRED, &above, &left);
        assert_eq!(ld[3][3], avg3(70, 80, 80));
        assert_eq!(ld[1][2], avg3(40, 50, 60));
        let rd = predict_subblock(B_RD_PRED, &above, &left);
        assert_eq!(rd[0][0], avg3(15, 50, 10));
        assert_eq!(rd[3][0], avg3(45, 35, 25));
        assert_eq!(rd[0][3], avg3(20, 30, 40));
        let hu = predict_subblock(B_HU_PRED, &above, &left);
        assert_eq!(hu[3], [45; 4]);
        assert_eq!(hu[0][0], avg2(15, 25));
        let tm = predict_subblock(B_TM_PRED, &above, &left);
        assert_eq!(tm[0][0], 0);
        assert_eq!(tm[3][3], 35);
        for mode in 0..10 {
            // Every mode fills the whole block from in-range edges.
            let block = predict_subblock(mode, &[127; 9], &[129; 4]);
            assert!(block.iter().flatten().all(|&v| (127..=129).contains(&v)));
        }
    }
}

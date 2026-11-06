//! Intra prediction (RFC 6386 Section 12).
//!
//! Prediction works in a small buffer that holds one block and its edges:
//! row 0 is the row above (top-left first, then the block, then four
//! above-right pixels), column 0 is the column to the left. Outside the
//! frame the row above reads 127 and the column to the left reads 129.

/// Intra 16×16 and chroma modes.
pub const DC_PRED: u8 = 0;
pub const V_PRED: u8 = 1;
pub const H_PRED: u8 = 2;
pub const TM_PRED: u8 = 3;
pub const B_PRED: u8 = 4;

/// Sub-block modes.
pub const B_DC: u8 = 0;
pub const B_TM: u8 = 1;
pub const B_VE: u8 = 2;
pub const B_HE: u8 = 3;
pub const B_LD: u8 = 4;
pub const B_RD: u8 = 5;
pub const B_VR: u8 = 6;
pub const B_VL: u8 = 7;
pub const B_HD: u8 = 8;
pub const B_HU: u8 = 9;

/// Width of the luma work buffer: left column, 16 pixels, 4 above-right.
pub const LUMA_STRIDE: usize = 21;
/// Luma work buffer size: the above row plus 16 rows.
pub const LUMA_BUF: usize = LUMA_STRIDE * 17;
/// Width of a chroma work buffer: left column plus 8 pixels.
pub const CHROMA_STRIDE: usize = 9;
pub const CHROMA_BUF: usize = CHROMA_STRIDE * 9;

/// Fills the edges of a work buffer for the block at `(bx, by)` in a plane.
///
/// `size` is 16 or 8. `edge_right` is the above-right pixel count (4 for
/// luma, 0 for chroma).
pub fn load_edges(
    buf: &mut [u8],
    plane: &[u8],
    stride: usize,
    bx: usize,
    by: usize,
    size: usize,
    edge_right: usize,
) {
    let bstride = size + 1 + edge_right;
    let x0 = bx * size;
    let y0 = by * size;
    if by == 0 {
        buf[..bstride].fill(127);
    } else {
        let above = (y0 - 1) * stride;
        buf[0] = if bx == 0 { 129 } else { plane[above + x0 - 1] };
        buf[1..=size].copy_from_slice(&plane[above + x0..above + x0 + size]);
        if edge_right > 0 {
            let right = x0 + size;
            if right + edge_right <= stride {
                buf[size + 1..bstride]
                    .copy_from_slice(&plane[above + right..above + right + edge_right]);
            } else {
                let last = plane[above + stride - 1];
                buf[size + 1..bstride].fill(last);
            }
        }
    }
    for r in 0..size {
        buf[(r + 1) * bstride] = if bx == 0 {
            129
        } else {
            plane[(y0 + r) * stride + x0 - 1]
        };
    }
}

/// Copies a reconstructed block from a work buffer into the plane.
pub fn store(
    buf: &[u8],
    plane: &mut [u8],
    stride: usize,
    bx: usize,
    by: usize,
    size: usize,
    bstride: usize,
) {
    let x0 = bx * size;
    let y0 = by * size;
    for r in 0..size {
        let src = (r + 1) * bstride + 1;
        let dst = (y0 + r) * stride + x0;
        plane[dst..dst + size].copy_from_slice(&buf[src..src + size]);
    }
}

/// Predicts a whole 16×16 or 8×8 block into its work buffer.
///
/// `has_above` and `has_left` say whether the edges lie in the frame,
/// which only DC_PRED looks at.
pub fn predict_block(
    buf: &mut [u8],
    size: usize,
    bstride: usize,
    mode: u8,
    has_above: bool,
    has_left: bool,
) {
    let shift = size.trailing_zeros();
    match mode {
        V_PRED => {
            for r in 1..=size {
                let (top, rest) = buf.split_at_mut(bstride);
                let start = (r - 1) * bstride + 1;
                rest[start..start + size].copy_from_slice(&top[1..=size]);
            }
        }
        H_PRED => {
            for r in 1..=size {
                let left = buf[r * bstride];
                buf[r * bstride + 1..r * bstride + 1 + size].fill(left);
            }
        }
        TM_PRED => {
            let corner = i32::from(buf[0]);
            for r in 1..=size {
                let left = i32::from(buf[r * bstride]) - corner;
                for c in 1..=size {
                    buf[r * bstride + c] = (left + i32::from(buf[c])).clamp(0, 255) as u8;
                }
            }
        }
        _ => {
            let above: u32 = buf[1..=size].iter().map(|&p| u32::from(p)).sum();
            let left: u32 = (1..=size).map(|r| u32::from(buf[r * bstride])).sum();
            let dc = match (has_above, has_left) {
                (true, true) => (above + left + (1 << shift)) >> (shift + 1),
                (true, false) => (above + (1 << (shift - 1))) >> shift,
                (false, true) => (left + (1 << (shift - 1))) >> shift,
                (false, false) => 128,
            } as u8;
            for r in 1..=size {
                buf[r * bstride + 1..r * bstride + 1 + size].fill(dc);
            }
        }
    }
}

/// Copies the macroblock's above-right pixels down beside sub-block rows
/// 1 to 3, where column 3 reads them (Section 12.3).
pub fn copy_above_right(buf: &mut [u8]) {
    let mut right = [0u8; 4];
    right.copy_from_slice(&buf[17..21]);
    for r in [4, 8, 12] {
        buf[r * LUMA_STRIDE + 17..r * LUMA_STRIDE + 21].copy_from_slice(&right);
    }
}

#[inline]
fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((u16::from(a) + 2 * u16::from(b) + u16::from(c) + 2) >> 2) as u8
}

#[inline]
fn avg2(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b) + 1) >> 1) as u8
}

/// Predicts sub-block `(row, col)` of the luma work buffer in place.
pub fn predict_sub(buf: &mut [u8], row: usize, col: usize, mode: u8) {
    let s = LUMA_STRIDE;
    let top = 4 * row * s + 4 * col; // index of the top-left pixel
    // a[0] is top-left, a[1..9] the above and above-right pixels.
    let mut a = [0u8; 9];
    a.copy_from_slice(&buf[top..top + 9]);
    let l = [
        buf[top + s],
        buf[top + 2 * s],
        buf[top + 3 * s],
        buf[top + 4 * s],
    ];
    let p = a[0];
    let above = |i: usize| a[i + 1];
    let mut out = [0u8; 16];
    match mode {
        B_TM => {
            for r in 0..4 {
                for c in 0..4 {
                    out[4 * r + c] =
                        (i32::from(l[r]) + i32::from(above(c)) - i32::from(p)).clamp(0, 255) as u8;
                }
            }
        }
        B_VE => {
            for c in 0..4 {
                let v = avg3(a[c], a[c + 1], a[c + 2]);
                for r in 0..4 {
                    out[4 * r + c] = v;
                }
            }
        }
        B_HE => {
            let rows = [
                avg3(p, l[0], l[1]),
                avg3(l[0], l[1], l[2]),
                avg3(l[1], l[2], l[3]),
                avg3(l[2], l[3], l[3]),
            ];
            for r in 0..4 {
                out[4 * r..4 * r + 4].fill(rows[r]);
            }
        }
        B_LD => {
            for r in 0..4 {
                for c in 0..4 {
                    let i = r + c;
                    out[4 * r + c] = if i == 6 {
                        avg3(above(6), above(7), above(7))
                    } else {
                        avg3(above(i), above(i + 1), above(i + 2))
                    };
                }
            }
        }
        B_RD => {
            // Edge: l[3], l[2], l[1], l[0], p, a0..a3.
            let e = [
                l[3],
                l[2],
                l[1],
                l[0],
                p,
                above(0),
                above(1),
                above(2),
                above(3),
            ];
            for r in 0..4 {
                for c in 0..4 {
                    let i = 4 + c - r;
                    out[4 * r + c] = avg3(e[i - 1], e[i], e[i + 1]);
                }
            }
        }
        B_VR => {
            let e = [
                l[3],
                l[2],
                l[1],
                l[0],
                p,
                above(0),
                above(1),
                above(2),
                above(3),
            ];
            out[12] = avg3(e[1], e[2], e[3]);
            out[8] = avg3(e[2], e[3], e[4]);
            out[13] = avg3(e[3], e[4], e[5]);
            out[4] = out[13];
            out[9] = avg2(e[4], e[5]);
            out[0] = out[9];
            out[14] = avg3(e[4], e[5], e[6]);
            out[5] = out[14];
            out[10] = avg2(e[5], e[6]);
            out[1] = out[10];
            out[15] = avg3(e[5], e[6], e[7]);
            out[6] = out[15];
            out[11] = avg2(e[6], e[7]);
            out[2] = out[11];
            out[7] = avg3(e[6], e[7], e[8]);
            out[3] = avg2(e[7], e[8]);
        }
        B_VL => {
            let x = |i: usize| above(i);
            out[0] = avg2(x(0), x(1));
            out[4] = avg3(x(0), x(1), x(2));
            out[8] = avg2(x(1), x(2));
            out[1] = out[8];
            out[5] = avg3(x(1), x(2), x(3));
            out[12] = out[5];
            out[9] = avg2(x(2), x(3));
            out[2] = out[9];
            out[13] = avg3(x(2), x(3), x(4));
            out[6] = out[13];
            out[10] = avg2(x(3), x(4));
            out[3] = out[10];
            out[14] = avg3(x(3), x(4), x(5));
            out[7] = out[14];
            // The last two break the pattern (Section 12.3).
            out[11] = avg3(x(4), x(5), x(6));
            out[15] = avg3(x(5), x(6), x(7));
        }
        B_HD => {
            let e = [
                l[3],
                l[2],
                l[1],
                l[0],
                p,
                above(0),
                above(1),
                above(2),
                above(3),
            ];
            out[12] = avg2(e[0], e[1]);
            out[13] = avg3(e[0], e[1], e[2]);
            out[8] = avg2(e[1], e[2]);
            out[14] = out[8];
            out[9] = avg3(e[1], e[2], e[3]);
            out[15] = out[9];
            out[10] = avg2(e[2], e[3]);
            out[4] = out[10];
            out[11] = avg3(e[2], e[3], e[4]);
            out[5] = out[11];
            out[6] = avg2(e[3], e[4]);
            out[0] = out[6];
            out[7] = avg3(e[3], e[4], e[5]);
            out[1] = out[7];
            out[2] = avg3(e[4], e[5], e[6]);
            out[3] = avg3(e[5], e[6], e[7]);
        }
        B_HU => {
            out[0] = avg2(l[0], l[1]);
            out[1] = avg3(l[0], l[1], l[2]);
            out[2] = avg2(l[1], l[2]);
            out[4] = out[2];
            out[3] = avg3(l[1], l[2], l[3]);
            out[5] = out[3];
            out[6] = avg2(l[2], l[3]);
            out[8] = out[6];
            out[7] = avg3(l[2], l[3], l[3]);
            out[9] = out[7];
            for i in [10, 11, 12, 13, 14, 15] {
                out[i] = l[3];
            }
        }
        _ => {
            let sum: u32 = (0..4).map(|i| u32::from(above(i)) + u32::from(l[i])).sum();
            out.fill(((sum + 4) >> 3) as u8);
        }
    }
    for r in 0..4 {
        let start = top + (r + 1) * s + 1;
        buf[start..start + 4].copy_from_slice(&out[4 * r..4 * r + 4]);
    }
}

#[cfg(test)]
mod tests {
    use super::{B_HD, B_HU, B_LD, B_RD, B_VL, B_VR, LUMA_BUF, LUMA_STRIDE, predict_sub};

    /// Builds sub-block 0's edges from the reference decoder's layout and
    /// returns the predicted block.
    fn run(mode: u8, above: [u8; 8], left: [u8; 4], corner: u8) -> [u8; 16] {
        let mut buf = [0u8; LUMA_BUF];
        buf[0] = corner;
        buf[1..9].copy_from_slice(&above);
        for (r, &value) in left.iter().enumerate() {
            buf[(r + 1) * LUMA_STRIDE] = value;
        }
        predict_sub(&mut buf, 0, 0, mode);
        let mut out = [0u8; 16];
        for r in 0..4 {
            out[4 * r..4 * r + 4]
                .copy_from_slice(&buf[(r + 1) * LUMA_STRIDE + 1..(r + 1) * LUMA_STRIDE + 5]);
        }
        out
    }

    // Straight transcriptions of the Section 20 formulas, kept apart from
    // the tables above so a slip in one shows up against the other.
    fn reference(mode: u8, a: [u8; 8], l: [u8; 4], p: u8) -> [u8; 16] {
        let a = |i: isize| -> i32 {
            if i < 0 {
                i32::from(p)
            } else {
                i32::from(a[i as usize])
            }
        };
        let l = |i: usize| i32::from(l[i]);
        let p = i32::from(p);
        let f3 = |x: i32, y: i32, z: i32| ((x + 2 * y + z + 2) >> 2) as u8;
        let f2 = |x: i32, y: i32| ((x + y + 1) >> 1) as u8;
        let mut o = [0u8; 16];
        let mut set = |r: usize, c: usize, v: u8| o[4 * r + c] = v;
        match mode {
            B_LD => {
                for r in 0..4 {
                    for c in 0..4 {
                        let i = (r + c) as isize;
                        let last = if i == 6 { a(7) } else { a(i + 2) };
                        set(r, c, f3(a(i), a(i + 1), last));
                    }
                }
            }
            B_RD => {
                let pr = [
                    f3(l(0), p, a(0)),
                    f3(p, a(0), a(1)),
                    f3(a(0), a(1), a(2)),
                    f3(a(1), a(2), a(3)),
                    f3(l(1), l(0), p),
                    f3(l(2), l(1), l(0)),
                    f3(l(3), l(2), l(1)),
                ];
                let grid = [[0, 1, 2, 3], [4, 0, 1, 2], [5, 4, 0, 1], [6, 5, 4, 0]];
                for r in 0..4 {
                    for c in 0..4 {
                        set(r, c, pr[grid[r][c]]);
                    }
                }
            }
            B_VR => {
                let pr = [
                    f2(p, a(0)),
                    f2(a(0), a(1)),
                    f2(a(1), a(2)),
                    f2(a(2), a(3)),
                    f3(l(0), p, a(0)),
                    f3(p, a(0), a(1)),
                    f3(a(0), a(1), a(2)),
                    f3(a(1), a(2), a(3)),
                    f3(l(1), l(0), p),
                    f3(l(2), l(1), l(0)),
                ];
                let grid = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 0, 1, 2], [9, 4, 5, 6]];
                for r in 0..4 {
                    for c in 0..4 {
                        set(r, c, pr[grid[r][c]]);
                    }
                }
            }
            B_VL => {
                let pr = [
                    f2(a(0), a(1)),
                    f2(a(1), a(2)),
                    f2(a(2), a(3)),
                    f2(a(3), a(4)),
                    f3(a(0), a(1), a(2)),
                    f3(a(1), a(2), a(3)),
                    f3(a(2), a(3), a(4)),
                    f3(a(3), a(4), a(5)),
                    f3(a(4), a(5), a(6)),
                    f3(a(5), a(6), a(7)),
                ];
                let grid = [[0, 1, 2, 3], [4, 5, 6, 7], [1, 2, 3, 8], [5, 6, 7, 9]];
                for r in 0..4 {
                    for c in 0..4 {
                        set(r, c, pr[grid[r][c]]);
                    }
                }
            }
            B_HD => {
                let pr = [
                    f2(l(0), p),
                    f3(l(0), p, a(0)),
                    f3(p, a(0), a(1)),
                    f3(a(0), a(1), a(2)),
                    f2(l(1), l(0)),
                    f3(l(1), l(0), p),
                    f2(l(2), l(1)),
                    f3(l(2), l(1), l(0)),
                    f2(l(3), l(2)),
                    f3(l(3), l(2), l(1)),
                ];
                let grid = [[0, 1, 2, 3], [4, 5, 0, 1], [6, 7, 4, 5], [8, 9, 6, 7]];
                for r in 0..4 {
                    for c in 0..4 {
                        set(r, c, pr[grid[r][c]]);
                    }
                }
            }
            _ => {
                let pr = [
                    f2(l(0), l(1)),
                    f3(l(0), l(1), l(2)),
                    f2(l(1), l(2)),
                    f3(l(1), l(2), l(3)),
                    f2(l(2), l(3)),
                    f3(l(2), l(3), l(3)),
                    l(3) as u8,
                ];
                let grid = [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 6], [6, 6, 6, 6]];
                for r in 0..4 {
                    for c in 0..4 {
                        set(r, c, pr[grid[r][c]]);
                    }
                }
            }
        }
        o
    }

    #[test]
    fn diagonal_modes_match_the_reference_formulas() {
        let mut seed = 1u32;
        for _ in 0..500 {
            let mut next = || {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (seed >> 16) as u8
            };
            let above = [
                next(),
                next(),
                next(),
                next(),
                next(),
                next(),
                next(),
                next(),
            ];
            let left = [next(), next(), next(), next()];
            let corner = next();
            for mode in [B_LD, B_RD, B_VR, B_VL, B_HD, B_HU] {
                assert_eq!(
                    run(mode, above, left, corner),
                    reference(mode, above, left, corner),
                    "mode {mode}"
                );
            }
        }
    }
}

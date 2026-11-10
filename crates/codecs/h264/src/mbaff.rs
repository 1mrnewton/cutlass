//! Neighbours in a macroblock-adaptive frame.
//!
//! A frame coded this way is a grid of macroblock pairs, each 16 samples wide
//! and 32 tall. A pair is two frame macroblocks, one above the other, or two
//! field macroblocks, the top one holding the pair's even lines and the
//! bottom one its odd lines. The decoder stores a pair's macroblocks in rows
//! `2 * pair` and `2 * pair + 1` of the macroblock grid whichever kind it is.
//!
//! What lies left of, above and diagonal to a macroblock then depends on the
//! kind of both pairs. Clause 6.4.12.2 gives the answer for a luma location
//! relative to the macroblock's first sample, and [`locate`] is that table.

use super::super::Mb;

/// The macroblock holding a neighbouring luma location, by its column and
/// row in the macroblock grid, and the luma row of the location inside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Near {
    pub(super) mx: usize,
    pub(super) my: usize,
    pub(super) row: usize,
}

/// Clause 6.4.12.2, Table 6-4, for luma: the macroblock that holds the
/// location (`xn`, `yn`) relative to the first luma sample of the macroblock
/// at (`mx`, `my`), which is a field macroblock when `field` is set.
///
/// `grid` is the picture's macroblocks and how many there are in a row.
/// `xn` is −1 to 16 and `yn` is −1 to 15. `None` is a location outside the
/// picture, or one the table calls unavailable. A macroblock of another slice
/// or one not decoded yet is still returned; the caller decides that.
pub(super) fn locate(
    grid: (&[Mb], usize),
    at: (usize, usize),
    field: bool,
    xn: i32,
    yn: i32,
) -> Option<Near> {
    let (mbs, mb_w) = grid;
    let (mx, my) = at;
    let top = my.is_multiple_of(2);
    // Grid row of the top macroblock of this pair.
    let pair = my & !1;
    let field_pair = |px: usize, row: usize| mbs.get(row * mb_w + px).is_some_and(|mb| mb.field_mb);
    let near = |mx: usize, my: usize, row: usize| Some(Near { mx, my, row });
    if xn < 0 {
        let left = mx.checked_sub(1)?;
        let Ok(yn) = usize::try_from(yn) else {
            // Above and to the left.
            return match (field, top) {
                (false, true) => near(left, pair.checked_sub(1)?, 15),
                (false, false) => {
                    if field_pair(left, pair) {
                        near(left, pair + 1, 7)
                    } else {
                        near(left, pair, 15)
                    }
                }
                (true, true) => {
                    let above = pair.checked_sub(2)?;
                    if field_pair(left, above) {
                        near(left, above, 15)
                    } else {
                        near(left, above + 1, 14)
                    }
                }
                (true, false) => near(left, pair.checked_sub(1)?, 15),
            };
        };
        if yn > 15 {
            return None;
        }
        let beside = field_pair(left, pair);
        return match (field, beside) {
            // Two frame pairs or two field pairs: the macroblock in the same
            // half of the pair, at the same row.
            (false, false) | (true, true) => near(left, my, yn),
            // A frame macroblock beside a field pair: its lines alternate
            // between the two field macroblocks.
            (false, true) => {
                let line = yn + if top { 0 } else { 16 };
                near(left, pair + (yn & 1), line >> 1)
            }
            // A field macroblock beside a frame pair: its eight upper lines
            // lie beside the top macroblock and the rest beside the bottom.
            (true, false) => {
                let line = 2 * yn + usize::from(!top);
                near(left, pair + line / 16, line % 16)
            }
        };
    }
    if xn < 16 {
        let Ok(yn) = usize::try_from(yn) else {
            return match (field, top) {
                (false, true) => near(mx, pair.checked_sub(1)?, 15),
                (false, false) => near(mx, pair, 15),
                (true, true) => {
                    let above = pair.checked_sub(2)?;
                    if field_pair(mx, above) {
                        near(mx, above, 15)
                    } else {
                        near(mx, above + 1, 14)
                    }
                }
                (true, false) => near(mx, pair.checked_sub(1)?, 15),
            };
        };
        return (yn < 16).then_some(Near { mx, my, row: yn });
    }
    // Above and to the right. Nothing to the right at this height or below
    // is decoded yet, and neither is the pair right of a bottom frame
    // macroblock's upper edge.
    if yn >= 0 || mx + 1 >= mb_w {
        return None;
    }
    let right = mx + 1;
    match (field, top) {
        (false, true) => near(right, pair.checked_sub(1)?, 15),
        (false, false) => None,
        (true, true) => {
            let above = pair.checked_sub(2)?;
            if field_pair(right, above) {
                near(right, above, 15)
            } else {
                near(right, above + 1, 14)
            }
        }
        (true, false) => near(right, pair.checked_sub(1)?, 15),
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::Mb;
    use super::Near;

    /// A 3×4 grid of macroblocks: two rows of three pairs. `fields` names
    /// the pairs, by column and pair row, that are field pairs.
    fn grid(fields: &[(usize, usize)]) -> Vec<Mb> {
        let mut mbs = vec![Mb::new(0); 12];
        for &(px, pair) in fields {
            for half in 0..2 {
                mbs[(pair * 2 + half) * 3 + px].field_mb = true;
            }
        }
        mbs
    }

    fn locate(mbs: &[Mb], at: (usize, usize), field: bool, xn: i32, yn: i32) -> Option<Near> {
        super::locate((mbs, 3), at, field, xn, yn)
    }

    fn at(mx: usize, my: usize, row: usize) -> Option<Near> {
        Some(Near { mx, my, row })
    }

    #[test]
    fn frame_pairs_look_like_a_plain_frame() {
        let pic = grid(&[]);
        // Top macroblock of the middle pair in the second pair row.
        assert_eq!(locate(&pic, (1, 2), false, -1, 5), at(0, 2, 5));
        assert_eq!(locate(&pic, (1, 2), false, 3, -1), at(1, 1, 15));
        assert_eq!(locate(&pic, (1, 2), false, 16, -1), at(2, 1, 15));
        assert_eq!(locate(&pic, (1, 2), false, -1, -1), at(0, 1, 15));
        // Its bottom macroblock: the top one is above it, and nothing to
        // the upper right is decoded.
        assert_eq!(locate(&pic, (1, 3), false, 3, -1), at(1, 2, 15));
        assert_eq!(locate(&pic, (1, 3), false, -1, -1), at(0, 2, 15));
        assert_eq!(locate(&pic, (1, 3), false, 16, -1), None);
        assert_eq!(locate(&pic, (1, 3), false, -1, 9), at(0, 3, 9));
    }

    #[test]
    fn the_picture_edge_has_no_neighbour() {
        let pic = grid(&[(0, 0)]);
        assert_eq!(locate(&pic, (0, 0), true, -1, 0), None);
        assert_eq!(locate(&pic, (0, 0), true, 0, -1), None);
        // The bottom field macroblock of the first pair row starts on the
        // picture's second line: nothing is above it either.
        assert_eq!(locate(&pic, (0, 1), true, 0, -1), None);
        assert_eq!(locate(&pic, (2, 2), false, 16, -1), None);
        assert_eq!(locate(&pic, (1, 2), false, 16, 0), None);
    }

    #[test]
    fn a_frame_macroblock_beside_a_field_pair_alternates() {
        let pic = grid(&[(0, 1)]);
        // Top frame macroblock: even lines are the top field's, odd lines
        // the bottom field's, each at half the height.
        assert_eq!(locate(&pic, (1, 2), false, -1, 0), at(0, 2, 0));
        assert_eq!(locate(&pic, (1, 2), false, -1, 1), at(0, 3, 0));
        assert_eq!(locate(&pic, (1, 2), false, -1, 12), at(0, 2, 6));
        assert_eq!(locate(&pic, (1, 2), false, -1, 15), at(0, 3, 7));
        // Bottom frame macroblock: the lower half of each field.
        assert_eq!(locate(&pic, (1, 3), false, -1, 0), at(0, 2, 8));
        assert_eq!(locate(&pic, (1, 3), false, -1, 15), at(0, 3, 15));
        // Its upper-left sample is the last line above it, which is the
        // bottom field's line 7.
        assert_eq!(locate(&pic, (1, 3), false, -1, -1), at(0, 3, 7));
    }

    #[test]
    fn a_field_macroblock_beside_a_frame_pair_spans_both() {
        let pic = grid(&[(1, 1)]);
        // Top field: lines 0, 2, 4 … of the pair.
        assert_eq!(locate(&pic, (1, 2), true, -1, 0), at(0, 2, 0));
        assert_eq!(locate(&pic, (1, 2), true, -1, 7), at(0, 2, 14));
        assert_eq!(locate(&pic, (1, 2), true, -1, 8), at(0, 3, 0));
        assert_eq!(locate(&pic, (1, 2), true, -1, 15), at(0, 3, 14));
        // Bottom field: lines 1, 3, 5 ….
        assert_eq!(locate(&pic, (1, 3), true, -1, 0), at(0, 2, 1));
        assert_eq!(locate(&pic, (1, 3), true, -1, 7), at(0, 2, 15));
        assert_eq!(locate(&pic, (1, 3), true, -1, 8), at(0, 3, 1));
        assert_eq!(locate(&pic, (1, 3), true, -1, 15), at(0, 3, 15));
    }

    #[test]
    fn a_field_macroblock_looks_up_to_its_own_parity() {
        // Above is a frame pair: both fields end in its bottom macroblock,
        // the top field one line higher.
        let pic = grid(&[(1, 1)]);
        assert_eq!(locate(&pic, (1, 2), true, 4, -1), at(1, 1, 14));
        assert_eq!(locate(&pic, (1, 3), true, 4, -1), at(1, 1, 15));
        assert_eq!(locate(&pic, (1, 2), true, -1, -1), at(0, 1, 14));
        assert_eq!(locate(&pic, (1, 2), true, 16, -1), at(2, 1, 14));
        assert_eq!(locate(&pic, (1, 3), true, 16, -1), at(2, 1, 15));
        // Above is a field pair: each field continues its own.
        let pic = grid(&[(0, 0), (1, 0), (2, 0), (1, 1)]);
        assert_eq!(locate(&pic, (1, 2), true, 4, -1), at(1, 0, 15));
        assert_eq!(locate(&pic, (1, 3), true, 4, -1), at(1, 1, 15));
        assert_eq!(locate(&pic, (1, 2), true, -1, -1), at(0, 0, 15));
        assert_eq!(locate(&pic, (1, 3), true, -1, -1), at(0, 1, 15));
        assert_eq!(locate(&pic, (1, 2), true, 16, -1), at(2, 0, 15));
        // A frame macroblock under a field pair reads the bottom field.
        let pic = grid(&[(1, 0)]);
        assert_eq!(locate(&pic, (1, 2), false, 4, -1), at(1, 1, 15));
    }
}

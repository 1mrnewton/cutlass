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

use cutlass_decode::CodecError;

use crate::recon::{self, Sample};

use super::super::{Frame, Mb, Slice};
use super::{Block, McMotion, Mix};

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

/// Inter prediction for a field macroblock of a macroblock-adaptive frame
/// (clause 8.4.2.1). Its samples are every other line of its pair, and each
/// reference index names one field of a listed frame: index 2k is the field
/// of frame k with this macroblock's parity, index 2k + 1 the other field.
pub(super) fn predict_field<S: Sample>(
    pic: &mut Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    at: (usize, usize),
    mb: &Mb,
) -> Result<(), CodecError> {
    let (mx, my) = at;
    let parity = my & 1;
    // The macroblock's first line, counted in lines of its field.
    let top = my / 2 * 16;
    let motion = |index: usize| {
        let part = index / 8 * 2 + index % 4 / 2;
        McMotion {
            r0: mb.ref0[part],
            r1: mb.ref1[part],
            mv0: mb.mv0[index],
            mv1: mb.mv1[index],
        }
    };
    if mb.mv0 == [mb.mv0[0]; 16]
        && mb.mv1 == [mb.mv1[0]; 16]
        && mb.ref0 == [mb.ref0[0]; 4]
        && mb.ref1 == [mb.ref1[0]; 4]
    {
        let whole = Block {
            x: mx * 16,
            y: top,
            w: 16,
            h: 16,
        };
        return field_block(pic, dpb, slice, parity, whole, motion(0));
    }
    let same = |a: usize, b: usize| mb.mv0[a] == mb.mv0[b] && mb.mv1[a] == mb.mv1[b];
    for quarter in 0..4 {
        let first = quarter / 2 * 8 + quarter % 2 * 2;
        let blocks = [first, first + 1, first + 4, first + 5];
        // A reference index covers the quarter, so one vector for its four
        // blocks makes it a single 8×8 block.
        let (count, size) = if blocks.iter().all(|&index| same(first, index)) {
            (1, 8)
        } else {
            (4, 4)
        };
        for &index in blocks.iter().take(count) {
            let block = Block {
                x: mx * 16 + index % 4 * 4,
                y: top + index / 4 * 4,
                w: size,
                h: size,
            };
            field_block(pic, dpb, slice, parity, block, motion(index))?;
        }
    }
    Ok(())
}

/// One block of a field macroblock. `block.y` counts lines of the field
/// with `parity`, 0 for the top field.
fn field_block<S: Sample>(
    pic: &mut Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    parity: usize,
    block: Block,
    motion: McMotion,
) -> Result<(), CodecError> {
    let Block { x, y, w, h } = block;
    let McMotion { r0, r1, mv0, mv1 } = motion;
    let size = (pic.w, pic.h);
    let luma_mix = field_mix(pic, dpb, slice, parity, (r0, r1), -1);
    let stride = pic.w;
    if let Mix::Copy(list) = luma_mix {
        // One reference stored as it is: predict straight into the picture.
        let (reference, mv) = if list == 0 { (r0, mv0) } else { (r1, mv1) };
        let (frame, from) = field_source(dpb, slice, (list, reference), parity, size)?;
        let dst = &mut pic.y[(2 * y + parity) * stride + x..];
        field_luma(dst, 2 * stride, &frame.y, size, from, (x, y, w, h), mv);
    } else {
        let mut luma0 = [S::default(); 256];
        let mut luma1 = [S::default(); 256];
        for (list, reference, mv, out) in [(0, r0, mv0, &mut luma0), (1, r1, mv1, &mut luma1)] {
            if reference >= 0 {
                let (frame, from) = field_source(dpb, slice, (list, reference), parity, size)?;
                field_luma(out, w, &frame.y, size, from, (x, y, w, h), mv);
            }
        }
        // The field's lines are a plane of their own, twice the stride apart.
        super::store_block(
            &mut pic.y[parity * stride..],
            2 * stride,
            (x, y),
            (w, h),
            (&luma0, &luma1),
            luma_mix,
            S::default(),
        );
    }
    if pic.separate_plane {
        return Ok(());
    }
    let (origin, chroma_size) = match pic.chroma() {
        3 => ((x, y), (w, h)),
        2 => ((x / 2, y), (w / 2, h)),
        _ => ((x / 2, y / 2), (w / 2, h / 2)),
    };
    let stride = pic.chroma_stride();
    for cb in [true, false] {
        let mix = field_mix(pic, dpb, slice, parity, (r0, r1), if cb { 0 } else { 1 });
        if let Mix::Copy(list) = mix {
            let (reference, mv) = if list == 0 { (r0, mv0) } else { (r1, mv1) };
            let (frame, from) = field_source(dpb, slice, (list, reference), parity, size)?;
            let plane = if cb { &mut pic.u } else { &mut pic.v };
            let dst = &mut plane[(2 * origin.1 + parity) * stride + origin.0..];
            field_chroma(dst, 2 * stride, frame, cb, (parity, from), (x, y, w, h), mv);
            continue;
        }
        let mut chroma0 = [S::default(); 256];
        let mut chroma1 = [S::default(); 256];
        for (list, reference, mv, out) in [(0, r0, mv0, &mut chroma0), (1, r1, mv1, &mut chroma1)] {
            if reference >= 0 {
                let (frame, from) = field_source(dpb, slice, (list, reference), parity, size)?;
                let block = (x, y, w, h);
                field_chroma(out, chroma_size.0, frame, cb, (parity, from), block, mv);
            }
        }
        let plane = if cb { &mut pic.u } else { &mut pic.v };
        super::store_block(
            &mut plane[parity * stride..],
            2 * stride,
            origin,
            chroma_size,
            (&chroma0, &chroma1),
            mix,
            S::mid(),
        );
    }
    Ok(())
}

/// The frame a field reference index names in `list`, and which of its
/// fields: 0 the top, 1 the bottom. An even index is the field with the
/// macroblock's own `parity`.
fn field_source<'a, S: Sample>(
    dpb: &'a [Frame<S>],
    slice: &Slice<'_>,
    (list, reference): (usize, i8),
    parity: usize,
    size: (usize, usize),
) -> Result<(&'a Frame<S>, usize), CodecError> {
    let frame = super::reference_frame(dpb, slice, list, reference >> 1)?;
    // Fields are read out of whole frames of this picture's size. Pictures
    // of an interlaced sequence are never decoded on another thread.
    if frame.pending.is_some() || (frame.w, frame.h) != size || frame.field {
        return Err(CodecError::Unsupported {
            what: "field reference",
        });
    }
    Ok((frame, parity ^ usize::from(reference & 1 == 1)))
}

/// A luma block of one field of `plane`, moved by `mv`, whose vertical part
/// counts quarter lines of the field. `block` is in field lines too.
fn field_luma<S: Sample>(
    dst: &mut [S],
    dst_stride: usize,
    plane: &[S],
    (width, height): (usize, usize),
    from: usize,
    block: (usize, usize, usize, usize),
    mv: [i16; 2],
) {
    let (x, y, w, h) = block;
    let Some(field) = plane.get(from * width..) else {
        return;
    };
    let (mvx, mvy) = (i32::from(mv[0]), i32::from(mv[1]));
    recon::put_luma_block(
        dst,
        dst_stride,
        field,
        2 * width,
        (
            i32::try_from(width).unwrap_or(0),
            i32::try_from(height / 2).unwrap_or(0),
        ),
        recon::BlockMc {
            x: i32::try_from(x).unwrap_or(0) + (mvx >> 2),
            y: i32::try_from(y).unwrap_or(0) + (mvy >> 2),
            w,
            h,
            x_frac: mvx & 3,
            y_frac: mvy & 3,
        },
    );
}

/// The chroma block that goes with the luma `block` of a field macroblock.
/// `parities` is the macroblock's field, then the reference's.
fn field_chroma<S: Sample>(
    dst: &mut [S],
    dst_stride: usize,
    frame: &Frame<S>,
    cb: bool,
    parities: (usize, usize),
    block: (usize, usize, usize, usize),
    mv: [i16; 2],
) {
    let (x, y, w, h) = block;
    let (current, from) = parities;
    let plane: &[S] = if cb { &frame.u } else { &frame.v };
    let format = frame.chroma();
    if format == 3 {
        field_luma(dst, dst_stride, plane, (frame.w, frame.h), from, block, mv);
        return;
    }
    let stride = frame.w / 2;
    let Some(field) = plane.get(from * stride..) else {
        return;
    };
    let (mvx, mvy) = (i32::from(mv[0]), i32::from(mv[1]));
    let (x, y) = (i32::try_from(x).unwrap_or(0), i32::try_from(y).unwrap_or(0));
    let (lines, at) = if format == 2 {
        // 4:2:2 chroma has luma's lines: quarter-line steps, as eighths.
        let at = recon::BlockMc {
            x: x / 2 + (mvx >> 3),
            y: y + (mvy >> 2),
            w: w / 2,
            h,
            x_frac: mvx & 7,
            y_frac: (mvy << 1) & 7,
        };
        (frame.h / 2, at)
    } else {
        // Clause 8.4.1.4, Table 8-10: the two fields' chroma lines are not
        // at the same heights, so a vector into the other field is moved a
        // quarter of a chroma line: up from the top field, down from the
        // bottom one.
        let lift = if current == from {
            0
        } else if current == 0 {
            -2
        } else {
            2
        };
        let mvy = mvy + lift;
        let at = recon::BlockMc {
            x: x / 2 + (mvx >> 3),
            y: y / 2 + (mvy >> 3),
            w: w / 2,
            h: h / 2,
            x_frac: mvx & 7,
            y_frac: mvy & 7,
        };
        (frame.h / 4, at)
    };
    recon::put_chroma_block(
        dst,
        dst_stride,
        field,
        2 * stride,
        (
            i32::try_from(stride).unwrap_or(0),
            i32::try_from(lines).unwrap_or(0),
        ),
        at,
    );
}

/// How a field macroblock's two predictions are combined. Explicit weights
/// are sent for each frame of a list, and both of its fields use them
/// (clause 8.4.2.3); implicit weights follow the distances between fields.
fn field_mix<S: Sample>(
    pic: &Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    parity: usize,
    (r0, r1): (i8, i8),
    plane: i8,
) -> Mix {
    if r0 >= 0 && r1 >= 0 {
        if slice.weighted {
            Mix::Weighted(super::explicit_bi(slice, r0 >> 1, r1 >> 1, plane))
        } else if slice.pps.weighted_bipred == 2 && slice.kind == 1 {
            let (w0, w1) = field_implicit(pic, dpb, slice, parity, (r0, r1));
            Mix::Weighted(recon::BiWeight {
                w0,
                w1,
                o0: 0,
                o1: 0,
                denom: 5,
            })
        } else {
            Mix::Avg
        }
    } else if r0 >= 0 {
        super::uni_mix(slice, 0, r0 >> 1, plane)
    } else if r1 >= 0 {
        super::uni_mix(slice, 1, r1 >> 1, plane)
    } else {
        Mix::Fill
    }
}

/// Clause 8.4.2.3.1 for a field macroblock: the picture order distances are
/// between this macroblock's field and the two fields it predicts from.
fn field_implicit<S: Sample>(
    pic: &Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    parity: usize,
    (r0, r1): (i8, i8),
) -> (i32, i32) {
    let field = |list: &[usize], longs: &[bool], reference: i8| {
        let index = usize::try_from(reference >> 1).ok()?;
        let frame = dpb.get(*list.get(index)?)?;
        let from = parity ^ usize::from(reference & 1 == 1);
        Some((
            frame.field_poc[from],
            longs.get(index).copied().unwrap_or(true),
        ))
    };
    let (Some((poc0, long0)), Some((poc1, long1))) = (
        field(&slice.ref0, &slice.long0, r0),
        field(&slice.ref1, &slice.long1, r1),
    ) else {
        return (32, 32);
    };
    super::implicit_pair(pic.field_poc[parity], poc0, poc1, long0 || long1)
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

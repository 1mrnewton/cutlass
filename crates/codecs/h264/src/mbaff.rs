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

/// The block of a list-1 picture that lies where part of a macroblock of
/// this picture does: its macroblock, as a slot in the grid, and the 4×4
/// block in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Col {
    pub(super) slot: usize,
    pub(super) block: usize,
}

/// Clause 8.4.1.2.1, Table 8-8, for a macroblock-adaptive frame, which
/// always has `direct_8x8_inference_flag`: the co-located block for the 8×8
/// quarter (`qx`, `qy`) of the macroblock at grid position `at`.
///
/// When both pictures code the pair the same way it is the same
/// macroblock's corner block. A frame macroblock over a field pair takes
/// the field nearer to `current_poc`, this frame's order count, and the
/// half of it the macroblock covers. A field macroblock over a frame pair
/// has its upper half in the top macroblock and its lower half in the
/// bottom one.
pub(super) fn col_block<S: Sample>(
    col: &Frame<S>,
    current_poc: i32,
    at: (usize, usize),
    field: bool,
    (qx, qy): (usize, usize),
) -> Col {
    let (mx, my) = at;
    let pair = my & !1;
    let col_field = col
        .mb_at(pair * col.mb_w + mx)
        .is_some_and(|mb| mb.field_mb);
    let (row, block_row) = match (field, col_field) {
        (false, false) | (true, true) => (my, qy * 3),
        (false, true) => {
            let apart =
                |parity: usize| (i64::from(col.field_poc[parity]) - i64::from(current_poc)).abs();
            (pair + usize::from(apart(0) >= apart(1)), 2 * (my & 1) + qy)
        }
        (true, false) => (pair + qy, 2 * qy),
    };
    Col {
        slot: row * col.mb_w + mx,
        block: block_row * 4 + qx * 3,
    }
}

/// The three planes of the picture being deblocked, and its chroma format:
/// 1 is 4:2:0, 2 is 4:2:2, 3 is 4:4:4.
struct Planes<'a, S> {
    luma: &'a mut [S],
    cb: &'a mut [S],
    cr: &'a mut [S],
    format: u8,
}

/// Clause 8.7 for the macroblock rows `rows` of a macroblock-adaptive frame.
///
/// Macroblocks are filtered a pair at a time, top then bottom. A field
/// macroblock's edges run along its own field's lines. Where a pair meets a
/// pair of the other kind the edge between them is not one edge: see
/// [`mixed_left`] and [`under_fields`].
pub(super) fn deblock<S: Sample>(
    pic: &Frame<S>,
    format: u8,
    planes: (&mut [S], &mut [S], &mut [S]),
    rows: std::ops::Range<usize>,
) {
    let (luma, cb, cr) = planes;
    let mut planes = Planes {
        luma,
        cb,
        cr,
        format,
    };
    for pair in rows.start / 2..rows.end.min(pic.h / 16) / 2 {
        for mx in 0..pic.mb_w {
            deblock_mb(pic, &mut planes, mx, pair * 2);
            deblock_mb(pic, &mut planes, mx, pair * 2 + 1);
        }
    }
}

fn deblock_mb<S: Sample>(pic: &Frame<S>, planes: &mut Planes<'_, S>, mx: usize, my: usize) {
    let mb_w = pic.mb_w;
    let Some(cur) = pic.mb.get(my * mb_w + mx) else {
        return;
    };
    if cur.deblock == 1 {
        return;
    }
    let field = cur.field_mb;
    // No coefficients and one motion: no edge inside is filtered. For a
    // field macroblock one motion also means one field of the reference,
    // which the reference index tells, not the frame.
    let flat = super::inner_edges_flat(cur)
        && (!field || (cur.ref0 == [cur.ref0[0]; 4] && cur.ref1 == [cur.ref1[0]; 4]));
    // A slice may keep its filter inside itself.
    let allowed = |other: &&Mb| cur.deblock != 2 || other.slice == cur.slice;
    if let Some(left) = mx.checked_sub(1)
        && let Some(beside) = pic.mb.get((my & !1) * mb_w + left).filter(allowed)
    {
        if beside.field_mb == field {
            plain_edge(pic, planes, (mx, my), pic.mb.get(my * mb_w + left), 0, true);
        } else {
            mixed_left(pic, planes, (mx, my));
        }
    }
    for edge in 1..4 {
        if flat {
            break;
        }
        plain_edge(pic, planes, (mx, my), None, edge, true);
    }
    if let Ok(above) = usize::try_from(super::above_my(pic, mx, my))
        && let Some(over) = pic.mb.get(above * mb_w + mx).filter(allowed)
    {
        if !field && my.is_multiple_of(2) && over.field_mb {
            under_fields(pic, planes, (mx, my));
        } else {
            plain_edge(pic, planes, (mx, my), Some(over), 0, false);
        }
    }
    for edge in 1..4 {
        if flat {
            break;
        }
        plain_edge(pic, planes, (mx, my), None, edge, false);
    }
}

/// An edge whose two sides run along the same lines: one inside the
/// macroblock, or its left or upper edge with `across` on the other side.
/// `across` is the macroblock [`super::edge_qp`] finds there too.
fn plain_edge<S: Sample>(
    pic: &Frame<S>,
    planes: &mut Planes<'_, S>,
    (mx, my): (usize, usize),
    across: Option<&Mb>,
    edge: usize,
    vertical: bool,
) {
    let Some(cur) = pic.mb.get(my * pic.mb_w + mx) else {
        return;
    };
    let bs = strengths(cur, across, edge, vertical);
    if bs == [0; 4] {
        return;
    }
    let off = (i32::from(cur.off_a), i32::from(cur.off_b));
    let span = super::luma_span(my, cur.field_mb);
    // An 8×8 transform has no luma edge at 4 or 12.
    if !(cur.transform8 && edge % 2 == 1) {
        let at = if vertical {
            (mx * 16 + edge * 4, span.y)
        } else {
            (mx * 16, span.at(edge * 4))
        };
        let qp = super::edge_qp(pic, mx, my, true, true, vertical, edge == 0);
        let kind = (vertical, false, 4, span.step);
        super::filter_groups(planes.luma, pic.w, at, kind, bs, qp, off);
    }
    if !pic.separate_plane {
        let at = super::ChromaEdge {
            mx,
            my,
            edge,
            vertical,
            transform8: cur.transform8,
            format: planes.format,
        };
        super::filter_chroma(pic, (&mut *planes.cb, &mut *planes.cr), at, bs, off);
    }
}

/// Clause 8.7.2.1: the strength of each quarter of an edge of `cur`.
/// `across` is the macroblock on the far side of edge 0.
fn strengths(cur: &Mb, across: Option<&Mb>, edge: usize, vertical: bool) -> [u8; 4] {
    let other = across.unwrap_or(cur);
    if edge != 0 {
        if cur.intra {
            return [3; 4];
        }
    } else if cur.intra || other.intra {
        // The strongest filter is for a vertical macroblock edge, or a
        // horizontal one between two frame macroblocks.
        let frames = !cur.field_mb && !other.field_mb;
        return if vertical || frames { [4; 4] } else { [3; 4] };
    }
    // A field macroblock under a frame pair: the two sides do not share
    // lines, so their motion is not compared.
    let mixed = other.field_mb != cur.field_mb;
    let q_mask = super::coded_mask(cur);
    let p_mask = super::coded_mask(other);
    let before = if edge == 0 { 3 } else { edge - 1 };
    let mut out = [0u8; 4];
    for (line, strength) in out.iter_mut().enumerate() {
        let (q, p) = if vertical {
            ((edge, line), (before, line))
        } else {
            ((line, edge), (line, before))
        };
        *strength = if (q_mask >> (q.1 * 4 + q.0) | p_mask >> (p.1 * 4 + p.0)) & 1 == 1 {
            2
        } else if mixed {
            1
        } else {
            motion_between(other, p, cur, q)
        };
    }
    out
}

/// Whether the blocks `p` and `q` predict differently enough to filter the
/// edge between them. Both macroblocks are of one kind.
fn motion_between(p_mb: &Mb, p: (usize, usize), q_mb: &Mb, q: (usize, usize)) -> u8 {
    // The threshold is four quarter samples of the frame. A field
    // macroblock's vertical vector counts field lines, each two of those.
    let tall = if q_mb.field_mb { 2 } else { 4 };
    let apart = |a: [i16; 2], b: [i16; 2]| {
        (i32::from(a[0]) - i32::from(b[0])).abs() >= 4
            || (i32::from(a[1]) - i32::from(b[1])).abs() >= tall
    };
    super::motion_strength_by(preds(p_mb, p), preds(q_mb, q), apart)
}

/// What the block (`bx`, `by`) predicts from, on each list: the reference
/// and the vector. A field macroblock's reference is one field of a frame.
/// The two sides of an edge between field macroblocks are fields of one
/// parity, so the low bit of the index tells the fields apart.
fn preds(mb: &Mb, (bx, by): (usize, usize)) -> [(i8, [i16; 2]); 2] {
    let index = by * 4 + bx;
    let part = by / 2 * 2 + bx / 2;
    let picture = |slot: i8, reference: i8| {
        if slot < 0 {
            -1
        } else if mb.field_mb {
            slot.saturating_mul(2).saturating_add(reference & 1)
        } else {
            slot.saturating_mul(2)
        }
    };
    [
        (picture(mb.slot0[part], mb.ref0[part]), mb.mv0[index]),
        (picture(mb.slot1[part], mb.ref1[part]), mb.mv1[index]),
    ]
}

/// One plane's share of a macroblock: where its first sample is, how many
/// lines it has there, and how its samples are filtered.
#[derive(Clone, Copy)]
struct Share {
    x: usize,
    lines: usize,
    stride: usize,
    /// The chroma filter, which changes one sample each side.
    chroma: bool,
}

/// The planes to filter for a macroblock at column `mx`: luma, then Cb and
/// Cr unless they are separate pictures.
fn shares<S: Sample>(pic: &Frame<S>, format: u8, mx: usize) -> [Option<Share>; 3] {
    let luma = Share {
        x: mx * 16,
        lines: 16,
        stride: pic.w,
        chroma: false,
    };
    let chroma = match format {
        3 => luma,
        2 => Share {
            x: mx * 8,
            lines: 16,
            stride: pic.w / 2,
            chroma: true,
        },
        _ => Share {
            x: mx * 8,
            lines: 8,
            stride: pic.w / 2,
            chroma: true,
        },
    };
    let chroma = (!pic.separate_plane).then_some(chroma);
    [Some(luma), chroma, chroma]
}

/// The average quantiser across an edge between `cur` and `other`, for plane
/// 0 (luma), 1 (Cb) or 2 (Cr).
fn qp_between(cur: &Mb, other: &Mb, plane: usize) -> i32 {
    let of = |mb: &Mb| {
        i32::from(match plane {
            0 => mb.qp,
            1 => mb.qp_c,
            _ => mb.qp_cr,
        })
    };
    (of(cur) + of(other) + 1) >> 1
}

/// The left edge of a macroblock whose pair is not the kind of the pair to
/// its left (clause 8.7.1). Each line of the macroblock continues into the
/// macroblock that holds that line of the picture, so the edge has two
/// neighbours and eight strengths, one for each two luma lines.
///
/// A field macroblock's upper eight lines border the top frame macroblock
/// and the rest the bottom one. A frame macroblock's lines alternate
/// between the top field macroblock and the bottom one. Motion is not
/// compared across such an edge: its strength is at least 1.
fn mixed_left<S: Sample>(pic: &Frame<S>, planes: &mut Planes<'_, S>, (mx, my): (usize, usize)) {
    let mb_w = pic.mb_w;
    let pair = my & !1;
    let (Some(cur), Some(left)) = (pic.mb.get(my * mb_w + mx), mx.checked_sub(1)) else {
        return;
    };
    let (Some(upper), Some(lower)) = (
        pic.mb.get(pair * mb_w + left),
        pic.mb.get((pair + 1) * mb_w + left),
    ) else {
        return;
    };
    let field = cur.field_mb;
    let sides = [upper, lower];
    let q_mask = super::coded_mask(cur);
    let p_masks = [super::coded_mask(upper), super::coded_mask(lower)];
    let mut bs = [0u8; 8];
    for (i, strength) in bs.iter_mut().enumerate() {
        // The neighbour beside these two lines, and its row of blocks there.
        let (side, row) = if field {
            (i / 4, i % 4)
        } else {
            (i % 2, my % 2 * 2 + i / 4)
        };
        *strength = if cur.intra || sides[side].intra {
            4
        } else if (q_mask >> (i / 2 * 4) | p_masks[side] >> (row * 4 + 3)) & 1 == 1 {
            2
        } else {
            1
        };
    }
    // The strengths along each neighbour, top to bottom.
    let along = if field {
        [[bs[0], bs[1], bs[2], bs[3]], [bs[4], bs[5], bs[6], bs[7]]]
    } else {
        [[bs[0], bs[2], bs[4], bs[6]], [bs[1], bs[3], bs[5], bs[7]]]
    };
    let off = (i32::from(cur.off_a), i32::from(cur.off_b));
    for (plane, share) in shares(pic, planes.format, mx).into_iter().enumerate() {
        let Some(share) = share else {
            continue;
        };
        let samples = match plane {
            0 => &mut *planes.luma,
            1 => &mut *planes.cb,
            _ => &mut *planes.cr,
        };
        let span = super::chroma_span(my, share.lines, field);
        for (side, other) in sides.into_iter().enumerate() {
            // Half of a field macroblock's lines, or every other line of a
            // frame macroblock's: each time lines two picture rows apart.
            let y = if field {
                span.at(share.lines / 2 * side)
            } else {
                span.y + side
            };
            let kind = (true, share.chroma, share.lines / 8, 2);
            let qp = qp_between(cur, other, plane);
            super::filter_groups(
                samples,
                share.stride,
                (share.x, y),
                kind,
                along[side],
                qp,
                off,
            );
        }
    }
}

/// The upper edge of a top frame macroblock under a field pair (clause
/// 8.7.1). It is filtered twice, once along each field: the macroblock's
/// even lines with the top field macroblock above and its odd lines with the
/// bottom one, each three lines deep in steps of two. Motion is not
/// compared, and an intra side gives strength 3.
fn under_fields<S: Sample>(pic: &Frame<S>, planes: &mut Planes<'_, S>, (mx, my): (usize, usize)) {
    let mb_w = pic.mb_w;
    let (Some(cur), Some(above)) = (pic.mb.get(my * mb_w + mx), my.checked_sub(2)) else {
        return;
    };
    let q_mask = super::coded_mask(cur);
    let off = (i32::from(cur.off_a), i32::from(cur.off_b));
    for parity in 0..2 {
        let Some(over) = pic.mb.get((above + parity) * mb_w + mx) else {
            continue;
        };
        let p_mask = super::coded_mask(over);
        let mut bs = [3u8; 4];
        if !cur.intra && !over.intra {
            for (column, strength) in bs.iter_mut().enumerate() {
                let coded = (q_mask >> column | p_mask >> (12 + column)) & 1 == 1;
                *strength = if coded { 2 } else { 1 };
            }
        }
        for (plane, share) in shares(pic, planes.format, mx).into_iter().enumerate() {
            let Some(share) = share else {
                continue;
            };
            let samples = match plane {
                0 => &mut *planes.luma,
                1 => &mut *planes.cb,
                _ => &mut *planes.cr,
            };
            let y = my * share.lines + parity;
            // A quarter of the edge is four luma samples, two of 4:2:0 or
            // 4:2:2 chroma.
            let quarter = if share.chroma { 2 } else { 4 };
            let kind = (false, share.chroma, quarter, 2);
            let qp = qp_between(cur, over, plane);
            super::filter_groups(samples, share.stride, (share.x, y), kind, bs, qp, off);
        }
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

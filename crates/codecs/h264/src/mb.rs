//! Macroblock layer, clauses 7.3.5 and 8. One slice group is raster order.
//! Several groups walk to the next macroblock of the same group.

use cutlass_decode::CodecError;

use crate::cabac::{self, Cabac};
use crate::cavlc;
use crate::recon::{self, Sample};
use crate::set::{Sps, ZIGZAG4, ZIGZAG8};

use super::{Entropy, Frame, Header, Mb, Mod, Slice, blk_x, blk_y, put_block};

#[derive(Clone, Copy)]
struct Place {
    mx: usize,
    my: usize,
}

#[derive(Clone, Copy)]
struct Cell {
    bx: usize,
    by: usize,
}

struct LumaPaint<'a> {
    i16: bool,
    mode16: u8,
    chroma_mode: u8,
    dc: [i32; 16],
    ac: &'a [[i32; 16]; 16],
    block8: &'a [[i32; 64]; 4],
}

struct ChromaPaint<'a> {
    cb_dc: &'a [i32],
    cr_dc: &'a [i32],
    cb_ac: &'a [[i32; 16]],
    cr_ac: &'a [[i32; 16]],
}

struct ChromaResidual {
    cb_dc: [i32; 8],
    cr_dc: [i32; 8],
    cb_ac: [[i32; 16]; 8],
    cr_ac: [[i32; 16]; 8],
}

/// A list slot the initialisation process left as "no reference picture".
const NO_PICTURE: usize = usize::MAX;

/// SliceQPY after `mb_qp_delta`. The result stays in `-QpBdOffsetY..=51`.
fn next_qp<S: Sample>(qp: i32, delta: i32) -> i32 {
    let span = 52 + S::QP_BD;
    (qp + delta + span + S::QP_BD).rem_euclid(span) - S::QP_BD
}

fn chroma_deblock<S: Sample>(luma_qp: i32, offset: i32) -> i8 {
    recon::chroma_qps(luma_qp, offset, S::QP_BD).1 as i8
}

pub(crate) fn reference_lists<S: Sample>(
    dpb: &[Frame<S>],
    header: &Header,
    poc: i32,
    max_fn: u32,
    size: (usize, usize),
) -> Result<(Vec<usize>, Vec<usize>), CodecError> {
    let view = header.view_id;
    let (width, height) = size;
    let short = |frame: &&Frame<S>| {
        frame.referenced
            && !frame.long_term
            && !frame.nonexisting
            && frame.view_id == view
            && frame.w == width
            && frame.h == height
    };
    let mut shorts: Vec<usize> = dpb
        .iter()
        .enumerate()
        .filter(|(_, frame)| short(frame))
        .map(|(index, _)| index)
        .collect();
    let mut longs: Vec<usize> = dpb
        .iter()
        .enumerate()
        .filter(|(_, frame)| {
            frame.referenced
                && frame.long_term
                && !frame.nonexisting
                && frame.view_id == view
                && frame.w == width
                && frame.h == height
        })
        .map(|(index, _)| index)
        .collect();
    longs.sort_by_key(|&index| dpb[index].long_idx);
    if super::intra_slice(header.kind) {
        return Ok((Vec::new(), Vec::new()));
    }
    if super::syntax_kind(header.kind) == 0 {
        shorts.sort_by_key(|&index| {
            std::cmp::Reverse(pic_num(&dpb[index], header.frame_num, max_fn))
        });
        let mut list = shorts;
        list.extend(longs);
        append_interview(&mut list, dpb, poc, &header.interview_l0, size);
        let list = reorder(
            list,
            header.num_ref_l0,
            dpb,
            &header.ref_mod_l0,
            header.frame_num,
            max_fn,
        )?;
        return Ok((list, Vec::new()));
    }
    let mut past: Vec<usize> = shorts
        .iter()
        .copied()
        .filter(|&index| dpb[index].poc < poc)
        .collect();
    let mut future: Vec<usize> = shorts
        .iter()
        .copied()
        .filter(|&index| dpb[index].poc > poc)
        .collect();
    past.sort_by_key(|&index| std::cmp::Reverse(dpb[index].poc));
    future.sort_by_key(|&index| dpb[index].poc);
    let mut list0 = past.clone();
    list0.extend(future.iter().copied());
    list0.extend(longs.iter().copied());
    let mut list1 = future;
    list1.extend(past);
    list1.extend(longs);
    // Clause 8.2.4.2.3 switches the first two list 1 entries before the lists
    // are cut to the active lengths.
    if list0 == list1 && list1.len() > 1 {
        list1.swap(0, 1);
    }
    append_interview(&mut list0, dpb, poc, &header.interview_l0, size);
    append_interview(&mut list1, dpb, poc, &header.interview_l1, size);
    let list0 = reorder(
        list0,
        header.num_ref_l0,
        dpb,
        &header.ref_mod_l0,
        header.frame_num,
        max_fn,
    )?;
    let list1 = reorder(
        list1,
        header.num_ref_l1,
        dpb,
        &header.ref_mod_l1,
        header.frame_num,
        max_fn,
    )?;
    Ok((list0, list1))
}

fn append_interview<S: Sample>(
    list: &mut Vec<usize>,
    dpb: &[Frame<S>],
    poc: i32,
    views: &[u16],
    size: (usize, usize),
) {
    let (width, height) = size;
    for view in views {
        let Some(index) = dpb.iter().position(|frame| {
            frame.referenced
                && !frame.nonexisting
                && frame.inter_view
                && frame.view_id == *view
                && frame.poc == poc
                && frame.w == width
                && frame.h == height
        }) else {
            continue;
        };
        if !list.contains(&index) {
            list.push(index);
        }
    }
}

fn pic_num<S: Sample>(frame: &Frame<S>, current: u32, max_fn: u32) -> i32 {
    let mut number = i32::try_from(frame.frame_num).unwrap_or(0);
    if frame.frame_num > current {
        number -= i32::try_from(max_fn).unwrap_or(0);
    }
    number
}

/// Clause 8.2.4.2 pads a short initial list, then 8.2.4.3 may repeat a picture
/// so the active length is filled. A slot left as "no reference picture" stays
/// [`NO_PICTURE`] and is rejected when a macroblock uses it.
fn reorder<S: Sample>(
    mut list: Vec<usize>,
    active: usize,
    dpb: &[Frame<S>],
    mods: &[Mod],
    current: u32,
    max_fn: u32,
) -> Result<Vec<usize>, CodecError> {
    if active > 32 {
        return Err(CodecError::Limit { what: "ref idx" });
    }
    if list.len() > active {
        list.truncate(active);
    }
    list.resize(active, NO_PICTURE);
    if mods.is_empty() {
        return Ok(list);
    }
    let max_pic = i32::try_from(max_fn).map_err(|_| CodecError::Malformed {
        at_bit: 0,
        what: "ref list modification",
    })?;
    let curr = i32::try_from(current).map_err(|_| CodecError::Malformed {
        at_bit: 0,
        what: "ref list modification",
    })?;
    list.push(NO_PICTURE);
    let mut pred = curr;
    let mut ref_idx = 0usize;
    for modification in mods {
        if ref_idx >= active {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "ref list modification",
            });
        }
        match modification.idc {
            0 | 1 => {
                let pic_num = short_target(&mut pred, modification, curr, max_pic)?;
                let Some(picture) = short_index(dpb, pic_num, current, max_fn) else {
                    return Err(CodecError::Malformed {
                        at_bit: 0,
                        what: "ref list modification",
                    });
                };
                place_short(
                    &mut list,
                    ref_idx,
                    active,
                    picture,
                    pic_num,
                    dpb,
                    ShortNums {
                        current,
                        max_fn,
                        max_pic,
                    },
                );
                ref_idx += 1;
            }
            2 => {
                let Some(picture) = long_index(dpb, modification.arg) else {
                    return Err(CodecError::Malformed {
                        at_bit: 0,
                        what: "ref list modification",
                    });
                };
                place_long(&mut list, ref_idx, active, picture, modification.arg, dpb);
                ref_idx += 1;
            }
            _ => {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "ref list modification",
                });
            }
        }
    }
    list.truncate(active);
    Ok(list)
}

fn short_target(
    pred: &mut i32,
    modification: &Mod,
    curr: i32,
    max_pic: i32,
) -> Result<i32, CodecError> {
    let Some(sum) = modification.arg.checked_add(1) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "ref list modification",
        });
    };
    let Ok(abs) = i32::try_from(sum) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "ref list modification",
        });
    };
    if abs <= 0 || abs > max_pic || max_pic <= 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "ref list modification",
        });
    }
    let no_wrap = if modification.idc == 0 {
        if *pred < abs {
            *pred - abs + max_pic
        } else {
            *pred - abs
        }
    } else if *pred > max_pic - abs {
        *pred + abs - max_pic
    } else {
        *pred + abs
    };
    *pred = no_wrap;
    Ok(if no_wrap > curr {
        no_wrap - max_pic
    } else {
        no_wrap
    })
}

struct ShortNums {
    current: u32,
    max_fn: u32,
    max_pic: i32,
}

fn place_short<S: Sample>(
    list: &mut [usize],
    ref_idx: usize,
    active: usize,
    picture: usize,
    pic_num: i32,
    dpb: &[Frame<S>],
    nums: ShortNums,
) {
    for c in (ref_idx + 1..=active).rev() {
        list[c] = list[c - 1];
    }
    list[ref_idx] = picture;
    let mut nidx = ref_idx + 1;
    for c in (ref_idx + 1)..=active {
        if pic_num_f(dpb, list[c], &nums) != pic_num {
            list[nidx] = list[c];
            nidx += 1;
        }
    }
}

fn place_long<S: Sample>(
    list: &mut [usize],
    ref_idx: usize,
    active: usize,
    picture: usize,
    long_num: u32,
    dpb: &[Frame<S>],
) {
    for c in (ref_idx + 1..=active).rev() {
        list[c] = list[c - 1];
    }
    list[ref_idx] = picture;
    let mut nidx = ref_idx + 1;
    for c in (ref_idx + 1)..=active {
        if !same_long(dpb, list[c], long_num) {
            list[nidx] = list[c];
            nidx += 1;
        }
    }
}

fn short_index<S: Sample>(
    dpb: &[Frame<S>],
    number: i32,
    current: u32,
    max_fn: u32,
) -> Option<usize> {
    dpb.iter().position(|frame| {
        frame.referenced
            && !frame.long_term
            && !frame.nonexisting
            && pic_num(frame, current, max_fn) == number
    })
}

fn long_index<S: Sample>(dpb: &[Frame<S>], long_num: u32) -> Option<usize> {
    let Ok(long_num) = i32::try_from(long_num) else {
        return None;
    };
    dpb.iter().position(|frame| {
        frame.referenced && frame.long_term && !frame.nonexisting && frame.long_idx == long_num
    })
}

fn pic_num_f<S: Sample>(dpb: &[Frame<S>], slot: usize, nums: &ShortNums) -> i32 {
    let Some(frame) = dpb.get(slot) else {
        return nums.max_pic;
    };
    if !frame.referenced || frame.long_term {
        return nums.max_pic;
    }
    pic_num(frame, nums.current, nums.max_fn)
}

fn same_long<S: Sample>(dpb: &[Frame<S>], slot: usize, long_num: u32) -> bool {
    let Ok(long_num) = i32::try_from(long_num) else {
        return false;
    };
    dpb.get(slot).is_some_and(|frame| {
        frame.referenced && frame.long_term && !frame.nonexisting && frame.long_idx == long_num
    })
}

/// For a picture decoded on its own thread: when decoding has just passed
/// the end of a macroblock row, deblocks the rows that are now safe and
/// publishes the samples that makes final. A row is deblocked once the row
/// below it is decoded, since that row's intra prediction reads this one
/// unfiltered.
fn rows_decoded<S: Sample>(pic: &mut Frame<S>, addr: usize) {
    let Some(writer) = pic.writer else {
        return;
    };
    if pic.mb_w == 0 || !addr.is_multiple_of(pic.mb_w) {
        return;
    }
    let ready = (addr / pic.mb_w).saturating_sub(1);
    if ready <= writer.deblocked {
        return;
    }
    deblock_rows(pic, writer.deblocked..ready);
    let published = match pic.pending.clone() {
        Some(pending) => pending.publish(pic, writer.published, ready, false),
        None => writer.published,
    };
    pic.writer = Some(super::Writer {
        deblocked: ready,
        published,
    });
}

/// Deblocks and publishes the rows [`rows_decoded`] left, once every
/// macroblock of the picture is decoded.
pub(crate) fn finish_rows<S: Sample>(pic: &mut Frame<S>) {
    let Some(writer) = pic.writer else {
        return;
    };
    let rows = pic.h / 16;
    deblock_rows(pic, writer.deblocked..rows);
    let published = match pic.pending.clone() {
        Some(pending) => pending.publish(pic, writer.published, rows, true),
        None => writer.published,
    };
    pic.writer = Some(super::Writer {
        deblocked: rows,
        published,
    });
}

pub(crate) fn decode_mbs<S: Sample>(
    pic: &mut Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    slice_id: u16,
    next_mb: &mut usize,
    mut entropy: Entropy<'_, '_>,
    layer: Option<&Frame<S>>,
) -> Result<(), CodecError> {
    let total = pic.mb_w.saturating_mul(pic.h / 16);
    let mbaff = slice.sps.mbaff && !pic.field;
    let next = slice.group_next.as_slice();
    if !next.is_empty() && next.len() != total {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "slice group map",
        });
    }
    let mut addr = if next.is_empty() {
        *next_mb
    } else {
        slice.origin
    };
    let mut decoded = *next_mb;
    let mut qp = slice.qp;
    let mut rle = Rle {
        run: 0,
        kind: false,
    };
    while addr < total {
        if !super::intra_slice(slice.kind) {
            match &mut entropy {
                Entropy::Cavlc(bits) => {
                    let run = bits.ue()? as usize;
                    let vs_skip = if slice.vsp && run > 0 && vsp_ref_exist(pic, dpb, slice) {
                        bits.bit()?
                    } else {
                        false
                    };
                    for left in 0..run {
                        if addr >= total {
                            return Err(bits.malformed("skip run"));
                        }
                        fresh(pic, addr, mbaff, !next.is_empty())?;
                        let flag = skip_field_flag(pic, bits, slice_id, addr, mbaff, run - left)?;
                        if vs_skip {
                            if slice.kind > 1 {
                                return Err(CodecError::Unsupported { what: "3d" });
                            }
                            vsp_skip_mb(pic, dpb, slice, slice_id, addr, qp, flag)?;
                        } else {
                            skip_mb(pic, dpb, slice, slice_id, addr, qp, flag)?;
                        }
                        addr = step(next, addr, &mut decoded, total)?;
                        rows_decoded(pic, addr);
                    }
                    if addr >= total || (run > 0 && !bits.more_rbsp()) {
                        break;
                    }
                }
                Entropy::Cabac(cab) => {
                    let skipped = rle_skip(pic, cab, slice, slice_id, addr, mbaff, &mut rle)?;
                    if skipped {
                        fresh(pic, addr, mbaff, !next.is_empty())?;
                        let flag = cabac_skip_field(pic, cab, slice_id, addr, mbaff)?;
                        skip_mb(pic, dpb, slice, slice_id, addr, qp, flag)?;
                        let end = if !mbaff || pair_bottom(pic, addr, true) {
                            cab.terminate()?
                        } else {
                            false
                        };
                        addr = step(next, addr, &mut decoded, total)?;
                        rows_decoded(pic, addr);
                        if end {
                            break;
                        }
                        continue;
                    }
                }
            }
        }
        if addr >= total {
            break;
        }
        fresh(pic, addr, mbaff, !next.is_empty())?;
        let base_mode = if slice.adaptive_base {
            match &mut entropy {
                Entropy::Cavlc(bits) => bits.bit()?,
                Entropy::Cabac(_) => {
                    return Err(CodecError::Unsupported {
                        what: "inter-layer prediction",
                    });
                }
            }
        } else {
            slice.base_mode
        };
        qp = if base_mode {
            let Some(layer) = layer else {
                return Err(CodecError::Unsupported {
                    what: "inter-layer prediction",
                });
            };
            let residual = coded_residual_flag(slice, &mut entropy)?;
            let qp = inferred_base(pic, dpb, layer, slice, &mut entropy, addr, (slice_id, qp))?;
            if residual {
                if slice.tcoeff {
                    return Err(CodecError::Unsupported {
                        what: "inter-layer prediction",
                    });
                }
                add_layer_residual(pic, dpb, layer, slice, addr)?;
            }
            qp
        } else {
            let (qp, residual) =
                one_mb(pic, dpb, slice, slice_id, addr, (qp, layer), &mut entropy)?;
            if residual {
                let Some(layer) = layer else {
                    return Err(CodecError::Unsupported {
                        what: "inter-layer prediction",
                    });
                };
                if slice.tcoeff {
                    add_tcoeff_residual(pic, layer, slice, addr)?;
                } else {
                    add_layer_residual(pic, dpb, layer, slice, addr)?;
                }
            }
            qp
        };
        let bottom = !mbaff || pair_bottom(pic, addr, mbaff);
        let end = match &mut entropy {
            Entropy::Cavlc(bits) => bottom && !bits.more_rbsp(),
            Entropy::Cabac(cab) => {
                if bottom {
                    cab.terminate()?
                } else {
                    false
                }
            }
        };
        addr = step(next, addr, &mut decoded, total)?;
        rows_decoded(pic, addr);
        if end {
            break;
        }
    }
    if let Entropy::Cavlc(bits) = &entropy {
        bits.close()?;
    }
    *next_mb = decoded;
    Ok(())
}

fn fresh<S: Sample>(
    pic: &Frame<S>,
    addr: usize,
    mbaff: bool,
    grouped: bool,
) -> Result<(), CodecError> {
    if !grouped {
        return Ok(());
    }
    let (mx, my) = locate(pic, addr, mbaff);
    let slot = my * pic.mb_w + mx;
    if pic.mb.get(slot).is_some_and(|mb| mb.slice != u16::MAX) {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "first macroblock",
        });
    }
    Ok(())
}

fn step(
    next: &[usize],
    addr: usize,
    decoded: &mut usize,
    total: usize,
) -> Result<usize, CodecError> {
    *decoded = decoded.saturating_add(1);
    if *decoded > total {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "slice group map",
        });
    }
    Ok(if next.is_empty() {
        addr.saturating_add(1)
    } else {
        next.get(addr).copied().unwrap_or(total)
    })
}

fn pair_bottom<S: Sample>(pic: &Frame<S>, addr: usize, mbaff: bool) -> bool {
    locate(pic, addr, mbaff).1 % 2 == 1
}

/// `RunLength` and `mb_skip_run_type` for one slice. Both start clear.
struct Rle {
    run: u32,
    kind: bool,
}

/// J.7.3.4.1. A B slice with `enable_rle_skip_flag` reads `mb_skip_run_type`
/// when both neighbours are skipped or unavailable and the run has ended.
/// Type 1 infers the next skips. Type 0 still reads each `mb_skip_flag`.
fn rle_skip<S: Sample>(
    pic: &Frame<S>,
    cab: &mut Cabac<'_>,
    slice: &Slice<'_>,
    slice_id: u16,
    addr: usize,
    mbaff: bool,
    rle: &mut Rle,
) -> Result<bool, CodecError> {
    let ctx = rle_context(pic, slice, slice_id, addr, mbaff);
    if ctx && rle.run == 0 {
        rle.kind = cab.decision(1040)? == 1;
        rle.run = 16;
    } else if !ctx {
        rle.run = 0;
    }
    if ctx && rle.kind {
        rle.run = rle.run.saturating_sub(1);
        return Ok(true);
    }
    let skipped = cabac_skip(
        pic,
        cab,
        super::syntax_kind(slice.kind),
        slice_id,
        addr,
        mbaff,
    )?;
    if ctx && !skipped {
        rle.run = 0;
    }
    Ok(skipped)
}

fn rle_context<S: Sample>(
    pic: &Frame<S>,
    slice: &Slice<'_>,
    slice_id: u16,
    addr: usize,
    mbaff: bool,
) -> bool {
    if !slice.sps.rle || slice.kind != 1 {
        return false;
    }
    let (mx, my) = locate(pic, addr, mbaff);
    let left = neighbor(pic, mx as i32 - 1, my as i32, slice_id);
    let above = neighbor(pic, mx as i32, above_my(pic, mx, my), slice_id);
    left.is_none_or(|mb| mb.skip) && above.is_none_or(|mb| mb.skip)
}

fn cabac_skip<S: Sample>(
    pic: &Frame<S>,
    cab: &mut Cabac<'_>,
    kind: u8,
    slice_id: u16,
    addr: usize,
    mbaff: bool,
) -> Result<bool, CodecError> {
    let (mx, my) = locate(pic, addr, mbaff);
    let left = neighbor(pic, mx as i32 - 1, my as i32, slice_id);
    let above = neighbor(pic, mx as i32, above_my(pic, mx, my), slice_id);
    let mut ctx = 0;
    if left.is_some_and(|mb| !mb.skip) {
        ctx += 1;
    }
    if above.is_some_and(|mb| !mb.skip) {
        ctx += 1;
    }
    if kind == 1 {
        ctx += 13;
    }
    Ok(cab.decision(11 + ctx)? == 1)
}

fn skip_mb<S: Sample>(
    pic: &mut Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    slice_id: u16,
    addr: usize,
    qp: i32,
    field_mb: bool,
) -> Result<(), CodecError> {
    let mbaff = slice.sps.mbaff && !pic.field;
    let (mx, my) = locate(pic, addr, mbaff);
    let mut mb = Mb::new(slice_id);
    mb.field_mb = field_mb;
    mb.skip = true;
    mb.qp = qp as i8;
    mb.qp_c = chroma_deblock::<S>(qp, slice.pps.chroma_qp_offset);
    mb.qp_cr = chroma_deblock::<S>(qp, slice.pps.second_chroma_qp_offset);
    mb.off_a = slice.off_a as i8;
    mb.off_b = slice.off_b as i8;
    mb.deblock = slice.deblock;
    if super::syntax_kind(slice.kind) == 0 {
        if slice.vsp {
            return Err(CodecError::Unsupported { what: "3d" });
        }
        let (mv, reference) = if slice.dmvp {
            dmvp_skip_motion(pic, dpb, slice, mx, my)
                .unwrap_or_else(|| (p_skip_mv(pic, mx, my, slice_id), 0))
        } else {
            (p_skip_mv(pic, mx, my, slice_id), 0)
        };
        for part in 0..4 {
            mb.ref0[part] = reference;
        }
        for block in 0..16 {
            mb.mv0[block] = mv;
        }
        predict_inter(pic, dpb, slice, mx, my, &mb)?;
        if slice.kind == 3 {
            let luma = [[0i32; 16]; 16];
            let chroma = ChromaResidual {
                cb_dc: [0; 8],
                cr_dc: [0; 8],
                cb_ac: [[0; 16]; 8],
                cr_ac: [[0; 16]; 8],
            };
            write_sp(
                pic,
                slice,
                Place { mx, my },
                &mb,
                SpSamples {
                    luma: &luma,
                    chroma: &chroma,
                    intra: false,
                    qp,
                },
            )?;
        }
    } else {
        if slice.vsp {
            return Err(CodecError::Unsupported { what: "3d" });
        }
        direct_16(pic, dpb, slice, mx, my, &mut mb)?;
        predict_inter(pic, dpb, slice, mx, my, &mb)?;
    }
    bind_refs(&mut mb, slice);
    let slot = my * pic.mb_w + mx;
    pic.mb[slot] = mb;
    Ok(())
}

/// J.7.3.4.1. `mb_skip_type_flag` is present when a skip run has an interview reference.
fn vsp_ref_exist<S: Sample>(pic: &Frame<S>, dpb: &[Frame<S>], slice: &Slice<'_>) -> bool {
    let listed = |list: &[usize]| {
        list.iter().any(|&slot| {
            slot != NO_PICTURE
                && dpb
                    .get(slot)
                    .is_some_and(|frame| frame.view_id != pic.view_id)
        })
    };
    listed(&slice.ref0) || (slice.kind == 1 && listed(&slice.ref1))
}

/// J.8.2.1.3 and J.8.2.1.6. `MbVSSkipFlag` 1 splits the skip into four 8×8
/// disparities. A B slice also fills list 1 when that list names an interview
/// picture. `TextureFirstFlag` 1 uses reference view 0 for both lists.
fn vsp_skip_mb<S: Sample>(
    pic: &mut Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    slice_id: u16,
    addr: usize,
    qp: i32,
    field_mb: bool,
) -> Result<(), CodecError> {
    let mbaff = slice.sps.mbaff && !pic.field;
    let (mx, my) = locate(pic, addr, mbaff);
    let mut mb = Mb::new(slice_id);
    mb.field_mb = field_mb;
    mb.skip = true;
    mb.qp = qp as i8;
    mb.qp_c = chroma_deblock::<S>(qp, slice.pps.chroma_qp_offset);
    mb.qp_cr = chroma_deblock::<S>(qp, slice.pps.second_chroma_qp_offset);
    mb.off_a = slice.off_a as i8;
    mb.off_b = slice.off_b as i8;
    mb.deblock = slice.deblock;
    let reference = interview_index(pic, dpb, &slice.ref0).ok_or(CodecError::Malformed {
        at_bit: 0,
        what: "3d",
    })?;
    let reference1 = (slice.kind == 1)
        .then(|| interview_index(pic, dpb, &slice.ref1))
        .flatten();
    for index in 0..4 {
        let part = Part {
            x: (index & 1) * 8,
            y: (index >> 1) * 8,
            w: 8,
            h: 8,
            l0: true,
            l1: false,
        };
        let motion = synthesis_motion(pic, dpb, slice, (mx * 16, my * 16), &part)?;
        fill_ref(&mut mb.ref0, &mut mb.uses0, &part, reference);
        store_mv(&mut mb, &part, motion, 0);
        if let Some(reference1) = reference1 {
            let part1 = Part {
                l0: false,
                l1: true,
                ..part
            };
            fill_ref(&mut mb.ref1, &mut mb.uses1, &part1, reference1);
            store_mv(&mut mb, &part1, motion, 1);
        }
    }
    predict_inter(pic, dpb, slice, mx, my, &mb)?;
    bind_refs(&mut mb, slice);
    let slot = my * pic.mb_w + mx;
    pic.mb[slot] = mb;
    Ok(())
}

fn interview_index<S: Sample>(pic: &Frame<S>, dpb: &[Frame<S>], list: &[usize]) -> Option<i8> {
    for (index, &slot) in list.iter().enumerate() {
        if slot == NO_PICTURE {
            continue;
        }
        if dpb
            .get(slot)
            .is_some_and(|frame| frame.view_id != pic.view_id)
        {
            return i8::try_from(index).ok();
        }
    }
    None
}

/// J.8.2.1.2. `TextureFirstFlag` is 1, so the disparity starts at zero and the
/// base view's texture at this order count supplies the vector. The sample
/// lands on (7, 7) of this macroblock. A missing or intra base macroblock
/// leaves the ordinary skip.
fn dmvp_skip_motion<S: Sample>(
    pic: &Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    mx: usize,
    my: usize,
) -> Option<([i16; 2], i8)> {
    let base = base_texture(pic, dpb, slice)?;
    let slot = my.saturating_mul(base.mb_w).saturating_add(mx);
    let mb = base.mb.get(slot)?;
    if mb.intra {
        return None;
    }
    let reference = mb.ref0[0];
    if reference < 0 || usize::try_from(reference).ok()? >= slice.ref0.len() {
        return None;
    }
    // Disparity zero lands on sample (7, 7), the 4×4 at block 5.
    Some((mb.mv0[5], reference))
}

fn base_texture<'a, S: Sample>(
    pic: &Frame<S>,
    dpb: &'a [Frame<S>],
    slice: &Slice<'_>,
) -> Option<&'a Frame<S>> {
    let base_id = slice.sps.views.first().copied()?;
    if pic.view_id == base_id {
        return None;
    }
    dpb.iter().find(|frame| {
        frame.referenced
            && frame.inter_view
            && !frame.nonexisting
            && frame.view_id == base_id
            && frame.poc == pic.poc
            && frame.w == pic.w
            && frame.h == pic.h
    })
}

/// The coded reference is another picture of this view, not an interview picture.
fn same_view_reference<S: Sample>(
    pic: &Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    mb: &Mb,
    part: &Part,
) -> bool {
    let Ok(index) = usize::try_from(ref_of(mb, part.x / 4, part.y / 4, 0)) else {
        return false;
    };
    let Some(&slot) = slice.ref0.get(index) else {
        return false;
    };
    dpb.get(slot)
        .is_none_or(|frame| frame.view_id == pic.view_id)
}

/// J.8.2.1.7.2. Unavailable neighbors take the base view's motion when that
/// motion names the same reference index. The coded difference is added later.
fn dmvp_temporal_mvp<S: Sample>(
    pic: &Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    at: Place,
    part: &Part,
    mb: &Mb,
) -> [i16; 2] {
    let Place { mx, my } = at;
    let x = part.x / 4;
    let y = part.y / 4;
    let reference = ref_of(mb, x, y, 0);
    let derived = correspond_mv(pic, dpb, slice, (mx, my), part, reference);
    let mut a = part_motion(pic, mx, my, x as i32 - 1, y as i32, 0, mb);
    let mut b = part_motion(pic, mx, my, x as i32, y as i32 - 1, 0, mb);
    let above_right = part_motion(
        pic,
        mx,
        my,
        x as i32 + (part.w / 4) as i32,
        y as i32 - 1,
        0,
        mb,
    );
    let mut c = if above_right.available {
        above_right
    } else {
        part_motion(pic, mx, my, x as i32 - 1, y as i32 - 1, 0, mb)
    };
    for neighbor in [&mut a, &mut b, &mut c] {
        if !neighbor.available || neighbor.reference != reference {
            neighbor.mv = [0, 0];
            neighbor.reference = -1;
            neighbor.available = false;
        }
    }
    if part.w == 16 && part.h == 8 && y == 0 && b.reference == reference {
        return b.mv;
    }
    if part.w == 16 && part.h == 8 && y == 2 && a.reference == reference {
        return a.mv;
    }
    if part.w == 8 && part.h == 16 && x == 0 && a.reference == reference {
        return a.mv;
    }
    if part.w == 8 && part.h == 16 && x == 2 && c.reference == reference {
        return c.mv;
    }
    for neighbor in [&mut a, &mut b, &mut c] {
        if !neighbor.available || neighbor.reference != reference {
            neighbor.mv = derived;
        }
    }
    [
        median(a.mv[0], b.mv[0], c.mv[0]),
        median(a.mv[1], b.mv[1], c.mv[1]),
    ]
}

fn correspond_mv<S: Sample>(
    pic: &Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    at: (usize, usize),
    part: &Part,
    reference: i8,
) -> [i16; 2] {
    let Some(base) = base_texture(pic, dpb, slice) else {
        return [0, 0];
    };
    let (mx, my) = at;
    let slot = my.saturating_mul(base.mb_w).saturating_add(mx);
    let Some(stored) = base.mb.get(slot) else {
        return [0, 0];
    };
    if stored.intra {
        return [0, 0];
    }
    let (bx, by) = correspond_block(part);
    let Some(mv) = stored.mv0.get(by * 4 + bx) else {
        return [0, 0];
    };
    let Some(stored_ref) = stored.ref0.get((by / 2) * 2 + (bx / 2)) else {
        return [0, 0];
    };
    if *stored_ref == reference {
        *mv
    } else {
        [0, 0]
    }
}

/// Clause J.8.2.1.7.2. The 4×4 index is `4 * mbPartIdx + subMbPartIdx`, then
/// the scan in clause 6.4.3. An 8×4 partition's second block is not its origin.
fn correspond_block(part: &Part) -> (usize, usize) {
    let mb_part = (part.y / 8) * 2 + (part.x / 8);
    let sub = if part.w == 8 && part.h == 4 {
        (part.y % 8) / 4
    } else if part.w == 4 && part.h == 8 {
        (part.x % 8) / 4
    } else if part.w == 4 && part.h == 4 {
        ((part.y % 8) / 4) * 2 + ((part.x % 8) / 4)
    } else {
        0
    };
    let index = 4 * mb_part + sub;
    let x = inverse_raster(index / 4, 8, 8, 16, false) + inverse_raster(index % 4, 4, 4, 8, false);
    let y = inverse_raster(index / 4, 8, 8, 16, true) + inverse_raster(index % 4, 4, 4, 8, true);
    (x / 4, y / 4)
}

fn inverse_raster(index: usize, width: usize, height: usize, span: usize, vertical: bool) -> usize {
    let across = span / width;
    if vertical {
        (index / across) * height
    } else {
        (index % across) * width
    }
}

fn p_skip_mv<S: Sample>(pic: &Frame<S>, mx: usize, my: usize, slice: u16) -> [i16; 2] {
    let left = neighbor(pic, mx as i32 - 1, my as i32, slice);
    let above = neighbor(pic, mx as i32, above_my(pic, mx, my), slice);
    let zero = |mb: &Mb, bx: usize, by: usize| {
        let part = (by / 2) * 2 + (bx / 2);
        mb.ref0[part] == 0 && mb.mv0[by * 4 + bx] == [0, 0]
    };
    if left.is_none_or(|mb| zero(mb, 3, 0)) || above.is_none_or(|mb| zero(mb, 0, 3)) {
        [0, 0]
    } else {
        mvp(
            pic,
            Place { mx, my },
            Cell { bx: 0, by: 0 },
            (16, 16),
            0,
            0,
            &Mb::new(slice),
        )
    }
}

/// Base mode. An intra reference layer macroblock is copied, then a coded
/// residual is added. An inter one inherits list-0 motion and predicts from
/// the decoded picture buffer. Resampling and residual prediction are refused.
/// `residual_prediction_flag` when the slice asks for it on every macroblock.
/// An intra slice does not carry the flag. Otherwise the slice default applies.
fn coded_residual_flag(
    slice: &Slice<'_>,
    entropy: &mut Entropy<'_, '_>,
) -> Result<bool, CodecError> {
    if !slice.adaptive_residual {
        return Ok(slice.residual_pred);
    }
    if super::intra_slice(slice.kind) {
        return Ok(false);
    }
    match entropy {
        Entropy::Cavlc(bits) => bits.bit(),
        Entropy::Cabac(_) => Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        }),
    }
}

fn inferred_base<S: Sample>(
    pic: &mut Frame<S>,
    dpb: &[Frame<S>],
    layer: &Frame<S>,
    slice: &Slice<'_>,
    entropy: &mut Entropy<'_, '_>,
    addr: usize,
    state: (u16, i32),
) -> Result<i32, CodecError> {
    let (slice_id, qp) = state;
    let (mx, my) = locate(pic, addr, false);
    let slot = my.saturating_mul(pic.mb_w).saturating_add(mx);
    let intra = layer.mb.get(slot).is_some_and(|mb| mb.intra);
    if intra {
        intra_base(pic, layer, slice, slice_id, addr, qp, entropy)
    } else {
        motion_base(pic, dpb, layer, slice, entropy, addr, state)
    }
}

/// Intra base mode. The prediction is the co-located reference-layer samples.
/// A coded residual is a 4×4 or an 8×8 added to that prediction.
fn intra_base<S: Sample>(
    pic: &mut Frame<S>,
    layer: &Frame<S>,
    slice: &Slice<'_>,
    slice_id: u16,
    addr: usize,
    qp: i32,
    entropy: &mut Entropy<'_, '_>,
) -> Result<i32, CodecError> {
    if slice.sps.mbaff || pic.field || pic.separate_plane || pic.chroma() != 1 {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let (mx, my) = locate(pic, addr, false);
    let slot = my * pic.mb_w + mx;
    copy_reference(pic, layer, mx, my)?;
    let mut mb = Mb::new(slice_id);
    mb.intra = true;
    mb.off_a = slice.off_a as i8;
    mb.off_b = slice.off_b as i8;
    mb.deblock = slice.deblock;
    pic.mb[slot] = mb.clone();
    let (luma_cbp, chroma_cbp) = read_cbp(pic, entropy, mx, my, true, slice_id, slice.sps.chroma)?;
    if luma_cbp > 0 && slice.pps.transform_8x8 {
        mb.transform8 = transform8_flag(pic, entropy, mx, my, slice_id)?;
    }
    let mut qp = qp;
    if luma_cbp > 0 || chroma_cbp > 0 {
        let delta = qp_delta(entropy, prev_qpd(pic, mx, my, slice_id))?;
        qp = next_qp::<S>(qp, delta);
        mb.qpd = delta as i8;
    }
    mb.cbp = luma_cbp | (chroma_cbp << 4);
    if mb.transform8 {
        for i in 0..4 {
            if luma_cbp & (1 << i) == 0 {
                continue;
            }
            let block = read8(pic, entropy, mx, my, i, &mut mb, 0)?;
            let span = luma_span(my, false);
            super::add_scaled(
                &mut pic.y,
                pic.w,
                (mx * 16 + (i & 1) * 8, span.at((i >> 1) * 8), span.step),
                &block,
                (qp, qp_bypass::<S>(slice, qp)),
                &slice.pps.weight8[0],
                8,
            );
        }
    } else {
        for i in 0..16 {
            let bit = 1 << ((blk_y(i) / 2) * 2 + blk_x(i) / 2);
            if luma_cbp & bit == 0 {
                continue;
            }
            let block = read_block(
                pic,
                entropy,
                (2, 16, false),
                Place { mx, my },
                Cell {
                    bx: blk_x(i),
                    by: blk_y(i),
                },
                &mut mb,
                0,
            )?;
            let span = luma_span(my, false);
            super::add_scaled(
                &mut pic.y,
                pic.w,
                (mx * 16 + blk_x(i) * 4, span.at(blk_y(i) * 4), span.step),
                &block,
                (qp, qp_bypass::<S>(slice, qp)),
                &slice.pps.weight4[0],
                4,
            );
        }
    }
    let chroma = read_chroma(pic, entropy, mx, my, chroma_cbp, &mut mb)?;
    if chroma_cbp != 0 {
        recon_chroma(
            pic,
            slice,
            Place { mx, my },
            0,
            ChromaPaint {
                cb_dc: &chroma.cb_dc,
                cr_dc: &chroma.cr_dc,
                cb_ac: &chroma.cb_ac,
                cr_ac: &chroma.cr_ac,
            },
            (qp, true, slice_id),
            false,
        );
    }
    mb.qp = qp as i8;
    mb.qp_c = chroma_deblock::<S>(qp, slice.pps.chroma_qp_offset);
    mb.qp_cr = chroma_deblock::<S>(qp, slice.pps.second_chroma_qp_offset);
    pic.mb[slot] = mb;
    Ok(qp)
}

/// Inter base mode. Motion is the reference layer's vectors. The samples come
/// from the decoded picture buffer. A coded residual is a 4×4 or an 8×8.
/// A smaller reference layer scales those vectors onto the cropped window.
fn motion_base<S: Sample>(
    pic: &mut Frame<S>,
    dpb: &[Frame<S>],
    layer: &Frame<S>,
    slice: &Slice<'_>,
    entropy: &mut Entropy<'_, '_>,
    addr: usize,
    state: (u16, i32),
) -> Result<i32, CodecError> {
    let (slice_id, qp) = state;
    if slice.sps.mbaff || pic.field || pic.separate_plane || pic.chroma() != 1 {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let (mx, my) = locate(pic, addr, false);
    let slot = my.saturating_mul(pic.mb_w).saturating_add(mx);
    let mut mb = Mb::new(slice_id);
    mb.off_a = slice.off_a as i8;
    mb.off_b = slice.off_b as i8;
    mb.deblock = slice.deblock;
    if layer.w == pic.w && layer.h == pic.h {
        if layer.mb_w != pic.mb_w {
            return Err(CodecError::Unsupported {
                what: "inter-layer prediction",
            });
        }
        let Some(reference) = layer.mb.get(slot) else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "inter-layer prediction",
            });
        };
        if !has_prediction(reference) {
            return Err(CodecError::Unsupported {
                what: "inter-layer prediction",
            });
        }
        mb.mv0 = reference.mv0;
        mb.mv1 = reference.mv1;
        mb.ref0 = reference.ref0;
        mb.ref1 = reference.ref1;
    } else if matches!(slice.sps.svc_spatial, 1 | 2) {
        scaled_layer_motion(&mut mb, layer, dpb, slice, mx, my)?;
    } else {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    bind_refs(&mut mb, slice);
    pic.mb[slot] = mb.clone();
    let (luma_cbp, chroma_cbp) = read_cbp(pic, entropy, mx, my, false, slice_id, slice.sps.chroma)?;
    if luma_cbp > 0 && slice.pps.transform_8x8 {
        mb.transform8 = transform8_flag(pic, entropy, mx, my, slice_id)?;
    }
    let mut qp = qp;
    if luma_cbp > 0 || chroma_cbp > 0 {
        let delta = qp_delta(entropy, prev_qpd(pic, mx, my, slice_id))?;
        qp = next_qp::<S>(qp, delta);
        mb.qpd = delta as i8;
    }
    mb.cbp = luma_cbp | (chroma_cbp << 4);
    predict_inter(pic, dpb, slice, mx, my, &mb)?;
    if mb.transform8 {
        for i in 0..4 {
            if luma_cbp & (1 << i) == 0 {
                continue;
            }
            let block = read8(pic, entropy, mx, my, i, &mut mb, 0)?;
            let span = luma_span(my, false);
            super::add_scaled(
                &mut pic.y,
                pic.w,
                (mx * 16 + (i & 1) * 8, span.at((i >> 1) * 8), span.step),
                &block,
                (qp, qp_bypass::<S>(slice, qp)),
                &slice.pps.weight8[1],
                8,
            );
        }
    } else {
        for i in 0..16 {
            let bit = 1 << ((blk_y(i) / 2) * 2 + blk_x(i) / 2);
            if luma_cbp & bit == 0 {
                continue;
            }
            let block = read_block(
                pic,
                entropy,
                (2, 16, false),
                Place { mx, my },
                Cell {
                    bx: blk_x(i),
                    by: blk_y(i),
                },
                &mut mb,
                0,
            )?;
            let span = luma_span(my, false);
            super::add_scaled(
                &mut pic.y,
                pic.w,
                (mx * 16 + blk_x(i) * 4, span.at(blk_y(i) * 4), span.step),
                &block,
                (qp, qp_bypass::<S>(slice, qp)),
                &slice.pps.weight4[3],
                4,
            );
        }
    }
    let chroma = read_chroma(pic, entropy, mx, my, chroma_cbp, &mut mb)?;
    if chroma_cbp != 0 {
        recon_chroma(
            pic,
            slice,
            Place { mx, my },
            0,
            ChromaPaint {
                cb_dc: &chroma.cb_dc,
                cr_dc: &chroma.cr_dc,
                cb_ac: &chroma.cb_ac,
                cr_ac: &chroma.cr_ac,
            },
            (qp, false, slice_id),
            false,
        );
    }
    mb.qp = qp as i8;
    mb.qp_c = chroma_deblock::<S>(qp, slice.pps.chroma_qp_offset);
    mb.qp_cr = chroma_deblock::<S>(qp, slice.pps.second_chroma_qp_offset);
    pic.mb[slot] = mb;
    Ok(qp)
}

/// A reference-layer vector scaled onto the cropped enhancement window.
fn scaled_layer_motion<S: Sample>(
    mb: &mut Mb,
    layer: &Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    mx: usize,
    my: usize,
) -> Result<(), CodecError> {
    if layer.field || layer.mbaff || layer.w == 0 || layer.h == 0 {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let crop = &slice.scale;
    let left = sample_offset(crop.left)?;
    let top = sample_offset(crop.top)?;
    let right = sample_offset(crop.right)?;
    let bottom = sample_offset(crop.bottom)?;
    let scaled_w =
        i64::from(
            i32::try_from(slice.sps.width).map_err(|_| CodecError::Unsupported {
                what: "inter-layer prediction",
            })?,
        ) - i64::from(left)
            - i64::from(right);
    let scaled_h =
        i64::from(
            i32::try_from(slice.sps.height).map_err(|_| CodecError::Unsupported {
                what: "inter-layer prediction",
            })?,
        ) - i64::from(top)
            - i64::from(bottom);
    let (Ok(scaled_w), Ok(scaled_h)) = (i32::try_from(scaled_w), i32::try_from(scaled_h)) else {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    };
    if scaled_w <= 0 || scaled_h <= 0 || scaled_w % 2 != 0 || scaled_h % 2 != 0 {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    if !crop_covers(mx, my, left, top, scaled_w, scaled_h) {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let (Ok(base_w), Ok(base_h)) = (i32::try_from(layer.w), i32::try_from(layer.h)) else {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    };
    let x_axis = motion_axis(base_w, scaled_w, slice.sps.level_idc)?;
    let y_axis = motion_axis(base_h, scaled_h, slice.sps.level_idc)?;
    mb.mv0 = [[0; 2]; 16];
    mb.mv1 = [[0; 2]; 16];
    mb.ref0 = [-1; 4];
    mb.ref1 = [-1; 4];
    let mut seen = [false; 4];
    for by in 0..4 {
        for bx in 0..4 {
            let curr_x = sample_index(mx, bx)?;
            let curr_y = sample_index(my, by)?;
            let pos_x = base_sample(
                curr_x.checked_sub(left).ok_or(CodecError::Unsupported {
                    what: "inter-layer prediction",
                })?,
                x_axis,
            )?;
            let pos_y = base_sample(
                curr_y.checked_sub(top).ok_or(CodecError::Unsupported {
                    what: "inter-layer prediction",
                })?,
                y_axis,
            )?;
            let mb_x = (pos_x as usize) / 16;
            let mb_y = (pos_y as usize) / 16;
            let slot = mb_y.saturating_mul(layer.mb_w).saturating_add(mb_x);
            let Some(reference) = layer.mb.get(slot) else {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "inter-layer prediction",
                });
            };
            if reference.intra {
                return Err(CodecError::Unsupported {
                    what: "inter-layer prediction",
                });
            }
            let inside_x = (pos_x as usize) & 15;
            let inside_y = (pos_y as usize) & 15;
            let blk = (inside_y / 4) * 4 + (inside_x / 4);
            let part = (inside_y / 8) * 2 + (inside_x / 8);
            let index = by * 4 + bx;
            let source0 = reference.mv0[blk];
            let source1 = reference.mv1[blk];
            let current = (left, top, right, bottom);
            let crop0 = stored_crop(dpb, &slice.ref0, reference.ref0[part], current)?;
            let crop1 = stored_crop(dpb, &slice.ref1, reference.ref1[part], current)?;
            mb.mv0[index] = [
                crop_vector(
                    source0[0],
                    curr_x,
                    (left, right),
                    (crop0.0, crop0.2),
                    base_w,
                    scaled_w,
                )?,
                crop_vector(
                    source0[1],
                    curr_y,
                    (top, bottom),
                    (crop0.1, crop0.3),
                    base_h,
                    scaled_h,
                )?,
            ];
            mb.mv1[index] = [
                crop_vector(
                    source1[0],
                    curr_x,
                    (left, right),
                    (crop1.0, crop1.2),
                    base_w,
                    scaled_w,
                )?,
                crop_vector(
                    source1[1],
                    curr_y,
                    (top, bottom),
                    (crop1.1, crop1.3),
                    base_h,
                    scaled_h,
                )?,
            ];
            let enh = (by / 2) * 2 + (bx / 2);
            if !seen[enh] {
                mb.ref0[enh] = reference.ref0[part];
                mb.ref1[enh] = reference.ref1[part];
                seen[enh] = true;
            } else if mb.ref0[enh] != reference.ref0[part] || mb.ref1[enh] != reference.ref1[part] {
                return Err(CodecError::Unsupported {
                    what: "inter-layer prediction",
                });
            }
        }
    }
    if !has_prediction(mb) {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    Ok(())
}

fn crop_covers(mx: usize, my: usize, left: i32, top: i32, scaled_w: i32, scaled_h: i32) -> bool {
    let (Ok(mx), Ok(my)) = (i32::try_from(mx), i32::try_from(my)) else {
        return false;
    };
    let x0 = (left + 15) / 16;
    let y0 = (top + 15) / 16;
    let x1 = (left + scaled_w) / 16;
    let y1 = (top + scaled_h) / 16;
    mx >= x0 && mx < x1 && my >= y0 && my < y1
}

fn sample_index(mb: usize, block: usize) -> Result<i32, CodecError> {
    i32::try_from(mb * 16 + block * 4 + 1).map_err(|_| CodecError::Unsupported {
        what: "inter-layer prediction",
    })
}

#[derive(Clone, Copy)]
struct MotionAxis {
    scale: i64,
    add: i64,
    shift: i32,
    limit: i32,
}

fn motion_axis(reference: i32, scaled: i32, level: u8) -> Result<MotionAxis, CodecError> {
    if reference <= 0 || scaled <= 0 {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let shift = if level <= 30 {
        16
    } else {
        31 - ceil_log2(reference)
    };
    if !(1..=30).contains(&shift) {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let scale = ((i64::from(reference) << shift) + i64::from(scaled >> 1)) / i64::from(scaled);
    Ok(MotionAxis {
        scale,
        add: 1i64 << (shift - 1),
        shift,
        limit: reference,
    })
}

fn base_sample(curr: i32, axis: MotionAxis) -> Result<i32, CodecError> {
    let pos = i64::from(curr)
        .checked_mul(axis.scale)
        .and_then(|value| value.checked_add(axis.add))
        .ok_or(CodecError::Unsupported {
            what: "inter-layer prediction",
        })?;
    let pos = i32::try_from(pos >> axis.shift).unwrap_or(i32::MAX);
    if axis.limit <= 0 {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    Ok(pos.clamp(0, axis.limit - 1))
}

fn stored_crop<S: Sample>(
    dpb: &[Frame<S>],
    list: &[usize],
    index: i8,
    current: (i32, i32, i32, i32),
) -> Result<(i32, i32, i32, i32), CodecError> {
    if index < 0 {
        return Ok(current);
    }
    let Some(slot) = usize::try_from(index)
        .ok()
        .and_then(|at| list.get(at))
        .copied()
    else {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    };
    let Some(frame) = dpb.get(slot) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "inter-layer prediction",
        });
    };
    Ok((
        frame.svc_left,
        frame.svc_top,
        frame.svc_right,
        frame.svc_bottom,
    ))
}

/// Scale a reference-layer vector, then add the crop difference of its picture.
fn crop_vector(
    mv: i16,
    sample: i32,
    current: (i32, i32),
    reference: (i32, i32),
    base: i32,
    scaled: i32,
) -> Result<i16, CodecError> {
    let id_near = i64::from(current.0) - i64::from(reference.0);
    let id_span = i64::from(current.1) - i64::from(reference.1) + id_near;
    if id_near == 0 && id_span == 0 {
        return scale_vector(mv, base, scaled);
    }
    if base <= 0 || scaled <= 0 {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let span = i64::from(scaled) + id_span;
    if span <= 0 {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let ratio = ((span << 16) + i64::from(base >> 1)) / i64::from(base);
    let mut value = i64::from(mv)
        .checked_mul(ratio)
        .and_then(|product| product.checked_add(32_768))
        .ok_or(CodecError::Unsupported {
            what: "inter-layer prediction",
        })?;
    value >>= 16;
    let corr = ((id_span * 4) << 16) + i64::from(scaled >> 1);
    let corr = corr / i64::from(scaled);
    let at = i64::from(sample) - i64::from(current.0);
    let shift = at
        .checked_mul(corr)
        .and_then(|product| product.checked_add(32_768))
        .ok_or(CodecError::Unsupported {
            what: "inter-layer prediction",
        })?;
    value = value
        .checked_add(shift >> 16)
        .and_then(|sum| sum.checked_sub(4 * id_near))
        .ok_or(CodecError::Unsupported {
            what: "inter-layer prediction",
        })?;
    i16::try_from(value).map_err(|_| CodecError::Unsupported {
        what: "inter-layer prediction",
    })
}

fn scale_vector(mv: i16, reference: i32, scaled: i32) -> Result<i16, CodecError> {
    if reference <= 0 {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let ratio = ((i64::from(scaled) << 16) + i64::from(reference >> 1)) / i64::from(reference);
    let value = i64::from(mv)
        .checked_mul(ratio)
        .and_then(|product| product.checked_add(32_768))
        .ok_or(CodecError::Unsupported {
            what: "inter-layer prediction",
        })?;
    i16::try_from(value >> 16).map_err(|_| CodecError::Unsupported {
        what: "inter-layer prediction",
    })
}

pub(crate) struct ResidualPlanes {
    pub(crate) y: Vec<i32>,
    pub(crate) u: Vec<i32>,
    pub(crate) v: Vec<i32>,
}

/// Upsample an inter-layer residual onto the cropped enhancement window.
pub(crate) fn resample_residual<S: Sample>(
    layer: &Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
) -> Result<ResidualPlanes, CodecError> {
    if layer.field
        || layer.mbaff
        || slice.sps.mbaff
        || !slice.sps.frame_mbs_only
        || slice.sps.chroma != 1
    {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let crop = &slice.scale;
    let left = sample_offset(crop.left)?;
    let top = sample_offset(crop.top)?;
    let right = sample_offset(crop.right)?;
    let bottom = sample_offset(crop.bottom)?;
    let base = base_residual(layer, dpb)?;
    let out_w = i32::try_from(slice.sps.width).map_err(|_| CodecError::Unsupported {
        what: "inter-layer prediction",
    })?;
    let out_h = i32::try_from(slice.sps.height).map_err(|_| CodecError::Unsupported {
        what: "inter-layer prediction",
    })?;
    let scaled_w = out_w
        .checked_sub(left)
        .and_then(|width| width.checked_sub(right))
        .ok_or(CodecError::Unsupported {
            what: "inter-layer prediction",
        })?;
    let scaled_h = out_h
        .checked_sub(top)
        .and_then(|height| height.checked_sub(bottom))
        .ok_or(CodecError::Unsupported {
            what: "inter-layer prediction",
        })?;
    if scaled_w <= 0 || scaled_h <= 0 || scaled_w % 2 != 0 || scaled_h % 2 != 0 {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let base_w = i32::try_from(layer.w).map_err(|_| CodecError::Unsupported {
        what: "inter-layer prediction",
    })?;
    let base_h = i32::try_from(layer.h).map_err(|_| CodecError::Unsupported {
        what: "inter-layer prediction",
    })?;
    let level = slice.sps.level_idc;
    let y = upsample_residual(ResidualUp {
        src: &base.y,
        base_w,
        base_h,
        out_w,
        out_h,
        scaled_w,
        scaled_h,
        left,
        top,
        phase_x: 0,
        phase_y: 0,
        ref_phase_x: 0,
        ref_phase_y: 0,
        level,
        mbs: &layer.mb,
        mb_w: layer.mb_w,
        chroma: false,
    })?;
    let chroma = (
        i32::from(crop.chroma_x),
        i32::from(crop.chroma_y),
        i32::from(crop.ref_chroma_x),
        i32::from(crop.ref_chroma_y),
    );
    let u = upsample_residual(ResidualUp {
        src: &base.u,
        base_w: base_w / 2,
        base_h: base_h / 2,
        out_w: out_w / 2,
        out_h: out_h / 2,
        scaled_w: scaled_w / 2,
        scaled_h: scaled_h / 2,
        left: crop.left,
        top: crop.top,
        phase_x: chroma.0,
        phase_y: chroma.1,
        ref_phase_x: chroma.2,
        ref_phase_y: chroma.3,
        level,
        mbs: &layer.mb,
        mb_w: layer.mb_w,
        chroma: true,
    })?;
    let v = upsample_residual(ResidualUp {
        src: &base.v,
        base_w: base_w / 2,
        base_h: base_h / 2,
        out_w: out_w / 2,
        out_h: out_h / 2,
        scaled_w: scaled_w / 2,
        scaled_h: scaled_h / 2,
        left: crop.left,
        top: crop.top,
        phase_x: chroma.0,
        phase_y: chroma.1,
        ref_phase_x: chroma.2,
        ref_phase_y: chroma.3,
        level,
        mbs: &layer.mb,
        mb_w: layer.mb_w,
        chroma: true,
    })?;
    Ok(ResidualPlanes { y, u, v })
}

fn base_residual<S: Sample>(
    layer: &Frame<S>,
    dpb: &[Frame<S>],
) -> Result<ResidualPlanes, CodecError> {
    if layer.w == 0 || layer.h == 0 || layer.mb_w == 0 || layer.chroma() != 1 {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let mut y = vec![0i32; layer.w.saturating_mul(layer.h)];
    let cw = layer.w / 2;
    let ch = layer.h / 2;
    let mut u = vec![0i32; cw.saturating_mul(ch)];
    let mut v = vec![0i32; cw.saturating_mul(ch)];
    let mb_h = layer.h / 16;
    for my in 0..mb_h {
        for mx in 0..layer.mb_w {
            let slot = my.saturating_mul(layer.mb_w).saturating_add(mx);
            let Some(mb) = layer.mb.get(slot) else {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "inter-layer prediction",
                });
            };
            if mb.intra || !has_prediction(mb) {
                return Err(CodecError::Unsupported {
                    what: "inter-layer prediction",
                });
            }
            for by in 0..4 {
                for bx in 0..4 {
                    let index = by * 4 + bx;
                    let part = (by / 2) * 2 + (bx / 2);
                    let x = mx * 16 + bx * 4;
                    let y0 = my * 16 + by * 4;
                    let pred = layer_pred(
                        dpb,
                        (mb.ref0[part], mb.ref1[part]),
                        (mb.slot0[part], mb.slot1[part]),
                        (mb.mv0[index], mb.mv1[index]),
                        (x, y0),
                        4,
                    )?;
                    for row in 0..4 {
                        for col in 0..4 {
                            let at = (y0 + row).saturating_mul(layer.w).saturating_add(x + col);
                            let Some(sample) = layer.y.get(at) else {
                                return Err(CodecError::Malformed {
                                    at_bit: 0,
                                    what: "inter-layer prediction",
                                });
                            };
                            let Some(predicted) = pred.get(row * 4 + col) else {
                                return Err(CodecError::Malformed {
                                    at_bit: 0,
                                    what: "inter-layer prediction",
                                });
                            };
                            y[at] = Into::<i32>::into(*sample) - Into::<i32>::into(*predicted);
                        }
                    }
                    let chroma = layer_chroma(
                        dpb,
                        (mb.ref0[part], mb.ref1[part]),
                        (mb.slot0[part], mb.slot1[part]),
                        (mb.mv0[index], mb.mv1[index]),
                        (x, y0),
                    )?;
                    let cx = mx * 8 + bx * 2;
                    let cy = my * 8 + by * 2;
                    for row in 0..2 {
                        for col in 0..2 {
                            let at = (cy + row).saturating_mul(cw).saturating_add(cx + col);
                            let Some(sample_u) = layer.u.get(at) else {
                                return Err(CodecError::Malformed {
                                    at_bit: 0,
                                    what: "inter-layer prediction",
                                });
                            };
                            let Some(sample_v) = layer.v.get(at) else {
                                return Err(CodecError::Malformed {
                                    at_bit: 0,
                                    what: "inter-layer prediction",
                                });
                            };
                            let Some(pred_u) = chroma.0.get(row * 2 + col) else {
                                return Err(CodecError::Malformed {
                                    at_bit: 0,
                                    what: "inter-layer prediction",
                                });
                            };
                            let Some(pred_v) = chroma.1.get(row * 2 + col) else {
                                return Err(CodecError::Malformed {
                                    at_bit: 0,
                                    what: "inter-layer prediction",
                                });
                            };
                            u[at] = Into::<i32>::into(*sample_u) - Into::<i32>::into(*pred_u);
                            v[at] = Into::<i32>::into(*sample_v) - Into::<i32>::into(*pred_v);
                        }
                    }
                }
            }
        }
    }
    Ok(ResidualPlanes { y, u, v })
}

struct ResidualUp<'a> {
    src: &'a [i32],
    base_w: i32,
    base_h: i32,
    out_w: i32,
    out_h: i32,
    scaled_w: i32,
    scaled_h: i32,
    left: i32,
    top: i32,
    phase_x: i32,
    phase_y: i32,
    ref_phase_x: i32,
    ref_phase_y: i32,
    level: u8,
    mbs: &'a [Mb],
    mb_w: usize,
    chroma: bool,
}

fn upsample_residual(up: ResidualUp<'_>) -> Result<Vec<i32>, CodecError> {
    let ResidualUp {
        src,
        base_w,
        base_h,
        out_w,
        out_h,
        scaled_w,
        scaled_h,
        left,
        top,
        phase_x,
        phase_y,
        ref_phase_x,
        ref_phase_y,
        level,
        mbs,
        mb_w,
        chroma,
    } = up;
    if base_w <= 0 || base_h <= 0 || out_w <= 0 || out_h <= 0 || scaled_w <= 0 || scaled_h <= 0 {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let base_stride = usize::try_from(base_w).map_err(|_| CodecError::Unsupported {
        what: "inter-layer prediction",
    })?;
    let base_rows = usize::try_from(base_h).map_err(|_| CodecError::Unsupported {
        what: "inter-layer prediction",
    })?;
    if src.len() != base_stride.saturating_mul(base_rows) {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "inter-layer prediction",
        });
    }
    let (scale_x, add_x, shift_x, delta_x) =
        resample_axis(base_w, scaled_w, phase_x, ref_phase_x, level)?;
    let (scale_y, add_y, shift_y, delta_y) =
        resample_axis(base_h, scaled_h, phase_y, ref_phase_y, level)?;
    let out_stride = usize::try_from(out_w).map_err(|_| CodecError::Unsupported {
        what: "inter-layer prediction",
    })?;
    let out_rows = usize::try_from(out_h).map_err(|_| CodecError::Unsupported {
        what: "inter-layer prediction",
    })?;
    let mut horizontal = vec![0i32; base_rows.saturating_mul(out_stride)];
    let inside = |at: i32, origin: i32, span: i32| at >= origin && at < origin + span;
    for row in 0..base_h {
        for col in 0..out_w {
            if !inside(col, left, scaled_w) {
                *plane_at_mut(&mut horizontal, out_stride, row, col)? = 0;
                continue;
            }
            let (tap, reference) = resample_pos(col, left, scale_x, add_x, shift_x, delta_x)?;
            let x0 = reference.clamp(0, base_w - 1);
            let x1 = (reference + 1).clamp(0, base_w - 1);
            let s0 = *plane_at(src, base_stride, row, x0)?;
            let s1 = *plane_at(src, base_stride, row, x1)?;
            let blended = if residual_block(mbs, mb_w, x0, row, chroma)
                == residual_block(mbs, mb_w, x1, row, chroma)
            {
                i64::from(16 - tap) * i64::from(s0) + i64::from(tap) * i64::from(s1)
            } else if tap < 8 {
                i64::from(s0) << 4
            } else {
                i64::from(s1) << 4
            };
            *plane_at_mut(&mut horizontal, out_stride, row, col)? = i32::try_from(blended)
                .map_err(|_| CodecError::Unsupported {
                    what: "inter-layer prediction",
                })?;
        }
    }
    let mut output = vec![0i32; out_rows.saturating_mul(out_stride)];
    for col in 0..out_w {
        if !inside(col, left, scaled_w) {
            for row in 0..out_h {
                *plane_at_mut(&mut output, out_stride, row, col)? = 0;
            }
            continue;
        }
        let (tap_x, reference_x) = resample_pos(col, left, scale_x, add_x, shift_x, delta_x)?;
        let ref_x = (reference_x + (tap_x >> 3)).clamp(0, base_w - 1);
        for row in 0..out_h {
            if !inside(row, top, scaled_h) {
                *plane_at_mut(&mut output, out_stride, row, col)? = 0;
                continue;
            }
            let (tap, reference) = resample_pos(row, top, scale_y, add_y, shift_y, delta_y)?;
            let y0 = reference.clamp(0, base_h - 1);
            let y1 = (reference + 1).clamp(0, base_h - 1);
            let h0 = *plane_at(&horizontal, out_stride, y0, col)?;
            let h1 = *plane_at(&horizontal, out_stride, y1, col)?;
            let value = if residual_block(mbs, mb_w, ref_x, y0, chroma)
                == residual_block(mbs, mb_w, ref_x, y1, chroma)
            {
                (i64::from(16 - tap) * i64::from(h0) + i64::from(tap) * i64::from(h1) + 128) >> 8
            } else if tap < 8 {
                (i64::from(h0) + 8) >> 4
            } else {
                (i64::from(h1) + 8) >> 4
            };
            *plane_at_mut(&mut output, out_stride, row, col)? =
                i32::try_from(value).map_err(|_| CodecError::Unsupported {
                    what: "inter-layer prediction",
                })?;
        }
    }
    Ok(output)
}

fn plane_at(plane: &[i32], stride: usize, row: i32, col: i32) -> Result<&i32, CodecError> {
    let row = usize::try_from(row).map_err(|_| CodecError::Unsupported {
        what: "inter-layer prediction",
    })?;
    let col = usize::try_from(col).map_err(|_| CodecError::Unsupported {
        what: "inter-layer prediction",
    })?;
    plane
        .get(row.saturating_mul(stride).saturating_add(col))
        .ok_or(CodecError::Malformed {
            at_bit: 0,
            what: "inter-layer prediction",
        })
}

fn plane_at_mut(
    plane: &mut [i32],
    stride: usize,
    row: i32,
    col: i32,
) -> Result<&mut i32, CodecError> {
    let row = usize::try_from(row).map_err(|_| CodecError::Unsupported {
        what: "inter-layer prediction",
    })?;
    let col = usize::try_from(col).map_err(|_| CodecError::Unsupported {
        what: "inter-layer prediction",
    })?;
    plane
        .get_mut(row.saturating_mul(stride).saturating_add(col))
        .ok_or(CodecError::Malformed {
            at_bit: 0,
            what: "inter-layer prediction",
        })
}

fn residual_block(mbs: &[Mb], mb_w: usize, x: i32, y: i32, chroma: bool) -> i32 {
    if mb_w == 0 || x < 0 || y < 0 {
        return 0;
    }
    let mb_size = if chroma { 8 } else { 16 };
    let mb_x = x / mb_size;
    let mb_y = y / mb_size;
    let Some(width) = i32::try_from(mb_w).ok() else {
        return 0;
    };
    let slot = mb_y * width + mb_x;
    let local_x = x - mb_x * mb_size;
    let local_y = y - mb_y * mb_size;
    let transform8 = !chroma
        && usize::try_from(slot)
            .ok()
            .and_then(|index| mbs.get(index))
            .is_some_and(|mb| mb.transform8);
    if transform8 {
        let id = ((local_y >> 3) << 1) + (local_x >> 3) + (slot << 2);
        (id << 1) + 1
    } else {
        let id = ((local_y >> 2) << 2) + (local_x >> 2) + (slot << 4);
        id << 1
    }
}

fn add_resampled_residual<S: Sample>(
    pic: &mut Frame<S>,
    slice: &Slice<'_>,
    addr: usize,
) -> Result<(), CodecError> {
    if slice.residual_y.len() != pic.w.saturating_mul(pic.h) {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let chroma_w = pic.w / 2;
    let chroma_h = pic.h / 2;
    if slice.residual_u.len() != chroma_w.saturating_mul(chroma_h)
        || slice.residual_v.len() != chroma_w.saturating_mul(chroma_h)
    {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let (mx, my) = locate(pic, addr, false);
    for row in 0..16 {
        for col in 0..16 {
            let x = mx * 16 + col;
            let y = my * 16 + row;
            let at = y.saturating_mul(pic.w).saturating_add(x);
            let Some(delta) = slice.residual_y.get(at) else {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "inter-layer prediction",
                });
            };
            let Some(sample) = pic.y.get(at).copied() else {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "inter-layer prediction",
                });
            };
            let Some(slot) = pic.y.get_mut(at) else {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "inter-layer prediction",
                });
            };
            *slot = S::clip(Into::<i32>::into(sample) + *delta);
        }
    }
    for row in 0..8 {
        for col in 0..8 {
            let x = mx * 8 + col;
            let y = my * 8 + row;
            let at = y.saturating_mul(chroma_w).saturating_add(x);
            add_residual_sample(&mut pic.u, &slice.residual_u, at)?;
            add_residual_sample(&mut pic.v, &slice.residual_v, at)?;
        }
    }
    Ok(())
}

fn add_residual_sample<S: Sample>(
    plane: &mut [S],
    residual: &[i32],
    at: usize,
) -> Result<(), CodecError> {
    let Some(delta) = residual.get(at) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "inter-layer prediction",
        });
    };
    let Some(sample) = plane.get(at).copied() else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "inter-layer prediction",
        });
    };
    let Some(slot) = plane.get_mut(at) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "inter-layer prediction",
        });
    };
    *slot = S::clip(Into::<i32>::into(sample) + *delta);
    Ok(())
}

/// Inter-layer residual prediction. The residual is the reference-layer
/// reconstruction minus that layer's own list-0 prediction.
fn add_layer_residual<S: Sample>(
    pic: &mut Frame<S>,
    dpb: &[Frame<S>],
    layer: &Frame<S>,
    slice: &Slice<'_>,
    addr: usize,
) -> Result<(), CodecError> {
    if slice.sps.mbaff || pic.field || pic.separate_plane || pic.chroma() != 1 {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    if layer.w != pic.w || layer.h != pic.h || layer.mb_w != pic.mb_w {
        return add_resampled_residual(pic, slice, addr);
    }
    let (mx, my) = locate(pic, addr, false);
    let slot = my.saturating_mul(pic.mb_w).saturating_add(mx);
    let Some(reference_mb) = layer.mb.get(slot) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "inter-layer prediction",
        });
    };
    if reference_mb.intra || !has_prediction(reference_mb) {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let motion0 = reference_mb.mv0;
    let motion1 = reference_mb.mv1;
    let refs0 = reference_mb.ref0;
    let refs1 = reference_mb.ref1;
    let slots0 = reference_mb.slot0;
    let slots1 = reference_mb.slot1;
    let chroma_stride = pic.w / 2;
    for by in 0..4 {
        for bx in 0..4 {
            let index = by * 4 + bx;
            let part = (by / 2) * 2 + (bx / 2);
            let x = mx * 16 + bx * 4;
            let y = my * 16 + by * 4;
            let pred = layer_pred(
                dpb,
                (refs0[part], refs1[part]),
                (slots0[part], slots1[part]),
                (motion0[index], motion1[index]),
                (x, y),
                4,
            )?;
            add_predicted(&mut pic.y, &layer.y, &pred, pic.w, x, y, 4)?;
            let chroma = layer_chroma(
                dpb,
                (refs0[part], refs1[part]),
                (slots0[part], slots1[part]),
                (motion0[index], motion1[index]),
                (x, y),
            )?;
            let cx = mx * 8 + bx * 2;
            let cy = my * 8 + by * 2;
            add_predicted(&mut pic.u, &layer.u, &chroma.0, chroma_stride, cx, cy, 2)?;
            add_predicted(&mut pic.v, &layer.v, &chroma.1, chroma_stride, cx, cy, 2)?;
        }
    }
    Ok(())
}

/// G.8.5.2 then G.8.5.1 for one inter macroblock. The reference layer's luma
/// and 4:2:0 chroma levels are rescaled from its QP to this macroblock's QP,
/// this slice's levels are added, and one inverse transform updates the
/// prediction. A macroblock that codes no residual keeps the reference QP.
fn add_tcoeff_residual<S: Sample>(
    pic: &mut Frame<S>,
    layer: &Frame<S>,
    slice: &Slice<'_>,
    addr: usize,
) -> Result<(), CodecError> {
    if slice.sps.mbaff
        || pic.field
        || pic.separate_plane
        || pic.chroma() != 1
        || S::BITS != 8
        || layer.w != pic.w
        || layer.h != pic.h
        || layer.mb_w != pic.mb_w
    {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let (mx, my) = locate(pic, addr, false);
    let slot = my.saturating_mul(pic.mb_w).saturating_add(mx);
    let Some(reference_mb) = layer.mb.get(slot) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "inter-layer prediction",
        });
    };
    if reference_mb.intra || reference_mb.intra16 {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let stride = pic.level_stride();
    let start = slot.saturating_mul(stride);
    if layer.levels.len() < start + stride || pic.levels.len() < start + stride {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "inter-layer prediction",
        });
    }
    let Some(current) = pic.mb.get(slot) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "inter-layer prediction",
        });
    };
    if current.intra16 || reference_mb.transform8 != current.transform8 {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let ref_qp = i32::from(reference_mb.qp);
    // No coded residual: tQPY is the reference macroblock's QP.
    let cur_qp = if current.cbp == 0 {
        ref_qp
    } else {
        i32::from(current.qp)
    };
    let eight = current.transform8;
    if eight {
        let weight = slice.pps.weight8[1];
        for i in 0..4 {
            let mut block = [0i32; 64];
            for k in 0..64 {
                let at = start + i * 64 + k;
                let level =
                    rescale_level(layer.levels[at], ref_qp, cur_qp)?.saturating_add(pic.levels[at]);
                let (ry, rx) = (k / 8, k % 8);
                block[k] = recon::scale8(level, cur_qp + S::QP_BD, rx, ry, i32::from(weight[k]));
            }
            recon::idct8(&mut block);
            let span = luma_span(my, false);
            recon::add_block(
                &mut pic.y,
                pic.w,
                mx * 16 + (i & 1) * 8,
                span.at((i >> 1) * 8),
                &block,
                8,
                span.step,
            );
        }
    } else {
        let weight = slice.pps.weight4[3];
        for i in 0..16 {
            let mut block = [0i32; 16];
            for k in 0..16 {
                let at = start + i * 16 + k;
                let level =
                    rescale_level(layer.levels[at], ref_qp, cur_qp)?.saturating_add(pic.levels[at]);
                let (ry, rx) = (k / 4, k % 4);
                block[k] = recon::scale4(level, cur_qp + S::QP_BD, rx, ry, i32::from(weight[k]));
            }
            recon::idct4(&mut block);
            let span = luma_span(my, false);
            recon::add_block(
                &mut pic.y,
                pic.w,
                mx * 16 + blk_x(i) * 4,
                span.at(blk_y(i) * 4),
                &block,
                4,
                span.step,
            );
        }
    }
    add_tcoeff_chroma(pic, layer, slice, (mx, my, start), (ref_qp, cur_qp))?;
    Ok(())
}

/// G.8.5.1.2.2 for 4:2:0. DC levels take the chroma Hadamard. AC levels take
/// the ordinary 4×4 scale. Both use the chroma QP that belongs to `cur_qp`.
fn add_tcoeff_chroma<S: Sample>(
    pic: &mut Frame<S>,
    layer: &Frame<S>,
    slice: &Slice<'_>,
    at: (usize, usize, usize),
    qp: (i32, i32),
) -> Result<(), CodecError> {
    let (mx, my, start) = at;
    let (ref_qp, cur_qp) = qp;
    let chroma_stride = pic.w / 2;
    let offsets = [
        slice.pps.chroma_qp_offset,
        slice.pps.second_chroma_qp_offset,
    ];
    for (plane, offset) in offsets.into_iter().enumerate() {
        let (ref_c, _) = recon::chroma_qps(ref_qp, offset, S::QP_BD);
        let (cur_c, _) = recon::chroma_qps(cur_qp, offset, S::QP_BD);
        let list = 4 + plane;
        let weight = slice.pps.weight4[list];
        let origin = start + 256 + plane * 64;
        let mut dc = [0i32; 4];
        let mut ac = [[0i32; 16]; 4];
        for block in 0..4 {
            let coeffs = &mut ac[block];
            for (k, coeff) in coeffs.iter_mut().enumerate() {
                let index = origin + block * 16 + k;
                let level = rescale_level(layer.levels[index], ref_c, cur_c)?
                    .saturating_add(pic.levels[index]);
                if k == 0 {
                    dc[block] = level;
                } else {
                    *coeff = level;
                }
            }
        }
        let scaled = recon::chroma_dc(dc[0], dc[1], dc[2], dc[3], cur_c, i32::from(weight[0]));
        let plane_samples = if plane == 0 { &mut pic.u } else { &mut pic.v };
        let span = chroma_span(my, 8, false);
        for i in 0..4 {
            let mut block = [0i32; 16];
            for k in 0..16 {
                let (ry, rx) = (k / 4, k % 4);
                block[k] = if k == 0 {
                    scaled[i]
                } else {
                    recon::scale4(ac[i][k], cur_c, rx, ry, i32::from(weight[k]))
                };
            }
            recon::idct4(&mut block);
            recon::add_block(
                plane_samples,
                chroma_stride,
                mx * 8 + (i & 1) * 4,
                span.at((i >> 1) * 4),
                &block,
                4,
                span.step,
            );
        }
    }
    Ok(())
}

/// Table G-6 and equation G-172. The shift divides toward zero.
fn rescale_level(level: i32, ref_qp: i32, cur_qp: i32) -> Result<i32, CodecError> {
    const SCALE: [i32; 6] = [8, 9, 10, 11, 13, 14];
    let delta = ref_qp - cur_qp + 54;
    if !(0..108).contains(&delta) {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let index = usize::try_from(delta.rem_euclid(6)).unwrap_or(0);
    let shift = delta.div_euclid(6);
    if !(0..32).contains(&shift) {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let magnitude = (i64::from(SCALE[index]) * i64::from(level.unsigned_abs())) << shift;
    let scaled = i32::try_from(magnitude >> 12).unwrap_or(i32::MAX);
    Ok(if level < 0 { -scaled } else { scaled })
}

fn has_prediction(mb: &Mb) -> bool {
    (0..4).all(|index| mb.ref0[index] >= 0 || mb.ref1[index] >= 0)
}

fn layer_pred<S: Sample>(
    dpb: &[Frame<S>],
    refs: (i8, i8),
    slots: (i8, i8),
    motion: ([i16; 2], [i16; 2]),
    origin: (usize, usize),
    n: usize,
) -> Result<Vec<S>, CodecError> {
    let (x, y) = origin;
    let mut pred = vec![S::default(); n * n];
    if refs.0 >= 0 {
        let frame = dpb_slot(dpb, slots.0)?;
        predict_luma(&mut pred, frame, x, y, n, n, motion.0);
    }
    if refs.1 >= 0 {
        let frame = dpb_slot(dpb, slots.1)?;
        let mut other = vec![S::default(); n * n];
        predict_luma(&mut other, frame, x, y, n, n, motion.1);
        if refs.0 >= 0 {
            for (sample, averaged) in pred.iter_mut().zip(other) {
                *sample = recon::avg(*sample, averaged);
            }
        } else {
            pred = other;
        }
    }
    Ok(pred)
}

fn layer_chroma<S: Sample>(
    dpb: &[Frame<S>],
    refs: (i8, i8),
    slots: (i8, i8),
    motion: ([i16; 2], [i16; 2]),
    origin: (usize, usize),
) -> Result<([S; 4], [S; 4]), CodecError> {
    let mut cu = [S::default(); 4];
    let mut cv = [S::default(); 4];
    if refs.0 >= 0 {
        let frame = dpb_slot(dpb, slots.0)?;
        predict_chroma((&mut cu, &mut cv), 2, frame, origin, (2, 2), motion.0);
    }
    if refs.1 >= 0 {
        let frame = dpb_slot(dpb, slots.1)?;
        let mut ou = [S::default(); 4];
        let mut ov = [S::default(); 4];
        predict_chroma((&mut ou, &mut ov), 2, frame, origin, (2, 2), motion.1);
        if refs.0 >= 0 {
            for i in 0..4 {
                cu[i] = recon::avg(cu[i], ou[i]);
                cv[i] = recon::avg(cv[i], ov[i]);
            }
        } else {
            cu = ou;
            cv = ov;
        }
    }
    Ok((cu, cv))
}

fn dpb_slot<S: Sample>(dpb: &[Frame<S>], slot: i8) -> Result<&Frame<S>, CodecError> {
    let Ok(index) = usize::try_from(slot) else {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    };
    let Some(frame) = dpb.get(index) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "inter-layer prediction",
        });
    };
    if frame.nonexisting {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "inter-layer prediction",
        });
    }
    Ok(frame)
}

fn add_predicted<S: Sample>(
    dst: &mut [S],
    layer: &[S],
    predicted: &[S],
    stride: usize,
    x: usize,
    y: usize,
    n: usize,
) -> Result<(), CodecError> {
    for row in 0..n {
        let at = (y + row).saturating_mul(stride).saturating_add(x);
        let Some(into) = dst.get_mut(at..at + n) else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "inter-layer prediction",
            });
        };
        let Some(from_layer) = layer.get(at..at + n) else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "inter-layer prediction",
            });
        };
        let Some(from_pred) = predicted.get(row * n..row * n + n) else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "inter-layer prediction",
            });
        };
        for i in 0..n {
            let delta = Into::<i32>::into(from_layer[i]) - Into::<i32>::into(from_pred[i]);
            into[i] = S::clip(Into::<i32>::into(into[i]) + delta);
        }
    }
    Ok(())
}

fn copy_reference<S: Sample>(
    pic: &mut Frame<S>,
    layer: &Frame<S>,
    mx: usize,
    my: usize,
) -> Result<(), CodecError> {
    if layer.w != pic.w || layer.h != pic.h || layer.mb_w != pic.mb_w {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let slot = my * pic.mb_w + mx;
    let Some(reference) = layer.mb.get(slot) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "inter-layer prediction",
        });
    };
    if !reference.intra {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let x0 = mx.saturating_mul(16);
    let y0 = my.saturating_mul(16);
    if x0.saturating_add(16) > pic.w || y0.saturating_add(16) > pic.h {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "inter-layer prediction",
        });
    }
    for row in 0..16 {
        let src = (y0 + row) * pic.w + x0;
        let Some(from) = layer.y.get(src..src + 16) else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "inter-layer prediction",
            });
        };
        let Some(into) = pic.y.get_mut(src..src + 16) else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "inter-layer prediction",
            });
        };
        into.copy_from_slice(from);
    }
    let stride = pic.w / 2;
    let cx = mx.saturating_mul(8);
    let cy = my.saturating_mul(8);
    for row in 0..8 {
        let src = (cy + row) * stride + cx;
        let Some(from) = layer.u.get(src..src + 8) else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "inter-layer prediction",
            });
        };
        let Some(into) = pic.u.get_mut(src..src + 8) else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "inter-layer prediction",
            });
        };
        into.copy_from_slice(from);
        let Some(from) = layer.v.get(src..src + 8) else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "inter-layer prediction",
            });
        };
        let Some(into) = pic.v.get_mut(src..src + 8) else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "inter-layer prediction",
            });
        };
        into.copy_from_slice(from);
    }
    Ok(())
}

fn one_mb<S: Sample>(
    pic: &mut Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    slice_id: u16,
    addr: usize,
    coded: (i32, Option<&Frame<S>>),
    entropy: &mut Entropy<'_, '_>,
) -> Result<(i32, bool), CodecError> {
    let (qp, layer) = coded;
    let mbaff = slice.sps.mbaff && !pic.field;
    let (mx, my) = locate(pic, addr, mbaff);
    let slot = my * pic.mb_w + mx;
    let mut mb = Mb::new(slice_id);
    mb.field_mb = field_flag(pic, entropy, slice_id, mx, my, mbaff)?;
    mb.off_a = slice.off_a as i8;
    mb.off_b = slice.off_b as i8;
    mb.deblock = slice.deblock;
    pic.mb[slot] = mb.clone();
    let (intra, mut spec) = mb_type(pic, entropy, slice.kind, mx, my, slice_id)?;
    if spec == SI_MB {
        mb.si = true;
        spec = 0;
    }
    mb.intra = intra;
    mb.direct16 = !intra && slice.kind == 1 && spec == 0;
    if intra && spec == 25 {
        mb.pcm = true;
        read_pcm(pic, entropy, mx, my, mb.field_mb, slice)?;
        mb.nnz_y = [16; 16];
        mb.nnz_c = [16; 32];
        mb.qp = 0;
        pic.mb[slot] = mb;
        return Ok((qp, false));
    }
    let mut qp = qp;
    let residual = if intra {
        decode_intra(
            pic,
            entropy,
            slice,
            Place { mx, my },
            spec,
            &mut mb,
            &mut qp,
        )?;
        false
    } else {
        decode_inter(
            pic,
            dpb,
            entropy,
            slice,
            Place { mx, my },
            spec,
            (&mut mb, &mut qp, layer),
        )?
    };
    mb.qp = qp as i8;
    mb.qp_c = chroma_deblock::<S>(qp, slice.pps.chroma_qp_offset);
    mb.qp_cr = chroma_deblock::<S>(qp, slice.pps.second_chroma_qp_offset);
    if !mb.intra {
        bind_refs(&mut mb, slice);
    }
    pic.mb[slot] = mb;
    Ok((qp, residual))
}

fn field_flag<S: Sample>(
    pic: &Frame<S>,
    entropy: &mut Entropy<'_, '_>,
    slice_id: u16,
    mx: usize,
    my: usize,
    mbaff: bool,
) -> Result<bool, CodecError> {
    if !mbaff {
        return Ok(false);
    }
    if my % 2 == 1 {
        return Ok(pic
            .mb
            .get((my - 1) * pic.mb_w + mx)
            .is_some_and(|mb| mb.field_mb));
    }
    match entropy {
        Entropy::Cavlc(bits) => bits.bit(),
        Entropy::Cabac(cab) => {
            let mut ctx = 0;
            if mx > 0
                && neighbor(pic, mx as i32 - 1, my as i32, slice_id).is_some_and(|mb| mb.field_mb)
            {
                ctx += 1;
            }
            if my >= 2
                && neighbor(pic, mx as i32, my as i32 - 2, slice_id).is_some_and(|mb| mb.field_mb)
            {
                ctx += 1;
            }
            Ok(cab.decision(70 + ctx)? == 1)
        }
    }
}

/// CAVLC reads `mb_field_decoding_flag` on the last skipped macroblock of a pair.
fn skip_field_flag<S: Sample>(
    pic: &Frame<S>,
    bits: &mut crate::bits::Bits<'_>,
    slice_id: u16,
    addr: usize,
    mbaff: bool,
    remaining: usize,
) -> Result<bool, CodecError> {
    if !mbaff {
        return Ok(false);
    }
    let (mx, my) = locate(pic, addr, true);
    if my % 2 == 1 {
        return Ok(pic
            .mb
            .get((my - 1) * pic.mb_w + mx)
            .is_some_and(|mb| mb.field_mb));
    }
    if remaining == 1 {
        return bits.bit();
    }
    Ok(inferred_field(pic, mx, my, slice_id))
}

fn cabac_skip_field<S: Sample>(
    pic: &Frame<S>,
    cab: &mut Cabac<'_>,
    slice_id: u16,
    addr: usize,
    mbaff: bool,
) -> Result<bool, CodecError> {
    if !mbaff {
        return Ok(false);
    }
    let (mx, my) = locate(pic, addr, true);
    if my % 2 == 1 {
        return Ok(pic
            .mb
            .get((my - 1) * pic.mb_w + mx)
            .is_some_and(|mb| mb.field_mb));
    }
    field_flag(pic, &mut Entropy::Cabac(cab), slice_id, mx, my, true)
}

fn inferred_field<S: Sample>(pic: &Frame<S>, mx: usize, my: usize, slice: u16) -> bool {
    if let Some(left) = neighbor(pic, mx as i32 - 1, my as i32, slice) {
        return left.field_mb;
    }
    neighbor(pic, mx as i32, my as i32 - 2, slice).is_some_and(|mb| mb.field_mb)
}

fn locate<S: Sample>(pic: &Frame<S>, addr: usize, mbaff: bool) -> (usize, usize) {
    if !mbaff {
        return xy(pic, addr);
    }
    let pair = addr / 2;
    let mx = pair % pic.mb_w;
    let row = pair / pic.mb_w;
    (mx, row * 2 + (addr % 2))
}

#[derive(Clone, Copy)]
struct Span {
    y: usize,
    step: usize,
}

fn luma_span(my: usize, field: bool) -> Span {
    if field {
        Span {
            y: (my & !1) * 16 + (my & 1),
            step: 2,
        }
    } else {
        Span {
            y: my * 16,
            step: 1,
        }
    }
}

fn chroma_span(my: usize, rows: usize, field: bool) -> Span {
    if field {
        Span {
            y: (my & !1) * rows + (my & 1),
            step: 2,
        }
    } else {
        Span {
            y: my * rows,
            step: 1,
        }
    }
}

impl Span {
    fn at(self, local: usize) -> usize {
        self.y + local * self.step
    }
}

fn block_span<S: Sample>(pic: &Frame<S>, my: usize, field: bool, plane: u8) -> Span {
    if plane == 0 || pic.chroma() == 3 {
        luma_span(my, field)
    } else {
        let rows = if pic.chroma() == 2 { 16 } else { 8 };
        chroma_span(my, rows, field)
    }
}

fn above_my<S: Sample>(pic: &Frame<S>, mx: usize, my: usize) -> i32 {
    let field = pic.mb.get(my * pic.mb_w + mx).is_some_and(|mb| mb.field_mb);
    my as i32 - if field { 2 } else { 1 }
}

/// CAVLC `mb_type` 0 in an SI slice. It is not an intra spec.
const SI_MB: u32 = 32;

fn mb_type<S: Sample>(
    pic: &Frame<S>,
    entropy: &mut Entropy<'_, '_>,
    kind: u8,
    mx: usize,
    my: usize,
    slice: u16,
) -> Result<(bool, u32), CodecError> {
    match entropy {
        Entropy::Cavlc(bits) => {
            let value = bits.ue()?;
            if kind == 4 {
                if value == 0 {
                    return Ok((true, SI_MB));
                }
                if value > 26 {
                    return Err(bits.malformed("intra mb type"));
                }
                return Ok((true, value - 1));
            }
            if kind == 2 {
                return Ok((true, value));
            }
            let intra_base = if super::syntax_kind(kind) == 0 { 5 } else { 23 };
            if value >= intra_base {
                Ok((true, value - intra_base))
            } else {
                Ok((false, value))
            }
        }
        Entropy::Cabac(cab) => cabac_mb_type(pic, cab, kind, mx, my, slice),
    }
}

fn cabac_mb_type<S: Sample>(
    pic: &Frame<S>,
    cab: &mut Cabac<'_>,
    kind: u8,
    mx: usize,
    my: usize,
    slice: u16,
) -> Result<(bool, u32), CodecError> {
    if kind == 1 {
        let ctx = direct_ctx(pic, mx, my, slice);
        if cab.decision(27 + ctx)? == 0 {
            return Ok((false, 0));
        }
        if cab.decision(27 + 3)? == 0 {
            return Ok((false, 1 + u32::from(cab.decision(27 + 5)?)));
        }
        let mut bits = u32::from(cab.decision(27 + 4)?) << 3;
        bits += u32::from(cab.decision(27 + 5)?) << 2;
        bits += u32::from(cab.decision(27 + 5)?) << 1;
        bits += u32::from(cab.decision(27 + 5)?);
        if bits < 8 {
            return Ok((false, bits + 3));
        }
        if bits == 13 {
            return Ok((true, cabac_intra_type(pic, cab, 32, false, mx, my, slice)?));
        }
        if bits == 14 {
            return Ok((false, 11));
        }
        if bits == 15 {
            return Ok((false, 22));
        }
        bits = (bits << 1) + u32::from(cab.decision(27 + 5)?);
        return Ok((false, bits - 4));
    }
    if super::syntax_kind(kind) == 0 {
        if cab.decision(14)? == 0 {
            if cab.decision(15)? == 0 {
                return Ok((false, 3 * u32::from(cab.decision(16)?)));
            }
            return Ok((false, 2 - u32::from(cab.decision(17)?)));
        }
        return Ok((true, cabac_intra_type(pic, cab, 17, false, mx, my, slice)?));
    }
    if kind == 4 {
        if cab.decision(0)? == 0 {
            return Ok((true, SI_MB));
        }
        return Ok((true, cabac_intra_type(pic, cab, 3, true, mx, my, slice)?));
    }
    Ok((true, cabac_intra_type(pic, cab, 3, true, mx, my, slice)?))
}

fn direct_ctx<S: Sample>(pic: &Frame<S>, mx: usize, my: usize, slice: u16) -> usize {
    // Clause 9.3.3.1.1.3: a missing neighbor, B_Skip, or B_Direct_16x16 contributes 0.
    let coded = |mb: &Mb| !mb.skip && !mb.direct16;
    let left = neighbor(pic, mx as i32 - 1, my as i32, slice).is_some_and(coded);
    let above = neighbor(pic, mx as i32, above_my(pic, mx, my), slice).is_some_and(coded);
    usize::from(left) + usize::from(above)
}

fn cabac_intra_type<S: Sample>(
    pic: &Frame<S>,
    cab: &mut Cabac<'_>,
    base: usize,
    intra_slice: bool,
    mx: usize,
    my: usize,
    slice: u16,
) -> Result<u32, CodecError> {
    let intra16 = |mb: &Mb| mb.intra16 || mb.pcm;
    if intra_slice {
        let mut ctx = 0;
        if neighbor(pic, mx as i32 - 1, my as i32, slice).is_some_and(intra16) {
            ctx += 1;
        }
        if neighbor(pic, mx as i32, above_my(pic, mx, my), slice).is_some_and(intra16) {
            ctx += 1;
        }
        if cab.decision(base + ctx)? == 0 {
            return Ok(0);
        }
    } else if cab.decision(base)? == 0 {
        return Ok(0);
    }
    if cab.terminate()? {
        return Ok(25);
    }
    let shift = usize::from(intra_slice);
    // An I slice skips the two prefix contexts before the I_16x16 suffix.
    let origin = base + shift * 2;
    let mut kind = 1u32;
    kind += 12 * u32::from(cab.decision(origin + 1)?);
    if cab.decision(origin + 2)? == 1 {
        kind += 4 + 4 * u32::from(cab.decision(origin + 2 + shift)?);
    }
    kind += 2 * u32::from(cab.decision(origin + 3 + shift)?);
    kind += u32::from(cab.decision(origin + 3 + 2 * shift)?);
    Ok(kind)
}

fn decode_intra<S: Sample>(
    pic: &mut Frame<S>,
    entropy: &mut Entropy<'_, '_>,
    slice: &Slice<'_>,
    at: Place,
    spec: u32,
    mb: &mut Mb,
    qp: &mut i32,
) -> Result<(), CodecError> {
    let Place { mx, my } = at;
    if spec > 25 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "intra mb type",
        });
    }
    let i16 = (1..25).contains(&spec);
    mb.intra16 = i16;
    let (mode16, cbp_luma, cbp_chroma) = if i16 {
        let mode = ((spec - 1) % 4) as u8;
        let chroma = (((spec - 1) / 4) % 3) as u8;
        let luma = u8::from(spec >= 13) * 15;
        (mode, luma, chroma)
    } else {
        (0, 0, 0)
    };
    let mut transform8 = false;
    if !i16 && slice.pps.transform_8x8 {
        transform8 = transform8_flag(pic, entropy, mx, my, mb.slice)?;
    }
    mb.transform8 = transform8;
    if !i16 {
        if transform8 {
            read_intra8(pic, entropy, slice, mx, my, mb)?;
        } else {
            read_intra4(pic, entropy, slice, mx, my, mb)?;
        }
    }
    if slice.sps.chroma == 3 && i16 && cbp_chroma != 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "chroma cbp",
        });
    }
    let chroma = if slice.sps.chroma == 3 {
        0
    } else {
        read_chroma_mode(pic, entropy, mx, my, mb.slice)?
    };
    mb.chroma_mode = chroma;
    let (luma_cbp, chroma_cbp) = if i16 {
        (cbp_luma, u8::from(slice.sps.chroma != 3) * cbp_chroma)
    } else {
        read_cbp(pic, entropy, mx, my, true, mb.slice, slice.sps.chroma)?
    };
    mb.cbp = luma_cbp | (chroma_cbp << 4);
    if luma_cbp > 0 && slice.pps.transform_8x8 && !i16 && !transform8 {
        // Inter-only syntax position. Intra already consumed the flag.
    }
    let mut delta = 0;
    if luma_cbp > 0 || chroma_cbp > 0 || i16 {
        delta = qp_delta(entropy, prev_qpd(pic, mx, my, mb.slice))?;
        *qp = next_qp::<S>(*qp, delta);
    }
    mb.qpd = delta as i8;
    let mut luma_dc = [0i32; 16];
    let mut ac = [[0i32; 16]; 16];
    let mut block8 = [[0i32; 64]; 4];
    if i16 {
        luma_dc = read_block(
            pic,
            entropy,
            (0, 16, true),
            Place { mx, my },
            Cell { bx: 0, by: 0 },
            mb,
            0,
        )?;
        if luma_cbp != 0 {
            for (i, slot) in ac.iter_mut().enumerate() {
                *slot = read_block(
                    pic,
                    entropy,
                    (1, 15, false),
                    Place { mx, my },
                    Cell {
                        bx: blk_x(i),
                        by: blk_y(i),
                    },
                    mb,
                    0,
                )?;
            }
        }
    } else if transform8 {
        for (i, slot) in block8.iter_mut().enumerate() {
            if luma_cbp & (1 << i) != 0 {
                *slot = read8(pic, entropy, mx, my, i, mb, 0)?;
            }
        }
    } else {
        for (i, slot) in ac.iter_mut().enumerate() {
            let bit = 1 << ((blk_y(i) / 2) * 2 + blk_x(i) / 2);
            if luma_cbp & bit != 0 {
                *slot = read_block(
                    pic,
                    entropy,
                    (2, 16, false),
                    Place { mx, my },
                    Cell {
                        bx: blk_x(i),
                        by: blk_y(i),
                    },
                    mb,
                    0,
                )?;
            }
        }
    }
    let chroma_res = if slice.sps.chroma == 3 {
        ChromaResidual {
            cb_dc: [0; 8],
            cr_dc: [0; 8],
            cb_ac: [[0; 16]; 8],
            cr_ac: [[0; 16]; 8],
        }
    } else {
        read_chroma(pic, entropy, mx, my, chroma_cbp, mb)?
    };
    if mb.si {
        recon_si(pic, slice, Place { mx, my }, mb, &ac, &chroma_res, *qp)?;
        return Ok(());
    }
    recon_intra(
        pic,
        slice,
        Place { mx, my },
        mb,
        LumaPaint {
            i16,
            mode16,
            chroma_mode: chroma,
            dc: luma_dc,
            ac: &ac,
            block8: &block8,
        },
        ChromaPaint {
            cb_dc: &chroma_res.cb_dc,
            cr_dc: &chroma_res.cr_dc,
            cb_ac: &chroma_res.cb_ac,
            cr_ac: &chroma_res.cr_ac,
        },
        *qp,
    );
    if slice.sps.chroma == 3 && !slice.sps.separate {
        recon_444(
            pic,
            entropy,
            slice,
            Place { mx, my },
            mb,
            *qp,
            i16.then_some(mode16),
        )?;
    }
    Ok(())
}

fn read_intra4<S: Sample>(
    pic: &Frame<S>,
    entropy: &mut Entropy<'_, '_>,
    slice: &Slice<'_>,
    mx: usize,
    my: usize,
    mb: &mut Mb,
) -> Result<u8, CodecError> {
    for i in 0..16 {
        let bx = blk_x(i);
        let by = blk_y(i);
        let pred = pred_mode(
            mode_at(
                pic,
                mx,
                my,
                bx as i32 - 1,
                by as i32,
                mb,
                slice.pps.constrained_intra,
            ),
            mode_at(
                pic,
                mx,
                my,
                bx as i32,
                by as i32 - 1,
                mb,
                slice.pps.constrained_intra,
            ),
        );
        let mode = read_mode(entropy, pred)?;
        mb.mode4[by * 4 + bx] = mode;
    }
    let _ = slice;
    Ok(0)
}

fn read_intra8<S: Sample>(
    pic: &Frame<S>,
    entropy: &mut Entropy<'_, '_>,
    slice: &Slice<'_>,
    mx: usize,
    my: usize,
    mb: &mut Mb,
) -> Result<u8, CodecError> {
    for i in 0..4 {
        let bx = (i & 1) * 2;
        let by = (i >> 1) * 2;
        let pred = pred_mode(
            mode_at(
                pic,
                mx,
                my,
                bx as i32 - 1,
                by as i32,
                mb,
                slice.pps.constrained_intra,
            ),
            mode_at(
                pic,
                mx,
                my,
                bx as i32,
                by as i32 - 1,
                mb,
                slice.pps.constrained_intra,
            ),
        );
        let mode = read_mode(entropy, pred)?;
        for dy in 0..2 {
            for dx in 0..2 {
                mb.mode4[(by + dy) * 4 + bx + dx] = mode;
            }
        }
    }
    let _ = slice;
    Ok(0)
}

fn mode_at<S: Sample>(
    pic: &Frame<S>,
    mx: usize,
    my: usize,
    bx: i32,
    by: i32,
    current: &Mb,
    constrained: bool,
) -> Option<u8> {
    let (mbx, ox) = if bx < 0 {
        (mx as i32 - 1, 3)
    } else if bx > 3 {
        (mx as i32 + 1, 0)
    } else {
        (mx as i32, bx as usize)
    };
    let (mby, oy) = if by < 0 {
        (above_my(pic, mx, my), 3)
    } else if by > 3 {
        (my as i32 + 1, 0)
    } else {
        (my as i32, by as usize)
    };
    if mbx == mx as i32 && mby == my as i32 {
        return Some(current.mode4[oy * 4 + ox]);
    }
    let mb = neighbor(pic, mbx, mby, current.slice)?;
    // Clause 8.3.1.1: a neighbour that is not Intra_4x4 or Intra_8x8 contributes DC.
    // An inter neighbour with constrained intra makes both predictors DC.
    if !mb.intra && constrained {
        return None;
    }
    if mb.intra16 || mb.pcm || !mb.intra {
        return Some(2);
    }
    Some(mb.mode4[oy * 4 + ox])
}

fn pred_mode(left: Option<u8>, above: Option<u8>) -> u8 {
    match (left, above) {
        (Some(left), Some(above)) => left.min(above),
        _ => 2,
    }
}

fn read_mode(entropy: &mut Entropy<'_, '_>, pred: u8) -> Result<u8, CodecError> {
    match entropy {
        Entropy::Cavlc(bits) => {
            if bits.bit()? {
                Ok(pred)
            } else {
                let rem = bits.u(3)? as u8;
                Ok(if rem < pred { rem } else { rem + 1 })
            }
        }
        Entropy::Cabac(cab) => {
            if cab.decision(68)? == 1 {
                return Ok(pred);
            }
            let mut rem = 0u8;
            for shift in 0..3 {
                rem |= cab.decision(69)? << shift;
            }
            Ok(if rem < pred { rem } else { rem + 1 })
        }
    }
}

fn read_chroma_mode<S: Sample>(
    pic: &Frame<S>,
    entropy: &mut Entropy<'_, '_>,
    mx: usize,
    my: usize,
    slice: u16,
) -> Result<u8, CodecError> {
    match entropy {
        Entropy::Cavlc(bits) => {
            let mode = bits.ue()?;
            if mode > 3 {
                return Err(bits.malformed("chroma mode"));
            }
            Ok(mode as u8)
        }
        Entropy::Cabac(cab) => {
            let mut ctx = 0;
            if neighbor(pic, mx as i32 - 1, my as i32, slice)
                .is_some_and(|mb| mb.intra && !mb.pcm && mb.chroma_mode != 0)
            {
                ctx += 1;
            }
            if neighbor(pic, mx as i32, above_my(pic, mx, my), slice)
                .is_some_and(|mb| mb.intra && !mb.pcm && mb.chroma_mode != 0)
            {
                ctx += 1;
            }
            if cab.decision(64 + ctx)? == 0 {
                return Ok(0);
            }
            if cab.decision(64 + 3)? == 0 {
                return Ok(1);
            }
            if cab.decision(64 + 3)? == 0 {
                Ok(2)
            } else {
                Ok(3)
            }
        }
    }
}

fn read_cbp<S: Sample>(
    pic: &Frame<S>,
    entropy: &mut Entropy<'_, '_>,
    mx: usize,
    my: usize,
    intra: bool,
    slice: u16,
    chroma: u8,
) -> Result<(u8, u8), CodecError> {
    match entropy {
        Entropy::Cavlc(bits) => {
            let code = bits.ue()? as usize;
            if chroma == 3 {
                let Some(value) = (if intra {
                    super::CBP_INTRA_444
                } else {
                    super::CBP_INTER_444
                })
                .get(code)
                .copied() else {
                    return Err(bits.malformed("cbp"));
                };
                return Ok((value, 0));
            }
            if code >= 48 {
                return Err(bits.malformed("cbp"));
            }
            let value = if intra {
                super::CBP_INTRA[code]
            } else {
                super::CBP_INTER[code]
            };
            Ok((value & 15, value >> 4))
        }
        Entropy::Cabac(cab) => {
            let mut luma = 0u8;
            for bit in 0..4 {
                let ctx = cbp_ctx(pic, mx, my, bit, luma, slice);
                if cab.decision(73 + ctx)? == 1 {
                    luma |= 1 << bit;
                }
            }
            if chroma == 3 {
                return Ok((luma, 0));
            }
            let left = cbp_chroma(pic, mx as i32 - 1, my as i32, slice);
            let above = cbp_chroma(pic, mx as i32, above_my(pic, mx, my), slice);
            let mut ctx = usize::from(left > 0) + 2 * usize::from(above > 0);
            if cab.decision(77 + ctx)? == 0 {
                return Ok((luma, 0));
            }
            ctx = 4 + usize::from(left == 2) + 2 * usize::from(above == 2);
            Ok((luma, 1 + cab.decision(77 + ctx)?))
        }
    }
}

fn cbp_ctx<S: Sample>(
    pic: &Frame<S>,
    mx: usize,
    my: usize,
    bit: usize,
    current: u8,
    slice: u16,
) -> usize {
    let (left_bit, above_bit) = match bit {
        0 => (0x02, 0x04),
        1 => (0x01, 0x08),
        2 => (0x08, 0x01),
        _ => (0x04, 0x02),
    };
    let left_coded = if bit == 1 || bit == 3 {
        current & left_bit != 0
    } else {
        coded_bit(pic, mx as i32 - 1, my as i32, left_bit, slice)
    };
    let above_coded = if bit == 2 || bit == 3 {
        current & above_bit != 0
    } else {
        coded_bit(pic, mx as i32, above_my(pic, mx, my), above_bit, slice)
    };
    usize::from(!left_coded) + 2 * usize::from(!above_coded)
}

fn coded_bit<S: Sample>(pic: &Frame<S>, mx: i32, my: i32, bit: u8, slice: u16) -> bool {
    match neighbor(pic, mx, my, slice) {
        Some(mb) => mb.pcm || mb.cbp & bit != 0,
        None => true,
    }
}

fn cbp_chroma<S: Sample>(pic: &Frame<S>, mx: i32, my: i32, slice: u16) -> u8 {
    match neighbor(pic, mx, my, slice) {
        Some(mb) if mb.pcm => 2,
        Some(mb) => mb.cbp >> 4,
        None => 0,
    }
}

/// Clause 9.3.3.1.1.5. The first bin uses context 60 or 61. Later bins use 62, then 63.
fn qp_delta(entropy: &mut Entropy<'_, '_>, prev_nonzero: bool) -> Result<i32, CodecError> {
    match entropy {
        Entropy::Cavlc(bits) => bits.se(),
        Entropy::Cabac(cab) => {
            let mut ctx = usize::from(prev_nonzero);
            let mut code = 0u32;
            loop {
                if cab.decision(60 + ctx)? == 0 {
                    break;
                }
                code += 1;
                if code > 52 {
                    return Err(CodecError::Malformed {
                        at_bit: 0,
                        what: "qp delta",
                    });
                }
                ctx = if code == 1 { 2 } else { 3 };
            }
            Ok(if code.is_multiple_of(2) {
                -((code / 2) as i32)
            } else {
                (code / 2) as i32 + 1
            })
        }
    }
}

fn prev_qpd<S: Sample>(pic: &Frame<S>, mx: usize, my: usize, slice: u16) -> bool {
    let earlier = if pic.mbaff && my % 2 == 1 {
        (mx as i32, my as i32 - 1)
    } else if pic.mbaff && mx > 0 {
        (mx as i32 - 1, my as i32 + 1)
    } else {
        let addr = my * pic.mb_w + mx;
        if addr == 0 {
            return false;
        }
        return pic
            .mb
            .get(addr - 1)
            .is_some_and(|mb| mb.slice == slice && mb.qpd != 0);
    };
    coded_mb(pic, earlier.0, earlier.1).is_some_and(|mb| mb.slice == slice && mb.qpd != 0)
}

fn transform8_flag<S: Sample>(
    pic: &Frame<S>,
    entropy: &mut Entropy<'_, '_>,
    mx: usize,
    my: usize,
    slice: u16,
) -> Result<bool, CodecError> {
    let on = |mb: &Mb| mb.transform8;
    let left = neighbor(pic, mx as i32 - 1, my as i32, slice).is_some_and(on);
    let above = neighbor(pic, mx as i32, above_my(pic, mx, my), slice).is_some_and(on);
    flag(entropy, 399 + usize::from(left) + usize::from(above))
}

fn flag(entropy: &mut Entropy<'_, '_>, ctx: usize) -> Result<bool, CodecError> {
    match entropy {
        Entropy::Cavlc(bits) => bits.bit(),
        Entropy::Cabac(cab) => Ok(cab.decision(ctx)? == 1),
    }
}

const CBF_BASE: [usize; 14] = [
    85, 89, 93, 97, 101, 1012, 460, 464, 468, 1016, 472, 476, 480, 1020,
];

/// ctxBlockCat for a chroma plane that uses the luma residual process.
fn block_cat(cat: usize, plane: u8, chroma444: bool) -> usize {
    if !chroma444 || plane == 0 {
        return cat;
    }
    let row = match cat {
        0 => 0,
        1 => 1,
        2 => 2,
        5 => 3,
        _ => return cat,
    };
    const MAP: [[usize; 3]; 4] = [[0, 6, 10], [1, 7, 11], [2, 8, 12], [5, 9, 13]];
    MAP[row][usize::from(plane.min(2))]
}

fn chroma_slot(chroma: u8, plane: u8, bx: usize, by: usize) -> usize {
    let across = if chroma == 3 { 4 } else { 2 };
    usize::from(plane.saturating_sub(1)) * 16 + by * across + bx
}

fn chroma_pos(index: usize, blocks: usize) -> (usize, usize) {
    if blocks == 8 {
        let sub = index & 3;
        let band = index >> 2;
        (sub & 1, band * 2 + (sub >> 1))
    } else {
        (index & 1, index >> 1)
    }
}

fn read_block<S: Sample>(
    pic: &Frame<S>,
    entropy: &mut Entropy<'_, '_>,
    ask: (usize, usize, bool),
    at: Place,
    cell: Cell,
    mb: &mut Mb,
    plane: u8,
) -> Result<[i32; 16], CodecError> {
    let (cat, max, dc) = ask;
    let Place { mx, my } = at;
    let Cell { bx, by } = cell;
    let mut out = [0i32; 16];
    let coeffs = match entropy {
        Entropy::Cavlc(bits) => {
            let nc = if cat == 3 {
                if max == 8 { -2 } else { -1 }
            } else {
                nc_of(pic, Place { mx, my }, Cell { bx, by }, mb, plane)
            };
            let decoded = cavlc::residual(bits.residual(mb.intra)?, nc, max)?;
            for (index, value) in decoded.iter().enumerate() {
                out[index] = *value;
            }
            out
        }
        Entropy::Cabac(cab) => {
            let coded = block_cat(cat, plane, pic.chroma() == 3);
            // Luma DC (cat 0) has a coded_block_flag. Its neighbours are Intra_16x16 DC blocks.
            let inc = if cat == 0 {
                cbf_dc_plane(pic, mx, my, mb, plane)
            } else if cat == 3 {
                cbf_dc_chroma(pic, mx, my, mb, usize::from(plane.saturating_sub(1)))
            } else {
                cbf_inc(pic, Place { mx, my }, Cell { bx, by }, mb, plane, dc)
            };
            let Some(base) = CBF_BASE.get(coded).copied() else {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "cabac residual",
                });
            };
            if cab.decision(base + inc)? == 0 {
                return Ok(out);
            }
            if cat == 0 {
                if plane == 0 {
                    mb.dc_coded = true;
                } else if let Some(flag) = mb.dc16.get_mut(usize::from(plane - 1)) {
                    *flag = true;
                }
            } else if cat == 3 {
                let which = usize::from(plane.saturating_sub(1));
                if let Some(flag) = mb.dc_c.get_mut(which) {
                    *flag = true;
                }
            }
            cabac_residual(cab, coded, &mut out[..max], mb.field_mb || pic.field)?;
            out
        }
    };
    let count = coeffs.iter().take(max).filter(|value| **value != 0).count() as u8;
    if !dc {
        if plane == 0 {
            mb.nnz_y[by * 4 + bx] = count;
        } else if let Some(slot) = mb.nnz_c.get_mut(chroma_slot(pic.chroma(), plane, bx, by)) {
            *slot = count;
        }
    }
    Ok(coeffs)
}

fn read8<S: Sample>(
    pic: &Frame<S>,
    entropy: &mut Entropy<'_, '_>,
    mx: usize,
    my: usize,
    which: usize,
    mb: &mut Mb,
    plane: u8,
) -> Result<[i32; 64], CodecError> {
    let mut out = [0i32; 64];
    let bx = (which & 1) * 2;
    let by = (which >> 1) * 2;
    match entropy {
        Entropy::Cavlc(bits) => {
            for i4 in 0..4 {
                let x = bx + (i4 & 1);
                let y = by + (i4 >> 1);
                let nc = nc_of(pic, Place { mx, my }, Cell { bx: x, by: y }, mb, plane);
                let block = cavlc::residual(bits.residual(mb.intra)?, nc, 16)?;
                let count = block.iter().filter(|value| **value != 0).count() as u8;
                store_nnz(pic, mb, plane, x, y, count);
                for (index, value) in block.iter().enumerate() {
                    out[index * 4 + i4] = *value;
                }
            }
        }
        Entropy::Cabac(cab) => {
            let cat = block_cat(5, plane, pic.chroma() == 3 && !pic.separate_plane);
            // 4:2:0 and 4:2:2 omit the luma 8×8 coded-block flag; the CBP bit
            // is that flag. 4:4:4 reads it for luma and for both chroma planes.
            // Separate colour planes are monochrome, so that flag stays absent.
            if plane != 0 || (pic.chroma() == 3 && !pic.separate_plane) {
                let inc = cbf_inc(pic, Place { mx, my }, Cell { bx, by }, mb, plane, false);
                let Some(base) = CBF_BASE.get(cat).copied() else {
                    return Err(CodecError::Malformed {
                        at_bit: 0,
                        what: "cabac residual",
                    });
                };
                if cab.decision(base + inc)? == 0 {
                    return Ok(out);
                }
            }
            cabac_residual(cab, cat, &mut out, mb.field_mb || pic.field)?;
            let count = out.iter().filter(|value| **value != 0).count() as u8;
            for dy in 0..2 {
                for dx in 0..2 {
                    store_nnz(pic, mb, plane, bx + dx, by + dy, count);
                }
            }
        }
    }
    Ok(out)
}

fn store_nnz<S: Sample>(pic: &Frame<S>, mb: &mut Mb, plane: u8, x: usize, y: usize, count: u8) {
    if plane == 0 {
        mb.nnz_y[y * 4 + x] = count;
        return;
    }
    if let Some(slot) = mb.nnz_c.get_mut(chroma_slot(pic.chroma(), plane, x, y)) {
        *slot = count;
    }
}

fn read_chroma<S: Sample>(
    pic: &Frame<S>,
    entropy: &mut Entropy<'_, '_>,
    mx: usize,
    my: usize,
    cbp: u8,
    mb: &mut Mb,
) -> Result<ChromaResidual, CodecError> {
    let mut residual = ChromaResidual {
        cb_dc: [0; 8],
        cr_dc: [0; 8],
        cb_ac: [[0; 16]; 8],
        cr_ac: [[0; 16]; 8],
    };
    if cbp == 0 {
        return Ok(residual);
    }
    let blocks = if pic.chroma() == 2 { 8 } else { 4 };
    let dc0 = read_block(
        pic,
        entropy,
        (3, blocks, true),
        Place { mx, my },
        Cell { bx: 0, by: 0 },
        mb,
        1,
    )?;
    residual.cb_dc[..blocks].copy_from_slice(&dc0[..blocks]);
    let dc1 = read_block(
        pic,
        entropy,
        (3, blocks, true),
        Place { mx, my },
        Cell { bx: 0, by: 0 },
        mb,
        2,
    )?;
    residual.cr_dc[..blocks].copy_from_slice(&dc1[..blocks]);
    if cbp == 2 {
        for (plane, ac) in [
            (1u8, residual.cb_ac.as_mut_slice()),
            (2, residual.cr_ac.as_mut_slice()),
        ] {
            for (i, slot) in ac.iter_mut().take(blocks).enumerate() {
                let (bx, by) = chroma_pos(i, blocks);
                *slot = read_block(
                    pic,
                    entropy,
                    (4, 15, false),
                    Place { mx, my },
                    Cell { bx, by },
                    mb,
                    plane,
                )?;
            }
        }
    }
    Ok(residual)
}

fn nc_of<S: Sample>(pic: &Frame<S>, at: Place, cell: Cell, mb: &Mb, plane: u8) -> i32 {
    let Place { mx, my } = at;
    let Cell { bx, by } = cell;
    let left = nnz_at(pic, Place { mx, my }, (bx as i32 - 1, by as i32), mb, plane);
    let above = nnz_at(pic, Place { mx, my }, (bx as i32, by as i32 - 1), mb, plane);
    match (left, above) {
        (Some(a), Some(b)) => (i32::from(a) + i32::from(b) + 1) >> 1,
        (Some(a), None) => i32::from(a),
        (None, Some(b)) => i32::from(b),
        (None, None) => 0,
    }
}

fn nnz_at<S: Sample>(
    pic: &Frame<S>,
    at: Place,
    block: (i32, i32),
    current: &Mb,
    plane: u8,
) -> Option<u8> {
    let Place { mx, my } = at;
    let (bx, by) = block;
    let (gx, gy) = if plane == 0 {
        (4, 4)
    } else {
        match pic.chroma() {
            2 => (2, 4),
            3 => (4, 4),
            _ => (2, 2),
        }
    };
    let (mbx, ox) = shift(mx, bx, gx);
    let (mby, oy) = shift_field(my, by, gy, current.field_mb);
    let index = if plane == 0 {
        oy * 4 + ox
    } else {
        chroma_slot(pic.chroma(), plane, ox, oy)
    };
    if mbx == mx as i32 && mby == my as i32 {
        return Some(if plane == 0 {
            current.nnz_y[index]
        } else {
            current.nnz_c[index]
        });
    }
    let mb = neighbor(pic, mbx, mby, current.slice)?;
    if mb.pcm {
        return Some(16);
    }
    Some(if plane == 0 {
        mb.nnz_y[index]
    } else {
        mb.nnz_c[index]
    })
}

fn cbf_dc_plane<S: Sample>(pic: &Frame<S>, mx: usize, my: usize, mb: &Mb, plane: u8) -> usize {
    let flag = |dx: i32, dy: i32| -> usize {
        let Some(side) = neighbor(pic, mx as i32 + dx, my as i32 + dy, mb.slice) else {
            return usize::from(mb.intra);
        };
        if side.pcm {
            return 1;
        }
        if !side.intra16 {
            return 0;
        }
        if plane == 0 {
            usize::from(side.dc_coded)
        } else {
            side.dc16
                .get(usize::from(plane - 1))
                .copied()
                .map(usize::from)
                .unwrap_or(0)
        }
    };
    let up = if mb.field_mb { -2 } else { -1 };
    flag(-1, 0) + 2 * flag(0, up)
}

fn shift(mb: usize, block: i32, grid: usize) -> (i32, usize) {
    let last = grid - 1;
    if block < 0 {
        (mb as i32 - 1, last)
    } else if block >= grid as i32 {
        (mb as i32 + 1, 0)
    } else {
        (mb as i32, block as usize)
    }
}

fn shift_field(mb: usize, block: i32, grid: usize, field: bool) -> (i32, usize) {
    let (row, off) = shift(mb, block, grid);
    if field && block < 0 {
        (row - 1, off)
    } else {
        (row, off)
    }
}

/// Clause 9.3.3.1.1.9 for ctxBlockCat 3. The neighbour contributes its chroma DC flag.
fn cbf_dc_chroma<S: Sample>(pic: &Frame<S>, mx: usize, my: usize, mb: &Mb, which: usize) -> usize {
    let flag = |dx: i32, dy: i32| -> usize {
        let Some(side) = neighbor(pic, mx as i32 + dx, my as i32 + dy, mb.slice) else {
            return usize::from(mb.intra);
        };
        if side.pcm {
            return 1;
        }
        if side.cbp >> 4 == 0 {
            return 0;
        }
        side.dc_c.get(which).copied().map(usize::from).unwrap_or(0)
    };
    let up = if mb.field_mb { -2 } else { -1 };
    flag(-1, 0) + 2 * flag(0, up)
}

fn cbf_inc<S: Sample>(
    pic: &Frame<S>,
    at: Place,
    cell: Cell,
    mb: &Mb,
    plane: u8,
    dc: bool,
) -> usize {
    let Place { mx, my } = at;
    let Cell { bx, by } = cell;
    let left = nnz_at(pic, Place { mx, my }, (bx as i32 - 1, by as i32), mb, plane)
        .or(dc.then_some(u8::from(mb.intra)));
    let above = nnz_at(pic, Place { mx, my }, (bx as i32, by as i32 - 1), mb, plane)
        .or(dc.then_some(u8::from(mb.intra)));
    let left = left.unwrap_or(u8::from(mb.intra));
    let above = above.unwrap_or(u8::from(mb.intra));
    usize::from(left > 0) + 2 * usize::from(above > 0)
}

fn cabac_residual(
    cab: &mut Cabac<'_>,
    cat: usize,
    coeffs: &mut [i32],
    field: bool,
) -> Result<(), CodecError> {
    // Table 9-34. A field macroblock uses the second context block.
    const SIG: [[usize; 14]; 2] = [
        [
            105, 120, 134, 149, 152, 402, 484, 499, 513, 660, 528, 543, 557, 718,
        ],
        [
            277, 292, 306, 321, 324, 436, 776, 791, 805, 675, 820, 835, 849, 733,
        ],
    ];
    const LAST: [[usize; 14]; 2] = [
        [
            166, 181, 195, 210, 213, 417, 572, 587, 601, 690, 616, 631, 645, 748,
        ],
        [
            338, 353, 367, 382, 385, 451, 864, 879, 893, 699, 908, 923, 937, 757,
        ],
    ];
    const LEVEL: [usize; 14] = [
        227, 237, 247, 257, 266, 426, 952, 962, 972, 708, 982, 992, 1002, 766,
    ];
    let which = cat.min(13);
    let row = usize::from(field);
    let (sig_base, last_base, level_base) = (SIG[row][which], LAST[row][which], LEVEL[which]);
    let max = coeffs.len();
    let dc422 = cat == 3 && max == 8;
    let mut index = [0usize; 64];
    let mut count = 0usize;
    let last_coded = max.saturating_sub(1);
    let mut pos = 0;
    while pos < last_coded {
        let sig_ctx = sig_base
            + if matches!(cat, 5 | 9 | 13) {
                if field {
                    cabac::sig8_field(pos)
                } else {
                    cabac::sig8(pos)
                }
            } else if dc422 {
                const MAP: [usize; 7] = [0, 0, 1, 1, 2, 2, 2];
                MAP[pos.min(6)]
            } else if cat == 3 {
                pos.min(2)
            } else {
                pos
            };
        if cab.decision(sig_ctx)? == 1 {
            let Some(slot) = index.get_mut(count) else {
                return Err(CodecError::Limit {
                    what: "cabac residual",
                });
            };
            *slot = pos;
            count += 1;
            let last_ctx = last_base
                + if matches!(cat, 5 | 9 | 13) {
                    cabac::last8(pos)
                } else if dc422 {
                    const MAP: [usize; 7] = [0, 0, 1, 1, 2, 2, 2];
                    MAP[pos.min(6)]
                } else if cat == 3 {
                    pos.min(2)
                } else {
                    pos
                };
            if cab.decision(last_ctx)? == 1 {
                pos = max;
                break;
            }
        }
        pos += 1;
    }
    if pos == last_coded {
        let Some(slot) = index.get_mut(count) else {
            return Err(CodecError::Limit {
                what: "cabac residual",
            });
        };
        *slot = pos;
        count += 1;
    }
    if count == 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "cabac residual",
        });
    }
    let mut node = 0usize;
    let level1 = [1usize, 2, 3, 4, 0, 0, 0, 0];
    let mut gt1 = [5usize, 5, 5, 5, 6, 7, 8, 9];
    if dc422 {
        gt1[7] = 8;
    }
    let trans1 = [1usize, 2, 3, 3, 4, 5, 6, 7];
    let trans_gt = [4usize, 4, 4, 4, 5, 6, 7, 7];
    for scan in index[..count].iter().rev() {
        let scan = *scan;
        if cab.decision(level_base + level1[node])? == 0 {
            node = trans1[node];
            let sign = cab.bypass()?;
            coeffs[scan] = if sign == 1 { -1 } else { 1 };
        } else {
            let ctx = level_base + gt1[node];
            node = trans_gt[node];
            let mut abs = 2i32;
            while abs < 15 && cab.decision(ctx)? == 1 {
                abs += 1;
            }
            if abs >= 15 {
                let mut zeros = 0i32;
                while cab.bypass()? == 1 && zeros < 20 {
                    zeros += 1;
                }
                let mut mag = 1i32;
                for _ in 0..zeros {
                    mag = (mag << 1) | i32::from(cab.bypass()?);
                }
                abs = mag + 14;
            }
            let sign = cab.bypass()?;
            coeffs[scan] = if sign == 1 { -abs } else { abs };
        }
    }
    Ok(())
}

/// Clause 8.5. `qp` is SliceQPY. Bypass applies when the flag is set and QP′Y is 0.
fn qp_bypass<S: Sample>(slice: &Slice<'_>, qp: i32) -> bool {
    slice.sps.bypass && qp + S::QP_BD == 0
}

fn place4(coeffs: &[i32]) -> [i32; 16] {
    let mut block = [0i32; 16];
    for (index, &pos) in ZIGZAG4.iter().enumerate() {
        block[pos] = coeffs.get(index).copied().unwrap_or(0);
    }
    block
}

fn place8(coeffs: &[i32]) -> [i32; 64] {
    let mut block = [0i32; 64];
    for (index, &pos) in ZIGZAG8.iter().enumerate() {
        block[pos] = coeffs.get(index).copied().unwrap_or(0);
    }
    block
}

/// Clause 8.5.15. `horizontal` is horPredFlag.
fn prefix_residual(block: &mut [i32], width: usize, height: usize, horizontal: bool) {
    if horizontal {
        for y in 0..height {
            let mut sum = 0i32;
            for x in 0..width {
                let at = y * width + x;
                sum = sum.wrapping_add(block[at]);
                block[at] = sum;
            }
        }
    } else {
        for x in 0..width {
            let mut sum = 0i32;
            for y in 0..height {
                let at = y * width + x;
                sum = sum.wrapping_add(block[at]);
                block[at] = sum;
            }
        }
    }
}

fn i16_bypass(dc: &[i32; 16], ac: &[[i32; 16]; 16], coded: bool) -> [i32; 256] {
    let raster = luma_dc_raster(dc);
    let mut out = [0i32; 256];
    for (i, samples) in ac.iter().enumerate() {
        let bx = blk_x(i);
        let by = blk_y(i);
        let mut coeffs = [0i32; 16];
        coeffs[0] = raster[by * 4 + bx];
        if coded {
            coeffs[1..].copy_from_slice(&samples[..15]);
        }
        let block = place4(&coeffs);
        for y in 0..4 {
            for x in 0..4 {
                out[(by * 4 + y) * 16 + bx * 4 + x] = block[y * 4 + x];
            }
        }
    }
    out
}

fn add_i16_bypass<S: Sample>(
    plane: &mut [S],
    stride: usize,
    origin: (usize, usize, usize),
    dc: &[i32; 16],
    ac: &[[i32; 16]; 16],
    flags: (bool, u8),
) {
    let (x0, y0, step) = origin;
    let (coded, mode) = flags;
    let mut residual = i16_bypass(dc, ac, coded);
    if mode == 0 || mode == 1 {
        prefix_residual(&mut residual, 16, 16, mode == 1);
    }
    recon::add_block(plane, stride, x0, y0, &residual, 16, step);
}

fn add_chroma_bypass<S: Sample>(
    plane: &mut [S],
    stride: usize,
    origin: (usize, usize, usize),
    levels: (&[i32], &[[i32; 16]]),
    shape: (usize, usize, usize),
    intra_mode: Option<u8>,
) {
    let (dc, ac) = levels;
    let (blocks, width, height) = shape;
    let raster = if blocks == 8 {
        raster422(dc)
    } else {
        [0i32; 8]
    };
    let mut residual = [0i32; 128];
    for i in 0..blocks {
        let (bx, by) = chroma_pos(i, blocks);
        let mut coeffs = [0i32; 16];
        coeffs[0] = if blocks == 8 {
            raster[by * 2 + bx]
        } else {
            dc.get(i).copied().unwrap_or(0)
        };
        if let Some(block) = ac.get(i) {
            coeffs[1..].copy_from_slice(&block[..15]);
        }
        let levels = place4(&coeffs);
        for y in 0..4 {
            for x in 0..4 {
                residual[(by * 4 + y) * width + bx * 4 + x] = levels[y * 4 + x];
            }
        }
    }
    if let Some(mode) = intra_mode.filter(|mode| *mode == 1 || *mode == 2) {
        prefix_residual(&mut residual[..width * height], width, height, mode == 1);
    }
    let (x0, y0, step) = origin;
    for y in 0..height {
        for x in 0..width {
            let at = (y0 + y * step) * stride + x0 + x;
            let Some(sample) = plane.get(at).copied() else {
                continue;
            };
            let sum = sample.into().wrapping_add(residual[y * width + x]);
            if let Some(slot) = plane.get_mut(at) {
                *slot = S::clip(sum);
            }
        }
    }
}

fn recon_intra<S: Sample>(
    pic: &mut Frame<S>,
    slice: &Slice<'_>,
    at: Place,
    mb: &Mb,
    luma: LumaPaint<'_>,
    chroma: ChromaPaint<'_>,
    qp: i32,
) {
    let Place { mx, my } = at;
    let LumaPaint {
        i16,
        mode16,
        chroma_mode,
        dc: luma_dc,
        ac,
        block8,
    } = luma;
    let ChromaPaint {
        cb_dc,
        cr_dc,
        cb_ac,
        cr_ac,
    } = chroma;
    let x0 = mx * 16;
    let span = luma_span(my, mb.field_mb);
    let y0 = span.y;
    let step = span.step;
    // Neighbours inside this macroblock are already reconstructed. Publish it
    // before those samples are read back, so availability sees this slice.
    pic.mb[my * pic.mb_w + mx] = mb.clone();
    if i16 {
        let (above, left, above_ok, left_ok) =
            edge16(pic, mx, my, mb.slice, slice.pps.constrained_intra, 0);
        let corner = corner_sample(
            &pic.y,
            pic.w,
            pic.h,
            mx as i32 * 16 - 1,
            span.y as i32 - span.step as i32,
            above_ok && left_ok,
        );
        let pred = recon::pred16(mode16, &above, &left, above_ok, left_ok, corner);
        put_block(&mut pic.y, pic.w, x0, y0, &pred, 16, step);
        if qp_bypass::<S>(slice, qp) {
            add_i16_bypass(
                &mut pic.y,
                pic.w,
                (x0, y0, step),
                &luma_dc,
                ac,
                (mb.cbp & 15 != 0, mode16),
            );
        } else {
            let dc = recon::luma_dc(
                &luma_dc_raster(&luma_dc),
                qp + S::QP_BD,
                i32::from(slice.pps.weight4[0][0]),
            );
            for (i, samples) in ac.iter().enumerate() {
                let bx = blk_x(i);
                let by = blk_y(i);
                let mut coeffs = [0i32; 16];
                if mb.cbp & 15 != 0 {
                    coeffs[1..].copy_from_slice(&samples[..15]);
                }
                let mut block = [0i32; 16];
                for n in 0..16 {
                    let pos = ZIGZAG4[n];
                    let (ry, rx) = (pos / 4, pos % 4);
                    let level = if n == 0 { dc[by * 4 + bx] } else { coeffs[n] };
                    let weight = i32::from(slice.pps.weight4[0][pos]);
                    block[ry * 4 + rx] = if n == 0 {
                        level
                    } else {
                        recon::scale4(level, qp + S::QP_BD, rx, ry, weight)
                    };
                }
                if mb.cbp & 15 == 0 {
                    block[0] = dc[by * 4 + bx];
                }
                recon::idct4(&mut block);
                recon::add_block(
                    &mut pic.y,
                    pic.w,
                    x0 + bx * 4,
                    span.at(by * 4),
                    &block,
                    4,
                    step,
                );
            }
        }
    } else if mb.transform8 {
        for (i, samples) in block8.iter().enumerate() {
            let bx = (i & 1) * 8;
            let by = (i >> 1) * 8;
            let pred = pred8_block(
                pic,
                Place { mx, my },
                Cell {
                    bx: bx / 4,
                    by: by / 4,
                },
                mb,
                slice.pps.constrained_intra,
                0,
            );
            put_block(&mut pic.y, pic.w, x0 + bx, span.at(by), &pred, 8, step);
            let mode = mb.mode4[(by / 4) * 4 + bx / 4];
            let bypass = qp_bypass::<S>(slice, qp);
            if bypass && (mode == 0 || mode == 1) {
                let mut block = place8(samples);
                prefix_residual(&mut block, 8, 8, mode == 1);
                recon::add_block(&mut pic.y, pic.w, x0 + bx, span.at(by), &block, 8, step);
            } else {
                super::add_scaled(
                    &mut pic.y,
                    pic.w,
                    (x0 + bx, span.at(by), step),
                    samples,
                    (qp, bypass),
                    &slice.pps.weight8[0],
                    8,
                );
            }
        }
    } else {
        for (i, samples) in ac.iter().enumerate() {
            let bx = blk_x(i);
            let by = blk_y(i);
            let pred = pred4_block(
                pic,
                Place { mx, my },
                Cell { bx, by },
                mb,
                slice.pps.constrained_intra,
                0,
            );
            put_block(
                &mut pic.y,
                pic.w,
                x0 + bx * 4,
                span.at(by * 4),
                &pred,
                4,
                step,
            );
            let mode = mb.mode4[by * 4 + bx];
            let bypass = qp_bypass::<S>(slice, qp);
            if bypass && (mode == 0 || mode == 1) {
                let mut block = place4(samples);
                prefix_residual(&mut block, 4, 4, mode == 1);
                recon::add_block(
                    &mut pic.y,
                    pic.w,
                    x0 + bx * 4,
                    span.at(by * 4),
                    &block,
                    4,
                    step,
                );
            } else {
                super::add_scaled(
                    &mut pic.y,
                    pic.w,
                    (x0 + bx * 4, span.at(by * 4), step),
                    samples,
                    (qp, bypass),
                    &slice.pps.weight4[0],
                    4,
                );
            }
        }
    }
    if pic.chroma() != 3 {
        recon_chroma(
            pic,
            slice,
            Place { mx, my },
            chroma_mode,
            ChromaPaint {
                cb_dc,
                cr_dc,
                cb_ac,
                cr_ac,
            },
            (qp, true, mb.slice),
            true,
        );
    }
}

fn luma_dc_raster(scan: &[i32; 16]) -> [i32; 16] {
    let mut out = [0i32; 16];
    for i in 0..16 {
        out[ZIGZAG4[i]] = scan[i];
    }
    out
}

fn plane_samples<S: Sample>(pic: &Frame<S>, plane: u8) -> (&[S], usize, usize) {
    match plane {
        1 => (&pic.u, pic.chroma_stride(), pic.chroma_height()),
        2 => (&pic.v, pic.chroma_stride(), pic.chroma_height()),
        _ => (&pic.y, pic.w, pic.h),
    }
}

fn pred4_block<S: Sample>(
    pic: &Frame<S>,
    at: Place,
    cell: Cell,
    mb: &Mb,
    constrained: bool,
    plane: u8,
) -> [S; 16] {
    let Place { mx, my } = at;
    let Cell { bx, by } = cell;
    let (samples, stride, height) = plane_samples(pic, plane);
    let ax = mx * 4 + bx;
    let ay = my * 4 + by;
    let span = block_span(pic, my, mb.field_mb, plane);
    let row0 = span.at(by * 4) as i32;
    let down = span.step as i32;
    let mut above = [S::mid(); 8];
    let mut left = [S::mid(); 4];
    let here = (mx, my, bx, by);
    let above_ok = row_ok(pic, ax, ay as i32 - 1, mb.slice, constrained, here);
    let left_ok = col_ok(pic, ax as i32 - 1, ay, mb.slice, constrained, here);
    if above_ok {
        for (k, slot) in above[..4].iter_mut().enumerate() {
            *slot = pix(samples, stride, height, (ax * 4 + k) as i32, row0 - down);
        }
        let right_ok = above_right_ok(pic, mx, my, bx, by, mb.slice, constrained);
        let fill = above[3];
        for (k, slot) in above[4..].iter_mut().enumerate() {
            let x = ax * 4 + k + 4;
            *slot = if right_ok {
                pix(samples, stride, height, x as i32, row0 - down)
            } else {
                fill
            };
        }
    }
    if left_ok {
        for (k, slot) in left.iter_mut().enumerate() {
            *slot = pix(
                samples,
                stride,
                height,
                ax as i32 * 4 - 1,
                row0 + k as i32 * down,
            );
        }
    }
    let corner = if above_ok && left_ok {
        pix(samples, stride, height, ax as i32 * 4 - 1, row0 - down)
    } else if above_ok {
        above[0]
    } else if left_ok {
        left[0]
    } else {
        S::mid()
    };
    recon::pred4(
        mb.mode4[by * 4 + bx],
        &above,
        &left,
        corner,
        above_ok,
        left_ok,
    )
}

fn pred8_block<S: Sample>(
    pic: &Frame<S>,
    at: Place,
    cell: Cell,
    mb: &Mb,
    constrained: bool,
    plane: u8,
) -> [S; 64] {
    let Place { mx, my } = at;
    let Cell { bx, by } = cell;
    let (samples, stride, height) = plane_samples(pic, plane);
    let mode = mb.mode4[by * 4 + bx];
    let mut raw_above = [S::mid(); 16];
    let mut raw_left = [S::mid(); 8];
    let span = block_span(pic, my, mb.field_mb, plane);
    let x = (mx * 16 + bx * 4) as i32;
    let y = span.at(by * 4) as i32;
    let down = span.step as i32;
    let here = (mx, my, bx, by);
    let above_ok = row_ok(
        pic,
        mx * 4 + bx,
        (my * 4 + by) as i32 - 1,
        mb.slice,
        constrained,
        here,
    );
    let left_ok = col_ok(
        pic,
        (mx * 4 + bx) as i32 - 1,
        my * 4 + by,
        mb.slice,
        constrained,
        here,
    );
    if above_ok {
        for (k, slot) in raw_above[..8].iter_mut().enumerate() {
            *slot = pix(samples, stride, height, x + k as i32, y - down);
        }
        // Clause 8.3.2.2: p[8..15, -1] that are not yet constructed take p[7, -1].
        let fill = raw_above[7];
        let right = above8_right(pic, x, y, down, mb.slice, constrained);
        for (k, slot) in raw_above[8..].iter_mut().enumerate() {
            *slot = if right {
                pix(samples, stride, height, x + 8 + k as i32, y - down)
            } else {
                fill
            };
        }
        // Clause 8.3.2.2: when p[8..15, -1] are not available, they take p[7, -1]
        // before the clause 8.3.2.2.1 filter. That block is the one to the upper right.
        let upper_right = |dx: usize| {
            row_ok(
                pic,
                mx * 4 + bx + 2 + dx,
                (my * 4 + by) as i32 - 1,
                mb.slice,
                constrained,
                here,
            )
        };
        if upper_right(0) && upper_right(1) {
            for (k, slot) in raw_above[8..].iter_mut().enumerate() {
                *slot = pix(&pic.y, pic.w, pic.h, x + 8 + k as i32, y - 1);
            }
        } else {
            let edge = raw_above[7];
            raw_above[8..].fill(edge);
        }
    }
    if left_ok {
        for (k, slot) in raw_left.iter_mut().enumerate() {
            *slot = pix(samples, stride, height, x - 1, y + k as i32 * down);
        }
    }
    let corner = if above_ok && left_ok {
        pix(samples, stride, height, x - 1, y - down)
    } else {
        S::mid()
    };
    let above = filter_row(&raw_above, corner, above_ok, left_ok);
    let left = filter_col(&raw_left, corner, above_ok, left_ok);
    let corner = if above_ok && left_ok {
        avg3(raw_above[0], corner, raw_left[0])
    } else {
        corner
    };
    let mut out = [S::default(); 64];
    for row in 0..8 {
        for col in 0..8 {
            out[row * 8 + col] =
                pred8_at(mode, (col, row), &above, &left, corner, (above_ok, left_ok));
        }
    }
    out
}

fn pred8_at<S: Sample>(
    mode: u8,
    at: (usize, usize),
    above: &[S; 16],
    left: &[S; 8],
    corner: S,
    present: (bool, bool),
) -> S {
    let (x, y) = at;
    let (above_ok, left_ok) = present;
    match mode {
        0 if above_ok => above[x],
        1 if left_ok => left[y],
        3 if above_ok => pred8_down_left(x, y, above),
        4 if above_ok && left_ok => pred8_down_right(x, y, above, left, corner),
        5 if above_ok && left_ok => pred8_vertical_right(x, y, above, left, corner),
        6 if above_ok && left_ok => pred8_horizontal_down(x, y, above, left, corner),
        7 if above_ok => pred8_vertical_left(x, y, above),
        8 if left_ok => pred8_horizontal_up(x, y, left),
        _ => dc8(above, left, above_ok, left_ok),
    }
}

fn pred8_down_left<S: Sample>(x: usize, y: usize, above: &[S; 16]) -> S {
    if x == 7 && y == 7 {
        avg3(above[14], above[15], above[15])
    } else {
        avg3(above[x + y], above[x + y + 1], above[x + y + 2])
    }
}

fn pred8_down_right<S: Sample>(x: usize, y: usize, above: &[S; 16], left: &[S; 8], corner: S) -> S {
    let x = i32::try_from(x).unwrap_or(0);
    let y = i32::try_from(y).unwrap_or(0);
    if x > y {
        avg3(
            above_at(above, corner, x - y - 2),
            above_at(above, corner, x - y - 1),
            above_at(above, corner, x - y),
        )
    } else if x < y {
        avg3(
            left_at(left, corner, y - x - 2),
            left_at(left, corner, y - x - 1),
            left_at(left, corner, y - x),
        )
    } else {
        avg3(above[0], corner, left[0])
    }
}

fn pred8_vertical_right<S: Sample>(
    x: usize,
    y: usize,
    above: &[S; 16],
    left: &[S; 8],
    corner: S,
) -> S {
    let x = i32::try_from(x).unwrap_or(0);
    let y = i32::try_from(y).unwrap_or(0);
    let z = 2 * x - y;
    let shift = y >> 1;
    if z >= 0 && (z & 1) == 0 {
        let index = x - shift;
        avg2(
            above_at(above, corner, index - 1),
            above_at(above, corner, index),
        )
    } else if z > 0 {
        let index = x - shift;
        avg3(
            above_at(above, corner, index - 2),
            above_at(above, corner, index - 1),
            above_at(above, corner, index),
        )
    } else if z == -1 {
        avg3(left[0], corner, above[0])
    } else {
        let index = y - 2 * x;
        avg3(
            left_at(left, corner, index - 1),
            left_at(left, corner, index - 2),
            left_at(left, corner, index - 3),
        )
    }
}

fn pred8_horizontal_down<S: Sample>(
    x: usize,
    y: usize,
    above: &[S; 16],
    left: &[S; 8],
    corner: S,
) -> S {
    let x = i32::try_from(x).unwrap_or(0);
    let y = i32::try_from(y).unwrap_or(0);
    let z = 2 * y - x;
    let shift = x >> 1;
    if z >= 0 && (z & 1) == 0 {
        let index = y - shift;
        avg2(
            left_at(left, corner, index - 1),
            left_at(left, corner, index),
        )
    } else if z > 0 {
        let index = y - shift;
        avg3(
            left_at(left, corner, index - 2),
            left_at(left, corner, index - 1),
            left_at(left, corner, index),
        )
    } else if z == -1 {
        avg3(left[0], corner, above[0])
    } else {
        let index = x - 2 * y;
        avg3(
            above_at(above, corner, index - 1),
            above_at(above, corner, index - 2),
            above_at(above, corner, index - 3),
        )
    }
}

fn pred8_vertical_left<S: Sample>(x: usize, y: usize, above: &[S; 16]) -> S {
    let base = x + (y >> 1);
    if y.is_multiple_of(2) {
        avg2(above[base], above[base + 1])
    } else {
        avg3(above[base], above[base + 1], above[base + 2])
    }
}

fn pred8_horizontal_up<S: Sample>(x: usize, y: usize, left: &[S; 8]) -> S {
    let z = x + 2 * y;
    let base = y + (x >> 1);
    if matches!(z, 0 | 2 | 4 | 6 | 8 | 10 | 12) {
        avg2(left[base], left[base + 1])
    } else if matches!(z, 1 | 3 | 5 | 7 | 9 | 11) {
        avg3(left[base], left[base + 1], left[base + 2])
    } else if z == 13 {
        avg3(left[6], left[7], left[7])
    } else {
        left[7]
    }
}

fn above_at<S: Sample>(above: &[S; 16], corner: S, index: i32) -> S {
    let Ok(index) = usize::try_from(index) else {
        return corner;
    };
    above.get(index).copied().unwrap_or(corner)
}

fn left_at<S: Sample>(left: &[S; 8], corner: S, index: i32) -> S {
    let Ok(index) = usize::try_from(index) else {
        return corner;
    };
    left.get(index).copied().unwrap_or(corner)
}

fn avg2<S: Sample>(a: S, b: S) -> S {
    let (a, b): (i32, i32) = (a.into(), b.into());
    S::clip((a + b + 1) >> 1)
}

/// The eight samples to the right of an Intra_8x8 block's above row.
///
/// They belong to one macroblock. A later macroblock in this slice is not
/// constructed yet, so it is not a neighbour.
fn above8_right<S: Sample>(
    pic: &Frame<S>,
    x: i32,
    y: i32,
    step: i32,
    slice: u16,
    constrained: bool,
) -> bool {
    let sx = x + 8;
    let sy = y - step;
    if sx < 0 || sy < 0 {
        return false;
    }
    let Some(side) = neighbor(pic, sx / 16, sy / 16, slice) else {
        return false;
    };
    !constrained || side.intra
}

fn filter_row<S: Sample>(raw: &[S; 16], corner: S, above_ok: bool, left_ok: bool) -> [S; 16] {
    let mut out = *raw;
    if !above_ok {
        return out;
    }
    let left = if left_ok { corner } else { raw[0] };
    out[0] = avg3(left, raw[0], raw[1]);
    for i in 1..15 {
        out[i] = avg3(raw[i - 1], raw[i], raw[i + 1]);
    }
    out[15] = avg3(raw[14], raw[15], raw[15]);
    out
}

fn filter_col<S: Sample>(raw: &[S; 8], corner: S, above_ok: bool, left_ok: bool) -> [S; 8] {
    let mut out = *raw;
    if !left_ok {
        return out;
    }
    let up = if above_ok { corner } else { raw[0] };
    out[0] = avg3(up, raw[0], raw[1]);
    for i in 1..7 {
        out[i] = avg3(raw[i - 1], raw[i], raw[i + 1]);
    }
    out[7] = avg3(raw[6], raw[7], raw[7]);
    out
}

fn avg3<S: Sample>(a: S, b: S, c: S) -> S {
    let (a, b, c): (i32, i32, i32) = (a.into(), b.into(), c.into());
    S::clip((a + 2 * b + c + 2) >> 2)
}

fn dc8<S: Sample>(above: &[S], left: &[S], above_ok: bool, left_ok: bool) -> S {
    let sample = |v: &S| -> i32 { (*v).into() };
    if above_ok && left_ok {
        let sum =
            above[..8].iter().map(sample).sum::<i32>() + left[..8].iter().map(sample).sum::<i32>();
        S::clip((sum + 8) >> 4)
    } else if above_ok {
        let sum = above[..8].iter().map(sample).sum::<i32>();
        S::clip((sum + 4) >> 3)
    } else if left_ok {
        let sum = left[..8].iter().map(sample).sum::<i32>();
        S::clip((sum + 4) >> 3)
    } else {
        S::mid()
    }
}

fn recon_chroma<S: Sample>(
    pic: &mut Frame<S>,
    slice: &Slice<'_>,
    at: Place,
    mode: u8,
    paint: ChromaPaint<'_>,
    flags: (i32, bool, u16),
    spatial: bool,
) {
    let Place { mx, my } = at;
    if pic.chroma() == 2 {
        recon_chroma422(pic, slice, at, mode, paint, flags, spatial);
        return;
    }
    let ChromaPaint {
        cb_dc,
        cr_dc,
        cb_ac,
        cr_ac,
    } = paint;
    let (qp, intra, slice_id) = flags;
    let x0 = mx * 8;
    let field = pic.mb.get(my * pic.mb_w + mx).is_some_and(|mb| mb.field_mb);
    let span = chroma_span(my, 8, field);
    let y0 = span.y;
    let step = span.step;
    let (qp_cb, _) = recon::chroma_qps(qp, slice.pps.chroma_qp_offset, S::QP_BD);
    let (qp_cr, _) = recon::chroma_qps(qp, slice.pps.second_chroma_qp_offset, S::QP_BD);
    // Intra chroma uses lists 1 and 2. Inter chroma uses lists 4 and 5.
    let base = if intra { 1 } else { 4 };
    for (cb, dc, ac, qp_c, list) in [
        (true, cb_dc, cb_ac, qp_cb, base),
        (false, cr_dc, cr_ac, qp_cr, base + 1),
    ] {
        let pred = if intra && spatial {
            let (above, left, above_ok, left_ok) =
                edge_chroma(pic, mx, my, cb, slice_id, slice.pps.constrained_intra, 8);
            let plane = if cb { &pic.u } else { &pic.v };
            let corner = corner_sample(
                plane,
                pic.w / 2,
                pic.h / 2,
                mx as i32 * 8 - 1,
                span.y as i32 - span.step as i32,
                above_ok && left_ok,
            );
            Some(recon::pred_chroma(
                mode, &above, &left, above_ok, left_ok, corner,
            ))
        } else {
            None
        };
        let plane = if cb { &mut pic.u } else { &mut pic.v };
        let stride = pic.w / 2;
        if let Some(pred) = pred {
            put_block(plane, stride, x0, y0, &pred, 8, step);
        }
        if qp_bypass::<S>(slice, qp) {
            add_chroma_bypass(
                plane,
                stride,
                (x0, y0, step),
                (dc, ac),
                (4, 8, 8),
                intra.then_some(mode),
            );
            continue;
        }
        // No coefficients in this plane: every block below would add zero.
        if dc.iter().take(4).all(|&level| level == 0)
            && ac.iter().take(4).all(|block| block[..15] == [0; 15])
        {
            continue;
        }
        let weight = i32::from(slice.pps.weight4[list][0]);
        let scaled = recon::chroma_dc(dc[0], dc[1], dc[2], dc[3], qp_c, weight);
        for i in 0..4 {
            let bx = i & 1;
            let by = i >> 1;
            let mut coeffs = [0i32; 16];
            coeffs[1..].copy_from_slice(&ac[i][..15]);
            let mut block = [0i32; 16];
            block[0] = scaled[i];
            for n in 1..16 {
                // A zero level scales to zero.
                if coeffs[n] == 0 {
                    continue;
                }
                let pos = ZIGZAG4[n];
                let (ry, rx) = (pos / 4, pos % 4);
                let w = i32::from(slice.pps.weight4[list][pos]);
                block[ry * 4 + rx] = recon::scale4(coeffs[n], qp_c, rx, ry, w);
            }
            recon::add_idct4(
                plane,
                stride,
                x0 + bx * 4,
                span.at(by * 4),
                &mut block,
                step,
            );
        }
    }
}

fn recon_chroma422<S: Sample>(
    pic: &mut Frame<S>,
    slice: &Slice<'_>,
    at: Place,
    mode: u8,
    paint: ChromaPaint<'_>,
    flags: (i32, bool, u16),
    spatial: bool,
) {
    let Place { mx, my } = at;
    let ChromaPaint {
        cb_dc,
        cr_dc,
        cb_ac,
        cr_ac,
    } = paint;
    let (qp, intra, slice_id) = flags;
    let x0 = mx * 8;
    let field = pic.mb.get(my * pic.mb_w + mx).is_some_and(|mb| mb.field_mb);
    let span = chroma_span(my, 16, field);
    let y0 = span.y;
    let step = span.step;
    let (qp_cb, _) = recon::chroma_qps(qp, slice.pps.chroma_qp_offset, S::QP_BD);
    let (qp_cr, _) = recon::chroma_qps(qp, slice.pps.second_chroma_qp_offset, S::QP_BD);
    let base = if intra { 1 } else { 4 };
    for (cb, dc, ac, qp_c, list) in [
        (true, cb_dc, cb_ac, qp_cb, base),
        (false, cr_dc, cr_ac, qp_cr, base + 1),
    ] {
        if intra && spatial {
            let (above, left, above_ok, left_ok) =
                edge_chroma(pic, mx, my, cb, slice_id, slice.pps.constrained_intra, 16);
            let (samples, stride, height) = if cb {
                (&pic.u[..], pic.chroma_stride(), pic.chroma_height())
            } else {
                (&pic.v[..], pic.chroma_stride(), pic.chroma_height())
            };
            let corner = if above_ok && left_ok {
                pix(
                    samples,
                    stride,
                    height,
                    mx as i32 * 8 - 1,
                    span.y as i32 - span.step as i32,
                )
            } else {
                S::mid()
            };
            let pred = recon::pred_chroma422(mode, &above, &left, above_ok, left_ok, corner);
            let plane = if cb { &mut pic.u } else { &mut pic.v };
            for row in 0..16 {
                for col in 0..8 {
                    plane[(y0 + row * step) * stride + x0 + col] = pred[row * 8 + col];
                }
            }
        }
        let plane = if cb { &mut pic.u } else { &mut pic.v };
        let stride = pic.w / 2;
        if qp_bypass::<S>(slice, qp) {
            add_chroma_bypass(
                plane,
                stride,
                (x0, y0, step),
                (dc, ac),
                (8, 8, 16),
                intra.then_some(mode),
            );
            continue;
        }
        let weight = i32::from(slice.pps.weight4[list][0]);
        // 4:2:2 chroma DC is scaled at QP′C + 3. The eight-point Hadamard
        // needs that extra factor; 4:2:0 stays on the four-point scale.
        let scaled = recon::chroma_dc422(&raster422(dc), qp_c + 3, weight);
        for i in 0..8 {
            let (bx, by) = chroma_pos(i, 8);
            let mut coeffs = [0i32; 16];
            if let Some(block) = ac.get(i) {
                coeffs[1..].copy_from_slice(&block[..15]);
            }
            let mut samples = [0i32; 16];
            for n in 0..16 {
                let pos = ZIGZAG4[n];
                let (ry, rx) = (pos / 4, pos % 4);
                let level = if n == 0 {
                    scaled[by * 2 + bx]
                } else {
                    coeffs[n]
                };
                let w = i32::from(slice.pps.weight4[list][pos]);
                samples[ry * 4 + rx] = if n == 0 {
                    level
                } else {
                    recon::scale4(level, qp_c, rx, ry, w)
                };
            }
            recon::idct4(&mut samples);
            recon::add_block(
                plane,
                stride,
                x0 + bx * 4,
                span.at(by * 4),
                &samples,
                4,
                step,
            );
        }
    }
}

fn raster422(scan: &[i32]) -> [i32; 8] {
    const AT: [usize; 8] = [0, 2, 1, 4, 6, 3, 5, 7];
    let mut raster = [0i32; 8];
    for (index, slot) in AT.iter().enumerate() {
        raster[*slot] = scan.get(index).copied().unwrap_or(0);
    }
    raster
}

fn quant_list(plane: u8, intra: bool, eight: bool) -> usize {
    if eight {
        let base = usize::from(!intra);
        if plane == 0 {
            base
        } else {
            base + 2 * usize::from(plane)
        }
    } else {
        usize::from(!intra) * 3 + usize::from(plane)
    }
}

fn recon_444<S: Sample>(
    pic: &mut Frame<S>,
    entropy: &mut Entropy<'_, '_>,
    slice: &Slice<'_>,
    at: Place,
    mb: &mut Mb,
    qp: i32,
    mode16: Option<u8>,
) -> Result<(), CodecError> {
    let Place { mx, my } = at;
    for plane in [1u8, 2] {
        let offset = if plane == 1 {
            slice.pps.chroma_qp_offset
        } else {
            slice.pps.second_chroma_qp_offset
        };
        let (qp_c, qp_base) = recon::chroma_qps(qp, offset, S::QP_BD);
        paint_444(
            pic,
            entropy,
            slice,
            Place { mx, my },
            mb,
            (plane, qp_c, qp_base, mode16, qp_bypass::<S>(slice, qp)),
        )?;
    }
    Ok(())
}

fn paint_444<S: Sample>(
    pic: &mut Frame<S>,
    entropy: &mut Entropy<'_, '_>,
    slice: &Slice<'_>,
    at: Place,
    mb: &mut Mb,
    job: (u8, i32, i32, Option<u8>, bool),
) -> Result<(), CodecError> {
    let Place { mx, my } = at;
    let (plane, qp_c, qp_base, mode16, bypass) = job;
    let intra = mb.intra;
    let list = quant_list(plane, intra, mb.transform8);
    let x0 = mx * 16;
    let span = luma_span(my, mb.field_mb);
    let y0 = span.y;
    let step = span.step;
    if mb.intra16 {
        let mode = mode16.unwrap_or(2);
        let (above, left, above_ok, left_ok) =
            edge16(pic, mx, my, mb.slice, slice.pps.constrained_intra, plane);
        let (samples, stride, height) = plane_samples(pic, plane);
        let corner = corner_sample(
            samples,
            stride,
            height,
            mx as i32 * 16 - 1,
            span.y as i32 - span.step as i32,
            above_ok && left_ok,
        );
        let pred = recon::pred16(mode, &above, &left, above_ok, left_ok, corner);
        let (dest, dest_stride) = mutable_plane(pic, plane);
        put_block(dest, dest_stride, x0, y0, &pred, 16, step);
        let dc = read_block(
            pic,
            entropy,
            (0, 16, true),
            Place { mx, my },
            Cell { bx: 0, by: 0 },
            mb,
            plane,
        )?;
        let mut ac = [[0i32; 16]; 16];
        if mb.cbp & 15 != 0 {
            for (i, slot) in ac.iter_mut().enumerate() {
                *slot = read_block(
                    pic,
                    entropy,
                    (1, 15, false),
                    Place { mx, my },
                    Cell {
                        bx: blk_x(i),
                        by: blk_y(i),
                    },
                    mb,
                    plane,
                )?;
            }
        }
        if bypass {
            let (dest, dest_stride) = mutable_plane(pic, plane);
            add_i16_bypass(
                dest,
                dest_stride,
                (x0, y0, step),
                &dc,
                &ac,
                (mb.cbp & 15 != 0, mode),
            );
            return Ok(());
        }
        let scaled = recon::luma_dc(
            &luma_dc_raster(&dc),
            qp_c,
            i32::from(slice.pps.weight4[list][0]),
        );
        for (i, samples) in ac.iter().enumerate() {
            let bx = blk_x(i);
            let by = blk_y(i);
            let mut coeffs = [0i32; 16];
            if mb.cbp & 15 != 0 {
                coeffs[1..].copy_from_slice(&samples[..15]);
            }
            let mut block = [0i32; 16];
            for n in 0..16 {
                let pos = ZIGZAG4[n];
                let (ry, rx) = (pos / 4, pos % 4);
                let level = if n == 0 {
                    scaled[by * 4 + bx]
                } else {
                    coeffs[n]
                };
                let weight = i32::from(slice.pps.weight4[list][pos]);
                block[ry * 4 + rx] = if n == 0 {
                    level
                } else {
                    recon::scale4(level, qp_c, rx, ry, weight)
                };
            }
            if mb.cbp & 15 == 0 {
                block[0] = scaled[by * 4 + bx];
            }
            recon::idct4(&mut block);
            let (dest, dest_stride) = mutable_plane(pic, plane);
            recon::add_block(
                dest,
                dest_stride,
                x0 + bx * 4,
                span.at(by * 4),
                &block,
                4,
                step,
            );
        }
        return Ok(());
    }
    if mb.transform8 {
        let mut blocks = [[0i32; 64]; 4];
        for (i, slot) in blocks.iter_mut().enumerate() {
            if mb.cbp & (1 << i) != 0 {
                *slot = read8(pic, entropy, mx, my, i, mb, plane)?;
            }
        }
        for (i, samples) in blocks.iter().enumerate() {
            let bx = (i & 1) * 8;
            let by = (i >> 1) * 8;
            if intra {
                let pred = pred8_block(
                    pic,
                    Place { mx, my },
                    Cell {
                        bx: bx / 4,
                        by: by / 4,
                    },
                    mb,
                    slice.pps.constrained_intra,
                    plane,
                );
                let (dest, dest_stride) = mutable_plane(pic, plane);
                put_block(dest, dest_stride, x0 + bx, span.at(by), &pred, 8, step);
            }
            if mb.cbp & (1 << i) != 0 {
                let mode = mb.mode4[(by / 4) * 4 + bx / 4];
                let (dest, dest_stride) = mutable_plane(pic, plane);
                if bypass && intra && (mode == 0 || mode == 1) {
                    let mut block = place8(samples);
                    prefix_residual(&mut block, 8, 8, mode == 1);
                    recon::add_block(dest, dest_stride, x0 + bx, span.at(by), &block, 8, step);
                } else {
                    super::add_scaled(
                        dest,
                        dest_stride,
                        (x0 + bx, span.at(by), step),
                        samples,
                        (qp_base, bypass),
                        &slice.pps.weight8[list],
                        8,
                    );
                }
            }
        }
        return Ok(());
    }
    let mut ac = [[0i32; 16]; 16];
    for (i, slot) in ac.iter_mut().enumerate() {
        let bit = 1 << ((blk_y(i) / 2) * 2 + blk_x(i) / 2);
        if mb.cbp & bit != 0 {
            *slot = read_block(
                pic,
                entropy,
                (2, 16, false),
                Place { mx, my },
                Cell {
                    bx: blk_x(i),
                    by: blk_y(i),
                },
                mb,
                plane,
            )?;
        }
    }
    for (i, samples) in ac.iter().enumerate() {
        let bx = blk_x(i);
        let by = blk_y(i);
        let bit = 1 << ((by / 2) * 2 + bx / 2);
        if intra {
            let pred = pred4_block(
                pic,
                Place { mx, my },
                Cell { bx, by },
                mb,
                slice.pps.constrained_intra,
                plane,
            );
            let (dest, dest_stride) = mutable_plane(pic, plane);
            put_block(
                dest,
                dest_stride,
                x0 + bx * 4,
                span.at(by * 4),
                &pred,
                4,
                step,
            );
        }
        if mb.cbp & bit != 0 {
            let mode = mb.mode4[by * 4 + bx];
            let (dest, dest_stride) = mutable_plane(pic, plane);
            if bypass && intra && (mode == 0 || mode == 1) {
                let mut block = place4(samples);
                prefix_residual(&mut block, 4, 4, mode == 1);
                recon::add_block(
                    dest,
                    dest_stride,
                    x0 + bx * 4,
                    span.at(by * 4),
                    &block,
                    4,
                    step,
                );
            } else {
                super::add_scaled(
                    dest,
                    dest_stride,
                    (x0 + bx * 4, span.at(by * 4), step),
                    samples,
                    (qp_base, bypass),
                    &slice.pps.weight4[list],
                    4,
                );
            }
        }
    }
    Ok(())
}

fn mutable_plane<S: Sample>(pic: &mut Frame<S>, plane: u8) -> (&mut [S], usize) {
    let stride = if plane == 0 {
        pic.w
    } else {
        pic.chroma_stride()
    };
    let samples = match plane {
        1 => pic.u.as_mut_slice(),
        2 => pic.v.as_mut_slice(),
        _ => pic.y.as_mut_slice(),
    };
    (samples, stride)
}

#[cfg(test)]
mod intra8 {
    use cutlass_decode::{ColorMatrix, ColorPrimaries, ColorRange, Transfer};

    use super::super::{Frame, Mb};

    fn frame(width: usize, height: usize) -> Frame {
        let mb_w = width / 16;
        let mb_h = height / 16;
        Frame {
            y: vec![0; width * height].into(),
            u: vec![128; (width / 2) * (height / 2)].into(),
            v: vec![128; (width / 2) * (height / 2)].into(),
            mb: vec![Mb::new(u16::MAX); mb_w * mb_h].into(),
            w: width,
            h: height,
            mb_w,
            poc: 0,
            frame_num: 0,
            referenced: true,
            long_term: false,
            long_idx: -1,
            output: true,
            key: true,
            pts: 0,
            duration: 0,
            timescale: 1,
            crop_x: 0,
            crop_y: 0,
            vis_w: width as u32,
            vis_h: height as u32,
            range: ColorRange::Limited,
            primaries: ColorPrimaries::Bt709,
            transfer: Transfer::Bt709,
            matrix: ColorMatrix::Bt709,
            sample_aspect: cutlass_decode::SampleAspect::UNSPECIFIED,
            num_units: 0,
            time_scale: 0,
            fixed_rate: false,
            nonexisting: false,
            separate_plane: false,
            mbaff: false,
            field: false,
            bottom: false,
            max_ref: 1,
            reorder: 0,
            adaptive: false,
            mmco: Vec::new(),
            max_fn: 16,
            view_id: 0,
            inter_view: true,
            depth: false,
            dps_id: 0,
            shown: super::super::Shown::plain(0, 0),
            svc_left: 0,
            svc_top: 0,
            svc_right: 0,
            svc_bottom: 0,
            levels: Vec::new().into(),
            pending: None,
            writer: None,
        }
    }

    fn paint(pic: &mut Frame, x: usize, y: usize, sample: u8) {
        pic.y[y * pic.w + x] = sample;
    }

    /// Clause 8.3.2.2. The bottom-right 8×8's above-right samples sit in the next
    /// macroblock, which is not constructed yet. They repeat p[7, −1] before the
    /// reference filter, so DC stays on that sample instead of the cleared plane.
    #[test]
    fn intra8_bottom_right_repeats_the_last_above_sample() {
        let mut pic = frame(32, 16);
        let mut mb = Mb::new(7);
        mb.intra = true;
        mb.mode4[2 * 4 + 2] = 2;
        for col in 8..16 {
            paint(&mut pic, col, 7, 200);
        }
        paint(&mut pic, 7, 7, 200);
        for row in 8..16 {
            paint(&mut pic, 7, row, 200);
        }
        let pred = super::pred8_block(
            &pic,
            super::Place { mx: 0, my: 0 },
            super::Cell { bx: 2, by: 2 },
            &mb,
            false,
            0,
        );
        assert!(
            pred.iter().all(|sample| *sample == 200),
            "dc drifted to {}",
            pred[0]
        );

        pic.mb[1].slice = 7;
        pic.mb[1].intra = true;
        let pred = super::pred8_block(
            &pic,
            super::Place { mx: 0, my: 0 },
            super::Cell { bx: 2, by: 2 },
            &mb,
            false,
            0,
        );
        assert_eq!(pred[0], 197, "an available zero above-right must be read");
    }

    /// Clause 9.3.3.1.1.6. A direct partition keeps a reference for motion, and that
    /// reference does not raise the CABAC context of the next ref_idx.
    #[test]
    fn direct_neighbour_does_not_raise_the_ref_idx_context() {
        let mut pic = frame(32, 16);
        let mut left = Mb::new(3);
        left.direct16 = true;
        left.ref0 = [2, 2, 2, 2];
        pic.mb[0] = left;
        let current = Mb::new(3);
        assert_eq!(super::ref_ctx(&pic, 1, 0, 0, 0, 0, &current), 0);
        pic.mb[0].uses0 = 0b1111;
        assert_eq!(super::ref_ctx(&pic, 1, 0, 0, 0, 0, &current), 1);
    }

    /// Clause 8.4.1.3. A partition with predFlagL0 = 0 is still decoded, so C is
    /// that partition and not the above-left substitute.
    #[test]
    fn an_unused_list_still_occupies_the_neighbour_slot() {
        let mut pic = frame(32, 32);
        let mut left = Mb::new(3);
        left.ref0[1] = 0;
        left.ref0[3] = 0;
        left.mv0[7] = [0, 80];
        left.mv0[11] = [0, 80];
        pic.mb[2] = left;
        let mut current = Mb::new(3);
        current.ref0[0] = 0;
        super::mark_decoded(&mut current, 0, 0, 16, 8);
        let pred = super::mvp(
            &pic,
            super::Place { mx: 1, my: 1 },
            super::Cell { bx: 0, by: 2 },
            (8, 8),
            0,
            0,
            &current,
        );
        assert_eq!(pred, [0, 0]);
    }
}

#[cfg(test)]
mod oracle {
    use std::process::Command;

    use cutlass_decode::{CodecId, CodecParams, Packet, VideoDecoder};

    use crate::H264Decoder;

    fn ffmpeg() -> Option<String> {
        [
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "ffmpeg",
        ]
        .into_iter()
        .find(|path| Command::new(path).arg("-version").output().is_ok())
        .map(str::to_string)
    }

    #[test]
    fn baseline_intra_matches_ffmpeg() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-intra");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("intra.h264");
        let yuv = dir.join("intra.yuv");
        let encoded = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30:duration=1",
                "-frames:v",
                "1",
                "-c:v",
                "libx264",
                "-profile:v",
                "baseline",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=0:bframes=0:weightp=0:8x8dct=0:keyint=1:scenecut=0",
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "ffmpeg encode failed");
        let decoded = Command::new(&ffmpeg)
            .args(["-y", "-i"])
            .arg(&coded)
            .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "ffmpeg decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let params = CodecParams::video(CodecId::H264, 64, 64);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let packet = Packet::video(bytes, 0, 1, 30, true);
        let output = decoder.push(&packet).unwrap();
        let frames = if output.is_empty() {
            decoder.drain().unwrap()
        } else {
            output
        };
        assert_eq!(frames.len(), 1, "expected one picture");
        let picture = &frames[0].picture;
        let mut got = picture.y().to_vec();
        got.extend_from_slice(picture.u());
        got.extend_from_slice(picture.v());
        let mut count = 0usize;
        let mut max = 0u16;
        let mut worst = (0usize, 0u8, 0u8);
        let mut misses = Vec::new();
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a == b {
                continue;
            }
            let delta = u16::from(a.abs_diff(*b));
            count += 1;
            if delta >= max {
                max = delta;
                worst = (index, *a, *b);
            }
            if misses.len() < 8 {
                misses.push((index, *a, *b));
            }
        }
        if count != 0 {
            panic!("intra mismatches {count} max {max} worst {worst:?}: {misses:?}");
        }
    }

    #[test]
    fn baseline_ip_matches_ffmpeg() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-ip");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("ip.h264");
        let yuv = dir.join("ip.yuv");
        let encoded = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30",
                "-frames:v",
                "4",
                "-c:v",
                "libx264",
                "-profile:v",
                "baseline",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=0:bframes=0:weightp=0:8x8dct=0:keyint=4:scenecut=0:ref=1",
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "ffmpeg encode failed");
        let decoded = Command::new(&ffmpeg)
            .args(["-y", "-i"])
            .arg(&coded)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "ffmpeg decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let params = CodecParams::video(CodecId::H264, 64, 64);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let mut got = Vec::new();
        for unit in access_units(&bytes) {
            let packet = Packet::video(unit, 0, 1, 30, false);
            got.extend(pictures(&decoder.push(&packet).unwrap()));
        }
        got.extend(pictures(&decoder.drain().unwrap()));
        assert_eq!(got.len(), expected.len(), "picture bytes");
        let mut count = 0usize;
        let mut max = 0u8;
        let mut first = None;
        let mut worst = (0usize, 0u8, 0u8);
        let mut per = [0usize; 8];
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a == b {
                continue;
            }
            count += 1;
            let delta = a.abs_diff(*b);
            let frame = index / 6144;
            if frame < per.len() {
                per[frame] += 1;
            }
            if delta >= max {
                max = delta;
                worst = (index, *a, *b);
            }
            if first.is_none() {
                first = Some((index, *a, *b));
            }
        }
        if count != 0 {
            panic!("ip mismatches {count} max {max} first {first:?} worst {worst:?} per {per:?}");
        }
    }

    #[test]
    fn main_cabac_ipb_matches_ffmpeg() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-ipb");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("ipb.h264");
        let yuv = dir.join("ipb.yuv");
        let encoded = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30",
                "-frames:v",
                "4",
                "-c:v",
                "libx264",
                "-profile:v",
                "main",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=1:bframes=1:weightp=0:8x8dct=0:keyint=30:scenecut=0:ref=1:threads=1:b-pyramid=0:direct=spatial",
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "ffmpeg encode failed");
        let decoded = Command::new(&ffmpeg)
            .args(["-y", "-i"])
            .arg(&coded)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "ffmpeg decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let params = CodecParams::video(CodecId::H264, 64, 64);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let mut got = Vec::new();
        for (unit_index, unit) in access_units(&bytes).into_iter().enumerate() {
            let packet = Packet::video(unit, 0, 1, 30, false);
            match decoder.push(&packet) {
                Ok(frames) => got.extend(pictures(&frames)),
                Err(error) => panic!("push {unit_index} failed: {error:?}"),
            }
        }
        match decoder.drain() {
            Ok(frames) => got.extend(pictures(&frames)),
            Err(error) => panic!("drain failed: {error:?}"),
        }
        assert_eq!(got.len(), expected.len(), "picture bytes");
        let mut count = 0usize;
        let mut max = 0u8;
        let mut first = None;
        let mut per = [0usize; 8];
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a == b {
                continue;
            }
            count += 1;
            let delta = a.abs_diff(*b);
            let frame = index / 6144;
            if frame < per.len() {
                per[frame] += 1;
            }
            if delta >= max {
                max = delta;
            }
            if first.is_none() {
                first = Some((index, *a, *b));
            }
        }
        if count != 0 {
            panic!(
                "ipb mismatches {count} max {max} first {first:?} per {per:?} got {:?} want {:?}",
                &got[..16],
                &expected[..16]
            );
        }
    }

    #[test]
    fn main_cabac_temporal_direct_matches_ffmpeg() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-temporal");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("temporal.h264");
        let yuv = dir.join("temporal.yuv");
        let encoded = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30",
                "-frames:v",
                "4",
                "-c:v",
                "libx264",
                "-profile:v",
                "main",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=1:bframes=1:weightp=0:8x8dct=0:keyint=30:scenecut=0:ref=1:threads=1:b-pyramid=0:direct=temporal",
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "ffmpeg encode failed");
        let decoded = Command::new(&ffmpeg)
            .args(["-y", "-i"])
            .arg(&coded)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "ffmpeg decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let params = CodecParams::video(CodecId::H264, 64, 64);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let mut got = Vec::new();
        for (unit_index, unit) in access_units(&bytes).into_iter().enumerate() {
            let packet = Packet::video(unit, 0, 1, 30, false);
            match decoder.push(&packet) {
                Ok(frames) => got.extend(pictures(&frames)),
                Err(error) => panic!("push {unit_index} failed: {error:?}"),
            }
        }
        match decoder.drain() {
            Ok(frames) => got.extend(pictures(&frames)),
            Err(error) => panic!("drain failed: {error:?}"),
        }
        assert_eq!(got.len(), expected.len(), "picture bytes");
        let mut count = 0usize;
        let mut max = 0u8;
        let mut first = None;
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a == b {
                continue;
            }
            count += 1;
            let delta = a.abs_diff(*b);
            if delta >= max {
                max = delta;
            }
            if first.is_none() {
                first = Some((index, *a, *b));
            }
        }
        if count != 0 {
            panic!("temporal mismatches {count} max {max} first {first:?}");
        }
    }

    #[test]
    fn two_slices_match_ffmpeg_across_the_boundary() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-slices");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("slices.h264");
        let yuv = dir.join("slices.yuv");
        let encoded = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30",
                "-frames:v",
                "3",
                "-c:v",
                "libx264",
                "-profile:v",
                "main",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=1:bframes=0:slices=2:keyint=30:scenecut=0:weightp=0:8x8dct=0:threads=1",
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "ffmpeg encode failed");
        let decoded = Command::new(&ffmpeg)
            .args(["-y", "-i"])
            .arg(&coded)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "ffmpeg decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let params = CodecParams::video(CodecId::H264, 64, 64);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let mut got = Vec::new();
        for (unit_index, unit) in access_units(&bytes).into_iter().enumerate() {
            let packet = Packet::video(unit, 0, 1, 30, false);
            match decoder.push(&packet) {
                Ok(frames) => got.extend(pictures(&frames)),
                Err(error) => panic!("push {unit_index} failed: {error:?}"),
            }
        }
        match decoder.drain() {
            Ok(frames) => got.extend(pictures(&frames)),
            Err(error) => panic!("drain failed: {error:?}"),
        }
        assert_eq!(got.len(), expected.len(), "picture bytes");
        let mut count = 0usize;
        let mut max = 0u8;
        let mut first = None;
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a == b {
                continue;
            }
            count += 1;
            if a.abs_diff(*b) >= max {
                max = a.abs_diff(*b);
            }
            if first.is_none() {
                first = Some((index, *a, *b));
            }
        }
        if count != 0 {
            panic!("slice boundary mismatches {count} max {max} first {first:?}");
        }
    }

    #[test]
    fn cropped_picture_matches_ffmpeg() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-crop");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("crop.h264");
        let yuv = dir.join("crop.yuv");
        let encoded = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=70x50:rate=30",
                "-frames:v",
                "1",
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=1:bframes=0:keyint=1:scenecut=0:threads=1",
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "ffmpeg encode failed");
        let decoded = Command::new(&ffmpeg)
            .args(["-y", "-i"])
            .arg(&coded)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "ffmpeg decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let params = CodecParams::video(CodecId::H264, 70, 50);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let mut got = Vec::new();
        let mut size = (0u32, 0u32);
        for (unit_index, unit) in access_units(&bytes).into_iter().enumerate() {
            let packet = Packet::video(unit, 0, 1, 30, false);
            match decoder.push(&packet) {
                Ok(frames) => {
                    if let Some(frame) = frames.first() {
                        size = (frame.picture.width(), frame.picture.height());
                    }
                    got.extend(pictures(&frames));
                }
                Err(error) => panic!("push {unit_index} failed: {error:?}"),
            }
        }
        match decoder.drain() {
            Ok(frames) => {
                if let Some(frame) = frames.first() {
                    size = (frame.picture.width(), frame.picture.height());
                }
                got.extend(pictures(&frames));
            }
            Err(error) => panic!("drain failed: {error:?}"),
        }
        assert_eq!(size, (70, 50), "visible size");
        assert_eq!(got.len(), expected.len(), "picture bytes");
        let mut count = 0usize;
        let mut max = 0u8;
        let mut first = None;
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a == b {
                continue;
            }
            count += 1;
            if a.abs_diff(*b) >= max {
                max = a.abs_diff(*b);
            }
            if first.is_none() {
                first = Some((index, *a, *b));
            }
        }
        if count != 0 {
            panic!("crop mismatches {count} max {max} first {first:?}");
        }
    }

    #[test]
    fn jvt_quant_matrices_match_ffmpeg() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-cqm");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("cqm.h264");
        let yuv = dir.join("cqm.yuv");
        let encoded = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30",
                "-frames:v",
                "2",
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=1:bframes=0:8x8dct=1:keyint=30:scenecut=0:cqm=jvt:threads=1",
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "ffmpeg encode failed");
        let decoded = Command::new(&ffmpeg)
            .args(["-y", "-i"])
            .arg(&coded)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "ffmpeg decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let params = CodecParams::video(CodecId::H264, 64, 64);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let mut got = Vec::new();
        for (unit_index, unit) in access_units(&bytes).into_iter().enumerate() {
            let packet = Packet::video(unit, 0, 1, 30, false);
            match decoder.push(&packet) {
                Ok(frames) => got.extend(pictures(&frames)),
                Err(error) => panic!("push {unit_index} failed: {error:?}"),
            }
        }
        match decoder.drain() {
            Ok(frames) => got.extend(pictures(&frames)),
            Err(error) => panic!("drain failed: {error:?}"),
        }
        assert_eq!(got.len(), expected.len(), "picture bytes");
        let mut count = 0usize;
        let mut max = 0u8;
        let mut first = None;
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a == b {
                continue;
            }
            count += 1;
            if a.abs_diff(*b) >= max {
                max = a.abs_diff(*b);
            }
            if first.is_none() {
                first = Some((index, *a, *b));
            }
        }
        if count != 0 {
            panic!("quant matrix mismatches {count} max {max} first {first:?}");
        }
    }

    #[test]
    fn high_cabac_8x8_matches_ffmpeg() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-high");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("high.h264");
        let yuv = dir.join("high.yuv");
        let encoded = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30",
                "-frames:v",
                "4",
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=1:bframes=1:weightp=0:8x8dct=1:keyint=30:scenecut=0:ref=1:threads=1:b-pyramid=0:direct=spatial",
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "ffmpeg encode failed");
        let decoded = Command::new(&ffmpeg)
            .args(["-y", "-i"])
            .arg(&coded)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "ffmpeg decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let params = CodecParams::video(CodecId::H264, 64, 64);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let mut got = Vec::new();
        for (unit_index, unit) in access_units(&bytes).into_iter().enumerate() {
            let packet = Packet::video(unit, 0, 1, 30, false);
            match decoder.push(&packet) {
                Ok(frames) => got.extend(pictures(&frames)),
                Err(error) => panic!("push {unit_index} failed: {error:?}"),
            }
        }
        match decoder.drain() {
            Ok(frames) => got.extend(pictures(&frames)),
            Err(error) => panic!("drain failed: {error:?}"),
        }
        assert_eq!(got.len(), expected.len(), "picture bytes");
        let mut count = 0usize;
        let mut max = 0u8;
        let mut first = None;
        let mut per = [0usize; 8];
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a == b {
                continue;
            }
            count += 1;
            let delta = a.abs_diff(*b);
            let frame = index / 6144;
            if frame < per.len() {
                per[frame] += 1;
            }
            if delta >= max {
                max = delta;
            }
            if first.is_none() {
                first = Some((index, *a, *b));
            }
        }
        if count != 0 {
            panic!("high mismatches {count} max {max} first {first:?} per {per:?}");
        }
    }

    #[test]
    fn high_implicit_weights_match_ffmpeg() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-weightb");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("weightb.h264");
        let yuv = dir.join("weightb.yuv");
        let encoded = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30",
                "-frames:v",
                "6",
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=1:bframes=2:weightp=0:weightb=1:8x8dct=1:keyint=30:scenecut=0:ref=1:threads=1:b-pyramid=0:direct=spatial",
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "ffmpeg encode failed");
        let decoded = Command::new(&ffmpeg)
            .args(["-y", "-i"])
            .arg(&coded)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "ffmpeg decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let params = CodecParams::video(CodecId::H264, 64, 64);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let mut got = Vec::new();
        for (unit_index, unit) in access_units(&bytes).into_iter().enumerate() {
            let packet = Packet::video(unit, 0, 1, 30, false);
            match decoder.push(&packet) {
                Ok(frames) => got.extend(pictures(&frames)),
                Err(error) => panic!("push {unit_index} failed: {error:?}"),
            }
        }
        match decoder.drain() {
            Ok(frames) => got.extend(pictures(&frames)),
            Err(error) => panic!("drain failed: {error:?}"),
        }
        assert_eq!(got.len(), expected.len(), "picture bytes");
        let mut count = 0usize;
        let mut max = 0u8;
        let mut first = None;
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a != b {
                count += 1;
                max = max.max(a.abs_diff(*b));
                if first.is_none() {
                    first = Some((index, *a, *b));
                }
            }
        }
        assert_eq!(
            count, 0,
            "weightb mismatches {count} max {max} first {first:?}"
        );
    }

    #[test]
    fn main_explicit_weights_match_ffmpeg() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let dir = std::env::temp_dir().join("cutlass-h264-weightp");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("weightp.h264");
        let yuv = dir.join("weightp.yuv");
        let encoded = Command::new(&ffmpeg)
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x64:rate=30,fade=in:0:4",
                "-frames:v",
                "4",
                "-c:v",
                "libx264",
                "-profile:v",
                "main",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                "cabac=1:bframes=0:weightp=1:8x8dct=0:keyint=30:scenecut=0:ref=2:threads=1",
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "ffmpeg encode failed");
        let decoded = Command::new(&ffmpeg)
            .args(["-y", "-i"])
            .arg(&coded)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "ffmpeg decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let params = CodecParams::video(CodecId::H264, 64, 64);
        let mut decoder = H264Decoder::open(&params).unwrap();
        let mut got = Vec::new();
        for (unit_index, unit) in access_units(&bytes).into_iter().enumerate() {
            let packet = Packet::video(unit, 0, 1, 30, false);
            match decoder.push(&packet) {
                Ok(frames) => got.extend(pictures(&frames)),
                Err(error) => panic!("push {unit_index} failed: {error:?}"),
            }
        }
        match decoder.drain() {
            Ok(frames) => got.extend(pictures(&frames)),
            Err(error) => panic!("drain failed: {error:?}"),
        }
        assert_eq!(got.len(), expected.len(), "picture bytes");
        let mut count = 0usize;
        let mut max = 0u8;
        let mut first = None;
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a != b {
                count += 1;
                max = max.max(a.abs_diff(*b));
                if first.is_none() {
                    first = Some((index, *a, *b));
                }
            }
        }
        assert_eq!(
            count, 0,
            "weightp mismatches {count} max {max} first {first:?}"
        );
    }

    /// High pictures from x264, including a B 8×8 whose list-0 neighbor does not
    /// use list 0, and an intra gradient that selects plane and 8×8 prediction.
    #[test]
    fn high_profile_samples_match_ffmpeg() {
        let Some(ffmpeg) = ffmpeg() else {
            eprintln!("skip: ffmpeg is not installed");
            return;
        };
        let gradient = |size: &str| {
            format!(
                "color=c=black:s={size},format=yuv420p,geq=lum='X*255/W':cb='128+X*40/W':cr='128+Y*40/H'"
            )
        };
        let cases = [
            (
                "8x8-intra",
                80,
                64,
                "2",
                gradient("80x64"),
                "cabac=1:bframes=0:8x8dct=1:weightp=0:no-deblock=1:keyint=1:scenecut=0:qp=20",
            ),
            (
                "plane-intra",
                80,
                64,
                "2",
                gradient("80x64"),
                "cabac=1:bframes=0:8x8dct=0:weightp=0:no-deblock=1:keyint=1:scenecut=0:qp=20",
            ),
            (
                "b8x8-weightb",
                320,
                240,
                "4",
                "testsrc2=size=320x240:rate=30".to_owned(),
                "cabac=1:bframes=2:8x8dct=0:weightp=0:no-deblock=1:direct=spatial:partitions=b8x8,i8x8,i4x4:weightb=1",
            ),
            (
                "veryfast",
                1280,
                720,
                "4",
                "testsrc2=size=1280x720:rate=30".to_owned(),
                "cabac=1:bframes=2",
            ),
        ];
        for (label, width, height, frames, graph, params) in cases {
            same_pictures(&ffmpeg, label, width, height, frames, &graph, params);
        }
    }

    fn same_pictures(
        ffmpeg: &str,
        label: &str,
        width: u32,
        height: u32,
        frames: &str,
        graph: &str,
        params: &str,
    ) {
        let dir = std::env::temp_dir().join(format!("cutlass-h264-{label}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let coded = dir.join("s.h264");
        let yuv = dir.join("s.yuv");
        let encoded = Command::new(ffmpeg)
            .args([
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                graph,
                "-frames:v",
                frames,
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-x264-params",
                params,
                "-f",
                "h264",
            ])
            .arg(&coded)
            .status()
            .unwrap();
        assert!(encoded.success(), "{label} encode failed");
        let decoded = Command::new(ffmpeg)
            .args(["-y", "-loglevel", "error", "-threads", "1", "-i"])
            .arg(&coded)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
            .arg(&yuv)
            .status()
            .unwrap();
        assert!(decoded.success(), "{label} decode failed");
        let bytes = std::fs::read(&coded).unwrap();
        let expected = std::fs::read(&yuv).unwrap();
        let mut codec = CodecParams::video(CodecId::H264, width, height);
        codec.nal_length_size = 0;
        let mut decoder = H264Decoder::open(&codec).unwrap();
        let mut got = Vec::new();
        for (unit_index, unit) in access_units(&bytes).into_iter().enumerate() {
            let packet = Packet::video(unit, 0, 1, 30, false);
            match decoder.push(&packet) {
                Ok(frames) => got.extend(pictures(&frames)),
                Err(error) => panic!("{label} push {unit_index} failed: {error:?}"),
            }
        }
        match decoder.drain() {
            Ok(frames) => got.extend(pictures(&frames)),
            Err(error) => panic!("{label} drain failed: {error:?}"),
        }
        assert_eq!(got.len(), expected.len(), "{label} picture bytes");
        let mut count = 0usize;
        let mut max = 0u8;
        let mut first = None;
        for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a != b {
                count += 1;
                max = max.max(a.abs_diff(*b));
                if first.is_none() {
                    first = Some((index, *a, *b));
                }
            }
        }
        assert_eq!(
            count, 0,
            "{label} mismatches {count} max {max} first {first:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn pictures(frames: &[cutlass_decode::VideoOutput]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for frame in frames {
            bytes.extend_from_slice(frame.picture.y());
            bytes.extend_from_slice(frame.picture.u());
            bytes.extend_from_slice(frame.picture.v());
        }
        bytes
    }

    fn access_units(data: &[u8]) -> Vec<Vec<u8>> {
        let mut starts = Vec::new();
        let mut i = 0;
        while i + 3 < data.len() {
            if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
                let header = if i > 0 && data[i - 1] == 0 { i - 1 } else { i };
                starts.push((header, i + 3));
                i += 3;
            } else {
                i += 1;
            }
        }
        let mut units = Vec::new();
        let mut current = Vec::new();
        let mut saw_vcl = false;
        for (n, &(header, body)) in starts.iter().enumerate() {
            let end = starts
                .get(n + 1)
                .map(|(next, _)| *next)
                .unwrap_or(data.len());
            let kind = data[body] & 31;
            let vcl = kind == 1 || kind == 5;
            let starts_picture = vcl && data.get(body + 1).is_some_and(|byte| byte & 0x80 != 0);
            if (kind == 9 || (starts_picture && saw_vcl)) && !current.is_empty() {
                units.push(std::mem::take(&mut current));
                saw_vcl = false;
            }
            current.extend_from_slice(&data[header..end]);
            saw_vcl |= vcl;
        }
        if !current.is_empty() {
            units.push(current);
        }
        units
    }
}

fn corner_sample<S: Sample>(
    plane: &[S],
    stride: usize,
    height: usize,
    x: i32,
    y: i32,
    present: bool,
) -> S {
    if !present || x < 0 || y < 0 {
        return S::mid();
    }
    pix(plane, stride, height, x, y)
}

fn edge16<S: Sample>(
    pic: &Frame<S>,
    mx: usize,
    my: usize,
    slice: u16,
    constrained: bool,
    plane: u8,
) -> (Vec<S>, Vec<S>, bool, bool) {
    let (samples, stride, height) = plane_samples(pic, plane);
    let field = pic.mb.get(my * pic.mb_w + mx).is_some_and(|mb| mb.field_mb);
    let span = luma_span(my, field);
    let above_ok = neighbor(pic, mx as i32, above_my(pic, mx, my), slice)
        .is_some_and(|mb| !constrained || mb.intra);
    let left_ok = mx > 0
        && neighbor(pic, mx as i32 - 1, my as i32, slice)
            .is_some_and(|mb| !constrained || mb.intra);
    let mut above = vec![S::mid(); 17];
    let mut left = vec![S::mid(); 17];
    let above_y = span.y as i32 - span.step as i32;
    if above_ok {
        for (k, slot) in above[..16].iter_mut().enumerate() {
            *slot = pix(samples, stride, height, (mx * 16 + k) as i32, above_y);
        }
        above[16] = pix(samples, stride, height, (mx * 16 + 16) as i32, above_y);
    }
    if left_ok {
        for (k, slot) in left[..16].iter_mut().enumerate() {
            *slot = pix(
                samples,
                stride,
                height,
                mx as i32 * 16 - 1,
                span.at(k) as i32,
            );
        }
        left[16] = pix(
            samples,
            stride,
            height,
            mx as i32 * 16 - 1,
            span.at(16) as i32,
        );
    }
    (above, left, above_ok, left_ok)
}

fn edge_chroma<S: Sample>(
    pic: &Frame<S>,
    mx: usize,
    my: usize,
    cb: bool,
    slice_id: u16,
    constrained: bool,
    rows: usize,
) -> (Vec<S>, Vec<S>, bool, bool) {
    let plane = if cb { &pic.u } else { &pic.v };
    let stride = pic.chroma_stride();
    let height = pic.chroma_height();
    let field = pic.mb.get(my * pic.mb_w + mx).is_some_and(|mb| mb.field_mb);
    let span = chroma_span(my, rows, field);
    let above_ok = neighbor(pic, mx as i32, above_my(pic, mx, my), slice_id)
        .is_some_and(|mb| !constrained || mb.intra);
    let left_ok = mx > 0
        && neighbor(pic, mx as i32 - 1, my as i32, slice_id)
            .is_some_and(|mb| !constrained || mb.intra);
    let mut above = vec![S::mid(); 9];
    let mut left = vec![S::mid(); rows + 1];
    let above_y = span.y as i32 - span.step as i32;
    if above_ok {
        for (k, slot) in above[..8].iter_mut().enumerate() {
            *slot = pix(plane, stride, height, (mx * 8 + k) as i32, above_y);
        }
        above[8] = pix(plane, stride, height, (mx * 8 + 8) as i32, above_y);
    }
    if left_ok {
        for (k, slot) in left[..rows].iter_mut().enumerate() {
            *slot = pix(plane, stride, height, mx as i32 * 8 - 1, span.at(k) as i32);
        }
        left[rows] = pix(
            plane,
            stride,
            height,
            mx as i32 * 8 - 1,
            span.at(rows) as i32,
        );
    }
    (above, left, above_ok, left_ok)
}

fn row_ok<S: Sample>(
    pic: &Frame<S>,
    ax: usize,
    ay: i32,
    slice: u16,
    constrained: bool,
    here: (usize, usize, usize, usize),
) -> bool {
    if ay < 0 {
        return false;
    }
    let (cur_mx, cur_my, cur_bx, cur_by) = here;
    let mbx = (ax / 4) as i32;
    let mut mby = ay / 4;
    let field = pic
        .mb
        .get(cur_my * pic.mb_w + cur_mx)
        .is_some_and(|mb| mb.field_mb);
    if field && mby == cur_my as i32 - 1 {
        mby -= 1;
    }
    if mbx == cur_mx as i32 && mby == cur_my as i32 {
        return scan_of(ax % 4, ay as usize % 4) < scan_of(cur_bx, cur_by);
    }
    let Some(mb) = neighbor(pic, mbx, mby, slice) else {
        return false;
    };
    !constrained || mb.intra
}

fn col_ok<S: Sample>(
    pic: &Frame<S>,
    ax: i32,
    ay: usize,
    slice: u16,
    constrained: bool,
    here: (usize, usize, usize, usize),
) -> bool {
    if ax < 0 {
        return false;
    }
    let (cur_mx, cur_my, cur_bx, cur_by) = here;
    let mbx = ax / 4;
    let mby = (ay / 4) as i32;
    if mbx == cur_mx as i32 && mby == cur_my as i32 {
        return scan_of(ax as usize % 4, ay % 4) < scan_of(cur_bx, cur_by);
    }
    let Some(mb) = neighbor(pic, mbx, mby, slice) else {
        return false;
    };
    !constrained || mb.intra
}

fn above_right_ok<S: Sample>(
    pic: &Frame<S>,
    mx: usize,
    my: usize,
    bx: usize,
    by: usize,
    slice: u16,
    constrained: bool,
) -> bool {
    if by == 0 {
        let mbx = mx + usize::from(bx == 3);
        return neighbor(pic, mbx as i32, above_my(pic, mx, my), slice)
            .is_some_and(|mb| !constrained || mb.intra);
    }
    if bx == 3 {
        return false;
    }
    let target = scan_of(bx + 1, by - 1);
    let current = scan_of(bx, by);
    target < current
}

fn scan_of(bx: usize, by: usize) -> usize {
    match (bx, by) {
        (0, 0) => 0,
        (1, 0) => 1,
        (0, 1) => 2,
        (1, 1) => 3,
        (2, 0) => 4,
        (3, 0) => 5,
        (2, 1) => 6,
        (3, 1) => 7,
        (0, 2) => 8,
        (1, 2) => 9,
        (0, 3) => 10,
        (1, 3) => 11,
        (2, 2) => 12,
        (3, 2) => 13,
        (2, 3) => 14,
        _ => 15,
    }
}

struct SpSamples<'a> {
    luma: &'a [[i32; 16]; 16],
    chroma: &'a ChromaResidual,
    intra: bool,
    qp: i32,
}

struct SpCoded {
    qp: i32,
    luma_cbp: u8,
    chroma_cbp: u8,
}

fn sp_ok<S: Sample>(transform8: bool, chroma: u8) -> Result<(), CodecError> {
    if transform8 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "sp transform",
        });
    }
    if chroma != 1 {
        return Err(CodecError::Unsupported { what: "sp chroma" });
    }
    if S::BITS != 8 {
        return Err(CodecError::Unsupported { what: "sp" });
    }
    Ok(())
}

fn sp_inter<S: Sample>(
    pic: &mut Frame<S>,
    entropy: &mut Entropy<'_, '_>,
    slice: &Slice<'_>,
    at: Place,
    mb: &mut Mb,
    coded: SpCoded,
) -> Result<(), CodecError> {
    sp_ok::<S>(mb.transform8, slice.sps.chroma)?;
    let Place { mx, my } = at;
    let mut luma = [[0i32; 16]; 16];
    for (i, slot) in luma.iter_mut().enumerate() {
        let bit = 1 << ((blk_y(i) / 2) * 2 + blk_x(i) / 2);
        if coded.luma_cbp & bit != 0 {
            *slot = read_block(
                pic,
                entropy,
                (2, 16, false),
                Place { mx, my },
                Cell {
                    bx: blk_x(i),
                    by: blk_y(i),
                },
                mb,
                0,
            )?;
        }
    }
    let chroma = read_chroma(pic, entropy, mx, my, coded.chroma_cbp, mb)?;
    write_sp(
        pic,
        slice,
        at,
        mb,
        SpSamples {
            luma: &luma,
            chroma: &chroma,
            intra: false,
            qp: coded.qp,
        },
    )
}

fn write_sp<S: Sample>(
    pic: &mut Frame<S>,
    slice: &Slice<'_>,
    at: Place,
    mb: &Mb,
    samples: SpSamples<'_>,
) -> Result<(), CodecError> {
    sp_ok::<S>(false, slice.sps.chroma)?;
    let Place { mx, my } = at;
    let span = luma_span(my, mb.field_mb);
    let quant = crate::sp::SpQuant {
        from: samples.qp,
        to: slice.qs,
        switching: samples.intra || slice.sp_switch,
    };
    let list = if samples.intra { 0 } else { 3 };
    for (i, levels) in samples.luma.iter().enumerate() {
        let bx = blk_x(i);
        let by = blk_y(i);
        let x = mx * 16 + bx * 4;
        let y = span.at(by * 4);
        let pred = gather4(&pic.y, pic.w, x, y, span.step);
        let block = crate::sp::luma_block(&pred, levels, quant, &slice.pps.weight4[list]);
        scatter4(&mut pic.y, pic.w, x, y, span.step, &block);
    }
    let planes = [
        gather_chroma(pic, mx, my, mb.field_mb, true),
        gather_chroma(pic, mx, my, mb.field_mb, false),
    ];
    paint_chroma(
        pic,
        slice,
        at,
        SpChroma {
            planes: &planes,
            residual: samples.chroma,
            quant,
            intra: samples.intra,
            field: mb.field_mb,
        },
    );
    Ok(())
}

fn recon_si<S: Sample>(
    pic: &mut Frame<S>,
    slice: &Slice<'_>,
    at: Place,
    mb: &Mb,
    ac: &[[i32; 16]; 16],
    chroma: &ChromaResidual,
    qp: i32,
) -> Result<(), CodecError> {
    sp_ok::<S>(mb.transform8 || mb.intra16, slice.sps.chroma)?;
    let Place { mx, my } = at;
    let span = luma_span(my, mb.field_mb);
    let quant = crate::sp::SpQuant {
        from: qp,
        to: slice.qs,
        switching: true,
    };
    for (i, levels) in ac.iter().enumerate() {
        let cell = Cell {
            bx: blk_x(i),
            by: blk_y(i),
        };
        let predicted = pred4_block(pic, at, cell, mb, slice.pps.constrained_intra, 0);
        let mut pred = [0i32; 16];
        for (slot, sample) in pred.iter_mut().zip(predicted) {
            *slot = sample.into();
        }
        let block = crate::sp::luma_block(&pred, levels, quant, &slice.pps.weight4[0]);
        scatter4(
            &mut pic.y,
            pic.w,
            mx * 16 + cell.bx * 4,
            span.at(cell.by * 4),
            span.step,
            &block,
        );
    }
    let planes = [
        si_chroma_pred(pic, slice, at, mb, true),
        si_chroma_pred(pic, slice, at, mb, false),
    ];
    paint_chroma(
        pic,
        slice,
        at,
        SpChroma {
            planes: &planes,
            residual: chroma,
            quant,
            intra: true,
            field: mb.field_mb,
        },
    );
    Ok(())
}

struct SpChroma<'a> {
    planes: &'a [[i32; 64]; 2],
    residual: &'a ChromaResidual,
    quant: crate::sp::SpQuant,
    intra: bool,
    field: bool,
}

fn paint_chroma<S: Sample>(pic: &mut Frame<S>, slice: &Slice<'_>, at: Place, job: SpChroma<'_>) {
    let Place { mx, my } = at;
    let span = chroma_span(my, 8, job.field);
    let stride = pic.chroma_stride();
    let base = if job.intra { 1 } else { 4 };
    let offsets = [
        slice.pps.chroma_qp_offset,
        slice.pps.second_chroma_qp_offset,
    ];
    let dc = [&job.residual.cb_dc, &job.residual.cr_dc];
    let ac = [&job.residual.cb_ac, &job.residual.cr_ac];
    for (index, plane) in [&mut pic.u, &mut pic.v].into_iter().enumerate() {
        let (from, _) = recon::chroma_qps(job.quant.from, offsets[index], S::QP_BD);
        let (to, _) = recon::chroma_qps(job.quant.to, offsets[index], S::QP_BD);
        let mut levels = [0i32; 4];
        levels.copy_from_slice(&dc[index][..4]);
        let mut blocks = [[0i32; 16]; 4];
        blocks.copy_from_slice(&ac[index][..4]);
        let painted = crate::sp::chroma_plane(
            &job.planes[index],
            &blocks,
            &levels,
            crate::sp::SpQuant {
                from,
                to,
                switching: job.quant.switching,
            },
            &slice.pps.weight4[base + index],
        );
        scatter8(plane, stride, mx * 8, span.y, span.step, &painted);
    }
}

fn si_chroma_pred<S: Sample>(
    pic: &Frame<S>,
    slice: &Slice<'_>,
    at: Place,
    mb: &Mb,
    cb: bool,
) -> [i32; 64] {
    let Place { mx, my } = at;
    let (above, left, above_ok, left_ok) =
        edge_chroma(pic, mx, my, cb, mb.slice, slice.pps.constrained_intra, 8);
    let span = chroma_span(my, 8, mb.field_mb);
    let plane = if cb { &pic.u } else { &pic.v };
    let corner = corner_sample(
        plane,
        pic.chroma_stride(),
        pic.chroma_height(),
        mx as i32 * 8 - 1,
        span.y as i32 - span.step as i32,
        above_ok && left_ok,
    );
    let pred = recon::pred_chroma(mb.chroma_mode, &above, &left, above_ok, left_ok, corner);
    let mut out = [0i32; 64];
    for (slot, sample) in out.iter_mut().zip(pred) {
        *slot = sample.into();
    }
    out
}

fn gather_chroma<S: Sample>(
    pic: &Frame<S>,
    mx: usize,
    my: usize,
    field: bool,
    cb: bool,
) -> [i32; 64] {
    let span = chroma_span(my, 8, field);
    let plane = if cb { &pic.u } else { &pic.v };
    gather8(plane, pic.chroma_stride(), mx * 8, span.y, span.step)
}

fn gather4<S: Sample>(plane: &[S], stride: usize, x: usize, y: usize, step: usize) -> [i32; 16] {
    let mut out = [0i32; 16];
    for row in 0..4 {
        for col in 0..4 {
            out[row * 4 + col] = plane[(y + row * step) * stride + x + col].into();
        }
    }
    out
}

fn gather8<S: Sample>(plane: &[S], stride: usize, x: usize, y: usize, step: usize) -> [i32; 64] {
    let mut out = [0i32; 64];
    for row in 0..8 {
        for col in 0..8 {
            out[row * 8 + col] = plane[(y + row * step) * stride + x + col].into();
        }
    }
    out
}

fn scatter4<S: Sample>(
    plane: &mut [S],
    stride: usize,
    x: usize,
    y: usize,
    step: usize,
    src: &[i32; 16],
) {
    for row in 0..4 {
        for col in 0..4 {
            plane[(y + row * step) * stride + x + col] = S::clip(src[row * 4 + col]);
        }
    }
}

fn scatter8<S: Sample>(
    plane: &mut [S],
    stride: usize,
    x: usize,
    y: usize,
    step: usize,
    src: &[i32; 64],
) {
    for row in 0..8 {
        for col in 0..8 {
            plane[(y + row * step) * stride + x + col] = S::clip(src[row * 8 + col]);
        }
    }
}

fn pix<S: Sample>(plane: &[S], stride: usize, height: usize, x: i32, y: i32) -> S {
    if stride == 0 || height == 0 {
        return S::mid();
    }
    let x = x.clamp(0, stride as i32 - 1) as usize;
    let y = y.clamp(0, height as i32 - 1) as usize;
    plane.get(y * stride + x).copied().unwrap_or(S::mid())
}

fn decode_inter<S: Sample>(
    pic: &mut Frame<S>,
    dpb: &[Frame<S>],
    entropy: &mut Entropy<'_, '_>,
    slice: &Slice<'_>,
    at: Place,
    spec: u32,
    work: (&mut Mb, &mut i32, Option<&Frame<S>>),
) -> Result<bool, CodecError> {
    let Place { mx, my } = at;
    let (mb, qp, layer) = work;
    let sub = if slice.kind == 1 {
        spec == 22
    } else {
        spec == 3 || spec == 4
    };
    let vsp_block = (slice.kind == 0 && matches!(spec, 0..=3)) || (slice.kind == 1 && spec == 1);
    if slice.vsp && !vsp_block {
        return Err(CodecError::Unsupported { what: "3d" });
    }
    if slice.sps.alc && !pic.depth && super::syntax_kind(slice.kind) == 0 && matches!(spec, 0..=2) {
        mb.alc = read_flag(entropy)?;
        if mb.alc {
            mb.alc_spec = spec;
        }
    }
    if (slice.adaptive_motion || slice.motion_pred) && (sub || (slice.kind == 1 && spec == 0)) {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    if slice.kind == 1 && spec == 0 {
        direct_16(pic, dpb, slice, mx, my, mb)?;
    } else if sub {
        read_sub(pic, dpb, entropy, slice, Place { mx, my }, spec, mb)?;
    } else {
        read_parts(pic, dpb, entropy, slice, mb, (mx, my, spec), layer)?;
    }
    let residual = coded_residual_flag(slice, entropy)?;
    let (luma_cbp, chroma_cbp) = read_cbp(pic, entropy, mx, my, false, mb.slice, slice.sps.chroma)?;
    mb.cbp = luma_cbp | (chroma_cbp << 4);
    let direct16 = slice.kind == 1 && spec == 0;
    let allow8 = if direct16 {
        slice.sps.direct_8x8_inference
    } else {
        mb.partition8
    };
    if luma_cbp > 0 && slice.pps.transform_8x8 && allow8 {
        mb.transform8 = transform8_flag(pic, entropy, mx, my, mb.slice)?;
    }
    let mut delta = 0;
    if luma_cbp > 0 || chroma_cbp > 0 {
        delta = qp_delta(entropy, prev_qpd(pic, mx, my, mb.slice))?;
        *qp = next_qp::<S>(*qp, delta);
    }
    mb.qpd = delta as i8;
    predict_inter(pic, dpb, slice, mx, my, mb)?;
    let merge = residual && slice.tcoeff;
    if merge && (mb.field_mb || slice.sps.chroma != 1) {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    if slice.kind == 3 {
        sp_inter(
            pic,
            entropy,
            slice,
            Place { mx, my },
            mb,
            SpCoded {
                qp: *qp,
                luma_cbp,
                chroma_cbp,
            },
        )?;
        return Ok(residual);
    }
    if mb.transform8 {
        for i in 0..4 {
            if luma_cbp & (1 << i) != 0 {
                let block = read8(pic, entropy, mx, my, i, mb, 0)?;
                let bx = (i & 1) * 8;
                let by = (i >> 1) * 8;
                let span = luma_span(my, mb.field_mb);
                if !pic.levels.is_empty() {
                    pic.note_luma8(my * pic.mb_w + mx, i, &block);
                }
                if !merge {
                    super::add_scaled(
                        &mut pic.y,
                        pic.w,
                        (mx * 16 + bx, span.at(by), span.step),
                        &block,
                        (*qp, qp_bypass::<S>(slice, *qp)),
                        &slice.pps.weight8[1],
                        8,
                    );
                }
            }
        }
    } else {
        for i in 0..16 {
            let bit = 1 << ((blk_y(i) / 2) * 2 + blk_x(i) / 2);
            if luma_cbp & bit != 0 {
                let block = read_block(
                    pic,
                    entropy,
                    (2, 16, false),
                    Place { mx, my },
                    Cell {
                        bx: blk_x(i),
                        by: blk_y(i),
                    },
                    mb,
                    0,
                )?;
                let span = luma_span(my, mb.field_mb);
                if !pic.levels.is_empty() {
                    pic.note_luma4(my * pic.mb_w + mx, i, &block);
                }
                if !merge {
                    super::add_scaled(
                        &mut pic.y,
                        pic.w,
                        (mx * 16 + blk_x(i) * 4, span.at(blk_y(i) * 4), span.step),
                        &block,
                        (*qp, qp_bypass::<S>(slice, *qp)),
                        &slice.pps.weight4[3],
                        4,
                    );
                }
            }
        }
    }
    if slice.sps.chroma == 3 && !slice.sps.separate {
        recon_444(pic, entropy, slice, Place { mx, my }, mb, *qp, None)?;
    } else {
        let chroma_res = read_chroma(pic, entropy, mx, my, chroma_cbp, mb)?;
        if !pic.levels.is_empty() {
            let slot = my * pic.mb_w + mx;
            pic.note_chroma(slot, 0, &chroma_res.cb_dc, &chroma_res.cb_ac);
            pic.note_chroma(slot, 1, &chroma_res.cr_dc, &chroma_res.cr_ac);
        }
        if !merge {
            recon_chroma(
                pic,
                slice,
                Place { mx, my },
                0,
                ChromaPaint {
                    cb_dc: &chroma_res.cb_dc,
                    cr_dc: &chroma_res.cr_dc,
                    cb_ac: &chroma_res.cb_ac,
                    cr_ac: &chroma_res.cr_ac,
                },
                (*qp, false, mb.slice),
                false,
            );
        }
    }
    Ok(residual)
}

fn read_parts<S: Sample>(
    pic: &Frame<S>,
    dpb: &[Frame<S>],
    entropy: &mut Entropy<'_, '_>,
    slice: &Slice<'_>,
    mb: &mut Mb,
    at: (usize, usize, u32),
    layer: Option<&Frame<S>>,
) -> Result<(), CodecError> {
    let (mx, my, spec) = at;
    let parts = partitions(super::syntax_kind(slice.kind), spec);
    // One flag per partition; there are at most two.
    let mut from_layer0 = [false; 2];
    let mut from_layer1 = [false; 2];
    let mut synthesis0 = [false; 2];
    let mut synthesis1 = [false; 2];
    if slice.adaptive_motion || slice.motion_pred {
        for (index, part) in parts.iter().enumerate() {
            if part.l0 {
                from_layer0[index] = motion_prediction_flag(slice, entropy)?;
            }
        }
        for (index, part) in parts.iter().enumerate() {
            if part.l1 {
                from_layer1[index] = motion_prediction_flag(slice, entropy)?;
            }
        }
    }
    // Clause 7.3.5.1: every list 0 value, then every list 1 value.
    // A motion-prediction flag borrows that list's reference from the base.
    for (index, part) in parts.iter().enumerate() {
        if !part.l0 {
            continue;
        }
        if from_layer0[index] {
            let (_, reference) = layer_motion(layer, mx, my, part, 0)?;
            fill_ref(&mut mb.ref0, &mut mb.uses0, part, reference);
        } else if mb.alc {
            fill_ref(
                &mut mb.ref0,
                &mut mb.uses0,
                part,
                list0_interview(pic, dpb, slice)?,
            );
        } else if slice.num_ref_l0 > 1 {
            let r = read_te(
                entropy,
                slice.num_ref_l0,
                ref_ctx(pic, mx, my, part.x / 4, part.y / 4, 0, mb),
            )?;
            fill_ref(&mut mb.ref0, &mut mb.uses0, part, r as i8);
            if slice.vsp && interview_reference(pic, dpb, slice, mb, part, 0) {
                synthesis0[index] = read_flag(entropy)?;
            }
        } else {
            fill_ref(&mut mb.ref0, &mut mb.uses0, part, 0);
        }
    }
    for (index, part) in parts.iter().enumerate() {
        if !part.l1 {
            continue;
        }
        if from_layer1[index] {
            let (_, reference) = layer_motion(layer, mx, my, part, 1)?;
            fill_ref(&mut mb.ref1, &mut mb.uses1, part, reference);
        } else if slice.num_ref_l1 > 1 {
            let r = read_te(
                entropy,
                slice.num_ref_l1,
                ref_ctx(pic, mx, my, part.x / 4, part.y / 4, 1, mb),
            )?;
            fill_ref(&mut mb.ref1, &mut mb.uses1, part, r as i8);
            if slice.vsp && interview_reference(pic, dpb, slice, mb, part, 1) {
                synthesis1[index] = read_flag(entropy)?;
            }
        } else {
            fill_ref(&mut mb.ref1, &mut mb.uses1, part, 0);
        }
    }
    // A partition that does not use list 0 is still decoded. Its blocks stay
    // available so the next partition does not replace C with D.
    for part in parts.iter() {
        if !part.l0 {
            mark_decoded(mb, part.x, part.y, part.w, part.h);
        }
    }
    for (index, part) in parts.iter().enumerate() {
        if part.l0 && synthesis0[index] {
            store_mv(
                mb,
                part,
                synthesis_motion(pic, dpb, slice, (mx * 16, my * 16), part)?,
                0,
            );
            continue;
        }
        if part.l0 {
            let predicted = if from_layer0[index] {
                Some(layer_motion(layer, mx, my, part, 0)?.0)
            } else if slice.dmvp {
                if !same_view_reference(pic, dpb, slice, mb, part) {
                    return Err(CodecError::Unsupported { what: "3d" });
                }
                Some(dmvp_temporal_mvp(
                    pic,
                    dpb,
                    slice,
                    Place { mx, my },
                    part,
                    mb,
                ))
            } else {
                None
            };
            read_mvd_part(
                pic,
                entropy,
                slice,
                Place { mx, my },
                part,
                (0, predicted),
                mb,
            )?;
        }
    }
    for (index, part) in parts.iter().enumerate() {
        if part.l1 && synthesis1[index] {
            store_mv(
                mb,
                part,
                synthesis_motion(pic, dpb, slice, (mx * 16, my * 16), part)?,
                1,
            );
            continue;
        }
        if part.l1 {
            let predicted = if from_layer1[index] {
                Some(layer_motion(layer, mx, my, part, 1)?.0)
            } else {
                None
            };
            read_mvd_part(
                pic,
                entropy,
                slice,
                Place { mx, my },
                part,
                (1, predicted),
                mb,
            )?;
        }
    }
    Ok(())
}

fn read_flag(entropy: &mut Entropy<'_, '_>) -> Result<bool, CodecError> {
    match entropy {
        Entropy::Cavlc(bits) => bits.bit(),
        Entropy::Cabac(_) => Err(CodecError::Unsupported { what: "3d" }),
    }
}

fn interview_reference<S: Sample>(
    pic: &Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    mb: &Mb,
    part: &Part,
    list: usize,
) -> bool {
    let Ok(index) = usize::try_from(ref_of(mb, part.x / 4, part.y / 4, list)) else {
        return false;
    };
    let refs = if list == 0 { &slice.ref0 } else { &slice.ref1 };
    let Some(&slot) = refs.get(index) else {
        return false;
    };
    if slot == NO_PICTURE {
        return false;
    }
    dpb.get(slot)
        .is_some_and(|frame| frame.view_id != pic.view_id)
}

fn store_mv(mb: &mut Mb, part: &Part, mv: [i16; 2], list: usize) {
    let x = part.x / 4;
    let y = part.y / 4;
    for by in 0..(part.h / 4) {
        for bx in 0..(part.w / 4) {
            let index = (y + by) * 4 + x + bx;
            if list == 0 {
                mb.mv0[index] = mv;
                mb.mvd0[index] = [0, 0];
            } else {
                mb.mv1[index] = mv;
                mb.mvd1[index] = [0, 0];
            }
            mb.ready |= 1 << index;
        }
    }
}

/// J.8.2.1 with `bvsp_flag` 1 and `TextureFirstFlag` 1. The disparity is the
/// motion vector. `refViewId` is 0, and the depth picture is the base view.
fn synthesis_motion<S: Sample>(
    pic: &Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    origin: (usize, usize),
    part: &Part,
) -> Result<[i16; 2], CodecError> {
    let Some(base_id) = slice.sps.views.first().copied() else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "3d",
        });
    };
    let depth_w = slice.sps.depth_mb_width.saturating_mul(16);
    let depth_h = slice.sps.depth_mb_height.saturating_mul(16);
    let Some(depth) = dpb.iter().find(|frame| {
        frame.depth
            && frame.referenced
            && !frame.nonexisting
            && frame.view_id == base_id
            && frame.poc == pic.poc
            && frame.w == depth_w
            && frame.h == depth_h
    }) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "3d",
        });
    };
    let views = slice.sps.depth_views.iter().filter(|flag| **flag).count();
    let Some(src) = depth_param_index(&slice.sps.views, &slice.sps.depth_views, pic.view_id) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "3d",
        });
    };
    let Some(reference) = depth_param_index(&slice.sps.views, &slice.sps.depth_views, 0) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "3d",
        });
    };
    let scale = disparity_term(&slice.disparity_scale, src, reference, views);
    let offset = disparity_term(&slice.disparity_offset, src, reference, views);
    let max_depth = max_depth(
        depth,
        origin,
        part,
        (slice.sps.depth_hor_mult, slice.sps.depth_hor_rsh),
        (slice.sps.depth_ver_mult, slice.sps.depth_ver_rsh),
    );
    let mapped = i32::from(slice.ndr[usize::try_from(max_depth).unwrap_or(0).min(255)]);
    Ok([disparity_value(mapped, scale, offset), 0])
}

/// J.8.2.1.1 corners. `origin` is the macroblock's luma sample. Each axis is
/// `(texture * mult) >> shift`, then clipped to the depth picture. Absent
/// reduced resolution uses mult 2 and shift 0.
fn max_depth<S: Sample>(
    depth: &Frame<S>,
    origin: (usize, usize),
    part: &Part,
    hor: (u32, u32),
    ver: (u32, u32),
) -> i32 {
    let right = depth.w.saturating_sub(1);
    let bottom = depth.h.saturating_sub(1);
    let x = origin.0.saturating_add(part.x);
    let y = origin.1.saturating_add(part.y);
    let (hor_mult, hor_rsh) = hor;
    let (ver_mult, ver_rsh) = ver;
    let x0 = map_axis(x, hor_mult, hor_rsh, right);
    let y0 = map_axis(y, ver_mult, ver_rsh, bottom);
    let x1 = map_axis(
        x.saturating_add(part.w).saturating_sub(1),
        hor_mult,
        hor_rsh,
        right,
    );
    let y1 = map_axis(
        y.saturating_add(part.h).saturating_sub(1),
        ver_mult,
        ver_rsh,
        bottom,
    );
    let mut max = depth_sample(depth, x0, y0);
    for (x, y) in [(x1, y0), (x0, y1), (x1, y1)] {
        max = max.max(depth_sample(depth, x, y));
    }
    max
}

/// J-38. `(texture * mult) >> shift`, clipped to the depth picture.
fn map_axis(texture: usize, mult: u32, shift: u32, last: usize) -> usize {
    let mult = usize::try_from(mult).unwrap_or(usize::MAX);
    let scaled = texture.saturating_mul(mult);
    let shifted = if shift >= usize::BITS {
        0
    } else {
        scaled >> shift
    };
    shifted.min(last)
}

fn depth_sample<S: Sample>(frame: &Frame<S>, x: usize, y: usize) -> i32 {
    frame
        .y
        .get(y.saturating_mul(frame.w).saturating_add(x))
        .copied()
        .map(Into::into)
        .unwrap_or(0)
}

fn depth_param_index(views: &[u16], depth_views: &[bool], view_id: u16) -> Option<usize> {
    views
        .iter()
        .zip(depth_views)
        .filter(|(_, depth)| **depth)
        .position(|(id, _)| *id == view_id)
}

fn disparity_term(values: &[i32], src: usize, reference: usize, views: usize) -> i32 {
    values
        .get(src.saturating_mul(views).saturating_add(reference))
        .copied()
        .unwrap_or(0)
}

fn disparity_value(depth: i32, scale: i32, offset: i32) -> i16 {
    let sum = i64::from(depth) * i64::from(scale) + (i64::from(offset) << 8) + (1 << 13);
    let shifted = sum >> 14;
    let value = i32::try_from(shifted).unwrap_or(if shifted < 0 { i32::MIN } else { i32::MAX });
    super::clip_ref(value)
}

/// `motion_prediction_flag`. The reference layer supplies the predictor.
fn motion_prediction_flag(
    slice: &Slice<'_>,
    entropy: &mut Entropy<'_, '_>,
) -> Result<bool, CodecError> {
    if !slice.adaptive_motion {
        return Ok(slice.motion_pred);
    }
    match entropy {
        Entropy::Cavlc(bits) => bits.bit(),
        Entropy::Cabac(_) => Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        }),
    }
}

fn layer_motion<S: Sample>(
    layer: Option<&Frame<S>>,
    mx: usize,
    my: usize,
    part: &Part,
    list: usize,
) -> Result<([i16; 2], i8), CodecError> {
    let Some(layer) = layer else {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    };
    let slot = my.saturating_mul(layer.mb_w).saturating_add(mx);
    let Some(reference) = layer.mb.get(slot) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "inter-layer prediction",
        });
    };
    let bx = part.x / 4;
    let by = part.y / 4;
    let block = by * 4 + bx;
    let block8 = (by / 2) * 2 + (bx / 2);
    let (mv, index) = if list == 0 {
        (reference.mv0[block], reference.ref0[block8])
    } else {
        (reference.mv1[block], reference.ref1[block8])
    };
    if index < 0 {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    Ok((mv, index))
}

#[derive(Clone, Copy)]
struct Part {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    l0: bool,
    l1: bool,
}

/// A macroblock's partitions: none, one or two. Held in place, since this
/// is made for every inter macroblock.
struct Parts {
    items: [Part; 2],
    len: usize,
}

impl Parts {
    fn new(parts: &[Part]) -> Self {
        let mut items = [part(0, 0, 16, 16, false, false); 2];
        let len = parts.len().min(2);
        items[..len].copy_from_slice(&parts[..len]);
        Self { items, len }
    }
}

impl std::ops::Deref for Parts {
    type Target = [Part];

    fn deref(&self) -> &[Part] {
        &self.items[..self.len]
    }
}

fn partitions(kind: u8, spec: u32) -> Parts {
    if kind == 0 {
        return match spec {
            0 => Parts::new(&[part(0, 0, 16, 16, true, false)]),
            1 => Parts::new(&[
                part(0, 0, 16, 8, true, false),
                part(0, 8, 16, 8, true, false),
            ]),
            2 => Parts::new(&[
                part(0, 0, 8, 16, true, false),
                part(8, 0, 8, 16, true, false),
            ]),
            _ => Parts::new(&[]),
        };
    }
    let (a, b, wide) = match spec {
        1 => (1, 0, true),
        2 => (0, 1, true),
        3 => (1, 1, true),
        4 => (1, 1, true),
        5 => (1, 1, false),
        6 => (0, 1, true),
        7 => (0, 1, false),
        8 => (1, 0, true),
        9 => (1, 0, false),
        10 => (0, 1, true),
        11 => (0, 1, false),
        12 => (1, 1, true),
        13 => (1, 1, false),
        14 => (0, 1, true),
        15 => (0, 1, false),
        16 => (1, 1, true),
        17 => (1, 1, false),
        18 => (1, 0, true),
        19 => (1, 0, false),
        20 => (1, 1, true),
        21 => (1, 1, false),
        _ => return Parts::new(&[part(0, 0, 16, 16, true, false)]),
    };
    let _ = (a, b);
    let lists = b_lists(spec);
    if wide && matches!(spec, 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20) {
        Parts::new(&[
            part(0, 0, 16, 8, lists.0.0, lists.0.1),
            part(0, 8, 16, 8, lists.1.0, lists.1.1),
        ])
    } else if matches!(spec, 5 | 7 | 9 | 11 | 13 | 15 | 17 | 19 | 21) {
        Parts::new(&[
            part(0, 0, 8, 16, lists.0.0, lists.0.1),
            part(8, 0, 8, 16, lists.1.0, lists.1.1),
        ])
    } else {
        Parts::new(&[part(0, 0, 16, 16, lists.0.0, lists.0.1)])
    }
}

fn b_lists(spec: u32) -> ((bool, bool), (bool, bool)) {
    match spec {
        1 => ((true, false), (true, false)),
        2 => ((false, true), (false, true)),
        3 => ((true, true), (true, true)),
        4 | 5 => ((true, false), (true, false)),
        6 | 7 => ((false, true), (false, true)),
        8 | 9 => ((true, false), (false, true)),
        10 | 11 => ((false, true), (true, false)),
        12 | 13 => ((true, false), (true, true)),
        14 | 15 => ((false, true), (true, true)),
        16 | 17 => ((true, true), (true, false)),
        18 | 19 => ((true, true), (false, true)),
        _ => ((true, true), (true, true)),
    }
}

fn part(x: usize, y: usize, w: usize, h: usize, l0: bool, l1: bool) -> Part {
    Part { x, y, w, h, l0, l1 }
}

fn fill_ref(refs: &mut [i8; 4], uses: &mut u8, part: &Part, value: i8) {
    for y in (part.y / 8)..((part.y + part.h) / 8) {
        for x in (part.x / 8)..((part.x + part.w) / 8) {
            let index = y * 2 + x;
            refs[index] = value;
            *uses |= 1 << index;
        }
    }
}

fn read_te(entropy: &mut Entropy<'_, '_>, range: usize, ctx: usize) -> Result<u32, CodecError> {
    if range <= 1 {
        return Ok(0);
    }
    match entropy {
        Entropy::Cavlc(bits) => {
            if range == 2 {
                Ok(1 - u32::from(bits.bit()?))
            } else {
                bits.ue()
            }
        }
        Entropy::Cabac(cab) => {
            let mut value = 0u32;
            let mut context = ctx;
            while cab.decision(54 + context)? == 1 {
                value += 1;
                context = (context >> 2) + 4;
                if value >= range as u32 {
                    return Err(CodecError::Malformed {
                        at_bit: 0,
                        what: "ref idx",
                    });
                }
            }
            Ok(value)
        }
    }
}

fn ref_ctx<S: Sample>(
    pic: &Frame<S>,
    mx: usize,
    my: usize,
    bx: usize,
    by: usize,
    list: usize,
    mb: &Mb,
) -> usize {
    let left = ref_at(pic, mx, my, bx as i32 - 1, by as i32, list, mb);
    let above = ref_at(pic, mx, my, bx as i32, by as i32 - 1, list, mb);
    usize::from(left > 0) + 2 * usize::from(above > 0)
}

fn ref_at<S: Sample>(
    pic: &Frame<S>,
    mx: usize,
    my: usize,
    bx: i32,
    by: i32,
    list: usize,
    current: &Mb,
) -> i8 {
    let (mbx, ox) = if bx < 0 {
        (mx as i32 - 1, 1)
    } else {
        (mx as i32, (bx as usize) / 2)
    };
    let (mby, oy) = if by < 0 {
        (above_my(pic, mx, my), 1)
    } else {
        (my as i32, (by as usize) / 2)
    };
    let index = oy * 2 + ox.min(1);
    if mbx == mx as i32 && mby == my as i32 {
        return listed_ref(current, index, list);
    }
    neighbor(pic, mbx, mby, current.slice)
        .map(|mb| listed_ref(mb, index, list))
        .unwrap_or(-1)
}

/// Clause 9.3.3.1.1.6. Direct, skip, intra, and the other list contribute no reference.
fn listed_ref(mb: &Mb, index: usize, list: usize) -> i8 {
    let uses = if list == 0 { mb.uses0 } else { mb.uses1 };
    if uses & (1 << index) == 0 {
        return -1;
    }
    if list == 0 {
        mb.ref0[index]
    } else {
        mb.ref1[index]
    }
}

fn read_mvd_part<S: Sample>(
    pic: &Frame<S>,
    entropy: &mut Entropy<'_, '_>,
    slice: &Slice<'_>,
    at: Place,
    part: &Part,
    motion: (usize, Option<[i16; 2]>),
    mb: &mut Mb,
) -> Result<(), CodecError> {
    let (list, predicted) = motion;
    let Place { mx, my } = at;
    let x = part.x / 4;
    let y = part.y / 4;
    let pred = if let Some(mv) = predicted {
        mv
    } else {
        mvp(
            pic,
            Place { mx, my },
            Cell { bx: x, by: y },
            (part.w, part.h),
            list,
            ref_of(mb, x, y, list),
            mb,
        )
    };
    let dx = read_mvd(
        entropy,
        mvd_sum(pic, Place { mx, my }, Cell { bx: x, by: y }, (list, 0), mb),
        40,
    )?;
    let dy = read_mvd(
        entropy,
        mvd_sum(pic, Place { mx, my }, Cell { bx: x, by: y }, (list, 1), mb),
        47,
    )?;
    let mv = [
        super::clip_ref(i32::from(pred[0]) + dx),
        super::clip_ref(i32::from(pred[1]) + dy),
    ];
    let mvd = [super::clip_ref(dx), super::clip_ref(dy)];
    let bw = part.w / 4;
    let bh = part.h / 4;
    for by in 0..bh {
        for bx in 0..bw {
            let index = (y + by) * 4 + x + bx;
            if list == 0 {
                mb.mv0[index] = mv;
                mb.mvd0[index] = mvd;
            } else {
                mb.mv1[index] = mv;
                mb.mvd1[index] = mvd;
            }
            mb.ready |= 1 << index;
        }
    }
    let _ = slice;
    Ok(())
}

fn ref_of(mb: &Mb, bx: usize, by: usize, list: usize) -> i8 {
    let index = (by / 2) * 2 + (bx / 2);
    if list == 0 {
        mb.ref0[index]
    } else {
        mb.ref1[index]
    }
}

fn read_mvd(entropy: &mut Entropy<'_, '_>, sum: i32, base: usize) -> Result<i32, CodecError> {
    match entropy {
        Entropy::Cavlc(bits) => bits.se(),
        Entropy::Cabac(cab) => {
            let ctx = usize::from(sum > 2) + usize::from(sum > 32);
            if cab.decision(base + ctx)? == 0 {
                return Ok(0);
            }
            let mut value = 1i32;
            let mut context = base + 3;
            while value < 9 && cab.decision(context)? == 1 {
                if value < 4 {
                    context += 1;
                }
                value += 1;
            }
            if value >= 9 {
                let mut k = 3i32;
                while cab.bypass()? == 1 {
                    value += 1 << k;
                    k += 1;
                    if k > 16 {
                        return Err(CodecError::Malformed {
                            at_bit: 0,
                            what: "mvd",
                        });
                    }
                }
                for bit in (0..k).rev() {
                    value += i32::from(cab.bypass()?) << bit;
                }
            }
            let sign = cab.bypass()?;
            Ok(if sign == 1 { -value } else { value })
        }
    }
}

fn mvd_sum<S: Sample>(
    pic: &Frame<S>,
    at: Place,
    cell: Cell,
    which: (usize, usize),
    mb: &Mb,
) -> i32 {
    let Place { mx, my } = at;
    let Cell { bx, by } = cell;
    let (list, comp) = which;
    let left = mvd_at(
        pic,
        Place { mx, my },
        (bx as i32 - 1, by as i32),
        (list, comp),
        mb,
    );
    let above = mvd_at(
        pic,
        Place { mx, my },
        (bx as i32, by as i32 - 1),
        (list, comp),
        mb,
    );
    left.abs() + above.abs()
}

fn mvd_at<S: Sample>(
    pic: &Frame<S>,
    at: Place,
    cell: (i32, i32),
    which: (usize, usize),
    current: &Mb,
) -> i32 {
    let Place { mx, my } = at;
    let (bx, by) = cell;
    let (list, comp) = which;
    let (mbx, ox) = shift(mx, bx, 4);
    let (mby, oy) = shift_field(my, by, 4, current.field_mb);
    let index = oy * 4 + ox;
    if mbx == mx as i32 && mby == my as i32 {
        let mv = if list == 0 {
            current.mvd0[index]
        } else {
            current.mvd1[index]
        };
        return i32::from(mv[comp]);
    }
    neighbor(pic, mbx, mby, current.slice)
        .map(|mb| {
            i32::from(if list == 0 {
                mb.mvd0[index][comp]
            } else {
                mb.mvd1[index][comp]
            })
        })
        .unwrap_or(0)
}

#[derive(Clone, Copy)]
struct Motion {
    available: bool,
    mv: [i16; 2],
    reference: i8,
}

fn mvp<S: Sample>(
    pic: &Frame<S>,
    at: Place,
    cell: Cell,
    size: (usize, usize),
    list: usize,
    reference: i8,
    current: &Mb,
) -> [i16; 2] {
    let Place { mx, my } = at;
    let Cell { bx, by } = cell;
    let (w, h) = size;
    let a = part_motion(pic, mx, my, bx as i32 - 1, by as i32, list, current);
    let mut b = part_motion(pic, mx, my, bx as i32, by as i32 - 1, list, current);
    let above_right = part_motion(
        pic,
        mx,
        my,
        bx as i32 + (w / 4) as i32,
        by as i32 - 1,
        list,
        current,
    );
    let mut c = if above_right.available {
        above_right
    } else {
        part_motion(pic, mx, my, bx as i32 - 1, by as i32 - 1, list, current)
    };
    if w == 16 && h == 8 && by == 0 && b.reference == reference {
        return b.mv;
    }
    if w == 16 && h == 8 && by == 2 && a.reference == reference {
        return a.mv;
    }
    if w == 8 && h == 16 && bx == 0 && a.reference == reference {
        return a.mv;
    }
    if w == 8 && h == 16 && bx == 2 && c.reference == reference {
        return c.mv;
    }
    if !b.available && !c.available && a.available {
        b = a;
        c = a;
    }
    let mut same = [a, b, c].into_iter().filter(|n| n.reference == reference);
    if let (Some(only), None) = (same.next(), same.next()) {
        return only.mv;
    }
    [
        median(a.mv[0], b.mv[0], c.mv[0]),
        median(a.mv[1], b.mv[1], c.mv[1]),
    ]
}

fn median(a: i16, b: i16, c: i16) -> i16 {
    a.max(b).min(a.max(c)).min(b.max(c))
}

/// The partition is decoded, so later motion-vector prediction may see it.
fn mark_decoded(mb: &mut Mb, x: usize, y: usize, w: usize, h: usize) {
    let mut py = y;
    while py < y + h {
        let mut px = x;
        while px < x + w {
            let index = (py / 4) * 4 + (px / 4);
            mb.ready |= 1 << index;
            px += 4;
        }
        py += 4;
    }
}

fn part_motion<S: Sample>(
    pic: &Frame<S>,
    mx: usize,
    my: usize,
    bx: i32,
    by: i32,
    list: usize,
    current: &Mb,
) -> Motion {
    let (mbx, ox) = shift(mx, bx, 4);
    let (mby, oy) = shift_field(my, by, 4, current.field_mb);
    let index = oy * 4 + ox;
    if mbx == mx as i32 && mby == my as i32 {
        if current.ready & (1 << index) == 0 {
            return Motion {
                available: false,
                mv: [0, 0],
                reference: -1,
            };
        }
        return motion_of(current, index, list);
    }
    let Some(mb) = neighbor(pic, mbx, mby, current.slice) else {
        return Motion {
            available: false,
            mv: [0, 0],
            reference: -1,
        };
    };
    motion_of(mb, index, list)
}

fn motion_of(mb: &Mb, index: usize, list: usize) -> Motion {
    if mb.intra {
        return Motion {
            available: true,
            mv: [0, 0],
            reference: -1,
        };
    }
    let part = (index / 4 / 2) * 2 + ((index % 4) / 2);
    let reference = if list == 0 {
        mb.ref0[part]
    } else {
        mb.ref1[part]
    };
    if reference < 0 {
        return Motion {
            available: true,
            mv: [0, 0],
            reference: -1,
        };
    }
    let mv = if list == 0 {
        mb.mv0[index]
    } else {
        mb.mv1[index]
    };
    Motion {
        available: true,
        mv,
        reference,
    }
}

fn read_sub<S: Sample>(
    pic: &Frame<S>,
    dpb: &[Frame<S>],
    entropy: &mut Entropy<'_, '_>,
    slice: &Slice<'_>,
    at: Place,
    spec: u32,
    mb: &mut Mb,
) -> Result<(), CodecError> {
    let Place { mx, my } = at;
    let mut kinds = [0u32; 4];
    for slot in &mut kinds {
        *slot = match entropy {
            Entropy::Cavlc(bits) => bits.ue()?,
            Entropy::Cabac(cab) => {
                if super::syntax_kind(slice.kind) == 0 {
                    if cab.decision(21)? == 1 {
                        0
                    } else if cab.decision(22)? == 0 {
                        1
                    } else if cab.decision(23)? == 1 {
                        2
                    } else {
                        3
                    }
                } else if cab.decision(36)? == 0 {
                    0
                } else if cab.decision(37)? == 0 {
                    1 + u32::from(cab.decision(39)?)
                } else {
                    let mut kind = 3u32;
                    if cab.decision(38)? == 1 {
                        if cab.decision(39)? == 1 {
                            *slot = 11 + u32::from(cab.decision(39)?);
                            continue;
                        }
                        kind += 4;
                    }
                    kind += 2 * u32::from(cab.decision(39)?);
                    kind += u32::from(cab.decision(39)?);
                    kind
                }
            }
        };
    }
    for kind in kinds {
        if kind == 0 && slice.kind == 1 {
            if !slice.sps.direct_8x8_inference {
                mb.partition8 = false;
            }
            continue;
        }
        let (_, _, w, h) = sub_shape(super::syntax_kind(slice.kind), kind);
        if w < 8 || h < 8 {
            mb.partition8 = false;
        }
    }
    let ref0_forced = spec == 4;
    let direct = |kind: u32| kind == 0 && slice.kind == 1;
    if slice.vsp {
        for kind in kinds {
            let (_, _, w, h) = sub_shape(super::syntax_kind(slice.kind), kind);
            if !matches!((w, h), (8, 8) | (8, 4) | (4, 8) | (4, 4)) {
                return Err(CodecError::Unsupported { what: "3d" });
            }
        }
    }
    let mut synthesis0 = [false; 4];
    let mut synthesis1 = [false; 4];
    // Clause 7.3.5.2 and J.7.3.6.2: list 0 references, list 1 references, then deltas.
    // `bvsp_flag` follows the reference it belongs to.
    for (i, kind) in kinds.iter().copied().enumerate() {
        if direct(kind) {
            continue;
        }
        let (l0, _) = sub_lists(super::syntax_kind(slice.kind), kind);
        if !l0 {
            continue;
        }
        let x = (i & 1) * 2;
        let y = (i >> 1) * 2;
        let r = if ref0_forced || slice.num_ref_l0 <= 1 {
            0
        } else {
            read_te(entropy, slice.num_ref_l0, ref_ctx(pic, mx, my, x, y, 0, mb))? as i8
        };
        let index = (y / 2) * 2 + (x / 2);
        mb.ref0[index] = r;
        mb.uses0 |= 1 << index;
        if slice.vsp {
            let block = part(x * 4, y * 4, 8, 8, true, false);
            if interview_reference(pic, dpb, slice, mb, &block, 0) {
                synthesis0[i] = if ref0_forced || slice.num_ref_l0 <= 1 {
                    true
                } else {
                    read_flag(entropy)?
                };
            }
        }
    }
    for (i, kind) in kinds.iter().copied().enumerate() {
        if direct(kind) {
            continue;
        }
        let (_, l1) = sub_lists(super::syntax_kind(slice.kind), kind);
        if !l1 {
            continue;
        }
        let x = (i & 1) * 2;
        let y = (i >> 1) * 2;
        let r = if slice.num_ref_l1 <= 1 {
            0
        } else {
            read_te(entropy, slice.num_ref_l1, ref_ctx(pic, mx, my, x, y, 1, mb))? as i8
        };
        let index = (y / 2) * 2 + (x / 2);
        mb.ref1[index] = r;
        mb.uses1 |= 1 << index;
        if slice.vsp {
            let block = part(x * 4, y * 4, 8, 8, false, true);
            if interview_reference(pic, dpb, slice, mb, &block, 1) {
                synthesis1[i] = if slice.num_ref_l1 <= 1 {
                    true
                } else {
                    read_flag(entropy)?
                };
            }
        }
    }
    for (i, kind) in kinds.iter().copied().enumerate() {
        if direct(kind) {
            let x0 = (i & 1) * 8;
            let y0 = (i >> 1) * 8;
            direct_8(pic, dpb, slice, Place { mx, my }, (x0, y0), mb);
        }
    }
    for (i, kind) in kinds.iter().copied().enumerate() {
        if direct(kind) {
            continue;
        }
        let (l0, _, _, _) = sub_shape(super::syntax_kind(slice.kind), kind);
        if l0 {
            continue;
        }
        let x0 = (i & 1) * 8;
        let y0 = (i >> 1) * 8;
        mark_decoded(mb, x0, y0, 8, 8);
    }
    for list in 0..2 {
        for (i, kind) in kinds.iter().copied().enumerate() {
            if direct(kind) {
                continue;
            }
            let (l0, l1, w, h) = sub_shape(super::syntax_kind(slice.kind), kind);
            if (list == 0 && !l0) || (list == 1 && !l1) {
                continue;
            }
            let x0 = (i & 1) * 8;
            let y0 = (i >> 1) * 8;
            for (px, py) in sub_origins(x0, y0, w, h) {
                let part = part(px, py, w, h, list == 0, list == 1);
                let synthesis = if list == 0 {
                    synthesis0[i]
                } else {
                    synthesis1[i]
                };
                if synthesis {
                    // J.8.2.1 derives the disparity from the macroblock partition.
                    // A 4×4, 4×8, or 8×4 sub-partition still samples its 8×8.
                    let parent = Part {
                        x: x0,
                        y: y0,
                        w: 8,
                        h: 8,
                        l0: list == 0,
                        l1: list == 1,
                    };
                    store_mv(
                        mb,
                        &part,
                        synthesis_motion(pic, dpb, slice, (mx * 16, my * 16), &parent)?,
                        list,
                    );
                    continue;
                }
                let predicted = if slice.dmvp && list == 0 {
                    if !same_view_reference(pic, dpb, slice, mb, &part) {
                        return Err(CodecError::Unsupported { what: "3d" });
                    }
                    Some(dmvp_temporal_mvp(
                        pic,
                        dpb,
                        slice,
                        Place { mx, my },
                        &part,
                        mb,
                    ))
                } else {
                    None
                };
                read_mvd_part(
                    pic,
                    entropy,
                    slice,
                    Place { mx, my },
                    &part,
                    (list, predicted),
                    mb,
                )?;
            }
        }
    }
    Ok(())
}

/// The origins of an 8×8 block's sub-partitions of `w`×`h`, in decoding
/// order.
fn sub_origins(x0: usize, y0: usize, w: usize, h: usize) -> impl Iterator<Item = (usize, usize)> {
    let (w, h) = (w.max(1), h.max(1));
    (y0..y0 + 8)
        .step_by(h)
        .flat_map(move |y| (x0..x0 + 8).step_by(w).map(move |x| (x, y)))
}

fn sub_lists(kind: u8, spec: u32) -> (bool, bool) {
    if kind == 0 {
        return (true, false);
    }
    match spec {
        0 | 1 | 4 | 5 | 10 => (true, false),
        2 | 6 | 7 | 11 => (false, true),
        _ => (true, true),
    }
}

fn sub_shape(kind: u8, spec: u32) -> (bool, bool, usize, usize) {
    let (l0, l1) = sub_lists(kind, spec);
    let (w, h) = if kind == 0 {
        match spec {
            1 => (8, 4),
            2 => (4, 8),
            3 => (4, 4),
            _ => (8, 8),
        }
    } else {
        match spec {
            4 | 6 | 8 => (8, 4),
            5 | 7 | 9 => (4, 8),
            10..=12 => (4, 4),
            _ => (8, 8),
        }
    };
    (l0, l1, w, h)
}

fn direct_8<S: Sample>(
    pic: &Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    at: Place,
    origin: (usize, usize),
    mb: &mut Mb,
) {
    let Place { mx, my } = at;
    let (x, y) = origin;
    let mut blocks = [(0usize, 0usize); 4];
    let mut n = 0;
    for by in 0..2 {
        for bx in 0..2 {
            blocks[n] = (x / 4 + bx, y / 4 + by);
            n += 1;
        }
    }
    apply_direct(pic, dpb, slice, mx, my, mb, &blocks);
}

fn direct_16<S: Sample>(
    pic: &Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    mx: usize,
    my: usize,
    mb: &mut Mb,
) -> Result<(), CodecError> {
    let mut blocks = [(0usize, 0usize); 16];
    for (index, slot) in blocks.iter_mut().enumerate() {
        *slot = (index % 4, index / 4);
    }
    apply_direct(pic, dpb, slice, mx, my, mb, &blocks);
    Ok(())
}

/// Clause 8.4.1.2.3. Scale the co-located motion vector by the picture-order distance.
fn temporal_direct<S: Sample>(
    pic: &Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    mx: usize,
    my: usize,
    mb: &mut Mb,
    blocks: &[(usize, usize)],
) {
    let addr = my * pic.mb_w + mx;
    let col = slice
        .ref1
        .first()
        .and_then(|&index| dpb.get(index))
        .and_then(|frame| frame.mb_at(addr).map(|col| (frame.poc, col)));
    for &(bx, by) in blocks {
        let index = by * 4 + bx;
        let part = (by / 2) * 2 + (bx / 2);
        let (ref0, mv0, mv1) = temporal_block(slice, col, bx, by);
        mb.mv0[index] = mv0;
        mb.mv1[index] = mv1;
        mb.ref0[part] = ref0;
        mb.ref1[part] = 0;
        mb.ready |= 1 << index;
    }
}

fn temporal_block(
    slice: &Slice<'_>,
    col: Option<(i32, &Mb)>,
    bx: usize,
    by: usize,
) -> (i8, [i16; 2], [i16; 2]) {
    let Some((col_poc, col)) = col else {
        return (0, [0, 0], [0, 0]);
    };
    if col.intra || col.pcm {
        return (0, [0, 0], [0, 0]);
    }
    let part = (by / 2) * 2 + (bx / 2);
    let index = if slice.sps.direct_8x8_inference {
        (by & !1) * 4 + (bx & !1)
    } else {
        by * 4 + bx
    };
    let (mv_col, ref_poc) = if col.ref0[part] >= 0 {
        (col.mv0[index], col.ref_poc0[part])
    } else if col.ref1[part] >= 0 {
        (col.mv1[index], col.ref_poc1[part])
    } else {
        return (0, [0, 0], [0, 0]);
    };
    let ref0 = slice
        .poc0
        .iter()
        .position(|&poc| poc == ref_poc)
        .and_then(|index| i8::try_from(index).ok())
        .unwrap_or(0);
    let Some(&pic0) = slice.poc0.get(usize::from(u8::try_from(ref0).unwrap_or(0))) else {
        return (0, [0, 0], [0, 0]);
    };
    let long = slice
        .long0
        .get(usize::from(u8::try_from(ref0).unwrap_or(0)))
        .copied()
        .unwrap_or(false);
    let td = (col_poc - pic0).clamp(-128, 127);
    if long || td == 0 {
        return (ref0, mv_col, [0, 0]);
    }
    let tb = (slice.poc - pic0).clamp(-128, 127);
    let tx = (16_384 + (td.abs() / 2)) / td;
    let factor = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
    let mv0 = [scale_mv(factor, mv_col[0]), scale_mv(factor, mv_col[1])];
    let mv1 = [
        clip_mv(i32::from(mv0[0]) - i32::from(mv_col[0])),
        clip_mv(i32::from(mv0[1]) - i32::from(mv_col[1])),
    ];
    (ref0, mv0, mv1)
}

fn scale_mv(factor: i32, component: i16) -> i16 {
    clip_mv((factor * i32::from(component) + 128) >> 8)
}

fn clip_mv(value: i32) -> i16 {
    i16::try_from(value.clamp(-32_768, 32_767)).unwrap_or(0)
}

/// Clause 8.4.1.2.2. One reference pair for the macroblock; each 4×4 may be forced to a zero vector.
fn apply_direct<S: Sample>(
    pic: &Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    mx: usize,
    my: usize,
    mb: &mut Mb,
    blocks: &[(usize, usize)],
) {
    if !slice.spatial_direct {
        temporal_direct(pic, dpb, slice, mx, my, mb, blocks);
        return;
    }
    let (ref0, ref1, zero) = direct_refs(pic, mx, my, mb);
    let pred0 = if zero || ref0 < 0 {
        [0, 0]
    } else {
        mvp(
            pic,
            Place { mx, my },
            Cell { bx: 0, by: 0 },
            (16, 16),
            0,
            ref0,
            mb,
        )
    };
    let pred1 = if zero || ref1 < 0 {
        [0, 0]
    } else {
        mvp(
            pic,
            Place { mx, my },
            Cell { bx: 0, by: 0 },
            (16, 16),
            1,
            ref1,
            mb,
        )
    };
    let addr = my * pic.mb_w + mx;
    // The co-located block is asked about only for a reference index of 0,
    // and its macroblock is the same for every block here.
    let col_matters = !zero && (ref0 == 0 || ref1 == 0);
    let col_mb = if col_matters {
        col_macroblock(dpb, slice, addr)
    } else {
        None
    };
    for &(bx, by) in blocks {
        let index = by * 4 + bx;
        let col = if slice.sps.direct_8x8_inference {
            (by & !1) * 4 + (bx & !1)
        } else {
            index
        };
        let col_zero = col_matters && col_mb.is_some_and(|mb| col_zero(mb, col));
        mb.mv0[index] = if zero || ref0 < 0 || (ref0 == 0 && col_zero) {
            [0, 0]
        } else {
            pred0
        };
        mb.mv1[index] = if zero || ref1 < 0 || (ref1 == 0 && col_zero) {
            [0, 0]
        } else {
            pred1
        };
        let part = (by / 2) * 2 + bx / 2;
        mb.ref0[part] = if zero { 0 } else { ref0 };
        mb.ref1[part] = if zero { 0 } else { ref1 };
        mb.ready |= 1 << index;
    }
}

fn direct_refs<S: Sample>(pic: &Frame<S>, mx: usize, my: usize, mb: &Mb) -> (i8, i8, bool) {
    let reference = |list: usize| {
        let a = part_motion(pic, mx, my, -1, 0, list, mb);
        let b = part_motion(pic, mx, my, 0, -1, list, mb);
        let right = part_motion(pic, mx, my, 4, -1, list, mb);
        let c = if right.available {
            right
        } else {
            part_motion(pic, mx, my, -1, -1, list, mb)
        };
        min_positive(a.reference, min_positive(b.reference, c.reference))
    };
    let mut ref0 = reference(0);
    let mut ref1 = reference(1);
    if ref0 < 0 && ref1 < 0 {
        ref0 = 0;
        ref1 = 0;
        return (ref0, ref1, true);
    }
    (ref0, ref1, false)
}

fn min_positive(left: i8, right: i8) -> i8 {
    if left >= 0 && right >= 0 {
        left.min(right)
    } else {
        left.max(right)
    }
}

/// The co-located macroblock, when it is one whose motion can make a
/// direct block's vector zero: an inter macroblock of a short-term first
/// list-1 picture.
fn col_macroblock<'a, S: Sample>(
    dpb: &'a [Frame<S>],
    slice: &Slice<'_>,
    addr: usize,
) -> Option<&'a Mb> {
    let frame = dpb.get(*slice.ref1.first()?)?;
    if frame.long_term {
        return None;
    }
    let col = frame.mb_at(addr)?;
    (!col.intra && !col.pcm).then_some(col)
}

/// colZeroFlag of clause 8.4.1.2.2 for block `index` of that macroblock.
fn col_zero(col: &Mb, index: usize) -> bool {
    let part = (index / 4 / 2) * 2 + ((index % 4) / 2);
    let (mv, reference) = if col.ref0[part] >= 0 {
        (col.mv0[index], col.ref0[part])
    } else if col.ref1[part] >= 0 {
        (col.mv1[index], col.ref1[part])
    } else {
        return false;
    };
    reference == 0 && (-1..=1).contains(&mv[0]) && (-1..=1).contains(&mv[1])
}

fn list0_interview<S: Sample>(
    pic: &Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
) -> Result<i8, CodecError> {
    for (index, &slot) in slice.ref0.iter().enumerate().take(slice.num_ref_l0) {
        if slot == NO_PICTURE {
            continue;
        }
        let Some(frame) = dpb.get(slot) else {
            continue;
        };
        if frame.view_id != pic.view_id {
            return i8::try_from(index).map_err(|_| CodecError::Malformed {
                at_bit: 0,
                what: "3d",
            });
        }
    }
    Err(CodecError::Malformed {
        at_bit: 0,
        what: "3d",
    })
}

/// J.8.2.4. A partition that touches the picture edge uses weight 1 and a shift of 15.
/// Otherwise the weight is the ratio of the reconstructed neighbors to the reference neighbors.
fn alc_luma<S: Sample>(
    pic: &Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    origin: (usize, usize),
    size: (usize, usize),
    reference: i8,
    mv: [i16; 2],
) -> Result<(i32, i32), CodecError> {
    let frame = reference_frame(dpb, slice, 0, reference)?;
    Ok(alc_weight(pic, frame, origin, size, mv))
}

fn alc_weight<S: Sample>(
    pic: &Frame<S>,
    frame: &Frame<S>,
    origin: (usize, usize),
    size: (usize, usize),
    mv: [i16; 2],
) -> (i32, i32) {
    let (x_t, y_t) = origin;
    let (width, height) = size;
    let mvx = i32::from(mv[0]);
    let mvy = i32::from(mv[1]);
    let x_i = i32::try_from(x_t).unwrap_or(i32::MAX);
    let y_i = i32::try_from(y_t).unwrap_or(i32::MAX);
    let wide = i32::try_from(width).unwrap_or(i32::MAX);
    let high = i32::try_from(height).unwrap_or(i32::MAX);
    let pic_w = i32::try_from(pic.w).unwrap_or(i32::MAX);
    let pic_h = i32::try_from(pic.h).unwrap_or(i32::MAX);
    let left = i64::from(mvx) + (i64::from(x_i) - 1) * 4;
    let top = i64::from(mvy) + (i64::from(y_i) - 1) * 4;
    let right = i64::from(mvx) + i64::from(x_i.saturating_add(wide)) * 4;
    let bottom = i64::from(mvy) + i64::from(y_i.saturating_add(high)) * 4;
    if left < 0
        || top < 0
        || right >= i64::from(pic_w) * 4
        || bottom >= i64::from(pic_h) * 4
        || x_t == 0
        || y_t == 0
    {
        return (1, 15);
    }
    let frac = (mvx & 3, mvy & 3);
    let mut ref_sum = 1i64;
    let mut sum = 1i64;
    for row in 0..height {
        let rec = constructed(pic, x_t - 1, y_t + row);
        let sample = ref_luma(
            frame,
            x_i + (mvx >> 2) - 1,
            y_i + (mvy >> 2) + i32::try_from(row).unwrap_or(0),
            frac,
        );
        if (rec - sample).abs() < 31 {
            ref_sum += i64::from(sample);
            sum += i64::from(rec);
        }
    }
    for col in 0..width {
        let rec = constructed(pic, x_t + col, y_t - 1);
        let sample = ref_luma(
            frame,
            x_i + (mvx >> 2) + i32::try_from(col).unwrap_or(0),
            y_i + (mvy >> 2) - 1,
            frac,
        );
        if (rec - sample).abs() < 31 {
            ref_sum += i64::from(sample);
            sum += i64::from(rec);
        }
    }
    if (sum >> 4) == (ref_sum >> 4) {
        return (1, 0);
    }
    let weight = ((1i64 << 15) * sum + (ref_sum >> 1)) / ref_sum;
    (
        i32::try_from(weight).unwrap_or(if weight < 0 { i32::MIN } else { i32::MAX }),
        15,
    )
}

fn constructed<S: Sample>(pic: &Frame<S>, x: usize, y: usize) -> i32 {
    pic.y
        .get(y.saturating_mul(pic.w).saturating_add(x))
        .copied()
        .map(Into::into)
        .unwrap_or(0)
}

fn ref_luma<S: Sample>(frame: &Frame<S>, x: i32, y: i32, frac: (i32, i32)) -> i32 {
    let mut sample = [S::default()];
    recon::put_luma_block(
        &mut sample,
        1,
        &frame.y,
        frame.w,
        (
            i32::try_from(frame.w).unwrap_or(0),
            i32::try_from(frame.h).unwrap_or(0),
        ),
        recon::BlockMc {
            x,
            y,
            w: 1,
            h: 1,
            x_frac: frac.0,
            y_frac: frac.1,
        },
    );
    sample[0].into()
}

fn predict_inter<S: Sample>(
    pic: &mut Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    mx: usize,
    my: usize,
    mb: &Mb,
) -> Result<(), CodecError> {
    if mb.alc && slice.weighted {
        for part in partitions(super::syntax_kind(slice.kind), mb.alc_spec).iter() {
            if !part.l0 {
                continue;
            }
            let bx = part.x / 4;
            let by = part.y / 4;
            let r0 = ref_of(mb, bx, by, 0);
            let mv0 = mb.mv0[by * 4 + bx];
            let origin = (mx * 16 + part.x, my * 16 + part.y);
            let (weight, denom) = alc_luma(pic, dpb, slice, origin, (part.w, part.h), r0, mv0)?;
            mc_block(
                pic,
                dpb,
                slice,
                Block {
                    x: origin.0,
                    y: origin.1,
                    w: part.w,
                    h: part.h,
                },
                McMotion {
                    r0,
                    r1: -1,
                    mv0,
                    mv1: [0, 0],
                },
                Some(AlcWeight { weight, denom }),
            )?;
        }
        return Ok(());
    }
    // One motion for the whole macroblock, as in a skipped or 16×16 one:
    // the search for equal neighbours below would find just this.
    if mb.mv0 == [mb.mv0[0]; 16]
        && mb.mv1 == [mb.mv1[0]; 16]
        && mb.ref0 == [mb.ref0[0]; 4]
        && mb.ref1 == [mb.ref1[0]; 4]
    {
        return mc_block(
            pic,
            dpb,
            slice,
            Block {
                x: mx * 16,
                y: my * 16,
                w: 16,
                h: 16,
            },
            McMotion {
                r0: mb.ref0[0],
                r1: mb.ref1[0],
                mv0: mb.mv0[0],
                mv1: mb.mv1[0],
            },
            None,
        );
    }
    let mut done = [false; 16];
    for by in 0..4 {
        for bx in 0..4 {
            let index = by * 4 + bx;
            if done[index] {
                continue;
            }
            let r0 = ref_of(mb, bx, by, 0);
            let r1 = ref_of(mb, bx, by, 1);
            let mv0 = mb.mv0[index];
            let mv1 = mb.mv1[index];
            let mut bw = 1usize;
            while bx + bw < 4 && same_motion(mb, bx + bw, by, r0, r1, mv0, mv1) {
                bw += 1;
            }
            let mut bh = 1usize;
            while by + bh < 4 {
                let row = (0..bw).all(|dx| {
                    !done[(by + bh) * 4 + bx + dx]
                        && same_motion(mb, bx + dx, by + bh, r0, r1, mv0, mv1)
                });
                if !row {
                    break;
                }
                bh += 1;
            }
            for dy in 0..bh {
                for dx in 0..bw {
                    done[(by + dy) * 4 + bx + dx] = true;
                }
            }
            mc_block(
                pic,
                dpb,
                slice,
                Block {
                    x: mx * 16 + bx * 4,
                    y: my * 16 + by * 4,
                    w: bw * 4,
                    h: bh * 4,
                },
                McMotion { r0, r1, mv0, mv1 },
                None,
            )?;
        }
    }
    Ok(())
}

fn same_motion(
    mb: &Mb,
    bx: usize,
    by: usize,
    r0: i8,
    r1: i8,
    mv0: [i16; 2],
    mv1: [i16; 2],
) -> bool {
    let index = by * 4 + bx;
    ref_of(mb, bx, by, 0) == r0
        && ref_of(mb, bx, by, 1) == r1
        && mb.mv0[index] == mv0
        && mb.mv1[index] == mv1
}

struct Block {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

struct McMotion {
    r0: i8,
    r1: i8,
    mv0: [i16; 2],
    mv1: [i16; 2],
}

struct AlcWeight {
    weight: i32,
    denom: i32,
}

fn mc_block<S: Sample>(
    pic: &mut Frame<S>,
    dpb: &[Frame<S>],
    slice: &Slice<'_>,
    block: Block,
    motion: McMotion,
    alc: Option<AlcWeight>,
) -> Result<(), CodecError> {
    let Block { x, y, w, h } = block;
    let McMotion { r0, r1, mv0, mv1 } = motion;
    let chroma = pic.chroma();
    let separate = pic.separate_plane;
    let (cw, ch) = match chroma {
        3 => (w, h),
        2 => (w / 2, h),
        _ => (w / 2, h / 2),
    };
    let luma_mix = if let Some(AlcWeight { weight, denom }) = alc {
        Mix::Uni {
            weight,
            offset: 0,
            denom,
            from: 0,
        }
    } else {
        mix(slice, r0, r1, -1)
    };
    let chroma_mix = if alc.is_some() {
        Mix::Uni {
            weight: 1,
            offset: 0,
            denom: 0,
            from: 0,
        }
    } else {
        mix(slice, r0, r1, 0)
    };
    let chroma_v = if alc.is_some() {
        Mix::Uni {
            weight: 1,
            offset: 0,
            denom: 0,
            from: 0,
        }
    } else {
        mix(slice, r0, r1, 1)
    };
    // One reference stored as it is: predict straight into the picture.
    if let Mix::Copy(list) = luma_mix {
        let (reference, mv) = if list == 0 { (r0, mv0) } else { (r1, mv1) };
        let frame = reference_frame(dpb, slice, list, reference)?;
        let stride = pic.w;
        let dst = &mut pic.y[y * stride + x..];
        predict_block(dst, stride, frame, (x, y, w, h), mv);
    } else {
        let mut luma0 = [S::default(); 256];
        let mut luma1 = [S::default(); 256];
        if r0 >= 0 {
            let frame = reference_frame(dpb, slice, 0, r0)?;
            predict_luma(&mut luma0, frame, x, y, w, h, mv0);
        }
        if r1 >= 0 {
            let frame = reference_frame(dpb, slice, 1, r1)?;
            predict_luma(&mut luma1, frame, x, y, w, h, mv1);
        }
        store_block(
            &mut pic.y,
            pic.w,
            (x, y),
            (w, h),
            (&luma0, &luma1),
            luma_mix,
            S::default(),
        );
    }
    if !separate && cw > 0 && ch > 0 {
        let stride = if chroma == 3 { pic.w } else { pic.w / 2 };
        let origin = match chroma {
            3 => (x, y),
            2 => (x / 2, y),
            _ => (x / 2, y / 2),
        };
        let size = (cw, ch);
        if let (Mix::Copy(list), Mix::Copy(_)) = (chroma_mix, chroma_v) {
            let (reference, mv) = if list == 0 { (r0, mv0) } else { (r1, mv1) };
            let frame = reference_frame(dpb, slice, list, reference)?;
            let at = origin.1 * stride + origin.0;
            let planes = (&mut pic.u[at..], &mut pic.v[at..]);
            predict_chroma(planes, stride, frame, (x, y), size, mv);
            return Ok(());
        }
        let mut u0 = [S::default(); 256];
        let mut v0 = [S::default(); 256];
        let mut u1 = [S::default(); 256];
        let mut v1 = [S::default(); 256];
        if r0 >= 0 {
            let frame = reference_frame(dpb, slice, 0, r0)?;
            predict_chroma((&mut u0, &mut v0), cw, frame, (x, y), size, mv0);
        }
        if r1 >= 0 {
            let frame = reference_frame(dpb, slice, 1, r1)?;
            predict_chroma((&mut u1, &mut v1), cw, frame, (x, y), size, mv1);
        }
        store_block(
            &mut pic.u,
            stride,
            origin,
            size,
            (&u0, &u1),
            chroma_mix,
            S::mid(),
        );
        store_block(
            &mut pic.v,
            stride,
            origin,
            size,
            (&v0, &v1),
            chroma_v,
            S::mid(),
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Mix {
    Fill,
    Copy(usize),
    Uni {
        weight: i32,
        offset: i32,
        denom: i32,
        from: usize,
    },
    Avg,
    Weighted(recon::BiWeight),
}

fn mix(slice: &Slice<'_>, r0: i8, r1: i8, plane: i8) -> Mix {
    if r0 >= 0 && r1 >= 0 {
        if slice.weighted {
            Mix::Weighted(explicit_bi(slice, r0, r1, plane))
        } else if slice.pps.weighted_bipred == 2 && slice.kind == 1 {
            let (w0, w1) = implicit_weights(slice, r0, r1);
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
        uni_mix(slice, 0, r0, plane)
    } else if r1 >= 0 {
        uni_mix(slice, 1, r1, plane)
    } else {
        Mix::Fill
    }
}

fn explicit_bi(slice: &Slice<'_>, r0: i8, r1: i8, plane: i8) -> recon::BiWeight {
    let left = usize::try_from(r0).unwrap_or(0);
    let right = usize::try_from(r1).unwrap_or(0);
    if plane < 0 {
        recon::BiWeight {
            w0: slice.yw[0][left],
            w1: slice.yw[1][right],
            o0: slice.yo[0][left],
            o1: slice.yo[1][right],
            denom: slice.luma_denom,
        }
    } else {
        let plane = usize::try_from(plane).unwrap_or(0);
        recon::BiWeight {
            w0: slice.cw[0][left][plane],
            w1: slice.cw[1][right][plane],
            o0: slice.co[0][left][plane],
            o1: slice.co[1][right][plane],
            denom: slice.chroma_denom,
        }
    }
}

fn uni_mix(slice: &Slice<'_>, list: usize, reference: i8, plane: i8) -> Mix {
    if !slice.weighted {
        return Mix::Copy(list);
    }
    let index = usize::try_from(reference).unwrap_or(0);
    let (weight, offset, denom) = if plane < 0 {
        (
            slice.yw[list][index],
            slice.yo[list][index],
            slice.luma_denom,
        )
    } else {
        let plane = usize::try_from(plane).unwrap_or(0);
        (
            slice.cw[list][index][plane],
            slice.co[list][index][plane],
            slice.chroma_denom,
        )
    };
    // The default weight, 2^denom with no offset, gives the sample back.
    if offset == 0 && (0..15).contains(&denom) && weight == 1 << denom {
        return Mix::Copy(list);
    }
    Mix::Uni {
        weight,
        offset,
        denom,
        from: list,
    }
}

fn store_block<S: Sample>(
    dst: &mut [S],
    stride: usize,
    origin: (usize, usize),
    size: (usize, usize),
    pred: (&[S], &[S]),
    mix: Mix,
    fill: S,
) {
    let (x, y) = origin;
    let (w, h) = size;
    let (left, right) = pred;
    match mix {
        Mix::Fill => {
            for row in 0..h {
                let at = (y + row) * stride + x;
                dst[at..at + w].fill(fill);
            }
        }
        Mix::Copy(which) => {
            let src = if which == 0 { left } else { right };
            for row in 0..h {
                let at = (y + row) * stride + x;
                dst[at..at + w].copy_from_slice(&src[row * w..row * w + w]);
            }
        }
        Mix::Uni {
            weight,
            offset,
            denom,
            from,
        } => {
            let src = if from == 0 { left } else { right };
            for row in 0..h {
                let at = (y + row) * stride + x;
                let out = &mut dst[at..at + w];
                for (sample, &pred) in out.iter_mut().zip(&src[row * w..row * w + w]) {
                    *sample = recon::weight_uni(pred, weight, offset, denom);
                }
            }
        }
        Mix::Avg => {
            for row in 0..h {
                let at = (y + row) * stride + x;
                recon::avg_row(
                    &mut dst[at..at + w],
                    &left[row * w..row * w + w],
                    &right[row * w..row * w + w],
                );
            }
        }
        Mix::Weighted(weight)
            if weight.w0 == 32
                && weight.w1 == 32
                && weight.o0 == 0
                && weight.o1 == 0
                && weight.denom == 5 =>
        {
            for row in 0..h {
                let at = (y + row) * stride + x;
                recon::avg_row(
                    &mut dst[at..at + w],
                    &left[row * w..row * w + w],
                    &right[row * w..row * w + w],
                );
            }
        }
        Mix::Weighted(weight) => {
            for row in 0..h {
                let at = (y + row) * stride + x;
                recon::weight_row(
                    &mut dst[at..at + w],
                    &left[row * w..row * w + w],
                    &right[row * w..row * w + w],
                    weight,
                );
            }
        }
    }
}

fn reference_frame<'a, S: Sample>(
    dpb: &'a [Frame<S>],
    slice: &Slice<'_>,
    list: usize,
    reference: i8,
) -> Result<&'a Frame<S>, CodecError> {
    let refs = if list == 0 { &slice.ref0 } else { &slice.ref1 };
    let index = usize::try_from(reference).map_err(|_| CodecError::Malformed {
        at_bit: 0,
        what: "ref idx",
    })?;
    let Some(&frame_index) = refs.get(index) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "ref idx",
        });
    };
    if frame_index == NO_PICTURE {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "ref idx",
        });
    };
    let Some(frame) = dpb.get(frame_index) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "dpb",
        });
    };
    if frame.nonexisting {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "non-existing reference",
        });
    }
    Ok(frame)
}

fn predict_luma<S: Sample>(
    dst: &mut [S],
    frame: &Frame<S>,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    mv: [i16; 2],
) {
    predict_block(dst, w, frame, (x, y, w, h), mv);
}

/// A luma block of `frame` moved by `mv`, into `dst` with rows `stride`
/// apart. A picture another thread is decoding is read through the rows it
/// has published, waiting for them if need be; if that picture failed,
/// `dst` is left alone and the picture reading it fails when it ends.
fn predict_block<S: Sample>(
    dst: &mut [S],
    stride: usize,
    frame: &Frame<S>,
    block: (usize, usize, usize, usize),
    mv: [i16; 2],
) {
    let Some(published) = &frame.pending else {
        predict_qpel(dst, stride, &frame.y, (frame.w, frame.h), block, mv);
        return;
    };
    let (x, y, w, h) = block;
    let (mvx, mvy) = (i32::from(mv[0]), i32::from(mv[1]));
    let at = (
        i32::try_from(x).unwrap_or(0) + (mvx >> 2),
        i32::try_from(y).unwrap_or(0) + (mvy >> 2),
    );
    let Ok(window) = published.luma(at.0, at.1) else {
        return;
    };
    recon::put_luma_block(
        dst,
        stride,
        window.samples,
        window.stride,
        (
            i32::try_from(window.stride).unwrap_or(0),
            i32::try_from(window.rows).unwrap_or(0),
        ),
        recon::BlockMc {
            x: window.x,
            y: window.y,
            w,
            h,
            x_frac: mvx & 3,
            y_frac: mvy & 3,
        },
    );
}

/// `dst` starts at the block's first sample and its rows are `stride` apart.
fn predict_qpel<S: Sample>(
    dst: &mut [S],
    stride: usize,
    src: &[S],
    pic: (usize, usize),
    block: (usize, usize, usize, usize),
    mv: [i16; 2],
) {
    let (pic_w, pic_h) = pic;
    let (x, y, w, h) = block;
    let mvx = i32::from(mv[0]);
    let mvy = i32::from(mv[1]);
    recon::put_luma_block(
        dst,
        stride,
        src,
        pic_w,
        (
            i32::try_from(pic_w).unwrap_or(0),
            i32::try_from(pic_h).unwrap_or(0),
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

/// Each of `planes` starts at the block's first sample and its rows are
/// `dst_stride` apart.
fn predict_chroma<S: Sample>(
    planes: (&mut [S], &mut [S]),
    dst_stride: usize,
    frame: &Frame<S>,
    at: (usize, usize),
    size: (usize, usize),
    mv: [i16; 2],
) {
    let (u, v) = planes;
    let (x, y) = at;
    let (w, h) = size;
    // A picture another thread is decoding is 4:2:0 and is read through
    // its published rows, as in `predict_block`.
    if let Some(published) = &frame.pending {
        let (mvx, mvy) = (i32::from(mv[0]), i32::from(mv[1]));
        let at = (
            i32::try_from(x / 2).unwrap_or(0) + (mvx >> 3),
            i32::try_from(y / 2).unwrap_or(0) + (mvy >> 3),
        );
        for (cb, dst) in [(true, u), (false, v)] {
            let Ok(window) = published.chroma(cb, at.0, at.1) else {
                return;
            };
            recon::put_chroma_block(
                dst,
                dst_stride,
                window.samples,
                window.stride,
                (
                    i32::try_from(window.stride).unwrap_or(0),
                    i32::try_from(window.rows).unwrap_or(0),
                ),
                recon::BlockMc {
                    x: window.x,
                    y: window.y,
                    w,
                    h,
                    x_frac: mvx & 7,
                    y_frac: mvy & 7,
                },
            );
        }
        return;
    }
    if frame.chroma() == 3 {
        predict_qpel(
            u,
            dst_stride,
            &frame.u,
            (frame.w, frame.h),
            (x, y, w, h),
            mv,
        );
        predict_qpel(
            v,
            dst_stride,
            &frame.v,
            (frame.w, frame.h),
            (x, y, w, h),
            mv,
        );
        return;
    }
    let mvx = i32::from(mv[0]);
    let mvy = i32::from(mv[1]);
    let full_height = frame.chroma() == 2;
    let block = recon::BlockMc {
        x: i32::try_from(x / 2).unwrap_or(0) + (mvx >> 3),
        y: if full_height {
            i32::try_from(y).unwrap_or(0) + (mvy >> 2)
        } else {
            i32::try_from(y / 2).unwrap_or(0) + (mvy >> 3)
        },
        w,
        h,
        x_frac: mvx & 7,
        y_frac: if full_height { (mvy << 1) & 7 } else { mvy & 7 },
    };
    let pic_h = if full_height { frame.h } else { frame.h / 2 };
    let pic = (
        i32::try_from(frame.w / 2).unwrap_or(0),
        i32::try_from(pic_h).unwrap_or(0),
    );
    let stride = frame.w / 2;
    recon::put_chroma_block(u, dst_stride, &frame.u, stride, pic, block);
    recon::put_chroma_block(v, dst_stride, &frame.v, stride, pic, block);
}

/// Clause 8.4.3.2. Implicit bipred weights from the picture-order distance.
fn implicit_weights(slice: &Slice<'_>, r0: i8, r1: i8) -> (i32, i32) {
    let Some(poc0) = listed(&slice.poc0, r0) else {
        return (32, 32);
    };
    let Some(poc1) = listed(&slice.poc1, r1) else {
        return (32, 32);
    };
    let long = listed(&slice.long0, r0).unwrap_or(true) || listed(&slice.long1, r1).unwrap_or(true);
    let td = (poc1 - poc0).clamp(-128, 127);
    if td == 0 || long {
        return (32, 32);
    }
    let tb = (slice.poc - poc0).clamp(-128, 127);
    let tx = (16_384 + (td / 2).abs()) / td;
    let dist = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
    let w1 = dist >> 2;
    if !(-64..=128).contains(&w1) {
        return (32, 32);
    }
    (64 - w1, w1)
}

fn listed<T: Copy>(values: &[T], index: i8) -> Option<T> {
    let index = usize::try_from(index).ok()?;
    values.get(index).copied()
}

fn pcm_code<S: Sample>(bytes: &[u8], bit: &mut usize) -> Result<S, CodecError> {
    let mut value = 0u32;
    for _ in 0..S::BITS {
        let Some(byte) = bytes.get(*bit / 8).copied() else {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "pcm",
            });
        };
        let shift = 7 - (*bit % 8);
        value = (value << 1) | u32::from((byte >> shift) & 1);
        *bit += 1;
    }
    Ok(S::from_code(value))
}

fn read_pcm<S: Sample>(
    pic: &mut Frame<S>,
    entropy: &mut Entropy<'_, '_>,
    mx: usize,
    my: usize,
    field: bool,
    slice: &Slice<'_>,
) -> Result<(), CodecError> {
    let x0 = mx * 16;
    let span = luma_span(my, field);
    let y0 = span.y;
    let step = span.step;
    let separate = pic.separate_plane;
    match entropy {
        Entropy::Cavlc(bits) => {
            let bits = bits.residual(true)?;
            while bits.pos() % 8 != 0 {
                if bits.bit()? {
                    return Err(bits.malformed("pcm align"));
                }
            }
            for row in 0..16 {
                for col in 0..16 {
                    pic.y[(y0 + row * step) * pic.w + x0 + col] = S::from_code(bits.u(S::BITS)?);
                }
            }
            if !separate {
                let (cw, ch, ox, oy, stride, cstep) = pcm_chroma(pic, x0, my, field);
                for plane in [&mut pic.u, &mut pic.v] {
                    for row in 0..ch {
                        for col in 0..cw {
                            plane[(oy + row * cstep) * stride + ox + col] =
                                S::from_code(bits.u(S::BITS)?);
                        }
                    }
                }
            }
        }
        Entropy::Cabac(cab) => {
            let bytes = cab.tail();
            let mut bit = 0usize;
            for row in 0..16 {
                for col in 0..16 {
                    pic.y[(y0 + row * step) * pic.w + x0 + col] = pcm_code::<S>(bytes, &mut bit)?;
                }
            }
            if !separate {
                let (cw, ch, ox, oy, stride, cstep) = pcm_chroma(pic, x0, my, field);
                for plane in [&mut pic.u, &mut pic.v] {
                    for row in 0..ch {
                        for col in 0..cw {
                            plane[(oy + row * cstep) * stride + ox + col] =
                                pcm_code::<S>(bytes, &mut bit)?;
                        }
                    }
                }
            }
            let rest = bytes.get(bit.div_ceil(8)..).ok_or(CodecError::Truncated {
                at_bit: 0,
                what: "pcm",
            })?;
            **cab = Cabac::open(
                rest,
                slice.qp,
                if super::intra_slice(slice.kind) { 0 } else { 1 },
            )?;
        }
    }
    Ok(())
}

fn pcm_chroma<S: Sample>(
    pic: &Frame<S>,
    x0: usize,
    my: usize,
    field: bool,
) -> (usize, usize, usize, usize, usize, usize) {
    match pic.chroma() {
        3 => {
            let span = luma_span(my, field);
            (16, 16, x0, span.y, pic.w, span.step)
        }
        2 => {
            let span = chroma_span(my, 16, field);
            (8, 16, x0 / 2, span.y, pic.w / 2, span.step)
        }
        _ => {
            let span = chroma_span(my, 8, field);
            (8, 8, x0 / 2, span.y, pic.w / 2, span.step)
        }
    }
}

fn neighbor<S: Sample>(pic: &Frame<S>, mx: i32, my: i32, slice: u16) -> Option<&Mb> {
    if mx < 0 || my < 0 || mx as usize >= pic.mb_w || my as usize >= pic.h / 16 {
        return None;
    }
    let mb = pic.mb.get(my as usize * pic.mb_w + mx as usize)?;
    if mb.slice != slice {
        return None;
    }
    Some(mb)
}

fn xy<S: Sample>(pic: &Frame<S>, addr: usize) -> (usize, usize) {
    (addr % pic.mb_w, addr / pic.mb_w)
}

pub(crate) fn deblock<S: Sample>(pic: &mut Frame<S>) {
    let rows = pic.h / 16;
    deblock_rows(pic, 0..rows);
}

/// Deblocks the macroblock rows `rows` of a picture. The filter goes
/// through macroblocks in raster order, and a row reads only itself and
/// the row above, so the rows of a picture can be filtered in turn as they
/// are decoded, each once the row below it is decoded too: intra prediction
/// of that row reads this one unfiltered.
pub(crate) fn deblock_rows<S: Sample>(pic: &mut Frame<S>, rows: std::ops::Range<usize>) {
    let chroma = pic.chroma();
    // The planes leave the picture for the pass, so each is borrowed for
    // writing once and the macroblocks stay readable beside them.
    let mut y = std::mem::take(&mut pic.y);
    let mut u = std::mem::take(&mut pic.u);
    let mut v = std::mem::take(&mut pic.v);
    deblock_planes(pic, chroma, (&mut y, &mut u, &mut v), rows);
    pic.y = y;
    pic.u = u;
    pic.v = v;
}

/// A macroblock with no luma coefficients and one motion for all of it. No
/// edge inside it is filtered: each side of one predicts the same way and
/// neither holds coefficients, so its strength is 0.
fn inner_edges_flat(mb: &Mb) -> bool {
    !mb.intra && mb.nnz_y == [0; 16] && single_motion(mb)
}

fn deblock_planes<S: Sample>(
    pic: &Frame<S>,
    chroma: u8,
    planes: (&mut [S], &mut [S], &mut [S]),
    rows: std::ops::Range<usize>,
) {
    let (luma, cb, cr) = planes;
    let mb_w = pic.mb_w;
    let mb_h = pic.h / 16;
    let mbaff = pic.mbaff;
    let first = rows.start.min(mb_h).saturating_mul(mb_w);
    let count = rows.end.min(mb_h).saturating_mul(mb_w);
    for addr in first..count {
        let (mx, my) = if mbaff {
            let pair = addr / 2;
            (pair % mb_w, (pair / mb_w) * 2 + (addr % 2))
        } else {
            (addr % mb_w, addr / mb_w)
        };
        let index = my * mb_w + mx;
        if pic
            .mb
            .get(index)
            .is_some_and(|mb| mb.deblock == 1 || mb.pcm)
        {
            continue;
        }
        let transform8 = pic.mb.get(index).is_some_and(|mb| mb.transform8);
        let field = pic.mb.get(index).is_some_and(|mb| mb.field_mb);
        let flat = pic.mb.get(index).is_some_and(inner_edges_flat);
        // Worked out once here, not again for each of the eight edges.
        let known = pic
            .mb
            .get(index)
            .map_or((0, false), |mb| (coded_mask(mb), single_motion(mb)));
        let span = luma_span(my, field);
        let off = offsets(pic, index);
        for edge in 0..4 {
            if edge == 0 && (mx == 0 || !filter_boundary(pic, mx, my, true)) {
                continue;
            }
            if edge != 0 && flat {
                break;
            }
            let x = mx * 16 + edge * 4;
            let bs = edge_strength(pic, (mx, my), edge, true, known);
            // An 8×8 transform has no luma edge at 4 or 12.
            if x >= 4 && !(transform8 && (edge == 1 || edge == 3)) {
                let qpav = edge_qp(pic, mx, my, true, true, true, edge == 0);
                filter_groups(
                    luma,
                    pic.w,
                    (x, span.y),
                    (true, false, 4, span.step),
                    bs,
                    qpav,
                    off,
                );
            }
            if !pic.separate_plane && bs != [0; 4] {
                let at = ChromaEdge {
                    mx,
                    my,
                    edge,
                    vertical: true,
                    transform8,
                    format: chroma,
                };
                filter_chroma(pic, (cb, cr), at, bs, off);
            }
        }
        for edge in 0..4 {
            if edge == 0 && (above_my(pic, mx, my) < 0 || !filter_boundary(pic, mx, my, false)) {
                continue;
            }
            if edge != 0 && flat {
                break;
            }
            let y = span.at(edge * 4);
            let bs = edge_strength(pic, (mx, my), edge, false, known);
            if y >= 4 && !(transform8 && (edge == 1 || edge == 3)) {
                let qpav = edge_qp(pic, mx, my, true, true, false, edge == 0);
                filter_groups(
                    luma,
                    pic.w,
                    (mx * 16, y),
                    (false, false, 4, span.step),
                    bs,
                    qpav,
                    off,
                );
            }
            if !pic.separate_plane && bs != [0; 4] {
                let at = ChromaEdge {
                    mx,
                    my,
                    edge,
                    vertical: false,
                    transform8,
                    format: chroma,
                };
                filter_chroma(pic, (cb, cr), at, bs, off);
            }
        }
    }
}

/// One edge of a macroblock, for its chroma. `format` is the picture's
/// chroma format: 1 is 4:2:0, 2 is 4:2:2, 3 is 4:4:4.
#[derive(Clone, Copy)]
struct ChromaEdge {
    mx: usize,
    my: usize,
    edge: usize,
    vertical: bool,
    transform8: bool,
    format: u8,
}

/// `known` is the current macroblock's [`coded_mask`] and whether it has a
/// [`single_motion`].
fn edge_strength<S: Sample>(
    pic: &Frame<S>,
    at: (usize, usize),
    edge: usize,
    vertical: bool,
    known: (u16, bool),
) -> [u8; 4] {
    let (mx, my) = at;
    let (q_mask, single) = known;
    let Some(current) = pic.mb.get(my * pic.mb_w + mx) else {
        return [0; 4];
    };
    let neighbor = if edge == 0 {
        let (nx, ny) = if vertical {
            (mx as i32 - 1, my as i32)
        } else {
            (mx as i32, above_my(pic, mx, my))
        };
        coded_mb(pic, nx, ny)
    } else {
        None
    };
    // Intra covers the whole macroblock, so every sample of the edge agrees.
    if edge == 0 {
        if current.intra || neighbor.is_some_and(|mb| mb.intra) {
            return [4; 4];
        }
    } else if current.intra {
        return [3; 4];
    }
    // The blocks on the far side: the neighbour's last row or column across
    // the macroblock edge, this macroblock's previous one inside it.
    let (p_mask, across) = if edge == 0 {
        (neighbor.map_or(0, coded_mask), 3)
    } else {
        (q_mask, edge - 1)
    };
    // One motion for the whole macroblock: the two sides of an edge inside
    // it predict alike. Across the macroblock edge the motion answer is the
    // same for all four parts when the neighbour has one motion too.
    let apart = if !single {
        None
    } else if edge != 0 {
        Some(0)
    } else {
        neighbor
            .filter(|mb| single_motion(mb))
            .map(|mb| motion_strength(preds_at(mb, 3, 3), preds_at(current, 0, 0)))
    };
    let mut out = [0u8; 4];
    for (line, sample) in out.iter_mut().enumerate() {
        let (q_bit, p_bit) = if vertical {
            (line * 4 + edge, line * 4 + across)
        } else {
            (edge * 4 + line, across * 4 + line)
        };
        // Coefficients on either side settle it before any motion is read.
        *sample = if (q_mask >> q_bit | p_mask >> p_bit) & 1 == 1 {
            2
        } else if let Some(apart) = apart {
            apart
        } else {
            inter_strength(current, neighbor, edge, line, vertical)
        };
    }
    out
}

/// A bit per 4×4 luma block that counts as holding coefficients for the
/// deblocking strength, bit `y * 4 + x`. See [`luma_coded`].
fn coded_mask(mb: &Mb) -> u16 {
    if mb.pcm {
        return u16::MAX;
    }
    let mut mask = 0u16;
    for (index, &count) in mb.nnz_y.iter().enumerate() {
        mask |= u16::from(count > 0) << index;
    }
    if mb.transform8 {
        // The four blocks of an 8×8 block hold coefficients together.
        for corner in [0, 2, 8, 10] {
            if mask & (0x33 << corner) != 0 {
                mask |= 0x33 << corner;
            }
        }
    }
    mask
}

/// Every block of the macroblock predicts from the same pictures with the
/// same vectors.
fn single_motion(mb: &Mb) -> bool {
    mb.slot0 == [mb.slot0[0]; 4]
        && mb.slot1 == [mb.slot1[0]; 4]
        && mb.mv0 == [mb.mv0[0]; 16]
        && mb.mv1 == [mb.mv1[0]; 16]
}

/// The coefficients the 4×4 luma block at `(x, y)` counts for the
/// deblocking strength. With the 8×8 transform it is the whole 8×8 block
/// that holds coefficients or not (8.7.2.1); CAVLC still counts each 4×4
/// part on its own for its coefficient context, so the four are summed.
fn luma_coded(mb: &Mb, x: usize, y: usize) -> u8 {
    if !mb.transform8 {
        return mb.nnz_y[y * 4 + x];
    }
    let (left, top) = (x & !1, y & !1);
    let mut sum = 0u8;
    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        sum = sum.saturating_add(mb.nnz_y[(top + dy) * 4 + left + dx]);
    }
    sum
}

fn inter_strength(q: &Mb, neighbor: Option<&Mb>, edge: usize, line: usize, vertical: bool) -> u8 {
    let (qx, qy) = if vertical { (edge, line) } else { (line, edge) };
    let q_nnz = luma_coded(q, qx, qy);
    let q_pred = preds_at(q, qx, qy);
    let (p_nnz, p_pred) = if edge == 0 {
        let (ox, oy) = if vertical { (3, line) } else { (line, 3) };
        match neighbor {
            Some(p) => {
                let nnz = if p.pcm { 16 } else { luma_coded(p, ox, oy) };
                (nnz, preds_at(p, ox, oy))
            }
            None => (0, [(-1, [0, 0]), (-1, [0, 0])]),
        }
    } else {
        let (ox, oy) = if vertical {
            (edge - 1, line)
        } else {
            (line, edge - 1)
        };
        (luma_coded(q, ox, oy), preds_at(q, ox, oy))
    };
    if p_nnz > 0 || q_nnz > 0 {
        return 2;
    }
    motion_strength(p_pred, q_pred)
}

fn filter_groups<S: Sample>(
    plane: &mut [S],
    stride: usize,
    at: (usize, usize),
    kind: (bool, bool, usize, usize),
    bs: [u8; 4],
    qpav: i32,
    off: (i32, i32),
) {
    let (x, y) = at;
    let (vertical, chroma, step, row_step) = kind;
    let (offset_a, offset_b) = off;
    // A whole luma edge of 16 samples, or a whole chroma edge of 8.
    if (!chroma && step == 4) || (chroma && step == 2) {
        let whole = recon::Edge {
            vertical,
            bs: 0,
            qpav,
            offset_a,
            offset_b,
            chroma,
            len: step * 4,
            row_step,
        };
        if recon::filter_luma16(plane, stride, x, y, whole, bs) {
            return;
        }
    }
    if bs[0] != 0 && bs[1] == bs[0] && bs[2] == bs[0] && bs[3] == bs[0] {
        recon::filter_edge(
            plane,
            stride,
            x,
            y,
            recon::Edge {
                vertical,
                bs: bs[0],
                qpav,
                offset_a,
                offset_b,
                chroma,
                len: step * 4,
                row_step,
            },
        );
        return;
    }
    for (group, group_bs) in bs.into_iter().enumerate() {
        if group_bs == 0 {
            continue;
        }
        let shift = group * step * row_step;
        let (gx, gy) = if vertical {
            (x, y + shift)
        } else {
            (x + shift, y)
        };
        recon::filter_edge(
            plane,
            stride,
            gx,
            gy,
            recon::Edge {
                vertical,
                bs: group_bs,
                qpav,
                offset_a,
                offset_b,
                chroma,
                len: step,
                row_step,
            },
        );
    }
}

fn offsets<S: Sample>(pic: &Frame<S>, index: usize) -> (i32, i32) {
    pic.mb
        .get(index)
        .map(|mb| (i32::from(mb.off_a), i32::from(mb.off_b)))
        .unwrap_or((0, 0))
}

fn filter_chroma<S: Sample>(
    pic: &Frame<S>,
    planes: (&mut [S], &mut [S]),
    at: ChromaEdge,
    bs: [u8; 4],
    off: (i32, i32),
) {
    let ChromaEdge {
        mx,
        my,
        edge,
        vertical,
        transform8,
        format: chroma,
    } = at;
    let (plane_cb, plane_cr) = planes;
    let field = pic.mb.get(my * pic.mb_w + mx).is_some_and(|mb| mb.field_mb);
    if chroma == 3 {
        if transform8 && (edge == 1 || edge == 3) {
            return;
        }
        let span = luma_span(my, field);
        let pos = if vertical {
            mx * 16 + edge * 4
        } else {
            span.at(edge * 4)
        };
        if pos < 4 {
            return;
        }
        let origin = if vertical {
            (pos, span.y)
        } else {
            (mx * 16, pos)
        };
        let stride = pic.w;
        for cb in [true, false] {
            let qpc = edge_qp(pic, mx, my, false, cb, vertical, edge == 0);
            let plane = if cb { &mut *plane_cb } else { &mut *plane_cr };
            filter_groups(
                plane,
                stride,
                origin,
                (vertical, false, 4, span.step),
                bs,
                qpc,
                off,
            );
        }
        return;
    }
    let rows_full = chroma == 2;
    if vertical && !edge.is_multiple_of(2) {
        return;
    }
    if !vertical && !rows_full && !edge.is_multiple_of(2) {
        return;
    }
    let step = if vertical && rows_full { 4 } else { 2 };
    let rows = if rows_full { 16 } else { 8 };
    let span = chroma_span(my, rows, field);
    let along = if vertical {
        mx * 8 + edge * 2
    } else if rows_full {
        span.at(edge * 4)
    } else {
        span.at(edge * 2)
    };
    if along < 2 {
        return;
    }
    let origin = if vertical {
        (along, span.y)
    } else {
        (mx * 8, along)
    };
    let stride = pic.w / 2;
    for cb in [true, false] {
        let qpc = edge_qp(pic, mx, my, false, cb, vertical, edge == 0);
        let plane = if cb { &mut *plane_cb } else { &mut *plane_cr };
        filter_groups(
            plane,
            stride,
            origin,
            (vertical, true, step, span.step),
            bs,
            qpc,
            off,
        );
    }
}

fn edge_qp<S: Sample>(
    pic: &Frame<S>,
    mx: usize,
    my: usize,
    luma: bool,
    cb: bool,
    vertical: bool,
    boundary: bool,
) -> i32 {
    let index = my * pic.mb_w + mx;
    let current = pic
        .mb
        .get(index)
        .map(|mb| {
            if luma {
                i32::from(mb.qp)
            } else if cb {
                i32::from(mb.qp_c)
            } else {
                i32::from(mb.qp_cr)
            }
        })
        .unwrap_or(26);
    if !boundary {
        return current;
    }
    let (nx, ny) = if vertical {
        (mx as i32 - 1, my as i32)
    } else {
        (mx as i32, above_my(pic, mx, my))
    };
    let Some(other) = coded_mb(pic, nx, ny) else {
        return current;
    };
    let side = if luma {
        i32::from(other.qp)
    } else if cb {
        i32::from(other.qp_c)
    } else {
        i32::from(other.qp_cr)
    };
    (current + side + 1) >> 1
}

/// Clause 8.7.2.1. Compares the pictures actually referenced, on either list.
fn motion_strength(p: [(i8, [i16; 2]); 2], q: [(i8, [i16; 2]); 2]) -> u8 {
    let p_used = u8::from(p[0].0 >= 0) + u8::from(p[1].0 >= 0);
    let q_used = u8::from(q[0].0 >= 0) + u8::from(q[1].0 >= 0);
    if p_used != q_used {
        return 1;
    }
    if p_used == 0 {
        return 0;
    }
    if p_used == 1 {
        let p = if p[0].0 >= 0 { p[0] } else { p[1] };
        let q = if q[0].0 >= 0 { q[0] } else { q[1] };
        return u8::from(p.0 != q.0 || mv_apart(p.1, q.1));
    }
    if !same_pictures(p[0].0, p[1].0, q[0].0, q[1].0) {
        return 1;
    }
    if p[0].0 != p[1].0 {
        let q_mv = |slot: i8| {
            if q[0].0 == slot { q[0].1 } else { q[1].1 }
        };
        return u8::from(mv_apart(p[0].1, q_mv(p[0].0)) || mv_apart(p[1].1, q_mv(p[1].0)));
    }
    let same_list = mv_apart(p[0].1, q[0].1) || mv_apart(p[1].1, q[1].1);
    let crossed = mv_apart(p[0].1, q[1].1) || mv_apart(p[1].1, q[0].1);
    u8::from(same_list && crossed)
}

fn same_pictures(a: i8, b: i8, c: i8, d: i8) -> bool {
    (a == c && b == d) || (a == d && b == c)
}

fn mv_apart(a: [i16; 2], b: [i16; 2]) -> bool {
    (a[0] - b[0]).abs() >= 4 || (a[1] - b[1]).abs() >= 4
}

fn preds_at(mb: &Mb, bx: usize, by: usize) -> [(i8, [i16; 2]); 2] {
    let index = by * 4 + bx;
    let part = (by / 2) * 2 + (bx / 2);
    [
        (mb.slot0[part], mb.mv0[index]),
        (mb.slot1[part], mb.mv1[index]),
    ]
}

fn bind_refs(mb: &mut Mb, slice: &Slice<'_>) {
    for i in 0..4 {
        mb.slot0[i] = ref_slot(&slice.ref0, mb.ref0[i]);
        mb.slot1[i] = ref_slot(&slice.ref1, mb.ref1[i]);
        mb.ref_poc0[i] = listed_poc(&slice.poc0, mb.ref0[i]);
        mb.ref_poc1[i] = listed_poc(&slice.poc1, mb.ref1[i]);
    }
}

fn listed_poc(pocs: &[i32], index: i8) -> i32 {
    let Ok(index) = usize::try_from(index) else {
        return 0;
    };
    pocs.get(index).copied().unwrap_or(0)
}

fn ref_slot(list: &[usize], index: i8) -> i8 {
    let Ok(index) = usize::try_from(index) else {
        return -1;
    };
    let Some(slot) = list.get(index) else {
        return -1;
    };
    i8::try_from(*slot).unwrap_or(-1)
}

/// A macroblock edge is filtered across slices unless this slice asked not to.
fn filter_boundary<S: Sample>(pic: &Frame<S>, mx: usize, my: usize, vertical: bool) -> bool {
    let Some(mb) = coded_mb(pic, mx as i32, my as i32) else {
        return false;
    };
    if mb.deblock != 2 {
        return true;
    }
    let (nx, ny) = if vertical {
        (mx as i32 - 1, my as i32)
    } else {
        (mx as i32, above_my(pic, mx, my))
    };
    coded_mb(pic, nx, ny).is_some_and(|other| other.slice == mb.slice)
}

/// Picture-level intra resampling. The reference layer is an intra picture,
/// and each enhancement sample is the 16-phase intra filter of that picture.
/// Motion and residual resampling stay unsupported.
pub(crate) fn resample_intra<S: Sample>(
    base: &Frame<S>,
    sps: &Sps,
    scale: &crate::set::SvcScale,
) -> Result<Frame<S>, CodecError> {
    if base.field
        || base.mbaff
        || base.mb.is_empty()
        || sps.mbaff
        || !sps.frame_mbs_only
        || sps.separate
        || sps.chroma != 1
        || base.w == 0
        || base.h == 0
        || base.mb.iter().any(|mb| !mb.intra)
    {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let out_w = sps.width as usize;
    let out_h = sps.height as usize;
    if out_w == 0 || out_h == 0 || !out_w.is_multiple_of(16) || !out_h.is_multiple_of(16) {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "inter-layer prediction",
        });
    }
    let left = sample_offset(scale.left)?;
    let top = sample_offset(scale.top)?;
    let right = sample_offset(scale.right)?;
    let bottom = sample_offset(scale.bottom)?;
    let scaled_w = i64::from(sps.width) - i64::from(left) - i64::from(right);
    let scaled_h = i64::from(sps.height) - i64::from(top) - i64::from(bottom);
    let (Ok(scaled_w), Ok(scaled_h)) = (i32::try_from(scaled_w), i32::try_from(scaled_h)) else {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    };
    if scaled_w <= 0 || scaled_h <= 0 || scaled_w % 2 != 0 || scaled_h % 2 != 0 {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let y = resample_plane(
        &base.y,
        Plane {
            base_w: base.w,
            base_h: base.h,
            out_w,
            out_h,
            left,
            top,
            scaled_w,
            scaled_h,
            phase_x: 0,
            phase_y: 0,
            ref_phase_x: 0,
            ref_phase_y: 0,
            chroma: false,
            level: sps.level_idc,
        },
    )?;
    let u = resample_plane(
        &base.u,
        Plane {
            base_w: base.w / 2,
            base_h: base.h / 2,
            out_w: out_w / 2,
            out_h: out_h / 2,
            left: scale.left,
            top: scale.top,
            scaled_w: scaled_w / 2,
            scaled_h: scaled_h / 2,
            phase_x: i32::from(scale.chroma_x),
            phase_y: i32::from(scale.chroma_y),
            ref_phase_x: i32::from(scale.ref_chroma_x),
            ref_phase_y: i32::from(scale.ref_chroma_y),
            chroma: true,
            level: sps.level_idc,
        },
    )?;
    let v = resample_plane(
        &base.v,
        Plane {
            base_w: base.w / 2,
            base_h: base.h / 2,
            out_w: out_w / 2,
            out_h: out_h / 2,
            left: scale.left,
            top: scale.top,
            scaled_w: scaled_w / 2,
            scaled_h: scaled_h / 2,
            phase_x: i32::from(scale.chroma_x),
            phase_y: i32::from(scale.chroma_y),
            ref_phase_x: i32::from(scale.ref_chroma_x),
            ref_phase_y: i32::from(scale.ref_chroma_y),
            chroma: true,
            level: sps.level_idc,
        },
    )?;
    let mb_w = out_w / 16;
    let mb_h = out_h / 16;
    let mut intra = Mb::new(0);
    intra.intra = true;
    let mut frame = base.clone();
    frame.y = y.into();
    frame.u = u.into();
    frame.v = v.into();
    frame.w = out_w;
    frame.h = out_h;
    frame.mb_w = mb_w;
    frame.mb = vec![intra; mb_w.saturating_mul(mb_h)].into();
    Ok(frame)
}

struct Plane {
    base_w: usize,
    base_h: usize,
    out_w: usize,
    out_h: usize,
    left: i32,
    top: i32,
    scaled_w: i32,
    scaled_h: i32,
    phase_x: i32,
    phase_y: i32,
    ref_phase_x: i32,
    ref_phase_y: i32,
    chroma: bool,
    level: u8,
}

/// 16-phase luma taps for intra resampling. Each phase sums to 32.
const LUMA_PHASE: [[i32; 4]; 16] = [
    [0, 32, 0, 0],
    [-1, 32, 2, -1],
    [-2, 31, 4, -1],
    [-3, 30, 6, -1],
    [-3, 28, 8, -1],
    [-4, 26, 11, -1],
    [-4, 24, 14, -2],
    [-3, 22, 16, -3],
    [-3, 19, 19, -3],
    [-3, 16, 22, -3],
    [-2, 14, 24, -4],
    [-1, 11, 26, -4],
    [-1, 8, 28, -3],
    [-1, 6, 30, -3],
    [-1, 4, 31, -2],
    [-1, 2, 32, -1],
];

fn resample_plane<S: Sample>(samples: &[S], plane: Plane) -> Result<Vec<S>, CodecError> {
    if plane.base_w == 0 || plane.base_h == 0 || plane.out_w == 0 || plane.out_h == 0 {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let (scale_x, add_x, shift_x, delta_x) = resample_axis(
        plane.base_w as i32,
        plane.scaled_w,
        plane.phase_x,
        plane.ref_phase_x,
        plane.level,
    )?;
    let (scale_y, add_y, shift_y, delta_y) = resample_axis(
        plane.base_h as i32,
        plane.scaled_h,
        plane.phase_y,
        plane.ref_phase_y,
        plane.level,
    )?;
    let right = plane.out_w as i32 - plane.left - plane.scaled_w;
    let bottom = plane.out_h as i32 - plane.top - plane.scaled_h;
    let mut horizontal = vec![0i32; plane.base_h.saturating_mul(plane.out_w)];
    for row in 0..plane.base_h {
        for col in 0..plane.out_w {
            let at = row * plane.out_w + col;
            let x = col as i32;
            if x < plane.left || x >= plane.out_w as i32 - right {
                horizontal[at] = S::MID;
                continue;
            }
            let (phase, reference) = resample_pos(x, plane.left, scale_x, add_x, shift_x, delta_x)?;
            horizontal[at] = if plane.chroma {
                let taps = chroma_phase(phase);
                taps[0] * plane_sample(samples, plane.base_w, plane.base_h, reference, row as i32)?
                    + taps[1]
                        * plane_sample(
                            samples,
                            plane.base_w,
                            plane.base_h,
                            reference + 1,
                            row as i32,
                        )?
            } else {
                let taps = LUMA_PHASE[phase as usize];
                (0..4).try_fold(0i32, |sum, tap| {
                    Ok(sum
                        + taps[tap]
                            * plane_sample(
                                samples,
                                plane.base_w,
                                plane.base_h,
                                reference + tap as i32 - 1,
                                row as i32,
                            )?)
                })?
            };
        }
    }
    let mut output = vec![S::mid(); plane.out_w.saturating_mul(plane.out_h)];
    for row in 0..plane.out_h {
        for col in 0..plane.out_w {
            let x = col as i32;
            let y = row as i32;
            if x < plane.left
                || x >= plane.out_w as i32 - right
                || y < plane.top
                || y >= plane.out_h as i32 - bottom
            {
                continue;
            }
            let (phase, reference) = resample_pos(y, plane.top, scale_y, add_y, shift_y, delta_y)?;
            let acc = if plane.chroma {
                let taps = chroma_phase(phase);
                taps[0] * horizontal_sample(&horizontal, plane.out_w, plane.base_h, col, reference)?
                    + taps[1]
                        * horizontal_sample(
                            &horizontal,
                            plane.out_w,
                            plane.base_h,
                            col,
                            reference + 1,
                        )?
            } else {
                let taps = LUMA_PHASE[phase as usize];
                (0..4).try_fold(0i32, |sum, tap| {
                    Ok(sum
                        + taps[tap]
                            * horizontal_sample(
                                &horizontal,
                                plane.out_w,
                                plane.base_h,
                                col,
                                reference + tap as i32 - 1,
                            )?)
                })?
            };
            output[row * plane.out_w + col] = S::clip((acc + 512) >> 10);
        }
    }
    Ok(output)
}

fn chroma_phase(phase: i32) -> [i32; 2] {
    let phase = phase.clamp(0, 15);
    [32 - 2 * phase, 2 * phase]
}

fn resample_axis(
    reference: i32,
    scaled: i32,
    phase: i32,
    ref_phase: i32,
    level: u8,
) -> Result<(i64, i64, i32, i32), CodecError> {
    if reference <= 0 || scaled <= 0 {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let shift = if level <= 30 {
        16
    } else {
        31 - ceil_log2(reference)
    };
    if shift < 5 {
        return Err(CodecError::Unsupported {
            what: "inter-layer prediction",
        });
    }
    let scale = ((i64::from(reference) << shift) + i64::from(scaled >> 1)) / i64::from(scaled);
    let scaled_phase = (i64::from(reference) * i64::from(2 + phase)) << (shift - 2);
    let add = (scaled_phase + i64::from(scaled >> 1)) / i64::from(scaled) + (1i64 << (shift - 5));
    let delta = 4 * (2 + ref_phase);
    Ok((scale, add, shift - 4, delta))
}

pub(crate) fn sample_offset(coded: i32) -> Result<i32, CodecError> {
    i32::try_from(i64::from(coded) * 2).map_err(|_| CodecError::Unsupported {
        what: "inter-layer prediction",
    })
}

fn resample_pos(
    at: i32,
    offset: i32,
    scale: i64,
    add: i64,
    shift: i32,
    delta: i32,
) -> Result<(i32, i32), CodecError> {
    let pre = (i64::from(at - offset) * scale + add) >> shift;
    let pre = i32::try_from(pre).map_err(|_| CodecError::Unsupported {
        what: "inter-layer prediction",
    })?;
    let pos = pre.checked_sub(delta).ok_or(CodecError::Unsupported {
        what: "inter-layer prediction",
    })?;
    Ok((pos & 15, pos >> 4))
}

fn ceil_log2(value: i32) -> i32 {
    let Ok(value) = u32::try_from(value) else {
        return 0;
    };
    if value <= 1 {
        return 0;
    }
    let floor = 31 - value.leading_zeros() as i32;
    if value.is_power_of_two() {
        floor
    } else {
        floor + 1
    }
}

fn plane_sample<S: Sample>(
    samples: &[S],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
) -> Result<i32, CodecError> {
    let x = x.clamp(0, width as i32 - 1) as usize;
    let y = y.clamp(0, height as i32 - 1) as usize;
    samples
        .get(y * width + x)
        .map(|sample| (*sample).into())
        .ok_or(CodecError::Malformed {
            at_bit: 0,
            what: "inter-layer prediction",
        })
}

fn horizontal_sample(
    samples: &[i32],
    width: usize,
    height: usize,
    x: usize,
    y: i32,
) -> Result<i32, CodecError> {
    let y = y.clamp(0, height as i32 - 1) as usize;
    samples
        .get(y * width + x)
        .copied()
        .ok_or(CodecError::Malformed {
            at_bit: 0,
            what: "inter-layer prediction",
        })
}

fn coded_mb<S: Sample>(pic: &Frame<S>, mx: i32, my: i32) -> Option<&Mb> {
    if mx < 0 || my < 0 || mx as usize >= pic.mb_w || my as usize >= pic.h / 16 {
        return None;
    }
    let mb = pic.mb.get(my as usize * pic.mb_w + mx as usize)?;
    if mb.slice == u16::MAX { None } else { Some(mb) }
}

#[cfg(test)]
mod qp_wrap {
    use crate::recon::Depth;

    use super::next_qp;

    #[test]
    fn slice_qp_wraps_inside_each_bit_depth() {
        assert_eq!(next_qp::<u8>(51, 1), 0);
        assert_eq!(next_qp::<Depth<9>>(51, 1), -6);
        assert_eq!(next_qp::<Depth<10>>(51, 1), -12);
        assert_eq!(next_qp::<Depth<11>>(51, 1), -18);
        assert_eq!(next_qp::<Depth<12>>(51, 1), -24);
        assert_eq!(next_qp::<Depth<13>>(51, 1), -30);
        assert_eq!(next_qp::<Depth<14>>(51, 1), -36);
        assert_eq!(next_qp::<Depth<14>>(-36, -1), 51);
        assert_eq!(next_qp::<Depth<9>>(-6, 0), -6);
    }
}

//! The decoded picture buffer: reference marking (clause 8.2.5), the frames
//! a `frame_num` gap leaves (8.2.5.2), reference picture lists (8.2.4) and
//! the order pictures leave in. The software decoder keeps its pictures
//! here with their samples; [`crate::stateless`] keeps another decoder's
//! surfaces. Both implement [`Stored`] and share every rule.
//!
//! Two choices of the software decoder hold for both. A frame inferred for
//! a `frame_num` gap is marked and slides older references out, but is
//! never put in a reference list. And a picture with
//! `memory_management_control_operation` 5 keeps its order count.

use cutlass_decode::CodecError;

use crate::header::{MemoryManagementControlOperation, RefPicListModification, intra_slice};
use crate::poc::{Poc, ghost_order};
use crate::set::Sps;

/// A list slot the initialisation process left as "no reference picture".
pub(crate) const NO_PICTURE: usize = usize::MAX;

/// A picture in the buffer, as marking, list building and output see it.
pub(crate) trait Stored: Clone {
    /// `FrameNum`.
    fn frame_num(&self) -> u32;
    fn set_frame_num(&mut self, frame_num: u32);
    /// `PicOrderCnt`.
    fn poc(&self) -> i32;
    /// Luma width and height. Only pictures of the current size are
    /// references for one another.
    fn size(&self) -> (usize, usize);
    /// The frame was inferred for a `frame_num` gap.
    fn nonexisting(&self) -> bool;
    fn referenced(&self) -> bool;
    fn set_referenced(&mut self, referenced: bool);
    fn long_term(&self) -> bool;
    /// `LongTermFrameIdx`, or -1.
    fn long_idx(&self) -> i32;
    /// Marks the picture long-term with `LongTermFrameIdx` `idx`.
    fn set_long(&mut self, idx: i32);
    /// Clears the long-term mark and leaves the index.
    fn set_short(&mut self);
    /// Waiting to be output.
    fn output(&self) -> bool;
    fn set_output(&mut self, output: bool);
    /// `MaxFrameNum` of the picture's sequence.
    fn max_fn(&self) -> u32;
    /// `max_num_ref_frames` of the picture's sequence, 1 to 16.
    fn max_ref(&self) -> usize;
    /// Annex H `view_id`. 0 for an Annex A picture.
    fn view_id(&self) -> u16 {
        0
    }
    /// The picture may be an interview reference.
    fn inter_view(&self) -> bool {
        false
    }
}

/// `PicNum` of a short-term frame while `current` is decoded.
pub(crate) fn pic_num<T: Stored>(frame: &T, current: u32, max_fn: u32) -> i32 {
    let mut number = i32::try_from(frame.frame_num()).unwrap_or(0);
    if frame.frame_num() > current {
        number -= i32::try_from(max_fn).unwrap_or(0);
    }
    number
}

/// A decoded picture joins the buffer: it is marked (clause 8.2.5.3 when
/// `marking` is `None`, else the operations of 8.2.5.4), stored, and the
/// pictures past `reorder` waiting are returned in output order.
pub(crate) fn store<T: Stored>(
    dpb: &mut Vec<T>,
    mut pic: T,
    marking: Option<&[MemoryManagementControlOperation]>,
    poc: &mut Poc,
    reorder: usize,
) -> Result<Vec<T>, CodecError> {
    if pic.referenced() {
        match marking {
            Some(operations) => apply_mmco(dpb, &mut pic, operations, poc)?,
            None => sliding_window(dpb, &pic),
        }
    }
    dpb.push(pic);
    dpb.retain(|frame| frame.referenced() || frame.output());
    Ok(take_output(dpb, reorder))
}

/// Clause 8.2.5.1 for an IDR picture: every earlier picture stops being a
/// reference. With `no_output_of_prior_pics_flag` the ones waiting are
/// dropped; otherwise they are returned, in order, before the IDR.
pub(crate) fn begin_idr<T: Stored>(
    dpb: &mut Vec<T>,
    poc: &mut Poc,
    long_term_reference_flag: bool,
    no_output_of_prior_pics_flag: bool,
) -> Vec<T> {
    poc.max_long = if long_term_reference_flag {
        Some(0)
    } else {
        None
    };
    for frame in dpb.iter_mut() {
        frame.set_referenced(false);
        if no_output_of_prior_pics_flag {
            frame.set_output(false);
        }
    }
    take_output(dpb, 0)
}

/// Clause 8.2.5.3. The oldest short-term frames stop being references
/// while there are more than `max_num_ref_frames` with `pic`.
pub(crate) fn sliding_window<T: Stored>(dpb: &mut [T], pic: &T) {
    let max_ref = pic.max_ref().clamp(1, 16);
    let size = pic.size();
    let same = |frame: &&T| frame.size() == size;
    let mut short = dpb
        .iter()
        .filter(|frame| frame.referenced() && !frame.long_term() && same(frame))
        .count();
    let long = dpb
        .iter()
        .filter(|frame| frame.referenced() && frame.long_term() && same(frame))
        .count();
    short += 1;
    while short + long > max_ref && short > 0 {
        if let Some(pos) = oldest_short(dpb, pic.frame_num(), pic.max_fn(), size) {
            dpb[pos].set_referenced(false);
            short -= 1;
        } else {
            break;
        }
    }
}

/// Clause 8.2.5.4. Adaptive marking replaces the sliding window for this picture.
pub(crate) fn apply_mmco<T: Stored>(
    dpb: &mut [T],
    pic: &mut T,
    operations: &[MemoryManagementControlOperation],
    poc: &mut Poc,
) -> Result<(), CodecError> {
    let malformed = CodecError::Malformed {
        at_bit: 0,
        what: "mmco",
    };
    let mut current_long = false;
    for operation in operations {
        let op = operation.memory_management_control_operation;
        match op {
            1 | 3 => {
                let target = i32::try_from(pic.frame_num()).unwrap_or(0)
                    - i32::try_from(operation.difference_of_pic_nums_minus1).unwrap_or(0)
                    - 1;
                let Some(pos) = short_at(dpb, target, pic.frame_num(), pic.max_fn(), op == 3)
                else {
                    return Err(malformed);
                };
                if op == 1 {
                    dpb[pos].set_referenced(false);
                } else {
                    let idx = i32::try_from(operation.long_term_frame_idx).unwrap_or(i32::MAX);
                    assign_long(dpb, pos, idx, poc.max_long)?;
                }
            }
            2 => {
                let idx = i32::try_from(operation.long_term_pic_num).unwrap_or(i32::MAX);
                if let Some(frame) = dpb.iter_mut().find(|frame| {
                    frame.referenced() && frame.long_term() && frame.long_idx() == idx
                }) {
                    frame.set_referenced(false);
                }
            }
            4 => {
                let plus1 = operation.max_long_term_frame_idx_plus1;
                if plus1 > 16 {
                    return Err(malformed);
                }
                if plus1 == 0 {
                    poc.max_long = None;
                    for frame in dpb.iter_mut().filter(|frame| frame.long_term()) {
                        frame.set_referenced(false);
                    }
                } else {
                    let max = plus1 - 1;
                    poc.max_long = Some(u8::try_from(max).unwrap_or(0));
                    for frame in dpb.iter_mut() {
                        if frame.long_term() && frame.long_idx() > i32::try_from(max).unwrap_or(0) {
                            frame.set_referenced(false);
                        }
                    }
                }
            }
            5 => {
                for frame in dpb.iter_mut() {
                    frame.set_referenced(false);
                }
                pic.set_frame_num(0);
                poc.prev_frame_num = 0;
                poc.prev_frame_num_offset = 0;
                poc.max_long = None;
            }
            6 => {
                let idx = i32::try_from(operation.long_term_frame_idx).unwrap_or(i32::MAX);
                let allowed = poc
                    .max_long
                    .is_some_and(|max| idx >= 0 && idx <= i32::from(max));
                if !allowed {
                    return Err(malformed);
                }
                for frame in dpb.iter_mut() {
                    if frame.long_term() && frame.long_idx() == idx {
                        frame.set_referenced(false);
                    }
                }
                pic.set_long(idx);
                current_long = true;
            }
            _ => return Err(malformed),
        }
    }
    if !current_long {
        pic.set_short();
    }
    let refs = dpb.iter().filter(|frame| frame.referenced()).count() + 1;
    if refs > pic.max_ref() {
        return Err(CodecError::Limit {
            what: "reference frames",
        });
    }
    Ok(())
}

fn short_at<T: Stored>(
    dpb: &[T],
    number: i32,
    current: u32,
    max_fn: u32,
    coded_only: bool,
) -> Option<usize> {
    dpb.iter().position(|frame| {
        frame.referenced()
            && !frame.long_term()
            && !(coded_only && frame.nonexisting())
            && pic_num(frame, current, max_fn) == number
    })
}

fn assign_long<T: Stored>(
    dpb: &mut [T],
    pos: usize,
    idx: i32,
    max_long: Option<u8>,
) -> Result<(), CodecError> {
    let allowed = max_long.is_some_and(|max| idx >= 0 && idx <= i32::from(max));
    if !allowed {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "mmco",
        });
    }
    for (index, frame) in dpb.iter_mut().enumerate() {
        if index != pos && frame.long_term() && frame.long_idx() == idx {
            frame.set_referenced(false);
        }
    }
    dpb[pos].set_long(idx);
    Ok(())
}

fn oldest_short<T: Stored>(
    dpb: &[T],
    current: u32,
    max_fn: u32,
    size: (usize, usize),
) -> Option<usize> {
    dpb.iter()
        .enumerate()
        .filter(|(_, frame)| frame.referenced() && !frame.long_term() && frame.size() == size)
        .min_by_key(|(_, frame)| pic_num(*frame, current, max_fn))
        .map(|(index, _)| index)
}

/// Clause 8.2.5.2. Each missing `frame_num` becomes a non-existing
/// short-term reference, made by `absent` from its `frame_num` and order
/// count, and can slide an older reference out. Those frames are not
/// displayed and are not put in reference lists.
pub(crate) fn fill_frame_num_gap<T: Stored>(
    sps: &Sps,
    state: &mut Poc,
    dpb: &mut Vec<T>,
    frame_num: u32,
    mut absent: impl FnMut(u32, i32) -> Result<T, CodecError>,
) -> Result<(), CodecError> {
    if !state.have_ref {
        return Ok(());
    }
    let max_fn = 1u32 << sps.log2_max_frame_num();
    if max_fn == 0 {
        return Ok(());
    }
    let mut unused = state.prev_frame_num.wrapping_add(1) % max_fn;
    if frame_num == state.prev_frame_num || frame_num == unused {
        return Ok(());
    }
    if !sps.gaps_in_frame_num_value_allowed_flag {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "frame_num gap",
        });
    }
    let mut steps = 0u32;
    while unused != frame_num {
        if steps >= max_fn {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "frame_num gap",
            });
        }
        steps += 1;
        let (poc, offset) = ghost_order(sps, state, unused)?;
        let ghost = absent(unused, poc)?;
        sliding_window(dpb, &ghost);
        state.prev_frame_num = unused;
        state.prev_frame_num_offset = offset;
        dpb.push(ghost);
        dpb.retain(|frame| frame.referenced() || frame.output());
        unused = unused.wrapping_add(1) % max_fn;
    }
    Ok(())
}

/// The pictures whose turn it is to be output while more than `keep` wait,
/// smallest order count first. Non-existing frames are never returned.
pub(crate) fn take_output<T: Stored>(dpb: &mut Vec<T>, keep: usize) -> Vec<T> {
    let mut out = Vec::new();
    loop {
        let pending: Vec<usize> = dpb
            .iter()
            .enumerate()
            .filter(|(_, frame)| frame.output())
            .map(|(index, _)| index)
            .collect();
        if pending.len() <= keep {
            break;
        }
        let pos = pending
            .into_iter()
            .min_by_key(|&index| dpb[index].poc())
            .unwrap_or(0);
        dpb[pos].set_output(false);
        if !dpb[pos].nonexisting() {
            out.push(dpb[pos].clone());
        }
    }
    dpb.retain(|frame| frame.referenced() || frame.output());
    out
}

/// What a slice says about its reference lists.
pub(crate) struct ListSyntax<'a> {
    /// `slice_type % 5`.
    pub kind: u8,
    pub frame_num: u32,
    /// Annex H `view_id`. Only pictures of the same view are temporal references.
    pub view_id: u16,
    /// `num_ref_idx_lX_active_minus1 + 1` of each list.
    pub active: [usize; 2],
    pub modifications: [&'a [RefPicListModification]; 2],
    /// Interview reference views appended to each list.
    pub interview: [&'a [u16]; 2],
}

/// Clauses 8.2.4.2 and 8.2.4.3: the frame reference lists of a P, SP or B
/// slice as indices into `dpb`, cut or padded to the active lengths. A slot
/// left as "no reference picture" is [`NO_PICTURE`]; such slots come after
/// every picture. Only short-term and long-term frames of `size` that are
/// not inferred for a gap are listed.
pub(crate) fn reference_lists<T: Stored>(
    dpb: &[T],
    syntax: &ListSyntax<'_>,
    poc: i32,
    max_fn: u32,
    size: (usize, usize),
) -> Result<(Vec<usize>, Vec<usize>), CodecError> {
    let view = syntax.view_id;
    let usable = |frame: &T| {
        frame.referenced()
            && !frame.nonexisting()
            && frame.view_id() == view
            && frame.size() == size
    };
    let mut shorts: Vec<usize> = dpb
        .iter()
        .enumerate()
        .filter(|(_, frame)| usable(frame) && !frame.long_term())
        .map(|(index, _)| index)
        .collect();
    let mut longs: Vec<usize> = dpb
        .iter()
        .enumerate()
        .filter(|(_, frame)| usable(frame) && frame.long_term())
        .map(|(index, _)| index)
        .collect();
    longs.sort_by_key(|&index| dpb[index].long_idx());
    if intra_slice(syntax.kind) {
        return Ok((Vec::new(), Vec::new()));
    }
    let [active0, active1] = syntax.active;
    let [mods0, mods1] = syntax.modifications;
    if crate::header::syntax_kind(syntax.kind) == 0 {
        shorts.sort_by_key(|&index| {
            std::cmp::Reverse(pic_num(&dpb[index], syntax.frame_num, max_fn))
        });
        let mut list = shorts;
        list.extend(longs);
        append_interview(&mut list, dpb, poc, syntax.interview[0], size);
        let list = modify(list, active0, dpb, mods0, syntax.frame_num, max_fn)?;
        return Ok((list, Vec::new()));
    }
    let mut past: Vec<usize> = shorts
        .iter()
        .copied()
        .filter(|&index| dpb[index].poc() < poc)
        .collect();
    let mut future: Vec<usize> = shorts
        .iter()
        .copied()
        .filter(|&index| dpb[index].poc() > poc)
        .collect();
    past.sort_by_key(|&index| std::cmp::Reverse(dpb[index].poc()));
    future.sort_by_key(|&index| dpb[index].poc());
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
    append_interview(&mut list0, dpb, poc, syntax.interview[0], size);
    append_interview(&mut list1, dpb, poc, syntax.interview[1], size);
    let list0 = modify(list0, active0, dpb, mods0, syntax.frame_num, max_fn)?;
    let list1 = modify(list1, active1, dpb, mods1, syntax.frame_num, max_fn)?;
    Ok((list0, list1))
}

fn append_interview<T: Stored>(
    list: &mut Vec<usize>,
    dpb: &[T],
    poc: i32,
    views: &[u16],
    size: (usize, usize),
) {
    for view in views {
        let Some(index) = dpb.iter().position(|frame| {
            frame.referenced()
                && !frame.nonexisting()
                && frame.inter_view()
                && frame.view_id() == *view
                && frame.poc() == poc
                && frame.size() == size
        }) else {
            continue;
        };
        if !list.contains(&index) {
            list.push(index);
        }
    }
}

fn list_error() -> CodecError {
    CodecError::Malformed {
        at_bit: 0,
        what: "ref list modification",
    }
}

/// Clause 8.2.4.2 pads a short initial list, then 8.2.4.3 may repeat a picture
/// so the active length is filled. A slot left as "no reference picture" stays
/// [`NO_PICTURE`] and is rejected when a macroblock uses it.
fn modify<T: Stored>(
    mut list: Vec<usize>,
    active: usize,
    dpb: &[T],
    mods: &[RefPicListModification],
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
    let max_pic = i32::try_from(max_fn).map_err(|_| list_error())?;
    let curr = i32::try_from(current).map_err(|_| list_error())?;
    list.push(NO_PICTURE);
    let mut pred = curr;
    let mut ref_idx = 0usize;
    for modification in mods {
        if ref_idx >= active {
            return Err(list_error());
        }
        match modification.modification_of_pic_nums_idc {
            0 | 1 => {
                let number = short_target(&mut pred, modification, curr, max_pic)?;
                let Some(picture) = short_index(dpb, number, current, max_fn) else {
                    return Err(list_error());
                };
                let nums = ShortNums {
                    current,
                    max_fn,
                    max_pic,
                };
                place(&mut list, ref_idx, active, picture, |slot| {
                    short_num(dpb, slot, &nums) != number
                });
                ref_idx += 1;
            }
            2 => {
                let long_num = modification.long_term_pic_num;
                let Some(picture) = long_index(dpb, long_num) else {
                    return Err(list_error());
                };
                place(&mut list, ref_idx, active, picture, |slot| {
                    !same_long(dpb, slot, long_num)
                });
                ref_idx += 1;
            }
            _ => return Err(list_error()),
        }
    }
    list.truncate(active);
    Ok(list)
}

fn short_target(
    pred: &mut i32,
    modification: &RefPicListModification,
    curr: i32,
    max_pic: i32,
) -> Result<i32, CodecError> {
    let Some(sum) = modification.abs_diff_pic_num_minus1.checked_add(1) else {
        return Err(list_error());
    };
    let Ok(abs) = i32::try_from(sum) else {
        return Err(list_error());
    };
    if abs <= 0 || abs > max_pic || max_pic <= 0 {
        return Err(list_error());
    }
    let no_wrap = if modification.modification_of_pic_nums_idc == 0 {
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

/// Puts `picture` at `ref_idx`, moving the rest down, and keeps after it
/// only the entries `keep` passes: the same picture is not listed twice.
fn place(
    list: &mut [usize],
    ref_idx: usize,
    active: usize,
    picture: usize,
    keep: impl Fn(usize) -> bool,
) {
    for c in (ref_idx + 1..=active).rev() {
        list[c] = list[c - 1];
    }
    list[ref_idx] = picture;
    let mut nidx = ref_idx + 1;
    for c in (ref_idx + 1)..=active {
        if keep(list[c]) {
            list[nidx] = list[c];
            nidx += 1;
        }
    }
}

fn short_index<T: Stored>(dpb: &[T], number: i32, current: u32, max_fn: u32) -> Option<usize> {
    dpb.iter().position(|frame| {
        frame.referenced()
            && !frame.long_term()
            && !frame.nonexisting()
            && pic_num(frame, current, max_fn) == number
    })
}

fn long_index<T: Stored>(dpb: &[T], long_num: u32) -> Option<usize> {
    let Ok(long_num) = i32::try_from(long_num) else {
        return None;
    };
    dpb.iter().position(|frame| {
        frame.referenced()
            && frame.long_term()
            && !frame.nonexisting()
            && frame.long_idx() == long_num
    })
}

/// `PicNumF`: a short-term picture's `PicNum`, otherwise `MaxPicNum`.
fn short_num<T: Stored>(dpb: &[T], slot: usize, nums: &ShortNums) -> i32 {
    let Some(frame) = dpb.get(slot) else {
        return nums.max_pic;
    };
    if !frame.referenced() || frame.long_term() {
        return nums.max_pic;
    }
    pic_num(frame, nums.current, nums.max_fn)
}

fn same_long<T: Stored>(dpb: &[T], slot: usize, long_num: u32) -> bool {
    let Ok(long_num) = i32::try_from(long_num) else {
        return false;
    };
    dpb.get(slot).is_some_and(|frame| {
        frame.referenced()
            && frame.long_term()
            && !frame.nonexisting()
            && frame.long_idx() == long_num
    })
}

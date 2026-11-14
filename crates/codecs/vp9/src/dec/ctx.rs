//! Contexts of the mode info syntax elements from the blocks above and
//! left (libvpx `vp9_pred_common.c`).

use crate::geom::{ALTREF_FRAME, GOLDEN_FRAME, INTRA_FRAME, LAST_FRAME};

use super::frame::Mv;

const SENTINEL_FILTER: u8 = 3;

/// What a block leaves for the blocks after it to read.
#[derive(Clone, Copy, Debug)]
pub struct Nb {
    pub bsize: u8,
    /// The luma mode (for a block under 8×8, the last sub-block's).
    pub mode: u8,
    /// Sub-block luma modes, in raster order; all `mode` for 8×8 and up.
    pub bmode: [u8; 4],
    pub skip: bool,
    pub tx: u8,
    pub seg_pred: bool,
    /// 3 for intra blocks, which counts as no filter in the context.
    pub interp: u8,
    pub refs: [u8; 2],
    /// Sub-block vectors, raster order; all the block's for 8×8 and up.
    pub bmv: [[Mv; 2]; 4],
}

impl Default for Nb {
    fn default() -> Self {
        Self {
            bsize: 3,
            mode: 0,
            bmode: [0; 4],
            skip: false,
            tx: 0,
            seg_pred: false,
            interp: 3,
            refs: [0; 2],
            bmv: [[Mv::ZERO; 2]; 4],
        }
    }
}


pub(crate) fn has_second(mi: &Nb) -> bool {
    mi.refs[1] > INTRA_FRAME
}

pub(crate) fn is_inter(mi: &Nb) -> bool {
    mi.refs[0] > INTRA_FRAME
}

pub(crate) fn intra_inter_ctx(above: Option<&Nb>, left: Option<&Nb>) -> usize {
    match (above, left) {
        (Some(above), Some(left)) => {
            let above_intra = !is_inter(above);
            let left_intra = !is_inter(left);
            if left_intra && above_intra {
                3
            } else {
                usize::from(left_intra || above_intra)
            }
        }
        (Some(edge), None) | (None, Some(edge)) => 2 * usize::from(!is_inter(edge)),
        (None, None) => 0,
    }
}

pub(crate) fn interp_ctx(left: Option<u8>, above: Option<u8>) -> usize {
    let left = left.unwrap_or(SENTINEL_FILTER);
    let above = above.unwrap_or(SENTINEL_FILTER);
    if left == above || left == SENTINEL_FILTER {
        usize::from(if left == above { left } else { above })
    } else if above == SENTINEL_FILTER {
        usize::from(left)
    } else {
        usize::from(SENTINEL_FILTER)
    }
}

pub(crate) fn reference_mode_ctx(above: Option<&Nb>, left: Option<&Nb>, fixed: u8) -> usize {
    match (above, left) {
        (Some(above), Some(left)) => both_ref_mode(above, left, fixed),
        (Some(edge), None) | (None, Some(edge)) => one_ref_mode(edge, fixed),
        (None, None) => 1,
    }
}

pub(crate) fn both_ref_mode(above: &Nb, left: &Nb, fixed: u8) -> usize {
    if !has_second(above) && !has_second(left) {
        usize::from((above.refs[0] == fixed) ^ (left.refs[0] == fixed))
    } else if !has_second(above) {
        2 + usize::from(above.refs[0] == fixed || !is_inter(above))
    } else if !has_second(left) {
        2 + usize::from(left.refs[0] == fixed || !is_inter(left))
    } else {
        4
    }
}

pub(crate) fn one_ref_mode(edge: &Nb, fixed: u8) -> usize {
    if has_second(edge) {
        3
    } else {
        usize::from(edge.refs[0] == fixed)
    }
}

pub(crate) fn single_p1(above: Option<&Nb>, left: Option<&Nb>) -> usize {
    match (above, left) {
        (Some(above), Some(left)) => single_p1_both(above, left),
        (Some(edge), None) | (None, Some(edge)) => single_one_last(edge),
        (None, None) => 2,
    }
}

pub(crate) fn single_p1_both(above: &Nb, left: &Nb) -> usize {
    let above_intra = !is_inter(above);
    let left_intra = !is_inter(left);
    if above_intra && left_intra {
        2
    } else if above_intra || left_intra {
        single_one_last(if above_intra { left } else { above })
    } else {
        single_p1_inter(above, left)
    }
}

pub(crate) fn single_one_last(edge: &Nb) -> usize {
    if !is_inter(edge) {
        return 2;
    }
    if !has_second(edge) {
        4 * usize::from(edge.refs[0] == LAST_FRAME)
    } else {
        1 + usize::from(edge.refs[0] == LAST_FRAME || edge.refs[1] == LAST_FRAME)
    }
}

pub(crate) fn single_p1_inter(above: &Nb, left: &Nb) -> usize {
    let above_second = has_second(above);
    let left_second = has_second(left);
    if above_second && left_second {
        return 1 + usize::from(mentions(above, LAST_FRAME) || mentions(left, LAST_FRAME));
    }
    if above_second || left_second {
        let single = if above_second { left } else { above };
        let compound = if above_second { above } else { left };
        if single.refs[0] == LAST_FRAME {
            3 + usize::from(mentions(compound, LAST_FRAME))
        } else {
            usize::from(mentions(compound, LAST_FRAME))
        }
    } else {
        2 * usize::from(above.refs[0] == LAST_FRAME) + 2 * usize::from(left.refs[0] == LAST_FRAME)
    }
}

pub(crate) fn single_p2(above: Option<&Nb>, left: Option<&Nb>) -> usize {
    match (above, left) {
        (Some(above), Some(left)) => single_p2_both(above, left),
        (Some(edge), None) | (None, Some(edge)) => single_p2_one(edge),
        (None, None) => 2,
    }
}

pub(crate) fn single_p2_one(edge: &Nb) -> usize {
    if !is_inter(edge) || (edge.refs[0] == LAST_FRAME && !has_second(edge)) {
        2
    } else if !has_second(edge) {
        4 * usize::from(edge.refs[0] == GOLDEN_FRAME)
    } else {
        3 * usize::from(mentions(edge, GOLDEN_FRAME))
    }
}

pub(crate) fn single_p2_both(above: &Nb, left: &Nb) -> usize {
    let above_intra = !is_inter(above);
    let left_intra = !is_inter(left);
    if above_intra && left_intra {
        2
    } else if above_intra || left_intra {
        single_p2_mixed(if above_intra { left } else { above })
    } else {
        single_p2_inter(above, left)
    }
}

pub(crate) fn single_p2_mixed(edge: &Nb) -> usize {
    if !has_second(edge) {
        if edge.refs[0] == LAST_FRAME {
            3
        } else {
            4 * usize::from(edge.refs[0] == GOLDEN_FRAME)
        }
    } else {
        1 + 2 * usize::from(mentions(edge, GOLDEN_FRAME))
    }
}

pub(crate) fn single_p2_inter(above: &Nb, left: &Nb) -> usize {
    let above_second = has_second(above);
    let left_second = has_second(left);
    if above_second && left_second {
        if above.refs == left.refs {
            3 * usize::from(mentions(above, GOLDEN_FRAME) || mentions(left, GOLDEN_FRAME))
        } else {
            2
        }
    } else if above_second || left_second {
        p2_mixed_refs(above, left, above_second)
    } else {
        p2_two_singles(above, left)
    }
}

pub(crate) fn p2_mixed_refs(above: &Nb, left: &Nb, above_second: bool) -> usize {
    let single = if above_second { left } else { above };
    let compound = if above_second { above } else { left };
    if single.refs[0] == GOLDEN_FRAME {
        3 + usize::from(mentions(compound, GOLDEN_FRAME))
    } else if single.refs[0] == ALTREF_FRAME {
        usize::from(mentions(compound, GOLDEN_FRAME))
    } else {
        1 + 2 * usize::from(mentions(compound, GOLDEN_FRAME))
    }
}

pub(crate) fn p2_two_singles(above: &Nb, left: &Nb) -> usize {
    if above.refs[0] == LAST_FRAME && left.refs[0] == LAST_FRAME {
        3
    } else if above.refs[0] == LAST_FRAME || left.refs[0] == LAST_FRAME {
        let other = if above.refs[0] == LAST_FRAME {
            left.refs[0]
        } else {
            above.refs[0]
        };
        4 * usize::from(other == GOLDEN_FRAME)
    } else {
        2 * usize::from(above.refs[0] == GOLDEN_FRAME)
            + 2 * usize::from(left.refs[0] == GOLDEN_FRAME)
    }
}

pub(crate) fn comp_ref_ctx(above: Option<&Nb>, left: Option<&Nb>, header: &crate::header::Header) -> usize {
    let fixed = header.comp_fixed;
    let var0 = header.comp_var[0];
    let var1 = header.comp_var[1];
    let var_at = usize::from(header.sign_bias[usize::from(fixed)] == 0);
    match (above, left) {
        (Some(above), Some(left)) => comp_both(above, left, fixed, var0, var1, var_at),
        (Some(edge), None) | (None, Some(edge)) => comp_one(edge, var1, var_at),
        (None, None) => 2,
    }
}

pub(crate) fn comp_one(edge: &Nb, var: u8, var_at: usize) -> usize {
    if !is_inter(edge) {
        2
    } else if has_second(edge) {
        4 * usize::from(edge.refs[var_at] != var)
    } else {
        3 * usize::from(edge.refs[0] != var)
    }
}

pub(crate) fn comp_both(above: &Nb, left: &Nb, fixed: u8, var0: u8, var1: u8, var_at: usize) -> usize {
    let above_intra = !is_inter(above);
    let left_intra = !is_inter(left);
    if above_intra && left_intra {
        2
    } else if above_intra || left_intra {
        comp_one_edge(if above_intra { left } else { above }, var1, var_at)
    } else {
        comp_inter(above, left, fixed, var0, var1, var_at)
    }
}

pub(crate) fn comp_one_edge(edge: &Nb, var: u8, var_at: usize) -> usize {
    let seen = if has_second(edge) {
        edge.refs[var_at]
    } else {
        edge.refs[0]
    };
    1 + 2 * usize::from(seen != var)
}

pub(crate) fn comp_inter(above: &Nb, left: &Nb, fixed: u8, var0: u8, var1: u8, var_at: usize) -> usize {
    let above_single = !has_second(above);
    let left_single = !has_second(left);
    let above_ref = if above_single {
        above.refs[0]
    } else {
        above.refs[var_at]
    };
    let left_ref = if left_single {
        left.refs[0]
    } else {
        left.refs[var_at]
    };
    if above_ref == left_ref && var1 == above_ref {
        0
    } else if left_single && above_single {
        comp_two_singles(above_ref, left_ref, fixed, var0)
    } else if left_single || above_single {
        comp_single_and_compound(above_ref, left_ref, left_single, var1)
    } else if above_ref == left_ref {
        4
    } else {
        2
    }
}

pub(crate) fn comp_two_singles(above_ref: u8, left_ref: u8, fixed: u8, var0: u8) -> usize {
    let crossed =
        (above_ref == fixed && left_ref == var0) || (left_ref == fixed && above_ref == var0);
    if crossed {
        4
    } else if above_ref == left_ref {
        3
    } else {
        1
    }
}

pub(crate) fn comp_single_and_compound(above_ref: u8, left_ref: u8, left_single: bool, var: u8) -> usize {
    let compound = if left_single { above_ref } else { left_ref };
    let single = if left_single { left_ref } else { above_ref };
    if compound == var && single != var {
        1
    } else if single == var && compound != var {
        2
    } else {
        4
    }
}

pub(crate) fn mentions(mi: &Nb, frame: u8) -> bool {
    mi.refs[0] == frame || mi.refs[1] == frame
}


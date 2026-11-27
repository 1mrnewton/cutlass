//! The slice header, clause 7.3.3, up to and including
//! `slice_group_change_cycle`: what an Annex A slice codes, and what a
//! scalable, multiview or 3D slice codes the same way before its own
//! extension fields. The software decoder and [`crate::stateless`] read it
//! with [`read_start`] and [`read_rest`].
//!
//! The public fields are the syntax elements under their names in the
//! Recommendation. An element the slice does not code holds the value
//! clause 7.4.3 infers for it, or the default named on the field.

use cutlass_decode::CodecError;

use crate::bits::Bits;
use crate::set::{MAX_PPS, Pps, Sps};

/// One slice header (clause 7.3.3) of an Annex A slice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceHeader {
    /// `nal_ref_idc` of the NAL unit the slice is in.
    pub nal_ref_idc: u8,
    /// `IdrPicFlag`: the NAL unit is type 5.
    pub idr_pic_flag: bool,
    /// `first_mb_in_slice`.
    pub first_mb_in_slice: u32,
    /// `slice_type` as coded, 0 to 9. `slice_type % 5` is 0 P, 1 B, 2 I,
    /// 3 SP, 4 SI ([`SliceHeader::slice_kind`]); 5 to 9 say every slice of
    /// the picture has that type.
    pub slice_type: u8,
    /// `pic_parameter_set_id`.
    pub pic_parameter_set_id: u8,
    /// `colour_plane_id`. 0 unless `separate_colour_plane_flag`.
    pub colour_plane_id: u8,
    /// `frame_num`.
    pub frame_num: u32,
    /// `field_pic_flag`.
    pub field_pic_flag: bool,
    /// `bottom_field_flag`.
    pub bottom_field_flag: bool,
    /// `idr_pic_id`. 0 for a slice that is not IDR.
    pub idr_pic_id: u32,
    /// `pic_order_cnt_lsb`. 0 unless `pic_order_cnt_type` is 0.
    pub pic_order_cnt_lsb: u32,
    /// `delta_pic_order_cnt_bottom`. 0 when absent.
    pub delta_pic_order_cnt_bottom: i32,
    /// `delta_pic_order_cnt[0]` and `[1]`. 0 when absent.
    pub delta_pic_order_cnt: [i32; 2],
    /// `redundant_pic_cnt`. 0 when absent. A slice above 0 belongs to a
    /// redundant picture, which the decoders skip.
    pub redundant_pic_cnt: u32,
    /// `direct_spatial_mv_pred_flag`. Coded in B slices only; false elsewhere.
    pub direct_spatial_mv_pred_flag: bool,
    /// `num_ref_idx_active_override_flag`. False when absent.
    pub num_ref_idx_active_override_flag: bool,
    /// `num_ref_idx_l0_active_minus1`: the coded value with the override
    /// flag, else the PPS default. Not used by I and SI slices.
    pub num_ref_idx_l0_active_minus1: u8,
    /// `num_ref_idx_l1_active_minus1`, as `num_ref_idx_l0_active_minus1`.
    /// Used by B slices only.
    pub num_ref_idx_l1_active_minus1: u8,
    /// `ref_pic_list_modification()` for list 0, in coded order without the
    /// closing `modification_of_pic_nums_idc` 3. Empty when the flag is 0
    /// or the list closes at once.
    pub ref_pic_list_modification_l0: Vec<RefPicListModification>,
    /// The same for list 1. B slices only.
    pub ref_pic_list_modification_l1: Vec<RefPicListModification>,
    /// `pred_weight_table()`, present when `weighted_pred_flag` is 1 for a P
    /// or SP slice or `weighted_bipred_idc` is 1 for a B slice.
    pub pred_weight_table: Option<PredWeightTable>,
    /// `dec_ref_pic_marking()`. Every flag false and no operation when
    /// `nal_ref_idc` is 0.
    pub dec_ref_pic_marking: DecRefPicMarking,
    /// `cabac_init_idc`. 0 when absent.
    pub cabac_init_idc: u8,
    /// `slice_qp_delta`. SliceQPY is `26 + pic_init_qp_minus26 + slice_qp_delta`.
    pub slice_qp_delta: i32,
    /// `sp_for_switch_flag`. SP slices only.
    pub sp_for_switch_flag: bool,
    /// `slice_qs_delta`. SP and SI slices only.
    pub slice_qs_delta: i32,
    /// `disable_deblocking_filter_idc`. 0 when absent.
    pub disable_deblocking_filter_idc: u8,
    /// `slice_alpha_c0_offset_div2`. 0 when absent.
    pub slice_alpha_c0_offset_div2: i32,
    /// `slice_beta_offset_div2`. 0 when absent.
    pub slice_beta_offset_div2: i32,
    /// `slice_group_change_cycle`. 0 when absent.
    pub slice_group_change_cycle: u32,
}

/// One `modification_of_pic_nums_idc` and the value that follows it
/// (clause 7.3.3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefPicListModification {
    /// `modification_of_pic_nums_idc`: 0 subtracts and 1 adds
    /// `abs_diff_pic_num_minus1 + 1` to the predicted picture number; 2
    /// names a long-term picture.
    pub modification_of_pic_nums_idc: u8,
    /// `abs_diff_pic_num_minus1` for idc 0 and 1. Otherwise 0.
    pub abs_diff_pic_num_minus1: u32,
    /// `long_term_pic_num` for idc 2. Otherwise 0.
    pub long_term_pic_num: u32,
}

/// `pred_weight_table()` (clause 7.3.3.2) with every entry filled: an entry
/// whose flag is 0 holds the weight `2^denom` and offset 0 that clause
/// 7.4.3.2 infers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PredWeightTable {
    /// `luma_log2_weight_denom`, 0 to 7.
    pub luma_log2_weight_denom: u8,
    /// `chroma_log2_weight_denom`, 0 to 7.
    pub chroma_log2_weight_denom: u8,
    /// One entry for each of the `num_ref_idx_l0_active_minus1 + 1`
    /// list 0 indices.
    pub l0: Vec<PredWeight>,
    /// One entry for each list 1 index of a B slice. Empty otherwise.
    pub l1: Vec<PredWeight>,
}

/// The weights of one reference index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PredWeight {
    /// `luma_weight_lX_flag`.
    pub luma_weight_flag: bool,
    /// `luma_weight_lX[i]`, or `2^luma_log2_weight_denom` when the flag is 0.
    pub luma_weight: i32,
    /// `luma_offset_lX[i]` as coded (not scaled by the bit depth), or 0.
    pub luma_offset: i32,
    /// `chroma_weight_lX_flag`.
    pub chroma_weight_flag: bool,
    /// `chroma_weight_lX[i][j]` for Cb and Cr, or `2^chroma_log2_weight_denom`.
    pub chroma_weight: [i32; 2],
    /// `chroma_offset_lX[i][j]` for Cb and Cr as coded, or 0.
    pub chroma_offset: [i32; 2],
}

/// `dec_ref_pic_marking()` (clause 7.3.3.3).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DecRefPicMarking {
    /// `no_output_of_prior_pics_flag`. IDR pictures only.
    pub no_output_of_prior_pics_flag: bool,
    /// `long_term_reference_flag`. IDR pictures only.
    pub long_term_reference_flag: bool,
    /// `adaptive_ref_pic_marking_mode_flag`. Pictures that are not IDR only.
    pub adaptive_ref_pic_marking_mode_flag: bool,
    /// The operations in coded order, without the closing operation 0.
    pub operations: Vec<MemoryManagementControlOperation>,
}

/// One `memory_management_control_operation` and its values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryManagementControlOperation {
    /// `memory_management_control_operation`, 1 to 6.
    pub memory_management_control_operation: u8,
    /// `difference_of_pic_nums_minus1` for operations 1 and 3. Otherwise 0.
    pub difference_of_pic_nums_minus1: u32,
    /// `long_term_pic_num` for operation 2. Otherwise 0.
    pub long_term_pic_num: u32,
    /// `long_term_frame_idx` for operations 3 and 6. Otherwise 0.
    pub long_term_frame_idx: u32,
    /// `max_long_term_frame_idx_plus1` for operation 4. Otherwise 0.
    pub max_long_term_frame_idx_plus1: u32,
}

impl SliceHeader {
    /// `slice_type % 5`: 0 P, 1 B, 2 I, 3 SP, 4 SI.
    pub fn slice_kind(&self) -> u8 {
        self.slice_type % 5
    }

    /// `slice_qp_delta` added to the PPS start: SliceQPY.
    pub(crate) fn qp(&self, pps: &Pps) -> i32 {
        pps.init_qp() + self.slice_qp_delta
    }

    /// The elements picture order counts are worked out from.
    pub(crate) fn order(&self, sps: &Sps) -> crate::poc::OrderSyntax {
        crate::poc::OrderSyntax {
            frame_num: self.frame_num,
            nal_ref: self.nal_ref_idc != 0,
            field: self.field_pic_flag,
            poc_lsb: self.pic_order_cnt_lsb as i32,
            delta_poc: delta_poc(self, sps),
        }
    }

    /// The active list lengths the decoder builds: none for an I or SI
    /// slice, list 0 only for P and SP.
    pub(crate) fn active(&self) -> [usize; 2] {
        let kind = self.slice_kind();
        if intra_slice(kind) {
            return [0, 0];
        }
        let l0 = usize::from(self.num_ref_idx_l0_active_minus1) + 1;
        let l1 = usize::from(self.num_ref_idx_l1_active_minus1) + 1;
        [l0, l1]
    }
}

/// The two order-count differences the way the decoder keeps them: for
/// type 0 the second is `delta_pic_order_cnt_bottom`; for type 1 they are
/// `delta_pic_order_cnt`.
pub(crate) fn delta_poc(header: &SliceHeader, sps: &Sps) -> [i32; 2] {
    if sps.pic_order_cnt_type == 0 {
        [0, header.delta_pic_order_cnt_bottom]
    } else {
        header.delta_pic_order_cnt
    }
}

/// I and SI slices predict nothing from other pictures.
pub(crate) fn intra_slice(kind: u8) -> bool {
    matches!(kind, 2 | 4)
}

/// SP uses P syntax. SI uses I syntax. The raw kind stays on the slice for clause 8.6.
pub(crate) fn syntax_kind(kind: u8) -> u8 {
    match kind {
        3 => 0,
        4 => 2,
        other => other,
    }
}

/// `first_mb_in_slice`, `slice_type` and `pic_parameter_set_id`: what
/// finds the parameter sets the rest of the header is read with.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SliceStart {
    pub first_mb_in_slice: u32,
    pub slice_type: u8,
    pub pic_parameter_set_id: u8,
}

impl SliceStart {
    /// `slice_type % 5`.
    pub(crate) fn kind(&self) -> u8 {
        self.slice_type % 5
    }
}

pub(crate) fn read_start(bits: &mut Bits<'_>) -> Result<SliceStart, CodecError> {
    let first_mb_in_slice = bits.ue()?;
    let slice_type = bits.ue()?;
    let pps_id = bits.ue()?;
    if slice_type > 9 {
        return Err(bits.malformed("slice type"));
    }
    if pps_id >= MAX_PPS as u32 {
        return Err(bits.malformed("pps id"));
    }
    Ok(SliceStart {
        first_mb_in_slice,
        slice_type: slice_type as u8,
        pic_parameter_set_id: pps_id as u8,
    })
}

/// What the NAL unit, and a slice extension's header, say about how the
/// rest of the slice header is coded.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Context {
    /// `IdrPicFlag`.
    pub idr: bool,
    pub nal_ref_idc: u8,
    /// `quality_id` is 0, as it is for every slice that is not scalable. A
    /// quality enhancement leaves the reference lists, the weights and the
    /// marking to its base.
    pub base_quality: bool,
    /// A scalable slice whose SPS has `slice_header_restriction_flag` 0
    /// codes `store_ref_base_pic_flag` after the marking.
    pub store_ref_base: bool,
}

impl Context {
    /// An Annex A slice in a NAL unit with this header.
    pub(crate) fn annex_a(idr: bool, nal_ref_idc: u8) -> Self {
        Self {
            idr,
            nal_ref_idc,
            base_quality: true,
            store_ref_base: false,
        }
    }
}

/// The rest of the slice header after [`read_start`], up to and including
/// `slice_group_change_cycle`. `bits` is left on the next element.
pub(crate) fn read_rest(
    bits: &mut Bits<'_>,
    start: SliceStart,
    sps: &Sps,
    pps: &Pps,
    cx: Context,
) -> Result<SliceHeader, CodecError> {
    let kind = start.kind();
    let nal_ref = cx.nal_ref_idc != 0;
    let mut colour_plane_id = 0u8;
    if sps.separate_colour_plane_flag {
        let id = bits.u(2)?;
        if id > 2 {
            return Err(bits.malformed("colour plane"));
        }
        colour_plane_id = id as u8;
    }
    let frame_num = bits.u(sps.log2_max_frame_num())?;
    let mut field_pic_flag = false;
    let mut bottom_field_flag = false;
    if !sps.frame_mbs_only_flag {
        field_pic_flag = bits.bit()?;
        if field_pic_flag {
            bottom_field_flag = bits.bit()?;
        }
    }
    let mut idr_pic_id = 0;
    if cx.idr {
        idr_pic_id = bits.ue()?;
    }
    let mut pic_order_cnt_lsb = 0u32;
    let mut delta_pic_order_cnt_bottom = 0i32;
    let mut delta_pic_order_cnt = [0i32; 2];
    let bottom_present = pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag;
    if sps.pic_order_cnt_type == 0 {
        pic_order_cnt_lsb = bits.u(sps.log2_max_pic_order_cnt_lsb())?;
        if bottom_present {
            delta_pic_order_cnt_bottom = bits.se()?;
        }
    } else if sps.pic_order_cnt_type == 1 && !sps.delta_pic_order_always_zero_flag {
        delta_pic_order_cnt[0] = bits.se()?;
        if bottom_present {
            delta_pic_order_cnt[1] = bits.se()?;
        }
    }
    let mut redundant_pic_cnt = 0u32;
    if pps.redundant_pic_cnt_present_flag {
        redundant_pic_cnt = bits.ue()?;
        if redundant_pic_cnt > 127 {
            return Err(bits.malformed("redundant_pic_cnt"));
        }
    }
    let mut direct_spatial_mv_pred_flag = false;
    if kind == 1 {
        direct_spatial_mv_pred_flag = bits.bit()?;
    }
    let mut override_flag = false;
    let mut l0_minus1 = u32::from(pps.num_ref_idx_l0_default_active_minus1);
    let mut l1_minus1 = u32::from(pps.num_ref_idx_l1_default_active_minus1);
    let predicted = !intra_slice(kind) && cx.base_quality;
    if predicted {
        override_flag = bits.bit()?;
        if override_flag {
            l0_minus1 = bits.ue()?;
            if kind == 1 {
                l1_minus1 = bits.ue()?;
            }
        }
        if l0_minus1 >= 16 || l1_minus1 >= 16 {
            return Err(CodecError::Limit {
                what: "reference index",
            });
        }
    }
    let (l0_mods, l1_mods) = if predicted {
        (
            read_list_mod(bits)?,
            if kind == 1 {
                read_list_mod(bits)?
            } else {
                Vec::new()
            },
        )
    } else {
        (Vec::new(), Vec::new())
    };
    let mut pred_weight_table = None;
    if cx.base_quality
        && ((pps.weighted_pred_flag && syntax_kind(kind) == 0)
            || (pps.weighted_bipred_idc == 1 && kind == 1))
    {
        let luma_denom = bits.ue()?;
        let chroma_denom = bits.ue()?;
        if luma_denom > 7 || chroma_denom > 7 {
            return Err(bits.malformed("weight denom"));
        }
        let denom = (luma_denom as u8, chroma_denom as u8);
        let l0 = read_weights(bits, l0_minus1 as usize + 1, denom)?;
        let l1 = if kind == 1 {
            read_weights(bits, l1_minus1 as usize + 1, denom)?
        } else {
            Vec::new()
        };
        pred_weight_table = Some(PredWeightTable {
            luma_log2_weight_denom: denom.0,
            chroma_log2_weight_denom: denom.1,
            l0,
            l1,
        });
    }
    let mut marking = DecRefPicMarking::default();
    if nal_ref && cx.base_quality {
        if cx.idr {
            marking.no_output_of_prior_pics_flag = bits.bit()?;
            marking.long_term_reference_flag = bits.bit()?;
        } else if bits.bit()? {
            marking.adaptive_ref_pic_marking_mode_flag = true;
            marking.operations = read_mmco(bits)?;
        }
        if cx.store_ref_base && bits.bit()? {
            return Err(CodecError::Unsupported {
                what: "reference base picture",
            });
        }
    }
    let mut cabac_init_idc = 0u8;
    if pps.entropy_coding_mode_flag && !intra_slice(kind) {
        let idc = bits.ue()?;
        if idc > 2 {
            return Err(bits.malformed("cabac_init_idc"));
        }
        cabac_init_idc = idc as u8;
    }
    let slice_qp_delta = bits.se()?;
    let qp = pps.init_qp().saturating_add(slice_qp_delta);
    let qp_bd = 6 * i32::from(sps.bit_depth.saturating_sub(8));
    if !(-qp_bd..=51).contains(&qp) {
        return Err(bits.malformed("slice qp"));
    }
    let mut sp_for_switch_flag = false;
    let mut slice_qs_delta = 0i32;
    if matches!(kind, 3 | 4) {
        if kind == 3 {
            sp_for_switch_flag = bits.bit()?;
        }
        slice_qs_delta = bits.se()?;
        let qs = pps.init_qs().saturating_add(slice_qs_delta);
        if !(0..=51).contains(&qs) {
            return Err(bits.malformed("slice qs"));
        }
    }
    let mut disable_deblocking_filter_idc = 0u8;
    let mut slice_alpha_c0_offset_div2 = 0i32;
    let mut slice_beta_offset_div2 = 0i32;
    if pps.deblocking_filter_control_present_flag {
        let idc = bits.ue()?;
        if idc > 2 {
            return Err(bits.malformed("deblock idc"));
        }
        disable_deblocking_filter_idc = idc as u8;
        if idc != 1 {
            slice_alpha_c0_offset_div2 = bits.se()?;
            slice_beta_offset_div2 = bits.se()?;
            if !(-6..=6).contains(&slice_alpha_c0_offset_div2)
                || !(-6..=6).contains(&slice_beta_offset_div2)
            {
                return Err(bits.malformed("deblock offset"));
            }
        }
    }
    let mut slice_group_change_cycle = 0u32;
    if cx.base_quality && pps.groups.minus1 > 0 && (3..=5).contains(&pps.groups.map_type) {
        let units = crate::fmo::map_units(sps.mb_width, sps.mb_height, sps.frame_mbs_only_flag);
        let width = crate::fmo::cycle_bits(units, pps.groups.rate)?;
        slice_group_change_cycle = bits.u(width)?;
    }
    Ok(SliceHeader {
        nal_ref_idc: cx.nal_ref_idc,
        idr_pic_flag: cx.idr,
        first_mb_in_slice: start.first_mb_in_slice,
        slice_type: start.slice_type,
        pic_parameter_set_id: start.pic_parameter_set_id,
        colour_plane_id,
        frame_num,
        field_pic_flag,
        bottom_field_flag,
        idr_pic_id,
        pic_order_cnt_lsb,
        delta_pic_order_cnt_bottom,
        delta_pic_order_cnt,
        redundant_pic_cnt,
        direct_spatial_mv_pred_flag,
        num_ref_idx_active_override_flag: override_flag,
        num_ref_idx_l0_active_minus1: l0_minus1 as u8,
        num_ref_idx_l1_active_minus1: l1_minus1 as u8,
        ref_pic_list_modification_l0: l0_mods,
        ref_pic_list_modification_l1: l1_mods,
        pred_weight_table,
        dec_ref_pic_marking: marking,
        cabac_init_idc,
        slice_qp_delta,
        sp_for_switch_flag,
        slice_qs_delta,
        disable_deblocking_filter_idc,
        slice_alpha_c0_offset_div2,
        slice_beta_offset_div2,
        slice_group_change_cycle,
    })
}

fn read_list_mod(bits: &mut Bits<'_>) -> Result<Vec<RefPicListModification>, CodecError> {
    let mut mods = Vec::new();
    if !bits.bit()? {
        return Ok(mods);
    }
    loop {
        let idc = bits.ue()?;
        if idc == 3 {
            break;
        }
        if idc > 3 {
            return Err(bits.malformed("ref list modification"));
        }
        let value = bits.ue()?;
        mods.push(RefPicListModification {
            modification_of_pic_nums_idc: idc as u8,
            abs_diff_pic_num_minus1: if idc < 2 { value } else { 0 },
            long_term_pic_num: if idc == 2 { value } else { 0 },
        });
        if mods.len() > 32 {
            return Err(CodecError::Limit {
                what: "ref list modification",
            });
        }
    }
    Ok(mods)
}

/// `n` entries of one list. Chroma weights are coded for every chroma
/// format the decoder accepts.
fn read_weights(
    bits: &mut Bits<'_>,
    n: usize,
    (luma_denom, chroma_denom): (u8, u8),
) -> Result<Vec<PredWeight>, CodecError> {
    let mut list = Vec::with_capacity(n);
    for _ in 0..n {
        let mut entry = PredWeight {
            luma_weight_flag: false,
            luma_weight: 1 << luma_denom,
            luma_offset: 0,
            chroma_weight_flag: false,
            chroma_weight: [1 << chroma_denom; 2],
            chroma_offset: [0; 2],
        };
        entry.luma_weight_flag = bits.bit()?;
        if entry.luma_weight_flag {
            entry.luma_weight = bits.se()?;
            entry.luma_offset = bits.se()?;
        }
        entry.chroma_weight_flag = bits.bit()?;
        if entry.chroma_weight_flag {
            for c in 0..2 {
                entry.chroma_weight[c] = bits.se()?;
                entry.chroma_offset[c] = bits.se()?;
            }
        }
        list.push(entry);
    }
    Ok(list)
}

fn read_mmco(bits: &mut Bits<'_>) -> Result<Vec<MemoryManagementControlOperation>, CodecError> {
    let mut operations = Vec::new();
    loop {
        let op = bits.ue()?;
        if op == 0 {
            break;
        }
        if op > 6 {
            return Err(bits.malformed("mmco"));
        }
        let mut operation = MemoryManagementControlOperation {
            memory_management_control_operation: op as u8,
            difference_of_pic_nums_minus1: 0,
            long_term_pic_num: 0,
            long_term_frame_idx: 0,
            max_long_term_frame_idx_plus1: 0,
        };
        match op {
            1 | 3 => operation.difference_of_pic_nums_minus1 = bits.ue()?,
            2 => operation.long_term_pic_num = bits.ue()?,
            4 => operation.max_long_term_frame_idx_plus1 = bits.ue()?,
            _ => {}
        }
        if matches!(op, 3 | 6) {
            operation.long_term_frame_idx = bits.ue()?;
        }
        operations.push(operation);
    }
    Ok(operations)
}

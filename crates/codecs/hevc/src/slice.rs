//! Slice segment header (clause 7.3.6) and the weighted prediction table
//! (clause 7.3.6.3).
//!
//! The fields of [`SliceHeader`] are the syntax elements of the same name,
//! as coded or, when not coded, as clause 7.4.7 infers them.

use std::sync::Arc;

use cutlass_decode::CodecError;

use crate::bits::Bits;
use crate::nal::Nal;
use crate::ps::{MAX_DPB, Pps, Sps, StRps, parse_st_rps};

/// `slice_type` (Table 7-7). `slice_type as u8` is the coded value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SliceType {
    B = 0,
    P = 1,
    I = 2,
}

/// The explicit weights of one reference picture in `pred_weight_table()`.
/// A flag that is 0 leaves its deltas and offsets 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PredWeight {
    /// `luma_weight_l0_flag[i]` (or `_l1_`).
    pub luma_weight_flag: bool,
    /// `chroma_weight_l0_flag[i]` (or `_l1_`).
    pub chroma_weight_flag: bool,
    /// `delta_luma_weight_l0[i]` (or `_l1_`).
    pub delta_luma_weight: i32,
    /// `luma_offset_l0[i]` (or `_l1_`).
    pub luma_offset: i32,
    /// `delta_chroma_weight_l0[i][j]` (or `_l1_`), Cb then Cr.
    pub delta_chroma_weight: [i32; 2],
    /// `delta_chroma_offset_l0[i][j]` (or `_l1_`), Cb then Cr.
    pub delta_chroma_offset: [i32; 2],
    /// `ChromaOffsetL0[i][j]` (or `L1`, equation 7-56), Cb then Cr; 0 when
    /// the chroma flag is 0.
    pub chroma_offset: [i32; 2],
    /// `LumaWeightL0[i]`, `ChromaWeightL0[i][0]` and `[1]` (or `L1`).
    pub(crate) weight: [i32; 3],
    /// The offsets prediction adds: `luma_offset_l0[i]` and
    /// `ChromaOffsetL0[i][j]`, each shifted left by the bit depth less 8
    /// (clause 8.5.3.3.4.3).
    pub(crate) offset: [i32; 3],
}

/// `pred_weight_table()` (clause 7.3.6.3).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PredWeightTable {
    /// `luma_log2_weight_denom`.
    pub luma_log2_weight_denom: u8,
    /// `delta_chroma_log2_weight_denom`.
    pub delta_chroma_log2_weight_denom: i32,
    /// The weights of list 0 (`[0]`, `num_ref_idx_l0_active_minus1 + 1`
    /// entries) and list 1 (`[1]`, B slices only), by reference index.
    pub entries: [Vec<PredWeight>; 2],
}

impl PredWeightTable {
    /// `ChromaLog2WeightDenom`.
    pub fn chroma_log2_weight_denom(&self) -> u8 {
        (i32::from(self.luma_log2_weight_denom) + self.delta_chroma_log2_weight_denom) as u8
    }
}

/// One entry of the long-term reference pictures of a slice header, `i`
/// below `num_long_term_sps + num_long_term_pics`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LongTerm {
    /// `lt_idx_sps[i]` for an entry from the SPS candidates (`i` below
    /// `num_long_term_sps`), `None` for one coded in the header.
    pub lt_idx_sps: Option<u32>,
    /// `PocLsbLt[i]`: `lt_ref_pic_poc_lsb_sps[lt_idx_sps[i]]` or
    /// `poc_lsb_lt[i]`.
    pub poc_lsb_lt: u32,
    /// `UsedByCurrPicLt[i]`: `used_by_curr_pic_lt_sps_flag[lt_idx_sps[i]]`
    /// or `used_by_curr_pic_lt_flag[i]`.
    pub used_by_curr_pic_lt_flag: bool,
    /// `delta_poc_msb_present_flag[i]`.
    pub delta_poc_msb_present_flag: bool,
    /// `DeltaPocMsbCycleLt[i]` (equation 7-52): the coded
    /// `delta_poc_msb_cycle_lt` values summed from the first entry of the
    /// SPS candidates or of the header entries.
    pub delta_poc_msb_cycle_lt: u32,
}

/// A slice segment header, `slice_segment_header()` (clause 7.3.6.1).
///
/// The header of a dependent slice segment carries the values of the
/// independent one before it, with its own
/// `first_slice_segment_in_pic_flag`, `no_output_of_prior_pics_flag`,
/// `slice_pic_parameter_set_id`, `dependent_slice_segment_flag`,
/// `slice_segment_address` and entry points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceHeader {
    /// `first_slice_segment_in_pic_flag`.
    pub first_slice_segment_in_pic_flag: bool,
    /// `no_output_of_prior_pics_flag`.
    pub no_output_of_prior_pics_flag: bool,
    /// `slice_pic_parameter_set_id`.
    pub slice_pic_parameter_set_id: u8,
    /// `dependent_slice_segment_flag`.
    pub dependent_slice_segment_flag: bool,
    /// `slice_segment_address`.
    pub slice_segment_address: u32,
    /// `slice_type`.
    pub slice_type: SliceType,
    /// `pic_output_flag` (1 when not coded).
    pub pic_output_flag: bool,
    /// `slice_pic_order_cnt_lsb` (0 for an IDR picture).
    pub slice_pic_order_cnt_lsb: u32,
    /// `short_term_ref_pic_set_sps_flag`.
    pub short_term_ref_pic_set_sps_flag: bool,
    /// `short_term_ref_pic_set_idx`.
    pub short_term_ref_pic_set_idx: u32,
    /// The short-term reference picture set the picture uses: the SPS's
    /// `st_ref_pic_set(short_term_ref_pic_set_idx)` or the
    /// `st_ref_pic_set(num_short_term_ref_pic_sets)` coded here. Empty for
    /// an IDR picture.
    pub st_ref_pic_set: StRps,
    /// `num_long_term_sps`.
    pub num_long_term_sps: u32,
    /// `num_long_term_pics`.
    pub num_long_term_pics: u32,
    /// The long-term entries, `num_long_term_sps + num_long_term_pics`.
    pub long_term: Vec<LongTerm>,
    /// `slice_temporal_mvp_enabled_flag`.
    pub slice_temporal_mvp_enabled_flag: bool,
    /// `slice_sao_luma_flag`.
    pub slice_sao_luma_flag: bool,
    /// `slice_sao_chroma_flag`.
    pub slice_sao_chroma_flag: bool,
    /// `num_ref_idx_active_override_flag`.
    pub num_ref_idx_active_override_flag: bool,
    /// `num_ref_idx_l0_active_minus1`: coded, or the PPS default, in P and B
    /// slices; 0 in I slices.
    pub num_ref_idx_l0_active_minus1: u8,
    /// `num_ref_idx_l1_active_minus1`: coded, or the PPS default, in B
    /// slices; 0 in others.
    pub num_ref_idx_l1_active_minus1: u8,
    /// `ref_pic_list_modification_flag_l0`.
    pub ref_pic_list_modification_flag_l0: bool,
    /// `list_entry_l0[i]`, when `ref_pic_list_modification_flag_l0` is 1.
    pub list_entry_l0: Vec<u32>,
    /// `ref_pic_list_modification_flag_l1`.
    pub ref_pic_list_modification_flag_l1: bool,
    /// `list_entry_l1[i]`, when `ref_pic_list_modification_flag_l1` is 1.
    pub list_entry_l1: Vec<u32>,
    /// `mvd_l1_zero_flag`.
    pub mvd_l1_zero_flag: bool,
    /// `cabac_init_flag`.
    pub cabac_init_flag: bool,
    /// `collocated_from_l0_flag` (1 when not coded).
    pub collocated_from_l0_flag: bool,
    /// `collocated_ref_idx` (0 when not coded).
    pub collocated_ref_idx: u8,
    /// `pred_weight_table()`, in P slices with `weighted_pred_flag` and B
    /// slices with `weighted_bipred_flag`.
    pub pred_weight_table: Option<PredWeightTable>,
    /// `five_minus_max_num_merge_cand` (0 in I slices).
    pub five_minus_max_num_merge_cand: u8,
    /// `slice_qp_delta`.
    pub slice_qp_delta: i32,
    /// `slice_cb_qp_offset`.
    pub slice_cb_qp_offset: i32,
    /// `slice_cr_qp_offset`.
    pub slice_cr_qp_offset: i32,
    /// `deblocking_filter_override_flag`.
    pub deblocking_filter_override_flag: bool,
    /// `slice_deblocking_filter_disabled_flag` (the PPS's when not coded).
    pub slice_deblocking_filter_disabled_flag: bool,
    /// `slice_beta_offset_div2` (the PPS's when not coded).
    pub slice_beta_offset_div2: i32,
    /// `slice_tc_offset_div2` (the PPS's when not coded).
    pub slice_tc_offset_div2: i32,
    /// `slice_loop_filter_across_slices_enabled_flag` (the PPS's when not
    /// coded).
    pub slice_loop_filter_across_slices_enabled_flag: bool,
    /// `num_entry_point_offsets`.
    pub num_entry_point_offsets: u32,
    /// `offset_len_minus1`.
    pub offset_len_minus1: u8,
    /// `entry_point_offset_minus1[i]`, coded bytes (emulation prevention
    /// bytes counted).
    pub entry_point_offset_minus1: Vec<u32>,
    /// `slice_segment_header_extension_length`.
    pub slice_segment_header_extension_length: u32,
}

impl SliceHeader {
    /// `NumPicTotalCurr` (equation 7-55).
    pub fn num_pic_total_curr(&self) -> usize {
        self.st_ref_pic_set
            .used_by_curr_pic_s0
            .iter()
            .filter(|&&used| used)
            .count()
            + self
                .st_ref_pic_set
                .used_by_curr_pic_s1
                .iter()
                .filter(|&&used| used)
                .count()
            + self
                .long_term
                .iter()
                .filter(|lt| lt.used_by_curr_pic_lt_flag)
                .count()
    }

    /// The sizes of reference picture lists 0 and 1:
    /// `num_ref_idx_lX_active_minus1 + 1` for a list the slice type uses,
    /// 0 for one it does not.
    pub fn num_ref_idx_active(&self) -> [usize; 2] {
        let l0 = usize::from(self.num_ref_idx_l0_active_minus1) + 1;
        let l1 = usize::from(self.num_ref_idx_l1_active_minus1) + 1;
        match self.slice_type {
            SliceType::I => [0, 0],
            SliceType::P => [l0, 0],
            SliceType::B => [l0, l1],
        }
    }

    /// `list_entry_lX` of list `list` when it is modified.
    pub fn list_entry(&self, list: usize) -> Option<&[u32]> {
        match list {
            0 if self.ref_pic_list_modification_flag_l0 => Some(&self.list_entry_l0),
            1 if self.ref_pic_list_modification_flag_l1 => Some(&self.list_entry_l1),
            _ => None,
        }
    }

    /// `MaxNumMergeCand`.
    pub fn max_num_merge_cand(&self) -> u8 {
        5 - self.five_minus_max_num_merge_cand
    }
}

fn ceil_log2(value: u32) -> u32 {
    if value <= 1 {
        0
    } else {
        32 - (value - 1).leading_zeros()
    }
}

/// A parsed slice segment header and where its parts end.
pub struct Parsed {
    pub header: SliceHeader,
    pub sps: Arc<Sps>,
    pub pps: Arc<Pps>,
    /// Byte offset of `slice_segment_data()` in the RBSP after the NAL
    /// unit header.
    pub data_offset: usize,
    /// Bits of the `st_ref_pic_set()` coded in the header: 0 when there is
    /// none, as in a dependent slice segment.
    pub st_rps_bits: u32,
}

/// Parse a slice segment header. `prev` is the independent header a
/// dependent segment copies.
pub fn parse_header(
    nal: &Nal,
    sets: &dyn Fn(u8) -> Option<(Arc<Sps>, Arc<Pps>)>,
    prev: Option<&SliceHeader>,
) -> Result<Parsed, CodecError> {
    let mut bits = Bits::new(&nal.rbsp);
    let first_in_pic = bits.bit()?;
    let no_output_of_prior_pics = if nal.is_irap() { bits.bit()? } else { false };
    let pps_id = bits.ue_max(63, "slice_pic_parameter_set_id")? as u8;
    let Some((sps, pps)) = sets(pps_id) else {
        return Err(bits.malformed("slice references a missing parameter set"));
    };
    let ctbs = sps.ctb_width() * sps.ctb_height();
    let mut dependent = false;
    let mut segment_address = 0;
    if !first_in_pic {
        if pps.dependent_slice_segments_enabled_flag {
            dependent = bits.bit()?;
        }
        segment_address = bits.u(ceil_log2(ctbs))?;
        if segment_address >= ctbs {
            return Err(bits.malformed("slice_segment_address"));
        }
    }
    let mut st_rps_bits = 0;
    let mut header = if dependent {
        let Some(prev) = prev else {
            return Err(bits.malformed("dependent slice without a slice"));
        };
        let mut copy = prev.clone();
        copy.first_slice_segment_in_pic_flag = first_in_pic;
        copy.no_output_of_prior_pics_flag = no_output_of_prior_pics;
        copy.slice_pic_parameter_set_id = pps_id;
        copy.dependent_slice_segment_flag = true;
        copy.slice_segment_address = segment_address;
        copy.num_entry_point_offsets = 0;
        copy.offset_len_minus1 = 0;
        copy.entry_point_offset_minus1 = Vec::new();
        copy.slice_segment_header_extension_length = 0;
        copy
    } else {
        independent(
            &mut bits,
            nal,
            (&sps, &pps),
            Start {
                first_in_pic,
                no_output_of_prior_pics,
                pps_id,
                segment_address,
            },
            &mut st_rps_bits,
        )?
    };
    if pps.tiles_enabled_flag || pps.entropy_coding_sync_enabled_flag {
        let max = if pps.tiles_enabled_flag {
            ctbs
        } else {
            sps.ctb_height()
        };
        let count = bits.ue_max(max, "num_entry_point_offsets")?;
        header.num_entry_point_offsets = count;
        if count > 0 {
            let len_minus1 = bits.ue_max(31, "offset_len_minus1")?;
            header.offset_len_minus1 = len_minus1 as u8;
            for _ in 0..count {
                header
                    .entry_point_offset_minus1
                    .push(bits.u(len_minus1 + 1)?);
            }
        }
    }
    if pps.slice_segment_header_extension_present_flag {
        let len = bits.ue_max(256, "slice_segment_header_extension_length")?;
        header.slice_segment_header_extension_length = len;
        bits.skip(len as usize * 8)?;
    }
    // byte_alignment(): a one, then zeros.
    if !bits.bit()? {
        return Err(bits.malformed("alignment_bit_equal_to_one"));
    }
    while !bits.byte_aligned() {
        if bits.bit()? {
            return Err(bits.malformed("alignment_bit_equal_to_zero"));
        }
    }
    let data_offset = bits.pos() / 8;
    if data_offset >= nal.rbsp.len() {
        return Err(bits.truncated("slice data"));
    }
    Ok(Parsed {
        header,
        sps,
        pps,
        data_offset,
        st_rps_bits,
    })
}

/// The fields read before the independent part of a header.
struct Start {
    first_in_pic: bool,
    no_output_of_prior_pics: bool,
    pps_id: u8,
    segment_address: u32,
}

fn independent(
    bits: &mut Bits<'_>,
    nal: &Nal,
    (sps, pps): (&Sps, &Pps),
    start: Start,
    st_rps_bits: &mut u32,
) -> Result<SliceHeader, CodecError> {
    let Start {
        first_in_pic,
        no_output_of_prior_pics,
        pps_id,
        segment_address,
    } = start;
    bits.skip(usize::from(pps.num_extra_slice_header_bits))?;
    let slice_type = match bits.ue_max(2, "slice_type")? {
        0 => SliceType::B,
        1 => SliceType::P,
        _ => SliceType::I,
    };
    if nal.is_irap() && slice_type != SliceType::I {
        return Err(bits.malformed("an IRAP slice that is not intra"));
    }
    let pic_output = if pps.output_flag_present_flag {
        bits.bit()?
    } else {
        true
    };
    let mut header = SliceHeader {
        first_slice_segment_in_pic_flag: first_in_pic,
        no_output_of_prior_pics_flag: no_output_of_prior_pics,
        slice_pic_parameter_set_id: pps_id,
        dependent_slice_segment_flag: false,
        slice_segment_address: segment_address,
        slice_type,
        pic_output_flag: pic_output,
        slice_pic_order_cnt_lsb: 0,
        short_term_ref_pic_set_sps_flag: false,
        short_term_ref_pic_set_idx: 0,
        st_ref_pic_set: StRps::default(),
        num_long_term_sps: 0,
        num_long_term_pics: 0,
        long_term: Vec::new(),
        slice_temporal_mvp_enabled_flag: false,
        slice_sao_luma_flag: false,
        slice_sao_chroma_flag: false,
        num_ref_idx_active_override_flag: false,
        num_ref_idx_l0_active_minus1: 0,
        num_ref_idx_l1_active_minus1: 0,
        ref_pic_list_modification_flag_l0: false,
        list_entry_l0: Vec::new(),
        ref_pic_list_modification_flag_l1: false,
        list_entry_l1: Vec::new(),
        mvd_l1_zero_flag: false,
        cabac_init_flag: false,
        collocated_from_l0_flag: true,
        collocated_ref_idx: 0,
        pred_weight_table: None,
        five_minus_max_num_merge_cand: 0,
        slice_qp_delta: 0,
        slice_cb_qp_offset: 0,
        slice_cr_qp_offset: 0,
        deblocking_filter_override_flag: false,
        slice_deblocking_filter_disabled_flag: false,
        slice_beta_offset_div2: 0,
        slice_tc_offset_div2: 0,
        slice_loop_filter_across_slices_enabled_flag: false,
        num_entry_point_offsets: 0,
        offset_len_minus1: 0,
        entry_point_offset_minus1: Vec::new(),
        slice_segment_header_extension_length: 0,
    };
    if !nal.is_idr() {
        let poc_lsb = bits.u(u32::from(sps.log2_max_poc_lsb()))?;
        header.slice_pic_order_cnt_lsb = poc_lsb;
        header.short_term_ref_pic_set_sps_flag = bits.bit()?;
        if header.short_term_ref_pic_set_sps_flag {
            let count = sps.st_ref_pic_set.len() as u32;
            if count == 0 {
                return Err(bits.malformed("short_term_ref_pic_set_sps_flag"));
            }
            let index = bits.u(ceil_log2(count))?;
            let Some(set) = sps.st_ref_pic_set.get(index as usize) else {
                return Err(bits.malformed("short_term_ref_pic_set_idx"));
            };
            header.short_term_ref_pic_set_idx = index;
            header.st_ref_pic_set = set.clone();
        } else {
            let before = bits.pos();
            let sets = &sps.st_ref_pic_set;
            header.st_ref_pic_set = parse_st_rps(bits, sets, sets.len(), sets.len(), true)?;
            *st_rps_bits = (bits.pos() - before) as u32;
        }
        if sps.long_term_ref_pics_present_flag {
            long_term(bits, sps, &mut header)?;
        }
        if sps.sps_temporal_mvp_enabled_flag {
            header.slice_temporal_mvp_enabled_flag = bits.bit()?;
        }
    }
    if sps.sample_adaptive_offset_enabled_flag {
        header.slice_sao_luma_flag = bits.bit()?;
        header.slice_sao_chroma_flag = bits.bit()?;
    }
    if slice_type != SliceType::I {
        header.num_ref_idx_l0_active_minus1 = pps.num_ref_idx_l0_default_active_minus1;
        if slice_type == SliceType::B {
            header.num_ref_idx_l1_active_minus1 = pps.num_ref_idx_l1_default_active_minus1;
        }
        header.num_ref_idx_active_override_flag = bits.bit()?;
        if header.num_ref_idx_active_override_flag {
            header.num_ref_idx_l0_active_minus1 =
                bits.ue_max(14, "num_ref_idx_l0_active_minus1")? as u8;
            if slice_type == SliceType::B {
                header.num_ref_idx_l1_active_minus1 =
                    bits.ue_max(14, "num_ref_idx_l1_active_minus1")? as u8;
            }
        }
        let num_ref_idx = header.num_ref_idx_active();
        let total = header.num_pic_total_curr();
        if total == 0 {
            return Err(bits.malformed("an inter slice with no reference"));
        }
        if pps.lists_modification_present_flag && total > 1 {
            let len = ceil_log2(total as u32);
            let lists = if slice_type == SliceType::B { 2 } else { 1 };
            for (list, &count) in num_ref_idx.iter().enumerate().take(lists) {
                if bits.bit()? {
                    let mut entries = Vec::with_capacity(count);
                    for _ in 0..count {
                        let value = bits.u(len)?;
                        if value as usize >= total {
                            return Err(bits.malformed("list_entry"));
                        }
                        entries.push(value);
                    }
                    if list == 0 {
                        header.ref_pic_list_modification_flag_l0 = true;
                        header.list_entry_l0 = entries;
                    } else {
                        header.ref_pic_list_modification_flag_l1 = true;
                        header.list_entry_l1 = entries;
                    }
                }
            }
        }
        if slice_type == SliceType::B {
            header.mvd_l1_zero_flag = bits.bit()?;
        }
        if pps.cabac_init_present_flag {
            header.cabac_init_flag = bits.bit()?;
        }
        if header.slice_temporal_mvp_enabled_flag {
            if slice_type == SliceType::B {
                header.collocated_from_l0_flag = bits.bit()?;
            }
            let list = if header.collocated_from_l0_flag { 0 } else { 1 };
            if num_ref_idx[list] > 1 {
                header.collocated_ref_idx =
                    bits.ue_max(num_ref_idx[list] as u32 - 1, "collocated_ref_idx")? as u8;
            }
        }
        if (pps.weighted_pred_flag && slice_type == SliceType::P)
            || (pps.weighted_bipred_flag && slice_type == SliceType::B)
        {
            header.pred_weight_table = Some(pred_weight_table(bits, sps, slice_type, num_ref_idx)?);
        }
        header.five_minus_max_num_merge_cand =
            bits.ue_max(4, "five_minus_max_num_merge_cand")? as u8;
    }
    let qp_delta = bits.se()?;
    let qp = 26 + pps.init_qp_minus26 + qp_delta;
    if qp < -sps.qp_bd_offset() || qp > 51 {
        return Err(bits.malformed("slice_qp_delta"));
    }
    header.slice_qp_delta = qp_delta;
    if pps.pps_slice_chroma_qp_offsets_present_flag {
        let cb_qp_offset = bits.se_range(-12, 12, "slice_cb_qp_offset")?;
        let cr_qp_offset = bits.se_range(-12, 12, "slice_cr_qp_offset")?;
        if !(-12..=12).contains(&(pps.pps_cb_qp_offset + cb_qp_offset))
            || !(-12..=12).contains(&(pps.pps_cr_qp_offset + cr_qp_offset))
        {
            return Err(bits.malformed("chroma qp offset"));
        }
        header.slice_cb_qp_offset = cb_qp_offset;
        header.slice_cr_qp_offset = cr_qp_offset;
    }
    header.slice_deblocking_filter_disabled_flag = pps.pps_deblocking_filter_disabled_flag;
    header.slice_beta_offset_div2 = pps.pps_beta_offset_div2;
    header.slice_tc_offset_div2 = pps.pps_tc_offset_div2;
    if pps.deblocking_filter_override_enabled_flag {
        header.deblocking_filter_override_flag = bits.bit()?;
    }
    if header.deblocking_filter_override_flag {
        header.slice_deblocking_filter_disabled_flag = bits.bit()?;
        if !header.slice_deblocking_filter_disabled_flag {
            header.slice_beta_offset_div2 = bits.se_range(-6, 6, "slice_beta_offset_div2")?;
            header.slice_tc_offset_div2 = bits.se_range(-6, 6, "slice_tc_offset_div2")?;
        }
    }
    header.slice_loop_filter_across_slices_enabled_flag = if pps
        .pps_loop_filter_across_slices_enabled_flag
        && (header.slice_sao_luma_flag
            || header.slice_sao_chroma_flag
            || !header.slice_deblocking_filter_disabled_flag)
    {
        bits.bit()?
    } else {
        pps.pps_loop_filter_across_slices_enabled_flag
    };
    Ok(header)
}

/// The long-term entries of a slice header.
fn long_term(bits: &mut Bits<'_>, sps: &Sps, header: &mut SliceHeader) -> Result<(), CodecError> {
    let candidates = sps.lt_ref_pic_poc_lsb_sps.len();
    let from_sps = if candidates == 0 {
        0
    } else {
        bits.ue_max(candidates as u32, "num_long_term_sps")?
    };
    let from_header = bits.ue_max(MAX_DPB as u32, "num_long_term_pics")?;
    let total = from_sps + from_header;
    if total as usize + header.st_ref_pic_set.num_delta_pocs() > MAX_DPB {
        return Err(bits.malformed("long-term reference count"));
    }
    header.num_long_term_sps = from_sps;
    header.num_long_term_pics = from_header;
    let log2_max_poc_lsb = sps.log2_max_poc_lsb();
    let max_lsb = 1i64 << log2_max_poc_lsb;
    let poc_lsb = header.slice_pic_order_cnt_lsb;
    let mut msb_cycle = 0i64;
    for i in 0..total {
        let (lt_idx_sps, lsb, used) = if i < from_sps {
            let idx = bits.u(ceil_log2(candidates as u32))?;
            let (Some(&lsb), Some(&used)) = (
                sps.lt_ref_pic_poc_lsb_sps.get(idx as usize),
                sps.used_by_curr_pic_lt_sps_flag.get(idx as usize),
            ) else {
                return Err(bits.malformed("lt_idx_sps"));
            };
            (Some(idx), lsb, used)
        } else {
            (None, bits.u(u32::from(log2_max_poc_lsb))?, bits.bit()?)
        };
        let msb_present = bits.bit()?;
        if msb_present {
            let delta = i64::from(bits.ue()?);
            msb_cycle = if i == 0 || i == from_sps {
                delta
            } else {
                msb_cycle + delta
            };
            // The reference's order count less the current one's must fit:
            // the decoder adds them when it resolves the picture.
            let relative = i64::from(lsb) - msb_cycle * max_lsb - i64::from(poc_lsb);
            if i32::try_from(relative).is_err() {
                return Err(bits.malformed("delta_poc_msb_cycle_lt"));
            }
        } else if i == 0 || i == from_sps {
            msb_cycle = 0;
        }
        let Ok(cycle) = u32::try_from(msb_cycle) else {
            return Err(bits.malformed("delta_poc_msb_cycle_lt"));
        };
        header.long_term.push(LongTerm {
            lt_idx_sps,
            poc_lsb_lt: lsb,
            used_by_curr_pic_lt_flag: used,
            delta_poc_msb_present_flag: msb_present,
            delta_poc_msb_cycle_lt: cycle,
        });
    }
    Ok(())
}

impl LongTerm {
    /// The order count the entry names, and whether it is the whole count
    /// (`delta_poc_msb_present_flag`) rather than its least significant
    /// bits: `PocLtCurr` or `PocLtFoll` (equation 8-5), for a picture of
    /// order count `poc` whose slice header has `slice_pic_order_cnt_lsb`
    /// and whose SPS has `MaxPicOrderCntLsb` of `max_lsb`.
    pub(crate) fn target(&self, poc: i32, poc_lsb: u32, max_lsb: i32) -> (i32, bool) {
        if self.delta_poc_msb_present_flag {
            // In range: the header parse checked it.
            let relative = i64::from(self.poc_lsb_lt)
                - i64::from(self.delta_poc_msb_cycle_lt) * i64::from(max_lsb)
                - i64::from(poc_lsb);
            (poc.wrapping_add(relative as i32), true)
        } else {
            (self.poc_lsb_lt as i32, false)
        }
    }
}

fn pred_weight_table(
    bits: &mut Bits<'_>,
    sps: &Sps,
    slice_type: SliceType,
    num_ref_idx: [usize; 2],
) -> Result<PredWeightTable, CodecError> {
    let luma_denom = bits.ue_max(7, "luma_log2_weight_denom")?;
    let delta_chroma_denom = bits.se()?;
    let chroma_denom = i64::from(luma_denom) + i64::from(delta_chroma_denom);
    if !(0..=7).contains(&chroma_denom) {
        return Err(bits.malformed("delta_chroma_log2_weight_denom"));
    }
    let chroma_denom = chroma_denom as u32;
    let shift = i32::from(sps.bit_depth()) - 8;
    let mut table = PredWeightTable {
        luma_log2_weight_denom: luma_denom as u8,
        delta_chroma_log2_weight_denom: delta_chroma_denom,
        entries: [Vec::new(), Vec::new()],
    };
    let lists = if slice_type == SliceType::B { 2 } else { 1 };
    for (list, &count) in num_ref_idx.iter().enumerate().take(lists) {
        let mut luma_flags = [false; 16];
        let mut chroma_flags = [false; 16];
        for flag in luma_flags.iter_mut().take(count) {
            *flag = bits.bit()?;
        }
        for flag in chroma_flags.iter_mut().take(count) {
            *flag = bits.bit()?;
        }
        for i in 0..count {
            let mut entry = PredWeight {
                luma_weight_flag: luma_flags[i],
                chroma_weight_flag: chroma_flags[i],
                weight: [1 << luma_denom, 1 << chroma_denom, 1 << chroma_denom],
                ..PredWeight::default()
            };
            if luma_flags[i] {
                entry.delta_luma_weight = bits.se_range(-128, 127, "delta_luma_weight")?;
                entry.luma_offset = bits.se_range(-128, 127, "luma_offset")?;
                entry.weight[0] += entry.delta_luma_weight;
                entry.offset[0] = entry.luma_offset << shift;
            }
            if chroma_flags[i] {
                for c in 0..2 {
                    let delta_weight = bits.se_range(-128, 127, "delta_chroma_weight")?;
                    let w = (1 << chroma_denom) + delta_weight;
                    let delta = bits.se_range(-512, 511, "delta_chroma_offset")?;
                    let offset = (delta - ((128 * w) >> chroma_denom) + 128).clamp(-128, 127);
                    entry.delta_chroma_weight[c] = delta_weight;
                    entry.delta_chroma_offset[c] = delta;
                    entry.chroma_offset[c] = offset;
                    entry.weight[c + 1] = w;
                    entry.offset[c + 1] = offset << shift;
                }
            }
            table.entries[list].push(entry);
        }
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::ceil_log2;

    #[test]
    fn ceil_log2_counts_the_bits_of_an_index() {
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
    }
}

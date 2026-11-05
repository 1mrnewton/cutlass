//! Slice segment header (clause 7.3.6) and the weighted prediction table
//! (clause 7.3.6.3).

use cutlass_decode::CodecError;

use crate::bits::Bits;
use crate::nal::Nal;
use crate::ps::{MAX_DPB, Pps, Sps, StRps, parse_st_rps};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceType {
    B,
    P,
    I,
}

/// Explicit weights of one reference, luma then Cb and Cr.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Weight {
    pub weight: [i32; 3],
    pub offset: [i32; 3],
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Weights {
    pub log2_denom: [u32; 2],
    pub lists: [Vec<Weight>; 2],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LongTerm {
    pub poc: i32,
    /// Whether `poc` is the full order count rather than its LSBs.
    pub msb: bool,
    pub used: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceHeader {
    pub first_in_pic: bool,
    pub no_output_of_prior_pics: bool,
    pub pps_id: u8,
    pub dependent: bool,
    pub segment_address: u32,
    pub slice_type: SliceType,
    pub pic_output: bool,
    pub poc_lsb: u32,
    pub st_rps: StRps,
    /// Bits of a short-term set coded in the header.
    pub long_term: Vec<LongTerm>,
    pub temporal_mvp: bool,
    pub sao_luma: bool,
    pub sao_chroma: bool,
    pub num_ref_idx: [usize; 2],
    pub list_entry: [Option<Vec<u32>>; 2],
    pub mvd_l1_zero: bool,
    pub cabac_init: bool,
    pub collocated_from_l0: bool,
    pub collocated_ref_idx: usize,
    pub weights: Option<Weights>,
    pub max_merge_cand: u8,
    pub qp_delta: i32,
    pub cb_qp_offset: i32,
    pub cr_qp_offset: i32,
    pub deblocking_disabled: bool,
    pub beta_offset: i32,
    pub tc_offset: i32,
    pub loop_filter_across_slices: bool,
    /// Entry point offsets, coded bytes, one per substream after the first.
    pub entry_points: Vec<u32>,
    /// Byte offset of the slice data in the RBSP.
    pub data_offset: usize,
}

impl SliceHeader {
    /// `NumPicTotalCurr` (equation 7-55).
    pub fn num_pic_total_curr(&self) -> usize {
        self.st_rps
            .negative_used
            .iter()
            .filter(|&&used| used)
            .count()
            + self
                .st_rps
                .positive_used
                .iter()
                .filter(|&&used| used)
                .count()
            + self.long_term.iter().filter(|lt| lt.used).count()
    }
}

fn ceil_log2(value: u32) -> u32 {
    if value <= 1 {
        0
    } else {
        32 - (value - 1).leading_zeros()
    }
}

/// Parse a slice segment header. `prev` is the independent header a
/// dependent segment copies.
pub fn parse_header(
    nal: &Nal,
    sps_of: &dyn Fn(u8) -> Option<(Sps, Pps)>,
    prev: Option<&SliceHeader>,
) -> Result<(SliceHeader, Sps, Pps), CodecError> {
    let mut bits = Bits::new(&nal.rbsp);
    let first_in_pic = bits.bit()?;
    let no_output_of_prior_pics = if nal.is_irap() { bits.bit()? } else { false };
    let pps_id = bits.ue_max(63, "slice_pic_parameter_set_id")? as u8;
    let Some((sps, pps)) = sps_of(pps_id) else {
        return Err(bits.malformed("slice references a missing parameter set"));
    };
    let ctbs = sps.ctb_width() * sps.ctb_height();
    let mut dependent = false;
    let mut segment_address = 0;
    if !first_in_pic {
        if pps.dependent_slices {
            dependent = bits.bit()?;
        }
        segment_address = bits.u(ceil_log2(ctbs))?;
        if segment_address >= ctbs {
            return Err(bits.malformed("slice_segment_address"));
        }
    }
    let mut header = if dependent {
        let Some(prev) = prev else {
            return Err(bits.malformed("dependent slice without a slice"));
        };
        let mut copy = prev.clone();
        copy.first_in_pic = first_in_pic;
        copy.no_output_of_prior_pics = no_output_of_prior_pics;
        copy.pps_id = pps_id;
        copy.dependent = true;
        copy.segment_address = segment_address;
        copy.entry_points = Vec::new();
        copy
    } else {
        independent(
            &mut bits,
            nal,
            &sps,
            &pps,
            Start {
                first_in_pic,
                no_output_of_prior_pics,
                pps_id,
                segment_address,
            },
        )?
    };
    if pps.tiles || pps.entropy_sync {
        let max = if pps.tiles { ctbs } else { sps.ctb_height() };
        let count = bits.ue_max(max, "num_entry_point_offsets")?;
        if count > 0 {
            let len = bits.ue_max(31, "offset_len_minus1")? + 1;
            for _ in 0..count {
                header.entry_points.push(bits.u(len)?.saturating_add(1));
            }
        }
    }
    if pps.slice_header_extension {
        let len = bits.ue_max(256, "slice_segment_header_extension_length")?;
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
    header.data_offset = bits.pos() / 8;
    if header.data_offset >= nal.rbsp.len() {
        return Err(bits.truncated("slice data"));
    }
    Ok((header, sps, pps))
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
    sps: &Sps,
    pps: &Pps,
    start: Start,
) -> Result<SliceHeader, CodecError> {
    let Start {
        first_in_pic,
        no_output_of_prior_pics,
        pps_id,
        segment_address,
    } = start;
    bits.skip(usize::from(pps.extra_slice_header_bits))?;
    let slice_type = match bits.ue_max(2, "slice_type")? {
        0 => SliceType::B,
        1 => SliceType::P,
        _ => SliceType::I,
    };
    if nal.is_irap() && slice_type != SliceType::I {
        return Err(bits.malformed("an IRAP slice that is not intra"));
    }
    let pic_output = if pps.output_flag_present {
        bits.bit()?
    } else {
        true
    };
    let mut poc_lsb = 0;
    let mut st_rps = StRps::default();
    let mut long_term = Vec::new();
    let mut temporal_mvp = false;
    if !nal.is_idr() {
        poc_lsb = bits.u(u32::from(sps.log2_max_poc_lsb))?;
        if bits.bit()? {
            let count = sps.st_rps.len() as u32;
            if count == 0 {
                return Err(bits.malformed("short_term_ref_pic_set_sps_flag"));
            }
            let index = bits.u(ceil_log2(count))?;
            let Some(set) = sps.st_rps.get(index as usize) else {
                return Err(bits.malformed("short_term_ref_pic_set_idx"));
            };
            st_rps = set.clone();
        } else {
            st_rps = parse_st_rps(bits, &sps.st_rps, sps.st_rps.len(), sps.st_rps.len(), true)?;
        }
        if sps.long_term_present {
            let from_sps = if sps.lt_ref_pic_poc_lsb.is_empty() {
                0
            } else {
                bits.ue_max(sps.lt_ref_pic_poc_lsb.len() as u32, "num_long_term_sps")?
            };
            let from_header = bits.ue_max(MAX_DPB as u32, "num_long_term_pics")?;
            let total = from_sps + from_header;
            if total as usize + st_rps.negative.len() + st_rps.positive.len() > MAX_DPB {
                return Err(bits.malformed("long-term reference count"));
            }
            let max_lsb = 1i64 << sps.log2_max_poc_lsb;
            let mut msb_cycle = 0i64;
            for i in 0..total {
                let (lsb, used) = if i < from_sps {
                    let bits_len = ceil_log2(sps.lt_ref_pic_poc_lsb.len() as u32);
                    let idx = bits.u(bits_len)? as usize;
                    let (Some(&lsb), Some(&used)) = (
                        sps.lt_ref_pic_poc_lsb.get(idx),
                        sps.lt_used_by_curr.get(idx),
                    ) else {
                        return Err(bits.malformed("lt_idx_sps"));
                    };
                    (lsb, used)
                } else {
                    (bits.u(u32::from(sps.log2_max_poc_lsb))?, bits.bit()?)
                };
                let msb_present = bits.bit()?;
                if msb_present {
                    let delta = i64::from(bits.ue()?);
                    msb_cycle = if i == 0 || i == from_sps {
                        delta
                    } else {
                        msb_cycle + delta
                    };
                    // PocLsbLt with the cycle folded in. The decoder adds the
                    // current order count when it resolves the picture.
                    let relative = i64::from(lsb) - msb_cycle * max_lsb - i64::from(poc_lsb);
                    let Ok(relative) = i32::try_from(relative) else {
                        return Err(bits.malformed("delta_poc_msb_cycle_lt"));
                    };
                    long_term.push(LongTerm {
                        poc: relative,
                        msb: true,
                        used,
                    });
                } else {
                    if i == 0 || i == from_sps {
                        msb_cycle = 0;
                    }
                    long_term.push(LongTerm {
                        poc: lsb as i32,
                        msb: false,
                        used,
                    });
                }
            }
        }
        if sps.temporal_mvp {
            temporal_mvp = bits.bit()?;
        }
    }
    let mut sao_luma = false;
    let mut sao_chroma = false;
    if sps.sao {
        sao_luma = bits.bit()?;
        sao_chroma = bits.bit()?;
    }
    let mut num_ref_idx = [0usize; 2];
    let mut list_entry: [Option<Vec<u32>>; 2] = [None, None];
    let mut mvd_l1_zero = false;
    let mut cabac_init = false;
    let mut collocated_from_l0 = true;
    let mut collocated_ref_idx = 0;
    let mut weights = None;
    let mut max_merge_cand = 5;
    let mut header_tmp = SliceHeader {
        first_in_pic,
        no_output_of_prior_pics,
        pps_id,
        dependent: false,
        segment_address,
        slice_type,
        pic_output,
        poc_lsb,
        st_rps,
        long_term,
        temporal_mvp,
        sao_luma,
        sao_chroma,
        num_ref_idx,
        list_entry: [None, None],
        mvd_l1_zero,
        cabac_init,
        collocated_from_l0,
        collocated_ref_idx,
        weights: None,
        max_merge_cand,
        qp_delta: 0,
        cb_qp_offset: 0,
        cr_qp_offset: 0,
        deblocking_disabled: false,
        beta_offset: 0,
        tc_offset: 0,
        loop_filter_across_slices: false,
        entry_points: Vec::new(),
        data_offset: 0,
    };
    if slice_type != SliceType::I {
        num_ref_idx[0] = usize::from(pps.num_ref_idx_default[0]);
        if slice_type == SliceType::B {
            num_ref_idx[1] = usize::from(pps.num_ref_idx_default[1]);
        }
        if bits.bit()? {
            num_ref_idx[0] = bits.ue_max(14, "num_ref_idx_l0_active_minus1")? as usize + 1;
            if slice_type == SliceType::B {
                num_ref_idx[1] = bits.ue_max(14, "num_ref_idx_l1_active_minus1")? as usize + 1;
            }
        }
        let total = header_tmp.num_pic_total_curr();
        if total == 0 {
            return Err(bits.malformed("an inter slice with no reference"));
        }
        if pps.lists_modification && total > 1 {
            let len = ceil_log2(total as u32);
            let lists = if slice_type == SliceType::B { 2 } else { 1 };
            for (list, entry) in list_entry.iter_mut().enumerate().take(lists) {
                if bits.bit()? {
                    let mut entries = Vec::with_capacity(num_ref_idx[list]);
                    for _ in 0..num_ref_idx[list] {
                        let value = bits.u(len)?;
                        if value as usize >= total {
                            return Err(bits.malformed("list_entry"));
                        }
                        entries.push(value);
                    }
                    *entry = Some(entries);
                }
            }
        }
        if slice_type == SliceType::B {
            mvd_l1_zero = bits.bit()?;
        }
        if pps.cabac_init_present {
            cabac_init = bits.bit()?;
        }
        if temporal_mvp {
            if slice_type == SliceType::B {
                collocated_from_l0 = bits.bit()?;
            }
            let list = if collocated_from_l0 { 0 } else { 1 };
            if num_ref_idx[list] > 1 {
                collocated_ref_idx =
                    bits.ue_max(num_ref_idx[list] as u32 - 1, "collocated_ref_idx")? as usize;
            }
        }
        if (pps.weighted_pred && slice_type == SliceType::P)
            || (pps.weighted_bipred && slice_type == SliceType::B)
        {
            weights = Some(pred_weight_table(bits, sps, slice_type, num_ref_idx)?);
        }
        max_merge_cand = 5 - bits.ue_max(4, "five_minus_max_num_merge_cand")? as u8;
    }
    let qp_delta = bits.se()?;
    let qp = pps.init_qp + qp_delta;
    if qp < -sps.qp_bd_offset() || qp > 51 {
        return Err(bits.malformed("slice_qp_delta"));
    }
    let mut cb_qp_offset = 0;
    let mut cr_qp_offset = 0;
    if pps.slice_chroma_qp_offsets {
        cb_qp_offset = bits.se_range(-12, 12, "slice_cb_qp_offset")?;
        cr_qp_offset = bits.se_range(-12, 12, "slice_cr_qp_offset")?;
        if !(-12..=12).contains(&(pps.cb_qp_offset + cb_qp_offset))
            || !(-12..=12).contains(&(pps.cr_qp_offset + cr_qp_offset))
        {
            return Err(bits.malformed("chroma qp offset"));
        }
    }
    let mut deblocking_disabled = pps.deblocking_disabled;
    let mut beta_offset = pps.beta_offset;
    let mut tc_offset = pps.tc_offset;
    if pps.deblocking_override_enabled && bits.bit()? {
        deblocking_disabled = bits.bit()?;
        if !deblocking_disabled {
            beta_offset = 2 * bits.se_range(-6, 6, "slice_beta_offset_div2")?;
            tc_offset = 2 * bits.se_range(-6, 6, "slice_tc_offset_div2")?;
        }
    }
    let loop_filter_across_slices =
        if pps.loop_filter_across_slices && (sao_luma || sao_chroma || !deblocking_disabled) {
            bits.bit()?
        } else {
            pps.loop_filter_across_slices
        };
    header_tmp.num_ref_idx = num_ref_idx;
    header_tmp.list_entry = list_entry;
    header_tmp.mvd_l1_zero = mvd_l1_zero;
    header_tmp.cabac_init = cabac_init;
    header_tmp.collocated_from_l0 = collocated_from_l0;
    header_tmp.collocated_ref_idx = collocated_ref_idx;
    header_tmp.weights = weights;
    header_tmp.max_merge_cand = max_merge_cand;
    header_tmp.qp_delta = qp_delta;
    header_tmp.cb_qp_offset = cb_qp_offset;
    header_tmp.cr_qp_offset = cr_qp_offset;
    header_tmp.deblocking_disabled = deblocking_disabled;
    header_tmp.beta_offset = beta_offset;
    header_tmp.tc_offset = tc_offset;
    header_tmp.loop_filter_across_slices = loop_filter_across_slices;
    Ok(header_tmp)
}

fn pred_weight_table(
    bits: &mut Bits<'_>,
    sps: &Sps,
    slice_type: SliceType,
    num_ref_idx: [usize; 2],
) -> Result<Weights, CodecError> {
    let luma_denom = bits.ue_max(7, "luma_log2_weight_denom")?;
    let chroma_denom = i64::from(luma_denom) + i64::from(bits.se()?);
    if !(0..=7).contains(&chroma_denom) {
        return Err(bits.malformed("delta_chroma_log2_weight_denom"));
    }
    let chroma_denom = chroma_denom as u32;
    let shift = i32::from(sps.bit_depth) - 8;
    let mut weights = Weights {
        log2_denom: [luma_denom, chroma_denom],
        lists: [Vec::new(), Vec::new()],
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
            let mut weight = Weight {
                weight: [1 << luma_denom, 1 << chroma_denom, 1 << chroma_denom],
                offset: [0; 3],
            };
            if luma_flags[i] {
                weight.weight[0] += bits.se_range(-128, 127, "delta_luma_weight")?;
                weight.offset[0] = bits.se_range(-128, 127, "luma_offset")? << shift;
            }
            if chroma_flags[i] {
                for c in 1..3 {
                    let w =
                        (1 << chroma_denom) + bits.se_range(-128, 127, "delta_chroma_weight")?;
                    let delta = bits.se_range(-512, 511, "delta_chroma_offset")?;
                    let offset = (delta - ((128 * w) >> chroma_denom) + 128).clamp(-128, 127);
                    weight.weight[c] = w;
                    weight.offset[c] = offset << shift;
                }
            }
            weights.lists[list].push(weight);
        }
    }
    Ok(weights)
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

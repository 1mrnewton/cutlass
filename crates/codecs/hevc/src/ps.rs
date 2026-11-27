//! Sequence and picture parameter sets (clauses 7.3.2.2, 7.3.2.3, 7.3.4,
//! 7.3.7, and Annex E), and the tile scan of clause 6.5.1.
//!
//! The fields of [`Sps`], [`Pps`] and what they hold are the syntax
//! elements of the same name, as coded, or as inferred when not coded
//! (clause 7.4). Derived variables the decoding process uses are methods,
//! documented with the variable's name in the standard.

use cutlass_decode::{CodecError, ColorMatrix, ColorPrimaries, ColorRange, SampleAspect, Transfer};

use crate::bits::Bits;

/// Largest luma edge, the edge of a level 6.2 picture held to an 8:1 aspect.
pub const MAX_EDGE: u32 = 16_888;
/// Most luma samples in a picture: `MaxLumaPs` of level 6.2 (Table A.8).
pub const MAX_LUMA_SAMPLES: u64 = 35_651_584;
pub const MAX_DPB: usize = 16;

/// A short-term reference picture set, `st_ref_pic_set(stRpsIdx)` (clause
/// 7.3.7), with the lists clause 7.4.8 derives from it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StRps {
    /// `inter_ref_pic_set_prediction_flag`.
    pub inter_ref_pic_set_prediction_flag: bool,
    /// `delta_idx_minus1`: coded only in a slice header, 0 otherwise.
    pub delta_idx_minus1: u32,
    /// `delta_rps_sign`.
    pub delta_rps_sign: bool,
    /// `abs_delta_rps_minus1`.
    pub abs_delta_rps_minus1: u32,
    /// `used_by_curr_pic_flag[j]`, `j` from 0 to `NumDeltaPocs[RefRpsIdx]`,
    /// when the set is predicted; empty otherwise.
    pub used_by_curr_pic_flag: Vec<bool>,
    /// `use_delta_flag[j]` (1 when not coded), when the set is predicted;
    /// empty otherwise.
    pub use_delta_flag: Vec<bool>,
    /// `DeltaPocS0[i]`: the negative order count differences, closest first.
    /// Without prediction, the running sums of `delta_poc_s0_minus1 + 1`,
    /// negated.
    pub delta_poc_s0: Vec<i32>,
    /// `UsedByCurrPicS0[i]` (`used_by_curr_pic_s0_flag[i]` without
    /// prediction).
    pub used_by_curr_pic_s0: Vec<bool>,
    /// `DeltaPocS1[i]`: the positive order count differences, closest first.
    pub delta_poc_s1: Vec<i32>,
    /// `UsedByCurrPicS1[i]` (`used_by_curr_pic_s1_flag[i]` without
    /// prediction).
    pub used_by_curr_pic_s1: Vec<bool>,
}

impl StRps {
    /// `NumNegativePics`.
    pub fn num_negative_pics(&self) -> usize {
        self.delta_poc_s0.len()
    }

    /// `NumPositivePics`.
    pub fn num_positive_pics(&self) -> usize {
        self.delta_poc_s1.len()
    }

    /// `NumDeltaPocs`.
    pub fn num_delta_pocs(&self) -> usize {
        self.delta_poc_s0.len() + self.delta_poc_s1.len()
    }
}

/// The scaling factors of `scaling_list_data()` (clause 7.3.4), after
/// prediction from reference lists and default lists (clause 7.4.5).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalingList {
    /// `ScalingList[sizeId][matrixId][i]` at the raster position its
    /// up-right diagonal scan index `i` names: the first 16 entries of a
    /// 4×4 list (sizeId 0), all 64 of an 8×8 one. Of sizeId 3 only
    /// matrixId 0 and 3 are coded; the others hold the default lists.
    /// [`Self::coded`] gives a list in coded order.
    pub lists: [[[u8; 64]; 6]; 4],
    /// `scaling_list_dc_coef_minus8 + 8` (16 when predicted from a default
    /// list): `[sizeId - 2][matrixId]` for sizeId 2 and 3.
    pub dc: [[u8; 6]; 2],
}

impl ScalingList {
    pub fn default_lists() -> Self {
        let mut lists = [[[16u8; 64]; 6]; 4];
        for size in lists.iter_mut().skip(1) {
            for (matrix, list) in size.iter_mut().enumerate() {
                *list = if matrix < 3 {
                    DEFAULT_INTRA
                } else {
                    DEFAULT_INTER
                };
            }
        }
        Self {
            lists,
            dc: [[16; 6]; 2],
        }
    }

    /// The scaling lists a picture uses: the PPS's when
    /// `pps_scaling_list_data_present_flag` is 1, else the SPS's (the
    /// default lists when `sps_scaling_list_data_present_flag` is 0). `None`
    /// is flat scaling: `scaling_list_enabled_flag` is 0.
    pub fn active<'a>(sps: &'a Sps, pps: &'a Pps) -> Option<&'a Self> {
        pps.scaling_list.as_ref().or(sps.scaling_list.as_ref())
    }

    /// `ScalingList[sizeId][matrixId][i]` in coded (up-right diagonal scan)
    /// order: 16 entries for sizeId 0 (the rest 0), 64 for the others.
    /// Indices out of range give zeros.
    pub fn coded(&self, size_id: usize, matrix_id: usize) -> [u8; 64] {
        let mut out = [0u8; 64];
        let Some(list) = self.lists.get(size_id).and_then(|s| s.get(matrix_id)) else {
            return out;
        };
        let scan = if size_id == 0 {
            &DIAGONAL_4[..]
        } else {
            &DIAGONAL_8[..]
        };
        for (slot, &pos) in out.iter_mut().zip(scan) {
            *slot = list[usize::from(pos)];
        }
        out
    }

    /// The 4×4 lists (sizeId 0) of matrixId 0 to 5, in coded order.
    pub fn scaling_list_4x4(&self) -> [[u8; 16]; 6] {
        std::array::from_fn(|m| {
            let coded = self.coded(0, m);
            std::array::from_fn(|i| coded[i])
        })
    }

    /// The 8×8 lists (sizeId 1) of matrixId 0 to 5, in coded order.
    pub fn scaling_list_8x8(&self) -> [[u8; 64]; 6] {
        std::array::from_fn(|m| self.coded(1, m))
    }

    /// The 16×16 lists (sizeId 2) of matrixId 0 to 5, in coded order.
    pub fn scaling_list_16x16(&self) -> [[u8; 64]; 6] {
        std::array::from_fn(|m| self.coded(2, m))
    }

    /// The 32×32 lists (sizeId 3) of matrixId 0 (intra) and 3 (inter), in
    /// coded order.
    pub fn scaling_list_32x32(&self) -> [[u8; 64]; 2] {
        [self.coded(3, 0), self.coded(3, 3)]
    }

    /// The DC values of the 16×16 lists, matrixId 0 to 5.
    pub fn scaling_list_dc_16x16(&self) -> [u8; 6] {
        self.dc[0]
    }

    /// The DC values of the 32×32 lists of matrixId 0 and 3.
    pub fn scaling_list_dc_32x32(&self) -> [u8; 2] {
        [self.dc[1][0], self.dc[1][3]]
    }
}

const DEFAULT_INTRA: [u8; 64] = [
    16, 16, 16, 16, 17, 18, 21, 24, 16, 16, 16, 16, 17, 19, 22, 25, 16, 16, 17, 18, 20, 22, 25, 29,
    16, 16, 18, 21, 24, 27, 31, 36, 17, 17, 20, 24, 30, 35, 41, 47, 18, 19, 22, 27, 35, 44, 54, 65,
    21, 22, 25, 31, 41, 54, 70, 88, 24, 25, 29, 36, 47, 65, 88, 115,
];

const DEFAULT_INTER: [u8; 64] = [
    16, 16, 16, 16, 17, 18, 20, 24, 16, 16, 16, 17, 18, 20, 24, 25, 16, 16, 17, 18, 20, 24, 25, 28,
    16, 17, 18, 20, 24, 25, 28, 33, 17, 18, 20, 24, 25, 28, 33, 41, 18, 20, 24, 25, 28, 33, 41, 54,
    20, 24, 25, 28, 33, 41, 54, 71, 24, 25, 28, 33, 41, 54, 71, 91,
];

/// Raster positions of the up-right diagonal scan of a 4×4 block (clause
/// 6.5.3).
const DIAGONAL_4: [u8; 16] = diagonal::<16>(4);
/// Raster positions of the up-right diagonal scan of an 8×8 block.
const DIAGONAL_8: [u8; 64] = diagonal::<64>(8);

/// `profile_tier_level()` (clause 7.3.3): the general profile, tier and
/// level. Sub-layer values are skipped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProfileTierLevel {
    /// `general_profile_space`.
    pub general_profile_space: u8,
    /// `general_tier_flag`.
    pub general_tier_flag: bool,
    /// `general_profile_idc`: 1 Main, 2 Main 10, 3 Main Still Picture, 4
    /// the format range extensions profiles (Main 12 among them).
    pub general_profile_idc: u8,
    /// `general_profile_compatibility_flag[j]` at bit `31 - j`, as coded.
    pub general_profile_compatibility_flags: u32,
    /// `general_progressive_source_flag`.
    pub general_progressive_source_flag: bool,
    /// `general_interlaced_source_flag`.
    pub general_interlaced_source_flag: bool,
    /// `general_non_packed_constraint_flag`.
    pub general_non_packed_constraint_flag: bool,
    /// `general_frame_only_constraint_flag`.
    pub general_frame_only_constraint_flag: bool,
    /// The 43 bits after `general_frame_only_constraint_flag`
    /// (`general_max_12bit_constraint_flag` and the rest, or reserved) and
    /// the bit after them (`general_inbld_flag` or reserved), in the low 44
    /// bits, the first coded bit highest.
    pub general_constraint_flags: u64,
    /// `general_level_idc`: 30 times the level number.
    pub general_level_idc: u8,
}

/// A sequence parameter set, `seq_parameter_set_rbsp()` (clause 7.3.2.2).
///
/// Only 4:2:0 at equal luma and chroma depths of 8 to 12 bits with no
/// range-extension, multilayer, 3D or screen-content tool parses; anything
/// else is refused as [`CodecError::Unsupported`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sps {
    /// `sps_video_parameter_set_id`.
    pub sps_video_parameter_set_id: u8,
    /// `sps_max_sub_layers_minus1`.
    pub sps_max_sub_layers_minus1: u8,
    /// `sps_temporal_id_nesting_flag`.
    pub sps_temporal_id_nesting_flag: bool,
    /// `profile_tier_level(1, sps_max_sub_layers_minus1)`.
    pub profile_tier_level: ProfileTierLevel,
    /// `sps_seq_parameter_set_id`.
    pub sps_seq_parameter_set_id: u8,
    /// `chroma_format_idc`: always 1 (4:2:0).
    pub chroma_format_idc: u8,
    /// `separate_colour_plane_flag`: always 0.
    pub separate_colour_plane_flag: bool,
    /// `pic_width_in_luma_samples`.
    pub pic_width_in_luma_samples: u32,
    /// `pic_height_in_luma_samples`.
    pub pic_height_in_luma_samples: u32,
    /// `conformance_window_flag`.
    pub conformance_window_flag: bool,
    /// `conf_win_left_offset`, in units of `SubWidthC` luma samples.
    /// [`Self::crop`] gives the window in luma samples.
    pub conf_win_left_offset: u32,
    /// `conf_win_right_offset`.
    pub conf_win_right_offset: u32,
    /// `conf_win_top_offset`, in units of `SubHeightC` luma rows.
    pub conf_win_top_offset: u32,
    /// `conf_win_bottom_offset`.
    pub conf_win_bottom_offset: u32,
    /// `bit_depth_luma_minus8`.
    pub bit_depth_luma_minus8: u8,
    /// `bit_depth_chroma_minus8`: always equal to `bit_depth_luma_minus8`.
    pub bit_depth_chroma_minus8: u8,
    /// `log2_max_pic_order_cnt_lsb_minus4`.
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    /// `sps_sub_layer_ordering_info_present_flag`.
    pub sps_sub_layer_ordering_info_present_flag: bool,
    /// `sps_max_dec_pic_buffering_minus1[i]` for each sub-layer up to
    /// `sps_max_sub_layers_minus1`, those not coded inferred (clause
    /// 7.4.3.2.1); 0 past the last sub-layer. [`Self::max_dec_pic_buffering`]
    /// gives the value of the highest sub-layer.
    pub sps_max_dec_pic_buffering_minus1: [u8; 7],
    /// `sps_max_num_reorder_pics[i]`, as `sps_max_dec_pic_buffering_minus1`.
    pub sps_max_num_reorder_pics: [u8; 7],
    /// `sps_max_latency_increase_plus1[i]`, as
    /// `sps_max_dec_pic_buffering_minus1`.
    pub sps_max_latency_increase_plus1: [u32; 7],
    /// `log2_min_luma_coding_block_size_minus3`.
    pub log2_min_luma_coding_block_size_minus3: u8,
    /// `log2_diff_max_min_luma_coding_block_size`.
    pub log2_diff_max_min_luma_coding_block_size: u8,
    /// `log2_min_luma_transform_block_size_minus2`.
    pub log2_min_luma_transform_block_size_minus2: u8,
    /// `log2_diff_max_min_luma_transform_block_size`.
    pub log2_diff_max_min_luma_transform_block_size: u8,
    /// `max_transform_hierarchy_depth_inter`.
    pub max_transform_hierarchy_depth_inter: u8,
    /// `max_transform_hierarchy_depth_intra`.
    pub max_transform_hierarchy_depth_intra: u8,
    /// `scaling_list_enabled_flag`.
    pub scaling_list_enabled_flag: bool,
    /// `sps_scaling_list_data_present_flag`.
    pub sps_scaling_list_data_present_flag: bool,
    /// The SPS scaling lists when `scaling_list_enabled_flag` is 1: its
    /// `scaling_list_data()`, or the default lists when
    /// `sps_scaling_list_data_present_flag` is 0. [`ScalingList::active`]
    /// picks the lists a picture uses.
    pub scaling_list: Option<ScalingList>,
    /// `amp_enabled_flag`.
    pub amp_enabled_flag: bool,
    /// `sample_adaptive_offset_enabled_flag`.
    pub sample_adaptive_offset_enabled_flag: bool,
    /// `pcm_enabled_flag`.
    pub pcm_enabled_flag: bool,
    /// The PCM parameters when `pcm_enabled_flag` is 1.
    pub pcm: Option<Pcm>,
    /// `num_short_term_ref_pic_sets`.
    pub num_short_term_ref_pic_sets: u8,
    /// `st_ref_pic_set(i)` for `i` below `num_short_term_ref_pic_sets`.
    pub st_ref_pic_set: Vec<StRps>,
    /// `long_term_ref_pics_present_flag`.
    pub long_term_ref_pics_present_flag: bool,
    /// `num_long_term_ref_pics_sps`.
    pub num_long_term_ref_pics_sps: u8,
    /// `lt_ref_pic_poc_lsb_sps[i]`.
    pub lt_ref_pic_poc_lsb_sps: Vec<u32>,
    /// `used_by_curr_pic_lt_sps_flag[i]`.
    pub used_by_curr_pic_lt_sps_flag: Vec<bool>,
    /// `sps_temporal_mvp_enabled_flag`.
    pub sps_temporal_mvp_enabled_flag: bool,
    /// `strong_intra_smoothing_enabled_flag`.
    pub strong_intra_smoothing_enabled_flag: bool,
    /// `vui_parameters_present_flag`.
    pub vui_parameters_present_flag: bool,
    /// `vui_parameters()`, the inferred values when not present.
    pub vui: Vui,
}

/// The PCM parameters of an SPS (clause 7.3.2.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pcm {
    /// `pcm_sample_bit_depth_luma_minus1`.
    pub pcm_sample_bit_depth_luma_minus1: u8,
    /// `pcm_sample_bit_depth_chroma_minus1`.
    pub pcm_sample_bit_depth_chroma_minus1: u8,
    /// `log2_min_pcm_luma_coding_block_size_minus3`.
    pub log2_min_pcm_luma_coding_block_size_minus3: u8,
    /// `log2_diff_max_min_pcm_luma_coding_block_size`.
    pub log2_diff_max_min_pcm_luma_coding_block_size: u8,
    /// `pcm_loop_filter_disabled_flag`.
    pub pcm_loop_filter_disabled_flag: bool,
}

impl Pcm {
    /// `PcmBitDepthY`.
    pub fn bit_depth(&self) -> u8 {
        self.pcm_sample_bit_depth_luma_minus1 + 1
    }

    /// `PcmBitDepthC`.
    pub fn bit_depth_chroma(&self) -> u8 {
        self.pcm_sample_bit_depth_chroma_minus1 + 1
    }

    /// `Log2MinIpcmCbSizeY`.
    pub fn log2_min(&self) -> u8 {
        self.log2_min_pcm_luma_coding_block_size_minus3 + 3
    }

    /// `Log2MaxIpcmCbSizeY`.
    pub fn log2_max(&self) -> u8 {
        self.log2_min() + self.log2_diff_max_min_pcm_luma_coding_block_size
    }
}

/// The parts of `vui_parameters()` (clause E.2.1) this decoder keeps: aspect
/// ratio, colour description, chroma sample location and timing. A syntax
/// element not coded holds its inferred value (clause E.3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vui {
    /// `aspect_ratio_info_present_flag`.
    pub aspect_ratio_info_present_flag: bool,
    /// `aspect_ratio_idc`.
    pub aspect_ratio_idc: u8,
    /// `sar_width`.
    pub sar_width: u16,
    /// `sar_height`.
    pub sar_height: u16,
    /// `video_signal_type_present_flag`.
    pub video_signal_type_present_flag: bool,
    /// `video_format` (5, unspecified, when not coded).
    pub video_format: u8,
    /// `video_full_range_flag`.
    pub video_full_range_flag: bool,
    /// `colour_description_present_flag`.
    pub colour_description_present_flag: bool,
    /// `colour_primaries` (2, unspecified, when not coded).
    pub colour_primaries: u8,
    /// `transfer_characteristics` (2 when not coded).
    pub transfer_characteristics: u8,
    /// `matrix_coeffs` (2 when not coded).
    pub matrix_coeffs: u8,
    /// `chroma_loc_info_present_flag`.
    pub chroma_loc_info_present_flag: bool,
    /// `chroma_sample_loc_type_top_field`.
    pub chroma_sample_loc_type_top_field: u32,
    /// `chroma_sample_loc_type_bottom_field`.
    pub chroma_sample_loc_type_bottom_field: u32,
    /// `vui_timing_info_present_flag`.
    pub vui_timing_info_present_flag: bool,
    /// `vui_num_units_in_tick`.
    pub vui_num_units_in_tick: u32,
    /// `vui_time_scale`.
    pub vui_time_scale: u32,
}

impl Vui {
    fn unspecified() -> Self {
        Self {
            aspect_ratio_info_present_flag: false,
            aspect_ratio_idc: 0,
            sar_width: 0,
            sar_height: 0,
            video_signal_type_present_flag: false,
            video_format: 5,
            video_full_range_flag: false,
            colour_description_present_flag: false,
            colour_primaries: 2,
            transfer_characteristics: 2,
            matrix_coeffs: 2,
            chroma_loc_info_present_flag: false,
            chroma_sample_loc_type_top_field: 0,
            chroma_sample_loc_type_bottom_field: 0,
            vui_timing_info_present_flag: false,
            vui_num_units_in_tick: 0,
            vui_time_scale: 0,
        }
    }

    /// The sample range `video_full_range_flag` names.
    pub fn range(&self) -> ColorRange {
        if self.video_full_range_flag {
            ColorRange::Full
        } else {
            ColorRange::Limited
        }
    }

    /// The colour primaries `colour_primaries` names.
    pub fn primaries(&self) -> ColorPrimaries {
        primaries_of(self.colour_primaries)
    }

    /// The transfer function `transfer_characteristics` names.
    pub fn transfer(&self) -> Transfer {
        transfer_of(self.transfer_characteristics)
    }

    /// The matrix `matrix_coeffs` names.
    pub fn matrix(&self) -> ColorMatrix {
        matrix_of(self.matrix_coeffs)
    }

    /// The sample aspect ratio (Table E.1), unspecified when not coded.
    pub fn sample_aspect(&self) -> SampleAspect {
        if !self.aspect_ratio_info_present_flag {
            SampleAspect::UNSPECIFIED
        } else if self.aspect_ratio_idc == 255 {
            SampleAspect::ratio(u32::from(self.sar_width), u32::from(self.sar_height))
        } else {
            aspect(self.aspect_ratio_idc)
        }
    }
}

impl Sps {
    /// `BitDepthY`.
    pub fn bit_depth(&self) -> u8 {
        self.bit_depth_luma_minus8 + 8
    }

    /// `BitDepthC`.
    pub fn bit_depth_chroma(&self) -> u8 {
        self.bit_depth_chroma_minus8 + 8
    }

    /// `log2(MaxPicOrderCntLsb)`.
    pub fn log2_max_poc_lsb(&self) -> u8 {
        self.log2_max_pic_order_cnt_lsb_minus4 + 4
    }

    /// `MinCbLog2SizeY`.
    pub fn log2_min_cb(&self) -> u8 {
        self.log2_min_luma_coding_block_size_minus3 + 3
    }

    /// `CtbLog2SizeY`.
    pub fn log2_ctb(&self) -> u8 {
        self.log2_min_cb() + self.log2_diff_max_min_luma_coding_block_size
    }

    /// `MinTbLog2SizeY`.
    pub fn log2_min_tb(&self) -> u8 {
        self.log2_min_luma_transform_block_size_minus2 + 2
    }

    /// `MaxTbLog2SizeY`.
    pub fn log2_max_tb(&self) -> u8 {
        self.log2_min_tb() + self.log2_diff_max_min_luma_transform_block_size
    }

    /// `CtbSizeY`.
    pub fn ctb_size(&self) -> u32 {
        1 << self.log2_ctb()
    }

    /// `PicWidthInCtbsY`.
    pub fn ctb_width(&self) -> u32 {
        self.pic_width_in_luma_samples.div_ceil(self.ctb_size())
    }

    /// `PicHeightInCtbsY`.
    pub fn ctb_height(&self) -> u32 {
        self.pic_height_in_luma_samples.div_ceil(self.ctb_size())
    }

    /// `QpBdOffsetY`.
    pub fn qp_bd_offset(&self) -> i32 {
        6 * i32::from(self.bit_depth_luma_minus8)
    }

    /// `QpBdOffsetC`.
    pub fn qp_bd_offset_c(&self) -> i32 {
        6 * i32::from(self.bit_depth_chroma_minus8)
    }

    /// The conformance window in luma samples: left, right, top, bottom
    /// (`SubWidthC * conf_win_left_offset` and so on).
    pub fn crop(&self) -> [u32; 4] {
        let (sub_w, sub_h) = match self.chroma_format_idc {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        };
        [
            self.conf_win_left_offset.saturating_mul(sub_w),
            self.conf_win_right_offset.saturating_mul(sub_w),
            self.conf_win_top_offset.saturating_mul(sub_h),
            self.conf_win_bottom_offset.saturating_mul(sub_h),
        ]
    }

    /// `sps_max_dec_pic_buffering_minus1 + 1` of the highest sub-layer: the
    /// pictures the decoded picture buffer holds, the current one included.
    pub fn max_dec_pic_buffering(&self) -> usize {
        let top = usize::from(self.sps_max_sub_layers_minus1).min(6);
        usize::from(self.sps_max_dec_pic_buffering_minus1[top]) + 1
    }

    /// `sps_max_num_reorder_pics` of the highest sub-layer.
    pub fn max_num_reorder(&self) -> usize {
        let top = usize::from(self.sps_max_sub_layers_minus1).min(6);
        usize::from(self.sps_max_num_reorder_pics[top])
    }
}

pub fn parse_sps(rbsp: &[u8]) -> Result<Sps, CodecError> {
    let mut bits = Bits::new(rbsp);
    let sps_video_parameter_set_id = bits.u(4)? as u8;
    let sps_max_sub_layers_minus1 = bits.u(3)? as u8;
    if sps_max_sub_layers_minus1 > 6 {
        return Err(bits.malformed("sps_max_sub_layers_minus1"));
    }
    let max_sub_layers = sps_max_sub_layers_minus1 + 1;
    let sps_temporal_id_nesting_flag = bits.bit()?;
    let profile_tier_level = profile_tier_level(&mut bits, max_sub_layers)?;
    let id = bits.ue_max(15, "sps_seq_parameter_set_id")? as u8;
    let chroma_format_idc = bits.ue_max(3, "chroma_format_idc")? as u8;
    if chroma_format_idc == 3 && bits.bit()? {
        return Err(CodecError::Unsupported {
            what: "separate colour planes",
        });
    }
    let width = bits.ue()?;
    let height = bits.ue()?;
    if width == 0 || height == 0 {
        return Err(bits.malformed("picture size"));
    }
    if width > MAX_EDGE
        || height > MAX_EDGE
        || u64::from(width) * u64::from(height) > MAX_LUMA_SAMPLES
    {
        return Err(CodecError::Limit {
            what: "picture size",
        });
    }
    let mut window = [0u32; 4];
    let conformance_window_flag = bits.bit()?;
    if conformance_window_flag {
        let (sub_w, sub_h) = match chroma_format_idc {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        };
        let mut crop = [0u32; 4];
        for (n, (edge, coded)) in crop.iter_mut().zip(window.iter_mut()).enumerate() {
            let unit = if n < 2 { sub_w } else { sub_h };
            *coded = bits.ue()?;
            *edge = coded.saturating_mul(unit);
        }
        if crop[0].saturating_add(crop[1]) >= width || crop[2].saturating_add(crop[3]) >= height {
            return Err(bits.malformed("conformance window"));
        }
    }
    let bit_depth_luma_minus8 = bits.ue_max(8, "bit_depth_luma_minus8")? as u8;
    let bit_depth_chroma_minus8 = bits.ue_max(8, "bit_depth_chroma_minus8")? as u8;
    let (bit_depth, bit_depth_chroma) = (bit_depth_luma_minus8 + 8, bit_depth_chroma_minus8 + 8);
    let log2_max_pic_order_cnt_lsb_minus4 =
        bits.ue_max(12, "log2_max_pic_order_cnt_lsb_minus4")? as u8;
    let log2_max_poc_lsb = log2_max_pic_order_cnt_lsb_minus4 + 4;
    let ordering_all = bits.bit()?;
    let mut max_dec_pic_buffering_minus1 = [0u8; 7];
    let mut max_num_reorder = [0u8; 7];
    let mut max_latency_increase_plus1 = [0u32; 7];
    let first = if ordering_all { 0 } else { max_sub_layers - 1 };
    for i in first..max_sub_layers {
        let i = usize::from(i);
        let dec = bits.ue_max(MAX_DPB as u32 - 1, "sps_max_dec_pic_buffering_minus1")? + 1;
        let reorder = bits.ue_max(dec - 1, "sps_max_num_reorder_pics")?;
        max_dec_pic_buffering_minus1[i] = (dec - 1) as u8;
        max_num_reorder[i] = reorder as u8;
        max_latency_increase_plus1[i] = bits.ue()?;
    }
    for i in 0..usize::from(first) {
        let last = usize::from(first);
        max_dec_pic_buffering_minus1[i] = max_dec_pic_buffering_minus1[last];
        max_num_reorder[i] = max_num_reorder[last];
        max_latency_increase_plus1[i] = max_latency_increase_plus1[last];
    }
    let log2_min_cb_minus3 = bits.ue_max(3, "log2_min_luma_coding_block_size_minus3")? as u8;
    let log2_diff_cb = bits.ue_max(3, "log2_diff_max_min_luma_coding_block_size")? as u8;
    let log2_min_tb_minus2 = bits.ue_max(3, "log2_min_luma_transform_block_size_minus2")? as u8;
    let log2_diff_tb = bits.ue_max(3, "log2_diff_max_min_luma_transform_block_size")? as u8;
    let log2_min_cb = log2_min_cb_minus3 + 3;
    let log2_ctb = log2_min_cb + log2_diff_cb;
    let log2_min_tb = log2_min_tb_minus2 + 2;
    let log2_max_tb = log2_min_tb + log2_diff_tb;
    if !(4..=6).contains(&log2_ctb) || log2_min_tb >= log2_min_cb || log2_max_tb > log2_ctb.min(5) {
        return Err(bits.malformed("block sizes"));
    }
    if width % (1 << log2_min_cb) != 0 || height % (1 << log2_min_cb) != 0 {
        return Err(bits.malformed("picture size is not a multiple of the coding block"));
    }
    let depth_max = u32::from(log2_ctb - log2_min_tb);
    let max_th_depth_inter = bits.ue_max(depth_max, "max_transform_hierarchy_depth_inter")? as u8;
    let max_th_depth_intra = bits.ue_max(depth_max, "max_transform_hierarchy_depth_intra")? as u8;
    let scaling_list_enabled_flag = bits.bit()?;
    let mut sps_scaling_list_data_present_flag = false;
    let scaling_list = if scaling_list_enabled_flag {
        let mut list = ScalingList::default_lists();
        sps_scaling_list_data_present_flag = bits.bit()?;
        if sps_scaling_list_data_present_flag {
            scaling_list_data(&mut bits, &mut list)?;
        }
        Some(list)
    } else {
        None
    };
    let amp = bits.bit()?;
    let sao = bits.bit()?;
    let pcm_enabled_flag = bits.bit()?;
    let pcm = if pcm_enabled_flag {
        let pcm_depth_minus1 = bits.u(4)? as u8;
        let pcm_depth_c_minus1 = bits.u(4)? as u8;
        let log2_min_minus3 = bits.ue_max(2, "log2_min_pcm_luma_coding_block_size_minus3")? as u8;
        let log2_diff = bits.ue_max(2, "log2_diff_max_min_pcm_luma_coding_block_size")? as u8;
        let pcm = Pcm {
            pcm_sample_bit_depth_luma_minus1: pcm_depth_minus1,
            pcm_sample_bit_depth_chroma_minus1: pcm_depth_c_minus1,
            log2_min_pcm_luma_coding_block_size_minus3: log2_min_minus3,
            log2_diff_max_min_pcm_luma_coding_block_size: log2_diff,
            pcm_loop_filter_disabled_flag: false,
        };
        if pcm.bit_depth() > bit_depth
            || pcm.bit_depth_chroma() > bit_depth_chroma
            || pcm.log2_max() > log2_ctb.min(5)
        {
            return Err(bits.malformed("pcm parameters"));
        }
        Some(Pcm {
            pcm_loop_filter_disabled_flag: bits.bit()?,
            ..pcm
        })
    } else {
        None
    };
    let num_st = bits.ue_max(64, "num_short_term_ref_pic_sets")? as usize;
    let mut st_rps: Vec<StRps> = Vec::with_capacity(num_st);
    for index in 0..num_st {
        let rps = parse_st_rps(&mut bits, &st_rps, index, num_st, false)?;
        st_rps.push(rps);
    }
    let long_term_present = bits.bit()?;
    let mut lt_ref_pic_poc_lsb = Vec::new();
    let mut lt_used_by_curr = Vec::new();
    if long_term_present {
        let count = bits.ue_max(32, "num_long_term_ref_pics_sps")?;
        for _ in 0..count {
            lt_ref_pic_poc_lsb.push(bits.u(u32::from(log2_max_poc_lsb))?);
            lt_used_by_curr.push(bits.bit()?);
        }
    }
    let temporal_mvp = bits.bit()?;
    let strong_intra_smoothing = bits.bit()?;
    let vui_parameters_present_flag = bits.bit()?;
    let vui = if vui_parameters_present_flag {
        parse_vui(&mut bits, max_sub_layers)?
    } else {
        Vui::unspecified()
    };
    if bits.bit()? {
        let range = bits.bit()?;
        let multilayer = bits.bit()?;
        let three_d = bits.bit()?;
        let scc = bits.bit()?;
        let _four = bits.u(4)?;
        if range {
            for _ in 0..9 {
                if bits.bit()? {
                    return Err(CodecError::Unsupported {
                        what: "range extension coding tools",
                    });
                }
            }
        }
        if multilayer || three_d || scc {
            return Err(CodecError::Unsupported {
                what: "sps extension",
            });
        }
    }
    if chroma_format_idc != 1 {
        return Err(CodecError::Unsupported {
            what: "chroma format other than 4:2:0",
        });
    }
    if bit_depth != bit_depth_chroma || bit_depth > 12 {
        return Err(CodecError::Unsupported { what: "bit depth" });
    }
    let [left, right, top, bottom] = window;
    Ok(Sps {
        sps_video_parameter_set_id,
        sps_max_sub_layers_minus1,
        sps_temporal_id_nesting_flag,
        profile_tier_level,
        sps_seq_parameter_set_id: id,
        chroma_format_idc,
        separate_colour_plane_flag: false,
        pic_width_in_luma_samples: width,
        pic_height_in_luma_samples: height,
        conformance_window_flag,
        conf_win_left_offset: left,
        conf_win_right_offset: right,
        conf_win_top_offset: top,
        conf_win_bottom_offset: bottom,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
        log2_max_pic_order_cnt_lsb_minus4,
        sps_sub_layer_ordering_info_present_flag: ordering_all,
        sps_max_dec_pic_buffering_minus1: max_dec_pic_buffering_minus1,
        sps_max_num_reorder_pics: max_num_reorder,
        sps_max_latency_increase_plus1: max_latency_increase_plus1,
        log2_min_luma_coding_block_size_minus3: log2_min_cb_minus3,
        log2_diff_max_min_luma_coding_block_size: log2_diff_cb,
        log2_min_luma_transform_block_size_minus2: log2_min_tb_minus2,
        log2_diff_max_min_luma_transform_block_size: log2_diff_tb,
        max_transform_hierarchy_depth_inter: max_th_depth_inter,
        max_transform_hierarchy_depth_intra: max_th_depth_intra,
        scaling_list_enabled_flag,
        sps_scaling_list_data_present_flag,
        scaling_list,
        amp_enabled_flag: amp,
        sample_adaptive_offset_enabled_flag: sao,
        pcm_enabled_flag,
        pcm,
        num_short_term_ref_pic_sets: num_st as u8,
        st_ref_pic_set: st_rps,
        long_term_ref_pics_present_flag: long_term_present,
        num_long_term_ref_pics_sps: lt_ref_pic_poc_lsb.len() as u8,
        lt_ref_pic_poc_lsb_sps: lt_ref_pic_poc_lsb,
        used_by_curr_pic_lt_sps_flag: lt_used_by_curr,
        sps_temporal_mvp_enabled_flag: temporal_mvp,
        strong_intra_smoothing_enabled_flag: strong_intra_smoothing,
        vui_parameters_present_flag,
        vui,
    })
}

/// `profile_space`, `tier_flag`, `profile_idc`, 32 compatibility flags, 4
/// source flags, 43 constraint bits, and one more flag: 88 bits.
fn general_profile(bits: &mut Bits<'_>) -> Result<ProfileTierLevel, CodecError> {
    let general_profile_space = bits.u(2)? as u8;
    let general_tier_flag = bits.bit()?;
    let general_profile_idc = bits.u(5)? as u8;
    let general_profile_compatibility_flags = bits.u(32)?;
    let general_progressive_source_flag = bits.bit()?;
    let general_interlaced_source_flag = bits.bit()?;
    let general_non_packed_constraint_flag = bits.bit()?;
    let general_frame_only_constraint_flag = bits.bit()?;
    let high = u64::from(bits.u(32)?);
    let low = u64::from(bits.u(12)?);
    Ok(ProfileTierLevel {
        general_profile_space,
        general_tier_flag,
        general_profile_idc,
        general_profile_compatibility_flags,
        general_progressive_source_flag,
        general_interlaced_source_flag,
        general_non_packed_constraint_flag,
        general_frame_only_constraint_flag,
        general_constraint_flags: (high << 12) | low,
        general_level_idc: 0,
    })
}

fn profile_tier_level(
    bits: &mut Bits<'_>,
    max_sub_layers: u8,
) -> Result<ProfileTierLevel, CodecError> {
    let mut general = general_profile(bits)?;
    general.general_level_idc = bits.u(8)? as u8;
    let subs = usize::from(max_sub_layers - 1);
    let mut profile_present = [false; 8];
    let mut level_present = [false; 8];
    for i in 0..subs {
        profile_present[i] = bits.bit()?;
        level_present[i] = bits.bit()?;
    }
    if subs > 0 {
        bits.skip(2 * (8 - subs))?;
    }
    for i in 0..subs {
        if profile_present[i] {
            bits.skip(88)?;
        }
        if level_present[i] {
            bits.skip(8)?;
        }
    }
    Ok(general)
}

/// `st_ref_pic_set(stRpsIdx)` of clause 7.3.7, with the inter prediction of
/// clause 7.4.8.
pub fn parse_st_rps(
    bits: &mut Bits<'_>,
    sets: &[StRps],
    index: usize,
    num_sets: usize,
    in_slice: bool,
) -> Result<StRps, CodecError> {
    let predict = index != 0 && bits.bit()?;
    if predict {
        let delta_idx = if in_slice {
            bits.ue_max(index as u32 - 1, "delta_idx_minus1")? as usize + 1
        } else {
            1
        };
        let Some(reference) = index.checked_sub(delta_idx).and_then(|at| sets.get(at)) else {
            return Err(bits.malformed("delta_idx_minus1"));
        };
        let sign = bits.bit()?;
        let abs_minus1 = bits.ue_max(32_767, "abs_delta_rps_minus1")?;
        let abs = abs_minus1 as i32 + 1;
        let delta_rps = if sign { -abs } else { abs };
        let total = reference.num_delta_pocs();
        let mut used = [false; 33];
        let mut use_delta = [true; 33];
        for j in 0..=total {
            used[j] = bits.bit()?;
            if !used[j] {
                use_delta[j] = bits.bit()?;
            }
        }
        let neg_ref = reference.delta_poc_s0.len();
        let mut rps = StRps {
            inter_ref_pic_set_prediction_flag: true,
            delta_idx_minus1: (delta_idx - 1) as u32,
            delta_rps_sign: sign,
            abs_delta_rps_minus1: abs_minus1,
            used_by_curr_pic_flag: used[..=total].to_vec(),
            use_delta_flag: use_delta[..=total].to_vec(),
            ..StRps::default()
        };
        for j in (0..reference.delta_poc_s1.len()).rev() {
            let d = reference.delta_poc_s1[j] + delta_rps;
            if d < 0 && use_delta[neg_ref + j] {
                rps.delta_poc_s0.push(d);
                rps.used_by_curr_pic_s0.push(used[neg_ref + j]);
            }
        }
        if delta_rps < 0 && use_delta[total] {
            rps.delta_poc_s0.push(delta_rps);
            rps.used_by_curr_pic_s0.push(used[total]);
        }
        for j in 0..neg_ref {
            let d = reference.delta_poc_s0[j] + delta_rps;
            if d < 0 && use_delta[j] {
                rps.delta_poc_s0.push(d);
                rps.used_by_curr_pic_s0.push(used[j]);
            }
        }
        for j in (0..neg_ref).rev() {
            let d = reference.delta_poc_s0[j] + delta_rps;
            if d > 0 && use_delta[j] {
                rps.delta_poc_s1.push(d);
                rps.used_by_curr_pic_s1.push(used[j]);
            }
        }
        if delta_rps > 0 && use_delta[total] {
            rps.delta_poc_s1.push(delta_rps);
            rps.used_by_curr_pic_s1.push(used[total]);
        }
        for j in 0..reference.delta_poc_s1.len() {
            let d = reference.delta_poc_s1[j] + delta_rps;
            if d > 0 && use_delta[neg_ref + j] {
                rps.delta_poc_s1.push(d);
                rps.used_by_curr_pic_s1.push(used[neg_ref + j]);
            }
        }
        if rps.num_delta_pocs() > MAX_DPB {
            return Err(bits.malformed("short-term reference picture set"));
        }
        let _ = num_sets;
        Ok(rps)
    } else {
        let negatives = bits.ue_max(MAX_DPB as u32, "num_negative_pics")? as usize;
        let positives = bits.ue_max(MAX_DPB as u32, "num_positive_pics")? as usize;
        if negatives + positives > MAX_DPB {
            return Err(bits.malformed("short-term reference picture set"));
        }
        let mut rps = StRps::default();
        let mut poc = 0i32;
        for _ in 0..negatives {
            poc -= bits.ue_max(32_767, "delta_poc_s0_minus1")? as i32 + 1;
            rps.delta_poc_s0.push(poc);
            rps.used_by_curr_pic_s0.push(bits.bit()?);
        }
        poc = 0;
        for _ in 0..positives {
            poc += bits.ue_max(32_767, "delta_poc_s1_minus1")? as i32 + 1;
            rps.delta_poc_s1.push(poc);
            rps.used_by_curr_pic_s1.push(bits.bit()?);
        }
        Ok(rps)
    }
}

/// Diagonal scan positions of an `edge`×`edge` block, raster index, for
/// `N = edge * edge`.
const fn diagonal<const N: usize>(edge: i32) -> [u8; N] {
    let mut out = [0u8; N];
    let mut at = 0;
    let mut x = 0i32;
    let mut y = 0i32;
    while at < N {
        while y >= 0 {
            if x < edge && y < edge {
                out[at] = (y * edge + x) as u8;
                at += 1;
            }
            y -= 1;
            x += 1;
        }
        y = x;
        x = 0;
    }
    out
}

fn scaling_list_data(bits: &mut Bits<'_>, list: &mut ScalingList) -> Result<(), CodecError> {
    for size in 0..4usize {
        let step = if size == 3 { 3 } else { 1 };
        let mut matrix = 0usize;
        while matrix < 6 {
            if !bits.bit()? {
                let delta = bits.ue_max(5, "scaling_list_pred_matrix_id_delta")? as usize * step;
                if delta > matrix {
                    return Err(bits.malformed("scaling_list_pred_matrix_id_delta"));
                }
                if delta == 0 {
                    list.lists[size][matrix] = if size == 0 {
                        [16; 64]
                    } else if matrix < 3 {
                        DEFAULT_INTRA
                    } else {
                        DEFAULT_INTER
                    };
                    if size > 1 {
                        list.dc[size - 2][matrix] = 16;
                    }
                } else {
                    list.lists[size][matrix] = list.lists[size][matrix - delta];
                    if size > 1 {
                        list.dc[size - 2][matrix] = list.dc[size - 2][matrix - delta];
                    }
                }
            } else {
                let mut next = 8i32;
                if size > 1 {
                    let dc = bits.se_range(-7, 247, "scaling_list_dc_coef_minus8")? + 8;
                    next = dc;
                    list.dc[size - 2][matrix] = dc as u8;
                }
                let scan = if size == 0 {
                    &DIAGONAL_4[..]
                } else {
                    &DIAGONAL_8[..]
                };
                for &pos in scan {
                    let delta = bits.se_range(-128, 127, "scaling_list_delta_coef")?;
                    next = (next + delta + 256).rem_euclid(256);
                    list.lists[size][matrix][usize::from(pos)] = next as u8;
                }
            }
            matrix += step;
        }
    }
    Ok(())
}

fn parse_vui(bits: &mut Bits<'_>, max_sub_layers: u8) -> Result<Vui, CodecError> {
    let mut vui = Vui::unspecified();
    vui.aspect_ratio_info_present_flag = bits.bit()?;
    if vui.aspect_ratio_info_present_flag {
        vui.aspect_ratio_idc = bits.u(8)? as u8;
        if vui.aspect_ratio_idc == 255 {
            vui.sar_width = bits.u(16)? as u16;
            vui.sar_height = bits.u(16)? as u16;
        }
    }
    if bits.bit()? {
        let _overscan_appropriate = bits.bit()?;
    }
    vui.video_signal_type_present_flag = bits.bit()?;
    if vui.video_signal_type_present_flag {
        vui.video_format = bits.u(3)? as u8;
        vui.video_full_range_flag = bits.bit()?;
        vui.colour_description_present_flag = bits.bit()?;
        if vui.colour_description_present_flag {
            vui.colour_primaries = bits.u(8)? as u8;
            vui.transfer_characteristics = bits.u(8)? as u8;
            vui.matrix_coeffs = bits.u(8)? as u8;
        }
    }
    vui.chroma_loc_info_present_flag = bits.bit()?;
    if vui.chroma_loc_info_present_flag {
        vui.chroma_sample_loc_type_top_field = bits.ue()?;
        vui.chroma_sample_loc_type_bottom_field = bits.ue()?;
    }
    let _neutral_chroma = bits.bit()?;
    let _field_seq = bits.bit()?;
    let _frame_field_info = bits.bit()?;
    if bits.bit()? {
        for _ in 0..4 {
            let _offset = bits.ue()?;
        }
    }
    vui.vui_timing_info_present_flag = bits.bit()?;
    if vui.vui_timing_info_present_flag {
        vui.vui_num_units_in_tick = bits.u(32)?;
        vui.vui_time_scale = bits.u(32)?;
        if bits.bit()? {
            let _ticks = bits.ue()?;
        }
        if bits.bit()? {
            hrd_parameters(bits, true, max_sub_layers)?;
        }
    }
    if bits.bit()? {
        let _tiles_fixed = bits.bit()?;
        let _mv_over_boundaries = bits.bit()?;
        let _restricted_lists = bits.bit()?;
        for _ in 0..5 {
            let _ = bits.ue()?;
        }
    }
    Ok(vui)
}

fn hrd_parameters(bits: &mut Bits<'_>, common: bool, max_sub_layers: u8) -> Result<(), CodecError> {
    let mut nal = false;
    let mut vcl = false;
    let mut sub_pic = false;
    if common {
        nal = bits.bit()?;
        vcl = bits.bit()?;
        if nal || vcl {
            sub_pic = bits.bit()?;
            if sub_pic {
                bits.skip(8 + 5 + 1 + 5)?;
            }
            bits.skip(8)?;
            if sub_pic {
                bits.skip(4)?;
            }
            bits.skip(15)?;
        }
    }
    for _ in 0..max_sub_layers {
        let general = bits.bit()?;
        let within = if general { true } else { bits.bit()? };
        let mut low_delay = false;
        if within {
            let _duration = bits.ue()?;
        } else {
            low_delay = bits.bit()?;
        }
        let mut cpb = 1;
        if !low_delay {
            cpb = bits.ue_max(31, "cpb_cnt_minus1")? + 1;
        }
        for present in [nal, vcl] {
            if present {
                for _ in 0..cpb {
                    let _ = bits.ue()?;
                    let _ = bits.ue()?;
                    if sub_pic {
                        let _ = bits.ue()?;
                        let _ = bits.ue()?;
                    }
                    let _cbr = bits.bit()?;
                }
            }
        }
    }
    Ok(())
}

fn aspect(idc: u8) -> SampleAspect {
    const TABLE: [(u32, u32); 17] = [
        (0, 1),
        (1, 1),
        (12, 11),
        (10, 11),
        (16, 11),
        (40, 33),
        (24, 11),
        (20, 11),
        (32, 11),
        (80, 33),
        (18, 11),
        (15, 11),
        (64, 33),
        (160, 99),
        (4, 3),
        (3, 2),
        (2, 1),
    ];
    TABLE
        .get(usize::from(idc))
        .map_or(SampleAspect::UNSPECIFIED, |&(w, h)| {
            SampleAspect::ratio(w, h)
        })
}

fn primaries_of(idc: u8) -> ColorPrimaries {
    match idc {
        1 => ColorPrimaries::Bt709,
        4 => ColorPrimaries::Bt470m,
        5 | 6 => ColorPrimaries::Bt601,
        7 => ColorPrimaries::Smpte240m,
        8 => ColorPrimaries::Film,
        9 => ColorPrimaries::Bt2020,
        10 => ColorPrimaries::Smpte428,
        11 => ColorPrimaries::Smpte431,
        12 => ColorPrimaries::Smpte432,
        22 => ColorPrimaries::Ebu3213,
        _ => ColorPrimaries::Unspecified,
    }
}

fn transfer_of(idc: u8) -> Transfer {
    match idc {
        1 | 6 => Transfer::Bt709,
        4 | 5 => Transfer::Bt601,
        7 => Transfer::Smpte240m,
        8 => Transfer::Linear,
        9 => Transfer::Log100,
        10 => Transfer::LogSqrt,
        11 => Transfer::Iec61966_2_4,
        12 => Transfer::Bt1361,
        13 => Transfer::Srgb,
        14 | 15 => Transfer::Bt2020,
        16 => Transfer::Pq,
        17 => Transfer::Smpte428,
        18 => Transfer::Hlg,
        _ => Transfer::Unspecified,
    }
}

fn matrix_of(idc: u8) -> ColorMatrix {
    match idc {
        0 => ColorMatrix::Identity,
        1 => ColorMatrix::Bt709,
        4 => ColorMatrix::Fcc,
        5 | 6 => ColorMatrix::Bt601,
        7 => ColorMatrix::Smpte240m,
        8 => ColorMatrix::Ycgco,
        9 | 10 => ColorMatrix::Bt2020,
        11 => ColorMatrix::Smpte2085,
        12 => ColorMatrix::ChromaDerived,
        13 => ColorMatrix::ChromaDerivedConstant,
        14 => ColorMatrix::Ictcp,
        15 => ColorMatrix::IptC2,
        16 => ColorMatrix::YcgcoRe,
        17 => ColorMatrix::YcgcoRo,
        _ => ColorMatrix::Unspecified,
    }
}

/// A picture parameter set, `pic_parameter_set_rbsp()` (clause 7.3.2.3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pps {
    /// `pps_pic_parameter_set_id`.
    pub pps_pic_parameter_set_id: u8,
    /// `pps_seq_parameter_set_id`.
    pub pps_seq_parameter_set_id: u8,
    /// `dependent_slice_segments_enabled_flag`.
    pub dependent_slice_segments_enabled_flag: bool,
    /// `output_flag_present_flag`.
    pub output_flag_present_flag: bool,
    /// `num_extra_slice_header_bits`.
    pub num_extra_slice_header_bits: u8,
    /// `sign_data_hiding_enabled_flag`.
    pub sign_data_hiding_enabled_flag: bool,
    /// `cabac_init_present_flag`.
    pub cabac_init_present_flag: bool,
    /// `num_ref_idx_l0_default_active_minus1`.
    pub num_ref_idx_l0_default_active_minus1: u8,
    /// `num_ref_idx_l1_default_active_minus1`.
    pub num_ref_idx_l1_default_active_minus1: u8,
    /// `init_qp_minus26`.
    pub init_qp_minus26: i32,
    /// `constrained_intra_pred_flag`.
    pub constrained_intra_pred_flag: bool,
    /// `transform_skip_enabled_flag`.
    pub transform_skip_enabled_flag: bool,
    /// `cu_qp_delta_enabled_flag`.
    pub cu_qp_delta_enabled_flag: bool,
    /// `diff_cu_qp_delta_depth`.
    pub diff_cu_qp_delta_depth: u8,
    /// `pps_cb_qp_offset`.
    pub pps_cb_qp_offset: i32,
    /// `pps_cr_qp_offset`.
    pub pps_cr_qp_offset: i32,
    /// `pps_slice_chroma_qp_offsets_present_flag`.
    pub pps_slice_chroma_qp_offsets_present_flag: bool,
    /// `weighted_pred_flag`.
    pub weighted_pred_flag: bool,
    /// `weighted_bipred_flag`.
    pub weighted_bipred_flag: bool,
    /// `transquant_bypass_enabled_flag`.
    pub transquant_bypass_enabled_flag: bool,
    /// `tiles_enabled_flag`.
    pub tiles_enabled_flag: bool,
    /// `entropy_coding_sync_enabled_flag`.
    pub entropy_coding_sync_enabled_flag: bool,
    /// `num_tile_columns_minus1` (0 without tiles).
    pub num_tile_columns_minus1: u32,
    /// `num_tile_rows_minus1` (0 without tiles).
    pub num_tile_rows_minus1: u32,
    /// `uniform_spacing_flag` (1 without tiles).
    pub uniform_spacing_flag: bool,
    /// `column_width_minus1[i]`, `num_tile_columns_minus1` entries when
    /// `uniform_spacing_flag` is 0, else empty. [`Self::column_widths`]
    /// derives every width.
    pub column_width_minus1: Vec<u32>,
    /// `row_height_minus1[i]`, as `column_width_minus1`.
    /// [`Self::row_heights`] derives every height.
    pub row_height_minus1: Vec<u32>,
    /// `loop_filter_across_tiles_enabled_flag` (1 without tiles).
    pub loop_filter_across_tiles_enabled_flag: bool,
    /// `pps_loop_filter_across_slices_enabled_flag`.
    pub pps_loop_filter_across_slices_enabled_flag: bool,
    /// `deblocking_filter_control_present_flag`.
    pub deblocking_filter_control_present_flag: bool,
    /// `deblocking_filter_override_enabled_flag`.
    pub deblocking_filter_override_enabled_flag: bool,
    /// `pps_deblocking_filter_disabled_flag`.
    pub pps_deblocking_filter_disabled_flag: bool,
    /// `pps_beta_offset_div2`.
    pub pps_beta_offset_div2: i32,
    /// `pps_tc_offset_div2`.
    pub pps_tc_offset_div2: i32,
    /// `pps_scaling_list_data_present_flag`.
    pub pps_scaling_list_data_present_flag: bool,
    /// The PPS's `scaling_list_data()` when
    /// `pps_scaling_list_data_present_flag` is 1. [`ScalingList::active`]
    /// picks the lists a picture uses.
    pub scaling_list: Option<ScalingList>,
    /// `lists_modification_present_flag`.
    pub lists_modification_present_flag: bool,
    /// `log2_parallel_merge_level_minus2`.
    pub log2_parallel_merge_level_minus2: u8,
    /// `slice_segment_header_extension_present_flag`.
    pub slice_segment_header_extension_present_flag: bool,
}

impl Pps {
    /// `colWidth[i]` (equations 6-3 and 6-4): the width of each tile
    /// column in CTBs, derived also when `uniform_spacing_flag` is 1; one
    /// column of the whole picture without tiles.
    pub fn column_widths(&self, sps: &Sps) -> Result<Vec<u32>, CodecError> {
        spacing(
            self.num_tile_columns_minus1,
            &self.column_width_minus1,
            sps.ctb_width(),
            self.uniform_spacing_flag || !self.tiles_enabled_flag,
        )
    }

    /// `rowHeight[j]` (equations 6-5 and 6-6), as [`Self::column_widths`].
    pub fn row_heights(&self, sps: &Sps) -> Result<Vec<u32>, CodecError> {
        spacing(
            self.num_tile_rows_minus1,
            &self.row_height_minus1,
            sps.ctb_height(),
            self.uniform_spacing_flag || !self.tiles_enabled_flag,
        )
    }

    /// `Log2ParMrgLevel`.
    pub fn log2_parallel_merge_level(&self) -> u8 {
        self.log2_parallel_merge_level_minus2 + 2
    }
}

pub fn parse_pps(rbsp: &[u8]) -> Result<Pps, CodecError> {
    let mut bits = Bits::new(rbsp);
    let id = bits.ue_max(63, "pps_pic_parameter_set_id")? as u8;
    let sps_id = bits.ue_max(15, "pps_seq_parameter_set_id")? as u8;
    let dependent_slices = bits.bit()?;
    let output_flag_present = bits.bit()?;
    let extra_slice_header_bits = bits.u(3)? as u8;
    let sign_data_hiding = bits.bit()?;
    let cabac_init_present = bits.bit()?;
    let l0 = bits.ue_max(14, "num_ref_idx_l0_default_active_minus1")? as u8;
    let l1 = bits.ue_max(14, "num_ref_idx_l1_default_active_minus1")? as u8;
    let init_qp_minus26 = bits.se_range(-62, 25, "init_qp_minus26")?;
    let constrained_intra_pred = bits.bit()?;
    let transform_skip = bits.bit()?;
    let cu_qp_delta = bits.bit()?;
    let diff_cu_qp_delta_depth = if cu_qp_delta {
        bits.ue_max(3, "diff_cu_qp_delta_depth")? as u8
    } else {
        0
    };
    let cb_qp_offset = bits.se_range(-12, 12, "pps_cb_qp_offset")?;
    let cr_qp_offset = bits.se_range(-12, 12, "pps_cr_qp_offset")?;
    let slice_chroma_qp_offsets = bits.bit()?;
    let weighted_pred = bits.bit()?;
    let weighted_bipred = bits.bit()?;
    let transquant_bypass = bits.bit()?;
    let tiles = bits.bit()?;
    let entropy_sync = bits.bit()?;
    let mut num_tile_columns_minus1 = 0;
    let mut num_tile_rows_minus1 = 0;
    let mut column_width_minus1 = Vec::new();
    let mut row_height_minus1 = Vec::new();
    let mut uniform_spacing = true;
    let mut loop_filter_across_tiles = true;
    if tiles {
        num_tile_columns_minus1 = bits.ue_max(MAX_EDGE / 16, "num_tile_columns_minus1")?;
        num_tile_rows_minus1 = bits.ue_max(MAX_EDGE / 16, "num_tile_rows_minus1")?;
        uniform_spacing = bits.bit()?;
        if !uniform_spacing {
            for _ in 0..num_tile_columns_minus1 {
                column_width_minus1.push(bits.ue_max(MAX_EDGE, "column_width_minus1")?);
            }
            for _ in 0..num_tile_rows_minus1 {
                row_height_minus1.push(bits.ue_max(MAX_EDGE, "row_height_minus1")?);
            }
        }
        loop_filter_across_tiles = bits.bit()?;
    }
    let loop_filter_across_slices = bits.bit()?;
    let deblocking_filter_control_present_flag = bits.bit()?;
    let mut deblocking_override_enabled = false;
    let mut deblocking_disabled = false;
    let mut beta_offset_div2 = 0;
    let mut tc_offset_div2 = 0;
    if deblocking_filter_control_present_flag {
        deblocking_override_enabled = bits.bit()?;
        deblocking_disabled = bits.bit()?;
        if !deblocking_disabled {
            beta_offset_div2 = bits.se_range(-6, 6, "pps_beta_offset_div2")?;
            tc_offset_div2 = bits.se_range(-6, 6, "pps_tc_offset_div2")?;
        }
    }
    let pps_scaling_list_data_present_flag = bits.bit()?;
    let scaling_list = if pps_scaling_list_data_present_flag {
        let mut list = ScalingList::default_lists();
        scaling_list_data(&mut bits, &mut list)?;
        Some(list)
    } else {
        None
    };
    let lists_modification = bits.bit()?;
    let log2_parallel_merge_level_minus2 =
        bits.ue_max(4, "log2_parallel_merge_level_minus2")? as u8;
    let slice_header_extension = bits.bit()?;
    if bits.bit()? {
        let range = bits.bit()?;
        let multilayer = bits.bit()?;
        let three_d = bits.bit()?;
        let scc = bits.bit()?;
        let _four = bits.u(4)?;
        if range {
            if transform_skip && bits.ue()? != 0 {
                return Err(CodecError::Unsupported {
                    what: "transform skip larger than 4x4",
                });
            }
            if bits.bit()? || bits.bit()? {
                return Err(CodecError::Unsupported {
                    what: "range extension picture tools",
                });
            }
            if bits.ue()? != 0 || bits.ue()? != 0 {
                return Err(CodecError::Unsupported {
                    what: "sao offset scale",
                });
            }
        }
        if multilayer || three_d || scc {
            return Err(CodecError::Unsupported {
                what: "pps extension",
            });
        }
    }
    Ok(Pps {
        pps_pic_parameter_set_id: id,
        pps_seq_parameter_set_id: sps_id,
        dependent_slice_segments_enabled_flag: dependent_slices,
        output_flag_present_flag: output_flag_present,
        num_extra_slice_header_bits: extra_slice_header_bits,
        sign_data_hiding_enabled_flag: sign_data_hiding,
        cabac_init_present_flag: cabac_init_present,
        num_ref_idx_l0_default_active_minus1: l0,
        num_ref_idx_l1_default_active_minus1: l1,
        init_qp_minus26,
        constrained_intra_pred_flag: constrained_intra_pred,
        transform_skip_enabled_flag: transform_skip,
        cu_qp_delta_enabled_flag: cu_qp_delta,
        diff_cu_qp_delta_depth,
        pps_cb_qp_offset: cb_qp_offset,
        pps_cr_qp_offset: cr_qp_offset,
        pps_slice_chroma_qp_offsets_present_flag: slice_chroma_qp_offsets,
        weighted_pred_flag: weighted_pred,
        weighted_bipred_flag: weighted_bipred,
        transquant_bypass_enabled_flag: transquant_bypass,
        tiles_enabled_flag: tiles,
        entropy_coding_sync_enabled_flag: entropy_sync,
        num_tile_columns_minus1,
        num_tile_rows_minus1,
        uniform_spacing_flag: uniform_spacing,
        column_width_minus1,
        row_height_minus1,
        loop_filter_across_tiles_enabled_flag: loop_filter_across_tiles,
        pps_loop_filter_across_slices_enabled_flag: loop_filter_across_slices,
        deblocking_filter_control_present_flag,
        deblocking_filter_override_enabled_flag: deblocking_override_enabled,
        pps_deblocking_filter_disabled_flag: deblocking_disabled,
        pps_beta_offset_div2: beta_offset_div2,
        pps_tc_offset_div2: tc_offset_div2,
        pps_scaling_list_data_present_flag,
        scaling_list,
        lists_modification_present_flag: lists_modification,
        log2_parallel_merge_level_minus2,
        slice_segment_header_extension_present_flag: slice_header_extension,
    })
}

/// The CTB scan of one SPS and PPS pair (clause 6.5.1).
#[derive(Clone, Debug)]
pub struct Layout {
    pub ctb_width: u32,
    pub ctb_height: u32,
    pub log2_ctb: u8,
    pub rs_to_ts: Vec<u32>,
    pub ts_to_rs: Vec<u32>,
    /// Tile index of each CTB, raster order.
    pub tile_of: Vec<u16>,
    /// Column boundaries in CTBs, `columns + 1` entries.
    pub col_bd: Vec<u32>,
    /// Z-scan order of the minimum transform blocks inside a CTB, by row
    /// then column (clause 6.5.2), and their count per edge.
    pub zorder: Vec<u16>,
    pub zorder_edge: usize,
    /// `Log2MinTrafoSize`.
    pub log2_min_tb: u8,
}

impl Layout {
    pub fn new(sps: &Sps, pps: &Pps) -> Result<Self, CodecError> {
        let ctb_width = sps.ctb_width();
        let ctb_height = sps.ctb_height();
        let widths = pps.column_widths(sps)?;
        let heights = pps.row_heights(sps)?;
        let mut col_bd = vec![0u32];
        for w in &widths {
            col_bd.push(col_bd[col_bd.len() - 1] + w);
        }
        let mut row_bd = vec![0u32];
        for h in &heights {
            row_bd.push(row_bd[row_bd.len() - 1] + h);
        }
        let total = (ctb_width * ctb_height) as usize;
        let mut rs_to_ts = vec![0u32; total];
        let mut ts_to_rs = vec![0u32; total];
        let mut tile_of = vec![0u16; total];
        let mut ts = 0u32;
        let mut tile = 0u16;
        for tr in 0..heights.len() {
            for tc in 0..widths.len() {
                for y in row_bd[tr]..row_bd[tr + 1] {
                    for x in col_bd[tc]..col_bd[tc + 1] {
                        let rs = (y * ctb_width + x) as usize;
                        rs_to_ts[rs] = ts;
                        ts_to_rs[ts as usize] = rs as u32;
                        tile_of[rs] = tile;
                        ts += 1;
                    }
                }
                tile = tile.saturating_add(1);
            }
        }
        let log2_ctb = sps.log2_ctb();
        let log2_min_tb = sps.log2_min_tb();
        let edge_log2 = u32::from(log2_ctb.saturating_sub(log2_min_tb));
        let edge = 1usize << edge_log2;
        let mut zorder = vec![0u16; edge * edge];
        for (i, z) in zorder.iter_mut().enumerate() {
            let (tx, ty) = (i % edge, i / edge);
            let mut morton = 0usize;
            for bit in 0..edge_log2 {
                morton |= ((tx >> bit) & 1) << (2 * bit);
                morton |= ((ty >> bit) & 1) << (2 * bit + 1);
            }
            *z = morton as u16;
        }
        Ok(Self {
            ctb_width,
            ctb_height,
            log2_ctb,
            rs_to_ts,
            ts_to_rs,
            tile_of,
            col_bd,
            zorder,
            zorder_edge: edge,
            log2_min_tb,
        })
    }

    pub fn ctb_count(&self) -> u32 {
        self.ctb_width * self.ctb_height
    }

    /// First CTB column of the tile column holding column `x`.
    pub fn tile_col_start(&self, x: u32) -> u32 {
        let at = self.col_bd.partition_point(|&bd| bd <= x);
        self.col_bd[at.saturating_sub(1)]
    }
}

/// Tile column widths or row heights in CTBs: `count_minus1 + 1` of them
/// over `total` CTBs, evenly spread or as coded (`minus1`, one fewer, the
/// last taking the rest).
fn spacing(
    count_minus1: u32,
    minus1: &[u32],
    total: u32,
    uniform: bool,
) -> Result<Vec<u32>, CodecError> {
    let count = count_minus1.saturating_add(1);
    if count > total {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "tile count",
        });
    }
    if uniform {
        return Ok((0..count)
            .map(|i| ((i + 1) * total) / count - (i * total) / count)
            .collect());
    }
    let mut out: Vec<u32> = minus1.iter().map(|&w| w + 1).collect();
    let sum: u32 = out.iter().fold(0u32, |acc, &w| acc.saturating_add(w));
    if sum >= total {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "tile sizes",
        });
    }
    out.push(total - sum);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{Bits, DIAGONAL_4, ScalingList, StRps, parse_st_rps};

    #[test]
    fn diagonal_scan_runs_up_right() {
        assert_eq!(
            DIAGONAL_4,
            [0, 4, 1, 8, 5, 2, 12, 9, 6, 3, 13, 10, 7, 14, 11, 15]
        );
    }

    #[test]
    fn coded_order_undoes_the_raster_placement() {
        let mut list = ScalingList::default_lists();
        for (i, &pos) in DIAGONAL_4.iter().enumerate() {
            list.lists[0][2][usize::from(pos)] = i as u8;
        }
        let coded = list.scaling_list_4x4()[2];
        assert_eq!(coded, std::array::from_fn(|i| i as u8));
        // The default intra 8×8 list is symmetric, so its coded order
        // starts 16, 16, 16 and ends with the largest factor.
        let eight = list.scaling_list_8x8()[0];
        assert_eq!(eight[0], 16);
        assert_eq!(eight[63], 115);
        assert_eq!(list.scaling_list_32x32()[1][63], 91);
    }

    #[test]
    fn predicted_rps_adds_the_delta_and_sorts() {
        // Reference set: -1 (used), 1 (used).
        let reference = StRps {
            delta_poc_s0: vec![-1],
            used_by_curr_pic_s0: vec![true],
            delta_poc_s1: vec![1],
            used_by_curr_pic_s1: vec![true],
            ..StRps::default()
        };
        // inter_ref_pic_set_prediction_flag 1, sign 1, abs_delta_rps_minus1 0
        // (delta -1), then used flags 1, 1, 1 for -2, 0, and -1.
        // 1, 1, 1 (ue 0), 1, 1, 1 -> bits: 1 1 1 1 1 1
        let data = [0b1111_1110];
        let mut bits = Bits::new(&data);
        let rps = parse_st_rps(&mut bits, &[reference], 1, 2, false).unwrap();
        assert_eq!(rps.delta_poc_s0, vec![-1, -2]);
        assert!(rps.delta_poc_s1.is_empty());
        assert!(rps.inter_ref_pic_set_prediction_flag);
        assert!(rps.delta_rps_sign);
        assert_eq!(rps.used_by_curr_pic_flag, vec![true, true, true]);
    }
}

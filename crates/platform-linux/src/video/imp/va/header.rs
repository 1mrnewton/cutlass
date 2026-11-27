//! The parameter sets and slice headers of the VA-API encoders' streams.
//!
//! They are written as FFmpeg's VA-API encoders write them for a stream
//! without B pictures: one slice a picture, an IDR every so often, and P
//! pictures that each reference the picture before. Every picture is a
//! reference, so H.264 counts pictures in `frame_num` with
//! `pic_order_cnt_type` 2, and HEVC sends one negative picture in each
//! slice's own reference picture set.

use super::bits::Writer;
use crate::video::rate::Vui;

/// One picture of the stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub idr: bool,
    /// Pictures since the last IDR, which is 0.
    pub index: u32,
    /// Counts IDR pictures, for H.264's `idr_pic_id`.
    pub idr_id: u32,
}

/// Pictures per second as the VUI and VA-API's frame rate parameter take
/// them: both terms at most 16 bits, halved together until they are.
pub fn rate16(rate: (u32, u32)) -> (u32, u32) {
    let (mut numerator, mut denominator) = (rate.0.max(1), rate.1.max(1));
    while numerator > 0xffff || denominator > 0xffff {
        numerator = (numerator / 2).max(1);
        denominator = (denominator / 2).max(1);
    }
    (numerator, denominator)
}

fn video_signal(out: &mut Writer, colour: Option<Vui>) {
    out.flag(colour.is_some()); // video_signal_type_present_flag
    if let Some(colour) = colour {
        out.u(3, 5); // video_format: unspecified
        out.flag(colour.full_range);
        out.flag(colour.described); // colour_description_present_flag
        if colour.described {
            out.u(8, u32::from(colour.primaries));
            out.u(8, u32::from(colour.transfer));
            out.u(8, u32::from(colour.matrix));
        }
    }
}

/// The smallest level whose limits hold `need`, else the largest.
fn level<const N: usize>(levels: &[(u8, [u64; N])], need: [u64; N]) -> u8 {
    levels
        .iter()
        .find(|(_, limits)| need.iter().zip(limits).all(|(need, limit)| need <= limit))
        .or(levels.last())
        .map_or(0, |(idc, _)| *idc)
}

pub const H264_LOG2_MAX_FRAME_NUM: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H264Profile {
    ConstrainedBaseline,
    Main,
    High,
}

impl H264Profile {
    pub fn idc(self) -> u8 {
        match self {
            Self::ConstrainedBaseline => 66,
            Self::Main => 77,
            Self::High => 100,
        }
    }

    /// `constraint_set0_flag` to `constraint_set5_flag`, as FFmpeg sets
    /// them: frame pictures only, and no B pictures.
    fn constraints(self) -> [bool; 6] {
        match self {
            Self::ConstrainedBaseline => [true, true, false, false, false, false],
            Self::Main => [false, true, false, false, true, true],
            Self::High => [false, false, false, false, true, true],
        }
    }

    pub fn cabac(self) -> bool {
        self != Self::ConstrainedBaseline
    }

    pub fn transform_8x8(self) -> bool {
        self == Self::High
    }
}

/// What an H.264 stream's headers say.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H264 {
    pub profile: H264Profile,
    pub level_idc: u8,
    pub width_mbs: u32,
    pub height_mbs: u32,
    /// Luma samples cropped off the right and bottom, each even.
    pub crop: (u32, u32),
    pub init_qp: u8,
    pub colour: Option<Vui>,
    /// From [`rate16`].
    pub rate: (u32, u32),
}

impl H264 {
    pub fn frame_num(step: Step) -> u32 {
        step.index % (1 << H264_LOG2_MAX_FRAME_NUM)
    }

    /// The picture's order count, two a frame as `pic_order_cnt_type` 2
    /// derives it.
    pub fn poc(step: Step) -> i32 {
        i32::try_from(step.index.saturating_mul(2)).unwrap_or(i32::MAX)
    }

    pub fn sps(&self) -> Vec<u8> {
        let mut out = Writer::new();
        out.u(8, 0x67); // nal_ref_idc 3, nal_unit_type 7
        out.u(8, u32::from(self.profile.idc()));
        for flag in self.profile.constraints() {
            out.flag(flag);
        }
        out.u(2, 0); // reserved_zero_2bits
        out.u(8, u32::from(self.level_idc));
        out.ue(0); // seq_parameter_set_id
        if self.profile == H264Profile::High {
            out.ue(1); // chroma_format_idc
            out.ue(0); // bit_depth_luma_minus8
            out.ue(0); // bit_depth_chroma_minus8
            out.flag(false); // qpprime_y_zero_transform_bypass_flag
            out.flag(false); // seq_scaling_matrix_present_flag
        }
        out.ue(H264_LOG2_MAX_FRAME_NUM - 4);
        out.ue(2); // pic_order_cnt_type
        out.ue(1); // max_num_ref_frames
        out.flag(false); // gaps_in_frame_num_value_allowed_flag
        out.ue(self.width_mbs - 1);
        out.ue(self.height_mbs - 1);
        out.flag(true); // frame_mbs_only_flag
        out.flag(true); // direct_8x8_inference_flag
        let cropped = self.crop != (0, 0);
        out.flag(cropped); // frame_cropping_flag
        if cropped {
            // In units of two luma samples for 4:2:0 frames.
            out.ue(0);
            out.ue(self.crop.0 / 2);
            out.ue(0);
            out.ue(self.crop.1 / 2);
        }
        out.flag(true); // vui_parameters_present_flag
        out.flag(false); // aspect_ratio_info_present_flag
        out.flag(false); // overscan_info_present_flag
        video_signal(&mut out, self.colour);
        out.flag(false); // chroma_loc_info_present_flag
        out.flag(true); // timing_info_present_flag
        out.u(32, self.rate.1); // num_units_in_tick
        out.u(32, self.rate.0 * 2); // time_scale: a tick is a field
        out.flag(false); // fixed_frame_rate_flag
        out.flag(false); // nal_hrd_parameters_present_flag
        out.flag(false); // vcl_hrd_parameters_present_flag
        out.flag(false); // pic_struct_present_flag
        out.flag(true); // bitstream_restriction_flag
        out.flag(true); // motion_vectors_over_pic_boundaries_flag
        out.ue(0); // max_bytes_per_pic_denom
        out.ue(0); // max_bits_per_mb_denom
        out.ue(15); // log2_max_mv_length_horizontal
        out.ue(15); // log2_max_mv_length_vertical
        out.ue(0); // max_num_reorder_frames
        out.ue(1); // max_dec_frame_buffering
        out.align();
        out.into_bytes()
    }

    pub fn pps(&self) -> Vec<u8> {
        let mut out = Writer::new();
        out.u(8, 0x68); // nal_ref_idc 3, nal_unit_type 8
        out.ue(0); // pic_parameter_set_id
        out.ue(0); // seq_parameter_set_id
        out.flag(self.profile.cabac()); // entropy_coding_mode_flag
        out.flag(false); // bottom_field_pic_order_in_frame_present_flag
        out.ue(0); // num_slice_groups_minus1
        out.ue(0); // num_ref_idx_l0_default_active_minus1
        out.ue(0); // num_ref_idx_l1_default_active_minus1
        out.flag(false); // weighted_pred_flag
        out.u(2, 0); // weighted_bipred_idc
        out.se(i32::from(self.init_qp) - 26); // pic_init_qp_minus26
        out.se(0); // pic_init_qs_minus26
        out.se(0); // chroma_qp_index_offset
        out.flag(true); // deblocking_filter_control_present_flag
        out.flag(false); // constrained_intra_pred_flag
        out.flag(false); // redundant_pic_cnt_present_flag
        if self.profile.transform_8x8() {
            out.flag(true); // transform_8x8_mode_flag
            out.flag(false); // pic_scaling_matrix_present_flag
            out.se(0); // second_chroma_qp_index_offset
        }
        out.align();
        out.into_bytes()
    }

    /// The slice NAL unit up to its slice data, which the driver appends.
    pub fn slice_header(&self, step: Step) -> Writer {
        let mut out = Writer::new();
        let kind = if step.idr { 5 } else { 1 };
        out.u(8, (1 << 5) | kind); // nal_ref_idc 1
        out.ue(0); // first_mb_in_slice
        out.ue(if step.idr { 2 } else { 0 }); // slice_type: I or P
        out.ue(0); // pic_parameter_set_id
        out.u(H264_LOG2_MAX_FRAME_NUM, Self::frame_num(step));
        if step.idr {
            out.ue(step.idr_id & 0xffff); // idr_pic_id
        } else {
            out.flag(false); // num_ref_idx_active_override_flag
            out.flag(false); // ref_pic_list_modification_flag_l0
        }
        // dec_ref_pic_marking()
        if step.idr {
            out.flag(false); // no_output_of_prior_pics_flag
            out.flag(false); // long_term_reference_flag
        } else {
            out.flag(false); // adaptive_ref_pic_marking_mode_flag
        }
        if self.profile.cabac() && !step.idr {
            out.ue(0); // cabac_init_idc
        }
        out.se(0); // slice_qp_delta
        out.ue(0); // disable_deblocking_filter_idc
        out.se(0); // slice_alpha_c0_offset_div2
        out.se(0); // slice_beta_offset_div2
        out
    }

    /// The level of Table A-1 that holds the picture size, macroblock
    /// rate and bitrate.
    pub fn level(
        width_mbs: u32,
        height_mbs: u32,
        rate: (u32, u32),
        bitrate: u32,
        profile: H264Profile,
    ) -> u8 {
        // level_idc, then MaxFS, MaxMBPS (scaled by the rate's
        // denominator below), and MaxBR in bits a second.
        const LEVELS: [(u8, [u64; 3]); 19] = [
            (10, [99, 1_485, 64]),
            (11, [396, 3_000, 192]),
            (12, [396, 6_000, 384]),
            (13, [396, 11_880, 768]),
            (20, [396, 11_880, 2_000]),
            (21, [792, 19_800, 4_000]),
            (22, [1_620, 20_250, 4_000]),
            (30, [1_620, 40_500, 10_000]),
            (31, [3_600, 108_000, 14_000]),
            (32, [5_120, 216_000, 20_000]),
            (40, [8_192, 245_760, 20_000]),
            (41, [8_192, 245_760, 50_000]),
            (42, [8_704, 522_240, 50_000]),
            (50, [22_080, 589_824, 135_000]),
            (51, [36_864, 983_040, 240_000]),
            (52, [36_864, 2_073_600, 240_000]),
            (60, [139_264, 4_177_920, 240_000]),
            (61, [139_264, 8_355_840, 480_000]),
            (62, [139_264, 16_711_680, 800_000]),
        ];
        let (numerator, denominator) = (u64::from(rate.0), u64::from(rate.1.max(1)));
        // High allows 1250 bits for each 1000 of the table.
        let factor = if profile == H264Profile::High {
            1_250
        } else {
            1_000
        };
        let frame = u64::from(width_mbs) * u64::from(height_mbs);
        let side = u64::from(width_mbs.max(height_mbs));
        let levels: Vec<(u8, [u64; 4])> = LEVELS
            .iter()
            .map(|(idc, [fs, mbps, br])| (*idc, [*fs, mbps * denominator, br * factor, fs * 8]))
            .collect();
        level(
            &levels,
            [frame, frame * numerator, u64::from(bitrate), side * side],
        )
    }
}

pub const HEVC_LOG2_MAX_POC_LSB: u32 = 8;

/// HEVC block sizes, as log2 of their edge in luma samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blocks {
    pub min_cb: u32,
    pub ctb: u32,
    pub min_tb: u32,
    pub max_tb: u32,
    pub depth_inter: u32,
    pub depth_intra: u32,
}

impl Blocks {
    /// What FFmpeg uses when the driver does not say: 32x32 coding tree
    /// blocks, 8x8 coding blocks and transforms from 4x4 to 32x32.
    pub const DEFAULT: Self = Self {
        min_cb: 3,
        ctb: 5,
        min_tb: 2,
        max_tb: 5,
        depth_inter: 3,
        depth_intra: 3,
    };
}

/// HEVC coding tools, on or off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tools {
    pub amp: bool,
    pub sao: bool,
    pub temporal_mvp: bool,
    pub strong_intra_smoothing: bool,
    pub sign_data_hiding: bool,
    pub transform_skip: bool,
    pub cu_qp_delta: bool,
}

/// What an HEVC Main stream's headers say.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hevc {
    pub level_idc: u8,
    /// Coded size, multiples of the minimum coding block.
    pub width: u32,
    pub height: u32,
    /// Luma samples cropped off the right and bottom, each even.
    pub crop: (u32, u32),
    pub blocks: Blocks,
    pub tools: Tools,
    /// P pictures go as B slices with the same picture in both lists, for
    /// drivers that refuse P slices.
    pub gpb: bool,
    pub init_qp: u8,
    pub colour: Option<Vui>,
    /// From [`rate16`].
    pub rate: (u32, u32),
}

pub const HEVC_IDR_W_RADL: u32 = 19;
pub const HEVC_TRAIL_R: u32 = 1;

/// HEVC `slice_type`.
pub const HEVC_B: u8 = 0;
pub const HEVC_P: u8 = 1;
pub const HEVC_I: u8 = 2;

impl Hevc {
    pub fn poc(step: Step) -> i32 {
        i32::try_from(step.index).unwrap_or(i32::MAX)
    }

    pub fn nal_unit_type(step: Step) -> u32 {
        if step.idr {
            HEVC_IDR_W_RADL
        } else {
            HEVC_TRAIL_R
        }
    }

    pub fn slice_type(&self, step: Step) -> u8 {
        match (step.idr, self.gpb) {
            (true, _) => HEVC_I,
            (false, true) => HEVC_B,
            (false, false) => HEVC_P,
        }
    }

    /// The coding tree blocks that cover the picture.
    pub fn ctus(&self) -> u32 {
        let size = 1 << self.blocks.ctb;
        self.width.div_ceil(size) * self.height.div_ceil(size)
    }

    fn nal_header(out: &mut Writer, kind: u32) {
        out.u(8, kind << 1); // forbidden_zero_bit, nal_unit_type, layer high bit
        out.u(8, 1); // nuh_layer_id 0, nuh_temporal_id_plus1 1
    }

    fn profile_tier_level(&self, out: &mut Writer) {
        out.u(2, 0); // general_profile_space
        out.flag(false); // general_tier_flag: Main
        out.u(5, 1); // general_profile_idc: Main
        for profile in 0..32 {
            // Main streams are Main 10 streams too.
            out.flag(profile == 1 || profile == 2);
        }
        out.flag(true); // general_progressive_source_flag
        out.flag(false); // general_interlaced_source_flag
        out.flag(false); // general_non_packed_constraint_flag
        out.flag(true); // general_frame_only_constraint_flag
        out.bits(43, 0); // general_reserved_zero_43bits
        out.flag(false); // general_reserved_zero_bit
        out.u(8, u32::from(self.level_idc));
    }

    /// `sps_max_dec_pic_buffering_minus1` and the two after it: the
    /// current picture and one reference, no reordering.
    fn ordering(out: &mut Writer) {
        out.flag(true); // sub_layer_ordering_info_present_flag
        out.ue(1); // max_dec_pic_buffering_minus1
        out.ue(0); // max_num_reorder_pics
        out.ue(0); // max_latency_increase_plus1
    }

    pub fn vps(&self) -> Vec<u8> {
        let mut out = Writer::new();
        Self::nal_header(&mut out, 32);
        out.u(4, 0); // vps_video_parameter_set_id
        out.flag(true); // vps_base_layer_internal_flag
        out.flag(true); // vps_base_layer_available_flag
        out.u(6, 0); // vps_max_layers_minus1
        out.u(3, 0); // vps_max_sub_layers_minus1
        out.flag(true); // vps_temporal_id_nesting_flag
        out.u(16, 0xffff); // vps_reserved_0xffff_16bits
        self.profile_tier_level(&mut out);
        Self::ordering(&mut out);
        out.u(6, 0); // vps_max_layer_id
        out.ue(0); // vps_num_layer_sets_minus1
        out.flag(false); // vps_timing_info_present_flag
        out.flag(false); // vps_extension_flag
        out.align();
        out.into_bytes()
    }

    pub fn sps(&self) -> Vec<u8> {
        let mut out = Writer::new();
        Self::nal_header(&mut out, 33);
        out.u(4, 0); // sps_video_parameter_set_id
        out.u(3, 0); // sps_max_sub_layers_minus1
        out.flag(true); // sps_temporal_id_nesting_flag
        self.profile_tier_level(&mut out);
        out.ue(0); // sps_seq_parameter_set_id
        out.ue(1); // chroma_format_idc
        out.ue(self.width); // pic_width_in_luma_samples
        out.ue(self.height); // pic_height_in_luma_samples
        let cropped = self.crop != (0, 0);
        out.flag(cropped); // conformance_window_flag
        if cropped {
            // In chroma samples.
            out.ue(0);
            out.ue(self.crop.0 / 2);
            out.ue(0);
            out.ue(self.crop.1 / 2);
        }
        out.ue(0); // bit_depth_luma_minus8
        out.ue(0); // bit_depth_chroma_minus8
        out.ue(HEVC_LOG2_MAX_POC_LSB - 4);
        Self::ordering(&mut out);
        let blocks = self.blocks;
        out.ue(blocks.min_cb - 3);
        out.ue(blocks.ctb - blocks.min_cb);
        out.ue(blocks.min_tb - 2);
        out.ue(blocks.max_tb - blocks.min_tb);
        out.ue(blocks.depth_inter);
        out.ue(blocks.depth_intra);
        let tools = self.tools;
        out.flag(false); // scaling_list_enabled_flag
        out.flag(tools.amp);
        out.flag(tools.sao);
        out.flag(false); // pcm_enabled_flag
        out.ue(0); // num_short_term_ref_pic_sets: each slice has its own
        out.flag(false); // long_term_ref_pics_present_flag
        out.flag(tools.temporal_mvp);
        out.flag(tools.strong_intra_smoothing);
        out.flag(true); // vui_parameters_present_flag
        out.flag(false); // aspect_ratio_info_present_flag
        out.flag(false); // overscan_info_present_flag
        video_signal(&mut out, self.colour);
        out.flag(false); // chroma_loc_info_present_flag
        out.flag(false); // neutral_chroma_indication_flag
        out.flag(false); // field_seq_flag
        out.flag(false); // frame_field_info_present_flag
        out.flag(false); // default_display_window_flag
        out.flag(true); // vui_timing_info_present_flag
        out.u(32, self.rate.1); // vui_num_units_in_tick
        out.u(32, self.rate.0); // vui_time_scale
        out.flag(false); // vui_poc_proportional_to_timing_flag
        out.flag(false); // vui_hrd_parameters_present_flag
        out.flag(true); // bitstream_restriction_flag
        out.flag(false); // tiles_fixed_structure_flag
        out.flag(true); // motion_vectors_over_pic_boundaries_flag
        out.flag(true); // restricted_ref_pic_lists_flag
        out.ue(0); // min_spatial_segmentation_idc
        out.ue(0); // max_bytes_per_pic_denom
        out.ue(0); // max_bits_per_min_cu_denom
        out.ue(15); // log2_max_mv_length_horizontal
        out.ue(15); // log2_max_mv_length_vertical
        out.flag(false); // sps_extension_present_flag
        out.align();
        out.into_bytes()
    }

    pub fn pps(&self) -> Vec<u8> {
        let mut out = Writer::new();
        Self::nal_header(&mut out, 34);
        let tools = self.tools;
        out.ue(0); // pps_pic_parameter_set_id
        out.ue(0); // pps_seq_parameter_set_id
        out.flag(false); // dependent_slice_segments_enabled_flag
        out.flag(false); // output_flag_present_flag
        out.u(3, 0); // num_extra_slice_header_bits
        out.flag(tools.sign_data_hiding);
        out.flag(false); // cabac_init_present_flag
        out.ue(0); // num_ref_idx_l0_default_active_minus1
        out.ue(0); // num_ref_idx_l1_default_active_minus1
        out.se(i32::from(self.init_qp) - 26); // init_qp_minus26
        out.flag(false); // constrained_intra_pred_flag
        out.flag(tools.transform_skip);
        out.flag(tools.cu_qp_delta);
        if tools.cu_qp_delta {
            out.ue(0); // diff_cu_qp_delta_depth
        }
        out.se(0); // pps_cb_qp_offset
        out.se(0); // pps_cr_qp_offset
        out.flag(false); // pps_slice_chroma_qp_offsets_present_flag
        out.flag(false); // weighted_pred_flag
        out.flag(false); // weighted_bipred_flag
        out.flag(false); // transquant_bypass_enabled_flag
        out.flag(false); // tiles_enabled_flag
        out.flag(false); // entropy_coding_sync_enabled_flag
        out.flag(false); // pps_loop_filter_across_slices_enabled_flag
        out.flag(false); // deblocking_filter_control_present_flag
        out.flag(false); // pps_scaling_list_data_present_flag
        out.flag(false); // lists_modification_present_flag
        out.ue(0); // log2_parallel_merge_level_minus2
        out.flag(false); // slice_segment_header_extension_present_flag
        out.flag(false); // pps_extension_present_flag
        out.align();
        out.into_bytes()
    }

    /// The slice segment NAL unit up to its slice data, ending with the
    /// header's byte alignment.
    pub fn slice_header(&self, step: Step) -> Writer {
        let mut out = Writer::new();
        Self::nal_header(&mut out, Self::nal_unit_type(step));
        let tools = self.tools;
        let kind = self.slice_type(step);
        out.flag(true); // first_slice_segment_in_pic_flag
        if step.idr {
            out.flag(false); // no_output_of_prior_pics_flag
        }
        out.ue(0); // slice_pic_parameter_set_id
        out.ue(u32::from(kind));
        if !step.idr {
            let lsb = step.index % (1 << HEVC_LOG2_MAX_POC_LSB);
            out.u(HEVC_LOG2_MAX_POC_LSB, lsb); // slice_pic_order_cnt_lsb
            out.flag(false); // short_term_ref_pic_set_sps_flag
            out.ue(1); // num_negative_pics
            out.ue(0); // num_positive_pics
            out.ue(0); // delta_poc_s0_minus1: the picture before
            out.flag(true); // used_by_curr_pic_s0_flag
            if tools.temporal_mvp {
                out.flag(true); // slice_temporal_mvp_enabled_flag
            }
        }
        if tools.sao {
            out.flag(true); // slice_sao_luma_flag
            out.flag(true); // slice_sao_chroma_flag
        }
        if !step.idr {
            out.flag(false); // num_ref_idx_active_override_flag
            if kind == HEVC_B {
                out.flag(false); // mvd_l1_zero_flag
                if tools.temporal_mvp {
                    out.flag(true); // collocated_from_l0_flag
                }
            }
            out.ue(0); // five_minus_max_num_merge_cand
        }
        out.se(0); // slice_qp_delta
        out.align(); // byte_alignment()
        out
    }

    /// The Main tier level of Table A.8 that holds the picture size,
    /// sample rate and bitrate.
    pub fn level(width: u32, height: u32, rate: (u32, u32), bitrate: u32) -> u8 {
        // general_level_idc, then MaxLumaPs, MaxLumaSr (scaled by the
        // rate's denominator below) and MaxBR in bits a second.
        const LEVELS: [(u8, [u64; 3]); 13] = [
            (30, [36_864, 552_960, 128_000]),
            (60, [122_880, 3_686_400, 1_500_000]),
            (63, [245_760, 7_372_800, 3_000_000]),
            (90, [552_960, 16_588_800, 6_000_000]),
            (93, [983_040, 33_177_600, 10_000_000]),
            (120, [2_228_224, 66_846_720, 12_000_000]),
            (123, [2_228_224, 133_693_440, 20_000_000]),
            (150, [8_912_896, 267_386_880, 25_000_000]),
            (153, [8_912_896, 534_773_760, 40_000_000]),
            (156, [8_912_896, 1_069_547_520, 60_000_000]),
            (180, [35_651_584, 1_069_547_520, 60_000_000]),
            (183, [35_651_584, 2_139_095_040, 120_000_000]),
            (186, [35_651_584, 4_278_190_080, 240_000_000]),
        ];
        let (numerator, denominator) = (u64::from(rate.0), u64::from(rate.1.max(1)));
        let samples = u64::from(width) * u64::from(height);
        let side = u64::from(width.max(height));
        let levels: Vec<(u8, [u64; 4])> = LEVELS
            .iter()
            .map(|(idc, [ps, sr, br])| (*idc, [*ps, sr * denominator, *br, ps * 8]))
            .collect();
        level(
            &levels,
            [
                samples,
                samples * numerator,
                u64::from(bitrate),
                side * side,
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_shrink_to_sixteen_bits() {
        assert_eq!(rate16((30_000, 1001)), (30_000, 1001));
        assert_eq!(rate16((90_000, 3)), (45_000, 1));
        assert_eq!(rate16((0, 0)), (1, 1));
        let (numerator, denominator) = rate16((1_000_000, 33_333));
        assert!(numerator <= 0xffff && denominator <= 0xffff);
        assert_eq!(numerator / denominator, 30);
    }

    #[test]
    fn levels_follow_size_rate_and_bitrate() {
        let high = H264Profile::High;
        // 1080p30 at 16.6 Mbit/s fits 4.0 by size, rate and bits.
        assert_eq!(H264::level(120, 68, (30, 1), 16_588_800, high), 40);
        assert_eq!(H264::level(120, 68, (30, 1), 30_000_000, high), 41);
        assert_eq!(H264::level(120, 68, (60, 1), 8_000_000, high), 42);
        assert_eq!(H264::level(240, 135, (30, 1), 40_000_000, high), 51);
        assert_eq!(H264::level(4, 4, (30, 1), 64_000, H264Profile::Main), 10);
        // Past every level: the last.
        assert_eq!(H264::level(2000, 2000, (120, 1), 0, high), 62);
        assert_eq!(Hevc::level(1920, 1080, (30, 1), 10_000_000), 120);
        assert_eq!(Hevc::level(1920, 1080, (60, 1), 10_000_000), 123);
        assert_eq!(Hevc::level(3840, 2160, (30, 1), 20_000_000), 150);
        assert_eq!(Hevc::level(64, 64, (30, 1), 64_000), 30);
        // A thin picture needs a level whose square holds its long side.
        assert_eq!(Hevc::level(4096, 16, (30, 1), 64_000), 120);
    }
}

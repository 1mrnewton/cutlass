//! OBU, sequence header and frame header parsing (spec 5.3 – 5.9, 7.5, 7.8, 7.20, 7.21).

use cutlass_decode::CodecError;

use super::bits::{BitReader, R};

pub const OBU_SEQUENCE_HEADER: u8 = 1;
pub const OBU_TEMPORAL_DELIMITER: u8 = 2;
pub const OBU_FRAME_HEADER: u8 = 3;
pub const OBU_TILE_GROUP: u8 = 4;
pub const OBU_METADATA: u8 = 5;
pub const OBU_FRAME: u8 = 6;
pub const OBU_REDUNDANT_FRAME_HEADER: u8 = 7;
pub const OBU_TILE_LIST: u8 = 8;
pub const OBU_PADDING: u8 = 15;

pub const KEY_FRAME: u8 = 0;
pub const INTER_FRAME: u8 = 1;
pub const INTRA_ONLY_FRAME: u8 = 2;
pub const SWITCH_FRAME: u8 = 3;

pub const NUM_REF_FRAMES: usize = 8;
pub const REFS_PER_FRAME: usize = 7;
pub const PRIMARY_REF_NONE: u8 = 7;
pub const MAX_SEGMENTS: usize = 8;
pub const SEG_LVL_MAX: usize = 8;
pub const SEG_LVL_ALT_Q: usize = 0;
pub const SEG_LVL_REF_FRAME: usize = 5;
pub const SEG_LVL_SKIP: usize = 6;
pub const SEG_LVL_GLOBALMV: usize = 7;
pub const MAX_TILE_COLS: usize = 64;
pub const MAX_TILE_ROWS: usize = 64;

// Global motion types.
pub const IDENTITY: u8 = 0;
pub const TRANSLATION: u8 = 1;
pub const ROTZOOM: u8 = 2;
pub const AFFINE: u8 = 3;
pub const WARPEDMODEL_PREC_BITS: u32 = 16;

pub const RESTORE_NONE: u8 = 0;
pub const RESTORE_WIENER: u8 = 1;
pub const RESTORE_SGRPROJ: u8 = 2;
pub const RESTORE_SWITCHABLE: u8 = 3;

pub const ONLY_4X4: u8 = 0;
pub const TX_MODE_LARGEST: u8 = 1;
pub const TX_MODE_SELECT: u8 = 2;

pub const SWITCHABLE: u8 = 4;

const SEG_FEATURE_BITS: [u32; 8] = [8, 6, 6, 6, 6, 3, 0, 0];
const SEG_FEATURE_SIGNED: [bool; 8] = [true, true, true, true, true, false, false, false];
const SEG_FEATURE_MAX: [i32; 8] = [255, 63, 63, 63, 63, 7, 0, 0];

fn malformed(what: &'static str) -> CodecError {
    CodecError::Malformed { at_bit: 0, what }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OperatingPoint {
    pub idc: u16,
    pub seq_level_idx: u8,
    pub seq_tier: u8,
    pub decoder_model_present: bool,
    pub initial_display_delay_present: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SequenceHeader {
    pub profile: u8,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub timing_info_present: bool,
    pub equal_picture_interval: bool,
    pub decoder_model_info_present: bool,
    pub buffer_removal_time_length: u32,
    pub frame_presentation_time_length: u32,
    pub operating_points: Vec<OperatingPoint>,
    pub operating_point_idc: u16,
    pub frame_width_bits: u32,
    pub frame_height_bits: u32,
    pub max_frame_width: u32,
    pub max_frame_height: u32,
    pub frame_id_numbers_present: bool,
    pub delta_frame_id_length: u32,
    pub additional_frame_id_length: u32,
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    pub enable_order_hint: bool,
    pub enable_jnt_comp: bool,
    pub enable_ref_frame_mvs: bool,
    pub seq_force_screen_content_tools: u8,
    pub seq_force_integer_mv: u8,
    pub order_hint_bits: u32,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    // color config
    pub bit_depth: u32,
    pub mono_chrome: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub color_range: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub chroma_sample_position: u8,
    pub separate_uv_delta_q: bool,
    pub film_grain_params_present: bool,
}

impl SequenceHeader {
    pub fn num_planes(&self) -> usize {
        if self.mono_chrome { 1 } else { 3 }
    }
}

const SELECT_SCREEN_CONTENT_TOOLS: u8 = 2;
const SELECT_INTEGER_MV: u8 = 2;

pub fn parse_sequence_header(data: &[u8]) -> R<SequenceHeader> {
    let mut r = BitReader::new(data);
    let mut s = SequenceHeader {
        profile: r.f(3)? as u8,
        ..SequenceHeader::default()
    };
    if s.profile > 2 {
        return Err(CodecError::Unsupported {
            what: "av1 seq_profile",
        });
    }
    s.still_picture = r.flag()?;
    s.reduced_still_picture_header = r.flag()?;
    if s.reduced_still_picture_header {
        s.operating_points.push(OperatingPoint {
            seq_level_idx: r.f(5)? as u8,
            ..OperatingPoint::default()
        });
    } else {
        s.timing_info_present = r.flag()?;
        let mut buffer_delay_length = 0;
        if s.timing_info_present {
            r.f(32)?; // num_units_in_display_tick
            r.f(32)?; // time_scale
            s.equal_picture_interval = r.flag()?;
            if s.equal_picture_interval {
                r.uvlc()?;
            }
            s.decoder_model_info_present = r.flag()?;
            if s.decoder_model_info_present {
                buffer_delay_length = r.f(5)? + 1;
                r.f(32)?;
                s.buffer_removal_time_length = r.f(5)? + 1;
                s.frame_presentation_time_length = r.f(5)? + 1;
            }
        }
        let initial_display_delay_present = r.flag()?;
        let cnt = r.f(5)? + 1;
        for _ in 0..cnt {
            let mut op = OperatingPoint {
                idc: r.f(12)? as u16,
                seq_level_idx: r.f(5)? as u8,
                ..OperatingPoint::default()
            };
            if op.seq_level_idx > 7 {
                op.seq_tier = r.f(1)? as u8;
            }
            if s.decoder_model_info_present {
                op.decoder_model_present = r.flag()?;
                if op.decoder_model_present {
                    r.f(buffer_delay_length)?;
                    r.f(buffer_delay_length)?;
                    r.f(1)?;
                }
            }
            if initial_display_delay_present {
                op.initial_display_delay_present = r.flag()?;
                if op.initial_display_delay_present {
                    r.f(4)?;
                }
            }
            s.operating_points.push(op);
        }
    }
    // choose_operating_point: operating point 0.
    s.operating_point_idc = s.operating_points.first().map_or(0, |o| o.idc);
    s.frame_width_bits = r.f(4)? + 1;
    s.frame_height_bits = r.f(4)? + 1;
    s.max_frame_width = r.f(s.frame_width_bits)? + 1;
    s.max_frame_height = r.f(s.frame_height_bits)? + 1;
    if !s.reduced_still_picture_header {
        s.frame_id_numbers_present = r.flag()?;
    }
    if s.frame_id_numbers_present {
        s.delta_frame_id_length = r.f(4)? + 2;
        s.additional_frame_id_length = r.f(3)? + 1;
    }
    s.use_128x128_superblock = r.flag()?;
    s.enable_filter_intra = r.flag()?;
    s.enable_intra_edge_filter = r.flag()?;
    if s.reduced_still_picture_header {
        s.seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS;
        s.seq_force_integer_mv = SELECT_INTEGER_MV;
    } else {
        s.enable_interintra_compound = r.flag()?;
        s.enable_masked_compound = r.flag()?;
        s.enable_warped_motion = r.flag()?;
        s.enable_dual_filter = r.flag()?;
        s.enable_order_hint = r.flag()?;
        if s.enable_order_hint {
            s.enable_jnt_comp = r.flag()?;
            s.enable_ref_frame_mvs = r.flag()?;
        }
        let choose_sct = r.flag()?;
        s.seq_force_screen_content_tools = if choose_sct {
            SELECT_SCREEN_CONTENT_TOOLS
        } else {
            r.f(1)? as u8
        };
        if s.seq_force_screen_content_tools > 0 {
            let choose_imv = r.flag()?;
            s.seq_force_integer_mv = if choose_imv {
                SELECT_INTEGER_MV
            } else {
                r.f(1)? as u8
            };
        } else {
            s.seq_force_integer_mv = SELECT_INTEGER_MV;
        }
        if s.enable_order_hint {
            s.order_hint_bits = r.f(3)? + 1;
        }
    }
    s.enable_superres = r.flag()?;
    s.enable_cdef = r.flag()?;
    s.enable_restoration = r.flag()?;
    // color_config
    let high_bitdepth = r.flag()?;
    s.bit_depth = if s.profile == 2 && high_bitdepth {
        if r.flag()? { 12 } else { 10 }
    } else if high_bitdepth {
        10
    } else {
        8
    };
    s.mono_chrome = if s.profile == 1 { false } else { r.flag()? };
    let color_description_present = r.flag()?;
    if color_description_present {
        s.color_primaries = r.f(8)? as u8;
        s.transfer_characteristics = r.f(8)? as u8;
        s.matrix_coefficients = r.f(8)? as u8;
    } else {
        s.color_primaries = 2;
        s.transfer_characteristics = 2;
        s.matrix_coefficients = 2;
    }
    if s.mono_chrome {
        s.color_range = r.flag()?;
        s.subsampling_x = true;
        s.subsampling_y = true;
        s.separate_uv_delta_q = false;
    } else {
        if s.color_primaries == 1 && s.transfer_characteristics == 13 && s.matrix_coefficients == 0
        {
            s.color_range = true;
        } else {
            s.color_range = r.flag()?;
            match s.profile {
                0 => {
                    s.subsampling_x = true;
                    s.subsampling_y = true;
                }
                1 => {}
                _ => {
                    if s.bit_depth == 12 {
                        s.subsampling_x = r.flag()?;
                        if s.subsampling_x {
                            s.subsampling_y = r.flag()?;
                        }
                    } else {
                        s.subsampling_x = true;
                    }
                }
            }
            if s.subsampling_x && s.subsampling_y {
                s.chroma_sample_position = r.f(2)? as u8;
            }
        }
        s.separate_uv_delta_q = r.flag()?;
    }
    s.film_grain_params_present = r.flag()?;
    Ok(s)
}

/// Film grain syntax elements (spec 5.9.30) after load/update resolution.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GrainParams {
    pub apply_grain: bool,
    pub grain_seed: u16,
    pub update_grain: bool,
    pub num_y_points: u8,
    pub point_y_value: [u8; 16],
    pub point_y_scaling: [u8; 16],
    pub chroma_scaling_from_luma: bool,
    pub num_cb_points: u8,
    pub point_cb_value: [u8; 16],
    pub point_cb_scaling: [u8; 16],
    pub num_cr_points: u8,
    pub point_cr_value: [u8; 16],
    pub point_cr_scaling: [u8; 16],
    pub grain_scaling_minus_8: u8,
    pub ar_coeff_lag: u8,
    pub ar_coeffs_y_plus_128: [u8; 24],
    pub ar_coeffs_cb_plus_128: [u8; 25],
    pub ar_coeffs_cr_plus_128: [u8; 25],
    pub ar_coeff_shift_minus_6: u8,
    pub grain_scale_shift: u8,
    pub cb_mult: u8,
    pub cb_luma_mult: u8,
    pub cb_offset: u16,
    pub cr_mult: u8,
    pub cr_luma_mult: u8,
    pub cr_offset: u16,
    pub overlap_flag: bool,
    pub clip_to_restricted_range: bool,
}

/// Per-slot header state that later frame headers read (spec 7.20 / 7.21).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RefHeaderState {
    pub valid: bool,
    pub frame_id: u32,
    pub upscaled_width: u32,
    pub frame_width: u32,
    pub frame_height: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub mi_cols: u32,
    pub mi_rows: u32,
    pub frame_type: u8,
    pub order_hint: u32,
    pub saved_order_hints: [u32; 8],
    pub gm_params: [[i32; 6]; 8],
    pub loop_filter_ref_deltas: [i8; 8],
    pub loop_filter_mode_deltas: [i8; 2],
    pub feature_enabled: [[bool; 8]; 8],
    pub feature_data: [[i16; 8]; 8],
    pub grain: GrainParams,
    pub bit_depth: u32,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub showable: bool,
}

impl Default for RefHeaderState {
    fn default() -> Self {
        Self {
            valid: false,
            frame_id: 0,
            upscaled_width: 0,
            frame_width: 0,
            frame_height: 0,
            render_width: 0,
            render_height: 0,
            mi_cols: 0,
            mi_rows: 0,
            frame_type: 0,
            order_hint: 0,
            saved_order_hints: [0; 8],
            gm_params: default_gm_params(),
            loop_filter_ref_deltas: DEFAULT_LF_REF_DELTAS,
            loop_filter_mode_deltas: [0; 2],
            feature_enabled: [[false; 8]; 8],
            feature_data: [[0; 8]; 8],
            grain: GrainParams::default(),
            bit_depth: 8,
            subsampling_x: true,
            subsampling_y: true,
            showable: false,
        }
    }
}

pub const DEFAULT_LF_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];

pub fn default_gm_params() -> [[i32; 6]; 8] {
    let mut p = [[0i32; 6]; 8];
    for r in p.iter_mut() {
        r[2] = 1 << WARPEDMODEL_PREC_BITS;
        r[5] = 1 << WARPEDMODEL_PREC_BITS;
    }
    p
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TileInfo {
    pub cols: usize,
    pub rows: usize,
    pub cols_log2: u32,
    pub rows_log2: u32,
    pub mi_col_starts: Vec<u32>,
    pub mi_row_starts: Vec<u32>,
    pub context_update_tile_id: u32,
    pub tile_size_bytes: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Quant {
    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub delta_q_u_dc: i8,
    pub delta_q_u_ac: i8,
    pub delta_q_v_dc: i8,
    pub delta_q_v_ac: i8,
    pub using_qmatrix: bool,
    pub qm_y: u8,
    pub qm_u: u8,
    pub qm_v: u8,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Segmentation {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal_update: bool,
    pub update_data: bool,
    pub feature_enabled: [[bool; 8]; 8],
    pub feature_data: [[i16; 8]; 8],
    pub seg_id_pre_skip: bool,
    pub last_active_seg_id: u8,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoopFilter {
    pub level: [u8; 4],
    pub sharpness: u8,
    pub delta_enabled: bool,
    pub delta_update: bool,
    pub ref_deltas: [i8; 8],
    pub mode_deltas: [i8; 2],
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cdef {
    pub damping: u8,
    pub bits: u8,
    pub y_pri: [u8; 8],
    pub y_sec: [u8; 8],
    pub uv_pri: [u8; 8],
    pub uv_sec: [u8; 8],
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Restoration {
    pub frame_type: [u8; 3],
    pub uses_lr: bool,
    pub unit_size: [u32; 3],
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameHeader {
    pub show_existing_frame: bool,
    pub frame_to_show_map_idx: u8,
    pub frame_type: u8,
    pub frame_is_intra: bool,
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,
    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: bool,
    pub current_frame_id: u32,
    pub frame_size_override_flag: bool,
    pub order_hint: u32,
    pub primary_ref_frame: u8,
    pub refresh_frame_flags: u8,
    pub ref_order_hint: [u32; 8],
    pub frame_width: u32,
    pub frame_height: u32,
    pub upscaled_width: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub use_superres: bool,
    pub superres_denom: u32,
    pub mi_cols: u32,
    pub mi_rows: u32,
    pub allow_intrabc: bool,
    pub ref_frame_idx: [u8; 7],
    pub allow_high_precision_mv: bool,
    pub interpolation_filter: u8,
    pub is_motion_mode_switchable: bool,
    pub use_ref_frame_mvs: bool,
    /// OrderHints[ LAST_FRAME + i ] at index i + 1.
    pub order_hints: [u32; 8],
    pub ref_frame_sign_bias: [bool; 8],
    pub disable_frame_end_update_cdf: bool,
    pub tile_info: TileInfo,
    pub quant: Quant,
    pub segmentation: Segmentation,
    pub delta_q_present: bool,
    pub delta_q_res: u8,
    pub delta_lf_present: bool,
    pub delta_lf_res: u8,
    pub delta_lf_multi: bool,
    pub coded_lossless: bool,
    pub all_lossless: bool,
    pub lossless_array: [bool; 8],
    pub seg_qm_level: [[u8; 8]; 3],
    pub loop_filter: LoopFilter,
    pub cdef: Cdef,
    pub lr: Restoration,
    pub tx_mode: u8,
    pub reference_select: bool,
    pub skip_mode_present: bool,
    pub skip_mode_frame: [u8; 2],
    pub allow_warped_motion: bool,
    pub reduced_tx_set: bool,
    pub gm_type: [u8; 8],
    pub gm_params: [[i32; 6]; 8],
    pub grain: GrainParams,
    /// Byte length of the uncompressed header (after byte alignment).
    pub header_bytes: usize,
    pub temporal_id: u8,
    pub spatial_id: u8,
}

impl FrameHeader {
    pub fn get_relative_dist(&self, seq: &SequenceHeader, a: u32, b: u32) -> i32 {
        relative_dist(seq, a, b)
    }

    /// get_qindex( ignoreDeltaQ = 1, segmentId )
    pub fn qindex(&self, segment_id: usize) -> u8 {
        let base = i32::from(self.quant.base_q_idx);
        if self.seg_feature_active(segment_id, SEG_LVL_ALT_Q) {
            let data = i32::from(self.segmentation.feature_data[segment_id][SEG_LVL_ALT_Q]);
            (base + data).clamp(0, 255) as u8
        } else {
            base as u8
        }
    }

    pub fn seg_feature_active(&self, segment_id: usize, feature: usize) -> bool {
        self.segmentation.enabled && self.segmentation.feature_enabled[segment_id][feature]
    }
}

pub fn relative_dist(seq: &SequenceHeader, a: u32, b: u32) -> i32 {
    if !seq.enable_order_hint {
        return 0;
    }
    let diff = a as i32 - b as i32;
    let m = 1i32 << (seq.order_hint_bits - 1);
    (diff & (m - 1)) - (diff & m)
}

fn tile_log2(blk: u32, target: u32) -> u32 {
    let mut k = 0;
    while (blk << k) < target {
        k += 1;
    }
    k
}

/// Inputs from decoder state needed to parse a frame header.
pub struct HeaderContext<'a> {
    pub seq: &'a SequenceHeader,
    pub refs: &'a [RefHeaderState; 8],
    pub temporal_id: u8,
    pub spatial_id: u8,
}

/// Parses uncompressed_header(). The returned header has all derived values.
pub fn parse_frame_header(data: &[u8], ctx: &HeaderContext<'_>) -> R<FrameHeader> {
    let seq = ctx.seq;
    let mut r = BitReader::new(data);
    let mut h = FrameHeader {
        temporal_id: ctx.temporal_id,
        spatial_id: ctx.spatial_id,
        ..FrameHeader::default()
    };
    let mut ref_valid: [bool; 8] = std::array::from_fn(|i| ctx.refs[i].valid);
    let mut ref_order_hint: [u32; 8] = std::array::from_fn(|i| ctx.refs[i].order_hint);
    let id_len = if seq.frame_id_numbers_present {
        seq.additional_frame_id_length + seq.delta_frame_id_length
    } else {
        0
    };
    let all_frames = 0xffu8;
    if seq.reduced_still_picture_header {
        h.frame_type = KEY_FRAME;
        h.frame_is_intra = true;
        h.show_frame = true;
    } else {
        h.show_existing_frame = r.flag()?;
        if h.show_existing_frame {
            h.frame_to_show_map_idx = r.f(3)? as u8;
            if seq.decoder_model_info_present && !seq.equal_picture_interval {
                r.f(seq.frame_presentation_time_length)?;
            }
            if seq.frame_id_numbers_present {
                r.f(id_len)?;
            }
            let slot = &ctx.refs[h.frame_to_show_map_idx as usize];
            if !slot.valid {
                return Err(malformed("show_existing_frame of empty slot"));
            }
            h.frame_type = slot.frame_type;
            if h.frame_type == KEY_FRAME {
                h.refresh_frame_flags = all_frames;
            }
            if seq.film_grain_params_present {
                h.grain = slot.grain.clone();
            }
            h.show_frame = true;
            h.header_bytes = r.byte_pos();
            return Ok(h);
        }
        h.frame_type = r.f(2)? as u8;
        h.frame_is_intra = h.frame_type == INTRA_ONLY_FRAME || h.frame_type == KEY_FRAME;
        h.show_frame = r.flag()?;
        if h.show_frame && seq.decoder_model_info_present && !seq.equal_picture_interval {
            r.f(seq.frame_presentation_time_length)?;
        }
        h.showable_frame = if h.show_frame {
            h.frame_type != KEY_FRAME
        } else {
            r.flag()?
        };
        h.error_resilient_mode =
            if h.frame_type == SWITCH_FRAME || (h.frame_type == KEY_FRAME && h.show_frame) {
                true
            } else {
                r.flag()?
            };
    }
    if h.frame_type == KEY_FRAME && h.show_frame {
        ref_valid = [false; 8];
        ref_order_hint = [0; 8];
    }
    h.disable_cdf_update = r.flag()?;
    h.allow_screen_content_tools =
        if seq.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS {
            r.flag()?
        } else {
            seq.seq_force_screen_content_tools != 0
        };
    if h.allow_screen_content_tools {
        h.force_integer_mv = if seq.seq_force_integer_mv == SELECT_INTEGER_MV {
            r.flag()?
        } else {
            seq.seq_force_integer_mv != 0
        };
    }
    if h.frame_is_intra {
        h.force_integer_mv = true;
    }
    if seq.frame_id_numbers_present {
        h.current_frame_id = r.f(id_len)?;
        // mark_ref_frames
        let diff_len = seq.delta_frame_id_length;
        for i in 0..8 {
            let rid = ctx.refs[i].frame_id;
            let cur = h.current_frame_id;
            if cur > (1 << diff_len) {
                if rid > cur || rid < cur - (1 << diff_len) {
                    ref_valid[i] = false;
                }
            } else if rid > cur && rid < (1 << id_len) + cur - (1 << diff_len) {
                ref_valid[i] = false;
            }
        }
    }
    h.frame_size_override_flag = if h.frame_type == SWITCH_FRAME {
        true
    } else if seq.reduced_still_picture_header {
        false
    } else {
        r.flag()?
    };
    h.order_hint = r.f(seq.order_hint_bits)?;
    h.primary_ref_frame = if h.frame_is_intra || h.error_resilient_mode {
        PRIMARY_REF_NONE
    } else {
        r.f(3)? as u8
    };
    if seq.decoder_model_info_present {
        let present = r.flag()?;
        if present {
            for op in &seq.operating_points {
                if op.decoder_model_present {
                    let idc = u32::from(op.idc);
                    let in_t = (idc >> ctx.temporal_id) & 1;
                    let in_s = (idc >> (u32::from(ctx.spatial_id) + 8)) & 1;
                    if idc == 0 || (in_t != 0 && in_s != 0) {
                        r.f(seq.buffer_removal_time_length)?;
                    }
                }
            }
        }
    }
    h.refresh_frame_flags =
        if h.frame_type == SWITCH_FRAME || (h.frame_type == KEY_FRAME && h.show_frame) {
            all_frames
        } else {
            r.f(8)? as u8
        };
    if (!h.frame_is_intra || h.refresh_frame_flags != all_frames)
        && h.error_resilient_mode
        && seq.enable_order_hint
    {
        for i in 0..8 {
            let v = r.f(seq.order_hint_bits)?;
            h.ref_order_hint[i] = v;
            if v != ref_order_hint[i] {
                ref_valid[i] = false;
                // 7.20: the reference is replaced by a frame with this hint.
                ref_order_hint[i] = v;
            }
        }
    }
    h.ref_order_hint = ref_order_hint;
    if h.frame_is_intra {
        frame_size(&mut r, seq, &mut h)?;
        render_size(&mut r, &mut h)?;
        if h.allow_screen_content_tools && h.upscaled_width == h.frame_width {
            h.allow_intrabc = r.flag()?;
        }
    } else {
        let mut short_signaling = false;
        if seq.enable_order_hint {
            short_signaling = r.flag()?;
            if short_signaling {
                let last = r.f(3)? as u8;
                let gold = r.f(3)? as u8;
                h.ref_frame_idx = set_frame_refs(seq, &ref_order_hint, h.order_hint, last, gold)?;
            }
        }
        for i in 0..REFS_PER_FRAME {
            if !short_signaling {
                h.ref_frame_idx[i] = r.f(3)? as u8;
            }
            if seq.frame_id_numbers_present {
                let delta = r.f(seq.delta_frame_id_length)? + 1;
                let expected = (h.current_frame_id + (1 << id_len) - delta) % (1 << id_len);
                if ctx.refs[h.ref_frame_idx[i] as usize].frame_id != expected {
                    return Err(malformed("reference frame id"));
                }
            }
        }
        for i in 0..REFS_PER_FRAME {
            if !ref_valid[h.ref_frame_idx[i] as usize] {
                return Err(malformed("inter frame names an invalid reference"));
            }
        }
        if h.frame_size_override_flag && !h.error_resilient_mode {
            // frame_size_with_refs
            let mut found = false;
            for i in 0..REFS_PER_FRAME {
                if r.flag()? {
                    let s = &ctx.refs[h.ref_frame_idx[i] as usize];
                    h.upscaled_width = s.upscaled_width;
                    h.frame_width = h.upscaled_width;
                    h.frame_height = s.frame_height;
                    h.render_width = s.render_width;
                    h.render_height = s.render_height;
                    found = true;
                    break;
                }
            }
            if found {
                superres_params(&mut r, seq, &mut h)?;
                compute_image_size(&mut h);
            } else {
                frame_size(&mut r, seq, &mut h)?;
                render_size(&mut r, &mut h)?;
            }
        } else {
            frame_size(&mut r, seq, &mut h)?;
            render_size(&mut r, &mut h)?;
        }
        h.allow_high_precision_mv = if h.force_integer_mv { false } else { r.flag()? };
        let switchable = r.flag()?;
        h.interpolation_filter = if switchable {
            SWITCHABLE
        } else {
            r.f(2)? as u8
        };
        h.is_motion_mode_switchable = r.flag()?;
        h.use_ref_frame_mvs = if h.error_resilient_mode || !seq.enable_ref_frame_mvs {
            false
        } else {
            r.flag()?
        };
        for i in 0..REFS_PER_FRAME {
            let hint = ref_order_hint[h.ref_frame_idx[i] as usize];
            h.order_hints[i + 1] = hint;
            h.ref_frame_sign_bias[i + 1] =
                seq.enable_order_hint && relative_dist(seq, hint, h.order_hint) > 0;
        }
    }
    h.disable_frame_end_update_cdf = if seq.reduced_still_picture_header || h.disable_cdf_update {
        true
    } else {
        r.flag()?
    };
    // load_previous(): previous gm params, loop filter deltas, segmentation feature data.
    let prev = if h.primary_ref_frame == PRIMARY_REF_NONE {
        None
    } else {
        let slot = &ctx.refs[h.ref_frame_idx[h.primary_ref_frame as usize] as usize];
        if !slot.valid {
            return Err(malformed("primary reference frame"));
        }
        Some(slot)
    };
    let prev_gm = prev.map_or_else(default_gm_params, |s| s.gm_params);
    h.loop_filter.ref_deltas = prev.map_or(DEFAULT_LF_REF_DELTAS, |s| s.loop_filter_ref_deltas);
    h.loop_filter.mode_deltas = prev.map_or([0; 2], |s| s.loop_filter_mode_deltas);
    if let Some(s) = prev {
        h.segmentation.feature_enabled = s.feature_enabled;
        h.segmentation.feature_data = s.feature_data;
    }
    tile_info(&mut r, seq, &mut h)?;
    quantization_params(&mut r, seq, &mut h)?;
    segmentation_params(&mut r, &mut h)?;
    // delta_q_params
    if h.quant.base_q_idx > 0 {
        h.delta_q_present = r.flag()?;
    }
    if h.delta_q_present {
        h.delta_q_res = r.f(2)? as u8;
    }
    // delta_lf_params
    if h.delta_q_present {
        if !h.allow_intrabc {
            h.delta_lf_present = r.flag()?;
        }
        if h.delta_lf_present {
            h.delta_lf_res = r.f(2)? as u8;
            h.delta_lf_multi = r.flag()?;
        }
    }
    h.coded_lossless = true;
    for seg in 0..MAX_SEGMENTS {
        let q = h.qindex(seg);
        let qp = &h.quant;
        let lossless = q == 0
            && qp.delta_q_y_dc == 0
            && qp.delta_q_u_ac == 0
            && qp.delta_q_u_dc == 0
            && qp.delta_q_v_ac == 0
            && qp.delta_q_v_dc == 0;
        h.lossless_array[seg] = lossless;
        if !lossless {
            h.coded_lossless = false;
        }
        if qp.using_qmatrix {
            if lossless {
                h.seg_qm_level[0][seg] = 15;
                h.seg_qm_level[1][seg] = 15;
                h.seg_qm_level[2][seg] = 15;
            } else {
                h.seg_qm_level[0][seg] = qp.qm_y;
                h.seg_qm_level[1][seg] = qp.qm_u;
                h.seg_qm_level[2][seg] = qp.qm_v;
            }
        }
    }
    h.all_lossless = h.coded_lossless && h.frame_width == h.upscaled_width;
    loop_filter_params(&mut r, seq, &mut h)?;
    cdef_params(&mut r, seq, &mut h)?;
    lr_params(&mut r, seq, &mut h)?;
    // read_tx_mode
    h.tx_mode = if h.coded_lossless {
        ONLY_4X4
    } else if r.flag()? {
        TX_MODE_SELECT
    } else {
        TX_MODE_LARGEST
    };
    h.reference_select = if h.frame_is_intra { false } else { r.flag()? };
    skip_mode_params(&mut r, seq, &mut h, &ref_order_hint)?;
    h.allow_warped_motion =
        if h.frame_is_intra || h.error_resilient_mode || !seq.enable_warped_motion {
            false
        } else {
            r.flag()?
        };
    h.reduced_tx_set = r.flag()?;
    global_motion_params(&mut r, &mut h, &prev_gm)?;
    film_grain_params(&mut r, seq, &mut h, ctx.refs)?;
    r.byte_alignment();
    h.header_bytes = r.position() / 8;
    Ok(h)
}

fn superres_params(r: &mut BitReader<'_>, seq: &SequenceHeader, h: &mut FrameHeader) -> R<()> {
    h.use_superres = if seq.enable_superres {
        r.flag()?
    } else {
        false
    };
    h.superres_denom = if h.use_superres { r.f(3)? + 9 } else { 8 };
    h.upscaled_width = h.frame_width;
    h.frame_width = (h.upscaled_width * 8 + h.superres_denom / 2) / h.superres_denom;
    Ok(())
}

fn compute_image_size(h: &mut FrameHeader) {
    h.mi_cols = 2 * ((h.frame_width + 7) >> 3);
    h.mi_rows = 2 * ((h.frame_height + 7) >> 3);
}

fn frame_size(r: &mut BitReader<'_>, seq: &SequenceHeader, h: &mut FrameHeader) -> R<()> {
    if h.frame_size_override_flag {
        h.frame_width = r.f(seq.frame_width_bits)? + 1;
        h.frame_height = r.f(seq.frame_height_bits)? + 1;
    } else {
        h.frame_width = seq.max_frame_width;
        h.frame_height = seq.max_frame_height;
    }
    superres_params(r, seq, h)?;
    compute_image_size(h);
    Ok(())
}

fn render_size(r: &mut BitReader<'_>, h: &mut FrameHeader) -> R<()> {
    if r.flag()? {
        h.render_width = r.f(16)? + 1;
        h.render_height = r.f(16)? + 1;
    } else {
        h.render_width = h.upscaled_width;
        h.render_height = h.frame_height;
    }
    Ok(())
}

fn set_frame_refs(
    seq: &SequenceHeader,
    ref_order_hint: &[u32; 8],
    order_hint: u32,
    last: u8,
    gold: u8,
) -> R<[u8; 7]> {
    let mut idx = [-1i32; 7];
    idx[0] = i32::from(last);
    idx[3] = i32::from(gold);
    let mut used = [false; 8];
    used[last as usize] = true;
    used[gold as usize] = true;
    let cur = 1i32 << (seq.order_hint_bits - 1);
    let shifted: [i32; 8] =
        std::array::from_fn(|i| cur + relative_dist(seq, ref_order_hint[i], order_hint));
    if shifted[last as usize] >= cur || shifted[gold as usize] >= cur {
        return Err(malformed("short reference signaling"));
    }
    let find_latest_backward = |used: &[bool; 8]| {
        let mut r = -1i32;
        let mut latest = 0;
        for i in 0..8 {
            let hnt = shifted[i];
            if !used[i] && hnt >= cur && (r < 0 || hnt >= latest) {
                r = i as i32;
                latest = hnt;
            }
        }
        r
    };
    let find_earliest_backward = |used: &[bool; 8]| {
        let mut r = -1i32;
        let mut earliest = 0;
        for i in 0..8 {
            let hnt = shifted[i];
            if !used[i] && hnt >= cur && (r < 0 || hnt < earliest) {
                r = i as i32;
                earliest = hnt;
            }
        }
        r
    };
    let find_latest_forward = |used: &[bool; 8]| {
        let mut r = -1i32;
        let mut latest = 0;
        for i in 0..8 {
            let hnt = shifted[i];
            if !used[i] && hnt < cur && (r < 0 || hnt >= latest) {
                r = i as i32;
                latest = hnt;
            }
        }
        r
    };
    let rf = find_latest_backward(&used);
    if rf >= 0 {
        idx[6] = rf;
        used[rf as usize] = true;
    }
    let rf = find_earliest_backward(&used);
    if rf >= 0 {
        idx[4] = rf;
        used[rf as usize] = true;
    }
    let rf = find_earliest_backward(&used);
    if rf >= 0 {
        idx[5] = rf;
        used[rf as usize] = true;
    }
    // LAST2, LAST3, BWDREF, ALTREF2, ALTREF
    for ref_frame in [2usize, 3, 5, 6, 7] {
        if idx[ref_frame - 1] < 0 {
            let rf = find_latest_forward(&used);
            if rf >= 0 {
                idx[ref_frame - 1] = rf;
                used[rf as usize] = true;
            }
        }
    }
    let mut rf = -1i32;
    let mut earliest = 0;
    for i in 0..8 {
        let hnt = shifted[i];
        if rf < 0 || hnt < earliest {
            rf = i as i32;
            earliest = hnt;
        }
    }
    let mut out = [0u8; 7];
    for i in 0..7 {
        out[i] = if idx[i] < 0 { rf as u8 } else { idx[i] as u8 };
    }
    Ok(out)
}

fn tile_info(r: &mut BitReader<'_>, seq: &SequenceHeader, h: &mut FrameHeader) -> R<()> {
    let sb128 = seq.use_128x128_superblock;
    let sb_cols = if sb128 {
        (h.mi_cols + 31) >> 5
    } else {
        (h.mi_cols + 15) >> 4
    };
    let sb_rows = if sb128 {
        (h.mi_rows + 31) >> 5
    } else {
        (h.mi_rows + 15) >> 4
    };
    let sb_shift = if sb128 { 5 } else { 4 };
    let sb_size = sb_shift + 2;
    let max_tile_width_sb = 4096 >> sb_size;
    let mut max_tile_area_sb = (4096 * 2304) >> (2 * sb_size);
    let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS as u32));
    let max_log2_tile_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS as u32));
    let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));
    let t = &mut h.tile_info;
    t.mi_col_starts.clear();
    t.mi_row_starts.clear();
    let uniform = r.flag()?;
    if uniform {
        t.cols_log2 = min_log2_tile_cols;
        while t.cols_log2 < max_log2_tile_cols {
            if r.flag()? {
                t.cols_log2 += 1;
            } else {
                break;
            }
        }
        let tile_width_sb = (sb_cols + (1 << t.cols_log2) - 1) >> t.cols_log2;
        let mut start = 0;
        while start < sb_cols {
            t.mi_col_starts.push(start << sb_shift);
            start += tile_width_sb;
        }
        t.mi_col_starts.push(h.mi_cols);
        t.cols = t.mi_col_starts.len() - 1;
        let min_log2_tile_rows = min_log2_tiles.saturating_sub(t.cols_log2);
        t.rows_log2 = min_log2_tile_rows;
        while t.rows_log2 < max_log2_tile_rows {
            if r.flag()? {
                t.rows_log2 += 1;
            } else {
                break;
            }
        }
        let tile_height_sb = (sb_rows + (1 << t.rows_log2) - 1) >> t.rows_log2;
        let mut start = 0;
        while start < sb_rows {
            t.mi_row_starts.push(start << sb_shift);
            start += tile_height_sb;
        }
        t.mi_row_starts.push(h.mi_rows);
        t.rows = t.mi_row_starts.len() - 1;
    } else {
        let mut widest = 0;
        let mut start = 0;
        while start < sb_cols {
            t.mi_col_starts.push(start << sb_shift);
            let max_w = (sb_cols - start).min(max_tile_width_sb);
            let size = r.ns(max_w)? + 1;
            widest = widest.max(size);
            start += size;
            if t.mi_col_starts.len() > MAX_TILE_COLS {
                return Err(malformed("tile columns"));
            }
        }
        t.mi_col_starts.push(h.mi_cols);
        t.cols = t.mi_col_starts.len() - 1;
        t.cols_log2 = tile_log2(1, t.cols as u32);
        max_tile_area_sb = if min_log2_tiles > 0 {
            (sb_rows * sb_cols) >> (min_log2_tiles + 1)
        } else {
            sb_rows * sb_cols
        };
        let max_tile_height_sb = (max_tile_area_sb / widest).max(1);
        let mut start = 0;
        while start < sb_rows {
            t.mi_row_starts.push(start << sb_shift);
            let max_h = (sb_rows - start).min(max_tile_height_sb);
            let size = r.ns(max_h)? + 1;
            start += size;
            if t.mi_row_starts.len() > MAX_TILE_ROWS {
                return Err(malformed("tile rows"));
            }
        }
        t.mi_row_starts.push(h.mi_rows);
        t.rows = t.mi_row_starts.len() - 1;
        t.rows_log2 = tile_log2(1, t.rows as u32);
    }
    if t.cols_log2 > 0 || t.rows_log2 > 0 {
        t.context_update_tile_id = r.f(t.rows_log2 + t.cols_log2)?;
        if t.context_update_tile_id as usize >= t.cols * t.rows {
            return Err(malformed("context_update_tile_id"));
        }
        t.tile_size_bytes = r.f(2)? + 1;
    } else {
        t.context_update_tile_id = 0;
        t.tile_size_bytes = 4;
    }
    Ok(())
}

fn read_delta_q(r: &mut BitReader<'_>) -> R<i8> {
    if r.flag()? { Ok(r.su(7)? as i8) } else { Ok(0) }
}

fn quantization_params(r: &mut BitReader<'_>, seq: &SequenceHeader, h: &mut FrameHeader) -> R<()> {
    let q = &mut h.quant;
    q.base_q_idx = r.f(8)? as u8;
    q.delta_q_y_dc = read_delta_q(r)?;
    if seq.num_planes() > 1 {
        let diff_uv = if seq.separate_uv_delta_q {
            r.flag()?
        } else {
            false
        };
        q.delta_q_u_dc = read_delta_q(r)?;
        q.delta_q_u_ac = read_delta_q(r)?;
        if diff_uv {
            q.delta_q_v_dc = read_delta_q(r)?;
            q.delta_q_v_ac = read_delta_q(r)?;
        } else {
            q.delta_q_v_dc = q.delta_q_u_dc;
            q.delta_q_v_ac = q.delta_q_u_ac;
        }
    }
    q.using_qmatrix = r.flag()?;
    if q.using_qmatrix {
        q.qm_y = r.f(4)? as u8;
        q.qm_u = r.f(4)? as u8;
        q.qm_v = if seq.separate_uv_delta_q {
            r.f(4)? as u8
        } else {
            q.qm_u
        };
    }
    Ok(())
}

fn segmentation_params(r: &mut BitReader<'_>, h: &mut FrameHeader) -> R<()> {
    let primary_none = h.primary_ref_frame == PRIMARY_REF_NONE;
    let s = &mut h.segmentation;
    s.enabled = r.flag()?;
    if s.enabled {
        if primary_none {
            s.update_map = true;
            s.temporal_update = false;
            s.update_data = true;
        } else {
            s.update_map = r.flag()?;
            if s.update_map {
                s.temporal_update = r.flag()?;
            }
            s.update_data = r.flag()?;
        }
        if s.update_data {
            for i in 0..MAX_SEGMENTS {
                for j in 0..SEG_LVL_MAX {
                    let enabled = r.flag()?;
                    s.feature_enabled[i][j] = enabled;
                    let mut v = 0;
                    if enabled {
                        let bits = SEG_FEATURE_BITS[j];
                        let limit = SEG_FEATURE_MAX[j];
                        if SEG_FEATURE_SIGNED[j] {
                            v = r.su(1 + bits)?.clamp(-limit, limit);
                        } else {
                            v = (r.f(bits)? as i32).clamp(0, limit);
                        }
                    }
                    s.feature_data[i][j] = v as i16;
                }
            }
        }
    } else {
        s.feature_enabled = [[false; 8]; 8];
        s.feature_data = [[0; 8]; 8];
    }
    s.seg_id_pre_skip = false;
    s.last_active_seg_id = 0;
    for i in 0..MAX_SEGMENTS {
        for j in 0..SEG_LVL_MAX {
            if s.feature_enabled[i][j] {
                s.last_active_seg_id = i as u8;
                if j >= SEG_LVL_REF_FRAME {
                    s.seg_id_pre_skip = true;
                }
            }
        }
    }
    Ok(())
}

fn loop_filter_params(r: &mut BitReader<'_>, seq: &SequenceHeader, h: &mut FrameHeader) -> R<()> {
    let lf = &mut h.loop_filter;
    if h.coded_lossless || h.allow_intrabc {
        lf.level = [0; 4];
        lf.ref_deltas = DEFAULT_LF_REF_DELTAS;
        lf.mode_deltas = [0; 2];
        return Ok(());
    }
    lf.level[0] = r.f(6)? as u8;
    lf.level[1] = r.f(6)? as u8;
    if seq.num_planes() > 1 && (lf.level[0] != 0 || lf.level[1] != 0) {
        lf.level[2] = r.f(6)? as u8;
        lf.level[3] = r.f(6)? as u8;
    }
    lf.sharpness = r.f(3)? as u8;
    lf.delta_enabled = r.flag()?;
    if lf.delta_enabled {
        lf.delta_update = r.flag()?;
        if lf.delta_update {
            for i in 0..8 {
                if r.flag()? {
                    lf.ref_deltas[i] = r.su(7)? as i8;
                }
            }
            for i in 0..2 {
                if r.flag()? {
                    lf.mode_deltas[i] = r.su(7)? as i8;
                }
            }
        }
    }
    Ok(())
}

fn cdef_params(r: &mut BitReader<'_>, seq: &SequenceHeader, h: &mut FrameHeader) -> R<()> {
    let c = &mut h.cdef;
    if h.coded_lossless || h.allow_intrabc || !seq.enable_cdef {
        *c = Cdef {
            damping: 3,
            ..Cdef::default()
        };
        return Ok(());
    }
    c.damping = r.f(2)? as u8 + 3;
    c.bits = r.f(2)? as u8;
    for i in 0..(1usize << c.bits) {
        c.y_pri[i] = r.f(4)? as u8;
        c.y_sec[i] = r.f(2)? as u8;
        if c.y_sec[i] == 3 {
            c.y_sec[i] += 1;
        }
        if seq.num_planes() > 1 {
            c.uv_pri[i] = r.f(4)? as u8;
            c.uv_sec[i] = r.f(2)? as u8;
            if c.uv_sec[i] == 3 {
                c.uv_sec[i] += 1;
            }
        }
    }
    Ok(())
}

fn lr_params(r: &mut BitReader<'_>, seq: &SequenceHeader, h: &mut FrameHeader) -> R<()> {
    let lr = &mut h.lr;
    *lr = Restoration::default();
    if h.all_lossless || h.allow_intrabc || !seq.enable_restoration {
        return Ok(());
    }
    const REMAP: [u8; 4] = [
        RESTORE_NONE,
        RESTORE_SWITCHABLE,
        RESTORE_WIENER,
        RESTORE_SGRPROJ,
    ];
    let mut uses_chroma = false;
    for i in 0..seq.num_planes() {
        let t = REMAP[r.f(2)? as usize];
        lr.frame_type[i] = t;
        if t != RESTORE_NONE {
            lr.uses_lr = true;
            if i > 0 {
                uses_chroma = true;
            }
        }
    }
    if lr.uses_lr {
        let mut shift = r.f(1)?;
        if seq.use_128x128_superblock {
            shift += 1;
        } else if shift != 0 {
            shift += r.f(1)?;
        }
        lr.unit_size[0] = 256 >> (2 - shift);
        let uv_shift = if seq.subsampling_x && seq.subsampling_y && uses_chroma {
            r.f(1)?
        } else {
            0
        };
        lr.unit_size[1] = lr.unit_size[0] >> uv_shift;
        lr.unit_size[2] = lr.unit_size[0] >> uv_shift;
    }
    Ok(())
}

fn skip_mode_params(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
    ref_order_hint: &[u32; 8],
) -> R<()> {
    let mut allowed = false;
    if !(h.frame_is_intra || !h.reference_select || !seq.enable_order_hint) {
        let mut fwd_idx = -1i32;
        let mut bwd_idx = -1i32;
        let mut fwd_hint = 0u32;
        let mut bwd_hint = 0u32;
        for i in 0..REFS_PER_FRAME {
            let hint = ref_order_hint[h.ref_frame_idx[i] as usize];
            if relative_dist(seq, hint, h.order_hint) < 0 {
                if fwd_idx < 0 || relative_dist(seq, hint, fwd_hint) > 0 {
                    fwd_idx = i as i32;
                    fwd_hint = hint;
                }
            } else if relative_dist(seq, hint, h.order_hint) > 0
                && (bwd_idx < 0 || relative_dist(seq, hint, bwd_hint) < 0)
            {
                bwd_idx = i as i32;
                bwd_hint = hint;
            }
        }
        if fwd_idx < 0 {
            allowed = false;
        } else if bwd_idx >= 0 {
            allowed = true;
            h.skip_mode_frame = [
                1 + fwd_idx.min(bwd_idx) as u8,
                1 + fwd_idx.max(bwd_idx) as u8,
            ];
        } else {
            let mut sec_idx = -1i32;
            let mut sec_hint = 0u32;
            for i in 0..REFS_PER_FRAME {
                let hint = ref_order_hint[h.ref_frame_idx[i] as usize];
                if relative_dist(seq, hint, fwd_hint) < 0
                    && (sec_idx < 0 || relative_dist(seq, hint, sec_hint) > 0)
                {
                    sec_idx = i as i32;
                    sec_hint = hint;
                }
            }
            if sec_idx >= 0 {
                allowed = true;
                h.skip_mode_frame = [
                    1 + fwd_idx.min(sec_idx) as u8,
                    1 + fwd_idx.max(sec_idx) as u8,
                ];
            }
        }
    }
    h.skip_mode_present = if allowed { r.flag()? } else { false };
    Ok(())
}

fn decode_subexp(r: &mut BitReader<'_>, num_syms: u32) -> R<u32> {
    let mut i = 0u32;
    let mut mk = 0u32;
    let k = 3u32;
    loop {
        let b2 = if i != 0 { k + i - 1 } else { k };
        let a = 1u32 << b2;
        if num_syms <= mk + 3 * a {
            return Ok(r.ns(num_syms - mk)? + mk);
        }
        if r.flag()? {
            i += 1;
            mk += a;
        } else {
            return Ok(r.f(b2)? + mk);
        }
    }
}

fn inverse_recenter(r: i32, v: i32) -> i32 {
    if v > 2 * r {
        v
    } else if v & 1 != 0 {
        r - ((v + 1) >> 1)
    } else {
        r + (v >> 1)
    }
}

fn decode_unsigned_subexp_with_ref(rd: &mut BitReader<'_>, mx: i32, r: i32) -> R<i32> {
    let v = decode_subexp(rd, mx as u32)? as i32;
    if (r << 1) <= mx {
        Ok(inverse_recenter(r, v))
    } else {
        Ok(mx - 1 - inverse_recenter(mx - 1 - r, v))
    }
}

fn decode_signed_subexp_with_ref(rd: &mut BitReader<'_>, low: i32, high: i32, r: i32) -> R<i32> {
    let x = decode_unsigned_subexp_with_ref(rd, high - low, r - low)?;
    Ok(x + low)
}

fn global_motion_params(r: &mut BitReader<'_>, h: &mut FrameHeader, prev: &[[i32; 6]; 8]) -> R<()> {
    h.gm_params = default_gm_params();
    h.gm_type = [IDENTITY; 8];
    if h.frame_is_intra {
        return Ok(());
    }
    for rf in 1..8 {
        let ty = if r.flag()? {
            if r.flag()? {
                ROTZOOM
            } else if r.flag()? {
                TRANSLATION
            } else {
                AFFINE
            }
        } else {
            IDENTITY
        };
        h.gm_type[rf] = ty;
        let mut read = |idx: usize, h: &mut FrameHeader| -> R<()> {
            let mut abs_bits = 12u32;
            let mut prec_bits = 15u32;
            if idx < 2 {
                if ty == TRANSLATION {
                    let hp = u32::from(!h.allow_high_precision_mv);
                    abs_bits = 9 - hp;
                    prec_bits = 3 - hp;
                } else {
                    abs_bits = 12;
                    prec_bits = 6;
                }
            }
            let prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
            let round = if idx % 3 == 2 {
                1i32 << WARPEDMODEL_PREC_BITS
            } else {
                0
            };
            let sub = if idx % 3 == 2 { 1i32 << prec_bits } else { 0 };
            let mx = 1i32 << abs_bits;
            let rr = (prev[rf][idx] >> prec_diff) - sub;
            h.gm_params[rf][idx] =
                (decode_signed_subexp_with_ref(r, -mx, mx + 1, rr)? << prec_diff) + round;
            Ok(())
        };
        if ty >= ROTZOOM {
            read(2, h)?;
            read(3, h)?;
            if ty == AFFINE {
                read(4, h)?;
                read(5, h)?;
            } else {
                h.gm_params[rf][4] = -h.gm_params[rf][3];
                h.gm_params[rf][5] = h.gm_params[rf][2];
            }
        }
        if ty >= TRANSLATION {
            read(0, h)?;
            read(1, h)?;
        }
    }
    Ok(())
}

fn film_grain_params(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
    refs: &[RefHeaderState; 8],
) -> R<()> {
    h.grain = GrainParams::default();
    if !seq.film_grain_params_present || (!h.show_frame && !h.showable_frame) {
        return Ok(());
    }
    let g = &mut h.grain;
    g.apply_grain = r.flag()?;
    if !g.apply_grain {
        *g = GrainParams::default();
        return Ok(());
    }
    g.grain_seed = r.f(16)? as u16;
    g.update_grain = if h.frame_type == INTER_FRAME {
        r.flag()?
    } else {
        true
    };
    if !g.update_grain {
        let idx = r.f(3)? as usize;
        // It is a requirement that film_grain_params_ref_idx is one of ref_frame_idx.
        if !h.ref_frame_idx.iter().any(|&x| x as usize == idx) {
            return Err(malformed("film_grain_params_ref_idx"));
        }
        let seed = g.grain_seed;
        *g = refs[idx].grain.clone();
        g.grain_seed = seed;
        g.apply_grain = true;
        g.update_grain = false;
        return Ok(());
    }
    g.num_y_points = r.f(4)? as u8;
    if g.num_y_points > 14 {
        return Err(malformed("num_y_points"));
    }
    for i in 0..g.num_y_points as usize {
        g.point_y_value[i] = r.f(8)? as u8;
        g.point_y_scaling[i] = r.f(8)? as u8;
    }
    g.chroma_scaling_from_luma = if seq.mono_chrome { false } else { r.flag()? };
    if seq.mono_chrome
        || g.chroma_scaling_from_luma
        || (seq.subsampling_x && seq.subsampling_y && g.num_y_points == 0)
    {
        g.num_cb_points = 0;
        g.num_cr_points = 0;
    } else {
        g.num_cb_points = r.f(4)? as u8;
        if g.num_cb_points > 10 {
            return Err(malformed("num_cb_points"));
        }
        for i in 0..g.num_cb_points as usize {
            g.point_cb_value[i] = r.f(8)? as u8;
            g.point_cb_scaling[i] = r.f(8)? as u8;
        }
        g.num_cr_points = r.f(4)? as u8;
        if g.num_cr_points > 10 {
            return Err(malformed("num_cr_points"));
        }
        for i in 0..g.num_cr_points as usize {
            g.point_cr_value[i] = r.f(8)? as u8;
            g.point_cr_scaling[i] = r.f(8)? as u8;
        }
    }
    g.grain_scaling_minus_8 = r.f(2)? as u8;
    g.ar_coeff_lag = r.f(2)? as u8;
    let lag = usize::from(g.ar_coeff_lag);
    let num_pos_luma = 2 * lag * (lag + 1);
    let num_pos_chroma = if g.num_y_points != 0 {
        for i in 0..num_pos_luma {
            g.ar_coeffs_y_plus_128[i] = r.f(8)? as u8;
        }
        num_pos_luma + 1
    } else {
        num_pos_luma
    };
    if g.chroma_scaling_from_luma || g.num_cb_points != 0 {
        for i in 0..num_pos_chroma {
            g.ar_coeffs_cb_plus_128[i] = r.f(8)? as u8;
        }
    }
    if g.chroma_scaling_from_luma || g.num_cr_points != 0 {
        for i in 0..num_pos_chroma {
            g.ar_coeffs_cr_plus_128[i] = r.f(8)? as u8;
        }
    }
    g.ar_coeff_shift_minus_6 = r.f(2)? as u8;
    g.grain_scale_shift = r.f(2)? as u8;
    if g.num_cb_points != 0 {
        g.cb_mult = r.f(8)? as u8;
        g.cb_luma_mult = r.f(8)? as u8;
        g.cb_offset = r.f(9)? as u16;
    }
    if g.num_cr_points != 0 {
        g.cr_mult = r.f(8)? as u8;
        g.cr_luma_mult = r.f(8)? as u8;
        g.cr_offset = r.f(9)? as u16;
    }
    g.overlap_flag = r.flag()?;
    g.clip_to_restricted_range = r.flag()?;
    Ok(())
}

/// One OBU inside a temporal unit.
pub struct Obu<'a> {
    pub obu_type: u8,
    pub temporal_id: u8,
    pub spatial_id: u8,
    pub has_extension: bool,
    pub payload: &'a [u8],
}

/// Converts one Annex B temporal unit (`temporal_unit( sz )`, spec Annex B)
/// into the low-overhead format: every OBU gets `obu_has_size_field = 1`.
pub fn annexb_to_low_overhead(data: &[u8]) -> R<Vec<u8>> {
    let mut out = Vec::with_capacity(data.len() + 16);
    let mut rest = data;
    let take = |buf: &mut &[u8], what: &'static str| -> R<usize> {
        let (v, n) = super::bits::leb128(buf)?;
        let v = usize::try_from(v).map_err(|_| CodecError::Limit { what })?;
        if n + v > buf.len() {
            return Err(CodecError::Truncated {
                at_bit: (buf.len() * 8) as u64,
                what,
            });
        }
        *buf = &buf[n..];
        Ok(v)
    };
    while !rest.is_empty() {
        let tu_size = take(&mut rest, "temporal_unit_size")?;
        let (mut tu, after) = rest.split_at(tu_size);
        rest = after;
        while !tu.is_empty() {
            let fu_size = take(&mut tu, "frame_unit_size")?;
            let (mut fu, after) = tu.split_at(fu_size);
            tu = after;
            while !fu.is_empty() {
                let obu_len = take(&mut fu, "obu_length")?;
                let (obu, after) = fu.split_at(obu_len);
                fu = after;
                let Some(&h) = obu.first() else {
                    return Err(malformed("empty obu"));
                };
                let ext = usize::from(h & 4 != 0);
                if obu.len() < 1 + ext {
                    return Err(malformed("obu extension"));
                }
                let mut body = &obu[1 + ext..];
                if h & 2 != 0 {
                    // Already sized: keep only the declared payload.
                    let n = take(&mut body, "obu_size")?;
                    body = &body[..n];
                }
                out.push(h | 2);
                out.extend_from_slice(&obu[1..1 + ext]);
                let mut v = body.len() as u64;
                loop {
                    let b = (v & 0x7f) as u8;
                    v >>= 7;
                    if v == 0 {
                        out.push(b);
                        break;
                    }
                    out.push(b | 0x80);
                }
                out.extend_from_slice(body);
            }
        }
    }
    Ok(out)
}

/// Splits a low-overhead bitstream (or one Annex-B-free temporal unit) into OBUs.
pub fn split_obus(mut data: &[u8]) -> R<Vec<Obu<'_>>> {
    let mut out = Vec::new();
    while !data.is_empty() {
        let h = data[0];
        if h & 0x80 != 0 {
            return Err(malformed("obu_forbidden_bit"));
        }
        let obu_type = (h >> 3) & 15;
        let has_extension = h & 4 != 0;
        let has_size = h & 2 != 0;
        let mut pos = 1;
        let (mut temporal_id, mut spatial_id) = (0, 0);
        if has_extension {
            let e = *data.get(1).ok_or(CodecError::Truncated {
                at_bit: 8,
                what: "obu extension",
            })?;
            temporal_id = e >> 5;
            spatial_id = (e >> 3) & 3;
            pos = 2;
        }
        let size = if has_size {
            let (v, n) = super::bits::leb128(&data[pos..])?;
            pos += n;
            usize::try_from(v).map_err(|_| CodecError::Limit { what: "obu size" })?
        } else {
            data.len() - pos
        };
        let end = pos
            .checked_add(size)
            .ok_or(CodecError::Limit { what: "obu size" })?;
        if end > data.len() {
            return Err(CodecError::Truncated {
                at_bit: (data.len() * 8) as u64,
                what: "obu payload",
            });
        }
        out.push(Obu {
            obu_type,
            temporal_id,
            spatial_id,
            has_extension,
            payload: &data[pos..end],
        });
        data = &data[end..];
    }
    Ok(out)
}

#[cfg(test)]
mod annexb_tests {
    use super::{annexb_to_low_overhead, split_obus};

    fn leb(v: usize) -> Vec<u8> {
        let mut out = Vec::new();
        let mut v = v as u64;
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return out;
            }
            out.push(b | 0x80);
        }
    }

    #[test]
    fn annexb_units_become_sized_obus() {
        // A temporal delimiter (no size field) and a padding OBU with one.
        let td = vec![0x10u8];
        let pad = vec![0x7a, 0x02, 0xaa, 0xbb];
        let mut fu = leb(td.len());
        fu.extend_from_slice(&td);
        fu.extend(leb(pad.len()));
        fu.extend_from_slice(&pad);
        let mut tu = leb(fu.len());
        tu.extend_from_slice(&fu);
        let mut data = leb(tu.len());
        data.extend_from_slice(&tu);
        let low = annexb_to_low_overhead(&data).expect("convert");
        assert_eq!(low, vec![0x12, 0x00, 0x7a, 0x02, 0xaa, 0xbb]);
        let obus = split_obus(&low).expect("split");
        assert_eq!(obus.len(), 2);
        assert_eq!(obus[1].payload, &[0xaa, 0xbb]);
        assert!(annexb_to_low_overhead(&[0x05, 0x01]).is_err());
    }

    #[test]
    fn annexb_stream_decodes_like_low_overhead() {
        use crate::dec::decoder::{AnyPicture, Av1Core};
        use crate::enc::encoder::{Av1Enc, EncConfig, InputFrame, RateControl};
        let (w, h) = (48usize, 32usize);
        let mut cfg = EncConfig::new(w, h);
        cfg.rc = RateControl::Qindex(80);
        cfg.gop = 4;
        let mut enc = Av1Enc::new(cfg).expect("encoder");
        let mut units = Vec::new();
        for f in 0..6u8 {
            let y: Vec<u8> = (0..w * h)
                .map(|i| (i as u8).wrapping_mul(3).wrapping_add(f * 9))
                .collect();
            let c = vec![128u8; (w / 2) * (h / 2)];
            units.extend(
                enc.encode(&InputFrame {
                    planes: [&y, &c, &c],
                    strides: [w, w / 2, w / 2],
                })
                .expect("encode"),
            );
        }
        units.extend(enc.flush::<u8>().expect("flush"));
        let (mut a, mut b) = (Av1Core::new(), Av1Core::new());
        let mut shown = 0;
        for u in &units {
            // Annex B: each OBU without its size field, length-prefixed, in one
            // frame unit per temporal unit.
            let mut fu = Vec::new();
            for obu in split_obus(&u.data).expect("split") {
                let mut o = vec![(obu.obu_type << 3) | if obu.has_extension { 4 } else { 0 }];
                o.extend_from_slice(obu.payload);
                fu.extend(leb(o.len()));
                fu.extend(o);
            }
            let mut tu = leb(fu.len());
            tu.extend(fu);
            let mut ab = leb(tu.len());
            ab.extend(tu);
            let x = a.decode(&u.data).expect("low overhead");
            let y = b.decode_annexb(&ab).expect("annex b");
            assert_eq!(x.len(), y.len());
            for (p, q) in x.iter().zip(&y) {
                match (&*p.picture, &*q.picture) {
                    (AnyPicture::U8(p), AnyPicture::U8(q)) => {
                        assert_eq!(p.planes[0].data, q.planes[0].data);
                    }
                    _ => panic!("8-bit pictures expected"),
                }
                shown += 1;
            }
        }
        assert_eq!(shown, 6);
    }
}

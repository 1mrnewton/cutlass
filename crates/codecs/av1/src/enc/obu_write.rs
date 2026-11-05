//! Sequence header, frame header and tile group writers (spec 5.5, 5.9,
//! 5.11.1): the inverses of `dec::obu::parse_sequence_header` and
//! `dec::obu::parse_frame_header`.
//!
//! The writers take a header as the encoder's intent. Syntax-level choices
//! that the parsed structs do not record (timing values, whether a delta is
//! coded although it equals the predicted value, uniform tile spacing, ...)
//! come from the `*WriteOptions` structs; by default the shortest coding is
//! chosen. `frame_refs_short_signaling` is always 0.

use super::bitwriter::{BitWriter, write_obu};
use crate::dec::obu::{
    AFFINE, DEFAULT_LF_REF_DELTAS, FrameHeader, GrainParams, HeaderContext, IDENTITY, INTER_FRAME,
    INTRA_ONLY_FRAME, KEY_FRAME, MAX_SEGMENTS, MAX_TILE_COLS, MAX_TILE_ROWS,
    OBU_TEMPORAL_DELIMITER, PRIMARY_REF_NONE, REFS_PER_FRAME, RESTORE_NONE, RESTORE_SGRPROJ,
    RESTORE_SWITCHABLE, RESTORE_WIENER, ROTZOOM, RefHeaderState, SEG_LVL_MAX, SWITCH_FRAME,
    SWITCHABLE, SequenceHeader, TRANSLATION, TX_MODE_SELECT, TileInfo, WARPEDMODEL_PREC_BITS,
    default_gm_params, relative_dist,
};

const SELECT_SCREEN_CONTENT_TOOLS: u8 = 2;
const SELECT_INTEGER_MV: u8 = 2;
const SEG_FEATURE_BITS: [u32; 8] = [8, 6, 6, 6, 6, 3, 0, 0];
const SEG_FEATURE_SIGNED: [bool; 8] = [true, true, true, true, true, false, false, false];

/// Sequence header values that `SequenceHeader` does not keep.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceHeaderWriteOptions {
    pub num_units_in_display_tick: u32,
    pub time_scale: u32,
    pub num_ticks_per_picture_minus_1: u32,
    /// buffer_delay_length_minus_1 (5 bits).
    pub buffer_delay_length_minus_1: u8,
    pub num_units_in_decoding_tick: u32,
    /// Per operating point (missing entries are 0 / false).
    pub decoder_buffer_delay: Vec<u32>,
    pub encoder_buffer_delay: Vec<u32>,
    pub low_delay_mode_flag: Vec<bool>,
    pub initial_display_delay_minus_1: Vec<u8>,
    /// Code initial_display_delay_present_flag = 1 even if no operating
    /// point carries a delay.
    pub force_initial_display_delay_present: bool,
    /// Code color_description_present_flag = 1 for the default (2, 2, 2).
    pub force_color_description: bool,
}

impl Default for SequenceHeaderWriteOptions {
    fn default() -> Self {
        Self {
            num_units_in_display_tick: 1001,
            time_scale: 30000,
            num_ticks_per_picture_minus_1: 0,
            buffer_delay_length_minus_1: 23,
            num_units_in_decoding_tick: 1001,
            decoder_buffer_delay: Vec::new(),
            encoder_buffer_delay: Vec::new(),
            low_delay_mode_flag: Vec::new(),
            initial_display_delay_minus_1: Vec::new(),
            force_initial_display_delay_present: false,
            force_color_description: false,
        }
    }
}

/// sequence_header_obu() followed by trailing_bits(): the OBU payload.
pub fn write_sequence_header(seq: &SequenceHeader) -> Vec<u8> {
    write_sequence_header_with(seq, &SequenceHeaderWriteOptions::default())
}

pub fn write_sequence_header_with(seq: &SequenceHeader, o: &SequenceHeaderWriteOptions) -> Vec<u8> {
    let mut w = BitWriter::new();
    let s = seq;
    w.f(3, u32::from(s.profile));
    w.flag(s.still_picture);
    w.flag(s.reduced_still_picture_header);
    if s.reduced_still_picture_header {
        let level = s.operating_points.first().map_or(0, |op| op.seq_level_idx);
        w.f(5, u32::from(level));
    } else {
        w.flag(s.timing_info_present);
        let buffer_delay_length = u32::from(o.buffer_delay_length_minus_1 & 31) + 1;
        if s.timing_info_present {
            w.f(32, o.num_units_in_display_tick);
            w.f(32, o.time_scale);
            w.flag(s.equal_picture_interval);
            if s.equal_picture_interval {
                w.uvlc(o.num_ticks_per_picture_minus_1);
            }
            w.flag(s.decoder_model_info_present);
            if s.decoder_model_info_present {
                w.f(5, buffer_delay_length - 1);
                w.f(32, o.num_units_in_decoding_tick);
                w.f(5, s.buffer_removal_time_length.saturating_sub(1));
                w.f(5, s.frame_presentation_time_length.saturating_sub(1));
            }
        }
        let idd_present = o.force_initial_display_delay_present
            || s.operating_points
                .iter()
                .any(|op| op.initial_display_delay_present);
        w.flag(idd_present);
        w.f(5, s.operating_points.len().saturating_sub(1) as u32);
        for (i, op) in s.operating_points.iter().enumerate() {
            w.f(12, u32::from(op.idc));
            w.f(5, u32::from(op.seq_level_idx));
            if op.seq_level_idx > 7 {
                w.f(1, u32::from(op.seq_tier));
            }
            if s.decoder_model_info_present {
                w.flag(op.decoder_model_present);
                if op.decoder_model_present {
                    w.f(
                        buffer_delay_length,
                        o.decoder_buffer_delay.get(i).copied().unwrap_or(0),
                    );
                    w.f(
                        buffer_delay_length,
                        o.encoder_buffer_delay.get(i).copied().unwrap_or(0),
                    );
                    w.flag(o.low_delay_mode_flag.get(i).copied().unwrap_or(false));
                }
            }
            if idd_present {
                w.flag(op.initial_display_delay_present);
                if op.initial_display_delay_present {
                    let d = o.initial_display_delay_minus_1.get(i).copied().unwrap_or(9);
                    w.f(4, u32::from(d));
                }
            }
        }
    }
    w.f(4, s.frame_width_bits - 1);
    w.f(4, s.frame_height_bits - 1);
    w.f(s.frame_width_bits, s.max_frame_width - 1);
    w.f(s.frame_height_bits, s.max_frame_height - 1);
    if !s.reduced_still_picture_header {
        w.flag(s.frame_id_numbers_present);
    }
    if s.frame_id_numbers_present {
        w.f(4, s.delta_frame_id_length - 2);
        w.f(3, s.additional_frame_id_length - 1);
    }
    w.flag(s.use_128x128_superblock);
    w.flag(s.enable_filter_intra);
    w.flag(s.enable_intra_edge_filter);
    if !s.reduced_still_picture_header {
        w.flag(s.enable_interintra_compound);
        w.flag(s.enable_masked_compound);
        w.flag(s.enable_warped_motion);
        w.flag(s.enable_dual_filter);
        w.flag(s.enable_order_hint);
        if s.enable_order_hint {
            w.flag(s.enable_jnt_comp);
            w.flag(s.enable_ref_frame_mvs);
        }
        if s.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS {
            w.flag(true);
        } else {
            w.flag(false);
            w.f(1, u32::from(s.seq_force_screen_content_tools));
        }
        if s.seq_force_screen_content_tools > 0 {
            if s.seq_force_integer_mv == SELECT_INTEGER_MV {
                w.flag(true);
            } else {
                w.flag(false);
                w.f(1, u32::from(s.seq_force_integer_mv));
            }
        }
        if s.enable_order_hint {
            w.f(3, s.order_hint_bits - 1);
        }
    }
    w.flag(s.enable_superres);
    w.flag(s.enable_cdef);
    w.flag(s.enable_restoration);
    // color_config()
    let high_bitdepth = s.bit_depth > 8;
    w.flag(high_bitdepth);
    if s.profile == 2 && high_bitdepth {
        w.flag(s.bit_depth == 12);
    }
    if s.profile != 1 {
        w.flag(s.mono_chrome);
    }
    let (cp, tc, mc) = (
        s.color_primaries,
        s.transfer_characteristics,
        s.matrix_coefficients,
    );
    let color_description = o.force_color_description || !(cp == 2 && tc == 2 && mc == 2);
    w.flag(color_description);
    if color_description {
        w.f(8, u32::from(cp));
        w.f(8, u32::from(tc));
        w.f(8, u32::from(mc));
    }
    if s.mono_chrome {
        w.flag(s.color_range);
    } else {
        if !(cp == 1 && tc == 13 && mc == 0) {
            w.flag(s.color_range);
            if s.profile == 2 && s.bit_depth == 12 {
                w.flag(s.subsampling_x);
                if s.subsampling_x {
                    w.flag(s.subsampling_y);
                }
            }
            if s.subsampling_x && s.subsampling_y {
                w.f(2, u32::from(s.chroma_sample_position));
            }
        }
        w.flag(s.separate_uv_delta_q);
    }
    w.flag(s.film_grain_params_present);
    w.trailing_bits();
    w.into_bytes()
}

/// How frame_size_with_refs() codes the size of an inter frame that uses
/// frame_size_override_flag outside error resilient mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SizeFromRef {
    /// found_ref = 1 for the first reference whose size and render size
    /// match the frame; explicit sizes otherwise.
    #[default]
    Auto,
    /// Always code the size explicitly.
    Never,
    /// found_ref = 1 for this reference (0..7); it must match the frame.
    Ref(usize),
}

/// Frame header syntax choices that `FrameHeader` does not record.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameHeaderWriteOptions {
    /// temporal_point_info() value.
    pub frame_presentation_time: u32,
    /// buffer_removal_time_present_flag = 1 with these values, indexed by
    /// operating point (missing entries are 0).
    pub buffer_removal_times: Option<Vec<u32>>,
    pub size_from_ref: SizeFromRef,
    /// render_and_frame_size_different = 1 even when the render size equals
    /// the frame size.
    pub force_render_size: bool,
    /// Some(true) / Some(false) forces uniform / explicit tile spacing;
    /// None picks uniform when `TileInfo` is a uniform layout of its
    /// `cols_log2` and `rows_log2` (or carries no starts).
    pub uniform_tile_spacing: Option<bool>,
    /// diff_uv_delta = 1 even when the V deltas equal the U deltas.
    pub force_diff_uv_delta: bool,
    /// update_ref_delta / update_mode_delta = 1 for every delta (otherwise
    /// only for deltas that differ from the previous frame's).
    pub lf_delta_update_all: bool,
    /// film_grain_params_ref_idx when update_grain = 0; None searches the
    /// frame's references for one with equal parameters.
    pub film_grain_params_ref_idx: Option<u8>,
}

/// Superblock geometry and the tile count limits of tile_info().
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileLimits {
    pub sb_cols: u32,
    pub sb_rows: u32,
    pub sb_shift: u32,
    pub max_tile_width_sb: u32,
    pub max_tile_area_sb: u32,
    pub min_log2_tile_cols: u32,
    pub max_log2_tile_cols: u32,
    pub max_log2_tile_rows: u32,
    pub min_log2_tiles: u32,
}

fn tile_log2(blk: u32, target: u32) -> u32 {
    let mut k = 0;
    while (blk << k) < target {
        k += 1;
    }
    k
}

pub fn tile_limits(seq: &SequenceHeader, mi_cols: u32, mi_rows: u32) -> TileLimits {
    let sb128 = seq.use_128x128_superblock;
    let sb_shift = if sb128 { 5 } else { 4 };
    let sb_cols = (mi_cols + (1 << sb_shift) - 1) >> sb_shift;
    let sb_rows = (mi_rows + (1 << sb_shift) - 1) >> sb_shift;
    let sb_size = sb_shift + 2;
    let max_tile_width_sb = 4096 >> sb_size;
    let max_tile_area_sb = (4096 * 2304) >> (2 * sb_size);
    let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
    TileLimits {
        sb_cols,
        sb_rows,
        sb_shift,
        max_tile_width_sb,
        max_tile_area_sb,
        min_log2_tile_cols,
        max_log2_tile_cols: tile_log2(1, sb_cols.min(MAX_TILE_COLS as u32)),
        max_log2_tile_rows: tile_log2(1, sb_rows.min(MAX_TILE_ROWS as u32)),
        min_log2_tiles: min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols)),
    }
}

impl TileLimits {
    /// MinLog2TileRows for a uniform layout with `cols_log2`.
    pub fn min_log2_tile_rows(&self, cols_log2: u32) -> u32 {
        self.min_log2_tiles.saturating_sub(cols_log2)
    }

    /// maxTileHeightSb of an explicit layout whose widest column is
    /// `widest_sb` superblocks.
    pub fn max_tile_height_sb(&self, widest_sb: u32) -> u32 {
        let area = if self.min_log2_tiles > 0 {
            (self.sb_rows * self.sb_cols) >> (self.min_log2_tiles + 1)
        } else {
            self.sb_rows * self.sb_cols
        };
        (area / widest_sb.max(1)).max(1)
    }
}

fn uniform_starts(sb_count: u32, log2: u32, sb_shift: u32, mi_count: u32) -> Vec<u32> {
    let size = (sb_count + (1 << log2) - 1) >> log2;
    let mut v = Vec::new();
    let mut start = 0;
    while start < sb_count {
        v.push(start << sb_shift);
        start += size.max(1);
    }
    v.push(mi_count);
    v
}

/// The TileInfo the parser derives for uniform spacing with the given log2
/// counts (clamped to the allowed range). `context_update_tile_id` is 0 and
/// `tile_size_bytes` is 4.
pub fn uniform_tile_info(
    seq: &SequenceHeader,
    mi_cols: u32,
    mi_rows: u32,
    cols_log2: u32,
    rows_log2: u32,
) -> TileInfo {
    let l = tile_limits(seq, mi_cols, mi_rows);
    let cols_log2 = cols_log2.clamp(
        l.min_log2_tile_cols,
        l.max_log2_tile_cols.max(l.min_log2_tile_cols),
    );
    let min_rows = l.min_log2_tile_rows(cols_log2);
    let rows_log2 = rows_log2.clamp(min_rows, l.max_log2_tile_rows.max(min_rows));
    let mi_col_starts = uniform_starts(l.sb_cols, cols_log2, l.sb_shift, mi_cols);
    let mi_row_starts = uniform_starts(l.sb_rows, rows_log2, l.sb_shift, mi_rows);
    TileInfo {
        cols: mi_col_starts.len() - 1,
        rows: mi_row_starts.len() - 1,
        cols_log2,
        rows_log2,
        mi_col_starts,
        mi_row_starts,
        context_update_tile_id: 0,
        tile_size_bytes: 4,
    }
}

/// The TileInfo the parser derives for explicit spacing with the given
/// column widths and row heights in superblocks.
pub fn explicit_tile_info(
    seq: &SequenceHeader,
    mi_cols: u32,
    mi_rows: u32,
    widths_sb: &[u32],
    heights_sb: &[u32],
) -> TileInfo {
    let l = tile_limits(seq, mi_cols, mi_rows);
    let starts = |sizes: &[u32], end: u32| {
        let mut v = Vec::with_capacity(sizes.len() + 1);
        let mut s = 0;
        for &x in sizes {
            v.push(s << l.sb_shift);
            s += x;
        }
        v.push(end);
        v
    };
    let cols = widths_sb.len();
    let rows = heights_sb.len();
    TileInfo {
        cols,
        rows,
        cols_log2: tile_log2(1, cols as u32),
        rows_log2: tile_log2(1, rows as u32),
        mi_col_starts: starts(widths_sb, mi_cols),
        mi_row_starts: starts(heights_sb, mi_rows),
        context_update_tile_id: 0,
        tile_size_bytes: 4,
    }
}

/// The smallest TileSizeBytes that holds `max_tile_len` (the largest tile
/// that is not last in its tile group).
pub fn tile_size_bytes_for(max_tile_len: usize) -> u32 {
    let v = max_tile_len.saturating_sub(1);
    if v < 1 << 8 {
        1
    } else if v < 1 << 16 {
        2
    } else if v < 1 << 24 {
        3
    } else {
        4
    }
}

/// Upscaled width to coded frame width (spec 7.16).
pub fn downscaled_width(upscaled_width: u32, superres_denom: u32) -> u32 {
    let d = superres_denom.max(1);
    (upscaled_width * 8 + d / 2) / d
}

/// SkipModeFrame[0..2] when skip mode is allowed (spec 5.9.22).
pub fn skip_mode_frames(
    seq: &SequenceHeader,
    h: &FrameHeader,
    ref_order_hint: &[u32; 8],
) -> Option<[u8; 2]> {
    let frame_is_intra = h.frame_type == KEY_FRAME || h.frame_type == INTRA_ONLY_FRAME;
    if frame_is_intra || !h.reference_select || !seq.enable_order_hint {
        return None;
    }
    let mut fwd_idx = -1i32;
    let mut bwd_idx = -1i32;
    let mut fwd_hint = 0u32;
    let mut bwd_hint = 0u32;
    for i in 0..REFS_PER_FRAME {
        let hint = ref_order_hint[h.ref_frame_idx[i] as usize & 7];
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
        return None;
    }
    if bwd_idx >= 0 {
        return Some([
            1 + fwd_idx.min(bwd_idx) as u8,
            1 + fwd_idx.max(bwd_idx) as u8,
        ]);
    }
    let mut sec_idx = -1i32;
    let mut sec_hint = 0u32;
    for i in 0..REFS_PER_FRAME {
        let hint = ref_order_hint[h.ref_frame_idx[i] as usize & 7];
        if relative_dist(seq, hint, fwd_hint) < 0
            && (sec_idx < 0 || relative_dist(seq, hint, sec_hint) > 0)
        {
            sec_idx = i as i32;
            sec_hint = hint;
        }
    }
    if sec_idx >= 0 {
        Some([
            1 + fwd_idx.min(sec_idx) as u8,
            1 + fwd_idx.max(sec_idx) as u8,
        ])
    } else {
        None
    }
}

/// uncompressed_header() then byte_alignment(): the start of a FRAME OBU
/// payload (the tile group follows).
pub fn write_frame_header(
    h: &FrameHeader,
    ctx: &HeaderContext<'_>,
    o: &FrameHeaderWriteOptions,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    write_uncompressed_header(&mut w, h, ctx, o);
    w.byte_alignment();
    w.into_bytes()
}

/// uncompressed_header() then trailing_bits(): a FRAME_HEADER OBU payload.
pub fn write_frame_header_obu(
    h: &FrameHeader,
    ctx: &HeaderContext<'_>,
    o: &FrameHeaderWriteOptions,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    write_uncompressed_header(&mut w, h, ctx, o);
    w.trailing_bits();
    w.into_bytes()
}

fn write_delta_q(w: &mut BitWriter, v: i8) {
    w.flag(v != 0);
    if v != 0 {
        w.su(7, i32::from(v));
    }
}

/// Writes frame_size() (the size part) and superres_params().
fn write_frame_size(w: &mut BitWriter, seq: &SequenceHeader, h: &FrameHeader, override_flag: bool) {
    if override_flag {
        w.f(seq.frame_width_bits, h.upscaled_width - 1);
        w.f(seq.frame_height_bits, h.frame_height - 1);
    }
    write_superres(w, seq, h);
}

fn write_superres(w: &mut BitWriter, seq: &SequenceHeader, h: &FrameHeader) {
    if seq.enable_superres {
        w.flag(h.use_superres);
    }
    if seq.enable_superres && h.use_superres {
        w.f(3, h.superres_denom - 9);
    }
}

fn write_render_size(w: &mut BitWriter, h: &FrameHeader, force: bool) {
    let different =
        force || h.render_width != h.upscaled_width || h.render_height != h.frame_height;
    w.flag(different);
    if different {
        w.f(16, h.render_width - 1);
        w.f(16, h.render_height - 1);
    }
}

/// Writes uncompressed_header() (without the final alignment).
pub fn write_uncompressed_header(
    w: &mut BitWriter,
    h: &FrameHeader,
    ctx: &HeaderContext<'_>,
    o: &FrameHeaderWriteOptions,
) {
    let seq = ctx.seq;
    let refs = ctx.refs;
    let id_len = if seq.frame_id_numbers_present {
        seq.additional_frame_id_length + seq.delta_frame_id_length
    } else {
        0
    };
    let reduced = seq.reduced_still_picture_header;
    let frame_type = if reduced { KEY_FRAME } else { h.frame_type };
    let show_frame = reduced || h.show_frame;
    let frame_is_intra = frame_type == KEY_FRAME || frame_type == INTRA_ONLY_FRAME;
    let key_shown = frame_type == KEY_FRAME && show_frame;
    let mut error_resilient = h.error_resilient_mode;
    if !reduced {
        w.flag(h.show_existing_frame);
        if h.show_existing_frame {
            let idx = h.frame_to_show_map_idx & 7;
            w.f(3, u32::from(idx));
            if seq.decoder_model_info_present && !seq.equal_picture_interval {
                w.f(
                    seq.frame_presentation_time_length,
                    o.frame_presentation_time,
                );
            }
            if seq.frame_id_numbers_present {
                w.f(id_len, refs[idx as usize].frame_id);
            }
            return;
        }
        w.f(2, u32::from(frame_type));
        w.flag(show_frame);
        if show_frame && seq.decoder_model_info_present && !seq.equal_picture_interval {
            w.f(
                seq.frame_presentation_time_length,
                o.frame_presentation_time,
            );
        }
        if !show_frame {
            w.flag(h.showable_frame);
        }
        if frame_type == SWITCH_FRAME || key_shown {
            error_resilient = true;
        } else {
            w.flag(h.error_resilient_mode);
        }
    } else {
        error_resilient = false;
    }
    w.flag(h.disable_cdf_update);
    let allow_sct = if seq.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS {
        w.flag(h.allow_screen_content_tools);
        h.allow_screen_content_tools
    } else {
        seq.seq_force_screen_content_tools != 0
    };
    let mut force_integer_mv = false;
    if allow_sct {
        force_integer_mv = if seq.seq_force_integer_mv == SELECT_INTEGER_MV {
            w.flag(h.force_integer_mv);
            h.force_integer_mv
        } else {
            seq.seq_force_integer_mv != 0
        };
    }
    if frame_is_intra {
        force_integer_mv = true;
    }
    if seq.frame_id_numbers_present {
        w.f(id_len, h.current_frame_id);
    }
    let override_flag = if frame_type == SWITCH_FRAME {
        true
    } else if reduced {
        false
    } else {
        w.flag(h.frame_size_override_flag);
        h.frame_size_override_flag
    };
    w.f(seq.order_hint_bits, h.order_hint);
    let primary_ref_frame = if frame_is_intra || error_resilient {
        PRIMARY_REF_NONE
    } else {
        w.f(3, u32::from(h.primary_ref_frame));
        h.primary_ref_frame
    };
    if seq.decoder_model_info_present {
        w.flag(o.buffer_removal_times.is_some());
        if let Some(times) = &o.buffer_removal_times {
            for (k, op) in seq.operating_points.iter().enumerate() {
                if op.decoder_model_present {
                    let idc = u32::from(op.idc);
                    let in_t = (idc >> ctx.temporal_id) & 1;
                    let in_s = (idc >> (u32::from(ctx.spatial_id) + 8)) & 1;
                    if idc == 0 || (in_t != 0 && in_s != 0) {
                        w.f(
                            seq.buffer_removal_time_length,
                            times.get(k).copied().unwrap_or(0),
                        );
                    }
                }
            }
        }
    }
    let refresh = if frame_type == SWITCH_FRAME || key_shown {
        0xff
    } else {
        w.f(8, u32::from(h.refresh_frame_flags));
        h.refresh_frame_flags
    };
    let mut ref_order_hint: [u32; 8] = if key_shown {
        [0; 8]
    } else {
        std::array::from_fn(|i| refs[i].order_hint)
    };
    if (!frame_is_intra || refresh != 0xff) && error_resilient && seq.enable_order_hint {
        for i in 0..8 {
            w.f(seq.order_hint_bits, h.ref_order_hint[i]);
            ref_order_hint[i] = h.ref_order_hint[i];
        }
    }
    let superres_denom = if seq.enable_superres && h.use_superres {
        h.superres_denom
    } else {
        8
    };
    let frame_width = downscaled_width(h.upscaled_width, superres_denom);
    let mi_cols = 2 * ((frame_width + 7) >> 3);
    let mi_rows = 2 * ((h.frame_height + 7) >> 3);
    let mut allow_intrabc = false;
    let mut allow_high_precision_mv = false;
    if frame_is_intra {
        write_frame_size(w, seq, h, override_flag);
        write_render_size(w, h, o.force_render_size);
        if allow_sct && h.upscaled_width == frame_width {
            w.flag(h.allow_intrabc);
            allow_intrabc = h.allow_intrabc;
        }
    } else {
        if seq.enable_order_hint {
            w.flag(false); // frame_refs_short_signaling
        }
        for i in 0..REFS_PER_FRAME {
            let idx = h.ref_frame_idx[i] & 7;
            w.f(3, u32::from(idx));
            if seq.frame_id_numbers_present {
                let m = 1u32 << id_len;
                let delta = (h.current_frame_id + m - refs[idx as usize].frame_id % m) % m;
                w.f(seq.delta_frame_id_length, delta.wrapping_sub(1));
            }
        }
        if override_flag && !error_resilient {
            let matches = |i: usize| {
                let s = &refs[h.ref_frame_idx[i] as usize & 7];
                s.upscaled_width == h.upscaled_width
                    && s.frame_height == h.frame_height
                    && s.render_width == h.render_width
                    && s.render_height == h.render_height
            };
            let found = match o.size_from_ref {
                SizeFromRef::Auto => (0..REFS_PER_FRAME).find(|&i| matches(i)),
                SizeFromRef::Never => None,
                SizeFromRef::Ref(i) => Some(i.min(REFS_PER_FRAME - 1)),
            };
            match found {
                Some(i) => {
                    for _ in 0..i {
                        w.flag(false);
                    }
                    w.flag(true);
                    write_superres(w, seq, h);
                }
                None => {
                    for _ in 0..REFS_PER_FRAME {
                        w.flag(false);
                    }
                    write_frame_size(w, seq, h, override_flag);
                    write_render_size(w, h, o.force_render_size);
                }
            }
        } else {
            write_frame_size(w, seq, h, override_flag);
            write_render_size(w, h, o.force_render_size);
        }
        if !force_integer_mv {
            w.flag(h.allow_high_precision_mv);
            allow_high_precision_mv = h.allow_high_precision_mv;
        }
        w.flag(h.interpolation_filter == SWITCHABLE);
        if h.interpolation_filter != SWITCHABLE {
            w.f(2, u32::from(h.interpolation_filter));
        }
        w.flag(h.is_motion_mode_switchable);
        if !(error_resilient || !seq.enable_ref_frame_mvs) {
            w.flag(h.use_ref_frame_mvs);
        }
    }
    if !(reduced || h.disable_cdf_update) {
        w.flag(h.disable_frame_end_update_cdf);
    }
    let prev: Option<&RefHeaderState> = if primary_ref_frame == PRIMARY_REF_NONE {
        None
    } else {
        let slot =
            h.ref_frame_idx[usize::from(primary_ref_frame).min(REFS_PER_FRAME - 1)] as usize & 7;
        Some(&refs[slot])
    };
    write_tile_info(w, seq, h, mi_cols, mi_rows, o);

    // quantization_params()
    let q = &h.quant;
    w.f(8, u32::from(q.base_q_idx));
    write_delta_q(w, q.delta_q_y_dc);
    if seq.num_planes() > 1 {
        let diff_uv = seq.separate_uv_delta_q
            && (o.force_diff_uv_delta
                || q.delta_q_v_dc != q.delta_q_u_dc
                || q.delta_q_v_ac != q.delta_q_u_ac);
        if seq.separate_uv_delta_q {
            w.flag(diff_uv);
        }
        write_delta_q(w, q.delta_q_u_dc);
        write_delta_q(w, q.delta_q_u_ac);
        if diff_uv {
            write_delta_q(w, q.delta_q_v_dc);
            write_delta_q(w, q.delta_q_v_ac);
        }
    }
    w.flag(q.using_qmatrix);
    if q.using_qmatrix {
        w.f(4, u32::from(q.qm_y));
        w.f(4, u32::from(q.qm_u));
        if seq.separate_uv_delta_q {
            w.f(4, u32::from(q.qm_v));
        }
    }

    // segmentation_params()
    let s = &h.segmentation;
    w.flag(s.enabled);
    if s.enabled {
        let update_data = if primary_ref_frame == PRIMARY_REF_NONE {
            true
        } else {
            w.flag(s.update_map);
            if s.update_map {
                w.flag(s.temporal_update);
            }
            w.flag(s.update_data);
            s.update_data
        };
        if update_data {
            for i in 0..MAX_SEGMENTS {
                for j in 0..SEG_LVL_MAX {
                    let enabled = s.feature_enabled[i][j];
                    w.flag(enabled);
                    if enabled {
                        let bits = SEG_FEATURE_BITS[j];
                        let v = i32::from(s.feature_data[i][j]);
                        if SEG_FEATURE_SIGNED[j] {
                            w.su(1 + bits, v);
                        } else {
                            w.f(bits, v.max(0) as u32);
                        }
                    }
                }
            }
        }
    }

    // delta_q_params() / delta_lf_params()
    let mut delta_q_present = false;
    if q.base_q_idx > 0 {
        w.flag(h.delta_q_present);
        delta_q_present = h.delta_q_present;
    }
    if delta_q_present {
        w.f(2, u32::from(h.delta_q_res));
        let mut delta_lf_present = false;
        if !allow_intrabc {
            w.flag(h.delta_lf_present);
            delta_lf_present = h.delta_lf_present;
        }
        if delta_lf_present {
            w.f(2, u32::from(h.delta_lf_res));
            w.flag(h.delta_lf_multi);
        }
    }

    let mut coded_lossless = true;
    for seg in 0..MAX_SEGMENTS {
        let lossless = h.qindex(seg) == 0
            && q.delta_q_y_dc == 0
            && q.delta_q_u_ac == 0
            && q.delta_q_u_dc == 0
            && q.delta_q_v_ac == 0
            && q.delta_q_v_dc == 0;
        if !lossless {
            coded_lossless = false;
        }
    }
    let all_lossless = coded_lossless && frame_width == h.upscaled_width;

    // loop_filter_params()
    if !(coded_lossless || allow_intrabc) {
        let lf = &h.loop_filter;
        w.f(6, u32::from(lf.level[0]));
        w.f(6, u32::from(lf.level[1]));
        if seq.num_planes() > 1 && (lf.level[0] != 0 || lf.level[1] != 0) {
            w.f(6, u32::from(lf.level[2]));
            w.f(6, u32::from(lf.level[3]));
        }
        w.f(3, u32::from(lf.sharpness));
        w.flag(lf.delta_enabled);
        if lf.delta_enabled {
            w.flag(lf.delta_update);
            if lf.delta_update {
                let prev_ref = prev.map_or(DEFAULT_LF_REF_DELTAS, |p| p.loop_filter_ref_deltas);
                let prev_mode = prev.map_or([0; 2], |p| p.loop_filter_mode_deltas);
                for i in 0..8 {
                    let upd = o.lf_delta_update_all || lf.ref_deltas[i] != prev_ref[i];
                    w.flag(upd);
                    if upd {
                        w.su(7, i32::from(lf.ref_deltas[i]));
                    }
                }
                for i in 0..2 {
                    let upd = o.lf_delta_update_all || lf.mode_deltas[i] != prev_mode[i];
                    w.flag(upd);
                    if upd {
                        w.su(7, i32::from(lf.mode_deltas[i]));
                    }
                }
            }
        }
    }

    // cdef_params()
    if !(coded_lossless || allow_intrabc || !seq.enable_cdef) {
        let c = &h.cdef;
        let sec = |v: u8| u32::from(if v >= 3 { 3 } else { v });
        w.f(2, u32::from(c.damping.saturating_sub(3)));
        w.f(2, u32::from(c.bits));
        for i in 0..(1usize << (c.bits & 3)) {
            w.f(4, u32::from(c.y_pri[i]));
            w.f(2, sec(c.y_sec[i]));
            if seq.num_planes() > 1 {
                w.f(4, u32::from(c.uv_pri[i]));
                w.f(2, sec(c.uv_sec[i]));
            }
        }
    }

    // lr_params()
    if !(all_lossless || allow_intrabc || !seq.enable_restoration) {
        let lr = &h.lr;
        let mut uses_lr = false;
        let mut uses_chroma = false;
        for i in 0..seq.num_planes() {
            let t = lr.frame_type[i];
            let code = match t {
                RESTORE_SWITCHABLE => 1,
                RESTORE_WIENER => 2,
                RESTORE_SGRPROJ => 3,
                _ => 0,
            };
            w.f(2, code);
            if t != RESTORE_NONE {
                uses_lr = true;
                if i > 0 {
                    uses_chroma = true;
                }
            }
        }
        if uses_lr {
            let shift = (31 - lr.unit_size[0].max(64).leading_zeros()).saturating_sub(6);
            if seq.use_128x128_superblock {
                w.f(1, shift.saturating_sub(1));
            } else {
                w.flag(shift != 0);
                if shift != 0 {
                    w.f(1, shift - 1);
                }
            }
            if seq.subsampling_x && seq.subsampling_y && uses_chroma {
                w.flag(lr.unit_size[1] < lr.unit_size[0]);
            }
        }
    }

    // read_tx_mode()
    if !coded_lossless {
        w.flag(h.tx_mode == TX_MODE_SELECT);
    }
    // frame_reference_mode()
    let mut reference_select = false;
    if !frame_is_intra {
        w.flag(h.reference_select);
        reference_select = h.reference_select;
    }
    // skip_mode_params()
    let hs = FrameHeader {
        frame_type,
        reference_select,
        ref_frame_idx: h.ref_frame_idx,
        order_hint: h.order_hint,
        ..FrameHeader::default()
    };
    if skip_mode_frames(seq, &hs, &ref_order_hint).is_some() {
        w.flag(h.skip_mode_present);
    }
    if !(frame_is_intra || error_resilient || !seq.enable_warped_motion) {
        w.flag(h.allow_warped_motion);
    }
    w.flag(h.reduced_tx_set);

    // global_motion_params()
    if !frame_is_intra {
        let prev_gm = prev.map_or_else(default_gm_params, |p| p.gm_params);
        for rf in 1..8 {
            let ty = h.gm_type[rf];
            w.flag(ty != IDENTITY);
            if ty != IDENTITY {
                w.flag(ty == ROTZOOM);
                if ty != ROTZOOM {
                    w.flag(ty == TRANSLATION);
                }
            }
            let put = |w: &mut BitWriter, idx: usize| {
                let mut abs_bits = 12u32;
                let mut prec_bits = 15u32;
                if idx < 2 {
                    if ty == TRANSLATION {
                        let hp = u32::from(!allow_high_precision_mv);
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
                let r = (prev_gm[rf][idx] >> prec_diff) - sub;
                let x = (h.gm_params[rf][idx] - round) >> prec_diff;
                encode_signed_subexp_with_ref(w, -mx, mx + 1, r, x);
            };
            if ty >= ROTZOOM {
                put(w, 2);
                put(w, 3);
                if ty == AFFINE {
                    put(w, 4);
                    put(w, 5);
                }
            }
            if ty >= TRANSLATION {
                put(w, 0);
                put(w, 1);
            }
        }
    }

    // film_grain_params()
    let showable = if show_frame {
        frame_type != KEY_FRAME
    } else {
        h.showable_frame
    };
    if seq.film_grain_params_present && (show_frame || showable) {
        write_film_grain(w, seq, h, frame_type, refs, o);
    }
}

fn write_tile_info(
    w: &mut BitWriter,
    seq: &SequenceHeader,
    h: &FrameHeader,
    mi_cols: u32,
    mi_rows: u32,
    o: &FrameHeaderWriteOptions,
) {
    let t = &h.tile_info;
    let l = tile_limits(seq, mi_cols, mi_rows);
    let cols_ok = t.cols_log2 >= l.min_log2_tile_cols && t.cols_log2 <= l.max_log2_tile_cols;
    let min_rows = l.min_log2_tile_rows(t.cols_log2);
    let rows_ok = t.rows_log2 >= min_rows && t.rows_log2 <= l.max_log2_tile_rows;
    let empty = t.mi_col_starts.len() < 2 || t.mi_row_starts.len() < 2;
    let uniform = o.uniform_tile_spacing.unwrap_or_else(|| {
        empty
            || (cols_ok
                && rows_ok
                && t.mi_col_starts == uniform_starts(l.sb_cols, t.cols_log2, l.sb_shift, mi_cols)
                && t.mi_row_starts == uniform_starts(l.sb_rows, t.rows_log2, l.sb_shift, mi_rows))
    });
    w.flag(uniform);
    let (cols_log2, rows_log2) = if uniform {
        let cols_log2 = t.cols_log2.clamp(
            l.min_log2_tile_cols,
            l.max_log2_tile_cols.max(l.min_log2_tile_cols),
        );
        for _ in l.min_log2_tile_cols..cols_log2 {
            w.flag(true);
        }
        if cols_log2 < l.max_log2_tile_cols {
            w.flag(false);
        }
        let min_rows = l.min_log2_tile_rows(cols_log2);
        let rows_log2 = t
            .rows_log2
            .clamp(min_rows, l.max_log2_tile_rows.max(min_rows));
        for _ in min_rows..rows_log2 {
            w.flag(true);
        }
        if rows_log2 < l.max_log2_tile_rows {
            w.flag(false);
        }
        (cols_log2, rows_log2)
    } else {
        let sizes = |starts: &[u32]| -> Vec<u32> {
            starts
                .windows(2)
                .map(|p| {
                    let a = p[0] >> l.sb_shift;
                    let b = (p[1] + (1 << l.sb_shift) - 1) >> l.sb_shift;
                    b.saturating_sub(a).max(1)
                })
                .collect()
        };
        let widths = sizes(&t.mi_col_starts);
        let heights = sizes(&t.mi_row_starts);
        let mut start = 0;
        let mut widest = 0;
        for &wd in &widths {
            let max_w = (l.sb_cols - start.min(l.sb_cols)).min(l.max_tile_width_sb);
            w.ns(max_w, wd - 1);
            widest = widest.max(wd);
            start += wd;
        }
        let max_h_sb = l.max_tile_height_sb(widest);
        let mut start = 0;
        for &ht in &heights {
            let max_h = (l.sb_rows - start.min(l.sb_rows)).min(max_h_sb);
            w.ns(max_h, ht - 1);
            start += ht;
        }
        (
            tile_log2(1, widths.len() as u32),
            tile_log2(1, heights.len() as u32),
        )
    };
    if cols_log2 > 0 || rows_log2 > 0 {
        w.f(rows_log2 + cols_log2, t.context_update_tile_id);
        w.f(2, t.tile_size_bytes.clamp(1, 4) - 1);
    }
}

fn write_film_grain(
    w: &mut BitWriter,
    seq: &SequenceHeader,
    h: &FrameHeader,
    frame_type: u8,
    refs: &[RefHeaderState; 8],
    o: &FrameHeaderWriteOptions,
) {
    let g = &h.grain;
    w.flag(g.apply_grain);
    if !g.apply_grain {
        return;
    }
    w.f(16, u32::from(g.grain_seed));
    let update = if frame_type == INTER_FRAME {
        w.flag(g.update_grain);
        g.update_grain
    } else {
        true
    };
    if !update {
        let norm = |p: &GrainParams| GrainParams {
            grain_seed: 0,
            apply_grain: true,
            update_grain: false,
            ..p.clone()
        };
        let want = norm(g);
        let idx = o.film_grain_params_ref_idx.unwrap_or_else(|| {
            h.ref_frame_idx
                .iter()
                .copied()
                .find(|&i| norm(&refs[i as usize & 7].grain) == want)
                .unwrap_or(h.ref_frame_idx[0])
        });
        w.f(3, u32::from(idx));
        return;
    }
    w.f(4, u32::from(g.num_y_points));
    for i in 0..usize::from(g.num_y_points).min(16) {
        w.f(8, u32::from(g.point_y_value[i]));
        w.f(8, u32::from(g.point_y_scaling[i]));
    }
    if !seq.mono_chrome {
        w.flag(g.chroma_scaling_from_luma);
    }
    let csfl = !seq.mono_chrome && g.chroma_scaling_from_luma;
    let (num_cb, num_cr) = if seq.mono_chrome
        || csfl
        || (seq.subsampling_x && seq.subsampling_y && g.num_y_points == 0)
    {
        (0, 0)
    } else {
        w.f(4, u32::from(g.num_cb_points));
        for i in 0..usize::from(g.num_cb_points).min(16) {
            w.f(8, u32::from(g.point_cb_value[i]));
            w.f(8, u32::from(g.point_cb_scaling[i]));
        }
        w.f(4, u32::from(g.num_cr_points));
        for i in 0..usize::from(g.num_cr_points).min(16) {
            w.f(8, u32::from(g.point_cr_value[i]));
            w.f(8, u32::from(g.point_cr_scaling[i]));
        }
        (g.num_cb_points, g.num_cr_points)
    };
    w.f(2, u32::from(g.grain_scaling_minus_8));
    w.f(2, u32::from(g.ar_coeff_lag));
    let lag = usize::from(g.ar_coeff_lag & 3);
    let num_pos_luma = 2 * lag * (lag + 1);
    let num_pos_chroma = if g.num_y_points != 0 {
        for i in 0..num_pos_luma {
            w.f(8, u32::from(g.ar_coeffs_y_plus_128[i]));
        }
        num_pos_luma + 1
    } else {
        num_pos_luma
    };
    if csfl || num_cb != 0 {
        for i in 0..num_pos_chroma {
            w.f(8, u32::from(g.ar_coeffs_cb_plus_128[i]));
        }
    }
    if csfl || num_cr != 0 {
        for i in 0..num_pos_chroma {
            w.f(8, u32::from(g.ar_coeffs_cr_plus_128[i]));
        }
    }
    w.f(2, u32::from(g.ar_coeff_shift_minus_6));
    w.f(2, u32::from(g.grain_scale_shift));
    if num_cb != 0 {
        w.f(8, u32::from(g.cb_mult));
        w.f(8, u32::from(g.cb_luma_mult));
        w.f(9, u32::from(g.cb_offset));
    }
    if num_cr != 0 {
        w.f(8, u32::from(g.cr_mult));
        w.f(8, u32::from(g.cr_luma_mult));
        w.f(9, u32::from(g.cr_offset));
    }
    w.flag(g.overlap_flag);
    w.flag(g.clip_to_restricted_range);
}

/// The inverse of decode_subexp( numSyms ).
fn encode_subexp(w: &mut BitWriter, v: u32, num_syms: u32) {
    let mut i = 0u32;
    let mut mk = 0u32;
    let k = 3u32;
    loop {
        let b2 = if i != 0 { k + i - 1 } else { k };
        let a = 1u32 << b2;
        if num_syms <= mk + 3 * a {
            w.ns(num_syms - mk, v - mk);
            return;
        }
        let more = v >= mk + a;
        w.flag(more);
        if more {
            i += 1;
            mk += a;
        } else {
            w.f(b2, v - mk);
            return;
        }
    }
}

/// The inverse of inverse_recenter( r, v ).
fn recenter(r: i32, v: i32) -> i32 {
    if v > 2 * r {
        v
    } else if v >= r {
        (v - r) << 1
    } else {
        ((r - v) << 1) - 1
    }
}

fn encode_unsigned_subexp_with_ref(w: &mut BitWriter, mx: i32, r: i32, x: i32) {
    let v = if (r << 1) <= mx {
        recenter(r, x)
    } else {
        recenter(mx - 1 - r, mx - 1 - x)
    };
    encode_subexp(w, v.max(0) as u32, mx as u32);
}

fn encode_signed_subexp_with_ref(w: &mut BitWriter, low: i32, high: i32, r: i32, x: i32) {
    encode_unsigned_subexp_with_ref(w, high - low, r - low, x - low);
}

/// tile_group_obu() header: tile_start_and_end_present_flag (when the group
/// is not the whole frame), tg_start, tg_end, byte_alignment().
pub fn write_tile_group_header(w: &mut BitWriter, ti: &TileInfo, tg_start: usize, tg_end: usize) {
    let num_tiles = ti.cols * ti.rows;
    if num_tiles > 1 {
        let present = !(tg_start == 0 && tg_end + 1 == num_tiles);
        w.flag(present);
        if present {
            let bits = ti.cols_log2 + ti.rows_log2;
            w.f(bits, tg_start as u32);
            w.f(bits, tg_end as u32);
        }
    }
    w.byte_alignment();
}

/// A tile group OBU payload for tiles `tg_start..=tg_end` (`tiles` holds
/// their data in order): the header, then each tile but the last preceded by
/// tile_size_minus_1 in `ti.tile_size_bytes` little-endian bytes.
pub fn write_tile_group<T: AsRef<[u8]>>(
    ti: &TileInfo,
    tg_start: usize,
    tg_end: usize,
    tiles: &[T],
) -> Vec<u8> {
    let mut w = BitWriter::new();
    write_tile_group_header(&mut w, ti, tg_start, tg_end);
    let n = tiles.len();
    for (i, t) in tiles.iter().enumerate() {
        let t = t.as_ref();
        if i + 1 < n {
            w.le(ti.tile_size_bytes, t.len().saturating_sub(1) as u32);
        }
        w.bytes_raw(t);
    }
    w.into_bytes()
}

/// A temporal delimiter OBU (no extension).
pub fn write_temporal_delimiter(out: &mut Vec<u8>) {
    write_obu(out, OBU_TEMPORAL_DELIMITER, 0, 0, false, &[]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec::obu::{
        Cdef, LoopFilter, OBU_FRAME, OperatingPoint, Quant, Restoration, SEG_LVL_REF_FRAME,
        Segmentation, parse_frame_header, parse_sequence_header, split_obus,
    };

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n.max(1)
        }
        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            lo + self.below((hi - lo + 1) as u64) as i64
        }
        fn chance(&mut self, num: u64, den: u64) -> bool {
            self.below(den) < num
        }
        fn flag(&mut self) -> bool {
            self.below(2) == 1
        }
    }

    fn random_seq(rng: &mut Rng) -> SequenceHeader {
        let mut s = SequenceHeader {
            profile: rng.below(3) as u8,
            ..SequenceHeader::default()
        };
        s.reduced_still_picture_header = rng.chance(1, 8);
        s.still_picture = s.reduced_still_picture_header || rng.chance(1, 8);
        if s.reduced_still_picture_header {
            s.operating_points.push(OperatingPoint {
                seq_level_idx: rng.below(32) as u8,
                ..OperatingPoint::default()
            });
            s.seq_force_screen_content_tools = 2;
            s.seq_force_integer_mv = 2;
        } else {
            s.timing_info_present = rng.flag();
            if s.timing_info_present {
                s.equal_picture_interval = rng.flag();
                s.decoder_model_info_present = rng.flag();
                if s.decoder_model_info_present {
                    s.buffer_removal_time_length = 1 + rng.below(32) as u32;
                    s.frame_presentation_time_length = 1 + rng.below(32) as u32;
                }
            }
            let many = rng.flag();
            let n = 1 + rng.below(if many { 32 } else { 1 }) as usize;
            for _ in 0..n {
                let level = rng.below(32) as u8;
                let idc = if rng.chance(1, 3) {
                    0
                } else {
                    rng.below(4096) as u16
                };
                let tier = if level > 7 { rng.below(2) as u8 } else { 0 };
                let dm = s.decoder_model_info_present && rng.flag();
                let idd = rng.chance(1, 3);
                s.operating_points.push(OperatingPoint {
                    idc,
                    seq_level_idx: level,
                    seq_tier: tier,
                    decoder_model_present: dm,
                    initial_display_delay_present: idd,
                });
            }
            s.frame_id_numbers_present = rng.flag();
            if s.frame_id_numbers_present {
                s.delta_frame_id_length = 2 + rng.below(16) as u32;
                s.additional_frame_id_length = 1 + rng.below(8) as u32;
            }
            s.enable_interintra_compound = rng.flag();
            s.enable_masked_compound = rng.flag();
            s.enable_warped_motion = rng.flag();
            s.enable_dual_filter = rng.flag();
            s.enable_order_hint = rng.chance(3, 4);
            if s.enable_order_hint {
                s.enable_jnt_comp = rng.flag();
                s.enable_ref_frame_mvs = rng.flag();
                s.order_hint_bits = 1 + rng.below(8) as u32;
            }
            s.seq_force_screen_content_tools = rng.below(3) as u8;
            s.seq_force_integer_mv = if s.seq_force_screen_content_tools > 0 {
                rng.below(3) as u8
            } else {
                2
            };
        }
        s.operating_point_idc = s.operating_points[0].idc;
        // Mostly modest sizes; sometimes up to the 16-bit limit.
        let big = rng.chance(1, 5);
        s.frame_width_bits = 1 + rng.below(if big { 16 } else { 12 }) as u32;
        s.frame_height_bits = 1 + rng.below(if big { 16 } else { 12 }) as u32;
        s.max_frame_width = 1 + rng.below(1 << s.frame_width_bits) as u32;
        s.max_frame_height = 1 + rng.below(1 << s.frame_height_bits) as u32;
        s.use_128x128_superblock = rng.flag();
        s.enable_filter_intra = rng.flag();
        s.enable_intra_edge_filter = rng.flag();
        s.enable_superres = rng.flag();
        s.enable_cdef = rng.flag();
        s.enable_restoration = rng.flag();
        s.bit_depth = match (s.profile, rng.below(3)) {
            (2, 2) => 12,
            (_, 0) => 8,
            _ => 10,
        };
        s.mono_chrome = s.profile != 1 && rng.chance(1, 4);
        let cd = rng.below(4);
        let (cp, tc, mc) = match cd {
            0 => (2, 2, 2),
            1 => (1, 13, 0),
            _ => (
                rng.below(256) as u8,
                rng.below(256) as u8,
                rng.below(256) as u8,
            ),
        };
        s.color_primaries = cp;
        s.transfer_characteristics = tc;
        s.matrix_coefficients = mc;
        let srgb = cp == 1 && tc == 13 && mc == 0;
        if s.mono_chrome {
            s.color_range = rng.flag();
            s.subsampling_x = true;
            s.subsampling_y = true;
        } else if srgb {
            s.color_range = true;
            s.separate_uv_delta_q = rng.flag();
        } else {
            s.color_range = rng.flag();
            match s.profile {
                0 => {
                    s.subsampling_x = true;
                    s.subsampling_y = true;
                }
                1 => {}
                _ => {
                    if s.bit_depth == 12 {
                        s.subsampling_x = rng.flag();
                        s.subsampling_y = s.subsampling_x && rng.flag();
                    } else {
                        s.subsampling_x = true;
                    }
                }
            }
            if s.subsampling_x && s.subsampling_y {
                s.chroma_sample_position = rng.below(4) as u8;
            }
            s.separate_uv_delta_q = rng.flag();
        }
        s.film_grain_params_present = rng.flag();
        s
    }

    #[test]
    fn sequence_header_round_trip() {
        let mut rng = Rng(0x5eed_0001);
        for _ in 0..5000 {
            let s = random_seq(&mut rng);
            let n = s.operating_points.len();
            let o = SequenceHeaderWriteOptions {
                num_units_in_display_tick: rng.next() as u32,
                time_scale: rng.next() as u32,
                num_ticks_per_picture_minus_1: (rng.next() as u32) >> rng.below(32),
                buffer_delay_length_minus_1: rng.below(32) as u8,
                num_units_in_decoding_tick: rng.next() as u32,
                decoder_buffer_delay: (0..n).map(|_| rng.below(1 << 10) as u32).collect(),
                encoder_buffer_delay: (0..n).map(|_| rng.below(1 << 10) as u32).collect(),
                low_delay_mode_flag: (0..n).map(|_| rng.flag()).collect(),
                initial_display_delay_minus_1: (0..n).map(|_| rng.below(16) as u8).collect(),
                force_initial_display_delay_present: rng.flag(),
                force_color_description: rng.flag(),
            };
            let bytes = write_sequence_header_with(&s, &o);
            assert_eq!(parse_sequence_header(&bytes).unwrap(), s);
            let bytes = write_sequence_header(&s);
            assert_eq!(parse_sequence_header(&bytes).unwrap(), s);
        }
    }

    type Gm = ([u8; 8], [[i32; 6]; 8]);

    fn random_gm(rng: &mut Rng, hp: bool) -> Gm {
        let mut ty = [IDENTITY; 8];
        let mut p = default_gm_params();
        for rf in 1..8 {
            let t = rng.below(4) as u8;
            ty[rf] = t;
            let val = |rng: &mut Rng, idx: usize| -> i32 {
                let (abs_bits, prec_bits) = if idx < 2 {
                    if t == TRANSLATION {
                        let h = u32::from(!hp);
                        (9 - h, 3 - h)
                    } else {
                        (12, 6)
                    }
                } else {
                    (12, 15)
                };
                let prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
                let round = if idx % 3 == 2 { 1 << 16 } else { 0 };
                let mx = 1i64 << abs_bits;
                let x = match rng.below(4) {
                    0 => -mx,
                    1 => mx,
                    _ => rng.range(-mx, mx),
                };
                ((x as i32) << prec_diff) + round
            };
            if t >= ROTZOOM {
                p[rf][2] = val(rng, 2);
                p[rf][3] = val(rng, 3);
                if t == AFFINE {
                    p[rf][4] = val(rng, 4);
                    p[rf][5] = val(rng, 5);
                } else {
                    p[rf][4] = -p[rf][3];
                    p[rf][5] = p[rf][2];
                }
            }
            if t >= TRANSLATION {
                p[rf][0] = val(rng, 0);
                p[rf][1] = val(rng, 1);
            }
        }
        (ty, p)
    }

    fn random_grain(rng: &mut Rng, seq: &SequenceHeader) -> GrainParams {
        let mut g = GrainParams {
            apply_grain: true,
            update_grain: true,
            grain_seed: rng.below(65536) as u16,
            num_y_points: rng.below(15) as u8,
            ..GrainParams::default()
        };
        for i in 0..g.num_y_points as usize {
            g.point_y_value[i] = rng.below(256) as u8;
            g.point_y_scaling[i] = rng.below(256) as u8;
        }
        g.chroma_scaling_from_luma = !seq.mono_chrome && rng.chance(1, 4);
        if !(seq.mono_chrome
            || g.chroma_scaling_from_luma
            || (seq.subsampling_x && seq.subsampling_y && g.num_y_points == 0))
        {
            g.num_cb_points = rng.below(11) as u8;
            g.num_cr_points = rng.below(11) as u8;
            for i in 0..g.num_cb_points as usize {
                g.point_cb_value[i] = rng.below(256) as u8;
                g.point_cb_scaling[i] = rng.below(256) as u8;
            }
            for i in 0..g.num_cr_points as usize {
                g.point_cr_value[i] = rng.below(256) as u8;
                g.point_cr_scaling[i] = rng.below(256) as u8;
            }
        }
        g.grain_scaling_minus_8 = rng.below(4) as u8;
        g.ar_coeff_lag = rng.below(4) as u8;
        let lag = g.ar_coeff_lag as usize;
        let npl = 2 * lag * (lag + 1);
        let npc = if g.num_y_points != 0 {
            for i in 0..npl {
                g.ar_coeffs_y_plus_128[i] = rng.below(256) as u8;
            }
            npl + 1
        } else {
            npl
        };
        if g.chroma_scaling_from_luma || g.num_cb_points != 0 {
            for i in 0..npc {
                g.ar_coeffs_cb_plus_128[i] = rng.below(256) as u8;
            }
        }
        if g.chroma_scaling_from_luma || g.num_cr_points != 0 {
            for i in 0..npc {
                g.ar_coeffs_cr_plus_128[i] = rng.below(256) as u8;
            }
        }
        g.ar_coeff_shift_minus_6 = rng.below(4) as u8;
        g.grain_scale_shift = rng.below(4) as u8;
        if g.num_cb_points != 0 {
            g.cb_mult = rng.below(256) as u8;
            g.cb_luma_mult = rng.below(256) as u8;
            g.cb_offset = rng.below(512) as u16;
        }
        if g.num_cr_points != 0 {
            g.cr_mult = rng.below(256) as u8;
            g.cr_luma_mult = rng.below(256) as u8;
            g.cr_offset = rng.below(512) as u16;
        }
        g.overlap_flag = rng.flag();
        g.clip_to_restricted_range = rng.flag();
        g
    }

    fn random_feature(rng: &mut Rng, j: usize) -> i16 {
        let max = [255, 63, 63, 63, 63, 7, 0, 0][j];
        if SEG_FEATURE_SIGNED[j] {
            rng.range(-max, max) as i16
        } else {
            rng.range(0, max) as i16
        }
    }

    fn random_ref(rng: &mut Rng, seq: &SequenceHeader, cur_id: u32) -> RefHeaderState {
        let id_len = seq.additional_frame_id_length + seq.delta_frame_id_length;
        let up = 1 + rng.below(u64::from(seq.max_frame_width)) as u32;
        let fh = 1 + rng.below(u64::from(seq.max_frame_height)) as u32;
        let hp = rng.flag();
        let (_, gm) = random_gm(rng, hp);
        let mut fe = [[false; 8]; 8];
        let mut fd = [[0i16; 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                if rng.chance(1, 4) {
                    fe[i][j] = true;
                    fd[i][j] = random_feature(rng, j);
                }
            }
        }
        let mut grain = random_grain(rng, seq);
        grain.update_grain = rng.flag();
        let frame_id = if seq.frame_id_numbers_present {
            let k = 1 + rng.below((1u64 << seq.delta_frame_id_length) - 1) as u32;
            (cur_id + (1 << id_len) - k) % (1 << id_len)
        } else {
            0
        };
        let render_width = if rng.flag() {
            up
        } else {
            1 + rng.below(65536) as u32
        };
        let render_height = if rng.flag() {
            fh
        } else {
            1 + rng.below(65536) as u32
        };
        let order_hint = if seq.enable_order_hint {
            rng.below(1 << seq.order_hint_bits) as u32
        } else {
            0
        };
        RefHeaderState {
            valid: true,
            frame_id,
            upscaled_width: up,
            frame_width: up,
            frame_height: fh,
            render_width,
            render_height,
            mi_cols: 0,
            mi_rows: 0,
            frame_type: rng.below(4) as u8,
            order_hint,
            saved_order_hints: [0; 8],
            gm_params: gm,
            loop_filter_ref_deltas: std::array::from_fn(|_| rng.range(-63, 63) as i8),
            loop_filter_mode_deltas: std::array::from_fn(|_| rng.range(-63, 63) as i8),
            feature_enabled: fe,
            feature_data: fd,
            grain,
            bit_depth: seq.bit_depth,
            subsampling_x: seq.subsampling_x,
            subsampling_y: seq.subsampling_y,
            showable: rng.flag(),
        }
    }

    /// Builds a random header whose derived fields are what the parser
    /// derives, so parse(write(h)) must equal h.
    fn random_frame(
        rng: &mut Rng,
        seq: &SequenceHeader,
        refs: &[RefHeaderState; 8],
        cur_id: u32,
        tid: u8,
        sid: u8,
    ) -> (FrameHeader, FrameHeaderWriteOptions) {
        let mut o = FrameHeaderWriteOptions::default();
        let mut h = FrameHeader {
            temporal_id: tid,
            spatial_id: sid,
            ..FrameHeader::default()
        };
        if seq.decoder_model_info_present {
            o.frame_presentation_time = rng.below(1 << seq.frame_presentation_time_length) as u32;
            if rng.flag() {
                let len = seq.buffer_removal_time_length;
                o.buffer_removal_times = Some(
                    (0..seq.operating_points.len())
                        .map(|_| rng.below(1 << len) as u32)
                        .collect(),
                );
            }
        }
        let reduced = seq.reduced_still_picture_header;
        if !reduced && rng.chance(1, 12) {
            h.show_existing_frame = true;
            h.frame_to_show_map_idx = rng.below(8) as u8;
            let slot = &refs[h.frame_to_show_map_idx as usize];
            h.frame_type = slot.frame_type;
            if h.frame_type == KEY_FRAME {
                h.refresh_frame_flags = 0xff;
            }
            if seq.film_grain_params_present {
                h.grain = slot.grain.clone();
            }
            h.show_frame = true;
            return (h, o);
        }
        if reduced {
            h.frame_type = KEY_FRAME;
            h.show_frame = true;
        } else {
            h.frame_type = rng.below(4) as u8;
            h.show_frame = rng.flag();
            h.showable_frame = if h.show_frame {
                h.frame_type != KEY_FRAME
            } else {
                rng.flag()
            };
        }
        h.frame_is_intra = h.frame_type == KEY_FRAME || h.frame_type == INTRA_ONLY_FRAME;
        let key_shown = h.frame_type == KEY_FRAME && h.show_frame;
        h.error_resilient_mode = if reduced {
            false
        } else if h.frame_type == SWITCH_FRAME || key_shown {
            true
        } else {
            rng.flag()
        };
        h.disable_cdf_update = rng.flag();
        h.allow_screen_content_tools = if seq.seq_force_screen_content_tools == 2 {
            rng.flag()
        } else {
            seq.seq_force_screen_content_tools != 0
        };
        if h.allow_screen_content_tools {
            h.force_integer_mv = if seq.seq_force_integer_mv == 2 {
                rng.flag()
            } else {
                seq.seq_force_integer_mv != 0
            };
        }
        if h.frame_is_intra {
            h.force_integer_mv = true;
        }
        if seq.frame_id_numbers_present {
            h.current_frame_id = cur_id;
        }
        h.frame_size_override_flag = if h.frame_type == SWITCH_FRAME {
            true
        } else if reduced {
            false
        } else {
            rng.flag()
        };
        if seq.enable_order_hint {
            h.order_hint = rng.below(1 << seq.order_hint_bits) as u32;
        }
        h.primary_ref_frame = if h.frame_is_intra || h.error_resilient_mode {
            PRIMARY_REF_NONE
        } else {
            rng.below(7) as u8
        };
        h.refresh_frame_flags = if h.frame_type == SWITCH_FRAME || key_shown {
            0xff
        } else {
            rng.below(256) as u8
        };
        if !h.frame_is_intra {
            for i in 0..7 {
                h.ref_frame_idx[i] = rng.below(8) as u8;
            }
        }
        let mut roh: [u32; 8] = if key_shown {
            [0; 8]
        } else {
            std::array::from_fn(|i| refs[i].order_hint)
        };
        if (!h.frame_is_intra || h.refresh_frame_flags != 0xff)
            && h.error_resilient_mode
            && seq.enable_order_hint
        {
            for i in 0..8 {
                let used = !h.frame_is_intra && h.ref_frame_idx.contains(&(i as u8));
                if !used && rng.flag() {
                    roh[i] = rng.below(1 << seq.order_hint_bits) as u32;
                }
            }
        }
        h.ref_order_hint = roh;

        // Frame size.
        let with_refs = !h.frame_is_intra && h.frame_size_override_flag && !h.error_resilient_mode;
        if with_refs && rng.flag() {
            let i = rng.below(7) as usize;
            let s = &refs[h.ref_frame_idx[i] as usize];
            h.upscaled_width = s.upscaled_width;
            h.frame_height = s.frame_height;
            h.render_width = s.render_width;
            h.render_height = s.render_height;
            o.size_from_ref = if rng.flag() {
                SizeFromRef::Ref(i)
            } else {
                SizeFromRef::Auto
            };
        } else {
            if with_refs {
                o.size_from_ref = SizeFromRef::Never;
            }
            if h.frame_size_override_flag {
                h.upscaled_width = 1 + rng.below(1 << seq.frame_width_bits) as u32;
                h.frame_height = 1 + rng.below(1 << seq.frame_height_bits) as u32;
            } else {
                h.upscaled_width = seq.max_frame_width;
                h.frame_height = seq.max_frame_height;
            }
            if rng.flag() {
                h.render_width = h.upscaled_width;
                h.render_height = h.frame_height;
                o.force_render_size = rng.chance(1, 4);
            } else {
                h.render_width = 1 + rng.below(65536) as u32;
                h.render_height = 1 + rng.below(65536) as u32;
            }
        }
        h.use_superres = seq.enable_superres && rng.flag();
        h.superres_denom = if h.use_superres {
            9 + rng.below(8) as u32
        } else {
            8
        };
        h.frame_width = downscaled_width(h.upscaled_width, h.superres_denom);
        h.mi_cols = 2 * ((h.frame_width + 7) >> 3);
        h.mi_rows = 2 * ((h.frame_height + 7) >> 3);
        if h.frame_is_intra {
            if h.allow_screen_content_tools && h.upscaled_width == h.frame_width {
                h.allow_intrabc = rng.flag();
            }
        } else {
            h.allow_high_precision_mv = !h.force_integer_mv && rng.flag();
            h.interpolation_filter = rng.below(5) as u8;
            h.is_motion_mode_switchable = rng.flag();
            h.use_ref_frame_mvs =
                !(h.error_resilient_mode || !seq.enable_ref_frame_mvs) && rng.flag();
            for i in 0..7 {
                let hint = roh[h.ref_frame_idx[i] as usize];
                h.order_hints[i + 1] = hint;
                h.ref_frame_sign_bias[i + 1] =
                    seq.enable_order_hint && relative_dist(seq, hint, h.order_hint) > 0;
            }
        }
        h.disable_frame_end_update_cdf = reduced || h.disable_cdf_update || rng.flag();
        let prev = if h.primary_ref_frame == PRIMARY_REF_NONE {
            None
        } else {
            Some(&refs[h.ref_frame_idx[h.primary_ref_frame as usize] as usize])
        };

        // Tiles.
        let l = tile_limits(seq, h.mi_cols, h.mi_rows);
        let mut ti = if rng.flag() {
            let max_c = l.max_log2_tile_cols.max(l.min_log2_tile_cols);
            let c = rng.range(i64::from(l.min_log2_tile_cols), i64::from(max_c)) as u32;
            let mr = l.min_log2_tile_rows(c);
            let r = rng.range(i64::from(mr), i64::from(l.max_log2_tile_rows.max(mr))) as u32;
            if rng.chance(1, 4) {
                o.uniform_tile_spacing = Some(true);
            }
            uniform_tile_info(seq, h.mi_cols, h.mi_rows, c, r)
        } else {
            o.uniform_tile_spacing = if rng.flag() { Some(false) } else { None };
            let mut widths = Vec::new();
            let mut start = 0;
            while start < l.sb_cols {
                let left = l.sb_cols - start;
                let max_w = left.min(l.max_tile_width_sb);
                let slots = (MAX_TILE_COLS - widths.len()) as u32;
                let min_w = left.div_ceil(slots).clamp(1, max_w);
                let wd = if rng.flag() {
                    max_w
                } else {
                    rng.range(i64::from(min_w), i64::from(max_w)) as u32
                };
                widths.push(wd);
                start += wd;
            }
            let widest = widths.iter().copied().max().unwrap_or(1);
            let max_h_sb = l.max_tile_height_sb(widest);
            let mut heights = Vec::new();
            let mut start = 0;
            while start < l.sb_rows {
                let left = l.sb_rows - start;
                let max_h = left.min(max_h_sb);
                let slots = (MAX_TILE_ROWS - heights.len()) as u32;
                if slots == 0 {
                    break;
                }
                let min_h = left.div_ceil(slots).clamp(1, max_h);
                let ht = rng.range(i64::from(min_h), i64::from(max_h)) as u32;
                heights.push(ht);
                start += ht;
            }
            if start < l.sb_rows {
                // Too many rows for this width: use the uniform layout.
                o.uniform_tile_spacing = None;
                uniform_tile_info(seq, h.mi_cols, h.mi_rows, 0, 0)
            } else {
                explicit_tile_info(seq, h.mi_cols, h.mi_rows, &widths, &heights)
            }
        };
        if ti.cols_log2 > 0 || ti.rows_log2 > 0 {
            ti.context_update_tile_id = rng.below((ti.cols * ti.rows) as u64) as u32;
            ti.tile_size_bytes = 1 + rng.below(4) as u32;
        }
        h.tile_info = ti;

        // Quantizer.
        let planes = seq.num_planes();
        let dq = |rng: &mut Rng| -> i8 {
            if rng.flag() {
                0
            } else {
                rng.range(-64, 63) as i8
            }
        };
        let lossless_try = rng.chance(1, 6);
        let mut q = Quant {
            base_q_idx: if lossless_try {
                0
            } else {
                rng.below(256) as u8
            },
            ..Quant::default()
        };
        if !lossless_try {
            q.delta_q_y_dc = dq(rng);
            if planes > 1 {
                q.delta_q_u_dc = dq(rng);
                q.delta_q_u_ac = dq(rng);
                if seq.separate_uv_delta_q && rng.flag() {
                    q.delta_q_v_dc = dq(rng);
                    q.delta_q_v_ac = dq(rng);
                    o.force_diff_uv_delta = rng.flag();
                } else {
                    q.delta_q_v_dc = q.delta_q_u_dc;
                    q.delta_q_v_ac = q.delta_q_u_ac;
                }
            }
        }
        q.using_qmatrix = rng.flag();
        if q.using_qmatrix {
            q.qm_y = rng.below(16) as u8;
            q.qm_u = rng.below(16) as u8;
            q.qm_v = if seq.separate_uv_delta_q {
                rng.below(16) as u8
            } else {
                q.qm_u
            };
        }
        h.quant = q;

        // Segmentation.
        let mut s = Segmentation::default();
        if let Some(p) = prev {
            s.feature_enabled = p.feature_enabled;
            s.feature_data = p.feature_data;
        }
        s.enabled = rng.flag();
        if s.enabled {
            if prev.is_none() {
                s.update_map = true;
                s.update_data = true;
            } else {
                s.update_map = rng.flag();
                s.temporal_update = s.update_map && rng.flag();
                s.update_data = rng.flag();
            }
            if s.update_data {
                for i in 0..8 {
                    for j in 0..8 {
                        s.feature_enabled[i][j] = rng.chance(1, 3);
                        s.feature_data[i][j] = if s.feature_enabled[i][j] {
                            random_feature(rng, j)
                        } else {
                            0
                        };
                    }
                }
            }
        } else {
            s.feature_enabled = [[false; 8]; 8];
            s.feature_data = [[0; 8]; 8];
        }
        for i in 0..8 {
            for j in 0..8 {
                if s.feature_enabled[i][j] {
                    s.last_active_seg_id = i as u8;
                    if j >= SEG_LVL_REF_FRAME {
                        s.seg_id_pre_skip = true;
                    }
                }
            }
        }
        h.segmentation = s;

        if h.quant.base_q_idx > 0 {
            h.delta_q_present = rng.flag();
        }
        if h.delta_q_present {
            h.delta_q_res = rng.below(4) as u8;
            if !h.allow_intrabc {
                h.delta_lf_present = rng.flag();
            }
            if h.delta_lf_present {
                h.delta_lf_res = rng.below(4) as u8;
                h.delta_lf_multi = rng.flag();
            }
        }
        h.coded_lossless = true;
        for seg in 0..8 {
            let ll = {
                let qp = &h.quant;
                h.qindex(seg) == 0
                    && qp.delta_q_y_dc == 0
                    && qp.delta_q_u_ac == 0
                    && qp.delta_q_u_dc == 0
                    && qp.delta_q_v_ac == 0
                    && qp.delta_q_v_dc == 0
            };
            h.lossless_array[seg] = ll;
            h.coded_lossless &= ll;
            if h.quant.using_qmatrix {
                let lv = if ll {
                    [15; 3]
                } else {
                    [h.quant.qm_y, h.quant.qm_u, h.quant.qm_v]
                };
                for p in 0..3 {
                    h.seg_qm_level[p][seg] = lv[p];
                }
            }
        }
        h.all_lossless = h.coded_lossless && h.frame_width == h.upscaled_width;

        // Loop filter.
        let mut lf = LoopFilter {
            ref_deltas: prev.map_or(DEFAULT_LF_REF_DELTAS, |p| p.loop_filter_ref_deltas),
            mode_deltas: prev.map_or([0; 2], |p| p.loop_filter_mode_deltas),
            ..LoopFilter::default()
        };
        if h.coded_lossless || h.allow_intrabc {
            lf.ref_deltas = DEFAULT_LF_REF_DELTAS;
            lf.mode_deltas = [0; 2];
        } else {
            lf.level[0] = rng.below(64) as u8;
            lf.level[1] = rng.below(64) as u8;
            if planes > 1 && (lf.level[0] != 0 || lf.level[1] != 0) {
                lf.level[2] = rng.below(64) as u8;
                lf.level[3] = rng.below(64) as u8;
            }
            lf.sharpness = rng.below(8) as u8;
            lf.delta_enabled = rng.flag();
            if lf.delta_enabled {
                lf.delta_update = rng.flag();
                if lf.delta_update {
                    o.lf_delta_update_all = rng.chance(1, 4);
                    for i in 0..8 {
                        if rng.flag() {
                            lf.ref_deltas[i] = rng.range(-64, 63) as i8;
                        }
                    }
                    for i in 0..2 {
                        if rng.flag() {
                            lf.mode_deltas[i] = rng.range(-64, 63) as i8;
                        }
                    }
                }
            }
        }
        h.loop_filter = lf;

        // CDEF.
        let mut c = Cdef {
            damping: 3,
            ..Cdef::default()
        };
        if !(h.coded_lossless || h.allow_intrabc || !seq.enable_cdef) {
            c.damping = 3 + rng.below(4) as u8;
            c.bits = rng.below(4) as u8;
            let sec = |rng: &mut Rng| [0u8, 1, 2, 4][rng.below(4) as usize];
            for i in 0..(1usize << c.bits) {
                c.y_pri[i] = rng.below(16) as u8;
                c.y_sec[i] = sec(rng);
                if planes > 1 {
                    c.uv_pri[i] = rng.below(16) as u8;
                    c.uv_sec[i] = sec(rng);
                }
            }
        }
        h.cdef = c;

        // Loop restoration.
        let mut lr = Restoration::default();
        if !(h.all_lossless || h.allow_intrabc || !seq.enable_restoration) {
            let mut uses_chroma = false;
            for i in 0..planes {
                lr.frame_type[i] = rng.below(4) as u8;
                if lr.frame_type[i] != RESTORE_NONE {
                    lr.uses_lr = true;
                    uses_chroma |= i > 0;
                }
            }
            if lr.uses_lr {
                let shift = if seq.use_128x128_superblock {
                    1 + rng.below(2) as u32
                } else {
                    rng.below(3) as u32
                };
                lr.unit_size[0] = 256 >> (2 - shift);
                let uv = if seq.subsampling_x && seq.subsampling_y && uses_chroma {
                    rng.below(2) as u32
                } else {
                    0
                };
                lr.unit_size[1] = lr.unit_size[0] >> uv;
                lr.unit_size[2] = lr.unit_size[0] >> uv;
            }
        }
        h.lr = lr;

        h.tx_mode = if h.coded_lossless {
            0
        } else if rng.flag() {
            TX_MODE_SELECT
        } else {
            1
        };
        h.reference_select = !h.frame_is_intra && rng.flag();
        if let Some(f) = skip_mode_frames(seq, &h, &roh) {
            h.skip_mode_frame = f;
            h.skip_mode_present = rng.flag();
        }
        h.allow_warped_motion =
            !(h.frame_is_intra || h.error_resilient_mode || !seq.enable_warped_motion)
                && rng.flag();
        h.reduced_tx_set = rng.flag();
        h.gm_params = default_gm_params();
        if !h.frame_is_intra {
            let (ty, p) = random_gm(rng, h.allow_high_precision_mv);
            h.gm_type = ty;
            h.gm_params = p;
        }

        // Film grain.
        if seq.film_grain_params_present && (h.show_frame || h.showable_frame) && rng.chance(3, 4) {
            let mut g = random_grain(rng, seq);
            if h.frame_type == INTER_FRAME && rng.flag() {
                let idx = h.ref_frame_idx[rng.below(7) as usize];
                let seed = g.grain_seed;
                g = refs[idx as usize].grain.clone();
                g.grain_seed = seed;
                g.apply_grain = true;
                g.update_grain = false;
                if rng.flag() {
                    o.film_grain_params_ref_idx = Some(idx);
                }
            }
            h.grain = g;
        }
        (h, o)
    }

    #[test]
    fn frame_header_round_trip() {
        let mut rng = Rng(0x0bad_cafe);
        let mut kinds = [0usize; 4];
        let mut counts = [0usize; 12];
        for iter in 0..20_000 {
            let seq = random_seq(&mut rng);
            let id_len = seq.additional_frame_id_length + seq.delta_frame_id_length;
            let cur_id = if seq.frame_id_numbers_present {
                rng.below(1 << id_len) as u32
            } else {
                0
            };
            let refs: [RefHeaderState; 8] =
                std::array::from_fn(|_| random_ref(&mut rng, &seq, cur_id));
            let tid = rng.below(8) as u8;
            let sid = rng.below(4) as u8;
            let ctx = HeaderContext {
                seq: &seq,
                refs: &refs,
                temporal_id: tid,
                spatial_id: sid,
            };
            let (mut h, o) = random_frame(&mut rng, &seq, &refs, cur_id, tid, sid);
            for (frame_obu, bytes) in [
                (true, write_frame_header(&h, &ctx, &o)),
                (false, write_frame_header_obu(&h, &ctx, &o)),
            ] {
                let p = parse_frame_header(&bytes, &ctx)
                    .unwrap_or_else(|e| panic!("iter {iter}: {e:?}\n{h:#?}\n{seq:#?}"));
                if frame_obu && !h.show_existing_frame {
                    assert_eq!(p.header_bytes, bytes.len(), "iter {iter}");
                }
                h.header_bytes = p.header_bytes;
                assert_eq!(p, h, "iter {iter}\n{seq:#?}");
                // The parsed header written with default options parses the same.
                let o2 = FrameHeaderWriteOptions {
                    frame_presentation_time: o.frame_presentation_time,
                    buffer_removal_times: o.buffer_removal_times.clone(),
                    ..FrameHeaderWriteOptions::default()
                };
                let again = write_frame_header(&p, &ctx, &o2);
                let mut p2 = parse_frame_header(&again, &ctx).unwrap();
                p2.header_bytes = p.header_bytes;
                assert_eq!(p2, p, "iter {iter} (default options)");
                assert!(again.len() <= bytes.len() + 1, "iter {iter}");
            }
            kinds[h.frame_type as usize] += 1;
            let ti = &h.tile_info;
            let uniform = ti.mi_col_starts
                == uniform_tile_info(&seq, h.mi_cols, h.mi_rows, ti.cols_log2, ti.rows_log2)
                    .mi_col_starts;
            for (k, c) in [
                h.show_existing_frame,
                ti.cols * ti.rows > 1 && uniform,
                ti.cols * ti.rows > 1 && !uniform,
                h.allow_intrabc,
                h.grain.apply_grain && !h.grain.update_grain,
                h.grain.apply_grain && h.grain.update_grain,
                h.skip_mode_present,
                h.segmentation.enabled && h.segmentation.update_data,
                h.coded_lossless,
                h.use_superres,
                h.gm_type.contains(&AFFINE),
                h.lr.uses_lr && h.loop_filter.delta_update,
            ]
            .into_iter()
            .enumerate()
            {
                counts[k] += usize::from(c);
            }
        }
        assert!(kinds.iter().all(|&k| k > 1000), "{kinds:?}");
        assert!(counts.iter().all(|&k| k > 50), "{counts:?}");
    }

    #[test]
    fn tile_group_and_obus() {
        let mut rng = Rng(99);
        let mut seq = random_seq(&mut rng);
        seq.use_128x128_superblock = false;
        let mut ti = uniform_tile_info(&seq, 128, 64, 2, 1);
        assert_eq!((ti.cols, ti.rows), (4, 2));
        let tiles: Vec<Vec<u8>> = (0..8)
            .map(|i| (0..(i * 100 + 1)).map(|x| x as u8).collect())
            .collect();
        ti.tile_size_bytes = tile_size_bytes_for(701);
        assert_eq!(ti.tile_size_bytes, 2);
        assert_eq!(tile_size_bytes_for(256), 1);
        assert_eq!(tile_size_bytes_for(257), 2);
        for (a, b) in [(0usize, 7usize), (0, 2), (3, 7), (5, 5)] {
            let payload = write_tile_group(&ti, a, b, &tiles[a..=b]);
            // Parse as the decoder does.
            let mut r = crate::dec::bits::BitReader::new(&payload);
            let present = r.flag().unwrap();
            assert_eq!(present, !(a == 0 && b == 7));
            if present {
                assert_eq!(r.f(3).unwrap() as usize, a);
                assert_eq!(r.f(3).unwrap() as usize, b);
            }
            r.byte_alignment();
            let mut pos = r.position() / 8;
            for t in a..=b {
                let size = if t == b {
                    payload.len() - pos
                } else {
                    let v = usize::from(payload[pos]) | (usize::from(payload[pos + 1]) << 8);
                    pos += 2;
                    v + 1
                };
                assert_eq!(&payload[pos..pos + size], &tiles[t][..]);
                pos += size;
            }
        }
        let one = TileInfo {
            cols: 1,
            rows: 1,
            ..TileInfo::default()
        };
        assert_eq!(write_tile_group(&one, 0, 0, &[[7u8, 8]]), vec![7, 8]);

        let mut out = Vec::new();
        write_temporal_delimiter(&mut out);
        write_obu(&mut out, OBU_FRAME, 0, 0, false, &[1, 2]);
        let obus = split_obus(&out).unwrap();
        assert_eq!(obus[0].obu_type, OBU_TEMPORAL_DELIMITER);
        assert!(obus[0].payload.is_empty());
        assert_eq!(obus[1].payload, &[1, 2]);
        assert_eq!(out[..2], [0x12, 0x00]);
    }

    /// Mirror of the parser's decode_unsigned_subexp_with_ref.
    fn decode_unsigned(rd: &mut crate::dec::bits::BitReader<'_>, mx: i32, r: i32) -> i32 {
        let mut i = 0u32;
        let mut mk = 0u32;
        let v = loop {
            let b2 = if i != 0 { 3 + i - 1 } else { 3 };
            let a = 1u32 << b2;
            if mx as u32 <= mk + 3 * a {
                break rd.ns(mx as u32 - mk).unwrap() + mk;
            }
            if rd.flag().unwrap() {
                i += 1;
                mk += a;
            } else {
                break rd.f(b2).unwrap() + mk;
            }
        } as i32;
        let inv = |r: i32, v: i32| {
            if v > 2 * r {
                v
            } else if v & 1 != 0 {
                r - ((v + 1) >> 1)
            } else {
                r + (v >> 1)
            }
        };
        if (r << 1) <= mx {
            inv(r, v)
        } else {
            mx - 1 - inv(mx - 1 - r, v)
        }
    }

    #[test]
    fn subexp_round_trip() {
        for mx in [17i32, 33, 513, 1025, 8193] {
            for r in (0..mx).step_by(5) {
                let xs: Vec<i32> = (0..mx).step_by(3).chain([mx - 1]).collect();
                let mut w = BitWriter::new();
                for &x in &xs {
                    encode_unsigned_subexp_with_ref(&mut w, mx, r, x);
                }
                let data = w.into_bytes();
                let mut rd = crate::dec::bits::BitReader::new(&data);
                for &x in &xs {
                    assert_eq!(decode_unsigned(&mut rd, mx, r), x, "mx {mx} r {r}");
                }
            }
        }
    }
}

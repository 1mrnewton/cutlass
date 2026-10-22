//! Uncompressed frame header.
//!
//! This parses the syntax in the frame-header section, including tool flags
//! the reconstruction path does not yet apply. Reconstruction rejects those
//! tools before it writes a frame. Inter frames and show-existing frames are
//! parsed far enough to name the gap, then rejected: their reference state is
//! not implemented.

use crate::bits::BitReader;
use crate::error::DecodeError;
use crate::limits::DecoderLimits;
use crate::math::clip3;
use crate::sequence::SequenceHeader;

pub const KEY_FRAME: u8 = 0;
pub const INTER_FRAME: u8 = 1;
pub const INTRA_ONLY_FRAME: u8 = 2;
pub const SWITCH_FRAME: u8 = 3;
pub const PRIMARY_REF_NONE: u8 = 7;

const RESTORE_NONE: u8 = 0;
const SEG_LVL_MAX: usize = 8;
const SEG_LVL_REF_FRAME: usize = 5;
const SEG_FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 6, 6, 6, 3, 0, 0];
const SEG_FEATURE_SIGNED: [bool; SEG_LVL_MAX] = [true, true, true, true, true, false, false, false];
const SEG_FEATURE_MAX: [i32; SEG_LVL_MAX] = [255, 63, 63, 63, 63, 7, 0, 0];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxMode {
    Only4x4,
    Largest,
    Select,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    pub order_hint: u32,
    pub primary_ref_frame: u8,
    pub refresh_frame_flags: u8,
    pub allow_intrabc: bool,
    /// Coded luma width after the super-resolution downsample.
    pub frame_width: u32,
    pub frame_height: u32,
    pub upscaled_width: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub superres_denom: u32,
    pub mi_cols: u32,
    pub mi_rows: u32,
    pub tile_cols: u32,
    pub tile_rows: u32,
    pub tile_cols_log2: u32,
    pub tile_rows_log2: u32,
    pub mi_col_starts: Vec<u32>,
    pub mi_row_starts: Vec<u32>,
    pub context_update_tile_id: u32,
    pub tile_size_bytes: u32,
    pub base_q_idx: u8,
    pub delta_q_y_dc: i32,
    pub delta_q_u_dc: i32,
    pub delta_q_u_ac: i32,
    pub delta_q_v_dc: i32,
    pub delta_q_v_ac: i32,
    pub using_qmatrix: bool,
    pub segmentation_enabled: bool,
    pub seg_id_pre_skip: bool,
    pub last_active_seg_id: u8,
    pub delta_q_present: bool,
    pub delta_q_res: u8,
    pub delta_lf_present: bool,
    pub coded_lossless: bool,
    pub all_lossless: bool,
    pub loop_filter_level: [u8; 4],
    pub cdef_bits: u8,
    pub cdef_y_pri: [u8; 8],
    pub cdef_y_sec: [u8; 8],
    pub cdef_uv_pri: [u8; 8],
    pub cdef_uv_sec: [u8; 8],
    pub restoration_type: [u8; 3],
    pub tx_mode: TxMode,
    pub reduced_tx_set: bool,
    pub apply_grain: bool,
    pub grain_seed: u16,
}

pub fn parse_uncompressed(
    b: &mut BitReader<'_>,
    seq: &SequenceHeader,
    limits: &DecoderLimits,
) -> Result<FrameHeader, DecodeError> {
    let mut show_existing = false;
    let frame_to_show = 0u8;
    let frame_type;
    let frame_is_intra;
    let show_frame;
    let showable_frame;
    let error_resilient_mode;

    if seq.reduced_still_picture_header {
        frame_type = KEY_FRAME;
        frame_is_intra = true;
        show_frame = true;
        showable_frame = false;
        error_resilient_mode = true;
    } else {
        show_existing = b.f(1)? == 1;
        if show_existing {
            let _ = b.f(3)?;
            if seq.decoder_model_info_present && !seq.equal_picture_interval {
                let n = u32::from(seq.frame_presentation_time_length_minus_1) + 1;
                let _pts = b.f(n)?;
            }
            if seq.frame_id_numbers_present {
                let _id = b.f(seq.id_len())?;
            }
            if seq.film_grain_params_present {
                return Err(DecodeError::unsupported(
                    "show_existing grain load is not implemented",
                ));
            }
            return Err(DecodeError::unsupported(
                "show_existing frames need saved reference images",
            ));
        }
        frame_type = b.f(2)? as u8;
        frame_is_intra = frame_type == INTRA_ONLY_FRAME || frame_type == KEY_FRAME;
        show_frame = b.f(1)? == 1;
        if show_frame && seq.decoder_model_info_present && !seq.equal_picture_interval {
            let n = u32::from(seq.frame_presentation_time_length_minus_1) + 1;
            let _pts = b.f(n)?;
        }
        showable_frame = if show_frame {
            frame_type != KEY_FRAME
        } else {
            b.f(1)? == 1
        };
        error_resilient_mode =
            if frame_type == SWITCH_FRAME || (frame_type == KEY_FRAME && show_frame) {
                true
            } else {
                b.f(1)? == 1
            };
    }

    let disable_cdf_update = b.f(1)? == 1;
    let allow_screen_content_tools =
        if seq.seq_force_screen_content_tools == crate::sequence::SELECT_SCREEN_CONTENT_TOOLS {
            b.f(1)? == 1
        } else {
            seq.seq_force_screen_content_tools == 1
        };
    let mut force_integer_mv = if allow_screen_content_tools {
        if seq.seq_force_integer_mv == crate::sequence::SELECT_INTEGER_MV {
            b.f(1)? == 1
        } else {
            seq.seq_force_integer_mv == 1
        }
    } else {
        false
    };
    if frame_is_intra {
        force_integer_mv = true;
    }
    let current_frame_id = if seq.frame_id_numbers_present {
        b.f(seq.id_len())?
    } else {
        0
    };
    let frame_size_override = if frame_type == SWITCH_FRAME {
        true
    } else if seq.reduced_still_picture_header {
        false
    } else {
        b.f(1)? == 1
    };
    let order_hint = if seq.order_hint_bits == 0 {
        0
    } else {
        b.f(u32::from(seq.order_hint_bits))?
    };
    let primary_ref_frame = if frame_is_intra || error_resilient_mode {
        PRIMARY_REF_NONE
    } else {
        b.f(3)? as u8
    };

    if !frame_is_intra {
        return Err(DecodeError::unsupported(
            "inter frame reconstruction is not implemented",
        ));
    }

    let refresh_frame_flags =
        if frame_type == SWITCH_FRAME || (frame_type == KEY_FRAME && show_frame) {
            (1u16 << 8) - 1
        } else {
            u16::from(b.f(8)? as u8)
        } as u8;

    let (frame_width, frame_height, upscaled_width, render_width, render_height, superres_denom) =
        read_frame_size(b, seq, frame_size_override)?;
    if upscaled_width > limits.max_dimension
        || frame_height > limits.max_dimension
        || frame_width > limits.max_dimension
    {
        return Err(DecodeError::limit("frame dimensions exceed max_dimension"));
    }
    let pixels = u64::from(upscaled_width) * u64::from(frame_height);
    if pixels > limits.max_pixels {
        return Err(DecodeError::limit("frame pixels exceed max_pixels"));
    }
    let mi_cols = 2 * ((frame_width + 7) >> 3);
    let mi_rows = 2 * ((frame_height + 7) >> 3);

    let allow_intrabc = allow_screen_content_tools && upscaled_width == frame_width && b.f(1)? == 1;

    let disable_frame_end_update_cdf = seq.reduced_still_picture_header || disable_cdf_update;
    let _ = disable_frame_end_update_cdf;

    let (
        tile_cols,
        tile_rows,
        tile_cols_log2,
        tile_rows_log2,
        mi_col_starts,
        mi_row_starts,
        context_update_tile_id,
        tile_size_bytes,
    ) = tile_info(b, seq.use_128x128_superblock, mi_cols, mi_rows)?;

    let base_q_idx = b.f(8)? as u8;
    let delta_q_y_dc = read_delta_q(b)?;
    let (delta_q_u_dc, delta_q_u_ac, delta_q_v_dc, delta_q_v_ac) = if seq.num_planes() > 1 {
        let diff_uv = if seq.separate_uv_delta_q {
            b.f(1)? == 1
        } else {
            false
        };
        let u_dc = read_delta_q(b)?;
        let u_ac = read_delta_q(b)?;
        if diff_uv {
            (u_dc, u_ac, read_delta_q(b)?, read_delta_q(b)?)
        } else {
            (u_dc, u_ac, u_dc, u_ac)
        }
    } else {
        (0, 0, 0, 0)
    };
    let using_qmatrix = b.f(1)? == 1;
    if using_qmatrix {
        let _qm_y = b.f(4)?;
        let qm_u = b.f(4)?;
        if seq.separate_uv_delta_q {
            let _qm_v = b.f(4)?;
        } else {
            let _qm_v = qm_u;
        }
    }

    let (segmentation_enabled, seg_id_pre_skip, last_active_seg_id) =
        segmentation(b, primary_ref_frame)?;

    let mut coded_lossless = true;
    for _seg in 0..8 {
        // Segmentation features that change qindex are not applied yet.
        // With segmentation disabled every segment uses base_q_idx.
        let qindex = i32::from(base_q_idx);
        let lossless = qindex == 0
            && delta_q_y_dc == 0
            && delta_q_u_ac == 0
            && delta_q_u_dc == 0
            && delta_q_v_ac == 0
            && delta_q_v_dc == 0;
        if !lossless {
            coded_lossless = false;
        }
    }
    let all_lossless = coded_lossless && frame_width == upscaled_width;

    let (delta_q_present, delta_q_res) = if base_q_idx > 0 {
        let present = b.f(1)? == 1;
        let res = if present { b.f(2)? as u8 } else { 0 };
        (present, res)
    } else {
        (false, 0)
    };
    let delta_lf_present = if delta_q_present && !allow_intrabc {
        let present = b.f(1)? == 1;
        if present {
            let _res = b.f(2)?;
            let _multi = b.f(1)?;
        }
        present
    } else {
        false
    };

    let loop_filter_level = loop_filter(b, seq, coded_lossless, allow_intrabc)?;
    let (cdef_bits, cdef_y_pri, cdef_y_sec, cdef_uv_pri, cdef_uv_sec) =
        cdef_params(b, seq, coded_lossless, allow_intrabc)?;
    let restoration_type = restoration(b, seq, all_lossless, allow_intrabc)?;

    let tx_mode = if coded_lossless {
        TxMode::Only4x4
    } else if b.f(1)? == 1 {
        TxMode::Select
    } else {
        TxMode::Largest
    };

    let reduced_tx_set = b.f(1)? == 1;
    // Intra frames do not read global-motion parameters. The models stay identity.
    let (apply_grain, grain_seed) = film_grain(b, seq, show_frame, showable_frame, frame_type)?;

    Ok(FrameHeader {
        show_existing_frame: show_existing,
        frame_to_show_map_idx: frame_to_show,
        frame_type,
        frame_is_intra,
        show_frame,
        showable_frame,
        error_resilient_mode,
        disable_cdf_update,
        allow_screen_content_tools,
        force_integer_mv,
        current_frame_id,
        order_hint,
        primary_ref_frame,
        refresh_frame_flags,
        allow_intrabc,
        frame_width,
        frame_height,
        upscaled_width,
        render_width,
        render_height,
        superres_denom,
        mi_cols,
        mi_rows,
        tile_cols,
        tile_rows,
        tile_cols_log2,
        tile_rows_log2,
        mi_col_starts,
        mi_row_starts,
        context_update_tile_id,
        tile_size_bytes,
        base_q_idx,
        delta_q_y_dc,
        delta_q_u_dc,
        delta_q_u_ac,
        delta_q_v_dc,
        delta_q_v_ac,
        using_qmatrix,
        segmentation_enabled,
        seg_id_pre_skip,
        last_active_seg_id,
        delta_q_present,
        delta_q_res,
        delta_lf_present,
        coded_lossless,
        all_lossless,
        loop_filter_level,
        cdef_bits,
        cdef_y_pri,
        cdef_y_sec,
        cdef_uv_pri,
        cdef_uv_sec,
        restoration_type,
        tx_mode,
        reduced_tx_set,
        apply_grain,
        grain_seed,
    })
}

fn read_delta_q(b: &mut BitReader<'_>) -> Result<i32, DecodeError> {
    if b.f(1)? == 1 { b.su(7) } else { Ok(0) }
}

fn read_frame_size(
    b: &mut BitReader<'_>,
    seq: &SequenceHeader,
    frame_size_override: bool,
) -> Result<(u32, u32, u32, u32, u32, u32), DecodeError> {
    let (mut frame_width, frame_height) = if frame_size_override {
        let w = b.f(u32::from(seq.frame_width_bits))? + 1;
        let h = b.f(u32::from(seq.frame_height_bits))? + 1;
        (w, h)
    } else {
        (seq.max_frame_width, seq.max_frame_height)
    };
    let (upscaled, denom) = superres(b, seq, frame_width)?;
    frame_width = upscaled.0;
    let upscaled_width = upscaled.1;
    let (render_width, render_height) = render_size(b, upscaled_width, frame_height)?;
    Ok((
        frame_width,
        frame_height,
        upscaled_width,
        render_width,
        render_height,
        denom,
    ))
}

/// Returns `(coded_width, upscaled_width)`.
fn superres(
    b: &mut BitReader<'_>,
    seq: &SequenceHeader,
    frame_width: u32,
) -> Result<((u32, u32), u32), DecodeError> {
    let use_superres = if seq.enable_superres {
        b.f(1)? == 1
    } else {
        false
    };
    let denom = if use_superres { b.f(3)? + 9 } else { 8 };
    let upscaled = frame_width;
    let coded = ((u64::from(upscaled) * 8 + u64::from(denom / 2)) / u64::from(denom)) as u32;
    if coded == 0 {
        return Err(DecodeError::malformed("superres coded width is 0"));
    }
    Ok(((coded, upscaled), denom))
}

fn render_size(
    b: &mut BitReader<'_>,
    upscaled_width: u32,
    frame_height: u32,
) -> Result<(u32, u32), DecodeError> {
    if b.f(1)? == 1 {
        let w = b.f(16)? + 1;
        let h = b.f(16)? + 1;
        Ok((w, h))
    } else {
        Ok((upscaled_width, frame_height))
    }
}

fn tile_log2(blk: u32, target: u32) -> u32 {
    let mut k = 0u32;
    let mut size = u64::from(blk);
    while size < u64::from(target) {
        size <<= 1;
        k += 1;
        if k > 31 {
            break;
        }
    }
    k
}

#[allow(clippy::type_complexity, clippy::needless_late_init)]
fn tile_info(
    b: &mut BitReader<'_>,
    use_128: bool,
    mi_cols: u32,
    mi_rows: u32,
) -> Result<(u32, u32, u32, u32, Vec<u32>, Vec<u32>, u32, u32), DecodeError> {
    let sb_cols = if use_128 {
        (mi_cols + 31) >> 5
    } else {
        (mi_cols + 15) >> 4
    };
    let sb_rows = if use_128 {
        (mi_rows + 31) >> 5
    } else {
        (mi_rows + 15) >> 4
    };
    let sb_shift = if use_128 { 5 } else { 4 };
    let sb_size = sb_shift + 2;
    let max_tile_width_sb = 4096 >> sb_size;
    let max_tile_area_sb = (4096u32 * 2304) >> (2 * sb_size);
    let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(64));
    let max_log2_tile_rows = tile_log2(1, sb_rows.min(64));
    let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));

    let mut mi_col_starts = Vec::new();
    let mut mi_row_starts = Vec::new();
    let tile_cols;
    let tile_rows;
    let tile_cols_log2;
    let tile_rows_log2;

    if b.f(1)? == 1 {
        tile_cols_log2 = {
            let mut log = min_log2_tile_cols;
            while log < max_log2_tile_cols {
                if b.f(1)? == 1 {
                    log += 1;
                } else {
                    break;
                }
            }
            log
        };
        let tile_width_sb = (sb_cols + (1 << tile_cols_log2) - 1) >> tile_cols_log2;
        let mut start = 0u32;
        while start < sb_cols {
            mi_col_starts.push(start << sb_shift);
            start += tile_width_sb;
        }
        mi_col_starts.push(mi_cols);
        tile_cols = (mi_col_starts.len() - 1) as u32;

        let min_log2_tile_rows = min_log2_tiles.saturating_sub(tile_cols_log2);
        tile_rows_log2 = {
            let mut log = min_log2_tile_rows;
            while log < max_log2_tile_rows {
                if b.f(1)? == 1 {
                    log += 1;
                } else {
                    break;
                }
            }
            log
        };
        let tile_height_sb = (sb_rows + (1 << tile_rows_log2) - 1) >> tile_rows_log2;
        let mut start = 0u32;
        while start < sb_rows {
            mi_row_starts.push(start << sb_shift);
            start += tile_height_sb;
        }
        mi_row_starts.push(mi_rows);
        tile_rows = (mi_row_starts.len() - 1) as u32;
    } else {
        return Err(DecodeError::unsupported(
            "non-uniform tile spacing is parsed in a later increment",
        ));
    }

    let (context_update_tile_id, tile_size_bytes) = if tile_cols_log2 > 0 || tile_rows_log2 > 0 {
        let id = b.f(tile_rows_log2 + tile_cols_log2)?;
        let size_bytes = b.f(2)? + 1;
        (id, size_bytes)
    } else {
        (0, 0)
    };
    if tile_cols == 0 || tile_rows == 0 || tile_cols > 64 || tile_rows > 64 {
        return Err(DecodeError::malformed("tile count"));
    }
    Ok((
        tile_cols,
        tile_rows,
        tile_cols_log2,
        tile_rows_log2,
        mi_col_starts,
        mi_row_starts,
        context_update_tile_id,
        tile_size_bytes,
    ))
}

fn segmentation(b: &mut BitReader<'_>, primary_ref: u8) -> Result<(bool, bool, u8), DecodeError> {
    let enabled = b.f(1)? == 1;
    let mut enabled_feat = [[false; SEG_LVL_MAX]; 8];
    let mut data = [[0i32; SEG_LVL_MAX]; 8];
    if enabled {
        let (update_map, update_data) = if primary_ref == PRIMARY_REF_NONE {
            (true, true)
        } else {
            let update_map = b.f(1)? == 1;
            if update_map {
                let _temporal = b.f(1)?;
            }
            let update_data = b.f(1)? == 1;
            (update_map, update_data)
        };
        let _ = update_map;
        if update_data {
            for seg in 0..8 {
                for level in 0..SEG_LVL_MAX {
                    let on = b.f(1)? == 1;
                    enabled_feat[seg][level] = on;
                    if on {
                        let bits = SEG_FEATURE_BITS[level];
                        let limit = SEG_FEATURE_MAX[level];
                        let value = if SEG_FEATURE_SIGNED[level] {
                            b.su(1 + bits)?
                        } else {
                            b.f(bits)? as i32
                        };
                        let lo = if SEG_FEATURE_SIGNED[level] { -limit } else { 0 };
                        data[seg][level] = clip3(lo, limit, value);
                    }
                }
            }
        }
    }
    let mut pre_skip = false;
    let mut last = 0u8;
    for (seg, levels) in enabled_feat.iter().enumerate() {
        for (level, &on) in levels.iter().enumerate() {
            if on {
                last = seg as u8;
                if level >= SEG_LVL_REF_FRAME {
                    pre_skip = true;
                }
            }
        }
    }
    let _ = data;
    Ok((enabled, pre_skip, last))
}

fn loop_filter(
    b: &mut BitReader<'_>,
    seq: &SequenceHeader,
    coded_lossless: bool,
    allow_intrabc: bool,
) -> Result<[u8; 4], DecodeError> {
    if coded_lossless || allow_intrabc {
        return Ok([0; 4]);
    }
    let mut level = [0u8; 4];
    level[0] = b.f(6)? as u8;
    level[1] = b.f(6)? as u8;
    if seq.num_planes() > 1 && (level[0] != 0 || level[1] != 0) {
        level[2] = b.f(6)? as u8;
        level[3] = b.f(6)? as u8;
    }
    let _sharpness = b.f(3)?;
    if b.f(1)? == 1 && b.f(1)? == 1 {
        for _ in 0..8 {
            if b.f(1)? == 1 {
                let _delta = b.su(7)?;
            }
        }
        for _ in 0..2 {
            if b.f(1)? == 1 {
                let _delta = b.su(7)?;
            }
        }
    }
    Ok(level)
}

type CdefHeader = (u8, [u8; 8], [u8; 8], [u8; 8], [u8; 8]);

fn cdef_params(
    b: &mut BitReader<'_>,
    seq: &SequenceHeader,
    coded_lossless: bool,
    allow_intrabc: bool,
) -> Result<CdefHeader, DecodeError> {
    let mut y_pri = [0u8; 8];
    let mut y_sec = [0u8; 8];
    let mut uv_pri = [0u8; 8];
    let mut uv_sec = [0u8; 8];
    if coded_lossless || allow_intrabc || !seq.enable_cdef {
        return Ok((0, y_pri, y_sec, uv_pri, uv_sec));
    }
    let _damping = b.f(2)? + 3;
    let bits = b.f(2)? as u8;
    let n = 1usize << bits;
    for i in 0..n {
        y_pri[i] = b.f(4)? as u8;
        y_sec[i] = b.f(2)? as u8;
        if y_sec[i] == 3 {
            y_sec[i] += 1;
        }
        if seq.num_planes() > 1 {
            uv_pri[i] = b.f(4)? as u8;
            uv_sec[i] = b.f(2)? as u8;
            if uv_sec[i] == 3 {
                uv_sec[i] += 1;
            }
        }
    }
    Ok((bits, y_pri, y_sec, uv_pri, uv_sec))
}

fn restoration(
    b: &mut BitReader<'_>,
    seq: &SequenceHeader,
    all_lossless: bool,
    allow_intrabc: bool,
) -> Result<[u8; 3], DecodeError> {
    if all_lossless || allow_intrabc || !seq.enable_restoration {
        return Ok([RESTORE_NONE; 3]);
    }
    let mut kind = [RESTORE_NONE; 3];
    let mut uses = false;
    let mut uses_chroma = false;
    let remap = [0u8, 3, 1, 2]; // none, switchable, wiener, sgr
    for (plane, slot) in kind.iter_mut().take(seq.num_planes()).enumerate() {
        let lr_type = b.f(2)? as usize;
        *slot = remap[lr_type];
        if *slot != RESTORE_NONE {
            uses = true;
            if plane > 0 {
                uses_chroma = true;
            }
        }
    }
    if uses {
        let mut shift = b.f(1)?;
        if seq.use_128x128_superblock {
            shift += 1;
        } else if shift == 1 {
            shift += b.f(1)?;
        }
        let _size = 256u32 >> (2 - shift);
        if seq.subsampling_x == 1 && seq.subsampling_y == 1 && uses_chroma {
            let _uv = b.f(1)?;
        }
    }
    Ok(kind)
}

fn film_grain(
    b: &mut BitReader<'_>,
    seq: &SequenceHeader,
    show_frame: bool,
    showable: bool,
    frame_type: u8,
) -> Result<(bool, u16), DecodeError> {
    if !seq.film_grain_params_present || (!show_frame && !showable) {
        return Ok((false, 0));
    }
    if b.f(1)? == 0 {
        return Ok((false, 0));
    }
    let seed = b.f(16)? as u16;
    let update = if frame_type == INTER_FRAME {
        b.f(1)? == 1
    } else {
        true
    };
    if !update {
        let _idx = b.f(3)?;
        return Ok((true, seed));
    }
    let num_y = b.f(4)? as usize;
    if num_y > 14 {
        return Err(DecodeError::malformed("grain y points"));
    }
    for _ in 0..num_y {
        let _v = b.f(8)?;
        let _s = b.f(8)?;
    }
    let chroma_from_luma = if seq.mono_chrome { false } else { b.f(1)? == 1 };
    let (num_cb, num_cr) = if seq.mono_chrome
        || chroma_from_luma
        || (seq.subsampling_x == 1 && seq.subsampling_y == 1 && num_y == 0)
    {
        (0usize, 0usize)
    } else {
        let cb = b.f(4)? as usize;
        if cb > 10 {
            return Err(DecodeError::malformed("grain cb points"));
        }
        for _ in 0..cb {
            let _ = b.f(8)?;
            let _ = b.f(8)?;
        }
        let cr = b.f(4)? as usize;
        if cr > 10 {
            return Err(DecodeError::malformed("grain cr points"));
        }
        for _ in 0..cr {
            let _ = b.f(8)?;
            let _ = b.f(8)?;
        }
        (cb, cr)
    };
    let _scaling = b.f(2)?;
    let lag = b.f(2)?;
    let num_pos_luma = 2 * lag * (lag + 1);
    let num_pos_chroma = if num_y > 0 {
        for _ in 0..num_pos_luma {
            let _ = b.f(8)?;
        }
        num_pos_luma + 1
    } else {
        num_pos_luma
    };
    if chroma_from_luma || num_cb > 0 {
        for _ in 0..num_pos_chroma {
            let _ = b.f(8)?;
        }
    }
    if chroma_from_luma || num_cr > 0 {
        for _ in 0..num_pos_chroma {
            let _ = b.f(8)?;
        }
    }
    let _ar_shift = b.f(2)?;
    let _grain_shift = b.f(2)?;
    if num_cb > 0 {
        let _ = b.f(8)?;
        let _ = b.f(8)?;
        let _ = b.f(9)?;
    }
    if num_cr > 0 {
        let _ = b.f(8)?;
        let _ = b.f(8)?;
        let _ = b.f(9)?;
    }
    let _overlap = b.f(1)?;
    let _clip = b.f(1)?;
    Ok((true, seed))
}

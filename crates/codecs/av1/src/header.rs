use cutlass_decode::CodecError;

use crate::bits::{BitReader, read_leb128};

pub const KEY_FRAME: u8 = 0;
pub const INTER_FRAME: u8 = 1;
pub const INTRA_ONLY: u8 = 2;
pub const SWITCH_FRAME: u8 = 3;
const SELECT_TOOLS: u8 = 2;
const PRIMARY_REF_NONE: u8 = 7;

#[derive(Clone, Debug)]
pub struct Sequence {
    pub profile: u8,
    pub still: bool,
    pub reduced: bool,
    pub width: u32,
    pub height: u32,
    pub width_bits: u32,
    pub height_bits: u32,
    pub order_hint_bits: u8,
    pub enable_order_hint: bool,
    pub enable_intra_edge: bool,
    pub enable_filter_intra: bool,
    pub enable_interintra: bool,
    pub enable_masked: bool,
    pub enable_warped: bool,
    pub enable_dual_filter: bool,
    pub enable_jnt: bool,
    pub enable_ref_mvs: bool,
    pub enable_superres: bool,
    pub screen_select: bool,
    pub force_integer_select: bool,
    pub force_screen: u8,
    pub force_integer: u8,
    pub bit_depth: u8,
    pub monochrome: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub color_range_full: bool,
    pub separate_uv_delta_q: bool,
    pub film_grain: bool,
    pub frame_id_present: bool,
    pub id_len: u8,
    pub delta_frame_id_len: u8,
    pub use_128: bool,
}

#[derive(Clone, Debug)]
pub struct FrameInfo {
    pub frame_type: u8,
    pub show_frame: bool,
    pub showable: bool,
    pub show_existing: bool,
    pub frame_to_show: u8,
    pub error_resilient: bool,
    pub disable_cdf_update: bool,
    pub disable_frame_end_update: bool,
    pub allow_screen: bool,
    pub force_integer_mv: bool,
    pub intra: bool,
    pub width: u32,
    pub height: u32,
    pub upscaled_width: u32,
    pub order_hint: u32,
    pub refresh_flags: u8,
    pub primary_ref: u8,
    pub base_q: u8,
    pub delta_y_dc: i32,
    pub delta_u_dc: i32,
    pub delta_u_ac: i32,
    pub delta_v_dc: i32,
    pub delta_v_ac: i32,
    pub using_qmatrix: bool,
    pub tx_select: bool,
    pub reduced_tx: bool,
    pub reference_select: bool,
    pub skip_mode: bool,
    pub allow_intrabc: bool,
    pub allow_high_precision_mv: bool,
    pub interpolation: u8,
    pub filter_switchable: bool,
    pub motion_mode_switchable: bool,
    pub use_ref_frame_mvs: bool,
    pub allow_warped: bool,
    pub lf_level: [u8; 4],
    pub lf_sharpness: u8,
    pub lf_delta_enabled: bool,
    pub lf_ref_deltas: [i8; 8],
    pub lf_mode_deltas: [i8; 2],
    pub segmentation: bool,
    pub lossless: bool,
    pub all_lossless: bool,
    pub ref_frames: [u8; 7],
    pub tile: Vec<u8>,
    pub mi_cols: u32,
    pub mi_rows: u32,
}

pub fn parse_unit(
    bytes: &[u8],
    sequence: Option<&Sequence>,
) -> Result<(Sequence, FrameInfo), CodecError> {
    let mut bits = BitReader::new(bytes);
    let mut seq = sequence.cloned();
    let mut frame = None;
    while bits.bits_left() >= 8 {
        let start = bits.position();
        let header = bits.read(8)?;
        if header & 0x80 != 0 {
            return Err(malformed(&bits, "obu forbidden"));
        }
        let obu_type = ((header >> 3) & 0xf) as u8;
        let extension = (header >> 2) & 1 == 1;
        let has_size = (header >> 1) & 1 == 1;
        if !has_size {
            return Err(CodecError::Unsupported {
                what: "obu without a size",
            });
        }
        if extension {
            let _ = bits.read(8)?;
        }
        let size = read_leb128(&mut bits)? as usize;
        let payload_bits = size.saturating_mul(8);
        if bits.bits_left() < payload_bits {
            return Err(CodecError::Truncated {
                at_bit: bits.position() as u64,
                what: "obu",
            });
        }
        let payload_end = bits.position() + payload_bits;
        match obu_type {
            1 => {
                seq = Some(parse_sequence(&mut bits)?);
                trailing(&mut bits, payload_end)?;
            }
            2 => {}
            6 => {
                let sequence = seq.as_ref().ok_or(CodecError::InvalidApi {
                    what: "frame before sequence",
                })?;
                frame = Some(parse_frame(&mut bits, sequence, payload_end)?);
            }
            3 | 7 => {
                let sequence = seq.as_ref().ok_or(CodecError::InvalidApi {
                    what: "frame before sequence",
                })?;
                frame = Some(parse_frame_header_only(&mut bits, sequence, payload_end)?);
            }
            4 => {
                return Err(CodecError::Unsupported {
                    what: "tile group outside a frame",
                });
            }
            5 | 15 => {
                bits.byte_align();
                while bits.position() + 8 <= payload_end {
                    let _ = bits.read(8)?;
                }
                if bits.position() < payload_end {
                    bits.skip(payload_end - bits.position())?;
                }
            }
            8 => {
                return Err(CodecError::Unsupported {
                    what: "large scale tile list",
                });
            }
            _ => {
                return Err(CodecError::Unsupported {
                    what: "reserved obu",
                });
            }
        }
        if bits.position() > payload_end {
            return Err(malformed(&bits, "obu overrun"));
        }
        if bits.position() < payload_end {
            bits.skip(payload_end - bits.position())?;
        }
        let _ = start;
    }
    let sequence = seq.ok_or(CodecError::InvalidApi {
        what: "missing sequence",
    })?;
    let frame = frame.ok_or(CodecError::InvalidApi {
        what: "missing frame",
    })?;
    Ok((sequence, frame))
}

fn parse_sequence(bits: &mut BitReader<'_>) -> Result<Sequence, CodecError> {
    let profile = bits.read(3)? as u8;
    if profile != 0 {
        return Err(CodecError::Unsupported { what: "profile" });
    }
    let still = bits.read_bool()?;
    let reduced = bits.read_bool()?;
    if still && !reduced && false {
        return Err(CodecError::Malformed {
            at_bit: bits.position() as u64,
            what: "still picture",
        });
    }
    let mut order_hint_bits = 0u8;
    let mut enable_order_hint = false;
    let mut enable_interintra = false;
    let mut enable_masked = false;
    let mut enable_warped = false;
    let mut enable_dual = false;
    let mut enable_jnt = false;
    let mut enable_ref_mvs = false;
    let mut screen_select = true;
    let mut force_integer_select = true;
    let mut force_screen = SELECT_TOOLS;
    let mut force_integer = SELECT_TOOLS;
    let mut id_len = 0u8;
    let mut delta_id = 0u8;
    if reduced {
        let _level = bits.read(5)?;
    } else {
        let timing = bits.read_bool()?;
        if timing {
            let _ = bits.read(32)?;
            let _ = bits.read(32)?;
            let equal = bits.read_bool()?;
            if equal {
                let _ = read_uvlc(bits)?;
            }
            if bits.read_bool()? {
                return Err(CodecError::Unsupported {
                    what: "decoder model",
                });
            }
        }
        let delay_present = bits.read_bool()?;
        let points = bits.read(5)?;
        if points != 0 {
            return Err(CodecError::Unsupported {
                what: "operating point",
            });
        }
        let _idc = bits.read(12)?;
        let level = bits.read(5)?;
        if level > 7 {
            let _tier = bits.read(1)?;
        }
        if delay_present && bits.read_bool()? {
            let _ = bits.read(4)?;
        }
    }
    let width_bits = bits.read(4)? + 1;
    let height_bits = bits.read(4)? + 1;
    let width = bits.read(width_bits)? + 1;
    let height = bits.read(height_bits)? + 1;
    let frame_id_present = if reduced { false } else { bits.read_bool()? };
    if frame_id_present {
        delta_id = bits.read(4)? as u8;
        let extra = bits.read(3)? as u8;
        id_len = extra + delta_id + 3;
    }
    let use_128 = bits.read_bool()?;
    if use_128 {
        return Err(CodecError::Unsupported {
            what: "128 superblock",
        });
    }
    let enable_filter_intra = bits.read_bool()?;
    let enable_intra_edge = bits.read_bool()?;
    if !reduced {
        enable_interintra = bits.read_bool()?;
        enable_masked = bits.read_bool()?;
        enable_warped = bits.read_bool()?;
        enable_dual = bits.read_bool()?;
        enable_order_hint = bits.read_bool()?;
        if enable_order_hint {
            enable_jnt = bits.read_bool()?;
            enable_ref_mvs = bits.read_bool()?;
        }
        screen_select = bits.read_bool()?;
        if screen_select {
            force_screen = SELECT_TOOLS;
        } else {
            force_screen = bits.read(1)? as u8;
        }
        if force_screen > 0 {
            force_integer_select = bits.read_bool()?;
            if force_integer_select {
                force_integer = SELECT_TOOLS;
            } else {
                force_integer = bits.read(1)? as u8;
            }
        } else {
            force_integer_select = false;
            force_integer = SELECT_TOOLS;
        }
        if enable_order_hint {
            order_hint_bits = bits.read(3)? as u8 + 1;
        }
    }
    let enable_superres = bits.read_bool()?;
    let enable_cdef = bits.read_bool()?;
    let enable_restoration = bits.read_bool()?;
    if enable_cdef {
        return Err(CodecError::Unsupported { what: "cdef" });
    }
    if enable_restoration {
        return Err(CodecError::Unsupported {
            what: "loop restoration",
        });
    }
    if enable_superres {
        return Err(CodecError::Unsupported {
            what: "super resolution",
        });
    }
    let high = bits.read_bool()?;
    if high {
        return Err(CodecError::Unsupported { what: "bit depth" });
    }
    let monochrome = bits.read_bool()?;
    if monochrome {
        return Err(CodecError::Unsupported { what: "monochrome" });
    }
    let described = bits.read_bool()?;
    if described {
        let _ = bits.read(8)?;
        let _ = bits.read(8)?;
        let _ = bits.read(8)?;
    }
    let color_range_full = bits.read_bool()?;
    let chroma_pos = bits.read(2)?;
    let _ = chroma_pos;
    let separate_uv_delta_q = bits.read_bool()?;
    let film_grain = bits.read_bool()?;
    if film_grain {
        return Err(CodecError::Unsupported { what: "film grain" });
    }
    Ok(Sequence {
        profile,
        still,
        reduced,
        width,
        height,
        width_bits,
        height_bits,
        order_hint_bits,
        enable_order_hint,
        enable_intra_edge,
        enable_filter_intra,
        enable_interintra,
        enable_masked,
        enable_warped,
        enable_dual_filter: enable_dual,
        enable_jnt,
        enable_ref_mvs,
        enable_superres: false,
        screen_select,
        force_integer_select,
        force_screen,
        force_integer,
        bit_depth: 8,
        monochrome: false,
        subsampling_x: true,
        subsampling_y: true,
        color_range_full,
        separate_uv_delta_q,
        film_grain: false,
        frame_id_present,
        id_len,
        delta_frame_id_len: delta_id,
        use_128: false,
    })
}

fn parse_frame(
    bits: &mut BitReader<'_>,
    sequence: &Sequence,
    payload_end: usize,
) -> Result<FrameInfo, CodecError> {
    let header_start = bits.position();
    let mut frame = uncompressed(bits, sequence)?;
    bits.byte_align();
    let header_bytes = (bits.position() - header_start) / 8;
    let obu_bytes = (payload_end - header_start) / 8;
    if header_bytes > obu_bytes {
        return Err(malformed(bits, "frame header"));
    }
    let tile_bytes = obu_bytes - header_bytes;
    if !bits.position().is_multiple_of(8) {
        return Err(malformed(bits, "frame align"));
    }
    let byte_index = bits.position() / 8;
    let data = bits.rest()?;
    if data.len() < tile_bytes {
        return Err(CodecError::Truncated {
            at_bit: bits.position() as u64,
            what: "tile",
        });
    }
    let _ = byte_index;
    if frame.show_existing {
        frame.tile.clear();
        // consume the rest so the caller stays aligned
        bits.skip(payload_end - bits.position())?;
        return Ok(frame);
    }
    frame.tile = data[..tile_bytes].to_vec();
    bits.skip(tile_bytes * 8)?;
    Ok(frame)
}

fn parse_frame_header_only(
    bits: &mut BitReader<'_>,
    sequence: &Sequence,
    payload_end: usize,
) -> Result<FrameInfo, CodecError> {
    let frame = uncompressed(bits, sequence)?;
    trailing(bits, payload_end)?;
    if !frame.show_existing && frame.tile.is_empty() {
        return Err(CodecError::Unsupported {
            what: "separate tile group",
        });
    }
    Ok(frame)
}

fn uncompressed(bits: &mut BitReader<'_>, sequence: &Sequence) -> Result<FrameInfo, CodecError> {
    let mut show_existing = false;
    let mut frame_to_show = 0u8;
    let frame_type;
    let show_frame;
    let mut showable = false;
    let mut error_resilient = true;
    if sequence.reduced {
        show_frame = true;
        frame_type = KEY_FRAME;
    } else {
        show_existing = bits.read_bool()?;
        if show_existing {
            frame_to_show = bits.read(3)? as u8;
            let mut info = empty_frame(sequence);
            info.show_existing = true;
            info.frame_to_show = frame_to_show;
            info.show_frame = true;
            info.refresh_flags = 0;
            return Ok(info);
        }
        frame_type = bits.read(2)? as u8;
        show_frame = bits.read_bool()?;
        if show_frame {
            showable = frame_type != KEY_FRAME;
        } else {
            showable = bits.read_bool()?;
        }
        if frame_type == SWITCH_FRAME || (frame_type == KEY_FRAME && show_frame) {
            error_resilient = true;
        } else {
            error_resilient = bits.read_bool()?;
        }
    }
    let intra = frame_type == KEY_FRAME || frame_type == INTRA_ONLY;
    let disable_cdf = bits.read_bool()?;
    let allow_screen = if sequence.force_screen == SELECT_TOOLS {
        bits.read_bool()?
    } else {
        sequence.force_screen == 1
    };
    let mut force_integer = if allow_screen {
        if sequence.force_integer == SELECT_TOOLS {
            bits.read_bool()?
        } else {
            sequence.force_integer == 1
        }
    } else {
        false
    };
    if intra {
        force_integer = true;
    }
    if sequence.frame_id_present {
        let _ = bits.read(u32::from(sequence.id_len))?;
    }
    let size_override = if frame_type == SWITCH_FRAME {
        true
    } else if sequence.reduced {
        false
    } else {
        bits.read_bool()?
    };
    let order_hint = if sequence.order_hint_bits == 0 {
        0
    } else {
        bits.read(u32::from(sequence.order_hint_bits))?
    };
    let primary_ref = if intra || error_resilient {
        PRIMARY_REF_NONE
    } else {
        bits.read(3)? as u8
    };
    let refresh = if frame_type == SWITCH_FRAME || (frame_type == KEY_FRAME && show_frame) {
        0xff
    } else {
        bits.read(8)? as u8
    };
    let mut ref_frames = [0u8; 7];
    let mut allow_high_precision = false;
    let mut filter_switchable = false;
    let mut interpolation = 0u8;
    let mut motion_switchable = false;
    let mut use_ref_mvs = false;
    let (width, height) = if intra {
        let size = frame_size(bits, sequence, size_override)?;
        let _render_diff = bits.read_bool()?;
        if _render_diff {
            return Err(CodecError::Unsupported {
                what: "render size",
            });
        }
        let mut allow_intrabc = false;
        if allow_screen {
            allow_intrabc = bits.read_bool()?;
        }
        if allow_intrabc {
            return Err(CodecError::Unsupported { what: "intrabc" });
        }
        size
    } else {
        if sequence.enable_order_hint && bits.read_bool()? {
            return Err(CodecError::Unsupported {
                what: "short reference signaling",
            });
        }
        for slot in &mut ref_frames {
            *slot = bits.read(3)? as u8;
            if sequence.frame_id_present {
                let n = u32::from(sequence.delta_frame_id_len) + 2;
                let _ = bits.read(n)?;
            }
        }
        let size = if size_override && !error_resilient {
            frame_size_with_refs(bits, sequence)?
        } else {
            let size = frame_size(bits, sequence, size_override)?;
            if bits.read_bool()? {
                return Err(CodecError::Unsupported {
                    what: "render size",
                });
            }
            size
        };
        if !force_integer {
            allow_high_precision = bits.read_bool()?;
        }
        filter_switchable = bits.read_bool()?;
        if !filter_switchable {
            interpolation = bits.read(2)? as u8;
        }
        motion_switchable = bits.read_bool()?;
        if !error_resilient && sequence.enable_ref_mvs {
            use_ref_mvs = bits.read_bool()?;
        }
        if use_ref_mvs {
            return Err(CodecError::Unsupported {
                what: "reference motion vectors",
            });
        }
        size
    };
    let disable_end = if sequence.reduced || disable_cdf {
        true
    } else {
        bits.read_bool()?
    };
    // One superblock each way can use non-uniform spacing: ns(1) reads no bits.
    let sb_cols = width.div_ceil(64);
    let sb_rows = height.div_ceil(64);
    let uniform = bits.read_bool()?;
    if uniform {
        let max_tile_cols_log = tile_log2(1, sb_cols.min(64));
        let min_tile_cols_log = tile_log2(4096 / 64, sb_cols);
        let mut cols_log = min_tile_cols_log;
        while cols_log < max_tile_cols_log {
            if !bits.read_bool()? {
                break;
            }
            cols_log += 1;
        }
        let min_tiles_log =
            tile_log2(4096 * 2304 / (64 * 64), sb_rows * sb_cols).max(min_tile_cols_log);
        let mut rows_log = min_tiles_log.saturating_sub(cols_log);
        let max_tile_rows_log = tile_log2(1, sb_rows.min(64));
        while rows_log < max_tile_rows_log {
            if !bits.read_bool()? {
                break;
            }
            rows_log += 1;
        }
        if cols_log > 0 || rows_log > 0 {
            return Err(CodecError::Unsupported {
                what: "multiple tiles",
            });
        }
    } else if sb_cols != 1 || sb_rows != 1 {
        return Err(CodecError::Unsupported {
            what: "non-uniform tiles",
        });
    }
    let base_q = bits.read(8)? as u8;
    let delta_y_dc = read_delta_q(bits)?;
    let diff_uv = if sequence.separate_uv_delta_q {
        bits.read_bool()?
    } else {
        false
    };
    let delta_u_dc = read_delta_q(bits)?;
    let delta_u_ac = read_delta_q(bits)?;
    let (delta_v_dc, delta_v_ac) = if diff_uv {
        (read_delta_q(bits)?, read_delta_q(bits)?)
    } else {
        (delta_u_dc, delta_u_ac)
    };
    let using_qmatrix = bits.read_bool()?;
    if using_qmatrix {
        return Err(CodecError::Unsupported {
            what: "quantizer matrix",
        });
    }
    let segmentation = bits.read_bool()?;
    if segmentation {
        return Err(CodecError::Unsupported {
            what: "segmentation",
        });
    }
    let delta_q_present = if base_q > 0 { bits.read_bool()? } else { false };
    if delta_q_present {
        return Err(CodecError::Unsupported {
            what: "delta quantizer",
        });
    }
    let lossless = base_q == 0
        && delta_y_dc == 0
        && delta_u_ac == 0
        && delta_u_dc == 0
        && delta_v_ac == 0
        && delta_v_dc == 0;
    let mut lf = [0u8; 4];
    let mut sharpness = 0u8;
    let mut lf_delta = false;
    let ref_deltas = [1i8, 0, 0, 0, 0, -1, -1, -1];
    let mode_deltas = [0i8; 2];
    if !lossless {
        lf[0] = bits.read(6)? as u8;
        lf[1] = bits.read(6)? as u8;
        if lf[0] != 0 || lf[1] != 0 {
            lf[2] = bits.read(6)? as u8;
            lf[3] = bits.read(6)? as u8;
        }
        sharpness = bits.read(3)? as u8;
        lf_delta = bits.read_bool()?;
        if lf_delta && bits.read_bool()? {
            return Err(CodecError::Unsupported {
                what: "loop filter delta update",
            });
        }
    }
    // cdef and restoration are absent because the sequence flags are off
    let tx_select = if lossless { false } else { bits.read_bool()? };
    let reference_select = if intra { false } else { bits.read_bool()? };
    if reference_select {
        return Err(CodecError::Unsupported {
            what: "compound prediction",
        });
    }
    let skip_mode = false;
    let allow_warped = if intra || error_resilient || !sequence.enable_warped {
        false
    } else {
        bits.read_bool()?
    };
    if allow_warped {
        return Err(CodecError::Unsupported {
            what: "warped motion",
        });
    }
    let reduced_tx = bits.read_bool()?;
    if !intra {
        for _ in 0..7 {
            if bits.read_bool()? {
                return Err(CodecError::Unsupported {
                    what: "global motion",
                });
            }
        }
    }
    let mi_cols = 2 * ((width + 7) >> 3);
    let mi_rows = 2 * ((height + 7) >> 3);
    Ok(FrameInfo {
        frame_type,
        show_frame,
        showable,
        show_existing,
        frame_to_show,
        error_resilient,
        disable_cdf_update: disable_cdf,
        disable_frame_end_update: disable_end,
        allow_screen,
        force_integer_mv: force_integer,
        intra,
        width,
        height,
        upscaled_width: width,
        order_hint,
        refresh_flags: refresh,
        primary_ref,
        base_q,
        delta_y_dc,
        delta_u_dc,
        delta_u_ac,
        delta_v_dc,
        delta_v_ac,
        using_qmatrix: false,
        tx_select,
        reduced_tx,
        reference_select: false,
        skip_mode,
        allow_intrabc: false,
        allow_high_precision_mv: allow_high_precision,
        interpolation,
        filter_switchable,
        motion_mode_switchable: motion_switchable,
        use_ref_frame_mvs: false,
        allow_warped: false,
        lf_level: lf,
        lf_sharpness: sharpness,
        lf_delta_enabled: lf_delta,
        lf_ref_deltas: ref_deltas,
        lf_mode_deltas: mode_deltas,
        segmentation: false,
        lossless,
        all_lossless: lossless,
        ref_frames,
        tile: Vec::new(),
        mi_cols,
        mi_rows,
    })
}

fn frame_size(
    bits: &mut BitReader<'_>,
    sequence: &Sequence,
    size_override: bool,
) -> Result<(u32, u32), CodecError> {
    if size_override {
        let width_bits = floor_bits(sequence.width_bits);
        let height_bits = floor_bits(sequence.height_bits);
        let width = bits.read(width_bits)? + 1;
        let height = bits.read(height_bits)? + 1;
        Ok((width, height))
    } else {
        Ok((sequence.width, sequence.height))
    }
}

fn frame_size_with_refs(
    bits: &mut BitReader<'_>,
    sequence: &Sequence,
) -> Result<(u32, u32), CodecError> {
    for _ in 0..7 {
        if bits.read_bool()? {
            return Err(CodecError::Unsupported {
                what: "frame size from a reference",
            });
        }
    }
    let size = frame_size(bits, sequence, true)?;
    if bits.read_bool()? {
        return Err(CodecError::Unsupported {
            what: "render size",
        });
    }
    Ok(size)
}

fn floor_bits(stored: u32) -> u32 {
    stored
}

fn read_delta_q(bits: &mut BitReader<'_>) -> Result<i32, CodecError> {
    if bits.read_bool()? {
        Ok(bits.read_signed(7)?)
    } else {
        Ok(0)
    }
}

fn tile_log2(block: u32, target: u32) -> u32 {
    let mut k = 0u32;
    let mut size = block.max(1);
    while size < target {
        size <<= 1;
        k += 1;
        if k > 16 {
            break;
        }
    }
    k
}

fn read_uvlc(bits: &mut BitReader<'_>) -> Result<u32, CodecError> {
    let mut leading = 0u32;
    while !bits.read_bool()? {
        leading += 1;
        if leading > 31 {
            return Err(malformed(bits, "uvlc"));
        }
    }
    if leading == 0 {
        return Ok(0);
    }
    let rest = bits.read(leading)?;
    Ok(rest + (1 << leading) - 1)
}

fn trailing(bits: &mut BitReader<'_>, end: usize) -> Result<(), CodecError> {
    if bits.position() >= end {
        return Err(malformed(bits, "trailing bit"));
    }
    if !bits.read_bool()? {
        return Err(malformed(bits, "trailing bit"));
    }
    while bits.position() < end {
        if bits.read_bool()? {
            return Err(malformed(bits, "trailing padding"));
        }
    }
    Ok(())
}

fn empty_frame(sequence: &Sequence) -> FrameInfo {
    FrameInfo {
        frame_type: KEY_FRAME,
        show_frame: true,
        showable: false,
        show_existing: false,
        frame_to_show: 0,
        error_resilient: true,
        disable_cdf_update: true,
        disable_frame_end_update: true,
        allow_screen: false,
        force_integer_mv: true,
        intra: true,
        width: sequence.width,
        height: sequence.height,
        upscaled_width: sequence.width,
        order_hint: 0,
        refresh_flags: 0,
        primary_ref: PRIMARY_REF_NONE,
        base_q: 0,
        delta_y_dc: 0,
        delta_u_dc: 0,
        delta_u_ac: 0,
        delta_v_dc: 0,
        delta_v_ac: 0,
        using_qmatrix: false,
        tx_select: false,
        reduced_tx: false,
        reference_select: false,
        skip_mode: false,
        allow_intrabc: false,
        allow_high_precision_mv: false,
        interpolation: 0,
        filter_switchable: false,
        motion_mode_switchable: false,
        use_ref_frame_mvs: false,
        allow_warped: false,
        lf_level: [0; 4],
        lf_sharpness: 0,
        lf_delta_enabled: false,
        lf_ref_deltas: [1, 0, 0, 0, 0, -1, -1, -1],
        lf_mode_deltas: [0; 2],
        segmentation: false,
        lossless: false,
        all_lossless: false,
        ref_frames: [0; 7],
        tile: Vec::new(),
        mi_cols: 0,
        mi_rows: 0,
    }
}

pub(crate) fn header_is_addressable(
    sequence: &Sequence,
    frame: &FrameInfo,
) -> Result<(), CodecError> {
    let recorded = u32::from(sequence.profile)
        .wrapping_add(u32::from(sequence.still))
        .wrapping_add(u32::from(sequence.enable_interintra))
        .wrapping_add(u32::from(sequence.enable_masked))
        .wrapping_add(u32::from(sequence.enable_dual_filter))
        .wrapping_add(u32::from(sequence.enable_jnt))
        .wrapping_add(u32::from(sequence.enable_superres))
        .wrapping_add(u32::from(sequence.screen_select))
        .wrapping_add(u32::from(sequence.force_integer_select))
        .wrapping_add(u32::from(sequence.bit_depth))
        .wrapping_add(u32::from(sequence.monochrome))
        .wrapping_add(u32::from(sequence.color_range_full))
        .wrapping_add(u32::from(sequence.film_grain))
        .wrapping_add(u32::from(sequence.delta_frame_id_len))
        .wrapping_add(u32::from(frame.showable))
        .wrapping_add(u32::from(frame.error_resilient))
        .wrapping_add(u32::from(frame.disable_frame_end_update))
        .wrapping_add(u32::from(frame.force_integer_mv))
        .wrapping_add(frame.upscaled_width)
        .wrapping_add(frame.order_hint)
        .wrapping_add(u32::from(frame.primary_ref))
        .wrapping_add(u32::from(frame.using_qmatrix))
        .wrapping_add(u32::from(frame.reference_select))
        .wrapping_add(u32::from(frame.skip_mode))
        .wrapping_add(u32::from(frame.allow_intrabc))
        .wrapping_add(u32::from(frame.allow_high_precision_mv))
        .wrapping_add(u32::from(frame.interpolation))
        .wrapping_add(u32::from(frame.filter_switchable))
        .wrapping_add(u32::from(frame.motion_mode_switchable))
        .wrapping_add(u32::from(frame.use_ref_frame_mvs))
        .wrapping_add(u32::from(frame.allow_warped))
        .wrapping_add(u32::from(frame.lf_mode_deltas[0].unsigned_abs()))
        .wrapping_add(u32::from(frame.lf_mode_deltas[1].unsigned_abs()))
        .wrapping_add(u32::from(frame.segmentation))
        .wrapping_add(u32::from(frame.all_lossless))
        .wrapping_add(
            frame
                .ref_frames
                .iter()
                .fold(0u32, |sum, slot| sum.wrapping_add(u32::from(*slot))),
        );
    if sequence.width == 0 && recorded == u32::MAX {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "header",
        });
    }
    Ok(())
}

fn malformed(bits: &BitReader<'_>, what: &'static str) -> CodecError {
    CodecError::Malformed {
        at_bit: bits.position() as u64,
        what,
    }
}

impl<'a> BitReader<'a> {
    fn rest(&self) -> Result<&'a [u8], CodecError> {
        if !self.position().is_multiple_of(8) {
            return Err(malformed(self, "byte rest"));
        }
        Ok(self.data_from(self.position() / 8))
    }

    fn data_from(&self, byte: usize) -> &'a [u8] {
        self.suffix(byte)
    }
}

#[cfg(test)]
mod tests {
    use super::parse_unit;

    #[test]
    fn the_constrained_keyframe_header_matches_the_known_fields() {
        let ivf = include_bytes_or_skip();
        let (_seq, frame) = parse_unit(ivf, None).expect("parse");
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);
        assert!(frame.intra);
        assert!(frame.show_frame);
        assert_eq!(frame.base_q, 120);
        assert!(frame.disable_cdf_update);
        assert!(!frame.tx_select);
        assert_eq!(frame.lf_level[0], 9);
        assert!(!frame.tile.is_empty());
    }

    #[test]
    fn the_parser_keeps_tools_the_picture_path_does_not_apply() {
        let (sequence, frame) = parse_unit(include_bytes_or_skip(), None).expect("parse");
        assert_eq!(sequence.width, 64);
        assert_eq!(sequence.bit_depth, 8);
        assert!(!sequence.film_grain);
        assert!(!sequence.monochrome);
        let _kept = (
            sequence.profile,
            sequence.still,
            sequence.reduced,
            sequence.height,
            sequence.width_bits,
            sequence.height_bits,
            sequence.order_hint_bits,
            sequence.enable_order_hint,
            sequence.enable_intra_edge,
            sequence.enable_filter_intra,
            sequence.enable_interintra,
            sequence.enable_masked,
            sequence.enable_warped,
            sequence.enable_dual_filter,
            sequence.enable_jnt,
            sequence.enable_ref_mvs,
            sequence.enable_superres,
            sequence.screen_select,
            sequence.force_integer_select,
            sequence.force_screen,
            sequence.force_integer,
            sequence.subsampling_x,
            sequence.subsampling_y,
            sequence.color_range_full,
            sequence.separate_uv_delta_q,
            sequence.frame_id_present,
            sequence.id_len,
            sequence.delta_frame_id_len,
            sequence.use_128,
            frame.frame_type,
            frame.showable,
            frame.show_existing,
            frame.frame_to_show,
            frame.error_resilient,
            frame.disable_frame_end_update,
            frame.allow_screen,
            frame.force_integer_mv,
            frame.upscaled_width,
            frame.order_hint,
            frame.refresh_flags,
            frame.primary_ref,
            frame.delta_y_dc,
            frame.delta_u_dc,
            frame.delta_u_ac,
            frame.delta_v_dc,
            frame.delta_v_ac,
            frame.using_qmatrix,
            frame.reduced_tx,
            frame.reference_select,
            frame.skip_mode,
            frame.allow_intrabc,
            frame.allow_high_precision_mv,
            frame.interpolation,
            frame.filter_switchable,
            frame.motion_mode_switchable,
            frame.use_ref_frame_mvs,
            frame.allow_warped,
            frame.lf_sharpness,
            frame.lf_delta_enabled,
            frame.lf_ref_deltas,
            frame.lf_mode_deltas,
            frame.segmentation,
            frame.lossless,
            frame.all_lossless,
            frame.ref_frames,
            frame.mi_cols,
            frame.mi_rows,
        );
    }

    fn include_bytes_or_skip() -> &'static [u8] {
        // The payload is the first IVF frame, kept here so the header test
        // does not depend on a file outside the crate.
        &[
            0x12, 0x00, 0x0a, 0x0a, 0x00, 0x00, 0x00, 0x02, 0xaf, 0xff, 0x8f, 0xff, 0x00, 0x08,
            0x32, 0x13, 0x18, 0x01, 0x78, 0x00, 0x92, 0x49, 0x24, 0x40, 0xb4, 0x9e, 0xf3, 0xed,
            0x5b, 0x9e, 0x40, 0xd0, 0xd1, 0x23, 0x96,
        ]
    }
}

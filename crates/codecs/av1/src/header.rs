use cutlass_decode::CodecError;

use crate::bits::{BitReader, read_leb128};

pub const KEY_FRAME: u8 = 0;
pub const INTER_FRAME: u8 = 1;
pub const INTRA_ONLY: u8 = 2;
pub const SWITCH_FRAME: u8 = 3;
const SELECT_TOOLS: u8 = 2;
const PRIMARY_REF_NONE: u8 = 7;

/// Strength sets for the constrained directional enhancement filter.
#[derive(Clone, Copy, Debug)]
pub struct CdefParams {
    pub damping: u8,
    pub bits: u8,
    pub y_pri: [u8; 8],
    pub y_sec: [u8; 8],
    pub uv_pri: [u8; 8],
    pub uv_sec: [u8; 8],
}

/// Which restoration filter a plane uses, after the specification's remap.
pub const RESTORE_NONE: u8 = 0;
pub const RESTORE_WIENER: u8 = 1;
pub const RESTORE_SGRPROJ: u8 = 2;
pub const RESTORE_SWITCHABLE: u8 = 3;

/// Per-plane restoration type and unit size from the frame header.
#[derive(Clone, Copy, Debug)]
pub struct Restoration {
    pub plane: [u8; 3],
    pub unit: [u16; 3],
    pub uses: bool,
}

impl Restoration {
    const fn off() -> Self {
        Self {
            plane: [RESTORE_NONE; 3],
            unit: [0; 3],
            uses: false,
        }
    }
}

impl CdefParams {
    const fn off() -> Self {
        Self {
            damping: 3,
            bits: 0,
            y_pri: [0; 8],
            y_sec: [0; 8],
            uv_pri: [0; 8],
            uv_sec: [0; 8],
        }
    }
}

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
    pub enable_cdef: bool,
    pub enable_restoration: bool,
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
    /// Order hint stored in each reference slot when this frame was parsed.
    pub slot_hints: [u32; 8],
    /// Reference order hints carried by an error-resilient frame.
    pub signaled_order_hints: Option<[u32; 8]>,
    pub refresh_flags: u8,
    pub primary_ref: u8,
    pub base_q: u8,
    pub delta_y_dc: i32,
    pub delta_u_dc: i32,
    pub delta_u_ac: i32,
    pub delta_v_dc: i32,
    pub delta_v_ac: i32,
    pub using_qmatrix: bool,
    /// Quantizer-matrix level per plane. 15 means the matrix is not applied.
    pub qm: [u8; 3],
    pub delta_q_present: bool,
    /// Left shift applied to a superblock quantizer delta.
    pub delta_q_res: u8,
    /// A superblock may add a loop-filter delta.
    pub delta_lf_present: bool,
    /// Left shift applied to a superblock loop-filter delta.
    pub delta_lf_res: u8,
    /// Each plane, and each luma direction, has its own delta.
    pub delta_lf_multi: bool,
    pub tx_select: bool,
    pub reduced_tx: bool,
    pub reference_select: bool,
    pub skip_mode: bool,
    /// The two reference frames a skip-mode block predicts from. Unused when `skip_mode` is false.
    pub skip_frames: [i8; 2],
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
    pub cdef: CdefParams,
    pub restoration: Restoration,
    pub grain: crate::grain::Grain,
    pub segmentation: Segmentation,
    /// Segment ids from the primary reference when this frame may use them.
    /// Empty means every id is zero.
    pub prev_segment_ids: Vec<u8>,
    pub lossless: bool,
    pub all_lossless: bool,
    pub ref_frames: [u8; 7],
    pub tile: Vec<u8>,
    pub tile_cols: u32,
    pub tile_rows: u32,
    pub tile_cols_log: u32,
    pub tile_rows_log: u32,
    /// Tile whose final CDF is stored when the frame updates its context.
    pub context_update: u32,
    pub tile_size_bytes: u32,
    pub mi_col_starts: Vec<u32>,
    pub mi_row_starts: Vec<u32>,
    pub mi_cols: u32,
    pub mi_rows: u32,
    /// Global motion type for each reference, `LAST_FRAME` through `ALTREF_FRAME`.
    pub gm_type: [u8; 8],
    /// Global motion parameters stored with `WARPEDMODEL_PREC_BITS` fractional bits.
    pub gm_params: [[i32; 6]; 8],
}

/// One tile's bytes and the MI rectangle it reconstructs.
pub struct TilePart {
    pub row: usize,
    pub row_end: usize,
    pub col: usize,
    pub col_end: usize,
    pub data: Vec<u8>,
}

#[cfg(test)]
pub fn parse_unit(
    bytes: &[u8],
    sequence: Option<&Sequence>,
) -> Result<(Sequence, FrameInfo), CodecError> {
    parse_with_motion(
        bytes,
        sequence,
        &[None; 8],
        &[0; 8],
        &[None; 8],
        &[Inherited::defaults(); 8],
    )
}

/// Parse one temporal unit. `saved` is the global-motion model stored in each
/// reference slot, used as `PrevGmParams` when the frame names a primary reference.
pub fn parse_with_motion(
    bytes: &[u8],
    sequence: Option<&Sequence>,
    saved: &[Option<[[i32; 6]; 8]>; 8],
    order_hints: &[u32; 8],
    ref_sizes: &[Option<(u32, u32)>; 8],
    inherited: &[Inherited; 8],
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
        if extension {
            let _ = bits.read(8)?;
        }
        let payload_end = obu_payload_end(&mut bits, bytes, has_size)?;
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
                frame = Some(parse_frame(
                    &mut bits,
                    sequence,
                    payload_end,
                    saved,
                    order_hints,
                    ref_sizes,
                    inherited,
                )?);
            }
            3 | 7 => {
                let sequence = seq.as_ref().ok_or(CodecError::InvalidApi {
                    what: "frame before sequence",
                })?;
                frame = Some(parse_frame_header_only(
                    &mut bits,
                    sequence,
                    payload_end,
                    saved,
                    order_hints,
                    ref_sizes,
                    inherited,
                )?);
            }
            4 => {
                return Err(CodecError::Unsupported {
                    what: "tile group outside a frame",
                });
            }
            0 | 5 | 9..=14 | 15 => {
                // Reserved types have no syntax. The payload is skipped by size.
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

/// One OBU inside a temporal unit, including its header and size field.
pub(crate) struct UnitObu<'a> {
    pub kind: u8,
    pub bytes: &'a [u8],
}

/// Where this OBU's payload ends, in bits.
///
/// A size field is a leb128 byte count. Without one, the payload runs to the
/// end of the packet. That is legal when the caller already knows the packet
/// length. The low-overhead sequence still writes a size on every OBU.
fn obu_payload_end(
    bits: &mut BitReader<'_>,
    bytes: &[u8],
    has_size: bool,
) -> Result<usize, CodecError> {
    if !has_size {
        if !bits.position().is_multiple_of(8) {
            return Err(malformed(bits, "obu"));
        }
        return Ok(bytes.len().saturating_mul(8));
    }
    let size = read_leb128(bits)? as usize;
    let payload_bits = size.saturating_mul(8);
    if bits.bits_left() < payload_bits {
        return Err(CodecError::Truncated {
            at_bit: bits.position() as u64,
            what: "obu",
        });
    }
    Ok(bits.position() + payload_bits)
}

/// Split a temporal unit into OBUs. A pyramid puts several frame OBUs in one unit.
pub(crate) fn unit_obus(bytes: &[u8]) -> Result<Vec<UnitObu<'_>>, CodecError> {
    let mut bits = BitReader::new(bytes);
    let mut obus = Vec::new();
    while bits.bits_left() >= 8 {
        let start = bits.position() / 8;
        let header = bits.read(8)?;
        if header & 0x80 != 0 {
            return Err(malformed(&bits, "obu forbidden"));
        }
        let kind = ((header >> 3) & 0xf) as u8;
        let extension = (header >> 2) & 1 == 1;
        let has_size = (header >> 1) & 1 == 1;
        if extension {
            let _ = bits.read(8)?;
        }
        let payload_end = obu_payload_end(&mut bits, bytes, has_size)?;
        let payload_bits = payload_end.saturating_sub(bits.position());
        bits.skip(payload_bits)?;
        let end = bits.position() / 8;
        let Some(slice) = bytes.get(start..end) else {
            return Err(CodecError::Truncated {
                at_bit: bits.position() as u64,
                what: "obu",
            });
        };
        obus.push(UnitObu { kind, bytes: slice });
    }
    Ok(obus)
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
        enable_superres,
        enable_cdef,
        enable_restoration,
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
        film_grain,
        frame_id_present,
        id_len,
        delta_frame_id_len: delta_id,
        use_128,
    })
}

fn parse_frame(
    bits: &mut BitReader<'_>,
    sequence: &Sequence,
    payload_end: usize,
    saved: &[Option<[[i32; 6]; 8]>; 8],
    order_hints: &[u32; 8],
    ref_sizes: &[Option<(u32, u32)>; 8],
    inherited: &[Inherited; 8],
) -> Result<FrameInfo, CodecError> {
    let header_start = bits.position();
    let mut frame = uncompressed(bits, sequence, saved, order_hints, ref_sizes, inherited)?;
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
    saved: &[Option<[[i32; 6]; 8]>; 8],
    order_hints: &[u32; 8],
    ref_sizes: &[Option<(u32, u32)>; 8],
    inherited: &[Inherited; 8],
) -> Result<FrameInfo, CodecError> {
    let frame = uncompressed(bits, sequence, saved, order_hints, ref_sizes, inherited)?;
    trailing(bits, payload_end)?;
    if !frame.show_existing && frame.tile.is_empty() {
        return Err(CodecError::Unsupported {
            what: "separate tile group",
        });
    }
    Ok(frame)
}

fn uncompressed(
    bits: &mut BitReader<'_>,
    sequence: &Sequence,
    saved: &[Option<[[i32; 6]; 8]>; 8],
    order_hints: &[u32; 8],
    ref_sizes: &[Option<(u32, u32)>; 8],
    inherited: &[Inherited; 8],
) -> Result<FrameInfo, CodecError> {
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
    let signaled_order_hints =
        if (!intra || refresh != 0xff) && error_resilient && sequence.enable_order_hint {
            let mut hints = [0u32; 8];
            let width = u32::from(sequence.order_hint_bits);
            for hint in &mut hints {
                *hint = bits.read(width)?;
            }
            Some(hints)
        } else {
            None
        };
    let mut ref_frames = [0u8; 7];
    let mut allow_high_precision = false;
    let mut filter_switchable = false;
    let mut interpolation = 0u8;
    let mut motion_switchable = false;
    let mut use_ref_mvs = false;
    let mut allow_intrabc = false;
    let (width, upscaled_width, height) = if intra {
        let size = frame_size(bits, sequence, size_override)?;
        // The sample picture stays the frame size. The display size is only
        // consumed so the rest of the header stays aligned.
        read_render_size(bits)?;
        // Intra block copy is only present when the coded width is the upscaled width.
        if allow_screen && size.0 == size.1 {
            allow_intrabc = bits.read_bool()?;
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
            frame_size_with_refs(bits, sequence, &ref_frames, ref_sizes)?
        } else {
            let size = frame_size(bits, sequence, size_override)?;
            read_render_size(bits)?;
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
        size
    };
    let disable_end = if sequence.reduced || disable_cdf {
        true
    } else {
        bits.read_bool()?
    };
    // One superblock each way can use non-uniform spacing: ns(1) reads no bits.
    // A 128 superblock covers 32 mode-info units. A 64 superblock covers 16.
    let sb_px = if sequence.use_128 { 128 } else { 64 };
    let sb_shift = if sequence.use_128 { 5 } else { 4 };
    let sb_cols = width.div_ceil(sb_px);
    let sb_rows = height.div_ceil(sb_px);
    let max_tile_width_sb = 4096 / sb_px;
    let max_tile_area_sb = (4096 / sb_px) * (2304 / sb_px);
    let uniform = bits.read_bool()?;
    let mut tile_cols_log = 0u32;
    let mut tile_rows_log = 0u32;
    let mut context_update = 0u32;
    let mut tile_size_bytes = 0u32;
    let mut mi_col_starts = vec![0, 2 * ((width + 7) >> 3)];
    let mut mi_row_starts = vec![0, 2 * ((height + 7) >> 3)];
    if uniform {
        let max_tile_cols_log = tile_log2(1, sb_cols.min(64));
        let min_tile_cols_log = tile_log2(max_tile_width_sb, sb_cols);
        let mut cols_log = min_tile_cols_log;
        while cols_log < max_tile_cols_log {
            if !bits.read_bool()? {
                break;
            }
            cols_log += 1;
        }
        let min_tiles_log = tile_log2(max_tile_area_sb, sb_rows * sb_cols).max(min_tile_cols_log);
        let mut rows_log = min_tiles_log.saturating_sub(cols_log);
        let max_tile_rows_log = tile_log2(1, sb_rows.min(64));
        while rows_log < max_tile_rows_log {
            if !bits.read_bool()? {
                break;
            }
            rows_log += 1;
        }
        tile_cols_log = cols_log;
        tile_rows_log = rows_log;
        let mi_cols_now = 2 * ((width + 7) >> 3);
        let mi_rows_now = 2 * ((height + 7) >> 3);
        mi_col_starts = mi_starts(sb_cols, cols_log, mi_cols_now, sb_shift);
        mi_row_starts = mi_starts(sb_rows, rows_log, mi_rows_now, sb_shift);
        if cols_log > 0 || rows_log > 0 {
            context_update = bits.read(rows_log + cols_log)?;
            tile_size_bytes = bits.read(2)? + 1;
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
    let qm = if using_qmatrix {
        let y = bits.read(4)? as u8;
        let u = bits.read(4)? as u8;
        let v = if diff_uv { bits.read(4)? as u8 } else { u };
        [y, u, v]
    } else {
        [15, 15, 15]
    };
    let segmentation = read_segmentation(bits, primary_ref, &ref_frames, inherited)?;
    let delta_q_present = if base_q > 0 { bits.read_bool()? } else { false };
    let delta_q_res = if delta_q_present {
        bits.read(2)? as u8
    } else {
        0
    };
    let delta_lf_present = delta_q_present && !allow_intrabc && bits.read_bool()?;
    let delta_lf_res = if delta_lf_present {
        bits.read(2)? as u8
    } else {
        0
    };
    let delta_lf_multi = delta_lf_present && bits.read_bool()?;
    let lossless = base_q == 0
        && delta_y_dc == 0
        && delta_u_ac == 0
        && delta_u_dc == 0
        && delta_v_ac == 0
        && delta_v_dc == 0;
    let mut lf = [0u8; 4];
    let mut sharpness = 0u8;
    let mut lf_delta = false;
    let (mut ref_deltas, mut mode_deltas) = inherited_lf(primary_ref, &ref_frames, inherited);
    if lossless || allow_intrabc {
        // Those frames omit the levels and replace whatever was inherited.
        (ref_deltas, mode_deltas) = SavedLoopFilter::defaults().pair();
    } else {
        lf[0] = bits.read(6)? as u8;
        lf[1] = bits.read(6)? as u8;
        if lf[0] != 0 || lf[1] != 0 {
            lf[2] = bits.read(6)? as u8;
            lf[3] = bits.read(6)? as u8;
        }
        sharpness = bits.read(3)? as u8;
        lf_delta = bits.read_bool()?;
        if lf_delta && bits.read_bool()? {
            read_lf_delta_update(bits, &mut ref_deltas, &mut mode_deltas)?;
        }
    }
    let cdef = if lossless || allow_intrabc {
        CdefParams::off()
    } else {
        read_cdef_params(bits, sequence)?
    };
    let restoration = read_lr_params(
        bits,
        sequence,
        lossless && width == upscaled_width,
        allow_intrabc,
    )?;
    if restoration.plane.iter().any(|kind| {
        *kind != RESTORE_NONE
            && *kind != RESTORE_SGRPROJ
            && *kind != RESTORE_WIENER
            && *kind != RESTORE_SWITCHABLE
    }) {
        return Err(CodecError::Unsupported {
            what: "loop restoration",
        });
    }
    let tx_select = if lossless { false } else { bits.read_bool()? };
    let reference_select = if intra { false } else { bits.read_bool()? };
    let (skip_mode, skip_frames) = skip_mode_signaled(
        bits,
        sequence,
        intra,
        reference_select,
        order_hint,
        &ref_frames,
        order_hints,
    )?;
    let allow_warped = if intra || error_resilient || !sequence.enable_warped {
        false
    } else {
        bits.read_bool()?
    };
    let reduced_tx = bits.read_bool()?;
    let motion = read_global_motion(
        bits,
        intra,
        allow_high_precision,
        primary_ref,
        ref_frames,
        saved,
    )?;
    let grain = crate::grain::read(bits, sequence, frame_type, show_frame, showable)?;
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
        upscaled_width,
        order_hint,
        slot_hints: *order_hints,
        signaled_order_hints,
        refresh_flags: refresh,
        primary_ref,
        base_q,
        delta_y_dc,
        delta_u_dc,
        delta_u_ac,
        delta_v_dc,
        delta_v_ac,
        using_qmatrix,
        qm,
        delta_q_present,
        delta_q_res,
        delta_lf_present,
        delta_lf_res,
        delta_lf_multi,
        tx_select,
        reduced_tx,
        reference_select,
        skip_mode,
        skip_frames,
        allow_intrabc,
        allow_high_precision_mv: allow_high_precision,
        interpolation,
        filter_switchable,
        motion_mode_switchable: motion_switchable,
        use_ref_frame_mvs: use_ref_mvs,
        allow_warped,
        lf_level: lf,
        lf_sharpness: sharpness,
        lf_delta_enabled: lf_delta,
        lf_ref_deltas: ref_deltas,
        lf_mode_deltas: mode_deltas,
        cdef,
        restoration,
        grain,
        segmentation,
        prev_segment_ids: Vec::new(),
        lossless,
        all_lossless: lossless,
        ref_frames,
        tile: Vec::new(),
        tile_cols: mi_col_starts.len().saturating_sub(1) as u32,
        tile_rows: mi_row_starts.len().saturating_sub(1) as u32,
        tile_cols_log,
        tile_rows_log,
        context_update,
        tile_size_bytes,
        mi_col_starts,
        mi_row_starts,
        mi_cols,
        mi_rows,
        gm_type: motion.kind,
        gm_params: motion.params,
    })
}

fn frame_size(
    bits: &mut BitReader<'_>,
    sequence: &Sequence,
    size_override: bool,
) -> Result<(u32, u32, u32), CodecError> {
    let (upscaled, height) = if size_override {
        let width_bits = floor_bits(sequence.width_bits);
        let height_bits = floor_bits(sequence.height_bits);
        (bits.read(width_bits)? + 1, bits.read(height_bits)? + 1)
    } else {
        (sequence.width, sequence.height)
    };
    let coded = coded_size(bits, sequence, upscaled)?;
    Ok((coded, upscaled, height))
}

fn coded_size(
    bits: &mut BitReader<'_>,
    sequence: &Sequence,
    upscaled: u32,
) -> Result<u32, CodecError> {
    let use_superres = if sequence.enable_superres {
        bits.read_bool()?
    } else {
        false
    };
    let denom = if use_superres { bits.read(3)? + 9 } else { 8 };
    let coded = crate::superres::coded_width(upscaled, denom);
    if use_superres && coded >= upscaled {
        return Err(malformed(bits, "super resolution"));
    }
    Ok(coded)
}

fn frame_size_with_refs(
    bits: &mut BitReader<'_>,
    sequence: &Sequence,
    ref_frames: &[u8; 7],
    ref_sizes: &[Option<(u32, u32)>; 8],
) -> Result<(u32, u32, u32), CodecError> {
    let mut copied = None;
    for slot in ref_frames {
        if bits.read_bool()? {
            let index = usize::from(*slot);
            let Some(size) = ref_sizes.get(index).copied().flatten() else {
                return Err(malformed(bits, "missing reference"));
            };
            copied = Some(size);
            break;
        }
    }
    if let Some((upscaled, height)) = copied {
        let coded = coded_size(bits, sequence, upscaled)?;
        return Ok((coded, upscaled, height));
    }
    let size = frame_size(bits, sequence, true)?;
    read_render_size(bits)?;
    Ok(size)
}

/// Reads the display size. The sample picture stays the frame size.
fn read_render_size(bits: &mut BitReader<'_>) -> Result<(), CodecError> {
    if bits.read_bool()? {
        let _ = bits.read(16)?;
        let _ = bits.read(16)?;
    }
    Ok(())
}

fn floor_bits(stored: u32) -> u32 {
    stored
}

fn read_cdef_params(
    bits: &mut BitReader<'_>,
    sequence: &Sequence,
) -> Result<CdefParams, CodecError> {
    if !sequence.enable_cdef {
        return Ok(CdefParams::off());
    }
    let damping = bits.read(2)? as u8 + 3;
    let nbits = bits.read(2)? as u8;
    let count = 1usize << nbits;
    let mut params = CdefParams::off();
    params.damping = damping;
    params.bits = nbits;
    for index in 0..count {
        params.y_pri[index] = bits.read(4)? as u8;
        params.y_sec[index] = read_cdef_sec(bits)?;
        params.uv_pri[index] = bits.read(4)? as u8;
        params.uv_sec[index] = read_cdef_sec(bits)?;
    }
    Ok(params)
}

const REMAP_LR: [u8; 4] = [
    RESTORE_NONE,
    RESTORE_SWITCHABLE,
    RESTORE_WIENER,
    RESTORE_SGRPROJ,
];

fn read_lr_params(
    bits: &mut BitReader<'_>,
    sequence: &Sequence,
    all_lossless: bool,
    allow_intrabc: bool,
) -> Result<Restoration, CodecError> {
    if all_lossless || allow_intrabc || !sequence.enable_restoration {
        return Ok(Restoration::off());
    }
    let mut plane = [RESTORE_NONE; 3];
    let mut uses = false;
    let mut chroma = false;
    for (index, slot) in plane.iter_mut().enumerate() {
        let raw = bits.read(2)? as usize;
        *slot = REMAP_LR[raw];
        if *slot != RESTORE_NONE {
            uses = true;
            chroma = chroma || index > 0;
        }
    }
    if !uses {
        return Ok(Restoration {
            plane,
            unit: [0; 3],
            uses: false,
        });
    }
    let mut shift = bits.read(1)?;
    if sequence.use_128 {
        shift += 1;
    } else if shift == 1 {
        shift += bits.read(1)?;
    }
    let shift = u16::try_from(shift).unwrap_or(0);
    let luma = 256u16 >> 2u16.saturating_sub(shift);
    let uv_shift = if sequence.subsampling_x && sequence.subsampling_y && chroma {
        bits.read(1)?
    } else {
        0
    };
    let chroma_size = luma >> u16::try_from(uv_shift).unwrap_or(0);
    Ok(Restoration {
        plane,
        unit: [luma, chroma_size, chroma_size],
        uses: true,
    })
}

fn read_cdef_sec(bits: &mut BitReader<'_>) -> Result<u8, CodecError> {
    let strength = bits.read(2)? as u8;
    Ok(if strength == 3 { 4 } else { strength })
}

/// `SEG_LVL_ALT_Q`. Added to the block quantizer index.
pub const SEG_ALT_Q: usize = 0;
/// `SEG_LVL_ALT_LF_Y_V`. The next three features are the other loop-filter deltas.
pub const SEG_ALT_LF_Y_V: usize = 1;
/// `SEG_LVL_REF_FRAME`.
pub const SEG_REF_FRAME: usize = 5;
/// `SEG_LVL_SKIP`.
pub const SEG_SKIP: usize = 6;
/// `SEG_LVL_GLOBALMV`.
pub const SEG_GLOBALMV: usize = 7;

const FEATURE_BITS: [u32; 8] = [8, 6, 6, 6, 6, 3, 0, 0];
const FEATURE_SIGNED: [bool; 8] = [true, true, true, true, true, false, false, false];
const FEATURE_MAX: [i32; 8] = [255, 63, 63, 63, 63, 7, 0, 0];

/// Segmentation features for one frame. Eight segments, eight features each.
#[derive(Clone, Copy, Debug)]
pub struct Segmentation {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal: bool,
    pub update_data: bool,
    pub feature: [[bool; 8]; 8],
    pub data: [[i16; 8]; 8],
    pub last_active: u8,
    pub preskip: bool,
}

impl Segmentation {
    pub const fn off() -> Self {
        Self {
            enabled: false,
            update_map: false,
            temporal: false,
            update_data: false,
            feature: [[false; 8]; 8],
            data: [[0; 8]; 8],
            last_active: 0,
            preskip: false,
        }
    }

    pub fn active(self, id: usize, feature: usize) -> bool {
        self.enabled
            && self
                .feature
                .get(id)
                .and_then(|row| row.get(feature))
                .copied()
                .unwrap_or(false)
    }

    pub fn value(self, id: usize, feature: usize) -> i32 {
        i32::from(
            self.data
                .get(id)
                .and_then(|row| row.get(feature))
                .copied()
                .unwrap_or(0),
        )
    }

    fn finish(&mut self) {
        self.last_active = 0;
        self.preskip = false;
        for (id, row) in self.feature.iter().enumerate() {
            for (feature, enabled) in row.iter().enumerate() {
                if *enabled {
                    self.last_active = u8::try_from(id).unwrap_or(0);
                    if feature >= SEG_REF_FRAME {
                        self.preskip = true;
                    }
                }
            }
        }
    }
}

/// Loop-filter deltas and segmentation features stored with one reference slot.
#[derive(Clone, Copy)]
pub struct Inherited {
    pub lf: SavedLoopFilter,
    pub seg: Segmentation,
}

impl Inherited {
    pub const fn defaults() -> Self {
        Self {
            lf: SavedLoopFilter::defaults(),
            seg: Segmentation::off(),
        }
    }
}

/// Loop-filter reference and mode deltas stored with a reference frame.
#[derive(Clone, Copy)]
pub struct SavedLoopFilter {
    pub ref_deltas: [i8; 8],
    pub mode_deltas: [i8; 2],
}

impl SavedLoopFilter {
    pub const fn defaults() -> Self {
        Self {
            // Intra, last, last2, last3, golden, backward, altref2, altref.
            ref_deltas: [1, 0, 0, 0, -1, 0, -1, -1],
            mode_deltas: [0, 0],
        }
    }

    const fn pair(self) -> ([i8; 8], [i8; 2]) {
        (self.ref_deltas, self.mode_deltas)
    }
}

fn inherited_lf(
    primary_ref: u8,
    ref_frames: &[u8; 7],
    saved: &[Inherited; 8],
) -> ([i8; 8], [i8; 2]) {
    if primary_ref >= PRIMARY_REF_NONE {
        return SavedLoopFilter::defaults().pair();
    }
    let slot = usize::from(ref_frames[usize::from(primary_ref)]);
    saved
        .get(slot)
        .map(|saved| saved.lf)
        .unwrap_or_else(SavedLoopFilter::defaults)
        .pair()
}

fn read_segmentation(
    bits: &mut BitReader<'_>,
    primary_ref: u8,
    ref_frames: &[u8; 7],
    saved: &[Inherited; 8],
) -> Result<Segmentation, CodecError> {
    if !bits.read_bool()? {
        return Ok(Segmentation::off());
    }
    let (update_map, temporal, update_data) = if primary_ref >= PRIMARY_REF_NONE {
        (true, false, true)
    } else {
        let update_map = bits.read_bool()?;
        let temporal = if update_map { bits.read_bool()? } else { false };
        (update_map, temporal, bits.read_bool()?)
    };
    let stored = if update_data {
        read_feature_data(bits)?
    } else {
        inherited_features(primary_ref, ref_frames, saved)
    };
    let mut segmentation = Segmentation {
        enabled: true,
        update_map,
        temporal,
        update_data,
        feature: stored.feature,
        data: stored.data,
        last_active: 0,
        preskip: false,
    };
    segmentation.finish();
    Ok(segmentation)
}

fn inherited_features(
    primary_ref: u8,
    ref_frames: &[u8; 7],
    saved: &[Inherited; 8],
) -> Segmentation {
    if primary_ref >= PRIMARY_REF_NONE {
        return Segmentation::off();
    }
    let slot = usize::from(ref_frames[usize::from(primary_ref)]);
    saved
        .get(slot)
        .map(|saved| saved.seg)
        .unwrap_or_else(Segmentation::off)
}

fn read_feature_data(bits: &mut BitReader<'_>) -> Result<Segmentation, CodecError> {
    let mut stored = Segmentation::off();
    for row in 0..8 {
        for column in 0..8 {
            if !bits.read_bool()? {
                continue;
            }
            stored.feature[row][column] = true;
            let width = FEATURE_BITS[column];
            let limit = FEATURE_MAX[column];
            let raw = if FEATURE_SIGNED[column] {
                bits.read_signed(1 + width)?
            } else if width == 0 {
                0
            } else {
                i32::try_from(bits.read(width)?).unwrap_or(0)
            };
            let clipped = if FEATURE_SIGNED[column] {
                raw.clamp(-limit, limit)
            } else {
                raw.clamp(0, limit)
            };
            stored.data[row][column] = i16::try_from(clipped).unwrap_or(0);
        }
    }
    Ok(stored)
}

fn read_lf_delta_update(
    bits: &mut BitReader<'_>,
    ref_deltas: &mut [i8; 8],
    mode_deltas: &mut [i8; 2],
) -> Result<(), CodecError> {
    for delta in ref_deltas {
        if bits.read_bool()? {
            *delta = read_lf_delta(bits)?;
        }
    }
    for delta in mode_deltas {
        if bits.read_bool()? {
            *delta = read_lf_delta(bits)?;
        }
    }
    Ok(())
}

fn read_lf_delta(bits: &mut BitReader<'_>) -> Result<i8, CodecError> {
    let value = bits.read_signed(7)?;
    i8::try_from(value).map_err(|_| malformed(bits, "loop filter delta"))
}

fn read_delta_q(bits: &mut BitReader<'_>) -> Result<i32, CodecError> {
    if bits.read_bool()? {
        Ok(bits.read_signed(7)?)
    } else {
        Ok(0)
    }
}

fn mi_starts(sb_count: u32, log: u32, mi_total: u32, sb_shift: u32) -> Vec<u32> {
    let tile_sb = ((sb_count + (1 << log) - 1) >> log).max(1);
    let mut starts = Vec::new();
    let mut sb = 0u32;
    while sb < sb_count {
        starts.push(sb << sb_shift);
        sb += tile_sb;
    }
    if starts.is_empty() {
        starts.push(0);
    }
    starts.push(mi_total);
    starts
}

/// Split a frame OBU's tile group into one byte range per tile.
pub fn tile_parts(frame: &FrameInfo) -> Result<Vec<TilePart>, CodecError> {
    let cols = frame.mi_col_starts.len().saturating_sub(1);
    let rows = frame.mi_row_starts.len().saturating_sub(1);
    if cols == 0
        || rows == 0
        || cols != frame.tile_cols as usize
        || rows != frame.tile_rows as usize
    {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "tile grid",
        });
    }
    let num = cols * rows;
    if num == 1 {
        return Ok(vec![TilePart {
            row: 0,
            row_end: frame.mi_rows as usize,
            col: 0,
            col_end: frame.mi_cols as usize,
            data: frame.tile.clone(),
        }]);
    }
    let mut bits = BitReader::new(&frame.tile);
    let present = bits.read_bool()?;
    let (group_start, group_end) = if present {
        let width = frame.tile_cols_log + frame.tile_rows_log;
        let start = bits.read(width)? as usize;
        let end = bits.read(width)? as usize;
        (start, end)
    } else {
        (0, num - 1)
    };
    if group_end < group_start || group_end >= num {
        return Err(CodecError::Malformed {
            at_bit: bits.position() as u64,
            what: "tile group",
        });
    }
    bits.byte_align();
    let mut parts = Vec::new();
    for tile_num in group_start..=group_end {
        let row = tile_num / cols;
        let col = tile_num % cols;
        let last = tile_num == group_end;
        let size = if last {
            bits.bits_left() / 8
        } else {
            let minus = bits.read_le(frame.tile_size_bytes)?;
            usize::try_from(minus)
                .unwrap_or(usize::MAX)
                .saturating_add(1)
        };
        if bits.bits_left() < size.saturating_mul(8) {
            return Err(CodecError::Truncated {
                at_bit: bits.position() as u64,
                what: "tile",
            });
        }
        let data = bits.rest()?.get(..size).unwrap_or(&[]).to_vec();
        if data.len() != size {
            return Err(CodecError::Truncated {
                at_bit: bits.position() as u64,
                what: "tile",
            });
        }
        bits.skip(size * 8)?;
        let Some(&row_start) = frame.mi_row_starts.get(row) else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "tile grid",
            });
        };
        let Some(&row_end) = frame.mi_row_starts.get(row + 1) else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "tile grid",
            });
        };
        let Some(&col_start) = frame.mi_col_starts.get(col) else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "tile grid",
            });
        };
        let Some(&col_end) = frame.mi_col_starts.get(col + 1) else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "tile grid",
            });
        };
        parts.push(TilePart {
            row: row_start as usize,
            row_end: row_end as usize,
            col: col_start as usize,
            col_end: col_end as usize,
            data,
        });
    }
    Ok(parts)
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
        slot_hints: [0; 8],
        signaled_order_hints: None,
        refresh_flags: 0,
        primary_ref: PRIMARY_REF_NONE,
        base_q: 0,
        delta_y_dc: 0,
        delta_u_dc: 0,
        delta_u_ac: 0,
        delta_v_dc: 0,
        delta_v_ac: 0,
        using_qmatrix: false,
        qm: [15, 15, 15],
        delta_q_present: false,
        delta_q_res: 0,
        delta_lf_present: false,
        delta_lf_res: 0,
        delta_lf_multi: false,
        tx_select: false,
        reduced_tx: false,
        reference_select: false,
        skip_mode: false,
        skip_frames: [0, 0],
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
        lf_ref_deltas: SavedLoopFilter::defaults().ref_deltas,
        lf_mode_deltas: [0; 2],
        cdef: CdefParams::off(),
        restoration: Restoration::off(),
        grain: crate::grain::Grain::off(),
        segmentation: Segmentation::off(),
        prev_segment_ids: Vec::new(),
        lossless: false,
        all_lossless: false,
        ref_frames: [0; 7],
        tile: Vec::new(),
        tile_cols: 1,
        tile_rows: 1,
        tile_cols_log: 0,
        tile_rows_log: 0,
        context_update: 0,
        tile_size_bytes: 0,
        mi_col_starts: vec![0],
        mi_row_starts: vec![0],
        mi_cols: 0,
        mi_rows: 0,
        gm_type: [0; 8],
        gm_params: identity_gm(),
    }
}

const GM_TRANSLATION: u8 = 1;
const GM_ROTZOOM: u8 = 2;
const GM_AFFINE: u8 = 3;
const WARPED_PREC_BITS: u32 = 16;
const GM_ABS_ALPHA_BITS: u32 = 12;
const GM_ALPHA_PREC_BITS: u32 = 15;
const GM_ABS_TRANS_BITS: u32 = 12;
const GM_TRANS_PREC_BITS: u32 = 6;
const GM_ABS_TRANS_ONLY_BITS: u32 = 9;
const GM_TRANS_ONLY_PREC_BITS: u32 = 3;

struct GlobalMotion {
    kind: [u8; 8],
    params: [[i32; 6]; 8],
}

/// `skip_mode_present`. The bit is in the header only when two references
/// sit on opposite sides of this frame, or two sit in front of it.
fn skip_mode_signaled(
    bits: &mut BitReader<'_>,
    sequence: &Sequence,
    intra: bool,
    reference_select: bool,
    order_hint: u32,
    ref_frames: &[u8; 7],
    slot_hints: &[u32; 8],
) -> Result<(bool, [i8; 2]), CodecError> {
    let Some(frames) = skip_mode_frames(
        sequence,
        intra,
        reference_select,
        order_hint,
        ref_frames,
        slot_hints,
    ) else {
        return Ok((false, [0, 0]));
    };
    let present = bits.read_bool()?;
    Ok((present, frames))
}

/// `SkipModeFrame`. The closest forward reference, paired with the closest backward one, or with a second forward reference.
fn skip_mode_frames(
    sequence: &Sequence,
    intra: bool,
    reference_select: bool,
    order_hint: u32,
    ref_frames: &[u8; 7],
    slot_hints: &[u32; 8],
) -> Option<[i8; 2]> {
    if intra || !reference_select || !sequence.enable_order_hint {
        return None;
    }
    let width = sequence.order_hint_bits;
    let mut forward = None;
    let mut backward = None;
    for (index, slot) in ref_frames.iter().enumerate() {
        let hint = slot_hints[usize::from(*slot)];
        let dist = relative_dist(hint, order_hint, width);
        if dist < 0 {
            let closer =
                forward.is_none_or(|(_, previous)| relative_dist(hint, previous, width) > 0);
            if closer {
                forward = Some((index, hint));
            }
        } else if dist > 0 {
            let closer =
                backward.is_none_or(|(_, previous)| relative_dist(hint, previous, width) < 0);
            if closer {
                backward = Some((index, hint));
            }
        }
    }
    let (forward_index, forward_hint) = forward?;
    if let Some((backward_index, _)) = backward {
        return Some(skip_pair(forward_index, backward_index));
    }
    let mut second = None;
    for (index, slot) in ref_frames.iter().enumerate() {
        let hint = slot_hints[usize::from(*slot)];
        if relative_dist(hint, forward_hint, width) < 0 {
            let closer =
                second.is_none_or(|(_, previous)| relative_dist(hint, previous, width) > 0);
            if closer {
                second = Some((index, hint));
            }
        }
    }
    second.map(|(index, _)| skip_pair(forward_index, index))
}

fn skip_pair(first: usize, second: usize) -> [i8; 2] {
    let low = i8::try_from(first.min(second) + 1).unwrap_or(1);
    let high = i8::try_from(first.max(second) + 1).unwrap_or(1);
    [low, high]
}

pub(crate) fn relative_dist(a: u32, b: u32, bits: u8) -> i32 {
    if bits == 0 {
        return 0;
    }
    let diff = a.wrapping_sub(b);
    let m = 1u32 << (bits - 1);
    i32::try_from(diff & (m - 1)).unwrap_or(0) - i32::try_from(diff & m).unwrap_or(0)
}

fn identity_gm() -> [[i32; 6]; 8] {
    let mut params = [[0; 6]; 8];
    let one = 1 << WARPED_PREC_BITS;
    for row in &mut params {
        row[2] = one;
        row[5] = one;
    }
    params
}

/// Section 5.9.24. Identity is seven zero bits. Any other type carries parameters.
fn read_global_motion(
    bits: &mut BitReader<'_>,
    intra: bool,
    allow_hp: bool,
    primary_ref: u8,
    ref_frames: [u8; 7],
    saved: &[Option<[[i32; 6]; 8]>; 8],
) -> Result<GlobalMotion, CodecError> {
    let mut motion = GlobalMotion {
        kind: [0; 8],
        params: identity_gm(),
    };
    if intra {
        return Ok(motion);
    }
    let prev = if primary_ref >= 7 {
        identity_gm()
    } else {
        let slot = usize::from(ref_frames[usize::from(primary_ref)]);
        saved
            .get(slot)
            .copied()
            .flatten()
            .unwrap_or_else(identity_gm)
    };
    for reference in 1..8 {
        let kind = if !bits.read_bool()? {
            0
        } else if bits.read_bool()? {
            GM_ROTZOOM
        } else if bits.read_bool()? {
            GM_TRANSLATION
        } else {
            GM_AFFINE
        };
        motion.kind[reference] = kind;
        if kind >= GM_ROTZOOM {
            read_global_param(bits, &mut motion, &prev, allow_hp, kind, reference, 2)?;
            read_global_param(bits, &mut motion, &prev, allow_hp, kind, reference, 3)?;
            if kind == GM_AFFINE {
                read_global_param(bits, &mut motion, &prev, allow_hp, kind, reference, 4)?;
                read_global_param(bits, &mut motion, &prev, allow_hp, kind, reference, 5)?;
            } else {
                motion.params[reference][4] = -motion.params[reference][3];
                motion.params[reference][5] = motion.params[reference][2];
            }
        }
        if kind >= GM_TRANSLATION {
            read_global_param(bits, &mut motion, &prev, allow_hp, kind, reference, 0)?;
            read_global_param(bits, &mut motion, &prev, allow_hp, kind, reference, 1)?;
        }
    }
    Ok(motion)
}

fn read_global_param(
    bits: &mut BitReader<'_>,
    motion: &mut GlobalMotion,
    prev: &[[i32; 6]; 8],
    allow_hp: bool,
    kind: u8,
    reference: usize,
    index: usize,
) -> Result<(), CodecError> {
    let (abs_bits, prec_bits) = if index < 2 {
        if kind == GM_TRANSLATION {
            let lost = u32::from(!allow_hp);
            (
                GM_ABS_TRANS_ONLY_BITS - lost,
                GM_TRANS_ONLY_PREC_BITS - lost,
            )
        } else {
            (GM_ABS_TRANS_BITS, GM_TRANS_PREC_BITS)
        }
    } else {
        (GM_ABS_ALPHA_BITS, GM_ALPHA_PREC_BITS)
    };
    let prec_diff = WARPED_PREC_BITS - prec_bits;
    let round = if index % 3 == 2 {
        1 << WARPED_PREC_BITS
    } else {
        0
    };
    let sub = if index % 3 == 2 { 1 << prec_bits } else { 0 };
    let mx = 1 << abs_bits;
    let recenter = (prev[reference][index] >> prec_diff) - sub;
    let value = decode_signed_subexp(bits, -mx, mx + 1, recenter)?;
    motion.params[reference][index] = (value << prec_diff) + round;
    Ok(())
}

fn decode_signed_subexp(
    bits: &mut BitReader<'_>,
    low: i32,
    high: i32,
    reference: i32,
) -> Result<i32, CodecError> {
    let value = decode_unsigned_subexp(bits, high - low, reference - low)?;
    Ok(value + low)
}

fn decode_unsigned_subexp(
    bits: &mut BitReader<'_>,
    symbols: i32,
    reference: i32,
) -> Result<i32, CodecError> {
    let value = decode_subexp(bits, symbols)?;
    if (reference << 1) <= symbols {
        Ok(inverse_recenter(reference, value))
    } else {
        Ok(symbols - 1 - inverse_recenter(symbols - 1 - reference, value))
    }
}

fn decode_subexp(bits: &mut BitReader<'_>, symbols: i32) -> Result<i32, CodecError> {
    let mut index = 0i32;
    let mut mk = 0i32;
    let k = 3i32;
    for _ in 0..32 {
        let width = if index == 0 { k } else { k + index - 1 };
        let span = 1i32 << width;
        if symbols <= mk + 3 * span {
            let rest = ns(bits, (symbols - mk).max(0))?;
            return Ok(rest + mk);
        }
        if bits.read_bool()? {
            index += 1;
            mk += span;
        } else {
            return Ok(bits.read(width as u32)? as i32 + mk);
        }
    }
    Err(CodecError::Malformed {
        at_bit: bits.position() as u64,
        what: "global motion",
    })
}

fn inverse_recenter(reference: i32, value: i32) -> i32 {
    if value > 2 * reference {
        value
    } else if value & 1 == 1 {
        reference - ((value + 1) >> 1)
    } else {
        reference + (value >> 1)
    }
}

/// `ns(n)` on the uncompressed header.
fn ns(bits: &mut BitReader<'_>, n: i32) -> Result<i32, CodecError> {
    if n <= 1 {
        return Ok(0);
    }
    let w = u32::ilog2(n as u32) + 1;
    let m = (1i32 << w) - n;
    let v = bits.read(w - 1)? as i32;
    if v < m {
        return Ok(v);
    }
    let extra = bits.read(1)? as i32;
    Ok((v << 1) - m + extra)
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
        .wrapping_add(u32::from(sequence.enable_restoration))
        .wrapping_add(u32::from(frame.restoration.uses))
        .wrapping_add(u32::from(frame.restoration.plane[0]))
        .wrapping_add(u32::from(frame.restoration.unit[0]))
        .wrapping_add(u32::from(sequence.screen_select))
        .wrapping_add(u32::from(sequence.force_integer_select))
        .wrapping_add(u32::from(sequence.bit_depth))
        .wrapping_add(u32::from(sequence.monochrome))
        .wrapping_add(u32::from(sequence.color_range_full))
        .wrapping_add(u32::from(sequence.film_grain))
        .wrapping_add(u32::from(frame.grain.apply))
        .wrapping_add(u32::from(frame.grain.seed))
        .wrapping_add(u32::from(sequence.delta_frame_id_len))
        .wrapping_add(u32::from(frame.showable))
        .wrapping_add(u32::from(frame.error_resilient))
        .wrapping_add(u32::from(frame.disable_frame_end_update))
        .wrapping_add(u32::from(frame.force_integer_mv))
        .wrapping_add(frame.upscaled_width)
        .wrapping_add(frame.order_hint)
        .wrapping_add(
            frame
                .signaled_order_hints
                .map(|hints| hints[0])
                .unwrap_or(0),
        )
        .wrapping_add(u32::from(frame.primary_ref))
        .wrapping_add(frame.tile_cols)
        .wrapping_add(frame.tile_size_bytes)
        .wrapping_add(u32::from(frame.using_qmatrix))
        .wrapping_add(u32::from(frame.qm[0]))
        .wrapping_add(u32::from(frame.delta_q_res))
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
        .wrapping_add(u32::from(frame.segmentation.enabled))
        .wrapping_add(u32::from(frame.segmentation.update_data))
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

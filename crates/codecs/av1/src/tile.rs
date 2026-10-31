//! Intra tile decode: partitions, modes, coefficients, and reconstruction.

use cutlass_decode::CodecError;

use crate::cdef::{self, CdefInput};
use crate::deblock::{Edges, Picture, deblock_picture};
use crate::dims::{
    ADJUSTED_TX_SIZE, ADST_FLIPADST, BLOCK_8X8, BLOCK_128X128, COEFF_BASE_CTX_OFFSET, DC_PRED,
    DCT_DCT, DCT_FLIPADST, FLIPADST_ADST, FLIPADST_DCT, FLIPADST_FLIPADST, H_ADST, H_DCT,
    H_FLIPADST, IDTX, MAX_TX_DEPTH, MAX_TX_SIZE_RECT, MI_HEIGHT_LOG2, MI_WIDTH_LOG2, MODE_TO_TXFM,
    NUM_4X4_HIGH, NUM_4X4_WIDE, PARTITION_HORZ, PARTITION_HORZ_4, PARTITION_HORZ_A,
    PARTITION_HORZ_B, PARTITION_NONE, PARTITION_SPLIT, PARTITION_VERT, PARTITION_VERT_4,
    PARTITION_VERT_A, PARTITION_VERT_B, SMOOTH_PRED, SPLIT_TX_SIZE, TRANSFORM_ROW_SHIFT, TX_4X4,
    TX_8X8, TX_16X16, TX_16X32, TX_16X64, TX_32X16, TX_32X32, TX_32X64, TX_64X16, TX_64X32,
    TX_64X64, TX_HEIGHT, TX_HEIGHT_LOG2, TX_SIZE_SQR, TX_SIZE_SQR_UP, TX_WIDTH, TX_WIDTH_LOG2,
    UV_CFL_PRED, V_ADST, V_DCT, V_FLIPADST, V_PRED, block_height, block_width, intra_tx_type,
    partition_subsize, subsampled_size,
};
use crate::header::{CdefParams, FrameInfo, INTER_FRAME, Sequence};
use crate::header::{RESTORE_SGRPROJ, RESTORE_SWITCHABLE, RESTORE_WIENER};
use crate::idct::{InverseScratch, inverse_2d_into};
use crate::inter::{
    ALTREF_FRAME, ALTREF2_FRAME, BWDREF_FRAME, BlockPlace, Candidates, FieldSource,
    GLOBAL_GLOBALMV, GLOBALMV, GM_TRANSLATION, GOLDEN_FRAME, INTRA_FRAME, InterSample, LAST_FRAME,
    LAST2_FRAME, LAST3_FRAME, MOTION_LOCAL_WARP, MOTION_OBMC, MOTION_SIMPLE, Motion, MotionField,
    MotionGrid, NEAREST_NEARESTMV, NEARESTMV, NEARMV, NEWMV, NONE_FRAME, RefView, SIZE_GROUP,
    SaveQuery, average_compound, compound_mask_weight, difference_mask, distance_compound,
    find_stack, global_shear_valid, has_near_mv, inter_set_allows, inter_tx_set, inter_tx_type,
    interintra_weight, list_mode, local_warp_params, mask_compound, motion_unused, predict,
    predict_precise, predict_warp, project_global, project_motion_field, save_motion_field,
    saved_views, wedge_bits, wedge_mask,
};
use crate::predict::{ChromaFromLuma, IntraPred, PredictScratch, chroma_from_luma, predict_intra};
use crate::restore;
use crate::superres;
use crate::symbol::SymbolDecoder;
use crate::tables::{
    AC_QLOOKUP, DC_QLOOKUP, DEFAULT_ANGLE_DELTA_CDF, DEFAULT_CFL_ALPHA_CDF, DEFAULT_CFL_SIGN_CDF,
    DEFAULT_COEFF_BASE_CDF, DEFAULT_COEFF_BASE_EOB_CDF, DEFAULT_COEFF_BR_CDF, DEFAULT_DC_SIGN_CDF,
    DEFAULT_DRL_MODE_CDF, DEFAULT_EOB_EXTRA_CDF, DEFAULT_EOB_PT_16_CDF, DEFAULT_EOB_PT_32_CDF,
    DEFAULT_EOB_PT_64_CDF, DEFAULT_EOB_PT_128_CDF, DEFAULT_EOB_PT_256_CDF, DEFAULT_EOB_PT_512_CDF,
    DEFAULT_EOB_PT_1024_CDF, DEFAULT_INTER_TX_TYPE_SET1_CDF, DEFAULT_INTER_TX_TYPE_SET2_CDF,
    DEFAULT_INTER_TX_TYPE_SET3_CDF, DEFAULT_INTERP_FILTER_CDF, DEFAULT_INTRA_FRAME_Y_MODE_CDF,
    DEFAULT_INTRA_TX_TYPE_SET1_CDF, DEFAULT_INTRA_TX_TYPE_SET2_CDF, DEFAULT_IS_INTER_CDF,
    DEFAULT_MV_BIT_CDF, DEFAULT_MV_CLASS_CDF, DEFAULT_MV_CLASS0_BIT_CDF, DEFAULT_MV_CLASS0_FR_CDF,
    DEFAULT_MV_CLASS0_HP_CDF, DEFAULT_MV_FR_CDF, DEFAULT_MV_HP_CDF, DEFAULT_MV_JOINT_CDF,
    DEFAULT_MV_SIGN_CDF, DEFAULT_NEW_MV_CDF, DEFAULT_PARTITION_W8_CDF, DEFAULT_PARTITION_W16_CDF,
    DEFAULT_PARTITION_W32_CDF, DEFAULT_PARTITION_W64_CDF, DEFAULT_PARTITION_W128_CDF,
    DEFAULT_REF_MV_CDF, DEFAULT_SCAN_4X4, DEFAULT_SCAN_4X8, DEFAULT_SCAN_4X16, DEFAULT_SCAN_8X4,
    DEFAULT_SCAN_8X8, DEFAULT_SCAN_8X16, DEFAULT_SCAN_8X32, DEFAULT_SCAN_16X4, DEFAULT_SCAN_16X8,
    DEFAULT_SCAN_16X16, DEFAULT_SCAN_16X32, DEFAULT_SCAN_32X8, DEFAULT_SCAN_32X16,
    DEFAULT_SCAN_32X32, DEFAULT_SINGLE_REF_CDF, DEFAULT_SKIP_CDF, DEFAULT_TX_8X8_CDF,
    DEFAULT_TX_16X16_CDF, DEFAULT_TX_32X32_CDF, DEFAULT_TX_64X64_CDF, DEFAULT_TXB_SKIP_CDF,
    DEFAULT_TXFM_SPLIT_CDF, DEFAULT_UV_MODE_CFL_ALLOWED_CDF, DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF,
    DEFAULT_Y_MODE_CDF, DEFAULT_ZERO_MV_CDF, MCOL_SCAN_4X4, MCOL_SCAN_4X8, MCOL_SCAN_4X16,
    MCOL_SCAN_8X4, MCOL_SCAN_8X8, MCOL_SCAN_8X16, MCOL_SCAN_16X4, MCOL_SCAN_16X8, MCOL_SCAN_16X16,
    MROW_SCAN_4X4, MROW_SCAN_4X8, MROW_SCAN_4X16, MROW_SCAN_8X4, MROW_SCAN_8X8, MROW_SCAN_8X16,
    MROW_SCAN_16X4, MROW_SCAN_16X8, MROW_SCAN_16X16, SIG_REF_DIFF_OFFSET,
};

pub(crate) const MODE_CTX: [usize; 13] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];
const SB4: usize = 16;
/// `MAX_VARTX_DEPTH`. A selected inter transform splits at most twice.
const MAX_VARTX_DEPTH: usize = 2;

pub struct DecodedPicture {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

/// A reconstructed picture and the CDF context stored with its reference slots.
pub struct Reconstructed {
    pub picture: DecodedPicture,
    pub cdfs: Cdfs,
    pub motion: crate::inter::SavedMotion,
}

/// Decode one frame from its own bytes. Tests use this when the frame does not
/// name a primary reference.
#[cfg(test)]
pub fn decode_tile(sequence: &Sequence, frame: &FrameInfo) -> Result<Reconstructed, CodecError> {
    decode_with_refs(sequence, frame, &[], None)
}

pub fn decode_with_refs(
    sequence: &Sequence,
    frame: &FrameInfo,
    refs: &[Option<RefView<'_>>],
    carried: Option<&Cdfs>,
) -> Result<Reconstructed, CodecError> {
    if frame.lossless {
        return Err(CodecError::Unsupported { what: "lossless" });
    }
    if frame.allow_screen {
        return Err(CodecError::Unsupported {
            what: "screen content",
        });
    }
    if frame.width > 4096 || frame.height > 4096 {
        return Err(CodecError::Limit { what: "frame size" });
    }
    let visible_w = frame.width as usize;
    let visible_h = frame.height as usize;
    if visible_w == 0 || visible_h == 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "frame size",
        });
    }
    let buf_w = (frame.mi_cols as usize).saturating_mul(4);
    let buf_h = (frame.mi_rows as usize).saturating_mul(4);
    let parts = crate::header::tile_parts(frame)?;
    if parts.is_empty() {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "tile",
        });
    }
    let sb = if sequence.use_128 { 32 } else { SB4 };
    let initial = frame_cdfs(frame, carried)?;
    let mut lr = restore::Grid::new(frame)?;
    let mut end_cdfs = None;
    let mut tile = if parts.len() > 1 && !lr.active() {
        let (tile, end) = decode_tiles(sequence, frame, &parts, buf_w, buf_h, refs, &initial)?;
        end_cdfs = end;
        tile
    } else {
        let mut tile = Tile::new(
            sequence,
            frame,
            &parts[0].data,
            buf_w,
            buf_h,
            refs,
            &initial,
        )?;
        for (index, part) in parts.iter().enumerate() {
            if index > 0 {
                tile.restart(&part.data)?;
            }
            tile.decode_part(frame, part, sb, &mut lr)?;
            if part_tile_num(frame, part)? == frame.context_update as usize {
                end_cdfs = Some(tile.cdfs.clone());
            }
        }
        tile
    };
    let cdfs = if frame.disable_frame_end_update {
        initial
    } else {
        end_cdfs.unwrap_or(initial)
    };
    let sizes = tile.sizes.clone();
    let tx_sizes = tile.tx_sizes.clone();
    let refs = tile.block_ref.clone();
    let modes = tile.y_mode.clone();
    let edges = Edges {
        sizes: &sizes,
        tx: &tx_sizes,
        refs: &refs,
        modes: &modes,
        mi_cols: tile.mi_cols,
        mi_rows: tile.mi_rows,
        sub_x: tile.sub_x,
        sub_y: tile.sub_y,
        frame,
    };
    deblock_picture(
        Picture {
            y: &mut tile.y,
            y_w: buf_w,
            y_h: buf_h,
            u: &mut tile.u,
            v: &mut tile.v,
            c_w: tile.chroma_w,
            c_h: tile.chroma_h,
        },
        &edges,
    );
    let before = if lr.active() {
        Some((tile.y.clone(), tile.u.clone(), tile.v.clone()))
    } else {
        None
    };
    if tile.cdef_on {
        cdef::apply(&mut CdefInput {
            y: &mut tile.y,
            u: &mut tile.u,
            v: &mut tile.v,
            y_stride: buf_w,
            c_stride: tile.chroma_w,
            mi_cols: tile.mi_cols,
            mi_rows: tile.mi_rows,
            skips: &tile.skips,
            index: &tile.cdef_idx,
            params: tile.cdef,
        });
    }
    let chroma_w = visible_w.div_ceil(2);
    let chroma_h = visible_h.div_ceil(2);
    let up_w = frame.upscaled_width as usize;
    let (y, u, v, out_w) = if up_w > visible_w {
        let up_chroma = up_w.div_ceil(2);
        (
            superres::upscale(
                &tile.y,
                buf_w,
                visible_w,
                up_w,
                visible_h,
                buf_w.saturating_sub(1),
            ),
            superres::upscale(
                &tile.u,
                tile.chroma_w,
                chroma_w,
                up_chroma,
                chroma_h,
                tile.chroma_w.saturating_sub(1),
            ),
            superres::upscale(
                &tile.v,
                tile.chroma_w,
                chroma_w,
                up_chroma,
                chroma_h,
                tile.chroma_w.saturating_sub(1),
            ),
            up_w,
        )
    } else {
        (
            crop(&tile.y, buf_w, visible_w, visible_h),
            crop(&tile.u, tile.chroma_w, chroma_w, chroma_h),
            crop(&tile.v, tile.chroma_w, chroma_w, chroma_h),
            visible_w,
        )
    };
    let (y, u, v) = if let Some((pre_y, pre_u, pre_v)) = before {
        let (before_y, before_u, before_v) = if up_w > visible_w {
            let up_chroma = up_w.div_ceil(2);
            (
                superres::upscale(
                    &pre_y,
                    buf_w,
                    visible_w,
                    up_w,
                    visible_h,
                    buf_w.saturating_sub(1),
                ),
                superres::upscale(
                    &pre_u,
                    tile.chroma_w,
                    chroma_w,
                    up_chroma,
                    chroma_h,
                    tile.chroma_w.saturating_sub(1),
                ),
                superres::upscale(
                    &pre_v,
                    tile.chroma_w,
                    chroma_w,
                    up_chroma,
                    chroma_h,
                    tile.chroma_w.saturating_sub(1),
                ),
            )
        } else {
            (
                crop(&pre_y, buf_w, visible_w, visible_h),
                crop(&pre_u, tile.chroma_w, chroma_w, chroma_h),
                crop(&pre_v, tile.chroma_w, chroma_w, chroma_h),
            )
        };
        (
            restore_plane(&y, &before_y, out_w, visible_h, &lr, 0, frame),
            restore_plane(&u, &before_u, out_w.div_ceil(2), chroma_h, &lr, 1, frame),
            restore_plane(&v, &before_v, out_w.div_ceil(2), chroma_h, &lr, 2, frame),
        )
    } else {
        (y, u, v)
    };
    let motion = if sequence.enable_ref_mvs {
        save_motion_field(&SaveQuery {
            frame_type: frame.frame_type,
            mi_cols: tile.mi_cols,
            mi_rows: tile.mi_rows,
            order_hint: frame.order_hint,
            hint_bits: sequence.order_hint_bits,
            ref_slots: frame.ref_frames,
            slot_hints: &frame.slot_hints,
            block_ref: &tile.block_ref,
            block_ref1: &tile.block_ref1,
            mvs: &tile.mvs,
            mvs1: &tile.mvs1,
        })
    } else {
        motion_unused()
    };
    Ok(Reconstructed {
        picture: DecodedPicture {
            y,
            u,
            v,
            width: out_w,
            height: visible_h,
        },
        cdfs,
        motion,
    })
}

fn restore_plane(
    cdef: &[u8],
    before: &[u8],
    width: usize,
    height: usize,
    lr: &restore::Grid,
    plane: usize,
    frame: &FrameInfo,
) -> Vec<u8> {
    match frame.restoration.plane.get(plane).copied() {
        Some(RESTORE_WIENER) => restore::apply_wiener(cdef, before, width, height, lr, plane),
        Some(RESTORE_SGRPROJ) | Some(RESTORE_SWITCHABLE) => {
            restore::apply_sgr(cdef, before, width, height, lr, plane)
        }
        _ => cdef.to_vec(),
    }
}

/// Section 6.8.2. A primary reference supplies its saved CDFs with counters cleared.
/// An intra frame, or `PRIMARY_REF_NONE`, loads the default tables for this quantizer.
fn frame_cdfs(frame: &FrameInfo, carried: Option<&Cdfs>) -> Result<Cdfs, CodecError> {
    if !frame.intra && frame.primary_ref < 7 {
        let Some(saved) = carried else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "missing reference",
            });
        };
        let mut cdfs = saved.clone();
        cdfs.clear_counts();
        Ok(cdfs)
    } else {
        Ok(Cdfs::load(quantizer_context(frame.base_q)))
    }
}

fn quantizer_context(base_q: u8) -> usize {
    match base_q {
        0..=20 => 0,
        21..=60 => 1,
        61..=120 => 2,
        _ => 3,
    }
}

fn part_tile_num(frame: &FrameInfo, part: &crate::header::TilePart) -> Result<usize, CodecError> {
    let cols = frame.tile_cols as usize;
    if cols == 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "tile grid",
        });
    }
    let row = frame
        .mi_row_starts
        .iter()
        .position(|start| *start as usize == part.row)
        .ok_or(CodecError::Malformed {
            at_bit: 0,
            what: "tile grid",
        })?;
    let col = frame
        .mi_col_starts
        .iter()
        .position(|start| *start as usize == part.col)
        .ok_or(CodecError::Malformed {
            at_bit: 0,
            what: "tile grid",
        })?;
    Ok(row * cols + col)
}

/// Tiles reconstruct independently. Loop-filter rows then share the remaining cores.
fn decode_tiles<'a, 'r>(
    sequence: &Sequence,
    frame: &FrameInfo,
    parts: &'a [crate::header::TilePart],
    buf_w: usize,
    buf_h: usize,
    refs: &[Option<RefView<'r>>],
    initial: &Cdfs,
) -> Result<(Tile<'a, 'r>, Option<Cdfs>), CodecError> {
    let sb = if sequence.use_128 { 32 } else { SB4 };
    let decoded = std::thread::scope(|scope| {
        let mut joins = Vec::with_capacity(parts.len());
        for part in parts {
            joins.push(scope.spawn(|| {
                let mut tile = Tile::new(sequence, frame, &part.data, buf_w, buf_h, refs, initial)?;
                let mut grid = restore::Grid::new(frame)?;
                tile.decode_part(frame, part, sb, &mut grid)?;
                Ok(tile)
            }));
        }
        let mut tiles = Vec::with_capacity(joins.len());
        let mut error = None;
        for join in joins {
            match join.join() {
                Ok(Ok(tile)) => tiles.push(tile),
                Ok(Err(err)) => error = error.or(Some(err)),
                Err(_) => {
                    error = error.or(Some(CodecError::Malformed {
                        at_bit: 0,
                        what: "tile",
                    }));
                }
            }
        }
        if let Some(err) = error {
            return Err(err);
        }
        Ok(tiles)
    })?;
    let mut tiles = decoded;
    let context = frame.context_update as usize;
    let end_cdfs = tiles
        .iter()
        .find(|tile| tile.tile_num == context)
        .map(|tile| tile.cdfs.clone());
    let Some(mut base) = tiles.pop() else {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "tile",
        });
    };
    // `pop` took the last tile. Copy the earlier rectangles into it.
    for extra in tiles {
        copy_tile_region(&mut base, &extra);
    }
    Ok((base, end_cdfs))
}

fn copy_tile_region(dst: &mut Tile<'_, '_>, src: &Tile<'_, '_>) {
    let y0 = src.tile_row.saturating_mul(4);
    let y1 = src.tile_row_end.saturating_mul(4);
    let x0 = src.tile_col.saturating_mul(4);
    let x1 = src.tile_col_end.saturating_mul(4);
    copy_plane(&mut dst.y, &src.y, dst.width, x0, x1, y0, y1);
    let shift_x = usize::from(src.sub_x);
    let shift_y = usize::from(src.sub_y);
    copy_plane(
        &mut dst.u,
        &src.u,
        dst.chroma_w,
        x0 >> shift_x,
        x1 >> shift_x,
        y0 >> shift_y,
        y1 >> shift_y,
    );
    copy_plane(
        &mut dst.v,
        &src.v,
        dst.chroma_w,
        x0 >> shift_x,
        x1 >> shift_x,
        y0 >> shift_y,
        y1 >> shift_y,
    );
    let cols = dst.mi_cols;
    for row in src.tile_row..src.tile_row_end {
        let start = row.saturating_mul(cols).saturating_add(src.tile_col);
        let n = src.tile_col_end.saturating_sub(src.tile_col);
        copy_cells(&mut dst.sizes, &src.sizes, start, n);
        copy_cells(&mut dst.tx_sizes, &src.tx_sizes, start, n);
        copy_cells(&mut dst.skips, &src.skips, start, n);
        copy_cells(&mut dst.cdef_idx, &src.cdef_idx, start, n);
        copy_cells(&mut dst.mvs, &src.mvs, start, n);
        copy_cells(&mut dst.mvs1, &src.mvs1, start, n);
        copy_cells(&mut dst.block_ref, &src.block_ref, start, n);
        copy_cells(&mut dst.block_ref1, &src.block_ref1, start, n);
    }
}

fn copy_plane(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
) {
    let n = x1.saturating_sub(x0);
    if n == 0 || stride == 0 {
        return;
    }
    for y in y0..y1 {
        let start = y.saturating_mul(stride).saturating_add(x0);
        let Some(from) = src.get(start..start.saturating_add(n)) else {
            continue;
        };
        let Some(into) = dst.get_mut(start..start.saturating_add(n)) else {
            continue;
        };
        into.copy_from_slice(from);
    }
}

fn copy_cells<T: Copy>(dst: &mut [T], src: &[T], start: usize, n: usize) {
    let Some(from) = src.get(start..start.saturating_add(n)) else {
        return;
    };
    let Some(into) = dst.get_mut(start..start.saturating_add(n)) else {
        return;
    };
    into.copy_from_slice(from);
}

fn crop(src: &[u8], stride: usize, width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0u8; width.saturating_mul(height)];
    if stride == 0 || width == 0 || height == 0 {
        return out;
    }
    let n = width.min(stride);
    for row in 0..height {
        let from = row.saturating_mul(stride);
        let to = row.saturating_mul(width);
        if from + n <= src.len() && to + n <= out.len() {
            out[to..to + n].copy_from_slice(&src[from..from + n]);
        }
    }
    out
}

#[derive(Clone, Copy)]
struct PlaneRect {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    sub_x: bool,
    sub_y: bool,
}

struct Tile<'a, 'r> {
    symbols: SymbolDecoder<'a>,
    update: bool,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    width: usize,
    height: usize,
    chroma_w: usize,
    chroma_h: usize,
    mi_cols: usize,
    mi_rows: usize,
    tile_row: usize,
    tile_col: usize,
    tile_row_end: usize,
    tile_col_end: usize,
    tile_num: usize,
    initial_cdfs: Cdfs,
    y_mode: Vec<u8>,
    uv_mode: Vec<u8>,
    skips: Vec<u8>,
    sizes: Vec<u8>,
    tx_sizes: Vec<u8>,
    above_level: [Vec<u8>; 3],
    left_level: [Vec<u8>; 3],
    above_dc: [Vec<u8>; 3],
    left_dc: [Vec<u8>; 3],
    decoded: [Vec<u8>; 3],
    base_q: u8,
    qm: [u8; 3],
    delta_q_res: u8,
    current_q: u8,
    read_deltas: bool,
    deltas: [i32; 5],
    tx_select: bool,
    reduced_tx: bool,
    lossless: bool,
    edge_filter: bool,
    filter_intra: bool,
    filter_mode: Option<u8>,
    screen: bool,
    cfl_u: i32,
    cfl_v: i32,
    max_luma_w: usize,
    max_luma_h: usize,
    sub_x: bool,
    sub_y: bool,
    inter_frame: bool,
    force_integer: bool,
    allow_hp: bool,
    filter_switchable: bool,
    interpolation: u8,
    dual_filter: bool,
    interintra: bool,
    gm_type: [u8; 8],
    gm_params: [[i32; 6]; 8],
    frame_refs: [u8; 7],
    visible_w: usize,
    visible_h: usize,
    ref_y: [&'r [u8]; 8],
    ref_u: [&'r [u8]; 8],
    ref_v: [&'r [u8]; 8],
    ref_w: [usize; 8],
    ref_h: [usize; 8],
    ref_ok: [bool; 8],
    mvs: Vec<Motion>,
    mvs1: Vec<Motion>,
    reference_select: bool,
    masked_compound: bool,
    distance_compound: bool,
    order_hint: u32,
    order_hint_bits: u8,
    slot_hints: [u32; 8],
    sign_bias: [bool; 8],
    compound_idx: Vec<u8>,
    /// Packed compound mask: group, kind, wedge index, sign, and mask type.
    compound_mask: Vec<u16>,
    blend_mask: Vec<u8>,
    blend_w: usize,
    blend_ready: bool,
    skip_present: bool,
    skip_ref: [i8; 2],
    skip_modes: Vec<u8>,
    motion_modes: Vec<u8>,
    allow_warped: bool,
    motion_switchable: bool,
    use_ref_mvs: bool,
    motion_field: MotionField,
    block_ref: Vec<i8>,
    block_ref1: Vec<i8>,
    inter_flag: Vec<u8>,
    interp: Vec<[u8; 2]>,
    /// Interintra mode for each 4×4, or -1 when the block is not interintra.
    ii_mode: Vec<i8>,
    tx_kind: Vec<u8>,
    coding_inter: bool,
    cur_luma_row: usize,
    cur_luma_col: usize,
    cdef_on: bool,
    cdef: CdefParams,
    cdef_idx: Vec<i8>,
    cdfs: Cdfs,
    /// Luma transform origins for an inter block that selects its sizes.
    tx_origins: Vec<(u16, u16, u8)>,
    coeff_buf: Vec<i32>,
    quant_buf: Vec<i32>,
    residual_buf: Vec<i64>,
    idct_row: Vec<i64>,
    idct_out: Vec<i32>,
    predict: PredictScratch,
}

#[derive(Clone)]
pub(crate) struct Cdfs {
    partition: [Vec<u16>; 5],
    skip: Vec<u16>,
    y_mode: Vec<u16>,
    uv_yes: Vec<u16>,
    uv_no: Vec<u16>,
    angle: Vec<u16>,
    tx: [Vec<u16>; 4],
    intra_tx1: Vec<u16>,
    intra_tx2: Vec<u16>,
    txb_skip: Vec<u16>,
    eob: [Vec<u16>; 7],
    eob_extra: Vec<u16>,
    dc_sign: Vec<u16>,
    coeff_eob: Vec<u16>,
    coeff_base: Vec<u16>,
    coeff_br: Vec<u16>,
    cfl_sign: Vec<u16>,
    cfl_alpha: Vec<u16>,
    is_inter: Vec<u16>,
    single_ref: Vec<u16>,
    new_mv: Vec<u16>,
    zero_mv: Vec<u16>,
    ref_mv: Vec<u16>,
    drl: Vec<u16>,
    mv_joint: Vec<u16>,
    mv_sign: Vec<u16>,
    mv_class: Vec<u16>,
    mv_class0_bit: Vec<u16>,
    mv_class0_fr: Vec<u16>,
    mv_class0_hp: Vec<u16>,
    mv_bit: Vec<u16>,
    mv_fr: Vec<u16>,
    mv_hp: Vec<u16>,
    interp_filter: Vec<u16>,
    inter_tx1: Vec<u16>,
    inter_tx2: Vec<u16>,
    inter_tx3: Vec<u16>,
    txfm_split: Vec<u16>,
    intra_y: Vec<u16>,
    delta_q: Vec<u16>,
    use_filter_intra: Vec<u16>,
    filter_intra: Vec<u16>,
    inter_intra: Vec<u16>,
    inter_intra_mode: Vec<u16>,
    wedge_interintra: Vec<u16>,
    comp_mode: Vec<u16>,
    comp_ref_type: Vec<u16>,
    uni_comp_ref: Vec<u16>,
    comp_ref: Vec<u16>,
    comp_bwdref: Vec<u16>,
    compound_mode: Vec<u16>,
    compound_idx: Vec<u16>,
    comp_group: Vec<u16>,
    compound_type: Vec<u16>,
    wedge_index: Vec<u16>,
    skip_mode: Vec<u16>,
    motion_mode: Vec<u16>,
    obmc: Vec<u16>,
}

impl<'a, 'r> Tile<'a, 'r> {
    fn new(
        sequence: &Sequence,
        frame: &FrameInfo,
        bytes: &'a [u8],
        width: usize,
        height: usize,
        refs: &[Option<RefView<'r>>],
        initial: &Cdfs,
    ) -> Result<Self, CodecError> {
        let mi_cols = frame.mi_cols as usize;
        let mi_rows = frame.mi_rows as usize;
        let cells = mi_cols.saturating_mul(mi_rows);
        if cells == 0 || bytes.is_empty() {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "tile",
            });
        }
        let chroma_w = if sequence.subsampling_x {
            width.div_ceil(2)
        } else {
            width
        };
        let chroma_h = if sequence.subsampling_y {
            height.div_ceil(2)
        } else {
            height
        };
        let mut ref_y: [&'r [u8]; 8] = [&[]; 8];
        let mut ref_u: [&'r [u8]; 8] = [&[]; 8];
        let mut ref_v: [&'r [u8]; 8] = [&[]; 8];
        let mut ref_w = [0usize; 8];
        let mut ref_h = [0usize; 8];
        let mut ref_ok = [false; 8];
        for (slot, view) in refs.iter().enumerate().take(8) {
            if let Some(view) = view {
                ref_y[slot] = view.y;
                ref_u[slot] = view.u;
                ref_v[slot] = view.v;
                ref_w[slot] = view.width;
                ref_h[slot] = view.height;
                ref_ok[slot] = true;
            }
        }
        Ok(Self {
            symbols: SymbolDecoder::new(bytes)?,
            update: !frame.disable_cdf_update,
            y: vec![0; width * height],
            u: vec![128; chroma_w * chroma_h],
            v: vec![128; chroma_w * chroma_h],
            width,
            height,
            chroma_w,
            chroma_h,
            mi_cols,
            mi_rows,
            tile_row: 0,
            tile_col: 0,
            tile_row_end: frame.mi_rows as usize,
            tile_col_end: frame.mi_cols as usize,
            tile_num: 0,
            initial_cdfs: initial.clone(),
            y_mode: vec![DC_PRED; cells],
            uv_mode: vec![DC_PRED; cells],
            skips: vec![0; cells],
            sizes: vec![0; cells],
            tx_sizes: vec![0; cells],
            above_level: [vec![0; mi_cols], vec![0; mi_cols], vec![0; mi_cols]],
            left_level: [vec![0; mi_rows], vec![0; mi_rows], vec![0; mi_rows]],
            above_dc: [vec![0; mi_cols], vec![0; mi_cols], vec![0; mi_cols]],
            left_dc: [vec![0; mi_rows], vec![0; mi_rows], vec![0; mi_rows]],
            decoded: [vec![0; 40 * 40], vec![0; 40 * 40], vec![0; 40 * 40]],
            base_q: frame.base_q,
            qm: frame.qm,
            delta_q_res: frame.delta_q_res,
            current_q: frame.base_q,
            read_deltas: false,
            deltas: [
                frame.delta_y_dc,
                frame.delta_u_dc,
                frame.delta_u_ac,
                frame.delta_v_dc,
                frame.delta_v_ac,
            ],
            tx_select: frame.tx_select,
            reduced_tx: frame.reduced_tx,
            lossless: frame.lossless,
            edge_filter: sequence.enable_intra_edge,
            filter_intra: sequence.enable_filter_intra,
            filter_mode: None,
            screen: frame.allow_screen,
            cfl_u: 0,
            cfl_v: 0,
            max_luma_w: 0,
            max_luma_h: 0,
            sub_x: sequence.subsampling_x,
            sub_y: sequence.subsampling_y,
            inter_frame: frame.frame_type == INTER_FRAME,
            force_integer: frame.force_integer_mv,
            allow_hp: frame.allow_high_precision_mv,
            filter_switchable: frame.filter_switchable,
            interpolation: frame.interpolation,
            dual_filter: sequence.enable_dual_filter,
            interintra: sequence.enable_interintra,
            gm_type: frame.gm_type,
            gm_params: frame.gm_params,
            frame_refs: frame.ref_frames,
            visible_w: frame.width as usize,
            visible_h: frame.height as usize,
            ref_y,
            ref_u,
            ref_v,
            ref_w,
            ref_h,
            ref_ok,
            mvs: vec![Motion::default(); cells],
            mvs1: vec![Motion::default(); cells],
            reference_select: frame.reference_select,
            masked_compound: sequence.enable_masked,
            distance_compound: sequence.enable_jnt,
            order_hint: frame.order_hint,
            order_hint_bits: sequence.order_hint_bits,
            slot_hints: frame.slot_hints,
            sign_bias: sign_bias(sequence, frame),
            compound_idx: vec![1; cells],
            compound_mask: vec![0; cells],
            blend_mask: Vec::new(),
            blend_w: 0,
            blend_ready: false,
            skip_present: frame.skip_mode,
            skip_ref: frame.skip_frames,
            skip_modes: vec![0; cells],
            motion_modes: vec![MOTION_SIMPLE; cells],
            allow_warped: frame.allow_warped,
            motion_switchable: frame.motion_mode_switchable,
            use_ref_mvs: frame.use_ref_frame_mvs,
            motion_field: if frame.use_ref_frame_mvs {
                let saved = saved_views(refs);
                project_motion_field(&FieldSource {
                    mi_cols,
                    mi_rows,
                    order_hint: frame.order_hint,
                    hint_bits: sequence.order_hint_bits,
                    ref_slots: frame.ref_frames,
                    slot_hints: &frame.slot_hints,
                    saved: &saved,
                })
            } else {
                MotionField::empty()
            },
            block_ref: vec![0; cells],
            block_ref1: vec![NONE_FRAME; cells],
            inter_flag: vec![0; cells],
            interp: vec![[0, 0]; cells],
            ii_mode: vec![-1; cells],
            tx_kind: vec![DCT_DCT; cells],
            coding_inter: false,
            cur_luma_row: 0,
            cur_luma_col: 0,
            cdef_on: sequence.enable_cdef,
            cdef: frame.cdef,
            cdef_idx: vec![-1; cells],
            cdfs: initial.clone(),
            tx_origins: Vec::new(),
            coeff_buf: Vec::new(),
            quant_buf: Vec::new(),
            residual_buf: Vec::new(),
            idct_row: Vec::new(),
            idct_out: Vec::new(),
            predict: PredictScratch::new(),
        })
    }

    fn restart(&mut self, bytes: &'a [u8]) -> Result<(), CodecError> {
        self.symbols = SymbolDecoder::new(bytes)?;
        self.cdfs = self.initial_cdfs.clone();
        self.current_q = self.base_q;
        self.read_deltas = false;
        self.filter_mode = None;
        Ok(())
    }

    fn decode_part(
        &mut self,
        frame: &FrameInfo,
        part: &crate::header::TilePart,
        sb: usize,
        lr: &mut restore::Grid,
    ) -> Result<(), CodecError> {
        self.tile_row = part.row;
        self.tile_col = part.col;
        self.tile_row_end = part.row_end;
        self.tile_col_end = part.col_end;
        self.tile_num = part_tile_num(frame, part)?;
        self.clear_above(part.col, part.col_end);
        let mut row = part.row;
        while row < part.row_end {
            self.clear_left();
            let mut col = part.col;
            while col < part.col_end {
                self.clear_decoded(row, col);
                self.read_deltas = frame.delta_q_present;
                restore::read_superblock(&mut self.symbols, self.update, frame, lr, row, col, sb)?;
                let root = if sb == 32 { BLOCK_128X128 } else { 12 };
                self.decode_partition(row, col, root)?;
                col += sb;
            }
            row += sb;
        }
        self.symbols.finish()?;
        Ok(())
    }

    fn above_mi(&self, row: usize) -> bool {
        row > self.tile_row
    }

    fn left_mi(&self, col: usize) -> bool {
        col > self.tile_col
    }

    fn clear_above(&mut self, col: usize, col_end: usize) {
        for plane in 0..3 {
            let end = col_end.min(self.above_level[plane].len());
            for x in col..end {
                self.above_level[plane][x] = 0;
                self.above_dc[plane][x] = 0;
            }
        }
    }

    fn clear_left(&mut self) {
        for plane in 0..3 {
            self.left_level[plane].fill(0);
            self.left_dc[plane].fill(0);
        }
    }

    fn clear_decoded(&mut self, mi_row: usize, mi_col: usize) {
        for plane in 0..3 {
            let (sub_x, sub_y) = self.subsampling(plane);
            let sb_w = (self.mi_cols - mi_col) >> sub_x;
            let sb_h = (self.mi_rows - mi_row) >> sub_y;
            let sb_x = SB4 >> sub_x;
            let sb_y = SB4 >> sub_y;
            // Prediction stops at the tile. The row above this superblock is
            // already decoded, but it must not include the next tile's columns.
            let tile_x1 = (self.tile_col_end >> sub_x) as i32;
            let tile_y1 = (self.tile_row_end >> sub_y) as i32;
            let origin_x = (mi_col >> sub_x) as i32;
            let origin_y = (mi_row >> sub_y) as i32;
            for y in -1..=sb_y as i32 {
                for x in -1..=sb_x as i32 {
                    let above_ok = mi_row > self.tile_row;
                    let left_ok = mi_col > self.tile_col;
                    let col_in_tile = origin_x + x < tile_x1;
                    let row_in_tile = origin_y + y < tile_y1;
                    let flag = if (y < 0 && x < sb_w as i32 && above_ok && col_in_tile)
                        || (x < 0 && y < sb_h as i32 && left_ok && row_in_tile)
                    {
                        1
                    } else {
                        0
                    };
                    self.set_decoded(plane, y, x, flag);
                }
            }
            self.set_decoded(plane, sb_y as i32, -1, 0);
        }
    }

    fn set_decoded(&mut self, plane: usize, y: i32, x: i32, flag: u8) {
        let stride = 40;
        self.decoded[plane][((y + 1) as usize) * stride + (x + 1) as usize] = flag;
    }

    fn decoded_at(&self, plane: usize, y: i32, x: i32) -> bool {
        let stride = 40;
        self.decoded[plane][((y + 1) as usize) * stride + (x + 1) as usize] != 0
    }

    fn subsampling(&self, plane: usize) -> (usize, usize) {
        if plane == 0 {
            (0, 0)
        } else {
            (usize::from(self.sub_x), usize::from(self.sub_y))
        }
    }

    fn decode_partition(&mut self, row: usize, col: usize, bsize: usize) -> Result<(), CodecError> {
        if row >= self.mi_rows || col >= self.mi_cols {
            return Ok(());
        }
        let num4 = NUM_4X4_WIDE[bsize];
        let half = num4 >> 1;
        let has_rows = row + half < self.mi_rows;
        let has_cols = col + half < self.mi_cols;
        let partition = self.read_partition(row, col, bsize, has_rows, has_cols)?;
        let sub = partition_subsize(partition, bsize)
            .ok_or(CodecError::Unsupported { what: "partition" })?;
        if partition == PARTITION_NONE {
            self.decode_block(row, col, sub)?;
        } else if partition == PARTITION_HORZ {
            self.decode_block(row, col, sub)?;
            if has_rows {
                self.decode_block(row + half, col, sub)?;
            }
        } else if partition == PARTITION_VERT {
            self.decode_block(row, col, sub)?;
            if has_cols {
                self.decode_block(row, col + half, sub)?;
            }
        } else if partition == PARTITION_SPLIT {
            self.decode_partition(row, col, sub)?;
            self.decode_partition(row, col + half, sub)?;
            self.decode_partition(row + half, col, sub)?;
            self.decode_partition(row + half, col + half, sub)?;
        } else {
            self.place_uneven(partition, row, col, bsize, sub, half)?;
        }
        Ok(())
    }

    fn place_uneven(
        &mut self,
        partition: u32,
        row: usize,
        col: usize,
        bsize: usize,
        sub: usize,
        half: usize,
    ) -> Result<(), CodecError> {
        let split = partition_subsize(PARTITION_SPLIT, bsize).ok_or(CodecError::Malformed {
            at_bit: 0,
            what: "partition",
        })?;
        let quarter = half >> 1;
        if partition == PARTITION_HORZ_A {
            self.decode_block(row, col, split)?;
            self.decode_block(row, col + half, split)?;
            self.decode_block(row + half, col, sub)?;
        } else if partition == PARTITION_HORZ_B {
            self.decode_block(row, col, sub)?;
            self.decode_block(row + half, col, split)?;
            self.decode_block(row + half, col + half, split)?;
        } else if partition == PARTITION_VERT_A {
            self.decode_block(row, col, split)?;
            self.decode_block(row + half, col, split)?;
            self.decode_block(row, col + half, sub)?;
        } else if partition == PARTITION_VERT_B {
            self.decode_block(row, col, sub)?;
            self.decode_block(row, col + half, split)?;
            self.decode_block(row + half, col + half, split)?;
        } else if partition == PARTITION_HORZ_4 {
            for step in 0..4 {
                let at = row + quarter * step;
                if step < 3 || at < self.mi_rows {
                    self.decode_block(at, col, sub)?;
                }
            }
        } else if partition == PARTITION_VERT_4 {
            for step in 0..4 {
                let at = col + quarter * step;
                if step < 3 || at < self.mi_cols {
                    self.decode_block(row, at, sub)?;
                }
            }
        }
        Ok(())
    }

    fn read_partition(
        &mut self,
        row: usize,
        col: usize,
        bsize: usize,
        has_rows: bool,
        has_cols: bool,
    ) -> Result<u32, CodecError> {
        if bsize < BLOCK_8X8 {
            return Ok(PARTITION_NONE);
        }
        let bsl = MI_WIDTH_LOG2[bsize];
        let above = self.above_mi(row)
            && MI_WIDTH_LOG2[self.sizes[(row - 1) * self.mi_cols + col] as usize] < bsl;
        let left = self.left_mi(col)
            && MI_HEIGHT_LOG2[self.sizes[row * self.mi_cols + col - 1] as usize] < bsl;
        let ctx = usize::from(left) * 2 + usize::from(above);
        let shape = partition_shape(bsl);
        let slot = bsl.saturating_sub(1).min(4);
        if has_rows && has_cols {
            return read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.partition[slot],
                shape,
                &[ctx],
            );
        }
        if !has_rows && !has_cols {
            return Ok(PARTITION_SPLIT);
        }
        let mut built = {
            let table = &self.cdfs.partition[slot];
            let (start, len) = row_range(table.len(), shape, &[ctx])?;
            edge_partition_cdf(&table[start..start + len], bsize, has_cols)
        };
        let symbol = self.symbols.read(&mut built, false)?;
        if has_cols {
            Ok(if symbol == 0 {
                PARTITION_HORZ
            } else {
                PARTITION_SPLIT
            })
        } else {
            Ok(if symbol == 0 {
                PARTITION_VERT
            } else {
                PARTITION_SPLIT
            })
        }
    }

    fn decode_block(&mut self, row: usize, col: usize, bsize: usize) -> Result<(), CodecError> {
        if self.inter_frame {
            self.decode_inter_frame_block(row, col, bsize)
        } else {
            self.decode_intra_frame_block(row, col, bsize)
        }
    }

    fn decode_intra_frame_block(
        &mut self,
        row: usize,
        col: usize,
        bsize: usize,
    ) -> Result<(), CodecError> {
        let bw4 = NUM_4X4_WIDE[bsize];
        let bh4 = NUM_4X4_HIGH[bsize];
        let has_chroma = self.has_chroma(row, col, bw4, bh4);
        let avail_u = self.above_mi(row);
        let avail_l = self.left_mi(col);
        let skip_ctx = self.neighbor_sum(row, col);
        let skip = self.read_skip(skip_ctx)?;
        self.read_cdef(row, col, skip)?;
        self.read_quantizer_delta(bsize, skip)?;
        let y_mode = self.read_y_mode(row, col, avail_u, avail_l)?;
        let angle_y = self.read_angle(y_mode, bsize, true)?;
        let (uv_mode, angle_uv) = if has_chroma {
            let mode = self.read_uv_mode(bsize, y_mode)?;
            if mode == UV_CFL_PRED {
                (self.cfl_u, self.cfl_v) = self.read_cfl_alphas()?;
            }
            let angle = self.read_angle(mode, bsize, false)?;
            (mode, angle)
        } else {
            (DC_PRED, 0)
        };
        self.read_palette_gate(bsize)?;
        self.filter_mode = self.read_filter_intra(bsize, y_mode)?;
        let tx = self.read_tx_size(row, col, bsize, skip, false)?;
        if skip {
            self.reset_context(row, col, bw4, bh4, has_chroma);
        }
        let coded = CodedBlock {
            row,
            col,
            bw4,
            bh4,
            bsize,
            y_mode,
            uv_mode,
            angle_y,
            angle_uv,
            tx,
            skip,
            has_chroma,
            avail_u,
            avail_l,
            inter: false,
            var_tx: false,
            mv: Motion::default(),
            mv1: Motion::default(),
            ref0: 0,
            ref1: NONE_FRAME,
            ii: -1,
            interp: [0, 0],
            compound_idx: 1,
            compound_mask: 0,
            skip_mode: false,
            motion_mode: MOTION_SIMPLE,
        };
        self.fill_grid(&coded);
        self.predict_and_residual(&coded)?;
        Ok(())
    }

    fn decode_inter_frame_block(
        &mut self,
        row: usize,
        col: usize,
        bsize: usize,
    ) -> Result<(), CodecError> {
        let bw4 = NUM_4X4_WIDE[bsize];
        let bh4 = NUM_4X4_HIGH[bsize];
        let has_chroma = self.has_chroma(row, col, bw4, bh4);
        let avail_u = self.above_mi(row);
        let avail_l = self.left_mi(col);
        let skip_mode = self.read_block_skip_mode(row, col, bsize)?;
        let skip = if skip_mode {
            true
        } else {
            self.read_skip(self.neighbor_sum(row, col))?
        };
        self.read_cdef(row, col, skip)?;
        self.read_quantizer_delta(bsize, skip)?;
        self.filter_mode = None;
        let inter = if skip_mode {
            true
        } else {
            self.read_is_inter(row, col)?
        };
        let chosen = if inter {
            self.read_inter_block(row, col, bsize, skip_mode)?
        } else {
            self.read_intra_in_inter(bsize, has_chroma)?
        };
        let (tx, var_tx) = self.read_block_tx_size(row, col, bsize, skip, inter)?;
        if skip {
            self.reset_context(row, col, bw4, bh4, has_chroma);
        }
        let coded = CodedBlock {
            row,
            col,
            bw4,
            bh4,
            bsize,
            y_mode: chosen.y_mode,
            uv_mode: chosen.uv_mode,
            angle_y: chosen.angle_y,
            angle_uv: chosen.angle_uv,
            tx,
            skip,
            has_chroma,
            avail_u,
            avail_l,
            inter,
            var_tx,
            mv: chosen.mv,
            mv1: chosen.mv1,
            ref0: chosen.ref0,
            ref1: chosen.ref1,
            ii: chosen.ii,
            interp: chosen.interp,
            compound_idx: chosen.compound_idx,
            compound_mask: chosen.compound_mask,
            skip_mode,
            motion_mode: chosen.motion_mode,
        };
        self.fill_grid(&coded);
        self.predict_and_residual(&coded)
    }

    fn read_is_inter(&mut self, row: usize, col: usize) -> Result<bool, CodecError> {
        let ctx = self.is_inter_ctx(row, col);
        Ok(read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.is_inter,
            &[4, 3],
            &[ctx],
        )? == 1)
    }

    fn is_inter_ctx(&self, row: usize, col: usize) -> usize {
        let above = self.above_mi(row);
        let left = self.left_mi(col);
        let above_intra = above && self.inter_flag[(row - 1) * self.mi_cols + col] == 0;
        let left_intra = left && self.inter_flag[row * self.mi_cols + col - 1] == 0;
        if above && left {
            if left_intra && above_intra {
                3
            } else {
                usize::from(left_intra || above_intra)
            }
        } else if above || left {
            2 * usize::from(if above { above_intra } else { left_intra })
        } else {
            0
        }
    }

    fn read_intra_in_inter(
        &mut self,
        bsize: usize,
        has_chroma: bool,
    ) -> Result<Chosen, CodecError> {
        let ctx = SIZE_GROUP.get(bsize).copied().unwrap_or(0);
        let y_mode = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.intra_y,
            &[4, 14],
            &[ctx],
        )? as u8;
        let angle_y = self.read_angle(y_mode, bsize, true)?;
        let (uv_mode, angle_uv) = if has_chroma {
            let mode = self.read_uv_mode(bsize, y_mode)?;
            if mode == UV_CFL_PRED {
                (self.cfl_u, self.cfl_v) = self.read_cfl_alphas()?;
            }
            (mode, self.read_angle(mode, bsize, false)?)
        } else {
            (DC_PRED, 0)
        };
        self.read_palette_gate(bsize)?;
        self.filter_mode = self.read_filter_intra(bsize, y_mode)?;
        Ok(Chosen {
            y_mode,
            uv_mode,
            angle_y,
            angle_uv,
            mv: Motion::default(),
            mv1: Motion::default(),
            ref0: 0,
            ref1: NONE_FRAME,
            ii: -1,
            interp: [0, 0],
            compound_idx: 1,
            compound_mask: 0,
            motion_mode: MOTION_SIMPLE,
        })
    }

    fn read_block_skip_mode(
        &mut self,
        row: usize,
        col: usize,
        bsize: usize,
    ) -> Result<bool, CodecError> {
        if !self.skip_present || block_width(bsize) < 8 || block_height(bsize) < 8 {
            return Ok(false);
        }
        let mut ctx = 0;
        if self.above_mi(row) {
            ctx += usize::from(self.skip_modes[(row - 1) * self.mi_cols + col]);
        }
        if self.left_mi(col) {
            ctx += usize::from(self.skip_modes[row * self.mi_cols + col - 1]);
        }
        Ok(read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.skip_mode,
            &[3, 3],
            &[ctx],
        )? == 1)
    }

    fn read_inter_block(
        &mut self,
        row: usize,
        col: usize,
        bsize: usize,
        skip_mode: bool,
    ) -> Result<Chosen, CodecError> {
        let bw4 = NUM_4X4_WIDE[bsize];
        let bh4 = NUM_4X4_HIGH[bsize];
        let (reference, reference1) = if skip_mode {
            (self.skip_ref[0], self.skip_ref[1])
        } else {
            self.read_refs(row, col, bw4, bh4)?
        };
        let compound = reference1 > INTRA_FRAME;
        let global = self.global_mv(reference, row, col, bsize);
        let global1 = if compound {
            self.global_mv(reference1, row, col, bsize)
        } else {
            Motion::default()
        };
        let stack = find_stack(
            &MotionGrid {
                cols: self.mi_cols,
                rows: self.mi_rows,
                inter: &self.inter_flag,
                refs: &self.block_ref,
                refs1: &self.block_ref1,
                mvs: &self.mvs,
                mvs1: &self.mvs1,
                modes: &self.y_mode,
                sizes: &self.sizes,
            },
            &BlockPlace {
                row,
                col,
                bw4,
                bh4,
                bsize,
                reference,
                reference1,
                compound,
                global1,
                affine1: self.gm_affine(reference1),
                bias: self.sign_bias,
            },
            global,
            self.allow_hp,
            self.force_integer,
            self.gm_affine(reference),
            self.use_ref_mvs.then_some(&self.motion_field),
        );
        let y_mode = if skip_mode {
            NEAREST_NEARESTMV
        } else if compound {
            self.read_compound_mode(&stack)?
        } else {
            self.read_inter_mode(&stack)?
        };
        let mv_index = self.read_drl(y_mode, stack.found, &stack)?;
        let (mv, mv1) = self.assign_mvs(y_mode, mv_index, &stack, global, global1)?;
        let ii = if compound {
            -1
        } else {
            self.read_interintra(bsize)?
        };
        let motion_mode = self.read_motion_mode(MotionQuery {
            row,
            col,
            bsize,
            mode: y_mode,
            reference,
            reference1: if ii >= 0 { INTRA_FRAME } else { reference1 },
            mv,
            skip_mode,
        })?;
        let compound_choice = if skip_mode || !compound {
            CompoundChoice::average()
        } else {
            self.read_compound_choice(row, col, bsize, reference, reference1)?
        };
        // A skip-mode block carries no interpolation symbol. The frame filter
        // is used, or eight-tap regular when that filter is switchable.
        let interp = if skip_mode {
            if self.filter_switchable {
                [0, 0]
            } else {
                [self.interpolation, self.interpolation]
            }
        } else {
            self.read_interp(InterpQuery {
                row,
                col,
                bsize,
                mode: y_mode,
                reference,
                reference1,
                motion_mode,
            })?
        };
        Ok(Chosen {
            y_mode,
            uv_mode: DC_PRED,
            angle_y: 0,
            angle_uv: 0,
            mv,
            mv1,
            ref0: reference,
            ref1: if ii >= 0 { INTRA_FRAME } else { reference1 },
            ii,
            interp,
            compound_idx: compound_choice.idx,
            compound_mask: compound_choice.pack(),
            motion_mode,
        })
    }

    /// `read_compound_type`. A mask is a wedge or a difference weight. Otherwise the block averages or uses distance.
    fn read_compound_choice(
        &mut self,
        row: usize,
        col: usize,
        bsize: usize,
        reference: i8,
        reference1: i8,
    ) -> Result<CompoundChoice, CodecError> {
        if !self.masked_compound {
            let idx = if self.distance_compound {
                self.read_compound_idx(row, col, reference, reference1)?
            } else {
                1
            };
            return Ok(CompoundChoice::from_index(idx));
        }
        let ctx = self.comp_group_ctx(row, col);
        let group = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.comp_group,
            &[6, 3],
            &[ctx],
        )? as u8;
        if group == 0 {
            let idx = if self.distance_compound {
                self.read_compound_idx(row, col, reference, reference1)?
            } else {
                1
            };
            return Ok(CompoundChoice {
                idx,
                group: 0,
                kind: 0,
                wedge: 0,
                sign: 0,
                mask_type: 0,
            });
        }
        let bits = wedge_bits(bsize);
        if bits == 0 {
            let mask_type = u8::from(self.symbols.read_bool()?);
            return Ok(CompoundChoice {
                idx: 1,
                group,
                kind: MASK_DIFF,
                wedge: 0,
                sign: 0,
                mask_type,
            });
        }
        let symbol = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.compound_type,
            &[22, 3],
            &[bsize],
        )?;
        if symbol == 0 {
            let wedge = read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.wedge_index,
                &[22, 17],
                &[bsize],
            )? as u8;
            let sign = u8::from(self.symbols.read_bool()?);
            Ok(CompoundChoice {
                idx: 1,
                group,
                kind: MASK_WEDGE,
                wedge,
                sign,
                mask_type: 0,
            })
        } else {
            let mask_type = u8::from(self.symbols.read_bool()?);
            Ok(CompoundChoice {
                idx: 1,
                group,
                kind: MASK_DIFF,
                wedge: 0,
                sign: 0,
                mask_type,
            })
        }
    }

    fn comp_group_ctx(&self, row: usize, col: usize) -> usize {
        let mut ctx = 0;
        if self.above_mi(row) {
            ctx += self.group_neighbor((row - 1) * self.mi_cols + col);
        }
        if self.left_mi(col) {
            ctx += self.group_neighbor(row * self.mi_cols + col - 1);
        }
        ctx.min(5)
    }

    fn group_neighbor(&self, at: usize) -> usize {
        if self.block_ref1.get(at).copied().unwrap_or(NONE_FRAME) > INTRA_FRAME {
            usize::from(self.compound_mask.get(at).copied().unwrap_or(0) & 1)
        } else if self.block_ref.get(at).copied().unwrap_or(NONE_FRAME) == ALTREF_FRAME {
            3
        } else {
            0
        }
    }

    fn gm_affine(&self, reference: i8) -> bool {
        let index = usize::try_from(reference).unwrap_or(0);
        self.gm_type.get(index).copied().unwrap_or(0) > GM_TRANSLATION
    }

    fn read_compound_idx(
        &mut self,
        row: usize,
        col: usize,
        reference: i8,
        reference1: i8,
    ) -> Result<u8, CodecError> {
        let ctx = self.compound_idx_ctx(row, col, reference, reference1);
        Ok(read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.compound_idx,
            &[6, 3],
            &[ctx],
        )? as u8)
    }

    fn compound_idx_ctx(&self, row: usize, col: usize, reference: i8, reference1: i8) -> usize {
        let same =
            self.frame_dist(reference).unsigned_abs() == self.frame_dist(reference1).unsigned_abs();
        let mut ctx = if same { 3 } else { 0 };
        if self.above_mi(row) {
            ctx += self.compound_neighbor((row - 1) * self.mi_cols + col);
        }
        if self.left_mi(col) {
            ctx += self.compound_neighbor(row * self.mi_cols + col - 1);
        }
        ctx
    }

    fn compound_neighbor(&self, at: usize) -> usize {
        if self.block_ref1.get(at).copied().unwrap_or(NONE_FRAME) > INTRA_FRAME {
            usize::from(self.compound_idx.get(at).copied().unwrap_or(0))
        } else if self.block_ref.get(at).copied().unwrap_or(NONE_FRAME) == ALTREF_FRAME {
            1
        } else {
            0
        }
    }

    fn frame_dist(&self, reference: i8) -> i32 {
        let index = usize::try_from(reference).unwrap_or(0);
        if index == 0 || index > 7 {
            return 0;
        }
        let slot = usize::from(self.frame_refs[index - 1]);
        crate::header::relative_dist(self.slot_hints[slot], self.order_hint, self.order_hint_bits)
    }

    fn distance_weights(&self, reference: i8, reference1: i8) -> (i32, i32) {
        let d1 = clipped_dist(self.frame_dist(reference));
        let d0 = clipped_dist(self.frame_dist(reference1));
        let order = d0 <= d1;
        let band = if d0 == 0 || d1 == 0 {
            3
        } else {
            let mut band = 3;
            for (index, row) in DIST_WEIGHT.iter().enumerate().take(3) {
                let c0 = row[usize::from(order)];
                let c1 = row[usize::from(!order)];
                let left = d0 * c0;
                let right = d1 * c1;
                let split = if order { left > right } else { left < right };
                if split {
                    band = index;
                    break;
                }
            }
            band
        };
        let side = usize::from(order);
        (
            DIST_LOOKUP[band][side],
            DIST_LOOKUP[band][usize::from(!order)],
        )
    }

    fn read_refs(
        &mut self,
        row: usize,
        col: usize,
        bw4: usize,
        bh4: usize,
    ) -> Result<(i8, i8), CodecError> {
        let ctx = self.comp_mode_ctx(row, col);
        let compound = self.reference_select && bw4.min(bh4) >= 2 && {
            read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.comp_mode,
                &[5, 3],
                &[ctx],
            )? == 1
        };
        if compound {
            self.read_compound_refs(row, col)
        } else {
            Ok((self.read_single_ref(row, col)?, NONE_FRAME))
        }
    }

    fn read_compound_refs(&mut self, row: usize, col: usize) -> Result<(i8, i8), CodecError> {
        let ctx = self.comp_ref_type_ctx(row, col);
        let unidir = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.comp_ref_type,
            &[5, 3],
            &[ctx],
        )? == 0;
        if unidir {
            self.read_unidir_refs(row, col)
        } else {
            self.read_bidir_refs(row, col)
        }
    }

    fn read_unidir_refs(&mut self, row: usize, col: usize) -> Result<(i8, i8), CodecError> {
        let ctx = self.single_ref_ctx(row, col, 0);
        if self.uni_bit(0, ctx)? {
            return Ok((BWDREF_FRAME, ALTREF_FRAME));
        }
        let ctx = self.uni_p1_ctx(row, col);
        if self.uni_bit(1, ctx)? {
            let ctx = self.single_ref_ctx(row, col, 4);
            if self.uni_bit(2, ctx)? {
                Ok((LAST_FRAME, GOLDEN_FRAME))
            } else {
                Ok((LAST_FRAME, LAST3_FRAME))
            }
        } else {
            Ok((LAST_FRAME, LAST2_FRAME))
        }
    }

    fn read_bidir_refs(&mut self, row: usize, col: usize) -> Result<(i8, i8), CodecError> {
        let ctx = self.single_ref_ctx(row, col, 2);
        let reference = if self.comp_fwd_bit(0, ctx)? {
            let ctx = self.single_ref_ctx(row, col, 4);
            if self.comp_fwd_bit(2, ctx)? {
                GOLDEN_FRAME
            } else {
                LAST3_FRAME
            }
        } else {
            let ctx = self.single_ref_ctx(row, col, 3);
            if self.comp_fwd_bit(1, ctx)? {
                LAST2_FRAME
            } else {
                LAST_FRAME
            }
        };
        let ctx = self.single_ref_ctx(row, col, 1);
        let reference1 = if self.comp_bwd_bit(0, ctx)? {
            ALTREF_FRAME
        } else {
            let ctx = self.single_ref_ctx(row, col, 5);
            if self.comp_bwd_bit(1, ctx)? {
                ALTREF2_FRAME
            } else {
                BWDREF_FRAME
            }
        };
        Ok((reference, reference1))
    }

    fn uni_bit(&mut self, which: usize, ctx: usize) -> Result<bool, CodecError> {
        Ok(read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.uni_comp_ref,
            &[3, 3, 3],
            &[ctx, which],
        )? == 1)
    }

    fn comp_fwd_bit(&mut self, which: usize, ctx: usize) -> Result<bool, CodecError> {
        Ok(read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.comp_ref,
            &[3, 3, 3],
            &[ctx, which],
        )? == 1)
    }

    fn comp_bwd_bit(&mut self, which: usize, ctx: usize) -> Result<bool, CodecError> {
        Ok(read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.comp_bwdref,
            &[3, 2, 3],
            &[ctx, which],
        )? == 1)
    }

    fn uni_p1_ctx(&self, row: usize, col: usize) -> usize {
        let last2 = self.count_refs(row, col, LAST2_FRAME);
        let later =
            self.count_refs(row, col, LAST3_FRAME) + self.count_refs(row, col, GOLDEN_FRAME);
        ref_count_ctx(last2, later)
    }

    fn comp_mode_ctx(&self, row: usize, col: usize) -> usize {
        let above = self.above_mi(row);
        let left = self.left_mi(col);
        let (above_ref, _, above_intra, above_single) = self.side_refs(above, row, col, true);
        let (left_ref, _, left_intra, left_single) = self.side_refs(left, row, col, false);
        if above && left {
            if above_single && left_single {
                usize::from(backward(above_ref)) ^ usize::from(backward(left_ref))
            } else if above_single {
                2 + usize::from(backward(above_ref) || above_intra)
            } else if left_single {
                2 + usize::from(backward(left_ref) || left_intra)
            } else {
                4
            }
        } else if above {
            if above_single {
                usize::from(backward(above_ref))
            } else {
                3
            }
        } else if left {
            if left_single {
                usize::from(backward(left_ref))
            } else {
                3
            }
        } else {
            1
        }
    }

    fn side_refs(&self, avail: bool, row: usize, col: usize, above: bool) -> (i8, i8, bool, bool) {
        if !avail {
            return (INTRA_FRAME, NONE_FRAME, true, true);
        }
        let at = if above {
            (row - 1) * self.mi_cols + col
        } else {
            row * self.mi_cols + col - 1
        };
        let first = self.block_ref[at];
        let second = self.block_ref1[at];
        (first, second, first <= INTRA_FRAME, second <= INTRA_FRAME)
    }

    fn comp_ref_type_ctx(&self, row: usize, col: usize) -> usize {
        let above = self.above_mi(row);
        let left = self.left_mi(col);
        let (above0, above1, above_intra, above_single) = self.side_refs(above, row, col, true);
        let (left0, left1, left_intra, left_single) = self.side_refs(left, row, col, false);
        let above_comp = above && !above_intra && !above_single;
        let left_comp = left && !left_intra && !left_single;
        let above_uni = above_comp && same_direction(above0, above1);
        let left_uni = left_comp && same_direction(left0, left1);
        if above && !above_intra && left && !left_intra {
            let same = same_direction(above0, left0);
            if !above_comp && !left_comp {
                1 + 2 * usize::from(same)
            } else if !above_comp {
                if left_uni { 3 + usize::from(same) } else { 1 }
            } else if !left_comp {
                if above_uni { 3 + usize::from(same) } else { 1 }
            } else if !above_uni && !left_uni {
                0
            } else if !above_uni || !left_uni {
                2
            } else {
                3 + usize::from((above0 == BWDREF_FRAME) == (left0 == BWDREF_FRAME))
            }
        } else if above && left {
            if above_comp {
                1 + 2 * usize::from(above_uni)
            } else if left_comp {
                1 + 2 * usize::from(left_uni)
            } else {
                2
            }
        } else if above_comp {
            4 * usize::from(above_uni)
        } else if left_comp {
            4 * usize::from(left_uni)
        } else {
            2
        }
    }

    fn read_compound_mode(&mut self, stack: &Candidates) -> Result<u8, CodecError> {
        const MAP: [[usize; 5]; 3] = [[0, 1, 1, 1, 1], [1, 2, 3, 4, 4], [4, 4, 5, 6, 7]];
        let row = (stack.ref_ctx >> 1).min(2);
        let col = stack.new_ctx.min(4);
        let ctx = MAP[row][col];
        let mode = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.compound_mode,
            &[8, 9],
            &[ctx],
        )?;
        Ok(NEAREST_NEARESTMV + mode as u8)
    }

    fn read_single_ref(&mut self, row: usize, col: usize) -> Result<i8, CodecError> {
        let p1 = self.ref_bit(row, col, 0)?;
        if p1 {
            let p2 = self.ref_bit(row, col, 1)?;
            if !p2 {
                let p6 = self.ref_bit(row, col, 5)?;
                Ok(if p6 { 6 } else { 5 })
            } else {
                Ok(7)
            }
        } else {
            let p3 = self.ref_bit(row, col, 2)?;
            if p3 {
                let p5 = self.ref_bit(row, col, 4)?;
                Ok(if p5 { 4 } else { 3 })
            } else {
                let p4 = self.ref_bit(row, col, 3)?;
                Ok(if p4 { 2 } else { LAST_FRAME })
            }
        }
    }

    fn ref_bit(&mut self, row: usize, col: usize, which: usize) -> Result<bool, CodecError> {
        let ctx = self.single_ref_ctx(row, col, which);
        Ok(read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.single_ref,
            &[3, 6, 3],
            &[ctx, which],
        )? == 1)
    }

    fn single_ref_ctx(&self, row: usize, col: usize, which: usize) -> usize {
        let count = |frame: i8| self.count_refs(row, col, frame);
        let (left, right) = match which {
            0 => (
                count(1) + count(2) + count(3) + count(4),
                count(5) + count(6) + count(7),
            ),
            1 => (count(5) + count(6), count(7)),
            2 => (count(1) + count(2), count(3) + count(4)),
            3 => (count(1), count(2)),
            4 => (count(3), count(4)),
            _ => (count(5), count(6)),
        };
        if left < right {
            0
        } else if left == right {
            1
        } else {
            2
        }
    }

    fn count_refs(&self, row: usize, col: usize, frame: i8) -> usize {
        let mut count = 0;
        if self.above_mi(row) {
            let at = (row - 1) * self.mi_cols + col;
            count += usize::from(self.block_ref[at] == frame);
            count += usize::from(self.block_ref1[at] == frame);
        }
        if self.left_mi(col) {
            let at = row * self.mi_cols + col - 1;
            count += usize::from(self.block_ref[at] == frame);
            count += usize::from(self.block_ref1[at] == frame);
        }
        count
    }

    fn read_inter_mode(&mut self, stack: &Candidates) -> Result<u8, CodecError> {
        let new_mv = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.new_mv,
            &[6, 3],
            &[stack.new_ctx],
        )?;
        if new_mv == 0 {
            return Ok(NEWMV);
        }
        let zero_mv = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.zero_mv,
            &[2, 3],
            &[stack.zero_ctx],
        )?;
        if zero_mv == 0 {
            return Ok(GLOBALMV);
        }
        let ref_mv = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.ref_mv,
            &[6, 3],
            &[stack.ref_ctx],
        )?;
        Ok(if ref_mv == 0 { NEARESTMV } else { NEARMV })
    }

    fn read_drl(
        &mut self,
        mode: u8,
        found: usize,
        stack: &Candidates,
    ) -> Result<usize, CodecError> {
        if mode == NEWMV || mode == crate::inter::NEW_NEWMV {
            let mut index = 0usize;
            for idx in 0..2 {
                if found > idx + 1 {
                    let bit = self.drl_bit(stack.drl[idx])?;
                    if bit == 0 {
                        index = idx;
                        break;
                    }
                    index = idx + 1;
                }
            }
            return Ok(index);
        }
        if !has_near_mv(mode) {
            return Ok(0);
        }
        let mut index = 1usize;
        for idx in 1..3 {
            if found > idx + 1 {
                let bit = self.drl_bit(stack.drl[idx])?;
                if bit == 0 {
                    index = idx;
                    break;
                }
                index = idx + 1;
            }
        }
        Ok(index)
    }

    fn drl_bit(&mut self, ctx: usize) -> Result<u32, CodecError> {
        read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.drl,
            &[3, 3],
            &[ctx],
        )
    }

    fn global_mv(&self, reference: i8, row: usize, col: usize, bsize: usize) -> Motion {
        let index = usize::try_from(reference).unwrap_or(0);
        let kind = self.gm_type.get(index).copied().unwrap_or(0);
        let params = self.gm_params.get(index).copied().unwrap_or([0; 6]);
        project_global(
            kind,
            params,
            row,
            col,
            bsize,
            self.allow_hp,
            self.force_integer,
        )
    }

    /// Interintra is a single-reference blend. A wedge mask stays unsupported.
    fn read_interintra(&mut self, bsize: usize) -> Result<i8, CodecError> {
        if !self.interintra || !(3..=9).contains(&bsize) {
            return Ok(-1);
        }
        let ctx = SIZE_GROUP
            .get(bsize)
            .copied()
            .unwrap_or(1)
            .saturating_sub(1);
        let on = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.inter_intra,
            &[3, 3],
            &[ctx],
        )? == 1;
        if !on {
            return Ok(-1);
        }
        let mode = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.inter_intra_mode,
            &[3, 5],
            &[ctx],
        )? as i8;
        let wedge = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.wedge_interintra,
            &[22, 3],
            &[bsize],
        )? == 1;
        if wedge {
            return Err(CodecError::Unsupported {
                what: "interintra wedge",
            });
        }
        Ok(mode)
    }

    fn assign_mvs(
        &mut self,
        mode: u8,
        index: usize,
        stack: &Candidates,
        global: Motion,
        global1: Motion,
    ) -> Result<(Motion, Motion), CodecError> {
        let first = self.assign_one(mode, 0, index, stack, global)?;
        let second = if mode >= NEAREST_NEARESTMV {
            self.assign_one(mode, 1, index, stack, global1)?
        } else {
            Motion::default()
        };
        Ok((first, second))
    }

    fn assign_one(
        &mut self,
        y_mode: u8,
        list: usize,
        index: usize,
        stack: &Candidates,
        global: Motion,
    ) -> Result<Motion, CodecError> {
        let mode = list_mode(y_mode, list);
        let mut pos = if mode == NEARESTMV { 0 } else { index };
        if mode == NEWMV && stack.found <= 1 {
            pos = 0;
        }
        let pred = if mode == GLOBALMV {
            global
        } else if list == 0 {
            stack.mv[pos.min(7)]
        } else {
            stack.mv1[pos.min(7)]
        };
        if mode != NEWMV {
            return Ok(pred);
        }
        let diff = self.read_mv_delta()?;
        let mv = Motion {
            row: pred.row + diff.row,
            col: pred.col + diff.col,
        };
        if mv.row.abs() >= (1 << 14) || mv.col.abs() >= (1 << 14) {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "motion vector",
            });
        }
        Ok(mv)
    }

    fn read_mv_delta(&mut self) -> Result<Motion, CodecError> {
        let joint = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.mv_joint,
            &[5],
            &[],
        )?;
        let row = if joint == 2 || joint == 3 {
            self.read_mv_component(0)?
        } else {
            0
        };
        let col = if joint == 1 || joint == 3 {
            self.read_mv_component(1)?
        } else {
            0
        };
        Ok(Motion { row, col })
    }

    fn read_mv_component(&mut self, comp: usize) -> Result<i32, CodecError> {
        let sign = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.mv_sign,
            &[2, 3],
            &[comp],
        )? == 1;
        let class = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.mv_class,
            &[2, 12],
            &[comp],
        )? as i32;
        let mag = if class == 0 {
            self.class0_mag(comp)?
        } else {
            self.class_mag(comp, class)?
        };
        Ok(if sign { -mag } else { mag })
    }

    fn class0_mag(&mut self, comp: usize) -> Result<i32, CodecError> {
        let bit = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.mv_class0_bit,
            &[2, 3],
            &[comp],
        )?;
        let fr = if self.force_integer {
            3
        } else {
            read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.mv_class0_fr,
                &[2, 2, 5],
                &[comp, bit as usize],
            )?
        };
        let hp = if self.allow_hp {
            read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.mv_class0_hp,
                &[2, 3],
                &[comp],
            )?
        } else {
            1
        };
        Ok((((bit << 3) | (fr << 1) | hp) + 1) as i32)
    }

    fn class_mag(&mut self, comp: usize, class: i32) -> Result<i32, CodecError> {
        let mut bits = 0i32;
        for i in 0..class {
            let bit = read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.mv_bit,
                &[2, 10, 3],
                &[comp, i as usize],
            )?;
            bits |= (bit as i32) << i;
        }
        let fr = if self.force_integer {
            3
        } else {
            read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.mv_fr,
                &[2, 5],
                &[comp],
            )?
        };
        let hp = if self.allow_hp {
            read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.mv_hp,
                &[2, 3],
                &[comp],
            )?
        } else {
            1
        };
        Ok((1 << (class + 3)) + (((bits << 3) | ((fr as i32) << 1) | hp as i32) + 1))
    }

    /// `read_motion_mode`. A block that cannot warp reads the overlapped-motion bit.
    fn read_motion_mode(&mut self, query: MotionQuery) -> Result<u8, CodecError> {
        if query.skip_mode || !self.motion_switchable {
            return Ok(MOTION_SIMPLE);
        }
        if block_width(query.bsize).min(block_height(query.bsize)) < 8 {
            return Ok(MOTION_SIMPLE);
        }
        if !self.force_integer
            && (query.mode == GLOBALMV || query.mode == GLOBAL_GLOBALMV)
            && self.gm_kind(query.reference) > GM_TRANSLATION
        {
            return Ok(MOTION_SIMPLE);
        }
        let compound = query.reference1 > INTRA_FRAME;
        if compound
            || query.reference1 == INTRA_FRAME
            || !self.overlappable(query.row, query.col, query.bsize)
        {
            return Ok(MOTION_SIMPLE);
        }
        let samples = self.warp_samples(&query);
        let warp_ok = !self.force_integer
            && samples.count > 0
            && self.allow_warped
            && !self.ref_is_scaled(query.reference);
        if !warp_ok {
            let use_obmc = read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.obmc,
                &[22, 3],
                &[query.bsize],
            )? == 1;
            return Ok(if use_obmc { MOTION_OBMC } else { MOTION_SIMPLE });
        }
        let mode = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.motion_mode,
            &[22, 4],
            &[query.bsize],
        )? as u8;
        Ok(mode)
    }

    fn overlappable(&self, row: usize, col: usize, bsize: usize) -> bool {
        let w4 = NUM_4X4_WIDE[bsize];
        let h4 = NUM_4X4_HIGH[bsize];
        if self.above_mi(row) {
            let mut x4 = col;
            let end = self.mi_cols.min(col + w4);
            while x4 < end {
                let cand = x4 | 1;
                if cand < self.mi_cols {
                    let at = (row - 1) * self.mi_cols + cand;
                    if self.block_ref.get(at).copied().unwrap_or(INTRA_FRAME) > INTRA_FRAME {
                        return true;
                    }
                }
                x4 += 2;
            }
        }
        if self.left_mi(col) {
            let mut y4 = row;
            let end = self.mi_rows.min(row + h4);
            while y4 < end {
                let cand = y4 | 1;
                if cand < self.mi_rows {
                    let at = cand * self.mi_cols + col - 1;
                    if self.block_ref.get(at).copied().unwrap_or(INTRA_FRAME) > INTRA_FRAME {
                        return true;
                    }
                }
                y4 += 2;
            }
        }
        false
    }

    fn ref_is_scaled(&self, reference: i8) -> bool {
        let Ok(index) = usize::try_from(reference) else {
            return false;
        };
        if !(1..=7).contains(&index) {
            return false;
        }
        let slot = usize::from(self.frame_refs[index - 1]);
        slot < 8
            && self.ref_ok[slot]
            && (self.ref_w[slot] != self.visible_w || self.ref_h[slot] != self.visible_h)
    }

    /// `find_warp_samples`. The count is zero when no neighbor shares this reference.
    fn warp_samples(&self, query: &MotionQuery) -> WarpSamples {
        let mut scan = WarpSamples::default();
        let w4 = NUM_4X4_WIDE[query.bsize];
        let h4 = NUM_4X4_HIGH[query.bsize];
        let mut top_left = true;
        let mut top_right = true;
        let threshold = i32::try_from(
            block_width(query.bsize)
                .max(block_height(query.bsize))
                .clamp(16, 112),
        )
        .unwrap_or(112);
        if self.above_mi(query.row) {
            let src_w = NUM_4X4_WIDE[self.bsize_at(query.row - 1, query.col)];
            if w4 <= src_w {
                let offset = -((query.col as isize) & (src_w as isize - 1));
                if offset < 0 {
                    top_left = false;
                }
                if offset + src_w as isize > w4 as isize {
                    top_right = false;
                }
                self.add_warp_sample(&mut scan, query, -1, 0, threshold);
            } else {
                let mut along = 0;
                let limit = w4.min(self.mi_cols.saturating_sub(query.col));
                while along < limit {
                    let src_w = NUM_4X4_WIDE[self.bsize_at(query.row - 1, query.col + along)];
                    let step = w4.min(src_w).max(1);
                    self.add_warp_sample(&mut scan, query, -1, along as isize, threshold);
                    along += step;
                }
            }
        }
        if self.left_mi(query.col) {
            let src_h = NUM_4X4_HIGH[self.bsize_at(query.row, query.col - 1)];
            if h4 <= src_h {
                let offset = -((query.row as isize) & (src_h as isize - 1));
                if offset < 0 {
                    top_left = false;
                }
                self.add_warp_sample(&mut scan, query, 0, -1, threshold);
            } else {
                let mut along = 0;
                let limit = h4.min(self.mi_rows.saturating_sub(query.row));
                while along < limit {
                    let src_h = NUM_4X4_HIGH[self.bsize_at(query.row + along, query.col - 1)];
                    let step = h4.min(src_h).max(1);
                    self.add_warp_sample(&mut scan, query, along as isize, -1, threshold);
                    along += step;
                }
            }
        }
        if top_left {
            self.add_warp_sample(&mut scan, query, -1, -1, threshold);
        }
        if top_right && w4.max(h4) <= 16 {
            self.add_warp_sample(&mut scan, query, -1, w4 as isize, threshold);
        }
        if scan.count == 0 && scan.scanned > 0 {
            scan.count = 1;
        }
        scan
    }

    fn add_warp_sample(
        &self,
        scan: &mut WarpSamples,
        query: &MotionQuery,
        delta_row: isize,
        delta_col: isize,
        threshold: i32,
    ) {
        if scan.scanned >= 8 {
            return;
        }
        let mv_row = query.row as isize + delta_row;
        let mv_col = query.col as isize + delta_col;
        if mv_row < 0
            || mv_col < 0
            || mv_row >= self.mi_rows as isize
            || mv_col >= self.mi_cols as isize
        {
            return;
        }
        let at = mv_row as usize * self.mi_cols + mv_col as usize;
        if self.block_ref.get(at).copied().unwrap_or(NONE_FRAME) != query.reference
            || self.block_ref1.get(at).copied().unwrap_or(NONE_FRAME) != NONE_FRAME
        {
            return;
        }
        let cand_sz = usize::from(self.sizes.get(at).copied().unwrap_or(0));
        if cand_sz >= NUM_4X4_WIDE.len() {
            return;
        }
        let cand_w = NUM_4X4_WIDE[cand_sz];
        let cand_h = NUM_4X4_HIGH[cand_sz];
        if cand_w == 0 || cand_h == 0 {
            return;
        }
        let cand_row = (mv_row as usize) & !(cand_h - 1);
        let cand_col = (mv_col as usize) & !(cand_w - 1);
        let origin = cand_row * self.mi_cols + cand_col;
        let Some(neighbor) = self.mvs.get(origin).copied() else {
            return;
        };
        let mid_y = (cand_row * 4 + cand_h * 2 - 1) as i32;
        let mid_x = (cand_col * 4 + cand_w * 2 - 1) as i32;
        let valid =
            (neighbor.row - query.mv.row).abs() + (neighbor.col - query.mv.col).abs() <= threshold;
        scan.scanned += 1;
        if !valid && scan.scanned > 1 {
            return;
        }
        if scan.count >= scan.list.len() {
            return;
        }
        scan.list[scan.count] = [
            mid_y.saturating_mul(8),
            mid_x.saturating_mul(8),
            mid_y.saturating_mul(8).saturating_add(neighbor.row),
            mid_x.saturating_mul(8).saturating_add(neighbor.col),
        ];
        if valid {
            scan.count += 1;
        }
    }

    fn read_interp(&mut self, query: InterpQuery) -> Result<[u8; 2], CodecError> {
        if !self.filter_switchable {
            return Ok([self.interpolation, self.interpolation]);
        }
        let dirs = if self.dual_filter { 2 } else { 1 };
        let mut filter = [0u8; 2];
        for (dir, slot) in filter.iter_mut().enumerate().take(dirs) {
            if self.needs_interp(&query) {
                let ctx = self.interp_ctx(
                    query.row,
                    query.col,
                    dir,
                    query.reference,
                    query.reference1 > INTRA_FRAME,
                );
                *slot = read_sym(
                    &mut self.symbols,
                    self.update,
                    &mut self.cdfs.interp_filter,
                    &[16, 4],
                    &[ctx],
                )? as u8;
            }
        }
        if !self.dual_filter {
            filter[1] = filter[0];
        }
        Ok(filter)
    }

    fn needs_interp(&self, query: &InterpQuery) -> bool {
        if query.motion_mode == MOTION_LOCAL_WARP {
            return false;
        }
        let large = block_width(query.bsize).min(block_height(query.bsize)) >= 8;
        if large && query.mode == GLOBALMV {
            return self.gm_kind(query.reference) == GM_TRANSLATION;
        }
        if large && query.mode == GLOBAL_GLOBALMV {
            return self.gm_kind(query.reference) == GM_TRANSLATION
                || self.gm_kind(query.reference1) == GM_TRANSLATION;
        }
        true
    }

    fn gm_kind(&self, reference: i8) -> u8 {
        let index = usize::try_from(reference).unwrap_or(0);
        self.gm_type.get(index).copied().unwrap_or(0)
    }

    fn interp_ctx(
        &self,
        row: usize,
        col: usize,
        dir: usize,
        reference: i8,
        compound: bool,
    ) -> usize {
        let mut ctx = ((dir & 1) * 2 + usize::from(compound)) * 4;
        let mut left = 3usize;
        let mut above = 3usize;
        if self.left_mi(col) {
            let at = row * self.mi_cols + col - 1;
            if self.block_ref[at] == reference || self.block_ref1[at] == reference {
                left = usize::from(self.interp[at][dir]);
            }
        }
        if self.above_mi(row) {
            let at = (row - 1) * self.mi_cols + col;
            if self.block_ref[at] == reference || self.block_ref1[at] == reference {
                above = usize::from(self.interp[at][dir]);
            }
        }
        if left == above {
            ctx += left;
        } else if left == 3 {
            ctx += above;
        } else if above == 3 {
            ctx += left;
        } else {
            ctx += 3;
        }
        ctx
    }

    fn predict_inter_plane(&mut self, block: &Block) -> Result<(), CodecError> {
        let at = block.row * self.mi_cols + block.col;
        let frame_ref = self.block_ref[at];
        if frame_ref < LAST_FRAME {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "missing reference",
            });
        }
        let slot = usize::from(self.frame_refs[frame_ref as usize - 1]);
        if slot >= 8 || !self.ref_ok[slot] {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "missing reference",
            });
        }
        let mv = self.mvs[at];
        let filter = self.interp[at];
        let ref_w = self.ref_w[slot];
        let ref_h = self.ref_h[slot];
        let frame_w = self.visible_w;
        let frame_h = self.visible_h;
        let (sub_x, sub_y) = (block.sub_x != 0, block.sub_y != 0);
        let x = (block.col >> block.sub_x) * 4;
        let y = (block.row >> block.sub_y) * 4;
        let w = block_width(block.bsize) >> block.sub_x;
        let h = block_height(block.bsize) >> block.sub_y;
        let second = self.block_ref1[at];
        if second > INTRA_FRAME {
            return self.predict_compound(
                block,
                frame_ref,
                second,
                PlaneRect {
                    x,
                    y,
                    w,
                    h,
                    sub_x,
                    sub_y,
                },
            );
        }
        let plane_w = if block.plane == 0 {
            ref_w
        } else {
            ref_w.div_ceil(2)
        };
        let plane_h = if block.plane == 0 {
            ref_h
        } else {
            ref_h.div_ceil(2)
        };
        let ii = self.ii_mode.get(at).copied().unwrap_or(-1);
        let gm_index = usize::try_from(frame_ref).unwrap_or(0);
        let gm_kind = self.gm_type.get(gm_index).copied().unwrap_or(0);
        let gm_params = self.gm_params.get(gm_index).copied().unwrap_or([0; 6]);
        let local = if w >= 8 && h >= 8 && !self.force_integer {
            self.local_model(block.row, block.col, block.bsize, frame_ref, mv)
        } else {
            None
        };
        let global = block.y_mode == GLOBALMV
            && gm_kind > GM_TRANSLATION
            && w >= 8
            && h >= 8
            && !self.force_integer
            && global_shear_valid(gm_params);
        let warp_params = local.or(global.then_some(gm_params));
        if ii >= 0 {
            self.predict_block_intra(block, interintra_intra_mode(ii as u8), x, y, w, h)?;
            let mut packed = vec![0u8; w.saturating_mul(h)];
            let reference = self.ref_plane(slot, block.plane);
            let mut sample = InterSample {
                dst: &mut packed,
                stride: w,
                x,
                y,
                w,
                h,
                sub_x,
                sub_y,
                mv,
                filter,
                reference,
                ref_stride: plane_w,
                plane_w,
                plane_h,
                frame_w,
                frame_h,
                ref_w,
                ref_h,
                write_x: 0,
                write_y: 0,
            };
            if let Some(params) = warp_params {
                predict_warp(&mut sample, params)?;
            } else {
                predict(&mut sample)?;
            }
            self.blend_interintra(block.plane, ii as u8, (x, y, w, h), &packed);
            return Ok(());
        }
        let stride = if block.plane == 0 {
            self.width
        } else {
            self.chroma_w
        };
        let reference: &[u8] = match block.plane {
            0 => self.ref_y[slot],
            1 => self.ref_u[slot],
            _ => self.ref_v[slot],
        };
        {
            let dst: &mut [u8] = if block.plane == 0 {
                &mut self.y
            } else if block.plane == 1 {
                &mut self.u
            } else {
                &mut self.v
            };
            let mut sample = InterSample {
                dst,
                stride,
                x,
                y,
                w,
                h,
                sub_x,
                sub_y,
                mv,
                filter,
                reference,
                ref_stride: plane_w,
                plane_w,
                plane_h,
                frame_w,
                frame_h,
                ref_w,
                ref_h,
                write_x: x,
                write_y: y,
            };
            if let Some(params) = warp_params {
                predict_warp(&mut sample, params)?;
            } else {
                predict(&mut sample)?;
            }
        }
        if self.motion_at(block.row, block.col) == MOTION_OBMC {
            self.overlapped(block)?;
        }
        Ok(())
    }

    fn motion_at(&self, row: usize, col: usize) -> u8 {
        self.mi_at(row, col)
            .and_then(|at| self.motion_modes.get(at).copied())
            .unwrap_or(MOTION_SIMPLE)
    }

    fn local_model(
        &self,
        row: usize,
        col: usize,
        bsize: usize,
        reference: i8,
        mv: Motion,
    ) -> Option<[i32; 6]> {
        if self.motion_at(row, col) != MOTION_LOCAL_WARP {
            return None;
        }
        let query = MotionQuery {
            row,
            col,
            bsize,
            mode: MOTION_SIMPLE,
            reference,
            reference1: NONE_FRAME,
            mv,
            skip_mode: false,
        };
        let samples = self.warp_samples(&query);
        if samples.count == 0 {
            return None;
        }
        let mid_y = (row * 4 + NUM_4X4_HIGH[bsize] * 2 - 1) as i32;
        let mid_x = (col * 4 + NUM_4X4_WIDE[bsize] * 2 - 1) as i32;
        let params = local_warp_params(&samples.list[..samples.count], mv, mid_x, mid_y)?;
        global_shear_valid(params).then_some(params)
    }

    /// Overlapped motion. Above runs only when the plane block is at least 8×8.
    fn overlapped(&mut self, block: &Block) -> Result<(), CodecError> {
        let sub_x = block.sub_x;
        let sub_y = block.sub_y;
        let w = block_width(block.bsize) >> block.sub_x;
        let h = block_height(block.bsize) >> block.sub_y;
        let plane_bsize = if block.plane == 0 {
            block.bsize
        } else {
            subsampled_size(block.bsize, self.sub_x, self.sub_y)
        };
        if self.above_mi(block.row) && plane_bsize >= BLOCK_8X8 {
            let w4 = NUM_4X4_WIDE[block.bsize];
            let mut x4 = block.col;
            let mut count = 0;
            let limit = 4.min(MI_WIDTH_LOG2[block.bsize]);
            let end = self.mi_cols.min(block.col + w4);
            while count < limit && x4 < end {
                let cand_col = x4 | 1;
                let cand_row = block.row - 1;
                let step4 = self.neighbor_step(cand_row, cand_col, true);
                if self.neighbor_inter(cand_row, cand_col) {
                    count += 1;
                    let pred_w = w.min((step4 * 4) >> sub_x);
                    let pred_h = (h / 2).min(32 >> sub_y);
                    self.blend_overlap(
                        block,
                        Overlap {
                            pass: 0,
                            cand_row,
                            cand_col,
                            pred_w,
                            pred_h,
                            x4,
                            y4: block.row,
                        },
                    )?;
                }
                x4 += step4;
            }
        }
        if self.left_mi(block.col) {
            let h4 = NUM_4X4_HIGH[block.bsize];
            let mut y4 = block.row;
            let mut count = 0;
            let limit = 4.min(MI_HEIGHT_LOG2[block.bsize]);
            let end = self.mi_rows.min(block.row + h4);
            while count < limit && y4 < end {
                let cand_row = y4 | 1;
                let cand_col = block.col - 1;
                let step4 = self.neighbor_step(cand_row, cand_col, false);
                if self.neighbor_inter(cand_row, cand_col) {
                    count += 1;
                    let pred_w = (w / 2).min(32 >> sub_x);
                    let pred_h = h.min((step4 * 4) >> sub_y);
                    self.blend_overlap(
                        block,
                        Overlap {
                            pass: 1,
                            cand_row,
                            cand_col,
                            pred_w,
                            pred_h,
                            x4: block.col,
                            y4,
                        },
                    )?;
                }
                y4 += step4;
            }
        }
        Ok(())
    }

    fn neighbor_step(&self, row: usize, col: usize, wide: bool) -> usize {
        let Some(at) = self.mi_at(row, col) else {
            return 2;
        };
        let cand = usize::from(self.sizes.get(at).copied().unwrap_or(0));
        let blocks = if wide {
            NUM_4X4_WIDE.get(cand).copied().unwrap_or(1)
        } else {
            NUM_4X4_HIGH.get(cand).copied().unwrap_or(1)
        };
        blocks.clamp(2, 16)
    }

    fn neighbor_inter(&self, row: usize, col: usize) -> bool {
        self.mi_at(row, col)
            .is_some_and(|at| self.block_ref.get(at).copied().unwrap_or(INTRA_FRAME) > INTRA_FRAME)
    }

    fn blend_overlap(&mut self, block: &Block, overlap: Overlap) -> Result<(), CodecError> {
        if overlap.pred_w == 0 || overlap.pred_h == 0 {
            return Ok(());
        }
        let Some(at) = self.mi_at(overlap.cand_row, overlap.cand_col) else {
            return Ok(());
        };
        let reference = self.block_ref.get(at).copied().unwrap_or(NONE_FRAME);
        if reference < LAST_FRAME {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "missing reference",
            });
        }
        let slot = usize::from(self.frame_refs[reference as usize - 1]);
        if slot >= 8 || !self.ref_ok[slot] {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "missing reference",
            });
        }
        let mv = self.mvs.get(at).copied().unwrap_or_default();
        let filter = self.interp.get(at).copied().unwrap_or([0, 0]);
        let sub_x = block.sub_x != 0;
        let sub_y = block.sub_y != 0;
        let pred_x = (overlap.x4 * 4) >> block.sub_x;
        let pred_y = (overlap.y4 * 4) >> block.sub_y;
        let packed = self.predict_neighbor(NeighborPred {
            slot,
            plane: block.plane,
            mv,
            filter,
            x: pred_x,
            y: pred_y,
            w: overlap.pred_w,
            h: overlap.pred_h,
            sub_x,
            sub_y,
        })?;
        let length = if overlap.pass == 0 {
            overlap.pred_h
        } else {
            overlap.pred_w
        };
        let stride = if block.plane == 0 {
            self.width
        } else {
            self.chroma_w
        };
        let dst: &mut [u8] = if block.plane == 0 {
            &mut self.y
        } else if block.plane == 1 {
            &mut self.u
        } else {
            &mut self.v
        };
        for row in 0..overlap.pred_h {
            for col in 0..overlap.pred_w {
                let along = if overlap.pass == 0 { row } else { col };
                let weight = u32::from(obmc_weight(length, along));
                let dx = pred_x + col;
                let dy = pred_y + row;
                if dx >= stride {
                    continue;
                }
                let dest = dy * stride + dx;
                let Some(slot) = dst.get_mut(dest) else {
                    continue;
                };
                let pred = u32::from(*packed.get(row * overlap.pred_w + col).unwrap_or(&0));
                let curr = u32::from(*slot);
                *slot = ((weight * curr + (64 - weight) * pred + 32) >> 6) as u8;
            }
        }
        Ok(())
    }

    fn predict_neighbor(&self, pred: NeighborPred) -> Result<Vec<u8>, CodecError> {
        let ref_w = self.ref_w[pred.slot];
        let ref_h = self.ref_h[pred.slot];
        let plane_w = if pred.plane == 0 {
            ref_w
        } else {
            ref_w.div_ceil(2)
        };
        let plane_h = if pred.plane == 0 {
            ref_h
        } else {
            ref_h.div_ceil(2)
        };
        let reference = self.ref_plane(pred.slot, pred.plane);
        let mut packed = vec![0u8; pred.w.saturating_mul(pred.h)];
        let mut sample = InterSample {
            dst: &mut packed,
            stride: pred.w,
            x: pred.x,
            y: pred.y,
            w: pred.w,
            h: pred.h,
            sub_x: pred.sub_x,
            sub_y: pred.sub_y,
            mv: pred.mv,
            filter: pred.filter,
            reference,
            ref_stride: plane_w,
            plane_w,
            plane_h,
            frame_w: self.visible_w,
            frame_h: self.visible_h,
            ref_w,
            ref_h,
            write_x: 0,
            write_y: 0,
        };
        predict(&mut sample)?;
        Ok(packed)
    }

    fn predict_compound(
        &mut self,
        block: &Block,
        reference: i8,
        reference1: i8,
        place: PlaneRect,
    ) -> Result<(), CodecError> {
        let PlaneRect {
            x,
            y,
            w,
            h,
            sub_x,
            sub_y,
        } = place;
        let n = w.saturating_mul(h);
        let mut first = vec![0i32; n];
        let mut second = vec![0i32; n];
        self.predict_list(block, reference, 0, &mut first, place)?;
        self.predict_list(block, reference1, 1, &mut second, place)?;
        let at = block.row * self.mi_cols + block.col;
        let packed = self.compound_mask.get(at).copied().unwrap_or(0);
        let kind = mask_kind(packed);
        let weights = (kind == 0 && self.compound_idx.get(at).copied().unwrap_or(1) == 0)
            .then(|| self.distance_weights(reference, reference1));
        if kind != 0 && block.plane == 0 {
            self.fill_blend_mask(w, h, packed, &first, &second);
        } else if kind != 0 && !self.blend_ready {
            self.build_luma_mask(block, reference, reference1, packed)?;
        }
        let luma_w = if self.blend_w == 0 {
            block_width(block.bsize)
        } else {
            self.blend_w
        };
        let mut blended = vec![0u8; n];
        for row in 0..h {
            for col in 0..w {
                let sample = row * w + col;
                blended[sample] = if kind == 0 {
                    if let Some((fwd, bck)) = weights {
                        distance_compound(first[sample], second[sample], fwd, bck)
                    } else {
                        average_compound(first[sample], second[sample])
                    }
                } else {
                    let weight =
                        compound_mask_weight(&self.blend_mask, luma_w, col, row, sub_x, sub_y);
                    mask_compound(first[sample], second[sample], weight)
                };
            }
        }
        let stride = if block.plane == 0 {
            self.width
        } else {
            self.chroma_w
        };
        let dst: &mut [u8] = if block.plane == 0 {
            &mut self.y
        } else if block.plane == 1 {
            &mut self.u
        } else {
            &mut self.v
        };
        for row in 0..h {
            for col in 0..w {
                let dx = x + col;
                let dy = y + row;
                if dx < stride {
                    let dest = dy * stride + dx;
                    if let Some(slot) = dst.get_mut(dest) {
                        *slot = blended[row * w + col];
                    }
                }
            }
        }
        Ok(())
    }

    fn fill_blend_mask(&mut self, w: usize, h: usize, packed: u16, first: &[i32], second: &[i32]) {
        let n = w.saturating_mul(h);
        self.blend_mask.resize(n, 0);
        if mask_kind(packed) == MASK_WEDGE {
            wedge_mask(
                w,
                h,
                usize::from(mask_wedge(packed)),
                mask_sign(packed),
                &mut self.blend_mask,
            );
        } else {
            difference_mask(first, second, w, h, mask_type(packed), &mut self.blend_mask);
        }
        self.blend_w = w;
        self.blend_ready = true;
    }

    fn build_luma_mask(
        &mut self,
        block: &Block,
        reference: i8,
        reference1: i8,
        packed: u16,
    ) -> Result<(), CodecError> {
        let w = block_width(block.bsize);
        let h = block_height(block.bsize);
        let n = w.saturating_mul(h);
        let luma = Block {
            plane: 0,
            row: block.row,
            col: block.col,
            bsize: block.bsize,
            y_mode: block.y_mode,
            uv_mode: block.uv_mode,
            angle_y: block.angle_y,
            angle_uv: block.angle_uv,
            tx: block.tx,
            skip: block.skip,
            avail_u: block.avail_u,
            avail_l: block.avail_l,
            x4: 0,
            y4: 0,
            sub_x: 0,
            sub_y: 0,
            step_x: block.step_x,
            step_y: block.step_y,
            inter: true,
        };
        let place = PlaneRect {
            x: block.col * 4,
            y: block.row * 4,
            w,
            h,
            sub_x: false,
            sub_y: false,
        };
        let mut first = vec![0i32; n];
        let mut second = vec![0i32; n];
        self.predict_list(&luma, reference, 0, &mut first, place)?;
        self.predict_list(&luma, reference1, 1, &mut second, place)?;
        self.fill_blend_mask(w, h, packed, &first, &second);
        Ok(())
    }

    fn predict_list(
        &mut self,
        block: &Block,
        reference: i8,
        list: usize,
        dst: &mut [i32],
        place: PlaneRect,
    ) -> Result<(), CodecError> {
        let PlaneRect {
            x,
            y,
            w,
            h,
            sub_x,
            sub_y,
        } = place;
        if reference < LAST_FRAME {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "missing reference",
            });
        }
        let slot = usize::from(self.frame_refs[reference as usize - 1]);
        if slot >= 8 || !self.ref_ok[slot] {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "missing reference",
            });
        }
        let at = block.row * self.mi_cols + block.col;
        let mv = if list == 0 {
            self.mvs[at]
        } else {
            self.mvs1[at]
        };
        let ref_w = self.ref_w[slot];
        let ref_h = self.ref_h[slot];
        let plane_w = if block.plane == 0 {
            ref_w
        } else {
            ref_w.div_ceil(2)
        };
        let plane_h = if block.plane == 0 {
            ref_h
        } else {
            ref_h.div_ceil(2)
        };
        let mode = list_mode(block.y_mode, list);
        let gm_index = usize::try_from(reference).unwrap_or(0);
        let gm_kind = self.gm_type.get(gm_index).copied().unwrap_or(0);
        let gm_params = self.gm_params.get(gm_index).copied().unwrap_or([0; 6]);
        let warp = mode == GLOBALMV
            && gm_kind > GM_TRANSLATION
            && w >= 8
            && h >= 8
            && !self.force_integer
            && global_shear_valid(gm_params);
        let reference_plane = self.ref_plane(slot, block.plane);
        if warp {
            let mut packed = vec![0u8; w.saturating_mul(h)];
            let mut sample = InterSample {
                dst: &mut packed,
                stride: w,
                x,
                y,
                w,
                h,
                sub_x,
                sub_y,
                mv,
                filter: self.interp[at],
                reference: reference_plane,
                ref_stride: plane_w,
                plane_w,
                plane_h,
                frame_w: self.visible_w,
                frame_h: self.visible_h,
                ref_w,
                ref_h,
                write_x: 0,
                write_y: 0,
            };
            predict_warp(&mut sample, gm_params)?;
            for (slot, pixel) in dst.iter_mut().zip(packed) {
                *slot = i32::from(pixel) << 4;
            }
            return Ok(());
        }
        let mut unused = [];
        let sample = InterSample {
            dst: &mut unused,
            stride: w,
            x,
            y,
            w,
            h,
            sub_x,
            sub_y,
            mv,
            filter: self.interp[at],
            reference: reference_plane,
            ref_stride: plane_w,
            plane_w,
            plane_h,
            frame_w: self.visible_w,
            frame_h: self.visible_h,
            ref_w,
            ref_h,
            write_x: 0,
            write_y: 0,
        };
        predict_precise(&sample, dst)
    }

    fn ref_plane(&self, slot: usize, plane: usize) -> &'r [u8] {
        match plane {
            0 => self.ref_y[slot],
            1 => self.ref_u[slot],
            _ => self.ref_v[slot],
        }
    }

    /// Intra half of an interintra block, at the plane's own resolution.
    fn predict_block_intra(
        &mut self,
        block: &Block,
        mode: u8,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    ) -> Result<(), CodecError> {
        let (sub_x, sub_y) = self.subsampling(block.plane);
        let row_mask = (SB4 >> sub_y) - 1;
        let col_mask = (SB4 >> sub_x) - 1;
        let local_row = ((y >> 2) & row_mask) as i32;
        let local_col = ((x >> 2) & col_mask) as i32;
        let above_right = self.decoded_at(block.plane, local_row - 1, local_col + (w / 4) as i32);
        let below_left = self.decoded_at(block.plane, local_row + (h / 4) as i32, local_col - 1);
        let filter_type = self.smooth_neighbor(block.plane, block.row, block.col);
        let edge_filter = self.edge_filter;
        let (plane_w, plane_h, stride) = if block.plane == 0 {
            (self.width, self.height, self.width)
        } else {
            (self.chroma_w, self.chroma_h, self.chroma_w)
        };
        let pred = IntraPred {
            x,
            y,
            w,
            h,
            have_left: block.avail_l && x > 0,
            have_above: block.avail_u && y > 0,
            have_above_right: above_right,
            have_below_left: below_left,
            mode,
            angle_delta: 0,
            edge_filter,
            filter_type,
            max_x: plane_w.saturating_sub(1),
            max_y: plane_h.saturating_sub(1),
            filter: None,
        };
        let plane = if block.plane == 0 {
            &mut self.y
        } else if block.plane == 1 {
            &mut self.u
        } else {
            &mut self.v
        };
        predict_intra(plane, stride, &pred, &mut self.predict)
    }

    /// `COMPOUND_INTRA` blend. The mask is built at this plane's resolution.
    fn blend_interintra(
        &mut self,
        plane: usize,
        mode: u8,
        place: (usize, usize, usize, usize),
        inter: &[u8],
    ) {
        let (x, y, w, h) = place;
        let stride = if plane == 0 {
            self.width
        } else {
            self.chroma_w
        };
        let dst = if plane == 0 {
            &mut self.y
        } else if plane == 1 {
            &mut self.u
        } else {
            &mut self.v
        };
        let scale = 128 / h.max(w).max(1);
        for row in 0..h {
            for col in 0..w {
                let Some(sample) = inter.get(row * w + col).copied() else {
                    continue;
                };
                let at = (y + row) * stride + x + col;
                let Some(slot) = dst.get_mut(at) else {
                    continue;
                };
                let weight = interintra_weight(mode, row, col, scale);
                let intra = i32::from(*slot);
                let mixed = weight * intra + (64 - weight) * i32::from(sample);
                *slot = ((mixed + 32) >> 6).clamp(0, 255) as u8;
            }
        }
    }

    fn inter_tx(&mut self, plane: usize, tx: usize) -> Result<u8, CodecError> {
        let set = inter_tx_set(tx, self.reduced_tx);
        if set == 0 || self.base_q == 0 {
            return self.remember_tx(plane, tx, DCT_DCT);
        }
        if plane > 0 {
            let stored = self.luma_tx_kind();
            let kind = if inter_set_allows(set, stored) {
                stored
            } else {
                DCT_DCT
            };
            return Ok(kind);
        }
        let symbol = match set {
            1 => read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.inter_tx1,
                &[2, 17],
                &[TX_SIZE_SQR[tx]],
            )?,
            2 => read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.inter_tx2,
                &[13],
                &[],
            )?,
            _ => read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.inter_tx3,
                &[4, 3],
                &[TX_SIZE_SQR[tx]],
            )?,
        };
        let kind = inter_tx_type(set, symbol).ok_or(CodecError::Malformed {
            at_bit: 0,
            what: "tx type",
        })?;
        self.remember_tx(plane, tx, kind)
    }

    fn luma_tx_kind(&self) -> u8 {
        if self.cur_luma_row >= self.mi_rows || self.cur_luma_col >= self.mi_cols {
            return DCT_DCT;
        }
        self.tx_kind[self.cur_luma_row * self.mi_cols + self.cur_luma_col]
    }

    fn remember_tx(&mut self, plane: usize, tx: usize, kind: u8) -> Result<u8, CodecError> {
        if plane == 0 {
            let w4 = TX_WIDTH[tx] >> 2;
            let h4 = TX_HEIGHT[tx] >> 2;
            for y in 0..h4 {
                for x in 0..w4 {
                    let r = self.cur_luma_row + y;
                    let c = self.cur_luma_col + x;
                    if r < self.mi_rows && c < self.mi_cols {
                        self.tx_kind[r * self.mi_cols + c] = kind;
                    }
                }
            }
        }
        Ok(kind)
    }

    fn read_cdef(&mut self, row: usize, col: usize, skip: bool) -> Result<(), CodecError> {
        if skip || self.lossless || !self.cdef_on {
            return Ok(());
        }
        let aligned_row = row & !15;
        let aligned_col = col & !15;
        let at = aligned_row * self.mi_cols + aligned_col;
        if self.cdef_idx.get(at).copied().unwrap_or(0) >= 0 {
            return Ok(());
        }
        let index = self.symbols.read_literal(u32::from(self.cdef.bits))? as i8;
        if let Some(slot) = self.cdef_idx.get_mut(at) {
            *slot = index;
        }
        Ok(())
    }

    /// Superblock quantizer delta. The first coded block of the superblock
    /// reads it. A skipped 64×64 block leaves the running index unchanged.
    fn read_quantizer_delta(&mut self, bsize: usize, skip: bool) -> Result<(), CodecError> {
        let full_superblock = bsize == 12 || bsize == BLOCK_128X128;
        if self.read_deltas && !(full_superblock && skip) {
            let mut magnitude = read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.delta_q,
                &[5],
                &[0],
            )?;
            if magnitude == 3 {
                let extra = self.symbols.read_literal(3)? + 1;
                magnitude = self.symbols.read_literal(extra)? + (1 << extra) + 1;
            }
            if magnitude != 0 {
                let sign = i32::from(self.symbols.read_bool()?);
                let reduced = if sign == 1 {
                    -i32::try_from(magnitude).unwrap_or(0)
                } else {
                    i32::try_from(magnitude).unwrap_or(0)
                };
                let shifted = reduced << self.delta_q_res;
                let next = i32::from(self.current_q) + shifted;
                self.current_q = next.clamp(1, 255) as u8;
            }
        }
        self.read_deltas = false;
        Ok(())
    }

    /// Palette is signaled before filter intra when screen tools are on.
    fn read_palette_gate(&self, bsize: usize) -> Result<(), CodecError> {
        if self.screen
            && bsize >= BLOCK_8X8
            && block_width(bsize) <= 64
            && block_height(bsize) <= 64
        {
            return Err(CodecError::Unsupported { what: "palette" });
        }
        Ok(())
    }

    fn read_filter_intra(&mut self, bsize: usize, y_mode: u8) -> Result<Option<u8>, CodecError> {
        if !self.filter_intra
            || y_mode != DC_PRED
            || block_width(bsize).max(block_height(bsize)) > 32
        {
            return Ok(None);
        }
        let enabled = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.use_filter_intra,
            &[22, 3],
            &[bsize, 0],
        )? == 1;
        if !enabled {
            return Ok(None);
        }
        let mode = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.filter_intra,
            &[6],
            &[0],
        )?;
        u8::try_from(mode)
            .map(Some)
            .map_err(|_| CodecError::Malformed {
                at_bit: 0,
                what: "filter intra",
            })
    }

    fn read_skip(&mut self, ctx: usize) -> Result<bool, CodecError> {
        Ok(read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.skip,
            &[3, 3],
            &[ctx],
        )? == 1)
    }

    fn neighbor_sum(&self, row: usize, col: usize) -> usize {
        let mut ctx = 0;
        if self.above_mi(row) {
            ctx += usize::from(self.skips[(row - 1) * self.mi_cols + col]);
        }
        if self.left_mi(col) {
            ctx += usize::from(self.skips[row * self.mi_cols + col - 1]);
        }
        ctx
    }

    fn read_y_mode(
        &mut self,
        row: usize,
        col: usize,
        above: bool,
        left: bool,
    ) -> Result<u8, CodecError> {
        let above_mode = if above {
            self.y_mode[(row - 1) * self.mi_cols + col]
        } else {
            DC_PRED
        };
        let left_mode = if left {
            self.y_mode[row * self.mi_cols + col - 1]
        } else {
            DC_PRED
        };
        let a = MODE_CTX[above_mode as usize];
        let l = MODE_CTX[left_mode as usize];
        let mode = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.y_mode,
            &[5, 5, 14],
            &[a, l],
        )?;
        Ok(mode as u8)
    }

    fn read_uv_mode(&mut self, bsize: usize, y_mode: u8) -> Result<u8, CodecError> {
        let large = block_width(bsize).max(block_height(bsize)) > 32 || self.lossless;
        let mode = if large {
            read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.uv_no,
                &[13, 14],
                &[y_mode as usize],
            )?
        } else {
            read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.uv_yes,
                &[13, 15],
                &[y_mode as usize],
            )?
        };
        Ok(mode as u8)
    }

    fn read_cfl_alphas(&mut self) -> Result<(i32, i32), CodecError> {
        let signs = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.cfl_sign,
            &[9],
            &[],
        )?;
        let sign_u = (signs + 1) / 3;
        let sign_v = (signs + 1) % 3;
        let alpha = |ctx: u32,
                     sign: u32,
                     symbols: &mut SymbolDecoder<'_>,
                     update: bool,
                     table: &mut [u16]| {
            if sign == 0 {
                return Ok(0);
            }
            let symbol = read_sym(symbols, update, table, &[6, 17], &[ctx as usize])?;
            let magnitude = symbol as i32 + 1;
            Ok(if sign == 1 { -magnitude } else { magnitude })
        };
        let u_ctx = signs.saturating_sub(2);
        let v_ctx = [0, 3, 0, 1, 4, 0, 2, 5][signs as usize];
        let u = alpha(
            u_ctx,
            sign_u,
            &mut self.symbols,
            self.update,
            &mut self.cdfs.cfl_alpha,
        )?;
        let v = alpha(
            v_ctx,
            sign_v,
            &mut self.symbols,
            self.update,
            &mut self.cdfs.cfl_alpha,
        )?;
        Ok((u, v))
    }

    fn read_angle(&mut self, mode: u8, bsize: usize, _luma: bool) -> Result<i32, CodecError> {
        if bsize >= BLOCK_8X8 && (1..=8).contains(&mode) {
            let symbol = read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.angle,
                &[8, 8],
                &[(mode - 1) as usize],
            )?;
            Ok(symbol as i32 - 3)
        } else {
            Ok(0)
        }
    }

    fn read_tx_size(
        &mut self,
        row: usize,
        col: usize,
        bsize: usize,
        skip: bool,
        is_inter: bool,
    ) -> Result<usize, CodecError> {
        if self.lossless {
            return Ok(TX_4X4);
        }
        let max_rect = MAX_TX_SIZE_RECT[bsize];
        let allow = !skip || !is_inter;
        if bsize > 0 && self.tx_select && allow {
            let depth_max = MAX_TX_DEPTH[bsize];
            let which = match depth_max {
                4 => 3,
                3 => 2,
                2 => 1,
                _ => 0,
            };
            let ctx = self.tx_depth_ctx(row, col, max_rect);
            let shape: &[usize] = if which == 0 { &[3, 3] } else { &[3, 4] };
            let depth = read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.tx[which],
                shape,
                &[ctx],
            )? as usize;
            let mut tx = max_rect;
            for _ in 0..depth {
                tx = SPLIT_TX_SIZE[tx];
            }
            return Ok(tx);
        }
        Ok(max_rect)
    }

    /// Section 5.11.34. An inter block that is not skipped stores a transform tree.
    /// Every other block stores one size.
    fn read_block_tx_size(
        &mut self,
        row: usize,
        col: usize,
        bsize: usize,
        skip: bool,
        inter: bool,
    ) -> Result<(usize, bool), CodecError> {
        self.tx_origins.clear();
        let variable = self.tx_select && bsize > 0 && inter && !skip && !self.lossless;
        if !variable {
            let tx = self.read_tx_size(row, col, bsize, skip, inter)?;
            return Ok((tx, false));
        }
        let max_tx = MAX_TX_SIZE_RECT[bsize];
        let step_x = (TX_WIDTH[max_tx] / 4).max(1);
        let step_y = (TX_HEIGHT[max_tx] / 4).max(1);
        let bw4 = NUM_4X4_WIDE[bsize];
        let bh4 = NUM_4X4_HIGH[bsize];
        let mut y = row;
        while y < row + bh4 {
            let mut x = col;
            while x < col + bw4 {
                self.read_var_tx_size(y, x, max_tx, 0, (row, col, bsize))?;
                x += step_x;
            }
            y += step_y;
        }
        Ok((max_tx, true))
    }

    /// Section 5.11.35. `txfm_split` chooses whether this node divides.
    fn read_var_tx_size(
        &mut self,
        row: usize,
        col: usize,
        tx: usize,
        depth: usize,
        block: (usize, usize, usize),
    ) -> Result<(), CodecError> {
        if row >= self.mi_rows || col >= self.mi_cols {
            return Ok(());
        }
        let (block_row, block_col, bsize) = block;
        let split = if tx == TX_4X4 || depth == MAX_VARTX_DEPTH {
            false
        } else {
            let above = usize::from(self.above_tx_width(row, col, block_row) < width_of(tx));
            let left = usize::from(self.left_tx_height(row, col, block_col) < height_of(tx));
            let square = square_tx(64.min(block_width(bsize).max(block_height(bsize))));
            let sqr_up = TX_SIZE_SQR_UP.get(tx).copied().unwrap_or(square);
            let ctx = usize::from(sqr_up != square) * 3 + (4 - square) * 6 + above + left;
            read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.txfm_split,
                &[21, 3],
                &[ctx],
            )? == 1
        };
        let w4 = (width_of(tx) / 4).max(1);
        let h4 = (height_of(tx) / 4).max(1);
        if split {
            let sub = SPLIT_TX_SIZE[tx];
            let step_x = (width_of(sub) / 4).max(1);
            let step_y = (height_of(sub) / 4).max(1);
            let mut y = 0;
            while y < h4 {
                let mut x = 0;
                while x < w4 {
                    self.read_var_tx_size(row + y, col + x, sub, depth + 1, block)?;
                    x += step_x;
                }
                y += step_y;
            }
            return Ok(());
        }
        for y in 0..h4 {
            for x in 0..w4 {
                let at_row = row + y;
                let at_col = col + x;
                if at_row >= self.mi_rows || at_col >= self.mi_cols {
                    continue;
                }
                self.tx_sizes[at_row * self.mi_cols + at_col] = tx as u8;
            }
        }
        self.tx_origins.push((row as u16, col as u16, tx as u8));
        Ok(())
    }

    fn reconstruct_variable_luma(&mut self, coded: &CodedBlock) -> Result<(), CodecError> {
        let origins = std::mem::take(&mut self.tx_origins);
        let mut result = Ok(());
        for &(mi_row, mi_col, tx) in &origins {
            let tx = usize::from(tx);
            if tx >= TX_WIDTH.len() {
                result = Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "tx size",
                });
                break;
            }
            let x4 = usize::from(mi_col).saturating_sub(coded.col);
            let y4 = usize::from(mi_row).saturating_sub(coded.row);
            if let Err(error) = self.transform_block(Block {
                plane: 0,
                row: coded.row,
                col: coded.col,
                bsize: coded.bsize,
                y_mode: coded.y_mode,
                uv_mode: coded.uv_mode,
                angle_y: coded.angle_y,
                angle_uv: coded.angle_uv,
                tx,
                skip: coded.skip,
                avail_u: coded.avail_u,
                avail_l: coded.avail_l,
                x4,
                y4,
                sub_x: 0,
                sub_y: 0,
                step_x: (TX_WIDTH[tx] / 4).max(1),
                step_y: (TX_HEIGHT[tx] / 4).max(1),
                inter: true,
            }) {
                result = Err(error);
                break;
            }
        }
        self.tx_origins = origins;
        result
    }

    fn tx_depth_ctx(&self, row: usize, col: usize, max_rect: usize) -> usize {
        // An inter neighbor contributes its block size. An intra neighbor contributes its transform.
        let above = if !self.above_mi(row) {
            0
        } else if self.inter_at(row - 1, col) {
            block_width(self.bsize_at(row - 1, col))
        } else {
            self.above_tx_width(row, col, row)
        };
        let left = if !self.left_mi(col) {
            0
        } else if self.inter_at(row, col - 1) {
            block_height(self.bsize_at(row, col - 1))
        } else {
            self.left_tx_height(row, col, col)
        };
        usize::from(above >= width_of(max_rect)) + usize::from(left >= height_of(max_rect))
    }

    /// `get_above_tx_width`. At the block edge a missing neighbor is 64 wide.
    fn above_tx_width(&self, row: usize, col: usize, block_row: usize) -> usize {
        if row == block_row {
            if !self.above_mi(block_row) {
                return 64;
            }
            if self.skipped(row - 1, col) && self.inter_at(row - 1, col) {
                return block_width(self.bsize_at(row - 1, col));
            }
        }
        if row == 0 {
            return 0;
        }
        width_of(self.tx_at(row - 1, col))
    }

    /// `get_left_tx_height`. At the block edge a missing neighbor is 64 high.
    fn left_tx_height(&self, row: usize, col: usize, block_col: usize) -> usize {
        if col == block_col {
            if !self.left_mi(block_col) {
                return 64;
            }
            if self.skipped(row, col - 1) && self.inter_at(row, col - 1) {
                return block_height(self.bsize_at(row, col - 1));
            }
        }
        if col == 0 {
            return 0;
        }
        height_of(self.tx_at(row, col - 1))
    }

    fn inter_at(&self, row: usize, col: usize) -> bool {
        self.mi_at(row, col)
            .is_some_and(|at| self.inter_flag[at] != 0)
    }

    fn skipped(&self, row: usize, col: usize) -> bool {
        self.mi_at(row, col).is_some_and(|at| self.skips[at] != 0)
    }

    fn bsize_at(&self, row: usize, col: usize) -> usize {
        self.mi_at(row, col)
            .map(|at| usize::from(self.sizes[at]))
            .unwrap_or(0)
    }

    fn tx_at(&self, row: usize, col: usize) -> usize {
        self.mi_at(row, col)
            .map(|at| usize::from(self.tx_sizes[at]))
            .unwrap_or(0)
    }

    fn mi_at(&self, row: usize, col: usize) -> Option<usize> {
        if row >= self.mi_rows || col >= self.mi_cols {
            return None;
        }
        Some(row * self.mi_cols + col)
    }

    fn has_chroma(&self, row: usize, col: usize, bw4: usize, bh4: usize) -> bool {
        if bh4 == 1 && self.sub_y && row.is_multiple_of(2) {
            return false;
        }
        if bw4 == 1 && self.sub_x && col.is_multiple_of(2) {
            return false;
        }
        true
    }

    fn reset_context(&mut self, row: usize, col: usize, bw4: usize, bh4: usize, has_chroma: bool) {
        let planes = if has_chroma { 3 } else { 1 };
        for plane in 0..planes {
            let (sub_x, sub_y) = self.subsampling(plane);
            let x0 = col >> sub_x;
            let y0 = row >> sub_y;
            let x1 = (col + bw4) >> sub_x;
            let y1 = (row + bh4) >> sub_y;
            for x in x0..x1.min(self.above_level[plane].len()) {
                self.above_level[plane][x] = 0;
                self.above_dc[plane][x] = 0;
            }
            for y in y0..y1.min(self.left_level[plane].len()) {
                self.left_level[plane][y] = 0;
                self.left_dc[plane][y] = 0;
            }
        }
    }

    fn fill_grid(&mut self, coded: &CodedBlock) {
        let CodedBlock {
            row,
            col,
            bw4,
            bh4,
            y_mode,
            uv_mode,
            skip,
            bsize,
            tx,
            ..
        } = *coded;
        for y in 0..bh4 {
            for x in 0..bw4 {
                if row + y >= self.mi_rows || col + x >= self.mi_cols {
                    continue;
                }
                let at = (row + y) * self.mi_cols + col + x;
                self.y_mode[at] = y_mode;
                self.uv_mode[at] = uv_mode;
                self.skips[at] = u8::from(skip);
                self.sizes[at] = bsize as u8;
                if !coded.var_tx {
                    self.tx_sizes[at] = tx as u8;
                }
                self.inter_flag[at] = u8::from(coded.inter);
                self.mvs[at] = coded.mv;
                self.mvs1[at] = coded.mv1;
                self.block_ref[at] = coded.ref0;
                self.block_ref1[at] = coded.ref1;
                self.interp[at] = coded.interp;
                self.ii_mode[at] = coded.ii;
                self.compound_idx[at] = coded.compound_idx;
                self.compound_mask[at] = coded.compound_mask;
                self.skip_modes[at] = u8::from(coded.skip_mode);
                self.motion_modes[at] = coded.motion_mode;
            }
        }
    }

    fn predict_and_residual(&mut self, coded: &CodedBlock) -> Result<(), CodecError> {
        self.blend_ready = false;
        let CodedBlock {
            row,
            col,
            bsize,
            y_mode,
            uv_mode,
            angle_y,
            angle_uv,
            tx,
            skip,
            has_chroma,
            avail_u,
            avail_l,
            ..
        } = *coded;
        let planes = if has_chroma { 3 } else { 1 };
        if coded.var_tx {
            self.reconstruct_variable_luma(coded)?;
        }
        for plane in usize::from(coded.var_tx)..planes {
            let plane_size = if plane == 0 {
                bsize
            } else {
                subsampled_size(bsize, self.sub_x, self.sub_y)
            };
            let tx_size = self.plane_tx(plane, bsize, tx);
            let step_x = TX_WIDTH[tx_size] >> 2;
            let step_y = TX_HEIGHT[tx_size] >> 2;
            let num_w = NUM_4X4_WIDE[plane_size];
            let num_h = NUM_4X4_HIGH[plane_size];
            let (sub_x, sub_y) = self.subsampling(plane);
            let mut y4 = 0;
            while y4 < num_h {
                let mut x4 = 0;
                while x4 < num_w {
                    self.transform_block(Block {
                        plane,
                        row,
                        col,
                        bsize,
                        y_mode,
                        uv_mode,
                        angle_y,
                        angle_uv,
                        tx: tx_size,
                        skip,
                        avail_u,
                        avail_l,
                        x4,
                        y4,
                        sub_x,
                        sub_y,
                        step_x,
                        step_y,
                        inter: coded.inter,
                    })?;
                    x4 += step_x;
                }
                y4 += step_y;
            }
        }
        Ok(())
    }

    fn plane_tx(&self, plane: usize, bsize: usize, tx: usize) -> usize {
        if plane == 0 {
            return tx;
        }
        let uv = MAX_TX_SIZE_RECT[subsampled_size(bsize, self.sub_x, self.sub_y)];
        if TX_WIDTH[uv] == 64 || TX_HEIGHT[uv] == 64 {
            if TX_WIDTH[uv] == 16 {
                return TX_16X32;
            }
            if TX_HEIGHT[uv] == 16 {
                return TX_32X16;
            }
            return TX_32X32;
        }
        uv
    }

    fn transform_block(&mut self, block: Block) -> Result<(), CodecError> {
        let (sub_x, sub_y) = (block.sub_x, block.sub_y);
        let start_x = ((block.col >> sub_x) + block.x4) * 4;
        let start_y = ((block.row >> sub_y) + block.y4) * 4;
        let (plane_w, plane_h, stride) = if block.plane == 0 {
            (self.width, self.height, self.width)
        } else {
            (self.chroma_w, self.chroma_h, self.chroma_w)
        };
        if block.inter && block.x4 == 0 && block.y4 == 0 {
            self.predict_inter_plane(&block)?;
        }
        if start_x >= plane_w || start_y >= plane_h {
            return Ok(());
        }
        self.coding_inter = block.inter;
        self.cur_luma_row = block.row + (block.y4 << block.sub_y);
        self.cur_luma_col = block.col + (block.x4 << block.sub_x);
        // The decoded-neighbor grid is one superblock. Chroma 4:2:0 has half
        // as many 4×4 cells on each axis as luma, so the mask follows the plane.
        let row_mask = (SB4 >> block.sub_y) - 1;
        let col_mask = (SB4 >> block.sub_x) - 1;
        let local_row = ((start_y >> 2) & row_mask) as i32;
        let local_col = ((start_x >> 2) & col_mask) as i32;
        if !block.inter {
            self.predict_intra_block(
                &block,
                IntraAt {
                    start_x,
                    start_y,
                    stride,
                    plane_w,
                    plane_h,
                    local_row,
                    local_col,
                },
            )?;
        }
        if !block.skip {
            let tx_type = self.coeffs(CoeffPlace {
                plane: block.plane,
                start_x,
                start_y,
                tx: block.tx,
                bsize: block.bsize,
                y_mode: block.y_mode,
                uv_mode: block.uv_mode,
            })?;
            if let Some(tx_type) = tx_type {
                let quant = std::mem::take(&mut self.quant_buf);
                let decoded = if quant.iter().any(|level| *level != 0) {
                    self.reconstruct(block.plane, start_x, start_y, block.tx, tx_type, &quant)
                } else {
                    Ok(())
                };
                self.quant_buf = quant;
                decoded?;
            }
        }
        for y in 0..block.step_y {
            for x in 0..block.step_x {
                self.set_decoded(block.plane, local_row + y as i32, local_col + x as i32, 1);
            }
        }
        Ok(())
    }

    fn predict_intra_block(&mut self, block: &Block, at: IntraAt) -> Result<(), CodecError> {
        let IntraAt {
            start_x,
            start_y,
            stride,
            plane_w,
            plane_h,
            local_row,
            local_col,
        } = at;
        let mode = if block.plane == 0 {
            block.y_mode
        } else if block.uv_mode == UV_CFL_PRED {
            DC_PRED
        } else {
            block.uv_mode
        };
        let angle = if block.plane == 0 {
            block.angle_y
        } else {
            block.angle_uv
        };
        let have_left = block.avail_l || block.x4 > 0;
        let have_above = block.avail_u || block.y4 > 0;
        // local_row and local_col are already in this plane's 4×4 units.
        let above_right =
            self.decoded_at(block.plane, local_row - 1, local_col + block.step_x as i32);
        let below_left =
            self.decoded_at(block.plane, local_row + block.step_y as i32, local_col - 1);
        let filter_type = self.smooth_neighbor(block.plane, block.row, block.col);
        let log_w = TX_WIDTH_LOG2[block.tx] as usize;
        let log_h = TX_HEIGHT_LOG2[block.tx] as usize;
        let pred = IntraPred {
            x: start_x,
            y: start_y,
            w: 1 << log_w,
            h: 1 << log_h,
            have_left: have_left && start_x > 0,
            have_above: have_above && start_y > 0,
            have_above_right: above_right,
            have_below_left: below_left,
            mode,
            angle_delta: angle,
            edge_filter: self.edge_filter,
            filter_type,
            max_x: plane_w - 1,
            max_y: plane_h - 1,
            filter: if block.plane == 0 {
                self.filter_mode
            } else {
                None
            },
        };
        if block.plane == 0 {
            predict_intra(&mut self.y, stride, &pred, &mut self.predict)?;
            self.max_luma_w = start_x + pred.w;
            self.max_luma_h = start_y + pred.h;
        } else if block.plane == 1 {
            predict_intra(&mut self.u, stride, &pred, &mut self.predict)?;
            if block.uv_mode == UV_CFL_PRED {
                chroma_from_luma(
                    ChromaFromLuma {
                        luma: &self.y,
                        luma_stride: self.width,
                        chroma: &mut self.u,
                        chroma_stride: stride,
                        x: start_x,
                        y: start_y,
                        w: pred.w,
                        h: pred.h,
                        sub_x: self.sub_x,
                        sub_y: self.sub_y,
                        alpha: self.cfl_u,
                        max_luma_w: self.max_luma_w,
                        max_luma_h: self.max_luma_h,
                    },
                    self.predict.ac(),
                )?;
            }
        } else {
            predict_intra(&mut self.v, stride, &pred, &mut self.predict)?;
            if block.uv_mode == UV_CFL_PRED {
                chroma_from_luma(
                    ChromaFromLuma {
                        luma: &self.y,
                        luma_stride: self.width,
                        chroma: &mut self.v,
                        chroma_stride: stride,
                        x: start_x,
                        y: start_y,
                        w: pred.w,
                        h: pred.h,
                        sub_x: self.sub_x,
                        sub_y: self.sub_y,
                        alpha: self.cfl_v,
                        max_luma_w: self.max_luma_w,
                        max_luma_h: self.max_luma_h,
                    },
                    self.predict.ac(),
                )?;
            }
        }
        Ok(())
    }

    fn smooth_neighbor(&self, plane: usize, row: usize, col: usize) -> bool {
        let above = if self.above_mi(row) {
            let mode = if plane == 0 {
                self.y_mode[(row - 1) * self.mi_cols + col]
            } else {
                self.uv_mode[(row - 1) * self.mi_cols + col]
            };
            (9..=11).contains(&mode)
        } else {
            false
        };
        let left = if self.left_mi(col) {
            let mode = if plane == 0 {
                self.y_mode[row * self.mi_cols + col - 1]
            } else {
                self.uv_mode[row * self.mi_cols + col - 1]
            };
            (9..=11).contains(&mode)
        } else {
            false
        };
        above || left
    }

    fn coeffs(&mut self, place: CoeffPlace) -> Result<Option<u8>, CodecError> {
        let CoeffPlace {
            plane,
            start_x,
            start_y,
            tx,
            bsize,
            y_mode,
            uv_mode,
        } = place;
        let x4 = start_x >> 2;
        let y4 = start_y >> 2;
        let w4 = TX_WIDTH[tx] >> 2;
        let h4 = TX_HEIGHT[tx] >> 2;
        let tx_ctx = (TX_SIZE_SQR[tx] + TX_SIZE_SQR_UP[tx] + 1) >> 1;
        let ctx = self.skip_coeff_ctx(SkipAt {
            plane,
            x4,
            y4,
            w4,
            h4,
            tx,
            bsize,
        });
        let all_zero = read_sym(
            &mut self.symbols,
            self.update,
            &mut self.cdfs.txb_skip,
            &[5, 13, 3],
            &[tx_ctx, ctx],
        )? == 1;
        let seg = if tx == TX_16X64 || tx == TX_64X16 {
            512
        } else {
            (TX_WIDTH[tx] * TX_HEIGHT[tx]).min(1024)
        };
        let mut quant = std::mem::take(&mut self.quant_buf);
        if quant.len() < seg {
            quant.resize(seg, 0);
        }
        quant.truncate(seg);
        quant.fill(0);
        let decoded = (|| -> Result<Option<u8>, CodecError> {
            if all_zero {
                self.write_levels(
                    DcSpan {
                        plane,
                        x4,
                        y4,
                        w4,
                        h4,
                    },
                    0,
                    0,
                );
                return Ok(None);
            }
            let tx_type = self.tx_type(plane, tx, y_mode, uv_mode)?;
            let scan = scan_for(tx, tx_type);
            let eob = self.read_eob(tx, tx_ctx, plane, tx_type)?;
            self.read_levels(
                &mut quant,
                LevelScan {
                    scan,
                    eob,
                    tx,
                    tx_ctx,
                    plane,
                    tx_type,
                },
            )?;
            self.apply_signs_and_golomb(
                &mut quant,
                scan,
                eob,
                DcSpan {
                    plane,
                    x4,
                    y4,
                    w4,
                    h4,
                },
            )?;
            Ok(Some(tx_type))
        })();
        self.quant_buf = quant;
        decoded
    }

    fn skip_coeff_ctx(&self, at: SkipAt) -> usize {
        let SkipAt {
            plane,
            x4,
            y4,
            w4,
            h4,
            tx,
            bsize,
        } = at;
        let (sub_x, sub_y) = self.subsampling(plane);
        let max_x = self.mi_cols >> sub_x;
        let max_y = self.mi_rows >> sub_y;
        let plane_size = if plane == 0 {
            bsize
        } else {
            subsampled_size(bsize, self.sub_x, self.sub_y)
        };
        let bw = block_width(plane_size);
        let bh = block_height(plane_size);
        let w = TX_WIDTH[tx];
        let h = TX_HEIGHT[tx];
        let geom = Neighbor {
            x4,
            y4,
            w4,
            h4,
            max_x,
            max_y,
            bw,
            bh,
            tw: w,
            th: h,
        };
        if plane == 0 {
            luma_coeff_ctx(&self.above_level[0], &self.left_level[0], &geom)
        } else {
            chroma_coeff_ctx(
                &self.above_level[plane],
                &self.above_dc[plane],
                &self.left_level[plane],
                &self.left_dc[plane],
                &geom,
            )
        }
    }

    fn tx_type(
        &mut self,
        plane: usize,
        tx: usize,
        y_mode: u8,
        uv_mode: u8,
    ) -> Result<u8, CodecError> {
        if self.lossless || TX_SIZE_SQR_UP[tx] > TX_32X32 {
            return self.remember_tx(plane, tx, DCT_DCT);
        }
        if self.coding_inter {
            return self.inter_tx(plane, tx);
        }
        let set = tx_set(tx, self.reduced_tx);
        if set == 0 {
            return Ok(DCT_DCT);
        }
        if plane == 0 {
            if self.base_q == 0 {
                return Ok(DCT_DCT);
            }
            let sqr = TX_SIZE_SQR[tx];
            let dir = self.filter_mode.map(filter_intra_dir).unwrap_or(y_mode);
            let symbol = if set == 1 {
                read_sym(
                    &mut self.symbols,
                    self.update,
                    &mut self.cdfs.intra_tx1,
                    &[2, 13, 8],
                    &[sqr, usize::from(dir)],
                )?
            } else {
                read_sym(
                    &mut self.symbols,
                    self.update,
                    &mut self.cdfs.intra_tx2,
                    &[3, 13, 6],
                    &[sqr, usize::from(dir)],
                )?
            };
            return intra_tx_type(set, symbol).ok_or(CodecError::Malformed {
                at_bit: 0,
                what: "tx type",
            });
        }
        let mapped = MODE_TO_TXFM[uv_mode as usize];
        let allowed = intra_set_allows(set, mapped);
        Ok(if allowed { mapped } else { DCT_DCT })
    }

    fn read_eob(
        &mut self,
        tx: usize,
        tx_ctx: usize,
        plane: usize,
        tx_type: u8,
    ) -> Result<usize, CodecError> {
        let ptype = usize::from(plane > 0);
        let class_ctx = usize::from(tx_class(tx_type) != 0);
        let multi = TX_WIDTH_LOG2[tx].min(5) + TX_HEIGHT_LOG2[tx].min(5) - 4;
        let (which, len) = match multi {
            0 => (0, 6),
            1 => (1, 7),
            2 => (2, 8),
            3 => (3, 9),
            4 => (4, 10),
            5 => (5, 11),
            _ => (6, 12),
        };
        let eob_pt = if multi <= 4 {
            read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.eob[which],
                &[2, 2, len],
                &[ptype, class_ctx],
            )? + 1
        } else {
            read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.eob[which],
                &[2, len],
                &[ptype],
            )? + 1
        };
        let mut eob = if eob_pt < 2 {
            eob_pt
        } else {
            (1 << (eob_pt - 2)) + 1
        };
        let shift = eob_pt as i32 - 3;
        if shift >= 0 {
            let extra = read_sym(
                &mut self.symbols,
                self.update,
                &mut self.cdfs.eob_extra,
                &[5, 2, 9, 3],
                &[tx_ctx, ptype, (eob_pt - 3) as usize],
            )?;
            if extra == 1 {
                eob += 1 << shift;
            }
            let base = eob_pt.saturating_sub(2);
            for i in 1..base {
                let bit_shift = base - 1 - i;
                if self.symbols.read_bool()? {
                    eob += 1 << bit_shift;
                }
            }
        }
        Ok(eob as usize)
    }

    fn read_levels(&mut self, quant: &mut [i32], job: LevelScan<'_>) -> Result<(), CodecError> {
        let LevelScan {
            scan,
            eob,
            tx,
            tx_ctx,
            plane,
            tx_type,
        } = job;
        let ptype = usize::from(plane > 0);
        let mut c = eob;
        while c > 0 {
            c -= 1;
            let pos = scan[c] as usize;
            let level = if c + 1 == eob {
                let ctx = coeff_base_ctx(tx, tx_type, pos, c, true, quant)?;
                read_sym(
                    &mut self.symbols,
                    self.update,
                    &mut self.cdfs.coeff_eob,
                    &[5, 2, 4, 4],
                    &[tx_ctx, ptype, ctx],
                )? + 1
            } else {
                let ctx = coeff_base_ctx(tx, tx_type, pos, c, false, quant)?;
                read_sym(
                    &mut self.symbols,
                    self.update,
                    &mut self.cdfs.coeff_base,
                    &[5, 2, 42, 5],
                    &[tx_ctx, ptype, ctx],
                )?
            };
            let mut level = level;
            if level > 2 {
                for _ in 0..4 {
                    let ctx = br_ctx(tx, tx_type, pos, quant);
                    let br = read_sym(
                        &mut self.symbols,
                        self.update,
                        &mut self.cdfs.coeff_br,
                        &[5, 2, 21, 5],
                        &[tx_ctx.min(3), ptype, ctx],
                    )?;
                    level += br;
                    if br < 3 {
                        break;
                    }
                }
            }
            quant[pos] = level as i32;
        }
        Ok(())
    }

    fn apply_signs_and_golomb(
        &mut self,
        quant: &mut [i32],
        scan: &[u16],
        eob: usize,
        span: DcSpan,
    ) -> Result<(), CodecError> {
        let DcSpan {
            plane,
            x4,
            y4,
            w4,
            h4,
        } = span;
        let ptype = usize::from(plane > 0);
        let mut cul = 0i32;
        let mut dc_cat = 0u8;
        for (c, scanned) in scan.iter().enumerate().take(eob) {
            let pos = usize::from(*scanned);
            if quant[pos] == 0 {
                continue;
            }
            let sign = if c == 0 {
                let ctx = self.dc_sign_ctx(plane, x4, y4, w4, h4);
                read_sym(
                    &mut self.symbols,
                    self.update,
                    &mut self.cdfs.dc_sign,
                    &[2, 3, 3],
                    &[ptype, ctx],
                )? == 1
            } else {
                self.symbols.read_bool()?
            };
            if quant[pos] > 14 {
                quant[pos] = self.read_golomb()?;
            }
            if pos == 0 {
                dc_cat = if sign { 1 } else { 2 };
            }
            cul += quant[pos];
            if sign {
                quant[pos] = -quant[pos];
            }
        }
        self.write_levels(span, u8::try_from(cul.min(63)).unwrap_or(0), dc_cat);
        Ok(())
    }

    fn read_golomb(&mut self) -> Result<i32, CodecError> {
        let mut length = 0u32;
        loop {
            length += 1;
            if length > 20 {
                return Err(CodecError::Limit { what: "golomb" });
            }
            if self.symbols.read_bool()? {
                break;
            }
        }
        let mut x = 1i32;
        let data_bits = length.saturating_sub(1);
        for _ in 0..data_bits {
            let bit = i32::from(self.symbols.read_bool()?);
            x = (x << 1) | bit;
        }
        Ok(x + 14)
    }

    fn dc_sign_ctx(&self, plane: usize, x4: usize, y4: usize, w4: usize, h4: usize) -> usize {
        let (sub_x, sub_y) = self.subsampling(plane);
        let max_x = self.mi_cols >> sub_x;
        let max_y = self.mi_rows >> sub_y;
        let mut sign = 0i32;
        for k in 0..w4 {
            if x4 + k < max_x {
                match self.above_dc[plane][x4 + k] {
                    1 => sign -= 1,
                    2 => sign += 1,
                    _ => {}
                }
            }
        }
        for k in 0..h4 {
            if y4 + k < max_y {
                match self.left_dc[plane][y4 + k] {
                    1 => sign -= 1,
                    2 => sign += 1,
                    _ => {}
                }
            }
        }
        if sign < 0 {
            1
        } else if sign > 0 {
            2
        } else {
            0
        }
    }

    fn write_levels(&mut self, span: DcSpan, level: u8, dc: u8) {
        let DcSpan {
            plane,
            x4,
            y4,
            w4,
            h4,
        } = span;
        for i in 0..w4 {
            if x4 + i < self.above_level[plane].len() {
                self.above_level[plane][x4 + i] = level;
                self.above_dc[plane][x4 + i] = dc;
            }
        }
        for i in 0..h4 {
            if y4 + i < self.left_level[plane].len() {
                self.left_level[plane][y4 + i] = level;
                self.left_dc[plane][y4 + i] = dc;
            }
        }
    }

    fn reconstruct(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        tx: usize,
        tx_type: u8,
        quant: &[i32],
    ) -> Result<(), CodecError> {
        let w = TX_WIDTH[tx];
        let h = TX_HEIGHT[tx];
        let tw = w.min(32);
        let th = h.min(32);
        let denom = match tx {
            TX_32X32 | TX_16X32 | TX_32X16 | TX_16X64 | TX_64X16 => 2,
            TX_64X64 | TX_32X64 | TX_64X32 => 4,
            _ => 1,
        };
        let dc = self.quantizer(plane, true);
        let ac = self.quantizer(plane, false);
        let qm_level = if tx_type < IDTX { self.qm[plane] } else { 15 };
        let n = w.saturating_mul(h);
        let row_n = w.max(h);
        if self.coeff_buf.len() < n {
            self.coeff_buf.resize(n, 0);
        }
        self.coeff_buf[..n].fill(0);
        if self.residual_buf.len() < n {
            self.residual_buf.resize(n, 0);
        }
        if self.idct_row.len() < row_n {
            self.idct_row.resize(row_n, 0);
        }
        if self.idct_out.len() < n {
            self.idct_out.resize(n, 0);
        }
        for row in 0..th {
            for col in 0..tw {
                let q = if row == 0 && col == 0 { dc } else { ac };
                let q = crate::qm::scale(q, qm_level, plane > 0, tx, row, col, tw);
                let level = i64::from(quant[row * tw + col]);
                let dq = level * i64::from(q);
                let signed = if dq < 0 { -1 } else { 1 };
                let value = signed * ((dq.abs() & 0x00ff_ffff) / i64::from(denom));
                self.coeff_buf[row * w + col] =
                    i32::try_from(value.clamp(-32768, 32767)).unwrap_or(0);
            }
        }
        inverse_2d_into(
            &self.coeff_buf,
            &mut InverseScratch {
                residual: &mut self.residual_buf,
                row: &mut self.idct_row,
                out: &mut self.idct_out,
            },
            w,
            h,
            tx_type,
            TRANSFORM_ROW_SHIFT[tx],
        );
        let flip_ud = matches!(
            tx_type,
            FLIPADST_DCT | FLIPADST_ADST | V_FLIPADST | FLIPADST_FLIPADST
        );
        let flip_lr = matches!(
            tx_type,
            DCT_FLIPADST | ADST_FLIPADST | H_FLIPADST | FLIPADST_FLIPADST
        );
        let stride = if plane == 0 {
            self.width
        } else {
            self.chroma_w
        };
        let residual = &self.idct_out;
        let samples = if plane == 0 {
            &mut self.y
        } else if plane == 1 {
            &mut self.u
        } else {
            &mut self.v
        };
        for row in 0..h {
            for col in 0..w {
                let xx = if flip_lr { w - col - 1 } else { col };
                let yy = if flip_ud { h - row - 1 } else { row };
                let at = (y + yy) * stride + x + xx;
                let Some(slot) = samples.get_mut(at) else {
                    continue;
                };
                let sum = i32::from(*slot) + residual[row * w + col];
                *slot = sum.clamp(0, 255) as u8;
            }
        }
        Ok(())
    }

    fn quantizer(&self, plane: usize, dc: bool) -> i32 {
        let delta = if dc {
            match plane {
                0 => self.deltas[0],
                1 => self.deltas[1],
                _ => self.deltas[3],
            }
        } else {
            match plane {
                1 => self.deltas[2],
                2 => self.deltas[4],
                _ => 0,
            }
        };
        let q = (i32::from(self.current_q) + delta).clamp(0, 255) as usize;
        i32::from(if dc { DC_QLOOKUP[q] } else { AC_QLOOKUP[q] })
    }
}

fn sign_bias(sequence: &Sequence, frame: &FrameInfo) -> [bool; 8] {
    let mut bias = [false; 8];
    if !sequence.enable_order_hint {
        return bias;
    }
    for (index, slot) in frame.ref_frames.iter().enumerate() {
        let hint = frame.slot_hints[usize::from(*slot)];
        bias[index + 1] =
            crate::header::relative_dist(hint, frame.order_hint, sequence.order_hint_bits) > 0;
    }
    bias
}

fn clipped_dist(dist: i32) -> i32 {
    match i32::try_from(dist.unsigned_abs()) {
        Ok(value) => value.min(31),
        Err(_) => 31,
    }
}

const DIST_WEIGHT: [[i32; 2]; 4] = [[2, 3], [2, 5], [2, 7], [1, 31]];
const DIST_LOOKUP: [[i32; 2]; 4] = [[9, 7], [11, 5], [12, 4], [13, 3]];

fn ref_count_ctx(counts0: usize, counts1: usize) -> usize {
    if counts0 < counts1 {
        0
    } else if counts0 == counts1 {
        1
    } else {
        2
    }
}

fn backward(frame: i8) -> bool {
    (BWDREF_FRAME..=ALTREF_FRAME).contains(&frame)
}

fn same_direction(first: i8, second: i8) -> bool {
    (first >= BWDREF_FRAME) == (second >= BWDREF_FRAME)
}

const MASK_WEDGE: u8 = 1;
const MASK_DIFF: u8 = 2;

struct CompoundChoice {
    idx: u8,
    group: u8,
    kind: u8,
    wedge: u8,
    sign: u8,
    mask_type: u8,
}

impl CompoundChoice {
    fn average() -> Self {
        Self {
            idx: 1,
            group: 0,
            kind: 0,
            wedge: 0,
            sign: 0,
            mask_type: 0,
        }
    }

    fn from_index(idx: u8) -> Self {
        Self {
            idx,
            ..Self::average()
        }
    }

    fn pack(self) -> u16 {
        u16::from(self.group)
            | (u16::from(self.kind) << 1)
            | (u16::from(self.wedge) << 3)
            | (u16::from(self.sign) << 7)
            | (u16::from(self.mask_type) << 8)
    }
}

fn mask_kind(packed: u16) -> u8 {
    ((packed >> 1) & 3) as u8
}

fn mask_wedge(packed: u16) -> u8 {
    ((packed >> 3) & 15) as u8
}

fn mask_sign(packed: u16) -> u8 {
    ((packed >> 7) & 1) as u8
}

fn mask_type(packed: u16) -> u8 {
    ((packed >> 8) & 1) as u8
}

struct Chosen {
    y_mode: u8,
    uv_mode: u8,
    angle_y: i32,
    angle_uv: i32,
    mv: Motion,
    mv1: Motion,
    ref0: i8,
    ref1: i8,
    ii: i8,
    interp: [u8; 2],
    /// 1 averages the two predictors. 0 uses distance weights.
    compound_idx: u8,
    compound_mask: u16,
    motion_mode: u8,
}

struct CodedBlock {
    row: usize,
    col: usize,
    bw4: usize,
    bh4: usize,
    bsize: usize,
    y_mode: u8,
    uv_mode: u8,
    angle_y: i32,
    angle_uv: i32,
    tx: usize,
    skip: bool,
    has_chroma: bool,
    avail_u: bool,
    avail_l: bool,
    inter: bool,
    /// Luma sizes were stored per 4×4 by the variable transform tree.
    var_tx: bool,
    mv: Motion,
    mv1: Motion,
    ref0: i8,
    ref1: i8,
    ii: i8,
    interp: [u8; 2],
    compound_idx: u8,
    compound_mask: u16,
    skip_mode: bool,
    motion_mode: u8,
}

struct MotionQuery {
    row: usize,
    col: usize,
    bsize: usize,
    mode: u8,
    reference: i8,
    reference1: i8,
    mv: Motion,
    skip_mode: bool,
}

struct InterpQuery {
    row: usize,
    col: usize,
    bsize: usize,
    mode: u8,
    reference: i8,
    reference1: i8,
    motion_mode: u8,
}

#[derive(Clone, Copy, Default)]
struct WarpSamples {
    count: usize,
    scanned: usize,
    list: [[i32; 4]; 8],
}

struct Overlap {
    pass: u8,
    cand_row: usize,
    cand_col: usize,
    pred_w: usize,
    pred_h: usize,
    x4: usize,
    y4: usize,
}

struct NeighborPred {
    slot: usize,
    plane: usize,
    mv: Motion,
    filter: [u8; 2],
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    sub_x: bool,
    sub_y: bool,
}

struct CoeffPlace {
    plane: usize,
    start_x: usize,
    start_y: usize,
    tx: usize,
    bsize: usize,
    y_mode: u8,
    uv_mode: u8,
}

struct SkipAt {
    plane: usize,
    x4: usize,
    y4: usize,
    w4: usize,
    h4: usize,
    tx: usize,
    bsize: usize,
}

struct LevelScan<'a> {
    scan: &'a [u16],
    eob: usize,
    tx: usize,
    tx_ctx: usize,
    plane: usize,
    tx_type: u8,
}

#[derive(Clone, Copy)]
struct DcSpan {
    plane: usize,
    x4: usize,
    y4: usize,
    w4: usize,
    h4: usize,
}

struct IntraAt {
    start_x: usize,
    start_y: usize,
    stride: usize,
    plane_w: usize,
    plane_h: usize,
    local_row: i32,
    local_col: i32,
}

struct Block {
    plane: usize,
    row: usize,
    col: usize,
    bsize: usize,
    y_mode: u8,
    uv_mode: u8,
    angle_y: i32,
    angle_uv: i32,
    tx: usize,
    skip: bool,
    avail_u: bool,
    avail_l: bool,
    x4: usize,
    y4: usize,
    sub_x: usize,
    sub_y: usize,
    step_x: usize,
    step_y: usize,
    inter: bool,
}

pub(crate) struct Neighbor {
    pub x4: usize,
    pub y4: usize,
    pub w4: usize,
    pub h4: usize,
    pub max_x: usize,
    pub max_y: usize,
    pub bw: usize,
    pub bh: usize,
    pub tw: usize,
    pub th: usize,
}

pub(crate) fn luma_coeff_ctx(above: &[u8], left: &[u8], geom: &Neighbor) -> usize {
    let mut top = 0u8;
    let mut side = 0u8;
    for k in 0..geom.w4 {
        if geom.x4 + k < geom.max_x && geom.x4 + k < above.len() {
            top = top.max(above[geom.x4 + k]);
        }
    }
    for k in 0..geom.h4 {
        if geom.y4 + k < geom.max_y && geom.y4 + k < left.len() {
            side = side.max(left[geom.y4 + k]);
        }
    }
    if geom.bw == geom.tw && geom.bh == geom.th {
        0
    } else if top == 0 && side == 0 {
        1
    } else if top == 0 || side == 0 {
        2 + usize::from(top.max(side) > 3)
    } else if top.max(side) <= 3 {
        4
    } else if top.min(side) <= 3 {
        5
    } else {
        6
    }
}

pub(crate) fn chroma_coeff_ctx(
    above_level: &[u8],
    above_dc: &[u8],
    left_level: &[u8],
    left_dc: &[u8],
    geom: &Neighbor,
) -> usize {
    let mut above = 0u8;
    let mut left = 0u8;
    for i in 0..geom.w4 {
        if geom.x4 + i < geom.max_x && geom.x4 + i < above_level.len() {
            above |= above_level[geom.x4 + i];
            above |= above_dc.get(geom.x4 + i).copied().unwrap_or(0);
        }
    }
    for i in 0..geom.h4 {
        if geom.y4 + i < geom.max_y && geom.y4 + i < left_level.len() {
            left |= left_level[geom.y4 + i];
            left |= left_dc.get(geom.y4 + i).copied().unwrap_or(0);
        }
    }
    let mut ctx = usize::from(above != 0) + usize::from(left != 0) + 7;
    if geom.bw * geom.bh > geom.tw * geom.th {
        ctx += 3;
    }
    ctx
}

pub(crate) fn coeff_base_ctx(
    tx: usize,
    tx_type: u8,
    pos: usize,
    c: usize,
    eob: bool,
    quant: &[i32],
) -> Result<usize, CodecError> {
    let adj = ADJUSTED_TX_SIZE[tx];
    let bwl = TX_WIDTH_LOG2[adj] as usize;
    let width = 1usize << bwl;
    let height = TX_HEIGHT[adj];
    if eob {
        let limit = height << bwl;
        let raw = if c == 0 {
            38
        } else if c <= limit / 8 {
            39
        } else if c <= limit / 4 {
            40
        } else {
            41
        };
        return Ok(raw + 4 - 42);
    }
    let class = tx_class(tx_type);
    let row = pos >> bwl;
    let col = pos - (row << bwl);
    let mut mag = 0i32;
    for idx in 0..5 {
        let ref_row = row as i32 + i32::from(SIG_REF_DIFF_OFFSET[(class * 5 + idx) * 2]);
        let ref_col = col as i32 + i32::from(SIG_REF_DIFF_OFFSET[(class * 5 + idx) * 2 + 1]);
        if ref_row >= 0 && ref_col >= 0 && (ref_row as usize) < height && (ref_col as usize) < width
        {
            let at = (ref_row as usize) * width + ref_col as usize;
            if at < quant.len() {
                mag += quant[at].abs().min(3);
            }
        }
    }
    let ctx = ((mag + 1) >> 1).min(4) as usize;
    if class == 0 {
        if row == 0 && col == 0 {
            return Ok(0);
        }
        let off = COEFF_BASE_CTX_OFFSET[tx * 25 + row.min(4) * 5 + col.min(4)] as usize;
        return Ok(ctx + off);
    }
    // Vertical transforms (class 2) vary the context down the rows.
    let idx = if class == 2 { row } else { col };
    Ok(ctx + [26, 31, 36][idx.min(2)])
}

pub(crate) fn br_ctx(tx: usize, tx_type: u8, pos: usize, quant: &[i32]) -> usize {
    let adj = ADJUSTED_TX_SIZE[tx];
    let bwl = TX_WIDTH_LOG2[adj] as usize;
    let txw = TX_WIDTH[adj];
    let txh = TX_HEIGHT[adj];
    let row = pos >> bwl;
    let col = pos - (row << bwl);
    let class = tx_class(tx_type);
    const OFFSETS: [[i32; 6]; 3] = [[0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 0, 2], [0, 1, 1, 0, 2, 0]];
    let mut mag = 0i32;
    for idx in 0..3 {
        let ref_row = row as i32 + OFFSETS[class][idx * 2];
        let ref_col = col as i32 + OFFSETS[class][idx * 2 + 1];
        if ref_row >= 0
            && ref_col >= 0
            && (ref_row as usize) < txh
            && (ref_col as usize) < (1 << bwl)
        {
            let at = ref_row as usize * txw + ref_col as usize;
            if at < quant.len() {
                mag += quant[at].min(15);
            }
        }
    }
    let mag = ((mag + 1) >> 1).min(6) as usize;
    if pos == 0 {
        mag
    } else if class == 0 {
        mag + if row < 2 && col < 2 { 7 } else { 14 }
    } else if class == 1 {
        mag + if col == 0 { 7 } else { 14 }
    } else {
        mag + if row == 0 { 7 } else { 14 }
    }
}

/// Spec transform class. Horizontal-only is 1 and vertical-only is 2.
fn tx_class(tx_type: u8) -> usize {
    if matches!(tx_type, H_DCT | H_ADST | H_FLIPADST) {
        1
    } else if matches!(tx_type, V_DCT | V_ADST | V_FLIPADST) {
        2
    } else {
        0
    }
}

fn tx_set(tx: usize, reduced: bool) -> usize {
    if TX_SIZE_SQR_UP[tx] >= TX_32X32 {
        0
    } else if reduced || TX_SIZE_SQR[tx] == TX_16X16 {
        2
    } else {
        1
    }
}

fn intra_set_allows(set: usize, tx_type: u8) -> bool {
    const SET1: [u8; 16] = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0];
    const SET2: [u8; 16] = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0];
    let table = if set == 1 { &SET1 } else { &SET2 };
    table.get(tx_type as usize).copied().unwrap_or(0) == 1
}

pub(crate) fn scan_for(tx: usize, tx_type: u8) -> &'static [u16] {
    let prefer_row = matches!(tx_type, V_DCT | V_ADST | V_FLIPADST);
    let prefer_col = matches!(tx_type, H_DCT | H_ADST | H_FLIPADST);
    if prefer_row {
        return match tx {
            TX_4X4 => &MROW_SCAN_4X4,
            5 => &MROW_SCAN_4X8,
            6 => &MROW_SCAN_8X4,
            1 => &MROW_SCAN_8X8,
            7 => &MROW_SCAN_8X16,
            8 => &MROW_SCAN_16X8,
            TX_16X16 => &MROW_SCAN_16X16,
            13 => &MROW_SCAN_4X16,
            14 => &MROW_SCAN_16X4,
            _ => default_scan(tx),
        };
    }
    if prefer_col {
        return match tx {
            TX_4X4 => &MCOL_SCAN_4X4,
            5 => &MCOL_SCAN_4X8,
            6 => &MCOL_SCAN_8X4,
            1 => &MCOL_SCAN_8X8,
            7 => &MCOL_SCAN_8X16,
            8 => &MCOL_SCAN_16X8,
            TX_16X16 => &MCOL_SCAN_16X16,
            13 => &MCOL_SCAN_4X16,
            14 => &MCOL_SCAN_16X4,
            _ => default_scan(tx),
        };
    }
    default_scan(tx)
}

fn default_scan(tx: usize) -> &'static [u16] {
    match tx {
        TX_4X4 => &DEFAULT_SCAN_4X4,
        5 => &DEFAULT_SCAN_4X8,
        6 => &DEFAULT_SCAN_8X4,
        1 => &DEFAULT_SCAN_8X8,
        7 => &DEFAULT_SCAN_8X16,
        8 => &DEFAULT_SCAN_16X8,
        TX_16X16 => &DEFAULT_SCAN_16X16,
        9 => &DEFAULT_SCAN_16X32,
        10 => &DEFAULT_SCAN_32X16,
        TX_32X32 | TX_64X64 | 11 | 12 => &DEFAULT_SCAN_32X32,
        13 => &DEFAULT_SCAN_4X16,
        14 => &DEFAULT_SCAN_16X4,
        15 => &DEFAULT_SCAN_8X32,
        16 => &DEFAULT_SCAN_32X8,
        TX_16X64 => &DEFAULT_SCAN_16X32,
        TX_64X16 => &DEFAULT_SCAN_32X16,
        _ => &DEFAULT_SCAN_32X32,
    }
}

pub(crate) fn partition_shape(bsl: usize) -> &'static [usize] {
    match bsl {
        1 => &[4, 5],
        5 => &[4, 9],
        _ => &[4, 11],
    }
}

pub(crate) fn edge_partition_cdf(cdf: &[u16], bsize: usize, horizontal: bool) -> [u16; 3] {
    let symbols: &[usize] = if horizontal {
        if bsize == BLOCK_128X128 {
            &[2, 3, 4, 6, 7]
        } else {
            &[2, 3, 4, 6, 7, 9]
        }
    } else if bsize == BLOCK_128X128 {
        &[1, 3, 4, 5, 6]
    } else {
        &[1, 3, 4, 5, 6, 8]
    };
    let mut psum = 0u32;
    for symbol in symbols {
        let current = u32::from(*cdf.get(*symbol).unwrap_or(&0));
        let previous = if *symbol == 0 {
            0
        } else {
            u32::from(*cdf.get(symbol - 1).unwrap_or(&0))
        };
        psum += current.saturating_sub(previous);
    }
    [(32768u32.saturating_sub(psum)) as u16, 32768, 0]
}

/// `Default_Filter_Intra_Cdf`. One binary row per block size.
const FILTER_INTRA_CDF: [u16; 66] = [
    4621, 32768, 0, 6743, 32768, 0, 5893, 32768, 0, 7866, 32768, 0, 12551, 32768, 0, 9394, 32768,
    0, 12408, 32768, 0, 14301, 32768, 0, 12756, 32768, 0, 22343, 32768, 0, 16384, 32768, 0, 16384,
    32768, 0, 16384, 32768, 0, 16384, 32768, 0, 16384, 32768, 0, 16384, 32768, 0, 12770, 32768, 0,
    10368, 32768, 0, 20229, 32768, 0, 18101, 32768, 0, 16384, 32768, 0, 16384, 32768, 0,
];

fn width_of(tx: usize) -> usize {
    TX_WIDTH.get(tx).copied().unwrap_or(0)
}

fn height_of(tx: usize) -> usize {
    TX_HEIGHT.get(tx).copied().unwrap_or(0)
}

/// Square transform whose side is `pixels`. Sides above 64 use the 64-point transform.
fn square_tx(pixels: usize) -> usize {
    match pixels {
        4 => TX_4X4,
        8 => TX_8X8,
        16 => TX_16X16,
        32 => TX_32X32,
        _ => TX_64X64,
    }
}

/// `Filter_Intra_Mode_To_Intra_Dir`: DC, vertical, horizontal, D157, DC.
fn filter_intra_dir(mode: u8) -> u8 {
    const MAP: [u8; 5] = [0, 1, 2, 6, 0];
    MAP.get(usize::from(mode)).copied().unwrap_or(0)
}

fn read_sym(
    symbols: &mut SymbolDecoder<'_>,
    update: bool,
    table: &mut [u16],
    shape: &[usize],
    index: &[usize],
) -> Result<u32, CodecError> {
    let (start, len) = row_range(table.len(), shape, index)?;
    if len > 32 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "cdf",
        });
    }
    let mut row = [0u16; 32];
    row[..len].copy_from_slice(&table[start..start + len]);
    let symbol = symbols.read(&mut row[..len], update)?;
    if update {
        table[start..start + len].copy_from_slice(&row[..len]);
    }
    Ok(symbol)
}

pub(crate) fn row_range(
    len: usize,
    shape: &[usize],
    index: &[usize],
) -> Result<(usize, usize), CodecError> {
    let mut stride = len;
    let mut offset = 0usize;
    for (dim, idx) in shape.iter().zip(index.iter()) {
        if *dim == 0 || !stride.is_multiple_of(*dim) || *idx >= *dim {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "cdf index",
            });
        }
        stride /= dim;
        offset += idx * stride;
    }
    let width = *shape.last().unwrap_or(&0);
    if offset + width > len {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "cdf row",
        });
    }
    Ok((offset, width))
}

impl Cdfs {
    /// `load_cdfs`: every adaptation counter, the last entry of each row, goes back to zero.
    fn clear_counts(&mut self) {
        const PARTITION_WIDTH: [usize; 5] = [5, 11, 11, 11, 9];
        const TX_WIDTH: [usize; 4] = [3, 4, 4, 4];
        const EOB_WIDTH: [usize; 7] = [6, 7, 8, 9, 10, 11, 12];
        for (table, width) in self.partition.iter_mut().zip(PARTITION_WIDTH) {
            zero_counters(table, width);
        }
        for (table, width) in self.tx.iter_mut().zip(TX_WIDTH) {
            zero_counters(table, width);
        }
        for (table, width) in self.eob.iter_mut().zip(EOB_WIDTH) {
            zero_counters(table, width);
        }
        zero_counters(&mut self.skip, 3);
        zero_counters(&mut self.y_mode, 14);
        zero_counters(&mut self.uv_yes, 15);
        zero_counters(&mut self.uv_no, 14);
        zero_counters(&mut self.angle, 8);
        zero_counters(&mut self.intra_tx1, 8);
        zero_counters(&mut self.intra_tx2, 6);
        zero_counters(&mut self.txb_skip, 3);
        zero_counters(&mut self.eob_extra, 3);
        zero_counters(&mut self.dc_sign, 3);
        zero_counters(&mut self.coeff_eob, 4);
        zero_counters(&mut self.coeff_base, 5);
        zero_counters(&mut self.coeff_br, 5);
        zero_counters(&mut self.cfl_sign, 9);
        zero_counters(&mut self.cfl_alpha, 17);
        zero_counters(&mut self.is_inter, 3);
        zero_counters(&mut self.single_ref, 3);
        zero_counters(&mut self.new_mv, 3);
        zero_counters(&mut self.zero_mv, 3);
        zero_counters(&mut self.ref_mv, 3);
        zero_counters(&mut self.drl, 3);
        zero_counters(&mut self.mv_joint, 5);
        zero_counters(&mut self.mv_sign, 3);
        zero_counters(&mut self.mv_class, 12);
        zero_counters(&mut self.mv_class0_bit, 3);
        zero_counters(&mut self.mv_class0_fr, 5);
        zero_counters(&mut self.mv_class0_hp, 3);
        zero_counters(&mut self.mv_bit, 3);
        zero_counters(&mut self.mv_fr, 5);
        zero_counters(&mut self.mv_hp, 3);
        zero_counters(&mut self.interp_filter, 4);
        zero_counters(&mut self.inter_tx1, 17);
        zero_counters(&mut self.inter_tx2, 13);
        zero_counters(&mut self.inter_tx3, 3);
        zero_counters(&mut self.txfm_split, 3);
        zero_counters(&mut self.intra_y, 14);
        zero_counters(&mut self.delta_q, 5);
        zero_counters(&mut self.use_filter_intra, 3);
        zero_counters(&mut self.filter_intra, 6);
        zero_counters(&mut self.inter_intra, 3);
        zero_counters(&mut self.inter_intra_mode, 5);
        zero_counters(&mut self.wedge_interintra, 3);
        zero_counters(&mut self.comp_mode, 3);
        zero_counters(&mut self.comp_ref_type, 3);
        zero_counters(&mut self.uni_comp_ref, 3);
        zero_counters(&mut self.comp_ref, 3);
        zero_counters(&mut self.comp_bwdref, 3);
        zero_counters(&mut self.compound_mode, 9);
        zero_counters(&mut self.compound_idx, 3);
        zero_counters(&mut self.comp_group, 3);
        zero_counters(&mut self.compound_type, 3);
        zero_counters(&mut self.wedge_index, 17);
        zero_counters(&mut self.skip_mode, 3);
        zero_counters(&mut self.motion_mode, 4);
        zero_counters(&mut self.obmc, 3);
    }

    fn load(q: usize) -> Self {
        Self {
            partition: [
                DEFAULT_PARTITION_W8_CDF.to_vec(),
                DEFAULT_PARTITION_W16_CDF.to_vec(),
                DEFAULT_PARTITION_W32_CDF.to_vec(),
                DEFAULT_PARTITION_W64_CDF.to_vec(),
                DEFAULT_PARTITION_W128_CDF.to_vec(),
            ],
            skip: DEFAULT_SKIP_CDF.to_vec(),
            y_mode: DEFAULT_INTRA_FRAME_Y_MODE_CDF.to_vec(),
            uv_yes: DEFAULT_UV_MODE_CFL_ALLOWED_CDF.to_vec(),
            uv_no: DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF.to_vec(),
            angle: DEFAULT_ANGLE_DELTA_CDF.to_vec(),
            tx: [
                DEFAULT_TX_8X8_CDF.to_vec(),
                DEFAULT_TX_16X16_CDF.to_vec(),
                DEFAULT_TX_32X32_CDF.to_vec(),
                DEFAULT_TX_64X64_CDF.to_vec(),
            ],
            intra_tx1: DEFAULT_INTRA_TX_TYPE_SET1_CDF.to_vec(),
            intra_tx2: DEFAULT_INTRA_TX_TYPE_SET2_CDF.to_vec(),
            txb_skip: slice_q(&DEFAULT_TXB_SKIP_CDF, 4, q),
            eob: [
                slice_q(&DEFAULT_EOB_PT_16_CDF, 4, q),
                slice_q(&DEFAULT_EOB_PT_32_CDF, 4, q),
                slice_q(&DEFAULT_EOB_PT_64_CDF, 4, q),
                slice_q(&DEFAULT_EOB_PT_128_CDF, 4, q),
                slice_q(&DEFAULT_EOB_PT_256_CDF, 4, q),
                slice_q(&DEFAULT_EOB_PT_512_CDF, 4, q),
                slice_q(&DEFAULT_EOB_PT_1024_CDF, 4, q),
            ],
            eob_extra: slice_q(&DEFAULT_EOB_EXTRA_CDF, 4, q),
            dc_sign: slice_q(&DEFAULT_DC_SIGN_CDF, 4, q),
            coeff_eob: slice_q(&DEFAULT_COEFF_BASE_EOB_CDF, 4, q),
            coeff_base: slice_q(&DEFAULT_COEFF_BASE_CDF, 4, q),
            coeff_br: slice_q(&DEFAULT_COEFF_BR_CDF, 4, q),
            cfl_sign: DEFAULT_CFL_SIGN_CDF.to_vec(),
            cfl_alpha: DEFAULT_CFL_ALPHA_CDF.to_vec(),
            is_inter: DEFAULT_IS_INTER_CDF.to_vec(),
            single_ref: DEFAULT_SINGLE_REF_CDF.to_vec(),
            new_mv: DEFAULT_NEW_MV_CDF.to_vec(),
            zero_mv: DEFAULT_ZERO_MV_CDF.to_vec(),
            ref_mv: DEFAULT_REF_MV_CDF.to_vec(),
            drl: DEFAULT_DRL_MODE_CDF.to_vec(),
            mv_joint: DEFAULT_MV_JOINT_CDF.to_vec(),
            mv_sign: both_components(&DEFAULT_MV_SIGN_CDF),
            mv_class: DEFAULT_MV_CLASS_CDF.to_vec(),
            mv_class0_bit: both_components(&DEFAULT_MV_CLASS0_BIT_CDF),
            mv_class0_fr: DEFAULT_MV_CLASS0_FR_CDF.to_vec(),
            mv_class0_hp: both_components(&DEFAULT_MV_CLASS0_HP_CDF),
            mv_bit: both_components(&DEFAULT_MV_BIT_CDF),
            mv_fr: DEFAULT_MV_FR_CDF.to_vec(),
            mv_hp: both_components(&DEFAULT_MV_HP_CDF),
            interp_filter: DEFAULT_INTERP_FILTER_CDF.to_vec(),
            inter_tx1: DEFAULT_INTER_TX_TYPE_SET1_CDF.to_vec(),
            inter_tx2: DEFAULT_INTER_TX_TYPE_SET2_CDF.to_vec(),
            inter_tx3: DEFAULT_INTER_TX_TYPE_SET3_CDF.to_vec(),
            txfm_split: DEFAULT_TXFM_SPLIT_CDF.to_vec(),
            intra_y: DEFAULT_Y_MODE_CDF.to_vec(),
            delta_q: vec![28160, 32120, 32677, 32768, 0],
            use_filter_intra: FILTER_INTRA_CDF.to_vec(),
            filter_intra: vec![8949, 12776, 17211, 29558, 32768, 0],
            inter_intra: INTER_INTRA_CDF.to_vec(),
            inter_intra_mode: INTER_INTRA_MODE_CDF.to_vec(),
            wedge_interintra: WEDGE_INTERINTRA_CDF.to_vec(),
            comp_mode: COMP_MODE_CDF.to_vec(),
            comp_ref_type: COMP_REF_TYPE_CDF.to_vec(),
            uni_comp_ref: UNI_COMP_REF_CDF.to_vec(),
            comp_ref: COMP_REF_CDF.to_vec(),
            comp_bwdref: COMP_BWDREF_CDF.to_vec(),
            compound_mode: COMPOUND_MODE_CDF.to_vec(),
            compound_idx: COMPOUND_IDX_CDF.to_vec(),
            comp_group: COMP_GROUP_CDF.to_vec(),
            compound_type: COMPOUND_TYPE_CDF.to_vec(),
            wedge_index: WEDGE_INDEX_CDF.to_vec(),
            skip_mode: SKIP_MODE_CDF.to_vec(),
            motion_mode: MOTION_MODE_CDF.to_vec(),
            obmc: OBMC_CDF.to_vec(),
        }
    }
}

fn obmc_weight(length: usize, index: usize) -> u8 {
    const MASK_2: [u8; 2] = [45, 64];
    const MASK_4: [u8; 4] = [39, 50, 59, 64];
    const MASK_8: [u8; 8] = [36, 42, 48, 53, 57, 61, 64, 64];
    const MASK_16: [u8; 16] = [
        34, 37, 40, 43, 46, 49, 52, 54, 56, 58, 60, 61, 64, 64, 64, 64,
    ];
    const MASK_32: [u8; 32] = [
        33, 35, 36, 38, 40, 41, 43, 44, 45, 47, 48, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 60, 61,
        62, 64, 64, 64, 64, 64, 64, 64, 64,
    ];
    let mask: &[u8] = match length {
        2 => &MASK_2,
        4 => &MASK_4,
        8 => &MASK_8,
        16 => &MASK_16,
        _ => &MASK_32,
    };
    mask.get(index).copied().unwrap_or(64)
}

/// `Default_Motion_Mode_Cdf`. One row per block size: three modes, the end, and the counter.
const MOTION_MODE_CDF: [u16; 88] = [
    10923, 21845, 32768, 0, 10923, 21845, 32768, 0, 10923, 21845, 32768, 0, 7651, 24760, 32768, 0,
    4738, 24765, 32768, 0, 5391, 25528, 32768, 0, 19419, 26810, 32768, 0, 5123, 23606, 32768, 0,
    11606, 24308, 32768, 0, 26260, 29116, 32768, 0, 20360, 28062, 32768, 0, 21679, 26830, 32768, 0,
    29516, 30701, 32768, 0, 28898, 30397, 32768, 0, 30878, 31335, 32768, 0, 32507, 32558, 32768, 0,
    10923, 21845, 32768, 0, 10923, 21845, 32768, 0, 28799, 31390, 32768, 0, 26431, 30774, 32768, 0,
    28973, 31594, 32768, 0, 29742, 31203, 32768, 0,
];

/// `Default_Use_Obmc_Cdf`. One probability, the end, and the counter, per block size.
const OBMC_CDF: [u16; 66] = [
    16384, 32768, 0, 16384, 32768, 0, 16384, 32768, 0, 10437, 32768, 0, 9371, 32768, 0, 9301,
    32768, 0, 17432, 32768, 0, 14423, 32768, 0, 15142, 32768, 0, 25817, 32768, 0, 22823, 32768, 0,
    22083, 32768, 0, 30128, 32768, 0, 31014, 32768, 0, 31560, 32768, 0, 32638, 32768, 0, 16384,
    32768, 0, 16384, 32768, 0, 23664, 32768, 0, 20901, 32768, 0, 24008, 32768, 0, 26879, 32768, 0,
];

/// `Default_Inter_Intra_Cdf`. Three size groups, one probability, then the end and the counter.
const INTER_INTRA_CDF: [u16; 9] = [26887, 32768, 0, 27597, 32768, 0, 30237, 32768, 0];

/// `Default_Inter_Intra_Mode_Cdf`. Four modes, then the end and the counter.
const INTER_INTRA_MODE_CDF: [u16; 15] = [
    1875, 11082, 27332, 32768, 0, 2473, 9996, 26388, 32768, 0, 4238, 11537, 25926, 32768, 0,
];

/// `Default_Comp_Mode_Cdf`. Five contexts, one probability, the end, and the counter.
const COMP_MODE_CDF: [u16; 15] = [
    26828, 32768, 0, 24035, 32768, 0, 12031, 32768, 0, 10640, 32768, 0, 2901, 32768, 0,
];

/// `Default_Comp_Ref_Type_Cdf`.
const COMP_REF_TYPE_CDF: [u16; 15] = [
    1198, 32768, 0, 2070, 32768, 0, 9166, 32768, 0, 7499, 32768, 0, 22475, 32768, 0,
];

/// `Default_Uni_Comp_Ref_Cdf`. Three contexts, three symbols.
const UNI_COMP_REF_CDF: [u16; 27] = [
    5284, 32768, 0, 3865, 32768, 0, 3128, 32768, 0, 23152, 32768, 0, 14173, 32768, 0, 15270, 32768,
    0, 31774, 32768, 0, 25120, 32768, 0, 26710, 32768, 0,
];

/// `Default_Comp_Ref_Cdf`. Three contexts, three forward symbols.
const COMP_REF_CDF: [u16; 27] = [
    4946, 32768, 0, 9468, 32768, 0, 1503, 32768, 0, 19891, 32768, 0, 22441, 32768, 0, 15160, 32768,
    0, 30731, 32768, 0, 31059, 32768, 0, 27544, 32768, 0,
];

/// `Default_Comp_Bwd_Ref_Cdf`. Three contexts, two backward symbols.
const COMP_BWDREF_CDF: [u16; 18] = [
    2235, 32768, 0, 1423, 32768, 0, 17182, 32768, 0, 15175, 32768, 0, 30606, 32768, 0, 30489,
    32768, 0,
];

/// `Default_Compound_Idx_Cdf`. Six contexts, one probability, the end, and the counter.
const COMPOUND_IDX_CDF: [u16; 18] = [
    18244, 32768, 0, 12865, 32768, 0, 7053, 32768, 0, 13259, 32768, 0, 9334, 32768, 0, 4644, 32768,
    0,
];

/// `Default_Skip_Mode_Cdf`. The context is how many of the above and left blocks use skip mode.
const SKIP_MODE_CDF: [u16; 9] = [32621, 32768, 0, 20708, 32768, 0, 8127, 32768, 0];

/// `Default_Comp_Group_Idx_Cdf`. Six contexts.
const COMP_GROUP_CDF: [u16; 18] = [
    26607, 32768, 0, 22891, 32768, 0, 18840, 32768, 0, 24594, 32768, 0, 19934, 32768, 0, 22674,
    32768, 0,
];

/// `Default_Compound_Type_Cdf`. One row per block size. Symbol 0 is a wedge.
const COMPOUND_TYPE_CDF: [u16; 66] = [
    16384, 32768, 0, 16384, 32768, 0, 16384, 32768, 0, 23431, 32768, 0, 13171, 32768, 0, 11470,
    32768, 0, 9770, 32768, 0, 9100, 32768, 0, 8233, 32768, 0, 6172, 32768, 0, 16384, 32768, 0,
    16384, 32768, 0, 16384, 32768, 0, 16384, 32768, 0, 16384, 32768, 0, 16384, 32768, 0, 16384,
    32768, 0, 16384, 32768, 0, 11820, 32768, 0, 7701, 32768, 0, 16384, 32768, 0, 16384, 32768, 0,
];

/// `Default_Wedge_Index_Cdf`. Sixteen wedges per block size.
const WEDGE_INDEX_CDF: [u16; 374] = [
    2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624, 28672,
    30720, 32768, 0, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528,
    24576, 26624, 28672, 30720, 32768, 0, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
    18432, 20480, 22528, 24576, 26624, 28672, 30720, 32768, 0, 2438, 4440, 6599, 8663, 11005,
    12874, 15751, 18094, 20359, 22362, 24127, 25702, 27752, 29450, 31171, 32768, 0, 806, 3266,
    6005, 6738, 7218, 7367, 7771, 14588, 16323, 17367, 18452, 19422, 22839, 26127, 29629, 32768, 0,
    2779, 3738, 4683, 7213, 7775, 8017, 8655, 14357, 17939, 21332, 24520, 27470, 29456, 30529,
    31656, 32768, 0, 1684, 3625, 5675, 7108, 9302, 11274, 14429, 17144, 19163, 20961, 22884, 24471,
    26719, 28714, 30877, 32768, 0, 1142, 3491, 6277, 7314, 8089, 8355, 9023, 13624, 15369, 16730,
    18114, 19313, 22521, 26012, 29550, 32768, 0, 2742, 4195, 5727, 8035, 8980, 9336, 10146, 14124,
    17270, 20533, 23434, 25972, 27944, 29570, 31416, 32768, 0, 1727, 3948, 6101, 7796, 9841, 12344,
    15766, 18944, 20638, 22038, 23963, 25311, 26988, 28766, 31012, 32768, 0, 2048, 4096, 6144,
    8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624, 28672, 30720, 32768, 0,
    2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624, 28672,
    30720, 32768, 0, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528,
    24576, 26624, 28672, 30720, 32768, 0, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
    18432, 20480, 22528, 24576, 26624, 28672, 30720, 32768, 0, 2048, 4096, 6144, 8192, 10240,
    12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624, 28672, 30720, 32768, 0, 2048, 4096,
    6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624, 28672, 30720, 32768,
    0, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
    28672, 30720, 32768, 0, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480,
    22528, 24576, 26624, 28672, 30720, 32768, 0, 154, 987, 1925, 2051, 2088, 2111, 2151, 23033,
    23703, 24284, 24985, 25684, 27259, 28883, 30911, 32768, 0, 1135, 1322, 1493, 2635, 2696, 2737,
    2770, 21016, 22935, 25057, 27251, 29173, 30089, 30960, 31933, 32768, 0, 2048, 4096, 6144, 8192,
    10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624, 28672, 30720, 32768, 0, 2048,
    4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624, 28672, 30720,
    32768, 0,
];

/// `Default_Compound_Mode_Cdf`. Eight contexts, eight modes.
const COMPOUND_MODE_CDF: [u16; 72] = [
    7760, 13823, 15808, 17641, 19156, 20666, 26891, 32768, 0, 10730, 19452, 21145, 22749, 24039,
    25131, 28724, 32768, 0, 10664, 20221, 21588, 22906, 24295, 25387, 28436, 32768, 0, 13298,
    16984, 20471, 24182, 25067, 25736, 26422, 32768, 0, 18904, 23325, 25242, 27432, 27898, 28258,
    30758, 32768, 0, 10725, 17454, 20124, 22820, 24195, 25168, 26046, 32768, 0, 17125, 24273,
    25814, 27492, 28214, 28704, 30592, 32768, 0, 13046, 23214, 24505, 25942, 27435, 28442, 29330,
    32768, 0,
];

/// `Default_Wedge_Inter_Intra_Cdf`. Only sizes 8×8 through 32×32 are read.
const WEDGE_INTERINTRA_CDF: [u16; 66] = [
    16384, 32768, 0, 16384, 32768, 0, 16384, 32768, 0, 20036, 32768, 0, 24957, 32768, 0, 26704,
    32768, 0, 27530, 32768, 0, 29564, 32768, 0, 29444, 32768, 0, 26872, 32768, 0, 16384, 32768, 0,
    16384, 32768, 0, 16384, 32768, 0, 16384, 32768, 0, 16384, 32768, 0, 16384, 32768, 0, 16384,
    32768, 0, 16384, 32768, 0, 16384, 32768, 0, 16384, 32768, 0, 16384, 32768, 0, 16384, 32768, 0,
];

/// Map an interintra mode onto the intra predictor. 0 DC, 1 vertical, 2 horizontal, 3 smooth.
fn interintra_intra_mode(mode: u8) -> u8 {
    match mode {
        1 => V_PRED,
        2 => 2,
        3 => SMOOTH_PRED,
        _ => DC_PRED,
    }
}

/// The specification copies one default motion-vector row into each component.
fn both_components(row: &[u16]) -> Vec<u16> {
    let mut table = Vec::with_capacity(row.len() * 2);
    table.extend_from_slice(row);
    table.extend_from_slice(row);
    table
}

fn zero_counters(table: &mut [u16], width: usize) {
    if width == 0 {
        return;
    }
    let mut index = width - 1;
    while index < table.len() {
        table[index] = 0;
        index = index.saturating_add(width);
    }
}

pub(crate) fn slice_q(table: &[u16], parts: usize, q: usize) -> Vec<u16> {
    let chunk = table.len() / parts;
    table[q * chunk..(q + 1) * chunk].to_vec()
}

#[cfg(test)]
mod tests {
    use super::{br_ctx, coeff_base_ctx, decode_tile};
    use crate::dims::{H_DCT, V_DCT};
    use crate::header::parse_unit;

    #[test]
    fn load_cdfs_clears_adaptation_counters() {
        let mut cdfs = super::Cdfs::load(1);
        let kept = cdfs.skip[0];
        cdfs.skip[2] = 32;
        cdfs.skip[5] = 17;
        cdfs.coeff_base[4] = 9;
        cdfs.clear_counts();
        assert_eq!(cdfs.skip[0], kept);
        assert_eq!(cdfs.skip[2], 0);
        assert_eq!(cdfs.skip[5], 0);
        assert_eq!(cdfs.coeff_base[4], 0);
        assert_ne!(cdfs.coeff_base[0], 0);
    }

    #[test]
    fn one_dimensional_transforms_use_the_spec_class() {
        let quant = [0i32; 16];
        let mut marked = quant;
        marked[2] = 3;
        // A vertical transform looks down the column, so a sample two columns over is ignored.
        assert_eq!(
            coeff_base_ctx(0, V_DCT, 0, 0, false, &marked).expect("ctx"),
            26
        );
        // A horizontal transform looks along the row and counts that sample.
        assert_eq!(
            coeff_base_ctx(0, H_DCT, 0, 0, false, &marked).expect("ctx"),
            28
        );
        assert_eq!(br_ctx(0, V_DCT, 1, &quant), 7);
        assert_eq!(br_ctx(0, H_DCT, 1, &quant), 14);
    }

    #[test]
    fn the_constrained_keyframe_matches_the_reference_step() {
        let bytes = [
            0x12, 0x00, 0x0a, 0x0a, 0x00, 0x00, 0x00, 0x02, 0xaf, 0xff, 0x8f, 0xff, 0x00, 0x08,
            0x32, 0x13, 0x18, 0x01, 0x78, 0x00, 0x92, 0x49, 0x24, 0x40, 0xb4, 0x9e, 0xf3, 0xed,
            0x5b, 0x9e, 0x40, 0xd0, 0xd1, 0x23, 0x96,
        ];
        let (sequence, frame) = parse_unit(&bytes, None).expect("parse");
        let picture = decode_tile(&sequence, &frame).expect("decode").picture;
        assert_eq!(picture.width, 64);
        assert_eq!(picture.height, 64);
        for row in 0..64 {
            for col in 0..64 {
                let expected = if col < 32 { 40 } else { 180 };
                assert_eq!(picture.y[row * 64 + col], expected, "y {row},{col}");
            }
        }
        assert!(picture.u.iter().all(|sample| *sample == 128));
        assert!(picture.v.iter().all(|sample| *sample == 128));
    }

    #[test]
    fn independent_aomenc_frames_match_dav1d_when_present() {
        let cases = [
            (
                "/tmp/av1-src/step128.ivf",
                "/tmp/av1-src/step128.yuv",
                128,
                128,
            ),
            (
                "/tmp/av1-src/ramp-simple.ivf",
                "/tmp/av1-src/ramp-simple.yuv",
                96,
                80,
            ),
            (
                "/tmp/av1-src/cfl-min.ivf",
                "/tmp/av1-src/cfl-min.yuv",
                96,
                80,
            ),
            (
                "/tmp/av1-src/ab-only.ivf",
                "/tmp/av1-src/ab-only.yuv",
                96,
                80,
            ),
            ("/tmp/av1-src/angle.ivf", "/tmp/av1-src/angle.yuv", 96, 80),
            ("/tmp/av1-src/recttx.ivf", "/tmp/av1-src/recttx.yuv", 96, 80),
            ("/tmp/av1-src/tx64.ivf", "/tmp/av1-src/tx64.yuv", 96, 80),
        ];
        for (ivf, yuv, width, height) in cases {
            let width: usize = width;
            let height: usize = height;
            let ivf_bytes = match std::fs::read(ivf) {
                Ok(bytes) => bytes,
                Err(_) => continue,
            };
            let expected = match std::fs::read(yuv) {
                Ok(bytes) => bytes,
                Err(_) => continue,
            };
            let frames = ivf_payloads(&ivf_bytes);
            let mut sequence = None;
            let mut got = Vec::new();
            for payload in &frames {
                let (seq, frame) = parse_unit(payload, sequence.as_ref()).expect("parse");
                let picture = decode_tile(&seq, &frame)
                    .unwrap_or_else(|err| panic!("{ivf}: {err:?}"))
                    .picture;
                got.extend_from_slice(&picture.y);
                got.extend_from_slice(&picture.u);
                got.extend_from_slice(&picture.v);
                sequence = Some(seq);
            }
            let frame_size = width * height + width.div_ceil(2) * height.div_ceil(2) * 2;
            let n = frames.len() * frame_size;
            assert_eq!(got.len(), n);
            assert_eq!(
                got,
                expected[..n],
                "{ivf} differs from the reference picture"
            );
        }
    }

    fn ivf_payloads(data: &[u8]) -> Vec<&[u8]> {
        let mut off = 32;
        let mut frames = Vec::new();
        while off + 12 <= data.len() {
            let size = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
            frames.push(&data[off + 12..off + 12 + size]);
            off += 12 + size;
        }
        frames
    }
}

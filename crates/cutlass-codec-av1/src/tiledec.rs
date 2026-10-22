//! Intra tile reconstruction for the configurations this build can finish.
//!
//! The supported slice is an 8-bit 4:2:0 intra frame, one tile, DC / vertical /
//! horizontal prediction, lossless 4×4 residuals, no screen-content tools, no
//! segmentation, no delta-q, no super-resolution, no loop filter, no CDEF, no
//! restoration, and no film grain.
//! Other legal headers parse and then return [`DecodeError::Unsupported`]
//! before any pixel is published.

use crate::bits::BitReader;
use crate::cdf::{self, KF_Y_MODE, SKIP};
use crate::coeff::{self, LevelMaps};
use crate::error::DecodeError;
use crate::geom::{
    BLOCK_4X4, BLOCK_8X8, DC_PRED, H_PRED, INTRA_MODE_CONTEXT, MI_SIZE, PARTITION_HORZ,
    PARTITION_NONE, PARTITION_SPLIT, PARTITION_VERT, V_PRED, block_height, block_width,
    mi_height_log2, mi_width_log2, num_4x4_wide, partition_subsize, subsampled_block_size,
    superblock_size,
};
use crate::header::{FrameHeader, TxMode};
use crate::intra;
use crate::limits::DecoderLimits;
use crate::sequence::SequenceHeader;
use crate::symbol::SymbolDecoder;
use crate::txfm;

pub struct Reconstructed {
    pub header: FrameHeader,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub chroma_w: u32,
    pub chroma_h: u32,
}

pub fn decode_shown_frame(
    header: FrameHeader,
    tiles: &[u8],
    seq: &SequenceHeader,
    _limits: &DecoderLimits,
) -> Result<Reconstructed, DecodeError> {
    ensure_supported(seq, &header)?;
    if !header.show_frame {
        return Err(DecodeError::unsupported(
            "hidden frames are not stored as references in this build",
        ));
    }
    if header.tile_cols != 1 || header.tile_rows != 1 {
        return Err(DecodeError::unsupported(
            "multiple tiles are not reconstructed in this build",
        ));
    }
    let mut y = Plane::new(header.frame_width, header.frame_height)?;
    let (cw, ch) = chroma_size(header.frame_width, header.frame_height, seq);
    let mut u = Plane::new(cw, ch)?;
    let mut v = Plane::new(cw, ch)?;
    let mut state = TileState {
        mi_cols: header.mi_cols,
        mi_rows: header.mi_rows,
        mi_size: vec![0u8; (header.mi_cols as usize) * (header.mi_rows as usize)],
        y_mode: vec![DC_PRED; (header.mi_cols as usize) * (header.mi_rows as usize)],
        skip: vec![0u8; (header.mi_cols as usize) * (header.mi_rows as usize)],
        levels: LevelMaps::new(header.mi_cols as usize, header.mi_rows as usize),
    };
    let mut sym = SymbolDecoder::init(BitReader::new(tiles), tiles.len(), false)?;
    decode_tile(&mut sym, seq, &header, &mut state, &mut y, &mut u, &mut v)?;
    sym.exit()?;
    Ok(Reconstructed {
        width: header.frame_width,
        height: header.frame_height,
        chroma_w: cw,
        chroma_h: ch,
        y: y.to_u8()?,
        u: u.to_u8()?,
        v: v.to_u8()?,
        header,
    })
}

fn ensure_supported(seq: &SequenceHeader, header: &FrameHeader) -> Result<(), DecodeError> {
    if seq.bit_depth != 8 || seq.mono_chrome || seq.subsampling_x != 1 || seq.subsampling_y != 1 {
        return Err(DecodeError::unsupported(
            "this build reconstructs 8-bit 4:2:0 only",
        ));
    }
    if !header.frame_is_intra {
        return Err(DecodeError::unsupported("inter frames"));
    }
    if header.allow_intrabc || header.allow_screen_content_tools || seq.enable_filter_intra {
        return Err(DecodeError::unsupported(
            "screen-content tools are not reconstructed in this build",
        ));
    }
    if header.segmentation_enabled || header.delta_q_present || header.using_qmatrix {
        return Err(DecodeError::unsupported(
            "segmentation, delta-q, and quantization matrices are not reconstructed in this build",
        ));
    }
    if header.superres_denom != 8 {
        return Err(DecodeError::unsupported(
            "super-resolution is not reconstructed in this build",
        ));
    }
    if header.restoration_type.iter().any(|&t| t != 0) {
        return Err(DecodeError::unsupported(
            "loop restoration is not reconstructed in this build",
        ));
    }
    if header.loop_filter_level.iter().any(|&l| l != 0) {
        return Err(DecodeError::unsupported(
            "deblocking is not reconstructed in this build",
        ));
    }
    if cdef_active(header) {
        return Err(DecodeError::unsupported(
            "CDEF is not reconstructed in this build",
        ));
    }
    if header.apply_grain {
        return Err(DecodeError::unsupported(
            "film grain synthesis is not reconstructed in this build",
        ));
    }
    if header.disable_cdf_update {
        Ok(())
    } else {
        Err(DecodeError::unsupported(
            "CDF adaptation is not retained in this build",
        ))
    }
}

fn cdef_active(header: &FrameHeader) -> bool {
    header.cdef_bits != 0
        || header.cdef_y_pri.iter().any(|&s| s != 0)
        || header.cdef_y_sec.iter().any(|&s| s != 0)
        || header.cdef_uv_pri.iter().any(|&s| s != 0)
        || header.cdef_uv_sec.iter().any(|&s| s != 0)
}

fn chroma_size(w: u32, h: u32, seq: &SequenceHeader) -> (u32, u32) {
    (
        (w + u32::from(seq.subsampling_x)) >> seq.subsampling_x,
        (h + u32::from(seq.subsampling_y)) >> seq.subsampling_y,
    )
}

struct Plane {
    data: Vec<u16>,
    width: usize,
    height: usize,
    stride: usize,
}

impl Plane {
    fn new(width: u32, height: u32) -> Result<Self, DecodeError> {
        let width = width as usize;
        let height = height as usize;
        let stride = width;
        let len = stride
            .checked_mul(height)
            .ok_or_else(|| DecodeError::limit("plane allocation"))?;
        Ok(Self {
            data: vec![0; len],
            width,
            height,
            stride,
        })
    }

    fn to_u8(&self) -> Result<Vec<u8>, DecodeError> {
        let mut out = Vec::with_capacity(self.width * self.height);
        for row in 0..self.height {
            for col in 0..self.width {
                let s = self.data[row * self.stride + col];
                if s > 255 {
                    return Err(DecodeError::malformed("8-bit sample exceeded 255"));
                }
                out.push(s as u8);
            }
        }
        Ok(out)
    }
}

struct TileState {
    mi_cols: u32,
    mi_rows: u32,
    mi_size: Vec<u8>,
    y_mode: Vec<u8>,
    skip: Vec<u8>,
    levels: LevelMaps,
}

impl TileState {
    fn inside(&self, r: i32, c: i32) -> bool {
        r >= 0 && c >= 0 && (r as u32) < self.mi_rows && (c as u32) < self.mi_cols
    }

    fn index(&self, r: u32, c: u32) -> usize {
        (r as usize) * (self.mi_cols as usize) + c as usize
    }
}

fn decode_tile(
    sym: &mut SymbolDecoder<'_>,
    seq: &SequenceHeader,
    header: &FrameHeader,
    state: &mut TileState,
    y: &mut Plane,
    u: &mut Plane,
    v: &mut Plane,
) -> Result<(), DecodeError> {
    let sb = superblock_size(seq.use_128x128_superblock);
    let sb4 = num_4x4_wide(sb).ok_or_else(|| DecodeError::malformed("superblock"))?;
    let mut r = 0u32;
    while r < header.mi_rows {
        let mut c = 0u32;
        while c < header.mi_cols {
            decode_partition(sym, seq, header, state, y, u, v, r, c, sb, 0)?;
            c += sb4;
        }
        r += sb4;
    }
    Ok(())
}

fn decode_partition(
    sym: &mut SymbolDecoder<'_>,
    seq: &SequenceHeader,
    header: &FrameHeader,
    state: &mut TileState,
    y: &mut Plane,
    u: &mut Plane,
    v: &mut Plane,
    r: u32,
    c: u32,
    bsize: usize,
    depth: u32,
) -> Result<(), DecodeError> {
    if r >= state.mi_rows || c >= state.mi_cols {
        return Ok(());
    }
    if depth > 8 {
        return Err(DecodeError::malformed("partition recursion"));
    }
    let avail_u = state.inside(r as i32 - 1, c as i32);
    let avail_l = state.inside(r as i32, c as i32 - 1);
    let n4 = num_4x4_wide(bsize).ok_or_else(|| DecodeError::malformed("block size"))?;
    let half = n4 >> 1;
    let has_rows = r + half < state.mi_rows;
    let has_cols = c + half < state.mi_cols;
    let bsl = mi_width_log2(bsize).ok_or_else(|| DecodeError::malformed("bsl"))?;
    let partition = if bsize < BLOCK_8X8 {
        PARTITION_NONE
    } else if has_rows && has_cols {
        let above = avail_u
            && mi_width_log2(state.mi_size[state.index(r - 1, c)] as usize).unwrap_or(0) < bsl;
        let left = avail_l
            && mi_height_log2(state.mi_size[state.index(r, c - 1)] as usize).unwrap_or(0) < bsl;
        let ctx = u32::from(left) * 2 + u32::from(above);
        let cdf =
            cdf::partition_cdf(bsl, ctx).ok_or_else(|| DecodeError::malformed("partition cdf"))?;
        let mut cdf = cdf.to_vec();
        sym.read_symbol(&mut cdf)? as u8
    } else if has_cols {
        // Vertical edge of the frame. The split-or-horizontal symbol is a
        // derived CDF. This build only finishes frames whose edge partitions
        // are fully implied, which is the case below 8x8 and for blocks that
        // are not a horizontal remainder.
        return Err(DecodeError::unsupported(
            "horizontal edge partition symbols are not reconstructed in this build",
        ));
    } else if has_rows {
        return Err(DecodeError::unsupported(
            "vertical edge partition symbols are not reconstructed in this build",
        ));
    } else {
        PARTITION_SPLIT
    };

    let sub = partition_subsize(partition, bsize)
        .ok_or_else(|| DecodeError::unsupported("partition shape is not reconstructed"))?;
    match partition {
        PARTITION_NONE => decode_block(sym, seq, header, state, y, u, v, r, c, sub),
        PARTITION_HORZ => {
            decode_block(sym, seq, header, state, y, u, v, r, c, sub)?;
            if has_rows {
                decode_block(sym, seq, header, state, y, u, v, r + half, c, sub)?;
            }
            Ok(())
        }
        PARTITION_VERT => {
            decode_block(sym, seq, header, state, y, u, v, r, c, sub)?;
            if has_cols {
                decode_block(sym, seq, header, state, y, u, v, r, c + half, sub)?;
            }
            Ok(())
        }
        PARTITION_SPLIT => {
            let split = partition_subsize(PARTITION_SPLIT, bsize)
                .ok_or_else(|| DecodeError::malformed("split size"))?;
            decode_partition(sym, seq, header, state, y, u, v, r, c, split, depth + 1)?;
            decode_partition(
                sym,
                seq,
                header,
                state,
                y,
                u,
                v,
                r,
                c + half,
                split,
                depth + 1,
            )?;
            decode_partition(
                sym,
                seq,
                header,
                state,
                y,
                u,
                v,
                r + half,
                c,
                split,
                depth + 1,
            )?;
            decode_partition(
                sym,
                seq,
                header,
                state,
                y,
                u,
                v,
                r + half,
                c + half,
                split,
                depth + 1,
            )?;
            Ok(())
        }
        _ => Err(DecodeError::unsupported(
            "asymmetric partitions are not reconstructed in this build",
        )),
    }
}

/// Lossless CfL is allowed only when the chroma residual is 4×4. Otherwise
/// it is allowed for luma blocks whose larger side is at most 32.
fn uv_cfl_allowed(
    header: &FrameHeader,
    bsize: usize,
    seq: &SequenceHeader,
) -> Result<bool, DecodeError> {
    if header.coded_lossless {
        let plane = subsampled_block_size(bsize, seq.subsampling_x, seq.subsampling_y)
            .ok_or_else(|| DecodeError::malformed("chroma block size"))?;
        Ok(plane == BLOCK_4X4)
    } else {
        let w = block_width(bsize).ok_or_else(|| DecodeError::malformed("block width"))?;
        let h = block_height(bsize).ok_or_else(|| DecodeError::malformed("block height"))?;
        Ok(w.max(h) <= 32)
    }
}

fn decode_block(
    sym: &mut SymbolDecoder<'_>,
    seq: &SequenceHeader,
    header: &FrameHeader,
    state: &mut TileState,
    y: &mut Plane,
    u: &mut Plane,
    v: &mut Plane,
    r: u32,
    c: u32,
    bsize: usize,
) -> Result<(), DecodeError> {
    let bw = block_width(bsize).ok_or_else(|| DecodeError::malformed("block width"))?;
    let bh = block_height(bsize).ok_or_else(|| DecodeError::malformed("block height"))?;
    let bw4 = bw / MI_SIZE;
    let bh4 = bh / MI_SIZE;
    // A 4-wide or 4-tall block on an even MI column/row does not code chroma
    // when that axis is subsampled. Monochrome never enters the chroma path.
    let has_chroma = !seq.mono_chrome
        && !(bh4 == 1 && seq.subsampling_y == 1 && (r & 1) == 0)
        && !(bw4 == 1 && seq.subsampling_x == 1 && (c & 1) == 0);
    let avail_u = state.inside(r as i32 - 1, c as i32);
    let avail_l = state.inside(r as i32, c as i32 - 1);
    let skip_ctx = u32::from(avail_u && state.skip[state.index(r - 1, c)] != 0)
        + u32::from(avail_l && state.skip[state.index(r, c - 1)] != 0);
    let mut skip_cdf = SKIP[skip_ctx as usize].to_vec();
    let skip = sym.read_symbol(&mut skip_cdf)?;
    if skip == 0 && !header.coded_lossless {
        return Err(DecodeError::unsupported(
            "non-lossless coefficient blocks are not reconstructed in this build",
        ));
    }
    let above_mode = if avail_u {
        state.y_mode[state.index(r - 1, c)]
    } else {
        DC_PRED
    };
    let left_mode = if avail_l {
        state.y_mode[state.index(r, c - 1)]
    } else {
        DC_PRED
    };
    let above_ctx = INTRA_MODE_CONTEXT[above_mode as usize] as usize;
    let left_ctx = INTRA_MODE_CONTEXT[left_mode as usize] as usize;
    let mut y_cdf = KF_Y_MODE[above_ctx][left_ctx].to_vec();
    let y_mode = sym.read_symbol(&mut y_cdf)? as u8;
    if y_mode != DC_PRED && y_mode != V_PRED && y_mode != H_PRED {
        return Err(DecodeError::unsupported(
            "directional, smooth, and Paeth predictors are not reconstructed in this build",
        ));
    }
    let mut uv_mode = DC_PRED;
    let mut avail_u_chroma = false;
    let mut avail_l_chroma = false;
    if has_chroma {
        let cfl_allowed = uv_cfl_allowed(header, bsize, seq)?;
        let uv_cdf = cdf::uv_mode_cdf(cfl_allowed, y_mode)
            .ok_or_else(|| DecodeError::malformed("uv mode cdf"))?;
        let mut uv_cdf = uv_cdf.to_vec();
        uv_mode = sym.read_symbol(&mut uv_cdf)? as u8;
        if uv_mode != DC_PRED && uv_mode != V_PRED && uv_mode != H_PRED {
            return Err(DecodeError::unsupported(
                "directional, smooth, Paeth, and CfL chroma predictors are not reconstructed in this build",
            ));
        }
        avail_u_chroma = if seq.subsampling_y == 1 && bh4 == 1 {
            state.inside(r as i32 - 2, c as i32)
        } else {
            avail_u
        };
        avail_l_chroma = if seq.subsampling_x == 1 && bw4 == 1 {
            state.inside(r as i32, c as i32 - 2)
        } else {
            avail_l
        };
    }
    if header.tx_mode == TxMode::Select && !header.coded_lossless {
        return Err(DecodeError::unsupported(
            "selectable transform sizes are not reconstructed in this build",
        ));
    }
    let read_coeffs = skip == 0;
    paint_block(
        sym,
        &mut state.levels,
        seq,
        header,
        y,
        r,
        c,
        bw,
        bh,
        y_mode,
        avail_l,
        avail_u,
        false,
        0,
        read_coeffs,
    )?;
    if has_chroma {
        paint_block(
            sym,
            &mut state.levels,
            seq,
            header,
            u,
            r,
            c,
            bw,
            bh,
            uv_mode,
            avail_l_chroma,
            avail_u_chroma,
            true,
            1,
            read_coeffs,
        )?;
        paint_block(
            sym,
            &mut state.levels,
            seq,
            header,
            v,
            r,
            c,
            bw,
            bh,
            uv_mode,
            avail_l_chroma,
            avail_u_chroma,
            true,
            2,
            read_coeffs,
        )?;
    }
    for row in r..r + bh4 {
        for col in c..c + bw4 {
            if row < state.mi_rows && col < state.mi_cols {
                let i = state.index(row, col);
                state.mi_size[i] = bsize as u8;
                state.y_mode[i] = y_mode;
                state.skip[i] = skip as u8;
            }
        }
    }
    Ok(())
}

fn paint_block(
    sym: &mut SymbolDecoder<'_>,
    levels: &mut LevelMaps,
    seq: &SequenceHeader,
    header: &FrameHeader,
    plane: &mut Plane,
    mi_r: u32,
    mi_c: u32,
    block_w: u32,
    block_h: u32,
    mode: u8,
    avail_l: bool,
    avail_u: bool,
    chroma: bool,
    plane_index: usize,
    read_coeffs: bool,
) -> Result<(), DecodeError> {
    let sub_x = if chroma { seq.subsampling_x } else { 0 };
    let sub_y = if chroma { seq.subsampling_y } else { 0 };
    let origin_x = ((mi_c >> sub_x) * MI_SIZE) as usize;
    let origin_y = ((mi_r >> sub_y) * MI_SIZE) as usize;
    let plane_w = ((block_w >> sub_x).max(4)) as usize;
    let plane_h = ((block_h >> sub_y).max(4)) as usize;
    let tx = if header.coded_lossless {
        4
    } else {
        plane_w.max(plane_h)
    };
    if !header.coded_lossless
        && (plane_w > 64 || plane_h > 64 || tx != plane_w || plane_w != plane_h)
    {
        return Err(DecodeError::unsupported(
            "only lossless 4x4 or a single square largest transform is reconstructed",
        ));
    }
    let step = if header.coded_lossless { 4 } else { plane_w };
    let max_x = plane.width;
    let max_y = plane.height;
    let mut y = 0usize;
    while y < plane_h {
        let mut x = 0usize;
        while x < plane_w {
            let sx = origin_x + x;
            let sy = origin_y + y;
            if sx < max_x && sy < max_y {
                let w = step.min(plane_w - x).min(max_x - sx);
                let h = step.min(plane_h - y).min(max_y - sy);
                let have_left = avail_l || x > 0;
                let have_above = avail_u || y > 0;
                let mut left = vec![0u16; h];
                let mut above = vec![0u16; w];
                if have_left {
                    for (i, slot) in left.iter_mut().enumerate() {
                        *slot = plane.data[(sy + i) * plane.stride + (sx - 1)];
                    }
                }
                if have_above {
                    for (j, slot) in above.iter_mut().enumerate() {
                        *slot = plane.data[(sy - 1) * plane.stride + (sx + j)];
                    }
                }
                let mut pred = vec![0u16; w * h];
                match mode {
                    DC_PRED => intra::predict_dc(
                        seq.bit_depth,
                        have_left,
                        have_above,
                        &left,
                        &above,
                        w,
                        h,
                        &mut pred,
                        w,
                    ),
                    V_PRED => {
                        let row = vertical_reference(
                            seq.bit_depth,
                            have_left,
                            have_above,
                            &left,
                            &above,
                            w,
                        );
                        intra::predict_vertical(&row, w, h, &mut pred, w);
                    }
                    H_PRED => {
                        let col = horizontal_reference(
                            seq.bit_depth,
                            have_left,
                            have_above,
                            &left,
                            &above,
                            h,
                        );
                        intra::predict_horizontal(&col, w, h, &mut pred, w);
                    }
                    _ => return Err(DecodeError::unsupported("intra mode is not reconstructed")),
                }
                for i in 0..h {
                    for j in 0..w {
                        plane.data[(sy + i) * plane.stride + (sx + j)] = pred[i * w + j];
                    }
                }
                if read_coeffs {
                    if w != 4 || h != 4 {
                        return Err(DecodeError::unsupported(
                            "coefficient blocks that do not fill a 4x4 are not reconstructed",
                        ));
                    }
                    let x4 = sx / 4;
                    let y4 = sy / 4;
                    let quant = coeff::read_4x4(
                        sym,
                        header.base_q_idx,
                        plane_index,
                        plane_w,
                        plane_h,
                        x4,
                        y4,
                        plane.width / 4,
                        plane.height / 4,
                        levels,
                    )?;
                    if quant.iter().any(|&q| q != 0) {
                        let (dc_q, ac_q) = plane_quant(header, plane_index, seq.bit_depth)?;
                        txfm::add_lossless_wht4x4(
                            &mut plane.data,
                            plane.stride,
                            sx,
                            sy,
                            &quant,
                            dc_q,
                            ac_q,
                            seq.bit_depth,
                        )?;
                    }
                }
            }
            x += step;
        }
        y += step;
    }
    Ok(())
}

fn plane_quant(
    header: &FrameHeader,
    plane: usize,
    bit_depth: u8,
) -> Result<(i32, i32), DecodeError> {
    let q = i32::from(header.base_q_idx);
    let (dc_delta, ac_delta) = match plane {
        0 => (header.delta_q_y_dc, 0),
        1 => (header.delta_q_u_dc, header.delta_q_u_ac),
        2 => (header.delta_q_v_dc, header.delta_q_v_ac),
        _ => return Err(DecodeError::malformed("quantizer plane")),
    };
    Ok((
        txfm::dc_quant(bit_depth, q + dc_delta)?,
        txfm::ac_quant(bit_depth, q + ac_delta)?,
    ))
}

fn vertical_reference(
    bit_depth: u8,
    have_left: bool,
    have_above: bool,
    left: &[u16],
    above: &[u16],
    w: usize,
) -> Vec<u16> {
    if have_above {
        above.to_vec()
    } else if have_left {
        vec![left[0]; w]
    } else {
        vec![(1u16 << (bit_depth - 1)) - 1; w]
    }
}

fn horizontal_reference(
    bit_depth: u8,
    have_left: bool,
    have_above: bool,
    left: &[u16],
    above: &[u16],
    h: usize,
) -> Vec<u16> {
    if have_left {
        left.to_vec()
    } else if have_above {
        vec![above[0]; h]
    } else {
        vec![(1u16 << (bit_depth - 1)) + 1; h]
    }
}

//! Intra tile decode: partitions, modes, coefficients, and reconstruction.

use cutlass_decode::CodecError;

use crate::cdef::{self, CdefInput};
use crate::deblock::{Edges, deblock_plane};
use crate::dims::{
    ADJUSTED_TX_SIZE, ADST_FLIPADST, BLOCK_8X8, BLOCK_128X128, COEFF_BASE_CTX_OFFSET, DC_PRED,
    DCT_DCT, DCT_FLIPADST, FLIPADST_ADST, FLIPADST_DCT, FLIPADST_FLIPADST, H_ADST, H_DCT,
    H_FLIPADST, IDTX, MAX_TX_DEPTH, MAX_TX_SIZE_RECT, MI_HEIGHT_LOG2, MI_WIDTH_LOG2, MODE_TO_TXFM,
    NUM_4X4_HIGH, NUM_4X4_WIDE, PARTITION_HORZ, PARTITION_HORZ_4, PARTITION_HORZ_A,
    PARTITION_HORZ_B, PARTITION_NONE, PARTITION_SPLIT, PARTITION_VERT, PARTITION_VERT_4,
    PARTITION_VERT_A, PARTITION_VERT_B, SPLIT_TX_SIZE, TRANSFORM_ROW_SHIFT, TX_4X4, TX_16X16,
    TX_16X32, TX_16X64, TX_32X16, TX_32X32, TX_32X64, TX_64X16, TX_64X32, TX_64X64, TX_HEIGHT,
    TX_HEIGHT_LOG2, TX_SIZE_SQR, TX_SIZE_SQR_UP, TX_WIDTH, TX_WIDTH_LOG2, UV_CFL_PRED, V_ADST,
    V_DCT, V_FLIPADST, block_height, block_width, intra_tx_type, partition_subsize,
    subsampled_size,
};
use crate::header::{CdefParams, FrameInfo, INTER_FRAME, Sequence};
use crate::header::{RESTORE_SGRPROJ, RESTORE_SWITCHABLE, RESTORE_WIENER};
use crate::idct::inverse_2d;
use crate::inter::{
    BlockPlace, Candidates, GLOBALMV, InterSample, LAST_FRAME, Motion, MotionGrid, NEARESTMV,
    NEARMV, NEWMV, NONE_FRAME, RefView, SIZE_GROUP, find_stack, inter_set_allows, inter_tx_set,
    inter_tx_type, predict,
};
use crate::predict::{ChromaFromLuma, IntraPred, chroma_from_luma, predict_intra};
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
    DEFAULT_UV_MODE_CFL_ALLOWED_CDF, DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF, DEFAULT_Y_MODE_CDF,
    DEFAULT_ZERO_MV_CDF, MCOL_SCAN_4X4, MCOL_SCAN_4X8, MCOL_SCAN_4X16, MCOL_SCAN_8X4,
    MCOL_SCAN_8X8, MCOL_SCAN_8X16, MCOL_SCAN_16X4, MCOL_SCAN_16X8, MCOL_SCAN_16X16, MROW_SCAN_4X4,
    MROW_SCAN_4X8, MROW_SCAN_4X16, MROW_SCAN_8X4, MROW_SCAN_8X8, MROW_SCAN_8X16, MROW_SCAN_16X4,
    MROW_SCAN_16X8, MROW_SCAN_16X16, SIG_REF_DIFF_OFFSET,
};

pub(crate) const MODE_CTX: [usize; 13] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];
const SB4: usize = 16;

pub struct DecodedPicture {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

pub fn decode_tile(sequence: &Sequence, frame: &FrameInfo) -> Result<DecodedPicture, CodecError> {
    decode_with_refs(sequence, frame, &[])
}

pub fn decode_with_refs(
    sequence: &Sequence,
    frame: &FrameInfo,
    refs: &[Option<RefView<'_>>],
) -> Result<DecodedPicture, CodecError> {
    if frame.frame_type == INTER_FRAME {
        if sequence.enable_interintra {
            return Err(CodecError::Unsupported { what: "interintra" });
        }
        if !frame.error_resilient {
            return Err(CodecError::Unsupported { what: "cdf carry" });
        }
        if frame.tx_select {
            return Err(CodecError::Unsupported {
                what: "variable transform size",
            });
        }
        if frame.motion_mode_switchable {
            return Err(CodecError::Unsupported {
                what: "overlapped motion",
            });
        }
    }
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
    let Some(first) = parts.first() else {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "tile",
        });
    };
    let mut tile = Tile::new(sequence, frame, &first.data, buf_w, buf_h, refs)?;
    let mut lr = restore::Grid::new(frame)?;
    let sb = if sequence.use_128 { 32 } else { SB4 };
    for (index, part) in parts.iter().enumerate() {
        if index > 0 {
            tile.restart(&part.data)?;
        }
        tile.tile_row = part.row;
        tile.tile_col = part.col;
        tile.clear_above(part.col, part.col_end);
        let mut row = part.row;
        while row < part.row_end {
            tile.clear_left();
            let mut col = part.col;
            while col < part.col_end {
                tile.clear_decoded(row, col);
                tile.read_deltas = frame.delta_q_present;
                restore::read_superblock(
                    &mut tile.symbols,
                    tile.update,
                    frame,
                    &mut lr,
                    row,
                    col,
                    sb,
                )?;
                let root = if sb == 32 { BLOCK_128X128 } else { 12 };
                tile.decode_partition(row, col, root)?;
                col += sb;
            }
            row += sb;
        }
        tile.symbols.finish()?;
    }
    let sizes = tile.sizes.clone();
    let tx_sizes = tile.tx_sizes.clone();
    let edges = Edges {
        sizes: &sizes,
        tx: &tx_sizes,
        mi_cols: tile.mi_cols,
        mi_rows: tile.mi_rows,
        sub_x: tile.sub_x,
        sub_y: tile.sub_y,
        frame,
    };
    deblock_plane(&mut tile.y, buf_w, buf_h, 0, &edges);
    deblock_plane(&mut tile.u, tile.chroma_w, tile.chroma_h, 1, &edges);
    deblock_plane(&mut tile.v, tile.chroma_w, tile.chroma_h, 2, &edges);
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
    Ok(DecodedPicture {
        y,
        u,
        v,
        width: out_w,
        height: visible_h,
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
    q_index: usize,
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
    block_ref: Vec<i8>,
    block_ref1: Vec<i8>,
    inter_flag: Vec<u8>,
    interp: Vec<[u8; 2]>,
    tx_kind: Vec<u8>,
    coding_inter: bool,
    cur_luma_row: usize,
    cur_luma_col: usize,
    cdef_on: bool,
    cdef: CdefParams,
    cdef_idx: Vec<i8>,
    cdfs: Cdfs,
}

struct Cdfs {
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
    intra_y: Vec<u16>,
    delta_q: Vec<u16>,
    use_filter_intra: Vec<u16>,
    filter_intra: Vec<u16>,
}

impl<'a, 'r> Tile<'a, 'r> {
    fn new(
        sequence: &Sequence,
        frame: &FrameInfo,
        bytes: &'a [u8],
        width: usize,
        height: usize,
        refs: &[Option<RefView<'r>>],
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
        let q_ctx = match frame.base_q {
            0..=20 => 0,
            21..=60 => 1,
            61..=120 => 2,
            _ => 3,
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
            q_index: q_ctx,
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
            block_ref: vec![0; cells],
            block_ref1: vec![NONE_FRAME; cells],
            inter_flag: vec![0; cells],
            interp: vec![[0, 0]; cells],
            tx_kind: vec![DCT_DCT; cells],
            coding_inter: false,
            cur_luma_row: 0,
            cur_luma_col: 0,
            cdef_on: sequence.enable_cdef,
            cdef: frame.cdef,
            cdef_idx: vec![-1; cells],
            cdfs: Cdfs::load(q_ctx),
        })
    }

    fn restart(&mut self, bytes: &'a [u8]) -> Result<(), CodecError> {
        self.symbols = SymbolDecoder::new(bytes)?;
        self.cdfs = Cdfs::load(self.q_index);
        self.current_q = self.base_q;
        self.read_deltas = false;
        self.filter_mode = None;
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
            for y in -1..=sb_y as i32 {
                for x in -1..=sb_x as i32 {
                    let above_ok = mi_row > self.tile_row;
                    let left_ok = mi_col > self.tile_col;
                    let flag = if (y < 0 && x < sb_w as i32 && above_ok)
                        || (x < 0 && y < sb_h as i32 && left_ok)
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
            mv: Motion::default(),
            ref0: 0,
            ref1: NONE_FRAME,
            interp: [0, 0],
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
        let skip = self.read_skip(self.neighbor_sum(row, col))?;
        self.read_cdef(row, col, skip)?;
        self.read_quantizer_delta(bsize, skip)?;
        self.filter_mode = None;
        let inter = self.read_is_inter(row, col)?;
        let chosen = if inter {
            self.read_inter_block(row, col, bsize, bw4, bh4)?
        } else {
            self.read_intra_in_inter(bsize, has_chroma)?
        };
        let tx = self.read_tx_size(row, col, bsize, skip, inter)?;
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
            mv: chosen.mv,
            ref0: chosen.ref0,
            ref1: chosen.ref1,
            interp: chosen.interp,
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
            ref0: 0,
            ref1: NONE_FRAME,
            interp: [0, 0],
        })
    }

    fn read_inter_block(
        &mut self,
        row: usize,
        col: usize,
        bsize: usize,
        bw4: usize,
        bh4: usize,
    ) -> Result<Chosen, CodecError> {
        let reference = self.read_single_ref(row, col)?;
        let stack = find_stack(
            &MotionGrid {
                cols: self.mi_cols,
                rows: self.mi_rows,
                inter: &self.inter_flag,
                refs: &self.block_ref,
                refs1: &self.block_ref1,
                mvs: &self.mvs,
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
            },
            self.allow_hp,
            self.force_integer,
        );
        let y_mode = self.read_inter_mode(&stack)?;
        let mv_index = self.read_drl(y_mode, stack.found, &stack)?;
        let mv = self.assign_mv(y_mode, mv_index, &stack)?;
        let interp = self.read_interp(row, col, bsize, y_mode, reference)?;
        Ok(Chosen {
            y_mode,
            uv_mode: DC_PRED,
            angle_y: 0,
            angle_uv: 0,
            mv,
            ref0: reference,
            ref1: NONE_FRAME,
            interp,
        })
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
            &[0],
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
        if mode == NEWMV {
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
        if mode != NEARMV {
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

    fn assign_mv(
        &mut self,
        mode: u8,
        index: usize,
        stack: &Candidates,
    ) -> Result<Motion, CodecError> {
        let mut pos = if mode == NEARESTMV { 0 } else { index };
        if mode == NEWMV && stack.found <= 1 {
            pos = 0;
        }
        let pred = if mode == GLOBALMV {
            Motion::default()
        } else {
            stack.mv[pos.min(7)]
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
            &[3],
            &[],
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
            &[3],
            &[],
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
                &[3],
                &[],
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
                &[10, 3],
                &[i as usize],
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
                &[3],
                &[],
            )?
        } else {
            1
        };
        Ok((1 << (class + 3)) + (((bits << 3) | ((fr as i32) << 1) | hp as i32) + 1))
    }

    fn read_interp(
        &mut self,
        row: usize,
        col: usize,
        bsize: usize,
        mode: u8,
        reference: i8,
    ) -> Result<[u8; 2], CodecError> {
        if !self.filter_switchable {
            return Ok([self.interpolation, self.interpolation]);
        }
        let dirs = if self.dual_filter { 2 } else { 1 };
        let mut filter = [0u8; 2];
        for (dir, slot) in filter.iter_mut().enumerate().take(dirs) {
            if self.needs_interp(bsize, mode) {
                let ctx = self.interp_ctx(row, col, dir, reference);
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

    fn needs_interp(&self, bsize: usize, mode: u8) -> bool {
        let large = block_width(bsize).min(block_height(bsize)) >= 8;
        !(large && mode == GLOBALMV)
    }

    fn interp_ctx(&self, row: usize, col: usize, dir: usize, reference: i8) -> usize {
        let mut ctx = (dir & 1) * 8;
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
        let reference: &[u8] = match block.plane {
            0 => self.ref_y[slot],
            1 => self.ref_u[slot],
            _ => self.ref_v[slot],
        };
        let (sub_x, sub_y) = (block.sub_x != 0, block.sub_y != 0);
        let x = (block.col >> block.sub_x) * 4;
        let y = (block.row >> block.sub_y) * 4;
        let w = block_width(block.bsize) >> block.sub_x;
        let h = block_height(block.bsize) >> block.sub_y;
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
        predict(&mut InterSample {
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
        })
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

    fn tx_depth_ctx(&self, row: usize, col: usize, max_rect: usize) -> usize {
        // An unavailable neighbor contributes width 0. A coded neighbor contributes its transform size.
        let above = if !self.above_mi(row) {
            0
        } else {
            TX_WIDTH[self.tx_sizes[(row - 1) * self.mi_cols + col] as usize]
        };
        let left = if !self.left_mi(col) {
            0
        } else {
            TX_HEIGHT[self.tx_sizes[row * self.mi_cols + col - 1] as usize]
        };
        usize::from(above >= TX_WIDTH[max_rect]) + usize::from(left >= TX_HEIGHT[max_rect])
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
                self.tx_sizes[at] = tx as u8;
                self.inter_flag[at] = u8::from(coded.inter);
                self.mvs[at] = coded.mv;
                self.block_ref[at] = coded.ref0;
                self.block_ref1[at] = coded.ref1;
                self.interp[at] = coded.interp;
            }
        }
    }

    fn predict_and_residual(&mut self, coded: &CodedBlock) -> Result<(), CodecError> {
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
        for plane in 0..planes {
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
        let local_row = ((start_y >> 2) & (SB4 - 1)) as i32;
        let local_col = ((start_x >> 2) & (SB4 - 1)) as i32;
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
            let (quant, tx_type) = self.coeffs(CoeffPlace {
                plane: block.plane,
                start_x,
                start_y,
                tx: block.tx,
                bsize: block.bsize,
                y_mode: block.y_mode,
                uv_mode: block.uv_mode,
            })?;
            if quant.iter().any(|level| *level != 0) {
                self.reconstruct(block.plane, start_x, start_y, block.tx, tx_type, &quant)?;
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
            predict_intra(&mut self.y, stride, &pred)?;
            self.max_luma_w = start_x + pred.w;
            self.max_luma_h = start_y + pred.h;
        } else if block.plane == 1 {
            predict_intra(&mut self.u, stride, &pred)?;
            if block.uv_mode == UV_CFL_PRED {
                chroma_from_luma(ChromaFromLuma {
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
                })?;
            }
        } else {
            predict_intra(&mut self.v, stride, &pred)?;
            if block.uv_mode == UV_CFL_PRED {
                chroma_from_luma(ChromaFromLuma {
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
                })?;
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

    fn coeffs(&mut self, place: CoeffPlace) -> Result<(Vec<i32>, u8), CodecError> {
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
        let mut quant = vec![0i32; seg];
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
            return Ok((quant, DCT_DCT));
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
        Ok((quant, tx_type))
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
        let mut coeff = vec![0i32; w * h];
        for row in 0..th {
            for col in 0..tw {
                let q = if row == 0 && col == 0 { dc } else { ac };
                let q = crate::qm::scale(q, qm_level, plane > 0, tx, row, col, tw);
                let level = i64::from(quant[row * tw + col]);
                let dq = level * i64::from(q);
                let signed = if dq < 0 { -1 } else { 1 };
                let value = signed * ((dq.abs() & 0x00ff_ffff) / i64::from(denom));
                coeff[row * w + col] = i32::try_from(value.clamp(-32768, 32767)).unwrap_or(0);
            }
        }
        let residual = inverse_2d(&coeff, w, h, tx_type, TRANSFORM_ROW_SHIFT[tx]);
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

struct Chosen {
    y_mode: u8,
    uv_mode: u8,
    angle_y: i32,
    angle_uv: i32,
    mv: Motion,
    ref0: i8,
    ref1: i8,
    interp: [u8; 2],
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
    mv: Motion,
    ref0: i8,
    ref1: i8,
    interp: [u8; 2],
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
            mv_sign: DEFAULT_MV_SIGN_CDF.to_vec(),
            mv_class: DEFAULT_MV_CLASS_CDF.to_vec(),
            mv_class0_bit: DEFAULT_MV_CLASS0_BIT_CDF.to_vec(),
            mv_class0_fr: DEFAULT_MV_CLASS0_FR_CDF.to_vec(),
            mv_class0_hp: DEFAULT_MV_CLASS0_HP_CDF.to_vec(),
            mv_bit: DEFAULT_MV_BIT_CDF.to_vec(),
            mv_fr: DEFAULT_MV_FR_CDF.to_vec(),
            mv_hp: DEFAULT_MV_HP_CDF.to_vec(),
            interp_filter: DEFAULT_INTERP_FILTER_CDF.to_vec(),
            inter_tx1: DEFAULT_INTER_TX_TYPE_SET1_CDF.to_vec(),
            inter_tx2: DEFAULT_INTER_TX_TYPE_SET2_CDF.to_vec(),
            inter_tx3: DEFAULT_INTER_TX_TYPE_SET3_CDF.to_vec(),
            intra_y: DEFAULT_Y_MODE_CDF.to_vec(),
            delta_q: vec![28160, 32120, 32677, 32768, 0],
            use_filter_intra: FILTER_INTRA_CDF.to_vec(),
            filter_intra: vec![8949, 12776, 17211, 29558, 32768, 0],
        }
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
        let picture = decode_tile(&sequence, &frame).expect("decode");
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
                let picture =
                    decode_tile(&seq, &frame).unwrap_or_else(|err| panic!("{ivf}: {err:?}"));
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

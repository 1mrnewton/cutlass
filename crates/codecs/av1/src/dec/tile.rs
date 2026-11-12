//! Tile decoding: partition, mode info, transform sizes, palette and coefficients
//! (spec 5.11 with the CDF selection of 8.3.2).

use std::ops::Range;

use cutlass_decode::CodecError;

use crate::dsp::Pixel;

use super::coder::Coder;
use crate::spec::tables::{
    ADJUSTED_TX_SIZE, COEFF_BASE_CTX_OFFSET, COEFF_BASE_POS_CTX_OFFSET, COMPOUND_MODE_CTX_MAP,
    DEFAULT_SCAN_4X4, DEFAULT_SCAN_4X8, DEFAULT_SCAN_4X16, DEFAULT_SCAN_8X4, DEFAULT_SCAN_8X8,
    DEFAULT_SCAN_8X16, DEFAULT_SCAN_8X32, DEFAULT_SCAN_16X4, DEFAULT_SCAN_16X8, DEFAULT_SCAN_16X16,
    DEFAULT_SCAN_16X32, DEFAULT_SCAN_32X8, DEFAULT_SCAN_32X16, DEFAULT_SCAN_32X32,
    FILTER_INTRA_MODE_TO_INTRA_DIR, INTRA_MODE_CONTEXT, MAX_TX_DEPTH_TABLE, MAX_TX_SIZE_RECT,
    MCOL_SCAN_4X4, MCOL_SCAN_4X8, MCOL_SCAN_4X16, MCOL_SCAN_8X4, MCOL_SCAN_8X8, MCOL_SCAN_8X16,
    MCOL_SCAN_16X4, MCOL_SCAN_16X8, MCOL_SCAN_16X16, MODE_TO_TXFM, MROW_SCAN_4X4, MROW_SCAN_4X8,
    MROW_SCAN_4X16, MROW_SCAN_8X4, MROW_SCAN_8X8, MROW_SCAN_8X16, MROW_SCAN_16X4, MROW_SCAN_16X8,
    MROW_SCAN_16X16, PALETTE_COLOR_CONTEXT, PALETTE_COLOR_HASH_MULTIPLIERS, PARTITION_SUBSIZE,
    SGR_PARAMS, SIG_REF_DIFF_OFFSET, SIZE_GROUP, SPLIT_TX_SIZE, TX_SIZE_SQR, TX_SIZE_SQR_UP,
    TX_TYPE_IN_SET_INTER, TX_TYPE_IN_SET_INTRA, TX_TYPE_INTER_INV_SET1, TX_TYPE_INTER_INV_SET2,
    TX_TYPE_INTER_INV_SET3, TX_TYPE_INTRA_INV_SET1, TX_TYPE_INTRA_INV_SET2, WEDGE_BITS,
};

use super::cdf::CdfContext;
use super::decoder::Mirror;
use super::frame::{FrameState, Picture, lf_dims};
use super::msac::{Hot, Msac};
use super::obu::{
    FrameHeader, RESTORE_NONE, RESTORE_SGRPROJ, RESTORE_SWITCHABLE, RESTORE_WIENER,
    SEG_LVL_GLOBALMV, SEG_LVL_REF_FRAME, SEG_LVL_SKIP, SWITCHABLE, SequenceHeader, TX_MODE_SELECT,
};
use super::types::{
    ALTREF_FRAME, ALTREF2_FRAME, BLOCK_4X4, BLOCK_8X8, BLOCK_32X32, BLOCK_64X64, BLOCK_128X128,
    BWDREF_FRAME, COMPOUND_AVERAGE, COMPOUND_DIFFWTD, COMPOUND_DISTANCE, COMPOUND_INTRA,
    COMPOUND_WEDGE, DC_PRED, GLOBAL_GLOBALMV, GLOBALMV, GOLDEN_FRAME, H_ADST, H_DCT, H_FLIPADST,
    IDTX, INTRA_FRAME, LAST_FRAME, LAST2_FRAME, LAST3_FRAME, LOCALWARP, MiFields, MiInfo, Mv,
    NEAR_NEARMV, NEAR_NEWMV, NEAREST_NEARESTMV, NEAREST_NEWMV, NEARESTMV, NEARMV, NEW_NEARESTMV,
    NEW_NEARMV, NEW_NEWMV, NEWMV, NONE, OBMC, SIMPLE, TX_4X4, TX_16X32, TX_16X64, TX_32X16,
    TX_32X32, TX_64X16, TX_64X64, UV_CFL_PRED, V_ADST, V_DCT, V_FLIPADST, bh4, block_height,
    block_width, bw4, find_tx_size, is_directional, mi_height_log2, mi_width_log2,
    plane_residual_size, tx_h, tx_h_log2, tx_w, tx_w_log2,
};

pub type R<T> = Result<T, CodecError>;

/// Called after each superblock row a tile decodes, with the frame state,
/// the picture, the reconstruction log (when the tile records one) and the
/// luma 4x4 rows `r0..r1` of the row.
pub type RowHook<'h, P> =
    dyn FnMut(&FrameState, &Picture<P>, Option<&mut ReconLog>, usize, usize, bool) + 'h;

/// The loop restoration units of `plane` that the superblocks covering
/// luma 4x4 rows `rows` and columns `cols` read (see
/// [`TileDec::read_lr`]): the unit rows and the unit columns. The spans of
/// neighbouring superblocks meet, so those of a tile's superblocks make
/// the span of the tile.
pub fn lr_unit_span(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    plane: usize,
    rows: Range<usize>,
    cols: Range<usize>,
) -> (Range<usize>, Range<usize>) {
    let (sx, sy) = if plane > 0 {
        (u32::from(seq.subsampling_x), u32::from(seq.subsampling_y))
    } else {
        (0, 0)
    };
    let unit_size = hdr.lr.unit_size[plane] as usize;
    let row_start = (rows.start * (4 >> sy)).div_ceil(unit_size);
    let row_end = fs.lr_rows[plane].min((rows.end * (4 >> sy)).div_ceil(unit_size));
    let (num, den) = if hdr.use_superres {
        ((4 >> sx) * hdr.superres_denom as usize, unit_size * 8)
    } else {
        (4 >> sx, unit_size)
    };
    let col_start = (cols.start * num).div_ceil(den);
    let col_end = fs.lr_cols[plane].min((cols.end * num).div_ceil(den));
    (row_start..row_end, col_start..col_end)
}

/// While recording, the superblocks after which the log is handed on
/// before the end of their row (see [`TileDec::decode_tile_rows`]).
const LOG_CHUNK_SBS: usize = 4;

/// What reconstruction needs of the blocks a tile decoder parsed while
/// recording (see [`TileDec::rec`]): the superblocks, blocks and transform
/// blocks in decoding order, with the dequantized coefficients.
#[derive(Default)]
pub struct ReconLog {
    pub sbs: Vec<SbRec>,
    pub blocks: Vec<BlockRec>,
    pub ops: Vec<TxOp>,
    pub coefs: Vec<i32>,
    /// Palette colors and color maps of palette blocks.
    pub extra: Vec<u16>,
    /// Warp samples of local warp blocks.
    pub cands: Vec<[i32; 4]>,
}

impl ReconLog {
    pub fn clear(&mut self) {
        self.sbs.clear();
        self.blocks.clear();
        self.ops.clear();
        self.coefs.clear();
        self.extra.clear();
        self.cands.clear();
    }
}

/// A superblock of a [`ReconLog`]: its position, its tile's bounds and its
/// first block.
#[derive(Clone, Copy)]
pub struct SbRec {
    pub r: usize,
    pub c: usize,
    pub tile: [usize; 4],
    pub first_block: usize,
}

/// A block of a [`ReconLog`]: what prediction and reconstruction read of
/// its state when prediction starts, its first transform block and, when
/// it has any, where its palette (colors and color maps, in
/// [`ReconLog::extra`]) and its local warp samples (in [`ReconLog::cands`])
/// start.
pub struct BlockRec {
    pub b: RecBlock,
    pub first_op: u32,
    pub extra: Option<(u32, u32)>,
}

/// The fields of a [`Block`] that reconstruction reads (the motion vector
/// prediction outputs, palette colors and warp samples stay with the
/// parse), and the block's loop filter deltas.
#[derive(Clone, Copy)]
pub struct RecBlock {
    pub mi_row: u32,
    pub mi_col: u32,
    pub mv: [Mv; 2],
    pub ref_frame: [i8; 2],
    pub delta_lf: [i8; 4],
    pub bsize: u8,
    pub has_chroma: bool,
    pub avail_u: bool,
    pub avail_l: bool,
    pub avail_u_chroma: bool,
    pub avail_l_chroma: bool,
    pub skip: bool,
    pub skip_mode: bool,
    pub segment_id: u8,
    pub lossless: bool,
    pub is_inter: bool,
    pub use_intrabc: bool,
    pub y_mode: u8,
    pub uv_mode: u8,
    pub angle_delta_y: i8,
    pub angle_delta_uv: i8,
    pub cfl_alpha_u: i8,
    pub cfl_alpha_v: i8,
    pub use_filter_intra: bool,
    pub filter_intra_mode: u8,
    pub palette_size_y: u8,
    pub palette_size_uv: u8,
    pub tx_size: u8,
    pub interp_filter: [u8; 2],
    pub motion_mode: u8,
    pub interintra: bool,
    pub interintra_mode: u8,
    pub wedge_interintra: bool,
    pub wedge_index: u8,
    pub wedge_sign: u8,
    pub mask_type: u8,
    pub compound_type: u8,
    pub comp_group_idx: u8,
    pub compound_idx: u8,
    pub num_samples: u8,
}

impl RecBlock {
    fn of(b: &Block, delta_lf: [i8; 4]) -> Self {
        Self {
            mi_row: b.mi_row as u32,
            mi_col: b.mi_col as u32,
            mv: b.mv,
            ref_frame: b.ref_frame,
            delta_lf,
            bsize: b.bsize,
            has_chroma: b.has_chroma,
            avail_u: b.avail_u,
            avail_l: b.avail_l,
            avail_u_chroma: b.avail_u_chroma,
            avail_l_chroma: b.avail_l_chroma,
            skip: b.skip,
            skip_mode: b.skip_mode,
            segment_id: b.segment_id,
            lossless: b.lossless,
            is_inter: b.is_inter,
            use_intrabc: b.use_intrabc,
            y_mode: b.y_mode,
            uv_mode: b.uv_mode,
            angle_delta_y: b.angle_delta_y,
            angle_delta_uv: b.angle_delta_uv,
            cfl_alpha_u: b.cfl_alpha_u,
            cfl_alpha_v: b.cfl_alpha_v,
            use_filter_intra: b.use_filter_intra,
            filter_intra_mode: b.filter_intra_mode,
            palette_size_y: b.palette_size_y,
            palette_size_uv: b.palette_size_uv,
            tx_size: b.tx_size,
            interp_filter: b.interp_filter,
            motion_mode: b.motion_mode,
            interintra: b.interintra,
            interintra_mode: b.interintra_mode,
            wedge_interintra: b.wedge_interintra,
            wedge_index: b.wedge_index,
            wedge_sign: b.wedge_sign,
            mask_type: b.mask_type,
            compound_type: b.compound_type,
            comp_group_idx: b.comp_group_idx,
            compound_idx: b.compound_idx,
            num_samples: b.num_samples.min(8) as u8,
        }
    }

    /// Writes the fields into `b` (the others keep stale values that
    /// reconstruction does not read).
    fn apply(&self, b: &mut Block) {
        b.mi_row = self.mi_row as usize;
        b.mi_col = self.mi_col as usize;
        b.mv = self.mv;
        b.ref_frame = self.ref_frame;
        b.bsize = self.bsize;
        b.has_chroma = self.has_chroma;
        b.avail_u = self.avail_u;
        b.avail_l = self.avail_l;
        b.avail_u_chroma = self.avail_u_chroma;
        b.avail_l_chroma = self.avail_l_chroma;
        b.skip = self.skip;
        b.skip_mode = self.skip_mode;
        b.segment_id = self.segment_id;
        b.lossless = self.lossless;
        b.is_inter = self.is_inter;
        b.use_intrabc = self.use_intrabc;
        b.y_mode = self.y_mode;
        b.uv_mode = self.uv_mode;
        b.angle_delta_y = self.angle_delta_y;
        b.angle_delta_uv = self.angle_delta_uv;
        b.cfl_alpha_u = self.cfl_alpha_u;
        b.cfl_alpha_v = self.cfl_alpha_v;
        b.use_filter_intra = self.use_filter_intra;
        b.filter_intra_mode = self.filter_intra_mode;
        b.palette_size_y = self.palette_size_y;
        b.palette_size_uv = self.palette_size_uv;
        b.tx_size = self.tx_size;
        b.interp_filter = self.interp_filter;
        b.motion_mode = self.motion_mode;
        b.interintra = self.interintra;
        b.interintra_mode = self.interintra_mode;
        b.wedge_interintra = self.wedge_interintra;
        b.wedge_index = self.wedge_index;
        b.wedge_sign = self.wedge_sign;
        b.mask_type = self.mask_type;
        b.compound_type = self.compound_type;
        b.comp_group_idx = self.comp_group_idx;
        b.compound_idx = self.compound_idx;
        b.num_samples = usize::from(self.num_samples);
    }
}

/// The mode info stored for block `b` (see [`TileDec::write_mode_info`]).
#[inline(always)]
fn mode_info_of(b: &Block) -> MiInfo {
    MiFields {
        ref_frame: b.ref_frame,
        mv: if b.ref_frame[1] > INTRA_FRAME {
            b.mv
        } else {
            [b.mv[0], Mv::ZERO]
        },
        bsize: b.bsize,
        y_mode: b.y_mode,
        is_inter: b.is_inter,
        skip: b.skip,
        skip_mode: b.skip_mode,
        segment_id: b.segment_id,
        tx_size: b.tx_size,
        palette_size: [b.palette_size_y, b.palette_size_uv],
        comp_group_idx: b.comp_group_idx,
        compound_idx: b.compound_idx,
        interp_filter: b.interp_filter,
        motion_mode: b.motion_mode,
    }
    .into()
}

/// A transform block of a [`ReconLog`]: the arguments of transform_block(),
/// and, when it has coefficients, their type and nonzero region, and its
/// nonzero coefficients: `ncoef` entries at `coef`, each `pos | value << 10`
/// with `pos` the index in a block of rows of `min(width, 32)`. With `raw`,
/// `value` is the signed level, dequantized on replay with `deq` (DC and
/// AC quantizers, log2 of the denominator), and the nonzero region is
/// found then.
#[derive(Clone, Copy)]
pub struct TxOp {
    pub plane: u8,
    pub tx: u8,
    pub tx_type: u8,
    pub coded: bool,
    pub nz_rows: u8,
    pub nz_cols: u8,
    pub x: u16,
    pub y: u16,
    pub base_x: u32,
    pub base_y: u32,
    pub coef: u32,
    pub ncoef: u16,
    pub raw: bool,
    pub deq: (i32, i32, u8),
}

pub fn malformed(what: &'static str) -> CodecError {
    CodecError::Malformed { at_bit: 0, what }
}

#[derive(Clone, Copy)]
enum DeltaKind {
    Q,
    Lf,
    LfMulti(usize),
}

/// Syntax values of the block being decoded.
#[derive(Clone)]
pub struct Block {
    pub mi_row: usize,
    pub mi_col: usize,
    pub bsize: u8,
    pub has_chroma: bool,
    pub avail_u: bool,
    pub avail_l: bool,
    pub avail_u_chroma: bool,
    pub avail_l_chroma: bool,
    pub skip: bool,
    pub skip_mode: bool,
    pub segment_id: u8,
    pub lossless: bool,
    pub is_inter: bool,
    pub use_intrabc: bool,
    pub y_mode: u8,
    pub uv_mode: u8,
    pub angle_delta_y: i8,
    pub angle_delta_uv: i8,
    pub cfl_alpha_u: i8,
    pub cfl_alpha_v: i8,
    pub use_filter_intra: bool,
    pub filter_intra_mode: u8,
    pub palette_size_y: u8,
    pub palette_size_uv: u8,
    pub palette_y: [u16; 8],
    pub palette_u: [u16; 8],
    pub palette_v: [u16; 8],
    pub tx_size: u8,
    pub ref_frame: [i8; 2],
    pub mv: [Mv; 2],
    pub interp_filter: [u8; 2],
    pub motion_mode: u8,
    pub interintra: bool,
    pub interintra_mode: u8,
    pub wedge_interintra: bool,
    pub wedge_index: u8,
    pub wedge_sign: u8,
    pub mask_type: u8,
    pub compound_type: u8,
    pub comp_group_idx: u8,
    pub compound_idx: u8,
    pub ref_mv_idx: usize,
    // MV prediction outputs.
    pub num_mv_found: usize,
    pub new_mv_count: usize,
    pub ref_stack_mv: [[Mv; 2]; 9],
    pub weight_stack: [u32; 9],
    pub global_mvs: [Mv; 2],
    pub drl_ctx_stack: [u8; 9],
    pub new_mv_ctx: usize,
    pub ref_mv_ctx: usize,
    pub zero_mv_ctx: usize,
    pub num_samples: usize,
    pub num_samples_scanned: usize,
    pub cand_list: [[i32; 4]; 8],
    pub local_warp: [i32; 6],
    pub local_valid: bool,
    pub max_luma_w: usize,
    pub max_luma_h: usize,
}

impl Default for Block {
    fn default() -> Self {
        Self {
            mi_row: 0,
            mi_col: 0,
            bsize: 0,
            has_chroma: false,
            avail_u: false,
            avail_l: false,
            avail_u_chroma: false,
            avail_l_chroma: false,
            skip: false,
            skip_mode: false,
            segment_id: 0,
            lossless: false,
            is_inter: false,
            use_intrabc: false,
            y_mode: 0,
            uv_mode: 0,
            angle_delta_y: 0,
            angle_delta_uv: 0,
            cfl_alpha_u: 0,
            cfl_alpha_v: 0,
            use_filter_intra: false,
            filter_intra_mode: 0,
            palette_size_y: 0,
            palette_size_uv: 0,
            palette_y: [0; 8],
            palette_u: [0; 8],
            palette_v: [0; 8],
            tx_size: 0,
            ref_frame: [INTRA_FRAME, NONE],
            mv: [Mv::ZERO; 2],
            interp_filter: [0; 2],
            motion_mode: 0,
            interintra: false,
            interintra_mode: 0,
            wedge_interintra: false,
            wedge_index: 0,
            wedge_sign: 0,
            mask_type: 0,
            compound_type: COMPOUND_AVERAGE,
            comp_group_idx: 0,
            compound_idx: 0,
            ref_mv_idx: 0,
            num_mv_found: 0,
            new_mv_count: 0,
            ref_stack_mv: [[Mv::ZERO; 2]; 9],
            weight_stack: [0; 9],
            global_mvs: [Mv::ZERO; 2],
            drl_ctx_stack: [0; 9],
            new_mv_ctx: 0,
            ref_mv_ctx: 0,
            zero_mv_ctx: 0,
            num_samples: 0,
            num_samples_scanned: 0,
            cand_list: [[0; 4]; 8],
            local_warp: [0; 6],
            local_valid: false,
            max_luma_w: 0,
            max_luma_h: 0,
        }
    }
}

/// Encoder decisions for one superblock, consumed when the tile code runs
/// with a writing [`Coder`].
#[derive(Default)]
pub struct EncPlan {
    pub sb_row: usize,
    pub sb_col: usize,
    /// Partition at `((r - sb_row) * 32 + (c - sb_col)) * 6 + depth`, where
    /// depth is log2 of (superblock size / block size).
    pub partition: Vec<u8>,
    pub blocks: Vec<PlannedBlock>,
    /// `block_at[(r - sb_row) * 32 + (c - sb_col)]` is a block index + 1.
    pub block_at: Vec<u16>,
    pub txs: Vec<PlannedTx>,
    /// cdef_idx of each 64x64 inside the superblock, index (dr / 16) * 2 + dc / 16.
    pub cdef: [i8; 4],
    /// Target q index of each superblock of the frame (row-major,
    /// `sb_cols` per row) with its rate-distortion scale (squared quantizer
    /// step relative to the frame's); empty without delta q.
    pub sb_q: Vec<(i32, f64)>,
    pub sb_cols: usize,
    /// The frame's lambdas, which the superblock scale multiplies.
    pub base_lambda: (f64, f64, f64),
    /// The current superblock's target q index.
    pub sb_target: Option<i32>,
}

impl EncPlan {
    pub fn reset(&mut self, sb_row: usize, sb_col: usize) {
        self.sb_row = sb_row;
        self.sb_col = sb_col;
        self.partition.clear();
        self.partition.resize(32 * 32 * 6, 0);
        self.blocks.clear();
        self.block_at.clear();
        self.block_at.resize(32 * 32, 0);
        self.txs.clear();
    }

    pub fn set_partition(&mut self, r: usize, c: usize, depth: usize, p: u8) {
        let i = ((r - self.sb_row) * 32 + (c - self.sb_col)) * 6 + depth;
        self.partition[i] = p;
    }

    fn partition_at(&self, r: usize, c: usize, depth: usize) -> usize {
        let i = ((r.wrapping_sub(self.sb_row)) * 32 + c.wrapping_sub(self.sb_col)) * 6 + depth;
        self.partition.get(i).copied().unwrap_or(0) as usize
    }

    pub fn add_block(&mut self, b: PlannedBlock) {
        let i = (b.b.mi_row - self.sb_row) * 32 + (b.b.mi_col - self.sb_col);
        self.blocks.push(b);
        self.block_at[i] = self.blocks.len() as u16;
    }

    fn block(&self, r: usize, c: usize) -> Option<&PlannedBlock> {
        let i = (r.wrapping_sub(self.sb_row)) * 32 + c.wrapping_sub(self.sb_col);
        let k = *self.block_at.get(i)? as usize;
        self.blocks.get(k.checked_sub(1)?)
    }

    fn tx(&self, plane: usize, x: usize, y: usize) -> Option<&PlannedTx> {
        self.txs
            .iter()
            .find(|t| usize::from(t.plane) == plane && t.x as usize == x && t.y as usize == y)
    }
}

/// One block's planned syntax.
#[derive(Clone, Default)]
pub struct PlannedBlock {
    pub b: Block,
    /// CurrentQIndex after this block's delta (when delta q is present).
    pub q_index: i32,
    pub delta_lf: [i8; 4],
    pub color_map_y: Vec<u8>,
    pub color_map_uv: Vec<u8>,
    /// Inter transform size of each 4x4 of the block (row-major, block width
    /// in 4x4 units), used when the transform size is selected per region.
    pub inter_tx: Vec<u8>,
}

/// One transform block's planned coefficients.
#[derive(Clone, Default)]
pub struct PlannedTx {
    pub plane: u8,
    /// Position in the plane, in samples.
    pub x: u32,
    pub y: u32,
    pub tx_type: u8,
    /// Signed quantized levels in the decoder's `Quant` layout (row-major,
    /// stride min(w, 32)); all zero for an all-zero block.
    pub coefs: Vec<i32>,
}

/// A reference frame as the tile decoder sees it.
#[derive(Clone, Copy)]
pub struct RefView<'a, P> {
    pub pic: &'a Picture<P>,
    /// The samples are in [`TileDec::mirrors`] at this index instead of
    /// `pic`: the bands of final rows the reference published so far,
    /// while it still decodes.
    pub mirror: Option<usize>,
    pub upscaled_width: usize,
    pub frame_height: usize,
    pub frame_width: usize,
}

/// Rows `y0..y1` of plane `plane` of `rv` (see [`RefView::mirror`]): the
/// samples from row `y0` on and their stride, or `None` when no one band
/// holds them.
#[inline(always)]
pub fn ref_rows<'s, P>(
    rv: &'s RefView<'_, P>,
    mirrors: &'s [Mirror<P>],
    plane: usize,
    y0: usize,
    y1: usize,
) -> Option<(&'s [P], usize)> {
    match rv.mirror {
        Some(k) => mirrors.get(k)?.rows(plane, y0, y1),
        None => {
            let p = rv.pic.planes.get(plane)?;
            Some((p.data.get(y0 * p.stride..)?, p.stride))
        }
    }
}

pub struct TileDec<'a, P: Pixel, C: Coder = Msac<'a>> {
    pub seq: &'a SequenceHeader,
    pub hdr: &'a FrameHeader,
    pub fs: &'a mut FrameState,
    pub cur: &'a mut Picture<P>,
    /// Indexed by reference frame type 1..=7.
    pub refs: [Option<RefView<'a, P>>; 8],
    /// The bands of references still being decoded (see [`RefView::mirror`]).
    pub mirrors: Vec<Mirror<P>>,
    pub msac: C,
    /// The planned block (encoder only); the source of written values.
    pub pb: Block,
    pub plan: EncPlan,
    /// The source picture being encoded (encoder only).
    pub src: Option<&'a Picture<P>>,
    pub enc: crate::enc::quant::QuantParams,
    pub mode: crate::enc::rdo::ModeParams,
    pub me_scratch: crate::enc::me::MeScratch<P>,
    pub plan_q: i32,
    /// CFL alphas found by the chroma mode search (encoder only).
    pub cfl_choice: (i8, i8),
    /// Angle delta and filter intra mode found by the luma mode search.
    pub intra_choice: (i8, Option<u8>),
    /// Motion of the last partition estimate (seeds the next one).
    pub est_hint: (i32, i32),
    /// Interpolation filters of the inter choice being evaluated (encoder).
    pub cur_filters: [u8; 2],
    pub plan_delta_lf: [i8; 4],
    pub cdf: &'a mut CdfContext,
    pub mi_row_start: usize,
    pub mi_row_end: usize,
    pub mi_col_start: usize,
    pub mi_col_end: usize,
    pub sb_size: u8,
    pub sb_size4: usize,
    pub bd: u32,
    pub sub_x: bool,
    pub sub_y: bool,
    pub num_planes: usize,
    // Above context, indexed by plane 4x4 column (frame-wide).
    pub above_level: [Vec<u8>; 3],
    pub above_dc: [Vec<u8>; 3],
    pub above_seg_pred: Vec<u8>,
    // Left context, indexed by plane 4x4 row inside the superblock row.
    pub left_level: [[u8; 32]; 3],
    pub left_dc: [[u8; 32]; 3],
    pub left_seg_pred: [u8; 32],
    // Palette colors of the neighbors.
    pub above_palette: [Vec<[u16; 8]>; 2],
    pub left_palette: [[[u16; 8]; 32]; 2],
    pub delta_lf: [i8; 4],
    pub current_q_index: i32,
    pub read_deltas: bool,
    pub ref_lr_wiener: [[[i8; 3]; 2]; 3],
    pub ref_sgr_xqd: [[i8; 2]; 3],
    /// BlockDecoded[plane][y + 1][x + 1].
    pub block_decoded: [[[bool; 35]; 35]; 3],
    pub b: Block,
    pub color_map_y: Vec<u8>,
    pub color_map_uv: Vec<u8>,
    pub scratch: super::recon::Scratch<P>,
    /// When set, blocks are parsed but not predicted or reconstructed: what
    /// reconstruction needs is appended here for [`TileDec::replay`].
    pub rec: Option<Box<ReconLog>>,
    /// The luma row `cur` starts at (its chroma planes start at the
    /// subsampled row): reconstruction can write one band of rows of a
    /// picture. 0 for a whole picture.
    pub y0: usize,
    /// The mode-info row the grids of `fs` that reconstruction reads and
    /// writes (`mi`, `uv_modes`, `delta_lf`) start at. 0 for a whole
    /// frame's.
    pub mi_r0: usize,
    /// Which units of a block [`Self::write_mode_info`] writes.
    pub mi_fill: MiFill,
}

/// The units of a block whose mode info later readers need.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MiFill {
    /// All of them (the encoder reads any).
    All,
    /// Its bottom row and right column (a neighbour's above or left of
    /// them) and its first: what later blocks of reconstruction read.
    Edges,
    /// Those of [`Self::Edges`] and its odd rows: what parsing reads
    /// besides (the outer row and column scans of the motion vector
    /// search read odd rows and odd columns, by steps of 2 or more rows),
    /// the motion field (odd rows and columns) and the loop filter's
    /// levels (a block's first unit, or the odd row below it).
    OddRows,
}

#[inline]
fn neg_deinterleave(diff: i32, r: i32, max: i32) -> i32 {
    if r == 0 {
        return diff;
    }
    if r >= max - 1 {
        return max - diff - 1;
    }
    if 2 * r < max {
        if diff <= 2 * r {
            if diff & 1 != 0 {
                return r + ((diff + 1) >> 1);
            }
            return r - (diff >> 1);
        }
        diff
    } else {
        if diff <= 2 * (max - r - 1) {
            if diff & 1 != 0 {
                return r + ((diff + 1) >> 1);
            }
            return r - (diff >> 1);
        }
        max - (diff + 1)
    }
}

/// Inverse of neg_deinterleave (the encoder side of segment_id coding).
pub fn neg_interleave(x: i32, r: i32, max: i32) -> i32 {
    let diff = x - r;
    if r == 0 {
        return x;
    }
    if r >= max - 1 {
        return -x + max - 1;
    }
    if 2 * r < max {
        if diff.abs() <= r {
            return if diff > 0 {
                (diff << 1) - 1
            } else {
                (-diff) << 1
            };
        }
        x
    } else {
        if diff.abs() < max - r {
            return if diff > 0 {
                (diff << 1) - 1
            } else {
                (-diff) << 1
            };
        }
        (max - x) - 1
    }
}

/// The bytes of the first `n` (at most 16) of 16 lanes.
#[inline(always)]
fn lane_mask(n: usize) -> u128 {
    u128::MAX
        .checked_shr(128 - 8 * n.min(16) as u32)
        .unwrap_or(0)
}

/// Entries `at..at + n` (`n` at most 16) of a context array as the bytes of
/// a `u128`, the bytes from `n` on 0: a load and a mask rather than a loop
/// whose length changes from one transform block to the next. Entries past
/// the end of `d` read as 0.
#[inline(always)]
fn ctx_lanes(d: &[u8], at: usize, n: usize) -> u128 {
    match d.get(at..).and_then(|s| s.first_chunk::<16>()) {
        Some(c) => u128::from_le_bytes(*c) & lane_mask(n),
        None => (0..n.min(16)).fold(0, |x, i| {
            x | u128::from(d.get(at + i).copied().unwrap_or(0)) << (8 * i)
        }),
    }
}

/// Left context entries `ly0..ly0 + n` (wrapping at 32) as in
/// [`ctx_lanes`].
#[inline(always)]
fn left_lanes(d: &[u8; 32], ly0: usize, n: usize) -> u128 {
    if ly0 + n <= 32 {
        ctx_lanes(d, ly0, n)
    } else {
        (0..n.min(16)).fold(0, |x, i| x | u128::from(d[(ly0 + i) & 31]) << (8 * i))
    }
}

/// The sum of the bytes of `x` (each at most 15).
#[inline(always)]
fn byte_sum(x: u128) -> u32 {
    let h = |v: u64| (v.wrapping_mul(0x0101_0101_0101_0101) >> 56) as u32;
    h(x as u64) + h((x >> 64) as u64)
}

/// The dc_sign context weight of DC categories `x` (bytes 0, 1 or 2): the
/// number of 2s less the number of 1s.
#[inline(always)]
fn dc_weight(x: u128) -> i32 {
    // With S the sum and N the number of nonzero bytes: 2S - 3N.
    let nz = (x | x >> 1) & 0x0101_0101_0101_0101_0101_0101_0101_0101;
    2 * byte_sum(x) as i32 - 3 * byte_sum(nz) as i32
}

/// Sets entries `at..at + n` (`n` at most 16) of a context array to `v`,
/// those before the end of `d`.
#[inline(always)]
fn ctx_set(d: &mut [u8], at: usize, n: usize, v: u8) {
    match d.get_mut(at..).and_then(|s| s.first_chunk_mut::<16>()) {
        Some(c) => {
            let m = lane_mask(n);
            let x = (u128::from_le_bytes(*c) & !m) | ((u128::from(v) * (u128::MAX / 255)) & m);
            *c = x.to_le_bytes();
        }
        None => {
            for x in d.iter_mut().skip(at).take(n) {
                *x = v;
            }
        }
    }
}

/// Fills `h` rows of `w` entries of a map of stride `stride` from `at`
/// with `v`: one dispatch on the width for all the rows, and no library
/// call for the power-of-two widths.
#[inline(always)]
/// [`fill_rect`] of only the first unit, the bottom row and the right
/// column of the `w` x `h` rectangle at `at`.
fn fill_edges<T: Copy>(d: &mut [T], stride: usize, at: usize, w: usize, h: usize, v: T) {
    if w <= 2 && h <= 2 {
        return fill_rect(d, stride, at, w, h, v);
    }
    if h == 0 || w == 0 {
        return;
    }
    d[at] = v;
    let last = at + (h - 1) * stride;
    for m in &mut d[last..][..w] {
        *m = v;
    }
    for y in 0..h - 1 {
        d[at + y * stride + w - 1] = v;
    }
}

/// [`fill_edges`] and the odd rows of the `w` x `h` rectangle at `at`,
/// a block's: one taller than a unit starts at an even row.
#[inline(always)]
fn fill_odd_rows<T: Copy>(d: &mut [T], stride: usize, at: usize, w: usize, h: usize, v: T) {
    if h <= 2 || w == 0 {
        return fill_rect(d, stride, at, w, h, v);
    }
    d[at] = v;
    for y in 0..h {
        let row = at + y * stride;
        if y % 2 == 1 || y == h - 1 {
            for m in &mut d[row..][..w] {
                *m = v;
            }
        } else {
            d[row + w - 1] = v;
        }
    }
}

pub(crate) fn fill_rect<T: Copy>(d: &mut [T], stride: usize, at: usize, w: usize, h: usize, v: T) {
    if size_of::<T>() > 2 {
        // Larger values (mode info): a store per entry.
        for y in 0..h {
            for m in &mut d[at + y * stride..][..w] {
                *m = v;
            }
        }
        return;
    }
    macro_rules! rows {
        ($n:literal) => {
            for y in 0..h {
                d[at + y * stride..][..$n].copy_from_slice(&[v; $n]);
            }
        };
    }
    match w {
        1 => {
            for y in 0..h {
                d[at + y * stride] = v;
            }
        }
        2 => rows!(2),
        4 => rows!(4),
        8 => rows!(8),
        16 => rows!(16),
        32 => rows!(32),
        _ => {
            for y in 0..h {
                d[at + y * stride..][..w].fill(v);
            }
        }
    }
}

/// `d.fill(v)` without a library call for the short power-of-two runs of
/// 4x4 unit maps.
#[inline(always)]
pub(crate) fn fill_small<T: Copy>(d: &mut [T], v: T) {
    match d.len() {
        1 => d[0] = v,
        2 => d.copy_from_slice(&[v; 2]),
        4 => d.copy_from_slice(&[v; 4]),
        8 => d.copy_from_slice(&[v; 8]),
        16 => d.copy_from_slice(&[v; 16]),
        _ => d.fill(v),
    }
}

pub fn get_tx_class(tx_type: u8) -> usize {
    match tx_type {
        V_DCT | V_ADST | V_FLIPADST => 2,
        H_DCT | H_ADST | H_FLIPADST => 1,
        _ => 0,
    }
}

pub(crate) fn get_scan(tx: u8, tx_type: u8) -> &'static [u16] {
    if tx == TX_16X64 {
        return &DEFAULT_SCAN_16X32;
    }
    if tx == TX_64X16 {
        return &DEFAULT_SCAN_32X16;
    }
    if TX_SIZE_SQR_UP[tx as usize] == TX_64X64 {
        return &DEFAULT_SCAN_32X32;
    }
    let default = |t: u8| -> &'static [u16] {
        match t {
            0 => &DEFAULT_SCAN_4X4,
            5 => &DEFAULT_SCAN_4X8,
            6 => &DEFAULT_SCAN_8X4,
            1 => &DEFAULT_SCAN_8X8,
            7 => &DEFAULT_SCAN_8X16,
            8 => &DEFAULT_SCAN_16X8,
            2 => &DEFAULT_SCAN_16X16,
            9 => &DEFAULT_SCAN_16X32,
            10 => &DEFAULT_SCAN_32X16,
            13 => &DEFAULT_SCAN_4X16,
            14 => &DEFAULT_SCAN_16X4,
            15 => &DEFAULT_SCAN_8X32,
            16 => &DEFAULT_SCAN_32X8,
            _ => &DEFAULT_SCAN_32X32,
        }
    };
    if tx_type == IDTX {
        return default(tx);
    }
    let prefer_row = matches!(tx_type, V_DCT | V_ADST | V_FLIPADST);
    let prefer_col = matches!(tx_type, H_DCT | H_ADST | H_FLIPADST);
    if prefer_row {
        match tx {
            0 => &MROW_SCAN_4X4,
            5 => &MROW_SCAN_4X8,
            6 => &MROW_SCAN_8X4,
            1 => &MROW_SCAN_8X8,
            7 => &MROW_SCAN_8X16,
            8 => &MROW_SCAN_16X8,
            2 => &MROW_SCAN_16X16,
            13 => &MROW_SCAN_4X16,
            _ => &MROW_SCAN_16X4,
        }
    } else if prefer_col {
        match tx {
            0 => &MCOL_SCAN_4X4,
            5 => &MCOL_SCAN_4X8,
            6 => &MCOL_SCAN_8X4,
            1 => &MCOL_SCAN_8X8,
            7 => &MCOL_SCAN_8X16,
            8 => &MCOL_SCAN_16X8,
            2 => &MCOL_SCAN_16X16,
            13 => &MCOL_SCAN_4X16,
            _ => &MCOL_SCAN_16X4,
        }
    } else {
        default(tx)
    }
}

impl<'a, P: Pixel, C: Coder> TileDec<'a, P, C> {
    #[inline]
    fn is_inside(&self, r: isize, c: isize) -> bool {
        c >= self.mi_col_start as isize
            && c < self.mi_col_end as isize
            && r >= self.mi_row_start as isize
            && r < self.mi_row_end as isize
    }

    #[inline]
    pub fn mi(&self, r: usize, c: usize) -> &MiInfo {
        &self.fs.mi[(r - self.mi_r0) * self.fs.mi_stride + c]
    }

    /// The first row of plane `plane` in `cur` (see [`Self::y0`]).
    #[inline]
    pub fn row0(&self, plane: usize) -> usize {
        self.y0 >> self.plane_sub(plane).1
    }

    fn seg_feature_active(&self, feature: usize) -> bool {
        self.hdr
            .seg_feature_active(self.b.segment_id as usize, feature)
    }

    // ------------------------------------------------------------------
    // Tile entry point.

    pub fn decode_tile(&mut self) -> R<()> {
        self.begin_tile();
        let sb4 = self.sb_size4;
        let mut r = self.mi_row_start;
        while r < self.mi_row_end {
            self.begin_sb_row();
            let mut c = self.mi_col_start;
            while c < self.mi_col_end {
                self.code_superblock(r, c)?;
                c += sb4;
            }
            self.check_overread()?;
            r += sb4;
        }
        Ok(())
    }

    /// Refuses a tile whose symbols ran past the end of its data.
    fn check_overread(&self) -> R<()> {
        if self.msac.overread() {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "tile data",
            });
        }
        Ok(())
    }

    /// [`Self::decode_tile`], calling `on_row(fs, cur, log, r0, r1, true)`
    /// after each superblock row (luma 4x4 rows `r0..r1`). While recording
    /// it also calls `on_row(.., false)` every few superblocks inside a row,
    /// so reconstruction can follow closer behind.
    /// `before_row(fs, r1)` runs before each superblock row.
    pub fn decode_tile_rows(
        &mut self,
        on_row: &mut RowHook<'_, P>,
        before_row: &mut dyn FnMut(&mut FrameState, usize),
    ) -> R<()> {
        self.begin_tile();
        self.decode_rows(self.mi_row_start..self.mi_row_end, on_row, before_row)
    }

    /// The superblock rows of the tile starting in luma 4x4 rows `rows`
    /// (a part of [`Self::decode_tile_rows`], after [`Self::begin_tile`]
    /// and the rows above), with the same hooks.
    pub fn decode_rows(
        &mut self,
        rows: Range<usize>,
        on_row: &mut RowHook<'_, P>,
        before_row: &mut dyn FnMut(&mut FrameState, usize),
    ) -> R<()> {
        let sb4 = self.sb_size4;
        let mut r = rows.start.max(self.mi_row_start);
        while r < rows.end.min(self.mi_row_end) {
            self.begin_sb_row();
            let r1 = (r + sb4).min(self.mi_row_end);
            before_row(self.fs, r1);
            let mut c = self.mi_col_start;
            let mut n = 0;
            while c < self.mi_col_end {
                self.code_superblock(r, c)?;
                c += sb4;
                n += 1;
                if n % LOG_CHUNK_SBS == 0 && c < self.mi_col_end && self.rec.is_some() {
                    on_row(self.fs, self.cur, self.rec.as_deref_mut(), r, r1, false);
                }
            }
            self.check_overread()?;
            on_row(self.fs, self.cur, self.rec.as_deref_mut(), r, r1, true);
            r += sb4;
        }
        Ok(())
    }

    /// Start of decode_tile(): clears the above context and the per-tile state.
    pub fn begin_tile(&mut self) {
        // Each plane's contexts are indexed in that plane's 4x4 units.
        for p in 0..3 {
            let (sx, _) = self.plane_sub(p);
            let s = self.mi_col_start >> sx;
            let e = ((self.mi_col_end >> sx) + 32).min(self.above_seg_pred.len());
            for v in &mut self.above_level[p][s..e] {
                *v = 0;
            }
            for v in &mut self.above_dc[p][s..e] {
                *v = 0;
            }
        }
        let s = self.mi_col_start;
        let e = (self.mi_col_end + 32).min(self.above_seg_pred.len());
        for v in &mut self.above_seg_pred[s..e] {
            *v = 0;
        }
        self.delta_lf = [0; 4];
        self.current_q_index = i32::from(self.hdr.quant.base_q_idx);
        for p in 0..self.num_planes {
            self.ref_sgr_xqd[p] = [-32, 31];
            self.ref_lr_wiener[p] = [[3, -7, 15]; 2];
        }
    }

    /// clear_left_context() at the start of a superblock row.
    pub fn begin_sb_row(&mut self) {
        self.left_level = [[0; 32]; 3];
        self.left_dc = [[0; 32]; 3];
        self.left_seg_pred = [0; 32];
    }

    /// One superblock of decode_tile() (read_lr and decode_partition).
    pub fn code_superblock(&mut self, r: usize, c: usize) -> R<()> {
        self.read_deltas = self.hdr.delta_q_present;
        if C::WRITE && self.hdr.delta_q_present && !self.plan.sb_q.is_empty() {
            let sh = self.sb_size4.trailing_zeros();
            let i = (r >> sh) * self.plan.sb_cols + (c >> sh);
            if let Some(&(q, scale)) = self.plan.sb_q.get(i) {
                self.plan.sb_target = Some(q);
                let (l, ls, sk) = self.plan.base_lambda;
                self.enc.lambda = l * scale;
                self.mode.lambda_satd = ls * scale.sqrt();
                self.mode.skip_sse_per_px = sk * scale;
            }
        }
        self.clear_cdef(r, c);
        if let Some(log) = self.rec.as_deref_mut() {
            log.sbs.push(SbRec {
                r,
                c,
                tile: [
                    self.mi_row_start,
                    self.mi_row_end,
                    self.mi_col_start,
                    self.mi_col_end,
                ],
                first_block: log.blocks.len(),
            });
        }
        // While recording only luma's flags are read here (see `decoded`).
        let planes = if self.rec.is_some() {
            1
        } else {
            self.num_planes
        };
        self.clear_block_decoded_flags(r, c, planes);
        self.msac
            .mark(super::coder::SB_MARK | ((r as u32) << 15) | c as u32);
        self.read_lr(r, c, self.sb_size)?;
        self.decode_partition(r, c, self.sb_size)
    }

    fn clear_cdef(&mut self, r: usize, c: usize) {
        let st = self.fs.cdef_stride;
        self.fs.cdef_idx[(r >> 4) * st + (c >> 4)] = -1;
        if self.seq.use_128x128_superblock {
            self.fs.cdef_idx[(r >> 4) * st + (c >> 4) + 1] = -1;
            self.fs.cdef_idx[((r >> 4) + 1) * st + (c >> 4)] = -1;
            self.fs.cdef_idx[((r >> 4) + 1) * st + (c >> 4) + 1] = -1;
        }
    }

    fn clear_block_decoded_flags(&mut self, r: usize, c: usize, planes: usize) {
        let sb4 = self.sb_size4;
        for plane in 0..planes {
            let (sx, sy) = self.plane_sub(plane);
            let sbw4 = (self.mi_col_end - c) >> sx;
            let sbh4 = (self.mi_row_end - r) >> sy;
            let w = (sb4 >> sx) + 2;
            let h = (sb4 >> sy) + 2;
            let bd = &mut self.block_decoded[plane];
            // Row -1 is decoded left of the tile's right edge, column -1
            // above its bottom edge.
            let n = (sbw4 + 1).min(w);
            bd[0] = [false; 35];
            bd[0][..n].fill(true);
            for (y, row) in bd[1..h].iter_mut().enumerate() {
                *row = [false; 35];
                row[0] = y < sbh4;
            }
            bd[h - 1][0] = false;
        }
    }

    #[inline]
    pub fn plane_sub(&self, plane: usize) -> (u32, u32) {
        if plane > 0 {
            (u32::from(self.sub_x), u32::from(self.sub_y))
        } else {
            (0, 0)
        }
    }

    // ------------------------------------------------------------------
    // Loop restoration syntax (5.11.57, 5.11.58).

    fn read_lr(&mut self, r: usize, c: usize, bsize: u8) -> R<()> {
        if self.hdr.allow_intrabc {
            return Ok(());
        }
        let w = bw4(bsize);
        let h = bh4(bsize);
        for plane in 0..self.num_planes {
            let ftype = self.hdr.lr.frame_type[plane];
            if ftype == RESTORE_NONE {
                continue;
            }
            let (rows, cols) = lr_unit_span(self.seq, self.hdr, self.fs, plane, r..r + h, c..c + w);
            for ur in rows {
                for uc in cols.clone() {
                    self.read_lr_unit(plane, ur, uc, ftype)?;
                }
            }
        }
        Ok(())
    }

    fn subexp_bool(&mut self, num_syms: u32, k: u32, value: u32) -> u32 {
        let mut i = 0u32;
        let mut mk = 0u32;
        loop {
            let b2 = if i != 0 { k + i - 1 } else { k };
            let a = 1u32 << b2;
            if num_syms <= mk + 3 * a {
                return self.msac.ns(num_syms - mk, value.wrapping_sub(mk)) + mk;
            }
            let more = value >= mk + a;
            if self.msac.literal(1, u32::from(more)) != 0 {
                i += 1;
                mk += a;
            } else {
                return self.msac.literal(b2, value.wrapping_sub(mk)) + mk;
            }
        }
    }

    /// decode_signed_subexp_with_ref_bool; `target` is the value to write.
    fn signed_subexp_with_ref_bool(
        &mut self,
        low: i32,
        high: i32,
        k: u32,
        r: i32,
        target: i32,
    ) -> i32 {
        let mx = high - low;
        let r = r - low;
        let recenter = |r: i32, v: i32| {
            if v > 2 * r {
                v
            } else if v >= r {
                (v - r) * 2
            } else {
                (r - v) * 2 - 1
            }
        };
        let enc = if C::WRITE {
            let x = target - low;
            if (r << 1) <= mx {
                recenter(r, x)
            } else {
                recenter(mx - 1 - r, mx - 1 - x)
            }
        } else {
            0
        };
        let v = self.subexp_bool(mx as u32, k, enc.max(0) as u32) as i32;
        let inv = |r: i32, v: i32| {
            if v > 2 * r {
                v
            } else if v & 1 != 0 {
                r - ((v + 1) >> 1)
            } else {
                r + (v >> 1)
            }
        };
        let x = if (r << 1) <= mx {
            inv(r, v)
        } else {
            mx - 1 - inv(mx - 1 - r, v)
        };
        x + low
    }

    fn read_lr_unit(&mut self, plane: usize, ur: usize, uc: usize, ftype: u8) -> R<()> {
        let idx = ur * self.fs.lr_cols[plane] + uc;
        // When encoding, the unit holds the planned parameters.
        let plan = self.fs.lr[plane][idx];
        let rtype = if ftype == RESTORE_WIENER {
            if self
                .msac
                .bool_adapt(&mut self.cdf.use_wiener, plan.kind == RESTORE_WIENER)
            {
                RESTORE_WIENER
            } else {
                RESTORE_NONE
            }
        } else if ftype == RESTORE_SGRPROJ {
            if self
                .msac
                .bool_adapt(&mut self.cdf.use_sgrproj, plan.kind == RESTORE_SGRPROJ)
            {
                RESTORE_SGRPROJ
            } else {
                RESTORE_NONE
            }
        } else {
            debug_assert_eq!(ftype, RESTORE_SWITCHABLE);
            self.msac
                .symbol(&mut self.cdf.restoration_type, plan.kind as usize) as u8
        };
        let mut unit = self.fs.lr[plane][idx];
        unit.kind = rtype;
        const WMIN: [i32; 3] = [-5, -23, -17];
        const WMAX: [i32; 3] = [10, 8, 46];
        const WK: [u32; 3] = [1, 2, 3];
        if rtype == RESTORE_WIENER {
            for pass in 0..2 {
                let first = if plane > 0 {
                    unit.wiener[pass][0] = 0;
                    1
                } else {
                    0
                };
                for j in first..3 {
                    let v = self.signed_subexp_with_ref_bool(
                        WMIN[j],
                        WMAX[j] + 1,
                        WK[j],
                        i32::from(self.ref_lr_wiener[plane][pass][j]),
                        i32::from(plan.wiener[pass][j]),
                    );
                    unit.wiener[pass][j] = v as i8;
                    self.ref_lr_wiener[plane][pass][j] = v as i8;
                }
            }
        } else if rtype == RESTORE_SGRPROJ {
            let set = self.msac.literal(4, u32::from(plan.sgr_set)) as usize;
            unit.sgr_set = set as u8;
            const XMIN: [i32; 2] = [-96, -32];
            const XMAX: [i32; 2] = [31, 95];
            for i in 0..2 {
                let radius = SGR_PARAMS[set][i * 2];
                let v = if radius != 0 {
                    self.signed_subexp_with_ref_bool(
                        XMIN[i],
                        XMAX[i] + 1,
                        4,
                        i32::from(self.ref_sgr_xqd[plane][i]),
                        i32::from(plan.sgr_xqd[i]),
                    )
                } else if i == 1 {
                    ((1 << 7) - i32::from(self.ref_sgr_xqd[plane][0])).clamp(XMIN[i], XMAX[i])
                } else {
                    0
                };
                unit.sgr_xqd[i] = v as i8;
                self.ref_sgr_xqd[plane][i] = v as i8;
            }
        }
        self.fs.lr[plane][idx] = unit;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Partition (5.11.4).

    fn partition_ctx(&self, r: usize, c: usize, bsize: u8, avail_u: bool, avail_l: bool) -> usize {
        let bsl = mi_width_log2(bsize);
        let above = avail_u && mi_width_log2(self.mi(r - 1, c).bsize) < bsl;
        let left = avail_l && mi_height_log2(self.mi(r, c - 1).bsize) < bsl;
        usize::from(left) * 2 + usize::from(above)
    }

    pub(crate) fn decode_partition(&mut self, r: usize, c: usize, bsize: u8) -> R<()> {
        if r >= self.fs.mi_rows || c >= self.fs.mi_cols {
            return Ok(());
        }
        let avail_u = self.is_inside(r as isize - 1, c as isize);
        let avail_l = self.is_inside(r as isize, c as isize - 1);
        let num4x4 = bw4(bsize);
        let half = num4x4 >> 1;
        let quarter = half >> 1;
        let has_rows = (r + half) < self.fs.mi_rows;
        let has_cols = (c + half) < self.fs.mi_cols;
        // The encoder may code the node itself, trying its partitions.
        if C::WRITE
            && bsize >= BLOCK_8X8
            && self.plan.blocks.is_empty()
            && self.enc_partition_trials(r, c, bsize, has_rows && has_cols)?
        {
            return Ok(());
        }
        let partition: usize = if bsize < BLOCK_8X8 {
            0
        } else {
            let ctx = self.partition_ctx(r, c, bsize, avail_u, avail_l);
            let bsl = mi_width_log2(bsize);
            let want = if C::WRITE {
                if self.plan.blocks.is_empty() {
                    self.enc_partition(r, c, bsize, has_rows, has_cols)
                } else {
                    let depth = (self.sb_size4 / num4x4).trailing_zeros() as usize;
                    self.plan.partition_at(r, c, depth)
                }
            } else {
                0
            };
            if has_rows && has_cols {
                match bsl {
                    1 => self.msac.symbol(&mut self.cdf.partition_w8[ctx], want),
                    2 => self.msac.symbol(&mut self.cdf.partition_w16[ctx], want),
                    3 => self.msac.symbol(&mut self.cdf.partition_w32[ctx], want),
                    4 => self.msac.symbol(&mut self.cdf.partition_w64[ctx], want),
                    _ => self.msac.symbol(&mut self.cdf.partition_w128[ctx], want),
                }
            } else if has_cols || has_rows {
                // split_or_horz / split_or_vert: cdf[0] = 32768 - psum,
                // inverted form = psum.
                let mask: u16 = if has_cols {
                    (1 << 2) | (1 << 3) | (1 << 4) | (1 << 6) | (1 << 7)
                } else {
                    (1 << 1) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6)
                } | if bsize == BLOCK_128X128 {
                    0
                } else if has_cols {
                    1 << 9
                } else {
                    1 << 8
                };
                let v = want == 3;
                let m = &mut self.msac;
                let bit = match bsl {
                    1 => m.bool_masked(&self.cdf.partition_w8[ctx], mask, v),
                    2 => m.bool_masked(&self.cdf.partition_w16[ctx], mask, v),
                    3 => m.bool_masked(&self.cdf.partition_w32[ctx], mask, v),
                    4 => m.bool_masked(&self.cdf.partition_w64[ctx], mask, v),
                    _ => m.bool_masked(&self.cdf.partition_w128[ctx], mask, v),
                };
                if has_cols {
                    if bit { 3 } else { 1 }
                } else if bit {
                    3
                } else {
                    2
                }
            } else {
                3
            }
        };
        let sub = PARTITION_SUBSIZE[partition][bsize as usize];
        let split = PARTITION_SUBSIZE[3][bsize as usize];
        if sub == super::types::BLOCK_INVALID {
            return Err(malformed("partition"));
        }
        match partition {
            0 => self.decode_block(r, c, sub)?,
            1 => {
                self.decode_block(r, c, sub)?;
                if has_rows {
                    self.decode_block(r + half, c, sub)?;
                }
            }
            2 => {
                self.decode_block(r, c, sub)?;
                if has_cols {
                    self.decode_block(r, c + half, sub)?;
                }
            }
            3 => {
                self.decode_partition(r, c, sub)?;
                self.decode_partition(r, c + half, sub)?;
                self.decode_partition(r + half, c, sub)?;
                self.decode_partition(r + half, c + half, sub)?;
            }
            4 => {
                self.decode_block(r, c, split)?;
                self.decode_block(r, c + half, split)?;
                self.decode_block(r + half, c, sub)?;
            }
            5 => {
                self.decode_block(r, c, sub)?;
                self.decode_block(r + half, c, split)?;
                self.decode_block(r + half, c + half, split)?;
            }
            6 => {
                self.decode_block(r, c, split)?;
                self.decode_block(r + half, c, split)?;
                self.decode_block(r, c + half, sub)?;
            }
            7 => {
                self.decode_block(r, c, sub)?;
                self.decode_block(r, c + half, split)?;
                self.decode_block(r + half, c + half, split)?;
            }
            8 => {
                for i in 0..4 {
                    if i < 3 || r + quarter * 3 < self.fs.mi_rows {
                        self.decode_block(r + quarter * i, c, sub)?;
                    }
                }
            }
            _ => {
                for i in 0..4 {
                    if i < 3 || c + quarter * 3 < self.fs.mi_cols {
                        self.decode_block(r, c + quarter * i, sub)?;
                    }
                }
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Block (5.11.5).

    fn decode_block(&mut self, r: usize, c: usize, bsize: u8) -> R<()> {
        // The other fields keep the previous block's values until the
        // syntax below sets them.
        let w4 = bw4(bsize);
        let h4 = bh4(bsize);
        let has_chroma =
            if (h4 == 1 && self.sub_y && (r & 1) == 0) || (w4 == 1 && self.sub_x && (c & 1) == 0) {
                false
            } else {
                self.num_planes > 1
            };
        let avail_u = self.is_inside(r as isize - 1, c as isize);
        let avail_l = self.is_inside(r as isize, c as isize - 1);
        let (mut avail_u_chroma, mut avail_l_chroma) = (avail_u, avail_l);
        if has_chroma {
            if self.sub_y && h4 == 1 {
                avail_u_chroma = self.is_inside(r as isize - 2, c as isize);
            }
            if self.sub_x && w4 == 1 {
                avail_l_chroma = self.is_inside(r as isize, c as isize - 2);
            }
        } else {
            avail_u_chroma = false;
            avail_l_chroma = false;
        }
        let b = &mut self.b;
        b.mi_row = r;
        b.mi_col = c;
        b.bsize = bsize;
        b.has_chroma = has_chroma;
        if has_chroma
            && plane_residual_size(bsize, self.sub_x, self.sub_y) == super::types::BLOCK_INVALID
        {
            return Err(malformed("block size for this subsampling"));
        }
        b.avail_u = avail_u;
        b.avail_l = avail_l;
        b.avail_u_chroma = avail_u_chroma;
        b.avail_l_chroma = avail_l_chroma;
        if C::WRITE && self.plan.blocks.is_empty() {
            self.enc_block();
        } else if C::WRITE {
            let Some(pbk) = self.plan.block(r, c) else {
                return Err(malformed("no planned block"));
            };
            self.pb = pbk.b.clone();
            self.plan_q = pbk.q_index;
            self.plan_delta_lf = pbk.delta_lf;
            let n = pbk.color_map_y.len().min(self.color_map_y.len());
            self.color_map_y[..n].copy_from_slice(&pbk.color_map_y[..n]);
            let n = pbk.color_map_uv.len().min(self.color_map_uv.len());
            self.color_map_uv[..n].copy_from_slice(&pbk.color_map_uv[..n]);
        }
        if self.hdr.frame_is_intra {
            self.intra_frame_mode_info()?;
        } else {
            self.inter_frame_mode_info()?;
        }
        self.palette_tokens()?;
        self.read_block_tx_size()?;
        if self.b.skip {
            self.reset_block_context();
        }
        self.store_mode_info_pre();
        #[cfg(feature = "stats")]
        self.count_tools();
        if let Some(log) = self.rec.as_deref_mut() {
            let b = &self.b;
            let palette = b.palette_size_y != 0 || b.palette_size_uv != 0;
            let warp = b.motion_mode == LOCALWARP;
            let extra = (palette || warp).then(|| {
                let at = (log.extra.len() as u32, log.cands.len() as u32);
                if palette {
                    for c in [&b.palette_y, &b.palette_u, &b.palette_v] {
                        log.extra.extend_from_slice(c);
                    }
                    let maps = self.color_map_y.iter().chain(&self.color_map_uv);
                    log.extra.extend(maps.map(|&v| u16::from(v)));
                }
                if warp {
                    log.cands
                        .extend_from_slice(&b.cand_list[..b.num_samples.min(8)]);
                }
                at
            });
            log.blocks.push(BlockRec {
                b: RecBlock::of(b, self.delta_lf),
                first_op: log.ops.len() as u32,
                extra,
            });
        } else {
            self.compute_prediction()?;
        }
        self.residual()?;
        self.store_mode_info_post();
        Ok(())
    }

    fn reset_block_context(&mut self) {
        let b = &self.b;
        let planes = if b.has_chroma { 3 } else { 1 };
        let sb_mask = self.sb_size4 - 1;
        for plane in 0..planes {
            let (sx, sy) = self.plane_sub(plane);
            let x0 = b.mi_col >> sx;
            let x1 = (b.mi_col + bw4(b.bsize)) >> sx;
            for i in x0..x1 {
                self.above_level[plane][i] = 0;
                self.above_dc[plane][i] = 0;
            }
            let y0 = (b.mi_row & sb_mask) >> sy;
            let y1 = ((b.mi_row & sb_mask) + bh4(b.bsize)) >> sy;
            for i in y0..y1.min(32) {
                self.left_level[plane][i] = 0;
                self.left_dc[plane][i] = 0;
            }
        }
    }

    /// The part of the block's mode info that prediction needs (spec stores
    /// it before compute_prediction). Returns the mode info stored.
    fn store_mode_info_pre(&mut self) {
        let info = mode_info_of(&self.b);
        // Kept in memory so each store is a plain 16-byte copy (field by
        // field stores of the scalarized value cost several times more).
        let info = std::hint::black_box(info);
        self.write_mode_info(&info, self.delta_lf);
        let b = &self.b;
        let w4 = bw4(b.bsize).min(self.fs.mi_cols - b.mi_col);
        let h4 = bh4(b.bsize).min(self.fs.mi_rows - b.mi_row);
        let stride = self.fs.mi_stride;
        let at = b.mi_row * stride + b.mi_col;
        fill_rect(&mut self.fs.segment_ids, stride, at, w4, h4, b.segment_id);
        // Palette colors for later palette caches, which read as many
        // colors as the neighbor's mode info says it has: none of a
        // block without a palette.
        if self.hdr.allow_screen_content_tools && (b.palette_size_y | b.palette_size_uv) != 0 {
            let sb_mask = self.sb_size4 - 1;
            for x in 0..bw4(b.bsize) {
                if b.mi_col + x < self.above_palette[0].len() {
                    self.above_palette[0][b.mi_col + x] = b.palette_y;
                    self.above_palette[1][b.mi_col + x] = b.palette_u;
                }
            }
            for y in 0..bh4(b.bsize) {
                let ly = ((b.mi_row & sb_mask) + y) & 31;
                self.left_palette[0][ly] = b.palette_y;
                self.left_palette[1][ly] = b.palette_u;
            }
        }
    }

    /// Stores `info` as the mode info of every 4x4 of the block `self.b`
    /// (and its uv_mode for intra blocks with chroma).
    pub fn write_mode_info(&mut self, info: &MiInfo, delta_lf: [i8; 4]) {
        let b = &self.b;
        let w4 = bw4(b.bsize).min(self.fs.mi_cols - b.mi_col);
        let h4 = bh4(b.bsize).min(self.fs.mi_rows - b.mi_row);
        let stride = self.fs.mi_stride;
        let set_uv = b.ref_frame[0] == INTRA_FRAME && b.has_chroma;
        let at = (b.mi_row - self.mi_r0) * stride + b.mi_col;
        match self.mi_fill {
            MiFill::Edges => {
                fill_edges(&mut self.fs.mi, stride, at, w4, h4, *info);
                if set_uv {
                    fill_edges(&mut self.fs.uv_modes, stride, at, w4, h4, b.uv_mode);
                }
            }
            MiFill::OddRows => {
                fill_odd_rows(&mut self.fs.mi, stride, at, w4, h4, *info);
                if set_uv {
                    fill_rect(&mut self.fs.uv_modes, stride, at, w4, h4, b.uv_mode);
                }
            }
            MiFill::All => {
                // Stores of the value, not a library copy per row.
                fill_rect(&mut self.fs.mi, stride, at, w4, h4, *info);
                if set_uv {
                    fill_rect(&mut self.fs.uv_modes, stride, at, w4, h4, b.uv_mode);
                }
            }
        }
        if self.hdr.delta_lf_present {
            let n = self.fs.mi.len();
            if self.fs.delta_lf.len() != n {
                self.fs.delta_lf.resize(n, [0; 4]);
            }
            fill_rect(&mut self.fs.delta_lf, stride, at, w4, h4, delta_lf);
        }
    }

    fn store_mode_info_post(&mut self) {}

    #[cfg(feature = "stats")]
    fn count_tools(&self) {
        use super::stats::{BSIZE, QIDX, SKIP, TXSIZE, YMODE, add, hit};
        let b = &self.b;
        hit(if b.is_inter { 1 } else { 0 });
        let area = (bw4(b.bsize) * bh4(b.bsize)) as u64;
        add(&BSIZE, b.bsize as usize, area);
        add(&YMODE, b.y_mode as usize, area);
        if !b.is_inter && b.has_chroma {
            add(&super::stats::UVMODE, b.uv_mode as usize, area);
        }
        add(&TXSIZE, b.tx_size as usize, area);
        if b.skip {
            SKIP.fetch_add(area, std::sync::atomic::Ordering::Relaxed);
        }
        QIDX.store(
            u64::from(self.hdr.quant.base_q_idx),
            std::sync::atomic::Ordering::Relaxed,
        );
        if b.ref_frame[1] > INTRA_FRAME {
            hit(2);
            match b.compound_type {
                COMPOUND_WEDGE => hit(3),
                COMPOUND_DIFFWTD => hit(4),
                COMPOUND_DISTANCE => hit(5),
                _ => {}
            }
        }
        if b.interintra {
            hit(6);
        }
        if b.motion_mode == OBMC {
            hit(7);
        }
        if b.motion_mode == LOCALWARP {
            hit(8);
        }
        if b.is_inter
            && (b.y_mode == GLOBALMV || b.y_mode == GLOBAL_GLOBALMV)
            && self.hdr.gm_type[b.ref_frame[0].max(0) as usize] > super::obu::TRANSLATION
        {
            hit(9);
        }
        if b.is_inter && !b.use_intrabc && self.is_scaled(b.ref_frame[0]) {
            hit(10);
        }
        if b.palette_size_y > 0 || b.palette_size_uv > 0 {
            hit(11);
        }
        if b.use_intrabc {
            hit(12);
        }
        if b.use_filter_intra {
            hit(13);
        }
        if !b.is_inter && b.uv_mode == UV_CFL_PRED {
            hit(14);
        }
        if b.lossless {
            hit(15);
        }
    }

    // ------------------------------------------------------------------
    // Intra frame mode info (5.11.7).

    fn intra_frame_mode_info(&mut self) -> R<()> {
        self.b.skip = false;
        let pre_skip = self.hdr.segmentation.seg_id_pre_skip;
        if pre_skip {
            self.intra_segment_id()?;
        }
        self.b.skip_mode = false;
        self.read_skip();
        if !pre_skip {
            self.intra_segment_id()?;
        }
        self.read_cdef();
        self.read_delta_qindex();
        self.read_delta_lf();
        self.read_deltas = false;
        self.b.ref_frame = [INTRA_FRAME, NONE];
        self.b.use_intrabc = if self.hdr.allow_intrabc {
            self.msac
                .bool_adapt(&mut self.cdf.intrabc, self.pb.use_intrabc)
        } else {
            false
        };
        self.b.palette_size_y = 0;
        self.b.palette_size_uv = 0;
        self.b.use_filter_intra = false;
        self.b.interintra = false;
        self.b.motion_mode = SIMPLE;
        self.b.compound_type = COMPOUND_AVERAGE;
        self.b.mv = [Mv::ZERO; 2];
        self.b.angle_delta_y = 0;
        self.b.angle_delta_uv = 0;
        if self.b.use_intrabc {
            self.b.is_inter = true;
            self.b.y_mode = DC_PRED;
            self.b.uv_mode = DC_PRED;
            self.b.interp_filter = [3, 3];
            self.find_mv_stack(false);
            self.assign_mv(false)?;
        } else {
            self.b.is_inter = false;
            let above = if self.b.avail_u {
                self.mi(self.b.mi_row - 1, self.b.mi_col).y_mode
            } else {
                DC_PRED
            };
            let left = if self.b.avail_l {
                self.mi(self.b.mi_row, self.b.mi_col - 1).y_mode
            } else {
                DC_PRED
            };
            let a = INTRA_MODE_CONTEXT[above as usize] as usize;
            let l = INTRA_MODE_CONTEXT[left as usize] as usize;
            self.b.y_mode = self.msac.symbol(
                &mut self.cdf.intra_frame_y_mode[a][l],
                self.pb.y_mode as usize,
            ) as u8;
            self.intra_mode_info_common()?;
        }
        Ok(())
    }

    fn intra_mode_info_common(&mut self) -> R<()> {
        self.intra_angle_info_y();
        if self.b.has_chroma {
            self.read_uv_mode();
            if self.b.uv_mode == UV_CFL_PRED {
                self.read_cfl_alphas();
            }
            self.intra_angle_info_uv();
        } else {
            self.b.uv_mode = DC_PRED;
        }
        self.b.palette_size_y = 0;
        self.b.palette_size_uv = 0;
        let bs = self.b.bsize;
        if bs >= BLOCK_8X8
            && block_width(bs) <= 64
            && block_height(bs) <= 64
            && self.hdr.allow_screen_content_tools
        {
            self.palette_mode_info()?;
        }
        self.filter_intra_mode_info();
        Ok(())
    }

    fn intra_segment_id(&mut self) -> R<()> {
        if self.hdr.segmentation.enabled {
            self.read_segment_id()?;
        } else {
            self.b.segment_id = 0;
        }
        self.b.lossless = self.hdr.lossless_array[self.b.segment_id as usize];
        Ok(())
    }

    fn read_segment_id(&mut self) -> R<()> {
        let (r, c) = (self.b.mi_row, self.b.mi_col);
        let st = self.fs.mi_stride;
        let prev_ul = if self.b.avail_u && self.b.avail_l {
            i32::from(self.fs.segment_ids[(r - 1) * st + c - 1])
        } else {
            -1
        };
        let prev_u = if self.b.avail_u {
            i32::from(self.fs.segment_ids[(r - 1) * st + c])
        } else {
            -1
        };
        let prev_l = if self.b.avail_l {
            i32::from(self.fs.segment_ids[r * st + c - 1])
        } else {
            -1
        };
        let pred = if prev_u == -1 {
            if prev_l == -1 { 0 } else { prev_l }
        } else if prev_l == -1 || prev_ul == prev_u {
            prev_u
        } else {
            prev_l
        };
        if self.b.skip {
            self.b.segment_id = pred as u8;
        } else {
            let ctx = if prev_ul < 0 {
                0
            } else if prev_ul == prev_u && prev_ul == prev_l {
                2
            } else if prev_ul == prev_u || prev_ul == prev_l || prev_u == prev_l {
                1
            } else {
                0
            };
            let max = i32::from(self.hdr.segmentation.last_active_seg_id) + 1;
            let want = if C::WRITE {
                neg_interleave(i32::from(self.pb.segment_id), pred, max).max(0) as usize
            } else {
                0
            };
            let v = self.msac.symbol(&mut self.cdf.segment_id[ctx], want) as i32;
            let s = neg_deinterleave(v, pred, max);
            if !(0..8).contains(&s) {
                return Err(malformed("segment_id"));
            }
            self.b.segment_id = s as u8;
        }
        Ok(())
    }

    fn read_skip(&mut self) {
        if self.hdr.segmentation.seg_id_pre_skip && self.seg_feature_active(SEG_LVL_SKIP) {
            self.b.skip = true;
            return;
        }
        let mut ctx = 0;
        if self.b.avail_u {
            ctx += usize::from(self.mi(self.b.mi_row - 1, self.b.mi_col).skip());
        }
        if self.b.avail_l {
            ctx += usize::from(self.mi(self.b.mi_row, self.b.mi_col - 1).skip());
        }
        self.b.skip = self.msac.bool_adapt(&mut self.cdf.skip[ctx], self.pb.skip);
    }

    fn read_skip_mode(&mut self) {
        let bs = self.b.bsize;
        if self.seg_feature_active(SEG_LVL_SKIP)
            || self.seg_feature_active(SEG_LVL_REF_FRAME)
            || self.seg_feature_active(SEG_LVL_GLOBALMV)
            || !self.hdr.skip_mode_present
            || block_width(bs) < 8
            || block_height(bs) < 8
        {
            self.b.skip_mode = false;
            return;
        }
        let mut ctx = 0;
        if self.b.avail_u {
            ctx += usize::from(self.mi(self.b.mi_row - 1, self.b.mi_col).skip_mode());
        }
        if self.b.avail_l {
            ctx += usize::from(self.mi(self.b.mi_row, self.b.mi_col - 1).skip_mode());
        }
        self.b.skip_mode = self
            .msac
            .bool_adapt(&mut self.cdf.skip_mode[ctx], self.pb.skip_mode);
    }

    fn read_cdef(&mut self) {
        if self.b.skip || self.hdr.coded_lossless || !self.seq.enable_cdef || self.hdr.allow_intrabc
        {
            return;
        }
        let st = self.fs.cdef_stride;
        let r = self.b.mi_row & !15;
        let c = self.b.mi_col & !15;
        let idx = (r >> 4) * st + (c >> 4);
        if self.fs.cdef_idx[idx] == -1 {
            self.msac.mark(idx as u32);
            let want = if C::WRITE {
                let i = ((r - self.plan.sb_row) >> 4) * 2 + ((c - self.plan.sb_col) >> 4);
                self.plan.cdef[i.min(3)].max(0) as u32
            } else {
                0
            };
            let v = self.msac.literal(u32::from(self.hdr.cdef.bits), want) as i8;
            self.fs.cdef_idx[idx] = v;
            let w4 = bw4(self.b.bsize);
            let h4 = bh4(self.b.bsize);
            let mut i = r;
            while i < r + h4 {
                let mut j = c;
                while j < c + w4 {
                    self.fs.cdef_idx[(i >> 4) * st + (j >> 4)] = v;
                    j += 16;
                }
                i += 16;
            }
        }
    }

    /// delta_q_abs / delta_lf_abs with the escape for large values.
    fn write_delta_abs(&mut self, kind: DeltaKind, want: i32) -> i32 {
        let sym = want.min(3) as usize;
        let cdf = match kind {
            DeltaKind::Q => &mut self.cdf.delta_q,
            DeltaKind::Lf => &mut self.cdf.delta_lf,
            DeltaKind::LfMulti(i) => &mut self.cdf.delta_lf_multi[i],
        };
        let mut abs = self.msac.symbol(cdf, sym) as i32;
        if abs == 3 {
            let (rem_w, bits_w) = if C::WRITE {
                let v = (want - 1).max(2) as u32;
                let rem = 31 - v.leading_zeros();
                (rem, v - (1 << rem))
            } else {
                (1, 0)
            };
            let rem = self.msac.literal(3, rem_w - 1) + 1;
            let bits = self.msac.literal(rem, bits_w) as i32;
            abs = bits + (1 << rem) + 1;
        }
        abs
    }

    fn read_delta_qindex(&mut self) {
        if self.b.bsize == self.sb_size && self.b.skip {
            return;
        }
        if self.read_deltas {
            let want = (self.plan_q - self.current_q_index) >> self.hdr.delta_q_res;
            let abs = self.write_delta_abs(DeltaKind::Q, want.abs());
            if abs != 0 {
                let sign = self.msac.literal(1, u32::from(want < 0)) != 0;
                let red = if sign { -abs } else { abs };
                self.current_q_index =
                    (self.current_q_index + (red << self.hdr.delta_q_res)).clamp(1, 255);
            }
        }
    }

    fn read_delta_lf(&mut self) {
        if self.b.bsize == self.sb_size && self.b.skip {
            return;
        }
        if self.read_deltas && self.hdr.delta_lf_present {
            let count = if self.hdr.delta_lf_multi {
                if self.num_planes > 1 { 4 } else { 2 }
            } else {
                1
            };
            for i in 0..count {
                let want = (i32::from(self.plan_delta_lf[i]) - i32::from(self.delta_lf[i]))
                    >> self.hdr.delta_lf_res;
                let kind = if self.hdr.delta_lf_multi {
                    DeltaKind::LfMulti(i)
                } else {
                    DeltaKind::Lf
                };
                let abs = self.write_delta_abs(kind, want.abs());
                if abs != 0 {
                    let sign = self.msac.literal(1, u32::from(want < 0)) != 0;
                    let red = if sign { -abs } else { abs };
                    self.delta_lf[i] = (i32::from(self.delta_lf[i])
                        + (red << self.hdr.delta_lf_res))
                        .clamp(-63, 63) as i8;
                }
            }
        }
    }

    fn intra_angle_info_y(&mut self) {
        self.b.angle_delta_y = 0;
        if self.b.bsize >= BLOCK_8X8 && is_directional(self.b.y_mode) {
            let v = self.msac.symbol(
                &mut self.cdf.angle_delta[self.b.y_mode as usize - 1],
                (self.pb.angle_delta_y + 3) as usize,
            );
            self.b.angle_delta_y = v as i8 - 3;
        }
    }

    fn intra_angle_info_uv(&mut self) {
        self.b.angle_delta_uv = 0;
        if self.b.bsize >= BLOCK_8X8 && is_directional(self.b.uv_mode) {
            let v = self.msac.symbol(
                &mut self.cdf.angle_delta[self.b.uv_mode as usize - 1],
                (self.pb.angle_delta_uv + 3) as usize,
            );
            self.b.angle_delta_uv = v as i8 - 3;
        }
    }

    fn read_uv_mode(&mut self) {
        let bs = self.b.bsize;
        let cfl_allowed =
            if self.b.lossless && plane_residual_size(bs, self.sub_x, self.sub_y) == BLOCK_4X4 {
                true
            } else {
                !self.b.lossless && block_width(bs).max(block_height(bs)) <= 32
            };
        let y = self.b.y_mode as usize;
        self.b.uv_mode = if cfl_allowed {
            self.msac.symbol(
                &mut self.cdf.uv_mode_cfl_allowed[y],
                self.pb.uv_mode as usize,
            ) as u8
        } else {
            self.msac.symbol(
                &mut self.cdf.uv_mode_cfl_not_allowed[y],
                self.pb.uv_mode as usize,
            ) as u8
        };
    }

    fn read_cfl_alphas(&mut self) {
        let sgn = |a: i8| -> i32 {
            match a.signum() {
                0 => 0,
                -1 => 1,
                _ => 2,
            }
        };
        let (au, av) = (self.pb.cfl_alpha_u, self.pb.cfl_alpha_v);
        let want = (sgn(au) * 3 + sgn(av) - 1).max(0) as usize;
        let signs = self.msac.symbol(&mut self.cdf.cfl_sign, want) as i32;
        let sign_u = (signs + 1) / 3;
        let sign_v = (signs + 1) % 3;
        self.b.cfl_alpha_u = if sign_u != 0 {
            let ctx = ((sign_u - 1) * 3 + sign_v) as usize;
            let a = self.msac.symbol(
                &mut self.cdf.cfl_alpha[ctx],
                (au.unsigned_abs().max(1) - 1) as usize,
            ) as i8
                + 1;
            if sign_u == 1 { -a } else { a }
        } else {
            0
        };
        self.b.cfl_alpha_v = if sign_v != 0 {
            let ctx = ((sign_v - 1) * 3 + sign_u) as usize;
            let a = self.msac.symbol(
                &mut self.cdf.cfl_alpha[ctx],
                (av.unsigned_abs().max(1) - 1) as usize,
            ) as i8
                + 1;
            if sign_v == 1 { -a } else { a }
        } else {
            0
        };
    }

    fn filter_intra_mode_info(&mut self) {
        self.b.use_filter_intra = false;
        let bs = self.b.bsize;
        if self.seq.enable_filter_intra
            && self.b.y_mode == DC_PRED
            && self.b.palette_size_y == 0
            && block_width(bs).max(block_height(bs)) <= 32
        {
            self.b.use_filter_intra = self.msac.bool_adapt(
                &mut self.cdf.filter_intra[bs as usize],
                self.pb.use_filter_intra,
            );
            if self.b.use_filter_intra {
                self.b.filter_intra_mode = self.msac.symbol(
                    &mut self.cdf.filter_intra_mode,
                    self.pb.filter_intra_mode as usize,
                ) as u8;
            }
        }
    }

    // ------------------------------------------------------------------
    // Palette (5.11.46, 5.11.49).

    fn palette_cache(&self, plane_type: usize) -> ([u16; 16], usize) {
        let b = &self.b;
        let above_n = if !(b.mi_row * 4).is_multiple_of(64) {
            self.mi(b.mi_row - 1, b.mi_col).palette_size()[plane_type] as usize
        } else {
            0
        };
        let left_n = if b.avail_l {
            self.mi(b.mi_row, b.mi_col - 1).palette_size()[plane_type] as usize
        } else {
            0
        };
        let above = &self.above_palette[plane_type][b.mi_col];
        let ly = ((b.mi_row & (self.sb_size4 - 1)) & 31) as isize;
        let left = &self.left_palette[plane_type][ly as usize];
        let mut cache = [0u16; 16];
        let (mut ai, mut li, mut n) = (0, 0, 0);
        while ai < above_n && li < left_n {
            let ac = above[ai];
            let lc = left[li];
            if lc < ac {
                if n == 0 || lc != cache[n - 1] {
                    cache[n] = lc;
                    n += 1;
                }
                li += 1;
            } else {
                if n == 0 || ac != cache[n - 1] {
                    cache[n] = ac;
                    n += 1;
                }
                ai += 1;
                if lc == ac {
                    li += 1;
                }
            }
        }
        while ai < above_n {
            let v = above[ai];
            ai += 1;
            if n == 0 || v != cache[n - 1] {
                cache[n] = v;
                n += 1;
            }
        }
        while li < left_n {
            let v = left[li];
            li += 1;
            if n == 0 || v != cache[n - 1] {
                cache[n] = v;
                n += 1;
            }
        }
        (cache, n)
    }

    fn read_palette_colors(
        &mut self,
        plane_type: usize,
        n: usize,
        uv: bool,
        want: [u16; 8],
    ) -> [u16; 8] {
        let bd = self.bd;
        let (cache, cache_n) = self.palette_cache(plane_type);
        let want = &want[..n];
        // Planned colors not taken from the cache, ascending.
        let mut rest = [0u16; 8];
        let mut rest_n = 0;
        if C::WRITE {
            for &c in want {
                if !cache[..cache_n].contains(&c) {
                    rest[rest_n] = c;
                    rest_n += 1;
                }
            }
            rest[..rest_n].sort_unstable();
        }
        let mut colors = [0u16; 8];
        let mut idx = 0;
        let mut i = 0;
        while i < cache_n && idx < n {
            let use_it = want.contains(&cache[i]);
            if self.msac.literal(1, u32::from(use_it)) != 0 {
                colors[idx] = cache[i];
                idx += 1;
            }
            i += 1;
        }
        let mut ri = 0;
        if idx < n {
            colors[idx] = self.msac.literal(bd, u32::from(rest[ri])) as u16;
            ri += 1;
            idx += 1;
        }
        let mut bits = 0u32;
        if idx < n {
            let min_bits = bd - 3;
            let extra = if C::WRITE {
                let mut need = 0u32;
                for k in 1..rest_n {
                    let d = u32::from(rest[k] - rest[k - 1]) - u32::from(!uv);
                    need = need.max(32 - d.leading_zeros());
                }
                need.saturating_sub(min_bits).min(3)
            } else {
                0
            };
            bits = min_bits + self.msac.literal(2, extra);
        }
        let maxv = (1i32 << bd) - 1;
        while idx < n {
            let wd = if C::WRITE && ri < rest_n {
                (i32::from(rest[ri]) - i32::from(colors[idx - 1]) - i32::from(!uv)).max(0) as u32
            } else {
                0
            };
            ri += 1;
            let mut delta = self.msac.literal(bits, wd) as i32;
            if !uv {
                delta += 1;
            }
            let v = (i32::from(colors[idx - 1]) + delta).clamp(0, maxv);
            colors[idx] = v as u16;
            let range = (1i32 << bd) - v - i32::from(!uv);
            bits = bits.min(ceil_log2(range.max(0) as u32));
            idx += 1;
        }
        colors[..n].sort_unstable();
        colors
    }

    fn palette_mode_info(&mut self) -> R<()> {
        let bs = self.b.bsize;
        let bsize_ctx = (mi_width_log2(bs) + mi_height_log2(bs) - 2) as usize;
        if self.b.y_mode == DC_PRED {
            let mut ctx = 0;
            if self.b.avail_u && self.mi(self.b.mi_row - 1, self.b.mi_col).palette_size()[0] > 0 {
                ctx += 1;
            }
            if self.b.avail_l && self.mi(self.b.mi_row, self.b.mi_col - 1).palette_size()[0] > 0 {
                ctx += 1;
            }
            if self.msac.bool_adapt(
                &mut self.cdf.palette_y_mode[bsize_ctx][ctx],
                self.pb.palette_size_y > 0,
            ) {
                let want = (self.pb.palette_size_y as usize).max(2) - 2;
                let n = self
                    .msac
                    .symbol(&mut self.cdf.palette_y_size[bsize_ctx], want)
                    + 2;
                self.b.palette_size_y = n as u8;
                self.b.palette_y = self.read_palette_colors(0, n, false, self.pb.palette_y);
            }
        }
        if self.b.has_chroma && self.b.uv_mode == DC_PRED {
            let ctx = usize::from(self.b.palette_size_y > 0);
            if self.msac.bool_adapt(
                &mut self.cdf.palette_uv_mode[ctx],
                self.pb.palette_size_uv > 0,
            ) {
                let want = (self.pb.palette_size_uv as usize).max(2) - 2;
                let n = self
                    .msac
                    .symbol(&mut self.cdf.palette_uv_size[bsize_ctx], want)
                    + 2;
                self.b.palette_size_uv = n as u8;
                self.b.palette_u = self.read_palette_colors(1, n, true, self.pb.palette_u);
                let bd = self.bd;
                // The encoder writes V colors directly.
                if self.msac.literal(1, 0) != 0 {
                    let min_bits = bd - 4;
                    let max_val = 1i32 << bd;
                    let bits = min_bits + self.msac.literal(2, 0);
                    self.b.palette_v[0] = self.msac.literal(bd, 0) as u16;
                    for idx in 1..n {
                        let mut delta = self.msac.literal(bits, 0) as i32;
                        if delta != 0 && self.msac.literal(1, 0) != 0 {
                            delta = -delta;
                        }
                        let mut val = i32::from(self.b.palette_v[idx - 1]) + delta;
                        if val < 0 {
                            val += max_val;
                        }
                        if val >= max_val {
                            val -= max_val;
                        }
                        self.b.palette_v[idx] = val.clamp(0, max_val - 1) as u16;
                    }
                } else {
                    for idx in 0..n {
                        self.b.palette_v[idx] =
                            self.msac.literal(bd, u32::from(self.pb.palette_v[idx])) as u16;
                    }
                }
            }
        }
        Ok(())
    }

    fn palette_tokens(&mut self) -> R<()> {
        let b = &self.b;
        let bh = block_height(b.bsize);
        let bw = block_width(b.bsize);
        let on_h = bh.min((self.fs.mi_rows - b.mi_row) * 4);
        let on_w = bw.min((self.fs.mi_cols - b.mi_col) * 4);
        if b.palette_size_y != 0 {
            let n = b.palette_size_y as usize;
            let mut map = std::mem::take(&mut self.color_map_y);
            self.read_color_map(&mut map, n, bw, bh, on_w, on_h, false);
            self.color_map_y = map;
        }
        let b = &self.b;
        if b.palette_size_uv != 0 {
            let n = b.palette_size_uv as usize;
            let (sx, sy) = (usize::from(self.sub_x), usize::from(self.sub_y));
            let mut bw = bw >> sx;
            let mut bh = bh >> sy;
            let mut on_w = on_w >> sx;
            let mut on_h = on_h >> sy;
            if bw < 4 {
                bw += 2;
                on_w += 2;
            }
            if bh < 4 {
                bh += 2;
                on_h += 2;
            }
            let mut map = std::mem::take(&mut self.color_map_uv);
            self.read_color_map(&mut map, n, bw, bh, on_w, on_h, true);
            self.color_map_uv = map;
        }
        Ok(())
    }

    fn read_color_map(
        &mut self,
        map: &mut [u8],
        n: usize,
        bw: usize,
        bh: usize,
        on_w: usize,
        on_h: usize,
        uv: bool,
    ) {
        // stride is bw
        map[0] = self.msac.ns(n as u32, u32::from(map[0])) as u8;
        for i in 1..(on_h + on_w - 1) {
            let jmax = i.min(on_w - 1);
            let jmin = i.saturating_sub(on_h - 1);
            let mut j = jmax as isize;
            while j >= jmin as isize {
                let (r, c) = (i - j as usize, j as usize);
                // get_palette_color_context
                let mut scores = [0u32; 8];
                let mut order = [0u8, 1, 2, 3, 4, 5, 6, 7];
                if c > 0 {
                    scores[map[r * bw + c - 1] as usize] += 2;
                }
                if r > 0 && c > 0 {
                    scores[map[(r - 1) * bw + c - 1] as usize] += 1;
                }
                if r > 0 {
                    scores[map[(r - 1) * bw + c] as usize] += 2;
                }
                for i2 in 0..3 {
                    let mut max_score = scores[i2];
                    let mut max_idx = i2;
                    for j2 in i2 + 1..n {
                        if scores[j2] > max_score {
                            max_score = scores[j2];
                            max_idx = j2;
                        }
                    }
                    if max_idx != i2 {
                        let ms = scores[max_idx];
                        let mo = order[max_idx];
                        let mut k = max_idx;
                        while k > i2 {
                            scores[k] = scores[k - 1];
                            order[k] = order[k - 1];
                            k -= 1;
                        }
                        scores[i2] = ms;
                        order[i2] = mo;
                    }
                }
                let mut hash = 0u32;
                for i2 in 0..3 {
                    hash += scores[i2] * PALETTE_COLOR_HASH_MULTIPLIERS[i2] as u32;
                }
                let ctx = PALETTE_COLOR_CONTEXT[hash as usize].max(0) as usize;
                let cdf = &mut self.cdf;
                let w = if C::WRITE {
                    let target = map[r * bw + c];
                    order.iter().position(|&o| o == target).unwrap_or(0)
                } else {
                    0
                };
                let sym = if uv {
                    match n {
                        2 => self.msac.symbol(&mut cdf.palette_size_2_uv_color[ctx], w),
                        3 => self.msac.symbol(&mut cdf.palette_size_3_uv_color[ctx], w),
                        4 => self.msac.symbol(&mut cdf.palette_size_4_uv_color[ctx], w),
                        5 => self.msac.symbol(&mut cdf.palette_size_5_uv_color[ctx], w),
                        6 => self.msac.symbol(&mut cdf.palette_size_6_uv_color[ctx], w),
                        7 => self.msac.symbol(&mut cdf.palette_size_7_uv_color[ctx], w),
                        _ => self.msac.symbol(&mut cdf.palette_size_8_uv_color[ctx], w),
                    }
                } else {
                    match n {
                        2 => self.msac.symbol(&mut cdf.palette_size_2_y_color[ctx], w),
                        3 => self.msac.symbol(&mut cdf.palette_size_3_y_color[ctx], w),
                        4 => self.msac.symbol(&mut cdf.palette_size_4_y_color[ctx], w),
                        5 => self.msac.symbol(&mut cdf.palette_size_5_y_color[ctx], w),
                        6 => self.msac.symbol(&mut cdf.palette_size_6_y_color[ctx], w),
                        7 => self.msac.symbol(&mut cdf.palette_size_7_y_color[ctx], w),
                        _ => self.msac.symbol(&mut cdf.palette_size_8_y_color[ctx], w),
                    }
                };
                map[r * bw + c] = order[sym];
                j -= 1;
            }
        }
        for i in 0..on_h {
            let v = map[i * bw + on_w - 1];
            for j in on_w..bw {
                map[i * bw + j] = v;
            }
        }
        for i in on_h..bh {
            for j in 0..bw {
                map[i * bw + j] = map[(on_h - 1) * bw + j];
            }
        }
    }

    // ------------------------------------------------------------------
    // Transform size (5.11.15 – 5.11.17).

    fn above_tx_width(&self, row: usize, col: usize) -> usize {
        if row == self.b.mi_row {
            if !self.b.avail_u {
                return 64;
            }
            let m = self.mi(row - 1, col);
            if m.skip() && m.is_inter() {
                return block_width(m.bsize);
            }
        }
        tx_w(self.fs.inter_tx_sizes[(row - 1) * self.fs.mi_stride + col])
    }

    fn left_tx_height(&self, row: usize, col: usize) -> usize {
        if col == self.b.mi_col {
            if !self.b.avail_l {
                return 64;
            }
            let m = self.mi(row, col - 1);
            if m.skip() && m.is_inter() {
                return block_height(m.bsize);
            }
        }
        tx_h(self.fs.inter_tx_sizes[row * self.fs.mi_stride + col - 1])
    }

    fn read_tx_size(&mut self, allow_select: bool) {
        if self.b.lossless {
            self.b.tx_size = TX_4X4;
            return;
        }
        let bs = self.b.bsize;
        let max_rect = MAX_TX_SIZE_RECT[bs as usize];
        let max_depth = MAX_TX_DEPTH_TABLE[bs as usize];
        self.b.tx_size = max_rect;
        if bs > BLOCK_4X4 && allow_select && self.hdr.tx_mode == TX_MODE_SELECT {
            let (r, c) = (self.b.mi_row, self.b.mi_col);
            let above_w = if self.b.avail_u && self.mi(r - 1, c).is_inter() {
                block_width(self.mi(r - 1, c).bsize)
            } else if self.b.avail_u {
                self.above_tx_width(r, c)
            } else {
                0
            };
            let left_h = if self.b.avail_l && self.mi(r, c - 1).is_inter() {
                block_height(self.mi(r, c - 1).bsize)
            } else if self.b.avail_l {
                self.left_tx_height(r, c)
            } else {
                0
            };
            let ctx =
                usize::from(above_w >= tx_w(max_rect)) + usize::from(left_h >= tx_h(max_rect));
            let mut want = 0;
            if C::WRITE {
                let mut t = max_rect;
                while t != self.pb.tx_size && want < 4 {
                    t = SPLIT_TX_SIZE[t as usize];
                    want += 1;
                }
            }
            let depth = match max_depth {
                4 => self.msac.symbol(&mut self.cdf.tx_64x64[ctx], want),
                3 => self.msac.symbol(&mut self.cdf.tx_32x32[ctx], want),
                2 => self.msac.symbol(&mut self.cdf.tx_16x16[ctx], want),
                _ => self.msac.symbol(&mut self.cdf.tx_8x8[ctx], want),
            };
            for _ in 0..depth {
                self.b.tx_size = SPLIT_TX_SIZE[self.b.tx_size as usize];
            }
        }
    }

    fn read_block_tx_size(&mut self) -> R<()> {
        let bs = self.b.bsize;
        let w4 = bw4(bs);
        let h4 = bh4(bs);
        let st = self.fs.mi_stride;
        if self.hdr.tx_mode == TX_MODE_SELECT
            && bs > BLOCK_4X4
            && self.b.is_inter
            && !self.b.skip
            && !self.b.lossless
        {
            let max_tx = MAX_TX_SIZE_RECT[bs as usize];
            let tw4 = tx_w(max_tx) / 4;
            let th4 = tx_h(max_tx) / 4;
            let mut row = self.b.mi_row;
            while row < self.b.mi_row + h4 {
                let mut col = self.b.mi_col;
                while col < self.b.mi_col + w4 {
                    self.read_var_tx_size(row, col, max_tx, 0);
                    col += tw4;
                }
                row += th4;
            }
        } else {
            self.read_tx_size(!self.b.skip || !self.b.is_inter);
            let t = self.b.tx_size;
            let at = self.b.mi_row * st + self.b.mi_col;
            fill_rect(&mut self.fs.inter_tx_sizes, st, at, w4, h4, t);
        }
        Ok(())
    }

    fn planned_inter_tx(&self, row: usize, col: usize) -> u8 {
        if self.plan.blocks.is_empty() {
            // Encoder hooks: one transform size for the whole block.
            return self.pb.tx_size;
        }
        let Some(pbk) = self.plan.block(self.b.mi_row, self.b.mi_col) else {
            return TX_4X4;
        };
        let w4 = bw4(self.b.bsize);
        let i = (row - self.b.mi_row) * w4 + (col - self.b.mi_col);
        pbk.inter_tx.get(i).copied().unwrap_or(TX_4X4)
    }

    fn read_var_tx_size(&mut self, row: usize, col: usize, tx: u8, depth: u32) {
        if row >= self.fs.mi_rows || col >= self.fs.mi_cols {
            return;
        }
        let split = if tx == TX_4X4 || depth == 2 {
            false
        } else {
            let above = self.above_tx_width(row, col) < tx_w(tx);
            let left = self.left_tx_height(row, col) < tx_h(tx);
            let bs = self.b.bsize;
            let size = 64.min(block_width(bs).max(block_height(bs)));
            let max_tx = find_tx_size(size, size);
            let sqr_up = TX_SIZE_SQR_UP[tx as usize];
            let ctx = usize::from(sqr_up != max_tx) * 3
                + (4 - max_tx as usize) * 6
                + usize::from(above)
                + usize::from(left);
            let want = if C::WRITE {
                let pt = self.planned_inter_tx(row, col);
                tx_w(pt) < tx_w(tx) || tx_h(pt) < tx_h(tx)
            } else {
                false
            };
            self.msac.bool_adapt(&mut self.cdf.txfm_split[ctx], want)
        };
        let w4 = tx_w(tx) / 4;
        let h4 = tx_h(tx) / 4;
        if split {
            let sub = SPLIT_TX_SIZE[tx as usize];
            let sw = tx_w(sub) / 4;
            let sh = tx_h(sub) / 4;
            let mut i = 0;
            while i < h4 {
                let mut j = 0;
                while j < w4 {
                    self.read_var_tx_size(row + i, col + j, sub, depth + 1);
                    j += sw;
                }
                i += sh;
            }
        } else {
            let st = self.fs.mi_stride;
            for i in 0..h4 {
                for v in &mut self.fs.inter_tx_sizes[(row + i) * st + col..][..w4] {
                    *v = tx;
                }
            }
            self.b.tx_size = tx;
        }
    }

    // ------------------------------------------------------------------
    // Inter frame mode info (5.11.18 onward).

    fn inter_frame_mode_info(&mut self) -> R<()> {
        self.b.use_intrabc = false;
        self.b.skip = false;
        self.b.palette_size_y = 0;
        self.b.palette_size_uv = 0;
        self.b.use_filter_intra = false;
        self.b.interintra = false;
        self.b.angle_delta_y = 0;
        self.b.angle_delta_uv = 0;
        self.b.motion_mode = SIMPLE;
        self.b.compound_type = COMPOUND_AVERAGE;
        self.b.comp_group_idx = 0;
        self.b.compound_idx = 1;
        self.b.mv = [Mv::ZERO; 2];
        self.inter_segment_id(true)?;
        self.read_skip_mode();
        if self.b.skip_mode {
            self.b.skip = true;
        } else {
            self.read_skip();
        }
        if !self.hdr.segmentation.seg_id_pre_skip {
            self.inter_segment_id(false)?;
        }
        self.b.lossless = self.hdr.lossless_array[self.b.segment_id as usize];
        self.read_cdef();
        self.read_delta_qindex();
        self.read_delta_lf();
        self.read_deltas = false;
        self.read_is_inter();
        if self.b.is_inter {
            self.inter_block_mode_info()?;
        } else {
            self.b.ref_frame = [INTRA_FRAME, NONE];
            let ctx = SIZE_GROUP[self.b.bsize as usize] as usize;
            self.b.y_mode = self
                .msac
                .symbol(&mut self.cdf.y_mode[ctx], self.pb.y_mode as usize)
                as u8;
            self.intra_mode_info_common()?;
        }
        Ok(())
    }

    fn get_segment_id_pred(&self) -> u8 {
        let b = &self.b;
        let Some(prev) = &self.fs.prev_segment_ids else {
            return 0;
        };
        // Waits for the end of that frame's parse. A frame that failed
        // has no ids: this frame predicts 0 and fails with it.
        let prev = prev.segment_ids();
        let xm = (self.fs.mi_cols - b.mi_col).min(bw4(b.bsize));
        let ym = (self.fs.mi_rows - b.mi_row).min(bh4(b.bsize));
        let mut seg = 7;
        for y in 0..ym {
            let at = (b.mi_row + y) * self.fs.mi_stride + b.mi_col;
            for &id in prev.get(at..at + xm).unwrap_or(&[0]) {
                seg = seg.min(id);
            }
        }
        seg
    }

    fn set_seg_pred_ctx(&mut self, v: u8) {
        let b = &self.b;
        for i in 0..bw4(b.bsize) {
            if b.mi_col + i < self.above_seg_pred.len() {
                self.above_seg_pred[b.mi_col + i] = v;
            }
        }
        let ly = b.mi_row & (self.sb_size4 - 1);
        for i in 0..bh4(b.bsize) {
            if ly + i < 32 {
                self.left_seg_pred[ly + i] = v;
            }
        }
    }

    fn inter_segment_id(&mut self, pre_skip: bool) -> R<()> {
        let seg = &self.hdr.segmentation;
        if seg.enabled {
            let predicted = self.get_segment_id_pred();
            if seg.update_map {
                if pre_skip && !seg.seg_id_pre_skip {
                    self.b.segment_id = 0;
                    return Ok(());
                }
                if !pre_skip && self.b.skip {
                    self.set_seg_pred_ctx(0);
                    self.read_segment_id()?;
                    return Ok(());
                }
                if seg.temporal_update {
                    let ly = self.b.mi_row & (self.sb_size4 - 1);
                    let ctx = usize::from(self.left_seg_pred[ly & 31])
                        + usize::from(self.above_seg_pred[self.b.mi_col]);
                    let p = self.msac.bool_adapt(
                        &mut self.cdf.segment_id_predicted[ctx],
                        self.pb.segment_id == predicted,
                    );
                    if p {
                        self.b.segment_id = predicted;
                    } else {
                        self.read_segment_id()?;
                    }
                    self.set_seg_pred_ctx(u8::from(p));
                } else {
                    self.read_segment_id()?;
                }
            } else {
                self.b.segment_id = predicted;
            }
        } else {
            self.b.segment_id = 0;
        }
        Ok(())
    }

    fn read_is_inter(&mut self) {
        if self.b.skip_mode {
            self.b.is_inter = true;
        } else if self.seg_feature_active(SEG_LVL_REF_FRAME) {
            self.b.is_inter = self.hdr.segmentation.feature_data[self.b.segment_id as usize]
                [SEG_LVL_REF_FRAME]
                != i16::from(INTRA_FRAME);
        } else if self.seg_feature_active(SEG_LVL_GLOBALMV) {
            self.b.is_inter = true;
        } else {
            let (au, al) = (self.b.avail_u, self.b.avail_l);
            let ai = au && self.mi(self.b.mi_row - 1, self.b.mi_col).ref_frame[0] <= INTRA_FRAME;
            let li = al && self.mi(self.b.mi_row, self.b.mi_col - 1).ref_frame[0] <= INTRA_FRAME;
            let ctx = if au && al {
                if li && ai { 3 } else { usize::from(li || ai) }
            } else if au || al {
                2 * usize::from(if au { ai } else { li })
            } else {
                0
            };
            self.b.is_inter = self
                .msac
                .bool_adapt(&mut self.cdf.is_inter[ctx], self.pb.is_inter);
        }
    }

    fn neighbor_refs(&self) -> ([i8; 2], [i8; 2]) {
        let b = &self.b;
        let above = if b.avail_u {
            self.mi(b.mi_row - 1, b.mi_col).ref_frame
        } else {
            [INTRA_FRAME, NONE]
        };
        let left = if b.avail_l {
            self.mi(b.mi_row, b.mi_col - 1).ref_frame
        } else {
            [INTRA_FRAME, NONE]
        };
        (above, left)
    }

    fn count_refs(&self, f: i8, above: [i8; 2], left: [i8; 2]) -> u32 {
        let mut c = 0;
        if self.b.avail_u {
            c += u32::from(above[0] == f) + u32::from(above[1] == f);
        }
        if self.b.avail_l {
            c += u32::from(left[0] == f) + u32::from(left[1] == f);
        }
        c
    }

    fn ref_ctx(&self, a: &[i8], b2: &[i8], above: [i8; 2], left: [i8; 2]) -> usize {
        let c0: u32 = a.iter().map(|&f| self.count_refs(f, above, left)).sum();
        let c1: u32 = b2.iter().map(|&f| self.count_refs(f, above, left)).sum();
        match c0.cmp(&c1) {
            std::cmp::Ordering::Less => 0,
            std::cmp::Ordering::Equal => 1,
            std::cmp::Ordering::Greater => 2,
        }
    }

    fn read_ref_frames(&mut self) -> R<()> {
        if self.b.skip_mode {
            self.b.ref_frame = [
                self.hdr.skip_mode_frame[0] as i8,
                self.hdr.skip_mode_frame[1] as i8,
            ];
            return Ok(());
        }
        if self.seg_feature_active(SEG_LVL_REF_FRAME) {
            let v =
                self.hdr.segmentation.feature_data[self.b.segment_id as usize][SEG_LVL_REF_FRAME];
            self.b.ref_frame = [v as i8, NONE];
            return Ok(());
        }
        if self.seg_feature_active(SEG_LVL_SKIP) || self.seg_feature_active(SEG_LVL_GLOBALMV) {
            self.b.ref_frame = [LAST_FRAME, NONE];
            return Ok(());
        }
        let (above, left) = self.neighbor_refs();
        let [pr0, pr1] = self.pb.ref_frame;
        let (au, al) = (self.b.avail_u, self.b.avail_l);
        let above_intra = above[0] <= INTRA_FRAME;
        let left_intra = left[0] <= INTRA_FRAME;
        let above_single = above[1] <= INTRA_FRAME;
        let left_single = left[1] <= INTRA_FRAME;
        let bwd = |f: i8| (BWDREF_FRAME..=ALTREF_FRAME).contains(&f);
        let comp = if self.hdr.reference_select && bw4(self.b.bsize).min(bh4(self.b.bsize)) >= 2 {
            let ctx = if au && al {
                if above_single && left_single {
                    usize::from(bwd(above[0]) ^ bwd(left[0]))
                } else if above_single {
                    2 + usize::from(bwd(above[0]) || above_intra)
                } else if left_single {
                    2 + usize::from(bwd(left[0]) || left_intra)
                } else {
                    4
                }
            } else if au {
                if above_single {
                    usize::from(bwd(above[0]))
                } else {
                    3
                }
            } else if al {
                if left_single {
                    usize::from(bwd(left[0]))
                } else {
                    3
                }
            } else {
                1
            };
            self.msac
                .bool_adapt(&mut self.cdf.comp_mode[ctx], pr1 > INTRA_FRAME)
        } else {
            false
        };
        const FWD: [i8; 4] = [LAST_FRAME, LAST2_FRAME, LAST3_FRAME, GOLDEN_FRAME];
        const BWD: [i8; 3] = [BWDREF_FRAME, ALTREF2_FRAME, ALTREF_FRAME];
        if comp {
            // comp_ref_type
            let samedir = |a: i8, b: i8| (a >= BWDREF_FRAME) == (b >= BWDREF_FRAME);
            let above_comp = au && !above_intra && !above_single;
            let left_comp = al && !left_intra && !left_single;
            let above_uni = above_comp && samedir(above[0], above[1]);
            let left_uni = left_comp && samedir(left[0], left[1]);
            let ctx = if au && !above_intra && al && !left_intra {
                let sd = usize::from(samedir(above[0], left[0]));
                if !above_comp && !left_comp {
                    1 + 2 * sd
                } else if !above_comp {
                    if !left_uni { 1 } else { 3 + sd }
                } else if !left_comp {
                    if !above_uni { 1 } else { 3 + sd }
                } else if !above_uni && !left_uni {
                    0
                } else if !above_uni || !left_uni {
                    2
                } else {
                    3 + usize::from((above[0] == BWDREF_FRAME) == (left[0] == BWDREF_FRAME))
                }
            } else if au && al {
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
            };
            let uni = !self.msac.bool_adapt(
                &mut self.cdf.comp_ref_type[ctx],
                (pr0 >= BWDREF_FRAME) != (pr1 >= BWDREF_FRAME),
            );
            if uni {
                let ctx = self.ref_ctx(&FWD, &BWD, above, left);
                if self
                    .msac
                    .bool_adapt(&mut self.cdf.uni_comp_ref[ctx][0], pr0 == BWDREF_FRAME)
                {
                    self.b.ref_frame = [BWDREF_FRAME, ALTREF_FRAME];
                } else {
                    let ctx1 =
                        self.ref_ctx(&[LAST2_FRAME], &[LAST3_FRAME, GOLDEN_FRAME], above, left);
                    if self
                        .msac
                        .bool_adapt(&mut self.cdf.uni_comp_ref[ctx1][1], pr1 != LAST2_FRAME)
                    {
                        let ctx2 = self.ref_ctx(&[LAST3_FRAME], &[GOLDEN_FRAME], above, left);
                        if self
                            .msac
                            .bool_adapt(&mut self.cdf.uni_comp_ref[ctx2][2], pr1 == GOLDEN_FRAME)
                        {
                            self.b.ref_frame = [LAST_FRAME, GOLDEN_FRAME];
                        } else {
                            self.b.ref_frame = [LAST_FRAME, LAST3_FRAME];
                        }
                    } else {
                        self.b.ref_frame = [LAST_FRAME, LAST2_FRAME];
                    }
                }
            } else {
                let ctx = self.ref_ctx(
                    &[LAST_FRAME, LAST2_FRAME],
                    &[LAST3_FRAME, GOLDEN_FRAME],
                    above,
                    left,
                );
                if !self.msac.bool_adapt(
                    &mut self.cdf.comp_ref[ctx][0],
                    pr0 == LAST3_FRAME || pr0 == GOLDEN_FRAME,
                ) {
                    let ctx1 = self.ref_ctx(&[LAST_FRAME], &[LAST2_FRAME], above, left);
                    self.b.ref_frame[0] = if self
                        .msac
                        .bool_adapt(&mut self.cdf.comp_ref[ctx1][1], pr0 == LAST2_FRAME)
                    {
                        LAST2_FRAME
                    } else {
                        LAST_FRAME
                    };
                } else {
                    let ctx2 = self.ref_ctx(&[LAST3_FRAME], &[GOLDEN_FRAME], above, left);
                    self.b.ref_frame[0] = if self
                        .msac
                        .bool_adapt(&mut self.cdf.comp_ref[ctx2][2], pr0 == GOLDEN_FRAME)
                    {
                        GOLDEN_FRAME
                    } else {
                        LAST3_FRAME
                    };
                }
                let ctx =
                    self.ref_ctx(&[BWDREF_FRAME, ALTREF2_FRAME], &[ALTREF_FRAME], above, left);
                if !self
                    .msac
                    .bool_adapt(&mut self.cdf.comp_bwd_ref[ctx][0], pr1 == ALTREF_FRAME)
                {
                    let ctx1 = self.ref_ctx(&[BWDREF_FRAME], &[ALTREF2_FRAME], above, left);
                    self.b.ref_frame[1] = if self
                        .msac
                        .bool_adapt(&mut self.cdf.comp_bwd_ref[ctx1][1], pr1 == ALTREF2_FRAME)
                    {
                        ALTREF2_FRAME
                    } else {
                        BWDREF_FRAME
                    };
                } else {
                    self.b.ref_frame[1] = ALTREF_FRAME;
                }
            }
        } else {
            let ctx = self.ref_ctx(&FWD, &BWD, above, left);
            if self
                .msac
                .bool_adapt(&mut self.cdf.single_ref[ctx][0], pr0 >= BWDREF_FRAME)
            {
                let ctx2 =
                    self.ref_ctx(&[BWDREF_FRAME, ALTREF2_FRAME], &[ALTREF_FRAME], above, left);
                if !self
                    .msac
                    .bool_adapt(&mut self.cdf.single_ref[ctx2][1], pr0 == ALTREF_FRAME)
                {
                    let ctx6 = self.ref_ctx(&[BWDREF_FRAME], &[ALTREF2_FRAME], above, left);
                    self.b.ref_frame[0] = if self
                        .msac
                        .bool_adapt(&mut self.cdf.single_ref[ctx6][5], pr0 == ALTREF2_FRAME)
                    {
                        ALTREF2_FRAME
                    } else {
                        BWDREF_FRAME
                    };
                } else {
                    self.b.ref_frame[0] = ALTREF_FRAME;
                }
            } else {
                let ctx3 = self.ref_ctx(
                    &[LAST_FRAME, LAST2_FRAME],
                    &[LAST3_FRAME, GOLDEN_FRAME],
                    above,
                    left,
                );
                if self.msac.bool_adapt(
                    &mut self.cdf.single_ref[ctx3][2],
                    pr0 == LAST3_FRAME || pr0 == GOLDEN_FRAME,
                ) {
                    let ctx5 = self.ref_ctx(&[LAST3_FRAME], &[GOLDEN_FRAME], above, left);
                    self.b.ref_frame[0] = if self
                        .msac
                        .bool_adapt(&mut self.cdf.single_ref[ctx5][4], pr0 == GOLDEN_FRAME)
                    {
                        GOLDEN_FRAME
                    } else {
                        LAST3_FRAME
                    };
                } else {
                    let ctx4 = self.ref_ctx(&[LAST_FRAME], &[LAST2_FRAME], above, left);
                    self.b.ref_frame[0] = if self
                        .msac
                        .bool_adapt(&mut self.cdf.single_ref[ctx4][3], pr0 == LAST2_FRAME)
                    {
                        LAST2_FRAME
                    } else {
                        LAST_FRAME
                    };
                }
            }
            self.b.ref_frame[1] = NONE;
        }
        Ok(())
    }

    fn inter_block_mode_info(&mut self) -> R<()> {
        self.b.palette_size_y = 0;
        self.b.palette_size_uv = 0;
        self.read_ref_frames()?;
        for r in self.b.ref_frame {
            if r > INTRA_FRAME && self.refs[r as usize].is_none() {
                return Err(malformed("missing reference"));
            }
        }
        let is_compound = self.b.ref_frame[1] > INTRA_FRAME;
        self.find_mv_stack(is_compound);
        if self.b.skip_mode {
            self.b.y_mode = NEAREST_NEARESTMV;
        } else if self.seg_feature_active(SEG_LVL_SKIP) || self.seg_feature_active(SEG_LVL_GLOBALMV)
        {
            self.b.y_mode = GLOBALMV;
        } else if is_compound {
            let ctx =
                COMPOUND_MODE_CTX_MAP[self.b.ref_mv_ctx >> 1][self.b.new_mv_ctx.min(4)] as usize;
            self.b.y_mode = NEAREST_NEARESTMV
                + self.msac.symbol(
                    &mut self.cdf.compound_mode[ctx],
                    self.pb.y_mode.saturating_sub(NEAREST_NEARESTMV) as usize,
                ) as u8;
        } else {
            let new_mv = self.msac.bool_adapt(
                &mut self.cdf.new_mv[self.b.new_mv_ctx],
                self.pb.y_mode != NEWMV,
            );
            if !new_mv {
                self.b.y_mode = NEWMV;
            } else {
                let zero = self.msac.bool_adapt(
                    &mut self.cdf.zero_mv[self.b.zero_mv_ctx],
                    self.pb.y_mode != GLOBALMV,
                );
                if !zero {
                    self.b.y_mode = GLOBALMV;
                } else {
                    let rm = self.msac.bool_adapt(
                        &mut self.cdf.ref_mv[self.b.ref_mv_ctx],
                        self.pb.y_mode == NEARMV,
                    );
                    self.b.y_mode = if !rm { NEARESTMV } else { NEARMV };
                }
            }
        }
        self.b.ref_mv_idx = 0;
        let ym = self.b.y_mode;
        if ym == NEWMV || ym == NEW_NEWMV {
            for idx in 0..2 {
                if self.b.num_mv_found > idx + 1 {
                    let ctx = self.b.drl_ctx_stack[idx] as usize;
                    let drl = self
                        .msac
                        .bool_adapt(&mut self.cdf.drl_mode[ctx], self.pb.ref_mv_idx > idx);
                    if !drl {
                        self.b.ref_mv_idx = idx;
                        break;
                    }
                    self.b.ref_mv_idx = idx + 1;
                }
            }
        } else if matches!(ym, NEARMV | NEAR_NEARMV | NEAR_NEWMV | NEW_NEARMV) {
            self.b.ref_mv_idx = 1;
            for idx in 1..3 {
                if self.b.num_mv_found > idx + 1 {
                    let ctx = self.b.drl_ctx_stack[idx] as usize;
                    let drl = self
                        .msac
                        .bool_adapt(&mut self.cdf.drl_mode[ctx], self.pb.ref_mv_idx > idx);
                    if !drl {
                        self.b.ref_mv_idx = idx;
                        break;
                    }
                    self.b.ref_mv_idx = idx + 1;
                }
            }
        }
        self.assign_mv(is_compound)?;
        self.read_interintra_mode(is_compound);
        self.read_motion_mode(is_compound);
        self.read_compound_type(is_compound);
        if self.hdr.interpolation_filter == SWITCHABLE {
            let dirs = if self.seq.enable_dual_filter { 2 } else { 1 };
            for dir in 0..dirs {
                if self.needs_interp_filter() {
                    let ctx = self.interp_filter_ctx(dir);
                    self.b.interp_filter[dir] = self.msac.symbol(
                        &mut self.cdf.interp_filter[ctx],
                        self.pb.interp_filter[dir] as usize,
                    ) as u8;
                } else {
                    self.b.interp_filter[dir] = 0;
                }
            }
            if !self.seq.enable_dual_filter {
                self.b.interp_filter[1] = self.b.interp_filter[0];
            }
        } else {
            self.b.interp_filter = [self.hdr.interpolation_filter; 2];
        }
        Ok(())
    }

    fn needs_interp_filter(&self) -> bool {
        let bs = self.b.bsize;
        let large = block_width(bs).min(block_height(bs)) >= 8;
        if self.b.skip_mode || self.b.motion_mode == LOCALWARP {
            false
        } else if large && self.b.y_mode == GLOBALMV {
            self.hdr.gm_type[self.b.ref_frame[0] as usize] == super::obu::TRANSLATION
        } else if large && self.b.y_mode == GLOBAL_GLOBALMV {
            self.hdr.gm_type[self.b.ref_frame[0] as usize] == super::obu::TRANSLATION
                || self.hdr.gm_type[self.b.ref_frame[1] as usize] == super::obu::TRANSLATION
        } else {
            true
        }
    }

    fn interp_filter_ctx(&self, dir: usize) -> usize {
        let b = &self.b;
        let mut ctx = ((dir & 1) * 2 + usize::from(b.ref_frame[1] > INTRA_FRAME)) * 4;
        let mut left_type = 3u8;
        let mut above_type = 3u8;
        if b.avail_l {
            let m = self.mi(b.mi_row, b.mi_col - 1);
            if m.ref_frame[0] == b.ref_frame[0] || m.ref_frame[1] == b.ref_frame[0] {
                left_type = m.interp_filter()[dir];
            }
        }
        if b.avail_u {
            let m = self.mi(b.mi_row - 1, b.mi_col);
            if m.ref_frame[0] == b.ref_frame[0] || m.ref_frame[1] == b.ref_frame[0] {
                above_type = m.interp_filter()[dir];
            }
        }
        ctx += if left_type == above_type {
            left_type as usize
        } else if left_type == 3 {
            above_type as usize
        } else if above_type == 3 {
            left_type as usize
        } else {
            3
        };
        ctx
    }

    fn read_interintra_mode(&mut self, is_compound: bool) {
        let bs = self.b.bsize;
        if !self.b.skip_mode
            && self.seq.enable_interintra_compound
            && !is_compound
            && (BLOCK_8X8..=BLOCK_32X32).contains(&bs)
        {
            let ctx = SIZE_GROUP[bs as usize] as usize - 1;
            self.b.interintra = self
                .msac
                .bool_adapt(&mut self.cdf.inter_intra[ctx], self.pb.interintra);
            if self.b.interintra {
                self.b.interintra_mode = self.msac.symbol(
                    &mut self.cdf.inter_intra_mode[ctx],
                    self.pb.interintra_mode as usize,
                ) as u8;
                self.b.ref_frame[1] = INTRA_FRAME;
                self.b.angle_delta_y = 0;
                self.b.angle_delta_uv = 0;
                self.b.use_filter_intra = false;
                self.b.wedge_interintra = self.msac.bool_adapt(
                    &mut self.cdf.wedge_inter_intra[bs as usize],
                    self.pb.wedge_interintra,
                );
                if self.b.wedge_interintra {
                    self.b.wedge_index = self.msac.symbol(
                        &mut self.cdf.wedge_index[bs as usize],
                        self.pb.wedge_index as usize,
                    ) as u8;
                    self.b.wedge_sign = 0;
                }
            }
        } else {
            self.b.interintra = false;
        }
    }

    pub fn has_overlappable_candidates(&self) -> bool {
        let b = &self.b;
        if b.avail_u {
            let w4 = bw4(b.bsize);
            let mut x4 = b.mi_col;
            while x4 < self.fs.mi_cols.min(b.mi_col + w4) {
                let x = (x4 | 1).min(self.fs.mi_cols - 1);
                if self.mi(b.mi_row - 1, x).ref_frame[0] > INTRA_FRAME {
                    return true;
                }
                x4 += 2;
            }
        }
        if b.avail_l {
            let h4 = bh4(b.bsize);
            let mut y4 = b.mi_row;
            while y4 < self.fs.mi_rows.min(b.mi_row + h4) {
                let y = (y4 | 1).min(self.fs.mi_rows - 1);
                if self.mi(y, b.mi_col - 1).ref_frame[0] > INTRA_FRAME {
                    return true;
                }
                y4 += 2;
            }
        }
        false
    }

    pub fn is_scaled(&self, ref_frame: i8) -> bool {
        let Some(rv) = &self.refs[ref_frame as usize] else {
            return false;
        };
        let fw = self.hdr.frame_width as usize;
        let fh = self.hdr.frame_height as usize;
        let xs = ((rv.upscaled_width << 14) + fw / 2) / fw;
        let ys = ((rv.frame_height << 14) + fh / 2) / fh;
        xs != 1 << 14 || ys != 1 << 14
    }

    fn read_motion_mode(&mut self, is_compound: bool) {
        self.b.motion_mode = SIMPLE;
        let b = &self.b;
        let bs = b.bsize;
        if b.skip_mode || !self.hdr.is_motion_mode_switchable {
            return;
        }
        if block_width(bs).min(block_height(bs)) < 8 {
            return;
        }
        if !self.hdr.force_integer_mv
            && (b.y_mode == GLOBALMV || b.y_mode == GLOBAL_GLOBALMV)
            && self.hdr.gm_type[b.ref_frame[0] as usize] > super::obu::TRANSLATION
        {
            return;
        }
        if is_compound || b.ref_frame[1] == INTRA_FRAME || !self.has_overlappable_candidates() {
            return;
        }
        self.find_warp_samples();
        let b = &self.b;
        if self.hdr.force_integer_mv
            || b.num_samples == 0
            || !self.hdr.allow_warped_motion
            || self.is_scaled(b.ref_frame[0])
        {
            let obmc = self.msac.bool_adapt(
                &mut self.cdf.use_obmc[bs as usize],
                self.pb.motion_mode == OBMC,
            );
            self.b.motion_mode = if obmc { OBMC } else { SIMPLE };
        } else {
            self.b.motion_mode = self.msac.symbol(
                &mut self.cdf.motion_mode[bs as usize],
                self.pb.motion_mode as usize,
            ) as u8;
        }
    }

    fn read_compound_type(&mut self, is_compound: bool) {
        self.b.comp_group_idx = 0;
        self.b.compound_idx = 1;
        if self.b.skip_mode {
            self.b.compound_type = COMPOUND_AVERAGE;
            return;
        }
        if is_compound {
            let bs = self.b.bsize;
            let n = WEDGE_BITS[bs as usize];
            let (above, left) = self.neighbor_refs();
            let (au, al) = (self.b.avail_u, self.b.avail_l);
            let (r, c) = (self.b.mi_row, self.b.mi_col);
            if self.seq.enable_masked_compound {
                let mut ctx = 0;
                if au {
                    if above[1] > INTRA_FRAME {
                        ctx += self.mi(r - 1, c).comp_group_idx() as usize;
                    } else if above[0] == ALTREF_FRAME {
                        ctx += 3;
                    }
                }
                if al {
                    if left[1] > INTRA_FRAME {
                        ctx += self.mi(r, c - 1).comp_group_idx() as usize;
                    } else if left[0] == ALTREF_FRAME {
                        ctx += 3;
                    }
                }
                self.b.comp_group_idx = u8::from(self.msac.bool_adapt(
                    &mut self.cdf.comp_group_idx[ctx.min(5)],
                    self.pb.comp_group_idx != 0,
                ));
            }
            if self.b.comp_group_idx == 0 {
                if self.seq.enable_jnt_comp {
                    let h = self.hdr;
                    let fwd = h
                        .get_relative_dist(
                            self.seq,
                            h.order_hints[self.b.ref_frame[0] as usize],
                            h.order_hint,
                        )
                        .abs();
                    let bck = h
                        .get_relative_dist(
                            self.seq,
                            h.order_hints[self.b.ref_frame[1] as usize],
                            h.order_hint,
                        )
                        .abs();
                    let mut ctx = if fwd == bck { 3 } else { 0 };
                    if au {
                        if above[1] > INTRA_FRAME {
                            ctx += self.mi(r - 1, c).compound_idx() as usize;
                        } else if above[0] == ALTREF_FRAME {
                            ctx += 1;
                        }
                    }
                    if al {
                        if left[1] > INTRA_FRAME {
                            ctx += self.mi(r, c - 1).compound_idx() as usize;
                        } else if left[0] == ALTREF_FRAME {
                            ctx += 1;
                        }
                    }
                    self.b.compound_idx = u8::from(
                        self.msac
                            .bool_adapt(&mut self.cdf.compound_idx[ctx], self.pb.compound_idx != 0),
                    );
                    self.b.compound_type = if self.b.compound_idx != 0 {
                        COMPOUND_AVERAGE
                    } else {
                        COMPOUND_DISTANCE
                    };
                } else {
                    self.b.compound_type = COMPOUND_AVERAGE;
                }
            } else if n == 0 {
                self.b.compound_type = COMPOUND_DIFFWTD;
            } else {
                self.b.compound_type = if self.msac.bool_adapt(
                    &mut self.cdf.compound_type[bs as usize],
                    self.pb.compound_type == COMPOUND_DIFFWTD,
                ) {
                    COMPOUND_DIFFWTD
                } else {
                    COMPOUND_WEDGE
                };
            }
            if self.b.compound_type == COMPOUND_WEDGE {
                self.b.wedge_index = self.msac.symbol(
                    &mut self.cdf.wedge_index[bs as usize],
                    self.pb.wedge_index as usize,
                ) as u8;
                self.b.wedge_sign = self.msac.literal(1, u32::from(self.pb.wedge_sign)) as u8;
            } else if self.b.compound_type == COMPOUND_DIFFWTD {
                self.b.mask_type = self.msac.literal(1, u32::from(self.pb.mask_type)) as u8;
            }
        } else if self.b.interintra {
            self.b.compound_type = if self.b.wedge_interintra {
                COMPOUND_WEDGE
            } else {
                COMPOUND_INTRA
            };
        } else {
            self.b.compound_type = COMPOUND_AVERAGE;
        }
    }

    fn get_mode(&self, ref_list: usize) -> u8 {
        let ym = self.b.y_mode;
        if ref_list == 0 {
            if ym < NEAREST_NEARESTMV {
                ym
            } else if matches!(ym, NEW_NEWMV | NEW_NEARESTMV | NEW_NEARMV) {
                NEWMV
            } else if matches!(ym, NEAREST_NEARESTMV | NEAREST_NEWMV) {
                NEARESTMV
            } else if matches!(ym, NEAR_NEARMV | NEAR_NEWMV) {
                NEARMV
            } else {
                GLOBALMV
            }
        } else if matches!(ym, NEW_NEWMV | NEAREST_NEWMV | NEAR_NEWMV) {
            NEWMV
        } else if matches!(ym, NEAREST_NEARESTMV | NEW_NEARESTMV) {
            NEARESTMV
        } else if matches!(ym, NEAR_NEARMV | NEW_NEARMV) {
            NEARMV
        } else {
            GLOBALMV
        }
    }

    fn assign_mv(&mut self, is_compound: bool) -> R<()> {
        for i in 0..1 + usize::from(is_compound) {
            let comp_mode = if self.b.use_intrabc {
                NEWMV
            } else {
                self.get_mode(i)
            };
            let pred = if self.b.use_intrabc {
                let mut p = self.b.ref_stack_mv[0][0];
                if p == Mv::ZERO {
                    p = self.b.ref_stack_mv[1][0];
                }
                if p == Mv::ZERO {
                    let sb4 = self.sb_size4 as i32;
                    if (self.b.mi_row as i32) - sb4 < self.mi_row_start as i32 {
                        p = Mv::new(0, -(sb4 * 4 + 256) * 8);
                    } else {
                        p = Mv::new(-(sb4 * 4 * 8), 0);
                    }
                }
                p
            } else if comp_mode == GLOBALMV {
                self.b.global_mvs[i]
            } else {
                let mut pos = if comp_mode == NEARESTMV {
                    0
                } else {
                    self.b.ref_mv_idx
                };
                if comp_mode == NEWMV && self.b.num_mv_found <= 1 {
                    pos = 0;
                }
                self.b.ref_stack_mv[pos][i]
            };
            if comp_mode == NEWMV {
                let want = (
                    i32::from(self.pb.mv[i].row) - i32::from(pred.row),
                    i32::from(self.pb.mv[i].col) - i32::from(pred.col),
                );
                let d = self.read_mv(want);
                self.b.mv[i] = Mv::new(i32::from(pred.row) + d.0, i32::from(pred.col) + d.1);
            } else {
                self.b.mv[i] = pred;
            }
        }
        if self.b.use_intrabc && !self.is_mv_valid_intrabc() {
            return Err(malformed("intrabc motion vector"));
        }
        if !self.b.use_intrabc {
            for i in 0..1 + usize::from(is_compound) {
                let mv = self.b.mv[i];
                if !(-(1 << 14) + 1..(1 << 14)).contains(&i32::from(mv.row))
                    || !(-(1 << 14) + 1..(1 << 14)).contains(&i32::from(mv.col))
                {
                    return Err(malformed("motion vector range"));
                }
            }
        }
        Ok(())
    }

    fn is_mv_valid_intrabc(&self) -> bool {
        let b = &self.b;
        let mv = b.mv[0];
        if (mv.row & 7) != 0 || (mv.col & 7) != 0 {
            return false;
        }
        let bw = block_width(b.bsize) as i32;
        let bh = block_height(b.bsize) as i32;
        let delta_row = i32::from(mv.row) >> 3;
        let delta_col = i32::from(mv.col) >> 3;
        let src_top = b.mi_row as i32 * 4 + delta_row;
        let src_left = b.mi_col as i32 * 4 + delta_col;
        let src_bottom = src_top + bh;
        let src_right = src_left + bw;
        if b.has_chroma {
            if bw < 8 && self.sub_x && src_left - 4 < self.mi_col_start as i32 * 4 {
                return false;
            }
            if bh < 8 && self.sub_y && src_top - 4 < self.mi_row_start as i32 * 4 {
                return false;
            }
        }
        if src_top < self.mi_row_start as i32 * 4
            || src_left < self.mi_col_start as i32 * 4
            || src_bottom > self.mi_row_end as i32 * 4
            || src_right > self.mi_col_end as i32 * 4
        {
            return false;
        }
        let sb_size = self.sb_size4 as i32 * 4;
        let sb_h = sb_size;
        let active_sb_row = (b.mi_row as i32 * 4) / sb_size;
        let active_sb64_col = (b.mi_col as i32 * 4) >> 6;
        let src_sb_row = (src_bottom - 1) / sb_h;
        let src_sb64_col = (src_right - 1) >> 6;
        let total_sb64_per_row = (((self.mi_col_end - self.mi_col_start) as i32 - 1) >> 4) + 1;
        let active_sb64 = active_sb_row * total_sb64_per_row + active_sb64_col;
        let src_sb64 = src_sb_row * total_sb64_per_row + src_sb64_col;
        if src_sb64 >= active_sb64 - 4 {
            return false;
        }
        let gradient = 1 + 4 + i32::from(sb_size > 64);
        let wf_offset = gradient * (active_sb_row - src_sb_row);
        if src_sb_row > active_sb_row || src_sb64_col >= active_sb64_col - 4 + wf_offset {
            return false;
        }
        true
    }

    fn read_mv_component(&mut self, ctx: usize, comp: usize, want: i32) -> i32 {
        // Encoder side: split the magnitude into class, offset bits, fraction and hp.
        let (w_sign, w_class, w_int, w_fr, w_hp) = if C::WRITE {
            let z = (want.unsigned_abs().max(1) - 1) as usize;
            let class = if z >= 2 * 4096 {
                10
            } else if z < 16 {
                0
            } else {
                (31 - ((z >> 3) as u32).leading_zeros()) as usize
            };
            let base = if class == 0 { 0 } else { 2 << (class + 2) };
            let off = z - base;
            (want < 0, class, off >> 3, (off >> 1) & 3, off & 1)
        } else {
            (false, 0, 0, 0, 0)
        };
        let cdf = &mut self.cdf;
        let sign = self.msac.bool_adapt(&mut cdf.mv_sign[ctx][comp], w_sign);
        let class = self.msac.symbol(&mut cdf.mv_class[ctx][comp], w_class);
        let mag = if class == 0 {
            let bit = usize::from(
                self.msac
                    .bool_adapt(&mut cdf.mv_class0_bit[ctx][comp], w_int != 0),
            );
            let fr = if self.hdr.force_integer_mv {
                3
            } else {
                self.msac
                    .symbol(&mut cdf.mv_class0_fr[ctx][comp][bit], w_fr)
            };
            let hp = if self.hdr.allow_high_precision_mv {
                usize::from(
                    self.msac
                        .bool_adapt(&mut cdf.mv_class0_hp[ctx][comp], w_hp != 0),
                )
            } else {
                1
            };
            ((bit << 3) | (fr << 1) | hp) + 1
        } else {
            let mut d = 0usize;
            for i in 0..class {
                d |= usize::from(
                    self.msac
                        .bool_adapt(&mut cdf.mv_bit[ctx][comp][i], (w_int >> i) & 1 != 0),
                ) << i;
            }
            let mut m = 2usize << (class + 2);
            let fr = if self.hdr.force_integer_mv {
                3
            } else {
                self.msac.symbol(&mut cdf.mv_fr[ctx][comp], w_fr)
            };
            let hp = if self.hdr.allow_high_precision_mv {
                usize::from(self.msac.bool_adapt(&mut cdf.mv_hp[ctx][comp], w_hp != 0))
            } else {
                1
            };
            m += ((d << 3) | (fr << 1) | hp) + 1;
            m
        };
        if sign { -(mag as i32) } else { mag as i32 }
    }

    /// Reads (or writes) one motion vector difference; `want` is the
    /// difference to write.
    fn read_mv(&mut self, want: (i32, i32)) -> (i32, i32) {
        let ctx = usize::from(self.b.use_intrabc);
        let wj = usize::from(want.0 != 0) * 2 + usize::from(want.1 != 0);
        let joint = self.msac.symbol(&mut self.cdf.mv_joint[ctx], wj);
        let mut d = (0, 0);
        if joint == 2 || joint == 3 {
            d.0 = self.read_mv_component(ctx, 0, want.0);
        }
        if joint == 1 || joint == 3 {
            d.1 = self.read_mv_component(ctx, 1, want.1);
        }
        d
    }

    // ------------------------------------------------------------------
    // Residual (5.11.34 onward).

    fn residual(&mut self) -> R<()> {
        // The fields used below (the whole block is large to copy).
        struct B {
            bsize: u8,
            mi_row: usize,
            mi_col: usize,
            has_chroma: bool,
            lossless: bool,
            is_inter: bool,
            tx_size: u8,
        }
        let b = B {
            bsize: self.b.bsize,
            mi_row: self.b.mi_row,
            mi_col: self.b.mi_col,
            has_chroma: self.b.has_chroma,
            lossless: self.b.lossless,
            is_inter: self.b.is_inter,
            tx_size: self.b.tx_size,
        };
        let bw = block_width(b.bsize);
        let bh = block_height(b.bsize);
        let width_chunks = (bw >> 6).max(1);
        let height_chunks = (bh >> 6).max(1);
        let chunk_size = if width_chunks > 1 || height_chunks > 1 {
            BLOCK_64X64
        } else {
            b.bsize
        };
        for cy in 0..height_chunks {
            for cx in 0..width_chunks {
                let row_chunk = b.mi_row + (cy << 4);
                let col_chunk = b.mi_col + (cx << 4);
                let planes = 1 + 2 * usize::from(b.has_chroma);
                for plane in 0..planes {
                    let tx = if b.lossless {
                        TX_4X4
                    } else {
                        self.get_tx_size(plane, b.tx_size)
                    };
                    let step_x = tx_w(tx) >> 2;
                    let step_y = tx_h(tx) >> 2;
                    let (sx, sy) = self.plane_sub(plane);
                    let psz = plane_residual_size(chunk_size, sx != 0, sy != 0);
                    let n4w = bw4(psz);
                    let n4h = bh4(psz);
                    let base_x = (col_chunk >> sx) * 4;
                    let base_y = (row_chunk >> sy) * 4;
                    if b.is_inter && !b.lossless && plane == 0 {
                        self.transform_tree(base_x, base_y, n4w * 4, n4h * 4)?;
                    } else {
                        let bxb = (b.mi_col >> sx) * 4;
                        let byb = (b.mi_row >> sy) * 4;
                        let mut y = 0;
                        while y < n4h {
                            let mut x = 0;
                            while x < n4w {
                                self.transform_block(
                                    plane,
                                    bxb,
                                    byb,
                                    tx,
                                    x + ((cx << 4) >> sx),
                                    y + ((cy << 4) >> sy),
                                )?;
                                x += step_x;
                            }
                            y += step_y;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn get_tx_size(&self, plane: usize, tx: u8) -> u8 {
        if plane == 0 {
            return tx;
        }
        let uv_tx =
            MAX_TX_SIZE_RECT[plane_residual_size(self.b.bsize, self.sub_x, self.sub_y) as usize];
        if tx_w(uv_tx) == 64 || tx_h(uv_tx) == 64 {
            if tx_w(uv_tx) == 16 {
                return TX_16X32;
            }
            if tx_h(uv_tx) == 16 {
                return TX_32X16;
            }
            return TX_32X32;
        }
        uv_tx
    }

    fn transform_tree(&mut self, sx: usize, sy: usize, w: usize, h: usize) -> R<()> {
        let max_x = self.fs.mi_cols * 4;
        let max_y = self.fs.mi_rows * 4;
        if sx >= max_x || sy >= max_y {
            return Ok(());
        }
        let row = sy >> 2;
        let col = sx >> 2;
        let luma_tx = self.fs.inter_tx_sizes[row * self.fs.mi_stride + col];
        if w <= tx_w(luma_tx) && h <= tx_h(luma_tx) {
            let tx = find_tx_size(w, h);
            self.transform_block(0, sx, sy, tx, 0, 0)?;
        } else if w > h {
            self.transform_tree(sx, sy, w / 2, h)?;
            self.transform_tree(sx + w / 2, sy, w / 2, h)?;
        } else if w < h {
            self.transform_tree(sx, sy, w, h / 2)?;
            self.transform_tree(sx, sy + h / 2, w, h / 2)?;
        } else {
            self.transform_tree(sx, sy, w / 2, h / 2)?;
            self.transform_tree(sx + w / 2, sy, w / 2, h / 2)?;
            self.transform_tree(sx, sy + h / 2, w / 2, h / 2)?;
            self.transform_tree(sx + w / 2, sy + h / 2, w / 2, h / 2)?;
        }
        Ok(())
    }

    fn transform_block(
        &mut self,
        plane: usize,
        base_x: usize,
        base_y: usize,
        tx: u8,
        x: usize,
        y: usize,
    ) -> R<()> {
        let start_x = base_x + 4 * x;
        let start_y = base_y + 4 * y;
        let (sx, sy) = self.plane_sub(plane);
        let max_x = (self.fs.mi_cols * 4) >> sx;
        let max_y = (self.fs.mi_rows * 4) >> sy;
        if start_x >= max_x || start_y >= max_y {
            return Ok(());
        }
        let recording = self.rec.is_some();
        if !recording {
            self.tx_predict(plane, start_x, start_y, x, y, tx);
        }
        let coef = self.rec.as_ref().map_or(0, |l| l.coefs.len());
        self.scratch.logged = false;
        let mut coded = false;
        if !self.b.skip {
            let eob = self.coeffs(plane, start_x, start_y, tx)?;
            if eob > 0 {
                if recording {
                    coded = true;
                } else {
                    self.reconstruct(plane, start_x, start_y, tx);
                }
            }
        }
        let row = (start_y << sy) >> 2;
        let col = (start_x << sx) >> 2;
        let step_x = tx_w(tx) >> 2;
        let step_y = tx_h(tx) >> 2;
        let st = self.fs.mi_stride;
        let at = (row >> sy) * st + (col >> sx);
        let dims = if self.b.skip && self.b.ref_frame[0] > INTRA_FRAME {
            let psz = plane_residual_size(self.b.bsize, sx != 0, sy != 0);
            lf_dims(
                block_width(psz).trailing_zeros() - 2,
                block_height(psz).trailing_zeros() - 2,
            )
        } else {
            lf_dims(tx_w_log2(tx) - 2, tx_h_log2(tx) - 2)
        };
        fill_rect(&mut self.fs.lf_tx[plane], st, at, step_x, step_y, dims);
        if let Some(log) = self.rec.as_deref_mut() {
            let sc = &mut self.scratch;
            let (nz_rows, nz_cols) = if coded && sc.logged {
                // Found when replayed.
                (0, 0)
            } else if coded {
                let tw = tx_w(tx).min(32);
                let (nr, nc) = (sc.nz_rows.clamp(1, 32), sc.nz_cols.clamp(1, tw));
                // The coefficients are in `coefs`: log the nonzero ones.
                for i in 0..nr {
                    for (j, v) in sc.coefs[i * tw..][..nc].iter_mut().enumerate() {
                        if *v != 0 {
                            log.coefs.push((*v << 10) | (i * tw + j) as i32);
                            *v = 0;
                        }
                    }
                }
                (nr, nc)
            } else {
                (0, 0)
            };
            log.ops.push(TxOp {
                plane: plane as u8,
                tx,
                tx_type: sc.tx_type,
                coded,
                nz_rows: nz_rows as u8,
                nz_cols: nz_cols as u8,
                x: x as u16,
                y: y as u16,
                base_x: base_x as u32,
                base_y: base_y as u32,
                coef: coef as u32,
                ncoef: (log.coefs.len() - coef) as u16,
                raw: coded && sc.logged,
                deq: sc.deq,
            });
        }
        // Also while recording: mode info prediction asks whether the 4x4s
        // above right are decoded yet (see `decoded`).
        if plane == 0 || self.rec.is_none() {
            self.tx_mark_decoded(plane, start_x, start_y, tx);
        }
        Ok(())
    }

    /// Prediction of transform_block() (intra and palette blocks).
    fn tx_predict(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        x: usize,
        y: usize,
        tx: u8,
    ) {
        let (sx, sy) = self.plane_sub(plane);
        let row = (start_y << sy) >> 2;
        let col = (start_x << sx) >> 2;
        let sb_mask = self.sb_size4 - 1;
        let sub_row = row & sb_mask;
        let sub_col = col & sb_mask;
        let step_x = tx_w(tx) >> 2;
        let step_y = tx_h(tx) >> 2;
        if !self.b.is_inter {
            let palette = if plane == 0 {
                self.b.palette_size_y
            } else {
                self.b.palette_size_uv
            };
            if palette != 0 {
                self.predict_palette(plane, start_x, start_y, x, y, tx);
            } else {
                let is_cfl = plane > 0 && self.b.uv_mode == UV_CFL_PRED;
                let mode = if plane == 0 {
                    self.b.y_mode
                } else if is_cfl {
                    DC_PRED
                } else {
                    self.b.uv_mode
                };
                let have_left = (if plane == 0 {
                    self.b.avail_l
                } else {
                    self.b.avail_l_chroma
                }) || x > 0;
                let have_above = (if plane == 0 {
                    self.b.avail_u
                } else {
                    self.b.avail_u_chroma
                }) || y > 0;
                let bdf = &self.block_decoded[plane];
                let have_above_right = bdf[sub_row >> sy][(sub_col >> sx) + step_x + 1];
                let have_below_left = bdf[(sub_row >> sy) + step_y + 1][sub_col >> sx];
                self.predict_intra(
                    plane,
                    start_x,
                    start_y,
                    have_left,
                    have_above,
                    have_above_right,
                    have_below_left,
                    mode,
                    tx_w_log2(tx),
                    tx_h_log2(tx),
                );
                if is_cfl {
                    self.predict_cfl(plane, start_x, start_y, tx);
                }
            }
            if plane == 0 {
                self.b.max_luma_w = start_x + step_x * 4;
                self.b.max_luma_h = start_y + step_y * 4;
            }
        }
    }

    /// Marks the transform block decoded (BlockDecoded).
    fn tx_mark_decoded(&mut self, plane: usize, start_x: usize, start_y: usize, tx: u8) {
        let (sx, sy) = self.plane_sub(plane);
        let sb_mask = self.sb_size4 - 1;
        let sub_row = ((start_y << sy) >> 2) & sb_mask;
        let sub_col = ((start_x << sx) >> 2) & sb_mask;
        let step_x = tx_w(tx) >> 2;
        let step_y = tx_h(tx) >> 2;
        let bdf = &mut self.block_decoded[plane];
        let x0 = ((sub_col >> sx) + 1).min(35);
        let x1 = (x0 + step_x).min(35);
        let y0 = ((sub_row >> sy) + 1).min(35);
        let y1 = (y0 + step_y).min(35);
        for r in &mut bdf[y0..y1] {
            fill_small(&mut r[x0..x1], true);
        }
    }

    /// Predicts and reconstructs the blocks of `log`, recorded by a tile
    /// decoder of the same frame whose mode info `self.fs` holds, from the
    /// superblock and block `from` on (both 0 to start).
    ///
    /// When the prediction of a block fails, `from` is left at that block:
    /// a caller that can mend the cause (rows of a reference that were not
    /// published yet, see [`Scratch::want_rows`]) calls again with it, and
    /// the block is predicted afresh. Nothing of it was reconstructed.
    pub fn replay(&mut self, log: &ReconLog, from: &mut (usize, usize)) -> R<()> {
        let mut bi = from.1;
        for (si, sb) in log.sbs.iter().enumerate().skip(from.0) {
            [
                self.mi_row_start,
                self.mi_row_end,
                self.mi_col_start,
                self.mi_col_end,
            ] = sb.tile;
            // A superblock taken up again part way keeps the flags of the
            // blocks it has done.
            if bi <= sb.first_block {
                self.clear_block_decoded_flags(sb.r, sb.c, self.num_planes);
            }
            let end = log
                .sbs
                .get(si + 1)
                .map_or(log.blocks.len(), |n| n.first_block);
            while bi < end {
                let rec = &log.blocks[bi];
                rec.b.apply(&mut self.b);
                let info = std::hint::black_box(mode_info_of(&self.b));
                self.write_mode_info(&info, rec.b.delta_lf);
                if let Some((at, cands)) = rec.extra {
                    let b = &mut self.b;
                    if b.palette_size_y != 0 || b.palette_size_uv != 0 {
                        let src = &log.extra[at as usize..];
                        for (k, c) in [&mut b.palette_y, &mut b.palette_u, &mut b.palette_v]
                            .into_iter()
                            .enumerate()
                        {
                            c.copy_from_slice(&src[8 * k..8 * k + 8]);
                        }
                        let ny = self.color_map_y.len();
                        let nuv = self.color_map_uv.len();
                        for (d, &v) in self.color_map_y.iter_mut().zip(&src[24..24 + ny]) {
                            *d = v as u8;
                        }
                        for (d, &v) in self.color_map_uv.iter_mut().zip(&src[24 + ny..][..nuv]) {
                            *d = v as u8;
                        }
                    }
                    if b.motion_mode == LOCALWARP {
                        let n = b.num_samples;
                        b.cand_list[..n].copy_from_slice(&log.cands[cands as usize..][..n]);
                    }
                }
                if let Err(e) = self.compute_prediction() {
                    *from = (si, bi);
                    return Err(e);
                }
                let op_end = log
                    .blocks
                    .get(bi + 1)
                    .map_or(log.ops.len(), |n| n.first_op as usize);
                for op in &log.ops[rec.first_op as usize..op_end] {
                    self.replay_tx(op, &log.coefs);
                }
                bi += 1;
            }
        }
        Ok(())
    }

    fn replay_tx(&mut self, op: &TxOp, coefs: &[i32]) {
        let plane = usize::from(op.plane);
        let (x, y) = (usize::from(op.x), usize::from(op.y));
        let start_x = op.base_x as usize + 4 * x;
        let start_y = op.base_y as usize + 4 * y;
        self.tx_predict(plane, start_x, start_y, x, y, op.tx);
        if op.coded {
            let (mut nr, mut nc) = (usize::from(op.nz_rows), usize::from(op.nz_cols));
            let dq = &mut self.scratch.coefs;
            let ops = &coefs[op.coef as usize..][..usize::from(op.ncoef)];
            if op.raw {
                let (dc, ac, shift) = op.deq;
                let lim = 1i64 << (7 + self.bd);
                let cbwl = tx_w(op.tx).min(32).trailing_zeros();
                let (mut rows, mut cols) = (0, 0);
                for &e in ops {
                    let pos = (e & 1023) as usize;
                    let q = e >> 10;
                    let qv = if pos == 0 { dc } else { ac };
                    dq[pos] = dequant_level(q.abs(), qv, q < 0, u32::from(shift), lim);
                    rows = rows.max(pos >> cbwl);
                    cols = cols.max(pos & ((1 << cbwl) - 1));
                }
                (nr, nc) = (rows + 1, cols + 1);
            } else {
                for &e in ops {
                    dq[(e & 1023) as usize] = e >> 10;
                }
            }
            self.scratch.tx_type = op.tx_type;
            self.scratch.nz_rows = nr;
            self.scratch.nz_cols = nc;
            self.reconstruct(plane, start_x, start_y, op.tx);
        }
        self.tx_mark_decoded(plane, start_x, start_y, op.tx);
    }

    pub fn tx_set_for(&self, tx: u8) -> usize {
        self.get_tx_set(tx)
    }

    pub fn tx_type_allowed(&self, set: usize, t: u8) -> bool {
        self.is_tx_type_in_set(set, t)
    }

    fn get_tx_set(&self, tx: u8) -> usize {
        let sqr = TX_SIZE_SQR[tx as usize];
        let sqr_up = TX_SIZE_SQR_UP[tx as usize];
        if sqr_up > TX_32X32 {
            return 0;
        }
        if self.b.is_inter {
            if self.hdr.reduced_tx_set || sqr_up == TX_32X32 {
                3
            } else if sqr == 2 {
                2
            } else {
                1
            }
        } else if sqr_up == TX_32X32 {
            0
        } else if self.hdr.reduced_tx_set || sqr == 2 {
            2
        } else {
            1
        }
    }

    fn is_tx_type_in_set(&self, set: usize, t: u8) -> bool {
        if self.b.is_inter {
            TX_TYPE_IN_SET_INTER[set][t as usize] != 0
        } else {
            TX_TYPE_IN_SET_INTRA[set][t as usize] != 0
        }
    }

    pub fn compute_tx_type(&self, plane: usize, tx: u8, bx: usize, by: usize) -> u8 {
        let sqr_up = TX_SIZE_SQR_UP[tx as usize];
        if self.b.lossless || sqr_up > TX_32X32 {
            return 0;
        }
        let set = self.get_tx_set(tx);
        let st = self.fs.mi_stride;
        if plane == 0 {
            return self.fs.tx_types[by * st + bx];
        }
        if self.b.is_inter {
            let x4 = self.b.mi_col.max(bx << u32::from(self.sub_x));
            let y4 = self.b.mi_row.max(by << u32::from(self.sub_y));
            let t = self.fs.tx_types[y4 * st + x4];
            if !self.is_tx_type_in_set(set, t) {
                return 0;
            }
            return t;
        }
        let t = MODE_TO_TXFM[self.b.uv_mode as usize];
        if !self.is_tx_type_in_set(set, t) {
            return 0;
        }
        t
    }

    fn read_tx_type(&mut self, x4: usize, y4: usize, tx: u8, want: u8) {
        let pos = |inv: &[u8]| inv.iter().position(|&v| v == want).unwrap_or(0);
        let set = self.get_tx_set(tx);
        let q = if self.hdr.segmentation.enabled {
            i32::from(self.hdr.qindex(self.b.segment_id as usize))
        } else {
            i32::from(self.hdr.quant.base_q_idx)
        };
        let mut t = 0u8;
        if set > 0 && q > 0 {
            let sqr = TX_SIZE_SQR[tx as usize] as usize;
            if self.b.is_inter {
                t = match set {
                    1 => {
                        TX_TYPE_INTER_INV_SET1[self.msac.symbol(
                            &mut self.cdf.inter_tx_type_set1[sqr],
                            pos(&TX_TYPE_INTER_INV_SET1),
                        )]
                    }
                    2 => {
                        TX_TYPE_INTER_INV_SET2[self.msac.symbol(
                            &mut self.cdf.inter_tx_type_set2,
                            pos(&TX_TYPE_INTER_INV_SET2),
                        )]
                    }
                    _ => {
                        TX_TYPE_INTER_INV_SET3[usize::from(self.msac.bool_adapt(
                            &mut self.cdf.inter_tx_type_set3[sqr],
                            pos(&TX_TYPE_INTER_INV_SET3) == 1,
                        ))]
                    }
                };
            } else {
                let dir = if self.b.use_filter_intra {
                    FILTER_INTRA_MODE_TO_INTRA_DIR[self.b.filter_intra_mode as usize] as usize
                } else {
                    self.b.y_mode as usize
                };
                t = if set == 1 {
                    TX_TYPE_INTRA_INV_SET1[self.msac.symbol(
                        &mut self.cdf.intra_tx_type_set1[sqr][dir],
                        pos(&TX_TYPE_INTRA_INV_SET1),
                    )]
                } else {
                    TX_TYPE_INTRA_INV_SET2[self.msac.symbol(
                        &mut self.cdf.intra_tx_type_set2[sqr][dir],
                        pos(&TX_TYPE_INTRA_INV_SET2),
                    )]
                };
            }
        }
        let st = self.fs.mi_stride;
        let (w4, h4) = (tx_w(tx) >> 2, tx_h(tx) >> 2);
        fill_rect(&mut self.fs.tx_types, st, y4 * st + x4, w4, h4, t);
    }

    /// coeffs() — reads one transform block's coefficients into the
    /// dequantized buffer and returns eob.
    fn coeffs(&mut self, plane: usize, start_x: usize, start_y: usize, tx: u8) -> R<usize> {
        let x4 = start_x >> 2;
        let y4 = start_y >> 2;
        let w4 = tx_w(tx) >> 2;
        let h4 = tx_h(tx) >> 2;
        let tx_sz_ctx =
            ((TX_SIZE_SQR[tx as usize] + TX_SIZE_SQR_UP[tx as usize] + 1) >> 1) as usize;
        let ptype = usize::from(plane > 0);
        let (sx, sy) = self.plane_sub(plane);
        let max_x4 = self.fs.mi_cols >> sx;
        let max_y4 = self.fs.mi_rows >> sy;
        let sb_mask = (self.sb_size4 - 1) >> sy;
        // all_zero context
        let bsize = plane_residual_size(self.b.bsize, sx != 0, sy != 0);
        let bw = block_width(bsize);
        let bh = block_height(bsize);
        let w = tx_w(tx);
        let h = tx_h(tx);
        let ly0 = y4 & sb_mask;
        // The context entries inside the frame.
        let (na, nl) = (
            w4.min(max_x4.saturating_sub(x4)),
            h4.min(max_y4.saturating_sub(y4)),
        );
        let ctx = if plane == 0 {
            // Only whether the largest level is 0 or above 3 matters.
            const ABOVE3: u128 = 0xFC * (u128::MAX / 255);
            let top = ctx_lanes(&self.above_level[plane], x4, na);
            let left = left_lanes(&self.left_level[plane], ly0, nl);
            let (tz, lz) = (top == 0, left == 0);
            let (tg, lg) = (top & ABOVE3 != 0, left & ABOVE3 != 0);
            if bw == w && bh == h {
                0
            } else if tz && lz {
                1
            } else if tz || lz {
                2 + usize::from(tg || lg)
            } else if !tg && !lg {
                4
            } else if !tg || !lg {
                5
            } else {
                6
            }
        } else {
            let above = ctx_lanes(&self.above_level[plane], x4, na)
                | ctx_lanes(&self.above_dc[plane], x4, na);
            let left = left_lanes(&self.left_level[plane], ly0, nl)
                | left_lanes(&self.left_dc[plane], ly0, nl);
            let mut c = usize::from(above != 0) + usize::from(left != 0) + 7;
            if bw * bh > w * h {
                c += 3;
            }
            c
        };
        // Encoder: the planned coefficients and their end of block.
        let mut want_eob = 0usize;
        let mut want_type = 0u8;
        if C::WRITE {
            if let Some(pt) = self.plan.tx(plane, start_x, start_y) {
                want_type = pt.tx_type;
                let want = &mut self.scratch.want;
                for (i, v) in want.iter_mut().enumerate() {
                    *v = pt.coefs.get(i).copied().unwrap_or(0);
                }
            } else {
                // Quantize the source against the prediction now in the frame.
                want_type = self.enc_quantize(plane, start_x, start_y, tx);
            }
            let scan = get_scan(tx, want_type);
            for (c, &p) in scan.iter().enumerate() {
                if self.scratch.want[p as usize] != 0 {
                    want_eob = c + 1;
                }
            }
        }
        let all_zero = self
            .msac
            .bool_adapt(&mut self.cdf.txb_skip[tx_sz_ctx][ctx], want_eob == 0);
        let mut eob = 0usize;
        let mut cul_level = 0u32;
        let mut dc_category = 0u8;
        if all_zero {
            if plane == 0 {
                let st = self.fs.mi_stride;
                fill_rect(&mut self.fs.tx_types, st, y4 * st + x4, w4, h4, 0);
            }
        } else {
            if plane == 0 {
                self.read_tx_type(x4, y4, tx, want_type);
            }
            let tx_type = self.compute_tx_type(plane, tx, x4, y4);
            self.scratch.tx_type = tx_type;
            #[cfg(feature = "stats")]
            if plane == 0 {
                super::stats::add(
                    &super::stats::TXTYPE,
                    tx_type as usize,
                    (tx_w(tx) * tx_h(tx) / 16) as u64,
                );
            }
            let scan = get_scan(tx, tx_type);
            let qinfo = self.dequant_info(plane, tx_type, tx);
            let tx_class = get_tx_class(tx_type);
            let eob_multisize = tx_w_log2(tx).min(5) + tx_h_log2(tx).min(5) - 4;
            let ectx = usize::from(tx_class != 0);
            let cdf = &mut *self.cdf;
            let scratch = &mut self.scratch;
            let above_dc = &self.above_dc[plane];
            let left_dc = &self.left_dc[plane];
            let bd = self.bd;
            let rec = self.rec.as_deref_mut();
            // The symbols of the block, with the decoder state in locals.
            let (e, cul, dcc) = self.msac.with_local(
                #[inline(always)]
                |msac| {
                    let we = want_eob.max(1);
                    let w_pt = if we <= 2 {
                        we
                    } else {
                        (31 - ((we - 1) as u32).leading_zeros()) as usize + 2
                    };
                    let w_extra = we.saturating_sub(if w_pt < 2 {
                        w_pt
                    } else {
                        (1 << (w_pt - 2)) + 1
                    });
                    let wp = w_pt - 1;
                    let eob_pt = 1 + match eob_multisize {
                        0 => msac.symbol(&mut cdf.eob_pt_16[ptype][ectx], wp),
                        1 => msac.symbol(&mut cdf.eob_pt_32[ptype][ectx], wp),
                        2 => msac.symbol(&mut cdf.eob_pt_64[ptype][ectx], wp),
                        3 => msac.symbol(&mut cdf.eob_pt_128[ptype][ectx], wp),
                        4 => msac.symbol(&mut cdf.eob_pt_256[ptype][ectx], wp),
                        5 => msac.symbol(&mut cdf.eob_pt_512[ptype], wp),
                        _ => msac.symbol(&mut cdf.eob_pt_1024[ptype], wp),
                    };
                    let mut eob = if eob_pt < 2 {
                        eob_pt
                    } else {
                        (1 << (eob_pt - 2)) + 1
                    };
                    let eob_shift = eob_pt as i32 - 3;
                    if eob_shift >= 0 {
                        if msac.bool_adapt(
                            &mut cdf.eob_extra[tx_sz_ctx][ptype][eob_pt - 3],
                            (w_extra >> eob_shift) & 1 != 0,
                        ) {
                            eob += 1 << eob_shift;
                        }
                        for i in 1..(eob_pt as i32 - 2).max(0) {
                            let sh = (eob_pt as i32 - 2).max(0) - 1 - i;
                            if msac.literal(1, ((w_extra >> sh) & 1) as u32) != 0 {
                                eob += 1 << sh;
                            }
                        }
                    }
                    // Level decoding on a padded levels buffer.
                    let adj = ADJUSTED_TX_SIZE[tx as usize];
                    let bwl = tx_w_log2(adj);
                    let width = 1usize << bwl;
                    let height = tx_h(adj);
                    let lstride = width + 4;
                    let levels = &mut *scratch.levels;
                    let nzc = &mut *scratch.nz_scan;
                    let coeff_base_cdf = &mut cdf.coeff_base[tx_sz_ctx][ptype];
                    let coeff_br_cdf = &mut cdf.coeff_br[tx_sz_ctx.min(3)][ptype];
                    // The eob coefficient.
                    let n = height << bwl;
                    let last = eob - 1;
                    let ctx = if last == 0 {
                        0
                    } else if last <= n / 8 {
                        1
                    } else if last <= n / 4 {
                        2
                    } else {
                        3
                    };
                    let want = &scratch.want;
                    let wfirst = if C::WRITE {
                        (want[scan[last] as usize].unsigned_abs().clamp(1, 3) - 1) as usize
                    } else {
                        0
                    };
                    let first = 1 + msac
                        .symbol(&mut cdf.coeff_base_eob[tx_sz_ctx][ptype][ctx], wfirst)
                        as u32;
                    let tab = level_table(tx as usize, tx_class);
                    let args = LevelArgs {
                        scan,
                        tab,
                        eob,
                        lstride,
                        first,
                        want,
                    };
                    macro_rules! levels_ls {
                        ($class:literal, $ls:literal) => {
                            match msac.decoder(|m| {
                                read_levels_dec::<$class, $ls>(
                                    m,
                                    coeff_base_cdf,
                                    coeff_br_cdf,
                                    levels,
                                    nzc,
                                    &args,
                                )
                            }) {
                                Some(n) => n,
                                None => read_levels::<$class, $ls, C>(
                                    msac,
                                    coeff_base_cdf,
                                    coeff_br_cdf,
                                    levels,
                                    nzc,
                                    &args,
                                ),
                            }
                        };
                    }
                    macro_rules! levels {
                        ($class:literal) => {
                            match bwl {
                                2 => levels_ls!($class, 8),
                                3 => levels_ls!($class, 12),
                                4 => levels_ls!($class, 20),
                                _ => levels_ls!($class, 36),
                            }
                        };
                    }
                    let n_nz = match tx_class {
                        0 => levels!(0),
                        1 => levels!(1),
                        _ => levels!(2),
                    };
                    // Signs and Golomb remainder.
                    let lim = 1i64 << (7 + bd);
                    let dshift = qinfo.denom.trailing_zeros();
                    let cbwl = bwl.min(5);
                    // dc_sign ctx
                    let dc_ctx = || {
                        let dc_sign = dc_weight(ctx_lanes(above_dc, x4, na))
                            + dc_weight(left_lanes(left_dc, ly0, nl));
                        match dc_sign.cmp(&0) {
                            std::cmp::Ordering::Less => 1,
                            std::cmp::Ordering::Greater => 2,
                            std::cmp::Ordering::Equal => 0,
                        }
                    };
                    let nz = &nzc[..n_nz];
                    let fast = if qinfo.qm.is_none() {
                        let deq = Dequant {
                            dc: qinfo.dc,
                            ac: qinfo.ac,
                            shift: dshift,
                            lim,
                            cbwl,
                        };
                        let (dq, dc_cdf) = (&mut scratch.coefs, &mut cdf.dc_sign[ptype]);
                        let logged = &mut scratch.logged;
                        scratch.deq = (qinfo.dc, qinfo.ac, dshift as u8);
                        msac.decoder(|m| {
                            if let Some(log) = rec {
                                // Recording: the coefficients go straight
                                // to the log (see `TxOp`).
                                let out = &mut log.coefs;
                                out.reserve(nz.len());
                                *logged = true;
                                Some(signs_dec::<true>(
                                    m,
                                    nz,
                                    levels,
                                    dc_cdf,
                                    dc_ctx,
                                    &deq,
                                    #[inline(always)]
                                    |_, p, v| out.push((v << 10) | p as i32),
                                ))
                            } else {
                                let dq = dq.first_chunk_mut::<1024>()?;
                                Some(signs_dec::<false>(
                                    m,
                                    nz,
                                    levels,
                                    dc_cdf,
                                    dc_ctx,
                                    &deq,
                                    #[inline(always)]
                                    |_, p, v| dq[p & 1023] = v,
                                ))
                            }
                        })
                        .flatten()
                    } else {
                        None
                    };
                    let (cul_level, dc_category, nz_rows, nz_cols) = match fast {
                        Some(r) => r,
                        None => {
                            let dq = &mut scratch.coefs;
                            let mut nz_rows = 0usize;
                            let mut nz_cols = 0usize;
                            let mut cul_level = 0u32;
                            let mut dc_category = 0u8;
                            // The nonzero coefficients, in scan order.
                            for &e in nz.iter().rev() {
                                let li = (e & LEVELS_MASK) as usize;
                                let mut q = i32::from(levels[li] >> 8);
                                levels[li] = 0;
                                let pos = (e >> 22) as usize;
                                let wpos = if C::WRITE { want[pos] } else { 0 };
                                let sign = if pos == 0 {
                                    msac.bool_adapt(&mut cdf.dc_sign[ptype][dc_ctx()], wpos < 0)
                                } else {
                                    msac.literal(1, u32::from(wpos < 0)) != 0
                                };
                                if q > 14 {
                                    let g = msac.golomb(wpos.unsigned_abs().saturating_sub(15));
                                    q = (g as i32).wrapping_add(15);
                                }
                                if pos == 0 {
                                    dc_category = if sign { 1 } else { 2 };
                                }
                                let q = q & 0xFFFFF;
                                cul_level += q as u32;
                                // Dequantize (7.12.3 step 1).
                                let i = pos >> cbwl;
                                let j = pos - (i << cbwl);
                                let qv = if pos == 0 { qinfo.dc } else { qinfo.ac };
                                let q2 = match qinfo.qm {
                                    Some(m) => crate::dsp::round2(qv * i32::from(m[pos]), 5),
                                    None => qv,
                                };
                                let dqv = ((i64::from(q) * i64::from(q2)) & 0xFFFFFF) >> dshift;
                                let v = if sign { -dqv } else { dqv };
                                dq[pos] = v.clamp(-lim, lim - 1) as i32;
                                nz_rows = nz_rows.max(i + 1);
                                nz_cols = nz_cols.max(j + 1);
                            }
                            (cul_level, dc_category, nz_rows, nz_cols)
                        }
                    };
                    // Levels read as zero were never written.
                    scratch.nz_rows = nz_rows.max(1);
                    scratch.nz_cols = nz_cols.max(1);
                    (eob, cul_level.min(63), dc_category)
                },
            );
            eob = e;
            cul_level = cul;
            dc_category = dcc;
        }
        ctx_set(&mut self.above_level[plane], x4, w4, cul_level as u8);
        ctx_set(&mut self.above_dc[plane], x4, w4, dc_category);
        if ly0 + h4 <= 32 {
            ctx_set(&mut self.left_level[plane], ly0, h4, cul_level as u8);
            ctx_set(&mut self.left_dc[plane], ly0, h4, dc_category);
        } else {
            for i in 0..h4 {
                let ly = (ly0 + i) & 31;
                self.left_level[plane][ly] = cul_level as u8;
                self.left_dc[plane][ly] = dc_category;
            }
        }
        Ok(eob)
    }
}

struct LevelArgs<'s> {
    scan: &'s [u16],
    /// [`level_table`] of the transform size and class.
    tab: &'s [u32],
    /// Planned signed levels (encoder), indexed by position.
    want: &'s [i32],
    eob: usize,
    lstride: usize,
    first: u32,
}

/// Per transform size and class, one entry per scan index `c`: the
/// position's index in the padded levels buffer (bits 0..11, see
/// [`LEVELS_MASK`]), its coeff_base context offset (bits 11..17), its
/// coeff_br context offset (bits 17..22) and the position itself (bits
/// 22..32). Entry 0 (the DC) is 0.
pub(crate) fn level_table(tx: usize, class: usize) -> &'static [u32] {
    use std::sync::OnceLock;
    static TABLES: OnceLock<Vec<Vec<u32>>> = OnceLock::new();
    let t = TABLES.get_or_init(|| {
        let mut out = Vec::with_capacity(19 * 3);
        for tx in 0..19 {
            for class in 0..3 {
                let adj = ADJUSTED_TX_SIZE[tx];
                let bwl = tx_w_log2(adj);
                let mask = (1usize << bwl) - 1;
                // A tx_type of the class picks the scan.
                let tx_type = [0, H_DCT, V_DCT][class];
                let scan = get_scan(tx as u8, tx_type);
                let v = scan
                    .iter()
                    .map(|&p| {
                        let pos = p as usize;
                        let row = pos >> bwl;
                        let col = pos & mask;
                        let li = pos + (row << 2);
                        let (base, near) = match class {
                            0 => (
                                COEFF_BASE_CTX_OFFSET[tx][row.min(4)][col.min(4)],
                                row < 2 && col < 2,
                            ),
                            1 => (COEFF_BASE_POS_CTX_OFFSET[col.min(2)], col == 0),
                            _ => (COEFF_BASE_POS_CTX_OFFSET[row.min(2)], row == 0),
                        };
                        let (base, br) = if pos == 0 {
                            (0, 0)
                        } else {
                            (u32::from(base), if near { 7 } else { 14 })
                        };
                        li as u32 | base << 11 | br << 17 | (pos as u32) << 22
                    })
                    .collect();
                out.push(v);
            }
        }
        out
    });
    &t[tx * 3 + class]
}

/// The value kept in the levels buffer for a coefficient level `l` (at most
/// 15): `min(l, 3)` in the low byte, `l` in the high byte, so one sum of
/// neighbours gives both context magnitudes.
#[inline(always)]
fn level_entry(l: u32) -> u16 {
    (l.min(3) | (l << 8)) as u16
}

/// Reads coeff_br for a coefficient whose base level is 3; returns the level.
#[inline(always)]
fn read_br<C: Coder>(msac: &mut C, cdf: &mut u64, wl: u32) -> u32 {
    let mut level = 3;
    let mut rem = wl.saturating_sub(3);
    for _ in 0..4 {
        let wb = rem.min(3);
        rem -= wb;
        let br = symbol_packed(msac, cdf, wb as usize);
        level += br;
        if br < 3 {
            break;
        }
    }
    level
}

/// [`Coder::symbol_packed`].
#[inline(always)]
fn symbol_packed<C: Coder>(msac: &mut C, cdf: &mut u64, want: usize) -> u32 {
    msac.symbol_packed(cdf, want) as u32
}

/// Length of the padded levels buffer: any index below `LEVELS_MASK + 1`
/// plus the farthest neighbour offset (4 rows of at most 63) stays inside.
pub const LEVELS_LEN: usize = 2048 + 4 * 64 + 4;
/// Mask of the levels buffer index in a [`level_table`] entry.
const LEVELS_MASK: u32 = 0x7FF;

/// Reads coeff_base / coeff_br for scan positions eob-1 down to 0 (the
/// eob position's base level is `args.first`). `levels` is a zeroed padded
/// buffer with stride `LS` (= width + 4) of [`level_entry`] values.
/// Returns the number of scan indices written to `nz`: the eob position,
/// then every nonzero one below it in decreasing order.
#[inline(always)]
fn read_levels<const CLASS: usize, const LS: usize, C: Coder>(
    msac: &mut C,
    base_cdf: &mut [u64; 42],
    br_cdf: &mut [u64; 21],
    levels: &mut [u16; LEVELS_LEN],
    nz: &mut [u32; 1024],
    a: &LevelArgs<'_>,
) -> usize {
    debug_assert_eq!(a.lstride, LS);
    let ls = LS;
    let offs: [usize; 5] = std::array::from_fn(|i| {
        SIG_REF_DIFF_OFFSET[CLASS][i][0] as usize * ls + SIG_REF_DIFF_OFFSET[CLASS][i][1] as usize
    });
    let want = |c: usize| {
        if C::WRITE {
            a.want[a.scan[c] as usize].unsigned_abs()
        } else {
            0
        }
    };
    // Sums of the neighbour entries: (3 magnitude neighbours, all 5).
    let sums = |levels: &[u16; LEVELS_LEN], li: usize| {
        let s3 = u32::from(levels[li + offs[0]])
            + u32::from(levels[li + offs[1]])
            + u32::from(levels[li + offs[2]]);
        (
            s3,
            s3 + u32::from(levels[li + offs[3]]) + u32::from(levels[li + offs[4]]),
        )
    };
    let last = a.eob - 1;
    // The eob coefficient (nonzero).
    nz[0] = a.tab[last];
    let mut n = 1;
    {
        let e = a.tab[last];
        let li = (e & LEVELS_MASK) as usize;
        let mut level = a.first;
        if level > 2 {
            let (s3, _) = sums(levels, li);
            let mag = (((s3 >> 8) + 1) >> 1).min(6);
            level = read_br(
                msac,
                &mut br_cdf[(mag + ((e >> 17) & 0x1F)) as usize],
                want(last),
            );
        }
        levels[li] = level_entry(level);
    }
    if last == 0 {
        return n;
    }
    for (c, &e) in a.tab[..last].iter().enumerate().skip(1).rev() {
        let li = (e & LEVELS_MASK) as usize;
        let (s3, s5) = sums(levels, li);
        let m = (((s5 & 0xFF) + 1) >> 1).min(4);
        let ctx = (m + ((e >> 11) & 0x3F)) as usize;
        let wl = want(c);
        let mut level = symbol_packed(msac, &mut base_cdf[ctx], wl.min(3) as usize);
        if level > 2 {
            let mag = (((s3 >> 8) + 1) >> 1).min(6);
            level = read_br(msac, &mut br_cdf[(mag + ((e >> 17) & 0x1F)) as usize], wl);
        }
        levels[li] = level_entry(level);
        nz[n & 1023] = e;
        n += usize::from(level != 0);
    }
    // The DC coefficient.
    let (s3, s5) = sums(levels, 0);
    let ctx = if CLASS == 0 {
        0
    } else {
        ((((s5 & 0xFF) + 1) >> 1).min(4) + u32::from(COEFF_BASE_POS_CTX_OFFSET[0])) as usize
    };
    let wl = want(0);
    let mut level = symbol_packed(msac, &mut base_cdf[ctx], wl.min(3) as usize);
    if level > 2 {
        let mag = (((s3 >> 8) + 1) >> 1).min(6);
        level = read_br(msac, &mut br_cdf[mag as usize], wl);
    }
    levels[0] = level_entry(level);
    nz[n & 1023] = 0;
    n + usize::from(level != 0)
}

/// `min((x + 1) >> 1, 4)` for the coeff_base context.
const BASE_MAG: [u8; 16] = [0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4];

/// The neighbour offsets in a levels buffer of stride `LS`.
#[inline(always)]
fn level_offsets<const CLASS: usize, const LS: usize>() -> [usize; 5] {
    std::array::from_fn(|i| {
        SIG_REF_DIFF_OFFSET[CLASS][i][0] as usize * LS + SIG_REF_DIFF_OFFSET[CLASS][i][1] as usize
    })
}

/// [`read_levels`] when decoding: the same syntax, on a [`Hot`] decoder
/// state, with the coeff_br symbols out of line. Inside the loop live only
/// the decoder state, the scan index, the buffers and the nonzero count.
#[inline(never)]
fn read_levels_dec<const CLASS: usize, const LS: usize>(
    m: &mut Msac<'_>,
    base_cdf: &mut [u64; 42],
    br_cdf: &mut [u64; 21],
    levels: &mut [u16; LEVELS_LEN],
    nz: &mut [u32; 1024],
    a: &LevelArgs<'_>,
) -> usize {
    if !m.allow_update {
        return read_levels::<CLASS, LS, Msac<'_>>(m, base_cdf, br_cdf, levels, nz, a);
    }
    let offs = level_offsets::<CLASS, LS>();
    let mut h = Hot::new(m);
    let last = a.eob - 1;
    // The eob coefficient (nonzero).
    nz[0] = a.tab[last];
    {
        let e = a.tab[last];
        let li = (e & LEVELS_MASK) as usize;
        let mut entry = level_entry(a.first);
        if a.first > 2 {
            (h, entry) = br_dec::<CLASS, LS>(h, br_cdf, levels, li, (e >> 17) & 0x1F);
        }
        levels[li] = entry;
    }
    if last == 0 {
        h.done();
        return 1;
    }
    let mut n = 1;
    let tab = &a.tab[..last];
    for &e in tab[1..].iter().rev() {
        let li = (e & LEVELS_MASK) as usize;
        let s5 = u32::from(levels[li + offs[0]])
            + u32::from(levels[li + offs[1]])
            + u32::from(levels[li + offs[2]])
            + u32::from(levels[li + offs[3]])
            + u32::from(levels[li + offs[4]]);
        let ctx = ((e >> 11) & 0x3F) as usize + usize::from(BASE_MAG[(s5 & 0xF) as usize]);
        let level = h.sym4(&mut base_cdf[ctx]);
        let entry = if level > 2 {
            let entry;
            (h, entry) = br_dec::<CLASS, LS>(h, br_cdf, levels, li, (e >> 17) & 0x1F);
            entry
        } else {
            (level * 0x101) as u16
        };
        levels[li] = entry;
        nz[n & 1023] = e;
        n += usize::from(level != 0);
    }
    // The DC coefficient.
    let ctx = if CLASS == 0 {
        0
    } else {
        let s5: u32 = offs.iter().map(|&o| u32::from(levels[o])).sum();
        usize::from(BASE_MAG[(s5 & 0xF) as usize] + COEFF_BASE_POS_CTX_OFFSET[0])
    };
    let level = h.sym4(&mut base_cdf[ctx]);
    let mut entry = (level * 0x101) as u16;
    if level > 2 {
        (h, entry) = br_dec::<CLASS, LS>(h, br_cdf, levels, 0, 0);
    }
    levels[0] = entry;
    nz[n & 1023] = 0;
    h.done();
    n + usize::from(level != 0)
}

/// Dequantizes level `q` (7.12.3 step 1) with quantizer `qv`: `shift` is
/// log2 of the denominator, the result is clamped to `-lim..lim`.
#[inline(always)]
fn dequant_level(q: i32, qv: i32, sign: bool, shift: u32, lim: i64) -> i32 {
    let dqv = ((i64::from(q) * i64::from(qv)) & 0xFF_FFFF) >> shift;
    let v = if sign { -dqv } else { dqv };
    v.max(-lim).min(lim - 1) as i32
}

/// Dequantization of one transform block without a quantizer matrix.
struct Dequant {
    dc: i32,
    ac: i32,
    /// log2 of the denominator.
    shift: u32,
    /// The coefficients are clamped to `-lim..lim`.
    lim: i64,
    /// log2 of the coefficient row stride.
    cbwl: u32,
}

/// The signs and Golomb remainders of the nonzero levels `nz` (from
/// [`read_levels_dec`]) when decoding, without a quantizer matrix: hands
/// each dequantized coefficient to `put(k, pos, value)`, `k` counting them
/// in scan order, and clears their levels. Returns the cumulative level,
/// the DC category and the rows and columns holding nonzero coefficients.
///
/// With `RAW`, `value` is the signed level instead (see [`dequant_level`])
/// and the rows and columns are not tracked (both 0).
#[inline(never)]
fn signs_dec<const RAW: bool>(
    m: &mut Msac<'_>,
    nz: &[u32],
    levels: &mut [u16; LEVELS_LEN],
    dc_cdf: &mut [[u16; 2]; 3],
    dc_ctx: impl FnOnce() -> usize,
    d: &Dequant,
    mut put: impl FnMut(usize, usize, i32),
) -> (u32, u8, usize, usize) {
    let dequant = |q: i32, qv: i32, sign: bool| {
        if RAW {
            if sign { -q } else { q }
        } else {
            dequant_level(q, qv, sign, d.shift, d.lim)
        }
    };
    let mut cul = 0u32;
    let mut dc_category = 0u8;
    let (mut rows, mut cols) = (0usize, 0usize);
    let mut rest = nz;
    let mut k = 0;
    // In scan order the DC comes first; its entry is 0.
    if let Some((&0, more)) = nz.split_last() {
        rest = more;
        let sign = m.bool_adapt(&mut dc_cdf[dc_ctx()]);
        let mut q = i32::from(levels[0] >> 8);
        levels[0] = 0;
        if q > 14 {
            q = (m.golomb() as i32).wrapping_add(15);
        }
        let q = q & 0xF_FFFF;
        cul += q as u32;
        dc_category = if sign { 1 } else { 2 };
        put(0, 0, dequant(q, d.dc, sign));
        if !RAW {
            (rows, cols) = (1, 1);
        }
        k = 1;
    }
    let cmask = (1usize << d.cbwl) - 1;
    let mut h = Hot::new(m);
    for &e in rest.iter().rev() {
        let li = (e & LEVELS_MASK) as usize;
        let pos = (e >> 22) as usize;
        let mut q = i32::from(levels[li] >> 8);
        levels[li] = 0;
        let sign = h.bool_equi();
        if q > 14 {
            q = (h.golomb() as i32).wrapping_add(15);
        }
        let q = q & 0xF_FFFF;
        cul += q as u32;
        put(k, pos, dequant(q, d.ac, sign));
        k += 1;
        if !RAW {
            rows = rows.max((pos >> d.cbwl) + 1);
            cols = cols.max((pos & cmask) + 1);
        }
    }
    h.done();
    (cul, dc_category, rows, cols)
}

/// The coeff_br symbols of the coefficient at `li` of the levels buffer
/// (`br_off`: its context offset); returns its [`level_entry`]. Out of line
/// and given the state by value, so the loop calling it keeps its registers.
#[cold]
#[inline(never)]
fn br_dec<'m, 'a, const CLASS: usize, const LS: usize>(
    mut h: Hot<'m, 'a>,
    br_cdf: &mut [u64; 21],
    levels: &[u16; LEVELS_LEN],
    li: usize,
    br_off: u32,
) -> (Hot<'m, 'a>, u16) {
    let offs = level_offsets::<CLASS, LS>();
    let s3 = u32::from(levels[li + offs[0]])
        + u32::from(levels[li + offs[1]])
        + u32::from(levels[li + offs[2]]);
    let mag = (((s3 >> 8) + 1) >> 1).min(6);
    let cdf = &mut br_cdf[(mag + br_off) as usize];
    let mut level = 3;
    for _ in 0..4 {
        let br = h.sym4(cdf);
        level += br;
        if br < 3 {
            break;
        }
    }
    (h, level_entry(level))
}

fn ceil_log2(x: u32) -> u32 {
    if x < 2 {
        return 0;
    }
    let mut i = 1;
    let mut p = 2;
    while p < x {
        i += 1;
        p <<= 1;
    }
    i
}

/// Quantizer values for one transform block.
pub struct DequantInfo {
    pub dc: i32,
    pub ac: i32,
    pub denom: i32,
    pub qm: Option<&'static [u8]>,
}

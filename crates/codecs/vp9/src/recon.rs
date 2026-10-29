//! Profile 0 reconstruction.
//!
//! Partition walk, mode decode, and coefficient reconstruction follow
//! libvpx `decode_partition` and `decode_block`. Inter prediction scales a
//! reference when its crop differs from the frame. A loop filter level of
//! zero skips the filter.

#![allow(dead_code)]

mod mc;
mod mv;

pub(crate) use mc::{Job, RefPlane, render};
pub use mc::{Motion, RefSlots, Reference};

/// Right and bottom samples past the aligned plane. libvpx allocates
/// `VP9_DEC_BORDER_IN_PIXELS` (32) so a transform that crosses `y_width` or
/// `uv_width` is stored in that border. A stride equal to the aligned width
/// wraps those samples onto the next row.
const FRAME_BORDER: usize = 32;

/// A reconstructed picture plus the counts and motion the next frame adapts from.
pub struct Recon {
    pub picture: Picture,
    pub counts: Counts,
    pub motion: Motion,
    pub segmap: Vec<u8>,
}

use cutlass_decode::CodecError;

use crate::bits::BoolDecoder;
use crate::coef::decode_coefs;
use crate::geom::{
    self, BLOCK_4X4, BLOCK_4X8, BLOCK_8X4, BLOCK_8X8, BLOCK_64X64, BLOCK_INVALID, DCT_DCT,
    INTRA_FRAME, PARTITION_HORZ, PARTITION_NONE, PARTITION_SPLIT, PARTITION_VERT, TX_4X4, TX_8X8,
    TX_16X16, TX_32X32, TX_MODE_SELECT, align8, intra_tx_type, max_tx, mi_count, mi_width_log2,
    partition_context_byte, subsize, tx_mode_biggest, uv_tx_420,
};
use crate::header::{Header, Parsed};
use crate::intra;
use crate::lpf;
use crate::prob::{self, Counts, FrameContext};
use crate::tables::{
    KF_UV_MODE_PROB, KF_Y_MODE_PROB, NEIGHBORS_COL_4X4, NEIGHBORS_COL_8X8, NEIGHBORS_COL_16X16,
    NEIGHBORS_DEFAULT_4X4, NEIGHBORS_DEFAULT_8X8, NEIGHBORS_DEFAULT_16X16, NEIGHBORS_DEFAULT_32X32,
    NEIGHBORS_ROW_4X4, NEIGHBORS_ROW_8X8, NEIGHBORS_ROW_16X16, SCAN_COL_4X4, SCAN_COL_8X8,
    SCAN_COL_16X16, SCAN_DEFAULT_4X4, SCAN_DEFAULT_8X8, SCAN_DEFAULT_16X16, SCAN_DEFAULT_32X32,
    SCAN_ROW_4X4, SCAN_ROW_8X8, SCAN_ROW_16X16,
};
use crate::txfm::{iht_add, iwht4x4_add};

pub struct Picture {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy)]
struct Mi {
    bsize: u8,
    mode: u8,
    uv: u8,
    bmi: [u8; 4],
    skip: bool,
    tx: u8,
    seg: u8,
    interp: u8,
    refs: [u8; 2],
    mv: [mc::Mv; 2],
    sub_mv: [[mc::Mv; 2]; 4],
    seg_predicted: bool,
}

#[derive(Clone, Copy)]
struct Split {
    mi_row: usize,
    mi_col: usize,
    partition: u8,
    sub: u8,
    n4: u32,
    n8: u32,
    hbs: usize,
    has_rows: bool,
    has_cols: bool,
}

#[derive(Clone, Copy)]
struct BlockPos {
    mi_row: usize,
    mi_col: usize,
    bw: usize,
    bh: usize,
}

struct BlockAt {
    mi_row: usize,
    mi_col: usize,
    bsize: u8,
    x_mis: usize,
    y_mis: usize,
    bw: usize,
    bh: usize,
}

#[derive(Clone, Copy)]
struct TxPos {
    plane: usize,
    mi_row: usize,
    mi_col: usize,
    tx: u8,
    row: usize,
    col: usize,
    n4_w: usize,
    ss: u32,
    shift_w: usize,
    shift_h: usize,
}

#[derive(Clone, Copy)]
struct Residual {
    mode: u8,
    x: usize,
    y: usize,
    ctx: usize,
}

struct Inv<'a> {
    coef: &'a [i32],
    stride: usize,
    n: usize,
    kind: u8,
    lossless: bool,
    x: usize,
    y: usize,
}

impl Default for Mi {
    fn default() -> Self {
        Self {
            bsize: BLOCK_8X8,
            mode: geom::DC,
            uv: geom::DC,
            bmi: [geom::DC; 4],
            skip: false,
            tx: TX_4X4,
            seg: 0,
            interp: 3,
            refs: [INTRA_FRAME, INTRA_FRAME],
            mv: [mc::Mv::default(); 2],
            sub_mv: [[mc::Mv::default(); 2]; 4],
            seg_predicted: false,
        }
    }
}

struct Scratch<'a> {
    header: &'a Header,
    fc: &'a FrameContext,
    counts: Counts,
    collect: bool,
    part: [[u8; 3]; 16],
    grid: Vec<Mi>,
    segmap: Vec<u8>,
    mi_cols: usize,
    mi_rows: usize,
    tile_col: usize,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    y_stride: usize,
    uv_stride: usize,
    /// Aligned plane size, libvpx `y_width` / `y_height`. The stride is wider
    /// by `FRAME_BORDER` so a transform that extends past this edge stays on
    /// its row.
    plane_w: usize,
    plane_h: usize,
    uv_plane_w: usize,
    uv_plane_h: usize,
    crop_w: usize,
    crop_h: usize,
    uv_w: usize,
    uv_h: usize,
    above_part: Vec<u8>,
    left_part: [u8; 8],
    above_y: Vec<u8>,
    above_u: Vec<u8>,
    above_v: Vec<u8>,
    left_y: [u8; 16],
    left_u: [u8; 16],
    left_v: [u8; 16],
    coef: Vec<i32>,
    levels: lpf::Levels,
    masks: Vec<lpf::Mask>,
    sb_cols: usize,
    slots: &'a RefSlots<'a>,
    tile_col_end: usize,
    prev: Option<&'a Motion>,
    last_seg: &'a [u8],
}

pub fn reconstruct<'a>(
    parsed: &'a Parsed<'_>,
    slots: &'a RefSlots<'a>,
    prev: Option<&'a Motion>,
    last_seg: &'a [u8],
) -> Result<Recon, CodecError> {
    if parsed.header.show_existing {
        return Err(CodecError::Unsupported {
            what: "vp9 show_existing",
        });
    }
    if !parsed.header.is_intra() && parsed.header.use_prev_mvs && prev.is_none() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "previous motion",
        });
    }
    let prev = if parsed.header.use_prev_mvs {
        prev
    } else {
        None
    };
    let mut frame = Scratch::new(&parsed.header, &parsed.fc, slots, prev, last_seg)?;
    frame.decode_tiles(parsed.tiles)?;
    frame.loop_filter()?;
    let picture = frame.picture()?;
    let motion = frame.motion();
    Ok(Recon {
        picture,
        counts: frame.counts,
        motion,
        segmap: frame.segmap,
    })
}

impl<'a> Scratch<'a> {
    fn new(
        header: &'a Header,
        fc: &'a FrameContext,
        slots: &'a RefSlots<'a>,
        prev: Option<&'a Motion>,
        last_seg: &'a [u8],
    ) -> Result<Self, CodecError> {
        let width = header.width;
        let height = header.height;
        if width == 0 || height == 0 || width > geom::MAX_EDGE || height > geom::MAX_EDGE {
            return Err(CodecError::Limit {
                what: "vp9 frame edge",
            });
        }
        let mi_cols = mi_count(width);
        let mi_rows = mi_count(height);
        let aligned = mi_cols.div_ceil(8) * 8;
        let coded_w = usize::try_from(align8(width)).unwrap_or(0);
        let coded_h = usize::try_from(align8(height)).unwrap_or(0);
        let uv_w = coded_w / 2;
        let uv_h = coded_h / 2;
        let y_stride = coded_w.saturating_add(FRAME_BORDER);
        let uv_stride = uv_w.saturating_add(FRAME_BORDER);
        let y_rows = coded_h.saturating_add(FRAME_BORDER);
        let uv_rows = uv_h.saturating_add(FRAME_BORDER);
        let cells = mi_cols.saturating_mul(mi_rows);
        Ok(Self {
            header,
            fc,
            counts: Counts::new(),
            collect: !header.error_resilient && !header.frame_parallel,
            part: if header.is_intra() {
                prob::kf_partition()
            } else {
                fc.partition
            },
            grid: vec![Mi::default(); cells],
            segmap: vec![0u8; cells],
            mi_cols,
            mi_rows,
            tile_col: 0,
            y: vec![0u8; y_stride.saturating_mul(y_rows)],
            u: vec![0u8; uv_stride.saturating_mul(uv_rows)],
            v: vec![0u8; uv_stride.saturating_mul(uv_rows)],
            y_stride,
            uv_stride,
            plane_w: coded_w,
            plane_h: coded_h,
            uv_plane_w: uv_w,
            uv_plane_h: uv_h,
            crop_w: usize::try_from(width).unwrap_or(0),
            crop_h: usize::try_from(height).unwrap_or(0),
            uv_w: usize::try_from(width.div_ceil(2)).unwrap_or(0),
            uv_h: usize::try_from(height.div_ceil(2)).unwrap_or(0),
            above_part: vec![0u8; aligned],
            left_part: [0; 8],
            above_y: vec![0u8; aligned * 2],
            above_u: vec![0u8; aligned],
            above_v: vec![0u8; aligned],
            left_y: [0; 16],
            left_u: [0; 16],
            left_v: [0; 16],
            coef: vec![0i32; 32 * 32],
            levels: lpf::levels(header),
            masks: vec![
                lpf::Mask::default();
                if header.filter_level > 0 {
                    mi_cols.div_ceil(8) * mi_rows.div_ceil(8)
                } else {
                    0
                }
            ],
            sb_cols: mi_cols.div_ceil(8),
            slots,
            tile_col_end: mi_cols,
            prev,
            last_seg,
        })
    }

    fn motion(&self) -> Motion {
        Motion {
            cols: self.mi_cols,
            cells: self
                .grid
                .iter()
                .map(|mi| mc::MotionCell {
                    mv: mi.mv,
                    refs: mi.refs,
                })
                .collect(),
        }
    }

    fn decode_tiles(&mut self, mut data: &[u8]) -> Result<(), CodecError> {
        let cols = 1usize << self.header.log2_tile_cols;
        let rows = 1usize << self.header.log2_tile_rows;
        for tr in 0..rows {
            let row0 = tile_span(tr, self.mi_rows, self.header.log2_tile_rows);
            let row1 = tile_span(tr + 1, self.mi_rows, self.header.log2_tile_rows);
            for tc in 0..cols {
                let last = tr + 1 == rows && tc + 1 == cols;
                let tile = take_tile(&mut data, last)?;
                let col0 = tile_span(tc, self.mi_cols, self.header.log2_tile_cols);
                let col1 = tile_span(tc + 1, self.mi_cols, self.header.log2_tile_cols);
                self.decode_tile(tile, row0, row1, col0, col1)?;
            }
        }
        Ok(())
    }

    fn decode_tile(
        &mut self,
        data: &[u8],
        row0: usize,
        row1: usize,
        col0: usize,
        col1: usize,
    ) -> Result<(), CodecError> {
        let mut reader = BoolDecoder::new(data)?;
        self.tile_col = col0;
        self.tile_col_end = col1;
        for mi_row in (row0..row1).step_by(8) {
            self.left_part = [0; 8];
            self.left_y = [0; 16];
            self.left_u = [0; 16];
            self.left_v = [0; 16];
            for mi_col in (col0..col1).step_by(8) {
                self.decode_partition(&mut reader, mi_row, mi_col, BLOCK_64X64, 4)?;
            }
        }
        reader.finish()
    }

    fn decode_partition(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi_row: usize,
        mi_col: usize,
        bsize: u8,
        n4: u32,
    ) -> Result<(), CodecError> {
        if mi_row >= self.mi_rows || mi_col >= self.mi_cols {
            return Ok(());
        }
        let n8 = n4 - 1;
        let hbs = (1usize << n8) >> 1;
        let has_rows = mi_row + hbs < self.mi_rows;
        let has_cols = mi_col + hbs < self.mi_cols;
        let partition = self.read_partition(reader, mi_row, mi_col, has_rows, has_cols, n8)?;
        let sub = subsize(partition, bsize);
        if sub == BLOCK_INVALID {
            return Err(bad("partition"));
        }
        if hbs == 0 {
            self.decode_block(reader, mi_row, mi_col, sub, 1, 1)?;
        } else {
            self.descend(
                reader,
                Split {
                    mi_row,
                    mi_col,
                    partition,
                    sub,
                    n4,
                    n8,
                    hbs,
                    has_rows,
                    has_cols,
                },
            )?;
        }
        if bsize >= BLOCK_8X8 && (bsize == BLOCK_8X8 || partition != PARTITION_SPLIT) {
            self.update_partition(mi_row, mi_col, sub, bsize)?;
        }
        Ok(())
    }

    fn descend(&mut self, reader: &mut BoolDecoder<'_>, split: Split) -> Result<(), CodecError> {
        let Split {
            mi_row,
            mi_col,
            partition,
            sub,
            n4,
            n8,
            hbs,
            has_rows,
            has_cols,
        } = split;
        match partition {
            PARTITION_NONE => self.decode_block(reader, mi_row, mi_col, sub, n4, n4),
            PARTITION_HORZ => {
                self.decode_block(reader, mi_row, mi_col, sub, n4, n8)?;
                if has_rows {
                    self.decode_block(reader, mi_row + hbs, mi_col, sub, n4, n8)?;
                }
                Ok(())
            }
            PARTITION_VERT => {
                self.decode_block(reader, mi_row, mi_col, sub, n8, n4)?;
                if has_cols {
                    self.decode_block(reader, mi_row, mi_col + hbs, sub, n8, n4)?;
                }
                Ok(())
            }
            _ => {
                self.decode_partition(reader, mi_row, mi_col, sub, n8)?;
                self.decode_partition(reader, mi_row, mi_col + hbs, sub, n8)?;
                self.decode_partition(reader, mi_row + hbs, mi_col, sub, n8)?;
                self.decode_partition(reader, mi_row + hbs, mi_col + hbs, sub, n8)
            }
        }
    }

    fn read_partition(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi_row: usize,
        mi_col: usize,
        has_rows: bool,
        has_cols: bool,
        bsl: u32,
    ) -> Result<u8, CodecError> {
        let ctx = self.partition_ctx(mi_row, mi_col, bsl);
        let probs = self.part[ctx];
        let partition = if has_rows && has_cols {
            prob::read_tree(reader, &prob::PARTITION_TREE, &probs)?
        } else if !has_rows && has_cols {
            if reader.read(u32::from(probs[1]))? == 1 {
                PARTITION_SPLIT
            } else {
                PARTITION_HORZ
            }
        } else if has_rows && !has_cols {
            if reader.read(u32::from(probs[2]))? == 1 {
                PARTITION_SPLIT
            } else {
                PARTITION_VERT
            }
        } else {
            PARTITION_SPLIT
        };
        if self.collect {
            let slot = &mut self.counts.partition[ctx][usize::from(partition)];
            *slot = slot.saturating_add(1);
        }
        Ok(partition)
    }

    fn partition_ctx(&self, mi_row: usize, mi_col: usize, bsl: u32) -> usize {
        let above = usize::from(self.above_part.get(mi_col).copied().unwrap_or(0) >> bsl) & 1;
        let left = usize::from(self.left_part[mi_row & 7] >> bsl) & 1;
        left * 2 + above + (bsl as usize) * 4
    }

    fn update_partition(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        subsize: u8,
        parent: u8,
    ) -> Result<(), CodecError> {
        let n = 1usize << mi_width_log2(parent);
        let (above, left) = partition_context_byte(subsize);
        let above_end = mi_col + n;
        let above_slot = self
            .above_part
            .get_mut(mi_col..above_end)
            .ok_or(bad("partition context"))?;
        above_slot.fill(above);
        let left_at = mi_row & 7;
        let left_slot = self
            .left_part
            .get_mut(left_at..left_at + n)
            .ok_or(bad("partition context"))?;
        left_slot.fill(left);
        Ok(())
    }

    fn decode_block(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi_row: usize,
        mi_col: usize,
        bsize: u8,
        bwl: u32,
        bhl: u32,
    ) -> Result<(), CodecError> {
        if bwl == 0 || bhl == 0 {
            return Err(bad("block size"));
        }
        let bw = 1usize << (bwl - 1);
        let bh = 1usize << (bhl - 1);
        let x_mis = bw.min(self.mi_cols - mi_col);
        let y_mis = bh.min(self.mi_rows - mi_row);
        let at = BlockAt {
            mi_row,
            mi_col,
            bsize,
            x_mis,
            y_mis,
            bw,
            bh,
        };
        let mi = if self.header.is_intra() {
            self.read_intra_mode(reader, mi_row, mi_col, bsize, x_mis, y_mis)?
        } else {
            self.read_inter_frame(reader, at)?
        };
        if mi.skip {
            self.reset_skip(mi_row, mi_col, bw, bh)?;
        }
        let pos = BlockPos {
            mi_row,
            mi_col,
            bw,
            bh,
        };
        if mi.refs[0] == INTRA_FRAME {
            self.recon_intra(reader, &mi, pos)?;
        } else {
            self.recon_inter(reader, &mi, pos)?;
        }
        self.store_mi(mi, mi_row, mi_col, x_mis, y_mis);
        self.note_filter(&mi, mi_row, mi_col, bw, bh);
        Ok(())
    }

    fn read_intra_mode(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi_row: usize,
        mi_col: usize,
        bsize: u8,
        x_mis: usize,
        y_mis: usize,
    ) -> Result<Mi, CodecError> {
        let (seg, predicted) = self.read_segment(reader, mi_row, mi_col, x_mis, y_mis)?;
        let mut mi = Mi {
            bsize,
            seg,
            seg_predicted: predicted,
            ..Mi::default()
        };
        mi.skip = self.read_skip(reader, mi_row, mi_col, mi.seg)?;
        mi.tx = self.read_tx(reader, bsize, mi_row, mi_col, true)?;
        self.read_y_modes(reader, &mut mi, mi_row, mi_col)?;
        let mode = usize::from(mi.mode);
        mi.uv = prob::read_tree(reader, &prob::INTRA_TREE, &KF_UV_MODE_PROB[mode])?;
        Ok(mi)
    }

    fn read_segment(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi_row: usize,
        mi_col: usize,
        x_mis: usize,
        y_mis: usize,
    ) -> Result<(u8, bool), CodecError> {
        if !self.header.seg.enabled {
            return Ok((0, false));
        }
        if self.header.is_intra() {
            return self.intra_segment(reader, mi_row, mi_col, x_mis, y_mis);
        }
        self.inter_segment(reader, mi_row, mi_col, x_mis, y_mis)
    }

    fn intra_segment(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi_row: usize,
        mi_col: usize,
        x_mis: usize,
        y_mis: usize,
    ) -> Result<(u8, bool), CodecError> {
        if !self.header.seg.update_map {
            self.copy_segment(mi_row, mi_col, x_mis, y_mis);
            return Ok((0, false));
        }
        let id = self.read_seg_id(reader)?;
        self.write_segment(mi_row, mi_col, x_mis, y_mis, id);
        Ok((id, false))
    }

    fn inter_segment(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi_row: usize,
        mi_col: usize,
        x_mis: usize,
        y_mis: usize,
    ) -> Result<(u8, bool), CodecError> {
        let predicted = self.predicted_segment(mi_row, mi_col, x_mis, y_mis)?;
        if !self.header.seg.update_map {
            self.copy_segment(mi_row, mi_col, x_mis, y_mis);
            return Ok((predicted, false));
        }
        if !self.header.seg.temporal {
            let id = self.read_seg_id(reader)?;
            self.write_segment(mi_row, mi_col, x_mis, y_mis, id);
            return Ok((id, false));
        }
        let ctx = self.seg_pred_ctx(mi_row, mi_col);
        let prob = u32::from(self.header.seg.pred_probs.get(ctx).copied().unwrap_or(255));
        let keep = reader.read(prob)? == 1;
        let id = if keep {
            predicted
        } else {
            self.read_seg_id(reader)?
        };
        self.write_segment(mi_row, mi_col, x_mis, y_mis, id);
        Ok((id, keep))
    }

    fn read_seg_id(&self, reader: &mut BoolDecoder<'_>) -> Result<u8, CodecError> {
        let id = prob::read_tree(reader, &prob::SEG_TREE, &self.header.seg.tree_probs)?;
        if id > 7 {
            return Err(bad("segment id"));
        }
        Ok(id)
    }

    fn predicted_segment(
        &self,
        mi_row: usize,
        mi_col: usize,
        x_mis: usize,
        y_mis: usize,
    ) -> Result<u8, CodecError> {
        let mut id = u8::MAX;
        let mut saw = false;
        for y in 0..y_mis {
            for x in 0..x_mis {
                let i = (mi_row + y) * self.mi_cols + mi_col + x;
                let Some(value) = self.last_seg.get(i).copied() else {
                    continue;
                };
                saw = true;
                id = id.min(value);
            }
        }
        if !saw {
            return Ok(0);
        }
        if id > 7 {
            return Err(bad("segment id"));
        }
        Ok(id)
    }

    fn seg_pred_ctx(&self, mi_row: usize, mi_col: usize) -> usize {
        let above = self
            .above_mi(mi_row, mi_col)
            .is_some_and(|mi| mi.seg_predicted);
        let left = self
            .left_mi(mi_row, mi_col)
            .is_some_and(|mi| mi.seg_predicted);
        usize::from(above) + usize::from(left)
    }

    fn copy_segment(&mut self, mi_row: usize, mi_col: usize, x_mis: usize, y_mis: usize) {
        for y in 0..y_mis {
            for x in 0..x_mis {
                let i = (mi_row + y) * self.mi_cols + mi_col + x;
                let value = self.last_seg.get(i).copied().unwrap_or(0);
                if let Some(slot) = self.segmap.get_mut(i) {
                    *slot = value;
                }
            }
        }
    }

    fn write_segment(&mut self, mi_row: usize, mi_col: usize, x_mis: usize, y_mis: usize, id: u8) {
        for y in 0..y_mis {
            for x in 0..x_mis {
                let i = (mi_row + y) * self.mi_cols + mi_col + x;
                if let Some(slot) = self.segmap.get_mut(i) {
                    *slot = id;
                }
            }
        }
    }

    fn read_skip(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi_row: usize,
        mi_col: usize,
        seg: u8,
    ) -> Result<bool, CodecError> {
        let seg_i = usize::from(seg);
        if self.header.seg.enabled && self.header.seg.feature_on[seg_i][3] {
            return Ok(true);
        }
        let ctx = self.above_skip(mi_row, mi_col) + self.left_skip(mi_row, mi_col);
        let skip = reader.read(u32::from(self.fc.skip[ctx]))? == 1;
        if self.collect {
            let bit = usize::from(skip);
            self.counts.skip[ctx][bit] = self.counts.skip[ctx][bit].saturating_add(1);
        }
        Ok(skip)
    }

    fn above_skip(&self, mi_row: usize, mi_col: usize) -> usize {
        if mi_row == 0 {
            0
        } else {
            usize::from(self.grid[(mi_row - 1) * self.mi_cols + mi_col].skip)
        }
    }

    fn left_skip(&self, mi_row: usize, mi_col: usize) -> usize {
        if mi_col > self.tile_col {
            usize::from(self.grid[mi_row * self.mi_cols + mi_col - 1].skip)
        } else {
            0
        }
    }

    fn read_tx(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        bsize: u8,
        mi_row: usize,
        mi_col: usize,
        allow: bool,
    ) -> Result<u8, CodecError> {
        let limit = max_tx(bsize);
        let biggest = tx_mode_biggest(self.header.tx_mode);
        if allow && self.header.tx_mode == TX_MODE_SELECT && bsize >= BLOCK_8X8 {
            let ctx = self.tx_ctx(limit, mi_row, mi_col);
            self.read_selected_tx(reader, limit, ctx)
        } else {
            Ok(limit.min(biggest))
        }
    }

    fn tx_ctx(&self, limit: u8, mi_row: usize, mi_col: usize) -> usize {
        let above = self.above_mi(mi_row, mi_col);
        let left = self.left_mi(mi_row, mi_col);
        let mut above_tx = neighbor_tx(above, limit);
        let mut left_tx = neighbor_tx(left, limit);
        if left.is_none() {
            left_tx = above_tx;
        }
        if above.is_none() {
            above_tx = left_tx;
        }
        usize::from(above_tx + left_tx > limit)
    }

    fn above_mi(&self, mi_row: usize, mi_col: usize) -> Option<&Mi> {
        if mi_row == 0 {
            None
        } else {
            self.grid.get((mi_row - 1) * self.mi_cols + mi_col)
        }
    }

    fn left_mi(&self, mi_row: usize, mi_col: usize) -> Option<&Mi> {
        if mi_col > self.tile_col {
            self.grid.get(mi_row * self.mi_cols + mi_col - 1)
        } else {
            None
        }
    }

    fn read_selected_tx(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        limit: u8,
        ctx: usize,
    ) -> Result<u8, CodecError> {
        let probs: &[u8] = match limit {
            TX_8X8 => &self.fc.tx8[ctx],
            TX_16X16 => &self.fc.tx16[ctx],
            _ => &self.fc.tx32[ctx],
        };
        let mut tx = u8::try_from(reader.read(u32::from(probs[0]))?).unwrap_or(0);
        if tx != TX_4X4 && limit >= TX_16X16 {
            tx = tx.saturating_add(u8::try_from(reader.read(u32::from(probs[1]))?).unwrap_or(0));
            if tx != TX_8X8 && limit >= TX_32X32 {
                tx =
                    tx.saturating_add(u8::try_from(reader.read(u32::from(probs[2]))?).unwrap_or(0));
            }
        }
        if self.collect {
            self.count_tx(limit, ctx, tx);
        }
        Ok(tx)
    }

    fn count_tx(&mut self, limit: u8, ctx: usize, tx: u8) {
        let tx = usize::from(tx);
        match limit {
            TX_8X8 => {
                self.counts.tx8[ctx][tx] = self.counts.tx8[ctx][tx].saturating_add(1);
            }
            TX_16X16 => {
                self.counts.tx16[ctx][tx] = self.counts.tx16[ctx][tx].saturating_add(1);
            }
            _ => {
                self.counts.tx32[ctx][tx] = self.counts.tx32[ctx][tx].saturating_add(1);
            }
        }
    }

    fn read_y_modes(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi: &mut Mi,
        mi_row: usize,
        mi_col: usize,
    ) -> Result<(), CodecError> {
        match mi.bsize {
            BLOCK_4X4 => {
                for i in 0..4 {
                    mi.bmi[i] = self.read_kf_y(reader, mi, mi_row, mi_col, i)?;
                }
                mi.mode = mi.bmi[3];
            }
            BLOCK_4X8 => {
                let top = self.read_kf_y(reader, mi, mi_row, mi_col, 0)?;
                mi.bmi[0] = top;
                mi.bmi[2] = top;
                let bottom = self.read_kf_y(reader, mi, mi_row, mi_col, 1)?;
                mi.bmi[1] = bottom;
                mi.bmi[3] = bottom;
                mi.mode = bottom;
            }
            BLOCK_8X4 => {
                let left = self.read_kf_y(reader, mi, mi_row, mi_col, 0)?;
                mi.bmi[0] = left;
                mi.bmi[1] = left;
                let right = self.read_kf_y(reader, mi, mi_row, mi_col, 2)?;
                mi.bmi[2] = right;
                mi.bmi[3] = right;
                mi.mode = right;
            }
            _ => {
                let mode = self.read_kf_y(reader, mi, mi_row, mi_col, 0)?;
                mi.mode = mode;
                mi.bmi = [mode; 4];
            }
        }
        Ok(())
    }

    fn read_kf_y(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi: &Mi,
        mi_row: usize,
        mi_col: usize,
        block: usize,
    ) -> Result<u8, CodecError> {
        let above = usize::from(self.above_mode(mi, mi_row, mi_col, block));
        let left = usize::from(self.left_mode(mi, mi_row, mi_col, block));
        prob::read_tree(reader, &prob::INTRA_TREE, &KF_Y_MODE_PROB[above][left])
    }

    fn above_mode(&self, mi: &Mi, mi_row: usize, mi_col: usize, block: usize) -> u8 {
        if block <= 1 {
            match self.above_mi(mi_row, mi_col) {
                Some(above) if above.refs[0] == INTRA_FRAME => y_mode(above, block + 2),
                _ => geom::DC,
            }
        } else {
            mi.bmi[block - 2]
        }
    }

    fn left_mode(&self, mi: &Mi, mi_row: usize, mi_col: usize, block: usize) -> u8 {
        if block == 0 || block == 2 {
            match self.left_mi(mi_row, mi_col) {
                Some(left) if left.refs[0] == INTRA_FRAME => y_mode(left, block + 1),
                _ => geom::DC,
            }
        } else {
            mi.bmi[block - 1]
        }
    }

    fn reset_skip(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bw: usize,
        bh: usize,
    ) -> Result<(), CodecError> {
        self.zero_plane(0, mi_row, mi_col, bw << 1, bh << 1)?;
        self.zero_plane(1, mi_row, mi_col, bw, bh)?;
        self.zero_plane(2, mi_row, mi_col, bw, bh)
    }

    fn zero_plane(
        &mut self,
        plane: usize,
        mi_row: usize,
        mi_col: usize,
        n4_w: usize,
        n4_h: usize,
    ) -> Result<(), CodecError> {
        let ss = usize::from(plane > 0);
        let above_at = (mi_col * 2) >> ss;
        let left_at = ((mi_row * 2) & 15) >> ss;
        match plane {
            0 => {
                paint_zero(&mut self.above_y, above_at, n4_w)?;
                paint_zero(&mut self.left_y, left_at, n4_h)
            }
            1 => {
                paint_zero(&mut self.above_u, above_at, n4_w)?;
                paint_zero(&mut self.left_u, left_at, n4_h)
            }
            _ => {
                paint_zero(&mut self.above_v, above_at, n4_w)?;
                paint_zero(&mut self.left_v, left_at, n4_h)
            }
        }
    }

    fn recon_intra(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi: &Mi,
        pos: BlockPos,
    ) -> Result<(), CodecError> {
        for plane in 0..3 {
            self.recon_plane(reader, mi, plane, pos, false)?;
        }
        Ok(())
    }

    fn recon_plane(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi: &Mi,
        plane: usize,
        pos: BlockPos,
        inter: bool,
    ) -> Result<(), CodecError> {
        let ss = u32::from(plane > 0);
        let tx = if plane == 0 {
            mi.tx
        } else {
            uv_tx_420(mi.bsize, mi.tx)
        };
        let n4_w = (pos.bw << 1) >> ss;
        let n4_h = (pos.bh << 1) >> ss;
        let step = 1usize << tx;
        let right = edge_delta(self.mi_cols, pos.bw, pos.mi_col);
        let bottom = edge_delta(self.mi_rows, pos.bh, pos.mi_row);
        let visit_w = limit_blocks(n4_w, right, ss);
        let visit_h = limit_blocks(n4_h, bottom, ss);
        let shift_w = if right >= 0 { 0 } else { visit_w };
        let shift_h = if bottom >= 0 { 0 } else { visit_h };
        for row in (0..visit_h).step_by(step) {
            for col in (0..visit_w).step_by(step) {
                let at = TxPos {
                    plane,
                    mi_row: pos.mi_row,
                    mi_col: pos.mi_col,
                    tx,
                    row,
                    col,
                    n4_w,
                    ss,
                    shift_w,
                    shift_h,
                };
                self.recon_tx(reader, mi, &at, inter)?;
            }
        }
        Ok(())
    }

    fn recon_tx(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi: &Mi,
        at: &TxPos,
        inter: bool,
    ) -> Result<(), CodecError> {
        let mode = block_mode(mi, at.plane, at.row, at.col);
        let x = ((at.mi_col << 3) >> at.ss) + (at.col << 2);
        let y = ((at.mi_row << 3) >> at.ss) + (at.row << 2);
        if !inter {
            let have_above = at.row > 0 || at.mi_row > 0;
            let have_left = at.col > 0 || at.mi_col > self.tile_col;
            let have_right = at.col + (1usize << at.tx) < at.n4_w;
            let (crop_w, crop_h, stride, dest) = self.plane_mut(at.plane);
            intra::predict(
                mode,
                dest,
                intra::Site {
                    tx: at.tx,
                    stride,
                    crop_w,
                    crop_h,
                    x,
                    y,
                    have_left,
                    have_above,
                    have_right,
                },
            );
        }
        if mi.skip {
            return Ok(());
        }
        let ctx = self.coef_ctx(at)?;
        let eob = self.decode_tx(reader, mi, at, Residual { mode, x, y, ctx }, inter)?;
        self.write_eob(at, eob)
    }

    fn plane_mut(&mut self, plane: usize) -> (usize, usize, usize, &mut [u8]) {
        // Intra edges use the aligned plane, libvpx `y_width` / `uv_width`,
        // not the stride. The stride includes the frame border.
        match plane {
            0 => (self.plane_w, self.plane_h, self.y_stride, &mut self.y),
            1 => (
                self.uv_plane_w,
                self.uv_plane_h,
                self.uv_stride,
                &mut self.u,
            ),
            _ => (
                self.uv_plane_w,
                self.uv_plane_h,
                self.uv_stride,
                &mut self.v,
            ),
        }
    }

    fn coef_ctx(&self, at: &TxPos) -> Result<usize, CodecError> {
        let ss = usize::from(at.plane > 0);
        let n = 1usize << at.tx;
        let above_at = ((at.mi_col * 2) >> ss) + at.col;
        let left_at = (((at.mi_row * 2) & 15) >> ss) + at.row;
        let above = plane_above(self, at.plane);
        let left = plane_left(self, at.plane);
        let above_nz = region_nz(above, above_at, n)?;
        let left_nz = region_nz(left, left_at, n)?;
        Ok(usize::from(above_nz) + usize::from(left_nz))
    }

    fn decode_tx(
        &mut self,
        reader: &mut BoolDecoder<'_>,
        mi: &Mi,
        at: &TxPos,
        residual: Residual,
        inter: bool,
    ) -> Result<usize, CodecError> {
        let kind = if inter || at.plane > 0 || self.header.lossless || at.tx == TX_32X32 {
            DCT_DCT
        } else {
            intra_tx_type(residual.mode)
        };
        let (scan, neighbors) = scan_pair(at.tx, kind);
        self.coef.fill(0);
        let seg = usize::from(mi.seg);
        let dq = if at.plane == 0 {
            self.header.y_dq[seg]
        } else {
            self.header.uv_dq[seg]
        };
        let eob = decode_coefs(
            reader,
            self.fc,
            &mut self.counts,
            self.collect,
            usize::from(at.tx),
            usize::from(at.plane > 0),
            inter,
            dq,
            residual.ctx,
            scan,
            neighbors,
            &mut self.coef,
        )?;
        if eob > 0 {
            let inv = Inv {
                coef: &self.coef,
                stride: if at.plane == 0 {
                    self.y_stride
                } else {
                    self.uv_stride
                },
                n: 4usize << at.tx,
                kind,
                lossless: self.header.lossless,
                x: residual.x,
                y: residual.y,
            };
            match at.plane {
                0 => add_inverse(&mut self.y, &inv)?,
                1 => add_inverse(&mut self.u, &inv)?,
                _ => add_inverse(&mut self.v, &inv)?,
            }
        }
        Ok(eob)
    }

    fn write_eob(&mut self, at: &TxPos, eob: usize) -> Result<(), CodecError> {
        let ss = usize::from(at.plane > 0);
        let n = 1usize << at.tx;
        let above_at = ((at.mi_col * 2) >> ss) + at.col;
        let left_at = (((at.mi_row * 2) & 15) >> ss) + at.row;
        let shift_a = shift_blocks(at.shift_w, at.col, n);
        let shift_l = shift_blocks(at.shift_h, at.row, n);
        let (above, left): (&mut [u8], &mut [u8]) = match at.plane {
            0 => (&mut self.above_y, &mut self.left_y),
            1 => (&mut self.above_u, &mut self.left_u),
            _ => (&mut self.above_v, &mut self.left_v),
        };
        paint_eob(above, above_at, n, eob, shift_a)?;
        paint_eob(left, left_at, n, eob, shift_l)
    }

    fn store_mi(&mut self, mi: Mi, mi_row: usize, mi_col: usize, x_mis: usize, y_mis: usize) {
        for y in 0..y_mis {
            for x in 0..x_mis {
                let i = (mi_row + y) * self.mi_cols + mi_col + x;
                if let Some(slot) = self.grid.get_mut(i) {
                    *slot = mi;
                }
            }
        }
    }

    fn note_filter(&mut self, mi: &Mi, mi_row: usize, mi_col: usize, bw: usize, bh: usize) {
        if self.header.filter_level == 0 {
            return;
        }
        let index = (mi_row / 8) * self.sb_cols + mi_col / 8;
        let Some(mask) = self.masks.get_mut(index) else {
            return;
        };
        lpf::build(
            mask,
            &self.levels,
            &lpf::Block {
                bsize: mi.bsize,
                tx: mi.tx,
                skip: mi.skip,
                inter: mi.refs[0] != INTRA_FRAME,
                seg: mi.seg,
                mode: mi.mode,
                reference: mi.refs[0],
                bw,
                bh,
            },
            mi_row,
            mi_col,
        );
    }

    fn loop_filter(&mut self) -> Result<(), CodecError> {
        if self.header.filter_level == 0 {
            return Ok(());
        }
        let levels = self.levels;
        let sb_rows = self.mi_rows.div_ceil(8);
        for sb_row in 0..sb_rows {
            for sb_col in 0..self.sb_cols {
                let index = sb_row * self.sb_cols + sb_col;
                let Some(mask) = self.masks.get(index).copied() else {
                    return Err(bad("loop filter mask"));
                };
                let mut mask = mask;
                lpf::filter_sb(
                    &mut mask,
                    &levels,
                    &mut lpf::Planes {
                        y: &mut self.y,
                        u: &mut self.u,
                        v: &mut self.v,
                        y_stride: self.y_stride,
                        uv_stride: self.uv_stride,
                    },
                    lpf::Origin {
                        mi_row: sb_row * 8,
                        mi_col: sb_col * 8,
                        mi_rows: self.mi_rows,
                        mi_cols: self.mi_cols,
                    },
                )?;
            }
        }
        Ok(())
    }

    fn picture(&self) -> Result<Picture, CodecError> {
        Ok(Picture {
            y: crop_plane(&self.y, self.y_stride, self.crop_w, self.crop_h)?,
            u: crop_plane(&self.u, self.uv_stride, self.uv_w, self.uv_h)?,
            v: crop_plane(&self.v, self.uv_stride, self.uv_w, self.uv_h)?,
            width: self.header.width,
            height: self.header.height,
        })
    }
}

fn neighbor_tx(mi: Option<&Mi>, limit: u8) -> u8 {
    match mi {
        Some(mi) if !mi.skip => mi.tx,
        _ => limit,
    }
}

fn y_mode(mi: &Mi, block: usize) -> u8 {
    if mi.bsize < BLOCK_8X8 {
        mi.bmi[block]
    } else {
        mi.mode
    }
}

fn block_mode(mi: &Mi, plane: usize, row: usize, col: usize) -> u8 {
    if plane == 0 && mi.bsize < BLOCK_8X8 {
        mi.bmi[(row << 1) + col]
    } else if plane == 0 {
        mi.mode
    } else {
        mi.uv
    }
}

fn plane_above<'a>(frame: &'a Scratch<'a>, plane: usize) -> &'a [u8] {
    match plane {
        0 => &frame.above_y,
        1 => &frame.above_u,
        _ => &frame.above_v,
    }
}

fn plane_left<'a>(frame: &'a Scratch<'a>, plane: usize) -> &'a [u8] {
    match plane {
        0 => &frame.left_y,
        1 => &frame.left_u,
        _ => &frame.left_v,
    }
}

fn region_nz(buf: &[u8], at: usize, n: usize) -> Result<bool, CodecError> {
    let slot = buf.get(at..at + n).ok_or(bad("entropy context"))?;
    Ok(slot.iter().any(|byte| *byte != 0))
}

fn paint_zero(buf: &mut [u8], at: usize, n: usize) -> Result<(), CodecError> {
    let slot = buf.get_mut(at..at + n).ok_or(bad("entropy context"))?;
    slot.fill(0);
    Ok(())
}

fn paint_eob(
    buf: &mut [u8],
    at: usize,
    n: usize,
    eob: usize,
    shift: usize,
) -> Result<(), CodecError> {
    let slot = buf.get_mut(at..at + n).ok_or(bad("entropy context"))?;
    let on = u8::from(eob > 0);
    let keep = n.saturating_sub(shift);
    for (i, byte) in slot.iter_mut().enumerate() {
        *byte = if i < keep { on } else { 0 };
    }
    Ok(())
}

fn shift_blocks(max_blocks: usize, at: usize, tx_blocks: usize) -> usize {
    if max_blocks > 0 && tx_blocks + at > max_blocks {
        tx_blocks - (max_blocks - at)
    } else {
        0
    }
}

fn edge_delta(mis: usize, block: usize, at: usize) -> i32 {
    let delta = mis as i32 - block as i32 - at as i32;
    delta.saturating_mul(64)
}

fn limit_blocks(n4: usize, edge: i32, ss: u32) -> usize {
    if edge >= 0 {
        n4
    } else {
        let max = n4 as i32 + (edge >> (5 + ss));
        usize::try_from(max.max(0)).unwrap_or(0)
    }
}

fn add_inverse(dest: &mut [u8], inv: &Inv<'_>) -> Result<(), CodecError> {
    let start = inv.y * inv.stride + inv.x;
    let need = (inv.n - 1) * inv.stride + inv.n;
    let Some(slot) = dest.get_mut(start..).filter(|tail| tail.len() >= need) else {
        return Err(bad("inverse plane"));
    };
    if inv.lossless {
        iwht4x4_add(inv.coef, slot, inv.stride);
    } else {
        iht_add(inv.coef, slot, inv.stride, inv.n, inv.kind);
    }
    Ok(())
}

fn tile_span(index: usize, mis: usize, log2: u8) -> usize {
    let sb = mis.div_ceil(8);
    let shifted = (index.saturating_mul(sb)) >> u32::from(log2);
    (shifted << 3).min(mis)
}

fn take_tile<'a>(data: &mut &'a [u8], last: bool) -> Result<&'a [u8], CodecError> {
    if last {
        let tile = *data;
        *data = &[];
        return Ok(tile);
    }
    if data.len() < 4 {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "tile size",
        });
    }
    let size = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
    *data = &data[4..];
    if data.len() < size {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "tile",
        });
    }
    let (tile, rest) = data.split_at(size);
    *data = rest;
    Ok(tile)
}

fn crop_plane(
    src: &[u8],
    stride: usize,
    width: usize,
    height: usize,
) -> Result<Vec<u8>, CodecError> {
    let mut out = vec![0u8; width.saturating_mul(height)];
    for row in 0..height {
        let from = row * stride;
        let to = row * width;
        let src_row = src.get(from..from + width).ok_or(bad("crop"))?;
        out[to..to + width].copy_from_slice(src_row);
    }
    Ok(out)
}

fn scan_pair(tx: u8, kind: u8) -> (&'static [u16], &'static [u16]) {
    match (tx, kind) {
        (TX_4X4, 1) => (&SCAN_ROW_4X4, &NEIGHBORS_ROW_4X4),
        (TX_4X4, 2) => (&SCAN_COL_4X4, &NEIGHBORS_COL_4X4),
        (TX_4X4, _) => (&SCAN_DEFAULT_4X4, &NEIGHBORS_DEFAULT_4X4),
        (TX_8X8, 1) => (&SCAN_ROW_8X8, &NEIGHBORS_ROW_8X8),
        (TX_8X8, 2) => (&SCAN_COL_8X8, &NEIGHBORS_COL_8X8),
        (TX_8X8, _) => (&SCAN_DEFAULT_8X8, &NEIGHBORS_DEFAULT_8X8),
        (TX_16X16, 1) => (&SCAN_ROW_16X16, &NEIGHBORS_ROW_16X16),
        (TX_16X16, 2) => (&SCAN_COL_16X16, &NEIGHBORS_COL_16X16),
        (TX_16X16, _) => (&SCAN_DEFAULT_16X16, &NEIGHBORS_DEFAULT_16X16),
        _ => (&SCAN_DEFAULT_32X32, &NEIGHBORS_DEFAULT_32X32),
    }
}

fn bad(what: &'static str) -> CodecError {
    CodecError::Malformed { at_bit: 0, what }
}

#[cfg(test)]
mod tests {
    use super::{RefSlots, reconstruct};
    use crate::header::{Persist, parse};

    /// libvpx-vp9 lossless 16×16 red keyframe. ffmpeg raw yuv420p is Y=81, U=90, V=240.
    const RED_16: [u8; 42] = [
        0x82, 0x49, 0x83, 0x42, 0x00, 0x00, 0xf0, 0x00, 0xf6, 0x00, 0x38, 0x24, 0x1c, 0x18, 0x00,
        0x00, 0x00, 0x20, 0x00, 0x00, 0x10, 0xbf, 0xff, 0xff, 0xc8, 0xd7, 0x80, 0x31, 0xff, 0xff,
        0xff, 0xec, 0xfc, 0x87, 0x0f, 0xff, 0xff, 0xff, 0x6c, 0x8c, 0x00, 0x00,
    ];

    #[test]
    fn lossless_red_keyframe_matches_ffmpeg() {
        let parsed = parse(&Persist::new(), &RED_16).expect("header");
        let pic = reconstruct(&parsed, &RefSlots::none(), None, &[])
            .expect("recon")
            .picture;
        assert_eq!((pic.width, pic.height), (16, 16));
        assert_eq!(pic.y.len(), 256);
        assert_eq!(pic.u.len(), 64);
        assert_eq!(pic.v.len(), 64);
        let y = pic.y.iter().position(|sample| *sample != 81);
        let u = pic.u.iter().position(|sample| *sample != 90);
        let v = pic.v.iter().position(|sample| *sample != 240);
        assert_eq!(y, None, "Y {}", y.map(|i| pic.y[i]).unwrap_or(0));
        assert_eq!(u, None, "U {}", u.map(|i| pic.u[i]).unwrap_or(0));
        assert_eq!(v, None, "V {}", v.map(|i| pic.v[i]).unwrap_or(0));
    }

    #[test]
    fn lossy_keyframes_match_ffvp9() {
        let cases = [
            (
                "red32",
                include_bytes!("../fixtures/vp9_red32_q_0.bin").as_slice(),
                include_bytes!("../fixtures/vp9_red32_q.yuv").as_slice(),
                32u32,
                32u32,
            ),
            (
                "src32",
                include_bytes!("../fixtures/vp9_src32_0.bin").as_slice(),
                include_bytes!("../fixtures/vp9_src32.yuv").as_slice(),
                32,
                32,
            ),
            (
                "src64",
                include_bytes!("../fixtures/vp9_src64_0.bin").as_slice(),
                include_bytes!("../fixtures/vp9_src64_0.yuv").as_slice(),
                64,
                64,
            ),
        ];
        for (name, payload, raw, w, h) in cases {
            let parsed = parse(&Persist::new(), payload).expect("header");
            let pic = reconstruct(&parsed, &RefSlots::none(), None, &[])
                .unwrap_or_else(|err| panic!("{name} recon {err}"))
                .picture;
            assert_eq!((pic.width, pic.height), (w, h), "{name}");
            let got = [&pic.y[..], &pic.u[..], &pic.v[..]].concat();
            assert_eq!(got.len(), raw.len(), "{name}");
            if let Some(i) = got.iter().zip(raw).position(|(a, b)| a != b) {
                panic!("{name} mismatch at {i} got {} want {}", got[i], raw[i]);
            }
        }
    }
}

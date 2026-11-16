//! In-loop filter for profile 0.
//!
//! Edge selection follows libvpx `vp9_build_mask` and `vp9_adjust_mask`.
//! Pixel updates follow the scalar filters in `vpx_dsp/loopfilter.c`.
//! A frame whose level is zero is left unchanged, which is what libvpx does
//! even when a segment requests a different level. Taps that leave the frame
//! buffer read zero. The buffer keeps a border past the aligned plane, and a
//! transform may have written real samples into that border.

use crate::geom::{self, uv_tx_420};
use crate::header::Header;

const LEFT_TX: [u64; 4] = [
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0x5555_5555_5555_5555,
    0x1111_1111_1111_1111,
];
const ABOVE_TX: [u64; 4] = [
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0x00ff_00ff_00ff_00ff,
    0x0000_00ff_0000_00ff,
];
const LEFT_PRED: [u64; 13] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_0001,
    0x0000_0000_0000_0001,
    0x0000_0000_0000_0001,
    0x0000_0000_0000_0101,
    0x0000_0000_0000_0001,
    0x0000_0000_0000_0101,
    0x0000_0000_0101_0101,
    0x0000_0000_0000_0101,
    0x0000_0000_0101_0101,
    0x0101_0101_0101_0101,
    0x0000_0000_0101_0101,
    0x0101_0101_0101_0101,
];
const ABOVE_PRED: [u64; 13] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_0001,
    0x0000_0000_0000_0001,
    0x0000_0000_0000_0001,
    0x0000_0000_0000_0001,
    0x0000_0000_0000_0003,
    0x0000_0000_0000_0003,
    0x0000_0000_0000_0003,
    0x0000_0000_0000_000f,
    0x0000_0000_0000_000f,
    0x0000_0000_0000_000f,
    0x0000_0000_0000_00ff,
    0x0000_0000_0000_00ff,
];
const SIZE_Y: [u64; 13] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_0001,
    0x0000_0000_0000_0001,
    0x0000_0000_0000_0001,
    0x0000_0000_0000_0101,
    0x0000_0000_0000_0003,
    0x0000_0000_0000_0303,
    0x0000_0000_0303_0303,
    0x0000_0000_0000_0f0f,
    0x0000_0000_0f0f_0f0f,
    0x0f0f_0f0f_0f0f_0f0f,
    0x0000_0000_ffff_ffff,
    0xffff_ffff_ffff_ffff,
];
const LEFT_BORDER: u64 = 0x1111_1111_1111_1111;
const ABOVE_BORDER: u64 = 0x0000_00ff_0000_00ff;
const LEFT_TX_UV: [u16; 4] = [0xffff, 0xffff, 0x5555, 0x1111];
const ABOVE_TX_UV: [u16; 4] = [0xffff, 0xffff, 0x0f0f, 0x000f];
const LEFT_PRED_UV: [u16; 13] = [
    0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0011, 0x0001, 0x0011, 0x1111, 0x0011,
    0x1111,
];
const ABOVE_PRED_UV: [u16; 13] = [
    0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0003, 0x0003, 0x0003, 0x000f,
    0x000f,
];
const SIZE_UV: [u16; 13] = [
    0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0011, 0x0003, 0x0033, 0x3333, 0x00ff,
    0xffff,
];
const LEFT_BORDER_UV: u16 = 0x1111;
const ABOVE_BORDER_UV: u16 = 0x000f;
const FIRST_UV: [[u8; 8]; 8] = [
    [1, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
];

#[derive(Clone, Copy, Default)]
struct Thresh {
    lim: u8,
    blimit: u8,
    hev: u8,
}

#[derive(Clone, Copy)]
pub struct Levels {
    thr: [Thresh; 64],
    lvl: [[[u8; 2]; 4]; 8],
}

impl Levels {
    /// The thresholds of filter level `level`.
    pub(crate) fn dsp(&self, level: u8) -> crate::dsp::lf::Thresh {
        let t = self.thr[usize::from(level & 63)];
        crate::dsp::lf::Thresh {
            blimit: t.blimit,
            limit: t.lim,
            hev: t.hev,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Mask {
    pub(crate) left_y: [u64; 4],
    pub(crate) above_y: [u64; 4],
    pub(crate) int_y: u64,
    pub(crate) left_uv: [u16; 4],
    pub(crate) above_uv: [u16; 4],
    pub(crate) int_uv: u16,
    pub(crate) lfl_y: [u8; 64],
}

impl Default for Mask {
    fn default() -> Self {
        Self {
            left_y: [0; 4],
            above_y: [0; 4],
            int_y: 0,
            left_uv: [0; 4],
            above_uv: [0; 4],
            int_uv: 0,
            lfl_y: [0; 64],
        }
    }
}

pub struct Block {
    pub bsize: u8,
    pub tx: u8,
    pub skip: bool,
    pub inter: bool,
    pub seg: u8,
    pub mode: u8,
    pub reference: u8,
    pub bw: usize,
    pub bh: usize,
}

pub fn levels(header: &Header) -> Levels {
    let mut thr = [Thresh::default(); 64];
    let sharp = u32::from(header.sharpness.min(7));
    let shift = u32::from(sharp > 0) + u32::from(sharp > 4);
    for lvl in 0u32..64 {
        let mut limit = lvl >> shift;
        if sharp > 0 {
            limit = limit.min(9u32.saturating_sub(sharp));
        }
        if limit < 1 {
            limit = 1;
        }
        let blimit = 2 * (lvl + 2) + limit;
        thr[lvl as usize] = Thresh {
            lim: u8::try_from(limit).unwrap_or(1),
            blimit: u8::try_from(blimit).unwrap_or(1),
            hev: u8::try_from(lvl >> 4).unwrap_or(0),
        };
    }
    Levels {
        thr,
        lvl: segment_levels(header),
    }
}

pub fn build(mask: &mut Mask, levels: &Levels, block: &Block, mi_row: usize, mi_col: usize) {
    let bsize = usize::from(block.bsize);
    if bsize >= 13 {
        return;
    }
    let tx = usize::from(block.tx);
    if tx > 3 {
        return;
    }
    let Some(level) = levels.level(block.seg, block.reference, block.mode) else {
        return;
    };
    if level == 0 {
        return;
    }
    let uv_tx = usize::from(uv_tx_420(block.bsize, block.tx));
    let row_in = mi_row & 7;
    let col_in = mi_col & 7;
    let shift_y = col_in + (row_in << 3);
    let shift_uv = (col_in >> 1) + ((row_in >> 1) << 2);
    let build_uv = FIRST_UV[row_in][col_in] == 1;
    paint_level(mask, level, shift_y, block.bw, block.bh);
    mask.above_y[tx] |= ABOVE_PRED[bsize] << shift_y;
    mask.left_y[tx] |= LEFT_PRED[bsize] << shift_y;
    if build_uv {
        mask.above_uv[uv_tx] |= ABOVE_PRED_UV[bsize] << shift_uv;
        mask.left_uv[uv_tx] |= LEFT_PRED_UV[bsize] << shift_uv;
    }
    if block.skip && block.inter {
        return;
    }
    mask.above_y[tx] |= (SIZE_Y[bsize] & ABOVE_TX[tx]) << shift_y;
    mask.left_y[tx] |= (SIZE_Y[bsize] & LEFT_TX[tx]) << shift_y;
    if build_uv {
        mask.above_uv[uv_tx] |= (SIZE_UV[bsize] & ABOVE_TX_UV[uv_tx]) << shift_uv;
        mask.left_uv[uv_tx] |= (SIZE_UV[bsize] & LEFT_TX_UV[uv_tx]) << shift_uv;
    }
    if tx == 0 {
        mask.int_y |= SIZE_Y[bsize] << shift_y;
    }
    if build_uv && uv_tx == 0 {
        mask.int_uv |= SIZE_UV[bsize] << shift_uv;
    }
}

impl Levels {
    pub(crate) fn level(&self, seg: u8, reference: u8, mode: u8) -> Option<u8> {
        let seg = usize::from(seg);
        let reference = usize::from(reference);
        if seg >= 8 || reference >= 4 {
            return None;
        }
        let mode = usize::from(matches!(mode, geom::NEAREST | geom::NEAR | geom::NEW));
        Some(self.lvl[seg][reference][mode])
    }
}

fn segment_levels(header: &Header) -> [[[u8; 2]; 4]; 8] {
    let scale = 1i32 << (i32::from(header.filter_level) >> 5);
    let mut lvl = [[[0u8; 2]; 4]; 8];
    for (seg, row) in lvl.iter_mut().enumerate() {
        let base = segment_base(header, seg);
        if header.delta_enabled {
            row[0][0] = clamp_level(base + i32::from(header.ref_deltas[0]) * scale);
            for (reference, modes) in row.iter_mut().enumerate().skip(1) {
                for (mode, slot) in modes.iter_mut().enumerate() {
                    let delta = i32::from(header.ref_deltas[reference])
                        + i32::from(header.mode_deltas[mode]);
                    *slot = clamp_level(base + delta * scale);
                }
            }
        } else {
            let value = clamp_level(base);
            for modes in &mut *row {
                modes.fill(value);
            }
        }
    }
    lvl
}

fn segment_base(header: &Header, seg: usize) -> i32 {
    let mut base = i32::from(header.filter_level);
    if header.seg.enabled && header.seg.feature_on[seg][1] {
        let data = i32::from(header.seg.feature_data[seg][1]);
        base = if header.seg.abs_delta {
            data
        } else {
            base + data
        };
        base = base.clamp(0, 63);
    }
    base
}

fn clamp_level(value: i32) -> u8 {
    u8::try_from(value.clamp(0, 63)).unwrap_or(0)
}

fn paint_level(mask: &mut Mask, level: u8, shift_y: usize, bw: usize, bh: usize) {
    let mut index = shift_y;
    for _ in 0..bh {
        for dx in 0..bw {
            if let Some(slot) = mask.lfl_y.get_mut(index + dx) {
                *slot = level;
            }
        }
        index += 8;
    }
}

impl Mask {
    pub(crate) fn adjust(&mut self, mi_row: usize, mi_col: usize, mi_rows: usize, mi_cols: usize) {
        self.left_y[2] |= self.left_y[3];
        self.above_y[2] |= self.above_y[3];
        self.left_uv[2] |= self.left_uv[3];
        self.above_uv[2] |= self.above_uv[3];
        promote_border(&mut self.left_y, LEFT_BORDER);
        promote_border(&mut self.above_y, ABOVE_BORDER);
        promote_border_uv(&mut self.left_uv, LEFT_BORDER_UV);
        promote_border_uv(&mut self.above_uv, ABOVE_BORDER_UV);
        if mi_row + 8 > mi_rows {
            clip_rows(self, mi_rows - mi_row);
        }
        if mi_col + 8 > mi_cols {
            clip_cols(self, mi_cols - mi_col);
        }
        if mi_col == 0 {
            for slot in &mut self.left_y[..3] {
                *slot &= 0xfefe_fefe_fefe_fefe;
            }
            for slot in &mut self.left_uv[..3] {
                *slot &= 0xeeee;
            }
        }
    }
}

fn promote_border(masks: &mut [u64; 4], border: u64) {
    masks[1] |= masks[0] & border;
    masks[0] &= !border;
}

fn promote_border_uv(masks: &mut [u16; 4], border: u16) {
    masks[1] |= masks[0] & border;
    masks[0] &= !border;
}

fn clip_rows(mask: &mut Mask, rows: usize) {
    let mask_y = (1u64 << (rows * 8)) - 1;
    let uv_bits = ((rows + 1) >> 1) << 2;
    let mask_uv =
        u16::try_from((1u32 << u32::try_from(uv_bits).unwrap_or(0)) - 1).unwrap_or(0xffff);
    for i in 0..3 {
        mask.left_y[i] &= mask_y;
        mask.above_y[i] &= mask_y;
        mask.left_uv[i] &= mask_uv;
        mask.above_uv[i] &= mask_uv;
    }
    mask.int_y &= mask_y;
    mask.int_uv &= mask_uv;
    if rows == 1 {
        mask.above_uv[1] |= mask.above_uv[2];
        mask.above_uv[2] = 0;
    }
    if rows == 5 {
        mask.above_uv[1] |= mask.above_uv[2] & 0xff00;
        mask.above_uv[2] &= 0x00ff;
    }
}

fn clip_cols(mask: &mut Mask, columns: usize) {
    let low = (1u64 << columns) - 1;
    let mask_y = low.wrapping_mul(0x0101_0101_0101_0101);
    let outer = (1u32 << u32::try_from((columns + 1) >> 1).unwrap_or(0)) - 1;
    let inner = (1u32 << u32::try_from(columns >> 1).unwrap_or(0)) - 1;
    let mask_uv = u16::try_from(outer * 0x1111).unwrap_or(0xffff);
    let mask_int = u16::try_from(inner * 0x1111).unwrap_or(0);
    for i in 0..3 {
        mask.left_y[i] &= mask_y;
        mask.above_y[i] &= mask_y;
        mask.left_uv[i] &= mask_uv;
        mask.above_uv[i] &= mask_uv;
    }
    mask.int_y &= mask_y;
    mask.int_uv &= mask_int;
    if columns == 1 {
        mask.left_uv[1] |= mask.left_uv[2];
        mask.left_uv[2] = 0;
    }
    if columns == 5 {
        mask.left_uv[1] |= mask.left_uv[2] & 0xcccc;
        mask.left_uv[2] &= 0x3333;
    }
}

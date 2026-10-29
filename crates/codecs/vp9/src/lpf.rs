//! In-loop filter for profile 0.
//!
//! Edge selection follows libvpx `vp9_build_mask` and `vp9_adjust_mask`.
//! Pixel updates follow the scalar filters in `vpx_dsp/loopfilter.c`.
//! A frame whose level is zero is left unchanged, which is what libvpx does
//! even when a segment requests a different level.

use cutlass_decode::CodecError;

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

#[derive(Clone, Copy)]
pub struct Mask {
    left_y: [u64; 4],
    above_y: [u64; 4],
    int_y: u64,
    left_uv: [u16; 4],
    above_uv: [u16; 4],
    int_uv: u16,
    lfl_y: [u8; 64],
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

pub struct Planes<'a> {
    pub y: &'a mut [u8],
    pub u: &'a mut [u8],
    pub v: &'a mut [u8],
    pub y_stride: usize,
    pub uv_stride: usize,
}

#[derive(Clone, Copy)]
pub struct Origin {
    pub mi_row: usize,
    pub mi_col: usize,
    pub mi_rows: usize,
    pub mi_cols: usize,
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

pub fn filter_sb(
    mask: &mut Mask,
    levels: &Levels,
    planes: &mut Planes<'_>,
    origin: Origin,
) -> Result<(), CodecError> {
    mask.adjust(origin.mi_row, origin.mi_col, origin.mi_rows, origin.mi_cols);
    filter_luma(mask, levels, planes.y, planes.y_stride, &origin)?;
    filter_chroma(mask, levels, planes.u, planes.uv_stride, &origin)?;
    filter_chroma(mask, levels, planes.v, planes.uv_stride, &origin)?;
    Ok(())
}

impl Levels {
    fn level(&self, seg: u8, reference: u8, mode: u8) -> Option<u8> {
        let seg = usize::from(seg);
        let reference = usize::from(reference);
        if seg >= 8 || reference >= 4 {
            return None;
        }
        let mode = usize::from(matches!(mode, geom::NEAREST | geom::NEAR | geom::NEW));
        Some(self.lvl[seg][reference][mode])
    }

    fn thresh(&self, level: u8) -> Option<Thresh> {
        self.thr.get(usize::from(level)).copied()
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
    fn adjust(&mut self, mi_row: usize, mi_col: usize, mi_rows: usize, mi_cols: usize) {
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

fn filter_luma(
    mask: &Mask,
    levels: &Levels,
    plane: &mut [u8],
    stride: usize,
    origin: &Origin,
) -> Result<(), CodecError> {
    for row in 0..8 {
        if origin.mi_row + row >= origin.mi_rows {
            break;
        }
        let cols = origin.mi_cols.saturating_sub(origin.mi_col).min(8);
        let y = (origin.mi_row + row) * 8;
        let lfl = &mask.lfl_y[row * 8..(row + 1) * 8];
        vertical_edges(
            plane,
            stride,
            origin.mi_col * 8,
            y,
            cols,
            &EdgeRun {
                wide: row_byte(mask.left_y[2], row),
                mid: row_byte(mask.left_y[1], row),
                narrow: row_byte(mask.left_y[0], row),
                inner: row_byte(mask.int_y, row),
                lfl,
                levels,
            },
        )?;
    }
    for row in 0..8 {
        if origin.mi_row + row >= origin.mi_rows {
            break;
        }
        let cols = origin.mi_cols.saturating_sub(origin.mi_col).min(8);
        let y = (origin.mi_row + row) * 8;
        let lfl = &mask.lfl_y[row * 8..(row + 1) * 8];
        let edge = origin.mi_row + row != 0;
        horizontal_edges(
            plane,
            stride,
            origin.mi_col * 8,
            y,
            cols,
            &EdgeRun {
                wide: if edge {
                    row_byte(mask.above_y[2], row)
                } else {
                    0
                },
                mid: if edge {
                    row_byte(mask.above_y[1], row)
                } else {
                    0
                },
                narrow: if edge {
                    row_byte(mask.above_y[0], row)
                } else {
                    0
                },
                inner: row_byte(mask.int_y, row),
                lfl,
                levels,
            },
        )?;
    }
    Ok(())
}

fn filter_chroma(
    mask: &Mask,
    levels: &Levels,
    plane: &mut [u8],
    stride: usize,
    origin: &Origin,
) -> Result<(), CodecError> {
    for row in 0..4 {
        let mi = origin.mi_row + row * 2;
        if mi >= origin.mi_rows {
            break;
        }
        let cols = origin
            .mi_cols
            .saturating_sub(origin.mi_col)
            .div_ceil(2)
            .min(4);
        let y = origin.mi_row * 4 + row * 8;
        let lfl = chroma_levels(&mask.lfl_y, row);
        vertical_edges(
            plane,
            stride,
            origin.mi_col * 4,
            y,
            cols,
            &EdgeRun {
                wide: uv_nibble(mask.left_uv[2], row),
                mid: uv_nibble(mask.left_uv[1], row),
                narrow: uv_nibble(mask.left_uv[0], row),
                inner: uv_nibble(mask.int_uv, row),
                lfl: &lfl,
                levels,
            },
        )?;
    }
    for row in 0..4 {
        let mi = origin.mi_row + row * 2;
        if mi >= origin.mi_rows {
            break;
        }
        let cols = origin
            .mi_cols
            .saturating_sub(origin.mi_col)
            .div_ceil(2)
            .min(4);
        let y = origin.mi_row * 4 + row * 8;
        let lfl = chroma_levels(&mask.lfl_y, row);
        let edge = mi != 0;
        let inner = if mi + 1 == origin.mi_rows {
            0
        } else {
            uv_nibble(mask.int_uv, row)
        };
        horizontal_edges(
            plane,
            stride,
            origin.mi_col * 4,
            y,
            cols,
            &EdgeRun {
                wide: if edge {
                    uv_nibble(mask.above_uv[2], row)
                } else {
                    0
                },
                mid: if edge {
                    uv_nibble(mask.above_uv[1], row)
                } else {
                    0
                },
                narrow: if edge {
                    uv_nibble(mask.above_uv[0], row)
                } else {
                    0
                },
                inner,
                lfl: &lfl,
                levels,
            },
        )?;
    }
    Ok(())
}

struct EdgeRun<'a> {
    wide: u8,
    mid: u8,
    narrow: u8,
    inner: u8,
    lfl: &'a [u8],
    levels: &'a Levels,
}

fn vertical_edges(
    plane: &mut [u8],
    stride: usize,
    x0: usize,
    y: usize,
    cols: usize,
    run: &EdgeRun<'_>,
) -> Result<(), CodecError> {
    for col in 0..cols.min(run.lfl.len()) {
        let bit = 1u8 << col;
        let x = x0 + col * 8;
        let Some(thr) = run.levels.thresh(run.lfl[col]) else {
            return Err(bad("loop filter level"));
        };
        if run.wide & bit != 0 {
            vert16(plane, stride, x, y, &thr)?;
        } else if run.mid & bit != 0 {
            vert8(plane, stride, x, y, &thr)?;
        } else if run.narrow & bit != 0 {
            vert4(plane, stride, x, y, &thr)?;
        }
        if run.inner & bit != 0 {
            vert4(plane, stride, x + 4, y, &thr)?;
        }
    }
    Ok(())
}

fn horizontal_edges(
    plane: &mut [u8],
    stride: usize,
    x0: usize,
    y: usize,
    cols: usize,
    run: &EdgeRun<'_>,
) -> Result<(), CodecError> {
    for col in 0..cols.min(run.lfl.len()) {
        let bit = 1u8 << col;
        let x = x0 + col * 8;
        let Some(thr) = run.levels.thresh(run.lfl[col]) else {
            return Err(bad("loop filter level"));
        };
        if run.wide & bit != 0 {
            horiz16(plane, stride, x, y, &thr)?;
        } else if run.mid & bit != 0 {
            horiz8(plane, stride, x, y, &thr)?;
        } else if run.narrow & bit != 0 {
            horiz4(plane, stride, x, y, &thr)?;
        }
        if run.inner & bit != 0 {
            horiz4(plane, stride, x, y + 4, &thr)?;
        }
    }
    Ok(())
}

fn row_byte(mask: u64, row: usize) -> u8 {
    u8::try_from((mask >> (row * 8)) & 0xff).unwrap_or(0)
}

fn uv_nibble(mask: u16, row: usize) -> u8 {
    u8::try_from((u32::from(mask) >> (row * 4)) & 0xf).unwrap_or(0)
}

fn chroma_levels(lfl: &[u8; 64], row: usize) -> [u8; 4] {
    std::array::from_fn(|col| lfl.get(row * 16 + col * 2).copied().unwrap_or(0))
}

fn vert4(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    thr: &Thresh,
) -> Result<(), CodecError> {
    for row in 0..8 {
        let at = index(plane, stride, x, y + row, 4, 4)?;
        let p = side(plane, at, 4);
        let q = [plane[at], plane[at + 1], plane[at + 2], plane[at + 3]];
        let mask = filter_mask(thr, &p, &q);
        let mut p1 = p[2];
        let mut p0 = p[3];
        let mut q0 = q[0];
        let mut q1 = q[1];
        filter4(mask, thr.hev, &mut p1, &mut p0, &mut q0, &mut q1);
        plane[at - 2] = p1;
        plane[at - 1] = p0;
        plane[at] = q0;
        plane[at + 1] = q1;
    }
    Ok(())
}

fn vert8(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    thr: &Thresh,
) -> Result<(), CodecError> {
    for row in 0..8 {
        let at = index(plane, stride, x, y + row, 4, 4)?;
        let p = side(plane, at, 4);
        let q = [plane[at], plane[at + 1], plane[at + 2], plane[at + 3]];
        let wide = filter8(thr, &p, &q);
        plane[at - 3] = wide[0];
        plane[at - 2] = wide[1];
        plane[at - 1] = wide[2];
        plane[at] = wide[3];
        plane[at + 1] = wide[4];
        plane[at + 2] = wide[5];
    }
    Ok(())
}

fn vert16(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    thr: &Thresh,
) -> Result<(), CodecError> {
    for row in 0..8 {
        let at = index(plane, stride, x, y + row, 8, 8)?;
        let p = side(plane, at, 8);
        let mut q = [0u8; 8];
        q.copy_from_slice(&plane[at..at + 8]);
        let wide = filter16(thr, &p, &q);
        plane[at - 7..at].copy_from_slice(&wide[..7]);
        plane[at..at + 7].copy_from_slice(&wide[7..]);
    }
    Ok(())
}

fn horiz4(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    thr: &Thresh,
) -> Result<(), CodecError> {
    for col in 0..8 {
        let samples = column(plane, stride, x + col, y, 4, 4)?;
        let mask = filter_mask(thr, &samples[..4], &samples[4..]);
        let mut p1 = samples[2];
        let mut p0 = samples[3];
        let mut q0 = samples[4];
        let mut q1 = samples[5];
        filter4(mask, thr.hev, &mut p1, &mut p0, &mut q0, &mut q1);
        put_column(plane, stride, x + col, y, &[p1, p0, q0, q1], 2)?;
    }
    Ok(())
}

fn horiz8(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    thr: &Thresh,
) -> Result<(), CodecError> {
    for col in 0..8 {
        let samples = column(plane, stride, x + col, y, 4, 4)?;
        let wide = filter8(thr, &samples[..4], &samples[4..]);
        put_column(plane, stride, x + col, y, &wide, 3)?;
    }
    Ok(())
}

fn horiz16(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    thr: &Thresh,
) -> Result<(), CodecError> {
    for col in 0..8 {
        let samples = column(plane, stride, x + col, y, 8, 8)?;
        let wide = filter16(thr, &samples[..8], &samples[8..]);
        put_column(plane, stride, x + col, y, &wide, 7)?;
    }
    Ok(())
}

fn index(
    plane: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    left: usize,
    right: usize,
) -> Result<usize, CodecError> {
    let height = plane.len() / stride.max(1);
    if stride == 0 || x < left || y >= height || x + right > stride {
        return Err(bad("loop filter edge"));
    }
    Ok(y * stride + x)
}

fn side(plane: &[u8], at: usize, n: usize) -> [u8; 8] {
    let mut out = [0u8; 8];
    for i in 0..n {
        out[n - 1 - i] = plane[at - 1 - i];
    }
    out
}

fn column(
    plane: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    above: usize,
    below: usize,
) -> Result<[u8; 16], CodecError> {
    let height = plane.len() / stride.max(1);
    if stride == 0 || y < above || y + below > height || x >= stride {
        return Err(bad("loop filter edge"));
    }
    let mut out = [0u8; 16];
    for (i, slot) in out.iter_mut().enumerate().take(above + below) {
        let row = y - above + i;
        *slot = plane[row * stride + x];
    }
    Ok(out)
}

fn put_column(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    samples: &[u8],
    radius: usize,
) -> Result<(), CodecError> {
    if y < radius || x >= stride {
        return Err(bad("loop filter edge"));
    }
    for (i, sample) in samples.iter().enumerate() {
        let row = y - radius + i;
        let at = row * stride + x;
        let Some(slot) = plane.get_mut(at) else {
            return Err(bad("loop filter edge"));
        };
        *slot = *sample;
    }
    Ok(())
}

fn filter_mask(thr: &Thresh, p: &[u8], q: &[u8]) -> i8 {
    // `p` is p3, p2, p1, p0. `q` is q0, q1, q2, q3.
    let mut reject = false;
    reject |= p[0].abs_diff(p[1]) > thr.lim;
    reject |= p[1].abs_diff(p[2]) > thr.lim;
    reject |= p[2].abs_diff(p[3]) > thr.lim;
    reject |= q[0].abs_diff(q[1]) > thr.lim;
    reject |= q[1].abs_diff(q[2]) > thr.lim;
    reject |= q[2].abs_diff(q[3]) > thr.lim;
    let gap = i32::from(p[3].abs_diff(q[0])) * 2 + i32::from(p[2].abs_diff(q[1]) / 2);
    reject |= gap > i32::from(thr.blimit);
    if reject { 0 } else { -1 }
}

fn flat4(thresh: u8, p: &[u8], q: &[u8]) -> i8 {
    let p0 = p[3];
    let q0 = q[0];
    let mut reject = false;
    reject |= p[2].abs_diff(p0) > thresh;
    reject |= q[1].abs_diff(q0) > thresh;
    reject |= p[1].abs_diff(p0) > thresh;
    reject |= q[2].abs_diff(q0) > thresh;
    reject |= p[0].abs_diff(p0) > thresh;
    reject |= q[3].abs_diff(q0) > thresh;
    if reject { 0 } else { -1 }
}

fn flat5(p4: u8, p: &[u8], q: &[u8], q4: u8) -> i8 {
    if flat4(1, p, q) == 0 || p4.abs_diff(p[3]) > 1 || q4.abs_diff(q[0]) > 1 {
        0
    } else {
        -1
    }
}

fn filter4(mask: i8, hev_thr: u8, p1: &mut u8, p0: &mut u8, q0: &mut u8, q1: &mut u8) {
    let ps1 = i32::from(signed(*p1));
    let ps0 = i32::from(signed(*p0));
    let qs0 = i32::from(signed(*q0));
    let qs1 = i32::from(signed(*q1));
    let hev = i32::from(hev_mask(hev_thr, *p1, *p0, *q0, *q1));
    let mask = i32::from(mask);
    let mut filter = i32::from(clamp_i8(ps1 - qs1)) & hev;
    filter = i32::from(clamp_i8(filter + 3 * (qs0 - ps0))) & mask;
    let filter1 = i32::from(clamp_i8(filter + 4) >> 3);
    let filter2 = i32::from(clamp_i8(filter + 3) >> 3);
    *q0 = unsigned(clamp_i8(qs0 - filter1));
    *p0 = unsigned(clamp_i8(ps0 + filter2));
    let outer = round_pow2(filter1, 1) & !hev;
    *q1 = unsigned(clamp_i8(qs1 - outer));
    *p1 = unsigned(clamp_i8(ps1 + outer));
}

fn filter8(thr: &Thresh, p: &[u8], q: &[u8]) -> [u8; 6] {
    let mask = filter_mask(thr, p, q);
    let flat = flat4(1, p, q);
    if flat != 0 && mask != 0 {
        let (p3, p2, p1, p0) = (
            i32::from(p[0]),
            i32::from(p[1]),
            i32::from(p[2]),
            i32::from(p[3]),
        );
        let (q0, q1, q2, q3) = (
            i32::from(q[0]),
            i32::from(q[1]),
            i32::from(q[2]),
            i32::from(q[3]),
        );
        [
            rnd3(p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0),
            rnd3(p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1),
            rnd3(p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2),
            rnd3(p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3),
            rnd3(p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3),
            rnd3(p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3),
        ]
    } else {
        let p2 = p[1];
        let mut p1 = p[2];
        let mut p0 = p[3];
        let mut q0 = q[0];
        let mut q1 = q[1];
        let q2 = q[2];
        filter4(mask, thr.hev, &mut p1, &mut p0, &mut q0, &mut q1);
        [p2, p1, p0, q0, q1, q2]
    }
}

fn filter16(thr: &Thresh, p: &[u8], q: &[u8]) -> [u8; 14] {
    let inner_p = [p[4], p[5], p[6], p[7]];
    let inner_q = [q[0], q[1], q[2], q[3]];
    let mask = filter_mask(thr, &inner_p, &inner_q);
    let flat = flat4(1, &inner_p, &inner_q);
    let flat2 = flat5(p[3], &inner_p, &inner_q, q[4]);
    if flat2 != 0 && flat != 0 && mask != 0 {
        wide15(p, q)
    } else {
        let short = filter8(thr, &inner_p, &inner_q);
        [
            p[1], p[2], p[3], p[4], short[0], short[1], short[2], short[3], short[4], short[5],
            q[3], q[4], q[5], q[6],
        ]
    }
}

fn wide15(p: &[u8], q: &[u8]) -> [u8; 14] {
    let p: [i32; 8] = std::array::from_fn(|i| i32::from(p[i]));
    let q: [i32; 8] = std::array::from_fn(|i| i32::from(q[i]));
    [
        rnd4(p[0] * 7 + p[1] * 2 + p[2] + p[3] + p[4] + p[5] + p[6] + p[7] + q[0]),
        rnd4(p[0] * 6 + p[1] + p[2] * 2 + p[3] + p[4] + p[5] + p[6] + p[7] + q[0] + q[1]),
        rnd4(p[0] * 5 + p[1] + p[2] + p[3] * 2 + p[4] + p[5] + p[6] + p[7] + q[0] + q[1] + q[2]),
        rnd4(
            p[0] * 4
                + p[1]
                + p[2]
                + p[3]
                + p[4] * 2
                + p[5]
                + p[6]
                + p[7]
                + q[0]
                + q[1]
                + q[2]
                + q[3],
        ),
        rnd4(
            p[0] * 3
                + p[1]
                + p[2]
                + p[3]
                + p[4]
                + p[5] * 2
                + p[6]
                + p[7]
                + q[0]
                + q[1]
                + q[2]
                + q[3]
                + q[4],
        ),
        rnd4(
            p[0] * 2
                + p[1]
                + p[2]
                + p[3]
                + p[4]
                + p[5]
                + p[6] * 2
                + p[7]
                + q[0]
                + q[1]
                + q[2]
                + q[3]
                + q[4]
                + q[5],
        ),
        rnd4(
            p[0] + p[1]
                + p[2]
                + p[3]
                + p[4]
                + p[5]
                + p[6]
                + p[7] * 2
                + q[0]
                + q[1]
                + q[2]
                + q[3]
                + q[4]
                + q[5]
                + q[6],
        ),
        rnd4(
            p[1] + p[2]
                + p[3]
                + p[4]
                + p[5]
                + p[6]
                + p[7]
                + q[0] * 2
                + q[1]
                + q[2]
                + q[3]
                + q[4]
                + q[5]
                + q[6]
                + q[7],
        ),
        rnd4(
            p[2] + p[3]
                + p[4]
                + p[5]
                + p[6]
                + p[7]
                + q[0]
                + q[1] * 2
                + q[2]
                + q[3]
                + q[4]
                + q[5]
                + q[6]
                + q[7] * 2,
        ),
        rnd4(
            p[3] + p[4]
                + p[5]
                + p[6]
                + p[7]
                + q[0]
                + q[1]
                + q[2] * 2
                + q[3]
                + q[4]
                + q[5]
                + q[6]
                + q[7] * 3,
        ),
        rnd4(
            p[4] + p[5]
                + p[6]
                + p[7]
                + q[0]
                + q[1]
                + q[2]
                + q[3] * 2
                + q[4]
                + q[5]
                + q[6]
                + q[7] * 4,
        ),
        rnd4(p[5] + p[6] + p[7] + q[0] + q[1] + q[2] + q[3] + q[4] * 2 + q[5] + q[6] + q[7] * 5),
        rnd4(p[6] + p[7] + q[0] + q[1] + q[2] + q[3] + q[4] + q[5] * 2 + q[6] + q[7] * 6),
        rnd4(p[7] + q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] * 2 + q[7] * 7),
    ]
}

fn hev_mask(thresh: u8, p1: u8, p0: u8, q0: u8, q1: u8) -> i8 {
    if p1.abs_diff(p0) > thresh || q1.abs_diff(q0) > thresh {
        -1
    } else {
        0
    }
}

fn signed(sample: u8) -> i8 {
    i8::from_ne_bytes([(sample ^ 0x80)])
}

fn unsigned(sample: i8) -> u8 {
    sample.cast_unsigned() ^ 0x80
}

fn clamp_i8(value: i32) -> i8 {
    i8::try_from(value.clamp(-128, 127)).unwrap_or(0)
}

fn round_pow2(value: i32, n: u32) -> i32 {
    (value + (1 << (n - 1))) >> n
}

fn rnd3(value: i32) -> u8 {
    u8::try_from(round_pow2(value, 3).clamp(0, 255)).unwrap_or(0)
}

fn rnd4(value: i32) -> u8 {
    u8::try_from(round_pow2(value, 4).clamp(0, 255)).unwrap_or(0)
}

fn bad(what: &'static str) -> CodecError {
    CodecError::Malformed { at_bit: 0, what }
}

//! Intra prediction from section 7.11.2 of the AV1 specification.

use cutlass_decode::CodecError;

use crate::dims::{D67_PRED, DC_PRED, PAETH_PRED, SMOOTH_PRED, V_PRED};
use crate::tables::{
    DR_INTRA_DERIVATIVE, INTRA_EDGE_KERNEL, MODE_TO_ANGLE, SM_WEIGHTS_TX_4X4, SM_WEIGHTS_TX_8X8,
    SM_WEIGHTS_TX_16X16, SM_WEIGHTS_TX_32X32, SM_WEIGHTS_TX_64X64,
};

const OFFSET: i32 = 2;

pub struct IntraPred {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    pub have_left: bool,
    pub have_above: bool,
    pub have_above_right: bool,
    pub have_below_left: bool,
    pub mode: u8,
    pub angle_delta: i32,
    pub edge_filter: bool,
    pub filter_type: bool,
    pub max_x: usize,
    pub max_y: usize,
    /// Recursive intra filter. `None` uses the ordinary intra mode.
    pub filter: Option<u8>,
}

/// Luma samples and the chroma block they correct.
pub struct ChromaFromLuma<'a> {
    pub luma: &'a [u8],
    pub luma_stride: usize,
    pub chroma: &'a mut [u8],
    pub chroma_stride: usize,
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    pub sub_x: bool,
    pub sub_y: bool,
    pub alpha: i32,
    pub max_luma_w: usize,
    pub max_luma_h: usize,
}

/// Buffers reused across intra blocks. A fresh buffer per block dominated decode time.
pub struct PredictScratch {
    pred: Vec<u8>,
    above: Vec<i32>,
    left: Vec<i32>,
    ac: Vec<i32>,
    dup: Vec<i32>,
    edge: Vec<i32>,
}

impl PredictScratch {
    pub fn new() -> Self {
        Self {
            pred: Vec::new(),
            above: Vec::new(),
            left: Vec::new(),
            ac: Vec::new(),
            dup: Vec::new(),
            edge: Vec::new(),
        }
    }

    pub fn ac(&mut self) -> &mut Vec<i32> {
        &mut self.ac
    }
}

impl Default for PredictScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Add the luma high-frequency residual to a DC chroma prediction.
pub fn chroma_from_luma(job: ChromaFromLuma<'_>, ac: &mut Vec<i32>) -> Result<(), CodecError> {
    let ChromaFromLuma {
        luma,
        luma_stride,
        chroma,
        chroma_stride,
        x: start_x,
        y: start_y,
        w,
        h,
        sub_x,
        sub_y,
        alpha,
        max_luma_w,
        max_luma_h,
    } = job;
    let sx = usize::from(sub_x);
    let sy = usize::from(sub_y);
    let shift = 3 - sx - sy;
    let mut average = 0i32;
    let n = w.saturating_mul(h);
    if ac.len() < n {
        ac.resize(n, 0);
    }
    let ac = &mut ac[..n];
    for i in 0..h {
        let luma_y = ((start_y + i) << sy).min(max_luma_h.saturating_sub(1 << sy));
        for j in 0..w {
            let luma_x = ((start_x + j) << sx).min(max_luma_w.saturating_sub(1 << sx));
            let mut total = 0i32;
            for dy in 0..=sy {
                for dx in 0..=sx {
                    let at = (luma_y + dy) * luma_stride + luma_x + dx;
                    let Some(sample) = luma.get(at) else {
                        return Err(CodecError::Malformed {
                            at_bit: 0,
                            what: "chroma from luma",
                        });
                    };
                    total += i32::from(*sample);
                }
            }
            let value = total << shift;
            ac[i * w + j] = value;
            average += value;
        }
    }
    let log = w.trailing_zeros() + h.trailing_zeros();
    average = round2_signed(average, log);
    for i in 0..h {
        for j in 0..w {
            let at = (start_y + i) * chroma_stride + start_x + j;
            let Some(dc) = chroma.get(at).copied() else {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "chroma from luma",
                });
            };
            let scaled = round2_signed(alpha * (ac[i * w + j] - average), 6);
            let Some(slot) = chroma.get_mut(at) else {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "chroma from luma",
                });
            };
            *slot = clip1(i32::from(dc) + scaled);
        }
    }
    Ok(())
}

pub fn predict_intra(
    plane: &mut [u8],
    stride: usize,
    block: &IntraPred,
    scratch: &mut PredictScratch,
) -> Result<(), CodecError> {
    let span = 4 * (block.w + block.h) + 8;
    let edge_len = span + OFFSET as usize;
    let pred_len = block.w.saturating_mul(block.h);
    grow_i32(&mut scratch.above, edge_len);
    grow_i32(&mut scratch.left, edge_len);
    grow_u8(&mut scratch.pred, pred_len);
    scratch.above[..edge_len].fill(0);
    scratch.left[..edge_len].fill(0);
    let PredictScratch {
        pred,
        above,
        left,
        dup,
        edge,
        ac: _,
    } = scratch;
    fill_edges(
        plane,
        stride,
        block,
        &mut above[..edge_len],
        &mut left[..edge_len],
    );
    if let Some(mode) = block.filter {
        filter_intra_pred(
            &above[..edge_len],
            &left[..edge_len],
            block,
            mode,
            &mut pred[..pred_len],
        )?;
    } else if directional(block.mode) {
        directional_pred(
            &mut above[..edge_len],
            &mut left[..edge_len],
            block,
            &mut pred[..pred_len],
            dup,
            edge,
        )?;
    } else if block.mode >= SMOOTH_PRED && block.mode < PAETH_PRED {
        smooth_pred(
            &above[..edge_len],
            &left[..edge_len],
            block,
            &mut pred[..pred_len],
        );
    } else if block.mode == DC_PRED {
        dc_pred(
            &above[..edge_len],
            &left[..edge_len],
            block,
            &mut pred[..pred_len],
        );
    } else {
        paeth_pred(
            &above[..edge_len],
            &left[..edge_len],
            block,
            &mut pred[..pred_len],
        );
    }
    for row in 0..block.h {
        for col in 0..block.w {
            let at = (block.y + row) * stride + block.x + col;
            // A coded block may extend past the visible picture. Those samples
            // are not stored.
            let Some(slot) = plane.get_mut(at) else {
                continue;
            };
            *slot = pred[row * block.w + col];
        }
    }
    Ok(())
}

fn grow_i32(buf: &mut Vec<i32>, n: usize) {
    if buf.len() < n {
        buf.resize(n, 0);
    }
}

fn grow_u8(buf: &mut Vec<u8>, n: usize) {
    if buf.len() < n {
        buf.resize(n, 0);
    }
}

/// `Intra_Filter_Taps` from the specification. Five modes, eight outputs, seven taps.
const FILTER_TAPS: [[[i8; 7]; 8]; 5] = [
    [
        [-6, 10, 0, 0, 0, 12, 0],
        [-5, 2, 10, 0, 0, 9, 0],
        [-3, 1, 1, 10, 0, 7, 0],
        [-3, 1, 1, 2, 10, 5, 0],
        [-4, 6, 0, 0, 0, 2, 12],
        [-3, 2, 6, 0, 0, 2, 9],
        [-3, 2, 2, 6, 0, 2, 7],
        [-3, 1, 2, 2, 6, 3, 5],
    ],
    [
        [-10, 16, 0, 0, 0, 10, 0],
        [-6, 0, 16, 0, 0, 6, 0],
        [-4, 0, 0, 16, 0, 4, 0],
        [-2, 0, 0, 0, 16, 2, 0],
        [-10, 16, 0, 0, 0, 0, 10],
        [-6, 0, 16, 0, 0, 0, 6],
        [-4, 0, 0, 16, 0, 0, 4],
        [-2, 0, 0, 0, 16, 0, 2],
    ],
    [
        [-8, 8, 0, 0, 0, 16, 0],
        [-8, 0, 8, 0, 0, 16, 0],
        [-8, 0, 0, 8, 0, 16, 0],
        [-8, 0, 0, 0, 8, 16, 0],
        [-4, 4, 0, 0, 0, 0, 16],
        [-4, 0, 4, 0, 0, 0, 16],
        [-4, 0, 0, 4, 0, 0, 16],
        [-4, 0, 0, 0, 4, 0, 16],
    ],
    [
        [-2, 8, 0, 0, 0, 10, 0],
        [-1, 3, 8, 0, 0, 6, 0],
        [-1, 2, 3, 8, 0, 4, 0],
        [0, 1, 2, 3, 8, 2, 0],
        [-1, 4, 0, 0, 0, 3, 10],
        [-1, 3, 4, 0, 0, 4, 6],
        [-1, 2, 3, 4, 0, 4, 4],
        [-1, 2, 2, 3, 4, 3, 3],
    ],
    [
        [-12, 14, 0, 0, 0, 14, 0],
        [-10, 0, 14, 0, 0, 12, 0],
        [-9, 0, 0, 14, 0, 11, 0],
        [-8, 0, 0, 0, 14, 10, 0],
        [-10, 12, 0, 0, 0, 0, 14],
        [-9, 1, 12, 0, 0, 0, 12],
        [-8, 0, 0, 12, 0, 1, 11],
        [-7, 0, 0, 1, 12, 1, 9],
    ],
];

fn filter_intra_pred(
    above: &[i32],
    left: &[i32],
    block: &IntraPred,
    mode: u8,
    pred: &mut [u8],
) -> Result<(), CodecError> {
    let Some(taps) = FILTER_TAPS.get(usize::from(mode)) else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "filter intra",
        });
    };
    let w4 = block.w >> 2;
    let h2 = block.h >> 1;
    for i2 in 0..h2 {
        for j4 in 0..w4 {
            let mut samples = [0i32; 7];
            for (i, sample) in samples.iter_mut().enumerate() {
                *sample = if i < 5 {
                    if i2 == 0 {
                        above[index((j4 << 2) as i32 + i as i32 - 1)]
                    } else if j4 == 0 && i == 0 {
                        left[index((i2 << 1) as i32 - 1)]
                    } else {
                        i32::from(pred[((i2 << 1) - 1) * block.w + (j4 << 2) + i - 1])
                    }
                } else if j4 == 0 {
                    left[index((i2 << 1) as i32 + i as i32 - 5)]
                } else {
                    i32::from(pred[((i2 << 1) + i - 5) * block.w + (j4 << 2) - 1])
                };
            }
            for i1 in 0..2 {
                for j1 in 0..4 {
                    let mut sum = 0i32;
                    let weights = taps[(i1 << 2) + j1];
                    for (weight, sample) in weights.iter().zip(samples) {
                        sum += i32::from(*weight) * sample;
                    }
                    pred[((i2 << 1) + i1) * block.w + (j4 << 2) + j1] =
                        clip1(round2_signed(sum, 4));
                }
            }
        }
    }
    Ok(())
}

fn directional(mode: u8) -> bool {
    (V_PRED..=D67_PRED).contains(&mode)
}

fn fill_edges(plane: &[u8], stride: usize, block: &IntraPred, above: &mut [i32], left: &mut [i32]) {
    let x = block.x as i32;
    let y = block.y as i32;
    if !block.have_above && block.have_left {
        let sample = i32::from(at(plane, stride, x - 1, y));
        fill(above, 0, block.w + block.h, sample);
    } else if !block.have_above {
        fill(above, 0, block.w + block.h, 127);
    } else {
        let limit = block.max_x.min(
            block.x
                + if block.have_above_right {
                    block.w * 2
                } else {
                    block.w
                }
                - 1,
        );
        for i in 0..(block.w + block.h) {
            above[index(i as i32)] =
                i32::from(at(plane, stride, (block.x + i).min(limit) as i32, y - 1));
        }
    }
    if !block.have_left && block.have_above {
        let sample = i32::from(at(plane, stride, x, y - 1));
        fill(left, 0, block.w + block.h, sample);
    } else if !block.have_left {
        fill(left, 0, block.w + block.h, 129);
    } else {
        let limit = block.max_y.min(
            block.y
                + if block.have_below_left {
                    block.h * 2
                } else {
                    block.h
                }
                - 1,
        );
        for i in 0..(block.w + block.h) {
            left[index(i as i32)] =
                i32::from(at(plane, stride, x - 1, (block.y + i).min(limit) as i32));
        }
    }
    let corner = if block.have_above && block.have_left {
        i32::from(at(plane, stride, x - 1, y - 1))
    } else if block.have_above {
        i32::from(at(plane, stride, x, y - 1))
    } else if block.have_left {
        i32::from(at(plane, stride, x - 1, y))
    } else {
        128
    };
    above[index(-1)] = corner;
    left[index(-1)] = corner;
}

fn at(plane: &[u8], stride: usize, x: i32, y: i32) -> u8 {
    plane[(y as usize) * stride + x as usize]
}

fn index(i: i32) -> usize {
    (i + OFFSET) as usize
}

fn fill(buf: &mut [i32], from: i32, count: usize, sample: i32) {
    for i in from..(from + count as i32) {
        buf[index(i)] = sample;
    }
}

fn dc_pred(above: &[i32], left: &[i32], block: &IntraPred, pred: &mut [u8]) {
    let avg = if block.have_left && block.have_above {
        let mut sum = 0i32;
        for k in 0..block.h {
            sum += left[index(k as i32)];
        }
        for k in 0..block.w {
            sum += above[index(k as i32)];
        }
        sum += ((block.w + block.h) >> 1) as i32;
        sum / (block.w + block.h) as i32
    } else if block.have_left {
        let mut sum = 0i32;
        for k in 0..block.h {
            sum += left[index(k as i32)];
        }
        (sum + ((block.h >> 1) as i32)) >> block.h.trailing_zeros()
    } else if block.have_above {
        let mut sum = 0i32;
        for k in 0..block.w {
            sum += above[index(k as i32)];
        }
        (sum + ((block.w >> 1) as i32)) >> block.w.trailing_zeros()
    } else {
        128
    };
    pred.fill(clip1(avg));
}

fn smooth_pred(above: &[i32], left: &[i32], block: &IntraPred, pred: &mut [u8]) {
    let weights_x = weights(block.w);
    let weights_y = weights(block.h);
    for row in 0..block.h {
        for col in 0..block.w {
            let value = if block.mode == SMOOTH_PRED {
                round2_i32(
                    weights_y[row] as i32 * above[index(col as i32)]
                        + (256 - weights_y[row] as i32) * left[index(block.h as i32 - 1)]
                        + weights_x[col] as i32 * left[index(row as i32)]
                        + (256 - weights_x[col] as i32) * above[index(block.w as i32 - 1)],
                    9,
                )
            } else if block.mode == SMOOTH_PRED + 1 {
                round2_i32(
                    weights_y[row] as i32 * above[index(col as i32)]
                        + (256 - weights_y[row] as i32) * left[index(block.h as i32 - 1)],
                    8,
                )
            } else {
                round2_i32(
                    weights_x[col] as i32 * left[index(row as i32)]
                        + (256 - weights_x[col] as i32) * above[index(block.w as i32 - 1)],
                    8,
                )
            };
            pred[row * block.w + col] = clip1(value);
        }
    }
}

fn weights(size: usize) -> &'static [u16] {
    match size {
        4 => &SM_WEIGHTS_TX_4X4,
        8 => &SM_WEIGHTS_TX_8X8,
        16 => &SM_WEIGHTS_TX_16X16,
        32 => &SM_WEIGHTS_TX_32X32,
        _ => &SM_WEIGHTS_TX_64X64,
    }
}

fn paeth_pred(above: &[i32], left: &[i32], block: &IntraPred, pred: &mut [u8]) {
    for row in 0..block.h {
        for col in 0..block.w {
            let top = above[index(col as i32)];
            let side = left[index(row as i32)];
            let corner = above[index(-1)];
            let base = top + side - corner;
            let left_cost = (base - side).abs();
            let top_cost = (base - top).abs();
            let corner_cost = (base - corner).abs();
            let sample = if left_cost <= top_cost && left_cost <= corner_cost {
                side
            } else if top_cost <= corner_cost {
                top
            } else {
                corner
            };
            pred[row * block.w + col] = clip1(sample);
        }
    }
}

fn directional_pred(
    above: &mut [i32],
    left: &mut [i32],
    block: &IntraPred,
    pred: &mut [u8],
    dup: &mut Vec<i32>,
    edge: &mut Vec<i32>,
) -> Result<(), CodecError> {
    let angle = i32::from(MODE_TO_ANGLE[block.mode as usize]) + block.angle_delta * 3;
    let mut upsample_above = 0i32;
    let mut upsample_left = 0i32;
    if block.edge_filter && angle != 90 && angle != 180 {
        filter_edges(above, left, block, angle, edge);
        upsample_above = upsample_edge(above, block, angle - 90, 0, dup);
        upsample_left = upsample_edge(left, block, angle - 180, 1, dup);
    }
    let dx = derivative(angle, true)?;
    let dy = derivative(angle, false)?;
    for row in 0..block.h {
        for col in 0..block.w {
            pred[row * block.w + col] = directional_sample(DirSample {
                above,
                left,
                block,
                angle,
                dx,
                dy,
                upsample_above,
                upsample_left,
                row,
                col,
            });
        }
    }
    Ok(())
}

fn derivative(angle: i32, horizontal: bool) -> Result<i32, CodecError> {
    let index = if horizontal {
        if angle < 90 {
            angle
        } else if (91..180).contains(&angle) {
            180 - angle
        } else {
            return Ok(0);
        }
    } else if (91..180).contains(&angle) {
        angle - 90
    } else if angle > 180 {
        270 - angle
    } else {
        return Ok(0);
    };
    DR_INTRA_DERIVATIVE
        .get(index as usize)
        .map(|value| i32::from(*value))
        .ok_or(CodecError::Malformed {
            at_bit: 0,
            what: "intra angle",
        })
}

struct DirSample<'a> {
    above: &'a [i32],
    left: &'a [i32],
    block: &'a IntraPred,
    angle: i32,
    dx: i32,
    dy: i32,
    upsample_above: i32,
    upsample_left: i32,
    row: usize,
    col: usize,
}

fn directional_sample(sample: DirSample<'_>) -> u8 {
    let DirSample {
        above,
        left,
        block,
        angle,
        dx,
        dy,
        upsample_above,
        upsample_left,
        row,
        col,
    } = sample;
    if angle < 90 {
        let idx = (row as i32 + 1) * dx;
        let base = (idx >> (6 - upsample_above)) + ((col as i32) << upsample_above);
        let shift = ((idx << upsample_above) >> 1) & 0x1f;
        let max_base = ((block.w + block.h - 1) as i32) << upsample_above;
        if base < max_base {
            clip1(round2_i32(
                above[index(base)] * (32 - shift) + above[index(base + 1)] * shift,
                5,
            ))
        } else {
            clip1(above[index(max_base)])
        }
    } else if (91..180).contains(&angle) {
        let idx = ((col as i32) << 6) - (row as i32 + 1) * dx;
        let base = idx >> (6 - upsample_above);
        if base >= -(1 << upsample_above) {
            let shift = ((idx << upsample_above) >> 1) & 0x1f;
            clip1(round2_i32(
                above[index(base)] * (32 - shift) + above[index(base + 1)] * shift,
                5,
            ))
        } else {
            let idx = ((row as i32) << 6) - (col as i32 + 1) * dy;
            let base = idx >> (6 - upsample_left);
            let shift = ((idx << upsample_left) >> 1) & 0x1f;
            clip1(round2_i32(
                left[index(base)] * (32 - shift) + left[index(base + 1)] * shift,
                5,
            ))
        }
    } else if angle > 180 {
        let idx = (col as i32 + 1) * dy;
        let base = (idx >> (6 - upsample_left)) + ((row as i32) << upsample_left);
        let shift = ((idx << upsample_left) >> 1) & 0x1f;
        clip1(round2_i32(
            left[index(base)] * (32 - shift) + left[index(base + 1)] * shift,
            5,
        ))
    } else if angle == 90 {
        clip1(above[index(col as i32)])
    } else {
        clip1(left[index(row as i32)])
    }
}

fn filter_edges(
    above: &mut [i32],
    left: &mut [i32],
    block: &IntraPred,
    angle: i32,
    edge: &mut Vec<i32>,
) {
    if (91..180).contains(&angle) && block.w + block.h >= 24 {
        let filtered = round2_i32(
            left[index(0)] * 5 + above[index(-1)] * 6 + above[index(0)] * 5,
            4,
        );
        above[index(-1)] = filtered;
        left[index(-1)] = filtered;
    }
    if block.have_above {
        let strength = edge_strength(block.w, block.h, block.filter_type, (angle - 90).abs());
        let num = block.w.min(block.max_x - block.x + 1) + if angle < 90 { block.h } else { 0 } + 1;
        filter_edge(above, num, strength, edge);
    }
    if block.have_left {
        let strength = edge_strength(block.w, block.h, block.filter_type, (angle - 180).abs());
        let num =
            block.h.min(block.max_y - block.y + 1) + if angle > 180 { block.w } else { 0 } + 1;
        filter_edge(left, num, strength, edge);
    }
}

fn upsample_edge(
    buf: &mut [i32],
    block: &IntraPred,
    delta: i32,
    dir: i32,
    dup: &mut Vec<i32>,
) -> i32 {
    let use_upsample = edge_upsample(block.w, block.h, block.filter_type, delta.abs());
    if use_upsample == 0 {
        return 0;
    }
    let num = if dir == 0 {
        block.w + if delta < 0 { block.h } else { 0 }
    } else {
        block.h + if delta > 0 { block.w } else { 0 }
    };
    let need = num + 3;
    grow_i32(dup, need);
    let dup = &mut dup[..need];
    dup[0] = buf[index(-1)];
    for i in -1..num as i32 {
        dup[(i + 2) as usize] = buf[index(i)];
    }
    dup[num + 2] = buf[index(num as i32 - 1)];
    buf[index(-2)] = dup[0];
    for i in 0..num {
        let s = -dup[i] + 9 * dup[i + 1] + 9 * dup[i + 2] - dup[i + 3];
        buf[index(2 * i as i32 - 1)] = i32::from(clip1(round2_i32(s, 4)));
        buf[index(2 * i as i32)] = dup[i + 2];
    }
    1
}

fn filter_edge(buf: &mut [i32], size: usize, strength: usize, edge: &mut Vec<i32>) {
    if strength == 0 || size == 0 {
        return;
    }
    grow_i32(edge, size);
    let edge = &mut edge[..size];
    for (i, slot) in edge.iter_mut().enumerate() {
        *slot = buf[index(i as i32 - 1)];
    }
    for i in 1..size {
        let mut sum = 0i32;
        for j in 0..5 {
            let k = (i as i32 - 2 + j as i32).clamp(0, size as i32 - 1) as usize;
            sum += i32::from(INTRA_EDGE_KERNEL[(strength - 1) * 5 + j]) * edge[k];
        }
        buf[index(i as i32 - 1)] = (sum + 8) >> 4;
    }
}

fn edge_strength(w: usize, h: usize, smooth: bool, delta: i32) -> usize {
    let wh = w + h;
    if !smooth {
        plain_strength(wh, delta)
    } else {
        smooth_strength(wh, delta)
    }
}

fn plain_strength(wh: usize, delta: i32) -> usize {
    if wh <= 8 {
        usize::from(delta >= 56)
    } else if wh <= 16 {
        usize::from(delta >= 40)
    } else if wh <= 24 {
        usize::from(delta >= 8) + usize::from(delta >= 16) + usize::from(delta >= 32)
    } else if wh <= 32 {
        1 + usize::from(delta >= 4) + usize::from(delta >= 32)
    } else {
        3
    }
}

fn smooth_strength(wh: usize, delta: i32) -> usize {
    if wh <= 8 {
        usize::from(delta >= 40) + usize::from(delta >= 64)
    } else if wh <= 16 {
        usize::from(delta >= 20) + usize::from(delta >= 48)
    } else if wh <= 24 {
        if delta >= 4 { 3 } else { 0 }
    } else {
        3
    }
}

fn edge_upsample(w: usize, h: usize, smooth: bool, delta: i32) -> i32 {
    if delta <= 0 || delta >= 40 {
        0
    } else if !smooth {
        i32::from(w + h <= 16)
    } else {
        i32::from(w + h <= 8)
    }
}

fn round2_signed(x: i32, n: u32) -> i32 {
    if x >= 0 {
        round2_i32(x, n)
    } else {
        -round2_i32(-x, n)
    }
}

fn round2_i32(x: i32, n: u32) -> i32 {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

fn clip1(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

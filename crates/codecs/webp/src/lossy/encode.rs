//! The VP8 key-frame encoder.
//!
//! The picture goes to 4:2:0 YUV, its macroblocks are grouped into
//! segments by how much quantization they can take, and each macroblock
//! picks its prediction by rate and distortion: four 16x16 luma modes
//! against ten modes for each 4x4 sub-block, and four chroma modes. The
//! token probabilities are then fitted to the coefficients, and the loop
//! filter strength set per segment. Settings follow libwebp's so that a
//! quality means about what it means to `cwebp -q`.

use cutlass_decode::CodecError;

use super::analysis::{SegmentShape, analyze};
use super::bool_encoder::{BoolEncoder, cost};
use super::dsp::{add_idct, inverse_wht};
use super::forward::{fdct, fwht, sse, texture_distortion};
use super::frame::{CHROMA_PITCH, LUMA_PITCH, decode, load_edges, loop_filter, store};
use super::header::{CoeffProbs, FilterKind, FilterStrength, Quant, filter_strength};
use super::predict::{self, B_DC, B_HE, B_TM, B_VE, Edges};
use super::source::Source;
use super::tables::{AC_Q, B_MODE_TREE, DEFAULT_COEFF_PROBS, KF_B_MODE_PROBS, ZIGZAG};
use super::tokens::{self, Cost, Stats};
use super::yuv::{Planes, to_rgba};
use crate::alpha::{self, Filter};
use crate::lossless::encode::{Config as LosslessConfig, encode_stream};

/// Settings of one lossy encode.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Config {
    /// 0 to 100, as `cwebp -q`.
    pub quality: u8,
    /// 0 to 6, as `cwebp -m`: higher tries more.
    pub method: u8,
    /// Keep the RGB under fully transparent pixels.
    pub exact: bool,
}

/// libwebp's spatial noise shaping strength, `cwebp -sns`'s default.
const SNS: i32 = 50;
/// `cwebp -f`'s default.
const FILTER_STRENGTH: i32 = 60;
const SEGMENTS: usize = 4;
/// Penalty, per block, on a flat area predicted by a mode other than DC.
const FLATNESS_PENALTY: u32 = 140;
/// Most AC levels a 4x4 luma block, or the chroma blocks, may have to
/// count as flat.
const FLATNESS_LIMIT_I4: usize = 3;
const FLATNESS_LIMIT_UV: usize = 2;
/// Distortion weight against rate in a score.
const RD_DISTO_MULT: u64 = 256;
/// Pictures of at most this many macroblocks (a 128x128 icon) search the
/// qualities for a frame that dominates.
const SMALL_PICTURE: usize = 64;
/// Fewest macroblocks between two refreshes of the costs.
const REFRESH_MIN: usize = 96;
/// Largest first partition: its size has 19 bits.
const MAX_FIRST_PARTITION: usize = (1 << 19) - 1;

const QFIX: u32 = 17;
/// Rounding biases of the quantizer, in 1/256ths, `[DC, AC]` for y1, y2, uv.
const BIAS: [[u32; 2]; 3] = [[96, 110], [96, 108], [110, 115]];
/// Extra magnitude for high luma frequencies, raster order.
const FREQ_SHARPENING: [u32; 16] = [
    0, 30, 60, 90, 30, 60, 90, 90, 60, 90, 90, 90, 90, 90, 90, 90,
];
const MAX_LEVEL: u32 = 2047;
/// Above this quality chroma DC errors are not diffused.
const ERROR_DIFFUSION_QUALITY: u8 = 98;

/// Quantize a DC alone, as libwebp's error diffusion does, and return half
/// the error.
fn quantize_dc(value: &mut i16, m: &Matrix) -> i32 {
    let v = i32::from(*value);
    let magnitude = v.unsigned_abs();
    if magnitude > m.zthresh[0] {
        let q = (((magnitude * m.iq[0] + m.bias[0]) >> QFIX) as i32) * m.q[0];
        let error = magnitude as i32 - q;
        *value = (if v < 0 { -q } else { q }) as i16;
        (if v < 0 { -error } else { error }) >> 1
    } else {
        *value = 0;
        v >> 1
    }
}

/// libwebp's `CorrectDCValues`: push part of each chroma DC error into the
/// next block's DC before quantizing it. Returns the three errors kept.
fn diffuse_dc(blocks: &mut [[i16; 16]], m: &Matrix, top: [i32; 2], left: [i32; 2]) -> [i32; 3] {
    const C1: i32 = 7;
    const C2: i32 = 8;
    let add = |block: &mut [i16; 16], below: i32, right: i32| {
        block[0] = (i32::from(block[0]) + ((C1 * below + C2 * right) >> 3)) as i16;
        quantize_dc(&mut block[0], m)
    };
    let e0 = add(&mut blocks[0], top[0], left[0]);
    let e1 = add(&mut blocks[1], top[1], e0);
    let e2 = add(&mut blocks[2], e0, left[1]);
    let e3 = add(&mut blocks[3], e1, e2);
    [e1, e2, e3]
}

/// One block type's quantizer, raster order.
#[derive(Clone, Copy)]
struct Matrix {
    q: [i32; 16],
    iq: [u32; 16],
    bias: [u32; 16],
    zthresh: [u32; 16],
    sharpen: [u32; 16],
}

impl Matrix {
    fn new(steps: [i32; 2], kind: usize) -> Self {
        let mut m = Self {
            q: [0; 16],
            iq: [0; 16],
            bias: [0; 16],
            zthresh: [0; 16],
            sharpen: [0; 16],
        };
        for (i, &sharpening) in FREQ_SHARPENING.iter().enumerate() {
            let ac = usize::from(i > 0);
            let q = steps[ac].max(1);
            m.q[i] = q;
            m.iq[i] = (1 << QFIX) / q as u32;
            m.bias[i] = BIAS[kind][ac] << (QFIX - 8);
            m.zthresh[i] = ((1 << QFIX) - 1 - m.bias[i]) / m.iq[i];
            if kind == 0 {
                m.sharpen[i] = (sharpening * q as u32) >> 11;
            }
        }
        m
    }

    /// Average step, libwebp's `ExpandMatrix` result.
    fn average(&self) -> i32 {
        (self.q.iter().sum::<i32>() + 8) >> 4
    }

    /// Quantize `coeffs` (raster) from scan position `first` into `levels`
    /// (scan order), leaving `coeffs` dequantized as the decoder sees them.
    /// Returns whether any level is not zero.
    fn quantize(&self, coeffs: &mut [i16; 16], levels: &mut [i16; 16], first: usize) -> bool {
        let mut any = false;
        levels[..first].fill(0);
        for n in first..16 {
            let j = ZIGZAG[n];
            let negative = coeffs[j] < 0;
            let magnitude = u32::from(coeffs[j].unsigned_abs()) + self.sharpen[j];
            if magnitude > self.zthresh[j] {
                let level = ((magnitude * self.iq[j] + self.bias[j]) >> QFIX).min(MAX_LEVEL) as i32;
                let level = if negative { -level } else { level };
                levels[n] = level as i16;
                // Stored in 16 bits, as the decoder does.
                coeffs[j] = (level * self.q[j]) as i16;
                any |= level != 0;
            } else {
                levels[n] = 0;
                coeffs[j] = 0;
            }
        }
        any
    }
}

/// Everything one segment codes with.
struct Segment {
    quant_index: i32,
    filter_level: i32,
    y1: Matrix,
    y2: Matrix,
    uv: Matrix,
    lambda_i16: u64,
    lambda_i4: u64,
    lambda_uv: u64,
    lambda_mode: u64,
    tlambda: u64,
}

/// libwebp's `QualityToCompression`.
fn quality_to_compression(quality: f64) -> f64 {
    let linear = if quality < 0.75 {
        quality * (2.0 / 3.0)
    } else {
        2.0 * quality - 1.0
    };
    linear.powf(1.0 / 3.0)
}

fn segments(
    shapes: &[SegmentShape],
    uv_alpha: i32,
    quality: u8,
    method: u8,
) -> (Vec<Segment>, [i32; 5]) {
    let amp = 0.9 * f64::from(SNS) / 100.0 / 128.0;
    let c_base = quality_to_compression(f64::from(quality) / 100.0);
    let dq_uv_ac = ((uv_alpha - 64) * (6 - -4) / (100 - 30) * SNS / 100).clamp(-4, 6);
    let dq_uv_dc = (-4 * SNS / 100).clamp(-15, 15);
    let deltas = [0, 0, 0, dq_uv_dc, dq_uv_ac];
    let tlambda_scale = if method >= 4 { SNS as u64 } else { 0 };
    let list = shapes
        .iter()
        .map(|shape| {
            let expn = 1.0 - amp * f64::from(shape.alpha);
            let c = c_base.powf(expn);
            let quant_index = ((127.0 * (1.0 - c)) as i32).clamp(0, 127);
            let quant = Quant::new(quant_index, deltas);
            let y1 = Matrix::new(quant.y1, 0);
            let y2 = Matrix::new(quant.y2, 1);
            let uv = Matrix::new(quant.uv, 2);
            let q_i4 = y1.average() as u64;
            let q_i16 = y2.average() as u64;
            let q_uv = uv.average() as u64;
            // Filter strength from the AC step, lowered for busy segments.
            let qstep = i32::from(AC_Q[quant_index as usize]) >> 2;
            let base = qstep.min(63);
            let f = base * 5 * FILTER_STRENGTH / (256 + shape.beta);
            let filter_level = if f < 2 { 0 } else { f.min(63) };
            Segment {
                quant_index,
                filter_level,
                y1,
                y2,
                uv,
                lambda_i16: 3 * q_i16 * q_i16,
                lambda_i4: (3 * q_i4 * q_i4) >> 7,
                lambda_uv: (3 * q_uv * q_uv) >> 6,
                lambda_mode: (q_i4 * q_i4) >> 7,
                tlambda: (tlambda_scale * q_i4) >> 5,
            }
        })
        .collect();
    (list, deltas)
}

/// Which neighbouring blocks had a non-zero level.
#[derive(Clone, Copy, Default)]
struct NonZero {
    y: [bool; 4],
    u: [bool; 2],
    v: [bool; 2],
    y2: bool,
}

/// One coded macroblock.
#[derive(Clone)]
struct Macroblock {
    segment: u8,
    is_4x4: bool,
    luma: [u8; 16],
    chroma: u8,
    /// Sixteen luma blocks, four U, four V, then Y2, in scan order.
    levels: [[i16; 16]; 25],
    /// No level is set.
    empty: bool,
}

fn i16_mode_cost(mode: u8) -> u32 {
    cost(true, 145)
        + match mode {
            B_TM => cost(true, 156) + cost(true, 128),
            B_HE => cost(true, 156) + cost(false, 128),
            B_VE => cost(false, 156) + cost(true, 163),
            _ => cost(false, 156) + cost(false, 163),
        }
}

fn uv_mode_cost(mode: u8) -> u32 {
    match mode {
        B_DC => cost(false, 142),
        B_VE => cost(true, 142) + cost(false, 114),
        B_TM => cost(true, 142) + cost(true, 114) + cost(true, 183),
        _ => cost(true, 142) + cost(true, 114) + cost(false, 183),
    }
}

/// Bits of every sub-block mode for each above and left mode.
fn sub_mode_costs() -> Vec<[[u32; 10]; 10]> {
    let mut out = vec![[[0u32; 10]; 10]; 10];
    for (above, row) in out.iter_mut().enumerate() {
        for (left, costs) in row.iter_mut().enumerate() {
            for (mode, slot) in costs.iter_mut().enumerate() {
                *slot = super::bool_encoder::tree_path(&B_MODE_TREE, mode)
                    .iter()
                    .map(|&(node, bit)| cost(bit, KF_B_MODE_PROBS[above][left][node >> 1]))
                    .sum();
            }
        }
    }
    out
}

/// A trial's result.
#[derive(Clone, Copy)]
struct Score {
    distortion: u64,
    rate: u64,
    score: u64,
}

impl Score {
    fn new(distortion: u64, rate: u64, lambda: u64) -> Self {
        Self {
            distortion,
            rate,
            score: rate * lambda + RD_DISTO_MULT * distortion,
        }
    }
}

struct Encoder<'a> {
    source: &'a Source,
    recon: Planes,
    segments: Vec<Segment>,
    segment_of: Vec<u8>,
    method: u8,
    quality: u8,
    probs: CoeffProbs,
    sub_costs: Vec<[[u32; 10]; 10]>,
}

/// State carried across a row and down the columns.
struct Context {
    top_nz: Vec<NonZero>,
    left_nz: NonZero,
    top_modes: Vec<[u8; 4]>,
    left_modes: [u8; 4],
    /// Chroma DC quantization error carried to the next blocks, per
    /// plane: below this macroblock, and right of it.
    top_error: Vec<[[i32; 2]; 2]>,
    left_error: [[i32; 2]; 2],
    /// Whether errors are carried at all.
    diffuse: bool,
}

impl Context {
    fn new(mb_w: usize, diffuse: bool) -> Self {
        Self {
            top_nz: vec![NonZero::default(); mb_w],
            left_nz: NonZero::default(),
            top_modes: vec![[B_DC; 4]; mb_w],
            left_modes: [B_DC; 4],
            top_error: vec![[[0; 2]; 2]; mb_w],
            left_error: [[0; 2]; 2],
            diffuse,
        }
    }

    fn start_row(&mut self) {
        self.left_nz = NonZero::default();
        self.left_modes = [B_DC; 4];
        self.left_error = [[0; 2]; 2];
    }
}

fn block_at(index: usize, pitch: usize) -> usize {
    (1 + (index / 4) * 4) * pitch + 1 + (index % 4) * 4
}

fn chroma_at(index: usize) -> usize {
    (1 + (index / 2) * 4) * CHROMA_PITCH + 1 + (index % 2) * 4
}

fn is_flat_source(src: &[u8], at: usize, stride: usize, size: usize) -> bool {
    let first = src[at];
    (0..size).all(|row| {
        src[at + row * stride..at + row * stride + size]
            .iter()
            .all(|&p| p == first)
    })
}

/// AC levels set in `blocks` (scan order).
fn ac_count(blocks: &[[i16; 16]]) -> usize {
    blocks
        .iter()
        .map(|levels| levels[1..].iter().filter(|&&l| l != 0).count())
        .sum()
}

/// Whether one block has at most `limit` AC levels.
fn is_flat(levels: &[i16; 16], limit: usize) -> bool {
    ac_count(std::slice::from_ref(levels)) <= limit
}

/// The best mode of one sub-block so far.
struct SubBlock {
    score: Score,
    mode: u8,
    levels: [i16; 16],
    pixels: [u8; 16],
    nz: bool,
}

/// Chroma result of one macroblock choice: the U and V work blocks hold
/// the reconstruction, and `errors` the DC errors kept for diffusion.
struct Chroma {
    mode: u8,
    levels: [[i16; 16]; 8],
    work: [[u8; CHROMA_PITCH * 9]; 2],
    errors: [[i32; 3]; 2],
}

/// Luma result of one macroblock choice.
struct Luma {
    work: [u8; LUMA_PITCH * 17],
    is_4x4: bool,
    modes: [u8; 16],
    levels: [[i16; 16]; 17],
    score: Score,
}

impl Encoder<'_> {
    fn luma_src(&self, mb_x: usize, mb_y: usize, block: usize) -> usize {
        (mb_y * 16 + (block / 4) * 4) * self.source.y_stride + mb_x * 16 + (block % 4) * 4
    }

    fn pick_i16(
        &self,
        mb_x: usize,
        mb_y: usize,
        seg: &Segment,
        top: &NonZero,
        left: &NonZero,
    ) -> Luma {
        let mut edges = [0u8; LUMA_PITCH * 17];
        load_edges(
            &mut edges,
            LUMA_PITCH,
            16,
            &self.recon.y,
            self.recon.y_stride,
            mb_x,
            mb_y,
        );
        let ys = self.source.y_stride;
        let src_at = mb_y * 16 * ys + mb_x * 16;
        let mut flat = is_flat_source(&self.source.y, src_at, ys, 16);
        let mut best: Option<Luma> = None;
        for mode in [B_DC, B_TM, B_VE, B_HE] {
            let mut work = edges;
            predict::block(
                &mut work,
                LUMA_PITCH + 1,
                LUMA_PITCH,
                16,
                mode,
                Edges {
                    above: mb_y > 0,
                    left: mb_x > 0,
                },
            );
            let mut coeffs = [[0i16; 16]; 16];
            let mut dc = [0i16; 16];
            for (b, block) in coeffs.iter_mut().enumerate() {
                *block = fdct(
                    &self.source.y,
                    self.luma_src(mb_x, mb_y, b),
                    ys,
                    &work,
                    block_at(b, LUMA_PITCH),
                    LUMA_PITCH,
                );
                dc[b] = block[0];
            }
            let mut levels = [[0i16; 16]; 17];
            let mut y2 = fwht(&dc);
            seg.y2.quantize(&mut y2, &mut levels[16], 0);
            let mut any_ac = false;
            for (b, block) in coeffs.iter_mut().enumerate() {
                any_ac |= seg.y1.quantize(block, &mut levels[b], 1);
            }
            let dcs = inverse_wht(&y2);
            for (b, block) in coeffs.iter_mut().enumerate() {
                block[0] = dcs[b];
                add_idct(block, &mut work, block_at(b, LUMA_PITCH), LUMA_PITCH);
            }
            let distortion = u64::from(sse(
                &self.source.y,
                src_at,
                ys,
                &work,
                LUMA_PITCH + 1,
                LUMA_PITCH,
                (16, 16),
            ));
            let mut texture = 0u64;
            if seg.tlambda > 0 {
                for b in 0..16 {
                    texture += texture_distortion(
                        &self.source.y,
                        self.luma_src(mb_x, mb_y, b),
                        ys,
                        &work,
                        block_at(b, LUMA_PITCH),
                        LUMA_PITCH,
                    ) as u64;
                }
                texture = (seg.tlambda * texture + 128) >> 8;
            }
            let mut rate = Cost::default();
            let mut top = *top;
            let mut left = *left;
            tokens::block(
                &mut rate,
                &self.probs,
                1,
                usize::from(top.y2) + usize::from(left.y2),
                0,
                &levels[16],
            );
            for (b, block) in levels[..16].iter().enumerate() {
                let (row, col) = (b / 4, b % 4);
                let ctx = usize::from(top.y[col]) + usize::from(left.y[row]);
                let nz = tokens::block(&mut rate, &self.probs, 0, ctx, 1, block);
                top.y[col] = nz;
                left.y[row] = nz;
            }
            let rate = u64::from(rate.0 + i16_mode_cost(mode));
            // A flat source that stays flat: insist on low distortion, as
            // libwebp does. Once a mode needs AC levels it no longer counts.
            flat &= !any_ac;
            let weight = if flat { 2 } else { 1 };
            let score = Score::new(weight * (distortion + texture), rate, seg.lambda_i16);
            if best.as_ref().is_none_or(|b| score.score < b.score.score) {
                let mut modes = [B_DC; 16];
                modes[0] = mode;
                best = Some(Luma {
                    work,
                    is_4x4: false,
                    modes,
                    levels,
                    score,
                });
            }
        }
        // Four modes were tried, so there is a best.
        best.unwrap_or_else(|| Luma {
            work: edges,
            is_4x4: false,
            modes: [B_DC; 16],
            levels: [[0; 16]; 17],
            score: Score::new(u64::MAX >> 16, 0, 0),
        })
    }

    /// The sub-block modes, or `None` when they cannot beat `limit`.
    fn pick_i4(
        &self,
        (mb_x, mb_y): (usize, usize),
        seg: &Segment,
        context: &Context,
        limit: u64,
    ) -> Option<Luma> {
        let top = &context.top_nz[mb_x];
        let left = &context.left_nz;
        let mut work = [0u8; LUMA_PITCH * 17];
        load_edges(
            &mut work,
            LUMA_PITCH,
            16,
            &self.recon.y,
            self.recon.y_stride,
            mb_x,
            mb_y,
        );
        if mb_y > 0 {
            let above = (mb_y * 16 - 1) * self.recon.y_stride;
            if mb_x + 1 < self.source.mb_w {
                let start = above + (mb_x + 1) * 16;
                work[17..21].copy_from_slice(&self.recon.y[start..start + 4]);
            } else {
                let last = work[16];
                work[17..21].fill(last);
            }
        }
        for row in [4, 8, 12] {
            work.copy_within(17..21, row * LUMA_PITCH + 17);
        }
        let ys = self.source.y_stride;
        let mut top_nz = top.y;
        let mut left_nz = left.y;
        let mut above_modes = context.top_modes[mb_x];
        let mut left_row = context.left_modes;
        let mut modes = [B_DC; 16];
        let mut levels = [[0i16; 16]; 17];
        let mut total = Score::new(0, u64::from(cost(false, 145)), seg.lambda_mode);
        for b in 0..16 {
            let (row, col) = (b / 4, b % 4);
            let at = block_at(b, LUMA_PITCH);
            let src_at = self.luma_src(mb_x, mb_y, b);
            let ctx = usize::from(top_nz[col]) + usize::from(left_nz[row]);
            let mut best: Option<SubBlock> = None;
            for mode in 0..10u8 {
                let mut trial = work;
                predict::sub_block(&mut trial, at, LUMA_PITCH, mode);
                let mut coeffs = fdct(&self.source.y, src_at, ys, &trial, at, LUMA_PITCH);
                let mut block_levels = [0i16; 16];
                let nz = seg.y1.quantize(&mut coeffs, &mut block_levels, 0);
                add_idct(&coeffs, &mut trial, at, LUMA_PITCH);
                let mode_rate = self.sub_costs[usize::from(above_modes[col])]
                    [usize::from(left_row[row])][usize::from(mode)];
                let distortion = u64::from(sse(
                    &self.source.y,
                    src_at,
                    ys,
                    &trial,
                    at,
                    LUMA_PITCH,
                    (4, 4),
                ));
                let texture = if seg.tlambda > 0 {
                    (seg.tlambda
                        * texture_distortion(&self.source.y, src_at, ys, &trial, at, LUMA_PITCH)
                            as u64
                        + 128)
                        >> 8
                } else {
                    0
                };
                // A complex mode for a nearly flat block pays a penalty.
                let mut rate = u64::from(mode_rate);
                if mode != B_DC && is_flat(&block_levels, FLATNESS_LIMIT_I4) {
                    rate += u64::from(FLATNESS_PENALTY);
                }
                // Without the levels' bits this is a lower bound: skip the
                // token walk when it cannot win.
                if best.as_ref().is_some_and(|b| {
                    Score::new(distortion + texture, rate, seg.lambda_i4).score >= b.score.score
                }) {
                    continue;
                }
                let mut tokens_rate = Cost::default();
                tokens::block(&mut tokens_rate, &self.probs, 3, ctx, 0, &block_levels);
                let rate = rate + u64::from(tokens_rate.0);
                let score = Score::new(distortion + texture, rate, seg.lambda_i4);
                if best.as_ref().is_none_or(|b| score.score < b.score.score) {
                    let mut pixels = [0u8; 16];
                    for r in 0..4 {
                        pixels[r * 4..r * 4 + 4]
                            .copy_from_slice(&trial[at + r * LUMA_PITCH..at + r * LUMA_PITCH + 4]);
                    }
                    best = Some(SubBlock {
                        score,
                        mode,
                        levels: block_levels,
                        pixels,
                        nz,
                    });
                }
            }
            let SubBlock {
                score,
                mode,
                levels: block_levels,
                pixels,
                nz,
            } = best?;
            for r in 0..4 {
                work[at + r * LUMA_PITCH..at + r * LUMA_PITCH + 4]
                    .copy_from_slice(&pixels[r * 4..r * 4 + 4]);
            }
            modes[b] = mode;
            levels[b] = block_levels;
            top_nz[col] = nz;
            left_nz[row] = nz;
            above_modes[col] = mode;
            left_row[row] = mode;
            total = Score::new(
                total.distortion + score.distortion,
                total.rate + score.rate,
                seg.lambda_mode,
            );
            if total.score >= limit {
                return None;
            }
        }
        Some(Luma {
            work,
            is_4x4: true,
            modes,
            levels,
            score: total,
        })
    }

    /// The chroma mode, its levels, reconstruction and kept DC errors.
    fn pick_uv(&self, mb_x: usize, mb_y: usize, seg: &Segment, context: &Context) -> Chroma {
        let (top, left) = (&context.top_nz[mb_x], &context.left_nz);
        let uvs = self.source.uv_stride;
        let mut edges = [[0u8; CHROMA_PITCH * 9]; 2];
        load_edges(
            &mut edges[0],
            CHROMA_PITCH,
            8,
            &self.recon.u,
            self.recon.uv_stride,
            mb_x,
            mb_y,
        );
        load_edges(
            &mut edges[1],
            CHROMA_PITCH,
            8,
            &self.recon.v,
            self.recon.uv_stride,
            mb_x,
            mb_y,
        );
        let src_at = mb_y * 8 * uvs + mb_x * 8;
        let mut best: Option<(u64, Chroma)> = None;
        for mode in [B_DC, B_TM, B_VE, B_HE] {
            let mut works = edges;
            let mut levels = [[0i16; 16]; 8];
            let mut errors = [[0i32; 3]; 2];
            let mut distortion = 0u64;
            for (p, (work, plane)) in works
                .iter_mut()
                .zip([&self.source.u, &self.source.v])
                .enumerate()
            {
                predict::block(
                    work,
                    CHROMA_PITCH + 1,
                    CHROMA_PITCH,
                    8,
                    mode,
                    Edges {
                        above: mb_y > 0,
                        left: mb_x > 0,
                    },
                );
                let mut coeffs = [[0i16; 16]; 4];
                for (b, block) in coeffs.iter_mut().enumerate() {
                    let s = src_at + (b / 2) * 4 * uvs + (b % 2) * 4;
                    *block = fdct(plane, s, uvs, work, chroma_at(b), CHROMA_PITCH);
                }
                if context.diffuse {
                    errors[p] = diffuse_dc(
                        &mut coeffs,
                        &seg.uv,
                        context.top_error[mb_x][p],
                        context.left_error[p],
                    );
                }
                for (b, block) in coeffs.iter_mut().enumerate() {
                    seg.uv.quantize(block, &mut levels[p * 4 + b], 0);
                    add_idct(block, work, chroma_at(b), CHROMA_PITCH);
                }
                distortion += u64::from(sse(
                    plane,
                    src_at,
                    uvs,
                    work,
                    CHROMA_PITCH + 1,
                    CHROMA_PITCH,
                    (8, 8),
                ));
            }
            let mut rate = Cost::default();
            for (p, (top_plane, left_plane)) in
                [(top.u, left.u), (top.v, left.v)].into_iter().enumerate()
            {
                let (mut t, mut l) = (top_plane, left_plane);
                for b in 0..4 {
                    let (row, col) = (b / 2, b % 2);
                    let nz = tokens::block(
                        &mut rate,
                        &self.probs,
                        2,
                        usize::from(t[col]) + usize::from(l[row]),
                        0,
                        &levels[p * 4 + b],
                    );
                    t[col] = nz;
                    l[row] = nz;
                }
            }
            let mut rate = u64::from(rate.0 + uv_mode_cost(mode));
            if mode != B_DC && ac_count(&levels) <= FLATNESS_LIMIT_UV {
                rate += u64::from(FLATNESS_PENALTY) * 8;
            }
            let score = Score::new(distortion, rate, seg.lambda_uv).score;
            if best.as_ref().is_none_or(|b| score < b.0) {
                best = Some((
                    score,
                    Chroma {
                        mode,
                        levels,
                        work: works,
                        errors,
                    },
                ));
            }
        }
        best.map_or(
            Chroma {
                mode: B_DC,
                levels: [[0; 16]; 8],
                work: edges,
                errors: [[0; 3]; 2],
            },
            |(_, chroma)| chroma,
        )
    }

    fn macroblock(
        &mut self,
        (mb_x, mb_y): (usize, usize),
        context: &mut Context,
        stats: &mut Stats,
    ) -> Macroblock {
        let segment = self.segment_of[mb_y * self.source.mb_w + mb_x];
        let seg = &self.segments[usize::from(segment)];
        let top = context.top_nz[mb_x];
        let left = context.left_nz;
        let i16 = self.pick_i16(mb_x, mb_y, seg, &top, &left);
        let luma = if self.method >= 1 {
            // Both sides weigh rate alike here, as libwebp's i4 total does.
            let limit = Score::new(i16.score.distortion, i16.score.rate, seg.lambda_mode).score;
            self.pick_i4((mb_x, mb_y), seg, context, limit)
                .unwrap_or(i16)
        } else {
            i16
        };
        let Chroma {
            mode: chroma,
            levels: uv_levels,
            work: uv_work,
            errors,
        } = self.pick_uv(mb_x, mb_y, seg, context);
        if context.diffuse {
            // libwebp's `StoreDiffusionErrors`.
            for (p, [e1, e2, e3]) in errors.into_iter().enumerate() {
                let left = (3 * e3) >> 2;
                context.left_error[p] = [e1, left];
                context.top_error[mb_x][p] = [e2, e3 - left];
            }
        }
        store(
            &luma.work,
            LUMA_PITCH,
            16,
            &mut self.recon.y,
            self.recon.y_stride,
            mb_x,
            mb_y,
        );
        store(
            &uv_work[0],
            CHROMA_PITCH,
            8,
            &mut self.recon.u,
            self.recon.uv_stride,
            mb_x,
            mb_y,
        );
        store(
            &uv_work[1],
            CHROMA_PITCH,
            8,
            &mut self.recon.v,
            self.recon.uv_stride,
            mb_x,
            mb_y,
        );
        let mut levels = [[0i16; 16]; 25];
        levels[..16].copy_from_slice(&luma.levels[..16]);
        levels[16..24].copy_from_slice(&uv_levels);
        if !luma.is_4x4 {
            levels[24] = luma.levels[16];
        }
        let empty = levels.iter().all(|block| block.iter().all(|&l| l == 0));
        let mb = Macroblock {
            segment,
            is_4x4: luma.is_4x4,
            luma: luma.modes,
            chroma,
            levels,
            empty,
        };
        // Contexts as the token writer leaves them, and the counts.
        write_tokens(
            stats,
            &self.probs,
            &mb,
            &mut context.top_nz[mb_x],
            &mut context.left_nz,
            false,
        );
        if mb.is_4x4 {
            for row in 0..4 {
                for col in 0..4 {
                    context.top_modes[mb_x][col] = mb.luma[row * 4 + col];
                }
                context.left_modes[row] = mb.luma[row * 4 + 3];
            }
        } else {
            context.top_modes[mb_x] = [mb.luma[0]; 4];
            context.left_modes = [mb.luma[0]; 4];
        }
        mb
    }

    /// Every macroblock in raster order. With `refresh`, the costs follow
    /// the probabilities fitted to the macroblocks so far, refreshed
    /// about eight times a picture, as libwebp's token loop does.
    fn encode_all(&mut self, refresh: bool) -> Vec<Macroblock> {
        let (mb_w, mb_h) = (self.source.mb_w, self.source.mb_h);
        let every = ((mb_w * mb_h) >> 3).max(REFRESH_MIN);
        let mut until_refresh = every;
        let mut stats = Stats::new();
        let mut context = Context::new(mb_w, self.quality <= ERROR_DIFFUSION_QUALITY);
        let mut out = Vec::with_capacity(mb_w * mb_h);
        for mb_y in 0..mb_h {
            context.start_row();
            for mb_x in 0..mb_w {
                if refresh {
                    if until_refresh == 0 {
                        self.probs = stats.probabilities();
                        until_refresh = every;
                    }
                    until_refresh -= 1;
                }
                out.push(self.macroblock((mb_x, mb_y), &mut context, &mut stats));
            }
        }
        out
    }
}

/// The tokens of one macroblock, updating the contexts. A skipped
/// macroblock writes nothing and clears them, as the decoder does.
fn write_tokens<S: tokens::Sink>(
    sink: &mut S,
    probs: &CoeffProbs,
    mb: &Macroblock,
    top: &mut NonZero,
    left: &mut NonZero,
    skip: bool,
) {
    if skip {
        let y2 = (top.y2, left.y2);
        *top = NonZero::default();
        *left = NonZero::default();
        if mb.is_4x4 {
            (top.y2, left.y2) = y2;
        }
        return;
    }
    let (first, luma_type) = if mb.is_4x4 {
        (0, 3)
    } else {
        let nz = tokens::block(
            sink,
            probs,
            1,
            usize::from(top.y2) + usize::from(left.y2),
            0,
            &mb.levels[24],
        );
        top.y2 = nz;
        left.y2 = nz;
        (1, 0)
    };
    for row in 0..4 {
        for col in 0..4 {
            let ctx = usize::from(top.y[col]) + usize::from(left.y[row]);
            let nz = tokens::block(
                sink,
                probs,
                luma_type,
                ctx,
                first,
                &mb.levels[row * 4 + col],
            );
            top.y[col] = nz;
            left.y[row] = nz;
        }
    }
    for (p, offset) in [(0, 16), (1, 20)] {
        let (t, l) = if p == 0 {
            (&mut top.u, &mut left.u)
        } else {
            (&mut top.v, &mut left.v)
        };
        for b in 0..4 {
            let (row, col) = (b / 2, b % 2);
            let ctx = usize::from(t[col]) + usize::from(l[row]);
            let nz = tokens::block(sink, probs, 2, ctx, 0, &mb.levels[offset + b]);
            t[col] = nz;
            l[row] = nz;
        }
    }
}

fn token_pass<S: tokens::Sink>(
    sink: &mut S,
    probs: &CoeffProbs,
    mbs: &[Macroblock],
    mb_w: usize,
    skip: bool,
) {
    let mut top = vec![NonZero::default(); mb_w];
    let mut left = NonZero::default();
    for (index, mb) in mbs.iter().enumerate() {
        if index % mb_w == 0 {
            left = NonZero::default();
        }
        write_tokens(
            sink,
            probs,
            mb,
            &mut top[index % mb_w],
            &mut left,
            skip && mb.empty,
        );
    }
}

/// Probability of 0 for a binary choice seen `zeros` and `ones` times.
fn probability(zeros: usize, ones: usize) -> u8 {
    let total = zeros + ones;
    (zeros * 255 + total / 2)
        .checked_div(total)
        .map_or(255, |p| p.clamp(1, 255) as u8)
}

/// Merge segments that ended up with the same settings.
fn simplify(segments: &mut Vec<Segment>, segment_of: &mut [u8]) {
    let mut map: Vec<u8> = Vec::with_capacity(segments.len());
    let mut kept: Vec<usize> = Vec::new();
    for (index, seg) in segments.iter().enumerate() {
        let same = kept.iter().position(|&k| {
            segments[k].quant_index == seg.quant_index
                && segments[k].filter_level == seg.filter_level
        });
        match same {
            Some(at) => map.push(at as u8),
            None => {
                map.push(kept.len() as u8);
                kept.push(index);
            }
        }
    }
    if kept.len() == segments.len() {
        return;
    }
    for s in segment_of.iter_mut() {
        *s = map[usize::from(*s)];
    }
    let mut index = 0;
    segments.retain(|_| {
        let keep = kept.contains(&index);
        index += 1;
        keep
    });
}

/// Mean SSIM of the luma of `a` against `b` over 8x8 windows, step 4.
fn ssim(a: &[u8], b: &[u8], stride: usize, width: usize, height: usize) -> f64 {
    const C1: f64 = 6.5025;
    const C2: f64 = 58.5225;
    let mut total = 0.0;
    let mut count = 0usize;
    let mut y = 0;
    while y + 8 <= height {
        let mut x = 0;
        while x + 8 <= width {
            let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for row in y..y + 8 {
                for col in x..x + 8 {
                    let p = f64::from(a[row * stride + col]);
                    let q = f64::from(b[row * stride + col]);
                    sa += p;
                    sb += q;
                    saa += p * p;
                    sbb += q * q;
                    sab += p * q;
                }
            }
            let n = 64.0;
            let (ma, mb) = (sa / n, sb / n);
            let va = saa / n - ma * ma;
            let vb = sbb / n - mb * mb;
            let cov = sab / n - ma * mb;
            total += ((2.0 * ma * mb + C1) * (2.0 * cov + C2))
                / ((ma * ma + mb * mb + C1) * (va + vb + C2));
            count += 1;
            x += 4;
        }
        y += 8 - 4;
    }
    if count == 0 {
        1.0
    } else {
        total / count as f64
    }
}

/// Filter levels per segment: libwebp's formula, and from method 4 the
/// multiple of it whose filtered luma is most like the source.
fn choose_filter(
    source: &Source,
    recon: &Planes,
    segments: &mut [Segment],
    mbs: &[Macroblock],
    method: u8,
    width: usize,
    height: usize,
) {
    if method < 4 {
        return;
    }
    let base: Vec<i32> = segments.iter().map(|s| s.filter_level).collect();
    let scales = [0, 2, 3, 4, 6, 8, 12, 16];
    // libwebp's own level, unless another is clearly better.
    const DEFAULT: usize = 3;
    const MARGIN: f64 = 1e-5;
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    let mut scores = [0.0f64; 8];
    for (score, &scale) in scores.iter_mut().zip(&scales) {
        let levels: Vec<i32> = base.iter().map(|&l| (l * scale / 4).min(63)).collect();
        let mut planes = Planes {
            width: recon.width,
            height: recon.height,
            y: recon.y.clone(),
            y_stride: recon.y_stride,
            u: recon.u.clone(),
            v: recon.v.clone(),
            uv_stride: recon.uv_stride,
        };
        if levels.iter().any(|&l| l > 0) {
            let filters = filter_list(&levels, mbs);
            loop_filter(&mut planes, source.mb_w, FilterKind::Normal, &filters);
        }
        // Luma counts four times as much as each chroma plane.
        let y = ssim(&source.y, &planes.y, source.y_stride, width, height);
        let u = ssim(&source.u, &planes.u, source.uv_stride, cw, ch);
        let v = ssim(&source.v, &planes.v, source.uv_stride, cw, ch);
        *score = (4.0 * y + u + v) / 6.0;
    }
    let mut best = (scores[DEFAULT] + MARGIN, DEFAULT);
    for (index, &score) in scores.iter().enumerate() {
        if score > best.0 {
            best = (score, index);
        }
    }
    for (seg, &level) in segments.iter_mut().zip(&base) {
        seg.filter_level = (level * scales[best.1] / 4).min(63);
    }
}

fn filter_list(levels: &[i32], mbs: &[Macroblock]) -> Vec<(FilterStrength, bool)> {
    mbs.iter()
        .map(|mb| {
            let level = levels[usize::from(mb.segment)];
            (filter_strength(level, 0), mb.is_4x4 || !mb.empty)
        })
        .collect()
}

/// A `VP8 ` key frame of `rgba` and, when any pixel is not opaque, an
/// `ALPH` payload.
///
/// A small picture also tries the other qualities and keeps a frame that
/// is no larger and no worse: there, rounding decides more than the
/// quantizer does, and a single colour can cost fewer bytes exactly than
/// approximately.
pub(crate) fn encode(
    rgba: &[u8],
    width: u32,
    height: u32,
    config: Config,
) -> Result<(Vec<u8>, Option<Vec<u8>>), CodecError> {
    let mut vp8 = encode_frame(rgba, width, height, config)?;
    let macroblocks = (width as usize).div_ceil(16) * (height as usize).div_ceil(16);
    if macroblocks <= SMALL_PICTURE {
        let mut error = frame_error(&vp8, rgba)?;
        for quality in (0..=100).step_by(10) {
            if quality == config.quality {
                continue;
            }
            let other = encode_frame(rgba, width, height, Config { quality, ..config })?;
            if other.len() > vp8.len() {
                continue;
            }
            let other_error = frame_error(&other, rgba)?;
            if other_error < error || (other_error == error && other.len() < vp8.len()) {
                (vp8, error) = (other, other_error);
            }
        }
    }
    let alpha = if rgba.as_chunks::<4>().0.iter().any(|p| p[3] != 255) {
        Some(alpha_chunk(rgba, width, height, config.method))
    } else {
        None
    };
    Ok((vp8, alpha))
}

/// Squared RGB error of a frame over the visible pixels of `rgba`.
fn frame_error(vp8: &[u8], rgba: &[u8]) -> Result<u64, CodecError> {
    let decoded = to_rgba(&decode(vp8)?);
    Ok(decoded
        .as_chunks::<4>()
        .0
        .iter()
        .zip(rgba.as_chunks::<4>().0)
        .filter(|(_, p)| p[3] != 0)
        .map(|(a, b)| {
            (0..3)
                .map(|c| {
                    let d = i64::from(a[c]) - i64::from(b[c]);
                    (d * d) as u64
                })
                .sum::<u64>()
        })
        .sum())
}

/// The `VP8 ` frame of `rgba` at `config`.
fn encode_frame(
    rgba: &[u8],
    width: u32,
    height: u32,
    config: Config,
) -> Result<Vec<u8>, CodecError> {
    let (w, h) = (width as usize, height as usize);
    let source = Source::new(rgba, w, h, config.exact);
    let analysis = analyze(&source, SEGMENTS);
    let (mut segment_list, deltas) = segments(
        &analysis.shapes,
        analysis.uv_alpha,
        config.quality,
        config.method,
    );
    let mut segment_of = analysis.segments;
    simplify(&mut segment_list, &mut segment_of);
    let recon = Planes {
        width: w,
        height: h,
        y: vec![0; source.y.len()],
        y_stride: source.y_stride,
        u: vec![0; source.u.len()],
        v: vec![0; source.v.len()],
        uv_stride: source.uv_stride,
    };
    let mut encoder = Encoder {
        source: &source,
        recon,
        segments: segment_list,
        segment_of,
        method: config.method,
        quality: config.quality,
        probs: DEFAULT_COEFF_PROBS,
        sub_costs: sub_mode_costs(),
    };
    let mut mbs = encoder.encode_all(config.method >= 3);
    let mb_w = source.mb_w;
    let mut probs = fitted(&mbs, mb_w);
    if config.method >= 5 {
        // Decide again with costs from the fitted probabilities.
        encoder.probs = probs;
        mbs = encoder.encode_all(false);
        probs = fitted(&mbs, mb_w);
    }
    let Encoder {
        recon,
        mut segments,
        segment_of,
        ..
    } = encoder;
    choose_filter(&source, &recon, &mut segments, &mbs, config.method, w, h);

    let empty = mbs.iter().filter(|mb| mb.empty).count();
    let skip_prob = probability(mbs.len() - empty, empty);
    let use_skip = skip_prob < 250;

    // Partition 0: the header and the modes.
    let mut first = BoolEncoder::with_capacity(mbs.len() * 4 + 1024);
    first.flag(false); // colour space
    first.flag(false); // clamping type
    let segmented = segments.len() > 1;
    first.flag(segmented);
    let mut tree = [255u8; 3];
    if segmented {
        let mut counts = [0usize; 4];
        for &s in &segment_of {
            counts[usize::from(s)] += 1;
        }
        tree = [
            probability(counts[0] + counts[1], counts[2] + counts[3]),
            probability(counts[0], counts[1]),
            probability(counts[2], counts[3]),
        ];
        first.flag(true); // update the map
        first.flag(true); // update the data
        first.flag(true); // absolute values
        for s in 0..4 {
            first.optional_signed(segments.get(s).map_or(0, |seg| seg.quant_index), 7);
        }
        for s in 0..4 {
            first.optional_signed(segments.get(s).map_or(0, |seg| seg.filter_level), 6);
        }
        for &prob in &tree {
            first.flag(prob != 255);
            if prob != 255 {
                first.literal(u32::from(prob), 8);
            }
        }
    }
    let level = segments.iter().map(|s| s.filter_level).max().unwrap_or(0);
    first.flag(false); // normal filter
    first.literal(level as u32, 6);
    first.literal(0, 3); // sharpness
    first.flag(false); // no filter deltas
    first.literal(0, 2); // one token partition
    first.literal(segments[0].quant_index as u32, 7);
    for delta in deltas {
        first.optional_signed(delta, 4);
    }
    first.flag(false); // refresh entropy probabilities
    tokens::write_updates(&mut first, &probs);
    first.flag(use_skip);
    if use_skip {
        first.literal(u32::from(skip_prob), 8);
    }
    let mut top_modes = vec![[B_DC; 4]; mb_w];
    let mut left_modes = [B_DC; 4];
    for (index, mb) in mbs.iter().enumerate() {
        let mb_x = index % mb_w;
        if mb_x == 0 {
            left_modes = [B_DC; 4];
        }
        if segmented {
            let s = mb.segment;
            first.put(s >= 2, tree[0]);
            if s >= 2 {
                first.put(s == 3, tree[2]);
            } else {
                first.put(s == 1, tree[1]);
            }
        }
        if use_skip {
            first.put(mb.empty, skip_prob);
        }
        first.put(!mb.is_4x4, 145);
        if mb.is_4x4 {
            for row in 0..4 {
                for col in 0..4 {
                    let mode = mb.luma[row * 4 + col];
                    let probs = &KF_B_MODE_PROBS[usize::from(top_modes[mb_x][col])]
                        [usize::from(left_modes[row])];
                    first.tree(&B_MODE_TREE, probs, usize::from(mode));
                    top_modes[mb_x][col] = mode;
                    left_modes[row] = mode;
                }
            }
        } else {
            let mode = mb.luma[0];
            match mode {
                B_TM => {
                    first.put(true, 156);
                    first.put(true, 128);
                }
                B_HE => {
                    first.put(true, 156);
                    first.put(false, 128);
                }
                B_VE => {
                    first.put(false, 156);
                    first.put(true, 163);
                }
                _ => {
                    first.put(false, 156);
                    first.put(false, 163);
                }
            }
            top_modes[mb_x] = [mode; 4];
            left_modes = [mode; 4];
        }
        match mb.chroma {
            B_DC => first.put(false, 142),
            B_VE => {
                first.put(true, 142);
                first.put(false, 114);
            }
            mode => {
                first.put(true, 142);
                first.put(true, 114);
                first.put(mode == B_TM, 183);
            }
        }
    }
    let first = first.finish();
    if first.len() > MAX_FIRST_PARTITION {
        return Err(CodecError::Limit {
            what: "vp8 first partition",
        });
    }
    let mut part = BoolEncoder::with_capacity(source.y.len() / 4);
    token_pass(&mut part, &probs, &mbs, mb_w, use_skip);
    let part = part.finish();

    let mut vp8 = Vec::with_capacity(10 + first.len() + part.len());
    let tag = (1u32 << 4) | ((first.len() as u32) << 5);
    vp8.extend_from_slice(&tag.to_le_bytes()[..3]);
    vp8.extend_from_slice(&[0x9d, 0x01, 0x2a]);
    vp8.extend_from_slice(&(width as u16).to_le_bytes());
    vp8.extend_from_slice(&(height as u16).to_le_bytes());
    vp8.extend_from_slice(&first);
    vp8.extend_from_slice(&part);
    Ok(vp8)
}

fn fitted(mbs: &[Macroblock], mb_w: usize) -> CoeffProbs {
    let mut stats = Stats::new();
    token_pass(&mut stats, &DEFAULT_COEFF_PROBS, mbs, mb_w, false);
    stats.probabilities()
}

/// An `ALPH` payload: the alpha plane as a headerless lossless stream,
/// with the prediction filter that codes smallest.
fn alpha_chunk(rgba: &[u8], width: u32, height: u32, method: u8) -> Vec<u8> {
    let plane: Vec<u8> = rgba.as_chunks::<4>().0.iter().map(|p| p[3]).collect();
    let filters: &[Filter] = if method >= 3 {
        &[
            Filter::None,
            Filter::Horizontal,
            Filter::Vertical,
            Filter::Gradient,
        ]
    } else {
        &[Filter::None]
    };
    let config = LosslessConfig {
        effort: method.min(6),
        exact: true,
    };
    let mut best: Option<Vec<u8>> = None;
    for &filter in filters {
        let filtered = alpha::filter(&plane, width as usize, filter);
        let argb: Vec<u32> = filtered
            .iter()
            .map(|&a| 0xff00_0000 | (u32::from(a) << 8))
            .collect();
        let mut payload = vec![1 | (filter.bits() << 2)];
        payload.extend_from_slice(&encode_stream(&argb, width, height, config));
        if best.as_ref().is_none_or(|b| payload.len() < b.len()) {
            best = Some(payload);
        }
    }
    best.unwrap_or_default()
}

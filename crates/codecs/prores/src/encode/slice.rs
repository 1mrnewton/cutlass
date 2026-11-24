//! One slice: forward transform, quantization, entropy coding, alpha.

use super::source::{MAX_COLUMNS, Picture};
use crate::bits::BitWriter;
use crate::dct::fdct;
use crate::header::Chroma;
use crate::tables::{
    FIRST_DC_CODEBOOK, FIRST_DC_SYMBOL, FIRST_LEVEL, FIRST_RUN, PROGRESSIVE_SCAN, dc_codebook,
    level_codebook, qscale, run_codebook,
};

/// Largest quantized magnitude written; every codeword then fits in 31 bits.
const MAX_QUANT: f32 = 8192.0;
/// Most blocks one component of a slice holds: 8 macroblocks of 4.
const MAX_BLOCKS: usize = 32;

/// One worker's memory for a slice: the samples it reads, their
/// coefficients, and what it quantizes and codes them through.
pub(crate) struct Work {
    /// 16 rows of the slice, `columns` samples each (chroma: its own width).
    ys: Vec<f32>,
    us: Vec<f32>,
    vs: Vec<f32>,
    /// Coefficients in coded order: index `i * blocks + b` holds block
    /// `b`'s coefficient at scan position `i`, times 8.
    y: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
    quantized: Vec<i32>,
    alpha_values: Vec<u16>,
    /// The slice's coded alpha, the same at every quantization index.
    alpha: Vec<u8>,
    luma_blocks: usize,
    chroma_blocks: usize,
}

impl Default for Work {
    fn default() -> Self {
        let samples = 16 * MAX_COLUMNS;
        let coefs = 64 * MAX_BLOCKS;
        Self {
            ys: vec![0.0; samples],
            us: vec![0.0; samples],
            vs: vec![0.0; samples],
            y: vec![0.0; coefs],
            u: vec![0.0; coefs],
            v: vec![0.0; coefs],
            quantized: vec![0; coefs],
            alpha_values: Vec::new(),
            alpha: Vec::new(),
            luma_blocks: 0,
            chroma_blocks: 0,
        }
    }
}

impl Work {
    /// Reads and transforms the slice of `mbs` macroblocks at macroblock
    /// `(mb_x, row)`, and codes its alpha.
    #[inline(always)]
    pub(crate) fn prepare(
        &mut self,
        picture: &Picture<'_>,
        basis: &[[f32; 8]; 8],
        row: usize,
        mb_x: usize,
        mbs: usize,
    ) {
        let mbs = mbs.clamp(1, 8);
        let columns = 16 * mbs;
        let chroma_columns = picture.chroma_columns(columns);
        self.luma_blocks = 4 * mbs;
        self.chroma_blocks = match picture.chroma {
            Chroma::Yuv422 => 2 * mbs,
            Chroma::Yuv444 => 4 * mbs,
        };
        picture.gather(
            row,
            mb_x,
            columns,
            &mut self.ys[..16 * columns],
            &mut self.us[..16 * chroma_columns],
            &mut self.vs[..16 * chroma_columns],
        );
        for block in 0..self.luma_blocks {
            let (mb, sub) = (block / 4, block % 4);
            let at = Block {
                columns,
                x: mb * 16 + (sub & 1) * 8,
                top: (sub >> 1) * 8,
                block,
                blocks: self.luma_blocks,
            };
            transform(&self.ys, at, &mut self.y, basis);
        }
        for block in 0..self.chroma_blocks {
            let (x, top) = match picture.chroma {
                Chroma::Yuv422 => ((block / 2) * 8, (block % 2) * 8),
                Chroma::Yuv444 => {
                    let sub = block % 4;
                    ((block / 4) * 16 + (sub >> 1) * 8, (sub & 1) * 8)
                }
            };
            let at = Block {
                columns: chroma_columns,
                x,
                top,
                block,
                blocks: self.chroma_blocks,
            };
            transform(&self.us, at, &mut self.u, basis);
            transform(&self.vs, at, &mut self.v, basis);
        }
        self.alpha.clear();
        if let Some(sixteen) = picture.alpha {
            picture.alpha(row, mb_x, columns, &mut self.alpha_values);
            encode_alpha(&self.alpha_values, sixteen, &mut self.alpha);
        }
    }

    /// Appends the prepared slice to `out`, coded at quantization index
    /// `index` with these steps. Returns its length in bytes.
    #[inline(always)]
    pub(crate) fn write(
        &mut self,
        index: u8,
        luma: &[f32; 64],
        chroma: &[f32; 64],
        out: &mut Vec<u8>,
    ) -> usize {
        let start = out.len();
        let alpha = !self.alpha.is_empty();
        let header_size: u8 = if alpha { 8 } else { 6 };
        out.push(header_size << 3);
        out.push(index);
        out.resize(start + usize::from(header_size), 0);
        let mut sizes = [0usize; 3];
        for (part, (coefs, blocks, steps)) in [
            (&self.y, self.luma_blocks, luma),
            (&self.u, self.chroma_blocks, chroma),
            (&self.v, self.chroma_blocks, chroma),
        ]
        .into_iter()
        .enumerate()
        {
            let quantized = &mut self.quantized[..64 * blocks];
            quantize(&coefs[..64 * blocks], blocks, steps, quantized);
            let before = out.len();
            let mut writer = BitWriter::onto(std::mem::take(out));
            code_component(&mut writer, quantized, blocks);
            *out = writer.finish();
            sizes[part] = out.len() - before;
        }
        // The sizes are 16-bit fields. A slice that overflows one is over
        // the slice limit as a whole, and the caller codes it again coarser.
        let field = |size: usize| (size.min(usize::from(u16::MAX)) as u16).to_be_bytes();
        out[start + 2..start + 4].copy_from_slice(&field(sizes[0]));
        out[start + 4..start + 6].copy_from_slice(&field(sizes[1]));
        if alpha {
            out[start + 6..start + 8].copy_from_slice(&field(sizes[2]));
        }
        out.extend_from_slice(&self.alpha);
        out.len() - start
    }

    /// Bytes of coded alpha in the prepared slice.
    pub(crate) fn alpha_len(&self) -> usize {
        self.alpha.len()
    }
}

/// A block of a slice: its corner in the slice's samples, and its place
/// among the component's `blocks` blocks.
#[derive(Clone, Copy)]
struct Block {
    columns: usize,
    x: usize,
    top: usize,
    block: usize,
    blocks: usize,
}

/// Transforms one block of a slice's samples into coded order.
#[inline(always)]
fn transform(samples: &[f32], at: Block, out: &mut [f32], basis: &[[f32; 8]; 8]) {
    let mut square = [0f32; 64];
    for r in 0..8 {
        let start = (at.top + r) * at.columns + at.x;
        if let Some(line) = samples.get(start..start + 8) {
            square[r * 8..r * 8 + 8].copy_from_slice(line);
        }
    }
    fdct(&mut square, basis);
    for (i, &raster) in PROGRESSIVE_SCAN.iter().enumerate() {
        if let Some(slot) = out.get_mut(i * at.blocks + at.block) {
            *slot = square[usize::from(raster)] * 8.0;
        }
    }
}

/// Where coded bits go: counted, or written.
pub(crate) trait Sink {
    fn code(&mut self, codebook: u8, value: u32);
    fn bit(&mut self, value: u32);
}

#[cfg(test)]
#[derive(Default)]
pub(crate) struct Counter {
    pub bits: u32,
}

#[cfg(test)]
impl Sink for Counter {
    #[inline]
    fn code(&mut self, codebook: u8, value: u32) {
        self.bits += crate::bits::code_len(codebook, value);
    }

    #[inline]
    fn bit(&mut self, _value: u32) {
        self.bits += 1;
    }
}

impl Sink for BitWriter {
    #[inline]
    fn code(&mut self, codebook: u8, value: u32) {
        BitWriter::code(self, codebook, value);
    }

    #[inline]
    fn bit(&mut self, value: u32) {
        self.put(1, value);
    }
}

/// Reciprocal step per scan position: `1 / (W[scan[i]] * qScale)`.
pub(crate) fn steps(matrix: &[u8; 64], index: u8) -> [f32; 64] {
    let q = qscale(index) as f32;
    let mut out = [0f32; 64];
    for (i, &raster) in PROGRESSIVE_SCAN.iter().enumerate() {
        out[i] = 1.0 / (f32::from(matrix[usize::from(raster)]) * q);
    }
    out
}

/// Two to the 23rd: from here up a single-precision value is a whole number.
const WHOLE: f32 = 8_388_608.0;

/// `value` rounded to the nearest whole number, halves away from zero,
/// and clamped to the largest magnitude written.
///
/// The magnitude is clamped, and adding [`WHOLE`] rounds it to a whole
/// number that sits in the low bits of the sum. One comes off where that
/// rounded up, which leaves the floor; the fraction left over, which is
/// exact, says whether to go up. This is `round` without a call or a
/// float-to-integer cast per coefficient, so the loop runs eight wide.
#[inline(always)]
fn round_clamped(value: f32) -> i32 {
    let size = value.abs();
    let size = if size < MAX_QUANT { size } else { MAX_QUANT };
    let lifted = size + WHOLE;
    let nearest = lifted - WHOLE;
    let above = nearest > size;
    let floor = nearest - if above { 1.0 } else { 0.0 };
    let whole =
        (lifted.to_bits() & 0x007f_ffff) as i32 - i32::from(above) + i32::from(size - floor >= 0.5);
    if value < 0.0 { -whole } else { whole }
}

/// Quantizes one component, coded order in and out.
#[inline(always)]
pub(crate) fn quantize(coefs: &[f32], blocks: usize, steps: &[f32; 64], out: &mut [i32]) {
    for ((src, dst), &step) in coefs
        .chunks_exact(blocks)
        .zip(out.chunks_exact_mut(blocks))
        .zip(steps)
    {
        for (d, &c) in dst.iter_mut().zip(src) {
            *d = round_clamped(c * step);
        }
    }
}

/// Codes one quantized component: DC differences, then interleaved runs and levels.
#[inline(always)]
pub(crate) fn code_component<S: Sink>(sink: &mut S, quantized: &[i32], blocks: usize) {
    let first = quantized[0];
    sink.code(FIRST_DC_CODEBOOK, signed_code(first));
    let mut previous_dc = first;
    let mut previous_code = FIRST_DC_SYMBOL;
    let mut negative = false;
    for &dc in &quantized[1..blocks] {
        let difference = dc - previous_dc;
        let code = if difference == 0 {
            negative = false;
            0
        } else {
            let magnitude = difference.unsigned_abs();
            let now_negative = difference < 0;
            let code = if now_negative == negative {
                2 * magnitude
            } else {
                2 * magnitude - 1
            };
            negative = now_negative;
            code
        };
        sink.code(dc_codebook(previous_code), code);
        previous_code = code;
        previous_dc = dc;
    }
    let mut last = blocks - 1;
    let mut run = FIRST_RUN;
    let mut level = FIRST_LEVEL;
    // Most coefficients are zero: whole groups of them are passed over
    // with one test.
    let ac = quantized.get(blocks..).unwrap_or_default();
    for (group, values) in ac.chunks(ZERO_GROUP).enumerate() {
        if values.iter().fold(0, |any, &value| any | value) == 0 {
            continue;
        }
        for (offset, &value) in values.iter().enumerate() {
            if value == 0 {
                continue;
            }
            let pos = blocks + group * ZERO_GROUP + offset;
            let this_run = (pos - last - 1) as u32;
            sink.code(run_codebook(run), this_run);
            let magnitude = value.unsigned_abs();
            sink.code(level_codebook(level), magnitude - 1);
            sink.bit(u32::from(value < 0));
            run = this_run;
            level = magnitude;
            last = pos;
        }
    }
}

/// Coefficients tested for zero at once.
const ZERO_GROUP: usize = 16;

#[inline]
fn signed_code(value: i32) -> u32 {
    if value >= 0 {
        2 * value.unsigned_abs()
    } else {
        2 * value.unsigned_abs() - 1
    }
}

/// Codes `values` as RDD 36 alpha: differences from the last value, with runs.
pub(crate) fn encode_alpha(values: &[u16], sixteen: bool, out: &mut Vec<u8>) {
    let (full, short, reach) = if sixteen { (16, 7, 64) } else { (8, 4, 8) };
    let mask: i32 = (1 << full) - 1;
    let mut previous = mask;
    let mut writer = BitWriter::onto(std::mem::take(out));
    let mut index = 0;
    while index < values.len() {
        let value = i32::from(values[index]) & mask;
        let mut difference = (value - previous) & mask;
        if difference > mask / 2 {
            difference -= mask + 1;
        }
        let magnitude = difference.unsigned_abs();
        if difference != 0 && magnitude <= reach {
            writer.put(1, 0);
            writer.put(short, 2 * (magnitude - 1) + u32::from(difference < 0));
        } else {
            writer.put(1, 1);
            writer.put(full, (difference & mask) as u32);
        }
        previous = value;
        index += 1;
        if index >= values.len() {
            break;
        }
        let repeats = values[index..]
            .iter()
            .take(2047)
            .take_while(|&&next| i32::from(next) & mask == value)
            .count();
        if repeats == 0 {
            writer.put(1, 1);
            continue;
        }
        writer.put(1, 0);
        if repeats <= 15 {
            writer.put(4, repeats as u32);
        } else {
            writer.put(4, 0);
            writer.put(11, repeats as u32);
        }
        index += repeats;
    }
    *out = writer.finish();
}

#[cfg(test)]
mod tests {
    use super::{Counter, MAX_QUANT, Sink, code_component, encode_alpha, quantize, round_clamped};
    use crate::bits::BitWriter;

    #[test]
    fn rounding_is_round_half_away_clamped() {
        let want = |value: f32| value.round().clamp(-MAX_QUANT, MAX_QUANT) as i32;
        let mut values = vec![0.0f32, -0.0, 1e30, -1e30, f32::INFINITY, f32::NEG_INFINITY];
        // Every half step up to past the clamp, and the floats around it.
        for k in 0..17_000i32 {
            let value = k as f32 * 0.5;
            for off in 0..4u32 {
                for candidate in [
                    f32::from_bits(value.to_bits().wrapping_add(off)),
                    f32::from_bits(value.to_bits().wrapping_sub(off)),
                ] {
                    values.push(candidate);
                    values.push(-candidate);
                }
            }
        }
        let mut state = 5u32;
        for _ in 0..400_000 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let unit = state as f32 / 4_294_967_296.0 - 0.5;
            values.push(unit * 3.0);
            values.push(unit * 300.0);
            values.push(unit * 20_000.0);
        }
        for value in values {
            if value.is_nan() {
                continue;
            }
            assert_eq!(round_clamped(value), want(value), "{value:e}");
        }
    }

    #[test]
    fn quantizing_a_component_rounds_every_coefficient() {
        let blocks = 8;
        let coefs: Vec<f32> = (0..64 * blocks)
            .map(|i| ((i * 7919) % 4001) as f32 - 2000.25)
            .collect();
        let mut steps = [0f32; 64];
        for (i, step) in steps.iter_mut().enumerate() {
            *step = 1.0 / (4.0 + i as f32);
        }
        let mut out = vec![0i32; coefs.len()];
        quantize(&coefs, blocks, &steps, &mut out);
        for (at, (&q, &c)) in out.iter().zip(&coefs).enumerate() {
            let want = (c * steps[at / blocks]).round() as i32;
            assert_eq!(q, want, "coefficient {at}");
        }
    }

    #[test]
    fn counting_matches_writing() {
        let blocks = 8;
        let mut quantized = vec![0i32; 64 * blocks];
        for (i, q) in quantized.iter_mut().enumerate() {
            if i % 7 == 0 || i < blocks {
                *q = ((i * 37) % 41) as i32 - 20;
            }
        }
        let mut counter = Counter::default();
        code_component(&mut counter, &quantized, blocks);
        let mut writer = BitWriter::default();
        code_component(&mut writer, &quantized, blocks);
        assert_eq!(counter.bits as usize, writer.len_bits());
        counter.bit(1);
        assert_eq!(counter.bits as usize, writer.len_bits() + 1);
    }

    #[test]
    fn alpha_round_trips_through_the_decoder_rules() {
        let mut values: Vec<u16> = vec![255; 40];
        values.extend((0..300).map(|i| (i * 13 % 256) as u16));
        values.extend(vec![7; 3000]);
        values.extend([8, 9, 7, 0, 255]);
        for sixteen in [false, true] {
            let coded: Vec<u16> = if sixteen {
                values.iter().map(|&v| v * 257).collect()
            } else {
                values.clone()
            };
            let mut bytes = Vec::new();
            encode_alpha(&coded, sixteen, &mut bytes);
            let mut out = Vec::new();
            crate::decode::unpack_alpha_for_tests(&bytes, sixteen, coded.len(), &mut out);
            assert_eq!(out, coded, "sixteen {sixteen}");
        }
    }
}

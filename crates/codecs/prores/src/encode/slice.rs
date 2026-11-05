//! One slice: forward transform, quantization, entropy coding, alpha.

use crate::bits::{BitWriter, code_len};
use crate::dct::fdct;
use crate::header::Chroma;
use crate::tables::{
    FIRST_DC_CODEBOOK, FIRST_DC_SYMBOL, FIRST_LEVEL, FIRST_RUN, PROGRESSIVE_SCAN, dc_codebook,
    level_codebook, qscale, run_codebook,
};

/// Largest quantized magnitude written; every codeword then fits in 31 bits.
const MAX_QUANT: f32 = 8192.0;

/// Encoder input in the transform domain: each sample less mid-grey,
/// scaled to RDD 36's 9-bit range, padded to whole macroblocks by
/// repeating the last row and column.
pub(crate) struct Planes {
    pub width: u32,
    pub height: u32,
    pub chroma: Chroma,
    pub luma_stride: usize,
    pub chroma_stride: usize,
    pub y: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub alpha: Option<AlphaPlane>,
}

/// Straight alpha, unpadded rows of `stride` samples (the padded width).
pub(crate) struct AlphaPlane {
    pub sixteen: bool,
    pub stride: usize,
    pub samples: Vec<u16>,
}

/// A slice's coefficients, each component in coded order: index
/// `i * blocks + b` holds block `b`'s coefficient at scan position `i`, times 8.
pub(crate) struct Prepared {
    pub luma_blocks: usize,
    pub chroma_blocks: usize,
    pub y: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub alpha: Vec<u8>,
}

impl Planes {
    /// Transforms one slice of `mbs` macroblocks at macroblock `(mb_x, row)`.
    pub(crate) fn prepare(&self, row: usize, mb_x: usize, mbs: usize) -> Prepared {
        let luma_blocks = 4 * mbs;
        let chroma_blocks = match self.chroma {
            Chroma::Yuv422 => 2 * mbs,
            Chroma::Yuv444 => 4 * mbs,
        };
        let mut y = vec![0f32; 64 * luma_blocks];
        for block in 0..luma_blocks {
            let (mb, sub) = (block / 4, block % 4);
            let x = (mb_x + mb) * 16 + (sub & 1) * 8;
            let top = row * 16 + (sub >> 1) * 8;
            transform(
                &self.y,
                self.luma_stride,
                x,
                top,
                block,
                luma_blocks,
                &mut y,
            );
        }
        let mut u = vec![0f32; 64 * chroma_blocks];
        let mut v = vec![0f32; 64 * chroma_blocks];
        for block in 0..chroma_blocks {
            let (x, top) = match self.chroma {
                Chroma::Yuv422 => ((mb_x + block / 2) * 8, row * 16 + (block % 2) * 8),
                Chroma::Yuv444 => {
                    let sub = block % 4;
                    (
                        (mb_x + block / 4) * 16 + (sub >> 1) * 8,
                        row * 16 + (sub & 1) * 8,
                    )
                }
            };
            transform(
                &self.u,
                self.chroma_stride,
                x,
                top,
                block,
                chroma_blocks,
                &mut u,
            );
            transform(
                &self.v,
                self.chroma_stride,
                x,
                top,
                block,
                chroma_blocks,
                &mut v,
            );
        }
        let alpha = match &self.alpha {
            Some(plane) => {
                let visible = (self.height as usize - row * 16).min(16);
                let columns = 16 * mbs;
                let x0 = mb_x * 16;
                let mut values = Vec::with_capacity(columns * visible);
                for r in 0..visible {
                    let start = (row * 16 + r) * plane.stride + x0;
                    values.extend_from_slice(&plane.samples[start..start + columns]);
                }
                encode_alpha(&values, plane.sixteen)
            }
            None => Vec::new(),
        };
        Prepared {
            luma_blocks,
            chroma_blocks,
            y,
            u,
            v,
            alpha,
        }
    }
}

fn transform(
    plane: &[f32],
    stride: usize,
    x: usize,
    top: usize,
    block: usize,
    blocks: usize,
    out: &mut [f32],
) {
    let mut samples = [0f32; 64];
    for r in 0..8 {
        let start = (top + r) * stride + x;
        samples[r * 8..r * 8 + 8].copy_from_slice(&plane[start..start + 8]);
    }
    fdct(&mut samples);
    for (i, &raster) in PROGRESSIVE_SCAN.iter().enumerate() {
        out[i * blocks + block] = samples[usize::from(raster)] * 8.0;
    }
}

/// Where coded bits go: counted, or written.
pub(crate) trait Sink {
    fn code(&mut self, codebook: u8, value: u32);
    fn bit(&mut self, value: u32);
}

#[derive(Default)]
pub(crate) struct Counter {
    pub bits: u32,
}

impl Sink for Counter {
    #[inline]
    fn code(&mut self, codebook: u8, value: u32) {
        self.bits += code_len(codebook, value);
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

/// Quantizes one component, coded order in and out.
pub(crate) fn quantize(coefs: &[f32], blocks: usize, steps: &[f32; 64], out: &mut Vec<i32>) {
    out.clear();
    out.resize(coefs.len(), 0);
    for (i, &step) in steps.iter().enumerate() {
        let src = &coefs[i * blocks..(i + 1) * blocks];
        let dst = &mut out[i * blocks..(i + 1) * blocks];
        for (d, &c) in dst.iter_mut().zip(src) {
            *d = (c * step).round().clamp(-MAX_QUANT, MAX_QUANT) as i32;
        }
    }
}

/// Codes one quantized component: DC differences, then interleaved runs and levels.
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
    for (pos, &value) in quantized.iter().enumerate().skip(blocks) {
        if value == 0 {
            continue;
        }
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

#[inline]
fn signed_code(value: i32) -> u32 {
    if value >= 0 {
        2 * value.unsigned_abs()
    } else {
        2 * value.unsigned_abs() - 1
    }
}

/// Codes `values` as RDD 36 alpha: differences from the last value, with runs.
pub(crate) fn encode_alpha(values: &[u16], sixteen: bool) -> Vec<u8> {
    let (full, short, reach) = if sixteen { (16, 7, 64) } else { (8, 4, 8) };
    let mask: i32 = (1 << full) - 1;
    let mut previous = mask;
    let mut writer = BitWriter::default();
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
    writer.finish()
}

/// Bytes a coded component takes.
#[inline]
pub(crate) fn bytes_of(bits: u32) -> u32 {
    bits.div_ceil(8)
}

/// Scratch for one worker.
#[derive(Default)]
pub(crate) struct Work {
    pub quantized: Vec<i32>,
}

/// Slice bytes without alpha at quantization index `index`, header included.
pub(crate) fn slice_bytes(
    prepared: &Prepared,
    luma: &[f32; 64],
    chroma: &[f32; 64],
    header: u32,
    work: &mut Work,
) -> u32 {
    let mut total = header;
    for (coefs, blocks, steps) in [
        (&prepared.y, prepared.luma_blocks, luma),
        (&prepared.u, prepared.chroma_blocks, chroma),
        (&prepared.v, prepared.chroma_blocks, chroma),
    ] {
        quantize(coefs, blocks, steps, &mut work.quantized);
        let mut counter = Counter::default();
        code_component(&mut counter, &work.quantized, blocks);
        total += bytes_of(counter.bits);
    }
    total
}

/// The whole coded slice at quantization index `index`.
pub(crate) fn write_slice(
    prepared: &Prepared,
    index: u8,
    luma: &[f32; 64],
    chroma: &[f32; 64],
    work: &mut Work,
) -> Vec<u8> {
    let alpha = !prepared.alpha.is_empty();
    let header_size: u8 = if alpha { 8 } else { 6 };
    let mut parts = Vec::with_capacity(3);
    for (coefs, blocks, steps) in [
        (&prepared.y, prepared.luma_blocks, luma),
        (&prepared.u, prepared.chroma_blocks, chroma),
        (&prepared.v, prepared.chroma_blocks, chroma),
    ] {
        quantize(coefs, blocks, steps, &mut work.quantized);
        let mut writer = BitWriter::default();
        code_component(&mut writer, &work.quantized, blocks);
        parts.push(writer.finish());
    }
    let body: usize = parts.iter().map(Vec::len).sum::<usize>() + prepared.alpha.len();
    let mut out = Vec::with_capacity(usize::from(header_size) + body);
    out.push(header_size << 3);
    out.push(index);
    out.extend_from_slice(&(parts[0].len() as u16).to_be_bytes());
    out.extend_from_slice(&(parts[1].len() as u16).to_be_bytes());
    if alpha {
        out.extend_from_slice(&(parts[2].len() as u16).to_be_bytes());
    }
    for part in &parts {
        out.extend_from_slice(part);
    }
    out.extend_from_slice(&prepared.alpha);
    out
}

#[cfg(test)]
mod tests {
    use super::{Counter, Sink, code_component, encode_alpha};
    use crate::bits::BitWriter;

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
            let bytes = encode_alpha(&coded, sixteen);
            let mut out = Vec::new();
            crate::decode::unpack_alpha_for_tests(&bytes, sixteen, coded.len(), &mut out);
            assert_eq!(out, coded, "sixteen {sixteen}");
        }
    }
}

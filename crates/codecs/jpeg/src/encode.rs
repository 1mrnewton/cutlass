//! Baseline JPEG encoder.
//!
//! Color is 4:2:0 and grey is 4:4:4. Quality 1..=100 scales the Annex K
//! quantizers the way the Independent JPEG Group does. `extra` empty is
//! quality 75. A single byte is that quality. Bitrate is not a JPEG mode.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_encode::{
    AudioInput, BitDepth, CodecError, CodecId, CodecParams, ColorType, Encoder, FrameInput, Packet,
    PictureInput, StillImage,
};

use crate::color::rgb_to_ycbcr;
use crate::dct::{fdct, quantize};
use crate::huffman::{HuffEncode, encode_table};
use crate::limits::dimensions_ok;
use crate::tables::{
    AC_CHROMA_COUNTS, AC_CHROMA_SYMBOLS, AC_LUMA_COUNTS, AC_LUMA_SYMBOLS, CHROMINANCE_QUANT,
    DC_CHROMA_COUNTS, DC_CHROMA_SYMBOLS, DC_LUMA_COUNTS, DC_LUMA_SYMBOLS, LUMINANCE_QUANT, ZIGZAG,
    quality_scale, scale_quant,
};

struct Bits {
    out: Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl Bits {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            acc: 0,
            nbits: 0,
        }
    }

    fn write(&mut self, value: u32, len: u32) -> Result<(), CodecError> {
        if len > 24 {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "coefficient",
            });
        }
        self.acc = (self.acc << len) | u64::from(value);
        self.nbits += len;
        while self.nbits >= 8 {
            self.nbits -= 8;
            let byte = u8::try_from((self.acc >> self.nbits) & 0xff).unwrap_or(0);
            self.acc &= (1u64 << self.nbits).saturating_sub(1);
            self.out.push(byte);
            if byte == 0xff {
                self.out.push(0);
            }
        }
        Ok(())
    }

    fn flush(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            let _ = self.write((1u32 << pad) - 1, pad);
        }
        self.out
    }
}

pub struct JpegEncoder {
    quality: u8,
    cancel: Arc<AtomicBool>,
}

impl JpegEncoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Jpeg {
            return Err(CodecError::Unsupported { what: "jpeg" });
        }
        let quality = match params.extra.as_slice() {
            [] => 75,
            [quality] if (1..=100).contains(quality) => *quality,
            _ => {
                return Err(CodecError::Unsupported {
                    what: "jpeg quality",
                });
            }
        };
        Ok(Self {
            quality,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }
}

impl Encoder for JpegEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Jpeg
    }

    fn push(&mut self, _input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::InvalidApi {
            what: "jpeg encodes a picture",
        })
    }

    fn push_frame(&mut self, _input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::Unsupported { what: "frame" })
    }

    fn push_picture(&mut self, input: &PictureInput) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let data = encode_jpeg(&input.image, self.quality, &self.cancel)?;
        Ok(vec![Packet {
            stream: 0,
            data,
            pts: 0,
            duration: 0,
            timescale: 0,
            keyframe: true,
            side: Vec::new(),
        }])
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.cancel.store(false, Ordering::Relaxed);
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

pub(crate) fn encode_jpeg(
    image: &StillImage,
    quality: u8,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, CodecError> {
    image.validate().map_err(|_| CodecError::InvalidApi {
        what: "jpeg picture",
    })?;
    if !dimensions_ok(image.width, image.height) {
        return Err(CodecError::Limit { what: "jpeg size" });
    }
    let color = match (image.color, image.depth) {
        (ColorType::Gray, BitDepth::Eight) => false,
        (ColorType::Rgb, BitDepth::Eight) => true,
        (ColorType::Rgba | ColorType::GrayAlpha, _) => {
            return Err(CodecError::Unsupported { what: "jpeg alpha" });
        }
        (ColorType::Indexed, _) => {
            return Err(CodecError::Unsupported {
                what: "jpeg indexed",
            });
        }
        (_, _) => {
            return Err(CodecError::Unsupported {
                what: "jpeg precision",
            });
        }
    };
    let scale = quality_scale(quality);
    let luma_q = scale_quant(&LUMINANCE_QUANT, scale);
    let chroma_q = scale_quant(&CHROMINANCE_QUANT, scale);
    let dc_luma = encode_table(&DC_LUMA_COUNTS, &DC_LUMA_SYMBOLS)?;
    let ac_luma = encode_table(&AC_LUMA_COUNTS, &AC_LUMA_SYMBOLS)?;
    let dc_chroma = encode_table(&DC_CHROMA_COUNTS, &DC_CHROMA_SYMBOLS)?;
    let ac_chroma = encode_table(&AC_CHROMA_COUNTS, &AC_CHROMA_SYMBOLS)?;

    let width = image.width;
    let height = image.height;
    let planes = planes(image, color)?;
    let mut file = Vec::new();
    file.extend_from_slice(&[0xff, 0xd8]);
    file.extend(jfif());
    file.extend(dqt(0, &luma_q));
    if color {
        file.extend(dqt(1, &chroma_q));
    }
    file.extend(sof0(width, height, color));
    file.extend(dht(0x00, &dc_luma));
    file.extend(dht(0x10, &ac_luma));
    if color {
        file.extend(dht(0x01, &dc_chroma));
        file.extend(dht(0x11, &ac_chroma));
    }
    file.extend(sos(color));

    let mut entropy = Bits::new();
    let mut pred = [0i32; 3];
    if color {
        let mcu_cols = usize::try_from(width.div_ceil(16)).unwrap_or(0);
        let mcu_rows = usize::try_from(height.div_ceil(16)).unwrap_or(0);
        for mcu_y in 0..mcu_rows {
            if cancel.load(Ordering::Relaxed) {
                return Err(CodecError::Cancelled);
            }
            for mcu_x in 0..mcu_cols {
                for local_y in 0..2 {
                    for local_x in 0..2 {
                        let block = luma_block(
                            &planes.y,
                            width,
                            height,
                            mcu_x * 16 + local_x * 8,
                            mcu_y * 16 + local_y * 8,
                        );
                        write_block(
                            &mut entropy,
                            &block,
                            &luma_q,
                            &mut pred[0],
                            &dc_luma,
                            &ac_luma,
                        )?;
                    }
                }
                let cb = chroma_block(&planes.cb, width, height, mcu_x * 8, mcu_y * 8);
                let cr = chroma_block(&planes.cr, width, height, mcu_x * 8, mcu_y * 8);
                write_block(
                    &mut entropy,
                    &cb,
                    &chroma_q,
                    &mut pred[1],
                    &dc_chroma,
                    &ac_chroma,
                )?;
                write_block(
                    &mut entropy,
                    &cr,
                    &chroma_q,
                    &mut pred[2],
                    &dc_chroma,
                    &ac_chroma,
                )?;
            }
        }
    } else {
        let cols = usize::try_from(width.div_ceil(8)).unwrap_or(0);
        let rows = usize::try_from(height.div_ceil(8)).unwrap_or(0);
        for row in 0..rows {
            if cancel.load(Ordering::Relaxed) {
                return Err(CodecError::Cancelled);
            }
            for col in 0..cols {
                let block = luma_block(&planes.y, width, height, col * 8, row * 8);
                write_block(
                    &mut entropy,
                    &block,
                    &luma_q,
                    &mut pred[0],
                    &dc_luma,
                    &ac_luma,
                )?;
            }
        }
    }
    file.extend(entropy.flush());
    file.extend_from_slice(&[0xff, 0xd9]);
    Ok(file)
}

struct Planes {
    y: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
}

fn planes(image: &StillImage, color: bool) -> Result<Planes, CodecError> {
    let width = usize::try_from(image.width).unwrap_or(0);
    let height = usize::try_from(image.height).unwrap_or(0);
    let pixels = width.checked_mul(height).ok_or(CodecError::Limit {
        what: "jpeg picture",
    })?;
    let mut y = vec![0u8; pixels];
    let mut cb = vec![0u8; pixels];
    let mut cr = vec![0u8; pixels];
    if color {
        for index in 0..pixels {
            let base = index * 3;
            let red = i32::from(u8::try_from(image.samples[base]).unwrap_or(0));
            let green = i32::from(u8::try_from(image.samples[base + 1]).unwrap_or(0));
            let blue = i32::from(u8::try_from(image.samples[base + 2]).unwrap_or(0));
            let (yy, ccb, ccr) = rgb_to_ycbcr(red, green, blue);
            y[index] = yy;
            cb[index] = ccb;
            cr[index] = ccr;
        }
    } else {
        for (slot, sample) in y.iter_mut().zip(&image.samples) {
            *slot = u8::try_from(*sample).unwrap_or(0);
        }
    }
    Ok(Planes { y, cb, cr })
}

fn luma_block(
    plane: &[u8],
    width: u32,
    height: u32,
    origin_x: usize,
    origin_y: usize,
) -> [f64; 64] {
    let mut block = [0.0; 64];
    let width = usize::try_from(width).unwrap_or(0);
    let height = usize::try_from(height).unwrap_or(0);
    for row in 0..8 {
        for col in 0..8 {
            let x = (origin_x + col).min(width.saturating_sub(1));
            let y = (origin_y + row).min(height.saturating_sub(1));
            let sample = plane.get(y * width + x).copied().unwrap_or(0);
            block[row * 8 + col] = f64::from(sample) - 128.0;
        }
    }
    block
}

fn chroma_block(
    plane: &[u8],
    width: u32,
    height: u32,
    origin_x: usize,
    origin_y: usize,
) -> [f64; 64] {
    let mut block = [0.0; 64];
    let width = usize::try_from(width).unwrap_or(1).max(1);
    let height = usize::try_from(height).unwrap_or(1).max(1);
    for row in 0..8 {
        for col in 0..8 {
            let mut sum = 0u32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let x = ((origin_x + col) * 2 + dx).min(width - 1);
                    let y = ((origin_y + row) * 2 + dy).min(height - 1);
                    sum += u32::from(plane.get(y * width + x).copied().unwrap_or(128));
                }
            }
            block[row * 8 + col] = f64::from(sum / 4) - 128.0;
        }
    }
    block
}

fn write_block(
    bits: &mut Bits,
    samples: &[f64; 64],
    quant: &[u16; 64],
    pred: &mut i32,
    dc: &HuffEncode,
    ac: &HuffEncode,
) -> Result<(), CodecError> {
    let coeffs = fdct(samples);
    let mut zig = [0i32; 64];
    for k in 0..64 {
        let natural = ZIGZAG[k];
        zig[k] = quantize(coeffs[natural], quant[natural]);
    }
    let diff = zig[0] - *pred;
    *pred = zig[0];
    write_magnitude(bits, diff, dc)?;
    let mut run = 0u32;
    let mut pending = false;
    for coeff in zig.iter().skip(1) {
        if *coeff == 0 {
            run += 1;
            pending = true;
            continue;
        }
        while run >= 16 {
            write_symbol(bits, 0xf0, ac)?;
            run -= 16;
        }
        write_magnitude_run(bits, run, *coeff, ac)?;
        run = 0;
        pending = false;
    }
    if pending {
        write_symbol(bits, 0x00, ac)?;
    }
    Ok(())
}

fn write_magnitude(bits: &mut Bits, value: i32, table: &HuffEncode) -> Result<(), CodecError> {
    let (size, extra) = category(value)?;
    write_symbol(bits, u32::from(size), table)?;
    if size > 0 {
        bits.write(extra, u32::from(size))?;
    }
    Ok(())
}

fn write_magnitude_run(
    bits: &mut Bits,
    run: u32,
    value: i32,
    table: &HuffEncode,
) -> Result<(), CodecError> {
    let (size, extra) = category(value)?;
    let symbol = (run << 4) | u32::from(size);
    write_symbol(bits, symbol, table)?;
    if size > 0 {
        bits.write(extra, u32::from(size))?;
    }
    Ok(())
}

fn write_symbol(bits: &mut Bits, symbol: u32, table: &HuffEncode) -> Result<(), CodecError> {
    let Some(index) = usize::try_from(symbol).ok() else {
        return Err(CodecError::Limit {
            what: "jpeg coefficient",
        });
    };
    let len = *table.len.get(index).unwrap_or(&0);
    if len == 0 {
        return Err(CodecError::Limit {
            what: "jpeg coefficient",
        });
    }
    bits.write(u32::from(table.code[index]), u32::from(len))
}

fn category(value: i32) -> Result<(u8, u32), CodecError> {
    if value == 0 {
        return Ok((0, 0));
    }
    let magnitude = value.unsigned_abs();
    let size = 32 - magnitude.leading_zeros();
    if size > 11 {
        return Err(CodecError::Limit {
            what: "jpeg coefficient",
        });
    }
    let bits = if value < 0 {
        u32::try_from(i64::from(value) + (1i64 << size) - 1).unwrap_or(0)
    } else {
        magnitude
    };
    Ok((u8::try_from(size).unwrap_or(0), bits))
}

fn jfif() -> Vec<u8> {
    let mut bytes = vec![0xff, 0xe0, 0x00, 0x10];
    bytes.extend_from_slice(b"JFIF\0");
    bytes.extend_from_slice(&[1, 1, 0, 0, 1, 0, 1, 0, 0]);
    bytes
}

fn dqt(id: u8, quant: &[u16; 64]) -> Vec<u8> {
    let mut body = Vec::with_capacity(67);
    body.push(id);
    for k in 0..64 {
        body.push(u8::try_from(quant[ZIGZAG[k]]).unwrap_or(1));
    }
    segment(0xdb, &body)
}

fn sof0(width: u32, height: u32, color: bool) -> Vec<u8> {
    let mut body = vec![8];
    body.extend(u16::try_from(height).unwrap_or(0).to_be_bytes());
    body.extend(u16::try_from(width).unwrap_or(0).to_be_bytes());
    if color {
        body.extend([3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]);
    } else {
        body.extend([1, 1, 0x11, 0]);
    }
    segment(0xc0, &body)
}

fn dht(class_id: u8, table: &HuffEncode) -> Vec<u8> {
    let mut body = Vec::new();
    body.push(class_id);
    body.extend_from_slice(&table.counts);
    body.extend_from_slice(&table.symbols);
    segment(0xc4, &body)
}

fn sos(color: bool) -> Vec<u8> {
    let body = if color {
        vec![3, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0]
    } else {
        vec![1, 1, 0x00, 0, 63, 0]
    };
    segment(0xda, &body)
}

fn segment(marker: u8, body: &[u8]) -> Vec<u8> {
    let len = u16::try_from(body.len() + 2).unwrap_or(u16::MAX);
    let mut bytes = vec![0xff, marker];
    bytes.extend(len.to_be_bytes());
    bytes.extend_from_slice(body);
    bytes
}

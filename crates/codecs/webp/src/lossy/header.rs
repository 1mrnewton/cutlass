//! The key-frame header of RFC 6386 sections 9 and 19.2.

use cutlass_decode::CodecError;

use super::bool_decoder::BoolDecoder;
use super::tables::{AC_Q, COEFF_UPDATE_PROBS, DC_Q, DEFAULT_COEFF_PROBS};
use crate::limits::picture_ok;

/// Token probabilities: `[block type][band][context][node]`.
pub(crate) type CoeffProbs = [[[[u8; 11]; 3]; 8]; 4];

/// Dequantization factors of one segment, `[DC, AC]`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Quant {
    pub y1: [i32; 2],
    pub y2: [i32; 2],
    pub uv: [i32; 2],
}

/// Loop-filter settings of one segment and macroblock kind.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FilterStrength {
    /// Edge limit before the `2 * limit + 1` of the edge test; 0 is off.
    pub limit: i32,
    pub interior: i32,
    pub hev: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FilterKind {
    None,
    Simple,
    Normal,
}

pub(crate) struct FrameHeader<'a> {
    pub width: u32,
    pub height: u32,
    /// Partition 0: the rest of the header and the modes.
    pub first: BoolDecoder<'a>,
    pub partitions: Vec<&'a [u8]>,
    pub update_segment_map: bool,
    pub segment_probs: [u8; 3],
    pub quant: [Quant; 4],
    pub filter: FilterKind,
    /// `[segment][is 4x4]`.
    pub strengths: [[FilterStrength; 2]; 4],
    pub coeff_probs: CoeffProbs,
    pub skip_prob: Option<u8>,
}

fn malformed(what: &'static str) -> CodecError {
    CodecError::Malformed { at_bit: 0, what }
}

/// Width and height from the ten-byte key-frame start, without decoding.
fn dimensions(data: &[u8]) -> Result<(u32, u32), CodecError> {
    if data.len() < 10 {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "vp8 frame header",
        });
    }
    let tag = u32::from(data[0]) | (u32::from(data[1]) << 8) | (u32::from(data[2]) << 16);
    if tag & 1 != 0 {
        return Err(CodecError::Unsupported {
            what: "vp8 inter frame in webp",
        });
    }
    if (tag >> 1) & 7 > 3 {
        return Err(malformed("vp8 profile"));
    }
    if (tag >> 4) & 1 == 0 {
        return Err(CodecError::Unsupported {
            what: "vp8 hidden frame",
        });
    }
    if data[3..6] != [0x9d, 0x01, 0x2a] {
        return Err(malformed("vp8 start code"));
    }
    let width = u32::from(u16::from_le_bytes([data[6], data[7]]) & 0x3fff);
    let height = u32::from(u16::from_le_bytes([data[8], data[9]]) & 0x3fff);
    picture_ok(width, height)?;
    Ok((width, height))
}

pub(crate) fn parse(data: &[u8]) -> Result<FrameHeader<'_>, CodecError> {
    let (width, height) = dimensions(data)?;
    let tag = u32::from(data[0]) | (u32::from(data[1]) << 8) | (u32::from(data[2]) << 16);
    let first_size = (tag >> 5) as usize;
    let rest = &data[10..];
    // libwebp refuses a first partition that fills the whole chunk.
    if first_size >= data.len() || first_size > rest.len() {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "vp8 first partition",
        });
    }
    let (first_bytes, tokens) = rest.split_at(first_size);
    let mut br = BoolDecoder::new(first_bytes);
    // Colour space and clamping type: both have one meaning in WebP.
    br.flag();
    br.flag();

    // Section 9.3: segmentation.
    let segmented = br.flag();
    let mut update_segment_map = false;
    // libwebp starts from absolute values when a key frame enables
    // segments without sending them.
    let mut absolute = true;
    let mut segment_quant = [0i32; 4];
    let mut segment_filter = [0i32; 4];
    let mut segment_probs = [255u8; 3];
    if segmented {
        update_segment_map = br.flag();
        if br.flag() {
            absolute = br.flag();
            for value in &mut segment_quant {
                *value = br.optional_signed(7);
            }
            for value in &mut segment_filter {
                *value = br.optional_signed(6);
            }
        }
        if update_segment_map {
            for prob in &mut segment_probs {
                *prob = if br.flag() {
                    // An 8-bit literal.
                    br.literal(8) as u8
                } else {
                    255
                };
            }
        }
    }

    // Section 9.6: the loop filter.
    let simple = br.flag();
    let level = br.literal(6).cast_signed();
    let sharpness = br.literal(3).cast_signed();
    let mut ref_delta = 0;
    let mut mode_delta = 0;
    if br.flag() && br.flag() {
        for index in 0..4 {
            let delta = br.optional_signed(6);
            if index == 0 {
                ref_delta = delta;
            }
        }
        for index in 0..4 {
            let delta = br.optional_signed(6);
            if index == 0 {
                mode_delta = delta;
            }
        }
    }
    if br.eof() {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "vp8 filter header",
        });
    }
    let filter = match (level, simple) {
        (0, _) => FilterKind::None,
        (_, true) => FilterKind::Simple,
        (_, false) => FilterKind::Normal,
    };

    // Section 9.5: token partitions.
    let count = 1usize << br.literal(2);
    let partitions = partitions(tokens, count)?;

    // Section 9.6: quantizers.
    let base = br.literal(7).cast_signed();
    let y1_dc = br.optional_signed(4);
    let y2_dc = br.optional_signed(4);
    let y2_ac = br.optional_signed(4);
    let uv_dc = br.optional_signed(4);
    let uv_ac = br.optional_signed(4);
    let mut quant = [Quant::default(); 4];
    for (segment, entry) in quant.iter_mut().enumerate() {
        let q = if segmented {
            segment_quant[segment] + if absolute { 0 } else { base }
        } else {
            base
        };
        let step = |table: &[i16; 128], delta: i32, max: i32| {
            i32::from(table[(q + delta).clamp(0, max).unsigned_abs() as usize])
        };
        entry.y1 = [step(&DC_Q, y1_dc, 127), step(&AC_Q, 0, 127)];
        entry.y2 = [
            step(&DC_Q, y2_dc, 127) * 2,
            ((step(&AC_Q, y2_ac, 127) * 101_581) >> 16).max(8),
        ];
        entry.uv = [step(&DC_Q, uv_dc, 117), step(&AC_Q, uv_ac, 127)];
    }

    // Key frames always refresh the probabilities; the flag is unused.
    br.flag();

    // Section 13.4: token probability updates.
    let mut coeff_probs = DEFAULT_COEFF_PROBS;
    for (plane, updates) in coeff_probs.iter_mut().zip(&COEFF_UPDATE_PROBS) {
        for (band, band_updates) in plane.iter_mut().zip(updates) {
            for (context, context_updates) in band.iter_mut().zip(band_updates) {
                for (prob, &update) in context.iter_mut().zip(context_updates) {
                    if br.bit(update) {
                        *prob = br.literal(8) as u8;
                    }
                }
            }
        }
    }
    let skip_prob = if br.flag() {
        Some(br.literal(8) as u8)
    } else {
        None
    };

    let mut strengths = [[FilterStrength::default(); 2]; 4];
    if filter != FilterKind::None {
        for (segment, kinds) in strengths.iter_mut().enumerate() {
            let base_level = if segmented {
                segment_filter[segment] + if absolute { 0 } else { level }
            } else {
                level
            };
            for (is_4x4, strength) in kinds.iter_mut().enumerate() {
                let mut level = base_level + ref_delta;
                if is_4x4 == 1 {
                    level += mode_delta;
                }
                *strength = filter_strength(level.clamp(0, 63), sharpness);
            }
        }
    }

    Ok(FrameHeader {
        width,
        height,
        first: br,
        partitions,
        update_segment_map,
        segment_probs,
        quant,
        filter,
        strengths,
        coeff_probs,
        skip_prob,
    })
}

/// Section 15.2: limits for one filter level.
fn filter_strength(level: i32, sharpness: i32) -> FilterStrength {
    if level == 0 {
        return FilterStrength::default();
    }
    let mut interior = level;
    if sharpness > 0 {
        interior >>= if sharpness > 4 { 2 } else { 1 };
        interior = interior.min(9 - sharpness);
    }
    let interior = interior.max(1);
    FilterStrength {
        limit: 2 * level + interior,
        interior,
        hev: if level >= 40 {
            2
        } else if level >= 15 {
            1
        } else {
            0
        },
    }
}

/// Split the token partitions. A declared size past the end is refused
/// (libwebp clamps it), and the last partition must not be empty.
fn partitions(data: &[u8], count: usize) -> Result<Vec<&[u8]>, CodecError> {
    let sizes_len = 3 * (count - 1);
    if data.len() < sizes_len {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "vp8 partition sizes",
        });
    }
    let (sizes, mut rest) = data.split_at(sizes_len);
    let mut parts = Vec::with_capacity(count);
    for size in sizes.chunks(3) {
        let size =
            usize::from(size[0]) | (usize::from(size[1]) << 8) | (usize::from(size[2]) << 16);
        if size > rest.len() {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "vp8 partition",
            });
        }
        let (part, tail) = rest.split_at(size);
        parts.push(part);
        rest = tail;
    }
    if rest.is_empty() {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "vp8 last partition",
        });
    }
    parts.push(rest);
    Ok(parts)
}

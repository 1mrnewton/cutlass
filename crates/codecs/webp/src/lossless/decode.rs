//! VP8L decoder (RFC 9649 section 3, "Specification for WebP Lossless
//! Bitstream").
//!
//! Pixels are ARGB words, alpha in the high byte, the layout the format
//! specifies. Transforms are read in order and undone in reverse. Each
//! transform type may appear once. Every entropy-coded image is bounded by
//! [`crate::limits`] before its buffer is allocated, and every decoded
//! symbol produces at least one pixel, so decoding work is linear in the
//! pixel count.

use cutlass_decode::CodecError;

use super::bits::BitReader;
use super::huffman::{Code, MAX_CODE_LENGTH};
use super::transform::{Transform, inverse};
use crate::limits::{MAX_TABLE_ENTRIES, picture_ok, pixel_count};

/// First byte of every VP8L bitstream.
pub(crate) const SIGNATURE: u8 = 0x2f;

/// Literal green values, then length prefixes, then color-cache indices.
pub(crate) const NUM_LITERALS: usize = 256;
pub(crate) const NUM_LENGTH_CODES: usize = 24;
pub(crate) const NUM_DISTANCE_CODES: usize = 40;
pub(crate) const MAX_CACHE_BITS: u32 = 11;

/// Order in which code-length code lengths are stored.
pub(crate) const CODE_LENGTH_ORDER: [usize; 19] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

/// The 120 short distance codes as (dx, dy), RFC 9649 section 3.6.
/// Distance code `n` names the pixel `dx + dy * width` back.
pub(crate) const DISTANCE_MAP: [(i8, i8); 120] = [
    (0, 1),
    (1, 0),
    (1, 1),
    (-1, 1),
    (0, 2),
    (2, 0),
    (1, 2),
    (-1, 2),
    (2, 1),
    (-2, 1),
    (2, 2),
    (-2, 2),
    (0, 3),
    (3, 0),
    (1, 3),
    (-1, 3),
    (3, 1),
    (-3, 1),
    (2, 3),
    (-2, 3),
    (3, 2),
    (-3, 2),
    (0, 4),
    (4, 0),
    (1, 4),
    (-1, 4),
    (4, 1),
    (-4, 1),
    (3, 3),
    (-3, 3),
    (2, 4),
    (-2, 4),
    (4, 2),
    (-4, 2),
    (0, 5),
    (3, 4),
    (-3, 4),
    (4, 3),
    (-4, 3),
    (5, 0),
    (1, 5),
    (-1, 5),
    (5, 1),
    (-5, 1),
    (2, 5),
    (-2, 5),
    (5, 2),
    (-5, 2),
    (4, 4),
    (-4, 4),
    (3, 5),
    (-3, 5),
    (5, 3),
    (-5, 3),
    (0, 6),
    (6, 0),
    (1, 6),
    (-1, 6),
    (6, 1),
    (-6, 1),
    (2, 6),
    (-2, 6),
    (6, 2),
    (-6, 2),
    (4, 5),
    (-4, 5),
    (5, 4),
    (-5, 4),
    (3, 6),
    (-3, 6),
    (6, 3),
    (-6, 3),
    (0, 7),
    (7, 0),
    (1, 7),
    (-1, 7),
    (5, 5),
    (-5, 5),
    (7, 1),
    (-7, 1),
    (4, 6),
    (-4, 6),
    (6, 4),
    (-6, 4),
    (2, 7),
    (-2, 7),
    (7, 2),
    (-7, 2),
    (3, 7),
    (-3, 7),
    (7, 3),
    (-7, 3),
    (5, 6),
    (-5, 6),
    (6, 5),
    (-6, 5),
    (8, 0),
    (4, 7),
    (-4, 7),
    (7, 4),
    (-7, 4),
    (8, 1),
    (8, 2),
    (6, 6),
    (-6, 6),
    (8, 3),
    (5, 7),
    (-5, 7),
    (7, 5),
    (-7, 5),
    (8, 4),
    (6, 7),
    (-6, 7),
    (7, 6),
    (-7, 6),
    (8, 5),
    (7, 7),
    (-7, 7),
    (8, 6),
    (8, 7),
];

/// Multiplier of the color-cache hash.
pub(crate) const CACHE_HASH: u32 = 0x1e35_a7bd;

/// Header of a VP8L bitstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Header {
    pub width: u32,
    pub height: u32,
    /// The `alpha_is_used` hint.
    pub alpha: bool,
}

/// Read the five-byte header, without decoding the image.
pub(crate) fn header(data: &[u8]) -> Result<Header, CodecError> {
    if data.len() < 5 {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "vp8l header",
        });
    }
    if data[0] != SIGNATURE {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "vp8l signature",
        });
    }
    let bits = u32::from_le_bytes([data[1], data[2], data[3], data[4]]);
    let version = bits >> 29;
    if version != 0 {
        return Err(CodecError::Malformed {
            at_bit: 37,
            what: "vp8l version",
        });
    }
    Ok(Header {
        width: (bits & 0x3fff) + 1,
        height: ((bits >> 14) & 0x3fff) + 1,
        alpha: (bits >> 28) & 1 == 1,
    })
}

/// Decode a complete VP8L bitstream into ARGB words, row by row.
pub(crate) fn decode(data: &[u8]) -> Result<(Header, Vec<u32>), CodecError> {
    let head = header(data)?;
    picture_ok(head.width, head.height)?;
    let mut reader = BitReader::new(data);
    reader.skip(40)?;
    let pixels = image_stream(&mut reader, head.width, head.height)?;
    Ok((head, pixels))
}

/// Transforms, then the spatially coded image, then the inverse transforms.
/// A headerless image stream of `width` x `height`, as an `ALPH` chunk
/// carries it (RFC 9649 section 2.7.1.2).
pub(crate) fn decode_stream(data: &[u8], width: u32, height: u32) -> Result<Vec<u32>, CodecError> {
    picture_ok(width, height)?;
    let mut reader = BitReader::new(data);
    image_stream(&mut reader, width, height)
}

fn image_stream(
    reader: &mut BitReader<'_>,
    width: u32,
    height: u32,
) -> Result<Vec<u32>, CodecError> {
    let mut transforms: Vec<Transform> = Vec::new();
    let mut seen = [false; 4];
    let mut coded_width = width;
    while reader.flag()? {
        let at_bit = reader.position();
        let kind = reader.read(2)? as usize;
        if seen[kind] {
            return Err(CodecError::Malformed {
                at_bit,
                what: "vp8l transform repeated",
            });
        }
        seen[kind] = true;
        let transform = read_transform(reader, kind, coded_width, height)?;
        if let Transform::ColorIndexing { bits, .. } = &transform {
            coded_width = subsample(coded_width, *bits);
        }
        transforms.push(transform);
    }
    let mut pixels = entropy_image(reader, coded_width, height, true)?;
    for transform in transforms.iter().rev() {
        pixels = inverse(transform, pixels, height)?;
    }
    Ok(pixels)
}

fn read_transform(
    reader: &mut BitReader<'_>,
    kind: usize,
    width: u32,
    height: u32,
) -> Result<Transform, CodecError> {
    match kind {
        0 | 1 => {
            let bits = reader.read(3)? + 2;
            let data = entropy_image(
                reader,
                subsample(width, bits),
                subsample(height, bits),
                false,
            )?;
            Ok(if kind == 0 {
                Transform::Predictor { bits, data, width }
            } else {
                Transform::Color { bits, data, width }
            })
        }
        2 => Ok(Transform::SubtractGreen),
        _ => {
            let size = reader.read(8)? + 1;
            let mut palette = entropy_image(reader, size, 1, false)?;
            for index in 1..palette.len() {
                palette[index] = add_pixels(palette[index], palette[index - 1]);
            }
            let bits = match size {
                0..=2 => 3,
                3..=4 => 2,
                5..=16 => 1,
                _ => 0,
            };
            Ok(Transform::ColorIndexing {
                bits,
                palette,
                width,
            })
        }
    }
}

/// `size` divided by `2^bits`, rounded up.
pub(crate) fn subsample(size: u32, bits: u32) -> u32 {
    (size + (1 << bits) - 1) >> bits
}

/// Per-channel sum modulo 256.
#[inline]
pub(crate) fn add_pixels(a: u32, b: u32) -> u32 {
    let alpha_green = (a & 0xff00_ff00).wrapping_add(b & 0xff00_ff00);
    let red_blue = (a & 0x00ff_00ff).wrapping_add(b & 0x00ff_00ff);
    (alpha_green & 0xff00_ff00) | (red_blue & 0x00ff_00ff)
}

/// The five prefix codes of one group: green with lengths and cache
/// indices, red, blue, alpha, distance.
struct Group {
    green: Code,
    red: Code,
    blue: Code,
    alpha: Code,
    distance: Code,
}

/// Decode one entropy-coded image of `width × height` pixels. Only the main
/// image (`main`) may carry an entropy image of prefix-code groups.
fn entropy_image(
    reader: &mut BitReader<'_>,
    width: u32,
    height: u32,
    main: bool,
) -> Result<Vec<u32>, CodecError> {
    let count = pixel_count(width, height)?;
    let cache_bits = if reader.flag()? {
        let at_bit = reader.position();
        let bits = reader.read(4)?;
        if !(1..=MAX_CACHE_BITS).contains(&bits) {
            return Err(CodecError::Malformed {
                at_bit,
                what: "vp8l color cache size",
            });
        }
        bits
    } else {
        0
    };
    let mut meta_bits = 0;
    let mut meta_width = 0;
    let mut meta: Vec<u32> = Vec::new();
    let mut group_count = 1usize;
    if main && reader.flag()? {
        meta_bits = reader.read(3)? + 2;
        meta_width = subsample(width, meta_bits);
        meta = entropy_image(reader, meta_width, subsample(height, meta_bits), false)?;
        let mut largest = 0u32;
        for pixel in &mut meta {
            *pixel = (*pixel >> 8) & 0xffff;
            largest = largest.max(*pixel);
        }
        group_count = largest as usize + 1;
    }
    let cache_size = if cache_bits > 0 {
        1usize << cache_bits
    } else {
        0
    };
    let mut groups = Vec::with_capacity(group_count.min(1024));
    let mut entries = 0usize;
    for _ in 0..group_count {
        let group = Group {
            green: read_code(reader, NUM_LITERALS + NUM_LENGTH_CODES + cache_size)?,
            red: read_code(reader, NUM_LITERALS)?,
            blue: read_code(reader, NUM_LITERALS)?,
            alpha: read_code(reader, NUM_LITERALS)?,
            distance: read_code(reader, NUM_DISTANCE_CODES)?,
        };
        entries += group.green.size()
            + group.red.size()
            + group.blue.size()
            + group.alpha.size()
            + group.distance.size();
        if entries > MAX_TABLE_ENTRIES {
            return Err(CodecError::Limit {
                what: "vp8l prefix tables",
            });
        }
        groups.push(group);
    }
    let mut pixels = vec![0u32; count];
    let mut cache = vec![0u32; cache_size];
    let cache_shift = 32 - cache_bits;
    let width = width as usize;
    let meta_mask = if meta_bits > 0 {
        (1usize << meta_bits) - 1
    } else {
        usize::MAX
    };
    let mut pos = 0usize;
    let mut col = 0usize;
    let mut row = 0usize;
    let mut group = &groups[0];
    let mut refresh = !meta.is_empty();
    while pos < count {
        if refresh || (meta_bits > 0 && col & meta_mask == 0) {
            if !meta.is_empty() {
                let tile = (row >> meta_bits) * meta_width as usize + (col >> meta_bits);
                group = &groups[meta[tile] as usize];
            }
            refresh = false;
        }
        let at_bit = reader.position();
        let symbol = usize::from(group.green.read(reader)?);
        if symbol < NUM_LITERALS {
            let red = u32::from(group.red.read(reader)?);
            let blue = u32::from(group.blue.read(reader)?);
            let alpha = u32::from(group.alpha.read(reader)?);
            let argb = (alpha << 24) | (red << 16) | ((symbol as u32) << 8) | blue;
            pixels[pos] = argb;
            if cache_bits > 0 {
                cache[(argb.wrapping_mul(CACHE_HASH) >> cache_shift) as usize] = argb;
            }
            pos += 1;
            col += 1;
            if col == width {
                col = 0;
                row += 1;
            }
        } else if symbol < NUM_LITERALS + NUM_LENGTH_CODES {
            let length = prefix_value(reader, (symbol - NUM_LITERALS) as u32)? as usize;
            let distance_symbol = u32::from(group.distance.read(reader)?);
            let code = prefix_value(reader, distance_symbol)? as usize;
            let distance = plane_distance(width, code);
            if distance > pos || count - pos < length {
                return Err(CodecError::Malformed {
                    at_bit,
                    what: "vp8l backward reference",
                });
            }
            for index in pos..pos + length {
                let argb = pixels[index - distance];
                pixels[index] = argb;
                if cache_bits > 0 {
                    cache[(argb.wrapping_mul(CACHE_HASH) >> cache_shift) as usize] = argb;
                }
            }
            pos += length;
            col += length;
            while col >= width {
                col -= width;
                row += 1;
            }
            refresh = true;
        } else {
            let key = symbol - NUM_LITERALS - NUM_LENGTH_CODES;
            let Some(&argb) = cache.get(key) else {
                return Err(CodecError::Malformed {
                    at_bit,
                    what: "vp8l color cache index",
                });
            };
            pixels[pos] = argb;
            cache[(argb.wrapping_mul(CACHE_HASH) >> cache_shift) as usize] = argb;
            pos += 1;
            col += 1;
            if col == width {
                col = 0;
                row += 1;
            }
        }
    }
    Ok(pixels)
}

/// Value of a length or distance prefix symbol and its extra bits.
fn prefix_value(reader: &mut BitReader<'_>, symbol: u32) -> Result<u32, CodecError> {
    if symbol < 4 {
        return Ok(symbol + 1);
    }
    let extra = (symbol - 2) >> 1;
    let offset = (2 + (symbol & 1)) << extra;
    Ok(offset + reader.read(extra)? + 1)
}

/// Pixel distance of a distance code (RFC 9649 section 3.6, "LZ77 Backward
/// Reference").
pub(crate) fn plane_distance(width: usize, code: usize) -> usize {
    if code > DISTANCE_MAP.len() {
        return code - DISTANCE_MAP.len();
    }
    let (dx, dy) = DISTANCE_MAP[code - 1];
    let distance = i64::from(dy) * width as i64 + i64::from(dx);
    if distance < 1 { 1 } else { distance as usize }
}

/// Read one prefix code over `alphabet` symbols.
fn read_code(reader: &mut BitReader<'_>, alphabet: usize) -> Result<Code, CodecError> {
    let at_bit = reader.position();
    let mut lengths = vec![0u8; alphabet];
    if reader.flag()? {
        // Simple code: one or two symbols.
        let two = reader.flag()?;
        let first = if reader.flag()? {
            reader.read(8)?
        } else {
            reader.read(1)?
        } as usize;
        if let Some(length) = lengths.get_mut(first) {
            *length = 1;
        }
        if two {
            let second = reader.read(8)? as usize;
            if let Some(length) = lengths.get_mut(second) {
                *length = 1;
            }
        }
        return Code::from_lengths(&lengths, at_bit);
    }
    // Normal code: lengths coded with a code-length code.
    let count = reader.read(4)? as usize + 4;
    let mut code_length_lengths = [0u8; 19];
    for &symbol in CODE_LENGTH_ORDER.iter().take(count) {
        code_length_lengths[symbol] = reader.read(3)? as u8;
    }
    let length_code = Code::from_lengths(&code_length_lengths, at_bit)?;
    let mut budget = if reader.flag()? {
        let width = 2 + 2 * reader.read(3)?;
        let max = 2 + reader.read(width)? as usize;
        if max > alphabet {
            return Err(CodecError::Malformed {
                at_bit,
                what: "vp8l code length count",
            });
        }
        max
    } else {
        alphabet
    };
    let mut symbol = 0usize;
    let mut previous = 8u8;
    while symbol < alphabet {
        if budget == 0 {
            break;
        }
        budget -= 1;
        let length = length_code.read(reader)?;
        if usize::from(length) <= MAX_CODE_LENGTH {
            let length = length as u8;
            lengths[symbol] = length;
            symbol += 1;
            if length != 0 {
                previous = length;
            }
            continue;
        }
        let (repeat, value) = match length {
            16 => (3 + reader.read(2)? as usize, previous),
            17 => (3 + reader.read(3)? as usize, 0),
            _ => (11 + reader.read(7)? as usize, 0),
        };
        if symbol + repeat > alphabet {
            return Err(CodecError::Malformed {
                at_bit,
                what: "vp8l code length repeat",
            });
        }
        lengths[symbol..symbol + repeat].fill(value);
        symbol += repeat;
    }
    Code::from_lengths(&lengths, at_bit)
}

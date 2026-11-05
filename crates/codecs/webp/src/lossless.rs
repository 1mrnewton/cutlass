//! VP8L, the lossless WebP bitstream (RFC 9649).
//!
//! An image is a list of transforms, then the entropy-coded ARGB pixels:
//! prefix-coded literals, LZ77 back references with the 2-D distance map,
//! and an optional color cache. A main image may switch prefix-code
//! groups per tile through an entropy image. The transforms are undone in
//! the reverse of the order they were read. Pixels are `0xAARRGGBB`.
//!
//! The same decoder reads a compressed `ALPH` chunk, which is this stream
//! with no header: the size comes from the picture it belongs to.

use cutlass_decode::CodecError;

use crate::bits::BitReader;
use crate::huffman::{Code, read_code};

/// Prefix-code table entries one image may hold at once, across all its
/// groups. A real encoder stays far below; a hostile stream that names
/// thousands of large groups stops here instead of exhausting memory.
const MAX_TABLE_ENTRIES: usize = 1 << 24;

/// Section 4.2.2: the 120 short distance codes as (dx, dy).
const DISTANCE_MAP: [(i8, i8); 120] = [
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

/// A decoded lossless image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Argb {
    pub width: usize,
    pub height: usize,
    pub has_alpha: bool,
    pub pixels: Vec<u32>,
}

/// Width and height from a `VP8L` header.
pub(crate) fn size(data: &[u8]) -> Result<(u32, u32), CodecError> {
    if data.len() < 5 {
        return Err(truncated(0, "vp8l header"));
    }
    if data[0] != 0x2F {
        return Err(malformed(0, "vp8l signature"));
    }
    let bits = u32::from_le_bytes([data[1], data[2], data[3], data[4]]);
    if bits >> 29 != 0 {
        return Err(CodecError::Unsupported {
            what: "vp8l version",
        });
    }
    Ok(((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1))
}

/// Decode a whole `VP8L` chunk.
pub(crate) fn decode(data: &[u8], max_pixels: usize) -> Result<Argb, CodecError> {
    let (width, height) = size(data)?;
    let has_alpha = (data[4] >> 4) & 1 == 1;
    let (width, height) = (to_usize(width), to_usize(height));
    let mut reader = BitReader::new(&data[5..]);
    let pixels = decode_stream(&mut reader, width, height, max_pixels)?;
    Ok(Argb {
        width,
        height,
        has_alpha,
        pixels,
    })
}

/// Decode a headerless stream of a known size, as an `ALPH` chunk holds.
pub(crate) fn decode_headless(
    data: &[u8],
    width: usize,
    height: usize,
    max_pixels: usize,
) -> Result<Vec<u32>, CodecError> {
    decode_stream(&mut BitReader::new(data), width, height, max_pixels)
}

enum Transform {
    Predictor {
        bits: u32,
        width: usize,
        modes: Vec<u32>,
    },
    Color {
        bits: u32,
        width: usize,
        elements: Vec<u32>,
    },
    SubtractGreen,
    Indexing {
        /// Width before packing, which the inverse restores.
        width: usize,
        bits: u32,
        table: Vec<u32>,
    },
}

fn decode_stream(
    reader: &mut BitReader<'_>,
    width: usize,
    height: usize,
    max_pixels: usize,
) -> Result<Vec<u32>, CodecError> {
    if width == 0 || height == 0 {
        return Err(malformed(0, "vp8l size"));
    }
    if width.checked_mul(height).is_none_or(|p| p > max_pixels) {
        return Err(CodecError::Limit {
            what: "vp8l image size",
        });
    }
    let mut transforms = Vec::new();
    let mut seen = [false; 4];
    let mut coded_width = width;
    while reader.read_flag()? {
        let kind = reader.read(2)? as usize;
        if std::mem::replace(&mut seen[kind], true) {
            return Err(malformed(reader.position(), "vp8l repeated transform"));
        }
        let transform = match kind {
            0 | 1 => {
                let bits = reader.read(3)? + 2;
                let (tiles_x, tiles_y) =
                    (coded_width.div_ceil(1 << bits), height.div_ceil(1 << bits));
                let image = decode_image(reader, tiles_x, tiles_y, false)?;
                if kind == 0 {
                    Transform::Predictor {
                        bits,
                        width: coded_width,
                        modes: image,
                    }
                } else {
                    Transform::Color {
                        bits,
                        width: coded_width,
                        elements: image,
                    }
                }
            }
            2 => Transform::SubtractGreen,
            _ => {
                let size = reader.read(8)? as usize + 1;
                let mut table = decode_image(reader, size, 1, false)?;
                for at in 1..table.len() {
                    table[at] = add_pixels(table[at], table[at - 1]);
                }
                let bits = match size {
                    0..=2 => 3,
                    3..=4 => 2,
                    5..=16 => 1,
                    _ => 0,
                };
                let before = coded_width;
                coded_width = coded_width.div_ceil(1 << bits);
                Transform::Indexing {
                    width: before,
                    bits,
                    table,
                }
            }
        };
        transforms.push(transform);
    }
    let mut pixels = decode_image(reader, coded_width, height, true)?;
    for transform in transforms.iter().rev() {
        pixels = match transform {
            Transform::Predictor { bits, width, modes } => {
                predict(&mut pixels, *width, *bits, modes)?;
                pixels
            }
            Transform::Color {
                bits,
                width,
                elements,
            } => {
                cross_color(&mut pixels, *width, *bits, elements);
                pixels
            }
            Transform::SubtractGreen => {
                for pixel in &mut pixels {
                    let green = (*pixel >> 8) & 0xFF;
                    *pixel = add_pixels(*pixel, (green << 16) | green);
                }
                pixels
            }
            Transform::Indexing { width, bits, table } => {
                unpack(&pixels, *width, height, *bits, table)
            }
        };
    }
    Ok(pixels)
}

/// The five codes of one group: green with lengths and cache, red, blue,
/// alpha, distance.
struct Group {
    codes: [Code; 5],
}

fn decode_image(
    reader: &mut BitReader<'_>,
    width: usize,
    height: usize,
    main: bool,
) -> Result<Vec<u32>, CodecError> {
    let cache_bits = if reader.read_flag()? {
        let bits = reader.read(4)?;
        if !(1..=11).contains(&bits) {
            return Err(malformed(reader.position(), "vp8l color cache"));
        }
        bits
    } else {
        0
    };
    let mut meta: Option<(u32, usize, Vec<u32>)> = None;
    let mut group_count = 1;
    if main && reader.read_flag()? {
        let bits = reader.read(3)? + 2;
        let tiles_x = width.div_ceil(1 << bits);
        let tiles_y = height.div_ceil(1 << bits);
        let image = decode_image(reader, tiles_x, tiles_y, false)?;
        let ids: Vec<u32> = image.iter().map(|p| (p >> 8) & 0xFFFF).collect();
        group_count = ids.iter().max().map_or(1, |&m| m as usize + 1);
        meta = Some((bits, tiles_x, ids));
    }
    // Only groups the entropy image names are kept; the rest are read
    // past so the stream stays in step.
    let mut used = vec![false; group_count];
    match &meta {
        Some((_, _, ids)) => {
            for &id in ids {
                used[id as usize] = true;
            }
        }
        None => used[0] = true,
    }
    let cache_size = if cache_bits > 0 { 1 << cache_bits } else { 0 };
    let alphabets = [256 + 24 + cache_size, 256, 256, 256, 40];
    let mut slot = vec![usize::MAX; group_count];
    let mut groups = Vec::new();
    let mut entries = 0_usize;
    for (id, &keep) in used.iter().enumerate() {
        let mut codes = Vec::with_capacity(5);
        for &alphabet in &alphabets {
            let code = read_code(reader, alphabet)?;
            if keep {
                entries = entries.saturating_add(code.entries());
                if entries > MAX_TABLE_ENTRIES {
                    return Err(CodecError::Limit {
                        what: "vp8l prefix code tables",
                    });
                }
                codes.push(code);
            }
        }
        if keep {
            let Ok(codes) = <[Code; 5]>::try_from(codes) else {
                return Err(malformed(reader.position(), "vp8l group"));
            };
            slot[id] = groups.len();
            groups.push(Group { codes });
        }
    }

    let total = width * height;
    let mut out = vec![0_u32; total];
    let mut cache = vec![0_u32; cache_size];
    let cache_shift = 32 - cache_bits;
    let mut cached = 0;
    let mut at = 0;
    while at < total {
        let group = match &meta {
            Some((bits, tiles_x, ids)) => {
                let (x, y) = (at % width, at / width);
                let id = ids[(y >> bits) * tiles_x + (x >> bits)] as usize;
                &groups[slot[id]]
            }
            None => &groups[0],
        };
        let green = usize::from(group.codes[0].read(reader)?);
        if green < 256 {
            let red = u32::from(group.codes[1].read(reader)?);
            let blue = u32::from(group.codes[2].read(reader)?);
            let alpha = u32::from(group.codes[3].read(reader)?);
            out[at] = (alpha << 24) | (red << 16) | ((green as u32) << 8) | blue;
            at += 1;
        } else if green < 280 {
            let length = prefix_value(reader, green - 256)?;
            let symbol = usize::from(group.codes[4].read(reader)?);
            let code = prefix_value(reader, symbol)?;
            let distance = distance(code, width);
            if distance > at || length > total - at {
                return Err(malformed(reader.position(), "vp8l back reference"));
            }
            for n in at..at + length {
                out[n] = out[n - distance];
            }
            at += length;
        } else {
            let index = green - 280;
            let Some(&color) = cache.get(index) else {
                return Err(malformed(reader.position(), "vp8l cache index"));
            };
            out[at] = color;
            at += 1;
        }
        if cache_size > 0 {
            for &pixel in &out[cached..at] {
                cache[(0x1E35_A7BD_u32.wrapping_mul(pixel) >> cache_shift) as usize] = pixel;
            }
            cached = at;
        }
    }
    Ok(out)
}

/// Section 4.2.2: a length or distance from its prefix symbol and extra bits.
fn prefix_value(reader: &mut BitReader<'_>, symbol: usize) -> Result<usize, CodecError> {
    if symbol < 4 {
        return Ok(symbol + 1);
    }
    let extra = u32::try_from((symbol - 2) >> 1).unwrap_or(0);
    let offset = (2 + (symbol & 1)) << extra;
    Ok(offset + reader.read(extra)? as usize + 1)
}

fn distance(code: usize, width: usize) -> usize {
    if code > 120 {
        return code - 120;
    }
    let (dx, dy) = DISTANCE_MAP[code - 1];
    let value = i64::from(dx) + i64::from(dy) * i64::try_from(width).unwrap_or(i64::MAX);
    usize::try_from(value.max(1)).unwrap_or(1)
}

fn add_pixels(a: u32, b: u32) -> u32 {
    let alpha_green = (a & 0xFF00_FF00).wrapping_add(b & 0xFF00_FF00) & 0xFF00_FF00;
    let red_blue = (a & 0x00FF_00FF).wrapping_add(b & 0x00FF_00FF) & 0x00FF_00FF;
    alpha_green | red_blue
}

fn channels(pixel: u32) -> [i32; 4] {
    [24, 16, 8, 0].map(|shift| ((pixel >> shift) & 0xFF) as i32)
}

fn from_channels(values: [i32; 4]) -> u32 {
    values
        .iter()
        .zip([24, 16, 8, 0])
        .fold(0, |pixel, (&v, shift)| {
            pixel | (((v & 0xFF) as u32) << shift)
        })
}

fn average2(a: u32, b: u32) -> u32 {
    (((a ^ b) & 0xFEFE_FEFE) >> 1) + (a & b)
}

fn select(left: u32, top: u32, top_left: u32) -> u32 {
    let (l, t, tl) = (channels(left), channels(top), channels(top_left));
    // Distance of the gradient estimate L + T - TL to each neighbour.
    let to_left: i32 = (0..4).map(|c| (t[c] - tl[c]).abs()).sum();
    let to_top: i32 = (0..4).map(|c| (l[c] - tl[c]).abs()).sum();
    if to_left < to_top { left } else { top }
}

fn clamp_add_subtract_full(a: u32, b: u32, c: u32) -> u32 {
    let (a, b, c) = (channels(a), channels(b), channels(c));
    from_channels([0, 1, 2, 3].map(|i| (a[i] + b[i] - c[i]).clamp(0, 255)))
}

fn clamp_add_subtract_half(a: u32, b: u32) -> u32 {
    let (a, b) = (channels(a), channels(b));
    from_channels([0, 1, 2, 3].map(|i| (a[i] + (a[i] - b[i]) / 2).clamp(0, 255)))
}

/// Undo the predictor transform (section 4.1).
fn predict(pixels: &mut [u32], width: usize, bits: u32, modes: &[u32]) -> Result<(), CodecError> {
    let tiles_x = width.div_ceil(1 << bits);
    for at in 0..pixels.len() {
        let (x, y) = (at % width, at / width);
        let prediction = if at == 0 {
            0xFF00_0000
        } else if y == 0 {
            pixels[at - 1]
        } else if x == 0 {
            pixels[at - width]
        } else {
            let mode = (modes[(y >> bits) * tiles_x + (x >> bits)] >> 8) & 0xF;
            let l = pixels[at - 1];
            let t = pixels[at - width];
            let tl = pixels[at - width - 1];
            // On the right edge this is the row's first pixel, as specified.
            let tr = pixels[at - width + 1];
            match mode {
                0 => 0xFF00_0000,
                1 => l,
                2 => t,
                3 => tr,
                4 => tl,
                5 => average2(average2(l, tr), t),
                6 => average2(l, tl),
                7 => average2(l, t),
                8 => average2(tl, t),
                9 => average2(t, tr),
                10 => average2(average2(l, tl), average2(t, tr)),
                11 => select(l, t, tl),
                12 => clamp_add_subtract_full(l, t, tl),
                13 => clamp_add_subtract_half(average2(l, t), tl),
                _ => return Err(malformed(0, "vp8l predictor mode")),
            }
        };
        pixels[at] = add_pixels(pixels[at], prediction);
    }
    Ok(())
}

fn color_delta(transform: u32, color: u32) -> i32 {
    let t = i32::from((transform & 0xFF) as u8 as i8);
    let c = i32::from((color & 0xFF) as u8 as i8);
    (t * c) >> 5
}

/// Undo the color transform (section 4.2).
fn cross_color(pixels: &mut [u32], width: usize, bits: u32, elements: &[u32]) {
    let tiles_x = width.div_ceil(1 << bits);
    for (at, pixel) in pixels.iter_mut().enumerate() {
        let (x, y) = (at % width, at / width);
        let element = elements[(y >> bits) * tiles_x + (x >> bits)];
        let (red_to_blue, green_to_blue, green_to_red) = (
            (element >> 16) & 0xFF,
            (element >> 8) & 0xFF,
            element & 0xFF,
        );
        let green = (*pixel >> 8) & 0xFF;
        let red = (((*pixel >> 16) & 0xFF) as i32 + color_delta(green_to_red, green)) & 0xFF;
        let red = red as u32;
        let blue = ((*pixel & 0xFF) as i32
            + color_delta(green_to_blue, green)
            + color_delta(red_to_blue, red))
            & 0xFF;
        *pixel = (*pixel & 0xFF00_FF00) | (red << 16) | blue as u32;
    }
}

/// Undo color indexing (section 4.4): look up each index, unpacking
/// several per pixel when the palette is small. An index past the table is
/// transparent black.
fn unpack(packed: &[u32], width: usize, height: usize, bits: u32, table: &[u32]) -> Vec<u32> {
    let packed_width = width.div_ceil(1 << bits);
    let per = 8 >> bits;
    let mask = (1_u32 << per) - 1;
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let source = packed[y * packed_width + (x >> bits)];
            let shift = (x as u32 & ((1 << bits) - 1)) * per;
            let index = ((source >> 8) >> shift) & mask;
            out.push(table.get(index as usize).copied().unwrap_or(0));
        }
    }
    out
}

fn to_usize(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

fn truncated(at: usize, what: &'static str) -> CodecError {
    CodecError::Truncated {
        at_bit: u64::try_from(at).unwrap_or(u64::MAX).saturating_mul(8),
        what,
    }
}

fn malformed(at_bit: u64, what: &'static str) -> CodecError {
    CodecError::Malformed { at_bit, what }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_values_and_distances_follow_the_spec() {
        let mut empty = BitReader::new(&[]);
        assert_eq!(prefix_value(&mut empty, 0).unwrap(), 1);
        assert_eq!(prefix_value(&mut empty, 3).unwrap(), 4);
        // Symbol 4: one extra bit over offset 4; symbol 5: over offset 6.
        let mut one = BitReader::new(&[0b1]);
        assert_eq!(prefix_value(&mut one, 4).unwrap(), 6);
        let mut zero = BitReader::new(&[0]);
        assert_eq!(prefix_value(&mut zero, 5).unwrap(), 7);
        // Code 1 is the pixel above; 2 the one to the left; past 120 is linear.
        assert_eq!(distance(1, 50), 50);
        assert_eq!(distance(2, 50), 1);
        assert_eq!(distance(4, 50), 49);
        assert_eq!(distance(125, 50), 5);
        // (-1, 1) on a 1-wide image would be 0, which becomes 1.
        assert_eq!(distance(4, 1), 1);
    }

    #[test]
    fn predictor_helpers_work_per_channel() {
        assert_eq!(add_pixels(0xFF01_02FF, 0x0101_0102), 0x0002_0301);
        assert_eq!(average2(0x0000_00FF, 0x0000_0001), 0x0000_0080);
        assert_eq!(average2(0xFF00_0000, 0x0100_0000), 0x8000_0000);
        // Select picks the neighbour closer to L + T - TL.
        assert_eq!(select(0x10, 0x80, 0x10), 0x80);
        assert_eq!(select(0x80, 0x10, 0x10), 0x80);
        assert_eq!(clamp_add_subtract_full(0xC8, 0xC8, 0x0A), 0xFF);
        // a + (a - b) / 2 truncates toward zero: 10 + (-5 / 2) = 8.
        assert_eq!(clamp_add_subtract_half(0x0A, 0x0F), 0x08);
        assert_eq!(color_delta(0xFF, 0x40), -2);
        assert_eq!(color_delta(0x20, 0x20), 32);
    }

    #[test]
    fn transforms_undo_in_reverse() {
        // Predictor mode 1 (left) on a 3x2 image of residuals.
        let mut pixels = vec![0x0000_0001, 1, 1, 1, 1, 1];
        predict(&mut pixels, 3, 2, &[0x0000_0100]).unwrap();
        assert_eq!(
            pixels,
            [
                0xFF00_0001,
                0xFF00_0002,
                0xFF00_0003,
                0xFF00_0002,
                0xFF00_0003,
                0xFF00_0004
            ]
        );
        assert!(predict(&mut [0; 4], 2, 2, &[0x0000_0E00]).is_err());
        // Cross color: green_to_red 0x20 adds green * 32 >> 5 = green.
        let mut pixels = vec![0xFF10_2030];
        cross_color(&mut pixels, 1, 2, &[0x0000_0020]);
        assert_eq!(pixels, [0xFF30_2030]);
        // Indexing: two colors, eight one-bit indexes per packed pixel.
        let table = [0xFF00_0000, 0xFFFF_FFFF];
        let out = unpack(&[0x0000_A500], 8, 1, 3, &table);
        let bits: Vec<u32> = out.iter().map(|&p| p & 1).collect();
        assert_eq!(bits, [1, 0, 1, 0, 0, 1, 0, 1]);
        // An index past the palette is transparent black.
        assert_eq!(unpack(&[0x0000_0500], 1, 1, 0, &table), [0]);
    }

    #[test]
    fn huge_and_bad_headers_are_refused() {
        assert!(size(&[0x2F, 0, 0]).is_err());
        assert!(size(&[0x2E, 0, 0, 0, 0]).is_err());
        assert!(matches!(
            size(&[0x2F, 0, 0, 0, 0x20]),
            Err(CodecError::Unsupported { .. })
        ));
        // 16384 x 16384 is past the limit before any pixel is read.
        let huge = [0x2F, 0xFF, 0xFF, 0xFF, 0x0F];
        assert_eq!(size(&huge).unwrap(), (16_384, 16_384));
        assert!(matches!(
            decode(&huge, 1 << 26),
            Err(CodecError::Limit { .. })
        ));
    }
}

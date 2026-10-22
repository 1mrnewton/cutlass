//! VP8L, the WebP lossless bitstream.
//!
//! The container hands this module one `VP8L` payload. Transforms are inverted
//! in reverse order (WebP lossless 4). Samples are bit-exact.

use crate::MAX_EDGE;
use crate::bits::Bits;
use crate::error::Error;
use crate::huff::{Budget, Huff, read_huff};

/// One decoded picture, still in ARGB. Alpha is bits 31..24, then red, green, blue.
pub(crate) struct Picture {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixels: Vec<u32>,
}

/// WebP lossless 5.2.2. `xi` counts pixels to the left of the current pixel.
const DISTANCE: [(i8, i8); 120] = [
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

/// `(0x1e35a7bd * color) >> (32 - bits)`. WebP lossless 5.2.3.
pub(crate) fn cache_index(color: u32, bits: u32) -> usize {
    let hash = color.wrapping_mul(0x1e35_a7bd) >> (32 - bits);
    usize::try_from(hash).expect("a color cache index fits")
}

/// Decodes one `VP8L` payload, including its signature byte.
pub(crate) fn decode_vp8l(data: &[u8]) -> Result<Picture, Error> {
    let mut bits = Bits::new(data);
    let mut budget = Budget::new();
    // WebP lossless 3.
    if bits.read(8)? != 0x2f {
        return Err(Error::Inconsistent("the lossless signature is not 0x2f"));
    }
    let width = bits.read(14)? + 1;
    let height = bits.read(14)? + 1;
    // The alpha bit is a hint and does not change the samples.
    let _alpha_used = bits.read(1)?;
    if bits.read(3)? != 0 {
        return Err(Error::Inconsistent("the version is not 0"));
    }
    pixel_count(width, height)?;
    let (coded_width, transforms) = read_transforms(&mut bits, width, height, &mut budget)?;
    let pixels = decode_image(&mut bits, coded_width, height, true, &mut budget)?;
    let pixels = apply(transforms, pixels, coded_width, height)?;
    let count = pixel_count(width, height)?;
    if pixels.len() != count {
        return Err(Error::Inconsistent("the image does not match its size"));
    }
    Ok(Picture {
        width,
        height,
        pixels,
    })
}

/// Pixel count when both edges are in range and the buffer is at most 256 MiB.
pub(crate) fn pixel_count(width: u32, height: u32) -> Result<usize, Error> {
    if width == 0 || height == 0 {
        return Err(Error::Inconsistent("an image edge is zero"));
    }
    if width > MAX_EDGE || height > MAX_EDGE {
        return Err(Error::TooLarge);
    }
    let count = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .ok_or(Error::TooLarge)?;
    let bytes = count.checked_mul(4).ok_or(Error::TooLarge)?;
    if bytes > crate::MAX_BYTES {
        return Err(Error::TooLarge);
    }
    Ok(count)
}

fn as_usize(value: u32) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_| Error::TooLarge)
}

enum Transform {
    Predictor {
        shift: u8,
        width: usize,
        cols: usize,
        modes: Vec<u32>,
    },
    Color {
        shift: u8,
        width: usize,
        cols: usize,
        elems: Vec<u32>,
    },
    Green,
    Index {
        table: Vec<u32>,
        width_bits: u8,
        full_width: usize,
    },
}

fn read_transforms(
    bits: &mut Bits<'_>,
    mut width: u32,
    height: u32,
    budget: &mut Budget,
) -> Result<(u32, Vec<Transform>), Error> {
    let mut seen = [false; 4];
    let mut transforms = Vec::new();
    loop {
        if bits.read(1)? == 0 {
            break;
        }
        let kind = bits.read(2)?;
        let Some(slot) = usize::try_from(kind)
            .ok()
            .and_then(|index| seen.get_mut(index))
        else {
            return Err(Error::Inconsistent("a transform is not supported"));
        };
        if *slot {
            return Err(Error::Inconsistent("a transform is repeated"));
        }
        *slot = true;
        match kind {
            0 => transforms.push(read_predictor(bits, width, height, budget)?),
            1 => transforms.push(read_color(bits, width, height, budget)?),
            2 => transforms.push(Transform::Green),
            3 => {
                let (xform, packed) = read_index(bits, width, budget)?;
                width = packed;
                transforms.push(xform);
            }
            _ => return Err(Error::Inconsistent("a transform is not supported")),
        }
    }
    Ok((width, transforms))
}

fn read_predictor(
    bits: &mut Bits<'_>,
    width: u32,
    height: u32,
    budget: &mut Budget,
) -> Result<Transform, Error> {
    let (shift, cols, modes) = read_blocks(bits, width, height, budget)?;
    Ok(Transform::Predictor {
        shift,
        width: as_usize(width)?,
        cols,
        modes,
    })
}

fn read_color(
    bits: &mut Bits<'_>,
    width: u32,
    height: u32,
    budget: &mut Budget,
) -> Result<Transform, Error> {
    let (shift, cols, elems) = read_blocks(bits, width, height, budget)?;
    Ok(Transform::Color {
        shift,
        width: as_usize(width)?,
        cols,
        elems,
    })
}

/// WebP lossless 4.1 and 4.2. The next 3 bits are the block size, then a sub image.
fn read_blocks(
    bits: &mut Bits<'_>,
    width: u32,
    height: u32,
    budget: &mut Budget,
) -> Result<(u8, usize, Vec<u32>), Error> {
    let size_bits = bits.read(3)? + 2;
    let span = 1u32
        .checked_shl(size_bits)
        .ok_or(Error::Inconsistent("a block size is too wide"))?;
    let cols = width.div_ceil(span);
    let rows = height.div_ceil(span);
    let modes = decode_image(bits, cols, rows, false, budget)?;
    let shift = u8::try_from(size_bits).expect("a block shift fits in a byte");
    Ok((shift, as_usize(cols)?, modes))
}

/// WebP lossless 4.4. Returns the transform and the packed image width.
fn read_index(
    bits: &mut Bits<'_>,
    width: u32,
    budget: &mut Budget,
) -> Result<(Transform, u32), Error> {
    let table_size = bits.read(8)? + 1;
    let raw = decode_image(bits, table_size, 1, false, budget)?;
    let mut table = Vec::with_capacity(raw.len());
    let mut a = 0u8;
    let mut r = 0u8;
    let mut g = 0u8;
    let mut b = 0u8;
    for pixel in raw {
        a = a.wrapping_add(alpha(pixel));
        r = r.wrapping_add(red(pixel));
        g = g.wrapping_add(green(pixel));
        b = b.wrapping_add(blue(pixel));
        table.push(pack(a, r, g, b));
    }
    let width_bits = if table_size <= 2 {
        3
    } else if table_size <= 4 {
        2
    } else if table_size <= 16 {
        1
    } else {
        0
    };
    let packed = width.div_ceil(1u32 << width_bits);
    let xform = Transform::Index {
        table,
        width_bits: u8::try_from(width_bits).expect("a bundle width fits in a byte"),
        full_width: as_usize(width)?,
    };
    Ok((xform, packed))
}

fn apply(
    transforms: Vec<Transform>,
    mut pixels: Vec<u32>,
    mut width: u32,
    height: u32,
) -> Result<Vec<u32>, Error> {
    for transform in transforms.into_iter().rev() {
        match transform {
            Transform::Predictor {
                shift,
                width: expect,
                cols,
                modes,
            } => {
                if as_usize(width)? != expect {
                    return Err(Error::Inconsistent("a predictor does not match the image"));
                }
                apply_predictor(&mut pixels, expect, shift, cols, &modes)?;
            }
            Transform::Color {
                shift,
                width: expect,
                cols,
                elems,
            } => {
                if as_usize(width)? != expect {
                    return Err(Error::Inconsistent(
                        "a color transform does not match the image",
                    ));
                }
                apply_color(&mut pixels, expect, shift, cols, &elems)?;
            }
            Transform::Green => apply_green(&mut pixels),
            Transform::Index {
                table,
                width_bits,
                full_width,
            } => {
                pixels = unpack_index(
                    &pixels,
                    as_usize(width)?,
                    full_width,
                    as_usize(height)?,
                    width_bits,
                    &table,
                )?;
                width = u32::try_from(full_width).map_err(|_| Error::TooLarge)?;
            }
        }
    }
    Ok(pixels)
}

/// WebP lossless 4.1. Borders ignore the stored mode.
fn apply_predictor(
    pixels: &mut [u32],
    width: usize,
    shift: u8,
    cols: usize,
    modes: &[u32],
) -> Result<(), Error> {
    if width == 0 || !pixels.len().is_multiple_of(width) {
        return Err(Error::Inconsistent("a predictor does not match the image"));
    }
    for pos in 0..pixels.len() {
        let residual = pixels[pos];
        let pred = predicted(modes, cols, shift, width, pixels, pos)?;
        pixels[pos] = add_px(residual, pred);
    }
    Ok(())
}

fn predicted(
    modes: &[u32],
    cols: usize,
    shift: u8,
    width: usize,
    pixels: &[u32],
    pos: usize,
) -> Result<u32, Error> {
    let x = pos % width;
    let y = pos / width;
    if x == 0 && y == 0 {
        return Ok(0xff00_0000);
    }
    if y == 0 {
        return Ok(pixels[pos - 1]);
    }
    if x == 0 {
        return Ok(pixels[pos - width]);
    }
    let mode = mode_at(modes, cols, shift, x, y)?;
    let left = pixels[pos - 1];
    let top = pixels[pos - width];
    let top_left = pixels[pos - width - 1];
    let top_right = if x + 1 == width {
        pixels[pos - x]
    } else {
        pixels[pos - width + 1]
    };
    match mode {
        0 => Ok(0xff00_0000),
        1 => Ok(left),
        2 => Ok(top),
        3 => Ok(top_right),
        4 => Ok(top_left),
        5 => Ok(avg_px(avg_px(left, top_right), top)),
        6 => Ok(avg_px(left, top_left)),
        7 => Ok(avg_px(left, top)),
        8 => Ok(avg_px(top_left, top)),
        9 => Ok(avg_px(top, top_right)),
        10 => Ok(avg_px(avg_px(left, top_left), avg_px(top, top_right))),
        11 => Ok(select(left, top, top_left)),
        12 => Ok(clamp_full(left, top, top_left)),
        13 => Ok(clamp_half(left, top, top_left)),
        _ => Err(Error::Inconsistent("a predictor is not one of 14")),
    }
}

fn mode_at(modes: &[u32], cols: usize, shift: u8, x: usize, y: usize) -> Result<u8, Error> {
    let bx = x >> usize::from(shift);
    let by = y >> usize::from(shift);
    let pixel = modes
        .get(by * cols + bx)
        .copied()
        .ok_or(Error::Inconsistent("a predictor block is missing"))?;
    Ok(green(pixel))
}

fn select(left: u32, top: u32, top_left: u32) -> u32 {
    let mut left_dist = 0i32;
    let mut top_dist = 0i32;
    for shift in [24u32, 16, 8, 0] {
        let guess = i32::from(chan(left, shift)) + i32::from(chan(top, shift))
            - i32::from(chan(top_left, shift));
        left_dist += (guess - i32::from(chan(left, shift))).abs();
        top_dist += (guess - i32::from(chan(top, shift))).abs();
    }
    if left_dist < top_dist { left } else { top }
}

fn clamp_full(left: u32, top: u32, top_left: u32) -> u32 {
    let ch = |shift: u32| {
        clamp_u8(
            i32::from(chan(left, shift)) + i32::from(chan(top, shift))
                - i32::from(chan(top_left, shift)),
        )
    };
    pack(ch(24), ch(16), ch(8), ch(0))
}

fn clamp_half(left: u32, top: u32, top_left: u32) -> u32 {
    let ch = |shift: u32| {
        let average = i32::from(avg(chan(left, shift), chan(top, shift)));
        let other = i32::from(chan(top_left, shift));
        clamp_u8(average + (average - other) / 2)
    };
    pack(ch(24), ch(16), ch(8), ch(0))
}

/// WebP lossless 4.2. Red is updated before it decorrelates blue.
fn apply_color(
    pixels: &mut [u32],
    width: usize,
    shift: u8,
    cols: usize,
    elems: &[u32],
) -> Result<(), Error> {
    if width == 0 || !pixels.len().is_multiple_of(width) {
        return Err(Error::Inconsistent(
            "a color transform does not match the image",
        ));
    }
    let height = pixels.len() / width;
    for y in 0..height {
        for x in 0..width {
            let elem = elems
                .get((y >> usize::from(shift)) * cols + (x >> usize::from(shift)))
                .copied()
                .ok_or(Error::Inconsistent("a color block is missing"))?;
            let at = y * width + x;
            pixels[at] = inverse_color(pixels[at], elem);
        }
    }
    Ok(())
}

fn inverse_color(pixel: u32, elem: u32) -> u32 {
    // Blue of the element is green-to-red, green is green-to-blue, red is red-to-blue.
    let green_to_red = blue(elem);
    let green_to_blue = green(elem);
    let red_to_blue = red(elem);
    let g = green(pixel);
    let red_sum = i32::from(red(pixel)) + color_delta(green_to_red, g);
    let new_red = low_byte(red_sum);
    let blue_sum =
        i32::from(blue(pixel)) + color_delta(green_to_blue, g) + color_delta(red_to_blue, new_red);
    pack(alpha(pixel), new_red, g, low_byte(blue_sum))
}

/// A signed 3.5 fixed-point multiply. WebP lossless 4.2.
fn color_delta(transform: u8, channel: u8) -> i32 {
    let transform = i32::from(i8::from_ne_bytes([transform]));
    let channel = i32::from(i8::from_ne_bytes([channel]));
    (transform * channel) >> 5
}

fn low_byte(value: i32) -> u8 {
    u8::try_from(value & 0xff).expect("eight bits fit in a byte")
}

/// WebP lossless 4.3.
fn apply_green(pixels: &mut [u32]) {
    for pixel in pixels {
        let g = green(*pixel);
        *pixel = pack(
            alpha(*pixel),
            red(*pixel).wrapping_add(g),
            g,
            blue(*pixel).wrapping_add(g),
        );
    }
}

fn unpack_index(
    pixels: &[u32],
    packed_width: usize,
    full_width: usize,
    height: usize,
    width_bits: u8,
    table: &[u32],
) -> Result<Vec<u32>, Error> {
    let count = full_width.checked_mul(height).ok_or(Error::TooLarge)?;
    let bytes = count.checked_mul(4).ok_or(Error::TooLarge)?;
    if bytes > crate::MAX_BYTES {
        return Err(Error::TooLarge);
    }
    let mut out = Vec::with_capacity(count);
    for y in 0..height {
        for x in 0..full_width {
            let index = bundled_index(pixels, packed_width, y, x, width_bits)?;
            // An index past the palette is transparent black. WebP lossless 4.4.
            out.push(table.get(index).copied().unwrap_or(0));
        }
    }
    Ok(out)
}

fn bundled_index(
    pixels: &[u32],
    packed_width: usize,
    y: usize,
    x: usize,
    width_bits: u8,
) -> Result<usize, Error> {
    if width_bits == 0 {
        let pixel = pixels
            .get(y * packed_width + x)
            .copied()
            .ok_or(Error::Inconsistent("a palette index is missing"))?;
        return Ok(usize::from(green(pixel)));
    }
    let px = x >> usize::from(width_bits);
    let pixel = pixels
        .get(y * packed_width + px)
        .copied()
        .ok_or(Error::Inconsistent("a palette index is missing"))?;
    let bits_per = 8usize >> usize::from(width_bits);
    let sub = x & ((1usize << usize::from(width_bits)) - 1);
    let shift = sub * bits_per;
    let mask = (1usize << bits_per) - 1;
    Ok((usize::from(green(pixel)) >> shift) & mask)
}

enum Meta {
    Flat,
    Grid {
        shift: u8,
        cols: usize,
        codes: Vec<u32>,
    },
}

struct Group {
    green: Huff,
    red: Huff,
    blue: Huff,
    alpha: Huff,
    dist: Huff,
}

struct Cache {
    bits: u32,
    slots: Vec<u32>,
}

impl Cache {
    fn new(bits: u32) -> Result<Self, Error> {
        if bits == 0 {
            return Ok(Self {
                bits: 0,
                slots: Vec::new(),
            });
        }
        let n = 1usize
            .checked_shl(bits)
            .ok_or(Error::Inconsistent("a color cache size is not supported"))?;
        Ok(Self {
            bits,
            slots: vec![0; n],
        })
    }

    fn insert(&mut self, color: u32) {
        if self.bits == 0 {
            return;
        }
        let index = cache_index(color, self.bits);
        let slot = self
            .slots
            .get_mut(index)
            .expect("a color cache hash lands in the table");
        *slot = color;
    }

    fn get(&self, index: usize) -> Result<u32, Error> {
        self.slots
            .get(index)
            .copied()
            .ok_or(Error::Inconsistent("a color cache index is out of range"))
    }
}

/// WebP lossless 5 and 6. `use_meta` is set only for the main image.
fn decode_image(
    bits: &mut Bits<'_>,
    width: u32,
    height: u32,
    use_meta: bool,
    budget: &mut Budget,
) -> Result<Vec<u32>, Error> {
    let count = pixel_count(width, height)?;
    let width_us = as_usize(width)?;
    let cache_bits = read_cache_bits(bits)?;
    let mut cache = Cache::new(cache_bits)?;
    let alphabet = 256 + 24 + cache.slots.len();
    let meta = if use_meta {
        read_meta(bits, width, height, budget)?
    } else {
        Meta::Flat
    };
    let groups = read_groups(bits, &meta, alphabet, budget)?;
    let mut pixels = vec![0u32; count];
    let mut pos = 0usize;
    while pos < count {
        let x = pos % width_us;
        let y = pos / width_us;
        let group_index = group_index(&groups, &meta, x, y)?;
        let symbol = u32::from(groups[group_index].green.decode(bits)?);
        if symbol < 256 {
            let g = u8::try_from(symbol).expect("a green literal fits in a byte");
            // WebP lossless 6.2.3. Red, then blue, then alpha.
            let red = literal(&groups[group_index].red, bits)?;
            let blue = literal(&groups[group_index].blue, bits)?;
            let alpha = literal(&groups[group_index].alpha, bits)?;
            let pixel = pack(alpha, red, g, blue);
            write_pixel(&mut pixels, &mut cache, &mut pos, pixel);
        } else if symbol < 256 + 24 {
            copy_run(
                bits,
                &groups[group_index].dist,
                &mut pixels,
                &mut cache,
                &mut pos,
                width_us,
                symbol - 256,
            )?;
        } else {
            let index = usize::try_from(symbol - (256 + 24)).map_err(|_| Error::TooLarge)?;
            let pixel = cache.get(index)?;
            write_pixel(&mut pixels, &mut cache, &mut pos, pixel);
        }
    }
    Ok(pixels)
}

fn write_pixel(pixels: &mut [u32], cache: &mut Cache, pos: &mut usize, pixel: u32) {
    pixels[*pos] = pixel;
    *pos += 1;
    cache.insert(pixel);
}

/// WebP lossless 5.2.2. Length extras come before the distance code.
fn copy_run(
    bits: &mut Bits<'_>,
    distance: &Huff,
    pixels: &mut [u32],
    cache: &mut Cache,
    pos: &mut usize,
    width: usize,
    len_prefix: u32,
) -> Result<(), Error> {
    let len = prefix_value(bits, len_prefix)?;
    if len == 0 || len > 4096 {
        return Err(Error::Inconsistent("a backward reference is too long"));
    }
    let dist_prefix = u32::from(distance.decode(bits)?);
    let dist_code = prefix_value(bits, dist_prefix)?;
    let dist = lookup_distance(dist_code, width, *pos)?;
    let len_us = usize::try_from(len).map_err(|_| Error::TooLarge)?;
    let end = pos.checked_add(len_us).ok_or(Error::Inconsistent(
        "a backward reference runs past the image",
    ))?;
    if end > pixels.len() {
        return Err(Error::Inconsistent(
            "a backward reference runs past the image",
        ));
    }
    for _ in 0..len_us {
        let pixel = pixels[*pos - dist];
        write_pixel(pixels, cache, pos, pixel);
    }
    Ok(())
}

/// WebP lossless 5.2.2. Prefix codes below 4 have no extra bits.
fn prefix_value(bits: &mut Bits<'_>, prefix: u32) -> Result<u32, Error> {
    if prefix > 39 {
        return Err(Error::Inconsistent("a prefix value is out of range"));
    }
    if prefix < 4 {
        return Ok(prefix + 1);
    }
    let extra = (prefix - 2) >> 1;
    let offset = (2 + (prefix & 1))
        .checked_shl(extra)
        .ok_or(Error::Inconsistent("a prefix value is out of range"))?;
    let more = bits.read(extra)?;
    offset
        .checked_add(more)
        .and_then(|value| value.checked_add(1))
        .ok_or(Error::Inconsistent("a prefix value is out of range"))
}

fn lookup_distance(code: u32, width: usize, pos: usize) -> Result<usize, Error> {
    let raw = if code == 0 {
        return Err(Error::Inconsistent("a backward reference is out of range"));
    } else if code <= 120 {
        let index = usize::try_from(code - 1).expect("a distance code fits");
        let (xi, yi) = DISTANCE
            .get(index)
            .copied()
            .ok_or(Error::Inconsistent("a backward reference is out of range"))?;
        let width_i = i32::try_from(width).map_err(|_| Error::TooLarge)?;
        let dist = i32::from(xi) + i32::from(yi) * width_i;
        if dist < 1 { 1 } else { dist }
    } else {
        i32::try_from(code - 120)
            .map_err(|_| Error::Inconsistent("a backward reference is too far"))?
    };
    let dist =
        usize::try_from(raw).map_err(|_| Error::Inconsistent("a backward reference is too far"))?;
    if dist == 0 || dist > pos {
        return Err(Error::Inconsistent("a backward reference is out of range"));
    }
    Ok(dist)
}

fn read_cache_bits(bits: &mut Bits<'_>) -> Result<u32, Error> {
    if bits.read(1)? == 0 {
        return Ok(0);
    }
    let n = bits.read(4)?;
    if !(1..=11).contains(&n) {
        return Err(Error::Inconsistent("a color cache size is not supported"));
    }
    Ok(n)
}

/// WebP lossless 6.2.2. Red and green together are the 16-bit group index.
fn read_meta(
    bits: &mut Bits<'_>,
    width: u32,
    height: u32,
    budget: &mut Budget,
) -> Result<Meta, Error> {
    if bits.read(1)? == 0 {
        return Ok(Meta::Flat);
    }
    let shift = bits.read(3)? + 2;
    let span = 1u32
        .checked_shl(shift)
        .ok_or(Error::Inconsistent("a block size is too wide"))?;
    let cols = width.div_ceil(span);
    let rows = height.div_ceil(span);
    let image = decode_image(bits, cols, rows, false, budget)?;
    let mut codes = Vec::with_capacity(image.len());
    for pixel in image {
        codes.push((pixel >> 8) & 0xffff);
    }
    let shift = u8::try_from(shift).expect("a block shift fits in a byte");
    Ok(Meta::Grid {
        shift,
        cols: as_usize(cols)?,
        codes,
    })
}

fn read_groups(
    bits: &mut Bits<'_>,
    meta: &Meta,
    alphabet: usize,
    budget: &mut Budget,
) -> Result<Vec<Group>, Error> {
    let count = match meta {
        Meta::Flat => 1,
        Meta::Grid { codes, .. } => {
            let max = codes.iter().copied().max().unwrap_or(0);
            usize::try_from(max)
                .map_err(|_| Error::TooLarge)?
                .checked_add(1)
                .ok_or(Error::TooLarge)?
        }
    };
    let mut groups = Vec::with_capacity(count.min(64));
    for _ in 0..count {
        groups.push(read_group(bits, alphabet, budget)?);
    }
    Ok(groups)
}

fn read_group(
    bits: &mut Bits<'_>,
    green_alphabet: usize,
    budget: &mut Budget,
) -> Result<Group, Error> {
    Ok(Group {
        green: read_huff(bits, green_alphabet, budget)?,
        red: read_huff(bits, 256, budget)?,
        blue: read_huff(bits, 256, budget)?,
        alpha: read_huff(bits, 256, budget)?,
        dist: read_huff(bits, 40, budget)?,
    })
}

fn group_index(groups: &[Group], meta: &Meta, x: usize, y: usize) -> Result<usize, Error> {
    let code = match meta {
        Meta::Flat => 0,
        Meta::Grid { shift, cols, codes } => {
            let index = (y >> usize::from(*shift)) * cols + (x >> usize::from(*shift));
            let code = codes
                .get(index)
                .copied()
                .ok_or(Error::Inconsistent("an entropy block is missing"))?;
            usize::try_from(code).map_err(|_| Error::TooLarge)?
        }
    };
    if code >= groups.len() {
        return Err(Error::Inconsistent("a prefix group is missing"));
    }
    Ok(code)
}

fn literal(huff: &Huff, bits: &mut Bits<'_>) -> Result<u8, Error> {
    let symbol = huff.decode(bits)?;
    u8::try_from(symbol).map_err(|_| Error::Inconsistent("a literal is not a byte"))
}

fn pack(a: u8, r: u8, g: u8, b: u8) -> u32 {
    (u32::from(a) << 24) | (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b)
}

pub(crate) fn into_rgba(pixels: &[u32]) -> Vec<u8> {
    let bytes = pixels
        .len()
        .checked_mul(4)
        .expect("the picture fits in memory");
    let mut out = vec![0u8; bytes];
    for (i, pixel) in pixels.iter().enumerate() {
        let at = i * 4;
        out[at] = red(*pixel);
        out[at + 1] = green(*pixel);
        out[at + 2] = blue(*pixel);
        out[at + 3] = alpha(*pixel);
    }
    out
}

fn chan(pixel: u32, shift: u32) -> u8 {
    u8::try_from((pixel >> shift) & 0xff).expect("a channel fits in a byte")
}

fn alpha(pixel: u32) -> u8 {
    chan(pixel, 24)
}

fn red(pixel: u32) -> u8 {
    chan(pixel, 16)
}

fn green(pixel: u32) -> u8 {
    chan(pixel, 8)
}

fn blue(pixel: u32) -> u8 {
    chan(pixel, 0)
}

fn avg(a: u8, b: u8) -> u8 {
    u8::try_from((u16::from(a) + u16::from(b)) / 2).expect("an average fits in a byte")
}

fn avg_px(a: u32, b: u32) -> u32 {
    pack(
        avg(alpha(a), alpha(b)),
        avg(red(a), red(b)),
        avg(green(a), green(b)),
        avg(blue(a), blue(b)),
    )
}

fn add_px(residual: u32, pred: u32) -> u32 {
    pack(
        alpha(residual).wrapping_add(alpha(pred)),
        red(residual).wrapping_add(red(pred)),
        green(residual).wrapping_add(green(pred)),
        blue(residual).wrapping_add(blue(pred)),
    )
}

fn clamp_u8(value: i32) -> u8 {
    if value <= 0 {
        0
    } else if value >= 255 {
        255
    } else {
        u8::try_from(value).expect("a clamped channel fits in a byte")
    }
}

#[cfg(test)]
mod tests {
    use super::DISTANCE;

    #[test]
    fn short_distances_follow_the_neighbor_map() {
        assert_eq!(DISTANCE[0], (0, 1));
        assert_eq!(DISTANCE[2], (1, 1));
        assert_eq!(DISTANCE[4], (0, 2));
        assert_eq!(DISTANCE[119], (8, 7));
    }
}

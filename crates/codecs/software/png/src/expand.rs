//! Unfilter scanlines and expand them to straight RGBA8.

use crate::error::Error;
use crate::image::{self, ColorKind, Info, Key};

pub(crate) fn paint(
    raw: &mut [u8],
    info: &Info,
    palette: &[[u8; 3]],
    key: &Key,
    pixels: &mut [u8],
) -> Result<(), Error> {
    let channels = info.color.channels();
    let bpp = image::filter_bytes(channels, info.bit_depth);
    let mut offset = 0usize;
    for pass in image::passes(info.interlace) {
        let width = image::span(info.width, pass.x0, pass.x_step);
        let height = image::span(info.height, pass.y0, pass.y_step);
        if width == 0 || height == 0 {
            continue;
        }
        let row_bytes = image::row_bytes(width, channels, info.bit_depth)?;
        let stride = row_bytes.checked_add(1).ok_or(Error::TooLarge)?;
        let rows = usize::try_from(height).map_err(|_| Error::TooLarge)?;
        let block = stride.checked_mul(rows).ok_or(Error::TooLarge)?;
        let end = offset.checked_add(block).ok_or(Error::TooLarge)?;
        let Some(scan) = raw.get_mut(offset..end) else {
            return Err(Error::Inconsistent(
                "the image data does not match the header",
            ));
        };
        unfilter(scan, row_bytes, bpp)?;
        for row_index in 0..rows {
            let row_at = row_index
                .checked_mul(stride)
                .and_then(|start| start.checked_add(1))
                .ok_or(Error::TooLarge)?;
            let Some(row) = scan.get(row_at..row_at + row_bytes) else {
                return Err(Error::Inconsistent(
                    "the image data does not match the header",
                ));
            };
            let row_in_pass = u32::try_from(row_index).map_err(|_| Error::TooLarge)?;
            for column in 0..width {
                let x = pass
                    .x0
                    .checked_add(column.checked_mul(pass.x_step).ok_or(Error::TooLarge)?)
                    .ok_or(Error::TooLarge)?;
                let y = pass
                    .y0
                    .checked_add(
                        row_in_pass
                            .checked_mul(pass.y_step)
                            .ok_or(Error::TooLarge)?,
                    )
                    .ok_or(Error::TooLarge)?;
                write_pixel(
                    &mut Target {
                        pixels,
                        info,
                        palette,
                        key,
                    },
                    row,
                    column,
                    x,
                    y,
                )?;
            }
        }
        offset = end;
    }
    if offset != raw.len() {
        return Err(Error::Inconsistent(
            "the image data does not match the header",
        ));
    }
    Ok(())
}

/// PNG 9.2. Bytes before the scanline, and the scanline above the first, are zero.
fn unfilter(scan: &mut [u8], row_bytes: usize, bpp: usize) -> Result<(), Error> {
    let stride = row_bytes.checked_add(1).ok_or(Error::TooLarge)?;
    if stride == 0 || !scan.len().is_multiple_of(stride) {
        return Err(Error::Inconsistent(
            "the image data does not match the header",
        ));
    }
    let rows = scan.len() / stride;
    for row in 0..rows {
        let start = row * stride;
        match scan[start] {
            0 => {}
            1 => add_predictor(scan, start, row_bytes, bpp, row, Predictor::Sub)?,
            2 => add_predictor(scan, start, row_bytes, bpp, row, Predictor::Up)?,
            3 => add_predictor(scan, start, row_bytes, bpp, row, Predictor::Average)?,
            4 => add_predictor(scan, start, row_bytes, bpp, row, Predictor::Paeth)?,
            _ => {
                return Err(Error::Unsupported("the filter type is not 0 through 4"));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Predictor {
    Sub,
    Up,
    Average,
    Paeth,
}

fn add_predictor(
    scan: &mut [u8],
    start: usize,
    row_bytes: usize,
    bpp: usize,
    row: usize,
    predictor: Predictor,
) -> Result<(), Error> {
    let stride = row_bytes + 1;
    for x in 0..row_bytes {
        let left = if x >= bpp {
            scan[start + 1 + x - bpp]
        } else {
            0
        };
        let up = if row == 0 {
            0
        } else {
            scan[(row - 1) * stride + 1 + x]
        };
        let up_left = if row == 0 || x < bpp {
            0
        } else {
            scan[(row - 1) * stride + 1 + x - bpp]
        };
        let predicted = match predictor {
            Predictor::Sub => left,
            Predictor::Up => up,
            Predictor::Average => average(left, up),
            Predictor::Paeth => paeth(left, up, up_left),
        };
        let at = start + 1 + x;
        scan[at] = scan[at].wrapping_add(predicted);
    }
    Ok(())
}

/// PNG 9.3. The sum uses more than eight bits, then rounds down.
fn average(left: u8, up: u8) -> u8 {
    let sum = u16::from(left) + u16::from(up);
    u8::try_from(sum / 2).expect("the average of two bytes fits in a byte")
}

/// PNG 9.4. The comparison order is the one the specification requires.
fn paeth(left: u8, up: u8, up_left: u8) -> u8 {
    let a = i32::from(left);
    let b = i32::from(up);
    let c = i32::from(up_left);
    let estimate = a + b - c;
    let toward_left = (estimate - a).abs();
    let toward_up = (estimate - b).abs();
    let toward_up_left = (estimate - c).abs();
    if toward_left <= toward_up && toward_left <= toward_up_left {
        left
    } else if toward_up <= toward_up_left {
        up
    } else {
        up_left
    }
}

struct Target<'a> {
    pixels: &'a mut [u8],
    info: &'a Info,
    palette: &'a [[u8; 3]],
    key: &'a Key,
}

fn write_pixel(
    target: &mut Target<'_>,
    row: &[u8],
    column: u32,
    x: u32,
    y: u32,
) -> Result<(), Error> {
    if x >= target.info.width || y >= target.info.height {
        return Err(Error::Inconsistent(
            "a pass pixel lands outside the picture",
        ));
    }
    let depth = target.info.bit_depth;
    let (red, green, blue, alpha) = match target.info.color {
        ColorKind::Grey => {
            let sample = read_sample(row, column, 0, 1, depth)?;
            let value = widen(sample, depth)?;
            let alpha = if grey_key(target.key, sample, depth)? {
                0
            } else {
                255
            };
            (value, value, value, alpha)
        }
        ColorKind::Truecolor => {
            let red = read_sample(row, column, 0, 3, depth)?;
            let green = read_sample(row, column, 1, 3, depth)?;
            let blue = read_sample(row, column, 2, 3, depth)?;
            let alpha = if rgb_key(target.key, red, green, blue, depth)? {
                0
            } else {
                255
            };
            (
                widen(red, depth)?,
                widen(green, depth)?,
                widen(blue, depth)?,
                alpha,
            )
        }
        ColorKind::GreyAlpha => {
            let sample = read_sample(row, column, 0, 2, depth)?;
            let alpha = read_sample(row, column, 1, 2, depth)?;
            let value = widen(sample, depth)?;
            (value, value, value, widen(alpha, depth)?)
        }
        ColorKind::TruecolorAlpha => {
            let red = read_sample(row, column, 0, 4, depth)?;
            let green = read_sample(row, column, 1, 4, depth)?;
            let blue = read_sample(row, column, 2, 4, depth)?;
            let alpha = read_sample(row, column, 3, 4, depth)?;
            (
                widen(red, depth)?,
                widen(green, depth)?,
                widen(blue, depth)?,
                widen(alpha, depth)?,
            )
        }
        ColorKind::Indexed => {
            let sample = read_sample(row, column, 0, 1, depth)?;
            let index = usize::from(sample);
            let Some(color) = target.palette.get(index) else {
                return Err(Error::Inconsistent("a pixel is outside the palette"));
            };
            let alpha = index_alpha(target.key, index)?;
            (color[0], color[1], color[2], alpha)
        }
    };
    let at = pixel_at(target.info.width, x, y)?;
    let Some(dest) = target.pixels.get_mut(at..at + 4) else {
        return Err(Error::Inconsistent(
            "the picture buffer does not match its size",
        ));
    };
    dest.copy_from_slice(&[red, green, blue, alpha]);
    Ok(())
}

fn grey_key(key: &Key, sample: u16, bit_depth: u8) -> Result<bool, Error> {
    match key {
        Key::None => Ok(false),
        Key::Grey(value) => Ok(sample == (value & significant(bit_depth))),
        Key::Rgb(_) | Key::Index(_) => Err(Error::Inconsistent(
            "the transparency key does not match the color type",
        )),
    }
}

fn rgb_key(key: &Key, red: u16, green: u16, blue: u16, bit_depth: u8) -> Result<bool, Error> {
    match key {
        Key::None => Ok(false),
        Key::Rgb(value) => {
            let mask = significant(bit_depth);
            Ok(red == (value[0] & mask) && green == (value[1] & mask) && blue == (value[2] & mask))
        }
        Key::Grey(_) | Key::Index(_) => Err(Error::Inconsistent(
            "the transparency key does not match the color type",
        )),
    }
}

fn index_alpha(key: &Key, index: usize) -> Result<u8, Error> {
    match key {
        Key::None => Ok(255),
        Key::Index(alphas) => Ok(alphas.get(index).copied().unwrap_or(255)),
        Key::Grey(_) | Key::Rgb(_) => Err(Error::Inconsistent(
            "the transparency key does not match the color type",
        )),
    }
}

/// PNG 11.3.1.1. Bits above the sample in a key are ignored.
fn significant(bit_depth: u8) -> u16 {
    match bit_depth {
        16 => u16::MAX,
        1 | 2 | 4 | 8 => (1u16 << bit_depth) - 1,
        _ => 0,
    }
}

/// PNG 12.4. Narrow samples repeat their high bits. A 16-bit sample keeps the top eight.
fn widen(sample: u16, bit_depth: u8) -> Result<u8, Error> {
    match bit_depth {
        16 => u8::try_from(sample >> 8).map_err(|_| Error::Inconsistent("a sample does not fit")),
        8 => u8::try_from(sample).map_err(|_| Error::Inconsistent("a sample does not fit")),
        1 | 2 | 4 => {
            let shift = 8 - bit_depth;
            let narrowed =
                u8::try_from(sample).map_err(|_| Error::Inconsistent("a sample does not fit"))?;
            let mut value = narrowed << shift;
            let mut filled = bit_depth;
            while filled < 8 {
                value |= value >> filled;
                filled = filled.saturating_mul(2);
            }
            Ok(value)
        }
        _ => Err(Error::Unsupported(
            "the bit depth is not allowed for the color type",
        )),
    }
}

fn read_sample(
    row: &[u8],
    pixel: u32,
    channel: u8,
    channels: u8,
    bit_depth: u8,
) -> Result<u16, Error> {
    let index = u64::from(pixel)
        .checked_mul(u64::from(channels))
        .and_then(|count| count.checked_add(u64::from(channel)))
        .ok_or(Error::TooLarge)?;
    if bit_depth == 16 {
        let at = usize::try_from(index.checked_mul(2).ok_or(Error::TooLarge)?)
            .map_err(|_| Error::TooLarge)?;
        let Some(bytes) = row.get(at..at + 2) else {
            return Err(Error::Inconsistent("a sample runs past the scanline"));
        };
        return Ok(u16::from_be_bytes([bytes[0], bytes[1]]));
    }
    if bit_depth == 8 {
        let at = usize::try_from(index).map_err(|_| Error::TooLarge)?;
        return row
            .get(at)
            .copied()
            .map(u16::from)
            .ok_or(Error::Inconsistent("a sample runs past the scanline"));
    }
    if !matches!(bit_depth, 1 | 2 | 4) {
        return Err(Error::Unsupported(
            "the bit depth is not allowed for the color type",
        ));
    }
    let bit_index = index
        .checked_mul(u64::from(bit_depth))
        .ok_or(Error::TooLarge)?;
    let at = usize::try_from(bit_index / 8).map_err(|_| Error::TooLarge)?;
    let Some(byte) = row.get(at).copied() else {
        return Err(Error::Inconsistent("a sample runs past the scanline"));
    };
    let offset = u8::try_from(bit_index % 8).expect("a bit offset within a byte fits in a byte");
    let Some(shift) = 8u8.checked_sub(bit_depth.saturating_add(offset)) else {
        return Err(Error::Inconsistent("a sample runs past the scanline"));
    };
    let mask = (1u8 << bit_depth) - 1;
    Ok(u16::from((byte >> shift) & mask))
}

fn pixel_at(width: u32, x: u32, y: u32) -> Result<usize, Error> {
    let width = usize::try_from(width).map_err(|_| Error::TooLarge)?;
    let x = usize::try_from(x).map_err(|_| Error::TooLarge)?;
    let y = usize::try_from(y).map_err(|_| Error::TooLarge)?;
    y.checked_mul(width)
        .and_then(|row| row.checked_add(x))
        .and_then(|pixel| pixel.checked_mul(4))
        .ok_or(Error::TooLarge)
}

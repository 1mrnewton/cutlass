//! Y′CbCr to the 8-bit RGBA working picture.
//!
//! Limited range follows ITU-R BT.601 / BT.709 digital levels: 8-bit luma is
//! `(Y − 16) / 219` and chroma is `(C − 128) / 224`. 10-bit uses the same
//! formulas scaled by four (`64`, `876`, `896`, neutral `512`). Full range
//! divides by `2^n − 1`, with chroma neutral at `2^(n−1)`, as in ITU-R BT.2100.
//!
//! The inverse matrix is the one in ITU-R BT.601, BT.709, and BT.2020:
//! `R = Y + 2 (1 − Kr) Cr`, and the same form for B and G. Kr and Kb are the
//! spec luma weights. Each chroma sample covers its subsampled cell with no
//! interpolation filter: 4:2:0 is the top-left 2×2, 4:2:2 is the left-right pair.

use cutlass_codec::access::{ColorRange, PixelFormat};

use crate::transfer;
use crate::{Depth, Error, Layout, Matrix, Picture, Request, Working};

const MAX_EDGE: u32 = crate::MAX_EDGE;

pub(crate) fn working(picture: &Picture<'_>, request: &Request) -> Result<Working, Error> {
    validate(picture)?;
    let (primary, blend) = transfer::mappers(picture, request)?;
    let pixels = usize::try_from(picture.width)
        .ok()
        .and_then(|width| usize::try_from(picture.height).ok()?.checked_mul(width))
        .and_then(|count| count.checked_mul(4))
        .ok_or(Error::Limit)?;
    let mut rgba = vec![0_u8; pixels];
    let chroma = chroma_size(picture.layout, picture.width, picture.height)?;
    let y_stride = stride(picture.y_stride, picture.width, picture.depth)?;
    let c_stride = stride(picture.chroma_stride, chroma.0, picture.depth)?;
    require_plane(picture.y, y_stride, picture.height)?;
    require_plane(picture.u, c_stride, chroma.1)?;
    require_plane(picture.v, c_stride, chroma.1)?;

    let mut index = 0_usize;
    for row in 0..picture.height {
        for col in 0..picture.width {
            let encoded = encoded_rgb(picture, y_stride, c_stride, col, row)?;
            let mut signal = primary.map(encoded);
            if let Some(other) = &blend {
                let base = other.map(encoded);
                signal = lerp(base, signal, request.segment.intensity);
            }
            let rgb = quantize(signal);
            let slot = rgba.get_mut(index..index + 4).ok_or(Error::Length)?;
            slot[0] = rgb[0];
            slot[1] = rgb[1];
            slot[2] = rgb[2];
            slot[3] = 255;
            index += 4;
        }
    }

    Ok(Working {
        width: picture.width,
        height: picture.height,
        format: PixelFormat::Rgba8,
        rgba,
        draft: request.draft,
        segment: request.segment,
    })
}

fn validate(picture: &Picture<'_>) -> Result<(), Error> {
    if picture.width == 0 || picture.height == 0 {
        return Err(Error::Empty);
    }
    if picture.width > MAX_EDGE || picture.height > MAX_EDGE {
        return Err(Error::Limit);
    }
    if picture.layout == Layout::Nv12 {
        return Err(Error::Layout(Layout::Nv12));
    }
    chroma_size(picture.layout, picture.width, picture.height)?;
    Ok(())
}

fn chroma_size(layout: Layout, width: u32, height: u32) -> Result<(u32, u32), Error> {
    match layout {
        Layout::Yuv420 => {
            if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
                return Err(Error::Odd);
            }
            Ok((width / 2, height / 2))
        }
        Layout::Yuv422 => {
            if !width.is_multiple_of(2) {
                return Err(Error::Odd);
            }
            Ok((width / 2, height))
        }
        Layout::Yuv444 => Ok((width, height)),
        Layout::Nv12 => Err(Error::Layout(Layout::Nv12)),
    }
}

fn stride(given: usize, width: u32, depth: Depth) -> Result<usize, Error> {
    let row = row_bytes(width, depth)?;
    if given == 0 {
        return Ok(row);
    }
    if given < row {
        return Err(Error::Length);
    }
    Ok(given)
}

fn row_bytes(width: u32, depth: Depth) -> Result<usize, Error> {
    let samples = usize::try_from(width).map_err(|_| Error::Limit)?;
    let bytes = match depth {
        Depth::Eight => 1,
        Depth::Ten => 2,
    };
    samples.checked_mul(bytes).ok_or(Error::Limit)
}

fn require_plane(samples: &[u8], stride: usize, height: u32) -> Result<(), Error> {
    let rows = usize::try_from(height).map_err(|_| Error::Limit)?;
    let need = stride.checked_mul(rows).ok_or(Error::Limit)?;
    if samples.len() < need {
        return Err(Error::Length);
    }
    Ok(())
}

fn encoded_rgb(
    picture: &Picture<'_>,
    y_stride: usize,
    c_stride: usize,
    x: u32,
    y: u32,
) -> Result<[f64; 3], Error> {
    let (cx, cy) = chroma_at(picture.layout, x, y);
    let luma = sample(picture.y, y_stride, x, y, picture.depth)?;
    let cb = sample(picture.u, c_stride, cx, cy, picture.depth)?;
    let cr = sample(picture.v, c_stride, cx, cy, picture.depth)?;
    Ok(ycbcr_to_rgb(
        luma,
        cb,
        cr,
        picture.depth,
        limited(picture.range),
        picture.matrix,
    ))
}

fn chroma_at(layout: Layout, x: u32, y: u32) -> (u32, u32) {
    match layout {
        Layout::Yuv420 => (x / 2, y / 2),
        Layout::Yuv422 => (x / 2, y),
        Layout::Yuv444 | Layout::Nv12 => (x, y),
    }
}

fn sample(plane: &[u8], stride: usize, x: u32, y: u32, depth: Depth) -> Result<u16, Error> {
    let row = usize::try_from(y).map_err(|_| Error::Limit)?;
    let col = usize::try_from(x).map_err(|_| Error::Limit)?;
    let origin = row.checked_mul(stride).ok_or(Error::Limit)?;
    match depth {
        Depth::Eight => {
            let index = origin.checked_add(col).ok_or(Error::Limit)?;
            plane
                .get(index)
                .copied()
                .map(u16::from)
                .ok_or(Error::Length)
        }
        Depth::Ten => {
            let index = origin
                .checked_add(col.checked_mul(2).ok_or(Error::Limit)?)
                .ok_or(Error::Limit)?;
            let Some(bytes) = plane.get(index..index + 2) else {
                return Err(Error::Length);
            };
            let Some(pair) = bytes.first().copied().zip(bytes.get(1).copied()) else {
                return Err(Error::Length);
            };
            let value = u16::from_le_bytes([pair.0, pair.1]);
            if value > 1023 {
                return Err(Error::Range);
            }
            Ok(value)
        }
    }
}

fn limited(range: ColorRange) -> bool {
    !matches!(range, ColorRange::Full)
}

fn ycbcr_to_rgb(y: u16, cb: u16, cr: u16, depth: Depth, limited: bool, matrix: Matrix) -> [f64; 3] {
    let bits = match depth {
        Depth::Eight => 8,
        Depth::Ten => 10,
    };
    let (kr, kb) = weights(matrix);
    let ey = expand(f64::from(y), bits, limited, false);
    let ecb = expand(f64::from(cb), bits, limited, true);
    let ecr = expand(f64::from(cr), bits, limited, true);
    let kg = 1.0 - kr - kb;
    let red = ey + 2.0 * (1.0 - kr) * ecr;
    let blue = ey + 2.0 * (1.0 - kb) * ecb;
    let green = ey - 2.0 * kb * (1.0 - kb) / kg * ecb - 2.0 * kr * (1.0 - kr) / kg * ecr;
    [clamp01(red), clamp01(green), clamp01(blue)]
}

fn weights(matrix: Matrix) -> (f64, f64) {
    match matrix {
        Matrix::Bt709 => (0.2126, 0.0722),
        Matrix::Bt601 => (0.299, 0.114),
        Matrix::Bt2020 => (0.2627, 0.0593),
    }
}

fn expand(code: f64, bits: u32, limited: bool, chroma: bool) -> f64 {
    let max_code = f64::from((1_u32 << bits) - 1);
    if !limited {
        if chroma {
            let neutral = f64::from(1_u32 << (bits - 1));
            return (code - neutral) / max_code;
        }
        return code / max_code;
    }
    let shift = f64::from(1_u32 << (bits - 8));
    if chroma {
        let neutral = f64::from(1_u32 << (bits - 1));
        return (code - neutral) / (224.0 * shift);
    }
    (code - 16.0 * shift) / (219.0 * shift)
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn lerp(from: [f64; 3], to: [f64; 3], t: f64) -> [f64; 3] {
    [
        from[0] * (1.0 - t) + to[0] * t,
        from[1] * (1.0 - t) + to[1] * t,
        from[2] * (1.0 - t) + to[2] * t,
    ]
}

pub(crate) fn quantize(signal: [f64; 3]) -> [u8; 3] {
    [byte(signal[0]), byte(signal[1]), byte(signal[2])]
}

fn byte(unit: f64) -> u8 {
    let rounded = (clamp01(unit) * 255.0).round();
    if rounded >= 255.0 {
        255
    } else if rounded <= 0.0 {
        0
    } else {
        // `rounded` is an integer in 1..=254.
        u8::try_from(rounded as u16).unwrap_or(0)
    }
}

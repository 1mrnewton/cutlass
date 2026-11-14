//! A PNG picture from a color strike, decoded to premultiplied RGBA.
//!
//! The samples are read as sRGB whatever gamma or profile the picture
//! names: the fonts that store pictures store them in sRGB. A picture is
//! refused before it is decoded when its header declares more than the
//! atlas holds, so a hostile header cannot ask for the memory.

use cutlass_decode::{
    BitDepth, CodecError, CodecId, CodecParams, ColorType, Decoder, Packet, StillImage,
    Transparency,
};
use cutlass_font::FontError;
use cutlass_png::PngDecoder;

use crate::TextError;

/// Most bytes one encoded picture may have. A picture as large as the
/// atlas holds, stored with no compression at 16 bits, is an eighth of this.
const MAX_PNG_BYTES: usize = 4 * 1024 * 1024;

/// Decode `png`, whose header declares `width` by `height`, to premultiplied
/// RGBA with row 0 at the top. Each edge is at most `max_edge`.
pub(super) fn decode(
    png: Vec<u8>,
    width: u32,
    height: u32,
    max_edge: u32,
) -> Result<Vec<u8>, TextError> {
    if width == 0 || height == 0 || width > max_edge || height > max_edge {
        return Err(TextError::Limit);
    }
    if png.len() > MAX_PNG_BYTES {
        return Err(TextError::Limit);
    }
    let mut decoder =
        PngDecoder::open(&CodecParams::video(CodecId::Png, width, height)).map_err(refusal)?;
    let pictures = decoder
        .decode_picture(&Packet::video(png, 0, 0, 0, true))
        .map_err(refusal)?;
    let Some(picture) = pictures.into_iter().next() else {
        return Err(TextError::Font(FontError::BadTable));
    };
    let image = picture.image;
    // The reader of the font and the decoder read the same header.
    if image.width != width || image.height != height {
        return Err(TextError::Font(FontError::BadTable));
    }
    premultiplied(&image)
}

fn refusal(error: CodecError) -> TextError {
    match error {
        CodecError::Limit { .. } => TextError::Limit,
        CodecError::Truncated { .. } | CodecError::NeedMoreData => {
            TextError::Font(FontError::Truncated)
        }
        CodecError::Unsupported { .. } => TextError::Font(FontError::Unsupported),
        _ => TextError::Font(FontError::BadTable),
    }
}

/// Every color type and bit depth a PNG has, as 8-bit premultiplied RGBA.
fn premultiplied(image: &StillImage) -> Result<Vec<u8>, TextError> {
    let pixels = usize::try_from(u64::from(image.width) * u64::from(image.height))
        .map_err(|_| TextError::Limit)?;
    let bytes = pixels.checked_mul(4).ok_or(TextError::Limit)?;
    let mut out = Vec::with_capacity(bytes);
    let depth = image.depth;
    let key = match &image.transparency {
        Transparency::ColorKey(key) => Some(key.as_slice()),
        Transparency::None | Transparency::PaletteAlpha(_) => None,
    };
    let keyed = |pixel: &[u16]| if key == Some(pixel) { 0 } else { u8::MAX };
    match image.color {
        ColorType::Gray => {
            for pixel in image.samples.as_chunks::<1>().0 {
                let gray = eight(pixel[0], depth);
                push(&mut out, [gray, gray, gray, keyed(pixel)]);
            }
        }
        ColorType::GrayAlpha => {
            for [gray, alpha] in image.samples.as_chunks::<2>().0 {
                let gray = eight(*gray, depth);
                push(&mut out, [gray, gray, gray, eight(*alpha, depth)]);
            }
        }
        ColorType::Rgb => {
            for pixel in image.samples.as_chunks::<3>().0 {
                let [red, green, blue] = pixel.map(|sample| eight(sample, depth));
                push(&mut out, [red, green, blue, keyed(pixel)]);
            }
        }
        ColorType::Rgba => {
            for pixel in image.samples.as_chunks::<4>().0 {
                push(&mut out, pixel.map(|sample| eight(sample, depth)));
            }
        }
        ColorType::Indexed => {
            let alphas = match &image.transparency {
                Transparency::PaletteAlpha(alphas) => alphas.as_slice(),
                Transparency::None | Transparency::ColorKey(_) => &[],
            };
            for index in &image.samples {
                let index = usize::from(*index);
                let Some(entry) = image.palette.get(index) else {
                    return Err(TextError::Font(FontError::BadTable));
                };
                // An entry past the alpha table is opaque.
                let alpha = alphas.get(index).copied().unwrap_or(u8::MAX);
                push(&mut out, [entry.red, entry.green, entry.blue, alpha]);
            }
        }
    }
    if out.len() != bytes {
        return Err(TextError::Font(FontError::BadTable));
    }
    Ok(out)
}

/// A sample of `depth` bits as eight, rounded.
fn eight(sample: u16, depth: BitDepth) -> u8 {
    let max = u32::from(depth.max_sample());
    let scaled = (u32::from(sample).min(max) * 255 + max / 2) / max;
    u8::try_from(scaled).unwrap_or(u8::MAX)
}

/// Append one straight RGBA pixel, premultiplied.
fn push(out: &mut Vec<u8>, [red, green, blue, alpha]: [u8; 4]) {
    let scale = |channel: u8| {
        let scaled = (u16::from(channel) * u16::from(alpha) + 127) / 255;
        u8::try_from(scaled).unwrap_or(u8::MAX)
    };
    out.extend([scale(red), scale(green), scale(blue), alpha]);
}

/// Flip an RGBA picture `width` pixels wide left to right, in place.
pub(super) fn mirror(rgba: &mut [u8], width: u32) -> Result<(), TextError> {
    let row = usize::try_from(width)
        .ok()
        .and_then(|width| width.checked_mul(4))
        .ok_or(TextError::Limit)?;
    if row == 0 {
        return Ok(());
    }
    for line in rgba.chunks_exact_mut(row) {
        line.as_chunks_mut::<4>().0.reverse();
    }
    Ok(())
}

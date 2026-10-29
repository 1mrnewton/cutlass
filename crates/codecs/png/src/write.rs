//! PNG datastream writer.
//!
//! The zlib stream is one final deflate block. Each scanline uses
//! the filter type with the smallest sum of absolute byte values. Adam7 is
//! emitted only when the caller asks for it.

use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::{CodecError, ColorType, ImageError, StillImage, TextItem, Transparency};

use crate::crc::{crc32, crc32_update};
use crate::limits::edge_ok;
use crate::raster::{
    Pass, adam_pass, bytes_per_pixel, filter_into, full_pass, pack, pixel_bits, pixel_offset,
    row_bytes,
};
use crate::zlib::deflate;

const IDAT_SPLIT: usize = 8_192;

pub(crate) fn encode_png(
    image: &StillImage,
    interlace: bool,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, CodecError> {
    if cancel.load(Ordering::Relaxed) {
        return Err(CodecError::Cancelled);
    }
    image.validate().map_err(image_err)?;
    if !edge_ok(image.width, image.height) {
        return Err(CodecError::Limit { what: "png image" });
    }
    let filtered = filter_image(image, interlace, cancel)?;
    let compressed = deflate(&filtered);
    let mut out = Vec::with_capacity(compressed.len() + 256);
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
    push_chunk(&mut out, b"IHDR", &ihdr(image, interlace))?;
    if let Some(values) = image.chromaticity {
        let mut data = [0u8; 32];
        for (index, value) in values.iter().enumerate() {
            data[index * 4..index * 4 + 4].copy_from_slice(&value.to_be_bytes());
        }
        push_chunk(&mut out, b"cHRM", &data)?;
    }
    if let Some(gamma) = image.gamma_times_100000 {
        push_chunk(&mut out, b"gAMA", &gamma.to_be_bytes())?;
    }
    if !image.icc_profile.is_empty() {
        push_chunk(&mut out, b"iCCP", &iccp(image)?)?;
    }
    if let Some(intent) = image.srgb_intent {
        push_chunk(&mut out, b"sRGB", &[intent])?;
    }
    if !image.significant_bits.is_empty() {
        push_chunk(&mut out, b"sBIT", &image.significant_bits)?;
    }
    if !image.palette.is_empty() {
        let mut data = Vec::with_capacity(image.palette.len() * 3);
        for entry in &image.palette {
            data.extend_from_slice(&[entry.red, entry.green, entry.blue]);
        }
        push_chunk(&mut out, b"PLTE", &data)?;
    }
    if let Some(data) = trns(image) {
        push_chunk(&mut out, b"tRNS", &data)?;
    }
    if let Some(data) = background(image) {
        push_chunk(&mut out, b"bKGD", &data)?;
    }
    if let Some(physical) = image.physical {
        let mut data = [0u8; 9];
        data[0..4].copy_from_slice(&physical.x.to_be_bytes());
        data[4..8].copy_from_slice(&physical.y.to_be_bytes());
        data[8] = physical.unit;
        push_chunk(&mut out, b"pHYs", &data)?;
    }
    for text in &image.texts {
        write_text(&mut out, text)?;
    }
    let mut offset = 0;
    while offset < compressed.len() {
        let end = (offset + IDAT_SPLIT).min(compressed.len());
        push_chunk(&mut out, b"IDAT", &compressed[offset..end])?;
        offset = end;
    }
    if compressed.is_empty() {
        push_chunk(&mut out, b"IDAT", &[])?;
    }
    push_chunk(&mut out, b"IEND", &[])?;
    Ok(out)
}

fn filter_image(
    image: &StillImage,
    interlace: bool,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, CodecError> {
    let channels = image.color.channels();
    let bits = pixel_bits(channels, image.depth);
    let bpp = bytes_per_pixel(bits);
    let mut out = Vec::new();
    if interlace {
        for index in 0..7 {
            let pass = adam_pass(index, image.width, image.height);
            if pass.cols == 0 || pass.rows == 0 {
                continue;
            }
            write_pass(image, cancel, &mut out, &pass, channels, bits, bpp)?;
        }
    } else {
        let pass = full_pass(image.width, image.height);
        write_pass(image, cancel, &mut out, &pass, channels, bits, bpp)?;
    }
    Ok(out)
}

fn write_pass(
    image: &StillImage,
    cancel: &AtomicBool,
    out: &mut Vec<u8>,
    pass: &Pass,
    channels: usize,
    bits: u32,
    bpp: usize,
) -> Result<(), CodecError> {
    let row_len = row_bytes(pass.cols, bits).ok_or(CodecError::Limit { what: "png image" })?;
    let mut prev = vec![0u8; row_len];
    let mut raw = vec![0u8; row_len];
    let mut trial = vec![0u8; row_len];
    let mut best = vec![0u8; row_len];
    let mut gathered = vec![0u16; usize::try_from(pass.cols).unwrap_or(0) * channels];
    for row_index in 0..pass.rows {
        if row_index % 32 == 0 && cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let y = pass.y0 + row_index * pass.dy;
        for col in 0..pass.cols {
            let x = pass.x0 + col * pass.dx;
            let from = pixel_offset(image.width, x, y, channels)
                .ok_or(CodecError::Limit { what: "png image" })?;
            let into = usize::try_from(col).unwrap_or(0) * channels;
            let end = from.saturating_add(channels);
            let into_end = into.saturating_add(channels);
            let Some(source) = image.samples.get(from..end) else {
                return Err(CodecError::InvalidApi { what: "png image" });
            };
            let Some(dest) = gathered.get_mut(into..into_end) else {
                return Err(CodecError::InvalidApi { what: "png image" });
            };
            dest.copy_from_slice(source);
        }
        pack(&gathered, image.depth, &mut raw);
        let (kind, filtered) = choose_filter(&raw, &prev, bpp, &mut trial, &mut best);
        out.push(kind);
        out.extend_from_slice(filtered);
        prev.copy_from_slice(&raw);
    }
    Ok(())
}

fn choose_filter<'a>(
    src: &[u8],
    prev: &[u8],
    bpp: usize,
    trial: &'a mut [u8],
    best: &'a mut [u8],
) -> (u8, &'a [u8]) {
    let mut best_kind = 0u8;
    let mut best_cost = i64::MAX;
    for kind in 0..5u8 {
        filter_into(kind, src, prev, bpp, trial);
        let mut cost = i64::from(kind);
        for &byte in trial.iter() {
            cost += i64::from((byte as i8).unsigned_abs());
        }
        if cost < best_cost {
            best_cost = cost;
            best_kind = kind;
            best.copy_from_slice(trial);
        }
    }
    (best_kind, best)
}

fn ihdr(image: &StillImage, interlace: bool) -> [u8; 13] {
    let mut data = [0u8; 13];
    data[0..4].copy_from_slice(&image.width.to_be_bytes());
    data[4..8].copy_from_slice(&image.height.to_be_bytes());
    data[8] = image.depth.bits();
    data[9] = match image.color {
        ColorType::Gray => 0,
        ColorType::Rgb => 2,
        ColorType::Indexed => 3,
        ColorType::GrayAlpha => 4,
        ColorType::Rgba => 6,
    };
    data[12] = u8::from(interlace);
    data
}

fn iccp(image: &StillImage) -> Result<Vec<u8>, CodecError> {
    let mut data = Vec::with_capacity(image.icc_name.len() + image.icc_profile.len() + 16);
    data.extend_from_slice(&image.icc_name);
    data.push(0);
    data.push(0);
    data.extend_from_slice(&deflate(&image.icc_profile));
    Ok(data)
}

fn trns(image: &StillImage) -> Option<Vec<u8>> {
    match &image.transparency {
        Transparency::None => None,
        Transparency::ColorKey(key) => Some(samples_be(key)),
        Transparency::PaletteAlpha(alpha) => Some(alpha.clone()),
    }
}

fn background(image: &StillImage) -> Option<Vec<u8>> {
    if image.background.is_empty() {
        return None;
    }
    if image.color == ColorType::Indexed {
        return Some(vec![image.background[0] as u8]);
    }
    Some(samples_be(&image.background))
}

fn samples_be(samples: &[u16]) -> Vec<u8> {
    let mut data = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        data.extend_from_slice(&sample.to_be_bytes());
    }
    data
}

fn write_text(out: &mut Vec<u8>, text: &TextItem) -> Result<(), CodecError> {
    let international = !text.language.is_empty() || !text.translated.is_empty();
    if international {
        let mut data = Vec::new();
        data.extend_from_slice(&text.keyword);
        data.push(0);
        data.push(u8::from(text.compressed));
        data.push(0);
        data.extend_from_slice(&text.language);
        data.push(0);
        data.extend_from_slice(&text.translated);
        data.push(0);
        if text.compressed {
            data.extend_from_slice(&deflate(&text.text));
        } else {
            data.extend_from_slice(&text.text);
        }
        push_chunk(out, b"iTXt", &data)
    } else if text.compressed {
        let mut data = Vec::new();
        data.extend_from_slice(&text.keyword);
        data.push(0);
        data.push(0);
        data.extend_from_slice(&deflate(&text.text));
        push_chunk(out, b"zTXt", &data)
    } else {
        let mut data = Vec::new();
        data.extend_from_slice(&text.keyword);
        data.push(0);
        data.extend_from_slice(&text.text);
        push_chunk(out, b"tEXt", &data)
    }
}

fn push_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) -> Result<(), CodecError> {
    let len = u32::try_from(data.len()).map_err(|_| CodecError::Limit { what: "chunk" })?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let crc = crc32_update(crc32(kind), data);
    out.extend_from_slice(&crc.to_be_bytes());
    Ok(())
}

fn image_err(err: ImageError) -> CodecError {
    match err {
        ImageError::Dimension => CodecError::Limit { what: "png image" },
        _ => CodecError::InvalidApi { what: "png image" },
    }
}

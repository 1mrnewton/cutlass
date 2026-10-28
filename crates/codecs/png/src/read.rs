//! PNG datastream reader.
//!
//! One buffer is one datastream: signature, chunks, and the concatenated
//! IDAT zlib stream. Samples are reconstructed in the declared domain.

use cutlass_decode::{
    BitDepth, CodecError, ColorType, ImageError, PhysicalPixel, Rgb8, StillImage, TextItem,
    Transparency,
};

use crate::crc::{crc32, crc32_update};
use crate::limits::{MAX_CHUNK, MAX_IDAT, MAX_INFLATED_META, edge_ok};
use crate::raster::{
    Pass, adam_pass, bytes_per_pixel, filtered_len, full_pass, pixel_bits, pixel_offset, row_bytes,
    sample_count, unfilter, unpack,
};
use crate::zlib::{ZlibError, inflate};

const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

pub(crate) fn decode_png(bytes: &[u8]) -> Result<StillImage, CodecError> {
    let parsed = parse(bytes)?;
    let channels = parsed.color.channels();
    let bits = pixel_bits(channels, parsed.depth);
    let expect = filtered_len(parsed.width, parsed.height, bits, parsed.interlace)
        .ok_or(CodecError::Limit { what: "png image" })?;
    let raw = inflate(&parsed.idat, expect).map_err(zlib_err)?;
    if raw.len() != expect {
        return Err(malformed(0, "idat length"));
    }
    let samples = reconstruct(&parsed, &raw)?;
    let mut image = StillImage::from_samples(
        parsed.width,
        parsed.height,
        parsed.color,
        parsed.depth,
        samples,
    );
    image.palette = parsed.palette;
    image.transparency = parsed.transparency;
    image.gamma_times_100000 = parsed.gamma;
    image.chromaticity = parsed.chromaticity;
    image.srgb_intent = parsed.srgb;
    image.icc_name = parsed.icc_name;
    image.icc_profile = parsed.icc_profile;
    image.significant_bits = parsed.significant_bits;
    image.background = parsed.background;
    image.physical = parsed.physical;
    image.texts = parsed.texts;
    image.checked().map_err(image_err)
}

struct Parsed {
    width: u32,
    height: u32,
    depth: BitDepth,
    color: ColorType,
    interlace: bool,
    palette: Vec<Rgb8>,
    transparency: Transparency,
    gamma: Option<u32>,
    chromaticity: Option<[u32; 8]>,
    srgb: Option<u8>,
    icc_name: Vec<u8>,
    icc_profile: Vec<u8>,
    significant_bits: Vec<u8>,
    background: Vec<u16>,
    physical: Option<PhysicalPixel>,
    texts: Vec<TextItem>,
    idat: Vec<u8>,
}

struct Flags {
    plte: bool,
    idat: bool,
    idat_closed: bool,
    trns: bool,
    bkgd: bool,
    gama: bool,
    chrm: bool,
    iccp: bool,
    srgb: bool,
    sbit: bool,
    phys: bool,
}

fn parse(bytes: &[u8]) -> Result<Parsed, CodecError> {
    if bytes.len() < 8 || bytes[..8] != SIGNATURE {
        return Err(malformed(0, "signature"));
    }
    let mut at = 8usize;
    let mut header = None;
    let mut parsed = empty_parsed();
    let mut flags = Flags {
        plte: false,
        idat: false,
        idat_closed: false,
        trns: false,
        bkgd: false,
        gama: false,
        chrm: false,
        iccp: false,
        srgb: false,
        sbit: false,
        phys: false,
    };
    let mut ended = false;
    while !ended {
        if bytes.len().saturating_sub(at) < 12 {
            return Err(truncated(at, "chunk"));
        }
        let len = u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
        if u64::from(len) > MAX_CHUNK as u64 {
            return Err(CodecError::Limit { what: "png chunk" });
        }
        let len = len as usize;
        let total = len.saturating_add(12);
        if bytes.len().saturating_sub(at) < total {
            return Err(truncated(at, "chunk"));
        }
        let kind = [bytes[at + 4], bytes[at + 5], bytes[at + 6], bytes[at + 7]];
        let data = &bytes[at + 8..at + 8 + len];
        let crc_at = at + 8 + len;
        let stored = u32::from_be_bytes([
            bytes[crc_at],
            bytes[crc_at + 1],
            bytes[crc_at + 2],
            bytes[crc_at + 3],
        ]);
        let actual = crc32_update(crc32(&kind), data);
        if stored != actual {
            return Err(malformed(at, "crc"));
        }
        if kind == *b"IEND" {
            if !data.is_empty() || !flags.idat {
                return Err(malformed(at, "iend"));
            }
            ended = true;
        } else if ended {
            return Err(malformed(at, "trailing"));
        } else {
            on_chunk(at, kind, data, &mut header, &mut parsed, &mut flags)?;
        }
        at += total;
    }
    if at != bytes.len() {
        return Err(malformed(at, "trailing"));
    }
    let Some((width, height, depth, color, interlace)) = header else {
        return Err(malformed(0, "ihdr"));
    };
    if color == ColorType::Indexed && parsed.palette.is_empty() {
        return Err(malformed(0, "plte"));
    }
    if matches!(color, ColorType::Gray | ColorType::GrayAlpha) && !parsed.palette.is_empty() {
        return Err(malformed(0, "plte"));
    }
    parsed.width = width;
    parsed.height = height;
    parsed.depth = depth;
    parsed.color = color;
    parsed.interlace = interlace;
    Ok(parsed)
}

fn empty_parsed() -> Parsed {
    Parsed {
        width: 0,
        height: 0,
        depth: BitDepth::Eight,
        color: ColorType::Gray,
        interlace: false,
        palette: Vec::new(),
        transparency: Transparency::None,
        gamma: None,
        chromaticity: None,
        srgb: None,
        icc_name: Vec::new(),
        icc_profile: Vec::new(),
        significant_bits: Vec::new(),
        background: Vec::new(),
        physical: None,
        texts: Vec::new(),
        idat: Vec::new(),
    }
}

fn on_chunk(
    at: usize,
    kind: [u8; 4],
    data: &[u8],
    header: &mut Option<(u32, u32, BitDepth, ColorType, bool)>,
    parsed: &mut Parsed,
    flags: &mut Flags,
) -> Result<(), CodecError> {
    if header.is_none() && kind != *b"IHDR" {
        return Err(malformed(at, "ihdr"));
    }
    if flags.idat && kind != *b"IDAT" {
        flags.idat_closed = true;
    }
    if kind == *b"IDAT" && flags.idat_closed {
        return Err(malformed(at, "idat order"));
    }
    match &kind {
        b"IHDR" => *header = Some(read_ihdr(at, data, header.is_some())?),
        b"PLTE" => read_plte(at, data, flags, parsed)?,
        b"IDAT" => read_idat(data, flags, parsed)?,
        b"tRNS" => read_trns(at, data, flags, header, parsed)?,
        b"gAMA" => read_gama(at, data, flags, parsed)?,
        b"cHRM" => read_chrm(at, data, flags, parsed)?,
        b"iCCP" => read_iccp(at, data, flags, parsed)?,
        b"sRGB" => read_srgb(at, data, flags, parsed)?,
        b"sBIT" => read_sbit(at, data, flags, parsed)?,
        b"bKGD" => read_bkgd(at, data, flags, header, parsed)?,
        b"pHYs" => read_phys(at, data, flags, parsed)?,
        b"tEXt" => parsed.texts.push(read_text(at, data, false)?),
        b"zTXt" => parsed.texts.push(read_text(at, data, true)?),
        b"iTXt" => parsed.texts.push(read_itxt(at, data)?),
        _ => {
            if kind[0] & 0x20 == 0 {
                return Err(malformed(at, "critical chunk"));
            }
        }
    }
    Ok(())
}

fn read_ihdr(
    at: usize,
    data: &[u8],
    seen: bool,
) -> Result<(u32, u32, BitDepth, ColorType, bool), CodecError> {
    if seen || data.len() != 13 {
        return Err(malformed(at, "ihdr"));
    }
    let width = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    let height = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
    if width == 0 || height == 0 || width > 0x7fff_ffff || height > 0x7fff_ffff {
        return Err(malformed(at, "dimension"));
    }
    if !edge_ok(width, height) {
        return Err(CodecError::Limit { what: "png image" });
    }
    let depth = bit_depth(data[8]).ok_or(malformed(at, "bit depth"))?;
    let color = color_type(data[9]).ok_or(malformed(at, "color type"))?;
    if !depth_ok(color, depth) {
        return Err(malformed(at, "bit depth"));
    }
    if data[10] != 0 {
        return Err(CodecError::Unsupported {
            what: "png compression method",
        });
    }
    if data[11] != 0 {
        return Err(CodecError::Unsupported {
            what: "png filter method",
        });
    }
    let interlace = match data[12] {
        0 => false,
        1 => true,
        _ => {
            return Err(CodecError::Unsupported {
                what: "png interlace",
            });
        }
    };
    Ok((width, height, depth, color, interlace))
}

fn read_plte(
    at: usize,
    data: &[u8],
    flags: &mut Flags,
    parsed: &mut Parsed,
) -> Result<(), CodecError> {
    if flags.plte
        || flags.idat
        || flags.trns
        || flags.bkgd
        || data.is_empty()
        || !data.len().is_multiple_of(3)
    {
        return Err(malformed(at, "plte"));
    }
    if data.len() / 3 > 256 {
        return Err(malformed(at, "plte"));
    }
    parsed.palette = data
        .as_chunks::<3>()
        .0
        .iter()
        .map(|entry| Rgb8 {
            red: entry[0],
            green: entry[1],
            blue: entry[2],
        })
        .collect();
    flags.plte = true;
    Ok(())
}

fn read_idat(data: &[u8], flags: &mut Flags, parsed: &mut Parsed) -> Result<(), CodecError> {
    if parsed.idat.len().saturating_add(data.len()) > MAX_IDAT {
        return Err(CodecError::Limit { what: "idat" });
    }
    parsed.idat.extend_from_slice(data);
    flags.idat = true;
    Ok(())
}

fn read_trns(
    at: usize,
    data: &[u8],
    flags: &mut Flags,
    header: &Option<(u32, u32, BitDepth, ColorType, bool)>,
    parsed: &mut Parsed,
) -> Result<(), CodecError> {
    if flags.trns || flags.idat {
        return Err(malformed(at, "trns"));
    }
    let (_, _, depth, color, _) = header.ok_or(malformed(at, "ihdr"))?;
    parsed.transparency = match color {
        ColorType::Gray if data.len() == 2 => {
            Transparency::ColorKey(vec![mask_sample(be16(data), depth)])
        }
        ColorType::Rgb if data.len() == 6 => Transparency::ColorKey(vec![
            mask_sample(be16(&data[0..2]), depth),
            mask_sample(be16(&data[2..4]), depth),
            mask_sample(be16(&data[4..6]), depth),
        ]),
        ColorType::Indexed
            if flags.plte && !data.is_empty() && data.len() <= parsed.palette.len() =>
        {
            Transparency::PaletteAlpha(data.to_vec())
        }
        ColorType::GrayAlpha | ColorType::Rgba => {
            return Err(malformed(at, "trns"));
        }
        _ => return Err(malformed(at, "trns")),
    };
    flags.trns = true;
    Ok(())
}

fn read_gama(
    at: usize,
    data: &[u8],
    flags: &mut Flags,
    parsed: &mut Parsed,
) -> Result<(), CodecError> {
    if flags.gama || data.len() != 4 {
        return Err(malformed(at, "gama"));
    }
    before_palette(at, flags)?;
    let gamma = be32(data);
    if gamma > 0x7fff_ffff {
        return Err(malformed(at, "gama"));
    }
    parsed.gamma = Some(gamma);
    flags.gama = true;
    Ok(())
}

fn read_chrm(
    at: usize,
    data: &[u8],
    flags: &mut Flags,
    parsed: &mut Parsed,
) -> Result<(), CodecError> {
    if flags.chrm || data.len() != 32 {
        return Err(malformed(at, "chrm"));
    }
    before_palette(at, flags)?;
    let mut values = [0u32; 8];
    for (index, chunk) in data.as_chunks::<4>().0.iter().enumerate() {
        let value = be32(chunk);
        if value > 0x7fff_ffff {
            return Err(malformed(at, "chrm"));
        }
        values[index] = value;
    }
    parsed.chromaticity = Some(values);
    flags.chrm = true;
    Ok(())
}

fn read_iccp(
    at: usize,
    data: &[u8],
    flags: &mut Flags,
    parsed: &mut Parsed,
) -> Result<(), CodecError> {
    if flags.iccp {
        return Err(malformed(at, "iccp"));
    }
    before_palette(at, flags)?;
    let (name, rest) = split_null(data).ok_or(malformed(at, "iccp"))?;
    if rest.first() != Some(&0) {
        return Err(malformed(at, "iccp"));
    }
    let profile = inflate(&rest[1..], MAX_INFLATED_META).map_err(zlib_err)?;
    if profile.is_empty() {
        return Err(malformed(at, "iccp"));
    }
    parsed.icc_name = name.to_vec();
    parsed.icc_profile = profile;
    flags.iccp = true;
    Ok(())
}

fn read_srgb(
    at: usize,
    data: &[u8],
    flags: &mut Flags,
    parsed: &mut Parsed,
) -> Result<(), CodecError> {
    if flags.srgb || data.len() != 1 || data[0] > 3 {
        return Err(malformed(at, "srgb"));
    }
    before_palette(at, flags)?;
    parsed.srgb = Some(data[0]);
    flags.srgb = true;
    Ok(())
}

fn read_sbit(
    at: usize,
    data: &[u8],
    flags: &mut Flags,
    parsed: &mut Parsed,
) -> Result<(), CodecError> {
    if flags.sbit || data.is_empty() || data.len() > 4 {
        return Err(malformed(at, "sbit"));
    }
    before_palette(at, flags)?;
    parsed.significant_bits = data.to_vec();
    flags.sbit = true;
    Ok(())
}

fn read_bkgd(
    at: usize,
    data: &[u8],
    flags: &mut Flags,
    header: &Option<(u32, u32, BitDepth, ColorType, bool)>,
    parsed: &mut Parsed,
) -> Result<(), CodecError> {
    if flags.bkgd || flags.idat {
        return Err(malformed(at, "bkgd"));
    }
    let (_, _, depth, color, _) = header.ok_or(malformed(at, "ihdr"))?;
    if !flags.plte && matches!(color, ColorType::Indexed) {
        return Err(malformed(at, "bkgd"));
    }
    parsed.background = match color {
        ColorType::Gray | ColorType::GrayAlpha if data.len() == 2 => {
            vec![mask_sample(be16(data), depth)]
        }
        ColorType::Rgb | ColorType::Rgba if data.len() == 6 => vec![
            mask_sample(be16(&data[0..2]), depth),
            mask_sample(be16(&data[2..4]), depth),
            mask_sample(be16(&data[4..6]), depth),
        ],
        ColorType::Indexed if data.len() == 1 => vec![u16::from(data[0])],
        _ => return Err(malformed(at, "bkgd")),
    };
    flags.bkgd = true;
    Ok(())
}

fn read_phys(
    at: usize,
    data: &[u8],
    flags: &mut Flags,
    parsed: &mut Parsed,
) -> Result<(), CodecError> {
    if flags.phys || flags.idat || data.len() != 9 || data[8] > 1 {
        return Err(malformed(at, "phys"));
    }
    let x = be32(&data[0..4]);
    let y = be32(&data[4..8]);
    if x > 0x7fff_ffff || y > 0x7fff_ffff {
        return Err(malformed(at, "phys"));
    }
    parsed.physical = Some(PhysicalPixel {
        x,
        y,
        unit: data[8],
    });
    flags.phys = true;
    Ok(())
}

fn read_text(at: usize, data: &[u8], compressed: bool) -> Result<TextItem, CodecError> {
    let (keyword, rest) = split_null(data).ok_or(malformed(at, "text"))?;
    let text = if compressed {
        if rest.first() != Some(&0) {
            return Err(malformed(at, "text"));
        }
        inflate(&rest[1..], MAX_INFLATED_META).map_err(zlib_err)?
    } else {
        rest.to_vec()
    };
    Ok(TextItem {
        keyword: keyword.to_vec(),
        language: Vec::new(),
        translated: Vec::new(),
        text,
        compressed,
    })
}

fn read_itxt(at: usize, data: &[u8]) -> Result<TextItem, CodecError> {
    let (keyword, rest) = split_null(data).ok_or(malformed(at, "text"))?;
    if rest.len() < 2 {
        return Err(malformed(at, "text"));
    }
    let flag = rest[0];
    let method = rest[1];
    if flag > 1 || (flag == 0 && method != 0) || (flag == 1 && method != 0) {
        return Err(malformed(at, "text"));
    }
    let (language, after_language) = split_null(&rest[2..]).ok_or(malformed(at, "text"))?;
    let (translated, payload) = split_null(after_language).ok_or(malformed(at, "text"))?;
    let text = if flag == 1 {
        inflate(payload, MAX_INFLATED_META).map_err(zlib_err)?
    } else {
        payload.to_vec()
    };
    Ok(TextItem {
        keyword: keyword.to_vec(),
        language: language.to_vec(),
        translated: translated.to_vec(),
        text,
        compressed: flag == 1,
    })
}

fn before_palette(at: usize, flags: &Flags) -> Result<(), CodecError> {
    if flags.plte || flags.idat {
        Err(malformed(at, "chunk order"))
    } else {
        Ok(())
    }
}

fn reconstruct(parsed: &Parsed, raw: &[u8]) -> Result<Vec<u16>, CodecError> {
    let channels = parsed.color.channels();
    let count = sample_count(parsed.width, parsed.height, channels)
        .ok_or(CodecError::Limit { what: "png image" })?;
    let mut samples = vec![0u16; count];
    let mut cursor = 0usize;
    if parsed.interlace {
        for index in 0..7 {
            let pass = adam_pass(index, parsed.width, parsed.height);
            if pass.cols == 0 || pass.rows == 0 {
                continue;
            }
            decode_pass(parsed, raw, &mut samples, &mut cursor, &pass)?;
        }
    } else {
        let pass = full_pass(parsed.width, parsed.height);
        decode_pass(parsed, raw, &mut samples, &mut cursor, &pass)?;
    }
    if cursor != raw.len() {
        return Err(malformed(cursor, "idat length"));
    }
    Ok(samples)
}

fn decode_pass(
    parsed: &Parsed,
    raw: &[u8],
    samples: &mut [u16],
    cursor: &mut usize,
    pass: &Pass,
) -> Result<(), CodecError> {
    let channels = parsed.color.channels();
    let bits = pixel_bits(channels, parsed.depth);
    let bpp = bytes_per_pixel(bits);
    let row_len = row_bytes(pass.cols, bits).ok_or(CodecError::Limit { what: "png image" })?;
    let mut prev = vec![0u8; row_len];
    let mut row = vec![0u8; row_len];
    let mut row_samples = vec![0u16; usize::try_from(pass.cols).unwrap_or(0) * channels];
    for row_index in 0..pass.rows {
        if *cursor >= raw.len() {
            return Err(truncated(*cursor, "scanline"));
        }
        let filter = raw[*cursor];
        *cursor += 1;
        let end = cursor.saturating_add(row_len);
        if end > raw.len() {
            return Err(truncated(*cursor, "scanline"));
        }
        row.copy_from_slice(&raw[*cursor..end]);
        *cursor = end;
        if unfilter(filter, &mut row, &prev, bpp).is_err() {
            return Err(malformed(*cursor, "filter"));
        }
        unpack(&row, parsed.depth, &mut row_samples);
        let y = pass.y0 + row_index * pass.dy;
        for col in 0..pass.cols {
            let x = pass.x0 + col * pass.dx;
            let from = usize::try_from(col).unwrap_or(0) * channels;
            let to = pixel_offset(parsed.width, x, y, channels)
                .ok_or(CodecError::Limit { what: "png image" })?;
            let from_end = from.saturating_add(channels);
            let to_end = to.saturating_add(channels);
            let Some(source) = row_samples.get(from..from_end) else {
                return Err(malformed(*cursor, "scanline"));
            };
            let Some(dest) = samples.get_mut(to..to_end) else {
                return Err(malformed(*cursor, "scanline"));
            };
            dest.copy_from_slice(source);
        }
        prev.copy_from_slice(&row);
    }
    Ok(())
}

fn bit_depth(byte: u8) -> Option<BitDepth> {
    match byte {
        1 => Some(BitDepth::One),
        2 => Some(BitDepth::Two),
        4 => Some(BitDepth::Four),
        8 => Some(BitDepth::Eight),
        16 => Some(BitDepth::Sixteen),
        _ => None,
    }
}

fn color_type(byte: u8) -> Option<ColorType> {
    match byte {
        0 => Some(ColorType::Gray),
        2 => Some(ColorType::Rgb),
        3 => Some(ColorType::Indexed),
        4 => Some(ColorType::GrayAlpha),
        6 => Some(ColorType::Rgba),
        _ => None,
    }
}

fn depth_ok(color: ColorType, depth: BitDepth) -> bool {
    match color {
        ColorType::Gray => true,
        ColorType::Indexed => !matches!(depth, BitDepth::Sixteen),
        ColorType::Rgb | ColorType::GrayAlpha | ColorType::Rgba => {
            matches!(depth, BitDepth::Eight | BitDepth::Sixteen)
        }
    }
}

fn mask_sample(value: u16, depth: BitDepth) -> u16 {
    value & depth.max_sample()
}

fn be16(bytes: &[u8]) -> u16 {
    u16::from_be_bytes([bytes[0], bytes[1]])
}

fn be32(bytes: &[u8]) -> u32 {
    u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn split_null(bytes: &[u8]) -> Option<(&[u8], &[u8])> {
    let index = bytes.iter().position(|byte| *byte == 0)?;
    Some((&bytes[..index], &bytes[index + 1..]))
}

fn image_err(err: ImageError) -> CodecError {
    let what = match err {
        ImageError::Palette => "png index",
        ImageError::Metadata => "png metadata",
        ImageError::Transparency => "png transparency",
        _ => "png image",
    };
    CodecError::Malformed { at_bit: 0, what }
}

fn zlib_err(err: ZlibError) -> CodecError {
    match err {
        ZlibError::Truncated => CodecError::Truncated {
            at_bit: 0,
            what: "zlib",
        },
        ZlibError::Malformed(what) => CodecError::Malformed { at_bit: 0, what },
        ZlibError::Limit => CodecError::Limit { what: "zlib" },
    }
}

fn malformed(at: usize, what: &'static str) -> CodecError {
    CodecError::Malformed {
        at_bit: byte_bit(at),
        what,
    }
}

fn truncated(at: usize, what: &'static str) -> CodecError {
    CodecError::Truncated {
        at_bit: byte_bit(at),
        what,
    }
}

fn byte_bit(at: usize) -> u64 {
    u64::try_from(at).unwrap_or(u64::MAX).saturating_mul(8)
}

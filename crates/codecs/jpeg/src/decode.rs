//! JPEG marker parser, Huffman scans, and the decoder trait.
//!
//! One packet is one image. `FF 00` inside a scan is a data byte. Arithmetic
//! SOF markers and `DAC` are refused before any coefficient is written.

use cutlass_decode::{
    BitDepth, CodecError, CodecId, CodecParams, ColorType, Decoder, Output, Packet, StillImage,
    StillOutput,
};

use crate::bits::Cursor;
use crate::color::ycbcr_to_rgb;
use crate::dct::{dequant_block, idct_u8};
use crate::huffman::{HuffDecode, decode_symbol, decode_table};
use crate::limits::{MAX_AL, MAX_INPUT, MAX_MARKERS, dimensions_ok};
use crate::tables::ZIGZAG;

const SOF0: u8 = 0xc0;
const SOF1: u8 = 0xc1;
const SOF2: u8 = 0xc2;
const SOF3: u8 = 0xc3;
const DHT: u8 = 0xc4;
const SOF5: u8 = 0xc5;
const SOF6: u8 = 0xc6;
const SOF7: u8 = 0xc7;
const SOF9: u8 = 0xc9;
const SOF10: u8 = 0xca;
const SOF11: u8 = 0xcb;
const DAC: u8 = 0xcc;
const SOF13: u8 = 0xcd;
const SOF14: u8 = 0xce;
const SOF15: u8 = 0xcf;
const RST0: u8 = 0xd0;
const SOI: u8 = 0xd8;
const EOI: u8 = 0xd9;
const SOS: u8 = 0xda;
const DQT: u8 = 0xdb;
const DNL: u8 = 0xdc;
const DRI: u8 = 0xdd;
const APP0: u8 = 0xe0;
const APP1: u8 = 0xe1;
const APP14: u8 = 0xee;
const COM: u8 = 0xfe;

/// Size and coding of one JPEG frame.
///
/// `orientation` is the EXIF tag when APP1 carries one. It is not applied
/// to the samples. `width` and `height` are the frame header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JpegInfo {
    pub width: u32,
    pub height: u32,
    pub components: u8,
    pub progressive: bool,
    pub orientation: Option<u16>,
}

struct Comp {
    id: u8,
    h: u8,
    v: u8,
    quant: usize,
    blocks_x: usize,
    blocks_y: usize,
    coeffs: Vec<i32>,
    plane: [i8; 64],
}

struct Frame {
    progressive: bool,
    baseline: bool,
    width: u32,
    height: u32,
    max_h: u8,
    max_v: u8,
    comps: Vec<Comp>,
}

struct Meta {
    orientation: Option<u16>,
    adobe: Option<u8>,
}

#[derive(Clone, Copy)]
struct ScanComp {
    index: usize,
    dc_table: usize,
    ac_table: usize,
}

struct Scan {
    comps: Vec<ScanComp>,
    ss: usize,
    se: usize,
    ah: u8,
    al: u8,
}

#[derive(Clone, Copy)]
enum Space {
    Gray,
    Rgb,
    YCbCr,
}

/// Read the frame size without decoding coefficients.
pub fn read_jpeg_info(bytes: &[u8]) -> Result<JpegInfo, CodecError> {
    parse(bytes, false).map(|parsed| parsed.info)
}

pub(crate) fn decode_jpeg(bytes: &[u8]) -> Result<StillImage, CodecError> {
    let parsed = parse(bytes, true)?;
    parsed.image.ok_or(CodecError::Malformed {
        at_bit: 0,
        what: "scan",
    })
}

struct Parsed {
    info: JpegInfo,
    image: Option<StillImage>,
}

fn parse(bytes: &[u8], full: bool) -> Result<Parsed, CodecError> {
    if bytes.len() > MAX_INPUT {
        return Err(CodecError::Limit { what: "jpeg bytes" });
    }
    let mut input = Cursor::new(bytes);
    if input.take_marker()? != SOI {
        return Err(input.malformed("start of image"));
    }
    let mut frame: Option<Frame> = None;
    let mut quant = [None, None, None, None];
    let mut dc_huff: [Option<HuffDecode>; 4] = [None, None, None, None];
    let mut ac_huff: [Option<HuffDecode>; 4] = [None, None, None, None];
    let mut restart: u16 = 0;
    let mut meta = Meta {
        orientation: None,
        adobe: None,
    };
    let mut scans = 0u32;
    let mut markers = 0usize;

    loop {
        markers += 1;
        if markers > MAX_MARKERS {
            return Err(CodecError::Limit {
                what: "jpeg markers",
            });
        }
        let marker = match input.take_marker() {
            Ok(marker) => marker,
            Err(err) if !full && frame.is_some() => {
                return Ok(info_only(frame.as_ref(), &meta, err));
            }
            Err(err) => return Err(err),
        };
        if marker == EOI {
            break;
        }
        if is_arithmetic(marker) {
            return Err(CodecError::Unsupported {
                what: "arithmetic coding",
            });
        }
        match marker {
            SOF0 | SOF1 | SOF2 => {
                if frame.is_some() {
                    return Err(input.malformed("frame"));
                }
                let body = read_segment(&mut input)?;
                frame = Some(parse_sof(marker, &body, input.at())?);
            }
            DQT => {
                let body = read_segment(&mut input)?;
                parse_dqt(&body, &mut quant, input.at())?;
            }
            DHT => {
                let body = read_segment(&mut input)?;
                parse_dht(&body, &mut dc_huff, &mut ac_huff, input.at())?;
            }
            DRI => {
                let body = read_segment(&mut input)?;
                restart = parse_dri(&body, input.at())?;
            }
            SOS => {
                let Some(frame) = frame.as_mut() else {
                    return Err(input.malformed("frame"));
                };
                let body = read_segment(&mut input)?;
                if !full {
                    return Ok(Parsed {
                        info: info_from(frame, &meta),
                        image: None,
                    });
                }
                decode_scan(
                    &mut input, frame, &body, &quant, &dc_huff, &ac_huff, restart,
                )?;
                scans += 1;
            }
            APP0 | APP1 | APP14 => {
                let body = read_segment(&mut input)?;
                note_app(marker, &body, &mut meta);
            }
            COM | 0xe2..=0xed | 0xef => {
                let _ = read_segment(&mut input)?;
            }
            DNL => {
                return Err(CodecError::Unsupported { what: "dnl" });
            }
            SOF3 => {
                return Err(CodecError::Unsupported {
                    what: "lossless jpeg",
                });
            }
            SOF5 | SOF6 | SOF7 => {
                return Err(CodecError::Unsupported {
                    what: "differential jpeg",
                });
            }
            0xf7 => {
                return Err(CodecError::Unsupported { what: "jpeg-ls" });
            }
            0xc8 => {
                return Err(CodecError::Unsupported {
                    what: "reserved jpeg",
                });
            }
            0xde => {
                return Err(CodecError::Unsupported {
                    what: "hierarchical jpeg",
                });
            }
            RST0..=0xd7 | SOI => return Err(input.malformed("marker")),
            _ => {
                return Err(CodecError::Unsupported {
                    what: "jpeg marker",
                });
            }
        }
    }

    let Some(frame) = frame else {
        return Err(input.malformed("frame"));
    };
    if !full {
        return Ok(Parsed {
            info: info_from(&frame, &meta),
            image: None,
        });
    }
    if scans == 0 {
        return Err(input.malformed("scan"));
    }
    let image = render(&frame, &quant, &meta)?;
    Ok(Parsed {
        info: info_from(&frame, &meta),
        image: Some(image),
    })
}

fn info_only(frame: Option<&Frame>, meta: &Meta, err: CodecError) -> Parsed {
    let Some(frame) = frame else {
        return Parsed {
            info: JpegInfo {
                width: 0,
                height: 0,
                components: 0,
                progressive: false,
                orientation: None,
            },
            image: None,
        };
    };
    let _ = err;
    Parsed {
        info: info_from(frame, meta),
        image: None,
    }
}

fn info_from(frame: &Frame, meta: &Meta) -> JpegInfo {
    JpegInfo {
        width: frame.width,
        height: frame.height,
        components: u8::try_from(frame.comps.len()).unwrap_or(0),
        progressive: frame.progressive,
        orientation: meta.orientation,
    }
}

fn is_arithmetic(marker: u8) -> bool {
    matches!(marker, SOF9 | SOF10 | SOF11 | DAC | SOF13 | SOF14 | SOF15)
}

fn read_segment(input: &mut Cursor<'_>) -> Result<Vec<u8>, CodecError> {
    Ok(input.segment()?.to_vec())
}

fn parse_sof(marker: u8, body: &[u8], at: u64) -> Result<Frame, CodecError> {
    if body.len() < 6 {
        return Err(truncated(at, "frame"));
    }
    let precision = body[0];
    if precision != 8 {
        return Err(CodecError::Unsupported {
            what: "jpeg precision",
        });
    }
    let height = u32::from(u16::from(body[1]) << 8 | u16::from(body[2]));
    let width = u32::from(u16::from(body[3]) << 8 | u16::from(body[4]));
    if height == 0 || width == 0 {
        return Err(CodecError::Unsupported { what: "dnl" });
    }
    if !dimensions_ok(width, height) {
        return Err(CodecError::Limit { what: "jpeg size" });
    }
    let nf = usize::from(body[5]);
    if nf != 1 && nf != 3 {
        let what = if nf == 4 { "cmyk" } else { "component count" };
        return Err(CodecError::Unsupported { what });
    }
    if body.len() != 6 + 3 * nf {
        return Err(malformed(at, "frame"));
    }
    let mut comps = Vec::with_capacity(nf);
    let mut max_h = 1u8;
    let mut max_v = 1u8;
    for index in 0..nf {
        let base = 6 + 3 * index;
        let id = body[base];
        let hv = body[base + 1];
        let h = hv >> 4;
        let v = hv & 15;
        let quant = usize::from(body[base + 2]);
        if !(1..=4).contains(&h) || !(1..=4).contains(&v) || quant > 3 {
            return Err(malformed(at, "sampling"));
        }
        if comps.iter().any(|comp: &Comp| comp.id == id) {
            return Err(malformed(at, "component"));
        }
        max_h = max_h.max(h);
        max_v = max_v.max(v);
        comps.push(Comp {
            id,
            h,
            v,
            quant,
            blocks_x: 0,
            blocks_y: 0,
            coeffs: Vec::new(),
            plane: [-1; 64],
        });
    }
    let (mcu_cols, mcu_rows) = mcu_grid(width, height, max_h, max_v)?;
    let mut blocks_in_mcu = 0usize;
    for comp in &mut comps {
        comp.blocks_x = mcu_cols
            .checked_mul(usize::from(comp.h))
            .ok_or(CodecError::Limit {
                what: "coefficient",
            })?;
        comp.blocks_y = mcu_rows
            .checked_mul(usize::from(comp.v))
            .ok_or(CodecError::Limit {
                what: "coefficient",
            })?;
        let len = comp
            .blocks_x
            .checked_mul(comp.blocks_y)
            .and_then(|blocks| blocks.checked_mul(64))
            .ok_or(CodecError::Limit {
                what: "coefficient",
            })?;
        comp.coeffs = zeros(len)?;
        blocks_in_mcu += usize::from(comp.h) * usize::from(comp.v);
    }
    let baseline = marker == SOF0;
    if baseline && blocks_in_mcu > 10 {
        return Err(malformed(at, "mcu"));
    }
    if blocks_in_mcu > 64 {
        return Err(CodecError::Limit { what: "mcu" });
    }
    Ok(Frame {
        progressive: marker == SOF2,
        baseline,
        width,
        height,
        max_h,
        max_v,
        comps,
    })
}

fn mcu_grid(width: u32, height: u32, max_h: u8, max_v: u8) -> Result<(usize, usize), CodecError> {
    let mcu_w = u32::from(max_h) * 8;
    let mcu_h = u32::from(max_v) * 8;
    let cols = usize::try_from(width.div_ceil(mcu_w))
        .map_err(|_| CodecError::Limit { what: "jpeg size" })?;
    let rows = usize::try_from(height.div_ceil(mcu_h))
        .map_err(|_| CodecError::Limit { what: "jpeg size" })?;
    if cols == 0 || rows == 0 {
        return Err(malformed(0, "frame"));
    }
    Ok((cols, rows))
}

fn zeros(len: usize) -> Result<Vec<i32>, CodecError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| CodecError::Limit {
            what: "coefficient",
        })?;
    values.resize(len, 0);
    Ok(values)
}

fn parse_dqt(body: &[u8], quant: &mut [Option<[u16; 64]>; 4], at: u64) -> Result<(), CodecError> {
    let mut index = 0usize;
    if body.is_empty() {
        return Err(malformed(at, "quantizer"));
    }
    while index < body.len() {
        let info = *body.get(index).ok_or_else(|| truncated(at, "quantizer"))?;
        index += 1;
        let precision = info >> 4;
        let id = usize::from(info & 15);
        if precision > 1 || id > 3 {
            return Err(malformed(at, "quantizer"));
        }
        let mut table = [0u16; 64];
        for k in 0..64 {
            let value = if precision == 0 {
                let byte = *body.get(index).ok_or_else(|| truncated(at, "quantizer"))?;
                index += 1;
                u16::from(byte)
            } else {
                let hi = *body.get(index).ok_or_else(|| truncated(at, "quantizer"))?;
                let lo = *body
                    .get(index + 1)
                    .ok_or_else(|| truncated(at, "quantizer"))?;
                index += 2;
                (u16::from(hi) << 8) | u16::from(lo)
            };
            if value == 0 {
                return Err(malformed(at, "quantizer"));
            }
            let Some(natural) = ZIGZAG.get(k).copied() else {
                return Err(malformed(at, "quantizer"));
            };
            table[natural] = value;
        }
        quant[id] = Some(table);
    }
    Ok(())
}

fn parse_dht(
    body: &[u8],
    dc: &mut [Option<HuffDecode>; 4],
    ac: &mut [Option<HuffDecode>; 4],
    at: u64,
) -> Result<(), CodecError> {
    let mut index = 0usize;
    if body.is_empty() {
        return Err(malformed(at, "huffman table"));
    }
    while index < body.len() {
        let info = *body
            .get(index)
            .ok_or_else(|| truncated(at, "huffman table"))?;
        index += 1;
        let class = info >> 4;
        let id = usize::from(info & 15);
        if class > 1 || id > 3 {
            return Err(malformed(at, "huffman table"));
        }
        let mut counts = [0u8; 16];
        let mut total = 0usize;
        for count in &mut counts {
            *count = *body
                .get(index)
                .ok_or_else(|| truncated(at, "huffman table"))?;
            index += 1;
            total += usize::from(*count);
        }
        if total > 256 {
            return Err(malformed(at, "huffman table"));
        }
        let end = index + total;
        if end > body.len() {
            return Err(truncated(at, "huffman table"));
        }
        let symbols = &body[index..end];
        index = end;
        let table = decode_table(&counts, symbols).map_err(|_| malformed(at, "huffman table"))?;
        if class == 0 {
            dc[id] = Some(table);
        } else {
            ac[id] = Some(table);
        }
    }
    Ok(())
}

fn parse_dri(body: &[u8], at: u64) -> Result<u16, CodecError> {
    if body.len() != 2 {
        return Err(malformed(at, "restart"));
    }
    Ok((u16::from(body[0]) << 8) | u16::from(body[1]))
}

fn note_app(marker: u8, body: &[u8], meta: &mut Meta) {
    match marker {
        APP0 if body.starts_with(b"JFIF\0") => {}
        APP1 => {
            if let Some(orientation) = exif_orientation(body) {
                meta.orientation = Some(orientation);
            }
        }
        APP14 if body.len() >= 12 && body.starts_with(b"Adobe") => {
            meta.adobe = Some(body[11]);
        }
        _ => {}
    }
}

fn exif_orientation(body: &[u8]) -> Option<u16> {
    let tiff = body.strip_prefix(b"Exif\0\0")?;
    if tiff.len() < 8 {
        return None;
    }
    let le = match tiff.get(..2) {
        Some(b"II") => true,
        Some(b"MM") => false,
        _ => return None,
    };
    if read_u16(tiff, 2, le) != Some(42) {
        return None;
    }
    let offset = usize::try_from(read_u32(tiff, 4, le)?).ok()?;
    let count = usize::from(read_u16(tiff, offset, le)?);
    let mut cursor = offset.checked_add(2)?;
    let count = count.min(64);
    for _ in 0..count {
        let entry = tiff.get(cursor..cursor + 12)?;
        cursor += 12;
        let tag = read_u16(entry, 0, le)?;
        if tag != 0x0112 {
            continue;
        }
        let kind = read_u16(entry, 2, le)?;
        let n = read_u32(entry, 4, le)?;
        if n != 1 || (kind != 3 && kind != 4) {
            return None;
        }
        let value = if kind == 3 {
            read_u16(entry, 8, le)?
        } else {
            u16::try_from(read_u32(entry, 8, le)?).ok()?
        };
        if (1..=8).contains(&value) {
            return Some(value);
        }
        return None;
    }
    None
}

fn read_u16(bytes: &[u8], offset: usize, le: bool) -> Option<u16> {
    let pair = bytes.get(offset..offset + 2)?;
    if le {
        Some(u16::from(pair[0]) | (u16::from(pair[1]) << 8))
    } else {
        Some((u16::from(pair[0]) << 8) | u16::from(pair[1]))
    }
}

fn read_u32(bytes: &[u8], offset: usize, le: bool) -> Option<u32> {
    let word = bytes.get(offset..offset + 4)?;
    if le {
        Some(
            u32::from(word[0])
                | (u32::from(word[1]) << 8)
                | (u32::from(word[2]) << 16)
                | (u32::from(word[3]) << 24),
        )
    } else {
        Some(
            (u32::from(word[0]) << 24)
                | (u32::from(word[1]) << 16)
                | (u32::from(word[2]) << 8)
                | u32::from(word[3]),
        )
    }
}

fn decode_scan(
    input: &mut Cursor<'_>,
    frame: &mut Frame,
    body: &[u8],
    quant: &[Option<[u16; 64]>; 4],
    dc_huff: &[Option<HuffDecode>; 4],
    ac_huff: &[Option<HuffDecode>; 4],
    restart: u16,
) -> Result<(), CodecError> {
    let scan = parse_sos(frame, body, input.at())?;
    for comp in &scan.comps {
        let component = &frame.comps[comp.index];
        if quant[component.quant].is_none() {
            return Err(input.malformed("quantizer"));
        }
        if frame.baseline && (comp.dc_table > 1 || comp.ac_table > 1) {
            return Err(input.malformed("baseline huffman"));
        }
    }
    accept_progression(frame, &scan, input.at())?;
    let (cols, rows, interleaved) = scan_grid(frame, &scan)?;
    let mut pred = vec![0i32; frame.comps.len()];
    let mut eobrun = 0u32;
    let mut since = 0u32;
    let mut rst = 0u8;
    let mcus = cols
        .checked_mul(rows)
        .ok_or(CodecError::Limit { what: "mcu" })?;
    for mcu in 0..mcus {
        if restart > 0 && since == u32::from(restart) {
            let marker = input.take_marker()?;
            if marker != RST0 + rst {
                return Err(input.malformed("restart"));
            }
            rst = (rst + 1) & 7;
            since = 0;
            pred.fill(0);
            eobrun = 0;
        }
        if interleaved {
            let mcu_x = mcu % cols;
            let mcu_y = mcu / cols;
            let mut walk = Walk {
                input: &mut *input,
                frame: &mut *frame,
                scan: &scan,
                dc: dc_huff,
                ac: ac_huff,
                pred: &mut pred,
                eobrun,
            };
            decode_interleaved_mcu(&mut walk, mcu_x, mcu_y)?;
            eobrun = walk.eobrun;
        } else {
            let comp = scan.comps[0].index;
            let blocks_x = frame.comps[comp].blocks_x;
            let bx = mcu % blocks_x;
            let by = mcu / blocks_x;
            let spec = scan.comps[0];
            let mut walk = Walk {
                input: &mut *input,
                frame: &mut *frame,
                scan: &scan,
                dc: dc_huff,
                ac: ac_huff,
                pred: &mut pred,
                eobrun,
            };
            decode_one_unit(&mut walk, spec, bx, by)?;
            eobrun = walk.eobrun;
        }
        since += 1;
    }
    if eobrun > 0 {
        return Err(input.malformed("eob"));
    }
    Ok(())
}

fn parse_sos(frame: &Frame, body: &[u8], at: u64) -> Result<Scan, CodecError> {
    if body.is_empty() {
        return Err(malformed(at, "scan"));
    }
    let ns = usize::from(body[0]);
    if ns == 0 || ns > 4 || body.len() != 4 + 2 * ns {
        return Err(malformed(at, "scan"));
    }
    let mut comps = Vec::with_capacity(ns);
    for index in 0..ns {
        let id = body[1 + 2 * index];
        let tables = body[2 + 2 * index];
        let Some(comp_index) = frame.comps.iter().position(|comp| comp.id == id) else {
            return Err(malformed(at, "component"));
        };
        if comps.iter().any(|comp: &ScanComp| comp.index == comp_index) {
            return Err(malformed(at, "component"));
        }
        comps.push(ScanComp {
            index: comp_index,
            dc_table: usize::from(tables >> 4),
            ac_table: usize::from(tables & 15),
        });
    }
    let ss = usize::from(body[body.len() - 3]);
    let se = usize::from(body[body.len() - 2]);
    let successive = body[body.len() - 1];
    let ah = successive >> 4;
    let al = successive & 15;
    if se > 63 || ss > se || al > MAX_AL {
        return Err(malformed(at, "progressive scan"));
    }
    if frame.progressive {
        if ss == 0 && se != 0 {
            return Err(malformed(at, "progressive scan"));
        }
        if ss > 0 && comps.len() != 1 {
            return Err(malformed(at, "progressive scan"));
        }
        if ah != 0 && al + 1 != ah {
            return Err(malformed(at, "progressive scan"));
        }
    } else if ss != 0 || se != 63 || ah != 0 || al != 0 {
        return Err(malformed(at, "scan"));
    }
    if comps
        .iter()
        .any(|comp| comp.dc_table > 3 || comp.ac_table > 3)
    {
        return Err(malformed(at, "huffman table"));
    }
    Ok(Scan {
        comps,
        ss,
        se,
        ah,
        al,
    })
}

fn malformed(at: u64, what: &'static str) -> CodecError {
    CodecError::Malformed { at_bit: at, what }
}

fn truncated(at: u64, what: &'static str) -> CodecError {
    CodecError::Truncated { at_bit: at, what }
}

fn accept_progression(frame: &mut Frame, scan: &Scan, at: u64) -> Result<(), CodecError> {
    for spec in &scan.comps {
        let comp = &mut frame.comps[spec.index];
        if scan.ah == 0 {
            for k in scan.ss..=scan.se {
                if comp.plane[k] >= 0 {
                    return Err(malformed(at, "progressive scan"));
                }
            }
        } else if (scan.ss..=scan.se).any(|k| comp.plane[k] != i8::try_from(scan.ah).unwrap_or(0)) {
            return Err(malformed(at, "progressive scan"));
        }
        if scan.ss > 0 && comp.plane[0] < 0 {
            return Err(malformed(at, "progressive scan"));
        }
        for k in scan.ss..=scan.se {
            comp.plane[k] = i8::try_from(scan.al).unwrap_or(0);
        }
    }
    Ok(())
}

fn scan_grid(frame: &Frame, scan: &Scan) -> Result<(usize, usize, bool), CodecError> {
    if scan.comps.len() > 1 {
        let (cols, rows) = mcu_grid(frame.width, frame.height, frame.max_h, frame.max_v)?;
        return Ok((cols, rows, true));
    }
    let comp = &frame.comps[scan.comps[0].index];
    if comp.blocks_x == 0 || comp.blocks_y == 0 {
        return Err(malformed(0, "frame"));
    }
    Ok((comp.blocks_x, comp.blocks_y, false))
}

struct Walk<'a, 'b> {
    input: &'a mut Cursor<'b>,
    frame: &'a mut Frame,
    scan: &'a Scan,
    dc: &'a [Option<HuffDecode>; 4],
    ac: &'a [Option<HuffDecode>; 4],
    pred: &'a mut [i32],
    eobrun: u32,
}

fn decode_interleaved_mcu(
    walk: &mut Walk<'_, '_>,
    mcu_x: usize,
    mcu_y: usize,
) -> Result<(), CodecError> {
    let count = walk.scan.comps.len();
    for at in 0..count {
        let spec = walk.scan.comps[at];
        let h = usize::from(walk.frame.comps[spec.index].h);
        let v = usize::from(walk.frame.comps[spec.index].v);
        for local_y in 0..v {
            for local_x in 0..h {
                decode_one_unit(walk, spec, mcu_x * h + local_x, mcu_y * v + local_y)?;
            }
        }
    }
    Ok(())
}

fn decode_one_unit(
    walk: &mut Walk<'_, '_>,
    spec: ScanComp,
    bx: usize,
    by: usize,
) -> Result<(), CodecError> {
    let mut block = load_block(&walk.frame.comps[spec.index], bx, by, walk.input)?;
    let predictor = walk
        .pred
        .get_mut(spec.index)
        .ok_or_else(|| walk.input.malformed("component"))?;
    let eobrun = if walk.scan.ss == 0 && walk.scan.se == 63 && !walk.frame.progressive {
        decode_baseline_block(walk.input, &mut block, &spec, walk.dc, walk.ac, predictor)?;
        0
    } else if walk.scan.ss == 0 {
        decode_dc_band(walk.input, &mut block, walk.scan, &spec, walk.dc, predictor)?;
        walk.eobrun
    } else {
        decode_ac_band(
            walk.input,
            &mut block,
            walk.scan,
            &spec,
            walk.ac,
            walk.eobrun,
        )?
    };
    store_block(
        &mut walk.frame.comps[spec.index],
        bx,
        by,
        &block,
        walk.input,
    )?;
    walk.eobrun = eobrun;
    Ok(())
}

fn decode_baseline_block(
    input: &mut Cursor<'_>,
    block: &mut [i32; 64],
    spec: &ScanComp,
    dc_huff: &[Option<HuffDecode>; 4],
    ac_huff: &[Option<HuffDecode>; 4],
    pred: &mut i32,
) -> Result<(), CodecError> {
    let dc = huff(dc_huff, spec.dc_table, input)?;
    let ac = huff(ac_huff, spec.ac_table, input)?;
    block[0] = read_dc(input, dc, pred, 0)?;
    let left = read_ac_first(input, block, ac, 1, 63, 0, 0)?;
    if left != 0 {
        return Err(input.malformed("eob"));
    }
    Ok(())
}

fn decode_dc_band(
    input: &mut Cursor<'_>,
    block: &mut [i32; 64],
    scan: &Scan,
    spec: &ScanComp,
    dc_huff: &[Option<HuffDecode>; 4],
    pred: &mut i32,
) -> Result<(), CodecError> {
    if scan.ah == 0 {
        let dc = huff(dc_huff, spec.dc_table, input)?;
        block[0] = read_dc(input, dc, pred, scan.al)?;
    } else {
        refine_dc(input, &mut block[0], scan.al)?;
    }
    Ok(())
}

fn decode_ac_band(
    input: &mut Cursor<'_>,
    block: &mut [i32; 64],
    scan: &Scan,
    spec: &ScanComp,
    ac_huff: &[Option<HuffDecode>; 4],
    eobrun: u32,
) -> Result<u32, CodecError> {
    let ac = huff(ac_huff, spec.ac_table, input)?;
    if scan.ah == 0 {
        read_ac_first(input, block, ac, scan.ss, scan.se, scan.al, eobrun)
    } else {
        read_ac_refine(input, block, ac, scan.ss, scan.se, scan.al, eobrun)
    }
}

fn huff<'a>(
    tables: &'a [Option<HuffDecode>; 4],
    id: usize,
    input: &Cursor<'_>,
) -> Result<&'a HuffDecode, CodecError> {
    tables
        .get(id)
        .and_then(Option::as_ref)
        .ok_or_else(|| input.malformed("huffman table"))
}

fn read_dc(
    input: &mut Cursor<'_>,
    table: &HuffDecode,
    pred: &mut i32,
    al: u8,
) -> Result<i32, CodecError> {
    let size = u32::from(decode_symbol(input, table)?);
    let diff = if size == 0 {
        0
    } else {
        extend(input.read_bits(size)?, size)?
    };
    let value = pred
        .checked_add(diff)
        .ok_or_else(|| input.malformed("coefficient"))?;
    *pred = value;
    shift_coeff(value, al, input)
}

fn refine_dc(input: &mut Cursor<'_>, coeff: &mut i32, al: u8) -> Result<(), CodecError> {
    if input.read_bit()? == 1 {
        *coeff |= 1 << al;
    }
    Ok(())
}

fn read_ac_first(
    input: &mut Cursor<'_>,
    block: &mut [i32; 64],
    table: &HuffDecode,
    ss: usize,
    se: usize,
    al: u8,
    eobrun: u32,
) -> Result<u32, CodecError> {
    if eobrun > 0 {
        return Ok(eobrun - 1);
    }
    let mut k = ss;
    while k <= se {
        let symbol = u32::from(decode_symbol(input, table)?);
        let run = symbol >> 4;
        let size = symbol & 15;
        if size == 0 {
            if run == 15 {
                if se - k + 1 < 16 {
                    return Err(input.malformed("coefficient"));
                }
                k += 16;
            } else {
                if run > 14 {
                    return Err(input.malformed("eob"));
                }
                let extra = if run == 0 { 0 } else { input.read_bits(run)? };
                let eob = (1u32 << run) + extra;
                return Ok(eob - 1);
            }
        } else {
            if k + usize::try_from(run).unwrap_or(usize::MAX) > se {
                return Err(input.malformed("coefficient"));
            }
            k += usize::from(u8::try_from(run).unwrap_or(0));
            let magnitude = extend(input.read_bits(size)?, size)?;
            let value = shift_coeff(magnitude, al, input)?;
            let natural = *ZIGZAG
                .get(k)
                .ok_or_else(|| input.malformed("coefficient"))?;
            block[natural] = value;
            k += 1;
        }
    }
    Ok(0)
}

fn read_ac_refine(
    input: &mut Cursor<'_>,
    block: &mut [i32; 64],
    table: &HuffDecode,
    ss: usize,
    se: usize,
    al: u8,
    eobrun: u32,
) -> Result<u32, CodecError> {
    let mut k = ss;
    if eobrun == 0 {
        while k <= se {
            let symbol = u32::from(decode_symbol(input, table)?);
            let mut run = i32::try_from(symbol >> 4).unwrap_or(0);
            let size = symbol & 15;
            let new_coef = if size == 0 {
                if run != 15 {
                    if !(0..=14).contains(&run) {
                        return Err(input.malformed("eob"));
                    }
                    let extra = if run == 0 {
                        0
                    } else {
                        input.read_bits(u32::try_from(run).unwrap_or(0))?
                    };
                    let eob = (1u32 << run) + extra;
                    refine_band(input, block, k, se, al)?;
                    return Ok(eob - 1);
                }
                0
            } else {
                if size != 1 {
                    return Err(input.malformed("refinement"));
                }
                let bit = input.read_bit()?;
                let step = 1i32 << al;
                if bit == 1 { step } else { -step }
            };
            loop {
                if k > se {
                    return Err(input.malformed("coefficient"));
                }
                let natural = *ZIGZAG
                    .get(k)
                    .ok_or_else(|| input.malformed("coefficient"))?;
                if block[natural] != 0 {
                    refine_one(input, &mut block[natural], al)?;
                } else {
                    run -= 1;
                    if run < 0 {
                        break;
                    }
                }
                k += 1;
            }
            if new_coef != 0 {
                if k > se {
                    return Err(input.malformed("coefficient"));
                }
                let natural = *ZIGZAG
                    .get(k)
                    .ok_or_else(|| input.malformed("coefficient"))?;
                block[natural] = new_coef;
            }
            k += 1;
        }
    }
    if eobrun > 0 {
        refine_band(input, block, k, se, al)?;
        return Ok(eobrun - 1);
    }
    Ok(0)
}

fn refine_band(
    input: &mut Cursor<'_>,
    block: &mut [i32; 64],
    from: usize,
    to: usize,
    al: u8,
) -> Result<(), CodecError> {
    if from > to {
        return Ok(());
    }
    for k in from..=to {
        let natural = *ZIGZAG
            .get(k)
            .ok_or_else(|| input.malformed("coefficient"))?;
        if block[natural] != 0 {
            refine_one(input, &mut block[natural], al)?;
        }
    }
    Ok(())
}

fn refine_one(input: &mut Cursor<'_>, coeff: &mut i32, al: u8) -> Result<(), CodecError> {
    if input.read_bit()? == 0 {
        return Ok(());
    }
    let step = 1i32 << al;
    if *coeff & step != 0 {
        return Ok(());
    }
    if *coeff >= 0 {
        *coeff += step;
    } else {
        *coeff -= step;
    }
    Ok(())
}

fn extend(bits: u32, size: u32) -> Result<i32, CodecError> {
    if !(1..=15).contains(&size) {
        return Err(malformed(0, "magnitude"));
    }
    let value = i64::from(bits);
    let magnitude = if value < (1i64 << (size - 1)) {
        value + ((-1i64) << size) + 1
    } else {
        value
    };
    i32::try_from(magnitude).map_err(|_| malformed(0, "coefficient"))
}

fn shift_coeff(value: i32, al: u8, input: &Cursor<'_>) -> Result<i32, CodecError> {
    let shifted = i64::from(value) << al;
    i32::try_from(shifted).map_err(|_| input.malformed("coefficient"))
}

fn block_start(comp: &Comp, bx: usize, by: usize, input: &Cursor<'_>) -> Result<usize, CodecError> {
    if bx >= comp.blocks_x || by >= comp.blocks_y {
        return Err(input.malformed("coefficient"));
    }
    by.checked_mul(comp.blocks_x)
        .and_then(|index| index.checked_add(bx))
        .and_then(|index| index.checked_mul(64))
        .ok_or_else(|| input.malformed("coefficient"))
}

fn load_block(
    comp: &Comp,
    bx: usize,
    by: usize,
    input: &Cursor<'_>,
) -> Result<[i32; 64], CodecError> {
    let start = block_start(comp, bx, by, input)?;
    let mut block = [0i32; 64];
    let Some(slice) = comp.coeffs.get(start..start + 64) else {
        return Err(input.malformed("coefficient"));
    };
    block.copy_from_slice(slice);
    Ok(block)
}

fn store_block(
    comp: &mut Comp,
    bx: usize,
    by: usize,
    block: &[i32; 64],
    input: &Cursor<'_>,
) -> Result<(), CodecError> {
    let start = block_start(comp, bx, by, input)?;
    let Some(slot) = comp.coeffs.get_mut(start..start + 64) else {
        return Err(input.malformed("coefficient"));
    };
    slot.copy_from_slice(block);
    Ok(())
}

fn render(
    frame: &Frame,
    quant: &[Option<[u16; 64]>; 4],
    meta: &Meta,
) -> Result<StillImage, CodecError> {
    let mut planes = Vec::with_capacity(frame.comps.len());
    for comp in &frame.comps {
        let table = quant[comp.quant].ok_or(malformed(0, "quantizer"))?;
        planes.push(paint(comp, &table)?);
    }
    let space = color_space(frame, meta);
    let (color, channels) = match space {
        Space::Gray => (ColorType::Gray, 1usize),
        Space::Rgb | Space::YCbCr => (ColorType::Rgb, 3usize),
    };
    let count = u64::from(frame.width)
        .checked_mul(u64::from(frame.height))
        .and_then(|pixels| pixels.checked_mul(channels as u64))
        .ok_or(CodecError::Limit {
            what: "jpeg picture",
        })?;
    let count = usize::try_from(count).map_err(|_| CodecError::Limit {
        what: "jpeg picture",
    })?;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(count)
        .map_err(|_| CodecError::Limit {
            what: "jpeg picture",
        })?;
    for y in 0..frame.height {
        for x in 0..frame.width {
            let pixel = composed(&planes, frame, x, y, space)?;
            let channels = if matches!(space, Space::Gray) { 1 } else { 3 };
            samples.extend(pixel.into_iter().take(channels));
        }
    }
    let image =
        StillImage::from_samples(frame.width, frame.height, color, BitDepth::Eight, samples);
    image.checked().map_err(|_| malformed(0, "picture"))
}

struct Plane {
    stride: usize,
    samples: Vec<u8>,
}

fn paint(comp: &Comp, quant: &[u16; 64]) -> Result<Plane, CodecError> {
    let stride = comp.blocks_x.checked_mul(8).ok_or(CodecError::Limit {
        what: "jpeg picture",
    })?;
    let rows = comp.blocks_y.checked_mul(8).ok_or(CodecError::Limit {
        what: "jpeg picture",
    })?;
    let len = stride.checked_mul(rows).ok_or(CodecError::Limit {
        what: "jpeg picture",
    })?;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(len)
        .map_err(|_| CodecError::Limit {
            what: "jpeg picture",
        })?;
    samples.resize(len, 0);
    for by in 0..comp.blocks_y {
        for bx in 0..comp.blocks_x {
            let start = (by * comp.blocks_x + bx) * 64;
            let Some(coeffs) = comp.coeffs.get(start..start + 64) else {
                return Err(malformed(0, "coefficient"));
            };
            let block = dequant_block(coeffs, quant).map_err(|_| malformed(0, "coefficient"))?;
            let pixels = idct_u8(&block);
            for row in 0..8 {
                for col in 0..8 {
                    let dest = (by * 8 + row) * stride + bx * 8 + col;
                    samples[dest] = pixels[row * 8 + col];
                }
            }
        }
    }
    Ok(Plane { stride, samples })
}

fn color_space(frame: &Frame, meta: &Meta) -> Space {
    if frame.comps.len() == 1 {
        return Space::Gray;
    }
    let ids = [
        frame.comps.first().map(|comp| comp.id).unwrap_or(0),
        frame.comps.get(1).map(|comp| comp.id).unwrap_or(0),
        frame.comps.get(2).map(|comp| comp.id).unwrap_or(0),
    ];
    if meta.adobe == Some(0) || ids == *b"RGB" {
        Space::Rgb
    } else {
        Space::YCbCr
    }
}

fn composed(
    planes: &[Plane],
    frame: &Frame,
    x: u32,
    y: u32,
    space: Space,
) -> Result<[u16; 3], CodecError> {
    let x = usize::try_from(x).unwrap_or(0);
    let y = usize::try_from(y).unwrap_or(0);
    match space {
        Space::Gray => Ok([
            u16::from(located(&planes[0], &frame.comps[0], frame, x, y)?),
            0,
            0,
        ]),
        Space::Rgb => Ok([
            u16::from(located(&planes[0], &frame.comps[0], frame, x, y)?),
            u16::from(located(&planes[1], &frame.comps[1], frame, x, y)?),
            u16::from(located(&planes[2], &frame.comps[2], frame, x, y)?),
        ]),
        Space::YCbCr => {
            let y_s = i32::from(located(&planes[0], &frame.comps[0], frame, x, y)?);
            let cb = i32::from(located(&planes[1], &frame.comps[1], frame, x, y)?);
            let cr = i32::from(located(&planes[2], &frame.comps[2], frame, x, y)?);
            let (red, green, blue) = ycbcr_to_rgb(y_s, cb, cr);
            Ok([u16::from(red), u16::from(green), u16::from(blue)])
        }
    }
}

fn located(
    plane: &Plane,
    comp: &Comp,
    frame: &Frame,
    x: usize,
    y: usize,
) -> Result<u8, CodecError> {
    let sx = x * usize::from(comp.h) / usize::from(frame.max_h);
    let sy = y * usize::from(comp.v) / usize::from(frame.max_v);
    plane
        .samples
        .get(sy * plane.stride + sx)
        .copied()
        .ok_or(malformed(0, "sample"))
}

/// Software JPEG decoder. One push is one complete image.
pub struct JpegDecoder;

impl JpegDecoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Jpeg {
            return Err(CodecError::Unsupported { what: "jpeg" });
        }
        Ok(Self)
    }
}

impl Decoder for JpegDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Jpeg
    }

    fn push(&mut self, _packet: &Packet) -> Result<Vec<Output>, CodecError> {
        Err(CodecError::InvalidApi {
            what: "jpeg produces a picture",
        })
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {}

    fn decode_picture(&mut self, packet: &Packet) -> Result<Vec<StillOutput>, CodecError> {
        if packet.data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        let image = decode_jpeg(&packet.data)?;
        Ok(vec![StillOutput {
            image,
            pts: packet.pts,
            duration: packet.duration,
            timescale: packet.timescale,
        }])
    }
}

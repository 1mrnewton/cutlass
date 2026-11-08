//! One JPEG image to component planes.
//!
//! Sequential scans reconstruct each block as it is read. Progressive scans
//! keep the coefficients until the end of the image. Arithmetic SOF markers
//! and `DAC` are refused before any coefficient is written.

use std::sync::Arc;

use cutlass_decode::CodecError;

use crate::bits::Cursor;
use crate::dct::{idct_block, idct_dc};
use crate::huffman::{HuffDecode, decode_symbol, decode_table};
use crate::limits::{MAX_AL, MAX_INPUT, MAX_MARKERS, dimensions_ok};
use crate::tables::{
    AC_CHROMA_COUNTS, AC_CHROMA_SYMBOLS, AC_LUMA_COUNTS, AC_LUMA_SYMBOLS, DC_CHROMA_COUNTS,
    DC_CHROMA_SYMBOLS, DC_LUMA_COUNTS, DC_LUMA_SYMBOLS, ZIGZAG,
};

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

/// Quantization and Huffman tables in effect.
///
/// A DQT or DHT segment replaces one slot. Tables persist from one image to
/// the next only when the caller passes the same value again, which is how
/// an abbreviated image (ITU-T T.81 B.5) finds tables an earlier one defined.
#[derive(Clone, Default)]
pub struct Tables {
    quant: [Option<[u16; 64]>; 4],
    dc: [Option<Arc<HuffDecode>>; 4],
    ac: [Option<Arc<HuffDecode>>; 4],
}

impl Tables {
    /// No tables. An image must define every table its scans use.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The Annex K.3 Huffman tables in ids 0 (luminance) and 1
    /// (chrominance), and no quantization tables.
    ///
    /// Motion JPEG frames leave these tables out; a DHT segment replaces them.
    #[must_use]
    pub fn standard_huffman() -> Self {
        let table =
            |counts: &[u8], symbols: &[u8]| decode_table(counts, symbols).ok().map(Arc::new);
        let mut tables = Self::default();
        tables.dc[0] = table(&DC_LUMA_COUNTS, &DC_LUMA_SYMBOLS);
        tables.ac[0] = table(&AC_LUMA_COUNTS, &AC_LUMA_SYMBOLS);
        tables.dc[1] = table(&DC_CHROMA_COUNTS, &DC_CHROMA_SYMBOLS);
        tables.ac[1] = table(&AC_CHROMA_COUNTS, &AC_CHROMA_SYMBOLS);
        tables
    }

    /// Whether quantization table `id` is defined.
    #[must_use]
    pub fn has_quant(&self, id: usize) -> bool {
        self.quant.get(id).is_some_and(Option::is_some)
    }

    /// Whether Huffman table `id` of `class` (0 DC, 1 AC) is defined.
    #[must_use]
    pub fn has_huffman(&self, class: u8, id: usize) -> bool {
        let slots = if class == 0 { &self.dc } else { &self.ac };
        slots.get(id).is_some_and(Option::is_some)
    }
}

impl std::fmt::Debug for Tables {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let present = |slots: [bool; 4]| slots.map(u8::from);
        f.debug_struct("Tables")
            .field(
                "quant",
                &present(self.quant.each_ref().map(Option::is_some)),
            )
            .field("dc", &present(self.dc.each_ref().map(Option::is_some)))
            .field("ac", &present(self.ac.each_ref().map(Option::is_some)))
            .finish()
    }
}

/// How the components of an image relate to color.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorModel {
    /// One component.
    Gray,
    /// Three components, JFIF YCbCr (ITU-T T.871).
    YCbCr,
    /// Three components, R, G, B: Adobe transform 0 or component ids `R`, `G`, `B`.
    Rgb,
}

/// One decoded component, tightly packed.
///
/// `width` and `height` are the component's own size from ITU-T T.81 A.1.1:
/// `ceil(X * h / Hmax)` by `ceil(Y * v / Vmax)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Component {
    pub id: u8,
    pub h: u8,
    pub v: u8,
    pub width: u32,
    pub height: u32,
    pub samples: Vec<u8>,
}

/// One decoded image, as component planes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub color: ColorModel,
    /// In frame-header order.
    pub components: Vec<Component>,
    pub max_h: u8,
    pub max_v: u8,
    pub progressive: bool,
    /// An APP0 `JFIF` segment was present.
    pub jfif: bool,
    /// A COM segment whose text, up to a NUL or a final newline, is
    /// `CS=ITU601`, declaring studio-range YCbCr.
    pub studio_range: bool,
    /// Byte 12 of a COM segment that starts `AVID`: 1 is NTSC, 2 is PAL.
    pub avid_standard: Option<u8>,
    /// EXIF orientation, when APP1 carries one. Not applied.
    pub orientation: Option<u16>,
}

/// An image and the number of bytes it used, through its EOI marker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decoded {
    pub image: Image,
    pub len: usize,
}

/// Frame header facts, read without decoding.
pub(crate) struct Header {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) components: u8,
    pub(crate) progressive: bool,
    pub(crate) orientation: Option<u16>,
}

/// Decode the image at the start of `bytes`.
///
/// `tables` supplies tables this image does not define, and receives the
/// tables it does define, even when the image then fails. A caller that
/// must keep its tables on failure passes a clone. Bytes after EOI are not
/// read; [`Decoded::len`] says where they start.
pub fn decode_image(bytes: &[u8], tables: &mut Tables) -> Result<Decoded, CodecError> {
    match walk(bytes, tables, false)? {
        Walked::Image(decoded) => Ok(decoded),
        Walked::Header(_) => Err(CodecError::Malformed {
            at_bit: 0,
            what: "scan",
        }),
    }
}

/// Read an abbreviated table-specification stream (ITU-T T.81 B.5): SOI,
/// DQT, DHT, APP, and COM segments, then EOI. `tables` changes only when the
/// whole stream reads. Returns the bytes used through EOI.
pub fn read_tables(bytes: &[u8], tables: &mut Tables) -> Result<usize, CodecError> {
    if bytes.len() > MAX_INPUT {
        return Err(CodecError::Limit { what: "jpeg bytes" });
    }
    let mut input = Cursor::new(bytes);
    if input.take_marker()? != SOI {
        return Err(input.malformed("start of image"));
    }
    let mut read = tables.clone();
    for _ in 0..MAX_MARKERS {
        match input.take_marker()? {
            EOI => {
                *tables = read;
                return Ok(input.offset());
            }
            DQT => {
                let body = input.segment()?;
                parse_dqt(body, &mut read.quant, input.at())?;
            }
            DHT => {
                let body = input.segment()?;
                parse_dht(body, &mut read, input.at())?;
            }
            APP0..=0xef | COM => {
                input.segment()?;
            }
            _ => return Err(input.malformed("table specification")),
        }
    }
    Err(CodecError::Limit {
        what: "jpeg markers",
    })
}

/// The frame header, stopping at the first scan. A frame header followed
/// by a broken segment still reports its size.
pub(crate) fn read_header(bytes: &[u8]) -> Result<Header, CodecError> {
    let mut tables = Tables::new();
    match walk(bytes, &mut tables, true)? {
        Walked::Header(header) => Ok(header),
        Walked::Image(decoded) => Ok(Header {
            width: decoded.image.width,
            height: decoded.image.height,
            components: u8::try_from(decoded.image.components.len()).unwrap_or(0),
            progressive: decoded.image.progressive,
            orientation: decoded.image.orientation,
        }),
    }
}

enum Walked {
    Header(Header),
    Image(Decoded),
}

struct Comp {
    id: u8,
    h: u8,
    v: u8,
    quant_id: usize,
    width: u32,
    height: u32,
    /// Blocks covering the component itself: a non-interleaved scan's grid.
    blocks_w: usize,
    blocks_h: usize,
    /// Blocks covering whole MCUs: an interleaved scan's grid and the plane.
    padded_w: usize,
    padded_h: usize,
    /// The table in effect when the component's first scan began.
    quant: Option<[u16; 64]>,
    plane: Vec<u8>,
    coeffs: Vec<i32>,
    /// Progressive: the low bit of each coefficient so far, -1 before any scan.
    progress: [i8; 64],
    scanned: bool,
}

struct Frame {
    progressive: bool,
    baseline: bool,
    width: u32,
    height: u32,
    max_h: u8,
    max_v: u8,
    mcu_cols: usize,
    mcu_rows: usize,
    comps: Vec<Comp>,
    allocated: bool,
}

#[derive(Default)]
struct Meta {
    orientation: Option<u16>,
    adobe: Option<u8>,
    jfif: bool,
    studio_range: bool,
    avid_standard: Option<u8>,
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

fn walk(bytes: &[u8], tables: &mut Tables, header_only: bool) -> Result<Walked, CodecError> {
    if bytes.len() > MAX_INPUT {
        return Err(CodecError::Limit { what: "jpeg bytes" });
    }
    let mut input = Cursor::new(bytes);
    if input.take_marker()? != SOI {
        return Err(input.malformed("start of image"));
    }
    let mut frame: Option<Frame> = None;
    let mut restart: u16 = 0;
    let mut meta = Meta::default();
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
            Err(_) if header_only && frame.is_some() => {
                return Ok(Walked::Header(header_of(frame.as_ref(), &meta)));
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
                let body = input.segment()?;
                frame = Some(parse_sof(marker, body, input.at())?);
            }
            DQT => {
                let body = input.segment()?;
                parse_dqt(body, &mut tables.quant, input.at())?;
            }
            DHT => {
                let body = input.segment()?;
                parse_dht(body, tables, input.at())?;
            }
            DRI => {
                let body = input.segment()?;
                restart = parse_dri(body, input.at())?;
            }
            SOS => {
                let Some(frame) = frame.as_mut() else {
                    return Err(input.malformed("frame"));
                };
                let body = input.segment()?;
                if header_only {
                    return Ok(Walked::Header(header_of(Some(frame), &meta)));
                }
                allocate(frame)?;
                let scan = parse_sos(frame, body, input.at())?;
                if frame.progressive {
                    progressive_scan(&mut input, frame, &scan, tables, restart)?;
                } else {
                    sequential_scan(&mut input, frame, &scan, tables, restart)?;
                }
                scans += 1;
            }
            APP0 | APP1 | APP14 | COM => {
                let body = input.segment()?;
                note(marker, body, &mut meta);
            }
            0xe2..=0xed | 0xef => {
                input.segment()?;
            }
            DNL => return Err(CodecError::Unsupported { what: "dnl" }),
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
            0xf7 => return Err(CodecError::Unsupported { what: "jpeg-ls" }),
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
    if header_only {
        return Ok(Walked::Header(header_of(Some(&frame), &meta)));
    }
    if scans == 0 {
        return Err(input.malformed("scan"));
    }
    let len = input.offset();
    let image = finish(frame, &meta)?;
    Ok(Walked::Image(Decoded { image, len }))
}

fn header_of(frame: Option<&Frame>, meta: &Meta) -> Header {
    let Some(frame) = frame else {
        return Header {
            width: 0,
            height: 0,
            components: 0,
            progressive: false,
            orientation: None,
        };
    };
    Header {
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

fn malformed(at: u64, what: &'static str) -> CodecError {
    CodecError::Malformed { at_bit: at, what }
}

fn truncated(at: u64, what: &'static str) -> CodecError {
    CodecError::Truncated { at_bit: at, what }
}

fn parse_sof(marker: u8, body: &[u8], at: u64) -> Result<Frame, CodecError> {
    if body.len() < 6 {
        return Err(truncated(at, "frame"));
    }
    if body[0] != 8 {
        return Err(CodecError::Unsupported {
            what: "jpeg precision",
        });
    }
    let height = u32::from(u16::from_be_bytes([body[1], body[2]]));
    let width = u32::from(u16::from_be_bytes([body[3], body[4]]));
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
    let mut comps: Vec<Comp> = Vec::with_capacity(nf);
    let mut max_h = 1u8;
    let mut max_v = 1u8;
    for index in 0..nf {
        let base = 6 + 3 * index;
        let id = body[base];
        let h = body[base + 1] >> 4;
        let v = body[base + 1] & 15;
        let quant_id = usize::from(body[base + 2]);
        if !(1..=4).contains(&h) || !(1..=4).contains(&v) || quant_id > 3 {
            return Err(malformed(at, "sampling"));
        }
        if comps.iter().any(|comp| comp.id == id) {
            return Err(malformed(at, "component"));
        }
        max_h = max_h.max(h);
        max_v = max_v.max(v);
        comps.push(Comp {
            id,
            h,
            v,
            quant_id,
            width: 0,
            height: 0,
            blocks_w: 0,
            blocks_h: 0,
            padded_w: 0,
            padded_h: 0,
            quant: None,
            plane: Vec::new(),
            coeffs: Vec::new(),
            progress: [-1; 64],
            scanned: false,
        });
    }
    let mcu_cols = usize::try_from(width.div_ceil(u32::from(max_h) * 8))
        .map_err(|_| CodecError::Limit { what: "jpeg size" })?;
    let mcu_rows = usize::try_from(height.div_ceil(u32::from(max_v) * 8))
        .map_err(|_| CodecError::Limit { what: "jpeg size" })?;
    let mut blocks_in_mcu = 0usize;
    for comp in &mut comps {
        comp.width = (width * u32::from(comp.h)).div_ceil(u32::from(max_h));
        comp.height = (height * u32::from(comp.v)).div_ceil(u32::from(max_v));
        comp.blocks_w = usize::try_from(comp.width.div_ceil(8)).unwrap_or(0);
        comp.blocks_h = usize::try_from(comp.height.div_ceil(8)).unwrap_or(0);
        comp.padded_w = mcu_cols * usize::from(comp.h);
        comp.padded_h = mcu_rows * usize::from(comp.v);
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
        mcu_cols,
        mcu_rows,
        comps,
        allocated: false,
    })
}

fn zeroed<T: Clone + Default>(len: usize) -> Result<Vec<T>, CodecError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| CodecError::Limit {
            what: "jpeg picture",
        })?;
    values.resize(len, T::default());
    Ok(values)
}

fn allocate(frame: &mut Frame) -> Result<(), CodecError> {
    if frame.allocated {
        return Ok(());
    }
    for comp in &mut frame.comps {
        let blocks = comp
            .padded_w
            .checked_mul(comp.padded_h)
            .ok_or(CodecError::Limit {
                what: "jpeg picture",
            })?;
        let samples = blocks.checked_mul(64).ok_or(CodecError::Limit {
            what: "jpeg picture",
        })?;
        comp.plane = zeroed(samples)?;
        if frame.progressive {
            comp.coeffs = zeroed(samples)?;
        }
    }
    frame.allocated = true;
    Ok(())
}

fn parse_dqt(body: &[u8], quant: &mut [Option<[u16; 64]>; 4], at: u64) -> Result<(), CodecError> {
    if body.is_empty() {
        return Err(malformed(at, "quantizer"));
    }
    let mut index = 0usize;
    while index < body.len() {
        let info = body[index];
        index += 1;
        let precision = info >> 4;
        let id = usize::from(info & 15);
        if precision > 1 || id > 3 {
            return Err(malformed(at, "quantizer"));
        }
        let mut table = [0u16; 64];
        for natural in ZIGZAG {
            let value = if precision == 0 {
                let byte = *body.get(index).ok_or_else(|| truncated(at, "quantizer"))?;
                index += 1;
                u16::from(byte)
            } else {
                let pair = body
                    .get(index..index + 2)
                    .ok_or_else(|| truncated(at, "quantizer"))?;
                index += 2;
                u16::from_be_bytes([pair[0], pair[1]])
            };
            if value == 0 {
                return Err(malformed(at, "quantizer"));
            }
            table[natural] = value;
        }
        quant[id] = Some(table);
    }
    Ok(())
}

fn parse_dht(body: &[u8], tables: &mut Tables, at: u64) -> Result<(), CodecError> {
    if body.is_empty() {
        return Err(malformed(at, "huffman table"));
    }
    let mut index = 0usize;
    while index < body.len() {
        let info = body[index];
        index += 1;
        let class = info >> 4;
        let id = usize::from(info & 15);
        if class > 1 || id > 3 {
            return Err(malformed(at, "huffman table"));
        }
        let counts = body
            .get(index..index + 16)
            .ok_or_else(|| truncated(at, "huffman table"))?;
        index += 16;
        let total: usize = counts.iter().map(|count| usize::from(*count)).sum();
        if total > 256 {
            return Err(malformed(at, "huffman table"));
        }
        let symbols = body
            .get(index..index + total)
            .ok_or_else(|| truncated(at, "huffman table"))?;
        index += total;
        let table = decode_table(counts, symbols).map_err(|_| malformed(at, "huffman table"))?;
        let slot = if class == 0 {
            &mut tables.dc[id]
        } else {
            &mut tables.ac[id]
        };
        *slot = Some(Arc::new(table));
    }
    Ok(())
}

fn parse_dri(body: &[u8], at: u64) -> Result<u16, CodecError> {
    match body {
        [hi, lo] => Ok(u16::from_be_bytes([*hi, *lo])),
        _ => Err(malformed(at, "restart")),
    }
}

fn note(marker: u8, body: &[u8], meta: &mut Meta) {
    match marker {
        APP0 if body.starts_with(b"JFIF\0") => meta.jfif = true,
        APP1 => {
            if let Some(orientation) = exif_orientation(body) {
                meta.orientation = Some(orientation);
            }
        }
        APP14 if body.len() >= 12 && body.starts_with(b"Adobe") => {
            meta.adobe = Some(body[11]);
        }
        COM => {
            // Read as a C string, as writers terminate it with NUL or a
            // newline.
            let text = body.strip_suffix(b"\n").unwrap_or(body);
            let text = text.split(|&byte| byte == 0).next().unwrap_or(text);
            if text == b"CS=ITU601" {
                meta.studio_range = true;
            } else if body.starts_with(b"AVID") {
                meta.avid_standard = body.get(12).copied();
            }
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
    let count = usize::from(read_u16(tiff, offset, le)?).min(64);
    let mut cursor = offset.checked_add(2)?;
    for _ in 0..count {
        let entry = tiff.get(cursor..cursor + 12)?;
        cursor += 12;
        if read_u16(entry, 0, le)? != 0x0112 {
            continue;
        }
        let kind = read_u16(entry, 2, le)?;
        if read_u32(entry, 4, le)? != 1 || (kind != 3 && kind != 4) {
            return None;
        }
        let value = if kind == 3 {
            read_u16(entry, 8, le)?
        } else {
            u16::try_from(read_u32(entry, 8, le)?).ok()?
        };
        return (1..=8).contains(&value).then_some(value);
    }
    None
}

fn read_u16(bytes: &[u8], offset: usize, le: bool) -> Option<u16> {
    let pair: [u8; 2] = bytes.get(offset..offset + 2)?.try_into().ok()?;
    Some(if le {
        u16::from_le_bytes(pair)
    } else {
        u16::from_be_bytes(pair)
    })
}

fn read_u32(bytes: &[u8], offset: usize, le: bool) -> Option<u32> {
    let word: [u8; 4] = bytes.get(offset..offset + 4)?.try_into().ok()?;
    Some(if le {
        u32::from_le_bytes(word)
    } else {
        u32::from_be_bytes(word)
    })
}

fn parse_sos(frame: &Frame, body: &[u8], at: u64) -> Result<Scan, CodecError> {
    let Some(&count) = body.first() else {
        return Err(malformed(at, "scan"));
    };
    let ns = usize::from(count);
    if ns == 0 || ns > 4 || body.len() != 4 + 2 * ns {
        return Err(malformed(at, "scan"));
    }
    let mut comps: Vec<ScanComp> = Vec::with_capacity(ns);
    for index in 0..ns {
        let id = body[1 + 2 * index];
        let tables = body[2 + 2 * index];
        let Some(comp_index) = frame.comps.iter().position(|comp| comp.id == id) else {
            return Err(malformed(at, "component"));
        };
        if comps.iter().any(|comp| comp.index == comp_index) {
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
    let ah = body[body.len() - 1] >> 4;
    let al = body[body.len() - 1] & 15;
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
    if frame.baseline
        && comps
            .iter()
            .any(|comp| comp.dc_table > 1 || comp.ac_table > 1)
    {
        return Err(malformed(at, "baseline huffman"));
    }
    Ok(Scan {
        comps,
        ss,
        se,
        ah,
        al,
    })
}

/// Latch each scan component's quantization table on its first scan.
fn latch_quant(
    input: &Cursor<'_>,
    frame: &mut Frame,
    scan: &Scan,
    tables: &Tables,
) -> Result<(), CodecError> {
    for spec in &scan.comps {
        let comp = &mut frame.comps[spec.index];
        if comp.quant.is_none() {
            comp.quant =
                Some(tables.quant[comp.quant_id].ok_or_else(|| input.malformed("quantizer"))?);
        }
    }
    Ok(())
}

fn huffman<'t>(
    slots: &'t [Option<Arc<HuffDecode>>; 4],
    id: usize,
    input: &Cursor<'_>,
) -> Result<&'t HuffDecode, CodecError> {
    slots
        .get(id)
        .and_then(Option::as_deref)
        .ok_or_else(|| input.malformed("huffman table"))
}

/// The MCU grid of a scan: whole MCUs when interleaved, the component's own
/// blocks when not (ITU-T T.81 A.2).
fn scan_grid(frame: &Frame, scan: &Scan) -> (usize, usize, bool) {
    if scan.comps.len() > 1 {
        return (frame.mcu_cols, frame.mcu_rows, true);
    }
    let comp = &frame.comps[scan.comps[0].index];
    (comp.blocks_w, comp.blocks_h, false)
}

/// Read the restart marker that must follow every `interval` MCUs.
fn restart_marker(input: &mut Cursor<'_>, next: &mut u8) -> Result<(), CodecError> {
    input.end_entropy()?;
    let marker = input.take_marker()?;
    if marker != RST0 + *next {
        return Err(input.malformed("restart"));
    }
    *next = (*next + 1) & 7;
    Ok(())
}

struct SeqComp<'t> {
    index: usize,
    dc: &'t HuffDecode,
    ac: &'t HuffDecode,
    quant: [u16; 64],
}

fn sequential_scan(
    input: &mut Cursor<'_>,
    frame: &mut Frame,
    scan: &Scan,
    tables: &Tables,
    restart: u16,
) -> Result<(), CodecError> {
    latch_quant(input, frame, scan, tables)?;
    let mut specs = Vec::with_capacity(scan.comps.len());
    for spec in &scan.comps {
        let comp = &mut frame.comps[spec.index];
        if comp.scanned {
            return Err(input.malformed("scan"));
        }
        comp.scanned = true;
        specs.push(SeqComp {
            index: spec.index,
            dc: huffman(&tables.dc, spec.dc_table, input)?,
            ac: huffman(&tables.ac, spec.ac_table, input)?,
            quant: comp.quant.ok_or_else(|| input.malformed("quantizer"))?,
        });
    }
    let (cols, rows, interleaved) = scan_grid(frame, scan);
    let mut pred = [0i32; 4];
    let mut block = [0i32; 64];
    let mut left = restart;
    let mut next_rst = 0u8;
    for mcu_y in 0..rows {
        for mcu_x in 0..cols {
            if restart > 0 {
                if left == 0 {
                    restart_marker(input, &mut next_rst)?;
                    pred = [0; 4];
                    left = restart;
                }
                left -= 1;
            }
            for (slot, spec) in specs.iter().enumerate() {
                let comp = &mut frame.comps[spec.index];
                let (h, v) = if interleaved {
                    (usize::from(comp.h), usize::from(comp.v))
                } else {
                    (1, 1)
                };
                let stride = comp.padded_w * 8;
                for local_y in 0..v {
                    for local_x in 0..h {
                        let bx = mcu_x * h + local_x;
                        let by = mcu_y * v + local_y;
                        let any_ac = sequential_block(input, spec, &mut pred[slot], &mut block)?;
                        let start = by * 8 * stride + bx * 8;
                        let Some(out) = comp.plane.get_mut(start..) else {
                            return Err(input.malformed("coefficient"));
                        };
                        if any_ac {
                            idct_block(&block, &spec.quant, out, stride);
                            block = [0; 64];
                        } else {
                            idct_dc(block[0], spec.quant[0], out, stride);
                        }
                    }
                }
            }
            input.check()?;
        }
    }
    input.end_entropy()
}

#[inline]
fn extend(bits: u32, size: u32) -> i32 {
    let value = i32::try_from(bits).unwrap_or(0);
    if value < 1 << (size - 1) {
        value - (1 << size) + 1
    } else {
        value
    }
}

/// One baseline or extended sequential block into `block`, natural order.
/// Returns whether any AC coefficient was coded.
#[inline]
fn sequential_block(
    input: &mut Cursor<'_>,
    spec: &SeqComp<'_>,
    pred: &mut i32,
    block: &mut [i32; 64],
) -> Result<bool, CodecError> {
    let size = u32::from(decode_symbol(input, spec.dc)?);
    if size > 15 {
        return Err(input.malformed("magnitude"));
    }
    let diff = if size == 0 {
        0
    } else {
        extend(input.read_bits(size)?, size)
    };
    *pred = pred
        .checked_add(diff)
        .ok_or_else(|| input.malformed("coefficient"))?;
    block[0] = *pred;
    let mut any_ac = false;
    let mut k = 1usize;
    while k < 64 {
        let symbol = decode_symbol(input, spec.ac)?;
        let run = usize::from(symbol >> 4);
        let size = u32::from(symbol & 15);
        if size == 0 {
            if run == 15 {
                k += 16;
                if k > 64 {
                    return Err(input.malformed("coefficient"));
                }
                continue;
            }
            if run != 0 {
                return Err(input.malformed("eob"));
            }
            break;
        }
        k += run;
        if k > 63 {
            return Err(input.malformed("coefficient"));
        }
        block[ZIGZAG[k]] = extend(input.read_bits(size)?, size);
        any_ac = true;
        k += 1;
    }
    Ok(any_ac)
}

fn accept_progression(frame: &mut Frame, scan: &Scan, at: u64) -> Result<(), CodecError> {
    let al = i8::try_from(scan.al).unwrap_or(0);
    let ah = i8::try_from(scan.ah).unwrap_or(0);
    for spec in &scan.comps {
        let comp = &mut frame.comps[spec.index];
        let band = &comp.progress[scan.ss..=scan.se];
        let fits = if scan.ah == 0 {
            band.iter().all(|bit| *bit < 0)
        } else {
            band.iter().all(|bit| *bit == ah)
        };
        if !fits || (scan.ss > 0 && comp.progress[0] < 0) {
            return Err(malformed(at, "progressive scan"));
        }
        comp.progress[scan.ss..=scan.se].fill(al);
        comp.scanned = true;
    }
    Ok(())
}

fn progressive_scan(
    input: &mut Cursor<'_>,
    frame: &mut Frame,
    scan: &Scan,
    tables: &Tables,
    restart: u16,
) -> Result<(), CodecError> {
    latch_quant(input, frame, scan, tables)?;
    accept_progression(frame, scan, input.at())?;
    let mut dc_tables = Vec::with_capacity(scan.comps.len());
    let mut ac_tables = Vec::with_capacity(scan.comps.len());
    for spec in &scan.comps {
        dc_tables.push(if scan.ss == 0 && scan.ah == 0 {
            Some(huffman(&tables.dc, spec.dc_table, input)?)
        } else {
            None
        });
        ac_tables.push(if scan.ss > 0 {
            Some(huffman(&tables.ac, spec.ac_table, input)?)
        } else {
            None
        });
    }
    let (cols, rows, interleaved) = scan_grid(frame, scan);
    let mut pred = [0i32; 4];
    let mut eobrun = 0u32;
    let mut left = restart;
    let mut next_rst = 0u8;
    for mcu_y in 0..rows {
        for mcu_x in 0..cols {
            if restart > 0 {
                if left == 0 {
                    if eobrun > 0 {
                        return Err(input.malformed("eob"));
                    }
                    restart_marker(input, &mut next_rst)?;
                    pred = [0; 4];
                    left = restart;
                }
                left -= 1;
            }
            for (slot, spec) in scan.comps.iter().enumerate() {
                let comp = &mut frame.comps[spec.index];
                let (h, v) = if interleaved {
                    (usize::from(comp.h), usize::from(comp.v))
                } else {
                    (1, 1)
                };
                for local_y in 0..v {
                    for local_x in 0..h {
                        let bx = mcu_x * h + local_x;
                        let by = mcu_y * v + local_y;
                        let start = (by * comp.padded_w + bx) * 64;
                        let Some(block) = comp.coeffs.get_mut(start..start + 64) else {
                            return Err(input.malformed("coefficient"));
                        };
                        if scan.ss == 0 {
                            if let Some(table) = dc_tables[slot] {
                                block[0] = first_dc(input, table, &mut pred[slot], scan.al)?;
                            } else if input.read_bit()? == 1 {
                                block[0] |= 1 << scan.al;
                            }
                        } else if let Some(table) = ac_tables[slot] {
                            eobrun = if scan.ah == 0 {
                                first_ac(input, block, table, scan, eobrun)?
                            } else {
                                refine_ac(input, block, table, scan, eobrun)?
                            };
                        }
                    }
                }
            }
            input.check()?;
        }
    }
    if eobrun > 0 {
        return Err(input.malformed("eob"));
    }
    input.end_entropy()
}

fn first_dc(
    input: &mut Cursor<'_>,
    table: &HuffDecode,
    pred: &mut i32,
    al: u8,
) -> Result<i32, CodecError> {
    let size = u32::from(decode_symbol(input, table)?);
    if size > 15 {
        return Err(input.malformed("magnitude"));
    }
    let diff = if size == 0 {
        0
    } else {
        extend(input.read_bits(size)?, size)
    };
    *pred = pred
        .checked_add(diff)
        .ok_or_else(|| input.malformed("coefficient"))?;
    shift_coeff(*pred, al, input)
}

fn shift_coeff(value: i32, al: u8, input: &Cursor<'_>) -> Result<i32, CodecError> {
    i32::try_from(i64::from(value) << al).map_err(|_| input.malformed("coefficient"))
}

fn first_ac(
    input: &mut Cursor<'_>,
    block: &mut [i32],
    table: &HuffDecode,
    scan: &Scan,
    eobrun: u32,
) -> Result<u32, CodecError> {
    if eobrun > 0 {
        return Ok(eobrun - 1);
    }
    let mut k = scan.ss;
    while k <= scan.se {
        let symbol = decode_symbol(input, table)?;
        let run = u32::from(symbol >> 4);
        let size = u32::from(symbol & 15);
        if size == 0 {
            if run == 15 {
                if scan.se + 1 - k < 16 {
                    return Err(input.malformed("coefficient"));
                }
                k += 16;
                continue;
            }
            let extra = input.read_bits(run)?;
            return Ok((1u32 << run) + extra - 1);
        }
        k += usize::try_from(run).unwrap_or(usize::MAX);
        if k > scan.se {
            return Err(input.malformed("coefficient"));
        }
        let value = extend(input.read_bits(size)?, size);
        block[ZIGZAG[k]] = shift_coeff(value, scan.al, input)?;
        k += 1;
    }
    Ok(0)
}

fn refine_ac(
    input: &mut Cursor<'_>,
    block: &mut [i32],
    table: &HuffDecode,
    scan: &Scan,
    eobrun: u32,
) -> Result<u32, CodecError> {
    let step = 1i32 << scan.al;
    let mut k = scan.ss;
    if eobrun == 0 {
        while k <= scan.se {
            let symbol = decode_symbol(input, table)?;
            let mut run = i32::from(symbol >> 4);
            let size = symbol & 15;
            let new_coef = if size == 0 {
                if run != 15 {
                    let extra = input.read_bits(u32::try_from(run).unwrap_or(0))?;
                    let eob = (1u32 << run) + extra;
                    refine_band(input, block, k, scan.se, step)?;
                    return Ok(eob - 1);
                }
                0
            } else {
                if size != 1 {
                    return Err(input.malformed("refinement"));
                }
                if input.read_bit()? == 1 { step } else { -step }
            };
            loop {
                if k > scan.se {
                    return Err(input.malformed("coefficient"));
                }
                let natural = ZIGZAG[k];
                if block[natural] != 0 {
                    refine_one(input, &mut block[natural], step)?;
                } else {
                    run -= 1;
                    if run < 0 {
                        break;
                    }
                }
                k += 1;
            }
            if new_coef != 0 {
                block[ZIGZAG[k]] = new_coef;
            }
            k += 1;
        }
        return Ok(0);
    }
    refine_band(input, block, k, scan.se, step)?;
    Ok(eobrun - 1)
}

fn refine_band(
    input: &mut Cursor<'_>,
    block: &mut [i32],
    from: usize,
    to: usize,
    step: i32,
) -> Result<(), CodecError> {
    for natural in ZIGZAG.get(from..=to).unwrap_or(&[]) {
        if block[*natural] != 0 {
            refine_one(input, &mut block[*natural], step)?;
        }
    }
    Ok(())
}

fn refine_one(input: &mut Cursor<'_>, coeff: &mut i32, step: i32) -> Result<(), CodecError> {
    if input.read_bit()? == 0 || *coeff & step != 0 {
        return Ok(());
    }
    if *coeff >= 0 {
        *coeff += step;
    } else {
        *coeff -= step;
    }
    Ok(())
}

fn finish(mut frame: Frame, meta: &Meta) -> Result<Image, CodecError> {
    for comp in &mut frame.comps {
        if !comp.scanned || (frame.progressive && comp.progress[0] < 0) {
            return Err(malformed(0, "scan"));
        }
    }
    if frame.progressive {
        for comp in &mut frame.comps {
            let quant = comp.quant.ok_or(malformed(0, "quantizer"))?;
            let stride = comp.padded_w * 8;
            let mut block = [0i32; 64];
            for by in 0..comp.padded_h {
                for bx in 0..comp.padded_w {
                    let start = (by * comp.padded_w + bx) * 64;
                    let Some(coeffs) = comp.coeffs.get(start..start + 64) else {
                        return Err(malformed(0, "coefficient"));
                    };
                    block.copy_from_slice(coeffs);
                    let Some(out) = comp.plane.get_mut(by * 8 * stride + bx * 8..) else {
                        return Err(malformed(0, "coefficient"));
                    };
                    idct_block(&block, &quant, out, stride);
                }
            }
            comp.coeffs = Vec::new();
        }
    }
    let color = color_model(&frame, meta);
    let components = frame
        .comps
        .into_iter()
        .map(|comp| {
            let mut samples = comp.plane;
            crop(
                &mut samples,
                comp.padded_w * 8,
                usize::try_from(comp.width).unwrap_or(0),
                usize::try_from(comp.height).unwrap_or(0),
            );
            Component {
                id: comp.id,
                h: comp.h,
                v: comp.v,
                width: comp.width,
                height: comp.height,
                samples,
            }
        })
        .collect();
    Ok(Image {
        width: frame.width,
        height: frame.height,
        color,
        components,
        max_h: frame.max_h,
        max_v: frame.max_v,
        progressive: frame.progressive,
        jfif: meta.jfif,
        studio_range: meta.studio_range,
        avid_standard: meta.avid_standard,
        orientation: meta.orientation,
    })
}

/// Pack the top-left `width` × `height` of a plane `stride` wide.
fn crop(plane: &mut Vec<u8>, stride: usize, width: usize, height: usize) {
    for row in 1..height {
        plane.copy_within(row * stride..row * stride + width, row * width);
    }
    plane.truncate(width * height);
}

fn color_model(frame: &Frame, meta: &Meta) -> ColorModel {
    if frame.comps.len() == 1 {
        return ColorModel::Gray;
    }
    let ids: Vec<u8> = frame.comps.iter().map(|comp| comp.id).collect();
    if meta.adobe == Some(0) || ids == b"RGB" {
        ColorModel::Rgb
    } else {
        ColorModel::YCbCr
    }
}

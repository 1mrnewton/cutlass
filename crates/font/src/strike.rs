//! Bitmap strikes from `CBLC`/`CBDT`, `EBLC`/`EBDT`, and `sbix`.
//!
//! A color strike stays premultiplied color, or the PNG picture the font
//! stores, still encoded: this crate has no decoder, so the caller decodes
//! it. A gray or mono strike is coverage. JPEG, TIFF, PDF, masks, and
//! component bitmaps are refused instead of being replaced with an empty
//! cell. A face with none of these tables has no strike.

use crate::{Directory, FontError, bytes_at, u16_at, u32_at};

const MAX_STRIKES: usize = 64;
const MAX_RANGES: usize = 256;
const MAX_DUPE: u8 = 4;
const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
/// Most segments read before a JPEG picture's frame header: tables,
/// application data and comments come first, a few of each.
const MAX_JPEG_SEGMENTS: usize = 64;

/// Pixels for one glyph in one strike.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitmapGlyph {
    pub width: u32,
    pub height: u32,
    /// Pixels from the horizontal origin to the left edge of the bitmap.
    pub bearing_x: i32,
    /// Pixels from the horizontal origin up to the top edge of the bitmap.
    pub bearing_y: i32,
    /// Strike pixels per em.
    pub ppem: u16,
    /// The picture is drawn flipped left to right. Only an `sbix` strike
    /// asks for this, for a glyph that faces the other way.
    pub mirrored: bool,
    pub pixels: BitmapPixels,
}

/// Coverage is one byte per pixel. Color is premultiplied RGBA, row 0 at the top.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BitmapPixels {
    Coverage(Vec<u8>),
    Color(Vec<u8>),
    /// One PNG picture as the font stores it. The glyph's `width` and
    /// `height` are the size its header declares; the bytes after the
    /// header are not checked.
    Png(Vec<u8>),
    /// One JPEG picture as the font stores it, which `sbix` allows. The
    /// glyph's `width` and `height` are the size its frame header declares;
    /// the bytes after that header are not checked. It has no alpha.
    Jpeg(Vec<u8>),
}

#[derive(Clone)]
pub(crate) struct Strikes<'a> {
    /// `CBLC`/`CBDT`, or `sbix` when the face has no color bitmap tables.
    color: Option<Source<'a>>,
    /// `EBLC`/`EBDT`.
    gray: Option<Source<'a>>,
}

#[derive(Clone)]
enum Source<'a> {
    Embedded {
        color: bool,
        location: &'a [u8],
        data: &'a [u8],
    },
    Sbix(&'a [u8]),
}

impl<'a> Strikes<'a> {
    pub(crate) fn parse(directory: &Directory<'a>) -> Result<Self, FontError> {
        let color = match embedded(directory, true)? {
            Some(source) => Some(source),
            None => sbix(directory)?,
        };
        let gray = embedded(directory, false)?;
        Ok(Self { color, gray })
    }

    pub(crate) fn is_some(&self) -> bool {
        self.color.is_some() || self.gray.is_some()
    }

    pub(crate) fn has_color(&self) -> bool {
        self.color.is_some()
    }

    /// The glyph from the largest strike that holds it: color first, then gray.
    pub(crate) fn glyph(
        &self,
        glyph_id: u16,
        num_glyphs: u16,
    ) -> Result<Option<BitmapGlyph>, FontError> {
        if glyph_id >= num_glyphs {
            return Err(FontError::UnknownGlyph);
        }
        for source in [&self.color, &self.gray].into_iter().flatten() {
            if let Some(glyph) = source.glyph(glyph_id, num_glyphs, u32::MAX)? {
                return Ok(Some(glyph));
            }
        }
        Ok(None)
    }

    /// The glyph from the largest color strike whose picture is at most
    /// `max_edge` pixels on each side. When every strike is larger, the
    /// smallest is returned and the caller decides.
    pub(crate) fn color_glyph(
        &self,
        glyph_id: u16,
        num_glyphs: u16,
        max_edge: u32,
    ) -> Result<Option<BitmapGlyph>, FontError> {
        if glyph_id >= num_glyphs {
            return Err(FontError::UnknownGlyph);
        }
        match &self.color {
            Some(source) => source.glyph(glyph_id, num_glyphs, max_edge),
            None => Ok(None),
        }
    }
}

impl Source<'_> {
    fn glyph(
        &self,
        glyph_id: u16,
        num_glyphs: u16,
        max_edge: u32,
    ) -> Result<Option<BitmapGlyph>, FontError> {
        match self {
            Self::Embedded {
                color,
                location,
                data,
            } => embedded_glyph(location, data, *color, glyph_id, max_edge),
            Self::Sbix(table) => sbix_glyph(table, glyph_id, num_glyphs, max_edge),
        }
    }
}

/// One glyph found in one strike, before its pixels are unpacked or copied.
struct Found<'a> {
    width: u32,
    height: u32,
    bearing_x: i32,
    bearing_y: i32,
    ppem: u16,
    mirrored: bool,
    data: Data<'a>,
}

enum Data<'a> {
    Empty,
    Packed {
        bytes: &'a [u8],
        bit_depth: u8,
        byte_aligned: bool,
        color_table: bool,
    },
    Png(&'a [u8]),
    Jpeg(&'a [u8]),
}

impl Found<'_> {
    fn fits(&self, max_edge: u32) -> bool {
        self.width <= max_edge && self.height <= max_edge
    }

    fn finish(self) -> Result<BitmapGlyph, FontError> {
        let pixels = match self.data {
            Data::Empty => BitmapPixels::Coverage(Vec::new()),
            Data::Packed {
                bytes,
                bit_depth,
                byte_aligned,
                color_table,
            } => unpack(
                bytes,
                self.width,
                self.height,
                bit_depth,
                byte_aligned,
                color_table,
            )?,
            Data::Png(bytes) => BitmapPixels::Png(bytes.to_vec()),
            Data::Jpeg(bytes) => BitmapPixels::Jpeg(bytes.to_vec()),
        };
        Ok(BitmapGlyph {
            width: self.width,
            height: self.height,
            bearing_x: self.bearing_x,
            bearing_y: self.bearing_y,
            ppem: self.ppem,
            mirrored: self.mirrored,
            pixels,
        })
    }
}

/// The first candidate that fits, from strikes offered largest first. When
/// none fits, the last one, which is the smallest. A strike whose graphic
/// this reader does not accept is passed over for the next; the refusal is
/// the answer only when no strike gave a picture.
fn pick<'a>(
    candidates: impl Iterator<Item = Result<Option<Found<'a>>, FontError>>,
    max_edge: u32,
) -> Result<Option<BitmapGlyph>, FontError> {
    let mut smallest = None;
    let mut refused = false;
    for candidate in candidates {
        match candidate {
            Ok(Some(found)) if found.fits(max_edge) => return found.finish().map(Some),
            Ok(Some(found)) => smallest = Some(found),
            Ok(None) => {}
            Err(FontError::Unsupported) => refused = true,
            Err(error) => return Err(error),
        }
    }
    match smallest {
        Some(found) => found.finish().map(Some),
        None if refused => Err(FontError::Unsupported),
        None => Ok(None),
    }
}

fn sbix<'a>(directory: &Directory<'a>) -> Result<Option<Source<'a>>, FontError> {
    match directory.table(b"sbix") {
        Ok(table) => {
            validate_sbix(table)?;
            Ok(Some(Source::Sbix(table)))
        }
        Err(FontError::MissingTable) => Ok(None),
        Err(error) => Err(error),
    }
}

fn embedded<'a>(directory: &Directory<'a>, color: bool) -> Result<Option<Source<'a>>, FontError> {
    let (location_tag, data_tag) = if color {
        (b"CBLC", b"CBDT")
    } else {
        (b"EBLC", b"EBDT")
    };
    let location = directory.table(location_tag);
    let data = directory.table(data_tag);
    match (location, data) {
        (Ok(location), Ok(data)) => {
            validate_embedded(location, data, color)?;
            Ok(Some(Source::Embedded {
                color,
                location,
                data,
            }))
        }
        (Err(FontError::MissingTable), Err(FontError::MissingTable)) => Ok(None),
        (Err(error), _) | (_, Err(error)) => Err(error),
    }
}

fn validate_embedded(location: &[u8], data: &[u8], color: bool) -> Result<(), FontError> {
    if location.is_empty() || data.is_empty() {
        return Err(FontError::EmptyTable);
    }
    let major = if color { 3 } else { 2 };
    if u16_at(location, 0)? != major || u16_at(location, 2)? != 0 {
        return Err(FontError::Unsupported);
    }
    if u16_at(data, 0)? != major || u16_at(data, 2)? != 0 {
        return Err(FontError::Unsupported);
    }
    let count = strike_count(location)?;
    for index in 0..count {
        let strike = strike_at(location, index)?;
        if strike.bit_depth == 32 && !color {
            return Err(FontError::Unsupported);
        }
        if !matches!(strike.bit_depth, 1 | 2 | 4 | 8 | 32) {
            return Err(FontError::Unsupported);
        }
        let list = bytes_at(location, strike.list_at, strike.ranges * 8)?;
        let mut previous: Option<u16> = None;
        for range in 0..strike.ranges {
            let at = range * 8;
            let first = u16_at(list, at)?;
            let last = u16_at(list, at + 2)?;
            if first > last || previous.is_some_and(|prior| first <= prior) {
                return Err(FontError::BadTable);
            }
            previous = Some(last);
            let subtable = strike.list_at
                + usize::try_from(u32_at(list, at + 4)?).map_err(|_| FontError::BadTable)?;
            let _header = bytes_at(location, subtable, 8)?;
        }
    }
    Ok(())
}

fn validate_sbix(table: &[u8]) -> Result<(), FontError> {
    if table.is_empty() {
        return Err(FontError::EmptyTable);
    }
    if table.len() < 8 {
        return Err(FontError::Truncated);
    }
    if u16_at(table, 0)? != 1 {
        return Err(FontError::Unsupported);
    }
    let flags = u16_at(table, 2)?;
    if flags != 1 && flags != 3 {
        return Err(FontError::BadTable);
    }
    let count = strike_count_sbix(table)?;
    for index in 0..count {
        let offset =
            usize::try_from(u32_at(table, 8 + index * 4)?).map_err(|_| FontError::BadTable)?;
        if offset >= table.len() {
            return Err(FontError::Truncated);
        }
    }
    Ok(())
}

struct Strike {
    list_at: usize,
    ranges: usize,
    bit_depth: u8,
    horizontal: bool,
    ppem: u16,
}

fn strike_count(table: &[u8]) -> Result<usize, FontError> {
    if table.len() < 8 {
        return Err(FontError::Truncated);
    }
    let count = usize::try_from(u32_at(table, 4)?).map_err(|_| FontError::BadTable)?;
    if count == 0 {
        return Err(FontError::EmptyTable);
    }
    if count > MAX_STRIKES {
        return Err(FontError::BadTable);
    }
    let bytes = count.checked_mul(48).ok_or(FontError::BadTable)?;
    if table.len() < 8 + bytes {
        return Err(FontError::Truncated);
    }
    Ok(count)
}

fn strike_at(location: &[u8], index: usize) -> Result<Strike, FontError> {
    let at = 8 + index * 48;
    let record = bytes_at(location, at, 48)?;
    let list_at = usize::try_from(u32_at(record, 0)?).map_err(|_| FontError::BadTable)?;
    let list_size = usize::try_from(u32_at(record, 4)?).map_err(|_| FontError::BadTable)?;
    let ranges = usize::try_from(u32_at(record, 8)?).map_err(|_| FontError::BadTable)?;
    if ranges > MAX_RANGES {
        return Err(FontError::BadTable);
    }
    let records = ranges.checked_mul(8).ok_or(FontError::BadTable)?;
    if list_size < records {
        return Err(FontError::BadTable);
    }
    let _list = bytes_at(location, list_at, list_size)?;
    let ppem = u16::from(record[45]);
    if ppem == 0 {
        return Err(FontError::BadTable);
    }
    Ok(Strike {
        list_at,
        ranges,
        bit_depth: record[46],
        horizontal: record[47] & 0x01 == 0x01,
        ppem,
    })
}

fn embedded_glyph(
    location: &[u8],
    data: &[u8],
    color: bool,
    glyph_id: u16,
    max_edge: u32,
) -> Result<Option<BitmapGlyph>, FontError> {
    let count = strike_count(location)?;
    let mut order: Vec<(u16, usize)> = Vec::with_capacity(count);
    for index in 0..count {
        order.push((strike_at(location, index)?.ppem, index));
    }
    order.sort_by(|left, right| right.0.cmp(&left.0).then(left.1.cmp(&right.1)));
    let candidates = order.into_iter().map(|(_, index)| {
        let strike = strike_at(location, index)?;
        range_hit(location, data, &strike, glyph_id)?
            .map(|hit| found(&hit, color))
            .transpose()
    });
    pick(candidates, max_edge)
}

type Located = (Option<u32>, Option<u32>, Option<Metrics>);

struct Hit<'a> {
    format: u16,
    bytes: &'a [u8],
    shared: Option<Metrics>,
    bit_depth: u8,
    horizontal: bool,
    ppem: u16,
}

#[derive(Clone, Copy)]
struct Metrics {
    height: u8,
    width: u8,
    bearing_x: i8,
    bearing_y: i8,
}

fn range_hit<'a>(
    location: &'a [u8],
    data: &'a [u8],
    strike: &Strike,
    glyph_id: u16,
) -> Result<Option<Hit<'a>>, FontError> {
    let list = bytes_at(location, strike.list_at, strike.ranges * 8)?;
    for range in 0..strike.ranges {
        let at = range * 8;
        let first = u16_at(list, at)?;
        let last = u16_at(list, at + 2)?;
        if glyph_id < first || glyph_id > last {
            continue;
        }
        let subtable = strike.list_at
            + usize::try_from(u32_at(list, at + 4)?).map_err(|_| FontError::BadTable)?;
        return locate(location, data, subtable, first, last, glyph_id, strike);
    }
    Ok(None)
}

fn locate<'a>(
    location: &'a [u8],
    data: &'a [u8],
    subtable: usize,
    first: u16,
    last: u16,
    glyph_id: u16,
    strike: &Strike,
) -> Result<Option<Hit<'a>>, FontError> {
    let header = bytes_at(location, subtable, 8)?;
    let index_format = u16_at(header, 0)?;
    let image_format = u16_at(header, 2)?;
    let image_at = u32_at(header, 4)?;
    let body = location.get(subtable + 8..).ok_or(FontError::Truncated)?;
    let slot = usize::from(glyph_id - first);
    let (start, end, shared) = match index_format {
        1 => offset_pair(body, slot, 4, true)?,
        3 => offset_pair(body, slot, 2, false)?,
        2 => constant_image(body, slot, true)?,
        5 => sparse_constant(body, glyph_id)?,
        4 => sparse_offset(body, glyph_id)?,
        _ => return Err(FontError::Unsupported),
    };
    let Some((start, end)) = start.zip(end) else {
        return Ok(None);
    };
    if end < start {
        return Err(FontError::BadTable);
    }
    let size = end - start;
    if size == 0 {
        return Ok(None);
    }
    let at = image_at.checked_add(start).ok_or(FontError::BadTable)?;
    let bytes = slice_at(data, at, size)?;
    let _ = last;
    Ok(Some(Hit {
        format: image_format,
        bytes,
        shared,
        bit_depth: strike.bit_depth,
        horizontal: strike.horizontal,
        ppem: strike.ppem,
    }))
}

fn offset_pair(body: &[u8], slot: usize, width: usize, wide: bool) -> Result<Located, FontError> {
    let at = slot.checked_mul(width).ok_or(FontError::BadTable)?;
    let pair = bytes_at(body, at, width * 2)?;
    let (start, end) = if wide {
        (u32_at(pair, 0)?, u32_at(pair, 4)?)
    } else {
        (u32::from(u16_at(pair, 0)?), u32::from(u16_at(pair, 2)?))
    };
    if start == end {
        Ok((None, None, None))
    } else {
        Ok((Some(start), Some(end), None))
    }
}

fn constant_image(body: &[u8], slot: usize, with_metrics: bool) -> Result<Located, FontError> {
    let image_size = u32_at(body, 0)?;
    let metrics = if with_metrics {
        Some(metrics_at(body, 4, true)?.0)
    } else {
        None
    };
    if image_size == 0 {
        return Ok((None, None, metrics));
    }
    let start = image_size
        .checked_mul(u32::try_from(slot).map_err(|_| FontError::BadTable)?)
        .ok_or(FontError::BadTable)?;
    let end = start.checked_add(image_size).ok_or(FontError::BadTable)?;
    Ok((Some(start), Some(end), metrics))
}

fn sparse_constant(body: &[u8], glyph_id: u16) -> Result<Located, FontError> {
    let image_size = u32_at(body, 0)?;
    let (metrics, next) = metrics_at(body, 4, true)?;
    let count = usize::try_from(u32_at(body, next)?).map_err(|_| FontError::BadTable)?;
    let ids = bytes_at(
        body,
        next + 4,
        count.checked_mul(2).ok_or(FontError::BadTable)?,
    )?;
    let mut previous: Option<u16> = None;
    for index in 0..count {
        let id = u16_at(ids, index * 2)?;
        if previous.is_some_and(|prior| id <= prior) {
            return Err(FontError::BadTable);
        }
        previous = Some(id);
        if id == glyph_id {
            if image_size == 0 {
                return Ok((None, None, Some(metrics)));
            }
            let start = image_size
                .checked_mul(u32::try_from(index).map_err(|_| FontError::BadTable)?)
                .ok_or(FontError::BadTable)?;
            let end = start.checked_add(image_size).ok_or(FontError::BadTable)?;
            return Ok((Some(start), Some(end), Some(metrics)));
        }
    }
    Ok((None, None, None))
}

fn sparse_offset(body: &[u8], glyph_id: u16) -> Result<Located, FontError> {
    let count = usize::try_from(u32_at(body, 0)?).map_err(|_| FontError::BadTable)?;
    let bytes = count
        .checked_add(1)
        .and_then(|pairs| pairs.checked_mul(4))
        .ok_or(FontError::BadTable)?;
    let pairs = bytes_at(body, 4, bytes)?;
    let mut previous: Option<u16> = None;
    for index in 0..count {
        let id = u16_at(pairs, index * 4)?;
        if previous.is_some_and(|prior| id <= prior) {
            return Err(FontError::BadTable);
        }
        previous = Some(id);
        if id == glyph_id {
            let start = u32::from(u16_at(pairs, index * 4 + 2)?);
            let end = u32::from(u16_at(pairs, index * 4 + 6)?);
            if start == end {
                return Ok((None, None, None));
            }
            return Ok((Some(start), Some(end), None));
        }
    }
    Ok((None, None, None))
}

fn found<'a>(hit: &Hit<'a>, color_table: bool) -> Result<Found<'a>, FontError> {
    match hit.format {
        1 | 2 | 5 | 6 | 7 => packed(hit, color_table),
        17..=19 if color_table => png(hit),
        _ => Err(FontError::Unsupported),
    }
}

fn packed<'a>(hit: &Hit<'a>, color_table: bool) -> Result<Found<'a>, FontError> {
    let (metrics, pixels_at) = if matches!(hit.format, 5) {
        let metrics = hit.shared.ok_or(FontError::BadTable)?;
        (metrics, 0)
    } else {
        let big = matches!(hit.format, 6 | 7);
        metrics_at(hit.bytes, 0, big)?
    };
    if !hit.horizontal && matches!(hit.format, 1 | 2) {
        return Err(FontError::Unsupported);
    }
    if metrics.width == 0 || metrics.height == 0 {
        return Ok(Found {
            width: 0,
            height: 0,
            bearing_x: 0,
            bearing_y: 0,
            ppem: hit.ppem,
            mirrored: false,
            data: Data::Empty,
        });
    }
    Ok(Found {
        width: u32::from(metrics.width),
        height: u32::from(metrics.height),
        bearing_x: i32::from(metrics.bearing_x),
        bearing_y: i32::from(metrics.bearing_y),
        ppem: hit.ppem,
        mirrored: false,
        data: Data::Packed {
            bytes: hit.bytes.get(pixels_at..).ok_or(FontError::Truncated)?,
            bit_depth: hit.bit_depth,
            byte_aligned: matches!(hit.format, 1 | 6),
            color_table,
        },
    })
}

/// Formats 17, 18 and 19: a PNG picture after small metrics, after big
/// metrics, or with the metrics its index subtable shares. The bearings
/// come from the metrics and the size from the picture's own header.
fn png<'a>(hit: &Hit<'a>) -> Result<Found<'a>, FontError> {
    let (metrics, at) = match hit.format {
        17 => metrics_at(hit.bytes, 0, false)?,
        18 => metrics_at(hit.bytes, 0, true)?,
        _ => (hit.shared.ok_or(FontError::BadTable)?, 0),
    };
    let len = usize::try_from(u32_at(hit.bytes, at)?).map_err(|_| FontError::BadTable)?;
    let picture = bytes_at(hit.bytes, at + 4, len)?;
    let (width, height) = png_size(picture)?;
    Ok(Found {
        width,
        height,
        bearing_x: i32::from(metrics.bearing_x),
        bearing_y: i32::from(metrics.bearing_y),
        ppem: hit.ppem,
        mirrored: false,
        data: Data::Png(picture),
    })
}

/// The width and height a PNG picture's header declares.
fn png_size(picture: &[u8]) -> Result<(u32, u32), FontError> {
    let header = bytes_at(picture, 0, 24)?;
    if header[..8] != PNG_SIGNATURE || &header[12..16] != b"IHDR" {
        return Err(FontError::BadTable);
    }
    let width = u32_at(header, 16)?;
    let height = u32_at(header, 20)?;
    if width == 0 || height == 0 {
        return Err(FontError::BadTable);
    }
    Ok((width, height))
}

/// The size the frame header of a JPEG picture declares.
///
/// The header is the first of its kind after the tables, application data
/// and comments that open the picture. A picture coded losslessly, in
/// layers or arithmetically is not read by the decoder and is refused here.
fn jpeg_size(picture: &[u8]) -> Result<(u32, u32), FontError> {
    if bytes_at(picture, 0, 2)? != [0xFF, 0xD8] {
        return Err(FontError::BadTable);
    }
    let mut at = 2_usize;
    for _ in 0..MAX_JPEG_SEGMENTS {
        let marker = bytes_at(picture, at, 2)?;
        if marker[0] != 0xFF {
            return Err(FontError::BadTable);
        }
        match marker[1] {
            // The baseline, extended and progressive frames.
            0xC0..=0xC2 => {
                let frame = bytes_at(picture, at + 2, 7)?;
                let height = u32::from(u16::from_be_bytes([frame[3], frame[4]]));
                let width = u32::from(u16::from_be_bytes([frame[5], frame[6]]));
                if width == 0 || height == 0 {
                    return Err(FontError::BadTable);
                }
                return Ok((width, height));
            }
            // Every other frame; 0xC4, 0xC8 and 0xCC are not frames.
            0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                return Err(FontError::Unsupported);
            }
            // The picture ends, or its samples start, with no frame named.
            0xD0..=0xDA | 0x00 | 0x01 | 0xFF => return Err(FontError::BadTable),
            _ => {}
        }
        let length = usize::from(u16_at(picture, at + 2)?);
        if length < 2 {
            return Err(FontError::BadTable);
        }
        at = at
            .checked_add(2)
            .and_then(|at| at.checked_add(length))
            .ok_or(FontError::BadTable)?;
    }
    Err(FontError::BadTable)
}

fn metrics_at(data: &[u8], offset: usize, big: bool) -> Result<(Metrics, usize), FontError> {
    let need = if big { 8 } else { 5 };
    let bytes = bytes_at(data, offset, need)?;
    Ok((
        Metrics {
            height: bytes[0],
            width: bytes[1],
            bearing_x: i8::from_be_bytes([bytes[2]]),
            bearing_y: i8::from_be_bytes([bytes[3]]),
        },
        offset + need,
    ))
}

fn unpack(
    data: &[u8],
    width: u32,
    height: u32,
    bit_depth: u8,
    byte_aligned: bool,
    color_table: bool,
) -> Result<BitmapPixels, FontError> {
    if bit_depth == 32 && !color_table {
        return Err(FontError::Unsupported);
    }
    let mut cursor = BitCursor { data, bit: 0 };
    let mut coverage = Vec::new();
    let mut color = Vec::new();
    for _ in 0..height {
        for _ in 0..width {
            let value = cursor.take(bit_depth)?;
            match bit_depth {
                1 => coverage.push(if value == 0 { 0 } else { 255 }),
                2 => coverage.push(u8::try_from(value * 85).map_err(|_| FontError::BadTable)?),
                4 => coverage.push(u8::try_from(value * 17).map_err(|_| FontError::BadTable)?),
                8 => coverage.push(u8::try_from(value).map_err(|_| FontError::BadTable)?),
                32 => {
                    let bytes = value.to_be_bytes();
                    color.extend_from_slice(&[bytes[2], bytes[1], bytes[0], bytes[3]]);
                }
                _ => return Err(FontError::Unsupported),
            }
        }
        if byte_aligned {
            cursor.align();
        }
    }
    if bit_depth == 32 {
        Ok(BitmapPixels::Color(color))
    } else {
        Ok(BitmapPixels::Coverage(coverage))
    }
}

struct BitCursor<'a> {
    data: &'a [u8],
    bit: usize,
}

impl BitCursor<'_> {
    fn take(&mut self, bits: u8) -> Result<u32, FontError> {
        let mut value = 0_u32;
        for _ in 0..bits {
            let byte = *self.data.get(self.bit / 8).ok_or(FontError::Truncated)?;
            let shift = 7 - (self.bit % 8);
            value = (value << 1) | u32::from((byte >> shift) & 1);
            self.bit += 1;
        }
        Ok(value)
    }

    fn align(&mut self) {
        let extra = self.bit % 8;
        if extra != 0 {
            self.bit += 8 - extra;
        }
    }
}

fn slice_at(data: &[u8], offset: u32, len: u32) -> Result<&[u8], FontError> {
    let offset = usize::try_from(offset).map_err(|_| FontError::BadTable)?;
    let len = usize::try_from(len).map_err(|_| FontError::BadTable)?;
    bytes_at(data, offset, len)
}

fn sbix_glyph(
    table: &[u8],
    glyph_id: u16,
    num_glyphs: u16,
    max_edge: u32,
) -> Result<Option<BitmapGlyph>, FontError> {
    let count = strike_count_sbix(table)?;
    let mut strikes = Vec::with_capacity(count);
    for index in 0..count {
        let offset =
            usize::try_from(u32_at(table, 8 + index * 4)?).map_err(|_| FontError::BadTable)?;
        let strike = table.get(offset..).ok_or(FontError::Truncated)?;
        if strike.len() < 4 {
            return Err(FontError::Truncated);
        }
        let ppem = u16_at(strike, 0)?;
        if ppem == 0 {
            return Err(FontError::BadTable);
        }
        strikes.push((ppem, offset));
    }
    strikes.sort_by(|left, right| right.0.cmp(&left.0).then(left.1.cmp(&right.1)));
    let candidates = strikes.into_iter().map(|(ppem, offset)| {
        let strike = SbixStrike {
            table,
            at: offset,
            num_glyphs,
            ppem,
        };
        strike.glyph(glyph_id, false, 0)
    });
    pick(candidates, max_edge)
}

fn strike_count_sbix(table: &[u8]) -> Result<usize, FontError> {
    let count = usize::try_from(u32_at(table, 4)?).map_err(|_| FontError::BadTable)?;
    if count == 0 {
        return Err(FontError::EmptyTable);
    }
    if count > MAX_STRIKES {
        return Err(FontError::BadTable);
    }
    let bytes = count.checked_mul(4).ok_or(FontError::BadTable)?;
    if table.len() < 8 + bytes {
        return Err(FontError::Truncated);
    }
    Ok(count)
}

struct SbixStrike<'a> {
    table: &'a [u8],
    at: usize,
    num_glyphs: u16,
    ppem: u16,
}

impl<'a> SbixStrike<'a> {
    /// The glyph's picture in this strike. A `dupe` or `flip` record names
    /// another glyph of the same strike, whose picture is used as it is or
    /// flipped left to right.
    fn glyph(
        &self,
        glyph_id: u16,
        mirrored: bool,
        depth: u8,
    ) -> Result<Option<Found<'a>>, FontError> {
        if depth > MAX_DUPE || glyph_id >= self.num_glyphs {
            return Err(FontError::BadTable);
        }
        let entries = usize::from(self.num_glyphs)
            .checked_add(1)
            .and_then(|count| count.checked_mul(4))
            .ok_or(FontError::BadTable)?;
        let array_at = self.at.checked_add(4).ok_or(FontError::BadTable)?;
        let array = bytes_at(self.table, array_at, entries)?;
        let pair = bytes_at(array, usize::from(glyph_id) * 4, 8)?;
        let start = usize::try_from(u32_at(pair, 0)?).map_err(|_| FontError::BadTable)?;
        let end = usize::try_from(u32_at(pair, 4)?).map_err(|_| FontError::BadTable)?;
        if start == end {
            return Ok(None);
        }
        if end < start {
            return Err(FontError::BadTable);
        }
        let record_at = self.at.checked_add(start).ok_or(FontError::BadTable)?;
        let record = bytes_at(self.table, record_at, end - start)?;
        if record.len() < 8 {
            return Err(FontError::Truncated);
        }
        let kind = &record[4..8];
        if kind == b"dupe" || kind == b"flip" {
            let next = u16_at(record, 8)?;
            let mirrored = mirrored != (kind == b"flip");
            return self.glyph(next, mirrored, depth + 1);
        }
        let picture = &record[8..];
        // A picture stored as TIFF or PDF, which the table also allows and
        // no face found holds, is not read.
        let ((width, height), data) = match kind {
            b"png " => (png_size(picture)?, Data::Png(picture)),
            b"jpg " => (jpeg_size(picture)?, Data::Jpeg(picture)),
            _ => return Err(FontError::Unsupported),
        };
        // The origin offsets place the picture's left and bottom edges.
        let origin_x = i16::from_be_bytes([record[0], record[1]]);
        let origin_y = i16::from_be_bytes([record[2], record[3]]);
        let rise = i32::try_from(height).map_err(|_| FontError::BadTable)?;
        Ok(Some(Found {
            width,
            height,
            bearing_x: i32::from(origin_x),
            bearing_y: i32::from(origin_y)
                .checked_add(rise)
                .ok_or(FontError::BadTable)?,
            ppem: self.ppem,
            mirrored,
            data,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::BitmapPixels;
    use crate::{FontError, read};

    const COLOR: &[u8] = include_bytes!("../fixtures/color-strike.otf");
    const GRAY: &[u8] = include_bytes!("../fixtures/gray-strike.otf");
    const NO_OUTLINE: &[u8] = include_bytes!("../fixtures/no-outline.otf");

    #[test]
    fn a_color_strike_keeps_premultiplied_channels() {
        let font = read(COLOR).expect("color");
        assert!(font.has_bitmap_strike());
        let glyph = font.bitmap(1).expect("glyph").expect("pixels");
        assert_eq!(glyph.width, 2);
        assert_eq!(glyph.height, 2);
        assert_eq!(glyph.bearing_x, 0);
        assert_eq!(glyph.bearing_y, 2);
        assert_eq!(glyph.ppem, 16);
        assert_eq!(
            glyph.pixels,
            BitmapPixels::Color(vec![
                255, 0, 0, 255, // opaque red
                0, 128, 0, 128, // half green, still premultiplied
                0, 0, 255, 255, // opaque blue
                0, 0, 0, 0,
            ])
        );
        assert!(font.bitmap(0).expect("missing").is_none());
    }

    #[test]
    fn a_gray_strike_is_coverage() {
        let font = read(GRAY).expect("gray");
        let glyph = font.bitmap(1).expect("glyph").expect("pixels");
        assert_eq!(glyph.pixels, BitmapPixels::Coverage(vec![255, 0, 128, 64]));
        assert_eq!(glyph.ppem, 16);
    }

    #[test]
    fn a_face_without_a_strike_has_none() {
        let font = read(NO_OUTLINE).expect("cff2");
        assert!(!font.has_bitmap_strike());
        assert!(font.bitmap(0).expect("none").is_none());
    }

    #[test]
    fn a_truncated_strike_or_an_encoded_graphic_is_an_error() {
        let mut short = COLOR.to_vec();
        short.truncate(short.len() - 4);
        assert!(matches!(
            read(&short).and_then(|font| font.bitmap(1)),
            Err(FontError::Truncated)
        ));
        assert!(matches!(
            read(&COLOR[..20]),
            Err(FontError::Truncated) | Err(FontError::BadTable)
        ));
        let mono = mono_font();
        let font = read(&mono).expect("mono");
        let glyph = font.bitmap(1).expect("mono").expect("pixels");
        assert_eq!(
            glyph.pixels,
            BitmapPixels::Coverage(vec![255, 0, 255, 255, 0, 0, 0, 0])
        );
        // Four bytes are not a PNG picture, in either kind of color strike.
        let png = png_font();
        let font = read(&png).expect("png");
        assert!(matches!(font.bitmap(1), Err(FontError::Truncated)));
        let sbix = sbix_font(b"png ");
        let font = read(&sbix).expect("sbix");
        assert!(font.has_bitmap_strike());
        assert!(matches!(font.bitmap(1), Err(FontError::Truncated)));
        // A record that names itself never ends.
        let dupe = sbix_font(b"dupe");
        assert!(matches!(
            read(&dupe).and_then(|font| font.bitmap(1)),
            Err(FontError::BadTable)
        ));
        let other = sbix_font(b"tiff");
        assert!(matches!(
            read(&other).and_then(|font| font.bitmap(1)),
            Err(FontError::Unsupported)
        ));
    }

    const SBIX: &[u8] = include_bytes!("../fixtures/sbix-png.ttf");
    const CBDT: &[u8] = include_bytes!("../fixtures/cbdt-png.ttf");

    #[test]
    fn an_sbix_png_picture_is_handed_back_encoded_with_its_place() {
        let font = read(SBIX).expect("sbix");
        assert!(font.has_outlines());
        assert!(font.has_color_strike());
        assert!(font.outline(2).expect("outline").contours.is_empty());
        let glyph = font.color_bitmap(2, 256).expect("glyph").expect("picture");
        assert_eq!((glyph.width, glyph.height, glyph.ppem), (2, 2, 2));
        assert_eq!((glyph.bearing_x, glyph.bearing_y), (0, 2));
        assert!(!glyph.mirrored);
        assert_eq!(
            glyph.pixels,
            BitmapPixels::Png(crate::fixture_color::rgba_picture())
        );
        // The origin offsets move the left and bottom edges.
        let moved = font.color_bitmap(14, 256).expect("glyph").expect("moved");
        assert_eq!((moved.bearing_x, moved.bearing_y), (3, 1));
        assert_eq!(moved.pixels, glyph.pixels);
        // A flipped record and a repeated one both use glyph 2's picture.
        let flipped = font.color_bitmap(7, 256).expect("glyph").expect("flip");
        assert!(flipped.mirrored);
        assert_eq!(flipped.pixels, glyph.pixels);
        let repeated = font.color_bitmap(8, 256).expect("glyph").expect("dupe");
        assert!(!repeated.mirrored);
        assert_eq!(repeated.pixels, glyph.pixels);
        // The letter has an outline and no picture.
        assert!(font.color_bitmap(1, 256).expect("letter").is_none());
        assert!(font.bitmap(1).expect("letter").is_none());
        assert_eq!(font.color_bitmap(20, 256), Err(FontError::UnknownGlyph));
    }

    #[test]
    fn the_largest_strike_that_fits_is_chosen() {
        let font = read(SBIX).expect("sbix");
        let largest = font.bitmap(2).expect("glyph").expect("largest");
        assert_eq!((largest.width, largest.ppem), (600, 600));
        let fitting = font.color_bitmap(2, 256).expect("glyph").expect("fits");
        assert_eq!((fitting.width, fitting.ppem), (2, 2));
        let exact = font.color_bitmap(2, 600).expect("glyph").expect("exact");
        assert_eq!(exact.ppem, 600);
        let tiny = font.color_bitmap(2, 1).expect("glyph").expect("tiny");
        assert_eq!((tiny.width, tiny.ppem), (1, 1));
        // When nothing fits, the smallest is handed back for the caller to refuse.
        let none = font.color_bitmap(2, 0).expect("glyph").expect("smallest");
        assert_eq!(none.ppem, 1);
        // The size a header declares is reported and not believed here.
        let bomb = font.color_bitmap(9, 256).expect("glyph").expect("bomb");
        assert_eq!((bomb.width, bomb.height), (100_000, 100_000));
    }

    #[test]
    fn an_sbix_jpeg_picture_is_handed_back_encoded_with_its_size() {
        let font = read(SBIX).expect("sbix");
        let glyph = font.color_bitmap(16, 256).expect("glyph").expect("picture");
        assert_eq!((glyph.width, glyph.height, glyph.ppem), (2, 2, 2));
        assert_eq!((glyph.bearing_x, glyph.bearing_y), (0, 2));
        assert!(!glyph.mirrored);
        let BitmapPixels::Jpeg(bytes) = &glyph.pixels else {
            panic!("not a jpeg: {:?}", glyph.pixels);
        };
        assert_eq!(bytes[..2], [0xFF, 0xD8]);
        assert_eq!(bytes[bytes.len() - 2..], [0xFF, 0xD9]);
        // The flipped copy is the same bytes, to be mirrored once decoded.
        let flipped = font.color_bitmap(19, 256).expect("glyph").expect("flip");
        assert!(flipped.mirrored);
        assert_eq!(flipped.pixels, glyph.pixels);
        // The size a frame header declares is reported and not believed here.
        let bomb = font.color_bitmap(17, 256).expect("glyph").expect("bomb");
        assert_eq!((bomb.width, bomb.height), (60_000, 60_000));
        // Cut short after its frame header, a picture still has its size.
        let short = font.color_bitmap(18, 256).expect("glyph").expect("short");
        assert_eq!((short.width, short.height), (2, 2));
    }

    #[test]
    fn the_size_of_a_jpeg_picture_is_read_from_its_frame_header_or_refused() {
        use super::jpeg_size;
        const PICTURE: &[u8] = include_bytes!("../fixtures/picture.jpg");
        assert_eq!(jpeg_size(PICTURE), Ok((2, 2)));
        let frame = PICTURE
            .windows(2)
            .position(|pair| pair == [0xFF, 0xC0])
            .expect("frame");
        // Every cut of the bytes before the size is an error and no panic.
        for end in 0..frame + 9 {
            assert!(jpeg_size(&PICTURE[..end]).is_err(), "{end}");
        }
        assert_eq!(jpeg_size(&PICTURE[..frame + 9]), Ok((2, 2)));
        // Not a JPEG picture, and one that ends with no frame.
        assert_eq!(jpeg_size(b"II*\0abcdefgh"), Err(FontError::BadTable));
        assert_eq!(
            jpeg_size(&[0xFF, 0xD8, 0xFF, 0xD9]),
            Err(FontError::BadTable)
        );
        // A frame with no pixels.
        let mut empty = PICTURE.to_vec();
        empty[frame + 5..frame + 7].copy_from_slice(&[0, 0]);
        assert_eq!(jpeg_size(&empty), Err(FontError::BadTable));
        // Lossless, layered and arithmetic frames are not decoded.
        for kind in [0xC3, 0xC5, 0xC9, 0xCA, 0xCF] {
            let mut other = PICTURE.to_vec();
            other[frame + 1] = kind;
            assert_eq!(jpeg_size(&other), Err(FontError::Unsupported), "{kind:x}");
        }
        // A segment whose length does not cover itself, and segments
        // without end, stop the read.
        let mut stuck = PICTURE.to_vec();
        stuck[4..6].copy_from_slice(&[0, 1]);
        assert_eq!(jpeg_size(&stuck), Err(FontError::BadTable));
        let mut endless = vec![0xFF, 0xD8];
        for _ in 0..100_000 {
            endless.extend([0xFF, 0xFE, 0, 2]);
        }
        let started = std::time::Instant::now();
        assert_eq!(jpeg_size(&endless), Err(FontError::BadTable));
        assert!(started.elapsed() < std::time::Duration::from_millis(50));
    }

    #[test]
    fn a_graphic_that_is_not_png_or_jpeg_is_refused() {
        let font = read(SBIX).expect("sbix");
        assert_eq!(font.color_bitmap(13, 256), Err(FontError::Unsupported));
        assert_eq!(font.bitmap(13), Err(FontError::Unsupported));
        // A strike with a graphic that is refused is passed over for one
        // that holds a PNG picture of the glyph.
        let passed = font.bitmap(15).expect("glyph").expect("picture");
        assert_eq!((passed.width, passed.ppem), (2, 2));
        assert!(matches!(passed.pixels, BitmapPixels::Png(_)));
    }

    #[test]
    fn cbdt_png_pictures_take_their_metrics_from_each_format() {
        let font = read(CBDT).expect("cbdt");
        assert!(font.has_color_strike());
        let small = font.color_bitmap(1, 256).expect("glyph").expect("small");
        assert_eq!((small.width, small.height, small.ppem), (2, 2, 2));
        assert_eq!((small.bearing_x, small.bearing_y), (0, 2));
        assert_eq!(
            small.pixels,
            BitmapPixels::Png(crate::fixture_color::rgba_picture())
        );
        let big = font.color_bitmap(2, 256).expect("glyph").expect("big");
        assert_eq!((big.bearing_x, big.bearing_y), (1, 3));
        assert_eq!(
            big.pixels,
            BitmapPixels::Png(crate::fixture_color::palette_picture())
        );
        let shared = font.color_bitmap(3, 256).expect("glyph").expect("shared");
        assert_eq!((shared.bearing_x, shared.bearing_y), (-1, 2));
        assert_eq!(
            shared.pixels,
            BitmapPixels::Png(crate::fixture_color::gray_alpha_picture())
        );
        assert!(font.color_bitmap(0, 256).expect("missing").is_none());
    }

    #[test]
    fn a_torn_png_strike_is_an_error() {
        // Cut inside the last picture: its bytes run past the table.
        let font = read(CBDT).expect("cbdt");
        let data_at = table_at(CBDT, b"CBDT");
        let mut short = CBDT.to_vec();
        let cut = data_at.1 - 8;
        set_table_length(&mut short, b"CBDT", cut);
        let torn = read(&short).expect("font");
        assert_eq!(torn.color_bitmap(3, 256), Err(FontError::Truncated));
        assert!(font.color_bitmap(3, 256).is_ok());
        // A picture that does not start as a PNG.
        let mut wrong = CBDT.to_vec();
        let at = data_at.0 + 4 + 5 + 4;
        wrong[at] = 0;
        let wrong = read(&wrong).expect("font");
        assert_eq!(wrong.color_bitmap(1, 256), Err(FontError::BadTable));
        // An sbix table cut inside its strikes.
        let mut cut_sbix = SBIX.to_vec();
        set_table_length(&mut cut_sbix, b"sbix", 40);
        assert!(matches!(
            read(&cut_sbix).and_then(|font| font.color_bitmap(2, 256)),
            Err(FontError::Truncated)
        ));
    }

    /// Where a table starts, and its length.
    fn table_at(bytes: &[u8], tag: &[u8; 4]) -> (usize, usize) {
        let at = record_at(bytes, tag);
        let read = |at: usize| {
            usize::try_from(u32::from_be_bytes([
                bytes[at],
                bytes[at + 1],
                bytes[at + 2],
                bytes[at + 3],
            ]))
            .expect("field")
        };
        (read(at + 8), read(at + 12))
    }

    fn record_at(bytes: &[u8], tag: &[u8; 4]) -> usize {
        let count = usize::from(u16::from_be_bytes([bytes[4], bytes[5]]));
        (0..count)
            .map(|index| 12 + index * 16)
            .find(|at| &bytes[*at..*at + 4] == tag)
            .expect("table")
    }

    fn set_table_length(bytes: &mut [u8], tag: &[u8; 4], length: usize) {
        let at = record_at(bytes, tag);
        bytes[at + 12..at + 16]
            .copy_from_slice(&u32::try_from(length).expect("length").to_be_bytes());
    }

    fn mono_font() -> Vec<u8> {
        let metrics = [1_u8, 8, 0, 1, 8];
        let mut image = metrics.to_vec();
        image.push(0b1011_0000);
        embedded_font(false, 1, 1, 1, &image, 6)
    }

    fn png_font() -> Vec<u8> {
        let mut image = vec![1, 1, 0, 1, 1];
        image.extend_from_slice(&4_u32.to_be_bytes());
        image.extend_from_slice(&[0, 1, 2, 3]);
        embedded_font(true, 32, 17, 1, &image, image.len() as u32)
    }

    fn embedded_font(
        color: bool,
        bit_depth: u8,
        image_format: u16,
        ppem: u8,
        image: &[u8],
        image_size: u32,
    ) -> Vec<u8> {
        let mut data = if color {
            vec![0, 3, 0, 0]
        } else {
            vec![0, 2, 0, 0]
        };
        data.extend_from_slice(image);
        let mut location = if color {
            vec![0, 3, 0, 0]
        } else {
            vec![0, 2, 0, 0]
        };
        location.extend_from_slice(&1_u32.to_be_bytes());
        location.extend_from_slice(&56_u32.to_be_bytes());
        location.extend_from_slice(&28_u32.to_be_bytes());
        location.extend_from_slice(&1_u32.to_be_bytes());
        location.extend_from_slice(&[0; 4]);
        location.extend_from_slice(&[0; 24]);
        location.extend_from_slice(&0_u16.to_be_bytes());
        location.extend_from_slice(&1_u16.to_be_bytes());
        location.push(ppem);
        location.push(ppem);
        location.push(bit_depth);
        location.push(1);
        location.extend_from_slice(&0_u16.to_be_bytes());
        location.extend_from_slice(&1_u16.to_be_bytes());
        location.extend_from_slice(&8_u32.to_be_bytes());
        location.extend_from_slice(&1_u16.to_be_bytes());
        location.extend_from_slice(&image_format.to_be_bytes());
        location.extend_from_slice(&4_u32.to_be_bytes());
        location.extend_from_slice(&0_u32.to_be_bytes());
        location.extend_from_slice(&0_u32.to_be_bytes());
        location.extend_from_slice(&image_size.to_be_bytes());
        let mut tables = base_tables();
        if color {
            tables.push((*b"CBDT", data));
            tables.push((*b"CBLC", location));
        } else {
            tables.push((*b"EBDT", data));
            tables.push((*b"EBLC", location));
        }
        sfnt(tables)
    }

    fn sbix_font(kind: &[u8; 4]) -> Vec<u8> {
        let mut table = Vec::new();
        table.extend_from_slice(&1_u16.to_be_bytes());
        table.extend_from_slice(&1_u16.to_be_bytes());
        table.extend_from_slice(&1_u32.to_be_bytes());
        table.extend_from_slice(&12_u32.to_be_bytes());
        table.extend_from_slice(&16_u16.to_be_bytes());
        table.extend_from_slice(&72_u16.to_be_bytes());
        let data_at = 16_u32;
        table.extend_from_slice(&data_at.to_be_bytes());
        table.extend_from_slice(&data_at.to_be_bytes());
        let end = data_at + 8 + if kind == b"dupe" { 2 } else { 4 };
        table.extend_from_slice(&end.to_be_bytes());
        table.extend_from_slice(&0_i16.to_be_bytes());
        table.extend_from_slice(&0_i16.to_be_bytes());
        table.extend_from_slice(kind);
        if kind == b"dupe" {
            table.extend_from_slice(&1_u16.to_be_bytes());
        } else {
            table.extend_from_slice(&[1, 2, 3, 4]);
        }
        let mut tables = base_tables();
        tables.push((*b"sbix", table));
        sfnt(tables)
    }

    fn base_tables() -> Vec<([u8; 4], Vec<u8>)> {
        vec![
            (*b"CFF2", vec![0x02]),
            (
                *b"cmap",
                hex("00000001000300010000000c0006000c0000004100010001"),
            ),
            (
                *b"head",
                hex(
                    "0001000000000000000000005f0f3cf5000003e800000000000000000000000000000000000000000000000000000000000000000000",
                ),
            ),
            (
                *b"hhea",
                hex("000000000000000000000000000000000000000000000000000000000000000000000002"),
            ),
            (*b"hmtx", hex("01f4000001f40000")),
            (*b"maxp", hex("000050000002")),
        ]
    }

    fn hex(text: &str) -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|index| u8::from_str_radix(&text[index..index + 2], 16).expect("hex"))
            .collect()
    }

    fn sfnt(mut tables: Vec<([u8; 4], Vec<u8>)>) -> Vec<u8> {
        tables.sort_by_key(|(tag, _)| *tag);
        let count = u16::try_from(tables.len()).expect("count");
        let mut offset = 12 + tables.len() * 16;
        let mut out = vec![0; offset];
        out[..4].copy_from_slice(b"OTTO");
        out[4..6].copy_from_slice(&count.to_be_bytes());
        for (index, (tag, data)) in tables.iter().enumerate() {
            let at = 12 + index * 16;
            out[at..at + 4].copy_from_slice(tag);
            out[at + 8..at + 12]
                .copy_from_slice(&u32::try_from(offset).expect("offset").to_be_bytes());
            out[at + 12..at + 16]
                .copy_from_slice(&u32::try_from(data.len()).expect("len").to_be_bytes());
            out.extend_from_slice(data);
            offset += data.len();
        }
        out
    }
}

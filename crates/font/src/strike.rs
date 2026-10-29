//! Bitmap strikes from `CBLC`/`CBDT`, `EBLC`/`EBDT`, and `sbix`.
//!
//! A color strike stays premultiplied color. A gray or mono strike is
//! coverage. PNG, JPEG, TIFF, PDF, masks, and component bitmaps are refused
//! instead of being replaced with an empty cell. A face with none of these
//! tables has no strike.

use crate::{Directory, FontError, bytes_at, u16_at, u32_at};

const MAX_STRIKES: usize = 64;
const MAX_RANGES: usize = 256;
const MAX_DUPE: u8 = 4;

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
    pub pixels: BitmapPixels,
}

/// Coverage is one byte per pixel. Color is premultiplied RGBA, row 0 at the top.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BitmapPixels {
    Coverage(Vec<u8>),
    Color(Vec<u8>),
}

pub(crate) struct Strikes<'a> {
    source: Option<Source<'a>>,
}

enum Source<'a> {
    Embedded {
        color: bool,
        location: &'a [u8],
        data: &'a [u8],
    },
    Sbix(&'a [u8]),
}

impl<'a> Strikes<'a> {
    pub(crate) fn none() -> Self {
        Self { source: None }
    }

    pub(crate) fn parse(directory: &Directory<'a>) -> Result<Self, FontError> {
        if let Some(source) = embedded(directory, true)? {
            return Ok(Self {
                source: Some(source),
            });
        }
        if let Some(source) = embedded(directory, false)? {
            return Ok(Self {
                source: Some(source),
            });
        }
        match directory.table(b"sbix") {
            Ok(table) => {
                validate_sbix(table)?;
                Ok(Self {
                    source: Some(Source::Sbix(table)),
                })
            }
            Err(FontError::MissingTable) => Ok(Self::none()),
            Err(error) => Err(error),
        }
    }

    pub(crate) fn is_some(&self) -> bool {
        self.source.is_some()
    }

    pub(crate) fn glyph(
        &self,
        glyph_id: u16,
        num_glyphs: u16,
    ) -> Result<Option<BitmapGlyph>, FontError> {
        if glyph_id >= num_glyphs {
            return Err(FontError::UnknownGlyph);
        }
        let Some(source) = &self.source else {
            return Ok(None);
        };
        match source {
            Source::Embedded {
                color,
                location,
                data,
            } => embedded_glyph(location, data, *color, glyph_id),
            Source::Sbix(table) => sbix_glyph(table, glyph_id, num_glyphs, 0),
        }
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
) -> Result<Option<BitmapGlyph>, FontError> {
    let count = strike_count(location)?;
    let mut order: Vec<(u16, usize)> = Vec::with_capacity(count);
    for index in 0..count {
        order.push((strike_at(location, index)?.ppem, index));
    }
    order.sort_by(|left, right| right.0.cmp(&left.0).then(left.1.cmp(&right.1)));
    for (_, index) in order {
        let strike = strike_at(location, index)?;
        if let Some(hit) = range_hit(location, data, &strike, glyph_id)? {
            return decode_hit(hit, color);
        }
    }
    Ok(None)
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

fn offset_pair(
    body: &[u8],
    slot: usize,
    width: usize,
    wide: bool,
) -> Result<Located, FontError> {
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

fn constant_image(
    body: &[u8],
    slot: usize,
    with_metrics: bool,
) -> Result<Located, FontError> {
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

fn sparse_constant(
    body: &[u8],
    glyph_id: u16,
) -> Result<Located, FontError> {
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

fn sparse_offset(
    body: &[u8],
    glyph_id: u16,
) -> Result<Located, FontError> {
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

fn decode_hit(hit: Hit<'_>, color_table: bool) -> Result<Option<BitmapGlyph>, FontError> {
    if matches!(hit.format, 8 | 9 | 17 | 18 | 19) {
        return refuse_encoded(&hit);
    }
    if !matches!(hit.format, 1 | 2 | 5 | 6 | 7) {
        return Err(FontError::Unsupported);
    }
    let (metrics, pixels_at) = if matches!(hit.format, 5) {
        let metrics = hit.shared.ok_or(FontError::BadTable)?;
        (metrics, 0)
    } else {
        let big = matches!(hit.format, 6 | 7);
        let (metrics, next) = metrics_at(hit.bytes, 0, big)?;
        (metrics, next)
    };
    if !hit.horizontal && matches!(hit.format, 1 | 2) {
        return Err(FontError::Unsupported);
    }
    if metrics.width == 0 || metrics.height == 0 {
        return Ok(Some(empty_glyph(hit.ppem)));
    }
    let byte_aligned = matches!(hit.format, 1 | 6);
    let packed = unpack(
        hit.bytes.get(pixels_at..).ok_or(FontError::Truncated)?,
        metrics.width,
        metrics.height,
        hit.bit_depth,
        byte_aligned,
        color_table,
    )?;
    Ok(Some(BitmapGlyph {
        width: u32::from(metrics.width),
        height: u32::from(metrics.height),
        bearing_x: i32::from(metrics.bearing_x),
        bearing_y: i32::from(metrics.bearing_y),
        ppem: hit.ppem,
        pixels: packed,
    }))
}

fn refuse_encoded(hit: &Hit<'_>) -> Result<Option<BitmapGlyph>, FontError> {
    match hit.format {
        17 => png_length(hit.bytes, 5)?,
        18 => png_length(hit.bytes, 8)?,
        19 => png_length(hit.bytes, 0)?,
        _ => {}
    }
    Err(FontError::Unsupported)
}

fn png_length(bytes: &[u8], metrics: usize) -> Result<(), FontError> {
    let len_at = metrics;
    let header = bytes_at(bytes, len_at, 4)?;
    let len = usize::try_from(u32_at(header, 0)?).map_err(|_| FontError::BadTable)?;
    let start = len_at.checked_add(4).ok_or(FontError::BadTable)?;
    if bytes.len() < start.checked_add(len).ok_or(FontError::BadTable)? {
        return Err(FontError::Truncated);
    }
    Ok(())
}

fn empty_glyph(ppem: u16) -> BitmapGlyph {
    BitmapGlyph {
        width: 0,
        height: 0,
        bearing_x: 0,
        bearing_y: 0,
        ppem,
        pixels: BitmapPixels::Coverage(Vec::new()),
    }
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
    width: u8,
    height: u8,
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
    depth: u8,
) -> Result<Option<BitmapGlyph>, FontError> {
    if depth > MAX_DUPE {
        return Err(FontError::BadTable);
    }
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
    for (ppem, offset) in strikes {
        if let Some(glyph) = sbix_at(table, offset, glyph_id, num_glyphs, ppem, depth)? {
            return Ok(Some(glyph));
        }
    }
    Ok(None)
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

fn sbix_at(
    table: &[u8],
    strike: usize,
    glyph_id: u16,
    num_glyphs: u16,
    ppem: u16,
    depth: u8,
) -> Result<Option<BitmapGlyph>, FontError> {
    let entries = usize::from(num_glyphs)
        .checked_add(1)
        .and_then(|count| count.checked_mul(4))
        .ok_or(FontError::BadTable)?;
    let array_at = strike.checked_add(4).ok_or(FontError::BadTable)?;
    let _entries = bytes_at(table, array_at, entries)?;
    let array = array_at
        .checked_add(
            usize::from(glyph_id)
                .checked_mul(4)
                .ok_or(FontError::BadTable)?,
        )
        .ok_or(FontError::BadTable)?;
    let pair = bytes_at(table, array, 8)?;
    let start = usize::try_from(u32_at(pair, 0)?).map_err(|_| FontError::BadTable)?;
    let end = usize::try_from(u32_at(pair, 4)?).map_err(|_| FontError::BadTable)?;
    if start == end {
        return Ok(None);
    }
    if end < start {
        return Err(FontError::BadTable);
    }
    let glyph = bytes_at(table, strike + start, end - start)?;
    if glyph.len() < 8 {
        return Err(FontError::Truncated);
    }
    let origin_x = i16::from_be_bytes([glyph[0], glyph[1]]);
    let origin_y = i16::from_be_bytes([glyph[2], glyph[3]]);
    let kind = &glyph[4..8];
    if kind == b"dupe" {
        if glyph.len() < 10 {
            return Err(FontError::Truncated);
        }
        let next = u16_at(glyph, 8)?;
        return sbix_glyph(table, next, num_glyphs, depth + 1);
    }
    if kind == b"png " || kind == b"jpg " || kind == b"tiff" || kind == b"pdf " || kind == b"mask" {
        return Err(FontError::Unsupported);
    }
    let _ = (origin_x, origin_y, ppem);
    Err(FontError::Unsupported)
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
        let png = png_font();
        let font = read(&png).expect("png");
        assert!(matches!(font.bitmap(1), Err(FontError::Unsupported)));
        let sbix = sbix_font(b"png ");
        let font = read(&sbix).expect("sbix");
        assert!(font.has_bitmap_strike());
        assert!(matches!(font.bitmap(1), Err(FontError::Unsupported)));
        let dupe = sbix_font(b"dupe");
        assert!(matches!(
            read(&dupe).and_then(|font| font.bitmap(1)),
            Err(FontError::BadTable) | Err(FontError::Truncated)
        ));
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

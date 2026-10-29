//! Reads a TrueType or OpenType font, including one face of a collection.
//!
//! The file yields a glyph id, an advance width, an outline, and horizontal
//! kerning pairs from a `kern` table. Glyph 0 is the missing glyph. A face
//! with `CFF2` and no `CFF ` table has no outlines this reader can draw.
//! Shaping, atlases, and meshes are not part of this crate.

#![forbid(unsafe_code)]

mod collection;
mod kern;
mod otf;
mod ttf;

use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

/// Advance width of one character, in font design units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Advance {
    pub character: char,
    pub glyph_id: u16,
    pub units: u16,
}

/// A closed contour in font design units.
///
/// The contour starts at `(x, y)`. Each segment ends at its own point, and
/// the last segment returns to the start.
#[derive(Clone, Debug, PartialEq)]
pub struct Contour {
    pub x: f32,
    pub y: f32,
    pub segments: Vec<Segment>,
}

/// One edge of a contour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Segment {
    Line {
        x: f32,
        y: f32,
    },
    Quad {
        cx: f32,
        cy: f32,
        x: f32,
        y: f32,
    },
    Cubic {
        c1x: f32,
        c1y: f32,
        c2x: f32,
        c2y: f32,
        x: f32,
        y: f32,
    },
}

/// The outlines of one glyph.
#[derive(Clone, Debug, PartialEq)]
pub struct Outline {
    pub contours: Vec<Contour>,
}

/// Why a font file could not be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FontError {
    /// The file ends before a header or a table that it declares.
    Truncated,
    /// The table directory is empty, or a required table has no bytes.
    EmptyTable,
    /// A table required to map characters or draw glyphs is absent.
    MissingTable,
    /// A table is present, but its fields disagree with the file.
    BadTable,
    /// The character is not in the font's character map.
    MissingGlyph { character: char },
    /// The glyph id is not in this font.
    UnknownGlyph,
    /// `index` is past the last face in the collection.
    CollectionIndex { index: u32, count: u32 },
    /// The bytes are a collection. Open one face with [`read_collection`].
    Collection,
    /// The outline format or encoding is not one this reader accepts.
    Unsupported,
}

impl fmt::Display for FontError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => formatter.write_str("font file is truncated"),
            Self::EmptyTable => formatter.write_str("font table is empty"),
            Self::MissingTable => formatter.write_str("font is missing a required table"),
            Self::BadTable => formatter.write_str("font table is inconsistent"),
            Self::MissingGlyph { character } => {
                write!(formatter, "font has no glyph for {character:?}")
            }
            Self::UnknownGlyph => formatter.write_str("glyph is not in the font"),
            Self::CollectionIndex { index, count } => {
                write!(formatter, "font collection has no face {index} of {count}")
            }
            Self::Collection => formatter.write_str("font file is a collection"),
            Self::Unsupported => formatter.write_str("font outline or encoding is unsupported"),
        }
    }
}

impl Error for FontError {}

/// Which outline tables a face carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FontKind {
    /// TrueType outlines in `glyf`.
    TrueType,
    /// Compact Font Format outlines in `CFF `.
    OpenType,
}

/// One face read from a `.ttf`, `.otf`, or `.ttc` file.
pub struct Font<'a> {
    kind: FontKind,
    units_per_em: u16,
    num_glyphs: u16,
    hmtx: Hmtx<'a>,
    cmap: Cmap<'a>,
    kern: kern::Kerning,
    outlines: Outlines<'a>,
}

/// Read one TrueType or OpenType face.
///
/// A collection (`ttcf`) is refused here. Use [`read_collection`] and a face index.
pub fn read(bytes: &[u8]) -> Result<Font<'_>, FontError> {
    if bytes.len() >= 4 && &bytes[..4] == b"ttcf" {
        return Err(FontError::Collection);
    }
    let directory = Directory::parse(bytes, 0)?;
    Font::from_directory(directory)
}

/// Read face `index` from a TrueType or OpenType collection.
pub fn read_collection(bytes: &[u8], index: u32) -> Result<Font<'_>, FontError> {
    let offset = collection::face_offset(bytes, index)?;
    let directory = Directory::parse(bytes, offset)?;
    Font::from_directory(directory)
}

impl fmt::Debug for Font<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Font")
            .field("kind", &self.kind)
            .field("units_per_em", &self.units_per_em)
            .field("glyph_count", &self.num_glyphs)
            .finish()
    }
}

impl Font<'_> {
    #[must_use]
    pub fn kind(&self) -> FontKind {
        self.kind
    }

    #[must_use]
    pub fn units_per_em(&self) -> u16 {
        self.units_per_em
    }

    #[must_use]
    pub fn glyph_count(&self) -> u16 {
        self.num_glyphs
    }

    /// Advance width of each character, in font design units.
    ///
    /// A character with no glyph fails the whole string. The file is left unread
    /// past that character, and nothing is substituted. [`Self::glyph_id`] reports
    /// that miss without failing the rest of a run.
    pub fn advances(&self, text: &str) -> Result<Vec<Advance>, FontError> {
        let mut advances = Vec::new();
        for character in text.chars() {
            let Some(glyph_id) = self.glyph_id(character)? else {
                return Err(FontError::MissingGlyph { character });
            };
            let units = self.advance(glyph_id)?;
            advances.push(Advance {
                character,
                glyph_id,
                units,
            });
        }
        Ok(advances)
    }

    /// Glyph id for one character.
    ///
    /// `Ok(None)` means the character map has no glyph. The missing glyph is
    /// still [`Self::missing_glyph`]. A map entry past the face's glyph count
    /// is a bad table.
    pub fn glyph_id(&self, character: char) -> Result<Option<u16>, FontError> {
        let Some(glyph_id) = self.cmap.glyph(u32::from(character))? else {
            return Ok(None);
        };
        if glyph_id >= self.num_glyphs {
            return Err(FontError::BadTable);
        }
        Ok(Some(glyph_id))
    }

    /// Glyph 0. OpenType requires this to be the face's missing glyph.
    #[must_use]
    pub fn missing_glyph(&self) -> u16 {
        debug_assert!(self.num_glyphs > 0);
        0
    }

    /// Outline of [`Self::missing_glyph`], in font design units.
    pub fn missing_outline(&self) -> Result<Outline, FontError> {
        self.outline(self.missing_glyph())
    }

    /// Advance width of one glyph, in font design units.
    pub fn advance(&self, glyph_id: u16) -> Result<u16, FontError> {
        if glyph_id >= self.num_glyphs {
            return Err(FontError::UnknownGlyph);
        }
        self.hmtx.advance(glyph_id)
    }

    /// `true` when this face has TrueType or CFF outlines.
    ///
    /// A `CFF2` face is readable and returns `false`. Sampling those outlines
    /// is [`FontError::Unsupported`].
    #[must_use]
    pub fn has_outlines(&self) -> bool {
        match &self.outlines {
            Outlines::TrueType(_) => true,
            Outlines::OpenType(source) => otf::has_outlines(source),
        }
    }

    /// Horizontal kerning for a glyph pair, in font design units.
    ///
    /// The value is added to the advance of `left` before `right` is placed.
    /// A face with no `kern` table, and a pair the table does not store, is 0.
    #[must_use]
    pub fn kerning(&self, left: u16, right: u16) -> i16 {
        self.kern.get(left, right)
    }

    /// Outline of one glyph, in font design units.
    pub fn outline(&self, glyph_id: u16) -> Result<Outline, FontError> {
        if glyph_id >= self.num_glyphs {
            return Err(FontError::UnknownGlyph);
        }
        match &self.outlines {
            Outlines::TrueType(source) => ttf::outline(source, glyph_id, self.num_glyphs),
            Outlines::OpenType(source) => otf::outline(source, glyph_id),
        }
    }

    fn from_directory(directory: Directory<'_>) -> Result<Font<'_>, FontError> {
        let head = parse_head(directory.table(b"head")?)?;
        let num_glyphs = parse_maxp(directory.table(b"maxp")?)?;
        let metrics = parse_hhea(directory.table(b"hhea")?, num_glyphs)?;
        let hmtx = Hmtx::parse(directory.table(b"hmtx")?, metrics, num_glyphs)?;
        let cmap = parse_cmap(directory.table(b"cmap")?)?;
        let kern = match directory.table(b"kern") {
            Ok(table) => kern::parse(table)?,
            Err(FontError::MissingTable) => kern::Kerning::empty(),
            Err(error) => return Err(error),
        };
        let outlines = match directory.kind {
            FontKind::TrueType => {
                Outlines::TrueType(ttf::source(&directory, head.loca, num_glyphs)?)
            }
            FontKind::OpenType => Outlines::OpenType(otf::source(&directory)?),
        };
        Ok(Font {
            kind: directory.kind,
            units_per_em: head.units_per_em,
            num_glyphs,
            hmtx,
            cmap,
            kern,
            outlines,
        })
    }
}

enum Outlines<'a> {
    TrueType(ttf::Source<'a>),
    OpenType(otf::Source<'a>),
}

impl fmt::Debug for Outlines<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TrueType(_) => formatter.write_str("TrueType"),
            Self::OpenType(_) => formatter.write_str("OpenType"),
        }
    }
}

struct Head {
    units_per_em: u16,
    loca: i16,
}

fn parse_head(table: &[u8]) -> Result<Head, FontError> {
    if table.len() < 54 {
        return Err(short(table));
    }
    if u32_at(table, 12)? != 0x5F0F_3CF5 {
        return Err(FontError::BadTable);
    }
    let units_per_em = u16_at(table, 18)?;
    if !(16..=16_384).contains(&units_per_em) {
        return Err(FontError::BadTable);
    }
    let loca = i16_at(table, 50)?;
    if loca != 0 && loca != 1 {
        return Err(FontError::BadTable);
    }
    let format = i16_at(table, 52)?;
    if format != 0 {
        return Err(FontError::Unsupported);
    }
    Ok(Head { units_per_em, loca })
}

fn parse_maxp(table: &[u8]) -> Result<u16, FontError> {
    if table.len() < 6 {
        return Err(short(table));
    }
    let version = u32_at(table, 0)?;
    if version != 0x0001_0000 && version != 0x0000_5000 {
        return Err(FontError::Unsupported);
    }
    let glyphs = u16_at(table, 4)?;
    if glyphs == 0 {
        return Err(FontError::EmptyTable);
    }
    Ok(glyphs)
}

fn parse_hhea(table: &[u8], num_glyphs: u16) -> Result<u16, FontError> {
    if table.len() < 36 {
        return Err(short(table));
    }
    let metrics = u16_at(table, 34)?;
    if metrics == 0 {
        return Err(FontError::EmptyTable);
    }
    if metrics > num_glyphs {
        return Err(FontError::BadTable);
    }
    Ok(metrics)
}

struct Hmtx<'a> {
    data: &'a [u8],
    metrics: u16,
}

impl<'a> Hmtx<'a> {
    fn parse(table: &'a [u8], metrics: u16, num_glyphs: u16) -> Result<Self, FontError> {
        let long = usize::from(metrics)
            .checked_mul(4)
            .ok_or(FontError::BadTable)?;
        let bearings = usize::from(num_glyphs - metrics)
            .checked_mul(2)
            .ok_or(FontError::BadTable)?;
        let need = long.checked_add(bearings).ok_or(FontError::BadTable)?;
        if table.len() < need {
            return Err(short(table));
        }
        Ok(Self {
            data: table,
            metrics,
        })
    }

    fn advance(&self, glyph: u16) -> Result<u16, FontError> {
        if self.metrics == 0 {
            return Err(FontError::EmptyTable);
        }
        let index = if glyph < self.metrics {
            glyph
        } else {
            self.metrics - 1
        };
        let offset = usize::from(index)
            .checked_mul(4)
            .ok_or(FontError::BadTable)?;
        u16_at(self.data, offset)
    }
}

enum Cmap<'a> {
    Format4(Format4<'a>),
    Format6(Format6<'a>),
    Groups(Groups<'a>),
}

impl Cmap<'_> {
    fn glyph(&self, codepoint: u32) -> Result<Option<u16>, FontError> {
        match self {
            Self::Format4(table) => table.glyph(codepoint),
            Self::Format6(table) => table.glyph(codepoint),
            Self::Groups(table) => table.glyph(codepoint),
        }
    }
}

struct Format4<'a> {
    data: &'a [u8],
    seg_count: usize,
    end_at: usize,
    start_at: usize,
    delta_at: usize,
    range_at: usize,
}

struct Format6<'a> {
    glyphs: &'a [u8],
    first: u16,
}

struct Groups<'a> {
    data: &'a [u8],
    count: usize,
    /// Format 13 stores one glyph id for the whole group. Format 12 adds the offset.
    constant: bool,
}

fn parse_cmap(table: &[u8]) -> Result<Cmap<'_>, FontError> {
    if table.len() < 4 {
        return Err(short(table));
    }
    if u16_at(table, 0)? != 0 {
        return Err(FontError::Unsupported);
    }
    let records = usize::from(u16_at(table, 2)?);
    if records == 0 {
        return Err(FontError::EmptyTable);
    }
    let record_bytes = records.checked_mul(8).ok_or(FontError::BadTable)?;
    let header = bytes_at(table, 4, record_bytes)?;
    let mut chosen: Option<(u8, usize)> = None;
    for index in 0..records {
        let at = index * 8;
        let platform = u16_at(header, at)?;
        let encoding = u16_at(header, at + 2)?;
        if !unicode_cmap(platform, encoding) {
            continue;
        }
        let offset = usize::try_from(u32_at(header, at + 4)?).map_err(|_| FontError::BadTable)?;
        let format = u16_at(table.get(offset..).ok_or(FontError::Truncated)?, 0)?;
        let Some(score) = cmap_score(format) else {
            continue;
        };
        if chosen.as_ref().is_some_and(|(best, _)| *best >= score) {
            continue;
        }
        chosen = Some((score, offset));
    }
    let Some((_, offset)) = chosen else {
        return Err(FontError::Unsupported);
    };
    let subtable = table.get(offset..).ok_or(FontError::Truncated)?;
    if subtable.len() < 2 {
        return Err(FontError::Truncated);
    }
    match u16_at(subtable, 0)? {
        4 => Ok(Cmap::Format4(parse_format4(subtable)?)),
        6 => Ok(Cmap::Format6(parse_format6(subtable)?)),
        12 => Ok(Cmap::Groups(parse_groups(subtable, false)?)),
        13 => Ok(Cmap::Groups(parse_groups(subtable, true)?)),
        _ => Err(FontError::Unsupported),
    }
}

fn unicode_cmap(platform: u16, encoding: u16) -> bool {
    matches!(
        (platform, encoding),
        (0, 0 | 1 | 2 | 3 | 4 | 6) | (3, 1 | 10)
    )
}

fn cmap_score(format: u16) -> Option<u8> {
    match format {
        12 => Some(4),
        13 => Some(3),
        4 => Some(2),
        6 => Some(1),
        _ => None,
    }
}

fn parse_format4(subtable: &[u8]) -> Result<Format4<'_>, FontError> {
    if subtable.len() < 14 {
        return Err(FontError::Truncated);
    }
    let length = usize::from(u16_at(subtable, 2)?);
    let data = bytes_at(subtable, 0, length)?;
    let seg_count_x2 = u16_at(data, 6)?;
    if !seg_count_x2.is_multiple_of(2) {
        return Err(FontError::BadTable);
    }
    let seg_count = usize::from(seg_count_x2 / 2);
    if seg_count == 0 {
        return Err(FontError::EmptyTable);
    }
    let covered = seg_count
        .checked_mul(8)
        .and_then(|arrays| arrays.checked_add(16))
        .ok_or(FontError::BadTable)?;
    if data.len() < covered {
        return Err(FontError::Truncated);
    }
    let end_at = 14;
    let start_at = end_at + seg_count * 2 + 2;
    let delta_at = start_at + seg_count * 2;
    let range_at = delta_at + seg_count * 2;
    let mut previous_end = 0u16;
    for index in 0..seg_count {
        let end = u16_at(data, end_at + index * 2)?;
        if index > 0 && end < previous_end {
            return Err(FontError::BadTable);
        }
        let start = u16_at(data, start_at + index * 2)?;
        if start > end {
            return Err(FontError::BadTable);
        }
        previous_end = end;
    }
    Ok(Format4 {
        data,
        seg_count,
        end_at,
        start_at,
        delta_at,
        range_at,
    })
}

impl Format4<'_> {
    fn glyph(&self, codepoint: u32) -> Result<Option<u16>, FontError> {
        let Ok(code) = u16::try_from(codepoint) else {
            return Ok(None);
        };
        let Some(index) = self.segment(code)? else {
            return Ok(None);
        };
        let start = u16_at(self.data, self.start_at + index * 2)?;
        if code < start {
            return Ok(None);
        }
        let delta = i16_at(self.data, self.delta_at + index * 2)?;
        let range = u16_at(self.data, self.range_at + index * 2)?;
        if range == 0 {
            return Ok(mapped(add_delta(code, delta)));
        }
        let slot = self.range_at + index * 2;
        let within = usize::from(code - start)
            .checked_mul(2)
            .ok_or(FontError::BadTable)?;
        let glyph_at = slot
            .checked_add(usize::from(range))
            .and_then(|at| at.checked_add(within))
            .ok_or(FontError::BadTable)?;
        let glyph = u16_at(self.data, glyph_at)?;
        if glyph == 0 {
            return Ok(None);
        }
        Ok(mapped(add_delta(glyph, delta)))
    }

    fn segment(&self, code: u16) -> Result<Option<usize>, FontError> {
        let mut low = 0;
        let mut high = self.seg_count;
        while low < high {
            let mid = low + (high - low) / 2;
            let end = u16_at(self.data, self.end_at + mid * 2)?;
            if end < code {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        if low == self.seg_count {
            Ok(None)
        } else {
            Ok(Some(low))
        }
    }
}

fn parse_format6(subtable: &[u8]) -> Result<Format6<'_>, FontError> {
    if subtable.len() < 10 {
        return Err(FontError::Truncated);
    }
    let length = usize::from(u16_at(subtable, 2)?);
    let data = bytes_at(subtable, 0, length)?;
    let first = u16_at(data, 6)?;
    let count = usize::from(u16_at(data, 8)?);
    if count == 0 {
        return Err(FontError::EmptyTable);
    }
    let glyphs = bytes_at(data, 10, count.checked_mul(2).ok_or(FontError::BadTable)?)?;
    Ok(Format6 { glyphs, first })
}

impl Format6<'_> {
    fn glyph(&self, codepoint: u32) -> Result<Option<u16>, FontError> {
        let Ok(code) = u16::try_from(codepoint) else {
            return Ok(None);
        };
        let count = self.glyphs.len() / 2;
        let Some(index) = usize::from(code).checked_sub(usize::from(self.first)) else {
            return Ok(None);
        };
        if index >= count {
            return Ok(None);
        }
        Ok(mapped(u16_at(self.glyphs, index * 2)?))
    }
}

fn parse_groups(subtable: &[u8], constant: bool) -> Result<Groups<'_>, FontError> {
    if subtable.len() < 16 {
        return Err(FontError::Truncated);
    }
    let length = usize::try_from(u32_at(subtable, 4)?).map_err(|_| FontError::BadTable)?;
    let data = bytes_at(subtable, 0, length)?;
    let count = usize::try_from(u32_at(data, 12)?).map_err(|_| FontError::BadTable)?;
    if count == 0 {
        return Err(FontError::EmptyTable);
    }
    let bytes = count.checked_mul(12).ok_or(FontError::BadTable)?;
    let groups = bytes_at(data, 16, bytes)?;
    let mut previous_end = 0u32;
    for index in 0..count {
        let start = u32_at(groups, index * 12)?;
        let end = u32_at(groups, index * 12 + 4)?;
        if start > end || (index > 0 && start <= previous_end) {
            return Err(FontError::BadTable);
        }
        previous_end = end;
    }
    Ok(Groups {
        data: groups,
        count,
        constant,
    })
}

impl Groups<'_> {
    fn glyph(&self, codepoint: u32) -> Result<Option<u16>, FontError> {
        let mut low = 0;
        let mut high = self.count;
        while low < high {
            let mid = low + (high - low) / 2;
            let at = mid * 12;
            let start = u32_at(self.data, at)?;
            if codepoint < start {
                high = mid;
                continue;
            }
            let end = u32_at(self.data, at + 4)?;
            if codepoint > end {
                low = mid + 1;
                continue;
            }
            let start_glyph = u32_at(self.data, at + 8)?;
            let glyph = if self.constant {
                start_glyph
            } else {
                start_glyph
                    .checked_add(codepoint - start)
                    .ok_or(FontError::BadTable)?
            };
            let Ok(glyph) = u16::try_from(glyph) else {
                return Err(FontError::BadTable);
            };
            return Ok(mapped(glyph));
        }
        Ok(None)
    }
}

fn mapped(glyph: u16) -> Option<u16> {
    if glyph == 0 { None } else { Some(glyph) }
}

fn add_delta(value: u16, delta: i16) -> u16 {
    let sum = i32::from(value) + i32::from(delta);
    let wrapped = sum.rem_euclid(65_536);
    u16::try_from(wrapped).unwrap_or(0)
}

struct Table {
    tag: [u8; 4],
    offset: u32,
    length: u32,
}

pub(crate) struct Directory<'a> {
    data: &'a [u8],
    kind: FontKind,
    tables: Vec<Table>,
}

impl<'a> Directory<'a> {
    fn parse(bytes: &'a [u8], at: usize) -> Result<Self, FontError> {
        let header = bytes.get(at..).ok_or(FontError::Truncated)?;
        if header.len() < 12 {
            return Err(FontError::Truncated);
        }
        let version = u32_at(header, 0)?;
        let kind = match version {
            0x0001_0000 | 0x7472_7565 => FontKind::TrueType,
            0x4F54_544F => FontKind::OpenType,
            _ => return Err(FontError::Unsupported),
        };
        let count = usize::from(u16_at(header, 4)?);
        if count == 0 {
            return Err(FontError::EmptyTable);
        }
        // A face carries a few dozen tables. A larger count is refused before
        // the directory is allocated.
        if count > 256 {
            return Err(FontError::BadTable);
        }
        let records = bytes_at(header, 12, count * 16)?;
        let mut tables = Vec::with_capacity(count);
        let mut previous: Option<[u8; 4]> = None;
        for index in 0..count {
            let at = index * 16;
            let mut tag = [0; 4];
            tag.copy_from_slice(&records[at..at + 4]);
            if previous.is_some_and(|prior| prior >= tag) {
                return Err(FontError::BadTable);
            }
            previous = Some(tag);
            let offset = u32_at(records, at + 8)?;
            let length = u32_at(records, at + 12)?;
            let end = offset.checked_add(length).ok_or(FontError::BadTable)?;
            if usize::try_from(end).map_err(|_| FontError::BadTable)? > bytes.len() {
                return Err(FontError::Truncated);
            }
            tables.push(Table {
                tag,
                offset,
                length,
            });
        }
        Ok(Self {
            data: bytes,
            kind,
            tables,
        })
    }

    pub(crate) fn kind(&self) -> FontKind {
        self.kind
    }

    pub(crate) fn table(&self, tag: &[u8; 4]) -> Result<&'a [u8], FontError> {
        let mut low = 0;
        let mut high = self.tables.len();
        while low < high {
            let mid = low + (high - low) / 2;
            match self.tables[mid].tag.cmp(tag) {
                Ordering::Less => low = mid + 1,
                Ordering::Greater => high = mid,
                Ordering::Equal => {
                    let record = &self.tables[mid];
                    let offset = usize::try_from(record.offset).map_err(|_| FontError::BadTable)?;
                    let length = usize::try_from(record.length).map_err(|_| FontError::BadTable)?;
                    return bytes_at(self.data, offset, length);
                }
            }
        }
        Err(FontError::MissingTable)
    }
}

pub(crate) fn bytes_at(data: &[u8], offset: usize, len: usize) -> Result<&[u8], FontError> {
    let end = offset.checked_add(len).ok_or(FontError::BadTable)?;
    data.get(offset..end).ok_or(FontError::Truncated)
}

pub(crate) fn u16_at(data: &[u8], offset: usize) -> Result<u16, FontError> {
    let bytes = bytes_at(data, offset, 2)?;
    Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
}

pub(crate) fn i16_at(data: &[u8], offset: usize) -> Result<i16, FontError> {
    let bytes = bytes_at(data, offset, 2)?;
    Ok(i16::from_be_bytes([bytes[0], bytes[1]]))
}

pub(crate) fn u32_at(data: &[u8], offset: usize) -> Result<u32, FontError> {
    let bytes = bytes_at(data, offset, 4)?;
    Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn short(table: &[u8]) -> FontError {
    if table.is_empty() {
        FontError::EmptyTable
    } else {
        FontError::Truncated
    }
}

#[cfg(test)]
mod tests {
    use super::{FontError, FontKind, Segment, read, read_collection, u16_at, u32_at};

    const SAMPLE_TTF: &[u8] = include_bytes!("../fixtures/sample.ttf");
    const SAMPLE_OTF: &[u8] = include_bytes!("../fixtures/sample.otf");
    const SAMPLE_TTC: &[u8] = include_bytes!("../fixtures/sample.ttc");
    const NO_OUTLINE: &[u8] = include_bytes!("../fixtures/no-outline.otf");

    #[test]
    fn advances_for_a_string_come_from_the_fixture() {
        let font = read(SAMPLE_TTF).expect("fixture");
        assert_eq!(font.kind(), FontKind::TrueType);
        assert_eq!(font.units_per_em(), 1000);
        assert_eq!(font.glyph_count(), 10);
        let advances = font.advances("AB").expect("advances");
        assert_eq!(advances.len(), 2);
        assert_eq!(advances[0].character, 'A');
        assert_eq!(advances[0].glyph_id, 2);
        assert_eq!(advances[0].units, 600);
        assert_eq!(advances[1].character, 'B');
        assert_eq!(advances[1].glyph_id, 3);
        assert_eq!(advances[1].units, 700);
        let spaced = font.advances("A B").expect("space");
        assert_eq!(
            spaced.iter().map(|item| item.units).collect::<Vec<_>>(),
            vec![600, 250, 700]
        );
        assert!(font.advances("").expect("empty").is_empty());
    }

    #[test]
    fn a_long_run_returns_one_advance_per_character() {
        let font = read(SAMPLE_TTF).expect("fixture");
        let text = "A".repeat(4_096);
        let advances = font.advances(&text).expect("advances");
        assert_eq!(advances.len(), 4_096);
        assert!(
            advances
                .iter()
                .all(|item| item.units == 600 && item.glyph_id == 2)
        );
    }

    #[test]
    fn a_missing_character_has_no_glyph_id_and_the_font_still_has_glyph_zero() {
        let font = read(SAMPLE_TTF).expect("fixture");
        assert!(font.has_outlines());
        assert_eq!(font.glyph_id('A').expect("map"), Some(2));
        assert_eq!(font.glyph_id('Z').expect("map"), None);
        assert_eq!(font.glyph_id('😀').expect("map"), None);
        assert_eq!(font.missing_glyph(), 0);
        assert_eq!(font.advance(0).expect("notdef"), 500);
        assert_eq!(font.advance(2).expect("A"), 600);
        assert_eq!(font.advance(10), Err(FontError::UnknownGlyph));
        let missing = font.missing_outline().expect("notdef outline");
        assert_eq!(missing, font.outline(0).expect("glyph 0"));
        assert!(missing.contours.is_empty());
        assert_eq!(font.kerning(2, 2), 0);
        assert_eq!(
            font.advances("Z"),
            Err(FontError::MissingGlyph { character: 'Z' })
        );
    }

    #[test]
    fn kerning_pairs_come_from_the_file() {
        let font = read(SAMPLE_TTF).expect("fixture");
        let pair = [0, 2, 0, 3, 0xff, 0xb0];
        assert!(
            SAMPLE_TTF.windows(pair.len()).any(|window| window == pair),
            "the fixture stores the pair"
        );
        assert_eq!(font.kerning(2, 3), -80);
        assert_eq!(font.kerning(3, 2), 40);
        assert_eq!(font.kerning(2, 2), 0);
        assert_eq!(font.advance(2).expect("A"), 600);
        assert_eq!(font.advance(3).expect("B"), 700);

        let mut shifted = SAMPLE_TTF.to_vec();
        let at = shifted
            .windows(pair.len())
            .position(|window| window == pair)
            .expect("pair");
        shifted[at + 4] = 0;
        shifted[at + 5] = 10;
        let changed = read(&shifted).expect("edited pair");
        assert_eq!(changed.kerning(2, 3), 10);

        let mut hostile = SAMPLE_TTF.to_vec();
        let kern_at = table_offset(&hostile, b"kern");
        hostile[kern_at + 10] = 0x01;
        hostile[kern_at + 11] = 0xF4;
        assert!(matches!(read(&hostile), Err(FontError::Truncated)));

        let otf = read(SAMPLE_OTF).expect("otf");
        assert_eq!(otf.kerning(2, 3), -80);
        assert_eq!(otf.kerning(3, 2), 40);
        let collection = read_collection(SAMPLE_TTC, 0).expect("face");
        assert_eq!(collection.kerning(2, 3), 0);
    }

    fn table_offset(bytes: &[u8], tag: &[u8; 4]) -> usize {
        let count = usize::from(u16::from_be_bytes([bytes[4], bytes[5]]));
        for index in 0..count {
            let at = 12 + index * 16;
            if &bytes[at..at + 4] == tag {
                return usize::try_from(u32::from_be_bytes([
                    bytes[at + 8],
                    bytes[at + 9],
                    bytes[at + 10],
                    bytes[at + 11],
                ]))
                .expect("offset");
            }
        }
        panic!("missing {tag:?}");
    }

    #[test]
    fn a_face_without_outlines_still_opens() {
        let font = read(NO_OUTLINE).expect("cff2");
        assert_eq!(font.kind(), FontKind::OpenType);
        assert!(!font.has_outlines());
        assert_eq!(font.glyph_count(), 1);
        assert_eq!(font.advance(font.missing_glyph()).expect("advance"), 500);
        assert_eq!(font.glyph_id('A').expect("map"), None);
        assert_eq!(font.missing_outline(), Err(FontError::Unsupported));
        assert_eq!(font.outline(0), Err(FontError::Unsupported));
    }

    #[test]
    fn a_missing_glyph_is_an_error() {
        let font = read(SAMPLE_TTF).expect("fixture");
        assert_eq!(
            font.advances("Z"),
            Err(FontError::MissingGlyph { character: 'Z' })
        );
        assert_eq!(
            font.advances("C"),
            Err(FontError::MissingGlyph { character: 'C' })
        );
        assert_eq!(
            font.advances("AB!"),
            Err(FontError::MissingGlyph { character: '!' })
        );
        assert_eq!(
            font.advances("😀"),
            Err(FontError::MissingGlyph { character: '😀' })
        );
    }

    #[test]
    fn an_empty_table_directory_is_an_error() {
        let mut bytes = vec![0, 1, 0, 0];
        bytes.extend_from_slice(&0u16.to_be_bytes());
        bytes.extend_from_slice(&0u16.to_be_bytes());
        bytes.extend_from_slice(&0u16.to_be_bytes());
        bytes.extend_from_slice(&0u16.to_be_bytes());
        assert_eq!(read(&bytes).expect_err("empty"), FontError::EmptyTable);
    }

    #[test]
    fn a_required_table_with_no_bytes_is_empty() {
        let bytes = directory_with_head(0, &[]);
        assert_eq!(read(&bytes).expect_err("empty"), FontError::EmptyTable);
    }

    #[test]
    fn a_truncated_file_is_an_error() {
        assert_eq!(
            read(&[0, 1, 0, 0]).expect_err("short"),
            FontError::Truncated
        );
        assert_eq!(read(&[]).expect_err("empty"), FontError::Truncated);
        let mut short = vec![0, 1, 0, 0, 0, 1, 0, 16, 0, 0, 0, 0];
        short.extend_from_slice(b"head");
        assert_eq!(read(&short).expect_err("record"), FontError::Truncated);
        let past_end = directory_with_head(54, &[]);
        assert_eq!(read(&past_end).expect_err("length"), FontError::Truncated);
    }

    #[test]
    fn a_table_offset_that_wraps_is_rejected() {
        let mut bytes = directory_with_head(1, &[0; 32]);
        let length_at = 12 + 12;
        bytes[length_at..length_at + 4].copy_from_slice(&u32::MAX.to_be_bytes());
        let offset_at = 12 + 8;
        bytes[offset_at..offset_at + 4].copy_from_slice(&1u32.to_be_bytes());
        assert_eq!(read(&bytes).expect_err("wrap"), FontError::BadTable);
    }

    #[test]
    fn too_many_tables_are_rejected_without_allocating_them() {
        let mut bytes = vec![0, 1, 0, 0];
        bytes.extend_from_slice(&10_000u16.to_be_bytes());
        bytes.extend_from_slice(&0u16.to_be_bytes());
        bytes.extend_from_slice(&0u16.to_be_bytes());
        bytes.extend_from_slice(&0u16.to_be_bytes());
        assert_eq!(read(&bytes).expect_err("wrap"), FontError::BadTable);
    }

    #[test]
    fn format4_uses_the_glyph_id_array_and_wraps_the_delta() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&4u16.to_be_bytes());
        bytes.extend_from_slice(&36u16.to_be_bytes());
        bytes.extend_from_slice(&0u16.to_be_bytes());
        bytes.extend_from_slice(&4u16.to_be_bytes());
        bytes.extend_from_slice(&4u16.to_be_bytes());
        bytes.extend_from_slice(&1u16.to_be_bytes());
        bytes.extend_from_slice(&0u16.to_be_bytes());
        bytes.extend_from_slice(&0x0042u16.to_be_bytes());
        bytes.extend_from_slice(&0xffffu16.to_be_bytes());
        bytes.extend_from_slice(&0u16.to_be_bytes());
        bytes.extend_from_slice(&0x0041u16.to_be_bytes());
        bytes.extend_from_slice(&0xffffu16.to_be_bytes());
        bytes.extend_from_slice(&0i16.to_be_bytes());
        bytes.extend_from_slice(&1i16.to_be_bytes());
        bytes.extend_from_slice(&4u16.to_be_bytes());
        bytes.extend_from_slice(&0u16.to_be_bytes());
        bytes.extend_from_slice(&2u16.to_be_bytes());
        bytes.extend_from_slice(&3u16.to_be_bytes());
        let table = super::parse_format4(&bytes).expect("format 4");
        assert_eq!(table.glyph(u32::from('A')).expect("A"), Some(2));
        assert_eq!(table.glyph(u32::from('B')).expect("B"), Some(3));
        assert_eq!(table.glyph(u32::from('C')).expect("C"), None);
        assert_eq!(table.glyph(0x40).expect("hole"), None);

        let mut delta = Vec::new();
        delta.extend_from_slice(&4u16.to_be_bytes());
        delta.extend_from_slice(&32u16.to_be_bytes());
        delta.extend_from_slice(&0u16.to_be_bytes());
        delta.extend_from_slice(&4u16.to_be_bytes());
        delta.extend_from_slice(&4u16.to_be_bytes());
        delta.extend_from_slice(&1u16.to_be_bytes());
        delta.extend_from_slice(&0u16.to_be_bytes());
        delta.extend_from_slice(&65u16.to_be_bytes());
        delta.extend_from_slice(&0xffffu16.to_be_bytes());
        delta.extend_from_slice(&0u16.to_be_bytes());
        delta.extend_from_slice(&65u16.to_be_bytes());
        delta.extend_from_slice(&0xffffu16.to_be_bytes());
        delta.extend_from_slice(&(-64i16).to_be_bytes());
        delta.extend_from_slice(&1i16.to_be_bytes());
        delta.extend_from_slice(&0u16.to_be_bytes());
        delta.extend_from_slice(&0u16.to_be_bytes());
        let table = super::parse_format4(&delta).expect("delta");
        assert_eq!(table.glyph(65).expect("delta"), Some(1));
        assert_eq!(table.glyph(0xffff).expect("missing"), None);
    }

    #[test]
    fn the_last_horizontal_metric_covers_later_glyphs() {
        let data = [0x01, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let hmtx = super::Hmtx::parse(&data, 1, 3).expect("hmtx");
        assert_eq!(hmtx.advance(0).expect("first"), 400);
        assert_eq!(hmtx.advance(2).expect("shared"), 400);
        assert!(matches!(
            super::Hmtx::parse(&[], 1, 1),
            Err(FontError::EmptyTable)
        ));
    }

    #[test]
    fn truetype_outlines_keep_lines_quads_and_components() {
        let font = read(SAMPLE_TTF).expect("fixture");
        let letter = font.outline(2).expect("A");
        assert_eq!(letter.contours.len(), 1);
        assert_eq!(letter.contours[0].x, 100.0);
        assert_eq!(letter.contours[0].y, 0.0);
        assert_eq!(
            letter.contours[0].segments,
            vec![
                Segment::Line { x: 500.0, y: 0.0 },
                Segment::Line { x: 500.0, y: 800.0 },
                Segment::Line { x: 100.0, y: 800.0 },
                Segment::Line { x: 100.0, y: 0.0 },
            ]
        );
        assert!(font.outline(1).expect("space").contours.is_empty());

        let curve = font.outline(6).expect("quad");
        assert_eq!(
            curve.contours[0].segments,
            vec![
                Segment::Quad {
                    cx: 100.0,
                    cy: 200.0,
                    x: 200.0,
                    y: 0.0,
                },
                Segment::Line { x: 0.0, y: 0.0 },
            ]
        );

        let bow = font.outline(7).expect("bow");
        assert_eq!(
            bow.contours[0].segments,
            vec![
                Segment::Quad {
                    cx: 0.0,
                    cy: 100.0,
                    x: 50.0,
                    y: 100.0,
                },
                Segment::Quad {
                    cx: 100.0,
                    cy: 100.0,
                    x: 100.0,
                    y: 0.0,
                },
                Segment::Line { x: 0.0, y: 0.0 },
            ]
        );

        let wave = font.outline(8).expect("wave");
        assert_eq!(wave.contours[0].x, 0.0);
        assert_eq!(wave.contours[0].y, 50.0);
        assert_eq!(
            wave.contours[0].segments,
            vec![
                Segment::Quad {
                    cx: 0.0,
                    cy: 100.0,
                    x: 50.0,
                    y: 100.0,
                },
                Segment::Quad {
                    cx: 100.0,
                    cy: 100.0,
                    x: 100.0,
                    y: 50.0,
                },
                Segment::Quad {
                    cx: 100.0,
                    cy: 0.0,
                    x: 50.0,
                    y: 0.0,
                },
                Segment::Quad {
                    cx: 0.0,
                    cy: 0.0,
                    x: 0.0,
                    y: 50.0,
                },
            ]
        );

        let shifted = font.outline(4).expect("component");
        assert_eq!(shifted.contours[0].x, 110.0);
        assert_eq!(shifted.contours[0].y, 20.0);
        assert_eq!(
            shifted.contours[0].segments[0],
            Segment::Line { x: 510.0, y: 20.0 }
        );

        let half = font.outline(5).expect("scaled");
        assert_eq!(half.contours[0].x, 50.0);
        assert_eq!(half.contours[0].y, 0.0);
        assert_eq!(
            half.contours[0].segments[1],
            Segment::Line { x: 250.0, y: 400.0 }
        );
        let matched = font.outline(9).expect("matched component");
        assert_eq!(matched.contours.len(), 2);
        assert_eq!(matched.contours[1].x, -300.0);
        assert_eq!(matched.contours[1].y, 0.0);
        assert_eq!(font.outline(10), Err(FontError::UnknownGlyph));
    }

    #[test]
    fn opentype_advances_and_cubic_outlines_come_from_the_fixture() {
        let font = read(SAMPLE_OTF).expect("fixture");
        assert_eq!(font.kind(), FontKind::OpenType);
        let advances = font.advances("AB").expect("advances");
        assert_eq!(advances[0].units, 640);
        assert_eq!(advances[0].glyph_id, 2);
        assert_eq!(advances[1].units, 740);
        assert_eq!(advances[1].glyph_id, 3);
        assert_eq!(
            font.advances("Q"),
            Err(FontError::MissingGlyph { character: 'Q' })
        );
        let letter = font.outline(2).expect("A");
        assert_eq!(letter.contours.len(), 1);
        assert_eq!(letter.contours[0].x, 100.0);
        assert_eq!(letter.contours[0].y, 0.0);
        assert_eq!(
            letter.contours[0].segments,
            vec![
                Segment::Line { x: 500.0, y: 0.0 },
                Segment::Line { x: 500.0, y: 800.0 },
                Segment::Line { x: 100.0, y: 800.0 },
                Segment::Line { x: 100.0, y: 0.0 },
            ]
        );
        assert!(font.outline(1).expect("space").contours.is_empty());
    }

    #[test]
    fn a_collection_index_resolves_one_face() {
        assert_eq!(
            read(SAMPLE_TTC).expect_err("collection"),
            FontError::Collection
        );
        let first = read_collection(SAMPLE_TTC, 0).expect("face 0");
        let second = read_collection(SAMPLE_TTC, 1).expect("face 1");
        assert_eq!(first.advances("A").expect("A")[0].units, 600);
        assert_eq!(second.advances("A").expect("A")[0].units, 800);
        assert_eq!(second.advances("B").expect("B")[0].units, 900);
        assert_eq!(
            read_collection(SAMPLE_TTC, 2).expect_err("index"),
            FontError::CollectionIndex { index: 2, count: 2 }
        );
        assert_eq!(
            read_collection(SAMPLE_TTF, 0).expect_err("not a collection"),
            FontError::BadTable
        );
        assert_eq!(
            read_collection(b"ttcf", 0).expect_err("short"),
            FontError::Truncated
        );
    }

    fn directory_with_head(length: u32, body: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0, 1, 0, 0, 0, 1, 0, 16, 0, 0, 0, 0];
        bytes.extend_from_slice(b"head");
        bytes.extend_from_slice(&0u32.to_be_bytes());
        let offset = u32::try_from(bytes.len() + 8).expect("offset");
        bytes.extend_from_slice(&offset.to_be_bytes());
        bytes.extend_from_slice(&length.to_be_bytes());
        bytes.extend_from_slice(body);
        let _ = (u16_at(&bytes, 0), u32_at(&bytes, 0));
        bytes
    }
}

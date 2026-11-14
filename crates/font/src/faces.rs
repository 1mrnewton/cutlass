//! The faces in a font file, and one face of a collection as its own file.
//!
//! [`list_faces`] reads through a callback, so a caller with a large
//! collection on disk reads only the headers, the table directories and the
//! `name` tables, not the outlines. [`single_face`] copies the tables of one
//! collection face into a standalone font that [`crate::read`] opens.

use crate::{Directory, FontError, Tag, bytes_at, collection, u16_at, u32_at};

/// The most faces a collection may list.
pub const MAX_COLLECTION_FACES: u32 = 256;
/// The largest `name` table [`list_faces`] reads, in bytes.
const MAX_NAME_TABLE: u32 = 256 * 1024;
/// The longest full name kept, in characters.
const MAX_NAME_CHARS: usize = 64;
/// The most tables a face directory may hold; [`Directory`] refuses more.
const MAX_TABLES: usize = 256;

/// One face in a font file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaceEntry {
    /// `None` for a single-face file, the face number in a collection.
    pub index: Option<u32>,
    /// The face's full name from its `name` table, when it has a readable one.
    pub full_name: Option<String>,
    /// The family the face belongs to: a regular face and its bold and
    /// italic share one.
    pub family: Option<String>,
    /// The family as its maker thinks of it, when the face names one
    /// apart from `family`: a face of many weights may give each weight
    /// a family of its own there, "Montserrat Thin", and the whole
    /// design's name here, "Montserrat".
    pub design_family: Option<String>,
    /// The face is the family's bold, or bold italic.
    pub bold: bool,
    /// The face is the family's italic, or bold italic.
    pub italic: bool,
    /// The instances a variable face names, such as Bold: each can be
    /// loaded as a face of its own with [`load_instance`]. Empty for a
    /// face that does not vary.
    pub instances: Vec<InstanceEntry>,
}

/// One named instance of a variable face.
#[derive(Clone, Debug, PartialEq)]
pub struct InstanceEntry {
    /// The instance's name, such as "Condensed Bold".
    pub name: String,
    /// The value of each of the face's axes, in the face's order, to
    /// hand to [`load_instance`], [`pin_instance`] or
    /// [`Font::set_variations`](crate::Font::set_variations).
    pub coordinates: Vec<(Tag, f32)>,
    /// The instance is the bold of the face's instances that are italic
    /// as it is: of each kind exactly one is, the one of weight 700 or,
    /// with none at 700, the nearest above 600. In a face with no axis
    /// of weight, whether the face itself is bold.
    pub bold: bool,
    /// The face is italic, or the instance is set italic or slanted.
    pub italic: bool,
    /// Every axis is at its default: the instance is the face as it is
    /// read with no setting.
    pub default: bool,
}

// A value of an axis is a 16.16 fixed-point number read from the file, so
// it is always a number and equal to itself.
impl Eq for InstanceEntry {}

/// The tag of the table that says which instance a face's bytes are: not
/// an OpenType table, but one this crate writes with [`pin_instance`] and
/// reads when it opens a face.
const PIN: &[u8; 4] = b"Cpin";
/// The largest `fvar` table [`list_faces`] reads, in bytes.
const MAX_FVAR_TABLE: u32 = 256 * 1024;
/// The most instances listed for one face.
const MAX_LISTED_INSTANCES: usize = 64;
/// The most instance records looked through for those.
const MAX_INSTANCE_RECORDS: usize = 256;

/// The faces in a font file. `read_at(offset, len)` returns exactly `len`
/// bytes at `offset`, or `None` when the file is shorter.
pub fn list_faces(
    mut read_at: impl FnMut(u64, usize) -> Option<Vec<u8>>,
) -> Result<Vec<FaceEntry>, FontError> {
    let mut read = |offset: u64, len: usize| {
        read_at(offset, len)
            .filter(|bytes| bytes.len() == len)
            .ok_or(FontError::Truncated)
    };
    let header = read(0, 12)?;
    if &header[..4] != b"ttcf" {
        sfnt_version(&header)?;
        return Ok(vec![entry_at(&mut read, 0, &header, None)?]);
    }
    let count = u32_at(&header, 8)?;
    if count == 0 {
        return Err(FontError::EmptyTable);
    }
    if count > MAX_COLLECTION_FACES {
        return Err(FontError::BadTable);
    }
    let offsets = read(12, count as usize * 4)?;
    let mut faces = Vec::with_capacity(count as usize);
    for index in 0..count {
        let offset = u64::from(u32_at(&offsets, index as usize * 4)?);
        let header = read(offset, 12)?;
        sfnt_version(&header)?;
        faces.push(entry_at(&mut read, offset, &header, Some(index))?);
    }
    Ok(faces)
}

/// The largest emoji strike a loaded face keeps, in pixels per em: the
/// largest picture a glyph atlas holds.
const MAX_STRIKE_PPEM: u16 = 256;
/// An `sbix` table under this many bytes is kept whole.
const SMALL_SBIX: usize = 8 * 1024 * 1024;

/// One face of a font file as a standalone font, read a table at a time.
///
/// `read_at(offset, len)` returns exactly `len` bytes at `offset`. `index`
/// is the face of a collection, and `None` for a file that is one face.
/// Only that face's tables are read, so one face of a large collection
/// costs what it holds and not the whole file. A large `sbix` table, which
/// holds every emoji at many sizes, is cut to one size: the largest, up to
/// 256 pixels per em, that `max_bytes` has room for. A face that still
/// comes to more than `max_bytes` is refused as unsupported.
pub fn load_face(
    mut read_at: impl FnMut(u64, usize) -> Option<Vec<u8>>,
    index: Option<u32>,
    max_bytes: usize,
) -> Result<Vec<u8>, FontError> {
    let mut read = |offset: u64, len: usize| {
        read_at(offset, len)
            .filter(|bytes| bytes.len() == len)
            .ok_or(FontError::Truncated)
    };
    let header = read(0, 12)?;
    let start = match (&header[..4] == b"ttcf", index) {
        (false, None) => 0,
        (false, Some(_)) => return Err(FontError::BadTable),
        (true, None) => return Err(FontError::Collection),
        (true, Some(index)) => {
            let count = u32_at(&header, 8)?;
            if index >= count {
                return Err(FontError::CollectionIndex { index, count });
            }
            if count > MAX_COLLECTION_FACES {
                return Err(FontError::BadTable);
            }
            u64::from(u32_at(&read(12 + u64::from(index) * 4, 4)?, 0)?)
        }
    };
    let header = read(start, 12)?;
    sfnt_version(&header)?;
    let version = u32_at(&header, 0)?;
    let count = usize::from(u16_at(&header, 4)?);
    if count == 0 {
        return Err(FontError::EmptyTable);
    }
    if count > MAX_TABLES {
        return Err(FontError::BadTable);
    }
    let records = read(start + 12, count * 16)?;
    let mut tables = Vec::with_capacity(count);
    let mut total = 12 + count * 16;
    for slot in 0..count {
        let at = slot * 16;
        let mut tag = [0; 4];
        tag.copy_from_slice(&records[at..at + 4]);
        let checksum = u32_at(&records, at + 4)?;
        let offset = u64::from(u32_at(&records, at + 8)?);
        let length =
            usize::try_from(u32_at(&records, at + 12)?).map_err(|_| FontError::BadTable)?;
        let room = max_bytes.saturating_sub(total);
        let data = if &tag == b"sbix" && length > SMALL_SBIX {
            one_strike(&mut read, offset, length, room)?
        } else if length > room {
            return Err(FontError::Unsupported);
        } else {
            read(offset, length)?
        };
        total += data.len().next_multiple_of(4);
        tables.push((tag, checksum, data));
    }
    let out = assemble(version, &tables)?;
    crate::read(&out)?;
    Ok(out)
}

/// An `sbix` table holding one strike of the table at `offset`: the
/// largest up to [`MAX_STRIKE_PPEM`] that fits in `room`, or the smallest
/// of all when every strike is larger than that.
fn one_strike(
    read: &mut impl FnMut(u64, usize) -> Result<Vec<u8>, FontError>,
    offset: u64,
    length: usize,
    room: usize,
) -> Result<Vec<u8>, FontError> {
    let header = read(offset, 8)?;
    let strikes = usize::try_from(u32_at(&header, 4)?).map_err(|_| FontError::BadTable)?;
    if strikes == 0 || strikes > MAX_STRIKES {
        return Err(FontError::BadTable);
    }
    let offsets = read(offset + 8, strikes * 4)?;
    // Each strike: its pixels per em, where it starts, and where it ends,
    // which is where the next one starts.
    let mut found: Vec<(u16, usize)> = Vec::with_capacity(strikes);
    for slot in 0..strikes {
        let at = usize::try_from(u32_at(&offsets, slot * 4)?).map_err(|_| FontError::BadTable)?;
        if at >= length {
            return Err(FontError::BadTable);
        }
        let ppem = u16_at(&read(offset + at as u64, 2)?, 0)?;
        found.push((ppem, at));
    }
    // Where each strike ends: where the next one starts.
    let size_of = |at: usize| {
        let end = found
            .iter()
            .map(|(_, other)| *other)
            .filter(|other| *other > at)
            .min()
            .unwrap_or(length);
        end - at
    };
    let fits = |at: usize| size_of(at) + 12 <= room;
    let (_, at) = found
        .iter()
        .copied()
        .filter(|(ppem, at)| *ppem <= MAX_STRIKE_PPEM && fits(*at))
        .max_by_key(|(ppem, _)| *ppem)
        .or_else(|| {
            found
                .iter()
                .copied()
                .min_by_key(|(ppem, _)| *ppem)
                .filter(|(_, at)| fits(*at))
        })
        .ok_or(FontError::Unsupported)?;
    let size = size_of(at);
    let mut table = Vec::with_capacity(size + 12);
    table.extend_from_slice(&header[..4]);
    table.extend_from_slice(&1_u32.to_be_bytes());
    table.extend_from_slice(&12_u32.to_be_bytes());
    table.extend(read(offset + at as u64, size)?);
    Ok(table)
}

/// The most strikes an `sbix` table is searched through.
const MAX_STRIKES: usize = 64;

/// A font file of `tables`: each a tag, its checksum, and its bytes.
pub(crate) fn assemble(
    version: u32,
    tables: &[([u8; 4], u32, Vec<u8>)],
) -> Result<Vec<u8>, FontError> {
    let count = tables.len();
    let count_u16 = u16::try_from(count).map_err(|_| FontError::BadTable)?;
    if count == 0 {
        return Err(FontError::EmptyTable);
    }
    let selector = count.ilog2();
    let search = 16_u16 << selector;
    let shift = count_u16 * 16 - search;
    let mut out = Vec::new();
    out.extend_from_slice(&version.to_be_bytes());
    out.extend_from_slice(&count_u16.to_be_bytes());
    out.extend_from_slice(&search.to_be_bytes());
    out.extend_from_slice(&u16::try_from(selector).unwrap_or(0).to_be_bytes());
    out.extend_from_slice(&shift.to_be_bytes());
    let records = out.len();
    out.resize(records + count * 16, 0);
    for (slot, (tag, checksum, data)) in tables.iter().enumerate() {
        while out.len() % 4 != 0 {
            out.push(0);
        }
        let offset = u32::try_from(out.len()).map_err(|_| FontError::BadTable)?;
        let length = u32::try_from(data.len()).map_err(|_| FontError::BadTable)?;
        out.extend_from_slice(data);
        let record = records + slot * 16;
        out[record..record + 4].copy_from_slice(tag);
        out[record + 4..record + 8].copy_from_slice(&checksum.to_be_bytes());
        out[record + 8..record + 12].copy_from_slice(&offset.to_be_bytes());
        out[record + 12..record + 16].copy_from_slice(&length.to_be_bytes());
    }
    while out.len() % 4 != 0 {
        out.push(0);
    }
    Ok(out)
}

/// Face `index` of a collection as a standalone font file.
pub fn single_face(bytes: &[u8], index: u32) -> Result<Vec<u8>, FontError> {
    let at = collection::face_offset(bytes, index)?;
    let directory = Directory::parse(bytes, at)?;
    let version = u32_at(bytes, at)?;
    let count = directory.tables.len();
    let count_u16 = u16::try_from(count).map_err(|_| FontError::BadTable)?;
    let selector = count.ilog2();
    let search = 16_u16 << selector;
    let shift = count_u16 * 16 - search;
    let mut out = Vec::new();
    out.extend_from_slice(&version.to_be_bytes());
    out.extend_from_slice(&count_u16.to_be_bytes());
    out.extend_from_slice(&search.to_be_bytes());
    out.extend_from_slice(&u16::try_from(selector).unwrap_or(0).to_be_bytes());
    out.extend_from_slice(&shift.to_be_bytes());
    let records = out.len();
    out.resize(records + count * 16, 0);
    for (slot, table) in directory.tables.iter().enumerate() {
        while out.len() % 4 != 0 {
            out.push(0);
        }
        let offset = u32::try_from(out.len()).map_err(|_| FontError::BadTable)?;
        let length = usize::try_from(table.length).map_err(|_| FontError::BadTable)?;
        let start = usize::try_from(table.offset).map_err(|_| FontError::BadTable)?;
        out.extend_from_slice(bytes_at(bytes, start, length)?);
        let checksum = u32_at(bytes, at + 12 + slot * 16 + 4)?;
        let record = records + slot * 16;
        out[record..record + 4].copy_from_slice(&table.tag);
        out[record + 4..record + 8].copy_from_slice(&checksum.to_be_bytes());
        out[record + 8..record + 12].copy_from_slice(&offset.to_be_bytes());
        out[record + 12..record + 16].copy_from_slice(&table.length.to_be_bytes());
    }
    while out.len() % 4 != 0 {
        out.push(0);
    }
    crate::read(&out)?;
    Ok(out)
}

fn sfnt_version(header: &[u8]) -> Result<(), FontError> {
    match u32_at(header, 0)? {
        0x0001_0000 | 0x7472_7565 | 0x4F54_544F => Ok(()),
        _ => Err(FontError::Unsupported),
    }
}

/// The names and style of the face whose directory starts at `offset`. A
/// missing or unreadable `name` table gives no names, and a face that
/// does not say is neither bold nor italic; a broken directory is an
/// error.
fn entry_at(
    read: &mut impl FnMut(u64, usize) -> Result<Vec<u8>, FontError>,
    offset: u64,
    header: &[u8],
    index: Option<u32>,
) -> Result<FaceEntry, FontError> {
    let count = usize::from(u16_at(header, 4)?);
    if count == 0 {
        return Err(FontError::EmptyTable);
    }
    if count > MAX_TABLES {
        return Err(FontError::BadTable);
    }
    let records = read(offset + 12, count * 16)?;
    let mut entry = FaceEntry {
        index,
        full_name: None,
        family: None,
        design_family: None,
        bold: false,
        italic: false,
        instances: Vec::new(),
    };
    // The style a face says it is in `OS/2` wins over the older `head`.
    let mut said = false;
    let (mut names, mut axes) = (None, None);
    for slot in 0..count {
        let at = slot * 16;
        let start = u64::from(u32_at(&records, at + 8)?);
        let length = u32_at(&records, at + 12)?;
        match &records[at..at + 4] {
            b"name" if length <= MAX_NAME_TABLE => {
                if let Ok(table) = read(start, length as usize) {
                    entry.full_name = name(&table, 4);
                    entry.family = name(&table, 1);
                    entry.design_family = name(&table, 16);
                    names = Some(table);
                }
            }
            b"fvar" if length <= MAX_FVAR_TABLE => {
                axes = read(start, length as usize).ok();
            }
            b"OS/2" if length >= 64 => {
                if let Ok(selection) = read(start + 62, 2).and_then(|bytes| u16_at(&bytes, 0)) {
                    entry.italic = selection & 0x01 != 0;
                    entry.bold = selection & 0x20 != 0;
                    said = true;
                }
            }
            b"head" if length >= 46 && !said => {
                if let Ok(style) = read(start + 44, 2).and_then(|bytes| u16_at(&bytes, 0)) {
                    entry.bold = style & 0x01 != 0;
                    entry.italic = style & 0x02 != 0;
                }
            }
            _ => {}
        }
    }
    if let Some(axes) = axes {
        entry.instances = instances(&axes, names.as_deref(), entry.bold, entry.italic);
    }
    Ok(entry)
}

/// The named instances of the `fvar` table `axes`, named from the `name`
/// table `names`, in a face that is itself `bold` and `italic` or not. A
/// table that does not read, and an instance with no readable name, is
/// left out.
fn instances(axes: &[u8], names: Option<&[u8]>, bold: bool, italic: bool) -> Vec<InstanceEntry> {
    let (Ok(Some(table)), Some(names)) = (crate::vary::Fvar::read(axes), names) else {
        return Vec::new();
    };
    let axes = table.axes(None);
    let value = |coordinates: &[(Tag, f32)], tag: &[u8; 4]| {
        coordinates
            .iter()
            .find(|(axis, _)| axis == tag)
            .map(|(_, value)| *value)
    };
    let mut listed = Vec::new();
    for (name_id, _, values) in table.records().take(MAX_INSTANCE_RECORDS) {
        if listed.len() >= MAX_LISTED_INSTANCES {
            break;
        }
        let Some(name) = Some(name_id)
            .filter(|id| *id != 0xFFFF)
            .and_then(|id| name(names, id))
        else {
            continue;
        };
        let coordinates: Vec<(Tag, f32)> = axes.iter().map(|axis| axis.tag).zip(values).collect();
        let slanted = value(&coordinates, b"ital").is_some_and(|ital| ital >= 0.5)
            || value(&coordinates, b"slnt").is_some_and(|slant| slant <= -1.0);
        listed.push(InstanceEntry {
            name,
            default: axes
                .iter()
                .zip(&coordinates)
                .all(|(axis, (_, value))| value - axis.default == 0.0),
            italic: italic || slanted,
            // In a face with no axis of weight every instance is as bold
            // as the face.
            bold: bold && value(&coordinates, b"wght").is_none(),
            coordinates,
        });
    }
    if axes.iter().any(|axis| &axis.tag == b"wght") {
        // How far an instance is from the weight of bold, and then from
        // the default of every other axis, as a part of the axis.
        let distance = |entry: &InstanceEntry| {
            let mut weight = f32::MAX;
            let mut others = 0.0;
            for (axis, (tag, value)) in axes.iter().zip(&entry.coordinates) {
                if tag == b"wght" {
                    weight = (value - 700.0).abs();
                } else if axis.max > axis.min {
                    others += (value - axis.default).abs() / (axis.max - axis.min);
                }
            }
            (weight, others)
        };
        for kind in [false, true] {
            let chosen = listed
                .iter()
                .enumerate()
                .filter(|(_, entry)| {
                    entry.italic == kind
                        && value(&entry.coordinates, b"wght").is_some_and(|weight| weight >= 600.0)
                })
                .min_by(|(_, one), (_, other)| {
                    distance(one)
                        .partial_cmp(&distance(other))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(index, _)| index);
            if let Some(index) = chosen {
                listed[index].bold = true;
            }
        }
    }
    listed
}

/// The settings a `Cpin` table holds: an axis and a value for each. A
/// table of another version, or one that says it holds more than it
/// does, holds nothing.
pub(crate) fn pinned(table: &[u8]) -> Vec<(Tag, f32)> {
    let read = || {
        if u16_at(table, 0).ok()? != 1 {
            return None;
        }
        let count = usize::from(u16_at(table, 2).ok()?);
        if count > crate::vary::MAX_AXES {
            return None;
        }
        let records = bytes_at(table, 4, count * 8).ok()?;
        Some(
            records
                .as_chunks::<8>()
                .0
                .iter()
                .map(|record| {
                    let raw = i32::from_be_bytes([record[4], record[5], record[6], record[7]]);
                    #[allow(clippy::cast_precision_loss)]
                    let value = raw as f32 / 65_536.0;
                    ([record[0], record[1], record[2], record[3]], value)
                })
                .collect(),
        )
    };
    read().unwrap_or_default()
}

/// A `Cpin` table of `coordinates`: a version, a count, and for each an
/// axis and a 16.16 fixed-point value. A value that is no number is left
/// out, and one too large to store is the largest that can be.
fn pin_table(coordinates: &[(Tag, f32)]) -> Vec<u8> {
    let kept: Vec<&(Tag, f32)> = coordinates
        .iter()
        .filter(|(_, value)| value.is_finite())
        .take(crate::vary::MAX_AXES)
        .collect();
    let mut table = vec![0, 1];
    table.extend(u16::try_from(kept.len()).unwrap_or(0).to_be_bytes());
    for (tag, value) in kept {
        table.extend(tag);
        #[allow(clippy::cast_possible_truncation)]
        let raw = (f64::from(*value) * 65_536.0)
            .round()
            .clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32;
        table.extend(raw.to_be_bytes());
    }
    table
}

/// The sum a table's directory record carries for it: its bytes as
/// 32-bit numbers, the last padded with zeros.
fn checksum(table: &[u8]) -> u32 {
    let (words, rest) = table.as_chunks::<4>();
    let mut sum = words.iter().fold(0_u32, |sum, word| {
        sum.wrapping_add(u32::from_be_bytes(*word))
    });
    if !rest.is_empty() {
        let mut last = [0; 4];
        last[..rest.len()].copy_from_slice(rest);
        sum = sum.wrapping_add(u32::from_be_bytes(last));
    }
    sum
}

/// The single face `bytes`, a variable one, as the instance at
/// `coordinates`: the same font with a table added that says where its
/// axes are set, so that [`crate::read`] opens it already set there, as
/// [`Font::varied`](crate::Font::varied) would set it. The coordinates
/// are an axis's tag and a value in the axis's own units, as
/// [`InstanceEntry::coordinates`] has them; an axis not named is at its
/// default. A face pinned before is pinned anew. A collection is refused:
/// take one face of it first, with [`single_face`] or [`load_instance`].
pub fn pin_instance(bytes: &[u8], coordinates: &[(Tag, f32)]) -> Result<Vec<u8>, FontError> {
    if bytes.starts_with(b"ttcf") {
        return Err(FontError::Collection);
    }
    let directory = Directory::parse(bytes, 0)?;
    let version = u32_at(bytes, 0)?;
    let mut tables = Vec::with_capacity(directory.tables.len() + 1);
    for (slot, table) in directory.tables.iter().enumerate() {
        if &table.tag == PIN {
            continue;
        }
        let start = usize::try_from(table.offset).map_err(|_| FontError::BadTable)?;
        let length = usize::try_from(table.length).map_err(|_| FontError::BadTable)?;
        tables.push((
            table.tag,
            u32_at(bytes, 12 + slot * 16 + 4)?,
            bytes_at(bytes, start, length)?.to_vec(),
        ));
    }
    let pin = pin_table(coordinates);
    tables.push((*PIN, checksum(&pin), pin));
    // A directory lists its tables in the order of their tags.
    tables.sort_by_key(|table| table.0);
    let out = assemble(version, &tables)?;
    crate::read(&out)?;
    Ok(out)
}

/// One face of a font file as a standalone font pinned to the instance at
/// `coordinates`: [`load_face`], then [`pin_instance`]. The face and its
/// pin together come to at most `max_bytes`.
pub fn load_instance(
    read_at: impl FnMut(u64, usize) -> Option<Vec<u8>>,
    index: Option<u32>,
    coordinates: &[(Tag, f32)],
    max_bytes: usize,
) -> Result<Vec<u8>, FontError> {
    // The pin's record in the directory, and the pin, padded.
    let pin = 16 + (4 + coordinates.len().min(crate::vary::MAX_AXES) * 8).next_multiple_of(4);
    let face = load_face(read_at, index, max_bytes.saturating_sub(pin))?;
    pin_instance(&face, coordinates)
}

/// One name from a `name` table: English Windows or Unicode first, then
/// a plain-ASCII Macintosh one in English, then a Windows entry in any
/// other language. A face may name itself in several, and no English
/// among its Windows names.
pub(crate) fn name(table: &[u8], id: u16) -> Option<String> {
    let count = usize::from(u16_at(table, 2).ok()?);
    let storage = usize::from(u16_at(table, 4).ok()?);
    let mut best: Option<(u8, String)> = None;
    for slot in 0..count {
        let record = bytes_at(table, 6 + slot * 12, 12).ok()?;
        let platform = u16_at(record, 0).ok()?;
        let encoding = u16_at(record, 2).ok()?;
        let language = u16_at(record, 4).ok()?;
        if u16_at(record, 6).ok()? != id {
            continue;
        }
        let length = usize::from(u16_at(record, 8).ok()?);
        let offset = usize::from(u16_at(record, 10).ok()?);
        let Ok(raw) = bytes_at(table, storage + offset, length) else {
            continue;
        };
        let (rank, text) = match (platform, encoding) {
            (3, 1 | 10) | (0, _) => {
                let rank = if platform == 0 || language == 0x0409 {
                    0
                } else {
                    2
                };
                (rank, utf16(raw)?)
            }
            (1, 0) if language == 0 && raw.is_ascii() => {
                (1, String::from_utf8_lossy(raw).into_owned())
            }
            _ => continue,
        };
        let text: String = text
            .trim()
            .chars()
            .filter(|character| !character.is_control())
            .take(MAX_NAME_CHARS)
            .collect();
        if text.is_empty() {
            continue;
        }
        if best.as_ref().is_none_or(|(have, _)| rank < *have) {
            best = Some((rank, text));
        }
    }
    best.map(|(_, text)| text)
}

fn utf16(raw: &[u8]) -> Option<String> {
    let (pairs, rest) = raw.as_chunks::<2>();
    if !rest.is_empty() {
        return None;
    }
    let units = pairs.iter().map(|pair| u16::from_be_bytes(*pair));
    char::decode_utf16(units)
        .collect::<Result<String, _>>()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read, read_collection};

    const SAMPLE_TTF: &[u8] = include_bytes!("../fixtures/sample.ttf");
    const SAMPLE_TTC: &[u8] = include_bytes!("../fixtures/sample.ttc");

    fn from(bytes: &[u8]) -> impl FnMut(u64, usize) -> Option<Vec<u8>> + '_ {
        move |offset, len| {
            let start = usize::try_from(offset).ok()?;
            bytes
                .get(start..start.checked_add(len)?)
                .map(<[u8]>::to_vec)
        }
    }

    #[test]
    fn a_single_file_lists_one_face_with_its_name() {
        let faces = list_faces(from(SAMPLE_TTF)).expect("faces");
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].index, None);
        assert!(
            faces[0]
                .full_name
                .as_deref()
                .is_some_and(|name| !name.is_empty())
        );
    }

    /// A `name` table of Windows English names.
    fn names(entries: &[(u16, &str)]) -> Vec<u8> {
        let count = u16::try_from(entries.len()).expect("count");
        let mut table = crate::fixture::be(&[0, count, 6 + count * 12]);
        let mut strings = Vec::new();
        for (id, text) in entries {
            let encoded: Vec<u8> = text.encode_utf16().flat_map(u16::to_be_bytes).collect();
            table.extend(crate::fixture::be(&[
                3,
                1,
                0x0409,
                *id,
                u16::try_from(encoded.len()).expect("length"),
                u16::try_from(strings.len()).expect("offset"),
            ]));
            strings.extend(encoded);
        }
        table.extend(strings);
        table
    }

    #[test]
    fn a_face_is_loaded_a_table_at_a_time() {
        // A single face reads back as itself, table for table.
        let whole = load_face(from(SAMPLE_TTF), None, 1 << 20).expect("single");
        let face = read(&whole).expect("font");
        assert_eq!(face.advance(2).expect("A"), 600);
        // A face of a collection is the same font `single_face` copies out.
        let second = load_face(from(SAMPLE_TTC), Some(1), 1 << 20).expect("second");
        assert_eq!(second, single_face(SAMPLE_TTC, 1).expect("copied"));
        // Only that face's tables are read: nothing past them.
        let mut reads = 0;
        let _first = load_face(
            |offset, len| {
                reads += 1;
                from(SAMPLE_TTC)(offset, len)
            },
            Some(0),
            1 << 20,
        )
        .expect("first");
        assert!(reads < 40, "{reads}");
        // The wrong kind of index, a face past the end, a face too large.
        assert_eq!(
            load_face(from(SAMPLE_TTC), None, 1 << 20),
            Err(FontError::Collection)
        );
        assert_eq!(
            load_face(from(SAMPLE_TTF), Some(0), 1 << 20),
            Err(FontError::BadTable)
        );
        assert_eq!(
            load_face(from(SAMPLE_TTC), Some(2), 1 << 20),
            Err(FontError::CollectionIndex { index: 2, count: 2 })
        );
        assert_eq!(
            load_face(from(SAMPLE_TTF), None, 64),
            Err(FontError::Unsupported)
        );
        assert_eq!(
            load_face(from(&[]), None, 1 << 20),
            Err(FontError::Truncated)
        );
    }

    #[test]
    fn a_large_emoji_table_is_cut_to_one_size() {
        // An `sbix` table of four strikes, larger as their pictures are,
        // and together past the size under which a table is kept whole.
        const UNIT: usize = SMALL_SBIX / 8;
        let strike = |ppem: u16, units: usize, fill: u8| {
            let mut bytes = crate::fixture::be(&[ppem, 72]);
            bytes.resize(units * UNIT, fill);
            bytes
        };
        let strikes = [
            strike(64, 1, 1),
            strike(96, 2, 2),
            strike(160, 4, 3),
            strike(300, 6, 4),
        ];
        let mut sbix = crate::fixture::be(&[1, 1, 0, 4]);
        let mut at = 8 + 16;
        for bytes in &strikes {
            sbix.extend(u32::try_from(at).expect("offset").to_be_bytes());
            at += bytes.len();
        }
        for bytes in &strikes {
            sbix.extend(bytes);
        }
        assert!(sbix.len() > SMALL_SBIX);
        let glyphs = [crate::fixture::boxed(500); 3];
        let file = crate::fixture::font(&glyphs, &[('A', 2)], &[(b"sbix", sbix)]);
        // The strike kept with `room` bytes for the face: its size in
        // pixels, and the table is that strike alone.
        let kept = |room: usize| {
            let loaded = load_face(from(&file), None, room)?;
            let table = table_of(&loaded, b"sbix");
            assert_eq!(&table[4..12], &[0, 0, 0, 1, 0, 0, 0, 12]);
            let ppem = u16::from_be_bytes([table[12], table[13]]);
            let (units, fill) = match ppem {
                64 => (1, 1),
                96 => (2, 2),
                160 => (4, 3),
                _ => panic!("a strike of {ppem}"),
            };
            assert_eq!(table.len(), 12 + units * UNIT);
            assert!(table[16..].iter().all(|byte| *byte == fill));
            Ok::<u16, FontError>(ppem)
        };
        // With room for all of them the sharpest an atlas holds is kept:
        // the strike of 300 is past that.
        assert_eq!(kept(64 << 20), Ok(160));
        // With less room, the sharpest that fits.
        assert_eq!(kept(3 * UNIT), Ok(96));
        assert_eq!(kept(3 * UNIT / 2), Ok(64));
        // With no room for even one strike the face is refused.
        assert_eq!(kept(UNIT / 2), Err(FontError::Unsupported));
    }

    #[test]
    fn strikes_all_past_the_atlas_keep_the_smallest() {
        const UNIT: usize = SMALL_SBIX;
        let mut sbix = crate::fixture::be(&[1, 1, 0, 2]);
        sbix.extend(16_u32.to_be_bytes());
        sbix.extend(u32::try_from(16 + UNIT).expect("offset").to_be_bytes());
        for ppem in [400_u16, 300] {
            let mut strike = crate::fixture::be(&[ppem, 72]);
            strike.resize(UNIT, 7);
            sbix.extend(strike);
        }
        let glyphs = [crate::fixture::boxed(500); 3];
        let file = crate::fixture::font(&glyphs, &[('A', 2)], &[(b"sbix", sbix)]);
        let loaded = load_face(from(&file), None, 64 << 20).expect("loaded");
        let table = table_of(&loaded, b"sbix");
        assert_eq!(&table[12..14], &300_u16.to_be_bytes());
    }

    fn table_of<'a>(font: &'a [u8], tag: &[u8; 4]) -> &'a [u8] {
        let count = usize::from(u16::from_be_bytes([font[4], font[5]]));
        let at = (0..count)
            .map(|slot| 12 + slot * 16)
            .find(|at| &font[*at..*at + 4] == tag)
            .expect("table");
        let word = |from: usize| {
            u32::from_be_bytes([font[from], font[from + 1], font[from + 2], font[from + 3]])
                as usize
        };
        &font[word(at + 8)..word(at + 8) + word(at + 12)]
    }

    #[test]
    fn a_face_names_its_family_and_says_when_it_is_bold_or_italic() {
        let glyphs = [crate::fixture::boxed(500); 3];
        let build = |selection: Option<u16>, mac_style: u16| {
            let mut extra = vec![(b"name", names(&[(1, "Fam"), (4, "Fam Bold Italic")]))];
            if let Some(selection) = selection {
                let mut os2 = vec![0_u8; 78];
                os2[62..64].copy_from_slice(&selection.to_be_bytes());
                extra.push((b"OS/2", os2));
            }
            let mut bytes = crate::fixture::font(&glyphs, &[('A', 2)], &extra);
            // The style in `head` is at byte 44 of that table.
            let head = (0..usize::from(u16::from_be_bytes([bytes[4], bytes[5]])))
                .map(|slot| 12 + slot * 16)
                .find(|at| &bytes[*at..*at + 4] == b"head")
                .expect("head");
            let start = u32::from_be_bytes([
                bytes[head + 8],
                bytes[head + 9],
                bytes[head + 10],
                bytes[head + 11],
            ]) as usize;
            bytes[start + 44..start + 46].copy_from_slice(&mac_style.to_be_bytes());
            bytes
        };
        let face = |bytes: &[u8]| list_faces(from(bytes)).expect("faces").remove(0);
        let both = face(&build(Some(0x21), 0));
        assert_eq!(both.family.as_deref(), Some("Fam"));
        assert_eq!(both.full_name.as_deref(), Some("Fam Bold Italic"));
        // It names no design apart from its family.
        assert_eq!(both.design_family, None);
        // A weight filed as a family of its own names the whole design.
        let glyphs = [crate::fixture::boxed(500); 3];
        let thin = crate::fixture::font(
            &glyphs,
            &[('A', 2)],
            &[(b"name", names(&[(1, "Fam Thin"), (16, "Fam")]))],
        );
        let thin = face(&thin);
        assert_eq!(thin.family.as_deref(), Some("Fam Thin"));
        assert_eq!(thin.design_family.as_deref(), Some("Fam"));
        assert!(both.bold && both.italic);
        let bold = face(&build(Some(0x20), 0));
        assert!(bold.bold && !bold.italic);
        // `OS/2` is believed over `head`; without it `head` says.
        let plain = face(&build(Some(0x40), 3));
        assert!(!plain.bold && !plain.italic);
        let old = face(&build(None, 2));
        assert!(!old.bold && old.italic);
        // The sample face has no family name and no style.
        let sample = face(SAMPLE_TTF);
        assert!(!sample.bold && !sample.italic);
    }

    #[test]
    fn a_collection_lists_each_face_and_reads_no_outlines() {
        let mut furthest = 0;
        let faces = list_faces(|offset, len| {
            furthest = furthest.max(offset as usize + len);
            from(SAMPLE_TTC)(offset, len)
        })
        .expect("faces");
        let indexes: Vec<_> = faces.iter().map(|face| face.index).collect();
        assert_eq!(indexes, [Some(0), Some(1)]);
        assert_eq!(faces[0].full_name.as_deref(), Some("Cutlass Sample"));
        assert!(furthest <= SAMPLE_TTC.len());
    }

    #[test]
    fn a_face_copied_out_of_a_collection_reads_like_the_face_inside() {
        for index in 0..2 {
            let own = single_face(SAMPLE_TTC, index).expect("copy");
            assert_eq!(own.len() % 4, 0);
            let copied = read(&own).expect("standalone");
            let inside = read_collection(SAMPLE_TTC, index).expect("inside");
            assert_eq!(copied.glyph_count(), inside.glyph_count());
            assert_eq!(copied.units_per_em(), inside.units_per_em());
            assert_eq!(
                copied.advances("A").expect("advance"),
                inside.advances("A").expect("advance")
            );
            assert_eq!(
                list_faces(from(&own)).expect("faces")[0].full_name,
                list_faces(from(SAMPLE_TTC)).expect("faces")[index as usize].full_name
            );
        }
    }

    #[test]
    fn bad_collections_and_indexes_are_refused() {
        assert_eq!(
            single_face(SAMPLE_TTC, 2).expect_err("index"),
            FontError::CollectionIndex { index: 2, count: 2 }
        );
        assert!(single_face(SAMPLE_TTF, 0).is_err());
        assert_eq!(
            list_faces(from(b"ttcf")).expect_err("short"),
            FontError::Truncated
        );
        assert_eq!(
            list_faces(from(b"wOFFwOFFwOFF")).expect_err("kind"),
            FontError::Unsupported
        );

        let mut huge = SAMPLE_TTC.to_vec();
        huge[8..12].copy_from_slice(&(MAX_COLLECTION_FACES + 1).to_be_bytes());
        assert_eq!(
            list_faces(from(&huge)).expect_err("count"),
            FontError::BadTable
        );
        let mut empty = SAMPLE_TTC.to_vec();
        empty[8..12].copy_from_slice(&0_u32.to_be_bytes());
        assert_eq!(
            list_faces(from(&empty)).expect_err("empty"),
            FontError::EmptyTable
        );
        // A face offset past the end is a short file, not a panic.
        let mut far = SAMPLE_TTC.to_vec();
        far[12..16].copy_from_slice(&u32::MAX.to_be_bytes());
        assert_eq!(
            list_faces(from(&far)).expect_err("far"),
            FontError::Truncated
        );
    }

    #[test]
    fn a_full_name_prefers_english_unicode_and_skips_junk() {
        // Two id-4 records: Macintosh ASCII, then Windows English.
        let mut table = vec![0, 0, 0, 2, 0, 30];
        table.extend_from_slice(&[0, 1, 0, 0, 0, 0, 0, 4, 0, 3, 0, 0]);
        table.extend_from_slice(&[0, 3, 0, 1, 4, 9, 0, 4, 0, 4, 0, 3]);
        table.extend_from_slice(b"Mac");
        table.extend_from_slice(&[0, b'W', 0, b'n']);
        assert_eq!(name(&table, 4).as_deref(), Some("Wn"));
        // The Windows name in another language, Urdu: the Macintosh
        // English one is the name, as for the Mac's own Arabic face.
        table[22..24].copy_from_slice(&0x0420_u16.to_be_bytes());
        assert_eq!(name(&table, 4).as_deref(), Some("Mac"));
        // Only the Windows one in that language: it is used.
        table[6..8].copy_from_slice(&7_u16.to_be_bytes());
        assert_eq!(name(&table, 4).as_deref(), Some("Wn"));
        table[6..8].copy_from_slice(&1_u16.to_be_bytes());
        // Only the Macintosh one: it is used.
        table[2..4].copy_from_slice(&1_u16.to_be_bytes());
        assert_eq!(name(&table, 4).as_deref(), Some("Mac"));
        // A record pointing past the table is skipped, not read.
        table[16..18].copy_from_slice(&900_u16.to_be_bytes());
        assert_eq!(name(&table, 4), None);
        assert_eq!(name(&[0, 0], 4), None);
    }
}

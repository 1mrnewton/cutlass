//! The faces in a font file, and one face of a collection as its own file.
//!
//! [`list_faces`] reads through a callback, so a caller with a large
//! collection on disk reads only the headers, the table directories and the
//! `name` tables, not the outlines. [`single_face`] copies the tables of one
//! collection face into a standalone font that [`crate::read`] opens.

use crate::{Directory, FontError, bytes_at, collection, u16_at, u32_at};

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
}

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
        return Ok(vec![FaceEntry {
            index: None,
            full_name: full_name_at(&mut read, 0, &header)?,
        }]);
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
        faces.push(FaceEntry {
            index: Some(index),
            full_name: full_name_at(&mut read, offset, &header)?,
        });
    }
    Ok(faces)
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

/// The full name of the face whose directory starts at `offset`. A missing
/// or unreadable `name` table gives `None`; a broken directory is an error.
fn full_name_at(
    read: &mut impl FnMut(u64, usize) -> Result<Vec<u8>, FontError>,
    offset: u64,
    header: &[u8],
) -> Result<Option<String>, FontError> {
    let count = usize::from(u16_at(header, 4)?);
    if count == 0 {
        return Err(FontError::EmptyTable);
    }
    if count > MAX_TABLES {
        return Err(FontError::BadTable);
    }
    let records = read(offset + 12, count * 16)?;
    for slot in 0..count {
        let at = slot * 16;
        if &records[at..at + 4] != b"name" {
            continue;
        }
        let start = u32_at(&records, at + 8)?;
        let length = u32_at(&records, at + 12)?;
        if length > MAX_NAME_TABLE {
            return Ok(None);
        }
        let Ok(table) = read(u64::from(start), length as usize) else {
            return Ok(None);
        };
        return Ok(full_name(&table));
    }
    Ok(None)
}

/// Name id 4 from a `name` table: English Windows or Unicode first, then any
/// Windows or Unicode entry, then a plain-ASCII Macintosh one.
pub(crate) fn full_name(table: &[u8]) -> Option<String> {
    let count = usize::from(u16_at(table, 2).ok()?);
    let storage = usize::from(u16_at(table, 4).ok()?);
    let mut best: Option<(u8, String)> = None;
    for slot in 0..count {
        let record = bytes_at(table, 6 + slot * 12, 12).ok()?;
        let platform = u16_at(record, 0).ok()?;
        let encoding = u16_at(record, 2).ok()?;
        let language = u16_at(record, 4).ok()?;
        if u16_at(record, 6).ok()? != 4 {
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
                    1
                };
                (rank, utf16(raw)?)
            }
            (1, 0) if language == 0 && raw.is_ascii() => {
                (2, String::from_utf8_lossy(raw).into_owned())
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
        assert_eq!(full_name(&table).as_deref(), Some("Wn"));
        // Only the Macintosh one: it is used.
        table[2..4].copy_from_slice(&1_u16.to_be_bytes());
        assert_eq!(full_name(&table).as_deref(), Some("Mac"));
        // A record pointing past the table is skipped, not read.
        table[16..18].copy_from_slice(&900_u16.to_be_bytes());
        assert_eq!(full_name(&table), None);
        assert_eq!(full_name(&[0, 0]), None);
    }
}

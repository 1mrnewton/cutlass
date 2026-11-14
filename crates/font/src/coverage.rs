//! Which characters a face has, read from its file without the rest of it.
//!
//! A face that may draw a character another face lacks is asked through
//! its character map alone: a few thousand bytes of a file that may hold
//! many megabytes of outlines and pictures.

use crate::{FontError, MAX_COLLECTION_FACES, coded_glyph, parse_cmap, u16_at, u32_at};

/// The largest character map kept for a face. The widest on a Mac, of a
/// face with every Chinese character, is under a megabyte.
pub const MAX_CHARACTER_MAP: usize = 4 * 1024 * 1024;
/// The most tables a face's directory is read for.
const MAX_TABLES: usize = 256;

/// One face's character map, held on its own.
#[derive(Clone, Debug)]
pub struct CharacterMap {
    table: Vec<u8>,
    glyphs: u16,
}

impl CharacterMap {
    /// Whether the face has a glyph for `character`, as
    /// [`Font::glyph_id`](crate::Font::glyph_id) would find one.
    #[must_use]
    pub fn has(&self, character: char) -> bool {
        let Ok((map, coding)) = parse_cmap(&self.table) else {
            return false;
        };
        matches!(
            coded_glyph(&map, coding, u32::from(character)),
            Ok(Some(glyph)) if glyph < self.glyphs
        )
    }
}

/// The character map of one face of a font file. `read_at(offset, len)`
/// returns exactly `len` bytes at `offset`; `index` is the face of a
/// collection, and `None` for a file that is one face. Only the face's
/// directory, its glyph count and its character map are read. A map
/// larger than [`MAX_CHARACTER_MAP`] is refused as unsupported.
pub fn load_character_map(
    mut read_at: impl FnMut(u64, usize) -> Option<Vec<u8>>,
    index: Option<u32>,
) -> Result<CharacterMap, FontError> {
    let mut read = |offset: u64, len: usize| {
        read_at(offset, len)
            .filter(|bytes| bytes.len() == len)
            .ok_or(FontError::Truncated)
    };
    let header = read(0, 12)?;
    let start = match (&header[..4] == b"ttcf", index) {
        (false, None | Some(0)) => 0,
        (false, Some(_)) => return Err(FontError::CollectionIndex { index: 0, count: 1 }),
        (true, index) => {
            let index = index.unwrap_or(0);
            let count = u32_at(&header, 8)?;
            if count > MAX_COLLECTION_FACES {
                return Err(FontError::BadTable);
            }
            if index >= count {
                return Err(FontError::CollectionIndex { index, count });
            }
            let at = 12 + u64::from(index) * 4;
            u64::from(u32_at(&read(at, 4)?, 0)?)
        }
    };
    let directory = read(start, 12)?;
    let count = usize::from(u16_at(&directory, 4)?);
    if count == 0 {
        return Err(FontError::EmptyTable);
    }
    if count > MAX_TABLES {
        return Err(FontError::BadTable);
    }
    let records = read(start + 12, count * 16)?;
    let mut table = None;
    let mut glyphs = None;
    for slot in 0..count {
        let at = slot * 16;
        let offset = u64::from(u32_at(&records, at + 8)?);
        let length =
            usize::try_from(u32_at(&records, at + 12)?).map_err(|_| FontError::BadTable)?;
        match &records[at..at + 4] {
            b"cmap" if length > MAX_CHARACTER_MAP => return Err(FontError::Unsupported),
            b"cmap" => table = Some(read(offset, length)?),
            b"maxp" if length >= 6 => glyphs = Some(u16_at(&read(offset + 4, 2)?, 0)?),
            _ => {}
        }
    }
    let (Some(table), Some(glyphs)) = (table, glyphs) else {
        return Err(FontError::MissingTable);
    };
    // A map that does not read is no map.
    parse_cmap(&table)?;
    Ok(CharacterMap { table, glyphs })
}

#[cfg(test)]
mod tests {
    use super::{CharacterMap, MAX_CHARACTER_MAP, load_character_map};
    use crate::{FontError, read, read_collection};

    const SAMPLE: &[u8] = include_bytes!("../fixtures/sample.ttf");
    const COLLECTION: &[u8] = include_bytes!("../fixtures/sample.ttc");
    const THAI: &[u8] = include_bytes!("../fixtures/thai.ttf");

    fn from(bytes: &[u8]) -> impl FnMut(u64, usize) -> Option<Vec<u8>> + '_ {
        |offset, len| {
            let start = usize::try_from(offset).ok()?;
            bytes
                .get(start..start.checked_add(len)?)
                .map(<[u8]>::to_vec)
        }
    }

    fn map_of(bytes: &[u8], index: Option<u32>) -> Result<CharacterMap, FontError> {
        load_character_map(from(bytes), index)
    }

    #[test]
    fn a_map_says_what_its_face_draws() {
        // Every character from a space to the end of Thai, and a few
        // beyond: the map alone and the whole face say the same.
        for (bytes, index) in [(SAMPLE, None), (THAI, None), (COLLECTION, Some(1))] {
            let face = match index {
                Some(index) => read_collection(bytes, index),
                None => read(bytes),
            }
            .expect("face");
            let map = map_of(bytes, index).expect("map");
            let mut drawn = 0;
            for character in (0x20..0xE80_u32)
                .chain([0xF020, 0xFFFD, 0x1F600, 0x10_FFFF])
                .filter_map(char::from_u32)
            {
                let whole = matches!(face.glyph_id(character), Ok(Some(_)));
                assert_eq!(map.has(character), whole, "{character:?}");
                drawn += u32::from(whole);
            }
            assert!(drawn > 0);
        }
        // A file of one face is its own first face, and has no second.
        assert!(map_of(SAMPLE, Some(0)).expect("map").has('A'));
        assert!(matches!(
            map_of(SAMPLE, Some(1)),
            Err(FontError::CollectionIndex { .. })
        ));
        assert!(matches!(
            map_of(COLLECTION, Some(9)),
            Err(FontError::CollectionIndex { .. })
        ));
    }

    #[test]
    fn only_the_directory_the_count_and_the_map_are_read() {
        let mut asked = 0_usize;
        let map = load_character_map(
            |offset, len| {
                asked += len;
                from(SAMPLE)(offset, len)
            },
            None,
        )
        .expect("map");
        assert!(map.has('A'));
        assert!(asked < SAMPLE.len(), "{asked} of {} bytes", SAMPLE.len());
    }

    #[test]
    fn a_torn_or_hostile_file_gives_no_map_and_is_read_in_time() {
        let started = std::time::Instant::now();
        // Cut at every length: a map, or an error, and never a read past
        // the end.
        for cut in 0..SAMPLE.len() {
            if let Ok(map) = map_of(&SAMPLE[..cut], None) {
                let _ = map.has('A');
            }
        }
        // Every byte of the directories turned to all ones and to nothing.
        for bytes in [SAMPLE, COLLECTION] {
            for at in 0..bytes.len().min(400) {
                for value in [0xFF, 0x00] {
                    let mut torn = bytes.to_vec();
                    torn[at] = value;
                    for index in [None, Some(1)] {
                        if let Ok(map) = map_of(&torn, index) {
                            let _ = map.has('A');
                            let _ = map.has('\u{10FFFF}');
                        }
                    }
                }
            }
        }
        // A map that says it is larger than any map is kept out before it
        // is read.
        let mut large = SAMPLE.to_vec();
        let count = usize::from(u16::from_be_bytes([large[4], large[5]]));
        let record = (0..count)
            .map(|slot| 12 + slot * 16)
            .find(|at| &large[*at..*at + 4] == b"cmap")
            .expect("cmap");
        let huge = u32::try_from(MAX_CHARACTER_MAP + 1).expect("size");
        large[record + 12..record + 16].copy_from_slice(&huge.to_be_bytes());
        let mut asked_most = 0;
        let refused = load_character_map(
            |offset, len| {
                asked_most = asked_most.max(len);
                from(&large)(offset, len)
            },
            None,
        );
        assert!(matches!(refused, Err(FontError::Unsupported)));
        assert!(asked_most < 4_096);
        // Nothing at all, and a reader that gives too few bytes.
        assert!(map_of(&[], None).is_err());
        assert!(load_character_map(|_, _| Some(vec![0; 3]), None).is_err());
        assert!(started.elapsed() < std::time::Duration::from_secs(10));
    }
}

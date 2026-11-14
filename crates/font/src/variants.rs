//! Variation sequences: the `cmap` subtable of format 14, which says what a
//! character draws when a variation selector follows it. An emoji face uses
//! it to give a digit its keycap picture, and a face of Han characters to
//! pick one written form of a character.

use crate::{FontError, bytes_at, u16_at, u32_at};

/// What a face draws for a character and the selector after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// The face does not list the pair: the selector changes nothing.
    Unlisted,
    /// The pair draws as the character alone does.
    Plain,
    /// The pair has a glyph of its own.
    Glyph(u16),
}

const HEADER: usize = 10;
const SELECTOR: usize = 11;
const RANGE: usize = 4;
const MAPPING: usize = 5;

/// The format 14 subtable of a face.
pub(crate) struct Variants<'a> {
    data: &'a [u8],
    selectors: usize,
}

fn u24_at(data: &[u8], at: usize) -> Result<u32, FontError> {
    let bytes = bytes_at(data, at, 3)?;
    Ok(u32::from(bytes[0]) << 16 | u32::from(bytes[1]) << 8 | u32::from(bytes[2]))
}

fn offset_at(data: &[u8], at: usize) -> Result<usize, FontError> {
    usize::try_from(u32_at(data, at)?).map_err(|_| FontError::BadTable)
}

/// The first of `count` records of `size` bytes from `at` whose first three
/// bytes are `wanted` or, with `ranged`, whose range holds it. Records are
/// in rising order, as the format requires; a table that is not finds less.
fn search(
    data: &[u8],
    at: usize,
    count: usize,
    size: usize,
    wanted: u32,
    ranged: bool,
) -> Result<Option<usize>, FontError> {
    let (mut low, mut high) = (0, count);
    while low < high {
        let mid = low + (high - low) / 2;
        let record = at
            .checked_add(mid.checked_mul(size).ok_or(FontError::BadTable)?)
            .ok_or(FontError::BadTable)?;
        let first = u24_at(data, record)?;
        let last = if ranged {
            first + u32::from(*data.get(record + 3).ok_or(FontError::Truncated)?)
        } else {
            first
        };
        if wanted < first {
            high = mid;
        } else if wanted > last {
            low = mid + 1;
        } else {
            return Ok(Some(record));
        }
    }
    Ok(None)
}

/// A list inside the subtable: its first record and how many it has, which
/// the subtable must hold in full.
fn list(data: &[u8], offset: usize, size: usize) -> Result<(usize, usize), FontError> {
    let count = offset_at(data, offset)?;
    let at = offset.checked_add(4).ok_or(FontError::BadTable)?;
    let bytes = count.checked_mul(size).ok_or(FontError::BadTable)?;
    let _records = bytes_at(data, at, bytes)?;
    Ok((at, count))
}

impl<'a> Variants<'a> {
    /// The format 14 subtable of a `cmap` table, when it has one. Such a
    /// subtable is stored under platform 0, encoding 5.
    pub(crate) fn find(table: &'a [u8]) -> Result<Option<Self>, FontError> {
        let records = usize::from(u16_at(table, 2)?);
        for index in 0..records {
            let at = 4 + index * 8;
            if (u16_at(table, at)?, u16_at(table, at + 2)?) != (0, 5) {
                continue;
            }
            let offset = offset_at(table, at + 4)?;
            let subtable = table.get(offset..).ok_or(FontError::Truncated)?;
            if u16_at(subtable, 0)? != 14 {
                continue;
            }
            let length = offset_at(subtable, 2)?;
            let data = bytes_at(subtable, 0, length)?;
            let selectors = offset_at(data, 6)?;
            let bytes = selectors.checked_mul(SELECTOR).ok_or(FontError::BadTable)?;
            let _records = bytes_at(data, HEADER, bytes)?;
            return Ok(Some(Self { data, selectors }));
        }
        Ok(None)
    }

    pub(crate) fn variant(&self, character: u32, selector: u32) -> Result<Variant, FontError> {
        let Some(record) = search(self.data, HEADER, self.selectors, SELECTOR, selector, false)?
        else {
            return Ok(Variant::Unlisted);
        };
        let plain = offset_at(self.data, record + 3)?;
        if plain != 0 {
            let (at, count) = list(self.data, plain, RANGE)?;
            if search(self.data, at, count, RANGE, character, true)?.is_some() {
                return Ok(Variant::Plain);
            }
        }
        let own = offset_at(self.data, record + 7)?;
        if own != 0 {
            let (at, count) = list(self.data, own, MAPPING)?;
            if let Some(mapping) = search(self.data, at, count, MAPPING, character, false)? {
                return Ok(Variant::Glyph(u16_at(self.data, mapping + 3)?));
            }
        }
        Ok(Variant::Unlisted)
    }
}

#[cfg(test)]
mod tests {
    use super::Variant;
    use crate::fixture::{boxed, character_map, font};
    use crate::{FontError, read};

    const EMOJI: char = '\u{FE0F}';
    const TEXT: char = '\u{FE0E}';

    #[test]
    fn a_pair_is_its_own_glyph_the_plain_glyph_or_unlisted() {
        let bytes = crate::fixture_fonts::varied();
        let face = read(&bytes).expect("face");
        assert_eq!(face.variant('1', EMOJI), Ok(Variant::Glyph(3)));
        assert_eq!(face.variant('A', EMOJI), Ok(Variant::Plain));
        assert_eq!(face.variant('A', TEXT), Ok(Variant::Glyph(5)));
        // A character the selector's lists lack, a selector the face
        // lacks, and the ends of the range of characters.
        assert_eq!(face.variant('1', TEXT), Ok(Variant::Unlisted));
        assert_eq!(face.variant('B', EMOJI), Ok(Variant::Unlisted));
        assert_eq!(face.variant('A', '\u{FE00}'), Ok(Variant::Unlisted));
        assert_eq!(face.variant('\0', EMOJI), Ok(Variant::Unlisted));
        assert_eq!(face.variant(char::MAX, EMOJI), Ok(Variant::Unlisted));
        // The characters still map as they did.
        assert_eq!(face.glyph_id('A'), Ok(Some(1)));
        assert_eq!(face.glyph_id(EMOJI), Ok(Some(4)));
    }

    #[test]
    fn a_symbol_face_draws_its_pictures_for_the_keys_that_type_them() {
        // A symbol face maps U+F041 and not `A`, under platform 3,
        // encoding 0.
        let glyphs = [boxed(500), boxed(500), boxed(500)];
        let pictures = [('\u{F041}', 1), ('\u{F0FF}', 2)];
        let mut cmap = character_map(&pictures, &[]);
        cmap[6..8].copy_from_slice(&[0, 0]);
        let bytes = font(&glyphs, &pictures, &[(b"cmap", cmap.clone())]);
        let face = read(&bytes).expect("symbol face");
        assert_eq!(face.glyph_id('A'), Ok(Some(1)));
        assert_eq!(face.glyph_id('\u{F041}'), Ok(Some(1)));
        assert_eq!(face.glyph_id('\u{FF}'), Ok(Some(2)));
        // Only the keys: a character past them is not moved there.
        assert_eq!(face.glyph_id('B'), Ok(None));
        assert_eq!(face.glyph_id('\u{141}'), Ok(None));
        // A face that also has a map of characters uses that one alone.
        let letters = character_map(&[('B', 2)], &[]);
        let mut both = vec![0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 0, 20, 0, 3, 0, 10];
        let second = u32::try_from(20 + cmap.len() - 12).expect("offset");
        both.extend(second.to_be_bytes());
        both.extend(&cmap[12..]);
        both.extend(&letters[12..]);
        let bytes = font(&glyphs, &pictures, &[(b"cmap", both)]);
        let face = read(&bytes).expect("both maps");
        assert_eq!(face.glyph_id('B'), Ok(Some(2)));
        assert_eq!(face.glyph_id('A'), Ok(None));
        assert_eq!(face.glyph_id('\u{F041}'), Ok(None));
    }

    #[test]
    fn a_face_without_the_subtable_lists_no_pair() {
        let bytes = font(&[boxed(500), boxed(500)], &[('A', 1)], &[]);
        let face = read(&bytes).expect("face");
        assert_eq!(face.variant('A', EMOJI), Ok(Variant::Unlisted));
    }

    /// The face of `pairs` with the bytes of its `cmap` table edited.
    fn edited(
        pairs: &[crate::fixture::Varied],
        edit: impl Fn(&mut Vec<u8>),
    ) -> Result<Vec<u8>, FontError> {
        let glyphs = [boxed(500), boxed(500), boxed(500)];
        let map = [('1', 2), ('A', 1)];
        let mut cmap = character_map(&map, pairs);
        edit(&mut cmap);
        let bytes = font(&glyphs, &map, &[(b"cmap", cmap)]);
        match read(&bytes) {
            Ok(_) => Ok(bytes),
            Err(error) => Err(error),
        }
    }

    /// Where the format 14 subtable starts in the `cmap` table built here.
    const SUBTABLE: usize = 20;

    #[test]
    fn a_hostile_subtable_is_an_error_and_never_a_read_past_it() {
        let pairs = [('1', EMOJI, Some(2)), ('A', EMOJI, None)];
        let bytes = edited(&pairs, |_| {}).expect("whole");
        let face = read(&bytes).expect("face");
        assert_eq!(face.variant('1', EMOJI), Ok(Variant::Glyph(2)));
        // A glyph the face does not have.
        let bytes = edited(&[('1', EMOJI, Some(9))], |_| {}).expect("opens");
        let face = read(&bytes).expect("face");
        assert_eq!(face.variant('1', EMOJI), Err(FontError::BadTable));
        // More selectors than the subtable holds, and a length past the
        // table: the face does not open.
        let many = edited(&pairs, |cmap| {
            cmap[SUBTABLE + 6..SUBTABLE + 10].copy_from_slice(&u32::MAX.to_be_bytes());
        });
        assert!(many.is_err());
        let long = edited(&pairs, |cmap| {
            cmap[SUBTABLE + 2..SUBTABLE + 6].copy_from_slice(&u32::MAX.to_be_bytes());
        });
        assert_eq!(long.err(), Some(FontError::Truncated));
        // A list that starts past the subtable, and one that counts more
        // records than it has: the pair is an error when it is asked for.
        for (field, value) in [(3, 0x00FF_FFFF_u32), (7, 0x00FF_FFFF)] {
            let bytes = edited(&pairs, |cmap| {
                let at = SUBTABLE + 10 + field;
                cmap[at..at + 4].copy_from_slice(&value.to_be_bytes());
            })
            .expect("opens");
            let face = read(&bytes).expect("face");
            assert_eq!(face.variant('1', EMOJI), Err(FontError::Truncated));
        }
        let bytes = edited(&pairs, |cmap| {
            let ranges = SUBTABLE + 10 + 11;
            cmap[ranges..ranges + 4].copy_from_slice(&u32::MAX.to_be_bytes());
        })
        .expect("opens");
        let face = read(&bytes).expect("face");
        assert!(face.variant('A', EMOJI).is_err());
        // Every cut of the table opens or is an error, and so is every pair.
        let whole = character_map(&[('1', 2), ('A', 1)], &pairs);
        for length in 0..whole.len() {
            if let Ok(bytes) = edited(&pairs, |cmap| cmap.truncate(length)) {
                let face = read(&bytes).expect("face");
                let _ = face.variant('1', EMOJI);
                let _ = face.variant('A', EMOJI);
            }
        }
    }

    #[test]
    fn a_pair_is_found_quickly_among_many() {
        let glyphs = [boxed(500), boxed(500)];
        let pairs: Vec<crate::fixture::Varied> = (0x4E00_u32..0x4E00 + 20_000)
            .filter_map(char::from_u32)
            .map(|character| (character, '\u{E0100}', Some(1)))
            .collect();
        let map = [('A', 1)];
        let cmap = character_map(&map, &pairs);
        let bytes = font(&glyphs, &map, &[(b"cmap", cmap)]);
        let face = read(&bytes).expect("face");
        let started = std::time::Instant::now();
        let mut found = 0;
        for round in 0..200_000_u32 {
            let character = char::from_u32(0x4E00 + round % 20_000).expect("character");
            if face.variant(character, '\u{E0100}') == Ok(Variant::Glyph(1)) {
                found += 1;
            }
        }
        assert_eq!(found, 200_000);
        // A search of fifteen steps each: far under a second even unoptimized.
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }
}

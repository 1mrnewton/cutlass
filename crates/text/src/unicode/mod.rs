//! Character properties from the Unicode Character Database.
//!
//! `data.bin` is written by `tools/unicode-tables/generate.py`, which reads
//! the properties out of ICU; its doc comment gives the layout. The numbers
//! in the modules below are ICU's own, so the table and this file agree. A
//! code point looks its record up in two steps: a block for its upper bits,
//! then its place in that block.

use std::sync::OnceLock;

static DATA: &[u8] = include_bytes!("data.bin");

/// Line breaking classes, as `ULineBreak` numbers them.
pub(crate) mod lb {
    pub(crate) const UNKNOWN: u8 = 0;
    pub(crate) const AMBIGUOUS: u8 = 1;
    pub(crate) const ALPHABETIC: u8 = 2;
    pub(crate) const BREAK_BOTH: u8 = 3;
    pub(crate) const BREAK_AFTER: u8 = 4;
    pub(crate) const BREAK_BEFORE: u8 = 5;
    pub(crate) const MANDATORY_BREAK: u8 = 6;
    pub(crate) const CONTINGENT_BREAK: u8 = 7;
    pub(crate) const CLOSE_PUNCTUATION: u8 = 8;
    pub(crate) const COMBINING_MARK: u8 = 9;
    pub(crate) const CARRIAGE_RETURN: u8 = 10;
    pub(crate) const EXCLAMATION: u8 = 11;
    pub(crate) const GLUE: u8 = 12;
    pub(crate) const HYPHEN: u8 = 13;
    pub(crate) const IDEOGRAPHIC: u8 = 14;
    pub(crate) const INSEPARABLE: u8 = 15;
    pub(crate) const INFIX_NUMERIC: u8 = 16;
    pub(crate) const LINE_FEED: u8 = 17;
    pub(crate) const NONSTARTER: u8 = 18;
    pub(crate) const NUMERIC: u8 = 19;
    pub(crate) const OPEN_PUNCTUATION: u8 = 20;
    pub(crate) const POSTFIX_NUMERIC: u8 = 21;
    pub(crate) const PREFIX_NUMERIC: u8 = 22;
    pub(crate) const QUOTATION: u8 = 23;
    pub(crate) const COMPLEX_CONTEXT: u8 = 24;
    pub(crate) const SURROGATE: u8 = 25;
    pub(crate) const SPACE: u8 = 26;
    pub(crate) const BREAK_SYMBOLS: u8 = 27;
    pub(crate) const ZERO_WIDTH_SPACE: u8 = 28;
    pub(crate) const NEXT_LINE: u8 = 29;
    pub(crate) const WORD_JOINER: u8 = 30;
    pub(crate) const H2: u8 = 31;
    pub(crate) const H3: u8 = 32;
    pub(crate) const JL: u8 = 33;
    pub(crate) const JT: u8 = 34;
    pub(crate) const JV: u8 = 35;
    pub(crate) const CLOSE_PARENTHESIS: u8 = 36;
    pub(crate) const CONDITIONAL_JAPANESE_STARTER: u8 = 37;
    pub(crate) const HEBREW_LETTER: u8 = 38;
    pub(crate) const REGIONAL_INDICATOR: u8 = 39;
    pub(crate) const E_BASE: u8 = 40;
    pub(crate) const E_MODIFIER: u8 = 41;
    pub(crate) const ZERO_WIDTH_JOINER: u8 = 42;
    pub(crate) const AKSARA: u8 = 43;
    pub(crate) const AKSARA_PREBASE: u8 = 44;
    pub(crate) const AKSARA_START: u8 = 45;
    pub(crate) const VIRAMA_FINAL: u8 = 46;
    pub(crate) const VIRAMA: u8 = 47;
    pub(crate) const UNAMBIGUOUS_HYPHEN: u8 = 48;
}

/// General categories, as `UCharCategory` numbers them.
pub(crate) mod gc {
    pub(crate) const UNASSIGNED: u8 = 0;
    pub(crate) const NONSPACING_MARK: u8 = 6;
    pub(crate) const SPACING_MARK: u8 = 8;
    pub(crate) const INITIAL_PUNCTUATION: u8 = 28;
    pub(crate) const FINAL_PUNCTUATION: u8 = 29;
}

/// East Asian widths, as `UEastAsianWidth` numbers them.
pub(crate) mod ea {
    pub(crate) const HALFWIDTH: u8 = 2;
    pub(crate) const FULLWIDTH: u8 = 3;
    pub(crate) const WIDE: u8 = 5;
}

/// The properties of one code point, packed as the generator packs them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Props(u64);

impl Props {
    fn bits(self, shift: u32, count: u32) -> u8 {
        let value = (self.0 >> shift) & ((1 << count) - 1);
        u8::try_from(value).unwrap_or(0)
    }

    /// One of [`gc`].
    pub(crate) fn category(self) -> u8 {
        self.bits(0, 5)
    }

    /// One of [`lb`].
    pub(crate) fn line_break(self) -> u8 {
        self.bits(13, 6)
    }

    /// One of [`ea`].
    pub(crate) fn east_asian(self) -> u8 {
        self.bits(19, 3)
    }

    pub(crate) fn pictographic(self) -> bool {
        self.bits(22, 1) == 1
    }
}

/// The tables inside [`DATA`], each a run of its bytes.
struct Tables {
    block_bits: u32,
    stage_one: &'static [u8],
    stage_two: &'static [u8],
    records: &'static [u8],
    version: [u8; 3],
}

impl Tables {
    /// Split the file into its tables. A file that does not read gives
    /// `None`, and every code point then has the properties of an
    /// unassigned one.
    fn read(data: &'static [u8]) -> Option<Self> {
        if data.get(..4)? != b"CUTU" || u16_at(data, 4)? != 1 {
            return None;
        }
        let block_bits = u32::from(u16_at(data, 6)?);
        if !(1..=8).contains(&block_bits) {
            return None;
        }
        let mut counts = [0_usize; 8];
        for (index, count) in counts.iter_mut().enumerate() {
            *count = usize::try_from(u32_at(data, 8 + index * 4)?).ok()?;
        }
        let version = [*data.get(40)?, *data.get(41)?, *data.get(42)?];
        let mut at = 44_usize;
        let mut take = |bytes: usize| {
            let end = at.checked_add(bytes)?;
            let table = data.get(at..end)?;
            at = end;
            Some(table)
        };
        let stage_one = take(counts[0].checked_mul(2)?)?;
        let stage_two = take(counts[1].checked_mul(2)?)?;
        let records = take(counts[2].checked_mul(8)?)?;
        Some(Self {
            block_bits,
            stage_one,
            stage_two,
            records,
            version,
        })
    }

    fn props(&self, code_point: u32) -> Props {
        let found = (|| {
            let block = usize::from(u16_at(
                self.stage_one,
                usize::try_from(code_point >> self.block_bits).ok()? * 2,
            )?);
            let within = usize::try_from(code_point & ((1 << self.block_bits) - 1)).ok()?;
            let record = usize::from(u16_at(
                self.stage_two,
                ((block << self.block_bits) + within) * 2,
            )?);
            let bytes = self.records.get(record * 8..record * 8 + 8)?;
            let mut packed = [0; 8];
            packed.copy_from_slice(bytes);
            Some(u64::from_le_bytes(packed))
        })();
        Props(found.unwrap_or(0))
    }
}

fn u16_at(data: &[u8], at: usize) -> Option<u16> {
    let bytes = data.get(at..at.checked_add(2)?)?;
    Some(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn u32_at(data: &[u8], at: usize) -> Option<u32> {
    let bytes = data.get(at..at.checked_add(4)?)?;
    Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn tables() -> Option<&'static Tables> {
    static TABLES: OnceLock<Option<Tables>> = OnceLock::new();
    TABLES.get_or_init(|| Tables::read(DATA)).as_ref()
}

/// The properties of `character`.
pub(crate) fn props(character: char) -> Props {
    tables().map_or(Props(0), |tables| tables.props(u32::from(character)))
}

/// The Unicode version the table was written from: major, minor, patch.
#[must_use]
pub fn unicode_version() -> [u8; 3] {
    tables().map_or([0; 3], |tables| tables.version)
}

#[cfg(test)]
mod tests {
    use super::{Tables, ea, gc, lb, props, unicode_version};

    #[test]
    fn the_table_reads_and_names_its_unicode_version() {
        assert!(Tables::read(super::DATA).is_some());
        assert_eq!(unicode_version(), [17, 0, 0]);
    }

    #[test]
    fn known_characters_have_their_properties() {
        assert_eq!(props('A').line_break(), lb::ALPHABETIC);
        assert_eq!(props(' ').line_break(), lb::SPACE);
        assert_eq!(props('\n').line_break(), lb::LINE_FEED);
        assert_eq!(props('-').line_break(), lb::HYPHEN);
        assert_eq!(props('7').line_break(), lb::NUMERIC);
        assert_eq!(props('(').line_break(), lb::OPEN_PUNCTUATION);
        assert_eq!(props('\u{2014}').line_break(), lb::BREAK_BOTH);
        assert_eq!(props('\u{2010}').line_break(), lb::UNAMBIGUOUS_HYPHEN);
        assert_eq!(props('\u{0301}').line_break(), lb::COMBINING_MARK);
        assert_eq!(props('\u{0301}').category(), gc::NONSPACING_MARK);
        assert_eq!(props('\u{093E}').category(), gc::SPACING_MARK);
        assert_eq!(props('\u{4E2D}').line_break(), lb::IDEOGRAPHIC);
        assert_eq!(props('\u{4E2D}').east_asian(), ea::WIDE);
        assert_eq!(props('\u{FF21}').east_asian(), ea::FULLWIDTH);
        assert_eq!(props('\u{0915}').line_break(), lb::ALPHABETIC);
        assert_eq!(props('\u{1F600}').line_break(), lb::IDEOGRAPHIC);
        assert!(props('\u{1F600}').pictographic());
        assert!(!props('A').pictographic());
        assert_eq!(props('\u{10FFFF}').category(), gc::UNASSIGNED);
    }

    #[test]
    fn a_torn_table_reads_as_nothing() {
        assert!(Tables::read(b"").is_none());
        assert!(Tables::read(b"CUTU\x01\x00\x06\x00").is_none());
        assert!(Tables::read(&super::DATA[..super::DATA.len() / 2]).is_none());
        let mut wrong = [0_u8; 64];
        wrong[..4].copy_from_slice(b"NOPE");
        let wrong: &'static [u8] = Box::leak(Box::new(wrong));
        assert!(Tables::read(wrong).is_none());
    }
}

//! Horizontal kerning pairs from a `kern` table.
//!
//! Format 0 stores one adjustment per glyph pair. A missing table means no
//! pairs. In a version-0 table a horizontal subtable in another format is
//! refused, so a pair is not dropped on the floor, and vertical subtables
//! are skipped. Apple's version-1 table is read for its format 0 subtables
//! alone: its other formats are state machines, which also move marks,
//! and those are left out.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::{FontError, i16_at, u16_at};

const MAX_PAIRS: usize = 65_536;

#[derive(Clone, Copy)]
struct Pair {
    left: u16,
    right: u16,
    value: i16,
}

pub(crate) struct Kerning {
    pairs: Vec<Pair>,
}

impl Kerning {
    pub(crate) fn empty() -> Self {
        Self { pairs: Vec::new() }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    pub(crate) fn get(&self, left: u16, right: u16) -> i16 {
        let mut low = 0;
        let mut high = self.pairs.len();
        while low < high {
            let mid = low + (high - low) / 2;
            let pair = &self.pairs[mid];
            match (pair.left, pair.right).cmp(&(left, right)) {
                Ordering::Less => low = mid + 1,
                Ordering::Greater => high = mid,
                Ordering::Equal => return pair.value,
            }
        }
        0
    }
}

pub(crate) fn parse(table: &[u8]) -> Result<Kerning, FontError> {
    if table.is_empty() {
        return Err(FontError::EmptyTable);
    }
    if table.len() < 4 {
        return Err(FontError::Truncated);
    }
    if u16_at(table, 0)? == 1 && u16_at(table, 2)? == 0 {
        // A version-1 table that does not read is no kerning, not a face
        // that does not open.
        return apple(table).map_err(|_| FontError::Unsupported);
    }
    if u16_at(table, 0)? != 0 {
        return Err(FontError::Unsupported);
    }
    let count = usize::from(u16_at(table, 2)?);
    let mut at = 4;
    let mut pairs = BTreeMap::new();
    for _ in 0..count {
        let sub = table.get(at..).ok_or(FontError::Truncated)?;
        if sub.len() < 6 {
            return Err(FontError::Truncated);
        }
        let sub_version = u16_at(sub, 0)?;
        let length = usize::from(u16_at(sub, 2)?);
        if length < 6 {
            return Err(FontError::BadTable);
        }
        let end = at.checked_add(length).ok_or(FontError::BadTable)?;
        let sub = table.get(at..end).ok_or(FontError::Truncated)?;
        let coverage = u16_at(sub, 4)?;
        let horizontal = coverage & 0x0001 == 0x0001;
        if horizontal {
            let minimum = coverage & 0x0002 == 0x0002;
            let cross_stream = coverage & 0x0004 == 0x0004;
            let replace = coverage & 0x0008 == 0x0008;
            let format = coverage >> 8;
            if sub_version != 0 || minimum || cross_stream || format != 0 {
                return Err(FontError::Unsupported);
            }
            read_pairs(sub, 6, replace, &mut pairs)?;
        }
        at = end;
    }
    let pairs = pairs
        .into_iter()
        .map(|((left, right), value)| Pair { left, right, value })
        .collect();
    Ok(Kerning { pairs })
}

/// The pairs of an Apple version-1 table's format 0 subtables.
fn apple(table: &[u8]) -> Result<Kerning, FontError> {
    let count = usize::try_from(crate::u32_at(table, 4)?).map_err(|_| FontError::BadTable)?;
    if count > 256 {
        return Err(FontError::BadTable);
    }
    let mut at = 8;
    let mut pairs = BTreeMap::new();
    for _ in 0..count {
        let length = usize::try_from(crate::u32_at(table, at)?).map_err(|_| FontError::BadTable)?;
        if length < 8 {
            return Err(FontError::BadTable);
        }
        let end = at.checked_add(length).ok_or(FontError::BadTable)?;
        let sub = table.get(at..end).ok_or(FontError::Truncated)?;
        let coverage = u16_at(sub, 4)?;
        // Not vertical, not across the line, not a variation, format 0.
        if coverage & 0xE0FF == 0 {
            read_pairs(sub, 8, false, &mut pairs)?;
        }
        at = end;
    }
    let pairs = pairs
        .into_iter()
        .map(|((left, right), value)| Pair { left, right, value })
        .collect();
    Ok(Kerning { pairs })
}

/// The pairs of one format 0 subtable, whose own header is `header`
/// bytes long.
fn read_pairs(
    sub: &[u8],
    header: usize,
    replace: bool,
    pairs: &mut BTreeMap<(u16, u16), i16>,
) -> Result<(), FontError> {
    let first = header + 8;
    if sub.len() < first {
        return Err(FontError::Truncated);
    }
    let count = usize::from(u16_at(sub, header)?);
    let bytes = count.checked_mul(6).ok_or(FontError::BadTable)?;
    let need = bytes.checked_add(first).ok_or(FontError::BadTable)?;
    if sub.len() < need {
        return Err(FontError::Truncated);
    }
    // Real faces repeat a pair or store one out of its place. The first
    // of a pair in a subtable is the one that counts, as it is the one a
    // search of a sorted list meets.
    let mut own: BTreeMap<(u16, u16), i16> = BTreeMap::new();
    for index in 0..count {
        let offset = first + index * 6;
        let left = u16_at(sub, offset)?;
        let right = u16_at(sub, offset + 2)?;
        let value = i16_at(sub, offset + 4)?;
        own.entry((left, right)).or_insert(value);
    }
    for (key, value) in own {
        if replace {
            pairs.insert(key, value);
        } else {
            pairs
                .entry(key)
                .and_modify(|current| *current = current.saturating_add(value))
                .or_insert(value);
        }
        if pairs.len() > MAX_PAIRS {
            return Err(FontError::BadTable);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse;
    use crate::FontError;

    fn format0(coverage: u16, pairs: &[(u16, u16, i16)]) -> Vec<u8> {
        let count = u16::try_from(pairs.len()).expect("count");
        let mut body = Vec::new();
        body.extend_from_slice(&count.to_be_bytes());
        body.extend_from_slice(&0u16.to_be_bytes());
        body.extend_from_slice(&0u16.to_be_bytes());
        body.extend_from_slice(&0u16.to_be_bytes());
        for (left, right, value) in pairs {
            body.extend_from_slice(&left.to_be_bytes());
            body.extend_from_slice(&right.to_be_bytes());
            body.extend_from_slice(&value.to_be_bytes());
        }
        let length = u16::try_from(6 + body.len()).expect("length");
        let mut sub = Vec::new();
        sub.extend_from_slice(&0u16.to_be_bytes());
        sub.extend_from_slice(&length.to_be_bytes());
        sub.extend_from_slice(&coverage.to_be_bytes());
        sub.extend(body);
        let mut table = Vec::new();
        table.extend_from_slice(&0u16.to_be_bytes());
        table.extend_from_slice(&1u16.to_be_bytes());
        table.extend(sub);
        table
    }

    fn with_subs(subs: &[Vec<u8>]) -> Vec<u8> {
        let count = u16::try_from(subs.len()).expect("count");
        let mut table = Vec::new();
        table.extend_from_slice(&0u16.to_be_bytes());
        table.extend_from_slice(&count.to_be_bytes());
        for sub in subs {
            table.extend_from_slice(sub);
        }
        table
    }

    fn subtable(coverage: u16, pairs: &[(u16, u16, i16)]) -> Vec<u8> {
        let table = format0(coverage, pairs);
        table[4..].to_vec()
    }

    #[test]
    fn an_apple_table_gives_the_pairs_of_its_pair_subtables() {
        // One format 0 subtable of two pairs, then a state-machine one.
        let mut table = vec![0, 1, 0, 0, 0, 0, 0, 2];
        let mut pairs = vec![0, 0, 0, 28, 0, 0, 0, 0, 0, 2, 0, 12, 0, 1, 0, 0];
        pairs.extend([0, 2, 0, 3, 0xFF, 0xB0, 0, 3, 0, 2, 0, 40]);
        table.extend(&pairs);
        table.extend([0, 0, 0, 12, 0, 1, 0, 0, 9, 9, 9, 9]);
        let kerning = parse(&table).expect("apple");
        assert_eq!((kerning.get(2, 3), kerning.get(3, 2)), (-80, 40));
        assert_eq!(kerning.get(2, 2), 0);
        assert!(!kerning.is_empty());
        // A torn one is no kerning at all.
        assert!(matches!(
            parse(&table[..table.len() - 14]),
            Err(FontError::Unsupported)
        ));
        assert!(matches!(
            parse(&[0, 1, 0, 0, 0, 0]),
            Err(FontError::Unsupported)
        ));
    }

    #[test]
    fn pairs_sum_until_an_override_replaces_them() {
        let first = subtable(1, &[(1, 2, 10), (2, 2, 4)]);
        let second = subtable(1, &[(1, 2, -3)]);
        let summed = parse(&with_subs(&[first, second])).expect("sum");
        assert_eq!(summed.get(1, 2), 7);
        assert_eq!(summed.get(2, 2), 4);
        assert_eq!(summed.get(9, 9), 0);

        let kept = subtable(1, &[(1, 2, 10)]);
        let replaced = subtable(0x0009, &[(1, 2, -5), (4, 1, 3)]);
        let kern = parse(&with_subs(&[kept, replaced])).expect("override");
        assert_eq!(kern.get(1, 2), -5);
        assert_eq!(kern.get(4, 1), 3);
    }

    #[test]
    fn a_vertical_subtable_is_skipped() {
        let vertical = subtable(0, &[(1, 1, 99)]);
        let horizontal = subtable(1, &[(2, 3, -8)]);
        let kern = parse(&with_subs(&[vertical, horizontal])).expect("kern");
        assert_eq!(kern.get(1, 1), 0);
        assert_eq!(kern.get(2, 3), -8);
    }

    #[test]
    fn a_repeated_or_misplaced_pair_is_read_and_the_first_counts() {
        // A widely installed bold face stores one pair twice.
        let kern = parse(&format0(1, &[(2, 1, 5), (1, 1, 4), (2, 1, 9), (1, 1, 4)])).expect("kern");
        assert_eq!(kern.get(2, 1), 5);
        assert_eq!(kern.get(1, 1), 4);
        assert_eq!(kern.get(1, 2), 0);
        // A second subtable still adds to the first.
        let more = with_subs(&[
            subtable(1, &[(1, 1, 4), (1, 1, 50)]),
            subtable(1, &[(1, 1, 3)]),
        ]);
        assert_eq!(parse(&more).expect("kern").get(1, 1), 7);
    }

    #[test]
    fn a_short_table_is_an_error() {
        assert!(matches!(parse(&[]), Err(FontError::EmptyTable)));
        assert!(matches!(parse(&[0, 0]), Err(FontError::Truncated)));
        assert!(matches!(parse(&[0, 1, 0, 0]), Err(FontError::Unsupported)));
        assert!(matches!(parse(&[0, 0, 0, 1]), Err(FontError::Truncated)));
        let mut short = format0(1, &[(1, 2, 5)]);
        short[11] = 50;
        assert!(matches!(parse(&short), Err(FontError::Truncated)));
    }

    #[test]
    fn a_horizontal_format_this_reader_cannot_apply_is_refused() {
        assert!(matches!(
            parse(&format0(0x0101, &[])),
            Err(FontError::Unsupported)
        ));
        assert!(matches!(
            parse(&format0(0x0003, &[])),
            Err(FontError::Unsupported)
        ));
        assert!(matches!(
            parse(&format0(0x0005, &[])),
            Err(FontError::Unsupported)
        ));
    }
}

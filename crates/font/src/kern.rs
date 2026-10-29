//! Horizontal kerning pairs from a version-0 `kern` table.
//!
//! Format 0 stores one adjustment per glyph pair. A missing table means no
//! pairs. A horizontal subtable in another format is refused, so a pair is
//! not dropped on the floor. Vertical subtables are skipped.

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
            read_pairs(sub, replace, &mut pairs)?;
        }
        at = end;
    }
    let pairs = pairs
        .into_iter()
        .map(|((left, right), value)| Pair { left, right, value })
        .collect();
    Ok(Kerning { pairs })
}

fn read_pairs(
    sub: &[u8],
    replace: bool,
    pairs: &mut BTreeMap<(u16, u16), i16>,
) -> Result<(), FontError> {
    if sub.len() < 14 {
        return Err(FontError::Truncated);
    }
    let count = usize::from(u16_at(sub, 6)?);
    let bytes = count.checked_mul(6).ok_or(FontError::BadTable)?;
    let need = bytes.checked_add(14).ok_or(FontError::BadTable)?;
    if sub.len() < need {
        return Err(FontError::Truncated);
    }
    let mut previous: Option<(u16, u16)> = None;
    for index in 0..count {
        let offset = 14 + index * 6;
        let left = u16_at(sub, offset)?;
        let right = u16_at(sub, offset + 2)?;
        let value = i16_at(sub, offset + 4)?;
        let key = (left, right);
        if previous.is_some_and(|prior| prior >= key) {
            return Err(FontError::BadTable);
        }
        previous = Some(key);
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
    fn a_short_or_unsorted_table_is_an_error() {
        assert!(matches!(parse(&[]), Err(FontError::EmptyTable)));
        assert!(matches!(parse(&[0, 0]), Err(FontError::Truncated)));
        assert!(matches!(parse(&[0, 1, 0, 0]), Err(FontError::Unsupported)));
        assert!(matches!(parse(&[0, 0, 0, 1]), Err(FontError::Truncated)));
        assert!(matches!(
            parse(&format0(1, &[(2, 1, 5), (1, 1, 4)])),
            Err(FontError::BadTable)
        ));
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

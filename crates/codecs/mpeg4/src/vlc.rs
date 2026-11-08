//! Variable length code lookup.
//!
//! Each table is a single level of `2^max_len` entries, so one peek and
//! one index decode any code. The largest MPEG-4 code is the 15-bit
//! reversible coefficient code.

use crate::bits::BitReader;

#[derive(Clone, Copy, Default)]
struct Entry {
    symbol: u16,
    /// Code length; zero marks a bit pattern no code starts with.
    len: u8,
}

/// A prefix code decoder.
pub(crate) struct Vlc {
    bits: u32,
    table: Vec<Entry>,
}

/// Why a code list cannot form a decoder.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum VlcError {
    /// A code is longer than 16 bits or has no bits.
    Length,
    /// Two codes share a prefix.
    Overlap,
}

impl Vlc {
    /// Builds from `(length, code, symbol)` triples.
    pub(crate) fn new(codes: &[(u8, u16, u16)]) -> Result<Self, VlcError> {
        let bits = u32::from(codes.iter().map(|c| c.0).max().unwrap_or(1));
        if bits > 16 || codes.iter().any(|c| c.0 == 0) {
            return Err(VlcError::Length);
        }
        let mut table = vec![Entry::default(); 1 << bits];
        for &(len, code, symbol) in codes {
            let shift = bits - u32::from(len);
            let start = usize::from(code) << shift;
            let end = start + (1usize << shift);
            let Some(slots) = table.get_mut(start..end) else {
                return Err(VlcError::Length);
            };
            for slot in slots {
                if slot.len != 0 {
                    return Err(VlcError::Overlap);
                }
                *slot = Entry { symbol, len };
            }
        }
        Ok(Self { bits, table })
    }

    /// A decoder that matches nothing. The tables fall back to it only if
    /// their constant codes were inconsistent, which a test rules out.
    pub(crate) fn empty() -> Self {
        Self {
            bits: 1,
            table: vec![Entry::default(); 2],
        }
    }

    /// Builds from a list of `(length, code)` where the symbol is the index.
    pub(crate) fn indexed(codes: &[(u8, u16)]) -> Result<Self, VlcError> {
        let triples: Vec<(u8, u16, u16)> = codes
            .iter()
            .enumerate()
            .map(|(i, &(len, code))| (len, code, i as u16))
            .collect();
        Self::new(&triples)
    }

    /// Decodes one symbol, or `None` when the bits start no code. A miss
    /// consumes nothing.
    #[inline]
    pub(crate) fn read(&self, r: &mut BitReader<'_>) -> Option<u16> {
        let index = r.peek(self.bits) as usize;
        let entry = self.table.get(index).copied().unwrap_or_default();
        if entry.len == 0 {
            return None;
        }
        r.skip(u32::from(entry.len));
        Some(entry.symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::{Vlc, VlcError};
    use crate::bits::BitReader;

    #[test]
    fn decodes_codes_and_misses_unknown_patterns() {
        let vlc = Vlc::new(&[(1, 0b1, 7), (2, 0b01, 8), (3, 0b001, 9)]).expect("vlc");
        // 1 01 001 000...
        let data = [0b1010_0100];
        let mut r = BitReader::new(&data);
        assert_eq!(vlc.read(&mut r), Some(7));
        assert_eq!(vlc.read(&mut r), Some(8));
        assert_eq!(vlc.read(&mut r), Some(9));
        let at = r.position();
        assert_eq!(vlc.read(&mut r), None);
        assert_eq!(r.position(), at, "a miss consumes nothing");
    }

    #[test]
    fn refuses_overlapping_codes() {
        assert_eq!(
            Vlc::new(&[(1, 0b1, 0), (2, 0b11, 1)]).err(),
            Some(VlcError::Overlap)
        );
        assert_eq!(Vlc::new(&[(0, 0, 0)]).err(), Some(VlcError::Length));
    }
}

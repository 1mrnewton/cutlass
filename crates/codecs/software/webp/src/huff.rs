//! Canonical prefix codes.
//!
//! WebP lossless 6.2.1. Code lengths are either a one- or two-symbol simple
//! code or a normal code whose lengths are themselves prefix-coded. The first
//! bit of a code is the high bit of its canonical value.

use crate::MAX_BYTES;
use crate::bits::Bits;
use crate::error::Error;

/// Running size of the prefix-code tables. `docs/codecs.md` caps the sum.
pub(crate) struct Budget {
    used: usize,
}

impl Budget {
    pub(crate) fn new() -> Self {
        Self { used: 0 }
    }

    pub(crate) fn charge(&mut self, bytes: usize) -> Result<(), Error> {
        self.used = self.used.checked_add(bytes).ok_or(Error::TooLarge)?;
        if self.used > MAX_BYTES {
            Err(Error::TooLarge)
        } else {
            Ok(())
        }
    }
}

/// One canonical tree. A single leaf consumes no bits.
pub(crate) struct Huff {
    single: Option<u16>,
    max_len: u8,
    /// Lowest canonical code of this length.
    min: [u32; 16],
    count: [u16; 16],
    /// Index into `symbols` of the first code of this length.
    base: [u16; 16],
    symbols: Vec<u16>,
}

impl Huff {
    /// The next symbol. A single leaf does not read.
    pub(crate) fn decode(&self, bits: &mut Bits<'_>) -> Result<u16, Error> {
        if let Some(symbol) = self.single {
            return Ok(symbol);
        }
        let mut code = 0u32;
        for len in 1..=self.max_len {
            code = (code << 1) | u32::from(bits.read_bit()?);
            let n = u32::from(self.count[usize::from(len)]);
            if n == 0 {
                continue;
            }
            let min = self.min[usize::from(len)];
            if code >= min && code - min < n {
                let off = usize::try_from(code - min).expect("a code offset fits");
                let at = usize::from(self.base[usize::from(len)]) + off;
                return self
                    .symbols
                    .get(at)
                    .copied()
                    .ok_or(Error::Inconsistent("a prefix code is not in the tree"));
            }
        }
        Err(Error::Inconsistent("a prefix code is not in the tree"))
    }
}

/// Reads one prefix code and charges its table against `budget`.
pub(crate) fn read_huff(
    bits: &mut Bits<'_>,
    alphabet: usize,
    budget: &mut Budget,
) -> Result<Huff, Error> {
    let lengths = read_lengths(bits, alphabet)?;
    let huff = build(&lengths)?;
    // A fixed cost per tree, plus the symbol list, so a flood of trees stops.
    budget.charge(64)?;
    budget.charge(huff.symbols.len().checked_mul(2).ok_or(Error::TooLarge)?)?;
    Ok(huff)
}

/// WebP lossless 6.2.1. The returned lengths cover symbols `0..len`.
pub(crate) fn read_lengths(bits: &mut Bits<'_>, alphabet: usize) -> Result<Vec<u8>, Error> {
    if bits.read(1)? == 1 {
        simple(bits, alphabet)
    } else {
        normal(bits, alphabet)
    }
}

fn simple(bits: &mut Bits<'_>, alphabet: usize) -> Result<Vec<u8>, Error> {
    let mut lengths = vec![0u8; alphabet];
    let count = bits.read(1)? + 1;
    let wide = bits.read(1)?;
    let width = if wide == 1 { 8 } else { 1 };
    let first = bits.read(width)?;
    put(&mut lengths, first)?;
    if count == 2 {
        put(&mut lengths, bits.read(8)?)?;
    }
    Ok(lengths)
}

fn put(lengths: &mut [u8], symbol: u32) -> Result<(), Error> {
    let index = usize::try_from(symbol).map_err(|_| Error::TooLarge)?;
    let Some(slot) = lengths.get_mut(index) else {
        return Err(Error::Inconsistent("a prefix symbol is out of range"));
    };
    *slot = 1;
    Ok(())
}

/// Order of the 19 code-length codes. WebP lossless 6.2.1.
const LENGTH_ORDER: [usize; 19] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

fn normal(bits: &mut Bits<'_>, alphabet: usize) -> Result<Vec<u8>, Error> {
    let listed = usize::try_from(4 + bits.read(4)?).expect("a nibble plus four fits");
    let mut meta_lengths = [0u8; 19];
    for slot in LENGTH_ORDER.iter().take(listed) {
        let len = bits.read(3)?;
        meta_lengths[*slot] = u8::try_from(len).expect("three bits fit in a byte");
    }
    let meta = build(&meta_lengths)?;
    let max_symbol = if bits.read(1)? == 0 {
        alphabet
    } else {
        let width = 2 + 2 * bits.read(3)?;
        let max_symbol = 2 + bits.read(width)?;
        let alphabet_u = u32::try_from(alphabet).unwrap_or(u32::MAX);
        if max_symbol > alphabet_u {
            return Err(Error::Inconsistent("a prefix alphabet is too large"));
        }
        usize::try_from(max_symbol).map_err(|_| Error::TooLarge)?
    };
    let mut lengths = vec![0u8; max_symbol];
    let mut prev: Option<u8> = None;
    let mut index = 0usize;
    while index < max_symbol {
        match meta.decode(bits)? {
            symbol @ 0..=15 => {
                let len = u8::try_from(symbol).expect("a code length fits in a byte");
                lengths[index] = len;
                if len != 0 {
                    prev = Some(len);
                }
                index += 1;
            }
            16 => {
                let value = prev.unwrap_or(8);
                let times = usize::try_from(3 + bits.read(2)?).expect("a repeat fits");
                repeat(&mut lengths, &mut index, times, value)?;
                prev = Some(value);
            }
            17 => {
                let times = usize::try_from(3 + bits.read(3)?).expect("a repeat fits");
                repeat(&mut lengths, &mut index, times, 0)?;
            }
            18 => {
                let times = usize::try_from(11 + bits.read(7)?).expect("a repeat fits");
                repeat(&mut lengths, &mut index, times, 0)?;
            }
            _ => return Err(Error::Inconsistent("a code length is out of range")),
        }
    }
    Ok(lengths)
}

/// Writes `times` copies of `value`, or errors if that passes `lengths`.
fn repeat(lengths: &mut [u8], index: &mut usize, times: usize, value: u8) -> Result<(), Error> {
    for _ in 0..times {
        let Some(slot) = lengths.get_mut(*index) else {
            return Err(Error::Inconsistent("a code length runs past the alphabet"));
        };
        *slot = value;
        *index += 1;
    }
    Ok(())
}

/// Builds a canonical tree from code lengths. A missing tail is length 0.
pub(crate) fn build(lengths: &[u8]) -> Result<Huff, Error> {
    let mut count = [0u16; 16];
    let mut nonzero = 0u16;
    let mut only = 0u16;
    let mut max_len = 0u8;
    for (symbol, &len) in lengths.iter().enumerate() {
        if len == 0 {
            continue;
        }
        if len > 15 {
            return Err(Error::Inconsistent("a prefix code is longer than 15 bits"));
        }
        count[usize::from(len)] = count[usize::from(len)]
            .checked_add(1)
            .ok_or(Error::Inconsistent("a prefix code overflows"))?;
        nonzero = nonzero
            .checked_add(1)
            .ok_or(Error::Inconsistent("a prefix code overflows"))?;
        only = u16::try_from(symbol).map_err(|_| Error::TooLarge)?;
        max_len = max_len.max(len);
    }
    if nonzero == 0 {
        return Err(Error::Inconsistent("a prefix code is empty"));
    }
    if nonzero == 1 {
        if lengths.get(usize::from(only)).copied() != Some(1) {
            return Err(Error::Inconsistent("a prefix code is not a full tree"));
        }
        return Ok(leaf(only));
    }
    // Sum of 2^(15-len) is 2^15 exactly when the tree is full.
    let mut left = 1u32 << 15;
    for (len, &n) in count.iter().enumerate().skip(1) {
        let used = u32::from(n) << (15 - len);
        if used > left {
            return Err(Error::Inconsistent("a prefix code is not a full tree"));
        }
        left -= used;
    }
    if left != 0 {
        return Err(Error::Inconsistent("a prefix code is not a full tree"));
    }
    let mut min = [0u32; 16];
    let mut code = 0u32;
    for len in 1..16 {
        code = code
            .checked_add(u32::from(count[len - 1]))
            .and_then(|sum| sum.checked_shl(1))
            .ok_or(Error::Inconsistent("a prefix code overflows"))?;
        min[len] = code;
    }
    let mut symbols = vec![0u16; usize::from(nonzero)];
    let mut base = [0u16; 16];
    let mut running = 0u16;
    for len in 1..16 {
        base[len] = running;
        running = running
            .checked_add(count[len])
            .ok_or(Error::Inconsistent("a prefix code overflows"))?;
    }
    let mut cursor = base;
    for (symbol, &len) in lengths.iter().enumerate() {
        if len == 0 {
            continue;
        }
        let at = usize::from(cursor[usize::from(len)]);
        let Some(slot) = symbols.get_mut(at) else {
            return Err(Error::Inconsistent("a prefix code overflows"));
        };
        *slot = u16::try_from(symbol).map_err(|_| Error::TooLarge)?;
        cursor[usize::from(len)] = cursor[usize::from(len)]
            .checked_add(1)
            .ok_or(Error::Inconsistent("a prefix code overflows"))?;
    }
    Ok(Huff {
        single: None,
        max_len,
        min,
        count,
        base,
        symbols,
    })
}

fn leaf(symbol: u16) -> Huff {
    Huff {
        single: Some(symbol),
        max_len: 0,
        min: [0; 16],
        count: [0; 16],
        base: [0; 16],
        symbols: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::{build, read_lengths};
    use crate::bits::{Bits, Writer};
    use crate::error::Error;

    fn stream(write: impl FnOnce(&mut Writer)) -> Vec<u8> {
        let mut writer = Writer::new();
        write(&mut writer);
        writer.finish()
    }

    #[test]
    fn an_empty_code_is_refused() {
        assert!(matches!(
            build(&[0, 0]),
            Err(Error::Inconsistent("a prefix code is empty"))
        ));
    }

    #[test]
    fn three_length_one_symbols_are_not_a_full_tree() {
        assert!(matches!(
            build(&[1, 1, 1]),
            Err(Error::Inconsistent("a prefix code is not a full tree"))
        ));
    }

    #[test]
    fn a_single_symbol_must_have_length_one() {
        assert!(matches!(
            build(&[2]),
            Err(Error::Inconsistent("a prefix code is not a full tree"))
        ));
    }

    #[test]
    fn a_single_leaf_consumes_no_bits() {
        let huff = build(&[1, 0]).expect("one leaf");
        let mut bits = Bits::new(&[0x01]);
        assert_eq!(huff.decode(&mut bits).expect("symbol"), 0);
        assert_eq!(bits.read(8).expect("untouched"), 1);
    }

    #[test]
    fn the_lower_symbol_takes_code_zero() {
        let huff = build(&[1, 1]).expect("two leaves");
        let bytes = stream(|w| {
            w.bit(0);
            w.bit(1);
        });
        let mut bits = Bits::new(&bytes);
        assert_eq!(huff.decode(&mut bits).expect("first"), 0);
        assert_eq!(huff.decode(&mut bits).expect("second"), 1);
    }

    #[test]
    fn longer_codes_start_after_the_shorter_ones() {
        let huff = build(&[1, 2, 2]).expect("full tree");
        let bytes = stream(|w| {
            w.bit(0);
            w.bit(1);
            w.bit(0);
            w.bit(1);
            w.bit(1);
        });
        let mut bits = Bits::new(&bytes);
        assert_eq!(huff.decode(&mut bits).expect("zero"), 0);
        assert_eq!(huff.decode(&mut bits).expect("one"), 1);
        assert_eq!(huff.decode(&mut bits).expect("two"), 2);
    }

    #[test]
    fn a_simple_leaf_names_its_symbol() {
        let bytes = stream(|w| {
            w.bit(1);
            w.bit(0);
            w.bit(1);
            w.bits(5, 8);
        });
        let lengths = read_lengths(&mut Bits::new(&bytes), 256).expect("simple");
        assert_eq!(lengths.iter().filter(|len| **len != 0).count(), 1);
        assert_eq!(lengths[5], 1);
        let huff = build(&lengths).expect("leaf");
        let mut bits = Bits::new(&[0xff]);
        assert_eq!(huff.decode(&mut bits).expect("symbol"), 5);
        assert_eq!(bits.read(8).expect("untouched"), 0xff);
    }

    #[test]
    fn a_simple_symbol_past_the_alphabet_is_refused() {
        let bytes = stream(|w| {
            w.bit(1);
            w.bit(0);
            w.bit(1);
            w.bits(5, 8);
        });
        assert!(matches!(
            read_lengths(&mut Bits::new(&bytes), 4),
            Err(Error::Inconsistent("a prefix symbol is out of range"))
        ));
    }

    #[test]
    fn two_simple_symbols_share_one_bit() {
        let bytes = stream(|w| {
            w.bit(1);
            w.bit(1);
            w.bit(0);
            w.bit(0);
            w.bits(5, 8);
            w.bit(1);
        });
        let mut bits = Bits::new(&bytes);
        let lengths = read_lengths(&mut bits, 256).expect("simple");
        let huff = build(&lengths).expect("tree");
        assert_eq!(huff.decode(&mut bits).expect("larger"), 5);
    }

    #[test]
    fn a_repeat_with_no_previous_length_uses_eight() {
        let bytes = stream(|w| {
            w.bit(0);
            w.bits(5, 4);
            for _ in 0..8 {
                w.bits(0, 3);
            }
            w.bits(1, 3);
            w.bit(1);
            w.bits(0, 3);
            w.bits(1, 2);
            w.bits(0, 2);
        });
        let lengths = read_lengths(&mut Bits::new(&bytes), 256).expect("lengths");
        assert_eq!(lengths, [8, 8, 8]);
        assert!(matches!(
            build(&lengths),
            Err(Error::Inconsistent("a prefix code is not a full tree"))
        ));
    }

    #[test]
    fn a_repeat_copies_the_previous_length() {
        let bytes = stream(|w| {
            w.bit(0);
            w.bits(5, 4);
            for len in [0, 0, 0, 0, 1, 0, 0, 0, 1] {
                w.bits(len, 3);
            }
            w.bit(1);
            w.bits(0, 3);
            w.bits(2, 2);
            w.bit(0);
            w.bit(1);
            w.bits(0, 2);
        });
        let lengths = read_lengths(&mut Bits::new(&bytes), 256).expect("lengths");
        assert_eq!(&lengths[..4], &[2, 2, 2, 2]);
        let huff = build(&lengths).expect("tree");
        let data = stream(|w| {
            w.bit(1);
            w.bit(0);
        });
        assert_eq!(huff.decode(&mut Bits::new(&data)).expect("symbol"), 2);
    }

    #[test]
    fn a_run_of_zeros_places_the_next_length() {
        let bytes = stream(|w| {
            w.bit(0);
            w.bits(0, 4);
            for len in [0, 1, 0, 1] {
                w.bits(len, 3);
            }
            w.bit(1);
            w.bits(1, 3);
            w.bits(10, 4);
            w.bit(1);
            w.bits(0, 7);
            w.bit(0);
        });
        let lengths = read_lengths(&mut Bits::new(&bytes), 256).expect("lengths");
        assert_eq!(lengths.iter().position(|len| *len == 1), Some(11));
        let huff = build(&lengths).expect("leaf");
        let mut bits = Bits::new(&[0x01]);
        assert_eq!(huff.decode(&mut bits).expect("symbol"), 11);
        assert_eq!(bits.read(8).expect("untouched"), 1);
    }

    #[test]
    fn an_explicit_alphabet_past_the_limit_is_refused() {
        let bytes = stream(|w| {
            w.bit(0);
            w.bits(0, 4);
            for len in [0, 0, 1, 0] {
                w.bits(len, 3);
            }
            w.bit(1);
            w.bits(0, 3);
            w.bits(3, 2);
        });
        assert!(matches!(
            read_lengths(&mut Bits::new(&bytes), 4),
            Err(Error::Inconsistent("a prefix alphabet is too large"))
        ));
    }
}

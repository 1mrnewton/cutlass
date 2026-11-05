//! Canonical prefix codes for VP8L (RFC 9649 section 3.7.2).
//!
//! A code is built from code lengths the same way as DEFLATE: shorter codes
//! first, ties in symbol order. VP8L reads bits least significant first, so
//! a table is indexed by the bit-reversed code. A code with exactly one
//! symbol takes zero bits. Any other code must be complete: an
//! over-subscribed or an incomplete set of lengths is malformed.

use cutlass_decode::CodecError;

use super::bits::BitReader;

/// Longest code VP8L allows.
pub(crate) const MAX_CODE_LENGTH: usize = 15;

/// Bits looked up in one step. Longer codes go through a second table.
const ROOT_BITS: u32 = 8;

/// One decoding entry: a symbol and its code length, or a link to a
/// second-level table.
#[derive(Clone, Copy, Default)]
struct Entry {
    /// Symbol, or the offset of the second-level table.
    value: u32,
    /// Code length, or `ROOT_BITS` plus the second-level index width.
    bits: u8,
    link: bool,
}

/// Decoding table for one prefix code.
pub(crate) struct Code {
    entries: Vec<Entry>,
    /// The one symbol of a zero-bit code.
    single: Option<u16>,
}

impl Code {
    /// Entries this code holds, for the image-wide table budget.
    pub(crate) fn size(&self) -> usize {
        self.entries.len()
    }

    /// Build from one length per symbol. Lengths above 15 are malformed.
    pub(crate) fn from_lengths(lengths: &[u8], at_bit: u64) -> Result<Self, CodecError> {
        let malformed = CodecError::Malformed {
            at_bit,
            what: "vp8l prefix code",
        };
        let mut count = [0u32; MAX_CODE_LENGTH + 1];
        let mut used = 0usize;
        let mut last = 0usize;
        for (symbol, &length) in lengths.iter().enumerate() {
            let length = usize::from(length);
            if length > MAX_CODE_LENGTH {
                return Err(malformed);
            }
            if length > 0 {
                count[length] += 1;
                used += 1;
                last = symbol;
            }
        }
        if used == 0 {
            return Err(malformed);
        }
        if used == 1 {
            let symbol = u16::try_from(last).map_err(|_| malformed)?;
            return Ok(Self {
                entries: Vec::new(),
                single: Some(symbol),
            });
        }
        let mut left = 1i64;
        for &n in &count[1..] {
            left = left * 2 - i64::from(n);
            if left < 0 {
                return Err(malformed);
            }
        }
        if left != 0 {
            return Err(malformed);
        }
        // First canonical code of each length.
        let mut next = [0u32; MAX_CODE_LENGTH + 2];
        let mut code = 0u32;
        for length in 1..=MAX_CODE_LENGTH {
            code = (code + count[length - 1]) << 1;
            next[length] = code;
        }
        let root_size = 1usize << ROOT_BITS;
        // Widest second-level table under each root prefix.
        let mut wide = [0u32; 1 << ROOT_BITS];
        let mut codes = Vec::with_capacity(used);
        for (symbol, &length) in lengths.iter().enumerate() {
            if length == 0 {
                continue;
            }
            let length = u32::from(length);
            let slot = &mut next[length as usize];
            let reversed = reverse(*slot, length);
            *slot += 1;
            if length > ROOT_BITS {
                let prefix = (reversed & ((1 << ROOT_BITS) - 1)) as usize;
                wide[prefix] = wide[prefix].max(length - ROOT_BITS);
            }
            codes.push((symbol, length, reversed));
        }
        let mut offsets = [0usize; 1 << ROOT_BITS];
        let mut total = root_size;
        for (prefix, &bits) in wide.iter().enumerate() {
            if bits > 0 {
                offsets[prefix] = total;
                total += 1usize << bits;
            }
        }
        let mut entries = vec![Entry::default(); total];
        for (prefix, &bits) in wide.iter().enumerate() {
            if bits > 0 {
                entries[prefix] = Entry {
                    value: u32::try_from(offsets[prefix]).map_err(|_| malformed)?,
                    bits: u8::try_from(ROOT_BITS + bits).map_err(|_| malformed)?,
                    link: true,
                };
            }
        }
        for (symbol, length, reversed) in codes {
            let value = u32::try_from(symbol).map_err(|_| malformed)?;
            let bits = u8::try_from(length).map_err(|_| malformed)?;
            if length <= ROOT_BITS {
                let step = 1usize << length;
                let mut index = reversed as usize;
                while index < root_size {
                    entries[index] = Entry {
                        value,
                        bits,
                        link: false,
                    };
                    index += step;
                }
            } else {
                let prefix = (reversed & ((1 << ROOT_BITS) - 1)) as usize;
                let sub_bits = wide[prefix];
                let sub_length = length - ROOT_BITS;
                let base = offsets[prefix];
                let step = 1usize << sub_length;
                let mut index = (reversed >> ROOT_BITS) as usize;
                while index < (1usize << sub_bits) {
                    entries[base + index] = Entry {
                        value,
                        bits,
                        link: false,
                    };
                    index += step;
                }
            }
        }
        Ok(Self {
            entries,
            single: None,
        })
    }

    /// Read one symbol.
    #[inline]
    pub(crate) fn read(&self, reader: &mut BitReader<'_>) -> Result<u16, CodecError> {
        if let Some(symbol) = self.single {
            return Ok(symbol);
        }
        let bits = reader.peek(MAX_CODE_LENGTH as u32);
        let mut entry = self.entries[(bits & ((1 << ROOT_BITS) - 1)) as usize];
        if entry.link {
            let sub = (bits >> ROOT_BITS) & ((1 << (u32::from(entry.bits) - ROOT_BITS)) - 1);
            let index = entry.value as usize + sub as usize;
            entry = self.entries.get(index).copied().unwrap_or_default();
        }
        reader.skip(u32::from(entry.bits))?;
        Ok(entry.value as u16)
    }
}

/// `code` with its low `length` bits in reverse order.
pub(crate) fn reverse(code: u32, length: u32) -> u32 {
    if length == 0 {
        return 0;
    }
    code.reverse_bits() >> (32 - length)
}

#[cfg(test)]
mod tests {
    use super::{Code, reverse};
    use crate::lossless::bits::{BitReader, BitWriter};

    #[test]
    fn reverse_flips_the_low_bits() {
        assert_eq!(reverse(0b001, 3), 0b100);
        assert_eq!(reverse(0b1101, 4), 0b1011);
        assert_eq!(reverse(0, 0), 0);
    }

    #[test]
    fn a_canonical_code_reads_back_each_symbol() {
        // Lengths of the DEFLATE example: A..H = 3,3,3,3,3,2,4,4.
        let lengths = [3, 3, 3, 3, 3, 2, 4, 4];
        let code = Code::from_lengths(&lengths, 0).unwrap();
        // Canonical codes: F=00, A=010, B=011, C=100, D=101, E=110, G=1110, H=1111.
        let codes: [(u16, u32, u32); 8] = [
            (5, 0b00, 2),
            (0, 0b010, 3),
            (1, 0b011, 3),
            (2, 0b100, 3),
            (3, 0b101, 3),
            (4, 0b110, 3),
            (6, 0b1110, 4),
            (7, 0b1111, 4),
        ];
        let mut writer = BitWriter::default();
        for &(_, bits, length) in &codes {
            writer.write(reverse(bits, length), length);
        }
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        for &(symbol, _, _) in &codes {
            assert_eq!(code.read(&mut reader).unwrap(), symbol);
        }
    }

    #[test]
    fn long_codes_go_through_the_second_table() {
        // 1, 2, ..., 14, 15, 15: complete and fifteen bits deep.
        let mut lengths: Vec<u8> = (1..=15).collect();
        lengths.push(15);
        let code = Code::from_lengths(&lengths, 0).unwrap();
        let mut writer = BitWriter::default();
        // Symbol 15 is the all-ones code of length 15; symbol 14 is 1..10.
        writer.write(reverse(0x7fff, 15), 15);
        writer.write(reverse(0x7ffe, 15), 15);
        writer.write(reverse(0b10, 2), 2);
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(code.read(&mut reader).unwrap(), 15);
        assert_eq!(code.read(&mut reader).unwrap(), 14);
        assert_eq!(code.read(&mut reader).unwrap(), 1);
    }

    #[test]
    fn one_symbol_takes_no_bits() {
        let code = Code::from_lengths(&[0, 0, 7, 0], 0).unwrap();
        let mut reader = BitReader::new(&[]);
        assert_eq!(code.read(&mut reader).unwrap(), 2);
        assert_eq!(reader.position(), 0);
    }

    #[test]
    fn over_subscribed_incomplete_and_empty_codes_are_refused() {
        assert!(Code::from_lengths(&[1, 1, 1], 0).is_err());
        assert!(Code::from_lengths(&[2, 2, 2], 0).is_err());
        assert!(Code::from_lengths(&[0, 0, 0], 0).is_err());
        assert!(Code::from_lengths(&[16, 1], 0).is_err());
    }
}

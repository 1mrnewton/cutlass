//! Huffman decode. T.81 Annex C.

use crate::bits::Bits;
use crate::error::Error;

pub(crate) struct Huff {
    /// Smallest code of each length. Index 0 is length 1.
    min_code: [i32; 16],
    /// Largest code of each length, or -1 when that length has no codes.
    max_code: [i32; 16],
    /// Index of the first symbol of that length.
    index: [u16; 16],
    symbols: Vec<u8>,
}

impl Huff {
    /// T.81 C.2. Codes are assigned in symbol order, short lengths first.
    /// A code of all ones is reserved for byte padding.
    pub(crate) fn build(counts: &[u8; 16], symbols: &[u8]) -> Result<Self, Error> {
        let total: usize = counts.iter().copied().map(usize::from).sum();
        if total != symbols.len() {
            return Err(Error::Inconsistent(
                "a Huffman table length does not match its counts",
            ));
        }
        if total > 256 {
            return Err(Error::Inconsistent("a Huffman table has too many symbols"));
        }
        let mut code = 0i32;
        let mut min_code = [0i32; 16];
        let mut max_code = [-1i32; 16];
        let mut index = [0u16; 16];
        let mut next = 0u16;
        for len in 0..16 {
            let count = i32::from(counts[len]);
            if count > 0 {
                let limit = 1i32 << (len + 1);
                if code < 0 || count > limit - code {
                    return Err(Error::Inconsistent(
                        "a Huffman code does not fit its length",
                    ));
                }
                min_code[len] = code;
                max_code[len] = code + count - 1;
                index[len] = next;
                next = next
                    .checked_add(u16::try_from(count).expect("a Huffman count fits"))
                    .ok_or(Error::Inconsistent("a Huffman table has too many symbols"))?;
                code += count;
            }
            if max_code[len] >= 0 {
                let all_ones = (1i32 << (len + 1)) - 1;
                if max_code[len] >= all_ones {
                    return Err(Error::Inconsistent("a Huffman code is all ones"));
                }
            }
            if len != 15 {
                code = code.checked_shl(1).ok_or(Error::Inconsistent(
                    "a Huffman code does not fit its length",
                ))?;
            }
        }
        Ok(Self {
            min_code,
            max_code,
            index,
            symbols: symbols.to_vec(),
        })
    }

    /// T.81 F.2.2.3. At most 16 bits. A code that is not in the table is an error.
    pub(crate) fn decode(&self, bits: &mut Bits<'_>) -> Result<u8, Error> {
        let mut code = 0i32;
        for len in 0..16 {
            code = (code << 1) | i32::from(bits.bit()?);
            if code <= self.max_code[len] {
                let delta = code - self.min_code[len];
                let at = usize::from(self.index[len])
                    + usize::try_from(delta)
                        .map_err(|_| Error::Inconsistent("a Huffman code is not in the table"))?;
                return self
                    .symbols
                    .get(at)
                    .copied()
                    .ok_or(Error::Inconsistent("a Huffman code is not in the table"));
            }
        }
        Err(Error::Inconsistent("a Huffman code is not in the table"))
    }
}

#[cfg(test)]
mod tests {
    use super::Huff;
    use crate::bits::Bits;

    fn dc_table() -> Huff {
        let mut counts = [0u8; 16];
        counts[3] = 12;
        let symbols: Vec<u8> = (0..12).collect();
        Huff::build(&counts, &symbols).expect("the DC table builds")
    }

    #[test]
    fn a_length_four_code_is_the_symbol() {
        let table = dc_table();
        // 0101 0000 is symbol 5, then symbol 0.
        let mut bits = Bits::new(&[0x50], 0);
        assert_eq!(table.decode(&mut bits).expect("first"), 5);
        assert_eq!(table.decode(&mut bits).expect("second"), 0);
    }

    #[test]
    fn an_all_ones_code_is_rejected() {
        let mut counts = [0u8; 16];
        counts[0] = 2;
        assert!(matches!(
            Huff::build(&counts, &[0, 1]),
            Err(crate::Error::Inconsistent("a Huffman code is all ones"))
        ));
    }
}

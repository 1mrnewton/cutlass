//! VP8L prefix codes (RFC 9649 section 3.7.2).
//!
//! A code is canonical: lengths alone define it, shorter codes first, and
//! within one length the smaller symbol first. Code bits arrive most
//! significant first inside the least-significant-first stream, so the
//! first bit read is the top bit of the code. A code with one used symbol
//! takes no bits. A code whose lengths leave the tree over- or
//! under-full is malformed.

use cutlass_decode::CodecError;

use crate::bits::BitReader;

/// Longest code length the format allows.
const MAX_LENGTH: usize = 15;
/// Bits resolved by one table lookup; longer codes walk the rest.
const ROOT_BITS: u32 = 8;

/// One decoding table.
#[derive(Clone, Debug)]
pub(crate) struct Code {
    /// Symbol when only one is used.
    single: Option<u16>,
    /// Indexed by the next `ROOT_BITS` bits: `(symbol, length)`, length 0
    /// meaning the code is longer than the root.
    root: Vec<(u16, u8)>,
    /// How many codes have each length.
    counts: [u16; MAX_LENGTH + 1],
    /// Symbols in canonical order.
    symbols: Vec<u16>,
}

impl Code {
    /// Build from a code length per symbol.
    pub(crate) fn from_lengths(lengths: &[u8]) -> Result<Self, CodecError> {
        let mut counts = [0_u16; MAX_LENGTH + 1];
        for &length in lengths {
            let length = usize::from(length);
            if length > MAX_LENGTH {
                return Err(malformed("vp8l code length"));
            }
            counts[length] += 1;
        }
        counts[0] = 0;
        let used: u32 = counts.iter().map(|&c| u32::from(c)).sum();
        if used == 0 {
            return Err(malformed("vp8l empty prefix code"));
        }
        if used == 1 {
            let symbol = lengths.iter().position(|&l| l > 0).unwrap_or(0);
            return Ok(Self::single(u16::try_from(symbol).unwrap_or(0)));
        }
        // The lengths must fill the code space exactly.
        let mut left: i32 = 1;
        for &count in &counts[1..] {
            left = left * 2 - i32::from(count);
            if left < 0 {
                return Err(malformed("vp8l over-full prefix code"));
            }
        }
        if left != 0 {
            return Err(malformed("vp8l incomplete prefix code"));
        }
        let mut offsets = [0_usize; MAX_LENGTH + 2];
        for length in 1..=MAX_LENGTH {
            offsets[length + 1] = offsets[length] + usize::from(counts[length]);
        }
        let mut symbols = vec![0_u16; offsets[MAX_LENGTH + 1]];
        for (symbol, &length) in lengths.iter().enumerate() {
            if length > 0 {
                let slot = &mut offsets[usize::from(length)];
                symbols[*slot] = u16::try_from(symbol).unwrap_or(0);
                *slot += 1;
            }
        }
        let mut root = vec![(0_u16, 0_u8); 1 << ROOT_BITS];
        let mut code = 0_u32;
        let mut index = 0;
        for length in 1..=ROOT_BITS {
            for _ in 0..counts[length as usize] {
                // The table is indexed by bits as read: reverse the code.
                let reversed = code.reverse_bits() >> (32 - length);
                let mut slot = reversed as usize;
                while slot < root.len() {
                    root[slot] = (symbols[index], u8::try_from(length).unwrap_or(0));
                    slot += 1 << length;
                }
                code += 1;
                index += 1;
            }
            code <<= 1;
        }
        Ok(Self {
            single: None,
            root,
            counts,
            symbols,
        })
    }

    pub(crate) fn single(symbol: u16) -> Self {
        Self {
            single: Some(symbol),
            root: Vec::new(),
            counts: [0; MAX_LENGTH + 1],
            symbols: Vec::new(),
        }
    }

    /// Stored entries, for the decoder's table budget.
    pub(crate) fn entries(&self) -> usize {
        self.root.len() + self.symbols.len() + 1
    }

    pub(crate) fn read(&self, reader: &mut BitReader<'_>) -> Result<u16, CodecError> {
        if let Some(symbol) = self.single {
            return Ok(symbol);
        }
        let peeked = reader.peek(ROOT_BITS) as usize;
        if let Some(&(symbol, length)) = self.root.get(peeked)
            && length > 0
        {
            reader.consume(u32::from(length))?;
            return Ok(symbol);
        }
        // Canonical walk, one bit at a time.
        let mut code: i32 = 0;
        let mut first: i32 = 0;
        let mut index: i32 = 0;
        for length in 1..=MAX_LENGTH {
            code |= i32::from(reader.read_flag()?);
            let count = i32::from(self.counts[length]);
            if code - first < count {
                let at = usize::try_from(index + code - first).unwrap_or(usize::MAX);
                return self
                    .symbols
                    .get(at)
                    .copied()
                    .ok_or_else(|| malformed("vp8l prefix code"));
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(malformed("vp8l prefix code"))
    }
}

/// Order the code-length code lengths are stored in.
const CODE_LENGTH_ORDER: [usize; 19] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

/// Read one prefix code over `alphabet` symbols (section 3.7.2.1).
pub(crate) fn read_code(reader: &mut BitReader<'_>, alphabet: usize) -> Result<Code, CodecError> {
    if reader.read_flag()? {
        // Simple code: one or two symbols, each of length one.
        let two = reader.read_flag()?;
        let first_bits = if reader.read_flag()? { 8 } else { 1 };
        let first = reader.read(first_bits)? as usize;
        let mut lengths = vec![0_u8; alphabet];
        *lengths
            .get_mut(first)
            .ok_or_else(|| malformed("vp8l simple code symbol"))? = 1;
        if two {
            let second = reader.read(8)? as usize;
            *lengths
                .get_mut(second)
                .ok_or_else(|| malformed("vp8l simple code symbol"))? = 1;
            if second == first {
                return Ok(Code::single(u16::try_from(first).unwrap_or(0)));
            }
            return Code::from_lengths(&lengths);
        }
        return Ok(Code::single(u16::try_from(first).unwrap_or(0)));
    }
    let count = 4 + reader.read(4)? as usize;
    let mut code_length_lengths = [0_u8; 19];
    for &symbol in CODE_LENGTH_ORDER.iter().take(count) {
        code_length_lengths[symbol] = u8::try_from(reader.read(3)?).unwrap_or(0);
    }
    let length_code = Code::from_lengths(&code_length_lengths)?;
    let mut max_symbol = alphabet;
    if reader.read_flag()? {
        let bits = 2 + 2 * reader.read(3)?;
        max_symbol = 2 + reader.read(bits)? as usize;
        if max_symbol > alphabet {
            return Err(malformed("vp8l max symbol"));
        }
    }
    let mut lengths = vec![0_u8; alphabet];
    let mut symbol = 0;
    let mut previous = 8_u8;
    while symbol < alphabet {
        if max_symbol == 0 {
            break;
        }
        max_symbol -= 1;
        let value = length_code.read(reader)?;
        if value < 16 {
            let length = u8::try_from(value).unwrap_or(0);
            lengths[symbol] = length;
            symbol += 1;
            if length != 0 {
                previous = length;
            }
            continue;
        }
        let (repeat, fill) = match value {
            16 => (3 + reader.read(2)? as usize, previous),
            17 => (3 + reader.read(3)? as usize, 0),
            _ => (11 + reader.read(7)? as usize, 0),
        };
        if symbol + repeat > alphabet {
            return Err(malformed("vp8l code length repeat"));
        }
        lengths[symbol..symbol + repeat].fill(fill);
        symbol += repeat;
    }
    Code::from_lengths(&lengths)
}

fn malformed(what: &'static str) -> CodecError {
    CodecError::Malformed { at_bit: 0, what }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack `(value, bits)` least significant first, as VP8L stores them.
    pub(crate) fn pack(fields: &[(u32, u32)]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut held = 0_u64;
        let mut count = 0;
        for &(value, bits) in fields {
            held |= u64::from(value) << count;
            count += bits;
            while count >= 8 {
                out.push((held & 0xFF) as u8);
                held >>= 8;
                count -= 8;
            }
        }
        if count > 0 {
            out.push(held as u8);
        }
        out
    }

    /// A canonical code word written in read order: top bit first.
    fn word(code: u32, length: u32) -> (u32, u32) {
        (code.reverse_bits() >> (32 - length), length)
    }

    #[test]
    fn canonical_codes_decode_short_and_long_words() {
        // Lengths: A=1, B=2, C=3, and twelve symbols of length 7 to 15.
        let mut lengths = vec![1_u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15];
        lengths.resize(20, 0);
        let code = Code::from_lengths(&lengths).unwrap();
        // Canonical codes: 0, 10, 110, ..., 111111111111110, 111111111111111.
        let mut fields = Vec::new();
        for (symbol, &length) in lengths.iter().enumerate().take(16) {
            let length = u32::from(length);
            let value = if symbol == 15 {
                (1 << 15) - 1
            } else {
                ((1 << length) - 1) & !1
            };
            fields.push(word(value, length));
        }
        let bytes = pack(&fields);
        let mut reader = BitReader::new(&bytes);
        for symbol in 0..16 {
            assert_eq!(code.read(&mut reader).unwrap(), symbol);
        }
    }

    #[test]
    fn bad_lengths_are_refused() {
        assert!(Code::from_lengths(&[0, 0, 0]).is_err());
        assert!(Code::from_lengths(&[1, 1, 1]).is_err(), "over-full");
        assert!(Code::from_lengths(&[1, 2, 0]).is_err(), "incomplete");
        assert!(Code::from_lengths(&[16, 1]).is_err());
        let single = Code::from_lengths(&[0, 0, 4, 0]).unwrap();
        let mut reader = BitReader::new(&[]);
        assert_eq!(single.read(&mut reader).unwrap(), 2);
    }

    #[test]
    fn simple_and_normal_codes_read_from_the_stream() {
        // Simple, two symbols: 8-bit first symbol 200, second 7.
        let bytes = pack(&[(1, 1), (1, 1), (1, 1), (200, 8), (7, 8), (1, 1), (0, 1)]);
        let mut reader = BitReader::new(&bytes);
        let code = read_code(&mut reader, 256).unwrap();
        // Canonical order: the smaller symbol, 7, is code 0.
        assert_eq!(code.read(&mut reader).unwrap(), 200);
        assert_eq!(code.read(&mut reader).unwrap(), 7);
        // Simple, one 1-bit symbol: takes no bits to read.
        let bytes = pack(&[(1, 1), (0, 1), (0, 1), (1, 1)]);
        let mut reader = BitReader::new(&bytes);
        assert_eq!(
            read_code(&mut reader, 40)
                .unwrap()
                .read(&mut reader)
                .unwrap(),
            1
        );
        // A simple symbol beyond the alphabet is malformed.
        let bytes = pack(&[(1, 1), (0, 1), (1, 1), (45, 8)]);
        assert!(read_code(&mut BitReader::new(&bytes), 40).is_err());

        // Normal code over 4 symbols, lengths [1, 2, 3, 3], spelled with a
        // code-length code where symbols 1, 2, 3 each have length 2 and
        // 0 has length 2: four code-length codes of two bits.
        // Stored order starts 17, 18, 0, 1, 2, 3: six entries (count 4 + 2).
        let mut fields = vec![(0, 1), (2, 4)];
        for length in [0, 0, 2, 2, 2, 2] {
            fields.push((length, 3));
        }
        fields.push((0, 1)); // no max symbol
        // Code-length symbols 0..3 have canonical codes 00, 01, 10, 11.
        for length in [1_u32, 2, 3, 3] {
            fields.push(word(length, 2));
        }
        // Then the symbols 3, 0, 2, 1 with codes 111, 0, 110, 10.
        fields.extend([word(7, 3), word(0, 1), word(6, 3), word(2, 2)]);
        let bytes = pack(&fields);
        let mut reader = BitReader::new(&bytes);
        let code = read_code(&mut reader, 4).unwrap();
        let read: Vec<u16> = (0..4).map(|_| code.read(&mut reader).unwrap()).collect();
        assert_eq!(read, [3, 0, 2, 1]);
    }

    #[test]
    fn repeat_codes_fill_runs() {
        // Code-length alphabet: 2 has length 1, 16 and 17 length 2.
        // Stored order 17, 18, 0, 1, 2, 3, 4, 5, 16: nine entries (4 + 5).
        let mut fields = vec![(0, 1), (5, 4)];
        for length in [2, 0, 0, 0, 1, 0, 0, 0, 2] {
            fields.push((length, 3));
        }
        fields.push((0, 1));
        // Canonical: 2 -> 0, 16 -> 10, 17 -> 11. A 2, then 16 repeats it
        // three times, then 17 writes 3 + 2 zeros: [2, 2, 2, 2, 0 x 5].
        fields.extend([word(0, 1), word(2, 2), (0, 2), word(3, 2), (2, 3)]);
        // Symbol 3 is code 11.
        fields.push(word(3, 2));
        let bytes = pack(&fields);
        let mut reader = BitReader::new(&bytes);
        let code = read_code(&mut reader, 9).unwrap();
        assert_eq!(code.read(&mut reader).unwrap(), 3);
        // A run past the alphabet is refused.
        let mut fields = vec![(0, 1), (5, 4)];
        for length in [2, 0, 0, 0, 1, 0, 0, 0, 2] {
            fields.push((length, 3));
        }
        fields.extend([(0, 1), word(0, 1), word(3, 2), (7, 3)]);
        let bytes = pack(&fields);
        assert!(read_code(&mut BitReader::new(&bytes), 9).is_err());
    }
}

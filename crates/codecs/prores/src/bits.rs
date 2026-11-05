//! MSB-first bit reading and writing, and the Rice / exp-Golomb codewords.

use cutlass_decode::CodecError;

/// Reads bits from one component's data. Past the end it reads zeros, and
/// [`Self::overrun`] says so.
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    end: usize,
    /// Bit offset of `data` in the frame, for error positions.
    base: u64,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8], base: u64) -> Self {
        Self {
            data,
            pos: 0,
            end: data.len().saturating_mul(8),
            base,
        }
    }

    /// The next 32 bits, zero-filled past the end.
    #[inline]
    pub(crate) fn peek32(&self) -> u32 {
        let byte = self.pos >> 3;
        let shift = self.pos & 7;
        let word = match self.data.get(byte..byte.saturating_add(8)) {
            Some(bytes) => match <[u8; 8]>::try_from(bytes) {
                Ok(array) => u64::from_be_bytes(array),
                Err(_) => 0,
            },
            None => self.tail_word(byte),
        };
        ((word << shift) >> 32) as u32
    }

    #[cold]
    fn tail_word(&self, byte: usize) -> u64 {
        let mut array = [0u8; 8];
        if let Some(rest) = self.data.get(byte..) {
            for (slot, value) in array.iter_mut().zip(rest) {
                *slot = *value;
            }
        }
        u64::from_be_bytes(array)
    }

    #[inline]
    pub(crate) fn skip(&mut self, bits: u32) {
        self.pos = self.pos.saturating_add(bits as usize);
    }

    /// Reads `bits` bits, 0 to 32.
    #[inline]
    pub(crate) fn read(&mut self, bits: u32) -> u32 {
        if bits == 0 {
            return 0;
        }
        let value = self.peek32() >> (32 - bits);
        self.skip(bits);
        value
    }

    /// Bits left before the end. Negative after an overrun.
    #[inline]
    pub(crate) fn left(&self) -> isize {
        (self.end as isize).saturating_sub(self.pos as isize)
    }

    /// True when a read went past the end.
    #[inline]
    pub(crate) fn overrun(&self) -> bool {
        self.pos > self.end
    }

    /// The frame bit offset of the next read.
    pub(crate) fn at(&self) -> u64 {
        self.base.saturating_add(self.pos as u64)
    }

    pub(crate) fn malformed(&self, what: &'static str) -> CodecError {
        CodecError::Malformed {
            at_bit: self.at(),
            what,
        }
    }

    /// True when every bit left is zero and fewer than 32 remain: the
    /// component's coefficients have ended.
    #[inline]
    pub(crate) fn at_end_of_data(&self) -> bool {
        let left = self.left();
        left <= 0 || (left < 32 && self.peek32() == 0)
    }

    /// Decodes one codeword of the RDD 36 hybrid Rice / exp-Golomb code.
    ///
    /// The codebook byte holds the Rice order in bits 7..5, the exp-Golomb
    /// order in bits 4..2, and the switch point in bits 1..0. A codeword
    /// longer than 31 bits is refused.
    #[inline]
    pub(crate) fn code(&mut self, codebook: u8) -> Result<u32, CodecError> {
        let buf = self.peek32();
        let switch = u32::from(codebook & 3);
        let rice = u32::from(codebook >> 5);
        let exp = u32::from((codebook >> 2) & 7);
        let zeros = buf.leading_zeros();
        if zeros > switch {
            let len = exp + 2 * zeros - switch;
            if len > 31 {
                return Err(self.malformed("codeword"));
            }
            let value = (buf >> (32 - len)) - (1 << exp) + ((switch + 1) << rice);
            self.skip(len);
            Ok(value)
        } else if rice > 0 {
            let len = zeros + 1 + rice;
            let value = (zeros << rice) + ((buf >> (32 - len)) & ((1 << rice) - 1));
            self.skip(len);
            Ok(value)
        } else {
            self.skip(zeros + 1);
            Ok(zeros)
        }
    }
}

/// Writes bits MSB first.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Default)]
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    acc: u64,
    filled: u32,
}

#[cfg_attr(not(test), allow(dead_code))]
impl BitWriter {
    /// Writes the low `bits` bits of `value`, 0 to 32.
    #[inline]
    pub(crate) fn put(&mut self, bits: u32, value: u32) {
        if bits == 0 {
            return;
        }
        let masked = u64::from(value) & ((1u64 << bits) - 1);
        self.acc = (self.acc << bits) | masked;
        self.filled += bits;
        while self.filled >= 8 {
            self.filled -= 8;
            self.bytes.push((self.acc >> self.filled) as u8);
        }
    }

    /// Writes `count` zero bits.
    #[inline]
    pub(crate) fn zeros(&mut self, mut count: u32) {
        while count > 0 {
            let step = count.min(32);
            self.put(step, 0);
            count -= step;
        }
    }

    /// Bits written so far.
    pub(crate) fn len_bits(&self) -> usize {
        self.bytes.len() * 8 + self.filled as usize
    }

    /// Pads with zero bits to a byte boundary and returns the bytes.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        if self.filled > 0 {
            let pad = 8 - self.filled;
            self.put(pad, 0);
        }
        self.bytes
    }

    /// Encodes one codeword, the inverse of [`BitReader::code`].
    #[inline]
    pub(crate) fn code(&mut self, codebook: u8, value: u32) {
        let (bits, zeros, payload) = code_parts(codebook, value);
        self.zeros(zeros);
        self.put(bits, payload);
    }
}

/// The codeword for `value`: `zeros` leading zeros, then `bits` bits of `payload`.
#[cfg_attr(not(test), allow(dead_code))]
#[inline]
pub(crate) fn code_parts(codebook: u8, value: u32) -> (u32, u32, u32) {
    let switch = u32::from(codebook & 3);
    let rice = u32::from(codebook >> 5);
    let exp = u32::from((codebook >> 2) & 7);
    let base = (switch + 1) << rice;
    if value < base {
        let zeros = value >> rice;
        let mask = (1u32 << rice) - 1;
        (1 + rice, zeros, (1 << rice) | (value & mask))
    } else {
        let x = value - base + (1 << exp);
        let bits = 32 - x.leading_zeros();
        let zeros = bits - exp + switch;
        (bits, zeros, x)
    }
}

/// Length of a codeword in bits.
#[cfg_attr(not(test), allow(dead_code))]
#[inline]
pub(crate) fn code_len(codebook: u8, value: u32) -> u32 {
    let (bits, zeros, _) = code_parts(codebook, value);
    bits + zeros
}

#[cfg(test)]
mod tests {
    use super::{BitReader, BitWriter, code_len};
    use crate::tables::{DC_CODEBOOK, FIRST_DC_CODEBOOK, LEVEL_CODEBOOK, RUN_CODEBOOK};

    #[test]
    fn every_codebook_round_trips_its_symbols() {
        let mut books = vec![FIRST_DC_CODEBOOK];
        books.extend_from_slice(&DC_CODEBOOK);
        books.extend_from_slice(&RUN_CODEBOOK);
        books.extend_from_slice(&LEVEL_CODEBOOK);
        for &book in &books {
            let mut writer = BitWriter::default();
            let values: Vec<u32> = (0..600)
                .chain([4_095, 16_383, 32_767])
                .filter(|&value| code_len(book, value) <= 31)
                .collect();
            assert!(values.len() >= 602, "book {book:#x}");
            let mut total = 0;
            for &value in &values {
                writer.code(book, value);
                total += code_len(book, value) as usize;
                assert_eq!(writer.len_bits(), total, "book {book:#x} value {value}");
            }
            let bytes = writer.finish();
            let mut reader = BitReader::new(&bytes, 0);
            for &value in &values {
                assert_eq!(reader.code(book).expect("code"), value, "book {book:#x}");
            }
            assert!(!reader.overrun());
        }
    }

    #[test]
    fn rice_and_exp_golomb_match_hand_coded_words() {
        // Codebook 0x04: Rice order 0 up to 0, then exp-Golomb order 1.
        // 0 is "1"; 1 is "010"; 2 is "011"; 3 is "00100".
        let bytes = [0b1010_0110, 0b0100_0000];
        let mut reader = BitReader::new(&bytes, 0);
        assert_eq!(reader.code(0x04).expect("0"), 0);
        assert_eq!(reader.code(0x04).expect("1"), 1);
        assert_eq!(reader.code(0x04).expect("2"), 2);
        assert_eq!(reader.code(0x04).expect("3"), 3);
    }

    #[test]
    fn a_run_of_zeros_is_refused_not_wrapped() {
        let bytes = [0u8; 8];
        let mut reader = BitReader::new(&bytes, 16);
        assert!(reader.code(0x04).is_err());
        let mut reader = BitReader::new(&[], 0);
        assert!(reader.code(FIRST_DC_CODEBOOK).is_err());
    }

    #[test]
    fn reads_past_the_end_are_zero_and_flagged() {
        let bytes = [0xff];
        let mut reader = BitReader::new(&bytes, 0);
        assert_eq!(reader.read(4), 0xf);
        assert!(!reader.at_end_of_data());
        assert_eq!(reader.read(8), 0xf0);
        assert!(reader.overrun());
        assert!(reader.at_end_of_data());
    }
}

//! MSB-first bit reading and writing.
//!
//! Every Layer III field is written most significant bit first
//! (ISO/IEC 11172-3 clause 2.1, `bslbf` and `uimsbf`).

use cutlass_decode::CodecError;

/// Reads bits from a byte slice, most significant bit first.
///
/// `end` is a bit limit inside the slice. A read that passes it is
/// [`CodecError::Truncated`]; a peek past it sees zero bits.
#[derive(Clone, Debug)]
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    end: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            end: data.len().saturating_mul(8),
        }
    }

    /// A reader over `data` that starts at bit `pos` and stops at bit `end`.
    pub(crate) fn window(data: &'a [u8], pos: usize, end: usize) -> Self {
        let limit = data.len().saturating_mul(8);
        Self {
            data,
            pos: pos.min(limit),
            end: end.min(limit),
        }
    }

    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    pub(crate) fn left(&self) -> usize {
        self.end.saturating_sub(self.pos)
    }

    /// The next 32 bits, left-justified. Bits past `end` read as zero.
    pub(crate) fn peek32(&self) -> u32 {
        let mut value = 0u64;
        let byte = self.pos / 8;
        for index in 0..5 {
            let next = self.data.get(byte + index).copied().unwrap_or(0);
            value = (value << 8) | u64::from(next);
        }
        let shift = self.pos % 8;
        let bits = (value >> (8 - shift)) as u32;
        let valid = self.left();
        if valid >= 32 {
            bits
        } else if valid == 0 {
            0
        } else {
            bits & !(u32::MAX >> valid)
        }
    }

    pub(crate) fn read(&mut self, count: u32, what: &'static str) -> Result<u32, CodecError> {
        if count == 0 {
            return Ok(0);
        }
        let count_usize = count as usize;
        if count > 32 || self.left() < count_usize {
            return Err(CodecError::Truncated {
                at_bit: self.pos as u64,
                what,
            });
        }
        let bits = self.peek32() >> (32 - count);
        self.pos += count_usize;
        Ok(bits)
    }

    pub(crate) fn bit(&mut self, what: &'static str) -> Result<bool, CodecError> {
        Ok(self.read(1, what)? == 1)
    }

    pub(crate) fn skip(&mut self, count: usize, what: &'static str) -> Result<(), CodecError> {
        if self.left() < count {
            return Err(CodecError::Truncated {
                at_bit: self.pos as u64,
                what,
            });
        }
        self.pos += count;
        Ok(())
    }
}

/// Appends bits to a byte vector, most significant bit first.
#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    acc: u64,
    filled: u32,
}

#[cfg(test)]
impl BitWriter {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Write the low `count` bits of `value`. `count` is at most 32.
    pub(crate) fn put(&mut self, value: u32, count: u32) {
        if count == 0 {
            return;
        }
        let count = count.min(32);
        let masked = u64::from(value) & ((1u64 << count) - 1);
        self.acc = (self.acc << count) | masked;
        self.filled += count;
        while self.filled >= 8 {
            self.filled -= 8;
            self.bytes.push((self.acc >> self.filled) as u8);
        }
        self.acc &= (1u64 << self.filled) - 1;
    }

    pub(crate) fn bits(&self) -> usize {
        self.bytes.len() * 8 + self.filled as usize
    }

    /// Pad with zero bits to the next byte and return the bytes.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        if self.filled > 0 {
            let pad = 8 - self.filled;
            self.put(0, pad);
        }
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::{BitReader, BitWriter};

    #[test]
    fn a_written_field_reads_back_at_every_offset() {
        for lead in 0..17u32 {
            let mut writer = BitWriter::new();
            writer.put(0, lead);
            writer.put(0x1_2345, 17);
            writer.put(0xDEAD_BEEF, 32);
            writer.put(1, 1);
            let bits = writer.bits();
            let bytes = writer.finish();
            assert_eq!(bits, lead as usize + 50);
            let mut reader = BitReader::new(&bytes);
            assert_eq!(reader.read(lead, "lead").expect("lead"), 0);
            assert_eq!(reader.read(17, "a").expect("a"), 0x1_2345);
            assert_eq!(reader.read(32, "b").expect("b"), 0xDEAD_BEEF);
            assert!(reader.bit("c").expect("c"));
        }
    }

    #[test]
    fn a_read_past_the_window_is_truncated_and_a_peek_sees_zeros() {
        let data = [0xFF, 0xFF, 0xFF];
        let mut reader = BitReader::window(&data, 4, 12);
        assert_eq!(reader.peek32(), 0xFF00_0000);
        assert_eq!(reader.read(8, "x").expect("eight"), 0xFF);
        assert!(reader.read(1, "x").is_err());
        assert_eq!(reader.pos(), 12);
        assert_eq!(reader.left(), 0);
    }
}

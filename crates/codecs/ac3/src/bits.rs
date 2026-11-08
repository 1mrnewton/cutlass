//! MSB-first bit reading and writing.
//!
//! Every AC-3 and E-AC-3 element arrives most significant bit first
//! (ATSC A/52:2018 clause 5.3 and Annex E clause 2.2).

use cutlass_decode::CodecError;

/// Reads bits from a byte slice, most significant bit first.
///
/// A read past the end is [`CodecError::Truncated`] and moves nothing.
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

    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    pub(crate) fn left(&self) -> usize {
        self.end.saturating_sub(self.pos)
    }

    /// `count` bits, `count` at most 32.
    pub(crate) fn read(&mut self, count: u32, what: &'static str) -> Result<u32, CodecError> {
        if count == 0 {
            return Ok(0);
        }
        let wanted = count as usize;
        if count > 32 || self.left() < wanted {
            return Err(CodecError::Truncated {
                at_bit: self.pos as u64,
                what,
            });
        }
        let byte = self.pos / 8;
        let mut window = 0u64;
        for index in 0..5 {
            let next = self.data.get(byte + index).copied().unwrap_or(0);
            window = (window << 8) | u64::from(next);
        }
        let shift = 40 - (self.pos % 8) - wanted;
        let mask = (1u64 << wanted) - 1;
        self.pos += wanted;
        Ok(((window >> shift) & mask) as u32)
    }

    pub(crate) fn bit(&mut self, what: &'static str) -> Result<bool, CodecError> {
        Ok(self.read(1, what)? == 1)
    }

    /// `count` bits as a two's complement value.
    pub(crate) fn signed(&mut self, count: u32, what: &'static str) -> Result<i32, CodecError> {
        let raw = self.read(count, what)?;
        if count == 0 {
            return Ok(0);
        }
        let shift = 32 - count;
        Ok(((raw << shift) as i32) >> shift)
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

/// Writes bits into a growing byte buffer, most significant bit first.
#[derive(Clone, Debug, Default)]
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    acc: u64,
    filled: u32,
}

impl BitWriter {
    pub(crate) fn with_capacity(bytes: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(bytes),
            acc: 0,
            filled: 0,
        }
    }

    /// The low `count` bits of `value`, `count` at most 32.
    pub(crate) fn put(&mut self, count: u32, value: u32) {
        if count == 0 {
            return;
        }
        let mask = if count >= 32 {
            u64::from(u32::MAX)
        } else {
            (1u64 << count) - 1
        };
        self.acc = (self.acc << count) | (u64::from(value) & mask);
        self.filled += count;
        while self.filled >= 8 {
            self.filled -= 8;
            self.bytes.push((self.acc >> self.filled) as u8);
        }
        self.acc &= (1u64 << self.filled) - 1;
    }

    pub(crate) fn flag(&mut self, value: bool) {
        self.put(1, u32::from(value));
    }

    /// The low `count` bits of a two's complement value.
    pub(crate) fn signed(&mut self, count: u32, value: i32) {
        self.put(count, value as u32);
    }

    pub(crate) fn bits(&self) -> usize {
        self.bytes.len() * 8 + self.filled as usize
    }

    /// Zero bits up to the next byte boundary, then the bytes.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        if self.filled > 0 {
            let pad = 8 - self.filled;
            self.put(pad, 0);
        }
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::{BitReader, BitWriter};

    #[test]
    fn written_fields_read_back_across_byte_boundaries() {
        let mut writer = BitWriter::default();
        writer.put(16, 0x0B77);
        writer.put(5, 16);
        writer.signed(7, -3);
        writer.put(32, 0xDEAD_BEEF);
        writer.flag(true);
        writer.put(11, 2047);
        assert_eq!(writer.bits(), 72);
        let bytes = writer.finish();
        assert_eq!(bytes.len(), 9);
        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read(16, "sync").ok(), Some(0x0B77));
        assert_eq!(reader.read(5, "bsid").ok(), Some(16));
        assert_eq!(reader.signed(7, "s").ok(), Some(-3));
        assert_eq!(reader.read(32, "w").ok(), Some(0xDEAD_BEEF));
        assert_eq!(reader.bit("f").ok(), Some(true));
        assert_eq!(reader.read(11, "frmsiz").ok(), Some(2047));
        assert_eq!(reader.left(), 0);
    }

    #[test]
    fn a_read_past_the_end_is_truncated_and_moves_nothing() {
        let bytes = [0xFF_u8, 0x00];
        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read(12, "a").ok(), Some(0xFF0));
        let error = reader.read(5, "past").expect_err("only four bits remain");
        assert!(matches!(
            error,
            cutlass_decode::CodecError::Truncated {
                at_bit: 12,
                what: "past"
            }
        ));
        assert_eq!(reader.pos(), 12);
        assert!(reader.skip(5, "skip").is_err());
        assert_eq!(reader.read(4, "rest").ok(), Some(0));
    }

    #[test]
    fn a_partial_byte_is_padded_with_zero_bits() {
        let mut writer = BitWriter::with_capacity(4);
        writer.put(3, 0b101);
        assert_eq!(writer.finish(), vec![0b1010_0000]);
    }
}

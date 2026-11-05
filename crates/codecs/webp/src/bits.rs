//! The VP8L bit reader: bits come least significant first.
//!
//! Reading past the last byte is [`CodecError::Truncated`]; a peek past the
//! end sees zeros, so a prefix-code table lookup near the end works and the
//! consume that follows decides whether the bits were really there.

use cutlass_decode::CodecError;

pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// Next byte to load into `held`.
    at: usize,
    held: u64,
    count: u32,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            at: 0,
            held: 0,
            count: 0,
        }
    }

    fn fill(&mut self) {
        while self.count <= 56 {
            let Some(&byte) = self.data.get(self.at) else {
                return;
            };
            self.held |= u64::from(byte) << self.count;
            self.at += 1;
            self.count += 8;
        }
    }

    /// The next `bits` bits (at most 32) without consuming them; bits past
    /// the end read as zero.
    pub(crate) fn peek(&mut self, bits: u32) -> u32 {
        self.fill();
        let mask = if bits >= 32 {
            u64::from(u32::MAX)
        } else {
            (1_u64 << bits) - 1
        };
        u32::try_from(self.held & mask).unwrap_or(u32::MAX)
    }

    pub(crate) fn consume(&mut self, bits: u32) -> Result<(), CodecError> {
        self.fill();
        if bits > self.count {
            return Err(CodecError::Truncated {
                at_bit: self.position(),
                what: "vp8l bitstream",
            });
        }
        self.held >>= bits;
        self.count -= bits;
        Ok(())
    }

    /// Read `bits` bits (at most 32) as an unsigned value.
    pub(crate) fn read(&mut self, bits: u32) -> Result<u32, CodecError> {
        let value = self.peek(bits);
        self.consume(bits)?;
        Ok(value)
    }

    pub(crate) fn read_flag(&mut self) -> Result<bool, CodecError> {
        Ok(self.read(1)? == 1)
    }

    /// How many bits have been consumed, for error positions.
    pub(crate) fn position(&self) -> u64 {
        let loaded = u64::try_from(self.at).unwrap_or(u64::MAX).saturating_mul(8);
        loaded.saturating_sub(u64::from(self.count))
    }
}

#[cfg(test)]
mod tests {
    use super::BitReader;

    #[test]
    fn bits_come_least_significant_first_and_stop_at_the_end() {
        let mut reader = BitReader::new(&[0b1010_1101, 0xFF, 0x01]);
        assert_eq!(reader.read(1).unwrap(), 1);
        assert_eq!(reader.read(3).unwrap(), 0b110);
        assert_eq!(reader.read(4).unwrap(), 0b1010);
        assert_eq!(reader.read(9).unwrap(), 0x1FF);
        assert_eq!(reader.position(), 17);
        assert_eq!(reader.peek(16), 0);
        assert_eq!(reader.read(7).unwrap(), 0);
        assert!(reader.read(1).is_err());
        let mut wide = BitReader::new(&[0x78, 0x56, 0x34, 0x12, 0xAB]);
        assert_eq!(wide.read(32).unwrap(), 0x1234_5678);
    }
}

//! Most-significant-bit-first reading and writing.
//!
//! FLAC stores every field big-endian and bit-packed (RFC 9639, section 5).
//! The reader refuses a read past the end of its buffer with
//! [`CodecError::Truncated`]; it never pads missing bits with zeros.

use cutlass_decode::CodecError;

/// Reads bits from one complete unit.
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    bit: u64,
    end: u64,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        let end = u64::try_from(data.len())
            .unwrap_or(u64::MAX / 8)
            .saturating_mul(8);
        Self { data, bit: 0, end }
    }

    /// A reader whose next read is at byte `start`.
    pub(crate) fn at_byte(data: &'a [u8], start: usize) -> Self {
        let mut reader = Self::new(data);
        reader.bit = u64::try_from(start)
            .unwrap_or(u64::MAX / 8)
            .saturating_mul(8)
            .min(reader.end);
        reader
    }

    /// Bit offset of the next read.
    pub(crate) fn position(&self) -> u64 {
        self.bit
    }

    pub(crate) fn remaining(&self) -> u64 {
        self.end - self.bit
    }

    /// The next 64 bits, left aligned, with zeros past the end of the buffer.
    /// At least 57 of them are real bits when that many remain.
    fn window(&self) -> u64 {
        let byte = usize::try_from(self.bit >> 3).unwrap_or(usize::MAX);
        let mut buf = [0u8; 8];
        if let Some(rest) = self.data.get(byte..) {
            let take = rest.len().min(8);
            buf[..take].copy_from_slice(&rest[..take]);
        }
        u64::from_be_bytes(buf) << (self.bit & 7)
    }

    /// Reads `count` bits, at most 56, as an unsigned value.
    pub(crate) fn read(&mut self, count: u32, what: &'static str) -> Result<u64, CodecError> {
        if count == 0 {
            return Ok(0);
        }
        if u64::from(count) > self.remaining() {
            return Err(CodecError::Truncated {
                at_bit: self.bit,
                what,
            });
        }
        let value = self.window() >> (64 - count);
        self.bit += u64::from(count);
        Ok(value)
    }

    /// Reads a two's complement value `count` bits wide, at most 33.
    /// Zero bits read nothing and return zero.
    pub(crate) fn read_signed(
        &mut self,
        count: u32,
        what: &'static str,
    ) -> Result<i64, CodecError> {
        if count == 0 {
            return Ok(0);
        }
        let raw = self.read(count, what)?;
        let shift = 64 - count;
        Ok(((raw << shift) as i64) >> shift)
    }

    /// Counts zero bits up to and including the terminating one bit.
    ///
    /// Returns the number of zeros. More than `limit` zeros is
    /// [`CodecError::Malformed`], so a hostile run cannot spin.
    pub(crate) fn unary(&mut self, limit: u64, what: &'static str) -> Result<u64, CodecError> {
        let mut zeros = 0u64;
        loop {
            let left = self.remaining();
            if left == 0 {
                return Err(CodecError::Truncated {
                    at_bit: self.bit,
                    what,
                });
            }
            let valid = left.min(57);
            let lead = u64::from(self.window().leading_zeros());
            if lead < valid {
                zeros += lead;
                self.bit += lead + 1;
                if zeros > limit {
                    return Err(CodecError::Malformed {
                        at_bit: self.bit,
                        what,
                    });
                }
                return Ok(zeros);
            }
            zeros += valid;
            self.bit += valid;
            if zeros > limit {
                return Err(CodecError::Malformed {
                    at_bit: self.bit,
                    what,
                });
            }
        }
    }

    /// Skips to the next byte boundary and returns the skipped bits.
    pub(crate) fn align(&mut self) -> Result<u64, CodecError> {
        let partial = u32::try_from((8 - (self.bit & 7)) & 7).unwrap_or(0);
        self.read(partial, "byte alignment")
    }
}

/// Writes bits into a growing byte buffer.
pub(crate) struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    pending: u32,
}

impl BitWriter {
    pub(crate) fn with_capacity(bytes: usize) -> Self {
        Self {
            out: Vec::with_capacity(bytes),
            acc: 0,
            pending: 0,
        }
    }

    /// Writes the low `count` bits of `value`, at most 32.
    pub(crate) fn put(&mut self, count: u32, value: u64) {
        if count == 0 {
            return;
        }
        let mask = (1u64 << count) - 1;
        self.acc = (self.acc << count) | (value & mask);
        self.pending += count;
        while self.pending >= 8 {
            self.pending -= 8;
            self.out.push((self.acc >> self.pending) as u8);
        }
        self.acc &= (1u64 << self.pending) - 1;
    }

    /// Writes up to 64 bits.
    pub(crate) fn put_long(&mut self, count: u32, value: u64) {
        if count > 32 {
            self.put(count - 32, value >> 32);
            self.put(32, value);
        } else {
            self.put(count, value);
        }
    }

    /// Writes a two's complement value `count` bits wide, at most 33.
    pub(crate) fn put_signed(&mut self, count: u32, value: i64) {
        self.put_long(count, value as u64);
    }

    /// Writes `zeros` zero bits and a one.
    pub(crate) fn put_unary(&mut self, zeros: u64) {
        let mut left = zeros;
        while left >= 32 {
            self.put(32, 0);
            left -= 32;
        }
        self.put(u32::try_from(left).unwrap_or(0) + 1, 1);
    }

    /// Pads with zero bits to a byte boundary.
    pub(crate) fn align(&mut self) {
        if self.pending > 0 {
            let pad = 8 - self.pending;
            self.put(pad, 0);
        }
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.out
    }

    /// Bits written so far.
    pub(crate) fn bit_len(&self) -> u64 {
        u64::try_from(self.out.len())
            .unwrap_or(u64::MAX / 8)
            .saturating_mul(8)
            + u64::from(self.pending)
    }

    /// Appends every bit `other` holds.
    pub(crate) fn append(&mut self, other: &Self) {
        for byte in &other.out {
            self.put(8, u64::from(*byte));
        }
        self.put(other.pending, other.acc);
    }

    /// Drops every bit, keeping the allocation.
    pub(crate) fn clear(&mut self) {
        self.out.clear();
        self.acc = 0;
        self.pending = 0;
    }

    /// The bytes. Call [`Self::align`] first.
    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::{BitReader, BitWriter};

    #[test]
    fn written_fields_read_back_in_order() {
        let mut writer = BitWriter::with_capacity(16);
        writer.put(3, 0b101);
        writer.put_signed(33, -5);
        writer.put_unary(40);
        writer.put(7, 0x55);
        writer.put_long(36, 0x9_8765_4321);
        writer.align();
        let bytes = writer.into_bytes();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read(3, "a").expect("a"), 0b101);
        assert_eq!(reader.read_signed(33, "b").expect("b"), -5);
        assert_eq!(reader.unary(64, "c").expect("c"), 40);
        assert_eq!(reader.read(7, "d").expect("d"), 0x55);
        assert_eq!(reader.read(36, "e").expect("e"), 0x9_8765_4321);
        reader.align().expect("align");
        assert_eq!(reader.remaining(), 0);
    }

    #[test]
    fn a_read_past_the_end_is_truncated_and_not_zero() {
        let mut reader = BitReader::new(&[0xFF]);
        assert!(reader.read(9, "x").is_err());
        assert_eq!(reader.position(), 0);
        let mut zeros = BitReader::new(&[0, 0]);
        assert!(zeros.unary(100, "u").is_err());
    }

    #[test]
    fn a_unary_run_past_its_limit_is_malformed() {
        let mut reader = BitReader::new(&[0, 0, 0, 1]);
        assert!(matches!(
            reader.unary(10, "u"),
            Err(cutlass_decode::CodecError::Malformed { .. })
        ));
    }
}

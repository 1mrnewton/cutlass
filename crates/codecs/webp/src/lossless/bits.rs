//! Least-significant-bit-first reader and writer for VP8L.
//!
//! RFC 9649 section 3.3: bits are read from the low end of each byte, in
//! byte order. A value of `n` bits has its least significant bit first.

use cutlass_decode::CodecError;

pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// Next byte to load into `window`.
    next: usize,
    /// Loaded bits, the next one in bit 0.
    window: u64,
    /// Valid bits in `window`.
    held: u32,
    /// Bits consumed so far.
    used: u64,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            next: 0,
            window: 0,
            held: 0,
            used: 0,
        }
    }

    /// Bit offset of the next unread bit.
    pub(crate) fn position(&self) -> u64 {
        self.used
    }

    fn fill(&mut self) {
        if self.held <= 32
            && let Some(chunk) = self.data.get(self.next..self.next + 4)
        {
            let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            self.window |= u64::from(word) << self.held;
            self.next += 4;
            self.held += 32;
        }
        while self.held <= 56 {
            let Some(&byte) = self.data.get(self.next) else {
                return;
            };
            self.window |= u64::from(byte) << self.held;
            self.next += 1;
            self.held += 8;
        }
    }

    /// The next `count` bits without consuming them, `count <= 32`. Bits
    /// past the end read as zero; [`Self::skip`] refuses to consume them.
    #[inline]
    pub(crate) fn peek(&mut self, count: u32) -> u32 {
        if self.held < count {
            self.fill();
        }
        let mask = if count >= 32 {
            u64::from(u32::MAX)
        } else {
            (1u64 << count) - 1
        };
        // The mask keeps at most 32 bits.
        u32::try_from(self.window & mask).unwrap_or(u32::MAX)
    }

    /// Consume `count` bits that [`Self::peek`] returned.
    #[inline]
    pub(crate) fn skip(&mut self, count: u32) -> Result<(), CodecError> {
        if self.held < count {
            self.fill();
            if self.held < count {
                return Err(CodecError::Truncated {
                    at_bit: self.used,
                    what: "vp8l bitstream",
                });
            }
        }
        self.window >>= count;
        self.held -= count;
        self.used += u64::from(count);
        Ok(())
    }

    /// Read `count` bits, `count <= 32`.
    #[inline]
    pub(crate) fn read(&mut self, count: u32) -> Result<u32, CodecError> {
        let value = self.peek(count);
        self.skip(count)?;
        Ok(value)
    }

    pub(crate) fn flag(&mut self) -> Result<bool, CodecError> {
        Ok(self.read(1)? == 1)
    }
}

/// Writer with the same bit order as [`BitReader`].
#[cfg(test)]
#[derive(Default)]
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    window: u64,
    held: u32,
}

#[cfg(test)]
impl BitWriter {
    /// Write the low `count` bits of `value`, `count <= 32`.
    #[inline]
    pub(crate) fn write(&mut self, value: u32, count: u32) {
        if count == 0 {
            return;
        }
        let masked = if count >= 32 {
            u64::from(value)
        } else {
            u64::from(value) & ((1u64 << count) - 1)
        };
        self.window |= masked << self.held;
        self.held += count;
        while self.held >= 8 {
            self.bytes.push((self.window & 0xff) as u8);
            self.window >>= 8;
            self.held -= 8;
        }
    }

    /// Bits written so far.
    pub(crate) fn bit_len(&self) -> u64 {
        (self.bytes.len() as u64) * 8 + u64::from(self.held)
    }

    /// The bytes, the last one padded with zero bits.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        if self.held > 0 {
            self.bytes.push((self.window & 0xff) as u8);
        }
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::{BitReader, BitWriter};

    #[test]
    fn bits_come_back_in_the_order_they_were_written() {
        let mut writer = BitWriter::default();
        let values = [
            (1u32, 1u32),
            (5, 3),
            (0x2f, 8),
            (0x3fff, 14),
            (0, 2),
            (0x1234_5678, 32),
        ];
        for (value, count) in values {
            writer.write(value, count);
        }
        assert_eq!(writer.bit_len(), 60);
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        for (value, count) in values {
            assert_eq!(reader.read(count).unwrap(), value);
        }
        assert_eq!(reader.position(), 60);
    }

    #[test]
    fn the_first_bit_is_the_low_bit_of_the_first_byte() {
        let mut reader = BitReader::new(&[0b1010_0110, 0xff]);
        assert_eq!(reader.read(1).unwrap(), 0);
        assert_eq!(reader.read(2).unwrap(), 0b11);
        assert_eq!(reader.read(5).unwrap(), 0b10100);
        assert_eq!(reader.read(8).unwrap(), 0xff);
    }

    #[test]
    fn reading_past_the_end_is_truncated() {
        let mut reader = BitReader::new(&[0xff]);
        assert_eq!(reader.read(8).unwrap(), 0xff);
        assert!(reader.read(1).is_err());
        let mut short = BitReader::new(&[1, 2]);
        assert_eq!(short.peek(24), 0x0201);
        assert!(short.skip(17).is_err());
    }
}

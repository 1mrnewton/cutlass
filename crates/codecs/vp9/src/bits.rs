#![allow(dead_code)]
//! Uncompressed bit IO and the VP9 bool coder.
//!
//! The bool decoder follows the specification's one-bit process. The bool
//! encoder follows libvpx `vpx_write`, including the shift that is reassigned
//! from the bit count before the final left shift.

use cutlass_decode::CodecError;

use crate::tables::VPX_NORM;

/// Most-significant-bit-first reader over one buffer.
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn bits_left(&self) -> usize {
        self.data.len().saturating_mul(8).saturating_sub(self.pos)
    }

    pub fn read(&mut self, n: u32) -> Result<u32, CodecError> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(CodecError::Limit {
                what: "bit read wider than 32",
            });
        }
        if self.bits_left() < usize::try_from(n).unwrap_or(usize::MAX) {
            return Err(CodecError::Truncated {
                at_bit: self.pos as u64,
                what: "bitstream",
            });
        }
        let mut left = n;
        let mut value = 0u32;
        while left > 0 {
            let byte = u32::from(self.data[self.pos / 8]);
            let used = (self.pos % 8) as u32;
            let available = 8 - used;
            let take = left.min(available);
            let shift = available - take;
            let mask = (1u32 << take) - 1;
            value = (value << take) | ((byte >> shift) & mask);
            self.pos += usize::try_from(take).unwrap_or(0);
            left -= take;
        }
        Ok(value)
    }

    pub fn byte_aligned(&self) -> bool {
        self.pos.is_multiple_of(8)
    }

    /// Bytes consumed, rounding a partial byte up.
    pub fn bytes_read(&self) -> usize {
        self.pos.div_ceil(8)
    }
}

/// Specification bool decoder over one partition.
pub struct BoolDecoder<'a> {
    data: &'a [u8],
    bit: usize,
    value: u32,
    range: u32,
    max_bits: i32,
}

impl<'a> BoolDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self, CodecError> {
        if data.is_empty() {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "bool coder",
            });
        }
        let mut dec = Self {
            data,
            bit: 8,
            value: u32::from(data[0]),
            range: 255,
            max_bits: (data.len() as i32).saturating_mul(8) - 8,
        };
        if dec.read(128)? != 0 {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "bool marker",
            });
        }
        Ok(dec)
    }

    pub fn read(&mut self, prob: u32) -> Result<u32, CodecError> {
        let split = 1 + (((self.range - 1) * prob) >> 8);
        let bit = if self.value < split {
            self.range = split;
            0
        } else {
            self.value -= split;
            self.range -= split;
            1
        };
        while self.range < 128 {
            let new_bit = if self.max_bits > 0 {
                self.max_bits -= 1;
                self.raw_bit()?
            } else {
                return Err(CodecError::Malformed {
                    at_bit: self.bit as u64,
                    what: "bool renorm",
                });
            };
            self.range *= 2;
            self.value = (self.value << 1) + new_bit;
        }
        Ok(bit)
    }

    pub fn read_literal(&mut self, n: u32) -> Result<u32, CodecError> {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | self.read(128)?;
        }
        Ok(value)
    }

    /// Trailing raw bits must be zero. This is `exit_bool`.
    pub fn finish(&mut self) -> Result<(), CodecError> {
        let bits = self.max_bits;
        if bits < 0 {
            return Err(CodecError::Malformed {
                at_bit: self.bit as u64,
                what: "bool padding",
            });
        }
        for _ in 0..bits {
            if self.raw_bit()? != 0 {
                return Err(CodecError::Malformed {
                    at_bit: self.bit as u64,
                    what: "bool padding",
                });
            }
        }
        self.max_bits = 0;
        Ok(())
    }

    fn raw_bit(&mut self) -> Result<u32, CodecError> {
        let byte_index = self.bit / 8;
        if byte_index >= self.data.len() {
            return Err(CodecError::Truncated {
                at_bit: self.bit as u64,
                what: "bool coder",
            });
        }
        let shift = 7 - (self.bit % 8);
        let bit = u32::from(self.data[byte_index] >> shift) & 1;
        self.bit += 1;
        Ok(bit)
    }
}

/// libvpx bool encoder.
pub struct BoolWriter {
    low: u32,
    range: u32,
    count: i32,
    buf: Vec<u8>,
    failed: bool,
}

impl BoolWriter {
    pub fn new() -> Self {
        let mut writer = Self {
            low: 0,
            range: 255,
            count: -24,
            buf: Vec::new(),
            failed: false,
        };
        writer.write(0, 128);
        writer
    }

    pub fn write(&mut self, bit: u32, prob: u32) {
        if self.failed {
            return;
        }
        let split = 1 + (((self.range - 1) * prob) >> 8);
        let mut range = split;
        let mut low = self.low;
        if bit != 0 {
            low = low.wrapping_add(split);
            range = self.range - split;
        }
        let norm = u32::from(VPX_NORM[usize::try_from(range).unwrap_or(0)]);
        let mut shift = norm;
        range <<= shift;
        self.count += i32::try_from(shift).unwrap_or(0);
        if self.count >= 0 {
            let offset = shift as i32 - self.count;
            if offset <= 0 {
                self.failed = true;
                return;
            }
            let offset_u = u32::try_from(offset).unwrap_or(0);
            if low.wrapping_shl(offset_u - 1) & 0x8000_0000 != 0 {
                let mut index = self.buf.len() as i32 - 1;
                while index >= 0 && self.buf[usize::try_from(index).unwrap_or(0)] == 0xff {
                    self.buf[usize::try_from(index).unwrap_or(0)] = 0;
                    index -= 1;
                }
                if index < 0 {
                    self.failed = true;
                    return;
                }
                let at = usize::try_from(index).unwrap_or(0);
                self.buf[at] = self.buf[at].wrapping_add(1);
            }
            let byte = ((low >> (24 - offset_u)) & 0xff) as u8;
            self.buf.push(byte);
            low = low.wrapping_shl(offset_u);
            shift = u32::try_from(self.count).unwrap_or(0);
            low &= 0x00ff_ffff;
            self.count -= 8;
        }
        low = low.wrapping_shl(shift);
        self.low = low;
        self.range = range;
    }

    pub fn write_literal(&mut self, value: u32, bits: u32) {
        for shift in (0..bits).rev() {
            self.write((value >> shift) & 1, 128);
        }
    }

    pub fn finish(mut self) -> Result<Vec<u8>, CodecError> {
        for _ in 0..32 {
            self.write(0, 128);
        }
        if self.failed {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "bool encoder carry",
            });
        }
        if let Some(last) = self.buf.last().copied()
            && last & 0xe0 == 0xc0
        {
            self.buf.push(0);
        }
        Ok(self.buf)
    }
}

impl Default for BoolWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{BitReader, BoolDecoder, BoolWriter};

    #[test]
    fn bool_round_trip_mixed_probabilities() {
        let mut writer = BoolWriter::new();
        let bits = [0u32, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1];
        let probs = [128u32, 16, 240, 200, 10, 250, 128, 64, 192, 32, 96, 160, 8];
        for (bit, prob) in bits.iter().zip(probs) {
            writer.write(*bit, prob);
        }
        writer.write_literal(0x15a, 12);
        let bytes = writer.finish().expect("encode");
        assert_eq!(bytes, [0x3d, 0x84, 0x09, 0x20, 0x00]);
        let mut dec = BoolDecoder::new(&bytes).expect("decode");
        for (bit, prob) in bits.iter().zip(probs) {
            assert_eq!(dec.read(prob).expect("bit"), *bit);
        }
        assert_eq!(dec.read_literal(12).expect("literal"), 0x15a);
        dec.finish().expect("padding");
    }

    #[test]
    fn literal_bits_are_big_endian() {
        let mut reader = BitReader::new(&[0b1011_0001, 0b1100_0000]);
        assert_eq!(reader.read(4).expect("nibble"), 0b1011);
        assert_eq!(reader.read(6).expect("six"), 0b00_0111);
        assert_eq!(reader.bytes_read(), 2);
    }

    #[test]
    fn marker_bit_must_be_clear() {
        let err = match BoolDecoder::new(&[0xff, 0x00]) {
            Err(err) => err,
            Ok(_) => panic!("marker bit was accepted"),
        };
        let message = err.to_string();
        assert!(message.contains("marker"), "{message}");
    }
}

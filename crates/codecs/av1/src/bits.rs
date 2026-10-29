use cutlass_decode::CodecError;

/// Most-significant-bit-first reader. The first bit is the high bit of the first byte.
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    #[must_use]
    pub fn position(&self) -> usize {
        self.pos
    }

    pub(crate) fn suffix(&self, byte: usize) -> &'a [u8] {
        self.data.get(byte..).unwrap_or(&[])
    }

    pub fn bit_at(&self, pos: usize) -> Result<bool, CodecError> {
        let index = pos / 8;
        if index >= self.data.len() {
            return Err(CodecError::Truncated {
                at_bit: pos as u64,
                what: "bit",
            });
        }
        let shift = 7 - (pos % 8);
        Ok(((self.data[index] >> shift) & 1) == 1)
    }

    #[must_use]
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
        if self.bits_left() < n as usize {
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
            self.pos += take as usize;
            left -= take;
        }
        Ok(value)
    }

    pub fn skip(&mut self, n: usize) -> Result<(), CodecError> {
        if self.bits_left() < n {
            return Err(CodecError::Truncated {
                at_bit: self.pos as u64,
                what: "bitstream",
            });
        }
        self.pos += n;
        Ok(())
    }

    pub fn read_bool(&mut self) -> Result<bool, CodecError> {
        Ok(self.read(1)? == 1)
    }

    /// Signed integer of `n` bits, high bit is the sign.
    pub fn read_signed(&mut self, n: u32) -> Result<i32, CodecError> {
        let raw = self.read(n)?;
        let sign = 1u32 << (n - 1);
        if raw & sign == 0 {
            Ok(raw as i32)
        } else {
            Ok(raw as i32 - (sign as i32 * 2))
        }
    }

    pub fn byte_align(&mut self) {
        let rem = self.pos % 8;
        if rem != 0 {
            self.pos += 8 - rem;
        }
    }

    /// Little-endian integer of `bytes` bytes, as used for a tile size.
    pub fn read_le(&mut self, bytes: u32) -> Result<u32, CodecError> {
        let mut value = 0u32;
        for i in 0..bytes {
            let byte = self.read(8)?;
            value |= byte << (8 * i);
        }
        Ok(value)
    }
}

/// Most-significant-bit-first writer. `finish` zero-pads the last byte.
pub struct BitWriter {
    data: Vec<u8>,
    acc: u8,
    bits: u8,
}

impl BitWriter {
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            acc: 0,
            bits: 0,
        }
    }

    pub fn write(&mut self, mut value: u32, n: u32) -> Result<(), CodecError> {
        if n == 0 {
            return Ok(());
        }
        if n > 32 {
            return Err(CodecError::Limit {
                what: "bit write wider than 32",
            });
        }
        if n < 32 {
            value &= (1u32 << n) - 1;
        }
        for shift in (0..n).rev() {
            let bit = ((value >> shift) & 1) as u8;
            self.acc = (self.acc << 1) | bit;
            self.bits += 1;
            if self.bits == 8 {
                self.data.push(self.acc);
                self.acc = 0;
                self.bits = 0;
            }
        }
        Ok(())
    }

    pub fn write_bool(&mut self, value: bool) -> Result<(), CodecError> {
        self.write(u32::from(value), 1)
    }

    pub fn write_signed(&mut self, value: i32, n: u32) -> Result<(), CodecError> {
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        self.write(value as u32 & mask, n)
    }

    pub fn align_zero(&mut self) {
        if self.bits > 0 {
            self.acc <<= 8 - self.bits;
            self.data.push(self.acc);
            self.acc = 0;
            self.bits = 0;
        }
    }

    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        self.align_zero();
        self.data
    }

    pub fn trailing_bits(&mut self) -> Result<(), CodecError> {
        self.write_bool(true)?;
        while self.bits != 0 {
            self.write_bool(false)?;
        }
        Ok(())
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

pub fn floor_log2(value: u32) -> u32 {
    31 - value.leading_zeros()
}

/// Unsigned LEB128, at most 8 bytes, values that fit in `u32`.
pub fn read_leb128(bits: &mut BitReader<'_>) -> Result<u32, CodecError> {
    let mut value = 0u32;
    for i in 0..8 {
        let byte = bits.read(8)?;
        let piece = u64::from(byte & 0x7f) << (i * 7);
        let next = u64::from(value) | piece;
        if next > u64::from(u32::MAX) {
            return Err(CodecError::Limit { what: "leb128" });
        }
        value = u32::try_from(next).unwrap_or(0);
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(CodecError::Malformed {
        at_bit: bits.position() as u64,
        what: "leb128",
    })
}

pub(crate) fn signed_and_little_endian_hold() -> bool {
    let mut writer = BitWriter::new();
    if writer.write_signed(-3, 4).is_err() {
        return false;
    }
    let bytes = writer.finish();
    let mut reader = BitReader::new(&bytes);
    let Ok(signed) = reader.read_signed(4) else {
        return false;
    };
    let Ok(le) = BitReader::new(&[0x02, 0x01]).read_le(2) else {
        return false;
    };
    signed == -3 && le == 0x0102
}

pub fn write_leb128(out: &mut Vec<u8>, mut value: u32) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BitReader, BitWriter, read_leb128, write_leb128};
    use cutlass_decode::CodecError;

    #[test]
    fn bits_cross_a_byte() {
        let mut writer = BitWriter::new();
        writer.write(0b101, 3).unwrap();
        writer.write(0b1111_0001, 8).unwrap();
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read(3).unwrap(), 0b101);
        assert_eq!(reader.read(8).unwrap(), 0b1111_0001);
    }

    #[test]
    fn a_leb128_past_u32_is_a_limit() {
        let mut reader = BitReader::new(&[0xff, 0xff, 0xff, 0xff, 0x7f]);
        assert!(matches!(
            read_leb128(&mut reader),
            Err(CodecError::Limit { .. })
        ));
    }

    #[test]
    fn leb128_round_trips_a_tile_size() {
        let mut bytes = Vec::new();
        write_leb128(&mut bytes, 300);
        let mut reader = BitReader::new(&bytes);
        assert_eq!(read_leb128(&mut reader).unwrap(), 300);
    }

    #[test]
    fn signed_values_round_trip() {
        let mut writer = BitWriter::new();
        writer.write_signed(-3, 4).unwrap();
        writer.write_signed(5, 4).unwrap();
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read_signed(4).unwrap(), -3);
        assert_eq!(reader.read_signed(4).unwrap(), 5);
    }

    #[test]
    fn little_endian_bytes_are_read_in_order() {
        let mut reader = BitReader::new(&[0x78, 0x56, 0x34, 0x12]);
        assert_eq!(reader.read_le(4).unwrap(), 0x1234_5678);
    }
}

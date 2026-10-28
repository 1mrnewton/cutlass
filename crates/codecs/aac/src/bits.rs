use cutlass_decode::CodecError;

/// Most-significant-bit-first reader over a borrowed buffer.
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

    pub fn byte_align(&mut self) -> Result<(), CodecError> {
        let rem = self.pos % 8;
        if rem != 0 {
            let skip = 8 - rem;
            if self.bits_left() < skip {
                return Err(CodecError::Truncated {
                    at_bit: self.pos as u64,
                    what: "byte align",
                });
            }
            self.pos += skip;
        }
        Ok(())
    }
}

pub trait BitSink {
    fn write_bits(&mut self, value: u32, n: u32) -> Result<(), CodecError>;
}

/// Most-significant-bit-first writer. `finish` zero-pads to a byte.
pub struct BitWriter {
    data: Vec<u8>,
    acc: u8,
    bits: u8,
    cap: usize,
}

impl BitWriter {
    pub fn with_cap(max_bytes: usize) -> Self {
        Self {
            data: Vec::new(),
            acc: 0,
            bits: 0,
            cap: max_bytes,
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        if self.bits > 0 {
            self.acc <<= 8 - self.bits;
            self.data.push(self.acc);
        }
        self.data
    }
}

impl BitSink for BitWriter {
    fn write_bits(&mut self, mut value: u32, n: u32) -> Result<(), CodecError> {
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
                if self.data.len() >= self.cap {
                    return Err(CodecError::Limit {
                        what: "access unit bytes",
                    });
                }
                self.data.push(self.acc);
                self.acc = 0;
                self.bits = 0;
            }
        }
        Ok(())
    }
}

pub struct BitCounter {
    pub bits: u32,
}

impl BitSink for BitCounter {
    fn write_bits(&mut self, _value: u32, n: u32) -> Result<(), CodecError> {
        self.bits = self.bits.saturating_add(n);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{BitReader, BitSink, BitWriter};

    #[test]
    fn round_trip_spans_byte_boundaries() {
        let mut w = BitWriter::with_cap(16);
        w.write_bits(0b101, 3).unwrap();
        w.write_bits(0b1111_0000, 8).unwrap();
        w.write_bits(0b11, 2).unwrap();
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read(3).unwrap(), 0b101);
        assert_eq!(r.read(8).unwrap(), 0b1111_0000);
        assert_eq!(r.read(2).unwrap(), 0b11);
    }

    #[test]
    fn truncated_read_names_the_bit() {
        let mut r = BitReader::new(&[0xFF]);
        assert!(r.read(8).is_ok());
        let err = r.read(1).unwrap_err();
        assert!(matches!(
            err,
            cutlass_decode::CodecError::Truncated { at_bit: 8, .. }
        ));
    }

    #[test]
    fn empty_read_is_zero() {
        let mut r = BitReader::new(&[]);
        assert_eq!(r.read(0).unwrap(), 0);
        assert!(r.read(1).is_err());
    }
}

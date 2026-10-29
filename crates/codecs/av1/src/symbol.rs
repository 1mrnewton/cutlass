use cutlass_decode::CodecError;

use crate::bits::{BitReader, floor_log2};

/// Symbol decoder from section 8.2 of the AV1 specification.
pub struct SymbolDecoder<'a> {
    bits: BitReader<'a>,
    value: u32,
    range: u32,
    max_bits: i32,
}

impl<'a> SymbolDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self, CodecError> {
        if data.is_empty() {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "symbol tile",
            });
        }
        let mut bits = BitReader::new(data);
        let num_bits = (data.len().saturating_mul(8)).min(15) as u32;
        let buf = bits.read(num_bits)?;
        let padded = buf << (15 - num_bits);
        Ok(Self {
            bits,
            value: ((1 << 15) - 1) ^ padded,
            range: 1 << 15,
            max_bits: (data.len() as i32).saturating_mul(8) - 15,
        })
    }

    pub fn read(&mut self, cdf: &mut [u16], update: bool) -> Result<u32, CodecError> {
        if cdf.len() < 3 {
            return Err(CodecError::Malformed {
                at_bit: self.bits.position() as u64,
                what: "cdf",
            });
        }
        let symbols = cdf.len() - 1;
        let mut cur = self.range;
        let mut symbol: i32 = -1;
        let prev = loop {
            symbol += 1;
            if symbol as usize >= symbols {
                return Err(CodecError::Malformed {
                    at_bit: self.bits.position() as u64,
                    what: "symbol",
                });
            }
            let prev = cur;
            let f = (1u32 << 15) - u32::from(cdf[symbol as usize]);
            cur = ((self.range >> 8) * (f >> 6)) >> 1;
            cur += 4 * (symbols as u32 - symbol as u32 - 1);
            if self.value >= cur {
                break prev;
            }
        };
        self.range = prev - cur;
        self.value -= cur;
        if self.range == 0 {
            return Err(CodecError::Malformed {
                at_bit: self.bits.position() as u64,
                what: "symbol range",
            });
        }
        self.renorm()?;
        let decoded = symbol as u32;
        if update {
            adapt(cdf, decoded as usize);
        }
        Ok(decoded)
    }

    pub fn read_bool(&mut self) -> Result<bool, CodecError> {
        let mut cdf = [1 << 14, 1 << 15, 0];
        Ok(self.read(&mut cdf, false)? == 1)
    }

    pub fn read_literal(&mut self, n: u32) -> Result<u32, CodecError> {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | u32::from(self.read_bool()?);
        }
        Ok(value)
    }

    /// Checks the tile trailing bit required by `exit_symbol`.
    pub fn finish(&mut self) -> Result<(), CodecError> {
        if self.max_bits < -14 {
            return Err(CodecError::Malformed {
                at_bit: self.bits.position() as u64,
                what: "symbol end",
            });
        }
        let position = self.bits.position() as i32;
        let trailing = position - (15 + self.max_bits).min(15);
        let skip = self.max_bits.max(0) as u32;
        if self.bits.bits_left() < skip as usize {
            return Err(CodecError::Truncated {
                at_bit: self.bits.position() as u64,
                what: "symbol padding",
            });
        }
        if skip > 0 {
            self.bits.skip(skip as usize)?;
        }
        let end = self.bits.position() as i32;
        if trailing < 0 || end <= trailing {
            return Err(CodecError::Malformed {
                at_bit: self.bits.position() as u64,
                what: "trailing bit",
            });
        }
        if !self.bits.bit_at(trailing as usize)? {
            return Err(CodecError::Malformed {
                at_bit: trailing as u64,
                what: "trailing bit",
            });
        }
        for pos in (trailing + 1)..end {
            if self.bits.bit_at(pos as usize)? {
                return Err(CodecError::Malformed {
                    at_bit: pos as u64,
                    what: "trailing padding",
                });
            }
        }
        Ok(())
    }

    fn renorm(&mut self) -> Result<(), CodecError> {
        let bits = 15 - floor_log2(self.range);
        self.range <<= bits;
        let available = self.max_bits.max(0) as u32;
        let num = bits.min(available);
        let new_data = if num == 0 { 0 } else { self.bits.read(num)? };
        let padded = new_data << (bits - num);
        self.value = padded ^ (((self.value + 1) << bits) - 1);
        self.max_bits -= bits as i32;
        Ok(())
    }
}

fn adapt(cdf: &mut [u16], symbol: usize) {
    let n = cdf.len() - 1;
    let count = usize::from(cdf[n]);
    let rate = 3 + u32::from(count > 15) + u32::from(count > 31) + floor_log2(n as u32).min(2);
    let mut tmp = 0u32;
    for (i, slot) in cdf.iter_mut().enumerate().take(n - 1) {
        if i == symbol {
            tmp = 1 << 15;
        }
        let current = u32::from(*slot);
        if tmp < current {
            *slot = (current - ((current - tmp) >> rate)) as u16;
        } else {
            *slot = (current + ((tmp - current) >> rate)) as u16;
        }
    }
    if cdf[n] < 32 {
        cdf[n] += 1;
    }
}

/// Symbol encoder. The splits match section 8.2, so [`SymbolDecoder`] reads the bytes back.
pub struct SymbolEncoder {
    low: u64,
    range: u32,
    cnt: i32,
    out: Vec<u8>,
}

impl SymbolEncoder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            low: 0,
            range: 0x8000,
            cnt: -9,
            out: Vec::new(),
        }
    }

    pub fn symbol(&mut self, symbol: usize, cdf: &[u16]) -> Result<(), CodecError> {
        if cdf.len() < 3 || symbol + 1 >= cdf.len() {
            return Err(CodecError::InvalidApi { what: "symbol" });
        }
        let nsyms = cdf.len() - 1;
        let narrowed = nsyms as u32 - 1;
        let fl = if symbol == 0 {
            32768
        } else {
            32768 - u32::from(cdf[symbol - 1])
        };
        let fh = 32768 - u32::from(cdf[symbol]);
        let range = self.range;
        if fl < 32768 {
            let upper = split(range, fl, narrowed - (symbol as u32 - 1));
            let lower = split(range, fh, narrowed - symbol as u32);
            if upper <= lower || upper > range {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "encode split",
                });
            }
            self.low += u64::from(range - upper);
            self.range = upper - lower;
        } else {
            let lower = split(range, fh, narrowed - symbol as u32);
            if lower >= range {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "encode split",
                });
            }
            self.range = range - lower;
        }
        self.renorm()
    }

    pub fn bool(&mut self, value: bool) -> Result<(), CodecError> {
        let cdf = [1 << 14, 1 << 15, 0];
        self.symbol(usize::from(value), &cdf)
    }

    pub fn literal(&mut self, n: u32, mut value: u32) -> Result<(), CodecError> {
        for shift in (0..n).rev() {
            self.bool(((value >> shift) & 1) == 1)?;
            value &= !(1 << shift);
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<Vec<u8>, CodecError> {
        let low = self.low;
        let mut cnt = self.cnt;
        let mut remain = 10 + cnt;
        let mask = 0x3fffu64;
        let mut end = ((low + mask) & !mask) | (mask + 1);
        if remain > 0 {
            let mut window = (1u64 << (cnt + 16)) - 1;
            while remain > 0 {
                let val = (end >> (cnt + 16)) as u16;
                self.push_byte((val & 0xff) as u8, val & 0x100 != 0)?;
                end &= window;
                remain -= 8;
                cnt -= 8;
                window >>= 8;
            }
        }
        Ok(self.out)
    }

    fn renorm(&mut self) -> Result<(), CodecError> {
        if self.range == 0 || self.range > 65535 {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "encode range",
            });
        }
        let shift = 15 - floor_log2(self.range).min(15);
        let mut cnt = self.cnt;
        let mut ready = cnt + shift as i32;
        let mut low = self.low;
        if ready >= 40 {
            let bytes = ((ready >> 3) + 1) as u32;
            cnt += 24 - (bytes as i32) * 8;
            if cnt < 0 {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "encode carry",
                });
            }
            let output = low >> cnt;
            low &= (1u64 << cnt) - 1;
            let carry_bit = 1u64 << (bytes * 8);
            let carry = output & carry_bit != 0;
            let output = output & (carry_bit - 1);
            for index in (0..bytes).rev() {
                let byte = ((output >> (8 * index)) & 0xff) as u8;
                let this_carry = carry && index + 1 == bytes;
                self.push_byte(byte, this_carry)?;
            }
            ready = cnt + shift as i32 - 24;
        }
        self.low = low << shift;
        self.range <<= shift;
        self.cnt = ready;
        Ok(())
    }

    fn push_byte(&mut self, byte: u8, carry: bool) -> Result<(), CodecError> {
        if carry {
            let mut index = self.out.len();
            loop {
                if index == 0 {
                    return Err(CodecError::Malformed {
                        at_bit: 0,
                        what: "encode carry",
                    });
                }
                index -= 1;
                let sum = u16::from(self.out[index]) + 1;
                self.out[index] = (sum & 0xff) as u8;
                if sum < 256 {
                    break;
                }
            }
        }
        self.out.push(byte);
        Ok(())
    }
}

impl Default for SymbolEncoder {
    fn default() -> Self {
        Self::new()
    }
}

fn split(range: u32, frequency: u32, symbols_after: u32) -> u32 {
    (((range >> 8) * (frequency >> 6)) >> 1) + 4 * symbols_after
}

pub(crate) fn literal_holds() -> bool {
    let mut encoder = SymbolEncoder::new();
    if encoder.literal(1, 1).is_err() {
        return false;
    }
    let Ok(bytes) = encoder.finish() else {
        return false;
    };
    let Ok(mut decoder) = SymbolDecoder::new(&bytes) else {
        return false;
    };
    decoder.read_literal(1).ok() == Some(1)
}

#[cfg(test)]
mod tests {
    use super::{SymbolDecoder, SymbolEncoder};

    #[test]
    fn symbols_round_trip_with_a_fixed_cdf() {
        let cdf = [8000u16, 16000, 24000, 32768, 0];
        let symbols = [0u32, 1, 2, 3, 1, 0, 3, 2, 2, 1];
        let mut encoder = SymbolEncoder::new();
        for symbol in symbols {
            encoder.symbol(symbol as usize, &cdf).unwrap();
        }
        let bytes = encoder.finish().unwrap();
        let mut decoder = SymbolDecoder::new(&bytes).unwrap();
        for symbol in symbols {
            assert_eq!(decoder.read(&mut cdf.clone(), false).unwrap(), symbol);
        }
        decoder.finish().unwrap();
    }

    #[test]
    fn a_literal_round_trips() {
        let mut encoder = SymbolEncoder::new();
        encoder.literal(3, 0b101).unwrap();
        let bytes = encoder.finish().unwrap();
        let mut decoder = SymbolDecoder::new(&bytes).unwrap();
        assert_eq!(decoder.read_literal(3).unwrap(), 0b101);
        decoder.finish().unwrap();
    }
}

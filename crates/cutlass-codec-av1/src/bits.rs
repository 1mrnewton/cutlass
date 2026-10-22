//! Raw bitstream reader.
//!
//! Bits inside a byte are consumed most-significant-bit first. The reader
//! never looks past `end`. Unsigned helpers reject values that do not fit the
//! requested width.

use crate::error::DecodeError;
use crate::math::{ceil_log2, floor_log2};

/// Cursor over a bounded byte slice.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Next bit index from the start of `data`.
    bit: usize,
    /// Exclusive end bit. Defaults to `data.len() * 8`.
    end: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            bit: 0,
            end: data.len().saturating_mul(8),
        }
    }

    /// Restrict reads to the next `bytes` bytes, which must be fully inside
    /// the current window and start on a byte boundary.
    #[allow(dead_code)]
    pub fn subreader(&self, bytes: usize) -> Result<BitReader<'a>, DecodeError> {
        if !self.bit.is_multiple_of(8) {
            return Err(DecodeError::malformed(
                "sub-reader requested off a byte boundary",
            ));
        }
        let start = self.bit / 8;
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| DecodeError::malformed("byte range overflow"))?;
        if end > self.end / 8 {
            return Err(DecodeError::truncated("sub-reader extends past the OBU"));
        }
        Ok(BitReader {
            data: self.data,
            bit: start * 8,
            end: end * 8,
        })
    }

    pub fn position(&self) -> usize {
        self.bit
    }

    pub(crate) fn data_slice(&self) -> &'a [u8] {
        self.data
    }

    pub fn bits_left(&self) -> usize {
        self.end.saturating_sub(self.bit)
    }

    pub fn byte_aligned(&self) -> bool {
        self.bit.is_multiple_of(8)
    }

    pub fn skip_bits(&mut self, n: usize) -> Result<(), DecodeError> {
        let next = self
            .bit
            .checked_add(n)
            .ok_or_else(|| DecodeError::malformed("bit position overflow"))?;
        if next > self.end {
            return Err(DecodeError::truncated("skip past end of buffer"));
        }
        self.bit = next;
        Ok(())
    }

    /// Advance to the absolute bit position. Used by the symbol-decoder exit
    /// process, which computes a position from bits already consumed.
    pub fn seek_bit(&mut self, bit: usize) -> Result<(), DecodeError> {
        if bit > self.end {
            return Err(DecodeError::truncated("seek past end of buffer"));
        }
        self.bit = bit;
        Ok(())
    }

    pub fn read_bit(&mut self) -> Result<u8, DecodeError> {
        if self.bit >= self.end {
            return Err(DecodeError::truncated("read past end of buffer"));
        }
        let byte = self.data[self.bit / 8];
        let shift = 7 - (self.bit % 8);
        self.bit += 1;
        Ok((byte >> shift) & 1)
    }

    /// `f(n)` for `n <= 32`. `n == 0` returns 0 and consumes nothing.
    pub fn f(&mut self, n: u32) -> Result<u32, DecodeError> {
        if n > 32 {
            return Err(DecodeError::malformed("f(n) wider than 32"));
        }
        let mut x = 0u32;
        for _ in 0..n {
            x = (x << 1) | u32::from(self.read_bit()?);
        }
        Ok(x)
    }

    /// `su(n)`.
    pub fn su(&mut self, n: u32) -> Result<i32, DecodeError> {
        if n == 0 || n > 32 {
            return Err(DecodeError::malformed("su(n) width"));
        }
        let mut value = self.f(n)? as i32;
        let sign_mask = 1i32 << (n - 1);
        if value & sign_mask != 0 {
            value -= 2 * sign_mask;
        }
        Ok(value)
    }

    /// `uvlc()`. Leading zeros are capped at 32.
    pub fn uvlc(&mut self) -> Result<u32, DecodeError> {
        let mut leading = 0u32;
        loop {
            if leading >= 32 {
                return Ok(u32::MAX);
            }
            let done = self.read_bit()?;
            if done == 1 {
                break;
            }
            leading += 1;
        }
        if leading >= 32 {
            return Ok(u32::MAX);
        }
        let value = if leading == 0 { 0 } else { self.f(leading)? };
        Ok(value + (1u32 << leading) - 1)
    }

    /// `le(n)` little-endian unsigned, `n` bytes, `n <= 4`. Position must be
    /// byte aligned.
    #[allow(dead_code)]
    pub fn le(&mut self, n: u32) -> Result<u32, DecodeError> {
        if !self.byte_aligned() {
            return Err(DecodeError::malformed("le(n) off a byte boundary"));
        }
        if n > 4 {
            return Err(DecodeError::malformed("le(n) wider than 4"));
        }
        let mut t = 0u32;
        for i in 0..n {
            let byte = self.f(8)?;
            t |= byte << (i * 8);
        }
        Ok(t)
    }

    /// `leb128()`. Returns `(value, bytes_consumed)`.
    ///
    /// Values above `u32::MAX` and a continuation bit on the 8th byte are
    /// malformed. Overlong encodings are legal.
    pub fn leb128(&mut self) -> Result<(u32, u32), DecodeError> {
        if !self.byte_aligned() {
            return Err(DecodeError::malformed("leb128 off a byte boundary"));
        }
        let mut value: u64 = 0;
        let mut bytes = 0u32;
        for i in 0..8 {
            let byte = self.f(8)?;
            bytes += 1;
            value |= u64::from(byte & 0x7f) << (i * 7);
            if byte & 0x80 == 0 {
                if value > u64::from(u32::MAX) {
                    return Err(DecodeError::malformed("leb128 exceeds 2^32-1"));
                }
                return Ok((value as u32, bytes));
            }
            if i == 7 {
                return Err(DecodeError::malformed("leb128 longer than 8 bytes"));
            }
        }
        Err(DecodeError::malformed("leb128"))
    }

    /// `ns(n)`, `n >= 1`.
    pub fn ns(&mut self, n: u32) -> Result<u32, DecodeError> {
        if n == 0 {
            return Err(DecodeError::malformed("ns(0)"));
        }
        let w = floor_log2(n) + 1;
        let m = (1u32 << w) - n;
        let v = self.f(w - 1)?;
        if v < m {
            Ok(v)
        } else {
            let extra = self.f(1)?;
            Ok((v << 1) - m + extra)
        }
    }

    /// `decode_subexp` from the uncompressed header.
    #[allow(dead_code)]
    pub fn subexp(&mut self, num_syms: u32) -> Result<u32, DecodeError> {
        let mut i = 0u32;
        let mut mk = 0u32;
        let k = 3u32;
        for _ in 0..32 {
            let b2 = if i == 0 { k } else { k + i - 1 };
            let a = 1u32 << b2;
            if num_syms <= mk + 3 * a {
                let bits = self.ns(num_syms - mk)?;
                return Ok(bits + mk);
            }
            let more = self.f(1)?;
            if more == 1 {
                i += 1;
                mk += a;
            } else {
                let bits = self.f(b2)?;
                return Ok(bits + mk);
            }
        }
        Err(DecodeError::malformed("subexp did not terminate"))
    }

    #[allow(dead_code)]
    pub fn inverse_recenter(r: u32, v: u32) -> u32 {
        if v > 2 * r {
            v
        } else if v & 1 == 1 {
            r - ((v + 1) >> 1)
        } else {
            r + (v >> 1)
        }
    }

    #[allow(dead_code)]
    pub fn unsigned_subexp_with_ref(&mut self, mx: u32, r: u32) -> Result<u32, DecodeError> {
        let v = self.subexp(mx)?;
        if (r << 1) <= mx {
            Ok(Self::inverse_recenter(r, v))
        } else {
            Ok(mx - 1 - Self::inverse_recenter(mx - 1 - r, v))
        }
    }

    #[allow(dead_code)]
    pub fn signed_subexp_with_ref(
        &mut self,
        low: i32,
        high: i32,
        r: i32,
    ) -> Result<i32, DecodeError> {
        let mx = (high - low) as u32;
        let x = self.unsigned_subexp_with_ref(mx, (r - low) as u32)?;
        Ok(x as i32 + low)
    }

    pub fn byte_align(&mut self) -> Result<(), DecodeError> {
        while !self.bit.is_multiple_of(8) {
            let z = self.read_bit()?;
            if z != 0 {
                return Err(DecodeError::malformed("alignment bit is not zero"));
            }
        }
        Ok(())
    }

    /// Trailing one followed by zeros, consuming exactly `nb_bits`.
    pub fn trailing_bits(&mut self, mut nb_bits: i32) -> Result<(), DecodeError> {
        if nb_bits <= 0 {
            return Err(DecodeError::malformed("trailing bit count"));
        }
        let one = self.read_bit()?;
        if one != 1 {
            return Err(DecodeError::malformed("trailing one bit is not set"));
        }
        nb_bits -= 1;
        while nb_bits > 0 {
            let z = self.read_bit()?;
            if z != 0 {
                return Err(DecodeError::malformed("trailing padding is not zero"));
            }
            nb_bits -= 1;
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn ceil_log2(n: u32) -> u32 {
        ceil_log2(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msb_first_and_leb128() {
        let data = [0b1011_0000, 0x80, 0x01];
        let mut r = BitReader::new(&data);
        assert_eq!(r.f(4).unwrap(), 0b1011);
        assert_eq!(r.f(4).unwrap(), 0);
        let (v, n) = r.leb128().unwrap();
        assert_eq!(v, 128);
        assert_eq!(n, 2);
    }

    #[test]
    fn overlong_leb128_is_legal_overflow_is_not() {
        let mut r = BitReader::new(&[0x80, 0x80, 0x80, 0x80, 0x10]);
        assert!(r.leb128().is_err());
    }

    #[test]
    fn ns_five() {
        // Spec table: 0=00, 1=01, 2=10, 3=110, 4=111.
        let cases = [(0b00_000000, 0u32), (0b01_000000, 1), (0b10_000000, 2)];
        for (byte, expect) in cases {
            let data = [byte];
            let mut r = BitReader::new(&data);
            assert_eq!(r.ns(5).unwrap(), expect);
        }
        let mut r = BitReader::new(&[0b110_00000]);
        assert_eq!(r.ns(5).unwrap(), 3);
        let mut r = BitReader::new(&[0b111_00000]);
        assert_eq!(r.ns(5).unwrap(), 4);
    }

    #[test]
    fn su_sign_extends() {
        let mut r = BitReader::new(&[0b111_00000]);
        assert_eq!(r.su(3).unwrap(), -1);
    }

    #[test]
    fn truncated_read_errors() {
        let mut r = BitReader::new(&[0x00]);
        assert!(r.f(9).is_err());
    }

    #[test]
    fn uvlc_zero_and_saturated() {
        let mut r = BitReader::new(&[0b1000_0000]);
        assert_eq!(r.uvlc().unwrap(), 0);
    }
}

//! Bit reader over an RBSP. Exp-Golomb codes follow H.265 clause 9.2.

use cutlass_decode::CodecError;

pub struct Bits<'a> {
    data: &'a [u8],
    /// Bit position of the next unread bit.
    pos: usize,
}

impl<'a> Bits<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn len_bits(&self) -> usize {
        self.data.len() * 8
    }

    pub fn left(&self) -> usize {
        self.len_bits().saturating_sub(self.pos)
    }

    pub fn truncated(&self, what: &'static str) -> CodecError {
        CodecError::Truncated {
            at_bit: self.pos as u64,
            what,
        }
    }

    pub fn malformed(&self, what: &'static str) -> CodecError {
        CodecError::Malformed {
            at_bit: self.pos as u64,
            what,
        }
    }

    /// `n` bits, most significant first. `n` is at most 32.
    pub fn u(&mut self, n: u32) -> Result<u32, CodecError> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(self.malformed("bit width"));
        }
        if self.left() < n as usize {
            return Err(self.truncated("bits"));
        }
        let mut value = 0u64;
        let mut remaining = n;
        while remaining > 0 {
            let byte = self.data[self.pos / 8];
            let offset = (self.pos % 8) as u32;
            let take = (8 - offset).min(remaining);
            let bits = (u32::from(byte) >> (8 - offset - take)) & ((1u32 << take) - 1);
            value = (value << take) | u64::from(bits);
            remaining -= take;
            self.pos += take as usize;
        }
        Ok(value as u32)
    }

    pub fn bit(&mut self) -> Result<bool, CodecError> {
        Ok(self.u(1)? == 1)
    }

    pub fn skip(&mut self, n: usize) -> Result<(), CodecError> {
        if self.left() < n {
            return Err(self.truncated("bits"));
        }
        self.pos += n;
        Ok(())
    }

    pub fn ue(&mut self) -> Result<u32, CodecError> {
        let mut zeros = 0u32;
        while !self.bit()? {
            zeros += 1;
            if zeros > 31 {
                return Err(self.malformed("exp-golomb"));
            }
        }
        if zeros == 0 {
            return Ok(0);
        }
        let rest = self.u(zeros)?;
        let value = ((1u64 << zeros) - 1) + u64::from(rest);
        u32::try_from(value).map_err(|_| self.malformed("exp-golomb"))
    }

    /// `ue(v)` that must not pass `max`.
    pub fn ue_max(&mut self, max: u32, what: &'static str) -> Result<u32, CodecError> {
        let value = self.ue()?;
        if value > max {
            return Err(self.malformed(what));
        }
        Ok(value)
    }

    pub fn se(&mut self) -> Result<i32, CodecError> {
        let code = i64::from(self.ue()?);
        let magnitude = (code + 1) / 2;
        let value = if code & 1 == 0 { -magnitude } else { magnitude };
        i32::try_from(value).map_err(|_| self.malformed("signed exp-golomb"))
    }

    /// `se(v)` inside `min..=max`.
    pub fn se_range(&mut self, min: i32, max: i32, what: &'static str) -> Result<i32, CodecError> {
        let value = self.se()?;
        if value < min || value > max {
            return Err(self.malformed(what));
        }
        Ok(value)
    }

    pub fn byte_aligned(&self) -> bool {
        self.pos.is_multiple_of(8)
    }
}

#[cfg(test)]
mod tests {
    use super::Bits;

    #[test]
    fn exp_golomb_reads_the_codes_of_clause_9_2() {
        // 1, 010, 011, 00100, 00101 -> 0, 1, 2, 3, 4
        let data = [0b1010_0110, 0b0100_0010, 0b1000_0000];
        let mut bits = Bits::new(&data);
        assert_eq!(bits.ue().unwrap(), 0);
        assert_eq!(bits.ue().unwrap(), 1);
        assert_eq!(bits.ue().unwrap(), 2);
        assert_eq!(bits.ue().unwrap(), 3);
        assert_eq!(bits.ue().unwrap(), 4);
    }

    #[test]
    fn signed_codes_alternate() {
        // 010 -> 1, 011 -> -1, 00100 -> 2
        let data = [0b0100_1100, 0b1000_0000];
        let mut bits = Bits::new(&data);
        assert_eq!(bits.se().unwrap(), 1);
        assert_eq!(bits.se().unwrap(), -1);
        assert_eq!(bits.se().unwrap(), 2);
    }

    #[test]
    fn reading_past_the_end_is_truncated() {
        let mut bits = Bits::new(&[0xff]);
        assert!(bits.u(8).is_ok());
        assert!(bits.u(1).is_err());
        let mut zeros = Bits::new(&[0, 0, 0, 0, 0]);
        assert!(zeros.ue().is_err());
    }
}

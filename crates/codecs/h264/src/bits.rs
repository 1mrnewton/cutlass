//! Bit reader over an RBSP. Exp-Golomb follows clause 9.1.

use cutlass_decode::CodecError;

pub struct Bits<'a> {
    data: &'a [u8],
    /// Next byte that has not been pulled into `cache`.
    index: usize,
    cache: u64,
    /// Valid low bits of `cache`.
    bits: u32,
}

impl<'a> Bits<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            index: 0,
            cache: 0,
            bits: 0,
        }
    }

    pub fn pos(&self) -> usize {
        self.index * 8 - self.bits as usize
    }

    fn fill(&mut self) {
        while self.bits <= 56 && self.index < self.data.len() {
            self.cache = (self.cache << 8) | u64::from(self.data[self.index]);
            self.index += 1;
            self.bits += 8;
        }
    }

    pub fn u(&mut self, n: u32) -> Result<u32, CodecError> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(self.malformed("bit width"));
        }
        self.fill();
        if self.bits < n {
            return Err(self.truncated("bits"));
        }
        self.bits -= n;
        let mask = (1u64 << n) - 1;
        Ok(((self.cache >> self.bits) & mask) as u32)
    }

    pub fn bit(&mut self) -> Result<bool, CodecError> {
        Ok(self.u(1)? == 1)
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
        Ok(((1u32 << zeros) - 1).saturating_add(rest))
    }

    pub fn se(&mut self) -> Result<i32, CodecError> {
        let code = self.ue()?;
        let magnitude = ((code >> 1) + (code & 1)) as i32;
        if code & 1 == 0 {
            Ok(-magnitude)
        } else {
            Ok(magnitude)
        }
    }

    /// Clause 7.2. `true` while a syntax element can still follow.
    pub fn more_rbsp(&self) -> bool {
        let start = self.pos();
        let total = self.data.len() * 8;
        if start >= total {
            return false;
        }
        let mut ones = 0u32;
        for abs in start..total {
            if bit_at(self.data, abs) {
                ones += 1;
            }
        }
        ones > 1 || (ones == 1 && !bit_at(self.data, start))
    }

    pub fn rbsp_trailing(&mut self) -> Result<(), CodecError> {
        if !self.bit()? {
            return Err(self.malformed("rbsp stop bit"));
        }
        while !self.pos().is_multiple_of(8) {
            if self.bit()? {
                return Err(self.malformed("rbsp alignment"));
            }
        }
        Ok(())
    }

    /// Skip to the next byte. CABAC requires the skipped bits to be one.
    pub fn cabac_align(&mut self) -> Result<(), CodecError> {
        while !self.pos().is_multiple_of(8) {
            if !self.bit()? {
                return Err(self.malformed("cabac alignment"));
            }
        }
        Ok(())
    }

    /// Bytes not yet consumed, including a partial current byte.
    pub fn rest(&self) -> &'a [u8] {
        let byte = self.pos() / 8;
        if byte >= self.data.len() {
            &[]
        } else {
            &self.data[byte..]
        }
    }

    pub fn truncated(&self, what: &'static str) -> CodecError {
        CodecError::Truncated {
            at_bit: self.pos() as u64,
            what,
        }
    }

    pub fn malformed(&self, what: &'static str) -> CodecError {
        CodecError::Malformed {
            at_bit: self.pos() as u64,
            what,
        }
    }
}

fn bit_at(data: &[u8], abs: usize) -> bool {
    let byte = data[abs / 8];
    ((byte >> (7 - (abs % 8))) & 1) == 1
}

pub fn vlc(
    bits: &mut Bits<'_>,
    table: &[crate::tables::Vlc],
) -> Result<crate::tables::Vlc, CodecError> {
    let mut acc = 0u32;
    for len in 1..=16 {
        acc = (acc << 1) | bits.u(1)?;
        for code in table {
            if u32::from(code.len) == len && u32::from(code.bits) == acc {
                return Ok(*code);
            }
        }
    }
    Err(bits.malformed("vlc"))
}

#[cfg(test)]
mod tests {
    use super::Bits;

    #[test]
    fn exp_golomb_reads_the_signed_map() {
        // 1, 010, 011, 00100 -> ue 0, 1, 2, 3 and se 0, 1, -1, 2.
        let mut bits = Bits::new(&[0b1010_0110, 0b0100_0000]);
        assert_eq!(bits.ue().unwrap(), 0);
        assert_eq!(bits.ue().unwrap(), 1);
        assert_eq!(bits.ue().unwrap(), 2);
        assert_eq!(bits.se().unwrap(), 2);
    }

    #[test]
    fn more_rbsp_stops_on_the_trailing_one() {
        let mut bits = Bits::new(&[0b1000_0000]);
        assert!(!bits.more_rbsp());
        bits.u(1).unwrap();
        assert!(!bits.more_rbsp());
    }
}

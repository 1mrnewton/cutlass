//! Bit reader over an RBSP. Exp-Golomb follows clause 9.1.

use cutlass_decode::CodecError;

pub struct Bits<'a> {
    data: &'a [u8],
    /// Next byte that has not been pulled into `cache`.
    index: usize,
    cache: u64,
    /// Valid low bits of `cache`.
    bits: u32,
    /// Position of the last bit of `data` that is set, the stop bit of an
    /// RBSP, or 0 when none is.
    stop: usize,
}

impl<'a> Bits<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            index: 0,
            cache: 0,
            bits: 0,
            stop: stop_bit(data),
        }
    }

    /// A reader that starts `bit` bits into `data`.
    pub fn at(data: &'a [u8], bit: usize) -> Result<Self, CodecError> {
        let mut bits = Self {
            data,
            index: (bit / 8).min(data.len()),
            cache: 0,
            bits: 0,
            stop: stop_bit(data),
        };
        let skip = (bit % 8) as u32;
        if skip > 0 {
            bits.u(skip)?;
        }
        Ok(bits)
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

    /// Clause 7.2. `true` while a syntax element can still follow: a bit
    /// after the next one is set. CAVLC asks once a macroblock and SEI once
    /// a message, so this does not look through the rest of the data.
    pub fn more_rbsp(&self) -> bool {
        self.pos() < self.stop
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

fn stop_bit(data: &[u8]) -> usize {
    data.iter()
        .rposition(|&byte| byte != 0)
        .map_or(0, |at| at * 8 + 7 - data[at].trailing_zeros() as usize)
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

    #[test]
    fn more_rbsp_is_true_while_a_bit_after_the_next_is_set() {
        for value in 0..=u16::MAX {
            let data = value.to_be_bytes();
            for at in 0..=16 {
                let bits = Bits::at(&data, at).unwrap();
                let set = (at + 1..16).any(|bit| (value >> (15 - bit)) & 1 == 1);
                assert_eq!(bits.more_rbsp(), set, "{value:016b} at {at}");
            }
        }
    }
}

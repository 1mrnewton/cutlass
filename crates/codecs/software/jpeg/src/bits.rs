//! Entropy bits. JPEG reads them most significant bit first.
//!
//! A `0xFF` data byte is stored as `0xFF 0x00` (T.81 B.1.1.5). A marker that
//! appears while a coefficient is still open is an error. Restart alignment
//! discards the pad bits the encoder stuffed into the last byte.

use crate::error::Error;

pub(crate) struct Bits<'a> {
    data: &'a [u8],
    base: u64,
    pos: usize,
    acc: u32,
    left: u8,
}

impl<'a> Bits<'a> {
    pub(crate) fn new(data: &'a [u8], base: usize) -> Self {
        Self {
            data,
            base: u64::try_from(base).unwrap_or(u64::MAX),
            pos: 0,
            acc: 0,
            left: 0,
        }
    }

    pub(crate) fn bit(&mut self) -> Result<u8, Error> {
        if self.left == 0 {
            let byte = self.next_byte()?;
            self.acc = u32::from(byte);
            self.left = 8;
        }
        self.left -= 1;
        Ok(u8::try_from((self.acc >> self.left) & 1).expect("a bit is 0 or 1"))
    }

    pub(crate) fn bits(&mut self, count: u8) -> Result<u16, Error> {
        if count > 16 {
            return Err(Error::Inconsistent("a coefficient is wider than 16 bits"));
        }
        let mut value = 0u16;
        for _ in 0..count {
            value = (value << 1) | u16::from(self.bit()?);
        }
        Ok(value)
    }

    /// Drop the bits left in the current byte. T.81 F.1.2.3, before a restart.
    pub(crate) fn discard(&mut self) {
        self.left = 0;
        self.acc = 0;
    }

    /// Read a marker at a byte boundary. Extra `0xFF` fill bytes are skipped.
    pub(crate) fn marker(&mut self) -> Result<u8, Error> {
        self.discard();
        if self.pos >= self.data.len() {
            return Err(self.ended(1));
        }
        if self.data[self.pos] != 0xFF {
            return Err(Error::Inconsistent("a restart marker is missing"));
        }
        self.pos += 1;
        while self.pos < self.data.len() && self.data[self.pos] == 0xFF {
            self.pos += 1;
        }
        if self.pos >= self.data.len() {
            return Err(self.ended(1));
        }
        let code = self.data[self.pos];
        self.pos += 1;
        Ok(code)
    }

    fn next_byte(&mut self) -> Result<u8, Error> {
        if self.pos >= self.data.len() {
            return Err(self.ended(1));
        }
        let byte = self.data[self.pos];
        self.pos += 1;
        if byte != 0xFF {
            return Ok(byte);
        }
        if self.pos >= self.data.len() {
            return Err(self.ended(1));
        }
        let next = self.data[self.pos];
        self.pos += 1;
        if next == 0x00 {
            return Ok(0xFF);
        }
        self.pos -= 2;
        Err(Error::Inconsistent("a marker sits inside a coefficient"))
    }

    fn ended(&self, need: usize) -> Error {
        Error::Truncated {
            at: self
                .base
                .saturating_add(u64::try_from(self.pos).unwrap_or(u64::MAX)),
            need: u64::try_from(need).unwrap_or(u64::MAX),
        }
    }
}

/// T.81 F.2.2.1. A size of zero is the zero difference and reads no bits.
pub(crate) fn extend(bits: u16, size: u8) -> Result<i32, Error> {
    if size == 0 {
        return Ok(0);
    }
    if size > 15 {
        return Err(Error::Inconsistent("a coefficient is wider than 15 bits"));
    }
    let magnitude = i32::from(bits);
    let threshold = 1i32 << (size - 1);
    let span = (1i32 << size) - 1;
    Ok(if magnitude < threshold {
        magnitude - span
    } else {
        magnitude
    })
}

#[cfg(test)]
mod tests {
    use super::{Bits, extend};

    #[test]
    fn extend_matches_the_sign_rule() {
        assert_eq!(extend(0, 0).expect("zero"), 0);
        assert_eq!(extend(0, 1).expect("minus one"), -1);
        assert_eq!(extend(1, 1).expect("one"), 1);
        assert_eq!(extend(15, 5).expect("minus sixteen"), -16);
        assert_eq!(extend(16, 5).expect("sixteen"), 16);
        assert_eq!(extend(80, 7).expect("eighty"), 80);
    }

    #[test]
    fn a_stuffed_ff_is_one_data_byte() {
        let mut bits = Bits::new(&[0xFF, 0x00], 0);
        let mut value = 0u16;
        for _ in 0..8 {
            value = (value << 1) | u16::from(bits.bit().expect("bit"));
        }
        assert_eq!(value, 0xFF);
    }
}

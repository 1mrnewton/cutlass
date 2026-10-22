//! Least-significant-bit-first bit reader.
//!
//! WebP lossless 1. The first bit read from a byte is its low bit.

use crate::error::Error;

/// A cursor over one byte slice.
pub(crate) struct Bits<'a> {
    data: &'a [u8],
    index: usize,
    /// Next bit inside `data[index]`, from 0 through 7.
    bit: u8,
}

impl<'a> Bits<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            index: 0,
            bit: 0,
        }
    }

    /// The next bit, as 0 or 1.
    pub(crate) fn read_bit(&mut self) -> Result<u8, Error> {
        let Some(byte) = self.data.get(self.index).copied() else {
            return Err(self.truncated());
        };
        let bit = (byte >> self.bit) & 1;
        self.bit += 1;
        if self.bit == 8 {
            self.bit = 0;
            self.index += 1;
        }
        Ok(bit)
    }

    /// `n` bits, with the first bit read stored as the low bit. `n` is at most 24.
    pub(crate) fn read(&mut self, n: u32) -> Result<u32, Error> {
        if n > 24 {
            return Err(Error::Inconsistent("a bit field is too wide"));
        }
        let mut value = 0u32;
        for shift in 0..n {
            value |= u32::from(self.read_bit()?) << shift;
        }
        Ok(value)
    }

    fn truncated(&self) -> Error {
        Error::Truncated {
            at: u64::try_from(self.index).unwrap_or(u64::MAX),
            need: 1,
        }
    }
}

/// Test streams. Bits are written in the same order [`Bits`] reads them.
#[cfg(test)]
pub(crate) struct Writer {
    data: Vec<u8>,
    acc: u8,
    n: u8,
}

#[cfg(test)]
impl Writer {
    pub(crate) fn new() -> Self {
        Self {
            data: Vec::new(),
            acc: 0,
            n: 0,
        }
    }

    pub(crate) fn bit(&mut self, value: u32) {
        if value & 1 == 1 {
            self.acc |= 1 << self.n;
        }
        self.n += 1;
        if self.n == 8 {
            self.data.push(self.acc);
            self.acc = 0;
            self.n = 0;
        }
    }

    pub(crate) fn bits(&mut self, value: u32, n: u32) {
        for shift in 0..n {
            self.bit((value >> shift) & 1);
        }
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.data.push(self.acc);
        }
        self.data
    }
}

//! Big-endian bit reading and writing.
//!
//! The reader reads zero bits past the end of its slice, so a lookahead near
//! the end never fails. A caller that consumed past the end sees it through
//! [`BitReader::overrun`] and reports [`CodecError::Truncated`].

use cutlass_decode::CodecError;

#[derive(Clone, Debug)]
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    /// Bit offset of `data[0]` inside the unit, for error positions.
    base: u64,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self::at(data, 0)
    }

    pub(crate) fn at(data: &'a [u8], base_bits: u64) -> Self {
        Self {
            data,
            pos: 0,
            base: base_bits,
        }
    }

    /// Absolute bit position, for error messages.
    pub(crate) fn at_bit(&self) -> u64 {
        self.base.saturating_add(self.pos as u64)
    }

    pub(crate) fn bits_left(&self) -> isize {
        (self.data.len() * 8) as isize - self.pos as isize
    }

    /// True once more bits were consumed than the slice holds.
    pub(crate) fn overrun(&self) -> bool {
        self.pos > self.data.len() * 8
    }

    /// The next 32 bits, zero past the end.
    #[inline]
    pub(crate) fn peek32(&self) -> u32 {
        let byte = self.pos >> 3;
        let shift = self.pos & 7;
        let word = if let Some(window) = self.data.get(byte..byte + 8) {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(window);
            u64::from_be_bytes(buf)
        } else {
            let mut buf = [0u8; 8];
            if let Some(tail) = self.data.get(byte..) {
                buf[..tail.len()].copy_from_slice(tail);
            }
            u64::from_be_bytes(buf)
        };
        ((word << shift) >> 32) as u32
    }

    /// The next `n` bits, 1 to 32, zero past the end.
    #[inline]
    pub(crate) fn peek(&self, n: u32) -> u32 {
        debug_assert!((1..=32).contains(&n));
        self.peek32() >> (32 - n)
    }

    #[inline]
    pub(crate) fn skip(&mut self, n: u32) {
        self.pos += n as usize;
    }

    /// Reads `n` bits, 0 to 32.
    #[inline]
    pub(crate) fn read(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let value = self.peek(n);
        self.pos += n as usize;
        value
    }

    #[inline]
    pub(crate) fn read_bit(&mut self) -> bool {
        self.read(1) != 0
    }

    /// Reads `n` bits and fails if the slice ran out.
    pub(crate) fn read_checked(&mut self, n: u32, what: &'static str) -> Result<u32, CodecError> {
        let value = self.read(n);
        if self.overrun() {
            return Err(CodecError::Truncated {
                at_bit: self.at_bit(),
                what,
            });
        }
        Ok(value)
    }

    /// A marker bit that must be one.
    pub(crate) fn marker(&mut self, what: &'static str) -> Result<(), CodecError> {
        let at = self.at_bit();
        if self.read_checked(1, what)? != 1 {
            return Err(CodecError::Malformed { at_bit: at, what });
        }
        Ok(())
    }

    pub(crate) fn truncated(&self, what: &'static str) -> CodecError {
        CodecError::Truncated {
            at_bit: self.at_bit(),
            what,
        }
    }

    pub(crate) fn malformed(&self, what: &'static str) -> CodecError {
        CodecError::Malformed {
            at_bit: self.at_bit(),
            what,
        }
    }

    /// Fails if the reader went past the end.
    pub(crate) fn check(&self, what: &'static str) -> Result<(), CodecError> {
        if self.overrun() {
            Err(self.truncated(what))
        } else {
            Ok(())
        }
    }
}

/// Big-endian bit writer.
#[derive(Clone, Debug, Default)]
pub(crate) struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    count: u32,
}

impl BitWriter {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn with_capacity(bytes: usize) -> Self {
        Self {
            out: Vec::with_capacity(bytes),
            acc: 0,
            count: 0,
        }
    }

    /// Writes the low `n` bits of `value`, `n` from 0 to 32.
    #[inline]
    pub(crate) fn put(&mut self, n: u32, value: u32) {
        if n == 0 {
            return;
        }
        let masked = u64::from(value) & ((1u64 << n) - 1);
        self.acc = (self.acc << n) | masked;
        self.count += n;
        while self.count >= 8 {
            self.count -= 8;
            self.out.push((self.acc >> self.count) as u8);
        }
        self.acc &= (1u64 << self.count) - 1;
    }

    #[inline]
    pub(crate) fn put_bit(&mut self, bit: bool) {
        self.put(1, u32::from(bit));
    }

    /// Bits written so far.
    pub(crate) fn bit_len(&self) -> u64 {
        self.out.len() as u64 * 8 + u64::from(self.count)
    }

    /// Pads with zero bits to a byte boundary.
    pub(crate) fn align(&mut self) {
        if self.count > 0 {
            let pad = 8 - self.count;
            self.put(pad, 0);
        }
    }

    /// Aligns, then writes `00 00 01 code`.
    pub(crate) fn start_code(&mut self, code: u8) {
        self.align();
        self.out.extend_from_slice(&[0, 0, 1, code]);
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
        self.align();
        self.out
    }
}

/// Byte offsets of every `00 00 01` prefix in `data`.
pub(crate) fn start_codes(data: &[u8]) -> Vec<usize> {
    let mut found = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        let third = data[i + 2];
        if third > 1 {
            i += 3;
        } else if third == 1 && data[i] == 0 && data[i + 1] == 0 {
            found.push(i);
            i += 3;
        } else {
            i += 1;
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_across_bytes_and_zero_fills_the_tail() {
        let data = [0b1010_1100, 0b0101_0011];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read(3), 0b101);
        assert_eq!(reader.read(7), 0b011_0001);
        assert_eq!(reader.peek(8), 0b0100_1100);
        assert_eq!(reader.read(6), 0b01_0011);
        assert!(!reader.overrun());
        assert_eq!(reader.read(4), 0);
        assert!(reader.overrun());
        assert!(reader.check("tail").is_err());
    }

    #[test]
    fn read_checked_names_the_truncated_field() {
        let mut reader = BitReader::at(&[0xff], 16);
        assert_eq!(reader.read_checked(8, "byte"), Ok(0xff));
        assert_eq!(
            reader.read_checked(1, "next"),
            Err(CodecError::Truncated {
                at_bit: 25,
                what: "next"
            })
        );
    }

    #[test]
    fn a_zero_marker_is_malformed() {
        let mut reader = BitReader::new(&[0x40]);
        assert!(reader.marker("first").is_err());
        assert!(reader.marker("second").is_ok());
    }

    #[test]
    fn writer_round_trips_through_the_reader() {
        let mut writer = BitWriter::new();
        let fields = [
            (1, 1),
            (5, 0x13),
            (12, 0xabc),
            (32, 0xdead_beef),
            (3, 5),
            (7, 0),
        ];
        for (n, value) in fields {
            writer.put(n, value);
        }
        let bits = writer.bit_len();
        assert_eq!(bits, 60);
        let bytes = writer.finish();
        assert_eq!(bytes.len(), 8);
        let mut reader = BitReader::new(&bytes);
        for (n, value) in fields {
            assert_eq!(reader.read(n), value);
        }
    }

    #[test]
    fn start_codes_are_found_at_every_alignment() {
        let data = [0, 0, 1, 0xb3, 0, 0, 0, 1, 0xb5, 0, 0, 1, 0, 0, 0, 1];
        assert_eq!(start_codes(&data), vec![0, 5, 9, 13]);
        assert!(start_codes(&[0, 0]).is_empty());
        assert!(start_codes(&[0, 1, 0, 0, 2]).is_empty());
    }
}

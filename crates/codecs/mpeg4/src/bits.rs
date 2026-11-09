//! MSB-first bit reading and writing.
//!
//! The reader never panics on short input: bits past the end read as zero
//! and [`BitReader::overrun`] reports it, so a syntax loop checks once per
//! macroblock instead of on every field.

use cutlass_decode::CodecError;

/// Reads bits most significant first from a byte slice.
#[derive(Clone)]
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// Bit position from the start of `data`.
    pos: usize,
    /// Length of `data` in bits.
    len: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            len: data.len() * 8,
        }
    }

    /// Bits consumed so far.
    pub(crate) fn position(&self) -> usize {
        self.pos
    }

    /// Bits left before the end; zero once the reader has run past it.
    pub(crate) fn left(&self) -> usize {
        self.len.saturating_sub(self.pos)
    }

    /// Whether a read went past the end of the input.
    pub(crate) fn overrun(&self) -> bool {
        self.pos > self.len
    }

    /// Moves to an absolute bit position.
    pub(crate) fn seek(&mut self, pos: usize) {
        self.pos = pos;
    }

    /// The next 32 bits without consuming them, zero-filled past the end.
    #[inline]
    pub(crate) fn peek32(&self) -> u32 {
        let byte = self.pos >> 3;
        let shift = (self.pos & 7) as u32;
        if let Some(chunk) = self.data.get(byte..byte + 8)
            && let Ok(bytes) = <[u8; 8]>::try_from(chunk)
        {
            return ((u64::from_be_bytes(bytes) << shift) >> 32) as u32;
        }
        let mut word = 0u64;
        // Five bytes cover 32 bits at any bit offset.
        for i in 0..5 {
            let b = self.data.get(byte + i).copied().unwrap_or(0);
            word = (word << 8) | u64::from(b);
        }
        ((word << shift) >> 8) as u32
    }

    /// The next `n` bits (0..=32) without consuming them.
    #[inline]
    pub(crate) fn peek(&self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        self.peek32() >> (32 - n)
    }

    #[inline]
    pub(crate) fn skip(&mut self, n: u32) {
        self.pos += n as usize;
    }

    /// Reads `n` bits (0..=32) as an unsigned value.
    #[inline]
    pub(crate) fn read(&mut self, n: u32) -> u32 {
        let v = self.peek(n);
        self.pos += n as usize;
        v
    }

    #[inline]
    pub(crate) fn bit(&mut self) -> bool {
        self.read(1) == 1
    }

    /// Reads `n` bits as a two's complement signed value.
    pub(crate) fn read_signed(&mut self, n: u32) -> i32 {
        if n == 0 {
            return 0;
        }
        let v = self.read(n);
        let shift = 32 - n;
        ((v << shift) as i32) >> shift
    }

    /// Reads the H.263 style signed field: `n` bits where a clear top bit
    /// means the negative value `v - (2^n - 1)`.
    pub(crate) fn read_xbits(&mut self, n: u32) -> i32 {
        if n == 0 {
            return 0;
        }
        let v = self.read(n) as i32;
        if v >> (n - 1) == 0 {
            v - ((1 << n) - 1)
        } else {
            v
        }
    }

    /// Consumes a marker bit, which must be one.
    pub(crate) fn marker(&mut self, what: &'static str) -> Result<(), CodecError> {
        let at_bit = self.pos as u64;
        if self.bit() {
            Ok(())
        } else {
            Err(CodecError::Malformed { at_bit, what })
        }
    }

    /// Bits until the next byte boundary, 1 to 8: MPEG-4 stuffing always
    /// takes at least one bit.
    pub(crate) fn bits_to_align(&self) -> u32 {
        8 - (self.pos & 7) as u32
    }

    /// Fails when the reader has run past the end.
    pub(crate) fn check(&self, what: &'static str) -> Result<(), CodecError> {
        if self.overrun() {
            Err(CodecError::Truncated {
                at_bit: self.len as u64,
                what,
            })
        } else {
            Ok(())
        }
    }

    pub(crate) fn malformed(&self, what: &'static str) -> CodecError {
        CodecError::Malformed {
            at_bit: self.pos as u64,
            what,
        }
    }
}

/// Writes bits most significant first.
#[derive(Default)]
pub(crate) struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    /// Bits held in `acc`, always below 8 between calls.
    held: u32,
}

impl BitWriter {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Writes the low `n` bits (0..=32) of `value`.
    #[inline]
    pub(crate) fn put(&mut self, n: u32, value: u32) {
        if n == 0 {
            return;
        }
        let masked = u64::from(value) & ((1u64 << n) - 1);
        self.acc = (self.acc << n) | masked;
        self.held += n;
        while self.held >= 8 {
            self.held -= 8;
            self.out.push((self.acc >> self.held) as u8);
        }
        self.acc &= (1u64 << self.held) - 1;
    }

    pub(crate) fn bit(&mut self, b: bool) {
        self.put(1, u32::from(b));
    }

    /// Writes `value` as an `n`-bit two's complement field.
    pub(crate) fn put_signed(&mut self, n: u32, value: i32) {
        self.put(n, value as u32);
    }

    /// MPEG-4 stuffing: a zero, then ones up to the byte boundary. At
    /// least one bit is always written.
    pub(crate) fn stuff(&mut self) {
        let n = 8 - self.held;
        self.put(n, (1u32 << (n - 1)) - 1);
    }

    /// Zero bits up to the byte boundary.
    pub(crate) fn align_zero(&mut self) {
        if self.held > 0 {
            self.put(8 - self.held, 0);
        }
    }

    /// Whether the writer sits on a byte boundary.
    #[cfg(test)]
    pub(crate) fn aligned(&self) -> bool {
        self.held == 0
    }

    /// Appends whole bytes; the writer must be aligned.
    pub(crate) fn bytes(&mut self, bytes: &[u8]) {
        if self.held == 0 {
            self.out.extend_from_slice(bytes);
        } else {
            for &b in bytes {
                self.put(8, u32::from(b));
            }
        }
    }

    /// The bytes written, zero-padding a partial last byte.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        self.align_zero();
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::{BitReader, BitWriter};

    #[test]
    fn reads_fields_across_bytes_and_past_the_end() {
        let data = [0b1010_1100, 0b0101_0011, 0xFF];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read(3), 0b101);
        assert_eq!(r.read(7), 0b011_0001);
        assert_eq!(r.peek(6), 0b010011);
        assert_eq!(r.read_signed(4), 0b0100);
        assert_eq!(r.read_signed(2), -1);
        assert!(!r.overrun());
        assert_eq!(r.read(8), 0xFF);
        assert_eq!(r.left(), 0);
        assert_eq!(r.read(16), 0, "zeros past the end");
        assert!(r.overrun());
        assert!(r.check("test").is_err());
    }

    #[test]
    fn xbits_maps_a_clear_top_bit_to_negative() {
        // 3 bits: 000 is -7, 011 is -4, 100 is 4, 111 is 7.
        for (bits, want) in [(0b000, -7), (0b011, -4), (0b100, 4), (0b111, 7)] {
            let byte = [(bits as u8) << 5];
            assert_eq!(BitReader::new(&byte).read_xbits(3), want);
        }
    }

    #[test]
    fn writer_round_trips_through_the_reader() {
        let mut w = BitWriter::new();
        w.put(3, 0b101);
        w.put(13, 0x1ABC);
        w.put_signed(5, -3);
        w.stuff();
        assert!(w.aligned());
        w.put(32, 0xDEAD_BEEF);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read(3), 0b101);
        assert_eq!(r.read(13), 0x1ABC);
        assert_eq!(r.read_signed(5), -3);
        // Stuffing after 21 bits: a zero and two ones.
        assert_eq!(r.read(3), 0b011);
        assert_eq!(r.read(32), 0xDEAD_BEEF);
    }

    #[test]
    fn stuffing_is_a_full_byte_when_aligned() {
        let mut w = BitWriter::new();
        w.stuff();
        assert_eq!(w.finish(), vec![0x7F]);
    }
}

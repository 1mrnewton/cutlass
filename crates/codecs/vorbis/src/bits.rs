//! Vorbis bit packing (Vorbis I section 2.1): the first bit of a byte is its
//! least significant, and a value's low bits come first.
//!
//! Reading past the end of a packet is the "end-of-packet" condition, not a
//! failure: the read returns `None`, and every later read returns `None`
//! too (section 2.1.8).

use cutlass_decode::CodecError;

/// Reads LSb-first values from one packet.
#[derive(Clone, Debug)]
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// Next byte to load into `cache`.
    next: usize,
    /// Unread bits, the next one at bit 0.
    cache: u64,
    /// Valid bits in `cache`.
    count: u32,
    /// Set once a read ran past the end.
    ended: bool,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            next: 0,
            cache: 0,
            count: 0,
            ended: false,
        }
    }

    fn refill(&mut self) {
        while self.count <= 56 {
            let Some(&byte) = self.data.get(self.next) else {
                break;
            };
            self.cache |= u64::from(byte) << self.count;
            self.count += 8;
            self.next += 1;
        }
    }

    /// Reads `width` bits (0 to 32) as an unsigned value.
    pub(crate) fn read(&mut self, width: u32) -> Option<u32> {
        if self.ended {
            return None;
        }
        if width == 0 {
            return Some(0);
        }
        if self.count < width {
            self.refill();
            if self.count < width {
                self.ended = true;
                self.count = 0;
                self.cache = 0;
                return None;
            }
        }
        let value = (self.cache & ((1_u64 << width) - 1)) as u32;
        self.cache >>= width;
        self.count -= width;
        Some(value)
    }

    pub(crate) fn read_bit(&mut self) -> Option<bool> {
        self.read(1).map(|bit| bit == 1)
    }

    /// [`Self::read`] where end of packet is an error (header decode).
    pub(crate) fn take(&mut self, width: u32, what: &'static str) -> Result<u32, CodecError> {
        let at_bit = self.position();
        self.read(width)
            .ok_or(CodecError::Truncated { at_bit, what })
    }

    pub(crate) fn take_bit(&mut self, what: &'static str) -> Result<bool, CodecError> {
        self.take(1, what).map(|bit| bit == 1)
    }

    /// The next `width` bits (at most 32) without consuming them, and how
    /// many of them the packet really has. Missing bits read as zero.
    pub(crate) fn peek(&mut self, width: u32) -> (u32, u32) {
        if self.ended {
            return (0, 0);
        }
        if self.count < width {
            self.refill();
        }
        let mask = if width >= 32 {
            u64::from(u32::MAX)
        } else {
            (1_u64 << width) - 1
        };
        ((self.cache & mask) as u32, self.count.min(width))
    }

    /// Consumes `width` bits a [`Self::peek`] showed were there.
    pub(crate) fn consume(&mut self, width: u32) {
        let width = width.min(self.count);
        self.cache = if width >= 64 { 0 } else { self.cache >> width };
        self.count -= width;
    }

    /// Marks the end-of-packet condition.
    pub(crate) fn end(&mut self) {
        self.ended = true;
        self.count = 0;
        self.cache = 0;
    }

    #[cfg(test)]
    pub(crate) fn ended(&self) -> bool {
        self.ended
    }

    /// Bits consumed so far.
    pub(crate) fn position(&self) -> u64 {
        if self.ended {
            return self.data.len() as u64 * 8;
        }
        self.next as u64 * 8 - u64::from(self.count)
    }

    /// Bits not yet read.
    pub(crate) fn remaining(&self) -> u64 {
        (self.data.len() as u64 * 8).saturating_sub(self.position())
    }
}

/// Writes LSb-first values.
#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    cache: u64,
    count: u32,
}

#[cfg(test)]
impl BitWriter {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Writes the low `width` bits (0 to 32) of `value`.
    pub(crate) fn write(&mut self, value: u32, width: u32) {
        if width == 0 {
            return;
        }
        let masked = u64::from(value) & ((1_u64 << width) - 1);
        self.cache |= masked << self.count;
        self.count += width;
        while self.count >= 8 {
            self.bytes.push(self.cache as u8);
            self.cache >>= 8;
            self.count -= 8;
        }
    }

    pub(crate) fn write_bit(&mut self, bit: bool) {
        self.write(u32::from(bit), 1);
    }

    pub(crate) fn write_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write(u32::from(byte), 8);
        }
    }

    /// Bits written so far.
    pub(crate) fn len(&self) -> u64 {
        self.bytes.len() as u64 * 8 + u64::from(self.count)
    }

    /// The packet, zero-padded to a whole byte.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        if self.count > 0 {
            self.bytes.push(self.cache as u8);
        }
        self.bytes
    }
}

/// The position of the highest set bit, counting from 1; 0 for 0
/// (`ilog`, Vorbis I section 9.2.1).
pub(crate) fn ilog(value: u32) -> u32 {
    32 - value.leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::{BitReader, BitWriter, ilog};

    #[test]
    fn values_pack_low_bit_first_as_the_spec_example() {
        // Sections 2.1.6 and 2.1.7: 12 (4 bits), -1 (3), 17 (7), 6969 (13).
        let mut writer = BitWriter::new();
        writer.write(12, 4);
        writer.write(0b111, 3);
        writer.write(17, 7);
        writer.write(6969, 13);
        let bytes = writer.finish();
        assert_eq!(bytes, [0b1111_1100, 0b0100_1000, 0b1100_1110, 0b0000_0110]);
        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read(2), Some(0b00));
        assert_eq!(reader.read(2), Some(0b11));
        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read(4), Some(12));
        assert_eq!(reader.read(3), Some(7));
        assert_eq!(reader.read(7), Some(17));
        assert_eq!(reader.read(13), Some(6969));
        assert_eq!(reader.position(), 27);
    }

    #[test]
    fn reading_past_the_end_sticks_at_end_of_packet() {
        let mut reader = BitReader::new(&[0xff]);
        assert_eq!(reader.read(6), Some(0x3f));
        assert_eq!(reader.read(3), None);
        assert!(reader.ended());
        assert_eq!(reader.read(1), None);
        assert_eq!(reader.read(0), None);
        assert_eq!(reader.peek(4), (0, 0));
    }

    #[test]
    fn a_full_width_value_round_trips() {
        let mut writer = BitWriter::new();
        writer.write(1, 1);
        writer.write(0xdead_beef, 32);
        writer.write(0x7fff_ffff, 31);
        assert_eq!(writer.len(), 64);
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read(1), Some(1));
        assert_eq!(reader.read(32), Some(0xdead_beef));
        assert_eq!(reader.read(31), Some(0x7fff_ffff));
        assert_eq!(reader.remaining(), 0);
    }

    #[test]
    fn peek_reports_how_many_bits_are_real() {
        let mut reader = BitReader::new(&[0b1010_0101]);
        assert_eq!(reader.peek(12), (0b1010_0101, 8));
        reader.consume(3);
        assert_eq!(reader.peek(8), (0b1_0100, 5));
        assert_eq!(reader.position(), 3);
    }

    #[test]
    fn ilog_counts_bits_as_the_spec_table() {
        assert_eq!(ilog(0), 0);
        assert_eq!(ilog(1), 1);
        assert_eq!(ilog(2), 2);
        assert_eq!(ilog(3), 2);
        assert_eq!(ilog(4), 3);
        assert_eq!(ilog(7), 3);
        assert_eq!(ilog(u32::MAX), 32);
    }
}

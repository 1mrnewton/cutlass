//! Big-endian bit reading and writing.
//!
//! ALAC packs every field most significant bit first. A read past the end
//! of a unit is [`CodecError::Truncated`]: a packet is one complete frame,
//! so running out of bits is never a request for more input.

use cutlass_decode::CodecError;

/// Reads bits from one complete access unit.
#[derive(Clone, Debug)]
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// Next bit to read, counted from the first bit of `data`.
    pos: usize,
    /// Bits in `data`.
    end: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            end: data.len().saturating_mul(8),
        }
    }

    #[cfg(test)]
    pub(crate) fn position(&self) -> usize {
        self.pos
    }

    /// Bits left in the unit.
    pub(crate) fn remaining(&self) -> usize {
        self.end - self.pos
    }

    pub(crate) fn at_bit(&self) -> u64 {
        u64::try_from(self.pos).unwrap_or(u64::MAX)
    }

    /// The 32 bits at the read position, zero past the end of the unit.
    /// Nothing is consumed. A later [`Self::skip`] checks the real length.
    #[inline]
    pub(crate) fn peek32(&self) -> u32 {
        let byte = self.pos >> 3;
        let mut word = 0u64;
        if let Some(chunk) = self.data.get(byte..byte + 5) {
            for &b in chunk {
                word = (word << 8) | u64::from(b);
            }
        } else {
            for index in 0..5 {
                word = (word << 8) | u64::from(self.data.get(byte + index).copied().unwrap_or(0));
            }
        }
        let offset = self.pos & 7;
        ((word << offset) >> 8) as u32
    }

    /// Consume `bits` bits. Past the end is [`CodecError::Truncated`].
    #[inline]
    pub(crate) fn skip(&mut self, bits: usize, what: &'static str) -> Result<(), CodecError> {
        let next = self.pos.checked_add(bits).ok_or(CodecError::Truncated {
            at_bit: self.at_bit(),
            what,
        })?;
        if next > self.end {
            return Err(CodecError::Truncated {
                at_bit: self.at_bit(),
                what,
            });
        }
        self.pos = next;
        Ok(())
    }

    /// Read `bits` bits, 0 through 32, as an unsigned value.
    #[inline]
    pub(crate) fn read(&mut self, bits: u32, what: &'static str) -> Result<u32, CodecError> {
        if bits == 0 {
            return Ok(0);
        }
        if bits > 32 {
            return Err(CodecError::InvalidApi { what: "bit count" });
        }
        let value = self.peek32() >> (32 - bits);
        self.skip(bits as usize, what)?;
        Ok(value)
    }

    /// Read `bits` bits, 1 through 32, as a two's-complement value.
    pub(crate) fn read_signed(&mut self, bits: u32, what: &'static str) -> Result<i32, CodecError> {
        let value = self.read(bits, what)?;
        Ok(sign_extend(value as i32, bits))
    }

    /// Move to the next byte boundary. Alignment inside the unit cannot fail.
    pub(crate) fn align(&mut self) {
        let aligned = (self.pos + 7) & !7;
        self.pos = aligned.min(self.end.max(self.pos));
    }
}

/// `value` reduced to its low `bits` bits and sign-extended, the way the
/// reference shifts left then arithmetically right. `bits` is 1 through 32.
#[inline]
pub(crate) fn sign_extend(value: i32, bits: u32) -> i32 {
    let shift = 32u32.saturating_sub(bits).min(31);
    if bits >= 32 {
        return value;
    }
    ((value as u32) << shift) as i32 >> shift
}

/// Somewhere the encoder puts bits: the packet, or a counter for a cost.
pub(crate) trait BitSink {
    /// Append the low `bits` bits of `value`, 0 through 32.
    fn put(&mut self, value: u32, bits: u32);
}

/// Counts bits without storing them.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BitCounter {
    pub(crate) bits: u64,
}

impl BitSink for BitCounter {
    #[inline]
    fn put(&mut self, _value: u32, bits: u32) {
        self.bits += u64::from(bits);
    }
}

/// Writes bits most significant first into a growing byte buffer.
#[derive(Clone, Debug, Default)]
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    acc: u64,
    acc_bits: u32,
}

impl BitWriter {
    pub(crate) fn with_capacity(bytes: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(bytes),
            acc: 0,
            acc_bits: 0,
        }
    }

    /// Bits written so far.
    pub(crate) fn position(&self) -> usize {
        self.bytes.len() * 8 + self.acc_bits as usize
    }

    /// Drop everything written after bit `mark`, an earlier [`Self::position`].
    pub(crate) fn rewind(&mut self, mark: usize) {
        if mark >= self.position() {
            return;
        }
        let byte = mark / 8;
        let rest = (mark % 8) as u32;
        if byte < self.bytes.len() {
            let partial = self.bytes.get(byte).copied().unwrap_or(0);
            self.bytes.truncate(byte);
            self.acc = if rest == 0 {
                0
            } else {
                u64::from(partial >> (8 - rest))
            };
            self.acc_bits = rest;
        } else {
            let drop = self.acc_bits - rest;
            self.acc >>= drop;
            self.acc_bits = rest;
        }
    }

    /// Pad with zero bits to the next byte boundary.
    pub(crate) fn align(&mut self) {
        let pad = (8 - self.acc_bits % 8) % 8;
        self.put(0, pad);
    }

    /// The bytes, once the stream ends on a byte boundary.
    #[cfg(test)]
    pub(crate) fn into_bytes(mut self) -> Vec<u8> {
        self.align();
        self.bytes
    }

    pub(crate) fn clear(&mut self) {
        self.bytes.clear();
        self.acc = 0;
        self.acc_bits = 0;
    }

    /// The finished bytes, copied out, leaving the writer cleared.
    pub(crate) fn take_bytes(&mut self) -> Vec<u8> {
        self.align();
        let bytes = std::mem::take(&mut self.bytes);
        self.clear();
        bytes
    }
}

impl BitSink for BitWriter {
    #[inline]
    fn put(&mut self, value: u32, bits: u32) {
        if bits == 0 {
            return;
        }
        let masked = if bits >= 32 {
            u64::from(value)
        } else {
            u64::from(value & ((1u32 << bits) - 1))
        };
        self.acc = (self.acc << bits) | masked;
        self.acc_bits += bits;
        while self.acc_bits >= 8 {
            self.acc_bits -= 8;
            self.bytes.push((self.acc >> self.acc_bits) as u8);
        }
        if self.acc_bits == 0 {
            self.acc = 0;
        } else {
            self.acc &= (1u64 << self.acc_bits) - 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BitReader, BitSink, BitWriter, sign_extend};

    #[test]
    fn written_fields_read_back_in_order() {
        let mut writer = BitWriter::default();
        let fields: [(u32, u32); 7] = [
            (5, 3),
            (0, 12),
            (0x1ff, 9),
            (0xdead_beef, 32),
            (1, 1),
            (0x12345, 17),
            (0x7fff_ffff, 31),
        ];
        for (value, bits) in fields {
            writer.put(value, bits);
        }
        let bytes = writer.into_bytes();
        let mut reader = BitReader::new(&bytes);
        for (value, bits) in fields {
            assert_eq!(reader.read(bits, "field").unwrap(), value);
        }
    }

    #[test]
    fn a_read_past_the_end_is_truncated() {
        let bytes = [0xff];
        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read(8, "byte").unwrap(), 0xff);
        assert!(matches!(
            reader.read(1, "bit"),
            Err(cutlass_decode::CodecError::Truncated { at_bit: 8, .. })
        ));
        assert_eq!(reader.position(), 8);
    }

    #[test]
    fn rewind_drops_the_tail_inside_and_across_bytes() {
        let mut writer = BitWriter::default();
        writer.put(0b101, 3);
        let mark = writer.position();
        writer.put(0xffff, 16);
        writer.rewind(mark);
        writer.put(0b11, 2);
        assert_eq!(writer.into_bytes(), vec![0b1011_1000]);

        let mut writer = BitWriter::default();
        writer.put(0b1, 1);
        let mark = writer.position();
        writer.put(0b11, 2);
        writer.rewind(mark);
        assert_eq!(writer.position(), 1);
        assert_eq!(writer.into_bytes(), vec![0b1000_0000]);
    }

    #[test]
    fn sign_extension_wraps_to_the_width() {
        assert_eq!(sign_extend(0xffff, 16), -1);
        assert_eq!(sign_extend(0x7fff, 16), 32767);
        assert_eq!(sign_extend(0x1_0000, 16), 0);
        assert_eq!(sign_extend(i32::MIN, 32), i32::MIN);
        assert_eq!(sign_extend(0x80_0000, 24), -8_388_608);
    }

    #[test]
    fn signed_reads_extend_the_top_bit() {
        let bytes = [0xff, 0xf0];
        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read_signed(12, "s12").unwrap(), -1);
    }
}

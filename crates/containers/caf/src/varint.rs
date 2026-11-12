//! The packet table's variable-length integers.
//!
//! Each byte carries seven bits, most significant group first. The high bit
//! is set on every byte except the last (CAF specification, "Packet Table
//! Chunk"; 128 is `0x81 0x00`). A value is at most
//! [`MAX_VARINT_BYTES`](crate::limits::MAX_VARINT_BYTES) bytes and fits in
//! a `u32`; anything longer or larger is refused, so a hostile table cannot
//! overflow a counter.

use crate::limits::MAX_VARINT_BYTES;

/// A table entry that does not decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarintError {
    /// More than [`MAX_VARINT_BYTES`] bytes had the continuation bit.
    TooLong,
    /// The value does not fit in a `u32`.
    TooLarge,
}

/// Decodes one integer a byte at a time, so it may span two reads.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Decoder {
    value: u64,
    len: u32,
}

impl Decoder {
    /// Adds one byte. `Some` is a finished value; the decoder is then empty.
    pub(crate) fn push(&mut self, byte: u8) -> Result<Option<u32>, VarintError> {
        self.len += 1;
        if self.len > MAX_VARINT_BYTES {
            return Err(VarintError::TooLong);
        }
        self.value = (self.value << 7) | u64::from(byte & 0x7F);
        if byte & 0x80 != 0 {
            return Ok(None);
        }
        let value = u32::try_from(self.value).map_err(|_| VarintError::TooLarge)?;
        *self = Self::default();
        Ok(Some(value))
    }

    /// True when a value was started and not finished.
    #[cfg(test)]
    pub(crate) const fn is_partial(&self) -> bool {
        self.len != 0
    }
}

/// Appends `value` in the table's encoding.
pub(crate) fn encode(value: u32, out: &mut Vec<u8>) {
    let mut started = false;
    for shift in [28_u32, 21, 14, 7] {
        let group = (value >> shift) & 0x7F;
        if group != 0 || started {
            started = true;
            out.push(u8::try_from(group).unwrap_or(0) | 0x80);
        }
    }
    out.push(u8::try_from(value & 0x7F).unwrap_or(0));
}

/// Bytes `encode` writes for `value`.
#[cfg(test)]
pub(crate) const fn encoded_len(value: u32) -> u64 {
    match value {
        0..=0x7F => 1,
        0x80..=0x3FFF => 2,
        0x4000..=0x1F_FFFF => 3,
        0x20_0000..=0x0FFF_FFFF => 4,
        _ => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::{Decoder, VarintError, encode, encoded_len};

    fn decode_all(bytes: &[u8]) -> Result<Vec<u32>, VarintError> {
        let mut decoder = Decoder::default();
        let mut out = Vec::new();
        for byte in bytes {
            if let Some(value) = decoder.push(*byte)? {
                out.push(value);
            }
        }
        assert!(!decoder.is_partial());
        Ok(out)
    }

    #[test]
    fn the_specification_examples_decode() {
        assert_eq!(decode_all(&[0x81, 0x00]), Ok(vec![128]));
        assert_eq!(decode_all(&[0x81, 0x80, 0x00]), Ok(vec![16_384]));
        assert_eq!(decode_all(&[0x00, 0x7F]), Ok(vec![0, 127]));
    }

    #[test]
    fn every_width_round_trips() {
        for value in [
            0,
            1,
            127,
            128,
            16_383,
            16_384,
            2_097_151,
            2_097_152,
            268_435_455,
            268_435_456,
            u32::MAX,
        ] {
            let mut out = Vec::new();
            encode(value, &mut out);
            assert_eq!(super::super::bytes::len_u64(out.len()), encoded_len(value));
            assert_eq!(decode_all(&out), Ok(vec![value]), "{value}");
        }
    }

    #[test]
    fn a_long_or_large_value_is_refused() {
        assert_eq!(
            decode_all(&[0x80, 0x80, 0x80, 0x80, 0x80, 0x01]),
            Err(VarintError::TooLong)
        );
        assert_eq!(
            decode_all(&[0xFF, 0xFF, 0xFF, 0xFF, 0x7F]),
            Err(VarintError::TooLarge)
        );
        assert_eq!(
            decode_all(&[0x8F, 0xFF, 0xFF, 0xFF, 0x7F]),
            Ok(vec![u32::MAX])
        );
    }

    #[test]
    fn a_value_may_span_two_reads() {
        let mut decoder = Decoder::default();
        assert_eq!(decoder.push(0x81), Ok(None));
        assert!(decoder.is_partial());
        assert_eq!(decoder.push(0x00), Ok(Some(128)));
        assert!(!decoder.is_partial());
    }
}

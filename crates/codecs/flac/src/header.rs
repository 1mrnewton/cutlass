//! The frame header (RFC 9639, section 9.1).

use cutlass_decode::CodecError;

use crate::bits::{BitReader, BitWriter};
use crate::crc::crc8;

/// Sample rates for codes 1 through 11 (table 15).
const RATES: [u32; 12] = [
    0, 88_200, 176_400, 192_000, 8_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 96_000,
];

/// Bit depths for codes 1, 2, 4, 5, 6 and 7 (table 17). Code 3 is reserved.
const DEPTHS: [u8; 8] = [0, 8, 12, 0, 16, 20, 24, 32];

/// How the coded channels map to output channels (table 16).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelAssignment {
    /// One to eight channels, each coded on its own.
    Independent(u8),
    /// Left, then side (left minus right).
    LeftSide,
    /// Side, then right.
    SideRight,
    /// Mid, then side.
    MidSide,
}

impl ChannelAssignment {
    #[must_use]
    pub const fn channels(self) -> u8 {
        match self {
            Self::Independent(count) => count,
            Self::LeftSide | Self::SideRight | Self::MidSide => 2,
        }
    }

    /// Whether coded channel `index` is a side channel, one bit wider.
    #[must_use]
    pub const fn is_side(self, index: usize) -> bool {
        match self {
            Self::Independent(_) => false,
            Self::LeftSide | Self::MidSide => index == 1,
            Self::SideRight => index == 0,
        }
    }

    const fn code(self) -> u64 {
        match self {
            Self::Independent(count) => count.saturating_sub(1) as u64,
            Self::LeftSide => 8,
            Self::SideRight => 9,
            Self::MidSide => 10,
        }
    }
}

/// One parsed frame header.
///
/// `sample_rate` and `bits` are `None` when the header defers to STREAMINFO.
/// `number` is the frame number in a fixed-block-size stream and the first
/// sample number in a variable-block-size stream. `len` counts the header
/// bytes including the CRC-8.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    pub variable_block: bool,
    pub block_size: u32,
    pub sample_rate: Option<u32>,
    pub assignment: ChannelAssignment,
    pub bits: Option<u8>,
    pub number: u64,
    pub len: usize,
}

/// Bytes in the longest frame header: sync and codes (4), a 7-byte coded
/// number, a 16-bit block size, a 16-bit sample rate, and the CRC-8.
pub const MAX_HEADER_LEN: usize = 16;

fn malformed(at_bit: u64, what: &'static str) -> CodecError {
    CodecError::Malformed { at_bit, what }
}

impl FrameHeader {
    /// Parses a header at the start of `bytes` and checks its CRC-8.
    pub fn parse(bytes: &[u8]) -> Result<Self, CodecError> {
        let mut reader = BitReader::new(bytes);
        let sync = reader.read(15, "frame sync")?;
        if sync != 0x7FFC {
            return Err(malformed(0, "frame sync"));
        }
        let variable_block = reader.read(1, "frame header")? == 1;
        let block_code = reader.read(4, "frame header")?;
        let rate_code = reader.read(4, "frame header")?;
        let channel_code = reader.read(4, "frame header")?;
        let depth_code = reader.read(3, "frame header")?;
        let reserved = reader.read(1, "frame header")?;
        if block_code == 0 {
            return Err(malformed(16, "reserved block size code"));
        }
        if rate_code == 15 {
            return Err(malformed(20, "forbidden sample rate code"));
        }
        let assignment = match channel_code {
            0..=7 => ChannelAssignment::Independent(u8::try_from(channel_code + 1).unwrap_or(8)),
            8 => ChannelAssignment::LeftSide,
            9 => ChannelAssignment::SideRight,
            10 => ChannelAssignment::MidSide,
            _ => return Err(malformed(24, "reserved channel assignment")),
        };
        if depth_code == 3 {
            return Err(malformed(28, "reserved bit depth code"));
        }
        if reserved != 0 {
            return Err(malformed(31, "reserved frame header bit"));
        }
        let number = read_coded_number(&mut reader, variable_block)?;
        let block_size = match block_code {
            1 => 192,
            2..=5 => 576 << (block_code - 2),
            6 => reader.read(8, "frame block size")? + 1,
            7 => {
                let raw = reader.read(16, "frame block size")?;
                if raw == 0xFFFF {
                    return Err(malformed(reader.position(), "forbidden block size 65536"));
                }
                raw + 1
            }
            _ => 256 << (block_code - 8),
        };
        let sample_rate = match rate_code {
            0 => None,
            1..=11 => Some(RATES[usize::try_from(rate_code).unwrap_or(0)]),
            12 => Some(u32::try_from(reader.read(8, "frame sample rate")? * 1000).unwrap_or(0)),
            13 => Some(u32::try_from(reader.read(16, "frame sample rate")?).unwrap_or(0)),
            _ => Some(u32::try_from(reader.read(16, "frame sample rate")? * 10).unwrap_or(0)),
        };
        if sample_rate == Some(0) {
            return Err(malformed(reader.position(), "zero frame sample rate"));
        }
        let bits = match depth_code {
            0 => None,
            code => Some(DEPTHS[usize::try_from(code).unwrap_or(0)]),
        };
        let crc_at = usize::try_from(reader.position() / 8).unwrap_or(usize::MAX);
        let stored = reader.read(8, "frame header crc")?;
        if u64::from(crc8(&bytes[..crc_at])) != stored {
            return Err(malformed(reader.position(), "frame header crc"));
        }
        Ok(Self {
            variable_block,
            block_size: u32::try_from(block_size).unwrap_or(0),
            sample_rate,
            assignment,
            bits,
            number,
            len: crc_at + 1,
        })
    }

    /// The header bytes with their CRC-8.
    ///
    /// A sample rate or bit depth without a code is written as "see
    /// STREAMINFO". `len` is ignored. A block size outside 1..=65535 or a
    /// coded number past its limit is [`CodecError::InvalidApi`].
    pub fn to_bytes(&self) -> Result<Vec<u8>, CodecError> {
        let mut writer = BitWriter::with_capacity(MAX_HEADER_LEN);
        self.write(&mut writer)?;
        Ok(writer.into_bytes())
    }

    pub(crate) fn write(&self, out: &mut BitWriter) -> Result<(), CodecError> {
        let (block_code, block_extra) = match self.block_size {
            192 => (1, None),
            576 | 1152 | 2304 | 4608 => (2 + (self.block_size / 576).trailing_zeros(), None),
            256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 => {
                (8 + (self.block_size / 256).trailing_zeros(), None)
            }
            1..=256 => (6, Some((8, self.block_size - 1))),
            257..=65535 => (7, Some((16, self.block_size - 1))),
            _ => return Err(CodecError::InvalidApi { what: "block size" }),
        };
        let (rate_code, rate_extra) = rate_code(self.sample_rate);
        let depth_code = self
            .bits
            .and_then(|bits| DEPTHS.iter().position(|depth| *depth == bits && bits != 0))
            .unwrap_or(0);
        let limit = if self.variable_block {
            (1u64 << 36) - 1
        } else {
            (1u64 << 31) - 1
        };
        if self.number > limit {
            return Err(CodecError::InvalidApi {
                what: "frame number",
            });
        }
        let mut header = BitWriter::with_capacity(MAX_HEADER_LEN);
        header.put(15, 0x7FFC);
        header.put(1, u64::from(self.variable_block));
        header.put(4, u64::from(block_code));
        header.put(4, rate_code);
        header.put(4, self.assignment.code());
        header.put(3, u64::try_from(depth_code).unwrap_or(0));
        header.put(1, 0);
        write_coded_number(&mut header, self.number);
        if let Some((bits, value)) = block_extra {
            header.put(bits, u64::from(value));
        }
        if let Some((bits, value)) = rate_extra {
            header.put(bits, value);
        }
        let crc = crc8(header.bytes());
        for byte in header.bytes() {
            out.put(8, u64::from(*byte));
        }
        out.put(8, u64::from(crc));
        Ok(())
    }
}

fn rate_code(rate: Option<u32>) -> (u64, Option<(u32, u64)>) {
    let Some(rate) = rate else {
        return (0, None);
    };
    if let Some(code) = RATES.iter().skip(1).position(|known| *known == rate) {
        return (u64::try_from(code + 1).unwrap_or(0), None);
    }
    if rate % 1000 == 0 && rate / 1000 <= 255 {
        return (12, Some((8, u64::from(rate / 1000))));
    }
    if rate <= 65_535 {
        return (13, Some((16, u64::from(rate))));
    }
    if rate % 10 == 0 && rate / 10 <= 65_535 {
        return (14, Some((16, u64::from(rate / 10))));
    }
    (0, None)
}

/// Reads the UTF-8-like coded number (section 9.1.5).
///
/// A fixed-block-size stream allows six bytes (31 bits); a variable one
/// seven bytes (36 bits).
fn read_coded_number(reader: &mut BitReader<'_>, variable: bool) -> Result<u64, CodecError> {
    let at = reader.position();
    let first = reader.read(8, "frame coded number")?;
    let ones = (first << 56).leading_ones();
    let (mut value, extra) = match ones {
        0 => (first, 0),
        2..=6 => (first & (0x7F >> ones), ones - 1),
        7 => (0, 6),
        _ => return Err(malformed(at, "frame coded number")),
    };
    if extra > if variable { 6 } else { 5 } {
        return Err(malformed(at, "frame coded number length"));
    }
    for _ in 0..extra {
        let byte = reader.read(8, "frame coded number")?;
        if byte & 0xC0 != 0x80 {
            return Err(malformed(reader.position(), "frame coded number"));
        }
        value = (value << 6) | (byte & 0x3F);
    }
    Ok(value)
}

fn write_coded_number(out: &mut BitWriter, value: u64) {
    if value < 0x80 {
        out.put(8, value);
        return;
    }
    let extra: u32 = match value {
        0..=0x7FF => 1,
        0x800..=0xFFFF => 2,
        0x1_0000..=0x1F_FFFF => 3,
        0x20_0000..=0x3FF_FFFF => 4,
        0x400_0000..=0x7FFF_FFFF => 5,
        _ => 6,
    };
    let lead = (0xFF00u64 >> (extra + 1)) & 0xFF;
    out.put(8, lead | (value >> (6 * extra)));
    for index in (0..extra).rev() {
        out.put(8, 0x80 | ((value >> (6 * index)) & 0x3F));
    }
}

#[cfg(test)]
mod tests {
    use super::{ChannelAssignment, FrameHeader};
    use crate::bits::BitWriter;

    fn encode(header: &FrameHeader) -> Vec<u8> {
        let mut writer = BitWriter::with_capacity(16);
        header.write(&mut writer).expect("write");
        writer.into_bytes()
    }

    #[test]
    fn a_header_flac_writes_parses() {
        // First frame header of `flac -5` on 44.1 kHz stereo 16-bit:
        // sync, 4096 samples, 44.1 kHz, mid/side, 16 bits, frame 0.
        let bytes = [0xFF, 0xF8, 0xC9, 0xA8, 0x00, 0xF5];
        let mut probe = bytes;
        probe[5] = crate::crc::crc8(&bytes[..5]);
        let header = FrameHeader::parse(&probe).expect("parse");
        assert_eq!(header.block_size, 4096);
        assert_eq!(header.sample_rate, Some(44_100));
        assert_eq!(header.assignment, ChannelAssignment::MidSide);
        assert_eq!(header.bits, Some(16));
        assert_eq!(header.number, 0);
        assert_eq!(header.len, 6);
        assert!(!header.variable_block);
    }

    #[test]
    fn every_field_form_round_trips() {
        let rates = [
            None,
            Some(44_100),
            Some(11_000),
            Some(11_025),
            Some(655_350),
            Some(700_001),
        ];
        let sizes = [1, 16, 192, 1152, 4096, 4097, 65_535];
        let numbers = [0u64, 0x7F, 0x80, 0x7FF, 0xFFFF, 0x1F_FFFF, 0x7FFF_FFFF];
        for rate in rates {
            for size in sizes {
                for number in numbers {
                    let header = FrameHeader {
                        variable_block: false,
                        block_size: size,
                        sample_rate: rate,
                        assignment: ChannelAssignment::Independent(6),
                        bits: Some(24),
                        number,
                        len: 0,
                    };
                    let parsed = FrameHeader::parse(&encode(&header)).expect("parse");
                    let expected_rate = if rate == Some(700_001) { None } else { rate };
                    assert_eq!(parsed.sample_rate, expected_rate);
                    assert_eq!(parsed.block_size, size);
                    assert_eq!(parsed.number, number);
                }
            }
        }
        let variable = FrameHeader {
            variable_block: true,
            block_size: 100,
            sample_rate: None,
            assignment: ChannelAssignment::SideRight,
            bits: Some(13),
            number: (1 << 36) - 1,
            len: 0,
        };
        let parsed = FrameHeader::parse(&encode(&variable)).expect("variable");
        assert_eq!(parsed.number, (1 << 36) - 1);
        assert_eq!(parsed.bits, None);
        assert_eq!(parsed.assignment, ChannelAssignment::SideRight);
    }

    #[test]
    fn reserved_codes_and_a_bad_crc_are_refused() {
        let header = FrameHeader {
            variable_block: false,
            block_size: 4096,
            sample_rate: Some(48_000),
            assignment: ChannelAssignment::LeftSide,
            bits: Some(16),
            number: 3,
            len: 0,
        };
        let good = encode(&header);
        let mut bad_crc = good.clone();
        *bad_crc.last_mut().expect("crc") ^= 1;
        assert!(FrameHeader::parse(&bad_crc).is_err());
        for (byte, mask, value) in [
            (2, 0xF0, 0x00),
            (2, 0x0F, 0x0F),
            (3, 0xF0, 0xB0),
            (3, 0x0E, 0x06),
            (3, 0x01, 0x01),
        ] {
            let mut bytes = good.clone();
            bytes[byte] = (bytes[byte] & !mask) | value;
            let at = bytes.len() - 1;
            bytes[at] = crate::crc::crc8(&bytes[..at]);
            assert!(
                FrameHeader::parse(&bytes).is_err(),
                "byte {byte} value {value:#x}"
            );
        }
        for cut in 0..good.len() {
            assert!(FrameHeader::parse(&good[..cut]).is_err());
        }
    }

    #[test]
    fn a_fixed_stream_number_past_31_bits_is_refused() {
        let header = FrameHeader {
            variable_block: false,
            block_size: 4096,
            sample_rate: None,
            assignment: ChannelAssignment::Independent(1),
            bits: None,
            number: 1 << 31,
            len: 0,
        };
        let mut writer = BitWriter::with_capacity(16);
        assert!(header.write(&mut writer).is_err());
    }
}

//! The parts of RFC 9639 framing a container needs: the STREAMINFO and
//! SEEKTABLE blocks, the frame header, and the two CRCs.
//!
//! The codec crate decodes frames; this crate only finds where they start
//! and end. The header fields are read here so the container stays apart
//! from the codec (the same split the MP3 family keeps).

/// The STREAMINFO body length.
pub const STREAMINFO_LEN: usize = 34;
/// One SEEKTABLE point.
pub const SEEK_POINT_LEN: usize = 18;
/// The sample number of a placeholder seek point.
pub const PLACEHOLDER: u64 = u64::MAX;
/// The longest frame header: sync to CRC-8 with a 7-byte coded number and
/// 16-bit block size and rate fields.
pub const MAX_HEADER_LEN: usize = 16;

const fn crc8_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u8;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 0x80 != 0 {
                (crc << 1) ^ 0x07
            } else {
                crc << 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

const fn crc16_table() -> [u16; 256] {
    let mut table = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = (i as u16) << 8;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x8005
            } else {
                crc << 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

static CRC8: [u8; 256] = crc8_table();
static CRC16: [u16; 256] = crc16_table();

/// CRC-8, polynomial 0x07, initial 0.
#[must_use]
pub fn crc8(data: &[u8]) -> u8 {
    data.iter()
        .fold(0u8, |crc, byte| CRC8[usize::from(crc ^ byte)])
}

/// Continues a CRC-16 (polynomial 0x8005, initial 0) over `byte`.
#[inline]
#[must_use]
pub fn crc16_byte(crc: u16, byte: u8) -> u16 {
    (crc << 8) ^ CRC16[usize::from(((crc >> 8) as u8) ^ byte)]
}

/// CRC-16 over `data`. A frame with its footer gives zero.
#[must_use]
pub fn crc16(data: &[u8]) -> u16 {
    data.iter().fold(0u16, |crc, byte| crc16_byte(crc, *byte))
}

/// The STREAMINFO fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamInfo {
    pub min_block: u16,
    pub max_block: u16,
    pub min_frame: u32,
    pub max_frame: u32,
    pub sample_rate: u32,
    pub channels: u8,
    pub bits: u8,
    /// Zero when unknown.
    pub total_samples: u64,
    /// All zero when unknown.
    pub md5: [u8; 16],
}

impl StreamInfo {
    /// Reads a 34-byte body. A sample rate of zero, a block size under 16,
    /// a minimum above the maximum, or a depth under 4 is refused.
    pub fn parse(body: &[u8]) -> Option<Self> {
        let body: &[u8; STREAMINFO_LEN] = body.try_into().ok()?;
        let min_block = u16::from_be_bytes([body[0], body[1]]);
        let max_block = u16::from_be_bytes([body[2], body[3]]);
        let min_frame = u32::from_be_bytes([0, body[4], body[5], body[6]]);
        let max_frame = u32::from_be_bytes([0, body[7], body[8], body[9]]);
        let packed = u64::from_be_bytes([
            body[10], body[11], body[12], body[13], body[14], body[15], body[16], body[17],
        ]);
        let sample_rate = u32::try_from(packed >> 44).ok()?;
        let channels = u8::try_from((packed >> 41) & 0x7).ok()? + 1;
        let bits = u8::try_from((packed >> 36) & 0x1F).ok()? + 1;
        let total_samples = packed & ((1 << 36) - 1);
        let mut md5 = [0u8; 16];
        md5.copy_from_slice(&body[18..]);
        if sample_rate == 0 || max_block < 16 || min_block > max_block || bits < 4 {
            return None;
        }
        Some(Self {
            min_block,
            max_block,
            min_frame,
            max_frame,
            sample_rate,
            channels,
            bits,
            total_samples,
            md5,
        })
    }

    /// Reads codec extradata: a bare body, or `fLaC` and a STREAMINFO block.
    #[must_use]
    pub fn from_extra(extra: &[u8]) -> Option<Self> {
        match extra.len() {
            STREAMINFO_LEN => Self::parse(extra),
            42 if &extra[..4] == b"fLaC" && extra[4] & 0x7F == 0 => Self::parse(&extra[8..]),
            _ => None,
        }
    }

    /// The 34-byte body. Fields past their width are cut to it.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; STREAMINFO_LEN] {
        let mut out = [0u8; STREAMINFO_LEN];
        out[..2].copy_from_slice(&self.min_block.to_be_bytes());
        out[2..4].copy_from_slice(&self.max_block.to_be_bytes());
        out[4..7].copy_from_slice(&self.min_frame.min(0xFF_FFFF).to_be_bytes()[1..]);
        out[7..10].copy_from_slice(&self.max_frame.min(0xFF_FFFF).to_be_bytes()[1..]);
        let packed = (u64::from(self.sample_rate & 0xF_FFFF) << 44)
            | (u64::from(self.channels.saturating_sub(1) & 0x7) << 41)
            | (u64::from(self.bits.saturating_sub(1) & 0x1F) << 36)
            | (self.total_samples & ((1 << 36) - 1));
        out[10..18].copy_from_slice(&packed.to_be_bytes());
        out[18..].copy_from_slice(&self.md5);
        out
    }
}

/// One SEEKTABLE row: the first sample of a frame and its byte offset from
/// the first frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeekPoint {
    pub sample: u64,
    pub offset: u64,
    pub samples: u16,
}

impl SeekPoint {
    #[must_use]
    pub fn parse(bytes: &[u8; SEEK_POINT_LEN]) -> Self {
        let sample = u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let offset = u64::from_be_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        Self {
            sample,
            offset,
            samples: u16::from_be_bytes([bytes[16], bytes[17]]),
        }
    }

    #[must_use]
    pub fn placeholder() -> Self {
        Self {
            sample: PLACEHOLDER,
            offset: 0,
            samples: 0,
        }
    }

    #[must_use]
    pub fn to_bytes(&self) -> [u8; SEEK_POINT_LEN] {
        let mut out = [0u8; SEEK_POINT_LEN];
        out[..8].copy_from_slice(&self.sample.to_be_bytes());
        out[8..16].copy_from_slice(&self.offset.to_be_bytes());
        out[16..].copy_from_slice(&self.samples.to_be_bytes());
        out
    }
}

/// The frame header fields framing depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    pub variable: bool,
    pub block_size: u32,
    /// `None` defers to STREAMINFO.
    pub sample_rate: Option<u32>,
    pub channels: u8,
    /// `None` defers to STREAMINFO.
    pub bits: Option<u8>,
    /// The frame number (fixed block size) or first sample (variable).
    pub number: u64,
    /// Header bytes, CRC-8 included.
    pub len: usize,
}

/// What a header read found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Header {
    Valid(FrameHeader),
    /// More bytes are needed to decide.
    Short,
    Invalid,
}

const RATES: [u32; 12] = [
    0, 88_200, 176_400, 192_000, 8_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 96_000,
];
const DEPTHS: [u8; 8] = [0, 8, 12, 0, 16, 20, 24, 32];

/// `true` when `bytes` start with a frame sync and a reserved zero bit.
#[inline]
#[must_use]
pub fn is_sync(first: u8, second: u8) -> bool {
    first == 0xFF && second & 0xFE == 0xF8
}

/// Parses a frame header at the start of `bytes`, checking every reserved
/// and forbidden code and the CRC-8.
#[must_use]
pub fn parse_header(bytes: &[u8]) -> Header {
    if bytes.len() < 2 {
        return Header::Short;
    }
    if !is_sync(bytes[0], bytes[1]) {
        return Header::Invalid;
    }
    if bytes.len() < 5 {
        return Header::Short;
    }
    let variable = bytes[1] & 1 == 1;
    let block_code = bytes[2] >> 4;
    let rate_code = bytes[2] & 0xF;
    let channel_code = bytes[3] >> 4;
    let depth_code = usize::from((bytes[3] >> 1) & 0x7);
    if block_code == 0 || rate_code == 15 || channel_code > 10 || depth_code == 3 {
        return Header::Invalid;
    }
    if bytes[3] & 1 != 0 {
        return Header::Invalid;
    }
    // The coded number.
    let first = bytes[4];
    let ones = first.leading_ones();
    let (mut number, extra) = match ones {
        0 => (u64::from(first), 0usize),
        2..=6 => (u64::from(first & (0x7F >> ones)), ones as usize - 1),
        7 => (0, 6),
        _ => return Header::Invalid,
    };
    if extra > if variable { 6 } else { 5 } {
        return Header::Invalid;
    }
    let mut at = 5;
    for _ in 0..extra {
        let Some(&byte) = bytes.get(at) else {
            return Header::Short;
        };
        if byte & 0xC0 != 0x80 {
            return Header::Invalid;
        }
        number = (number << 6) | u64::from(byte & 0x3F);
        at += 1;
    }
    let mut field = |len: usize| -> Option<Option<u32>> {
        let slice = bytes.get(at..at + len)?;
        at += len;
        Some(Some(
            slice.iter().fold(0u32, |v, b| (v << 8) | u32::from(*b)),
        ))
    };
    let block_extra = match block_code {
        6 => field(1),
        7 => field(2),
        _ => Some(None),
    };
    let Some(block_extra) = block_extra else {
        return Header::Short;
    };
    let rate_extra = match rate_code {
        12 => field(1),
        13 | 14 => field(2),
        _ => Some(None),
    };
    let Some(rate_extra) = rate_extra else {
        return Header::Short;
    };
    let Some(&crc) = bytes.get(at) else {
        return Header::Short;
    };
    if crc8(&bytes[..at]) != crc {
        return Header::Invalid;
    }
    let block_size = match (block_code, block_extra) {
        (1, _) => 192,
        (2..=5, _) => 576 << (block_code - 2),
        (6 | 7, Some(value)) => value + 1,
        (_, _) => 256 << (block_code - 8),
    };
    if block_size > 65_535 {
        return Header::Invalid;
    }
    let sample_rate = match (rate_code, rate_extra) {
        (0, _) => None,
        (12, Some(value)) => Some(value * 1000),
        (13, Some(value)) => Some(value),
        (14, Some(value)) => Some(value * 10),
        (code, _) => Some(RATES[usize::from(code)]),
    };
    if sample_rate == Some(0) {
        return Header::Invalid;
    }
    let channels = if channel_code < 8 {
        channel_code + 1
    } else {
        2
    };
    let bits = match DEPTHS[depth_code] {
        0 => None,
        bits => Some(bits),
    };
    Header::Valid(FrameHeader {
        variable,
        block_size,
        sample_rate,
        channels,
        bits,
        number,
        len: at + 1,
    })
}

#[cfg(test)]
mod tests {
    use super::{Header, SeekPoint, StreamInfo, crc8, crc16, parse_header};

    #[test]
    fn crcs_match_their_check_values() {
        assert_eq!(crc8(b"123456789"), 0xF4);
        assert_eq!(crc16(b"123456789"), 0xFEE8);
    }

    #[test]
    fn streaminfo_and_seek_points_round_trip() {
        let info = StreamInfo {
            min_block: 4096,
            max_block: 4096,
            min_frame: 14,
            max_frame: 9000,
            sample_rate: 44_100,
            channels: 2,
            bits: 16,
            total_samples: 123_456_789,
            md5: [7; 16],
        };
        assert_eq!(StreamInfo::parse(&info.to_bytes()), Some(info));
        let mut extra = b"fLaC\x80\x00\x00\x22".to_vec();
        extra.extend_from_slice(&info.to_bytes());
        assert_eq!(StreamInfo::from_extra(&extra), Some(info));
        assert_eq!(StreamInfo::from_extra(&extra[..20]), None);
        let point = SeekPoint {
            sample: 4096,
            offset: 77,
            samples: 4096,
        };
        assert_eq!(SeekPoint::parse(&point.to_bytes()), point);
    }

    #[test]
    fn a_header_parses_and_damage_is_refused() {
        // Fixed, 4096 samples, 44.1 kHz, left/side, 16 bits, frame 0.
        let mut bytes = vec![0xFF, 0xF8, 0xC9, 0x98, 0x00];
        bytes.push(crc8(&bytes));
        let Header::Valid(header) = parse_header(&bytes) else {
            panic!("valid header");
        };
        assert_eq!(
            (
                header.block_size,
                header.sample_rate,
                header.channels,
                header.bits,
                header.len
            ),
            (4096, Some(44_100), 2, Some(16), 6)
        );
        assert_eq!(parse_header(&bytes[..4]), Header::Short);
        for bit in 0..40 {
            let mut damaged = bytes.clone();
            damaged[bit / 8] ^= 0x80 >> (bit % 8);
            if let Header::Valid(other) = parse_header(&damaged) {
                panic!("bit {bit} passed the crc as {other:?}");
            }
        }
    }
}

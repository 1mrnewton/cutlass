//! The STREAMINFO metadata block body (RFC 9639, section 8.2).
//!
//! The codec configuration a container hands over is this 34-byte body.
//! [`StreamInfo::from_extra`] also accepts the body behind a `fLaC` marker
//! and its block header, the form FFmpeg accepts as extradata.

use cutlass_decode::CodecError;

use crate::bits::{BitReader, BitWriter};

/// Bytes in a STREAMINFO body.
pub const STREAMINFO_LEN: usize = 34;

/// Stream parameters and the whole-stream signature.
///
/// `min_frame` and `max_frame` are zero when unknown. `total_samples` is
/// zero when unknown. `md5` is all zeros when the encoder did not compute
/// one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamInfo {
    pub min_block: u16,
    pub max_block: u16,
    pub min_frame: u32,
    pub max_frame: u32,
    pub sample_rate: u32,
    pub channels: u8,
    pub bits: u8,
    pub total_samples: u64,
    pub md5: [u8; 16],
}

impl StreamInfo {
    /// Parses and validates a 34-byte body.
    ///
    /// A maximum block size under 16, a minimum above the maximum, a zero
    /// sample rate, and a bit depth under 4 are refused. A minimum block
    /// size under 16 is accepted: it only describes the stream.
    pub fn parse(body: &[u8]) -> Result<Self, CodecError> {
        if body.len() != STREAMINFO_LEN {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "streaminfo length",
            });
        }
        let mut reader = BitReader::new(body);
        let min_block = read_u16(&mut reader)?;
        let max_block = read_u16(&mut reader)?;
        let min_frame = read_u32(&mut reader, 24)?;
        let max_frame = read_u32(&mut reader, 24)?;
        let sample_rate = read_u32(&mut reader, 20)?;
        let channels = read_u32(&mut reader, 3)? + 1;
        let bits = read_u32(&mut reader, 5)? + 1;
        let total_samples = reader.read(36, "streaminfo")?;
        let mut md5 = [0u8; 16];
        md5.copy_from_slice(&body[18..34]);
        if max_block < 16 {
            return Err(CodecError::Malformed {
                at_bit: 16,
                what: "streaminfo maximum block size",
            });
        }
        if min_block > max_block {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "streaminfo minimum block size",
            });
        }
        if sample_rate == 0 {
            return Err(CodecError::Unsupported {
                what: "streaminfo without a sample rate",
            });
        }
        if bits < 4 {
            return Err(CodecError::Malformed {
                at_bit: 103,
                what: "streaminfo bit depth",
            });
        }
        Ok(Self {
            min_block,
            max_block,
            min_frame,
            max_frame,
            sample_rate,
            channels: u8::try_from(channels).unwrap_or(8),
            bits: u8::try_from(bits).unwrap_or(32),
            total_samples,
            md5,
        })
    }

    /// Reads codec configuration: empty is `None`; a 34-byte body; or the
    /// `fLaC` marker, a STREAMINFO block header, and the body, with any
    /// further metadata blocks after it ignored.
    pub fn from_extra(extra: &[u8]) -> Result<Option<Self>, CodecError> {
        if extra.is_empty() {
            return Ok(None);
        }
        if extra.len() == STREAMINFO_LEN {
            return Self::parse(extra).map(Some);
        }
        if extra.len() >= 8 + STREAMINFO_LEN && extra.starts_with(b"fLaC") {
            let kind = extra[4] & 0x7F;
            let len = u32::from_be_bytes([0, extra[5], extra[6], extra[7]]);
            if kind != 0 || len != 34 {
                return Err(CodecError::Malformed {
                    at_bit: 32,
                    what: "first metadata block is not streaminfo",
                });
            }
            return Self::parse(&extra[8..8 + STREAMINFO_LEN]).map(Some);
        }
        Err(CodecError::Malformed {
            at_bit: 0,
            what: "flac configuration length",
        })
    }

    /// The 34-byte body. Fields wider than their slot are cut to the slot.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; STREAMINFO_LEN] {
        let mut writer = BitWriter::with_capacity(STREAMINFO_LEN);
        writer.put(16, u64::from(self.min_block));
        writer.put(16, u64::from(self.max_block));
        writer.put(24, u64::from(self.min_frame));
        writer.put(24, u64::from(self.max_frame));
        writer.put(20, u64::from(self.sample_rate));
        writer.put(3, u64::from(self.channels.saturating_sub(1)));
        writer.put(5, u64::from(self.bits.saturating_sub(1)));
        writer.put_long(36, self.total_samples);
        let mut out = [0u8; STREAMINFO_LEN];
        out[..18].copy_from_slice(writer.bytes());
        out[18..].copy_from_slice(&self.md5);
        out
    }
}

fn read_u16(reader: &mut BitReader<'_>) -> Result<u16, CodecError> {
    let value = reader.read(16, "streaminfo")?;
    Ok(u16::try_from(value).unwrap_or(u16::MAX))
}

fn read_u32(reader: &mut BitReader<'_>, bits: u32) -> Result<u32, CodecError> {
    let value = reader.read(bits, "streaminfo")?;
    Ok(u32::try_from(value).unwrap_or(u32::MAX))
}

#[cfg(test)]
mod tests {
    use super::{STREAMINFO_LEN, StreamInfo};

    fn sample() -> StreamInfo {
        StreamInfo {
            min_block: 4096,
            max_block: 4096,
            min_frame: 14,
            max_frame: 12_345,
            sample_rate: 44_100,
            channels: 2,
            bits: 16,
            total_samples: 0xF_FFFF_FFFF,
            md5: [7; 16],
        }
    }

    #[test]
    fn a_body_round_trips() {
        let info = sample();
        let bytes = info.to_bytes();
        assert_eq!(StreamInfo::parse(&bytes).expect("parse"), info);
        // Hand-checked field placement: 4096 = 0x1000, rate 44100 = 0xAC44.
        assert_eq!(&bytes[..4], &[0x10, 0x00, 0x10, 0x00]);
        assert_eq!(&bytes[10..13], &[0x0A, 0xC4, 0x42]);
    }

    #[test]
    fn extradata_forms() {
        let body = sample().to_bytes();
        assert_eq!(StreamInfo::from_extra(&[]).expect("empty"), None);
        assert!(StreamInfo::from_extra(&body).expect("body").is_some());
        let mut marked = b"fLaC".to_vec();
        marked.extend_from_slice(&[0x80, 0, 0, 34]);
        marked.extend_from_slice(&body);
        assert!(StreamInfo::from_extra(&marked).expect("marked").is_some());
        marked[4] = 0x84;
        assert!(StreamInfo::from_extra(&marked).is_err());
        assert!(StreamInfo::from_extra(&body[..STREAMINFO_LEN - 1]).is_err());
    }

    #[test]
    fn impossible_values_are_refused() {
        let mut info = sample();
        info.max_block = 15;
        info.min_block = 15;
        assert!(StreamInfo::parse(&info.to_bytes()).is_err());
        let mut info = sample();
        info.min_block = 5000;
        assert!(StreamInfo::parse(&info.to_bytes()).is_err());
        let mut info = sample();
        info.sample_rate = 0;
        assert!(StreamInfo::parse(&info.to_bytes()).is_err());
        let mut info = sample();
        info.bits = 3;
        assert!(StreamInfo::parse(&info.to_bytes()).is_err());
    }
}

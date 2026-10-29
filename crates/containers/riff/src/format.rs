//! `fmt ` chunk: WAVEFORMAT, WAVEFORMATEX, and WAVEFORMATEXTENSIBLE.
//!
//! The channel mask is copied into stream extradata at
//! [`crate::CHANNEL_MASK_OFFSET`]. This crate does not decode samples.

use cutlass_codec::{CodecId, StreamCodec, UnknownCodec};
use cutlass_container::{MuxError, OpenError};

use crate::bytes::{guid_at, u16_at, u32_at};
use crate::limits::{CHANNEL_MASK_OFFSET, EXTRA_HEADER};

/// How sample bytes are split into packets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Framing {
    /// One interleaved sample frame. Duration is one tick of the sample rate.
    Samples,
    /// One `nBlockAlign` block. Duration comes from `fact`, `ds64`, or the block formula.
    Blocks,
    /// MPEG audio frames inside the data chunk.
    Mpeg,
    /// The data chunk is one packet when it fits in the packet limit.
    Whole,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Integer,
    Float,
    Compressed,
}

/// Audio description read from `fmt `.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WavFormat {
    tag: u16,
    channels: u16,
    sample_rate: u32,
    block_align: u16,
    bits: u16,
    valid_bits: u16,
    channel_mask: u32,
    kind: Kind,
    extensible: bool,
    codec: StreamCodec,
    extra: Vec<u8>,
    samples_per_block: Option<u64>,
}

impl WavFormat {
    #[must_use]
    pub const fn tag(&self) -> u16 {
        self.tag
    }

    #[must_use]
    pub const fn channels(&self) -> u16 {
        self.channels
    }

    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[must_use]
    pub const fn block_align(&self) -> u16 {
        self.block_align
    }

    #[must_use]
    pub const fn bits(&self) -> u16 {
        self.bits
    }

    #[must_use]
    pub const fn valid_bits(&self) -> u16 {
        self.valid_bits
    }

    #[must_use]
    pub const fn channel_mask(&self) -> u32 {
        self.channel_mask
    }

    #[must_use]
    pub const fn is_float(&self) -> bool {
        matches!(self.kind, Kind::Float)
    }

    #[must_use]
    pub const fn extensible(&self) -> bool {
        self.extensible
    }

    #[must_use]
    pub const fn codec(&self) -> &StreamCodec {
        &self.codec
    }

    #[must_use]
    pub const fn samples_per_block(&self) -> Option<u64> {
        self.samples_per_block
    }

    pub(crate) const fn framing(&self) -> Framing {
        match self.tag {
            0x0001 | 0x0003 | 0x0006 | 0x0007 => Framing::Samples,
            0x0050 | 0x0055 => Framing::Mpeg,
            _ => {
                if self.block_align > 1 {
                    Framing::Blocks
                } else {
                    Framing::Whole
                }
            }
        }
    }

    /// Linear PCM or float PCM, including an extensible subformat of those tags.
    #[must_use]
    pub fn is_linear_pcm(&self) -> bool {
        matches!(self.tag, 0x0001 | 0x0003)
            && matches!(self.codec, StreamCodec::Known(CodecId::Pcm))
    }

    /// Codec configuration handed to the decoder.
    ///
    /// ```text
    /// 0  u16 format tag, after an extensible subformat is resolved
    /// 2  u16 bits per sample
    /// 4  u16 valid bits, or samples per block when that count fits
    /// 6  u16 block align
    /// 8  u32 channel mask, zero when the file has no mask
    /// 12 u16 0 integer, 1 float, 2 compressed
    /// 14 u16 bit 0 set when the file used WAVEFORMATEXTENSIBLE
    /// 16 .. codec-specific bytes that followed the parsed header
    /// ```
    #[must_use]
    pub fn extradata(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(EXTRA_HEADER + self.extra.len());
        out.extend_from_slice(&self.tag.to_le_bytes());
        out.extend_from_slice(&self.bits.to_le_bytes());
        let field = if self.kind == Kind::Compressed {
            u16::try_from(self.samples_per_block.unwrap_or(0)).unwrap_or(0)
        } else {
            self.valid_bits
        };
        out.extend_from_slice(&field.to_le_bytes());
        out.extend_from_slice(&self.block_align.to_le_bytes());
        out.extend_from_slice(&self.channel_mask.to_le_bytes());
        let kind = match self.kind {
            Kind::Integer => 0u16,
            Kind::Float => 1,
            Kind::Compressed => 2,
        };
        out.extend_from_slice(&kind.to_le_bytes());
        let flags = u16::from(self.extensible);
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&self.extra);
        out
    }

    pub(crate) fn from_extradata(
        bytes: &[u8],
        channels: u16,
        rate: u32,
        codec: &StreamCodec,
    ) -> Result<Self, FormatError> {
        if bytes.is_empty() {
            return default_format(channels, rate, codec);
        }
        if bytes.len() < EXTRA_HEADER {
            return Err(FormatError::Hostile {
                declared: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
                limit: u64::try_from(EXTRA_HEADER).unwrap_or(u64::MAX),
            });
        }
        let tag = u16_at(bytes, 0).map_err(FormatError::from_open)?;
        let bits = u16_at(bytes, 2).map_err(FormatError::from_open)?;
        let field = u16_at(bytes, 4).map_err(FormatError::from_open)?;
        let block_align = u16_at(bytes, 6).map_err(FormatError::from_open)?;
        let channel_mask = u32_at(bytes, CHANNEL_MASK_OFFSET).map_err(FormatError::from_open)?;
        let kind_bits = u16_at(bytes, 12).map_err(FormatError::from_open)?;
        let flags = u16_at(bytes, 14).map_err(FormatError::from_open)?;
        let kind = match kind_bits {
            1 => Kind::Float,
            2 => Kind::Compressed,
            _ => Kind::Integer,
        };
        let (valid_bits, samples_per_block) = if kind == Kind::Compressed {
            let samples = if field == 0 {
                None
            } else {
                Some(u64::from(field))
            };
            (0, samples)
        } else {
            (field, None)
        };
        let mut format = Self {
            tag,
            channels,
            sample_rate: rate,
            block_align,
            bits,
            valid_bits,
            channel_mask,
            kind,
            extensible: flags & 1 == 1,
            codec: codec.clone(),
            extra: bytes[EXTRA_HEADER..].to_vec(),
            samples_per_block,
        };
        if format.block_align == 0 {
            format.block_align = inferred_align(bits, channels)?;
        }
        check_rate_channels(rate, channels)?;
        if format.framing() == Framing::Samples {
            check_sample_width(&format)?;
        }
        Ok(format)
    }

    pub(crate) fn fmt_payload(&self, avg_bytes: u32) -> Result<Vec<u8>, FormatError> {
        let extensible = self.extensible
            || self.channel_mask != 0
            || self.channels > 2
            || (self.valid_bits != 0
                && self.valid_bits != self.bits
                && self.kind != Kind::Compressed);
        let mut out = Vec::new();
        let stored_tag = if extensible { 0xFFFE } else { self.tag };
        out.extend_from_slice(&stored_tag.to_le_bytes());
        out.extend_from_slice(&self.channels.to_le_bytes());
        out.extend_from_slice(&self.sample_rate.to_le_bytes());
        out.extend_from_slice(&avg_bytes.to_le_bytes());
        out.extend_from_slice(&self.block_align.to_le_bytes());
        out.extend_from_slice(&self.bits.to_le_bytes());
        if extensible {
            out.extend_from_slice(&22u16.to_le_bytes());
            let valid = if self.valid_bits == 0 {
                self.bits
            } else {
                self.valid_bits
            };
            out.extend_from_slice(&valid.to_le_bytes());
            out.extend_from_slice(&self.channel_mask.to_le_bytes());
            write_guid(&mut out, self.tag);
            out.extend_from_slice(&self.extra);
        } else if self.tag != 0x0001 || !self.extra.is_empty() {
            let extra_len = u16::try_from(self.extra.len()).map_err(|_| FormatError::Hostile {
                declared: u64::try_from(self.extra.len()).unwrap_or(u64::MAX),
                limit: u64::from(u16::MAX),
            })?;
            out.extend_from_slice(&extra_len.to_le_bytes());
            out.extend_from_slice(&self.extra);
        }
        Ok(out)
    }
}

/// Bytes of one `fmt ` payload.
pub(crate) fn parse_fmt(payload: &[u8]) -> Result<WavFormat, FormatError> {
    if payload.len() < 14 {
        return Err(FormatError::Truncated {
            expected: 14,
            got: u64::try_from(payload.len()).unwrap_or(u64::MAX),
        });
    }
    let raw_tag = u16_at(payload, 0).map_err(FormatError::from_open)?;
    let channels = u16_at(payload, 2).map_err(FormatError::from_open)?;
    let sample_rate = u32_at(payload, 4).map_err(FormatError::from_open)?;
    let block_align = u16_at(payload, 12).map_err(FormatError::from_open)?;
    check_rate_channels(sample_rate, channels)?;
    if block_align == 0 {
        return Err(FormatError::Bad("block align is zero"));
    }
    let bits = if payload.len() == 14 {
        8
    } else if payload.len() == 15 {
        return Err(FormatError::Truncated {
            expected: 16,
            got: 15,
        });
    } else {
        u16_at(payload, 14).map_err(FormatError::from_open)?
    };

    let mut tag = raw_tag;
    let mut valid_bits = 0u16;
    let mut channel_mask = 0u32;
    let mut extensible = false;
    let mut extra = Vec::new();
    let mut guid_unknown: Option<[u8; 16]> = None;

    if payload.len() >= 18 && raw_tag != 0x0165 {
        let cb_size = u16_at(payload, 16).map_err(FormatError::from_open)?;
        let available = payload.len().saturating_sub(18);
        let cb = usize::from(cb_size).min(available);
        if raw_tag == 0xFFFE {
            if cb < 22 {
                return Err(FormatError::Truncated {
                    expected: 40,
                    got: u64::try_from(payload.len()).unwrap_or(u64::MAX),
                });
            }
            extensible = true;
            valid_bits = u16_at(payload, 18).map_err(FormatError::from_open)?;
            channel_mask = u32_at(payload, 20).map_err(FormatError::from_open)?;
            let guid = guid_at(payload, 24).map_err(FormatError::from_open)?;
            match subformat_tag(&guid) {
                Some(resolved) => tag = resolved,
                None => guid_unknown = Some(guid),
            }
            if cb > 22 {
                extra = payload[40..18 + cb].to_vec();
            }
        } else if cb > 0 {
            extra = payload[18..18 + cb].to_vec();
        }
    }

    if valid_bits > bits && bits > 0 {
        return Err(FormatError::Bad("valid bits exceed the container width"));
    }

    let codec = if let Some(guid) = guid_unknown {
        named_unknown(&guid_name(&guid))?
    } else {
        codec_for_tag(tag)?
    };
    let kind = kind_for(tag, &codec);
    let samples_per_block = samples_in_block(tag, channels, block_align, &extra);
    let format = WavFormat {
        tag,
        channels,
        sample_rate,
        block_align,
        bits,
        valid_bits,
        channel_mask,
        kind,
        extensible,
        codec,
        extra,
        samples_per_block,
    };
    if format.framing() == Framing::Samples {
        check_sample_width(&format)?;
    }
    Ok(format)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FormatError {
    Truncated { expected: u64, got: u64 },
    Hostile { declared: u64, limit: u64 },
    Bad(&'static str),
}

impl FormatError {
    pub(crate) fn into_open(self) -> OpenError {
        match self {
            Self::Truncated { expected, got } => OpenError::Truncated { expected, got },
            Self::Hostile { declared, limit } => OpenError::HostileSize { declared, limit },
            Self::Bad(message) => OpenError::Source {
                message: message.to_owned(),
            },
        }
    }

    pub(crate) fn into_mux(self) -> MuxError {
        match self {
            Self::Hostile { declared, limit } => MuxError::HostileSize { declared, limit },
            Self::Truncated { expected, got } => MuxError::Rejected {
                message: format!("wav header truncated: expected {expected}, got {got}"),
            },
            Self::Bad(message) => MuxError::Rejected {
                message: message.to_owned(),
            },
        }
    }

    fn from_open(error: OpenError) -> Self {
        match error {
            OpenError::Truncated { expected, got } => Self::Truncated { expected, got },
            OpenError::HostileSize { declared, limit } => Self::Hostile { declared, limit },
            OpenError::Unrecognized { .. }
            | OpenError::UnsupportedBrand { .. }
            | OpenError::DuplicateTrack { .. }
            | OpenError::Source { .. } => Self::Bad("fmt field"),
        }
    }
}

fn default_format(channels: u16, rate: u32, codec: &StreamCodec) -> Result<WavFormat, FormatError> {
    check_rate_channels(rate, channels)?;
    let (tag, bits, kind) = match codec {
        StreamCodec::Known(CodecId::Pcm) => (0x0001, 16, Kind::Integer),
        StreamCodec::Known(CodecId::Mp3) => (0x0055, 0, Kind::Compressed),
        StreamCodec::Known(CodecId::Aac) => (0x00FF, 0, Kind::Compressed),
        StreamCodec::Known(CodecId::Ac3) => (0x2000, 0, Kind::Compressed),
        StreamCodec::Known(CodecId::Flac) => (0xF1AC, 0, Kind::Compressed),
        StreamCodec::Known(_) => {
            return Err(FormatError::Bad("codec cannot be stored in wav"));
        }
        StreamCodec::Unknown(_) => {
            return Err(FormatError::Bad("unknown codec cannot be stored in wav"));
        }
    };
    let block_align = if kind == Kind::Integer {
        inferred_align(bits, channels)?
    } else {
        1
    };
    Ok(WavFormat {
        tag,
        channels,
        sample_rate: rate,
        block_align,
        bits,
        valid_bits: 0,
        channel_mask: 0,
        kind,
        extensible: false,
        codec: codec.clone(),
        extra: Vec::new(),
        samples_per_block: None,
    })
}

fn inferred_align(bits: u16, channels: u16) -> Result<u16, FormatError> {
    if bits == 0 || !bits.is_multiple_of(8) {
        return Err(FormatError::Bad("pcm width is not a whole number of bytes"));
    }
    let width = bits / 8;
    let align = u32::from(width) * u32::from(channels);
    u16::try_from(align).map_err(|_| FormatError::Bad("block align does not fit"))
}

fn check_rate_channels(rate: u32, channels: u16) -> Result<(), FormatError> {
    if rate == 0 {
        return Err(FormatError::Bad("sample rate is zero"));
    }
    if channels == 0 {
        return Err(FormatError::Bad("channel count is zero"));
    }
    Ok(())
}

fn check_sample_width(format: &WavFormat) -> Result<(), FormatError> {
    let allowed = match format.tag {
        0x0003 => matches!(format.bits, 32 | 64),
        0x0006 | 0x0007 => format.bits == 8,
        _ => matches!(format.bits, 8 | 16 | 24 | 32 | 64),
    };
    if !allowed {
        return Err(FormatError::Bad("pcm width is not supported"));
    }
    let align = inferred_align(format.bits, format.channels)?;
    if format.block_align != align {
        return Err(FormatError::Bad("block align does not match the pcm width"));
    }
    Ok(())
}

fn kind_for(tag: u16, codec: &StreamCodec) -> Kind {
    if tag == 0x0003 {
        Kind::Float
    } else if matches!(codec, StreamCodec::Known(CodecId::Pcm)) && tag == 0x0001 {
        Kind::Integer
    } else if matches!(tag, 0x0006 | 0x0007) {
        Kind::Compressed
    } else if matches!(codec, StreamCodec::Known(CodecId::Pcm)) {
        Kind::Integer
    } else {
        Kind::Compressed
    }
}

fn codec_for_tag(tag: u16) -> Result<StreamCodec, FormatError> {
    let known = match tag {
        0x0001 | 0x0003 => Some(CodecId::Pcm),
        0x0055 => Some(CodecId::Mp3),
        0x00FF | 0x1600 | 0x706D | 0x4143 | 0xA106 => Some(CodecId::Aac),
        0x2000 => Some(CodecId::Ac3),
        0xF1AC => Some(CodecId::Flac),
        _ => None,
    };
    if let Some(id) = known {
        return Ok(StreamCodec::Known(id));
    }
    let mut text = b"wav:".to_vec();
    push_hex_u16(&mut text, tag);
    named_unknown(&text)
}

fn samples_in_block(tag: u16, channels: u16, block_align: u16, extra: &[u8]) -> Option<u64> {
    match tag {
        0x0011 => ima_samples(channels, block_align),
        0x0002 => extra.get(..2).and_then(|bytes| {
            let value = u16::from_le_bytes([bytes[0], bytes[1]]);
            if value == 0 {
                None
            } else {
                Some(u64::from(value))
            }
        }),
        _ => None,
    }
}

fn ima_samples(channels: u16, block_align: u16) -> Option<u64> {
    let channels = u64::from(channels);
    let block = u64::from(block_align);
    let header = channels.checked_mul(4)?;
    if block <= header || channels == 0 {
        return None;
    }
    Some(1 + (block - header) * 2 / channels)
}

/// Media-subtype GUID tail: `{xxxxxxxx-0000-0010-8000-00aa00389b71}`.
fn subformat_tag(guid: &[u8; 16]) -> Option<u16> {
    const TAIL: [u8; 12] = [
        0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
    ];
    if guid.get(4..)? != TAIL {
        return None;
    }
    let data1 = u32::from_le_bytes([guid[0], guid[1], guid[2], guid[3]]);
    u16::try_from(data1).ok()
}

fn write_guid(out: &mut Vec<u8>, tag: u16) {
    out.extend_from_slice(&u32::from(tag).to_le_bytes());
    out.extend_from_slice(&0x0010_0000u32.to_le_bytes());
    out.extend_from_slice(&0xAA00_0080u32.to_le_bytes());
    out.extend_from_slice(&0x719B_3800u32.to_le_bytes());
}

fn named_unknown(bytes: &[u8]) -> Result<StreamCodec, FormatError> {
    let len = bytes.len().min(UnknownCodec::MAX_TAG);
    UnknownCodec::new(bytes[..len].to_vec())
        .map(StreamCodec::Unknown)
        .map_err(|_| FormatError::Bad("codec tag"))
}

fn guid_name(guid: &[u8; 16]) -> Vec<u8> {
    let mut out = b"wav-guid:".to_vec();
    for byte in guid {
        push_hex(&mut out, *byte);
    }
    out
}

fn push_hex_u16(out: &mut Vec<u8>, value: u16) {
    push_hex(out, u8::try_from(value >> 8).unwrap_or(0));
    push_hex(out, u8::try_from(value & 0x00FF).unwrap_or(0));
}

fn push_hex(out: &mut Vec<u8>, byte: u8) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let hi = usize::from(byte >> 4);
    let lo = usize::from(byte & 0x0F);
    out.push(HEX[hi]);
    out.push(HEX[lo]);
}

#[cfg(test)]
mod tests {
    use super::parse_fmt;
    use crate::limits::CHANNEL_MASK_OFFSET;
    use cutlass_codec::{CodecId, StreamCodec};

    fn pcm_fmt(channels: u16, rate: u32, bits: u16) -> Vec<u8> {
        let align = (bits / 8) * channels;
        let avg = rate * u32::from(align);
        let mut out = Vec::new();
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&avg.to_le_bytes());
        out.extend_from_slice(&align.to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out
    }

    #[test]
    fn classic_pcm_is_integer_and_has_no_mask() {
        let format = parse_fmt(&pcm_fmt(2, 44_100, 16)).expect("fmt");
        assert_eq!(format.tag(), 1);
        assert_eq!(format.sample_rate(), 44_100);
        assert_eq!(format.channels(), 2);
        assert_eq!(format.block_align(), 4);
        assert_eq!(format.channel_mask(), 0);
        assert!(!format.is_float());
        assert_eq!(format.codec(), &StreamCodec::Known(CodecId::Pcm));
        assert_ne!(format.sample_rate(), 1_000_000);
    }

    #[test]
    fn an_extensible_mask_is_at_the_documented_extradata_offset() {
        assert_eq!(CHANNEL_MASK_OFFSET, 8);
        let mut fmt = pcm_fmt(2, 48_000, 16);
        fmt[0] = 0xFE;
        fmt[1] = 0xFF;
        fmt.extend_from_slice(&22u16.to_le_bytes());
        fmt.extend_from_slice(&16u16.to_le_bytes());
        let mask = 0x0000_0003u32;
        fmt.extend_from_slice(&mask.to_le_bytes());
        fmt.extend_from_slice(&1u32.to_le_bytes());
        fmt.extend_from_slice(&0x0010_0000u32.to_le_bytes());
        fmt.extend_from_slice(&0xAA00_0080u32.to_le_bytes());
        fmt.extend_from_slice(&0x719B_3800u32.to_le_bytes());
        let format = parse_fmt(&fmt).expect("extensible");
        assert!(format.extensible());
        assert_eq!(format.channel_mask(), mask);
        assert_eq!(format.codec(), &StreamCodec::Known(CodecId::Pcm));
        let extra = format.extradata();
        assert_eq!(
            &extra[CHANNEL_MASK_OFFSET..CHANNEL_MASK_OFFSET + 4],
            &mask.to_le_bytes()
        );
    }

    #[test]
    fn an_unknown_subformat_is_a_named_codec() {
        let mut fmt = pcm_fmt(1, 8_000, 8);
        fmt[0] = 0xFE;
        fmt[1] = 0xFF;
        fmt.extend_from_slice(&22u16.to_le_bytes());
        fmt.extend_from_slice(&8u16.to_le_bytes());
        fmt.extend_from_slice(&0u32.to_le_bytes());
        fmt.extend_from_slice(&[0xAB; 16]);
        let format = parse_fmt(&fmt).expect("unknown");
        let StreamCodec::Unknown(tag) = format.codec() else {
            panic!("known codec");
        };
        let text = String::from_utf8_lossy(tag.tag());
        assert!(text.starts_with("wav-guid:"), "{text}");
    }

    #[test]
    fn a_short_fmt_is_truncated() {
        let error = parse_fmt(&[0; 8]).expect_err("short");
        assert!(matches!(
            error,
            super::FormatError::Truncated { expected: 14, .. }
        ));
    }
}

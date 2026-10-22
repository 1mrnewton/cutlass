//! An AIFF or AIFC form.

use std::num::NonZeroU32;

use cutlass_core::{AudioCodec, AudioStreamConfig, Packet, Timestamp};

use crate::Error;
use crate::iff::{self, Chunk, Cursor};

const FORM: [u8; 4] = *b"FORM";
const AIFF: [u8; 4] = *b"AIFF";
const AIFC: [u8; 4] = *b"AIFC";
const COMM: [u8; 4] = *b"COMM";
const SSND: [u8; 4] = *b"SSND";

const COMM_AIFF: usize = 18;
const COMM_ID: usize = 22;

/// A parsed AIFF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Movie {
    tracks: Vec<Track>,
}

impl Movie {
    /// Read a file that is already in memory.
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        let mut cursor = Cursor::new(data);
        let Some(form) = cursor.next_chunk()? else {
            return Err(Error::Inconsistent("the file has no FORM header"));
        };
        if form.id != FORM {
            return Err(Error::Inconsistent("the file has no FORM header"));
        }
        if form.data.len() < 4 {
            return Err(Error::Inconsistent("the file has no form"));
        }
        let kind: [u8; 4] = match form.data[..4].try_into() {
            Ok(kind) => kind,
            Err(_) => return Err(Error::Inconsistent("the file has no form")),
        };
        let aifc = match kind {
            AIFF => false,
            AIFC => true,
            _ => return Err(Error::Form),
        };
        let movie = parse_form(data, &form, aifc)?;
        if cursor.next_chunk()?.is_some() {
            return Err(Error::Inconsistent("the file has data after the form"));
        }
        Ok(movie)
    }

    /// The one audio track.
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }
}

/// The audio in the file. Samples are the sound chunk, in stored order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Track {
    timescale: NonZeroU32,
    kind: Kind,
    samples: Vec<Packet>,
}

impl Track {
    /// Ticks per second. This is the sample rate.
    pub fn timescale(&self) -> NonZeroU32 {
        self.timescale
    }

    /// Codec, and the settings a decoder needs.
    pub fn kind(&self) -> &Kind {
        &self.kind
    }

    /// Samples in stored order. `dts` is always none.
    pub fn samples(&self) -> &[Packet] {
        &self.samples
    }
}

/// What the track carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    Audio(AudioStreamConfig),
    /// A compression id this reader does not map. `codec_id` is that id, and
    /// `config_record` is the common chunk through the compression name.
    Other {
        codec_id: String,
        config_record: Vec<u8>,
    },
}

fn parse_form(file: &[u8], form: &Chunk<'_>, aifc: bool) -> Result<Movie, Error> {
    let mut cursor = iff::children(file, form)?;
    let mut common = None;
    let mut sound = None;
    while let Some(chunk) = cursor.next_chunk()? {
        match chunk.id {
            COMM => {
                if common.is_some() {
                    return Err(Error::Inconsistent("the common chunk is repeated"));
                }
                common = Some(chunk.data);
            }
            SSND => {
                if sound.is_some() {
                    return Err(Error::Inconsistent("the sound chunk is repeated"));
                }
                sound = Some(chunk.data);
            }
            _ => {}
        }
    }
    let common = common.ok_or(Error::Inconsistent("the file has no common chunk"))?;
    let header = read_common(common, aifc)?;
    let samples = match sound {
        Some(sound) => vec![packet(&header, sound_bytes(sound)?)?],
        None => Vec::new(),
    };
    Ok(Movie {
        tracks: vec![Track {
            timescale: header.rate,
            kind: header.kind,
            samples,
        }],
    })
}

struct Header {
    rate: NonZeroU32,
    kind: Kind,
    channels: u16,
    bits: u16,
    frames: u32,
    pcm: bool,
}

fn read_common(data: &[u8], aifc: bool) -> Result<Header, Error> {
    let config_end = config_len(data, aifc)?;
    let channels = match u16::try_from(i16_at(data, 0)?) {
        Ok(channels) => channels,
        Err(_) => return Err(Error::Inconsistent("the channel count is invalid")),
    };
    let frames = u32_at(data, 2)?;
    let bits = match u16::try_from(i16_at(data, 6)?) {
        Ok(bits) => bits,
        Err(_) => return Err(Error::Inconsistent("the sample width is invalid")),
    };
    let rate = match NonZeroU32::new(whole_rate(bytes_at(data, 8)?)?) {
        Some(rate) => rate,
        None => return Err(Error::Inconsistent("the sample rate is zero")),
    };
    let config_record = data[..config_end].to_vec();
    let (pcm, kind) = if aifc {
        let id = bytes_at::<4>(data, COMM_AIFF)?;
        if pcm_id(id) {
            (true, pcm_kind(rate, channels, config_record))
        } else {
            (
                false,
                Kind::Other {
                    codec_id: codec_id(id),
                    config_record,
                },
            )
        }
    } else {
        (true, pcm_kind(rate, channels, config_record))
    };
    if pcm {
        check_pcm(channels, bits)?;
    }
    Ok(Header {
        rate,
        kind,
        channels,
        bits,
        frames,
        pcm,
    })
}

fn pcm_kind(rate: NonZeroU32, channels: u16, config_record: Vec<u8>) -> Kind {
    Kind::Audio(AudioStreamConfig {
        codec: AudioCodec::Pcm,
        sample_rate: rate.get(),
        channels,
        config_record,
    })
}

fn config_len(data: &[u8], aifc: bool) -> Result<usize, Error> {
    if data.len() < COMM_AIFF {
        return Err(Error::Inconsistent("the common chunk is too short"));
    }
    if !aifc {
        return Ok(COMM_AIFF);
    }
    if data.len() < COMM_ID {
        return Err(Error::Inconsistent("the common chunk is too short"));
    }
    if data.len() < COMM_ID + 1 {
        return Err(Error::Inconsistent("the compression name does not fit"));
    }
    let name_len = usize::from(data[COMM_ID]);
    let end = COMM_ID
        .checked_add(1)
        .and_then(|start| start.checked_add(name_len))
        .ok_or(Error::TooLarge)?;
    if end > data.len() {
        return Err(Error::Inconsistent("the compression name does not fit"));
    }
    Ok(end)
}

fn pcm_id(id: [u8; 4]) -> bool {
    matches!(
        &id,
        b"NONE"
            | b"twos"
            | b"sowt"
            | b"raw "
            | b"in24"
            | b"in32"
            | b"fl32"
            | b"fl64"
            | b"FL32"
            | b"FL64"
    )
}

fn check_pcm(channels: u16, bits: u16) -> Result<(), Error> {
    if channels == 0 {
        return Err(Error::Inconsistent("the channel count is zero"));
    }
    if bits == 0 {
        return Err(Error::Inconsistent("the sample width is zero"));
    }
    if !bits.is_multiple_of(8) {
        return Err(Error::Inconsistent(
            "the sample width is not a whole number of bytes",
        ));
    }
    Ok(())
}

fn sound_bytes(data: &[u8]) -> Result<&[u8], Error> {
    if data.len() < 8 {
        return Err(Error::Inconsistent("the sound chunk is too short"));
    }
    let offset = usize::try_from(u32_at(data, 0)?).map_err(|_| Error::TooLarge)?;
    let start = 8usize.checked_add(offset).ok_or(Error::TooLarge)?;
    if start > data.len() {
        return Err(Error::Inconsistent("the sound offset does not fit"));
    }
    Ok(&data[start..])
}

fn packet(header: &Header, sound: &[u8]) -> Result<Packet, Error> {
    if header.pcm {
        let width = u32::from(header.bits / 8);
        let frame = u32::from(header.channels)
            .checked_mul(width)
            .ok_or(Error::TooLarge)?;
        let expected = u64::from(header.frames)
            .checked_mul(u64::from(frame))
            .ok_or(Error::TooLarge)?;
        let actual = u64::try_from(sound.len()).map_err(|_| Error::TooLarge)?;
        if actual != expected {
            return Err(Error::Inconsistent(
                "the sound data does not match the frame count",
            ));
        }
    }
    Ok(Packet {
        data: sound.to_vec(),
        pts: Timestamp::new(0, header.rate),
        dts: None,
        duration: Some(Timestamp::new(i64::from(header.frames), header.rate)),
        keyframe: true,
    })
}

/// An 80-bit SANE float as a positive whole sample rate.
fn whole_rate(bytes: [u8; 10]) -> Result<u32, Error> {
    let exponent = u16::from_be_bytes([bytes[0], bytes[1]]);
    let sign = exponent & 0x8000 != 0;
    let exponent = exponent & 0x7FFF;
    let mantissa = u64::from_be_bytes(bytes_at::<8>(&bytes, 2)?);
    if !sign && mantissa == 0 && exponent != 0x7FFF {
        return Err(Error::Inconsistent("the sample rate is zero"));
    }
    if sign || exponent == 0 || exponent == 0x7FFF {
        return Err(Error::Inconsistent("the sample rate is not a whole number"));
    }
    let shift = i32::from(exponent) - 16383 - 63;
    let value = if shift >= 0 {
        let shift = u32::try_from(shift).map_err(|_| Error::TooLarge)?;
        if shift >= 64 || mantissa.leading_zeros() < shift {
            return Err(Error::TooLarge);
        }
        mantissa << shift
    } else {
        let shift = u32::try_from(-shift).map_err(|_| Error::TooLarge)?;
        if shift >= 64 || mantissa & ((1u64 << shift) - 1) != 0 {
            return Err(Error::Inconsistent("the sample rate is not a whole number"));
        }
        mantissa >> shift
    };
    if value == 0 {
        return Err(Error::Inconsistent("the sample rate is zero"));
    }
    u32::try_from(value).map_err(|_| Error::TooLarge)
}

fn codec_id(fourcc: [u8; 4]) -> String {
    if let Ok(text) = std::str::from_utf8(&fourcc) {
        if text.bytes().all(|byte| (0x20..=0x7E).contains(&byte)) {
            return text.to_string();
        }
    }
    fourcc.iter().fold(String::new(), |mut text, byte| {
        text.push_str(&format!("{byte:02x}"));
        text
    })
}

fn u32_at(data: &[u8], offset: usize) -> Result<u32, Error> {
    Ok(u32::from_be_bytes(bytes_at(data, offset)?))
}

fn i16_at(data: &[u8], offset: usize) -> Result<i16, Error> {
    Ok(i16::from_be_bytes(bytes_at(data, offset)?))
}

fn bytes_at<const N: usize>(data: &[u8], offset: usize) -> Result<[u8; N], Error> {
    let end = offset.checked_add(N).ok_or(Error::TooLarge)?;
    let slice = data
        .get(offset..end)
        .ok_or(Error::Inconsistent("the common chunk is too short"))?;
    match slice.try_into() {
        Ok(bytes) => Ok(bytes),
        Err(_) => Err(Error::Inconsistent("the common chunk is too short")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_extended_rate_is_a_whole_number() {
        assert_eq!(
            whole_rate([0x40, 0x0E, 0xAC, 0x44, 0, 0, 0, 0, 0, 0]),
            Ok(44_100)
        );
        assert_eq!(
            whole_rate([0x40, 0x0D, 0xAC, 0x44, 0, 0, 0, 0, 0, 0]),
            Ok(22_050)
        );
    }

    #[test]
    fn a_zero_extended_rate_is_refused() {
        assert_eq!(
            whole_rate([0; 10]),
            Err(Error::Inconsistent("the sample rate is zero"))
        );
    }

    #[test]
    fn a_fractional_extended_rate_is_refused() {
        assert_eq!(
            whole_rate([0x40, 0x0E, 0xAC, 0x44, 0, 0, 0, 0, 0, 1]),
            Err(Error::Inconsistent("the sample rate is not a whole number"))
        );
    }

    #[test]
    fn a_negative_extended_rate_is_refused() {
        assert_eq!(
            whole_rate([0xC0, 0x0E, 0xAC, 0x44, 0, 0, 0, 0, 0, 0]),
            Err(Error::Inconsistent("the sample rate is not a whole number"))
        );
    }

    #[test]
    fn an_extended_rate_that_does_not_fit_is_too_large() {
        assert_eq!(
            whole_rate([0x40, 0x3E, 0x80, 0, 0, 0, 0, 0, 0, 0]),
            Err(Error::TooLarge)
        );
    }
}

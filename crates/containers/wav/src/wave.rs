//! A WAVE form.

use std::num::NonZeroU32;

use cutlass_core::{AudioCodec, AudioStreamConfig, Packet, Timestamp};

use crate::Error;
use crate::riff::{self, Chunk, Cursor};

const RIFF: [u8; 4] = *b"RIFF";
const WAVE: [u8; 4] = *b"WAVE";
const FMT: [u8; 4] = *b"fmt ";
const DATA: [u8; 4] = *b"data";

const PCM_HEADER: usize = 16;
const EXTENDED_HEADER: usize = 18;

/// A parsed WAV file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Movie {
    tracks: Vec<Track>,
}

impl Movie {
    /// Read a file that is already in memory.
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        let mut cursor = Cursor::new(data);
        let Some(riff) = cursor.next_chunk()? else {
            return Err(Error::Inconsistent("the file has no RIFF header"));
        };
        if riff.id != RIFF {
            return Err(Error::Inconsistent("the file has no RIFF header"));
        }
        if riff.data.len() < 4 {
            return Err(Error::Inconsistent("the file has no form"));
        }
        let form: [u8; 4] = match riff.data[..4].try_into() {
            Ok(form) => form,
            Err(_) => return Err(Error::Inconsistent("the file has no form")),
        };
        if form != WAVE {
            return Err(Error::Form);
        }
        let movie = parse_wave(data, &riff)?;
        if cursor.next_chunk()?.is_some() {
            return Err(Error::Inconsistent("the file has data after the wave"));
        }
        Ok(movie)
    }

    /// The one audio track.
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }
}

/// The audio in the file. Samples are the data chunk, in stored order.
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
    /// A format tag this reader does not map. `codec_id` is the tag as four
    /// hex digits, and `config_record` is the `fmt ` chunk through the
    /// declared extra.
    Other {
        codec_id: String,
        config_record: Vec<u8>,
    },
}

fn parse_wave(file: &[u8], riff: &Chunk<'_>) -> Result<Movie, Error> {
    let mut cursor = riff::children(file, riff)?;
    let mut format = None;
    let mut data = None;
    while let Some(chunk) = cursor.next_chunk()? {
        match chunk.id {
            FMT => {
                if format.is_some() {
                    return Err(Error::Inconsistent("the format chunk is repeated"));
                }
                format = Some(chunk.data);
            }
            DATA => {
                if data.is_some() {
                    return Err(Error::Inconsistent("the data chunk is repeated"));
                }
                data = Some(chunk.data);
            }
            _ => {}
        }
    }
    let format = format.ok_or(Error::Inconsistent("the file has no format"))?;
    let header = read_format(format)?;
    let samples = match data {
        Some(data) => vec![packet(&header, data)?],
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
    block_align: u16,
}

fn read_format(data: &[u8]) -> Result<Header, Error> {
    let config_end = config_len(data)?;
    let tag = u16_at(data, 0)?;
    let channels = u16_at(data, 2)?;
    let rate = u32_at(data, 4)?;
    let block_align = u16_at(data, 12)?;
    let bits = u16_at(data, 14)?;
    let rate = match NonZeroU32::new(rate) {
        Some(rate) => rate,
        None => return Err(Error::Inconsistent("the sample rate is zero")),
    };
    let config_record = data[..config_end].to_vec();
    let kind = match tag {
        0x0001 | 0x0003 => {
            check_pcm(channels, bits, block_align)?;
            Kind::Audio(AudioStreamConfig {
                codec: AudioCodec::Pcm,
                sample_rate: rate.get(),
                channels,
                config_record,
            })
        }
        _ => Kind::Other {
            codec_id: format!("{tag:04x}"),
            config_record,
        },
    };
    Ok(Header {
        rate,
        kind,
        block_align,
    })
}

fn config_len(data: &[u8]) -> Result<usize, Error> {
    if data.len() < PCM_HEADER {
        return Err(Error::Inconsistent("the wave header is too short"));
    }
    if data.len() == PCM_HEADER {
        return Ok(PCM_HEADER);
    }
    if data.len() < EXTENDED_HEADER {
        return Err(Error::Inconsistent("the wave header is too short"));
    }
    let extra = usize::from(u16_at(data, 16)?);
    let end = EXTENDED_HEADER.checked_add(extra).ok_or(Error::TooLarge)?;
    if end > data.len() {
        return Err(Error::Inconsistent("the wave extra does not fit"));
    }
    Ok(end)
}

fn check_pcm(channels: u16, bits: u16, block_align: u16) -> Result<(), Error> {
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
    let width = bits / 8;
    let frame = channels.checked_mul(width).ok_or(Error::TooLarge)?;
    if block_align != frame {
        return Err(Error::Inconsistent(
            "the block align does not match the sample",
        ));
    }
    Ok(())
}

fn packet(header: &Header, data: &[u8]) -> Result<Packet, Error> {
    if header.block_align == 0 {
        return Err(Error::Inconsistent("the block align is zero"));
    }
    let align = usize::from(header.block_align);
    if !data.len().is_multiple_of(align) {
        return Err(Error::Inconsistent(
            "the data size is not a multiple of the block align",
        ));
    }
    let frames = i64::try_from(data.len() / align).map_err(|_| Error::TooLarge)?;
    Ok(Packet {
        data: data.to_vec(),
        pts: Timestamp::new(0, header.rate),
        dts: None,
        duration: Some(Timestamp::new(frames, header.rate)),
        keyframe: true,
    })
}

fn u32_at(data: &[u8], offset: usize) -> Result<u32, Error> {
    Ok(u32::from_le_bytes(bytes_at(data, offset)?))
}

fn u16_at(data: &[u8], offset: usize) -> Result<u16, Error> {
    Ok(u16::from_le_bytes(bytes_at(data, offset)?))
}

fn bytes_at<const N: usize>(data: &[u8], offset: usize) -> Result<[u8; N], Error> {
    let end = offset.checked_add(N).ok_or(Error::TooLarge)?;
    let slice = data
        .get(offset..end)
        .ok_or(Error::Inconsistent("the wave header is too short"))?;
    match slice.try_into() {
        Ok(bytes) => Ok(bytes),
        Err(_) => Err(Error::Inconsistent("the wave header is too short")),
    }
}

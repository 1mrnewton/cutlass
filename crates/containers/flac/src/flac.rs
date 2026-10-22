//! A native FLAC file. RFC 9639. The audio after the metadata is one packet.

use std::num::NonZeroU32;

use cutlass_core::{AudioCodec, AudioStreamConfig, Packet, Timestamp};

use crate::Error;

const MARKER: &[u8] = b"fLaC";
const STREAMINFO: usize = 34;
const BLOCK_HEADER: usize = 4;

/// A parsed FLAC file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Movie {
    tracks: Vec<Track>,
}

impl Movie {
    /// Read a file that is already in memory.
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        if data.is_empty() {
            return Err(Error::Inconsistent("the file has no stream info"));
        }
        if data.len() < MARKER.len() || &data[..MARKER.len()] != MARKER {
            return Err(if data.len() < MARKER.len() {
                Error::Truncated { at: 0, need: 4 }
            } else {
                Error::Form
            });
        }
        let (info, audio_at) = metadata(&data[MARKER.len()..], MARKER.len())?;
        let audio = &data[audio_at..];
        let samples = packet(&info, audio)?;
        Ok(Movie {
            tracks: vec![Track {
                timescale: info.rate,
                config: AudioStreamConfig {
                    codec: AudioCodec::Flac,
                    sample_rate: info.rate.get(),
                    channels: info.channels,
                    config_record: info.record,
                },
                samples,
            }],
        })
    }

    /// The one audio track.
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }
}

/// The audio in the file. The sample list is the audio after the metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Track {
    timescale: NonZeroU32,
    config: AudioStreamConfig,
    samples: Vec<Packet>,
}

impl Track {
    /// Ticks per second. This is the sample rate.
    pub fn timescale(&self) -> NonZeroU32 {
        self.timescale
    }

    /// Codec, rate, and channel count. The config record is the stream info.
    pub fn config(&self) -> &AudioStreamConfig {
        &self.config
    }

    /// The audio packet, when the file has one. `dts` is always none.
    pub fn samples(&self) -> &[Packet] {
        &self.samples
    }
}

struct Info {
    rate: NonZeroU32,
    channels: u16,
    samples: u64,
    record: Vec<u8>,
}

fn metadata(data: &[u8], base: usize) -> Result<(Info, usize), Error> {
    let mut at = 0;
    let mut info = None;
    let mut first = true;
    loop {
        if at == data.len() {
            return Err(Error::Inconsistent("the metadata does not end"));
        }
        if data.len() - at < BLOCK_HEADER {
            return Err(Error::Truncated {
                at: locate(base, at)?,
                need: 4,
            });
        }
        let last = data[at] & 0x80 != 0;
        let kind = data[at] & 0x7F;
        let len = block_len(&data[at + 1..at + 4])?;
        let body_at = at + BLOCK_HEADER;
        let body_end = body_at.checked_add(len).ok_or(Error::TooLarge)?;
        if body_end > data.len() {
            return Err(Error::Truncated {
                at: locate(base, body_at)?,
                need: u64::try_from(len).map_err(|_| Error::TooLarge)?,
            });
        }
        if kind == 127 {
            return Err(Error::Inconsistent("the metadata block is invalid"));
        }
        if first && kind != 0 {
            return Err(Error::Inconsistent("the file has no stream info"));
        }
        first = false;
        if kind == 0 {
            if info.is_some() {
                return Err(Error::Inconsistent("the stream info is repeated"));
            }
            info = Some(stream_info(&data[body_at..body_end])?);
        }
        at = body_end;
        if last {
            let info = info.ok_or(Error::Inconsistent("the file has no stream info"))?;
            let audio_at = base.checked_add(at).ok_or(Error::TooLarge)?;
            return Ok((info, audio_at));
        }
    }
}

fn stream_info(data: &[u8]) -> Result<Info, Error> {
    if data.len() != STREAMINFO {
        return Err(Error::Inconsistent("the stream info is the wrong size"));
    }
    let packed = u64::from_be_bytes(bytes_at(data, 10)?);
    let rate = u32::try_from((packed >> 44) & 0x000F_FFFF).map_err(|_| Error::TooLarge)?;
    let rate = match NonZeroU32::new(rate) {
        Some(rate) => rate,
        None => return Err(Error::Inconsistent("the sample rate is zero")),
    };
    let channels = u16::try_from(((packed >> 41) & 0x7) + 1).map_err(|_| Error::TooLarge)?;
    Ok(Info {
        rate,
        channels,
        samples: packed & 0x000F_FFFF_FFFF,
        record: data.to_vec(),
    })
}

fn packet(info: &Info, audio: &[u8]) -> Result<Vec<Packet>, Error> {
    if audio.is_empty() {
        if info.samples > 0 {
            return Err(Error::Inconsistent("the stream has no audio"));
        }
        return Ok(Vec::new());
    }
    let duration = if info.samples == 0 {
        None
    } else {
        Some(Timestamp::new(
            i64::try_from(info.samples).map_err(|_| Error::TooLarge)?,
            info.rate,
        ))
    };
    Ok(vec![Packet {
        data: audio.to_vec(),
        pts: Timestamp::new(0, info.rate),
        dts: None,
        duration,
        keyframe: true,
    }])
}

fn block_len(bytes: &[u8]) -> Result<usize, Error> {
    let len = u32::from_be_bytes([0, bytes[0], bytes[1], bytes[2]]);
    usize::try_from(len).map_err(|_| Error::TooLarge)
}

fn locate(base: usize, at: usize) -> Result<u64, Error> {
    let pos = base.checked_add(at).ok_or(Error::TooLarge)?;
    u64::try_from(pos).map_err(|_| Error::TooLarge)
}

fn bytes_at(data: &[u8], offset: usize) -> Result<[u8; 8], Error> {
    let end = offset.checked_add(8).ok_or(Error::TooLarge)?;
    let slice = data
        .get(offset..end)
        .ok_or(Error::Inconsistent("the stream info is the wrong size"))?;
    match slice.try_into() {
        Ok(bytes) => Ok(bytes),
        Err(_) => Err(Error::Inconsistent("the stream info is the wrong size")),
    }
}

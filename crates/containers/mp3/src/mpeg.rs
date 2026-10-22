//! An MP3 file: an optional ID3v2 tag, frames, and an optional ID3v1 tag.

use std::num::NonZeroU32;

use cutlass_core::{AudioCodec, AudioStreamConfig, Packet, Timestamp};

use crate::Error;
use crate::frame::{self, Frame};

/// A parsed MP3 file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Movie {
    tracks: Vec<Track>,
}

impl Movie {
    /// Read a file that is already in memory.
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        if data.is_empty() {
            return Err(Error::Inconsistent("the file has no frames"));
        }
        let start = skip_id3v2(data)?;
        let end = id3v1_end(data);
        if start > end {
            return Err(Error::Inconsistent("the file has no frames"));
        }
        if end - start < 4 {
            return Err(if start == 0 {
                Error::Truncated {
                    at: u64::try_from(start).map_err(|_| Error::TooLarge)?,
                    need: 4,
                }
            } else {
                Error::Inconsistent("the file has no frames")
            });
        }
        if data[start] != 0xFF || data[start + 1] & 0xE0 != 0xE0 {
            return Err(Error::Form);
        }
        let audio = &data[start..end];
        let (stream, samples) = frames(audio, u64::try_from(start).map_err(|_| Error::TooLarge)?)?;
        let rate = match NonZeroU32::new(stream.sample_rate) {
            Some(rate) => rate,
            None => return Err(Error::Inconsistent("the sample rate is zero")),
        };
        Ok(Movie {
            tracks: vec![Track {
                timescale: rate,
                config: AudioStreamConfig {
                    codec: AudioCodec::Mp3,
                    sample_rate: stream.sample_rate,
                    channels: stream.channels,
                    config_record: Vec::new(),
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

/// The audio in the file. Samples are the frames, in stored order.
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

    /// Codec, rate, and channel count. The config record is empty.
    pub fn config(&self) -> &AudioStreamConfig {
        &self.config
    }

    /// Frames in stored order. `dts` is always none.
    pub fn samples(&self) -> &[Packet] {
        &self.samples
    }
}

fn frames(audio: &[u8], base: u64) -> Result<(Frame, Vec<Packet>), Error> {
    let rate = match NonZeroU32::new(frame::header(audio, 0)?.sample_rate) {
        Some(rate) => rate,
        None => return Err(Error::Inconsistent("the sample rate is zero")),
    };
    let mut samples = Vec::new();
    let mut at = 0;
    let mut pts = 0i64;
    let mut stream = None;
    while at < audio.len() {
        if audio.len() - at < 4 {
            return Err(Error::Inconsistent(
                "the file has data after the last frame",
            ));
        }
        let header = frame::header(audio, at)?;
        let stream = stream.get_or_insert(header);
        if !frame::same_stream(stream, &header) {
            return Err(Error::Inconsistent("a frame does not match the stream"));
        }
        let end = at.checked_add(header.len).ok_or(Error::TooLarge)?;
        if end > audio.len() {
            return Err(Error::Truncated {
                at: base
                    .checked_add(u64::try_from(at).map_err(|_| Error::TooLarge)?)
                    .ok_or(Error::TooLarge)?,
                need: u64::try_from(header.len).map_err(|_| Error::TooLarge)?,
            });
        }
        samples.push(Packet {
            data: audio[at..end].to_vec(),
            pts: Timestamp::new(pts, rate),
            dts: None,
            duration: Some(Timestamp::new(header.samples, rate)),
            keyframe: true,
        });
        pts = pts.checked_add(header.samples).ok_or(Error::TooLarge)?;
        at = end;
    }
    let stream = stream.ok_or(Error::Inconsistent("the file has no frames"))?;
    Ok((stream, samples))
}

fn skip_id3v2(data: &[u8]) -> Result<usize, Error> {
    if data.len() < 3 || &data[..3] != b"ID3" {
        return Ok(0);
    }
    if data.len() < 10 {
        return Err(Error::Truncated { at: 0, need: 10 });
    }
    let size = synchsafe(&data[6..10])?;
    let mut total = 10usize.checked_add(size).ok_or(Error::TooLarge)?;
    if data[5] & 0x10 != 0 {
        total = total.checked_add(10).ok_or(Error::TooLarge)?;
    }
    if total > data.len() {
        return Err(Error::Truncated {
            at: 0,
            need: u64::try_from(total).map_err(|_| Error::TooLarge)?,
        });
    }
    Ok(total)
}

fn synchsafe(bytes: &[u8]) -> Result<usize, Error> {
    if bytes.iter().any(|byte| byte & 0x80 != 0) {
        return Err(Error::Inconsistent("the tag size is not synchsafe"));
    }
    let size = (u32::from(bytes[0]) << 21)
        | (u32::from(bytes[1]) << 14)
        | (u32::from(bytes[2]) << 7)
        | u32::from(bytes[3]);
    usize::try_from(size).map_err(|_| Error::TooLarge)
}

fn id3v1_end(data: &[u8]) -> usize {
    if data.len() >= 128 && &data[data.len() - 128..data.len() - 125] == b"TAG" {
        data.len() - 128
    } else {
        data.len()
    }
}

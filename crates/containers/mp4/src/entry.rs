//! Sample descriptions for video and audio.

use cutlass_core::{AudioCodec, ColorTags, VideoCodec, VideoStreamConfig};

use crate::read::{self, Atom, Reader};
use crate::{Error, Fourcc};

/// An audio track, as the sample description states it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Audio {
    codec: Option<AudioCodec>,
    sample_entry: Fourcc,
    sample_rate: u32,
    channels: u16,
    config_record: Vec<u8>,
}

impl Audio {
    /// The codec named by the sample entry. `None` when this reader does not
    /// name that entry.
    pub fn codec(&self) -> Option<AudioCodec> {
        self.codec
    }

    /// The sample-entry code, such as `mp4a`.
    pub fn sample_entry(&self) -> Fourcc {
        self.sample_entry
    }

    /// Samples per second.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Channel count.
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Decoder configuration from the elementary stream descriptor, when the
    /// sample entry has one.
    pub fn config_record(&self) -> &[u8] {
        &self.config_record
    }
}

pub(crate) struct VideoEntry {
    pub(crate) config: VideoStreamConfig,
    pub(crate) color: Option<ColorTags>,
}

pub(crate) enum Description {
    Video(VideoEntry),
    Audio(Audio),
}

pub(crate) fn description(stsd: &Atom<'_>, handler: [u8; 4]) -> Result<Description, Error> {
    let (version, _, body) = read::full_box(stsd)?;
    expect_version(*b"stsd", version, &[0])?;
    let mut reader = Reader::new(body, stsd.payload_base()?);
    let count = reader.u32()?;
    let entries = read::atoms(reader.rest(), reader.file_pos()?)?;
    if count != 1 || entries.len() != 1 {
        return Err(Error::MultipleDescriptions);
    }
    let entry = &entries[0];
    if handler == *b"vide" {
        Ok(Description::Video(video(entry)?))
    } else {
        Ok(Description::Audio(audio(entry)?))
    }
}

fn video(entry: &Atom<'_>) -> Result<VideoEntry, Error> {
    let codec = video_codec(entry.kind).ok_or(Error::UnsupportedVideo(Fourcc::new(entry.kind)))?;
    if entry.payload.len() < VISUAL_PREFIX {
        return Err(Error::Truncated {
            at: entry.payload_base()?,
            need: u64::try_from(VISUAL_PREFIX).map_err(|_| Error::TooLarge)?,
        });
    }
    let width = u32::from(u16::from_be_bytes([entry.payload[24], entry.payload[25]]));
    let height = u32::from(u16::from_be_bytes([entry.payload[26], entry.payload[27]]));
    if width == 0 || height == 0 {
        return Err(Error::ZeroDimension);
    }
    let children = read::atoms(
        &entry.payload[VISUAL_PREFIX..],
        entry.payload_base()? + u64::try_from(VISUAL_PREFIX).map_err(|_| Error::TooLarge)?,
    )?;
    let config_record = video_config(entry.kind, &children)?;
    Ok(VideoEntry {
        config: VideoStreamConfig {
            codec,
            width,
            height,
            config_record,
        },
        color: color_tags(&children)?,
    })
}

fn color_tags(children: &[Atom<'_>]) -> Result<Option<ColorTags>, Error> {
    let Some(colr) = read::only(children, *b"colr")? else {
        return Ok(None);
    };
    let mut reader = Reader::new(colr.payload, colr.payload_base()?);
    let kind = reader.fourcc()?;
    match &kind {
        b"nclx" => Ok(Some(ColorTags {
            primaries: reader.u16()?,
            transfer: reader.u16()?,
            matrix: reader.u16()?,
            full_range: Some(reader.u8()? & 0x80 != 0),
        })),
        b"nclc" => Ok(Some(ColorTags {
            primaries: reader.u16()?,
            transfer: reader.u16()?,
            matrix: reader.u16()?,
            full_range: None,
        })),
        // An ICC profile names the colour another way. The numeric tags stay empty.
        b"rICC" | b"prof" => Ok(None),
        _ => Ok(None),
    }
}

fn video_codec(kind: [u8; 4]) -> Option<VideoCodec> {
    match &kind {
        b"avc1" | b"avc3" => Some(VideoCodec::H264),
        b"hvc1" | b"hev1" => Some(VideoCodec::Hevc),
        b"vp08" => Some(VideoCodec::Vp8),
        b"vp09" => Some(VideoCodec::Vp9),
        b"av01" => Some(VideoCodec::Av1),
        b"mp4v" => Some(VideoCodec::Mpeg4Part2),
        b"jpeg" | b"mjpa" | b"mjpb" | b"mjpg" => Some(VideoCodec::Mjpeg),
        b"apch" | b"apcn" | b"apcs" | b"apco" | b"ap4h" | b"ap4x" => Some(VideoCodec::ProRes),
        _ => None,
    }
}

fn video_config(kind: [u8; 4], children: &[Atom<'_>]) -> Result<Vec<u8>, Error> {
    let required = match &kind {
        b"avc1" | b"avc3" => Some(*b"avcC"),
        b"hvc1" | b"hev1" => Some(*b"hvcC"),
        b"vp09" => Some(*b"vpcC"),
        b"av01" => Some(*b"av1C"),
        _ => None,
    };
    let Some(name) = required else {
        return optional_config(kind, children);
    };
    let atom = read::required(children, name)?;
    Ok(atom.payload.to_vec())
}

fn optional_config(kind: [u8; 4], children: &[Atom<'_>]) -> Result<Vec<u8>, Error> {
    match &kind {
        b"vp08" => Ok(read::only(children, *b"vpcC")?
            .map(|atom| atom.payload.to_vec())
            .unwrap_or_default()),
        b"mp4v" => {
            let Some(esds) = read::only(children, *b"esds")? else {
                return Ok(Vec::new());
            };
            let (version, _, body) = read::full_box(esds)?;
            expect_version(*b"esds", version, &[0])?;
            let (_, config) = decoder_specific(body, esds.payload_base()?)?;
            Ok(config)
        }
        _ => Ok(Vec::new()),
    }
}

fn audio(entry: &Atom<'_>) -> Result<Audio, Error> {
    let header = audio_header(entry)?;
    let Some(rest) = entry.payload.get(header.prefix..) else {
        return Err(Error::Truncated {
            at: header.child_base,
            need: 1,
        });
    };
    let children = read::atoms(rest, header.child_base)?;
    let (object_type, config_record) = match find_esds(&children)? {
        Some(payload) => {
            if payload.len() < 4 {
                return Err(Error::Truncated {
                    at: header.child_base,
                    need: 4,
                });
            }
            if payload[0] != 0 {
                return Err(Error::BadVersion {
                    kind: Fourcc::new(*b"esds"),
                    version: payload[0],
                });
            }
            let base = header.child_base.checked_add(4).ok_or(Error::TooLarge)?;
            let (object_type, config) = decoder_specific(&payload[4..], base)?;
            (Some(object_type), config)
        }
        None => (None, Vec::new()),
    };
    Ok(Audio {
        codec: audio_codec(entry.kind, object_type),
        sample_entry: Fourcc::new(entry.kind),
        sample_rate: header.sample_rate,
        channels: header.channels,
        config_record,
    })
}

struct AudioHeader {
    prefix: usize,
    child_base: u64,
    sample_rate: u32,
    channels: u16,
}

fn audio_header(entry: &Atom<'_>) -> Result<AudioHeader, Error> {
    let mut reader = Reader::new(entry.payload, entry.payload_base()?);
    reader.skip(8)?;
    let version = reader.u16()?;
    reader.skip(2)?;
    reader.skip(4)?;
    let (channels, sample_rate, prefix) = match version {
        0 | 1 => {
            let channels = reader.u16()?;
            reader.skip(6)?;
            let sample_rate = rate_16_16(reader.u32()?)?;
            if version == 1 {
                reader.skip(16)?;
            }
            let prefix = if version == 0 { 28 } else { 44 };
            (channels, sample_rate, prefix)
        }
        2 => {
            reader.skip(16)?;
            let bits = reader.u64()?;
            let sample_rate = rate_f64(f64::from_bits(bits))?;
            let channels = u16::try_from(reader.u32()?).map_err(|_| Error::TooLarge)?;
            reader.skip(20)?;
            (channels, sample_rate, 64)
        }
        _ => {
            return Err(Error::Inconsistent(
                "audio sample entry version is not 0, 1, or 2",
            ));
        }
    };
    if channels == 0 {
        return Err(Error::ZeroChannels);
    }
    if reader.file_pos()?
        != entry.payload_base()? + u64::try_from(prefix).map_err(|_| Error::TooLarge)?
    {
        return Err(Error::Inconsistent(
            "audio sample entry header is the wrong length",
        ));
    }
    Ok(AudioHeader {
        prefix,
        child_base: reader.file_pos()?,
        sample_rate,
        channels,
    })
}

fn rate_16_16(raw: u32) -> Result<u32, Error> {
    let rate = (u64::from(raw).saturating_add(0x8000)) >> 16;
    let rate = u32::try_from(rate).map_err(|_| Error::TooLarge)?;
    if rate == 0 {
        Err(Error::BadSampleRate)
    } else {
        Ok(rate)
    }
}

fn rate_f64(rate: f64) -> Result<u32, Error> {
    if !rate.is_finite() || rate <= 0.0 || rate > f64::from(u32::MAX) {
        return Err(Error::BadSampleRate);
    }
    let rounded = rate.round();
    if rounded <= 0.0 || rounded > f64::from(u32::MAX) {
        return Err(Error::BadSampleRate);
    }
    // `rounded` is inside `1..=u32::MAX`, so the conversion keeps the value.
    let rate = rounded as u32;
    if rate == 0 {
        Err(Error::BadSampleRate)
    } else {
        Ok(rate)
    }
}

fn audio_codec(kind: [u8; 4], object_type: Option<u8>) -> Option<AudioCodec> {
    Some(match &kind {
        b"mp4a" => match object_type {
            Some(0x69 | 0x6b) => AudioCodec::Mp3,
            _ => AudioCodec::Aac,
        },
        b"alac" => AudioCodec::Alac,
        b"ac-3" | b"sac3" => AudioCodec::Ac3,
        b"sowt" | b"twos" | b"lpcm" | b"raw " | b"in24" | b"in32" | b"fl32" | b"fl64" => {
            AudioCodec::Pcm
        }
        b"fLaC" | b"flac" => AudioCodec::Flac,
        b"Opus" => AudioCodec::Opus,
        _ => return None,
    })
}

fn find_esds(atoms: &[Atom<'_>]) -> Result<Option<Vec<u8>>, Error> {
    if let Some(esds) = read::only(atoms, *b"esds")? {
        return Ok(Some(esds.payload.to_vec()));
    }
    let Some(wave) = read::only(atoms, *b"wave")? else {
        return Ok(None);
    };
    let nested = read::atoms(wave.payload, wave.payload_base()?)?;
    Ok(read::only(&nested, *b"esds")?.map(|atom| atom.payload.to_vec()))
}

fn decoder_specific(body: &[u8], base: u64) -> Result<(u8, Vec<u8>), Error> {
    let mut reader = Reader::new(body, base);
    let (tag, content) = descriptor(&mut reader)?;
    if tag != 0x03 {
        return Err(Error::BadDescriptor);
    }
    let mut descriptor_reader = Reader::new(&content, base);
    descriptor_reader.u16()?;
    let flags = descriptor_reader.u8()?;
    if flags & 0x80 != 0 {
        descriptor_reader.skip(2)?;
    }
    if flags & 0x40 != 0 {
        let len = usize::from(descriptor_reader.u8()?);
        descriptor_reader.skip(len)?;
    }
    if flags & 0x20 != 0 {
        descriptor_reader.skip(2)?;
    }
    let (tag, content) = descriptor(&mut descriptor_reader)?;
    if tag != 0x04 {
        return Err(Error::BadDescriptor);
    }
    let mut decoder = Reader::new(&content, base);
    let object_type = decoder.u8()?;
    decoder.skip(12)?;
    if decoder.remaining() == 0 {
        return Ok((object_type, Vec::new()));
    }
    let (tag, specific) = descriptor(&mut decoder)?;
    if tag != 0x05 {
        return Err(Error::BadDescriptor);
    }
    Ok((object_type, specific))
}

fn descriptor(reader: &mut Reader<'_>) -> Result<(u8, Vec<u8>), Error> {
    let tag = reader.u8()?;
    let mut size = 0usize;
    for _ in 0..4 {
        let byte = reader.u8()?;
        size = size
            .checked_shl(7)
            .ok_or(Error::TooLarge)?
            .checked_add(usize::from(byte & 0x7f))
            .ok_or(Error::TooLarge)?;
        if byte & 0x80 == 0 {
            return Ok((tag, reader.take(size)?.to_vec()));
        }
    }
    Err(Error::BadDescriptor)
}

fn expect_version(kind: [u8; 4], version: u8, allowed: &[u8]) -> Result<(), Error> {
    if allowed.contains(&version) {
        Ok(())
    } else {
        Err(Error::BadVersion {
            kind: Fourcc::new(kind),
            version,
        })
    }
}

const VISUAL_PREFIX: usize = 78;

/// Version 2 of the QuickTime sound description ends 64 bytes into the sample
/// entry payload: 8 bytes of sample-entry header, then 56 bytes of version 2.
const _: () = assert!(VISUAL_PREFIX == 78);

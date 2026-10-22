//! AVI headers and streams.

use std::num::NonZeroU32;

use cutlass_core::{AudioCodec, AudioStreamConfig, VideoCodec, VideoStreamConfig};

use crate::Error;
use crate::riff::{self, Chunk, Cursor};

const RIFF: [u8; 4] = *b"RIFF";
const AVI_FORM: [u8; 4] = *b"AVI ";
const LIST: [u8; 4] = *b"LIST";
const HDRL: [u8; 4] = *b"hdrl";
const STRL: [u8; 4] = *b"strl";
const AVIH: [u8; 4] = *b"avih";
const STRH: [u8; 4] = *b"strh";
const STRF: [u8; 4] = *b"strf";
const VIDS: [u8; 4] = *b"vids";
const AUDS: [u8; 4] = *b"auds";

const MAIN_HEADER: usize = 56;
const STREAM_HEADER: usize = 48;
const BITMAP_HEADER: usize = 40;
const WAVE_OLD: usize = 16;
const MAX_STREAMS: u32 = 100;

/// A parsed AVI file.
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
        if form != AVI_FORM {
            return Err(Error::Form);
        }
        let movie = parse_form(data, &riff)?;
        if cursor.next_chunk()?.is_some() {
            return Err(Error::Inconsistent("the file has data after the movie"));
        }
        Ok(movie)
    }

    /// Streams in header order. The number is the two-digit id a chunk uses.
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }
}

/// One stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Track {
    number: u32,
    timescale: NonZeroU32,
    kind: Kind,
}

impl Track {
    /// Zero-based stream number.
    pub fn number(&self) -> u32 {
        self.number
    }

    /// Ticks per second, from `dwRate` and `dwScale`.
    pub fn timescale(&self) -> NonZeroU32 {
        self.timescale
    }

    /// Codec, and the settings a decoder needs.
    pub fn kind(&self) -> &Kind {
        &self.kind
    }
}

/// What the stream carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    Video(VideoStreamConfig),
    Audio(AudioStreamConfig),
    /// A codec this reader does not map. `codec_id` is the fourcc or the wave
    /// format tag, and `config_record` is the extra bytes after that header.
    Other {
        codec_id: String,
        config_record: Vec<u8>,
    },
}

fn parse_form(file: &[u8], riff: &Chunk<'_>) -> Result<Movie, Error> {
    let mut cursor = riff::children(file, riff)?;
    let mut header = None;
    while let Some(chunk) = cursor.next_chunk()? {
        if chunk.id != LIST {
            continue;
        }
        let (kind, _) = riff::list_parts(&chunk)?;
        if kind != HDRL {
            continue;
        }
        if header.is_some() {
            return Err(Error::Inconsistent("the header is repeated"));
        }
        header = Some(parse_header(file, &chunk)?);
    }
    let tracks = header.ok_or(Error::Inconsistent("the file has no header"))?;
    Ok(Movie { tracks })
}

fn parse_header(file: &[u8], list: &Chunk<'_>) -> Result<Vec<Track>, Error> {
    let mut cursor = riff::children(file, list)?;
    let mut stream_count = None;
    let mut tracks = Vec::new();
    while let Some(chunk) = cursor.next_chunk()? {
        if chunk.id == AVIH {
            if stream_count.is_some() {
                return Err(Error::Inconsistent("the main header is repeated"));
            }
            stream_count = Some(main_streams(&chunk)?);
            continue;
        }
        if chunk.id != LIST {
            continue;
        }
        let (kind, _) = riff::list_parts(&chunk)?;
        if kind == STRL {
            let number = u32::try_from(tracks.len()).map_err(|_| Error::TooLarge)?;
            tracks.push(parse_stream(file, &chunk, number)?);
        }
    }
    let stream_count = stream_count.ok_or(Error::Inconsistent("the file has no main header"))?;
    let found = u32::try_from(tracks.len()).map_err(|_| Error::TooLarge)?;
    if stream_count != found {
        return Err(Error::Inconsistent("the stream count does not match"));
    }
    if stream_count > MAX_STREAMS {
        return Err(Error::Inconsistent("too many streams"));
    }
    Ok(tracks)
}

fn main_streams(chunk: &Chunk<'_>) -> Result<u32, Error> {
    if chunk.data.len() < MAIN_HEADER {
        return Err(Error::Inconsistent("the main header is too short"));
    }
    u32_at(chunk.data, 24)
}

fn parse_stream(file: &[u8], list: &Chunk<'_>, number: u32) -> Result<Track, Error> {
    let mut cursor = riff::children(file, list)?;
    let mut header = None;
    let mut format = None;
    while let Some(chunk) = cursor.next_chunk()? {
        match &chunk.id {
            id if *id == STRH => {
                if header.is_some() {
                    return Err(Error::Inconsistent("the stream header is repeated"));
                }
                header = Some(chunk.data);
            }
            id if *id == STRF => {
                if format.is_some() {
                    return Err(Error::Inconsistent("the stream format is repeated"));
                }
                format = Some(chunk.data);
            }
            _ => {}
        }
    }
    let header = header.ok_or(Error::Inconsistent("a stream has no header"))?;
    if header.len() < STREAM_HEADER {
        return Err(Error::Inconsistent("the stream header is too short"));
    }
    let stream_type = fourcc_at(header, 0)?;
    let handler = fourcc_at(header, 4)?;
    let scale = u32_at(header, 20)?;
    let rate = u32_at(header, 24)?;
    let timescale = timescale_of(scale, rate)?;
    let kind = match &stream_type {
        kind if *kind == VIDS => video_kind(format)?,
        kind if *kind == AUDS => audio_kind(format)?,
        _ => other_kind(stream_type, handler, format),
    };
    Ok(Track {
        number,
        timescale,
        kind,
    })
}

fn video_kind(format: Option<&[u8]>) -> Result<Kind, Error> {
    let Some(format) = format else {
        return Err(Error::Inconsistent("a video stream has no format"));
    };
    if format.len() < BITMAP_HEADER {
        return Err(Error::Inconsistent("the bitmap header is too short"));
    }
    let declared = usize::try_from(u32_at(format, 0)?).map_err(|_| Error::TooLarge)?;
    if declared < BITMAP_HEADER || declared > format.len() {
        return Err(Error::Inconsistent("the bitmap header is too short"));
    }
    let width = i32_at(format, 4)?;
    let height = i32_at(format, 8)?;
    let width =
        u32::try_from(width).map_err(|_| Error::Inconsistent("the video size is invalid"))?;
    if width == 0 || height == 0 {
        return Err(Error::Inconsistent("the video size is invalid"));
    }
    let height = height.unsigned_abs();
    let compression = fourcc_at(format, 16)?;
    let config_record = format[declared..].to_vec();
    if compression == [0, 0, 0, 0] {
        return Ok(Kind::Other {
            codec_id: "0000".to_string(),
            config_record,
        });
    }
    match video_codec(compression) {
        Some(codec) => Ok(Kind::Video(VideoStreamConfig {
            codec,
            width,
            height,
            config_record,
        })),
        None => Ok(Kind::Other {
            codec_id: codec_id(compression),
            config_record,
        }),
    }
}

fn audio_kind(format: Option<&[u8]>) -> Result<Kind, Error> {
    let Some(format) = format else {
        return Err(Error::Inconsistent("an audio stream has no format"));
    };
    if format.len() < WAVE_OLD {
        return Err(Error::Inconsistent("the wave header is too short"));
    }
    let tag = u16_at(format, 0)?;
    let channels = u16_at(format, 2)?;
    let sample_rate = u32_at(format, 4)?;
    let config_record = wave_extra(format)?;
    match audio_codec(tag) {
        Some(codec) => {
            if sample_rate == 0 {
                return Err(Error::Inconsistent("the sample rate is zero"));
            }
            if channels == 0 {
                return Err(Error::Inconsistent("the channel count is zero"));
            }
            Ok(Kind::Audio(AudioStreamConfig {
                codec,
                sample_rate,
                channels,
                config_record,
            }))
        }
        None => Ok(Kind::Other {
            codec_id: format!("{tag:04x}"),
            config_record,
        }),
    }
}

fn wave_extra(format: &[u8]) -> Result<Vec<u8>, Error> {
    if format.len() == WAVE_OLD {
        return Ok(Vec::new());
    }
    if format.len() < WAVE_OLD + 2 {
        return Err(Error::Inconsistent("the wave header is too short"));
    }
    let extra = usize::from(u16_at(format, 16)?);
    let start = WAVE_OLD + 2;
    let end = start.checked_add(extra).ok_or(Error::TooLarge)?;
    if end > format.len() {
        return Err(Error::Inconsistent("the wave extra does not fit"));
    }
    Ok(format[start..end].to_vec())
}

fn other_kind(stream_type: [u8; 4], handler: [u8; 4], format: Option<&[u8]>) -> Kind {
    let codec_id = if handler == [0, 0, 0, 0] {
        codec_id(stream_type)
    } else {
        codec_id(handler)
    };
    Kind::Other {
        codec_id,
        config_record: format.unwrap_or_default().to_vec(),
    }
}

fn video_codec(fourcc: [u8; 4]) -> Option<VideoCodec> {
    const MAP: &[(&[u8; 4], VideoCodec)] = &[
        (b"H264", VideoCodec::H264),
        (b"X264", VideoCodec::H264),
        (b"AVC1", VideoCodec::H264),
        (b"HEVC", VideoCodec::Hevc),
        (b"H265", VideoCodec::Hevc),
        (b"HEV1", VideoCodec::Hevc),
        (b"MJPG", VideoCodec::Mjpeg),
        (b"JPEG", VideoCodec::Mjpeg),
        (b"XVID", VideoCodec::Mpeg4Part2),
        (b"DIVX", VideoCodec::Mpeg4Part2),
        (b"DX50", VideoCodec::Mpeg4Part2),
        (b"MP4V", VideoCodec::Mpeg4Part2),
        (b"FMP4", VideoCodec::Mpeg4Part2),
        (b"MPG2", VideoCodec::Mpeg2),
        (b"VP80", VideoCodec::Vp8),
        (b"VP90", VideoCodec::Vp9),
        (b"AV01", VideoCodec::Av1),
        (b"apch", VideoCodec::ProRes),
        (b"apcn", VideoCodec::ProRes),
        (b"apcs", VideoCodec::ProRes),
        (b"apco", VideoCodec::ProRes),
        (b"ap4h", VideoCodec::ProRes),
    ];
    MAP.iter()
        .find(|(name, _)| fourcc.eq_ignore_ascii_case(*name))
        .map(|(_, codec)| *codec)
}

fn audio_codec(tag: u16) -> Option<AudioCodec> {
    match tag {
        0x0001 | 0x0003 => Some(AudioCodec::Pcm),
        0x0050 | 0x0055 => Some(AudioCodec::Mp3),
        0x00FF => Some(AudioCodec::Aac),
        0x2000 => Some(AudioCodec::Ac3),
        0xF1AC => Some(AudioCodec::Flac),
        _ => None,
    }
}

fn timescale_of(scale: u32, rate: u32) -> Result<NonZeroU32, Error> {
    if scale == 0 || rate == 0 {
        return Err(Error::Inconsistent("the stream scale or rate is zero"));
    }
    let timescale = rate / gcd(scale, rate);
    match NonZeroU32::new(timescale) {
        Some(timescale) => Ok(timescale),
        None => Err(Error::Inconsistent("the stream scale or rate is zero")),
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a
}

fn codec_id(fourcc: [u8; 4]) -> String {
    if fourcc == [0, 0, 0, 0] {
        return "0000".to_string();
    }
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

fn fourcc_at(data: &[u8], offset: usize) -> Result<[u8; 4], Error> {
    bytes_at(data, offset)
}

fn u32_at(data: &[u8], offset: usize) -> Result<u32, Error> {
    Ok(u32::from_le_bytes(bytes_at(data, offset)?))
}

fn u16_at(data: &[u8], offset: usize) -> Result<u16, Error> {
    Ok(u16::from_le_bytes(bytes_at(data, offset)?))
}

fn i32_at(data: &[u8], offset: usize) -> Result<i32, Error> {
    Ok(i32::from_le_bytes(bytes_at(data, offset)?))
}

fn bytes_at<const N: usize>(data: &[u8], offset: usize) -> Result<[u8; N], Error> {
    let end = offset.checked_add(N).ok_or(Error::TooLarge)?;
    let slice = data
        .get(offset..end)
        .ok_or(Error::Inconsistent("the header is too short"))?;
    match slice.try_into() {
        Ok(bytes) => Ok(bytes),
        Err(_) => Err(Error::Inconsistent("the header is too short")),
    }
}

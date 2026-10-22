//! Segment info and tracks.

use std::num::NonZeroU32;

use cutlass_core::{
    AudioCodec, AudioStreamConfig, Packet, Timestamp, VideoCodec, VideoStreamConfig,
};

use crate::Error;
use crate::block::Block;
use crate::ebml::{self, Element, Limits};

const EBML: u64 = 0x1A45_DFA3;
const DOC_TYPE: u64 = 0x4282;
const EBML_READ_VERSION: u64 = 0x42F7;
const MAX_ID_LENGTH: u64 = 0x42F2;
const MAX_SIZE_LENGTH: u64 = 0x42F3;
const VOID: u64 = 0xEC;
const CRC32: u64 = 0xBF;

const SEGMENT: u64 = 0x1853_8067;
const INFO: u64 = 0x1549_A966;
const TIMESTAMP_SCALE: u64 = 0x002A_D7B1;
const DURATION: u64 = 0x4489;

const TRACKS: u64 = 0x1654_AE6B;
const TRACK_ENTRY: u64 = 0xAE;
const TRACK_NUMBER: u64 = 0xD7;
const TRACK_TYPE: u64 = 0x83;
const CODEC_ID: u64 = 0x86;
const CODEC_PRIVATE: u64 = 0x63A2;
const CONTENT_ENCODINGS: u64 = 0x6D80;
const DEFAULT_DURATION: u64 = 0x0023_E383;

const VIDEO: u64 = 0xE0;
const PIXEL_WIDTH: u64 = 0xB0;
const PIXEL_HEIGHT: u64 = 0xBA;
const AUDIO: u64 = 0xE1;
const SAMPLING_FREQUENCY: u64 = 0xB5;
const CHANNELS: u64 = 0x9F;

const CLUSTER: u64 = 0x1F43_B675;
const CLUSTER_TIME: u64 = 0xE7;
const SIMPLE_BLOCK: u64 = 0xA3;
const BLOCK_GROUP: u64 = 0xA0;
const BLOCK: u64 = 0xA1;
const BLOCK_DURATION: u64 = 0x9B;
const REFERENCE_BLOCK: u64 = 0xFB;

/// Nanoseconds in one second. A timestamp scale is also in nanoseconds.
const NANOS_PER_SECOND: u64 = 1_000_000_000;
/// Matroska's default: one timestamp unit is one millisecond.
const DEFAULT_SCALE_NS: u64 = 1_000_000;
const DEFAULT_SAMPLE_RATE: u32 = 8_000;
const DEFAULT_CHANNELS: u16 = 1;

/// `matroska` or `webm`. Both use this reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocType {
    Matroska,
    Webm,
}

/// A parsed MKV or WebM file.
///
/// Sample times are the segment clock. There is no decode time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Movie {
    doc_type: DocType,
    timescale: NonZeroU32,
    duration: Option<i64>,
    tracks: Vec<Track>,
}

impl Movie {
    /// Read a file that is already in memory.
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        let mut cursor = ebml::Cursor::new(data, 0);
        let header = next_meaningful(&mut cursor, Limits::DEFAULT)?
            .ok_or(Error::Inconsistent("the file has no EBML header"))?;
        if header.id != EBML {
            return Err(Error::Inconsistent("the file has no EBML header"));
        }
        let (doc_type, limits) = read_header(header.body, header.body_at)?;
        let segment = next_meaningful(&mut cursor, limits)?
            .ok_or(Error::Inconsistent("the file has no segment"))?;
        if segment.id != SEGMENT {
            return Err(Error::Inconsistent("the file has no segment"));
        }
        if next_meaningful(&mut cursor, limits)?.is_some() {
            return Err(Error::Inconsistent("the file has data after the segment"));
        }
        parse_segment(data, segment.body, segment.body_at, limits, doc_type)
    }

    /// `matroska` or `webm`.
    pub fn doc_type(&self) -> DocType {
        self.doc_type
    }

    /// Ticks per second. The default timestamp scale is 1/1000 s, so this is
    /// 1000 unless the file set another scale.
    pub fn timescale(&self) -> NonZeroU32 {
        self.timescale
    }

    /// Segment duration in [`Self::timescale`] ticks. None when the file has
    /// no duration.
    pub fn duration(&self) -> Option<i64> {
        self.duration
    }

    /// Tracks in the order the file stores them.
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }
}

/// One track. Samples are in the order the clusters store them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Track {
    number: u64,
    kind: Kind,
    /// Default duration in nanoseconds, before it is converted to ticks.
    default_ns: Option<u64>,
    default_ticks: Option<i64>,
    samples: Vec<Packet>,
}

impl Track {
    /// Matroska track number. A block uses this, not the position in the list.
    pub fn number(&self) -> u64 {
        self.number
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
    Video(VideoStreamConfig),
    Audio(AudioStreamConfig),
    /// A codec this reader does not map. `codec_id` is the Matroska id, and
    /// `config_record` is `CodecPrivate` as stored.
    Other {
        codec_id: String,
        config_record: Vec<u8>,
    },
}

fn next_meaningful<'a>(
    cursor: &mut ebml::Cursor<'a>,
    limits: Limits,
) -> Result<Option<Element<'a>>, Error> {
    while let Some(element) = cursor.next(limits)? {
        if element.id == VOID || element.id == CRC32 {
            continue;
        }
        return Ok(Some(element));
    }
    Ok(None)
}

fn read_header(body: &[u8], body_at: u64) -> Result<(DocType, Limits), Error> {
    let mut doc_type = None;
    let mut read_version = None;
    let mut max_id = None;
    let mut max_size = None;
    ebml::for_each(body, body_at, Limits::DEFAULT, 1, |element| {
        match element.id {
            DOC_TYPE => {
                let name = ebml::text(element.body)?.to_string();
                once(&mut doc_type, name, "the doctype is repeated")?;
            }
            EBML_READ_VERSION => {
                once(
                    &mut read_version,
                    ebml::uint(element.body)?,
                    "the EBML read version is repeated",
                )?;
            }
            MAX_ID_LENGTH => {
                let value =
                    checked_limit(ebml::uint(element.body)?, 4, "the id length is too long")?;
                once(&mut max_id, value, "the id length is repeated")?;
            }
            MAX_SIZE_LENGTH => {
                let value =
                    checked_limit(ebml::uint(element.body)?, 8, "the size length is too long")?;
                once(&mut max_size, value, "the size length is repeated")?;
            }
            _ => {}
        }
        Ok(())
    })?;
    if let Some(version) = read_version {
        if version > 1 {
            return Err(Error::Version);
        }
    }
    let doc_type = match doc_type.as_deref() {
        Some("matroska") => DocType::Matroska,
        Some("webm") => DocType::Webm,
        Some(_) => return Err(Error::DocType),
        None => return Err(Error::Inconsistent("the file has no doctype")),
    };
    Ok((
        doc_type,
        Limits {
            max_id: max_id.unwrap_or(Limits::DEFAULT.max_id),
            max_size: max_size.unwrap_or(Limits::DEFAULT.max_size),
        },
    ))
}

fn parse_segment(
    file: &[u8],
    body: &[u8],
    body_at: u64,
    limits: Limits,
    doc_type: DocType,
) -> Result<Movie, Error> {
    let mut scale_ns = None;
    let mut duration = None;
    let mut tracks = None;
    let mut clusters = Vec::new();
    ebml::for_each(body, body_at, limits, 1, |element| {
        match element.id {
            INFO => {
                if scale_ns.is_some() || duration.is_some() {
                    return Err(Error::Inconsistent("the info element is repeated"));
                }
                let info = parse_info(element.body, element.body_at, limits)?;
                scale_ns = Some(info.scale_ns);
                duration = Some(info.duration);
            }
            TRACKS => {
                if tracks.is_some() {
                    return Err(Error::Inconsistent("the track list is repeated"));
                }
                tracks = Some(parse_tracks(element.body, element.body_at, limits)?);
            }
            CLUSTER => clusters.push(Span {
                at: element.body_at,
                len: element.body.len(),
            }),
            _ => {}
        }
        Ok(())
    })?;
    let scale_ns = scale_ns.unwrap_or(DEFAULT_SCALE_NS);
    let duration = duration.unwrap_or(None);
    let timescale = timescale_of(scale_ns)?;
    let duration = match duration {
        Some(value) => Some(i64::try_from(value).map_err(|_| Error::TooLarge)?),
        None => None,
    };
    let mut tracks = tracks.unwrap_or_default();
    set_default_ticks(&mut tracks, scale_ns)?;
    for cluster in clusters {
        let body = slice_at(file, cluster)?;
        read_cluster(body, cluster.at, limits, &mut tracks, timescale)?;
    }
    Ok(Movie {
        doc_type,
        timescale,
        duration,
        tracks,
    })
}

#[derive(Clone, Copy)]
struct Span {
    at: u64,
    len: usize,
}

fn slice_at(file: &[u8], span: Span) -> Result<&[u8], Error> {
    let start = usize::try_from(span.at).map_err(|_| Error::TooLarge)?;
    let end = start.checked_add(span.len).ok_or(Error::TooLarge)?;
    file.get(start..end).ok_or(Error::Truncated {
        at: span.at,
        need: u64::try_from(span.len).map_err(|_| Error::TooLarge)?,
    })
}

struct Info {
    scale_ns: u64,
    duration: Option<u64>,
}

fn parse_info(body: &[u8], body_at: u64, limits: Limits) -> Result<Info, Error> {
    let mut scale_ns = None;
    let mut duration = None;
    ebml::for_each(body, body_at, limits, 2, |element| {
        match element.id {
            TIMESTAMP_SCALE => {
                once(
                    &mut scale_ns,
                    ebml::uint(element.body)?,
                    "the timestamp scale is repeated",
                )?;
            }
            DURATION => {
                let value = whole_u64(
                    ebml::float(element.body)?,
                    "the duration is not a whole number",
                )?;
                once(&mut duration, value, "the duration is repeated")?;
            }
            _ => {}
        }
        Ok(())
    })?;
    Ok(Info {
        scale_ns: scale_ns.unwrap_or(DEFAULT_SCALE_NS),
        duration,
    })
}

fn parse_tracks(body: &[u8], body_at: u64, limits: Limits) -> Result<Vec<Track>, Error> {
    let mut tracks = Vec::new();
    ebml::for_each(body, body_at, limits, 2, |element| {
        if element.id == TRACK_ENTRY {
            tracks.push(parse_entry(element.body, element.body_at, limits)?);
        }
        Ok(())
    })?;
    let mut seen = Vec::new();
    for track in &tracks {
        if seen.contains(&track.number) {
            return Err(Error::Inconsistent("a track number is repeated"));
        }
        seen.push(track.number);
    }
    Ok(tracks)
}

struct Entry {
    number: Option<u64>,
    track_type: Option<u64>,
    codec_id: Option<String>,
    codec_private: Vec<u8>,
    video: Option<(u32, u32)>,
    audio: Option<AudioSettings>,
    default_ns: Option<u64>,
}

struct AudioSettings {
    sample_rate: Option<u32>,
    channels: Option<u16>,
}

fn parse_entry(body: &[u8], body_at: u64, limits: Limits) -> Result<Track, Error> {
    let mut entry = Entry {
        number: None,
        track_type: None,
        codec_id: None,
        codec_private: Vec::new(),
        video: None,
        audio: None,
        default_ns: None,
    };
    let mut saw_private = false;
    ebml::for_each(body, body_at, limits, 3, |element| {
        match element.id {
            TRACK_NUMBER => {
                once(
                    &mut entry.number,
                    ebml::uint(element.body)?,
                    "the track number is repeated",
                )?;
            }
            TRACK_TYPE => {
                once(
                    &mut entry.track_type,
                    ebml::uint(element.body)?,
                    "the track type is repeated",
                )?;
            }
            CODEC_ID => {
                let name = ebml::text(element.body)?.to_string();
                once(&mut entry.codec_id, name, "the codec id is repeated")?;
            }
            CODEC_PRIVATE => {
                if saw_private {
                    return Err(Error::Inconsistent("the codec config is repeated"));
                }
                saw_private = true;
                entry.codec_private = element.body.to_vec();
            }
            CONTENT_ENCODINGS => {
                return Err(Error::Inconsistent("a track is compressed or encrypted"));
            }
            DEFAULT_DURATION => {
                once(
                    &mut entry.default_ns,
                    ebml::uint(element.body)?,
                    "the default duration is repeated",
                )?;
            }
            VIDEO => {
                if entry.video.is_some() {
                    return Err(Error::Inconsistent("the video settings are repeated"));
                }
                entry.video = Some(parse_video(element.body, element.body_at, limits)?);
            }
            AUDIO => {
                if entry.audio.is_some() {
                    return Err(Error::Inconsistent("the audio settings are repeated"));
                }
                entry.audio = Some(parse_audio(element.body, element.body_at, limits)?);
            }
            _ => {}
        }
        Ok(())
    })?;
    finish_entry(entry)
}

fn parse_video(body: &[u8], body_at: u64, limits: Limits) -> Result<(u32, u32), Error> {
    let mut width = None;
    let mut height = None;
    ebml::for_each(body, body_at, limits, 4, |element| {
        match element.id {
            PIXEL_WIDTH => {
                once(
                    &mut width,
                    ebml::uint(element.body)?,
                    "the width is repeated",
                )?;
            }
            PIXEL_HEIGHT => {
                once(
                    &mut height,
                    ebml::uint(element.body)?,
                    "the height is repeated",
                )?;
            }
            _ => {}
        }
        Ok(())
    })?;
    let width = width.ok_or(Error::Inconsistent("a video track has no size"))?;
    let height = height.ok_or(Error::Inconsistent("a video track has no size"))?;
    if width == 0 || height == 0 {
        return Err(Error::Inconsistent(
            "video width and height must be non-zero",
        ));
    }
    let width = u32::try_from(width).map_err(|_| Error::TooLarge)?;
    let height = u32::try_from(height).map_err(|_| Error::TooLarge)?;
    Ok((width, height))
}

fn parse_audio(body: &[u8], body_at: u64, limits: Limits) -> Result<AudioSettings, Error> {
    let mut sample_rate = None;
    let mut channels = None;
    ebml::for_each(body, body_at, limits, 4, |element| {
        match element.id {
            SAMPLING_FREQUENCY => {
                let rate = positive_rate(ebml::float(element.body)?)?;
                once(&mut sample_rate, rate, "the sample rate is repeated")?;
            }
            CHANNELS => {
                let count = channel_count(ebml::uint(element.body)?)?;
                once(&mut channels, count, "the channel count is repeated")?;
            }
            _ => {}
        }
        Ok(())
    })?;
    Ok(AudioSettings {
        sample_rate,
        channels,
    })
}

fn finish_entry(entry: Entry) -> Result<Track, Error> {
    let number = entry
        .number
        .ok_or(Error::Inconsistent("a track has no number"))?;
    if number == 0 {
        return Err(Error::Inconsistent("a track number is zero"));
    }
    let track_type = entry
        .track_type
        .ok_or(Error::Inconsistent("a track has no type"))?;
    let codec_id = entry
        .codec_id
        .ok_or(Error::Inconsistent("a track has no codec"))?;
    let kind = match track_type {
        1 => video_kind(codec_id, entry.codec_private, entry.video, entry.audio)?,
        2 => audio_kind(codec_id, entry.codec_private, entry.video, entry.audio)?,
        _ => {
            if entry.video.is_some() || entry.audio.is_some() {
                return Err(Error::Inconsistent("a track has the wrong settings"));
            }
            Kind::Other {
                codec_id,
                config_record: entry.codec_private,
            }
        }
    };
    Ok(Track {
        number,
        kind,
        default_ns: entry.default_ns,
        default_ticks: None,
        samples: Vec::new(),
    })
}

fn video_kind(
    codec_id: String,
    config_record: Vec<u8>,
    video: Option<(u32, u32)>,
    audio: Option<AudioSettings>,
) -> Result<Kind, Error> {
    if audio.is_some() {
        return Err(Error::Inconsistent("a track has the wrong settings"));
    }
    let (width, height) = video.ok_or(Error::Inconsistent("a video track has no size"))?;
    if audio_codec(&codec_id).is_some() {
        return Err(Error::Inconsistent("the codec does not match the track"));
    }
    match video_codec(&codec_id) {
        Some(codec) => Ok(Kind::Video(VideoStreamConfig {
            codec,
            width,
            height,
            config_record,
        })),
        None => Ok(Kind::Other {
            codec_id,
            config_record,
        }),
    }
}

fn audio_kind(
    codec_id: String,
    config_record: Vec<u8>,
    video: Option<(u32, u32)>,
    audio: Option<AudioSettings>,
) -> Result<Kind, Error> {
    if video.is_some() {
        return Err(Error::Inconsistent("a track has the wrong settings"));
    }
    if video_codec(&codec_id).is_some() {
        return Err(Error::Inconsistent("the codec does not match the track"));
    }
    let settings = audio.unwrap_or(AudioSettings {
        sample_rate: None,
        channels: None,
    });
    match audio_codec(&codec_id) {
        Some(codec) => Ok(Kind::Audio(AudioStreamConfig {
            codec,
            sample_rate: settings.sample_rate.unwrap_or(DEFAULT_SAMPLE_RATE),
            channels: settings.channels.unwrap_or(DEFAULT_CHANNELS),
            config_record,
        })),
        None => Ok(Kind::Other {
            codec_id,
            config_record,
        }),
    }
}

fn video_codec(id: &str) -> Option<VideoCodec> {
    match id {
        "V_MPEG4/ISO/AVC" => Some(VideoCodec::H264),
        "V_MPEGH/ISO/HEVC" => Some(VideoCodec::Hevc),
        "V_VP8" => Some(VideoCodec::Vp8),
        "V_VP9" => Some(VideoCodec::Vp9),
        "V_AV1" => Some(VideoCodec::Av1),
        "V_MPEG2" => Some(VideoCodec::Mpeg2),
        "V_MPEG4/ISO/SP" | "V_MPEG4/ISO/ASP" | "V_MPEG4/ISO/AP" => Some(VideoCodec::Mpeg4Part2),
        "V_PRORES" => Some(VideoCodec::ProRes),
        "V_MJPEG" => Some(VideoCodec::Mjpeg),
        _ => None,
    }
}

fn audio_codec(id: &str) -> Option<AudioCodec> {
    if id == "A_AAC" || id.starts_with("A_AAC/") {
        return Some(AudioCodec::Aac);
    }
    match id {
        "A_MPEG/L3" => Some(AudioCodec::Mp3),
        "A_PCM/INT/LIT" | "A_PCM/INT/BIG" | "A_PCM/FLOAT/IEEE" => Some(AudioCodec::Pcm),
        "A_FLAC" => Some(AudioCodec::Flac),
        "A_OPUS" => Some(AudioCodec::Opus),
        "A_VORBIS" => Some(AudioCodec::Vorbis),
        "A_ALAC" => Some(AudioCodec::Alac),
        "A_AC3" => Some(AudioCodec::Ac3),
        _ => None,
    }
}

fn timescale_of(scale_ns: u64) -> Result<NonZeroU32, Error> {
    if scale_ns == 0 {
        return Err(Error::Inconsistent("the timestamp scale is zero"));
    }
    if scale_ns > NANOS_PER_SECOND || !NANOS_PER_SECOND.is_multiple_of(scale_ns) {
        return Err(Error::Inconsistent(
            "the timestamp scale does not divide one second",
        ));
    }
    let ticks = NANOS_PER_SECOND / scale_ns;
    let ticks = u32::try_from(ticks).map_err(|_| Error::TooLarge)?;
    match NonZeroU32::new(ticks) {
        Some(scale) => Ok(scale),
        None => Err(Error::Inconsistent(
            "the timestamp scale does not divide one second",
        )),
    }
}

/// `value` is a finite whole number in `0..=2^53`, so the cast keeps it.
fn whole_u64(value: f64, message: &'static str) -> Result<u64, Error> {
    if !value.is_finite() || value < 0.0 {
        return Err(Error::Inconsistent(message));
    }
    // 2^53 is the largest integer f64 stores exactly.
    let limit = (1u64 << 53) as f64;
    if value > limit {
        return Err(Error::TooLarge);
    }
    if value.fract() != 0.0 {
        return Err(Error::Inconsistent(message));
    }
    Ok(value as u64)
}

fn positive_rate(value: f64) -> Result<u32, Error> {
    let value = whole_u64(value, "the sample rate is not a whole number")?;
    if value == 0 {
        return Err(Error::Inconsistent("the sample rate is zero"));
    }
    u32::try_from(value).map_err(|_| Error::TooLarge)
}

fn channel_count(value: u64) -> Result<u16, Error> {
    if value == 0 {
        return Err(Error::Inconsistent("the channel count is zero"));
    }
    u16::try_from(value).map_err(|_| Error::TooLarge)
}

fn once<T>(slot: &mut Option<T>, value: T, message: &'static str) -> Result<(), Error> {
    if slot.is_some() {
        return Err(Error::Inconsistent(message));
    }
    *slot = Some(value);
    Ok(())
}

fn checked_limit(value: u64, max: u8, message: &'static str) -> Result<u8, Error> {
    let value = u8::try_from(value).map_err(|_| Error::Inconsistent(message))?;
    if value == 0 || value > max {
        return Err(Error::Inconsistent(message));
    }
    Ok(value)
}

fn set_default_ticks(tracks: &mut [Track], scale_ns: u64) -> Result<(), Error> {
    for track in tracks {
        track.default_ticks = match track.default_ns {
            Some(ns) => Some(ns_to_ticks(ns, scale_ns)?),
            None => None,
        };
    }
    Ok(())
}

fn ns_to_ticks(ns: u64, scale_ns: u64) -> Result<i64, Error> {
    if scale_ns == 0 || !ns.is_multiple_of(scale_ns) {
        return Err(Error::Inconsistent(
            "the default duration is not a whole number of timestamp units",
        ));
    }
    i64::try_from(ns / scale_ns).map_err(|_| Error::TooLarge)
}

fn read_cluster(
    body: &[u8],
    body_at: u64,
    limits: Limits,
    tracks: &mut [Track],
    timescale: NonZeroU32,
) -> Result<(), Error> {
    let mut timestamp = None;
    let mut pending = Vec::new();
    ebml::for_each(body, body_at, limits, 2, |element| {
        match element.id {
            CLUSTER_TIME => {
                once(
                    &mut timestamp,
                    ebml::uint(element.body)?,
                    "the cluster timestamp is repeated",
                )?;
            }
            SIMPLE_BLOCK => pending.push(Pending {
                block: crate::block::parse(element.body, element.body_at, limits)?,
                simple: true,
                duration: None,
                reference: false,
            }),
            BLOCK_GROUP => pending.push(parse_group(element.body, element.body_at, limits)?),
            _ => {}
        }
        Ok(())
    })?;
    let timestamp = timestamp.ok_or(Error::Inconsistent("a cluster has no timestamp"))?;
    for item in pending {
        push_block(item, timestamp, tracks, timescale)?;
    }
    Ok(())
}

struct Pending<'a> {
    block: Block<'a>,
    simple: bool,
    duration: Option<u64>,
    reference: bool,
}

fn parse_group<'a>(body: &'a [u8], body_at: u64, limits: Limits) -> Result<Pending<'a>, Error> {
    let mut block = None;
    let mut duration = None;
    let mut saw_reference = false;
    ebml::for_each(body, body_at, limits, 3, |element| {
        match element.id {
            BLOCK => {
                if block.is_some() {
                    return Err(Error::Inconsistent("a block group has two blocks"));
                }
                block = Some(crate::block::parse(element.body, element.body_at, limits)?);
            }
            SIMPLE_BLOCK => {
                return Err(Error::Inconsistent("a block group holds a simple block"));
            }
            BLOCK_DURATION => {
                once(
                    &mut duration,
                    ebml::uint(element.body)?,
                    "the block duration is repeated",
                )?;
            }
            REFERENCE_BLOCK => {
                if saw_reference {
                    return Err(Error::Inconsistent("the block reference is repeated"));
                }
                saw_reference = true;
                // Presence marks a non-keyframe. The value is a signed offset.
                ebml::uint(element.body)?;
            }
            _ => {}
        }
        Ok(())
    })?;
    let block = block.ok_or(Error::Inconsistent("a block group has no block"))?;
    Ok(Pending {
        block,
        simple: false,
        duration,
        reference: saw_reference,
    })
}

fn push_block(
    item: Pending<'_>,
    cluster_ts: u64,
    tracks: &mut [Track],
    timescale: NonZeroU32,
) -> Result<(), Error> {
    let index = tracks
        .iter()
        .position(|track| track.number == item.block.track)
        .ok_or(Error::Inconsistent("a block names an unknown track"))?;
    let audio = matches!(tracks[index].kind, Kind::Audio(_));
    let step = tracks[index].default_ticks;
    let frames = item.block.frames;
    if frames.len() > 1 {
        let step = step.ok_or(Error::Inconsistent("a laced block has no default duration"))?;
        if let Some(duration) = item.duration {
            let count = u64::try_from(frames.len()).map_err(|_| Error::TooLarge)?;
            let step_ticks = u64::try_from(step).map_err(|_| Error::TooLarge)?;
            let expected = step_ticks.checked_mul(count).ok_or(Error::TooLarge)?;
            if duration != expected {
                return Err(Error::Inconsistent(
                    "the block duration does not match the lace",
                ));
            }
        }
    }
    let mut time = block_time(cluster_ts, item.block.relative)?;
    let mut packets = Vec::with_capacity(frames.len());
    for (index, frame) in frames.iter().enumerate() {
        if index > 0 {
            let step = step.ok_or(Error::Inconsistent("a laced block has no default duration"))?;
            time = time.checked_add(step).ok_or(Error::TooLarge)?;
        }
        let duration = sample_duration(frames.len(), item.duration, step)?;
        packets.push(Packet {
            data: frame.to_vec(),
            pts: Timestamp::new(time, timescale),
            dts: None,
            duration: duration.map(|value| Timestamp::new(value, timescale)),
            keyframe: is_keyframe(
                audio,
                item.simple,
                item.block.keyframe_flag,
                item.reference,
                index,
            ),
        });
    }
    tracks[index].samples.append(&mut packets);
    Ok(())
}

fn sample_duration(
    count: usize,
    block_duration: Option<u64>,
    step: Option<i64>,
) -> Result<Option<i64>, Error> {
    if count > 1 {
        return Ok(step);
    }
    match block_duration {
        Some(duration) => Ok(Some(i64::try_from(duration).map_err(|_| Error::TooLarge)?)),
        None => Ok(step),
    }
}

fn is_keyframe(audio: bool, simple: bool, flag: bool, reference: bool, index: usize) -> bool {
    if audio {
        return true;
    }
    if index > 0 {
        return false;
    }
    if simple { flag } else { !reference }
}

fn block_time(cluster: u64, relative: i16) -> Result<i64, Error> {
    let cluster = i64::try_from(cluster).map_err(|_| Error::TooLarge)?;
    cluster
        .checked_add(i64::from(relative))
        .ok_or(Error::TooLarge)
}

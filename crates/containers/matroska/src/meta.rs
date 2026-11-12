//! The EBML header and the Segment's metadata elements, parsed from bytes
//! already held in memory: Info, Tracks, SeekHead, and Cues.
//!
//! Each element was bounded before it was read. Parsing here walks a fixed
//! schema, so nesting is never deeper than the schema (four levels inside
//! Tracks), whatever the file says.

use crate::ebml::{self, Child, Children, EbmlError};
use crate::error::{Error, Result};
use crate::ids;
use crate::limits::MAX_STRIPPED_HEADER;

/// The document type of the EBML header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DocType {
    Matroska,
    /// The WebM subset (<https://www.webmproject.org/docs/container/>).
    Webm,
}

impl DocType {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Matroska => "matroska",
            Self::Webm => "webm",
        }
    }
}

/// The parsed EBML header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EbmlHeader {
    pub doc_type: DocType,
    pub doc_type_version: u64,
    pub doc_type_read_version: u64,
}

/// The highest DocTypeReadVersion this reader implements (RFC 9559 is
/// Matroska version 4).
pub const READ_VERSION: u64 = 4;

/// Parses the data of the EBML element (RFC 8794, 11.2).
///
/// A DocType other than `matroska` or `webm` is [`Error::NotMatroska`]: it is
/// another EBML document. An EBML or DocType read version above what this
/// reader implements is [`Error::UnsupportedVersion`].
pub fn parse_ebml_header(data: &[u8]) -> Result<EbmlHeader> {
    let mut doc_type = None;
    let mut version = 1;
    let mut read_version = 1;
    let mut ebml_read_version = 1;
    let mut max_id = 4;
    let mut max_size = 8;
    for child in Children::new(data) {
        let child = child?;
        match child.id {
            ids::DOC_TYPE => doc_type = Some(ebml::text(child.data).to_vec()),
            ids::DOC_TYPE_VERSION => version = ebml::uint(child.id, child.data)?,
            ids::DOC_TYPE_READ_VERSION => read_version = ebml::uint(child.id, child.data)?,
            ids::EBML_READ_VERSION => ebml_read_version = ebml::uint(child.id, child.data)?,
            ids::EBML_MAX_ID_LENGTH => max_id = ebml::uint(child.id, child.data)?,
            ids::EBML_MAX_SIZE_LENGTH => max_size = ebml::uint(child.id, child.data)?,
            _ => {}
        }
    }
    let doc_type = match doc_type.as_deref() {
        Some(b"matroska") => DocType::Matroska,
        Some(b"webm") => DocType::Webm,
        _ => return Err(Error::NotMatroska),
    };
    if ebml_read_version != 1 {
        return Err(Error::UnsupportedVersion {
            what: "EBMLReadVersion",
            value: ebml_read_version,
        });
    }
    if read_version > READ_VERSION || read_version == 0 {
        return Err(Error::UnsupportedVersion {
            what: "DocTypeReadVersion",
            value: read_version,
        });
    }
    if max_id > 4 || max_id == 0 {
        return Err(Error::UnsupportedVersion {
            what: "EBMLMaxIDLength",
            value: max_id,
        });
    }
    if max_size > 8 || max_size == 0 {
        return Err(Error::UnsupportedVersion {
            what: "EBMLMaxSizeLength",
            value: max_size,
        });
    }
    Ok(EbmlHeader {
        doc_type,
        doc_type_version: version,
        doc_type_read_version: read_version,
    })
}

/// The Info element (RFC 9559, 5.1.2).
#[derive(Clone, Debug, PartialEq)]
pub struct Info {
    /// Nanoseconds per Segment Tick. Default 1,000,000.
    pub timestamp_scale: u64,
    /// Segment duration in Segment Ticks, when the file states one.
    pub duration: Option<f64>,
    pub muxing_app: Vec<u8>,
    pub writing_app: Vec<u8>,
    pub title: Vec<u8>,
}

impl Default for Info {
    fn default() -> Self {
        Self {
            timestamp_scale: 1_000_000,
            duration: None,
            muxing_app: Vec::new(),
            writing_app: Vec::new(),
            title: Vec::new(),
        }
    }
}

pub fn parse_info(data: &[u8]) -> Result<Info> {
    let mut info = Info::default();
    for child in Children::new(data) {
        let child = child?;
        match child.id {
            ids::TIMESTAMP_SCALE => info.timestamp_scale = ebml::uint(child.id, child.data)?,
            ids::DURATION => {
                let value = ebml::float(child.id, child.data)?;
                if value.is_finite() && value >= 0.0 {
                    info.duration = Some(value);
                }
            }
            ids::MUXING_APP => info.muxing_app = ebml::text(child.data).to_vec(),
            ids::WRITING_APP => info.writing_app = ebml::text(child.data).to_vec(),
            ids::TITLE => info.title = ebml::text(child.data).to_vec(),
            _ => {}
        }
    }
    if info.timestamp_scale == 0 {
        return Err(Error::malformed("TimestampScale is zero"));
    }
    Ok(info)
}

/// TrackType values (RFC 9559, 5.1.4.1.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackKind {
    Video,
    Audio,
    Other(u64),
}

/// How a track's frames are stored, from its ContentEncodings.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum Encoding {
    /// Frames are the codec's bytes.
    #[default]
    None,
    /// Each frame starts with these bytes, which the file left out
    /// (ContentCompAlgo 3, header stripping). They are put back on read.
    HeaderStripping(Vec<u8>),
    /// zlib, bzlib, lzo1x, or encryption. The frames are returned as they are
    /// stored and the stream's codec is reported as unknown.
    Opaque(&'static str),
}

/// One TrackEntry (RFC 9559, 5.1.4.1).
#[derive(Clone, Debug, PartialEq)]
pub struct TrackEntry {
    pub number: u64,
    pub uid: u64,
    pub kind: TrackKind,
    pub codec_id: Vec<u8>,
    pub codec_private: Vec<u8>,
    /// Nanoseconds per frame.
    pub default_duration: Option<u64>,
    /// Nanoseconds the decoder adds in front (RFC 9559, 5.1.4.1.25).
    pub codec_delay: u64,
    pub seek_pre_roll: u64,
    pub language: Option<Vec<u8>>,
    pub lacing: bool,
    pub pixel_width: u64,
    pub pixel_height: u64,
    pub display_width: Option<u64>,
    pub display_height: Option<u64>,
    pub sampling_frequency: f64,
    pub output_sampling_frequency: Option<f64>,
    pub channels: u64,
    pub bit_depth: Option<u64>,
    pub encoding: Encoding,
    /// CodecPrivate is stored with the same encoding as the frames.
    pub private_encoded: bool,
    pub track_timestamp_scale: f64,
}

impl TrackEntry {
    fn new() -> Self {
        Self {
            number: 0,
            uid: 0,
            kind: TrackKind::Other(0),
            codec_id: Vec::new(),
            codec_private: Vec::new(),
            default_duration: None,
            codec_delay: 0,
            seek_pre_roll: 0,
            language: None,
            lacing: true,
            pixel_width: 0,
            pixel_height: 0,
            display_width: None,
            display_height: None,
            sampling_frequency: 8000.0,
            output_sampling_frequency: None,
            channels: 1,
            bit_depth: None,
            encoding: Encoding::None,
            private_encoded: false,
            track_timestamp_scale: 1.0,
        }
    }
}

/// Parses the Tracks element. `max_tracks` and `max_private` are checked
/// before a track or a CodecPrivate is kept.
pub fn parse_tracks(data: &[u8], max_tracks: u32, max_private: u64) -> Result<Vec<TrackEntry>> {
    let mut tracks: Vec<TrackEntry> = Vec::new();
    for child in Children::new(data) {
        let child = child?;
        if child.id != ids::TRACK_ENTRY {
            continue;
        }
        let count = u64::try_from(tracks.len()).unwrap_or(u64::MAX);
        if count >= u64::from(max_tracks) {
            return Err(Error::Limit {
                what: "track count",
                value: count + 1,
                limit: u64::from(max_tracks),
            });
        }
        let track = parse_track(child.data, max_private)?;
        if tracks.iter().any(|other| other.number == track.number) {
            return Err(Error::malformed(format!(
                "track number {} appears twice",
                track.number
            )));
        }
        tracks.push(track);
    }
    Ok(tracks)
}

fn parse_track(data: &[u8], max_private: u64) -> Result<TrackEntry> {
    let mut track = TrackEntry::new();
    let mut kind = 0;
    for child in Children::new(data) {
        let child = child?;
        match child.id {
            ids::TRACK_NUMBER => track.number = ebml::uint(child.id, child.data)?,
            ids::TRACK_UID => track.uid = ebml::uint(child.id, child.data)?,
            ids::TRACK_TYPE => kind = ebml::uint(child.id, child.data)?,
            ids::FLAG_LACING => track.lacing = ebml::uint(child.id, child.data)? != 0,
            ids::CODEC_ID => track.codec_id = ebml::text(child.data).to_vec(),
            ids::CODEC_PRIVATE => {
                let len = u64::try_from(child.data.len()).unwrap_or(u64::MAX);
                if len > max_private {
                    return Err(Error::Limit {
                        what: "CodecPrivate bytes",
                        value: len,
                        limit: max_private,
                    });
                }
                track.codec_private = child.data.to_vec();
            }
            ids::DEFAULT_DURATION => {
                let value = ebml::uint(child.id, child.data)?;
                track.default_duration = (value > 0).then_some(value);
            }
            ids::CODEC_DELAY => track.codec_delay = ebml::uint(child.id, child.data)?,
            ids::SEEK_PRE_ROLL => track.seek_pre_roll = ebml::uint(child.id, child.data)?,
            ids::LANGUAGE => {
                if track.language.is_none() {
                    track.language = Some(ebml::text(child.data).to_vec());
                }
            }
            // The BCP 47 form wins over the legacy one (RFC 9559, 5.1.4.1.19).
            ids::LANGUAGE_BCP47 => track.language = Some(ebml::text(child.data).to_vec()),
            ids::TRACK_TIMESTAMP_SCALE => {
                track.track_timestamp_scale = ebml::float(child.id, child.data)?;
            }
            ids::VIDEO => parse_video(child.data, &mut track)?,
            ids::AUDIO => parse_audio(child.data, &mut track)?,
            ids::CONTENT_ENCODINGS => parse_encodings(child.data, &mut track)?,
            _ => {}
        }
    }
    if track.number == 0 {
        return Err(Error::malformed("a track has no TrackNumber"));
    }
    if track.number > u64::from(u32::MAX) {
        return Err(Error::unsupported(format!(
            "track number {} does not fit a track id",
            track.number
        )));
    }
    track.kind = match kind {
        1 => TrackKind::Video,
        2 => TrackKind::Audio,
        other => TrackKind::Other(other),
    };
    Ok(track)
}

fn parse_video(data: &[u8], track: &mut TrackEntry) -> Result<()> {
    for child in Children::new(data) {
        let child = child?;
        match child.id {
            ids::PIXEL_WIDTH => track.pixel_width = ebml::uint(child.id, child.data)?,
            ids::PIXEL_HEIGHT => track.pixel_height = ebml::uint(child.id, child.data)?,
            ids::DISPLAY_WIDTH => track.display_width = Some(ebml::uint(child.id, child.data)?),
            ids::DISPLAY_HEIGHT => {
                track.display_height = Some(ebml::uint(child.id, child.data)?);
            }
            _ => {}
        }
    }
    Ok(())
}

fn parse_audio(data: &[u8], track: &mut TrackEntry) -> Result<()> {
    for child in Children::new(data) {
        let child = child?;
        match child.id {
            ids::SAMPLING_FREQUENCY => {
                track.sampling_frequency = ebml::float(child.id, child.data)?;
            }
            ids::OUTPUT_SAMPLING_FREQUENCY => {
                track.output_sampling_frequency = Some(ebml::float(child.id, child.data)?);
            }
            ids::CHANNELS => track.channels = ebml::uint(child.id, child.data)?,
            ids::BIT_DEPTH => track.bit_depth = Some(ebml::uint(child.id, child.data)?),
            _ => {}
        }
    }
    Ok(())
}

/// ContentEncodings (RFC 9559, 5.1.4.1.31). One encoding is supported for
/// reading: header stripping on the frames, the CodecPrivate, or both.
/// More than one encoding, or any other algorithm, leaves the track opaque.
fn parse_encodings(data: &[u8], track: &mut TrackEntry) -> Result<()> {
    let mut found = Vec::new();
    for child in Children::new(data) {
        let child = child?;
        if child.id == ids::CONTENT_ENCODING {
            if found.len() >= 8 {
                return Err(Error::Limit {
                    what: "ContentEncoding count",
                    value: 9,
                    limit: 8,
                });
            }
            found.push(parse_encoding(child)?);
        }
    }
    match found.as_slice() {
        [] => {}
        [one] => {
            track.private_encoded = one.scope & 2 != 0;
            track.encoding = if one.scope & 1 == 0 && one.scope & 2 == 0 {
                Encoding::None
            } else {
                one.encoding.clone()
            };
        }
        _ => track.encoding = Encoding::Opaque("several content encodings"),
    }
    Ok(())
}

struct OneEncoding {
    scope: u64,
    encoding: Encoding,
}

fn parse_encoding(child: Child<'_>) -> Result<OneEncoding> {
    let mut scope = 1;
    let mut kind = 0;
    let mut algo = 0;
    let mut settings = Vec::new();
    for item in Children::new(child.data) {
        let item = item?;
        match item.id {
            ids::CONTENT_ENCODING_SCOPE => scope = ebml::uint(item.id, item.data)?,
            ids::CONTENT_ENCODING_TYPE => kind = ebml::uint(item.id, item.data)?,
            ids::CONTENT_COMPRESSION => {
                for part in Children::new(item.data) {
                    let part = part?;
                    match part.id {
                        ids::CONTENT_COMP_ALGO => algo = ebml::uint(part.id, part.data)?,
                        ids::CONTENT_COMP_SETTINGS => {
                            if part.data.len() > MAX_STRIPPED_HEADER {
                                return Err(Error::Limit {
                                    what: "ContentCompSettings bytes",
                                    value: u64::try_from(part.data.len()).unwrap_or(u64::MAX),
                                    limit: MAX_STRIPPED_HEADER as u64,
                                });
                            }
                            settings = part.data.to_vec();
                        }
                        _ => {}
                    }
                }
            }
            ids::CONTENT_ENCRYPTION => kind = 1,
            _ => {}
        }
    }
    let encoding = match (kind, algo) {
        (1, _) => Encoding::Opaque("encrypted"),
        (0, 3) => Encoding::HeaderStripping(settings),
        (0, 0) => Encoding::Opaque("zlib"),
        (0, 1) => Encoding::Opaque("bzlib"),
        (0, 2) => Encoding::Opaque("lzo1x"),
        _ => Encoding::Opaque("unknown content encoding"),
    };
    Ok(OneEncoding { scope, encoding })
}

/// One position of one cue point: a track's keyframe at `time`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cue {
    /// Segment Ticks.
    pub time: u64,
    pub track: u64,
    /// Segment Position of the Cluster (RFC 9559, section 16).
    pub cluster: u64,
    /// Offset of the block inside the Cluster's data, when stated.
    pub relative: Option<u64>,
}

/// Parses Cues into one list sorted by track, then time, then position.
/// At most `max_entries` entries are kept; more is a limit error.
pub fn parse_cues(data: &[u8], max_entries: u32) -> Result<Vec<Cue>> {
    let mut cues = Vec::new();
    for point in Children::new(data) {
        let point = point?;
        if point.id != ids::CUE_POINT {
            continue;
        }
        let mut time = None;
        let mut positions = Vec::new();
        for child in Children::new(point.data) {
            let child = child?;
            match child.id {
                ids::CUE_TIME => time = Some(ebml::uint(child.id, child.data)?),
                ids::CUE_TRACK_POSITIONS => positions.push(child.data),
                _ => {}
            }
        }
        let Some(time) = time else {
            continue;
        };
        for position in positions {
            let mut track = None;
            let mut cluster = None;
            let mut relative = None;
            for child in Children::new(position) {
                let child = child?;
                match child.id {
                    ids::CUE_TRACK => track = Some(ebml::uint(child.id, child.data)?),
                    ids::CUE_CLUSTER_POSITION => {
                        cluster = Some(ebml::uint(child.id, child.data)?);
                    }
                    ids::CUE_RELATIVE_POSITION => {
                        relative = Some(ebml::uint(child.id, child.data)?);
                    }
                    _ => {}
                }
            }
            let (Some(track), Some(cluster)) = (track, cluster) else {
                continue;
            };
            if cues.len() >= usize::try_from(max_entries).unwrap_or(usize::MAX) {
                return Err(Error::Limit {
                    what: "cue entries",
                    value: u64::from(max_entries) + 1,
                    limit: u64::from(max_entries),
                });
            }
            cues.push(Cue {
                time,
                track,
                cluster,
                relative,
            });
        }
    }
    cues.sort_unstable_by_key(|cue| (cue.track, cue.time, cue.cluster));
    Ok(cues)
}

/// One SeekHead entry: a top-level element and its Segment Position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeekEntry {
    pub id: u32,
    pub position: u64,
}

pub fn parse_seek_head(data: &[u8]) -> Result<Vec<SeekEntry>> {
    let mut entries = Vec::new();
    for seek in Children::new(data) {
        let seek = seek?;
        if seek.id != ids::SEEK {
            continue;
        }
        let mut id = None;
        let mut position = None;
        for child in Children::new(seek.data) {
            let child = child?;
            match child.id {
                ids::SEEK_ID => {
                    let (value, len) = ebml::read_id(child.data)?;
                    if len != child.data.len() {
                        return Err(Error::Ebml(EbmlError::BadLength {
                            id: child.id,
                            len: child.data.len(),
                        }));
                    }
                    id = Some(value);
                }
                ids::SEEK_POSITION => position = Some(ebml::uint(child.id, child.data)?),
                _ => {}
            }
        }
        if let (Some(id), Some(position)) = (id, position)
            && entries.len() < 1024
        {
            entries.push(SeekEntry { id, position });
        }
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::{
        Cue, DocType, Encoding, Error, TrackKind, parse_cues, parse_ebml_header, parse_info,
        parse_seek_head, parse_tracks,
    };
    use crate::ebml::{put_bytes, put_float, put_master, put_uint};
    use crate::ids;

    fn header(doc: &[u8], read_version: u64) -> Vec<u8> {
        let mut body = Vec::new();
        put_uint(&mut body, ids::EBML_VERSION, 1);
        put_uint(&mut body, ids::EBML_READ_VERSION, 1);
        put_bytes(&mut body, ids::DOC_TYPE, doc);
        put_uint(&mut body, ids::DOC_TYPE_VERSION, 4);
        put_uint(&mut body, ids::DOC_TYPE_READ_VERSION, read_version);
        body
    }

    #[test]
    fn the_header_names_matroska_or_webm_and_refuses_later_read_versions() {
        let parsed = parse_ebml_header(&header(b"webm", 2)).expect("webm");
        assert_eq!(parsed.doc_type, DocType::Webm);
        assert_eq!(parsed.doc_type_read_version, 2);
        assert_eq!(
            parse_ebml_header(&header(b"matroska\0", 4))
                .expect("padded")
                .doc_type,
            DocType::Matroska
        );
        assert_eq!(
            parse_ebml_header(&header(b"mkv", 1)),
            Err(Error::NotMatroska)
        );
        assert!(matches!(
            parse_ebml_header(&header(b"matroska", 5)),
            Err(Error::UnsupportedVersion { value: 5, .. })
        ));
    }

    #[test]
    fn info_defaults_to_milliseconds_and_refuses_a_zero_scale() {
        let info = parse_info(&[]).expect("empty");
        assert_eq!(info.timestamp_scale, 1_000_000);
        assert_eq!(info.duration, None);
        let mut body = Vec::new();
        put_uint(&mut body, ids::TIMESTAMP_SCALE, 100_000);
        put_float(&mut body, ids::DURATION, 1234.5);
        let info = parse_info(&body).expect("info");
        assert_eq!(info.timestamp_scale, 100_000);
        assert_eq!(info.duration, Some(1234.5));
        let mut zero = Vec::new();
        put_uint(&mut zero, ids::TIMESTAMP_SCALE, 0);
        assert!(parse_info(&zero).is_err());
    }

    fn track(number: u64, extra: impl Fn(&mut Vec<u8>)) -> Vec<u8> {
        let mut entry = Vec::new();
        put_uint(&mut entry, ids::TRACK_NUMBER, number);
        put_uint(&mut entry, ids::TRACK_TYPE, 2);
        put_bytes(&mut entry, ids::CODEC_ID, b"A_OPUS");
        extra(&mut entry);
        let mut out = Vec::new();
        put_master(&mut out, ids::TRACK_ENTRY, &entry);
        out
    }

    #[test]
    fn tracks_keep_numbers_unique_and_bounded() {
        let mut tracks = track(1, |_| {});
        tracks.extend(track(2, |_| {}));
        let parsed = parse_tracks(&tracks, 64, 1024).expect("tracks");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].kind, TrackKind::Audio);
        assert_eq!(parsed[0].sampling_frequency, 8000.0);
        assert_eq!(parsed[0].channels, 1);
        assert!(matches!(
            parse_tracks(&tracks, 1, 1024),
            Err(Error::Limit { .. })
        ));
        let mut twice = track(1, |_| {});
        twice.extend(track(1, |_| {}));
        assert!(parse_tracks(&twice, 64, 1024).is_err());
        let zero = track(0, |_| {});
        assert!(parse_tracks(&zero, 64, 1024).is_err());
        let private = track(1, |entry| put_bytes(entry, ids::CODEC_PRIVATE, &[0; 32]));
        assert!(matches!(
            parse_tracks(&private, 64, 16),
            Err(Error::Limit { value: 32, .. })
        ));
    }

    #[test]
    fn header_stripping_is_read_and_other_encodings_are_opaque() {
        let stripped = track(1, |entry| {
            let mut comp = Vec::new();
            put_uint(&mut comp, ids::CONTENT_COMP_ALGO, 3);
            put_bytes(&mut comp, ids::CONTENT_COMP_SETTINGS, &[0xFF, 0xFB]);
            let mut encoding = Vec::new();
            put_master(&mut encoding, ids::CONTENT_COMPRESSION, &comp);
            let mut encodings = Vec::new();
            put_master(&mut encodings, ids::CONTENT_ENCODING, &encoding);
            put_master(entry, ids::CONTENT_ENCODINGS, &encodings);
        });
        let parsed = parse_tracks(&stripped, 64, 1024).expect("stripped");
        assert_eq!(
            parsed[0].encoding,
            Encoding::HeaderStripping(vec![0xFF, 0xFB])
        );
        let zlib = track(1, |entry| {
            let mut comp = Vec::new();
            put_uint(&mut comp, ids::CONTENT_COMP_ALGO, 0);
            let mut encoding = Vec::new();
            put_master(&mut encoding, ids::CONTENT_COMPRESSION, &comp);
            let mut encodings = Vec::new();
            put_master(&mut encodings, ids::CONTENT_ENCODING, &encoding);
            put_master(entry, ids::CONTENT_ENCODINGS, &encodings);
        });
        let parsed = parse_tracks(&zlib, 64, 1024).expect("zlib");
        assert_eq!(parsed[0].encoding, Encoding::Opaque("zlib"));
    }

    #[test]
    fn cues_sort_by_track_then_time_and_respect_the_entry_limit() {
        let mut body = Vec::new();
        for (time, cluster) in [(2000_u64, 900_u64), (0, 100)] {
            let mut position = Vec::new();
            put_uint(&mut position, ids::CUE_TRACK, 1);
            put_uint(&mut position, ids::CUE_CLUSTER_POSITION, cluster);
            put_uint(&mut position, ids::CUE_RELATIVE_POSITION, 3);
            let mut point = Vec::new();
            put_uint(&mut point, ids::CUE_TIME, time);
            put_master(&mut point, ids::CUE_TRACK_POSITIONS, &position);
            put_master(&mut body, ids::CUE_POINT, &point);
        }
        let cues = parse_cues(&body, 16).expect("cues");
        assert_eq!(
            cues,
            vec![
                Cue {
                    time: 0,
                    track: 1,
                    cluster: 100,
                    relative: Some(3)
                },
                Cue {
                    time: 2000,
                    track: 1,
                    cluster: 900,
                    relative: Some(3)
                },
            ]
        );
        assert!(matches!(parse_cues(&body, 1), Err(Error::Limit { .. })));
    }

    #[test]
    fn a_seek_head_names_elements_by_id() {
        let mut seek = Vec::new();
        put_bytes(&mut seek, ids::SEEK_ID, &[0x1C, 0x53, 0xBB, 0x6B]);
        put_uint(&mut seek, ids::SEEK_POSITION, 4242);
        let mut body = Vec::new();
        put_master(&mut body, ids::SEEK, &seek);
        let entries = parse_seek_head(&body).expect("seek head");
        assert_eq!(entries[0].id, ids::CUES);
        assert_eq!(entries[0].position, 4242);
    }
}

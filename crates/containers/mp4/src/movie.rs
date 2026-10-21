//! The movie header, tracks, and edit lists.

use std::num::NonZeroU32;

use cutlass_core::{ColorTags, Packet, VideoStreamConfig};

use crate::entry::{self, Audio, Description};
use crate::read::{self, Atom, Reader};
use crate::tables;
use crate::{Error, Fourcc};

/// A parsed MP4 or MOV file.
///
/// Sample timestamps are media time. An edit list is stored on its track and
/// is not applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Movie {
    timescale: NonZeroU32,
    duration: u64,
    tracks: Vec<Track>,
}

impl Movie {
    /// Read a file that is already in memory.
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        let top = read::atoms(data, 0)?;
        if top.iter().any(|atom| atom.kind == *b"moof") {
            return Err(Error::Fragmented);
        }
        let moov = read::required(&top, *b"moov")?;
        let children = read::atoms(moov.payload, moov.payload_base()?)?;
        let (timescale, duration) = movie_header(read::required(&children, *b"mvhd")?)?;
        let mut tracks = Vec::new();
        for trak in children.iter().filter(|atom| atom.kind == *b"trak") {
            tracks.push(track(data, trak)?);
        }
        Ok(Self {
            timescale,
            duration,
            tracks,
        })
    }

    /// Timescale of the movie header. Edit-list segment durations use it.
    pub fn timescale(&self) -> NonZeroU32 {
        self.timescale
    }

    /// Duration from the movie header, in [`Self::timescale`] units.
    pub fn duration(&self) -> u64 {
        self.duration
    }

    /// Tracks in the order the file stores them.
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }
}

/// One track.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Track {
    id: u32,
    enabled: bool,
    timescale: NonZeroU32,
    duration: u64,
    kind: Kind,
    rotation: Rotation,
    display_width: u32,
    display_height: u32,
    edits: Vec<Edit>,
    samples: Vec<Packet>,
    frame_duration: Option<u32>,
    color: Option<ColorTags>,
}

impl Track {
    /// Track id from the track header.
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Whether the track header marks the track enabled.
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Media timescale. Sample timestamps use it.
    pub fn timescale(&self) -> NonZeroU32 {
        self.timescale
    }

    /// Duration from the media header, in [`Self::timescale`] units.
    pub fn duration(&self) -> u64 {
        self.duration
    }

    /// What the track holds.
    pub fn kind(&self) -> &Kind {
        &self.kind
    }

    /// Clockwise rotation from the track matrix.
    pub fn rotation(&self) -> Rotation {
        self.rotation
    }

    /// Colour tags from the sample entry, when the file has `nclx` or `nclc`.
    pub fn color(&self) -> Option<ColorTags> {
        self.color
    }

    /// Display width from the track header, after rotation.
    pub fn display_width(&self) -> u32 {
        self.display_width
    }

    /// Display height from the track header, after rotation.
    pub fn display_height(&self) -> u32 {
        self.display_height
    }

    /// Edit list. Empty when the file has none. Segment durations use the
    /// movie timescale, and media times use the track timescale.
    pub fn edits(&self) -> &[Edit] {
        &self.edits
    }

    /// Samples in decode order. Timestamps are media time.
    pub fn samples(&self) -> &[Packet] {
        &self.samples
    }

    /// `Some` when every sample has this duration, in [`Self::timescale`] units.
    /// `None` when the track is empty or the durations differ.
    pub fn frame_duration(&self) -> Option<u32> {
        self.frame_duration
    }
}

/// A video track, an audio track, or a track this reader does not demux.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// A video sample entry and its decoder configuration.
    Video(VideoStreamConfig),
    /// An audio sample entry.
    Audio(Audio),
    /// The handler type of a track that is left undemuxed, such as timecode.
    Other(Fourcc),
}

/// Clockwise rotation stored in a track matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rotation {
    /// The identity matrix.
    Zero,
    /// 90 degrees clockwise.
    Clockwise90,
    /// 180 degrees.
    Clockwise180,
    /// 270 degrees clockwise.
    Clockwise270,
    /// A matrix that is not one of those four.
    Other,
}

/// One edit-list entry, not applied to the samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edit {
    segment_duration: u64,
    media_time: i64,
    media_rate: u32,
}

impl Edit {
    /// Duration of the edit in the movie timescale.
    pub fn segment_duration(&self) -> u64 {
        self.segment_duration
    }

    /// Media time where this edit starts, in the track timescale.
    /// Negative one is an empty edit.
    pub fn media_time(&self) -> i64 {
        self.media_time
    }

    /// Playback rate as a 16.16 fixed-point number.
    pub fn media_rate(&self) -> u32 {
        self.media_rate
    }
}

fn track(file: &[u8], trak: &Atom<'_>) -> Result<Track, Error> {
    let children = read::atoms(trak.payload, trak.payload_base()?)?;
    let header = track_header(read::required(&children, *b"tkhd")?)?;
    let mdia = read::required(&children, *b"mdia")?;
    let media = read::atoms(mdia.payload, mdia.payload_base()?)?;
    let (timescale, duration) = media_header(read::required(&media, *b"mdhd")?)?;
    let handler = handler_type(read::required(&media, *b"hdlr")?)?;
    let edits = edits(&children)?;
    let (kind, color, samples) = if handler == *b"vide" || handler == *b"soun" {
        let minf = read::required(&media, *b"minf")?;
        let minf_children = read::atoms(minf.payload, minf.payload_base()?)?;
        let stbl = read::required(&minf_children, *b"stbl")?;
        let tables_atoms = read::atoms(stbl.payload, stbl.payload_base()?)?;
        let described = entry::description(read::required(&tables_atoms, *b"stsd")?, handler)?;
        let (kind, color) = match described {
            Description::Video(video) => (Kind::Video(video.config), video.color),
            Description::Audio(audio) => (Kind::Audio(audio), None),
        };
        let samples = tables::packets(file, timescale, &tables_atoms)?;
        (kind, color, samples)
    } else {
        (Kind::Other(Fourcc::new(handler)), None, Vec::new())
    };
    Ok(Track {
        id: header.id,
        enabled: header.enabled,
        timescale,
        duration,
        kind,
        rotation: header.rotation,
        display_width: header.display_width,
        display_height: header.display_height,
        edits,
        frame_duration: constant_duration(&samples),
        samples,
        color,
    })
}

struct Header {
    id: u32,
    enabled: bool,
    rotation: Rotation,
    display_width: u32,
    display_height: u32,
}

fn movie_header(atom: &Atom<'_>) -> Result<(NonZeroU32, u64), Error> {
    let (version, _, body) = read::full_box(atom)?;
    let mut reader = Reader::new(body, atom.payload_base()? + 4);
    let (timescale, duration) = match version {
        0 => {
            reader.skip(8)?;
            (reader.u32()?, u64::from(reader.u32()?))
        }
        1 => {
            reader.skip(16)?;
            (reader.u32()?, reader.u64()?)
        }
        _ => {
            return Err(Error::BadVersion {
                kind: Fourcc::new(*b"mvhd"),
                version,
            });
        }
    };
    let timescale = NonZeroU32::new(timescale).ok_or(Error::ZeroTimescale)?;
    Ok((timescale, duration))
}

fn track_header(atom: &Atom<'_>) -> Result<Header, Error> {
    let (version, flags, body) = read::full_box(atom)?;
    let mut reader = Reader::new(body, atom.payload_base()? + 4);
    let (id, matrix_at) = match version {
        0 => {
            reader.skip(8)?;
            let id = reader.u32()?;
            reader.skip(4)?;
            reader.u32()?;
            reader.skip(16)?;
            (id, reader.file_pos()?)
        }
        1 => {
            reader.skip(16)?;
            let id = reader.u32()?;
            reader.skip(4)?;
            reader.u64()?;
            reader.skip(16)?;
            (id, reader.file_pos()?)
        }
        _ => {
            return Err(Error::BadVersion {
                kind: Fourcc::new(*b"tkhd"),
                version,
            });
        }
    };
    if id == 0 {
        return Err(Error::BadTrackId);
    }
    let mut matrix = [0i32; 9];
    let mut matrix_reader = Reader::new(reader.rest(), matrix_at);
    for value in &mut matrix {
        *value = matrix_reader.i32()?;
    }
    let display_width = matrix_reader.u32()? >> 16;
    let display_height = matrix_reader.u32()? >> 16;
    Ok(Header {
        id,
        enabled: flags & 1 == 1,
        rotation: rotation(matrix),
        display_width,
        display_height,
    })
}

fn media_header(atom: &Atom<'_>) -> Result<(NonZeroU32, u64), Error> {
    let (version, _, body) = read::full_box(atom)?;
    let mut reader = Reader::new(body, atom.payload_base()? + 4);
    let (timescale, duration) = match version {
        0 => {
            reader.skip(8)?;
            (reader.u32()?, u64::from(reader.u32()?))
        }
        1 => {
            reader.skip(16)?;
            (reader.u32()?, reader.u64()?)
        }
        _ => {
            return Err(Error::BadVersion {
                kind: Fourcc::new(*b"mdhd"),
                version,
            });
        }
    };
    let timescale = NonZeroU32::new(timescale).ok_or(Error::ZeroTimescale)?;
    Ok((timescale, duration))
}

fn handler_type(atom: &Atom<'_>) -> Result<[u8; 4], Error> {
    let (_, _, body) = read::full_box(atom)?;
    let mut reader = Reader::new(body, atom.payload_base()? + 4);
    reader.skip(4)?;
    reader.fourcc()
}

fn edits(trak: &[Atom<'_>]) -> Result<Vec<Edit>, Error> {
    let Some(edts) = read::only(trak, *b"edts")? else {
        return Ok(Vec::new());
    };
    let children = read::atoms(edts.payload, edts.payload_base()?)?;
    let Some(elst) = read::only(&children, *b"elst")? else {
        return Ok(Vec::new());
    };
    let (version, _, body) = read::full_box(elst)?;
    let mut reader = Reader::new(body, elst.payload_base()? + 4);
    let count = reader.u32()?;
    let mut edits = Vec::new();
    match version {
        0 => {
            for _ in 0..count {
                edits.push(Edit {
                    segment_duration: u64::from(reader.u32()?),
                    media_time: i64::from(reader.i32()?),
                    media_rate: reader.u32()?,
                });
            }
        }
        1 => {
            for _ in 0..count {
                edits.push(Edit {
                    segment_duration: reader.u64()?,
                    media_time: reader.i64()?,
                    media_rate: reader.u32()?,
                });
            }
        }
        _ => {
            return Err(Error::BadVersion {
                kind: Fourcc::new(*b"elst"),
                version,
            });
        }
    }
    Ok(edits)
}

fn rotation(matrix: [i32; 9]) -> Rotation {
    let one = 0x0001_0000;
    let (a, b, c, d) = (matrix[0], matrix[1], matrix[3], matrix[4]);
    match (a, b, c, d) {
        (a, 0, 0, d) if a == one && d == one => Rotation::Zero,
        (0, b, c, 0) if b == one && c == -one => Rotation::Clockwise90,
        (a, 0, 0, d) if a == -one && d == -one => Rotation::Clockwise180,
        (0, b, c, 0) if b == -one && c == one => Rotation::Clockwise270,
        _ => Rotation::Other,
    }
}

fn constant_duration(samples: &[Packet]) -> Option<u32> {
    let first = samples.first()?.duration?;
    let value = u32::try_from(first.value).ok()?;
    if samples.iter().all(|sample| sample.duration == Some(first)) {
        Some(value)
    } else {
        None
    }
}

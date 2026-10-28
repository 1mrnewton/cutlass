//! `moov` and the sample tables it owns.
//!
//! Edit lists are stored on the track. They are not applied to packet timestamps.

use crate::index::{self, ChunkMap, CtsRun, Run, Sample, Sizes, Sync, Tables};
use crate::legacy::{Edit, Error, MediaKind, Stream, StreamParams, Timescale, VideoParams};
use crate::limits::{MAX_EDITS, MAX_SAMPLES, MAX_TABLE_ROWS, MAX_TRACKS};
use crate::parse::{self, Reader};
use crate::stsd::{self, SampleDesc};

#[derive(Clone, Copy)]
pub struct Trex {
    pub track_id: u32,
    pub duration: u32,
    pub size: u32,
    pub flags: u32,
}

pub struct BuiltTrack {
    pub stream: Stream,
    pub samples: Vec<Sample>,
    pub no_sync: bool,
    pub self_contained: bool,
}

pub struct Built {
    pub timescale: u32,
    pub duration: u64,
    pub tracks: Vec<BuiltTrack>,
    pub trex: Vec<Trex>,
    pub fragmented: bool,
}

struct Draft {
    id: u32,
    timescale: u32,
    language: [u8; 3],
    handler: [u8; 4],
    duration: u64,
    width: u32,
    height: u32,
    entries: Vec<SampleDesc>,
    stts: Vec<Run>,
    ctts: Vec<CtsRun>,
    stsc: Vec<ChunkMap>,
    sizes: Sizes,
    chunks: Vec<u64>,
    sync: Sync,
    partial_sync: Vec<u32>,
    edits: Vec<Edit>,
    refs_in_file: Vec<bool>,
}

pub fn parse_moov(payload: &[u8], budget: &mut u32, depth: u32) -> Result<Built, Error> {
    let depth = parse::enter(depth)?;
    let mut built = Built {
        timescale: 0,
        duration: 0,
        tracks: Vec::new(),
        trex: Vec::new(),
        fragmented: false,
    };
    for atom in parse::atoms(payload, 0, budget)? {
        match &atom.kind {
            b"mvhd" => parse_mvhd(atom.payload, &mut built)?,
            b"trak" => {
                if built.tracks.len() >= MAX_TRACKS {
                    return Err(Error::Limit {
                        what: "tracks",
                        value: u64::try_from(built.tracks.len()).unwrap_or(u64::MAX),
                    });
                }
                built.tracks.push(parse_trak(atom.payload, budget, depth)?);
            }
            b"mvex" => {
                built.fragmented = true;
                parse_mvex(atom.payload, &mut built, budget, depth)?;
            }
            _ => {}
        }
    }
    if built.timescale == 0 {
        return Err(Error::Unsupported {
            what: "movie timescale 0",
        });
    }
    Ok(built)
}

fn parse_mvhd(payload: &[u8], built: &mut Built) -> Result<(), Error> {
    let mut reader = Reader::new(payload, 0);
    let version = reader.u8()?;
    reader.skip(3)?;
    if version == 0 {
        reader.skip(8)?;
        built.timescale = reader.u32()?;
        built.duration = u64::from(reader.u32()?);
    } else {
        reader.skip(16)?;
        built.timescale = reader.u32()?;
        built.duration = reader.u64()?;
    }
    if built.timescale == 0 {
        return Err(Error::Unsupported {
            what: "movie timescale 0",
        });
    }
    Ok(())
}

fn parse_mvex(
    payload: &[u8],
    built: &mut Built,
    budget: &mut u32,
    depth: u32,
) -> Result<(), Error> {
    let _depth = parse::enter(depth)?;
    for atom in parse::atoms(payload, 0, budget)? {
        if atom.kind != *b"trex" {
            continue;
        }
        let mut reader = Reader::new(atom.payload, 0);
        reader.skip(4)?;
        let track_id = reader.u32()?;
        reader.skip(4)?;
        built.trex.push(Trex {
            track_id,
            duration: reader.u32()?,
            size: reader.u32()?,
            flags: reader.u32()?,
        });
    }
    Ok(())
}

fn parse_trak(payload: &[u8], budget: &mut u32, depth: u32) -> Result<BuiltTrack, Error> {
    let depth = parse::enter(depth)?;
    let mut draft = Draft {
        id: 0,
        timescale: 0,
        language: *b"und",
        handler: *b"none",
        duration: 0,
        width: 0,
        height: 0,
        entries: Vec::new(),
        stts: Vec::new(),
        ctts: Vec::new(),
        stsc: Vec::new(),
        sizes: Sizes::Constant { size: 0, count: 0 },
        chunks: Vec::new(),
        sync: Sync::All,
        partial_sync: Vec::new(),
        edits: Vec::new(),
        refs_in_file: Vec::new(),
    };
    for atom in parse::atoms(payload, 0, budget)? {
        match &atom.kind {
            b"tkhd" => parse_tkhd(atom.payload, &mut draft)?,
            b"edts" => parse_edts(atom.payload, &mut draft, budget, depth)?,
            b"mdia" => parse_mdia(atom.payload, &mut draft, budget, depth)?,
            _ => {}
        }
    }
    if draft.timescale == 0 {
        return Err(Error::Unsupported {
            what: "media timescale 0",
        });
    }
    let entry = draft.entries.first();
    let (codec, kind, extradata, color, sample_entry, data_ref) = match entry {
        Some(desc) => (
            desc.codec,
            fit_kind(desc.kind, draft.width, draft.height),
            desc.extradata.clone(),
            desc.color,
            desc.fourcc,
            desc.data_reference,
        ),
        None => {
            return Err(Error::Missing {
                what: "sample description",
            });
        }
    };
    let self_contained = data_ref == 0
        || draft.refs_in_file.is_empty()
        || draft
            .refs_in_file
            .get(usize::from(data_ref.saturating_sub(1)))
            .copied()
            .unwrap_or(false);
    let tables = Tables {
        stts: draft.stts,
        ctts: draft.ctts,
        stsc: draft.stsc,
        sizes: draft.sizes,
        chunks: draft.chunks,
        sync: draft.sync,
        partial_sync: draft.partial_sync,
    };
    let no_sync = index::no_sync_samples(&tables.sync, &tables.partial_sync);
    let samples = if tables.sizes.count() == 0 {
        Vec::new()
    } else {
        index::build(&tables)?
    };
    let duration_ticks = samples.iter().fold(0u64, |sum, sample| {
        sum.saturating_add(u64::from(sample.duration))
    });
    Ok(BuiltTrack {
        stream: Stream {
            params: StreamParams {
                track_id: draft.id,
                codec,
                kind,
                timescale: Timescale(draft.timescale),
                language: draft.language,
                extradata,
                duration_ticks: if duration_ticks == 0 {
                    draft.duration
                } else {
                    duration_ticks
                },
                sample_entry,
                color,
            },
            edits: draft.edits,
        },
        samples,
        no_sync,
        self_contained,
    })
}

fn fit_kind(kind: MediaKind, tkhd_width: u32, tkhd_height: u32) -> MediaKind {
    match kind {
        MediaKind::Video(video) if video.width == 0 && video.height == 0 => {
            MediaKind::Video(VideoParams {
                width: tkhd_width,
                height: tkhd_height,
            })
        }
        other => other,
    }
}

fn parse_tkhd(payload: &[u8], draft: &mut Draft) -> Result<(), Error> {
    let mut reader = Reader::new(payload, 0);
    let version = reader.u8()?;
    reader.skip(3)?;
    if version == 0 {
        reader.skip(8)?;
        draft.id = reader.u32()?;
        reader.skip(4)?;
        let _duration = reader.u32()?;
    } else {
        reader.skip(16)?;
        draft.id = reader.u32()?;
        reader.skip(4)?;
        let _duration = reader.u64()?;
    }
    reader.skip(16)?;
    reader.skip(36)?;
    let width = reader.u32()?;
    let height = reader.u32()?;
    draft.width = width >> 16;
    draft.height = height >> 16;
    Ok(())
}

fn parse_edts(
    payload: &[u8],
    draft: &mut Draft,
    budget: &mut u32,
    depth: u32,
) -> Result<(), Error> {
    let _depth = parse::enter(depth)?;
    for atom in parse::atoms(payload, 0, budget)? {
        if atom.kind == *b"elst" {
            parse_elst(atom.payload, draft)?;
        }
    }
    Ok(())
}

fn parse_elst(payload: &[u8], draft: &mut Draft) -> Result<(), Error> {
    let mut reader = Reader::new(payload, 0);
    let version = reader.u8()?;
    reader.skip(3)?;
    let count = reader.u32()?;
    if u64::from(count) > u64::try_from(MAX_EDITS).unwrap_or(0) {
        return Err(Error::Limit {
            what: "edits",
            value: u64::from(count),
        });
    }
    for _ in 0..count {
        let (duration, media_time) = if version == 0 {
            let duration = u64::from(reader.u32()?);
            let media = reader.i32()?;
            (duration, i64::from(media))
        } else {
            (reader.u64()?, reader.i64()?)
        };
        let media_rate = reader.u32()?;
        draft.edits.push(Edit {
            segment_duration: duration,
            media_time: if media_time == -1 {
                None
            } else {
                Some(media_time)
            },
            media_rate,
        });
    }
    Ok(())
}

fn parse_mdia(
    payload: &[u8],
    draft: &mut Draft,
    budget: &mut u32,
    depth: u32,
) -> Result<(), Error> {
    let depth = parse::enter(depth)?;
    for atom in parse::atoms(payload, 0, budget)? {
        match &atom.kind {
            b"mdhd" => parse_mdhd(atom.payload, draft)?,
            b"hdlr" => parse_hdlr(atom.payload, draft)?,
            b"minf" => parse_minf(atom.payload, draft, budget, depth)?,
            _ => {}
        }
    }
    Ok(())
}

fn parse_mdhd(payload: &[u8], draft: &mut Draft) -> Result<(), Error> {
    let mut reader = Reader::new(payload, 0);
    let version = reader.u8()?;
    reader.skip(3)?;
    if version == 0 {
        reader.skip(8)?;
        draft.timescale = reader.u32()?;
        draft.duration = u64::from(reader.u32()?);
    } else {
        reader.skip(16)?;
        draft.timescale = reader.u32()?;
        draft.duration = reader.u64()?;
    }
    let language = reader.u16()?;
    draft.language = unpack_language(language);
    if draft.timescale == 0 {
        return Err(Error::Unsupported {
            what: "media timescale 0",
        });
    }
    Ok(())
}

fn parse_hdlr(payload: &[u8], draft: &mut Draft) -> Result<(), Error> {
    let mut reader = Reader::new(payload, 0);
    reader.skip(4)?;
    reader.skip(4)?;
    let mut kind = [0u8; 4];
    kind.copy_from_slice(reader.take(4)?);
    draft.handler = kind;
    Ok(())
}

fn parse_minf(
    payload: &[u8],
    draft: &mut Draft,
    budget: &mut u32,
    depth: u32,
) -> Result<(), Error> {
    let depth = parse::enter(depth)?;
    for atom in parse::atoms(payload, 0, budget)? {
        match &atom.kind {
            b"dinf" => parse_dinf(atom.payload, draft, budget, depth)?,
            b"stbl" => parse_stbl(atom.payload, draft, budget, depth)?,
            _ => {}
        }
    }
    Ok(())
}

fn parse_dinf(
    payload: &[u8],
    draft: &mut Draft,
    budget: &mut u32,
    depth: u32,
) -> Result<(), Error> {
    let _depth = parse::enter(depth)?;
    for atom in parse::atoms(payload, 0, budget)? {
        if atom.kind == *b"dref" {
            parse_dref(atom.payload, draft, budget)?;
        }
    }
    Ok(())
}

fn parse_dref(payload: &[u8], draft: &mut Draft, budget: &mut u32) -> Result<(), Error> {
    let mut reader = Reader::new(payload, 0);
    reader.skip(4)?;
    let _count = reader.u32()?;
    for atom in parse::atoms(reader.remaining(), 0, budget)? {
        if atom.payload.len() < 4 {
            return Err(Error::Truncated { at: 0, needed: 4 });
        }
        let flags = u32::from_be_bytes([0, atom.payload[1], atom.payload[2], atom.payload[3]]);
        draft.refs_in_file.push(flags & 1 == 1);
    }
    Ok(())
}

fn parse_stbl(
    payload: &[u8],
    draft: &mut Draft,
    budget: &mut u32,
    depth: u32,
) -> Result<(), Error> {
    let depth = parse::enter(depth)?;
    for atom in parse::atoms(payload, 0, budget)? {
        match &atom.kind {
            b"stsd" => {
                draft.entries = stsd::parse_stsd(atom.payload, draft.handler, budget, depth)?;
            }
            b"stts" => draft.stts = parse_stts(atom.payload)?,
            b"ctts" => draft.ctts = parse_ctts(atom.payload)?,
            b"stsc" => draft.stsc = parse_stsc(atom.payload)?,
            b"stsz" => draft.sizes = parse_stsz(atom.payload)?,
            b"stz2" => draft.sizes = parse_stz2(atom.payload)?,
            b"stco" => draft.chunks = parse_offsets(atom.payload, false)?,
            b"co64" => draft.chunks = parse_offsets(atom.payload, true)?,
            b"stss" => draft.sync = Sync::Only(parse_sync_numbers(atom.payload)?),
            b"stps" => draft.partial_sync = parse_sync_numbers(atom.payload)?,
            _ => {}
        }
    }
    Ok(())
}

fn parse_stts(payload: &[u8]) -> Result<Vec<Run>, Error> {
    let mut reader = Reader::new(payload, 0);
    reader.skip(4)?;
    let count = reader.u32()?;
    check_rows(count)?;
    let mut rows = Vec::new();
    for _ in 0..count {
        rows.push(Run {
            count: reader.u32()?,
            delta: reader.u32()?,
        });
    }
    Ok(rows)
}

fn parse_ctts(payload: &[u8]) -> Result<Vec<CtsRun>, Error> {
    let mut reader = Reader::new(payload, 0);
    let version = reader.u8()?;
    reader.skip(3)?;
    let count = reader.u32()?;
    check_rows(count)?;
    let mut rows = Vec::new();
    for _ in 0..count {
        let samples = reader.u32()?;
        let offset = if version == 0 {
            i64::from(reader.u32()?)
        } else {
            i64::from(reader.i32()?)
        };
        if samples > 0 {
            rows.push(CtsRun {
                count: samples,
                offset,
            });
        }
    }
    Ok(rows)
}

fn parse_stsc(payload: &[u8]) -> Result<Vec<ChunkMap>, Error> {
    let mut reader = Reader::new(payload, 0);
    reader.skip(4)?;
    let count = reader.u32()?;
    check_rows(count)?;
    let mut rows = Vec::new();
    for _ in 0..count {
        rows.push(ChunkMap {
            first_chunk: reader.u32()?,
            samples_per_chunk: reader.u32()?,
        });
        let _description = reader.u32()?;
    }
    Ok(rows)
}

fn parse_stsz(payload: &[u8]) -> Result<Sizes, Error> {
    let mut reader = Reader::new(payload, 0);
    reader.skip(4)?;
    let sample_size = reader.u32()?;
    let count = reader.u32()?;
    if u64::from(count) > MAX_SAMPLES {
        return Err(Error::Limit {
            what: "samples",
            value: u64::from(count),
        });
    }
    if sample_size != 0 {
        return Ok(Sizes::Constant {
            size: sample_size,
            count: u64::from(count),
        });
    }
    let mut rows = Vec::new();
    for _ in 0..count {
        rows.push(reader.u32()?);
    }
    Ok(Sizes::Table(rows))
}

fn parse_stz2(payload: &[u8]) -> Result<Sizes, Error> {
    let mut reader = Reader::new(payload, 0);
    reader.skip(4)?;
    reader.skip(3)?;
    let field = reader.u8()?;
    let count = reader.u32()?;
    if u64::from(count) > MAX_SAMPLES {
        return Err(Error::Limit {
            what: "samples",
            value: u64::from(count),
        });
    }
    if !matches!(field, 4 | 8 | 16) {
        return Err(Error::Unsupported {
            what: "sample size field",
        });
    }
    let mut rows = Vec::new();
    if field == 4 {
        let mut pending: Option<u8> = None;
        for _ in 0..count {
            let nibble = match pending.take() {
                Some(low) => low,
                None => {
                    let byte = reader.u8()?;
                    pending = Some(byte & 0x0f);
                    byte >> 4
                }
            };
            rows.push(u32::from(nibble));
        }
    } else if field == 8 {
        for _ in 0..count {
            rows.push(u32::from(reader.u8()?));
        }
    } else {
        for _ in 0..count {
            rows.push(u32::from(reader.u16()?));
        }
    }
    Ok(Sizes::Table(rows))
}

fn parse_offsets(payload: &[u8], wide: bool) -> Result<Vec<u64>, Error> {
    let mut reader = Reader::new(payload, 0);
    reader.skip(4)?;
    let count = reader.u32()?;
    check_rows(count)?;
    let mut rows = Vec::new();
    for _ in 0..count {
        rows.push(if wide {
            reader.u64()?
        } else {
            u64::from(reader.u32()?)
        });
    }
    Ok(rows)
}

fn parse_sync_numbers(payload: &[u8]) -> Result<Vec<u32>, Error> {
    let mut reader = Reader::new(payload, 0);
    reader.skip(4)?;
    let count = reader.u32()?;
    check_rows(count)?;
    let mut rows = Vec::new();
    for _ in 0..count {
        let number = reader.u32()?;
        if let Some(index) = number.checked_sub(1) {
            rows.push(index);
        }
    }
    rows.sort_unstable();
    rows.dedup();
    Ok(rows)
}

fn check_rows(count: u32) -> Result<(), Error> {
    if usize::try_from(count).unwrap_or(usize::MAX) > MAX_TABLE_ROWS {
        Err(Error::Limit {
            what: "table rows",
            value: u64::from(count),
        })
    } else {
        Ok(())
    }
}

fn unpack_language(packed: u16) -> [u8; 3] {
    let byte = |shift: u16| -> u8 {
        let value = u8::try_from((packed >> shift) & 0x1f).unwrap_or(0);
        value.saturating_add(0x60)
    };
    [byte(10), byte(5), byte(0)]
}

#[must_use]
pub fn pack_language(language: [u8; 3]) -> u16 {
    let piece = |byte: u8| -> u16 { u16::from(byte.saturating_sub(0x60) & 0x1f) };
    (piece(language[0]) << 10) | (piece(language[1]) << 5) | piece(language[2])
}

trait SizeCount {
    fn count(&self) -> u64;
}

impl SizeCount for Sizes {
    fn count(&self) -> u64 {
        match self {
            Self::Constant { count, .. } => *count,
            Self::Table(rows) => u64::try_from(rows.len()).unwrap_or(u64::MAX),
        }
    }
}

//! Chunk tables from the OpenDML index, from `idx1`, or from a scan.
//!
//! Order of preference:
//!
//! 1. OpenDML: each stream's `indx` super index names `ix##` standard
//!    indexes, one per `RIFF` segment. An `indx` that is itself a standard
//!    index is read directly. Field indexes (`AVI_INDEX_2FIELD`) are read
//!    for their first-field offset. Every stream needs one, with at least
//!    one row; a writer that reserved an `indx` and never filled it (one
//!    `RIFF`, the index in `idx1`) falls through to `idx1`.
//! 2. `idx1`, which covers the first `RIFF` only. Its offsets are relative
//!    to the `movi` form type, as the specification says, or absolute, as
//!    some writers store them; the first entry's chunk header decides, and
//!    an offset from the first chunk in `movi` is tried last, as FFmpeg
//!    does. Any `AVIX` segment after it is scanned.
//! 3. A scan of every `movi` list that reads chunk headers, plus the first
//!    [`SNIFF_BYTES`] of each video chunk to find keyframes. After a broken
//!    header the scan looks [`MAX_RESYNC`] bytes ahead for the next one,
//!    then gives up on that list.
//!
//! An index that is malformed, or that names a chunk outside every `movi`
//! list, is dropped and the next source is tried; [`IndexSource`] says which
//! one was used and [`Fallback`] why. Running out of the entry budget
//! ([`cutlass_container::Limits::max_index_entries`]) is an open error.

use std::io::{Read, Seek};

use cutlass_container::{Limits, OpenError};

use crate::bytes::{id_at, u16_at, u32_at, u64_at};
use crate::format::StreamKind;
use crate::layout::{Layout, Movi};
use crate::limits::{MAX_RESYNC, MAX_SUPER_ENTRIES, SNIFF_BYTES};
use crate::sniff;
use crate::source::Source;
use crate::track::Track;

/// `AVIIF_LIST`: the entry is a `rec ` list.
pub const AVIIF_LIST: u32 = 0x01;
/// `AVIIF_KEYFRAME`.
pub const AVIIF_KEYFRAME: u32 = 0x10;
/// `AVIIF_NO_TIME`: palette changes and the like.
pub const AVIIF_NO_TIME: u32 = 0x100;

/// Where the chunk table came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexSource {
    /// `indx` super indexes and their `ix##` standard indexes.
    OpenDml,
    /// `idx1` alone.
    Idx1,
    /// `idx1` for the first `RIFF`, then a scan of the `AVIX` segments.
    Idx1AndScan,
    /// A scan of the `movi` lists.
    Scan,
}

/// Why an index the file has was not used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fallback {
    pub index: &'static str,
    pub reason: String,
}

enum Fail {
    Broken(String),
    Fatal(OpenError),
}

impl From<OpenError> for Fail {
    fn from(error: OpenError) -> Self {
        match error {
            OpenError::HostileSize { .. } => Self::Fatal(error),
            other => Self::Broken(other.to_string()),
        }
    }
}

fn broken(reason: impl Into<String>) -> Fail {
    Fail::Broken(reason.into())
}

pub(crate) struct Built {
    pub source: IndexSource,
    pub fallbacks: Vec<Fallback>,
    pub resyncs: u32,
}

pub(crate) fn build<R: Read + Seek>(
    source: &mut Source<R>,
    tracks: &mut [Track],
    layout: &Layout,
    limits: &Limits,
) -> Result<Built, OpenError> {
    let mut fallbacks = Vec::new();
    if tracks.iter().any(|track| track.super_index.is_some()) {
        let mut budget = u64::from(limits.max_index_entries);
        match read_odml(source, tracks, layout, limits, &mut budget) {
            Ok(()) => return Ok(finish(tracks, IndexSource::OpenDml, fallbacks, 0)),
            Err(Fail::Fatal(error)) => return Err(error),
            Err(Fail::Broken(reason)) => {
                fallbacks.push(Fallback {
                    index: "indx",
                    reason,
                });
                reset(tracks);
            }
        }
    }
    if let Some((at, len)) = layout.idx1
        && let Some(first) = layout.movi.first()
    {
        let mut budget = u64::from(limits.max_index_entries);
        match read_idx1(source, tracks, *first, at, len, &mut budget) {
            Ok(()) if layout.movi.len() == 1 => {
                return Ok(finish(tracks, IndexSource::Idx1, fallbacks, 0));
            }
            Ok(()) => {
                let resyncs = scan(source, tracks, &layout.movi[1..], &mut budget)?;
                return Ok(finish(tracks, IndexSource::Idx1AndScan, fallbacks, resyncs));
            }
            Err(Fail::Fatal(error)) => return Err(error),
            Err(Fail::Broken(reason)) => {
                fallbacks.push(Fallback {
                    index: "idx1",
                    reason,
                });
                reset(tracks);
            }
        }
    }
    let mut budget = u64::from(limits.max_index_entries);
    let resyncs = scan(source, tracks, &layout.movi, &mut budget)?;
    Ok(finish(tracks, IndexSource::Scan, fallbacks, resyncs))
}

fn finish(
    tracks: &mut [Track],
    source: IndexSource,
    fallbacks: Vec<Fallback>,
    resyncs: u32,
) -> Built {
    for track in tracks.iter_mut() {
        track.finish();
    }
    Built {
        source,
        fallbacks,
        resyncs,
    }
}

fn reset(tracks: &mut [Track]) {
    for track in tracks {
        track.reset();
    }
}

/// The stream number in a chunk id: two ASCII digits, `NNxx`.
pub(crate) fn stream_of(id: [u8; 4]) -> Option<u32> {
    let (tens, ones) = (id[0], id[1]);
    if !tens.is_ascii_digit() || !ones.is_ascii_digit() {
        return None;
    }
    Some(u32::from(tens - b'0') * 10 + u32::from(ones - b'0'))
}

/// The stream number in an OpenDML index id: `ix##` or `##ix`.
fn index_stream_of(id: [u8; 4]) -> Option<u32> {
    if &id[..2] == b"ix" {
        return stream_of([id[2], id[3], 0, 0]);
    }
    if &id[2..] == b"ix" {
        return stream_of(id);
    }
    None
}

fn read_idx1<R: Read + Seek>(
    source: &mut Source<R>,
    tracks: &mut [Track],
    movi: Movi,
    at: u64,
    len: u64,
    budget: &mut u64,
) -> Result<(), Fail> {
    let count = len / 16;
    if count == 0 {
        return Err(broken("idx1 is empty"));
    }
    let base = idx1_base(source, tracks, movi, at, count)?;
    let mut buffer = vec![0u8; 16 * 4096];
    let mut done = 0u64;
    while done < count {
        let batch = (count - done).min(4096);
        let bytes = usize::try_from(batch * 16).map_err(|_| broken("idx1 batch"))?;
        source
            .seek(at + done * 16)
            .and_then(|()| source.read_exact(&mut buffer[..bytes]))
            .map_err(|error| broken(error.to_string()))?;
        let (rows, _) = buffer[..bytes].as_chunks::<16>();
        for (row, bytes) in (done..).zip(rows) {
            let entry = Idx1Row::parse(bytes)?;
            let Some(track) = entry.track(tracks) else {
                continue;
            };
            let header = base
                .checked_add(u64::from(entry.offset))
                .ok_or_else(|| broken("idx1 offset overflows"))?;
            let data = header + 8;
            if !movi.holds(data, u64::from(entry.size)) {
                return Err(broken(format!("idx1 entry {row} points outside movi")));
            }
            let track = &mut tracks[track];
            track.push(data, entry.size, entry.flags & AVIIF_KEYFRAME != 0, budget)?;
        }
        done += batch;
    }
    Ok(())
}

struct Idx1Row {
    id: [u8; 4],
    flags: u32,
    offset: u32,
    size: u32,
}

impl Idx1Row {
    fn parse(row: &[u8]) -> Result<Self, Fail> {
        let field = |at| u32_at(row, at).map_err(|_| broken("idx1 row"));
        Ok(Self {
            id: id_at(row, 0).map_err(|_| broken("idx1 row"))?,
            flags: field(4)?,
            offset: field(8)?,
            size: field(12)?,
        })
    }

    /// The track this row indexes, or `None` for a `rec ` list, a palette
    /// change, or a stream number the header does not have.
    fn track(&self, tracks: &[Track]) -> Option<usize> {
        if self.flags & (AVIIF_LIST | AVIIF_NO_TIME) != 0 || &self.id[2..] == b"pc" {
            return None;
        }
        let number = stream_of(self.id)?;
        let index = usize::try_from(number).ok()?;
        (index < tracks.len()).then_some(index)
    }
}

/// The value added to an `idx1` offset to reach the chunk header.
fn idx1_base<R: Read + Seek>(
    source: &mut Source<R>,
    tracks: &[Track],
    movi: Movi,
    at: u64,
    count: u64,
) -> Result<u64, Fail> {
    let mut first = None;
    for row in 0..count.min(64) {
        source
            .seek(at + row * 16)
            .map_err(|error| broken(error.to_string()))?;
        let bytes: [u8; 16] = source
            .read_array()
            .map_err(|error| broken(error.to_string()))?;
        let entry = Idx1Row::parse(&bytes)?;
        if entry.track(tracks).is_some() && entry.size > 0 {
            first = Some(entry);
            break;
        }
    }
    let Some(first) = first else {
        return Err(broken("idx1 names no stream chunk"));
    };
    let offset = u64::from(first.offset);
    for base in [movi.list, 0] {
        if let Some(header) = base.checked_add(offset)
            && chunk_matches(source, header, first.id, first.size)
        {
            return Ok(base);
        }
    }
    if let Some(found) = first_chunk(source, movi, first.id)
        && found >= offset
        && chunk_matches(source, found, first.id, first.size)
    {
        return Ok(found - offset);
    }
    Err(broken("idx1 offsets do not point at their chunks"))
}

fn chunk_matches<R: Read + Seek>(
    source: &mut Source<R>,
    header: u64,
    id: [u8; 4],
    size: u32,
) -> bool {
    if source.seek(header).is_err() {
        return false;
    }
    let Ok(head) = source.read_array::<8>() else {
        return false;
    };
    head[..4] == id && u32::from_le_bytes([head[4], head[5], head[6], head[7]]) >= size
}

/// The first chunk named `id` in `movi`, past `rec ` lists and `JUNK`.
fn first_chunk<R: Read + Seek>(source: &mut Source<R>, movi: Movi, id: [u8; 4]) -> Option<u64> {
    let mut at = movi.data();
    for _ in 0..64 {
        if at + 8 > movi.end {
            return None;
        }
        source.seek(at).ok()?;
        let head: [u8; 8] = source.read_array().ok()?;
        if head[..4] == id {
            return Some(at);
        }
        if &head[..4] == b"LIST" {
            at += 12;
            continue;
        }
        let size = u64::from(u32::from_le_bytes([head[4], head[5], head[6], head[7]]));
        at = at.checked_add(8 + size + (size & 1))?;
    }
    None
}

fn read_odml<R: Read + Seek>(
    source: &mut Source<R>,
    tracks: &mut [Track],
    layout: &Layout,
    limits: &Limits,
    budget: &mut u64,
) -> Result<(), Fail> {
    for track in tracks.iter_mut() {
        let Some(indx) = track.super_index.clone() else {
            return Err(broken(format!("stream {} has no indx", track.number)));
        };
        let header = IndexHeader::parse(&indx)?;
        if header.stream != Some(track.number) {
            return Err(broken("indx chunk id names another stream"));
        }
        match header.kind {
            0 => {
                if header.longs != 4 {
                    return Err(broken("super index entries are not four longs"));
                }
                let room = (indx.len() - 24) / 16;
                let count = usize::try_from(header.entries).unwrap_or(usize::MAX);
                if count > room || header.entries > MAX_SUPER_ENTRIES {
                    return Err(broken("super index has more entries than bytes"));
                }
                if count == 0 {
                    return Err(broken("super index lists no standard index"));
                }
                for row in 0..count {
                    let at = 24 + row * 16;
                    let offset = u64_at(&indx, at).map_err(|_| broken("super index row"))?;
                    let size = u32_at(&indx, at + 8).map_err(|_| broken("super index row"))?;
                    read_standard(source, track, layout, limits, offset, size, budget)?;
                }
            }
            1 => standard_entries(&indx, &header, track, layout, budget)?,
            _ => return Err(broken("indx has an unknown index type")),
        }
    }
    Ok(())
}

struct IndexHeader {
    longs: u16,
    sub_type: u8,
    kind: u8,
    entries: u32,
    stream: Option<u32>,
}

impl IndexHeader {
    fn parse(bytes: &[u8]) -> Result<Self, Fail> {
        if bytes.len() < 24 {
            return Err(broken("index header is short"));
        }
        let longs = u16_at(bytes, 0).map_err(|_| broken("index header"))?;
        let id = id_at(bytes, 8).map_err(|_| broken("index header"))?;
        Ok(Self {
            longs,
            sub_type: bytes[2],
            kind: bytes[3],
            entries: u32_at(bytes, 4).map_err(|_| broken("index header"))?,
            stream: stream_of(id),
        })
    }
}

fn read_standard<R: Read + Seek>(
    source: &mut Source<R>,
    track: &mut Track,
    layout: &Layout,
    limits: &Limits,
    offset: u64,
    size: u32,
    budget: &mut u64,
) -> Result<(), Fail> {
    source
        .seek(offset)
        .map_err(|error| broken(format!("ix chunk: {error}")))?;
    let head: [u8; 8] = source
        .read_array()
        .map_err(|error| broken(format!("ix chunk: {error}")))?;
    if index_stream_of([head[0], head[1], head[2], head[3]]) != Some(track.number) {
        return Err(broken(
            "super index entry does not point at this stream's ix chunk",
        ));
    }
    let declared = u64::from(u32::from_le_bytes([head[4], head[5], head[6], head[7]]));
    if u64::from(size) < declared.saturating_add(8) && size != 0 {
        return Err(broken("ix chunk is larger than its super index entry"));
    }
    let limit = u64::from(limits.max_index_entries)
        .saturating_mul(12)
        .saturating_add(24);
    let bytes = source
        .read_vec(declared, limit)
        .map_err(|error| match error {
            cutlass_container::SourceError::HostileSize { declared, limit } => {
                Fail::Fatal(OpenError::HostileSize { declared, limit })
            }
            other => broken(format!("ix chunk: {other}")),
        })?;
    let header = IndexHeader::parse(&bytes)?;
    if header.kind != 1 || header.stream != Some(track.number) {
        return Err(broken("ix chunk is not a standard index of this stream"));
    }
    standard_entries(&bytes, &header, track, layout, budget)
}

fn standard_entries(
    bytes: &[u8],
    header: &IndexHeader,
    track: &mut Track,
    layout: &Layout,
    budget: &mut u64,
) -> Result<(), Fail> {
    let width = match (header.sub_type, header.longs) {
        (0, 2) => 8,
        (1, 3) => 12,
        _ => return Err(broken("standard index entry size")),
    };
    let base = u64_at(bytes, 12).map_err(|_| broken("standard index base"))?;
    let room = (bytes.len() - 24) / width;
    let count = usize::try_from(header.entries).unwrap_or(usize::MAX);
    if count > room {
        return Err(broken("standard index has more entries than bytes"));
    }
    for row in 0..count {
        let at = 24 + row * width;
        let offset = u32_at(bytes, at).map_err(|_| broken("standard index row"))?;
        let raw = u32_at(bytes, at + 4).map_err(|_| broken("standard index row"))?;
        let size = raw & 0x7FFF_FFFF;
        let data = base
            .checked_add(u64::from(offset))
            .ok_or_else(|| broken("standard index offset overflows"))?;
        if size > 0
            && !layout
                .movi
                .iter()
                .any(|movi| movi.holds(data, u64::from(size)))
        {
            return Err(broken("standard index entry points outside movi"));
        }
        track.push(data, size, raw & 0x8000_0000 == 0, budget)?;
    }
    Ok(())
}

/// Rebuilds the tables from chunk headers. Returns the number of resyncs.
fn scan<R: Read + Seek>(
    source: &mut Source<R>,
    tracks: &mut [Track],
    spans: &[Movi],
    budget: &mut u64,
) -> Result<u32, OpenError> {
    let mut resyncs = 0;
    for movi in spans {
        let mut at = movi.data();
        while at + 8 <= movi.end {
            let Ok(head) = source.seek(at).and_then(|()| source.read_array::<8>()) else {
                break;
            };
            let id = [head[0], head[1], head[2], head[3]];
            let size = u32::from_le_bytes([head[4], head[5], head[6], head[7]]);
            let payload = at + 8;
            let end = payload.saturating_add(u64::from(size));
            if &id == b"LIST" && size >= 4 && end <= movi.end {
                at = payload + 4;
                continue;
            }
            if !plausible(id) || end > movi.end {
                match resync(source, at + 1, movi.end, tracks.len()) {
                    Some(next) => {
                        resyncs += 1;
                        at = next;
                        continue;
                    }
                    None => break,
                }
            }
            if let Some(number) = stream_of(id)
                && let Ok(index) = usize::try_from(number)
                && index < tracks.len()
                && &id[2..] != b"pc"
            {
                let track = &mut tracks[index];
                let key = if track.kind == StreamKind::Video && !track.intra && size > 0 {
                    let take = u64::from(size).min(SNIFF_BYTES);
                    let head = source
                        .read_vec(take, SNIFF_BYTES)
                        .map_err(OpenError::from_header)?;
                    sniff::keyframe(track.codec, &head)
                } else {
                    true
                };
                track.push(payload, size, key, budget)?;
            }
            at = end + (end & 1);
        }
    }
    Ok(resyncs)
}

/// Chunk ids a scan accepts: `NNxx` with a letter pair, `ix##`, `JUNK`, `LIST`.
fn plausible(id: [u8; 4]) -> bool {
    let printable = id
        .iter()
        .all(|byte| byte.is_ascii_alphanumeric() || *byte == b' ');
    printable
        && (stream_of(id).is_some()
            || index_stream_of(id).is_some()
            || matches!(&id, b"JUNK" | b"LIST" | b"idx1"))
}

fn resync<R: Read + Seek>(
    source: &mut Source<R>,
    from: u64,
    end: u64,
    streams: usize,
) -> Option<u64> {
    let stop = from.saturating_add(MAX_RESYNC).min(end);
    let mut window = vec![0u8; 64 * 1024];
    let mut at = from;
    while at + 8 <= stop {
        let room = usize::try_from((end - at).min(64 * 1024)).ok()?;
        source.seek(at).ok()?;
        source.read_exact(&mut window[..room]).ok()?;
        for (offset, candidate) in window[..room].windows(8).enumerate() {
            let here = at + u64::try_from(offset).ok()?;
            if here >= stop {
                return None;
            }
            let id = [candidate[0], candidate[1], candidate[2], candidate[3]];
            let size = u64::from(u32::from_le_bytes([
                candidate[4],
                candidate[5],
                candidate[6],
                candidate[7],
            ]));
            let fits = here + 8 + size <= end;
            let stream = stream_of(id)
                .and_then(|number| usize::try_from(number).ok())
                .is_some_and(|number| {
                    number < streams && id[2].is_ascii_alphabetic() && id[3].is_ascii_alphabetic()
                });
            if fits && (stream || matches!(&id, b"LIST" | b"JUNK")) {
                return Some(here);
            }
        }
        let advance = u64::try_from(room.saturating_sub(7)).ok()?.max(1);
        at += advance;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{index_stream_of, plausible, stream_of};

    #[test]
    fn chunk_ids_name_their_stream() {
        assert_eq!(stream_of(*b"00dc"), Some(0));
        assert_eq!(stream_of(*b"17wb"), Some(17));
        assert_eq!(stream_of(*b"ix00"), None);
        assert_eq!(index_stream_of(*b"ix03"), Some(3));
        assert_eq!(index_stream_of(*b"03ix"), Some(3));
        assert_eq!(index_stream_of(*b"JUNK"), None);
        assert!(plausible(*b"01wb") && plausible(*b"JUNK") && !plausible([0xFF, 0, 1, 2]));
    }
}

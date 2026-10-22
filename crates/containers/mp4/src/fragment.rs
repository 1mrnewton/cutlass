//! Samples that live in movie fragments (`moof` / `traf` / `trun`).

use std::num::NonZeroU32;

use cutlass_core::{Packet, Timestamp};

use crate::read::{self, Atom, Reader};
use crate::{Error, Fourcc};

const BASE_DATA_OFFSET: u32 = 0x0000_0001;
const SAMPLE_DESCRIPTION: u32 = 0x0000_0002;
const DEFAULT_DURATION: u32 = 0x0000_0008;
const DEFAULT_SIZE: u32 = 0x0000_0010;
const DEFAULT_FLAGS: u32 = 0x0000_0020;
const DURATION_IS_EMPTY: u32 = 0x0001_0000;
const DEFAULT_BASE_IS_MOOF: u32 = 0x0002_0000;

const TRUN_DATA_OFFSET: u32 = 0x0000_0001;
const TRUN_FIRST_FLAGS: u32 = 0x0000_0004;
const TRUN_DURATION: u32 = 0x0000_0100;
const TRUN_SIZE: u32 = 0x0000_0200;
const TRUN_FLAGS: u32 = 0x0000_0400;
const TRUN_COMPOSITION: u32 = 0x0000_0800;

/// ISO/IEC 14496-12 `sample_is_non_sync_sample`. Clear means a keyframe.
const NON_SYNC: u32 = 0x0001_0000;

#[derive(Clone, Copy)]
pub(crate) struct Trex {
    track_id: u32,
    description: u32,
    duration: u32,
    size: u32,
    flags: u32,
}

pub(crate) struct TrackRef<'a> {
    pub(crate) id: u32,
    pub(crate) timescale: NonZeroU32,
    pub(crate) samples: &'a mut Vec<Packet>,
}

struct Tfhd {
    track_id: u32,
    base_offset: Option<u64>,
    base_is_moof: bool,
    description: Option<u32>,
    duration: Option<u32>,
    size: Option<u32>,
    sample_flags: Option<u32>,
}

struct RunDefaults {
    duration: u32,
    size: u32,
    flags: u32,
}

struct Placement<'a> {
    file: &'a [u8],
    base: u64,
    cursor: u64,
    dts: u64,
}

/// Defaults from `mvex` / `trex`. A movie with no `mvex` has none.
pub(crate) fn defaults(mvex: Option<&Atom<'_>>) -> Result<Vec<Trex>, Error> {
    let Some(mvex) = mvex else {
        return Ok(Vec::new());
    };
    let children = read::atoms(mvex.payload, mvex.payload_base()?)?;
    let mut found = Vec::<Trex>::new();
    for atom in children.iter().filter(|atom| atom.kind == *b"trex") {
        let trex = one_trex(atom)?;
        if found.iter().any(|item| item.track_id == trex.track_id) {
            return Err(Error::Inconsistent("a track has two fragment defaults"));
        }
        found.push(trex);
    }
    Ok(found)
}

/// Append every track run in this fragment onto the matching track.
pub(crate) fn append(
    file: &[u8],
    moof: &Atom<'_>,
    tracks: &mut [TrackRef<'_>],
    defaults: &[Trex],
) -> Result<(), Error> {
    let children = read::atoms(moof.payload, moof.payload_base()?)?;
    for traf in children.iter().filter(|atom| atom.kind == *b"traf") {
        read_traf(file, moof.start(), traf, tracks, defaults)?;
    }
    Ok(())
}

fn read_traf(
    file: &[u8],
    moof_start: u64,
    traf: &Atom<'_>,
    tracks: &mut [TrackRef<'_>],
    defaults: &[Trex],
) -> Result<(), Error> {
    let children = read::atoms(traf.payload, traf.payload_base()?)?;
    let header = track_fragment_header(read::required(&children, *b"tfhd")?)?;
    let position = tracks
        .iter()
        .position(|track| track.id == header.track_id)
        .ok_or(Error::Inconsistent(
            "a fragment names a track the movie does not have",
        ))?;
    let trex = trex_for(defaults, header.track_id)?;
    let index = header.description.unwrap_or(trex.description);
    if index > 1 {
        return Err(Error::MultipleDescriptions);
    }
    let run_defaults = RunDefaults {
        duration: header.duration.unwrap_or(trex.duration),
        size: header.size.unwrap_or(trex.size),
        flags: header.sample_flags.unwrap_or(trex.flags),
    };
    let runs: Vec<&Atom<'_>> = children
        .iter()
        .filter(|atom| atom.kind == *b"trun")
        .collect();
    if runs.is_empty() {
        return Ok(());
    }
    // An explicit base wins. `default-base-is-moof` is the first byte of this
    // moof. Guessing a base when the file states neither mis-places the samples.
    let base = match (header.base_offset, header.base_is_moof) {
        (Some(offset), _) => offset,
        (None, true) => moof_start,
        (None, false) => {
            return Err(Error::Inconsistent("a fragment has no data base"));
        }
    };
    let dts = match read::only(&children, *b"tfdt")? {
        Some(atom) => decode_time(atom)?,
        None => next_decode(tracks[position].samples)?,
    };
    let mut place = Placement {
        file,
        base,
        cursor: base,
        dts,
    };
    let timescale = tracks[position].timescale;
    for run in runs {
        read_run(
            &mut place,
            run,
            &run_defaults,
            timescale,
            tracks[position].samples,
        )?;
    }
    Ok(())
}

fn read_run(
    place: &mut Placement<'_>,
    run: &Atom<'_>,
    defaults: &RunDefaults,
    timescale: NonZeroU32,
    samples: &mut Vec<Packet>,
) -> Result<(), Error> {
    let (version, flags, body) = read::full_box(run)?;
    if version > 1 {
        return Err(Error::BadVersion {
            kind: Fourcc::new(*b"trun"),
            version,
        });
    }
    let mut reader = Reader::new(body, run.payload_base()? + 4);
    let count = usize::try_from(reader.u32()?).map_err(|_| Error::TooLarge)?;
    // One packet per sample. A count past the file cannot name real samples,
    // and it would allocate a packet for every count.
    if count > place.file.len() {
        return Err(Error::Inconsistent(
            "a fragment has more samples than the file",
        ));
    }
    let data_offset = if flagged(flags, TRUN_DATA_OFFSET) {
        Some(reader.i32()?)
    } else {
        None
    };
    let first_flags = if flagged(flags, TRUN_FIRST_FLAGS) {
        Some(reader.u32()?)
    } else {
        None
    };
    let mut pos = match data_offset {
        Some(offset) => add_offset(place.base, offset)?,
        None => place.cursor,
    };
    for index in 0..count {
        let duration = if flagged(flags, TRUN_DURATION) {
            reader.u32()?
        } else {
            defaults.duration
        };
        let size = if flagged(flags, TRUN_SIZE) {
            reader.u32()?
        } else {
            defaults.size
        };
        let sample_flags = if flagged(flags, TRUN_FLAGS) {
            reader.u32()?
        } else if index == 0 {
            first_flags.unwrap_or(defaults.flags)
        } else {
            defaults.flags
        };
        let composition = if flagged(flags, TRUN_COMPOSITION) {
            if version == 1 {
                i64::from(reader.i32()?)
            } else {
                i64::from(reader.u32()?)
            }
        } else {
            0
        };
        let bytes = copy_sample(place.file, pos, size)?;
        let dts = i64::try_from(place.dts).map_err(|_| Error::TooLarge)?;
        let pts = dts.checked_add(composition).ok_or(Error::TooLarge)?;
        samples.push(Packet {
            data: bytes,
            pts: Timestamp::new(pts, timescale),
            dts: Some(Timestamp::new(dts, timescale)),
            duration: Some(Timestamp::new(i64::from(duration), timescale)),
            keyframe: !flagged(sample_flags, NON_SYNC),
        });
        pos = pos.checked_add(u64::from(size)).ok_or(Error::TooLarge)?;
        place.dts = place
            .dts
            .checked_add(u64::from(duration))
            .ok_or(Error::TooLarge)?;
    }
    place.cursor = pos;
    Ok(())
}

fn track_fragment_header(atom: &Atom<'_>) -> Result<Tfhd, Error> {
    let (version, flags, body) = read::full_box(atom)?;
    if version != 0 {
        return Err(Error::BadVersion {
            kind: Fourcc::new(*b"tfhd"),
            version,
        });
    }
    let mut reader = Reader::new(body, atom.payload_base()? + 4);
    let track_id = reader.u32()?;
    if track_id == 0 {
        return Err(Error::BadTrackId);
    }
    let base_offset = if flagged(flags, BASE_DATA_OFFSET) {
        Some(reader.u64()?)
    } else {
        None
    };
    let description = if flagged(flags, SAMPLE_DESCRIPTION) {
        Some(reader.u32()?)
    } else {
        None
    };
    let mut duration = if flagged(flags, DEFAULT_DURATION) {
        Some(reader.u32()?)
    } else {
        None
    };
    if flagged(flags, DURATION_IS_EMPTY) {
        duration = Some(0);
    }
    let size = if flagged(flags, DEFAULT_SIZE) {
        Some(reader.u32()?)
    } else {
        None
    };
    let sample_flags = if flagged(flags, DEFAULT_FLAGS) {
        Some(reader.u32()?)
    } else {
        None
    };
    Ok(Tfhd {
        track_id,
        base_offset,
        base_is_moof: flagged(flags, DEFAULT_BASE_IS_MOOF),
        description,
        duration,
        size,
        sample_flags,
    })
}

fn decode_time(atom: &Atom<'_>) -> Result<u64, Error> {
    let (version, _, body) = read::full_box(atom)?;
    let mut reader = Reader::new(body, atom.payload_base()? + 4);
    match version {
        0 => Ok(u64::from(reader.u32()?)),
        1 => reader.u64(),
        _ => Err(Error::BadVersion {
            kind: Fourcc::new(*b"tfdt"),
            version,
        }),
    }
}

fn one_trex(atom: &Atom<'_>) -> Result<Trex, Error> {
    let (version, _, body) = read::full_box(atom)?;
    if version != 0 {
        return Err(Error::BadVersion {
            kind: Fourcc::new(*b"trex"),
            version,
        });
    }
    let mut reader = Reader::new(body, atom.payload_base()? + 4);
    let track_id = reader.u32()?;
    if track_id == 0 {
        return Err(Error::BadTrackId);
    }
    Ok(Trex {
        track_id,
        description: reader.u32()?,
        duration: reader.u32()?,
        size: reader.u32()?,
        flags: reader.u32()?,
    })
}

fn trex_for(defaults: &[Trex], id: u32) -> Result<Trex, Error> {
    let mut found = None;
    for item in defaults {
        if item.track_id != id {
            continue;
        }
        if found.is_some() {
            return Err(Error::Inconsistent("a track has two fragment defaults"));
        }
        found = Some(*item);
    }
    Ok(found.unwrap_or(Trex {
        track_id: id,
        description: 1,
        duration: 0,
        size: 0,
        flags: 0,
    }))
}

fn next_decode(samples: &[Packet]) -> Result<u64, Error> {
    let Some(last) = samples.last() else {
        return Ok(0);
    };
    let start = last.dts.unwrap_or(last.pts).value;
    let duration = last.duration.map(|stamp| stamp.value).unwrap_or(0);
    let end = start.checked_add(duration).ok_or(Error::TooLarge)?;
    if end < 0 {
        return Ok(0);
    }
    u64::try_from(end).map_err(|_| Error::TooLarge)
}

fn add_offset(base: u64, offset: i32) -> Result<u64, Error> {
    let base = i64::try_from(base).map_err(|_| Error::TooLarge)?;
    let at = base.checked_add(i64::from(offset)).ok_or(Error::TooLarge)?;
    u64::try_from(at).map_err(|_| Error::Inconsistent("a sample extends past the end of the file"))
}

fn copy_sample(file: &[u8], pos: u64, size: u32) -> Result<Vec<u8>, Error> {
    let start = read::usize_from_u64(pos)?;
    let len = read::usize_from_u64(u64::from(size))?;
    let Some(end) = start.checked_add(len) else {
        return Err(Error::TooLarge);
    };
    file.get(start..end)
        .map(<[u8]>::to_vec)
        .ok_or(Error::Inconsistent(
            "a sample extends past the end of the file",
        ))
}

fn flagged(flags: u32, bit: u32) -> bool {
    (flags & bit) != 0
}

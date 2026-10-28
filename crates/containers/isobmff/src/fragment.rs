//! Movie fragments. A progressive sample table is not this path.
//!
//! `trun` data offsets are relative to the base in `tfhd`. Decode time starts
//! at `tfdt` when that box is present.

use crate::index::Sample;
use crate::legacy::Error;
use crate::limits::{MAX_PACKET_BYTES, MAX_SAMPLES};
use crate::movie::Trex;
use crate::parse::{self, Reader};

const TFHD_BASE_DATA_OFFSET: u32 = 0x01;
const TFHD_STSD_ID: u32 = 0x02;
const TFHD_DEFAULT_DURATION: u32 = 0x08;
const TFHD_DEFAULT_SIZE: u32 = 0x10;
const TFHD_DEFAULT_FLAGS: u32 = 0x20;
const TFHD_DEFAULT_BASE_IS_MOOF: u32 = 0x02_0000;

const TRUN_DATA_OFFSET: u32 = 0x01;
const TRUN_FIRST_FLAGS: u32 = 0x04;
const TRUN_DURATION: u32 = 0x100;
const TRUN_SIZE: u32 = 0x200;
const TRUN_FLAGS: u32 = 0x400;
const TRUN_CTS: u32 = 0x800;

const SAMPLE_NON_SYNC: u32 = 0x0001_0000;
const SAMPLE_DEPENDS_YES: u32 = 0x0100_0000;

#[derive(Debug)]
pub struct SidxRef {
    pub track_id: u32,
    pub time: u64,
    pub offset: u64,
}

pub struct Tfra {
    pub track_id: u32,
    pub time: u64,
    pub moof_offset: u64,
}

pub struct FragmentSamples {
    pub track_id: u32,
    pub samples: Vec<Sample>,
}

pub fn parse_sidx(payload: &[u8], after_box: u64) -> Result<Vec<SidxRef>, Error> {
    let mut reader = Reader::new(payload, 0);
    let version = reader.u8()?;
    reader.skip(3)?;
    let track_id = reader.u32()?;
    let _timescale = reader.u32()?;
    let (mut time, first_offset) = if version == 0 {
        (u64::from(reader.u32()?), u64::from(reader.u32()?))
    } else {
        (reader.u64()?, reader.u64()?)
    };
    reader.skip(2)?;
    let count = reader.u16()?;
    let mut offset = after_box
        .checked_add(first_offset)
        .ok_or(Error::HostileSize {
            at: after_box,
            claimed: first_offset,
        })?;
    let mut refs = Vec::new();
    for _ in 0..count {
        let word = reader.u32()?;
        let duration = reader.u32()?;
        let _sap = reader.u32()?;
        if word & 0x8000_0000 != 0 {
            return Err(Error::Unsupported {
                what: "sidx reference to another index",
            });
        }
        refs.push(SidxRef {
            track_id,
            time,
            offset,
        });
        let size = u64::from(word & 0x7fff_ffff);
        offset = offset.checked_add(size).ok_or(Error::HostileSize {
            at: offset,
            claimed: size,
        })?;
        time = time.saturating_add(u64::from(duration));
    }
    Ok(refs)
}

pub fn parse_mfra(payload: &[u8], budget: &mut u32) -> Result<Vec<Tfra>, Error> {
    let mut out = Vec::new();
    for atom in parse::atoms(payload, 0, budget)? {
        if atom.kind == *b"tfra" {
            out.extend(parse_tfra(atom.payload)?);
        }
    }
    Ok(out)
}

fn parse_tfra(payload: &[u8]) -> Result<Vec<Tfra>, Error> {
    let mut reader = Reader::new(payload, 0);
    let version = reader.u8()?;
    reader.skip(3)?;
    let track_id = reader.u32()?;
    let lengths = reader.u32()?;
    let traf_bytes = usize::try_from(((lengths >> 4) & 0x3) + 1).unwrap_or(1);
    let trun_bytes = usize::try_from(((lengths >> 2) & 0x3) + 1).unwrap_or(1);
    let sample_bytes = usize::try_from((lengths & 0x3) + 1).unwrap_or(1);
    let count = reader.u32()?;
    let mut rows = Vec::new();
    for _ in 0..count {
        let time = if version == 0 {
            u64::from(reader.u32()?)
        } else {
            reader.u64()?
        };
        let moof_offset = if version == 0 {
            u64::from(reader.u32()?)
        } else {
            reader.u64()?
        };
        reader.skip(traf_bytes)?;
        reader.skip(trun_bytes)?;
        reader.skip(sample_bytes)?;
        rows.push(Tfra {
            track_id,
            time,
            moof_offset,
        });
    }
    Ok(rows)
}

pub fn parse_moof(
    payload: &[u8],
    moof_offset: u64,
    trex: &[Trex],
    next_dts: &[(u32, i64)],
    budget: &mut u32,
) -> Result<Vec<FragmentSamples>, Error> {
    let mut out: Vec<FragmentSamples> = Vec::new();
    let mut implicit = moof_offset;
    for atom in parse::atoms(payload, 0, budget)? {
        if atom.kind != *b"traf" {
            continue;
        }
        let parsed = parse_traf(
            atom.payload,
            moof_offset,
            &mut implicit,
            trex,
            next_dts,
            budget,
        )?;
        if let Some(slot) = out.iter_mut().find(|row| row.track_id == parsed.track_id) {
            if u64::try_from(slot.samples.len()).unwrap_or(u64::MAX)
                + u64::try_from(parsed.samples.len()).unwrap_or(0)
                > MAX_SAMPLES
            {
                return Err(Error::Limit {
                    what: "samples",
                    value: MAX_SAMPLES,
                });
            }
            slot.samples.extend(parsed.samples);
        } else {
            out.push(FragmentSamples {
                track_id: parsed.track_id,
                samples: parsed.samples,
            });
        }
    }
    Ok(out)
}

struct Traf {
    track_id: u32,
    samples: Vec<Sample>,
}

fn parse_traf(
    payload: &[u8],
    moof_offset: u64,
    implicit: &mut u64,
    trex: &[Trex],
    next_dts: &[(u32, i64)],
    budget: &mut u32,
) -> Result<Traf, Error> {
    let children = parse::atoms(payload, 0, budget)?;
    let Some(tfhd) = children.iter().find(|atom| atom.kind == *b"tfhd") else {
        return Err(Error::Missing { what: "tfhd" });
    };
    let header = parse_tfhd(tfhd.payload, moof_offset, trex)?;
    let base_time = children
        .iter()
        .find(|atom| atom.kind == *b"tfdt")
        .map(|atom| parse_tfdt(atom.payload))
        .transpose()?
        .or_else(|| {
            next_dts
                .iter()
                .find(|(id, _)| *id == header.track_id)
                .map(|(_, dts)| *dts)
        })
        .unwrap_or(0);
    let mut dts = base_time;
    let mut samples = Vec::new();
    for atom in children.iter().filter(|atom| atom.kind == *b"trun") {
        let run = parse_trun(atom.payload, &header, implicit, &mut dts)?;
        samples.extend(run);
        if u64::try_from(samples.len()).unwrap_or(u64::MAX) > MAX_SAMPLES {
            return Err(Error::Limit {
                what: "samples",
                value: MAX_SAMPLES,
            });
        }
    }
    Ok(Traf {
        track_id: header.track_id,
        samples,
    })
}

struct Tfhd {
    track_id: u32,
    base: u64,
    duration: u32,
    size: u32,
    flags: u32,
    base_is_explicit: bool,
}

fn parse_tfhd(payload: &[u8], moof_offset: u64, trex: &[Trex]) -> Result<Tfhd, Error> {
    if payload.len() < 8 {
        return Err(Error::Truncated { at: 0, needed: 8 });
    }
    let flags = u32::from_be_bytes([0, payload[1], payload[2], payload[3]]);
    let mut reader = Reader::new(&payload[4..], 0);
    let track_id = reader.u32()?;
    let defaults = trex.iter().find(|row| row.track_id == track_id);
    let base = if flags & TFHD_BASE_DATA_OFFSET != 0 {
        reader.u64()?
    } else {
        moof_offset
    };
    if flags & TFHD_STSD_ID != 0 {
        reader.u32()?;
    }
    let duration = if flags & TFHD_DEFAULT_DURATION != 0 {
        reader.u32()?
    } else {
        defaults.map(|row| row.duration).unwrap_or(0)
    };
    let size = if flags & TFHD_DEFAULT_SIZE != 0 {
        reader.u32()?
    } else {
        defaults.map(|row| row.size).unwrap_or(0)
    };
    let sample_flags = if flags & TFHD_DEFAULT_FLAGS != 0 {
        reader.u32()?
    } else {
        defaults.map(|row| row.flags).unwrap_or(0)
    };
    Ok(Tfhd {
        track_id,
        base,
        duration,
        size,
        flags: sample_flags,
        base_is_explicit: flags & (TFHD_BASE_DATA_OFFSET | TFHD_DEFAULT_BASE_IS_MOOF) != 0,
    })
}

fn parse_tfdt(payload: &[u8]) -> Result<i64, Error> {
    let mut reader = Reader::new(payload, 0);
    let version = reader.u8()?;
    reader.skip(3)?;
    let time = if version == 0 {
        u64::from(reader.u32()?)
    } else {
        reader.u64()?
    };
    i64::try_from(time).map_err(|_| Error::Limit {
        what: "fragment time",
        value: time,
    })
}

fn parse_trun(
    payload: &[u8],
    header: &Tfhd,
    implicit: &mut u64,
    dts: &mut i64,
) -> Result<Vec<Sample>, Error> {
    if payload.len() < 8 {
        return Err(Error::Truncated { at: 0, needed: 8 });
    }
    let version = payload[0];
    let flags = u32::from_be_bytes([0, payload[1], payload[2], payload[3]]);
    let mut reader = Reader::new(&payload[4..], 0);
    let count = reader.u32()?;
    if u64::from(count) > MAX_SAMPLES {
        return Err(Error::Limit {
            what: "samples",
            value: u64::from(count),
        });
    }
    let data_offset = if flags & TRUN_DATA_OFFSET != 0 {
        Some(reader.i32()?)
    } else {
        None
    };
    let first_flags = if flags & TRUN_FIRST_FLAGS != 0 {
        reader.u32()?
    } else {
        header.flags
    };
    let mut cursor = if let Some(delta) = data_offset {
        add_delta(header.base, delta)?
    } else if header.base_is_explicit {
        header.base
    } else {
        *implicit
    };
    let mut samples = Vec::new();
    for index in 0..count {
        let duration = if flags & TRUN_DURATION != 0 {
            reader.u32()?
        } else {
            header.duration
        };
        let size = if flags & TRUN_SIZE != 0 {
            reader.u32()?
        } else {
            header.size
        };
        let sample_flags = if flags & TRUN_FLAGS != 0 {
            reader.u32()?
        } else if index == 0 {
            first_flags
        } else {
            header.flags
        };
        let cts = if flags & TRUN_CTS != 0 {
            if version == 0 {
                i64::from(reader.u32()?)
            } else {
                i64::from(reader.i32()?)
            }
        } else {
            0
        };
        if u64::from(size) > MAX_PACKET_BYTES {
            return Err(Error::Limit {
                what: "packet",
                value: u64::from(size),
            });
        }
        if size == 0 {
            return Err(Error::Unsupported {
                what: "empty fragment sample",
            });
        }
        let Some(pts) = dts.checked_add(cts) else {
            return Err(Error::Limit {
                what: "presentation time",
                value: u64::from(index),
            });
        };
        samples.push(Sample {
            offset: cursor,
            size,
            dts: *dts,
            pts,
            duration,
            keyframe: sample_flags & (SAMPLE_NON_SYNC | SAMPLE_DEPENDS_YES) == 0,
        });
        cursor = cursor
            .checked_add(u64::from(size))
            .ok_or(Error::HostileSize {
                at: cursor,
                claimed: u64::from(size),
            })?;
        *dts = dts.checked_add(i64::from(duration)).ok_or(Error::Limit {
            what: "decode time",
            value: u64::from(index),
        })?;
    }
    *implicit = cursor;
    Ok(samples)
}

fn add_delta(base: u64, delta: i32) -> Result<u64, Error> {
    if delta >= 0 {
        base.checked_add(u64::from(u32::try_from(delta).unwrap_or(0)))
            .ok_or(Error::HostileSize {
                at: base,
                claimed: u64::try_from(delta).unwrap_or(0),
            })
    } else {
        base.checked_sub(u64::from(delta.unsigned_abs()))
            .ok_or(Error::HostileSize {
                at: base,
                claimed: u64::from(delta.unsigned_abs()),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::parse_sidx;
    use crate::legacy::Error;

    #[test]
    fn sidx_version_0_locates_the_reference_and_rejects_a_nested_index() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&[0, 0, 0, 0]);
        payload.extend_from_slice(&1u32.to_be_bytes());
        payload.extend_from_slice(&1_000u32.to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&8u32.to_be_bytes());
        payload.extend_from_slice(&0u16.to_be_bytes());
        payload.extend_from_slice(&1u16.to_be_bytes());
        payload.extend_from_slice(&100u32.to_be_bytes());
        payload.extend_from_slice(&1_000u32.to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes());
        let refs = parse_sidx(&payload, 40).expect("sidx");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].track_id, 1);
        assert_eq!(refs[0].time, 0);
        assert_eq!(refs[0].offset, 48);

        payload[24] = 0x80;
        let err = parse_sidx(&payload, 40).expect_err("nested");
        assert_eq!(
            err,
            Error::Unsupported {
                what: "sidx reference to another index"
            }
        );
    }
}

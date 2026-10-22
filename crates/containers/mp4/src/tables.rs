//! Sample tables to packets, in one pass over the samples.

use std::num::NonZeroU32;

use cutlass_core::{Packet, Timestamp};

use crate::read::{self, Atom, Reader};
use crate::{Error, Fourcc};

pub(crate) fn packets(
    file: &[u8],
    timescale: NonZeroU32,
    stbl: &[Atom<'_>],
) -> Result<Vec<Packet>, Error> {
    let sizes = sizes(stbl)?;
    let count = sizes.count();
    let stts_atom = required(stbl, *b"stts")?;
    let (stts_version, _, stts_body) = read::full_box(stts_atom)?;
    expect_version(*b"stts", stts_version, &[0])?;
    let (_, stts_entries) = entries(stts_body, 8)?;
    if sum_counts(stts_entries)? != count {
        return Err(Error::Inconsistent(
            "decoding time entries do not cover every sample",
        ));
    }
    let ctts_atom = only(stbl, *b"ctts")?;
    let ctts = match ctts_atom {
        None => None,
        Some(atom) => {
            let (version, _, body) = read::full_box(atom)?;
            expect_version(*b"ctts", version, &[0, 1])?;
            let (_, ctts_entries) = entries(body, 8)?;
            if sum_counts(ctts_entries)? != count {
                return Err(Error::Inconsistent(
                    "composition offsets do not cover every sample",
                ));
            }
            // Version 0 is unsigned in ISO/IEC 14496-12. QuickTime still writes
            // a negative offset in that version, as a two's-complement value.
            Some((ctts_entries, true))
        }
    };
    let chunks = chunk_offsets(stbl)?;
    let map = chunk_map(stbl)?;
    let sync = sync_samples(stbl, count)?;
    let total = sizes.total()?;
    if total > read::u64_from_usize(file.len())? {
        return Err(Error::Inconsistent(
            "sample bytes overlap or exceed the file",
        ));
    }

    let mut stts = Runs::new(stts_entries, false);
    let mut ctts = ctts.map(|(entries, signed)| Runs::new(entries, signed));
    let mut sync_at = 0usize;
    let mut sample_index = 0u32;
    let mut dts = 0u64;
    let mut entry = 0usize;
    let mut packets = Vec::with_capacity(read::usize_from_u64(u64::from(count))?);

    for (chunk_index, chunk_offset) in chunks.iter().copied().enumerate() {
        let chunk_number = u32::try_from(chunk_index)
            .map_err(|_| Error::TooLarge)?
            .checked_add(1)
            .ok_or(Error::TooLarge)?;
        while entry + 1 < map.len() && map[entry + 1].first_chunk <= chunk_number {
            entry += 1;
        }
        let Some(mapping) = map.get(entry) else {
            return Err(Error::Inconsistent("a chunk has no sample-to-chunk entry"));
        };
        if mapping.first_chunk > chunk_number {
            return Err(Error::Inconsistent("a chunk has no sample-to-chunk entry"));
        }
        let end_index = sample_index
            .checked_add(mapping.samples_per_chunk)
            .ok_or(Error::TooLarge)?;
        if end_index > count {
            return Err(Error::Inconsistent(
                "chunks contain more samples than the sample size table",
            ));
        }
        let mut offset = chunk_offset;
        for _ in 0..mapping.samples_per_chunk {
            if mapping.description != 1 {
                return Err(Error::MultipleDescriptions);
            }
            let size = sizes.get(sample_index)?;
            let next = offset.checked_add(u64::from(size)).ok_or(Error::TooLarge)?;
            let start = read::usize_from_u64(offset)?;
            let end = read::usize_from_u64(next)?;
            let Some(bytes) = file.get(start..end) else {
                return Err(Error::Inconsistent(
                    "a sample extends past the end of the file",
                ));
            };
            let duration = u32::try_from(stts.next_value()?).map_err(|_| Error::TooLarge)?;
            let composition = match ctts.as_mut() {
                Some(runs) => runs.next_value()?,
                None => 0,
            };
            let dts_i64 = i64::try_from(dts).map_err(|_| Error::TooLarge)?;
            let pts = dts_i64.checked_add(composition).ok_or(Error::TooLarge)?;
            let sample_number = sample_index.checked_add(1).ok_or(Error::TooLarge)?;
            let keyframe = match &sync {
                None => true,
                Some(numbers) => {
                    let is_sync = numbers.get(sync_at) == Some(&sample_number);
                    if is_sync {
                        sync_at += 1;
                    }
                    is_sync
                }
            };
            packets.push(Packet {
                data: bytes.to_vec(),
                pts: Timestamp::new(pts, timescale),
                dts: Some(Timestamp::new(dts_i64, timescale)),
                duration: Some(Timestamp::new(i64::from(duration), timescale)),
                keyframe,
            });
            dts = dts
                .checked_add(u64::from(duration))
                .ok_or(Error::TooLarge)?;
            offset = next;
            sample_index = sample_number;
        }
    }

    if sample_index != count {
        return Err(Error::Inconsistent(
            "chunks contain fewer samples than the sample size table",
        ));
    }
    Ok(packets)
}

struct ChunkMap {
    first_chunk: u32,
    samples_per_chunk: u32,
    description: u32,
}

enum Sizes<'a> {
    Fixed { count: u32, size: u32 },
    Table { count: u32, bytes: &'a [u8] },
    Bytes { count: u32, bytes: &'a [u8] },
    Nibbles { count: u32, bytes: &'a [u8] },
    Words { count: u32, bytes: &'a [u8] },
}

impl Sizes<'_> {
    fn get(&self, index: u32) -> Result<u32, Error> {
        let index = read::usize_from_u64(u64::from(index))?;
        match self {
            Self::Fixed { size, .. } => Ok(*size),
            Self::Table { bytes, .. } => {
                let at = index.checked_mul(4).ok_or(Error::TooLarge)?;
                let Some(row) = bytes.get(at..at + 4) else {
                    return Err(Error::Inconsistent("sample size table is short"));
                };
                Ok(u32::from_be_bytes([row[0], row[1], row[2], row[3]]))
            }
            Self::Bytes { bytes, .. } => {
                let Some(byte) = bytes.get(index) else {
                    return Err(Error::Inconsistent("sample size table is short"));
                };
                Ok(u32::from(*byte))
            }
            // ISO/IEC 14496-12 packs the 4-bit fields big-endian, so the first
            // sample is the high nibble.
            Self::Nibbles { bytes, .. } => {
                let Some(byte) = bytes.get(index / 2) else {
                    return Err(Error::Inconsistent("sample size table is short"));
                };
                let nibble = if index % 2 == 0 {
                    byte >> 4
                } else {
                    byte & 0x0f
                };
                Ok(u32::from(nibble))
            }
            Self::Words { bytes, .. } => {
                let at = index.checked_mul(2).ok_or(Error::TooLarge)?;
                let Some(row) = bytes.get(at..at + 2) else {
                    return Err(Error::Inconsistent("sample size table is short"));
                };
                Ok(u32::from(u16::from_be_bytes([row[0], row[1]])))
            }
        }
    }

    fn total(&self) -> Result<u64, Error> {
        let mut total = 0u64;
        for index in 0..self.count() {
            total = total
                .checked_add(u64::from(self.get(index)?))
                .ok_or(Error::TooLarge)?;
        }
        Ok(total)
    }

    fn count(&self) -> u32 {
        match self {
            Self::Fixed { count, .. }
            | Self::Table { count, .. }
            | Self::Bytes { count, .. }
            | Self::Nibbles { count, .. }
            | Self::Words { count, .. } => *count,
        }
    }
}

fn sizes<'a>(stbl: &[Atom<'a>]) -> Result<Sizes<'a>, Error> {
    match (only(stbl, *b"stsz")?, only(stbl, *b"stz2")?) {
        (Some(_), Some(_)) => Err(Error::DuplicateBox(Fourcc::new(*b"stsz"))),
        (None, None) => Err(Error::MissingBox(Fourcc::new(*b"stsz"))),
        (None, Some(compact)) => compact_sizes(compact),
        (Some(stsz), None) => {
            let (version, _, body) = read::full_box(stsz)?;
            expect_version(*b"stsz", version, &[0])?;
            let mut reader = Reader::new(body, stsz.payload_base()?);
            let sample_size = reader.u32()?;
            let count = reader.u32()?;
            if sample_size == 0 {
                Ok(Sizes::Table {
                    count,
                    bytes: reader.rest(),
                })
            } else {
                Ok(Sizes::Fixed {
                    count,
                    size: sample_size,
                })
            }
        }
    }
}

fn compact_sizes<'a>(atom: &Atom<'a>) -> Result<Sizes<'a>, Error> {
    let (version, _, body) = read::full_box(atom)?;
    expect_version(*b"stz2", version, &[0])?;
    let mut reader = Reader::new(body, atom.payload_base()?);
    let field_size = u8::try_from(reader.u32()? & 0xff).map_err(|_| Error::TooLarge)?;
    let count = reader.u32()?;
    let bytes = reader.rest();
    match field_size {
        4 => Ok(Sizes::Nibbles { count, bytes }),
        8 => Ok(Sizes::Bytes { count, bytes }),
        16 => Ok(Sizes::Words { count, bytes }),
        _ => Err(Error::Inconsistent(
            "compact sample size field is not 4, 8, or 16",
        )),
    }
}

fn chunk_offsets(stbl: &[Atom<'_>]) -> Result<Vec<u64>, Error> {
    let stco = only(stbl, *b"stco")?;
    let co64 = only(stbl, *b"co64")?;
    match (stco, co64) {
        (Some(_), Some(_)) => Err(Error::DuplicateBox(Fourcc::new(*b"stco"))),
        (None, None) => Err(Error::MissingBox(Fourcc::new(*b"stco"))),
        (Some(atom), None) => {
            let (version, _, body) = read::full_box(atom)?;
            expect_version(*b"stco", version, &[0])?;
            let (_, entries) = entries(body, 4)?;
            let mut reader = Reader::new(entries, atom.payload_base()?);
            let mut offsets = Vec::new();
            while reader.remaining() > 0 {
                offsets.push(u64::from(reader.u32()?));
            }
            Ok(offsets)
        }
        (None, Some(atom)) => {
            let (version, _, body) = read::full_box(atom)?;
            expect_version(*b"co64", version, &[0])?;
            let (_, entries) = entries(body, 8)?;
            let mut reader = Reader::new(entries, atom.payload_base()?);
            let mut offsets = Vec::new();
            while reader.remaining() > 0 {
                offsets.push(reader.u64()?);
            }
            Ok(offsets)
        }
    }
}

fn chunk_map(stbl: &[Atom<'_>]) -> Result<Vec<ChunkMap>, Error> {
    let atom = required(stbl, *b"stsc")?;
    let (version, _, body) = read::full_box(atom)?;
    expect_version(*b"stsc", version, &[0])?;
    let (_, entries) = entries(body, 12)?;
    let mut reader = Reader::new(entries, atom.payload_base()?);
    let mut map = Vec::new();
    let mut previous = 0u32;
    while reader.remaining() > 0 {
        let first_chunk = reader.u32()?;
        let samples_per_chunk = reader.u32()?;
        let description = reader.u32()?;
        if first_chunk == 0 || first_chunk <= previous {
            return Err(Error::Inconsistent(
                "sample-to-chunk entries are not in chunk order",
            ));
        }
        previous = first_chunk;
        map.push(ChunkMap {
            first_chunk,
            samples_per_chunk,
            description,
        });
    }
    Ok(map)
}

fn sync_samples(stbl: &[Atom<'_>], count: u32) -> Result<Option<Vec<u32>>, Error> {
    let Some(atom) = only(stbl, *b"stss")? else {
        return Ok(None);
    };
    let (version, _, body) = read::full_box(atom)?;
    expect_version(*b"stss", version, &[0])?;
    let (_, entries) = entries(body, 4)?;
    let mut reader = Reader::new(entries, atom.payload_base()?);
    let mut numbers = Vec::new();
    let mut previous = 0u32;
    while reader.remaining() > 0 {
        let number = reader.u32()?;
        if number == 0 || number <= previous || number > count {
            return Err(Error::Inconsistent(
                "sync sample is out of order or out of range",
            ));
        }
        previous = number;
        numbers.push(number);
    }
    Ok(Some(numbers))
}

struct Runs<'a> {
    reader: Reader<'a>,
    left: u32,
    value: i64,
    signed: bool,
}

impl<'a> Runs<'a> {
    fn new(entries: &'a [u8], signed: bool) -> Self {
        Self {
            reader: Reader::new(entries, 0),
            left: 0,
            value: 0,
            signed,
        }
    }

    fn next_value(&mut self) -> Result<i64, Error> {
        if self.left == 0 {
            if self.reader.remaining() == 0 {
                return Err(Error::Inconsistent("a sample table ran out of entries"));
            }
            self.left = self.reader.u32()?;
            self.value = if self.signed {
                i64::from(self.reader.i32()?)
            } else {
                i64::from(self.reader.u32()?)
            };
            if self.left == 0 {
                return Err(Error::Inconsistent(
                    "a sample table entry covers no samples",
                ));
            }
        }
        self.left -= 1;
        Ok(self.value)
    }
}

fn entries(body: &[u8], entry_len: usize) -> Result<(u32, &[u8]), Error> {
    let mut reader = Reader::new(body, 0);
    let count = reader.u32()?;
    let bytes = read::usize_from_u64(
        u64::from(count)
            .checked_mul(read::u64_from_usize(entry_len)?)
            .ok_or(Error::TooLarge)?,
    )?;
    Ok((count, reader.take(bytes)?))
}

fn sum_counts(entries: &[u8]) -> Result<u32, Error> {
    let mut reader = Reader::new(entries, 0);
    let mut total = 0u32;
    while reader.remaining() > 0 {
        let count = reader.u32()?;
        reader.skip(4)?;
        total = total.checked_add(count).ok_or(Error::TooLarge)?;
    }
    Ok(total)
}

fn required<'payload, 'list>(
    atoms: &'list [Atom<'payload>],
    kind: [u8; 4],
) -> Result<&'list Atom<'payload>, Error> {
    read::required(atoms, kind)
}

fn only<'payload, 'list>(
    atoms: &'list [Atom<'payload>],
    kind: [u8; 4],
) -> Result<Option<&'list Atom<'payload>>, Error> {
    read::only(atoms, kind)
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

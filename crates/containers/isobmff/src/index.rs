//! Sample locations from `stts`, `ctts`, `stsc`, `stsz`/`stz2`, `stco`/`co64`, and `stss`.
//!
//! Offsets are computed once. A later seek binary-searches this table and does
//! not walk the media payload.
//!
//! Constant-size audio whose every table sample is one tick (raw PCM, as
//! QuickTime stores it) is not expanded to one row per audio frame. Its
//! chunks are cut into packets of about [`GROUP_FRAMES`] frames, the way
//! FFmpeg's `mov_build_index` groups them, and each packet is worked out from
//! the chunk tables when it is asked for.

use crate::legacy::Error;
use crate::limits::{MAX_PACKET_BYTES, MAX_SAMPLES};

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub offset: u64,
    pub size: u32,
    pub dts: i64,
    pub pts: i64,
    pub duration: u32,
    pub keyframe: bool,
}

#[derive(Clone, Debug)]
pub struct Run {
    pub count: u32,
    pub delta: u32,
}

#[derive(Clone, Debug)]
pub struct CtsRun {
    pub count: u32,
    pub offset: i64,
}

#[derive(Clone, Debug)]
pub struct ChunkMap {
    pub first_chunk: u32,
    pub samples_per_chunk: u32,
}

pub enum Sizes {
    Constant { size: u32, count: u64 },
    Table(Vec<u32>),
}

impl Sizes {
    #[must_use]
    pub fn count(&self) -> u64 {
        match self {
            Self::Constant { count, .. } => *count,
            Self::Table(rows) => u64::try_from(rows.len()).unwrap_or(u64::MAX),
        }
    }

    fn get(&self, index: u64) -> Result<u32, Error> {
        match self {
            Self::Constant { size, count } if index < *count => Ok(*size),
            Self::Table(rows) => {
                let Some(at) = usize::try_from(index).ok().and_then(|i| rows.get(i)) else {
                    return Err(Error::Missing {
                        what: "sample size",
                    });
                };
                Ok(*at)
            }
            Self::Constant { .. } => Err(Error::Missing {
                what: "sample size",
            }),
        }
    }
}

/// Absent `stss` means every sample is a sync sample. An empty `stss` means none are.
pub enum Sync {
    All,
    Only(Vec<u32>),
}

pub struct Tables {
    pub stts: Vec<Run>,
    pub ctts: Vec<CtsRun>,
    pub stsc: Vec<ChunkMap>,
    pub sizes: Sizes,
    pub chunks: Vec<u64>,
    pub sync: Sync,
    pub partial_sync: Vec<u32>,
}

/// Audio frames in one packet of grouped constant-size audio.
pub const GROUP_FRAMES: u32 = 1024;

/// The block grouped audio is cut on: `frames` audio frames in `bytes` bytes.
///
/// Raw PCM is one frame of `channels * bytes per sample`. A version-1 sound
/// description of a compressed codec names a larger block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Framing {
    pub frames: u32,
    pub bytes: u32,
}

/// A run of chunks that all hold `chunk_frames` audio frames.
#[derive(Clone, Debug)]
struct GroupRun {
    first_packet: u64,
    first_frame: u64,
    first_chunk: usize,
    chunk_frames: u32,
    packet_frames: u32,
    framing: Framing,
}

impl GroupRun {
    fn packets_per_chunk(&self) -> u64 {
        u64::from(self.chunk_frames.div_ceil(self.packet_frames.max(1)))
    }

    fn bytes_of(&self, frames: u64) -> u64 {
        (frames / u64::from(self.framing.frames.max(1)))
            .saturating_mul(u64::from(self.framing.bytes))
    }
}

/// Packets of constant-size audio, computed from the chunk tables.
///
/// Memory is the chunk offsets and one row per `stsc` row, whatever the
/// track's length. Every packet is a sync sample and `pts` equals `dts`.
#[derive(Clone, Debug)]
pub struct Grouped {
    runs: Vec<GroupRun>,
    chunks: Vec<u64>,
    packets: u64,
    frames: u64,
}

impl Grouped {
    /// Audio frames in the track: its length in ticks.
    #[must_use]
    pub const fn frames(&self) -> u64 {
        self.frames
    }

    #[must_use]
    pub fn len(&self) -> usize {
        usize::try_from(self.packets).unwrap_or(usize::MAX)
    }

    #[must_use]
    pub fn get(&self, index: usize) -> Option<Sample> {
        let index = u64::try_from(index).ok()?;
        if index >= self.packets {
            return None;
        }
        let at = self
            .runs
            .partition_point(|run| run.first_packet <= index)
            .checked_sub(1)?;
        let run = self.runs.get(at)?;
        let within = index - run.first_packet;
        let per_chunk = run.packets_per_chunk().max(1);
        let chunk = usize::try_from(within / per_chunk).ok()?;
        let before = (within % per_chunk) * u64::from(run.packet_frames);
        let frames = u64::from(run.packet_frames).min(u64::from(run.chunk_frames) - before);
        let chunk_offset = *self.chunks.get(run.first_chunk.checked_add(chunk)?)?;
        let chunk_index = u64::try_from(chunk).ok()?;
        let dts = run.first_frame + chunk_index * u64::from(run.chunk_frames) + before;
        let dts = i64::try_from(dts).ok()?;
        Some(Sample {
            offset: chunk_offset.saturating_add(run.bytes_of(before)),
            size: u32::try_from(run.bytes_of(frames)).ok()?,
            dts,
            pts: dts,
            duration: u32::try_from(frames).ok()?,
            keyframe: true,
        })
    }
}

/// The samples of one track: a row each, or grouped constant-size audio.
#[derive(Clone, Debug)]
pub enum Samples {
    Table(Vec<Sample>),
    Grouped(Grouped),
}

impl Default for Samples {
    fn default() -> Self {
        Self::Table(Vec::new())
    }
}

impl Samples {
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Table(rows) => rows.len(),
            Self::Grouped(grouped) => grouped.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn get(&self, index: usize) -> Option<Sample> {
        match self {
            Self::Table(rows) => rows.get(index).copied(),
            Self::Grouped(grouped) => grouped.get(index),
        }
    }

    #[must_use]
    pub fn first(&self) -> Option<Sample> {
        self.get(0)
    }

    #[must_use]
    pub fn last(&self) -> Option<Sample> {
        self.get(self.len().checked_sub(1)?)
    }

    /// The latest `pts + duration` of any sample.
    #[must_use]
    pub fn presentation_end(&self) -> Option<i64> {
        match self {
            Self::Table(rows) => rows
                .iter()
                .map(|sample| sample.pts.saturating_add(i64::from(sample.duration)))
                .max(),
            Self::Grouped(grouped) => {
                (grouped.packets > 0).then(|| i64::try_from(grouped.frames).unwrap_or(i64::MAX))
            }
        }
    }

    /// The rows as a table a fragment can replace or extend. Grouped audio
    /// is written out one row per packet, within [`MAX_SAMPLES`].
    pub fn table_mut(&mut self) -> Result<&mut Vec<Sample>, Error> {
        if let Self::Grouped(grouped) = self {
            if grouped.packets > MAX_SAMPLES {
                return Err(Error::Limit {
                    what: "samples",
                    value: grouped.packets,
                });
            }
            let rows = (0..grouped.len())
                .filter_map(|index| grouped.get(index))
                .collect();
            *self = Self::Table(rows);
        }
        match self {
            Self::Table(rows) => Ok(rows),
            Self::Grouped(_) => Err(Error::Missing {
                what: "sample table",
            }),
        }
    }
}

/// Whether `tables` describe audio that [`group`] can cut into packets:
/// one constant sample size, and every sample one tick long with no
/// composition offset. FFmpeg's `mov_build_index` asks for one `stts` row;
/// several rows that all say one tick are the same timeline.
#[must_use]
pub fn groupable(tables: &Tables) -> bool {
    matches!(tables.sizes, Sizes::Constant { size, count } if size > 0 && count > 0)
        && tables.ctts.is_empty()
        && !tables.stts.is_empty()
        && tables.stts.iter().all(|run| run.delta == 1)
}

/// Cuts constant-size audio into packets per chunk.
///
/// A packet never crosses a chunk. It holds [`GROUP_FRAMES`] frames, or the
/// largest whole number of `framing` blocks under that, or one block when a
/// block is 160 frames or more; the last packet of a chunk holds the rest.
pub fn group(tables: &Tables, framing: Framing) -> Result<Grouped, Error> {
    let count = tables.sizes.count();
    if tables.stsc.is_empty() || tables.chunks.is_empty() {
        return Err(Error::Missing {
            what: "sample table",
        });
    }
    if framing.frames == 0 || framing.bytes == 0 {
        return Err(Error::Unsupported {
            what: "audio frame of no bytes",
        });
    }
    let packet_frames = if framing.frames >= 160 {
        framing.frames
    } else {
        (GROUP_FRAMES / framing.frames) * framing.frames
    };
    let packet_bytes = u64::from(packet_frames / framing.frames) * u64::from(framing.bytes);
    if packet_bytes > MAX_PACKET_BYTES {
        return Err(Error::Limit {
            what: "packet",
            value: packet_bytes,
        });
    }
    let mut out = Grouped {
        runs: Vec::new(),
        chunks: tables.chunks.clone(),
        packets: 0,
        frames: 0,
    };
    let mut next_chunk = 0usize;
    for (at, row) in tables.stsc.iter().enumerate() {
        if out.frames >= count {
            break;
        }
        let start = usize::try_from(row.first_chunk.saturating_sub(1))
            .unwrap_or(usize::MAX)
            .max(next_chunk);
        let end = tables
            .stsc
            .get(at + 1)
            .map(|next| usize::try_from(next.first_chunk.saturating_sub(1)).unwrap_or(usize::MAX))
            .unwrap_or(usize::MAX)
            .min(tables.chunks.len());
        if end <= start {
            continue;
        }
        next_chunk = end;
        if row.samples_per_chunk == 0 {
            continue;
        }
        let per_chunk = u64::from(row.samples_per_chunk);
        let chunks = u64::try_from(end - start).unwrap_or(u64::MAX);
        let left = count - out.frames;
        let whole = chunks.min(left / per_chunk);
        let mut push = |first_chunk: usize, chunk_frames: u32, chunks: u64| {
            if !chunk_frames.is_multiple_of(framing.frames) {
                return Err(Error::Unsupported {
                    what: "audio chunk that splits a frame",
                });
            }
            let run = GroupRun {
                first_packet: out.packets,
                first_frame: out.frames,
                first_chunk,
                chunk_frames,
                packet_frames,
                framing,
            };
            out.packets = out
                .packets
                .saturating_add(run.packets_per_chunk().saturating_mul(chunks));
            out.frames = out
                .frames
                .saturating_add(u64::from(chunk_frames).saturating_mul(chunks));
            out.runs.push(run);
            Ok(())
        };
        if whole > 0 {
            push(start, row.samples_per_chunk, whole)?;
        }
        let rest = left - whole * per_chunk;
        if whole < chunks && rest > 0 {
            // The size table ends inside this chunk.
            let first = start.saturating_add(usize::try_from(whole).unwrap_or(usize::MAX));
            push(first, u32::try_from(rest).unwrap_or(u32::MAX), 1)?;
        }
    }
    if out.frames != count {
        return Err(Error::Missing {
            what: "chunk covers samples",
        });
    }
    Ok(out)
}

pub fn build(tables: &Tables) -> Result<Vec<Sample>, Error> {
    let count = tables.sizes.count();
    if count > MAX_SAMPLES {
        return Err(Error::Limit {
            what: "samples",
            value: count,
        });
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    if tables.stts.is_empty() || tables.stsc.is_empty() || tables.chunks.is_empty() {
        return Err(Error::Missing {
            what: "sample table",
        });
    }
    let mut samples = Vec::new();
    let mut dts: i64 = 0;
    let mut sample_index: u64 = 0;
    let mut stts_at = 0usize;
    let mut stts_left = tables.stts.first().map(|run| run.count).unwrap_or(0);
    let mut ctts_at = 0usize;
    let mut ctts_left = tables.ctts.first().map(|run| run.count).unwrap_or(0);
    let mut map_at = 0usize;

    for (chunk_index, offset0) in tables.chunks.iter().copied().enumerate() {
        let Some(chunk_no) = u32::try_from(chunk_index)
            .ok()
            .and_then(|i| i.checked_add(1))
        else {
            return Err(Error::Limit {
                what: "chunks",
                value: u64::try_from(chunk_index).unwrap_or(u64::MAX),
            });
        };
        let Some(first) = tables.stsc.first() else {
            return Err(Error::Missing {
                what: "sample table",
            });
        };
        if chunk_no < first.first_chunk {
            continue;
        }
        while map_at + 1 < tables.stsc.len()
            && tables
                .stsc
                .get(map_at + 1)
                .is_some_and(|row| row.first_chunk <= chunk_no)
        {
            map_at += 1;
        }
        let per = tables
            .stsc
            .get(map_at)
            .map(|row| row.samples_per_chunk)
            .unwrap_or(0);
        let mut offset = offset0;
        for _ in 0..per {
            if sample_index >= count {
                break;
            }
            let size = tables.sizes.get(sample_index)?;
            if u64::from(size) > MAX_PACKET_BYTES {
                return Err(Error::Limit {
                    what: "packet",
                    value: u64::from(size),
                });
            }
            let duration = next_delta(&tables.stts, &mut stts_at, &mut stts_left)?;
            let cts = next_cts(&tables.ctts, &mut ctts_at, &mut ctts_left);
            let Some(pts) = dts.checked_add(cts) else {
                return Err(Error::Limit {
                    what: "presentation time",
                    value: sample_index,
                });
            };
            samples.push(Sample {
                offset,
                size,
                dts,
                pts,
                duration,
                keyframe: is_sync(&tables.sync, &tables.partial_sync, sample_index),
            });
            offset = offset
                .checked_add(u64::from(size))
                .ok_or(Error::HostileSize {
                    at: offset,
                    claimed: u64::from(size),
                })?;
            dts = dts.checked_add(i64::from(duration)).ok_or(Error::Limit {
                what: "decode time",
                value: sample_index,
            })?;
            sample_index += 1;
        }
    }
    if sample_index != count {
        return Err(Error::Missing {
            what: "chunk covers samples",
        });
    }
    Ok(samples)
}

fn next_delta(runs: &[Run], at: &mut usize, left: &mut u32) -> Result<u32, Error> {
    if *left == 0 {
        *at += 1;
        let Some(run) = runs.get(*at) else {
            return Err(Error::Missing {
                what: "time to sample",
            });
        };
        *left = run.count;
    }
    if *left == 0 {
        return Err(Error::Missing {
            what: "time to sample",
        });
    }
    *left -= 1;
    runs.get(*at).map(|run| run.delta).ok_or(Error::Missing {
        what: "time to sample",
    })
}

fn next_cts(runs: &[CtsRun], at: &mut usize, left: &mut u32) -> i64 {
    if runs.is_empty() {
        return 0;
    }
    if *left == 0 {
        *at += 1;
        *left = runs.get(*at).map(|run| run.count).unwrap_or(0);
    }
    if *left == 0 {
        return 0;
    }
    *left -= 1;
    runs.get(*at).map(|run| run.offset).unwrap_or(0)
}

fn is_sync(sync: &Sync, partial: &[u32], index: u64) -> bool {
    let Ok(number) = u32::try_from(index) else {
        return false;
    };
    match sync {
        Sync::All => true,
        Sync::Only(list) => {
            list.binary_search(&number).is_ok() || partial.binary_search(&number).is_ok()
        }
    }
}

#[must_use]
pub fn no_sync_samples(sync: &Sync, partial: &[u32]) -> bool {
    matches!(sync, Sync::Only(list) if list.is_empty() && partial.is_empty())
}

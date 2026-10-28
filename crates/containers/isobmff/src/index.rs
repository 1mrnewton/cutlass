//! Sample locations from `stts`, `ctts`, `stsc`, `stsz`/`stz2`, `stco`/`co64`, and `stss`.
//!
//! Offsets are computed once. A later seek binary-searches this table and does
//! not walk the media payload.

use crate::limits::{MAX_PACKET_BYTES, MAX_SAMPLES};
use cutlass_container::Error;

#[derive(Clone, Debug)]
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
    fn count(&self) -> u64 {
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

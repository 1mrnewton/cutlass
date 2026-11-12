//! The `pakt` chunk and the packet layout it describes.
//!
//! `CAFPacketTableHeader` is 24 bytes: `mNumberPackets` and
//! `mNumberValidFrames` (`SInt64`), then `mPrimingFrames` and
//! `mRemainderFrames` (`SInt32`). The entries that follow are one varint
//! per packet (its byte size when `mBytesPerPacket` is zero, or its frame
//! count when `mFramesPerPacket` is zero), or two (size, then frames) when
//! both are zero.
//!
//! The table is decoded in bounded blocks. `mNumberPackets` is checked
//! against [`Limits::max_index_entries`] and against the entry bytes before
//! anything is allocated: every entry takes at least one byte per varint, so
//! a count larger than the bytes present is a lie. The decoded table keeps
//! four bytes per packet for each varying field, plus one checkpoint
//! (offset and time) every [`CHECKPOINT_EVERY`] packets for seeking.

use std::io::Read;

use cutlass_container::{Input, Limits, OpenError};

use crate::bytes::{be_i32, be_i64, len_u64};
use crate::error::{hostile, malformed, truncated};
use crate::limits::{CHECKPOINT_EVERY, MAX_FRAMES_PER_PACKET, PAKT_BLOCK};
use crate::varint::{Decoder, VarintError};

/// Bytes of `CAFPacketTableHeader` before the entries.
pub const PAKT_HEADER_BYTES: u64 = 24;

/// The four header fields, checked to be non-negative.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PacketTableHeader {
    /// `mNumberPackets`. Zero for a constant format that only records frames.
    pub packets: u64,
    /// `mNumberValidFrames`.
    pub valid_frames: u64,
    /// `mPrimingFrames`.
    pub priming_frames: u32,
    /// `mRemainderFrames`.
    pub remainder_frames: u32,
}

impl PacketTableHeader {
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, OpenError> {
        let short = || truncated(PAKT_HEADER_BYTES, len_u64(bytes.len()));
        let packets = be_i64(bytes, 0).ok_or_else(short)?;
        let valid = be_i64(bytes, 8).ok_or_else(short)?;
        let priming = be_i32(bytes, 16).ok_or_else(short)?;
        let remainder = be_i32(bytes, 20).ok_or_else(short)?;
        let non_negative = |value: i64, what: &str| {
            u64::try_from(value).map_err(|_| malformed(format!("pakt {what} is negative")))
        };
        Ok(Self {
            packets: non_negative(packets, "packet count")?,
            valid_frames: non_negative(valid, "valid frame count")?,
            priming_frames: u32::try_from(priming)
                .map_err(|_| malformed("pakt priming frames are negative"))?,
            remainder_frames: u32::try_from(remainder)
                .map_err(|_| malformed("pakt remainder frames are negative"))?,
        })
    }

    #[must_use]
    pub fn to_bytes(&self) -> [u8; 24] {
        let mut out = [0_u8; 24];
        out[0..8].copy_from_slice(&self.packets.min(i64::MAX as u64).to_be_bytes());
        out[8..16].copy_from_slice(&self.valid_frames.min(i64::MAX as u64).to_be_bytes());
        out[16..20].copy_from_slice(&self.priming_frames.min(i32::MAX as u32).to_be_bytes());
        out[20..24].copy_from_slice(&self.remainder_frames.min(i32::MAX as u32).to_be_bytes());
        out
    }
}

/// One checkpoint: where packet `k * CHECKPOINT_EVERY` starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Checkpoint {
    offset: u64,
    time: u64,
}

/// Packet sizes and durations decoded from a table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Table {
    count: u64,
    /// Bytes of each packet, or empty when `constant_bytes` is set.
    sizes: Vec<u32>,
    /// Frames of each packet, or empty when `constant_frames` is set.
    frames: Vec<u32>,
    constant_bytes: u32,
    constant_frames: u32,
    checkpoints: Vec<Checkpoint>,
    total_bytes: u64,
    total_frames: u64,
}

/// Where a packet starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Located {
    pub index: u64,
    pub offset: u64,
    pub time: u64,
    /// Table entries or checkpoints looked at.
    pub visited: u32,
}

/// How the packets in `data` are laid out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Packets {
    /// Every packet has `bytes` bytes and `frames` frames.
    Constant { bytes: u32, frames: u32, count: u64 },
    /// Sizes, durations, or both come from the packet table.
    Table(Table),
}

impl Packets {
    pub(crate) fn count(&self) -> u64 {
        match self {
            Self::Constant { count, .. } => *count,
            Self::Table(table) => table.count,
        }
    }

    pub(crate) fn total_frames(&self) -> u64 {
        match self {
            Self::Constant { frames, count, .. } => u64::from(*frames).saturating_mul(*count),
            Self::Table(table) => table.total_frames,
        }
    }

    pub(crate) fn total_bytes(&self) -> u64 {
        match self {
            Self::Constant { bytes, count, .. } => u64::from(*bytes).saturating_mul(*count),
            Self::Table(table) => table.total_bytes,
        }
    }

    /// Bytes and frames of packet `index`. The caller keeps `index` below
    /// [`Self::count`].
    pub(crate) fn entry(&self, index: u64) -> (u32, u32) {
        match self {
            Self::Constant { bytes, frames, .. } => (*bytes, *frames),
            Self::Table(table) => table.entry(index),
        }
    }

    /// The packet that holds frame `time`, which is below the total.
    pub(crate) fn locate(&self, time: u64) -> Located {
        match self {
            Self::Constant { bytes, frames, .. } => {
                let index = time / u64::from((*frames).max(1));
                Located {
                    index,
                    offset: index.saturating_mul(u64::from(*bytes)),
                    time: index.saturating_mul(u64::from(*frames)),
                    visited: 0,
                }
            }
            Self::Table(table) => table.locate(time),
        }
    }
}

impl Table {
    fn entry(&self, index: u64) -> (u32, u32) {
        let at = usize::try_from(index).unwrap_or(usize::MAX);
        let bytes = if self.sizes.is_empty() {
            self.constant_bytes
        } else {
            self.sizes.get(at).copied().unwrap_or(0)
        };
        let frames = if self.frames.is_empty() {
            self.constant_frames
        } else {
            self.frames.get(at).copied().unwrap_or(0)
        };
        (bytes, frames)
    }

    fn locate(&self, time: u64) -> Located {
        let mut visited = 0_u32;
        let checkpoint = if self.frames.is_empty() {
            let index = time / u64::from(self.constant_frames.max(1));
            index / CHECKPOINT_EVERY
        } else {
            // The last checkpoint at or before `time`.
            let mut lo = 0_usize;
            let mut hi = self.checkpoints.len();
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                visited = visited.saturating_add(1);
                if self.checkpoints[mid].time <= time {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            len_u64(lo)
        };
        let at = usize::try_from(checkpoint).unwrap_or(usize::MAX);
        let start = self
            .checkpoints
            .get(at)
            .copied()
            .unwrap_or(Checkpoint { offset: 0, time: 0 });
        let mut located = Located {
            index: checkpoint.saturating_mul(CHECKPOINT_EVERY),
            offset: start.offset,
            time: start.time,
            visited,
        };
        while located.index + 1 < self.count {
            let (bytes, frames) = self.entry(located.index);
            let next = located.time.saturating_add(u64::from(frames));
            if next > time {
                break;
            }
            located.visited = located.visited.saturating_add(1);
            located.index += 1;
            located.offset = located.offset.saturating_add(u64::from(bytes));
            located.time = next;
        }
        located
    }

    /// Reads the entries that follow a packet table header.
    ///
    /// `entry_bytes` is the chunk size less the header. `constant_bytes`
    /// and `constant_frames` are the `desc` fields; a zero one is read from
    /// the table.
    pub(crate) fn read<R: Read>(
        input: &mut Input<R>,
        header: &PacketTableHeader,
        entry_bytes: u64,
        constant_bytes: u32,
        constant_frames: u32,
        limits: &Limits,
    ) -> Result<Self, OpenError> {
        let count = header.packets;
        let per_packet = u64::from(constant_bytes == 0) + u64::from(constant_frames == 0);
        if count > u64::from(limits.max_index_entries) {
            return Err(hostile(count, u64::from(limits.max_index_entries)));
        }
        let need = count.saturating_mul(per_packet);
        if need > entry_bytes {
            return Err(hostile(need, entry_bytes));
        }
        let mut sizes = Vec::new();
        let mut frames = Vec::new();
        let len = usize::try_from(count)
            .map_err(|_| hostile(count, u64::from(limits.max_index_entries)))?;
        if constant_bytes == 0 {
            reserve(&mut sizes, len, count)?;
        }
        if constant_frames == 0 {
            reserve(&mut frames, len, count)?;
        }
        let checkpoint_count = count.div_ceil(CHECKPOINT_EVERY);
        let mut checkpoints = Vec::new();
        reserve(
            &mut checkpoints,
            usize::try_from(checkpoint_count).unwrap_or(usize::MAX),
            checkpoint_count,
        )?;
        let mut table = Self {
            count,
            sizes,
            frames,
            constant_bytes,
            constant_frames,
            checkpoints,
            total_bytes: 0,
            total_frames: 0,
        };
        table.decode(input, entry_bytes, per_packet, limits)?;
        Ok(table)
    }

    fn decode<R: Read>(
        &mut self,
        input: &mut Input<R>,
        entry_bytes: u64,
        per_packet: u64,
        limits: &Limits,
    ) -> Result<(), OpenError> {
        let mut left = entry_bytes;
        let mut block =
            vec![0_u8; PAKT_BLOCK.min(usize::try_from(entry_bytes).unwrap_or(PAKT_BLOCK))];
        let mut decoder = Decoder::default();
        let mut pending: Option<u32> = None;
        let mut done = 0_u64;
        while done < self.count {
            if left == 0 {
                return Err(truncated(entry_bytes, entry_bytes));
            }
            let take = usize::try_from(left.min(len_u64(block.len()))).unwrap_or(block.len());
            let chunk = &mut block[..take];
            input.read_exact(chunk).map_err(OpenError::from_header)?;
            left -= len_u64(take);
            for byte in chunk.iter() {
                let Some(value) = decoder.push(*byte).map_err(varint_error)? else {
                    continue;
                };
                if per_packet == 2 && pending.is_none() {
                    pending = Some(value);
                    continue;
                }
                let (bytes, frames) = match (self.constant_bytes, self.constant_frames) {
                    (0, 0) => (pending.take().unwrap_or(0), value),
                    (0, frames) => (value, frames),
                    (bytes, _) => (bytes, value),
                };
                self.push(bytes, frames, limits)?;
                done += 1;
                if done == self.count {
                    break;
                }
            }
        }
        // The bytes after the last entry are not part of the table. They are
        // read past here so the input sits at the end of the chunk.
        if left > 0 {
            skip(input, left)?;
        }
        Ok(())
    }

    fn push(&mut self, bytes: u32, frames: u32, limits: &Limits) -> Result<(), OpenError> {
        if u64::from(bytes) > limits.max_packet_bytes {
            return Err(hostile(u64::from(bytes), limits.max_packet_bytes));
        }
        if frames == 0 {
            return Err(malformed("a pakt entry has no frames"));
        }
        if frames > MAX_FRAMES_PER_PACKET {
            return Err(hostile(u64::from(frames), u64::from(MAX_FRAMES_PER_PACKET)));
        }
        let seen = self.seen();
        if seen.is_multiple_of(CHECKPOINT_EVERY) {
            self.checkpoints.push(Checkpoint {
                offset: self.total_bytes,
                time: self.total_frames,
            });
        }
        if self.constant_bytes == 0 {
            self.sizes.push(bytes);
        }
        if self.constant_frames == 0 {
            self.frames.push(frames);
        }
        self.total_bytes = self
            .total_bytes
            .checked_add(u64::from(bytes))
            .ok_or_else(|| malformed("pakt byte total overflows"))?;
        self.total_frames = self
            .total_frames
            .checked_add(u64::from(frames))
            .filter(|total| i64::try_from(*total).is_ok())
            .ok_or_else(|| malformed("pakt frame total overflows"))?;
        Ok(())
    }

    /// Entries pushed so far.
    fn seen(&self) -> u64 {
        len_u64(self.sizes.len().max(self.frames.len()))
    }
}

fn reserve<T>(vec: &mut Vec<T>, len: usize, declared: u64) -> Result<(), OpenError> {
    vec.try_reserve_exact(len)
        .map_err(|_| hostile(declared, declared.saturating_sub(1)))
}

fn varint_error(error: VarintError) -> OpenError {
    match error {
        VarintError::TooLong => malformed("pakt entry is longer than five bytes"),
        VarintError::TooLarge => malformed("pakt entry does not fit in 32 bits"),
    }
}

/// Reads and drops `len` bytes.
pub(crate) fn skip<R: Read>(input: &mut Input<R>, mut len: u64) -> Result<(), OpenError> {
    let mut scratch = [0_u8; 8192];
    while len > 0 {
        let take = usize::try_from(len.min(8192)).unwrap_or(8192);
        input
            .read_exact(&mut scratch[..take])
            .map_err(OpenError::from_header)?;
        len -= len_u64(take);
    }
    Ok(())
}

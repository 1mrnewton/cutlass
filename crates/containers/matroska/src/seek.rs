//! Seeking by Cues, or by Cluster headers when a file has none.
//!
//! A seek picks the last cue point of the track at or before the target
//! (any track's cue points when the track has none of its own; the Cluster
//! headers when the file has no Cues) and walks block headers from that
//! Cluster. The landing is the last keyframe of the track at or before the
//! target. No frame byte is inspected on the walk: a block's track, time
//! and flags are in its first few octets, and the rest is skipped, inside
//! the read-ahead buffer when it is there and with a seek when not.
//! When the walk finds no keyframe at or before the target, it starts again
//! one cue point earlier, a bounded number of times.
//!
//! The landing's `ordinal` is the block's place in its Cluster, counting
//! from one: Matroska keeps no per-track frame number.
//!
//! Each walk keeps the block headers it passed, by the Cluster it started
//! from, up to [`MAX_WALKED_BLOCKS`] in all. A later walk from the same
//! Cluster goes over the kept headers first and reads on from where the
//! last walk stopped only when it needs more, so seeking back and forth
//! over the same stretch reads nothing again. The file does not change
//! under the demuxer, so a kept header stays true.

use std::collections::HashMap;

use std::io::{Read, Seek};

use cutlass_container::{Landing, SeekError, SeekLanding, SeekTo};

use crate::block::{self, KEYFRAME};
use crate::demux::{Cluster, CueState, Cursor, MatroskaDemux, Starts};
use crate::ebml::{self, EbmlError, Size};
use crate::error::{Error, Result};
use crate::ids;
use crate::limits::{MAX_CLUSTER_SCAN, MAX_CUES_BYTES, MAX_SEEK_WALK, MAX_WALKED_BLOCKS};
use crate::meta;
use crate::source::Header;

/// Cue points a seek steps back over before it settles for the first
/// keyframe after the target.
const MAX_STEP_BACK: usize = 16;

/// Elements read inside one Cluster looking for its Timestamp while the
/// Cluster index is built.
const MAX_CHILDREN_BEFORE_TIMESTAMP: u32 = 64;

/// A block of the seek track, seen on the walk.
#[derive(Clone, Copy, Debug)]
struct Found {
    time: i64,
    keyframe: bool,
    /// Offset of the block element's ID.
    position: u64,
    /// The Cluster as the reader would have it just before this block.
    cluster: Cluster,
}

#[derive(Default)]
struct Walk {
    best: Option<Found>,
    after: Option<Found>,
    first: Option<Found>,
    preroll: u32,
    visited: u32,
}

impl Walk {
    /// Takes in one block of the seek track. True when the walk is done.
    fn see(&mut self, found: Found, frames: u32, target: i64) -> bool {
        if self.first.is_none() {
            self.first = Some(found);
        }
        if found.time > target {
            if self.best.is_some() {
                return true;
            }
            if found.keyframe {
                self.after = Some(found);
                return true;
            }
            return false;
        }
        if found.keyframe {
            self.best = Some(found);
            self.preroll = 0;
        }
        if self.best.is_some() && found.time < target {
            self.preroll = self.preroll.saturating_add(frames);
        }
        false
    }
}

/// One block header a walk passed.
#[derive(Clone, Copy, Debug)]
struct Passed {
    track: u64,
    /// In packet ticks; `None` when the Cluster had no Timestamp before it
    /// or the time does not convert.
    time: Option<i64>,
    keyframe: bool,
    frames: u32,
    position: u64,
    cluster: Cluster,
}

/// Where a kept walk stopped.
#[derive(Clone, Copy, Debug)]
enum Resume {
    /// Reading on starts here, inside `cluster` or between Clusters.
    At {
        position: u64,
        cluster: Option<Cluster>,
    },
    /// The walk ran into the end of the Segment or the input.
    End,
}

/// The block headers walks from one Cluster passed, in file order.
#[derive(Debug)]
struct Trail {
    blocks: Vec<Passed>,
    resume: Resume,
}

/// Walks kept between seeks, by the Cluster offset each started from.
#[derive(Debug, Default)]
pub(crate) struct Walked {
    trails: HashMap<u64, Trail>,
    blocks: usize,
}

impl Walked {
    /// Room for one more block in `trail`, the walk in hand, which is not
    /// counted in `blocks`. When the total is reached, every other kept
    /// walk is dropped.
    fn room(&mut self, trail: &Trail) -> bool {
        if self.blocks.saturating_add(trail.blocks.len()) < MAX_WALKED_BLOCKS {
            return true;
        }
        self.trails.clear();
        self.blocks = 0;
        trail.blocks.len() < MAX_WALKED_BLOCKS
    }
}

impl<R: Read + Seek> MatroskaDemux<R> {
    pub(crate) fn seek_to(&mut self, to: SeekTo) -> Result<SeekLanding> {
        if !self.source.random() {
            return Err(Error::Seek(SeekError::ForwardOnly));
        }
        let number = u64::from(to.track);
        if !self
            .track_entries()
            .iter()
            .any(|entry| entry.number == number)
        {
            return Err(Error::Seek(SeekError::UnknownTrack));
        }
        let past_end = to.end.is_some_and(|end| to.time >= end)
            || (to.end.is_none()
                && self
                    .duration_ticks()
                    .is_some_and(|duration| duration > 0 && to.time >= duration));
        if past_end {
            self.cursor = Cursor {
                finished: true,
                ..Cursor::default()
            };
            return Ok(SeekLanding {
                landing: Landing::End,
                index_used: false,
                payload_scans: 0,
                entries_visited: 0,
            });
        }
        let saved = (self.cursor.clone(), self.source.position());
        match self.seek_walk(number, to.time) {
            Ok(landing) => Ok(landing),
            Err(error) => {
                let (cursor, position) = saved;
                self.cursor = cursor;
                let _ = self.source.seek(position);
                Err(error)
            }
        }
    }

    fn seek_walk(&mut self, number: u64, target: i64) -> Result<SeekLanding> {
        let starts = match self.starts.take() {
            Some(starts) if starts.track == number => starts,
            _ => {
                let (points, from_cues) = self.starts(number)?;
                Starts {
                    track: number,
                    points,
                    from_cues,
                }
            }
        };
        let landing = self.walk_starts(&starts.points, starts.from_cues, number, target);
        self.starts = Some(starts);
        landing
    }

    fn walk_starts(
        &mut self,
        starts: &[(i64, u64)],
        index_used: bool,
        number: u64,
        target: i64,
    ) -> Result<SeekLanding> {
        if starts.is_empty() {
            return Err(Error::Seek(SeekError::Empty));
        }
        // The last start at or before the target, else the first.
        let mut steps = 0_u32;
        let mut at = starts.partition_point(|(time, _)| {
            steps = steps.saturating_add(1);
            *time <= target
        });
        at = at.saturating_sub(1);
        let mut visited = steps;
        let mut tries = 0;
        let mut fallback: Option<Found> = None;
        loop {
            let walk = self.walk_from(starts[at].1, number, target)?;
            visited = visited.saturating_add(walk.visited);
            if let Some(best) = walk.best {
                return Ok(self.land(best, walk.preroll, target, index_used, visited));
            }
            fallback = fallback.or(walk.after).or(walk.first);
            tries += 1;
            if at == 0 || tries > MAX_STEP_BACK {
                break;
            }
            at -= 1;
        }
        match fallback {
            Some(found) => Ok(self.land(found, 0, target, index_used, visited)),
            None => Err(Error::Seek(SeekError::Empty)),
        }
    }

    fn land(
        &mut self,
        found: Found,
        preroll: u32,
        target: i64,
        index_used: bool,
        visited: u32,
    ) -> SeekLanding {
        let ordinal = found.cluster.blocks.saturating_add(1);
        self.cursor = Cursor {
            cluster: Some(found.cluster),
            ..Cursor::default()
        };
        // The walk read this far, so the offset is inside the input.
        let _ = self.source.seek(found.position);
        SeekLanding {
            landing: Landing::Packet {
                ordinal,
                time: found.time,
                keyframe: found.keyframe,
                preroll: if found.time >= target { 0 } else { preroll },
                byte_position: found.position,
            },
            index_used,
            payload_scans: 0,
            entries_visited: visited,
        }
    }

    /// Where walks may start: (time in packet ticks, Cluster offset), sorted
    /// by time, and whether they came from Cues.
    fn starts(&mut self, number: u64) -> Result<(Vec<(i64, u64)>, bool)> {
        self.load_cues();
        if let CueState::Loaded(cues) = &self.cues {
            let own: Vec<&meta::Cue> = cues.iter().filter(|cue| cue.track == number).collect();
            let chosen: Vec<&meta::Cue> = if own.is_empty() {
                cues.iter().collect()
            } else {
                own
            };
            let mut starts = Vec::with_capacity(chosen.len());
            for cue in chosen {
                let Some(time) = i64::try_from(cue.time)
                    .ok()
                    .and_then(|time| self.clock.from_segment(time))
                else {
                    continue;
                };
                let Some(position) = self.segment_data.checked_add(cue.cluster) else {
                    continue;
                };
                if self
                    .source
                    .length()
                    .is_some_and(|length| position >= length)
                {
                    continue;
                }
                starts.push((time, position));
            }
            starts.sort_unstable();
            starts.dedup();
            if !starts.is_empty() {
                return Ok((starts, true));
            }
        }
        let clusters = self.cluster_index()?;
        let mut starts = Vec::with_capacity(clusters.len());
        for (time, position) in clusters {
            if let Some(time) = self.clock.from_segment(time) {
                starts.push((time, position));
            }
        }
        // Cluster timestamps only go up in a valid file; sort in case.
        starts.sort_unstable();
        Ok((starts, false))
    }

    /// Reads the Cues the SeekHead points at, once. A Cues element that does
    /// not parse is the same as none: seek falls back to Cluster headers.
    pub(crate) fn load_cues(&mut self) {
        if !matches!(self.cues, CueState::Unread) {
            return;
        }
        self.cues = match self.read_cues() {
            Ok(Some(cues)) if !cues.is_empty() => CueState::Loaded(cues),
            _ => CueState::Absent,
        };
    }

    fn read_cues(&mut self) -> Result<Option<Vec<meta::Cue>>> {
        let mut entries = self.seek_head.clone();
        if !entries.iter().any(|entry| entry.id == ids::CUES) {
            // One level of SeekHead pointing at a second SeekHead.
            for entry in self.seek_head.clone() {
                if entry.id != ids::SEEK_HEAD {
                    continue;
                }
                let Some(header) = self.element_at(entry.position, ids::SEEK_HEAD)? else {
                    continue;
                };
                let Size::Known(size) = header.size else {
                    continue;
                };
                let data = self.source.bytes(size, self.limits.max_header_bytes)?;
                entries.extend(meta::parse_seek_head(&data)?);
            }
        }
        let Some(entry) = entries.iter().find(|entry| entry.id == ids::CUES) else {
            return Ok(None);
        };
        let Some(header) = self.element_at(entry.position, ids::CUES)? else {
            return Ok(None);
        };
        let Size::Known(size) = header.size else {
            return Ok(None);
        };
        let data = self.source.bytes(size, MAX_CUES_BYTES)?;
        meta::parse_cues(&data, self.limits.max_index_entries).map(Some)
    }

    /// The header of the element at a Segment Position, when it has `id`.
    fn element_at(&mut self, segment_position: u64, id: u32) -> Result<Option<Header>> {
        let Some(at) = self.segment_data.checked_add(segment_position) else {
            return Ok(None);
        };
        if self.source.length().is_some_and(|length| at >= length) {
            return Ok(None);
        }
        self.source.seek(at)?;
        Ok(self.source.header()?.filter(|header| header.id == id))
    }

    /// Cluster timestamps and offsets, read from Cluster headers once.
    fn cluster_index(&mut self) -> Result<Vec<(i64, u64)>> {
        if let Some(clusters) = &self.clusters {
            return Ok(clusters.clone());
        }
        let mut clusters = Vec::new();
        if let Some(first) = self.first_cluster {
            self.source.seek(first)?;
            let mut pending: Option<Header> = None;
            for _ in 0..MAX_CLUSTER_SCAN {
                if self
                    .segment_end
                    .is_some_and(|end| self.source.position() >= end)
                {
                    break;
                }
                let header = match pending.take() {
                    Some(header) => header,
                    None => match self.source.header() {
                        Ok(Some(header)) => header,
                        // The index stops at the first broken element:
                        // seeks before it still work.
                        Ok(None) | Err(_) => break,
                    },
                };
                match header.id {
                    ids::CLUSTER => match self.cluster_timestamp(header) {
                        Ok((timestamp, next)) => {
                            if let Some(timestamp) = timestamp {
                                clusters.push((timestamp, header.start));
                            }
                            pending = next;
                        }
                        Err(_) => break,
                    },
                    ids::EBML | ids::SEGMENT => break,
                    _ => match header.size {
                        Size::Known(size) => {
                            if self.source.skip(size).is_err() {
                                break;
                            }
                        }
                        Size::Unknown => break,
                    },
                }
            }
        }
        self.clusters = Some(clusters.clone());
        Ok(clusters)
    }

    /// Reads a Cluster's Timestamp and moves past the Cluster. An unknown
    /// size Cluster is walked header by header to its end; the header that
    /// ended it comes back.
    fn cluster_timestamp(&mut self, cluster: Header) -> Result<(Option<i64>, Option<Header>)> {
        let mut timestamp = None;
        let mut children = 0_u32;
        loop {
            if cluster
                .end()
                .is_some_and(|end| self.source.position() >= end)
            {
                return Ok((timestamp, None));
            }
            if (timestamp.is_some() || children >= MAX_CHILDREN_BEFORE_TIMESTAMP)
                && let Some(end) = cluster.end()
            {
                self.source.seek(end)?;
                return Ok((timestamp, None));
            }
            let Some(child) = self.source.header()? else {
                return Ok((timestamp, None));
            };
            if cluster.end().is_none() && ids::ends_cluster(child.id) {
                return Ok((timestamp, Some(child)));
            }
            let Size::Known(size) = child.size else {
                return Err(Error::Ebml(EbmlError::UnknownSize { id: child.id }));
            };
            children = children.saturating_add(1);
            if child.id == ids::TIMESTAMP && timestamp.is_none() {
                let data = self.source.bytes(size, 8)?;
                timestamp = i64::try_from(ebml::uint(child.id, &data)?).ok();
            } else {
                self.source.skip(size)?;
            }
        }
    }

    /// Walks block headers from the Cluster at `start`. A broken or
    /// truncated element ends the walk; what was found before it stands.
    fn walk_from(&mut self, start: u64, number: u64, target: i64) -> Result<Walk> {
        let mut walk = Walk::default();
        match self.walk_inner(start, number, target, &mut walk) {
            Err(error) if walk.first.is_none() => Err(error),
            _ => Ok(walk),
        }
    }

    fn walk_inner(&mut self, start: u64, number: u64, target: i64, walk: &mut Walk) -> Result<()> {
        let mut trail = self.walked.trails.remove(&start).unwrap_or(Trail {
            blocks: Vec::new(),
            resume: Resume::At {
                position: start,
                cluster: None,
            },
        });
        self.walked.blocks = self.walked.blocks.saturating_sub(trail.blocks.len());
        let kept = (trail.blocks.len(), trail.resume);
        let result = self.walk_trail(&mut trail, number, target, walk);
        if result.is_err() {
            // What the failed read added is not kept, so the next walk
            // fails the same way.
            trail.blocks.truncate(kept.0);
            trail.resume = kept.1;
        }
        self.walked.blocks = self.walked.blocks.saturating_add(trail.blocks.len());
        self.walked.trails.insert(start, trail);
        result
    }

    /// Goes over a kept walk, then reads on from where it stopped while the
    /// walk needs more, keeping what it reads.
    fn walk_trail(
        &mut self,
        trail: &mut Trail,
        number: u64,
        target: i64,
        walk: &mut Walk,
    ) -> Result<()> {
        for passed in &trail.blocks {
            if walk.visited >= MAX_SEEK_WALK {
                return Ok(());
            }
            walk.visited = walk.visited.saturating_add(1);
            if passed.track != number {
                continue;
            }
            let Some(time) = passed.time else {
                continue;
            };
            let found = Found {
                time,
                keyframe: passed.keyframe,
                position: passed.position,
                cluster: passed.cluster,
            };
            if walk.see(found, passed.frames, target) {
                return Ok(());
            }
        }
        let Resume::At {
            position,
            mut cluster,
        } = trail.resume
        else {
            return Ok(());
        };
        let mut keeping = true;
        self.source.seek(position)?;
        let mut pending: Option<Header> = None;
        while walk.visited < MAX_SEEK_WALK {
            if let Some(current) = cluster
                && current.end.is_some_and(|end| self.source.position() >= end)
            {
                cluster = None;
            }
            let header = match pending.take() {
                Some(header) => header,
                None => {
                    if cluster.is_none()
                        && self
                            .segment_end
                            .is_some_and(|end| self.source.position() >= end)
                    {
                        break;
                    }
                    match self.source.header()? {
                        Some(header) => header,
                        None => break,
                    }
                }
            };
            let Some(current) = cluster.as_mut() else {
                match header.id {
                    ids::CLUSTER => {
                        cluster = Some(Cluster {
                            start: header.start,
                            end: header.end(),
                            timestamp: None,
                            blocks: 0,
                        });
                    }
                    ids::EBML | ids::SEGMENT => break,
                    _ => match header.size {
                        Size::Known(size) => self.source.skip(size)?,
                        Size::Unknown => {
                            return Err(Error::Ebml(EbmlError::UnknownSize { id: header.id }));
                        }
                    },
                }
                continue;
            };
            if current.end.is_none() && ids::ends_cluster(header.id) {
                cluster = None;
                pending = Some(header);
                continue;
            }
            let Size::Known(size) = header.size else {
                return Err(Error::Ebml(EbmlError::UnknownSize { id: header.id }));
            };
            let before = *current;
            let seen = match header.id {
                ids::TIMESTAMP => {
                    let data = self.source.bytes(size, 8)?;
                    current.timestamp = i64::try_from(ebml::uint(header.id, &data)?).ok();
                    None
                }
                ids::SIMPLE_BLOCK => {
                    current.blocks = current.blocks.saturating_add(1);
                    Some(self.simple_block_head(header, size)?)
                }
                ids::BLOCK_GROUP => {
                    current.blocks = current.blocks.saturating_add(1);
                    self.group_head(header, size)?
                }
                _ => {
                    self.source.skip(size)?;
                    None
                }
            };
            let Some((head, keyframe, frames)) = seen else {
                continue;
            };
            walk.visited = walk.visited.saturating_add(1);
            let time = before
                .timestamp
                .and_then(|base| base.checked_add(i64::from(head.relative)))
                .and_then(|segment| self.clock.from_segment(segment));
            keeping = keeping && self.walked.room(trail);
            if keeping {
                trail.blocks.push(Passed {
                    track: head.track,
                    time,
                    keyframe,
                    frames,
                    position: header.start,
                    cluster: before,
                });
                trail.resume = Resume::At {
                    position: self.source.position(),
                    cluster,
                };
            }
            if head.track != number {
                continue;
            }
            let Some(time) = time else {
                continue;
            };
            let found = Found {
                time,
                keyframe,
                position: header.start,
                cluster: before,
            };
            if walk.see(found, frames, target) {
                return Ok(());
            }
        }
        if keeping && walk.visited < MAX_SEEK_WALK {
            trail.resume = Resume::End;
        }
        Ok(())
    }

    /// Reads a SimpleBlock's header and skips its frames. Returns the
    /// header, the keyframe flag and the frame count.
    fn simple_block_head(
        &mut self,
        element: Header,
        size: u64,
    ) -> Result<(block::BlockHeader, bool, u32)> {
        let head = self.block_head(size)?;
        let frames = self.lace_frames(head, size)?;
        self.skip_to(element.end())?;
        Ok((head, head.flags & KEYFRAME != 0, frames))
    }

    /// Reads the Block header inside a BlockGroup and whether a
    /// ReferenceBlock marks it as not a keyframe. Frame bytes are skipped.
    fn group_head(
        &mut self,
        element: Header,
        size: u64,
    ) -> Result<Option<(block::BlockHeader, bool, u32)>> {
        let end = element
            .data
            .checked_add(size)
            .ok_or(Error::malformed("BlockGroup size overflows"))?;
        let mut head = None;
        let mut referenced = false;
        while self.source.position() < end {
            let Some(child) = self.source.header()? else {
                return Err(Error::malformed("BlockGroup runs past the end"));
            };
            let Some(child_end) = child.end().filter(|child_end| *child_end <= end) else {
                return Err(Error::malformed("BlockGroup child runs past the group"));
            };
            let child_size = child_end - child.data;
            match child.id {
                ids::BLOCK if head.is_none() => {
                    let parsed = self.block_head(child_size)?;
                    let frames = self.lace_frames(parsed, child_size)?;
                    head = Some((parsed, frames));
                    self.source.seek(child_end)?;
                }
                ids::REFERENCE_BLOCK => {
                    referenced = true;
                    self.source.skip(child_size)?;
                }
                _ => self.source.skip(child_size)?,
            }
        }
        Ok(head.map(|(parsed, frames)| (parsed, !referenced, frames)))
    }

    pub(crate) fn block_head(&mut self, size: u64) -> Result<block::BlockHeader> {
        let mut bytes = [0_u8; block::MAX_HEADER];
        self.source.fill(&mut bytes[..1])?;
        let len = block::header_len(bytes[0]).ok_or(Error::Ebml(EbmlError::BadSize))?;
        if u64::try_from(len).unwrap_or(u64::MAX) > size {
            return Err(Error::malformed("block is shorter than its header"));
        }
        self.source.fill(&mut bytes[1..len])?;
        block::parse_header(&bytes[..len])
    }

    /// The frame count of a laced block: the one octet after its header.
    fn lace_frames(&mut self, head: block::BlockHeader, size: u64) -> Result<u32> {
        if head.lacing() == 0 || u64::try_from(head.len).unwrap_or(u64::MAX) >= size {
            return Ok(1);
        }
        let mut count = [0_u8; 1];
        self.source.fill(&mut count)?;
        Ok(block::lace_count(count[0]))
    }

    fn skip_to(&mut self, end: Option<u64>) -> Result<()> {
        match end {
            Some(end) => self.source.seek(end),
            None => Err(Error::malformed("block size is unknown")),
        }
    }
}

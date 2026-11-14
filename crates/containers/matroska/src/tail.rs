//! The DiscardPadding at the end of each track, read once at open.
//!
//! A track's trailing padding is the DiscardPadding of its last block
//! (RFC 9559, section 5.1.3.5.8), so the demuxer reads the block headers of
//! the file's last cue point's Cluster and what follows it, skipping frame
//! bytes, and keeps each track's last block's padding. A file without Cues
//! is not walked: finding its last Cluster would mean walking them all.

use std::io::{Read, Seek};

use crate::demux::{CueState, MatroskaDemux};
use crate::ebml::{self, Size};
use crate::error::Result;
use crate::ids;
use crate::limits::MAX_TAIL_BLOCKS;

impl<R: Read + Seek> MatroskaDemux<R> {
    /// Each track's DiscardPadding in nanoseconds, from its last block, for
    /// the tracks whose last block has one. Empty when the file has no Cues
    /// or the walk ran into something it could not read.
    pub(crate) fn end_padding(&mut self) -> Vec<(u64, u64)> {
        self.load_cues();
        let CueState::Loaded(cues) = &self.cues else {
            return Vec::new();
        };
        let Some(last) = cues.iter().map(|cue| cue.cluster).max() else {
            return Vec::new();
        };
        let Some(start) = self.segment_data.checked_add(last) else {
            return Vec::new();
        };
        let mut found = Vec::new();
        match self.walk_tail(start, &mut found) {
            Ok(()) => found
                .into_iter()
                .filter(|(_, padding)| *padding > 0)
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Walks element headers from `start` to the end of the Segment. A
    /// Cluster is entered whatever its size, since its children's IDs are
    /// not top-level IDs; any other element of known size is skipped.
    fn walk_tail(&mut self, start: u64, found: &mut Vec<(u64, u64)>) -> Result<()> {
        self.source.seek(start)?;
        let mut blocks = 0_u32;
        while blocks < MAX_TAIL_BLOCKS {
            if self
                .segment_end
                .is_some_and(|end| self.source.position() >= end)
            {
                return Ok(());
            }
            let Some(header) = self.source.header()? else {
                return Ok(());
            };
            let size = match (header.id, header.size) {
                (ids::CLUSTER, _) => continue,
                (ids::EBML | ids::SEGMENT, _) => return Ok(()),
                (_, Size::Known(size)) => size,
                (_, Size::Unknown) => return Ok(()),
            };
            let end = header
                .end()
                .ok_or(crate::error::Error::malformed("element size overflows"))?;
            let (track, padding) = match header.id {
                ids::SIMPLE_BLOCK => (self.block_head(size)?.track, 0),
                ids::BLOCK_GROUP => self.group_padding(end)?,
                _ => {
                    self.source.seek(end)?;
                    continue;
                }
            };
            self.source.seek(end)?;
            blocks += 1;
            match found.iter_mut().find(|(number, _)| *number == track) {
                Some(entry) => entry.1 = padding,
                None => found.push((track, padding)),
            }
        }
        // Too many blocks after the last cue point to be sure which is last.
        found.clear();
        Ok(())
    }

    /// The track and DiscardPadding of the BlockGroup whose data ends at
    /// `end`. A negative padding counts as none.
    fn group_padding(&mut self, end: u64) -> Result<(u64, u64)> {
        let mut track = None;
        let mut padding = 0;
        while self.source.position() < end {
            let Some(child) = self.source.header()? else {
                break;
            };
            let Some(child_end) = child.end().filter(|child_end| *child_end <= end) else {
                return Err(crate::error::Error::malformed(
                    "BlockGroup child runs past the group",
                ));
            };
            let child_size = child_end - child.data;
            match child.id {
                ids::BLOCK if track.is_none() => track = Some(self.block_head(child_size)?.track),
                ids::DISCARD_PADDING => {
                    let data = self.source.bytes(child_size, 8)?;
                    padding = u64::try_from(ebml::int(child.id, &data)?).unwrap_or(0);
                }
                _ => {}
            }
            self.source.seek(child_end)?;
        }
        let track = track.ok_or(crate::error::Error::malformed("BlockGroup has no Block"))?;
        Ok((track, padding))
    }
}

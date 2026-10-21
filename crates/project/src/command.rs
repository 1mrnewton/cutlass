//! Edits the interface and the agent both send.
//!
//! A command does no I/O. The project assigns every new id and returns it.

use crate::asset::Asset;
use crate::id::{AssetId, SegmentId, TrackId};
use crate::rate::SequenceRate;
use crate::segment::Segment;
use crate::settings::{AspectRatio, Color, ColorSpace};
use crate::time::{Speed, TimeRange};

/// One edit. Several of these can be one undo step.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    AddAsset(Asset),
    Rename(String),
    SetCanvas {
        width: u32,
        height: u32,
        ratio: AspectRatio,
    },
    SetRate(SequenceRate),
    SetColorSpace(ColorSpace),
    SetBackground(Color),
    SetMagnet(bool),
    SetLinkage(bool),
    /// `track: None` opens a track of the segment's kind.
    Place {
        track: Option<TrackId>,
        segment: Segment,
    },
    RemoveSegment(SegmentId),
    Trim {
        segment: SegmentId,
        source: TimeRange,
    },
    SetSpeed {
        segment: SegmentId,
        speed: Speed,
    },
    /// `track: None` keeps the segment on its track. `start` is microseconds.
    Move {
        segment: SegmentId,
        track: Option<TrackId>,
        start: i64,
    },
    /// `at` is a sequence time strictly inside the segment. The left piece keeps
    /// the id. The outcome is the new right piece.
    Split {
        segment: SegmentId,
        at: i64,
    },
}

/// What one applied command produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Unchanged,
    Asset(AssetId),
    Segment(SegmentId),
    /// The segment landed on a track this command opened.
    Opened {
        segment: SegmentId,
        track: TrackId,
    },
}

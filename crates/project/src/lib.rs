//! The project model: settings, assets, and one timeline.
//!
//! The crate does no I/O. The shell mints the project id and passes source
//! strings through. The project assigns every other id. Commands and the undo
//! stacks live here. `docs/project.md` is the model.

#![deny(unsafe_code)]

mod asset;
mod command;
mod error;
mod history;
mod id;
mod project;
mod rate;
mod segment;
mod settings;
mod time;
mod timeline;
mod track;

pub use asset::{Asset, AssetKind};
pub use command::{Command, Outcome};
pub use error::Error;
pub use history::History;
pub use id::{AssetId, ProjectId, SegmentId, TimelineId, TrackId};
pub use project::Project;
pub use rate::{FrameRate, SequenceRate};
pub use segment::{Channels, Keyframe, KeyframeValue, Segment, Transform};
pub use settings::{AspectRatio, Canvas, Color, ColorSpace, Settings};
pub use time::{SEQUENCE_TIMESCALE, Speed, TimeRange, timeline_duration};
pub use timeline::Timeline;
pub use track::{Track, TrackKind};

#[cfg(test)]
pub(crate) mod test_support {
    use std::num::NonZeroU32;

    use crate::id::{AssetId, SegmentId, TimelineId, TrackId};

    pub(crate) fn asset_id(value: u64) -> AssetId {
        AssetId::raw(value)
    }

    pub(crate) fn track_id(value: u64) -> TrackId {
        TrackId::raw(value)
    }

    pub(crate) fn segment_id(value: u64) -> SegmentId {
        SegmentId::raw(value)
    }

    pub(crate) fn timeline_id(value: u64) -> TimelineId {
        TimelineId::raw(value)
    }

    pub(crate) fn scale(value: u32) -> NonZeroU32 {
        NonZeroU32::new(value).unwrap()
    }
}

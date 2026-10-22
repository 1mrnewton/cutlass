//! The project model: settings, assets, one sequence, and nested timelines.
//!
//! The crate does not open a path. [`Project::to_json`] and [`Project::from_json`]
//! are version [`FILE_VERSION`] of the project file. The shell mints the project
//! id, passes source strings through, and reads and writes the path. The project
//! assigns every other id. Undo and redo stay in [`History`] for the session.
//! `docs/project.md` is the model.

#![deny(unsafe_code)]

mod animation;
mod asset;
mod attachment;
mod command;
mod curve;
mod error;
mod file;
mod history;
mod id;
mod mark;
mod project;
mod rate;
mod segment;
mod settings;
mod time;
mod timeline;
mod track;
mod transition;

pub use animation::{Animation, AnimationKind};
pub use asset::{Asset, AssetKind};
pub use attachment::{Attachment, Parameter, ParameterValue};
pub use command::{Command, Outcome, TrimEdge, commands_for_trim};
pub use curve::{SpeedCurve, SpeedPoint};
pub use error::Error;
pub use file::FILE_VERSION;
pub use history::History;
pub use id::{AssetId, ProjectId, SegmentId, TimelineId, TrackId};
pub use project::Project;
pub use rate::{FrameRate, SequenceRate};
pub use segment::{
    BezierEase, Blend, Channels, Crop, Easing, Keyframe, KeyframeProperty, KeyframeValue, Segment,
    Transform,
};
pub use settings::{AspectRatio, Canvas, Color, ColorSpace, Settings};
pub use time::{SEQUENCE_TIMESCALE, Speed, TimeRange, source_ticks, timeline_duration};
pub use timeline::Timeline;
pub use track::{Track, TrackKind};
pub use transition::Transition;

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

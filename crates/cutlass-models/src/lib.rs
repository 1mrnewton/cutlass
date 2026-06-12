//! Cutlass editing data model: project, media pool, timeline, tracks, clips.
//!
//! Design goals:
//! - **Correct types**: strongly-typed IDs (no mixing a [`TrackId`] with a
//!   [`ClipId`]), OTIO-style [`RationalTime`] (exact NTSC rates), and explicit
//!   timeline-vs-source ranges each carrying their own rate.
//! - **Fast lookup**: entities are stored in hash maps keyed by ID, so finding
//!   a clip/track/media by ID is O(1). The timeline keeps a `ClipId -> TrackId`
//!   index so a clip can be located across all tracks without scanning.
//! - **Independent testing**: this crate has no dependency on decode/render, so
//!   the model can be exercised in isolation.
//!
//! A [`Project`] owns one [`Timeline`] and a media pool of [`MediaSource`]s.

mod clip;
mod error;
mod ids;
mod media;
mod metadata;
mod param;
mod persist;
mod project;
mod schema;
mod serde_map;
mod time;
mod timeline;
mod track;

/// Fast hash map for integer-keyed entities (FxHash; no DoS resistance needed
/// for an in-process editing model, but much faster than SipHash for `u64`).
pub type Map<K, V> = rustc_hash::FxHashMap<K, V>;

pub use clip::{
    AnimatedTransform, Clip, ClipParam, ClipSource, ClipTransform, Generator, MAX_CLIP_VOLUME,
    ParamValue, Shape, TextAlignH, TextAlignV, TextBackground, TextCase, TextShadow, TextStroke,
    TextStyle, audio_gain_at,
};
pub use error::ModelError;
pub use param::{Easing, Keyframe, Lerp, Param};
pub use ids::{ClipId, LinkId, MarkerId, MediaId, ProjectId, TrackId};
pub use media::{MediaKind, MediaSource, STILL_DEFAULT_DURATION_TICKS, STILL_TICK_RATE};
pub use metadata::ProjectMetadata;
pub use persist::{PROJECT_FILE_EXTENSION, PROJECT_FILE_VERSION};
pub use project::Project;
pub use schema::{ProjectSchema, PROJECT_SCHEMA_KIND, PROJECT_SCHEMA_VERSION};
pub use time::{
    Rational, RationalTime, TimeRange, check_same_rate, rate_eq, resample, time_add, time_sub,
};
pub use timeline::{Marker, MarkerColor, Timeline};
pub use track::{Track, TrackKind};

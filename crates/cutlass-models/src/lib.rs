//! Cutlass editing data model: project, media pool, timeline, tracks, clips.
//!
//! Design goals:
//! - **Correct types**: strongly-typed IDs (no mixing a [`TrackId`] with a
//!   [`ClipId`]), rational frame rates (exact for 23.976/29.97), and explicit
//!   timeline-vs-source frame ranges.
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
mod project;
mod time;
mod timeline;
mod track;

/// Fast hash map for integer-keyed entities (FxHash; no DoS resistance needed
/// for an in-process editing model, but much faster than SipHash for `u64`).
pub type Map<K, V> = rustc_hash::FxHashMap<K, V>;

pub use clip::{Clip, ClipSource, Generator, Shape};
pub use error::ModelError;
pub use ids::{ClipId, MediaId, ProjectId, TrackId};
pub use media::MediaSource;
pub use project::Project;
pub use time::{Rational, TimeRange, convert_frames};
pub use timeline::Timeline;
pub use track::{Track, TrackKind};

//! Editor domain model (pure Rust).
//!
//! Tracks and clips are stored in `HashMap`s keyed by id, with sibling
//! `*_order` vectors for stable iteration. This is the **only** shape
//! the command layer (`crate::command`) and the agent are allowed to
//! mutate — the Slint-facing projection is one-way and lives in
//! `crate::projector`.

pub mod clip;
pub mod color;
pub mod project;
pub mod rational;
pub mod rational_time;
pub mod sample;
pub mod sequence;
pub mod time_range;
pub mod track;

pub use clip::Clip;
pub use color::Color;
pub use project::Project;
pub use rational::Rational;
pub use rational_time::RationalTime;
pub use sequence::Sequence;
pub use time_range::TimeRange;
pub use track::{Track, TrackKind};

pub use sample::sample_project;

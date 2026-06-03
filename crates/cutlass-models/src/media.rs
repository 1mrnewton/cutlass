use std::path::{Path, PathBuf};

use crate::ids::MediaId;
use crate::time::{Rational, TimeRange};

/// An imported source file in the project's media pool.
///
/// This is the *asset*, not a placement on the timeline; many [`Clip`](crate::Clip)s
/// can reference the same `MediaSource`.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaSource {
    pub id: MediaId,
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    /// Native frame rate of the source.
    pub frame_rate: Rational,
    /// Total length of the source, in source frames.
    pub duration: i64,
    pub has_audio: bool,
}

impl MediaSource {
    /// Create a media source with a freshly allocated [`MediaId`].
    pub fn new(
        path: impl Into<PathBuf>,
        width: u32,
        height: u32,
        frame_rate: Rational,
        duration: i64,
        has_audio: bool,
    ) -> Self {
        Self {
            id: MediaId::next(),
            path: path.into(),
            width,
            height,
            frame_rate,
            duration: duration.max(0),
            has_audio,
        }
    }

    /// The full extent of the source as a frame range `[0, duration)`.
    pub fn full_range(&self) -> TimeRange {
        TimeRange::new(0, self.duration)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

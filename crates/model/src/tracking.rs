//! A tracked target's path on the canvas, as the tracking job hands it to
//! the `Track` request.

use serde::{Deserialize, Serialize};

use crate::time::Micros;

/// The record type of a tracking result in `video_trackings`.
pub const VIDEO_TRACKING: &str = "video_tracking";

/// What the target's move may be. The reference editor's `tracker_type`
/// is 0 for an object and 1 for a plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrackMode {
    /// Move, scale and turn.
    Object,
    /// Four corners that move apart, as a tilting screen's do.
    Plane,
}

impl TrackMode {
    /// The reference editor's `tracker_type`.
    #[must_use]
    pub const fn tracker_type(self) -> i64 {
        match self {
            Self::Object => 0,
            Self::Plane => 1,
        }
    }
}

/// The target's four corners at one timeline time, in canvas transform
/// units: 0 is the middle, 1 is half the canvas, y up.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackSample {
    pub time: Micros,
    pub corners: [(f64, f64); 4],
}

/// A track: the mode it ran in and the target's corners frame by frame, in
/// timeline order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackPath {
    pub mode: TrackMode,
    pub samples: Vec<TrackSample>,
}

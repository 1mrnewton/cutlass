//! The rows a phone draws, and the clip edits it sends.
//!
//! Rows follow the editor's lanes: file order, an empty ordinary track left
//! out, and main kept when it is empty. A split keeps a frame on each side.

use std::path::Path;

use cutlass_editor::{
    GestureError, GestureOutcome, MoveGesture, Pointer, TimelineScale, TrimGesture, lanes,
    move_gesture, trim_gesture,
};
use cutlass_model::{MaterialRef, Segment, TrackAttribute, TrackType};
use cutlass_session::{Draft, Micros, Request, SegmentId, TrackId, TrimEdge};

use crate::error::FfiError;
use crate::media::{fresh, ids};

/// One clip on a row.
#[derive(Clone, Debug, PartialEq)]
pub struct PhoneClip {
    pub id: SegmentId,
    /// The file name, the text, or the segment id.
    pub name: String,
    pub start: Micros,
    pub duration: Micros,
    /// Linear gain: 1 is unity and 0 is silence.
    pub volume: f64,
    /// Constant speed: 2 plays twice as fast.
    pub speed: f64,
}

/// One row, top first.
#[derive(Clone, Debug, PartialEq)]
pub struct PhoneLane {
    pub track_id: TrackId,
    pub kind: TrackType,
    pub main: bool,
    pub locked: bool,
    pub clips: Vec<PhoneClip>,
}

/// The rows of `draft`, top first.
#[must_use]
pub fn phone_lanes(draft: &Draft) -> Vec<PhoneLane> {
    lanes(draft)
        .into_iter()
        .filter_map(|lane| {
            let track = draft.tracks.get(lane.track_index)?;
            Some(PhoneLane {
                track_id: lane.track_id,
                kind: lane.kind,
                main: lane.main,
                locked: track.attribute.contains(TrackAttribute::LOCK),
                clips: track
                    .segments
                    .iter()
                    .map(|segment| PhoneClip {
                        id: segment.id.clone(),
                        name: clip_name(draft, segment),
                        start: segment.target.start,
                        duration: segment.target.duration,
                        volume: segment.volume,
                        speed: segment.speed,
                    })
                    .collect(),
            })
        })
        .collect()
}

fn clip_name(draft: &Draft, segment: &Segment) -> String {
    let file = |path: &str| {
        Path::new(path).file_name().map_or_else(
            || path.to_owned(),
            |name| name.to_string_lossy().into_owned(),
        )
    };
    match draft.materials.get(segment.material_id.as_str()) {
        Some(MaterialRef::Video(video)) if !video.path.is_empty() => file(&video.path),
        Some(MaterialRef::Audio(audio)) if !audio.path.is_empty() => file(&audio.path),
        Some(MaterialRef::Text(text)) if text.words().is_some_and(|words| !words.is_empty()) => {
            text.words().unwrap_or_default().into_owned()
        }
        Some(MaterialRef::VideoEffect(effect)) if !effect.name.is_empty() => effect.name.clone(),
        _ => segment.id.as_str().to_owned(),
    }
}

/// The split request for `segment_id` at `at`, and the right piece's id.
///
/// Each piece keeps at least one frame at the draft rate.
pub(crate) fn split_request(
    draft: &Draft,
    segment_id: &SegmentId,
    at: Micros,
) -> Result<(Request, SegmentId), FfiError> {
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == segment_id)
        .ok_or_else(|| refused("the clip is not on the timeline"))?;
    let margin = draft
        .frame_rate()
        .and_then(|rate| rate.frame_start(1))
        .map_or(1, |time| time.0.max(1));
    let start = segment.target.start.0;
    let end = segment.target.end().0;
    if at.0 < start.saturating_add(margin) || at.0 > end.saturating_sub(margin) {
        return Err(refused("split is not available there"));
    }
    let right_id = SegmentId::new(fresh(&ids(draft), "seg"));
    Ok((
        Request::Split {
            segment_id: segment_id.clone(),
            at,
            right_id: right_id.clone(),
        },
        right_id,
    ))
}

/// Where a dragged clip lands, in the phone's own words: a row, a start,
/// and how near a cut, the playhead, or the timeline's start must be to pull
/// it there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Drag {
    pub row: u32,
    pub time: Micros,
    pub snap: Micros,
    pub playhead: Micros,
}

/// The move request for `segment_id` dropped at `drag`, and where it starts.
///
/// The shared move gesture places it on the row under the finger, or the
/// nearest row above of the same kind, and may snap the start. A drop that
/// would need a new row, or a row that does not exist, is refused.
pub(crate) fn move_request(
    draft: &Draft,
    segment_id: &SegmentId,
    drag: Drag,
) -> Result<(Request, Micros), FfiError> {
    let outcome = move_gesture(
        draft,
        &MoveGesture {
            segment_id: segment_id.clone(),
            pointer: Pointer {
                x: drag.time.0.max(0) as f64,
                y: f64::from(drag.row) + 0.5,
            },
            scale: UNIT,
            playhead: drag.playhead,
            snap_threshold: drag.snap,
        },
    )
    .map_err(gesture_refused)?;
    match outcome {
        GestureOutcome::Edit(request) => {
            let start = match &request {
                Request::Move { start, .. } => *start,
                _ => drag.time,
            };
            Ok((request, start))
        }
        GestureOutcome::Miss => Err(refused("the clip was dropped off the rows")),
        GestureOutcome::NewLane { .. } => Err(refused("there is no row there for this clip")),
    }
}

/// The trim request for the `start` or end edge of `segment_id` dragged to
/// `drag.time`, and where the edge lands.
///
/// The edge may snap, stays inside the clip's media, and stops at its
/// neighbor on the row. Each clip keeps at least one microsecond.
pub(crate) fn trim_request(
    draft: &Draft,
    segment_id: &SegmentId,
    start: bool,
    drag: Drag,
) -> Result<(Request, Micros), FfiError> {
    let plan = trim_gesture(
        draft,
        &TrimGesture {
            segment_id: segment_id.clone(),
            edge: if start {
                TrimEdge::Start
            } else {
                TrimEdge::End
            },
            pointer: Pointer {
                x: drag.time.0 as f64,
                y: 0.0,
            },
            scale: UNIT,
            playhead: drag.playhead,
            snap_threshold: drag.snap,
        },
    )
    .map_err(gesture_refused)?;
    Ok((plan.request(), plan.time))
}

/// One pixel per microsecond and one pixel per row, so a phone passes times
/// and rows and the gesture's own geometry stays in Rust.
const UNIT: TimelineScale = TimelineScale {
    lane_height: 1.0,
    pixel_size: 1.0,
    empty_lanes: false,
};

fn gesture_refused(error: GestureError) -> FfiError {
    refused(match error {
        GestureError::UnknownSegment => "the clip is not on the timeline",
        GestureError::BadScale => "the time is not on the timeline",
        GestureError::EmptyDuration => "the clip has no length",
    })
}

fn refused(reason: &str) -> FfiError {
    FfiError::Refused {
        reason: reason.to_owned(),
    }
}

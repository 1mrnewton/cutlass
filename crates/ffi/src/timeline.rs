//! The rows a phone draws, and the clip edits it sends.
//!
//! Rows follow the editor's lanes: file order, an empty ordinary track left
//! out, and main kept when it is empty. A split keeps a frame on each side.

use std::path::Path;

use cutlass_editor::lanes;
use cutlass_model::{MaterialRef, Segment, TrackAttribute, TrackType};
use cutlass_session::{Draft, Micros, Request, SegmentId, TrackId};

use crate::error::FfiError;
use crate::media::{fresh, ids};

/// One clip on a row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneClip {
    pub id: SegmentId,
    /// The file name, the text, or the segment id.
    pub name: String,
    pub start: Micros,
    pub duration: Micros,
}

/// One row, top first.
#[derive(Clone, Debug, PartialEq, Eq)]
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
        Some(MaterialRef::Text(text)) if !text.content.is_empty() => text.content.clone(),
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

fn refused(reason: &str) -> FfiError {
    FfiError::Refused {
        reason: reason.to_owned(),
    }
}

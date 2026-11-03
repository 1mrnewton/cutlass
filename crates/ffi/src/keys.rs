//! A clip's keyframes as the timeline shows them, and the keyframe button.

use cutlass_model::{Draft, key_offsets};
use cutlass_session::{Micros, Request, SegmentId};

/// A clip's keyframes, as timeline times.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneKeys {
    /// Each keyframe's timeline time, earliest first.
    pub times: Vec<Micros>,
    /// The keyframe within a frame of the time asked, when one is.
    pub here: Option<Micros>,
    start: Micros,
    duration: Micros,
}

impl PhoneKeys {
    pub(crate) fn of(draft: &Draft, segment_id: &SegmentId, time: Micros) -> Option<Self> {
        let segment = draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .find(|segment| segment.id == *segment_id)?;
        let start = segment.target.start;
        let times: Vec<Micros> = key_offsets(segment)
            .into_iter()
            .map(|offset| Micros(start.0.saturating_add(offset.0)))
            .collect();
        let reach = frame(draft.fps);
        let here = times
            .iter()
            .copied()
            .filter(|key| (key.0 - time.0).abs() <= reach)
            .min_by_key(|key| (key.0 - time.0).abs());
        Some(Self {
            times,
            here,
            start,
            duration: segment.target.duration,
        })
    }

    /// The request the keyframe button makes at `time`: remove the key here,
    /// else add one when `time` is on the clip.
    pub(crate) fn toggle(&self, segment_id: &SegmentId, time: Micros) -> Option<Request> {
        if let Some(here) = self.here {
            return Some(Request::RemoveKeyframe {
                segment_id: segment_id.clone(),
                offset: Micros(here.0 - self.start.0),
            });
        }
        let offset = time.0.checked_sub(self.start.0)?;
        (0..=self.duration.0)
            .contains(&offset)
            .then(|| Request::AddKeyframe {
                segment_id: segment_id.clone(),
                offset: Micros(offset),
            })
    }
}

/// One frame at `fps` in microseconds; 30 when the rate is not usable.
fn frame(fps: f64) -> i64 {
    let fps = if fps.is_finite() && fps >= 1.0 {
        fps
    } else {
        30.0
    };
    // A frame of at least one frame per second fits easily.
    #[expect(clippy::cast_possible_truncation, reason = "at most a million")]
    let frame = (1_000_000.0 / fps).round() as i64;
    frame
}

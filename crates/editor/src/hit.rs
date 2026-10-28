use cutlass_model::{Draft, Micros, SegmentId, TimeRange};

use crate::lane::{Lane, lanes};

/// A point on the timeline. `x` is pixels from time zero. `y` is pixels from the top lane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pointer {
    pub x: f64,
    pub y: f64,
}

/// How the pointer maps onto the lanes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimelineScale {
    /// Height of one lane, in pixels.
    pub lane_height: f64,
    /// Pixels per microsecond.
    pub pixel_size: f64,
}

/// The lane and time under a pointer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    pub lane: Lane,
    pub time: Micros,
    pub segment_id: Option<SegmentId>,
}

/// The pixel size or the lane height cannot map a pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BadScale;

/// The lane and time under `pointer`.
///
/// [`None`] is a miss: the pointer is above the first lane, or below the last.
/// A miss does not name a segment. The draft is not changed.
pub fn hit_test(
    draft: &Draft,
    scale: TimelineScale,
    pointer: Pointer,
) -> Result<Option<Hit>, BadScale> {
    let time = time_at(scale.pixel_size, pointer.x).ok_or(BadScale)?;
    let row = row_at(scale.lane_height, pointer.y)?;
    let rows = lanes(draft);
    let Some(row) = row else {
        return Ok(None);
    };
    let Some(lane) = rows.get(row).cloned() else {
        return Ok(None);
    };
    let segment_id = draft
        .tracks
        .get(lane.track_index)
        .and_then(|track| segment_at(track, time));
    Ok(Some(Hit {
        lane,
        time,
        segment_id,
    }))
}

fn row_at(lane_height: f64, y: f64) -> Result<Option<usize>, BadScale> {
    if !lane_height.is_finite() || lane_height <= 0.0 || !y.is_finite() {
        return Err(BadScale);
    }
    if y < 0.0 {
        return Ok(None);
    }
    let row = (y / lane_height).floor();
    if !row.is_finite() || row < 0.0 || row >= usize::MAX as f64 {
        return Ok(None);
    }
    Ok(Some(row as usize))
}

fn time_at(pixel_size: f64, x: f64) -> Option<Micros> {
    if !pixel_size.is_finite() || pixel_size <= 0.0 || !x.is_finite() {
        return None;
    }
    let raw = x / pixel_size;
    if !raw.is_finite() {
        return None;
    }
    let rounded = raw.round();
    let limit = (1_u64 << 63) as f64;
    if rounded < i64::MIN as f64 || rounded >= limit {
        return None;
    }
    Some(Micros(rounded as i64))
}

fn segment_at(track: &cutlass_model::Track, time: Micros) -> Option<SegmentId> {
    track
        .segments
        .iter()
        .find_map(|segment| contains(&segment.target, time).then(|| segment.id.clone()))
}

fn contains(range: &TimeRange, time: Micros) -> bool {
    if range.duration.0 <= 0 {
        return false;
    }
    let Some(end) = range.start.0.checked_add(range.duration.0) else {
        return false;
    };
    time.0 >= range.start.0 && time.0 < end
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, Segment, SegmentId, TimeRange, TrackId};

    use super::*;

    fn scale() -> TimelineScale {
        TimelineScale {
            lane_height: 40.0,
            pixel_size: 1.0,
        }
    }

    fn draft_with(start: i64, duration: i64) -> Draft {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        draft.tracks[0].segments.push(Segment::new(
            "clip",
            "media",
            TimeRange::new(start, duration),
        ));
        draft
    }

    #[test]
    fn a_pointer_outside_the_lanes_misses() {
        let draft = Draft::empty("D");
        let above = hit_test(&draft, scale(), Pointer { x: 10.0, y: -0.1 }).unwrap();
        let below = hit_test(&draft, scale(), Pointer { x: 10.0, y: 40.0 }).unwrap();
        assert!(above.is_none());
        assert!(below.is_none());
        assert!(draft.tracks[0].segments.is_empty());
    }

    #[test]
    fn a_pointer_on_a_segment_names_that_lane_and_time() {
        let draft = draft_with(1_000, 5_000);
        let before = draft.clone();
        let hit = hit_test(
            &draft,
            scale(),
            Pointer {
                x: 1_500.0,
                y: 10.0,
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(draft, before);
        assert_eq!(hit.lane.track_id.as_str(), "main");
        assert!(hit.lane.main);
        assert_eq!(hit.time, Micros(1_500));
        assert_eq!(hit.segment_id.as_ref().map(SegmentId::as_str), Some("clip"));
    }

    #[test]
    fn the_segment_span_is_half_open() {
        let mut draft = draft_with(0, 100);
        draft.tracks[0]
            .segments
            .push(Segment::new("next", "media", TimeRange::new(100, 50)));
        let at_end = hit_test(&draft, scale(), Pointer { x: 100.0, y: 1.0 })
            .unwrap()
            .unwrap();
        assert_eq!(
            at_end.segment_id.as_ref().map(SegmentId::as_str),
            Some("next")
        );
        let gap = hit_test(&draft_with(0, 100), scale(), Pointer { x: 100.0, y: 1.0 })
            .unwrap()
            .unwrap();
        assert!(gap.segment_id.is_none());
        assert_eq!(gap.time, Micros(100));
    }

    #[test]
    fn an_empty_main_lane_still_answers_the_time() {
        let draft = Draft::empty("D");
        let hit = hit_test(&draft, scale(), Pointer { x: 0.0, y: 0.0 })
            .unwrap()
            .unwrap();
        assert!(hit.lane.main);
        assert!(hit.segment_id.is_none());
        assert_eq!(hit.time, Micros(0));
    }

    #[test]
    fn a_bad_scale_is_not_a_miss() {
        let draft = Draft::empty("D");
        let bad_height = TimelineScale {
            lane_height: 0.0,
            pixel_size: 1.0,
        };
        let bad_zoom = TimelineScale {
            lane_height: 40.0,
            pixel_size: 0.0,
        };
        assert_eq!(
            hit_test(&draft, bad_height, Pointer { x: 0.0, y: 0.0 }),
            Err(BadScale)
        );
        assert_eq!(
            hit_test(&draft, bad_zoom, Pointer { x: 0.0, y: 0.0 }),
            Err(BadScale)
        );
    }

    #[test]
    fn the_last_segment_on_a_lane_is_still_found() {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        for index in 0..128 {
            let start = index * 10;
            draft.tracks[0].segments.push(Segment::new(
                format!("s-{index}"),
                "media",
                TimeRange::new(start, 10),
            ));
        }
        let hit = hit_test(&draft, scale(), Pointer { x: 1_275.0, y: 1.0 })
            .unwrap()
            .unwrap();
        assert_eq!(
            hit.segment_id.as_ref().map(SegmentId::as_str),
            Some("s-127")
        );
    }
}

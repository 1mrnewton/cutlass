use cutlass_model::{Draft, MaterialRef, Micros, Segment, SegmentId, TimeRange};

use crate::request::TrimEdge;

use super::speed::map_timeline_to_source;
use super::{EditError, ensure_unlocked};
use super::{locate, overlaps_except};

pub fn trim(
    draft: &mut Draft,
    segment_id: &SegmentId,
    edge: TrimEdge,
    time: Micros,
) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let current = draft.tracks[track_index].segments[segment_index].clone();
    let mut next = current.clone();
    let speed = super::speed::timeline_speed(draft, &current);
    apply_edge(&mut next, edge, time, speed)?;
    within_media(draft, &next, edge)?;
    if !super::magnet::packs(draft.config.maintrack_adsorb, &draft.tracks[track_index])
        && overlaps_except(&draft.tracks[track_index], &next, segment_id)?
    {
        return Err(EditError::Overlap);
    }
    draft.tracks[track_index].segments[segment_index] = next;
    Ok(())
}

fn apply_edge(
    segment: &mut Segment,
    edge: TrimEdge,
    time: Micros,
    speed: f64,
) -> Result<(), EditError> {
    let end = segment
        .target
        .start
        .0
        .checked_add(segment.target.duration.0)
        .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
    match edge {
        TrimEdge::Start => {
            let duration = end
                .checked_sub(time.0)
                .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
            if duration <= 0 {
                return Err(EditError::Range(
                    cutlass_model::RangeError::NonPositiveDuration,
                ));
            }
            let delta = time
                .0
                .checked_sub(segment.target.start.0)
                .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
            segment.target.start = time;
            segment.target.duration = Micros(duration);
            // A key stays on the moment of footage it was set on.
            cutlass_model::shift_keys(segment, -delta);
            if let Some(source) = segment.source_range.as_mut() {
                let source_delta = map_timeline_to_source(delta, speed)?;
                let source_duration = map_timeline_to_source(duration, speed)?;
                shift_source_start(source, source_delta, source_duration)?;
            }
        }
        TrimEdge::End => {
            let duration = time
                .0
                .checked_sub(segment.target.start.0)
                .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
            if duration <= 0 {
                return Err(EditError::Range(
                    cutlass_model::RangeError::NonPositiveDuration,
                ));
            }
            segment.target.duration = Micros(duration);
            if let Some(source) = segment.source_range.as_mut() {
                let source_duration = map_timeline_to_source(duration, speed)?;
                if source_duration <= 0 {
                    return Err(EditError::Range(
                        cutlass_model::RangeError::NonPositiveDuration,
                    ));
                }
                source.duration = Micros(source_duration);
            }
        }
    }
    Ok(())
}

/// The trimmed edge stays on the media: the start at or after its first
/// frame, the end at or before its length when the material has one.
fn within_media(draft: &Draft, segment: &Segment, edge: TrimEdge) -> Result<(), EditError> {
    let Some(source) = segment.source_range.as_ref() else {
        return Ok(());
    };
    let length = match draft.materials.get(segment.material_id.as_str()) {
        Some(MaterialRef::Video(video)) => video.duration.0,
        Some(MaterialRef::Audio(audio)) => audio.duration.0,
        _ => 0,
    };
    let past = match edge {
        TrimEdge::Start => source.start.0 < 0,
        TrimEdge::End => length > 0 && source.start.0.saturating_add(source.duration.0) > length,
    };
    if past {
        return Err(EditError::SourceBounds);
    }
    Ok(())
}

fn shift_source_start(source: &mut TimeRange, delta: i64, duration: i64) -> Result<(), EditError> {
    if duration <= 0 {
        return Err(EditError::Range(
            cutlass_model::RangeError::NonPositiveDuration,
        ));
    }
    let start = source
        .start
        .0
        .checked_add(delta)
        .ok_or(EditError::Range(cutlass_model::RangeError::EndOverflow))?;
    source.start = Micros(start);
    source.duration = Micros(duration);
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, MaterialVideo, Micros, Segment, TimeRange};

    use crate::edit::EditError;
    use crate::request::{Request, TrimEdge};
    use crate::session::Session;

    /// A clip showing 1 s to 3 s of a 4 s video, at 2 s on the timeline.
    fn open() -> Session {
        let mut draft = Draft::empty("D");
        draft.config.maintrack_adsorb = false;
        draft.tracks[0].id = "main".into();
        draft.materials.videos.push(MaterialVideo {
            id: "v".into(),
            duration: Micros(4_000_000),
            ..MaterialVideo::default()
        });
        let mut session = Session::open(draft);
        let mut clip = Segment::new("a", "v", TimeRange::new(2_000_000, 2_000_000));
        clip.source_range = Some(TimeRange::new(1_000_000, 2_000_000));
        session
            .apply(
                0,
                &[Request::Place {
                    track_id: "main".into(),
                    segment: Box::new(clip),
                }],
            )
            .unwrap();
        session
    }

    fn trim(edge: TrimEdge, time: i64) -> Request {
        Request::Trim {
            segment_id: "a".into(),
            edge,
            time: Micros(time),
        }
    }

    #[test]
    fn a_trim_reaches_the_media_ends_and_no_further() {
        let mut session = open();
        assert_eq!(
            session.apply(1, &[trim(TrimEdge::Start, 999_999)]),
            Err(EditError::SourceBounds)
        );
        assert_eq!(
            session.apply(1, &[trim(TrimEdge::End, 5_000_001)]),
            Err(EditError::SourceBounds)
        );
        session
            .apply(1, &[trim(TrimEdge::Start, 1_000_000)])
            .unwrap();
        session.apply(2, &[trim(TrimEdge::End, 5_000_000)]).unwrap();
        let source = session.draft().tracks[0].segments[0].source_range.clone();
        assert_eq!(source, Some(TimeRange::new(0, 4_000_000)));
    }
}

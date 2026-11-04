//! Which lanes take which clips, and where a clip may start.

use cutlass_model::{Draft, MaterialRef, Micros, Segment, Track, TrackType};

use super::EditError;

/// Refuse a start before zero, or a lane of another kind than the clip's
/// material. A lane with no set kind, a mixed lane, or a material the pools
/// do not hold yet takes anything.
pub(super) fn check(
    draft: &Draft,
    track: &Track,
    segment: &Segment,
    start: Micros,
) -> Result<(), EditError> {
    if start.0 < 0 {
        return Err(EditError::BeforeZero);
    }
    let fits = match draft.materials.get(segment.material_id.as_str()) {
        Some(MaterialRef::Video(_) | MaterialRef::Draft(_)) => {
            matches!(track.kind, TrackType::Video | TrackType::Composition)
        }
        Some(MaterialRef::Audio(_)) => track.kind == TrackType::Audio,
        Some(MaterialRef::Text(_)) => matches!(
            track.kind,
            TrackType::Text | TrackType::TextToVideoText | TrackType::Sticker
        ),
        _ => true,
    };
    let open = matches!(
        track.kind,
        TrackType::None | TrackType::Mixed | TrackType::Other(_)
    );
    if fits || open {
        Ok(())
    } else {
        Err(EditError::LaneKind)
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, MaterialAudio, Micros, Segment, TimeRange, Track, TrackType};

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    #[test]
    fn a_clip_keeps_to_its_kind_of_lane_and_after_zero() {
        let mut draft = Draft::empty("D");
        draft.config.maintrack_adsorb = false;
        draft.tracks[0].id = "main".into();
        let mut sound = Track::main("sound");
        sound.kind = TrackType::Audio;
        draft.tracks.push(sound);
        draft.materials.audios.push(MaterialAudio {
            id: "tone".into(),
            ..MaterialAudio::default()
        });
        let mut session = Session::open(draft);
        let place = |track: &str, start: i64| Request::Place {
            track_id: track.into(),
            segment: Box::new(Segment::new("a", "tone", TimeRange::new(start, 1_000_000))),
        };
        assert_eq!(
            session.apply(0, &[place("main", 0)]),
            Err(EditError::LaneKind)
        );
        assert_eq!(
            session.apply(0, &[place("sound", -1)]),
            Err(EditError::BeforeZero)
        );
        session.apply(0, &[place("sound", 0)]).unwrap();
        let shift = |track: &str, start: i64| Request::Move {
            segment_id: "a".into(),
            track_id: track.into(),
            start: Micros(start),
        };
        assert_eq!(
            session.apply(1, &[shift("main", 0)]),
            Err(EditError::LaneKind)
        );
        assert_eq!(
            session.apply(1, &[shift("sound", -5)]),
            Err(EditError::BeforeZero)
        );
        session.apply(1, &[shift("sound", 2_000_000)]).unwrap();
    }
}

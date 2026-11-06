//! Which lanes take which clips, and where a clip may start.
//!
//! A still picture (a sticker, a GIF, a photo or an image) may also sit on a
//! sticker lane. A compound clip may also sit on a lane of the kind it was
//! made on, so one made of text keeps to text lanes.

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
        Some(MaterialRef::Video(video)) => {
            matches!(track.kind, TrackType::Video | TrackType::Composition)
                || (track.kind == TrackType::Sticker
                    && matches!(video.kind.as_str(), "sticker" | "gif" | "photo" | "image"))
        }
        Some(MaterialRef::Draft(nest)) => {
            matches!(track.kind, TrackType::Video | TrackType::Composition)
                || nest
                    .draft
                    .as_ref()
                    .and_then(|inside| inside.tracks.first())
                    .is_some_and(|host| host.kind == track.kind)
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
    fn a_still_may_sit_on_a_sticker_lane_and_a_movie_may_not() {
        use cutlass_model::MaterialVideo;

        let mut draft = Draft::empty("D");
        draft.config.maintrack_adsorb = false;
        let mut stickers = Track::main("stickers");
        stickers.kind = TrackType::Sticker;
        draft.tracks.push(stickers);
        for (id, kind) in [("heart", "sticker"), ("loop", "gif"), ("movie", "video")] {
            let mut video = MaterialVideo::new(id);
            video.kind = kind.to_owned();
            draft.materials.videos.push(video);
        }
        let mut session = Session::open(draft);
        let place = |id: &str, start: i64| Request::Place {
            track_id: "stickers".into(),
            segment: Box::new(Segment::new(id, id, TimeRange::new(start, 1_000_000))),
        };
        session.apply(0, &[place("heart", 0)]).unwrap();
        session
            .apply(session.revision(), &[place("loop", 1_000_000)])
            .unwrap();
        assert_eq!(
            session.apply(session.revision(), &[place("movie", 2_000_000)]),
            Err(EditError::LaneKind)
        );
    }

    #[test]
    fn a_compound_made_of_text_moves_on_text_lanes_and_into_a_new_one() {
        use cutlass_model::{MaterialDraft, MaterialText, TrackId};

        let mut draft = Draft::empty("D");
        draft.config.maintrack_adsorb = false;
        let mut words = Track::main("words");
        words.kind = TrackType::Text;
        let mut inside = Draft::empty("inside");
        inside.tracks = vec![words.clone()];
        words
            .segments
            .push(Segment::new("nest", "nest", TimeRange::new(0, 1_000_000)));
        draft.tracks.push(words);
        draft.materials.texts.push(MaterialText {
            id: "unused".into(),
            ..MaterialText::default()
        });
        draft.materials.drafts.push(MaterialDraft {
            id: "nest".into(),
            draft: Some(Box::new(inside)),
            ..MaterialDraft::default()
        });
        let mut session = Session::open(draft);
        session
            .apply(
                0,
                &[Request::Move {
                    segment_id: "nest".into(),
                    track_id: "words".into(),
                    start: Micros(2_000_000),
                }],
            )
            .unwrap();
        let index = crate::new_lane_index(&session.draft().tracks, &TrackType::Text);
        session
            .apply(
                session.revision(),
                &[
                    Request::InsertTrackAt {
                        track_id: TrackId::new("more"),
                        kind: TrackType::Text,
                        index,
                    },
                    Request::Move {
                        segment_id: "nest".into(),
                        track_id: "more".into(),
                        start: Micros(0),
                    },
                ],
            )
            .unwrap();
    }

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

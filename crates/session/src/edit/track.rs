use cutlass_model::{Draft, Track, TrackAttribute, TrackFlag, TrackId, TrackType};

use super::EditError;

pub fn set_track_attribute(
    draft: &mut Draft,
    track_id: &TrackId,
    bit: i32,
    on: bool,
) -> Result<(), EditError> {
    if !known(bit) {
        return Err(EditError::UnknownAttribute);
    }
    {
        let Some(track) = draft.tracks.iter_mut().find(|track| &track.id == track_id) else {
            return Err(EditError::MissingTrack);
        };
        if on {
            track.attribute.0 |= bit;
        } else {
            track.attribute.0 &= !bit;
        }
    }
    if bit == TrackAttribute::SOLO {
        apply_solo(draft);
    }
    Ok(())
}

fn known(bit: i32) -> bool {
    matches!(
        bit,
        TrackAttribute::MUTE
            | TrackAttribute::HIDE
            | TrackAttribute::LOCK
            | TrackAttribute::SOLO
            | TrackAttribute::MUTE_BY_SOLO
    )
}

/// Add an empty lane.
///
/// Video and mixed lanes use the overlay flag and sit immediately above the
/// main track. Audio sits immediately under it. Any other lane sits above
/// the first video track. An empty id, a type that is not a lane, or an id
/// already on the draft is refused. The request does not place a segment.
pub fn insert_track(
    draft: &mut Draft,
    track_id: &TrackId,
    kind: &TrackType,
) -> Result<(), EditError> {
    if track_id.as_str().is_empty() || matches!(kind, TrackType::None) {
        return Err(EditError::TrackKind);
    }
    if draft.tracks.iter().any(|track| &track.id == track_id) {
        return Err(EditError::TrackKind);
    }
    let mut track = Track::main(track_id.clone());
    track.kind = kind.clone();
    track.flag = if matches!(kind, TrackType::Video | TrackType::Mixed) {
        TrackFlag::SubVideo
    } else {
        TrackFlag::None
    };
    let index = lane_index(&draft.tracks, kind);
    draft.tracks.insert(index, track);
    Ok(())
}

/// Video overlays sit just above the main track. Audio sits just under it.
/// Other lanes sit above the first video track.
fn lane_index(tracks: &[Track], kind: &TrackType) -> usize {
    match kind {
        TrackType::Audio => tracks
            .iter()
            .position(|track| track.is_main())
            .map(|index| index.saturating_add(1))
            .unwrap_or(tracks.len()),
        TrackType::Video | TrackType::Mixed => tracks
            .iter()
            .position(|track| track.is_main())
            .unwrap_or(tracks.len()),
        _ => tracks
            .iter()
            .position(|track| {
                matches!(track.kind, TrackType::Video | TrackType::Mixed) || track.is_main()
            })
            .unwrap_or(0),
    }
}

/// Solo is not a gain. Other tracks receive muted-by-solo. A solo track does not.
fn apply_solo(draft: &mut Draft) {
    let any_solo = draft
        .tracks
        .iter()
        .any(|track| track.attribute.contains(TrackAttribute::SOLO));
    for track in &mut draft.tracks {
        let muted = any_solo && !track.attribute.contains(TrackAttribute::SOLO);
        if muted {
            track.attribute.0 |= TrackAttribute::MUTE_BY_SOLO;
        } else {
            track.attribute.0 &= !TrackAttribute::MUTE_BY_SOLO;
        }
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Micros, Segment, TimeRange, TrackId, TrackType};

    use super::{EditError, insert_track};
    use crate::Session;
    use crate::request::Request;

    fn open() -> (Session, TrackId) {
        let mut draft = cutlass_model::Draft::empty("D");
        let main = TrackId::new("main");
        draft.tracks[0].id = main.clone();
        (Session::open(draft), main)
    }

    #[test]
    fn a_video_lane_is_an_overlay_above_the_main_track() {
        let (mut session, main) = open();
        session
            .apply(
                0,
                &[Request::InsertTrack {
                    track_id: TrackId::new("over"),
                    kind: TrackType::Video,
                }],
            )
            .unwrap();
        let tracks = &session.draft().tracks;
        assert_eq!(tracks.len(), 2);
        assert!(tracks[0].is_sub_video());
        assert_eq!(tracks[0].id.as_str(), "over");
        assert!(tracks[0].segments.is_empty());
        assert!(tracks[1].is_main());
        assert_eq!(tracks[1].id, main);
    }

    #[test]
    fn an_audio_lane_sits_under_the_main_track_and_one_undo_removes_the_place() {
        let (mut session, main) = open();
        session
            .apply(
                0,
                &[
                    Request::InsertTrack {
                        track_id: TrackId::new("sound"),
                        kind: TrackType::Audio,
                    },
                    Request::Place {
                        track_id: TrackId::new("sound"),
                        segment: Box::new(Segment::new(
                            "clip",
                            "song",
                            TimeRange::new(0, 1_000_000),
                        )),
                    },
                ],
            )
            .unwrap();
        assert_eq!(session.draft().tracks.len(), 2);
        assert_eq!(session.draft().tracks[0].id, main);
        assert_eq!(session.draft().tracks[1].kind, TrackType::Audio);
        assert!(!session.draft().tracks[1].is_main());
        assert_eq!(session.draft().tracks[1].segments.len(), 1);
        assert_eq!(session.draft().duration, Micros(1_000_000));
        session.undo().unwrap();
        assert_eq!(session.draft().tracks.len(), 1);
        assert!(session.draft().tracks[0].segments.is_empty());
        assert_eq!(session.draft().duration, Micros(0));
    }

    #[test]
    fn a_repeated_track_id_leaves_the_draft() {
        let (mut session, _) = open();
        let error = session
            .apply(
                0,
                &[
                    Request::InsertTrack {
                        track_id: TrackId::new("sound"),
                        kind: TrackType::Audio,
                    },
                    Request::InsertTrack {
                        track_id: TrackId::new("sound"),
                        kind: TrackType::Audio,
                    },
                ],
            )
            .unwrap_err();
        assert_eq!(error, EditError::TrackKind);
        assert_eq!(session.draft().tracks.len(), 1);
        assert_eq!(session.revision(), 0);
    }

    #[test]
    fn an_empty_id_and_a_none_type_are_refused() {
        let mut draft = cutlass_model::Draft::empty("D");
        let error = insert_track(&mut draft, &TrackId::new(""), &TrackType::Audio).unwrap_err();
        assert_eq!(error, EditError::TrackKind);
        let error = insert_track(&mut draft, &TrackId::new("lane"), &TrackType::None).unwrap_err();
        assert_eq!(error, EditError::TrackKind);
        assert_eq!(draft.tracks.len(), 1);
    }

    #[test]
    fn a_text_lane_sits_above_video() {
        let mut draft = cutlass_model::Draft::empty("D");
        insert_track(&mut draft, &TrackId::new("words"), &TrackType::Text).unwrap();
        assert_eq!(draft.tracks[0].kind, TrackType::Text);
        assert!(draft.tracks[1].is_main());
        assert_eq!(draft.tracks[0].flag, cutlass_model::TrackFlag::None);
    }
}

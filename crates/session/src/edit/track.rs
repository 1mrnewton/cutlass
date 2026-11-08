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
    let track = new_lane(draft, track_id, kind)?;
    let index = new_lane_index(&draft.tracks, kind);
    draft.tracks.insert(index, track);
    Ok(())
}

/// Add an empty lane at `index`. The index must sit inside the kind's block
/// (`lane_fits`). The other refusals are `insert_track`'s.
pub fn insert_track_at(
    draft: &mut Draft,
    track_id: &TrackId,
    kind: &TrackType,
    index: usize,
) -> Result<(), EditError> {
    let track = new_lane(draft, track_id, kind)?;
    if !lane_fits(&draft.tracks, kind, index) {
        return Err(EditError::TrackKind);
    }
    draft.tracks.insert(index, track);
    Ok(())
}

/// A new lane of `kind` may open at `index` in `tracks`.
///
/// Audio sits under the main track and every other kind above it. With no
/// main track any index up to the end fits. An index past the end does not.
#[must_use]
pub fn lane_fits(tracks: &[Track], kind: &TrackType, index: usize) -> bool {
    if index > tracks.len() || matches!(kind, TrackType::None) {
        return false;
    }
    let Some(main) = tracks.iter().position(Track::is_main) else {
        return true;
    };
    if matches!(kind, TrackType::Audio) {
        index > main
    } else {
        index <= main
    }
}

fn new_lane(draft: &Draft, track_id: &TrackId, kind: &TrackType) -> Result<Track, EditError> {
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
    Ok(track)
}

/// Take a lane off the draft with its clips.
///
/// The main track and a locked track are refused. Each clip leaves the way
/// a delete takes it, so its marks, beats and attachments go too; linkage
/// then removes dependents on other lanes.
pub fn remove_track(draft: &mut Draft, track_id: &TrackId) -> Result<(), EditError> {
    let Some(track) = draft.tracks.iter().find(|track| &track.id == track_id) else {
        return Err(EditError::MissingTrack);
    };
    if track.is_main() {
        return Err(EditError::MainTrack);
    }
    super::ensure_unlocked(track.attribute)?;
    let ids: Vec<_> = track
        .segments
        .iter()
        .map(|segment| segment.id.clone())
        .collect();
    for id in &ids {
        super::linkage::remove(draft, id)?;
    }
    draft.tracks.retain(|track| &track.id != track_id);
    Ok(())
}

/// Name a lane. A blank name gives the lane back its default name.
pub fn rename_track(draft: &mut Draft, track_id: &TrackId, name: &str) -> Result<(), EditError> {
    let Some(track) = draft.tracks.iter_mut().find(|track| &track.id == track_id) else {
        return Err(EditError::MissingTrack);
    };
    let name = name.trim();
    track.name = name.to_owned();
    track.is_default_name = name.is_empty();
    Ok(())
}

/// Video overlays sit just above the main track. Audio sits just under it.
/// Other lanes sit above the first video track.
/// Where `InsertTrack` opens a lane of `kind`: an overlay just above the main
/// track, audio just under it, any other kind above the first video track.
#[must_use]
pub fn new_lane_index(tracks: &[Track], kind: &TrackType) -> usize {
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

    use super::{EditError, insert_track, lane_fits};
    use crate::Session;
    use crate::request::Request;
    use cutlass_model::TrackAttribute;

    fn open() -> (Session, TrackId) {
        let mut draft = crate::edit::test_draft("D");
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
    fn a_lane_opens_at_its_index_inside_its_kinds_block() {
        let (mut session, main) = open();
        session
            .apply(
                0,
                &[
                    Request::InsertTrackAt {
                        track_id: TrackId::new("words"),
                        kind: TrackType::Text,
                        index: 0,
                    },
                    Request::InsertTrackAt {
                        track_id: TrackId::new("over"),
                        kind: TrackType::Video,
                        index: 0,
                    },
                    Request::InsertTrackAt {
                        track_id: TrackId::new("sound"),
                        kind: TrackType::Audio,
                        index: 3,
                    },
                ],
            )
            .unwrap();
        let ids: Vec<&str> = session
            .draft()
            .tracks
            .iter()
            .map(|track| track.id.as_str())
            .collect();
        assert_eq!(ids, ["over", "words", main.as_str(), "sound"]);
        assert!(session.draft().tracks[0].is_sub_video());
        session.undo().unwrap();
        assert_eq!(session.draft().tracks.len(), 1);
    }

    #[test]
    fn a_lane_outside_its_block_or_past_the_end_is_refused() {
        let (mut session, _) = open();
        for (kind, index) in [
            (TrackType::Audio, 0),
            (TrackType::Video, 1),
            (TrackType::Text, 1),
            (TrackType::Audio, 2),
            (TrackType::None, 0),
        ] {
            let error = session
                .apply(
                    0,
                    &[Request::InsertTrackAt {
                        track_id: TrackId::new("lane"),
                        kind,
                        index,
                    }],
                )
                .unwrap_err();
            assert_eq!(error, EditError::TrackKind);
        }
        assert_eq!(session.draft().tracks.len(), 1);
        assert_eq!(session.revision(), 0);
        let tracks = &session.draft().tracks;
        assert!(lane_fits(tracks, &TrackType::Audio, 1));
        assert!(lane_fits(tracks, &TrackType::Sticker, 0));
        assert!(lane_fits(&[], &TrackType::Audio, 0));
        assert!(!lane_fits(&[], &TrackType::Audio, 1));
    }

    #[test]
    fn an_edit_that_empties_a_lane_takes_it_and_an_added_empty_lane_stays() {
        let (mut session, main) = open();
        session
            .apply(
                0,
                &[
                    Request::InsertTrack {
                        track_id: TrackId::new("over"),
                        kind: TrackType::Video,
                    },
                    Request::Place {
                        track_id: TrackId::new("over"),
                        segment: Box::new(Segment::new("o", "pic", TimeRange::new(0, 1_000_000))),
                    },
                    Request::InsertTrack {
                        track_id: TrackId::new("spare"),
                        kind: TrackType::Text,
                    },
                    Request::Place {
                        track_id: main.clone(),
                        segment: Box::new(Segment::new("m", "pic", TimeRange::new(0, 1_000_000))),
                    },
                ],
            )
            .unwrap();
        let before = session.draft().clone();
        session
            .apply(
                1,
                &[Request::Remove {
                    segment_id: "o".into(),
                }],
            )
            .unwrap();
        let ids: Vec<&str> = session
            .draft()
            .tracks
            .iter()
            .map(|track| track.id.as_str())
            .collect();
        assert_eq!(ids, ["spare", main.as_str()]);
        session.undo().unwrap();
        assert_eq!(session.draft(), &before);

        // Main stays when its last clip goes.
        let revision = session.revision();
        session
            .apply(
                revision,
                &[Request::Remove {
                    segment_id: "m".into(),
                }],
            )
            .unwrap();
        assert!(session.draft().tracks.iter().any(|track| track.id == main));
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
        let mut draft = crate::edit::test_draft("D");
        let error = insert_track(&mut draft, &TrackId::new(""), &TrackType::Audio).unwrap_err();
        assert_eq!(error, EditError::TrackKind);
        let error = insert_track(&mut draft, &TrackId::new("lane"), &TrackType::None).unwrap_err();
        assert_eq!(error, EditError::TrackKind);
        assert_eq!(draft.tracks.len(), 1);
    }

    #[test]
    fn a_text_lane_sits_above_video() {
        let mut draft = crate::edit::test_draft("D");
        insert_track(&mut draft, &TrackId::new("words"), &TrackType::Text).unwrap();
        assert_eq!(draft.tracks[0].kind, TrackType::Text);
        assert!(draft.tracks[1].is_main());
        assert_eq!(draft.tracks[0].flag, cutlass_model::TrackFlag::None);
    }

    fn with_sound() -> Session {
        let (mut session, _) = open();
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
        session
    }

    #[test]
    fn a_removed_lane_takes_its_clips_and_one_undo_brings_them_back() {
        let mut session = with_sound();
        session
            .apply(
                1,
                &[Request::RemoveTrack {
                    track_id: TrackId::new("sound"),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().tracks.len(), 1);
        assert_eq!(session.draft().duration, Micros(0));
        session.undo().unwrap();
        assert_eq!(session.draft().tracks.len(), 2);
        assert_eq!(session.draft().tracks[1].segments.len(), 1);
    }

    #[test]
    fn the_main_lane_a_locked_lane_and_a_missing_lane_stay() {
        let mut session = with_sound();
        let main = session.draft().tracks[0].id.clone();
        let remove = |id: &TrackId| Request::RemoveTrack {
            track_id: id.clone(),
        };
        assert_eq!(
            session.apply(1, &[remove(&main)]).unwrap_err(),
            EditError::MainTrack
        );
        assert_eq!(
            session
                .apply(1, &[remove(&TrackId::new("gone"))])
                .unwrap_err(),
            EditError::MissingTrack
        );
        session
            .apply(
                1,
                &[Request::SetTrackAttribute {
                    track_id: TrackId::new("sound"),
                    bit: TrackAttribute::LOCK,
                    on: true,
                }],
            )
            .unwrap();
        assert_eq!(
            session
                .apply(2, &[remove(&TrackId::new("sound"))])
                .unwrap_err(),
            EditError::LockedTrack
        );
        assert_eq!(session.draft().tracks.len(), 2);
    }

    #[test]
    fn a_named_lane_keeps_its_name_and_a_blank_one_goes_back_to_the_default() {
        let mut session = with_sound();
        let rename = |name: &str| Request::RenameTrack {
            track_id: TrackId::new("sound"),
            name: name.to_owned(),
        };
        session.apply(1, &[rename("  Voice ")]).unwrap();
        let track = &session.draft().tracks[1];
        assert_eq!(
            (track.name.as_str(), track.is_default_name),
            ("Voice", false)
        );
        session.apply(2, &[rename("   ")]).unwrap();
        let track = &session.draft().tracks[1];
        assert_eq!((track.name.as_str(), track.is_default_name), ("", true));
    }
}

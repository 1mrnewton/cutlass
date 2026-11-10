//! Groups, and attributes carried from one clip to others.
//!
//! Clips that share a non-empty `group_id` move by the same delta in one
//! step; the linkage pass does that. A group is made by giving the clips
//! one id, and undone by clearing it.

use cutlass_model::{Clip, Draft, SegmentId, TrackType, UniformScale};
use serde::{Deserialize, Serialize};

use super::{EditError, ensure_unlocked, locate};

/// Give each clip of `segment_ids` the group `group_id`, or take them out
/// of their groups with an empty one. Every clip must be on the timeline,
/// on a row that is not locked, or none is changed.
pub fn set_group(
    draft: &mut Draft,
    segment_ids: &[SegmentId],
    group_id: &str,
) -> Result<(), EditError> {
    let mut places = Vec::with_capacity(segment_ids.len());
    for segment_id in segment_ids {
        let (track_index, segment_index) = locate(draft, segment_id)?;
        ensure_unlocked(draft.tracks[track_index].attribute)?;
        places.push((track_index, segment_index));
    }
    for (track_index, segment_index) in places {
        group_id.clone_into(&mut draft.tracks[track_index].segments[segment_index].group_id);
    }
    Ok(())
}

/// What Copy attributes takes from a clip. Each part is pasted onto the
/// clips that have such a thing: the placement onto clips that draw, the
/// volume onto clips that sound.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClipAttributes {
    /// Position, scale, rotation, opacity and flips.
    pub clip: Option<Clip>,
    pub uniform_scale: Option<UniformScale>,
    pub volume: Option<f64>,
}

/// A clip on a row of `kind` draws a picture that can be placed.
fn draws(kind: &TrackType) -> bool {
    matches!(
        kind,
        TrackType::Video | TrackType::Mixed | TrackType::Text | TrackType::Sticker
    )
}

/// A clip on a row of `kind` has a volume.
fn sounds(kind: &TrackType) -> bool {
    matches!(kind, TrackType::Video | TrackType::Mixed | TrackType::Audio)
}

/// Paste `attributes` onto each clip of `segment_ids` that can take them.
/// Every clip must be on the timeline, on a row that is not locked, and at
/// least one must take something, or none is changed. A value that is not
/// a finite volume is refused.
pub fn paste_attributes(
    draft: &mut Draft,
    segment_ids: &[SegmentId],
    attributes: &ClipAttributes,
) -> Result<(), EditError> {
    if attributes
        .volume
        .is_some_and(|volume| !volume.is_finite() || volume < 0.0)
    {
        return Err(EditError::Volume);
    }
    let mut places = Vec::with_capacity(segment_ids.len());
    let mut taken = false;
    for segment_id in segment_ids {
        let (track_index, segment_index) = locate(draft, segment_id)?;
        let track = &draft.tracks[track_index];
        ensure_unlocked(track.attribute)?;
        let picture = draws(&track.kind) && attributes.clip.is_some();
        let sound = sounds(&track.kind) && attributes.volume.is_some();
        taken |= picture || sound;
        places.push((track_index, segment_index, picture, sound));
    }
    if !taken {
        return Err(EditError::TrackKind);
    }
    for (track_index, segment_index, picture, sound) in places {
        let segment = &mut draft.tracks[track_index].segments[segment_index];
        if picture {
            if let Some(clip) = &attributes.clip {
                segment.clip = clip.clone();
            }
            if let Some(uniform_scale) = &attributes.uniform_scale {
                segment.uniform_scale = uniform_scale.clone();
            }
        }
        if sound && let Some(volume) = attributes.volume {
            segment.volume = volume;
            if volume > 0.0 {
                segment.last_nonzero_volume = volume;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Clip, Draft, MaterialAudio, MaterialId, MaterialVideo, Micros, Point, Segment, TimeRange,
        Track, TrackAttribute, TrackId, TrackType, UniformScale,
    };

    use super::ClipAttributes;
    use crate::EditError;
    use crate::request::Request;
    use crate::session::Session;

    /// Two pictures on main and a sound under them.
    fn session() -> Session {
        let mut draft = Draft::empty("D");
        draft.config.maintrack_adsorb = false;
        draft.tracks[0].id = TrackId::new("main");
        draft.materials.videos.push(MaterialVideo::new("m"));
        for (id, start) in [("a", 0), ("b", 2_000_000)] {
            draft.tracks[0]
                .segments
                .push(Segment::new(id, "m", TimeRange::new(start, 1_000_000)));
        }
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("tone"),
            ..MaterialAudio::default()
        });
        let mut sound = Track::main("sound");
        sound.kind = TrackType::Audio;
        sound
            .segments
            .push(Segment::new("s", "tone", TimeRange::new(0, 1_000_000)));
        draft.tracks.push(sound);
        Session::open(draft)
    }

    fn clip<'a>(session: &'a Session, id: &str) -> &'a Segment {
        session
            .draft()
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .find(|segment| segment.id.as_str() == id)
            .unwrap()
    }

    fn group(ids: &[&str], group_id: &str) -> Request {
        Request::SetGroup {
            segment_ids: ids.iter().map(|id| (*id).into()).collect(),
            group_id: group_id.to_owned(),
        }
    }

    #[test]
    fn grouped_clips_move_together_and_ungrouped_ones_do_not() {
        let mut session = session();
        session.apply(0, &[group(&["a", "s"], "g1")]).unwrap();
        assert_eq!(clip(&session, "a").group_id, "g1");
        assert_eq!(clip(&session, "s").group_id, "g1");
        assert_eq!(clip(&session, "b").group_id, "");

        // The picture moved a second later takes the sound with it.
        let nudge = |start: i64| Request::Move {
            segment_id: "a".into(),
            track_id: TrackId::new("main"),
            start: Micros(start),
        };
        session.apply(1, &[nudge(500_000)]).unwrap();
        assert_eq!(clip(&session, "a").target.start, Micros(500_000));
        assert_eq!(clip(&session, "s").target.start, Micros(500_000));
        assert_eq!(clip(&session, "b").target.start, Micros(2_000_000));

        // Ungrouped, it moves alone. One undo brings the group back.
        session.apply(2, &[group(&["a", "s"], "")]).unwrap();
        assert_eq!(clip(&session, "s").group_id, "");
        session.apply(3, &[nudge(0)]).unwrap();
        assert_eq!(clip(&session, "a").target.start, Micros(0));
        assert_eq!(clip(&session, "s").target.start, Micros(500_000));
        session.undo().unwrap();
        session.undo().unwrap();
        assert_eq!(clip(&session, "s").group_id, "g1");
    }

    /// With the magnet on: `a` then `b` packed on main, and a sound `s`
    /// under `b`, grouped with it.
    fn magnetic() -> Session {
        let mut draft = session().draft().clone();
        draft.config.maintrack_adsorb = true;
        draft.tracks[0].segments[1].target = TimeRange::new(1_000_000, 1_000_000);
        draft.tracks[1].segments[0].target = TimeRange::new(1_000_000, 1_000_000);
        let mut session = Session::open(draft);
        session.apply(0, &[group(&["b", "s"], "g")]).unwrap();
        session
    }

    fn start(session: &Session, id: &str) -> i64 {
        clip(session, id).target.start.0
    }

    #[test]
    fn a_group_held_by_the_magnet_comes_back_whole() {
        // With linkage on and off: a group is not a link.
        for linked in [true, false] {
            let mut session = magnetic();
            let mut settings = session.linkage();
            settings.enabled = linked;
            session.set_linkage(settings);

            // The main clip dragged a little later: the magnet puts it
            // back, and its sound comes back with it.
            let moved = |id: &str, row: &str, to: i64| Request::Move {
                segment_id: id.into(),
                track_id: TrackId::new(row),
                start: Micros(to),
            };
            session.apply(1, &[moved("b", "main", 1_400_000)]).unwrap();
            assert_eq!(start(&session, "b"), 1_000_000, "linked {linked}");
            assert_eq!(start(&session, "s"), 1_000_000, "linked {linked}");

            // The sound dragged instead takes the main clip with it, which
            // the magnet puts back, and so the sound too.
            session.apply(2, &[moved("s", "sound", 1_700_000)]).unwrap();
            assert_eq!(start(&session, "b"), 1_000_000);
            assert_eq!(start(&session, "s"), 1_000_000);

            // The clip before it deleted: the main clip closes the gap,
            // and its sound goes with it.
            session
                .apply(
                    3,
                    &[Request::Remove {
                        segment_id: "a".into(),
                    }],
                )
                .unwrap();
            assert_eq!(start(&session, "b"), 0);
            assert_eq!(start(&session, "s"), 0);
        }
    }

    #[test]
    fn a_group_with_two_clips_on_the_main_lane_moves_its_others_once() {
        // Both main clips are in the group with the sound under the second.
        let mut draft = session().draft().clone();
        draft.config.maintrack_adsorb = true;
        draft.tracks[0].segments[1].target = TimeRange::new(1_000_000, 1_000_000);
        draft.tracks[1].segments[0].target = TimeRange::new(1_000_000, 1_000_000);
        // A clip ahead of them, to delete.
        draft.tracks[0]
            .segments
            .insert(0, Segment::new("first", "m", TimeRange::new(0, 500_000)));
        draft.tracks[0].segments[1].target = TimeRange::new(500_000, 1_000_000);
        draft.tracks[0].segments[2].target = TimeRange::new(1_500_000, 1_000_000);
        draft.tracks[1].segments[0].target = TimeRange::new(1_500_000, 1_000_000);
        let mut session = Session::open(draft);
        session.apply(0, &[group(&["a", "b", "s"], "g")]).unwrap();
        session
            .apply(
                1,
                &[Request::Remove {
                    segment_id: "first".into(),
                }],
            )
            .unwrap();
        // Both main clips closed up by half a second; the sound moved by
        // that much, not by it twice.
        assert_eq!((start(&session, "a"), start(&session, "b")), (0, 1_000_000));
        assert_eq!(start(&session, "s"), 1_000_000);
    }

    #[test]
    fn a_group_with_a_missing_or_locked_clip_changes_nothing() {
        let mut session = session();
        let before = session.draft().clone();
        assert!(session.apply(0, &[group(&["a", "nowhere"], "g")]).is_err());
        assert_eq!(session.draft(), &before);
        let mut draft = before.clone();
        draft.tracks[1].attribute = TrackAttribute(TrackAttribute::LOCK);
        let mut locked = Session::open(draft.clone());
        assert_eq!(
            locked.apply(0, &[group(&["a", "s"], "g")]).unwrap_err(),
            EditError::LockedTrack
        );
        assert_eq!(locked.draft(), &draft);
    }

    fn attributes() -> ClipAttributes {
        let placed = Clip {
            scale: Point::new(0.5, 0.5),
            rotation: 30.0,
            transform: Point::new(0.25, -0.5),
            alpha: 0.4,
            ..Clip::default()
        };
        ClipAttributes {
            clip: Some(placed),
            uniform_scale: Some(UniformScale {
                on: false,
                value: 0.5,
                rest: Default::default(),
            }),
            volume: Some(0.3),
        }
    }

    #[test]
    fn attributes_land_on_the_clips_that_can_take_them() {
        let mut session = session();
        let paste = |ids: &[&str], attributes: ClipAttributes| Request::PasteAttributes {
            segment_ids: ids.iter().map(|id| (*id).into()).collect(),
            attributes: Box::new(attributes),
        };
        let before = session.draft().clone();
        session
            .apply(0, &[paste(&["b", "s"], attributes())])
            .unwrap();
        // The picture takes the placement and the volume.
        let b = clip(&session, "b");
        assert_eq!(b.clip.rotation, 30.0);
        assert_eq!(b.clip.alpha, 0.4);
        assert_eq!(b.clip.transform, Point::new(0.25, -0.5));
        assert!(!b.uniform_scale.on);
        assert_eq!((b.volume, b.last_nonzero_volume), (0.3, 0.3));
        // The sound takes the volume and is not placed.
        let s = clip(&session, "s");
        assert_eq!(s.volume, 0.3);
        assert_eq!(s.clip, Clip::default());
        // The clip copied from, and where the clips sit, are as they were.
        assert_eq!(clip(&session, "a"), &before.tracks[0].segments[0]);
        assert_eq!(b.target, TimeRange::new(2_000_000, 1_000_000));
        session.undo().unwrap();
        assert_eq!(session.draft(), &before);

        // A muted copy mutes, and what the volume was before stays known.
        let mut quiet = attributes();
        quiet.volume = Some(0.0);
        session.apply(2, &[paste(&["b"], quiet)]).unwrap();
        let b = clip(&session, "b");
        assert_eq!(b.volume, 0.0);
        assert_eq!(b.last_nonzero_volume, 1.0);

        // A placement alone gives a sound nothing, so nothing is pasted.
        let placement = ClipAttributes {
            volume: None,
            ..attributes()
        };
        let before = session.draft().clone();
        assert_eq!(
            session.apply(3, &[paste(&["s"], placement)]).unwrap_err(),
            EditError::TrackKind
        );
        // A volume that is no number, and a clip that is not there.
        let mut bad = attributes();
        bad.volume = Some(f64::NAN);
        assert_eq!(
            session.apply(3, &[paste(&["b"], bad)]).unwrap_err(),
            EditError::Volume
        );
        assert!(
            session
                .apply(3, &[paste(&["b", "gone"], attributes())])
                .is_err()
        );
        assert_eq!(session.draft(), &before);
    }
}

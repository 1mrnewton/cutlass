//! Sync video and audio: which of the picked clips are lined up by their
//! sound, and the requests that move them once the sound has been matched.
//!
//! One picture clip stays where it is. Every other picked clip with sound
//! moves so its sound falls on the same sound in that clip. Matching the
//! sound itself is `cutlass_audio::align`; a screen reads the files and
//! hands the shifts it found to [`sync_requests`].

use cutlass_model::{Draft, MaterialRef, Micros, Segment, SegmentId, Track, TrackId, TrackType};
use cutlass_session::Request;

/// The clips a sync works on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyncPlan {
    /// The picture clip that stays where it is.
    pub reference: SegmentId,
    /// The clips lined up to it, in timeline order.
    pub others: Vec<SegmentId>,
}

/// Why the picked clips cannot be synced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncMiss {
    /// Fewer than two of the picked clips have sound that can be matched.
    TooFew,
    /// None of them is a picture clip for the others to follow.
    NoPicture,
}

impl std::fmt::Display for SyncMiss {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::TooFew => "pick a video clip and the sound that goes with it",
            Self::NoPicture => "one of the clips must be a video for the others to follow",
        })
    }
}

impl std::error::Error for SyncMiss {}

/// A clip whose sound can be matched: a movie or a sound file, playing
/// forwards at its own speed. A clip slowed, sped up or reversed plays its
/// sound at another pace, and its shift would not be one number.
fn matchable(draft: &Draft, track: &Track, segment: &Segment) -> bool {
    let media = match draft.materials.get(segment.material_id.as_str()) {
        Some(MaterialRef::Video(video)) => !video.path.is_empty(),
        Some(MaterialRef::Audio(audio)) => !audio.path.is_empty(),
        _ => false,
    };
    media
        && matches!(track.kind, TrackType::Video | TrackType::Audio)
        && !segment.reverse
        && (segment.speed - 1.0).abs() < 1.0e-9
        && segment.target.duration.0 > 0
}

/// The clips of `selection` a sync works on: the picture clip that stays,
/// which is the earliest on the main track, else the earliest picture
/// clip, and the rest.
pub fn sync_plan(draft: &Draft, selection: &[SegmentId]) -> Result<SyncPlan, SyncMiss> {
    let mut picked: Vec<(&Track, &Segment)> = draft
        .tracks
        .iter()
        .flat_map(|track| track.segments.iter().map(move |segment| (track, segment)))
        .filter(|(track, segment)| {
            selection.contains(&segment.id) && matchable(draft, track, segment)
        })
        .collect();
    if picked.len() < 2 {
        return Err(SyncMiss::TooFew);
    }
    picked.sort_by_key(|(_, segment)| segment.target.start);
    let reference = picked
        .iter()
        .filter(|(track, _)| track.kind == TrackType::Video)
        .min_by_key(|(track, segment)| (!track.is_main(), segment.target.start))
        .map(|(_, segment)| segment.id.clone())
        .ok_or(SyncMiss::NoPicture)?;
    let others = picked
        .iter()
        .map(|(_, segment)| segment.id.clone())
        .filter(|id| *id != reference)
        .collect();
    Ok(SyncPlan { reference, others })
}

/// The requests that line clips up to `reference`, as one undo step.
///
/// `shifts` gives, for each clip, how long after the reference's start its
/// own start belongs, in microseconds; negative is before. A clip stays on
/// its lane when the lane is free there. Otherwise, and for a clip on the
/// main track while the magnet would take it back, a lane of its kind
/// opens for it, named by `fresh`. A clip that would start before the
/// timeline does is left where it is and returned in the second list.
pub fn sync_requests(
    draft: &Draft,
    reference: &SegmentId,
    shifts: &[(SegmentId, i64)],
    mut fresh: impl FnMut() -> TrackId,
) -> (Vec<Request>, Vec<SegmentId>) {
    let find = |id: &SegmentId| {
        draft
            .tracks
            .iter()
            .flat_map(|track| track.segments.iter().map(move |segment| (track, segment)))
            .find(|(_, segment)| segment.id == *id)
    };
    let Some((_, anchor)) = find(reference) else {
        return (
            Vec::new(),
            shifts.iter().map(|(id, _)| id.clone()).collect(),
        );
    };
    // Where every clip sits as the moves are planned: lane, id, start, end.
    let mut placed: Vec<(TrackId, SegmentId, i64, i64)> = draft
        .tracks
        .iter()
        .flat_map(|track| {
            track.segments.iter().map(|segment| {
                (
                    track.id.clone(),
                    segment.id.clone(),
                    segment.target.start.0,
                    segment
                        .target
                        .start
                        .0
                        .saturating_add(segment.target.duration.0),
                )
            })
        })
        .collect();
    let mut requests = Vec::new();
    let mut left = Vec::new();
    for (id, shift) in shifts {
        let Some((track, segment)) = find(id) else {
            left.push(id.clone());
            continue;
        };
        let start = anchor.target.start.0.saturating_add(*shift);
        let end = start.saturating_add(segment.target.duration.0);
        if start < 0 {
            left.push(id.clone());
            continue;
        }
        if start == segment.target.start.0 {
            continue;
        }
        let packed = track.is_main() && draft.config.maintrack_adsorb;
        let clear = !placed.iter().any(|(lane, other, from, to)| {
            *lane == track.id && other != id && start < *to && *from < end
        });
        let track_id = if clear && !packed {
            track.id.clone()
        } else {
            let track_id = fresh();
            requests.push(Request::InsertTrack {
                track_id: track_id.clone(),
                kind: track.kind.clone(),
            });
            track_id
        };
        if let Some(entry) = placed.iter_mut().find(|(_, other, _, _)| other == id) {
            *entry = (track_id.clone(), id.clone(), start, end);
        }
        requests.push(Request::Move {
            segment_id: id.clone(),
            track_id,
            start: Micros(start),
        });
    }
    (requests, left)
}

#[cfg(test)]
mod tests {
    use cutlass_model::{MaterialAudio, MaterialId, MaterialVideo, TimeRange};
    use cutlass_session::Session;

    use super::*;

    fn ids(names: &[&str]) -> Vec<SegmentId> {
        names.iter().map(|name| SegmentId::new(*name)).collect()
    }

    /// A camera clip on the main track from two seconds, a second camera
    /// above it, and two recorder clips on one sound lane.
    fn draft() -> Draft {
        let mut draft = Draft::empty("D");
        draft.config.maintrack_adsorb = false;
        for id in ["cam", "cam2"] {
            draft.materials.videos.push(MaterialVideo {
                id: MaterialId::new(id),
                path: format!("/m/{id}.mov"),
                duration: Micros(60_000_000),
                ..MaterialVideo::default()
            });
        }
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("rec"),
            path: "/m/rec.wav".to_owned(),
            duration: Micros(60_000_000),
            ..MaterialAudio::default()
        });
        draft.tracks[0].segments.push(Segment::new(
            "video",
            "cam",
            TimeRange::new(2_000_000, 10_000_000),
        ));
        let mut above = Track::main("above");
        above.flag = cutlass_model::TrackFlag::SubVideo;
        above
            .segments
            .push(Segment::new("second", "cam2", TimeRange::new(0, 4_000_000)));
        draft.tracks.push(above);
        let mut sound = Track::main("sound");
        sound.kind = TrackType::Audio;
        sound
            .segments
            .push(Segment::new("take", "rec", TimeRange::new(0, 5_000_000)));
        sound.segments.push(Segment::new(
            "next",
            "rec",
            TimeRange::new(20_000_000, 5_000_000),
        ));
        draft.tracks.push(sound);
        draft
    }

    #[test]
    fn the_main_track_video_stays_and_the_rest_follow() {
        let draft = draft();
        let plan = sync_plan(&draft, &ids(&["take", "second", "video", "next"])).unwrap();
        assert_eq!(plan.reference, SegmentId::new("video"));
        assert_eq!(plan.others, ids(&["second", "take", "next"]));
        // Without the main clip, the earliest picture clip stays.
        let plan = sync_plan(&draft, &ids(&["take", "second"])).unwrap();
        assert_eq!(plan.reference, SegmentId::new("second"));
        assert_eq!(plan.others, ids(&["take"]));
    }

    #[test]
    fn too_few_clips_no_picture_and_clips_off_their_pace_are_refused() {
        let mut draft = draft();
        assert_eq!(sync_plan(&draft, &[]), Err(SyncMiss::TooFew));
        assert_eq!(sync_plan(&draft, &ids(&["video"])), Err(SyncMiss::TooFew));
        assert_eq!(
            sync_plan(&draft, &ids(&["video", "gone"])),
            Err(SyncMiss::TooFew)
        );
        assert_eq!(
            sync_plan(&draft, &ids(&["take", "next"])),
            Err(SyncMiss::NoPicture)
        );
        // A clip at another speed, or reversed, is not matched.
        draft.tracks[2].segments[0].speed = 2.0;
        assert_eq!(
            sync_plan(&draft, &ids(&["video", "take"])),
            Err(SyncMiss::TooFew)
        );
        draft.tracks[2].segments[0].speed = 1.0;
        draft.tracks[2].segments[0].reverse = true;
        assert_eq!(
            sync_plan(&draft, &ids(&["video", "take"])),
            Err(SyncMiss::TooFew)
        );
        // A title among the picked clips is passed over.
        let mut text = Track::main("text");
        text.kind = TrackType::Text;
        text.segments
            .push(Segment::new("title", "none", TimeRange::new(0, 1_000_000)));
        draft.tracks.push(text);
        assert_eq!(
            sync_plan(&draft, &ids(&["video", "title"])),
            Err(SyncMiss::TooFew)
        );
        assert!(!SyncMiss::TooFew.to_string().is_empty());
        assert!(!SyncMiss::NoPicture.to_string().is_empty());
    }

    #[test]
    fn clips_move_to_their_shift_and_take_a_new_lane_when_theirs_is_taken() {
        let draft = draft();
        let mut made = 0;
        let mut fresh = || {
            made += 1;
            TrackId::new(format!("new-{made}"))
        };
        let reference = SegmentId::new("video");
        // "take" belongs 1.5 s after the video starts; "next" belongs half a
        // second later still, over where "take" then is; "second" belongs
        // before the timeline starts.
        let shifts = [
            (SegmentId::new("take"), 1_500_000),
            (SegmentId::new("next"), 2_000_000),
            (SegmentId::new("second"), -3_000_000),
            (SegmentId::new("gone"), 0),
        ];
        let (requests, left) = sync_requests(&draft, &reference, &shifts, &mut fresh);
        assert_eq!(left, ids(&["second", "gone"]));
        let mut session = Session::open(draft.clone());
        session.apply(0, &requests).unwrap();
        let after = session.draft();
        let at = |id: &str| {
            after
                .tracks
                .iter()
                .flat_map(|track| track.segments.iter().map(move |segment| (track, segment)))
                .find(|(_, segment)| segment.id.as_str() == id)
                .map(|(track, segment)| (track.id.as_str().to_owned(), segment.target.start.0))
                .unwrap()
        };
        assert_eq!(at("video"), ("main".to_owned(), 2_000_000));
        assert_eq!(at("take"), ("sound".to_owned(), 3_500_000));
        assert_eq!(at("next"), ("new-1".to_owned(), 4_000_000));
        assert_eq!(at("second"), ("above".to_owned(), 0));
        let lane = after
            .tracks
            .iter()
            .find(|track| track.id.as_str() == "new-1")
            .unwrap();
        assert_eq!(lane.kind, TrackType::Audio);
        // One undo puts every clip back.
        session.undo().unwrap();
        assert_eq!(session.draft().tracks, draft.tracks);

        // A clip already in place asks for nothing, and an unknown
        // reference moves nothing.
        let (requests, left) = sync_requests(
            &draft,
            &reference,
            &[(SegmentId::new("take"), -2_000_000)],
            &mut fresh,
        );
        assert!(requests.is_empty() && left.is_empty());
        let (requests, left) = sync_requests(
            &draft,
            &SegmentId::new("gone"),
            &[(SegmentId::new("take"), 0)],
            &mut fresh,
        );
        assert!(requests.is_empty());
        assert_eq!(left, ids(&["take"]));
    }

    #[test]
    fn a_clip_on_the_packed_main_track_moves_to_a_lane_of_its_own() {
        let mut draft = draft();
        draft.config.maintrack_adsorb = true;
        draft.tracks[0].segments[0].target.start = Micros(0);
        draft.tracks[0].segments.push(Segment::new(
            "b-roll",
            "cam2",
            TimeRange::new(10_000_000, 4_000_000),
        ));
        let (requests, left) = sync_requests(
            &draft,
            &SegmentId::new("video"),
            &[(SegmentId::new("b-roll"), 3_000_000)],
            || TrackId::new("new"),
        );
        assert!(left.is_empty());
        assert_eq!(
            requests,
            [
                Request::InsertTrack {
                    track_id: TrackId::new("new"),
                    kind: TrackType::Video,
                },
                Request::Move {
                    segment_id: SegmentId::new("b-roll"),
                    track_id: TrackId::new("new"),
                    start: Micros(3_000_000),
                },
            ]
        );
    }
}

//! Where a library item goes when it is added at the end of its lane.
//!
//! A picture goes after the last clip on the main lane. Anything else, or a
//! picture while main is locked, goes after the last clip on the first
//! unlocked lane of its kind. With no such lane, a new lane of that kind
//! opens and the clip starts at zero.

use cutlass_model::{Draft, MetaType, Micros, Track, TrackAttribute, TrackId, TrackType};

use crate::lane::lane_category;

/// The lane an appended clip lands on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AppendLane {
    /// An existing track.
    Track(TrackId),
    /// A track of this kind that the request must insert first.
    New(TrackType),
}

/// Where an appended clip lands and when it starts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Append {
    pub lane: AppendLane,
    pub start: Micros,
}

/// The lane kind and length of a library item, or `None` when the draft
/// has no video or audio material with that id.
///
/// A video material lands by its meta type and falls back to video. An
/// audio material lands on audio.
#[must_use]
pub fn material_span(draft: &Draft, id: &str) -> Option<(TrackType, Micros)> {
    if let Some(video) = draft
        .materials
        .videos
        .iter()
        .find(|item| item.id.as_str() == id)
    {
        let kind = MetaType::parse(&video.kind)
            .and_then(lane_category)
            .unwrap_or(TrackType::Video);
        return Some((kind, video.duration));
    }
    let audio = draft
        .materials
        .audios
        .iter()
        .find(|item| item.id.as_str() == id)?;
    Some((TrackType::Audio, audio.duration))
}

/// The end of the lane a clip of `kind` is added to.
#[must_use]
pub fn append_placement(draft: &Draft, kind: TrackType) -> Append {
    let open = |track: &&Track| !track.attribute.contains(TrackAttribute::LOCK);
    let main = if kind == TrackType::Video {
        draft.main_track().filter(open)
    } else {
        None
    };
    let target = main.or_else(|| {
        draft
            .tracks
            .iter()
            .filter(open)
            .find(|track| track.kind == kind)
    });
    match target {
        Some(track) => Append {
            lane: AppendLane::Track(track.id.clone()),
            start: Micros(
                track
                    .segments
                    .iter()
                    .map(|segment| segment.target.end().0)
                    .max()
                    .unwrap_or(0),
            ),
        },
        None => Append {
            lane: AppendLane::New(kind),
            start: Micros(0),
        },
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, MaterialAudio, MaterialId, MaterialVideo, Micros, Segment, TimeRange, Track,
        TrackAttribute, TrackId, TrackType,
    };

    use super::{Append, AppendLane, append_placement, material_span};

    fn draft() -> Draft {
        let mut draft = Draft::empty("draft");
        let mut photo = MaterialVideo::new("photo");
        photo.kind = "photo".to_owned();
        photo.duration = Micros(3_000_000);
        draft.materials.videos.push(photo);
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("song"),
            duration: Micros(2_000_000),
            ..MaterialAudio::default()
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "photo", TimeRange::new(0, 1_000_000)));
        draft.tracks[0].segments.push(Segment::new(
            "b",
            "photo",
            TimeRange::new(1_500_000, 500_000),
        ));
        draft
    }

    #[test]
    fn a_picture_goes_after_the_last_main_clip() {
        let draft = draft();
        assert_eq!(
            material_span(&draft, "photo"),
            Some((TrackType::Video, Micros(3_000_000)))
        );
        assert_eq!(
            append_placement(&draft, TrackType::Video),
            Append {
                lane: AppendLane::Track(draft.tracks[0].id.clone()),
                start: Micros(2_000_000),
            }
        );
    }

    #[test]
    fn a_sound_opens_an_audio_lane_then_follows_its_last_clip() {
        let mut draft = draft();
        assert_eq!(
            material_span(&draft, "song"),
            Some((TrackType::Audio, Micros(2_000_000)))
        );
        assert_eq!(
            append_placement(&draft, TrackType::Audio),
            Append {
                lane: AppendLane::New(TrackType::Audio),
                start: Micros(0),
            }
        );
        let mut sound = Track::main("sound");
        sound.kind = TrackType::Audio;
        sound.segments.push(Segment::new(
            "s",
            "song",
            TimeRange::new(250_000, 2_000_000),
        ));
        draft.tracks.push(sound);
        assert_eq!(
            append_placement(&draft, TrackType::Audio),
            Append {
                lane: AppendLane::Track(TrackId::new("sound")),
                start: Micros(2_250_000),
            }
        );
    }

    #[test]
    fn a_locked_main_lane_sends_a_picture_to_a_new_lane() {
        let mut draft = draft();
        draft.tracks[0].attribute = TrackAttribute(TrackAttribute::LOCK);
        assert_eq!(
            append_placement(&draft, TrackType::Video).lane,
            AppendLane::New(TrackType::Video)
        );
    }

    #[test]
    fn an_unknown_material_has_no_span() {
        assert_eq!(material_span(&draft(), "missing"), None);
    }
}

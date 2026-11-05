//! Small rules about what a clip or the timeline allows.

use cutlass_model::{Draft, MaterialRef, Segment, TrackAttribute, TrackType};
use cutlass_session::Request;

/// The clip draws a picture that can turn, flip and move on the canvas:
/// video, text, a compound clip, or a sticker. A clip with no material
/// counts when its row holds titles, stickers or compounds.
#[must_use]
pub fn can_turn(draft: &Draft, segment: &Segment, kind: &TrackType) -> bool {
    match draft.materials.get(segment.material_id.as_str()) {
        Some(MaterialRef::Video(_) | MaterialRef::Text(_) | MaterialRef::Draft(_)) => true,
        Some(MaterialRef::Other(_)) => draft
            .materials
            .stickers
            .iter()
            .any(|item| item.id == segment.material_id),
        Some(_) => false,
        None => matches!(
            kind,
            TrackType::Text
                | TrackType::TextToVideoText
                | TrackType::Sticker
                | TrackType::Composition
        ),
    }
}

/// The request that flips the magnet. `None` when the main track is locked,
/// since turning it on would move the main clips.
#[must_use]
pub fn magnet_request(draft: &Draft) -> Option<Request> {
    let locked = draft
        .main_track()
        .is_some_and(|track| track.attribute.contains(TrackAttribute::LOCK));
    (!locked).then_some(Request::SetMagnet {
        on: !draft.config.maintrack_adsorb,
    })
}

#[cfg(test)]
mod tests {
    use cutlass_model::{MaterialAudio, MaterialId, MaterialVideo, TimeRange};

    use super::*;

    #[test]
    fn pictures_turn_and_sounds_do_not() {
        let mut draft = Draft::empty("D");
        draft.materials.videos.push(MaterialVideo {
            id: MaterialId::new("v"),
            ..Default::default()
        });
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("a"),
            ..Default::default()
        });
        let video = Segment::new("1", "v", TimeRange::new(0, 1));
        let sound = Segment::new("2", "a", TimeRange::new(0, 1));
        let bare = Segment::new("3", "none", TimeRange::new(0, 1));
        assert!(can_turn(&draft, &video, &TrackType::Video));
        assert!(!can_turn(&draft, &sound, &TrackType::Audio));
        assert!(can_turn(&draft, &bare, &TrackType::Sticker));
        assert!(!can_turn(&draft, &bare, &TrackType::Audio));
    }

    #[test]
    fn the_magnet_flips_unless_the_main_track_is_locked() {
        let mut draft = Draft::empty("D");
        let on = !draft.config.maintrack_adsorb;
        assert_eq!(magnet_request(&draft), Some(Request::SetMagnet { on }));
        draft.tracks[0].attribute = TrackAttribute(TrackAttribute::LOCK);
        assert_eq!(magnet_request(&draft), None);
    }
}

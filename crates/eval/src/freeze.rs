//! The Freeze toggle: which frame a clip holds, or that it plays again.

use cutlass_model::{Draft, FreezeInfo, MaterialRef, Micros, SegmentId};

use crate::freeze_timestamp;

/// Why Freeze cannot toggle this clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreezeMiss {
    /// No clip has this id.
    UnknownSegment,
    /// The clip is not video.
    NotVideo,
    /// No frame of the clip maps to a source time.
    NoFrame,
}

/// What Freeze stores on the clip's video: `None` plays it again when it is
/// frozen; otherwise the frame under `playhead`, or the clip's first frame
/// when the playhead is elsewhere.
///
/// The hold is stored on the video, so another clip that uses the same
/// media holds the same frame.
pub fn freeze_toggle(
    draft: &Draft,
    segment_id: &SegmentId,
    playhead: Micros,
) -> Result<Option<FreezeInfo>, FreezeMiss> {
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == segment_id)
        .ok_or(FreezeMiss::UnknownSegment)?;
    let Some(MaterialRef::Video(video)) = draft.materials.get(segment.material_id.as_str()) else {
        return Err(FreezeMiss::NotVideo);
    };
    if video.freeze.is_some() {
        return Ok(None);
    }
    let start = segment.target.start;
    let inside = playhead.0 >= start.0 && playhead.0 < segment.target.end().0;
    let time = if inside { playhead } else { start };
    let timestamp = freeze_timestamp(draft, segment, time).ok_or(FreezeMiss::NoFrame)?;
    Ok(Some(FreezeInfo {
        source_material_id: video.id.as_str().to_owned(),
        source_material_path: video.path.clone(),
        reverse_material_path: video.reverse_path.clone(),
        timestamp,
        rest: Default::default(),
    }))
}

#[cfg(test)]
mod tests {
    use cutlass_model::{MaterialVideo, Segment, TimeRange};

    use super::*;

    fn draft(material: Option<MaterialVideo>) -> Draft {
        let mut draft = Draft::empty("D");
        let mut segment = Segment::new("a", "v", TimeRange::new(1_000_000, 2_000_000));
        segment.source_range = Some(TimeRange::new(0, 2_000_000));
        draft.tracks[0].segments.push(segment);
        if let Some(video) = material {
            draft.materials.videos.push(video);
        }
        draft
    }

    fn video() -> MaterialVideo {
        MaterialVideo {
            id: cutlass_model::MaterialId::new("v"),
            path: "/clip.mov".to_owned(),
            ..Default::default()
        }
    }

    #[test]
    fn the_playhead_frame_is_held_or_the_first_when_outside() {
        let draft = draft(Some(video()));
        let id = SegmentId::new("a");
        let held = freeze_toggle(&draft, &id, Micros(1_500_000))
            .unwrap()
            .unwrap();
        assert_eq!(held.timestamp, Micros(500_000));
        assert_eq!(held.source_material_path, "/clip.mov");
        let first = freeze_toggle(&draft, &id, Micros(9_000_000))
            .unwrap()
            .unwrap();
        assert_eq!(first.timestamp, Micros(0));
    }

    #[test]
    fn a_frozen_clip_plays_again() {
        let mut frozen = video();
        frozen.freeze = Some(FreezeInfo::default());
        let draft = draft(Some(frozen));
        assert_eq!(
            freeze_toggle(&draft, &SegmentId::new("a"), Micros(0)),
            Ok(None)
        );
    }

    #[test]
    fn only_video_on_the_timeline_freezes() {
        let draft = draft(None);
        assert_eq!(
            freeze_toggle(&draft, &SegmentId::new("a"), Micros(0)),
            Err(FreezeMiss::NotVideo)
        );
        assert_eq!(
            freeze_toggle(&draft, &SegmentId::new("z"), Micros(0)),
            Err(FreezeMiss::UnknownSegment)
        );
    }
}

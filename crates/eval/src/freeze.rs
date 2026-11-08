//! Freeze: which frame a still clip holds, and where it goes.

use std::path::Path;

use cutlass_encode::{
    BitDepth, CodecId, CodecParams, ColorType, EncodeFactory, PictureInput, StillImage,
};
use cutlass_engine::{ReaderFactory, decoded_picture};
use cutlass_model::{Draft, FreezeInfo, MaterialRef, Micros, SegmentId, TimeRange};
use cutlass_picture::Picture;

use crate::freeze_timestamp;
use crate::media::orient;

/// A playhead this close to a clip's edge freezes at that edge: the still
/// goes before or after the clip, which is not split.
pub const FREEZE_EDGE: Micros = Micros(33_000);

/// How long a new still lasts.
pub const FREEZE_DURATION: Micros = Micros(3_000_000);

/// Why Freeze cannot hold a frame of this clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreezeMiss {
    /// No clip has this id.
    UnknownSegment,
    /// The clip is not video.
    NotVideo,
    /// No frame of the clip maps to a source time.
    NoFrame,
    /// The playhead is not on the clip.
    Outside,
    /// The clip already holds one frame.
    AlreadyHeld,
}

/// Where a still goes and the frame it holds.
#[derive(Clone, Debug, PartialEq)]
pub struct FreezePoint {
    /// Timeline time the still starts at: the playhead, or the clip's start
    /// or end when the playhead is within [`FREEZE_EDGE`] of it.
    pub at: Micros,
    /// The hold to store on the still's own video.
    pub freeze: FreezeInfo,
}

/// The frame under `playhead` on the clip, and where its still goes.
///
/// The playhead must be on the clip. Near an edge the still takes the
/// clip's first or last frame and goes at that edge. The reversed file is
/// named only when the clip is reversed.
pub fn freeze_point(
    draft: &Draft,
    segment_id: &SegmentId,
    playhead: Micros,
) -> Result<FreezePoint, FreezeMiss> {
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
        return Err(FreezeMiss::AlreadyHeld);
    }
    if video.kind == "photo" {
        return Err(FreezeMiss::NotVideo);
    }
    let start = segment.target.start.0;
    let end = segment.target.end().0;
    if playhead.0 < start || playhead.0 > end {
        return Err(FreezeMiss::Outside);
    }
    let (at, frame) = if playhead.0 - start <= FREEZE_EDGE.0 {
        (start, start)
    } else if end - playhead.0 <= FREEZE_EDGE.0 {
        // The last frame shown is just before the end.
        (end, (end - 1).max(start))
    } else {
        (playhead.0, playhead.0)
    };
    let timestamp = freeze_timestamp(draft, segment, Micros(frame)).ok_or(FreezeMiss::NoFrame)?;
    Ok(FreezePoint {
        at: Micros(at),
        freeze: FreezeInfo {
            source_material_id: video.id.as_str().to_owned(),
            source_material_path: video.path.clone(),
            reverse_material_path: if segment.reverse {
                video.reverse_path.clone()
            } else {
                String::new()
            },
            timestamp,
            rest: Default::default(),
        },
    })
}

/// A held frame saved as a PNG, upright as the file asks it to be shown.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrozenImage {
    pub width: u32,
    pub height: u32,
    pub png: Vec<u8>,
}

/// The frame of the file at `path` at source time `at`, encoded as a PNG
/// for a freeze still. `None` when the file cannot be read there or no PNG
/// encoder is registered.
#[must_use]
pub fn freeze_image(
    media: &ReaderFactory,
    encoders: &EncodeFactory,
    path: &Path,
    at: Micros,
) -> Option<FrozenImage> {
    let mut opened = media.open(path).ok()?;
    let display = opened.info().display;
    let length = opened
        .info()
        .duration
        .map_or(i64::MAX / 2, |end| end.0.max(1));
    let held = opened.decode_at(at, &TimeRange::new(0, length)).ok()?;
    let Picture::Cpu(bytes) = decoded_picture(&held.frame).ok()? else {
        return None;
    };
    let bytes = orient(bytes, display)?;
    let (width, height) = (bytes.width(), bytes.height());
    let samples = bytes.rgba().iter().copied().map(u16::from).collect();
    let image = StillImage::from_samples(width, height, ColorType::Rgba, BitDepth::Eight, samples)
        .checked()
        .ok()?;
    let mut encoder = encoders
        .open(&CodecParams::video(CodecId::Png, width, height))
        .ok()?;
    let packets = encoder.push_picture(&PictureInput { image }).ok()?;
    let png = packets.into_iter().next()?.data;
    Some(FrozenImage { width, height, png })
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
    fn the_still_holds_the_playhead_frame_or_an_edge_frame_at_that_edge() {
        let draft = draft(Some(video()));
        let id = SegmentId::new("a");
        let middle = freeze_point(&draft, &id, Micros(1_500_000)).unwrap();
        assert_eq!(middle.at, Micros(1_500_000));
        assert_eq!(middle.freeze.timestamp, Micros(500_000));
        assert_eq!(middle.freeze.source_material_path, "/clip.mov");
        let near_start = freeze_point(&draft, &id, Micros(1_020_000)).unwrap();
        assert_eq!(near_start.at, Micros(1_000_000));
        assert_eq!(near_start.freeze.timestamp, Micros(0));
        let near_end = freeze_point(&draft, &id, Micros(2_990_000)).unwrap();
        assert_eq!(near_end.at, Micros(3_000_000));
        assert_eq!(near_end.freeze.timestamp, Micros(1_999_999));
        assert_eq!(
            freeze_point(&draft, &id, Micros(9_000_000)),
            Err(FreezeMiss::Outside)
        );
        assert_eq!(
            freeze_point(&draft, &id, Micros(900_000)),
            Err(FreezeMiss::Outside)
        );
    }

    #[test]
    fn the_reversed_file_is_named_only_for_a_reversed_clip() {
        let mut reversible = video();
        reversible.reverse_path = "/clip.reversed.mov".to_owned();
        let mut draft = draft(Some(reversible));
        let id = SegmentId::new("a");
        let forward = freeze_point(&draft, &id, Micros(1_500_000)).unwrap();
        assert_eq!(forward.freeze.reverse_material_path, "");
        draft.tracks[0].segments[0].reverse = true;
        let backward = freeze_point(&draft, &id, Micros(1_500_000)).unwrap();
        assert_eq!(backward.freeze.reverse_material_path, "/clip.reversed.mov");
    }

    #[test]
    fn a_held_clip_or_one_that_is_not_video_is_refused() {
        let mut frozen = video();
        frozen.freeze = Some(FreezeInfo::default());
        let draft_held = draft(Some(frozen));
        assert_eq!(
            freeze_point(&draft_held, &SegmentId::new("a"), Micros(1_500_000)),
            Err(FreezeMiss::AlreadyHeld)
        );
        let bare = draft(None);
        assert_eq!(
            freeze_point(&bare, &SegmentId::new("a"), Micros(1_500_000)),
            Err(FreezeMiss::NotVideo)
        );
        assert_eq!(
            freeze_point(&bare, &SegmentId::new("z"), Micros(0)),
            Err(FreezeMiss::UnknownSegment)
        );
    }
}

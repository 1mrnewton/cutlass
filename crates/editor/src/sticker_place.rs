//! Placing a sticker: how long it lasts, how large it starts, and the
//! requests that add its library item the first time and its clip, so the
//! desktop and the phone place a sticker the same way.

use cutlass_model::{Draft, MaterialId, MaterialVideo, MetaType, Micros, SegmentId, TrackId};
use cutlass_session::{ImportedMedium, Request};

use crate::{MediaPlace, media_requests};

/// How long a still sticker lasts when it is placed.
pub const STICKER_LENGTH: Micros = Micros(3_000_000);

/// A placed sticker's size against the canvas it would fill.
pub const STICKER_SCALE: f64 = 0.4;

/// How long a sticker lasts when it is placed: an animation of `seconds`
/// a round plays as many whole rounds as fit in [`STICKER_LENGTH`], at
/// least one, so its clip ends where a round ends; a still lasts
/// [`STICKER_LENGTH`].
#[must_use]
pub fn sticker_length(seconds: Option<f64>) -> Micros {
    let Some(seconds) = seconds.filter(|seconds| seconds.is_finite() && *seconds > 0.0) else {
        return STICKER_LENGTH;
    };
    let round = (seconds * 1_000_000.0).round();
    let rounds = (STICKER_LENGTH.0 as f64 / round).floor().max(1.0);
    Micros((round * rounds) as i64)
}

/// The sticker file stored at `stored` in the project, `width` by
/// `height` pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StickerFile<'a> {
    pub stored: &'a str,
    pub width: u32,
    pub height: u32,
}

/// Where a sticker goes and its new ids. `material_id` names the library
/// item only when the file is not in the library yet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StickerPlace {
    pub track_id: TrackId,
    /// Open a sticker row as `track_id` first.
    pub new_lane: bool,
    pub segment_id: SegmentId,
    pub material_id: MaterialId,
    pub start: Micros,
    pub length: Micros,
}

/// The requests that place `file` as a sticker, as one undo step: its
/// library item the first time (later stickers of the same file share
/// it), the row when it is new, the clip, and its starting size.
#[must_use]
pub fn sticker_requests(
    draft: &Draft,
    file: StickerFile<'_>,
    place: &StickerPlace,
    linkage: bool,
) -> Vec<Request> {
    let mut requests = Vec::new();
    let known = draft
        .materials
        .videos
        .iter()
        .find(|video| video.path == file.stored)
        .map(|video| video.id.clone());
    let material_id = known.unwrap_or_else(|| {
        let mut material = MaterialVideo::new(place.material_id.as_str());
        material.kind = MetaType::Sticker.as_str().to_owned();
        file.stored.clone_into(&mut material.path);
        // A picture has no length of its own; an animation plays round
        // for as long as its clip lasts.
        material.duration = MaterialVideo::MAPPING_DURATION.max(place.length);
        material.width = i32::try_from(file.width).unwrap_or(0);
        material.height = i32::try_from(file.height).unwrap_or(0);
        requests.push(Request::Import {
            medium: ImportedMedium::Video(Box::new(material)),
        });
        place.material_id.clone()
    });
    let media = MediaPlace {
        track_id: place.track_id.clone(),
        segment_id: place.segment_id.clone(),
        material_id: material_id.as_str().to_owned(),
        start: place.start,
        duration: place.length,
        new_lane: place.new_lane.then_some(cutlass_model::TrackType::Sticker),
    };
    requests.extend(media_requests(draft, &media, linkage));
    requests.push(Request::SetScale {
        segment_id: place.segment_id.clone(),
        scale: STICKER_SCALE,
    });
    requests
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, Micros, TrackType};
    use cutlass_session::{Request, Session};

    use super::*;

    #[test]
    fn an_animation_lasts_whole_rounds_and_a_still_the_set_length() {
        assert_eq!(sticker_length(Some(13.0 / 6.0)), Micros(2_166_667));
        assert_eq!(sticker_length(Some(0.9)), Micros(2_700_000));
        assert_eq!(sticker_length(Some(5.0)), Micros(5_000_000));
        assert_eq!(sticker_length(None), STICKER_LENGTH);
        assert_eq!(sticker_length(Some(f64::NAN)), STICKER_LENGTH);
    }

    #[test]
    fn a_second_sticker_of_the_same_file_shares_its_library_item() {
        let mut session = Session::open(Draft::empty("D"));
        let file = StickerFile {
            stored: "Resources/sticker-star.png",
            width: 512,
            height: 512,
        };
        let place = |n: u32, start: i64, new_lane: bool| StickerPlace {
            track_id: TrackId::new("track-s"),
            new_lane,
            segment_id: SegmentId::new(format!("seg-{n}")),
            material_id: MaterialId::new(format!("mat-{n}")),
            start: Micros(start),
            length: STICKER_LENGTH,
        };
        let first = sticker_requests(session.draft(), file, &place(1, 0, true), false);
        assert!(matches!(first[0], Request::Import { .. }));
        session.apply(session.revision(), &first).unwrap();
        let second = sticker_requests(session.draft(), file, &place(2, 4_000_000, false), false);
        assert!(
            !second
                .iter()
                .any(|request| matches!(request, Request::Import { .. }))
        );
        session.apply(session.revision(), &second).unwrap();
        let draft = session.draft();
        assert_eq!(draft.materials.videos.len(), 1);
        let row = draft
            .tracks
            .iter()
            .find(|track| track.kind == TrackType::Sticker)
            .unwrap();
        assert_eq!(row.segments.len(), 2);
        assert!(
            row.segments
                .iter()
                .all(|segment| segment.material_id.as_str() == "mat-1")
        );
        assert!(
            row.segments
                .iter()
                .all(|segment| (segment.clip.scale.x - STICKER_SCALE).abs() < 1e-9)
        );
    }
}

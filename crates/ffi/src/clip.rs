//! The clip tools a phone bar offers on a selected clip: what each can do
//! to the clip and how the clip stands, and the requests they send.
//!
//! Each tool is one undo step. The rules are the desktop's: a mirror, turn,
//! crop or opacity belongs on a picture, a fade or reverse on a picture or a
//! sound, a freeze or an extract on a video, a beat on a sound row.

use cutlass_model::{
    Crop, Draft, MaterialRef, Micros, Segment, SegmentId, TrackAttribute, TrackId, TrackType,
    segment_beats, source_time_at, track_holds_beats, user_beats,
};
use cutlass_session::Request;

use crate::error::FfiError;

/// How a selected clip stands, and which clip tools can change it.
#[derive(Clone, Debug, PartialEq)]
pub struct PhoneClipState {
    /// A picture that mirrors, turns and fades out of sight.
    pub picture: bool,
    /// Flipped left to right.
    pub mirrored: bool,
    /// Degrees, as stored.
    pub rotation: f64,
    /// A video whose frame can be cropped.
    pub can_crop: bool,
    /// The crop ratio the clip keeps: empty for the whole frame, one of
    /// [`CROP_RATIOS`], or `custom` for a crop made elsewhere.
    pub crop: String,
    /// How solid the picture is: 1 solid, 0 clear.
    pub opacity: f64,
    /// A picture or a sound, which plays backward and fades.
    pub plays: bool,
    pub reversed: bool,
    pub fade_in: Micros,
    pub fade_out: Micros,
    /// A video that is not already a still.
    pub can_freeze: bool,
    /// A video whose sound still plays from it.
    pub can_extract: bool,
    /// A video whose sound was extracted to a clip of its own.
    pub extracted: bool,
    /// On a row that keeps beats.
    pub can_beat: bool,
    /// A beat sits on the clip at the time asked about.
    pub beat_here: bool,
    /// The clip is on the main row.
    pub on_main: bool,
    /// A picture on another row, which can move to main.
    pub can_splice: bool,
}

/// The crop ratios the Crop panel offers after the whole frame.
pub const CROP_RATIOS: [&str; 6] = ["1:1", "4:5", "9:16", "16:9", "4:3", "3:4"];

/// The clip `segment_id` and its row.
pub(crate) fn locate<'a>(
    draft: &'a Draft,
    segment_id: &SegmentId,
) -> Option<(&'a Segment, &'a cutlass_model::Track)> {
    draft.tracks.iter().find_map(|track| {
        track
            .segments
            .iter()
            .find(|segment| &segment.id == segment_id)
            .map(|segment| (segment, track))
    })
}

fn refused(reason: &str) -> FfiError {
    FfiError::Refused {
        reason: reason.to_owned(),
    }
}

/// The clip and its row, refused when no row holds it or its row is locked.
pub(crate) fn unlocked<'a>(
    draft: &'a Draft,
    segment_id: &SegmentId,
) -> Result<(&'a Segment, &'a cutlass_model::Track), FfiError> {
    let (segment, track) =
        locate(draft, segment_id).ok_or_else(|| refused("the clip is not on the timeline"))?;
    if track.attribute.contains(TrackAttribute::LOCK) {
        return Err(refused("the clip's row is locked"));
    }
    Ok((segment, track))
}

/// How `segment_id` stands at `time`. `None` when no row holds it.
#[must_use]
pub fn clip_state(draft: &Draft, segment_id: &SegmentId, time: Micros) -> Option<PhoneClipState> {
    let (segment, track) = locate(draft, segment_id)?;
    let material = draft.materials.get(segment.material_id.as_str());
    let video = match material {
        Some(MaterialRef::Video(video)) => Some(video),
        _ => None,
    };
    let picture = cutlass_editor::can_turn(draft, segment, &track.kind);
    let plays = matches!(
        material,
        Some(MaterialRef::Video(_) | MaterialRef::Audio(_))
    );
    let (fade_in, fade_out) = fades(draft, segment);
    let can_beat = track_holds_beats(&track.kind);
    let beat_here = can_beat && beat_at(draft, segment, time).is_some();
    let extracted = cutlass_engine::original_sound_is_separate(draft, segment);
    Some(PhoneClipState {
        picture,
        mirrored: picture && segment.clip.flip.horizontal,
        rotation: segment.clip.rotation,
        can_crop: video.is_some(),
        crop: video.map_or_else(String::new, |video| {
            crop_name(&video.crop, video.width, video.height)
        }),
        opacity: segment.clip.alpha,
        plays,
        reversed: plays && segment.reverse,
        fade_in: Micros(fade_in),
        fade_out: Micros(fade_out),
        can_freeze: video.is_some_and(|video| video.freeze.is_none() && video.kind != "photo"),
        can_extract: video.is_some_and(|video| video.kind != "photo") && !extracted,
        extracted,
        can_beat,
        beat_here,
        on_main: track.is_main(),
        can_splice: !track.is_main() && track.kind == TrackType::Video && video.is_some(),
    })
}

/// The fade-in and fade-out lengths stored on the clip, in microseconds.
pub(crate) fn fades(draft: &Draft, segment: &Segment) -> (i64, i64) {
    for id in &segment.extra_material_refs {
        if let Some(MaterialRef::Fade(fade)) = draft.materials.get(id.as_str()) {
            return (fade.fade_in_duration.0, fade.fade_out_duration.0);
        }
    }
    (0, 0)
}

/// The stored source time of the beat within a frame of `time` on the clip.
fn beat_at(draft: &Draft, segment: &Segment, time: Micros) -> Option<i64> {
    let raw = segment_beats(&draft.materials, segment)?;
    let frame = frame_length(draft);
    user_beats(raw).into_iter().find(|source| {
        cutlass_model::beat_timeline_time(segment, *source)
            .is_some_and(|shown| (shown.0 - time.0).abs() < frame)
    })
}

fn frame_length(draft: &Draft) -> i64 {
    let fps = if draft.fps.is_finite() && draft.fps > 0.0 {
        draft.fps
    } else {
        30.0
    };
    (1_000_000.0 / fps).round() as i64
}

/// The request that adds a beat at `time` on a sound clip, or takes off
/// the beat already within a frame of it. `fresh` names a new beats material.
pub(crate) fn beat_request(
    draft: &Draft,
    segment_id: &SegmentId,
    time: Micros,
    fresh: &mut dyn FnMut(&str) -> String,
) -> Result<Request, FfiError> {
    let (segment, track) = unlocked(draft, segment_id)?;
    if !track_holds_beats(&track.kind) {
        return Err(refused("a beat belongs on a sound"));
    }
    if let Some(source) = beat_at(draft, segment, time) {
        return Ok(Request::RemoveBeat {
            segment_id: segment_id.clone(),
            source: Micros(source),
        });
    }
    let source =
        source_time_at(segment, time).ok_or_else(|| refused("move the playhead onto the clip"))?;
    Ok(Request::AddBeat {
        segment_id: segment_id.clone(),
        source: Micros(source),
        material_id: cutlass_model::MaterialId::new(fresh("beat")),
    })
}

/// A quarter turn clockwise from `degrees`.
#[must_use]
pub fn clockwise_quarter(degrees: f64) -> f64 {
    let current = if degrees.is_finite() { degrees } else { 0.0 };
    (current - 90.0).rem_euclid(360.0)
}

/// The middle of a `width` by `height` frame at `ratio` (`w:h`), as a crop
/// quad in fractions of the frame, y down. An empty ratio is the whole
/// frame. `None` for a ratio that does not read or a frame with no size.
#[must_use]
pub fn crop_for(ratio: &str, width: i32, height: i32) -> Option<Crop> {
    if ratio.is_empty() {
        return Some(Crop::default());
    }
    let (w, h) = ratio.split_once(':')?;
    let (w, h) = (w.trim().parse::<f64>().ok()?, h.trim().parse::<f64>().ok()?);
    if !(w > 0.0 && h > 0.0 && width > 0 && height > 0) {
        return None;
    }
    let want = w / h;
    let have = f64::from(width) / f64::from(height);
    let (across, down) = if want < have {
        (want / have, 1.0)
    } else {
        (1.0, have / want)
    };
    let (left, top) = ((1.0 - across) / 2.0, (1.0 - down) / 2.0);
    let (right, bottom) = (left + across, top + down);
    Some(Crop {
        upper_left_x: left,
        upper_left_y: top,
        upper_right_x: right,
        upper_right_y: top,
        lower_left_x: left,
        lower_left_y: bottom,
        lower_right_x: right,
        lower_right_y: bottom,
    })
}

/// The ratio a stored crop keeps, as [`PhoneClipState::crop`] names it.
fn crop_name(crop: &Crop, width: i32, height: i32) -> String {
    let near = |a: &Crop, b: &Crop| {
        [
            (a.upper_left_x, b.upper_left_x),
            (a.upper_left_y, b.upper_left_y),
            (a.upper_right_x, b.upper_right_x),
            (a.upper_right_y, b.upper_right_y),
            (a.lower_left_x, b.lower_left_x),
            (a.lower_left_y, b.lower_left_y),
            (a.lower_right_x, b.lower_right_x),
            (a.lower_right_y, b.lower_right_y),
        ]
        .iter()
        .all(|(x, y)| (x - y).abs() < 1e-6)
    };
    if *crop == Crop::default() {
        return String::new();
    }
    CROP_RATIOS
        .iter()
        .find(|ratio| crop_for(ratio, width, height).is_some_and(|made| near(&made, crop)))
        .map_or_else(|| "custom".to_owned(), |ratio| (*ratio).to_owned())
}

/// The request that crops the video clip to the middle at `ratio`, or
/// shows its whole frame for an empty ratio.
pub(crate) fn crop_request(
    draft: &Draft,
    segment_id: &SegmentId,
    ratio: &str,
) -> Result<Request, FfiError> {
    let (segment, _) = unlocked(draft, segment_id)?;
    let Some(MaterialRef::Video(video)) = draft.materials.get(segment.material_id.as_str()) else {
        return Err(refused("a crop belongs on a video or a photo"));
    };
    let crop = crop_for(ratio, video.width, video.height)
        .ok_or_else(|| refused("that ratio does not fit the clip"))?;
    Ok(Request::SetCrop {
        segment_id: segment_id.clone(),
        crop,
    })
}

/// The main row's id.
pub(crate) fn main_row(draft: &Draft) -> Result<TrackId, FfiError> {
    draft
        .tracks
        .iter()
        .find(|track| track.is_main())
        .map(|track| track.id.clone())
        .ok_or_else(|| refused("the project has no main row"))
}

/// The first unlocked picture row above main with room from `at` for
/// `length`.
pub(crate) fn overlay_row(draft: &Draft, at: Micros, length: Micros) -> Option<TrackId> {
    let end = at.0.saturating_add(length.0);
    draft
        .tracks
        .iter()
        .find(|track| {
            track.kind == TrackType::Video
                && !track.is_main()
                && !track.attribute.contains(TrackAttribute::LOCK)
                && track
                    .segments
                    .iter()
                    .all(|segment| segment.target.end().0 <= at.0 || segment.target.start.0 >= end)
        })
        .map(|track| track.id.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_square_crop_keeps_the_middle_of_a_wide_frame() {
        let crop = crop_for("1:1", 1920, 1080).unwrap();
        let side = 1080.0 / 1920.0;
        assert!((crop.upper_left_x - (1.0 - side) / 2.0).abs() < 1e-9);
        assert!((crop.lower_right_x - (1.0 + side) / 2.0).abs() < 1e-9);
        assert_eq!((crop.upper_left_y, crop.lower_right_y), (0.0, 1.0));
        assert_eq!(crop_name(&crop, 1920, 1080), "1:1");
    }

    #[test]
    fn a_wide_crop_keeps_the_middle_band_of_a_tall_frame() {
        let crop = crop_for("16:9", 1080, 1920).unwrap();
        assert_eq!((crop.upper_left_x, crop.lower_right_x), (0.0, 1.0));
        let band = (1080.0 / 1920.0) / (16.0 / 9.0);
        assert!((crop.lower_left_y - crop.upper_left_y - band).abs() < 1e-9);
        assert_eq!(crop_name(&crop, 1080, 1920), "16:9");
    }

    #[test]
    fn the_whole_frame_is_the_empty_crop_and_a_bad_ratio_is_refused() {
        assert_eq!(crop_for("", 10, 10), Some(Crop::default()));
        assert_eq!(crop_name(&Crop::default(), 10, 10), "");
        for bad in ["square", "0:1", "1:-1", "1:"] {
            assert_eq!(crop_for(bad, 10, 10), None, "{bad}");
        }
        assert_eq!(crop_for("1:1", 0, 10), None);
        let mut odd = crop_for("1:1", 1920, 1080).unwrap();
        odd.upper_left_x += 0.01;
        assert_eq!(crop_name(&odd, 1920, 1080), "custom");
    }

    #[test]
    fn four_quarter_turns_come_back_upright() {
        let mut degrees = 0.0;
        for expected in [270.0, 180.0, 90.0, 0.0] {
            degrees = clockwise_quarter(degrees);
            assert_eq!(degrees, expected);
        }
        assert_eq!(clockwise_quarter(f64::NAN), 270.0);
    }
}

use cutlass_model::{CanvasRatio, Draft, MaterialCanvas, MaterialId, MaterialRef, SegmentId};

use super::speed::locate;
use super::{EditError, ensure_unlocked};
use crate::request::Request;

/// The largest canvas side the preview keeps. A side outside 1 through this is refused.
const MAX_CANVAS_SIDE: i32 = 7_680;

/// Store the color under every layer.
///
/// An empty string clears it, and the preview stays clear. Six hex digits,
/// with an optional `#`, are opaque. Eight hex digits are `AARRGGBB`, with
/// an optional `#` or `0x`. The canvas size and ratio stay.
pub fn set_background(draft: &mut Draft, color: &str) -> Result<(), EditError> {
    let color = color.trim();
    if color.is_empty() {
        draft.canvas_config.background = None;
        return Ok(());
    }
    if !canvas_color(color) {
        return Err(EditError::CanvasColor);
    }
    draft.canvas_config.background = Some(color.to_owned());
    Ok(())
}

/// Store the frame size and the ratio name.
///
/// Each side is from 1 through 7680. The ratio is one of the project shapes,
/// including `original` and `custom`. The background and the frame rate stay.
pub fn set_size(draft: &mut Draft, width: i32, height: i32, ratio: &str) -> Result<(), EditError> {
    if !(1..=MAX_CANVAS_SIDE).contains(&width) || !(1..=MAX_CANVAS_SIDE).contains(&height) {
        return Err(EditError::CanvasSize);
    }
    if CanvasRatio::parse(ratio).as_str() != ratio {
        return Err(EditError::CanvasRatio);
    }
    draft.canvas_config.width = width;
    draft.canvas_config.height = height;
    draft.canvas_config.ratio = ratio.to_owned();
    Ok(())
}

/// A ratio the canvas can take and the frame it stores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanvasShape {
    pub ratio: &'static str,
    pub width: i32,
    pub height: i32,
}

const fn shape(ratio: &'static str, width: i32, height: i32) -> CanvasShape {
    CanvasShape {
        ratio,
        width,
        height,
    }
}

/// The named shapes, in the order a picker lists them. 1920×1080 is 16:9.
pub const CANVAS_SHAPES: [CanvasShape; 13] = [
    shape("16:9", 1_920, 1_080),
    shape("9:16", 1_080, 1_920),
    shape("1:1", 1_080, 1_080),
    shape("4:3", 1_440, 1_080),
    shape("3:4", 1_080, 1_440),
    shape("4:5", 1_080, 1_350),
    shape("2:1", 1_920, 960),
    shape("21:9", 2_520, 1_080),
    shape("2.35:1", 1_920, 817),
    shape("1.85:1", 1_920, 1_038),
    shape("1:2", 960, 1_920),
    shape("128:27", 5_120, 1_080),
    shape("1.125:2.436", 1_125, 2_436),
];

/// The frame `ratio` gives the canvas of `draft`.
///
/// `original` is the upright size of the first picture on the main lane,
/// scaled down to fit 7680 on the long side, or 1920×1080 when the lane
/// has no picture. A named ratio is its shape. `custom` and an unknown
/// name are refused.
pub fn canvas_for(draft: &Draft, ratio: &str) -> Result<CanvasShape, EditError> {
    if ratio == CanvasRatio::Original.as_str() {
        let (width, height) = first_picture(draft).unwrap_or((1_920, 1_080));
        return Ok(shape(CanvasRatio::Original.as_str(), width, height));
    }
    CANVAS_SHAPES
        .iter()
        .find(|shape| shape.ratio == ratio)
        .copied()
        .ok_or(EditError::CanvasRatio)
}

/// The request that gives the canvas `ratio`, sized by [`canvas_for`].
pub fn ratio_request(draft: &Draft, ratio: &str) -> Result<Request, EditError> {
    let shape = canvas_for(draft, ratio)?;
    Ok(Request::SetCanvasSize {
        width: shape.width,
        height: shape.height,
        ratio: shape.ratio.to_owned(),
    })
}

/// The canvas request for putting `material_id` first on an empty main
/// lane: an `original` canvas takes that picture's size. Any other ratio,
/// a lane that already has a picture, or a sound is `None`.
pub fn follow_first_picture(draft: &Draft, material_id: &str) -> Option<Request> {
    if draft.canvas_config.ratio_kind() != CanvasRatio::Original || first_picture(draft).is_some() {
        return None;
    }
    let (width, height) = picture_size(draft, material_id)?;
    Some(Request::SetCanvasSize {
        width,
        height,
        ratio: CanvasRatio::Original.as_str().to_owned(),
    })
}

fn first_picture(draft: &Draft) -> Option<(i32, i32)> {
    draft
        .main_track()?
        .segments
        .iter()
        .find_map(|segment| picture_size(draft, segment.material_id.as_str()))
}

/// The upright size of a picture material, fitted within 7680.
fn picture_size(draft: &Draft, material_id: &str) -> Option<(i32, i32)> {
    let Some(MaterialRef::Video(video)) = draft.materials.get(material_id) else {
        return None;
    };
    let (width, height) = (i64::from(video.width), i64::from(video.height));
    if width <= 0 || height <= 0 {
        return None;
    }
    let long = width.max(height);
    let side = |edge: i64| {
        let fitted = if long > i64::from(MAX_CANVAS_SIDE) {
            (edge * i64::from(MAX_CANVAS_SIDE) + long / 2) / long
        } else {
            edge
        };
        i32::try_from(fitted.max(1)).unwrap_or(MAX_CANVAS_SIDE)
    };
    Some((side(width), side(height)))
}

/// Store the color in the bars around a fitted picture.
///
/// An empty string lets the project color show. A video or a composition
/// can hold the color. An existing blur or image keeps its kind. The crop,
/// the scale, and the project canvas stay.
pub fn set_letterbox(
    draft: &mut Draft,
    segment_id: &SegmentId,
    color: &str,
    canvas_id: &MaterialId,
) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let material_id = draft.tracks[track_index].segments[segment_index]
        .material_id
        .clone();
    if material_id.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    match draft.materials.get(material_id.as_str()) {
        Some(MaterialRef::Video(_) | MaterialRef::Draft(_)) => {}
        Some(_) => return Err(EditError::Letterbox),
        None => return Err(EditError::MissingMaterial),
    }
    let color = color.trim();
    if !color.is_empty() && !canvas_color(color) {
        return Err(EditError::CanvasColor);
    }
    let existing = draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .iter()
        .find(|material_id| {
            matches!(
                draft.materials.get(material_id.as_str()),
                Some(MaterialRef::Canvas(_))
            )
        })
        .cloned();
    if let Some(existing_id) = existing {
        let existing_id = super::own::own(draft, track_index, segment_index, &existing_id)?;
        let Some(canvas) = draft
            .materials
            .canvases
            .iter_mut()
            .find(|canvas| canvas.id == existing_id)
        else {
            return Err(EditError::MissingMaterial);
        };
        canvas.color = color.to_owned();
        return Ok(());
    }
    if color.is_empty() {
        return Ok(());
    }
    if canvas_id.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    if draft.materials.get(canvas_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    draft.materials.canvases.push(MaterialCanvas {
        id: canvas_id.clone(),
        kind: "canvas_color".to_owned(),
        color: color.to_owned(),
        ..MaterialCanvas::default()
    });
    draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .push(canvas_id.clone());
    Ok(())
}

/// Cover the canvas behind a picture with a blurred copy of it.
///
/// `Some` turns the segment's canvas into a blur, keeping its color, or
/// adds one named `canvas_id`. `None` removes the segment's canvas, and the
/// material too once nothing else uses it, so the project color shows. A
/// video or a composition can hold it.
pub fn set_letterbox_blur(
    draft: &mut Draft,
    segment_id: &SegmentId,
    blur: Option<f64>,
    canvas_id: &MaterialId,
) -> Result<(), EditError> {
    if blur.is_some_and(|blur| !(0.0..=1.0).contains(&blur)) {
        return Err(EditError::LetterboxBlur);
    }
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let material_id = draft.tracks[track_index].segments[segment_index]
        .material_id
        .clone();
    match draft.materials.get(material_id.as_str()) {
        Some(MaterialRef::Video(_) | MaterialRef::Draft(_)) => {}
        Some(_) => return Err(EditError::Letterbox),
        None => return Err(EditError::MissingMaterial),
    }
    let existing = segment_canvas(draft, track_index, segment_index);
    let Some(blur) = blur else {
        if let Some(existing_id) = existing {
            let refs = &mut draft.tracks[track_index].segments[segment_index].extra_material_refs;
            refs.retain(|id| *id != existing_id);
            let used = draft.tracks.iter().any(|track| {
                track
                    .segments
                    .iter()
                    .any(|segment| segment.extra_material_refs.contains(&existing_id))
            });
            if !used {
                draft
                    .materials
                    .canvases
                    .retain(|canvas| canvas.id != existing_id);
            }
        }
        return Ok(());
    };
    if let Some(existing_id) = existing {
        let existing_id = super::own::own(draft, track_index, segment_index, &existing_id)?;
        let Some(canvas) = draft
            .materials
            .canvases
            .iter_mut()
            .find(|canvas| canvas.id == existing_id)
        else {
            return Err(EditError::MissingMaterial);
        };
        canvas.kind = "canvas_blur".to_owned();
        canvas.blur = blur;
        return Ok(());
    }
    if canvas_id.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    if draft.materials.get(canvas_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    draft.materials.canvases.push(MaterialCanvas {
        id: canvas_id.clone(),
        kind: "canvas_blur".to_owned(),
        blur,
        ..MaterialCanvas::default()
    });
    draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .push(canvas_id.clone());
    Ok(())
}

/// The canvas material a segment refers to, if any.
fn segment_canvas(draft: &Draft, track_index: usize, segment_index: usize) -> Option<MaterialId> {
    draft.tracks[track_index].segments[segment_index]
        .extra_material_refs
        .iter()
        .find(|material_id| {
            matches!(
                draft.materials.get(material_id.as_str()),
                Some(MaterialRef::Canvas(_))
            )
        })
        .cloned()
}

/// A color the preview paints as the project canvas.
///
/// Eight digits are read as `AARRGGBB` before a six-digit `RRGGBB` color.
fn canvas_color(text: &str) -> bool {
    if packed(text) {
        return true;
    }
    let hex = text.strip_prefix('#').unwrap_or(text);
    hex.len() == 6 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn packed(text: &str) -> bool {
    let hex = text
        .strip_prefix("0x")
        .or_else(|| text.strip_prefix("0X"))
        .or_else(|| text.strip_prefix('#'))
        .unwrap_or(text);
    hex.len() == 8 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, MaterialCanvas, MaterialId, MaterialText, MaterialVideo, Segment, TimeRange,
        TrackAttribute,
    };

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    #[test]
    fn a_canvas_color_sits_under_the_frame_and_an_empty_string_clears_it() {
        let mut session = Session::open(Draft::empty("D"));
        let size = session.draft().canvas_config.clone();
        session
            .apply(
                0,
                &[Request::SetCanvasBackground {
                    color: "  #336699  ".to_owned(),
                }],
            )
            .unwrap();
        assert_eq!(
            session.draft().canvas_config.background.as_deref(),
            Some("#336699")
        );
        assert_eq!(session.draft().canvas_config.width, size.width);
        assert_eq!(session.draft().canvas_config.height, size.height);
        assert_eq!(session.draft().canvas_config.ratio, size.ratio);

        session
            .apply(
                1,
                &[Request::SetCanvasBackground {
                    color: String::new(),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().canvas_config.background, None);
        session.undo().unwrap();
        assert_eq!(
            session.draft().canvas_config.background.as_deref(),
            Some("#336699")
        );

        session
            .apply(
                3,
                &[Request::SetCanvasBackground {
                    color: "0xFF336699".to_owned(),
                }],
            )
            .unwrap();
        assert_eq!(
            session.draft().canvas_config.background.as_deref(),
            Some("0xFF336699")
        );

        let before = session.draft().clone();
        let error = session
            .apply(
                4,
                &[Request::SetCanvasBackground {
                    color: "white".to_owned(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::CanvasColor);
        let error = session
            .apply(
                4,
                &[Request::SetCanvasBackground {
                    color: "0x336699".to_owned(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::CanvasColor);
        assert_eq!(session.draft(), &before);
    }

    #[test]
    fn a_canvas_size_sets_the_frame_and_a_side_past_7680_is_refused() {
        let mut session = Session::open(Draft::empty("D"));
        let fps = session.draft().fps;
        session
            .apply(
                0,
                &[Request::SetCanvasBackground {
                    color: "#336699".to_owned(),
                }],
            )
            .unwrap();
        session
            .apply(
                1,
                &[Request::SetCanvasSize {
                    width: 1_080,
                    height: 1_920,
                    ratio: "9:16".to_owned(),
                }],
            )
            .unwrap();
        let canvas = &session.draft().canvas_config;
        assert_eq!(canvas.width, 1_080);
        assert_eq!(canvas.height, 1_920);
        assert_eq!(canvas.ratio, "9:16");
        assert_eq!(canvas.background.as_deref(), Some("#336699"));
        assert_eq!(session.draft().fps, fps);
        session.undo().unwrap();
        assert_eq!(session.draft().canvas_config.width, 1_920);
        assert_eq!(session.draft().canvas_config.ratio, "original");

        session
            .apply(
                3,
                &[Request::SetCanvasSize {
                    width: 7_680,
                    height: 1,
                    ratio: "custom".to_owned(),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().canvas_config.width, 7_680);

        let before = session.draft().clone();
        let error = session
            .apply(
                4,
                &[Request::SetCanvasSize {
                    width: 0,
                    height: 1080,
                    ratio: "16:9".to_owned(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::CanvasSize);
        let error = session
            .apply(
                4,
                &[Request::SetCanvasSize {
                    width: 7_681,
                    height: 1080,
                    ratio: "16:9".to_owned(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::CanvasSize);
        let error = session
            .apply(
                4,
                &[Request::SetCanvasSize {
                    width: 1920,
                    height: 1080,
                    ratio: "wide".to_owned(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::CanvasRatio);
        assert_eq!(session.draft(), &before);
    }

    fn picture(draft: &mut Draft, id: &str, width: i32, height: i32) {
        let mut video = MaterialVideo::new(id);
        video.width = width;
        video.height = height;
        draft.materials.videos.push(video);
    }

    #[test]
    fn a_ratio_names_its_frame_and_original_follows_the_first_picture() {
        let mut draft = Draft::empty("D");
        assert_eq!(
            super::canvas_for(&draft, "9:16").map(|shape| (shape.width, shape.height)),
            Ok((1_080, 1_920))
        );
        assert_eq!(
            super::canvas_for(&draft, "original").map(|shape| (shape.width, shape.height)),
            Ok((1_920, 1_080))
        );
        assert_eq!(
            super::canvas_for(&draft, "custom"),
            Err(EditError::CanvasRatio)
        );
        assert_eq!(
            super::canvas_for(&draft, "wide"),
            Err(EditError::CanvasRatio)
        );

        picture(&mut draft, "tall", 1_170, 2_532);
        picture(&mut draft, "huge", 8_000, 4_000);
        draft.tracks[0]
            .segments
            .push(Segment::new("tall", "tall", TimeRange::new(0, 1_000_000)));
        assert_eq!(
            super::canvas_for(&draft, "original").map(|shape| (shape.width, shape.height)),
            Ok((1_170, 2_532))
        );
        draft.tracks[0].segments[0].material_id = MaterialId::new("huge");
        assert_eq!(
            super::canvas_for(&draft, "original").map(|shape| (shape.width, shape.height)),
            Ok((7_680, 3_840))
        );

        let mut session = Session::open(draft);
        let request = super::ratio_request(session.draft(), "4:5").unwrap();
        session.apply(0, &[request]).unwrap();
        let canvas = &session.draft().canvas_config;
        assert_eq!(
            (canvas.width, canvas.height, canvas.ratio.as_str()),
            (1_080, 1_350, "4:5")
        );
    }

    #[test]
    fn only_an_original_canvas_with_no_picture_yet_follows_the_first_one() {
        let mut draft = Draft::empty("D");
        picture(&mut draft, "tall", 1_080, 1_920);
        assert_eq!(
            super::follow_first_picture(&draft, "tall"),
            Some(Request::SetCanvasSize {
                width: 1_080,
                height: 1_920,
                ratio: "original".to_owned(),
            })
        );
        assert_eq!(super::follow_first_picture(&draft, "missing"), None);

        draft.canvas_config.ratio = "16:9".to_owned();
        assert_eq!(super::follow_first_picture(&draft, "tall"), None);

        draft.canvas_config.ratio = "original".to_owned();
        draft.tracks[0]
            .segments
            .push(Segment::new("tall", "tall", TimeRange::new(0, 1_000_000)));
        assert_eq!(super::follow_first_picture(&draft, "tall"), None);
    }

    #[test]
    fn a_blur_covers_the_canvas_behind_a_picture_and_none_removes_it() {
        let mut draft = Draft::empty("D");
        draft.materials.videos.push(MaterialVideo::new("picture"));
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 1_000_000),
        ));
        let mut session = Session::open(draft);
        let blur = |blur: Option<f64>, canvas_id: &str| Request::SetLetterboxBlur {
            segment_id: "picture".into(),
            blur,
            canvas_id: MaterialId::new(canvas_id),
        };
        session.apply(0, &[blur(Some(0.1), "back")]).unwrap();
        let canvas = &session.draft().materials.canvases[0];
        assert_eq!((canvas.kind.as_str(), canvas.blur), ("canvas_blur", 0.1));
        assert_eq!(
            session.draft().tracks[0].segments[0].extra_material_refs,
            [MaterialId::new("back")]
        );

        // A second blur changes the same canvas; the new id is not used.
        session.apply(1, &[blur(Some(0.2), "unused")]).unwrap();
        assert_eq!(session.draft().materials.canvases.len(), 1);
        assert_eq!(session.draft().materials.canvases[0].blur, 0.2);

        session.apply(2, &[blur(None, "")]).unwrap();
        assert!(session.draft().materials.canvases.is_empty());
        assert!(
            session.draft().tracks[0].segments[0]
                .extra_material_refs
                .is_empty()
        );
        session.undo().unwrap();
        assert_eq!(session.draft().materials.canvases[0].blur, 0.2);

        let before = session.draft().clone();
        let revision = session.revision();
        for wrong in [-0.1, 1.5, f64::NAN] {
            assert_eq!(
                session.apply(revision, &[blur(Some(wrong), "x")]),
                Err(EditError::LetterboxBlur)
            );
        }
        assert_eq!(session.draft(), &before);
    }

    #[test]
    fn a_letterbox_color_stays_on_the_picture_and_a_word_is_refused() {
        let mut draft = Draft::empty("D");
        draft.config.maintrack_adsorb = false;
        draft.materials.videos.push(MaterialVideo::new("picture"));
        draft.materials.videos.push(MaterialVideo::new("blurred"));
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 1_000_000),
        ));
        draft.tracks[0].segments.push(Segment::new(
            "blurred",
            "blurred",
            TimeRange::new(1_000_000, 1_000_000),
        ));
        draft.materials.canvases.push(MaterialCanvas {
            id: MaterialId::new("blur"),
            kind: "canvas_blur".to_owned(),
            blur: 0.25,
            ..MaterialCanvas::default()
        });
        draft.tracks[0].segments[1]
            .extra_material_refs
            .push(MaterialId::new("blur"));
        draft.materials.texts.push(MaterialText::new("caption"));
        let mut words = cutlass_model::Track::main("words");
        words.kind = cutlass_model::TrackType::Text;
        words.segments.push(Segment::new(
            "caption",
            "caption",
            TimeRange::new(0, 1_000_000),
        ));
        draft.tracks.push(words);
        let mut session = Session::open(draft);

        let error = session
            .apply(
                0,
                &[Request::SetLetterbox {
                    segment_id: "caption".into(),
                    color: "#FF0000".to_owned(),
                    canvas_id: MaterialId::new("bars"),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::Letterbox);

        session
            .apply(
                0,
                &[Request::SetLetterbox {
                    segment_id: "picture".into(),
                    color: "  #FF0000  ".to_owned(),
                    canvas_id: MaterialId::new("bars"),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().materials.canvases.len(), 2);
        assert_eq!(session.draft().materials.canvases[1].kind, "canvas_color");
        assert_eq!(session.draft().materials.canvases[1].color, "#FF0000");
        assert_eq!(
            session.draft().tracks[0].segments[0].extra_material_refs,
            vec![MaterialId::new("bars")]
        );
        session
            .apply(
                1,
                &[Request::SetLetterbox {
                    segment_id: "picture".into(),
                    color: String::new(),
                    canvas_id: MaterialId::new("other"),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().materials.canvases.len(), 2);
        assert_eq!(session.draft().materials.canvases[1].color, "");
        session.undo().unwrap();
        assert_eq!(session.draft().materials.canvases[1].color, "#FF0000");

        let error = session
            .apply(
                3,
                &[Request::SetLetterbox {
                    segment_id: "picture".into(),
                    color: "white".to_owned(),
                    canvas_id: MaterialId::new("other"),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::CanvasColor);
        assert_eq!(session.draft().materials.canvases[1].color, "#FF0000");

        session
            .apply(
                3,
                &[Request::SetLetterbox {
                    segment_id: "blurred".into(),
                    color: "#00FF00".to_owned(),
                    canvas_id: MaterialId::new("other"),
                }],
            )
            .unwrap();
        let blur = &session.draft().materials.canvases[0];
        assert_eq!(blur.kind, "canvas_blur");
        assert_eq!(blur.blur, 0.25);
        assert_eq!(blur.color, "#00FF00");
        assert_eq!(session.draft().materials.canvases.len(), 2);

        let lane = session.draft().tracks[0].id.clone();
        session
            .apply(
                4,
                &[Request::SetTrackAttribute {
                    track_id: lane,
                    bit: TrackAttribute::LOCK,
                    on: true,
                }],
            )
            .unwrap();
        let error = session
            .apply(
                5,
                &[Request::SetLetterbox {
                    segment_id: "picture".into(),
                    color: "#0000FF".to_owned(),
                    canvas_id: MaterialId::new("later"),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert_eq!(session.draft().materials.canvases[1].color, "#FF0000");
    }
}

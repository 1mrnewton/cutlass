use cutlass_model::{CanvasRatio, Draft, MaterialCanvas, MaterialId, MaterialRef, SegmentId};

use super::speed::locate;
use super::{EditError, ensure_unlocked};

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

use cutlass_model::{Draft, MaterialText, SegmentId};

use super::speed::locate;
use super::{EditError, ensure_unlocked};

/// Replace the text material's string. Word columns stay as stored.
pub fn set_text(draft: &mut Draft, segment_id: &SegmentId, content: &str) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let material_id = draft.tracks[track_index].segments[segment_index]
        .material_id
        .clone();
    if material_id.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    if let Some(text) = draft
        .materials
        .texts
        .iter_mut()
        .find(|text| text.id == material_id)
    {
        text.content = content.to_owned();
        return Ok(());
    }
    if draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    draft.materials.texts.push(MaterialText {
        id: material_id,
        content: content.to_owned(),
        ..MaterialText::default()
    });
    Ok(())
}

/// The largest size a line can store. A larger size does not draw.
const MAX_TEXT_SIZE: f64 = 512.0;
/// The widest wrap box. A wider box passes the picture edge.
const MAX_LINE_WIDTH: f64 = 8_192.0;

/// Store the text size in pixels. An empty `font_path` takes `face`.
/// The string and the word columns stay.
pub fn set_text_size(
    draft: &mut Draft,
    segment_id: &SegmentId,
    size: f64,
    face: &str,
) -> Result<(), EditError> {
    let index = text_index(draft, segment_id)?;
    if !size.is_finite() || size <= 0.0 || size > MAX_TEXT_SIZE {
        return Err(EditError::TextSize);
    }
    let text = &mut draft.materials.texts[index];
    text.font_size = size;
    fill_face(text, face);
    Ok(())
}

/// Store the text color. An empty string is white. An empty `font_path` takes `face`.
pub fn set_text_color(
    draft: &mut Draft,
    segment_id: &SegmentId,
    color: &str,
    face: &str,
) -> Result<(), EditError> {
    let index = text_index(draft, segment_id)?;
    let color = color.trim();
    if !color.is_empty() && !hex_color(color) {
        return Err(EditError::TextColor);
    }
    let text = &mut draft.materials.texts[index];
    text.text_color = color.to_owned();
    fill_face(text, face);
    Ok(())
}

/// Store left (`0`), center (`1`), or right (`2`).
///
/// An empty line width takes `line_width` when that number is positive, so
/// center and right have a box to move in. An empty `font_path` takes `face`.
pub fn set_text_align(
    draft: &mut Draft,
    segment_id: &SegmentId,
    alignment: i32,
    face: &str,
    line_width: f64,
) -> Result<(), EditError> {
    let index = text_index(draft, segment_id)?;
    if !(0..=2).contains(&alignment) {
        return Err(EditError::TextAlign);
    }
    let text = &mut draft.materials.texts[index];
    text.alignment = alignment;
    if text.line_max_width == 0.0 && line_width.is_finite() && line_width > 0.0 {
        text.line_max_width = line_width;
    }
    fill_face(text, face);
    Ok(())
}

/// Extra space between letters, as a fraction of the text size.
/// `-1` removes one size of gap. `0` is the normal gap. `4` is the widest.
pub fn set_letter_spacing(
    draft: &mut Draft,
    segment_id: &SegmentId,
    spacing: f64,
    face: &str,
) -> Result<(), EditError> {
    let index = text_index(draft, segment_id)?;
    if !spacing.is_finite() || !(-1.0..=4.0).contains(&spacing) {
        return Err(EditError::LetterSpacing);
    }
    let text = &mut draft.materials.texts[index];
    text.letter_spacing = spacing;
    fill_face(text, face);
    Ok(())
}

/// Extra space between lines, as a fraction of the text size.
/// The gap must stay open, so `-1` is refused. `0` is the normal gap.
pub fn set_line_spacing(
    draft: &mut Draft,
    segment_id: &SegmentId,
    spacing: f64,
    face: &str,
) -> Result<(), EditError> {
    let index = text_index(draft, segment_id)?;
    if !spacing.is_finite() || spacing <= -1.0 || spacing > 4.0 {
        return Err(EditError::LineSpacing);
    }
    let text = &mut draft.materials.texts[index];
    text.line_spacing = spacing;
    fill_face(text, face);
    Ok(())
}

/// Store the wrap width in pixels. `0` keeps the run on one line unless the
/// string itself breaks. A positive width wraps the next letter that would
/// pass it, and it is the box center and right move inside.
pub fn set_line_width(
    draft: &mut Draft,
    segment_id: &SegmentId,
    width: f64,
    face: &str,
) -> Result<(), EditError> {
    let index = text_index(draft, segment_id)?;
    if !width.is_finite() || width < 0.0 || width > MAX_LINE_WIDTH {
        return Err(EditError::LineWidth);
    }
    let text = &mut draft.materials.texts[index];
    text.line_max_width = width;
    fill_face(text, face);
    Ok(())
}

/// Store the face file name. The string, the size, and the word columns stay.
pub fn set_text_font(
    draft: &mut Draft,
    segment_id: &SegmentId,
    face: &str,
) -> Result<(), EditError> {
    let index = text_index(draft, segment_id)?;
    let face = face.trim();
    if !face_file(face) {
        return Err(EditError::TextFont);
    }
    draft.materials.texts[index].font_path = face.to_owned();
    Ok(())
}

fn face_file(face: &str) -> bool {
    !face.is_empty() && face.len() <= 255 && !face.contains(['/', '\\', '\0'])
}

fn text_index(draft: &Draft, segment_id: &SegmentId) -> Result<usize, EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let material_id = &draft.tracks[track_index].segments[segment_index].material_id;
    if material_id.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    if let Some(index) = draft
        .materials
        .texts
        .iter()
        .position(|text| text.id == *material_id)
    {
        return Ok(index);
    }
    if draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::NotText);
    }
    Err(EditError::MissingMaterial)
}

fn fill_face(text: &mut MaterialText, face: &str) {
    if text.font_path.is_empty() && !face.is_empty() {
        text.font_path = face.to_owned();
    }
}

fn hex_color(text: &str) -> bool {
    let body = text.strip_prefix('#').unwrap_or(text);
    (body.len() == 6 || body.len() == 8) && body.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use cutlass_model::{
        Draft, MaterialId, MaterialText, MaterialVideo, Segment, TimeRange, TrackAttribute, TrackId,
    };

    use crate::edit::EditError;
    use crate::request::Request;
    use crate::session::Session;

    fn open() -> Session {
        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        draft.materials.texts.push(MaterialText {
            id: MaterialId::new("caption"),
            content: "A".to_owned(),
            words: cutlass_model::WordColumns {
                text: vec!["A".to_owned()],
                start_time: vec![0],
                end_time: vec![100],
                ..cutlass_model::WordColumns::default()
            },
            ..MaterialText::default()
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "caption", TimeRange::new(0, 1_000_000)));
        Session::open(draft)
    }

    #[test]
    fn size_color_and_alignment_style_the_line_and_keep_the_words() {
        let mut session = open();
        session
            .apply(
                0,
                &[Request::SetTextSize {
                    segment_id: "a".into(),
                    size: 48.0,
                    face: "face.ttf".to_owned(),
                }],
            )
            .unwrap();
        let text = &session.draft().materials.texts[0];
        assert_eq!(text.font_size, 48.0);
        assert_eq!(text.font_path, "face.ttf");
        assert_eq!(text.content, "A");
        assert_eq!(text.words.text, ["A"]);

        session
            .apply(
                1,
                &[Request::SetTextSize {
                    segment_id: "a".into(),
                    size: 64.0,
                    face: "other.ttf".to_owned(),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().materials.texts[0].font_path, "face.ttf");
        assert_eq!(session.draft().materials.texts[0].font_size, 64.0);

        session
            .apply(
                2,
                &[Request::SetTextColor {
                    segment_id: "a".into(),
                    color: "  #ff0000 ".to_owned(),
                    face: "other.ttf".to_owned(),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().materials.texts[0].text_color, "#ff0000");

        session
            .apply(
                3,
                &[Request::SetTextAlign {
                    segment_id: "a".into(),
                    alignment: 1,
                    face: "other.ttf".to_owned(),
                    line_width: 200.0,
                }],
            )
            .unwrap();
        let text = &session.draft().materials.texts[0];
        assert_eq!(text.alignment, 1);
        assert_eq!(text.line_max_width, 200.0);
        session
            .apply(
                4,
                &[Request::SetTextAlign {
                    segment_id: "a".into(),
                    alignment: 2,
                    face: "other.ttf".to_owned(),
                    line_width: 40.0,
                }],
            )
            .unwrap();
        assert_eq!(session.draft().materials.texts[0].line_max_width, 200.0);
        session.undo().unwrap();
        assert_eq!(session.draft().materials.texts[0].alignment, 1);
        assert_eq!(session.draft().materials.texts[0].words.text, ["A"]);
    }

    #[test]
    fn a_bad_style_a_picture_and_a_locked_lane_leave_the_line() {
        let mut session = open();
        let before = session.draft().clone();
        let error = session
            .apply(
                0,
                &[Request::SetTextSize {
                    segment_id: "a".into(),
                    size: 0.0,
                    face: "face.ttf".to_owned(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::TextSize);
        let error = session
            .apply(
                0,
                &[Request::SetTextColor {
                    segment_id: "a".into(),
                    color: "#gg0000".to_owned(),
                    face: "face.ttf".to_owned(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::TextColor);
        let error = session
            .apply(
                0,
                &[Request::SetTextAlign {
                    segment_id: "a".into(),
                    alignment: 9,
                    face: "face.ttf".to_owned(),
                    line_width: 10.0,
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::TextAlign);
        assert_eq!(session.draft(), &before);

        let mut picture = Draft::empty("D");
        picture.tracks[0].id = TrackId::new("main");
        picture.materials.videos.push(MaterialVideo::new("picture"));
        picture.tracks[0]
            .segments
            .push(Segment::new("a", "picture", TimeRange::new(0, 1_000_000)));
        let mut session = Session::open(picture);
        let error = session
            .apply(
                0,
                &[Request::SetTextSize {
                    segment_id: "a".into(),
                    size: 32.0,
                    face: "face.ttf".to_owned(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::NotText);

        let mut session = open();
        session
            .apply(
                0,
                &[Request::SetTrackAttribute {
                    track_id: TrackId::new("main"),
                    bit: TrackAttribute::LOCK,
                    on: true,
                }],
            )
            .unwrap();
        let error = session
            .apply(
                1,
                &[Request::SetTextSize {
                    segment_id: "a".into(),
                    size: 32.0,
                    face: "face.ttf".to_owned(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert_eq!(session.draft().materials.texts[0].font_size, 0.0);
    }

    #[test]
    fn spacing_opens_the_gaps_and_a_closed_line_is_refused() {
        let mut session = open();
        session
            .apply(
                0,
                &[Request::SetLetterSpacing {
                    segment_id: "a".into(),
                    spacing: 1.0,
                    face: "face.ttf".to_owned(),
                }],
            )
            .unwrap();
        let text = &session.draft().materials.texts[0];
        assert_eq!(text.letter_spacing, 1.0);
        assert_eq!(text.line_spacing, 0.0);
        assert_eq!(text.font_path, "face.ttf");
        assert_eq!(text.font_size, 0.0);
        assert_eq!(text.words.text, ["A"]);

        session
            .apply(
                1,
                &[Request::SetLineSpacing {
                    segment_id: "a".into(),
                    spacing: 0.5,
                    face: "other.ttf".to_owned(),
                }],
            )
            .unwrap();
        let text = &session.draft().materials.texts[0];
        assert_eq!(text.line_spacing, 0.5);
        assert_eq!(text.letter_spacing, 1.0);
        assert_eq!(text.font_path, "face.ttf");
        session.undo().unwrap();
        assert_eq!(session.draft().materials.texts[0].line_spacing, 0.0);
        assert_eq!(session.draft().materials.texts[0].letter_spacing, 1.0);

        let before = session.draft().clone();
        let error = session
            .apply(
                3,
                &[Request::SetLetterSpacing {
                    segment_id: "a".into(),
                    spacing: 4.1,
                    face: "face.ttf".to_owned(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::LetterSpacing);
        let error = session
            .apply(
                3,
                &[Request::SetLineSpacing {
                    segment_id: "a".into(),
                    spacing: -1.0,
                    face: "face.ttf".to_owned(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::LineSpacing);
        assert_eq!(session.draft(), &before);
    }

    #[test]
    fn a_line_width_wraps_and_zero_keeps_one_line() {
        let mut session = open();
        session
            .apply(
                0,
                &[Request::SetLineWidth {
                    segment_id: "a".into(),
                    width: 40.0,
                    face: "face.ttf".to_owned(),
                }],
            )
            .unwrap();
        let text = &session.draft().materials.texts[0];
        assert_eq!(text.line_max_width, 40.0);
        assert_eq!(text.alignment, 0);
        assert_eq!(text.font_path, "face.ttf");
        assert_eq!(text.font_size, 0.0);
        assert_eq!(text.content, "A");
        assert_eq!(text.words.text, ["A"]);

        session
            .apply(
                1,
                &[Request::SetLineWidth {
                    segment_id: "a".into(),
                    width: 0.0,
                    face: "other.ttf".to_owned(),
                }],
            )
            .unwrap();
        assert_eq!(session.draft().materials.texts[0].line_max_width, 0.0);
        assert_eq!(session.draft().materials.texts[0].font_path, "face.ttf");
        session.undo().unwrap();
        assert_eq!(session.draft().materials.texts[0].line_max_width, 40.0);

        let before = session.draft().clone();
        let error = session
            .apply(
                3,
                &[Request::SetLineWidth {
                    segment_id: "a".into(),
                    width: -1.0,
                    face: "face.ttf".to_owned(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::LineWidth);
        let error = session
            .apply(
                3,
                &[Request::SetLineWidth {
                    segment_id: "a".into(),
                    width: 8_193.0,
                    face: "face.ttf".to_owned(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::LineWidth);
        assert_eq!(session.draft(), &before);
    }

    #[test]
    fn a_face_name_stays_on_the_text_and_a_path_is_refused() {
        let mut session = open();
        session
            .apply(
                0,
                &[Request::SetTextFont {
                    segment_id: "a".into(),
                    face: "  sample.ttf  ".to_owned(),
                }],
            )
            .unwrap();
        let text = &session.draft().materials.texts[0];
        assert_eq!(text.font_path, "sample.ttf");
        assert_eq!(text.font_size, 0.0);
        assert_eq!(text.content, "A");
        assert_eq!(text.words.text, ["A"]);
        session.undo().unwrap();
        assert_eq!(session.draft().materials.texts[0].font_path, "");

        let before = session.draft().clone();
        for face in ["", "faces/sample.ttf", "faces\\sample.ttf"] {
            let error = session
                .apply(
                    2,
                    &[Request::SetTextFont {
                        segment_id: "a".into(),
                        face: face.to_owned(),
                    }],
                )
                .unwrap_err();
            assert_eq!(error, EditError::TextFont);
        }
        assert_eq!(session.draft(), &before);

        let lane = session.draft().tracks[0].id.clone();
        session
            .apply(
                2,
                &[Request::SetTrackAttribute {
                    track_id: lane,
                    bit: TrackAttribute::LOCK,
                    on: true,
                }],
            )
            .unwrap();
        let error = session
            .apply(
                3,
                &[Request::SetTextFont {
                    segment_id: "a".into(),
                    face: "sample.ttf".to_owned(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert_eq!(session.draft().materials.texts[0].font_path, "");
    }
}

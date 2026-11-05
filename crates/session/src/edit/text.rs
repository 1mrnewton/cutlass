use cutlass_model::{
    Draft, MaterialEffect, MaterialId, MaterialText, Segment, SegmentId, TimeRange, TrackId,
    TrackType, segment_bloom,
};

use super::speed::locate;
use super::{EditError, ensure_unlocked};

/// Replace the text material's words. A rich title keeps its styled
/// ranges. Word columns stay as stored.
pub fn set_text(draft: &mut Draft, segment_id: &SegmentId, content: &str) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    let material_id = draft.tracks[track_index].segments[segment_index]
        .material_id
        .clone();
    if material_id.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    let material_id = super::own::own(draft, track_index, segment_index, &material_id)?;
    if let Some(text) = draft
        .materials
        .texts
        .iter_mut()
        .find(|text| text.id == material_id)
    {
        text.set_words(content);
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

/// The size a new text clip starts at, in pixels of the canvas.
pub const NEW_TEXT_SIZE: f64 = 80.0;

/// Place a new text clip on `track_id` over `target`, saying `content` in
/// `face`, white, at [`NEW_TEXT_SIZE`]. A missing track is added as text.
/// Blank content, a face that is not a file name, or an id already in use
/// changes nothing.
pub fn add_text(
    draft: &mut Draft,
    track_id: &TrackId,
    segment_id: &SegmentId,
    material_id: &MaterialId,
    target: TimeRange,
    content: &str,
    face: &str,
) -> Result<(), EditError> {
    if content.trim().is_empty() {
        return Err(EditError::TextEmpty);
    }
    let face = face.trim();
    if !face_file(face) {
        return Err(EditError::TextFont);
    }
    if draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    if let Some(track) = draft.tracks.iter().find(|track| &track.id == track_id) {
        if !matches!(track.kind, TrackType::Text) {
            return Err(EditError::TrackKind);
        }
        ensure_unlocked(track.attribute)?;
    } else {
        super::track::insert_track(draft, track_id, &TrackType::Text)?;
    }
    let mut segment = Segment::new(segment_id.clone(), material_id.clone(), target);
    segment.render_index = TrackType::Text.render_floor().unwrap_or(0);
    super::place::place(draft, track_id, segment)?;
    draft.materials.texts.push(MaterialText {
        id: material_id.clone(),
        kind: "text".to_owned(),
        content: content.to_owned(),
        font_path: face.to_owned(),
        font_size: NEW_TEXT_SIZE,
        ..MaterialText::new(material_id.clone())
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
    let index = own_text(draft, segment_id)?;
    if !size.is_finite() || size <= 0.0 || size > MAX_TEXT_SIZE {
        return Err(EditError::TextSize);
    }
    let text = &mut draft.materials.texts[index];
    text.font_size = size;
    fill_face(text, face);
    Ok(())
}

/// Store the text color, and paint every range of a rich title in it. An
/// empty string is white. An empty `font_path` takes `face`.
pub fn set_text_color(
    draft: &mut Draft,
    segment_id: &SegmentId,
    color: &str,
    face: &str,
) -> Result<(), EditError> {
    let index = own_text(draft, segment_id)?;
    let color = color.trim();
    if !color.is_empty() && !hex_color(color) {
        return Err(EditError::TextColor);
    }
    let text = &mut draft.materials.texts[index];
    text.text_color = color.to_owned();
    text.paint_rich(unit_rgb(if color.is_empty() { "#FFFFFF" } else { color }));
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
    let index = own_text(draft, segment_id)?;
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
    let index = own_text(draft, segment_id)?;
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
    let index = own_text(draft, segment_id)?;
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
    let index = own_text(draft, segment_id)?;
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
    let index = own_text(draft, segment_id)?;
    let face = face.trim();
    if !face_file(face) {
        return Err(EditError::TextFont);
    }
    draft.materials.texts[index].font_path = face.to_owned();
    Ok(())
}

/// The widest outline, in ems.
pub const MAX_OUTLINE: f64 = 0.25;
/// The farthest shadow, in hundredths of an em.
pub const MAX_SHADOW_DISTANCE: f64 = 100.0;
/// Room around the letters a new background box gets, in ems.
const BACKGROUND_PAD: (f64, f64) = (0.2, 0.1);

/// Ring the letters in `color`, `width` ems wide, fully opaque. An empty
/// color takes the outline off and keeps the width.
pub fn set_text_outline(
    draft: &mut Draft,
    segment_id: &SegmentId,
    color: &str,
    width: f64,
) -> Result<(), EditError> {
    let index = own_text(draft, segment_id)?;
    let color = style_color(color)?;
    if !(0.0..=MAX_OUTLINE).contains(&width) {
        return Err(EditError::TextStyle);
    }
    let text = &mut draft.materials.texts[index];
    text.border_color = color.to_owned();
    text.border_width = width;
    if !color.is_empty() {
        text.border_alpha = 1.0;
    }
    Ok(())
}

/// Cast a shadow in `color`. `shape` is the distance in hundredths of an
/// em, the angle in degrees, and the blur from 0 to 1. An empty color takes
/// the shadow off and keeps the rest. A shadow with no opacity yet is made
/// nearly opaque.
pub fn set_text_shadow(
    draft: &mut Draft,
    segment_id: &SegmentId,
    color: &str,
    shape: [f64; 3],
) -> Result<(), EditError> {
    let index = own_text(draft, segment_id)?;
    let color = style_color(color)?;
    let [distance, angle, blur] = shape;
    if !(0.0..=MAX_SHADOW_DISTANCE).contains(&distance)
        || !angle.is_finite()
        || !(0.0..=1.0).contains(&blur)
    {
        return Err(EditError::TextStyle);
    }
    let text = &mut draft.materials.texts[index];
    text.has_shadow = !color.is_empty();
    if !color.is_empty() {
        text.shadow_color = color.to_owned();
        if text.shadow_alpha <= 0.0 {
            text.shadow_alpha = 0.9;
        }
    }
    text.shadow_distance = distance;
    text.shadow_angle = angle;
    text.shadow_smoothing = blur;
    Ok(())
}

/// Fill a box behind the letters in `color` at `alpha` from 0 to 1, its
/// corners rounded by `round` from 0 to 1. An empty color takes the box
/// off. A box with no room around the letters yet gets some.
pub fn set_text_background(
    draft: &mut Draft,
    segment_id: &SegmentId,
    color: &str,
    alpha: f64,
    round: f64,
) -> Result<(), EditError> {
    let index = own_text(draft, segment_id)?;
    let color = style_color(color)?;
    if !(0.0..=1.0).contains(&alpha) || !(0.0..=1.0).contains(&round) {
        return Err(EditError::TextStyle);
    }
    let text = &mut draft.materials.texts[index];
    text.background_style = i32::from(!color.is_empty());
    text.background_color = color.to_owned();
    text.background_alpha = alpha;
    text.background_round_radius = round;
    if !color.is_empty() && text.background_width <= 0.0 && text.background_height <= 0.0 {
        (text.background_width, text.background_height) = BACKGROUND_PAD;
    }
    Ok(())
}

/// Draw the caption in the style keyed `style`, or plain when it is empty.
pub fn set_caption_style(
    draft: &mut Draft,
    segment_id: &SegmentId,
    style: &str,
) -> Result<(), EditError> {
    let style = style.trim();
    if !style.is_empty() && !cutlass_model::CAPTION_STYLE_KEYS.contains(&style) {
        return Err(EditError::TextStyle);
    }
    let index = own_text(draft, segment_id)?;
    style.clone_into(&mut draft.materials.texts[index].caption_style);
    Ok(())
}

/// The longest a caption's letters wait one after another.
pub const MAX_CAPTION_STAGGER_MS: i64 = 500;

/// Mark word `word` of the caption as a keyword, or take the mark off. A
/// word the caption does not have is refused.
pub fn set_caption_keyword(
    draft: &mut Draft,
    segment_id: &SegmentId,
    word: usize,
    on: bool,
) -> Result<(), EditError> {
    let index = own_text(draft, segment_id)?;
    let text = &mut draft.materials.texts[index];
    let columns = if text.current_words.text.is_empty() {
        &text.words
    } else {
        &text.current_words
    };
    if word >= columns.text.len() {
        return Err(EditError::TextStyle);
    }
    let marked = &mut text.caption_keywords;
    marked.retain(|kept| *kept != word);
    if on {
        marked.push(word);
        marked.sort_unstable();
    }
    Ok(())
}

/// Start each letter of the caption's words `millis` after the one before.
pub fn set_caption_stagger(
    draft: &mut Draft,
    segment_id: &SegmentId,
    millis: i64,
) -> Result<(), EditError> {
    if !(0..=MAX_CAPTION_STAGGER_MS).contains(&millis) {
        return Err(EditError::TextStyle);
    }
    let index = own_text(draft, segment_id)?;
    draft.materials.texts[index].caption_stagger_ms = millis;
    Ok(())
}

/// The room a bubble leaves around the letters, in ems, when it has none.
const BUBBLE_PAD: (f64, f64) = (0.35, 0.3);

/// Shape the background as the bubble keyed `bubble`, or a plain box again
/// when it is empty. A title with no background yet gets a white one, its
/// white letters turn black, and it gets room around the letters.
pub fn set_text_bubble(
    draft: &mut Draft,
    segment_id: &SegmentId,
    bubble: &str,
) -> Result<(), EditError> {
    let bubble = bubble.trim();
    if !bubble.is_empty() && cutlass_model::Bubble::of(bubble).is_none() {
        return Err(EditError::TextStyle);
    }
    let index = own_text(draft, segment_id)?;
    let text = &mut draft.materials.texts[index];
    text.bubble = bubble.to_owned();
    if bubble.is_empty() {
        return Ok(());
    }
    if text.background_style == 0 || text.background_color.is_empty() {
        text.background_style = 1;
        text.background_color = "#FFFFFF".to_owned();
        text.background_alpha = 1.0;
        if text.text_color.is_empty() || text.text_color.eq_ignore_ascii_case("#FFFFFF") {
            text.text_color = "#000000".to_owned();
        }
    }
    if text.background_width < BUBBLE_PAD.0 && text.background_height < BUBBLE_PAD.1 {
        (text.background_width, text.background_height) = BUBBLE_PAD;
    }
    Ok(())
}

/// Make the letters glow in `color`. `glow` is the strength and the range,
/// each from 0 to 1. The glow is a bloom effect among the clip's sidecars:
/// one only this clip points at is changed in place, and otherwise a new
/// one named `glow_id` takes its place, so a split clip's halves glow on
/// their own. An empty color takes the glow off, and drops the effect when
/// no clip points at it any more.
pub fn set_text_glow(
    draft: &mut Draft,
    segment_id: &SegmentId,
    glow_id: &MaterialId,
    color: &str,
    glow: [f64; 2],
) -> Result<(), EditError> {
    text_index(draft, segment_id)?;
    let color = style_color(color)?;
    let [strength, range] = glow;
    if !(0.0..=1.0).contains(&strength) || !(0.0..=1.0).contains(&range) {
        return Err(EditError::TextStyle);
    }
    let (track_index, segment_index) = locate(draft, segment_id)?;
    let segment = &draft.tracks[track_index].segments[segment_index];
    let existing = segment_bloom(&draft.materials, segment).map(|effect| effect.id.clone());
    let shared = existing
        .as_ref()
        .is_some_and(|id| pointed_at(draft, id) > 1);
    if color.is_empty() {
        let Some(id) = existing else {
            return Ok(());
        };
        draft.tracks[track_index].segments[segment_index]
            .extra_material_refs
            .retain(|kept| *kept != id);
        if !shared {
            draft.materials.effects.retain(|effect| effect.id != id);
        }
        return Ok(());
    }
    let mut bloom = existing
        .as_ref()
        .and_then(|id| {
            draft
                .materials
                .effects
                .iter()
                .find(|effect| &effect.id == id)
        })
        .and_then(MaterialEffect::bloom)
        .unwrap_or_default();
    bloom.color = color.to_owned();
    bloom.strength = strength;
    bloom.range = range;
    if let Some(id) = existing.as_ref().filter(|_| !shared) {
        if let Some(effect) = draft
            .materials
            .effects
            .iter_mut()
            .find(|effect| &effect.id == id)
        {
            effect.set_bloom(&bloom);
        }
        return Ok(());
    }
    if glow_id.is_empty() {
        return Err(EditError::MissingMaterial);
    }
    if draft.materials.get(glow_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    draft
        .materials
        .effects
        .push(MaterialEffect::glow(glow_id.clone(), &bloom));
    let refs = &mut draft.tracks[track_index].segments[segment_index].extra_material_refs;
    if let Some(id) = existing {
        refs.retain(|kept| *kept != id);
    }
    refs.push(glow_id.clone());
    Ok(())
}

/// How many clips point at material `id`.
fn pointed_at(draft: &Draft, id: &MaterialId) -> usize {
    draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .filter(|segment| segment.extra_material_refs.contains(id))
        .count()
}

/// A trimmed color that is empty or hex.
fn style_color(color: &str) -> Result<&str, EditError> {
    let color = color.trim();
    if color.is_empty() || hex_color(color) {
        Ok(color)
    } else {
        Err(EditError::TextColor)
    }
}

fn face_file(face: &str) -> bool {
    !face.is_empty() && face.len() <= 255 && !face.contains(['/', '\\', '\0'])
}

/// The index of the segment's own text material, copied first when
/// another segment shares it.
fn own_text(draft: &mut Draft, segment_id: &SegmentId) -> Result<usize, EditError> {
    text_index(draft, segment_id)?;
    let (track_index, segment_index) = locate(draft, segment_id)?;
    let shared = draft.tracks[track_index].segments[segment_index]
        .material_id
        .clone();
    let id = super::own::own(draft, track_index, segment_index, &shared)?;
    draft
        .materials
        .texts
        .iter()
        .position(|text| text.id == id)
        .ok_or(EditError::MissingMaterial)
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

/// The red, green, and blue of a hex color checked by [`hex_color`], each
/// from 0 to 1.
fn unit_rgb(color: &str) -> [f64; 3] {
    let body = color.strip_prefix('#').unwrap_or(color);
    let channel = |at: usize| {
        body.get(at..at + 2)
            .and_then(|pair| u8::from_str_radix(pair, 16).ok())
            .map_or(1.0, |byte| f64::from(byte) / 255.0)
    };
    [channel(0), channel(2), channel(4)]
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
    fn outline_shadow_and_background_go_on_and_come_off() {
        let mut session = open();
        let outline = |color: &str, width: f64| Request::SetTextOutline {
            segment_id: "a".into(),
            color: color.to_owned(),
            width,
        };
        let shadow = |color: &str| Request::SetTextShadow {
            segment_id: "a".into(),
            color: color.to_owned(),
            distance: 8.0,
            angle: -45.0,
            blur: 0.4,
        };
        let background = |color: &str, alpha: f64| Request::SetTextBackground {
            segment_id: "a".into(),
            color: color.to_owned(),
            alpha,
            round: 0.5,
        };
        session
            .apply(
                0,
                &[
                    outline("#000000", 0.08),
                    shadow("#000000"),
                    background("#FFCC00", 0.8),
                ],
            )
            .unwrap();
        let text = &session.draft().materials.texts[0];
        assert_eq!(
            (text.border_color.as_str(), text.border_width),
            ("#000000", 0.08)
        );
        assert_eq!(text.border_alpha, 1.0);
        assert!(text.has_shadow);
        assert_eq!(text.shadow_alpha, 0.9);
        assert_eq!(
            (
                text.shadow_distance,
                text.shadow_angle,
                text.shadow_smoothing
            ),
            (8.0, -45.0, 0.4)
        );
        assert_eq!(text.background_style, 1);
        assert_eq!(text.background_alpha, 0.8);
        assert_eq!(text.background_round_radius, 0.5);
        assert!(text.background_width > 0.0 && text.background_height > 0.0);
        assert_eq!(text.words.text, ["A"]);

        session
            .apply(1, &[outline("", 0.08), shadow(""), background("", 0.8)])
            .unwrap();
        let text = &session.draft().materials.texts[0];
        assert!(text.border_color.is_empty());
        assert!(!text.has_shadow);
        assert_eq!(text.background_style, 0);
        assert!(text.background_color.is_empty());

        let bubble = |key: &str| Request::SetTextBubble {
            segment_id: "a".into(),
            bubble: key.to_owned(),
        };
        let revision = session.revision();
        session.apply(revision, &[bubble("speech")]).unwrap();
        let text = &session.draft().materials.texts[0];
        assert_eq!(text.bubble, "speech");
        assert_eq!(
            (text.background_style, text.background_color.as_str()),
            (1, "#FFFFFF")
        );
        assert!(text.background_width >= 0.35);
        let revision = session.revision();
        session.apply(revision, &[bubble("")]).unwrap();
        let text = &session.draft().materials.texts[0];
        assert!(text.bubble.is_empty());
        // The box stays until the background is taken off.
        assert_eq!(text.background_style, 1);

        let caption = |key: &str| Request::SetCaptionStyle {
            segment_id: "a".into(),
            style: key.to_owned(),
        };
        let revision = session.revision();
        session.apply(revision, &[caption("karaoke")]).unwrap();
        assert_eq!(session.draft().materials.texts[0].caption_style, "karaoke");
        let revision = session.revision();
        session.apply(revision, &[caption("")]).unwrap();
        assert!(session.draft().materials.texts[0].caption_style.is_empty());

        // Keywords and stagger are kept on the material, one undo each.
        let keyword = |word: usize, on: bool| Request::SetCaptionKeyword {
            segment_id: "a".into(),
            word,
            on,
        };
        let revision = session.revision();
        session
            .apply(revision, &[keyword(0, true), keyword(0, true)])
            .unwrap();
        assert_eq!(session.draft().materials.texts[0].caption_keywords, [0]);
        let revision = session.revision();
        session.apply(revision, &[keyword(0, false)]).unwrap();
        assert!(
            session.draft().materials.texts[0]
                .caption_keywords
                .is_empty()
        );
        let stagger = |millis: i64| Request::SetCaptionStagger {
            segment_id: "a".into(),
            millis,
        };
        let revision = session.revision();
        session.apply(revision, &[stagger(40)]).unwrap();
        assert_eq!(session.draft().materials.texts[0].caption_stagger_ms, 40);
        session.undo().unwrap();
        assert_eq!(session.draft().materials.texts[0].caption_stagger_ms, 0);
        for refused in [keyword(1, true), stagger(-1), stagger(501)] {
            let revision = session.revision();
            assert_eq!(
                session.apply(revision, &[refused]),
                Err(EditError::TextStyle)
            );
        }

        for (request, error) in [
            (caption("loud"), EditError::TextStyle),
            (bubble("cloud"), EditError::TextStyle),
            (outline("#000000", 1.0), EditError::TextStyle),
            (outline("black", 0.1), EditError::TextColor),
            (background("#000000", 2.0), EditError::TextStyle),
            (
                Request::SetTextShadow {
                    segment_id: "a".into(),
                    color: "#000000".to_owned(),
                    distance: 8.0,
                    angle: f64::NAN,
                    blur: 0.4,
                },
                EditError::TextStyle,
            ),
        ] {
            let revision = session.revision();
            assert_eq!(session.apply(revision, &[request]).unwrap_err(), error);
        }
    }

    #[test]
    fn a_glow_is_a_bloom_effect_each_half_of_a_split_owns() {
        let mut session = open();
        let glow = |segment: &str, id: &str, color: &str, strength: f64| Request::SetTextGlow {
            segment_id: segment.into(),
            glow_id: MaterialId::new(id),
            color: color.to_owned(),
            strength,
            range: 0.5,
        };
        let bloom = |session: &Session, id: &str| {
            session
                .draft()
                .materials
                .effects
                .iter()
                .find(|effect| effect.id.as_str() == id)
                .and_then(cutlass_model::MaterialEffect::bloom)
        };
        session
            .apply(0, &[glow("a", "g1", "#FFCC00", 0.6)])
            .unwrap();
        let first = bloom(&session, "g1").unwrap();
        assert_eq!(
            (first.color.as_str(), first.strength, first.range),
            ("#FFCC00", 0.6, 0.5)
        );
        assert_eq!(session.draft().materials.effects[0].kind, "bloom");
        assert_eq!(
            session.draft().tracks[0].segments[0].extra_material_refs,
            [MaterialId::new("g1")]
        );

        // Its own glow changes in place; the new id goes unused.
        session
            .apply(1, &[glow("a", "unused", "#FFCC00", 0.9)])
            .unwrap();
        assert_eq!(bloom(&session, "g1").unwrap().strength, 0.9);
        assert!(bloom(&session, "unused").is_none());

        // After a split, a change to one half gives it a glow of its own.
        session
            .apply(
                2,
                &[Request::Split {
                    segment_id: "a".into(),
                    at: cutlass_model::Micros(500_000),
                    right_id: "b".into(),
                }],
            )
            .unwrap();
        session
            .apply(3, &[glow("b", "g2", "#00FF00", 0.3)])
            .unwrap();
        assert_eq!(bloom(&session, "g1").unwrap().color, "#FFCC00");
        assert_eq!(bloom(&session, "g2").unwrap().color, "#00FF00");
        let refs = |session: &Session, index: usize| {
            session.draft().tracks[0].segments[index]
                .extra_material_refs
                .clone()
        };
        assert_eq!(refs(&session, 1), [MaterialId::new("g2")]);

        // Taking it off drops the effect nobody points at.
        session.apply(4, &[glow("a", "", "", 0.0)]).unwrap();
        assert!(refs(&session, 0).is_empty());
        assert!(bloom(&session, "g1").is_none());
        session.apply(5, &[glow("a", "", "", 0.0)]).unwrap();

        for (request, error) in [
            (glow("a", "g3", "#FFFFFF", 1.5), EditError::TextStyle),
            (glow("a", "g3", "glow", 0.5), EditError::TextColor),
            (glow("a", "", "#FFFFFF", 0.5), EditError::MissingMaterial),
            (
                glow("a", "caption", "#FFFFFF", 0.5),
                EditError::DuplicateMaterial,
            ),
        ] {
            assert_eq!(session.apply(6, &[request]).unwrap_err(), error);
        }
    }

    #[test]
    fn a_rich_title_keeps_its_ranges_through_new_words_and_a_color() {
        let mut draft = Draft::empty("D");
        draft.materials.texts.push(MaterialText {
            id: MaterialId::new("rich"),
            is_rich_text: true,
            content: r#"{"text":"Hi","styles":[{"range":[0,1],"bold":true},{"range":[1,2]}]}"#
                .to_owned(),
            ..MaterialText::default()
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "rich", TimeRange::new(0, 1_000_000)));
        let mut session = Session::open(draft);
        session
            .apply(
                0,
                &[Request::SetTextContent {
                    segment_id: "a".into(),
                    content: "Hi you".to_owned(),
                }],
            )
            .unwrap();
        let text = &session.draft().materials.texts[0];
        let rich = text.rich().unwrap();
        assert_eq!(rich.text, "Hi you");
        assert_eq!(rich.styles[1].range, [1, 6]);
        assert_eq!(text.words().unwrap(), "Hi you");

        session
            .apply(
                1,
                &[Request::SetTextColor {
                    segment_id: "a".into(),
                    color: "#FF0000".to_owned(),
                    face: String::new(),
                }],
            )
            .unwrap();
        let rich = session.draft().materials.texts[0].rich().unwrap();
        assert!(
            rich.styles
                .iter()
                .all(|style| style.solid() == Some([1.0, 0.0, 0.0, 1.0]))
        );
        assert_eq!(rich.styles[0].rest.get("bold"), Some(&true.into()));
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

    fn add(track: &str, segment: &str, start: i64, content: &str, face: &str) -> Request {
        Request::AddText {
            track_id: TrackId::new(track),
            segment_id: cutlass_model::SegmentId::new(segment),
            material_id: MaterialId::new(format!("m-{segment}")),
            start: cutlass_model::Micros(start),
            duration: cutlass_model::Micros(2_000_000),
            content: content.to_owned(),
            face: face.to_owned(),
        }
    }

    #[test]
    fn a_new_text_clip_gets_its_row_its_words_and_its_face_in_one_step() {
        let mut session = open();
        session
            .apply(0, &[add("t1", "hello", 500_000, "Hello", "Arial.ttf")])
            .unwrap();
        assert_eq!(session.revision(), 1);
        let draft = session.draft();
        let row = draft
            .tracks
            .iter()
            .find(|track| track.id.as_str() == "t1")
            .unwrap();
        assert_eq!(row.kind, cutlass_model::TrackType::Text);
        assert_eq!(row.segments[0].target, TimeRange::new(500_000, 2_000_000));
        let text = draft
            .materials
            .texts
            .iter()
            .find(|text| text.id.as_str() == "m-hello")
            .unwrap();
        assert_eq!(text.content, "Hello");
        assert_eq!(text.font_path, "Arial.ttf");
        assert_eq!(text.font_size, super::NEW_TEXT_SIZE);

        // A second clip goes on the same row after the first.
        session
            .apply(1, &[add("t1", "later", 2_500_000, "Later", "Arial.ttf")])
            .unwrap();
        session.undo().unwrap();
        session.undo().unwrap();
        assert!(
            session
                .draft()
                .tracks
                .iter()
                .all(|track| track.id.as_str() != "t1")
        );
        assert_eq!(session.draft().materials.texts.len(), 1);
    }

    #[test]
    fn a_bad_new_text_clip_changes_nothing() {
        let mut session = open();
        let before = session.draft().clone();
        for (request, expected) in [
            (add("t1", "x", 0, "  ", "Arial.ttf"), EditError::TextEmpty),
            (add("t1", "x", 0, "Hi", ""), EditError::TextFont),
            (add("t1", "x", 0, "Hi", "../evil.ttf"), EditError::TextFont),
            (add("main", "x", 0, "Hi", "Arial.ttf"), EditError::TrackKind),
            (
                add("t1", "a", 0, "Hi", "Arial.ttf"),
                EditError::DuplicateSegment,
            ),
        ] {
            assert_eq!(session.apply(0, &[request]).unwrap_err(), expected);
            assert_eq!(session.draft(), &before);
        }
        session
            .apply(0, &[add("t1", "x", 0, "Hi", "Arial.ttf")])
            .unwrap();
        let overlap = add("t1", "y", 1_000_000, "Hi", "Arial.ttf");
        let after = session.draft().clone();
        assert_eq!(
            session.apply(1, &[overlap]).unwrap_err(),
            EditError::Overlap
        );
        assert_eq!(session.draft(), &after);
    }
}

//! Looks the user saves from a title or caption with "Save as preset", and
//! the requests that give another text that look. Every shell keeps the
//! same file, under the Cutlass folder's `Presets`.

use cutlass_model::{Draft, MaterialRef, Segment, SegmentId};
use cutlass_session::{Request, TextPattern};
use serde_json::{Map, Value, json};

use crate::inspector::text_look;

/// A saved look: the face, size, color, pattern, alignment and spacing,
/// and the outline, shadow and box. The words are not part of it.
#[derive(Clone, Debug, PartialEq)]
pub struct SavedLook {
    pub name: String,
    pub font_path: String,
    pub font_size: f64,
    pub color: String,
    pub pattern: TextPattern,
    pub alignment: i32,
    pub letter_spacing: f64,
    pub line_spacing: f64,
    /// Empty when off.
    pub outline_color: String,
    pub outline_width: f64,
    /// Empty when off.
    pub shadow_color: String,
    pub shadow_distance: f64,
    pub shadow_angle: f64,
    pub shadow_blur: f64,
    /// Empty when off.
    pub background_color: String,
    pub background_alpha: f64,
    pub background_round: f64,
}

/// The look of text clip `segment`, named `name`; `None` when it is not
/// text.
#[must_use]
pub fn saved_look(draft: &Draft, segment: &Segment, name: &str) -> Option<SavedLook> {
    let Some(MaterialRef::Text(text)) = draft.materials.get(segment.material_id.as_str()) else {
        return None;
    };
    let look = text_look(draft, segment)?;
    let alignment = if text.typesetting == 1 && (0..=2).contains(&text.alignment) {
        text.alignment + 3
    } else {
        text.alignment
    };
    Some(SavedLook {
        name: name.to_owned(),
        font_path: text.font_path.clone(),
        font_size: text.font_size,
        color: text.text_color.clone(),
        pattern: TextPattern::of(text),
        alignment,
        letter_spacing: text.letter_spacing,
        line_spacing: text.line_spacing,
        outline_color: look.outline_color,
        outline_width: look.outline_width,
        shadow_color: look.shadow_color,
        shadow_distance: look.shadow_distance,
        shadow_angle: look.shadow_angle,
        shadow_blur: look.shadow_blur,
        background_color: look.background_color,
        background_alpha: look.background_alpha,
        background_round: look.background_round,
    })
}

/// The requests that give `segment_id` the whole `look`, one undo step
/// when applied together. A text with no face takes `face`; `line_width`
/// is the box an alignment moves in when the text has none.
#[must_use]
pub fn look_requests(
    segment_id: &SegmentId,
    look: &SavedLook,
    face: &str,
    line_width: f64,
) -> Vec<Request> {
    let id = || segment_id.clone();
    let face = if look.font_path.is_empty() {
        face.to_owned()
    } else {
        look.font_path.clone()
    };
    let mut requests = Vec::new();
    if !look.font_path.is_empty() {
        requests.push(Request::SetTextFont {
            segment_id: id(),
            face: face.clone(),
        });
    }
    requests.extend([
        Request::SetTextSize {
            segment_id: id(),
            size: look.font_size,
            face: face.clone(),
        },
        Request::SetTextColor {
            segment_id: id(),
            color: look.color.clone(),
            face: face.clone(),
        },
        Request::SetTextPattern {
            segment_id: id(),
            bold: look.pattern.bold,
            italic: look.pattern.italic,
            underline: look.pattern.underline,
            face: face.clone(),
        },
        Request::SetTextAlign {
            segment_id: id(),
            alignment: look.alignment,
            face: face.clone(),
            line_width,
        },
        Request::SetLetterSpacing {
            segment_id: id(),
            spacing: look.letter_spacing,
            face: face.clone(),
        },
        Request::SetLineSpacing {
            segment_id: id(),
            spacing: look.line_spacing,
            face,
        },
        Request::SetTextOutline {
            segment_id: id(),
            color: look.outline_color.clone(),
            width: look.outline_width,
        },
        Request::SetTextShadow {
            segment_id: id(),
            color: look.shadow_color.clone(),
            distance: look.shadow_distance,
            angle: look.shadow_angle,
            blur: look.shadow_blur,
        },
        Request::SetTextBackground {
            segment_id: id(),
            color: look.background_color.clone(),
            alpha: look.background_alpha,
            round: look.background_round,
        },
    ]);
    requests
}

/// The looks in a saved file. A file that does not read, or a look in it
/// that does not, is left out.
#[must_use]
pub fn read_looks(bytes: &[u8]) -> Vec<SavedLook> {
    let Ok(Value::Object(file)) = serde_json::from_slice::<Value>(bytes) else {
        return Vec::new();
    };
    let Some(Value::Array(looks)) = file.get("looks") else {
        return Vec::new();
    };
    looks
        .iter()
        .filter_map(|look| look.as_object().and_then(read_look))
        .collect()
}

/// `looks` as a saved file.
#[must_use]
pub fn write_looks(looks: &[SavedLook]) -> Vec<u8> {
    let looks: Vec<Value> = looks.iter().map(write_look).collect();
    let file = json!({ "version": 1, "looks": looks });
    serde_json::to_vec_pretty(&file).unwrap_or_default()
}

fn write_look(look: &SavedLook) -> Value {
    json!({
        "name": look.name,
        "font_path": look.font_path,
        "font_size": look.font_size,
        "color": look.color,
        "bold": look.pattern.bold,
        "italic": look.pattern.italic,
        "underline": look.pattern.underline,
        "alignment": look.alignment,
        "letter_spacing": look.letter_spacing,
        "line_spacing": look.line_spacing,
        "outline_color": look.outline_color,
        "outline_width": look.outline_width,
        "shadow_color": look.shadow_color,
        "shadow_distance": look.shadow_distance,
        "shadow_angle": look.shadow_angle,
        "shadow_blur": look.shadow_blur,
        "background_color": look.background_color,
        "background_alpha": look.background_alpha,
        "background_round": look.background_round,
    })
}

fn read_look(look: &Map<String, Value>) -> Option<SavedLook> {
    let text = |key: &str| look.get(key)?.as_str().map(str::to_owned);
    let number = |key: &str| look.get(key)?.as_f64().filter(|value| value.is_finite());
    let flag = |key: &str| look.get(key).and_then(Value::as_bool).unwrap_or(false);
    let font_size = number("font_size").filter(|size| *size > 0.0)?;
    let alignment = look
        .get("alignment")
        .and_then(Value::as_i64)
        .and_then(|value| i32::try_from(value).ok())
        .filter(|value| (0..=5).contains(value))
        .unwrap_or(0);
    Some(SavedLook {
        name: text("name")?,
        font_path: text("font_path").unwrap_or_default(),
        font_size,
        color: text("color").unwrap_or_default(),
        pattern: TextPattern {
            bold: flag("bold"),
            italic: flag("italic"),
            underline: flag("underline"),
        },
        alignment,
        letter_spacing: number("letter_spacing").unwrap_or(0.0),
        line_spacing: number("line_spacing").unwrap_or(0.0),
        outline_color: text("outline_color").unwrap_or_default(),
        outline_width: number("outline_width").unwrap_or(0.0),
        shadow_color: text("shadow_color").unwrap_or_default(),
        shadow_distance: number("shadow_distance").unwrap_or(0.0),
        shadow_angle: number("shadow_angle").unwrap_or(0.0),
        shadow_blur: number("shadow_blur").unwrap_or(0.0),
        background_color: text("background_color").unwrap_or_default(),
        background_alpha: number("background_alpha").unwrap_or(1.0),
        background_round: number("background_round").unwrap_or(0.0),
    })
}

#[cfg(test)]
mod tests {
    use super::{SavedLook, read_looks, write_looks};
    use cutlass_session::TextPattern;

    fn look() -> SavedLook {
        SavedLook {
            name: "Mine".to_owned(),
            font_path: "A.ttf".to_owned(),
            font_size: 48.0,
            color: "#FF0000".to_owned(),
            pattern: TextPattern {
                bold: true,
                italic: false,
                underline: true,
            },
            alignment: 4,
            letter_spacing: 0.1,
            line_spacing: 0.2,
            outline_color: "#000000".to_owned(),
            outline_width: 0.06,
            shadow_color: String::new(),
            shadow_distance: 8.0,
            shadow_angle: -45.0,
            shadow_blur: 0.3,
            background_color: "#FFFFFF".to_owned(),
            background_alpha: 0.5,
            background_round: 0.3,
        }
    }

    #[test]
    fn saved_looks_read_back_as_written() {
        let looks = vec![
            look(),
            SavedLook {
                name: "Other".to_owned(),
                ..look()
            },
        ];
        assert_eq!(read_looks(&write_looks(&looks)), looks);
        assert_eq!(read_looks(&write_looks(&[])), []);
    }

    #[test]
    fn a_file_or_look_that_does_not_read_is_left_out() {
        assert!(read_looks(b"not json").is_empty());
        assert!(read_looks(b"[]").is_empty());
        let file = br#"{"looks": [{"name": "No size"}, {"name": "Sized", "font_size": 20}]}"#;
        let read = read_looks(file);
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].name, "Sized");
        assert_eq!(read[0].alignment, 0);
    }
}

//! Looks the user saves from a title or caption with "Save as preset", and
//! the requests that give another text that look. Every shell keeps the
//! same file, under the Cutlass folder's `Presets`.

use cutlass_model::{Draft, MaterialId, MaterialRef, Segment, SegmentId};
use cutlass_session::{Request, TextPattern};
use serde_json::{Map, Value, json};

use crate::inspector::text_look;

/// A saved look: the face, size, color, pattern, alignment and spacing,
/// and the outline, shadow, box, glow and curve. The words are not part
/// of it. A part kept only since presets first were is `None` in a file
/// saved before it, and a text given that look keeps its own.
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
    pub shadow_alpha: Option<f64>,
    pub background_box: Option<SavedBox>,
    pub glow: Option<SavedGlow>,
    /// Whether the lines bend, and by how many degrees of arc.
    pub curve: Option<(bool, f64)>,
}

/// How a saved look lays its box out: one for each line or one for all,
/// the room around the letters across and down, and how far right and up
/// it sits, in ems.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SavedBox {
    pub lines: bool,
    pub room: [f64; 2],
    pub offset: [f64; 2],
}

/// A saved look's glow: its color, empty when off, intensity and range,
/// and its rays and their lean right and up in degrees.
#[derive(Clone, Debug, PartialEq)]
pub struct SavedGlow {
    pub color: String,
    pub strength: f64,
    pub range: f64,
    pub rays: bool,
    pub lean: [f64; 2],
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
        shadow_alpha: Some(look.shadow_alpha),
        background_box: Some(SavedBox {
            lines: look.background_lines,
            room: [look.background_width, look.background_height],
            offset: [look.background_offset_x, look.background_offset_y],
        }),
        glow: Some(SavedGlow {
            color: look.glow_color,
            strength: look.glow_strength,
            range: look.glow_range,
            rays: look.glow_rays,
            lean: [look.glow_lean_x, look.glow_lean_y],
        }),
        curve: Some((look.curve_on, look.curve)),
    })
}

/// The requests that give `segment_id` the whole `look` but its glow, one
/// undo step when applied together. A text with no face takes `face`;
/// `line_width` is the box an alignment moves in when the text has none.
/// The glow is a material of each clip's own: see [`look_glow_request`].
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
    if let Some(opacity) = look.shadow_alpha {
        requests.push(Request::SetTextShadowOpacity {
            segment_id: id(),
            opacity,
        });
    }
    if let Some(layout) = look.background_box {
        requests.push(Request::SetTextBackgroundBox {
            segment_id: id(),
            lines: layout.lines,
            width: layout.room[0],
            height: layout.room[1],
            offset_x: layout.offset[0],
            offset_y: layout.offset[1],
        });
    }
    if let Some((on, degrees)) = look.curve {
        requests.push(Request::SetTextCurve {
            segment_id: id(),
            on,
            degrees,
        });
    }
    requests
}

/// The request that gives `segment_id` `look`'s glow, a new one named
/// `glow_id` when the clip has none of its own; `None` for a look saved
/// before glows were kept.
#[must_use]
pub fn look_glow_request(
    segment_id: &SegmentId,
    look: &SavedLook,
    glow_id: MaterialId,
) -> Option<Request> {
    let glow = look.glow.as_ref()?;
    Some(Request::SetTextGlow {
        segment_id: segment_id.clone(),
        glow_id,
        color: glow.color.clone(),
        strength: glow.strength,
        range: glow.range,
        rays: Some(glow.rays),
        lean: Some(glow.lean),
    })
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
        "shadow_alpha": look.shadow_alpha,
        "background_box": look.background_box.map(|layout| json!({
            "lines": layout.lines,
            "room": layout.room,
            "offset": layout.offset,
        })),
        "glow": look.glow.as_ref().map(|glow| json!({
            "color": glow.color,
            "strength": glow.strength,
            "range": glow.range,
            "rays": glow.rays,
            "lean": glow.lean,
        })),
        "curve": look.curve.map(|(on, degrees)| json!({ "on": on, "degrees": degrees })),
    })
}

/// Two finite numbers under `key` in `part`.
fn pair(part: &Map<String, Value>, key: &str) -> Option<[f64; 2]> {
    match part.get(key)?.as_array()?.as_slice() {
        [first, second] => Some([
            first.as_f64().filter(|value| value.is_finite())?,
            second.as_f64().filter(|value| value.is_finite())?,
        ]),
        _ => None,
    }
}

fn read_look(look: &Map<String, Value>) -> Option<SavedLook> {
    let text = |key: &str| look.get(key)?.as_str().map(str::to_owned);
    let number = |key: &str| look.get(key)?.as_f64().filter(|value| value.is_finite());
    let flag = |key: &str| look.get(key).and_then(Value::as_bool).unwrap_or(false);
    let part = |key: &str| look.get(key).and_then(Value::as_object);
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
        shadow_alpha: number("shadow_alpha"),
        background_box: part("background_box").and_then(|part| {
            Some(SavedBox {
                lines: part.get("lines").and_then(Value::as_bool).unwrap_or(false),
                room: pair(part, "room")?,
                offset: pair(part, "offset")?,
            })
        }),
        glow: part("glow").and_then(|part| {
            let number = |key: &str| part.get(key)?.as_f64().filter(|value| value.is_finite());
            Some(SavedGlow {
                color: part.get("color")?.as_str()?.to_owned(),
                strength: number("strength")?,
                range: number("range")?,
                rays: part.get("rays").and_then(Value::as_bool).unwrap_or(false),
                lean: pair(part, "lean").unwrap_or([0.0, 0.0]),
            })
        }),
        curve: part("curve").and_then(|part| {
            Some((
                part.get("on").and_then(Value::as_bool).unwrap_or(false),
                part.get("degrees")?
                    .as_f64()
                    .filter(|value| value.is_finite())?,
            ))
        }),
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
            shadow_alpha: Some(0.9),
            background_box: Some(super::SavedBox {
                lines: true,
                room: [0.2, 0.1],
                offset: [0.3, -0.2],
            }),
            glow: Some(super::SavedGlow {
                color: "#FFCC00".to_owned(),
                strength: 0.3,
                range: 0.2,
                rays: true,
                lean: [-1.0, 3.0],
            }),
            curve: Some((true, 72.0)),
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
        // A look saved before the glow, the curve and the rest were kept
        // has none of them, so a text given it keeps its own.
        assert_eq!(
            (
                read[0].shadow_alpha,
                read[0].background_box,
                &read[0].glow,
                read[0].curve
            ),
            (None, None, &None, None)
        );
    }

    #[test]
    fn a_look_gives_its_box_curve_and_glow_and_an_old_one_does_not() {
        use cutlass_session::Request;
        let id = cutlass_model::SegmentId::new("t");
        let has = |requests: &[Request], name: &str| {
            requests
                .iter()
                .any(|request| format!("{request:?}").starts_with(name))
        };
        let requests = super::look_requests(&id, &look(), "A.ttf", 1080.0);
        for name in [
            "SetTextShadowOpacity",
            "SetTextBackgroundBox",
            "SetTextCurve",
        ] {
            assert!(has(&requests, name), "{name}");
        }
        let glow = super::look_glow_request(&id, &look(), "g".into()).unwrap();
        assert!(matches!(
            glow,
            Request::SetTextGlow {
                rays: Some(true),
                ..
            }
        ));

        let old = SavedLook {
            shadow_alpha: None,
            background_box: None,
            glow: None,
            curve: None,
            ..look()
        };
        let requests = super::look_requests(&id, &old, "A.ttf", 1080.0);
        for name in [
            "SetTextShadowOpacity",
            "SetTextBackgroundBox",
            "SetTextCurve",
        ] {
            assert!(!has(&requests, name), "{name}");
        }
        assert!(super::look_glow_request(&id, &old, "g".into()).is_none());
    }
}

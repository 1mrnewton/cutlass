//! Text clips on the phone: add one at the playhead, change its words, its
//! face, or its style, and read them back. Faces come from the folders the shell
//! names; the draft keeps a face's file name.

use cutlass_model::{Draft, MaterialId, MaterialRef, MaterialText, TrackType};
use cutlass_session::{Micros, Request, SegmentId};

use crate::media::{fresh, room};

/// How long a new text clip lasts.
pub const NEW_TEXT_LENGTH: Micros = Micros(3_000_000);

/// A face the phone can set text in. `name` is what the draft stores.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneFont {
    pub name: String,
    pub label: String,
}

/// What a text clip says, the face it says it in, and its style.
#[derive(Clone, Debug, PartialEq)]
pub struct PhoneText {
    pub content: String,
    pub face: String,
    pub style: PhoneTextStyle,
}

/// How a title looks. A part with an empty color is off.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PhoneTextStyle {
    /// Pixels.
    pub size: f64,
    /// Hex; empty is white.
    pub color: String,
    pub outline_color: String,
    /// Ems.
    pub outline_width: f64,
    pub shadow_color: String,
    /// Hundredths of an em.
    pub shadow_distance: f64,
    /// 0 to 1.
    pub shadow_blur: f64,
    pub background_color: String,
    /// 0 to 1.
    pub background_alpha: f64,
    /// 0 square to 1 round.
    pub background_round: f64,
}

/// One change to a title's look.
#[derive(Clone, Debug, PartialEq)]
pub enum TextStyleChange {
    /// Pixels.
    Size(f64),
    /// Hex; empty is white.
    Color(String),
    /// An empty color takes the outline off. Width in ems.
    Outline { color: String, width: f64 },
    /// An empty color takes the shadow off. Distance in hundredths of an
    /// em, blur from 0 to 1.
    Shadow {
        color: String,
        distance: f64,
        blur: f64,
    },
    /// An empty color takes the box off. Alpha and round from 0 to 1.
    Background {
        color: String,
        alpha: f64,
        round: f64,
    },
}

/// Where a new shadow falls: down and to the right.
const SHADOW_ANGLE: f64 = -45.0;

/// The request for `change` on `segment_id`. A shadow keeps the angle a
/// shadow already has and falls down and right otherwise.
pub(crate) fn style_request(
    draft: &Draft,
    segment_id: &SegmentId,
    change: TextStyleChange,
) -> Request {
    let segment_id = segment_id.clone();
    match change {
        TextStyleChange::Size(size) => Request::SetTextSize {
            segment_id,
            size,
            face: String::new(),
        },
        TextStyleChange::Color(color) => Request::SetTextColor {
            segment_id,
            color,
            face: String::new(),
        },
        TextStyleChange::Outline { color, width } => Request::SetTextOutline {
            segment_id,
            color,
            width,
        },
        TextStyleChange::Shadow {
            color,
            distance,
            blur,
        } => {
            let angle = material(draft, &segment_id)
                .filter(|text| text.has_shadow && text.shadow_angle.is_finite())
                .map_or(SHADOW_ANGLE, |text| text.shadow_angle);
            Request::SetTextShadow {
                segment_id,
                color,
                distance,
                angle,
                blur,
            }
        }
        TextStyleChange::Background {
            color,
            alpha,
            round,
        } => Request::SetTextBackground {
            segment_id,
            color,
            alpha,
            round,
        },
    }
}

fn material<'a>(draft: &'a Draft, segment_id: &SegmentId) -> Option<&'a MaterialText> {
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == segment_id)?;
    match draft.materials.get(segment.material_id.as_str())? {
        MaterialRef::Text(text) => Some(text),
        _ => None,
    }
}

/// The faces in `faces`, each a name and its label for people.
#[must_use]
pub fn fonts(faces: Vec<(String, String)>) -> Vec<PhoneFont> {
    faces
        .into_iter()
        .map(|(name, label)| PhoneFont { name, label })
        .collect()
}

/// The request that adds `content` in `face` at `at` for
/// [`NEW_TEXT_LENGTH`], and the new clip's id. It goes on the first text
/// row with room, else on a new one.
pub(crate) fn add_request(
    draft: &Draft,
    content: &str,
    face: &str,
    at: Micros,
) -> (Request, SegmentId) {
    let (track_id, segment_id, taken) = room(draft, TrackType::Text, at, NEW_TEXT_LENGTH);
    let at = Micros(at.0.max(0));
    let material_id = MaterialId::new(fresh(&taken, "text"));
    let request = Request::AddText {
        track_id,
        segment_id: segment_id.clone(),
        material_id,
        start: at,
        duration: NEW_TEXT_LENGTH,
        content: content.to_owned(),
        face: face.to_owned(),
    };
    (request, segment_id)
}

/// What `segment_id` says, or `None` when it is not a text clip.
#[must_use]
pub fn text_of(draft: &Draft, segment_id: &SegmentId) -> Option<PhoneText> {
    let text = material(draft, segment_id)?;
    let on = |on: bool, color: &str| if on { color.to_owned() } else { String::new() };
    Some(PhoneText {
        content: text.content.clone(),
        face: text.font_path.clone(),
        style: PhoneTextStyle {
            size: text.font_size,
            color: text.text_color.clone(),
            outline_color: text.border_color.clone(),
            outline_width: text.border_width,
            shadow_color: on(text.has_shadow, &text.shadow_color),
            shadow_distance: text.shadow_distance,
            shadow_blur: text.shadow_smoothing,
            background_color: on(text.background_style != 0, &text.background_color),
            background_alpha: text.background_alpha,
            background_round: text.background_round_radius,
        },
    })
}

//! The text panel's Effects page: looks that set a title's fill, outline,
//! shadow and glow together in one step.
//!
//! Each part an effect leaves out is taken off, so picking one look after
//! another never stacks them. None takes every part off and keeps the
//! letters' color.

use cutlass_model::{MaterialId, SegmentId};
use cutlass_session::Request;

use crate::inspector::TextLook;

/// One look on the Effects page. A color of `""` leaves that part off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextEffect {
    pub id: &'static str,
    pub name: &'static str,
    /// The letters' color, or `None` to keep theirs.
    pub fill: Option<&'static str>,
    /// Color and width in ems.
    pub outline: (&'static str, f64),
    /// Color, distance in hundredths of an em, angle in degrees, and blur.
    pub shadow: (&'static str, f64, f64, f64),
    /// Color, strength and range.
    pub glow: (&'static str, f64, f64),
    /// The groups the Effects page files it under, as
    /// [`EFFECT_CATEGORIES`] names them.
    pub categories: &'static [&'static str],
}

/// The groups the Effects page offers, in its order.
pub const EFFECT_CATEGORIES: &[&str] = &[
    "Trending", "Classic", "Shadow", "Glow", "Neon", "Bold", "Retro",
];

const OFF_SHADOW: (&str, f64, f64, f64) = ("", 0.0, crate::inspector::SHADOW_ANGLE, 0.0);

/// The looks, in the order the page shows them after None.
pub const TEXT_EFFECTS: &[TextEffect] = &[
    TextEffect {
        id: "outline",
        name: "Outline",
        fill: None,
        outline: ("#000000", 0.08),
        shadow: OFF_SHADOW,
        glow: ("", 0.0, 0.0),
        categories: &["Trending", "Classic"],
    },
    TextEffect {
        id: "shadow",
        name: "Shadow",
        fill: None,
        outline: ("", 0.0),
        shadow: ("#000000", 8.0, -45.0, 0.5),
        glow: ("", 0.0, 0.0),
        categories: &["Classic", "Shadow"],
    },
    TextEffect {
        id: "glow",
        name: "Glow",
        fill: None,
        outline: ("", 0.0),
        shadow: OFF_SHADOW,
        glow: ("#FFFFFF", 0.6, 0.5),
        categories: &["Glow"],
    },
    TextEffect {
        id: "neon",
        name: "Neon",
        fill: Some("#FFFFFF"),
        outline: ("", 0.0),
        shadow: OFF_SHADOW,
        glow: ("#FF2BD6", 0.8, 0.6),
        categories: &["Trending", "Neon"],
    },
    TextEffect {
        id: "pop",
        name: "Pop",
        fill: Some("#FFD400"),
        outline: ("#000000", 0.1),
        shadow: ("#000000", 10.0, -45.0, 0.0),
        glow: ("", 0.0, 0.0),
        categories: &["Trending", "Bold"],
    },
    TextEffect {
        id: "retro",
        name: "Retro",
        fill: Some("#FF5A36"),
        outline: ("#FFFFFF", 0.06),
        shadow: ("#2A1B6B", 12.0, -45.0, 0.0),
        glow: ("", 0.0, 0.0),
        categories: &["Retro"],
    },
    TextEffect {
        id: "thick-outline",
        name: "Thick outline",
        fill: None,
        outline: ("#000000", 0.16),
        shadow: OFF_SHADOW,
        glow: ("", 0.0, 0.0),
        categories: &["Classic", "Bold"],
    },
    TextEffect {
        id: "white-outline",
        name: "White outline",
        fill: Some("#111111"),
        outline: ("#FFFFFF", 0.1),
        shadow: OFF_SHADOW,
        glow: ("", 0.0, 0.0),
        categories: &["Classic"],
    },
    TextEffect {
        id: "soft-shadow",
        name: "Soft shadow",
        fill: None,
        outline: ("", 0.0),
        shadow: ("#000000", 6.0, -45.0, 1.0),
        glow: ("", 0.0, 0.0),
        categories: &["Shadow"],
    },
    TextEffect {
        id: "long-shadow",
        name: "Long shadow",
        fill: None,
        outline: ("", 0.0),
        shadow: ("#000000", 18.0, -45.0, 0.0),
        glow: ("", 0.0, 0.0),
        categories: &["Shadow", "Retro"],
    },
    TextEffect {
        id: "haze",
        name: "Haze",
        fill: None,
        outline: ("", 0.0),
        shadow: ("#000000", 0.0, -45.0, 1.0),
        glow: ("", 0.0, 0.0),
        categories: &["Shadow"],
    },
    TextEffect {
        id: "pink-drop",
        name: "Pink drop",
        fill: Some("#FFFFFF"),
        outline: ("", 0.0),
        shadow: ("#FF4FA3", 10.0, -45.0, 0.0),
        glow: ("", 0.0, 0.0),
        categories: &["Shadow", "Bold"],
    },
    TextEffect {
        id: "gold-glow",
        name: "Gold glow",
        fill: Some("#FFE08A"),
        outline: ("", 0.0),
        shadow: OFF_SHADOW,
        glow: ("#FFB800", 0.7, 0.5),
        categories: &["Trending", "Glow"],
    },
    TextEffect {
        id: "ice-glow",
        name: "Ice glow",
        fill: Some("#FFFFFF"),
        outline: ("", 0.0),
        shadow: OFF_SHADOW,
        glow: ("#3EE6FF", 0.7, 0.5),
        categories: &["Glow"],
    },
    TextEffect {
        id: "outlined-glow",
        name: "Outlined glow",
        fill: Some("#FFFFFF"),
        outline: ("#000000", 0.06),
        shadow: OFF_SHADOW,
        glow: ("#FFFFFF", 0.5, 0.4),
        categories: &["Glow", "Classic"],
    },
    TextEffect {
        id: "neon-blue",
        name: "Neon blue",
        fill: Some("#FFFFFF"),
        outline: ("", 0.0),
        shadow: OFF_SHADOW,
        glow: ("#00C2FF", 0.9, 0.7),
        categories: &["Neon"],
    },
    TextEffect {
        id: "neon-green",
        name: "Neon green",
        fill: Some("#FFFFFF"),
        outline: ("", 0.0),
        shadow: OFF_SHADOW,
        glow: ("#7CFF3B", 0.9, 0.7),
        categories: &["Neon"],
    },
    TextEffect {
        id: "neon-red",
        name: "Neon red",
        fill: Some("#FFFFFF"),
        outline: ("", 0.0),
        shadow: OFF_SHADOW,
        glow: ("#FF2E2E", 0.9, 0.7),
        categories: &["Neon"],
    },
    TextEffect {
        id: "comic",
        name: "Comic",
        fill: Some("#FFFFFF"),
        outline: ("#000000", 0.12),
        shadow: ("#FF2E2E", 10.0, -45.0, 0.0),
        glow: ("", 0.0, 0.0),
        categories: &["Trending", "Bold"],
    },
    TextEffect {
        id: "sticker",
        name: "Sticker",
        fill: Some("#FF4FA3"),
        outline: ("#FFFFFF", 0.14),
        shadow: ("#000000", 4.0, -45.0, 0.3),
        glow: ("", 0.0, 0.0),
        categories: &["Bold"],
    },
    TextEffect {
        id: "lime-pop",
        name: "Lime pop",
        fill: Some("#B6FF3B"),
        outline: ("#000000", 0.1),
        shadow: ("#000000", 6.0, -45.0, 0.0),
        glow: ("", 0.0, 0.0),
        categories: &["Bold"],
    },
    TextEffect {
        id: "candy",
        name: "Candy",
        fill: Some("#FFFFFF"),
        outline: ("#FF4FA3", 0.1),
        shadow: ("#7A2BFF", 8.0, -45.0, 0.0),
        glow: ("", 0.0, 0.0),
        categories: &["Trending", "Retro"],
    },
    TextEffect {
        id: "sunset",
        name: "Sunset",
        fill: Some("#FFD400"),
        outline: ("#FF5A36", 0.08),
        shadow: ("#7A1B3B", 12.0, -45.0, 0.0),
        glow: ("", 0.0, 0.0),
        categories: &["Retro"],
    },
    TextEffect {
        id: "arcade",
        name: "Arcade",
        fill: Some("#3EE6FF"),
        outline: ("#1B1B6B", 0.1),
        shadow: ("#FF2BD6", 8.0, -45.0, 0.0),
        glow: ("", 0.0, 0.0),
        categories: &["Retro"],
    },
];

/// The look with no outline, shadow or glow.
pub const NO_TEXT_EFFECT: TextEffect = TextEffect {
    id: "none",
    name: "None",
    fill: None,
    outline: ("", 0.0),
    shadow: OFF_SHADOW,
    glow: ("", 0.0, 0.0),
    categories: &[],
};

/// The requests that give `segment_id` the look `effect`, in one step. A
/// glow that is new is a bloom effect named `glow_id`; `face` is the
/// shell's face, for a fill.
#[must_use]
pub fn text_effect_requests(
    effect: &TextEffect,
    segment_id: &SegmentId,
    glow_id: MaterialId,
    face: &str,
) -> Vec<Request> {
    let mut requests = Vec::new();
    if let Some(fill) = effect.fill {
        requests.push(Request::SetTextColor {
            segment_id: segment_id.clone(),
            color: fill.to_owned(),
            face: face.to_owned(),
        });
    }
    let (color, width) = effect.outline;
    requests.push(Request::SetTextOutline {
        segment_id: segment_id.clone(),
        color: color.to_owned(),
        width,
    });
    let (color, distance, angle, blur) = effect.shadow;
    requests.push(Request::SetTextShadow {
        segment_id: segment_id.clone(),
        color: color.to_owned(),
        distance,
        angle,
        blur,
    });
    let (color, strength, range) = effect.glow;
    requests.push(Request::SetTextGlow {
        segment_id: segment_id.clone(),
        glow_id,
        color: color.to_owned(),
        strength,
        range,
    });
    requests
}

/// The look a title wears, by its outline, shadow and glow: `None` when it
/// is none of them, as after a part was changed by hand.
#[must_use]
pub fn text_effect_of(look: &TextLook) -> Option<&'static TextEffect> {
    std::iter::once(&NO_TEXT_EFFECT)
        .chain(TEXT_EFFECTS)
        .find(|effect| wears(look, effect))
}

fn wears(look: &TextLook, effect: &TextEffect) -> bool {
    let same = |a: &str, b: &str| a.eq_ignore_ascii_case(b);
    let near = |a: f64, b: f64| (a - b).abs() < 1e-6;
    let (outline, width) = effect.outline;
    let (shadow, distance, _, blur) = effect.shadow;
    let (glow, strength, range) = effect.glow;
    same(&look.outline_color, outline)
        && (outline.is_empty() || near(look.outline_width, width))
        && same(&look.shadow_color, shadow)
        && (shadow.is_empty()
            || (near(look.shadow_distance, distance) && near(look.shadow_blur, blur)))
        && same(&look.glow_color, glow)
        && (glow.is_empty() || (near(look.glow_strength, strength) && near(look.glow_range, range)))
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, MaterialText, Segment, TimeRange};
    use cutlass_session::Session;

    use super::*;
    use crate::inspector::text_look;

    fn titled() -> Session {
        let mut draft = Draft::empty("draft");
        let mut text = MaterialText::new("words");
        text.content = "Hi".to_owned();
        text.text_color = "#FFFFFF".to_owned();
        draft.materials.texts.push(text);
        draft.tracks[0]
            .segments
            .push(Segment::new("title", "words", TimeRange::new(0, 1_000_000)));
        Session::open(draft)
    }

    #[test]
    fn each_look_is_one_step_that_replaces_the_last_and_reads_back() {
        let mut session = titled();
        let id = SegmentId::new("title");
        let look = |session: &Session| {
            let draft = session.draft();
            text_look(draft, &draft.tracks[0].segments[0]).unwrap()
        };
        assert_eq!(text_effect_of(&look(&session)).map(|e| e.id), Some("none"));
        for (at, effect) in TEXT_EFFECTS.iter().enumerate() {
            let glow_id = MaterialId::new(format!("glow-{at}"));
            let revision = session.revision();
            session
                .apply(
                    revision,
                    &text_effect_requests(effect, &id, glow_id, "face.ttf"),
                )
                .unwrap();
            assert_eq!(
                text_effect_of(&look(&session)).map(|e| e.id),
                Some(effect.id)
            );
        }
        // The last look's fill is kept.
        let last = TEXT_EFFECTS.last().unwrap().fill.unwrap();
        assert_eq!(session.draft().materials.texts[0].text_color, last);
        let revision = session.revision();
        session
            .apply(
                revision,
                &text_effect_requests(&NO_TEXT_EFFECT, &id, MaterialId::new("g"), "face.ttf"),
            )
            .unwrap();
        let plain = look(&session);
        assert_eq!(text_effect_of(&plain).map(|e| e.id), Some("none"));
        assert!(plain.glow_color.is_empty() && plain.shadow_color.is_empty());
        // None keeps the letters' color.
        assert_eq!(session.draft().materials.texts[0].text_color, last);
    }
}

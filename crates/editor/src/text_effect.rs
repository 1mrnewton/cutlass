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
}

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
    },
    TextEffect {
        id: "shadow",
        name: "Shadow",
        fill: None,
        outline: ("", 0.0),
        shadow: ("#000000", 8.0, -45.0, 0.5),
        glow: ("", 0.0, 0.0),
    },
    TextEffect {
        id: "glow",
        name: "Glow",
        fill: None,
        outline: ("", 0.0),
        shadow: OFF_SHADOW,
        glow: ("#FFFFFF", 0.6, 0.5),
    },
    TextEffect {
        id: "neon",
        name: "Neon",
        fill: Some("#FFFFFF"),
        outline: ("", 0.0),
        shadow: OFF_SHADOW,
        glow: ("#FF2BD6", 0.8, 0.6),
    },
    TextEffect {
        id: "pop",
        name: "Pop",
        fill: Some("#FFD400"),
        outline: ("#000000", 0.1),
        shadow: ("#000000", 10.0, -45.0, 0.0),
        glow: ("", 0.0, 0.0),
    },
    TextEffect {
        id: "retro",
        name: "Retro",
        fill: Some("#FF5A36"),
        outline: ("#FFFFFF", 0.06),
        shadow: ("#2A1B6B", 12.0, -45.0, 0.0),
        glow: ("", 0.0, 0.0),
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
        let retro = &session.draft().materials.texts[0];
        assert_eq!(retro.text_color, "#FF5A36");
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
        assert_eq!(session.draft().materials.texts[0].text_color, "#FF5A36");
    }
}

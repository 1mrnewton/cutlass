//! The text shelf's title styles, and the requests that place one.
//!
//! A style is a plain text clip with a size, a color, a place on the
//! canvas, and an in animation from the shared presets. Every screen adds
//! a title through here, so the phone and the desktop place the same one.

use cutlass_model::{
    ANIMATION_PRESETS, AnimType, AnimationPreset, MaterialId, Micros, SegmentId, TrackId,
};
use cutlass_session::Request;

/// How long a new title lasts.
pub const TITLE_LENGTH: Micros = Micros(3_000_000);

/// A card's id starts with this, so a drop can tell a title from a file.
pub const TITLE_CARD_PREFIX: &str = "title:";

/// One card on the text shelf.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TitleStyle {
    pub id: &'static str,
    /// The card's name, and the words a new clip says.
    pub name: &'static str,
    /// The card's second line: how the title comes in.
    pub meta: &'static str,
    pub size: f64,
    /// `#RRGGBB`.
    pub color: &'static str,
    /// Up from the canvas middle, in canvas half-heights.
    pub y: f64,
    /// The in animation preset's name, if the title has one.
    pub enter: Option<&'static str>,
}

/// The text shelf, in order.
pub const TITLE_STYLES: &[TitleStyle] = &[
    TitleStyle {
        id: "default",
        name: "Default text",
        meta: "Plain",
        size: 80.0,
        color: "#FFFFFF",
        y: 0.0,
        enter: None,
    },
    TitleStyle {
        id: "serif",
        name: "Serif title",
        meta: "Rise",
        size: 96.0,
        color: "#E7D7B4",
        y: 0.0,
        enter: Some("Slide Up"),
    },
    TitleStyle {
        id: "lower-third",
        name: "Lower third",
        meta: "Slide",
        size: 56.0,
        color: "#EFC56A",
        y: -0.7,
        enter: Some("Slide Right"),
    },
    TitleStyle {
        id: "end-card",
        name: "End card",
        meta: "Fade",
        size: 88.0,
        color: "#F0E2C4",
        y: 0.0,
        enter: Some("Fade In"),
    },
];

impl TitleStyle {
    /// The library card id for this style.
    #[must_use]
    pub fn card_id(&self) -> String {
        format!("{TITLE_CARD_PREFIX}{}", self.id)
    }

    /// The in animation this style uses.
    #[must_use]
    pub fn enter_preset(&self) -> Option<&'static AnimationPreset> {
        let name = self.enter?;
        ANIMATION_PRESETS
            .iter()
            .find(|preset| preset.kind == AnimType::In && preset.name == name)
    }
}

/// The style a library card id names.
#[must_use]
pub fn title_style(card_id: &str) -> Option<&'static TitleStyle> {
    let id = card_id.strip_prefix(TITLE_CARD_PREFIX)?;
    TITLE_STYLES.iter().find(|style| style.id == id)
}

/// Where a new title goes and what it is called.
#[derive(Clone, Debug, PartialEq)]
pub struct TitlePlace {
    /// A text row; a missing one is added.
    pub track_id: TrackId,
    pub segment_id: SegmentId,
    pub material_id: MaterialId,
    pub start: Micros,
    /// The face file the words are drawn in.
    pub face: String,
}

/// The requests that place `style` as one undo step: the clip, then its
/// size, color, place, and in animation where they differ from a plain one.
#[must_use]
pub fn title_requests(style: &TitleStyle, place: &TitlePlace) -> Vec<Request> {
    let segment_id = place.segment_id.clone();
    let mut requests = vec![Request::AddText {
        track_id: place.track_id.clone(),
        segment_id: segment_id.clone(),
        material_id: place.material_id.clone(),
        start: Micros(place.start.0.max(0)),
        duration: TITLE_LENGTH,
        content: style.name.to_owned(),
        face: place.face.clone(),
    }];
    if (style.size - cutlass_session::NEW_TEXT_SIZE).abs() > f64::EPSILON {
        requests.push(Request::SetTextSize {
            segment_id: segment_id.clone(),
            size: style.size,
            face: place.face.clone(),
        });
    }
    if !style.color.eq_ignore_ascii_case("#FFFFFF") {
        requests.push(Request::SetTextColor {
            segment_id: segment_id.clone(),
            color: style.color.to_owned(),
            face: place.face.clone(),
        });
    }
    if style.y != 0.0 {
        requests.push(Request::SetPosition {
            segment_id: segment_id.clone(),
            x: 0.0,
            y: style.y,
        });
    }
    if let Some(preset) = style.enter_preset() {
        requests.push(Request::SetAnimation {
            segment_id,
            kind: AnimType::In,
            animation: Some(preset.resource_id.to_owned()),
            duration: preset.duration,
        });
    }
    requests
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, TrackType, animated_at, segment_animations};
    use cutlass_session::Session;

    use super::*;

    fn place() -> TitlePlace {
        TitlePlace {
            track_id: TrackId::new("text-row"),
            segment_id: SegmentId::new("title-1"),
            material_id: MaterialId::new("text-1"),
            start: Micros(1_000_000),
            face: "Arial.ttf".to_owned(),
        }
    }

    #[test]
    fn every_style_names_a_card_and_a_real_animation() {
        for style in TITLE_STYLES {
            assert_eq!(title_style(&style.card_id()), Some(style));
            assert_eq!(style.enter.is_some(), style.enter_preset().is_some());
        }
        assert_eq!(title_style("default"), None);
        assert_eq!(title_style("title:nope"), None);
    }

    #[test]
    fn a_lower_third_lands_low_in_its_color_and_slides_in() {
        let style = title_style("title:lower-third").unwrap();
        let mut session = Session::open(Draft::empty("D"));
        session.apply(0, &title_requests(style, &place())).unwrap();
        let draft = session.draft();
        let track = draft
            .tracks
            .iter()
            .find(|track| track.id.as_str() == "text-row")
            .unwrap();
        assert_eq!(track.kind, TrackType::Text);
        let segment = &track.segments[0];
        assert_eq!(segment.target.start, Micros(1_000_000));
        assert_eq!(segment.target.duration, TITLE_LENGTH);
        let text = &draft.materials.texts[0];
        assert_eq!(text.words().unwrap(), "Lower third");
        assert!((text.font_size - 56.0).abs() < 1e-9);
        assert_eq!(text.text_color, "#EFC56A");
        assert!((segment.clip.transform.y + 0.7).abs() < 1e-9);
        let animations = segment_animations(draft, segment).unwrap();
        assert_eq!(animations.animations.len(), 1);
        assert_eq!(
            animations.animations[0].resource_id,
            "cutlass-anim-in-slide-right"
        );
        assert!(animated_at(draft, segment, Micros(1_000_000)).is_some());
    }

    #[test]
    fn default_text_is_a_plain_clip() {
        let requests = title_requests(&TITLE_STYLES[0], &place());
        assert_eq!(requests.len(), 1);
        assert!(matches!(requests[0], Request::AddText { .. }));
    }
}

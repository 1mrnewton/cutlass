//! Title templates: a scene of text widgets, each with its own words, size,
//! color, place, timing and animations, evaluated over time.
//!
//! A template lands as one compound clip, so it moves and trims as one
//! piece and plays its widgets inside; opening it, or ungrouping it, edits
//! the words. Every screen places a template through here.

use cutlass_model::{
    ANIMATION_PRESETS, AnimType, AnimationPreset, MaterialId, Micros, SegmentId, TrackId,
};
use cutlass_session::Request;

/// A card's id starts with this, so a drop can tell a template from a file.
pub const TEMPLATE_CARD_PREFIX: &str = "template:";

/// One text widget in a template.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Widget {
    pub words: &'static str,
    pub size: f64,
    /// `#RRGGBB`.
    pub color: &'static str,
    /// Right of the canvas middle, in canvas half-widths.
    pub x: f64,
    /// Up from the canvas middle, in canvas half-heights.
    pub y: f64,
    /// When the widget starts, from the template's start.
    pub start: Micros,
    pub length: Micros,
    /// The in, out and loop presets' names, if it has them.
    pub enter: Option<&'static str>,
    pub leave: Option<&'static str>,
    pub looped: Option<&'static str>,
}

const fn widget(words: &'static str, size: f64, color: &'static str, y: f64) -> Widget {
    Widget {
        words,
        size,
        color,
        x: 0.0,
        y,
        start: Micros(0),
        length: Micros(4_000_000),
        enter: None,
        leave: None,
        looped: None,
    }
}

/// One card on the templates shelf.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TitleTemplate {
    pub id: &'static str,
    pub name: &'static str,
    /// The card's second line.
    pub meta: &'static str,
    /// The card's color: its first widget's.
    pub widgets: &'static [Widget],
}

/// The templates shelf, in order.
pub const TITLE_TEMPLATES: &[TitleTemplate] = &[
    TitleTemplate {
        id: "headline",
        name: "Headline",
        meta: "Title and line",
        widgets: &[
            Widget {
                enter: Some("Slide Up"),
                leave: Some("Fade Out"),
                ..widget("Big News", 110.0, "#FFFFFF", 0.12)
            },
            Widget {
                start: Micros(300_000),
                length: Micros(3_700_000),
                enter: Some("Fade In"),
                leave: Some("Fade Out"),
                ..widget("The story in one line", 48.0, "#FFD23F", -0.12)
            },
        ],
    },
    TitleTemplate {
        id: "name-tag",
        name: "Name tag",
        meta: "Lower third",
        widgets: &[
            Widget {
                x: -0.45,
                enter: Some("Slide Right"),
                leave: Some("Slide Left"),
                ..widget("Alex Morgan", 64.0, "#FFFFFF", -0.62)
            },
            Widget {
                x: -0.45,
                start: Micros(250_000),
                length: Micros(3_750_000),
                enter: Some("Slide Right"),
                leave: Some("Slide Left"),
                ..widget("Host", 40.0, "#5CE1E6", -0.78)
            },
        ],
    },
    TitleTemplate {
        id: "countdown",
        name: "Countdown",
        meta: "3, 2, 1",
        widgets: &[
            Widget {
                length: Micros(1_000_000),
                enter: Some("Pop"),
                ..widget("3", 220.0, "#FFFFFF", 0.0)
            },
            Widget {
                start: Micros(1_000_000),
                length: Micros(1_000_000),
                enter: Some("Pop"),
                ..widget("2", 220.0, "#FFFFFF", 0.0)
            },
            Widget {
                start: Micros(2_000_000),
                length: Micros(1_000_000),
                enter: Some("Pop"),
                ..widget("1", 220.0, "#FF4D6D", 0.0)
            },
        ],
    },
    TitleTemplate {
        id: "follow",
        name: "Follow for more",
        meta: "Pulses",
        widgets: &[
            Widget {
                enter: Some("Zoom In"),
                ..widget("Thanks for watching", 64.0, "#FFFFFF", 0.1)
            },
            Widget {
                start: Micros(400_000),
                length: Micros(3_600_000),
                enter: Some("Fade In"),
                looped: Some("Pulse"),
                ..widget("Follow for more", 56.0, "#FF4D6D", -0.1)
            },
        ],
    },
    TitleTemplate {
        id: "quote",
        name: "Quote",
        meta: "Types on",
        widgets: &[
            Widget {
                enter: Some("Typewriter"),
                ..widget("\u{201c}Make it simple.\u{201d}", 84.0, "#F0E2C4", 0.08)
            },
            Widget {
                start: Micros(1_200_000),
                length: Micros(2_800_000),
                enter: Some("Fade In"),
                ..widget("\u{2014} Someone wise", 44.0, "#BFB3A0", -0.14)
            },
        ],
    },
];

impl TitleTemplate {
    /// The library card id for this template.
    #[must_use]
    pub fn card_id(&self) -> String {
        format!("{TEMPLATE_CARD_PREFIX}{}", self.id)
    }

    /// From the template's start to its last widget's end.
    #[must_use]
    pub fn length(&self) -> Micros {
        Micros(
            self.widgets
                .iter()
                .map(|widget| widget.start.0 + widget.length.0)
                .max()
                .unwrap_or(0),
        )
    }

    /// The card's color: its first widget's.
    #[must_use]
    pub fn color(&self) -> &'static str {
        self.widgets
            .first()
            .map_or("#FFFFFF", |widget| widget.color)
    }
}

/// The template a library card id names.
#[must_use]
pub fn title_template(card_id: &str) -> Option<&'static TitleTemplate> {
    let id = card_id.strip_prefix(TEMPLATE_CARD_PREFIX)?;
    TITLE_TEMPLATES.iter().find(|template| template.id == id)
}

fn preset(kind: AnimType, name: Option<&str>) -> Option<&'static AnimationPreset> {
    let name = name?;
    ANIMATION_PRESETS
        .iter()
        .find(|preset| preset.kind == kind && preset.name == name)
}

/// Where a template goes and the ids its pieces take.
#[derive(Clone, Debug, PartialEq)]
pub struct TemplatePlace {
    /// The text row the compound clip lands on; a missing one is added.
    pub track_id: TrackId,
    pub start: Micros,
    /// The face file the words are drawn in.
    pub face: String,
    /// The compound clip and the draft it plays.
    pub segment_id: SegmentId,
    pub material_id: MaterialId,
    /// One row, clip and text per widget, in order. The first row is
    /// `track_id`; the others are new and go again once grouped.
    pub widgets: Vec<(TrackId, SegmentId, MaterialId)>,
}

/// The requests that place `template` as one undo step: each widget on a
/// row of its own with its size, color, place and animations, then the
/// widgets grouped into one compound clip and the extra rows taken away.
#[must_use]
pub fn template_requests(template: &TitleTemplate, place: &TemplatePlace) -> Vec<Request> {
    let mut requests = Vec::new();
    let start = place.start.0.max(0);
    for (index, (widget, (track_id, segment_id, material_id))) in
        template.widgets.iter().zip(&place.widgets).enumerate()
    {
        let track_id = if index == 0 {
            place.track_id.clone()
        } else {
            track_id.clone()
        };
        requests.push(Request::AddText {
            track_id,
            segment_id: segment_id.clone(),
            material_id: material_id.clone(),
            start: Micros(start + widget.start.0),
            duration: widget.length,
            content: widget.words.to_owned(),
            face: place.face.clone(),
        });
        requests.push(Request::SetTextSize {
            segment_id: segment_id.clone(),
            size: widget.size,
            face: place.face.clone(),
        });
        requests.push(Request::SetTextColor {
            segment_id: segment_id.clone(),
            color: widget.color.to_owned(),
            face: place.face.clone(),
        });
        if widget.x != 0.0 || widget.y != 0.0 {
            requests.push(Request::SetPosition {
                segment_id: segment_id.clone(),
                x: widget.x,
                y: widget.y,
            });
        }
        for (kind, name) in [
            (AnimType::In, widget.enter),
            (AnimType::Out, widget.leave),
            (AnimType::Loop, widget.looped),
        ] {
            if let Some(preset) = preset(kind, name) {
                requests.push(Request::SetAnimation {
                    segment_id: segment_id.clone(),
                    kind,
                    animation: Some(preset.resource_id.to_owned()),
                    duration: preset.duration,
                });
            }
        }
    }
    requests.push(Request::CreateCompound {
        segment_ids: place
            .widgets
            .iter()
            .map(|(_, segment_id, _)| segment_id.clone())
            .collect(),
        segment_id: place.segment_id.clone(),
        material_id: place.material_id.clone(),
    });
    for (track_id, _, _) in place.widgets.iter().skip(1) {
        requests.push(Request::RemoveTrack {
            track_id: track_id.clone(),
        });
    }
    requests
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, MaterialRef, TrackType, segment_animations};
    use cutlass_session::Session;

    use super::*;

    fn place(template: &TitleTemplate, track: &str) -> TemplatePlace {
        TemplatePlace {
            track_id: TrackId::new(track),
            start: Micros(1_000_000),
            face: "Arial.ttf".to_owned(),
            segment_id: SegmentId::new("template"),
            material_id: MaterialId::new("template-draft"),
            widgets: (0..template.widgets.len())
                .map(|n| {
                    (
                        TrackId::new(format!("row-{n}")),
                        SegmentId::new(format!("widget-{n}")),
                        MaterialId::new(format!("words-{n}")),
                    )
                })
                .collect(),
        }
    }

    #[test]
    fn every_template_names_a_card_and_real_animations() {
        for template in TITLE_TEMPLATES {
            assert_eq!(title_template(&template.card_id()), Some(template));
            assert!(!template.widgets.is_empty());
            assert!(template.length().0 > 0);
            for widget in template.widgets {
                for (kind, name) in [
                    (AnimType::In, widget.enter),
                    (AnimType::Out, widget.leave),
                    (AnimType::Loop, widget.looped),
                ] {
                    assert_eq!(name.is_some(), preset(kind, name).is_some(), "{name:?}");
                }
            }
        }
        assert_eq!(title_template("headline"), None);
        assert_eq!(title_template("template:nope"), None);
    }

    #[test]
    fn a_template_lands_as_one_compound_clip_and_undoes_in_one_step() {
        let template = title_template("template:headline").unwrap();
        let mut session = Session::open(Draft::empty("D"));
        let before = session.draft().clone();
        session
            .apply(0, &template_requests(template, &place(template, "words")))
            .unwrap();
        let draft = session.draft();
        let rows: Vec<_> = draft
            .tracks
            .iter()
            .filter(|track| track.kind == TrackType::Text)
            .collect();
        assert_eq!(rows.len(), 1, "the extra rows went once grouped");
        assert_eq!(rows[0].id.as_str(), "words");
        let compound = &rows[0].segments[0];
        assert_eq!(compound.id.as_str(), "template");
        assert_eq!(compound.target.start, Micros(1_000_000));
        assert_eq!(compound.target.duration, template.length());
        let Some(MaterialRef::Draft(material)) = draft.materials.get("template-draft") else {
            panic!("compound");
        };
        let inside = material.draft.as_deref().unwrap();
        let widgets: Vec<_> = inside
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .collect();
        assert_eq!(widgets.len(), 2);
        let line = widgets
            .iter()
            .find(|segment| segment.id.as_str() == "widget-1")
            .unwrap();
        assert_eq!(line.target.start, Micros(300_000));
        let text = inside
            .materials
            .texts
            .iter()
            .find(|text| text.id.as_str() == "words-1")
            .unwrap();
        assert_eq!(text.words().unwrap(), "The story in one line");
        assert_eq!(text.text_color, "#FFD23F");
        let animations = segment_animations(inside, line).unwrap();
        assert_eq!(animations.animations.len(), 2, "in and out");

        session.undo().unwrap();
        assert_eq!(session.draft(), &before);
    }

    #[test]
    fn a_widget_with_a_loop_keeps_it() {
        let template = title_template("template:follow").unwrap();
        let mut session = Session::open(Draft::empty("D"));
        session
            .apply(0, &template_requests(template, &place(template, "words")))
            .unwrap();
        let Some(MaterialRef::Draft(material)) = session.draft().materials.get("template-draft")
        else {
            panic!("compound");
        };
        let inside = material.draft.as_deref().unwrap();
        let follow = inside
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .find(|segment| segment.id.as_str() == "widget-1")
            .unwrap();
        let kinds: Vec<_> = segment_animations(inside, follow)
            .unwrap()
            .animations
            .iter()
            .map(|animation| animation.kind.clone())
            .collect();
        assert!(kinds.iter().any(|kind| kind == "loop"), "{kinds:?}");
    }
}

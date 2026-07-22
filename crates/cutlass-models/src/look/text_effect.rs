// --- Text effect presets --------------------------------------------------------------

use crate::clip::{TextBackground, TextShadow, TextStroke};

/// A text effect preset (CapCut text effects): a named combination of the
/// stroke / shadow / background treatments [`crate::TextStyle`] already
/// persists. Applying a preset bakes these fields onto the style (see
/// [`crate::Generator::resolve_presets`]), so the file stays self-describing
/// and renderers never need the catalog.
#[derive(Debug, Clone, PartialEq)]
pub struct TextEffectSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub stroke: Option<TextStroke>,
    pub shadow: Option<TextShadow>,
    pub background: Option<TextBackground>,
}

const TEXT_EFFECTS: &[TextEffectSpec] = &[
    TextEffectSpec {
        id: "neon",
        label: "Neon",
        stroke: Some(TextStroke {
            rgba: crate::Param::Constant([57, 255, 20, 255]),
            width: crate::Param::Constant(4.0),
        }),
        shadow: Some(TextShadow {
            rgba: crate::Param::Constant([57, 255, 20, 200]),
            blur: crate::Param::Constant(0.35),
            distance: crate::Param::Constant(0.0),
        }),
        background: None,
    },
    TextEffectSpec {
        id: "shadow",
        label: "Shadow",
        stroke: None,
        shadow: Some(TextShadow {
            rgba: crate::Param::Constant([0, 0, 0, 230]),
            blur: crate::Param::Constant(0.15),
            distance: crate::Param::Constant(8.0),
        }),
        background: None,
    },
    TextEffectSpec {
        id: "outline",
        label: "Outline",
        stroke: Some(TextStroke {
            rgba: crate::Param::Constant([0, 0, 0, 255]),
            width: crate::Param::Constant(8.0),
        }),
        shadow: None,
        background: None,
    },
    TextEffectSpec {
        id: "glow",
        label: "Glow",
        stroke: None,
        shadow: Some(TextShadow {
            rgba: crate::Param::Constant([255, 255, 255, 220]),
            blur: crate::Param::Constant(0.4),
            distance: crate::Param::Constant(0.0),
        }),
        background: None,
    },
    TextEffectSpec {
        id: "retro",
        label: "Retro",
        stroke: Some(TextStroke {
            rgba: crate::Param::Constant([255, 140, 60, 255]),
            width: crate::Param::Constant(5.0),
        }),
        shadow: Some(TextShadow {
            rgba: crate::Param::Constant([120, 40, 160, 255]),
            blur: crate::Param::Constant(0.05),
            distance: crate::Param::Constant(10.0),
        }),
        background: None,
    },
    TextEffectSpec {
        id: "chrome",
        label: "Chrome",
        stroke: Some(TextStroke {
            rgba: crate::Param::Constant([230, 230, 240, 255]),
            width: crate::Param::Constant(3.0),
        }),
        shadow: Some(TextShadow {
            rgba: crate::Param::Constant([40, 60, 90, 200]),
            blur: crate::Param::Constant(0.2),
            distance: crate::Param::Constant(6.0),
        }),
        background: None,
    },
];

/// Every text effect preset (UI browsing order).
pub fn text_effect_catalog() -> &'static [TextEffectSpec] {
    TEXT_EFFECTS
}

/// The catalog entry for `id`, or `None`.
pub fn text_effect_spec(id: &str) -> Option<&'static TextEffectSpec> {
    TEXT_EFFECTS.iter().find(|s| s.id == id)
}

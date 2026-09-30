// --- Caption templates ----------------------------------------------------------------

use crate::clip::{
    TextAlignH, TextAlignV, TextBackground, TextCase, TextShadow, TextStroke, TextStyle,
};
use crate::look::AnimationRef;
use crate::param::Param;

use super::group::CaptionStyle;
use super::highlight::{CaptionHighlight, CaptionHighlightMode};
use super::layout::CaptionLayout;

/// Glyph outline in a template (reference pixels, 1080p baseline).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaptionStrokeSpec {
    pub rgba: [u8; 4],
    pub width: f32,
}

/// Drop shadow / glow in a template.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaptionShadowSpec {
    pub rgba: [u8; 4],
    /// Blur as a fraction of the font size.
    pub blur: f32,
    /// Offset distance in reference pixels (`0` = a centered glow).
    pub distance: f32,
}

/// Filled card behind the cue in a template.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaptionPlateSpec {
    pub rgba: [u8; 4],
    /// Corner rounding, `0.0` (square) ..= `1.0` (pill).
    pub radius: f32,
}

/// One caption template: a named look plus the segmentation rules that suit it.
///
/// Templates are the caption equivalent of the text effect catalog — the
/// validation *and* UI source of truth, so no shell hard-codes a preset list.
/// Applying one bakes concrete fields onto the group's [`CaptionStyle`], so
/// project files stay self-describing and renderers never consult the catalog.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptionTemplateSpec {
    pub id: &'static str,
    pub label: &'static str,
    /// Font family (`""` = the system default).
    pub font: &'static str,
    /// Font size in reference pixels (1080p baseline).
    pub size: f32,
    pub bold: bool,
    pub case: TextCase,
    /// Fill color (RGBA, 0-255).
    pub fill: [u8; 4],
    pub stroke: Option<CaptionStrokeSpec>,
    pub shadow: Option<CaptionShadowSpec>,
    pub plate: Option<CaptionPlateSpec>,
    /// Characters per line this look reads well at.
    pub max_chars_per_line: u16,
    pub max_lines: u8,
    pub highlight: Option<CaptionHighlight>,
    /// Entrance animation catalog id.
    pub animation_in: Option<&'static str>,
    /// Looping presence animation catalog id (mutually exclusive with
    /// `animation_in`, per the CapCut combo rule).
    pub animation_combo: Option<&'static str>,
}

const CAPTION_TEMPLATES: &[CaptionTemplateSpec] = &[
    CaptionTemplateSpec {
        id: "clean",
        label: "Clean",
        font: "",
        size: 72.0,
        bold: true,
        case: TextCase::Normal,
        fill: [255, 255, 255, 255],
        stroke: Some(CaptionStrokeSpec {
            rgba: [0, 0, 0, 255],
            width: 6.0,
        }),
        shadow: None,
        plate: None,
        max_chars_per_line: 32,
        max_lines: 2,
        highlight: None,
        animation_in: None,
        animation_combo: None,
    },
    CaptionTemplateSpec {
        id: "bold_box",
        label: "Bold box",
        font: "",
        size: 66.0,
        bold: true,
        case: TextCase::Upper,
        fill: [255, 255, 255, 255],
        stroke: None,
        shadow: None,
        plate: Some(CaptionPlateSpec {
            rgba: [0, 0, 0, 220],
            radius: 0.25,
        }),
        max_chars_per_line: 28,
        max_lines: 2,
        highlight: None,
        animation_in: None,
        animation_combo: None,
    },
    CaptionTemplateSpec {
        id: "karaoke_pop",
        label: "Karaoke pop",
        font: "",
        size: 78.0,
        bold: true,
        case: TextCase::Upper,
        fill: [255, 255, 255, 255],
        stroke: Some(CaptionStrokeSpec {
            rgba: [0, 0, 0, 255],
            width: 8.0,
        }),
        shadow: None,
        plate: None,
        max_chars_per_line: 24,
        max_lines: 1,
        highlight: Some(CaptionHighlight {
            mode: CaptionHighlightMode::Word,
            fill: [255, 216, 0, 255],
            plate: None,
            plate_radius: 0.0,
            scale: 1.08,
        }),
        animation_in: Some("char_pop_in"),
        animation_combo: None,
    },
    CaptionTemplateSpec {
        id: "glow",
        label: "Glow",
        font: "",
        size: 74.0,
        bold: true,
        case: TextCase::Normal,
        fill: [255, 255, 255, 255],
        stroke: None,
        shadow: Some(CaptionShadowSpec {
            rgba: [120, 220, 255, 220],
            blur: 0.4,
            distance: 0.0,
        }),
        plate: None,
        max_chars_per_line: 30,
        max_lines: 2,
        highlight: None,
        animation_in: Some("char_fade_in"),
        animation_combo: None,
    },
    CaptionTemplateSpec {
        id: "outline",
        label: "Outline",
        font: "",
        size: 84.0,
        bold: true,
        case: TextCase::Upper,
        fill: [255, 255, 255, 255],
        stroke: Some(CaptionStrokeSpec {
            rgba: [0, 0, 0, 255],
            width: 12.0,
        }),
        shadow: None,
        plate: None,
        max_chars_per_line: 20,
        max_lines: 1,
        highlight: Some(CaptionHighlight {
            mode: CaptionHighlightMode::Word,
            fill: [57, 255, 20, 255],
            plate: None,
            plate_radius: 0.0,
            scale: 1.0,
        }),
        animation_in: None,
        animation_combo: None,
    },
    CaptionTemplateSpec {
        id: "multiline",
        label: "Multiline",
        font: "",
        size: 56.0,
        bold: false,
        case: TextCase::Normal,
        fill: [255, 255, 255, 255],
        stroke: Some(CaptionStrokeSpec {
            rgba: [0, 0, 0, 255],
            width: 4.0,
        }),
        shadow: None,
        plate: None,
        max_chars_per_line: 40,
        max_lines: 3,
        highlight: None,
        animation_in: None,
        animation_combo: None,
    },
];

/// Every caption template (UI browsing order).
pub fn caption_template_catalog() -> &'static [CaptionTemplateSpec] {
    CAPTION_TEMPLATES
}

/// The catalog entry for `id`, or `None`.
pub fn caption_template_spec(id: &str) -> Option<&'static CaptionTemplateSpec> {
    CAPTION_TEMPLATES.iter().find(|s| s.id == id)
}

impl CaptionTemplateSpec {
    /// The shared look this template applies, placed in the caption safe area
    /// derived from [`Self::layout`].
    pub fn style(&self) -> CaptionStyle {
        let layout = self.layout();
        CaptionStyle {
            text: TextStyle {
                font: self.font.to_owned(),
                size: Param::Constant(self.size),
                bold: self.bold,
                italic: false,
                underline: false,
                case: self.case,
                fill: Param::Constant(self.fill),
                align_h: TextAlignH::Center,
                align_v: TextAlignV::Middle,
                stroke: self.stroke.map(|s| TextStroke {
                    rgba: Param::Constant(s.rgba),
                    width: Param::Constant(s.width),
                }),
                background: self.plate.map(|p| TextBackground {
                    rgba: Param::Constant(p.rgba),
                    radius: Param::Constant(p.radius),
                }),
                shadow: self.shadow.map(|s| TextShadow {
                    rgba: Param::Constant(s.rgba),
                    blur: Param::Constant(s.blur),
                    distance: Param::Constant(s.distance),
                }),
                // Treatments above are baked, so no preset owns them.
                effect_preset: None,
                ..TextStyle::default()
            },
            animation_in: self.animation_in.map(AnimationRef::new),
            animation_combo: self.animation_combo.map(AnimationRef::new),
            position: [0.0, layout.position_y()],
            ..CaptionStyle::default()
        }
    }

    /// The segmentation rules this look reads well at (default rules with the
    /// template's line budget).
    pub fn layout(&self) -> CaptionLayout {
        CaptionLayout {
            max_chars_per_line: self.max_chars_per_line,
            max_lines: self.max_lines,
            ..CaptionLayout::default()
        }
    }

    /// The highlight this template plays with, if any.
    pub fn highlight(&self) -> Option<CaptionHighlight> {
        self.highlight.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::look::animation_spec;

    #[test]
    fn catalog_ids_are_unique_and_findable() {
        let catalog = caption_template_catalog();
        assert!(!catalog.is_empty());
        for spec in catalog {
            assert_eq!(caption_template_spec(spec.id).map(|s| s.id), Some(spec.id));
            assert_eq!(
                catalog.iter().filter(|s| s.id == spec.id).count(),
                1,
                "duplicate template id {}",
                spec.id
            );
        }
        assert!(caption_template_spec("nope").is_none());
    }

    #[test]
    fn every_template_builds_a_valid_style_and_layout() {
        for spec in caption_template_catalog() {
            let style = spec.style();
            assert!(
                style.validate().is_ok(),
                "template {} built an invalid style",
                spec.id
            );
            assert!(
                spec.layout().validate().is_ok(),
                "template {} built an invalid layout",
                spec.id
            );
            if let Some(highlight) = spec.highlight() {
                assert!(
                    highlight.validate().is_ok(),
                    "template {} built an invalid highlight",
                    spec.id
                );
            }
        }
    }

    #[test]
    fn every_template_animation_exists_and_fits_its_slot() {
        for spec in caption_template_catalog() {
            if let Some(id) = spec.animation_in {
                let found = animation_spec(id).unwrap_or_else(|| panic!("unknown animation {id}"));
                assert_eq!(found.slot, crate::look::AnimationSlot::In);
            }
            if let Some(id) = spec.animation_combo {
                let found = animation_spec(id).unwrap_or_else(|| panic!("unknown animation {id}"));
                assert_eq!(found.slot, crate::look::AnimationSlot::Combo);
            }
            assert!(
                spec.animation_in.is_none() || spec.animation_combo.is_none(),
                "template {} sets both an entrance and a combo",
                spec.id
            );
        }
    }

    #[test]
    fn templates_bake_treatments_rather_than_naming_a_preset() {
        let outline = caption_template_spec("outline").unwrap().style();
        assert!(outline.text.stroke.is_some());
        assert_eq!(outline.text.effect_preset, None);
    }

    #[test]
    fn karaoke_template_highlights_words() {
        let spec = caption_template_spec("karaoke_pop").unwrap();
        let highlight = spec.highlight().expect("karaoke template highlights");
        assert_eq!(highlight.mode, CaptionHighlightMode::Word);
        assert!(highlight.scale > 1.0);
        assert_eq!(spec.max_lines, 1, "karaoke reads one line at a time");
    }

    #[test]
    fn styles_land_in_the_caption_safe_area() {
        for spec in caption_template_catalog() {
            let style = spec.style();
            assert!(
                style.position[1] > 0.0,
                "template {} must place captions below center",
                spec.id
            );
        }
    }
}

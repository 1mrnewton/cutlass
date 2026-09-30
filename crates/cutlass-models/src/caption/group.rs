// --- Caption groups -------------------------------------------------------------------

use serde::{Deserialize, Serialize};

use crate::clip::TextStyle;
use crate::error::ModelError;
use crate::ids::{CaptionGroupId, MediaId, TrackId};
use crate::look::{AnimationRef, AnimationSlot, LayerStyles, animation_spec};

use super::highlight::CaptionHighlight;
use super::layout::CaptionLayout;

/// Which cues a group-wide restyle rewrites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptionStyleScope {
    /// Rewrite every cue, discarding individual styling (CapCut "Apply to
    /// all"). Clears each cue's override flag.
    #[default]
    All,
    /// Leave hand-styled cues (`style_override`) untouched.
    KeepOverrides,
}

/// The subtitle file formats captions can be imported from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptionFileFormat {
    /// SubRip (`.srt`).
    Srt,
    /// WebVTT (`.vtt`).
    Vtt,
}

impl CaptionFileFormat {
    /// Stable wire id (the serde name), also the file extension.
    pub const fn id(self) -> &'static str {
        match self {
            Self::Srt => "srt",
            Self::Vtt => "vtt",
        }
    }

    /// The format for a file extension, case-insensitively.
    pub fn from_extension(extension: &str) -> Option<Self> {
        match extension
            .trim_start_matches('.')
            .to_ascii_lowercase()
            .as_str()
        {
            "srt" => Some(Self::Srt),
            "vtt" | "webvtt" => Some(Self::Vtt),
            _ => None,
        }
    }
}

/// Where a caption group's cues came from — the provenance that decides
/// whether re-running recognition is meaningful and what the UI calls the
/// group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CaptionSource {
    /// Typed in the editor.
    Manual,
    /// Parsed from a subtitle file.
    Imported { format: CaptionFileFormat },
    /// Produced by speech recognition over one media asset's audio.
    Auto {
        media: MediaId,
        /// BCP-47-ish language tag the recognizer was told to use, or `None`
        /// for auto-detect.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        language: Option<String>,
        /// Recognizer model id, so a re-run can be compared against the
        /// original.
        model: String,
    },
}

impl CaptionSource {
    /// Whether this group's cues can be regenerated from audio.
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto { .. })
    }
}

/// The shared look a caption group applies to its cues.
///
/// This is the group's *template*, not the render truth: every cue clip carries
/// its own `TextStyle`, so cues render, animate, and export exactly like any
/// other text clip and a single line can be emphasized without leaving the
/// group. Applying a style writes through to the member cues (see
/// [`CaptionStyleScope`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptionStyle {
    /// Typography and glyph treatments shared by every cue.
    #[serde(default)]
    pub text: TextStyle,
    /// Layer-quad styles (shadow / glow / outline / plate) shared by every cue.
    #[serde(default, skip_serializing_if = "LayerStyles::is_empty")]
    pub styles: LayerStyles,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animation_in: Option<AnimationRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animation_out: Option<AnimationRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animation_combo: Option<AnimationRef>,
    /// Canvas placement for every cue, in the
    /// [`ClipTransform::position`](crate::ClipTransform) convention (offset
    /// from the canvas center as a fraction of canvas size, `+y` down).
    #[serde(default)]
    pub position: [f32; 2],
    /// Uniform scale for every cue (`1.0` = the style's own size).
    #[serde(default = "unit_scale")]
    pub scale: f32,
}

fn unit_scale() -> f32 {
    1.0
}

impl Default for CaptionStyle {
    fn default() -> Self {
        Self {
            text: TextStyle::default(),
            styles: LayerStyles::default(),
            animation_in: None,
            animation_out: None,
            animation_combo: None,
            position: [0.0, CaptionLayout::default().position_y()],
            scale: 1.0,
        }
    }
}

impl CaptionStyle {
    pub fn validate(&self) -> Result<(), ModelError> {
        self.text.validate()?;
        self.styles.validate()?;
        validate_slot(&self.animation_in, AnimationSlot::In)?;
        validate_slot(&self.animation_out, AnimationSlot::Out)?;
        validate_slot(&self.animation_combo, AnimationSlot::Combo)?;
        if self.animation_combo.is_some()
            && (self.animation_in.is_some() || self.animation_out.is_some())
        {
            return Err(ModelError::InvalidParam(
                "a caption combo animation excludes in/out animations".into(),
            ));
        }
        if !self.position.iter().all(|v| v.is_finite()) {
            return Err(ModelError::InvalidParam(
                "caption position must be finite".into(),
            ));
        }
        if !self.scale.is_finite() || !(0.01..=10.0).contains(&self.scale) {
            return Err(ModelError::InvalidParam(
                "caption scale must be finite and within 0.01..=10".into(),
            ));
        }
        Ok(())
    }

    /// The animation slots as `(slot, ref)` pairs, including cleared slots, so
    /// callers can write all three through to a cue in one pass.
    pub fn animation_slots(&self) -> [(AnimationSlot, Option<AnimationRef>); 3] {
        [
            (AnimationSlot::In, self.animation_in.clone()),
            (AnimationSlot::Out, self.animation_out.clone()),
            (AnimationSlot::Combo, self.animation_combo.clone()),
        ]
    }
}

/// Caption presets are text presets: per-character reveals are welcome, so
/// `text_only` catalog entries are allowed here (unlike a generic clip).
fn validate_slot(animation: &Option<AnimationRef>, slot: AnimationSlot) -> Result<(), ModelError> {
    let Some(animation) = animation else {
        return Ok(());
    };
    let spec = animation_spec(&animation.id)
        .ok_or_else(|| ModelError::InvalidParam(format!("unknown animation '{}'", animation.id)))?;
    if spec.slot != slot {
        return Err(ModelError::InvalidParam(format!(
            "animation '{}' does not fit that slot",
            animation.id
        )));
    }
    animation.clone().normalized_for(spec).map(|_| ())
}

/// A set of caption cues sharing one style, layout, and provenance.
///
/// The cues themselves are ordinary text clips on `track`; this is the entity
/// that makes "restyle every caption" and "re-run recognition" single
/// operations rather than N-clip loops.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptionGroup {
    pub id: CaptionGroupId,
    /// Display name for the caption list ("Auto captions (English)").
    pub label: String,
    /// The text lane holding this group's cues.
    pub track: TrackId,
    pub style: CaptionStyle,
    #[serde(default)]
    pub layout: CaptionLayout,
    pub source: CaptionSource,
    /// Caption template id (see
    /// [`caption_template_catalog`](super::caption_template_catalog)) the style
    /// came from, for showing the selected chip. `None` once hand-edited.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    /// Word/line highlighting for playback. `None` (and absent from saves) for
    /// plain captions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub highlight: Option<CaptionHighlight>,
}

impl CaptionGroup {
    /// A group with default styling on `track`.
    pub fn new(track: TrackId, label: impl Into<String>, source: CaptionSource) -> Self {
        Self {
            id: CaptionGroupId::next(),
            label: label.into(),
            track,
            style: CaptionStyle::default(),
            layout: CaptionLayout::default(),
            source,
            template: None,
            highlight: None,
        }
    }

    /// Whether cues in this group should be highlighted as they play.
    pub fn highlights(&self) -> Option<&CaptionHighlight> {
        self.highlight.as_ref().filter(|h| h.is_active())
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if self.label.len() > 256 {
            return Err(ModelError::InvalidParam(
                "caption group label is too long (max 256 bytes)".into(),
            ));
        }
        self.style.validate()?;
        self.layout.validate()?;
        if let Some(highlight) = &self.highlight {
            highlight.validate()?;
        }
        if let Some(template) = &self.template
            && super::caption_template_spec(template).is_none()
        {
            return Err(ModelError::InvalidParam(format!(
                "unknown caption template '{template}'"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group() -> CaptionGroup {
        CaptionGroup::new(TrackId::from_raw(1), "Captions", CaptionSource::Manual)
    }

    #[test]
    fn default_group_validates_and_sits_in_the_safe_area() {
        let group = group();
        assert!(group.validate().is_ok());
        assert!(group.style.position[1] > 0.0, "captions sit below center");
        assert!(group.highlights().is_none());
    }

    #[test]
    fn validate_rejects_a_mismatched_animation_slot() {
        let mut group = group();
        group.style.animation_in = Some(AnimationRef::new("fade_out"));
        assert!(group.validate().is_err());
    }

    #[test]
    fn validate_accepts_text_only_per_character_presets() {
        let mut group = group();
        group.style.animation_in = Some(AnimationRef::new("char_typewriter"));
        assert!(group.validate().is_ok());
    }

    #[test]
    fn validate_rejects_unknown_animations_and_templates() {
        let mut group = group();
        group.style.animation_combo = Some(AnimationRef::new("nope"));
        assert!(group.validate().is_err());

        let mut group = self::group();
        group.template = Some("nope".into());
        assert!(group.validate().is_err());
    }

    #[test]
    fn validate_rejects_combo_alongside_edge_animations() {
        let mut group = group();
        group.style.animation_in = Some(AnimationRef::new("fade_in"));
        group.style.animation_combo = Some(AnimationRef::new("pulse"));
        assert!(group.validate().is_err());
    }

    #[test]
    fn highlights_ignores_an_off_highlight() {
        let mut group = group();
        group.highlight = Some(CaptionHighlight::default());
        assert!(group.highlights().is_none());
        group.highlight = Some(CaptionHighlight::word([255, 0, 0, 255]));
        assert!(group.highlights().is_some());
    }

    #[test]
    fn file_formats_map_from_extensions() {
        assert_eq!(
            CaptionFileFormat::from_extension(".SRT"),
            Some(CaptionFileFormat::Srt)
        );
        assert_eq!(
            CaptionFileFormat::from_extension("vtt"),
            Some(CaptionFileFormat::Vtt)
        );
        assert_eq!(CaptionFileFormat::from_extension("txt"), None);
    }

    #[test]
    fn source_roundtrips_through_json() {
        let auto = CaptionSource::Auto {
            media: MediaId::from_raw(3),
            language: Some("en".into()),
            model: "ggml-base.en".into(),
        };
        let json = serde_json::to_string(&auto).unwrap();
        assert_eq!(
            serde_json::from_str::<CaptionSource>(&json).unwrap(),
            auto,
            "auto provenance must survive a save"
        );
        assert!(auto.is_auto());
        assert!(!CaptionSource::Manual.is_auto());
    }
}

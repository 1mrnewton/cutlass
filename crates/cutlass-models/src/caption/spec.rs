// --- Caption creation specs -----------------------------------------------------------

use serde::{Deserialize, Serialize};

use crate::error::ModelError;
use crate::ids::{CaptionGroupId, TrackId};
use crate::time::TimeRange;

use super::cue::{CaptionCue, CaptionWord};
use super::group::{CaptionGroup, CaptionSource, CaptionStyle};
use super::highlight::CaptionHighlight;
use super::layout::CaptionLayout;
use super::template::caption_template_spec;

/// How many cues one caption group may hold. A feature-length transcript is a
/// few thousand lines; the ceiling exists so a malformed import can't try to
/// place a million clips in one command.
pub const MAX_CAPTION_CUES: usize = 5_000;

/// Everything needed to create a [`CaptionGroup`], minus the id the timeline
/// allocates.
///
/// `style`, `layout`, and `highlight` each fall back to the named `template`
/// (and then to the defaults) when omitted, so callers can say "captions in the
/// karaoke look" without restating its fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptionGroupSpec {
    /// The text lane the cues land on.
    pub track: TrackId,
    /// Display name for the caption list.
    pub label: String,
    pub source: CaptionSource,
    /// Caption template id (see
    /// [`caption_template_catalog`](super::caption_template_catalog)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<CaptionStyle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<CaptionLayout>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub highlight: Option<CaptionHighlight>,
}

impl CaptionGroupSpec {
    /// A spec for hand-typed captions on `track`.
    pub fn manual(track: TrackId, label: impl Into<String>) -> Self {
        Self {
            track,
            label: label.into(),
            source: CaptionSource::Manual,
            template: None,
            style: None,
            layout: None,
            highlight: None,
        }
    }

    pub fn with_template(mut self, template: impl Into<String>) -> Self {
        self.template = Some(template.into());
        self
    }

    /// Resolve into a group with a freshly allocated id, filling unset fields
    /// from the template. Validates before allocating, so a rejected spec never
    /// advances the id allocator.
    pub fn resolve(&self) -> Result<CaptionGroup, ModelError> {
        let template = match &self.template {
            Some(id) => Some(caption_template_spec(id).ok_or_else(|| {
                ModelError::InvalidParam(format!("unknown caption template '{id}'"))
            })?),
            None => None,
        };
        let style = self
            .style
            .clone()
            .or_else(|| template.map(|t| t.style()))
            .unwrap_or_default();
        let layout = self
            .layout
            .or_else(|| template.map(|t| t.layout()))
            .unwrap_or_default();
        let highlight = self
            .highlight
            .clone()
            .or_else(|| template.and_then(|t| t.highlight()));

        style.validate()?;
        layout.validate()?;
        if let Some(highlight) = &highlight {
            highlight.validate()?;
        }

        let group = CaptionGroup {
            id: CaptionGroupId::next(),
            label: self.label.clone(),
            track: self.track,
            style,
            layout,
            source: self.source.clone(),
            template: self.template.clone(),
            highlight,
        };
        group.validate()?;
        Ok(group)
    }
}

/// One caption cue to create: its text, timeline placement, and the metadata
/// that recognition or a subtitle file supplied.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptionCueSpec {
    /// The line's text, newlines included for a multi-line cue.
    pub text: String,
    /// Where the cue sits on the timeline (at the timeline rate).
    pub timeline: TimeRange,
    /// Per-word timings for highlight rendering, clip-relative.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<CaptionWord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

impl CaptionCueSpec {
    pub fn new(text: impl Into<String>, timeline: TimeRange) -> Self {
        Self {
            text: text.into(),
            timeline,
            words: Vec::new(),
            speaker: None,
            confidence: None,
        }
    }

    pub fn with_words(mut self, words: Vec<CaptionWord>) -> Self {
        self.words = words;
        self
    }

    /// The cue metadata this spec describes, bound to `group` at `index`.
    pub fn cue(&self, group: CaptionGroupId, index: u32) -> CaptionCue {
        CaptionCue {
            group,
            index,
            words: self.words.clone(),
            speaker: self.speaker.clone(),
            confidence: self.confidence,
            text_edited: false,
            style_override: false,
        }
    }

    /// Validate the spec's own contents (text, word timings). Placement is
    /// validated by the timeline, which knows the lane and its neighbors.
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.timeline.is_empty() || self.timeline.start.value < 0 {
            return Err(ModelError::InvalidRange);
        }
        // A blank cue would render nothing and be unselectable in the list.
        if self.text.trim().is_empty() {
            return Err(ModelError::InvalidParam("a caption cue needs text".into()));
        }
        self.cue(CaptionGroupId::from_raw(0), 0)
            .validate(&self.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::Rational;

    fn range(start: i64, duration: i64) -> TimeRange {
        TimeRange::at_rate(start, duration, Rational::new(30, 1))
    }

    #[test]
    fn spec_resolves_template_style_layout_and_highlight() {
        let spec =
            CaptionGroupSpec::manual(TrackId::from_raw(1), "Captions").with_template("karaoke_pop");
        let group = spec.resolve().unwrap();
        assert_eq!(group.template.as_deref(), Some("karaoke_pop"));
        assert_eq!(group.layout.max_lines, 1);
        assert!(group.highlights().is_some());
        assert!(group.style.animation_in.is_some());
    }

    #[test]
    fn explicit_fields_win_over_the_template() {
        let layout = CaptionLayout {
            max_lines: 3,
            ..CaptionLayout::default()
        };
        let mut spec =
            CaptionGroupSpec::manual(TrackId::from_raw(1), "Captions").with_template("karaoke_pop");
        spec.layout = Some(layout);
        let group = spec.resolve().unwrap();
        assert_eq!(group.layout.max_lines, 3);
    }

    #[test]
    fn resolve_rejects_an_unknown_template() {
        let spec = CaptionGroupSpec::manual(TrackId::from_raw(1), "Captions").with_template("nope");
        assert!(spec.resolve().is_err());
    }

    #[test]
    fn resolve_allocates_distinct_ids() {
        let spec = CaptionGroupSpec::manual(TrackId::from_raw(1), "Captions");
        assert_ne!(spec.resolve().unwrap().id, spec.resolve().unwrap().id);
    }

    #[test]
    fn cue_spec_validates_text_and_placement() {
        assert!(
            CaptionCueSpec::new("Hello", range(0, 30))
                .validate()
                .is_ok()
        );
        assert!(CaptionCueSpec::new("   ", range(0, 30)).validate().is_err());
        assert!(CaptionCueSpec::new("Hi", range(0, 0)).validate().is_err());
        assert!(CaptionCueSpec::new("Hi", range(-5, 30)).validate().is_err());
    }

    #[test]
    fn cue_spec_validates_word_ranges_against_its_own_text() {
        let spec =
            CaptionCueSpec::new("hi", range(0, 30)).with_words(vec![CaptionWord::new(0, 10, 0..9)]);
        assert!(spec.validate().is_err(), "range past the text is rejected");
    }
}

// --- Caption cue metadata -------------------------------------------------------------

use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::error::ModelError;
use crate::ids::CaptionGroupId;

/// Ceiling on per-cue word timings. A caption line is a handful of words; this
/// only exists so a malformed import can't hand the renderer an unbounded
/// table to binary-search every frame.
pub const MAX_CAPTION_WORDS: usize = 512;

/// One word of a caption cue with its own timing, for karaoke/word-highlight
/// rendering.
///
/// Times are **clip-relative milliseconds** — they ride the cue when it moves,
/// and survive a trim (the sampler clamps rather than the model rejecting, so
/// shortening a cue never invalidates its timings).
///
/// The word's text is not duplicated: `range` is a byte range into the cue
/// clip's [`Generator::Text`](crate::Generator::Text) content, which maps
/// straight onto the rasterizer's cluster ranges.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptionWord {
    /// Start offset from the cue's start, in milliseconds.
    pub start_ms: u32,
    /// End offset from the cue's start, in milliseconds (`>= start_ms`).
    pub end_ms: u32,
    /// Byte range of this word within the cue's text.
    pub range: Range<u32>,
}

impl CaptionWord {
    pub fn new(start_ms: u32, end_ms: u32, range: Range<u32>) -> Self {
        Self {
            start_ms,
            end_ms,
            range,
        }
    }

    /// The word's byte range as a `usize` range, for slicing the cue text.
    pub fn byte_range(&self) -> Range<usize> {
        self.range.start as usize..self.range.end as usize
    }

    /// The word's text, or `""` when the range does not land on character
    /// boundaries of `text` (a defensive read for legacy/hand-edited files).
    pub fn text<'a>(&self, text: &'a str) -> &'a str {
        text.get(self.byte_range()).unwrap_or_default()
    }
}

/// Caption metadata carried by one cue clip.
///
/// The cue's *text* and *style* live where every text clip's do — in
/// `Generator::Text` — so cues inherit typography, treatments, keyframes, and
/// look animations unchanged. This struct only adds what a caption needs on
/// top: which group it belongs to, its order within that group, optional word
/// timings, and the flags that let a group-wide restyle skip hand-tuned lines.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptionCue {
    /// The group this cue belongs to.
    pub group: CaptionGroupId,
    /// Zero-based order within the group, ascending with timeline position.
    /// Kept dense by every caption edit so the cue list reads as "line N".
    pub index: u32,
    /// Per-word timings for highlight/karaoke rendering. Empty when unknown
    /// (manual typing, an SRT without word timings, or after a text edit that
    /// invalidated them).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<CaptionWord>,
    /// Speaker label from diarization or manual tagging.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    /// Recognition confidence in `0..=1`, for flagging lines worth reviewing.
    /// `None` for anything not machine-transcribed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// The user re-worded this line, so re-running recognition must not
    /// silently overwrite it.
    #[serde(default, skip_serializing_if = "is_false")]
    pub text_edited: bool,
    /// The user styled this line individually, so a group restyle with
    /// [`CaptionStyleScope::KeepOverrides`](super::CaptionStyleScope) leaves
    /// it alone.
    #[serde(default, skip_serializing_if = "is_false")]
    pub style_override: bool,
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_false(b: &bool) -> bool {
    !*b
}

impl CaptionCue {
    /// A cue with no word timings — the manual-typing and plain-SRT case.
    pub fn new(group: CaptionGroupId, index: u32) -> Self {
        Self {
            group,
            index,
            words: Vec::new(),
            speaker: None,
            confidence: None,
            text_edited: false,
            style_override: false,
        }
    }

    /// Whether this cue can drive word-level highlighting.
    pub fn has_word_timings(&self) -> bool {
        !self.words.is_empty()
    }

    /// Validate the cue against the text it annotates.
    ///
    /// Word ranges must be ascending, non-overlapping, inside `text`, and on
    /// character boundaries; word times must be ascending. Times are
    /// deliberately *not* checked against the cue's duration: trimming a cue
    /// shorter must not invalidate the project, so
    /// [`active_word_at`](Self::active_word_at) clamps instead.
    pub fn validate(&self, text: &str) -> Result<(), ModelError> {
        if self.words.len() > MAX_CAPTION_WORDS {
            return Err(ModelError::InvalidParam(format!(
                "caption cue has {} word timings (max {MAX_CAPTION_WORDS})",
                self.words.len()
            )));
        }
        if let Some(confidence) = self.confidence
            && (!confidence.is_finite() || !(0.0..=1.0).contains(&confidence))
        {
            return Err(ModelError::InvalidParam(
                "caption confidence must be finite and within 0..=1".into(),
            ));
        }

        let len = u32::try_from(text.len()).unwrap_or(u32::MAX);
        let mut prev_end = 0u32;
        let mut prev_end_ms = 0u32;
        for word in &self.words {
            if word.range.start < prev_end || word.range.end < word.range.start {
                return Err(ModelError::InvalidParam(
                    "caption word ranges must be ascending and non-overlapping".into(),
                ));
            }
            if word.range.end > len {
                return Err(ModelError::InvalidParam(
                    "caption word range is outside the cue text".into(),
                ));
            }
            if text.get(word.byte_range()).is_none() {
                return Err(ModelError::InvalidParam(
                    "caption word range must fall on character boundaries".into(),
                ));
            }
            if word.end_ms < word.start_ms || word.start_ms < prev_end_ms {
                return Err(ModelError::InvalidParam(
                    "caption word times must be ascending".into(),
                ));
            }
            prev_end = word.range.end;
            prev_end_ms = word.end_ms;
        }
        Ok(())
    }

    /// Index of the word active at `ms` (clip-relative milliseconds).
    ///
    /// Before the first word's start, and inside a gap between words, this
    /// holds the most recently *started* word so a highlight never flickers
    /// off mid-line. `None` only when there are no timings, or when `ms` is
    /// before the first word begins. `O(log n)`, safe to call per frame.
    pub fn active_word_at(&self, ms: u32) -> Option<usize> {
        if self.words.is_empty() {
            return None;
        }
        // Last word whose start is at or before `ms`.
        let after = self.words.partition_point(|w| w.start_ms <= ms);
        after.checked_sub(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group() -> CaptionGroupId {
        CaptionGroupId::from_raw(7)
    }

    fn cue(words: Vec<CaptionWord>) -> CaptionCue {
        CaptionCue {
            words,
            ..CaptionCue::new(group(), 0)
        }
    }

    #[test]
    fn word_text_slices_the_cue_content() {
        let word = CaptionWord::new(0, 100, 6..11);
        assert_eq!(word.text("hello world"), "world");
    }

    #[test]
    fn word_text_is_empty_for_a_split_character() {
        // 'é' is two bytes: a range ending inside it is not a boundary.
        let word = CaptionWord::new(0, 100, 0..1);
        assert_eq!(word.text("é"), "");
    }

    #[test]
    fn validate_accepts_ascending_non_overlapping_words() {
        let cue = cue(vec![
            CaptionWord::new(0, 200, 0..5),
            CaptionWord::new(200, 400, 6..11),
        ]);
        assert!(cue.validate("hello world").is_ok());
    }

    #[test]
    fn validate_rejects_overlapping_ranges() {
        let cue = cue(vec![
            CaptionWord::new(0, 200, 0..7),
            CaptionWord::new(200, 400, 6..11),
        ]);
        assert!(cue.validate("hello world").is_err());
    }

    #[test]
    fn validate_rejects_ranges_past_the_text() {
        let cue = cue(vec![CaptionWord::new(0, 200, 0..99)]);
        assert!(cue.validate("hello").is_err());
    }

    #[test]
    fn validate_rejects_ranges_off_a_character_boundary() {
        let cue = cue(vec![CaptionWord::new(0, 200, 0..1)]);
        assert!(cue.validate("é").is_err());
    }

    #[test]
    fn validate_rejects_backwards_times() {
        let cue = cue(vec![
            CaptionWord::new(0, 400, 0..5),
            CaptionWord::new(200, 300, 6..11),
        ]);
        assert!(cue.validate("hello world").is_err());
    }

    #[test]
    fn validate_rejects_out_of_range_confidence() {
        let mut cue = cue(Vec::new());
        cue.confidence = Some(1.5);
        assert!(cue.validate("hi").is_err());
        cue.confidence = Some(f32::NAN);
        assert!(cue.validate("hi").is_err());
        cue.confidence = Some(0.5);
        assert!(cue.validate("hi").is_ok());
    }

    #[test]
    fn validate_rejects_too_many_words() {
        let words = (0..=MAX_CAPTION_WORDS)
            .map(|i| CaptionWord::new(i as u32, i as u32, 0..0))
            .collect();
        assert!(cue(words).validate("").is_err());
    }

    #[test]
    fn active_word_holds_the_last_started_word() {
        let cue = cue(vec![
            CaptionWord::new(0, 100, 0..5),
            CaptionWord::new(300, 400, 6..11),
        ]);
        assert_eq!(cue.active_word_at(0), Some(0));
        assert_eq!(cue.active_word_at(150), Some(0), "gap holds the prior word");
        assert_eq!(cue.active_word_at(300), Some(1));
        assert_eq!(cue.active_word_at(9_999), Some(1), "past the end holds");
    }

    #[test]
    fn active_word_is_none_before_the_first_word_and_without_timings() {
        let cue = cue(vec![CaptionWord::new(50, 100, 0..5)]);
        assert_eq!(cue.active_word_at(10), None);
        assert_eq!(CaptionCue::new(group(), 0).active_word_at(0), None);
    }

    #[test]
    fn cue_omits_defaults_from_saves() {
        let json = serde_json::to_string(&CaptionCue::new(group(), 3)).unwrap();
        assert_eq!(json, r#"{"group":7,"index":3}"#);
    }
}

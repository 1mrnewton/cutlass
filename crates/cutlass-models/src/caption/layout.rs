// --- Caption layout rules -------------------------------------------------------------

use serde::{Deserialize, Serialize};

use crate::error::ModelError;

/// Bounds for [`CaptionLayout`], deliberately wider than the inspector ranges
/// so API and AI clients can go further than the sliders without handing the
/// segmenter degenerate rules.
pub const MIN_CAPTION_CHARS_PER_LINE: u16 = 8;
pub const MAX_CAPTION_CHARS_PER_LINE: u16 = 256;
pub const MAX_CAPTION_LINES: u8 = 6;
pub const MAX_CAPTION_DURATION_MS: u32 = 30_000;

/// How a caption group breaks speech into readable lines, and where those
/// lines sit on the canvas.
///
/// These are the *segmentation* rules: the recognizer and subtitle importers
/// consult them when splitting words into cues, and re-flowing a group applies
/// them again. They are not render-time properties — once a cue exists it is an
/// ordinary text clip whose wrapping comes from its own style.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CaptionLayout {
    /// Soft ceiling on characters per line before the segmenter breaks.
    pub max_chars_per_line: u16,
    /// Lines allowed in one cue before it becomes a new cue.
    pub max_lines: u8,
    /// Shortest a cue may be held, even for a single quick word.
    pub min_duration_ms: u32,
    /// Longest a cue may run before it is split.
    pub max_duration_ms: u32,
    /// Gap left between consecutive cues, so lines visibly change.
    pub min_gap_ms: u32,
    /// Distance from the canvas bottom to the caption block, as a fraction of
    /// canvas height. The default keeps captions above the bottom-20% UI zone
    /// of TikTok and Reels.
    pub safe_area_bottom: f32,
}

/// Keeps captions clear of the short-form platforms' bottom UI overlay.
pub const DEFAULT_SAFE_AREA_BOTTOM: f32 = 0.18;

impl Default for CaptionLayout {
    fn default() -> Self {
        Self {
            max_chars_per_line: 32,
            max_lines: 2,
            min_duration_ms: 600,
            max_duration_ms: 5_000,
            min_gap_ms: 40,
            safe_area_bottom: DEFAULT_SAFE_AREA_BOTTOM,
        }
    }
}

impl CaptionLayout {
    /// Total characters one cue may hold across all its lines.
    pub fn max_chars_per_cue(&self) -> usize {
        usize::from(self.max_chars_per_line) * usize::from(self.max_lines.max(1))
    }

    /// Normalized vertical offset from the canvas center for the caption
    /// block, in the [`ClipTransform::position`](crate::ClipTransform)
    /// convention (`+y` down, fraction of canvas height). Anchoring the block
    /// `safe_area_bottom` above the bottom edge means offsetting it
    /// `0.5 - safe_area_bottom` down from the center.
    pub fn position_y(&self) -> f32 {
        0.5 - self.safe_area_bottom
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if !(MIN_CAPTION_CHARS_PER_LINE..=MAX_CAPTION_CHARS_PER_LINE)
            .contains(&self.max_chars_per_line)
        {
            return Err(ModelError::InvalidParam(format!(
                "caption max_chars_per_line must be within \
                 {MIN_CAPTION_CHARS_PER_LINE}..={MAX_CAPTION_CHARS_PER_LINE}"
            )));
        }
        if self.max_lines == 0 || self.max_lines > MAX_CAPTION_LINES {
            return Err(ModelError::InvalidParam(format!(
                "caption max_lines must be within 1..={MAX_CAPTION_LINES}"
            )));
        }
        if self.min_duration_ms == 0
            || self.max_duration_ms > MAX_CAPTION_DURATION_MS
            || self.min_duration_ms > self.max_duration_ms
        {
            return Err(ModelError::InvalidParam(format!(
                "caption durations must satisfy 0 < min <= max <= {MAX_CAPTION_DURATION_MS} ms"
            )));
        }
        if self.min_gap_ms > self.min_duration_ms {
            return Err(ModelError::InvalidParam(
                "caption min_gap_ms must not exceed min_duration_ms".into(),
            ));
        }
        if !self.safe_area_bottom.is_finite() || !(0.0..=0.5).contains(&self.safe_area_bottom) {
            return Err(ModelError::InvalidParam(
                "caption safe_area_bottom must be finite and within 0..=0.5".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_layout_is_valid_and_two_short_lines() {
        let layout = CaptionLayout::default();
        assert!(layout.validate().is_ok());
        assert_eq!(layout.max_chars_per_cue(), 64);
    }

    #[test]
    fn position_y_sits_below_the_center() {
        let layout = CaptionLayout::default();
        assert!((layout.position_y() - 0.32).abs() < 1e-6);
    }

    #[test]
    fn validate_rejects_degenerate_rules() {
        let bad = |mutate: fn(&mut CaptionLayout)| {
            let mut layout = CaptionLayout::default();
            mutate(&mut layout);
            assert!(layout.validate().is_err());
        };
        bad(|l| l.max_chars_per_line = 1);
        bad(|l| l.max_chars_per_line = u16::MAX);
        bad(|l| l.max_lines = 0);
        bad(|l| l.max_lines = 99);
        bad(|l| l.min_duration_ms = 0);
        bad(|l| l.max_duration_ms = 60_000);
        bad(|l| {
            l.min_duration_ms = 4_000;
            l.max_duration_ms = 1_000;
        });
        bad(|l| l.min_gap_ms = 10_000);
        bad(|l| l.safe_area_bottom = 0.9);
        bad(|l| l.safe_area_bottom = f32::NAN);
    }
}

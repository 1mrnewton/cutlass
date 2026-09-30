// --- Caption word highlighting ---------------------------------------------------------

use serde::{Deserialize, Serialize};

use crate::error::ModelError;

/// Bounds for the active-word emphasis scale.
pub const MIN_HIGHLIGHT_SCALE: f32 = 0.25;
pub const MAX_HIGHLIGHT_SCALE: f32 = 4.0;

/// What a caption group highlights as it plays (CapCut "Highlight captions").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptionHighlightMode {
    /// No highlight — the whole cue renders in its own style.
    #[default]
    Off,
    /// Emphasize the word being spoken (karaoke). Requires word timings on the
    /// cue; without them the cue renders unhighlighted.
    Word,
    /// Emphasize every word up to and including the one being spoken
    /// (progressive fill).
    Line,
}

impl CaptionHighlightMode {
    /// Stable wire/catalog id (the serde name).
    pub const fn id(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Word => "word",
            Self::Line => "line",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Word => "Word",
            Self::Line => "Line",
        }
    }

    /// Whether this mode needs per-word timings to render.
    pub const fn needs_word_timings(self) -> bool {
        !matches!(self, Self::Off)
    }

    pub const ALL: [Self; 3] = [Self::Off, Self::Word, Self::Line];
}

/// How the spoken word is picked out of a caption cue.
///
/// This is a group-level property, sampled per frame at resolve time against
/// the cue's [`CaptionWord`](super::CaptionWord) timings. It deliberately does
/// not live in `TextStyle`: the highlight is a function of playhead time, not a
/// property of the glyph run, and cues without timings must still render.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptionHighlight {
    pub mode: CaptionHighlightMode,
    /// Fill for the highlighted word(s) (RGBA, 0-255).
    pub fill: [u8; 4],
    /// Plate drawn behind the highlighted word(s), or `None` for a pure color
    /// swap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plate: Option<[u8; 4]>,
    /// Plate corner rounding, `0.0` (square) ..= `1.0` (pill) — the
    /// [`TextBackground::radius`](crate::TextBackground) convention.
    #[serde(default)]
    pub plate_radius: f32,
    /// Size multiplier for the highlighted word (`1.0` = no emphasis).
    #[serde(default = "unit_scale")]
    pub scale: f32,
}

fn unit_scale() -> f32 {
    1.0
}

impl Default for CaptionHighlight {
    fn default() -> Self {
        Self {
            mode: CaptionHighlightMode::Off,
            fill: [255, 216, 0, 255],
            plate: None,
            plate_radius: 0.0,
            scale: 1.0,
        }
    }
}

impl CaptionHighlight {
    /// A karaoke word highlight in `fill`.
    pub fn word(fill: [u8; 4]) -> Self {
        Self {
            mode: CaptionHighlightMode::Word,
            fill,
            ..Self::default()
        }
    }

    /// Whether this highlight would change any pixels.
    pub fn is_active(&self) -> bool {
        self.mode.needs_word_timings()
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if !self.plate_radius.is_finite() || !(0.0..=1.0).contains(&self.plate_radius) {
            return Err(ModelError::InvalidParam(
                "caption highlight plate radius must be finite and within 0..=1".into(),
            ));
        }
        if !self.scale.is_finite()
            || !(MIN_HIGHLIGHT_SCALE..=MAX_HIGHLIGHT_SCALE).contains(&self.scale)
        {
            return Err(ModelError::InvalidParam(format!(
                "caption highlight scale must be finite and within \
                 {MIN_HIGHLIGHT_SCALE}..={MAX_HIGHLIGHT_SCALE}"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_mode_is_inactive_and_needs_no_timings() {
        let highlight = CaptionHighlight::default();
        assert!(!highlight.is_active());
        assert!(!highlight.mode.needs_word_timings());
        assert!(highlight.validate().is_ok());
    }

    #[test]
    fn word_mode_is_active_and_needs_timings() {
        let highlight = CaptionHighlight::word([255, 0, 0, 255]);
        assert!(highlight.is_active());
        assert!(highlight.mode.needs_word_timings());
        assert_eq!(highlight.fill, [255, 0, 0, 255]);
    }

    #[test]
    fn validate_rejects_out_of_range_scale_and_radius() {
        let mut highlight = CaptionHighlight::word([0, 0, 0, 255]);
        highlight.scale = 0.0;
        assert!(highlight.validate().is_err());
        highlight.scale = 99.0;
        assert!(highlight.validate().is_err());
        highlight.scale = 1.2;
        highlight.plate_radius = 2.0;
        assert!(highlight.validate().is_err());
        highlight.plate_radius = 1.0;
        assert!(highlight.validate().is_ok());
    }

    #[test]
    fn mode_ids_are_stable_wire_strings() {
        for mode in CaptionHighlightMode::ALL {
            let json = serde_json::to_string(&mode).unwrap();
            assert_eq!(json, format!("\"{}\"", mode.id()));
        }
    }
}

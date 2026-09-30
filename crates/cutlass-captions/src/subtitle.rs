// --- Subtitle cues: the file-shaped view of a caption group ---------------------------

use cutlass_models::{CaptionCueSpec, CaptionFileFormat, CaptionLayout, TimeRange};

use crate::error::CaptionError;
use crate::format::numbered_lines;
use crate::reflow::{estimate_word_timings, wrap};
use crate::timing::{Placement, separate_spans, snap_spans};

#[cfg(test)]
mod tests;

/// One cue as a subtitle file states it: a span in milliseconds and text whose
/// newlines are the file's line breaks.
///
/// The neutral middle for both formats — SRT and WebVTT parse into it, both
/// writers render from it, and placing it on a timeline happens in exactly one
/// place ([`place_subtitles`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubtitleCue {
    pub start_ms: u32,
    pub end_ms: u32,
    pub text: String,
}

impl SubtitleCue {
    pub fn new(start_ms: u32, end_ms: u32, text: impl Into<String>) -> Self {
        Self {
            start_ms,
            end_ms,
            text: text.into(),
        }
    }
}

/// How an imported subtitle file becomes cue clips.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImportOptions {
    /// The group's rules. Only the minimum hold and gap apply on import — the
    /// file's own durations are otherwise authoritative, since someone already
    /// timed them.
    pub layout: CaptionLayout,
    pub placement: Placement,
    /// Re-wrap each cue to `layout.max_chars_per_line` instead of keeping the
    /// file's line breaks.
    pub rewrap: bool,
    /// Estimate per-word timings so imported subtitles can drive the word
    /// highlight (a subtitle file has none).
    pub estimate_words: bool,
}

impl ImportOptions {
    pub fn new(placement: Placement) -> Self {
        Self {
            layout: CaptionLayout::default(),
            placement,
            rewrap: false,
            estimate_words: false,
        }
    }

    pub fn with_layout(mut self, layout: CaptionLayout) -> Self {
        self.layout = layout;
        self
    }
}

/// Turn parsed subtitle cues into cue specs ready for `AddCaptionGroup`.
///
/// Overlapping cues (two speakers talking over each other, which a subtitle
/// file may express but one lane of clips cannot) are resolved by holding the
/// later cue until the earlier one ends.
pub fn place_subtitles(
    cues: &[SubtitleCue],
    options: &ImportOptions,
) -> Result<Vec<CaptionCueSpec>, CaptionError> {
    options.layout.validate()?;
    let usable: Vec<&SubtitleCue> = cues.iter().filter(|cue| !cue.text.is_empty()).collect();
    if usable.is_empty() {
        return Ok(Vec::new());
    }

    let mut spans: Vec<(u32, u32)> = Vec::with_capacity(usable.len());
    let mut previous_end = 0u32;
    for cue in &usable {
        let start = cue.start_ms.max(previous_end);
        let end = cue
            .end_ms
            .max(start.saturating_add(options.layout.min_duration_ms));
        previous_end = end;
        spans.push((start, end));
    }
    separate_spans(&mut spans, options.layout.min_gap_ms);
    let ranges = snap_spans(options.placement, &spans);

    let mut specs = Vec::with_capacity(usable.len());
    for ((cue, span), timeline) in usable.iter().zip(&spans).zip(ranges) {
        let text = if options.rewrap {
            wrap(&cue.text, options.layout.max_chars_per_line)
        } else {
            cue.text.clone()
        };
        let clip_start_ms = options
            .placement
            .ms(timeline.start.value - options.placement.offset_ticks);
        let words = if options.estimate_words {
            estimate_word_timings(
                &text,
                span.0.saturating_sub(clip_start_ms),
                span.1.saturating_sub(clip_start_ms),
            )
        } else {
            Vec::new()
        };
        let spec = CaptionCueSpec {
            text,
            timeline,
            words,
            speaker: None,
            confidence: None,
        };
        spec.validate()?;
        specs.push(spec);
    }
    Ok(specs)
}

/// Read a subtitle file whose format is whatever it turns out to be.
///
/// Import dialogs get handed files whose extension may lie, and both formats
/// are recognizable from their first line, so sniffing beats trusting the name.
pub fn parse_subtitles(input: &str) -> Result<(CaptionFileFormat, Vec<SubtitleCue>), CaptionError> {
    let format = detect_format(input);
    let cues = match format {
        CaptionFileFormat::Srt => crate::srt::parse_srt(input)?,
        CaptionFileFormat::Vtt => crate::vtt::parse_vtt(input)?,
    };
    Ok((format, cues))
}

/// Which format `input` is written in, by its header.
pub fn detect_format(input: &str) -> CaptionFileFormat {
    let header = numbered_lines(input)
        .into_iter()
        .find(|(_, line)| !line.trim().is_empty())
        .map(|(_, line)| line)
        .unwrap_or_default();
    if header.trim_start().starts_with("WEBVTT") {
        CaptionFileFormat::Vtt
    } else {
        CaptionFileFormat::Srt
    }
}

/// Convert placed cue clips back into subtitle cues, for sidecar export.
///
/// Takes `(timeline range, text)` pairs so the caller does the model walk and
/// this stays a pure conversion.
pub fn subtitles_from_clips<I>(clips: I, placement: Placement) -> Vec<SubtitleCue>
where
    I: IntoIterator<Item = (TimeRange, String)>,
{
    let mut cues: Vec<SubtitleCue> = clips
        .into_iter()
        .map(|(range, text)| SubtitleCue {
            start_ms: placement.ms(range.start.value - placement.offset_ticks),
            end_ms: placement.ms(range.end_tick() - placement.offset_ticks),
            text,
        })
        .collect();
    cues.sort_by_key(|cue| (cue.start_ms, cue.end_ms));
    cues
}

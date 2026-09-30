// --- Timed words → readable cues ------------------------------------------------------

use cutlass_models::{CaptionCueSpec, CaptionLayout, CaptionWord, MAX_CAPTION_CUES};

use crate::error::CaptionError;
use crate::timing::{Placement, separate_spans, snap_spans};

#[cfg(test)]
mod tests;

/// Silence between two words that ends a caption line, in milliseconds.
///
/// Short enough to catch a breath between clauses, long enough to ignore the
/// gaps inside normal speech.
pub const DEFAULT_PAUSE_BREAK_MS: u32 = 400;

/// One spoken word with its own timing — the neutral input to segmentation.
///
/// Deliberately not tied to any recognizer: the desktop maps Whisper's
/// `TranscriptWord` into this, and a subtitle importer or a test can build it
/// by hand.
#[derive(Debug, Clone, PartialEq)]
pub struct TimedWord {
    /// Start in milliseconds, on whatever clock the caller is placing from
    /// (asset time for a per-asset transcription).
    pub start_ms: u32,
    /// End in milliseconds.
    pub end_ms: u32,
    pub text: String,
    /// Recognition confidence in `0..=1`, averaged into the cue's confidence
    /// so the UI can flag lines worth reviewing.
    pub confidence: Option<f32>,
}

impl TimedWord {
    pub fn new(start_ms: u32, end_ms: u32, text: impl Into<String>) -> Self {
        Self {
            start_ms,
            end_ms,
            text: text.into(),
            confidence: None,
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }
}

/// How to turn words into cues.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmentOptions {
    /// The group's segmentation rules.
    pub layout: CaptionLayout,
    /// Where the resulting cues land on the timeline.
    pub placement: Placement,
    /// Silence that forces a new cue. Zero disables pause breaking.
    pub pause_break_ms: u32,
    /// Start a new cue after `.`, `!`, or `?`, so a line is one sentence
    /// wherever the sentence is short enough to fit.
    pub break_on_sentence: bool,
}

impl SegmentOptions {
    pub fn new(placement: Placement) -> Self {
        Self {
            layout: CaptionLayout::default(),
            placement,
            pause_break_ms: DEFAULT_PAUSE_BREAK_MS,
            break_on_sentence: true,
        }
    }

    pub fn with_layout(mut self, layout: CaptionLayout) -> Self {
        self.layout = layout;
        self
    }
}

/// Break `words` into cue specs ready for `AddCaptionGroup`.
///
/// Lines break on sentence ends and speech pauses first, then on the layout's
/// character and line limits, then on its maximum duration — the priority that
/// keeps a line a unit of meaning rather than a unit of width. Cues are held
/// for at least `min_duration_ms`, separated by at least `min_gap_ms`, and
/// snapped to whole frames without ever overlapping.
///
/// Runs in one pass over the words, so a feature-length transcript segments in
/// linear time with one allocation per cue.
pub fn segment(
    words: &[TimedWord],
    options: &SegmentOptions,
) -> Result<Vec<CaptionCueSpec>, CaptionError> {
    options.layout.validate()?;
    let words = normalize(words);
    if words.is_empty() {
        return Ok(Vec::new());
    }

    let grouped = group_into_lines(&words, options);
    if grouped.len() > MAX_CAPTION_CUES {
        return Err(CaptionError::TooManyCues {
            count: grouped.len(),
            max: MAX_CAPTION_CUES,
        });
    }

    let drafts: Vec<Draft> = grouped
        .iter()
        .map(|lines| Draft::build(&words, lines, &options.layout))
        .collect();
    place(&drafts, options)
}

/// A word with its text trimmed, its length measured once, and its timing
/// forced forward — everything the packer needs without re-deriving it per
/// candidate break.
struct Word<'a> {
    start_ms: u32,
    end_ms: u32,
    text: &'a str,
    chars: usize,
    confidence: Option<f32>,
    ends_sentence: bool,
}

fn normalize(words: &[TimedWord]) -> Vec<Word<'_>> {
    let mut normalized: Vec<Word<'_>> = Vec::with_capacity(words.len());
    let mut previous_end = 0u32;
    for word in words {
        let text = word.text.trim();
        if text.is_empty() {
            continue;
        }
        let start_ms = word.start_ms.max(previous_end);
        let end_ms = word.end_ms.max(start_ms);
        previous_end = end_ms;
        normalized.push(Word {
            start_ms,
            end_ms,
            text,
            chars: text.chars().count(),
            confidence: word.confidence,
            ends_sentence: ends_sentence(text),
        });
    }
    normalized
}

/// True when `text` closes a sentence, looking past a trailing quote or
/// bracket so `he said "stop!"` still breaks.
fn ends_sentence(text: &str) -> bool {
    text.trim_end_matches(['"', '\'', ')', ']', '»', '”', '’'])
        .chars()
        .next_back()
        .is_some_and(|c| matches!(c, '.' | '!' | '?' | '…' | '。' | '！' | '？'))
}

/// Pack words into cues, each a list of lines of word indices.
fn group_into_lines(words: &[Word<'_>], options: &SegmentOptions) -> Vec<Vec<Vec<usize>>> {
    let layout = &options.layout;
    let max_chars = usize::from(layout.max_chars_per_line);
    let max_lines = usize::from(layout.max_lines.max(1));

    let mut cues: Vec<Vec<Vec<usize>>> = Vec::new();
    let mut lines: Vec<Vec<usize>> = Vec::new();
    let mut line: Vec<usize> = Vec::new();
    let mut line_chars = 0usize;
    let mut cue_start_ms = 0u32;

    for (index, word) in words.iter().enumerate() {
        if !(lines.is_empty() && line.is_empty()) {
            let previous = &words[index - 1];
            let gap = word.start_ms.saturating_sub(previous.end_ms);
            let paused = options.pause_break_ms > 0 && gap >= options.pause_break_ms;
            let sentence = options.break_on_sentence && previous.ends_sentence;
            let overlong = word.end_ms.saturating_sub(cue_start_ms) > layout.max_duration_ms;
            if paused || sentence || overlong {
                flush(&mut cues, &mut lines, &mut line, &mut line_chars);
            }
        }

        if !line.is_empty() && line_chars + 1 + word.chars > max_chars {
            if lines.len() + 1 >= max_lines {
                flush(&mut cues, &mut lines, &mut line, &mut line_chars);
            } else {
                lines.push(std::mem::take(&mut line));
                line_chars = 0;
            }
        }

        if lines.is_empty() && line.is_empty() {
            cue_start_ms = word.start_ms;
        }
        if !line.is_empty() {
            line_chars += 1;
        }
        line_chars += word.chars;
        line.push(index);
    }

    flush(&mut cues, &mut lines, &mut line, &mut line_chars);
    cues
}

/// Close the in-progress line and cue, if either holds anything.
fn flush(
    cues: &mut Vec<Vec<Vec<usize>>>,
    lines: &mut Vec<Vec<usize>>,
    line: &mut Vec<usize>,
    line_chars: &mut usize,
) {
    if !line.is_empty() {
        lines.push(std::mem::take(line));
    }
    *line_chars = 0;
    if !lines.is_empty() {
        cues.push(std::mem::take(lines));
    }
}

/// A cue's rendered text, word table, and span before frame snapping.
struct Draft {
    text: String,
    words: Vec<CaptionWord>,
    start_ms: u32,
    end_ms: u32,
    confidence: Option<f32>,
}

impl Draft {
    fn build(words: &[Word<'_>], lines: &[Vec<usize>], layout: &CaptionLayout) -> Self {
        let mut text = String::new();
        let mut table = Vec::new();
        let mut confidence_sum = 0f64;
        let mut confidence_count = 0u32;
        let (mut start_ms, mut end_ms) = (u32::MAX, 0u32);

        for (line_index, line) in lines.iter().enumerate() {
            if line_index > 0 {
                text.push('\n');
            }
            for (word_index, &index) in line.iter().enumerate() {
                if word_index > 0 {
                    text.push(' ');
                }
                let word = &words[index];
                let offset = text.len();
                text.push_str(word.text);
                table.push(CaptionWord {
                    start_ms: word.start_ms,
                    end_ms: word.end_ms,
                    range: offset as u32..text.len() as u32,
                });
                start_ms = start_ms.min(word.start_ms);
                end_ms = end_ms.max(word.end_ms);
                if let Some(confidence) = word.confidence {
                    confidence_sum += f64::from(confidence);
                    confidence_count += 1;
                }
            }
        }

        let start_ms = start_ms.min(end_ms);
        let mut duration = end_ms - start_ms;
        duration = duration.clamp(layout.min_duration_ms, layout.max_duration_ms.max(1));
        Self {
            text,
            words: table,
            start_ms,
            end_ms: start_ms.saturating_add(duration),
            confidence: (confidence_count > 0)
                .then(|| (confidence_sum / f64::from(confidence_count)) as f32),
        }
    }
}

/// Separate, snap to frames, and rebase word timings onto the placed clips.
fn place(drafts: &[Draft], options: &SegmentOptions) -> Result<Vec<CaptionCueSpec>, CaptionError> {
    let placement = options.placement;
    let mut spans: Vec<(u32, u32)> = drafts
        .iter()
        .map(|draft| (draft.start_ms, draft.end_ms))
        .collect();
    separate_spans(&mut spans, options.layout.min_gap_ms);
    let ranges = snap_spans(placement, &spans);

    let mut specs = Vec::with_capacity(drafts.len());
    for (draft, timeline) in drafts.iter().zip(ranges) {
        // Word times are clip-relative, and snapping moved the clip by up to
        // half a frame: rebase against where the clip actually landed so the
        // highlight stays on the audio.
        let clip_start_ms = placement.ms(timeline.start.value - placement.offset_ticks);
        let words = draft
            .words
            .iter()
            .map(|word| CaptionWord {
                start_ms: word.start_ms.saturating_sub(clip_start_ms),
                end_ms: word.end_ms.saturating_sub(clip_start_ms),
                range: word.range.clone(),
            })
            .collect();
        let spec = CaptionCueSpec {
            text: draft.text.clone(),
            timeline,
            words,
            speaker: None,
            confidence: draft.confidence,
        };
        spec.validate()?;
        specs.push(spec);
    }
    Ok(specs)
}

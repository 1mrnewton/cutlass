//! Owned, normalized transcription output.

/// A complete transcription.
///
/// Segment text is joined with exactly one ASCII space between non-empty
/// segments. Leading, trailing, and repeated whitespace from Whisper is
/// normalized deterministically.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    text: String,
    segments: Vec<TranscriptSegment>,
}

impl Transcript {
    /// Returns the normalized text for the complete recording.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns the normalized, time-ordered segments.
    #[must_use]
    pub fn segments(&self) -> &[TranscriptSegment] {
        &self.segments
    }

    /// Returns whether the transcription contains no spoken text.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Consumes the transcript and returns its text and segments.
    #[must_use]
    pub fn into_parts(self) -> (String, Vec<TranscriptSegment>) {
        (self.text, self.segments)
    }

    #[cfg(test)]
    pub(crate) fn test_fixture(segments: Vec<TranscriptSegment>) -> Self {
        let text = join_segment_text(&segments);
        Self { text, segments }
    }
}

/// One time-ordered transcription segment.
///
/// Timestamps are explicit centiseconds, the native unit returned by
/// `whisper.cpp`. They are always non-negative and `end_centiseconds()` is
/// always greater than or equal to `start_centiseconds()`.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptSegment {
    start_cs: u64,
    end_cs: u64,
    text: String,
    words: Vec<TranscriptWord>,
}

impl TranscriptSegment {
    /// Returns the segment start in centiseconds.
    #[must_use]
    pub fn start_centiseconds(&self) -> u64 {
        self.start_cs
    }

    /// Returns the segment end in centiseconds.
    #[must_use]
    pub fn end_centiseconds(&self) -> u64 {
        self.end_cs
    }

    /// Returns the segment start in finite seconds.
    #[must_use]
    pub fn start_seconds(&self) -> f64 {
        centiseconds_to_seconds(self.start_cs)
    }

    /// Returns the segment end in finite seconds.
    #[must_use]
    pub fn end_seconds(&self) -> f64 {
        centiseconds_to_seconds(self.end_cs)
    }

    /// Returns the normalized segment text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns normalized words when token timestamps were requested.
    ///
    /// The slice is empty when token timestamps were disabled or Whisper did
    /// not provide usable token timestamps.
    #[must_use]
    pub fn words(&self) -> &[TranscriptWord] {
        &self.words
    }

    /// Consumes the segment and returns its timestamps, text, and words.
    #[must_use]
    pub fn into_parts(self) -> (u64, u64, String, Vec<TranscriptWord>) {
        (self.start_cs, self.end_cs, self.text, self.words)
    }

    #[cfg(test)]
    pub(crate) fn test_fixture(
        start_cs: u64,
        end_cs: u64,
        text: impl Into<String>,
        words: Vec<TranscriptWord>,
    ) -> Self {
        Self {
            start_cs,
            end_cs,
            text: text.into(),
            words,
        }
    }
}

/// One word, or one merged group of Whisper subword tokens.
///
/// Word ranges are non-negative, ordered, and contained by their segment.
/// `probability()` is always finite and in the inclusive range `0.0..=1.0`.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptWord {
    start_cs: u64,
    end_cs: u64,
    text: String,
    probability: f32,
}

impl TranscriptWord {
    /// Returns the word start in centiseconds.
    #[must_use]
    pub fn start_centiseconds(&self) -> u64 {
        self.start_cs
    }

    /// Returns the word end in centiseconds.
    #[must_use]
    pub fn end_centiseconds(&self) -> u64 {
        self.end_cs
    }

    /// Returns the word start in finite seconds.
    #[must_use]
    pub fn start_seconds(&self) -> f64 {
        centiseconds_to_seconds(self.start_cs)
    }

    /// Returns the word end in finite seconds.
    #[must_use]
    pub fn end_seconds(&self) -> f64 {
        centiseconds_to_seconds(self.end_cs)
    }

    /// Returns the merged word text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns the mean probability of the merged subword tokens.
    #[must_use]
    pub fn probability(&self) -> f32 {
        self.probability
    }

    /// Consumes the word and returns its timestamps, text, and probability.
    #[must_use]
    pub fn into_parts(self) -> (u64, u64, String, f32) {
        (self.start_cs, self.end_cs, self.text, self.probability)
    }

    #[cfg(test)]
    pub(crate) fn test_fixture(
        start_cs: u64,
        end_cs: u64,
        text: impl Into<String>,
        probability: f32,
    ) -> Self {
        Self {
            start_cs,
            end_cs,
            text: text.into(),
            probability,
        }
    }
}

pub(super) struct RawSegment {
    pub(super) start_cs: i64,
    pub(super) end_cs: i64,
    pub(super) text: String,
    pub(super) tokens: Vec<RawToken>,
}

pub(super) struct RawToken {
    pub(super) text: String,
    pub(super) timestamps_cs: Option<(i64, i64)>,
    pub(super) probability: f32,
}

pub(super) fn normalize_transcript(raw_segments: Vec<RawSegment>) -> Transcript {
    let mut previous_segment_end = 0;
    let mut segments = Vec::with_capacity(raw_segments.len());

    for raw in raw_segments {
        let start_cs = nonnegative_centiseconds(raw.start_cs).max(previous_segment_end);
        let end_cs = nonnegative_centiseconds(raw.end_cs).max(start_cs);
        let words = normalize_words(raw.tokens, start_cs, end_cs);
        let mut text = normalize_segment_text(&raw.text);
        if text.is_empty() && !words.is_empty() {
            text = join_word_text(&words);
        }

        segments.push(TranscriptSegment {
            start_cs,
            end_cs,
            text,
            words,
        });
        previous_segment_end = end_cs;
    }

    let text = join_segment_text(&segments);
    Transcript { text, segments }
}

fn normalize_words(
    raw_tokens: Vec<RawToken>,
    segment_start: u64,
    segment_end: u64,
) -> Vec<TranscriptWord> {
    let mut accumulators = Vec::new();
    let mut current: Option<WordAccumulator> = None;

    for token in raw_tokens {
        let trimmed = token.text.trim();
        if trimmed.is_empty() || is_special_whisper_token(trimmed) {
            continue;
        }

        let Some((raw_start, raw_end)) = token.timestamps_cs else {
            continue;
        };
        let starts_word = token.text.chars().next().is_some_and(char::is_whitespace);
        let piece = if starts_word {
            token.text.trim_start()
        } else {
            token.text.as_str()
        };
        if piece.is_empty() {
            continue;
        }

        let start_cs = nonnegative_centiseconds(raw_start);
        let end_cs = nonnegative_centiseconds(raw_end).max(start_cs);
        if starts_word && let Some(word) = current.take() {
            accumulators.push(word);
        }

        if let Some(word) = current.as_mut() {
            word.push_piece(piece, start_cs, end_cs, token.probability);
        } else {
            current = Some(WordAccumulator::new(
                piece,
                start_cs,
                end_cs,
                token.probability,
            ));
        }
    }

    if let Some(word) = current {
        accumulators.push(word);
    }

    let mut words = Vec::with_capacity(accumulators.len());
    let mut previous_word_end = segment_start;
    for word in accumulators {
        let text = normalize_spacing(&word.text);
        if text.is_empty() {
            continue;
        }

        let start_cs = word
            .start_cs
            .clamp(segment_start, segment_end)
            .max(previous_word_end);
        let end_cs = word.end_cs.clamp(start_cs, segment_end);
        let probability = (word.probability_sum / f64::from(word.probability_count)) as f32;
        words.push(TranscriptWord {
            start_cs,
            end_cs,
            text,
            probability: normalize_probability(probability),
        });
        previous_word_end = end_cs;
    }

    words
}

struct WordAccumulator {
    start_cs: u64,
    end_cs: u64,
    text: String,
    probability_sum: f64,
    probability_count: u32,
}

impl WordAccumulator {
    fn new(text: &str, start_cs: u64, end_cs: u64, probability: f32) -> Self {
        Self {
            start_cs,
            end_cs,
            text: text.to_owned(),
            probability_sum: f64::from(normalize_probability(probability)),
            probability_count: 1,
        }
    }

    fn push_piece(&mut self, text: &str, start_cs: u64, end_cs: u64, probability: f32) {
        self.start_cs = self.start_cs.min(start_cs);
        self.end_cs = self.end_cs.max(end_cs);
        self.text.push_str(text);
        self.probability_sum += f64::from(normalize_probability(probability));
        self.probability_count += 1;
    }
}

fn normalize_probability(probability: f32) -> f32 {
    if probability.is_nan() {
        0.0
    } else {
        probability.clamp(0.0, 1.0)
    }
}

fn nonnegative_centiseconds(value: i64) -> u64 {
    u64::try_from(value).unwrap_or(0)
}

fn centiseconds_to_seconds(value: u64) -> f64 {
    value as f64 / 100.0
}

fn normalize_segment_text(text: &str) -> String {
    text.split_whitespace()
        .filter(|piece| !is_special_whisper_token(piece))
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalize_spacing(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn join_segment_text(segments: &[TranscriptSegment]) -> String {
    segments
        .iter()
        .map(TranscriptSegment::text)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn join_word_text(words: &[TranscriptWord]) -> String {
    words
        .iter()
        .map(TranscriptWord::text)
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_special_whisper_token(text: &str) -> bool {
    let text = text.trim();
    if text.starts_with("<|") && text.ends_with("|>") {
        return true;
    }

    let Some(inner) = text
        .strip_prefix('[')
        .and_then(|text| text.strip_suffix(']'))
    else {
        return false;
    };
    !inner.is_empty()
        && inner.chars().all(|character| {
            character.is_ascii_uppercase()
                || character.is_ascii_digit()
                || matches!(character, '_' | '-' | ':' | '.')
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_tokens_segments_and_probabilities() {
        let transcript = normalize_transcript(vec![
            RawSegment {
                start_cs: -10,
                end_cs: 50,
                text: "  Hello   world!  ".into(),
                tokens: vec![
                    token("<|startoftranscript|>", 0, 0, 1.0),
                    token(" Hel", -5, 20, f32::NAN),
                    token("lo", 10, 25, f32::INFINITY),
                    token(" world", 15, 45, -2.0),
                    token("!", 44, 60, 0.8),
                    token("[EOT]", 50, 50, 1.0),
                ],
            },
            RawSegment {
                start_cs: 40,
                end_cs: 30,
                text: "  Again  ".into(),
                tokens: vec![token(" Again", 35, 35, 0.5)],
            },
        ]);

        assert_eq!(transcript.text(), "Hello world! Again");
        assert_eq!(transcript.segments().len(), 2);

        let first = &transcript.segments()[0];
        assert_eq!((first.start_cs, first.end_cs), (0, 50));
        assert_eq!(first.words.len(), 2);
        assert_eq!(first.words[0].text(), "Hello");
        assert_eq!(
            (
                first.words[0].start_centiseconds(),
                first.words[0].end_centiseconds()
            ),
            (0, 25)
        );
        assert_eq!(first.words[0].probability(), 0.5);
        assert_eq!(first.words[1].text(), "world!");
        assert_eq!(
            (
                first.words[1].start_centiseconds(),
                first.words[1].end_centiseconds()
            ),
            (25, 50)
        );
        assert!((first.words[1].probability() - 0.4).abs() < f32::EPSILON);

        let second = &transcript.segments()[1];
        assert_eq!((second.start_cs, second.end_cs), (50, 50));
        assert_eq!(
            (
                second.words[0].start_centiseconds(),
                second.words[0].end_centiseconds()
            ),
            (50, 50)
        );
        assert!(second.start_seconds().is_finite());
        assert!(second.end_seconds().is_finite());
    }

    #[test]
    fn skips_missing_timestamps_and_uses_word_text_as_fallback() {
        let transcript = normalize_transcript(vec![RawSegment {
            start_cs: 10,
            end_cs: 20,
            text: " [BLANK_AUDIO] ".into(),
            tokens: vec![
                RawToken {
                    text: " ignored".into(),
                    timestamps_cs: None,
                    probability: 1.0,
                },
                token(" useful", 10, 20, 0.9),
            ],
        }]);

        assert_eq!(transcript.text(), "useful");
        assert_eq!(transcript.segments()[0].text(), "useful");
        assert_eq!(transcript.segments()[0].words()[0].text(), "useful");
    }

    #[test]
    fn does_not_treat_lowercase_bracketed_speech_as_special() {
        assert!(!is_special_whisper_token("[hello]"));
        assert!(is_special_whisper_token("[MUSIC]"));
        assert!(is_special_whisper_token("<|endoftext|>"));
    }

    fn token(text: &str, start_cs: i64, end_cs: i64, probability: f32) -> RawToken {
        RawToken {
            text: text.into(),
            timestamps_cs: Some((start_cs, end_cs)),
            probability,
        }
    }
}

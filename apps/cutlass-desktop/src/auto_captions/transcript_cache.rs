// --- Cached transcripts -------------------------------------------------------------

use cutlass_analysis::moments::{
    AnalyzerIdentity, AnalyzerVersion, MAX_MOMENTS_QUERY_LIMIT, MediaContentKey, MomentBatchKey,
    MomentKind, MomentQuery, MomentRecord,
};
use cutlass_transcription::{
    TRANSCRIPT_MOMENT_PAYLOAD_VERSION, Transcript, transcript_to_moment_batch,
};
use tracing::warn;

use crate::analysis_index::{AnalysisIndexError, AnalysisIndexService};

use super::pipeline::RecognizedWord;

/// Who produced the cached words. Bound to the model because `base.en` and a
/// larger model disagree about wording, and a cache hit must not silently
/// substitute one for the other.
const ANALYZER: &str = "cutlass-transcription/ggml-base.en";

fn batch_key(content_key: MediaContentKey) -> MomentBatchKey {
    MomentBatchKey::new(content_key, identity(), version())
}

fn identity() -> AnalyzerIdentity {
    AnalyzerIdentity::new(ANALYZER).expect("the transcription analyzer identity is well-formed")
}

fn version() -> AnalyzerVersion {
    AnalyzerVersion::new(TRANSCRIPT_MOMENT_PAYLOAD_VERSION)
        .expect("the transcript payload version is non-zero")
}

/// Recognized words for this exact file, if this exact model already ran on it.
///
/// Returns `None` — meaning "transcribe it" — rather than a partial answer for
/// anything unexpected: a missing batch, more words than one query returns, or a
/// payload this build cannot read.
pub(super) fn load(
    index: &AnalysisIndexService,
    content_key: MediaContentKey,
    cancelled: &dyn Fn() -> bool,
) -> Result<Option<Vec<RecognizedWord>>, AnalysisIndexError> {
    if !index.batch_exists(&batch_key(content_key), cancelled)? {
        return Ok(None);
    }

    let query = MomentQuery::new(content_key).with_kind(MomentKind::TRANSCRIPT_WORD);
    let result = index.query_with_limit(&query, MAX_MOMENTS_QUERY_LIMIT, cancelled)?;
    if result.truncated() {
        // Silently captioning only the first N words would look like the
        // recognizer gave up halfway, so the cache declines instead.
        warn!(
            words = result.len(),
            "cached transcript exceeds one query; transcribing again"
        );
        return Ok(None);
    }

    let words: Vec<RecognizedWord> = result
        .records()
        .iter()
        .filter(|record| record.analyzer_identity() == &identity())
        .filter_map(cached_word)
        .collect();
    Ok((!words.is_empty()).then_some(words))
}

/// Cache a fresh transcript for the next run over the same file.
pub(super) fn store(
    index: &AnalysisIndexService,
    content_key: MediaContentKey,
    transcript: &Transcript,
    cancelled: &dyn Fn() -> bool,
) -> Result<(), String> {
    let batch = transcript_to_moment_batch(content_key, identity(), version(), transcript)
        .map_err(|error| error.to_string())?;
    index
        .replace_batch(&batch, cancelled)
        .map_err(|error| error.to_string())
}

/// One `transcript_word` record back into a recognized word.
///
/// The payload is the canonical JSON documented by
/// `TRANSCRIPT_MOMENT_PAYLOAD_VERSION`; only its text is needed, since the
/// record's own span and confidence carry the timing and probability.
fn cached_word(record: &MomentRecord) -> Option<RecognizedWord> {
    let payload = record.payload()?;
    let value: serde_json::Value = serde_json::from_str(payload.as_str()).ok()?;
    let text = value.get("text")?.as_str()?;
    if text.trim().is_empty() {
        return None;
    }
    Some(RecognizedWord {
        start_ms: seconds_to_ms(record.span().start_seconds()),
        end_ms: seconds_to_ms(record.span().end_seconds()),
        text: text.to_owned(),
        confidence: Some(record.confidence().get()),
    })
}

fn seconds_to_ms(seconds: f64) -> u32 {
    let milliseconds = (seconds * 1000.0).round();
    if milliseconds.is_finite() {
        milliseconds.clamp(0.0, f64::from(u32::MAX)) as u32
    } else {
        0
    }
}

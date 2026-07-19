use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use thiserror::Error;
use whisper_rs::{
    DtwMode, DtwModelPreset, FullParams, SamplingStrategy, WhisperContext,
    WhisperContextParameters, WhisperTokenData,
};

use crate::Transcript;
use crate::transcript::{RawSegment, RawToken, normalize_transcript};

const MAX_THREAD_COUNT: usize = 64;
const MAX_LANGUAGE_LENGTH: usize = 32;
const MAX_PCM_SAMPLES: usize = i32::MAX as usize;

/// Options controlling one Whisper transcription.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionOptions {
    /// Whisper language code or language name.
    ///
    /// `None` and `Some("auto")` request automatic language detection.
    pub language: Option<String>,
    /// Translate recognized speech to English instead of transcribing it.
    pub translate: bool,
    /// Request word-level token timestamps.
    ///
    /// Enabled by default. Disabling this leaves each segment's word list
    /// empty while retaining segment timestamps.
    pub token_timestamps: bool,
    /// Number of CPU threads used by the Whisper decoder.
    ///
    /// Valid values are in `1..=64`.
    pub thread_count: usize,
}

impl Default for TranscriptionOptions {
    fn default() -> Self {
        let available = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(4);
        Self {
            language: None,
            translate: false,
            token_timestamps: true,
            thread_count: available.clamp(1, 8),
        }
    }
}

impl TranscriptionOptions {
    /// Validates values before any model file is opened.
    ///
    /// # Errors
    ///
    /// Returns [`TranscriptionOptionsError`] for an unsupported language,
    /// surrounding whitespace, a NUL byte, or an out-of-range thread count.
    pub fn validate(&self) -> Result<(), TranscriptionOptionsError> {
        if !(1..=MAX_THREAD_COUNT).contains(&self.thread_count) {
            return Err(TranscriptionOptionsError::InvalidThreadCount {
                provided: self.thread_count,
                maximum: MAX_THREAD_COUNT,
            });
        }

        let Some(language) = self.language.as_deref() else {
            return Ok(());
        };
        if language.is_empty()
            || language.len() > MAX_LANGUAGE_LENGTH
            || language.trim() != language
            || language.contains('\0')
        {
            return Err(TranscriptionOptionsError::InvalidLanguage {
                language: language.to_owned(),
            });
        }
        if language != "auto" && whisper_rs::get_lang_id(language).is_none() {
            return Err(TranscriptionOptionsError::InvalidLanguage {
                language: language.to_owned(),
            });
        }

        Ok(())
    }

    fn whisper_language(&self) -> Option<&str> {
        self.language
            .as_deref()
            .filter(|language| *language != "auto")
    }
}

/// A validated transcription option error.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TranscriptionOptionsError {
    /// The requested language is malformed or unknown to Whisper.
    #[error("invalid or unsupported Whisper language `{language}`")]
    InvalidLanguage {
        /// The rejected language value.
        language: String,
    },
    /// The requested decoder thread count is outside the supported bounds.
    #[error("thread count {provided} is invalid; expected 1..={maximum}")]
    InvalidThreadCount {
        /// The rejected thread count.
        provided: usize,
        /// The maximum accepted thread count.
        maximum: usize,
    },
}

/// Cooperative cancellation observed by [`transcribe_pcm`].
///
/// Implementations must be fast and should not block. A panic is caught and
/// treated as a cancellation request before control can cross the C callback
/// boundary.
pub trait CancellationCheck: Send + Sync + 'static {
    /// Returns `true` when transcription should stop.
    fn is_cancelled(&self) -> bool;
}

impl<F> CancellationCheck for F
where
    F: Fn() -> bool + Send + Sync + 'static,
{
    fn is_cancelled(&self) -> bool {
        self()
    }
}

/// A cancellation check that never requests cancellation.
#[derive(Debug, Default, Clone, Copy)]
pub struct NeverCancel;

impl CancellationCheck for NeverCancel {
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// An error produced before, during, or after Whisper inference.
#[derive(Debug, Error)]
pub enum TranscriptionError {
    /// Transcription options failed validation.
    #[error(transparent)]
    InvalidOptions(#[from] TranscriptionOptionsError),
    /// Whisper cannot safely process an empty PCM buffer.
    #[error("PCM input is empty; expected 16 kHz mono samples")]
    EmptyPcm,
    /// The PCM slice cannot be represented by whisper.cpp's signed length.
    #[error("PCM input has {samples} samples; maximum supported length is {maximum}")]
    PcmTooLong {
        /// Rejected sample count.
        samples: usize,
        /// Maximum sample count accepted by whisper.cpp.
        maximum: usize,
    },
    /// One PCM sample was NaN or infinite.
    #[error("PCM sample at index {index} is not finite")]
    NonFinitePcm {
        /// Index of the rejected sample.
        index: usize,
    },
    /// The caller requested cancellation.
    #[error("transcription was cancelled")]
    Cancelled,
    /// `whisper.cpp` or its safe Rust wrapper returned an error.
    #[error("Whisper transcription failed: {0}")]
    Whisper(#[from] whisper_rs::WhisperError),
}

/// Transcribes 16 kHz mono floating-point PCM with a local Whisper model.
///
/// `pcm` must already be mono at [`crate::WHISPER_SAMPLE_RATE`] and contain only
/// finite samples. Options, PCM, and cancellation are checked before the model
/// is loaded.
///
/// Cancellation is checked immediately before model loading, immediately
/// after model loading, from whisper-rs's safe abort callback during inference,
/// and after inference. The current safe whisper-rs API cannot interrupt the
/// model-loading step itself, so a cancellation arriving during model loading
/// is returned as soon as that step completes.
///
/// When `options.token_timestamps` is enabled, known official model filenames
/// select whisper.cpp's matching DTW preset. Unknown filenames still use
/// Whisper's regular token timestamp path.
///
/// # Errors
///
/// Returns [`TranscriptionError`] for invalid input, cancellation, model-load
/// failures, inference failures, or malformed data returned by Whisper.
pub fn transcribe_pcm<C>(
    model_path: impl AsRef<Path>,
    pcm: &[f32],
    options: &TranscriptionOptions,
    cancellation: Arc<C>,
) -> Result<Transcript, TranscriptionError>
where
    C: CancellationCheck + ?Sized,
{
    options.validate()?;
    if pcm.is_empty() {
        return Err(TranscriptionError::EmptyPcm);
    }
    if pcm.len() > MAX_PCM_SAMPLES {
        return Err(TranscriptionError::PcmTooLong {
            samples: pcm.len(),
            maximum: MAX_PCM_SAMPLES,
        });
    }
    if let Some(index) = pcm.iter().position(|sample| !sample.is_finite()) {
        return Err(TranscriptionError::NonFinitePcm { index });
    }
    if cancellation_requested(cancellation.as_ref()) {
        return Err(TranscriptionError::Cancelled);
    }

    let model_path = model_path.as_ref();
    let mut context_parameters = WhisperContextParameters::default();
    if options.token_timestamps
        && let Some(model_preset) = dtw_preset_from_model_path(model_path)
    {
        context_parameters.dtw_parameters.mode = DtwMode::ModelPreset { model_preset };
    }

    let context = WhisperContext::new_with_params(model_path, context_parameters)?;
    if cancellation_requested(cancellation.as_ref()) {
        return Err(TranscriptionError::Cancelled);
    }

    let mut parameters = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    parameters.set_language(options.whisper_language());
    parameters.set_translate(options.translate);
    parameters.set_n_threads(options.thread_count as i32);
    parameters.set_print_special(false);
    parameters.set_print_progress(false);
    parameters.set_print_realtime(false);
    parameters.set_print_timestamps(false);
    parameters.set_token_timestamps(options.token_timestamps);
    parameters.set_split_on_word(options.token_timestamps);

    let cancellation_observed = Arc::new(AtomicBool::new(false));
    let callback_observed = Arc::clone(&cancellation_observed);
    let callback_cancellation = Arc::clone(&cancellation);
    let abort_callback: Box<dyn FnMut() -> bool> = Box::new(move || {
        let requested = cancellation_requested(callback_cancellation.as_ref());
        if requested {
            callback_observed.store(true, Ordering::Relaxed);
        }
        requested
    });
    parameters.set_abort_callback_safe::<_, Box<dyn FnMut() -> bool>>(Some(abort_callback));

    let mut state = context.create_state()?;
    if cancellation_requested(cancellation.as_ref()) {
        return Err(TranscriptionError::Cancelled);
    }

    let inference_result = state.full(parameters, pcm);
    if cancellation_observed.load(Ordering::Relaxed)
        || cancellation_requested(cancellation.as_ref())
    {
        return Err(TranscriptionError::Cancelled);
    }
    inference_result?;

    let mut raw_segments = Vec::new();
    for segment in state.as_iter() {
        let tokens = if options.token_timestamps {
            collect_segment_tokens(&segment)?
        } else {
            Vec::new()
        };
        raw_segments.push(RawSegment {
            start_cs: segment.start_timestamp(),
            end_cs: segment.end_timestamp(),
            text: segment.to_str_lossy()?.into_owned(),
            tokens,
        });
    }

    Ok(normalize_transcript(raw_segments))
}

fn cancellation_requested<C>(cancellation: &C) -> bool
where
    C: CancellationCheck + ?Sized,
{
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cancellation.is_cancelled()))
        .unwrap_or(true)
}

fn collect_segment_tokens(
    segment: &whisper_rs::WhisperSegment<'_>,
) -> Result<Vec<RawToken>, whisper_rs::WhisperError> {
    let mut tokens = Vec::with_capacity(segment.n_tokens().max(0) as usize);
    for token_index in 0..segment.n_tokens() {
        let Some(token) = segment.get_token(token_index) else {
            continue;
        };
        tokens.push(RawToken {
            text: token.to_str_lossy()?.into_owned(),
            timestamps_cs: token_timestamps_cs(&token.token_data()),
            probability: token.token_probability(),
        });
    }
    Ok(tokens)
}

fn token_timestamps_cs(data: &WhisperTokenData) -> Option<(i64, i64)> {
    if data.t0 >= 0 {
        Some((data.t0, data.t1.max(data.t0)))
    } else if data.t_dtw >= 0 {
        Some((data.t_dtw, data.t_dtw))
    } else {
        None
    }
}

fn dtw_preset_from_model_path(path: &Path) -> Option<DtwModelPreset> {
    let name = path.file_name()?.to_string_lossy().to_ascii_lowercase();

    if contains_any(&name, &["large-v3-turbo", "large_v3_turbo"]) {
        Some(DtwModelPreset::LargeV3Turbo)
    } else if contains_any(&name, &["large-v3", "large_v3"]) {
        Some(DtwModelPreset::LargeV3)
    } else if contains_any(&name, &["large-v2", "large_v2"]) {
        Some(DtwModelPreset::LargeV2)
    } else if contains_any(&name, &["large-v1", "large_v1"])
        || (name.contains("large")
            && !contains_any(&name, &["large-v2", "large_v2", "large-v3", "large_v3"]))
    {
        Some(DtwModelPreset::LargeV1)
    } else if contains_any(&name, &["medium.en", "medium-en", "medium_en"]) {
        Some(DtwModelPreset::MediumEn)
    } else if name.contains("medium") {
        Some(DtwModelPreset::Medium)
    } else if contains_any(&name, &["small.en", "small-en", "small_en"]) {
        Some(DtwModelPreset::SmallEn)
    } else if name.contains("small") {
        Some(DtwModelPreset::Small)
    } else if contains_any(&name, &["base.en", "base-en", "base_en"]) {
        Some(DtwModelPreset::BaseEn)
    } else if name.contains("base") {
        Some(DtwModelPreset::Base)
    } else if contains_any(&name, &["tiny.en", "tiny-en", "tiny_en"]) {
        Some(DtwModelPreset::TinyEn)
    } else if name.contains("tiny") {
        Some(DtwModelPreset::Tiny)
    } else {
        None
    }
}

fn contains_any(name: &str, candidates: &[&str]) -> bool {
    candidates.iter().any(|candidate| name.contains(candidate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid_and_timestamped() {
        let options = TranscriptionOptions::default();
        assert!(options.validate().is_ok());
        assert!(options.token_timestamps);
        assert!((1..=8).contains(&options.thread_count));
    }

    #[test]
    fn options_are_validated_before_model_loading() {
        let invalid_language = TranscriptionOptions {
            language: Some("not-a-whisper-language".into()),
            ..TranscriptionOptions::default()
        };
        let error = transcribe_pcm(
            "/definitely/missing/model.bin",
            &[0.0],
            &invalid_language,
            Arc::new(NeverCancel),
        )
        .expect_err("invalid options must fail before loading a model");
        assert!(matches!(
            error,
            TranscriptionError::InvalidOptions(TranscriptionOptionsError::InvalidLanguage { .. })
        ));

        let invalid_threads = TranscriptionOptions {
            thread_count: 0,
            ..TranscriptionOptions::default()
        };
        let error = transcribe_pcm(
            "/definitely/missing/model.bin",
            &[0.0],
            &invalid_threads,
            Arc::new(NeverCancel),
        )
        .expect_err("invalid options must fail before loading a model");
        assert!(matches!(
            error,
            TranscriptionError::InvalidOptions(
                TranscriptionOptionsError::InvalidThreadCount { .. }
            )
        ));
    }

    #[test]
    fn non_finite_pcm_is_rejected_before_model_loading() {
        let error = transcribe_pcm(
            "/definitely/missing/model.bin",
            &[0.0, f32::NAN],
            &TranscriptionOptions::default(),
            Arc::new(NeverCancel),
        )
        .expect_err("non-finite PCM must fail before loading a model");
        assert!(matches!(
            error,
            TranscriptionError::NonFinitePcm { index: 1 }
        ));
    }

    #[test]
    fn cancellation_is_checked_before_model_loading() {
        let error = transcribe_pcm(
            "/definitely/missing/model.bin",
            &[0.0],
            &TranscriptionOptions::default(),
            Arc::new(|| true),
        )
        .expect_err("cancellation must win before the missing model is opened");
        assert!(matches!(error, TranscriptionError::Cancelled));
    }

    #[test]
    fn language_validation_rejects_nul_without_panicking() {
        let options = TranscriptionOptions {
            language: Some("en\0ignored".into()),
            ..TranscriptionOptions::default()
        };
        assert!(matches!(
            options.validate(),
            Err(TranscriptionOptionsError::InvalidLanguage { .. })
        ));
    }

    #[test]
    fn selects_dtw_presets_from_official_and_quantized_names() {
        assert!(matches!(
            dtw_preset_from_model_path(Path::new("ggml-base.en.bin")),
            Some(DtwModelPreset::BaseEn)
        ));
        assert!(matches!(
            dtw_preset_from_model_path(Path::new("ggml-small-q5_1.bin")),
            Some(DtwModelPreset::Small)
        ));
        assert!(matches!(
            dtw_preset_from_model_path(Path::new("ggml-large-v3-turbo-q5_0.bin")),
            Some(DtwModelPreset::LargeV3Turbo)
        ));
        assert!(dtw_preset_from_model_path(Path::new("custom.bin")).is_none());
    }
}

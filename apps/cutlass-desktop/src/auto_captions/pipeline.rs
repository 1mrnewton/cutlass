// --- Speech recognition -------------------------------------------------------------

use std::error::Error;
use std::fmt;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cutlass_decoder::{ReadMonoPcmError, read_mono_pcm_with_cancel};
use cutlass_jobs::JobContext;
use cutlass_storage::CacheId;
use cutlass_transcription::{
    DownloadError, DownloadReader, HttpDownloader, ModelDownloader, ModelManager, ModelSpec,
    Transcript, TranscriptionError, TranscriptionOptions, WHISPER_SAMPLE_RATE, WhisperModel,
    transcribe_pcm_observed,
};
use tracing::warn;

use crate::CaptionBackend;
use crate::analysis_index::{
    AnalysisIndexError, AnalysisIndexService, validate_media_content_with_cancel,
};
use crate::cache_registry::{CacheRegistry, CoordinatedCacheError};

use super::progress::{Progress, Stage, report, with_pump};
use super::{AutoCaptionRequest, MAX_AUDIO_MINUTES, transcript_cache};

/// One recognized word on the asset's own clock.
#[derive(Debug, Clone)]
pub(super) struct RecognizedWord {
    pub(super) start_ms: u32,
    pub(super) end_ms: u32,
    pub(super) text: String,
    pub(super) confidence: Option<f32>,
}

/// Everything recognition produces for the caption placer.
#[derive(Debug, Clone)]
pub(super) struct Recognized {
    pub(super) words: Vec<RecognizedWord>,
    pub(super) language: Option<String>,
    /// Whether the words came from the moments cache instead of inference.
    pub(super) from_cache: bool,
}

/// Why recognition stopped, phrased for the dialog to show verbatim.
#[derive(Debug)]
pub(super) enum RecognizeError {
    Cancelled,
    /// The media file could not be read or changed while being identified.
    Media(String),
    /// The speech model could not be installed or verified.
    Model(String),
    /// The clip's audio could not be decoded.
    Decode(String),
    /// The clip is longer than one transcription pass accepts.
    TooLong,
    /// Inference itself failed.
    Transcribe(String),
}

impl fmt::Display for RecognizeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("Auto captions cancelled"),
            Self::Media(detail) => write!(formatter, "Could not read the clip's file: {detail}"),
            Self::Model(detail) => {
                write!(formatter, "Could not install the speech model: {detail}")
            }
            Self::Decode(detail) => {
                write!(formatter, "Could not decode the clip's audio: {detail}")
            }
            Self::TooLong => write!(
                formatter,
                "This clip's audio is longer than {MAX_AUDIO_MINUTES} minutes — caption it in shorter pieces"
            ),
            Self::Transcribe(detail) => write!(formatter, "Transcription failed: {detail}"),
        }
    }
}

impl Error for RecognizeError {}

impl From<AnalysisIndexError> for RecognizeError {
    fn from(error: AnalysisIndexError) -> Self {
        match error {
            AnalysisIndexError::Cancelled => Self::Cancelled,
            other => Self::Media(other.to_string()),
        }
    }
}

/// Turn a clip's audio into recognized words.
///
/// The file is transcribed whole, not just the clip's window: the transcript is
/// keyed by content hash and cached, so captioning a second clip of the same
/// source — or the same clip after an undo — costs a hash rather than another
/// inference run. Trimming to the clip happens after recognition.
pub(super) fn recognize(
    context: &JobContext,
    caches: &CacheRegistry,
    backend: &slint::Weak<CaptionBackend<'static>>,
    request: &AutoCaptionRequest,
) -> Result<Recognized, RecognizeError> {
    let cancelled = || context.cancelled();
    report(context, backend, Stage::Preparing, 0.0, None);
    let validated = validate_media_content_with_cancel(&request.path, &cancelled)?;
    let content_key = validated.content_key();
    let index = AnalysisIndexService::new(caches.clone());

    if let Some(words) = transcript_cache::load(&index, content_key, &cancelled)? {
        report(context, backend, Stage::Cached, 1.0, None);
        return Ok(Recognized {
            words,
            language: request.language.clone(),
            from_cache: true,
        });
    }

    let model = install_model(context, caches, backend)?;
    let pcm = decode_audio(context, backend, validated.path())?;
    let transcript = transcribe(context, backend, &model, &pcm, request.language.clone())?;
    drop(pcm);

    // A failed cache write costs the next run an inference pass but nothing
    // else, so it must not lose captions the user already waited for.
    if let Err(error) = transcript_cache::store(&index, content_key, &transcript, &cancelled) {
        warn!("auto captions could not cache the transcript: {error}");
    }

    Ok(Recognized {
        words: words_of(&transcript),
        language: request.language.clone(),
        from_cache: false,
    })
}

fn words_of(transcript: &Transcript) -> Vec<RecognizedWord> {
    transcript
        .segments()
        .iter()
        .flat_map(|segment| segment.words())
        .filter(|word| !word.text().trim().is_empty())
        .map(|word| RecognizedWord {
            start_ms: centiseconds_to_ms(word.start_centiseconds()),
            end_ms: centiseconds_to_ms(word.end_centiseconds()),
            text: word.text().to_owned(),
            confidence: Some(word.probability()),
        })
        .collect()
}

pub(super) fn centiseconds_to_ms(centiseconds: u64) -> u32 {
    u32::try_from(centiseconds.saturating_mul(10)).unwrap_or(u32::MAX)
}

/// Install and verify `ggml-base.en`, reporting download bytes as they arrive.
///
/// The AI-model cache gate is held for the whole install, which is what stops a
/// relocation from moving the directory out from under the download.
fn install_model(
    context: &JobContext,
    caches: &CacheRegistry,
    backend: &slint::Weak<CaptionBackend<'static>>,
) -> Result<PathBuf, RecognizeError> {
    let cancelled = || context.cancelled();
    let installed = with_pump(context, backend, Stage::Model, |progress| {
        let downloader = ObservedDownloader {
            inner: HttpDownloader::default(),
            progress: Arc::clone(progress),
        };
        caches.with_disk_cache_root(CacheId::AiModels, &cancelled, |root| {
            let manager = ModelManager::new(root).map_err(|error| error.to_string())?;
            // Deliberately not bound to the job's commit handshake: that would
            // make the whole job uncancellable from here on, and the user must
            // still be able to abandon a multi-minute inference run. A model
            // file that lands during a cancel is verified and simply reused.
            manager
                .ensure_with_cancellation(WhisperModel::BaseEn, &downloader, &cancelled)
                .map_err(|error| error.to_string())
        })
    });

    match installed {
        Ok(path) => Ok(path),
        Err(CoordinatedCacheError::Callback(detail)) => {
            if context.cancelled() {
                Err(RecognizeError::Cancelled)
            } else {
                Err(RecognizeError::Model(detail))
            }
        }
        Err(CoordinatedCacheError::Coordination(error)) => {
            if context.cancelled() {
                Err(RecognizeError::Cancelled)
            } else {
                Err(RecognizeError::Model(error.to_string()))
            }
        }
    }
}

/// Decode the whole file to the 16 kHz mono PCM whisper.cpp requires.
fn decode_audio(
    context: &JobContext,
    backend: &slint::Weak<CaptionBackend<'static>>,
    path: &Path,
) -> Result<Vec<f32>, RecognizeError> {
    report(context, backend, Stage::Decoding, 0.0, None);
    let max_frames = MAX_AUDIO_MINUTES as usize * 60 * WHISPER_SAMPLE_RATE as usize;
    read_mono_pcm_with_cancel(path, WHISPER_SAMPLE_RATE, max_frames, || {
        context.cancelled()
    })
    .map_err(|error| match error {
        ReadMonoPcmError::Cancelled => RecognizeError::Cancelled,
        ReadMonoPcmError::LimitExceeded { .. } => RecognizeError::TooLong,
        other => RecognizeError::Decode(other.to_string()),
    })
}

fn transcribe(
    context: &JobContext,
    backend: &slint::Weak<CaptionBackend<'static>>,
    model: &Path,
    pcm: &[f32],
    language: Option<String>,
) -> Result<Transcript, RecognizeError> {
    let options = TranscriptionOptions {
        language,
        token_timestamps: true,
        ..TranscriptionOptions::default()
    };
    let token = context.cancellation_token();
    with_pump(context, backend, Stage::Transcribing, |progress| {
        let observed = Arc::clone(progress);
        transcribe_pcm_observed(
            model,
            pcm,
            &options,
            Arc::new(move || token.cancelled()),
            Arc::new(move |percent: u8| observed.set_percent(percent)),
        )
    })
    .map_err(|error| match error {
        TranscriptionError::Cancelled => RecognizeError::Cancelled,
        other => RecognizeError::Transcribe(other.to_string()),
    })
}

/// A downloader that reports the bytes it streams.
///
/// The model manager owns integrity checking and does not report progress, so
/// the count is taken where the bytes actually cross the wire.
struct ObservedDownloader {
    inner: HttpDownloader,
    progress: Arc<Progress>,
}

impl ModelDownloader for ObservedDownloader {
    fn download(&self, spec: &ModelSpec) -> Result<DownloadReader, DownloadError> {
        let reader = self.inner.download(spec)?;
        Ok(Box::new(CountingReader {
            inner: reader,
            read: 0,
            total: spec.exact_bytes(),
            progress: Arc::clone(&self.progress),
        }))
    }
}

struct CountingReader {
    inner: DownloadReader,
    read: u64,
    total: u64,
    progress: Arc<Progress>,
}

impl Read for CountingReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let count = self.inner.read(buffer)?;
        self.read = self.read.saturating_add(count as u64);
        self.progress.set_bytes(self.read, self.total);
        Ok(count)
    }
}

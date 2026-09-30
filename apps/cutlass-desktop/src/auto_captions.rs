//! Auto captions: speech in a clip becomes a caption group.
//!
//! One [`JobManager`] job does the slow work — install the Whisper model, decode
//! the clip's audio to 16 kHz mono, transcribe it, cache the transcript — and
//! hands the recognized words to `cutlass-captions` for segmentation. The cue
//! specs then travel to the preview worker, which applies the single
//! `AddCaptionGroup` edit that puts the captions on the timeline.
//!
//! Nothing here touches the engine or the UI directly: the job thread owns none
//! of that state, so results cross back through the worker queue and the Slint
//! event loop the same way an export's progress does.

mod pipeline;
mod progress;
#[cfg(test)]
mod tests;
mod transcript_cache;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use cutlass_captions::{Placement, SegmentOptions, TimedWord, segment};
use cutlass_jobs::{JobCompletion, JobContext, JobId, JobManager};
use cutlass_models::{CaptionCueSpec, CaptionLayout, Rational};
use cutlass_transcription::WhisperModel;
use tracing::{info, warn};

use crate::CaptionBackend;
use crate::cache_registry::CacheRegistry;
use crate::preview_worker::{TranscribedCaptions, WorkerHandle};

use pipeline::Recognized;
use progress::{Stage, publish, report};

/// Longest audio one job will transcribe. Whisper holds the decoded PCM in
/// memory (an hour of 16 kHz mono is ~230 MB) and inference is roughly linear,
/// so a feature-length source is captioned clip by clip rather than in one go.
const MAX_AUDIO_MINUTES: u32 = 60;

/// Everything one auto-caption job needs, snapshotted on the UI thread from the
/// projection so the job never reads engine or Slint state.
#[derive(Debug, Clone)]
pub(crate) struct AutoCaptionRequest {
    /// Media file to transcribe.
    pub(crate) path: PathBuf,
    /// Pool id of that file, recorded as the group's provenance.
    pub(crate) media: String,
    /// Asset name, for the job label and the caption group's label.
    pub(crate) name: String,
    /// Sequence tick the clip starts at — where caption time zero lands.
    pub(crate) start_tick: i64,
    /// The clip's length on the timeline. Speech past its out-point belongs to
    /// footage that isn't in the edit, so those words are dropped.
    pub(crate) duration_ticks: i64,
    /// Seconds into the file the clip starts playing (its in-point).
    pub(crate) source_in_seconds: f64,
    /// Playback rate: a 2× clip plays its audio twice as fast, so recognized
    /// asset time compresses into half as much timeline time.
    pub(crate) speed: f64,
    pub(crate) rate: Rational,
    /// Whisper language code, or `None` to detect it.
    pub(crate) language: Option<String>,
    /// Caption template the new group wears.
    pub(crate) template: String,
    /// Segmentation rules — the dialog's characters-per-line lands here.
    pub(crate) layout: CaptionLayout,
}

impl AutoCaptionRequest {
    /// Seconds of the source this clip actually plays.
    fn window_seconds(&self) -> (f64, f64) {
        let start = self.source_in_seconds.max(0.0);
        let played = self.timeline_seconds() * self.speed.max(f64::MIN_POSITIVE);
        (start, start + played.max(0.0))
    }

    /// The clip's length in seconds of timeline time.
    fn timeline_seconds(&self) -> f64 {
        let rate = &self.rate;
        if rate.num <= 0 || rate.den <= 0 {
            return 0.0;
        }
        self.duration_ticks.max(0) as f64 * f64::from(rate.den) / f64::from(rate.num)
    }
}

/// Starts and cancels auto-caption jobs, and keeps the dialog's view of the
/// running one up to date.
///
/// Cloneable: the UI callbacks each hold one, all sharing the single job slot so
/// a second request can be refused rather than queued behind a job the user
/// can't see.
#[derive(Clone)]
pub(crate) struct AutoCaptionService {
    jobs: JobManager,
    caches: CacheRegistry,
    worker: WorkerHandle,
    backend: slint::Weak<CaptionBackend<'static>>,
    running: Arc<Mutex<Option<JobId>>>,
}

impl AutoCaptionService {
    pub(crate) fn new(
        jobs: JobManager,
        caches: CacheRegistry,
        worker: WorkerHandle,
        backend: slint::Weak<CaptionBackend<'static>>,
    ) -> Self {
        Self {
            jobs,
            caches,
            worker,
            backend,
            running: Arc::new(Mutex::new(None)),
        }
    }

    /// Transcribe one clip and place the captions it produces.
    ///
    /// Returns without starting anything when a job is already running: model
    /// installs and inference are both heavy, and two of them would compete for
    /// the same CPU while racing to add groups to the same lane.
    pub(crate) fn start(&self, request: AutoCaptionRequest) {
        let mut slot = self.running.lock().expect("auto-caption job slot poisoned");
        if let Some(id) = *slot
            && self
                .jobs
                .get(id)
                .is_some_and(|snapshot| !snapshot.state.is_terminal())
        {
            warn!(%id, "auto captions refused: a transcription is already running");
            return;
        }

        let label = format!("Transcribing {}", request.name);
        publish(&self.backend, progress::running(Stage::Preparing, 0.0));
        let service = self.clone();
        *slot = Some(
            self.jobs
                .spawn_with_completion(label, move |context| service.run(context, &request)),
        );
    }

    /// Ask the running job to stop. Cancellation is cooperative: the job exits
    /// at its next checkpoint, and a model install that already began its
    /// atomic commit finishes that step first.
    pub(crate) fn cancel(&self) {
        let Some(id) = *self.running.lock().expect("auto-caption job slot poisoned") else {
            return;
        };
        if self.jobs.cancel(id) {
            info!(%id, "auto captions cancellation requested");
            publish(
                &self.backend,
                progress::running(Stage::Cancelling, 0.0).indeterminate(),
            );
        }
    }

    /// The job body: recognize, segment, and hand the cues to the worker.
    fn run(
        &self,
        context: &JobContext,
        request: &AutoCaptionRequest,
    ) -> Result<JobCompletion, String> {
        let recognized = match pipeline::recognize(context, &self.caches, &self.backend, request) {
            Ok(recognized) => recognized,
            Err(error) => {
                let message = error.to_string();
                // Cancellation is the user's own doing, so it reports as a
                // dismissable outcome rather than a failure.
                publish(&self.backend, progress::outcome(&error));
                return Err(message);
            }
        };

        report(context, &self.backend, Stage::Placing, 0.0, None);
        let cues = match place(request, &recognized) {
            Ok(cues) => cues,
            Err(message) => {
                publish(&self.backend, progress::failed(&message));
                return Err(message);
            }
        };
        if cues.is_empty() {
            let message = format!("No speech found in {}", request.name);
            publish(&self.backend, progress::failed(&message));
            return Err(message);
        }

        // The worker owns the engine, so the placement round-trips: the dialog
        // reports what the edit actually did, and a rejected group can't be
        // announced as a success.
        let placed = self
            .worker
            .add_transcribed_captions(Box::new(TranscribedCaptions {
                cues,
                label: format!("{} captions", request.name),
                template: request.template.clone(),
                layout: request.layout,
                media: request.media.clone(),
                language: recognized.language.clone(),
                model: WhisperModel::BaseEn.spec().id().to_owned(),
            }));
        let count = match placed {
            Some(Ok(count)) => count,
            Some(Err(message)) => {
                publish(&self.backend, progress::failed(&message));
                return Err(message);
            }
            None => {
                let message = "The editor stopped before the captions were added".to_owned();
                publish(&self.backend, progress::failed(&message));
                return Err(message);
            }
        };
        info!(
            cues = count,
            words = recognized.words.len(),
            cached = recognized.from_cache,
            "auto captions placed"
        );
        publish(
            &self.backend,
            progress::completed(&format!(
                "Added {count} caption{} from {}",
                if count == 1 { "" } else { "s" },
                request.name
            )),
        );

        JobCompletion::new(format!("{count} caption cues"))
            .with_output("cues", count.to_string())
            .and_then(|completion| {
                completion.with_output("words", recognized.words.len().to_string())
            })
            .and_then(|completion| {
                completion.with_output(
                    "cached",
                    if recognized.from_cache {
                        "true"
                    } else {
                        "false"
                    },
                )
            })
            .map_err(|error| error.to_string())
    }
}

/// Recognized words → cue specs on the timeline.
///
/// Words arrive on the asset's clock; the clip's in-point and speed move them
/// onto the timeline, and anything outside the clip's window is discarded before
/// segmentation so a cue can never spill past the footage it transcribes.
fn place(
    request: &AutoCaptionRequest,
    recognized: &Recognized,
) -> Result<Vec<CaptionCueSpec>, String> {
    let (window_start, window_end) = request.window_seconds();
    let speed = request.speed.max(f64::MIN_POSITIVE);
    let limit_ms = (request.timeline_seconds() * 1000.0).round().max(0.0);

    let words: Vec<TimedWord> = recognized
        .words
        .iter()
        .filter(|word| {
            let start = f64::from(word.start_ms) / 1000.0;
            start < window_end && f64::from(word.end_ms) / 1000.0 > window_start
        })
        .map(|word| {
            let local = |ms: u32| {
                let seconds = (f64::from(ms) / 1000.0 - window_start).max(0.0) / speed;
                (seconds * 1000.0).round().clamp(0.0, limit_ms) as u32
            };
            TimedWord {
                start_ms: local(word.start_ms),
                end_ms: local(word.end_ms),
                text: word.text.clone(),
                confidence: word.confidence,
            }
        })
        .filter(|word| word.start_ms < limit_ms as u32)
        .collect();

    let placement = Placement::new(request.rate, request.start_tick.max(0));
    let options = SegmentOptions::new(placement).with_layout(request.layout);
    segment(&words, &options).map_err(|error| error.to_string())
}

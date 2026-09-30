// --- Auto-caption progress ------------------------------------------------------------

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use cutlass_jobs::JobContext;
use tracing::error;

use crate::CaptionBackend;

use super::pipeline::RecognizeError;

/// How often a blocking step's progress reaches the dialog — the same ~10 Hz
/// the export job publishes at, which is smooth to watch and cheap to deliver.
const TICK: Duration = Duration::from_millis(100);

/// One phase of a transcription and the slice of the bar it fills.
///
/// The weights are rough wall-clock shares: on a first run the 148 MB model
/// download dominates, and after that inference does. A skipped phase simply
/// jumps the bar forward, which reads as progress rather than a stall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Stage {
    Preparing,
    Cached,
    Model,
    Decoding,
    Transcribing,
    Placing,
    Cancelling,
}

impl Stage {
    const fn span(self) -> (f32, f32) {
        match self {
            Self::Preparing => (0.0, 0.05),
            Self::Model => (0.05, 0.35),
            Self::Decoding => (0.35, 0.45),
            Self::Transcribing | Self::Cached => (0.45, 0.97),
            Self::Placing => (0.97, 1.0),
            Self::Cancelling => (0.0, 0.0),
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Preparing => "Checking the clip's audio…",
            Self::Cached => "Reading the cached transcript…",
            Self::Model => "Installing the speech model…",
            Self::Decoding => "Decoding audio…",
            Self::Transcribing => "Transcribing speech…",
            Self::Placing => "Writing captions…",
            Self::Cancelling => "Cancelling…",
        }
    }

    /// Phases that cannot say how far along they are, so the dialog animates
    /// instead of showing a bar that never moves.
    const fn is_indeterminate(self) -> bool {
        matches!(
            self,
            Self::Preparing | Self::Cached | Self::Decoding | Self::Placing | Self::Cancelling
        )
    }
}

/// One snapshot of the auto-caption job for the Slint `CaptionBackend` global.
#[derive(Debug, Clone, Default)]
pub(super) struct AutoCaptionUi {
    running: bool,
    progress: f32,
    stage: String,
    indeterminate: bool,
    completed: bool,
    failed: bool,
    status: String,
}

impl AutoCaptionUi {
    pub(super) fn indeterminate(mut self) -> Self {
        self.indeterminate = true;
        self
    }
}

/// A running job at `fraction` of `stage`.
pub(super) fn running(stage: Stage, fraction: f32) -> AutoCaptionUi {
    let (start, end) = stage.span();
    AutoCaptionUi {
        running: true,
        progress: (start + (end - start) * fraction.clamp(0.0, 1.0)).clamp(0.0, 1.0),
        stage: stage.label().to_owned(),
        indeterminate: stage.is_indeterminate(),
        ..Default::default()
    }
}

pub(super) fn completed(status: &str) -> AutoCaptionUi {
    AutoCaptionUi {
        progress: 1.0,
        completed: true,
        status: status.to_owned(),
        ..Default::default()
    }
}

pub(super) fn failed(status: &str) -> AutoCaptionUi {
    AutoCaptionUi {
        failed: true,
        status: status.to_owned(),
        ..Default::default()
    }
}

/// The terminal state for a recognition failure. A cancellation is the user's
/// own doing, so it reads as a plain outcome rather than an error.
pub(super) fn outcome(error: &RecognizeError) -> AutoCaptionUi {
    if matches!(error, RecognizeError::Cancelled) {
        failed("Auto captions cancelled")
    } else {
        failed(&error.to_string())
    }
}

pub(super) fn publish(backend: &slint::Weak<CaptionBackend<'static>>, state: AutoCaptionUi) {
    let backend = backend.clone();
    if let Err(e) = slint::invoke_from_event_loop(move || {
        if let Some(backend) = backend.upgrade() {
            backend.set_auto_running(state.running);
            backend.set_auto_progress(state.progress);
            backend.set_auto_stage(state.stage.into());
            backend.set_auto_indeterminate(state.indeterminate);
            backend.set_auto_completed(state.completed);
            backend.set_auto_failed(state.failed);
            backend.set_auto_status(state.status.into());
        }
    }) {
        error!("failed to publish auto-caption progress to the UI: {e}");
    }
}

/// A step's own progress, written by the step and read by the pump.
///
/// Counters rather than a callback because the writers are `'static` C-facing
/// callbacks: they can hold an `Arc<Progress>` but not a borrow of the job
/// context that [`JobContext::set_progress`] needs.
#[derive(Debug, Default)]
pub(super) struct Progress {
    per_mille: AtomicU32,
    done_bytes: AtomicU64,
    total_bytes: AtomicU64,
}

impl Progress {
    /// Report `done` of `total` bytes — a download, whose detail line shows the
    /// sizes so a slow connection still looks alive.
    pub(super) fn set_bytes(&self, done: u64, total: u64) {
        self.done_bytes.store(done, Ordering::Relaxed);
        self.total_bytes.store(total, Ordering::Relaxed);
        if let Some(per_mille) = (done.min(total) * 1000).checked_div(total) {
            self.per_mille.store(per_mille as u32, Ordering::Relaxed);
        }
    }

    /// Report whole-step completion in percent (whisper.cpp's own unit).
    pub(super) fn set_percent(&self, percent: u8) {
        self.per_mille
            .store(u32::from(percent.min(100)) * 10, Ordering::Relaxed);
    }

    fn fraction(&self) -> f32 {
        self.per_mille.load(Ordering::Relaxed) as f32 / 1000.0
    }

    fn detail(&self) -> Option<String> {
        let total = self.total_bytes.load(Ordering::Relaxed);
        if total == 0 {
            return None;
        }
        let done = self.done_bytes.load(Ordering::Relaxed).min(total);
        Some(format!(
            "{} MB of {} MB",
            done / 1_000_000,
            total / 1_000_000
        ))
    }
}

/// Run a blocking step, publishing the progress it reports while it runs.
///
/// The step gets an [`Arc<Progress>`] it can hand to a `'static` callback, and a
/// scoped thread — which *can* borrow the job context — turns those counters
/// into job-registry and dialog updates. The pump is woken the moment the step
/// returns, so it adds no latency to the phase.
pub(super) fn with_pump<T>(
    context: &JobContext,
    backend: &slint::Weak<CaptionBackend<'static>>,
    stage: Stage,
    step: impl FnOnce(&Arc<Progress>) -> T,
) -> T {
    let progress = Arc::new(Progress::default());
    let finished = Mutex::new(false);
    let wake = Condvar::new();

    std::thread::scope(|scope| {
        let pumped = Arc::clone(&progress);
        let gate = &finished;
        let signal = &wake;
        scope.spawn(move || {
            let mut done = gate.lock().expect("auto-caption pump lock");
            while !*done {
                let (guard, _) = signal
                    .wait_timeout(done, TICK)
                    .expect("auto-caption pump lock");
                done = guard;
                if !*done {
                    report(context, backend, stage, pumped.fraction(), pumped.detail());
                }
            }
        });

        let value = step(&progress);
        *finished.lock().expect("auto-caption pump lock") = true;
        wake.notify_all();
        value
    })
}

/// Publish one phase's position to both the job registry and the dialog.
pub(super) fn report(
    context: &JobContext,
    backend: &slint::Weak<CaptionBackend<'static>>,
    stage: Stage,
    fraction: f32,
    detail: Option<String>,
) {
    let state = running(stage, fraction);
    let line = match &detail {
        Some(detail) => format!("{} {detail}", stage.label()),
        None => stage.label().to_owned(),
    };
    context.set_progress(state.progress, line);
    publish(
        backend,
        AutoCaptionUi {
            stage: match detail {
                Some(detail) => format!("{} · {detail}", stage.label()),
                None => stage.label().to_owned(),
            },
            ..state
        },
    );
}

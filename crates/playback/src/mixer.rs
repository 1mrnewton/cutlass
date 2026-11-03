//! Mix preview audio on its own thread.
//!
//! The mixer keeps the live stream [`LEAD`] frames ahead of the device in
//! 20 ms windows, so the window's thread never mixes. It owns its own file
//! readers and a copy of the draft it plays. A new play or a stop clears the
//! stream and starts a new generation under the same lock the mixer appends
//! under, so a window mixed for an old position never reaches the device.

use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use cutlass_eval::{Heard, ReaderFactory};
use cutlass_model::{Draft, Micros};

use crate::apple::{AudioOutput, AudioSink, RATE};
use crate::preview::{PreviewMix, window_bytes};

/// Frames kept streamed ahead of the device: about 128 ms.
pub const LEAD: usize = 6_144;
/// One mixed window: 20 ms, a whole number of microseconds, so windows meet
/// exactly.
pub const WINDOW: u32 = RATE / 50;
/// How often a full stream is checked for room.
const POLL: Duration = Duration::from_millis(5);

/// What the mixer plays: a draft from one time to the end of its cut.
#[derive(Clone, Debug)]
pub struct MixJob {
    pub draft: Draft,
    pub revision: u64,
    pub root: Option<PathBuf>,
    pub start: Micros,
    pub end: Micros,
}

impl MixJob {
    /// The timeline time `frames` after the start.
    #[must_use]
    pub fn time_at(&self, frames: u64) -> Micros {
        let micros = i128::from(frames) * 1_000_000 / i128::from(RATE);
        Micros(
            self.start
                .0
                .saturating_add(i64::try_from(micros).unwrap_or(i64::MAX)),
        )
    }
}

/// The preview's mixing thread.
pub struct Mixer {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    sink: AudioSink,
}

struct State {
    job: Option<Arc<MixJob>>,
    /// Frames streamed for `job`.
    mixed: u64,
    /// Changes on every play and stop. A window mixed for another
    /// generation is dropped.
    generation: u64,
    stop: bool,
}

impl Mixer {
    /// A mixer streaming into `output`, or `None` when the host has no
    /// device stream.
    #[must_use]
    pub fn start(output: &AudioOutput, readers: ReaderFactory) -> Option<Self> {
        let sink = output.sink()?;
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                job: None,
                mixed: 0,
                generation: 0,
                stop: false,
            }),
            wake: Condvar::new(),
            sink,
        });
        let worker_shared = Arc::clone(&shared);
        let worker = std::thread::Builder::new()
            .name("cutlass-mixer".to_owned())
            .spawn(move || run(&worker_shared, &readers))
            .ok()?;
        Some(Self {
            shared,
            worker: Some(worker),
        })
    }

    /// Drop what is streamed and play `job` from its start.
    pub fn play(&self, job: MixJob) {
        let mut state = lock(&self.shared.state);
        self.shared.sink.clear();
        state.job = Some(Arc::new(job));
        state.mixed = 0;
        state.generation = state.generation.wrapping_add(1);
        self.shared.wake.notify_all();
    }

    /// Drop what is streamed and mix nothing.
    pub fn stop(&self) {
        let mut state = lock(&self.shared.state);
        self.shared.sink.clear();
        state.job = None;
        state.mixed = 0;
        state.generation = state.generation.wrapping_add(1);
    }

    /// Frames streamed for the playing job.
    #[must_use]
    pub fn mixed(&self) -> u64 {
        lock(&self.shared.state).mixed
    }

    /// The draft revision being played, if any.
    #[must_use]
    pub fn revision(&self) -> Option<u64> {
        lock(&self.shared.state)
            .job
            .as_ref()
            .map(|job| job.revision)
    }
}

impl Drop for Mixer {
    fn drop(&mut self) {
        {
            let mut state = lock(&self.shared.state);
            state.stop = true;
            state.job = None;
            self.shared.sink.clear();
        }
        self.shared.wake.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn lock(state: &Mutex<State>) -> MutexGuard<'_, State> {
    state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn run(shared: &Shared, readers: &ReaderFactory) {
    let mut heard = Heard::new();
    let mut state = lock(&shared.state);
    loop {
        if state.stop {
            return;
        }
        let Some(job) = state.job.clone() else {
            state = shared
                .wake
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            continue;
        };
        let at = job.time_at(state.mixed);
        if at.0 >= job.end.0 || shared.sink.buffered() >= LEAD {
            state = shared
                .wake
                .wait_timeout(state, POLL)
                .map(|(state, _)| state)
                .unwrap_or_else(|poison| poison.into_inner().0);
            continue;
        }
        let generation = state.generation;
        drop(state);
        let bytes = window_bytes(
            PreviewMix {
                draft: &job.draft,
                revision: job.revision,
                time: at,
                frames: WINDOW,
                readers,
                heard: &mut heard,
                root: job.root.as_deref(),
            },
            WINDOW,
        );
        state = lock(&shared.state);
        if state.generation == generation {
            shared.sink.append(&bytes);
            state.mixed = state.mixed.saturating_add(u64::from(WINDOW));
        }
    }
}

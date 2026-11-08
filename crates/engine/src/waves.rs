//! Timeline waveforms, read on their own thread.
//!
//! A file's peaks are the loudest sample, across every channel, in each
//! [`BUCKETS_PER_SECOND`] slice of its audio, counted from the first decoded
//! sample. The worker reads one file start to end and publishes what it has
//! every [`PUBLISH_SECONDS`] of audio, so a long file draws as it is read.
//! A lookup never reads and never waits. While paused the worker stops
//! between buffers, so playback keeps the decoder to itself.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;

use cutlass_codec::access::Pcm;
use cutlass_model::Micros;

use crate::error::EngineError;
use crate::reader::ReaderFactory;

/// Peak buckets in one second of audio.
pub const BUCKETS_PER_SECOND: u32 = 100;

/// Seconds of audio read between two publishes of a file's peaks.
const PUBLISH_SECONDS: u32 = 10;

/// Files whose peaks are kept. The oldest is dropped past this.
const MAX_FILES: usize = 64;

/// A file's peaks, 0 to 1, one per bucket.
#[derive(Clone, Debug, PartialEq)]
pub struct Peaks {
    pub levels: Vec<f32>,
    /// The whole file has been read.
    pub complete: bool,
}

impl Peaks {
    /// The loudest level from `from` up to `to`, source time. A span inside
    /// one bucket reads that bucket. Past what was read is silence.
    #[must_use]
    pub fn level(&self, from: Micros, to: Micros) -> f32 {
        let bucket = |time: Micros| {
            usize::try_from(time.0.max(0) * i64::from(BUCKETS_PER_SECOND) / 1_000_000).unwrap_or(0)
        };
        let (first, last) = (bucket(from.min(to)), bucket(from.max(to)));
        self.levels
            .get(first..=last.min(self.levels.len().saturating_sub(1)))
            .map_or(0.0, |span| span.iter().copied().fold(0.0, f32::max))
    }

    /// The loudest level in each of `columns` equal spans of `duration`, the
    /// file's length in microseconds. Without a length, the peaks say how
    /// long the file is. Past what was read is silence.
    #[must_use]
    pub fn spread(&self, duration: i64, columns: usize) -> Vec<f32> {
        let length = if duration > 0 {
            duration
        } else {
            i64::try_from(self.levels.len())
                .unwrap_or(i64::MAX)
                .saturating_mul(1_000_000)
                / i64::from(BUCKETS_PER_SECOND)
        };
        let count = i64::try_from(columns.max(1)).unwrap_or(i64::MAX);
        (0..count)
            .map(|column| {
                let from = Micros(length.saturating_mul(column) / count);
                let to = Micros((length.saturating_mul(column + 1) / count - 1).max(from.0));
                self.level(from, to)
            })
            .collect()
    }
}

/// The waveform store and the thread that fills it.
pub struct Waveforms {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
}

#[derive(Default)]
struct State {
    stored: HashMap<PathBuf, Arc<Peaks>>,
    order: VecDeque<PathBuf>,
    wanted: VecDeque<PathBuf>,
    failed: HashSet<PathBuf>,
    /// The file the worker is reading now.
    reading: Option<PathBuf>,
    paused: bool,
    closed: bool,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Wait out a pause. False once the store is closing.
    fn resume(&self) -> bool {
        let mut state = self.lock();
        while state.paused && !state.closed {
            state = self
                .wake
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        !state.closed
    }
}

impl Waveforms {
    /// Start the worker. `ready` runs on the worker after each publish, so
    /// the caller can draw again.
    pub fn start(factory: ReaderFactory, ready: impl Fn() + Send + 'static) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
        });
        let worker = {
            let shared = Arc::clone(&shared);
            std::thread::Builder::new()
                .name("cutlass-waveforms".to_owned())
                .spawn(move || work(&shared, &factory, &ready))
                .ok()
        };
        Self { shared, worker }
    }

    /// The peaks read so far for `path`. This never reads.
    #[must_use]
    pub fn get(&self, path: &std::path::Path) -> Option<Arc<Peaks>> {
        self.shared.lock().stored.get(path).cloned()
    }

    /// Ask for `paths`, most wanted first. The asks replace the earlier ones;
    /// a file already read, being read, or failed is not asked again.
    pub fn want(&self, paths: impl IntoIterator<Item = PathBuf>) {
        let mut state = self.shared.lock();
        let mut seen = HashSet::new();
        let wanted: VecDeque<PathBuf> = paths
            .into_iter()
            .filter(|path| {
                !state.stored.contains_key(path)
                    && !state.failed.contains(path)
                    && state.reading.as_ref() != Some(path)
            })
            .filter(|path| seen.insert(path.clone()))
            .collect();
        state.wanted = wanted;
        drop(state);
        self.shared.wake.notify_all();
    }

    /// Stop reading between buffers, or carry on.
    pub fn set_paused(&self, paused: bool) {
        self.shared.lock().paused = paused;
        self.shared.wake.notify_all();
    }
}

impl Drop for Waveforms {
    fn drop(&mut self) {
        self.shared.lock().closed = true;
        self.shared.wake.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn work(shared: &Shared, factory: &ReaderFactory, ready: &dyn Fn()) {
    loop {
        let path = {
            let mut state = shared.lock();
            loop {
                if state.closed {
                    return;
                }
                if !state.paused
                    && let Some(path) = state.wanted.pop_front()
                {
                    state.reading = Some(path.clone());
                    break path;
                }
                state = shared
                    .wake
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        };
        let publish = |peaks: Peaks| {
            let mut state = shared.lock();
            if !state.stored.contains_key(&path) {
                state.order.push_back(path.clone());
            }
            state.stored.insert(path.clone(), Arc::new(peaks));
            while state.order.len() > MAX_FILES {
                if let Some(oldest) = state.order.pop_front() {
                    state.stored.remove(&oldest);
                }
            }
            drop(state);
            ready();
        };
        let read = read_peaks(factory, &path, shared, &publish);
        let mut state = shared.lock();
        state.reading = None;
        if read.is_err() && !state.stored.contains_key(&path) {
            state.failed.insert(path);
            drop(state);
            ready();
        }
    }
}

/// Read `path` start to end, publishing its peaks as they grow.
fn read_peaks(
    factory: &ReaderFactory,
    path: &std::path::Path,
    shared: &Shared,
    publish: &dyn Fn(Peaks),
) -> Result<(), EngineError> {
    let mut media = factory.open(path)?;
    let mut folder = Folder::default();
    let mut last_publish = 0_u64;
    while shared.resume() {
        let Some(ready) = media.pull_audio()? else {
            folder.finish();
            publish(Peaks {
                levels: folder.levels,
                complete: true,
            });
            return Ok(());
        };
        folder.fold(&ready.pcm)?;
        if folder.seconds() >= last_publish + u64::from(PUBLISH_SECONDS) {
            last_publish = folder.seconds();
            publish(Peaks {
                levels: folder.levels.clone(),
                complete: false,
            });
        }
    }
    Ok(())
}

/// Folds decoded buffers into peak buckets.
#[derive(Default)]
struct Folder {
    levels: Vec<f32>,
    /// Frames folded so far, and the rate they arrived at.
    frames: u64,
    rate: u32,
    /// The loudest sample of the bucket still filling.
    open: f32,
}

impl Folder {
    fn fold(&mut self, pcm: &Pcm) -> Result<(), EngineError> {
        if pcm.rate == 0 || pcm.layout.is_empty() {
            return Err(EngineError::Layout);
        }
        if self.rate == 0 {
            self.rate = pcm.rate;
        }
        let channels = pcm.layout.len();
        let level_at = |frame: usize| -> f32 {
            match (pcm.as_f32(), pcm.as_i16()) {
                (Some(samples), _) => samples[frame * channels..(frame + 1) * channels]
                    .iter()
                    .map(|sample| {
                        if sample.is_finite() {
                            sample.abs()
                        } else {
                            0.0
                        }
                    })
                    .fold(0.0, f32::max),
                (None, Some(samples)) => samples[frame * channels..(frame + 1) * channels]
                    .iter()
                    .map(|sample| f32::from(*sample).abs() / 32_768.0)
                    .fold(0.0, f32::max),
                (None, None) => 0.0,
            }
        };
        let frames = usize::try_from(pcm.samples).map_err(|_| EngineError::Layout)?;
        for frame in 0..frames {
            let bucket = self.frames * u64::from(BUCKETS_PER_SECOND) / u64::from(self.rate);
            let bucket = usize::try_from(bucket).map_err(|_| EngineError::Layout)?;
            while self.levels.len() < bucket {
                self.levels.push(self.open.min(1.0));
                self.open = 0.0;
            }
            self.open = self.open.max(level_at(frame));
            self.frames += 1;
        }
        Ok(())
    }

    /// Close the bucket still filling.
    fn finish(&mut self) {
        if self.frames > 0 {
            self.levels.push(self.open.min(1.0));
            self.open = 0.0;
        }
    }

    fn seconds(&self) -> u64 {
        if self.rate == 0 {
            return 0;
        }
        self.frames / u64::from(self.rate)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_spread_takes_the_loudest_level_in_each_column() {
        let peaks = Peaks {
            levels: vec![0.1, 0.9, 0.2, 0.3, 0.0, 0.5],
            complete: true,
        };
        // 60 ms of sound across three columns, 20 ms each.
        assert_eq!(peaks.spread(60_000, 3), [0.9, 0.3, 0.5]);
        // Without a stored length, the peaks say how long the file is.
        assert_eq!(peaks.spread(0, 2), [0.9, 0.5]);
    }

    use super::*;
    use cutlass_codec::access::ChannelLayout;

    #[test]
    fn buckets_keep_the_loudest_sample_of_any_channel() {
        let mut folder = Folder::default();
        // 200 stereo frames at 10 kHz: two 10 ms buckets of 100 frames.
        let mut samples = vec![0.0_f32; 400];
        samples[10] = -0.5;
        samples[201] = 0.25;
        samples[399] = 0.75;
        let pcm = Pcm::from_f32(10_000, ChannelLayout::stereo(), samples).unwrap();
        folder.fold(&pcm).unwrap();
        folder.finish();
        assert_eq!(folder.levels, [0.5, 0.75]);
    }

    #[test]
    fn a_level_spans_the_buckets_it_covers_and_past_the_end_is_silent() {
        let peaks = Peaks {
            levels: vec![0.1, 0.9, 0.2],
            complete: true,
        };
        assert_eq!(peaks.level(Micros(0), Micros(5_000)), 0.1);
        assert_eq!(peaks.level(Micros(0), Micros(25_000)), 0.9);
        assert_eq!(peaks.level(Micros(20_000), Micros(20_000)), 0.2);
        assert_eq!(peaks.level(Micros(90_000), Micros(95_000)), 0.0);
    }

    #[test]
    fn a_missing_file_fails_and_is_not_asked_again() {
        let (sender, receiver) = std::sync::mpsc::channel();
        let waves = Waveforms::start(ReaderFactory::new(), move || {
            let _ = sender.send(());
        });
        let path = PathBuf::from("/nowhere/missing.wav");
        waves.want([path.clone()]);
        receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("the worker answered");
        assert!(waves.get(&path).is_none());
        waves.want([path]);
        assert!(waves.shared.lock().wanted.is_empty());
    }
}

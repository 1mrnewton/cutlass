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
use std::time::{Duration, Instant};

use cutlass_codec::access::Pcm;
use cutlass_model::Micros;

use crate::error::EngineError;
use crate::reader::ReaderFactory;

/// Peak buckets in one second of audio.
pub const BUCKETS_PER_SECOND: u32 = 100;

/// Seconds of audio read between two publishes of a file's peaks.
const PUBLISH_SECONDS: u32 = 10;

/// Files whose peaks are kept. The oldest is dropped past this, and one ask
/// takes no more than this many files, so the files in view all fit and
/// reading one never drops another that is in view.
const MAX_FILES: usize = 64;

/// How long a file that failed is left alone before an ask tries it again:
/// long enough that a broken file is not reread every frame, short enough
/// that one still being written or copied in draws once it is whole.
const RETRY_AFTER: Duration = Duration::from_secs(30);

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
            usize::try_from(time.0.max(0).saturating_mul(i64::from(BUCKETS_PER_SECOND)) / 1_000_000)
                .unwrap_or(usize::MAX)
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
    /// The files the last ask named, which eviction keeps while it can.
    view: HashSet<PathBuf>,
    /// When each file last failed to read.
    failed: HashMap<PathBuf, Instant>,
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

    /// Ask for `paths`, most wanted first, the first [`MAX_FILES`] of them.
    /// The asks replace the earlier ones. A file read to its end or being
    /// read is not asked again; one whose read stopped part way is; one that
    /// failed is asked again once [`RETRY_AFTER`] has passed.
    pub fn want(&self, paths: impl IntoIterator<Item = PathBuf>) {
        let now = Instant::now();
        let mut state = self.shared.lock();
        let mut view = HashSet::new();
        let named: Vec<PathBuf> = paths
            .into_iter()
            .filter(|path| view.insert(path.clone()))
            .take(MAX_FILES)
            .collect();
        view.clear();
        view.extend(named.iter().cloned());
        state
            .failed
            .retain(|_, when| now.duration_since(*when) < RETRY_AFTER);
        let wanted: VecDeque<PathBuf> = named
            .into_iter()
            .filter(|path| {
                !state.stored.get(path).is_some_and(|peaks| peaks.complete)
                    && !state.failed.contains_key(path)
                    && state.reading.as_ref() != Some(path)
            })
            .collect();
        state.wanted = wanted;
        state.view = view;
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
            state.keep(&path, peaks);
            drop(state);
            ready();
        };
        let read = read_peaks(factory, &path, shared, &publish);
        let mut state = shared.lock();
        state.reading = None;
        if read.is_err() {
            // What was read before the error stays drawn; the rest is asked
            // for again after the wait.
            state.failed.insert(path, Instant::now());
            drop(state);
            ready();
        }
    }
}

impl State {
    /// Store `peaks` for `path` as the newest file, dropping the oldest past
    /// [`MAX_FILES`]: first those out of view, then the oldest in view.
    fn keep(&mut self, path: &PathBuf, peaks: Peaks) {
        self.order.retain(|kept| kept != path);
        self.order.push_back(path.clone());
        self.stored.insert(path.clone(), Arc::new(peaks));
        while self.order.len() > MAX_FILES {
            let out = self
                .order
                .iter()
                .position(|kept| !self.view.contains(kept))
                .unwrap_or(0);
            if let Some(oldest) = self.order.remove(out) {
                self.stored.remove(&oldest);
            }
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
    /// Microseconds folded at rates before the current one.
    before_us: u64,
    /// Frames folded at the current rate, and that rate.
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
        if self.rate != pcm.rate {
            // A stream that changes rate part way (an AAC switch, a joined
            // recording) keeps its buckets on time: what came before counts
            // as time, and the new rate counts from there.
            self.before_us = self.elapsed_us();
            self.frames = 0;
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
            let bucket = self.bucket();
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
        self.elapsed_us() / 1_000_000
    }

    /// Microseconds folded so far, rounded down.
    fn elapsed_us(&self) -> u64 {
        if self.rate == 0 {
            return self.before_us;
        }
        let now = u128::from(self.frames) * 1_000_000 / u128::from(self.rate);
        self.before_us
            .saturating_add(u64::try_from(now).unwrap_or(u64::MAX))
    }

    /// The bucket the next frame falls in.
    fn bucket(&self) -> u64 {
        let rate = u128::from(self.rate.max(1));
        let micros = u128::from(self.before_us) * rate + u128::from(self.frames) * 1_000_000;
        let bucket = micros * u128::from(BUCKETS_PER_SECOND) / (rate * 1_000_000);
        u64::try_from(bucket).unwrap_or(u64::MAX)
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
    fn a_missing_file_fails_and_is_asked_again_only_after_the_wait() {
        let (sender, receiver) = std::sync::mpsc::channel();
        let waves = Waveforms::start(ReaderFactory::new(), move || {
            let _ = sender.send(());
        });
        let path = PathBuf::from("/nowhere/missing.wav");
        waves.want([path.clone()]);
        receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("the worker answered");
        assert!(waves.get(&path).is_none());
        waves.set_paused(true);
        waves.want([path.clone()]);
        assert!(waves.shared.lock().wanted.is_empty());
        // Once the wait has passed, the next ask tries the file again.
        let long_ago = Instant::now()
            .checked_sub(RETRY_AFTER + Duration::from_secs(1))
            .expect("clock");
        waves.shared.lock().failed.insert(path.clone(), long_ago);
        waves.want([path.clone()]);
        assert_eq!(waves.shared.lock().wanted, [path]);
    }

    #[test]
    fn a_part_read_file_is_asked_again_and_a_whole_one_is_not() {
        let waves = Waveforms::start(ReaderFactory::new(), || {});
        waves.set_paused(true);
        let (part, whole) = (PathBuf::from("/a/part.wav"), PathBuf::from("/a/whole.wav"));
        {
            let mut state = waves.shared.lock();
            let peaks = |complete| Peaks {
                levels: vec![0.5],
                complete,
            };
            state.keep(&part, peaks(false));
            state.keep(&whole, peaks(true));
        }
        waves.want([whole, part.clone()]);
        assert_eq!(waves.shared.lock().wanted, [part]);
    }

    #[test]
    fn more_files_in_view_than_are_kept_do_not_push_each_other_out() {
        let waves = Waveforms::start(ReaderFactory::new(), || {});
        waves.set_paused(true);
        let path = |n: usize| PathBuf::from(format!("/a/{n}.wav"));
        // One ask names 70 files: only the first 64 are taken.
        waves.want((0..70).map(path));
        assert_eq!(waves.shared.lock().wanted.len(), MAX_FILES);
        // Storing them all, plus one out of view stored first, drops the
        // one out of view and keeps every file in view.
        let mut state = waves.shared.lock();
        let peaks = || Peaks {
            levels: vec![0.5],
            complete: true,
        };
        state.view.remove(&path(0));
        state.keep(&path(999), peaks());
        state.view.insert(path(0));
        for n in 0..MAX_FILES {
            state.keep(&path(n), peaks());
        }
        assert_eq!(state.stored.len(), MAX_FILES);
        assert!(!state.stored.contains_key(&path(999)));
        assert!((0..MAX_FILES).all(|n| state.stored.contains_key(&path(n))));
        // Storing a file again moves it to the newest, with no copy left
        // behind in the order.
        state.keep(&path(0), peaks());
        assert_eq!(state.order.len(), MAX_FILES);
        assert_eq!(state.order.back(), Some(&path(0)));
    }

    #[test]
    fn a_rate_change_keeps_the_buckets_on_time() {
        let mut folder = Folder::default();
        // 15 ms at 10 kHz, then 15 ms at 20 kHz, a sound in each half's
        // last millisecond: buckets 1 and 2 of three 10 ms buckets.
        let mut first = vec![0.0_f32; 150];
        first[140] = 0.5;
        let mut second = vec![0.0_f32; 300];
        second[290] = 0.75;
        folder
            .fold(&Pcm::from_f32(10_000, ChannelLayout::mono(), first).unwrap())
            .unwrap();
        folder
            .fold(&Pcm::from_f32(20_000, ChannelLayout::mono(), second).unwrap())
            .unwrap();
        folder.finish();
        assert_eq!(folder.levels, [0.0, 0.5, 0.75]);
        assert_eq!(folder.elapsed_us(), 30_000);
        // A time far past any file does not overflow.
        let peaks = Peaks {
            levels: vec![0.1],
            complete: true,
        };
        assert_eq!(peaks.level(Micros(i64::MAX - 1), Micros(i64::MAX)), 0.0);
    }
}

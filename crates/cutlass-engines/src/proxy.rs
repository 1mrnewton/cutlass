//! Background proxy builds: the disk tier beneath the in-RAM [`FrameCache`].
//!
//! On import we kick off a background transcode of each source into a 1080p
//! **all-intra H.264** proxy (see [`cutlass_decode::build_proxy`]). Once a proxy
//! is on disk, the [`MediaPool`](crate::MediaPool) reads frames from it instead
//! of the long-GOP source, turning a 0.4–1.6 s cold seek into a flat ~9 ms one.
//! See `docs/proxy-cache/research.md` for the measurements and the full design.
//!
//! ## Scheduling (P2)
//!
//! Builds run on a small pool of **lanes** — independent worker threads, each
//! with its own ffmpeg decoder (so they never share state with the live reader).
//! Lanes pull from one shared **priority queue**, so importing several clips at
//! once builds them in parallel and work-steals: whichever lane is free grabs the
//! highest-priority pending job. The default mix is one software lane (the fastest
//! single pipeline) plus one hardware-decode lane (offloads the CPU so a second
//! concurrent build doesn't fight the SW lane for cores).
//!
//! The queue supports two interactive levers:
//! - **playhead priority**: [`ProxyService::prioritize`] bumps the clip under the
//!   playhead so it's the next thing built.
//! - **pause**: [`ProxyService::set_paused`] parks the lanes during heavy
//!   interaction (scrub/playback) so background transcodes don't steal cycles.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use cutlass_decode::{build_proxy_with, HwAccel, ProxyBuildOptions, ProxyConfig, ProxyStats};
use cutlass_models::MediaId;
use rustc_hash::FxHasher;
use tracing::{debug, warn};

/// Bump to invalidate every proxy on disk after a format/codec/quality change.
/// v2: raised the all-intra H.264 bitrate 12→60 Mbps to kill grain artifacts.
/// v3: switched to constant-quality software libx264 (CRF) — cleaner per bit
/// than VideoToolbox, which grained even at high bitrate.
const CACHE_VERSION: u32 = 3;

/// State of a media's on-disk proxy.
#[derive(Debug, Clone)]
pub enum ProxyStatus {
    /// No proxy requested.
    None,
    /// A background build is in flight; `progress` is in `0.0..=1.0`.
    Building { progress: f32 },
    /// Proxy is on disk at this path and registered as the fast reader.
    Ready(PathBuf),
    /// The build failed; the source reader is used permanently.
    Failed(String),
}

impl ProxyStatus {
    pub fn is_ready(&self) -> bool {
        matches!(self, ProxyStatus::Ready(_))
    }

    /// Build progress in `0.0..=1.0`, or `None` when not currently building.
    pub fn progress(&self) -> Option<f32> {
        match self {
            ProxyStatus::Building { progress } => Some(*progress),
            _ => None,
        }
    }
}

/// How a single lane decodes the source during a build.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LaneConfig {
    pub decode: HwAccel,
    pub decode_threads: u32,
}

/// A queued transcode job. `priority` orders the shared queue (higher first);
/// `total_frames` lets the worker turn an encoded-frame count into a percentage.
pub(crate) struct RenderJob {
    pub media: MediaId,
    pub source: PathBuf,
    pub output: PathBuf,
    pub config: ProxyConfig,
    pub total_frames: u64,
    pub priority: i64,
}

/// A completed (or failed) build reported back from a lane.
pub(crate) struct ProxyDone {
    pub media: MediaId,
    pub output: PathBuf,
    pub result: Result<ProxyStats, String>,
}

/// A message from a lane back to the pool.
pub(crate) enum ProxyMsg {
    /// Periodic progress for an in-flight build.
    Progress { media: MediaId, progress: f32 },
    /// A build finished (successfully or not).
    Done(ProxyDone),
}

/// Shared, lane-fed priority queue.
struct Queue {
    inner: Mutex<QueueInner>,
    cv: Condvar,
}

struct QueueInner {
    jobs: Vec<RenderJob>,
    /// When set, lanes park instead of pulling work (interactive throttle).
    paused: bool,
    /// When set, lanes exit their loop.
    shutdown: bool,
}

impl QueueInner {
    /// Index of the next job to run: highest priority, ties broken by insertion
    /// order (earliest first), so equal-priority imports stay FIFO.
    fn best_index(&self) -> Option<usize> {
        self.jobs
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| {
                // Higher priority wins; on a tie the lower index is "greater".
                a.priority.cmp(&b.priority).then(ib.cmp(ia))
            })
            .map(|(i, _)| i)
    }
}

impl Queue {
    fn push(&self, job: RenderJob) {
        self.inner.lock().unwrap().jobs.push(job);
        self.cv.notify_one();
    }

    /// Raise the priority of `media`'s pending job (no-op if already running or
    /// not queued). Only ever raises, so a later, lower bump can't demote.
    fn prioritize(&self, media: MediaId, priority: i64) {
        let mut g = self.inner.lock().unwrap();
        let mut changed = false;
        for job in g.jobs.iter_mut() {
            if job.media == media && priority > job.priority {
                job.priority = priority;
                changed = true;
            }
        }
        drop(g);
        if changed {
            self.cv.notify_all();
        }
    }

    fn set_paused(&self, paused: bool) {
        self.inner.lock().unwrap().paused = paused;
        if !paused {
            self.cv.notify_all();
        }
    }

    fn shutdown(&self) {
        self.inner.lock().unwrap().shutdown = true;
        self.cv.notify_all();
    }
}

/// Owns the lane threads plus the shared queue and the completion channel.
///
/// Dropping the service signals shutdown; lanes finish any in-flight build and
/// then exit (drop is non-blocking — it doesn't join).
pub(crate) struct ProxyService {
    queue: Arc<Queue>,
    msg_rx: Receiver<ProxyMsg>,
}

impl ProxyService {
    /// Spawn the default lane mix (one SW lane + one HW-decode lane).
    pub fn spawn() -> Self {
        Self::spawn_lanes(default_lanes())
    }

    pub fn spawn_lanes(lanes: Vec<LaneConfig>) -> Self {
        let queue = Arc::new(Queue {
            inner: Mutex::new(QueueInner {
                jobs: Vec::new(),
                paused: false,
                shutdown: false,
            }),
            cv: Condvar::new(),
        });
        let (msg_tx, msg_rx) = mpsc::channel::<ProxyMsg>();

        for (i, lane) in lanes.into_iter().enumerate() {
            let queue = Arc::clone(&queue);
            let tx = msg_tx.clone();
            thread::Builder::new()
                .name(format!("cutlass-proxy-{i}"))
                .spawn(move || run_lane(queue, lane, tx))
                .expect("spawn cutlass-proxy lane");
        }
        // Drop the template sender so `msg_rx` closes once every lane exits.
        drop(msg_tx);

        Self { queue, msg_rx }
    }

    pub fn submit(&self, job: RenderJob) {
        self.queue.push(job);
    }

    pub fn prioritize(&self, media: MediaId, priority: i64) {
        self.queue.prioritize(media, priority);
    }

    pub fn set_paused(&self, paused: bool) {
        self.queue.set_paused(paused);
    }

    /// Non-blocking drain of one lane message, if any is ready.
    pub fn try_recv(&self) -> Option<ProxyMsg> {
        self.msg_rx.try_recv().ok()
    }
}

impl Drop for ProxyService {
    fn drop(&mut self) {
        // Park/wake lanes so they observe shutdown and exit; don't block on join.
        self.queue.shutdown();
    }
}

/// A lane's run loop: pull the best job, build it, report progress + result.
fn run_lane(queue: Arc<Queue>, lane: LaneConfig, tx: Sender<ProxyMsg>) {
    loop {
        let job = {
            let mut g = queue.inner.lock().unwrap();
            loop {
                if g.shutdown {
                    return;
                }
                if !g.paused
                    && let Some(idx) = g.best_index()
                {
                    break g.jobs.remove(idx);
                }
                g = queue.cv.wait(g).unwrap();
            }
        };
        build_one(&job, lane, &tx);
    }
}

fn build_one(job: &RenderJob, lane: LaneConfig, tx: &Sender<ProxyMsg>) {
    debug!(
        media = %job.media,
        ?job.output,
        decode = lane.decode.name(),
        "building proxy"
    );
    let opts = ProxyBuildOptions {
        decode: lane.decode,
        decode_threads: lane.decode_threads,
    };
    let media = job.media;
    let total = job.total_frames.max(1) as f32;
    let progress_tx = tx.clone();
    // Coalesce to whole-percent steps so the channel doesn't see one msg/frame.
    let mut last_pct: i32 = -1;
    let mut on_progress = move |done: u64| {
        let p = (done as f32 / total).clamp(0.0, 0.99);
        let pct = (p * 100.0) as i32;
        if pct != last_pct {
            last_pct = pct;
            let _ = progress_tx.send(ProxyMsg::Progress { media, progress: p });
        }
    };
    let result = build_proxy_with(&job.source, &job.output, job.config, opts, Some(&mut on_progress))
        .map_err(|e| e.to_string());
    let _ = tx.send(ProxyMsg::Done(ProxyDone {
        media: job.media,
        output: job.output.clone(),
        result,
    }));
}

/// Default lanes: a software lane (all cores — the fastest single pipeline) plus
/// a hardware-decode lane that runs a *second* concurrent build cheaply on the
/// CPU. `Auto` falls back to software where no GPU decode exists, so the second
/// lane stays correct everywhere (just capped so it can't oversubscribe).
fn default_lanes() -> Vec<LaneConfig> {
    vec![
        LaneConfig {
            decode: HwAccel::None,
            decode_threads: 0,
        },
        LaneConfig {
            decode: HwAccel::Auto,
            decode_threads: 2,
        },
    ]
}

/// Where proxies live: content-addressed, outside the project, safe to delete.
pub fn proxy_cache_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join("Library/Caches/cutlass/proxies");
    }
    std::env::temp_dir().join("cutlass/proxies")
}

/// Stable cache key for a source + proxy resolution. Content-addressed via path,
/// size, and mtime so edits to the file invalidate it but moves/renames within a
/// project do not re-key gratuitously.
pub fn render_hash(source: &Path, target_height: u32) -> String {
    let mut h = FxHasher::default();
    CACHE_VERSION.hash(&mut h);
    source.to_string_lossy().hash(&mut h);
    if let Ok(meta) = std::fs::metadata(source) {
        meta.len().hash(&mut h);
        if let Ok(mtime) = meta.modified()
            && let Ok(dur) = mtime.duration_since(std::time::UNIX_EPOCH)
        {
            dur.as_secs().hash(&mut h);
        }
    }
    target_height.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// The proxy file path for a source under the cache dir.
pub fn proxy_path(source: &Path, target_height: u32) -> PathBuf {
    proxy_cache_dir().join(format!("{}.mp4", render_hash(source, target_height)))
}

/// Byte-bounded LRU eviction over the on-disk proxy directory.
///
/// Mirrors [`FrameCache`](crate::FrameCache)'s budget but for disk: the disk
/// footprint of intra proxies is large, so without this the cache grows without
/// bound (the lesson from Filmora's 42 GB `.Render`).
pub(crate) struct DiskBudget {
    cap_bytes: u64,
}

impl DiskBudget {
    pub fn new(cap_bytes: u64) -> Self {
        Self { cap_bytes }
    }

    /// Delete least-recently-modified proxies until the directory fits the cap.
    /// `keep` paths (currently-open proxies) are never deleted. Best-effort.
    pub fn prune(&self, dir: &Path, keep: &HashSet<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut files: Vec<(PathBuf, u64, std::time::SystemTime)> = Vec::new();
        let mut total: u64 = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(meta) = entry.metadata() else { continue };
            if !meta.is_file() {
                continue;
            }
            total += meta.len();
            let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
            files.push((path, meta.len(), mtime));
        }
        if total <= self.cap_bytes {
            return;
        }
        // Oldest first.
        files.sort_by_key(|(_, _, mtime)| *mtime);
        for (path, len, _) in files {
            if total <= self.cap_bytes {
                break;
            }
            if keep.contains(&path) {
                continue;
            }
            match std::fs::remove_file(&path) {
                Ok(()) => {
                    total = total.saturating_sub(len);
                    debug!(?path, "evicted proxy to fit disk budget");
                }
                Err(e) => warn!(?path, error = %e, "failed to evict proxy"),
            }
        }
    }
}

/// Default disk budget for proxies: 50 GiB.
pub const DEFAULT_DISK_BUDGET_BYTES: u64 = 50 * 1024 * 1024 * 1024;

#[cfg(test)]
mod tests {
    use super::*;

    fn job(media: u64, priority: i64) -> RenderJob {
        RenderJob {
            media: MediaId::from_raw(media),
            source: PathBuf::from("/tmp/src.mp4"),
            output: PathBuf::from("/tmp/out.mp4"),
            config: ProxyConfig::default(),
            total_frames: 100,
            priority,
        }
    }

    #[test]
    fn best_index_picks_highest_priority() {
        let q = QueueInner {
            jobs: vec![job(1, 0), job(2, 5), job(3, 2)],
            paused: false,
            shutdown: false,
        };
        assert_eq!(q.best_index(), Some(1), "media 2 has the highest priority");
    }

    #[test]
    fn best_index_breaks_ties_fifo() {
        let q = QueueInner {
            jobs: vec![job(1, 3), job(2, 3), job(3, 3)],
            paused: false,
            shutdown: false,
        };
        assert_eq!(q.best_index(), Some(0), "equal priority stays FIFO");
    }

    #[test]
    fn prioritize_only_raises_and_reorders() {
        let queue = Queue {
            inner: Mutex::new(QueueInner {
                jobs: vec![job(1, 0), job(2, 0)],
                paused: false,
                shutdown: false,
            }),
            cv: Condvar::new(),
        };
        queue.prioritize(MediaId::from_raw(2), 10);
        {
            let g = queue.inner.lock().unwrap();
            assert_eq!(g.best_index(), Some(1), "bumped media 2 now wins");
        }
        // A lower bump must not demote it.
        queue.prioritize(MediaId::from_raw(2), 1);
        let g = queue.inner.lock().unwrap();
        assert_eq!(g.jobs[1].priority, 10, "priority only ever rises");
    }

    #[test]
    fn render_hash_is_stable_and_resolution_specific() {
        let p = Path::new("/tmp/does-not-exist.mp4");
        let a = render_hash(p, 1080);
        let b = render_hash(p, 1080);
        let c = render_hash(p, 540);
        assert_eq!(a, b, "same inputs hash the same");
        assert_ne!(a, c, "resolution is part of the key");
        assert_eq!(a.len(), 16);
    }

    #[test]
    fn proxy_path_lives_under_cache_dir() {
        let p = proxy_path(Path::new("/tmp/x.mp4"), 1080);
        assert!(p.starts_with(proxy_cache_dir()));
        assert_eq!(p.extension().unwrap(), "mp4");
    }
}

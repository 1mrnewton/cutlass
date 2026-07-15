//! Timeline clip content: filmstrip frames for video clips, waveform tiles
//! for audio clips.
//!
//! Architecture mirrors the library thumbnails (`src/thumbnails.rs`) plus the
//! ruler's viewport-virtualized pure callback (`src/ruler.rs`):
//!
//! * `ClipView` asks [`StripBackend`] for the tiles covering its *visible*
//!   slice. The resolvers here run on the UI thread and only do cache
//!   lookups — misses enqueue decode work and return empty-image tiles.
//! * A dedicated `cutlass-strips` thread decodes frames / computes audio
//!   peaks / rasterizes waveform tiles, newest request first (the user is
//!   looking at what they asked for last).
//! * Finished tiles land in UI-thread registries via `invoke_from_event_loop`
//!   and bump `StripBackend.generation`, which every resolver call takes as
//!   an argument — Slint re-evaluates the visible tile models automatically.
//!
//! Cache keys are anchored to *media time*, quantized to a power-of-two grid
//! of seconds chosen from the zoom level. Powers of two nest: every frame
//! decoded for a coarse zoom sits on the grid of every finer zoom, so zooming
//! reuses instead of re-decoding. Trims and moves shift the clip's window
//! over the same grid — also pure cache hits.
//!
//! Frame decode rides the scratch-project renderer (the mobile thumbnailer
//! pattern): one media-spanning clip fit-rendered per grid time, on a private
//! headless [`Renderer`] whose per-media decoder cache makes ascending
//! same-media targets roll forward instead of seeking.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, unbounded};
use cutlass_decoder::{AudioPeaks, audio_peaks_per_second};
use cutlass_models::{MediaSource, Project, Rational, RationalTime, TimeRange, TrackKind};
use cutlass_render::Renderer;
use slint::{Image, ModelRc, Rgba8Pixel, SharedPixelBuffer, VecModel};
use tracing::{error, info};

use crate::interaction::InteractionGate;
use crate::{StripBackend, StripTile};

/// While the interaction gate is closed, poll for new requests at this
/// cadence instead of decoding tiles — the backlog waits, registrations
/// still apply immediately.
const GATE_POLL: Duration = Duration::from_millis(100);

/// Filmstrip tile width range on screen: `[MIN, 2·MIN)` px. The power-of-two
/// time interval per tile is the smallest one at least `MIN` px wide.
const FRAME_TILE_MIN_PX: f64 = 64.0;
/// Waveform tiles are wider (bars need room; fewer images per clip).
const WAVE_TILE_MIN_PX: f64 = 128.0;

/// Decode box for filmstrip frames (2× a ~55px lane for hidpi; tiles crop
/// with `image-fit: cover`, so one decode size serves every aspect).
const FRAME_MAX_W: u32 = 256;
const FRAME_MAX_H: u32 = 128;

/// Waveform tile raster size. Rendered once per (tile, zoom-bucket) and
/// stretched to the on-screen tile width (≤ 2× — bars stay legible).
const WAVE_IMG_W: usize = 256;
const WAVE_IMG_H: usize = 112;
/// Bar pitch in the waveform raster (3px bar + 1px gap).
const WAVE_BAR_PITCH: usize = 4;
/// Waveform bars over the lane-colored clip card.
const WAVE_BAR_RGBA: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xD8];

/// Peak-file resolution: one peak per 10ms of audio, computed once per media.
const PEAKS_PER_SECOND: f64 = 100.0;
/// Hard cap on a peak file (≈ 5.5h at 100/s, 8 MB of f32).
const MAX_PEAKS: usize = 2_000_000;

/// Finest tile interval: 2⁻⁶ s. Keeps `interval_us` exact (15625 µs) so grid
/// keys never drift; below this tiles would repeat frames anyway.
const MIN_K: i32 = -6;
/// Coarsest interval (2²⁶ s ≈ 2 years — effectively "one tile").
const MAX_K: i32 = 26;

/// Upper bound on tiles returned per resolver call; the visible window plus
/// margin stays far below this, it only guards degenerate inputs.
const MAX_TILES: i64 = 256;

/// `ClipView` quantizes its visible window to buckets of this many px before
/// calling the resolvers, so scrolling only re-resolves every 256px.
const VIEW_BUCKET_PX: f64 = 256.0;

/// UI-side image cache caps (per cache). At ~116 KB per filmstrip frame this
/// bounds GPU/CPU image memory near 120 MB worst case.
const CACHE_CAP: usize = 1024;

// ---------------------------------------------------------------------------
// Tile planning (pure, unit-tested)
// ---------------------------------------------------------------------------

/// A planned tile range: indices `i0..=i1` on the power-of-two media-time
/// grid, plus the mapping back to clip-local pixels.
#[derive(Debug, Clone, PartialEq)]
struct TileGrid {
    /// Tile interval exponent: each tile covers `2^k` seconds of source.
    k: i32,
    /// Exact microseconds per tile (`2^k * 1e6`; exact for `k ≥ MIN_K`).
    interval_us: i64,
    i0: i64,
    i1: i64,
    /// Screen px per second of source at the current zoom.
    px_per_s: f64,
    /// Media time (seconds) at the clip's left edge.
    origin_s: f64,
}

impl TileGrid {
    fn interval_s(&self) -> f64 {
        self.interval_us as f64 / 1e6
    }

    /// Clip-local x of tile `i`'s left edge (can be negative for the first,
    /// partially hidden tile; the clip clips it).
    fn x_px(&self, i: i64) -> f64 {
        (i as f64 * self.interval_s() - self.origin_s) * self.px_per_s
    }

    fn width_px(&self) -> f64 {
        self.interval_s() * self.px_per_s
    }

    fn key_us(&self, i: i64) -> i64 {
        i * self.interval_us
    }
}

/// Smallest `k` such that a `2^k`-second tile is at least `min_px` wide.
fn choose_k(px_per_s: f64, min_px: f64) -> i32 {
    let k = (min_px / px_per_s).log2().ceil();
    if k.is_nan() {
        return MAX_K;
    }
    (k as i32).clamp(MIN_K, MAX_K)
}

fn interval_us_for(k: i32) -> i64 {
    if k >= 0 {
        1_000_000_i64 << k
    } else {
        1_000_000_i64 >> (-k)
    }
}

/// Plan the tiles covering the visible slice of a clip.
///
/// * `source_in_s` — media time at the clip's left edge.
/// * `duration_ticks` — clip length in sequence ticks; with `zoom` (px/tick)
///   this bounds the clip-local pixel range.
/// * `fps_*` — sequence tick rate (ticks per second = num/den).
/// * `speed` — clip playback rate (M1): a 2× clip squeezes two source
///   seconds into each timeline second, so px-per-source-second halves.
///   Degenerate values fall back to 1×.
/// * `from_bucket..=to_bucket` — visible window in `VIEW_BUCKET_PX` units of
///   clip-local x (from `ClipView`; may extend past the clip).
///
/// Returns `None` when nothing is visible or the inputs are degenerate.
#[allow(clippy::too_many_arguments)] // mirrors the positional Slint callback
fn plan_tiles(
    source_in_s: f64,
    duration_ticks: i32,
    fps_num: i32,
    fps_den: i32,
    speed: f64,
    zoom: f64,
    from_bucket: i32,
    to_bucket: i32,
    min_tile_px: f64,
) -> Option<TileGrid> {
    if duration_ticks <= 0 || fps_num <= 0 || fps_den <= 0 || zoom <= 0.0 || zoom.is_nan() {
        return None;
    }
    let speed = if speed.is_finite() && speed > 0.0 {
        speed
    } else {
        1.0
    };
    let ticks_per_s = f64::from(fps_num) / f64::from(fps_den);
    let px_per_s = zoom * ticks_per_s / speed;
    if px_per_s <= 0.0 || !px_per_s.is_finite() {
        return None;
    }

    let k = choose_k(px_per_s, min_tile_px);
    let interval_us = interval_us_for(k);
    let interval_s = interval_us as f64 / 1e6;
    let tile_px = interval_s * px_per_s;

    // Visible clip-local px range, padded by one tile on each side, clamped
    // to the clip's extent.
    let clip_w_px = f64::from(duration_ticks) * zoom;
    let lo = (f64::from(from_bucket) * VIEW_BUCKET_PX - tile_px).max(0.0);
    let hi = ((f64::from(to_bucket) + 1.0) * VIEW_BUCKET_PX + tile_px).min(clip_w_px);
    if hi <= lo {
        return None;
    }

    let origin_s = source_in_s.max(0.0);
    let t_lo = origin_s + lo / px_per_s;
    let t_hi = origin_s + hi / px_per_s;
    let i0 = (t_lo / interval_s).floor().max(0.0) as i64;
    let mut i1 = (t_hi / interval_s).floor() as i64;
    i1 = i1.max(i0).min(i0 + MAX_TILES - 1);

    Some(TileGrid {
        k,
        interval_us,
        i0,
        i1,
        px_per_s,
        origin_s,
    })
}

// ---------------------------------------------------------------------------
// UI-thread tile registries
// ---------------------------------------------------------------------------

struct CacheEntry {
    image: Image,
    last_used: u64,
}

/// Memory attributed to one UI-thread strip image registry.
///
/// The byte count estimates an RGBA8 backing store from each image's intrinsic
/// pixel dimensions. It does not include map, renderer, or GPU overhead.
#[allow(dead_code)] // Consumed by follow-up settings/agent cache controls.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct StripImageCacheStats {
    pub(crate) entry_count: usize,
    pub(crate) estimated_rgba_bytes: u64,
}

/// Separate snapshots for the filmstrip-frame and waveform-tile registries.
#[allow(dead_code)] // Consumed by follow-up settings/agent cache controls.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct StripCacheStats {
    pub(crate) filmstrips: StripImageCacheStats,
    pub(crate) waves: StripImageCacheStats,
}

thread_local! {
    /// Decoded filmstrip frames, keyed by (media id, media time µs on the
    /// power-of-two grid). Frame images are zoom-independent.
    static FILMSTRIPS: RefCell<HashMap<(u64, i64), CacheEntry>> = RefCell::new(HashMap::new());
    /// Rasterized waveform tiles, keyed by (media id, media time µs, k) —
    /// the raster's px-per-second depends on the zoom bucket `k`.
    static WAVES: RefCell<HashMap<(u64, i64, i32), CacheEntry>> = RefCell::new(HashMap::new());
    /// Keys already sent to the worker and not yet delivered. Guarantees a
    /// key is requested at most once, no matter how many clips share media
    /// or how often the resolvers re-run.
    static PENDING_FRAMES: RefCell<HashSet<(u64, i64)>> = RefCell::new(HashSet::new());
    static PENDING_WAVES: RefCell<HashSet<(u64, i64, i32)>> = RefCell::new(HashSet::new());
    /// Monotonic touch counter for LRU eviction.
    static USE_TICK: Cell<u64> = const { Cell::new(0) };
}

fn estimated_rgba_bytes(image: &Image) -> u64 {
    let size = image.size();
    u64::from(size.width)
        .saturating_mul(u64::from(size.height))
        .saturating_mul(4)
}

fn image_cache_stats<'a>(images: impl Iterator<Item = &'a Image>) -> StripImageCacheStats {
    images.fold(StripImageCacheStats::default(), |mut stats, image| {
        stats.entry_count = stats.entry_count.saturating_add(1);
        stats.estimated_rgba_bytes = stats
            .estimated_rgba_bytes
            .saturating_add(estimated_rgba_bytes(image));
        stats
    })
}

/// Snapshot both strip image caches (Slint UI thread only).
#[allow(dead_code)] // Consumed by follow-up settings/agent cache controls.
pub(crate) fn cache_stats() -> StripCacheStats {
    let filmstrips = FILMSTRIPS.with(|cache| {
        let cache = cache.borrow();
        image_cache_stats(cache.values().map(|entry| &entry.image))
    });
    let waves = WAVES.with(|cache| {
        let cache = cache.borrow();
        image_cache_stats(cache.values().map(|entry| &entry.image))
    });
    StripCacheStats { filmstrips, waves }
}

/// Clear all UI-thread strip caches and return their pre-clear image stats.
///
/// Pending keys are cleared with the images so in-flight requests cannot leave
/// future misses permanently suppressed. Worker media maps, peak files, and
/// renderers are deliberately retained.
#[allow(dead_code)] // Consumed by follow-up settings/agent cache controls.
pub(crate) fn clear_all_caches() -> StripCacheStats {
    let filmstrips = clear_filmstrips();
    let waves = clear_waveforms();
    StripCacheStats { filmstrips, waves }
}

/// Clear only filmstrip images and their pending-request suppression.
#[allow(dead_code)] // Consumed by follow-up settings/agent cache controls.
pub(crate) fn clear_filmstrips() -> StripImageCacheStats {
    let filmstrips = FILMSTRIPS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let stats = image_cache_stats(cache.values().map(|entry| &entry.image));
        cache.clear();
        stats
    });
    PENDING_FRAMES.with(|pending| pending.borrow_mut().clear());
    reset_use_tick_if_empty();
    filmstrips
}

/// Clear only waveform images and their pending-request suppression.
#[allow(dead_code)] // Consumed by follow-up settings/agent cache controls.
pub(crate) fn clear_waveforms() -> StripImageCacheStats {
    let waves = WAVES.with(|cache| {
        let mut cache = cache.borrow_mut();
        let stats = image_cache_stats(cache.values().map(|entry| &entry.image));
        cache.clear();
        stats
    });
    PENDING_WAVES.with(|pending| pending.borrow_mut().clear());
    reset_use_tick_if_empty();
    waves
}

fn reset_use_tick_if_empty() {
    let empty = FILMSTRIPS.with(|cache| cache.borrow().is_empty())
        && WAVES.with(|cache| cache.borrow().is_empty());
    if empty {
        USE_TICK.with(|tick| tick.set(0));
    }
}

fn next_use_tick() -> u64 {
    USE_TICK.with(|t| {
        let v = t.get() + 1;
        t.set(v);
        v
    })
}

/// Drop the least-recently-used entries once a cache exceeds its cap.
/// Visible tiles are touched on every resolve, so they always survive.
fn evict_lru<K: Clone + Eq + Hash>(map: &mut HashMap<K, CacheEntry>) {
    if map.len() <= CACHE_CAP {
        return;
    }
    let mut ages: Vec<(u64, K)> = map
        .iter()
        .map(|(key, entry)| (entry.last_used, key.clone()))
        .collect();
    ages.sort_unstable_by_key(|(age, _)| *age);
    let target = CACHE_CAP * 3 / 4;
    for (_, key) in ages.into_iter().take(map.len() - target) {
        map.remove(&key);
    }
}

fn empty_tiles() -> ModelRc<StripTile> {
    ModelRc::from(Rc::new(VecModel::from(Vec::<StripTile>::new())))
}

/// Resolve the visible filmstrip tiles for a video clip (UI thread; pure
/// lookups — misses are queued on the strip worker and come back via a
/// `generation` bump).
#[allow(clippy::too_many_arguments)] // mirrors the positional Slint callback
pub fn filmstrip_tiles(
    handle: &StripHandle,
    media_id: &str,
    source_in_s: f32,
    duration_ticks: i32,
    fps_num: i32,
    fps_den: i32,
    speed: f32,
    zoom: f32,
    from_bucket: i32,
    to_bucket: i32,
) -> ModelRc<StripTile> {
    let Ok(media) = media_id.parse::<u64>() else {
        return empty_tiles();
    };
    let Some(grid) = plan_tiles(
        f64::from(source_in_s),
        duration_ticks,
        fps_num,
        fps_den,
        f64::from(speed),
        f64::from(zoom),
        from_bucket,
        to_bucket,
        FRAME_TILE_MIN_PX,
    ) else {
        return empty_tiles();
    };

    let tick = next_use_tick();
    let mut tiles = Vec::with_capacity((grid.i1 - grid.i0 + 1) as usize);
    let mut missing: Vec<i64> = Vec::new();

    FILMSTRIPS.with(|cache| {
        let mut cache = cache.borrow_mut();
        for i in grid.i0..=grid.i1 {
            let key_us = grid.key_us(i);
            let image = match cache.get_mut(&(media, key_us)) {
                Some(entry) => {
                    entry.last_used = tick;
                    entry.image.clone()
                }
                None => {
                    missing.push(key_us);
                    Image::default()
                }
            };
            tiles.push(StripTile {
                x: grid.x_px(i) as f32,
                width: grid.width_px() as f32,
                image,
            });
        }
    });

    let to_request = PENDING_FRAMES.with(|pending| {
        let mut pending = pending.borrow_mut();
        missing.retain(|&t_us| pending.insert((media, t_us)));
        missing
    });
    if !to_request.is_empty() {
        handle.request_filmstrip(media, to_request);
    }

    ModelRc::from(Rc::new(VecModel::from(tiles)))
}

/// Resolve the visible waveform tiles for an audio clip (UI thread).
#[allow(clippy::too_many_arguments)] // mirrors the positional Slint callback
pub fn waveform_tiles(
    handle: &StripHandle,
    media_id: &str,
    source_in_s: f32,
    duration_ticks: i32,
    fps_num: i32,
    fps_den: i32,
    speed: f32,
    zoom: f32,
    from_bucket: i32,
    to_bucket: i32,
) -> ModelRc<StripTile> {
    let Ok(media) = media_id.parse::<u64>() else {
        return empty_tiles();
    };
    let Some(grid) = plan_tiles(
        f64::from(source_in_s),
        duration_ticks,
        fps_num,
        fps_den,
        f64::from(speed),
        f64::from(zoom),
        from_bucket,
        to_bucket,
        WAVE_TILE_MIN_PX,
    ) else {
        return empty_tiles();
    };

    let tick = next_use_tick();
    let mut tiles = Vec::with_capacity((grid.i1 - grid.i0 + 1) as usize);
    let mut missing: Vec<i64> = Vec::new();

    WAVES.with(|cache| {
        let mut cache = cache.borrow_mut();
        for i in grid.i0..=grid.i1 {
            let key_us = grid.key_us(i);
            let image = match cache.get_mut(&(media, key_us, grid.k)) {
                Some(entry) => {
                    entry.last_used = tick;
                    entry.image.clone()
                }
                None => {
                    missing.push(key_us);
                    Image::default()
                }
            };
            tiles.push(StripTile {
                x: grid.x_px(i) as f32,
                width: grid.width_px() as f32,
                image,
            });
        }
    });

    let to_request = PENDING_WAVES.with(|pending| {
        let mut pending = pending.borrow_mut();
        missing.retain(|&t_us| pending.insert((media, t_us, grid.k)));
        missing
    });
    if !to_request.is_empty() {
        handle.request_waveform(media, grid.k, to_request);
    }

    ModelRc::from(Rc::new(VecModel::from(tiles)))
}

// ---------------------------------------------------------------------------
// Strip worker (decode + raster off the UI thread)
// ---------------------------------------------------------------------------

enum StripMsg {
    /// Media imported: remember its path (requests only carry the id).
    Register { media_id: u64, path: PathBuf },
    /// A preview proxy landed for `media_id`: decode *future filmstrip
    /// frames* from this smaller short-GOP file instead of the original.
    /// Waveforms are untouched — proxies carry no audio.
    Proxy { media_id: u64, path: PathBuf },
    /// Proxy cache storage moved: return every future filmstrip decode to its
    /// registered original and discard scratch projects/decoder cursors that
    /// may still hold an old proxy path. Waveform state remains warm.
    ClearProxies,
    /// Decode the frames at `times_us` (media time, grid-aligned).
    Filmstrip { media_id: u64, times_us: Vec<i64> },
    /// Rasterize the waveform tiles starting at `times_us`, each spanning
    /// `2^k` seconds.
    Waveform {
        media_id: u64,
        k: i32,
        times_us: Vec<i64>,
    },
}

/// Cheap, cloneable sender to the strip thread (UI thread requests tiles;
/// the preview worker registers imports).
#[derive(Clone)]
pub struct StripHandle {
    tx: Sender<StripMsg>,
}

impl StripHandle {
    pub fn register_media(&self, media_id: u64, path: PathBuf) {
        let _ = self.tx.send(StripMsg::Register { media_id, path });
    }

    /// Point future filmstrip decodes for `media_id` at its preview proxy.
    pub fn register_proxy(&self, media_id: u64, path: PathBuf) {
        let _ = self.tx.send(StripMsg::Proxy { media_id, path });
    }

    /// Remove every preview-proxy substitution and its scratch decode state.
    ///
    /// Original media registrations and waveform peak files are preserved.
    pub fn clear_proxies(&self) {
        let _ = self.tx.send(StripMsg::ClearProxies);
    }

    fn request_filmstrip(&self, media_id: u64, times_us: Vec<i64>) {
        let _ = self.tx.send(StripMsg::Filmstrip { media_id, times_us });
    }

    fn request_waveform(&self, media_id: u64, k: i32, times_us: Vec<i64>) {
        let _ = self.tx.send(StripMsg::Waveform {
            media_id,
            k,
            times_us,
        });
    }
}

pub struct StripWorker {
    handle: StripHandle,
    _join: JoinHandle<()>,
}

impl StripWorker {
    pub fn spawn(
        backend: slint::Weak<StripBackend<'static>>,
        gate: Arc<InteractionGate>,
    ) -> Result<Self, String> {
        let (tx, rx) = unbounded::<StripMsg>();
        let join = std::thread::Builder::new()
            .name("cutlass-strips".into())
            .spawn(move || worker_loop(&rx, &backend, &gate))
            .map_err(|e| e.to_string())?;

        Ok(Self {
            handle: StripHandle { tx },
            _join: join,
        })
    }

    pub fn handle(&self) -> StripHandle {
        self.handle.clone()
    }
}

/// A media source wrapped as a full-length scratch clip, ready to fit-render
/// any media time — built once per media on first filmstrip demand.
struct ScratchClip {
    project: Project,
    rate: Rational,
    duration_frames: i64,
}

#[derive(Default)]
struct WorkerState {
    paths: HashMap<u64, PathBuf>,
    /// Preview-proxy overrides for *frame* decode: when present, the scratch
    /// clip opens this file instead of `paths[id]`. Audio peaks always read
    /// `paths` — proxies are video-only.
    proxies: HashMap<u64, PathBuf>,
    /// Private renderer for frame tiles (own GPU queue + per-media decoder
    /// cache, so ascending same-media targets roll forward, never contending
    /// with the preview renderer). Lazy: brought up on first frame demand.
    renderer: Option<Renderer>,
    /// Scratch single-clip projects, one per media on first demand.
    scratch: HashMap<u64, ScratchClip>,
    /// Media whose probe/renderer failed — don't retry on every tile.
    frames_failed: HashSet<u64>,
    /// Peak files, computed once per media on first waveform demand.
    peaks: HashMap<u64, AudioPeaks>,
    /// Media whose peak decode failed — don't retry on every tile.
    peaks_failed: HashSet<u64>,
}

/// Newest-first work loop: registrations apply immediately, tile work is
/// processed LIFO so the zoom level / viewport the user is looking at *now*
/// fills in before stale requests from half a second ago. While the
/// interaction gate reports scrub/playback the backlog holds — tile decode
/// and composite would contend with the preview for the same decode engine
/// and iGPU (field logs: ~100 ms composites for 228×128 tiles under load).
fn worker_loop(
    rx: &Receiver<StripMsg>,
    backend: &slint::Weak<StripBackend<'static>>,
    gate: &InteractionGate,
) {
    let mut state = WorkerState::default();
    let mut backlog: Vec<StripMsg> = Vec::new();

    loop {
        if backlog.is_empty() {
            match rx.recv() {
                Ok(msg) => triage(msg, &mut state, &mut backlog),
                Err(_) => return,
            }
        }
        while let Ok(msg) = rx.try_recv() {
            triage(msg, &mut state, &mut backlog);
        }
        // Defer (never drop) tile work while the user scrubs or plays:
        // the UI-side pending sets stay valid because every deferred
        // request is still eventually processed and delivered.
        if gate.busy() {
            match rx.recv_timeout(GATE_POLL) {
                Ok(msg) => triage(msg, &mut state, &mut backlog),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return,
            }
            continue;
        }
        if let Some(work) = backlog.pop() {
            match work {
                StripMsg::Filmstrip { media_id, times_us } => {
                    process_filmstrip(&mut state, media_id, &times_us, backend);
                }
                StripMsg::Waveform {
                    media_id,
                    k,
                    times_us,
                } => {
                    process_waveform(&mut state, media_id, k, &times_us, backend);
                }
                StripMsg::Register { .. } | StripMsg::Proxy { .. } | StripMsg::ClearProxies => {
                    unreachable!("registrations apply in triage")
                }
            }
        }
    }
}

fn triage(msg: StripMsg, state: &mut WorkerState, backlog: &mut Vec<StripMsg>) {
    match msg {
        StripMsg::Register { media_id, path } => {
            info!(media_id, path = %path.display(), "strip worker registered media");
            // Re-registration (relink) invalidates the cached scratch clip
            // and failure latches so the new file gets a fresh chance — and
            // the old file's proxy, which no longer matches the content.
            if state.paths.get(&media_id) != Some(&path) {
                state.proxies.remove(&media_id);
                state.scratch.remove(&media_id);
                state.peaks.remove(&media_id);
                state.frames_failed.remove(&media_id);
                state.peaks_failed.remove(&media_id);
            }
            state.paths.insert(media_id, path);
        }
        StripMsg::Proxy { media_id, path } => {
            info!(media_id, proxy = %path.display(), "strip worker switched to proxy");
            // Tiles already delivered stay (same content); the dropped
            // scratch clip makes every *future* decode open the proxy.
            if state.proxies.get(&media_id) != Some(&path) {
                state.scratch.remove(&media_id);
                state.frames_failed.remove(&media_id);
                state.proxies.insert(media_id, path);
            }
        }
        StripMsg::ClearProxies => clear_proxy_substitutions(state),
        work => backlog.push(work),
    }
}

/// Return filmstrip decoding to registered originals after proxy storage moves.
///
/// Scratch projects and the private renderer are both discarded: either can
/// retain decoder state opened on a proxy path. Audio always reads `paths`, so
/// its peak files and failure latch deliberately survive this reset.
fn clear_proxy_substitutions(state: &mut WorkerState) {
    let proxy_count = state.proxies.len();
    state.proxies.clear();
    state.renderer = None;
    state.scratch.clear();
    state.frames_failed.clear();
    info!(
        proxy_count,
        "strip worker cleared proxy substitutions and scratch decode state"
    );
}

fn process_filmstrip(
    state: &mut WorkerState,
    media_id: u64,
    times_us: &[i64],
    backend: &slint::Weak<StripBackend<'static>>,
) {
    // Ascending targets let the decoder roll forward through one GOP pass
    // instead of ping-ponging seeks (requests arrive LIFO-jumbled).
    let mut order: Vec<i64> = times_us.to_vec();
    order.sort_unstable();

    let mut results: Vec<(i64, Option<RgbaImage>)> = Vec::with_capacity(order.len());
    for &t_us in &order {
        let image = render_strip_frame(state, media_id, t_us);
        results.push((t_us, image));
    }

    // Undelivered targets become empty entries: the pending set clears and
    // the negative cache entry stops an endless re-request loop.
    deliver_frames(media_id, results, backend);
}

/// Fit-render the media's frame at `t_us` through its scratch clip — built
/// over the preview proxy when one has registered (same content, far
/// cheaper decode). `None` (a placeholder tile) when the media can't be
/// probed or rendered.
fn render_strip_frame(state: &mut WorkerState, media_id: u64, t_us: i64) -> Option<RgbaImage> {
    if state.frames_failed.contains(&media_id) {
        return None;
    }
    if !state.scratch.contains_key(&media_id) {
        let Some(path) = state
            .proxies
            .get(&media_id)
            .or_else(|| state.paths.get(&media_id))
        else {
            error!(media_id, "filmstrip request for unregistered media");
            state.frames_failed.insert(media_id);
            return None;
        };
        match scratch_clip(path) {
            Ok(scratch) => {
                state.scratch.insert(media_id, scratch);
            }
            Err(e) => {
                error!(media_id, "filmstrip source open failed: {e}");
                state.frames_failed.insert(media_id);
                return None;
            }
        }
    }
    let renderer = match &mut state.renderer {
        Some(renderer) => renderer,
        None => match Renderer::new_headless() {
            Ok(renderer) => state.renderer.insert(renderer),
            Err(e) => {
                error!("strip renderer init failed: {e}");
                state.frames_failed.insert(media_id);
                return None;
            }
        },
    };
    let scratch = &state.scratch[&media_id];

    let max_frame = (scratch.duration_frames - 1).max(0);
    let seconds = t_us as f64 / 1e6;
    let frame = ((seconds * f64::from(scratch.rate.num) / f64::from(scratch.rate.den)).floor()
        as i64)
        .clamp(0, max_frame);
    match renderer.render_frame_fit(
        &scratch.project,
        RationalTime::new(frame, scratch.rate),
        FRAME_MAX_W,
        FRAME_MAX_H,
    ) {
        Ok(image) => Some(RgbaImage {
            width: image.width,
            height: image.height,
            rgba: image.pixels,
        }),
        Err(e) => {
            error!(media_id, frame, "filmstrip render failed: {e}");
            None
        }
    }
}

/// Probe `path` and wrap it as a full-length single-clip project (the mobile
/// thumbnailer pattern) — stills get one cached frame across every slot.
fn scratch_clip(path: &std::path::Path) -> Result<ScratchClip, String> {
    let probe = cutlass_decoder::probe(path).map_err(|e| e.to_string())?;
    let source = if probe.is_image {
        MediaSource::image(path, probe.width, probe.height)
    } else {
        MediaSource::new(
            path,
            probe.width,
            probe.height,
            probe.frame_rate,
            probe.frame_count.max(1),
            probe.has_audio,
        )
    };
    let rate = source.frame_rate;
    let frames = source.duration.value.max(1);
    let mut project = Project::new("strip", rate);
    let media = project.add_media(source);
    let track = project.add_track(TrackKind::Video, "Media");
    project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(0, frames, rate),
            RationalTime::new(0, rate),
        )
        .map_err(|e| e.to_string())?;
    Ok(ScratchClip {
        project,
        rate,
        duration_frames: frames,
    })
}

fn process_waveform(
    state: &mut WorkerState,
    media_id: u64,
    k: i32,
    times_us: &[i64],
    backend: &slint::Weak<StripBackend<'static>>,
) {
    let peaks = ensure_peaks(state, media_id);

    let span_s = interval_us_for(k) as f64 / 1e6;
    let tiles: Vec<(i64, Option<RgbaImage>)> = times_us
        .iter()
        .map(|&t_us| {
            let image = peaks.map(|p| RgbaImage {
                width: WAVE_IMG_W as u32,
                height: WAVE_IMG_H as u32,
                rgba: render_wave_tile(
                    &p.peaks,
                    p.per_second,
                    t_us as f64 / 1e6,
                    span_s,
                    WAVE_IMG_W,
                    WAVE_IMG_H,
                ),
            });
            (t_us, image)
        })
        .collect();

    deliver_waves(media_id, k, tiles, backend);
}

/// The peak file for `media_id`, computing it on first demand (the long
/// full-file decode happens exactly once per media).
fn ensure_peaks(state: &mut WorkerState, media_id: u64) -> Option<&AudioPeaks> {
    if !state.peaks.contains_key(&media_id) && !state.peaks_failed.contains(&media_id) {
        let Some(path) = state.paths.get(&media_id) else {
            error!(media_id, "waveform request for unregistered media");
            state.peaks_failed.insert(media_id);
            return None;
        };
        match audio_peaks_per_second(path, PEAKS_PER_SECOND, MAX_PEAKS) {
            Ok(peaks) => {
                info!(
                    media_id,
                    peaks = peaks.peaks.len(),
                    per_second = peaks.per_second,
                    "computed audio peak file"
                );
                state.peaks.insert(media_id, peaks);
            }
            Err(e) => {
                error!(media_id, path = %path.display(), "audio peak decode failed: {e}");
                state.peaks_failed.insert(media_id);
            }
        }
    }
    state.peaks.get(&media_id)
}

/// `Send`-able RGBA payload (slint images must be built on the UI thread).
struct RgbaImage {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

impl RgbaImage {
    fn into_image(self) -> Image {
        let buffer =
            SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(&self.rgba, self.width, self.height);
        Image::from_rgba8(buffer)
    }
}

fn deliver_frames(
    media_id: u64,
    frames: Vec<(i64, Option<RgbaImage>)>,
    backend: &slint::Weak<StripBackend<'static>>,
) {
    if frames.is_empty() {
        return;
    }
    let backend = backend.clone();
    if let Err(e) = slint::invoke_from_event_loop(move || {
        let tick = next_use_tick();
        FILMSTRIPS.with(|cache| {
            let mut cache = cache.borrow_mut();
            for (t_us, image) in frames {
                PENDING_FRAMES.with(|p| p.borrow_mut().remove(&(media_id, t_us)));
                let image = image.map(RgbaImage::into_image).unwrap_or_default();
                cache.insert(
                    (media_id, t_us),
                    CacheEntry {
                        image,
                        last_used: tick,
                    },
                );
            }
            evict_lru(&mut cache);
        });
        bump_generation(&backend);
    }) {
        error!(media_id, "failed to deliver filmstrip frames to UI: {e}");
    }
}

fn deliver_waves(
    media_id: u64,
    k: i32,
    tiles: Vec<(i64, Option<RgbaImage>)>,
    backend: &slint::Weak<StripBackend<'static>>,
) {
    if tiles.is_empty() {
        return;
    }
    let backend = backend.clone();
    if let Err(e) = slint::invoke_from_event_loop(move || {
        let tick = next_use_tick();
        WAVES.with(|cache| {
            let mut cache = cache.borrow_mut();
            for (t_us, image) in tiles {
                PENDING_WAVES.with(|p| p.borrow_mut().remove(&(media_id, t_us, k)));
                let image = image.map(RgbaImage::into_image).unwrap_or_default();
                cache.insert(
                    (media_id, t_us, k),
                    CacheEntry {
                        image,
                        last_used: tick,
                    },
                );
            }
            evict_lru(&mut cache);
        });
        bump_generation(&backend);
    }) {
        error!(media_id, "failed to deliver waveform tiles to UI: {e}");
    }
}

/// Nudge every visible tile model to re-resolve (they all read `generation`).
fn bump_generation(backend: &slint::Weak<StripBackend<'static>>) {
    if let Some(backend) = backend.upgrade() {
        backend.set_generation(backend.get_generation().wrapping_add(1));
    }
}

// ---------------------------------------------------------------------------
// Waveform raster
// ---------------------------------------------------------------------------

/// Draw mirrored peak bars for `[t0, t0 + span)` seconds onto a transparent
/// tile — the lane-colored clip card shows through, CapCut-style.
fn render_wave_tile(
    peaks: &[f32],
    per_second: f64,
    t0_s: f64,
    span_s: f64,
    width: usize,
    height: usize,
) -> Vec<u8> {
    let mut rgba = vec![0u8; width * height * 4];
    if peaks.is_empty() || per_second <= 0.0 || span_s <= 0.0 {
        return rgba;
    }

    let bars = width / WAVE_BAR_PITCH;
    let bar_w = WAVE_BAR_PITCH - 1;
    let mid = height / 2;
    let max_half = (height / 2).saturating_sub(2).max(1);
    let bar_span_s = span_s / bars as f64;

    for bar in 0..bars {
        let start_s = t0_s + bar as f64 * bar_span_s;
        let lo = (start_s * per_second).floor().max(0.0) as usize;
        if lo >= peaks.len() {
            // Tile extends past the audio — happens when the clip's tick
            // duration outlives the decoded peaks (container vs stream
            // duration). Must check before the clamp below or it panics.
            break;
        }
        let hi = (((start_s + bar_span_s) * per_second).ceil() as usize).clamp(lo + 1, peaks.len());
        let peak = peaks[lo..hi].iter().copied().fold(0.0f32, f32::max);

        // Quiet audio keeps a visible 1px center line, like CapCut.
        let half =
            ((f64::from(peak.clamp(0.0, 1.0)) * max_half as f64) as usize).clamp(1, max_half);
        let x0 = bar * WAVE_BAR_PITCH;
        for y in mid.saturating_sub(half)..(mid + half).min(height) {
            let row = y * width * 4;
            for x in x0..(x0 + bar_w).min(width) {
                let px = row + x * 4;
                rgba[px..px + 4].copy_from_slice(&WAVE_BAR_RGBA);
            }
        }
    }

    rgba
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StripCacheCleanup;

    impl Drop for StripCacheCleanup {
        fn drop(&mut self) {
            clear_all_caches();
        }
    }

    fn test_image(width: u32, height: u32) -> Image {
        let rgba = vec![0; width as usize * height as usize * 4];
        Image::from_rgba8(SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(
            &rgba, width, height,
        ))
    }

    #[test]
    fn strip_cache_stats_and_clear_are_exact_and_idempotent() {
        clear_all_caches();
        let _cleanup = StripCacheCleanup;

        FILMSTRIPS.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.insert(
                (1, 0),
                CacheEntry {
                    image: test_image(2, 3),
                    last_used: 1,
                },
            );
            cache.insert(
                (1, 1_000_000),
                CacheEntry {
                    image: test_image(4, 5),
                    last_used: 2,
                },
            );
        });
        WAVES.with(|cache| {
            cache.borrow_mut().insert(
                (2, 0, -1),
                CacheEntry {
                    image: test_image(7, 2),
                    last_used: 3,
                },
            );
        });

        let expected = StripCacheStats {
            filmstrips: StripImageCacheStats {
                entry_count: 2,
                estimated_rgba_bytes: 2 * 3 * 4 + 4 * 5 * 4,
            },
            waves: StripImageCacheStats {
                entry_count: 1,
                estimated_rgba_bytes: 7 * 2 * 4,
            },
        };
        assert_eq!(cache_stats(), expected);
        assert_eq!(clear_all_caches(), expected);
        assert_eq!(cache_stats(), StripCacheStats::default());
        assert_eq!(clear_all_caches(), StripCacheStats::default());
    }

    #[test]
    fn strip_proxy_clear_is_idempotent_and_preserves_original_and_waveform_state() {
        let media_id = 17;
        let original = PathBuf::from("/media/original.mov");
        let mut state = WorkerState::default();
        state.paths.insert(media_id, original.clone());
        state
            .proxies
            .insert(media_id, PathBuf::from("/old-proxies/proxy.mp4"));
        state.scratch.insert(
            media_id,
            ScratchClip {
                project: Project::new("scratch", Rational::FPS_30),
                rate: Rational::FPS_30,
                duration_frames: 30,
            },
        );
        state.frames_failed.insert(media_id);
        state.peaks.insert(
            media_id,
            AudioPeaks {
                per_second: 100.0,
                peaks: vec![0.25, 0.5],
            },
        );
        state.peaks_failed.insert(99);

        let (tx, rx) = unbounded();
        let handle = StripHandle { tx };
        let mut backlog = Vec::new();
        handle.clear_proxies();
        triage(rx.recv().unwrap(), &mut state, &mut backlog);

        assert_eq!(state.paths.get(&media_id), Some(&original));
        assert!(state.proxies.is_empty());
        assert!(state.scratch.is_empty());
        assert!(state.frames_failed.is_empty());
        assert!(state.renderer.is_none());
        assert_eq!(state.peaks[&media_id].peaks, vec![0.25, 0.5]);
        assert!(state.peaks_failed.contains(&99));
        assert!(backlog.is_empty());

        handle.clear_proxies();
        triage(rx.recv().unwrap(), &mut state, &mut backlog);
        assert_eq!(state.paths.get(&media_id), Some(&original));
        assert!(state.proxies.is_empty());
        assert!(state.scratch.is_empty());
        assert_eq!(state.peaks[&media_id].peaks, vec![0.25, 0.5]);
        assert!(state.peaks_failed.contains(&99));
        assert!(backlog.is_empty());
    }

    #[test]
    fn clearing_strip_caches_resets_pending_request_suppression() {
        clear_all_caches();
        let _cleanup = StripCacheCleanup;
        let frame_key = (41, 2_000_000);
        let wave_key = (42, 3_000_000, 1);

        PENDING_FRAMES.with(|pending| {
            let mut pending = pending.borrow_mut();
            assert!(pending.insert(frame_key));
            assert!(!pending.insert(frame_key), "duplicate frame is suppressed");
        });
        PENDING_WAVES.with(|pending| {
            let mut pending = pending.borrow_mut();
            assert!(pending.insert(wave_key));
            assert!(!pending.insert(wave_key), "duplicate wave is suppressed");
        });
        USE_TICK.with(|tick| tick.set(99));

        assert_eq!(clear_all_caches(), StripCacheStats::default());
        assert_eq!(USE_TICK.with(Cell::get), 0);
        PENDING_FRAMES.with(|pending| {
            assert!(
                pending.borrow_mut().insert(frame_key),
                "frame can be requested again after clear"
            );
        });
        PENDING_WAVES.with(|pending| {
            assert!(
                pending.borrow_mut().insert(wave_key),
                "wave can be requested again after clear"
            );
        });
    }

    #[test]
    fn strip_cache_kinds_can_be_cleared_independently() {
        clear_all_caches();
        let _cleanup = StripCacheCleanup;
        FILMSTRIPS.with(|cache| {
            cache.borrow_mut().insert(
                (1, 0),
                CacheEntry {
                    image: test_image(2, 2),
                    last_used: 1,
                },
            );
        });
        WAVES.with(|cache| {
            cache.borrow_mut().insert(
                (2, 0, 0),
                CacheEntry {
                    image: test_image(3, 2),
                    last_used: 2,
                },
            );
        });
        PENDING_FRAMES.with(|pending| {
            pending.borrow_mut().insert((1, 0));
        });
        PENDING_WAVES.with(|pending| {
            pending.borrow_mut().insert((2, 0, 0));
        });
        USE_TICK.with(|tick| tick.set(7));

        assert_eq!(
            clear_filmstrips(),
            StripImageCacheStats {
                entry_count: 1,
                estimated_rgba_bytes: 16,
            }
        );
        assert_eq!(cache_stats().filmstrips, StripImageCacheStats::default());
        assert_eq!(cache_stats().waves.entry_count, 1);
        assert!(PENDING_FRAMES.with(|pending| pending.borrow().is_empty()));
        assert_eq!(PENDING_WAVES.with(|pending| pending.borrow().len()), 1);
        assert_eq!(USE_TICK.with(Cell::get), 7);

        assert_eq!(clear_waveforms().entry_count, 1);
        assert!(PENDING_WAVES.with(|pending| pending.borrow().is_empty()));
        assert_eq!(USE_TICK.with(Cell::get), 0);
    }

    #[test]
    fn choose_k_picks_smallest_interval_at_least_min_px() {
        // 100 px/s: 1s tile = 100px ≥ 64, 0.5s = 50px < 64 → k = 0.
        assert_eq!(choose_k(100.0, 64.0), 0);
        // 600 px/s: 0.125s = 75px ≥ 64, 0.0625s = 37.5px → k = -3.
        assert_eq!(choose_k(600.0, 64.0), -3);
        // 1 px/s: need 64s → next power of two is 64 → k = 6.
        assert_eq!(choose_k(1.0, 64.0), 6);
        // Degenerate zooms clamp instead of exploding.
        assert_eq!(choose_k(1e-12, 64.0), MAX_K);
        assert_eq!(choose_k(1e12, 64.0), MIN_K);
    }

    #[test]
    fn interval_us_is_exact_for_all_allowed_k() {
        assert_eq!(interval_us_for(0), 1_000_000);
        assert_eq!(interval_us_for(2), 4_000_000);
        assert_eq!(interval_us_for(-2), 250_000);
        assert_eq!(interval_us_for(MIN_K), 15_625);
        // Exactness: interval reconstructs 2^k seconds with no remainder.
        for k in MIN_K..=6 {
            let us = interval_us_for(k);
            assert_eq!(us as f64 / 1e6, 2.0_f64.powi(k), "k={k}");
        }
    }

    #[test]
    fn coarser_grids_nest_inside_finer_ones() {
        // Every key on the k+1 grid is also on the k grid: zooming in reuses
        // every frame already decoded while zoomed out.
        for k in MIN_K..10 {
            assert_eq!(interval_us_for(k + 1) % interval_us_for(k), 0);
        }
    }

    #[test]
    fn plan_covers_the_visible_window() {
        // 30fps, zoom 2px/tick → 60 px/s → 1s tiles are 60px < 64 → k=1:
        // 2s tiles, 120px each.
        let grid = plan_tiles(0.0, 3000, 30, 1, 1.0, 2.0, 0, 3, 64.0).expect("grid");
        assert_eq!(grid.k, 1);
        assert_eq!(grid.i0, 0);
        // Window: 0..1024px (+1 tile pad) of a 6000px clip → ~9 tiles.
        assert!(grid.i1 >= 8 && grid.i1 <= 10, "i1 = {}", grid.i1);
        assert_eq!(grid.x_px(0), 0.0);
        assert!((grid.width_px() - 120.0).abs() < 1e-9);
    }

    #[test]
    fn plan_clamps_to_the_clip_extent() {
        // Visible window far past the clip's end → nothing.
        assert_eq!(plan_tiles(0.0, 100, 30, 1, 1.0, 1.0, 10, 20, 64.0), None);
        // Window before the clip (negative buckets) → nothing.
        assert_eq!(plan_tiles(0.0, 100, 30, 1, 1.0, 1.0, -20, -10, 64.0), None);
    }

    #[test]
    fn plan_anchors_tiles_to_media_time() {
        // A clip trimmed 3.5s into its source at k=1 (2s tiles): the first
        // visible tile is the grid tile containing 3.5s (i=1 → 2s), drawn
        // 1.5s-worth of px to the left of the clip edge.
        let grid = plan_tiles(3.5, 3000, 30, 1, 1.0, 2.0, 0, 3, 64.0).expect("grid");
        assert_eq!(grid.k, 1);
        assert_eq!(grid.i0, 1);
        assert_eq!(grid.key_us(grid.i0), 2_000_000);
        assert!((grid.x_px(1) - (2.0 - 3.5) * 60.0).abs() < 1e-9);
    }

    #[test]
    fn plan_rejects_degenerate_inputs() {
        assert_eq!(plan_tiles(0.0, 0, 30, 1, 1.0, 1.0, 0, 3, 64.0), None);
        assert_eq!(plan_tiles(0.0, 100, 0, 1, 1.0, 1.0, 0, 3, 64.0), None);
        assert_eq!(plan_tiles(0.0, 100, 30, 1, 1.0, 0.0, 0, 3, 64.0), None);
    }

    #[test]
    fn plan_scales_px_per_source_second_by_speed() {
        // Same zoom, 2× speed: each source second occupies half the px, so
        // the source window covered by the same clip width doubles.
        let normal = plan_tiles(0.0, 3000, 30, 1, 1.0, 2.0, 0, 3, 64.0).expect("grid");
        let double = plan_tiles(0.0, 3000, 30, 1, 2.0, 2.0, 0, 3, 64.0).expect("grid");
        assert!((normal.px_per_s - 60.0).abs() < 1e-9);
        assert!((double.px_per_s - 30.0).abs() < 1e-9);
        // Degenerate speeds fall back to 1×.
        let fallback = plan_tiles(0.0, 3000, 30, 1, 0.0, 2.0, 0, 3, 64.0).expect("grid");
        assert!((fallback.px_per_s - 60.0).abs() < 1e-9);
    }

    #[test]
    fn plan_caps_the_tile_count() {
        // Absurd window: tile count stays bounded.
        let grid =
            plan_tiles(0.0, i32::MAX, 1000, 1, 1.0, 100.0, 0, i32::MAX / 256, 64.0).expect("grid");
        assert!(grid.i1 - grid.i0 < MAX_TILES);
    }

    #[test]
    fn wave_tile_paints_mirrored_bars_on_transparent_ground() {
        let peaks = vec![1.0f32; 200]; // 2s of loud audio at 100/s
        let rgba = render_wave_tile(&peaks, 100.0, 0.0, 2.0, 64, 32);
        assert_eq!(rgba.len(), 64 * 32 * 4);

        let px = |x: usize, y: usize| {
            let i = (y * 64 + x) * 4;
            [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
        };
        // Bar column at center rows is painted; the gap column is not.
        assert_eq!(px(0, 16), WAVE_BAR_RGBA);
        assert_eq!(px(3, 16), [0, 0, 0, 0], "gap between bars stays clear");
        // Mirrored: symmetric rows above/below the midline both painted.
        assert_eq!(px(0, 4), px(0, 27));
        // Top margin rows stay clear (max_half leaves 2px headroom).
        assert_eq!(px(0, 0), [0, 0, 0, 0]);
    }

    #[test]
    fn wave_tile_silence_keeps_a_center_line() {
        let peaks = vec![0.0f32; 100];
        let rgba = render_wave_tile(&peaks, 100.0, 0.0, 1.0, 16, 32);
        let center = (16 * 16) * 4;
        assert_eq!(&rgba[center..center + 4], &WAVE_BAR_RGBA);
        let top = 16 * 4_usize;
        assert_eq!(&rgba[top..top + 4], &[0, 0, 0, 0]);
    }

    #[test]
    fn wave_tile_past_the_end_of_audio_does_not_panic() {
        // 115 peaks ≈ 1.15s of audio at 100/s, but the tile spans [1.0, 2.0)s:
        // later bars start past the last peak (a clip whose tick duration
        // outlives its decoded audio). Used to panic in clamp() with
        // `min > max. min = 119, max = 115`.
        let peaks = vec![0.5f32; 115];
        let rgba = render_wave_tile(&peaks, 100.0, 1.0, 1.0, 64, 32);
        assert_eq!(rgba.len(), 64 * 32 * 4);
        // Bars fully past the audio stay transparent.
        let last_bar_x = 60; // bar 15 of 16 covers [1.9375, 2.0)s
        let center = (16 * 64 + last_bar_x) * 4;
        assert_eq!(&rgba[center..center + 4], &[0, 0, 0, 0]);
    }

    #[test]
    fn wave_tile_with_no_peaks_is_fully_transparent() {
        let rgba = render_wave_tile(&[], 100.0, 0.0, 1.0, 16, 16);
        assert!(rgba.iter().all(|&b| b == 0));
    }

    #[test]
    fn scratch_clip_renders_fixture_frames() {
        // End-to-end decode spine over the checked-in fixture: probe →
        // scratch project → headless fit-render, exactly what the strip
        // worker does per tile. Skips when the fixture or a GPU is missing.
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../apps/cutlass-ios-macos/cutlass-ios-macos/Fixtures/demo1.mp4");
        if !fixture.exists() {
            return;
        }
        // The fixture is checked in, so existence alone isn't enough: platforms
        // with no native video decoder (e.g. Linux CI) can't open it, so skip
        // rather than unwrap an Unsupported error.
        let Ok(scratch) = scratch_clip(&fixture) else {
            return;
        };
        assert!(scratch.duration_frames > 0);
        let Ok(mut renderer) = Renderer::new_headless() else {
            return;
        };
        let image = renderer
            .render_frame_fit(
                &scratch.project,
                RationalTime::new(0, scratch.rate),
                FRAME_MAX_W,
                FRAME_MAX_H,
            )
            .expect("fit render");
        assert!(image.width > 0 && image.width <= FRAME_MAX_W);
        assert!(image.height > 0 && image.height <= FRAME_MAX_H);
        assert_eq!(
            image.pixels.len(),
            (image.width * image.height * 4) as usize
        );
    }

    #[test]
    fn lru_eviction_keeps_recently_used_entries() {
        let mut map: HashMap<u32, CacheEntry> = HashMap::new();
        for i in 0..(CACHE_CAP as u32 + 100) {
            map.insert(
                i,
                CacheEntry {
                    image: Image::default(),
                    last_used: u64::from(i),
                },
            );
        }
        evict_lru(&mut map);
        assert!(map.len() <= CACHE_CAP);
        // The newest entries survive, the oldest are gone.
        assert!(map.contains_key(&(CACHE_CAP as u32 + 99)));
        assert!(!map.contains_key(&0));
    }
}

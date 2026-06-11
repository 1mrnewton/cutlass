//! Timeline clip content: filmstrip frames for video clips, waveform tiles
//! for audio clips (Phase 8).
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

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::path::PathBuf;
use std::rc::Rc;
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender, unbounded};
use cutlass_decoder::{AudioPeaks, audio_peaks_per_second, video_strip};
use slint::{Image, ModelRc, Rgba8Pixel, SharedPixelBuffer, VecModel};
use tracing::{error, info};

use crate::{StripBackend, StripTile};

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
    zoom: f64,
    from_bucket: i32,
    to_bucket: i32,
    min_tile_px: f64,
) -> Option<TileGrid> {
    if duration_ticks <= 0 || fps_num <= 0 || fps_den <= 0 || zoom <= 0.0 || zoom.is_nan() {
        return None;
    }
    let ticks_per_s = f64::from(fps_num) / f64::from(fps_den);
    let px_per_s = zoom * ticks_per_s;
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

    fn request_filmstrip(&self, media_id: u64, times_us: Vec<i64>) {
        let _ = self.tx.send(StripMsg::Filmstrip { media_id, times_us });
    }

    fn request_waveform(&self, media_id: u64, k: i32, times_us: Vec<i64>) {
        let _ = self.tx.send(StripMsg::Waveform { media_id, k, times_us });
    }
}

pub struct StripWorker {
    handle: StripHandle,
    _join: JoinHandle<()>,
}

impl StripWorker {
    pub fn spawn(backend: slint::Weak<StripBackend<'static>>) -> Result<Self, String> {
        let (tx, rx) = unbounded::<StripMsg>();
        let join = std::thread::Builder::new()
            .name("cutlass-strips".into())
            .spawn(move || worker_loop(&rx, &backend))
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

struct WorkerState {
    paths: HashMap<u64, PathBuf>,
    /// Peak files, computed once per media on first waveform demand.
    peaks: HashMap<u64, AudioPeaks>,
    /// Media whose peak decode failed — don't retry on every tile.
    peaks_failed: HashSet<u64>,
}

/// Newest-first work loop: registrations apply immediately, tile work is
/// processed LIFO so the zoom level / viewport the user is looking at *now*
/// fills in before stale requests from half a second ago.
fn worker_loop(rx: &Receiver<StripMsg>, backend: &slint::Weak<StripBackend<'static>>) {
    let mut state = WorkerState {
        paths: HashMap::new(),
        peaks: HashMap::new(),
        peaks_failed: HashSet::new(),
    };
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
        if let Some(work) = backlog.pop() {
            match work {
                StripMsg::Filmstrip { media_id, times_us } => {
                    process_filmstrip(&state, media_id, &times_us, backend);
                }
                StripMsg::Waveform { media_id, k, times_us } => {
                    process_waveform(&mut state, media_id, k, &times_us, backend);
                }
                StripMsg::Register { .. } => unreachable!("registrations apply in triage"),
            }
        }
    }
}

fn triage(msg: StripMsg, state: &mut WorkerState, backlog: &mut Vec<StripMsg>) {
    match msg {
        StripMsg::Register { media_id, path } => {
            info!(media_id, path = %path.display(), "strip worker registered media");
            state.paths.insert(media_id, path);
        }
        work => backlog.push(work),
    }
}

fn process_filmstrip(
    state: &WorkerState,
    media_id: u64,
    times_us: &[i64],
    backend: &slint::Weak<StripBackend<'static>>,
) {
    let Some(path) = state.paths.get(&media_id) else {
        error!(media_id, "filmstrip request for unregistered media");
        deliver_frames(media_id, times_us.iter().map(|&t| (t, None)).collect(), backend);
        return;
    };

    let times_s: Vec<f64> = times_us.iter().map(|&us| us as f64 / 1e6).collect();
    let mut results: Vec<Option<RgbaImage>> = (0..times_us.len()).map(|_| None).collect();
    let outcome = video_strip(path, &times_s, FRAME_MAX_W, FRAME_MAX_H, &mut |i, img| {
        results[i] = Some(RgbaImage {
            width: img.width,
            height: img.height,
            rgba: img.rgba,
        });
    });
    if let Err(e) = outcome {
        error!(media_id, path = %path.display(), "filmstrip decode failed: {e}");
    }

    // Undelivered targets become empty entries: the pending set clears and
    // the negative cache entry stops an endless re-request loop.
    deliver_frames(
        media_id,
        times_us.iter().copied().zip(results).collect(),
        backend,
    );
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
        let buffer = SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(
            &self.rgba,
            self.width,
            self.height,
        );
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
                cache.insert((media_id, t_us), CacheEntry { image, last_used: tick });
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
                cache.insert((media_id, t_us, k), CacheEntry { image, last_used: tick });
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
        let hi = (((start_s + bar_span_s) * per_second).ceil() as usize).clamp(lo + 1, peaks.len());
        if lo >= peaks.len() {
            break; // tile extends past the audio (shouldn't happen mid-clip)
        }
        let peak = peaks[lo..hi].iter().copied().fold(0.0f32, f32::max);

        // Quiet audio keeps a visible 1px center line, like CapCut.
        let half = ((f64::from(peak.clamp(0.0, 1.0)) * max_half as f64) as usize)
            .clamp(1, max_half);
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
        // 30fps, zoom 2px/tick → 60 px/s → k=0 (1s tiles, 60px each... < 64).
        // Actually 60 < 64 → k=1: 2s tiles, 120px.
        let grid = plan_tiles(0.0, 3000, 30, 1, 2.0, 0, 3, 64.0).expect("grid");
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
        assert_eq!(plan_tiles(0.0, 100, 30, 1, 1.0, 10, 20, 64.0), None);
        // Window before the clip (negative buckets) → nothing.
        assert_eq!(plan_tiles(0.0, 100, 30, 1, 1.0, -20, -10, 64.0), None);
    }

    #[test]
    fn plan_anchors_tiles_to_media_time() {
        // A clip trimmed 3.5s into its source at k=1 (2s tiles): the first
        // visible tile is the grid tile containing 3.5s (i=1 → 2s), drawn
        // 1.5s-worth of px to the left of the clip edge.
        let grid = plan_tiles(3.5, 3000, 30, 1, 2.0, 0, 3, 64.0).expect("grid");
        assert_eq!(grid.k, 1);
        assert_eq!(grid.i0, 1);
        assert_eq!(grid.key_us(grid.i0), 2_000_000);
        assert!((grid.x_px(1) - (2.0 - 3.5) * 60.0).abs() < 1e-9);
    }

    #[test]
    fn plan_rejects_degenerate_inputs() {
        assert_eq!(plan_tiles(0.0, 0, 30, 1, 1.0, 0, 3, 64.0), None);
        assert_eq!(plan_tiles(0.0, 100, 0, 1, 1.0, 0, 3, 64.0), None);
        assert_eq!(plan_tiles(0.0, 100, 30, 1, 0.0, 0, 3, 64.0), None);
    }

    #[test]
    fn plan_caps_the_tile_count() {
        // Absurd window: tile count stays bounded.
        let grid = plan_tiles(0.0, i32::MAX, 1000, 1, 100.0, 0, i32::MAX / 256, 64.0)
            .expect("grid");
        assert!(grid.i1 - grid.i0 + 1 <= MAX_TILES);
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
    fn wave_tile_with_no_peaks_is_fully_transparent() {
        let rgba = render_wave_tile(&[], 100.0, 0.0, 1.0, 16, 16);
        assert!(rgba.iter().all(|&b| b == 0));
    }

    #[test]
    fn lru_eviction_keeps_recently_used_entries() {
        let mut map: HashMap<u32, CacheEntry> = HashMap::new();
        for i in 0..(CACHE_CAP as u32 + 100) {
            map.insert(i, CacheEntry { image: Image::default(), last_used: u64::from(i) });
        }
        evict_lru(&mut map);
        assert!(map.len() <= CACHE_CAP);
        // The newest entries survive, the oldest are gone.
        assert!(map.contains_key(&(CACHE_CAP as u32 + 99)));
        assert!(!map.contains_key(&0));
    }
}

//! Mobile FFI bridge for the Cutlass test apps.
//!
//! This is the thin native surface the iOS and Android scaffolds call to prove
//! the editor runs on a device's real GPU: it brings up a headless `wgpu` device
//! (Metal on iOS, Vulkan/GLES on Android), composites/renders a frame, and hands
//! back the RGBA pixels for the app to display.
//!
//! Two front doors, one core:
//! - **iOS** links the `staticlib` and calls the plain C ABI
//!   ([`cutlass_render_demo`] / [`cutlass_image_free`]).
//! - **Android** loads the `cdylib` and calls the JNI entry points
//!   (`CutlassNative.*`), which return ARGB_8888 ints ready for a `Bitmap`.
//!
//! Everything that touches media goes through [`cutlass_engine::Engine`], so the
//! on-device harness exercises the same render path as the desktop editor — and,
//! crucially, the engine decodes video through the platform-native
//! (FFmpeg-free) backend on mobile (see `cutlass-engine`'s `engine_native_decode`
//! build cfg). The only per-platform code is the marshalling in this crate.

use std::ffi::{CString, c_char};
use std::path::{Path, PathBuf};

use cutlass_compositor::{
    CompositeLayer, Compositor, CompositorConfig, GpuContext, LayerPlacement, RgbaImage,
};
use cutlass_engine::{ApplyOutcome, Command, Engine, EngineConfig, EditOutcome, RgbaFrame};
use cutlass_models::{Generator, Project, Rational, RationalTime, TimeRange, TrackKind};

/// Render the built-in demo scene at `width`×`height` straight through the
/// compositor (solid layers only — no decode, no fonts), so it renders
/// identically on every device and isolates "does this device's GPU pipeline
/// work" from codec/asset concerns. `None` on degenerate dims or GPU failure.
fn render_demo_rgba(width: u32, height: u32) -> Option<RgbaImage> {
    if width == 0 || height == 0 {
        return None;
    }

    let gpu = match GpuContext::new_headless_blocking() {
        Ok(gpu) => gpu,
        Err(e) => {
            log::error!("demo: GPU init failed: {e}");
            return None;
        }
    };
    let mut compositor = match Compositor::new(&gpu) {
        Ok(c) => c,
        Err(e) => {
            log::error!("demo: compositor init failed: {e}");
            return None;
        }
    };

    let config = CompositorConfig::new(width, height).with_background([18, 22, 30]);
    let (w, h) = (width as f32, height as f32);
    let layers = [
        // A large centered block — fills most of the canvas.
        CompositeLayer::solid(
            [235, 78, 58, 255],
            LayerPlacement {
                center: [w * 0.5, h * 0.5],
                size: [w * 0.62, h * 0.62],
                rotation: 0.0,
                opacity: 1.0,
            },
        ),
        // A teal square upper-left (proves placement + y-down).
        CompositeLayer::solid(
            [46, 196, 182, 255],
            LayerPlacement {
                center: [w * 0.3, h * 0.32],
                size: [w * 0.2, w * 0.2],
                rotation: 0.0,
                opacity: 1.0,
            },
        ),
        // A rotated, translucent amber square lower-right (proves rotation +
        // alpha blending).
        CompositeLayer::solid(
            [255, 196, 61, 235],
            LayerPlacement {
                center: [w * 0.7, h * 0.7],
                size: [w * 0.2, w * 0.2],
                rotation: 0.6,
                opacity: 1.0,
            },
        ),
    ];

    compositor.composite(&gpu, &config, &layers).ok()
}

/// Build a headless [`Engine`] over `project`, caching under `cache_dir`
/// (falling back to a temp directory when the app didn't supply one).
fn new_engine_in(project: Project, cache_dir: Option<PathBuf>) -> Option<Engine> {
    let cache_dir = cache_dir.unwrap_or_else(|| std::env::temp_dir().join("cutlass-mobile-cache"));
    let config = EngineConfig {
        cache_dir,
        ..EngineConfig::default()
    };
    match Engine::with_project(config, project) {
        Ok(engine) => Some(engine),
        Err(e) => {
            log::error!("engine init failed: {e}");
            None
        }
    }
}

/// Build a headless [`Engine`] over `project`, caching to a temp directory.
fn new_engine(project: Project) -> Option<Engine> {
    new_engine_in(project, None)
}

/// Decode + composite the first frame of the video at `path` through the engine.
/// `None` on probe/render failure or if the GPU is unavailable.
fn render_file_frame_rgba(path: &Path) -> Option<RgbaFrame> {
    let mut preview = CutlassPreview::video(path)?;
    preview.render_seconds(0.0)
}

/// An RGBA8 image handed across the C ABI. `data` points at `len`
/// (`= width * height * 4`) bytes; `data` is null when rendering failed.
#[repr(C)]
pub struct CutlassImage {
    pub data: *mut u8,
    pub len: usize,
    pub width: u32,
    pub height: u32,
}

impl CutlassImage {
    fn empty() -> Self {
        Self {
            data: std::ptr::null_mut(),
            len: 0,
            width: 0,
            height: 0,
        }
    }

    fn from_parts(width: u32, height: u32, bytes: Vec<u8>) -> Self {
        let boxed: Box<[u8]> = bytes.into_boxed_slice();
        let len = boxed.len();
        let data = Box::into_raw(boxed) as *mut u8;
        Self {
            data,
            len,
            width,
            height,
        }
    }
}

/// Render the demo scene to a freshly allocated RGBA8 buffer.
///
/// On failure the returned [`CutlassImage`] has `data == null`. Every non-null
/// result must be released exactly once with [`cutlass_image_free`].
///
/// # Safety
/// The returned buffer is owned by the callee until passed back to
/// [`cutlass_image_free`]; do not free it any other way.
#[unsafe(no_mangle)]
pub extern "C" fn cutlass_render_demo(width: u32, height: u32) -> CutlassImage {
    match render_demo_rgba(width, height) {
        Some(img) => CutlassImage::from_parts(img.width, img.height, img.bytes),
        None => CutlassImage::empty(),
    }
}

/// Decode + composite the first frame of the video file at `path_utf8` (a UTF-8
/// path of `path_len` bytes).
///
/// `max_width`/`max_height` are advisory — the engine picks the preview canvas
/// size. On failure the returned [`CutlassImage`] has `data == null`. Every
/// non-null result must be released exactly once with [`cutlass_image_free`].
///
/// # Safety
/// `path_utf8` must point to `path_len` initialized bytes (no NUL terminator
/// required). The returned buffer is owned by the callee until freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_render_file_frame(
    path_utf8: *const u8,
    path_len: usize,
    max_width: u32,
    max_height: u32,
) -> CutlassImage {
    let _ = (max_width, max_height);
    if path_utf8.is_null() || path_len == 0 {
        return CutlassImage::empty();
    }
    // SAFETY: caller guarantees `path_len` initialized bytes at `path_utf8`.
    let bytes = unsafe { std::slice::from_raw_parts(path_utf8, path_len) };
    let Ok(path) = std::str::from_utf8(bytes) else {
        return CutlassImage::empty();
    };
    match render_file_frame_rgba(Path::new(path)) {
        Some(frame) => CutlassImage::from_parts(frame.width, frame.height, frame.bytes),
        None => CutlassImage::empty(),
    }
}

/// Free a buffer returned by an image-producing call. A null/empty image is a
/// no-op, so double-free of an already-empty handle is safe.
///
/// # Safety
/// `img` must be a value previously returned by this crate and not already
/// freed; `data`/`len` must not have been modified.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_image_free(img: CutlassImage) {
    if img.data.is_null() || img.len == 0 {
        return;
    }
    // SAFETY: `data`/`len` came from `Box::<[u8]>::into_raw` in `from_parts`.
    unsafe {
        let slice = std::slice::from_raw_parts_mut(img.data, img.len);
        drop(Box::from_raw(slice as *mut [u8]));
    }
}

// ---------------------------------------------------------------------------
// Interactive preview
//
// Scrubbing is the first real-editor workload, and it's where per-call engine
// setup would fall over: each slider tick must reuse the GPU device, compositor,
// and (for video) the decoder cache. `CutlassPreview` holds an `Engine` for the
// session; the app keeps an opaque handle and calls `render` per frame. The
// engine caches decoders internally, so a video scrub seeks within one open
// decoder instead of reopening per frame.
// ---------------------------------------------------------------------------

/// A live preview session: a persistent engine bound to a project, so scrubbing
/// only pays for the frame at `t`.
///
/// Not thread-safe: `render` takes `&mut self`, so the caller must serialize
/// calls on a single handle (a scrubber issues them one at a time).
pub struct CutlassPreview {
    engine: Engine,
    fps: Rational,
    duration_frames: i64,
}

impl CutlassPreview {
    /// A synthetic, file-free preview: a full-canvas color sweeping through the
    /// hue wheel over ~6s. Needs no assets or decoder, so it proves the scrub
    /// path (slider time -> composited frame) on any device, even offline.
    fn demo() -> Option<Self> {
        let fps = Rational::FPS_30;
        let total = 180i64; // 6s @ 30fps
        let step = 6i64; // a new color every 0.2s
        let mut project = Project::new("preview-demo", fps);
        let track = project.add_track(TrackKind::Sticker, "BG");
        let mut start = 0i64;
        while start < total {
            let len = step.min(total - start);
            let rgba = hue_sweep(start as f32 / total as f32);
            let span = TimeRange::at_rate(start, len, fps);
            if let Err(e) = project.add_generated(track, Generator::SolidColor { rgba }, span) {
                log::error!("preview demo: add_generated failed: {e}");
                return None;
            }
            start += step;
        }
        let engine = new_engine(project)?;
        Some(Self {
            engine,
            fps,
            duration_frames: total,
        })
    }

    /// A preview that scrubs the video file at `path`.
    fn video(path: &Path) -> Option<Self> {
        let probe = match cutlass_probe::probe(path) {
            Ok(probe) => probe,
            Err(e) => {
                log::error!("preview video: probe failed for {}: {e}", path.display());
                return None;
            }
        };
        let fps = probe.frame_rate;
        let frames = probe.duration_ticks.max(1);
        let mut project = Project::new("preview-video", fps);
        let media = project.add_media(probe.to_media_source(path));
        let track = project.add_track(TrackKind::Video, "Video");
        if let Err(e) = project.add_clip(
            track,
            media,
            TimeRange::at_rate(0, frames, fps),
            RationalTime::new(0, fps),
        ) {
            log::error!("preview video: add_clip failed: {e}");
            return None;
        }
        let engine = new_engine(project)?;
        Some(Self {
            engine,
            fps,
            duration_frames: frames,
        })
    }

    fn duration_seconds(&self) -> f64 {
        let fps = self.fps.as_f64();
        if fps <= 0.0 {
            return 0.0;
        }
        self.duration_frames as f64 / fps
    }

    fn render_seconds(&mut self, seconds: f64) -> Option<RgbaFrame> {
        let max_frame = (self.duration_frames - 1).max(0);
        let frame = (seconds.max(0.0) * self.fps.as_f64()).round() as i64;
        let frame = frame.clamp(0, max_frame);
        match self.engine.get_frame(RationalTime::new(frame, self.fps)) {
            Ok(image) => Some(image),
            Err(e) => {
                log::error!("preview render at frame {frame} failed: {e}");
                None
            }
        }
    }
}

/// An opaque RGBA sweep through the hue wheel as `t` runs 0->1 (always opaque).
fn hue_sweep(t: f32) -> [u8; 4] {
    let h = (t.rem_euclid(1.0) * 6.0).rem_euclid(6.0);
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    let (r, g, b) = match h as u32 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    [
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
        255,
    ]
}

/// Open the synthetic demo preview. Returns null if the GPU/engine can't be
/// brought up. Free the handle with [`cutlass_preview_close`].
#[unsafe(no_mangle)]
pub extern "C" fn cutlass_preview_open_demo() -> *mut CutlassPreview {
    match CutlassPreview::demo() {
        Some(preview) => Box::into_raw(Box::new(preview)),
        None => std::ptr::null_mut(),
    }
}

/// Open a preview that scrubs the video at `path_utf8` (UTF-8, `path_len`
/// bytes). Returns null if the file can't be probed or the GPU is unavailable.
///
/// # Safety
/// `path_utf8` must point to `path_len` initialized bytes. The returned handle
/// is owned by the callee until passed to [`cutlass_preview_close`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_preview_open_video(
    path_utf8: *const u8,
    path_len: usize,
) -> *mut CutlassPreview {
    if path_utf8.is_null() || path_len == 0 {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees `path_len` initialized bytes at `path_utf8`.
    let bytes = unsafe { std::slice::from_raw_parts(path_utf8, path_len) };
    let Ok(path) = std::str::from_utf8(bytes) else {
        return std::ptr::null_mut();
    };
    match CutlassPreview::video(Path::new(path)) {
        Some(preview) => Box::into_raw(Box::new(preview)),
        None => std::ptr::null_mut(),
    }
}

/// Total scrub length in seconds. `0.0` for a null handle.
///
/// # Safety
/// `handle` must be null or a live pointer from `cutlass_preview_open_*`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_preview_duration_seconds(handle: *const CutlassPreview) -> f64 {
    if handle.is_null() {
        return 0.0;
    }
    // SAFETY: caller guarantees a live, exclusive-enough handle for a read.
    unsafe { &*handle }.duration_seconds()
}

/// Render the preview frame at `seconds` (clamped to the project's range).
///
/// On failure the returned [`CutlassImage`] has `data == null`. Every non-null
/// result must be released exactly once with [`cutlass_image_free`].
///
/// # Safety
/// `handle` must be null or a live pointer from `cutlass_preview_open_*`, and
/// must not be used concurrently from another thread (render takes `&mut`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_preview_render(
    handle: *mut CutlassPreview,
    seconds: f64,
) -> CutlassImage {
    if handle.is_null() {
        return CutlassImage::empty();
    }
    // SAFETY: caller guarantees a live handle used non-concurrently.
    let preview = unsafe { &mut *handle };
    match preview.render_seconds(seconds) {
        Some(frame) => CutlassImage::from_parts(frame.width, frame.height, frame.bytes),
        None => CutlassImage::empty(),
    }
}

/// Free a handle from `cutlass_preview_open_*`. Null is a no-op.
///
/// # Safety
/// `handle` must be null or a value previously returned by an open call and not
/// already freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_preview_close(handle: *mut CutlassPreview) {
    if handle.is_null() {
        return;
    }
    // SAFETY: `handle` came from `Box::into_raw` and is freed exactly once.
    drop(unsafe { Box::from_raw(handle) });
}

// ---------------------------------------------------------------------------
// Editing session: the JSON command/state bridge
//
// `CutlassPreview` above is render-only. `CutlassSession` is the real-editor
// surface: it owns an `Engine` and exposes the same one-way contract the future
// AI agent will use — Swift sends a `Command` as JSON (`apply`), reads a curated
// project read-model back as JSON (`state_json`), and renders preview frames at
// a time (`render`). Edits flow in as commands; the UI never mutates engine
// state directly, so undo/redo and persistence stay the engine's job.
// ---------------------------------------------------------------------------

/// A live editing session: a persistent engine plus the JSON bridge the app
/// drives it through.
///
/// Not thread-safe: every mutating call takes `&mut self`, so the caller must
/// serialize calls on a single handle.
pub struct CutlassSession {
    engine: Engine,
}

/// The result of a command/undo/redo, serialized back to the app so it can
/// refresh selection and toolbar state without a second round-trip.
#[derive(serde::Serialize)]
struct CommandResult {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    /// The clip a creating edit produced (AddClip/SplitClip/…), for selection.
    #[serde(skip_serializing_if = "Option::is_none")]
    created_clip: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    created_track: Option<u64>,
    /// The media id a successful import/relink produced.
    #[serde(skip_serializing_if = "Option::is_none")]
    media: Option<u64>,
    revision: u64,
    dirty: bool,
    can_undo: bool,
    can_redo: bool,
}

/// Curated project read-model (the wire shape the app renders the timeline
/// from). Decoupled from the on-disk `.cutlass` schema on purpose: it only
/// carries what the v1 editor needs, so the schema can evolve independently.
#[derive(serde::Serialize)]
struct ProjectStateDto {
    name: String,
    fps: RationalDto,
    /// Total timeline length in `fps` ticks (end of the last-ending clip).
    duration_ticks: i64,
    revision: u64,
    dirty: bool,
    can_undo: bool,
    can_redo: bool,
    canvas: CanvasDto,
    media: Vec<MediaDto>,
    /// Tracks bottom-to-top in stacking order (audio floor first).
    tracks: Vec<TrackDto>,
}

#[derive(serde::Serialize)]
struct RationalDto {
    num: i32,
    den: i32,
}

#[derive(serde::Serialize)]
struct CanvasDto {
    aspect: String,
    background: [u8; 3],
}

#[derive(serde::Serialize)]
struct MediaDto {
    id: u64,
    path: String,
    fps: RationalDto,
    duration_ticks: i64,
    width: u32,
    height: u32,
    is_image: bool,
    has_audio: bool,
}

#[derive(serde::Serialize)]
struct TrackDto {
    id: u64,
    kind: String,
    name: String,
    enabled: bool,
    muted: bool,
    locked: bool,
    clips: Vec<ClipDto>,
}

#[derive(serde::Serialize)]
struct ClipDto {
    id: u64,
    /// `"media"` or `"generated"`.
    kind: &'static str,
    /// Timeline placement, in project-fps ticks.
    start: i64,
    duration: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    media_id: Option<u64>,
    label: String,
}

fn rational_dto(r: Rational) -> RationalDto {
    RationalDto {
        num: r.num,
        den: r.den,
    }
}

/// Pull the ids worth surfacing out of an apply outcome (new clip/track to
/// select, the media an import produced). A wildcard arm keeps this resilient
/// to the `Exported` variant only existing on non-mobile builds and to future
/// outcome kinds.
fn outcome_ids(outcome: &ApplyOutcome) -> (Option<u64>, Option<u64>, Option<u64>) {
    match outcome {
        ApplyOutcome::Imported { media }
        | ApplyOutcome::RemovedMedia { media }
        | ApplyOutcome::Relinked { media } => (None, None, Some(media.raw())),
        ApplyOutcome::Edited(EditOutcome::Created(clip)) => (Some(clip.raw()), None, None),
        ApplyOutcome::Edited(EditOutcome::CreatedTrack(track)) => (None, Some(track.raw()), None),
        _ => (None, None, None),
    }
}

impl CutlassSession {
    /// A fresh, empty project at `fps` with one video track ready for a drop,
    /// caching under `cache_dir` (the app's sandbox).
    fn new(fps: Rational, cache_dir: Option<PathBuf>) -> Option<Self> {
        let mut project = Project::new("Untitled", fps);
        // v1 lands imports on a single main video track, so create it up front:
        // the app can `AddClip` immediately without an `AddTrack` dance.
        project.add_track(TrackKind::Video, "Video");
        let engine = new_engine_in(project, cache_dir)?;
        Some(Self { engine })
    }

    /// Reopen a `.cutlass` project file (every referenced media path must
    /// exist), caching under `cache_dir`.
    fn open(path: &Path, cache_dir: Option<PathBuf>) -> Option<Self> {
        // Stand up a throwaway engine, then let `Open` replace the session from
        // disk — the same path the desktop editor uses.
        let mut engine = new_engine_in(Project::new("Untitled", Rational::FPS_30), cache_dir)?;
        if let Err(e) = engine.apply(Command::Project(cutlass_engine::ProjectCommand::Open {
            path: path.to_path_buf(),
        })) {
            log::error!("session open failed for {}: {e}", path.display());
            return None;
        }
        Some(Self { engine })
    }

    fn command_result(&self, outcome: Result<ApplyOutcome, cutlass_engine::EngineError>) -> String {
        let result = match outcome {
            Ok(o) => {
                let (created_clip, created_track, media) = outcome_ids(&o);
                CommandResult {
                    ok: true,
                    error: None,
                    created_clip,
                    created_track,
                    media,
                    revision: self.engine.revision(),
                    dirty: self.engine.is_dirty(),
                    can_undo: self.engine.can_undo(),
                    can_redo: self.engine.can_redo(),
                }
            }
            Err(e) => CommandResult {
                ok: false,
                error: Some(e.to_string()),
                created_clip: None,
                created_track: None,
                media: None,
                revision: self.engine.revision(),
                dirty: self.engine.is_dirty(),
                can_undo: self.engine.can_undo(),
                can_redo: self.engine.can_redo(),
            },
        };
        serde_json::to_string(&result).unwrap_or_else(|_| "{\"ok\":false}".to_string())
    }

    /// Apply a JSON-encoded [`Command`], returning a [`CommandResult`] JSON.
    fn apply_json(&mut self, json: &str) -> String {
        let command: Command = match serde_json::from_str(json) {
            Ok(c) => c,
            Err(e) => {
                let result = CommandResult {
                    ok: false,
                    error: Some(format!("invalid command json: {e}")),
                    created_clip: None,
                    created_track: None,
                    media: None,
                    revision: self.engine.revision(),
                    dirty: self.engine.is_dirty(),
                    can_undo: self.engine.can_undo(),
                    can_redo: self.engine.can_redo(),
                };
                return serde_json::to_string(&result).unwrap_or_else(|_| "{\"ok\":false}".into());
            }
        };
        let outcome = self.engine.apply(command);
        self.command_result(outcome)
    }

    fn undo(&mut self) -> String {
        let did = self.engine.undo();
        let result = CommandResult {
            ok: did,
            error: (!did).then(|| "nothing to undo".to_string()),
            created_clip: None,
            created_track: None,
            media: None,
            revision: self.engine.revision(),
            dirty: self.engine.is_dirty(),
            can_undo: self.engine.can_undo(),
            can_redo: self.engine.can_redo(),
        };
        serde_json::to_string(&result).unwrap_or_else(|_| "{\"ok\":false}".into())
    }

    fn redo(&mut self) -> String {
        let did = self.engine.redo();
        let result = CommandResult {
            ok: did,
            error: (!did).then(|| "nothing to redo".to_string()),
            created_clip: None,
            created_track: None,
            media: None,
            revision: self.engine.revision(),
            dirty: self.engine.is_dirty(),
            can_undo: self.engine.can_undo(),
            can_redo: self.engine.can_redo(),
        };
        serde_json::to_string(&result).unwrap_or_else(|_| "{\"ok\":false}".into())
    }

    /// Probe + register a media file (`ProjectCommand::Import`), returning a
    /// JSON object with the new media id and the metadata the app needs to
    /// build an `AddClip` (frame rate, full-source duration, dimensions).
    fn import(&mut self, path: &Path) -> String {
        let outcome = self
            .engine
            .apply(Command::Project(cutlass_engine::ProjectCommand::Import {
                path: path.to_path_buf(),
            }));
        let media_id = match outcome {
            Ok(ApplyOutcome::Imported { media }) => media,
            Ok(_) => {
                return serde_json::json!({
                    "ok": false,
                    "error": "import returned an unexpected outcome",
                })
                .to_string();
            }
            Err(e) => {
                return serde_json::json!({ "ok": false, "error": e.to_string() }).to_string();
            }
        };
        let Some(media) = self.engine.project().media(media_id) else {
            return serde_json::json!({ "ok": false, "error": "media vanished after import" })
                .to_string();
        };
        serde_json::json!({
            "ok": true,
            "media": media_id.raw(),
            "fps": rational_dto(media.frame_rate),
            "duration_ticks": media.duration.value,
            "width": media.width,
            "height": media.height,
            "is_image": media.is_image,
            "has_audio": media.has_audio,
            "revision": self.engine.revision(),
            "dirty": self.engine.is_dirty(),
            "can_undo": self.engine.can_undo(),
            "can_redo": self.engine.can_redo(),
        })
        .to_string()
    }

    fn save(&mut self, path: &Path) -> bool {
        match self
            .engine
            .apply(Command::Project(cutlass_engine::ProjectCommand::Save {
                path: path.to_path_buf(),
            })) {
            Ok(_) => true,
            Err(e) => {
                log::error!("session save failed for {}: {e}", path.display());
                false
            }
        }
    }

    /// Serialize the curated project read-model.
    fn state_json(&self) -> String {
        let project = self.engine.project();
        let timeline = project.timeline();
        let fps = timeline.frame_rate;
        let canvas = timeline.canvas();

        let media = project
            .media_iter()
            .map(|m| MediaDto {
                id: m.id.raw(),
                path: m.path().to_string_lossy().into_owned(),
                fps: rational_dto(m.frame_rate),
                duration_ticks: m.duration.value,
                width: m.width,
                height: m.height,
                is_image: m.is_image,
                has_audio: m.has_audio,
            })
            .collect();

        let tracks = timeline
            .tracks_ordered()
            .map(|track| {
                let clips = track
                    .clips_ordered()
                    .into_iter()
                    .map(|clip| {
                        let media_id = clip.media().map(|m| m.raw());
                        let label = media_id
                            .and_then(|id| project.media(cutlass_models::MediaId::from_raw(id)))
                            .map(|m| {
                                m.path()
                                    .file_name()
                                    .map(|n| n.to_string_lossy().into_owned())
                                    .unwrap_or_else(|| "Clip".to_string())
                            })
                            .unwrap_or_else(|| "Generated".to_string());
                        ClipDto {
                            id: clip.id.raw(),
                            kind: if clip.is_generated() { "generated" } else { "media" },
                            start: clip.timeline.start.value,
                            duration: clip.timeline.duration.value,
                            media_id,
                            label,
                        }
                    })
                    .collect();
                TrackDto {
                    id: track.id.raw(),
                    kind: format!("{:?}", track.kind).to_lowercase(),
                    name: track.name.clone(),
                    enabled: track.enabled,
                    muted: track.muted,
                    locked: track.locked,
                    clips,
                }
            })
            .collect();

        let dto = ProjectStateDto {
            name: project.name.clone(),
            fps: rational_dto(fps),
            duration_ticks: timeline.duration().value,
            revision: self.engine.revision(),
            dirty: self.engine.is_dirty(),
            can_undo: self.engine.can_undo(),
            can_redo: self.engine.can_redo(),
            canvas: CanvasDto {
                aspect: canvas.aspect.name().to_string(),
                background: canvas.background,
            },
            media,
            tracks,
        };
        serde_json::to_string(&dto).unwrap_or_else(|_| "{}".to_string())
    }

    /// Render the preview frame at `seconds`, clamped to the current timeline.
    fn render_seconds(&mut self, seconds: f64) -> Option<RgbaFrame> {
        let fps = self.engine.project().timeline().frame_rate;
        let duration_frames = self.engine.project().timeline().duration().value;
        let max_frame = (duration_frames - 1).max(0);
        let frame = (seconds.max(0.0) * fps.as_f64()).round() as i64;
        let frame = frame.clamp(0, max_frame);
        match self.engine.get_frame(RationalTime::new(frame, fps)) {
            Ok(image) => Some(image),
            Err(e) => {
                log::error!("session render at frame {frame} failed: {e}");
                None
            }
        }
    }
}

/// Read a UTF-8 path from a `(ptr, len)` pair. `None` on null/empty/non-UTF-8.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes (no NUL terminator required).
unsafe fn path_from_raw(ptr: *const u8, len: usize) -> Option<PathBuf> {
    if ptr.is_null() || len == 0 {
        return None;
    }
    // SAFETY: caller guarantees `len` initialized bytes at `ptr`.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    std::str::from_utf8(bytes).ok().map(PathBuf::from)
}

/// Move a Rust `String` onto the heap as a NUL-terminated C string. Free it
/// with [`cutlass_string_free`]. Returns null only if the string contained an
/// interior NUL (JSON never does).
fn into_c_string(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c) => c.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Create a new, empty editing session at `fps_num`/`fps_den` (e.g. 30/1),
/// caching under `cache_dir_utf8` (the app's sandbox dir). The session starts
/// with one video track. Returns null on invalid fps or engine/GPU failure;
/// free with [`cutlass_session_close`].
///
/// # Safety
/// `cache_dir_utf8` must be null or point to `cache_dir_len` initialized bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_new(
    fps_num: i32,
    fps_den: i32,
    cache_dir_utf8: *const u8,
    cache_dir_len: usize,
) -> *mut CutlassSession {
    if fps_num <= 0 || fps_den <= 0 {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees the cache-dir buffer's bounds.
    let cache_dir = unsafe { path_from_raw(cache_dir_utf8, cache_dir_len) };
    match CutlassSession::new(Rational::new(fps_num, fps_den), cache_dir) {
        Some(session) => Box::into_raw(Box::new(session)),
        None => std::ptr::null_mut(),
    }
}

/// Open a `.cutlass` project file into a new session, caching under
/// `cache_dir_utf8`. Returns null on failure; free with
/// [`cutlass_session_close`].
///
/// # Safety
/// Both `(ptr, len)` buffers must be null or point to that many initialized
/// bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_open(
    path_utf8: *const u8,
    path_len: usize,
    cache_dir_utf8: *const u8,
    cache_dir_len: usize,
) -> *mut CutlassSession {
    // SAFETY: caller guarantees both buffers' bounds.
    let Some(path) = (unsafe { path_from_raw(path_utf8, path_len) }) else {
        return std::ptr::null_mut();
    };
    let cache_dir = unsafe { path_from_raw(cache_dir_utf8, cache_dir_len) };
    match CutlassSession::open(&path, cache_dir) {
        Some(session) => Box::into_raw(Box::new(session)),
        None => std::ptr::null_mut(),
    }
}

/// Save the session to a `.cutlass` file at `path_utf8`. Returns `true` on
/// success.
///
/// # Safety
/// `handle` must be a live session pointer; `path_utf8` must point to
/// `path_len` initialized bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_save(
    handle: *mut CutlassSession,
    path_utf8: *const u8,
    path_len: usize,
) -> bool {
    if handle.is_null() {
        return false;
    }
    // SAFETY: caller guarantees a live, non-concurrent handle and buffer.
    let session = unsafe { &mut *handle };
    let Some(path) = (unsafe { path_from_raw(path_utf8, path_len) }) else {
        return false;
    };
    session.save(&path)
}

/// Apply a JSON-encoded command, returning a heap C string with the
/// [`CommandResult`] JSON (free with [`cutlass_string_free`]).
///
/// # Safety
/// `handle` must be a live session pointer; `json_utf8` must point to
/// `json_len` initialized bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_apply(
    handle: *mut CutlassSession,
    json_utf8: *const u8,
    json_len: usize,
) -> *mut c_char {
    if handle.is_null() || json_utf8.is_null() || json_len == 0 {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees a live handle and the json buffer's bounds.
    let session = unsafe { &mut *handle };
    let bytes = unsafe { std::slice::from_raw_parts(json_utf8, json_len) };
    let Ok(json) = std::str::from_utf8(bytes) else {
        return into_c_string("{\"ok\":false,\"error\":\"command json was not utf-8\"}".into());
    };
    into_c_string(session.apply_json(json))
}

/// Undo the last edit. Returns a [`CommandResult`] JSON C string.
///
/// # Safety
/// `handle` must be a live session pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_undo(handle: *mut CutlassSession) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees a live, non-concurrent handle.
    into_c_string(unsafe { &mut *handle }.undo())
}

/// Redo the last undone edit. Returns a [`CommandResult`] JSON C string.
///
/// # Safety
/// `handle` must be a live session pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_redo(handle: *mut CutlassSession) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees a live, non-concurrent handle.
    into_c_string(unsafe { &mut *handle }.redo())
}

/// Import a media file into the pool. Returns a JSON C string with the new
/// media id and metadata for building an `AddClip` (free with
/// [`cutlass_string_free`]).
///
/// # Safety
/// `handle` must be a live session pointer; `path_utf8` must point to
/// `path_len` initialized bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_import(
    handle: *mut CutlassSession,
    path_utf8: *const u8,
    path_len: usize,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees a live handle and the path buffer's bounds.
    let session = unsafe { &mut *handle };
    let Some(path) = (unsafe { path_from_raw(path_utf8, path_len) }) else {
        return into_c_string("{\"ok\":false,\"error\":\"invalid import path\"}".into());
    };
    into_c_string(session.import(&path))
}

/// Serialize the curated project read-model. Returns a JSON C string (free with
/// [`cutlass_string_free`]).
///
/// # Safety
/// `handle` must be a live session pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_state_json(handle: *const CutlassSession) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees a live handle, read-only here.
    into_c_string(unsafe { &*handle }.state_json())
}

/// Render the session's preview frame at `seconds` (clamped to the timeline).
/// On failure the returned [`CutlassImage`] has `data == null`. Free non-null
/// results with [`cutlass_image_free`].
///
/// # Safety
/// `handle` must be a live session pointer used non-concurrently (render takes
/// `&mut`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_render(
    handle: *mut CutlassSession,
    seconds: f64,
) -> CutlassImage {
    if handle.is_null() {
        return CutlassImage::empty();
    }
    // SAFETY: caller guarantees a live handle used non-concurrently.
    let session = unsafe { &mut *handle };
    match session.render_seconds(seconds) {
        Some(frame) => CutlassImage::from_parts(frame.width, frame.height, frame.bytes),
        None => CutlassImage::empty(),
    }
}

/// Free a session handle from `cutlass_session_new`/`_open`. Null is a no-op.
///
/// # Safety
/// `handle` must be null or a value previously returned by an open call and not
/// already freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_session_close(handle: *mut CutlassSession) {
    if handle.is_null() {
        return;
    }
    // SAFETY: `handle` came from `Box::into_raw` and is freed exactly once.
    drop(unsafe { Box::from_raw(handle) });
}

/// Free a JSON C string returned by a `cutlass_session_*` call. Null is a
/// no-op.
///
/// # Safety
/// `ptr` must be null or a value returned by this crate and not already freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_string_free(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: `ptr` came from `CString::into_raw` in `into_c_string`.
    drop(unsafe { CString::from_raw(ptr) });
}

#[cfg(target_os = "android")]
mod android_jni {
    use std::path::Path;

    use jni::JNIEnv;
    use jni::objects::{JClass, JString};
    use jni::sys::{jdouble, jint, jintArray, jlong};

    use super::{CutlassPreview, render_demo_rgba, render_file_frame_rgba};

    /// Route `log::*` to logcat (tag `cutlass`) once. Idempotent.
    fn init_logging() {
        use std::sync::Once;
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            android_logger::init_once(
                android_logger::Config::default()
                    .with_tag("cutlass")
                    .with_max_level(log::LevelFilter::Debug),
            );
        });
    }

    fn empty_array(env: &JNIEnv) -> jintArray {
        env.new_int_array(0)
            .map(|a| a.into_raw())
            .unwrap_or(std::ptr::null_mut())
    }

    /// Marshal RGBA bytes into a Java `int[]` laid out as
    /// `[width, height, argb_0, argb_1, …]` — the leading dimensions let the
    /// caller build the `Bitmap` without knowing the (fitted) output size up
    /// front. Pixels are ARGB_8888 (what `Bitmap.createBitmap` expects).
    fn rgba_to_argb_array(env: &JNIEnv, width: u32, height: u32, pixels: &[u8]) -> jintArray {
        let count = (width as usize).saturating_mul(height as usize);
        let mut argb = vec![0i32; count + 2];
        argb[0] = width as i32;
        argb[1] = height as i32;
        for (slot, px) in argb[2..].iter_mut().zip(pixels.chunks_exact(4)) {
            let r = i32::from(px[0]);
            let g = i32::from(px[1]);
            let b = i32::from(px[2]);
            let a = i32::from(px[3]);
            *slot = (a << 24) | (r << 16) | (g << 8) | b;
        }
        match env.new_int_array(argb.len() as jint) {
            Ok(array) => {
                if env.set_int_array_region(&array, 0, &argb).is_err() {
                    return empty_array(env);
                }
                array.into_raw()
            }
            Err(_) => empty_array(env),
        }
    }

    /// `CutlassNative.renderDemo(width, height)` -> `[width, height, argb…]`.
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_com_scytheralpha_cutlass_1android_CutlassNative_renderDemo<'local>(
        env: JNIEnv<'local>,
        _class: JClass<'local>,
        width: jint,
        height: jint,
    ) -> jintArray {
        init_logging();
        match render_demo_rgba(width.max(0) as u32, height.max(0) as u32) {
            Some(image) => rgba_to_argb_array(&env, image.width, image.height, &image.bytes),
            None => empty_array(&env),
        }
    }

    /// `CutlassNative.renderFileFrame(path, maxWidth, maxHeight)`: first frame of
    /// the video at `path` as ARGB_8888 pixels. Empty array on failure.
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_com_scytheralpha_cutlass_1android_CutlassNative_renderFileFrame<
        'local,
    >(
        mut env: JNIEnv<'local>,
        _class: JClass<'local>,
        path: JString<'local>,
        _max_width: jint,
        _max_height: jint,
    ) -> jintArray {
        let Ok(path) = env.get_string(&path) else {
            return empty_array(&env);
        };
        let path: String = path.into();
        match render_file_frame_rgba(Path::new(&path)) {
            Some(frame) => rgba_to_argb_array(&env, frame.width, frame.height, &frame.bytes),
            None => empty_array(&env),
        }
    }

    /// `CutlassNative.previewOpenDemo()` -> handle as `jlong` (`0` on failure).
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_com_scytheralpha_cutlass_1android_CutlassNative_previewOpenDemo<
        'local,
    >(
        _env: JNIEnv<'local>,
        _class: JClass<'local>,
    ) -> jlong {
        init_logging();
        match CutlassPreview::demo() {
            Some(preview) => Box::into_raw(Box::new(preview)) as jlong,
            None => 0,
        }
    }

    /// `CutlassNative.previewOpenVideo(path)` -> handle as `jlong` (`0` on fail).
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_com_scytheralpha_cutlass_1android_CutlassNative_previewOpenVideo<
        'local,
    >(
        mut env: JNIEnv<'local>,
        _class: JClass<'local>,
        path: JString<'local>,
    ) -> jlong {
        init_logging();
        let Ok(path) = env.get_string(&path) else {
            return 0;
        };
        let path: String = path.into();
        match CutlassPreview::video(Path::new(&path)) {
            Some(preview) => Box::into_raw(Box::new(preview)) as jlong,
            None => 0,
        }
    }

    /// `CutlassNative.previewDurationSeconds(handle)`: total scrub length.
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_com_scytheralpha_cutlass_1android_CutlassNative_previewDurationSeconds<
        'local,
    >(
        _env: JNIEnv<'local>,
        _class: JClass<'local>,
        handle: jlong,
    ) -> jdouble {
        if handle == 0 {
            return 0.0;
        }
        // SAFETY: `handle` is a live pointer from `previewOpen*`, read-only here.
        let preview = unsafe { &*(handle as *const CutlassPreview) };
        preview.duration_seconds()
    }

    /// `CutlassNative.previewRender(handle, seconds)` -> `[width, height, argb…]`.
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_com_scytheralpha_cutlass_1android_CutlassNative_previewRender<
        'local,
    >(
        env: JNIEnv<'local>,
        _class: JClass<'local>,
        handle: jlong,
        seconds: jdouble,
    ) -> jintArray {
        if handle == 0 {
            return empty_array(&env);
        }
        // SAFETY: `handle` is a live pointer used non-concurrently (UI serializes
        // scrub renders on one handle).
        let preview = unsafe { &mut *(handle as *mut CutlassPreview) };
        match preview.render_seconds(seconds) {
            Some(frame) => rgba_to_argb_array(&env, frame.width, frame.height, &frame.bytes),
            None => empty_array(&env),
        }
    }

    /// `CutlassNative.previewClose(handle)`: release a preview handle.
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_com_scytheralpha_cutlass_1android_CutlassNative_previewClose<
        'local,
    >(
        _env: JNIEnv<'local>,
        _class: JClass<'local>,
        handle: jlong,
    ) {
        if handle != 0 {
            // SAFETY: `handle` came from `Box::into_raw` in `previewOpen*` and is
            // freed exactly once.
            drop(unsafe { Box::from_raw(handle as *mut CutlassPreview) });
        }
    }
}

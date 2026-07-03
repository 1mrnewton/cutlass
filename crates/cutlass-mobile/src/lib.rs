//! Mobile FFI bridge for the Cutlass test apps.
//!
//! This is the thin native surface the blank iOS and Android scaffolds in
//! `apps/` call to prove the engine runs on a device's real GPU: it brings up a
//! headless `wgpu` device (Metal on iOS, Vulkan/GLES on Android), composites a
//! small built-in scene, and hands back the RGBA pixels for the app to display.
//!
//! Two front doors, one core:
//! - **iOS** links the `staticlib` and calls the plain C ABI
//!   ([`cutlass_render_demo`] / [`cutlass_image_free`]).
//! - **Android** loads the `cdylib` and calls the JNI entry point
//!   (`CutlassNative.renderDemo`), which returns ARGB_8888 ints ready for a
//!   `Bitmap`.
//!
//! Keeping the GPU/compositor work in shared Rust means the on-device harness
//! exercises the same path as export and the future preview — the only
//! per-platform code is the marshalling in this crate.

use std::path::Path;

use cutlass_compositor::{
    CompositeLayer, Compositor, CompositorConfig, GpuContext, LayerPlacement, RgbaImage,
};
use cutlass_decoder::OutputMode;
use cutlass_models::{Generator, MediaSource, Project, Rational, RationalTime, TimeRange, TrackKind};
use cutlass_render::Renderer;

/// Render the built-in demo scene at `width`×`height`. `None` if the dimensions
/// are degenerate or the GPU device/compositor can't be brought up.
///
/// The scene is solid layers only (no decode, no fonts) so it renders
/// identically on every device and isolates "does this device's GPU pipeline
/// work" from codec/asset concerns.
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
    let compositor = Compositor::new(&gpu);

    let config = CompositorConfig::new(width, height).with_background([18, 22, 30, 255]);
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
        // A teal square in the upper-left quadrant (proves placement + y-down).
        CompositeLayer::solid(
            [46, 196, 182, 255],
            LayerPlacement {
                center: [w * 0.3, h * 0.32],
                size: [w * 0.2, w * 0.2],
                rotation: 0.0,
                opacity: 1.0,
            },
        ),
        // A rotated, slightly translucent amber square lower-right (proves
        // rotation + alpha blending).
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

    compositor.render(&gpu, &config, &layers).ok()
}

/// Fit `src` inside `max` preserving aspect ratio, never upscaling. Always
/// returns at least 1×1.
fn fit_within(src: (u32, u32), max: (u32, u32)) -> (u32, u32) {
    let (sw, sh) = src;
    let (mw, mh) = (max.0.max(1), max.1.max(1));
    if sw == 0 || sh == 0 {
        return (mw, mh);
    }
    let scale = (f64::from(mw) / f64::from(sw))
        .min(f64::from(mh) / f64::from(sh))
        .min(1.0);
    let w = ((f64::from(sw) * scale).round() as u32).max(1);
    let h = ((f64::from(sh) * scale).round() as u32).max(1);
    (w, h)
}

/// Decode the first frame of `path` in `mode` and composite it full-canvas onto
/// a canvas that fits within `max`. `None` on any decode/composite failure.
fn compose_first_frame(
    gpu: &GpuContext,
    compositor: &Compositor,
    path: &Path,
    mode: OutputMode,
    max: (u32, u32),
) -> Option<RgbaImage> {
    let mut decoder = cutlass_decoder::open_video_decoder(path, mode).ok()?;
    let display = decoder.info().display_size;
    let frame = decoder.next_frame().ok()??;

    let (cw, ch) = fit_within(display, max);
    let config = CompositorConfig::new(cw, ch);
    let layer = CompositeLayer::frame(&frame, LayerPlacement::full_canvas(&config));
    compositor.render(gpu, &config, &[layer]).ok()
}

/// Decode + composite the first frame of the video at `path`, scaled to fit
/// `max_width`×`max_height`. `None` if the file can't be opened/decoded or the
/// GPU is unavailable.
///
/// Tries the zero-copy GPU decode path first (the fast path on Apple) and falls
/// back to the portable CPU path — so this automatically picks up zero-copy on
/// platforms once their import lands, with no caller change.
fn render_file_frame_rgba(path: &Path, max_width: u32, max_height: u32) -> Option<RgbaImage> {
    if max_width == 0 || max_height == 0 {
        return None;
    }
    let gpu = GpuContext::new_headless_blocking().ok()?;
    let compositor = Compositor::new(&gpu);
    let max = (max_width, max_height);

    compose_first_frame(&gpu, &compositor, path, OutputMode::Gpu, max)
        .or_else(|| compose_first_frame(&gpu, &compositor, path, OutputMode::Cpu, max))
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

    fn from_rgba(img: RgbaImage) -> Self {
        let width = img.width;
        let height = img.height;
        let boxed: Box<[u8]> = img.pixels.into_boxed_slice();
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
        Some(img) => CutlassImage::from_rgba(img),
        None => CutlassImage::empty(),
    }
}

/// Decode + composite the first frame of the video file at `path_utf8` (a UTF-8
/// path of `path_len` bytes), scaled to fit `max_width`×`max_height`.
///
/// On failure the returned [`CutlassImage`] has `data == null`. Every non-null
/// result must be released exactly once with [`cutlass_image_free`].
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
    if path_utf8.is_null() || path_len == 0 {
        return CutlassImage::empty();
    }
    // SAFETY: caller guarantees `path_len` initialized bytes at `path_utf8`.
    let bytes = unsafe { std::slice::from_raw_parts(path_utf8, path_len) };
    let Ok(path) = std::str::from_utf8(bytes) else {
        return CutlassImage::empty();
    };
    match render_file_frame_rgba(Path::new(path), max_width, max_height) {
        Some(img) => CutlassImage::from_rgba(img),
        None => CutlassImage::empty(),
    }
}

/// Free a buffer returned by [`cutlass_render_demo`]. A null/empty image is a
/// no-op, so double-free of an already-empty handle is safe.
///
/// # Safety
/// `img` must be a value previously returned by [`cutlass_render_demo`] and not
/// already freed; `data`/`len` must not have been modified.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_image_free(img: CutlassImage) {
    if img.data.is_null() || img.len == 0 {
        return;
    }
    // SAFETY: `data`/`len` came from `Box::<[u8]>::into_raw` in `from_rgba`.
    unsafe {
        let slice = std::slice::from_raw_parts_mut(img.data, img.len);
        drop(Box::from_raw(slice as *mut [u8]));
    }
}

// ---------------------------------------------------------------------------
// Interactive preview
//
// Scrubbing is the first real-editor workload, and it's where the per-call
// `GpuContext`/decoder setup above would fall over: each slider tick must reuse
// the device, compositor, and (for video) the decoder cache. `CutlassPreview`
// holds that state for the session; the app keeps an opaque handle and calls
// `render` per frame. `Renderer` already caches decoders internally, so a video
// scrub seeks within one open decoder instead of reopening per frame.
// ---------------------------------------------------------------------------

/// A live preview session: a persistent renderer (GPU device + decoder cache)
/// bound to a project, so scrubbing only pays for the frame at `t`.
///
/// Not thread-safe: `render` takes `&mut self`, so the caller must serialize
/// calls on a single handle (a scrubber issues them one at a time).
pub struct CutlassPreview {
    renderer: Renderer,
    project: Project,
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
        let renderer = match Renderer::new_headless() {
            Ok(renderer) => renderer,
            Err(e) => {
                log::error!("preview demo: renderer init failed: {e}");
                return None;
            }
        };
        Some(Self {
            renderer,
            project,
            fps,
            duration_frames: total,
        })
    }

    /// A preview that scrubs the video file at `path`. The canvas follows the
    /// footage (`CanvasAspect::Auto`).
    fn video(path: &Path) -> Option<Self> {
        let probe = match cutlass_decoder::probe(path) {
            Ok(probe) => probe,
            Err(e) => {
                log::error!("preview video: probe failed for {}: {e}", path.display());
                return None;
            }
        };
        let fps = probe.frame_rate;
        let frames = probe.frame_count.max(1);
        let mut project = Project::new("preview-video", fps);
        let media = project.add_media(MediaSource::new(
            path,
            probe.width,
            probe.height,
            fps,
            frames,
            probe.has_audio,
        ));
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
        let renderer = match Renderer::new_headless() {
            Ok(renderer) => renderer,
            Err(e) => {
                log::error!("preview video: renderer init failed: {e}");
                return None;
            }
        };
        Some(Self {
            renderer,
            project,
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

    fn render_seconds(&mut self, seconds: f64) -> Option<RgbaImage> {
        let max_frame = (self.duration_frames - 1).max(0);
        let frame = (seconds.max(0.0) * self.fps.as_f64()).round() as i64;
        let frame = frame.clamp(0, max_frame);
        match self
            .renderer
            .render_frame(&self.project, RationalTime::new(frame, self.fps))
        {
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

/// Open the synthetic demo preview. Returns null if the GPU/renderer can't be
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
        Some(img) => CutlassImage::from_rgba(img),
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

#[cfg(target_os = "android")]
mod android_jni {
    use std::path::Path;

    use cutlass_compositor::RgbaImage;
    use jni::JNIEnv;
    use jni::objects::{JClass, JString};
    use jni::sys::{jdouble, jint, jintArray, jlong};

    use super::{CutlassPreview, render_demo_rgba, render_file_frame_rgba};

    /// Route `log::*` to logcat (tag `cutlass`) once, so native failures are
    /// visible via `adb logcat -s cutlass`. Idempotent.
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

    /// Marshal an [`RgbaImage`] into a Java `int[]` laid out as
    /// `[width, height, argb_0, argb_1, …]` — the leading dimensions let the
    /// caller build the `Bitmap` without knowing the (fitted) output size up
    /// front. Pixels are ARGB_8888 (what `Bitmap.createBitmap` expects).
    fn rgba_to_argb_array(env: &JNIEnv, image: &RgbaImage) -> jintArray {
        let count = (image.width as usize).saturating_mul(image.height as usize);
        let mut argb = vec![0i32; count + 2];
        argb[0] = image.width as i32;
        argb[1] = image.height as i32;
        for (slot, px) in argb[2..].iter_mut().zip(image.pixels.chunks_exact(4)) {
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

    /// `com.scytheralpha.cutlass_android.CutlassNative.renderDemo(width, height)`.
    /// Returns `width * height` ARGB_8888 pixels, or an empty array on failure.
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_com_scytheralpha_cutlass_1android_CutlassNative_renderDemo<'local>(
        env: JNIEnv<'local>,
        _class: JClass<'local>,
        width: jint,
        height: jint,
    ) -> jintArray {
        init_logging();
        match render_demo_rgba(width.max(0) as u32, height.max(0) as u32) {
            Some(image) => rgba_to_argb_array(&env, &image),
            None => empty_array(&env),
        }
    }

    /// `CutlassNative.renderFileFrame(path, maxWidth, maxHeight)`: decode +
    /// composite the first frame of the video at `path`, scaled to fit, as
    /// ARGB_8888 pixels. Empty array on failure.
    #[unsafe(no_mangle)]
    pub extern "system" fn Java_com_scytheralpha_cutlass_1android_CutlassNative_renderFileFrame<
        'local,
    >(
        mut env: JNIEnv<'local>,
        _class: JClass<'local>,
        path: JString<'local>,
        max_width: jint,
        max_height: jint,
    ) -> jintArray {
        let Ok(path) = env.get_string(&path) else {
            return empty_array(&env);
        };
        let path: String = path.into();
        match render_file_frame_rgba(
            Path::new(&path),
            max_width.max(0) as u32,
            max_height.max(0) as u32,
        ) {
            Some(image) => rgba_to_argb_array(&env, &image),
            None => empty_array(&env),
        }
    }

    /// `CutlassNative.previewOpenDemo()`: open the synthetic scrub demo. Returns
    /// a handle as a `jlong` (`0` on failure) — pass it to the other
    /// `preview*` calls and free it with `previewClose`.
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

    /// `CutlassNative.previewOpenVideo(path)`: open a video scrub session.
    /// Returns a handle as a `jlong` (`0` on failure).
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

    /// `CutlassNative.previewRender(handle, seconds)`: the frame at `seconds` as
    /// `[width, height, argb…]`. Empty array on failure.
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
            Some(image) => rgba_to_argb_array(&env, &image),
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

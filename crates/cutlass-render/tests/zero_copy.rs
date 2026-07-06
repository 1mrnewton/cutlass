//! Zero-copy GPU decode path parity (Apple + Windows).
//!
//! A hardware-decoded GPU surface imported into `wgpu` must composite to the
//! same pixels as the portable CPU plane-upload path. This exercises the whole
//! import seam end-to-end on a real device:
//!
//! - Apple: VideoToolbox decode → IOSurface-backed NV12 `CVPixelBuffer` →
//!   `CVMetalTextureCache` → `wgpu` textures → YUV composite → readback.
//! - Windows: Media Foundation D3D11 decode → shared NV12 texture + fence →
//!   D3D12 `OpenSharedHandle` → `wgpu` NV12 plane views → YUV composite →
//!   readback.

#![cfg(any(target_vendor = "apple", target_os = "windows"))]

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use cutlass_compositor::{
    CompositeLayer, Compositor, CompositorConfig, GpuContext, LayerPlacement, RgbaImage,
};
use cutlass_core::RationalTime;
use cutlass_decoder::{OutputMode, open_video_decoder};
use cutlass_models::{
    CanvasAspect, CanvasSettings, Generator, MediaSource, Project, Rational, TimeRange, TrackKind,
};
use cutlass_render::{Renderer, export_to_file};

/// The wgpu backend whose zero-copy import path this platform implements.
#[cfg(target_vendor = "apple")]
const ZERO_COPY_BACKEND: wgpu::Backend = wgpu::Backend::Metal;
#[cfg(target_os = "windows")]
const ZERO_COPY_BACKEND: wgpu::Backend = wgpu::Backend::Dx12;

fn temp_mp4() -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("cutlass_zerocopy_{nanos}.mp4"))
}

/// The H.264 clip the test decodes through both paths.
enum TestClip {
    /// Exported by the test itself (platforms with a native encoder): a
    /// uniform solid-color clip whose decoded content we can assert on.
    Generated(std::path::PathBuf),
    /// Supplied via `CUTLASS_TEST_MEDIA` (platforms without an encoder):
    /// arbitrary real-world content, so only path parity can be asserted.
    External(std::path::PathBuf),
}

impl TestClip {
    fn path(&self) -> &Path {
        match self {
            TestClip::Generated(p) | TestClip::External(p) => p,
        }
    }

    fn is_generated(&self) -> bool {
        matches!(self, TestClip::Generated(_))
    }
}

impl Drop for TestClip {
    fn drop(&mut self) {
        if let TestClip::Generated(p) = self {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// Produce the clip both decode paths read, or `None` (with a skip note) when
/// this platform can't make one and no external file was supplied.
fn test_clip() -> Option<TestClip> {
    // No native Windows video encoder yet, so the test can't author its own
    // clip there; accept any local H.264 MP4 via the environment instead.
    if let Some(path) = std::env::var_os("CUTLASS_TEST_MEDIA") {
        return Some(TestClip::External(path.into()));
    }
    if cfg!(target_vendor = "apple") {
        let path = temp_mp4();
        export_solid_clip(&path);
        return Some(TestClip::Generated(path));
    }
    eprintln!("skipping: no native encoder and CUTLASS_TEST_MEDIA is unset");
    None
}

/// Export a short, uniform solid-color 1920×1080 clip so the decoded frame is a
/// single flat color — robust to any per-plane stride/padding differences
/// between the CPU and GPU import paths.
fn export_solid_clip(path: &Path) {
    let fps = Rational::FPS_30;
    let span = TimeRange::at_rate(0, 3, fps);

    let mut project = Project::new("zero-copy", fps);
    project.timeline_mut().set_canvas(CanvasSettings {
        aspect: CanvasAspect::Wide16x9,
        background: [20, 30, 40],
    });
    let bg = project.add_track(TrackKind::Sticker, "BG");
    project
        .add_generated(
            bg,
            Generator::SolidColor {
                rgba: [40, 80, 160, 255],
            },
            span,
        )
        .unwrap();

    let mut renderer = Renderer::new_headless().expect("headless renderer");
    let frames = export_to_file(&mut renderer, &project, path).expect("export mp4");
    assert_eq!(frames, 3);
}

/// Decode the first frame of `path` in `mode`, composite it full-canvas onto a
/// small canvas, and read it back.
fn first_frame_composited(
    path: &Path,
    mode: OutputMode,
    gpu: &GpuContext,
    comp: &mut Compositor,
) -> RgbaImage {
    let mut decoder = open_video_decoder(path, mode).expect("open decoder");
    let frame = decoder.next_frame().expect("decode").expect("a frame");
    match mode {
        OutputMode::Gpu => assert!(frame.is_gpu(), "gpu mode must yield a GPU frame"),
        OutputMode::Cpu => assert!(frame.is_cpu(), "cpu mode must yield a CPU frame"),
    }

    let config = CompositorConfig::new(64, 64);
    let layer = CompositeLayer::frame(&frame, LayerPlacement::full_canvas(&config));
    comp.render(gpu, &config, &[layer])
        .expect("composite frame")
}

#[test]
fn gpu_import_matches_cpu_upload() {
    let Ok(gpu) = GpuContext::new_headless_blocking() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // The zero-copy import path needs the platform's native backend; skip on
    // software/other backends.
    if gpu.adapter.get_info().backend != ZERO_COPY_BACKEND {
        eprintln!(
            "skipping: adapter is {:?}, not {ZERO_COPY_BACKEND:?}",
            gpu.adapter.get_info().backend
        );
        return;
    }
    let mut comp = Compositor::new(&gpu);

    let Some(clip) = test_clip() else { return };

    let cpu = first_frame_composited(clip.path(), OutputMode::Cpu, &gpu, &mut comp);
    let zero_copy = first_frame_composited(clip.path(), OutputMode::Gpu, &gpu, &mut comp);

    // Every sampled pixel from the zero-copy import must match the CPU upload
    // within unorm rounding tolerance.
    for &(x, y) in &[(8u32, 8u32), (32, 32), (56, 56), (16, 48), (48, 16)] {
        let a = cpu.pixel(x, y);
        let b = zero_copy.pixel(x, y);
        for ch in 0..4 {
            let d = i32::from(a[ch]) - i32::from(b[ch]);
            assert!(
                d.abs() <= 2,
                "pixel ({x},{y}) channel {ch}: cpu {a:?} vs zero-copy {b:?}"
            );
        }
    }

    // Sanity: it composited an opaque frame (not a transparent/empty canvas).
    let center = zero_copy.pixel(32, 32);
    assert_eq!(center[3], 255, "frame must be opaque");
    if clip.is_generated() {
        // The decoded solid must be in the right ballpark of the encoded
        // [40,80,160] (external clips have arbitrary content).
        assert!(
            center[2] > center[0],
            "encoded color was blue-dominant; got {center:?}"
        );
    }
}

/// Drive the whole [`Renderer`] over a project that imports decoded media. On
/// Apple and Windows the renderer defaults to zero-copy GPU decode (with a CPU
/// fallback); this smoke-tests that media frames decode, import/compose, and
/// read back to the expected color through the public render API.
#[test]
fn renderer_renders_decoded_media() {
    if GpuContext::new_headless_blocking().is_err() {
        eprintln!("skipping: no GPU adapter");
        return;
    }

    let Some(clip) = test_clip() else { return };
    let info = cutlass_decoder::probe(clip.path()).expect("probe clip");
    let fps = info.frame_rate;
    let frames = info.frame_count.clamp(1, 3);

    let mut project = Project::new("decoded-media", fps);
    project.timeline_mut().set_canvas(CanvasSettings {
        aspect: CanvasAspect::Wide16x9,
        background: [0, 0, 0],
    });
    let media = project.add_media(MediaSource::new(
        clip.path(),
        info.width,
        info.height,
        fps,
        frames,
        info.has_audio,
    ));
    let track = project.add_track(TrackKind::Video, "Video");
    project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(0, frames, fps),
            RationalTime::new(0, fps),
        )
        .expect("place media clip");

    let mut renderer = Renderer::new_headless().expect("headless renderer");
    let img = renderer
        .render_frame(&project, RationalTime::new(0, fps))
        .expect("render decoded media frame");

    // The clip fills the 16:9 canvas; the center is decoded video content.
    let center = img.pixel(img.width / 2, img.height / 2);
    assert_eq!(center[3], 255, "frame must be opaque");
    if clip.is_generated() {
        // The center is the decoded solid color, which was blue-dominant.
        assert!(
            center[2] > center[0],
            "encoded color was blue-dominant; got {center:?}"
        );
    }
}

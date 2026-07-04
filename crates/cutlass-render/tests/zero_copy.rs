//! Zero-copy GPU decode path parity (Apple).
//!
//! A hardware-decoded `CVPixelBuffer` imported into `wgpu` via
//! `CVMetalTextureCache` must composite to the same pixels as the portable CPU
//! plane-upload path. This exercises the whole import seam end-to-end on a real
//! Metal device: VideoToolbox decode → IOSurface-backed NV12 surface →
//! `CVMetalTextureCache` → `wgpu` textures → YUV composite → readback.

#![cfg(target_vendor = "apple")]

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use cutlass_compositor::{
    CompositeLayer, Compositor, CompositorConfig, GpuContext, LayerPlacement, RgbaImage,
};
use cutlass_core::{RationalTime, VideoDecoder};
use cutlass_decoder::{AvfDecoder, OutputMode};
use cutlass_models::{
    CanvasAspect, CanvasSettings, Generator, MediaSource, Project, Rational, TimeRange, TrackKind,
};
use cutlass_render::{Renderer, export_to_file};

fn temp_mp4() -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("cutlass_zerocopy_{nanos}.mp4"))
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
    let mut decoder = AvfDecoder::open(path, mode).expect("open decoder");
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
    // The zero-copy import path is Metal-only; skip on software/other backends.
    if gpu.adapter.get_info().backend != wgpu::Backend::Metal {
        eprintln!("skipping: adapter is not Metal");
        return;
    }
    let mut comp = Compositor::new(&gpu);

    let path = temp_mp4();
    export_solid_clip(&path);

    let cpu = first_frame_composited(&path, OutputMode::Cpu, &gpu, &mut comp);
    let zero_copy = first_frame_composited(&path, OutputMode::Gpu, &gpu, &mut comp);
    let _ = std::fs::remove_file(&path);

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

    // Sanity: it composited an opaque frame (not a transparent/empty canvas),
    // and the decoded solid is in the right ballpark of the encoded [40,80,160].
    let center = zero_copy.pixel(32, 32);
    assert_eq!(center[3], 255, "frame must be opaque");
    assert!(
        center[2] > center[0],
        "encoded color was blue-dominant; got {center:?}"
    );
}

/// Drive the whole [`Renderer`] over a project that imports decoded media. On
/// Apple the renderer defaults to zero-copy GPU decode (with a CPU fallback);
/// this smoke-tests that media frames decode, import/compose, and read back to
/// the expected color through the public render API.
#[test]
fn renderer_renders_decoded_media() {
    if GpuContext::new_headless_blocking().is_err() {
        eprintln!("skipping: no GPU adapter");
        return;
    }

    let fps = Rational::FPS_30;
    let path = temp_mp4();
    export_solid_clip(&path);

    let mut project = Project::new("decoded-media", fps);
    project.timeline_mut().set_canvas(CanvasSettings {
        aspect: CanvasAspect::Wide16x9,
        background: [0, 0, 0],
    });
    let media = project.add_media(MediaSource::new(&path, 1920, 1080, fps, 3, false));
    let track = project.add_track(TrackKind::Video, "Video");
    project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(0, 3, fps),
            RationalTime::new(0, fps),
        )
        .expect("place media clip");

    let mut renderer = Renderer::new_headless().expect("headless renderer");
    let img = renderer
        .render_frame(&project, RationalTime::new(0, fps))
        .expect("render decoded media frame");
    let _ = std::fs::remove_file(&path);

    // The clip fills the 16:9 canvas; the center is the decoded solid color.
    let center = img.pixel(img.width / 2, img.height / 2);
    assert_eq!(center[3], 255, "frame must be opaque");
    assert!(
        center[2] > center[0],
        "encoded color was blue-dominant; got {center:?}"
    );
}

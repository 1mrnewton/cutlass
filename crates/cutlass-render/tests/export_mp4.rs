//! End-to-end: resolve → GPU render → native encode → decode back. Apple-only,
//! since that's where the native encoder currently exists.

#![cfg(target_vendor = "apple")]

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use cutlass_compositor::{
    CompositeLayer, Compositor, CompositorConfig, GpuContext, LayerPlacement,
};
use cutlass_core::VideoDecoder;
use cutlass_decoder::{AvfDecoder, OutputMode};
use cutlass_models::{
    CanvasAspect, CanvasSettings, ChromaKey, Generator, Mask, MaskKind, MediaSource, Project,
    Rational, TextStyle, TimeRange, TrackKind,
};
use cutlass_render::{
    ExportSettings, RenderError, Renderer, export_to_file, export_to_file_observed,
};

fn write_solid_png(path: &Path, width: u32, height: u32, rgba: [u8; 4]) {
    let file = std::fs::File::create(path).expect("create png");
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("png header");
    let flat: Vec<u8> = rgba
        .iter()
        .copied()
        .cycle()
        .take((width * height * 4) as usize)
        .collect();
    writer.write_image_data(&flat).expect("png data");
}

fn decode_mp4_corner(path: &Path, x: u32, y: u32) -> [u8; 4] {
    let gpu = GpuContext::new_headless_blocking().expect("gpu");
    let mut decoder = AvfDecoder::open(path, OutputMode::Cpu).expect("open export");
    let frame = decoder.next_frame().expect("decode").expect("frame");
    let mut comp = Compositor::new(&gpu);
    let config = CompositorConfig::new(frame.width(), frame.height());
    let layer = CompositeLayer::frame(&frame, LayerPlacement::full_canvas(&config));
    let img = comp
        .render(&gpu, &config, &[layer])
        .expect("composite decoded");
    img.pixel(x, y)
}

fn temp_mp4() -> PathBuf {
    // A timestamp alone collides: macOS SystemTime ticks in microseconds and
    // parallel tests reach here in the same one. The counter disambiguates.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("cutlass_render_export_{nanos}_{seq}.mp4"))
}

/// A quick 3-frame generated timeline (solid + title) on a 16:9 canvas.
fn test_project() -> Project {
    let fps = Rational::FPS_30;
    let span = TimeRange::at_rate(0, 3, fps);

    let mut project = Project::new("export-test", fps);
    project.timeline_mut().set_canvas(CanvasSettings {
        aspect: CanvasAspect::Wide16x9,
        background: [10, 10, 16],
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

    let titles = project.add_track(TrackKind::Text, "Titles");
    project
        .add_generated(
            titles,
            Generator::Text {
                content: "Hi".into(),
                style: TextStyle {
                    size: 160.0.into(),
                    ..TextStyle::default()
                },
            },
            span,
        )
        .unwrap();
    project
}

#[test]
fn export_with_mask_and_chroma_differs_from_unmasked() {
    let dir = tempfile::tempdir().unwrap();
    let png_path = dir.path().join("green.png");
    write_solid_png(&png_path, 1920, 1080, [0, 255, 0, 255]);

    let fps = Rational::FPS_30;

    let mut unmasked = Project::new("unmasked", fps);
    unmasked.timeline_mut().set_canvas(CanvasSettings {
        aspect: CanvasAspect::Wide16x9,
        background: [0, 0, 0],
    });
    let media = unmasked.add_media(MediaSource::image(&png_path, 1920, 1080));
    let window = unmasked.media(media).unwrap().full_range();
    let track = unmasked.add_track(TrackKind::Video, "V1");
    unmasked
        .add_clip(
            track,
            media,
            window,
            cutlass_core::RationalTime::new(0, fps),
        )
        .unwrap();

    let mut masked = Project::new("masked", fps);
    masked.timeline_mut().set_canvas(CanvasSettings {
        aspect: CanvasAspect::Wide16x9,
        background: [0, 0, 0],
    });
    let media2 = masked.add_media(MediaSource::image(&png_path, 1920, 1080));
    let window2 = masked.media(media2).unwrap().full_range();
    let track2 = masked.add_track(TrackKind::Video, "V1");
    let clip = masked
        .add_clip(
            track2,
            media2,
            window2,
            cutlass_core::RationalTime::new(0, fps),
        )
        .unwrap();
    masked
        .set_clip_mask(clip, Some(Mask::new(MaskKind::Circle)))
        .unwrap();
    masked
        .set_clip_chroma_key(
            clip,
            Some(ChromaKey {
                rgb: [0, 255, 0],
                strength: 0.5.into(),
                shadow: 0.0.into(),
            }),
        )
        .unwrap();

    let mut renderer = Renderer::new_headless().expect("headless renderer");
    let unmasked_path = temp_mp4();
    let masked_path = temp_mp4();
    export_to_file(&mut renderer, &unmasked, &unmasked_path).expect("export unmasked");
    export_to_file(&mut renderer, &masked, &masked_path).expect("export masked");

    let corner_unmasked = decode_mp4_corner(&unmasked_path, 10, 10);
    let corner_masked = decode_mp4_corner(&masked_path, 10, 10);
    assert!(
        corner_unmasked[1] > 200,
        "unmasked corner should stay green, got {corner_unmasked:?}"
    );
    assert!(
        corner_masked[1] < 32,
        "masked corner should be background, got {corner_masked:?}"
    );
    assert_ne!(
        corner_unmasked, corner_masked,
        "mask+chroma export must differ from unmasked at corners"
    );

    let _ = std::fs::remove_file(&unmasked_path);
    let _ = std::fs::remove_file(&masked_path);
}

#[test]
fn exports_a_timeline_to_a_decodable_mp4() {
    let project = test_project();
    let mut renderer = Renderer::new_headless().expect("headless renderer");
    let path = temp_mp4();
    std::fs::write(&path, b"previous export").expect("seed existing destination");
    let frames = export_to_file(&mut renderer, &project, &path).expect("export mp4");
    assert_eq!(frames, 3);

    let meta = std::fs::metadata(&path).expect("output exists");
    assert!(meta.len() > 0, "encoded file is empty");

    let mut decoder = AvfDecoder::open(&path, OutputMode::Cpu).expect("reopen output");
    assert_eq!(decoder.info().display_size, (1920, 1080));
    let frame = decoder.next_frame().expect("decode").expect("a frame");
    assert_eq!((frame.width(), frame.height()), (1920, 1080));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn observed_export_scales_output_and_reports_progress() {
    // The desktop/mobile export-job path: explicit settings + observer.
    let project = test_project();
    let mut renderer = Renderer::new_headless().expect("headless renderer");
    let path = temp_mp4();

    let mut calls: Vec<(u64, u64)> = Vec::new();
    let settings = ExportSettings {
        size: (1280, 720),
        frame_rate: Rational::FPS_30,
    };
    let frames = export_to_file_observed(&mut renderer, &project, &path, settings, &mut |d, t| {
        calls.push((d, t));
        true
    })
    .expect("export mp4");
    assert_eq!(frames, 3);

    // The observer announces the total up front and lands on (total, total).
    assert_eq!(calls.first(), Some(&(0, 3)));
    assert_eq!(calls.last(), Some(&(3, 3)));

    let decoder = AvfDecoder::open(&path, OutputMode::Cpu).expect("reopen output");
    assert_eq!(decoder.info().display_size, (1280, 720));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn cancelled_export_returns_cancelled() {
    let project = test_project();
    let mut renderer = Renderer::new_headless().expect("headless renderer");
    let path = temp_mp4();
    let previous = b"previous export";
    std::fs::write(&path, previous).expect("seed existing destination");

    let settings = ExportSettings::for_project(&project);
    let result = export_to_file_observed(&mut renderer, &project, &path, settings, &mut |d, _| {
        d < 1 // stop after the first frame is in
    });
    assert!(matches!(result, Err(RenderError::Cancelled)));
    assert_eq!(
        std::fs::read(&path).expect("read preserved destination"),
        previous
    );

    let _ = std::fs::remove_file(&path);
}

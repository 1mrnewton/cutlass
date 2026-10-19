//! End-to-end proof that the engine can render a real video frame through the
//! platform-native (FFmpeg-free) decode path: synthesize an mp4 with the native
//! encoder, then drive `Engine::get_frame` with the native decoder backend.
//!
//! Only built when the engine is compiled with native decode
//! (`--features native-decode`, or implicitly on iOS/Android). On the desktop
//! default build this file compiles to nothing.

#![cfg(engine_native_decode)]

use std::time::{SystemTime, UNIX_EPOCH};

use cutlass_core::{
    ColorSpace, CpuImage, EncoderConfig, FrameData, PixelFormat, Plane, Rect, Rotation, VideoEncoder,
};
use cutlass_engine::{Engine, EngineConfig};
use cutlass_models::{MediaSource, Project, Rational, RationalTime, TimeRange, TrackKind};

fn temp_mp4(tag: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("cutlass_engine_native_{tag}_{nanos}.mp4"))
}

/// Write `frames` solid-blue RGBA frames at `w`x`h`, 30fps, into an mp4 via the
/// native encoder.
fn synthesize_mp4(path: &std::path::Path, w: u32, h: u32, frames: i64) {
    let core_rate = cutlass_core::Rational::FPS_30;
    let config = EncoderConfig::new((w, h), core_rate, ColorSpace::BT709);
    let mut encoder = cutlass_encoder::AvfEncoder::open(path, config).expect("open native encoder");
    for i in 0..frames {
        let mut pixels = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            pixels.extend_from_slice(&[20, 120, 220, 255]); // opaque blue, RGBA
        }
        let plane = Plane::new(pixels, w as usize * 4, h as usize);
        let frame = cutlass_core::VideoFrame::new(
            cutlass_core::RationalTime::new(i, core_rate),
            PixelFormat::Rgba8,
            ColorSpace::BT709,
            (w, h),
            Rect::from_size(w, h),
            Rotation::None,
            FrameData::Cpu(CpuImage::new(vec![plane])),
        );
        encoder.push(&frame).expect("push frame");
    }
    encoder.finish().expect("finish encode");
}

#[test]
fn engine_renders_a_frame_through_the_native_decoder() {
    let (w, h) = (320u32, 240u32);
    let frames = 6i64;
    let path = temp_mp4("render");
    synthesize_mp4(&path, w, h, frames);

    let fps = Rational::FPS_30;
    let mut project = Project::new("native-decode-test", fps);
    let media = project.add_media(MediaSource::new(&path, w, h, fps, frames, false));
    let track = project.add_track(TrackKind::Video, "Video");
    project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(0, frames, fps),
            RationalTime::new(0, fps),
        )
        .expect("add clip");

    let tmp = tempfile::tempdir().expect("temp cache dir");
    let config = EngineConfig {
        cache_dir: tmp.path().to_path_buf(),
        ..EngineConfig::default()
    };
    let mut engine = Engine::with_project(config, project).expect("engine");

    let frame = engine
        .get_frame(RationalTime::new(0, fps))
        .expect("render frame through native decoder");

    assert!(frame.width > 0 && frame.height > 0, "non-empty canvas");
    assert_eq!(
        frame.bytes.len(),
        frame.width as usize * frame.height as usize * 4,
        "RGBA byte count matches dimensions"
    );
    // The source is solid blue, so the composited canvas can't be all-zero.
    assert!(
        frame.bytes.iter().any(|&b| b != 0),
        "rendered frame should carry the decoded blue content"
    );

    let _ = std::fs::remove_file(&path);
}

//! End-to-end: resolve → GPU render → native encode → decode back. Apple-only,
//! since that's where the native encoder currently exists.

#![cfg(target_vendor = "apple")]

use std::time::{SystemTime, UNIX_EPOCH};

use cutlass_core::VideoDecoder;
use cutlass_decoder::{AvfDecoder, OutputMode};
use cutlass_models::{
    CanvasAspect, CanvasSettings, Generator, Project, Rational, TextStyle, TimeRange, TrackKind,
};
use cutlass_render::{Renderer, export_to_file};

fn temp_mp4() -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("cutlass_render_export_{nanos}.mp4"))
}

#[test]
fn exports_a_timeline_to_a_decodable_mp4() {
    let fps = Rational::FPS_30;
    let span = TimeRange::at_rate(0, 3, fps); // keep it quick: 3 frames

    let mut project = Project::new("export-test", fps);
    project.timeline_mut().set_canvas(CanvasSettings {
        aspect: CanvasAspect::Wide16x9,
        background: [10, 10, 16],
    });

    let bg = project.add_track(TrackKind::Sticker, "BG");
    project
        .add_generated(bg, Generator::SolidColor { rgba: [40, 80, 160, 255] }, span)
        .unwrap();

    let titles = project.add_track(TrackKind::Text, "Titles");
    project
        .add_generated(
            titles,
            Generator::Text {
                content: "Hi".into(),
                style: TextStyle { size: 160.0, ..TextStyle::default() },
            },
            span,
        )
        .unwrap();

    let mut renderer = Renderer::new_headless().expect("headless renderer");
    let path = temp_mp4();
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

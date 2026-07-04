//! End-to-end export: composite a short timeline to a PNG sequence on a
//! headless GPU. Skips cleanly when no GPU adapter is available (CI).

use cutlass_models::{Generator, Project, Rational, TimeRange, TrackKind};
use cutlass_render::{
    ExportSettings, PngSequenceEncoder, Renderer, RenderError, export, export_config,
    export_observed,
};

const FPS_24: Rational = Rational::FPS_24;

/// A 4-frame solid timeline: enough frames to observe progress and cancel
/// mid-way, cheap enough to composite in a blink.
fn solid_project(frames: i64) -> Project {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [200, 40, 40, 255],
            },
            TimeRange::at_rate(0, frames, FPS_24),
        )
        .unwrap();
    project
}

#[test]
fn exports_a_solid_timeline_to_a_png_sequence() {
    let mut project = Project::new("p", FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    // A 3-frame timeline keeps the GPU loop quick.
    project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [0, 128, 255, 255],
            },
            TimeRange::at_rate(0, 3, FPS_24),
        )
        .unwrap();

    // The export config reflects the resolved canvas and rate regardless of GPU.
    let config = export_config(&project);
    assert_eq!(config.size, (1920, 1080));
    assert_eq!(config.frame_rate, FPS_24);

    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };

    let dir = tempfile::tempdir().unwrap();
    let mut encoder = PngSequenceEncoder::new(dir.path()).unwrap();
    let frames = export(&mut renderer, &project, &mut encoder).unwrap();

    assert_eq!(frames, 3);
    assert_eq!(encoder.frames_written(), 3);
    for i in 0..3 {
        let path = dir.path().join(format!("frame_{i:05}.png"));
        assert!(path.exists(), "missing {}", path.display());
        // The file should be a decodable 1920×1080 PNG.
        let decoder = png::Decoder::new(std::fs::File::open(&path).unwrap());
        let reader = decoder.read_info().unwrap();
        let info = reader.info();
        assert_eq!((info.width, info.height), (1920, 1080));
    }
}

#[test]
fn observed_export_reports_progress_and_completes() {
    let project = solid_project(4);
    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };

    let dir = tempfile::tempdir().unwrap();
    let mut encoder = PngSequenceEncoder::new(dir.path()).unwrap();
    let mut seen: Vec<(u64, u64)> = Vec::new();
    let frames = export_observed(
        &mut renderer,
        &project,
        &mut encoder,
        ExportSettings::for_project(&project),
        &mut |done, total| {
            seen.push((done, total));
            true
        },
    )
    .unwrap();

    assert_eq!(frames, 4);
    // One call before each frame plus the final completion tick.
    assert_eq!(
        seen,
        vec![(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)],
        "observer sees monotonically increasing progress"
    );
}

#[test]
fn observed_export_cancels_between_frames() {
    let project = solid_project(4);
    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };

    let dir = tempfile::tempdir().unwrap();
    let mut encoder = PngSequenceEncoder::new(dir.path()).unwrap();
    let result = export_observed(
        &mut renderer,
        &project,
        &mut encoder,
        ExportSettings::for_project(&project),
        &mut |done, _| done < 2,
    );

    assert!(matches!(result, Err(RenderError::Cancelled)));
    // Frames 0 and 1 were pushed before the cancel landed; nothing after.
    assert_eq!(encoder.frames_written(), 2);
}

#[test]
fn observed_export_honors_size_and_rate_overrides() {
    let project = solid_project(4); // 4 frames @ 24 = 1/6 s
    let Ok(mut renderer) = Renderer::new_headless() else {
        eprintln!("skipping: no headless GPU available");
        return;
    };

    // Half rate over the same wall-clock span → 2 frames; quarter-size output.
    let settings = ExportSettings {
        size: (480, 270),
        frame_rate: Rational::new(12, 1),
    };
    let dir = tempfile::tempdir().unwrap();
    let mut encoder = PngSequenceEncoder::new(dir.path()).unwrap();
    let frames = export_observed(
        &mut renderer,
        &project,
        &mut encoder,
        settings,
        &mut |_, _| true,
    )
    .unwrap();

    assert_eq!(frames, 2);
    let decoder = png::Decoder::new(std::fs::File::open(dir.path().join("frame_00000.png")).unwrap());
    let reader = decoder.read_info().unwrap();
    let info = reader.info();
    assert_eq!((info.width, info.height), (480, 270));
}

#[test]
fn export_settings_even_rounding() {
    let settings = ExportSettings {
        size: (481, 271),
        frame_rate: FPS_24,
    }
    .evened();
    assert_eq!(settings.size, (480, 270));

    let tiny = ExportSettings {
        size: (1, 0),
        frame_rate: FPS_24,
    }
    .evened();
    assert_eq!(tiny.size, (2, 2));
}

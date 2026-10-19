//! End-to-end export: composite a short timeline to a PNG sequence on a
//! headless GPU. Skips cleanly when no GPU adapter is available (CI).

use cutlass_models::{Generator, Project, Rational, TimeRange, TrackKind};
use cutlass_render::{PngSequenceEncoder, Renderer, export, export_config};

const FPS_24: Rational = Rational::FPS_24;

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

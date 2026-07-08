//! Export hot path: composite + H.264 encode for short timelines.
//!
//! Run: `cargo bench -p cutlass-engine --bench export`

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use cutlass_commands::{Command, EditCommand};
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig};
use cutlass_models::{Generator, Rational, TimeRange, TrackKind};

fn tr(start: i64, duration: i64) -> TimeRange {
    TimeRange::at_rate(start, duration, Rational::FPS_24)
}

fn engine_with_solid(frames: i64) -> (tempfile::TempDir, Engine) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = EngineConfig {
        cache_dir: dir.path().join("cache"),
        cache_budget_bytes: 512 * 1024 * 1024,
        undo_limit: 8,
        ..Default::default()
    };
    let mut engine = Engine::new(config).expect("engine");
    let track = match engine
        .apply(Command::Edit(EditCommand::AddTrack {
            kind: TrackKind::Video,
            name: "V1".into(),
            index: None,
            pinned: false,
        }))
        .expect("track")
    {
        ApplyOutcome::Edited(cutlass_commands::EditOutcome::CreatedTrack(id)) => id,
        o => panic!("{o:?}"),
    };
    engine
        .apply(Command::Edit(EditCommand::AddGenerated {
            track,
            generator: Generator::SolidColor {
                rgba: [30, 60, 90, 255],
            },
            timeline: tr(0, frames),
        }))
        .expect("clip");
    (dir, engine)
}

fn bench_export_solid(c: &mut Criterion) {
    const FRAMES: i64 = 48;
    let bytes_per_frame = 1920u64 * 1080 * 4;

    let mut group = c.benchmark_group("export/timeline");
    group.throughput(Throughput::Bytes(bytes_per_frame * FRAMES as u64));
    group.warm_up_time(std::time::Duration::from_secs(2));
    group.measurement_time(std::time::Duration::from_secs(8));

    group.bench_function("solid_48f_1080p_mp4", |b| {
        b.iter(|| {
            let (dir, mut engine) = engine_with_solid(FRAMES);
            let out = dir.path().join("bench_export.mp4");
            let _ = std::fs::remove_file(&out);
            engine
                .apply(cutlass_commands::Command::Project(
                    cutlass_commands::ProjectCommand::Export { path: out },
                ))
                .expect("export");
        });
    });

    group.finish();
}

criterion_group!(benches, bench_export_solid);
criterion_main!(benches);

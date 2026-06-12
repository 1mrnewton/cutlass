//! Engine preview hot path: `get_frame` (solid + optional media asset).
//!
//! Run:
//!   `cargo bench -p cutlass-engine --bench preview`
//!   `CUTLASS_BENCH_ASSET=assets/foo.mp4 cargo bench -p cutlass-engine --bench preview`

use std::path::{Path, PathBuf};

use cutlass_commands::{Command, EditCommand, ProjectCommand};
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig};
use cutlass_models::{
    ClipParam, Easing, Generator, ParamValue, Rational, RationalTime, TimeRange, TrackKind,
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets")
}

fn bench_asset() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CUTLASS_BENCH_ASSET") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    let dir = assets_dir();
    for name in [
        "15531444_1920_1080_24fps.mp4",
        "6137050-hd_1920_1080_24fps.mp4",
    ] {
        let path = dir.join(name);
        if path.exists() {
            return Some(path);
        }
    }
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|ext| ext == "mp4"))
}

fn rt(tick: i64) -> RationalTime {
    RationalTime::new(tick, Rational::FPS_24)
}

fn tr(start: i64, duration: i64) -> TimeRange {
    TimeRange::at_rate(start, duration, Rational::FPS_24)
}

fn engine_with_solid_clip(frames: i64) -> (tempfile::TempDir, Engine, cutlass_models::ClipId) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = EngineConfig {
        cache_dir: dir.path().join("cache"),
        cache_budget_bytes: 512 * 1024 * 1024,
        undo_limit: 8,
        ..Default::default()
    };
    let mut engine = Engine::new(config).expect("engine");
    // Solid generators live on sticker lanes (lane typing).
    let track = match engine
        .apply(Command::Edit(EditCommand::AddTrack {
            kind: TrackKind::Sticker,
            name: "ST1".into(),
            index: None,
        }))
        .expect("track")
    {
        ApplyOutcome::Edited(cutlass_commands::EditOutcome::CreatedTrack(id)) => id,
        o => panic!("{o:?}"),
    };
    let clip = match engine
        .apply(Command::Edit(EditCommand::AddGenerated {
            track,
            generator: Generator::SolidColor {
                rgba: [40, 80, 120, 255],
            },
            timeline: tr(0, frames),
        }))
        .expect("clip")
    {
        ApplyOutcome::Edited(cutlass_commands::EditOutcome::Created(id)) => id,
        o => panic!("{o:?}"),
    };
    (dir, engine, clip)
}

fn engine_with_media(path: &Path, source_frames: i64) -> (tempfile::TempDir, Engine) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = EngineConfig {
        cache_dir: dir.path().join("cache"),
        cache_budget_bytes: 512 * 1024 * 1024,
        undo_limit: 8,
        ..Default::default()
    };
    let mut engine = Engine::new(config).expect("engine");
    let media = match engine
        .apply(Command::Project(ProjectCommand::Import {
            path: path.to_path_buf(),
        }))
        .expect("import")
    {
        ApplyOutcome::Imported { media } => media,
        o => panic!("{o:?}"),
    };
    let track = match engine
        .apply(Command::Edit(EditCommand::AddTrack {
            kind: TrackKind::Video,
            name: "V1".into(),
            index: None,
        }))
        .expect("track")
    {
        ApplyOutcome::Edited(cutlass_commands::EditOutcome::CreatedTrack(id)) => id,
        o => panic!("{o:?}"),
    };
    engine
        .apply(Command::Edit(EditCommand::AddClip {
            track,
            media,
            source: tr(0, source_frames),
            start: rt(0),
        }))
        .expect("clip");
    (dir, engine)
}

fn bench_get_frame_solid(c: &mut Criterion) {
    let (_dir, mut engine, clip) = engine_with_solid_clip(120);
    let (w, h) = (1920u32, 1080u32);
    let bytes = (w as u64) * (h as u64) * 4;

    let mut group = c.benchmark_group("preview/get_frame");
    group.throughput(Throughput::Bytes(bytes));
    group.bench_function("solid_1080p_warm", |b| {
        b.iter(|| engine.get_frame(rt(0)).expect("frame"));
    });

    // Same frame with keyframed transform params: guards the marginal cost
    // of M2 param sampling on the per-frame hot path (binary search + eased
    // lerp per property — should be noise next to composite + readback).
    for (param, a, b_) in [
        (ClipParam::Opacity, 0.2, 1.0),
        (ClipParam::Scale, 0.5, 1.0),
        (ClipParam::Rotation, 0.0, 180.0),
    ] {
        for (tick, value) in [(0i64, a), (119, b_)] {
            engine
                .apply(Command::Edit(EditCommand::SetParamKeyframe {
                    clip,
                    param,
                    at: rt(tick),
                    value: ParamValue::Scalar(value),
                    easing: Easing::EaseInOut,
                }))
                .expect("keyframe");
        }
    }
    group.bench_function("solid_1080p_animated_warm", |b| {
        b.iter(|| engine.get_frame(rt(60)).expect("frame"));
    });
    group.finish();
}

fn bench_get_frame_media(c: &mut Criterion) {
    let Some(path) = bench_asset() else {
        eprintln!("preview bench: no CUTLASS_BENCH_ASSET or assets/*.mp4, skipping media cases");
        return;
    };
    let (_dir, mut engine) = engine_with_media(&path, 120);
    let media = engine
        .project()
        .media_iter()
        .next()
        .expect("media");
    let bytes = (media.width as u64) * (media.height as u64) * 4;

    let mut group = c.benchmark_group("preview/get_frame");
    group.throughput(Throughput::Bytes(bytes));
    group.sample_size(20);

    group.bench_function("media_cold_tick0", |b| {
        b.iter(|| {
            let (_d, mut e) = engine_with_media(&path, 120);
            e.get_frame(rt(0)).expect("frame")
        });
    });

    // Warm cache: same timeline tick, repeated decode skipped via disk cache.
    engine.get_frame(rt(0)).expect("prime");
    group.bench_function("media_warm_tick0", |b| {
        b.iter(|| engine.get_frame(rt(0)).expect("frame"));
    });

    group.finish();
}

criterion_group!(benches, bench_get_frame_solid, bench_get_frame_media);
criterion_main!(benches);

//! Engine preview hot path: `get_frame` (solid + optional media asset).
//!
//! Run:
//!   `cargo bench -p cutlass-engine --bench preview`
//!   `CUTLASS_BENCH_ASSET=local-assets/assets/foo.mp4 cargo bench -p cutlass-engine --bench preview`

use std::path::{Path, PathBuf};
use std::time::Instant;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use cutlass_commands::{Command, EditCommand, ProjectCommand};
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig};
use cutlass_models::{
    ClipParam, Easing, Generator, ParamValue, Rational, RationalTime, TimeRange, TrackKind,
};

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../local-assets/assets")
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
            pinned: false,
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
            pinned: false,
        }))
        .expect("track")
    {
        ApplyOutcome::Edited(cutlass_commands::EditOutcome::CreatedTrack(id)) => id,
        o => panic!("{o:?}"),
    };
    // The source window is expressed at the media's own frame rate (which may
    // not be 24); the timeline start stays at the timeline rate. Clamp the
    // requested span to what the asset actually holds so short clips don't
    // run off the end of the source.
    let (media_rate, avail) = {
        let m = engine.project().media(media).expect("media");
        (m.frame_rate, m.duration.value)
    };
    let frames = source_frames.min(avail.saturating_sub(1)).max(1);
    let tl_rate = engine.project().timeline().frame_rate;
    engine
        .apply(Command::Edit(EditCommand::AddClip {
            track,
            media,
            source: TimeRange::at_rate(0, frames, media_rate),
            start: RationalTime::new(0, tl_rate),
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
        eprintln!(
            "preview bench: no CUTLASS_BENCH_ASSET or local-assets/assets/*.mp4, skipping media cases"
        );
        return;
    };
    let (_dir, engine) = engine_with_media(&path, 120);
    let media = engine.project().media_iter().next().expect("media");
    let bytes = (media.width as u64) * (media.height as u64) * 4;
    let rate = engine.project().timeline().frame_rate;
    let at0 = RationalTime::new(0, rate);

    let mut group = c.benchmark_group("preview/get_frame");
    group.throughput(Throughput::Bytes(bytes));
    group.sample_size(20);

    group.bench_function("media_cold_tick0", |b| {
        b.iter(|| {
            let (_d, mut e) = engine_with_media(&path, 120);
            e.get_frame(at0).expect("frame")
        });
    });

    // Warm cache: same timeline tick, repeated decode skipped via disk cache.
    // Isolates unpack + GPU upload + composite + RGBA readback at the preview
    // resolution (PREVIEW_MAX_HEIGHT) — no decode, no cache write.
    let (_d, mut e) = engine_with_media(&path, 120);
    let r = e.project().timeline().frame_rate;
    let prime = RationalTime::new(0, r);
    let warm = e.get_frame(prime).expect("prime");
    let (pw, ph) = (warm.width, warm.height);
    group.bench_function(format!("media_warm_{pw}x{ph}"), |b| {
        b.iter(|| e.get_frame(prime).expect("frame"));
    });

    group.finish();
}

/// Sequential playback: sweep consecutive timeline ticks, each a cache miss,
/// the way pressing play does (decode → pack/cache-write → GPU composite →
/// RGBA readback per frame). This is the number that decides whether playback
/// keeps cadence: `elem/s` reads as the sustainable preview FPS. With a clip
/// span far larger than the cache budget, revisited ticks have long since been
/// evicted, so every frame stays a real miss.
fn bench_playback_media(c: &mut Criterion) {
    let Some(path) = bench_asset() else {
        eprintln!("playback bench: no asset, skipping");
        return;
    };
    // A long source span so the sweep never lives entirely inside the cache.
    let span = 600;

    let mut group = c.benchmark_group("preview/playback");
    group.throughput(Throughput::Elements(1));
    group.sample_size(10);

    // Preview composites at PREVIEW_MAX_HEIGHT (decode stays native, then a
    // single downscale before the cache). The label carries both the source
    // resolution and what the preview actually composites/reads back, so
    // flipping the constant produces directly comparable runs.
    let (_dir, mut engine) = engine_with_media(&path, span);
    let media = engine.project().media_iter().next().expect("media");
    let (sw, sh) = (media.width, media.height);
    let rate = engine.project().timeline().frame_rate;
    let total = engine.project().timeline().duration().value.max(1);
    let frame = engine.get_frame(RationalTime::new(0, rate)).expect("frame");
    let (pw, ph) = (frame.width, frame.height);

    let mut next: i64 = 0;
    group.bench_function(format!("media_{sw}x{sh}_preview_{pw}x{ph}"), |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let tick = next % total;
                next += 1;
                let _ = engine
                    .get_frame(RationalTime::new(tick, rate))
                    .expect("frame");
            }
            start.elapsed()
        });
    });
    group.finish();
}

/// Sequential playback the way the preview worker actually drives it:
/// render the playhead tick, then read-ahead the next few ticks on the SAME
/// engine/decoder (mirrors `preview_worker::prefetch_ahead`). This exposes
/// decoder-position thrash that the prefetch-free `bench_playback_media`
/// can't see — if a displayed frame misses the cache after read-ahead has
/// rolled the decoder forward, the re-decode is a backward seek.
fn bench_playback_prefetch(c: &mut Criterion) {
    let Some(path) = bench_asset() else {
        eprintln!("playback_prefetch bench: no asset, skipping");
        return;
    };
    let span = 600;
    const READ_AHEAD: i64 = 4;

    let mut group = c.benchmark_group("preview/playback_prefetch");
    group.throughput(Throughput::Elements(1));
    group.sample_size(10);

    let (_dir, mut engine) = engine_with_media(&path, span);
    let rate = engine.project().timeline().frame_rate;
    let total = engine.project().timeline().duration().value.max(1);
    let frame = engine.get_frame(RationalTime::new(0, rate)).expect("frame");
    let (pw, ph) = (frame.width, frame.height);

    let mut next: i64 = 0;
    group.bench_function(format!("media_preview_{pw}x{ph}"), |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let tick = next % total;
                next += 1;
                let _ = engine
                    .get_frame(RationalTime::new(tick, rate))
                    .expect("frame");
                for ahead in 1..=READ_AHEAD {
                    let t = (tick + ahead).min(total - 1);
                    let _ = engine.prefetch(RationalTime::new(t, rate));
                }
            }
            start.elapsed()
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_get_frame_solid,
    bench_get_frame_media,
    bench_playback_media,
    bench_playback_prefetch
);
criterion_main!(benches);

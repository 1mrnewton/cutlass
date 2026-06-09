//! Shared helpers for `cutlass-engine` integration tests.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use cutlass_commands::{Command, EditCommand, EditOutcome};
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig};
use cutlass_models::{Rational, RationalTime, TimeRange, TrackId, TrackKind};

pub fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets")
}

pub fn small_video_asset() -> Option<PathBuf> {
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

pub fn temp_engine() -> (tempfile::TempDir, Engine) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = EngineConfig {
        cache_dir: dir.path().join("cache"),
        cache_budget_bytes: 64 * 1024 * 1024,
        undo_limit: 32,
    };
    let engine = Engine::new(config).expect("engine");
    (dir, engine)
}

pub fn rt(value: i64) -> RationalTime {
    RationalTime::new(value, Rational::FPS_24)
}

pub fn tr(start: i64, duration: i64) -> TimeRange {
    TimeRange::at_rate(start, duration, Rational::FPS_24)
}

pub fn add_track(engine: &mut Engine, kind: TrackKind, name: &str) -> TrackId {
    match engine
        .apply(Command::Edit(EditCommand::AddTrack {
            kind,
            name: name.into(),
        }))
        .expect("add track")
    {
        ApplyOutcome::Edited(EditOutcome::CreatedTrack(id)) => id,
        other => panic!("expected CreatedTrack, got {other:?}"),
    }
}

pub fn import_asset(engine: &mut Engine, path: &Path) -> cutlass_models::MediaId {
    use cutlass_commands::ProjectCommand;
    match engine
        .apply(Command::Project(ProjectCommand::Import {
            path: path.to_path_buf(),
        }))
        .expect("import")
    {
        ApplyOutcome::Imported { media } => media,
        other => panic!("expected import outcome, got {other:?}"),
    }
}

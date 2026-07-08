//! Shared helpers for `cutlass-engine` integration tests.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use cutlass_commands::{Command, EditCommand, EditOutcome, ProjectCommand};
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig};
use cutlass_models::{
    ClipId, Generator, MediaId, Rational, RationalTime, TimeRange, TrackId, TrackKind,
};

pub fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../local-assets/assets")
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

pub fn image_asset() -> Option<PathBuf> {
    let path = assets_dir().join("texture.png");
    path.exists().then_some(path)
}

pub fn temp_engine() -> (tempfile::TempDir, Engine) {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = Engine::new(EngineConfig { undo_limit: 32 }).expect("engine");
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
            index: None,
            pinned: false,
        }))
        .expect("add track")
    {
        ApplyOutcome::Edited(EditOutcome::CreatedTrack(id)) => id,
        other => panic!("expected CreatedTrack, got {other:?}"),
    }
}

pub fn import_asset(engine: &mut Engine, path: &Path) -> MediaId {
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

pub fn remove_media(engine: &mut Engine, media: MediaId) {
    match engine
        .apply(Command::Project(ProjectCommand::RemoveMedia { media }))
        .expect("remove media")
    {
        ApplyOutcome::RemovedMedia { media: removed } => assert_eq!(removed, media),
        other => panic!("expected RemovedMedia, got {other:?}"),
    }
}

pub fn created_clip(outcome: ApplyOutcome) -> ClipId {
    match outcome {
        ApplyOutcome::Edited(EditOutcome::Created(id)) => id,
        other => panic!("expected Created, got {other:?}"),
    }
}

/// Track kind that accepts clips produced by `generator`.
pub fn track_kind_for(generator: &Generator) -> TrackKind {
    TrackKind::for_generator(generator).expect("generator maps to a track kind")
}

pub fn add_track_for_generator(engine: &mut Engine, generator: &Generator, name: &str) -> TrackId {
    add_track(engine, track_kind_for(generator), name)
}

pub fn add_generated(
    engine: &mut Engine,
    track: TrackId,
    generator: Generator,
    timeline: TimeRange,
) -> ClipId {
    created_clip(
        engine
            .apply(Command::Edit(EditCommand::AddGenerated {
                track,
                generator,
                timeline,
            }))
            .expect("add generated"),
    )
}

pub fn add_media_clip(
    engine: &mut Engine,
    track: TrackId,
    media: MediaId,
    source: TimeRange,
    start: RationalTime,
) -> ClipId {
    created_clip(
        engine
            .apply(Command::Edit(EditCommand::AddClip {
                track,
                media,
                source,
                start,
            }))
            .expect("add clip"),
    )
}

pub fn export_to(engine: &mut Engine, path: &Path) -> u64 {
    match engine
        .apply(Command::Project(ProjectCommand::Export {
            path: path.to_path_buf(),
        }))
        .expect("export command")
    {
        ApplyOutcome::Exported { frames } => frames,
        other => panic!("expected Exported, got {other:?}"),
    }
}

pub fn save_project(engine: &mut Engine, path: &Path) {
    match engine
        .apply(Command::Project(ProjectCommand::Save {
            path: path.to_path_buf(),
        }))
        .expect("save")
    {
        ApplyOutcome::Saved => {}
        other => panic!("expected Saved, got {other:?}"),
    }
}

pub fn open_project(engine: &mut Engine, path: &Path) {
    match engine
        .apply(Command::Project(ProjectCommand::Open {
            path: path.to_path_buf(),
        }))
        .expect("open")
    {
        ApplyOutcome::Opened => {}
        other => panic!("expected Opened, got {other:?}"),
    }
}

pub fn load_project(engine: &mut Engine, path: &Path) {
    match engine
        .apply(Command::Project(ProjectCommand::Load {
            path: path.to_path_buf(),
        }))
        .expect("load")
    {
        ApplyOutcome::Loaded => {}
        other => panic!("expected Loaded, got {other:?}"),
    }
}

/// All MP4 assets under `local-assets/assets/`, sorted for deterministic tests.
pub fn video_assets() -> Vec<PathBuf> {
    let dir = assets_dir();
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|ext| ext == "mp4"))
        .collect();
    paths.sort();
    paths
}

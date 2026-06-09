//! Save / open / load project file workflows.

mod common;

use std::path::PathBuf;

use common::{import_asset, rt, small_video_asset, temp_engine, tr};
use cutlass_commands::{Command, EditCommand, ProjectCommand};
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig};
use cutlass_models::{Project, Rational, TrackKind};

fn engine_config(cache_dir: PathBuf) -> EngineConfig {
    EngineConfig {
        cache_dir,
        cache_budget_bytes: 64 * 1024 * 1024,
        undo_limit: 8,
    }
}

#[test]
fn save_and_open_roundtrip_restores_session() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (dir, mut engine) = temp_engine();
    let media_id = import_asset(&mut engine, &path);
    let track = common::add_track(&mut engine, TrackKind::Video, "V1");
    engine
        .apply(Command::Edit(EditCommand::AddClip {
            track,
            media: media_id,
            source: tr(0, 48),
            start: rt(0),
        }))
        .expect("add clip");
    assert!(engine.can_undo());

    let project_file = dir.path().join("session.cutlass");
    assert!(matches!(
        engine
            .apply(Command::Project(ProjectCommand::Save {
                path: project_file.clone(),
            }))
            .expect("save"),
        ApplyOutcome::Saved
    ));
    assert_eq!(engine.project_path(), Some(&project_file));
    assert!(engine.can_undo(), "save does not push undo but keeps prior history");

    let clip_id = engine
        .project()
        .timeline()
        .tracks_ordered()
        .flat_map(|t| t.clips())
        .next()
        .expect("clip")
        .id;
    engine
        .apply(Command::Edit(EditCommand::RippleDelete { clip: clip_id }))
        .expect("clear timeline");
    assert_eq!(engine.project().timeline().clip_count(), 0);

    assert!(matches!(
        engine
            .apply(Command::Project(ProjectCommand::Open {
                path: project_file.clone(),
            }))
            .expect("open"),
        ApplyOutcome::Opened
    ));
    assert_eq!(engine.project().timeline().clip_count(), 1);
    assert_eq!(engine.project().media_count(), 1);
    assert!(!engine.can_undo(), "open clears undo history");
}

#[test]
fn open_fails_when_media_missing() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let (dir, mut engine) = temp_engine();
    import_asset(&mut engine, &path);

    let project_file = dir.path().join("offline.cutlass");
    engine
        .apply(Command::Project(ProjectCommand::Save {
            path: project_file.clone(),
        }))
        .expect("save");

    let missing = dir.path().join("gone.mp4");
    let mut offline_project = Project::new("offline", Rational::FPS_24);
    offline_project.add_media(cutlass_models::MediaSource::new(
        &missing,
        1920,
        1080,
        Rational::FPS_24,
        100,
        false,
    ));
    let mut engine2 = Engine::with_project(
        engine_config(dir.path().join("cache2")),
        offline_project,
    )
    .expect("engine");
    let offline = dir.path().join("missing_media.cutlass");
    engine2
        .apply(Command::Project(ProjectCommand::Save { path: offline.clone() }))
        .expect("save offline");

    let err = engine
        .apply(Command::Project(ProjectCommand::Open { path: offline }))
        .unwrap_err();
    assert!(format!("{err}").contains("not found"));
}

#[test]
fn load_tolerates_missing_media() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("ghost.mp4");
    let mut fixture = Project::new("ghost", Rational::FPS_24);
    fixture.add_media(cutlass_models::MediaSource::new(
        &missing,
        1280,
        720,
        Rational::FPS_24,
        48,
        false,
    ));
    fixture.add_track(TrackKind::Video, "V1");

    let mut engine = Engine::with_project(
        engine_config(dir.path().join("cache")),
        fixture,
    )
    .expect("engine");

    let project_file = dir.path().join("ghost.cutlass");
    engine
        .apply(Command::Project(ProjectCommand::Save {
            path: project_file.clone(),
        }))
        .expect("save");

    let mut engine2 = Engine::new(engine_config(dir.path().join("cache3"))).expect("engine");
    assert!(matches!(
        engine2
            .apply(Command::Project(ProjectCommand::Load {
                path: project_file,
            }))
            .expect("load"),
        ApplyOutcome::Loaded
    ));
    assert_eq!(engine2.project().media_count(), 1);
    assert_eq!(
        engine2.project().media_iter().next().unwrap().path(),
        PathBuf::from(missing).as_path()
    );
}

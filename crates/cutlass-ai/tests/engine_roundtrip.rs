//! Wire commands → validation → a real `Engine`: the full Phase 1 path.
//!
//! One engine instance for the whole scenario (engine construction spins a
//! headless GPU context). Media is a pool entry with a synthetic path —
//! edit commands never decode, so no file is needed.

use cutlass_ai::wire::{self, WireCommand, WireGenerator};
use cutlass_ai::{summarize, validate};
use cutlass_commands::{Command, EditOutcome};
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig};
use cutlass_models::{MediaSource, Project, Rational};

const R24: Rational = Rational::FPS_24;

fn temp_engine_with(project: Project) -> (tempfile::TempDir, Engine) {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = EngineConfig { undo_limit: 64 };
    let engine = Engine::with_project(config, project).expect("engine");
    (dir, engine)
}

fn apply(engine: &mut Engine, command: WireCommand) -> ApplyOutcome {
    let lowered = validate(&command, engine.project())
        .unwrap_or_else(|r| panic!("{command:?} rejected: {r}"));
    engine
        .apply(lowered)
        .unwrap_or_else(|e| panic!("{command:?} failed in engine: {e}"))
}

fn created_clip(outcome: ApplyOutcome) -> u64 {
    match outcome {
        ApplyOutcome::Edited(EditOutcome::Created(id)) => id.raw(),
        other => panic!("expected created clip, got {other:?}"),
    }
}

fn created_track(outcome: ApplyOutcome) -> u64 {
    match outcome {
        ApplyOutcome::Edited(EditOutcome::CreatedTrack(id)) => id.raw(),
        other => panic!("expected created track, got {other:?}"),
    }
}

#[test]
fn prompt_sized_scenario_round_trips_and_unwinds() {
    let mut project = Project::new("agent-fixture", R24);
    let media = project
        .add_media(MediaSource::new(
            "/tmp/agent-roundtrip.mp4",
            1920,
            1080,
            R24,
            60 * 24,
            true,
        ))
        .raw();
    let (_dir, mut engine) = temp_engine_with(project);

    // "Lay down 10s of footage, cut it at 4s, keep the tail trimmed to 4s,
    // ripple the head away, then add a styled title and link it."
    let video = created_track(apply(
        &mut engine,
        WireCommand::AddTrack(wire::AddTrack {
            kind: wire::WireTrackKind::Video,
            name: "V1".into(),
            index: None,
        }),
    ));
    let head = created_clip(apply(
        &mut engine,
        WireCommand::AddClip(wire::AddClip {
            track: video,
            media,
            source_start: 0.0,
            source_duration: 10.0,
            start: 0.0,
        }),
    ));
    let tail = created_clip(apply(
        &mut engine,
        WireCommand::SplitClip(wire::SplitClip {
            clip: head,
            at: 4.0,
        }),
    ));

    // Trim the tail from [4s, 10s) to [6s, 10s): head-trim advances the
    // source in-point by the same 2s.
    apply(
        &mut engine,
        WireCommand::TrimClip(wire::TrimClip {
            clip: tail,
            start: 6.0,
            duration: 4.0,
        }),
    );
    {
        let clip = engine
            .project()
            .clip(cutlass_models::ClipId::from_raw(tail))
            .unwrap();
        assert_eq!(clip.timeline.start.value, 144); // 6s * 24
        assert_eq!(clip.timeline.duration.value, 96); // 4s * 24
        assert_eq!(clip.source_range().unwrap().start.value, 144);
    }

    // Ripple the head away: the tail slides left by the head's 4s.
    apply(
        &mut engine,
        WireCommand::RippleDelete(wire::RippleDelete { clip: head }),
    );
    assert_eq!(
        engine
            .project()
            .clip(cutlass_models::ClipId::from_raw(tail))
            .unwrap()
            .timeline
            .start
            .value,
        48 // 6s - 4s = 2s * 24
    );

    let titles = created_track(apply(
        &mut engine,
        WireCommand::AddTrack(wire::AddTrack {
            kind: wire::WireTrackKind::Text,
            name: "Titles".into(),
            index: None,
        }),
    ));
    let title = created_clip(apply(
        &mut engine,
        WireCommand::AddGenerated(wire::AddGenerated {
            track: titles,
            generator: WireGenerator::Text {
                content: "INTRO".into(),
            },
            start: 0.0,
            duration: 3.0,
        }),
    ));
    apply(
        &mut engine,
        WireCommand::SetGenerator(wire::SetGenerator {
            clip: title,
            generator: WireGenerator::Text {
                content: "OUTRO".into(),
            },
        }),
    );
    apply(
        &mut engine,
        WireCommand::SetClipTransform(wire::SetClipTransform {
            clip: title,
            position_x: None,
            position_y: Some(0.3),
            anchor_x: None,
            anchor_y: None,
            scale: Some(0.5),
            rotation: None,
            opacity: None,
        }),
    );
    apply(
        &mut engine,
        WireCommand::LinkClips(wire::LinkClips {
            clips: vec![tail, title],
        }),
    );

    // The summary the model would see reflects all of it.
    let summary = summarize(engine.project());
    assert_eq!(summary.tracks.len(), 2);
    let v1 = &summary.tracks[0];
    assert_eq!(v1.clips.len(), 1);
    assert_eq!(v1.clips[0].id, tail);
    assert_eq!(v1.clips[0].start_seconds, 2.0);
    let t1 = &summary.tracks[1];
    assert_eq!(
        t1.clips[0].content,
        cutlass_ai::describe::ClipContent::Text {
            text: "OUTRO".into()
        }
    );
    assert_eq!(t1.clips[0].link, v1.clips[0].link);
    assert!(v1.clips[0].link.is_some());

    // Ten applied commands = ten history entries; a full unwind leaves
    // the timeline empty (every wire command is exactly as undoable as a
    // gesture).
    let mut undone = 0;
    while engine.undo() {
        undone += 1;
    }
    assert_eq!(undone, 10);
    assert_eq!(engine.project().timeline().track_count(), 0);
    assert_eq!(engine.project().timeline().clip_count(), 0);
}

#[test]
fn engine_rejections_leave_state_untouched() {
    let mut project = Project::new("agent-rejects", R24);
    let media = project
        .add_media(MediaSource::new(
            "/tmp/agent-rejects.mp4",
            1920,
            1080,
            R24,
            20 * 24,
            true,
        ))
        .raw();
    let (_dir, mut engine) = temp_engine_with(project);

    let video = created_track(apply(
        &mut engine,
        WireCommand::AddTrack(wire::AddTrack {
            kind: wire::WireTrackKind::Video,
            name: "V1".into(),
            index: None,
        }),
    ));
    created_clip(apply(
        &mut engine,
        WireCommand::AddClip(wire::AddClip {
            track: video,
            media,
            source_start: 0.0,
            source_duration: 5.0,
            start: 0.0,
        }),
    ));

    // Validation passes (overlap is the engine's call), the engine rejects,
    // and nothing changed — the loop feeds this error back to the model.
    let overlapping = validate(
        &WireCommand::AddClip(wire::AddClip {
            track: video,
            media,
            source_start: 0.0,
            source_duration: 5.0,
            start: 2.0,
        }),
        engine.project(),
    )
    .expect("overlap is not validation's call");
    let before_clips = engine.project().timeline().clip_count();
    assert!(engine.apply(overlapping).is_err());
    assert_eq!(engine.project().timeline().clip_count(), before_clips);

    // And a failed apply must not have pushed an undo entry: one undo
    // removes the clip, the next removes the track, then history is empty.
    assert!(engine.undo());
    assert!(engine.undo());
    assert!(!engine.undo());
}

#[test]
fn validate_is_pure_against_engine_state() {
    let (_dir, engine) = temp_engine_with(Project::new("empty", R24));

    let rejection = validate(
        &WireCommand::RemoveClip(wire::RemoveClip { clip: 1 }),
        engine.project(),
    )
    .unwrap_err();
    assert!(rejection.message.contains("does not exist"));

    // Lowering never mutates: the project is untouched by validation.
    assert_eq!(engine.project().timeline().track_count(), 0);

    // Round-trip a serialized plan entry (the dry-run path).
    let plan = serde_json::json!({
        "command": "add_track", "kind": "video", "name": "V1"
    });
    let wire: WireCommand = serde_json::from_value(plan).unwrap();
    let lowered = validate(&wire, engine.project()).unwrap();
    assert!(matches!(lowered, Command::Edit(_)));
}

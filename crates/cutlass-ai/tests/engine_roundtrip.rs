//! Wire commands → validation → a real `Engine`: the full Phase 1 path.
//!
//! One engine instance for the whole scenario (engine construction spins a
//! headless GPU context). Media is a pool entry with a synthetic path —
//! edit commands never decode, so no file is needed.

use cutlass_ai::wire::{self, WireCommand, WireGenerator};
use cutlass_ai::{summarize, validate};
use cutlass_commands::{Command, EditOutcome};
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig};
use cutlass_models::{
    AnimationRef, AudioRole, ClipParam, Easing, Generator, LinkId, MediaSource, ParamValue,
    Project, Rational, RationalTime, TimeRange, TrackKind,
};

const R24: Rational = Rational::FPS_24;

fn engine_with(project: Project) -> Engine {
    let config = EngineConfig { undo_limit: 64 };
    Engine::with_project(config, project).expect("engine")
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
    let mut engine = engine_with(project);

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
fn move_effect_lowers_applies_and_undoes_exactly() {
    let mut project = Project::new("agent effect order", R24);
    let track = project.add_track(TrackKind::Text, "Titles");
    let clip = project
        .add_generated(
            track,
            Generator::text("TITLE"),
            TimeRange::at_rate(0, 48, R24),
        )
        .unwrap();
    for effect in ["gaussian_blur", "glitch", "vignette"] {
        project.add_effect(clip, effect).unwrap();
    }
    project.set_effect_param(clip, 0, 0, 12.0).unwrap();
    for (tick, value, easing) in [(0, 0.2, Easing::EaseIn), (24, 0.8, Easing::EaseOut)] {
        project
            .set_param_keyframe(
                clip,
                ClipParam::Effect {
                    effect: 1,
                    param: 0,
                },
                RationalTime::new(tick, R24),
                ParamValue::Scalar(value),
                easing,
            )
            .unwrap();
    }
    project.set_effect_param(clip, 2, 0, 0.75).unwrap();
    let before = project.clip(clip).unwrap().effects.clone();
    let expected = vec![before[1].clone(), before[2].clone(), before[0].clone()];
    let mut engine = engine_with(project);

    let outcome = apply(
        &mut engine,
        WireCommand::MoveEffect(wire::MoveEffect {
            clip: clip.raw(),
            from_index: 0,
            to_index: 2,
        }),
    );
    assert!(matches!(
        outcome,
        ApplyOutcome::Edited(EditOutcome::Updated(id)) if id == clip
    ));
    assert_eq!(engine.project().clip(clip).unwrap().effects, expected);

    assert!(engine.undo());
    assert_eq!(engine.project().clip(clip).unwrap().effects, before);
    assert!(engine.redo());
    assert_eq!(engine.project().clip(clip).unwrap().effects, expected);
}

#[test]
fn extract_audio_lowers_applies_and_undoes_as_one_edit() {
    let mut project = Project::new("agent extract", R24);
    let media = project.add_media(MediaSource::new(
        "/tmp/agent-extract.mp4",
        1920,
        1080,
        R24,
        480,
        true,
    ));
    let video_track = project.add_track(TrackKind::Video, "V1");
    let audio_track = project.add_track(TrackKind::Audio, "A1");
    let video = project
        .add_clip(
            video_track,
            media,
            TimeRange::at_rate(24, 96, R24),
            RationalTime::new(48, R24),
        )
        .unwrap();
    let mut engine = engine_with(project);

    let outcome = apply(
        &mut engine,
        WireCommand::ExtractAudio(wire::ExtractAudio {
            clip: video.raw(),
            track: audio_track.raw(),
        }),
    );
    let audio = match outcome {
        ApplyOutcome::Edited(EditOutcome::Created(id)) => id,
        other => panic!("expected Created, got {other:?}"),
    };
    let snapshot = engine.project().clip(audio).unwrap().clone();
    assert_eq!(
        snapshot.timeline,
        engine.project().clip(video).unwrap().timeline
    );
    assert_eq!(snapshot.audio_role, Some(AudioRole::Extracted));
    assert_eq!(
        engine.project().timeline().track_of(audio),
        Some(audio_track)
    );
    assert!(!engine.project().timeline().carries_own_audio(video));

    assert!(engine.undo());
    assert!(engine.project().clip(audio).is_none());
    assert!(
        engine
            .project()
            .timeline()
            .track(audio_track)
            .is_some_and(|track| track.is_empty())
    );
    assert!(engine.project().timeline().carries_own_audio(video));
    assert!(
        !engine.undo(),
        "the extraction was exactly one history entry"
    );

    assert!(engine.redo());
    assert_eq!(engine.project().clip(audio).unwrap(), &snapshot);
}

#[test]
fn duplicate_clip_wire_preserves_properties_and_round_trips_one_edit() {
    let mut project = Project::new("agent duplicate", R24);
    let source_track = project.add_track(TrackKind::Sticker, "Source");
    let destination_track = project.add_track(TrackKind::Sticker, "Copies");
    let source = project
        .add_generated(
            source_track,
            Generator::SolidColor {
                rgba: [17, 34, 51, 255],
            },
            TimeRange::at_rate(0, 48, R24),
        )
        .unwrap();
    let companion = project
        .add_generated(
            source_track,
            Generator::SolidColor {
                rgba: [51, 34, 17, 255],
            },
            TimeRange::at_rate(48, 48, R24),
        )
        .unwrap();
    let link = LinkId::next();
    for id in [source, companion] {
        project.timeline_mut().clip_mut(id).unwrap().link = Some(link);
    }
    project
        .set_param_keyframe(
            source,
            ClipParam::Scale,
            RationalTime::new(24, R24),
            ParamValue::Scalar(1.5),
            Easing::EaseOut,
        )
        .unwrap();
    project.add_effect(source, "gaussian_blur").unwrap();
    project.set_effect_param(source, 0, 0, 9.0).unwrap();
    project
        .timeline_mut()
        .clip_mut(source)
        .unwrap()
        .animation_combo = Some(AnimationRef::new("pulse"));
    let source_before = project.clip(source).unwrap().clone();
    let companion_before = project.clip(companion).unwrap().clone();
    let mut engine = engine_with(project);

    let outcome = apply(
        &mut engine,
        WireCommand::DuplicateClip(wire::DuplicateClip {
            clip: source.raw(),
            to_track: destination_track.raw(),
            start: 5.0,
        }),
    );
    let duplicate = match outcome {
        ApplyOutcome::Edited(EditOutcome::Created(id)) => id,
        other => panic!("expected Created, got {other:?}"),
    };
    assert_ne!(duplicate, source);

    let duplicate_before_undo = engine.project().clip(duplicate).unwrap().clone();
    let mut expected = source_before.clone();
    expected.id = duplicate;
    expected.timeline.start = RationalTime::new(120, R24);
    expected.link = None;
    assert_eq!(duplicate_before_undo, expected);
    assert_eq!(
        engine.project().timeline().track_of(duplicate),
        Some(destination_track)
    );
    assert_eq!(engine.project().clip(source).unwrap(), &source_before);
    assert_eq!(engine.project().clip(companion).unwrap(), &companion_before);

    assert!(engine.undo());
    assert!(engine.project().clip(duplicate).is_none());
    assert_eq!(engine.project().clip(source).unwrap(), &source_before);
    assert!(
        !engine.undo(),
        "the duplicate command was exactly one history entry"
    );

    assert!(engine.redo());
    assert_eq!(
        engine.project().clip(duplicate).unwrap(),
        &duplicate_before_undo,
        "redo restores the same id and complete clip state"
    );
}

#[test]
fn unlink_one_member_dissolves_complete_group_and_undoes_once() {
    let mut project = Project::new("agent-unlink", R24);
    let track = project.add_track(TrackKind::Sticker, "Overlays");
    let clips = [
        project
            .add_generated(
                track,
                Generator::SolidColor {
                    rgba: [255, 0, 0, 255],
                },
                TimeRange::at_rate(0, 24, R24),
            )
            .unwrap(),
        project
            .add_generated(
                track,
                Generator::SolidColor {
                    rgba: [0, 255, 0, 255],
                },
                TimeRange::at_rate(24, 24, R24),
            )
            .unwrap(),
        project
            .add_generated(
                track,
                Generator::SolidColor {
                    rgba: [0, 0, 255, 255],
                },
                TimeRange::at_rate(48, 24, R24),
            )
            .unwrap(),
    ];
    let link = LinkId::next();
    for clip in clips {
        project.timeline_mut().clip_mut(clip).unwrap().link = Some(link);
    }
    let mut engine = engine_with(project);

    let outcome = apply(
        &mut engine,
        WireCommand::UnlinkClips(wire::UnlinkClips {
            clips: vec![clips[1].raw()],
        }),
    );
    assert!(matches!(
        outcome,
        ApplyOutcome::Edited(EditOutcome::Updated(id)) if id == clips[1]
    ));
    for clip in clips {
        assert_eq!(engine.project().clip(clip).unwrap().link, None);
    }

    assert!(engine.undo(), "the unlink is one history step");
    for clip in clips {
        assert_eq!(engine.project().clip(clip).unwrap().link, Some(link));
    }
    assert!(!engine.undo(), "the fixture itself created no history");

    assert!(engine.redo());
    for clip in clips {
        assert_eq!(engine.project().clip(clip).unwrap().link, None);
    }
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
    let mut engine = engine_with(project);

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
    let engine = engine_with(Project::new("empty", R24));

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

//! Production freeze-frame command integration, validation, and history.

use cutlass_commands::{Command, EditCommand, EditOutcome};
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig};
use cutlass_models::{
    AnimationRef, ClipId, Easing, Generator, Keyframe, LinkId, MediaId, MediaSource, Param,
    Project, Rational, RationalTime, TimeRange, TrackId, TrackKind,
};

const FPS: Rational = Rational::FPS_24;

fn rt(value: i64) -> RationalTime {
    RationalTime::new(value, FPS)
}

fn tr(start: i64, duration: i64) -> TimeRange {
    TimeRange::at_rate(start, duration, FPS)
}

fn engine(project: Project) -> Engine {
    Engine::with_project(EngineConfig { undo_limit: 32 }, project).expect("engine")
}

fn snapshot(project: &Project) -> Vec<u8> {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("snapshot.cutlass");
    project.save_to_file(&path).expect("save project snapshot");
    std::fs::read(path).expect("read project snapshot")
}

struct VideoFixture {
    project: Project,
    media: MediaId,
    track: TrackId,
    source: ClipId,
    next: ClipId,
}

fn video_fixture(next_start: i64) -> VideoFixture {
    let mut project = Project::new("freeze command", FPS);
    let media = project.add_media(MediaSource::new(
        "/tmp/freeze-engine.mp4",
        1920,
        1080,
        FPS,
        1_000,
        true,
    ));
    let track = project.add_track(TrackKind::Video, "V1");
    let source = project
        .add_clip(track, media, tr(100, 50), rt(10))
        .expect("source");
    let next = project
        .add_clip(track, media, tr(300, 20), rt(next_start))
        .expect("next");
    VideoFixture {
        project,
        media,
        track,
        source,
        next,
    }
}

fn apply_freeze(
    engine: &mut Engine,
    clip: ClipId,
    at: RationalTime,
    duration: RationalTime,
) -> ClipId {
    match engine
        .apply(Command::Edit(EditCommand::FreezeFrame {
            clip,
            at,
            duration,
        }))
        .expect("freeze frame")
    {
        ApplyOutcome::Edited(EditOutcome::Created(id)) => id,
        other => panic!("expected Created, got {other:?}"),
    }
}

fn track_layout(engine: &Engine, track: TrackId) -> Vec<(ClipId, i64, i64)> {
    engine
        .project()
        .timeline()
        .track(track)
        .unwrap()
        .clips_ordered()
        .into_iter()
        .map(|clip| {
            (
                clip.id,
                clip.timeline.start.value,
                clip.timeline.duration.value,
            )
        })
        .collect()
}

#[test]
fn inserts_at_start_middle_and_end_while_preserving_existing_gap() {
    for (at, expected_count) in [(10, 3), (30, 4), (60, 3)] {
        let fixture = video_fixture(80);
        let source_before = fixture.project.clip(fixture.source).unwrap().clone();
        let held_at = if at == 60 { 59 } else { at };
        let expected_source = source_before.source_time_at(rt(held_at)).unwrap().unwrap();
        let mut engine = engine(fixture.project);

        let freeze = apply_freeze(&mut engine, fixture.source, rt(at), rt(6));
        let frozen = engine.project().clip(freeze).unwrap();
        assert!(frozen.freeze_frame);
        assert_eq!(frozen.timeline, tr(at, 6));
        assert_eq!(
            frozen.source_range(),
            Some(TimeRange::at_rate(expected_source.value, 1, FPS))
        );
        for tick in at..at + 6 {
            assert_eq!(
                frozen.source_time_at(rt(tick)).unwrap(),
                Some(expected_source)
            );
        }

        let layout = track_layout(&engine, fixture.track);
        assert_eq!(layout.len(), expected_count);
        assert_eq!(
            engine.project().clip(fixture.next).unwrap().start().value,
            86,
            "the later clip shifts and the original 20-tick gap survives"
        );
        match at {
            10 => {
                assert_eq!(
                    layout,
                    vec![
                        (freeze, 10, 6),
                        (fixture.source, 16, 50),
                        (fixture.next, 86, 20),
                    ]
                );
            }
            30 => {
                let tail = layout
                    .iter()
                    .find(|(id, start, _)| {
                        *id != freeze
                            && *id != fixture.source
                            && *id != fixture.next
                            && *start == 36
                    })
                    .expect("split tail")
                    .0;
                assert_eq!(
                    layout,
                    vec![
                        (fixture.source, 10, 20),
                        (freeze, 30, 6),
                        (tail, 36, 30),
                        (fixture.next, 86, 20),
                    ]
                );
            }
            60 => {
                assert_eq!(
                    layout,
                    vec![
                        (fixture.source, 10, 50),
                        (freeze, 60, 6),
                        (fixture.next, 86, 20),
                    ]
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn middle_freeze_is_one_undo_step_and_redo_restores_stable_ids() {
    let mut fixture = video_fixture(80);
    let orphan = LinkId::next();
    fixture
        .project
        .timeline_mut()
        .clip_mut(fixture.source)
        .unwrap()
        .link = Some(orphan);
    let before = snapshot(&fixture.project);
    let mut engine = engine(fixture.project);

    let freeze = apply_freeze(&mut engine, fixture.source, rt(30), rt(7));
    let tail = track_layout(&engine, fixture.track)
        .into_iter()
        .map(|(id, _, _)| id)
        .find(|id| *id != fixture.source && *id != fixture.next && *id != freeze)
        .expect("split tail");
    let freeze_after = engine.project().clip(freeze).unwrap().clone();
    let tail_after = engine.project().clip(tail).unwrap().clone();
    let after = snapshot(engine.project());
    assert_eq!(engine.project().clip(fixture.source).unwrap().link, None);

    assert!(engine.undo());
    assert_eq!(snapshot(engine.project()), before);
    assert_eq!(
        engine.project().clip(fixture.source).unwrap().link,
        Some(orphan)
    );
    assert!(
        !engine.undo(),
        "the command records exactly one history entry"
    );

    assert!(engine.redo());
    assert_eq!(snapshot(engine.project()), after);
    assert_eq!(engine.project().clip(freeze).unwrap(), &freeze_after);
    assert_eq!(engine.project().clip(tail).unwrap(), &tail_after);
}

#[test]
fn reverse_and_speed_curve_choose_the_pre_mutation_held_frame() {
    // Constant-speed reverse.
    let mut project = Project::new("reverse freeze", FPS);
    let media = project.add_media(MediaSource::new(
        "/tmp/reverse.mp4",
        1920,
        1080,
        FPS,
        1_000,
        false,
    ));
    let track = project.add_track(TrackKind::Video, "V1");
    let source = project.add_clip(track, media, tr(100, 120), rt(0)).unwrap();
    project
        .set_clip_speed(source, Rational::new(2, 1), true)
        .unwrap();
    let at = rt(20);
    let expected = project
        .clip(source)
        .unwrap()
        .source_time_at(at)
        .unwrap()
        .unwrap();
    let mut reverse_engine = engine(project);
    let frozen = apply_freeze(&mut reverse_engine, source, at, rt(5));
    assert_eq!(
        reverse_engine
            .project()
            .clip(frozen)
            .unwrap()
            .source_range(),
        Some(TimeRange::at_rate(expected.value, 1, FPS))
    );

    // Integrated speed ramp.
    let mut project = Project::new("curve freeze", FPS);
    let media = project.add_media(MediaSource::new(
        "/tmp/curve.mp4",
        1920,
        1080,
        FPS,
        1_000,
        false,
    ));
    let track = project.add_track(TrackKind::Video, "V1");
    let source = project.add_clip(track, media, tr(200, 120), rt(0)).unwrap();
    project
        .set_clip_speed_curve(
            source,
            Some(Param::Keyframed {
                keyframes: vec![
                    Keyframe {
                        tick: 0,
                        value: 0.5,
                        easing: Easing::Linear,
                    },
                    Keyframe {
                        tick: 1_000,
                        value: 2.0,
                        easing: Easing::Linear,
                    },
                ],
            }),
        )
        .unwrap();
    let at = rt(70);
    let source_before = project.clip(source).unwrap();
    let expected = source_before.source_time_at(at).unwrap().unwrap();
    assert_ne!(
        source_before.source_time_at(rt(69)).unwrap(),
        Some(expected),
        "chosen split must advance a source frame"
    );
    let mut curve_engine = engine(project);
    let frozen = apply_freeze(&mut curve_engine, source, at, rt(8));
    assert_eq!(
        curve_engine.project().clip(frozen).unwrap().source_range(),
        Some(TimeRange::at_rate(expected.value, 1, FPS))
    );
}

#[test]
fn mixed_rate_freeze_holds_one_native_source_frame() {
    let source_rate = Rational::FPS_30;
    let mut project = Project::new("mixed-rate freeze", FPS);
    let media = project.add_media(MediaSource::new(
        "/tmp/mixed-rate.mp4",
        1920,
        1080,
        source_rate,
        300,
        false,
    ));
    let track = project.add_track(TrackKind::Video, "V1");
    let source = project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(30, 150, source_rate),
            rt(0),
        )
        .unwrap();
    let at = rt(48);
    let expected = project
        .clip(source)
        .unwrap()
        .source_time_at(at)
        .unwrap()
        .unwrap();
    let mut engine = engine(project);

    let frozen = apply_freeze(&mut engine, source, at, rt(12));
    let frozen = engine.project().clip(frozen).unwrap();

    assert_eq!(expected.rate, source_rate);
    assert_eq!(
        frozen.source_range(),
        Some(TimeRange::at_rate(expected.value, 1, source_rate))
    );
    for tick in at.value..at.value + 12 {
        assert_eq!(
            frozen.source_time_at(rt(tick)).unwrap(),
            Some(expected),
            "held frame changed at timeline tick {tick}"
        );
    }
}

#[test]
fn structural_transition_pruning_undo_and_redo_match_other_edits() {
    let mut fixture = video_fixture(60);
    fixture
        .project
        .add_transition(fixture.source, "crossfade")
        .unwrap();
    let mut engine = engine(fixture.project);

    let freeze = apply_freeze(&mut engine, fixture.source, rt(60), rt(5));
    assert!(
        engine
            .project()
            .timeline()
            .track(fixture.track)
            .unwrap()
            .transition_at(fixture.source)
            .is_none()
    );
    assert!(engine.project().clip(freeze).is_some());

    assert!(engine.undo());
    let transition = engine
        .project()
        .timeline()
        .track(fixture.track)
        .unwrap()
        .transition_at(fixture.source)
        .expect("undo restores source junction");
    assert_eq!(transition.right, fixture.next);
    assert!(engine.project().clip(freeze).is_none());

    assert!(engine.redo());
    assert!(
        engine
            .project()
            .timeline()
            .track(fixture.track)
            .unwrap()
            .transition_at(fixture.source)
            .is_none()
    );
    assert!(engine.project().clip(freeze).is_some());
}

fn freeze_command(clip: ClipId, at: RationalTime, duration: RationalTime) -> EditCommand {
    EditCommand::FreezeFrame { clip, at, duration }
}

fn assert_rejected_unchanged(project: Project, command: EditCommand, message_fragment: &str) {
    let mut engine = engine(project);
    let before = snapshot(engine.project());
    let revision = engine.revision();
    let error = engine
        .apply(Command::Edit(command))
        .expect_err("command must reject");
    assert!(
        error.to_string().contains(message_fragment),
        "expected '{message_fragment}' in '{error}'"
    );
    assert_eq!(
        snapshot(engine.project()),
        before,
        "rejected freeze mutated the project"
    );
    assert_eq!(engine.revision(), revision);
    assert!(!engine.can_undo(), "rejected freeze added history");
    assert!(!engine.can_redo());
}

#[test]
fn source_kind_link_lock_and_existing_freeze_rejections_are_atomic() {
    let mut fixture = video_fixture(80);
    let link = LinkId::next();
    for clip in [fixture.source, fixture.next] {
        fixture.project.timeline_mut().clip_mut(clip).unwrap().link = Some(link);
    }
    assert_rejected_unchanged(
        fixture.project,
        freeze_command(fixture.source, rt(30), rt(5)),
        "unlink the group",
    );

    // A later detached video/audio pair would be shifted on only the video
    // lane. Reject rather than silently desynchronizing that live group.
    let mut fixture = video_fixture(80);
    let audio_track = fixture.project.add_track(TrackKind::Audio, "A1");
    let audio = fixture
        .project
        .add_clip(audio_track, fixture.media, tr(300, 20), rt(80))
        .unwrap();
    let link = LinkId::next();
    fixture
        .project
        .timeline_mut()
        .clip_mut(fixture.next)
        .unwrap()
        .link = Some(link);
    fixture.project.timeline_mut().clip_mut(audio).unwrap().link = Some(link);
    assert_rejected_unchanged(
        fixture.project,
        freeze_command(fixture.source, rt(60), rt(5)),
        "desynchronize",
    );

    let mut fixture = video_fixture(80);
    fixture
        .project
        .timeline_mut()
        .track_mut(fixture.track)
        .unwrap()
        .locked = true;
    assert_rejected_unchanged(
        fixture.project,
        freeze_command(fixture.source, rt(30), rt(5)),
        "locked",
    );

    let mut project = Project::new("image", FPS);
    let image = project.add_media(MediaSource::image("/tmp/still.png", 1920, 1080));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project
        .add_clip(
            track,
            image,
            project.media(image).unwrap().full_range(),
            rt(0),
        )
        .unwrap();
    assert_rejected_unchanged(project, freeze_command(clip, rt(0), rt(5)), "video media");

    let mut project = Project::new("audio-only media", FPS);
    let audio = project.add_media(MediaSource::new("/tmp/audio.wav", 0, 0, FPS, 240, true));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project.add_clip(track, audio, tr(0, 40), rt(0)).unwrap();
    assert_rejected_unchanged(project, freeze_command(clip, rt(0), rt(5)), "video media");

    let mut project = Project::new("audio lane", FPS);
    let media = project.add_media(MediaSource::new(
        "/tmp/video-on-audio-lane.mp4",
        1920,
        1080,
        FPS,
        240,
        true,
    ));
    let track = project.add_track(TrackKind::Audio, "A1");
    let clip = project.add_clip(track, media, tr(0, 40), rt(0)).unwrap();
    assert_rejected_unchanged(project, freeze_command(clip, rt(0), rt(5)), "video track");

    let mut project = Project::new("generated", FPS);
    let track = project.add_track(TrackKind::Sticker, "Stickers");
    let clip = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [1, 2, 3, 255],
            },
            tr(0, 20),
        )
        .unwrap();
    assert_rejected_unchanged(project, freeze_command(clip, rt(0), rt(5)), "video track");

    let mut fixture = video_fixture(80);
    fixture
        .project
        .timeline_mut()
        .clip_mut(fixture.source)
        .unwrap()
        .freeze_frame = true;
    assert_rejected_unchanged(
        fixture.project,
        freeze_command(fixture.source, rt(30), rt(5)),
        "already a freeze frame",
    );
}

#[test]
fn time_and_edge_animation_rejections_are_atomic() {
    let fixture = video_fixture(80);
    assert_rejected_unchanged(
        fixture.project,
        freeze_command(
            fixture.source,
            rt(30),
            RationalTime::new(5, Rational::FPS_30),
        ),
        "rate mismatch",
    );

    for duration in [0, -5] {
        let fixture = video_fixture(80);
        assert_rejected_unchanged(
            fixture.project,
            freeze_command(fixture.source, rt(30), rt(duration)),
            "invalid time range",
        );
    }

    let mut fixture = video_fixture(80);
    fixture
        .project
        .timeline_mut()
        .clip_mut(fixture.source)
        .unwrap()
        .animation_in = Some(AnimationRef::new("fade_in"));
    assert_rejected_unchanged(
        fixture.project,
        freeze_command(fixture.source, rt(10), rt(5)),
        "animation",
    );

    let mut fixture = video_fixture(80);
    fixture
        .project
        .timeline_mut()
        .clip_mut(fixture.source)
        .unwrap()
        .animation_out = Some(AnimationRef::new("fade_out"));
    assert_rejected_unchanged(
        fixture.project,
        freeze_command(fixture.source, rt(60), rt(5)),
        "animation",
    );

    for at in [10, 60] {
        let mut fixture = video_fixture(80);
        fixture
            .project
            .timeline_mut()
            .clip_mut(fixture.source)
            .unwrap()
            .animation_combo = Some(AnimationRef::new("pulse"));
        assert_rejected_unchanged(
            fixture.project,
            freeze_command(fixture.source, rt(at), rt(5)),
            "animation",
        );
    }
}

#[test]
fn overlap_and_mid_split_failures_leave_project_and_history_untouched() {
    // Public clip fields can expose a corrupt imported/plugin-authored track.
    // FreezeFrame fails closed before opening a hole in that state.
    let mut fixture = video_fixture(80);
    let blocker = fixture
        .project
        .add_clip(fixture.track, fixture.media, tr(500, 5), rt(0))
        .unwrap();
    fixture
        .project
        .timeline_mut()
        .clip_mut(blocker)
        .unwrap()
        .timeline
        .duration = rt(30);
    assert_rejected_unchanged(
        fixture.project,
        freeze_command(fixture.source, rt(10), rt(6)),
        "overlaps",
    );

    // This one passes command preflight, clears an orphan link internally,
    // then the hardened split continuity check rejects inside the entrance.
    // The link-clear inverse must run before the error escapes.
    let mut fixture = video_fixture(80);
    let orphan = LinkId::next();
    let source = fixture
        .project
        .timeline_mut()
        .clip_mut(fixture.source)
        .unwrap();
    source.link = Some(orphan);
    source.animation_in = Some(AnimationRef::new("fade_in"));
    assert_rejected_unchanged(
        fixture.project,
        freeze_command(fixture.source, rt(15), rt(6)),
        "entrance",
    );
}

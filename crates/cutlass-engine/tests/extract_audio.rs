//! Public command, placement policy, validation, and history coverage for
//! production audio extraction.

use cutlass_commands::{Command, EditCommand, EditOutcome};
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig};
use cutlass_models::{
    AudioRole, Clip, ClipId, Easing, Generator, Keyframe, LinkId, MediaId, MediaSource, Param,
    Project, Rational, RationalTime, TimeRange, TrackId, TrackKind,
};

const FPS: Rational = Rational::FPS_24;

fn tr(start: i64, duration: i64) -> TimeRange {
    TimeRange::at_rate(start, duration, FPS)
}

fn rt(value: i64) -> RationalTime {
    RationalTime::new(value, FPS)
}

fn engine(project: Project) -> Engine {
    Engine::with_project(EngineConfig { undo_limit: 32 }, project).expect("engine")
}

fn video_fixture(has_audio: bool) -> (Project, ClipId, MediaId, TrackId) {
    let mut project = Project::new("extract", FPS);
    let media = project.add_media(MediaSource::new(
        "/tmp/extract-engine.mp4",
        1920,
        1080,
        FPS,
        1_000,
        has_audio,
    ));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project
        .add_clip(track, media, tr(40, 120), rt(24))
        .expect("video clip");
    (project, clip, media, track)
}

fn extract(
    engine: &mut Engine,
    clip: ClipId,
    to_track: Option<TrackId>,
) -> Result<ClipId, cutlass_engine::EngineError> {
    match engine.apply(Command::Edit(EditCommand::ExtractAudio { clip, to_track }))? {
        ApplyOutcome::Edited(EditOutcome::Created(id)) => Ok(id),
        other => panic!("expected Created, got {other:?}"),
    }
}

fn audio_tracks(engine: &Engine) -> Vec<TrackId> {
    engine
        .project()
        .timeline()
        .tracks_ordered()
        .filter(|track| track.kind == TrackKind::Audio)
        .map(|track| track.id)
        .collect()
}

fn project_snapshot(project: &Project) -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("snapshot.cutlass");
    project.save_to_file(&path).unwrap();
    std::fs::read(path).unwrap()
}

fn expected_companion(source: &Clip, id: ClipId) -> Clip {
    let mut expected = Clip::from_media(
        source.media().expect("media"),
        source.source_range().expect("source range"),
        source.timeline,
    );
    expected.id = id;
    expected.speed = source.speed;
    expected.reversed = source.reversed;
    expected.speed_curve = source.speed_curve.clone();
    expected.preserve_pitch = source.preserve_pitch;
    expected.volume = source.volume.clone();
    expected.pan = source.pan.clone();
    expected.fade_in = source.fade_in;
    expected.fade_out = source.fade_out;
    expected.denoise = source.denoise;
    expected.beats = source.beats.clone();
    expected.audio_role = Some(AudioRole::Extracted);
    expected
}

#[test]
fn auto_extract_is_exact_and_undo_redo_restores_identical_entities() {
    let (mut project, video, _media, _video_track) = video_fixture(true);
    let orphan = LinkId::next();
    {
        let source = project.timeline_mut().clip_mut(video).unwrap();
        source.link = Some(orphan);
        source.timeline.duration = rt(73);
        source.speed = Rational::new(3, 2);
        source.reversed = true;
        source.speed_curve = Param::Keyframed {
            keyframes: vec![
                Keyframe {
                    tick: 0,
                    value: 0.5,
                    easing: Easing::EaseIn,
                    tangents: None,
                },
                Keyframe {
                    tick: 1_000,
                    value: 1.75,
                    easing: Easing::EaseOut,
                    tangents: None,
                },
            ],
        };
        source.preserve_pitch = false;
        source.volume = Param::Keyframed {
            keyframes: vec![
                Keyframe {
                    tick: 0,
                    value: 0.2,
                    easing: Easing::Linear,
                    tangents: None,
                },
                Keyframe {
                    tick: 72,
                    value: 0.9,
                    easing: Easing::EaseInOut,
                    tangents: None,
                },
            ],
        };
        source.fade_in = 5;
        source.fade_out = 9;
        source.denoise = true;
        source.beats = vec![48, 72, 100];
        source.flip_h = true;
        source.animation_combo = Some(cutlass_models::AnimationRef::new("pulse"));
        source.replaceable = Some(cutlass_models::Replaceable::new(2));
    }
    let source_before = project.clip(video).unwrap().clone();
    let media_count = project.media_count();
    let mut engine = engine(project);

    let audio = extract(&mut engine, video, None).expect("extract");
    let track = engine.project().timeline().track_of(audio).unwrap();
    let track_model = engine.project().timeline().track(track).unwrap();
    assert_eq!(track_model.kind, TrackKind::Audio);
    assert_eq!(track_model.name, "A1");
    assert!(!track_model.pinned);

    let source_after = engine.project().clip(video).unwrap().clone();
    let audio_after = engine.project().clip(audio).unwrap().clone();
    let mut expected = expected_companion(&source_before, audio);
    expected.link = audio_after.link;
    assert_eq!(audio_after, expected);
    assert_eq!(audio_after.timeline, source_before.timeline);
    assert_eq!(source_after.link, audio_after.link);
    assert!(source_after.link.is_some());
    assert_ne!(source_after.link, Some(orphan));
    assert!(!engine.project().timeline().carries_own_audio(video));
    assert!(engine.project().timeline().carries_own_audio(audio));
    assert_eq!(engine.project().media_count(), media_count);
    assert_eq!(audio_tracks(&engine), vec![track]);

    assert!(engine.undo(), "one command is one undo entry");
    assert!(!engine.undo(), "the fixture itself added no history");
    assert!(engine.project().clip(audio).is_none());
    assert!(engine.project().timeline().track(track).is_none());
    assert_eq!(engine.project().clip(video).unwrap(), &source_before);
    assert_eq!(engine.project().clip(video).unwrap().link, Some(orphan));
    assert!(engine.project().timeline().carries_own_audio(video));

    assert!(engine.redo());
    assert_eq!(engine.project().timeline().track_of(audio), Some(track));
    assert_eq!(engine.project().clip(audio).unwrap(), &audio_after);
    assert_eq!(engine.project().clip(video).unwrap(), &source_after);
}

#[test]
fn automatic_destination_uses_first_unlocked_exact_fit_lane() {
    let (mut project, video, media, _video_track) = video_fixture(true);
    // The final retimed span is [24, 84). Its unretimed source reserve would
    // extend to 144, so an occupant beginning at 84 catches regressions that
    // search with the oversized prototype rather than the final range.
    {
        let source = project.timeline_mut().clip_mut(video).unwrap();
        source.timeline.duration = rt(60);
        source.speed = Rational::new(2, 1);
    }
    let locked = project.add_track(TrackKind::Audio, "Locked");
    project.timeline_mut().track_mut(locked).unwrap().locked = true;
    let exact_fit = project.add_track(TrackKind::Audio, "Exact fit");
    project
        .add_clip(exact_fit, media, tr(0, 48), rt(84))
        .unwrap();
    let later_free = project.add_track(TrackKind::Audio, "Later");
    let mut engine = engine(project);

    let audio = extract(&mut engine, video, None).expect("extract");
    assert_eq!(
        engine.project().timeline().track_of(audio),
        Some(exact_fit),
        "the first unlocked lane fits the exact final [24, 84) range"
    );
    assert_eq!(audio_tracks(&engine), vec![locked, exact_fit, later_free]);
}

#[test]
fn explicit_destination_survives_undo_and_redo() {
    let (mut project, video, _media, _video_track) = video_fixture(true);
    let destination = project.add_track(TrackKind::Audio, "Dialogue");
    let mut engine = engine(project);

    let audio = extract(&mut engine, video, Some(destination)).expect("extract");
    let audio_snapshot = engine.project().clip(audio).unwrap().clone();
    assert_eq!(
        engine.project().timeline().track_of(audio),
        Some(destination)
    );

    assert!(engine.undo());
    let track = engine
        .project()
        .timeline()
        .track(destination)
        .expect("preexisting lane remains");
    assert!(track.is_empty());
    assert!(engine.project().clip(audio).is_none());

    assert!(engine.redo());
    assert_eq!(
        engine.project().timeline().track_of(audio),
        Some(destination)
    );
    assert_eq!(engine.project().clip(audio).unwrap(), &audio_snapshot);
}

fn assert_rejected_unchanged(project: Project, command: EditCommand, message_fragment: &str) {
    let mut engine = engine(project);
    let before = project_snapshot(engine.project());
    let error = engine
        .apply(Command::Edit(command))
        .expect_err("command must reject");
    assert!(
        error.to_string().contains(message_fragment),
        "expected '{message_fragment}' in '{error}'"
    );
    assert_eq!(
        project_snapshot(engine.project()),
        before,
        "rejection changed project state"
    );
    assert!(!engine.can_undo(), "rejection must not add history");
}

#[test]
fn source_and_explicit_destination_rejections_are_atomic() {
    let (project, video, _media, _) = video_fixture(false);
    assert_rejected_unchanged(
        project,
        EditCommand::ExtractAudio {
            clip: video,
            to_track: None,
        },
        "no audio stream",
    );

    let (mut project, video, _media, video_track) = video_fixture(true);
    project
        .timeline_mut()
        .track_mut(video_track)
        .unwrap()
        .locked = true;
    assert_rejected_unchanged(
        project,
        EditCommand::ExtractAudio {
            clip: video,
            to_track: None,
        },
        "source video track",
    );

    let mut project = Project::new("image", FPS);
    let image = project.add_media(MediaSource::image("/tmp/still.png", 1920, 1080));
    project.media_mut(image).unwrap().has_audio = true;
    let video_track = project.add_track(TrackKind::Video, "V1");
    let image_source = project.media(image).unwrap().full_range();
    let image_clip = project
        .add_clip(video_track, image, image_source, rt(0))
        .unwrap();
    assert_rejected_unchanged(
        project,
        EditCommand::ExtractAudio {
            clip: image_clip,
            to_track: None,
        },
        "does not reference video media",
    );

    let mut project = Project::new("generated", FPS);
    let text = project.add_track(TrackKind::Text, "Text");
    let generated = project
        .add_generated(text, Generator::text("title"), tr(0, 24))
        .unwrap();
    assert_rejected_unchanged(
        project,
        EditCommand::ExtractAudio {
            clip: generated,
            to_track: None,
        },
        "media-backed",
    );

    let mut project = Project::new("wrong lane", FPS);
    let media = project.add_media(MediaSource::new("/tmp/audio.wav", 0, 0, FPS, 120, true));
    let audio_track = project.add_track(TrackKind::Audio, "A1");
    let audio_clip = project
        .add_clip(audio_track, media, tr(0, 48), rt(0))
        .unwrap();
    assert_rejected_unchanged(
        project,
        EditCommand::ExtractAudio {
            clip: audio_clip,
            to_track: None,
        },
        "video track",
    );

    let (project, _video, _media, _) = video_fixture(true);
    assert_rejected_unchanged(
        project,
        EditCommand::ExtractAudio {
            clip: ClipId::from_raw(u64::MAX),
            to_track: None,
        },
        "unknown clip",
    );

    let (project, video, _media, video_track) = video_fixture(true);
    assert_rejected_unchanged(
        project,
        EditCommand::ExtractAudio {
            clip: video,
            to_track: Some(video_track),
        },
        "cannot hold this clip",
    );

    let (project, video, _media, _) = video_fixture(true);
    assert_rejected_unchanged(
        project,
        EditCommand::ExtractAudio {
            clip: video,
            to_track: Some(TrackId::from_raw(u64::MAX)),
        },
        "unknown track",
    );

    let (mut project, video, _media, _) = video_fixture(true);
    let locked = project.add_track(TrackKind::Audio, "Locked");
    project.timeline_mut().track_mut(locked).unwrap().locked = true;
    assert_rejected_unchanged(
        project,
        EditCommand::ExtractAudio {
            clip: video,
            to_track: Some(locked),
        },
        "locked",
    );

    let (mut project, video, media, _) = video_fixture(true);
    let occupied = project.add_track(TrackKind::Audio, "Occupied");
    project
        .add_clip(occupied, media, tr(0, 48), rt(48))
        .unwrap();
    assert_rejected_unchanged(
        project,
        EditCommand::ExtractAudio {
            clip: video,
            to_track: Some(occupied),
        },
        "overlaps",
    );
}

#[test]
fn linked_and_already_extracted_sources_reject_without_mutation() {
    let (mut project, video, media, video_track) = video_fixture(true);
    let other = project
        .add_clip(video_track, media, tr(0, 24), rt(200))
        .unwrap();
    let group = LinkId::next();
    project.timeline_mut().clip_mut(video).unwrap().link = Some(group);
    project.timeline_mut().clip_mut(other).unwrap().link = Some(group);
    assert_rejected_unchanged(
        project,
        EditCommand::ExtractAudio {
            clip: video,
            to_track: None,
        },
        "unlink the group",
    );

    let (project, video, _media, _) = video_fixture(true);
    let mut engine = engine(project);
    extract(&mut engine, video, None).unwrap();
    let before = project_snapshot(engine.project());
    let error = extract(&mut engine, video, None).expect_err("second extraction");
    assert!(error.to_string().contains("already has extracted audio"));
    assert_eq!(project_snapshot(engine.project()), before);
    assert!(engine.can_undo(), "the first extraction remains undoable");
    assert!(!engine.can_redo());
}

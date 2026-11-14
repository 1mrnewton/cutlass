use cutlass_eval::{copy_ask, enhance_ask};
use cutlass_model::{
    Draft, MaterialAudio, MaterialVideo, Micros, MotionBlur, Segment, TimeRange, Track,
    TrackAttribute,
};
use cutlass_settings::Settings;
use cutlass_store::enhanced_file;

use super::*;

/// A project with a movie clip `v`, a photo clip `p` and a sound `a`.
fn open(name: &str, movie: Option<&[u8]>) -> (Shell, PathBuf) {
    open_with(name, movie, |_| {})
}

/// [`open`], with `edit` making changes to the draft before it is saved.
fn open_with(name: &str, movie: Option<&[u8]>, edit: impl FnOnce(&mut Draft)) -> (Shell, PathBuf) {
    let temp = std::env::temp_dir().join(format!("cutlass-enhance-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    let mut draft = Draft::empty("draft");
    draft.fps = 10.0;
    draft.duration = Micros(2_000_000);
    draft.materials.videos.push(MaterialVideo {
        id: "m".into(),
        kind: "video".into(),
        width: 320,
        height: 180,
        duration: Micros(2_000_000),
        path: "walk.mp4".into(),
        ..MaterialVideo::default()
    });
    draft.materials.videos.push(MaterialVideo {
        id: "still".into(),
        kind: "photo".into(),
        path: "walk.mp4".into(),
        ..MaterialVideo::default()
    });
    draft.materials.audios.push(MaterialAudio {
        id: "tone".into(),
        ..MaterialAudio::default()
    });
    let mut clip = Segment::new("v", "m", TimeRange::new(0, 2_000_000));
    clip.source_range = Some(TimeRange::new(0, 2_000_000));
    draft.tracks[0].segments.push(clip);
    let mut overlay = Track::main("overlay");
    overlay.kind = cutlass_model::TrackType::Video;
    overlay
        .segments
        .push(Segment::new("p", "still", TimeRange::new(0, 1_000_000)));
    draft.tracks.push(overlay);
    let mut sound = Track::main("sound");
    sound.kind = cutlass_model::TrackType::Audio;
    sound
        .segments
        .push(Segment::new("a", "tone", TimeRange::new(0, 1_000_000)));
    draft.tracks.push(sound);
    edit(&mut draft);
    cutlass_store::Folder::new(draft).save(&temp).unwrap();
    if let Some(bytes) = movie {
        std::fs::write(temp.join("walk.mp4"), bytes).unwrap();
    }
    let (decoders, encoders) = crate::register_codecs();
    let mut shell = Shell::new(decoders, encoders, Settings::default());
    shell.open_folder(&temp).unwrap();
    (shell, temp)
}

/// Keep a copy of the clip at `level`, as a finished job would.
fn enhanced(shell: &Shell, temp: &std::path::Path, level: i64) {
    copied(shell, temp, level, 0);
}

/// Keep a copy of the clip at these levels, as a finished job would.
fn copied(shell: &Shell, temp: &std::path::Path, level: i64, denoise: i64) {
    flowed(shell, temp, "v", level, denoise, 0);
}

/// Keep a copy of clip `id` at these levels and frame rate.
fn flowed(shell: &Shell, temp: &std::path::Path, id: &str, level: i64, denoise: i64, fps: i64) {
    let ask = enhance_ask(
        shell.draft().unwrap(),
        Some(temp),
        &id.into(),
        level,
        denoise,
        fps,
    )
    .unwrap();
    let file = enhanced_file(temp, &ask.key(), &ask.range).unwrap();
    std::fs::write(file, b"a copied movie").unwrap();
}

#[test]
fn a_clip_with_a_copy_takes_its_level_at_once_and_undo_returns() {
    let (mut shell, temp) = open("copied", Some(b"a movie"));
    enhanced(&shell, &temp, Algorithm::HD);
    enhanced(&shell, &temp, Algorithm::UHD);
    shell.select(Some("v".into()));
    assert_eq!(shell.enhance_level(), 0);
    shell.set_enhance(Algorithm::HD).unwrap();
    assert!(shell.job().is_none(), "nothing to make");
    assert_eq!(shell.enhance_level(), Algorithm::HD);
    shell.set_enhance(Algorithm::UHD).unwrap();
    assert_eq!(shell.enhance_level(), Algorithm::UHD);
    assert!(shell.enhance_progress().is_none());
    shell.set_enhance(0).unwrap();
    assert_eq!(shell.enhance_level(), 0);
    shell.undo().unwrap();
    assert_eq!(shell.enhance_level(), Algorithm::UHD);
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn turning_it_off_needs_no_copy_model_or_gpu() {
    let (mut shell, temp) = open("off", Some(b"a movie"));
    shell.select(Some("v".into()));
    let revision = shell.revision();
    shell.set_enhance(0).unwrap();
    assert!(shell.job().is_none());
    assert_eq!(shell.enhance_level(), 0);
    assert!(shell.revision() >= revision);
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn a_photo_a_sound_nothing_selected_or_a_missing_file_is_refused() {
    let (mut shell, temp) = open("refused", None);
    assert!(shell.set_enhance(1).is_err(), "nothing selected");
    for id in ["p", "a"] {
        shell.select(Some(id.into()));
        assert!(shell.set_enhance(1).is_err(), "{id}");
    }
    shell.select(Some("v".into()));
    let revision = shell.revision();
    assert!(
        shell.set_enhance(1).is_err(),
        "the movie is not in the folder"
    );
    assert_eq!(shell.revision(), revision);
    assert!(shell.job().is_none());
    assert_eq!(shell.enhance_level(), 0);
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn a_denoised_clip_with_a_copy_takes_its_level_at_once_beside_enhance() {
    let (mut shell, temp) = open("denoised", Some(b"a movie"));
    copied(&shell, &temp, 0, Algorithm::WEAK);
    copied(&shell, &temp, 0, Algorithm::STRONG);
    copied(&shell, &temp, Algorithm::HD, Algorithm::STRONG);
    shell.select(Some("v".into()));
    assert_eq!(shell.denoise_level(), 0);
    shell.set_denoise(Algorithm::WEAK).unwrap();
    assert!(shell.job().is_none(), "nothing to make");
    shell.set_denoise(Algorithm::STRONG).unwrap();
    assert_eq!(shell.denoise_level(), Algorithm::STRONG);
    // Enhance on as well: the copy of both is there.
    shell.set_enhance(Algorithm::HD).unwrap();
    assert!(shell.job().is_none());
    assert_eq!(
        (shell.enhance_level(), shell.denoise_level()),
        (Algorithm::HD, Algorithm::STRONG)
    );
    assert!(shell.denoise_progress().is_none());
    // Enhance off again: the denoised copy draws the clip.
    shell.set_enhance(0).unwrap();
    assert!(shell.job().is_none());
    assert_eq!(shell.denoise_level(), Algorithm::STRONG);
    shell.undo().unwrap();
    assert_eq!(shell.enhance_level(), Algorithm::HD);
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn turning_one_off_never_fails_even_when_the_others_copy_cannot_start() {
    let (mut shell, temp) = open("one-off", Some(b"a movie"));
    copied(&shell, &temp, Algorithm::HD, 0);
    copied(&shell, &temp, Algorithm::HD, Algorithm::WEAK);
    shell.select(Some("v".into()));
    shell.set_enhance(Algorithm::HD).unwrap();
    shell.set_denoise(Algorithm::WEAK).unwrap();
    assert!(shell.job().is_none());
    // The movie goes, so a copy only denoised cannot be asked for.
    std::fs::remove_file(temp.join("walk.mp4")).unwrap();
    shell.set_enhance(0).unwrap();
    assert!(shell.job().is_none());
    assert_eq!(
        (shell.enhance_level(), shell.denoise_level()),
        (0, Algorithm::WEAK)
    );
    // Turning it on is refused, and the level stays.
    assert!(shell.set_denoise(Algorithm::STRONG).is_err());
    assert_eq!(shell.denoise_level(), Algorithm::WEAK);
    shell.set_denoise(0).unwrap();
    assert_eq!(shell.denoise_level(), 0);
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn a_photo_or_a_sound_is_not_denoised() {
    let (mut shell, temp) = open("denoise-refused", None);
    assert!(shell.set_denoise(1).is_err(), "nothing selected");
    for id in ["p", "a"] {
        shell.select(Some(id.into()));
        assert!(shell.set_denoise(1).is_err(), "{id}");
        assert_eq!(shell.denoise_level(), 0);
    }
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn an_interpolated_clip_with_a_copy_takes_its_rate_at_once_beside_the_others() {
    let (mut shell, temp) = open("flowed", Some(b"a movie"));
    for fps in [30, 60] {
        flowed(&shell, &temp, "v", 0, 0, fps);
    }
    flowed(&shell, &temp, "v", Algorithm::HD, 0, 60);
    shell.select(Some("v".into()));
    assert_eq!(shell.interpolate_level(), 0);
    shell.set_interpolate(30).unwrap();
    assert!(shell.job().is_none(), "nothing to make");
    assert_eq!(shell.interpolate_level(), 30);
    shell.set_interpolate(60).unwrap();
    shell.set_enhance(Algorithm::HD).unwrap();
    assert!(shell.job().is_none(), "the copy of both is there");
    assert_eq!(
        (shell.enhance_level(), shell.interpolate_level()),
        (Algorithm::HD, 60)
    );
    assert!(shell.interpolate_progress().is_none());
    let revision = shell.revision();
    assert!(shell.set_interpolate(24).is_err(), "not an offered rate");
    assert_eq!(shell.revision(), revision);
    shell.set_interpolate(0).unwrap();
    assert_eq!(shell.interpolate_level(), 0);
    assert_eq!(shell.enhance_level(), Algorithm::HD);
    shell.undo().unwrap();
    assert_eq!(shell.interpolate_level(), 60);
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn apply_to_all_fills_every_movie_clip_in_one_undo_step() {
    // A second movie clip after the first, and a locked lane's clip.
    let (mut shell, temp) = open_with("all", Some(b"a movie"), |draft| {
        let mut second = draft.tracks[0].segments[0].clone();
        second.id = "w".into();
        second.target = TimeRange::new(2_000_000, 1_000_000);
        second.source_range = Some(TimeRange::new(0, 1_000_000));
        draft.tracks[0].segments.push(second);
        let mut locked = Track::main("locked");
        locked.kind = cutlass_model::TrackType::Video;
        locked.attribute = TrackAttribute(TrackAttribute::LOCK);
        let mut held = draft.tracks[0].segments[0].clone();
        held.id = "l".into();
        locked.segments.push(held);
        draft.tracks.push(locked);
    });
    for id in ["v", "w"] {
        flowed(&shell, &temp, id, 0, 0, 50);
    }
    let fps = |shell: &Shell, id: &str| levels_of(shell.draft().unwrap(), &id.into()).interpolate;
    shell.set_interpolate_all(50).unwrap();
    assert!(shell.job().is_none(), "both copies are there");
    assert_eq!((fps(&shell, "v"), fps(&shell, "w")), (50, 50));
    assert_eq!(fps(&shell, "l"), 0, "a locked lane stays");
    assert_eq!(fps(&shell, "p"), 0, "a photo stays");
    assert!(shell.set_interpolate_all(24).is_err());
    shell.set_interpolate_all(0).unwrap();
    assert_eq!((fps(&shell, "v"), fps(&shell, "w")), (0, 0));
    shell.undo().unwrap();
    assert_eq!((fps(&shell, "v"), fps(&shell, "w")), (50, 50));
    shell.undo().unwrap();
    assert_eq!((fps(&shell, "v"), fps(&shell, "w")), (0, 0));
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn apply_to_all_with_no_file_to_fill_is_refused_and_the_clips_stay() {
    let (mut shell, temp) = open("all-refused", None);
    let revision = shell.revision();
    assert!(shell.set_interpolate_all(30).is_err());
    assert_eq!(shell.revision(), revision);
    assert!(shell.job().is_none());
    // Off still goes through, with nothing to make.
    shell.set_interpolate_all(0).unwrap();
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn a_new_speed_waits_for_the_slider_to_be_let_go_before_its_copy_is_made() {
    let (mut shell, temp) = open("speed", Some(b"a movie"));
    flowed(&shell, &temp, "v", 0, 0, 30);
    shell.select(Some("v".into()));
    shell.set_interpolate(30).unwrap();
    // Dragged: each value is stored, and the copy waits.
    shell.hold_edits(true);
    shell.set_speed(0.75).unwrap();
    shell.set_speed(0.5).unwrap();
    assert_eq!(shell.refill, Some("v".into()));
    assert!(shell.job().is_none());
    // The copy for half speed is there, so letting go has nothing to make.
    flowed(&shell, &temp, "v", 0, 0, 30);
    shell.hold_edits(false);
    assert_eq!(shell.refill, None);
    assert!(shell.job().is_none());
    assert_eq!(shell.interpolate_level(), 30);
    // A clip without Optical flow waits for nothing.
    shell.set_interpolate(0).unwrap();
    shell.hold_edits(true);
    shell.set_speed(2.0).unwrap();
    shell.hold_edits(false);
    assert!(shell.job().is_none());
    let _ = std::fs::remove_dir_all(&temp);
}

fn blur(times: i64) -> MotionBlur {
    MotionBlur {
        blur: 0.8,
        blend: 0.1,
        direction: MotionBlur::FORWARD,
        times,
    }
}

/// Keep a copy of clip `v` blurred as `blur` says, as a finished job would.
fn blurred(shell: &Shell, temp: &std::path::Path, blur: &MotionBlur) {
    let ask = copy_ask(
        shell.draft().unwrap(),
        Some(temp),
        &"v".into(),
        0,
        0,
        0,
        Some(blur),
    )
    .unwrap();
    let file = enhanced_file(temp, &ask.key(), &ask.range).unwrap();
    std::fs::write(file, b"a blurred movie").unwrap();
}

#[test]
fn a_clip_with_a_blurred_copy_takes_its_blur_at_once_and_undo_returns() {
    let (mut shell, temp) = open("blurred", Some(b"a movie"));
    blurred(&shell, &temp, &blur(4));
    shell.select(Some("v".into()));
    assert_eq!(shell.motion_blur(), None);
    shell.set_motion_blur(Some(blur(4))).unwrap();
    assert!(shell.job().is_none(), "nothing to make");
    assert_eq!(shell.motion_blur(), Some(blur(4)));
    assert!(shell.motion_blur_progress().is_none());
    shell.set_motion_blur(None).unwrap();
    assert_eq!(shell.motion_blur(), None);
    shell.undo().unwrap();
    assert_eq!(shell.motion_blur(), Some(blur(4)));
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn a_photo_or_a_wrong_blur_is_not_blurred_and_off_needs_no_copy() {
    let (mut shell, temp) = open("blur-refused", Some(b"a movie"));
    shell.select(Some("p".into()));
    assert!(shell.set_motion_blur(Some(blur(1))).is_err());
    shell.select(Some("v".into()));
    shell.set_motion_blur(None).unwrap();
    assert!(shell.job().is_none());
    blurred(&shell, &temp, &blur(3));
    assert!(
        shell.set_motion_blur(Some(blur(3))).is_err(),
        "3 times is not offered"
    );
    assert_eq!(shell.motion_blur(), None);
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn a_keyed_blend_keys_without_the_blur_and_shows_at_the_playhead() {
    let (mut shell, temp) = open("blur-keyed", Some(b"a movie"));
    blurred(&shell, &temp, &blur(4));
    shell.select(Some("v".into()));
    shell.set_motion_blur(Some(blur(4))).unwrap();
    shell
        .toggle_setting_key(KeyedSetting::MotionBlurBlend)
        .unwrap();
    shell.seek(Micros(1_000_000));
    let mut more = blur(4);
    more.blend = 0.5;
    shell.set_motion_blur(Some(more)).unwrap();
    let times = |shell: &Shell, setting| shell.setting_keys(setting).unwrap().times;
    assert_eq!(times(&shell, KeyedSetting::MotionBlurBlend).len(), 2);
    assert!(times(&shell, KeyedSetting::MotionBlurLength).is_empty());
    // The stored Blend stays; the playhead shows the keyed one.
    shell.seek(Micros(500_000));
    let shown = shell.motion_blur().unwrap();
    assert!((shown.blend - 0.3).abs() < 1e-6);
    assert_eq!(shown.blur, 0.8);
    assert_eq!(shell.stored_motion_blur().unwrap().blend, 0.1);
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn remove_flicker_takes_its_copy_at_once_and_undo_returns() {
    let (mut shell, temp) = open("deflicker-copied", Some(b"a movie"));
    let ask = clip_copy_ask(
        shell.draft().unwrap(),
        Some(&temp),
        &"v".into(),
        &CopyLevels {
            deflicker: Algorithm::DEFLICKER_ON,
            ..CopyLevels::default()
        },
    )
    .unwrap();
    let file = enhanced_file(&temp, &ask.key(), &ask.range).unwrap();
    assert!(
        file.parent()
            .unwrap()
            .ends_with("Resources/video_deflicker")
    );
    std::fs::write(file, b"an evened movie").unwrap();
    shell.select(Some("v".into()));
    assert_eq!(shell.deflicker_level(), 0);
    shell.set_deflicker(Algorithm::DEFLICKER_ON).unwrap();
    assert!(shell.job().is_none(), "nothing to make");
    assert_eq!(shell.deflicker_level(), Algorithm::DEFLICKER_ON);
    assert!(shell.deflicker_progress().is_none());
    shell.set_deflicker(0).unwrap();
    assert_eq!(shell.deflicker_level(), 0);
    shell.undo().unwrap();
    assert_eq!(shell.deflicker_level(), Algorithm::DEFLICKER_ON);
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn remove_flicker_with_no_copy_starts_its_own_job_without_a_gpu_check() {
    let (mut shell, temp) = open("deflicker-job", Some(b"a movie"));
    shell.select(Some("v".into()));
    shell.set_deflicker(Algorithm::DEFLICKER_ON).unwrap();
    assert_eq!(shell.deflicker_level(), Algorithm::DEFLICKER_ON);
    assert_eq!(
        shell.job().map(|job| job.label),
        Some(DEFLICKER_JOB.to_owned())
    );
    // The movie is not one, so the job fails, and the level stays.
    let until = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while shell.poll_job() && std::time::Instant::now() < until {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(shell.job().is_none());
    assert_eq!(shell.deflicker_level(), Algorithm::DEFLICKER_ON);
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn remove_flicker_refuses_a_photo_a_sound_and_nothing_selected() {
    let (mut shell, temp) = open("deflicker-refused", Some(b"a movie"));
    assert!(shell.set_deflicker(1).is_err(), "nothing selected");
    for id in ["p", "a"] {
        shell.select(Some(id.into()));
        assert!(shell.set_deflicker(1).is_err(), "{id}");
    }
    let _ = std::fs::remove_dir_all(&temp);
}

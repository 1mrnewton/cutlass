use cutlass_model::{Draft, MaterialAudio, MaterialVideo, Micros, Segment, TimeRange, Track};
use cutlass_motion::{Move, Path as CameraPath};
use cutlass_settings::Settings;

use super::*;

/// A project with a movie clip `v`, a photo clip `p` and a sound `a`.
fn open(name: &str, movie: Option<&[u8]>) -> (Shell, std::path::PathBuf) {
    let temp = std::env::temp_dir().join(format!("cutlass-stable-{name}-{}", std::process::id()));
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
    draft.tracks[0]
        .segments
        .push(Segment::new("v", "m", TimeRange::new(0, 2_000_000)));
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
    cutlass_store::Folder::new(draft).save(&temp).unwrap();
    if let Some(bytes) = movie {
        std::fs::write(temp.join("walk.mp4"), bytes).unwrap();
    }
    let (decoders, encoders) = crate::register_codecs();
    let mut shell = Shell::new(decoders, encoders, Settings::default());
    shell.open_folder(&temp).unwrap();
    (shell, temp)
}

fn land(shell: &mut Shell) {
    let until = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while shell.poll_job() && std::time::Instant::now() < until {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

/// Keep a measured path for the movie, as a finished job would.
fn measured(shell: &Shell, temp: &std::path::Path) {
    let ask = stable_ask(shell.draft().unwrap(), Some(temp), &"v".into()).unwrap();
    let path = CameraPath {
        width: 320,
        height: 180,
        rate: ask.rate,
        moves: vec![
            Move {
                dx: 2.0,
                confidence: 1.0,
                ..Move::STILL
            };
            19
        ],
    };
    save_camera_path(temp, &key(&ask), &path).unwrap();
}

#[test]
fn a_measured_clip_takes_its_level_at_once_and_undo_returns() {
    let (mut shell, temp) = open("measured", Some(b"a movie"));
    measured(&shell, &temp);
    shell.select(Some("v".into()));
    assert_eq!(shell.stable_level(), 0);
    shell.set_stable(VideoStable::RECOMMENDED).unwrap();
    assert!(shell.job().is_none(), "nothing to measure");
    assert_eq!(shell.stable_level(), 2);
    shell.set_stable(VideoStable::MOST_STABLE).unwrap();
    assert_eq!(shell.stable_level(), 3);
    shell.set_stable(0).unwrap();
    assert_eq!(shell.stable_level(), 0);
    shell.undo().unwrap();
    assert_eq!(shell.stable_level(), 3);
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn an_unmeasured_clip_is_measured_as_a_job_and_a_failure_is_told() {
    let (mut shell, temp) = open("unmeasured", Some(b"not a movie at all"));
    shell.select(Some("v".into()));
    shell.set_stable(VideoStable::RECOMMENDED).unwrap();
    assert_eq!(shell.stable_level(), 2, "the level is stored straight away");
    let job = shell.job().unwrap();
    assert_eq!(job.label, STABILIZE_JOB);
    assert!(shell.stable_progress().is_some());
    // A second clip waits for the first job.
    assert!(shell.set_stable(VideoStable::MINIMUM_CUT).is_err());
    land(&mut shell);
    assert!(shell.job().is_none());
    assert!(shell.stable_progress().is_none());
    assert!(
        shell.notice().contains("could not be steadied"),
        "{}",
        shell.notice()
    );
    assert_eq!(shell.stable_level(), 2);
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn a_photo_a_sound_nothing_selected_or_a_missing_file_is_refused() {
    let (mut shell, temp) = open("refused", None);
    assert!(shell.set_stable(2).is_err(), "nothing selected");
    for id in ["p", "a"] {
        shell.select(Some(id.into()));
        assert!(shell.set_stable(2).is_err(), "{id}");
    }
    shell.select(Some("v".into()));
    let revision = shell.revision();
    assert!(
        shell.set_stable(2).is_err(),
        "the movie is not in the folder"
    );
    assert_eq!(shell.revision(), revision);
    assert!(shell.job().is_none());
    let _ = std::fs::remove_dir_all(&temp);
}

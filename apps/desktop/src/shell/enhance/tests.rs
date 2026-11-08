use cutlass_model::{Draft, MaterialAudio, MaterialVideo, Micros, Segment, TimeRange, Track};
use cutlass_settings::Settings;
use cutlass_store::enhanced_file;

use super::*;

/// A project with a movie clip `v`, a photo clip `p` and a sound `a`.
fn open(name: &str, movie: Option<&[u8]>) -> (Shell, PathBuf) {
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
    let ask = enhance_ask(shell.draft().unwrap(), Some(temp), &"v".into(), level, 0).unwrap();
    let file = enhanced_file(temp, &ask.key(), &ask.range).unwrap();
    std::fs::write(file, b"an enhanced movie").unwrap();
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

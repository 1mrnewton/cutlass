use std::path::PathBuf;

use cutlass_model::{Draft, MaterialVideo, Micros, Segment, TimeRange, Track};
use cutlass_motion::Subject;
use cutlass_settings::Settings;
use cutlass_store::SubjectPath;

use super::*;

/// A project with a movie clip `v` and a photo clip `p`, the movie's file
/// holding `movie` when there is one.
fn open(name: &str, movie: Option<&[u8]>) -> (Shell, PathBuf) {
    let temp = std::env::temp_dir().join(format!("cutlass-reframe-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    let mut draft = Draft::empty("draft");
    draft.fps = 10.0;
    draft.duration = Micros(2_000_000);
    for (id, kind) in [("m", "video"), ("still", "photo")] {
        draft.materials.videos.push(MaterialVideo {
            id: id.into(),
            kind: kind.into(),
            width: 320,
            height: 180,
            duration: Micros(2_000_000),
            path: "walk.mp4".into(),
            ..MaterialVideo::default()
        });
    }
    let mut clip = Segment::new("v", "m", TimeRange::new(0, 2_000_000));
    clip.source_range = Some(TimeRange::new(0, 2_000_000));
    draft.tracks[0].segments.push(clip);
    let mut overlay = Track::main("overlay");
    overlay.kind = cutlass_model::TrackType::Video;
    overlay
        .segments
        .push(Segment::new("p", "still", TimeRange::new(0, 1_000_000)));
    draft.tracks.push(overlay);
    cutlass_store::Folder::new(draft).save(&temp).unwrap();
    if let Some(bytes) = movie {
        std::fs::write(temp.join("walk.mp4"), bytes).unwrap();
    }
    let (decoders, encoders) = crate::register_codecs();
    let mut shell = Shell::new(decoders, encoders, Settings::default());
    shell.open_folder(&temp).unwrap();
    (shell, temp)
}

/// Keep clip `v`'s subject track, as a finished job would.
fn keep_track(shell: &Shell, temp: &std::path::Path) {
    let ask = reframe_ask(shell.draft().unwrap(), Some(temp), &"v".into()).unwrap();
    let path = SubjectPath {
        width: 320,
        height: 180,
        subjects: vec![Subject::NONE; 10],
    };
    save_subject_path(temp, &ask.key(), &path).unwrap();
}

#[test]
fn a_clip_with_its_track_takes_each_shape_at_once_and_undo_returns() {
    let (mut shell, temp) = open("cut", Some(b"a movie"));
    keep_track(&shell, &temp);
    shell.select(Some("v".into()));
    assert_eq!(shell.reframe(), None);
    shell
        .set_reframe("9:16", Reframe::DEFAULT, Reframe::FAST)
        .unwrap();
    assert!(shell.job().is_none(), "nothing to find");
    let cut = shell.reframe().unwrap();
    assert_eq!((cut.ratio.as_str(), cut.speed), ("9:16", Reframe::FAST));
    shell
        .set_reframe("1:1", Reframe::STABLE, Reframe::SLOW)
        .unwrap();
    assert_eq!(shell.reframe().unwrap().stability, Reframe::STABLE);
    assert!(shell.reframe_progress().is_none());
    shell
        .set_reframe("", Reframe::DEFAULT, Reframe::DEFAULT)
        .unwrap();
    assert_eq!(shell.reframe(), None);
    shell.undo().unwrap();
    assert_eq!(shell.reframe().unwrap().ratio, "1:1");
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn turning_it_off_needs_no_track_or_model() {
    let (mut shell, temp) = open("off", Some(b"a movie"));
    shell.select(Some("v".into()));
    shell.set_reframe("", 2, 2).unwrap();
    assert!(shell.job().is_none());
    assert_eq!(shell.reframe(), None);
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn a_photo_nothing_selected_a_missing_file_or_a_wrong_setting_is_refused() {
    let (mut shell, temp) = open("refused", None);
    assert!(shell.set_reframe("9:16", 2, 2).is_err());
    shell.select(Some("p".into()));
    assert!(matches!(
        shell.set_reframe("9:16", 2, 2),
        Err(ShellError::Unavailable(_))
    ));
    shell.select(Some("v".into()));
    assert!(shell.set_reframe("9:16", 2, 2).is_err(), "no file to read");
    std::fs::write(temp.join("walk.mp4"), b"a movie").unwrap();
    keep_track(&shell, &temp);
    for (ratio, stability, speed) in [("7:5", 2, 2), ("9:16", 0, 2), ("9:16", 2, 5)] {
        assert!(
            shell.set_reframe(ratio, stability, speed).is_err(),
            "{ratio}"
        );
    }
    assert_eq!(shell.reframe(), None);
    let _ = std::fs::remove_dir_all(&temp);
}

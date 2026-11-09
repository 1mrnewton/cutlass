use std::path::PathBuf;

use cutlass_model::{Draft, MaterialVideo, Micros, Segment, TimeRange, Track};
use cutlass_settings::Settings;
use cutlass_store::relight_file;

use super::*;

/// A project with a movie clip `v` and a photo clip `p`, the movie's file
/// holding `movie` when there is one.
fn open(name: &str, movie: Option<&[u8]>) -> (Shell, PathBuf) {
    let temp = std::env::temp_dir().join(format!("cutlass-relight-{name}-{}", std::process::id()));
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

/// Keep clip `v`'s depth, as a finished job would.
fn keep_depth(shell: &Shell, temp: &std::path::Path) {
    let ask = relight_ask(shell.draft().unwrap(), Some(temp), &"v".into()).unwrap();
    let file = relight_file(temp, &ask.key(), &ask.range).unwrap();
    std::fs::write(file, b"a depth movie").unwrap();
}

#[test]
fn a_clip_with_its_depth_takes_each_light_at_once_and_undo_returns() {
    let (mut shell, temp) = open("lit", Some(b"a movie"));
    keep_depth(&shell, &temp);
    shell.select(Some("v".into()));
    assert_eq!(shell.relight(), None);
    shell.set_relight("warm_key", 0.6, "", 135.0).unwrap();
    assert!(shell.job().is_none(), "nothing to make");
    let lit = shell.relight().unwrap();
    assert_eq!((lit.preset.as_str(), lit.strength), ("warm_key", 0.6));
    shell
        .set_relight(Relight::CUSTOM, 1.0, "#ff8800", 90.0)
        .unwrap();
    let custom = shell.relight().unwrap();
    assert_eq!(
        (custom.preset.as_str(), custom.color.as_str(), custom.angle),
        (Relight::CUSTOM, "#ff8800", 90.0)
    );
    assert!(shell.relight_progress().is_none());
    shell.set_relight("", 1.0, "", 0.0).unwrap();
    assert_eq!(shell.relight(), None);
    shell.undo().unwrap();
    assert_eq!(shell.relight().unwrap().preset, Relight::CUSTOM);
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn turning_it_off_needs_no_depth_or_model() {
    let (mut shell, temp) = open("off", Some(b"a movie"));
    shell.select(Some("v".into()));
    shell.set_relight("", 1.0, "", 0.0).unwrap();
    assert!(shell.job().is_none());
    assert_eq!(shell.relight(), None);
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn a_photo_nothing_selected_a_missing_file_or_a_wrong_light_is_refused() {
    let (mut shell, temp) = open("refused", None);
    assert!(shell.set_relight("warm_key", 1.0, "", 0.0).is_err());
    shell.select(Some("p".into()));
    assert!(matches!(
        shell.set_relight("warm_key", 1.0, "", 0.0),
        Err(ShellError::Unavailable(_))
    ));
    shell.select(Some("v".into()));
    assert!(
        shell.set_relight("warm_key", 1.0, "", 0.0).is_err(),
        "no file to read"
    );
    assert_eq!(shell.relight(), None);
    std::fs::write(temp.join("walk.mp4"), b"a movie").unwrap();
    keep_depth(&shell, &temp);
    for (preset, strength, color) in [
        ("Warm Key", 1.0, ""),
        ("warm_key", 1.5, ""),
        ("warm_key", 1.0, "orange"),
    ] {
        assert!(
            shell.set_relight(preset, strength, color, 0.0).is_err(),
            "{preset} {strength} {color}"
        );
    }
    assert_eq!(shell.relight(), None);
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn a_relight_strength_keys_without_its_color_or_angle() {
    let (mut shell, temp) = open("keyed", Some(b"a movie"));
    keep_depth(&shell, &temp);
    shell.select(Some("v".into()));
    shell
        .set_relight(Relight::CUSTOM, 0.2, "#ff8800", 90.0)
        .unwrap();
    shell.toggle_setting_key(KeyedSetting::RelightStrength).unwrap();
    shell.seek(Micros(1_000_000));
    shell
        .set_relight(Relight::CUSTOM, 0.6, "#ff8800", 90.0)
        .unwrap();
    // Not keyed: stored all along the clip.
    shell
        .set_relight(Relight::CUSTOM, 0.6, "#0088ff", 45.0)
        .unwrap();
    let times = |shell: &Shell, setting| shell.setting_keys(setting).unwrap().times;
    assert_eq!(times(&shell, KeyedSetting::RelightStrength).len(), 2);
    assert!(times(&shell, KeyedSetting::RelightColor).is_empty());
    assert!(times(&shell, KeyedSetting::RelightAngle).is_empty());
    shell.seek(Micros(500_000));
    let lit = shell.relight().unwrap();
    assert!((lit.strength - 0.4).abs() < 1e-6);
    assert_eq!((lit.color.as_str(), lit.angle), ("#0088ff", 45.0));
    // A keyed color plays as keyed.
    shell.toggle_setting_key(KeyedSetting::RelightColor).unwrap();
    assert_eq!(shell.relight().unwrap().color, "#0088ff");
    let _ = std::fs::remove_dir_all(&temp);
}

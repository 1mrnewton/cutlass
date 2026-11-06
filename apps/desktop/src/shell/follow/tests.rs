use cutlass_model::{MaterialVideo, Segment, TimeRange, Track, TrackPath, TrackSample};
use cutlass_settings::Settings;

use super::*;

/// A project with a video clip `v` over two seconds, a sticker-row clip `s`
/// over it, and a clip `x` on a locked row.
fn open(name: &str) -> (Shell, std::path::PathBuf) {
    let temp = std::env::temp_dir().join(format!("cutlass-track-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    let mut draft = Draft::empty("draft");
    draft.canvas_config.width = 1920;
    draft.canvas_config.height = 1080;
    draft.duration = Micros(2_000_000);
    draft.materials.videos.push(MaterialVideo {
        id: "m".into(),
        width: 1920,
        height: 1080,
        path: "missing.mp4".into(),
        ..MaterialVideo::default()
    });
    draft.tracks[0]
        .segments
        .push(Segment::new("v", "m", TimeRange::new(0, 2_000_000)));
    let mut stickers = Track::main("stickers");
    stickers.kind = TrackType::Sticker;
    stickers
        .segments
        .push(Segment::new("s", "m", TimeRange::new(500_000, 1_000_000)));
    draft.tracks.push(stickers);
    let mut locked = Track::main("locked");
    locked.kind = TrackType::Sticker;
    locked.attribute = TrackAttribute(TrackAttribute::LOCK);
    locked
        .segments
        .push(Segment::new("x", "m", TimeRange::new(0, 1_000_000)));
    draft.tracks.push(locked);
    cutlass_store::Folder::new(draft).save(&temp).unwrap();
    let (decoders, encoders) = crate::register_codecs();
    let mut shell = Shell::new(decoders, encoders, Settings::default());
    shell.open_folder(&temp).unwrap();
    (shell, temp)
}

#[test]
fn a_track_is_set_up_on_a_video_clip_with_a_box_and_a_follower() {
    let (mut shell, temp) = open("setup");
    assert!(shell.start_tracking().is_err(), "nothing selected");
    shell.select(Some("s".into()));
    assert!(!shell.track_enabled(), "a sticker is not tracked");
    shell.select(Some("v".into()));
    shell.seek(Micros(700_000));
    assert!(shell.track_enabled());
    shell.start_tracking().unwrap();
    let tracking = shell.tracking().unwrap();
    assert_eq!(tracking.time, Micros(700_000));
    assert_eq!(tracking.mode, TrackMode::Object);
    assert!(!shell.track_ready());

    assert!(
        shell.set_track_box((10.0, 10.0), (12.0, 40.0)).is_err(),
        "too thin"
    );
    shell.set_track_box((400.0, 300.0), (200.0, 100.0)).unwrap();
    assert_eq!(
        shell.tracking().unwrap().canvas_box,
        Some(((200.0, 100.0), (400.0, 300.0)))
    );
    let choices = shell.follower_choices();
    assert_eq!(
        choices
            .iter()
            .map(|c| c.segment_id.as_str())
            .collect::<Vec<_>>(),
        vec!["s"],
        "not the target, not a locked row"
    );
    shell.toggle_follower(&"s".into());
    assert!(shell.track_ready());
    shell.toggle_follower(&"s".into());
    assert!(!shell.track_ready());
    shell.set_track_mode(TrackMode::Plane);
    assert_eq!(shell.tracking().unwrap().mode, TrackMode::Plane);
    shell.stop_tracking();
    assert!(shell.tracking().is_none());
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn a_tracked_path_is_read_back_in_canvas_pixels_for_both_clips() {
    let (mut shell, temp) = open("path");
    let square = |x: f64| {
        [
            (x - 0.1, 0.1),
            (x + 0.1, 0.1),
            (x + 0.1, -0.1),
            (x - 0.1, -0.1),
        ]
    };
    let path = TrackPath {
        mode: TrackMode::Object,
        samples: (0..10)
            .map(|i| TrackSample {
                time: Micros(500_000 + i * 33_333),
                corners: square(0.01 * i as f64),
            })
            .collect(),
    };
    shell
        .apply(Request::Track {
            target: "v".into(),
            followers: vec!["s".into()],
            tracking_id: "t".into(),
            anchor: Micros(500_000),
            path,
        })
        .unwrap();
    shell.select(Some("v".into()));
    let tracked = shell.tracked_path().unwrap();
    assert_eq!(tracked.frames.len(), 10);
    // The first box: 0.1 canvas halves each side of the middle.
    let first = tracked.frames[0].1;
    assert!((first[0].0 - 864.0).abs() < 1e-6 && (first[0].1 - 486.0).abs() < 1e-6);
    assert!(tracked.at(Micros(400_000)).is_none(), "before the path");
    assert_eq!(tracked.at(Micros(510_000)), Some(first));
    shell.select(Some("s".into()));
    assert!(shell.tracked_path().is_some(), "the follower shows it too");
    shell.select(Some("x".into()));
    assert!(shell.tracked_path().is_none());
    let _ = std::fs::remove_dir_all(&temp);
}

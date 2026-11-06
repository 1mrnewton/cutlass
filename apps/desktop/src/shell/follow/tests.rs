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
            masks: Vec::new(),
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

fn sample(time: i64, x: f64) -> TrackSample {
    TrackSample {
        time: Micros(time),
        corners: [(x, 0.0), (x + 0.1, 0.0), (x + 0.1, -0.1), (x, -0.1)],
    }
}

#[test]
fn a_fix_tracks_away_from_the_anchor_and_both_ways_on_it() {
    let anchor = Micros(1_000_000);
    assert_eq!(fix_span(Micros(1_500_000), anchor, 30.0), TrackSpan::After);
    assert_eq!(fix_span(Micros(400_000), anchor, 30.0), TrackSpan::Before);
    assert_eq!(fix_span(Micros(1_010_000), anchor, 30.0), TrackSpan::Both);
    assert_eq!(fix_span(Micros(990_000), anchor, f64::NAN), TrackSpan::Both);
}

#[test]
fn a_fixed_path_keeps_the_old_one_on_the_anchors_side() {
    let old = TrackPath {
        mode: TrackMode::Object,
        samples: (0..10).map(|k| sample(k * 100, 0.0)).collect(),
    };
    let after = TrackPath {
        mode: TrackMode::Object,
        samples: (6..12).map(|k| sample(k * 100, 0.5)).collect(),
    };
    let joined_after = joined(&old, after, Micros(600), TrackSpan::After);
    let times: Vec<i64> = joined_after.samples.iter().map(|s| s.time.0).collect();
    assert_eq!(times, (0..12).map(|k| k * 100).collect::<Vec<_>>());
    assert!(
        joined_after.samples[..6]
            .iter()
            .all(|s| s.corners[0].0 == 0.0)
    );
    assert!(
        joined_after.samples[6..]
            .iter()
            .all(|s| s.corners[0].0 == 0.5)
    );

    let before = TrackPath {
        mode: TrackMode::Object,
        samples: (0..=3).map(|k| sample(k * 100, 0.5)).collect(),
    };
    let joined_before = joined(&old, before, Micros(300), TrackSpan::Before);
    let times: Vec<i64> = joined_before.samples.iter().map(|s| s.time.0).collect();
    assert_eq!(times, (0..10).map(|k| k * 100).collect::<Vec<_>>());
    assert!(
        joined_before.samples[..4]
            .iter()
            .all(|s| s.corners[0].0 == 0.5)
    );
    assert!(
        joined_before.samples[4..]
            .iter()
            .all(|s| s.corners[0].0 == 0.0)
    );
}

#[test]
fn fixing_a_frame_starts_from_the_tracks_mode_followers_and_box() {
    let (mut shell, temp) = open("fix");
    shell.select(Some("v".into()));
    shell.seek(Micros(700_000));
    assert!(!shell.fix_enabled(), "nothing tracked yet");
    let path = TrackPath {
        mode: TrackMode::Plane,
        samples: (0..10)
            .map(|i| sample(500_000 + i * 33_333, 0.01 * i as f64))
            .collect(),
    };
    shell
        .apply(Request::Track {
            target: "v".into(),
            followers: vec!["s".into()],
            tracking_id: "t".into(),
            anchor: Micros(500_000),
            path: path.clone(),
            masks: Vec::new(),
        })
        .unwrap();
    shell.select(Some("v".into()));
    shell.seek(Micros(700_000));
    assert!(shell.fix_enabled());
    shell.start_fixing().unwrap();
    let tracking = shell.tracking().unwrap().clone();
    assert_eq!(tracking.mode, TrackMode::Plane);
    assert_eq!(tracking.followers, vec![SegmentId::from("s")]);
    assert_eq!(tracking.time, Micros(700_000));
    let fixing = tracking.fixing.unwrap();
    assert_eq!(fixing.anchor, Micros(500_000));
    assert_eq!(fixing.path.samples.len(), 10);
    // The box at 700 ms is the sample at 699,994 µs: x from 0.06 to 0.16
    // canvas halves, y from 0 down to -0.1.
    let ((x0, y0), (x1, y1)) = tracking.canvas_box.unwrap();
    assert!((x0 - 1.06 * 960.0).abs() < 1e-6 && (x1 - 1.16 * 960.0).abs() < 1e-6);
    assert!((y0 - 540.0).abs() < 1e-6 && (y1 - 1.1 * 540.0).abs() < 1e-6);
    assert!(shell.track_ready());
    shell.stop_tracking();
    shell.select(Some("s".into()));
    assert!(
        !shell.fix_enabled(),
        "a follower's track is fixed on its target"
    );
    let _ = std::fs::remove_dir_all(&temp);
}

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
    let masks = shell.mask_choices();
    assert_eq!(
        masks
            .iter()
            .map(|c| c.segment_id.as_str())
            .collect::<Vec<_>>(),
        vec!["v", "s"],
        "the target's own mask first, not a locked row"
    );
    shell.toggle_follower(&"s".into());
    shell.toggle_mask_follower(&"s".into());
    let tracking = shell.tracking().unwrap();
    assert!(tracking.followers.is_empty(), "a pose or a mask, not both");
    assert_eq!(tracking.masks, vec![SegmentId::from("s")]);
    assert!(shell.track_ready(), "a mask is enough");
    shell.toggle_mask_follower(&"s".into());
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

#[test]
fn a_mask_follower_lands_keyed_and_is_kept_when_fixing() {
    let (mut shell, temp) = open("mask");
    let path = TrackPath {
        mode: TrackMode::Object,
        samples: (0..10)
            .map(|i| sample(500_000 + i * 33_333, 0.01 * i as f64))
            .collect(),
    };
    let draft = shell.session_draft().unwrap();
    let own = clip_path(draft, &"v".into(), &path).unwrap();
    // The clip fills the canvas, so its frame is the canvas.
    assert!((own.samples[3].corners[1].0 - path.samples[3].corners[1].0).abs() < 1e-9);
    shell
        .apply(Request::Track {
            target: "v".into(),
            followers: Vec::new(),
            tracking_id: "t".into(),
            anchor: Micros(500_000),
            path,
            masks: vec![("v".into(), own)],
        })
        .unwrap();
    let draft = shell.session_draft().unwrap();
    let clip = &draft.tracks[0].segments[0];
    assert!(
        cutlass_effects::segment_mask(draft, clip).is_some(),
        "a box mask"
    );
    shell.select(Some("v".into()));
    shell.seek(Micros(700_000));
    shell.start_fixing().unwrap();
    let tracking = shell.tracking().unwrap();
    assert_eq!(tracking.masks, vec![SegmentId::from("v")]);
    assert!(tracking.followers.is_empty());
    let _ = std::fs::remove_dir_all(&temp);
}

/// A ten-frame path from half a second, a box sliding right.
fn slide() -> TrackPath {
    TrackPath {
        mode: TrackMode::Object,
        samples: (0..10)
            .map(|i| {
                let x = 0.01 * i as f64;
                TrackSample {
                    time: Micros(500_000 + i * 33_333),
                    corners: [
                        (x - 0.1, 0.1),
                        (x + 0.1, 0.1),
                        (x + 0.1, -0.1),
                        (x - 0.1, -0.1),
                    ],
                }
            })
            .collect(),
    }
}

#[test]
fn a_sticker_sets_up_a_track_on_the_video_drawn_under_it() {
    let (mut shell, temp) = open("follower-setup");
    assert!(shell.start_following().is_err(), "nothing selected");
    shell.select(Some("s".into()));
    shell.seek(Micros(1_800_000));
    assert!(
        shell.follow_target().is_err(),
        "the playhead is off the sticker"
    );
    shell.seek(Micros(700_000));
    assert_eq!(shell.follow_target().unwrap(), SegmentId::from("v"));
    shell.start_following().unwrap();
    let tracking = shell.tracking().unwrap();
    assert_eq!(tracking.segment_id, SegmentId::from("v"));
    assert_eq!(tracking.followers, vec![SegmentId::from("s")]);
    assert!(tracking.masks.is_empty());
    assert_eq!(tracking.time, Micros(700_000));
    shell.set_track_box((200.0, 100.0), (400.0, 300.0)).unwrap();
    assert!(shell.track_ready(), "the item itself follows");
    shell.select(Some("x".into()));
    assert!(
        shell.start_following().is_err(),
        "a locked row cannot follow"
    );
    shell.select(Some("v".into()));
    assert!(
        shell.follow_target().is_err(),
        "nothing is drawn under the bottom video"
    );
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn the_video_followed_is_the_nearest_one_drawn_below() {
    let (mut shell, temp) = open("follower-below");
    let mut draft = shell.session_draft().unwrap().clone();
    // A second video row above the sticker's, and a photo row below it.
    let mut over = Track::main("over");
    over.segments
        .push(Segment::new("w", "m", TimeRange::new(0, 2_000_000)));
    draft.tracks.push(over);
    draft.materials.videos.push(MaterialVideo {
        id: "p".into(),
        kind: "photo".into(),
        ..MaterialVideo::default()
    });
    let mut photo = Track::main("photo");
    photo
        .segments
        .push(Segment::new("still", "p", TimeRange::new(0, 2_000_000)));
    draft.tracks.insert(1, photo);
    let stacked = temp.with_extension("stacked");
    let _ = std::fs::remove_dir_all(&stacked);
    cutlass_store::Folder::new(draft).save(&stacked).unwrap();
    shell.open_folder(&stacked).unwrap();
    shell.select(Some("s".into()));
    shell.seek(Micros(700_000));
    assert_eq!(
        shell.follow_target().unwrap(),
        SegmentId::from("v"),
        "not the video above it, not a photo"
    );
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::remove_dir_all(&stacked);
}

#[test]
fn a_follower_reads_its_track_fixes_a_frame_and_stops_following() {
    let (mut shell, temp) = open("follower-stop");
    shell
        .apply(Request::Track {
            target: "v".into(),
            followers: vec!["s".into()],
            tracking_id: "t".into(),
            anchor: Micros(500_000),
            path: slide(),
            masks: Vec::new(),
        })
        .unwrap();
    shell.select(Some("s".into()));
    let following = shell.following().unwrap();
    assert_eq!(following.target, SegmentId::from("v"));
    assert!(!following.by_mask);
    shell.seek(Micros(600_000));
    shell.start_fixing_following().unwrap();
    let tracking = shell.tracking().unwrap();
    assert_eq!(tracking.segment_id, SegmentId::from("v"));
    assert_eq!(tracking.followers, vec![SegmentId::from("s")]);
    assert!(tracking.fixing.is_some());
    assert!(
        tracking.canvas_box.is_some(),
        "the tracked box at the playhead"
    );
    shell.stop_following().unwrap();
    assert!(shell.tracking().is_none(), "the fix it was setting up goes");
    assert!(shell.following().is_none());
    let draft = shell.session_draft().unwrap();
    assert!(draft.materials.video_trackings.is_empty());
    shell.undo().unwrap();
    assert!(shell.following().is_some(), "one undo step");
    shell.select(Some("v".into()));
    assert!(shell.following().is_none(), "the target follows nothing");
    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn a_video_clips_mask_sets_up_a_track_on_the_clip_itself() {
    let (mut shell, temp) = open("mask-follow");
    shell.select(Some("s".into()));
    assert!(
        shell.start_mask_following().is_err(),
        "a sticker's mask has no video of its own to track"
    );
    shell.select(Some("v".into()));
    shell.seek(Micros(700_000));
    shell.start_mask_following().unwrap();
    let tracking = shell.tracking().unwrap();
    assert_eq!(tracking.segment_id, SegmentId::from("v"));
    assert_eq!(tracking.masks, vec![SegmentId::from("v")]);
    assert!(tracking.followers.is_empty());
    let own = clip_path(shell.session_draft().unwrap(), &"v".into(), &slide()).unwrap();
    shell
        .apply(Request::Track {
            target: "v".into(),
            followers: Vec::new(),
            tracking_id: "t".into(),
            anchor: Micros(500_000),
            path: slide(),
            masks: vec![("v".into(), own)],
        })
        .unwrap();
    let following = shell.following().unwrap();
    assert!(following.by_mask);
    assert_eq!(following.target, SegmentId::from("v"));
    let _ = std::fs::remove_dir_all(&temp);
}

use cutlass_model::{
    MaterialVideo, Micros, Segment, TimeRange, Track, TrackAttribute, TrackMode, TrackPath,
    TrackSample, TrackType, VIDEO_TRACKING, clip_at, key_offsets,
};

use super::*;

/// A 1920 by 1080 canvas with a video clip `v` over two seconds and a
/// sticker `s` on its own track over the same span, its middle at
/// (0.5, 0.25) in canvas units.
fn draft() -> Draft {
    let mut draft = Draft::empty("D");
    draft.canvas_config.width = 1920;
    draft.canvas_config.height = 1080;
    draft.materials.videos.push(MaterialVideo {
        id: "m".into(),
        width: 1920,
        height: 1080,
        ..MaterialVideo::default()
    });
    draft.tracks[0]
        .segments
        .push(Segment::new("v", "m", TimeRange::new(0, 2_000_000)));
    let mut sticker = Segment::new("s", "sticker-art", TimeRange::new(0, 2_000_000));
    sticker.clip.transform.x = 0.5;
    sticker.clip.transform.y = 0.25;
    let mut lane = Track::main("stickers");
    lane.kind = TrackType::Sticker;
    lane.segments.push(sticker);
    draft.tracks.push(lane);
    draft
}

/// A square target, `half` canvas halves across on x and as many pixels
/// tall, at `middle`, turned `degrees` counterclockwise.
fn square(middle: (f64, f64), half: f64, degrees: f64) -> [(f64, f64); 4] {
    let (sin, cos) = degrees.to_radians().sin_cos();
    [(-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)].map(|(x, y)| {
        // In pixels, y up, then back to canvas units.
        let (px, py) = (x * half * 960.0, y * half * 960.0);
        let (rx, ry) = (cos * px - sin * py, sin * px + cos * py);
        (middle.0 + rx / 960.0, middle.1 + ry / 540.0)
    })
}

/// Frames at 30 a second over the clip, with `at(t)` giving the target's
/// middle, half side and turn at frame `t`.
fn path(frames: usize, at: impl Fn(f64) -> ((f64, f64), f64, f64)) -> TrackPath {
    TrackPath {
        mode: TrackMode::Object,
        samples: (0..frames)
            .map(|i| {
                let (middle, half, turn) = at(i as f64);
                TrackSample {
                    time: Micros(i as i64 * 1_000_000 / 30),
                    corners: square(middle, half, turn),
                }
            })
            .collect(),
    }
}

fn run(draft: &mut Draft, path: &TrackPath) -> Result<(), EditError> {
    track(
        draft,
        &"v".into(),
        &["s".into()],
        &"t1".into(),
        Micros(0),
        path,
    )
}

fn sticker(draft: &Draft) -> &Segment {
    &draft.tracks[1].segments[0]
}

#[test]
fn a_sticker_follows_a_target_that_slides_with_two_keys() {
    let mut draft = draft();
    let slide = path(60, |t| ((-0.5 + 0.01 * t, 0.0), 0.1, 0.0));
    run(&mut draft, &slide).unwrap();
    let segment = sticker(&draft);
    // A straight slide needs only its ends.
    assert_eq!(key_offsets(segment).len(), 2);
    for frame in [0_i64, 17, 59] {
        let time = Micros(frame * 1_000_000 / 30);
        let (clip, _) = clip_at(segment, &draft.keyframe_graph_list, time).unwrap();
        assert!(
            (clip.transform.x - (0.5 + 0.01 * frame as f64)).abs() < 1e-6,
            "{frame}"
        );
        assert!((clip.transform.y - 0.25).abs() < 1e-9);
        assert!(clip.rotation.abs() < 1e-9);
    }
}

#[test]
fn a_sticker_turns_and_grows_about_a_turning_growing_target() {
    let mut draft = draft();
    let spin = path(60, |t| ((0.0, 0.0), 0.1 * (1.0 + 0.01 * t), 1.5 * t));
    run(&mut draft, &spin).unwrap();
    let segment = sticker(&draft);
    let keys = key_offsets(segment).len();
    assert!(keys > 2 && keys < 60, "{keys} keys");
    let frame = 40.0_f64;
    let time = Micros(40 * 1_000_000 / 30);
    let (clip, uniform) = clip_at(segment, &draft.keyframe_graph_list, time).unwrap();
    // The sticker's offset from the target, in pixels, turned and scaled.
    let (k, turn) = (1.0 + 0.01 * frame, (1.5 * frame).to_radians());
    let offset = (0.5 * 960.0, 0.25 * 540.0);
    let want = (
        k * (turn.cos() * offset.0 - turn.sin() * offset.1) / 960.0,
        k * (turn.sin() * offset.0 + turn.cos() * offset.1) / 540.0,
    );
    assert!(
        (clip.transform.x - want.0).abs() < 0.004,
        "{} {}",
        clip.transform.x,
        want.0
    );
    assert!(
        (clip.transform.y - want.1).abs() < 0.004,
        "{} {}",
        clip.transform.y,
        want.1
    );
    assert!(
        (clip.rotation - 1.5 * frame).abs() < 0.3,
        "{}",
        clip.rotation
    );
    // A new clip scales both ways at once.
    assert!(uniform.on);
    assert!((uniform.value - k).abs() < 0.002, "{}", uniform.value);
}

#[test]
fn the_path_is_kept_once_and_a_new_track_takes_its_place() {
    let mut draft = draft();
    let slide = path(30, |t| ((0.01 * t, 0.0), 0.1, 0.0));
    run(&mut draft, &slide).unwrap();
    assert_eq!(draft.materials.video_trackings.len(), 1);
    let record = &draft.materials.video_trackings[0];
    assert_eq!(record.kind, VIDEO_TRACKING);
    assert_eq!(record.fields["tracker_type"], 0);
    assert_eq!(record.fields["segment_id"], "v");
    assert_eq!(record.fields["path"].as_array().map(Vec::len), Some(30));
    assert!(sticker(&draft).extra_material_refs.contains(&"t1".into()));

    track(
        &mut draft,
        &"v".into(),
        &["s".into()],
        &"t2".into(),
        Micros(0),
        &slide,
    )
    .unwrap();
    assert_eq!(draft.materials.video_trackings.len(), 1);
    assert_eq!(draft.materials.video_trackings[0].id, "t2".into());
    let refs = &sticker(&draft).extra_material_refs;
    assert!(refs.contains(&"t2".into()) && !refs.contains(&"t1".into()));
}

#[test]
fn keys_outside_the_tracked_span_stay() {
    let mut draft = draft();
    let segment = &mut draft.tracks[1].segments[0];
    put_key(segment, "KFTypePositionX", Micros(1_900_000), 0.9);
    put_key(segment, "KFTypePositionX", Micros(500_000), 0.3);
    put_key(segment, ALPHA, Micros(500_000), 0.5);
    let slide = path(30, |t| ((0.01 * t, 0.0), 0.1, 0.0));
    run(&mut draft, &slide).unwrap();
    let offsets = key_offsets(sticker(&draft));
    assert!(offsets.contains(&Micros(1_900_000)), "{offsets:?}");
    assert!(offsets.contains(&Micros(500_000)), "the opacity key stays");
    let x = sticker(&draft)
        .common_keyframes
        .iter()
        .find(|group| group.property_type == "KFTypePositionX")
        .unwrap();
    assert!(
        x.keyframe_list
            .iter()
            .all(|key| key.time_offset != Micros(500_000))
    );
}

const ALPHA: &str = cutlass_model::ALPHA_KEY;

#[test]
fn a_bad_path_or_follower_is_refused() {
    let base = draft();
    let good = path(10, |t| ((0.01 * t, 0.0), 0.1, 0.0));
    let refused =
        |draft: &mut Draft, followers: &[&str], id: &str, anchor: i64, path: &TrackPath| {
            let followers: Vec<SegmentId> = followers.iter().map(|&f| f.into()).collect();
            track(
                draft,
                &"v".into(),
                &followers,
                &id.into(),
                Micros(anchor),
                path,
            )
            .is_err()
        };
    let mut d = base.clone();
    assert!(
        refused(&mut d, &["s"], "t", 5_000_000, &good),
        "anchor off the path"
    );
    assert!(refused(&mut d, &[], "t", 0, &good), "no follower");
    assert!(
        refused(&mut d, &["v"], "t", 0, &good),
        "the target follows itself"
    );
    assert!(refused(&mut d, &["s", "s"], "t", 0, &good), "twice");
    assert!(refused(&mut d, &["s"], "m", 0, &good), "a used id");
    assert!(refused(&mut d, &["s"], "", 0, &good), "no id");
    let mut backwards = good.clone();
    backwards.samples.reverse();
    assert!(refused(&mut d, &["s"], "t", 0, &backwards));
    let mut nan = good.clone();
    nan.samples[3].corners[1].0 = f64::NAN;
    assert!(refused(&mut d, &["s"], "t", 0, &nan));
    let mut late = good.clone();
    late.samples[9].time = Micros(3_000_000);
    assert!(refused(&mut d, &["s"], "t", 0, &late), "past the clip");
    let short = TrackPath {
        mode: TrackMode::Object,
        samples: good.samples[..1].to_vec(),
    };
    assert!(refused(&mut d, &["s"], "t", 0, &short));
    let mut locked = base.clone();
    locked.tracks[1].attribute = TrackAttribute(TrackAttribute::LOCK);
    assert!(refused(&mut locked, &["s"], "t", 0, &good));
    let mut audio = base.clone();
    audio.tracks[1].kind = TrackType::Audio;
    assert!(refused(&mut audio, &["s"], "t", 0, &good));
    let mut away = base.clone();
    away.tracks[1].segments[0].target = TimeRange::new(5_000_000, 1_000_000);
    assert!(refused(&mut away, &["s"], "t", 0, &good), "not on the path");
}

#[test]
fn thinning_keeps_the_ends_and_the_bends() {
    let line: Vec<(f64, f64)> = (0..10)
        .map(|i| (f64::from(i), 2.0 * f64::from(i)))
        .collect();
    assert_eq!(thin(&line, 0.01), vec![0, 9]);
    let bend: Vec<(f64, f64)> = (0..11)
        .map(|i| (f64::from(i), f64::from(5 - (i - 5_i32).abs())))
        .collect();
    assert_eq!(thin(&bend, 0.01), vec![0, 5, 10]);
    assert!(thin(&[], 1.0).is_empty());
    assert_eq!(thin(&[(0.0, 1.0)], 1.0), vec![0]);
}

use cutlass_model::{
    MaterialVideo, Micros, Segment, TimeRange, Track, TrackAttribute, TrackMode, TrackPath,
    TrackSample, TrackType, VIDEO_TRACKING, clip_at, key_offsets,
};

use cutlass_effects::{Mask, MaskShape};

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
        &[],
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
        &[],
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
                &[],
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

#[test]
fn a_key_at_the_anchor_keeps_the_followers_own_pose_there() {
    let mut draft = draft();
    let slide = path(60, |t| ((-0.5 + 0.01 * t, 0.0), 0.1, 0.0));
    let anchor = Micros(20 * 1_000_000 / 30);
    track(
        &mut draft,
        &"v".into(),
        &["s".into()],
        &[],
        &"t1".into(),
        anchor,
        &slide,
    )
    .unwrap();
    let segment = sticker(&draft);
    assert_eq!(key_offsets(segment).len(), 3, "the ends and the anchor");
    let (clip, _) = clip_at(segment, &draft.keyframe_graph_list, anchor).unwrap();
    assert_eq!((clip.transform.x, clip.transform.y), (0.5, 0.25));
}

/// A box in a clip's own frame: `half` across and up from `middle`, turned
/// `degrees` counterclockwise, over `frames` frames at 30 a second.
fn clip_path(frames: usize, at: impl Fn(f64) -> ((f64, f64), (f64, f64), f64)) -> TrackPath {
    TrackPath {
        mode: TrackMode::Object,
        samples: (0..frames)
            .map(|i| {
                let (middle, half, turn) = at(i as f64);
                let (sin, cos) = turn.to_radians().sin_cos();
                let corners = [(-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)].map(
                    |(x, y): (f64, f64)| {
                        let (px, py) = (x * half.0, y * half.1);
                        (
                            middle.0 + cos * px - sin * py,
                            middle.1 + sin * px + cos * py,
                        )
                    },
                );
                TrackSample {
                    time: Micros(i as i64 * 1_000_000 / 30),
                    corners,
                }
            })
            .collect(),
    }
}

fn mask_at(draft: &Draft, time: i64) -> Mask {
    let segment = &draft.tracks[0].segments[0];
    cutlass_effects::segment_mask(draft, segment).unwrap().at(
        segment,
        &draft.keyframe_graph_list,
        Micros(time),
    )
}

fn close(a: f32, b: f64) -> bool {
    (f64::from(a) - b).abs() < 1e-3
}

#[test]
fn a_clip_with_no_mask_gets_a_box_that_follows_the_target() {
    let mut draft = draft();
    let canvas = path(30, |t| ((-0.5 + 0.02 * t, 0.0), 0.1, 0.0));
    let own = clip_path(30, |t| ((-0.5 + 0.02 * t, 0.0), (0.1, 0.2), 0.0));
    track(
        &mut draft,
        &"v".into(),
        &[],
        &[("v".into(), own)],
        &"t1".into(),
        Micros(0),
        &canvas,
    )
    .unwrap();
    let start = mask_at(&draft, 0);
    assert_eq!(start.shape, MaskShape::Rectangle);
    assert!(close(start.x, -0.5) && close(start.y, 0.0));
    assert!(close(start.width, 0.1) && close(start.height, 0.2));
    let later = mask_at(&draft, 20 * 1_000_000 / 30);
    assert!(close(later.x, -0.1), "{}", later.x);
    let segment = &draft.tracks[0].segments[0];
    assert_eq!(key_offsets(segment).len(), 2, "a slide thins to its ends");
    assert!(
        segment
            .extra_material_refs
            .iter()
            .any(|id| id.as_str() == "t1")
    );
    let record = &draft.materials.video_trackings[0];
    assert_eq!(record.fields["masks"], serde_json::json!(["v"]));
}

#[test]
fn a_mask_keeps_its_place_size_and_turn_against_the_box() {
    let mut draft = draft();
    let mask = Mask {
        x: 0.2,
        y: 0.0,
        width: 0.05,
        height: 0.1,
        rotation: 10.0,
        feather: 0.3,
        ..Mask::new(MaskShape::Circle)
    };
    crate::edit::mask::set_mask(&mut draft, &"v".into(), &"mask".into(), Some(&mask)).unwrap();
    // The box sits at the middle, half 0.1 by 0.1, and turns 90 degrees
    // while it grows to twice its size.
    let canvas = path(31, |t| ((0.0, 0.0), 0.1, 3.0 * t));
    let own = clip_path(31, |t| {
        let grow = 1.0 + t / 30.0;
        ((0.0, 0.0), (0.1 * grow, 0.1 * grow), 3.0 * t)
    });
    track(
        &mut draft,
        &"v".into(),
        &[],
        &[("v".into(), own)],
        &"t1".into(),
        Micros(0),
        &canvas,
    )
    .unwrap();
    let end = mask_at(&draft, 1_000_000);
    assert_eq!(end.shape, MaskShape::Circle);
    // Two box halves to the right, turned a quarter, is two halves up.
    assert!(
        close(end.x, 0.0) && close(end.y, 0.4),
        "{} {}",
        end.x,
        end.y
    );
    assert!(close(end.width, 0.1) && close(end.height, 0.2));
    assert!(close(end.rotation, 100.0), "{}", end.rotation);
    assert!(close(end.feather, 0.3), "feather stays");
}

#[test]
fn a_clip_follows_by_pose_or_by_mask_not_both() {
    let mut draft = draft();
    let canvas = path(10, |t| ((0.01 * t, 0.0), 0.1, 0.0));
    let own = clip_path(10, |t| ((0.01 * t, 0.0), (0.1, 0.2), 0.0));
    let both = track(
        &mut draft,
        &"v".into(),
        &["s".into()],
        &[("s".into(), own.clone())],
        &"t1".into(),
        Micros(0),
        &canvas,
    );
    assert!(both.is_err());
    let none = track(
        &mut draft,
        &"v".into(),
        &[],
        &[],
        &"t1".into(),
        Micros(0),
        &canvas,
    );
    assert!(none.is_err());
    let twice = track(
        &mut draft,
        &"v".into(),
        &[],
        &[("v".into(), own.clone()), ("v".into(), own)],
        &"t1".into(),
        Micros(0),
        &canvas,
    );
    assert!(twice.is_err());
}

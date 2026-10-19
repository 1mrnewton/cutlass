//! Timeline scrub and keyframe-seek integration workflows.
//!
//! These drive the FFmpeg `Decoder`, so they only build/run when the `ffmpeg`
//! feature is on (FFmpeg-free targets use the platform-native backends instead).
#![cfg(feature = "ffmpeg")]

mod common;

use std::time::Duration;

use common::{assert_frame_shape, build_index, open_software, small_video_asset, target_ticks};

#[test]
fn keyframe_index_and_seek_to_frame_agree() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let index = build_index(&path);
    let mut dec = open_software(&path);

    for ms in [0_u64, 100, 250, 500, 750, 1_000] {
        let target = Duration::from_millis(ms);
        let ticks = target_ticks(&index, target);
        let kf = index
            .keyframe_at_or_before_ticks(ticks)
            .expect("keyframe at or before target");

        let frame = dec
            .seek_to_frame(target)
            .expect("seek")
            .expect("decoded frame");
        assert_frame_shape(&frame);
        assert!(
            frame.pts_ticks >= ticks,
            "frame pts {pts} should be >= target ticks {ticks} at {ms}ms",
            pts = frame.pts_ticks,
        );
        assert!(
            frame.pts_ticks >= kf,
            "frame pts {pts} should be >= keyframe {kf} at {ms}ms",
            pts = frame.pts_ticks,
        );

        let gop = index.gop_containing(ticks).expect("gop for target");
        assert!(gop.contains(kf));
        assert!(frame.pts_ticks >= gop.start);
    }
}

#[test]
fn scrub_session_jumps_non_monotonically() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let index = build_index(&path);
    let mut dec = open_software(&path);

    let targets = [
        Duration::from_millis(0),
        Duration::from_millis(400),
        Duration::from_millis(150),
        Duration::from_millis(900),
        Duration::from_millis(50),
    ];

    for target in targets {
        let ticks = target_ticks(&index, target);
        let frame = dec.seek_to_frame(target).expect("seek").expect("frame");
        assert!(frame.pts_ticks >= ticks);
        assert_frame_shape(&frame);
    }
}

#[test]
fn dirty_seek_lands_at_or_after_target() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let index = build_index(&path);

    let mut dec = open_software(&path);
    for ms in [200_u64, 350, 500] {
        let target = Duration::from_millis(ms);
        let ticks = target_ticks(&index, target);
        let frame = dec
            .seek_dirty_to_frame(target)
            .expect("dirty seek")
            .unwrap_or_else(|| panic!("dirty seek returned no frame at {ms}ms"));
        assert!(frame.pts_ticks >= ticks);
        assert_frame_shape(&frame);
    }
}

#[test]
fn seek_us_rescales_monotonically_with_keyframes() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let index = build_index(&path);
    let us: Vec<i64> = index
        .keyframe_ticks()
        .iter()
        .map(|&t| index.ticks_to_av_time_base(t))
        .collect();
    assert!(us.windows(2).all(|w| w[0] <= w[1]));
}

#[test]
fn seek_to_start_returns_first_presented_frame() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let index = build_index(&path);
    let mut dec = open_software(&path);

    let first_kf = index.keyframe_ticks()[0];
    let frame = dec
        .seek_to_frame(Duration::ZERO)
        .expect("seek")
        .expect("first frame");
    assert!(frame.pts_ticks >= first_kf);
    assert_frame_shape(&frame);
}

#[test]
fn reopen_decoder_rebuilds_independent_session() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let index = build_index(&path);

    let mut a = open_software(&path);
    let _ = a
        .seek_to_frame(Duration::from_millis(500))
        .expect("seek a")
        .expect("frame a");

    let mut b = open_software(&path);
    let frame = b
        .seek_to_frame(Duration::from_millis(100))
        .expect("seek b")
        .expect("frame b");
    let ticks = target_ticks(&index, Duration::from_millis(100));
    assert!(frame.pts_ticks >= ticks);
}

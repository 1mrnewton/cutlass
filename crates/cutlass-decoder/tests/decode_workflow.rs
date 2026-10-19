//! End-to-end decode workflows: open, sequential read, plane layout, init.
//!
//! These drive the FFmpeg `Decoder`, so they only build/run when the `ffmpeg`
//! feature is on (FFmpeg-free targets use the platform-native backends instead).
#![cfg(feature = "ffmpeg")]

mod common;

use std::time::Duration;

use common::{
    any_mp4_in_assets, assert_frame_shape, assert_yuv420p_plane_layout, decode_first_frame,
    open_auto, open_software, small_video_asset,
};
use cutlass_decoder::{self, HwAccel, PixelFormat};

#[test]
fn init_smoke() {
    cutlass_decoder::init();
}

#[test]
fn software_decode_first_frame_has_valid_planes() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let mut dec = open_software(&path);
    assert_eq!(dec.info().hw_accel, HwAccel::None);

    let frame = decode_first_frame(&mut dec);
    assert_frame_shape(&frame);
    assert_eq!(frame.width, dec.info().width);
    assert_eq!(frame.height, dec.info().height);

    if frame.format == PixelFormat::Yuv420p {
        assert_yuv420p_plane_layout(&frame);
    }
}

#[test]
fn sequential_decode_keeps_monotonic_pts() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let mut dec = open_software(&path);
    let mut prev_pts = i64::MIN;
    for _ in 0..12 {
        let Some(frame) = dec.next_frame().expect("decode") else {
            break;
        };
        assert!(frame.pts_ticks >= prev_pts);
        prev_pts = frame.pts_ticks;
        assert_frame_shape(&frame);
    }
    assert!(prev_pts > i64::MIN, "expected at least one decoded frame");
}

#[test]
fn forward_decode_then_backward_seek_resumes() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let mut dec = open_software(&path);

    for _ in 0..8 {
        let _ = dec.next_frame().expect("decode");
    }

    let early = Duration::from_millis(100);
    let frame = dec
        .seek_to_frame(early)
        .expect("seek")
        .expect("frame after backward seek");
    assert_frame_shape(&frame);
    assert!(dec.duration().is_some_and(|d| d > Duration::ZERO));
}

#[test]
fn hw_auto_matches_software_dimensions() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let sw = open_software(&path);
    let auto = open_auto(&path);

    assert_eq!(sw.info().width, auto.info().width);
    assert_eq!(sw.info().height, auto.info().height);
}

#[test]
fn hw_auto_decodes_to_supported_pixel_format() {
    let Some(path) = small_video_asset() else {
        return;
    };
    let mut dec = open_auto(&path);
    let frame = decode_first_frame(&mut dec);
    assert_frame_shape(&frame);
    assert!(matches!(
        frame.format,
        PixelFormat::Yuv420p | PixelFormat::Nv12 | PixelFormat::Rgba8
    ));
}

#[test]
fn multiple_assets_open_and_decode() {
    let Some(dir) = any_mp4_in_assets().map(|p| p.parent().unwrap().to_path_buf()) else {
        return;
    };
    let mut opened = 0usize;
    for entry in std::fs::read_dir(dir).expect("read assets").flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "mp4") {
            let mut dec = open_software(&path);
            let frame = decode_first_frame(&mut dec);
            assert_frame_shape(&frame);
            opened += 1;
            if opened >= 3 {
                break;
            }
        }
    }
    assert!(
        opened >= 1,
        "expected at least one mp4 in local-assets/assets/"
    );
}

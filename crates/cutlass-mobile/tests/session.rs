//! End-to-end exercise of the JSON command/state bridge over the C ABI:
//! synthesize an mp4 with the native encoder, then drive a `CutlassSession`
//! through the v1 editor flow — new -> import -> AddClip -> SplitClip -> read
//! state -> Save -> reopen -> state matches. Proves the bridge the SwiftUI app
//! (and the future AI agent) talks to.

#![cfg(target_vendor = "apple")] // synthesis uses the AVFoundation encoder

use std::time::{SystemTime, UNIX_EPOCH};

use cutlass_core::{
    ColorSpace, CpuImage, EncoderConfig, FrameData, PixelFormat, Plane, Rect, Rotation,
    VideoEncoder,
};
use cutlass_mobile::{
    cutlass_session_apply, cutlass_session_close, cutlass_session_import, cutlass_session_new,
    cutlass_session_open, cutlass_session_save, cutlass_session_state_json, cutlass_string_free,
};
use serde_json::Value;

fn temp_path(tag: &str, ext: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("cutlass_session_{tag}_{nanos}.{ext}"))
}

fn synthesize_mp4(path: &std::path::Path, w: u32, h: u32, frames: i64) {
    let rate = cutlass_core::Rational::FPS_30;
    let config = EncoderConfig::new((w, h), rate, ColorSpace::BT709);
    let mut encoder = cutlass_encoder::AvfEncoder::open(path, config).expect("open native encoder");
    for i in 0..frames {
        let mut pixels = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            pixels.extend_from_slice(&[20, 120, 220, 255]); // opaque blue, RGBA
        }
        let plane = Plane::new(pixels, w as usize * 4, h as usize);
        let frame = cutlass_core::VideoFrame::new(
            cutlass_core::RationalTime::new(i, rate),
            PixelFormat::Rgba8,
            ColorSpace::BT709,
            (w, h),
            Rect::from_size(w, h),
            Rotation::None,
            FrameData::Cpu(CpuImage::new(vec![plane])),
        );
        encoder.push(&frame).expect("push frame");
    }
    encoder.finish().expect("finish encode");
}

/// Marshal a Rust string into a C-ABI JSON call and parse the returned C string
/// back into JSON, freeing the heap string.
fn take_json(ptr: *mut std::os::raw::c_char) -> Value {
    assert!(!ptr.is_null(), "bridge returned a null json string");
    // SAFETY: `ptr` came from a `cutlass_session_*` call (CString::into_raw).
    let s = unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: free exactly once, with the matching allocator.
    unsafe { cutlass_string_free(ptr) };
    serde_json::from_str(&s).expect("bridge returned valid json")
}

fn state_of(session: *const cutlass_mobile::CutlassSession) -> Value {
    // SAFETY: `session` is a live handle, read-only here.
    take_json(unsafe { cutlass_session_state_json(session) })
}

#[test]
fn session_round_trips_through_the_json_bridge() {
    let (w, h) = (320u32, 240u32);
    let media_path = temp_path("media", "mp4");
    synthesize_mp4(&media_path, w, h, 30); // 1s @ 30fps

    // --- new --------------------------------------------------------------
    // SAFETY: null cache dir -> the bridge falls back to a temp dir.
    let session =
        unsafe { cutlass_session_new(30, 1, std::ptr::null(), 0) };
    assert!(!session.is_null(), "session should open on a headless host");

    let state = state_of(session);
    let track_id = state["tracks"][0]["id"]
        .as_u64()
        .expect("the new project has one video track");
    assert_eq!(state["tracks"].as_array().unwrap().len(), 1);
    assert!(state["tracks"][0]["clips"].as_array().unwrap().is_empty());

    // --- import -----------------------------------------------------------
    let media_str = media_path.to_str().unwrap();
    // SAFETY: `media_str` outlives the call; bytes are valid UTF-8.
    let import = take_json(unsafe {
        cutlass_session_import(session, media_str.as_ptr(), media_str.len())
    });
    assert_eq!(import["ok"], Value::Bool(true), "import: {import}");
    let media_id = import["media"].as_u64().unwrap();
    let fps_num = import["fps"]["num"].as_i64().unwrap();
    let fps_den = import["fps"]["den"].as_i64().unwrap();
    let src_dur = import["duration_ticks"].as_i64().unwrap();
    assert!(src_dur > 0);

    // --- AddClip (built as a JSON Command, the agent's path) ---------------
    let add = serde_json::json!({
        "Edit": { "AddClip": {
            "track": track_id,
            "media": media_id,
            "source": {
                "start": { "value": 0, "rate": { "num": fps_num, "den": fps_den } },
                "duration": { "value": src_dur, "rate": { "num": fps_num, "den": fps_den } }
            },
            "start": { "value": 0, "rate": { "num": 30, "den": 1 } }
        }}
    })
    .to_string();
    // SAFETY: `add` outlives the call.
    let add_res = take_json(unsafe {
        cutlass_session_apply(session, add.as_ptr(), add.len())
    });
    assert_eq!(add_res["ok"], Value::Bool(true), "add_clip: {add_res}");
    assert!(add_res["can_undo"].as_bool().unwrap());

    let state = state_of(session);
    let clips = state["tracks"][0]["clips"].as_array().unwrap();
    assert_eq!(clips.len(), 1, "one clip placed");
    let clip_id = clips[0]["id"].as_u64().unwrap();
    let clip_dur = clips[0]["duration"].as_i64().unwrap();
    assert!(clip_dur >= 2, "clip long enough to split");

    // --- SplitClip at the midpoint ---------------------------------------
    let at = clip_dur / 2;
    let split = serde_json::json!({
        "Edit": { "SplitClip": {
            "clip": clip_id,
            "at": { "value": at, "rate": { "num": 30, "den": 1 } }
        }}
    })
    .to_string();
    // SAFETY: `split` outlives the call.
    let split_res = take_json(unsafe {
        cutlass_session_apply(session, split.as_ptr(), split.len())
    });
    assert_eq!(split_res["ok"], Value::Bool(true), "split: {split_res}");

    let state = state_of(session);
    assert_eq!(
        state["tracks"][0]["clips"].as_array().unwrap().len(),
        2,
        "split produced two clips"
    );
    assert_eq!(state["media"].as_array().unwrap().len(), 1);

    // --- Save -------------------------------------------------------------
    let project_path = temp_path("project", "cutlass");
    let project_str = project_path.to_str().unwrap();
    // SAFETY: `project_str` outlives the call.
    let saved =
        unsafe { cutlass_session_save(session, project_str.as_ptr(), project_str.len()) };
    assert!(saved, "save should succeed");
    assert!(project_path.exists());

    // SAFETY: close the live handle exactly once.
    unsafe { cutlass_session_close(session) };

    // --- Reopen and compare ----------------------------------------------
    // SAFETY: project path is valid UTF-8; null cache dir.
    let reopened = unsafe {
        cutlass_session_open(
            project_str.as_ptr(),
            project_str.len(),
            std::ptr::null(),
            0,
        )
    };
    assert!(!reopened.is_null(), "reopen should succeed");

    let state = state_of(reopened);
    assert_eq!(
        state["tracks"][0]["clips"].as_array().unwrap().len(),
        2,
        "reopened project keeps both split halves"
    );
    assert_eq!(state["media"].as_array().unwrap().len(), 1);

    // SAFETY: close the reopened handle exactly once.
    unsafe { cutlass_session_close(reopened) };

    let _ = std::fs::remove_file(&media_path);
    let _ = std::fs::remove_file(&project_path);
}

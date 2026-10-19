//! The synthetic scrub preview drives the engine on a real (headless) GPU with
//! no assets, so it runs on any host. Exercises the C ABI surface end to end.

use cutlass_mobile::{
    cutlass_image_free, cutlass_preview_close, cutlass_preview_duration_seconds,
    cutlass_preview_open_demo, cutlass_preview_render,
};

#[test]
fn demo_preview_renders_distinct_frames_across_the_scrub() {
    let handle = cutlass_preview_open_demo();
    assert!(!handle.is_null(), "demo preview should open on a headless GPU");

    // SAFETY: `handle` is the live, non-null pointer just returned.
    let duration = unsafe { cutlass_preview_duration_seconds(handle) };
    assert!(duration > 0.0, "demo has a positive duration");

    let frame_at = |seconds: f64| {
        // SAFETY: single-threaded test, one handle, serialized calls.
        let img = unsafe { cutlass_preview_render(handle, seconds) };
        assert!(!img.data.is_null(), "render must produce pixels");
        assert_eq!(img.len, img.width as usize * img.height as usize * 4);
        // SAFETY: copy the pixels out before freeing the buffer.
        let bytes = unsafe { std::slice::from_raw_parts(img.data, img.len) }.to_vec();
        // SAFETY: `img` came from `cutlass_preview_render`, freed exactly once.
        unsafe { cutlass_image_free(img) };
        bytes
    };

    let start = frame_at(0.0);
    let mid = frame_at(duration * 0.5);
    assert!(start.iter().any(|&b| b != 0), "first frame has content");
    assert_ne!(
        start, mid,
        "the hue sweep must change across the timeline (scrub works)"
    );

    // SAFETY: close the live handle exactly once.
    unsafe { cutlass_preview_close(handle) };
}

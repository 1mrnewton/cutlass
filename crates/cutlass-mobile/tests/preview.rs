//! Interactive preview scrub path through the C ABI: open a session, query its
//! duration, render at several times, and confirm scrubbing moves the picture.

use cutlass_compositor::GpuContext;
use cutlass_mobile::{
    cutlass_image_free, cutlass_preview_close, cutlass_preview_duration_seconds,
    cutlass_preview_open_demo, cutlass_preview_render,
};

fn center_rgba(img: &cutlass_mobile::CutlassImage) -> [u8; 4] {
    // SAFETY: `img` came from the FFI; data/len describe a valid RGBA buffer.
    let pixels = unsafe { std::slice::from_raw_parts(img.data, img.len) };
    let cx = img.width / 2;
    let cy = img.height / 2;
    let idx = ((cy * img.width + cx) * 4) as usize;
    [
        pixels[idx],
        pixels[idx + 1],
        pixels[idx + 2],
        pixels[idx + 3],
    ]
}

#[test]
fn demo_preview_scrubs() {
    if GpuContext::new_headless_blocking().is_err() {
        eprintln!("skipping: no GPU adapter");
        return;
    }

    let handle = cutlass_preview_open_demo();
    assert!(!handle.is_null(), "demo preview failed to open");

    // SAFETY: `handle` is the live session from the open call above.
    let duration = unsafe { cutlass_preview_duration_seconds(handle) };
    assert!(
        (duration - 6.0).abs() < 1e-6,
        "expected ~6s demo, got {duration}"
    );

    // Render at the start and near the end; the hue sweep must differ.
    // SAFETY: serialized renders on a single live handle.
    let first = unsafe { cutlass_preview_render(handle, 0.0) };
    let last = unsafe { cutlass_preview_render(handle, duration - 0.1) };

    assert!(!first.data.is_null() && !last.data.is_null());
    assert!(first.width > 0 && first.height > 0);
    assert_eq!(
        first.len,
        (first.width as usize) * (first.height as usize) * 4
    );

    let a = center_rgba(&first);
    let b = center_rgba(&last);
    assert_eq!(a[3], 255, "frames must be opaque");
    assert_ne!(a, b, "scrubbing should change the frame (got {a:?} twice)");

    // SAFETY: each image is released exactly once.
    unsafe {
        cutlass_image_free(first);
        cutlass_image_free(last);
    }
    // SAFETY: the handle is closed exactly once.
    unsafe { cutlass_preview_close(handle) };
}

#[test]
fn null_handle_is_safe() {
    // SAFETY: null is an explicitly handled, no-op input for each call.
    unsafe {
        assert_eq!(cutlass_preview_duration_seconds(std::ptr::null()), 0.0);
        let img = cutlass_preview_render(std::ptr::null_mut(), 1.0);
        assert!(img.data.is_null());
        cutlass_image_free(img);
        cutlass_preview_close(std::ptr::null_mut());
    }
}

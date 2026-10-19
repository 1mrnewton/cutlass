//! End-to-end: export a clip to mp4, then decode + composite its first frame
//! back through the public FFI. Exercises the real platform decoder, the
//! compositor's frame path, and the C-ABI marshalling/ownership.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use cutlass_compositor::GpuContext;
use cutlass_mobile::{cutlass_image_free, cutlass_render_file_frame};
use cutlass_models::{
    CanvasAspect, CanvasSettings, Generator, Project, Rational, TimeRange, TrackKind,
};
use cutlass_render::{Renderer, export_to_file};

fn temp_mp4() -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("cutlass_mobile_fileframe_{nanos}.mp4"))
}

/// Export a short solid blue (`[40,80,160]`) 16:9 clip.
fn export_solid_clip(path: &Path) {
    let fps = Rational::FPS_30;
    let span = TimeRange::at_rate(0, 3, fps);

    let mut project = Project::new("mobile-file-frame", fps);
    project.timeline_mut().set_canvas(CanvasSettings {
        aspect: CanvasAspect::Wide16x9,
        background: [20, 30, 40],
    });
    let track = project.add_track(TrackKind::Sticker, "BG");
    project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [40, 80, 160, 255],
            },
            span,
        )
        .unwrap();

    let mut renderer = Renderer::new_headless().expect("headless renderer");
    let frames = export_to_file(&mut renderer, &project, path).expect("export mp4");
    assert_eq!(frames, 3);
}

#[test]
fn decodes_exported_clip_through_ffi() {
    if GpuContext::new_headless_blocking().is_err() {
        eprintln!("skipping: no GPU adapter");
        return;
    }

    let path = temp_mp4();
    export_solid_clip(&path);
    let path_str = path.to_str().expect("utf-8 path");

    // SAFETY: `path_str` bytes live for the call; the returned image is freed below.
    let image = unsafe {
        cutlass_render_file_frame(path_str.as_ptr(), path_str.len(), 320, 240)
    };
    let _ = std::fs::remove_file(&path);

    assert!(!image.data.is_null(), "decode/composite produced no frame");
    // 16:9 source fit into 320x240 -> 320x180.
    assert!(image.width <= 320 && image.height <= 240);
    assert_eq!(image.len, (image.width as usize) * (image.height as usize) * 4);

    // SAFETY: `image` came straight from the FFI and is freed exactly once.
    let pixels = unsafe { std::slice::from_raw_parts(image.data, image.len) };
    let cx = image.width / 2;
    let cy = image.height / 2;
    let idx = ((cy * image.width + cx) * 4) as usize;
    let center = &pixels[idx..idx + 4];
    assert_eq!(center[3], 255, "frame must be opaque");
    assert!(
        center[2] > center[0],
        "encoded color was blue-dominant; got {center:?}"
    );

    // SAFETY: release the single owning handle.
    unsafe { cutlass_image_free(image) };
}

#[test]
fn missing_file_returns_null() {
    let bogus = "/no/such/cutlass/file.mp4";
    // SAFETY: valid pointer/len; a failed decode yields a null-data image.
    let image = unsafe { cutlass_render_file_frame(bogus.as_ptr(), bogus.len(), 64, 64) };
    assert!(image.data.is_null());
    unsafe { cutlass_image_free(image) };
}

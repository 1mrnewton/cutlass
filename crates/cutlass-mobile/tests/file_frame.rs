//! Hermetic file-frame round-trip through the C ABI: synthesize an mp4 with the
//! native encoder, then render its first frame back via
//! `cutlass_render_file_frame`. The engine probes + decodes the file (FFmpeg on
//! a default desktop build; the platform-native backend under
//! `--features native-decode` or on device).

#![cfg(target_vendor = "apple")] // synthesis uses the AVFoundation encoder

use std::time::{SystemTime, UNIX_EPOCH};

use cutlass_core::{
    ColorSpace, CpuImage, EncoderConfig, FrameData, PixelFormat, Plane, Rect, Rotation,
    VideoEncoder,
};
use cutlass_mobile::{cutlass_image_free, cutlass_render_file_frame};

fn temp_mp4(tag: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("cutlass_mobile_{tag}_{nanos}.mp4"))
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

#[test]
fn renders_first_frame_of_a_synthesized_mp4() {
    let (w, h) = (320u32, 240u32);
    let path = temp_mp4("file_frame");
    synthesize_mp4(&path, w, h, 6);

    let path_str = path.to_str().expect("utf-8 path");
    // SAFETY: `path_str` outlives the call; bytes are valid UTF-8.
    let img = unsafe {
        cutlass_render_file_frame(path_str.as_ptr(), path_str.len(), 1280, 720)
    };

    assert!(!img.data.is_null(), "file frame should render");
    assert!(img.width > 0 && img.height > 0);
    assert_eq!(img.len, img.width as usize * img.height as usize * 4);
    // SAFETY: read the buffer before freeing.
    let any_nonzero = unsafe { std::slice::from_raw_parts(img.data, img.len) }
        .iter()
        .any(|&b| b != 0);
    assert!(any_nonzero, "decoded blue content should be present");

    // SAFETY: `img` came from the bridge, freed exactly once.
    unsafe { cutlass_image_free(img) };
    let _ = std::fs::remove_file(&path);
}

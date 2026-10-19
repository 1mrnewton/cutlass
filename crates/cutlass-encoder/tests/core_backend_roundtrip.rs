//! Round-trips the FFmpeg `cutlass-core` backends: encode RGBA frames to an MP4
//! through [`FfmpegEncoder`], then decode them back through [`FfmpegDecoder`],
//! using only the trait surface. This pins the desktop dev path end to end.

use cutlass_core::{
    ColorSpace, CpuImage, EncoderConfig, FrameData, PixelFormat, Plane, Rational, RationalTime,
    Rect, Rotation, VideoDecoder, VideoEncoder, VideoFrame,
};
use cutlass_decoder::FfmpegDecoder;
use cutlass_encoder::FfmpegEncoder;

fn rgba_frame(index: i64, width: u32, height: u32) -> VideoFrame {
    // A flat color that shifts per frame, with full opacity. Tight RGBA stride.
    let r = (index * 8) as u8;
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
    for _ in 0..(width as usize * height as usize) {
        pixels.extend_from_slice(&[r, 64, 128, 255]);
    }
    let plane = Plane::new(pixels, width as usize * 4, height as usize);
    VideoFrame::new(
        RationalTime::new(index, Rational::FPS_30),
        PixelFormat::Rgba8,
        ColorSpace::BT709,
        (width, height),
        Rect::from_size(width, height),
        Rotation::None,
        FrameData::Cpu(CpuImage::new(vec![plane])),
    )
}

#[test]
fn ffmpeg_backends_roundtrip_rgba_to_mp4() {
    let (width, height) = (64u32, 48u32);
    let frames = 15;

    let dir = std::env::temp_dir();
    let out = dir.join(format!(
        "cutlass_core_backend_roundtrip_{}.mp4",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&out);

    let mut encoder = FfmpegEncoder::open(
        &out,
        EncoderConfig::new((width, height), Rational::FPS_30, ColorSpace::BT709),
    )
    .expect("open encoder");

    for i in 0..frames {
        encoder
            .push(&rgba_frame(i, width, height))
            .expect("push frame");
    }
    encoder.finish().expect("finish encode");

    assert!(out.exists(), "encoder produced an output file");

    let mut decoder = FfmpegDecoder::open(&out).expect("open decoder");
    assert_eq!(decoder.info().display_size, (width, height));
    assert_eq!(decoder.info().pixel_format, PixelFormat::Yuv420p);

    let frame = decoder
        .frame_at(RationalTime::new(0, Rational::FPS_30))
        .expect("frame_at")
        .expect("a frame at t=0");
    assert!(frame.is_cpu());
    assert_eq!(frame.width(), width);
    assert_eq!(frame.height(), height);
    assert_eq!(frame.format, PixelFormat::Yuv420p);

    // The first frame's pts must be at the source start.
    assert_eq!(frame.pts.value, 0);

    let _ = std::fs::remove_file(&out);
}

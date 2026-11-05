//! The encoder trait: stills of every sample layout, and animations
//! packet by packet.

use cutlass_decode::{CodecId, CodecParams, Decoder, Packet};
use cutlass_encode::{
    BitDepth, ColorType, CpuFrame, Encoder, FrameInput, FrameTime, PictureInput, PixelFormat, Rgb8,
    StillImage, Transparency,
};

use super::animation::{check_animations, decode_packets};
use super::lossless_encode::webpinfo_ok;
use super::oracle::{decode_still, dwebp_rgba, scratch, synthetic_rgba, tool};
use crate::{WebpDecoder, WebpEncoder, WebpOptions};

fn encoder(options: WebpOptions) -> WebpEncoder {
    let mut params = CodecParams::video(CodecId::Webp, 1, 1);
    params.extra = options.to_extra();
    WebpEncoder::open(&params).expect("open")
}

fn still(options: WebpOptions, image: StillImage) -> Vec<u8> {
    let packets = encoder(options)
        .push_picture(&PictureInput { image })
        .expect("encode");
    assert_eq!(packets.len(), 1);
    packets[0].data.clone()
}

#[test]
fn options_round_trip_through_extra() {
    for options in [
        WebpOptions::default(),
        WebpOptions::lossless(),
        WebpOptions {
            lossless: true,
            quality: 0,
            effort: 9,
            exact: true,
        },
    ] {
        assert_eq!(
            WebpOptions::from_extra(&options.to_extra()).ok(),
            Some(options)
        );
    }
    assert_eq!(
        WebpOptions::from_extra(&[]).ok(),
        Some(WebpOptions::default())
    );
    for bad in [
        &[2u8, 75, 4, 0][..],
        &[0, 101, 4, 0],
        &[0, 75, 7, 0],
        &[1, 75, 10, 0],
        &[0, 75, 4],
    ] {
        assert!(WebpOptions::from_extra(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn every_still_layout_encodes_losslessly() {
    let (width, height) = (13u32, 9u32);
    let rgba = synthetic_rgba(width, height);
    let n = (width * height) as usize;
    let gray16: Vec<u16> = (0..n).map(|i| (i * 600) as u16).collect();
    let mut indexed = StillImage::from_samples(
        width,
        height,
        ColorType::Indexed,
        BitDepth::Four,
        (0..n).map(|i| (i % 3) as u16).collect(),
    );
    indexed.palette = vec![
        Rgb8 {
            red: 255,
            green: 0,
            blue: 0,
        },
        Rgb8 {
            red: 0,
            green: 255,
            blue: 0,
        },
        Rgb8 {
            red: 0,
            green: 0,
            blue: 255,
        },
    ];
    indexed.transparency = Transparency::PaletteAlpha(vec![255, 128]);
    let cases: Vec<(StillImage, Vec<u8>)> = vec![
        (
            StillImage::from_samples(
                width,
                height,
                ColorType::Rgba,
                BitDepth::Eight,
                rgba.iter().map(|&v| u16::from(v)).collect(),
            ),
            rgba.clone(),
        ),
        (
            StillImage::from_samples(
                width,
                height,
                ColorType::Rgb,
                BitDepth::Eight,
                rgba.as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|p| [p[0], p[1], p[2]])
                    .map(u16::from)
                    .collect(),
            ),
            rgba.as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect(),
        ),
        (
            StillImage::from_samples(
                width,
                height,
                ColorType::Gray,
                BitDepth::Sixteen,
                gray16.clone(),
            ),
            gray16
                .iter()
                .flat_map(|&g| {
                    let v = ((u32::from(g) * 255 + 32767) / 65535) as u8;
                    [v, v, v, 255]
                })
                .collect(),
        ),
        (
            indexed,
            (0..n)
                .flat_map(|i| [[255, 0, 0, 255], [0, 255, 0, 128], [0, 0, 255, 255]][i % 3])
                .collect(),
        ),
    ];
    let dwebp = tool("dwebp");
    let dir = scratch("encoder-stills");
    for (index, (image, expected)) in cases.into_iter().enumerate() {
        let options = WebpOptions {
            exact: true,
            ..WebpOptions::lossless()
        };
        let bytes = still(options, image);
        assert_eq!(decode_still(&bytes).rgba, expected, "case {index}");
        if let Some(dwebp) = &dwebp {
            let path = dir.join(format!("{index}.webp"));
            std::fs::write(&path, &bytes).expect("write");
            assert_eq!(dwebp_rgba(dwebp, &path).2, expected, "case {index} dwebp");
        }
    }
}

#[test]
fn an_icc_profile_is_written_and_read_back() {
    let mut image = StillImage::from_samples(4, 4, ColorType::Rgb, BitDepth::Eight, vec![7; 48]);
    image.icc_profile = b"not really a profile, but bytes".to_vec();
    let bytes = still(WebpOptions::lossless(), image);
    let mut decoder = WebpDecoder::open(&CodecParams::video(CodecId::Webp, 1, 1)).expect("open");
    let outputs = decoder
        .decode_picture(&Packet::video(bytes.clone(), 0, 0, 1, true))
        .expect("decode");
    assert_eq!(
        outputs[0].image.icc_profile,
        b"not really a profile, but bytes"
    );
    assert_eq!(decode_still(&bytes).rgba, [7, 7, 7, 255].repeat(16));
    if let Some(webpinfo) = tool("webpinfo") {
        let dir = scratch("encoder-icc");
        let path = dir.join("icc.webp");
        std::fs::write(&path, &bytes).expect("write");
        assert!(webpinfo_ok(&webpinfo, &path));
    }
}

/// Frames of a square moving over a gradient, some of them repeated, and
/// with transparent holes.
fn frames(count: usize, width: u32, height: u32) -> Vec<Vec<u8>> {
    let base = synthetic_rgba(width, height);
    (0..count)
        .map(|index| {
            let mut frame = base.clone();
            let step = (index / 2) as u32 * 3;
            for y in 0..height {
                for x in 0..width {
                    let inside = (x + 20).abs_diff(10 + step) < 6 && y.abs_diff(height / 2) < 5;
                    if inside {
                        let at = ((y * width + x) * 4) as usize;
                        frame[at..at + 4].copy_from_slice(&[
                            250,
                            250,
                            20,
                            if x % 4 == 0 { 0 } else { 255 },
                        ]);
                    }
                }
            }
            frame
        })
        .collect()
}

pub(crate) fn encode_animation(
    options: WebpOptions,
    frames: &[Vec<u8>],
    width: u32,
    height: u32,
) -> Vec<Packet> {
    let mut encoder = encoder(options);
    for (index, frame) in frames.iter().enumerate() {
        let time = FrameTime::new(index as i64 * 3, 3, 30).expect("time");
        let cpu =
            CpuFrame::new(width, height, PixelFormat::Rgba8, frame.clone(), time).expect("frame");
        assert!(
            encoder
                .push_frame(&FrameInput::Cpu(cpu))
                .expect("push")
                .is_empty()
        );
    }
    let packets = encoder.drain().expect("drain");
    assert_eq!(packets.len(), frames.len());
    packets
}

#[test]
fn lossless_animations_come_back_frame_for_frame() {
    let (width, height) = (41u32, 23u32);
    let source = frames(7, width, height);
    let options = WebpOptions {
        exact: true,
        ..WebpOptions::lossless()
    };
    let packets = encode_animation(options, &source, width, height);
    for (index, packet) in packets.iter().enumerate() {
        assert_eq!(
            (packet.pts, packet.duration, packet.timescale),
            (index as i64 * 3, 3, 30)
        );
    }
    let data: Vec<Vec<u8>> = packets.iter().map(|p| p.data.clone()).collect();
    let mut decoder = WebpDecoder::open(&CodecParams::video(CodecId::Webp, 1, 1)).expect("open");
    let decoded = decode_packets(&mut decoder, &data);
    assert!(decoded == source);
    let file: Vec<u8> = data.concat();
    let declared = u32::from_le_bytes([file[4], file[5], file[6], file[7]]) as usize;
    assert_eq!(declared + 8, file.len());
    let dir = scratch("encoder-anim");
    let path = dir.join("anim.webp");
    std::fs::write(&path, &file).expect("write");
    if let Some(webpinfo) = tool("webpinfo") {
        assert!(webpinfo_ok(&webpinfo, &path));
    }
    check_animations(&dir, &[path]);
}

#[test]
fn frames_of_another_size_and_reset_behave() {
    let mut encoder = encoder(WebpOptions::lossless());
    let time = FrameTime::new(0, 1, 10).expect("time");
    let push = |encoder: &mut WebpEncoder, w: u32, h: u32| {
        let frame = CpuFrame::new(
            w,
            h,
            PixelFormat::Rgba8,
            vec![9; (w * h * 4) as usize],
            time,
        )
        .expect("frame");
        encoder.push_frame(&FrameInput::Cpu(frame))
    };
    push(&mut encoder, 4, 4).expect("first");
    assert!(push(&mut encoder, 5, 4).is_err());
    encoder.reset();
    assert!(encoder.drain().expect("drain").is_empty());
    push(&mut encoder, 5, 4).expect("after reset");
    assert_eq!(encoder.drain().expect("drain").len(), 1);
}

#[test]
fn lossy_animations_and_stills_decode_like_libwebp() {
    let (width, height) = (41u32, 23u32);
    let source = frames(5, width, height);
    let packets = encode_animation(WebpOptions::default(), &source, width, height);
    let file: Vec<u8> = packets.iter().flat_map(|p| p.data.clone()).collect();
    let dir = scratch("encoder-lossy-anim");
    let path = dir.join("anim.webp");
    std::fs::write(&path, &file).expect("write");
    if let Some(webpinfo) = tool("webpinfo") {
        assert!(webpinfo_ok(&webpinfo, &path));
    }
    check_animations(&dir, &[path]);

    let rgba = synthetic_rgba(width, height);
    let image = StillImage::from_samples(
        width,
        height,
        ColorType::Rgba,
        BitDepth::Eight,
        rgba.iter().map(|&v| u16::from(v)).collect(),
    );
    let bytes = still(WebpOptions::default(), image);
    let ours = decode_still(&bytes);
    // Alpha stays exact through the lossless `ALPH` chunk.
    for (a, b) in ours
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .zip(rgba.as_chunks::<4>().0)
    {
        assert_eq!(a[3], b[3]);
    }
    if let Some(dwebp) = tool("dwebp") {
        let path = dir.join("still.webp");
        std::fs::write(&path, &bytes).expect("write");
        assert!(dwebp_rgba(&dwebp, &path).2 == ours.rgba);
    }
}

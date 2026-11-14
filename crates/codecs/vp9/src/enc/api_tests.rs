use super::Vp9Encoder;
use crate::bits::BitWriter;
use crate::decode::{Decoder, Vp9Decoder};
use crate::geom;
use crate::header;
use crate::register_decoders;
use crate::register_encoders;
use cutlass_decode::video::VideoDecoder;
use cutlass_decode::{CodecError, VideoDecodeFactory};
use cutlass_encode::video::{VideoEncoder, VideoInput};
use cutlass_encode::{
    CodecId, CodecParams, Packet, PixelFormat, Plane, VideoEncodeFactory, VideoFrame,
    request_video_cancel,
};

fn solid(width: u32, height: u32, y: u8, u: u8, v: u8) -> VideoFrame {
    let chroma_w = width / 2;
    let chroma_h = height / 2;
    let luma = usize::try_from(width * height).expect("luma");
    let chroma = usize::try_from(chroma_w * chroma_h).expect("chroma");
    VideoFrame::yuv420p(
        width,
        height,
        Plane::tight(width, height, vec![y; luma]).expect("y"),
        Plane::tight(chroma_w, chroma_h, vec![u; chroma]).expect("u"),
        Plane::tight(chroma_w, chroma_h, vec![v; chroma]).expect("v"),
    )
    .expect("frame")
}

fn input(frame: VideoFrame) -> VideoInput {
    VideoInput {
        frame,
        pts: 0,
        duration: 1,
        timescale: 30,
    }
}

#[test]
fn lossless_constant_matches_the_source() {
    let frame = solid(16, 16, 40, 90, 140);
    let mut encoder = Vp9Encoder::with_quantizer(16, 16, 0).expect("encoder");
    let packets = encoder.push(&input(frame.clone())).expect("encode");
    assert_eq!(packets.len(), 1);
    assert!(packets[0].keyframe);
    let header = parsed_header(&packets[0].data).expect("header");
    assert!(header.key_frame);
    assert!(header.lossless);
    assert!(!header.error_resilient);
    assert_eq!(header.width, 16);
    assert_eq!(header.height, 16);
    let picture = Decoder::new()
        .decode(&packets[0].data)
        .expect("decode")
        .expect("shown");
    assert_eq!(picture.y, frame.y.samples);
    assert_eq!(picture.u, frame.u.samples);
    assert_eq!(picture.v, frame.v.samples);
}

#[test]
fn color_and_render_size_are_on_the_picture() {
    use cutlass_decode::{ColorPrimaries, ColorRange, MatrixCoefficients, Transfer};

    let frame = solid(16, 16, 40, 90, 140);
    let mut encoder = Vp9Encoder::with_quantizer(16, 16, 0).expect("encoder");
    let packets = encoder.push(&input(frame)).expect("encode");
    let params = CodecParams::with_format(CodecId::Vp9, 16, 16, PixelFormat::Yuv420p);
    let mut decoder = Vp9Decoder::open(&params).expect("decoder");
    let shown = decoder
        .push_all(&packets[0])
        .expect("decode")
        .pop()
        .expect("shown")
        .frame;
    assert_eq!(shown.width, 16);
    assert_eq!(shown.height, 16);
    assert_eq!(shown.y.width, 16);
    assert_eq!(shown.y.height, 16);
    assert_eq!(shown.render_width, 16);
    assert_eq!(shown.render_height, 16);
    assert_eq!(shown.range, ColorRange::Studio);
    assert_eq!(shown.primaries, ColorPrimaries::Bt601);
    assert_eq!(shown.transfer, Transfer::Bt601);
    assert_eq!(shown.matrix, MatrixCoefficients::Bt601);

    let rewritten = render_size(&packets[0].data, 32, 18);
    let header = parsed_header(&rewritten).expect("header");
    assert_eq!(header.width, 16);
    assert_eq!(header.height, 16);
    assert_eq!(header.render_width, 32);
    assert_eq!(header.render_height, 18);
    let mut decoder = Vp9Decoder::open(&params).expect("decoder");
    let again = decoder
        .push_all(&Packet::video(rewritten, 0, 1, 30, true))
        .expect("decode")
        .pop()
        .expect("shown")
        .frame;
    assert_eq!(again.width, 16);
    assert_eq!(again.height, 16);
    assert_eq!(again.y.samples, shown.y.samples);
    assert_eq!(again.u.samples, shown.u.samples);
    assert_eq!(again.v.samples, shown.v.samples);
    assert_eq!(again.render_width, 32);
    assert_eq!(again.render_height, 18);
    assert_eq!(again.range, ColorRange::Studio);
    assert_eq!(again.primaries, ColorPrimaries::Bt601);

    let mut bt709 = bits_of(&packets[0].data);
    bt709[32] = 0;
    bt709[33] = 1;
    bt709[34] = 0;
    bt709[35] = 1;
    let mut decoder = Vp9Decoder::open(&params).expect("decoder");
    let colored = decoder
        .push_all(&Packet::video(bytes_of(&bt709), 0, 1, 30, true))
        .expect("decode")
        .pop()
        .expect("shown")
        .frame;
    assert_eq!(colored.primaries, ColorPrimaries::Bt709);
    assert_eq!(colored.transfer, Transfer::Bt709);
    assert_eq!(colored.matrix, MatrixCoefficients::Bt709);
    assert_eq!(colored.range, ColorRange::Full);
    assert_eq!(colored.y.samples, shown.y.samples);

    assert_color_space(
        &params,
        &packets[0].data,
        &shown.y.samples,
        3,
        ColorPrimaries::Bt601,
        Transfer::Bt601,
        MatrixCoefficients::Bt601,
    );
    assert_color_space(
        &params,
        &packets[0].data,
        &shown.y.samples,
        5,
        ColorPrimaries::Bt2020,
        Transfer::Bt2020,
        MatrixCoefficients::Bt2020,
    );
    for space in [0u8, 4, 6] {
        assert_color_space(
            &params,
            &packets[0].data,
            &shown.y.samples,
            space,
            ColorPrimaries::Unspecified,
            Transfer::Unspecified,
            MatrixCoefficients::Unspecified,
        );
    }
    let mut rgb = bits_of(&packets[0].data);
    rgb[32] = 1;
    rgb[33] = 1;
    rgb[34] = 1;
    let err = Vp9Decoder::open(&params)
        .expect("decoder")
        .push_all(&Packet::video(bytes_of(&rgb), 0, 1, 30, true))
        .expect_err("profile 0 rgb");
    assert!(
        err.to_string().contains("profile 0 rgb"),
        "profile 0 rgb was {err}"
    );

    let original = parsed_header(&packets[0].data).expect("header");
    let mut intra = BitWriter::new();
    intra.write(2, 2);
    intra.write(0, 2);
    intra.write_bit(false);
    intra.write_bit(true);
    intra.write_bit(false);
    intra.write_bit(true);
    intra.write_bit(true);
    intra.write(0x49, 8);
    intra.write(0x83, 8);
    intra.write(0x42, 8);
    intra.write(0xff, 8);
    intra.write(15, 16);
    intra.write(15, 16);
    intra.write_bit(false);
    intra.write(0, 2);
    intra.write(0, 6);
    intra.write(0, 3);
    intra.write_bit(false);
    intra.write(0, 8);
    intra.write(0, 3);
    intra.write_bit(false);
    write_tile_bits(&mut intra, 16);
    intra.write(u32::try_from(original.compressed_bytes).unwrap_or(0), 16);
    let mut intra_bytes = intra.finish();
    intra_bytes.extend_from_slice(&packets[0].data[original.header_bytes..]);
    let hidden = decoder
        .push_all(&Packet::video(intra_bytes, 1, 1, 30, false))
        .expect("intra-only");
    assert!(hidden.is_empty());
    let kept = decoder
        .push_all(&Packet::video(vec![0x88], 2, 1, 30, false))
        .expect("show existing")
        .pop()
        .expect("shown")
        .frame;
    assert_eq!(kept.width, 16);
    assert_eq!(kept.height, 16);
    assert_eq!(kept.y.samples, shown.y.samples);
    assert_eq!(kept.range, ColorRange::Studio);
    assert_eq!(kept.primaries, ColorPrimaries::Bt601);
    assert_eq!(kept.transfer, Transfer::Bt601);
    assert_eq!(kept.matrix, MatrixCoefficients::Bt601);
}

fn assert_color_space(
    params: &CodecParams,
    frame: &[u8],
    samples: &[u8],
    space: u8,
    primaries: cutlass_decode::ColorPrimaries,
    transfer: cutlass_decode::Transfer,
    matrix: cutlass_decode::MatrixCoefficients,
) {
    let mut bits = bits_of(frame);
    bits[32] = (space >> 2) & 1;
    bits[33] = (space >> 1) & 1;
    bits[34] = space & 1;
    let shown = Vp9Decoder::open(params)
        .expect("decoder")
        .push_all(&Packet::video(bytes_of(&bits), 0, 1, 30, true))
        .unwrap_or_else(|err| panic!("space {space}: {err}"))
        .pop()
        .unwrap_or_else(|| panic!("space {space} showed nothing"))
        .frame;
    assert_eq!(shown.primaries, primaries, "space {space}");
    assert_eq!(shown.transfer, transfer, "space {space}");
    assert_eq!(shown.matrix, matrix, "space {space}");
    assert_eq!(
        shown.range,
        cutlass_decode::ColorRange::Studio,
        "space {space}"
    );
    assert_eq!(shown.y.samples, samples, "space {space}");
}

fn bits_of(frame: &[u8]) -> Vec<u8> {
    let mut bits = Vec::new();
    for byte in frame {
        for shift in (0..8).rev() {
            bits.push((byte >> shift) & 1);
        }
    }
    bits
}

fn bytes_of(bits: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0u8; bits.len().div_ceil(8)];
    for (index, bit) in bits.iter().enumerate() {
        if *bit == 1 {
            bytes[index / 8] |= 1 << (7 - (index % 8));
        }
    }
    bytes
}

fn render_size(frame: &[u8], render_w: u32, render_h: u32) -> Vec<u8> {
    let mut bits = Vec::new();
    for byte in frame {
        for shift in (0..8).rev() {
            bits.push((byte >> shift) & 1);
        }
    }
    bits[68] = 1;
    let mut extra = Vec::new();
    for value in [render_w - 1, render_h - 1] {
        for shift in (0..16).rev() {
            extra.push(((value >> shift) & 1) as u8);
        }
    }
    let mut out = bits[..=68].to_vec();
    out.extend(extra);
    out.extend_from_slice(&bits[69..]);
    let mut bytes = vec![0u8; out.len().div_ceil(8)];
    for (index, bit) in out.iter().enumerate() {
        if *bit == 1 {
            bytes[index / 8] |= 1 << (7 - (index % 8));
        }
    }
    bytes
}

#[test]
fn lossless_ramp_and_odd_blocks_match_the_source() {
    let mut frames = Vec::new();
    frames.push(ramp(8, 8));
    frames.push(ramp(16, 16));
    frames.push(ramp(24, 16));
    frames.push(solid(64, 64, 200, 20, 80));
    for frame in frames {
        let mut encoder =
            Vp9Encoder::with_quantizer(frame.width, frame.height, 0).expect("encoder");
        let packets = encoder.push(&input(frame.clone())).expect("encode");
        let picture = Decoder::new()
            .decode(&packets[0].data)
            .expect("decode")
            .expect("shown");
        assert_eq!(picture.y, frame.y.samples);
        assert_eq!(picture.u, frame.u.samples);
        assert_eq!(picture.v, frame.v.samples);
    }
}

fn ramp(width: u32, height: u32) -> VideoFrame {
    let chroma_w = width / 2;
    let chroma_h = height / 2;
    let mut y = Vec::new();
    let mut u = Vec::new();
    let mut v = Vec::new();
    for row in 0..height {
        for col in 0..width {
            y.push(u8::try_from((row * 3 + col * 5) % 251).unwrap_or(0));
        }
    }
    for row in 0..chroma_h {
        for col in 0..chroma_w {
            u.push(u8::try_from((row * 9 + col) % 200).unwrap_or(0));
            v.push(u8::try_from((col * 7 + 30) % 180).unwrap_or(0));
        }
    }
    VideoFrame::yuv420p(
        width,
        height,
        Plane::tight(width, height, y).expect("y"),
        Plane::tight(chroma_w, chroma_h, u).expect("u"),
        Plane::tight(chroma_w, chroma_h, v).expect("v"),
    )
    .expect("frame")
}

#[test]
fn lossy_keyframes_meet_the_psnr_gates() {
    assert_psnr(solid(64, 64, 64, 128, 128), 32, 45.0, "flat");
    assert_psnr(ramp(64, 64), 32, 38.0, "ramp");
    assert_psnr(checker(64, 64), 32, 28.0, "checker");
}

fn assert_psnr(frame: VideoFrame, qindex: u8, floor: f64, name: &str) {
    let mut encoder =
        Vp9Encoder::with_quantizer(frame.width, frame.height, qindex).expect("encoder");
    let packets = encoder.push(&input(frame.clone())).expect("encode");
    let picture = Decoder::new()
        .decode(&packets[0].data)
        .expect("decode")
        .expect("shown");
    let score = psnr(&frame.y.samples, &picture.y);
    assert!(score >= floor, "{name} psnr {score} floor {floor}");
}

fn checker(width: u32, height: u32) -> VideoFrame {
    let chroma_w = width / 2;
    let chroma_h = height / 2;
    let mut y = Vec::new();
    let mut u = Vec::new();
    let mut v = Vec::new();
    for row in 0..height {
        for col in 0..width {
            let on = ((row / 4) + (col / 4)).is_multiple_of(2);
            y.push(if on { 220 } else { 30 });
        }
    }
    for row in 0..chroma_h {
        for col in 0..chroma_w {
            let on = ((row / 2) + (col / 2)).is_multiple_of(2);
            u.push(if on { 160 } else { 80 });
            v.push(if on { 140 } else { 100 });
        }
    }
    VideoFrame::yuv420p(
        width,
        height,
        Plane::tight(width, height, y).expect("y"),
        Plane::tight(chroma_w, chroma_h, u).expect("u"),
        Plane::tight(chroma_w, chroma_h, v).expect("v"),
    )
    .expect("frame")
}

fn psnr(src: &[u8], decoded: &[u8]) -> f64 {
    let mut error = 0.0f64;
    for (left, right) in src.iter().zip(decoded) {
        let delta = f64::from(i32::from(*left) - i32::from(*right));
        error += delta * delta;
    }
    if error == 0.0 {
        return 100.0;
    }
    let count = f64::from(u32::try_from(src.len()).unwrap_or(u32::MAX));
    10.0 * (255.0 * 255.0 * count / error).log10()
}

#[test]
fn lossless_motion_matches_the_source() {
    let mut encoder = Vp9Encoder::with_quantizer(32, 16, 0).expect("encoder");
    let mut decoder = Decoder::new();
    for index in 0..8 {
        let frame = moving(32, 16, index);
        let packets = encoder.push(&input(frame.clone())).expect("encode");
        assert_eq!(packets[0].keyframe, index == 0);
        let picture = decoder
            .decode(&packets[0].data)
            .expect("decode")
            .expect("shown");
        assert_eq!(
            first_mismatch(&picture.y, &frame.y.samples),
            None,
            "y {index}"
        );
        assert_eq!(
            first_mismatch(&picture.u, &frame.u.samples),
            None,
            "u {index}"
        );
        assert_eq!(
            first_mismatch(&picture.v, &frame.v.samples),
            None,
            "v {index}"
        );
    }
}

#[test]
fn lossy_motion_meets_the_psnr_gate() {
    let mut encoder = Vp9Encoder::with_quantizer(64, 64, 32).expect("encoder");
    let mut decoder = Decoder::new();
    let mut scores = Vec::new();
    for index in 0..16 {
        let frame = moving(64, 64, index);
        let packets = encoder.push(&input(frame.clone())).expect("encode");
        let picture = decoder
            .decode(&packets[0].data)
            .expect("decode")
            .expect("shown");
        scores.push(psnr(&frame.y.samples, &picture.y));
    }
    let mean = scores.iter().sum::<f64>() / f64::from(u32::try_from(scores.len()).unwrap_or(1));
    assert!(mean >= 32.0, "mean psnr {mean} scores {scores:?}");
}

fn moving(width: u32, height: u32, shift: u32) -> VideoFrame {
    let chroma_w = width / 2;
    let chroma_h = height / 2;
    let mut y = Vec::new();
    let mut u = Vec::new();
    let mut v = Vec::new();
    for row in 0..height {
        for col in 0..width {
            let src = i32::try_from(col).unwrap_or(0) - i32::try_from(shift).unwrap_or(0);
            let sample = if src < 0 {
                16
            } else {
                u8::try_from((u32::try_from(src).unwrap_or(0) * 9 + row * 3) % 220 + 20)
                    .unwrap_or(20)
            };
            y.push(sample);
        }
    }
    let chroma_shift = shift / 2;
    for row in 0..chroma_h {
        for col in 0..chroma_w {
            let src = i32::try_from(col).unwrap_or(0) - i32::try_from(chroma_shift).unwrap_or(0);
            let sample = if src < 0 {
                90
            } else {
                u8::try_from((u32::try_from(src).unwrap_or(0) * 5 + row) % 80 + 80).unwrap_or(80)
            };
            u.push(sample);
            v.push(sample.saturating_add(20));
        }
    }
    VideoFrame::yuv420p(
        width,
        height,
        Plane::tight(width, height, y).expect("y"),
        Plane::tight(chroma_w, chroma_h, u).expect("u"),
        Plane::tight(chroma_w, chroma_h, v).expect("v"),
    )
    .expect("frame")
}

fn first_mismatch(got: &[u8], want: &[u8]) -> Option<usize> {
    got.iter().zip(want).position(|(left, right)| left != right)
}

#[test]
fn thirty_frames_stay_near_500_kbps() {
    let mut params = CodecParams::with_format(CodecId::Vp9, 160, 96, PixelFormat::Yuv420p);
    params.bitrate = 500_000;
    let mut encoder = Vp9Encoder::open(&params).expect("encoder");
    let mut total = 0usize;
    for index in 0..30u32 {
        let frame = moving(160, 96, index);
        let packets = encoder
            .push(&VideoInput {
                frame,
                pts: i64::from(index),
                duration: 1,
                timescale: 30,
            })
            .expect("encode");
        total += packets[0].data.len();
    }
    let target = 500_000.0 / 8.0;
    let ratio = total as f64 / target;
    assert!(
        (0.70..=1.25).contains(&ratio),
        "bytes {total} ratio {ratio}"
    );
}

#[test]
fn factory_bitrate_zero_is_the_default_quantizer() {
    let params = CodecParams::with_format(CodecId::Vp9, 16, 16, PixelFormat::Yuv420p);
    let encoder = Vp9Encoder::open(&params).expect("open");
    assert_eq!(encoder.qindex(), super::DEFAULT_Q);
}

#[test]
fn lifecycle_keeps_references_until_reset_or_cancel() {
    let mut encoder = Vp9Encoder::with_quantizer(16, 16, 0).expect("encoder");
    assert!(encoder.drain().expect("drain").is_empty());
    let first = encoder
        .push(&input(solid(16, 16, 10, 20, 30)))
        .expect("key");
    assert!(first[0].keyframe);
    let second = encoder.push(&input(moving(16, 16, 1))).expect("inter");
    assert!(!second[0].keyframe);
    let bad = encoder.push(&input(solid(32, 16, 1, 2, 3)));
    assert!(matches!(
        bad,
        Err(CodecError::InvalidApi {
            what: "picture size"
        })
    ));
    let third = encoder
        .push(&input(moving(16, 16, 2)))
        .expect("still inter");
    assert!(!third[0].keyframe);
    encoder.reset();
    let again = encoder
        .push(&input(solid(16, 16, 10, 20, 30)))
        .expect("reset key");
    assert!(again[0].keyframe);
    assert!(encoder.drain().expect("drain again").is_empty());
    request_video_cancel(&encoder.cancel_flag());
    let cancelled = encoder.push(&input(solid(16, 16, 10, 20, 30)));
    assert!(matches!(cancelled, Err(CodecError::Cancelled)));
}

#[test]
fn open_rejects_the_sizes_and_formats_outside_profile_0() {
    assert!(matches!(
        Vp9Encoder::with_quantizer(0, 8, 0),
        Err(CodecError::InvalidApi { .. })
    ));
    // Any size codes: partial superblocks and odd edges.
    assert!(Vp9Encoder::with_quantizer(12, 8, 0).is_ok());
    assert!(Vp9Encoder::with_quantizer(13, 7, 0).is_ok());
    assert!(matches!(
        Vp9Encoder::with_quantizer(65537, 8, 0),
        Err(CodecError::Limit { .. })
    ));
    let rgba = CodecParams::with_format(CodecId::Vp9, 16, 16, PixelFormat::Rgba8);
    assert!(matches!(
        Vp9Encoder::open(&rgba),
        Err(CodecError::Unsupported { .. })
    ));
    let wide = CodecParams::with_format(CodecId::Vp9, 4097, 8, PixelFormat::Yuv420p);
    assert!(Vp9Decoder::open(&wide).is_ok());
    let past_edge = CodecParams::with_format(CodecId::Vp9, 16_833, 8, PixelFormat::Yuv420p);
    assert!(matches!(
        Vp9Decoder::open(&past_edge),
        Err(CodecError::Limit { .. })
    ));
    let past_area = CodecParams::with_format(CodecId::Vp9, 16_832, 2_119, PixelFormat::Yuv420p);
    assert!(matches!(
        Vp9Decoder::open(&past_area),
        Err(CodecError::Limit { .. })
    ));
    let params = CodecParams::with_format(CodecId::Vp9, 16, 16, PixelFormat::Yuv420p);
    let mut decoder = Vp9Decoder::open(&params).expect("decoder");
    let empty = decoder.push_all(&Packet::video(Vec::new(), 0, 1, 30, true));
    assert!(matches!(empty, Err(CodecError::InvalidApi { .. })));
    let err = decoder
        .push_all(&Packet::video(vec![0xb0], 0, 1, 30, true))
        .expect_err("truncated profile 3");
    assert!(!err.to_string().contains("vp9 profile"), "{err}");
    let err = decoder
        .push_all(&Packet::video(vec![0x90], 0, 1, 30, true))
        .expect_err("truncated profile 2");
    assert!(!err.to_string().contains("vp9 profile"), "{err}");
    let err = match Decoder::new().decode(&oversized_edge()) {
        Err(err) => err,
        Ok(_) => panic!("oversized edge was accepted"),
    };
    assert!(matches!(err, CodecError::Limit { .. }), "{err}");
    let err = match Decoder::new().decode(&oversized_area()) {
        Err(err) => err,
        Ok(_) => panic!("oversized area was accepted"),
    };
    assert!(matches!(err, CodecError::Limit { .. }), "{err}");
    assert!(decoder.drain().expect("drain").is_empty());
    decoder.reset();
}

#[test]
fn show_existing_and_a_superframe_round_trip() {
    let params = CodecParams::with_format(CodecId::Vp9, 16, 16, PixelFormat::Yuv420p);
    let mut encoder = Vp9Encoder::with_quantizer(16, 16, 0).expect("encoder");
    let key = encoder
        .push(&input(solid(16, 16, 40, 90, 140)))
        .expect("key")
        .remove(0);
    let inter = encoder
        .push(&input(moving(16, 16, 1)))
        .expect("inter")
        .remove(0);
    let mut decoder = Vp9Decoder::open(&params).expect("decoder");
    let shown = decoder.push_all(&key).expect("key picture");
    let repeated = decoder
        .push_all(&Packet::video(vec![0x88], 9, 1, 30, false))
        .expect("show existing");
    assert_eq!(repeated.len(), 1);
    assert_eq!(repeated[0].pts, 9);
    assert_eq!(repeated[0].frame.y.samples, shown[0].frame.y.samples);
    assert_eq!(repeated[0].frame.u.samples, shown[0].frame.u.samples);
    assert_eq!(repeated[0].frame.v.samples, shown[0].frame.v.samples);

    let mut fresh = Vp9Decoder::open(&params).expect("fresh");
    let both = fresh
        .push_all(&Packet::video(
            superframe(&[&key.data, &inter.data]),
            0,
            1,
            30,
            true,
        ))
        .expect("superframe");
    assert_eq!(both.len(), 2);
    assert_eq!(both[0].frame.y.samples, shown[0].frame.y.samples);
    let mut separate = Decoder::new();
    separate.decode(&key.data).expect("key").expect("shown");
    let second = separate.decode(&inter.data).expect("inter").expect("shown");
    assert_eq!(both[1].frame.y.samples, second.y);

    let broken = superframe(&[&key.data, &[0x87, 0x00]]);
    let err = fresh.push_all(&Packet::video(broken, 1, 1, 30, true));
    assert!(err.is_err());
    let still = fresh
        .push_all(&Packet::video(vec![0x88], 4, 1, 30, false))
        .expect("refs kept");
    assert_eq!(still[0].frame.y.samples, second.y);

    let Some(ffmpeg) = ffmpeg_bin() else {
        require_or_skip("ffmpeg");
        return;
    };
    let plane = 16 * 16 * 3 / 2;
    let repeated_ivf = wrap_ivf(16, 16, &[key.data.clone(), vec![0x88]]);
    let repeated_yuv = ffmpeg_yuv(&ffmpeg, "vp9", &repeated_ivf);
    assert_eq!(repeated_yuv.len(), plane * 2);
    assert_eq!(&repeated_yuv[..plane], &repeated_yuv[plane..]);
    let packed = superframe(&[&key.data, &inter.data]);
    let wrapped = wrap_ivf(16, 16, &[packed]);
    let native = ffmpeg_yuv(&ffmpeg, "vp9", &wrapped);
    let mut expect = Vec::new();
    expect.extend_from_slice(&shown[0].frame.y.samples);
    expect.extend_from_slice(&shown[0].frame.u.samples);
    expect.extend_from_slice(&shown[0].frame.v.samples);
    expect.extend_from_slice(&second.y);
    expect.extend_from_slice(&second.u);
    expect.extend_from_slice(&second.v);
    assert_eq!(native, expect);
    // ffmpeg's libvpx-vp9 wrapper returns the last picture of one superframe packet.
    let libvpx = ffmpeg_yuv(&ffmpeg, "libvpx-vp9", &wrapped);
    assert_eq!(libvpx, expect[plane..]);
}

fn superframe(frames: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for frame in frames {
        out.extend_from_slice(frame);
    }
    let count = u8::try_from(frames.len() - 1).expect("count");
    let marker = 0xc8 | count;
    out.push(marker);
    for frame in frames {
        let size = u16::try_from(frame.len()).expect("size");
        out.extend_from_slice(&size.to_le_bytes());
    }
    out.push(marker);
    out
}

#[test]
fn factories_open_the_profile_0_codec() {
    let params = CodecParams::with_format(CodecId::Vp9, 16, 16, PixelFormat::Yuv420p);
    let mut encoders = VideoEncodeFactory::new();
    register_encoders(&mut encoders);
    let mut encoder = encoders.open(&params).expect("encoder");
    let mut decoders = VideoDecodeFactory::new();
    register_decoders(&mut decoders);
    let mut decoder = decoders.open(&params).expect("decoder");
    let packets = encoder
        .push(&input(solid(16, 16, 40, 90, 140)))
        .expect("push");
    let shown = decoder.push_all(&packets[0]).expect("decode");
    assert_eq!(shown.len(), 1);
    // The default quantizer leaves a flat picture within a step or two.
    assert!(
        shown[0]
            .frame
            .y
            .samples
            .iter()
            .all(|&v| v.abs_diff(40) <= 2)
    );
    assert_eq!(encoder.codec(), CodecId::Vp9);
    assert_eq!(decoder.codec(), CodecId::Vp9);
}

#[test]
fn independent_decoders_match_our_yuv() {
    let Some(ffmpeg) = ffmpeg_bin() else {
        require_or_skip("ffmpeg");
        return;
    };
    check_oracles(&ffmpeg, 0);
    check_oracles(&ffmpeg, 32);
}

fn check_oracles(ffmpeg: &std::path::Path, qindex: u8) {
    let mut encoder = Vp9Encoder::with_quantizer(32, 16, qindex).expect("encoder");
    let mut decoder = Decoder::new();
    let mut frames = Vec::new();
    let mut expect = Vec::new();
    for index in 0..4 {
        let frame = moving(32, 16, index);
        let packets = encoder.push(&input(frame)).expect("encode");
        let picture = decoder
            .decode(&packets[0].data)
            .expect("ours")
            .expect("shown");
        expect.extend_from_slice(&picture.y);
        expect.extend_from_slice(&picture.u);
        expect.extend_from_slice(&picture.v);
        frames.push(packets[0].data.clone());
    }
    let wrapped = wrap_ivf(32, 16, &frames);
    for name in ["vp9", "libvpx-vp9"] {
        let yuv = ffmpeg_yuv(ffmpeg, name, &wrapped);
        assert_eq!(
            first_mismatch(&yuv, &expect),
            None,
            "{name} q {qindex} mismatch"
        );
    }
    let Some(gst) = gst_bin() else {
        require_or_skip("gstreamer");
        return;
    };
    let yuv = gstreamer_yuv(&gst, &wrapped);
    assert_eq!(first_mismatch(&yuv, &expect), None, "gstreamer q {qindex}");
}

fn oversized_edge() -> Vec<u8> {
    let mut bits = BitWriter::new();
    bits.write(2, 2);
    bits.write(0, 2);
    bits.write_bit(false);
    bits.write_bit(false);
    bits.write_bit(true);
    bits.write_bit(true);
    bits.write(0x49, 8);
    bits.write(0x83, 8);
    bits.write(0x42, 8);
    bits.write(1, 3);
    bits.write_bit(false);
    bits.write(16_832, 16);
    bits.write(7, 16);
    bits.finish()
}

fn oversized_area() -> Vec<u8> {
    let mut bits = BitWriter::new();
    bits.write(2, 2);
    bits.write(0, 2);
    bits.write_bit(false);
    bits.write_bit(false);
    bits.write_bit(true);
    bits.write_bit(true);
    bits.write(0x49, 8);
    bits.write(0x83, 8);
    bits.write(0x42, 8);
    bits.write(1, 3);
    bits.write_bit(false);
    bits.write(16_831, 16);
    bits.write(2_118, 16);
    bits.finish()
}

fn wrap_ivf(width: u32, height: u32, frames: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"DKIF");
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"VP90");
    out.extend_from_slice(&u16::try_from(width).expect("width").to_le_bytes());
    out.extend_from_slice(&u16::try_from(height).expect("height").to_le_bytes());
    out.extend_from_slice(&30u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&u32::try_from(frames.len()).expect("count").to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    for (index, frame) in frames.iter().enumerate() {
        out.extend_from_slice(&u32::try_from(frame.len()).expect("size").to_le_bytes());
        let pts = u64::try_from(index).expect("pts");
        out.extend_from_slice(&pts.to_le_bytes());
        out.extend_from_slice(frame);
    }
    out
}

fn require_or_skip(tool: &str) {
    if std::env::var_os("VP9_REQUIRE_ORACLES").is_some() {
        panic!("{tool} is required");
    }
    eprintln!("skipped {tool} oracle: {tool} was not found");
}

fn ffmpeg_bin() -> Option<std::path::PathBuf> {
    for candidate in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ] {
        let path = std::path::PathBuf::from(candidate);
        if std::process::Command::new(&path)
            .arg("-version")
            .output()
            .is_ok_and(|output| output.status.success())
        {
            return Some(path);
        }
    }
    None
}

fn gst_bin() -> Option<std::path::PathBuf> {
    let path = std::path::PathBuf::from("gst-launch-1.0");
    std::process::Command::new(&path)
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
        .then_some(path)
}

fn unique_dir() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

fn ffmpeg_yuv(ffmpeg: &std::path::Path, decoder: &str, ivf: &[u8]) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!(
        "cutlass-vp9-ff-{}-{}-{}",
        std::process::id(),
        decoder,
        unique_dir()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("dir");
    let input = dir.join("in.ivf");
    let output = dir.join("out.yuv");
    std::fs::write(&input, ivf).expect("ivf");
    let status = std::process::Command::new(ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-c:v",
            decoder,
            "-i",
        ])
        .arg(&input)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&output)
        .output()
        .expect("ffmpeg");
    assert!(
        status.status.success(),
        "{decoder} {}",
        String::from_utf8_lossy(&status.stderr)
    );
    let bytes = std::fs::read(&output).expect("yuv");
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

fn gstreamer_yuv(gst: &std::path::Path, ivf: &[u8]) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!(
        "cutlass-vp9-gst-{}-{}",
        std::process::id(),
        unique_dir()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("dir");
    let input = dir.join("in.ivf");
    let output = dir.join("out.yuv");
    std::fs::write(&input, ivf).expect("ivf");
    let in_loc = format!("location={}", input.display());
    let out_loc = format!("location={}", output.display());
    let status = std::process::Command::new(gst)
        .args([
            "-q",
            "filesrc",
            &in_loc,
            "!",
            "ivfparse",
            "!",
            "vp9dec",
            "!",
            "video/x-raw,format=I420",
            "!",
            "filesink",
            &out_loc,
        ])
        .output()
        .expect("gst");
    assert!(
        status.status.success(),
        "gstreamer {}",
        String::from_utf8_lossy(&status.stderr)
    );
    let bytes = std::fs::read(&output).expect("yuv");
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

#[test]
fn release_320x240_meets_realtime_decode() {
    if cfg!(debug_assertions) {
        eprintln!("320x240 timing is measured in release");
        return;
    }
    let frames = 30u32;
    let mut encoder = Vp9Encoder::with_quantizer(320, 240, 32).expect("encoder");
    let mut packets = Vec::new();
    let encode_started = std::time::Instant::now();
    for index in 0..frames {
        let packet = encoder
            .push(&VideoInput {
                frame: moving(320, 240, index),
                pts: i64::from(index),
                duration: 1,
                timescale: 30,
            })
            .expect("encode");
        packets.push(packet[0].data.clone());
    }
    let encode_elapsed = encode_started.elapsed().as_secs_f64();
    let mut decoder = Decoder::new();
    let decode_started = std::time::Instant::now();
    for packet in &packets {
        decoder.decode(packet).expect("decode").expect("shown");
    }
    let decode_elapsed = decode_started.elapsed().as_secs_f64();
    let decode_fps = f64::from(frames) / decode_elapsed;
    let encode_fps = f64::from(frames) / encode_elapsed;
    let hwm = resident_kib();
    eprintln!(
        "vp9 320x240 q32 frames {frames} decode_fps {decode_fps:.1} encode_fps {encode_fps:.1} hwm_kib {hwm}"
    );
    assert!(
        decode_fps >= 30.0,
        "decode {decode_fps:.1} fps, need 30 for 1x realtime"
    );
}

fn resident_kib() -> u64 {
    let text = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("VmHWM:") else {
            continue;
        };
        let n = rest.split_whitespace().next().unwrap_or("0");
        return n.parse().unwrap_or(0);
    }
    0
}

/// Decoding helpers: a push and then a drain, for the frames the frame
/// threads hold back.
trait PushAll {
    fn push_all(
        &mut self,
        packet: &Packet,
    ) -> Result<Vec<cutlass_decode::video::VideoOutput>, CodecError>;
}

impl<T: VideoDecoder + ?Sized> PushAll for T {
    fn push_all(
        &mut self,
        packet: &Packet,
    ) -> Result<Vec<cutlass_decode::video::VideoOutput>, CodecError> {
        let mut out = self.push(packet)?;
        out.extend(self.drain()?);
        Ok(out)
    }
}

fn parsed_header(bytes: &[u8]) -> Result<header::Header, CodecError> {
    Ok(header::parse(&header::Persist::new(), bytes)?.header)
}

fn write_tile_bits(bits: &mut BitWriter, width: u32) {
    let (min_log2, max_log2) = header::tile_limits(geom::mi_count(width));
    if max_log2 > min_log2 {
        bits.write_bit(false);
    }
    bits.write_bit(false);
}

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
    assert_eq!(picture.y, vis(&frame.y));
    assert_eq!(picture.u, vis(&frame.u));
    assert_eq!(picture.v, vis(&frame.v));
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
    assert_eq!(vis(&again.y), vis(&shown.y));
    assert_eq!(vis(&again.u), vis(&shown.u));
    assert_eq!(vis(&again.v), vis(&shown.v));
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
    assert_eq!(vis(&colored.y), vis(&shown.y));

    assert_color_space(
        &params,
        &packets[0].data,
        &vis(&shown.y),
        3,
        ColorPrimaries::Bt601,
        Transfer::Bt601,
        MatrixCoefficients::Bt601,
    );
    assert_color_space(
        &params,
        &packets[0].data,
        &vis(&shown.y),
        5,
        ColorPrimaries::Bt2020,
        Transfer::Bt2020,
        MatrixCoefficients::Bt2020,
    );
    for space in [0u8, 4, 6] {
        assert_color_space(
            &params,
            &packets[0].data,
            &vis(&shown.y),
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
    assert_eq!(vis(&kept.y), vis(&shown.y));
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
    assert_eq!(vis(&shown.y), samples, "space {space}");
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
        assert_eq!(picture.y, vis(&frame.y));
        assert_eq!(picture.u, vis(&frame.u));
        assert_eq!(picture.v, vis(&frame.v));
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
    let score = psnr(&vis(&frame.y), &picture.y);
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
            first_mismatch(&picture.y, &vis(&frame.y)),
            None,
            "y {index}"
        );
        assert_eq!(
            first_mismatch(&picture.u, &vis(&frame.u)),
            None,
            "u {index}"
        );
        assert_eq!(
            first_mismatch(&picture.v, &vis(&frame.v)),
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
        scores.push(psnr(&vis(&frame.y), &picture.y));
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
fn thirty_frames_stay_near_250_kbps() {
    let mut params = CodecParams::with_format(CodecId::Vp9, 160, 96, PixelFormat::Yuv420p);
    // At more, this simple picture takes every bit it can at the finest
    // quantizer.
    params.bitrate = 250_000;
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
        total += packets.iter().map(|p| p.data.len()).sum::<usize>();
    }
    total += encoder
        .drain()
        .expect("drain")
        .iter()
        .map(|p| p.data.len())
        .sum::<usize>();
    let target = 250_000.0 / 8.0;
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
    assert_eq!(vis(&repeated[0].frame.y), vis(&shown[0].frame.y));
    assert_eq!(vis(&repeated[0].frame.u), vis(&shown[0].frame.u));
    assert_eq!(vis(&repeated[0].frame.v), vis(&shown[0].frame.v));

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
    assert_eq!(vis(&both[0].frame.y), vis(&shown[0].frame.y));
    let mut separate = Decoder::new();
    separate.decode(&key.data).expect("key").expect("shown");
    let second = separate.decode(&inter.data).expect("inter").expect("shown");
    assert_eq!(vis(&both[1].frame.y), second.y);

    let broken = superframe(&[&key.data, &[0x87, 0x00]]);
    let err = fresh.push_all(&Packet::video(broken, 1, 1, 30, true));
    assert!(err.is_err());
    let still = fresh
        .push_all(&Packet::video(vec![0x88], 4, 1, 30, false))
        .expect("refs kept");
    assert_eq!(vis(&still[0].frame.y), second.y);

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
    expect.extend_from_slice(&vis(&shown[0].frame.y));
    expect.extend_from_slice(&vis(&shown[0].frame.u));
    expect.extend_from_slice(&vis(&shown[0].frame.v));
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
    // With the default lookahead the picture comes out of the drain.
    let mut packets = encoder
        .push(&input(solid(16, 16, 40, 90, 140)))
        .expect("push");
    packets.extend(encoder.drain().expect("drain"));
    assert_eq!(packets.len(), 1);
    let shown = decoder.push_all(&packets[0]).expect("decode");
    assert_eq!(shown.len(), 1);
    // The default quantizer leaves a flat picture within a step or two.
    assert!(vis(&shown[0].frame.y).iter().all(|&v| v.abs_diff(40) <= 2));
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

#[test]
fn lookahead_codes_alt_refs_and_every_picture_comes_out() {
    let opts = super::Vp9Options {
        qindex: 120,
        lag: Some(16),
        threads: 2,
        ..super::Vp9Options::default()
    };
    let mut encoder = Vp9Encoder::with_options(64, 64, opts).expect("encoder");
    let mut packets = Vec::new();
    for index in 0..30u32 {
        // Nearly still: an alt-ref pays.
        packets.extend(
            encoder
                .push(&VideoInput {
                    frame: moving(64, 64, index / 8),
                    pts: i64::from(index),
                    duration: 1,
                    timescale: 30,
                })
                .expect("push"),
        );
    }
    assert!(packets.len() < 30, "the lookahead holds pictures back");
    packets.extend(VideoEncoder::drain(&mut encoder).expect("drain"));
    assert_eq!(packets.len(), 30);
    let pts: Vec<i64> = packets.iter().map(|p| p.pts).collect();
    assert_eq!(pts, (0..30).collect::<Vec<i64>>());
    // At least one packet carries a hidden alt-ref in a superframe.
    assert!(
        packets
            .iter()
            .any(|p| p.data.last().is_some_and(|&m| m & 0xe0 == 0xc0))
    );
    let params = CodecParams::with_format(CodecId::Vp9, 64, 64, PixelFormat::Yuv420p);
    let mut decoder = Vp9Decoder::open(&params).expect("decoder");
    let mut shown = 0;
    for p in &packets {
        shown += decoder.push_all(p).expect("decode").len();
    }
    assert_eq!(shown, 30);
}

/// A plane's visible rows: a shared (8-bit) plane is packed out of its
/// padded stride; a copied plane is already tight.
fn vis(p: &Plane) -> Vec<u8> {
    if p.samples.is_shared() {
        p.packed(1)
    } else {
        p.samples.to_vec()
    }
}

#[test]
fn tile_rows_restart_the_coder_and_change_no_picture() {
    // 288 rows: five superblock rows over four tile rows.
    let run = |tile_rows: u8| {
        let opts = super::Vp9Options {
            qindex: 100,
            tile_rows,
            threads: 2,
            ..super::Vp9Options::default()
        };
        let mut encoder = Vp9Encoder::with_options(160, 288, opts).expect("encoder");
        let mut packets = Vec::new();
        for index in 0..6u32 {
            packets.extend(
                encoder
                    .push(&VideoInput {
                        frame: moving(160, 288, index * 3),
                        pts: i64::from(index),
                        duration: 1,
                        timescale: 30,
                    })
                    .expect("push"),
            );
        }
        packets.extend(VideoEncoder::drain(&mut encoder).expect("drain"));
        let params = CodecParams::with_format(CodecId::Vp9, 160, 288, PixelFormat::Yuv420p);
        let mut decoder = Vp9Decoder::open(&params).expect("decoder");
        let mut pictures = Vec::new();
        for p in &packets {
            for out in decoder.push_all(p).expect("decode") {
                let f = &out.frame;
                pictures.push([vis(&f.y), vis(&f.u), vis(&f.v)]);
            }
        }
        (packets, pictures)
    };
    let (one, one_pics) = run(0);
    let (four, four_pics) = run(2);
    assert_eq!(one_pics.len(), 6);
    assert_eq!(one_pics, four_pics, "the same pictures");
    let key = parsed_header(&four[0].data).expect("header");
    assert_eq!(key.log2_tile_rows, 2);
    assert_ne!(
        one.iter().map(|p| p.data.len()).sum::<usize>(),
        four.iter().map(|p| p.data.len()).sum::<usize>(),
        "four coders, four flushes"
    );
}

#[test]
fn a_constant_bitrate_holds_every_half_second_near_the_rate() {
    let window = |cbr_buffer_ms: u32| {
        let opts = super::Vp9Options {
            bitrate: 250_000,
            cbr_buffer_ms,
            threads: 2,
            speed: 9,
            ..super::Vp9Options::default()
        };
        let mut encoder = Vp9Encoder::with_options(160, 96, opts).expect("encoder");
        let mut sizes = Vec::new();
        for index in 0..60u32 {
            let packets = encoder
                .push(&VideoInput {
                    frame: moving(160, 96, index * 2),
                    pts: i64::from(index),
                    duration: 1,
                    timescale: 30,
                })
                .expect("push");
            sizes.extend(packets.iter().map(|p| p.data.len()));
        }
        sizes.extend(
            VideoEncoder::drain(&mut encoder)
                .expect("drain")
                .iter()
                .map(|p| p.data.len()),
        );
        assert_eq!(sizes.len(), 60);
        // The fullest window of 15 frames, over the rate's share.
        let per = 250_000.0 / 8.0 / 2.0;
        sizes
            .windows(15)
            .map(|w| w.iter().sum::<usize>() as f64 / per)
            .fold(0.0, f64::max)
    };
    let (average, constant) = (window(0), window(1000));
    assert!(constant < 1.35, "a window at {constant:.2} of the rate");
    assert!(constant <= average, "{constant:.2} against {average:.2}");
    assert_eq!(Vp9Encoder::qindex_of_level(62), 249);
    assert_eq!(Vp9Encoder::qindex_of_level(63), 255);
    assert_eq!(Vp9Encoder::qindex_of_level(10), 40);
}

#[test]
fn a_first_pass_codes_near_the_quantizer_the_bitrate_takes() {
    let at = |bitrate: u32| {
        let opts = super::Vp9Options {
            bitrate,
            qindex: 77,
            ..super::Vp9Options::default()
        };
        Vp9Encoder::with_options(1280, 720, opts)
            .expect("encoder")
            .first_pass_qindex(1.0 / 30.0)
    };
    // No bitrate: the encoder's own quantizer.
    assert_eq!(at(0), 77);
    // Fewer bits, a coarser quantizer, within the first pass's range.
    let (low, mid, high) = (at(500_000), at(2_000_000), at(8_000_000));
    assert!(low > mid && mid > high, "{low} {mid} {high}");
    assert!((40..=200).contains(&low) && (40..=200).contains(&high));
    assert!((100..=160).contains(&mid), "{mid}");
}

#[test]
fn a_second_pass_spends_the_bitrate_where_the_first_found_detail() {
    // Twenty still pictures, then twenty moving: the second pass gives the
    // moving ones the bits.
    let frame = |index: u32| moving(160, 96, if index < 20 { 0 } else { index * 3 });
    let run = |opts: super::Vp9Options, first: Option<(&[u64], u8)>| {
        let mut encoder = Vp9Encoder::with_options(160, 96, opts).expect("encoder");
        if let Some((bits, q)) = first {
            encoder.set_second_pass(bits, q);
        }
        let mut sizes = Vec::new();
        for index in 0..40u32 {
            let packets = encoder
                .push(&VideoInput {
                    frame: frame(index),
                    pts: i64::from(index),
                    duration: 1,
                    timescale: 30,
                })
                .expect("push");
            sizes.extend(packets.iter().map(|p| p.data.len() as u64 * 8));
        }
        sizes.extend(
            VideoEncoder::drain(&mut encoder)
                .expect("drain")
                .iter()
                .map(|p| p.data.len() as u64 * 8),
        );
        sizes
    };
    let base = super::Vp9Options {
        threads: 2,
        speed: 8,
        ..super::Vp9Options::default()
    };
    let first = run(
        super::Vp9Options {
            qindex: 120,
            speed: 10,
            ..base
        },
        None,
    );
    assert_eq!(first.len(), 40);
    let rate = 150_000u32;
    let second = run(
        super::Vp9Options {
            bitrate: rate,
            ..base
        },
        Some((&first, 120)),
    );
    let total: u64 = second.iter().sum();
    let target = f64::from(rate) * 40.0 / 30.0;
    let ratio = total as f64 / target;
    assert!((0.85..=1.15).contains(&ratio), "{ratio:.2} of the bitrate");
    // The moving half takes most of it.
    let moving: u64 = second[20..].iter().sum();
    assert!(moving * 2 > total, "{moving} of {total}");
}

/// Frame `index` of a pan by 3 luma samples a frame over a picture with
/// fine detail in its colour: each chroma sample is the picture's at its
/// place, so a pan by an odd number of luma samples shows chroma that no
/// sub-sample filter predicts (sampled afresh, as a camera's 4:2:0 is).
fn colour_pan(width: u32, height: u32, index: u32) -> VideoFrame {
    let at = |x: u32, y: u32, k: u32| -> u8 {
        let mut v = (x + index * 3).wrapping_mul(0x9e37_79b1) ^ y.wrapping_mul(0x85eb_ca77) ^ k;
        v ^= v >> 15;
        v = v.wrapping_mul(0x2c1b_3c6d);
        v ^= v >> 12;
        u8::try_from(48 + (v & 0x7f)).unwrap_or(48)
    };
    let (cw, ch) = (width / 2, height / 2);
    let y: Vec<u8> = (0..height)
        .flat_map(|r| (0..width).map(move |c| (r, c)))
        .map(|(r, c)| at(c, r, 0))
        .collect();
    let plane = |k: u32| -> Vec<u8> {
        (0..ch)
            .flat_map(|r| (0..cw).map(move |c| (r, c)))
            .map(|(r, c)| at(c * 2, r * 2, k))
            .collect()
    };
    VideoFrame::yuv420p(
        width,
        height,
        Plane::tight(width, height, y).expect("y"),
        Plane::tight(cw, ch, plane(1)).expect("u"),
        Plane::tight(cw, ch, plane(2)).expect("v"),
    )
    .expect("frame")
}

#[test]
fn a_second_pass_predicts_chroma_from_a_picture_moved_by_whole_chroma_samples() {
    // Two passes keep the last pictures in spare slots, and GOLDEN reads
    // the one the content moved from by an even number of luma samples:
    // the stream decodes, and some frames' GOLDEN is such a slot.
    let (w, h, n) = (192u32, 128u32, 24u32);
    let run = |opts: super::Vp9Options, first: Option<(&[u64], u8)>| {
        let mut encoder = Vp9Encoder::with_options(w, h, opts).expect("encoder");
        if let Some((bits, q)) = first {
            encoder.set_second_pass(bits, q);
        }
        let mut packets = Vec::new();
        for index in 0..n {
            packets.extend(
                encoder
                    .push(&VideoInput {
                        frame: colour_pan(w, h, index),
                        pts: i64::from(index),
                        duration: 1,
                        timescale: 30,
                    })
                    .expect("push"),
            );
        }
        packets.extend(VideoEncoder::drain(&mut encoder).expect("drain"));
        packets
    };
    let base = super::Vp9Options {
        threads: 2,
        speed: 6,
        ..super::Vp9Options::default()
    };
    let first: Vec<u64> = run(
        super::Vp9Options {
            qindex: 120,
            speed: 10,
            ..base
        },
        None,
    )
    .iter()
    .map(|p| p.data.len() as u64 * 8)
    .collect();
    let second = run(
        super::Vp9Options {
            bitrate: 400_000,
            ..base
        },
        Some((&first, 120)),
    );
    assert_eq!(second.len(), n as usize);
    let mut decoder = Decoder::new();
    let mut pooled = 0;
    for (index, packet) in second.iter().enumerate() {
        let mut picture = None;
        for frame in crate::decode::split_superframe(&packet.data).expect("frames") {
            // An inter frame's uncompressed header: marker, profile 0,
            // show_existing, frame_type, show_frame, error_resilient,
            // intra_only (hidden frames), reset_frame_context, the refresh
            // flags, then LAST's slot and sign bias and GOLDEN's slot.
            let bit = |i: usize| (frame[i / 8] >> (7 - i % 8)) & 1;
            if bit(5) == 1 {
                let at = 8 + usize::from(bit(6) == 0) + 2 + 8 + 4;
                let golden = (bit(at) << 2) | (bit(at + 1) << 1) | bit(at + 2);
                pooled += usize::from(golden >= 3);
            }
            picture = decoder.decode(frame).expect("decode").or(picture);
        }
        let picture = picture.expect("shown");
        let src = colour_pan(w, h, u32::try_from(index).unwrap_or(0));
        let (y, u) = (
            psnr(&vis(&src.y), &picture.y),
            psnr(&vis(&src.u), &picture.u),
        );
        assert!(y > 25.0 && u > 15.0, "frame {index}: {y:.1} / {u:.1} dB");
    }
    assert!(
        pooled * 3 > n as usize,
        "{pooled} frames read GOLDEN from the pool"
    );
}

#[test]
fn a_cut_starts_a_golden_frame_and_keeps_the_stream_going() {
    // Ten frames of a moving ramp, then ten of a smooth gradient: nothing
    // in the second predicts from the first better than its own mean.
    let gradient = |shift: u32| {
        let mut f = moving(128, 128, 0);
        for (i, v) in f.y.samples.make_mut().iter_mut().enumerate() {
            let (r, c) = (i / 128, i % 128);
            *v = (40 + (r + c + shift as usize) / 4).min(255) as u8;
        }
        f
    };
    let opts = super::Vp9Options {
        qindex: 100,
        lag: Some(0),
        threads: 1,
        speed: 6,
        ..super::Vp9Options::default()
    };
    let mut encoder = Vp9Encoder::with_options(128, 128, opts).expect("encoder");
    let mut packets = Vec::new();
    for index in 0..20u32 {
        let frame = if index < 10 {
            moving(128, 128, index)
        } else {
            gradient(index)
        };
        packets.extend(
            encoder
                .push(&VideoInput {
                    frame,
                    pts: i64::from(index),
                    duration: 1,
                    timescale: 30,
                })
                .expect("push"),
        );
    }
    assert_eq!(packets.len(), 20);
    // Each header read against the state of a decoder that has decoded
    // the frames before it.
    let mut dec = Decoder::new();
    let headers: Vec<header::Header> = packets
        .iter()
        .map(|p| {
            let h = header::parse(dec.persist(), &p.data)
                .expect("header")
                .header;
            dec.decode(&p.data).expect("decode");
            h
        })
        .collect();
    // The cut is not a key frame, but it refreshes LAST and GOLDEN (and,
    // as every inter frame at this speed, one slot of the reference pool:
    // exactly one of slots 3 to 7).
    assert!(!headers[10].key_frame);
    let pool_slot = |h: &header::Header| {
        let pool = h.refresh_flags & 0xf8;
        assert_eq!(pool.count_ones(), 1, "one pool slot");
        pool
    };
    assert_eq!(headers[10].refresh_flags & 0x07, 0x03);
    pool_slot(&headers[10]);
    // The frames around it refresh only LAST, of the first three.
    assert_eq!(headers[9].refresh_flags & 0x07, 0x01);
    assert_eq!(headers[11].refresh_flags & 0x07, 0x01);
    // Each in a pool slot of its own.
    assert_ne!(pool_slot(&headers[9]), pool_slot(&headers[11]));
    // And the stream decodes through it.
    let params = CodecParams::with_format(CodecId::Vp9, 128, 128, PixelFormat::Yuv420p);
    let mut decoder = Vp9Decoder::open(&params).expect("decoder");
    let mut shown = 0;
    for p in &packets {
        shown += decoder.push_all(p).expect("decode").len();
    }
    assert_eq!(shown, 20);
}

#[test]
fn profile_1_chroma_codes_and_lossless_comes_back_exact() {
    for (chroma, (sx, sy)) in [
        (super::Chroma::Yuv444, (0u32, 0u32)),
        (super::Chroma::Yuv422, (1, 0)),
        (super::Chroma::Yuv440, (0, 1)),
    ] {
        let (w, h) = (96u32, 64u32);
        let dims = [(w, h), (w >> sx, h >> sy), (w >> sx, h >> sy)];
        let planes: Vec<Vec<u8>> = (0..3u32)
            .map(|p| {
                let (pw, ph) = dims[p as usize];
                (0..pw * ph)
                    .map(|i| {
                        let (r, c) = (i / pw, i % pw);
                        ((r * 3 + c * (p + 2) + p * 40) % 230 + 10) as u8
                    })
                    .collect()
            })
            .collect();
        let opts = super::Vp9Options {
            qindex: 0,
            chroma,
            lag: Some(0),
            threads: 1,
            ..super::Vp9Options::default()
        };
        let mut encoder = Vp9Encoder::with_options(w, h, opts).expect("encoder");
        let s = |p: usize| dims[p].0 as usize;
        let (data, _) = encoder
            .encode_planes(
                [(&planes[0], s(0)), (&planes[1], s(1)), (&planes[2], s(2))],
                1.0 / 30.0,
            )
            .expect("encode");
        let key = parsed_header(&data).expect("header");
        assert_eq!(key.profile, 1, "{chroma:?}");
        assert_eq!((key.ss_x, key.ss_y), (sx, sy));
        assert!(key.lossless);
        let mut decoder = Decoder::new();
        let picture = decoder.decode(&data).expect("decode").expect("shown");
        for (p, (got, want)) in [&picture.y, &picture.u, &picture.v]
            .into_iter()
            .zip(&planes)
            .enumerate()
        {
            assert_eq!(got, want, "{chroma:?} plane {p}");
        }
    }
}

/// Planes of `bd`-bit samples, `w`×`h` with chroma `ss` subsampled, a
/// gradient and a moving texture over the whole range (picture `t`).
fn deep_planes(w: u32, h: u32, (sx, sy): (u32, u32), bd: u8, t: u32) -> [Vec<u16>; 3] {
    let max = (1u32 << bd) - 1;
    let dims = [
        (w, h),
        (w.div_ceil(1 << sx), h.div_ceil(1 << sy)),
        (w.div_ceil(1 << sx), h.div_ceil(1 << sy)),
    ];
    std::array::from_fn(|p| {
        let (pw, ph) = dims[p];
        let mut x = 0x1234_5678u32 ^ (p as u32 * 977) ^ (t * 7919);
        (0..pw * ph)
            .map(|i| {
                let (r, c) = (i / pw, i % pw + t * 2);
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let wave = (r * 5 + c * (p as u32 + 3) + p as u32 * 300) % 180;
                (wave * max / 180 + (x >> 28)).min(max) as u16
            })
            .collect()
    })
}

/// The picture a packet shows, through every frame of a superframe.
fn shown_of(decoder: &mut Decoder, data: &[u8]) -> Option<crate::decode::Picture> {
    let mut shown = None;
    for f in crate::decode::split_superframe(data).expect("superframe") {
        if let Some(p) = decoder.decode(f).expect("decode") {
            shown = Some(p);
        }
    }
    shown
}

/// A decoded picture's little-endian 16-bit plane as samples.
fn le16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| u16::from_le_bytes(b))
        .collect()
}

#[test]
fn profiles_2_and_3_lossless_come_back_exact() {
    for (bd, chroma, (sx, sy), profile) in [
        (10u8, super::Chroma::Yuv420, (1u32, 1u32), 2u8),
        (12, super::Chroma::Yuv420, (1, 1), 2),
        (10, super::Chroma::Yuv444, (0, 0), 3),
        (12, super::Chroma::Yuv422, (1, 0), 3),
        (12, super::Chroma::Yuv440, (0, 1), 3),
    ] {
        // An odd size: partial superblocks and chroma rounded up.
        let (w, h) = (75u32, 45u32);
        let opts = super::Vp9Options {
            qindex: 0,
            chroma,
            bit_depth: bd,
            lag: Some(0),
            threads: 1,
            speed: 4,
            ..super::Vp9Options::default()
        };
        let mut encoder = Vp9Encoder::with_options(w, h, opts).expect("encoder");
        let mut decoder = Decoder::new();
        for t in 0..3 {
            let planes = deep_planes(w, h, (sx, sy), bd, t);
            let cw = w.div_ceil(1 << sx) as usize;
            let (data, key) = encoder
                .encode_planes16(
                    [(&planes[0], w as usize), (&planes[1], cw), (&planes[2], cw)],
                    1.0 / 30.0,
                )
                .expect("encode");
            assert_eq!(key, t == 0);
            if t == 0 {
                let head = parsed_header(&data).expect("header");
                assert_eq!(head.profile, profile, "{bd} {chroma:?}");
                assert_eq!(head.bit_depth, bd);
                assert_eq!((head.ss_x, head.ss_y), (sx, sy));
                assert!(head.lossless);
            }
            let picture = decoder.decode(&data).expect("decode").expect("shown");
            assert_eq!(picture.bit_depth, bd);
            for (p, (got, want)) in [&picture.y, &picture.u, &picture.v]
                .into_iter()
                .zip(&planes)
                .enumerate()
            {
                assert_eq!(&le16(got), want, "{bd} {chroma:?} picture {t} plane {p}");
            }
        }
    }
}

#[test]
fn high_bit_depth_lossy_decodes_as_the_encoder_reconstructs() {
    for (bd, speed, lag) in [(10u8, 2u8, 0u32), (12, 6, 0), (10, 8, 16), (12, 10, 0)] {
        let (w, h) = (96u32, 72u32);
        let opts = super::Vp9Options {
            qindex: 80,
            bit_depth: bd,
            lag: Some(lag),
            threads: 2,
            speed,
            ..super::Vp9Options::default()
        };
        let mut encoder = Vp9Encoder::with_options(w, h, opts).expect("encoder");
        let mut decoder = Decoder::new();
        let pictures: Vec<[Vec<u16>; 3]> =
            (0..20).map(|t| deep_planes(w, h, (1, 1), bd, t)).collect();
        let mut shown = Vec::new();
        let cw = w.div_ceil(2) as usize;
        for (t, planes) in pictures.iter().enumerate() {
            let coded = encoder
                .push_planes16(
                    [(&planes[0], w as usize), (&planes[1], cw), (&planes[2], cw)],
                    1.0 / 30.0,
                    (t as i64, 1, 30),
                )
                .expect("encode");
            for (data, _, _) in coded {
                let picture = shown_of(&mut decoder, &data).expect("shown");
                if lag == 0 {
                    // Each picture decodes as the encoder reconstructed it.
                    let super::Lane::Deep(core) = &encoder.lane else {
                        panic!("a 10- or 12-bit encoder");
                    };
                    let last = core.enc.last().expect("reconstruction");
                    last.buf.wait(usize::MAX);
                    let (rec, stride) = last.buf.plane(0);
                    let y = le16(&picture.y);
                    let w = w as usize;
                    for r in 0..h as usize {
                        assert_eq!(
                            &y[r * w..(r + 1) * w],
                            &rec[r * stride..r * stride + w],
                            "{bd}-bit speed {speed} row {r}"
                        );
                    }
                }
                shown.push(picture);
            }
        }
        for (data, _, _) in encoder.flush().expect("flush") {
            shown.push(shown_of(&mut decoder, &data).expect("shown"));
        }
        assert_eq!(shown.len(), pictures.len(), "{bd}-bit speed {speed}");
        // Close to the source: each picture's luma PSNR against the bit
        // depth's peak.
        let peak = f64::from((1u32 << bd) - 1);
        for (t, (pic, src)) in shown.iter().zip(&pictures).enumerate() {
            assert_eq!(pic.bit_depth, bd);
            let y = le16(&pic.y);
            let sse: f64 = y
                .iter()
                .zip(&src[0])
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum();
            let psnr = 10.0 * (peak * peak / (sse / y.len() as f64).max(1e-9)).log10();
            assert!(
                psnr > 30.0,
                "{bd}-bit speed {speed} picture {t}: {psnr:.1} dB"
            );
        }
    }
}

#[test]
fn ten_bit_frames_on_the_codec_contract() {
    let (w, h) = (48u32, 32u32);
    let mut params = CodecParams::video(CodecId::Vp9, w, h);
    params.pixel = PixelFormat::Yuv420p10;
    let mut encoder = Vp9Encoder::open(&params).expect("encoder");
    let planes = deep_planes(w, h, (1, 1), 10, 0);
    let bytes = |v: &[u16]| v.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>();
    let frame = VideoFrame::planar(
        PixelFormat::Yuv420p10,
        w,
        h,
        Plane::bytes(w, h, 2, bytes(&planes[0])).expect("y"),
        Plane::bytes(w / 2, h / 2, 2, bytes(&planes[1])).expect("u"),
        Plane::bytes(w / 2, h / 2, 2, bytes(&planes[2])).expect("v"),
    )
    .expect("frame");
    let mut packets = encoder.push(&input(frame)).expect("encode");
    packets.extend(encoder.drain().expect("drain"));
    assert_eq!(packets.len(), 1);
    let head = parsed_header(&packets[0].data).expect("header");
    assert_eq!((head.profile, head.bit_depth), (2, 10));
    let picture = Decoder::new()
        .decode(&packets[0].data)
        .expect("decode")
        .expect("shown");
    assert_eq!(picture.bit_depth, 10);
    // An 8-bit frame, or 8-bit planes, are refused.
    assert!(encoder.push(&input(solid(w, h, 1, 2, 3))).is_err());
    let y = vec![0u8; (w * h) as usize];
    let c = vec![0u8; (w * h / 4) as usize];
    assert!(
        encoder
            .encode_planes([(&y, 48), (&c, 24), (&c, 24)], 1.0 / 30.0)
            .is_err()
    );
    // So is a sample past 10 bits.
    let mut over = planes.clone();
    over[0][5] = 1024;
    assert!(
        encoder
            .encode_planes16([(&over[0], 48), (&over[1], 24), (&over[2], 24)], 1.0 / 30.0)
            .is_err()
    );
}

#[test]
fn lossless_with_alt_refs_comes_back_exact() {
    // An inter block after an alt-ref once dropped its residual where the
    // prediction was close, which a lossless frame cannot.
    for bd in [8u8, 10] {
        let (w, h) = (64u32, 48u32);
        let opts = super::Vp9Options {
            qindex: 0,
            bit_depth: bd,
            lag: Some(16),
            threads: 1,
            speed: 6,
            ..super::Vp9Options::default()
        };
        let mut encoder = Vp9Encoder::with_options(w, h, opts).expect("encoder");
        // Nearly still, with a little noise: the alt-ref predicts it
        // closely.
        let pictures: Vec<[Vec<u16>; 3]> = (0..20u32)
            .map(|t| {
                let mut p = deep_planes(w, h, (1, 1), bd, 0);
                let mut x = t.wrapping_mul(2_654_435_761);
                // Still but for a little noise in the top rows: most blocks
                // skip, which makes skipping cheap for the rest.
                for v in p[0][..w as usize * 8].iter_mut() {
                    x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *v = (*v).saturating_sub((x >> 31) as u16);
                }
                p
            })
            .collect();
        let mut packets = Vec::new();
        let cw = (w / 2) as usize;
        for (t, p) in pictures.iter().enumerate() {
            let clock = (t as i64, 1, 30);
            let coded = if bd == 8 {
                let b: Vec<Vec<u8>> = p
                    .iter()
                    .map(|v| v.iter().map(|&s| s as u8).collect())
                    .collect();
                encoder.push_planes(
                    [(&b[0], w as usize), (&b[1], cw), (&b[2], cw)],
                    1.0 / 30.0,
                    clock,
                )
            } else {
                encoder.push_planes16(
                    [(&p[0], w as usize), (&p[1], cw), (&p[2], cw)],
                    1.0 / 30.0,
                    clock,
                )
            };
            packets.extend(coded.expect("encode"));
        }
        packets.extend(encoder.flush().expect("flush"));
        assert!(
            packets
                .iter()
                .any(|(d, _, _)| d.last().is_some_and(|&m| m & 0xe0 == 0xc0)),
            "{bd}-bit: an alt-ref"
        );
        let mut decoder = Decoder::new();
        for (t, ((data, _, _), src)) in packets.iter().zip(&pictures).enumerate() {
            let pic = shown_of(&mut decoder, data).expect("shown");
            for (p, (got, want)) in [&pic.y, &pic.u, &pic.v].into_iter().zip(src).enumerate() {
                let got: Vec<u16> = if bd == 8 {
                    got.iter().map(|&s| u16::from(s)).collect()
                } else {
                    le16(got)
                };
                assert_eq!(&got, want, "{bd}-bit picture {t} plane {p}");
            }
        }
    }
}

#[test]
fn samples_past_the_bit_depth_are_refused() {
    for bd in [10u8, 12] {
        let (w, h) = (64u32, 48u32);
        let opts = super::Vp9Options {
            bit_depth: bd,
            threads: 1,
            ..super::Vp9Options::default()
        };
        let mut encoder = Vp9Encoder::with_options(w, h, opts).expect("encoder");
        let max = (1u16 << bd) - 1;
        let cw = (w / 2) as usize;
        let mut y = vec![max; (w * h) as usize];
        let c = vec![max / 2; cw * (h / 2) as usize];
        assert!(
            encoder
                .push_planes16(
                    [(&y, w as usize), (&c, cw), (&c, cw)],
                    1.0 / 30.0,
                    (0, 1, 30)
                )
                .is_ok(),
            "{bd}-bit: the largest sample"
        );
        y[(w * h) as usize - 1] = max + 1;
        assert!(
            encoder
                .push_planes16(
                    [(&y, w as usize), (&c, cw), (&c, cw)],
                    1.0 / 30.0,
                    (1, 1, 30)
                )
                .is_err(),
            "{bd}-bit: one past it"
        );
    }
}

/// Encodes `n` pictures of `frame(index)`; returns the packets.
fn encode_with(
    opts: super::Vp9Options,
    (w, h): (u32, u32),
    n: u32,
    frame: impl Fn(u32) -> VideoFrame,
) -> Vec<Packet> {
    let mut encoder = Vp9Encoder::with_options(w, h, opts).expect("encoder");
    let mut packets = Vec::new();
    for index in 0..n {
        packets.extend(
            encoder
                .push(&VideoInput {
                    frame: frame(index),
                    pts: i64::from(index),
                    duration: 1,
                    timescale: 30,
                })
                .expect("push"),
        );
    }
    packets.extend(VideoEncoder::drain(&mut encoder).expect("drain"));
    packets
}

/// The headers of single-frame packets, each read against the state of a
/// decoder that has decoded the frames before it.
fn headers_of(packets: &[Packet]) -> Vec<header::Header> {
    let mut dec = Decoder::new();
    packets
        .iter()
        .map(|p| {
            let h = header::parse(dec.persist(), &p.data)
                .expect("header")
                .header;
            dec.decode(&p.data).expect("decode");
            h
        })
        .collect()
}

/// The packets decode, on the codec contract, to `n` pictures.
fn assert_decodes(packets: &[Packet], (w, h): (u32, u32), n: usize) {
    let params = CodecParams::with_format(CodecId::Vp9, w, h, PixelFormat::Yuv420p);
    let mut decoder = Vp9Decoder::open(&params).expect("decoder");
    let mut shown = 0;
    for p in packets {
        shown += decoder.push_all(p).expect("decode").len();
    }
    assert_eq!(shown, n);
}

fn packet_bytes(packets: &[Packet]) -> usize {
    packets.iter().map(|p| p.data.len()).sum()
}

fn lag0(qindex: u8) -> super::Vp9Options {
    super::Vp9Options {
        qindex,
        lag: Some(0),
        threads: 2,
        ..super::Vp9Options::default()
    }
}

#[test]
fn stream_signals_take_their_options() {
    use super::ColorSpace;
    let opts = super::Vp9Options {
        sharpness: 5,
        color_space: ColorSpace::Bt709,
        full_range: true,
        render_size: Some((96, 40)),
        frame_parallel: true,
        ..lag0(100)
    };
    let packets = encode_with(opts, (64, 48), 4, |i| moving(64, 48, i));
    for (i, h) in headers_of(&packets).iter().enumerate() {
        assert_eq!(h.sharpness, 5, "frame {i}");
        assert_eq!((h.render_width, h.render_height), (96, 40), "frame {i}");
        assert!(h.frame_parallel && !h.error_resilient, "frame {i}");
        assert_eq!((h.color_space, h.color_range), (2, 1), "frame {i}");
    }
    assert_decodes(&packets, (64, 48), 4);
    // The defaults: BT.601 studio range, the picture's own size, sharpness
    // 0, backward adaptation.
    let h = &headers_of(&encode_with(lag0(100), (64, 48), 2, |i| moving(64, 48, i)))[1];
    assert_eq!((h.color_space, h.color_range, h.sharpness), (1, 0, 0));
    assert_eq!((h.render_width, h.render_height), (64, 48));
    assert!(!h.frame_parallel && h.refresh_context);
    // The configuration names the level and the range.
    let encoder = Vp9Encoder::with_options(
        64,
        48,
        super::Vp9Options {
            level: 31,
            full_range: true,
            ..lag0(100)
        },
    )
    .expect("encoder");
    let vpcc = encoder.vpcc_with_colour([1, 1, 1]);
    assert_eq!(vpcc[5], 31);
    assert_eq!(vpcc[6] & 1, 1);
}

#[test]
fn error_resilient_frames_stand_without_the_one_before() {
    let opts = super::Vp9Options {
        error_resilient: true,
        ..lag0(100)
    };
    let packets = encode_with(opts, (64, 48), 6, |i| moving(64, 48, i));
    for (i, h) in headers_of(&packets).iter().enumerate() {
        assert!(h.error_resilient && h.frame_parallel, "frame {i}");
        assert!(!h.use_prev_mvs, "frame {i}");
    }
    assert_decodes(&packets, (64, 48), 6);
    // With alt-refs (and no compound prediction, which error resilient
    // frames leave out) and at a bitrate.
    let arf = super::Vp9Options {
        error_resilient: true,
        bitrate: 300_000,
        lag: Some(16),
        ..lag0(100)
    };
    let with_arf = encode_with(arf, (64, 48), 30, |i| noisy(64, 48, i / 8, i));
    assert_decodes(&with_arf, (64, 48), 30);
    // With frame 2 lost, the frames after it still decode (from the
    // pictures the decoder has).
    let mut dec = Decoder::new();
    for (i, p) in packets.iter().enumerate() {
        if i != 2 {
            assert!(dec.decode(&p.data).expect("decode").is_some(), "frame {i}");
        }
    }
}

#[test]
fn srgb_takes_4_4_4_only() {
    use super::{Chroma, ColorSpace};
    let rgb = super::Vp9Options {
        color_space: ColorSpace::Srgb,
        ..lag0(0)
    };
    assert!(Vp9Encoder::with_options(16, 16, rgb).is_err());
    let mut encoder = Vp9Encoder::with_options(
        16,
        16,
        super::Vp9Options {
            chroma: Chroma::Yuv444,
            ..rgb
        },
    )
    .expect("encoder");
    let (g, b, r) = (vec![30u8; 256], vec![200u8; 256], vec![90u8; 256]);
    let (data, key) = encoder
        .encode_planes([(&g, 16), (&b, 16), (&r, 16)], 1.0 / 30.0)
        .expect("encode");
    assert!(key);
    let h = parsed_header(&data).expect("header");
    assert_eq!((h.profile, h.color_space, h.ss_x, h.ss_y), (1, 7, 0, 0));
    // Lossless: the planes come back as they went in.
    let pic = Decoder::new()
        .decode(&data)
        .expect("decode")
        .expect("shown");
    assert_eq!((pic.y, pic.u, pic.v), (g, b, r));
}

#[test]
fn options_outside_their_range_are_refused() {
    for opts in [
        super::Vp9Options {
            sharpness: 8,
            ..lag0(60)
        },
        super::Vp9Options {
            arnr_strength: Some(7),
            ..lag0(60)
        },
        super::Vp9Options {
            arnr_max_frames: Some(0),
            ..lag0(60)
        },
        super::Vp9Options {
            tile_columns: Some(7),
            ..lag0(60)
        },
        super::Vp9Options {
            render_size: Some((0, 10)),
            ..lag0(60)
        },
    ] {
        assert!(Vp9Encoder::with_options(64, 64, opts).is_err(), "{opts:?}");
    }
}

/// `moving(w, h, shift)` under faint noise of its own in each picture
/// `i`.
fn noisy(w: u32, h: u32, shift: u32, i: u32) -> VideoFrame {
    let mut f = moving(w, h, shift);
    for (k, v) in f.y.samples.make_mut().iter_mut().enumerate() {
        let n = ((k as u32) ^ i.wrapping_mul(0x9e37_79b9)).wrapping_mul(2_654_435_761) >> 30;
        *v = v.saturating_add(n as u8);
    }
    f
}

#[test]
fn a_static_threshold_skips_the_residual_of_near_blocks() {
    // A still picture under faint noise: each inter frame's residual is
    // the noise.
    let noisy = |i: u32| noisy(128, 64, 0, i);
    let size = |t: u32| {
        let opts = super::Vp9Options {
            static_thresh: t,
            speed: 6,
            ..lag0(20)
        };
        let packets = encode_with(opts, (128, 64), 6, noisy);
        assert_decodes(&packets, (128, 64), 6);
        packet_bytes(&packets[1..])
    };
    let (off, on) = (size(0), size(2000));
    assert!(on * 2 < off, "inter frames: {on} bytes with, {off} without");
}

#[test]
fn alt_ref_options_shape_the_alt_ref() {
    let base = super::Vp9Options {
        qindex: 120,
        lag: Some(16),
        threads: 2,
        ..super::Vp9Options::default()
    };
    let still = |i: u32| noisy(64, 64, i / 8, i);
    let has_arf = |packets: &[Packet]| {
        packets
            .iter()
            .any(|p| p.data.last().is_some_and(|&m| m & 0xe0 == 0xc0))
    };
    let plain = encode_with(base, (64, 64), 30, still);
    assert!(has_arf(&plain));
    let none = encode_with(
        super::Vp9Options {
            alt_ref: false,
            ..base
        },
        (64, 64),
        30,
        still,
    );
    assert!(!has_arf(&none));
    assert_decodes(&none, (64, 64), 30);
    // Another filter makes another alt-ref.
    for opts in [
        super::Vp9Options {
            arnr_strength: Some(0),
            ..base
        },
        super::Vp9Options {
            arnr_max_frames: Some(1),
            ..base
        },
        super::Vp9Options {
            arnr_strength: Some(6),
            arnr_max_frames: Some(15),
            ..base
        },
    ] {
        let packets = encode_with(opts, (64, 64), 30, still);
        assert!(has_arf(&packets));
        assert!(
            packets.iter().zip(&plain).any(|(a, b)| a.data != b.data),
            "{opts:?}"
        );
        assert_decodes(&packets, (64, 64), 30);
    }
}

#[test]
fn constrained_quality_holds_the_quantizer_at_its_level() {
    let opts = super::Vp9Options {
        bitrate: 4_000_000,
        speed: 6,
        ..lag0(60)
    };
    let cq = super::Vp9Options {
        cq_level: Some(140),
        ..opts
    };
    let free = encode_with(opts, (128, 64), 10, |i| moving(128, 64, i));
    let held = encode_with(cq, (128, 64), 10, |i| moving(128, 64, i));
    for (i, h) in headers_of(&held).iter().enumerate().skip(1) {
        assert!(h.base_qindex >= 140, "frame {i}: {}", h.base_qindex);
    }
    assert!(packet_bytes(&held) < packet_bytes(&free));
    assert_decodes(&held, (128, 64), 10);
}

#[test]
fn a_limit_on_key_frames_makes_them_smaller() {
    let opts = super::Vp9Options {
        bitrate: 200_000,
        speed: 6,
        ..lag0(60)
    };
    let capped = super::Vp9Options {
        max_intra_pct: 100,
        ..opts
    };
    let key = |o| {
        encode_with(o, (128, 64), 3, |i| moving(128, 64, i))[0]
            .data
            .len()
    };
    assert!(key(capped) < key(opts));
}

#[test]
fn dead_bands_leave_what_the_rate_overshot_or_undershot() {
    // The key frame takes far more than its share; the frames after pay
    // it back, but those after an overshoot band pay back less, and those
    // after an undershoot band spend less of a saving.
    let opts = super::Vp9Options {
        bitrate: 300_000,
        speed: 6,
        ..lag0(60)
    };
    let bytes = |o| {
        let p = encode_with(o, (128, 64), 20, |i| noisy(128, 64, i, i));
        assert_decodes(&p, (128, 64), 20);
        packet_bytes(&p[1..])
    };
    let plain = bytes(opts);
    let over = bytes(super::Vp9Options {
        overshoot_pct: 100,
        ..opts
    });
    assert!(over > plain, "{over} with the band, {plain} without");
    // A rate far above what still pictures cost leaves a saving.
    let rich = super::Vp9Options {
        bitrate: 3_000_000,
        ..opts
    };
    let still = |o| {
        let p = encode_with(o, (128, 64), 20, |_| moving(128, 64, 3));
        packet_bytes(&p[1..])
    };
    let under = still(super::Vp9Options {
        undershoot_pct: 100,
        ..rich
    });
    assert!(under <= still(rich));
}

#[test]
fn a_cut_past_keyint_min_is_a_key_frame() {
    let gradient = |i: u32| {
        if i < 10 {
            return moving(128, 128, i);
        }
        let mut f = moving(128, 128, 0);
        for (k, v) in f.y.samples.make_mut().iter_mut().enumerate() {
            let (r, c) = (k / 128, k % 128);
            *v = (40 + (r + c + i as usize) / 4).min(255) as u8;
        }
        f
    };
    for (min, key) in [(0, false), (5, true), (11, false)] {
        let opts = super::Vp9Options {
            keyint_min: min,
            speed: 6,
            ..lag0(100)
        };
        let packets = encode_with(opts, (128, 128), 14, gradient);
        let headers = headers_of(&packets);
        assert_eq!(headers[10].key_frame, key, "keyint_min {min}");
        assert_eq!(packets[10].keyframe, key, "keyint_min {min}");
        assert!(!headers[11].key_frame);
    }
}

#[test]
fn tile_columns_follow_their_option() {
    let opts = super::Vp9Options {
        tile_columns: Some(1),
        threads: 1,
        ..lag0(100)
    };
    let packets = encode_with(opts, (640, 64), 2, |i| moving(640, 64, i));
    for h in headers_of(&packets) {
        assert_eq!(h.log2_tile_cols, 1);
    }
    assert_decodes(&packets, (640, 64), 2);
}

#[test]
fn the_first_key_frame_is_planned_from_the_lookahead() {
    // Still pictures: the lookahead's probe finds the inter frames cost
    // next to nothing, and the key frame takes at most twenty frames'
    // share of the bitrate (with a guess of an inter frame's cost it
    // took a fraction of that); the stream stays near its rate.
    let (w, h) = (320u32, 192u32);
    let rate = 400_000u32;
    let opts = super::Vp9Options {
        bitrate: rate,
        speed: 6,
        lag: Some(16),
        threads: 2,
        ..super::Vp9Options::default()
    };
    let packets = encode_with(opts, (w, h), 60, |i| noisy(w, h, 0, i));
    assert_decodes(&packets, (w, h), 60);
    let share = f64::from(rate) / 30.0 / 8.0;
    let key = packets[0].data.len() as f64;
    assert!(
        key > 5.0 * share && key < 30.0 * share,
        "{key} bytes, {share} a frame"
    );
    let total = packet_bytes(&packets) as f64 * 8.0 / 2.0;
    assert!((total / f64::from(rate) - 1.0).abs() < 0.3, "{total} b/s");
}

/// Luma of full-range noise, new in every picture: no quantizer the rate
/// control has codes it within a small rate.
fn static_noise(w: u32, h: u32, i: u32) -> VideoFrame {
    let mut f = moving(w, h, 0);
    for (k, v) in f.y.samples.make_mut().iter_mut().enumerate() {
        let n = ((k as u32) ^ i.wrapping_mul(0x9e37_79b9)).wrapping_mul(2_654_435_761) >> 24;
        *v = n as u8;
    }
    f
}

#[test]
fn a_low_buffer_drops_frames() {
    let (w, h) = (128u32, 64u32);
    let n = 60;
    let opts = super::Vp9Options {
        bitrate: 60_000,
        cbr_buffer_ms: 500,
        speed: 9,
        ..lag0(60)
    };
    let run = |o: super::Vp9Options| {
        let p = encode_with(o, (w, h), n, |i| static_noise(w, h, i));
        assert_decodes(&p, (w, h), p.len());
        p
    };
    let kept = run(opts);
    assert_eq!(kept.len(), n as usize);
    let dropping = run(super::Vp9Options {
        drop_frame_pct: 30,
        ..opts
    });
    // Some frames are dropped, the key frame and those coded keep their
    // place, and the stream comes near the rate (at the coarsest
    // quantizer it is four times over without).
    assert!(dropping.len() < kept.len(), "{} packets", dropping.len());
    assert!(dropping.len() > 5, "{} packets", dropping.len());
    assert!(dropping[0].keyframe && dropping[0].pts == 0);
    assert!(dropping.windows(2).all(|p| p[0].pts < p[1].pts));
    let rate = |p: &[Packet]| packet_bytes(p) as f64 * 8.0 / (f64::from(n) / 30.0);
    let (with, without) = (rate(&dropping), rate(&kept));
    assert!(without > 2.0 * 60_000.0, "{without} without dropping");
    assert!(with < 1.5 * 60_000.0, "{with} with dropping");
    // A rate the pictures fit in drops none.
    let easy = encode_with(
        super::Vp9Options {
            drop_frame_pct: 30,
            bitrate: 400_000,
            ..opts
        },
        (w, h),
        n,
        |i| moving(w, h, i),
    );
    assert_eq!(easy.len(), n as usize);
    let past = super::Vp9Options {
        drop_frame_pct: 101,
        ..opts
    };
    assert!(Vp9Encoder::with_options(w, h, past).is_err());
}

/// Each packet's header and the segment map its frame leaves.
fn segmaps_of(packets: &[Packet]) -> Vec<(header::Header, Vec<u8>)> {
    let mut dec = Decoder::new();
    packets
        .iter()
        .map(|p| {
            let h = header::parse(dec.persist(), &p.data)
                .expect("header")
                .header;
            dec.decode(&p.data).expect("decode");
            (h, dec.last_segmap().to_vec())
        })
        .collect()
}

#[test]
fn cyclic_refresh_moves_a_band_of_finer_superblocks() {
    // 10×2 superblocks: a tenth is two a frame, the next two each time.
    let (w, h) = (640u32, 128u32);
    let opts = super::Vp9Options {
        aq_mode: super::AqMode::CyclicRefresh,
        speed: 9,
        ..lag0(120)
    };
    let packets = encode_with(opts, (w, h), 5, |i| moving(w, h, i));
    assert_decodes(&packets, (w, h), 5);
    let maps = segmaps_of(&packets);
    assert!(!maps[0].0.seg.enabled, "the key frame refreshes all");
    let mi_cols = 80;
    for (i, (hd, map)) in maps.iter().enumerate().skip(1) {
        assert!(hd.seg.enabled && hd.seg.update_map, "frame {i}");
        assert!(hd.seg.feature_on[1][0] && hd.seg.feature_data[1][0] < 0);
        assert!(!hd.seg.feature_on[0][0]);
        // Superblocks 2i − 2 and 2i − 1 of the first row are segment 1.
        for (k, &s) in map.iter().enumerate() {
            let sb = (k / mi_cols / 8) * 10 + (k % mi_cols) / 8;
            assert_eq!(s, u8::from(sb / 2 == i - 1), "frame {i} block {k}");
        }
    }
    let both = super::Vp9Options { aq: true, ..opts };
    assert!(Vp9Encoder::with_options(w, h, both).is_err());
}

#[test]
fn equator_aq_codes_the_polar_rows_coarser() {
    let (w, h) = (128u32, 256u32);
    let opts = super::Vp9Options {
        aq_mode: super::AqMode::Equator360,
        speed: 6,
        ..lag0(100)
    };
    let packets = encode_with(opts, (w, h), 3, |i| noisy(w, h, i, i));
    assert_decodes(&packets, (w, h), 3);
    for (hd, map) in segmaps_of(&packets) {
        let s = &hd.seg;
        assert!(s.enabled);
        assert_eq!(s.feature_data[0][0], 0);
        assert!(0 < s.feature_data[1][0] && s.feature_data[1][0] < s.feature_data[2][0]);
        // (A block takes its top-left row's segment.)
        assert_eq!(map[0], 2);
        assert_eq!(map[16 * 16], 0);
    }
}

/// A screen: dark text-like strokes on a light page that scrolls up by
/// whole rows, a line of new strokes appearing at the bottom each frame.
fn page(w: u32, h: u32, i: u32) -> VideoFrame {
    let mut f = solid(w, h, 230, 128, 128);
    for (k, v) in f.y.samples.make_mut().iter_mut().enumerate() {
        let (r, c) = ((k as u32 / w) + 3 * i, k as u32 % w);
        let glyph = (r / 12).wrapping_mul(0x9e37_79b9) ^ (c / 7).wrapping_mul(0x85eb_ca6b);
        if r % 12 < 9 && c % 7 < 5 && (glyph >> 7) % 3 != 0 && ((r * 5 + c * 3) % 4 != 0) {
            *v = 30;
        }
    }
    f
}

#[test]
fn screen_content_codes_no_alt_ref() {
    let (w, h) = (128u32, 96u32);
    let opts = super::Vp9Options {
        qindex: 120,
        lag: Some(16),
        speed: 6,
        threads: 2,
        ..super::Vp9Options::default()
    };
    let superframes = |p: &[Packet]| {
        p.iter()
            .filter(|p| p.data.last().is_some_and(|&m| m & 0xe0 == 0xc0))
            .count()
    };
    let camera = encode_with(opts, (w, h), 30, |i| page(w, h, i / 4));
    assert!(superframes(&camera) > 0);
    let screen = encode_with(
        super::Vp9Options {
            tune_content: super::TuneContent::Screen,
            ..opts
        },
        (w, h),
        30,
        |i| page(w, h, i / 4),
    );
    assert_eq!(superframes(&screen), 0);
    assert_eq!(screen.len(), 30);
    assert_decodes(&screen, (w, h), 30);
}

#[test]
fn ssim_tuning_moves_bits_to_flat_blocks() {
    // Left: a gentle ramp under faint noise; right: heavy noise.
    let (w, h) = (128u32, 64u32);
    let picture = || {
        let mut f = moving(w, h, 0);
        for (k, v) in f.y.samples.make_mut().iter_mut().enumerate() {
            let (r, c) = (k as u32 / w, k as u32 % w);
            let n = (k as u32).wrapping_mul(2_654_435_761) >> 24;
            *v = if c < 64 {
                (60 + r + c / 4 + n % 12) as u8
            } else {
                (30 + n * 3 / 4) as u8
            };
        }
        f
    };
    let err = |tune| {
        let opts = super::Vp9Options { tune, ..lag0(80) };
        let p = encode_with(opts, (w, h), 1, |_| picture());
        let decoded = Decoder::new()
            .decode(&p[0].data)
            .expect("decode")
            .expect("shown");
        let src = vis(&picture().y);
        let (mut flat, mut busy) = (0u64, 0u64);
        for (k, (&a, &b)) in src.iter().zip(&decoded.y).enumerate() {
            let d = u64::from(a.abs_diff(b)).pow(2);
            if k as u32 % w < 64 {
                flat += d
            } else {
                busy += d
            }
        }
        (flat, busy)
    };
    let (psnr_flat, psnr_busy) = err(super::Tune::Psnr);
    let (ssim_flat, ssim_busy) = err(super::Tune::Ssim);
    assert!(ssim_flat < psnr_flat, "{ssim_flat} against {psnr_flat}");
    assert!(ssim_busy >= psnr_busy, "{ssim_busy} against {psnr_busy}");
}

#[test]
fn the_denoiser_brings_a_noisy_still_picture_near_the_clean_one() {
    // A still picture under fresh noise each frame, coded losslessly: what
    // comes out is the denoised source.
    let (w, h) = (64u32, 48u32);
    let clean = moving(w, h, 0);
    let noisy = |i: u32| {
        let mut f = clean.clone();
        for (k, v) in f.y.samples.make_mut().iter_mut().enumerate() {
            let n = ((k as u32) ^ i.wrapping_mul(0x9e37_79b9)).wrapping_mul(2_654_435_761) >> 29;
            *v = (i32::from(*v) + n as i32 - 4).clamp(0, 255) as u8;
        }
        f
    };
    let err = |level: u8| {
        let opts = super::Vp9Options {
            noise_sensitivity: level,
            ..lag0(0)
        };
        let packets = encode_with(opts, (w, h), 8, noisy);
        let mut dec = Decoder::new();
        let mut last = None;
        for p in &packets {
            last = dec.decode(&p.data).expect("decode");
        }
        let pic = last.expect("shown");
        // Chroma comes through as it went in.
        assert_eq!(pic.u, vis(&clean.u));
        let src = vis(&clean.y);
        src.iter()
            .zip(&pic.y)
            .map(|(&a, &b)| u64::from(a.abs_diff(b)).pow(2))
            .sum::<u64>()
    };
    let (raw, denoised) = (err(0), err(1));
    assert!(denoised * 2 < raw, "{denoised} against {raw}");
    let past = super::Vp9Options {
        noise_sensitivity: 7,
        ..lag0(0)
    };
    assert!(Vp9Encoder::with_options(w, h, past).is_err());
}

#[test]
fn each_temporal_layer_decodes_without_those_above() {
    let (w, h) = (128u32, 64u32);
    for (layers, pattern) in [(2u8, &[0u8, 1][..]), (3, &[0, 2, 1, 2][..])] {
        let opts = super::Vp9Options {
            temporal_layers: layers,
            bitrate: 300_000,
            cbr_buffer_ms: 1000,
            speed: 9,
            lag: Some(16),
            ..lag0(100)
        };
        let mut encoder = Vp9Encoder::with_options(w, h, opts).expect("encoder");
        let mut coded = Vec::new();
        for i in 0..24u32 {
            let p = encoder
                .push(&VideoInput {
                    frame: noisy(w, h, i, i),
                    pts: i64::from(i),
                    duration: 1,
                    timescale: 30,
                })
                .expect("push");
            // No lookahead in layers: each picture comes out at once.
            assert_eq!(p.len(), 1);
            coded.push((encoder.temporal_layer(), p[0].data.clone()));
        }
        let mut dec = Decoder::new();
        for (i, (layer, data)) in coded.iter().enumerate() {
            assert_eq!(*layer, pattern[i % pattern.len()], "picture {i}");
            let h = header::parse(dec.persist(), data).expect("header").header;
            dec.decode(data).expect("decode");
            assert!(h.error_resilient || i == 0);
            if i > 0 {
                let refresh = match layer {
                    0 => 0x01,
                    1 => 0x02,
                    _ => 0x00,
                };
                assert_eq!(h.refresh_flags, refresh, "picture {i}");
            }
        }
        // All the layers, then each set of the lower ones alone: the
        // pictures of a set are those of all.
        let decode = |top: u8| -> Vec<(usize, Vec<u8>)> {
            let mut dec = Decoder::new();
            coded
                .iter()
                .enumerate()
                .filter(|(_, (l, _))| *l <= top)
                .map(|(i, (_, d))| (i, dec.decode(d).expect("decode").expect("shown").y))
                .collect()
        };
        let all = decode(layers - 1);
        assert_eq!(all.len(), 24);
        for top in 0..layers - 1 {
            let some = decode(top);
            assert!(some.len() < 24);
            for (i, y) in &some {
                assert_eq!(y, &all[*i].1, "{layers} layers, up to {top}, picture {i}");
            }
        }
    }
    let past = super::Vp9Options {
        temporal_layers: 4,
        ..lag0(100)
    };
    assert!(Vp9Encoder::with_options(w, h, past).is_err());
}

#[test]
fn a_region_of_interest_map_codes_its_segments_features() {
    // 128×64: 16×8 blocks. The top two rows of blocks skip, the left half
    // of the rest is finer, filtered less and predicts from LAST only.
    let (w, h) = (128u32, 64u32);
    let mut roi = super::RoiMap {
        map: vec![0; 16 * 8],
        ..super::RoiMap::default()
    };
    for (k, s) in roi.map.iter_mut().enumerate() {
        *s = if k < 32 {
            2
        } else if k % 16 < 8 {
            1
        } else {
            0
        };
    }
    roi.delta_q[1] = -30;
    roi.delta_lf[1] = -10;
    roi.ref_frame[1] = Some(1);
    roi.skip[2] = true;
    // (Unfiltered, so that the skipped rows away from the segments below
    // are the picture before's.)
    roi.delta_lf[2] = -63;
    for speed in [2u8, 6, 9, 10] {
        let opts = super::Vp9Options {
            speed,
            lag: Some(16),
            ..lag0(120)
        };
        let mut encoder = Vp9Encoder::with_options(w, h, opts).expect("encoder");
        encoder.set_roi_map(Some(roi.clone())).expect("roi");
        let mut packets = Vec::new();
        for i in 0..20u32 {
            packets.extend(
                encoder
                    .push(&VideoInput {
                        frame: noisy(w, h, i, i),
                        pts: i64::from(i),
                        duration: 1,
                        timescale: 30,
                    })
                    .expect("push"),
            );
        }
        packets.extend(VideoEncoder::drain(&mut encoder).expect("drain"));
        assert_decodes(&packets, (w, h), 20);
        // Each shown picture: its header's features, and the first eight
        // skipped rows the picture before's samples (zero vector, no
        // residual, no loop filter).
        let mut dec = Decoder::new();
        let mut before: Option<Vec<u8>> = None;
        for p in &packets {
            for f in crate::decode::split_superframe(&p.data).expect("superframe") {
                let hd = header::parse(dec.persist(), f).expect("header").header;
                let shown = dec.decode(f).expect("decode");
                if !hd.show_frame {
                    continue;
                }
                let s = &hd.seg;
                assert!(s.enabled, "speed {speed}");
                assert_eq!(s.feature_data[1][0], -30);
                assert_eq!(s.feature_data[1][1], -10);
                let inter = !hd.key_frame;
                assert_eq!(s.feature_on[1][2], inter);
                assert_eq!(s.feature_on[2][3], inter);
                assert_eq!(dec.last_segmap(), &roi.map[..]);
                let y = shown.expect("shown").y;
                if let (true, Some(b)) = (inter, &before) {
                    assert_eq!(&y[..8 * 128], &b[..8 * 128], "speed {speed}");
                }
                before = Some(y);
            }
        }
    }
    let mut encoder = Vp9Encoder::with_options(w, h, lag0(120)).expect("encoder");
    let wrong = |f: &dyn Fn(&mut super::RoiMap)| {
        let mut r = roi.clone();
        f(&mut r);
        r
    };
    for bad in [
        wrong(&|r| r.map.pop().map_or((), |_| ())),
        wrong(&|r| r.map[0] = 8),
        wrong(&|r| r.delta_lf[0] = 64),
        wrong(&|r| r.ref_frame[0] = Some(4)),
        wrong(&|r| r.ref_frame[2] = Some(0)),
    ] {
        assert!(encoder.set_roi_map(Some(bad)).is_err());
    }
    let mut aq = Vp9Encoder::with_options(
        w,
        h,
        super::Vp9Options {
            aq: true,
            ..lag0(120)
        },
    )
    .expect("encoder");
    assert!(aq.set_roi_map(Some(roi)).is_err());
}

#[test]
fn a_finer_region_of_interest_comes_back_nearer_the_source() {
    // The left half of the picture 40 indices finer: it comes back nearer
    // the source than without the map (its blocks weigh rate against
    // error at their own quantizer), the right half no nearer.
    let (w, h) = (128u32, 64u32);
    let mut roi = super::RoiMap {
        map: (0..16 * 8).map(|k| u8::from(k % 16 < 8)).collect(),
        ..super::RoiMap::default()
    };
    roi.delta_q[1] = -40;
    for speed in [6u8, 9] {
        let err = |map: Option<super::RoiMap>| {
            let opts = super::Vp9Options { speed, ..lag0(120) };
            let mut encoder = Vp9Encoder::with_options(w, h, opts).expect("encoder");
            encoder.set_roi_map(map).expect("roi");
            let mut dec = Decoder::new();
            let (mut left, mut right) = (0u64, 0u64);
            for i in 0..4 {
                let p = encoder.push(&input(noisy(w, h, i, i))).expect("push");
                let y = dec.decode(&p[0].data).expect("decode").expect("shown").y;
                for (k, (&a, &b)) in vis(&noisy(w, h, i, i).y).iter().zip(&y).enumerate() {
                    let d = u64::from(a.abs_diff(b)).pow(2);
                    if k as u32 % w < 64 {
                        left += d
                    } else {
                        right += d
                    }
                }
            }
            (left, right)
        };
        let (plain, with) = (err(None), err(Some(roi.clone())));
        assert!(
            with.0 * 3 < plain.0 * 2,
            "speed {speed}: {with:?} against {plain:?}"
        );
        assert!(
            with.1 * 10 >= plain.1 * 9,
            "speed {speed}: {with:?} against {plain:?}"
        );
    }
}

#[test]
fn a_corpus_gives_its_hard_clip_the_bits_of_its_easy_one() {
    // An easy clip (still) and a hard one (a noisy pan), two passes each,
    // at a rate both spend.
    let (w, h) = (160u32, 96u32);
    let clips: [&dyn Fn(u32) -> VideoFrame; 2] = [&|_| moving(w, h, 0), &|i| noisy(w, h, i * 3, i)];
    let base = super::Vp9Options {
        threads: 2,
        speed: 8,
        ..super::Vp9Options::default()
    };
    let rate = 60_000u32;
    let run = |clip: &dyn Fn(u32) -> VideoFrame, corpus: Option<u32>| -> (u32, u64) {
        let first_opts = super::Vp9Options {
            qindex: 120,
            speed: 10,
            ..base
        };
        let first: Vec<u64> = encode_with(first_opts, (w, h), 30, clip)
            .iter()
            .map(|p| p.data.len() as u64 * 8)
            .collect();
        let opts = super::Vp9Options {
            bitrate: rate,
            corpus_complexity: corpus.unwrap_or(0),
            ..base
        };
        let mut encoder = Vp9Encoder::with_options(w, h, opts).expect("encoder");
        let complexity = encoder.first_pass_complexity(&first, 120);
        encoder.set_second_pass(&first, 120);
        let mut bits = 0;
        for i in 0..30u32 {
            let p = encoder.push(&input(clip(i))).expect("push");
            bits += p.iter().map(|p| p.data.len() as u64 * 8).sum::<u64>();
        }
        let p = VideoEncoder::drain(&mut encoder).expect("drain");
        bits += p.iter().map(|p| p.data.len() as u64 * 8).sum::<u64>();
        (complexity, bits)
    };
    let alone: Vec<(u32, u64)> = clips.iter().map(|c| run(*c, None)).collect();
    let (easy, hard) = (alone[0].0, alone[1].0);
    assert!(hard > 2 * easy.max(1), "{easy} and {hard}");
    let corpus = (easy + hard) / 2;
    let together: Vec<u64> = clips.iter().map(|c| run(*c, Some(corpus)).1).collect();
    // The easy clip spends less than the bitrate and the hard one more;
    // the two come to about what they spent alone.
    assert!(
        together[0] < alone[0].1,
        "{} against {}",
        together[0],
        alone[0].1
    );
    assert!(
        together[1] > alone[1].1,
        "{} against {}",
        together[1],
        alone[1].1
    );
    let (sum, before) = (together[0] + together[1], alone[0].1 + alone[1].1);
    assert!(
        sum as f64 > 0.75 * before as f64 && (sum as f64) < 1.25 * before as f64,
        "{sum} against {before}"
    );
}

#[test]
fn layered_alt_refs_are_shown_as_they_are() {
    // A scrolling page at a fixed quantizer with the lookahead: with layers,
    // hidden alt-refs within the golden groups are shown in their
    // targets' places as they are (`show_existing_frame`, one byte),
    // and the pictures decode near the source, as with one layer.
    let (w, h) = (128u32, 96u32);
    let clip = |i: u32| page(w, h, i / 4);
    let opts = super::Vp9Options {
        qindex: 120,
        lag: Some(16),
        speed: 6,
        threads: 2,
        ..super::Vp9Options::default()
    };
    let shows = |p: &[Packet]| {
        p.iter()
            .filter(|p| p.data.len() == 1 && p.data[0] >> 3 & 1 == 1)
            .count()
    };
    let worst = |p: &[Packet]| {
        let params = CodecParams::with_format(CodecId::Vp9, w, h, PixelFormat::Yuv420p);
        let mut decoder = Vp9Decoder::open(&params).expect("decoder");
        let mut out = Vec::new();
        for p in p {
            out.extend(decoder.push_all(p).expect("decode"));
        }
        assert_eq!(out.len(), 30);
        out.iter()
            .enumerate()
            .map(|(i, o)| psnr(&vis(&clip(i as u32).y), &vis(&o.frame.y)))
            .fold(f64::INFINITY, f64::min)
    };
    let one = encode_with(opts, (w, h), 30, clip);
    assert_eq!(shows(&one), 0);
    let base = worst(&one);
    let mut n = Vec::new();
    for layers in [2u8, 3] {
        let p = encode_with(
            super::Vp9Options {
                alt_ref_layers: layers,
                ..opts
            },
            (w, h),
            30,
            clip,
        );
        assert_eq!(p.len(), 30);
        n.push(shows(&p));
        let q = worst(&p);
        assert!(q > base - 1.5, "{layers} layers: {q} against {base}");
    }
    // One in the golden group of twelve with an alt-ref, three with three
    // layers (at 6, then 3 and 9).
    assert_eq!(n, [1, 3]);
    let past = super::Vp9Options {
        alt_ref_layers: 4,
        ..opts
    };
    assert!(Vp9Encoder::with_options(w, h, past).is_err());
}

use cutlass_decode::{
    CodecError, CodecId, CodecParams, ColorMatrix, ColorPrimaries, ColorRange, DecodeFactory,
    Decoder as EngineTrait, MatrixCoefficients, Output, Packet, PictureOutput, PictureSamples,
    PixelFormat, Plane, Transfer, VideoFrame,
};
use cutlass_encode::video::{VideoEncoder as _, VideoInput};
use cutlass_encode::{CpuFrame, EncodeFactory, FrameInput, FrameTime};

use super::{EngineDecoder, register_engine_decoders};
use crate::decode::{Decoder, Picture, split_superframe};
use crate::{Vp9Encoder, Vp9Options, register_engine_encoders};

/// A decoded picture as the engine takes it: size, format and bytes.
#[derive(Debug, PartialEq, Eq)]
struct Shown {
    width: u32,
    height: u32,
    pixel: PixelFormat,
    bytes: Vec<u8>,
}

fn pictures(outputs: Vec<Output>) -> Vec<PictureOutput> {
    outputs
        .into_iter()
        .map(|output| match output {
            Output::Picture(picture) => picture,
            Output::Audio(_) => panic!("audio"),
        })
        .collect()
}

fn shown(picture: &PictureOutput) -> Shown {
    let PictureSamples::Cpu(bytes) = picture.samples() else {
        panic!("platform buffer");
    };
    let info = picture.info();
    Shown {
        width: info.width,
        height: info.height,
        pixel: info.pixel,
        bytes: bytes.to_vec(),
    }
}

/// Every packet (its index as the pts) through `decoder`, then the drain.
fn decode_all(decoder: &mut dyn EngineTrait, packets: &[Vec<u8>]) -> Vec<PictureOutput> {
    let mut out = Vec::new();
    for (index, data) in packets.iter().enumerate() {
        let packet = Packet::video(data.clone(), index as i64, 1, 30, index == 0);
        out.extend(pictures(decoder.push(&packet).expect("push")));
    }
    out.extend(pictures(decoder.drain().expect("drain")));
    out
}

/// `plane` at `w` by `h`, its last column and row repeated.
fn padded(plane: &[u8], own_w: usize, own_h: usize, sample: usize, w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for row in 0..h {
        let at = row.min(own_h - 1) * own_w * sample;
        let samples = &plane[at..at + own_w * sample];
        out.extend_from_slice(samples);
        for _ in own_w..w {
            out.extend_from_slice(&samples[samples.len() - sample..]);
        }
    }
    out
}

/// What the adapter should give for a picture of the crate's own decoder.
fn expected(picture: &Picture) -> Shown {
    let deep = picture.bit_depth > 8;
    let sample = if deep { 2 } else { 1 };
    let pixel = match (picture.bit_depth, picture.ss_x, picture.ss_y) {
        (8, 1, 1) => PixelFormat::Yuv420p,
        (10, 1, 1) => PixelFormat::Yuv420p10,
        (12, 1, 1) => PixelFormat::Yuv420p12,
        (8, 1, 0) => PixelFormat::Yuv422p,
        (10, 1, 0) => PixelFormat::Yuv422p10,
        (8, 0, 1) => PixelFormat::Yuv440p,
        (8, 0, 0) => PixelFormat::Yuv444p,
        (10, 0, 0) => PixelFormat::Yuv444p10,
        (12, 0, 0) => PixelFormat::Yuv444p12,
        other => panic!("format {other:?}"),
    };
    let (own_w, own_h) = (picture.width as usize, picture.height as usize);
    let (mut w, mut h) = (own_w, own_h);
    if picture.ss_x == 1 && picture.ss_y == 1 {
        w += w & 1;
        h += h & 1;
    }
    let chroma = |edge: usize, ss: u32| (edge + ss as usize) >> ss;
    let (own_cw, own_ch) = (chroma(own_w, picture.ss_x), chroma(own_h, picture.ss_y));
    let (cw, ch) = (chroma(w, picture.ss_x), chroma(h, picture.ss_y));
    let mut bytes = padded(&picture.y, own_w, own_h, sample, w, h);
    bytes.extend(padded(&picture.u, own_cw, own_ch, sample, cw, ch));
    bytes.extend(padded(&picture.v, own_cw, own_ch, sample, cw, ch));
    Shown {
        width: w as u32,
        height: h as u32,
        pixel,
        bytes,
    }
}

/// The crate's own decode of `packets`: every shown frame, in order.
fn reference(packets: &[Vec<u8>]) -> Vec<Picture> {
    let mut decoder = Decoder::new();
    let mut out = Vec::new();
    for data in packets {
        for frame in split_superframe(data).expect("superframe") {
            out.extend(decoder.decode(frame).expect("decode"));
        }
    }
    out
}

fn params() -> CodecParams {
    CodecParams::video(CodecId::Vp9, 64, 64)
}

/// A decoder from the factory playback opens (one thread a core), and
/// ones on the caller's thread and on three workers.
fn decoders(params: &CodecParams) -> Vec<(&'static str, Box<dyn EngineTrait>)> {
    let mut factory = DecodeFactory::new();
    register_engine_decoders(&mut factory);
    vec![
        ("factory", factory.open(params).expect("factory")),
        (
            "one thread",
            Box::new(EngineDecoder::open_with_threads(params, 1).expect("one")),
        ),
        (
            "three threads",
            Box::new(EngineDecoder::open_with_threads(params, 3).expect("three")),
        ),
    ]
}

/// Asserts every decoder of [`decoders`] gives `packets` as the crate's
/// own decoder does, one picture a shown frame.
fn assert_matches_the_crate(name: &str, params: &CodecParams, packets: &[Vec<u8>]) -> Vec<Shown> {
    let want: Vec<Shown> = reference(packets).iter().map(expected).collect();
    assert!(!want.is_empty(), "{name}");
    for (threads, mut decoder) in decoders(params) {
        assert_eq!(decoder.codec(), CodecId::Vp9);
        let got = decode_all(decoder.as_mut(), packets);
        assert_eq!(got.len(), want.len(), "{name} {threads}: pictures");
        for (index, (got, want)) in got.iter().zip(&want).enumerate() {
            assert_eq!(&shown(got), want, "{name} {threads}: picture {index}");
        }
        let times: Vec<i64> = got.iter().map(PictureOutput::pts).collect();
        assert!(times.is_sorted(), "{name} {threads}: {times:?}");
    }
    want
}

/// A fixture's name, its packets, and the pictures its oracle decoded.
type FixtureSet = (&'static str, Vec<Vec<u8>>, &'static [u8]);

macro_rules! fixture {
    ($($name:literal),+) => {
        vec![$(include_bytes!(concat!("../fixtures/", $name, ".bin")).to_vec()),+]
    };
}

/// The crate's fixtures (libvpx streams, with the pictures FFmpeg and
/// libvpx decode them to) come out of the adapter as they do out of the
/// crate's own decoder, and as those pictures: plain and alt-ref inter
/// frames, error resilience, segmentation, a size no block divides, and
/// references of another size.
#[test]
fn the_fixtures_decode_as_the_crate_decodes_them() {
    let sets: [FixtureSet; 6] = [
        (
            "normal32",
            fixture!("vp9_normal32_0", "vp9_normal32_1", "vp9_normal32_2"),
            include_bytes!("../fixtures/vp9_normal32.yuv"),
        ),
        (
            "altref64",
            fixture!(
                "vp9_altref64_0",
                "vp9_altref64_1",
                "vp9_altref64_2",
                "vp9_altref64_3",
                "vp9_altref64_4",
                "vp9_altref64_5",
                "vp9_altref64_6",
                "vp9_altref64_7"
            ),
            include_bytes!("../fixtures/vp9_altref64.yuv"),
        ),
        (
            "inter32",
            fixture!("vp9_inter32_0", "vp9_inter32_1"),
            include_bytes!("../fixtures/vp9_inter32.yuv"),
        ),
        (
            "aq",
            fixture!(
                "vp9_aq_0", "vp9_aq_1", "vp9_aq_2", "vp9_aq_3", "vp9_aq_4", "vp9_aq_5", "vp9_aq_6",
                "vp9_aq_7"
            ),
            include_bytes!("../fixtures/vp9_aq.yuv"),
        ),
        (
            "odd",
            fixture!("vp9_odd_0", "vp9_odd_1"),
            include_bytes!("../fixtures/vp9_odd.yuv"),
        ),
        (
            "scale",
            fixture!("vp9_scale_0", "vp9_scale_1", "vp9_scale_2"),
            include_bytes!("../fixtures/vp9_scale.yuv"),
        ),
    ];
    for (name, packets, yuv) in sets {
        let got = assert_matches_the_crate(name, &params(), &packets);
        assert_eq!(got.len(), packets.len(), "{name}");
        let bytes: Vec<u8> = got.iter().flat_map(|p| p.bytes.iter().copied()).collect();
        assert_eq!(bytes, yuv, "{name}: the oracle's pictures");
        assert!(got.iter().all(|p| p.pixel == PixelFormat::Yuv420p));
    }
}

/// One plane of a moving pattern, `depth` bits a sample.
fn pattern(w: u32, h: u32, plane: u32, depth: u8, t: u32) -> Vec<u16> {
    let max = (1u32 << depth) - 1;
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w + 2 * t, i / w);
            let wave = (x * (plane + 3) + y * 5 + plane * 70) % 160;
            (wave * max / 200 + max / 10) as u16
        })
        .collect()
}

/// A picture of [`pattern`] in `format`.
fn frame(format: PixelFormat, w: u32, h: u32, t: u32) -> VideoFrame {
    let (cw, ch) = format.chroma_wh(w, h).expect("planar");
    let depth = match format {
        PixelFormat::Yuv420p10 | PixelFormat::Yuv444p10 => 10,
        PixelFormat::Yuv420p12 => 12,
        _ => 8,
    };
    let plane = |index: u32, pw: u32, ph: u32| {
        let samples = pattern(pw, ph, index, depth, t);
        if depth == 8 {
            Plane::tight(pw, ph, samples.iter().map(|&s| s as u8).collect()).expect("plane")
        } else {
            let bytes = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
            Plane::bytes(pw, ph, 2, bytes).expect("plane")
        }
    };
    VideoFrame::planar(
        format,
        w,
        h,
        plane(0, w, h),
        plane(1, cw, ch),
        plane(2, cw, ch),
    )
    .expect("frame")
}

/// `count` pictures of [`frame`] through this crate's encoder.
fn encoded(format: PixelFormat, w: u32, h: u32, count: u32) -> Vec<Vec<u8>> {
    let mut params = CodecParams::with_format(CodecId::Vp9, w, h, format);
    params.matrix = MatrixCoefficients::Bt709;
    let mut encoder = Vp9Encoder::open(&params).expect("encoder");
    let mut packets = Vec::new();
    for t in 0..count {
        let input = VideoInput {
            frame: frame(format, w, h, t),
            pts: i64::from(t),
            duration: 1,
            timescale: 30,
        };
        packets.extend(encoder.push(&input).expect("encode"));
    }
    packets.extend(encoder.drain().expect("drain"));
    packets.into_iter().map(|packet| packet.data).collect()
}

/// Each profile's pictures come out in the stream's own format: 8-bit
/// planes, or little-endian 16-bit ones for 10 and 12 bits, 4:2:0 and
/// other chroma sizes. The streams have hidden alt-ref frames in
/// superframes, which give no picture of their own.
#[test]
fn every_profile_comes_out_in_its_own_format() {
    for (format, sample) in [
        (PixelFormat::Yuv420p, 1),
        (PixelFormat::Yuv420p10, 2),
        (PixelFormat::Yuv420p12, 2),
        (PixelFormat::Yuv444p, 1),
        (PixelFormat::Yuv422p, 1),
        (PixelFormat::Yuv444p10, 2),
    ] {
        let name = format!("{format:?}");
        let packets = encoded(format, 48, 32, 20);
        assert_eq!(packets.len(), 20, "{name}");
        let frames: usize = packets
            .iter()
            .map(|data| split_superframe(data).expect("superframe").len())
            .sum();
        assert!(frames > 20, "{name}: hidden frames in superframes");
        let mut read = CodecParams::video(CodecId::Vp9, 48, 32);
        read.pixel = format;
        let got = assert_matches_the_crate(&name, &read, &packets);
        assert_eq!(got.len(), 20, "{name}");
        let (cw, ch) = format.chroma_wh(48, 32).expect("planar");
        for picture in &got {
            assert_eq!((picture.width, picture.height), (48, 32));
            assert_eq!(picture.pixel, format, "{name}");
            assert_eq!(
                picture.bytes.len(),
                (48 * 32 + 2 * cw * ch) as usize * sample
            );
        }
    }
}

/// The engine's 4:2:0 pictures have even edges: an odd stream repeats its
/// last column and row, at 8 and at 10 bits, and keeps its own size as
/// the render size. Other chroma sizes keep odd edges.
#[test]
fn odd_sizes_repeat_their_last_column_and_row() {
    for (format, size) in [
        (PixelFormat::Yuv420p, (34, 18)),
        (PixelFormat::Yuv420p10, (34, 18)),
        (PixelFormat::Yuv444p, (33, 17)),
    ] {
        let name = format!("{format:?}");
        let packets = encoded(format, 33, 17, 3);
        let got =
            assert_matches_the_crate(&name, &CodecParams::video(CodecId::Vp9, 33, 17), &packets);
        assert!(got.iter().all(|p| (p.width, p.height) == size), "{name}");

        let mut decoder = EngineDecoder::open_with_threads(&params(), 1).expect("open");
        let out = decode_all(&mut decoder, &packets);
        let info = out[0].info();
        assert_eq!((info.render_width, info.render_height), (33, 17), "{name}");
    }
}

/// A key frame of [`frame`] alone, and the inter frame after it.
fn key_and_inter() -> (Vec<u8>, Vec<u8>) {
    let mut encoder = Vp9Encoder::with_options(
        32,
        32,
        Vp9Options {
            lag: Some(0),
            ..Vp9Options::default()
        },
    )
    .expect("encoder");
    let mut packets = Vec::new();
    for t in 0..2 {
        let input = VideoInput {
            frame: frame(PixelFormat::Yuv420p, 32, 32, t),
            pts: i64::from(t),
            duration: 1,
            timescale: 30,
        };
        packets.extend(encoder.push(&input).expect("encode"));
    }
    packets.extend(encoder.drain().expect("drain"));
    assert_eq!(packets.len(), 2);
    let inter = packets.remove(1).data;
    (packets.remove(0).data, inter)
}

fn video(data: Vec<u8>, pts: i64) -> Packet {
    Packet::video(data, pts, 1, 30, false)
}

#[test]
fn open_takes_vp9_alone_and_any_container_size() {
    let other = CodecParams::video(CodecId::Vp8, 32, 32);
    assert!(matches!(
        EngineDecoder::open(&other),
        Err(CodecError::Unsupported { what: "codec" })
    ));
    let mut factory = DecodeFactory::new();
    register_engine_decoders(&mut factory);
    assert!(factory.open(&other).is_err());

    // A container that names no size or format: the frame header has both.
    let (key, _) = key_and_inter();
    let mut bare = CodecParams::video(CodecId::Vp9, 0, 0);
    bare.pixel = PixelFormat::None;
    let mut decoder = EngineDecoder::open_with_threads(&bare, 1).expect("bare");
    let out = pictures(decoder.push(&video(key, 0)).expect("key"));
    assert_eq!((out[0].info().width, out[0].info().height), (32, 32));
}

/// Malformed packets are errors, never a picture or a panic, and leave
/// the references as they were: the stream goes on from the next packet.
#[test]
fn malformed_packets_are_refused_and_the_stream_goes_on() {
    let (key, inter) = key_and_inter();
    let want = reference(&[key.clone(), inter.clone()]);
    for threads in [1, 3] {
        let mut decoder = EngineDecoder::open_with_threads(&params(), threads).expect("open");
        assert!(
            decoder
                .push(&video(Vec::new(), 0))
                .expect("empty")
                .is_empty()
        );
        assert!(decoder.push(&video(vec![0xff; 40], 0)).is_err(), "garbage");
        // An inter frame before any key frame has nothing to predict from.
        assert!(decoder.push(&video(inter.clone(), 0)).is_err(), "no key");
        let mut out = pictures(decoder.push(&video(key.clone(), 5)).expect("key"));
        // A superframe index that names more bytes than the packet has.
        let mut index = key.clone();
        index.extend_from_slice(&[0xc8, 0xff, 0xff, 0xc8]);
        assert!(decoder.push(&video(index, 6)).is_err(), "superframe");
        assert!(decoder.push(&video(vec![0x82, 0x49], 6)).is_err(), "cut");
        out.extend(pictures(
            decoder.push(&video(inter.clone(), 7)).expect("inter"),
        ));
        out.extend(pictures(decoder.drain().expect("drain")));
        assert_eq!(out.len(), 2, "{threads} threads");
        assert_eq!(shown(&out[0]), expected(&want[0]));
        assert_eq!(shown(&out[1]), expected(&want[1]));
        assert_eq!((out[0].pts(), out[1].pts()), (5, 7));
        assert_eq!(out[1].timescale(), 30);
    }
}

/// Every cut of a key frame and of an inter frame, and every byte of them
/// flipped in turn: an error or a picture of the stream's size, no panic.
#[test]
fn cut_and_flipped_packets_do_not_panic() {
    let (key, inter) = key_and_inter();
    for threads in [1, 2] {
        let mut decoder = EngineDecoder::open_with_threads(&params(), threads).expect("open");
        let check = |data: Vec<u8>, decoder: &mut EngineDecoder| {
            let mut out = decoder.push(&video(data, 0)).unwrap_or_default();
            out.extend(decoder.drain().unwrap_or_default());
            for picture in pictures(out) {
                let info = picture.info();
                assert!(info.width <= 65536 && info.height <= 65536);
            }
        };
        for source in [&key, &inter] {
            for cut in 0..source.len() {
                decoder.reset();
                check(key.clone(), &mut decoder);
                check(source[..cut].to_vec(), &mut decoder);
            }
            for at in 0..source.len() {
                let mut flipped = source.clone();
                flipped[at] ^= 0x55;
                decoder.reset();
                check(key.clone(), &mut decoder);
                check(flipped, &mut decoder);
            }
        }
    }
}

/// A seek resets: the references and the pictures still to come out are
/// dropped, and decoding starts again at a key frame. A drain gives what
/// is left and the stream goes on after it.
#[test]
fn reset_drops_references_and_pending_pictures() {
    let (key, inter) = key_and_inter();
    let want = reference(&[key.clone(), inter.clone()]);
    for threads in [1, 4] {
        let mut decoder = EngineDecoder::open_with_threads(&params(), threads).expect("open");
        decoder.push(&video(key.clone(), 0)).expect("key");
        decoder.reset();
        assert!(decoder.drain().expect("drain").is_empty(), "dropped");
        assert!(decoder.push(&video(inter.clone(), 1)).is_err(), "no key");

        let mut out = pictures(decoder.push(&video(key.clone(), 10)).expect("key"));
        out.extend(pictures(decoder.drain().expect("drain")));
        assert_eq!(out.len(), 1);
        assert!(decoder.drain().expect("again").is_empty());
        out.extend(pictures(
            decoder.push(&video(inter.clone(), 11)).expect("inter"),
        ));
        out.extend(pictures(decoder.drain().expect("drain")));
        assert_eq!(out.len(), 2);
        assert_eq!(shown(&out[0]), expected(&want[0]));
        assert_eq!(shown(&out[1]), expected(&want[1]));
        assert_eq!((out[0].pts(), out[1].pts()), (10, 11));
    }
}

fn info_of(read: &CodecParams, packet: &[u8]) -> cutlass_decode::PictureInfo {
    let mut decoder = EngineDecoder::open_with_threads(read, 1).expect("open");
    let out = pictures(decoder.push(&video(packet.to_vec(), 0)).expect("key"));
    out[0].info()
}

/// The matrix and the range are the frame header's. The primaries and the
/// transfer are the container's when it names them, as an HDR file's
/// `vpcC` does, and follow the header's colour space otherwise.
#[test]
fn colour_is_the_headers_and_the_containers() {
    let packets = encoded(PixelFormat::Yuv420p10, 32, 32, 1);
    let key = &packets[0];

    let plain = info_of(&params(), key);
    assert_eq!(plain.matrix, ColorMatrix::Bt709);
    assert_eq!(plain.primaries, ColorPrimaries::Bt709);
    assert_eq!(plain.transfer, Transfer::Bt709);
    assert_eq!(plain.range, ColorRange::Limited);
    assert_eq!(
        (plain.cicp_primaries, plain.cicp_transfer, plain.cicp_matrix),
        (1, 1, 1)
    );

    // vpcC: version 1, profile 2, level 1.0, 10 bits 4:2:0 studio range,
    // BT.2020 primaries, PQ, BT.2020 matrix.
    let mut hdr = params();
    hdr.extra = vec![1, 0, 0, 0, 2, 10, 0xa2, 9, 16, 9, 0, 0];
    let info = info_of(&hdr, key);
    assert_eq!(info.pixel, PixelFormat::Yuv420p10);
    assert_eq!(info.primaries, ColorPrimaries::Bt2020);
    assert_eq!(info.transfer, Transfer::Pq);
    assert_eq!((info.cicp_primaries, info.cicp_transfer), (9, 16));
    // The header names BT.709 as its colour space, and stays the matrix.
    assert_eq!((info.matrix, info.cicp_matrix), (ColorMatrix::Bt709, 1));

    // The parameters' own fields go before the payload's.
    hdr.transfer = Transfer::Hlg;
    let info = info_of(&hdr, key);
    assert_eq!(info.primaries, ColorPrimaries::Bt2020);
    assert_eq!((info.transfer, info.cicp_transfer), (Transfer::Hlg, 18));

    // A payload that is not a version 1 vpcC, or names no colour, is none.
    for extra in [
        vec![0, 0, 0, 0, 2, 10, 0xa2, 9, 16, 9, 0, 0],
        vec![1, 0, 0, 0, 2, 10, 0xa2, 2, 2, 2, 0, 0],
        vec![1, 0, 0, 0, 2, 10, 0xa2, 9],
    ] {
        let mut read = params();
        read.extra = extra;
        assert_eq!(info_of(&read, key), plain);
    }

    // A header with no colour space (libvpx's default) takes the
    // container's matrix too.
    let unknown = include_bytes!("../fixtures/vp9_normal32_0.bin");
    let bare = info_of(&params(), unknown);
    assert_eq!(bare.matrix, ColorMatrix::Unspecified);
    assert_eq!(bare.primaries, ColorPrimaries::Unspecified);
    assert_eq!(bare.range, ColorRange::Limited);
    let mut named = params();
    named.extra = vec![1, 0, 0, 0, 0, 10, 0x82, 1, 1, 1, 0, 0];
    let info = info_of(&named, unknown);
    assert_eq!((info.matrix, info.cicp_matrix), (ColorMatrix::Bt709, 1));
    assert_eq!(info.primaries, ColorPrimaries::Bt709);
}

/// Export's path and back: RGBA frames through the factory export opens,
/// then the packets through the factory playback opens.
#[test]
fn export_and_playback_reach_vp9_through_the_engine_factories() {
    let mut encoders = EncodeFactory::new();
    register_engine_encoders(&mut encoders);
    let mut write = CodecParams::with_format(CodecId::Vp9, 64, 32, PixelFormat::Rgba8);
    write.bitrate = 400_000;
    let mut encoder = encoders.open(&write).expect("encoder");
    let mut packets = Vec::new();
    for index in 0..20 {
        let time = FrameTime::new(index, 1, 30).expect("time");
        // Orange: RGB 255, 128, 0.
        let rgba = [255_u8, 128, 0, 255].repeat(64 * 32);
        let frame = CpuFrame::new(64, 32, PixelFormat::Rgba8, rgba, time).expect("frame");
        packets.extend(encoder.push_frame(&FrameInput::Cpu(frame)).expect("push"));
    }
    packets.extend(encoder.drain().expect("drain"));
    assert_eq!(packets.len(), 20);

    let mut decoders = DecodeFactory::new();
    register_engine_decoders(&mut decoders);
    let mut decoder = decoders
        .open(&CodecParams::video(CodecId::Vp9, 64, 32))
        .expect("decoder");
    let mut out = Vec::new();
    for packet in &packets {
        out.extend(pictures(decoder.push(packet).expect("decode")));
    }
    out.extend(pictures(decoder.drain().expect("drain")));
    assert_eq!(out.len(), 20);
    for (index, picture) in out.iter().enumerate() {
        let info = picture.info();
        assert_eq!((info.width, info.height), (64, 32));
        assert_eq!(info.pixel, PixelFormat::Yuv420p);
        assert_eq!(info.matrix, ColorMatrix::Bt709);
        assert_eq!(picture.pts(), index as i64);
        let PictureSamples::Cpu(bytes) = picture.samples() else {
            panic!("platform buffer");
        };
        // BT.709 studio range orange, as export converts it: Y 141, Cb 59,
        // Cr 189.
        let (y, rest) = bytes.split_at(64 * 32);
        let (u, v) = rest.split_at(rest.len() / 2);
        assert!(y.iter().all(|&s| s.abs_diff(141) <= 3), "luma");
        assert!(u.iter().all(|&s| s.abs_diff(59) <= 3), "cb");
        assert!(v.iter().all(|&s| s.abs_diff(189) <= 3), "cr");
    }
}

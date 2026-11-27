//! Tests that need the GPU, and the declines that need none.
//!
//! A test that needs a backend this machine lacks prints "skipped:" and
//! passes, unless `CUTLASS_REQUIRE_NVIDIA` (or `CUTLASS_REQUIRE_VAAPI`) is
//! set, so a run on the test machine cannot pass by skipping.

use cutlass_decode::{
    CodecError, CodecId, CodecParams, DecodeFactory, Decoder, Output, Packet, PictureOutput,
    PictureSamples, PixelFormat, SideData, nal,
};
use cutlass_encode::{CpuFrame, EncodeFactory, Encoder, FrameInput, FrameTime, PlatformFrame};

use super::{cuda, nvdec, nvenc};
use crate::video::fallback::rust_decoder;

const REQUIRE_NVIDIA: &str = "CUTLASS_REQUIRE_NVIDIA";

/// Whether the NVIDIA tests can run here. A missing GPU fails when
/// [`REQUIRE_NVIDIA`] is set.
fn nvidia(what: &str) -> bool {
    let present = cuda::device().is_some() && nvdec::nvcuvid().is_some() && nvenc::api().is_some();
    if !present {
        assert!(
            std::env::var_os(REQUIRE_NVIDIA).is_none(),
            "{REQUIRE_NVIDIA} is set but there is no NVIDIA GPU with NVDEC and NVENC"
        );
        eprintln!("skipped: {what}: no NVIDIA GPU");
    }
    present
}

/// A planar 4:2:0 picture with a gradient that moves with `index`.
fn picture(width: u32, height: u32, index: u32) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let mut bytes = Vec::with_capacity(w * h * 3 / 2);
    for y in 0..h {
        for x in 0..w {
            bytes.push((16 + (x * 2 + y + index as usize * 3) % 200) as u8);
        }
    }
    for plane in 0..2 {
        for y in 0..h / 2 {
            for x in 0..w / 2 {
                let along = if plane == 0 { x * 2 } else { y * 2 };
                let size = if plane == 0 { w } else { h };
                bytes.push((64 + along * 128 / size.max(1)) as u8);
            }
        }
    }
    bytes
}

fn frame(width: u32, height: u32, index: u32) -> FrameInput {
    FrameInput::Cpu(
        CpuFrame::new(
            width,
            height,
            PixelFormat::Yuv420p,
            picture(width, height, index),
            FrameTime::new(i64::from(index), 1, 30).expect("time"),
        )
        .expect("frame"),
    )
}

fn encoder_params(codec: CodecId, width: u32, height: u32) -> CodecParams {
    CodecParams::with_format(codec, width, height, PixelFormat::Yuv420p)
}

fn encode(encoder: &mut dyn Encoder, width: u32, height: u32, frames: u32) -> Vec<Packet> {
    let mut packets = Vec::new();
    for index in 0..frames {
        packets.extend(
            encoder
                .push_frame(&frame(width, height, index))
                .expect("encode"),
        );
    }
    packets.extend(encoder.drain().expect("drain"));
    packets
}

fn config(packets: &[Packet]) -> Vec<u8> {
    packets
        .iter()
        .find_map(|packet| {
            packet.side.iter().find_map(|side| match side {
                SideData::Config(bytes) if !bytes.is_empty() => Some(bytes.clone()),
                _ => None,
            })
        })
        .unwrap_or_default()
}

fn decode_all(decoder: &mut dyn Decoder, packets: &[Packet]) -> Vec<PictureOutput> {
    let mut outputs = Vec::new();
    for packet in packets {
        outputs.extend(decoder.push(packet).expect("decode"));
    }
    outputs.extend(decoder.drain().expect("drain"));
    outputs
        .into_iter()
        .map(|output| match output {
            Output::Picture(picture) => picture,
            Output::Audio(_) => panic!("picture"),
        })
        .collect()
}

fn samples(picture: &PictureOutput) -> &[u8] {
    match picture.samples() {
        PictureSamples::Cpu(bytes) => bytes,
        _ => panic!("cpu samples"),
    }
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let error: u64 = a
        .iter()
        .zip(b)
        .map(|(a, b)| u64::from(a.abs_diff(*b)).pow(2))
        .sum();
    if error == 0 {
        return f64::INFINITY;
    }
    let mse = error as f64 / a.len() as f64;
    10.0 * (255.0 * 255.0 / mse).log10()
}

fn decoder_params(codec: CodecId, width: u32, height: u32, extra: Vec<u8>) -> CodecParams {
    let mut params = CodecParams::video(codec, width, height);
    params.extra = extra;
    params
}

/// The sets a configuration carries, as NAL unit types.
fn config_sets(codec: CodecId, config: &[u8]) -> Vec<u8> {
    match codec {
        CodecId::H264 => {
            let avc = nal::avc_config(config).expect("avcC parses");
            avc.sps
                .iter()
                .chain(&avc.pps)
                .filter_map(|set| nal::h264_type(set))
                .collect()
        }
        _ => nal::hevc_config(config)
            .expect("hvcC parses")
            .nals
            .iter()
            .filter_map(|set| nal::hevc_type(set))
            .collect(),
    }
}

#[test]
fn nvenc_round_trips_through_the_rust_decoder() {
    if !nvidia("nvenc round trip") {
        return;
    }
    for codec in [CodecId::H264, CodecId::Hevc] {
        for (width, height) in [(320, 240), (1920, 1080)] {
            let mut encoder =
                nvenc::open_encoder(&encoder_params(codec, width, height)).expect("nvenc opens");
            let packets = encode(encoder.as_mut(), width, height, 40);
            assert_eq!(packets.len(), 40, "{codec:?} {width}x{height}");
            let keyframes: Vec<usize> = packets
                .iter()
                .enumerate()
                .filter(|(_, packet)| packet.keyframe)
                .map(|(index, _)| index)
                .collect();
            assert_eq!(keyframes, [0, 30], "{codec:?}: an IDR every 30 pictures");
            let extra = config(&packets);
            let sets = config_sets(codec, &extra);
            let wanted: &[u8] = if codec == CodecId::H264 {
                &[7, 8]
            } else {
                &[32, 33, 34]
            };
            assert_eq!(sets, wanted, "{codec:?} configuration");
            let mut decoder =
                rust_decoder(&decoder_params(codec, width, height, extra)).expect("rust decoder");
            let pictures = decode_all(decoder.as_mut(), &packets);
            assert_eq!(pictures.len(), 40);
            for (index, picture) in pictures.iter().enumerate() {
                assert_eq!(picture.pts(), index as i64);
                let original = picture_bytes(width, height, index);
                let quality = psnr(samples(picture), &original);
                assert!(
                    quality >= 35.0,
                    "{codec:?} {width}x{height} picture {index}: {quality:.1} dB"
                );
            }
            eprintln!("{codec:?} {width}x{height}: NVENC round trip ok");
        }
    }
}

fn picture_bytes(width: u32, height: u32, index: usize) -> Vec<u8> {
    picture(width, height, u32::try_from(index).expect("index"))
}

#[test]
fn nvenc_reset_starts_again_on_a_keyframe_with_the_config() {
    if !nvidia("nvenc reset") {
        return;
    }
    for codec in [CodecId::H264, CodecId::Hevc] {
        let mut encoder =
            nvenc::open_encoder(&encoder_params(codec, 320, 240)).expect("nvenc opens");
        let first = encode(encoder.as_mut(), 320, 240, 3);
        assert!(!config(&first).is_empty());
        encoder.reset();
        let again = encode(encoder.as_mut(), 320, 240, 3);
        assert_eq!(again.len(), 3);
        assert!(again[0].keyframe, "{codec:?}");
        assert!(
            again[0]
                .side
                .iter()
                .any(|side| matches!(side, SideData::Config(_))),
            "{codec:?}: the config comes again"
        );
        assert_eq!(again[0].pts, 0);
    }
}

#[test]
fn nvenc_takes_rgba_and_an_ntsc_rate() {
    if !nvidia("nvenc rgba") {
        return;
    }
    let (width, height) = (640, 360);
    let mut params = encoder_params(CodecId::H264, width, height);
    params.pixel = PixelFormat::Rgba8;
    params.constant_rate = true;
    params.bitrate = 4_000_000;
    let mut encoder = nvenc::open_encoder(&params).expect("nvenc opens");
    let mut packets = Vec::new();
    for index in 0..10 {
        let frame = CpuFrame::new(
            width,
            height,
            PixelFormat::Rgba8,
            [30, 200, 60, 255].repeat((width * height) as usize),
            FrameTime::new(index * 1001, 1001, 30_000).expect("time"),
        )
        .expect("frame");
        packets.extend(encoder.push_frame(&FrameInput::Cpu(frame)).expect("encode"));
    }
    packets.extend(encoder.drain().expect("drain"));
    assert_eq!(packets.len(), 10);
    assert_eq!(packets[9].pts, 9 * 1001);
    assert_eq!(packets[9].timescale, 30_000);
}

/// The Rust encoder's stream, decoded on NVDEC and by the Rust decoder,
/// gives the same bytes.
#[test]
fn nvdec_matches_the_rust_decoder() {
    if !nvidia("nvdec against rust") {
        return;
    }
    let mut cases = Vec::new();
    let mut rust = EncodeFactory::new();
    cutlass_h264::register_encoders(&mut rust);
    cutlass_av1::register_encoders(&mut rust);
    for (codec, size) in [(CodecId::H264, (320, 240)), (CodecId::Av1, (320, 240))] {
        let (width, height) = size;
        let mut encoder = rust
            .open(&encoder_params(codec, width, height))
            .expect("rust encoder");
        cases.push((codec, size, encode(encoder.as_mut(), width, height, 12)));
    }
    let mut encoder =
        nvenc::open_encoder(&encoder_params(CodecId::Hevc, 320, 240)).expect("nvenc hevc");
    cases.push((
        CodecId::Hevc,
        (320, 240),
        encode(encoder.as_mut(), 320, 240, 12),
    ));

    for (codec, (width, height), packets) in cases {
        let params = decoder_params(codec, width, height, config(&packets));
        let mut hardware = match nvdec::open(&params) {
            Ok(decoder) => decoder,
            Err(CodecError::Unsupported { what }) => {
                eprintln!("skipped: NVDEC {codec:?} on this GPU: {what}");
                continue;
            }
            Err(error) => panic!("{codec:?}: {error}"),
        };
        let mut software = rust_decoder(&params).expect("rust decoder");
        let ours = decode_all(&mut hardware, &packets);
        let theirs = decode_all(software.as_mut(), &packets);
        assert!(hardware.hardware(), "{codec:?} stayed on NVDEC");
        assert_eq!(ours.len(), packets.len(), "{codec:?}");
        assert_eq!(ours.len(), theirs.len(), "{codec:?}");
        for (index, (ours, theirs)) in ours.iter().zip(&theirs).enumerate() {
            assert_eq!(ours.pts(), theirs.pts(), "{codec:?} {index}");
            let (a, b) = (ours.info(), theirs.info());
            assert_eq!((a.width, a.height), (b.width, b.height), "{codec:?}");
            assert!(
                samples(ours) == samples(theirs),
                "{codec:?} picture {index} differs from the Rust decoder"
            );
        }
        eprintln!("{codec:?}: NVDEC matches the Rust decoder");
    }
}

/// Annex B with the parameter sets in band, and no configuration, decodes.
#[test]
fn nvdec_takes_annex_b_without_a_config() {
    if !nvidia("nvdec annex b") {
        return;
    }
    let mut encoder = nvenc::open_encoder(&encoder_params(CodecId::H264, 320, 240)).expect("nvenc");
    let packets = encode(encoder.as_mut(), 320, 240, 5);
    let sets = nal::avc_config(&config(&packets)).expect("avcC");
    let annex_b: Vec<Packet> = packets
        .iter()
        .enumerate()
        .map(|(index, packet)| {
            let mut units = Vec::new();
            if index == 0 {
                for set in sets.sps.iter().chain(&sets.pps) {
                    units.push(nal::Nal { bytes: set.clone() });
                }
            }
            units.extend(nal::split_lengths(&packet.data, 4).expect("lengths"));
            Packet {
                data: nal::to_annex_b(&units),
                side: Vec::new(),
                ..packet.clone()
            }
        })
        .collect();
    let mut decoder =
        nvdec::open(&decoder_params(CodecId::H264, 320, 240, Vec::new())).expect("nvdec opens");
    let pictures = decode_all(&mut decoder, &annex_b);
    assert_eq!(pictures.len(), 5);
    assert!(decoder.hardware());
}

/// A stream smaller than NVDEC takes still decodes, on the Rust decoder.
#[test]
fn a_picture_nvdec_refuses_falls_back_to_rust() {
    if !nvidia("nvdec fallback") {
        return;
    }
    let (width, height) = (128, 72);
    let Ok(mut encoder) = nvenc::open_encoder(&encoder_params(CodecId::Hevc, width, height)) else {
        eprintln!("skipped: NVENC does not encode HEVC at {width}x{height}");
        return;
    };
    let packets = encode(encoder.as_mut(), width, height, 4);
    let params = decoder_params(CodecId::Hevc, width, height, config(&packets));
    let mut decoder = nvdec::open(&params).expect("nvdec opens");
    let pictures = decode_all(&mut decoder, &packets);
    let theirs = decode_all(rust_decoder(&params).expect("rust").as_mut(), &packets);
    assert_eq!(pictures.len(), 4);
    for (ours, theirs) in pictures.iter().zip(&theirs) {
        assert!(samples(ours) == samples(theirs));
    }
    eprintln!(
        "HEVC {width}x{height}: decoded on {}",
        if decoder.hardware() {
            "NVDEC"
        } else {
            "the Rust decoder"
        }
    );
}

#[test]
fn nvdec_decodes_again_after_drain_and_reset() {
    if !nvidia("nvdec drain and reset") {
        return;
    }
    let mut encoder = nvenc::open_encoder(&encoder_params(CodecId::H264, 320, 240)).expect("nvenc");
    let packets = encode(encoder.as_mut(), 320, 240, 6);
    let params = decoder_params(CodecId::H264, 320, 240, config(&packets));
    let mut decoder = nvdec::open(&params).expect("nvdec opens");
    assert_eq!(decode_all(&mut decoder, &packets).len(), 6);
    assert_eq!(decode_all(&mut decoder, &packets).len(), 6, "after drain");
    decoder.push(&packets[0]).expect("push");
    decoder.reset();
    assert_eq!(decode_all(&mut decoder, &packets).len(), 6, "after reset");
    assert!(decoder.hardware());
}

#[test]
fn nvenc_declines_what_it_does_not_encode() {
    let decline = |params: &CodecParams| {
        matches!(
            nvenc::open_encoder(params),
            Err(CodecError::Unsupported { .. })
        )
    };
    assert!(decline(&encoder_params(CodecId::Av1, 320, 240)));
    assert!(decline(&encoder_params(CodecId::H264, 321, 240)));
    assert!(decline(&encoder_params(CodecId::H264, 9000, 240)));
    let mut ten_bit = encoder_params(CodecId::Hevc, 320, 240);
    ten_bit.pixel = PixelFormat::Yuv420p10;
    assert!(decline(&ten_bit));
}

#[test]
fn nvenc_refuses_a_platform_frame_and_a_wrong_size() {
    if !nvidia("nvenc frame checks") {
        return;
    }
    let mut encoder = nvenc::open_encoder(&encoder_params(CodecId::H264, 320, 240)).expect("nvenc");
    let time = FrameTime::new(0, 1, 30).expect("time");
    let platform = FrameInput::Platform(PlatformFrame::new(320, 240, 1, time).expect("frame"));
    assert!(matches!(
        encoder.push_frame(&platform),
        Err(CodecError::Unsupported { .. })
    ));
    assert!(matches!(
        encoder.push_frame(&frame(160, 120, 0)),
        Err(CodecError::Unsupported { .. })
    ));
}

/// With no GPU, the factories reach the Rust codecs; HEVC encode, which
/// has no Rust encoder, is the one thing left unsupported.
#[test]
fn the_factories_fall_through_to_rust() {
    let mut decoders = DecodeFactory::new();
    super::register_decoders(&mut decoders);
    cutlass_h264::register_decoders(&mut decoders);
    cutlass_hevc::register_decoders(&mut decoders);
    cutlass_av1::register_decoders(&mut decoders);
    for codec in [CodecId::H264, CodecId::Hevc, CodecId::Av1] {
        decoders
            .open(&CodecParams::video(codec, 320, 240))
            .expect("a decoder opens");
    }
    let mut encoders = EncodeFactory::new();
    super::register_encoders(&mut encoders);
    cutlass_h264::register_encoders(&mut encoders);
    encoders
        .open(&encoder_params(CodecId::H264, 320, 240))
        .expect("an H.264 encoder opens");
    let hevc = encoders.open(&encoder_params(CodecId::Hevc, 320, 240));
    if cuda::device().is_none() {
        assert!(matches!(hevc, Err(CodecError::Unsupported { .. })));
    }
}

#[test]
fn registering_with_video_off_registers_nothing() {
    let mut decoders = DecodeFactory::new();
    super::register_decoders_with(&mut decoders, false);
    assert!(matches!(
        decoders.open(&CodecParams::video(CodecId::H264, 320, 240)),
        Err(CodecError::Unsupported { .. })
    ));
    let mut encoders = EncodeFactory::new();
    super::register_encoders_with(&mut encoders, false);
    assert!(matches!(
        encoders.open(&encoder_params(CodecId::Hevc, 320, 240)),
        Err(CodecError::Unsupported { .. })
    ));
}

#[test]
fn the_probe_says_something_for_each_backend() {
    let lines = super::probe();
    assert!(lines[0].starts_with(crate::BACKEND_VAR));
    assert!(lines.iter().any(|line| line.starts_with("NVIDIA")));
    for line in &lines {
        eprintln!("{line}");
    }
}

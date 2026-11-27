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

use super::{cuda, nvdec, nvenc, va};
use crate::video::fallback::rust_decoder;

const REQUIRE_NVIDIA: &str = "CUTLASS_REQUIRE_NVIDIA";
const REQUIRE_VAAPI: &str = "CUTLASS_REQUIRE_VAAPI";

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
    if cuda::device().is_none() && !va::hevc_enc::available() {
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
    assert!(lines.iter().any(|line| line.starts_with("VA-API")));
    for line in &lines {
        eprintln!("{line}");
    }
}

/// Whether the VA-API tests can run here. A missing render node fails
/// when [`REQUIRE_VAAPI`] is set.
fn vaapi(what: &str) -> bool {
    let present = va::hevc::available();
    if !present {
        assert!(
            std::env::var_os(REQUIRE_VAAPI).is_none(),
            "{REQUIRE_VAAPI} is set but no render node decodes HEVC"
        );
        eprintln!("skipped: {what}: no VA-API render node decodes HEVC");
    }
    present
}

/// An Annex B HEVC stream split into access units (clause 7.4.2.4.4), as
/// packets 1/30 s apart.
pub(super) fn hevc_access_units(data: &[u8]) -> Vec<Packet> {
    let mut units: Vec<Vec<nal::Nal>> = Vec::new();
    let mut current: Vec<nal::Nal> = Vec::new();
    let mut has_slice = false;
    for unit in nal::annex_b(data).expect("annex b") {
        let bytes = &unit.bytes;
        if bytes.len() < 3 {
            continue;
        }
        let kind = (bytes[0] >> 1) & 0x3f;
        let slice = kind <= 21;
        let starts = if slice {
            bytes[2] & 0x80 != 0
        } else {
            matches!(kind, 32..=35 | 39 | 41..=44 | 48..=55)
        };
        if starts && has_slice {
            units.push(std::mem::take(&mut current));
            has_slice = false;
        }
        has_slice |= slice;
        current.push(unit);
    }
    if !current.is_empty() {
        units.push(current);
    }
    units
        .iter()
        .enumerate()
        .map(|(index, unit)| {
            let keyframe = unit
                .iter()
                .any(|nal| matches!(nal::hevc_type(&nal.bytes), Some(16..=23)));
            Packet::video(nal::to_annex_b(unit), index as i64, 1, 30, keyframe)
        })
        .collect()
}

/// Whether ffmpeg with `encoder` (libx264, libx265) is here.
fn has_encoder(encoder: &str) -> bool {
    std::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output()
        .is_ok_and(|out| String::from_utf8_lossy(&out.stdout).contains(encoder))
}

/// A raw stream from ffmpeg's `encoder` of a moving test pattern, or
/// `None` when ffmpeg is missing or fails.
fn ffmpeg_encode(
    encoder: &str,
    size: (u32, u32),
    frames: u32,
    pix: &str,
    params: &str,
    extra: &[&str],
) -> Option<Vec<u8>> {
    if !has_encoder(encoder) {
        return None;
    }
    let (format, flag) = if encoder == "libx265" {
        ("hevc", "-x265-params")
    } else {
        ("h264", "-x264-params")
    };
    let path = std::env::temp_dir().join(format!(
        "cutlass-vaapi-{}-{encoder}-{}x{}-{pix}-{}.{format}",
        std::process::id(),
        size.0,
        size.1,
        params.len()
    ));
    let source = format!("testsrc2=size={}x{}:rate=30", size.0, size.1);
    let encoded = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i", &source])
        .args(["-frames:v", &frames.to_string(), "-pix_fmt", pix])
        .args(["-c:v", encoder, flag, params])
        .args(extra)
        .args(["-f", format])
        .arg(&path)
        .output()
        .ok()?;
    let bytes = std::fs::read(&path).ok();
    let _ = std::fs::remove_file(&path);
    if !encoded.status.success() {
        eprintln!(
            "skipped: {encoder} could not encode {params}: {}",
            String::from_utf8_lossy(&encoded.stderr).trim()
        );
        return None;
    }
    bytes
}

/// An HEVC stream from libx265, which runs with a thread pool: some builds
/// crash encoding slices without one.
fn x265(size: (u32, u32), frames: u32, pix: &str, params: &str) -> Option<Vec<u8>> {
    ffmpeg_encode(
        "libx265",
        size,
        frames,
        pix,
        &format!("pools=4:{params}"),
        &[],
    )
}

/// An H.264 stream of `profile` from libx264.
fn x264(size: (u32, u32), frames: u32, profile: &str, params: &str) -> Option<Vec<u8>> {
    ffmpeg_encode(
        "libx264",
        size,
        frames,
        "yuv420p",
        params,
        &["-profile:v", profile],
    )
}

/// An Annex B H.264 stream split into access units (clause 7.4.1.2.3), as
/// packets 1/30 s apart.
pub(super) fn h264_access_units(data: &[u8]) -> Vec<Packet> {
    let mut units: Vec<Vec<nal::Nal>> = Vec::new();
    let mut current: Vec<nal::Nal> = Vec::new();
    let mut has_slice = false;
    for unit in nal::annex_b(data).expect("annex b") {
        let bytes = &unit.bytes;
        if bytes.len() < 2 {
            continue;
        }
        let kind = bytes[0] & 0x1f;
        let slice = matches!(kind, 1 | 5);
        // first_mb_in_slice is ue(v): 0 codes as a single 1 bit.
        let starts = if slice {
            bytes[1] & 0x80 != 0
        } else {
            matches!(kind, 6..=9 | 14..=18)
        };
        if starts && has_slice {
            units.push(std::mem::take(&mut current));
            has_slice = false;
        }
        has_slice |= slice;
        current.push(unit);
    }
    if !current.is_empty() {
        units.push(current);
    }
    units
        .iter()
        .enumerate()
        .map(|(index, unit)| {
            let keyframe = unit.iter().any(|nal| nal::h264_type(&nal.bytes) == Some(5));
            Packet::video(nal::to_annex_b(unit), index as i64, 1, 30, keyframe)
        })
        .collect()
}

/// The H.264 streams the VA-API tests decode, with the parameters to open
/// them: the Rust encoder's, and x264 streams in each profile with B
/// frames, explicit and implicit weights, slices, scaling matrices and a
/// cropped 1080p when ffmpeg is here.
pub(super) fn h264_streams() -> Vec<(String, Vec<Packet>, CodecParams)> {
    let mut rust = EncodeFactory::new();
    cutlass_h264::register_encoders(&mut rust);
    let mut encoder = rust
        .open(&encoder_params(CodecId::H264, 320, 240))
        .expect("rust encoder");
    let packets = encode(encoder.as_mut(), 320, 240, 12);
    let params = decoder_params(CodecId::H264, 320, 240, config(&packets));
    let mut streams = vec![(String::from("rust encoder 320x240"), packets, params)];
    if !has_encoder("libx264") {
        eprintln!("skipped: x264 streams: no ffmpeg with libx264");
        return streams;
    }
    let cases = [
        ((640, 360), 20, "baseline", "keyint=10:ref=2"),
        (
            (640, 360),
            20,
            "main",
            "bframes=2:ref=3:keyint=10:weightp=1",
        ),
        (
            (1280, 720),
            30,
            "high",
            "bframes=3:ref=4:b-pyramid=normal:weightp=2:keyint=30",
        ),
        (
            (1280, 720),
            12,
            "high",
            "bframes=2:weightb=1:direct=temporal:keyint=12",
        ),
        (
            (1280, 720),
            12,
            "high",
            "slices=4:8x8dct=1:cqm=jvt:keyint=12",
        ),
        ((1920, 1080), 12, "high", "bframes=3:keyint=60:open-gop=1"),
    ];
    for (size, frames, profile, params) in cases {
        if let Some(bytes) = x264(size, frames, profile, params) {
            let mut decode = CodecParams::video(CodecId::H264, 0, 0);
            decode.nal_length_size = 0;
            streams.push((
                format!("x264 {}x{} {profile} {params}", size.0, size.1),
                h264_access_units(&bytes),
                decode,
            ));
        }
    }
    streams
}

/// The HEVC streams the VA-API tests decode: the crate fixture, and x265
/// streams with B-frames, weighted prediction, 10 bits, slices with
/// wavefronts, scaling lists and a cropped 1080p when ffmpeg is here.
pub(super) fn hevc_streams() -> Vec<(String, Vec<u8>, bool)> {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../codecs/hevc/tests/fixtures/testsrc-64x48.hevc");
    // The fixture is smaller than some GPUs decode; it may fall back.
    let mut streams = vec![(
        String::from("fixture 64x48"),
        std::fs::read(fixture).expect("fixture"),
        false,
    )];
    let cases = [
        (
            (1280, 720),
            30,
            "yuv420p",
            "bframes=3:ref=4:keyint=30:weightp=1:weightb=1",
        ),
        ((1280, 720), 20, "yuv420p10le", "bframes=2:keyint=20"),
        (
            (1280, 720),
            12,
            "yuv420p",
            "bframes=0:keyint=12:slices=4:wpp=1",
        ),
        (
            (640, 360),
            12,
            "yuv420p",
            "bframes=2:keyint=6:scaling-list=default",
        ),
        (
            (1920, 1080),
            12,
            "yuv420p",
            "bframes=3:keyint=60:open-gop=1",
        ),
    ];
    if !has_encoder("libx265") {
        eprintln!("skipped: x265 streams: no ffmpeg with libx265");
        return streams;
    }
    for (size, frames, pix, params) in cases {
        if let Some(bytes) = x265(size, frames, pix, params) {
            streams.push((
                format!("x265 {}x{} {pix} {params}", size.0, size.1),
                bytes,
                true,
            ));
        }
    }
    streams
}

#[test]
fn vaapi_hevc_matches_the_rust_decoder() {
    if !vaapi("vaapi hevc against rust") {
        return;
    }
    for (name, bytes, must_decode) in hevc_streams() {
        let packets = hevc_access_units(&bytes);
        let mut params = CodecParams::video(CodecId::Hevc, 0, 0);
        params.nal_length_size = 0;
        let mut hardware = va::hevc::open(&params).expect("va-api opens");
        let mut software = rust_decoder(&params).expect("rust decoder");
        let ours = decode_all(&mut hardware, &packets);
        let theirs = decode_all(software.as_mut(), &packets);
        if !hardware.hardware() {
            assert!(!must_decode, "{name}: VA-API handed the stream to Rust");
            eprintln!("{name}: VA-API refused it; the Rust decoder took it");
            continue;
        }
        assert_eq!(ours.len(), theirs.len(), "{name}");
        assert!(!ours.is_empty(), "{name}");
        for (index, (ours, theirs)) in ours.iter().zip(&theirs).enumerate() {
            assert_eq!(ours.pts(), theirs.pts(), "{name} {index}");
            let (a, b) = (ours.info(), theirs.info());
            assert_eq!((a.width, a.height), (b.width, b.height), "{name}");
            assert!(
                samples(ours) == samples(theirs),
                "{name}: picture {index} differs from the Rust decoder (PSNR {:.1} dB)",
                psnr(samples(ours), samples(theirs))
            );
        }
        eprintln!("{name}: VA-API matches the Rust decoder");
    }
}

#[test]
fn vaapi_hevc_decodes_again_after_drain_and_reset() {
    if !vaapi("vaapi hevc drain and reset") {
        return;
    }
    let bytes = x265((640, 360), 8, "yuv420p", "bframes=2:keyint=8").unwrap_or_else(|| {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../codecs/hevc/tests/fixtures/testsrc-64x48.hevc");
        std::fs::read(fixture).expect("fixture")
    });
    let packets = hevc_access_units(&bytes);
    let mut params = CodecParams::video(CodecId::Hevc, 0, 0);
    params.nal_length_size = 0;
    let mut decoder = va::hevc::open(&params).expect("va-api opens");
    let first = decode_all(&mut decoder, &packets).len();
    assert!(first > 0);
    assert_eq!(
        decode_all(&mut decoder, &packets).len(),
        first,
        "after drain"
    );
    decoder.push(&packets[0]).expect("push");
    decoder.reset();
    assert_eq!(
        decode_all(&mut decoder, &packets).len(),
        first,
        "after reset"
    );
}

/// With libva missing or no render node, the VA-API opener declines.
#[test]
fn vaapi_declines_without_a_render_node() {
    let opened = va::hevc::open(&CodecParams::video(CodecId::Hevc, 320, 240));
    if !va::hevc::available() {
        assert!(matches!(opened, Err(CodecError::Unsupported { .. })));
    }
    assert!(matches!(
        va::hevc::open(&CodecParams::video(CodecId::H264, 320, 240)),
        Err(CodecError::Unsupported { .. })
    ));
    let opened = va::h264::open(&CodecParams::video(CodecId::H264, 320, 240));
    if !va::h264::available() {
        assert!(matches!(opened, Err(CodecError::Unsupported { .. })));
    }
}

/// Whether the VA-API H.264 tests can run here.
fn vaapi_h264(what: &str) -> bool {
    let present = va::h264::available();
    if !present {
        assert!(
            std::env::var_os(REQUIRE_VAAPI).is_none(),
            "{REQUIRE_VAAPI} is set but no render node decodes H.264"
        );
        eprintln!("skipped: {what}: no VA-API render node decodes H.264");
    }
    present
}

#[test]
fn vaapi_h264_matches_the_rust_decoder() {
    if !vaapi_h264("vaapi h264 against rust") {
        return;
    }
    for (name, packets, params) in h264_streams() {
        let mut hardware = va::h264::open(&params).expect("va-api opens");
        let mut software = rust_decoder(&params).expect("rust decoder");
        let ours = decode_all(&mut hardware, &packets);
        let theirs = decode_all(software.as_mut(), &packets);
        assert!(
            hardware.hardware(),
            "{name}: VA-API handed the stream to Rust"
        );
        assert_eq!(ours.len(), theirs.len(), "{name}");
        assert!(!ours.is_empty(), "{name}");
        for (index, (ours, theirs)) in ours.iter().zip(&theirs).enumerate() {
            assert_eq!(ours.pts(), theirs.pts(), "{name} {index}");
            let (a, b) = (ours.info(), theirs.info());
            assert_eq!((a.width, a.height), (b.width, b.height), "{name}");
            assert!(
                samples(ours) == samples(theirs),
                "{name}: picture {index} differs from the Rust decoder (PSNR {:.1} dB)",
                psnr(samples(ours), samples(theirs))
            );
        }
        eprintln!("{name}: VA-API matches the Rust decoder");
    }
}

#[test]
fn vaapi_h264_decodes_again_after_drain_and_reset() {
    if !vaapi_h264("vaapi h264 drain and reset") {
        return;
    }
    let (_, packets, params) = h264_streams().remove(0);
    let mut decoder = va::h264::open(&params).expect("va-api opens");
    assert_eq!(decode_all(&mut decoder, &packets).len(), packets.len());
    assert_eq!(
        decode_all(&mut decoder, &packets).len(),
        packets.len(),
        "after drain"
    );
    decoder.push(&packets[0]).expect("push");
    decoder.reset();
    assert_eq!(
        decode_all(&mut decoder, &packets).len(),
        packets.len(),
        "after reset"
    );
    assert!(decoder.hardware());
}

/// An interlaced stream is the Rust decoder's: the front end refuses it at
/// its first SPS and the fallback replays it.
#[test]
fn an_interlaced_stream_falls_back_to_rust() {
    if !vaapi_h264("vaapi h264 interlaced fallback") {
        return;
    }
    let Some(bytes) = x264((640, 360), 6, "high", "interlaced=1:bframes=0") else {
        eprintln!("skipped: no interlaced x264 stream");
        return;
    };
    let packets = h264_access_units(&bytes);
    let mut params = CodecParams::video(CodecId::H264, 0, 0);
    params.nal_length_size = 0;
    let mut decoder = va::h264::open(&params).expect("va-api opens");
    let pictures = decode_all(&mut decoder, &packets);
    assert!(!decoder.hardware());
    assert!(!pictures.is_empty());
}

/// Whether VA-API encodes `codec` here. A missing encoder fails when
/// [`REQUIRE_VAAPI`] is set.
fn vaapi_encode(codec: CodecId, what: &str) -> bool {
    let present = match codec {
        CodecId::H264 => va::h264_enc::available(),
        _ => va::hevc_enc::available(),
    };
    if !present {
        assert!(
            std::env::var_os(REQUIRE_VAAPI).is_none(),
            "{REQUIRE_VAAPI} is set but no render node encodes {codec:?}"
        );
        eprintln!("skipped: {what}: no VA-API render node encodes {codec:?}");
    }
    present
}

fn va_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    match params.codec {
        CodecId::H264 => va::h264_enc::open_encoder(params),
        _ => va::hevc_enc::open_encoder(params),
    }
}

#[test]
fn vaapi_encode_round_trips_through_the_rust_decoder() {
    for codec in [CodecId::H264, CodecId::Hevc] {
        if !vaapi_encode(codec, "vaapi encode round trip") {
            continue;
        }
        for (width, height) in [(320, 240), (1920, 1080), (642, 362)] {
            let mut encoder =
                va_encoder(&encoder_params(codec, width, height)).expect("va-api opens");
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
                assert_eq!(
                    (picture.info().width, picture.info().height),
                    (width, height)
                );
                let original = picture_bytes(width, height, index);
                let quality = psnr(samples(picture), &original);
                assert!(
                    quality >= 35.0,
                    "{codec:?} {width}x{height} picture {index}: {quality:.1} dB"
                );
            }
            eprintln!("{codec:?} {width}x{height}: VA-API encode round trip ok");
        }
    }
}

#[test]
fn vaapi_encode_reset_starts_again_on_a_keyframe_with_the_config() {
    for codec in [CodecId::H264, CodecId::Hevc] {
        if !vaapi_encode(codec, "vaapi encode reset") {
            continue;
        }
        let mut encoder = va_encoder(&encoder_params(codec, 320, 240)).expect("va-api opens");
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
        let mut decoder =
            rust_decoder(&decoder_params(codec, 320, 240, config(&again))).expect("rust decoder");
        assert_eq!(decode_all(decoder.as_mut(), &again).len(), 3);
    }
}

#[test]
fn vaapi_encode_takes_rgba_an_ntsc_rate_and_a_constant_rate() {
    for codec in [CodecId::H264, CodecId::Hevc] {
        if !vaapi_encode(codec, "vaapi encode rgba") {
            continue;
        }
        let (width, height) = (640, 360);
        let mut params = encoder_params(codec, width, height);
        params.pixel = PixelFormat::Rgba8;
        params.constant_rate = true;
        params.bitrate = 4_000_000;
        let mut encoder = va_encoder(&params).expect("va-api opens");
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
        assert_eq!(packets.len(), 10, "{codec:?}");
        assert_eq!(packets[9].pts, 9 * 1001);
        assert_eq!(packets[9].timescale, 30_000);
        let mut decoder = rust_decoder(&decoder_params(codec, width, height, config(&packets)))
            .expect("rust decoder");
        let pictures = decode_all(decoder.as_mut(), &packets);
        assert_eq!(pictures.len(), 10, "{codec:?}");
    }
}

#[test]
fn vaapi_encode_declines_what_it_does_not_encode() {
    let decline =
        |params: &CodecParams| matches!(va_encoder(params), Err(CodecError::Unsupported { .. }));
    assert!(decline(&encoder_params(CodecId::Av1, 320, 240)));
    for codec in [CodecId::H264, CodecId::Hevc] {
        assert!(decline(&encoder_params(codec, 321, 240)));
        assert!(decline(&encoder_params(codec, 9000, 240)));
        let mut ten_bit = encoder_params(codec, 320, 240);
        ten_bit.pixel = PixelFormat::Yuv420p10;
        assert!(decline(&ten_bit));
    }
    assert!(matches!(
        va::h264_enc::open_encoder(&encoder_params(CodecId::Hevc, 320, 240)),
        Err(CodecError::Unsupported { .. })
    ));
    assert!(matches!(
        va::hevc_enc::open_encoder(&encoder_params(CodecId::H264, 320, 240)),
        Err(CodecError::Unsupported { .. })
    ));
    if !va::h264_enc::available() {
        assert!(decline(&encoder_params(CodecId::H264, 320, 240)));
    }
    if !va::hevc_enc::available() {
        assert!(decline(&encoder_params(CodecId::Hevc, 320, 240)));
    }
}

#[test]
fn vaapi_encode_refuses_a_platform_frame_and_a_wrong_size() {
    for codec in [CodecId::H264, CodecId::Hevc] {
        if !vaapi_encode(codec, "vaapi encode frame checks") {
            continue;
        }
        let mut encoder = va_encoder(&encoder_params(codec, 320, 240)).expect("va-api opens");
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
}

/// VA-API's H.264 stream, decoded again on VA-API, matches the Rust
/// decoder's pictures of it.
#[test]
fn vaapi_decodes_what_vaapi_encodes() {
    if !vaapi_encode(CodecId::H264, "vaapi encode then decode") || !vaapi_h264("vaapi decode") {
        return;
    }
    let (width, height) = (640, 360);
    let mut encoder =
        va_encoder(&encoder_params(CodecId::H264, width, height)).expect("va-api opens");
    let packets = encode(encoder.as_mut(), width, height, 35);
    let params = decoder_params(CodecId::H264, width, height, config(&packets));
    let mut hardware = va::h264::open(&params).expect("va-api decoder");
    let mut software = rust_decoder(&params).expect("rust decoder");
    let ours = decode_all(&mut hardware, &packets);
    let theirs = decode_all(software.as_mut(), &packets);
    assert_eq!(ours.len(), theirs.len());
    for (index, (a, b)) in ours.iter().zip(&theirs).enumerate() {
        assert_eq!(a.pts(), b.pts());
        assert_eq!(samples(a), samples(b), "picture {index}");
    }
}

//! MediaCodec on the device: these run on an Android phone or emulator.
//!
//! H.264 and AV1 decoding are exact, so MediaCodec's pictures must match
//! the Rust decoders' sample for sample on streams the Rust encoders write.

use cutlass_decode::{
    CodecError, CodecId, CodecParams, DecodeFactory, Decoder, Output, Packet, PictureOutput,
    PictureSamples, PixelFormat, SideData,
};
use cutlass_encode::{CpuFrame, Encoder, FrameInput, FrameTime, VideoEncoder, VideoInput, Yuv420p};

use super::decoder::open_decoder_on;
use super::encoder::open_encoder_on;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

/// A gradient that slides one step a picture, so motion is coded.
fn planes(step: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (WIDTH as usize, HEIGHT as usize);
    let shift = step as usize * 3;
    let y = (0..w * h)
        .map(|at| ((at % w + at / w + shift) % 220 + 16) as u8)
        .collect();
    let u = (0..w * h / 4)
        .map(|at| ((at % (w / 2)) * 2 % 200 + 28) as u8)
        .collect();
    let v = (0..w * h / 4)
        .map(|at| ((at / (w / 2) + shift) % 200 + 28) as u8)
        .collect();
    (y, u, v)
}

/// The luma an encoder makes from the RGBA a picture becomes on the input
/// surface, with luma weights `(red, blue)`: BT.709 or BT.601, limited range.
fn surface_luma(step: u32, (red, blue): (f64, f64)) -> Vec<u8> {
    let (mut y, u, v) = planes(step);
    y.extend(u);
    y.extend(v);
    let nv12 = cutlass_color::nv12::from_planar(WIDTH, HEIGHT, &y).expect("nv12");
    let layout = cutlass_color::nv12::Layout::packed(WIDTH, HEIGHT);
    let rgba = cutlass_color::nv12::to_rgba(layout, &nv12).expect("rgba");
    let green = 1.0 - red - blue;
    rgba.as_chunks::<4>()
        .0
        .iter()
        .map(|&[r, g, b, _]| {
            let [r, g, b] = [r, g, b].map(f64::from);
            let luma = red * r + green * g + blue * b;
            (16.0 + luma * 219.0 / 255.0).round() as u8
        })
        .collect()
}

fn h264_stream(pictures: u32) -> (CodecParams, Vec<Packet>) {
    let mut params = CodecParams::video(CodecId::H264, WIDTH, HEIGHT);
    params.nal_length_size = 4;
    let mut encoder = cutlass_h264::H264Encoder::open(&params).expect("encoder");
    let mut packets = Vec::new();
    for step in 0..pictures {
        let (y, u, v) = planes(step);
        let picture = Yuv420p::try_new(WIDTH, HEIGHT, y, u, v).expect("picture");
        packets.extend(
            encoder
                .push(&VideoInput {
                    picture,
                    pts: i64::from(step) * 1001,
                    duration: 1001,
                    timescale: 30_000,
                    force_keyframe: step == 0,
                })
                .expect("encode"),
        );
    }
    packets.extend(encoder.drain().expect("drain"));
    if let Some(config) = packets.iter().find_map(config_of) {
        params.extra = config;
    }
    (params, packets)
}

fn av1_stream(pictures: u32) -> (CodecParams, Vec<Packet>) {
    let params = CodecParams::with_format(CodecId::Av1, WIDTH, HEIGHT, PixelFormat::Yuv420p);
    let mut encoder = cutlass_av1::Av1Encoder::open(&params).expect("encoder");
    let mut packets = Vec::new();
    for step in 0..pictures {
        let (mut y, u, v) = planes(step);
        y.extend(u);
        y.extend(v);
        let time = FrameTime::new(i64::from(step), 1, 25).expect("time");
        let frame = CpuFrame::new(WIDTH, HEIGHT, PixelFormat::Yuv420p, y, time).expect("frame");
        packets.extend(encoder.push_frame(&FrameInput::Cpu(frame)).expect("encode"));
    }
    packets.extend(encoder.drain().expect("drain"));
    let mut params = params;
    if let Some(config) = packets.iter().find_map(config_of) {
        params.extra = config;
    }
    (params, packets)
}

fn config_of(packet: &Packet) -> Option<Vec<u8>> {
    packet.side.iter().find_map(|side| match side {
        SideData::Config(bytes) if !bytes.is_empty() => Some(bytes.clone()),
        _ => None,
    })
}

fn decode_all(decoder: &mut dyn Decoder, packets: &[Packet]) -> Vec<PictureOutput> {
    let mut outputs = Vec::new();
    for packet in packets {
        outputs.extend(decoder.push(packet).expect("push"));
    }
    outputs.extend(decoder.drain().expect("drain"));
    pictures(outputs)
}

fn pictures(outputs: Vec<Output>) -> Vec<PictureOutput> {
    let mut pictures: Vec<PictureOutput> = outputs
        .into_iter()
        .filter_map(|output| match output {
            Output::Picture(picture) => Some(picture),
            Output::Audio(_) => None,
        })
        .collect();
    pictures.sort_by_key(PictureOutput::pts);
    pictures
}

fn rust_pictures(params: &CodecParams, packets: &[Packet]) -> Vec<PictureOutput> {
    let mut factory = DecodeFactory::new();
    cutlass_h264::register_decoders(&mut factory);
    cutlass_av1::register_decoders(&mut factory);
    let mut decoder = factory.open(params).expect("rust decoder");
    decode_all(decoder.as_mut(), packets)
}

fn bytes(picture: &PictureOutput) -> &[u8] {
    match picture.samples() {
        PictureSamples::Cpu(bytes) => bytes,
        PictureSamples::Platform { .. } | PictureSamples::Surface(_) => panic!("not cpu bytes"),
    }
}

fn assert_same(got: &[PictureOutput], want: &[PictureOutput], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: picture count");
    for (index, (got, want)) in got.iter().zip(want).enumerate() {
        assert_eq!(got.pts(), want.pts(), "{what}: picture {index} time");
        assert_eq!(got.info().width, want.info().width, "{what}: width");
        assert_eq!(got.info().height, want.info().height, "{what}: height");
        assert_eq!(got.info().pixel, PixelFormat::Yuv420p, "{what}: pixel");
        assert!(
            bytes(got) == bytes(want),
            "{what}: picture {index} samples differ"
        );
    }
}

#[test]
fn software_h264_matches_the_rust_decoder() {
    let (mut params, packets) = h264_stream(12);
    params.pixel = PixelFormat::Yuv420p;
    let want = rust_pictures(&params, &packets);
    assert_eq!(want.len(), 12);
    let mut decoder = open_decoder_on(&params, false).expect("software decoder");
    assert!(!decoder.hardware(), "{} is not software", decoder.name());
    assert_same(&decode_all(&mut decoder, &packets), &want, decoder.name());
}

#[test]
fn the_platform_h264_decoder_matches_the_rust_decoder() {
    let (mut params, packets) = h264_stream(12);
    params.pixel = PixelFormat::Yuv420p;
    let want = rust_pictures(&params, &packets);
    let mut decoder = open_decoder_on(&params, true).expect("platform decoder");
    let name = decoder.name().to_owned();
    assert_same(&decode_all(&mut decoder, &packets), &want, &name);
}

#[test]
fn software_av1_matches_the_rust_decoder() {
    let (params, packets) = av1_stream(6);
    let want = rust_pictures(&params, &packets);
    assert_eq!(want.len(), 6);
    let mut decoder = open_decoder_on(&params, false).expect("software decoder");
    assert_same(&decode_all(&mut decoder, &packets), &want, decoder.name());
}

#[test]
fn a_drained_decoder_decodes_again_and_a_second_drain_is_empty() {
    let (mut params, packets) = h264_stream(6);
    params.pixel = PixelFormat::Yuv420p;
    let mut decoder = open_decoder_on(&params, false).expect("decoder");
    let first = decode_all(&mut decoder, &packets);
    assert_eq!(first.len(), 6);
    assert!(decoder.drain().expect("second drain").is_empty());
    let again = decode_all(&mut decoder, &packets);
    assert_same(&again, &first, "after a drain");
    decoder.push(&packets[0]).expect("push");
    decoder.reset();
    let after_reset = decode_all(&mut decoder, &packets);
    assert_same(&after_reset, &first, "after a reset");
}

#[test]
fn an_rgba_request_gives_rgba_pictures() {
    let (mut params, packets) = h264_stream(3);
    params.pixel = PixelFormat::Rgba8;
    let mut decoder = open_decoder_on(&params, false).expect("decoder");
    let pictures = decode_all(&mut decoder, &packets);
    assert_eq!(pictures.len(), 3);
    for picture in &pictures {
        assert_eq!(picture.info().pixel, PixelFormat::Rgba8);
        assert_eq!(bytes(picture).len(), WIDTH as usize * HEIGHT as usize * 4);
    }
}

#[test]
fn bad_input_is_an_error_and_not_a_crash() {
    let (mut params, packets) = h264_stream(2);
    params.pixel = PixelFormat::Yuv420p;
    let mut decoder = open_decoder_on(&params, false).expect("decoder");
    let empty = Packet::video(Vec::new(), 0, 1, 30, true);
    assert!(matches!(
        decoder.push(&empty),
        Err(CodecError::InvalidApi { .. })
    ));
    let garbage = Packet::video(vec![0, 0, 0, 3, 0xff, 0xee, 0xdd], 0, 1, 30, true);
    let _ = decoder.push(&garbage);
    let _ = decoder.drain();
    decoder.reset();
    assert_eq!(decode_all(&mut decoder, &packets).len(), 2);
    let mut zero = params.clone();
    zero.width = 0;
    assert!(open_decoder_on(&zero, false).is_err());
    let mut huge = params;
    huge.width = 20_000;
    assert!(matches!(
        open_decoder_on(&huge, false),
        Err(CodecError::Unsupported { .. })
    ));
}

#[test]
fn the_factory_opens_mediacodec_for_each_video_id() {
    let mut factory = DecodeFactory::new();
    super::register_decoders_with(&mut factory, false);
    for codec in [CodecId::H264, CodecId::Hevc, CodecId::Vp9, CodecId::Av1] {
        let params = CodecParams::with_format(codec, WIDTH, HEIGHT, PixelFormat::Yuv420p);
        let decoder = factory.open(&params);
        assert!(decoder.is_ok(), "{codec:?}: {:?}", decoder.err());
    }
}

/// Peak signal to noise of two 8-bit planes, in decibels.
fn psnr(got: &[u8], want: &[u8]) -> f64 {
    let error: f64 = got
        .iter()
        .zip(want)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
        .sum::<f64>()
        / want.len() as f64;
    if error == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0_f64 * 255.0 / error).log10()
}

fn encode_frames(encoder: &mut dyn Encoder, pictures: u32, pixel: PixelFormat) -> Vec<Packet> {
    let mut packets = Vec::new();
    for step in 0..pictures {
        let (mut y, u, v) = planes(step);
        let bytes = if pixel == PixelFormat::Rgba8 {
            y.iter()
                .flat_map(|luma| [*luma, *luma, *luma, 255])
                .collect()
        } else {
            y.extend(u);
            y.extend(v);
            y
        };
        let time = FrameTime::new(i64::from(step) * 1001, 1001, 30_000).expect("time");
        let frame = CpuFrame::new(WIDTH, HEIGHT, pixel, bytes, time).expect("frame");
        packets.extend(encoder.push_frame(&FrameInput::Cpu(frame)).expect("encode"));
    }
    packets.extend(encoder.drain().expect("drain"));
    packets
}

fn check_encoder(codec: CodecId, hardware: bool) {
    let params = CodecParams::with_format(codec, WIDTH, HEIGHT, PixelFormat::Yuv420p);
    let mut encoder = match open_encoder_on(&params, hardware) {
        Ok(encoder) => encoder,
        Err(error) => panic!("{codec:?} encoder: {error:?}"),
    };
    let name = encoder.name().to_owned();
    let packets = encode_frames(&mut encoder, 15, PixelFormat::Yuv420p);
    assert_eq!(packets.len(), 15, "{name}: one unit a picture");
    assert!(packets[0].keyframe, "{name}: the first unit is a keyframe");
    let times: Vec<i64> = packets.iter().map(|packet| packet.pts).collect();
    let want: Vec<i64> = (0..15).map(|step| step * 1001).collect();
    assert_eq!(times, want, "{name}: units keep their pictures' times");
    let config = config_of(&packets[0]).expect("configuration record on the keyframe");
    assert!(
        packets[1..]
            .iter()
            .all(|packet| config_of(packet).is_none()),
        "{name}: only the first keyframe carries the record"
    );
    let mut decode = params.clone();
    decode.extra = config;
    decode.nal_length_size = 4;
    let mut factory = DecodeFactory::new();
    cutlass_h264::register_decoders(&mut factory);
    cutlass_hevc::register_decoders(&mut factory);
    let mut decoder = factory.open(&decode).expect("rust decoder");
    let pictures = decode_all(decoder.as_mut(), &packets);
    assert_eq!(
        pictures.len(),
        15,
        "{name}: the Rust decoder reads every unit"
    );
    let luma = WIDTH as usize * HEIGHT as usize;
    for (step, picture) in pictures.iter().enumerate() {
        // Phones' encoders turn the surface's RGB into YUV with their own
        // matrix: the emulator's always with BT.601, and none says which.
        let got = &bytes(picture)[..luma];
        let quality = [(0.2126, 0.0722), (0.299, 0.114)]
            .map(|weights| psnr(got, &surface_luma(step as u32, weights)))
            .into_iter()
            .fold(0.0, f64::max);
        assert!(quality > 30.0, "{name}: picture {step} at {quality:.1} dB");
    }
}

#[test]
fn software_h264_encode_reads_back_in_rust() {
    check_encoder(CodecId::H264, false);
}

#[test]
fn the_platform_h264_encoder_reads_back_in_rust() {
    check_encoder(CodecId::H264, true);
}

#[test]
fn software_hevc_encode_reads_back_in_rust() {
    check_encoder(CodecId::Hevc, false);
}

#[test]
fn rgba_input_is_encoded_and_a_reset_starts_with_a_keyframe() {
    let params = CodecParams::with_format(CodecId::H264, WIDTH, HEIGHT, PixelFormat::Rgba8);
    let mut encoder = open_encoder_on(&params, false).expect("encoder");
    let first = encode_frames(&mut encoder, 4, PixelFormat::Rgba8);
    assert_eq!(first.len(), 4);
    assert!(encoder.drain().expect("second drain").is_empty());
    encoder.reset();
    let again = encode_frames(&mut encoder, 2, PixelFormat::Rgba8);
    assert!(again[0].keyframe);
    assert!(
        config_of(&again[0]).is_some(),
        "the record again after a reset"
    );
}

#[test]
fn the_encoder_refuses_what_it_cannot_take() {
    let mut params = CodecParams::with_format(CodecId::H264, WIDTH, HEIGHT, PixelFormat::Yuv420p);
    let mut encoder = open_encoder_on(&params, false).expect("encoder");
    let small = CpuFrame::new(
        16,
        16,
        PixelFormat::Yuv420p,
        vec![0; 16 * 16 * 3 / 2],
        FrameTime::new(0, 1, 30).expect("time"),
    )
    .expect("frame");
    assert!(encoder.push_frame(&FrameInput::Cpu(small)).is_err());
    encoder
        .cancel_flag()
        .store(true, std::sync::atomic::Ordering::Relaxed);
    let (mut y, u, v) = planes(0);
    y.extend(u);
    y.extend(v);
    let frame = CpuFrame::new(
        WIDTH,
        HEIGHT,
        PixelFormat::Yuv420p,
        y,
        FrameTime::new(0, 1, 30).expect("time"),
    )
    .expect("frame");
    assert!(matches!(
        encoder.push_frame(&FrameInput::Cpu(frame)),
        Err(CodecError::Cancelled)
    ));
    params.width = 321;
    assert!(open_encoder_on(&params, false).is_err());
    params.width = WIDTH;
    params.codec = CodecId::Vp9;
    assert!(matches!(
        open_encoder_on(&params, false),
        Err(CodecError::Unsupported { .. })
    ));
}

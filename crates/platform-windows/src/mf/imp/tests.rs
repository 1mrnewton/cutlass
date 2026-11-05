use cutlass_decode::{
    CodecError, CodecId, CodecParams, ColorPrimaries, DecodeFactory, Decoder, Output, Packet,
    PictureOutput, PictureSamples, PixelFormat, SideData,
};
use cutlass_encode::{CpuFrame, EncodeFactory, Encoder, FrameInput, FrameTime, PlatformFrame};

use super::video::{open_decoder, open_decoder_on, open_encoder, open_encoder_on};
use crate::nal;

fn video_params(width: u32, height: u32) -> CodecParams {
    codec_params(CodecId::H264, width, height)
}

fn codec_params(codec: CodecId, width: u32, height: u32) -> CodecParams {
    let mut params = CodecParams::with_format(codec, width, height, PixelFormat::Rgba8);
    params.bitrate = 500_000;
    params.primaries = ColorPrimaries::Bt709;
    params
}

fn solid(width: u32, height: u32, pixel: [u8; 4], pts: i64) -> FrameInput {
    let bytes = pixel.repeat(width as usize * height as usize);
    FrameInput::Cpu(
        CpuFrame::new(
            width,
            height,
            PixelFormat::Rgba8,
            bytes,
            FrameTime::new(pts, 1, 30).expect("time"),
        )
        .expect("frame"),
    )
}

fn encode(encoder: &mut dyn Encoder, width: u32, height: u32, frames: i64) -> Vec<Packet> {
    let mut packets = Vec::new();
    for pts in 0..frames {
        packets.extend(
            encoder
                .push_frame(&solid(width, height, [220, 10, 10, 255], pts))
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
        .expect("avcC")
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

fn mean_channel(bytes: &[u8], channel: usize) -> u64 {
    let pixels = bytes.as_chunks::<4>().0;
    pixels
        .iter()
        .map(|pixel| u64::from(pixel[channel]))
        .sum::<u64>()
        / pixels.len() as u64
}

#[test]
fn h264_round_trips_a_solid_color() {
    let mut encoder = open_encoder(&video_params(64, 48)).expect("encode open");
    let packets = encode(encoder.as_mut(), 64, 48, 6);
    assert_eq!(packets.len(), 6);
    let first = &packets[0];
    assert!(first.keyframe, "the first unit is a keyframe");
    assert!(
        first
            .side
            .iter()
            .any(|side| matches!(side, SideData::Config(_))),
        "the first keyframe carries the avcC"
    );
    for (index, packet) in packets.iter().enumerate() {
        assert_eq!(
            (packet.pts, packet.duration, packet.timescale),
            (index as i64, 1, 30)
        );
        assert!(nal::split_lengths(&packet.data, 4).is_ok(), "length framed");
    }
    let avc = nal::avc_config(&config(&packets)).expect("avcC parses");
    assert_eq!(avc.profile, 66, "Baseline");

    let mut params = video_params(64, 48);
    params.extra = config(&packets);
    let mut decoder = open_decoder(&params).expect("decode open");
    let pictures = decode_all(decoder.as_mut(), &packets);
    assert_eq!(pictures.len(), 6);
    for (index, picture) in pictures.iter().enumerate() {
        assert_eq!(picture.pts(), index as i64);
        assert_eq!(picture.timescale(), 30);
        let info = picture.info();
        assert_eq!(
            (info.width, info.height, info.pixel),
            (64, 48, PixelFormat::Rgba8)
        );
        let PictureSamples::Cpu(bytes) = picture.samples() else {
            panic!("cpu");
        };
        let (red, green, blue) = (
            mean_channel(bytes, 0),
            mean_channel(bytes, 1),
            mean_channel(bytes, 2),
        );
        assert!(
            red > 180 && green < 50 && blue < 50,
            "red came back as {red} {green} {blue}"
        );
    }
    assert!(decoder.drain().expect("second drain").is_empty());
}

#[test]
fn h264_decodes_to_cpu_yuv_when_asked() {
    let mut encoder = open_encoder(&video_params(64, 48)).expect("encode open");
    let packets = encode(encoder.as_mut(), 64, 48, 2);
    let mut params = CodecParams::with_format(CodecId::H264, 64, 48, PixelFormat::Yuv420p);
    params.extra = config(&packets);
    let mut decoder = open_decoder(&params).expect("decode open");
    let pictures = decode_all(decoder.as_mut(), &packets);
    let picture = pictures.first().expect("a picture");
    assert_eq!(picture.info().pixel, PixelFormat::Yuv420p);
    let PictureSamples::Cpu(bytes) = picture.samples() else {
        panic!("cpu");
    };
    assert_eq!(bytes.len(), 64 * 48 * 3 / 2);
    let mean = |plane: &[u8]| {
        plane.iter().map(|&value| u64::from(value)).sum::<u64>() / plane.len() as u64
    };
    let (y, rest) = bytes.split_at(64 * 48);
    let (u, v) = rest.split_at(32 * 24);
    let (y, u, v) = (mean(y), mean(u), mean(v));
    assert!((40..=90).contains(&y), "red luma {y}");
    assert!(u < 128 && v > 180, "red chroma u {u} v {v}");
}

#[test]
fn annex_b_units_decode_without_an_avcc() {
    let mut encoder = open_encoder(&video_params(64, 48)).expect("encode open");
    let packets = encode(encoder.as_mut(), 64, 48, 3);
    let avc = nal::avc_config(&config(&packets)).expect("avcC");
    let mut annex_b = Vec::new();
    for (index, packet) in packets.iter().enumerate() {
        let mut nals = Vec::new();
        if index == 0 {
            nals.extend(
                avc.sps
                    .iter()
                    .chain(&avc.pps)
                    .cloned()
                    .map(|bytes| nal::Nal { bytes }),
            );
        }
        nals.extend(nal::split_lengths(&packet.data, 4).expect("lengths"));
        annex_b.push(Packet {
            data: nal::to_annex_b(&nals),
            side: Vec::new(),
            ..packet.clone()
        });
    }
    let mut decoder = open_decoder(&video_params(64, 48)).expect("decode open");
    assert_eq!(decode_all(decoder.as_mut(), &annex_b).len(), 3);
}

#[test]
fn a_frame_size_that_is_not_a_multiple_of_sixteen_is_cropped() {
    let mut encoder = open_encoder(&video_params(200, 120)).expect("encode open");
    let packets = encode(encoder.as_mut(), 200, 120, 2);
    let mut params = video_params(200, 120);
    params.extra = config(&packets);
    let mut decoder = open_decoder(&params).expect("decode open");
    let pictures = decode_all(decoder.as_mut(), &packets);
    let info = pictures.first().expect("a picture").info();
    assert_eq!((info.width, info.height), (200, 120));
}

#[test]
fn reset_starts_again_on_a_keyframe_with_the_config() {
    let mut encoder = open_encoder(&video_params(64, 48)).expect("encode open");
    let _ = encode(encoder.as_mut(), 64, 48, 2);
    encoder.reset();
    let again = encode(encoder.as_mut(), 64, 48, 2);
    assert!(again[0].keyframe);
    assert!(
        again[0]
            .side
            .iter()
            .any(|side| matches!(side, SideData::Config(_)))
    );

    let mut params = video_params(64, 48);
    params.extra = config(&again);
    let mut decoder = open_decoder(&params).expect("decode open");
    let _ = decoder.push(&again[0]).expect("decode");
    decoder.reset();
    assert!(decoder.drain().expect("drain").is_empty());
    assert_eq!(decode_all(decoder.as_mut(), &again).len(), 2);
}

#[test]
fn a_platform_frame_is_declined() {
    let mut encoder = open_encoder(&video_params(64, 48)).expect("encode open");
    let frame =
        PlatformFrame::new(64, 48, 7, FrameTime::new(0, 1, 30).expect("time")).expect("frame");
    assert!(matches!(
        encoder.push_frame(&FrameInput::Platform(frame)),
        Err(CodecError::Unsupported { .. })
    ));
}

#[test]
fn an_odd_size_or_another_codec_is_declined() {
    assert!(matches!(
        open_decoder(&video_params(63, 48)),
        Err(CodecError::Unsupported { .. })
    ));
    assert!(matches!(
        open_encoder(&video_params(64, 47)),
        Err(CodecError::Unsupported { .. })
    ));
    let mut vp8 = video_params(64, 48);
    vp8.codec = CodecId::Vp8;
    assert!(matches!(
        open_decoder(&vp8),
        Err(CodecError::Unsupported { .. })
    ));
}

#[test]
fn an_empty_packet_is_an_invalid_call() {
    let mut decoder = open_decoder(&video_params(64, 48)).expect("open");
    let err = decoder.push(&Packet {
        stream: 0,
        data: Vec::new(),
        pts: 0,
        duration: 1,
        timescale: 30,
        keyframe: true,
        side: Vec::new(),
    });
    assert_eq!(
        err,
        Err(CodecError::InvalidApi {
            what: "empty packet"
        })
    );
}

#[test]
fn random_packets_fail_closed() {
    let mut decoder = open_decoder(&video_params(64, 48)).expect("open");
    let mut state = 0x1234_5678_u32;
    for _ in 0..32 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let len = (state % 48) as usize + 1;
        let data = (0..len)
            .map(|index| (state.wrapping_add(index as u32) % 256) as u8)
            .collect();
        let result = decoder.push(&Packet {
            stream: 0,
            data,
            pts: 0,
            duration: 1,
            timescale: 30,
            keyframe: false,
            side: Vec::new(),
        });
        assert!(
            result.as_ref().is_err() || result.as_ref().is_ok_and(Vec::is_empty),
            "a random packet made a picture"
        );
    }
}

#[test]
fn the_factory_uses_media_foundation_first() {
    let mut encoders = EncodeFactory::new();
    super::register_encoders(&mut encoders);
    let mut encoder = encoders.open(&video_params(64, 48)).expect("encoder");
    let packets = encode(encoder.as_mut(), 64, 48, 1);

    let mut decoders = DecodeFactory::new();
    super::register_decoders(&mut decoders);
    let mut params = video_params(64, 48);
    params.extra = config(&packets);
    let mut decoder = decoders.open(&params).expect("decoder");
    assert_eq!(decode_all(decoder.as_mut(), &packets).len(), 1);

    let mut none = DecodeFactory::new();
    super::register_decoders_with(&mut none, false);
    assert!(none.open(&params).is_err());
}

/// Encode red on one engine and decode on another, and check the colour.
/// Returns whether each side really ran on the GPU.
fn round_trip(
    codec: CodecId,
    (width, height): (u32, u32),
    encode_on_gpu: bool,
    decode_on_gpu: bool,
) -> Result<(bool, bool), CodecError> {
    let mut encoder = open_encoder_on(&codec_params(codec, width, height), encode_on_gpu)?;
    let encoded_on_gpu = encoder.hardware();
    let packets = encode(&mut encoder, width, height, 4);
    assert_eq!(packets.len(), 4, "{codec:?} {encoded_on_gpu}");
    assert!(packets[0].keyframe);
    let mut params = codec_params(codec, width, height);
    params.extra = config(&packets);
    let mut decoder = open_decoder_on(&params, decode_on_gpu)?;
    let pictures = decode_all(&mut decoder, &packets);
    let decoded_on_gpu = decoder.hardware();
    assert_eq!(
        pictures.len(),
        4,
        "{codec:?} decode on gpu {decoded_on_gpu}"
    );
    for (index, picture) in pictures.iter().enumerate() {
        assert_eq!(picture.pts(), index as i64);
        let info = picture.info();
        assert_eq!((info.width, info.height), (width, height));
        let PictureSamples::Cpu(bytes) = picture.samples() else {
            panic!("cpu");
        };
        let (red, green, blue) = (
            mean_channel(bytes, 0),
            mean_channel(bytes, 1),
            mean_channel(bytes, 2),
        );
        assert!(
            red > 180 && green < 50 && blue < 50,
            "{codec:?} gpu encode {encoded_on_gpu} decode {decoded_on_gpu}: {red} {green} {blue}"
        );
    }
    Ok((encoded_on_gpu, decoded_on_gpu))
}

#[test]
fn h264_round_trips_on_every_engine() {
    for size in [(64, 48), (1920, 1080)] {
        for (encode_on_gpu, decode_on_gpu) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let ran = round_trip(CodecId::H264, size, encode_on_gpu, decode_on_gpu).expect("h264");
            eprintln!("h264 {size:?}: asked gpu {encode_on_gpu}/{decode_on_gpu}, ran {ran:?}");
            if !encode_on_gpu {
                assert!(!ran.0, "a software encode stays in software");
            }
            if !decode_on_gpu {
                assert!(!ran.1, "a software decode stays in software");
            }
        }
    }
}

#[test]
fn hevc_round_trips_when_windows_can() {
    for size in [(64, 48), (1920, 1080)] {
        for (encode_on_gpu, decode_on_gpu) in [(true, true), (true, false)] {
            match round_trip(CodecId::Hevc, size, encode_on_gpu, decode_on_gpu) {
                Ok(ran) => eprintln!(
                    "hevc {size:?}: asked gpu {encode_on_gpu}/{decode_on_gpu}, ran {ran:?}"
                ),
                Err(CodecError::Unsupported { what }) => {
                    eprintln!("hevc {size:?}: this Windows declines ({what})");
                }
                Err(error) => panic!("hevc {size:?}: {error}"),
            }
        }
    }
}

#[test]
fn an_hevc_config_is_a_real_hvcc() {
    let Ok(mut encoder) = open_encoder_on(&codec_params(CodecId::Hevc, 64, 48), true) else {
        return;
    };
    let packets = encode(&mut encoder, 64, 48, 1);
    let bytes = config(&packets);
    assert_eq!(bytes[0], 1, "configurationVersion");
    let sets = nal::hevc_config(&bytes).expect("hvcC parses");
    let kinds: Vec<_> = sets.nals.iter().map(|set| nal::hevc_type(set)).collect();
    assert_eq!(kinds, [Some(32), Some(33), Some(34)]);
}

/// A hardware HEVC encoder asked for B pictures emits units out of display
/// order (H.264 is asked for Baseline, which has none). Each decoder, on the GPU and off it, gives the pictures back in
/// display order with their own times.
#[test]
fn b_pictures_come_back_in_display_order() {
    super::video::TEST_B_PICTURES.with(|count| count.set(2));
    let encoder = open_encoder_on(&codec_params(CodecId::Hevc, 320, 240), true);
    super::video::TEST_B_PICTURES.with(|count| count.set(0));
    let mut encoder = match encoder {
        Ok(encoder) => encoder,
        Err(CodecError::Unsupported { what }) => {
            eprintln!("skipped: {what} cannot encode HEVC here");
            return;
        }
        Err(error) => panic!("encoder: {error:?}"),
    };
    if !encoder.hardware() {
        eprintln!("skipped: no hardware encoder to make B pictures");
        return;
    }
    let mut packets = Vec::new();
    for pts in 0..30 {
        let shade = (pts * 8) as u8;
        packets.extend(
            encoder
                .push_frame(&solid(320, 240, [shade, 255 - shade, 40, 255], pts))
                .expect("encode"),
        );
    }
    packets.extend(encoder.drain().expect("drain"));
    let order: Vec<i64> = packets.iter().map(|packet| packet.pts).collect();
    let reordered = order.windows(2).any(|pair| pair[1] < pair[0]);
    eprintln!("decode order {order:?}");
    if !reordered {
        eprintln!("skipped: the encoder made no B pictures");
        return;
    }
    let mut params = codec_params(CodecId::Hevc, 320, 240);
    params.extra = config(&packets);
    for on_gpu in [true, false] {
        let mut decoder = open_decoder_on(&params, on_gpu).expect("decoder");
        let pictures = decode_all(&mut decoder, &packets);
        let shown: Vec<i64> = pictures.iter().map(PictureOutput::pts).collect();
        assert_eq!(shown, (0..30).collect::<Vec<_>>(), "gpu {on_gpu}");
        for picture in &pictures {
            let PictureSamples::Cpu(bytes) = picture.samples() else {
                panic!("cpu");
            };
            let want = (picture.pts() * 8) as u64;
            let red = mean_channel(bytes, 0);
            assert!(
                red.abs_diff(want) <= 12,
                "gpu {on_gpu} pts {} red {red} want {want}",
                picture.pts()
            );
        }
    }
}

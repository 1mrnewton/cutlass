//! Decodes the shared fixtures and compares every frame with the reference
//! decoder's output.

#![cfg(target_vendor = "apple")]
#![expect(clippy::unwrap_used, reason = "a failed codec test should panic")]

mod support;

use cutlass_core::{
    ColorTags, ContentLight, MasteringDisplay, Rotation, VideoCodec, VideoStreamConfig,
};
use cutlass_test_fixtures::VideoFixture;
use cutlass_videotoolbox::{Decoder, DecoderOptions, Error};
use support::{decode_all, frame_crc};

fn assert_matches_reference(fixture: &VideoFixture) {
    let mut decoder = Decoder::new(&fixture.config, DecoderOptions::default()).unwrap();
    let frames = decode_all(&mut decoder, &fixture.packets);

    let pts: Vec<_> = frames.iter().map(|frame| frame.pts).collect();
    let expected_pts: Vec<_> = fixture.frames.iter().map(|frame| frame.pts).collect();
    assert_eq!(
        pts, expected_pts,
        "{}: frames not in presentation order",
        fixture.name
    );

    for (frame, expected) in frames.iter().zip(&fixture.frames) {
        let image = &frame.image;
        assert_eq!(
            image.format(),
            Some((fixture.pixel_format, fixture.color_range)),
            "{}: output format",
            fixture.name
        );
        assert_eq!(
            (image.width(), image.height()),
            (
                fixture.config.width as usize,
                fixture.config.height as usize
            ),
            "{}: output size",
            fixture.name
        );
        assert_eq!(
            frame_crc(image, fixture.pixel_format),
            expected.crc32,
            "{}: frame at pts {} differs from the reference decoder",
            fixture.name,
            expected.pts.value
        );
        assert!(
            frame.color.is_none() && frame.mastering.is_none() && frame.content_light.is_none(),
            "{}: an untagged stream leaves the frame untagged",
            fixture.name
        );
        assert_eq!(
            frame.rotation,
            Rotation::Zero,
            "{}: an upright stream stays upright",
            fixture.name
        );
    }
}

#[test]
fn a_frame_keeps_the_colour_it_was_opened_with() {
    let fixture = cutlass_test_fixtures::video("h264-high-bframes-128x72");
    let tags = ColorTags {
        primaries: 1,
        transfer: 1,
        matrix: 1,
        full_range: Some(false),
    };
    let display = MasteringDisplay {
        primaries_x: [1, 2, 3],
        primaries_y: [4, 5, 6],
        white_x: 7,
        white_y: 8,
        max_luminance: 10_000_000,
        min_luminance: 0,
    };
    let light = ContentLight {
        max_content_light_level: 1_000,
        max_pic_average_light_level: 0,
    };
    let mut decoder = Decoder::new(
        &fixture.config,
        DecoderOptions {
            color: Some(tags),
            mastering: Some(display),
            content_light: Some(light),
            rotation: Rotation::Clockwise90,
            ..DecoderOptions::default()
        },
    )
    .unwrap();
    let frames = decode_all(&mut decoder, &fixture.packets);
    assert!(!frames.is_empty());
    for frame in &frames {
        assert_eq!(frame.color, Some(tags));
        assert_eq!(frame.mastering, Some(display));
        assert_eq!(frame.content_light, Some(light));
        assert_eq!(frame.rotation, Rotation::Clockwise90);
    }
}

#[test]
fn h264_high_with_b_frames_matches_reference() {
    assert_matches_reference(&cutlass_test_fixtures::video("h264-high-bframes-128x72"));
}

#[test]
fn h264_full_range_stays_full_range() {
    assert_matches_reference(&cutlass_test_fixtures::video("h264-high-full-range-128x72"));
}

#[test]
fn h264_size_that_is_not_a_multiple_of_16_is_cropped() {
    assert_matches_reference(&cutlass_test_fixtures::video("h264-high-cropped-100x58"));
}

#[test]
fn hevc_main_with_b_frames_matches_reference() {
    assert_matches_reference(&cutlass_test_fixtures::video("hevc-main-bframes-128x72"));
}

#[test]
fn hevc_main10_decodes_to_p010() {
    assert_matches_reference(&cutlass_test_fixtures::video("hevc-main10-bframes-128x72"));
}

#[test]
fn hevc_b_frames_stay_in_order_when_decode_time_starts_at_zero() {
    let fixture = cutlass_test_fixtures::video("hevc-main-bframes-128x72");
    let shift = fixture
        .packets
        .iter()
        .filter_map(|packet| packet.dts.map(|time| time.value))
        .min()
        .unwrap();
    let mut packets = fixture.packets.clone();
    for packet in &mut packets {
        if let Some(time) = packet.dts.as_mut() {
            time.value -= shift;
        }
    }
    let offset = cutlass_videotoolbox::composition_offset(&packets);
    assert!(offset < 0);
    let mut decoder = Decoder::new(
        &fixture.config,
        DecoderOptions {
            composition_offset: offset,
            ..DecoderOptions::default()
        },
    )
    .unwrap();
    let frames = decode_all(&mut decoder, &packets);
    let pts: Vec<_> = frames.iter().map(|frame| frame.pts).collect();
    let expected: Vec<_> = fixture.frames.iter().map(|frame| frame.pts).collect();
    assert_eq!(pts, expected);
}

#[test]
fn reset_then_second_gop_matches_reference() {
    let fixture = cutlass_test_fixtures::video("h264-high-bframes-128x72");
    let mut decoder = Decoder::new(&fixture.config, DecoderOptions::default()).unwrap();
    decode_all(&mut decoder, &fixture.packets[..3]);
    decoder.reset().unwrap();

    let second_gop = fixture
        .packets
        .iter()
        .position(|packet| packet.keyframe && packet.pts.value > 0)
        .unwrap();
    let frames = decode_all(&mut decoder, &fixture.packets[second_gop..]);
    let expected = &fixture.frames[fixture.frames.len() - frames.len()..];
    assert_eq!(frames.len(), fixture.packets.len() - second_gop);
    for (frame, expected) in frames.iter().zip(expected) {
        assert_eq!(frame.pts, expected.pts);
        assert_eq!(
            frame_crc(&frame.image, fixture.pixel_format),
            expected.crc32
        );
    }
}

#[test]
fn rejects_truncated_avcc() {
    let mut fixture = cutlass_test_fixtures::video("h264-high-bframes-128x72");
    fixture.config.config_record.truncate(10);
    let error = Decoder::new(&fixture.config, DecoderOptions::default())
        .err()
        .unwrap();
    assert!(matches!(error, Error::InvalidConfig(_)), "{error}");
}

#[test]
fn vp8_is_unsupported() {
    let config = VideoStreamConfig {
        codec: VideoCodec::Vp8,
        width: 64,
        height: 64,
        config_record: Vec::new(),
    };
    let error = Decoder::new(&config, DecoderOptions::default())
        .err()
        .unwrap();
    assert!(error.is_unsupported(), "{error}");
}

#[test]
fn rejects_empty_packet() {
    let fixture = cutlass_test_fixtures::video("h264-high-bframes-128x72");
    let mut decoder = Decoder::new(&fixture.config, DecoderOptions::default()).unwrap();
    let mut packet = fixture.packets[0].clone();
    packet.data.clear();
    assert!(matches!(
        decoder.send_packet(&packet),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn garbage_packet_is_an_error_not_a_crash() {
    let fixture = cutlass_test_fixtures::video("h264-high-bframes-128x72");
    let mut decoder = Decoder::new(&fixture.config, DecoderOptions::default()).unwrap();
    let mut packet = fixture.packets[0].clone();
    packet.data = vec![0xff; 64];
    let sent = decoder.send_packet(&packet);
    let flushed = decoder.flush();
    let output: Vec<_> = std::iter::from_fn(|| decoder.receive_frame()).collect();
    assert!(
        sent.is_err() || flushed.is_err() || output.iter().any(Result::is_err) || output.is_empty(),
        "garbage decoded cleanly into {} frames",
        output.len()
    );
}

//! Encodes synthetic frames, decodes the result, and checks the picture
//! survives.

#![cfg(target_vendor = "apple")]
#![expect(clippy::unwrap_used, reason = "a failed codec test should panic")]
#![expect(clippy::print_stderr, reason = "a skipped hardware test prints why")]

mod support;

use std::num::NonZeroU32;

use cutlass_core::{Packet, PixelFormat, Timestamp, VideoCodec, VideoStreamConfig};
use cutlass_videotoolbox::{Decoder, DecoderOptions, Encoder, EncoderConfig, Error, PixelBuffer};
use support::decode_all;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 180;
const FRAMES: usize = 30;
const TIMESCALE: NonZeroU32 = NonZeroU32::new(30).unwrap();

/// A diagonal ramp that moves with the frame index, in 8-bit luma code
/// values.
fn luma(x: usize, y: usize, index: usize) -> u16 {
    ((x + y + index * 4) % 220 + 16) as u16
}

fn fill(image: &mut PixelBuffer, format: PixelFormat, index: usize) {
    image
        .write_planes(|planes| {
            let [luma_plane, chroma_plane] = planes else {
                panic!("expected two planes, got {}", planes.len());
            };
            for y in 0..luma_plane.height {
                let row = &mut luma_plane.data[y * luma_plane.bytes_per_row..];
                for x in 0..luma_plane.width {
                    match format {
                        PixelFormat::Nv12 => row[x] = luma(x, y, index) as u8,
                        PixelFormat::P010 => {
                            let sample = (luma(x, y, index) << 2) << 6;
                            row[2 * x..2 * x + 2].copy_from_slice(&sample.to_le_bytes());
                        }
                    }
                }
            }
            let neutral: &[u8] = match format {
                PixelFormat::Nv12 => &[128],
                PixelFormat::P010 => &[0x00, 0x80],
            };
            for (byte, value) in chroma_plane.data.iter_mut().zip(neutral.iter().cycle()) {
                *byte = *value;
            }
        })
        .unwrap();
}

/// Peak signal-to-noise ratio of the decoded luma against the source, in dB,
/// on the 8-bit scale.
fn luma_psnr(image: &PixelBuffer, format: PixelFormat, index: usize) -> f64 {
    image
        .read_planes(|planes| {
            let plane = &planes[0];
            let mut squared_error = 0.0;
            for y in 0..plane.height {
                let row = &plane.data[y * plane.bytes_per_row..];
                for x in 0..plane.width {
                    let decoded = match format {
                        PixelFormat::Nv12 => f64::from(row[x]),
                        PixelFormat::P010 => {
                            f64::from(u16::from_le_bytes([row[2 * x], row[2 * x + 1]]) >> 6) / 4.0
                        }
                    };
                    let error = decoded - f64::from(luma(x, y, index));
                    squared_error += error * error;
                }
            }
            let mse = squared_error / (plane.width * plane.height) as f64;
            10.0 * (255.0 * 255.0 / mse.max(1e-9)).log10()
        })
        .unwrap()
}

/// `None` when this machine has no encoder for the config.
fn encode(
    config: &EncoderConfig,
    force_keyframe_at: Option<usize>,
) -> Option<(Vec<Packet>, Vec<u8>)> {
    let mut encoder = match Encoder::new(config) {
        Ok(encoder) => encoder,
        Err(error) if error.is_unsupported() => {
            eprintln!("skipping: no {:?} encoder here: {error}", config.codec);
            return None;
        }
        Err(error) => panic!("creating encoder: {error}"),
    };
    let mut packets = Vec::new();
    for index in 0..FRAMES {
        let mut image = encoder.create_pixel_buffer().unwrap();
        fill(&mut image, config.pixel_format, index);
        let pts = Timestamp::new(index as i64, TIMESCALE);
        let duration = Some(Timestamp::new(1, TIMESCALE));
        encoder
            .send_frame(image, pts, duration, force_keyframe_at == Some(index))
            .unwrap();
        while let Some(packet) = encoder.receive_packet() {
            packets.push(packet.unwrap());
        }
    }
    encoder.flush().unwrap();
    while let Some(packet) = encoder.receive_packet() {
        packets.push(packet.unwrap());
    }
    let mut previous_dts = None;
    for packet in &packets {
        let Some(dts) = packet.dts else { continue };
        assert!(
            dts <= packet.pts,
            "decode time {} is after presentation time {}",
            dts.value,
            packet.pts.value
        );
        if let Some(previous) = previous_dts {
            assert!(previous < dts, "decode times must increase");
        }
        previous_dts = Some(dts);
    }
    let config_record = encoder.config_record().unwrap().to_vec();
    Some((packets, config_record))
}

fn assert_round_trip(codec: VideoCodec, format: PixelFormat) {
    let config = EncoderConfig {
        pixel_format: format,
        keyframe_interval: Some(10),
        frame_rate: Some(30.0),
        ..EncoderConfig::new(codec, WIDTH, HEIGHT)
    };
    let Some((packets, config_record)) = encode(&config, None) else {
        return;
    };
    assert_eq!(packets.len(), FRAMES);
    assert!(packets[0].keyframe, "first packet must be a keyframe");
    assert_eq!(config_record.first(), Some(&1), "config record version");

    let stream = VideoStreamConfig {
        codec,
        width: WIDTH,
        height: HEIGHT,
        config_record,
    };
    let mut decoder = Decoder::new(&stream, DecoderOptions::default()).unwrap();
    let frames = decode_all(&mut decoder, &packets);
    assert_eq!(frames.len(), FRAMES);
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(
            frame.pts,
            Timestamp::new(index as i64, TIMESCALE),
            "frame order"
        );
        assert_eq!(frame.image.format().map(|(format, _)| format), Some(format));
        let psnr = luma_psnr(&frame.image, format, index);
        assert!(psnr > 35.0, "frame {index}: luma PSNR {psnr:.1} dB");
    }
}

#[test]
fn h264_round_trip_keeps_the_picture() {
    assert_round_trip(VideoCodec::H264, PixelFormat::Nv12);
}

#[test]
fn hevc_round_trip_keeps_the_picture() {
    assert_round_trip(VideoCodec::Hevc, PixelFormat::Nv12);
}

#[test]
fn hevc_10_bit_round_trip_keeps_the_picture() {
    assert_round_trip(VideoCodec::Hevc, PixelFormat::P010);
}

#[test]
fn forced_keyframe_comes_out_as_keyframe() {
    let config = EncoderConfig {
        keyframe_interval: Some(1000),
        ..EncoderConfig::new(VideoCodec::H264, WIDTH, HEIGHT)
    };
    let Some((packets, _)) = encode(&config, Some(7)) else {
        return;
    };
    let forced = packets.iter().find(|packet| packet.pts.value == 7).unwrap();
    assert!(forced.keyframe);
    assert_eq!(packets.iter().filter(|packet| packet.keyframe).count(), 2);
}

#[test]
fn b_frames_make_decode_order_differ_from_presentation_order() {
    let Some((packets, _)) = encode(&EncoderConfig::new(VideoCodec::H264, WIDTH, HEIGHT), None)
    else {
        return;
    };
    let pts: Vec<i64> = packets.iter().map(|packet| packet.pts.value).collect();
    assert!(
        pts.windows(2).any(|pair| pair[1] < pair[0]),
        "no reordering in {pts:?}"
    );
}

#[test]
fn h264_10_bit_is_unsupported() {
    let config = EncoderConfig {
        pixel_format: PixelFormat::P010,
        ..EncoderConfig::new(VideoCodec::H264, WIDTH, HEIGHT)
    };
    let error = Encoder::new(&config).err().unwrap();
    assert!(
        matches!(error, Error::EncodeUnsupported(VideoCodec::H264)),
        "{error}"
    );
}

#[test]
fn decoded_frames_reject_writes_unless_videotoolbox_marks_them_modifiable() {
    let fixture = cutlass_test_fixtures::video("h264-high-bframes-128x72");
    let mut decoder = Decoder::new(&fixture.config, DecoderOptions::default()).unwrap();
    let mut frame = decode_all(&mut decoder, &fixture.packets).remove(0);
    let before = support::frame_crc(&frame.image, fixture.pixel_format);
    match frame.image.write_planes(|planes| planes[0].data.fill(0)) {
        Ok(()) => assert_ne!(
            support::frame_crc(&frame.image, fixture.pixel_format),
            before
        ),
        Err(Error::InvalidInput(_)) => {
            assert_eq!(
                support::frame_crc(&frame.image, fixture.pixel_format),
                before
            );
        }
        Err(error) => panic!("unexpected error: {error}"),
    }
}

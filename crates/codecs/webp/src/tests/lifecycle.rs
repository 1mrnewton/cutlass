//! The decoder and encoder through the calls the engine makes, in the
//! orders it can make them: repeat, reset, drain, cancel and refuse.

use std::sync::atomic::Ordering;

use cutlass_decode::{CodecError, CodecId, CodecParams, Decoder, Packet};
use cutlass_encode::{
    BitDepth, ColorType, CpuFrame, Encoder, FrameInput, FrameTime, PictureInput, PixelFormat,
    StillImage,
};

use super::animation::split;
use super::encoder::encode_animation;
use super::oracle::synthetic_rgba;
use crate::{WebpDecoder, WebpEncoder, WebpOptions};

fn decoder() -> WebpDecoder {
    WebpDecoder::open(&CodecParams::video(CodecId::Webp, 0, 0)).expect("decoder")
}

fn encoder() -> WebpEncoder {
    let mut params = CodecParams::video(CodecId::Webp, 0, 0);
    params.extra = WebpOptions::lossless().to_extra();
    WebpEncoder::open(&params).expect("encoder")
}

fn picture(width: u32, height: u32) -> PictureInput {
    let samples = synthetic_rgba(width, height)
        .into_iter()
        .map(u16::from)
        .collect();
    PictureInput {
        image: StillImage::from_samples(width, height, ColorType::Rgba, BitDepth::Eight, samples),
    }
}

fn frame(index: i64, width: u32, height: u32) -> FrameInput {
    let mut rgba = synthetic_rgba(width, height);
    rgba[0] = index as u8;
    let time = FrameTime::new(index * 100, 100, 1000).expect("time");
    FrameInput::Cpu(CpuFrame::new(width, height, PixelFormat::Rgba8, rgba, time).expect("frame"))
}

fn still_packet(data: Vec<u8>) -> Packet {
    Packet::video(data, 0, 0, 0, true)
}

#[test]
fn the_decoder_answers_stills_alike_and_refuses_what_it_cannot_take() {
    let file = encoder().push_picture(&picture(9, 7)).expect("encode")[0]
        .data
        .clone();
    let mut decoder = decoder();
    assert!(WebpDecoder::open(&CodecParams::video(CodecId::Png, 0, 0)).is_err());
    assert!(matches!(
        decoder.push(&still_packet(file.clone())),
        Err(CodecError::InvalidApi { .. })
    ));
    assert!(matches!(
        decoder.decode_picture(&still_packet(Vec::new())),
        Err(CodecError::InvalidApi { .. })
    ));
    assert!(decoder.drain().expect("drain").is_empty());
    let first = decoder
        .decode_picture(&still_packet(file.clone()))
        .expect("first");
    // A broken packet in between changes nothing for the next one.
    assert!(
        decoder
            .decode_picture(&still_packet(file[..20].to_vec()))
            .is_err()
    );
    let second = decoder
        .decode_picture(&still_packet(file.clone()))
        .expect("second");
    decoder.reset();
    let third = decoder.decode_picture(&still_packet(file)).expect("third");
    assert_eq!(first[0].image, second[0].image);
    assert_eq!(first[0].image, third[0].image);
    assert!(decoder.drain().expect("drain").is_empty());
}

#[test]
fn a_reset_forgets_the_canvas_and_a_new_first_packet_starts_again() {
    let frames: Vec<Vec<u8>> = (0..3)
        .map(|index| {
            let mut rgba = synthetic_rgba(12, 10);
            rgba[(index * 4) as usize] = 255;
            rgba
        })
        .collect();
    let packets = encode_animation(WebpOptions::lossless(), &frames, 12, 10);
    let mut file = Vec::new();
    for packet in &packets {
        file.extend_from_slice(&packet.data);
    }
    let parts = split(&file);
    assert_eq!(parts.len(), 3);
    let mut decoder = decoder();
    let outputs: Vec<_> = parts
        .iter()
        .map(|part| {
            decoder
                .decode_picture(&still_packet(part.clone()))
                .expect("frame")
        })
        .collect();
    decoder.reset();
    // Without the first packet there is no canvas to draw on.
    assert!(
        decoder
            .decode_picture(&still_packet(parts[1].clone()))
            .is_err()
    );
    // Starting over gives the same canvases.
    for (part, output) in parts.iter().zip(&outputs) {
        let again = decoder
            .decode_picture(&still_packet(part.clone()))
            .expect("again");
        assert_eq!(again[0].image, output[0].image);
    }
    // A still in the middle of an animation decodes on its own.
    let still = encoder().push_picture(&picture(5, 5)).expect("still")[0]
        .data
        .clone();
    assert_eq!(
        decoder.decode_picture(&still_packet(still)).expect("still")[0]
            .image
            .width,
        5
    );
}

#[test]
fn the_encoder_drains_once_resets_and_starts_a_new_file_after_a_drain() {
    let mut encoder = encoder();
    assert_eq!(encoder.codec(), CodecId::Webp);
    assert!(encoder.drain().expect("empty drain").is_empty());
    for index in 0..3 {
        assert!(
            encoder
                .push_frame(&frame(index, 8, 6))
                .expect("push")
                .is_empty()
        );
    }
    // A still while frames wait is its own file and leaves them alone.
    let still = encoder.push_picture(&picture(4, 4)).expect("still");
    assert_eq!(&still[0].data[..4], b"RIFF");
    let first = encoder.drain().expect("drain");
    assert_eq!(first.len(), 3);
    assert_eq!(&first[0].data[..4], b"RIFF");
    assert_eq!(&first[1].data[..4], b"ANMF");
    assert!(encoder.drain().expect("second drain").is_empty());

    // After a drain the next frame starts a new file.
    assert!(
        encoder
            .push_frame(&frame(0, 8, 6))
            .expect("push")
            .is_empty()
    );
    let again = encoder.drain().expect("drain");
    assert_eq!(again.len(), 1);
    assert_eq!(&again[0].data[..4], b"RIFF");

    // A reset drops the frames that wait.
    encoder.push_frame(&frame(0, 8, 6)).expect("push");
    encoder.push_frame(&frame(1, 8, 6)).expect("push");
    encoder.reset();
    assert!(encoder.drain().expect("after reset").is_empty());
}

#[test]
fn the_encoder_refuses_audio_other_pixels_and_a_raised_cancel_flag() {
    let mut encoder = encoder();
    let pcm = cutlass_encode::AudioInput {
        pcm: cutlass_encode::Pcm::from_f32(
            48_000,
            cutlass_encode::ChannelLayout::mono(),
            vec![0.0],
        )
        .expect("pcm"),
    };
    assert!(matches!(
        encoder.push(&pcm),
        Err(CodecError::InvalidApi { .. })
    ));
    let yuv = CpuFrame::new(
        4,
        4,
        PixelFormat::Yuv420p,
        vec![0; 24],
        FrameTime::new(0, 1, 30).expect("time"),
    );
    if let Ok(yuv) = yuv {
        assert!(encoder.push_frame(&FrameInput::Cpu(yuv)).is_err());
    }

    let cancel = encoder.cancel_flag();
    encoder.push_frame(&frame(0, 8, 6)).expect("push");
    cancel.store(true, Ordering::Relaxed);
    assert!(matches!(
        encoder.push_frame(&frame(1, 8, 6)),
        Err(CodecError::Cancelled)
    ));
    assert!(matches!(
        encoder.push_picture(&picture(4, 4)),
        Err(CodecError::Cancelled)
    ));
    assert!(matches!(encoder.drain(), Err(CodecError::Cancelled)));
    // Lowered again, the encoder carries on from a reset.
    cancel.store(false, Ordering::Relaxed);
    encoder.reset();
    encoder.push_frame(&frame(0, 8, 6)).expect("push");
    assert_eq!(encoder.drain().expect("drain").len(), 1);
}

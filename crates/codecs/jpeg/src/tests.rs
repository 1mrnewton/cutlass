//! Behavior locked for the JPEG support record.
//!
//! Hand-built streams are the CI proof. ffmpeg and cjpeg, when installed,
//! are the independent files. `../local-files` is skipped when it is absent.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use cutlass_decode::{
    BitDepth, CodecError, CodecId, CodecParams, ColorType, DecodeFactory, Decoder, Packet,
    StillImage,
};
use cutlass_encode::{EncodeFactory, Encoder, PictureInput};

use crate::decode::{JpegDecoder, decode_jpeg, read_jpeg_info};
use crate::encode::{JpegEncoder, encode_jpeg};
use crate::tables::{AC_LUMA_COUNTS, AC_LUMA_SYMBOLS, DC_LUMA_COUNTS, DC_LUMA_SYMBOLS};

fn params(extra: &[u8]) -> CodecParams {
    let mut params = CodecParams::audio(CodecId::Jpeg, 1, cutlass_decode::ChannelLayout::mono());
    params.extra = extra.to_vec();
    params
}

fn decode(bytes: &[u8]) -> Result<StillImage, CodecError> {
    let mut decoder = JpegDecoder::open(&params(&[])).expect("open");
    let packet = Packet::audio(bytes.to_vec(), 7, 1, 30);
    let output = decoder.decode_picture(&packet)?;
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].pts, 7);
    Ok(output.into_iter().next().expect("picture").image)
}

fn gray(width: u16, height: u16, entropy: &[u8]) -> Vec<u8> {
    let mut file = vec![0xff, 0xd8];
    file.extend(dqt(&[1u8; 64]));
    file.extend(sof(0xc0, width, height, &[(1, 0x11, 0)]));
    file.extend(dht(0x00, &DC_LUMA_COUNTS, &DC_LUMA_SYMBOLS));
    file.extend(dht(0x10, &AC_LUMA_COUNTS, &AC_LUMA_SYMBOLS));
    file.extend(sos(&[(1, 0x00)], 0, 63, 0, 0));
    file.extend_from_slice(entropy);
    file.extend_from_slice(&[0xff, 0xd9]);
    file
}

fn segment(marker: u8, body: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0xff, marker];
    let len = u16::try_from(body.len() + 2).expect("segment");
    bytes.extend(len.to_be_bytes());
    bytes.extend_from_slice(body);
    bytes
}

fn dqt(zigzag: &[u8]) -> Vec<u8> {
    let mut body = vec![0];
    body.extend_from_slice(zigzag);
    segment(0xdb, &body)
}

fn dht(class_id: u8, counts: &[u8], symbols: &[u8]) -> Vec<u8> {
    let mut body = vec![class_id];
    body.extend_from_slice(counts);
    body.extend_from_slice(symbols);
    segment(0xc4, &body)
}

fn sof(marker: u8, width: u16, height: u16, comps: &[(u8, u8, u8)]) -> Vec<u8> {
    let mut body = vec![8];
    body.extend(height.to_be_bytes());
    body.extend(width.to_be_bytes());
    body.push(u8::try_from(comps.len()).expect("components"));
    for (id, sampling, quant) in comps {
        body.extend([*id, *sampling, *quant]);
    }
    segment(marker, &body)
}

fn sos(comps: &[(u8, u8)], ss: u8, se: u8, ah: u8, al: u8) -> Vec<u8> {
    let mut body = vec![u8::try_from(comps.len()).expect("scan")];
    for (id, tables) in comps {
        body.extend([*id, *tables]);
    }
    body.extend([ss, se, (ah << 4) | al]);
    segment(0xda, &body)
}

fn assert_flat(image: &StillImage, value: u16) {
    assert!(image.samples.iter().all(|sample| *sample == value));
}

#[test]
fn a_zero_block_is_mid_grey() {
    let image = decode(&gray(8, 8, &[0x2b])).expect("decode");
    assert_eq!(image.width, 8);
    assert_eq!(image.height, 8);
    assert_eq!(image.color, ColorType::Gray);
    assert_eq!(image.depth, BitDepth::Eight);
    assert_eq!(image.samples.len(), 64);
    assert_flat(&image, 128);
}

#[test]
fn a_dc_of_1016_is_white() {
    let image = decode(&gray(8, 8, &[0xfe, 0xfe, 0x2b])).expect("decode");
    assert_flat(&image, 255);
}

#[test]
fn one_horizontal_coefficient_matches_the_idct() {
    let image = decode(&gray(8, 8, &[0x3e, 0x32, 0x57])).expect("decode");
    let row = [145u16, 143, 138, 131, 125, 118, 113, 111];
    assert_eq!(image.samples.len(), 64);
    for y in 0..8 {
        assert_eq!(&image.samples[y * 8..y * 8 + 8], &row);
    }
}

#[test]
fn extended_sequential_uses_the_same_scan() {
    let mut file = gray(8, 8, &[0x2b]);
    let sof_at = file.iter().position(|byte| *byte == 0xc0).expect("sof");
    file[sof_at] = 0xc1;
    let image = decode(&file).expect("sof1");
    assert_flat(&image, 128);
}

#[test]
fn subsampled_420_keeps_the_mcu_padding_blocks() {
    let mut file = vec![0xff, 0xd8];
    file.extend(dqt(&[1u8; 64]));
    file.extend(sof(0xc0, 8, 8, &[(1, 0x22, 0), (2, 0x11, 0), (3, 0x11, 0)]));
    file.extend(dht(0x00, &DC_LUMA_COUNTS, &DC_LUMA_SYMBOLS));
    file.extend(dht(0x10, &AC_LUMA_COUNTS, &AC_LUMA_SYMBOLS));
    file.extend(sos(&[(1, 0x00), (2, 0x00), (3, 0x00)], 0, 63, 0, 0));
    file.extend_from_slice(&[0x28, 0xa2, 0x8a, 0x28, 0xaf, 0xff, 0xd9]);
    let image = decode(&file).expect("420");
    assert_eq!(image.color, ColorType::Rgb);
    assert_eq!(image.samples.len(), 8 * 8 * 3);
    assert_flat(&image, 128);
}

#[test]
fn progressive_ac_matches_the_sequential_block() {
    let mut file = vec![0xff, 0xd8];
    file.extend(dqt(&[1u8; 64]));
    file.extend(sof(0xc2, 8, 8, &[(1, 0x11, 0)]));
    file.extend(dht(0x00, &DC_LUMA_COUNTS, &DC_LUMA_SYMBOLS));
    file.extend(dht(0x10, &AC_LUMA_COUNTS, &AC_LUMA_SYMBOLS));
    file.extend(sos(&[(1, 0x00)], 0, 0, 0, 0));
    file.push(0x3f);
    file.extend(sos(&[(1, 0x00)], 1, 63, 0, 0));
    file.extend_from_slice(&[0xf8, 0xc9, 0x5f, 0xff, 0xd9]);
    let image = decode(&file).expect("progressive");
    let row = [145u16, 143, 138, 131, 125, 118, 113, 111];
    for y in 0..8 {
        assert_eq!(&image.samples[y * 8..y * 8 + 8], &row);
    }
    let info = read_jpeg_info(&file).expect("info");
    assert!(info.progressive);
    assert_eq!((info.width, info.height, info.components), (8, 8, 1));
}

#[test]
fn successive_ac_correction_adds_the_low_bit() {
    let mut file = vec![0xff, 0xd8];
    file.extend(dqt(&[1u8; 64]));
    file.extend(sof(0xc2, 8, 8, &[(1, 0x11, 0)]));
    file.extend(dht(0x00, &DC_LUMA_COUNTS, &DC_LUMA_SYMBOLS));
    file.extend(dht(0x10, &AC_LUMA_COUNTS, &AC_LUMA_SYMBOLS));
    file.extend(sos(&[(1, 0x00)], 0, 0, 0, 0));
    file.push(0x3f);
    file.extend(sos(&[(1, 0x00)], 1, 63, 0, 1));
    file.push(0x35);
    file.extend(sos(&[(1, 0x00)], 1, 63, 1, 0));
    file.extend_from_slice(&[0xaf, 0xff, 0xd9]);
    let image = decode(&file).expect("refine");
    let row = [129u16, 128, 128, 128, 128, 128, 128, 127];
    for y in 0..8 {
        assert_eq!(&image.samples[y * 8..y * 8 + 8], &row);
    }
}

#[test]
fn successive_dc_shifts_before_the_idct() {
    let mut file = vec![0xff, 0xd8];
    file.extend(dqt(&[1u8; 64]));
    file.extend(sof(0xc2, 8, 8, &[(1, 0x11, 0)]));
    file.extend(dht(0x00, &DC_LUMA_COUNTS, &DC_LUMA_SYMBOLS));
    file.extend(sos(&[(1, 0x00)], 0, 0, 0, 1));
    file.extend_from_slice(&[0xfd, 0xfc]);
    file.extend(sos(&[(1, 0x00)], 0, 0, 1, 0));
    file.extend_from_slice(&[0x7f, 0xff, 0xd9]);
    let image = decode(&file).expect("successive");
    assert_flat(&image, 255);
}

#[test]
fn a_restart_resets_the_dc_predictor() {
    let mut file = vec![0xff, 0xd8];
    file.extend(dqt(&[1u8; 64]));
    file.extend(sof(0xc0, 16, 8, &[(1, 0x11, 0)]));
    file.extend(segment(0xdd, &[0x00, 0x01]));
    file.extend(dht(0x00, &DC_LUMA_COUNTS, &DC_LUMA_SYMBOLS));
    file.extend(dht(0x10, &AC_LUMA_COUNTS, &AC_LUMA_SYMBOLS));
    file.extend(sos(&[(1, 0x00)], 0, 63, 0, 0));
    file.extend_from_slice(&[0xfe, 0xfe, 0x2b, 0xff, 0xd0, 0x2b, 0xff, 0xd9]);
    let image = decode(&file).expect("restart");
    assert_eq!(image.width, 16);
    for y in 0..8 {
        assert!(image.samples[y * 16..y * 16 + 8].iter().all(|s| *s == 255));
        assert!(
            image.samples[y * 16 + 8..y * 16 + 16]
                .iter()
                .all(|s| *s == 128)
        );
    }
}

#[test]
fn size_comes_from_the_frame_and_orientation_is_not_applied() {
    let mut file = vec![0xff, 0xd8];
    file.extend(segment(0xe1, &exif(6)));
    file.extend(dqt(&[1u8; 64]));
    file.extend(sof(0xc0, 8, 4, &[(1, 0x11, 0)]));
    let info = read_jpeg_info(&file).expect("header");
    assert_eq!((info.width, info.height, info.orientation), (8, 4, Some(6)));
    assert!(decode_jpeg(&file).is_err());
    file.extend(dht(0x00, &DC_LUMA_COUNTS, &DC_LUMA_SYMBOLS));
    file.extend(dht(0x10, &AC_LUMA_COUNTS, &AC_LUMA_SYMBOLS));
    file.extend(sos(&[(1, 0x00)], 0, 63, 0, 0));
    file.extend_from_slice(&[0x2b, 0xff, 0xd9, 0x00, 0x11]);
    let image = decode(&file).expect("oriented");
    assert_eq!((image.width, image.height), (8, 4));
    assert_eq!(image.samples.len(), 32);
    assert_flat(&image, 128);
}

#[test]
fn a_broken_exif_block_does_not_fail_the_picture() {
    let mut file = gray(8, 8, &[0x2b]);
    let junk = segment(0xe1, b"Exif\0\0not-a-tiff");
    file.splice(2..2, junk);
    let image = decode(&file).expect("exif");
    assert_flat(&image, 128);
    assert_eq!(read_jpeg_info(&file).expect("info").orientation, None);
}

#[test]
fn arithmetic_coding_is_refused_by_name() {
    for marker in [0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf, 0xcc] {
        let err = decode(&[0xff, 0xd8, 0xff, marker, 0x00, 0x02]).expect_err("arith");
        assert!(
            matches!(
                err,
                CodecError::Unsupported {
                    what: "arithmetic coding"
                }
            ),
            "{marker:#x} {err}"
        );
        let info = read_jpeg_info(&[0xff, 0xd8, 0xff, marker, 0x00, 0x02]);
        assert!(matches!(
            info,
            Err(CodecError::Unsupported {
                what: "arithmetic coding"
            })
        ));
    }
}

#[test]
fn named_unsupported_frames_stay_named() {
    let cases = [
        (0xc3u8, "lossless jpeg"),
        (0xc5, "differential jpeg"),
        (0xf7, "jpeg-ls"),
        (0xc8, "reserved jpeg"),
        (0xde, "hierarchical jpeg"),
        (0xdc, "dnl"),
    ];
    for (marker, what) in cases {
        let err = decode(&[0xff, 0xd8, 0xff, marker, 0x00, 0x02]).expect_err("named");
        assert!(
            matches!(err, CodecError::Unsupported { what: got } if got == what),
            "{marker:#x} {err}"
        );
    }
}

#[test]
fn hostile_bitstreams_return_errors() {
    assert!(decode(&[]).is_err());
    assert!(decode(&[0xff]).is_err());
    assert!(decode(&[0xff, 0xd8]).is_err());
    let huge = sof(0xc0, 65535, 65535, &[(1, 0x11, 0)]);
    let mut oversize = vec![0xff, 0xd8];
    oversize.extend(huge);
    match decode(&oversize) {
        Err(CodecError::Limit { what: "jpeg size" }) => {}
        other => panic!("{other:?}"),
    }
    match decode(&[0xff, 0xd8, 0xff, 0xdb, 0x00, 0x00]) {
        Err(CodecError::Malformed { what: "length", .. }) => {}
        other => panic!("{other:?}"),
    }
    match decode(&[0xff, 0xd8, 0xff, 0xdb, 0xff, 0xff]) {
        Err(CodecError::Truncated { .. }) => {}
        other => panic!("{other:?}"),
    }
    let sos_first = {
        let mut file = vec![0xff, 0xd8];
        file.extend(sos(&[(1, 0x00)], 0, 63, 0, 0));
        file
    };
    match decode(&sos_first) {
        Err(CodecError::Malformed { what: "frame", .. }) => {}
        other => panic!("{other:?}"),
    }
    let bad_sample = {
        let mut file = vec![0xff, 0xd8];
        file.extend(sof(0xc0, 8, 8, &[(1, 0x01, 0)]));
        file
    };
    match decode(&bad_sample) {
        Err(CodecError::Malformed {
            what: "sampling", ..
        }) => {}
        other => panic!("{other:?}"),
    }
    let mut short = gray(8, 8, &[]);
    short.truncate(short.len() - 2);
    assert!(decode(&short).is_err());
    let mut decoder = JpegDecoder::open(&params(&[])).expect("open");
    assert!(
        decoder
            .decode_picture(&Packet::audio(vec![0xff], 0, 0, 0))
            .is_err()
    );
    let good = decoder
        .decode_picture(&Packet::audio(gray(8, 8, &[0x2b]), 0, 0, 0))
        .expect("next");
    assert_flat(&good[0].image, 128);
    let mut bomb = vec![0xff, 0xd8];
    for _ in 0..4097 {
        bomb.extend(segment(0xfe, &[]));
    }
    match decode(&bomb) {
        Err(CodecError::Limit {
            what: "jpeg markers",
        }) => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn precision_cmyk_and_component_count_are_named() {
    let twelve = {
        let mut body = sof(0xc0, 8, 8, &[(1, 0x11, 0)]);
        let sof_at = body.iter().position(|byte| *byte == 0xc0).expect("sof");
        body[sof_at + 3] = 12;
        let mut file = vec![0xff, 0xd8];
        file.extend(body);
        file
    };
    match decode(&twelve) {
        Err(CodecError::Unsupported {
            what: "jpeg precision",
        }) => {}
        other => panic!("{other:?}"),
    }
    match decode_sof_components(4) {
        Err(CodecError::Unsupported { what: "cmyk" }) => {}
        other => panic!("{other:?}"),
    }
    match decode_sof_components(2) {
        Err(CodecError::Unsupported {
            what: "component count",
        }) => {}
        other => panic!("{other:?}"),
    }
}

fn decode_sof_components(count: u8) -> Result<StillImage, CodecError> {
    let comps: Vec<(u8, u8, u8)> = (1..=count).map(|id| (id, 0x11, 0)).collect();
    let mut file = vec![0xff, 0xd8];
    file.extend(sof(0xc0, 8, 8, &comps));
    decode(&file)
}

#[test]
fn the_factory_selects_jpeg_and_rejects_the_wrong_calls() {
    let mut decoders = DecodeFactory::new();
    let mut encoders = EncodeFactory::new();
    crate::register_decoders(&mut decoders);
    crate::register_encoders(&mut encoders);
    let decoder = decoders.open(&params(&[])).expect("decoder");
    assert_eq!(decoder.codec(), CodecId::Jpeg);
    let encoder = encoders.open(&params(&[90])).expect("encoder");
    assert_eq!(encoder.codec(), CodecId::Jpeg);
    assert!(encoders.open(&params(&[0])).is_err());
    let mut opened = JpegDecoder::open(&params(&[])).expect("open");
    assert!(matches!(
        opened.push(&Packet::audio(vec![1], 0, 0, 0)),
        Err(CodecError::InvalidApi {
            what: "jpeg produces a picture"
        })
    ));
    assert!(matches!(
        opened.decode_picture(&Packet::audio(vec![], 0, 0, 0)),
        Err(CodecError::InvalidApi {
            what: "empty packet"
        })
    ));
    assert!(opened.drain().expect("drain").is_empty());
    assert!(opened.drain().expect("drain").is_empty());
    let mut encoder = JpegEncoder::open(&params(&[])).expect("encoder");
    assert!(
        encoder
            .push(&cutlass_encode::AudioInput {
                pcm: cutlass_decode::Pcm::from_f32(
                    48_000,
                    cutlass_decode::ChannelLayout::mono(),
                    vec![0.0],
                )
                .expect("pcm"),
            })
            .is_err()
    );
    encoder.cancel_flag().store(true, Ordering::Relaxed);
    let image = still(8, 8, ColorType::Gray, vec![128; 64]);
    assert!(matches!(
        encoder.push_picture(&PictureInput {
            image: image.clone()
        }),
        Err(CodecError::Cancelled)
    ));
    encoder.reset();
    let packets = encoder
        .push_picture(&PictureInput { image })
        .expect("encode");
    assert_eq!(packets.len(), 1);
    assert!(packets[0].keyframe);
    assert!(encoder.drain().expect("drain").is_empty());
}

#[test]
fn baseline_encoder_round_trips_flat_and_odd_sizes() {
    let gray_image = still(8, 8, ColorType::Gray, vec![128; 64]);
    let encoded =
        encode_jpeg(&gray_image, 100, &std::sync::atomic::AtomicBool::new(false)).expect("encode");
    let back = decode(&encoded).expect("decode");
    assert_eq!(back.color, ColorType::Gray);
    assert_flat(&back, 128);

    let white = still(8, 8, ColorType::Gray, vec![255; 64]);
    let back = decode(
        &encode_jpeg(&white, 100, &std::sync::atomic::AtomicBool::new(false)).expect("white"),
    )
    .expect("white");
    assert_flat(&back, 255);

    let mut rgb = Vec::new();
    for _ in 0..64 {
        rgb.extend([40u16, 80, 200]);
    }
    let color = still(8, 8, ColorType::Rgb, rgb);
    let back = decode(
        &encode_jpeg(&color, 100, &std::sync::atomic::AtomicBool::new(false)).expect("color"),
    )
    .expect("color");
    assert_eq!(
        (back.width, back.height, back.color),
        (8, 8, ColorType::Rgb)
    );
    assert!(mean_abs(&back.samples, &color.samples) <= 8.0);

    for (width, height) in [(1u32, 1), (3, 5), (15, 17), (17, 19)] {
        let samples = vec![90u16; (width * height) as usize];
        let image = still(width, height, ColorType::Gray, samples);
        let back = decode(
            &encode_jpeg(&image, 90, &std::sync::atomic::AtomicBool::new(false)).expect("odd"),
        )
        .expect("odd");
        assert_eq!((back.width, back.height), (width, height));
        assert!(mean_abs(&back.samples, &image.samples) <= 8.0);
    }
}

#[test]
fn the_encoder_refuses_alpha_and_changes_size_with_quality() {
    let alpha = StillImage::from_samples(2, 2, ColorType::Rgba, BitDepth::Eight, vec![1; 16]);
    let err =
        encode_jpeg(&alpha, 75, &std::sync::atomic::AtomicBool::new(false)).expect_err("alpha");
    assert!(matches!(
        err,
        CodecError::Unsupported { what: "jpeg alpha" }
    ));
    let mut noisy = Vec::new();
    let mut state = 0x1234_5678u64;
    for _ in 0..32 * 32 * 3 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        noisy.push((state >> 33) as u16 & 255);
    }
    let image = still(32, 32, ColorType::Rgb, noisy);
    let fine = encode_jpeg(&image, 90, &std::sync::atomic::AtomicBool::new(false)).expect("90");
    let coarse = encode_jpeg(&image, 10, &std::sync::atomic::AtomicBool::new(false)).expect("10");
    assert!(fine.len() > coarse.len());
}

#[test]
fn an_independent_decoder_opens_this_encoder() {
    let Some(ffmpeg) = tool("ffmpeg") else {
        require_or_skip("ffmpeg");
        return;
    };
    let mut samples = Vec::new();
    for y in 0..32u16 {
        for x in 0..32u16 {
            samples.extend([x * 8, y * 8, 128]);
        }
    }
    let image = still(32, 32, ColorType::Rgb, samples.clone());
    let encoded =
        encode_jpeg(&image, 90, &std::sync::atomic::AtomicBool::new(false)).expect("encode");
    let raw = ffmpeg_raw(&ffmpeg, &encoded);
    assert_eq!(raw.len(), 32 * 32 * 3);
    let foreign: Vec<u16> = raw.iter().map(|byte| u16::from(*byte)).collect();
    assert!(
        mean_abs(&foreign, &samples) <= 20.0,
        "{}",
        mean_abs(&foreign, &samples)
    );
}

#[test]
fn an_independent_stream_matches_size_format_and_count() {
    let Some(cjpeg) = tool("cjpeg") else {
        require_or_skip("cjpeg");
        return;
    };
    let Some(ffmpeg) = tool("ffmpeg") else {
        require_or_skip("ffmpeg");
        return;
    };
    let ppm = ppm(16, 12);
    let baseline = Command::new(&cjpeg)
        .args(["-quality", "90", "-sample", "2x2"])
        .output_from(ppm.as_bytes())
        .expect("cjpeg");
    assert!(baseline.status.success(), "{baseline:?}");
    let image = decode(&baseline.stdout).expect("baseline");
    assert_eq!(
        (image.width, image.height, image.color),
        (16, 12, ColorType::Rgb)
    );
    assert_eq!(image.samples.len(), 16 * 12 * 3);
    let progressive = Command::new(&cjpeg)
        .args(["-progressive", "-quality", "80"])
        .output_from(ppm.as_bytes())
        .expect("progressive");
    assert!(progressive.status.success(), "{progressive:?}");
    let image = decode(&progressive.stdout).expect("progressive");
    assert_eq!(image.samples.len(), 16 * 12 * 3);
    assert!(
        read_jpeg_info(&progressive.stdout)
            .expect("info")
            .progressive
    );
    let restarted = Command::new(&cjpeg)
        .args(["-restart", "1", "-quality", "75", "-grayscale"])
        .output_from(pgm(16, 8).as_bytes())
        .expect("restart");
    assert!(restarted.status.success(), "{restarted:?}");
    let image = decode(&restarted.stdout).expect("restart");
    assert_eq!(
        (image.width, image.height, image.color),
        (16, 8, ColorType::Gray)
    );
    let mjpeg = ffmpeg_jpeg(&ffmpeg, &rgb_bytes(8, 8), 8, 8, "rgb24");
    let image = decode(&mjpeg).expect("ffmpeg");
    assert_eq!((image.width, image.height), (8, 8));
    assert_eq!(image.samples.len(), 8 * 8 * 3);
}

#[test]
fn local_files_decode_when_the_folder_is_present() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let Some(checkout) = root.ancestors().nth(3) else {
        return;
    };
    let folder = checkout.join("../local-files");
    if !folder.is_dir() {
        return;
    }
    let mut found = false;
    if let Ok(entries) = std::fs::read_dir(&folder) {
        for entry in entries.flatten() {
            let path = entry.path();
            let ext = path
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if ext == "jpg" || ext == "jpeg" {
                found = true;
                let bytes = std::fs::read(&path).expect("read");
                let info = read_jpeg_info(&bytes).expect("info");
                let image = decode(&bytes).expect("jpeg");
                assert_eq!((image.width, image.height), (info.width, info.height));
                assert!(!image.samples.is_empty());
            }
        }
    }
    if found {
        return;
    }
    let Some(ffmpeg) = tool("ffmpeg") else {
        return;
    };
    let Some(video) = first_video(&folder) else {
        return;
    };
    let jpeg = ffmpeg_frame(&ffmpeg, &video);
    let image = decode(&jpeg).expect("frame");
    assert!(image.width > 0 && image.height > 0);
    assert!(!image.samples.is_empty());
}

#[test]
#[ignore = "release timing budget for one 1920x1080 frame"]
fn release_full_hd_budget() {
    let mut samples = Vec::with_capacity(1920 * 1080 * 3);
    for y in 0..1080u32 {
        for x in 0..1920u32 {
            samples.extend([(x % 256) as u16, (y % 256) as u16, 128]);
        }
    }
    let image = still(1920, 1080, ColorType::Rgb, samples);
    let started = std::time::Instant::now();
    let encoded =
        encode_jpeg(&image, 75, &std::sync::atomic::AtomicBool::new(false)).expect("encode");
    let encode_ms = started.elapsed().as_millis();
    let started = std::time::Instant::now();
    let back = decode(&encoded).expect("decode");
    let decode_ms = started.elapsed().as_millis();
    eprintln!(
        "fullhd encode_ms={encode_ms} decode_ms={decode_ms} bytes={}",
        encoded.len()
    );
    assert_eq!(
        (back.width, back.height, back.color),
        (1920, 1080, ColorType::Rgb)
    );
    assert!(encode_ms < 500, "encode {encode_ms} ms");
    assert!(decode_ms < 250, "decode {decode_ms} ms");
}

#[test]
#[ignore = "decoder fuzz campaign; set JPEG_FUZZ_SECS"]
fn decoder_fuzz_campaign() {
    let seconds: u64 = std::env::var("JPEG_FUZZ_SECS")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(30 * 60);
    let seed = 0x4a50_4500_ffd8_ffd9_u64;
    let mut rng = Rng(seed);
    let mut decoder = JpegDecoder::open(&params(&[])).expect("open");
    let valid = gray(8, 8, &[0xfe, 0xfe, 0x2b]);
    let started = std::time::Instant::now();
    let mut cases = 0u64;
    while started.elapsed().as_secs() < seconds {
        let bytes = if cases.is_multiple_of(3) {
            let mut mutated = valid.clone();
            if !mutated.is_empty() {
                let at = (rng.next() as usize) % mutated.len();
                mutated[at] ^= (rng.next() as u8) | 1;
            }
            mutated
        } else {
            let len = (rng.next() as usize) % 4096;
            (0..len).map(|_| rng.next() as u8).collect()
        };
        let _ = decoder.decode_picture(&Packet::audio(bytes, 0, 0, 0));
        cases += 1;
    }
    eprintln!(
        "fuzz_cases={cases} seed={seed:#x} secs={}",
        started.elapsed().as_secs()
    );
}

fn exif(orientation: u16) -> Vec<u8> {
    let mut tiff = vec![b'I', b'I', 42, 0, 8, 0, 0, 0, 1, 0];
    tiff.extend(0x0112u16.to_le_bytes());
    tiff.extend(3u16.to_le_bytes());
    tiff.extend(1u32.to_le_bytes());
    tiff.extend(orientation.to_le_bytes());
    tiff.extend([0, 0]);
    tiff.extend([0, 0, 0, 0]);
    let mut body = b"Exif\0\0".to_vec();
    body.extend(tiff);
    body
}

fn still(width: u32, height: u32, color: ColorType, samples: Vec<u16>) -> StillImage {
    StillImage::from_samples(width, height, color, BitDepth::Eight, samples)
        .checked()
        .expect("still")
}

fn mean_abs(left: &[u16], right: &[u16]) -> f64 {
    let n = left.len().min(right.len()).max(1);
    let sum: u64 = left
        .iter()
        .zip(right)
        .map(|(a, b)| u64::from(a.abs_diff(*b)))
        .sum();
    sum as f64 / n as f64
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
}

fn ppm(width: u32, height: u32) -> String {
    let mut text = format!("P3\n{width} {height}\n255\n");
    for y in 0..height {
        for x in 0..width {
            text.push_str(&format!("{} {} {} ", x * 8, y * 8, 40));
        }
        text.push('\n');
    }
    text
}

fn pgm(width: u32, height: u32) -> String {
    let mut text = format!("P2\n{width} {height}\n255\n");
    for y in 0..height {
        for x in 0..width {
            text.push_str(&format!("{} ", (x + y) * 8));
        }
        text.push('\n');
    }
    text
}

fn rgb_bytes(width: u32, height: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    for y in 0..height {
        for x in 0..width {
            bytes.extend([x as u8, y as u8, 128]);
        }
    }
    bytes
}

trait OutputFrom {
    fn output_from(&mut self, input: &[u8]) -> std::io::Result<std::process::Output>;
}

impl OutputFrom for Command {
    fn output_from(&mut self, input: &[u8]) -> std::io::Result<std::process::Output> {
        use std::io::Write;
        use std::process::Stdio;
        let mut child = self
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        child.stdin.take().expect("stdin").write_all(input)?;
        child.wait_with_output()
    }
}

fn tool(name: &str) -> Option<PathBuf> {
    let path = Command::new("which").arg(name).output().ok()?;
    if !path.status.success() {
        return None;
    }
    let text = String::from_utf8(path.stdout).ok()?;
    let line = text.lines().next()?.trim();
    if line.is_empty() {
        None
    } else {
        Some(PathBuf::from(line))
    }
}

fn require_or_skip(name: &str) {
    if std::env::var_os("JPEG_REQUIRE_ORACLES").is_some() {
        panic!("{name} is required");
    }
}

fn scratch(label: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("cutlass-jpeg-{label}-{n}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
}

fn ffmpeg_raw(ffmpeg: &Path, jpeg: &[u8]) -> Vec<u8> {
    let dir = scratch("raw");
    let input = dir.join("in.jpg");
    let output = dir.join("out.raw");
    std::fs::write(&input, jpeg).expect("write");
    let status = Command::new(ffmpeg)
        .args(["-y", "-i"])
        .arg(&input)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
        .arg(&output)
        .output()
        .expect("ffmpeg");
    assert!(status.status.success(), "{status:?}");
    let bytes = std::fs::read(&output).expect("raw");
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

fn ffmpeg_jpeg(ffmpeg: &Path, raw: &[u8], width: u32, height: u32, pix: &str) -> Vec<u8> {
    let dir = scratch("jpg");
    let input = dir.join("in.raw");
    let output = dir.join("out.jpg");
    std::fs::write(&input, raw).expect("write");
    let status = Command::new(ffmpeg)
        .args(["-y", "-f", "rawvideo", "-pix_fmt", pix, "-s"])
        .arg(format!("{width}x{height}"))
        .arg("-i")
        .arg(&input)
        .args(["-frames:v", "1", "-c:v", "mjpeg", "-q:v", "3"])
        .arg(&output)
        .output()
        .expect("ffmpeg");
    assert!(status.status.success(), "{status:?}");
    let bytes = std::fs::read(&output).expect("jpg");
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

fn ffmpeg_frame(ffmpeg: &Path, video: &Path) -> Vec<u8> {
    let dir = scratch("frame");
    let output = dir.join("frame.jpg");
    let status = Command::new(ffmpeg)
        .args(["-y", "-i"])
        .arg(video)
        .args(["-frames:v", "1", "-q:v", "3"])
        .arg(&output)
        .output()
        .expect("frame");
    assert!(status.status.success(), "{status:?}");
    let bytes = std::fs::read(&output).expect("jpg");
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

fn first_video(folder: &Path) -> Option<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(folder)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            matches!(
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase()
                    .as_str(),
                "mp4" | "mov" | "webm"
            )
        })
        .collect();
    paths.sort_by_key(|path| path.metadata().map(|meta| meta.len()).unwrap_or(u64::MAX));
    paths.into_iter().next()
}

//! Decoder, encoder, lifecycle, and independent-oracle checks.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::Ordering;

use cutlass_decode::{
    BitDepth, CodecId, CodecParams, ColorType, DecodeFactory, Decoder, Packet, PhysicalPixel, Rgb8,
    StillImage, TextItem, Transparency,
};
use cutlass_encode::{EncodeFactory, Encoder, PictureInput};

use crate::crc::{crc32, crc32_update};
use crate::{PngDecoder, PngEncoder, register_decoders, register_encoders};

const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

#[test]
fn a_hand_built_grey_row_keeps_exact_samples() {
    let image = decode_bytes(&png_of(ihdr(2, 1, 8, 0, 0), &[0, 10, 30], &[])).expect("decode");
    assert_eq!(image.samples, vec![10, 30]);
    assert_eq!(image.color, ColorType::Gray);
    assert_eq!(image.depth, BitDepth::Eight);
}

#[test]
fn sub_and_paeth_rows_reconstruct_the_hand_computed_pixels() {
    let sub = decode_bytes(&png_of(ihdr(2, 1, 8, 0, 0), &[1, 10, 20], &[])).expect("sub");
    assert_eq!(sub.samples, vec![10, 30]);
    let up = decode_bytes(&png_of(ihdr(1, 2, 8, 0, 0), &[0, 10, 2, 20], &[])).expect("up");
    assert_eq!(up.samples, vec![10, 30]);
    let average =
        decode_bytes(&png_of(ihdr(2, 2, 8, 0, 0), &[0, 10, 20, 3, 25, 25], &[])).expect("average");
    assert_eq!(average.samples, vec![10, 20, 30, 50]);
    let paeth =
        decode_bytes(&png_of(ihdr(2, 2, 8, 0, 0), &[0, 10, 20, 4, 20, 20], &[])).expect("paeth");
    assert_eq!(paeth.samples, vec![10, 20, 30, 50]);
}

#[test]
fn one_bit_and_sixteen_bit_samples_stay_in_their_depth() {
    let bits = decode_bytes(&png_of(ihdr(2, 1, 1, 0, 0), &[0, 0b1000_0000], &[])).expect("bit");
    assert_eq!(bits.samples, vec![1, 0]);
    let deep = decode_bytes(&png_of(ihdr(1, 1, 16, 0, 0), &[0, 0x01, 0x02], &[])).expect("deep");
    assert_eq!(deep.samples, vec![0x0102]);
    assert_eq!(deep.depth, BitDepth::Sixteen);
}

#[test]
fn indexed_pixels_stay_indices_and_transparency_stays_a_table() {
    let plte = chunk(b"PLTE", &[1, 2, 3, 4, 5, 6]);
    let trns = chunk(b"tRNS", &[0]);
    let mut extra = plte;
    extra.extend(trns);
    let image = decode_bytes(&png_of(ihdr(2, 1, 8, 3, 0), &[0, 1, 0], &extra)).expect("indexed");
    assert_eq!(image.samples, vec![1, 0]);
    assert_eq!(image.palette.len(), 2);
    assert_eq!(image.transparency, Transparency::PaletteAlpha(vec![0]));
}

#[test]
fn adam7_places_each_pass_back_on_the_full_grid() {
    let raw = [0, 1, 0, 2, 0, 3, 4];
    let image = decode_bytes(&png_of(ihdr(2, 2, 8, 0, 1), &raw, &[])).expect("adam7");
    assert_eq!(image.samples, vec![1, 2, 3, 4]);
}

#[test]
fn adam7_places_one_bit_pixels_from_each_pass() {
    // 8×1 passes: x=0, x=4, then x=2 and x=6 together, then x=1, 3, 5, 7.
    let raw = [0, 0b1000_0000, 0, 0b1000_0000, 0, 0b1100_0000, 0, 0];
    let image = decode_bytes(&png_of(ihdr(8, 1, 1, 0, 1), &raw, &[])).expect("bit adam7");
    assert_eq!(image.samples, vec![1, 0, 1, 0, 1, 0, 1, 0]);
}

#[test]
fn adam7_places_two_bit_pixels_from_each_pass() {
    // 4×1 passes: x=0 is 3, x=2 is 1, then x=1 and x=3 are 2 and 0.
    let raw = [0, 0b1100_0000, 0, 0b0100_0000, 0, 0b1000_0000];
    let image = decode_bytes(&png_of(ihdr(4, 1, 2, 0, 1), &raw, &[])).expect("two bit");
    assert_eq!(image.samples, vec![3, 2, 1, 0]);
}

#[test]
fn adam7_places_four_bit_pixels_from_each_pass() {
    // 2×1 passes: x=0 is 15, then x=1 is 7.
    let raw = [0, 0b1111_0000, 0, 0b0111_0000];
    let image = decode_bytes(&png_of(ihdr(2, 1, 4, 0, 1), &raw, &[])).expect("four bit");
    assert_eq!(image.samples, vec![15, 7]);
}

#[test]
fn truecolor_alpha_and_a_grey_key_keep_their_domains() {
    let rgba = decode_bytes(&png_of(ihdr(1, 1, 8, 6, 0), &[0, 9, 8, 7, 6], &[])).expect("rgba");
    assert_eq!(rgba.samples, vec![9, 8, 7, 6]);
    assert_eq!(rgba.color, ColorType::Rgba);
    let trns = chunk(b"tRNS", &[0, 4]);
    let gray = decode_bytes(&png_of(ihdr(1, 1, 8, 0, 0), &[0, 4], &trns)).expect("key");
    assert_eq!(gray.transparency, Transparency::ColorKey(vec![4]));
    assert_eq!(gray.samples, vec![4]);
}

#[test]
fn metadata_is_kept_and_not_applied_to_samples() {
    let mut extra = chunk(b"cHRM", &chromaticity());
    extra.extend(chunk(b"gAMA", &45_455u32.to_be_bytes()));
    extra.extend(chunk(b"sRGB", &[0]));
    extra.extend(chunk(b"pHYs", &phys(1, 2)));
    extra.extend(chunk(b"tEXt", b"Title\0plain"));
    let image = decode_bytes(&png_of(ihdr(1, 1, 8, 2, 0), &[0, 1, 2, 3], &extra)).expect("meta");
    assert_eq!(image.samples, vec![1, 2, 3]);
    assert_eq!(image.chromaticity, Some(chromaticity_values()));
    assert_eq!(image.gamma_times_100000, Some(45_455));
    assert_eq!(image.srgb_intent, Some(0));
    assert_eq!(
        image.physical,
        Some(PhysicalPixel {
            x: 1,
            y: 2,
            unit: 1
        })
    );
    assert_eq!(image.texts[0].text, b"plain");
}

#[test]
fn the_encoder_round_trips_every_required_domain() {
    let cases = [
        still(2, 2, ColorType::Gray, BitDepth::One, vec![1, 0, 0, 1]),
        still(2, 2, ColorType::Gray, BitDepth::Two, vec![0, 1, 2, 3]),
        still(2, 2, ColorType::Gray, BitDepth::Four, vec![0, 7, 15, 1]),
        still(
            3,
            2,
            ColorType::Gray,
            BitDepth::Eight,
            vec![0, 1, 127, 128, 254, 255],
        ),
        still(1, 2, ColorType::Gray, BitDepth::Sixteen, vec![0, 65535]),
        still(1, 1, ColorType::Rgb, BitDepth::Eight, vec![1, 2, 3]),
        still(1, 1, ColorType::Rgb, BitDepth::Sixteen, vec![1, 256, 65535]),
        still(1, 1, ColorType::GrayAlpha, BitDepth::Eight, vec![8, 9]),
        still(1, 1, ColorType::GrayAlpha, BitDepth::Sixteen, vec![8, 9]),
        still(1, 1, ColorType::Rgba, BitDepth::Eight, vec![1, 2, 3, 4]),
        still(1, 1, ColorType::Rgba, BitDepth::Sixteen, vec![1, 2, 3, 4]),
    ];
    for image in cases {
        assert_round_trip(&image, false);
        assert_round_trip(&image, true);
    }
    let mut indexed =
        StillImage::from_samples(2, 2, ColorType::Indexed, BitDepth::Two, vec![0, 1, 2, 1]);
    indexed.palette = vec![rgb(0, 0, 0), rgb(1, 2, 3), rgb(4, 5, 6), rgb(7, 8, 9)];
    indexed.transparency = Transparency::PaletteAlpha(vec![0, 255]);
    let indexed = indexed.checked().expect("palette");
    assert_round_trip(&indexed, false);
}

#[test]
fn gamma_text_and_a_profile_round_trip_apart_from_the_samples() {
    let mut image = still(1, 1, ColorType::Gray, BitDepth::Eight, vec![3]);
    image.gamma_times_100000 = Some(45_455);
    image.srgb_intent = Some(1);
    image.physical = Some(PhysicalPixel {
        x: 72,
        y: 72,
        unit: 1,
    });
    image.background = vec![3];
    image.significant_bits = vec![8];
    image.texts.push(TextItem {
        keyword: b"Comment".to_vec(),
        language: Vec::new(),
        translated: Vec::new(),
        text: b"kept".to_vec(),
        compressed: true,
    });
    image.icc_name = b"Profile".to_vec();
    image.icc_profile = b"icc-bytes".to_vec();
    let image = image.checked().expect("meta");
    let back = round_trip(&image, false);
    assert_eq!(back.samples, image.samples);
    assert_eq!(back.gamma_times_100000, image.gamma_times_100000);
    assert_eq!(back.srgb_intent, Some(1));
    assert_eq!(back.texts[0].text, b"kept");
    assert!(back.texts[0].compressed);
    assert_eq!(back.icc_profile, b"icc-bytes");
    assert_eq!(back.background, vec![3]);
}

#[test]
fn malformed_streams_fail_and_leave_no_picture() {
    let good = png_of(ihdr(1, 1, 8, 0, 0), &[0, 7], &[]);
    let mut bad_crc = good.clone();
    bad_crc[20] ^= 0xff;
    assert!(decode_bytes(&bad_crc).is_err());
    assert!(decode_bytes(&good[..good.len() - 5]).is_err());
    assert!(decode_bytes(b"not a png").is_err());
    assert!(decode_bytes(&png_of(ihdr(1, 1, 8, 0, 0), &[5, 7], &[])).is_err());
    let mut loco = ihdr(1, 1, 8, 0, 0);
    loco[11] = 64;
    let err = decode_bytes(&png_of(loco, &[0, 7], &[])).expect_err("loco");
    assert!(matches!(
        err,
        cutlass_decode::CodecError::Unsupported {
            what: "png filter method"
        }
    ));
    assert!(decode_bytes(&png_of(ihdr(1, 1, 8, 3, 0), &[0, 0], &[])).is_err());
    let plte = chunk(b"PLTE", &[1, 2, 3]);
    assert!(decode_bytes(&png_of(ihdr(1, 1, 8, 0, 0), &[0, 1], &plte)).is_err());
    let trns = chunk(b"tRNS", &[0, 1]);
    assert!(decode_bytes(&png_of(ihdr(1, 1, 8, 6, 0), &[0, 1, 2, 3, 4], &trns)).is_err());
    let err = decode_bytes(&png_of(ihdr(20_000, 1, 8, 0, 0), &[0, 1], &[])).expect_err("edge");
    assert!(matches!(
        err,
        cutlass_decode::CodecError::Limit { what: "png image" }
    ));
    let mut late_gamma = chunk(b"PLTE", &[1, 2, 3, 4, 5, 6]);
    late_gamma.extend(chunk(b"gAMA", &1_000u32.to_be_bytes()));
    assert!(decode_bytes(&png_of(ihdr(1, 1, 8, 3, 0), &[0, 0], &late_gamma)).is_err());
    assert!(decode_bytes(&png_of(ihdr(0, 1, 8, 0, 0), &[0], &[])).is_err());
    let mut trailing = good.clone();
    trailing.push(0);
    assert!(decode_bytes(&trailing).is_err());
    let mut split = Vec::new();
    split.extend_from_slice(&SIGNATURE);
    split.extend(chunk(b"IHDR", &ihdr(1, 1, 8, 0, 0)));
    let zlib = stored_zlib(&[0, 7]);
    split.extend(chunk(b"IDAT", &zlib[..1]));
    split.extend(chunk(b"tEXt", b"A\0b"));
    split.extend(chunk(b"IDAT", &zlib[1..]));
    split.extend(chunk(b"IEND", &[]));
    assert!(decode_bytes(&split).is_err());
    let mut critical = Vec::new();
    critical.extend_from_slice(&SIGNATURE);
    critical.extend(chunk(b"IHDR", &ihdr(1, 1, 8, 0, 0)));
    critical.extend(chunk(b"ORNG", &[]));
    critical.extend(chunk(b"IDAT", &stored_zlib(&[0, 7])));
    critical.extend(chunk(b"IEND", &[]));
    assert!(decode_bytes(&critical).is_err());
}

#[test]
fn an_unknown_ancillary_chunk_does_not_hide_the_samples() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&SIGNATURE);
    bytes.extend(chunk(b"IHDR", &ihdr(1, 1, 8, 0, 0)));
    bytes.extend(chunk(b"orNG", &[1, 2, 3]));
    bytes.extend(chunk(b"acTL", &[0, 0, 0, 1, 0, 0, 0, 0]));
    bytes.extend(chunk(b"IDAT", &stored_zlib(&[0, 9])));
    bytes.extend(chunk(b"IEND", &[]));
    let image = decode_bytes(&bytes).expect("ancillary");
    assert_eq!(image.samples, vec![9]);
}

#[test]
fn one_byte_idat_chunks_are_one_zlib_stream() {
    let zlib = stored_zlib(&[0, 4]);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&SIGNATURE);
    bytes.extend(chunk(b"IHDR", &ihdr(1, 1, 8, 0, 0)));
    bytes.extend(chunk(b"IDAT", &[]));
    for byte in zlib {
        bytes.extend(chunk(b"IDAT", &[byte]));
    }
    bytes.extend(chunk(b"IEND", &[]));
    let image = decode_bytes(&bytes).expect("split");
    assert_eq!(image.samples, vec![4]);
}

#[test]
fn the_factory_uses_this_codec_and_audio_calls_are_refused() {
    let mut decoders = DecodeFactory::new();
    register_decoders(&mut decoders);
    register_decoders(&mut decoders);
    let params = png_params(false);
    let mut decoder = decoders.open(&params).expect("open");
    assert_eq!(decoder.codec(), CodecId::Png);
    let packet = Packet::audio(png_of(ihdr(1, 1, 8, 0, 0), &[0, 1], &[]), 3, 0, 1);
    assert!(decoder.push(&packet).is_err());
    let picture = decoder.decode_picture(&packet).expect("picture");
    assert_eq!(picture[0].samples(), &[1]);
    assert_eq!(picture[0].pts, 3);
    assert!(decoder.drain().expect("drain").is_empty());
    assert!(decoder.drain().expect("drain again").is_empty());
    assert!(
        decoder
            .decode_picture(&Packet::audio(Vec::new(), 0, 0, 0))
            .is_err()
    );

    let mut encoders = EncodeFactory::new();
    register_encoders(&mut encoders);
    let mut encoder = encoders.open(&params).expect("encoder");
    let image = still(1, 1, ColorType::Gray, BitDepth::Eight, vec![1]);
    assert!(
        encoder
            .push(&cutlass_encode::AudioInput {
                pcm: cutlass_decode::Pcm::from_f32(
                    48_000,
                    cutlass_decode::ChannelLayout::mono(),
                    vec![0.0]
                )
                .expect("pcm"),
            })
            .is_err()
    );
    let packets = encoder
        .push_picture(&PictureInput { image })
        .expect("encode");
    assert_eq!(packets.len(), 1);
    assert!(packets[0].keyframe);
    assert!(encoder.drain().expect("drain").is_empty());
    let flag = encoder.cancel_flag();
    flag.store(true, Ordering::Relaxed);
    let err = encoder
        .push_picture(&PictureInput {
            image: still(1, 1, ColorType::Gray, BitDepth::Eight, vec![2]),
        })
        .expect_err("cancel");
    assert!(matches!(err, cutlass_decode::CodecError::Cancelled));
    encoder.reset();
    assert!(
        encoder
            .push_picture(&PictureInput {
                image: still(1, 1, ColorType::Gray, BitDepth::Eight, vec![2]),
            })
            .is_ok()
    );
}

#[test]
fn a_text_keyword_with_a_leading_space_is_refused() {
    let text = chunk(b"tEXt", b" Title\0plain");
    assert!(decode_bytes(&png_of(ihdr(1, 1, 8, 0, 0), &[0, 1], &text)).is_err());
}

#[test]
fn text_after_idat_is_kept() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&SIGNATURE);
    bytes.extend(chunk(b"IHDR", &ihdr(1, 1, 8, 0, 0)));
    bytes.extend(chunk(b"IDAT", &stored_zlib(&[0, 4])));
    bytes.extend(chunk(b"tEXt", b"Title\0after"));
    bytes.extend(chunk(b"IEND", &[]));
    let image = decode_bytes(&bytes).expect("after");
    assert_eq!(image.samples, vec![4]);
    assert_eq!(image.texts[0].text, b"after");
}

#[test]
fn an_international_text_chunk_round_trips() {
    let mut image = still(1, 1, ColorType::Gray, BitDepth::Eight, vec![1]);
    image.texts.push(TextItem {
        keyword: b"Comment".to_vec(),
        language: b"en".to_vec(),
        translated: b"Note".to_vec(),
        text: b"kept".to_vec(),
        compressed: true,
    });
    let image = image.checked().expect("text");
    let back = round_trip(&image, false);
    assert_eq!(back.samples, vec![1]);
    assert_eq!(back.texts[0].language, b"en");
    assert_eq!(back.texts[0].translated, b"Note");
    assert_eq!(back.texts[0].text, b"kept");
    assert!(back.texts[0].compressed);
}

#[test]
fn an_animation_control_chunk_does_not_hide_the_samples() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&SIGNATURE);
    bytes.extend(chunk(b"IHDR", &ihdr(1, 1, 8, 0, 0)));
    bytes.extend(chunk(b"fcTL", &[0; 26]));
    bytes.extend(chunk(b"IDAT", &stored_zlib(&[0, 6])));
    bytes.extend(chunk(b"IEND", &[]));
    let image = decode_bytes(&bytes).expect("fctl");
    assert_eq!(image.samples, vec![6]);
}

#[test]
fn a_frame_data_chunk_before_idat_is_ignored() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&SIGNATURE);
    bytes.extend(chunk(b"IHDR", &ihdr(1, 1, 8, 0, 0)));
    bytes.extend(chunk(b"fdAT", &[0, 0, 0, 1, 0]));
    bytes.extend(chunk(b"IDAT", &stored_zlib(&[0, 5])));
    bytes.extend(chunk(b"IEND", &[]));
    let image = decode_bytes(&bytes).expect("fdat");
    assert_eq!(image.samples, vec![5]);
}

#[test]
fn a_frame_data_chunk_between_idat_chunks_is_an_error() {
    let zlib = stored_zlib(&[0, 5]);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&SIGNATURE);
    bytes.extend(chunk(b"IHDR", &ihdr(1, 1, 8, 0, 0)));
    bytes.extend(chunk(b"IDAT", &zlib[..1]));
    bytes.extend(chunk(b"fdAT", &[0, 0, 0, 2]));
    bytes.extend(chunk(b"IDAT", &zlib[1..]));
    bytes.extend(chunk(b"IEND", &[]));
    assert!(decode_bytes(&bytes).is_err());
}

#[test]
fn a_failed_picture_does_not_block_the_next_one() {
    let mut decoder = PngDecoder::open(&png_params(false)).expect("open");
    let bad = Packet::audio(b"nope".to_vec(), 0, 0, 0);
    assert!(decoder.decode_picture(&bad).is_err());
    let good = Packet::audio(png_of(ihdr(1, 1, 8, 0, 0), &[0, 7], &[]), 0, 0, 0);
    let picture = decoder.decode_picture(&good).expect("next");
    assert_eq!(picture[0].samples(), &[7]);
}

#[test]
#[ignore = "release timing budget for one 1920x1080 frame"]
fn full_hd_rgba_stays_inside_the_time_budget() {
    if cfg!(debug_assertions) {
        return;
    }
    let mut samples = Vec::with_capacity(1920 * 1080 * 4);
    for y in 0..1080u32 {
        for x in 0..1920u32 {
            let value = ((x.wrapping_mul(3).wrapping_add(y)) & 255) as u16;
            samples.extend([
                value,
                value.wrapping_add(20) & 255,
                value.wrapping_add(40) & 255,
                255,
            ]);
        }
    }
    let image = still(1920, 1080, ColorType::Rgba, BitDepth::Eight, samples);
    let started = std::time::Instant::now();
    let encoded = encode_bytes(&image, false);
    let encode_ms = started.elapsed().as_millis();
    let started = std::time::Instant::now();
    let back = decode_bytes(&encoded).expect("decode");
    let decode_ms = started.elapsed().as_millis();
    eprintln!(
        "encode_ms={encode_ms} decode_ms={decode_ms} bytes={}",
        encoded.len()
    );
    assert!(encode_ms < 500, "encode {encode_ms} ms");
    assert!(decode_ms < 250, "decode {decode_ms} ms");
    assert_eq!(back.samples, image.samples);

    let mut rng = Rng(1);
    let noisy = still(
        1920,
        1080,
        ColorType::Rgba,
        BitDepth::Eight,
        (0..1920 * 1080 * 4)
            .map(|_| ((rng.next() >> 33) as u16) & 255)
            .collect(),
    );
    let started = std::time::Instant::now();
    let encoded = encode_bytes(&noisy, false);
    let encode_ms = started.elapsed().as_millis();
    let started = std::time::Instant::now();
    let back = decode_bytes(&encoded).expect("decode noise");
    let decode_ms = started.elapsed().as_millis();
    eprintln!(
        "noise encode_ms={encode_ms} decode_ms={decode_ms} bytes={}",
        encoded.len()
    );
    assert!(encode_ms < 500, "encode noise {encode_ms} ms");
    assert!(decode_ms < 250, "decode noise {decode_ms} ms");
    assert_eq!(back.samples, noisy.samples);
}

#[test]
#[ignore = "decoder fuzz campaign; set PNG_FUZZ_SECS"]
fn decoder_fuzz_campaign() {
    let seconds: u64 = std::env::var("PNG_FUZZ_SECS")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(30 * 60);
    let seed = 0x504e_47f0_0d0a_1a0a_u64;
    let mut rng = Rng(seed);
    let mut decoder = PngDecoder::open(&png_params(false)).expect("open");
    let valid = png_of(
        ihdr(2, 2, 8, 2, 0),
        &[0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 1, 2, 3],
        &[],
    );
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
            let len = (rng.next() as usize) % 2048;
            (0..len).map(|_| rng.next() as u8).collect()
        };
        let packet = Packet::audio(bytes, 0, 0, 0);
        let _ = decoder.decode_picture(&packet);
        cases += 1;
    }
    eprintln!(
        "fuzz_cases={cases} seed={seed:#x} secs={}",
        started.elapsed().as_secs()
    );
}

#[test]
fn random_images_round_trip_and_random_bytes_fail_closed() {
    let mut rng = Rng(0x1234_5678_9abc_def0);
    for _ in 0..24 {
        let image = random_image(&mut rng);
        assert_round_trip(&image, rng.next().is_multiple_of(2));
    }
    for _ in 0..48 {
        let len = (rng.next() as usize) % 80;
        let bytes: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        let mut decoder = PngDecoder::open(&png_params(false)).expect("open");
        let packet = Packet::audio(bytes, 0, 0, 0);
        assert!(decoder.decode_picture(&packet).is_err());
    }
}

#[test]
fn ffmpeg_and_imagemagick_match_exact_samples_when_present() {
    let Some(ffmpeg) = tool("ffmpeg") else {
        require_or_skip("ffmpeg");
        return;
    };
    let image = still(4, 3, ColorType::Rgb, BitDepth::Eight, ramp(4 * 3 * 3));
    let encoded = encode_bytes(&image, false);
    let foreign = ffmpeg_raw(&ffmpeg, &encoded, "rgb24");
    assert_eq!(foreign, packed8(&image.samples));
    let theirs = ffmpeg_png(&ffmpeg, &packed8(&image.samples), 4, 3, "rgb24", None);
    let decoded = decode_bytes(&theirs).expect("ffmpeg png");
    assert_eq!(decoded.samples, image.samples);
    let mixed = ffmpeg_png(
        &ffmpeg,
        &packed8(&image.samples),
        4,
        3,
        "rgb24",
        Some("mixed"),
    );
    assert_eq!(decode_bytes(&mixed).expect("mixed").samples, image.samples);

    let rgba = still(2, 2, ColorType::Rgba, BitDepth::Eight, ramp(16));
    assert_eq!(
        ffmpeg_raw(&ffmpeg, &encode_bytes(&rgba, false), "rgba"),
        packed8(&rgba.samples)
    );
    let ya = still(2, 2, ColorType::GrayAlpha, BitDepth::Eight, ramp(8));
    assert_eq!(
        ffmpeg_raw(&ffmpeg, &encode_bytes(&ya, false), "ya8"),
        packed8(&ya.samples)
    );
    let deep = still(
        2,
        1,
        ColorType::Gray,
        BitDepth::Sixteen,
        vec![0x0102, 0xffff],
    );
    assert_eq!(
        ffmpeg_raw(&ffmpeg, &encode_bytes(&deep, false), "gray16be"),
        [0x01, 0x02, 0xff, 0xff]
    );
    let deep_png = ffmpeg_png(&ffmpeg, &[0x01, 0x02, 0x00, 0x03], 2, 1, "gray16be", None);
    assert_eq!(
        decode_bytes(&deep_png).expect("gray16").samples,
        vec![0x0102, 0x0003]
    );
    let rgb16 = still(1, 1, ColorType::Rgb, BitDepth::Sixteen, vec![1, 2, 65535]);
    assert_eq!(
        ffmpeg_raw(&ffmpeg, &encode_bytes(&rgb16, false), "rgb48be"),
        [0x00, 0x01, 0x00, 0x02, 0xff, 0xff]
    );
    let mut indexed =
        StillImage::from_samples(2, 1, ColorType::Indexed, BitDepth::Eight, vec![1, 0]);
    indexed.palette = vec![rgb(9, 8, 7), rgb(1, 2, 3)];
    let indexed = indexed.checked().expect("indexed");
    let pal = ffmpeg_raw(&ffmpeg, &encode_bytes(&indexed, false), "pal8");
    assert_eq!(&pal[..2], &[1, 0]);
    let foreign_idx = ffmpeg_png(&ffmpeg, &[1, 0], 2, 1, "pal8", None);
    let decoded_idx = decode_bytes(&foreign_idx).expect("pal8");
    assert_eq!(decoded_idx.samples, vec![1, 0]);
    assert_eq!(decoded_idx.color, ColorType::Indexed);

    let Some(magick) = tool("magick") else {
        require_or_skip("magick");
        return;
    };
    let gray = still(4, 2, ColorType::Gray, BitDepth::Eight, ramp(8));
    let png = encode_bytes(&gray, true);
    let raw = magick_gray(&magick, &png);
    assert_eq!(raw, packed8(&gray.samples));
    let adam = magick_png(&magick, &packed8(&gray.samples), 4, 2);
    let back = decode_bytes(&adam).expect("magick png");
    assert_eq!(back.samples, gray.samples);
}

#[test]
fn ffmpeg_sixteen_bit_truecolor_matches_exact_samples() {
    let Some(ffmpeg) = tool("ffmpeg") else {
        require_or_skip("ffmpeg");
        return;
    };
    let png = ffmpeg_png(
        &ffmpeg,
        &[0x00, 0x01, 0x02, 0x03, 0xff, 0xfe],
        1,
        1,
        "rgb48be",
        None,
    );
    let image = decode_bytes(&png).expect("rgb48");
    assert_eq!(image.color, ColorType::Rgb);
    assert_eq!(image.depth, BitDepth::Sixteen);
    assert_eq!(image.samples, vec![0x0001, 0x0203, 0xfffe]);
}

#[test]
fn ffmpeg_truecolor_alpha_matches_exact_samples() {
    let Some(ffmpeg) = tool("ffmpeg") else {
        require_or_skip("ffmpeg");
        return;
    };
    let png = ffmpeg_png(&ffmpeg, &[9, 8, 7, 6, 5, 4, 3, 2], 2, 1, "rgba", None);
    let image = decode_bytes(&png).expect("rgba");
    assert_eq!(image.color, ColorType::Rgba);
    assert_eq!(image.depth, BitDepth::Eight);
    assert_eq!(image.samples, vec![9, 8, 7, 6, 5, 4, 3, 2]);
}

#[test]
fn ffmpeg_grey_alpha_matches_exact_samples() {
    let Some(ffmpeg) = tool("ffmpeg") else {
        require_or_skip("ffmpeg");
        return;
    };
    let png = ffmpeg_png(&ffmpeg, &[8, 9, 10, 11], 2, 1, "ya8", None);
    let image = decode_bytes(&png).expect("ya8");
    assert_eq!(image.color, ColorType::GrayAlpha);
    assert_eq!(image.depth, BitDepth::Eight);
    assert_eq!(image.samples, vec![8, 9, 10, 11]);
}

#[test]
fn ffmpeg_sixteen_bit_grey_alpha_matches_exact_samples() {
    let Some(ffmpeg) = tool("ffmpeg") else {
        require_or_skip("ffmpeg");
        return;
    };
    let png = ffmpeg_png(
        &ffmpeg,
        &[0x00, 0x10, 0x01, 0x00, 0xff, 0xff, 0x00, 0x01],
        2,
        1,
        "ya16be",
        None,
    );
    let image = decode_bytes(&png).expect("ya16");
    assert_eq!(image.color, ColorType::GrayAlpha);
    assert_eq!(image.depth, BitDepth::Sixteen);
    assert_eq!(image.samples, vec![0x0010, 0x0100, 0xffff, 0x0001]);
}

#[test]
fn ffmpeg_sixteen_bit_truecolor_alpha_matches_exact_samples() {
    let Some(ffmpeg) = tool("ffmpeg") else {
        require_or_skip("ffmpeg");
        return;
    };
    let png = ffmpeg_png(
        &ffmpeg,
        &[0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0xff],
        1,
        1,
        "rgba64be",
        None,
    );
    let image = decode_bytes(&png).expect("rgba64");
    assert_eq!(image.color, ColorType::Rgba);
    assert_eq!(image.depth, BitDepth::Sixteen);
    assert_eq!(image.samples, vec![1, 2, 3, 255]);
}

#[test]
fn ffmpeg_one_bit_grey_matches_exact_samples() {
    let Some(ffmpeg) = tool("ffmpeg") else {
        require_or_skip("ffmpeg");
        return;
    };
    let png = ffmpeg_png(&ffmpeg, &[0b1011_0010], 8, 1, "monob", None);
    let image = decode_bytes(&png).expect("monob");
    assert_eq!(image.color, ColorType::Gray);
    assert_eq!(image.depth, BitDepth::One);
    assert_eq!(image.samples, vec![1, 0, 1, 1, 0, 0, 1, 0]);
}

#[test]
fn a_local_png_matches_ffmpeg_when_the_file_is_present() {
    let Some(path) = local_png() else {
        return;
    };
    let Some(ffmpeg) = tool("ffmpeg") else {
        require_or_skip("ffmpeg");
        return;
    };
    let bytes = std::fs::read(&path).expect("read");
    let image = decode_bytes(&bytes).expect("local");
    let foreign = ffmpeg_rgba(&ffmpeg, &bytes);
    assert_eq!(image.color, ColorType::Rgba);
    assert_eq!(image.depth, BitDepth::Eight);
    assert_eq!(packed8(&image.samples), foreign);
    let encoded = encode_bytes(&image, false);
    let again = ffmpeg_rgba(&ffmpeg, &encoded);
    assert_eq!(again, foreign);
}

fn assert_round_trip(image: &StillImage, interlace: bool) {
    let back = round_trip(image, interlace);
    assert_eq!(back.samples, image.samples);
    assert_eq!(back.color, image.color);
    assert_eq!(back.depth, image.depth);
    assert_eq!(back.palette, image.palette);
    assert_eq!(back.transparency, image.transparency);
}

fn round_trip(image: &StillImage, interlace: bool) -> StillImage {
    decode_bytes(&encode_bytes(image, interlace)).expect("round trip")
}

fn encode_bytes(image: &StillImage, interlace: bool) -> Vec<u8> {
    let mut encoder = PngEncoder::open(&png_params(interlace)).expect("open");
    let packets = encoder
        .push_picture(&PictureInput {
            image: image.clone(),
        })
        .expect("encode");
    packets[0].data.clone()
}

fn decode_bytes(bytes: &[u8]) -> Result<StillImage, cutlass_decode::CodecError> {
    let mut decoder = PngDecoder::open(&png_params(false)).expect("open");
    let packet = Packet::audio(bytes.to_vec(), 0, 0, 0);
    Ok(decoder.decode_picture(&packet)?.remove(0).image)
}

fn png_params(interlace: bool) -> CodecParams {
    let mut params = CodecParams::audio(CodecId::Png, 1, cutlass_decode::ChannelLayout::mono());
    params.codec = CodecId::Png;
    if interlace {
        params.extra = vec![1];
    }
    params
}

fn still(
    width: u32,
    height: u32,
    color: ColorType,
    depth: BitDepth,
    samples: Vec<u16>,
) -> StillImage {
    StillImage::from_samples(width, height, color, depth, samples)
        .checked()
        .expect("still")
}

fn rgb(red: u8, green: u8, blue: u8) -> Rgb8 {
    Rgb8 { red, green, blue }
}

fn ihdr(width: u32, height: u32, depth: u8, color: u8, interlace: u8) -> [u8; 13] {
    let mut data = [0u8; 13];
    data[0..4].copy_from_slice(&width.to_be_bytes());
    data[4..8].copy_from_slice(&height.to_be_bytes());
    data[8] = depth;
    data[9] = color;
    data[12] = interlace;
    data
}

fn phys(x: u32, y: u32) -> [u8; 9] {
    let mut data = [0u8; 9];
    data[0..4].copy_from_slice(&x.to_be_bytes());
    data[4..8].copy_from_slice(&y.to_be_bytes());
    data[8] = 1;
    data
}

fn chromaticity_values() -> [u32; 8] {
    [31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000]
}

fn chromaticity() -> [u8; 32] {
    let mut data = [0u8; 32];
    for (index, value) in chromaticity_values().iter().enumerate() {
        data[index * 4..index * 4 + 4].copy_from_slice(&value.to_be_bytes());
    }
    data
}

fn png_of(header: [u8; 13], filtered: &[u8], extra: &[u8]) -> Vec<u8> {
    let mut out = SIGNATURE.to_vec();
    out.extend(chunk(b"IHDR", &header));
    out.extend_from_slice(extra);
    out.extend(chunk(b"IDAT", &stored_zlib(filtered)));
    out.extend(chunk(b"IEND", &[]));
    out
}

fn chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&u32::try_from(data.len()).unwrap_or(0).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let crc = crc32_update(crc32(kind), data);
    out.extend_from_slice(&crc.to_be_bytes());
    out
}

fn stored_zlib(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01, 0x01];
    let len = u16::try_from(data.len()).unwrap_or(0);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&(!len).to_le_bytes());
    out.extend_from_slice(data);
    out.extend_from_slice(&adler(data).to_be_bytes());
    out
}

fn adler(data: &[u8]) -> u32 {
    let mut s1 = 1u32;
    let mut s2 = 0u32;
    for &byte in data {
        s1 = (s1 + u32::from(byte)) % 65_521;
        s2 = (s2 + s1) % 65_521;
    }
    (s2 << 16) | s1
}

fn packed8(samples: &[u16]) -> Vec<u8> {
    samples.iter().map(|sample| *sample as u8).collect()
}

fn ramp(len: usize) -> Vec<u16> {
    (0..len).map(|index| (index as u16) % 251).collect()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.0
    }
}

fn random_image(rng: &mut Rng) -> StillImage {
    let width = 1 + (rng.next() % 6) as u32;
    let height = 1 + (rng.next() % 6) as u32;
    let (color, depth) = match rng.next() % 6 {
        0 => (ColorType::Gray, BitDepth::Eight),
        1 => (ColorType::Gray, BitDepth::One),
        2 => (ColorType::Rgb, BitDepth::Eight),
        3 => (ColorType::Rgba, BitDepth::Eight),
        4 => (ColorType::GrayAlpha, BitDepth::Sixteen),
        _ => (ColorType::Indexed, BitDepth::Four),
    };
    let channels = color.channels();
    let count = (width * height) as usize * channels;
    let max = u32::from(depth.max_sample());
    let samples = (0..count)
        .map(|_| (rng.next() as u32 % (max + 1)) as u16)
        .collect();
    let mut image = StillImage::from_samples(width, height, color, depth, samples);
    if color == ColorType::Indexed {
        let entries = 1usize << depth.bits();
        image.palette = (0..entries).map(|index| rgb(index as u8, 2, 3)).collect();
        for sample in &mut image.samples {
            *sample %= entries as u16;
        }
    }
    image.checked().expect("random")
}

trait Samples {
    fn samples(&self) -> &[u16];
}

impl Samples for cutlass_decode::PictureOutput {
    fn samples(&self) -> &[u16] {
        &self.image.samples
    }
}

fn tool(name: &str) -> Option<PathBuf> {
    let path = Command::new("which")
        .arg(name)
        .output()
        .ok()
        .filter(|output| output.status.success())?;
    let text = String::from_utf8(path.stdout).ok()?;
    let line = text.lines().next()?.trim();
    if line.is_empty() {
        None
    } else {
        Some(PathBuf::from(line))
    }
}

fn require_or_skip(name: &str) {
    if std::env::var_os("PNG_REQUIRE_ORACLES").is_some() {
        panic!("{name} is required");
    }
}

fn ffmpeg_png(
    ffmpeg: &PathBuf,
    raw: &[u8],
    width: u32,
    height: u32,
    pix: &str,
    pred: Option<&str>,
) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!("cutlass-png-ff-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("in.raw");
    let output = dir.join("out.png");
    std::fs::write(&input, raw).unwrap();
    let mut command = Command::new(ffmpeg);
    command
        .args([
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix,
            "-s",
            &format!("{width}x{height}"),
            "-i",
        ])
        .arg(&input);
    if let Some(pred) = pred {
        command.args(["-pred", pred]);
    }
    let status = command.arg(&output).output().unwrap();
    assert!(status.status.success(), "{status:?}");
    let bytes = std::fs::read(&output).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

fn ffmpeg_rgba(ffmpeg: &PathBuf, png: &[u8]) -> Vec<u8> {
    ffmpeg_raw(ffmpeg, png, "rgba")
}

fn ffmpeg_raw(ffmpeg: &PathBuf, png: &[u8], pix: &str) -> Vec<u8> {
    let mut child = Command::new(ffmpeg)
        .args([
            "-v", "error", "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", pix, "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(png).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "{output:?}");
    output.stdout
}

fn magick_png(magick: &PathBuf, raw: &[u8], width: u32, height: u32) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!("cutlass-png-magick-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("in.gray");
    let output = dir.join("out.png");
    std::fs::write(&input, raw).unwrap();
    let status = Command::new(magick)
        .args(["-size", &format!("{width}x{height}"), "-depth", "8"])
        .arg(format!("gray:{}", input.display()))
        .args(["-interlace", "PNG"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(status.status.success(), "{status:?}");
    let bytes = std::fs::read(&output).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

fn magick_gray(magick: &PathBuf, png: &[u8]) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!("cutlass-png-gray-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("in.png");
    let output = dir.join("out.gray");
    std::fs::write(&input, png).unwrap();
    let status = Command::new(magick)
        .arg(&input)
        .args(["-depth", "8"])
        .arg(format!("gray:{}", output.display()))
        .output()
        .unwrap();
    assert!(status.status.success(), "{status:?}");
    let bytes = std::fs::read(&output).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

fn local_png() -> Option<PathBuf> {
    let checkout = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)?
        .to_path_buf();
    let path = checkout.join("../local-files/04-motion-graphics.png");
    path.is_file().then_some(path)
}

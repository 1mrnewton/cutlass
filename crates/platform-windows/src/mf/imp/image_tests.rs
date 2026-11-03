use std::path::PathBuf;

use cutlass_decode::{
    BitDepth, CodecError, CodecId, CodecParams, ColorType, DecodeFactory, Packet, PixelFormat,
    StillImage,
};
use cutlass_encode::{EncodeFactory, PictureInput};

use super::image::{decode_bytes, encode_image, open_decoder, open_encoder};

fn still(
    width: u32,
    height: u32,
    depth: BitDepth,
    pixel: impl Fn(u32, u32) -> [u16; 4],
) -> StillImage {
    let mut samples = Vec::new();
    for y in 0..height {
        for x in 0..width {
            samples.extend(pixel(x, y));
        }
    }
    StillImage::from_samples(width, height, ColorType::Rgba, depth, samples)
        .checked()
        .expect("still")
}

fn params(codec: CodecId) -> CodecParams {
    CodecParams::with_format(codec, 0, 0, PixelFormat::Rgba8)
}

fn packet(data: Vec<u8>) -> Packet {
    Packet {
        stream: 0,
        data,
        pts: 3,
        duration: 1,
        timescale: 1,
        keyframe: true,
        side: Vec::new(),
    }
}

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../engine/fixtures")
        .join(name);
    std::fs::read(path).expect("fixture")
}

#[test]
fn a_png_round_trips_exactly_with_alpha() {
    let image = still(5, 3, BitDepth::Eight, |x, y| {
        [x as u16 * 40, y as u16 * 80, 200, 50 + x as u16 * 30]
    });
    let mut encoder = open_encoder(&params(CodecId::Png)).expect("open");
    let packets = encoder
        .push_picture(&PictureInput {
            image: image.clone(),
        })
        .expect("encode");
    assert!(packets[0].data.starts_with(b"\x89PNG"));
    let mut decoder = open_decoder(&params(CodecId::Png)).expect("open");
    let decoded = decoder
        .decode_picture(&packet(packets[0].data.clone()))
        .expect("decode");
    assert_eq!(decoded[0].pts, 3);
    assert_eq!(decoded[0].image.samples, image.samples);
    assert_eq!(decoded[0].image.depth, BitDepth::Eight);
}

#[test]
fn a_sixteen_bit_png_keeps_sixteen_bits() {
    let image = still(4, 2, BitDepth::Sixteen, |x, y| {
        [x as u16 * 16_001 + 7, y as u16 * 30_000 + 3, 65_535, 40_000]
    });
    let bytes = encode_image(CodecId::Png, &image).expect("encode");
    assert_eq!(bytes[24], 16, "the IHDR says 16 bits");
    let decoded = decode_bytes(&bytes).expect("decode");
    assert_eq!(decoded.depth, BitDepth::Sixteen);
    assert_eq!(decoded.samples, image.samples);

    // The Rust PNG decoder reads the same file to the same samples.
    let mut rust = DecodeFactory::new();
    cutlass_png::register_decoders(&mut rust);
    let mut decoder = rust.open(&params(CodecId::Png)).expect("rust png");
    let theirs = decoder.decode_picture(&packet(bytes)).expect("rust decode");
    assert_eq!(theirs[0].image.samples, image.samples);
}

#[test]
fn a_rust_png_decodes_on_windows() {
    let image = still(3, 3, BitDepth::Eight, |x, y| {
        [x as u16 * 90, y as u16 * 90, 17, 255]
    });
    let mut rust = EncodeFactory::new();
    cutlass_png::register_encoders(&mut rust);
    let mut encoder = rust.open(&params(CodecId::Png)).expect("rust png");
    let packets = encoder
        .push_picture(&PictureInput {
            image: image.clone(),
        })
        .expect("encode");
    let decoded = decode_bytes(&packets[0].data).expect("decode");
    assert_eq!(decoded.samples, image.samples);
}

#[test]
fn jpeg_keeps_the_size_and_the_red() {
    let image = still(32, 16, BitDepth::Eight, |_, _| [220, 10, 10, 255]);
    let bytes = encode_image(CodecId::Jpeg, &image).expect("encode");
    assert!(bytes.starts_with(&[0xFF, 0xD8]));
    let decoded = decode_bytes(&bytes).expect("decode");
    assert_eq!((decoded.width, decoded.height), (32, 16));
    for pixel in decoded.samples.as_chunks::<4>().0 {
        assert!(
            pixel[0] > 190 && pixel[1] < 40 && pixel[2] < 40 && pixel[3] == 255,
            "{pixel:?}"
        );
    }
}

#[test]
fn the_pattern_fixture_matches_the_rust_jpeg_decoder() {
    let bytes = fixture("pattern.jpg");
    let ours = decode_bytes(&bytes).expect("wic");
    let mut rust = DecodeFactory::new();
    cutlass_jpeg::register_decoders(&mut rust);
    let mut decoder = rust.open(&params(CodecId::Jpeg)).expect("rust jpeg");
    let theirs = decoder.decode_picture(&packet(bytes)).expect("rust decode");
    let theirs = &theirs[0].image;
    assert_eq!((ours.width, ours.height), (theirs.width, theirs.height));
    let channels = match theirs.color {
        ColorType::Rgb => 3,
        ColorType::Rgba => 4,
        ColorType::Gray => 1,
        other => panic!("{other:?}"),
    };
    let mut worst = 0;
    for (mine, other) in ours
        .samples
        .as_chunks::<4>()
        .0
        .iter()
        .zip(theirs.samples.chunks_exact(channels))
    {
        for channel in 0..3 {
            let want = other[channel.min(channels - 1)];
            worst = worst.max(mine[channel].abs_diff(want));
        }
    }
    eprintln!("pattern.jpg: worst sample difference {worst}");
    assert!(worst <= 8, "worst sample difference {worst}");
}

#[test]
fn exif_orientation_turns_the_picture_upright() {
    // Left half red, right half blue, then tagged "rotate 90 clockwise".
    let image = still(4, 2, BitDepth::Eight, |x, _| {
        if x < 2 {
            [255, 0, 0, 255]
        } else {
            [0, 0, 255, 255]
        }
    });
    let plain = encode_image(CodecId::Jpeg, &image).expect("encode");
    let exif: &[u8] = &[
        b'E', b'x', b'i', b'f', 0, 0, b'I', b'I', 42, 0, 8, 0, 0, 0, 1, 0, 0x12, 0x01, 3, 0, 1, 0,
        0, 0, 6, 0, 0, 0, 0, 0, 0, 0,
    ];
    let mut tagged = vec![0xFF, 0xD8, 0xFF, 0xE1];
    tagged.extend_from_slice(&u16::try_from(exif.len() + 2).unwrap().to_be_bytes());
    tagged.extend_from_slice(exif);
    tagged.extend_from_slice(&plain[2..]);
    let decoded = decode_bytes(&tagged).expect("decode");
    assert_eq!(
        (decoded.width, decoded.height),
        (2, 4),
        "turned on its side"
    );
    // Turned clockwise, the left half (red) is now the top.
    let top = &decoded.samples[0..4];
    let bottom = &decoded.samples[decoded.samples.len() - 4..];
    assert!(top[0] > 200 && top[2] < 60, "top {top:?}");
    assert!(bottom[2] > 200 && bottom[0] < 60, "bottom {bottom:?}");
}

#[test]
fn hostile_and_truncated_files_fail_closed() {
    for name in ["hostile.jpg", "truncated.jpg"] {
        let result = decode_bytes(&fixture(name));
        eprintln!(
            "{name}: {:?}",
            result.as_ref().map(|image| (image.width, image.height))
        );
        if let Ok(image) = result {
            assert_eq!(
                image.samples.len(),
                image.width as usize * image.height as usize * 4,
                "{name} decoded to a whole picture"
            );
        }
    }
    let mut state = 0x1234_5678_u32;
    for _ in 0..32 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let len = (state % 64) as usize + 1;
        let bytes: Vec<u8> = (0..len)
            .map(|index| (state.wrapping_add(index as u32) % 256) as u8)
            .collect();
        assert!(decode_bytes(&bytes).is_err());
    }
}

#[test]
fn indexed_and_empty_input_are_refused() {
    let mut encoder = open_encoder(&params(CodecId::Png)).expect("open");
    let mut indexed = still(1, 1, BitDepth::Eight, |_, _| [0, 0, 0, 0]);
    indexed.color = ColorType::Indexed;
    indexed.samples = vec![0];
    indexed.palette = vec![cutlass_decode::Rgb8 {
        red: 1,
        green: 2,
        blue: 3,
    }];
    assert!(matches!(
        encoder.push_picture(&PictureInput { image: indexed }),
        Err(CodecError::Unsupported { .. })
    ));
    let mut decoder = open_decoder(&params(CodecId::Jpeg)).expect("open");
    assert_eq!(
        decoder.decode_picture(&packet(Vec::new())),
        Err(CodecError::InvalidApi {
            what: "empty packet"
        })
    );
    assert!(matches!(
        open_decoder(&params(CodecId::Webp)),
        Err(CodecError::Unsupported { .. })
    ));
}

//! Hostile and broken input: every case returns an error, never a panic,
//! a hang or an allocation the limits do not allow. Streams are built bit
//! by bit with the encoder's writer so each one breaks exactly one rule.

use cutlass_decode::{CodecError, CodecId, CodecParams, Decoder, Packet};

use super::lossy_encode::lossy_file;
use super::oracle::synthetic_rgba;
use crate::WebpDecoder;
use crate::lossless::bits::BitWriter;
use crate::lossless::codes::{Code, write_code};
use crate::lossless::decode::decode as decode_vp8l;
use crate::lossless::encode::{Config, encode as encode_vp8l};
use crate::riff::{self, Bitstream, write_still};

fn decoder() -> WebpDecoder {
    WebpDecoder::open(&CodecParams::video(CodecId::Webp, 0, 0)).expect("decoder")
}

fn decode(bytes: &[u8]) -> Result<usize, CodecError> {
    decoder()
        .decode_picture(&Packet::video(bytes.to_vec(), 0, 0, 0, true))
        .map(|pictures| pictures.len())
}

fn lossless_still(vp8l: &[u8], width: u32, height: u32) -> Vec<u8> {
    write_still(Bitstream::Lossless(vp8l), width, height, false, None)
}

/// The five-byte VP8L header of an opaque `width x height` picture.
fn header(writer: &mut BitWriter, width: u32, height: u32) {
    writer.write(0x2f, 8);
    writer.write(width - 1, 14);
    writer.write(height - 1, 14);
    writer.write(0, 1);
    writer.write(0, 3);
}

/// The five prefix codes of one group, each with one symbol except green,
/// which has `green` (length one each). Every copy uses distance symbol
/// `distance`. Returns the green code.
fn group(writer: &mut BitWriter, green_alphabet: usize, green: &[usize], distance: usize) -> Code {
    let mut lengths = vec![0u8; green_alphabet];
    for &symbol in green {
        lengths[symbol] = 1;
    }
    let code = write_code(writer, &lengths);
    for (alphabet, symbol) in [(256, 0), (256, 0), (256, 0), (40, distance)] {
        let mut lengths = vec![0u8; alphabet];
        lengths[symbol] = 1;
        write_code(writer, &lengths);
    }
    code
}

fn assert_malformed(result: Result<impl std::fmt::Debug, CodecError>, what: &str) {
    match result {
        Err(CodecError::Malformed { what: got, .. }) => assert_eq!(got, what),
        other => panic!("expected malformed {what}, got {other:?}"),
    }
}

#[test]
fn a_backward_reference_before_the_first_pixel_is_refused() {
    let mut writer = BitWriter::with_capacity(64);
    header(&mut writer, 4, 1);
    writer.write(0, 1); // no transform
    writer.write(0, 1); // no colour cache
    writer.write(0, 1); // one group
    // Green: literal 0 and length prefix 0 (a copy of one pixel), from
    // distance symbol 1: the pixel to the left.
    let green = group(&mut writer, 280, &[0, 256], 1);
    // The first pixel copies one from before the picture.
    green.put(&mut writer, 256);
    let stream = writer.finish();
    assert_malformed(decode_vp8l(&stream), "vp8l backward reference");
    assert_malformed(
        decode(&lossless_still(&stream, 4, 1)),
        "vp8l backward reference",
    );

    // The same copy after one literal is legal and fills the row.
    let mut writer = BitWriter::with_capacity(64);
    header(&mut writer, 4, 1);
    writer.write(0, 3);
    let green = group(&mut writer, 280, &[0, 256], 1);
    green.put(&mut writer, 0);
    for _ in 0..3 {
        green.put(&mut writer, 256);
    }
    let (_, pixels) = decode_vp8l(&writer.finish()).expect("legal copy");
    assert_eq!(pixels, vec![0; 4]);
}

#[test]
fn a_copy_past_the_last_pixel_is_refused() {
    let mut writer = BitWriter::with_capacity(64);
    header(&mut writer, 2, 1);
    writer.write(0, 3);
    // Length prefix 4 copies five pixels: past the end of a 2x1 picture.
    let green = group(&mut writer, 280, &[0, 260], 1);
    green.put(&mut writer, 0);
    green.put(&mut writer, 260);
    writer.write(0, 1); // the length's extra bit
    assert_malformed(decode_vp8l(&writer.finish()), "vp8l backward reference");
}

#[test]
fn an_over_subscribed_code_length_code_is_refused() {
    let mut writer = BitWriter::with_capacity(64);
    header(&mut writer, 1, 1);
    writer.write(0, 3);
    writer.write(0, 1); // a normal code
    writer.write(0, 4); // four code length code lengths: 17, 18, 0, 1
    for _ in 0..4 {
        writer.write(1, 3); // all of length one: over-subscribed
    }
    let result = decode_vp8l(&writer.finish());
    assert!(
        matches!(result, Err(CodecError::Malformed { .. })),
        "{result:?}"
    );
}

#[test]
fn a_repeated_transform_is_refused() {
    let mut writer = BitWriter::with_capacity(64);
    header(&mut writer, 1, 1);
    for _ in 0..2 {
        writer.write(1, 1); // a transform
        writer.write(2, 2); // subtract green
    }
    writer.write(0, 1);
    assert_malformed(decode_vp8l(&writer.finish()), "vp8l transform repeated");
}

#[test]
fn a_colour_cache_outside_one_to_eleven_bits_is_refused() {
    for bits in [0, 12, 15] {
        let mut writer = BitWriter::with_capacity(64);
        header(&mut writer, 1, 1);
        writer.write(0, 1);
        writer.write(1, 1);
        writer.write(bits, 4);
        assert_malformed(decode_vp8l(&writer.finish()), "vp8l color cache size");
    }
    // A cache index is a green symbol, so the alphabet bounds it: with a
    // two-entry cache, index 1 is the last symbol, and reads back the colour
    // the first pixel put in the cache.
    let mut writer = BitWriter::with_capacity(64);
    header(&mut writer, 2, 1);
    writer.write(0, 1);
    writer.write(1, 1);
    writer.write(1, 4);
    writer.write(0, 1);
    let green = group(&mut writer, 282, &[0, 281], 0);
    green.put(&mut writer, 0);
    green.put(&mut writer, 281);
    let result = decode_vp8l(&writer.finish());
    // The black pixel hashes to slot 0, so slot 1 is still empty (zero).
    assert_eq!(result.expect("in range").1, vec![0, 0]);
}

#[test]
fn every_truncation_of_a_bitstream_is_an_error() {
    let rgba = synthetic_rgba(23, 17);
    let argb: Vec<u32> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| u32::from_be_bytes([p[3], p[0], p[1], p[2]]))
        .collect();
    let vp8l = encode_vp8l(
        &argb,
        23,
        17,
        Config {
            effort: 5,
            exact: false,
        },
    );
    for end in 0..vp8l.len() - 1 {
        let file = write_still(Bitstream::Lossless(&vp8l[..end]), 23, 17, true, None);
        assert!(decode(&file).is_err(), "VP8L cut at {end}");
    }

    let file = lossy_file(&rgba, 23, 17, 60, 4);
    let chunks = riff::chunks(&file);
    let (vp8, alpha) = (&chunks[2].1, &chunks[1].1);
    assert_eq!((&chunks[1].0, &chunks[2].0), (b"ALPH", b"VP8 "));
    for end in 0..vp8.len() - 1 {
        let cut = Bitstream::Lossy {
            vp8: &vp8[..end],
            alpha: Some(alpha),
        };
        let result = decode(&write_still(cut, 23, 17, true, None));
        assert!(result.is_err(), "VP8 cut at {end}");
    }
    for end in 0..alpha.len() - 1 {
        let cut = Bitstream::Lossy {
            vp8,
            alpha: Some(&alpha[..end]),
        };
        assert!(
            decode(&write_still(cut, 23, 17, true, None)).is_err(),
            "ALPH cut at {end}"
        );
    }
    // And every cut of the whole file.
    for end in 0..file.len() - 1 {
        assert!(decode(&file[..end]).is_err(), "file cut at {end}");
    }
}

#[test]
fn a_partition_past_the_end_and_a_bool_decoder_overrun_are_errors() {
    let rgba = synthetic_rgba(40, 40);
    let file = lossy_file(&rgba, 40, 40, 80, 4);
    let chunks = riff::chunks(&file);
    let vp8 = chunks
        .iter()
        .find(|(fourcc, _)| fourcc == b"VP8 ")
        .map(|(_, data)| data.clone())
        .expect("vp8");
    let first = (u32::from_le_bytes([vp8[0], vp8[1], vp8[2], 0]) >> 5) as usize;
    let with = |vp8: &[u8]| {
        let alpha = &chunks[1].1;
        decode(&write_still(
            Bitstream::Lossy {
                vp8,
                alpha: Some(alpha),
            },
            40,
            40,
            true,
            None,
        ))
    };
    assert_eq!(with(&vp8).expect("intact"), 1);

    // The first partition claims more bytes than the frame has.
    let mut long = vp8.clone();
    let tag = (u32::from_le_bytes([long[0], long[1], long[2], 0]) & 0x1f) | ((1 << 19) - 1) << 5;
    long[..3].copy_from_slice(&tag.to_le_bytes()[..3]);
    assert!(matches!(
        with(&long),
        Err(CodecError::Truncated { .. } | CodecError::Malformed { .. })
    ));

    // The token partition stops early: the bool decoder runs past its end.
    let short = &vp8[..10 + first + (vp8.len() - 10 - first) / 2];
    assert!(matches!(with(short), Err(CodecError::Truncated { .. })));

    // Not a key frame.
    let mut inter = vp8.clone();
    inter[0] |= 1;
    assert!(with(&inter).is_err());
}

#[test]
fn pictures_and_canvases_past_the_limits_are_refused_before_allocating() {
    // A VP8L header for 16383 x 16383 and nothing else: legal WebP, over
    // the pixel cap.
    let mut writer = BitWriter::with_capacity(8);
    header(&mut writer, 16_383, 16_383);
    let bomb = writer.finish();
    assert!(matches!(decode_vp8l(&bomb), Err(CodecError::Limit { .. })));
    let file = riff::file(&[(b"VP8L", &bomb)]);
    assert!(matches!(decode(&file), Err(CodecError::Limit { .. })));

    // An animation canvas of 2^24 x 2^24 with one tiny frame.
    let mut tiny = BitWriter::with_capacity(16);
    header(&mut tiny, 1, 1);
    tiny.write(0, 3);
    group(&mut tiny, 280, &[0], 0);
    let tiny = tiny.finish();
    let mut anmf = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0];
    riff::put_chunk(&mut anmf, b"VP8L", &tiny);
    let huge = riff::file(&[
        (b"VP8X", &riff::vp8x(0x02, 1 << 24, 1 << 24)),
        (b"ANIM", &[0; 6]),
        (b"ANMF", &anmf),
    ]);
    assert!(matches!(decode(&huge), Err(CodecError::Limit { .. })));

    // More frames in one packet than the cap: refused, not composed.
    let mut chunks: Vec<(&[u8; 4], &[u8])> = vec![
        (b"VP8X", &[0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        (b"ANIM", &[0; 6]),
    ];
    for _ in 0..=crate::limits::MAX_FRAMES_PER_PACKET {
        chunks.push((b"ANMF", &anmf));
    }
    let many = riff::file(&chunks);
    assert!(matches!(decode(&many), Err(CodecError::Limit { .. })));
}

#[test]
fn a_frame_outside_the_canvas_is_refused() {
    let mut tiny = BitWriter::with_capacity(16);
    header(&mut tiny, 2, 2);
    tiny.write(0, 3);
    group(&mut tiny, 280, &[0], 0);
    let tiny = tiny.finish();
    // x = 2 * 2 = 4 on a 4 x 4 canvas, so the 2 x 2 frame runs to 6.
    let mut anmf = vec![2, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 100, 0, 0, 0];
    riff::put_chunk(&mut anmf, b"VP8L", &tiny);
    let file = riff::file(&[
        (b"VP8X", &riff::vp8x(0x02, 4, 4)),
        (b"ANIM", &[0; 6]),
        (b"ANMF", &anmf),
    ]);
    assert_malformed(decode(&file), "webp frame outside the canvas");
}

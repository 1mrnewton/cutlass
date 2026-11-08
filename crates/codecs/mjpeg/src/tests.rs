//! Crate tests: framing, fields, tables, lifecycle, and the encoder.
//!
//! Fixtures here come from the `cutlass-jpeg` encoder and are edited the way
//! a Motion JPEG writer would: DHT left out, tables only in an earlier
//! packet, two fields in one packet, a format A header, a comment. Frames
//! from independent encoders are in `oracle.rs`.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use cutlass_decode::video::VideoDecoder;
use cutlass_decode::{
    CodecError, CodecId, CodecParams, ColorRange, DecodeFactory, Decoder, MatrixCoefficients,
    Output, Packet, PixelFormat, Plane, VideoDecodeFactory, VideoFrame,
};
use cutlass_encode::video::{VideoEncoder, VideoInput};
use cutlass_encode::{CpuFrame, EncodeFactory, Encoder, FrameInput, FrameTime};
use cutlass_jpeg::{Coefficients, Planes, Sampling, Tables, WriteSettings, decode_image};

use crate::{
    FieldOrder, MAX_PACKET, MjpegDecoder, MjpegEncoder, MjpegPictureDecoder, RateMode,
    register_decoders, register_encoders, register_video_decoders,
};

pub(crate) fn params(width: u32, height: u32, extra: &[u8]) -> CodecParams {
    let mut params = CodecParams::video(CodecId::Mjpeg, width, height);
    params.extra = extra.to_vec();
    params
}

/// A picture with edges, gradients, and noise, so every coefficient band
/// and every chroma filter is exercised.
pub(crate) fn detailed_rgb(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = (state % 23) as i32 - 11;
            let edge = if (x / 7 + y / 5) % 2 == 0 { 60 } else { -60 };
            let r = (x * 255 / width.max(1)) as i32 + noise;
            let g = (y * 255 / height.max(1)) as i32 + edge / 2;
            let b = ((x + y + seed) % 256) as i32 - noise;
            for value in [r, g, b] {
                out.push(value.clamp(0, 255) as u8);
            }
        }
    }
    out
}

pub(crate) fn encode(
    rgb: &[u8],
    width: u32,
    height: u32,
    sampling: Sampling,
    quality: u8,
) -> Vec<u8> {
    let mut planes = Planes::default();
    planes
        .from_rgb(rgb, 3, width, height, sampling)
        .expect("planes");
    let mut coefficients = Coefficients::new();
    coefficients
        .transform(&planes.source().expect("source"))
        .expect("transform");
    let mut out = Vec::new();
    coefficients
        .write(
            &WriteSettings {
                quality,
                restart_interval: 0,
                jfif: true,
            },
            &mut out,
        )
        .expect("write");
    out
}

/// The marker segments before the first scan, as (marker, start, end).
pub(crate) fn segments(jpeg: &[u8]) -> Vec<(u8, usize, usize)> {
    let mut out = Vec::new();
    let mut at = 2;
    while jpeg[at + 1] != 0xda {
        let len = usize::from(u16::from_be_bytes([jpeg[at + 2], jpeg[at + 3]]));
        out.push((jpeg[at + 1], at, at + 2 + len));
        at += 2 + len;
    }
    out
}

/// `jpeg` without its segments of the markers in `drop`.
pub(crate) fn strip(jpeg: &[u8], drop: &[u8]) -> Vec<u8> {
    let mut out = jpeg[..2].to_vec();
    let mut scan_at = 2;
    for (marker, start, end) in segments(jpeg) {
        if !drop.contains(&marker) {
            out.extend_from_slice(&jpeg[start..end]);
        }
        scan_at = end;
    }
    out.extend_from_slice(&jpeg[scan_at..]);
    out
}

/// `jpeg` with a segment inserted after SOI.
pub(crate) fn with_segment(jpeg: &[u8], marker: u8, body: &[u8]) -> Vec<u8> {
    let mut out = jpeg[..2].to_vec();
    out.extend_from_slice(&[0xff, marker]);
    out.extend_from_slice(&u16::try_from(body.len() + 2).expect("len").to_be_bytes());
    out.extend_from_slice(body);
    out.extend_from_slice(&jpeg[2..]);
    out
}

fn packet(data: Vec<u8>, pts: i64) -> Packet {
    Packet::video(data, pts, 1001, 30_000, true)
}

/// Rows `first`, `first + 2`, ... of an interleaved picture.
pub(crate) fn field_rows(rgb: &[u8], width: u32, height: u32, first: u32) -> (Vec<u8>, u32) {
    let row = (width * 3) as usize;
    let rows: Vec<u32> = (first..height).step_by(2).collect();
    let mut out = Vec::with_capacity(rows.len() * row);
    for y in &rows {
        out.extend_from_slice(&rgb[*y as usize * row..(*y as usize + 1) * row]);
    }
    (out, u32::try_from(rows.len()).expect("rows"))
}

/// Even rows from `a`, odd rows from `b`, cut to `rows` rows of `row` bytes.
pub(crate) fn interleave(a: &[u8], b: &[u8], row: usize, rows: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(row * rows);
    for index in 0..rows {
        let field = if index % 2 == 0 { a } else { b };
        out.extend_from_slice(&field[(index / 2) * row..(index / 2 + 1) * row]);
    }
    out
}

fn rgba_of(decoder: &mut MjpegDecoder, data: &[u8]) -> Vec<u8> {
    let picture = decoder.decode(data).expect("decode");
    let mut rgba = Vec::new();
    picture.to_rgba(&mut rgba).expect("rgba");
    rgba
}

fn image_rgba(jpeg: &[u8]) -> Vec<u8> {
    let mut tables = Tables::standard_huffman();
    let decoded = decode_image(jpeg, &mut tables).expect("image");
    let mut rgba = Vec::new();
    decoded.image.to_rgba(&mut rgba).expect("rgba");
    rgba
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse = a
        .iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    10.0 * (255.0 * 255.0 / mse.max(1e-9)).log10()
}

fn rgb_of_rgba(rgba: &[u8]) -> Vec<u8> {
    rgba.as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect()
}

#[test]
fn a_frame_without_dht_uses_the_annex_k_tables() {
    let rgb = detailed_rgb(48, 32, 1);
    let full = encode(&rgb, 48, 32, Sampling::Yuv420, 80);
    let bare = strip(&full, &[0xc4]);
    assert!(bare.len() < full.len());
    let mut decoder = MjpegDecoder::open(&params(48, 32, &[])).expect("open");
    assert_eq!(rgba_of(&mut decoder, &bare), image_rgba(&full));
    // The still decoder does not assume tables a Motion JPEG frame may omit.
    assert!(matches!(
        decode_image(&bare, &mut Tables::new()),
        Err(CodecError::Malformed { .. })
    ));
}

#[test]
fn tables_carry_to_an_abbreviated_packet_and_reset_drops_them() {
    let first = encode(&detailed_rgb(32, 16, 2), 32, 16, Sampling::Yuv422, 60);
    let second_full = encode(&detailed_rgb(32, 16, 3), 32, 16, Sampling::Yuv422, 60);
    let second = strip(&second_full, &[0xdb, 0xc4]);
    let mut decoder = MjpegDecoder::open(&params(32, 16, &[])).expect("open");
    assert!(matches!(
        decoder.decode(&second),
        Err(CodecError::Malformed { .. })
    ));
    decoder.decode(&first).expect("first");
    assert_eq!(rgba_of(&mut decoder, &second), image_rgba(&second_full));
    VideoDecoder::reset(&mut decoder);
    assert!(matches!(
        decoder.decode(&second),
        Err(CodecError::Malformed { .. })
    ));
}

#[test]
fn tables_in_the_extradata_serve_every_packet() {
    let full = encode(&detailed_rgb(24, 24, 4), 24, 24, Sampling::Yuv444, 90);
    let mut table_stream = vec![0xff, 0xd8];
    for (marker, start, end) in segments(&full) {
        if matches!(marker, 0xdb | 0xc4) {
            table_stream.extend_from_slice(&full[start..end]);
        }
    }
    table_stream.extend_from_slice(&[0xff, 0xd9]);
    let abbreviated = strip(&full, &[0xdb, 0xc4]);
    let mut decoder = MjpegDecoder::open(&params(24, 24, &table_stream)).expect("open");
    assert_eq!(rgba_of(&mut decoder, &abbreviated), image_rgba(&full));
    VideoDecoder::reset(&mut decoder);
    assert_eq!(rgba_of(&mut decoder, &abbreviated), image_rgba(&full));
}

#[test]
fn a_failed_packet_keeps_the_tables_it_did_not_finish() {
    let good = encode(&detailed_rgb(32, 32, 5), 32, 32, Sampling::Yuv420, 85);
    let other_full = encode(&detailed_rgb(32, 32, 6), 32, 32, Sampling::Yuv420, 85);
    let abbreviated = strip(&other_full, &[0xdb]);
    // New tables at quality 10, then a scan cut short.
    let coarse = encode(&detailed_rgb(32, 32, 7), 32, 32, Sampling::Yuv420, 10);
    let broken = &coarse[..coarse.len() - 40];
    let mut decoder = MjpegDecoder::open(&params(32, 32, &[])).expect("open");
    decoder.decode(&good).expect("good");
    assert!(decoder.decode(broken).is_err());
    assert_eq!(rgba_of(&mut decoder, &abbreviated), image_rgba(&other_full));
}

struct FieldFixture {
    rgb: Vec<u8>,
    top: Vec<u8>,
    bottom: Vec<u8>,
    field_height: u32,
}

fn field_fixture(width: u32, height: u32, sampling: Sampling) -> FieldFixture {
    let rgb = detailed_rgb(width, height, 8);
    let (top_rgb, top_rows) = field_rows(&rgb, width, height, 0);
    let (bottom_rgb, bottom_rows) = field_rows(&rgb, width, height, 1);
    // Both fields code the same number of rows; an odd frame repeats the
    // last bottom row, as a field encoder pads.
    let mut bottom_rgb = bottom_rgb;
    if bottom_rows < top_rows {
        let row = (width * 3) as usize;
        let last = bottom_rgb[bottom_rgb.len() - row..].to_vec();
        bottom_rgb.extend_from_slice(&last);
    }
    FieldFixture {
        top: encode(&top_rgb, width, top_rows, sampling, 90),
        bottom: encode(&bottom_rgb, width, top_rows, sampling, 90),
        rgb,
        field_height: top_rows,
    }
}

#[test]
fn two_fields_weave_into_the_container_frame() {
    for (height, sampling) in [
        (48, Sampling::Yuv420),
        (47, Sampling::Yuv420),
        (40, Sampling::Yuv422),
        (34, Sampling::Yuv444),
        (30, Sampling::Gray),
    ] {
        let width = 40;
        let fixture = field_fixture(width, height, sampling);
        let mut data = fixture.top.clone();
        data.extend_from_slice(&fixture.bottom);
        let mut decoder = MjpegDecoder::open(&params(width, height, &[])).expect("open");
        let picture = decoder.decode(&data).expect("pair");
        assert_eq!((picture.width, picture.height), (width, height));
        assert_eq!(picture.fields, Some(FieldOrder::TopFirst));
        assert_eq!(decoder.last_field_order(), Some(FieldOrder::TopFirst));

        let mut rgba = Vec::new();
        picture.to_rgba(&mut rgba).expect("rgba");
        let want = interleave(
            &image_rgba(&fixture.top),
            &image_rgba(&fixture.bottom),
            width as usize * 4,
            height as usize,
        );
        assert_eq!(rgba, want, "height {height}");

        let frame = picture.into_frame().expect("frame");
        assert_eq!((frame.width, frame.height), (width, height));
        let top = decode_image(&fixture.top, &mut Tables::standard_huffman()).expect("top");
        let bottom =
            decode_image(&fixture.bottom, &mut Tables::standard_huffman()).expect("bottom");
        for (index, plane) in [&frame.y, &frame.u, &frame.v].into_iter().enumerate() {
            let Some(a) = top.image.components.get(index) else {
                break;
            };
            let b = &bottom.image.components[index];
            let want = interleave(
                &a.samples,
                &b.samples,
                a.width as usize,
                plane.height as usize,
            );
            assert_eq!(plane.samples, want, "height {height} plane {index}");
        }
        // Even rows hold the top field: the frame is nearer the source it
        // was split from than the same fields swapped.
        if sampling != Sampling::Gray {
            let swapped = interleave(
                &image_rgba(&fixture.bottom),
                &image_rgba(&fixture.top),
                width as usize * 4,
                height as usize,
            );
            let right = psnr(&rgb_of_rgba(&rgba), &fixture.rgb);
            let wrong = psnr(&rgb_of_rgba(&swapped), &fixture.rgb);
            assert!(right > wrong + 1.0, "height {height}: {right} vs {wrong}");
        }
    }
}

#[test]
fn the_bottom_field_first_comes_from_fiel_then_the_avid_comment() {
    let (width, height) = (32, 32);
    let fixture = field_fixture(width, height, Sampling::Yuv420);
    let mut bottom_first = fixture.bottom.clone();
    bottom_first.extend_from_slice(&fixture.top);
    let want = interleave(
        &image_rgba(&fixture.top),
        &image_rgba(&fixture.bottom),
        width as usize * 4,
        height as usize,
    );

    let mut decoder =
        MjpegDecoder::open(&params(width, height, &[b'j', b'p', b'e', b'g', 2, 6])).expect("open");
    let picture = decoder.decode(&bottom_first).expect("fiel 6");
    assert_eq!(picture.fields, Some(FieldOrder::BottomFirst));
    let mut rgba = Vec::new();
    picture.to_rgba(&mut rgba).expect("rgba");
    assert_eq!(rgba, want);

    let mut avid = b"AVID".to_vec();
    avid.resize(16, 0);
    avid[12] = 1;
    let mut tagged = with_segment(&fixture.bottom, 0xfe, &avid);
    tagged.extend_from_slice(&fixture.top);
    let mut decoder = MjpegDecoder::open(&params(width, height, &[])).expect("open");
    assert_eq!(rgba_of(&mut decoder, &tagged), want);
    assert_eq!(decoder.last_field_order(), Some(FieldOrder::BottomFirst));

    // `fiel` wins over the comment.
    let mut decoder =
        MjpegDecoder::open(&params(width, height, &[b'j', b'p', b'e', b'g', 2, 1])).expect("open");
    decoder.decode(&tagged).expect("fiel 1");
    assert_eq!(decoder.last_field_order(), Some(FieldOrder::TopFirst));

    avid[12] = 2;
    let mut pal = with_segment(&fixture.top, 0xfe, &avid);
    pal.extend_from_slice(&fixture.bottom);
    let mut decoder = MjpegDecoder::open(&params(width, height, &[])).expect("open");
    assert_eq!(rgba_of(&mut decoder, &pal), want);
}

#[test]
fn format_a_headers_and_padding_between_fields_are_skipped() {
    let (width, height) = (32, 32);
    let fixture = field_fixture(width, height, Sampling::Yuv420);
    let mut header = vec![0, 0, 0, 0];
    header.extend_from_slice(b"mjpg");
    header.resize(40, 0);
    let mut data = with_segment(&fixture.top, 0xe1, &header);
    data.extend_from_slice(&[0, 0, 0, 0]);
    data.extend_from_slice(&with_segment(&fixture.bottom, 0xe1, &header));
    let mut decoder = MjpegDecoder::open(&params(width, height, b"mjpa")).expect("open");
    let want = interleave(
        &image_rgba(&fixture.top),
        &image_rgba(&fixture.bottom),
        width as usize * 4,
        height as usize,
    );
    assert_eq!(rgba_of(&mut decoder, &data), want);
}

#[test]
fn field_packets_that_do_not_pair_are_refused() {
    let (width, height) = (32, 32);
    let fixture = field_fixture(width, height, Sampling::Yuv420);
    let mut decoder = MjpegDecoder::open(&params(width, height, &[])).expect("open");
    assert!(matches!(
        decoder.decode(&fixture.top),
        Err(CodecError::Unsupported {
            what: "one field per packet"
        })
    ));
    let other = encode(
        &detailed_rgb(width, fixture.field_height, 9),
        width,
        fixture.field_height,
        Sampling::Yuv444,
        90,
    );
    let mut mixed = fixture.top.clone();
    mixed.extend_from_slice(&other);
    assert!(matches!(
        decoder.decode(&mixed),
        Err(CodecError::Malformed {
            what: "second field",
            ..
        })
    ));
    let mut cut = fixture.top.clone();
    cut.extend_from_slice(&fixture.bottom[..fixture.bottom.len() / 2]);
    assert!(decoder.decode(&cut).is_err());

    // Without a container height, or at a height that is not a field pair,
    // the first image is the frame.
    let mut pair = fixture.top.clone();
    pair.extend_from_slice(&fixture.bottom);
    for (w, h) in [
        (0, 0),
        (width, fixture.field_height),
        (width, 2 * fixture.field_height + 1),
    ] {
        let mut decoder = MjpegDecoder::open(&params(w, h, &[])).expect("open");
        let picture = decoder.decode(&pair).expect("frame");
        assert_eq!(picture.fields, None);
        assert_eq!(picture.height, fixture.field_height);
    }
}

#[test]
fn format_b_and_foreign_extradata_are_refused() {
    assert!(matches!(
        MjpegDecoder::open(&params(16, 16, b"mjpb")),
        Err(CodecError::Unsupported {
            what: "motion jpeg format b"
        })
    ));
    let mut decoder = MjpegDecoder::open(&params(16, 16, &[])).expect("open");
    let mut data = vec![0, 0, 0, 0];
    data.extend_from_slice(b"mjpg");
    data.resize(64, 0);
    assert!(matches!(
        decoder.decode(&data),
        Err(CodecError::Unsupported {
            what: "motion jpeg format b"
        })
    ));
    assert!(matches!(
        MjpegDecoder::open(&params(16, 16, b"avc1")),
        Err(CodecError::Unsupported { .. })
    ));
    let mut wrong = params(16, 16, &[]);
    wrong.codec = CodecId::Jpeg;
    assert!(MjpegDecoder::open(&wrong).is_err());
}

#[test]
fn planar_frames_keep_their_coded_sampling_and_odd_sizes() {
    for (sampling, format, chroma) in [
        (Sampling::Yuv420, PixelFormat::Yuv420p, (12, 9)),
        (Sampling::Yuv422, PixelFormat::Yuv422p, (12, 17)),
        (Sampling::Yuv444, PixelFormat::Yuv444p, (23, 17)),
    ] {
        let jpeg = encode(&detailed_rgb(23, 17, 10), 23, 17, sampling, 90);
        let mut decoder = MjpegDecoder::open(&params(23, 17, &[])).expect("open");
        let frame = decoder
            .decode(&jpeg)
            .expect("decode")
            .into_frame()
            .expect("frame");
        assert_eq!(frame.format, format);
        assert_eq!((frame.width, frame.height), (23, 17));
        assert_eq!((frame.u.width, frame.u.height), chroma);
        assert_eq!(frame.range, ColorRange::Full);
        assert_eq!(frame.matrix, MatrixCoefficients::Bt601);
        let image = decode_image(&jpeg, &mut Tables::new())
            .expect("image")
            .image;
        assert_eq!(frame.y.samples, image.components[0].samples);
        assert_eq!(frame.v.samples, image.components[2].samples);
    }
    let gray = encode(&detailed_rgb(9, 7, 11), 9, 7, Sampling::Gray, 90);
    let mut decoder = MjpegDecoder::open(&params(9, 7, &[])).expect("open");
    let frame = decoder
        .decode(&gray)
        .expect("gray")
        .into_frame()
        .expect("frame");
    assert_eq!(frame.format, PixelFormat::Gray8);
    assert_eq!(frame.y.samples.len(), 63);
    assert!(frame.u.samples.is_empty());
}

#[test]
fn a_studio_range_comment_tags_the_frame_and_the_rgba_conversion() {
    let jpeg = encode(&vec![0; 16 * 16 * 3], 16, 16, Sampling::Yuv420, 100);
    let studio = with_segment(&jpeg, 0xfe, b"CS=ITU601");
    let mut decoder = MjpegDecoder::open(&params(16, 16, &[])).expect("open");
    // Full-range black is luma 0, which studio range clips to black too;
    // a luma of 16 is studio black and full-range dark grey.
    let frame = decoder
        .decode(&studio)
        .expect("studio")
        .into_frame()
        .expect("frame");
    assert_eq!(frame.range, ColorRange::Limited);

    let grey = vec![16u8; 16 * 16 * 3];
    let mut planes = Planes::default();
    planes
        .from_rgb(&grey, 3, 16, 16, Sampling::Yuv444)
        .expect("planes");
    let luma = planes.y[0];
    let plain = encode(&grey, 16, 16, Sampling::Yuv444, 100);
    let mut full = Vec::new();
    decoder
        .decode(&plain)
        .expect("plain")
        .to_rgba(&mut full)
        .expect("rgba");
    let mut limited = Vec::new();
    decoder
        .decode(&with_segment(&plain, 0xfe, b"CS=ITU601\n"))
        .expect("tagged")
        .to_rgba(&mut limited)
        .expect("rgba");
    assert_eq!(full[0], 16);
    let want = ((76_309 * (i32::from(luma) - 16) + 32_768) >> 16).clamp(0, 255);
    assert_eq!(i32::from(limited[0]), want);
}

#[test]
fn every_truncation_and_bit_flip_fails_closed_and_the_next_packet_decodes() {
    let good = encode(&detailed_rgb(24, 16, 12), 24, 16, Sampling::Yuv420, 70);
    let want = image_rgba(&good);
    let mut decoder = MjpegDecoder::open(&params(24, 16, &[])).expect("open");
    for cut in 0..good.len() - 1 {
        assert!(decoder.decode(&good[..cut]).is_err(), "cut {cut}");
        assert_eq!(rgba_of(&mut decoder, &good), want, "after cut {cut}");
    }
    let mut state = 0x1234_5678u32;
    for _ in 0..2000 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let mut flipped = good.clone();
        let at = state as usize % flipped.len();
        flipped[at] ^= 1 << (state >> 29);
        let _ = decoder.decode(&flipped);
        assert_eq!(rgba_of(&mut decoder, &good), want);
    }
    assert!(matches!(
        decoder.decode(&vec![0xff; MAX_PACKET + 1]),
        Err(CodecError::Limit { .. })
    ));
}

#[test]
fn lifecycle_empty_drain_cancel_and_reset() {
    let jpeg = encode(&detailed_rgb(16, 16, 13), 16, 16, Sampling::Yuv420, 75);
    let mut decoder = MjpegDecoder::open(&params(16, 16, &[])).expect("open");
    assert!(matches!(
        VideoDecoder::push(&mut decoder, &packet(Vec::new(), 0)),
        Err(CodecError::InvalidApi { .. })
    ));
    let out = VideoDecoder::push(&mut decoder, &packet(jpeg.clone(), 3003)).expect("push");
    assert_eq!(out.len(), 1);
    assert_eq!(
        (out[0].pts, out[0].duration, out[0].timescale),
        (3003, 1001, 30_000)
    );
    assert!(VideoDecoder::drain(&mut decoder).expect("drain").is_empty());
    assert!(VideoDecoder::drain(&mut decoder).expect("drain").is_empty());
    let cancel = decoder.cancel_flag();
    cancel.store(true, Ordering::Relaxed);
    assert_eq!(
        VideoDecoder::push(&mut decoder, &packet(jpeg.clone(), 0)),
        Err(CodecError::Cancelled)
    );
    VideoDecoder::reset(&mut decoder);
    assert!(!cancel.load(Ordering::Relaxed));
    assert!(VideoDecoder::push(&mut decoder, &packet(jpeg, 0)).is_ok());
}

#[test]
fn the_picture_decoder_hands_the_engine_rgba_with_the_packet_time() {
    let rgb = detailed_rgb(21, 13, 14);
    let jpeg = encode(&rgb, 21, 13, Sampling::Yuv420, 92);
    let mut decoders = DecodeFactory::new();
    register_decoders(&mut decoders);
    let mut decoder = decoders.open(&params(21, 13, &[])).expect("open");
    assert_eq!(decoder.codec(), CodecId::Mjpeg);
    let out = decoder.push(&packet(jpeg.clone(), 2002)).expect("push");
    let Ok([Output::Picture(picture)]) = <[Output; 1]>::try_from(out) else {
        panic!("one picture");
    };
    assert_eq!(picture.info().pixel, PixelFormat::Rgba8);
    assert_eq!((picture.info().width, picture.info().height), (21, 13));
    assert_eq!(picture.pts(), 2002);
    assert_eq!(picture.into_cpu().expect("cpu"), image_rgba(&jpeg));
    assert!(decoder.drain().expect("drain").is_empty());
    assert!(decoder.push(&packet(Vec::new(), 0)).is_err());
    decoder.reset();

    let mut video = VideoDecodeFactory::new();
    register_video_decoders(&mut video);
    assert!(video.open(&params(21, 13, &[])).is_ok());
    let mut picture_decoder = MjpegPictureDecoder::open(&params(0, 0, &[])).expect("open");
    picture_decoder.push(&packet(jpeg, 0)).expect("no size");
    assert_eq!(picture_decoder.inner().last_field_order(), None);
}

#[test]
fn size_and_sampling_may_change_between_packets() {
    let stream = [
        (32, 16, Sampling::Yuv420, PixelFormat::Yuv420p),
        (23, 17, Sampling::Yuv422, PixelFormat::Yuv422p),
        (9, 7, Sampling::Gray, PixelFormat::Gray8),
        (40, 24, Sampling::Yuv444, PixelFormat::Yuv444p),
        (32, 16, Sampling::Yuv420, PixelFormat::Yuv420p),
    ];
    let mut planar = MjpegDecoder::open(&params(32, 16, &[])).expect("open");
    let mut engine = MjpegPictureDecoder::open(&params(32, 16, &[])).expect("open");
    for (index, (width, height, sampling, format)) in stream.into_iter().enumerate() {
        let jpeg = encode(
            &detailed_rgb(width, height, 30 + index as u32),
            width,
            height,
            sampling,
            85,
        );
        let frame = planar
            .decode(&jpeg)
            .expect("decode")
            .into_frame()
            .expect("frame");
        assert_eq!(
            (frame.width, frame.height, frame.format),
            (width, height, format)
        );
        let fresh = MjpegDecoder::open(&params(width, height, &[]))
            .expect("open")
            .decode(&jpeg)
            .expect("fresh")
            .into_frame()
            .expect("frame");
        assert_eq!(frame.y.samples, fresh.y.samples, "{index}");
        assert_eq!(frame.v.samples, fresh.v.samples, "{index}");

        let out = engine
            .push(&packet(jpeg.clone(), index as i64))
            .expect("push");
        let Ok([Output::Picture(picture)]) = <[Output; 1]>::try_from(out) else {
            panic!("one picture");
        };
        assert_eq!(
            (picture.info().width, picture.info().height),
            (width, height)
        );
        assert_eq!(
            picture.into_cpu().expect("cpu"),
            image_rgba(&jpeg),
            "{index}"
        );
    }

    let mut any_size = MjpegEncoder::open(&params(0, 0, &[80])).expect("open");
    for (width, height) in [(16, 16), (33, 9), (16, 16)] {
        let rgba: Vec<u8> = detailed_rgb(width, height, 40)
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect();
        let jpeg = any_size
            .encode_rgba(width, height, &rgba, 1, 25)
            .expect("encode");
        let frame = planar
            .decode(&jpeg)
            .expect("decode")
            .into_frame()
            .expect("frame");
        assert_eq!((frame.width, frame.height), (width, height));
    }
    let mut fixed = MjpegEncoder::open(&params(16, 16, &[80])).expect("open");
    assert!(matches!(
        fixed.encode_rgba(33, 9, &[0; 33 * 9 * 4], 1, 25),
        Err(CodecError::InvalidApi { .. })
    ));
    assert!(fixed.encode_rgba(16, 16, &[0; 16 * 16 * 4], 1, 25).is_ok());
}

fn rgba_frame(rgb: &[u8], width: u32, height: u32, pts: i64) -> FrameInput {
    let mut rgba = Vec::with_capacity(rgb.len() / 3 * 4);
    for pixel in rgb.as_chunks::<3>().0 {
        rgba.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
    }
    let time = FrameTime::new(pts, 1001, 30_000).expect("time");
    FrameInput::Cpu(CpuFrame::new(width, height, PixelFormat::Rgba8, rgba, time).expect("frame"))
}

#[test]
fn the_encoder_writes_frames_this_decoder_and_the_core_read_back() {
    let (width, height) = (45, 31);
    let rgb = detailed_rgb(width, height, 15);
    let mut encoders = EncodeFactory::new();
    register_encoders(&mut encoders);
    let mut quality_params = params(width, height, &[90]);
    quality_params.pixel = PixelFormat::Rgba8;
    let mut encoder = encoders.open(&quality_params).expect("open");
    let packets = encoder
        .push_frame(&rgba_frame(&rgb, width, height, 5005))
        .expect("encode");
    assert_eq!(packets.len(), 1);
    let packet = &packets[0];
    assert!(packet.keyframe);
    assert_eq!(
        (packet.pts, packet.duration, packet.timescale),
        (5005, 1001, 30_000)
    );
    // The same bytes as the core's encoder at the same quality and sampling.
    assert_eq!(
        packet.data,
        encode(&rgb, width, height, Sampling::Yuv420, 90)
    );
    let mut decoder = MjpegDecoder::open(&params(width, height, &[])).expect("open");
    let back = rgba_of(&mut decoder, &packet.data);
    assert!(psnr(&rgb_of_rgba(&back), &rgb) > 25.0);
    assert!(encoder.drain().expect("drain").is_empty());
}

#[test]
fn quality_changes_size_and_bad_settings_are_refused() {
    let rgb = detailed_rgb(64, 48, 16);
    let size = |quality: u8| {
        let mut encoder = MjpegEncoder::open(&params(64, 48, &[quality])).expect("open");
        let FrameInput::Cpu(frame) = rgba_frame(&rgb, 64, 48, 0) else {
            unreachable!()
        };
        encoder
            .encode_rgba(64, 48, frame.bytes(), 1001, 30_000)
            .expect("encode")
            .len()
    };
    assert!(size(20) < size(60) && size(60) < size(95));
    for extra in [&[0u8][..], &[101], &[50, 50]] {
        assert!(matches!(
            MjpegEncoder::open(&params(64, 48, extra)),
            Err(CodecError::Unsupported { .. })
        ));
    }
    assert!(
        MjpegEncoder::open(&params(64, 48, &[]))
            .expect("open")
            .with_mode(RateMode::Quality(0))
            .is_err()
    );
    assert!(matches!(
        MjpegEncoder::open(&params(20_000, 16, &[])),
        Err(CodecError::Limit { .. })
    ));
    let mut gray = params(64, 48, &[]);
    gray.pixel = PixelFormat::Yuv440p;
    assert!(MjpegEncoder::open(&gray).is_err());
    let mut encoder = MjpegEncoder::open(&params(64, 48, &[])).expect("open");
    assert_eq!(encoder.mode(), RateMode::Quality(75));
    assert!(matches!(
        encoder.push_frame(&rgba_frame(&rgb, 64, 48, 0)),
        Ok(packets) if packets.len() == 1
    ));
    let small = detailed_rgb(32, 48, 1);
    assert!(matches!(
        encoder.push_frame(&rgba_frame(&small, 32, 48, 0)),
        Err(CodecError::InvalidApi { .. })
    ));
    let pcm =
        cutlass_encode::Pcm::from_f32(48_000, cutlass_encode::ChannelLayout::mono(), vec![0.0])
            .expect("pcm");
    assert!(Encoder::push(&mut encoder, &cutlass_encode::AudioInput { pcm }).is_err());
}

fn planar_frame(format: PixelFormat, width: u32, height: u32) -> VideoFrame {
    let plane = |w: u32, h: u32, seed: u32| {
        let samples = (0..w * h)
            .map(|i| ((i * 7 + seed * 31 + i / w * 3) % 256) as u8)
            .collect();
        Plane::tight(w, h, samples).expect("plane")
    };
    let (cw, ch) = format.chroma_wh(width, height).unwrap_or((0, 0));
    let mut frame = if format == PixelFormat::Gray8 {
        let mut frame = VideoFrame::blank_yuv420p(width, height).expect("blank");
        frame.format = PixelFormat::Gray8;
        frame.y = plane(width, height, 1);
        frame.u = Plane::tight(0, 0, Vec::new()).expect("u");
        frame.v = Plane::tight(0, 0, Vec::new()).expect("v");
        frame
    } else {
        VideoFrame::planar(
            format,
            width,
            height,
            plane(width, height, 1),
            plane(cw, ch, 2),
            plane(cw, ch, 3),
        )
        .expect("frame")
    };
    frame.range = ColorRange::Full;
    frame
}

#[test]
fn planar_input_keeps_its_sampling_through_encode_and_decode() {
    for format in [
        PixelFormat::Gray8,
        PixelFormat::Yuv420p,
        PixelFormat::Yuv422p,
        PixelFormat::Yuv444p,
    ] {
        let source = planar_frame(format, 27, 19);
        let mut encoder = MjpegEncoder::open(&params(27, 19, &[95])).expect("open");
        let packets = VideoEncoder::push(
            &mut encoder,
            &VideoInput {
                frame: source.clone(),
                pts: 7,
                duration: 1,
                timescale: 24,
            },
        )
        .expect("encode");
        let mut decoder = MjpegDecoder::open(&params(27, 19, &[])).expect("open");
        let frame = decoder
            .decode(&packets[0].data)
            .expect("decode")
            .into_frame()
            .expect("frame");
        assert_eq!(frame.format, format);
        assert_eq!(frame.u.width, source.u.width);
        assert!(
            psnr(&frame.y.samples, &source.y.samples) > 30.0,
            "{format:?}"
        );
    }
    let mut studio = planar_frame(PixelFormat::Yuv420p, 16, 16);
    studio.range = ColorRange::Limited;
    let mut encoder = MjpegEncoder::open(&params(16, 16, &[])).expect("open");
    assert!(matches!(
        encoder.encode_frame(&studio, 1, 24),
        Err(CodecError::Unsupported {
            what: "studio range input"
        })
    ));
    let mut hd = planar_frame(PixelFormat::Yuv420p, 16, 16);
    hd.matrix = MatrixCoefficients::Bt709;
    assert!(encoder.encode_frame(&hd, 1, 24).is_err());
    let odd = planar_frame(PixelFormat::Yuv440p, 16, 16);
    assert!(encoder.encode_frame(&odd, 1, 24).is_err());
}

#[test]
fn cancel_stops_the_encoder_until_reset() {
    let rgb = detailed_rgb(16, 16, 17);
    let mut encoder = MjpegEncoder::open(&params(16, 16, &[])).expect("open");
    let flag = Encoder::cancel_flag(&encoder);
    flag.store(true, Ordering::Relaxed);
    assert_eq!(
        encoder.push_frame(&rgba_frame(&rgb, 16, 16, 0)),
        Err(CodecError::Cancelled)
    );
    Encoder::reset(&mut encoder);
    assert!(encoder.push_frame(&rgba_frame(&rgb, 16, 16, 0)).is_ok());
    assert!(Arc::ptr_eq(&flag, &VideoEncoder::cancel_flag(&encoder)));
}

/// A clip that pans across a detailed picture, so every frame differs.
fn panning_clip(width: u32, height: u32, frames: u32) -> Vec<Vec<u8>> {
    let wide = detailed_rgb(width + frames * 3, height, 18);
    let stride = ((width + frames * 3) * 3) as usize;
    (0..frames)
        .map(|frame| {
            let mut out = Vec::with_capacity((width * height * 3) as usize);
            for y in 0..height as usize {
                let at = y * stride + frame as usize * 9;
                out.extend_from_slice(&wide[at..at + (width * 3) as usize]);
            }
            out
        })
        .collect()
}

#[test]
fn bitrate_mode_holds_the_mean_rate_within_ten_percent() {
    let (width, height, frames) = (160, 96, 30u32);
    let clip = panning_clip(width, height, frames);
    // Measure the clip at quality 50 and 90, then target the middle.
    let sizes: Vec<usize> = [50u8, 90]
        .iter()
        .map(|q| {
            clip.iter()
                .map(|rgb| encode(rgb, width, height, Sampling::Yuv420, *q).len())
                .sum()
        })
        .collect();
    let mean_bytes = (sizes[0] + sizes[1]) as f64 / 2.0 / f64::from(frames);
    let bitrate = (mean_bytes * 8.0 * 30_000.0 / 1001.0) as u32;
    let mut rate_params = params(width, height, &[]);
    rate_params.bitrate = bitrate;
    let mut encoder = MjpegEncoder::open(&rate_params).expect("open");
    assert_eq!(encoder.mode(), RateMode::Bitrate(u64::from(bitrate)));
    let mut total = 0usize;
    let mut qualities = Vec::new();
    for (index, rgb) in clip.iter().enumerate() {
        let packets = encoder
            .push_frame(&rgba_frame(rgb, width, height, index as i64 * 1001))
            .expect("encode");
        total += packets[0].data.len();
        qualities.push(encoder.last_quality().expect("quality"));
    }
    let achieved = total as f64 * 8.0 / (f64::from(frames) * 1001.0 / 30_000.0);
    let error = (achieved - f64::from(bitrate)).abs() / f64::from(bitrate);
    assert!(
        error < 0.10,
        "achieved {achieved} target {bitrate} qualities {qualities:?}"
    );
    assert!(
        qualities.iter().all(|q| (50..=90).contains(q)),
        "{qualities:?}"
    );

    // A frame with no duration has no share of a rate.
    let FrameInput::Cpu(frame) = rgba_frame(&clip[0], width, height, 0) else {
        unreachable!()
    };
    assert!(matches!(
        encoder.encode_rgba(width, height, frame.bytes(), 0, 30_000),
        Err(CodecError::InvalidApi {
            what: "frame duration"
        })
    ));
}

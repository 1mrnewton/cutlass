//! Malformed input and lifecycle: every bad frame is refused, none panics,
//! and a refused frame leaves the decoder as it was.

use std::sync::atomic::Ordering;

use cutlass_decode::video::VideoDecoder;
use cutlass_decode::{CodecError, Packet, PixelFormat};
use cutlass_encode::video::{VideoEncoder, VideoInput};

use crate::bits::BitWriter;
use crate::tests::{busy, params, plane16};
use crate::{Profile, ProresDecoder, ProresEncoder};

fn valid(profile: Profile, width: u32, height: u32, alpha: bool) -> Vec<u8> {
    let format = if profile.is_444() {
        PixelFormat::Yuv444p12
    } else {
        PixelFormat::Yuv422p10
    };
    let mut frame = busy(format, width, height);
    if alpha {
        frame = frame
            .with_alpha(8, {
                let samples = (0..width * height).map(|i| (i % 251) as u8).collect();
                cutlass_decode::Plane::bytes(width, height, 1, samples).expect("alpha")
            })
            .expect("alpha");
    }
    ProresEncoder::open(&params(profile))
        .expect("open")
        .encode(&frame)
        .expect("encode")
}

fn decoder() -> ProresDecoder {
    ProresDecoder::open(&params(Profile::Hq))
        .expect("open")
        .with_threads(1)
}

fn decode(bytes: &[u8]) -> Result<(), CodecError> {
    decoder().decode(bytes).map(|_| ())
}

/// Offsets in a frame from [`crate::encode`]: 148-byte header, one picture.
const HEADER: usize = 8;
const PICTURE: usize = 8 + 148;
const TABLE: usize = PICTURE + 8;

fn malformed(result: Result<(), CodecError>, what: &str) {
    match result {
        Err(CodecError::Malformed { what: got, .. }) => assert_eq!(got, what),
        other => panic!("expected malformed {what}, got {other:?}"),
    }
}

/// A one-macroblock 4:2:2 frame whose single slice has the given parts.
fn one_slice(index: u8, y: &[u8], cb: &[u8], cr: &[u8], alpha: Option<&[u8]>) -> Vec<u8> {
    let header_size: u8 = if alpha.is_some() { 8 } else { 6 };
    let mut slice = vec![header_size << 3, index];
    slice.extend_from_slice(&(y.len() as u16).to_be_bytes());
    slice.extend_from_slice(&(cb.len() as u16).to_be_bytes());
    if alpha.is_some() {
        slice.extend_from_slice(&(cr.len() as u16).to_be_bytes());
    }
    slice.extend_from_slice(y);
    slice.extend_from_slice(cb);
    slice.extend_from_slice(cr);
    if let Some(alpha) = alpha {
        slice.extend_from_slice(alpha);
    }
    let mut frame = Vec::new();
    let picture_size = 8 + 2 + slice.len();
    let frame_size = 8 + 20 + picture_size;
    frame.extend_from_slice(&(frame_size as u32).to_be_bytes());
    frame.extend_from_slice(b"icpf");
    frame.extend_from_slice(&20u16.to_be_bytes());
    frame.extend_from_slice(&[0, 1]);
    frame.extend_from_slice(b"test");
    frame.extend_from_slice(&16u16.to_be_bytes());
    frame.extend_from_slice(&16u16.to_be_bytes());
    frame.push(if alpha.is_some() { 3 << 6 } else { 2 << 6 });
    frame.extend_from_slice(&[0, 1, 1, 1]);
    frame.push(u8::from(alpha.is_some()));
    frame.extend_from_slice(&[0, 0]);
    frame.push(8 << 3);
    frame.extend_from_slice(&(picture_size as u32).to_be_bytes());
    frame.extend_from_slice(&1u16.to_be_bytes());
    frame.push(3 << 4);
    frame.extend_from_slice(&(slice.len() as u16).to_be_bytes());
    frame.extend_from_slice(&slice);
    frame
}

/// A flat component: every DC zero, no AC.
fn flat(blocks: usize) -> Vec<u8> {
    let mut w = BitWriter::default();
    w.code(crate::tables::FIRST_DC_CODEBOOK, 0);
    let mut previous = crate::tables::FIRST_DC_SYMBOL;
    for _ in 1..blocks {
        w.code(crate::tables::dc_codebook(previous), 0);
        previous = 0;
    }
    w.finish()
}

#[test]
fn a_hand_built_flat_frame_decodes_to_mid_grey() {
    let frame = one_slice(4, &flat(4), &flat(2), &flat(2), None);
    let decoded = decoder().decode(&frame).expect("decode");
    assert!(
        decoded
            .frame
            .y
            .samples
            .as_chunks::<2>()
            .0
            .iter()
            .all(|pair| u16::from_le_bytes(*pair) == 512)
    );
}

#[test]
fn every_truncation_is_refused() {
    let frame = valid(Profile::Standard, 48, 32, false);
    assert!(decode(&frame).is_ok());
    for len in 0..frame.len() {
        let result = decode(&frame[..len]);
        assert!(
            matches!(
                result,
                Err(CodecError::Truncated { .. }
                    | CodecError::Malformed { .. }
                    | CodecError::InvalidApi { .. })
            ),
            "length {len}: {result:?}"
        );
    }
    // The frame says it is longer than the packet: truncated, not read past.
    let mut short = frame.clone();
    short.truncate(frame.len() - 1);
    assert!(matches!(decode(&short), Err(CodecError::Truncated { .. })));
}

#[test]
fn bad_headers_are_named() {
    let frame = valid(Profile::P4444, 32, 32, true);
    let patch = |at: usize, value: u8| {
        let mut bad = frame.clone();
        bad[at] = value;
        bad
    };
    malformed(decode(&patch(4, b'x')), "frame identifier");
    // Frame header: byte n is at HEADER + n.
    assert!(matches!(
        decode(&patch(HEADER + 3, 2)),
        Err(CodecError::Unsupported { .. })
    ));
    let mut zero_width = frame.clone();
    zero_width[HEADER + 8..HEADER + 10].copy_from_slice(&[0, 0]);
    malformed(decode(&zero_width), "frame dimensions");
    let mut huge = frame.clone();
    huge[HEADER + 8..HEADER + 10].copy_from_slice(&[0xff, 0xff]);
    assert!(matches!(decode(&huge), Err(CodecError::Limit { .. })));
    malformed(decode(&patch(HEADER + 12, 1 << 6)), "chroma format");
    malformed(
        decode(&patch(HEADER + 12, (3 << 6) | (3 << 2))),
        "interlace mode",
    );
    malformed(decode(&patch(HEADER + 17, 3)), "alpha channel type");
    malformed(decode(&patch(HEADER + 20, 0)), "quantization weight");
    let mut small_header = frame.clone();
    small_header[HEADER..HEADER + 2].copy_from_slice(&19u16.to_be_bytes());
    malformed(decode(&small_header), "frame header size");
    // Picture header.
    malformed(decode(&patch(PICTURE, 7 << 3)), "picture header size");
    let mut long_picture = frame.clone();
    long_picture[PICTURE + 1..PICTURE + 5].copy_from_slice(&u32::MAX.to_be_bytes());
    assert!(matches!(
        decode(&long_picture),
        Err(CodecError::Truncated { .. })
    ));
}

#[test]
fn slice_sizes_that_overrun_the_frame_are_refused() {
    let frame = valid(Profile::Hq, 64, 32, false);
    let mut bad = frame.clone();
    bad[TABLE..TABLE + 2].copy_from_slice(&u16::MAX.to_be_bytes());
    malformed(decode(&bad), "slice sizes overrun the picture");
    let mut tiny = frame.clone();
    tiny[TABLE..TABLE + 2].copy_from_slice(&3u16.to_be_bytes());
    malformed(decode(&tiny), "slice size");
}

#[test]
fn quantization_indices_outside_1_to_224_are_refused() {
    for index in [0u8, 225, 255] {
        let frame = one_slice(index, &flat(4), &flat(2), &flat(2), None);
        malformed(decode(&frame), "quantization index");
    }
    let frame = one_slice(224, &flat(4), &flat(2), &flat(2), None);
    assert!(decode(&frame).is_ok());
}

#[test]
fn component_sizes_past_the_slice_are_refused() {
    let mut frame = one_slice(4, &flat(4), &flat(2), &flat(2), None);
    // The only slice follows the 8-byte prefix, the 20-byte frame header,
    // the picture header and a one-entry table.
    let slice = 8 + 20 + 8 + 2;
    // Its Y size field.
    frame[slice + 2..slice + 4].copy_from_slice(&500u16.to_be_bytes());
    malformed(decode(&frame), "slice component sizes");
}

#[test]
fn a_coefficient_run_past_the_block_is_refused() {
    let mut w = BitWriter::default();
    w.code(crate::tables::FIRST_DC_CODEBOOK, 0);
    let mut previous = crate::tables::FIRST_DC_SYMBOL;
    for _ in 1..4 {
        w.code(crate::tables::dc_codebook(previous), 0);
        previous = 0;
    }
    // Four blocks hold 256 positions; a run of 300 runs off the end.
    w.code(crate::tables::run_codebook(crate::tables::FIRST_RUN), 300);
    w.code(crate::tables::level_codebook(crate::tables::FIRST_LEVEL), 0);
    w.put(1, 0);
    let y = w.finish();
    let frame = one_slice(4, &y, &flat(2), &flat(2), None);
    malformed(decode(&frame), "coefficient run past the block");
}

#[test]
fn a_codeword_past_31_bits_is_refused() {
    let y = vec![0u8; 8];
    let frame = one_slice(4, &y, &flat(2), &flat(2), None);
    malformed(decode(&frame), "codeword");
}

#[test]
fn alpha_runs_past_the_slice_clamp_and_short_alpha_is_refused() {
    // 4444 has four chroma blocks; this frame header says 4:4:4 with alpha.
    let alpha_values = 16 * 16;
    let mut w = BitWriter::default();
    w.put(1, 1);
    w.put(8, 9);
    w.put(1, 0);
    w.put(4, 0);
    w.put(11, 2047);
    let long_run = w.finish();
    let frame = one_slice(4, &flat(4), &flat(4), &flat(4), Some(&long_run));
    let decoded = decoder().decode(&frame).expect("a long run clamps");
    let alpha = decoded.frame.alpha.expect("alpha");
    assert_eq!(alpha.plane.samples.len(), alpha_values);
    assert!(alpha.plane.samples.iter().all(|&a| a == 8));

    let frame = one_slice(4, &flat(4), &flat(4), &flat(4), Some(&[]));
    malformed(decode(&frame), "empty alpha data");
    let mut w = BitWriter::default();
    w.put(1, 1);
    w.put(8, 9);
    w.put(1, 1);
    let cut = w.finish();
    let frame = one_slice(4, &flat(4), &flat(4), &flat(4), Some(&cut));
    malformed(decode(&frame), "alpha data overrun");
}

#[test]
fn random_damage_never_panics() {
    let frames = [
        valid(Profile::Proxy, 40, 24, false),
        valid(Profile::Xq, 24, 40, true),
    ];
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for round in 0..4_000 {
        let source = &frames[round % 2];
        let mut bad = source.clone();
        let flips = 1 + next() % 8;
        for _ in 0..flips {
            let at = (next() % bad.len() as u64) as usize;
            bad[at] ^= 1 << (next() % 8);
        }
        let _ = decode(&bad);
    }
}

/// A longer campaign of the same: `cargo test --release -p cutlass-prores
/// fuzz_campaign -- --ignored`. `PRORES_FUZZ_SECONDS` sets the budget.
#[test]
#[ignore = "time-budgeted campaign; run by hand"]
fn fuzz_campaign() {
    let seconds: u64 = std::env::var("PRORES_FUZZ_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let frames = [
        valid(Profile::Proxy, 40, 24, false),
        valid(Profile::Hq, 72, 40, false),
        valid(Profile::P4444, 33, 17, true),
        valid(Profile::Xq, 24, 40, true),
    ];
    let start = std::time::Instant::now();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut cases = 0u64;
    let mut refused = 0u64;
    while start.elapsed().as_secs() < seconds {
        let source = &frames[(next() % 4) as usize];
        let mut bad = source.clone();
        match next() % 4 {
            0 => {
                for _ in 0..1 + next() % 16 {
                    let at = (next() % bad.len() as u64) as usize;
                    bad[at] ^= 1 << (next() % 8);
                }
            }
            1 => {
                let at = (next() % bad.len() as u64) as usize;
                bad.truncate(at);
            }
            2 => {
                let at = (next() % bad.len() as u64) as usize;
                bad[at] = (next() & 0xff) as u8;
                let span = (next() % 64) as usize;
                let end = (at + span).min(bad.len());
                for byte in &mut bad[at..end] {
                    *byte = (next() & 0xff) as u8;
                }
            }
            _ => {
                let header = 8 + 148 + 8;
                let at = header + (next() % 64) as usize;
                if at + 2 <= bad.len() {
                    let value = (next() & 0xffff) as u16;
                    bad[at..at + 2].copy_from_slice(&value.to_be_bytes());
                }
            }
        }
        if decode(&bad).is_err() {
            refused += 1;
        }
        cases += 1;
    }
    eprintln!("fuzz campaign: {cases} cases in {seconds} s, {refused} refused, no panics");
}

#[test]
fn an_empty_packet_and_a_cancel_are_api_errors() {
    let mut decoder = decoder();
    assert!(matches!(
        decoder.decode(&[]),
        Err(CodecError::InvalidApi { .. })
    ));
    let packet = Packet::video(Vec::new(), 0, 1, 30, true);
    assert!(VideoDecoder::push(&mut decoder, &packet).is_err());
    let flag = decoder.cancel_flag();
    flag.store(true, Ordering::Relaxed);
    let frame = valid(Profile::Lt, 32, 16, false);
    assert_eq!(
        decoder.decode(&frame).map(|_| ()),
        Err(CodecError::Cancelled)
    );
    VideoDecoder::reset(&mut decoder);
    assert!(decoder.decode(&frame).is_ok());
}

#[test]
fn a_refused_frame_leaves_the_decoder_as_it_was() {
    let mut decoder = decoder();
    let frame = valid(Profile::Standard, 48, 32, false);
    let first = decoder.decode(&frame).expect("first");
    let mut bad = frame.clone();
    bad[TABLE..TABLE + 2].copy_from_slice(&u16::MAX.to_be_bytes());
    assert!(decoder.decode(&bad).is_err());
    let again = decoder.decode(&frame).expect("again");
    assert_eq!(first, again);
    assert!(VideoDecoder::drain(&mut decoder).expect("drain").is_empty());
    VideoDecoder::reset(&mut decoder);
    assert_eq!(decoder.decode(&frame).expect("after reset"), first);
}

#[test]
fn the_video_traits_carry_time_through() {
    let source = busy(PixelFormat::Yuv422p10, 32, 16);
    let mut encoder = ProresEncoder::open(&params(Profile::Proxy)).expect("open");
    let packets = VideoEncoder::push(
        &mut encoder,
        &VideoInput {
            frame: source,
            pts: 1001,
            duration: 1001,
            timescale: 30_000,
        },
    )
    .expect("push");
    assert_eq!(packets.len(), 1);
    assert!(packets[0].keyframe);
    assert!(VideoEncoder::drain(&mut encoder).expect("drain").is_empty());
    let mut decoder = decoder();
    let out = VideoDecoder::push(&mut decoder, &packets[0]).expect("decode");
    assert_eq!(out.len(), 1);
    assert_eq!(
        (out[0].pts, out[0].duration, out[0].timescale),
        (1001, 1001, 30_000)
    );
    assert_eq!(out[0].frame.format, PixelFormat::Yuv422p10);
}

#[test]
fn the_encoder_refuses_what_it_cannot_code() {
    let mut params = params(Profile::Hq);
    params.width = 32;
    params.height = 16;
    let mut encoder = ProresEncoder::open(&params).expect("open");
    let wrong = busy(PixelFormat::Yuv422p10, 48, 16);
    assert!(matches!(
        encoder.encode(&wrong),
        Err(CodecError::InvalidApi { .. })
    ));
    assert!(encoder.encode_rgba(32, 16, &[0; 7]).is_err());
    assert!(encoder.encode_rgba(0, 16, &[]).is_err());
    let mut bad = crate::tests::params(Profile::Hq);
    bad.extra = b"avc1".to_vec();
    assert!(ProresEncoder::open(&bad).is_err());
    let flag = VideoEncoder::cancel_flag(&encoder);
    flag.store(true, Ordering::Relaxed);
    let right = busy(PixelFormat::Yuv422p10, 32, 16);
    assert_eq!(encoder.encode(&right), Err(CodecError::Cancelled));
    VideoEncoder::reset(&mut encoder);
    assert!(encoder.encode(&right).is_ok());
    let alpha = plane16(32, 16, |_, _| 0);
    let with_alpha = right.with_alpha(16, alpha).expect("alpha");
    // A 4:2:2 profile drops alpha rather than refusing the frame.
    let bytes = encoder.encode(&with_alpha).expect("encode");
    let decoded = decoder().decode(&bytes).expect("decode");
    assert!(decoded.frame.alpha.is_none());
}

#[test]
fn the_smallest_pictures_round_trip() {
    for (width, height) in [(1, 1), (1, 17), (17, 1), (2, 2), (15, 31)] {
        for profile in [Profile::Proxy, Profile::P4444] {
            let format = if profile.is_444() {
                PixelFormat::Yuv444p12
            } else {
                PixelFormat::Yuv422p10
            };
            let source = busy(format, width, height);
            let bytes = ProresEncoder::open(&params(profile))
                .expect("open")
                .encode(&source)
                .expect("encode");
            let decoded = decoder().decode(&bytes).expect("decode");
            assert_eq!((decoded.frame.width, decoded.frame.height), (width, height));
        }
    }
}

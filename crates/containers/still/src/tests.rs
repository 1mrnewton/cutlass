//! Identification, packet split, hostile input, seek, and mux.

use std::cell::Cell;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::rc::Rc;

use cutlass_codec::{
    CodecId, FrameSize, Packet, StreamCodec, StreamParams, Timescale, Timestamp, UnknownCodec,
};
use cutlass_container::{
    Demux, IndexPlace, Input, Landing, Mux, MuxError, MuxMode, MuxStatus, NewStream, OpenError,
    Probe, ProbeResult, ReadError, SeekError, SeekTo, TrackDuration, TrackId,
};

use crate::png::append_chunk;
use crate::{
    GIF_DEFAULT_DELAY, GIF_TIMESCALE, MAX_CHUNK_DEPTH, MAX_FILE_BYTES, MAX_FRAMES, PROBE_BYTES,
    StillDemux, StillKind, StillMux, StillProbe, UNSUPPORTED, WEBP_DEFAULT_DELAY_MS,
    WEBP_TIMESCALE, probe, still_limits,
};

const JPEG: &[u8] = &[
    0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF,
    0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0x00, 0xFF, 0xD9,
];

const PNG_HEX: &str = "89504e470d0a1a0a0000000d49484452000000010000000108000000003a7e9b550000000a49444154789c63a8070000810080d394534a0000000049454e44ae426082";

const IMAGE: [u8; 15] = [
    0x2C, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x02, 0x02, 0x4C, 0x01, 0x00,
];

#[test]
fn the_support_matrix_names_the_clock_the_bounds_and_what_is_left_out() {
    assert_eq!(GIF_TIMESCALE, 100);
    assert_eq!(GIF_DEFAULT_DELAY, 10);
    assert_eq!(WEBP_TIMESCALE, 1000);
    assert_eq!(WEBP_DEFAULT_DELAY_MS, 100);
    assert_eq!(MAX_FILE_BYTES, 32 * 1024 * 1024);
    assert_eq!(MAX_FRAMES, 4096);
    assert_eq!(MAX_CHUNK_DEPTH, 2);
    assert_eq!(PROBE_BYTES, 12);
    for name in [
        "apng frame sequence",
        "heic",
        "avif",
        "pixel decode",
        "huffman",
        "inflate",
        "lzw",
        "fragmented output",
        "invented frame rate",
        "infinite loop as packets",
        "gif plain text",
    ] {
        assert!(UNSUPPORTED.contains(&name), "the matrix dropped {name}");
    }
}

#[test]
fn probe_accepts_a_finished_signature_and_waits_on_a_real_prefix() {
    assert!(matches!(
        probe(&[0xFF, 0xD8]),
        ProbeResult::Accept { examined: 2, .. }
    ));
    assert!(matches!(
        probe(&[0xFF]),
        ProbeResult::NeedBytes { have: 1, need: 2 }
    ));
    assert!(matches!(probe(&[0xFF, 0x00]), ProbeResult::Refuse { .. }));

    let png = png_bytes();
    assert!(matches!(
        probe(&png[..8]),
        ProbeResult::Accept { examined: 8, .. }
    ));
    assert!(matches!(
        probe(&png[..4]),
        ProbeResult::NeedBytes { have: 4, need: 8 }
    ));
    assert!(matches!(probe(&[0x89, b'X']), ProbeResult::Refuse { .. }));

    assert!(matches!(
        probe(b"GIF89a"),
        ProbeResult::Accept { examined: 6, .. }
    ));
    assert!(matches!(
        probe(b"GIF87a"),
        ProbeResult::Accept { examined: 6, .. }
    ));
    assert!(matches!(
        probe(b"GIF"),
        ProbeResult::NeedBytes { have: 3, need: 6 }
    ));
    assert!(matches!(probe(b"GX"), ProbeResult::Refuse { .. }));
    assert!(matches!(probe(b"GIF99a"), ProbeResult::Refuse { .. }));

    let webp = still_webp();
    assert!(matches!(
        probe(&webp[..12]),
        ProbeResult::Accept { examined: 12, .. }
    ));
    assert!(matches!(
        probe(b"RIFF"),
        ProbeResult::NeedBytes { have: 4, need: 12 }
    ));
    assert!(matches!(probe(b"RX"), ProbeResult::Refuse { .. }));
    assert!(matches!(probe(b"RIFF????AVI "), ProbeResult::Refuse { .. }));
    assert!(matches!(probe(b"RIFF????WAVE"), ProbeResult::Refuse { .. }));
    assert!(matches!(
        probe(&[0x00, 0x00, 0x00, 0x18, b'f', b't', b'y', b'p']),
        ProbeResult::Refuse { examined: 1 }
    ));
    assert!(matches!(
        probe(&[0x1A, 0x45, 0xDF, 0xA3]),
        ProbeResult::Refuse { examined: 1 }
    ));
    assert_eq!(StillProbe.family(), cutlass_codec::ContainerId::Still);
    assert_eq!(StillProbe::HEADER_BYTES, PROBE_BYTES);
    assert!(matches!(
        StillProbe.probe(b"GIF89aextra"),
        ProbeResult::Accept { .. }
    ));
}

#[test]
fn a_jpeg_png_or_still_webp_is_one_packet_with_no_invented_clock() {
    assert_eq!(JPEG.len(), 28);
    let png = png_bytes();
    assert_eq!(png.len(), 67);
    let webp = still_webp();
    let loss = lossless_webp();
    let extended = extended_still_webp();
    for (bytes, kind, width, height) in [
        (JPEG, StillKind::Jpeg, 1, 1),
        (png.as_slice(), StillKind::Png, 1, 1),
        (webp.as_slice(), StillKind::Webp, 8, 8),
        (loss.as_slice(), StillKind::Webp, 4, 4),
        (extended.as_slice(), StillKind::Webp, 8, 8),
    ] {
        let mut demux = open_ok(bytes);
        assert_eq!(demux.kind(), kind);
        assert!(!demux.animated());
        assert_eq!(demux.loop_count(), None);
        assert_eq!(demux.media_end(), None);
        assert_eq!(demux.frame_count(), 1);
        assert_eq!(demux.consumed(), u64::try_from(bytes.len()).expect("len"));
        let stream = &demux.streams()[0];
        assert!(stream.params().extradata().is_empty());
        assert_eq!(stream.params().codec(), &StreamCodec::Known(kind.codec()));
        assert_eq!(stream.timescale().ticks_per_second(), 0);
        assert!(stream.params().sample_rate().is_none());
        assert!(stream.edits().is_empty());
        assert!(demux.gaps().is_empty());
        let size = stream.params().size().expect("size");
        assert_eq!((size.width(), size.height()), (width, height));
        let packet = demux.read_packet().expect("read").expect("packet");
        assert_eq!(packet.bytes(), bytes);
        assert!(packet.timestamp().is_missing());
        assert!(!packet.timescale().is_clock());
        assert!(packet.keyframe());
        assert_eq!(packet.stream(), 0);
        assert!(demux.read_packet().expect("end").is_none());
    }
}

#[test]
fn trailing_bytes_after_the_picture_stay_in_the_one_packet() {
    let mut jpeg = JPEG.to_vec();
    jpeg.push(0x00);
    let mut png = png_bytes();
    png.push(0x00);
    for bytes in [jpeg, png] {
        let mut demux = open_ok(&bytes);
        let packet = demux.read_packet().expect("read").expect("packet");
        assert_eq!(packet.bytes(), bytes);
    }
}

#[test]
fn an_apng_file_stays_one_picture() {
    let mut bytes = vec![0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n'];
    append_chunk(
        &mut bytes,
        b"IHDR",
        &[0, 0, 0, 1, 0, 0, 0, 1, 8, 0, 0, 0, 0],
    );
    append_chunk(&mut bytes, b"acTL", &[0, 0, 0, 2, 0, 0, 0, 0]);
    append_chunk(
        &mut bytes,
        b"IDAT",
        &[0x78, 0x9C, 0x63, 0xA8, 0x07, 0x00, 0x00, 0x81, 0x00, 0x80],
    );
    append_chunk(&mut bytes, b"IEND", &[]);
    let mut demux = open_ok(&bytes);
    assert_eq!(demux.kind(), StillKind::Png);
    assert!(!demux.animated());
    assert_eq!(demux.frame_count(), 1);
    let packet = demux.read_packet().expect("read").expect("packet");
    assert_eq!(packet.bytes(), bytes);
    assert!(packet.timestamp().is_missing());
}

#[test]
fn a_gif_is_one_packet_per_frame_in_file_order() {
    let file = unequal_gif();
    assert_eq!(file.len(), 66);
    let mut demux = open_ok(&file);
    assert_eq!(demux.kind(), StillKind::Gif);
    assert!(demux.animated());
    assert_eq!(demux.loop_count(), None);
    assert_eq!(demux.media_end(), Some(25));
    assert_eq!(demux.frame_count(), 2);
    let stream = &demux.streams()[0];
    assert_eq!(stream.timescale().ticks_per_second(), GIF_TIMESCALE);
    assert!(stream.params().extradata().is_empty());
    assert_eq!(stream.params().codec(), &StreamCodec::Known(CodecId::Gif));
    assert_eq!(demux.frame(0), Some(frame_control(1, None)));
    assert_eq!(demux.frame(1), Some(frame_control(2, None)));
    let packets = read_all(&mut demux);
    assert_eq!(packets.len(), 2);
    assert_eq!(packets[0].bytes(), &file[..42]);
    assert_eq!(packets[1].bytes(), &file[42..]);
    assert_eq!(stamp(&packets[0]), (0, 5));
    assert_eq!(stamp(&packets[1]), (5, 20));
    assert!(packets[0].keyframe());
    assert!(!packets[1].keyframe());
    assert_eq!(packets[0].timescale().ticks_per_second(), GIF_TIMESCALE);
    let mut joined = Vec::new();
    for packet in &packets {
        joined.extend_from_slice(packet.bytes());
    }
    assert_eq!(joined, file);
}

#[test]
fn a_missing_or_zero_gif_delay_uses_the_recorded_default_without_rewriting_bytes() {
    let missing = gif_without_control();
    assert_eq!(missing.len(), 35);
    let mut demux = open_ok(&missing);
    assert!(!demux.animated());
    assert_eq!(demux.media_end(), Some(GIF_DEFAULT_DELAY));
    let packet = demux.read_packet().expect("read").expect("packet");
    assert_eq!(packet.bytes(), missing);
    assert_eq!(stamp(&packet), (0, GIF_DEFAULT_DELAY));

    let zero = gif_with_delay(0, 0x04);
    assert_eq!(zero.len(), 43);
    let mut demux = open_ok(&zero);
    let packet = demux.read_packet().expect("read").expect("packet");
    assert_eq!(packet.bytes(), zero);
    assert_eq!(stamp(&packet), (0, GIF_DEFAULT_DELAY));
    assert!(
        packet
            .bytes()
            .windows(4)
            .any(|window| window == [0x04, 0x00, 0x00, 0x00])
    );
}

#[test]
fn a_loop_count_does_not_repeat_the_packets() {
    let mut file = gif_header();
    file.extend(netscape(0));
    file.extend(gce(0x04, 5));
    file.extend(IMAGE);
    file.extend(gce(0x08, 20));
    file.extend(IMAGE);
    file.push(0x3B);
    let mut demux = open_ok(&file);
    assert_eq!(demux.loop_count(), Some(0));
    assert_eq!(demux.frame_count(), 2);
    let packets = read_all(&mut demux);
    assert_eq!(packets.len(), 2);
    assert_eq!(stamp(&packets[0]), (0, 5));
    assert_eq!(stamp(&packets[1]), (5, 20));
}

#[test]
fn gif87a_and_a_plain_text_extension_are_one_timed_picture() {
    let mut file = b"GIF87a".to_vec();
    file.extend([0x01, 0x00, 0x01, 0x00, 0x80, 0x00, 0x00]);
    file.extend([0x00, 0x00, 0x00, 0xFF, 0x00, 0x00]);
    file.extend([0x21, 0x01, 0x00]);
    file.extend(IMAGE);
    file.push(0x3B);
    let mut demux = open_ok(&file);
    assert_eq!(demux.frame_count(), 1);
    assert_eq!(demux.loop_count(), None);
    let packet = demux.read_packet().expect("read").expect("packet");
    assert_eq!(packet.bytes(), file);
    assert_eq!(stamp(&packet), (0, GIF_DEFAULT_DELAY));
}

#[test]
fn animated_webp_frames_keep_file_order_and_the_stored_delay() {
    let file = animated_webp(100, 0x02, 400, 0x01);
    let second = second_anmf(&file);
    let mut demux = open_ok(&file);
    assert_eq!(demux.kind(), StillKind::Webp);
    assert!(demux.animated());
    assert_eq!(demux.loop_count(), Some(0));
    assert_eq!(demux.media_end(), Some(500));
    assert_eq!(demux.frame_count(), 2);
    assert_eq!(
        demux.streams()[0].timescale().ticks_per_second(),
        WEBP_TIMESCALE
    );
    assert!(demux.streams()[0].params().extradata().is_empty());
    assert_eq!(demux.frame(0), Some(frame_control(0, Some(1))));
    assert_eq!(demux.frame(1), Some(frame_control(1, Some(0))));
    let packets = read_all(&mut demux);
    assert_eq!(packets[0].bytes(), &file[..second]);
    assert_eq!(packets[1].bytes(), &file[second..]);
    assert!(packets[0].bytes().starts_with(b"RIFF"));
    assert!(packets[1].bytes().starts_with(b"ANMF"));
    assert_eq!(stamp(&packets[0]), (0, 100));
    assert_eq!(stamp(&packets[1]), (100, 400));
    assert!(packets[0].keyframe());
    assert!(!packets[1].keyframe());
    assert_eq!(packets[0].timescale().ticks_per_second(), WEBP_TIMESCALE);
}

#[test]
fn a_webp_duration_of_zero_uses_the_default_and_a_short_duration_is_kept() {
    let zero = one_frame_webp(0);
    let mut demux = open_ok(&zero);
    let packet = demux.read_packet().expect("read").expect("packet");
    assert_eq!(stamp(&packet), (0, WEBP_DEFAULT_DELAY_MS));
    assert_eq!(packet.bytes(), zero);

    let short = one_frame_webp(5);
    let mut demux = open_ok(&short);
    let packet = demux.read_packet().expect("read").expect("packet");
    assert_eq!(stamp(&packet), (0, 5));
}

#[test]
fn a_truncated_signature_or_body_is_an_error_with_no_packets() {
    assert_open_error(&[0xFF], |error| {
        matches!(
            error,
            OpenError::Truncated {
                expected: 2,
                got: 1
            }
        )
    });
    assert_open_error(&png_bytes()[..4], |error| {
        matches!(
            error,
            OpenError::Truncated {
                expected: 8,
                got: 4
            }
        )
    });
    assert_open_error(b"GIF", |error| {
        matches!(
            error,
            OpenError::Truncated {
                expected: 6,
                got: 3
            }
        )
    });
    assert_open_error(b"RIFF", |error| {
        matches!(
            error,
            OpenError::Truncated {
                expected: 12,
                got: 4
            }
        )
    });
    assert_open_error(&[0xFF, 0xD8], |error| {
        matches!(error, OpenError::Truncated { .. })
    });
    assert_open_error(&png_bytes()[..8], |error| {
        matches!(error, OpenError::Truncated { .. })
    });
    assert_open_error(b"GIF89a", |error| {
        matches!(error, OpenError::Truncated { .. })
    });
    assert_open_error(b"RIFF\x04\x00\x00\x00WEBP", |error| {
        matches!(
            error,
            OpenError::Truncated { .. } | OpenError::Source { .. }
        )
    });
}

#[test]
fn another_family_is_refused_and_a_huge_declared_length_is_not_allocated() {
    let movie = [
        0x00, 0x00, 0x00, 0x18, b'f', b't', b'y', b'p', b'i', b's', b'o', b'm',
    ];
    let reads = Rc::new(Cell::new(0_u64));
    let error = open_counted(&movie, u64::MAX, Rc::clone(&reads)).expect_err("movie");
    assert!(matches!(error, OpenError::Unrecognized { .. }));
    assert!(!error.streams_published());
    assert_eq!(reads.get(), 12);

    let reads = Rc::new(Cell::new(0_u64));
    let error = open_counted(JPEG, u64::MAX, Rc::clone(&reads)).expect_err("cap");
    assert!(matches!(
        error,
        OpenError::HostileSize {
            declared: u64::MAX,
            limit
        } if limit == MAX_FILE_BYTES
    ));
    assert!(!error.streams_published());
    assert_eq!(reads.get(), 12);
}

#[test]
fn a_declared_size_past_the_remaining_bytes_is_hostile_and_publishes_nothing() {
    let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0, 0xFF, 0xFF, 0x00];
    assert_hostile(&jpeg);
    jpeg.clear();
    jpeg.extend([0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n']);
    jpeg.extend(u32::MAX.to_be_bytes());
    jpeg.extend(b"IHDR");
    assert_hostile(&jpeg);

    let mut webp = b"RIFF".to_vec();
    webp.extend(u32::MAX.to_le_bytes());
    webp.extend(b"WEBP");
    assert_hostile(&webp);

    let mut gif = gif_header();
    gif.extend([
        0x2C, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x02, 200, 0x4C,
    ]);
    assert_hostile(&gif);
    assert_hostile(&nested_anmf());
}

#[test]
fn a_bad_png_crc_is_refused() {
    let mut png = png_bytes();
    let idat = png.iter().position(|byte| *byte == b'I').expect("idat");
    png[idat + 8] ^= 0xFF;
    let error = open_bytes(&png).expect_err("crc");
    assert!(matches!(error, OpenError::Source { .. }));
    assert!(!error.streams_published());
    assert!(error.to_string().contains("png chunk crc"));
}

#[test]
fn a_gif_past_the_frame_limit_returns_no_demuxer() {
    let over = gif_frames(usize::try_from(MAX_FRAMES).expect("frames") + 1);
    let error = open_bytes(&over).expect_err("limit");
    assert!(matches!(
        error,
        OpenError::HostileSize {
            declared,
            limit
        } if declared == u64::from(MAX_FRAMES) + 1 && limit == u64::from(MAX_FRAMES)
    ));
    assert!(!error.streams_published());

    let exact = gif_frames(usize::try_from(MAX_FRAMES).expect("frames"));
    let demux = open_ok(&exact);
    assert_eq!(
        demux.frame_count(),
        usize::try_from(MAX_FRAMES).expect("frames")
    );
}

#[test]
fn seek_uses_the_index_and_prerolls_to_the_first_gif_frame() {
    let mut demux = open_ok(JPEG);
    let consumed = demux.consumed();
    let landing = demux
        .seek(SeekTo {
            track: 0,
            time: 0,
            end: None,
        })
        .expect("seek");
    assert_eq!(landing.payload_scans, 0);
    assert!(landing.index_used);
    assert!(matches!(
        landing.landing,
        Landing::Packet { ordinal: 0, .. }
    ));
    let past = demux
        .seek(SeekTo {
            track: 0,
            time: 1,
            end: None,
        })
        .expect("past");
    assert!(matches!(past.landing, Landing::End));
    assert_eq!(past.payload_scans, 0);
    assert_eq!(demux.consumed(), consumed);
    assert!(demux.read_packet().expect("end").is_none());

    let file = unequal_gif();
    let mut demux = open_ok(&file);
    let landing = demux
        .seek(SeekTo {
            track: 0,
            time: 5,
            end: None,
        })
        .expect("mid");
    match landing.landing {
        Landing::Packet {
            ordinal,
            keyframe,
            preroll,
            ..
        } => {
            assert_eq!(ordinal, 0);
            assert!(keyframe);
            assert!(preroll >= 1);
        }
        Landing::End => panic!("mid-file seek landed at the end"),
    }
    assert_eq!(landing.payload_scans, 0);
    let packet = demux.read_packet().expect("read").expect("preroll");
    assert_eq!(packet.bytes(), &file[..42]);
    let end = demux
        .seek(SeekTo {
            track: 0,
            time: 25,
            end: None,
        })
        .expect("end");
    assert!(matches!(end.landing, Landing::End));
}

#[test]
fn a_forward_input_refuses_seek_and_still_returns_the_first_packet() {
    let mut demux =
        StillDemux::open(Input::forward(Cursor::new(JPEG.to_vec()), still_limits())).expect("open");
    let error = demux
        .seek(SeekTo {
            track: 0,
            time: 0,
            end: None,
        })
        .expect_err("forward");
    assert!(matches!(error, ReadError::Seek(SeekError::ForwardOnly)));
    assert!(error.streams_remain());
    assert!(!error.packet_returned());
    let packet = demux.read_packet().expect("read").expect("packet");
    assert_eq!(packet.bytes(), JPEG);
}

#[test]
fn cancel_stops_the_next_read_and_keeps_the_streams() {
    let mut demux = open_ok(JPEG);
    demux.cancel();
    let error = demux.read_packet().expect_err("cancel");
    assert!(matches!(error, ReadError::Cancelled));
    assert!(error.streams_remain());
    assert!(!error.packet_returned());
    assert_eq!(demux.streams().len(), 1);
    let seek = demux
        .seek(SeekTo {
            track: 0,
            time: 0,
            end: None,
        })
        .expect_err("seek");
    assert!(matches!(seek, ReadError::Cancelled));
}

#[test]
fn mux_writes_a_single_picture_back_unchanged() {
    for (bytes, codec) in [
        (JPEG, CodecId::Jpeg),
        (png_bytes().as_slice(), CodecId::Png),
        (still_webp().as_slice(), CodecId::Webp),
        (lossless_webp().as_slice(), CodecId::Webp),
    ] {
        let mut demux = open_ok(bytes);
        let packet = demux.read_packet().expect("read").expect("packet");
        let written = finish_one(&packet, codec, 0, 0);
        assert_eq!(written, bytes);
        let mut again = open_ok(&written);
        let round = again.read_packet().expect("read").expect("packet");
        assert_eq!(round.bytes(), bytes);
        assert!(round.timestamp().is_missing());
    }
}

#[test]
fn mux_writes_gif_and_animated_webp_frames_and_a_loop_count() {
    let file = unequal_gif();
    let mut demux = open_ok(&file);
    let packets = read_all(&mut demux);
    let written = finish_timed(&packets, CodecId::Gif, GIF_TIMESCALE, 25);
    assert_eq!(written, file);

    let mut mux = StillMux::new();
    add_stream(&mut mux, CodecId::Gif, GIF_TIMESCALE);
    mux.set_loop_count(7);
    for packet in &packets {
        mux.write_packet(packet).expect("write");
    }
    let finished = finish_mux(&mut mux, GIF_TIMESCALE, 25);
    assert_eq!(finished.mode(), MuxMode::Progressive);
    assert_eq!(finished.index(), IndexPlace::Absent);
    assert!(finished.interleaved());
    let bytes = mux.bytes().expect("file").to_vec();
    assert_ne!(bytes, file);
    let again = open_ok(&bytes);
    assert_eq!(again.loop_count(), Some(7));
    assert_eq!(again.media_end(), Some(25));
    assert_eq!(again.frame_count(), 2);

    let webp = animated_webp(100, 0x02, 400, 0x00);
    let mut demux = open_ok(&webp);
    let packets = read_all(&mut demux);
    let written = finish_timed(&packets, CodecId::Webp, WEBP_TIMESCALE, 500);
    assert_eq!(written, webp);
    let mut mux = StillMux::new();
    add_stream(&mut mux, CodecId::Webp, WEBP_TIMESCALE);
    mux.set_loop_count(3);
    for packet in &packets {
        mux.write_packet(packet).expect("write");
    }
    let _ = finish_mux(&mut mux, WEBP_TIMESCALE, 500);
    let bytes = mux.bytes().expect("webp").to_vec();
    let again = open_ok(&bytes);
    assert_eq!(again.loop_count(), Some(3));
    assert_eq!(again.frame_count(), 2);
    assert_eq!(again.media_end(), Some(500));
}

#[test]
fn a_failed_mux_has_no_bytes_and_no_durations() {
    let mut mux = StillMux::new();
    add_stream(&mut mux, CodecId::Jpeg, 0);
    let packet = Packet::try_new(
        0,
        b"not a jpeg".to_vec(),
        Timestamp::missing(),
        Timescale::new(0),
        true,
    )
    .expect("packet");
    mux.write_packet(&packet).expect("write");
    let failed = mux.finish(&[duration(0, 0)]).expect_err("finish");
    assert!(!failed.is_complete());
    assert!(failed.durations().is_none());
    assert!(mux.bytes().is_none());
    assert_eq!(mux.status(), MuxStatus::Failed);

    let mut mux = StillMux::new();
    add_stream(&mut mux, CodecId::Jpeg, 0);
    let hostile = {
        let mut bytes = vec![0xFF, 0xD8, 0xFF, 0xE0, 0xFF, 0xFF];
        bytes.push(0);
        Packet::try_new(0, bytes, Timestamp::missing(), Timescale::new(0), true).expect("packet")
    };
    mux.write_packet(&hostile).expect("write");
    let failed = mux.finish(&[duration(0, 0)]).expect_err("hostile");
    assert!(matches!(failed.reason(), MuxError::HostileSize { .. }));
    assert!(failed.durations().is_none());
    assert!(mux.bytes().is_none());
}

#[test]
fn mux_rejects_a_second_stream_and_a_codec_this_family_does_not_write() {
    let mut mux = StillMux::new();
    add_stream(&mut mux, CodecId::Jpeg, 0);
    let error = mux
        .add_stream(new_stream(CodecId::Png, 0))
        .expect_err("second");
    assert!(matches!(error, MuxError::TooManyTracks { limit: 1, .. }));
    assert_eq!(mux.status(), MuxStatus::Failed);

    let mut mux = StillMux::new();
    let error = mux
        .add_stream(new_stream(CodecId::H264, 0))
        .expect_err("h264");
    assert!(matches!(error, MuxError::UnknownCodec(tag) if tag.tag() == b"other"));
    assert_eq!(mux.status(), MuxStatus::Failed);
    let failed = mux.finish(&[]).expect_err("finish");
    assert!(failed.durations().is_none());

    let mut mux = StillMux::new();
    let unknown = StreamParams::video(
        StreamCodec::Unknown(UnknownCodec::new(b"xyz").expect("tag")),
        FrameSize::try_new(1, 1).expect("size"),
        Vec::new(),
    );
    let error = mux
        .add_stream(NewStream {
            params: unknown,
            timescale: Timescale::new(0),
            language: None,
        })
        .expect_err("unknown");
    assert!(matches!(error, MuxError::UnknownCodec(tag) if tag.tag() == b"xyz"));
}

#[test]
fn interleave_keeps_a_clockless_picture_and_refuses_a_timed_packet_with_no_time() {
    let mut mux = StillMux::new();
    add_stream(&mut mux, CodecId::Jpeg, 0);
    let packet = Packet::try_new(
        0,
        JPEG.to_vec(),
        Timestamp::missing(),
        Timescale::new(0),
        true,
    )
    .expect("packet");
    mux.write_packet(&packet).expect("write");
    mux.interleave().expect("clockless");
    assert_eq!(mux.status(), MuxStatus::Building);

    let mut mux = StillMux::new();
    add_stream(&mut mux, CodecId::Gif, GIF_TIMESCALE);
    let packet = Packet::try_new(
        0,
        unequal_gif(),
        Timestamp::missing(),
        Timescale::new(GIF_TIMESCALE),
        true,
    )
    .expect("packet");
    mux.write_packet(&packet).expect("write");
    let error = mux.interleave().expect_err("time");
    assert!(matches!(error, MuxError::MissingTimestamp));
    assert_eq!(mux.status(), MuxStatus::Building);
}

#[test]
fn the_duration_handed_to_finish_has_to_match_the_packets() {
    let mut mux = StillMux::new();
    add_stream(&mut mux, CodecId::Jpeg, 0);
    let packet = Packet::try_new(
        0,
        JPEG.to_vec(),
        Timestamp::missing(),
        Timescale::new(0),
        true,
    )
    .expect("packet");
    mux.write_packet(&packet).expect("write");
    let failed = mux.finish(&[duration(25, 1)]).expect_err("rate");
    assert!(matches!(
        failed.reason(),
        MuxError::TimescaleMismatch { .. } | MuxError::DurationMismatch { .. }
    ));
    assert!(failed.durations().is_none());
    assert!(mux.bytes().is_none());
}

fn frame_control(disposal: u8, blend: Option<u8>) -> crate::FrameControl {
    crate::FrameControl {
        disposal,
        webp_blend: blend,
    }
}

fn stamp(packet: &Packet) -> (i64, i64) {
    (
        packet.timestamp().presentation().expect("pts"),
        packet.timestamp().duration().expect("duration"),
    )
}

fn open_bytes(bytes: &[u8]) -> Result<StillDemux<Cursor<Vec<u8>>>, OpenError> {
    let len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    StillDemux::open(Input::random(
        Cursor::new(bytes.to_vec()),
        len,
        still_limits(),
    ))
}

fn open_ok(bytes: &[u8]) -> StillDemux<Cursor<Vec<u8>>> {
    open_bytes(bytes).expect("open")
}

fn read_all(demux: &mut impl Demux) -> Vec<Packet> {
    let mut packets = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        packets.push(packet);
    }
    packets
}

fn assert_open_error(bytes: &[u8], pred: impl Fn(&OpenError) -> bool) {
    let error = open_bytes(bytes).expect_err("open");
    assert!(pred(&error), "{error}");
    assert!(!error.streams_published());
}

fn assert_hostile(bytes: &[u8]) {
    let error = open_bytes(bytes).expect_err("hostile");
    assert!(matches!(error, OpenError::HostileSize { .. }), "{error}");
    assert!(!error.streams_published());
}

fn png_bytes() -> Vec<u8> {
    hex(PNG_HEX)
}

fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&text[index..index + 2], 16).expect("hex"))
        .collect()
}

fn gif_header() -> Vec<u8> {
    let mut out = b"GIF89a".to_vec();
    out.extend([0x01, 0x00, 0x01, 0x00, 0x80, 0x00, 0x00]);
    out.extend([0x00, 0x00, 0x00, 0xFF, 0x00, 0x00]);
    out
}

fn gce(packed: u8, delay: u16) -> [u8; 8] {
    let delay = delay.to_le_bytes();
    [0x21, 0xF9, 0x04, packed, delay[0], delay[1], 0x00, 0x00]
}

fn netscape(count: u16) -> [u8; 19] {
    let mut block = *b"!\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00";
    let count = count.to_le_bytes();
    block[16] = count[0];
    block[17] = count[1];
    block
}

fn unequal_gif() -> Vec<u8> {
    let mut out = gif_header();
    out.extend(gce(0x04, 5));
    out.extend(IMAGE);
    out.extend(gce(0x08, 20));
    out.extend(IMAGE);
    out.push(0x3B);
    out
}

fn gif_without_control() -> Vec<u8> {
    let mut out = gif_header();
    out.extend(IMAGE);
    out.push(0x3B);
    out
}

fn gif_with_delay(delay: u16, packed: u8) -> Vec<u8> {
    let mut out = gif_header();
    out.extend(gce(packed, delay));
    out.extend(IMAGE);
    out.push(0x3B);
    out
}

fn gif_frames(count: usize) -> Vec<u8> {
    let mut out = gif_header();
    for index in 0..count {
        let delay = if index % 2 == 0 { 5 } else { 20 };
        out.extend(gce(0x04, delay));
        out.extend(IMAGE);
    }
    out.push(0x3B);
    out
}

fn riff(body: &[u8]) -> Vec<u8> {
    let mut out = b"RIFF".to_vec();
    let size = u32::try_from(body.len()).expect("riff");
    out.extend(size.to_le_bytes());
    out.extend(body);
    out
}

fn chunk(fourcc: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = fourcc.to_vec();
    let size = u32::try_from(data.len()).expect("chunk");
    out.extend(size.to_le_bytes());
    out.extend(data);
    if data.len() % 2 == 1 {
        out.push(0);
    }
    out
}

fn vp8_payload() -> [u8; 10] {
    [0xD0, 0x01, 0x00, 0x9D, 0x01, 0x2A, 0x08, 0x00, 0x08, 0x00]
}

fn still_webp() -> Vec<u8> {
    let mut body = b"WEBP".to_vec();
    body.extend(chunk(b"VP8 ", &vp8_payload()));
    riff(&body)
}

fn lossless_webp() -> Vec<u8> {
    let mut body = b"WEBP".to_vec();
    body.extend(chunk(b"VP8L", &[0x2F, 0x03, 0xC0, 0x00, 0x00]));
    riff(&body)
}

fn vp8x(animated: bool) -> [u8; 10] {
    let mut data = [0_u8; 10];
    data[0] = if animated { 0x02 } else { 0 };
    data[4] = 7;
    data[7] = 7;
    data
}

fn extended_still_webp() -> Vec<u8> {
    let mut body = b"WEBP".to_vec();
    body.extend(chunk(b"VP8X", &vp8x(false)));
    body.extend(chunk(b"VP8 ", &vp8_payload()));
    riff(&body)
}

fn anim_chunk(loop_count: u16) -> [u8; 6] {
    let loop_count = loop_count.to_le_bytes();
    [0xFF, 0xFF, 0xFF, 0xFF, loop_count[0], loop_count[1]]
}

fn anmf(duration: u32, flags: u8, inner: &[u8]) -> Vec<u8> {
    let mut header = [0_u8; 16];
    header[6] = 7;
    header[9] = 7;
    let duration = duration.to_le_bytes();
    header[12] = duration[0];
    header[13] = duration[1];
    header[14] = duration[2];
    header[15] = flags;
    let mut data = header.to_vec();
    data.extend(inner);
    chunk(b"ANMF", &data)
}

fn animated_webp(first: u32, first_flags: u8, second: u32, second_flags: u8) -> Vec<u8> {
    let vp8 = chunk(b"VP8 ", &vp8_payload());
    let mut body = b"WEBP".to_vec();
    body.extend(chunk(b"VP8X", &vp8x(true)));
    body.extend(chunk(b"ANIM", &anim_chunk(0)));
    body.extend(anmf(first, first_flags, &vp8));
    body.extend(anmf(second, second_flags, &vp8));
    riff(&body)
}

fn one_frame_webp(duration: u32) -> Vec<u8> {
    let vp8 = chunk(b"VP8 ", &vp8_payload());
    let mut body = b"WEBP".to_vec();
    body.extend(chunk(b"VP8X", &vp8x(true)));
    body.extend(chunk(b"ANIM", &anim_chunk(1)));
    body.extend(anmf(duration, 0x00, &vp8));
    riff(&body)
}

fn nested_anmf() -> Vec<u8> {
    let inner = chunk(b"ANMF", &[0_u8; 16]);
    let mut body = b"WEBP".to_vec();
    body.extend(chunk(b"VP8X", &vp8x(true)));
    body.extend(chunk(b"ANIM", &anim_chunk(0)));
    body.extend(anmf(100, 0, &inner));
    riff(&body)
}

fn second_anmf(file: &[u8]) -> usize {
    let first = file
        .windows(4)
        .position(|window| window == b"ANMF")
        .expect("first");
    file[first + 4..]
        .windows(4)
        .position(|window| window == b"ANMF")
        .map(|index| first + 4 + index)
        .expect("second")
}

#[derive(Debug)]
struct Counted {
    inner: Cursor<Vec<u8>>,
    reads: Rc<Cell<u64>>,
}

impl Read for Counted {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let amount = self.inner.read(buf)?;
        self.reads.set(
            self.reads
                .get()
                .saturating_add(u64::try_from(amount).unwrap_or(0)),
        );
        Ok(amount)
    }
}

impl Seek for Counted {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

fn open_counted(
    bytes: &[u8],
    length: u64,
    reads: Rc<Cell<u64>>,
) -> Result<StillDemux<Counted>, OpenError> {
    StillDemux::open(Input::random(
        Counted {
            inner: Cursor::new(bytes.to_vec()),
            reads,
        },
        length,
        still_limits(),
    ))
}

fn new_stream(codec: CodecId, timescale: u32) -> NewStream {
    NewStream {
        params: StreamParams::video(
            StreamCodec::Known(codec),
            FrameSize::try_new(1, 1).expect("size"),
            Vec::new(),
        ),
        timescale: Timescale::new(timescale),
        language: None,
    }
}

fn add_stream(mux: &mut StillMux, codec: CodecId, timescale: u32) {
    mux.add_stream(new_stream(codec, timescale))
        .expect("stream");
}

fn duration(timescale: u32, ticks: i64) -> TrackDuration {
    TrackDuration {
        track: TrackId::new(0),
        timescale: Timescale::new(timescale),
        ticks,
    }
}

fn finish_mux(mux: &mut StillMux, timescale: u32, ticks: i64) -> cutlass_container::FinishedMux {
    mux.finish(&[duration(timescale, ticks)]).expect("finish")
}

fn finish_one(packet: &Packet, codec: CodecId, timescale: u32, ticks: i64) -> Vec<u8> {
    let mut mux = StillMux::new();
    add_stream(&mut mux, codec, timescale);
    mux.write_packet(packet).expect("write");
    let _ = finish_mux(&mut mux, timescale, ticks);
    mux.bytes().expect("bytes").to_vec()
}

fn finish_timed(packets: &[Packet], codec: CodecId, timescale: u32, ticks: i64) -> Vec<u8> {
    let mut mux = StillMux::new();
    add_stream(&mut mux, codec, timescale);
    for packet in packets {
        mux.write_packet(packet).expect("write");
    }
    let _ = finish_mux(&mut mux, timescale, ticks);
    mux.bytes().expect("bytes").to_vec()
}

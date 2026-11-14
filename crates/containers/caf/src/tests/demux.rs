//! Demux of well-formed files: packets, times, extradata, trim and seek.

use cutlass_codec::{CodecId, StreamCodec};
use cutlass_container::{Demux, Landing, ReadError, SeekError, SeekTo};

use super::build::{
    aac_cookie, chunk, data_body, desc_body, file, open, open_forward, pakt_body, payload_of,
    pcm_file, sized_chunk,
};

fn seek(time: i64) -> SeekTo {
    SeekTo {
        track: 0,
        time,
        end: None,
    }
}

#[test]
fn pcm_is_returned_in_whole_frames_of_up_to_4096_bytes() {
    let (bytes, payload) = pcm_file(2_500);
    let mut demux = open(&bytes).expect("open");
    let stream = demux.streams()[0].clone();
    assert_eq!(stream.params().codec(), &StreamCodec::Known(CodecId::Pcm));
    assert_eq!(stream.params().extradata(), [1, 0, 0]);
    assert_eq!(stream.params().sample_rate(), Some(48_000));
    assert_eq!(stream.params().channels(), Some(2));
    assert_eq!(stream.timescale().ticks_per_second(), 48_000);
    assert!(stream.edits().is_empty());
    assert_eq!(stream.trim().encoder_delay(), None);
    assert_eq!(demux.packet_count(), Some(2_500));
    let mut joined: Vec<u8> = Vec::new();
    let mut expect_pts = 0;
    let mut sizes = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        assert_eq!(packet.timestamp().presentation(), Some(expect_pts));
        assert_eq!(packet.timestamp().decode(), Some(expect_pts));
        let frames = packet.timestamp().duration().expect("duration");
        assert_eq!(
            frames * 4,
            i64::try_from(packet.bytes().len()).expect("len")
        );
        assert!(packet.keyframe());
        expect_pts += frames;
        sizes.push(packet.bytes().len());
        joined.extend(packet.bytes());
    }
    assert_eq!(sizes, [4096, 4096, 1808]);
    assert_eq!(joined, payload);
    assert_eq!(expect_pts, 2_500);
    assert!(demux.read_packet().expect("end").is_none());
}

#[test]
fn a_pcm_seek_lands_on_the_frame_and_reads_no_payload() {
    let (bytes, payload) = pcm_file(3_000);
    let mut demux = open(&bytes).expect("open");
    let before = demux.consumed();
    let landing = demux.seek(seek(1_001)).expect("seek");
    assert_eq!(demux.consumed(), before);
    assert_eq!(landing.payload_scans, 0);
    let Landing::Packet {
        ordinal,
        time,
        preroll,
        byte_position,
        ..
    } = landing.landing
    else {
        panic!("landing {landing:?}");
    };
    assert_eq!((ordinal, time, preroll), (1_001, 1_001, 0));
    assert_eq!(byte_position, demux.data_offset() + 4_004);
    let packet = demux.read_packet().expect("read").expect("packet");
    assert_eq!(packet.timestamp().presentation(), Some(1_001));
    assert_eq!(&packet.bytes()[..4], &payload[4_004..4_008]);
    assert_eq!(demux.seek(seek(3_000)).expect("end").landing, Landing::End);
    assert!(demux.read_packet().expect("end").is_none());
    let first = demux.seek(seek(-5)).expect("before");
    assert!(matches!(
        first.landing,
        Landing::Packet {
            time: 0,
            preroll: 0,
            ..
        }
    ));
    let last = demux.seek(seek(2_999)).expect("last");
    assert!(matches!(last.landing, Landing::Packet { time: 2_999, .. }));
    let packet = demux.read_packet().expect("read").expect("last packet");
    assert_eq!(packet.bytes(), &payload[11_996..]);
    let ended = SeekTo {
        track: 0,
        time: 10,
        end: Some(10),
    };
    assert_eq!(demux.seek(ended).expect("end").landing, Landing::End);
}

#[test]
fn data_of_size_minus_one_runs_to_the_end_of_the_file() {
    let (_, payload) = pcm_file(1_500);
    let bytes = file(&[
        chunk(b"desc", &desc_body(48_000.0, b"lpcm", 2, 4, 1, 2, 16)),
        sized_chunk(b"data", -1, &data_body(&payload)),
    ]);
    let mut demux = open(&bytes).expect("open");
    assert!(demux.is_open_ended());
    assert_eq!(demux.packet_count(), Some(1_500));
    let mut joined: Vec<u8> = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        joined.extend(packet.bytes());
    }
    assert_eq!(joined, payload);

    // A forward reader does not know where the file ends: it reads whole
    // frames until the source stops, and a partial last frame is dropped.
    let mut cut = bytes.clone();
    cut.push(0xAB);
    let mut forward = open_forward(&cut).expect("forward");
    assert_eq!(forward.packet_count(), None);
    let mut joined: Vec<u8> = Vec::new();
    let mut pts = 0;
    while let Some(packet) = forward.read_packet().expect("read") {
        assert_eq!(packet.timestamp().presentation(), Some(pts));
        pts += packet.timestamp().duration().expect("duration");
        joined.extend(packet.bytes());
    }
    assert_eq!(joined, payload);
    assert_eq!(
        forward.seek(seek(0)),
        Err(ReadError::Seek(SeekError::ForwardOnly))
    );
}

fn aac_file(sizes: &[u32], pakt_after_data: bool) -> Vec<u8> {
    let payload = payload_of(sizes);
    let pakt = chunk(
        b"pakt",
        &pakt_body(
            i64::try_from(sizes.len()).expect("len"),
            i64::try_from(sizes.len() * 1024 - 2112 - 100).expect("valid"),
            2112,
            100,
            sizes,
        ),
    );
    let mut chunks = vec![
        chunk(b"desc", &desc_body(48_000.0, b"aac ", 0, 0, 1024, 2, 0)),
        chunk(b"kuki", &aac_cookie()),
        chunk(b"chan", &[0, 0x65, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]),
    ];
    if pakt_after_data {
        chunks.push(chunk(b"data", &data_body(&payload)));
        chunks.push(pakt);
    } else {
        chunks.push(pakt);
        chunks.push(chunk(b"free", &[0; 17]));
        chunks.push(chunk(b"data", &data_body(&payload)));
    }
    file(&chunks)
}

fn sizes(count: u32) -> Vec<u32> {
    (0..count).map(|index| 100 + (index * 37) % 300).collect()
}

#[test]
fn variable_packets_follow_the_table_wherever_it_sits() {
    let sizes = sizes(200);
    let payload = payload_of(&sizes);
    for after in [false, true] {
        let mut demux = open(&aac_file(&sizes, after)).expect("open");
        let stream = demux.streams()[0].clone();
        assert_eq!(stream.params().codec(), &StreamCodec::Known(CodecId::Aac));
        assert_eq!(stream.params().extradata(), [0x11, 0x90]);
        assert_eq!(stream.trim().encoder_delay(), Some(2112));
        assert_eq!(stream.trim().trailing_padding(), Some(100));
        assert_eq!(stream.edits().len(), 1);
        assert_eq!(stream.edits()[0].media_time(), Some(2112));
        assert_eq!(stream.edits()[0].duration(), 200 * 1024 - 2112 - 100);
        assert_eq!(demux.channel_layout().expect("chan").tag, crate::TAG_STEREO);
        let mut at = 0;
        for (index, size) in sizes.iter().enumerate() {
            let packet = demux.read_packet().expect("read").expect("packet");
            let index = i64::try_from(index).expect("index");
            assert_eq!(packet.timestamp().presentation(), Some(index * 1024));
            assert_eq!(packet.timestamp().duration(), Some(1024));
            let len = usize::try_from(*size).expect("size");
            assert_eq!(packet.bytes(), &payload[at..at + len]);
            at += len;
        }
        assert!(demux.read_packet().expect("end").is_none());
    }
}

#[test]
fn a_variable_seek_uses_the_table_and_reads_no_payload() {
    let sizes = sizes(300);
    let payload = payload_of(&sizes);
    let mut demux = open(&aac_file(&sizes, false)).expect("open");
    for (time, packet, preroll) in [
        (0, 0_u32, 0_u32),
        (1024, 1, 0),
        (1025, 1, 1),
        (64 * 1024 + 5, 64, 1),
        (299 * 1024 + 1023, 299, 1),
        (-1, 0, 0),
    ] {
        let before = demux.consumed();
        let landing = demux.seek(seek(time)).expect("seek");
        assert_eq!(demux.consumed(), before, "{time}");
        let Landing::Packet {
            ordinal,
            time: landed,
            preroll: got,
            byte_position,
            ..
        } = landing.landing
        else {
            panic!("{time}: {landing:?}");
        };
        assert_eq!((ordinal, got), (packet, preroll), "{time}");
        assert_eq!(landed, i64::from(packet) * 1024);
        let offset: u32 = sizes[..usize::try_from(packet).expect("index")]
            .iter()
            .sum();
        assert_eq!(byte_position, demux.data_offset() + u64::from(offset));
        let read = demux.read_packet().expect("read").expect("packet");
        assert_eq!(read.timestamp().presentation(), Some(landed));
        let start = usize::try_from(offset).expect("offset");
        assert_eq!(read.bytes()[0], payload[start]);
    }
    assert_eq!(
        demux.seek(seek(300 * 1024)).expect("end").landing,
        Landing::End
    );
    assert!(demux.read_packet().expect("end").is_none());
}

#[test]
fn variable_frames_and_sizes_come_in_pairs() {
    let entries: Vec<(u32, u32)> = (0..150)
        .map(|i| (50 + i % 7, 4096 + (i % 3) * 512))
        .collect();
    let mut table = Vec::new();
    for (size, frames) in &entries {
        table.push(*size);
        table.push(*frames);
    }
    let sizes: Vec<u32> = entries.iter().map(|entry| entry.0).collect();
    let total: u32 = entries.iter().map(|entry| entry.1).sum();
    let bytes = file(&[
        chunk(b"desc", &desc_body(44_100.0, b"flac", 0, 0, 0, 2, 0)),
        chunk(b"data", &data_body(&payload_of(&sizes))),
        chunk(b"pakt", &pakt_body(150, i64::from(total), 0, 0, &table)),
    ]);
    let mut demux = open(&bytes).expect("open");
    assert_eq!(
        demux.streams()[0].params().codec(),
        &StreamCodec::Known(CodecId::Flac)
    );
    assert!(demux.streams()[0].params().extradata().is_empty());
    assert!(demux.streams()[0].edits().is_empty());
    assert_eq!(demux.total_frames(), Some(u64::from(total)));
    let mut pts = 0_i64;
    let mut starts = Vec::new();
    for (size, frames) in &entries {
        let packet = demux.read_packet().expect("read").expect("packet");
        assert_eq!(packet.timestamp().presentation(), Some(pts));
        assert_eq!(packet.timestamp().duration(), Some(i64::from(*frames)));
        assert_eq!(packet.bytes().len(), usize::try_from(*size).expect("size"));
        starts.push(pts);
        pts += i64::from(*frames);
    }
    for (index, start) in starts.iter().enumerate() {
        for probe in [*start, *start + 1, *start + 4095] {
            let landing = demux.seek(seek(probe)).expect("seek");
            let Landing::Packet { ordinal, time, .. } = landing.landing else {
                panic!("{probe}");
            };
            let expect = if probe >= starts.get(index + 1).copied().unwrap_or(i64::MAX) {
                index + 1
            } else {
                index
            };
            assert_eq!(
                usize::try_from(ordinal).expect("ordinal"),
                expect,
                "{probe}"
            );
            assert_eq!(time, starts[expect]);
            assert!(landing.entries_visited <= 80, "{}", landing.entries_visited);
        }
    }
}

#[test]
fn constant_bytes_with_variable_frames_read_frames_from_the_table() {
    let frames = [10_u32, 20, 30, 40];
    let bytes = file(&[
        chunk(b"desc", &desc_body(8_000.0, b"zzzz", 0, 3, 0, 1, 0)),
        chunk(b"pakt", &pakt_body(4, 100, 0, 0, &frames)),
        chunk(b"data", &data_body(&payload_of(&[3, 3, 3, 3]))),
    ]);
    let mut demux = open(&bytes).expect("open");
    assert!(demux.streams()[0].params().codec().is_unknown());
    let mut pts = 0;
    for count in frames {
        let packet = demux.read_packet().expect("read").expect("packet");
        assert_eq!(packet.bytes().len(), 3);
        assert_eq!(packet.timestamp().presentation(), Some(pts));
        pts += i64::from(count);
    }
    let landing = demux.seek(seek(35)).expect("seek");
    assert!(matches!(
        landing.landing,
        Landing::Packet {
            ordinal: 2,
            time: 30,
            preroll: 1,
            ..
        }
    ));
}

#[test]
fn an_unknown_constant_format_is_still_framed() {
    // IMA4: 34 bytes and 64 frames per channel per packet.
    let payload = payload_of(&[34, 34, 34]);
    let bytes = file(&[
        chunk(b"desc", &desc_body(44_100.0, b"ima4", 0, 34, 64, 1, 0)),
        chunk(b"pakt", &pakt_body(0, 150, 0, 42, &[])),
        chunk(b"data", &data_body(&payload)),
    ]);
    let mut demux = open(&bytes).expect("open");
    let stream = demux.streams()[0].clone();
    assert_eq!(
        stream.params().codec(),
        &StreamCodec::Unknown(cutlass_codec::UnknownCodec::new(b"ima4".to_vec()).expect("tag"))
    );
    assert_eq!(stream.trim().trailing_padding(), Some(42));
    assert_eq!(stream.edits()[0].duration(), 150);
    for index in 0..3 {
        let packet = demux.read_packet().expect("read").expect("packet");
        assert_eq!(packet.bytes().len(), 34);
        assert_eq!(packet.timestamp().presentation(), Some(index * 64));
        assert_eq!(packet.timestamp().duration(), Some(64));
    }
    assert!(demux.read_packet().expect("end").is_none());
    let landing = demux.seek(seek(130)).expect("seek");
    assert!(matches!(
        landing.landing,
        Landing::Packet {
            ordinal: 2,
            time: 128,
            preroll: 1,
            ..
        }
    ));
}

#[test]
fn a_forward_input_reads_a_table_before_data_and_refuses_one_after() {
    let sizes = sizes(20);
    let mut demux = open_forward(&aac_file(&sizes, false)).expect("forward");
    let mut count = 0;
    while demux.read_packet().expect("read").is_some() {
        count += 1;
    }
    assert_eq!(count, 20);
    assert_eq!(
        demux.seek(seek(0)),
        Err(ReadError::Seek(SeekError::ForwardOnly))
    );
    let error = open_forward(&aac_file(&sizes, true))
        .err()
        .expect("refused");
    assert!(error.to_string().contains("before data"), "{error}");
}

#[test]
fn info_chan_and_unknown_chunks_are_kept_or_skipped() {
    let mut info = 2_u32.to_be_bytes().to_vec();
    info.extend(b"title\0Song\0encoder\0Test\0");
    let (pcm, payload) = pcm_file(10);
    let data_at = pcm
        .windows(4)
        .position(|window| window == b"data")
        .expect("data");
    let mut bytes = pcm[..data_at].to_vec();
    bytes.extend(chunk(b"info", &info));
    bytes.extend(chunk(b"uuid", &[7; 40]));
    bytes.extend(chunk(b"strg", &[0, 0, 0, 0]));
    bytes.extend(&pcm[data_at..]);
    bytes.extend(chunk(b"mark", &[0; 8]));
    let mut demux = open(&bytes).expect("open");
    assert_eq!(demux.info().len(), 2);
    assert_eq!(demux.info()[0].key, "title");
    assert_eq!(demux.info()[1].value, "Test");
    let kinds: Vec<[u8; 4]> = demux.chunks().iter().map(|chunk| chunk.kind).collect();
    assert_eq!(
        kinds,
        [*b"desc", *b"info", *b"uuid", *b"strg", *b"data", *b"mark"]
    );
    assert_eq!(demux.stopped_at(), None);
    assert_eq!(demux.edit_count(), 1);
    let packet = demux.read_packet().expect("read").expect("packet");
    assert_eq!(packet.bytes(), payload);
}

#[test]
fn a_bad_chunk_after_data_ends_the_walk_and_is_reported() {
    // The tail of a macOS system sound: after `data` and `info`, a chunk
    // header one byte off claims an absurd size.
    let (pcm, payload) = pcm_file(10);
    let mut bytes = pcm.clone();
    let tail = bytes.len();
    bytes.extend(sized_chunk(b"\0ID3", 0x2000_0000_0000_0000, &[0; 20]));
    let mut demux = open(&bytes).expect("open");
    assert_eq!(demux.stopped_at(), Some(u64::try_from(tail).expect("tail")));
    assert_eq!(
        demux.read_packet().expect("read").expect("packet").bytes(),
        payload
    );
    let mut negative = pcm.clone();
    negative.extend(sized_chunk(b"junk", -5, &[]));
    assert!(open(&negative).expect("open").stopped_at().is_some());
    let mut short = pcm;
    short.extend([1, 2, 3, 4, 5]);
    assert!(open(&short).expect("open").stopped_at().is_some());
}

#[test]
fn opus_gets_an_opus_head_with_the_pre_skip_at_48_khz() {
    let sizes = [60_u32, 61, 62];
    let bytes = file(&[
        chunk(b"desc", &desc_body(24_000.0, b"opus", 0, 0, 480, 1, 0)),
        chunk(b"kuki", &[0; 28]),
        chunk(b"pakt", &pakt_body(3, 1_200, 156, 84, &sizes)),
        chunk(b"data", &data_body(&payload_of(&sizes))),
    ]);
    let demux = open(&bytes).expect("open");
    let stream = &demux.streams()[0];
    assert_eq!(stream.params().codec(), &StreamCodec::Known(CodecId::Opus));
    let head = stream.params().extradata();
    assert_eq!(&head[..8], b"OpusHead");
    assert_eq!(head[9], 1);
    assert_eq!(u16::from_le_bytes([head[10], head[11]]), 312);
    assert_eq!(stream.timescale().ticks_per_second(), 24_000);
    assert_eq!(demux.cookie(), Some(&[0_u8; 28][..]));
}

#[test]
fn reads_and_seeks_stop_after_cancel() {
    let (bytes, _) = pcm_file(10);
    let mut demux = open(&bytes).expect("open");
    demux.cancel();
    assert_eq!(demux.read_packet(), Err(ReadError::Cancelled));
    assert_eq!(demux.seek(seek(0)), Err(ReadError::Cancelled));
    assert_eq!(
        demux.seek(SeekTo {
            track: 1,
            time: 0,
            end: None
        }),
        Err(ReadError::Cancelled)
    );
}

#[test]
fn a_file_with_no_packets_opens_and_refuses_to_seek() {
    let bytes = file(&[
        chunk(b"desc", &desc_body(48_000.0, b"aac ", 0, 0, 1024, 2, 0)),
        chunk(b"kuki", &aac_cookie()),
        chunk(b"pakt", &pakt_body(0, 0, 0, 0, &[])),
        chunk(b"data", &data_body(&[])),
    ]);
    let mut demux = open(&bytes).expect("open");
    assert_eq!(demux.packet_count(), Some(0));
    assert!(demux.read_packet().expect("read").is_none());
    assert_eq!(demux.seek(seek(0)), Err(ReadError::Seek(SeekError::Empty)));
    let track = SeekTo {
        track: 3,
        time: 0,
        end: None,
    };
    assert_eq!(
        demux.seek(track),
        Err(ReadError::Seek(SeekError::UnknownTrack))
    );
}

#[test]
fn kept_chunks_are_read_within_their_limit_and_others_are_only_listed() {
    use super::build::{chunk, data_body, desc_body, file, open_forward};
    use crate::MAX_KEPT_BYTES;
    let desc = chunk(b"desc", &desc_body(8_000.0, b"lpcm", 0, 2, 1, 1, 16));
    let big = usize::try_from(MAX_KEPT_BYTES).expect("fits") - 20;
    let bytes = file(&[
        desc,
        chunk(b"uuid", &[1; 15]),
        chunk(b"midi", &vec![2; big]),
        chunk(b"uuid", &[3; 16]),
        chunk(b"mark", &[4; 8]),
        chunk(b"ovvw", &[5; 16]),
        chunk(b"data", &data_body(&[0, 1, 0, 2])),
        chunk(b"strg", &[0, 0, 0, 0]),
    ]);
    let demux = open(&bytes).expect("open");
    let kept: Vec<([u8; 4], usize)> = demux
        .kept_chunks()
        .iter()
        .map(|chunk| (chunk.kind(), chunk.body().len()))
        .collect();
    // A uuid shorter than its UUID is dropped; the second uuid fits the
    // budget, the mark does not; ovvw is not kept; strg after data is.
    assert_eq!(kept, vec![(*b"midi", big), (*b"uuid", 16), (*b"strg", 4)]);
    assert_eq!(demux.chunks().len(), 8);
    let forward = open_forward(&bytes).expect("forward");
    assert_eq!(forward.kept_chunks().len(), 2);
}

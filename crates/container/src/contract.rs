use crate::{
    Demux, Edit, FinishedMux, Gap, HeaderVerdict, IndexEntry, IndexPlace, Input, Landing, Limits,
    ListDemux, ListRequest, Mux, MuxDraft, MuxError, MuxMode, MuxStatus, NewStream, OpenError,
    PacketPlan, ReadError, SeekError, SeekTo, Stream, TrackDuration, TrackId, Trim,
};
use cutlass_codec::{
    CodecId, ContainerId, FrameSize, Packet, StreamCodec, StreamParams, Timescale, Timestamp,
    UnknownCodec,
};
use std::io::Cursor;

fn scale(ticks: u32) -> Timescale {
    Timescale::new(ticks)
}

fn time(presentation: i64, duration: i64) -> Timestamp {
    Timestamp::try_new(None, Some(presentation), Some(duration)).expect("time")
}

fn packet(stream: u32, bytes: &[u8], presentation: i64, duration: i64, ticks: u32) -> Packet {
    Packet::try_new(
        stream,
        bytes.to_vec(),
        time(presentation, duration),
        scale(ticks),
        true,
    )
    .expect("packet")
}

fn video(id: u32) -> Stream {
    let size = FrameSize::try_new(16, 16).expect("size");
    Stream::new(
        TrackId::new(id),
        StreamParams::video(StreamCodec::Known(CodecId::H264), size, vec![1, 2]),
        scale(30_000),
        None,
        Vec::new(),
        Trim::default(),
    )
}

fn random(bytes: &[u8]) -> Input<Cursor<Vec<u8>>> {
    let length = u64::try_from(bytes.len()).expect("len");
    Input::random(Cursor::new(bytes.to_vec()), length, Limits::standard())
}

fn open_list(
    bytes: Vec<u8>,
    streams: Vec<Stream>,
    plans: Vec<PacketPlan>,
    index: Vec<IndexEntry>,
    gaps: Vec<Gap>,
) -> ListDemux<Cursor<Vec<u8>>> {
    let length = u64::try_from(bytes.len()).expect("len");
    ListDemux::open(ListRequest {
        input: Input::random(Cursor::new(bytes), length, Limits::standard()),
        family: ContainerId::Isobmff,
        header_bytes: 4,
        verdict: HeaderVerdict::Match,
        streams,
        packets: plans,
        index,
        gaps,
    })
    .expect("open")
}

#[test]
fn open_failures_publish_no_streams_and_read_failures_keep_them() {
    let short = ListDemux::open(ListRequest {
        input: random(b"ab"),
        family: ContainerId::Isobmff,
        header_bytes: 4,
        verdict: HeaderVerdict::Match,
        streams: vec![video(0)],
        packets: Vec::new(),
        index: Vec::new(),
        gaps: Vec::new(),
    });
    let Err(failure) = short else {
        panic!("short file opened");
    };
    assert_eq!(
        failure.error,
        OpenError::Truncated {
            expected: 4,
            got: 2
        }
    );
    assert!(!failure.error.streams_published());
    assert_eq!(failure.input.consumed(), 0);

    let unrecognized = ListDemux::open(ListRequest {
        input: random(b"STL!"),
        family: ContainerId::Still,
        header_bytes: 4,
        verdict: HeaderVerdict::Unrecognized,
        streams: vec![video(0)],
        packets: Vec::new(),
        index: Vec::new(),
        gaps: Vec::new(),
    });
    let Err(failure) = unrecognized else {
        panic!("unrecognized file opened");
    };
    assert!(matches!(failure.error, OpenError::Unrecognized { .. }));
    assert_eq!(failure.input.consumed(), 0);

    let brand = ListDemux::open(ListRequest {
        input: random(b"MOV\xff"),
        family: ContainerId::Isobmff,
        header_bytes: 4,
        verdict: HeaderVerdict::UnsupportedBrand,
        streams: vec![video(0)],
        packets: Vec::new(),
        index: Vec::new(),
        gaps: Vec::new(),
    });
    let Err(failure) = brand else {
        panic!("unsupported brand opened");
    };
    assert!(matches!(failure.error, OpenError::UnsupportedBrand { .. }));

    let mut bytes = b"HDR!".to_vec();
    bytes.extend_from_slice(b"ab");
    let mut demux = ListDemux::open(ListRequest {
        input: Input::random(Cursor::new(bytes), 100, Limits::standard()),
        family: ContainerId::Isobmff,
        header_bytes: 4,
        verdict: HeaderVerdict::Match,
        streams: vec![video(0)],
        packets: vec![PacketPlan {
            stream: 0,
            timestamp: time(0, 1),
            timescale: scale(30_000),
            keyframe: true,
            size: 4,
        }],
        index: Vec::new(),
        gaps: Vec::new(),
    })
    .expect("header fits");
    let error = demux.read_packet().expect_err("truncated unit");
    assert!(matches!(
        error,
        ReadError::Truncated {
            expected: 4,
            got: 2
        }
    ));
    assert!(!error.packet_returned());
    assert!(error.streams_remain());
    assert_eq!(demux.streams().len(), 1);
    assert_eq!(demux.read_packet().expect_err("sticky"), error);
}

#[test]
fn a_forward_short_packet_is_not_a_truncated_open() {
    let mut bytes = b"HDR!".to_vec();
    bytes.extend_from_slice(b"ab");
    let mut demux = ListDemux::open(ListRequest {
        input: Input::forward(Cursor::new(bytes), Limits::standard()),
        family: ContainerId::Isobmff,
        header_bytes: 4,
        verdict: HeaderVerdict::Match,
        streams: vec![video(0)],
        packets: vec![PacketPlan {
            stream: 0,
            timestamp: time(0, 1),
            timescale: scale(30_000),
            keyframe: true,
            size: 4,
        }],
        index: Vec::new(),
        gaps: Vec::new(),
    })
    .expect("header");
    assert!(matches!(
        demux.read_packet().expect_err("short"),
        ReadError::ShortRead {
            expected: 4,
            got: 2
        }
    ));
}

#[test]
fn packets_stay_in_decode_order_and_in_the_file_timescale() {
    let mut bytes = b"HDR!".to_vec();
    bytes.extend_from_slice(b"aaaa");
    bytes.extend_from_slice(b"bbbb");
    let edit = Edit::try_new(Some(100), 30_000, 1, 1).expect("edit");
    let mut stream = video(0);
    stream = Stream::new(
        stream.id(),
        stream.params().clone(),
        stream.timescale(),
        None,
        vec![edit],
        Trim::new(Some(1_024), None),
    );
    let gap = Gap::try_new(TrackId::new(0), 0, 10).expect("gap");
    let mut demux = open_list(
        bytes,
        vec![stream],
        vec![
            PacketPlan {
                stream: 0,
                timestamp: time(20, 10),
                timescale: scale(30_000),
                keyframe: true,
                size: 4,
            },
            PacketPlan {
                stream: 0,
                timestamp: time(0, 10),
                timescale: scale(30_000),
                keyframe: false,
                size: 4,
            },
        ],
        Vec::new(),
        vec![gap],
    );
    let first = demux.read_packet().expect("read").expect("packet");
    let second = demux.read_packet().expect("read").expect("packet");
    assert_eq!(first.bytes(), b"aaaa");
    assert_eq!(second.bytes(), b"bbbb");
    assert_eq!(first.timestamp().presentation(), Some(20));
    assert_eq!(second.timestamp().presentation(), Some(0));
    assert_eq!(first.timescale().ticks_per_second(), 30_000);
    assert_ne!(
        first.timestamp().presentation(),
        Some(20_i64 * 1_000_000 / 30_000)
    );
    assert_eq!(demux.streams()[0].edits()[0].media_time(), Some(100));
    assert_eq!(demux.streams()[0].trim().encoder_delay(), Some(1_024));
    assert_eq!(demux.gaps()[0].duration(), 10);
    assert!(demux.read_packet().expect("end").is_none());
}

#[test]
fn an_unknown_codec_is_kept_on_the_stream() {
    let unknown = UnknownCodec::new(b"zz99".to_vec()).expect("tag");
    let size = FrameSize::try_new(8, 8).expect("size");
    let stream = Stream::new(
        TrackId::new(0),
        StreamParams::video(StreamCodec::Unknown(unknown), size, Vec::new()),
        scale(1),
        None,
        Vec::new(),
        Trim::default(),
    );
    let mut demux = open_list(
        b"HDR!".to_vec(),
        vec![stream],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    assert!(demux.streams()[0].params().codec().is_unknown());
    assert!(demux.read_packet().expect("empty").is_none());
}

#[test]
fn an_index_seek_does_not_read_the_payload_it_skips() {
    let mut bytes = b"HDR!".to_vec();
    bytes.extend_from_slice(b"aaaa");
    bytes.extend_from_slice(b"bbbb");
    bytes.extend_from_slice(b"cccc");
    let index = vec![
        IndexEntry {
            track: 0,
            time: 0,
            ordinal: 0,
            keyframe: true,
            byte_position: 4,
        },
        IndexEntry {
            track: 0,
            time: 10,
            ordinal: 1,
            keyframe: false,
            byte_position: 8,
        },
        IndexEntry {
            track: 0,
            time: 20,
            ordinal: 2,
            keyframe: true,
            byte_position: 12,
        },
    ];
    let mut demux = open_list(
        bytes,
        vec![video(0)],
        vec![
            PacketPlan {
                stream: 0,
                timestamp: time(0, 10),
                timescale: scale(30_000),
                keyframe: true,
                size: 4,
            },
            PacketPlan {
                stream: 0,
                timestamp: time(10, 10),
                timescale: scale(30_000),
                keyframe: false,
                size: 4,
            },
            PacketPlan {
                stream: 0,
                timestamp: time(20, 10),
                timescale: scale(30_000),
                keyframe: true,
                size: 4,
            },
        ],
        index,
        Vec::new(),
    );
    let landing = demux
        .seek(SeekTo {
            track: 0,
            time: 20,
            end: Some(40),
        })
        .expect("seek");
    assert!(landing.index_used);
    assert_eq!(landing.payload_scans, 0);
    match landing.landing {
        Landing::Packet {
            ordinal,
            keyframe,
            preroll,
            ..
        } => {
            assert_eq!(ordinal, 2);
            assert!(keyframe);
            assert_eq!(preroll, 0);
        }
        Landing::End => panic!("landed at the end"),
    }
    let packet = demux.read_packet().expect("read").expect("packet");
    assert_eq!(packet.bytes(), b"cccc");
    assert_eq!(demux.consumed(), 8);
}

#[test]
fn seek_between_keyframes_suppresses_packets_from_before_the_landing() {
    let mut bytes = b"HDR!".to_vec();
    bytes.extend_from_slice(b"aaaa");
    bytes.extend_from_slice(b"bbbb");
    bytes.extend_from_slice(b"cccc");
    let mut demux = open_list(
        bytes,
        vec![video(0)],
        vec![
            PacketPlan {
                stream: 0,
                timestamp: time(0, 10),
                timescale: scale(30_000),
                keyframe: true,
                size: 4,
            },
            PacketPlan {
                stream: 0,
                timestamp: time(10, 10),
                timescale: scale(30_000),
                keyframe: false,
                size: 4,
            },
            PacketPlan {
                stream: 0,
                timestamp: time(20, 10),
                timescale: scale(30_000),
                keyframe: true,
                size: 4,
            },
        ],
        Vec::new(),
        Vec::new(),
    );
    let first = demux.read_packet().expect("read").expect("first");
    assert_eq!(first.bytes(), b"aaaa");
    let landing = demux
        .seek(SeekTo {
            track: 0,
            time: 15,
            end: Some(40),
        })
        .expect("seek");
    match landing.landing {
        Landing::Packet {
            ordinal,
            preroll,
            keyframe,
            ..
        } => {
            assert_eq!(ordinal, 0);
            assert!(keyframe);
            assert_eq!(preroll, 2);
        }
        Landing::End => panic!("end"),
    }
    let again = demux.read_packet().expect("read").expect("keyframe");
    assert_eq!(again.bytes(), b"aaaa");
    let landing = demux
        .seek(SeekTo {
            track: 0,
            time: 50,
            end: Some(40),
        })
        .expect("past");
    assert!(matches!(landing.landing, Landing::End));
    assert!(demux.read_packet().expect("end").is_none());
}

#[test]
fn a_failed_seek_does_not_move_the_next_packet() {
    let mut bytes = b"HDR!".to_vec();
    bytes.extend_from_slice(b"aaaa");
    let mut demux = open_list(
        bytes.clone(),
        vec![video(0)],
        vec![PacketPlan {
            stream: 0,
            timestamp: time(0, 1),
            timescale: scale(30_000),
            keyframe: true,
            size: 4,
        }],
        Vec::new(),
        Vec::new(),
    );
    let error = demux
        .seek(SeekTo {
            track: 9,
            time: 0,
            end: None,
        })
        .expect_err("track");
    assert_eq!(error, ReadError::Seek(SeekError::UnknownTrack));
    assert_eq!(
        demux.read_packet().expect("read").expect("packet").bytes(),
        b"aaaa"
    );

    let mut forward = ListDemux::open(ListRequest {
        input: Input::forward(Cursor::new(bytes), Limits::standard()),
        family: ContainerId::Isobmff,
        header_bytes: 4,
        verdict: HeaderVerdict::Match,
        streams: vec![video(0)],
        packets: vec![PacketPlan {
            stream: 0,
            timestamp: time(0, 1),
            timescale: scale(30_000),
            keyframe: true,
            size: 4,
        }],
        index: Vec::new(),
        gaps: Vec::new(),
    })
    .expect("open");
    assert_eq!(
        forward
            .seek(SeekTo {
                track: 0,
                time: 0,
                end: None
            })
            .expect_err("forward"),
        ReadError::Seek(SeekError::ForwardOnly)
    );
    assert_eq!(forward.position(), 4);
    assert_eq!(
        forward
            .read_packet()
            .expect("read")
            .expect("packet")
            .bytes(),
        b"aaaa"
    );
}

#[test]
fn cancel_stops_the_next_packet_and_keeps_the_one_already_returned() {
    let mut bytes = b"HDR!".to_vec();
    bytes.extend_from_slice(b"aaaa");
    bytes.extend_from_slice(b"bbbb");
    let mut demux = open_list(
        bytes,
        vec![video(0)],
        vec![
            PacketPlan {
                stream: 0,
                timestamp: time(0, 1),
                timescale: scale(30_000),
                keyframe: true,
                size: 4,
            },
            PacketPlan {
                stream: 0,
                timestamp: time(1, 1),
                timescale: scale(30_000),
                keyframe: true,
                size: 4,
            },
        ],
        Vec::new(),
        Vec::new(),
    );
    let first = demux.read_packet().expect("read").expect("packet");
    demux.cancel();
    assert_eq!(
        demux.read_packet().expect_err("cancel"),
        ReadError::Cancelled
    );
    assert_eq!(first.bytes(), b"aaaa");
    assert_eq!(demux.streams().len(), 1);
}

#[test]
fn a_hostile_packet_size_is_a_read_error_and_allocates_nothing_of_that_size() {
    let mut demux = open_list(
        b"HDR!".to_vec(),
        vec![video(0)],
        vec![PacketPlan {
            stream: 0,
            timestamp: time(0, 1),
            timescale: scale(30_000),
            keyframe: true,
            size: u64::MAX,
        }],
        Vec::new(),
        Vec::new(),
    );
    let error = demux.read_packet().expect_err("hostile");
    assert!(matches!(
        error,
        ReadError::HostileSize {
            declared: u64::MAX,
            ..
        }
    ));
    assert_eq!(demux.consumed(), 4);
    assert_eq!(demux.streams().len(), 1);
}

#[test]
fn the_traits_are_object_safe() {
    let demux = open_list(
        b"HDR!".to_vec(),
        vec![video(0)],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let mut demux: Box<dyn Demux> = Box::new(demux);
    assert_eq!(demux.family(), ContainerId::Isobmff);
    assert!(demux.read_packet().expect("end").is_none());

    let mux: Box<dyn Mux> = Box::new(MuxDraft::new(
        ContainerId::Isobmff,
        MuxMode::Progressive,
        IndexPlace::End,
        Limits::standard(),
    ));
    assert_eq!(mux.family(), ContainerId::Isobmff);
    assert_eq!(mux.status(), MuxStatus::Building);
}

#[test]
fn a_failed_mux_does_not_publish_a_duration() {
    let mut mux = MuxDraft::new(
        ContainerId::Isobmff,
        MuxMode::Progressive,
        IndexPlace::End,
        Limits::standard(),
    );
    let unknown = UnknownCodec::new(b"zz99".to_vec()).expect("tag");
    let size = FrameSize::try_new(16, 16).expect("size");
    let error = mux
        .add_stream(NewStream {
            params: StreamParams::video(StreamCodec::Unknown(unknown), size, Vec::new()),
            timescale: scale(30_000),
            language: None,
        })
        .expect_err("unknown");
    assert!(matches!(error, MuxError::UnknownCodec(_)));
    let failed = mux.finish(&[]).expect_err("finish");
    assert!(!failed.is_complete());
    assert!(failed.durations().is_none());
    assert_eq!(mux.status(), MuxStatus::Failed);
}

#[test]
fn finish_refuses_a_duration_that_does_not_match_the_payload() {
    let mut mux = MuxDraft::new(
        ContainerId::Adts,
        MuxMode::Progressive,
        IndexPlace::Absent,
        Limits::standard(),
    );
    let id = mux
        .add_stream(NewStream {
            params: StreamParams::try_audio(StreamCodec::Known(CodecId::Aac), 48_000, 1, vec![9])
                .expect("audio"),
            timescale: scale(48_000),
            language: None,
        })
        .expect("stream");
    let sample = packet(id.get(), &[1, 2, 3], 0, 1_024, 48_000);
    mux.write_packet(&sample).expect("write");
    assert_eq!(mux.packets().expect("building")[0].bytes(), &[1, 2, 3]);
    let failed = mux
        .finish(&[TrackDuration {
            track: id,
            timescale: scale(48_000),
            ticks: 2_000,
        }])
        .expect_err("mismatch");
    assert!(!failed.is_complete());
    assert!(failed.durations().is_none());
    assert!(matches!(failed.reason(), MuxError::DurationMismatch { .. }));
    assert_eq!(failed.payload_bytes(), 3);
}

#[test]
fn a_positive_duration_with_no_packets_is_not_complete() {
    let mut mux = MuxDraft::new(
        ContainerId::Riff,
        MuxMode::Progressive,
        IndexPlace::Start,
        Limits::standard(),
    );
    let id = mux
        .add_stream(NewStream {
            params: StreamParams::try_audio(
                StreamCodec::Known(CodecId::Pcm),
                48_000,
                1,
                Vec::new(),
            )
            .expect("audio"),
            timescale: scale(48_000),
            language: None,
        })
        .expect("stream");
    let failed = mux
        .finish(&[TrackDuration {
            track: id,
            timescale: scale(48_000),
            ticks: 48_000,
        }])
        .expect_err("short");
    assert!(matches!(failed.reason(), MuxError::ShortPayload { .. }));
    assert!(failed.durations().is_none());
}

#[test]
fn interleave_keeps_file_ticks_and_a_second_finish_is_not_a_new_file() {
    let mut mux = MuxDraft::new(
        ContainerId::Isobmff,
        MuxMode::Progressive,
        IndexPlace::End,
        Limits::standard(),
    );
    let video_id = mux
        .add_stream(NewStream {
            params: StreamParams::video(
                StreamCodec::Known(CodecId::H264),
                FrameSize::try_new(16, 16).expect("size"),
                vec![1],
            ),
            timescale: scale(90_000),
            language: None,
        })
        .expect("video");
    let audio_id = mux
        .add_stream(NewStream {
            params: StreamParams::try_audio(StreamCodec::Known(CodecId::Aac), 48_000, 2, vec![2])
                .expect("audio"),
            timescale: scale(48_000),
            language: None,
        })
        .expect("audio");
    mux.write_packet(&packet(video_id.get(), b"v0", 0, 4_500, 90_000))
        .expect("v0");
    mux.write_packet(&packet(video_id.get(), b"v1", 45_000, 4_500, 90_000))
        .expect("v1");
    mux.write_packet(&packet(audio_id.get(), b"a0", 0, 24_000, 48_000))
        .expect("a0");
    mux.write_packet(&packet(audio_id.get(), b"a1", 48_000, 24_000, 48_000))
        .expect("a1");
    mux.interleave().expect("order");
    let finished = mux
        .finish(&[
            TrackDuration {
                track: video_id,
                timescale: scale(90_000),
                ticks: 9_000,
            },
            TrackDuration {
                track: audio_id,
                timescale: scale(48_000),
                ticks: 48_000,
            },
        ])
        .expect("finish");
    assert!(finished.is_complete());
    assert!(finished.interleaved());
    assert_eq!(finished.mode(), MuxMode::Progressive);
    assert_eq!(finished.index(), IndexPlace::End);
    assert_eq!(finished.durations().len(), 2);
    let presentations: Vec<Option<i64>> = finished
        .packets()
        .iter()
        .map(|item| item.timestamp().presentation())
        .collect();
    assert_eq!(
        presentations,
        vec![Some(0), Some(0), Some(45_000), Some(48_000)]
    );
    assert_eq!(
        finished.packets()[2].timestamp().presentation(),
        Some(45_000)
    );
    assert_ne!(
        finished.packets()[2].timestamp().presentation(),
        Some(500_000)
    );
    assert_eq!(finished.packets()[0].bytes(), b"v0");
    assert_eq!(finished.packets()[1].bytes(), b"a0");

    let again = mux.finish(&[]).expect_err("second");
    assert!(!again.is_complete());
    assert!(again.durations().is_none());
    assert_eq!(finished.durations()[0].ticks, 9_000);
}

#[test]
fn a_missing_timestamp_is_not_invented_and_fragmented_is_its_own_mode() {
    let mut mux = MuxDraft::new(
        ContainerId::Matroska,
        MuxMode::Fragmented,
        IndexPlace::Start,
        Limits::standard(),
    );
    let id = mux
        .add_stream(NewStream {
            params: StreamParams::video(
                StreamCodec::Known(CodecId::Vp9),
                FrameSize::try_new(16, 16).expect("size"),
                Vec::new(),
            ),
            timescale: scale(1_000),
            language: None,
        })
        .expect("stream");
    let untimed = Packet::try_new(id.get(), vec![7], Timestamp::missing(), scale(1_000), true)
        .expect("packet");
    mux.write_packet(&untimed).expect("write");
    assert_eq!(
        mux.interleave().expect_err("missing"),
        MuxError::MissingTimestamp
    );
    assert!(
        mux.packets().expect("still building")[0]
            .timestamp()
            .is_missing()
    );
    assert_eq!(mux.status(), MuxStatus::Building);

    let mut fragmented = MuxDraft::new(
        ContainerId::Isobmff,
        MuxMode::Fragmented,
        IndexPlace::Start,
        Limits::standard(),
    );
    let mut progressive = MuxDraft::new(
        ContainerId::Isobmff,
        MuxMode::Progressive,
        IndexPlace::End,
        Limits::standard(),
    );
    assert_ne!(
        finish_empty(&mut fragmented).mode(),
        finish_empty(&mut progressive).mode()
    );
}

fn finish_empty(mux: &mut MuxDraft) -> FinishedMux {
    mux.finish(&[]).expect("empty")
}

#[test]
fn grouped_tracks_are_not_interleaved_until_reordered() {
    let mut mux = MuxDraft::new(
        ContainerId::Isobmff,
        MuxMode::Progressive,
        IndexPlace::End,
        Limits::standard(),
    );
    let video_id = mux
        .add_stream(NewStream {
            params: StreamParams::video(
                StreamCodec::Known(CodecId::H264),
                FrameSize::try_new(16, 16).expect("size"),
                Vec::new(),
            ),
            timescale: scale(30),
            language: None,
        })
        .expect("video");
    let audio_id = mux
        .add_stream(NewStream {
            params: StreamParams::try_audio(
                StreamCodec::Known(CodecId::Aac),
                48_000,
                1,
                Vec::new(),
            )
            .expect("audio"),
            timescale: scale(30),
            language: None,
        })
        .expect("audio");
    mux.write_packet(&packet(video_id.get(), b"v0", 0, 1, 30))
        .expect("v0");
    mux.write_packet(&packet(video_id.get(), b"v1", 1, 1, 30))
        .expect("v1");
    mux.write_packet(&packet(audio_id.get(), b"a0", 0, 1, 30))
        .expect("a0");
    mux.write_packet(&packet(audio_id.get(), b"a1", 1, 1, 30))
        .expect("a1");
    let grouped = mux
        .finish(&[
            TrackDuration {
                track: video_id,
                timescale: scale(30),
                ticks: 2,
            },
            TrackDuration {
                track: audio_id,
                timescale: scale(30),
                ticks: 2,
            },
        ])
        .expect("grouped");
    assert!(!grouped.interleaved());
}

#[test]
fn a_round_trip_keeps_bytes_and_file_ticks() {
    let mut mux = MuxDraft::new(
        ContainerId::Isobmff,
        MuxMode::Progressive,
        IndexPlace::End,
        Limits::standard(),
    );
    let id = mux
        .add_stream(NewStream {
            params: StreamParams::video(
                StreamCodec::Known(CodecId::H264),
                FrameSize::try_new(16, 16).expect("size"),
                vec![4],
            ),
            timescale: scale(90_000),
            language: None,
        })
        .expect("stream");
    mux.write_packet(&packet(id.get(), b"unit", 90_000, 3_000, 90_000))
        .expect("write");
    let finished = mux
        .finish(&[TrackDuration {
            track: id,
            timescale: scale(90_000),
            ticks: 3_000,
        }])
        .expect("finish");
    let mut bytes = b"HDR!".to_vec();
    bytes.extend_from_slice(finished.packets()[0].bytes());
    let stream = Stream::new(
        TrackId::new(id.get()),
        StreamParams::video(
            StreamCodec::Known(CodecId::H264),
            FrameSize::try_new(16, 16).expect("size"),
            vec![4],
        ),
        scale(90_000),
        None,
        Vec::new(),
        Trim::default(),
    );
    let mut demux = open_list(
        bytes,
        vec![stream],
        vec![PacketPlan {
            stream: id.get(),
            timestamp: finished.packets()[0].timestamp(),
            timescale: finished.packets()[0].timescale(),
            keyframe: finished.packets()[0].keyframe(),
            size: u64::try_from(finished.packets()[0].bytes().len()).expect("size"),
        }],
        Vec::new(),
        Vec::new(),
    );
    let read = demux.read_packet().expect("read").expect("packet");
    assert_eq!(read.bytes(), b"unit");
    assert_eq!(read.timestamp().presentation(), Some(90_000));
    assert_eq!(read.timestamp().duration(), Some(3_000));
    assert_eq!(read.timescale().ticks_per_second(), 90_000);
    assert!(read.keyframe());
}

#[test]
fn demux_and_input_are_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Input<Cursor<Vec<u8>>>>();
    assert_send::<ListDemux<Cursor<Vec<u8>>>>();
    assert_send::<MuxDraft>();
}

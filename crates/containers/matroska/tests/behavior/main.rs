//! What the demuxer does with hand-built files: timestamps, lacing, block
//! groups, live (unknown-size) layouts, header stripping, timestamp
//! scales, metadata after the Clusters, seeking with and without Cues, and
//! malformed input.
//!
//! `clippy.toml` allows `expect` and `panic` in `#[cfg(test)]` modules.
//! This file is the harness, so the same exception is named here.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

mod build;
mod mux;

use std::io::Cursor;

use build::{
    Track, block_head, bytes, cluster, cues, ebml_header, ebml_laced, element, file, fixed, group,
    info, live_file, simple, tracks, uint, unknown, unknown_cluster, with_seek_head, xiph,
};
use cutlass_codec::{CodecId, Packet, StreamCodec};
use cutlass_container::{Demux, Landing, OpenError, ReadError, SeekError, SeekTo};
use cutlass_matroska::{MatroskaDemux, ids};

type Opened = MatroskaDemux<Cursor<Vec<u8>>>;

fn open(bytes: Vec<u8>) -> Result<Opened, OpenError> {
    let length = u64::try_from(bytes.len()).expect("length");
    MatroskaDemux::open(Cursor::new(bytes), length)
}

fn all(demux: &mut Opened) -> Vec<Packet> {
    let mut out = Vec::new();
    while let Some(packet) = demux.read_packet().expect("packet") {
        out.push(packet);
    }
    out
}

/// (track, pts, duration, bytes, keyframe) of each packet.
type Row = (u32, Option<i64>, Option<i64>, Vec<u8>, bool);

fn rows(packets: &[Packet]) -> Vec<Row> {
    packets
        .iter()
        .map(|packet| {
            let time = packet.timestamp();
            (
                packet.stream(),
                time.presentation(),
                time.duration(),
                packet.bytes().to_vec(),
                packet.keyframe(),
            )
        })
        .collect()
}

fn h264() -> Track {
    Track::video(1, "V_MPEG4/ISO/AVC").with_private(&[1, 0x64, 0, 0x1F, 0xFF, 0xE0, 0])
}

fn mp3() -> Track {
    Track::audio(2, "A_MPEG/L3", 48_000.0, 2)
}

#[test]
fn a_plain_file_gives_every_frame_in_file_order_with_its_track_and_time() {
    let bytes = file(
        "matroska",
        &[
            info(1_000_000, Some(120.0)),
            tracks(&[h264(), mp3()]),
            cluster(
                0,
                &[
                    simple(1, 0, true, b"v0"),
                    simple(2, 0, true, b"a0"),
                    simple(1, 40, false, b"v1"),
                ],
            ),
            cluster(80, &[simple(1, 0, true, b"v2"), simple(2, -5, true, b"a1")]),
        ],
    );
    let mut demux = open(bytes).expect("open");
    assert_eq!(demux.family(), cutlass_codec::ContainerId::Matroska);
    let streams = demux.streams();
    assert_eq!(streams.len(), 2);
    assert_eq!(streams[0].id().get(), 1);
    assert_eq!(
        streams[0].params().codec(),
        &StreamCodec::Known(CodecId::H264)
    );
    assert_eq!(
        streams[0].params().extradata(),
        &[1, 0x64, 0, 0x1F, 0xFF, 0xE0, 0]
    );
    let size = streams[0].params().size().expect("size");
    assert_eq!((size.width(), size.height()), (64, 48));
    assert_eq!(streams[0].timescale().ticks_per_second(), 1000);
    assert_eq!(
        streams[1].params().codec(),
        &StreamCodec::Known(CodecId::Mp3)
    );
    assert_eq!(streams[1].params().sample_rate(), Some(48_000));
    assert_eq!(streams[1].params().channels(), Some(2));
    assert!(streams[1].edits().is_empty());
    assert_eq!(demux.duration_ticks(), Some(120));
    let got = rows(&all(&mut demux));
    assert_eq!(
        got,
        vec![
            (1, Some(0), None, b"v0".to_vec(), true),
            (2, Some(0), None, b"a0".to_vec(), true),
            (1, Some(40), None, b"v1".to_vec(), false),
            (1, Some(80), None, b"v2".to_vec(), true),
            (2, Some(75), None, b"a1".to_vec(), true),
        ]
    );
    // The end stays the end.
    assert!(demux.read_packet().expect("end").is_none());
    assert!(demux.consumed() > 0);
}

#[test]
fn laced_frames_take_default_duration_steps_rounded_to_the_tick() {
    // A third of a frame period does not land on a millisecond: frames
    // start at the rounded multiples, and durations are the differences.
    let audio = Track::audio(2, "A_MPEG/L3", 48_000.0, 2).with_default_duration(33_333_333);
    let frames: [&[u8]; 3] = [b"aaaa", b"bbbb", b"cccc"];
    let bytes = file(
        "matroska",
        &[
            info(1_000_000, None),
            tracks(&[audio]),
            cluster(
                1000,
                &[
                    xiph(2, 0, &frames),
                    ebml_laced(2, 100, &[b"x", b"yyyyyyy", b"zz"]),
                    fixed(2, 200, &frames),
                    simple(2, 300, true, b"one"),
                ],
            ),
        ],
    );
    let mut demux = open(bytes).expect("open");
    let got: Vec<(Option<i64>, Option<i64>, Vec<u8>)> = rows(&all(&mut demux))
        .into_iter()
        .map(|row| (row.1, row.2, row.3))
        .collect();
    assert_eq!(
        got,
        vec![
            (Some(1000), Some(33), b"aaaa".to_vec()),
            (Some(1033), Some(34), b"bbbb".to_vec()),
            (Some(1067), Some(33), b"cccc".to_vec()),
            (Some(1100), Some(33), b"x".to_vec()),
            (Some(1133), Some(34), b"yyyyyyy".to_vec()),
            (Some(1167), Some(33), b"zz".to_vec()),
            (Some(1200), Some(33), b"aaaa".to_vec()),
            (Some(1233), Some(34), b"bbbb".to_vec()),
            (Some(1267), Some(33), b"cccc".to_vec()),
            (Some(1300), Some(33), b"one".to_vec()),
        ]
    );
}

#[test]
fn laced_frames_without_any_duration_have_no_time_after_the_first() {
    let bytes = file(
        "matroska",
        &[
            info(1_000_000, None),
            tracks(&[mp3()]),
            cluster(0, &[xiph(2, 7, &[b"a", b"b"])]),
        ],
    );
    let got = rows(&all(&mut open(bytes).expect("open")));
    assert_eq!(got[0].1, Some(7));
    assert_eq!(got[1].1, None);
    assert_eq!(got[0].2, None);
}

#[test]
fn block_groups_give_block_duration_and_references_clear_the_keyframe() {
    let bytes = file(
        "matroska",
        &[
            info(1_000_000, None),
            tracks(&[h264()]),
            cluster(
                0,
                &[
                    group(1, 0, b"key", Some(40), None),
                    group(1, 40, b"p", Some(40), Some(-40)),
                    group(1, 80, b"last", Some(25), Some(-40)),
                ],
            ),
        ],
    );
    let got = rows(&all(&mut open(bytes).expect("open")));
    assert_eq!(
        got,
        vec![
            (1, Some(0), Some(40), b"key".to_vec(), true),
            (1, Some(40), Some(40), b"p".to_vec(), false),
            (1, Some(80), Some(25), b"last".to_vec(), false),
        ]
    );
}

#[test]
fn a_block_duration_over_a_lace_is_shared_out() {
    let mut block = block_head(2, 0, 0x02);
    block.push(2);
    block.extend(cutlass_matroska::block::xiph_sizes(&[1, 1]));
    block.extend_from_slice(b"abc");
    let mut body = element(ids::BLOCK, &block);
    body.extend(uint(ids::BLOCK_DURATION, 10));
    let bytes = file(
        "matroska",
        &[
            info(1_000_000, None),
            tracks(&[mp3()]),
            cluster(0, &[element(ids::BLOCK_GROUP, &body)]),
        ],
    );
    let got = rows(&all(&mut open(bytes).expect("open")));
    let times: Vec<(Option<i64>, Option<i64>)> = got.iter().map(|row| (row.1, row.2)).collect();
    assert_eq!(
        times,
        vec![(Some(0), Some(3)), (Some(3), Some(3)), (Some(6), Some(4))]
    );
}

fn live_bytes() -> Vec<u8> {
    live_file(
        "webm",
        &[
            info(1_000_000, None),
            tracks(&[Track::video(1, "V_VP8"), mp3()]),
            unknown_cluster(0, &[simple(1, 0, true, b"v0"), simple(2, 0, true, b"a0")]),
            unknown_cluster(
                1000,
                &[simple(1, 0, true, b"v1"), simple(2, 10, true, b"a1")],
            ),
            // Cues end the last unknown-size Cluster.
            cues(&[(0, 1, 0)]),
        ],
    )
}

#[test]
fn unknown_size_segments_and_clusters_read_the_same_forward_and_random() {
    let expected = vec![
        (1, Some(0), None, b"v0".to_vec(), true),
        (2, Some(0), None, b"a0".to_vec(), true),
        (1, Some(1000), None, b"v1".to_vec(), true),
        (2, Some(1010), None, b"a1".to_vec(), true),
    ];
    let mut random = open(live_bytes()).expect("random");
    assert_eq!(
        random.ebml_header().doc_type,
        cutlass_matroska::meta::DocType::Webm
    );
    assert_eq!(rows(&all(&mut random)), expected);
    let mut forward = MatroskaDemux::open_forward(Cursor::new(live_bytes())).expect("forward");
    let mut got = Vec::new();
    while let Some(packet) = forward.read_packet().expect("packet") {
        got.push(packet);
    }
    assert_eq!(rows(&got), expected);
    let refused = forward.seek(SeekTo {
        track: 1,
        time: 0,
        end: None,
    });
    assert_eq!(refused, Err(ReadError::Seek(SeekError::ForwardOnly)));
}

#[test]
fn an_unknown_size_cluster_ends_at_the_next_cluster_without_cues() {
    let bytes = live_file(
        "matroska",
        &[
            info(1_000_000, None),
            tracks(&[mp3()]),
            unknown_cluster(0, &[simple(2, 0, true, b"a")]),
            cluster(500, &[simple(2, 0, true, b"b")]),
            unknown_cluster(900, &[simple(2, 1, true, b"c")]),
        ],
    );
    let mut demux = open(bytes).expect("open");
    let times: Vec<Option<i64>> = rows(&all(&mut demux)).iter().map(|row| row.1).collect();
    assert_eq!(times, vec![Some(0), Some(500), Some(901)]);
    let landing = demux
        .seek(SeekTo {
            track: 2,
            time: 600,
            end: None,
        })
        .expect("seek");
    assert!(matches!(landing.landing, Landing::Packet { time: 500, .. }));
}

fn encodings(children: &[u8]) -> Vec<u8> {
    let encoding = element(ids::CONTENT_ENCODING, children);
    element(ids::CONTENT_ENCODINGS, &encoding)
}

#[test]
fn header_stripping_puts_the_bytes_back_on_every_frame() {
    let mut compression = uint(ids::CONTENT_COMP_ALGO, 3);
    compression.extend(bytes(ids::CONTENT_COMP_SETTINGS, &[0xFF, 0xFB]));
    let stripped = encodings(&element(ids::CONTENT_COMPRESSION, &compression));
    let track = mp3()
        .with_default_duration(24_000_000)
        .with_extra(&stripped);
    let file_bytes = file(
        "matroska",
        &[
            info(1_000_000, None),
            tracks(&[track]),
            cluster(
                0,
                &[simple(2, 0, true, b"\x90\x00"), xiph(2, 24, &[b"a", b"b"])],
            ),
        ],
    );
    let mut demux = open(file_bytes).expect("open");
    assert_eq!(
        demux.streams()[0].params().codec(),
        &StreamCodec::Known(CodecId::Mp3)
    );
    let payloads: Vec<Vec<u8>> = rows(&all(&mut demux))
        .into_iter()
        .map(|row| row.3)
        .collect();
    assert_eq!(
        payloads,
        vec![
            vec![0xFF, 0xFB, 0x90, 0x00],
            vec![0xFF, 0xFB, b'a'],
            vec![0xFF, 0xFB, b'b'],
        ]
    );
}

#[test]
fn encrypted_and_zlib_tracks_open_as_unknown_codecs_with_their_bytes_as_stored() {
    let encrypted = encodings(&{
        let mut children = uint(ids::CONTENT_ENCODING_TYPE, 1);
        children.extend(element(ids::CONTENT_ENCRYPTION, &[]));
        children
    });
    let zlib = encodings(&element(
        ids::CONTENT_COMPRESSION,
        &uint(ids::CONTENT_COMP_ALGO, 0),
    ));
    for extra in [encrypted, zlib] {
        let bytes = file(
            "matroska",
            &[
                info(1_000_000, None),
                tracks(&[mp3().with_extra(&extra)]),
                cluster(0, &[simple(2, 0, true, b"sealed")]),
            ],
        );
        let mut demux = open(bytes).expect("open");
        assert!(matches!(
            demux.streams()[0].params().codec(),
            StreamCodec::Unknown(_)
        ));
        assert_eq!(rows(&all(&mut demux))[0].3, b"sealed".to_vec());
    }
}

#[test]
fn timestamp_scales_keep_the_file_clock() {
    // 100 microseconds per tick: ten thousand ticks a second, unchanged.
    let bytes = file(
        "matroska",
        &[
            info(100_000, Some(5.0)),
            tracks(&[mp3()]),
            cluster(12_345, &[simple(2, 3, true, b"a")]),
        ],
    );
    let mut demux = open(bytes).expect("open");
    assert_eq!(demux.streams()[0].timescale().ticks_per_second(), 10_000);
    assert_eq!(rows(&all(&mut demux))[0].1, Some(12_348));
    // A scale that does not divide a second: nanoseconds, exactly.
    let bytes = file(
        "matroska",
        &[
            info(3, None),
            tracks(&[mp3()]),
            cluster(1_000, &[simple(2, -1, true, b"a")]),
        ],
    );
    let mut demux = open(bytes).expect("open");
    assert_eq!(
        demux.streams()[0].timescale().ticks_per_second(),
        1_000_000_000
    );
    assert_eq!(rows(&all(&mut demux))[0].1, Some(2_997));
    // Zero is not a scale.
    let bytes = file("matroska", &[info(0, None), tracks(&[mp3()])]);
    assert!(open(bytes).is_err());
}

#[test]
fn info_and_tracks_after_the_clusters_are_found_through_the_seek_head() {
    let bytes = with_seek_head(
        &[ids::INFO, ids::TRACKS],
        &[
            cluster(0, &[simple(2, 0, true, b"a"), simple(2, 20, true, b"b")]),
            info(1_000_000, Some(40.0)),
            tracks(&[mp3()]),
        ],
    );
    let mut demux = open(bytes.clone()).expect("open");
    assert_eq!(demux.streams().len(), 1);
    assert_eq!(rows(&all(&mut demux)).len(), 2);
    // A forward reader cannot look ahead for them.
    assert!(MatroskaDemux::open_forward(Cursor::new(bytes)).is_err());
}

fn opus_head(pre_skip: u16) -> Vec<u8> {
    let mut head = b"OpusHead".to_vec();
    head.push(1);
    head.push(2);
    head.extend_from_slice(&pre_skip.to_le_bytes());
    head.extend_from_slice(&48_000_u32.to_le_bytes());
    head.extend_from_slice(&[0, 0, 0]);
    head
}

#[test]
fn opus_codec_delay_is_an_edit_and_the_pre_skip_a_trim() {
    let mut extra = uint(ids::CODEC_DELAY, 6_500_000);
    extra.extend(uint(ids::SEEK_PRE_ROLL, 80_000_000));
    let track = Track::audio(1, "A_OPUS", 48_000.0, 2)
        .with_private(&opus_head(312))
        .with_extra(&extra);
    let bytes = file(
        "webm",
        &[
            info(1_000_000, Some(1000.0)),
            tracks(&[track]),
            cluster(0, &[simple(1, 0, true, b"o")]),
        ],
    );
    let demux = open(bytes).expect("open");
    let stream = &demux.streams()[0];
    assert_eq!(stream.params().codec(), &StreamCodec::Known(CodecId::Opus));
    assert_eq!(stream.params().sample_rate(), Some(48_000));
    let edit = stream.edits()[0];
    // 6.5 ms rounds to 7 ticks of a millisecond.
    assert_eq!(edit.media_time(), Some(7));
    assert_eq!(edit.duration(), 993);
    assert_eq!(stream.trim().encoder_delay(), Some(312));
}

/// Two whole files one after the other, as a live server chains them: the
/// first Segment is read, forward and random, and the reader ends cleanly
/// where it ends. FFmpeg and GStreamer read on into the second Segment with
/// its times starting over, which one track list on one time line cannot
/// hold, so this reader keeps to the first.
#[test]
fn a_second_segment_after_the_first_is_not_read() {
    let part = |second: u8| {
        file(
            "webm",
            &[
                info(1_000_000, Some(40.0)),
                tracks(&[mp3()]),
                cluster(
                    0,
                    &[
                        simple(2, 0, true, &[second; 4]),
                        simple(2, 20, true, &[second; 4]),
                    ],
                ),
            ],
        )
    };
    let mut bytes = part(1);
    bytes.extend(part(2));
    for forward in [false, true] {
        let mut demux = if forward {
            MatroskaDemux::open_forward(Cursor::new(bytes.clone())).expect("open")
        } else {
            open(bytes.clone()).expect("open")
        };
        let got = all(&mut demux);
        assert_eq!(got.len(), 2, "forward {forward}");
        assert!(got.iter().all(|packet| packet.bytes() == [1; 4]));
        assert!(demux.read_packet().expect("end").is_none());
        if !forward {
            assert_eq!(landing_time(&seek(&mut demux, 2, 30)), Some(20));
            assert_eq!(all(&mut demux).len(), 1);
        }
    }
}

/// A BlockGroup with DiscardPadding, in nanoseconds.
fn discarding(track: u64, relative: i16, duration: Option<u64>, discard_ns: u64) -> Vec<u8> {
    let mut block = block_head(track, relative, 0);
    block.extend_from_slice(b"pad");
    let mut body = element(ids::BLOCK, &block);
    if let Some(duration) = duration {
        body.extend(uint(ids::BLOCK_DURATION, duration));
    }
    // A signed integer: eight octets keep the sign bit clear.
    body.extend(bytes(ids::DISCARD_PADDING, &discard_ns.to_be_bytes()));
    element(ids::BLOCK_GROUP, &body)
}

/// One Cluster of an Opus track with a 20 ms DefaultDuration and an AAC
/// track, each ending in a block with DiscardPadding: the Opus one has no
/// BlockDuration, the AAC one has the played part as its BlockDuration.
/// With `with_cues`, a Cues element points at the Cluster.
fn padded(with_cues: bool) -> Vec<u8> {
    let opus = Track::audio(1, "A_OPUS", 48_000.0, 2)
        .with_private(&opus_head(312))
        .with_default_duration(20_000_000)
        .with_extra(&uint(ids::CODEC_DELAY, 6_500_000));
    let aac = Track::audio(2, "A_AAC", 48_000.0, 2)
        .with_private(&[0x11, 0x90])
        .with_extra(&uint(ids::CODEC_DELAY, 21_333_333));
    let blocks = vec![
        simple(1, 0, true, b"o"),
        simple(2, 0, true, b"a"),
        discarding(1, 20, None, 6_500_000),
        // 1024 samples less 480 of padding play 11 ms.
        discarding(2, 21, Some(11), 10_000_000),
    ];
    let head = vec![info(1_000_000, Some(42.0)), tracks(&[opus, aac])];
    let mut position = 0_u64;
    loop {
        let mut children = head.clone();
        let mut wanted = vec![ids::INFO, ids::TRACKS];
        if with_cues {
            children.push(cues(&[(0, 1, position)]));
            wanted.push(ids::CUES);
        }
        children.push(cluster(0, &blocks));
        let bytes = with_seek_head(&wanted, &children);
        let at = bytes
            .windows(4)
            .position(|window| window == [0x1F, 0x43, 0xB6, 0x75])
            .expect("Cluster");
        let found = (at - segment_start(&bytes)) as u64;
        if !with_cues || found == position {
            return bytes;
        }
        position = found;
    }
}

#[test]
fn discard_padding_shortens_an_opus_packet_and_is_the_tracks_end_trim() {
    for with_cues in [true, false] {
        let mut demux = open(padded(with_cues)).expect("open");
        let trims: Vec<Option<u64>> = demux
            .streams()
            .iter()
            .map(|stream| stream.trim().trailing_padding())
            .collect();
        // The last block's padding, in samples, once the last Cluster is
        // found through the Cues.
        let want = if with_cues {
            vec![Some(312), Some(480)]
        } else {
            vec![None, None]
        };
        assert_eq!(trims, want);
        let got: Vec<(u32, Option<i64>)> = all(&mut demux)
            .iter()
            .map(|packet| (packet.stream(), packet.timestamp().duration()))
            .collect();
        // Opus plays 20 less 6.5 ms, rounded to the tick; AAC keeps its
        // whole 21 ms frame and the trim states the padding.
        assert_eq!(
            got,
            vec![(1, Some(20)), (2, None), (1, Some(13)), (2, Some(21))]
        );
    }
}

/// Three Clusters a second apart. Video: a keyframe at each Cluster start
/// and a frame every 100 ms. Audio: a keyframe every 100 ms, offset 50 ms.
fn seekable(with_cues: bool, cue_tracks: &[u64]) -> Vec<u8> {
    let mut clusters = Vec::new();
    for second in 0..3_u64 {
        let mut blocks = Vec::new();
        for step in 0..10_i16 {
            blocks.push(simple(
                1,
                step * 100,
                step == 0,
                &[u8::try_from(second).expect("s"); 300],
            ));
            blocks.push(simple(2, step * 100 + 50, true, b"aa"));
        }
        clusters.push(cluster(second * 1000, &blocks));
    }
    let head = vec![info(1_000_000, Some(3000.0)), tracks(&[h264(), mp3()])];
    let mut ids_wanted = vec![ids::INFO, ids::TRACKS];
    if with_cues {
        ids_wanted.push(ids::CUES);
    }
    let mut children = head.clone();
    children.extend(clusters.iter().cloned());
    if !with_cues {
        return with_seek_head(&ids_wanted, &children);
    }
    let mut cue_points = Vec::new();
    // Positions depend on the SeekHead size, which depends on the Cues
    // position; settle by iterating.
    let mut guess = 0_u64;
    loop {
        let points: Vec<(u64, u64, u64)> = cue_points.clone();
        let mut all = head.clone();
        all.extend(clusters.iter().cloned());
        all.push(cues(&points));
        let bytes = with_seek_head(&ids_wanted, &all);
        let segment_start = segment_start(&bytes);
        // Find each Cluster's ID in the file.
        let mut found = Vec::new();
        let mut at = 0;
        while let Some(offset) = bytes[at..]
            .windows(4)
            .position(|window| window == [0x1F, 0x43, 0xB6, 0x75])
        {
            found.push((at + offset - segment_start) as u64);
            at += offset + 4;
        }
        let mut next = Vec::new();
        for (index, position) in found.iter().enumerate() {
            for track in cue_tracks {
                next.push((index as u64 * 1000, *track, *position));
            }
        }
        if next == cue_points && guess > 0 {
            return bytes;
        }
        cue_points = next;
        guess += 1;
        assert!(guess < 10, "cue layout does not settle");
    }
}

/// Offset of the Segment's data: after the EBML header, the Segment ID
/// and its size.
fn segment_start(bytes: &[u8]) -> usize {
    let id = ebml_header("matroska").len() + 4;
    let (_, size_len) = cutlass_matroska::ebml::read_size(&bytes[id..]).expect("size");
    id + size_len
}

fn seek(demux: &mut Opened, track: u32, time: i64) -> cutlass_container::SeekLanding {
    demux
        .seek(SeekTo {
            track,
            time,
            end: None,
        })
        .expect("seek")
}

fn landing_time(landing: &cutlass_container::SeekLanding) -> Option<i64> {
    match landing.landing {
        Landing::Packet { time, .. } => Some(time),
        Landing::End => None,
    }
}

#[test]
fn seeks_land_on_the_last_keyframe_at_or_before_the_time_with_or_without_cues() {
    for (with_cues, cue_tracks) in [(true, vec![1_u64]), (true, vec![1, 2]), (false, vec![])] {
        let bytes = seekable(with_cues, &cue_tracks);
        let mut demux = open(bytes).expect("open");
        let landing = seek(&mut demux, 1, 1_500);
        assert_eq!(landing_time(&landing), Some(1_000), "cues {with_cues}");
        assert_eq!(landing.index_used, with_cues);
        assert_eq!(landing.payload_scans, 0);
        let Landing::Packet {
            keyframe, preroll, ..
        } = landing.landing
        else {
            panic!("end");
        };
        assert!(keyframe);
        // Video frames at 1000..=1400 are before 1500.
        assert_eq!(preroll, 5);
        let next = demux.read_packet().expect("read").expect("packet");
        assert_eq!(next.stream(), 1);
        assert_eq!(next.timestamp().presentation(), Some(1_000));
        assert!(next.keyframe());

        // Audio: every frame is a keyframe.
        let landing = seek(&mut demux, 2, 1_575);
        assert_eq!(landing_time(&landing), Some(1_550));
        let next = demux.read_packet().expect("read").expect("packet");
        assert_eq!(
            (next.stream(), next.timestamp().presentation()),
            (2, Some(1_550))
        );

        // Exactly on a keyframe, before the first, and at zero.
        assert_eq!(landing_time(&seek(&mut demux, 1, 2_000)), Some(2_000));
        assert_eq!(landing_time(&seek(&mut demux, 1, 0)), Some(0));
        assert_eq!(landing_time(&seek(&mut demux, 1, -40)), Some(0));
        assert_eq!(landing_time(&seek(&mut demux, 2, 10)), Some(50));
        // Late in the last Cluster.
        assert_eq!(landing_time(&seek(&mut demux, 1, 2_999)), Some(2_000));
        // At and past the stated duration.
        assert_eq!(seek(&mut demux, 1, 3_000).landing, Landing::End);
        assert!(demux.read_packet().expect("end").is_none());
        let stated_end = demux
            .seek(SeekTo {
                track: 1,
                time: 500,
                end: Some(400),
            })
            .expect("seek");
        assert_eq!(stated_end.landing, Landing::End);
        // A seek after the end reads again from the landing.
        assert_eq!(landing_time(&seek(&mut demux, 1, 700)), Some(0));
        let rest = all(&mut demux);
        assert_eq!(rest.len(), 60);
        // Unknown tracks are refused and leave the reader where it was.
        assert_eq!(
            demux.seek(SeekTo {
                track: 9,
                time: 0,
                end: None
            }),
            Err(ReadError::Seek(SeekError::UnknownTrack))
        );
    }
}

#[test]
fn seeks_over_a_stretch_walked_before_land_as_a_fresh_reader_does() {
    for (with_cues, cue_tracks) in [(true, vec![1_u64]), (true, vec![1, 2]), (false, vec![])] {
        let bytes = seekable(with_cues, &cue_tracks);
        let mut kept = open(bytes.clone()).expect("open");
        // Back and forth, so later seeks go over walks kept from earlier.
        let times: Vec<i64> = (0..120_i64).map(|i| (i * 7919) % 3_100 - 40).collect();
        for (index, time) in times.iter().enumerate() {
            let track = if index % 3 == 0 { 2 } else { 1 };
            let mut fresh = open(bytes.clone()).expect("open");
            let expected = seek(&mut fresh, track, *time);
            let landing = seek(&mut kept, track, *time);
            assert_eq!(
                landing, expected,
                "cues {with_cues} track {track} at {time}"
            );
            assert_eq!(
                kept.read_packet().expect("read"),
                fresh.read_packet().expect("read"),
                "cues {with_cues} track {track} at {time}"
            );
        }
    }
}

#[test]
fn seeks_land_the_same_once_the_kept_walks_are_full() {
    // 60 one-second Clusters of 500 blocks: more than the 16,384 kept.
    let mut children = vec![info(1_000_000, Some(60_000.0)), tracks(&[h264()])];
    for second in 0..60_u64 {
        let blocks: Vec<Vec<u8>> = (0..500_i16)
            .map(|step| simple(1, step * 2, step % 50 == 0, b"v"))
            .collect();
        children.push(cluster(second * 1000, &blocks));
    }
    let bytes = with_seek_head(&[ids::INFO, ids::TRACKS], &children);
    let mut kept = open(bytes.clone()).expect("open");
    for index in 0..150_i64 {
        let time = (index * 7919) % 60_000;
        let mut fresh = open(bytes.clone()).expect("open");
        assert_eq!(
            seek(&mut kept, 1, time),
            seek(&mut fresh, 1, time),
            "at {time}"
        );
        assert_eq!(
            kept.read_packet().expect("read"),
            fresh.read_packet().expect("read"),
            "at {time}"
        );
    }
}

#[test]
fn cues_that_point_nowhere_fall_back_to_the_cluster_headers() {
    let bytes = with_seek_head(
        &[ids::INFO, ids::TRACKS, ids::CUES],
        &[
            info(1_000_000, None),
            tracks(&[mp3()]),
            cues(&[(0, 2, 9_999_999)]),
            cluster(0, &[simple(2, 0, true, b"a")]),
            cluster(100, &[simple(2, 0, true, b"b")]),
        ],
    );
    let mut demux = open(bytes).expect("open");
    let landing = seek(&mut demux, 2, 150);
    assert_eq!(landing_time(&landing), Some(100));
    assert!(!landing.index_used);
}

#[test]
fn a_cancelled_reader_stops() {
    let bytes = seekable(false, &[]);
    let mut demux = open(bytes).expect("open");
    assert!(demux.read_packet().expect("first").is_some());
    demux.cancel();
    assert_eq!(demux.read_packet(), Err(ReadError::Cancelled));
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
fn blocks_of_unnamed_tracks_are_skipped_and_untimed_blocks_have_no_time() {
    // A block before the Cluster's Timestamp has no base time.
    let mut early = simple(2, 5, true, b"early");
    early.extend(uint(ids::TIMESTAMP, 100));
    early.extend(simple(7, 0, true, b"stray"));
    early.extend(simple(2, 0, true, b"late"));
    let bytes = file(
        "matroska",
        &[
            info(1_000_000, None),
            tracks(&[mp3()]),
            element(ids::CLUSTER, &early),
        ],
    );
    let got = rows(&all(&mut open(bytes).expect("open")));
    assert_eq!(
        got.iter()
            .map(|row| (row.1, row.3.clone()))
            .collect::<Vec<_>>(),
        vec![(None, b"early".to_vec()), (Some(100), b"late".to_vec())]
    );
}

#[test]
fn other_documents_and_newer_versions_are_told_apart() {
    let mut other = ebml_header("matroska");
    let at = other
        .windows(8)
        .position(|window| window == b"matroska")
        .expect("doctype");
    other[at..at + 8].copy_from_slice(b"notmkvxx");
    other.extend(element(ids::SEGMENT, &[]));
    assert_eq!(
        open(other).err(),
        Some(OpenError::Unrecognized {
            family: cutlass_codec::ContainerId::Matroska
        })
    );
    let mut newer = Vec::new();
    let mut body = bytes(ids::DOC_TYPE, b"matroska");
    body.extend(uint(ids::DOC_TYPE_READ_VERSION, 9));
    newer.extend(element(ids::EBML, &body));
    newer.extend(element(ids::SEGMENT, &[]));
    assert_eq!(
        open(newer).err(),
        Some(OpenError::UnsupportedBrand {
            family: cutlass_codec::ContainerId::Matroska
        })
    );
    assert!(matches!(
        open(b"RIFF\0\0\0\0WAVE".to_vec()).err(),
        Some(OpenError::Unrecognized { .. })
    ));
    assert!(matches!(
        open(Vec::new()).err(),
        Some(OpenError::Truncated { .. })
    ));
}

#[test]
fn a_segment_without_tracks_does_not_open() {
    let bytes = file(
        "matroska",
        &[
            info(1_000_000, None),
            cluster(0, &[simple(1, 0, true, b"x")]),
        ],
    );
    assert!(open(bytes).is_err());
}

#[test]
fn duplicate_track_numbers_and_too_many_tracks_are_refused() {
    let bytes = file("matroska", &[tracks(&[mp3(), mp3()])]);
    assert_eq!(open(bytes).err(), Some(OpenError::DuplicateTrack { id: 2 }));
    let many: Vec<Track> = (1..=65)
        .map(|number| Track::audio(number, "A_MPEG/L3", 48_000.0, 2))
        .collect();
    let bytes = file("matroska", &[tracks(&many)]);
    assert!(matches!(
        open(bytes).err(),
        Some(OpenError::HostileSize { .. })
    ));
}

#[test]
fn a_track_timestamp_scale_other_than_one_is_refused() {
    let mut scale = Vec::new();
    cutlass_matroska::ebml::put_float(&mut scale, ids::TRACK_TIMESTAMP_SCALE, 2.0);
    let bytes = file(
        "matroska",
        &[
            info(1_000_000, None),
            tracks(&[mp3().with_extra(&scale)]),
            cluster(0, &[simple(2, 0, true, b"x")]),
        ],
    );
    let Some(OpenError::Source { message }) = open(bytes).err() else {
        panic!("opened");
    };
    assert!(message.contains("TrackTimestampScale"), "{message}");
    // A scale of exactly one is the default and opens.
    let mut one = Vec::new();
    cutlass_matroska::ebml::put_float(&mut one, ids::TRACK_TIMESTAMP_SCALE, 1.0);
    let bytes = file(
        "matroska",
        &[
            info(1_000_000, None),
            tracks(&[mp3().with_extra(&one)]),
            cluster(0, &[simple(2, 0, true, b"x")]),
        ],
    );
    assert!(open(bytes).is_ok());
}

/// An element header with an eight-octet size.
fn huge(id: u32, size: u64) -> Vec<u8> {
    let mut out = Vec::new();
    cutlass_matroska::ebml::put_id(&mut out, id);
    cutlass_matroska::ebml::put_size_width(&mut out, size, 8);
    out
}

#[test]
fn hostile_sizes_are_refused_before_anything_that_size_is_allocated() {
    // A Tracks element that claims a terabyte.
    let mut bytes = ebml_header("matroska");
    bytes.extend(unknown(ids::SEGMENT, &huge(ids::TRACKS, 1 << 40)));
    assert!(matches!(
        open(bytes).err(),
        Some(OpenError::HostileSize { .. } | OpenError::Truncated { .. })
    ));
    // A block that claims a gigabyte inside an unknown-size Cluster.
    let mut body = uint(ids::TIMESTAMP, 0);
    body.extend(huge(ids::SIMPLE_BLOCK, 1 << 30));
    body.extend_from_slice(&[0x82, 0, 0, 0x80, 1, 2, 3]);
    let bytes = live_file(
        "matroska",
        &[
            info(1_000_000, None),
            tracks(&[mp3()]),
            unknown(ids::CLUSTER, &body),
        ],
    );
    let mut demux = open(bytes.clone()).expect("open");
    assert!(matches!(
        demux.read_packet(),
        Err(ReadError::HostileSize { .. } | ReadError::Truncated { .. })
    ));
    // The same on a forward reader, which has no length to check against.
    let mut forward = MatroskaDemux::open_forward(Cursor::new(bytes)).expect("forward");
    assert!(matches!(
        forward.read_packet(),
        Err(ReadError::HostileSize { .. })
    ));
    // A child that runs past its Cluster.
    let mut body = uint(ids::TIMESTAMP, 0);
    body.extend(huge(ids::SIMPLE_BLOCK, 100));
    let bytes = file(
        "matroska",
        &[
            info(1_000_000, None),
            tracks(&[mp3()]),
            element(ids::CLUSTER, &body),
        ],
    );
    assert!(open(bytes).expect("open").read_packet().is_err());
}

#[test]
fn bad_lacing_is_an_error_not_a_panic() {
    let mut block = block_head(2, 0, 0x80 | 0x02);
    block.extend_from_slice(&[3, 255, 255, 255]);
    let bytes = file(
        "matroska",
        &[
            info(1_000_000, None),
            tracks(&[mp3()]),
            cluster(0, &[element(ids::SIMPLE_BLOCK, &block)]),
        ],
    );
    assert!(open(bytes).expect("open").read_packet().is_err());
}

#[test]
fn every_truncation_opens_or_fails_cleanly_and_reads_a_prefix() {
    let whole = seekable(true, &[1]);
    let mut full = open(whole.clone()).expect("open");
    let expected = rows(&all(&mut full));
    for cut in 0..whole.len() {
        let Ok(mut demux) = open(whole[..cut].to_vec()) else {
            continue;
        };
        let mut got = Vec::new();
        while let Ok(Some(packet)) = demux.read_packet() {
            got.push(packet);
        }
        let got = rows(&got);
        assert!(got.len() <= expected.len());
        assert_eq!(got, expected[..got.len()], "cut {cut}");
        // A seek on the cut file answers or fails; it does not panic.
        let _ = demux.seek(SeekTo {
            track: 1,
            time: 1_500,
            end: None,
        });
    }
}

#[test]
fn probe_accepts_built_files() {
    let bytes = seekable(false, &[]);
    assert!(matches!(
        cutlass_matroska::probe(&bytes[..64]),
        cutlass_container::ProbeResult::Accept { .. }
    ));
}

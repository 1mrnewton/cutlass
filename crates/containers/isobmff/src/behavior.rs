//! Demux and mux behavior that does not depend on the muxer agreeing with itself
//! for every row. Hand-built files cover `stco`, `stz2`, an edit list, an empty
//! sync table, and hostile boxes. Mux round trips cover the writer.

use crate::{
    CodecId, Demux, Edit, Error, FileBrand, IsoBmff, IsoBmffFile, IsoBmffMux, MediaKind, Mux,
    MuxOptions, Packet, SliceSource, StreamParams, Timescale,
};
use std::io::{Cursor, Seek, SeekFrom, Write};

fn video() -> StreamParams {
    let mut params = StreamParams::video(CodecId::H264, Timescale(30), 16, 8);
    params.extradata = vec![1, 66, 0, 10];
    params.language = *b"eng";
    params
}

fn audio() -> StreamParams {
    let mut params = StreamParams::audio(
        CodecId::Aac,
        Timescale(48_000),
        48_000,
        2,
        16,
        Default::default(),
    );
    params.extradata = vec![0x12, 0x10];
    params
}

fn packet(dts: i64, pts: i64, duration: u32, payload: &[u8], keyframe: bool, scale: u32) -> Packet {
    Packet {
        track_id: 0,
        payload: payload.to_vec(),
        dts,
        pts,
        duration,
        timescale: Timescale(scale),
        keyframe,
    }
}

fn write_movie(options: MuxOptions, tracks: Vec<(StreamParams, Vec<Packet>)>) -> Vec<u8> {
    let mut mux = IsoBmffMux::new(Cursor::new(Vec::new()), options);
    for (params, packets) in tracks {
        let id = mux.add_stream(&params).expect("stream");
        let scale = params.timescale;
        for mut packet in packets {
            packet.track_id = id;
            packet.timescale = scale;
            mux.write_packet(&packet).expect("packet");
        }
    }
    mux.finish_into().expect("finish").into_inner()
}

fn open(bytes: &[u8]) -> IsoBmff {
    let mut source = SliceSource::new(bytes);
    IsoBmff::open(&mut source).expect("open")
}

fn must_err(bytes: &[u8]) -> Error {
    match IsoBmff::open(&mut SliceSource::new(bytes)) {
        Ok(_) => panic!("opened a file that should fail"),
        Err(err) => err,
    }
}

fn read_all(bytes: &[u8]) -> Vec<Packet> {
    let mut movie = open(bytes);
    let mut source = SliceSource::new(bytes);
    let mut packets = Vec::new();
    while let Some(packet) = movie.read_packet(&mut source).expect("read") {
        packets.push(packet);
    }
    packets
}

#[test]
fn the_shared_demux_returns_the_same_sample_bytes() {
    let bytes = write_movie(
        MuxOptions::default(),
        vec![(video(), vec![packet(0, 3, 1, &[9, 8, 7], true, 30)])],
    );
    let mut file = IsoBmffFile::open(Cursor::new(bytes)).expect("open");
    assert_eq!(
        cutlass_container::Demux::family(&file),
        cutlass_codec::ContainerId::Isobmff
    );
    let streams = cutlass_container::Demux::streams(&file);
    assert_eq!(streams.len(), 1);
    assert_eq!(streams[0].timescale().ticks_per_second(), 30);
    assert_eq!(
        streams[0].params().codec(),
        &cutlass_codec::StreamCodec::Known(cutlass_codec::CodecId::H264)
    );
    assert_eq!(streams[0].params().extradata(), &[1, 66, 0, 10]);
    let before = cutlass_container::Demux::consumed(&file);
    let packet = cutlass_container::Demux::read_packet(&mut file)
        .expect("read")
        .expect("packet");
    assert_eq!(packet.bytes(), &[9, 8, 7]);
    assert_eq!(packet.timestamp().decode(), Some(0));
    assert_eq!(packet.timestamp().presentation(), Some(3));
    assert_eq!(packet.timestamp().duration(), Some(1));
    assert_eq!(packet.timescale().ticks_per_second(), 30);
    assert!(packet.keyframe());
    assert!(cutlass_container::Demux::consumed(&file) > before);
    let consumed = cutlass_container::Demux::consumed(&file);
    let landing = cutlass_container::Demux::seek(
        &mut file,
        cutlass_container::SeekTo {
            track: packet.stream(),
            time: 0,
            end: None,
        },
    )
    .expect("seek");
    assert!(landing.index_used);
    assert!(landing.entries_visited > 0);
    assert_eq!(landing.payload_scans, 0);
    assert_eq!(cutlass_container::Demux::consumed(&file), consumed);
    match landing.landing {
        cutlass_container::Landing::Packet {
            time,
            keyframe,
            byte_position,
            ..
        } => {
            assert_eq!(time, 3);
            assert!(keyframe);
            assert!(byte_position > 0);
        }
        cutlass_container::Landing::End => panic!("seek landed at the end"),
    }
    let again = cutlass_container::Demux::read_packet(&mut file)
        .expect("read")
        .expect("packet");
    assert_eq!(again.bytes(), &[9, 8, 7]);
    let failed = cutlass_container::Demux::seek(
        &mut file,
        cutlass_container::SeekTo {
            track: 99,
            time: 0,
            end: None,
        },
    );
    assert!(matches!(
        failed,
        Err(cutlass_container::ReadError::Seek(
            cutlass_container::SeekError::UnknownTrack
        ))
    ));
}

#[test]
fn moov_after_mdat_round_trips_times_and_bytes() {
    let bytes = write_movie(
        MuxOptions::default(),
        vec![(
            video(),
            vec![
                packet(0, 0, 30, b"one", true, 30),
                packet(30, 30, 30, b"two", true, 30),
            ],
        )],
    );
    assert!(bytes.windows(4).any(|word| word == b"mdat"));
    assert!(bytes.windows(4).any(|word| word == b"moov"));
    let movie = open(&bytes);
    assert_eq!(movie.movie_timescale().get(), 30);
    assert_eq!(movie.brands()[0], *b"isom");
    let stream = &movie.streams()[0];
    assert_eq!(stream.params.codec, CodecId::H264);
    assert_eq!(stream.params.extradata, vec![1, 66, 0, 10]);
    assert_eq!(stream.params.language, *b"eng");
    match stream.params.kind {
        MediaKind::Video(video) => {
            assert_eq!((video.width, video.height), (16, 8));
        }
        MediaKind::Audio(_) | MediaKind::Data => panic!("video track"),
    }
    let packets = read_all(&bytes);
    assert_eq!(packets.len(), 2);
    assert_eq!(packets[0].payload, b"one");
    assert_eq!(packets[1].payload, b"two");
    assert_eq!(packets[0].dts, 0);
    assert_eq!(packets[1].dts, 30);
    assert!(packets[0].keyframe);
}

#[test]
fn moov_before_mdat_matches_the_end_layout() {
    let samples = vec![
        packet(0, 10, 10, b"aaa", true, 30),
        packet(10, 0, 10, b"bbb", false, 30),
        packet(20, 20, 10, b"ccc", true, 30),
    ];
    let early = MuxOptions {
        moov_at_end: false,
        ..MuxOptions::default()
    };
    let first = read_all(&write_movie(early, vec![(video(), samples.clone())]));
    let last = read_all(&write_movie(
        MuxOptions::default(),
        vec![(video(), samples)],
    ));
    assert_eq!(first, last);
    assert_eq!(first[1].pts, 0);
    assert_eq!(first[1].dts, 10);
    assert!(!first[1].keyframe);
}

#[test]
fn an_edit_list_is_exposed_and_not_applied_to_packet_time() {
    let mut mux = IsoBmffMux::new(Cursor::new(Vec::new()), MuxOptions::default());
    let id = mux.add_stream(&video()).unwrap();
    mux.set_edits(
        id,
        vec![Edit {
            segment_duration: 30,
            media_time: Some(30),
            media_rate: 0x0001_0000,
        }],
    )
    .unwrap();
    let mut sample = packet(0, 0, 30, b"pic", true, 30);
    sample.track_id = id;
    mux.write_packet(&sample).unwrap();
    let bytes = mux.finish_into().unwrap().into_inner();
    let movie = open(&bytes);
    let edit = movie.streams()[0].edits[0];
    assert_eq!(edit.media_time, Some(30));
    assert_eq!(edit.segment_duration, 30);
    assert_eq!(read_all(&bytes)[0].dts, 0);
}

#[test]
fn interleaved_tracks_read_in_file_order() {
    let mut mux = IsoBmffMux::new(Cursor::new(Vec::new()), MuxOptions::default());
    let video_id = mux.add_stream(&video()).unwrap();
    let audio_id = mux.add_stream(&audio()).unwrap();
    for (id, scale, sample) in [
        (video_id, 30, packet(0, 0, 30, b"V0", true, 30)),
        (audio_id, 48_000, packet(0, 0, 1024, b"A0", true, 48_000)),
        (video_id, 30, packet(30, 30, 30, b"V1", true, 30)),
    ] {
        let mut sample = sample;
        sample.track_id = id;
        sample.timescale = Timescale(scale);
        mux.write_packet(&sample).unwrap();
    }
    let bytes = mux.finish_into().unwrap().into_inner();
    let packets = read_all(&bytes);
    let payloads: Vec<_> = packets
        .iter()
        .map(|packet| packet.payload.as_slice())
        .collect();
    assert_eq!(
        payloads,
        [b"V0".as_slice(), b"A0".as_slice(), b"V1".as_slice()]
    );
    assert!(
        packets
            .iter()
            .any(|packet| packet.timescale.get() == 48_000)
    );
    assert_eq!(open(&bytes).brands()[0], *b"isom");
}

#[test]
fn brands_select_the_file_type() {
    for (brand, major) in [
        (FileBrand::Mov, *b"qt  "),
        (FileBrand::M4a, *b"M4A "),
        (FileBrand::M4v, *b"M4V "),
        (FileBrand::ThreeGp, *b"3gp4"),
    ] {
        let options = MuxOptions {
            brand,
            ..MuxOptions::default()
        };
        let bytes = write_movie(
            options,
            vec![(video(), vec![packet(0, 0, 30, b"x", true, 30)])],
        );
        assert_eq!(open(&bytes).brands()[0], major, "{brand:?}");
        assert!(crate::probe(&bytes));
    }
}

#[test]
fn seek_lands_on_the_sync_sample_at_or_before_the_time() {
    let bytes = write_movie(
        MuxOptions::default(),
        vec![(
            video(),
            vec![
                packet(0, 0, 10, b"K0", true, 30),
                packet(10, 10, 10, b"D1", false, 30),
                packet(20, 20, 10, b"K2", true, 30),
            ],
        )],
    );
    let mut movie = open(&bytes);
    let mut source = SliceSource::new(&bytes);
    let at = movie.seek(&mut source, 1, 15).unwrap();
    assert_eq!(at.dts, 0);
    assert!(at.keyframe);
    assert!(at.used_index);
    assert_eq!(at.preroll, 1);
    assert_eq!(
        movie.read_packet(&mut source).unwrap().unwrap().payload,
        b"K0"
    );

    let at = movie.seek(&mut source, 1, 20).unwrap();
    assert_eq!(at.dts, 20);
    assert_eq!(at.preroll, 0);
    assert_eq!(
        movie.read_packet(&mut source).unwrap().unwrap().payload,
        b"K2"
    );

    let at = movie.seek(&mut source, 1, -5).unwrap();
    assert_eq!(at.dts, 0);
    let end = movie.seek(&mut source, 1, 1_000).unwrap();
    assert!(!end.keyframe);
    assert!(movie.read_packet(&mut source).unwrap().is_none());
}

#[test]
fn a_failed_seek_leaves_the_next_packet() {
    let bytes = write_movie(
        MuxOptions::default(),
        vec![(video(), vec![packet(0, 0, 30, b"stay", true, 30)])],
    );
    let mut movie = open(&bytes);
    let mut source = SliceSource::new(&bytes);
    let err = movie.seek(&mut source, 99, 0).unwrap_err();
    assert_eq!(err, Error::Missing { what: "track" });
    assert_eq!(
        movie.read_packet(&mut source).unwrap().unwrap().payload,
        b"stay"
    );
}

#[test]
fn fragmented_seek_crosses_to_the_sync_sample() {
    let options = MuxOptions {
        fragmented: true,
        samples_per_fragment: 1,
        ..MuxOptions::default()
    };
    let bytes = write_movie(
        options,
        vec![(
            video(),
            vec![
                packet(0, 0, 10, b"K0", true, 30),
                packet(10, 10, 10, b"D1", false, 30),
                packet(20, 20, 10, b"K2", true, 30),
            ],
        )],
    );
    assert_eq!(open(&bytes).brands()[0], *b"iso5");
    let packets = read_all(&bytes);
    assert_eq!(packets[1].payload, b"D1");
    assert_eq!(packets[1].dts, 10);
    let mut movie = open(&bytes);
    let mut source = SliceSource::new(&bytes);
    let at = movie.seek(&mut source, 1, 15).unwrap();
    assert_eq!(at.dts, 0);
    assert!(at.keyframe);
    assert_eq!(
        movie.read_packet(&mut source).unwrap().unwrap().payload,
        b"K0"
    );
    assert_eq!(
        movie.read_packet(&mut source).unwrap().unwrap().payload,
        b"D1"
    );
}

#[test]
fn a_fragment_keeps_the_caller_interleave() {
    let mut mux = IsoBmffMux::new(
        Cursor::new(Vec::new()),
        MuxOptions {
            fragmented: true,
            samples_per_fragment: 4,
            ..MuxOptions::default()
        },
    );
    let video_id = mux.add_stream(&video()).unwrap();
    let audio_id = mux.add_stream(&audio()).unwrap();
    let mut samples = [
        (video_id, packet(0, 0, 30, b"V0", true, 30)),
        (audio_id, packet(0, 2, 1_024, b"A0", true, 48_000)),
        (video_id, packet(30, 31, 30, b"V1", false, 30)),
        (audio_id, packet(1_024, 1_025, 1_024, b"A1", true, 48_000)),
    ];
    for (id, sample) in &mut samples {
        sample.track_id = *id;
        mux.write_packet(sample).unwrap();
    }
    let bytes = mux.finish_into().unwrap().into_inner();
    let packets = read_all(&bytes);
    let payloads: Vec<_> = packets
        .iter()
        .map(|packet| packet.payload.as_slice())
        .collect();
    assert_eq!(payloads, [b"V0".as_slice(), b"A0", b"V1", b"A1"]);
    assert_eq!(packets[1].pts - packets[1].dts, 2);
    assert_eq!(packets[2].pts - packets[2].dts, 1);
}

#[test]
fn dropping_the_muxer_does_not_write_the_index() {
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut mux = IsoBmffMux::new(&mut cursor, MuxOptions::default());
        let id = mux.add_stream(&video()).unwrap();
        let mut sample = packet(0, 0, 30, b"abcd", true, 30);
        sample.track_id = id;
        mux.write_packet(&sample).unwrap();
    }
    let bytes = cursor.into_inner();
    assert!(bytes.windows(4).any(|word| word == b"mdat"));
    assert!(!bytes.windows(4).any(|word| word == b"moov"));
    assert_eq!(must_err(&bytes), Error::HostileSize { at: 28, claimed: 0 });
}

#[test]
fn the_writer_rejects_a_broken_timeline_and_a_foreign_packet() {
    let mut mux = IsoBmffMux::new(Cursor::new(Vec::new()), MuxOptions::default());
    let id = mux.add_stream(&video()).unwrap();
    let mut sample = packet(5, 5, 30, b"late", true, 30);
    sample.track_id = id;
    assert!(matches!(
        mux.write_packet(&sample),
        Err(Error::Unsupported { .. })
    ));
    sample.dts = 0;
    sample.pts = 0;
    sample.timescale = Timescale(1);
    assert!(matches!(
        mux.write_packet(&sample),
        Err(Error::Unsupported { .. })
    ));
    let mut zero = video();
    zero.timescale = Timescale(0);
    assert!(mux.add_stream(&zero).is_err());
}

/// A writer that refuses the payload once, then accepts every later write.
struct FailPayload {
    inner: Cursor<Vec<u8>>,
    armed: bool,
}

impl Write for FailPayload {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.armed && buf == b"payl" {
            self.armed = false;
            return Err(std::io::Error::other("payload"));
        }
        self.inner.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl Seek for FailPayload {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

#[test]
fn a_failed_payload_write_keeps_the_decode_time() {
    let mut mux = IsoBmffMux::new(
        FailPayload {
            inner: Cursor::new(Vec::new()),
            armed: true,
        },
        MuxOptions::default(),
    );
    let id = mux.add_stream(&video()).unwrap();
    let mut sample = packet(0, 0, 30, b"payl", true, 30);
    sample.track_id = id;
    assert!(matches!(mux.write_packet(&sample), Err(Error::Io { .. })));
    mux.write_packet(&sample).unwrap();
    let bytes = mux.finish_into().unwrap().inner.into_inner();
    let packets = read_all(&bytes);
    assert_eq!(packets.len(), 1);
    assert_eq!(packets[0].payload, b"payl");
    assert_eq!(packets[0].dts, 0);
}

#[test]
fn an_unknown_codec_still_returns_its_packets() {
    let mut params = StreamParams::video(CodecId::Unknown(*b"zzzz"), Timescale(1), 2, 2);
    params.sample_entry = *b"zzzz";
    let bytes = write_movie(
        MuxOptions::default(),
        vec![(params, vec![packet(0, 0, 1, b"raw", true, 1)])],
    );
    let movie = open(&bytes);
    assert_eq!(movie.streams()[0].params.codec, CodecId::Unknown(*b"zzzz"));
    assert_eq!(read_all(&bytes)[0].payload, b"raw");
}

#[test]
fn probe_and_open_reject_the_wrong_family_and_a_short_header() {
    assert!(!crate::probe(b""));
    assert!(!crate::probe(b"RIFF____"));
    assert_eq!(must_err(b""), Error::Truncated { at: 0, needed: 8 });
    assert_eq!(must_err(b"RIFF____"), Error::NotThisFamily);
    let unknown = box_of(b"ftyp", b"xxxx\0\0\0\0");
    assert!(matches!(
        IsoBmff::open(&mut SliceSource::new(&unknown)),
        Err(Error::UnsupportedBrand { .. })
    ));
}

#[test]
fn a_hand_built_stco_file_keeps_chunk_samples_edits_and_sync() {
    let bytes = hand_built(HandBuilt {
        wide_offsets: false,
        compact_sizes: false,
        sync: SyncTable::FirstOnly,
        negative_cts: true,
        edit: true,
        external: false,
        outside: false,
    });
    let movie = open(&bytes);
    let edit = &movie.streams()[0].edits[0];
    assert_eq!(edit.media_time, Some(1000));
    assert!(edit.media_rate == 0x0001_0000);
    let packets = read_all(&bytes);
    assert_eq!(packets.len(), 2);
    assert_eq!(packets[0].payload, b"abc");
    assert_eq!(packets[1].payload, b"WXYZ");
    assert_eq!(packets[0].dts, 0);
    assert_eq!(packets[1].dts, 1000);
    assert_eq!(packets[0].pts, 1000);
    assert_eq!(packets[1].pts, 0);
    assert!(packets[0].keyframe);
    assert!(!packets[1].keyframe);
    let mut movie = open(&bytes);
    let mut source = SliceSource::new(&bytes);
    let at = movie.seek(&mut source, 1, 1500).unwrap();
    assert_eq!(at.dts, 0);
    assert_eq!(at.preroll, 1);
}

#[test]
fn compact_sizes_and_wide_offsets_read_the_same_samples() {
    let bytes = hand_built(HandBuilt {
        wide_offsets: true,
        compact_sizes: true,
        sync: SyncTable::Absent,
        negative_cts: false,
        edit: false,
        external: false,
        outside: false,
    });
    let packets = read_all(&bytes);
    assert!(packets.iter().all(|packet| packet.keyframe));
    assert_eq!(packets[0].payload, b"abc");
    assert_eq!(packets[1].payload.len(), 4);
    assert_eq!(packets[1].payload, b"WXYZ");
}

#[test]
fn an_empty_sync_table_can_be_read_and_cannot_be_seeked() {
    let bytes = hand_built(HandBuilt {
        wide_offsets: false,
        compact_sizes: false,
        sync: SyncTable::Empty,
        negative_cts: false,
        edit: false,
        external: false,
        outside: false,
    });
    assert_eq!(read_all(&bytes).len(), 2);
    let mut movie = open(&bytes);
    let mut source = SliceSource::new(&bytes);
    let err = movie.seek(&mut source, 1, 0).unwrap_err();
    assert!(matches!(err, Error::SeekLimited { .. }));
    assert_eq!(
        movie.read_packet(&mut source).unwrap().unwrap().payload,
        b"abc"
    );
}

#[test]
fn a_sample_outside_mdat_and_an_external_url_fail_closed() {
    let outside = hand_built(HandBuilt {
        wide_offsets: false,
        compact_sizes: false,
        sync: SyncTable::Absent,
        negative_cts: false,
        edit: false,
        external: false,
        outside: true,
    });
    let mut movie = open(&outside);
    let err = movie
        .read_packet(&mut SliceSource::new(&outside))
        .unwrap_err();
    assert!(matches!(err, Error::SampleExtendsPastMedia { .. }));

    let external = hand_built(HandBuilt {
        wide_offsets: false,
        compact_sizes: false,
        sync: SyncTable::Absent,
        negative_cts: false,
        edit: false,
        external: true,
        outside: false,
    });
    assert_eq!(must_err(&external), Error::ExternalReference);
}

#[test]
fn hostile_size_and_deep_wave_boxes_are_errors() {
    let hostile = [0, 0, 0, 40, b'm', b'o', b'o', b'v'];
    assert!(matches!(
        IsoBmff::open(&mut SliceSource::new(&hostile)),
        Err(Error::HostileSize { .. })
    ));
    let nested = nested_wave_movie(40);
    assert!(matches!(
        IsoBmff::open(&mut SliceSource::new(&nested)),
        Err(Error::Nesting { .. })
    ));
}

#[test]
fn durations_accumulate_in_the_file_timescale() {
    let mut samples = Vec::new();
    let mut dts = 0i64;
    for duration in [1u32, 2, 2, 5, 1, 8, 3] {
        samples.push(packet(
            dts,
            dts,
            duration,
            &[u8::try_from(duration).unwrap()],
            true,
            1000,
        ));
        dts += i64::from(duration);
    }
    let mut params = video();
    params.timescale = Timescale(1000);
    let packets = read_all(&write_movie(MuxOptions::default(), vec![(params, samples)]));
    let mut expect = 0i64;
    for packet in packets {
        assert_eq!(packet.dts, expect);
        assert_eq!(packet.pts, expect);
        expect += i64::from(packet.duration);
    }
    assert_eq!(expect, 1 + 2 + 2 + 5 + 1 + 8 + 3);
}

#[test]
fn a_heif_item_is_one_packet() {
    let tile = b"hevc-tile";
    let bytes = heif_item(tile);
    assert!(crate::probe(&bytes));
    let movie = open(&bytes);
    assert_eq!(movie.streams()[0].params.codec, CodecId::Hevc);
    assert_eq!(movie.streams()[0].params.timescale.get(), 1);
    let packets = read_all(&bytes);
    assert_eq!(packets.len(), 1);
    assert_eq!(packets[0].payload, tile);
    assert_eq!(packets[0].dts, 0);
    assert!(packets[0].keyframe);
}

#[test]
fn seek_reads_the_landed_sample_and_not_the_media_prefix() {
    let bytes = write_movie(
        MuxOptions::default(),
        vec![(
            video(),
            vec![
                packet(0, 0, 10, &[1u8; 64], true, 30),
                packet(10, 10, 10, &[2u8; 64], true, 30),
                packet(20, 20, 10, &[3u8; 64], true, 30),
            ],
        )],
    );
    let mut source = Counting::new(bytes);
    let mut movie = IsoBmff::open(&mut source).unwrap();
    source.reads.clear();
    let at = movie.seek(&mut source, 1, 20).unwrap();
    assert_eq!(at.dts, 20);
    assert!(
        source.reads.iter().all(|(_, len)| *len <= 16),
        "seek walked media bytes: {:?}",
        source.reads
    );
    let packet = movie.read_packet(&mut source).unwrap().unwrap();
    assert_eq!(packet.payload, vec![3u8; 64]);
    assert!(source.reads.iter().any(|(_, len)| *len == 64));
}

struct Counting {
    data: Vec<u8>,
    reads: Vec<(u64, usize)>,
}

impl Counting {
    fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            reads: Vec::new(),
        }
    }
}

impl crate::Source for Counting {
    fn len(&self) -> u64 {
        u64::try_from(self.data.len()).unwrap_or(0)
    }

    fn read_exact_at(&mut self, at: u64, buf: &mut [u8]) -> Result<(), Error> {
        self.reads.push((at, buf.len()));
        SliceSource::new(&self.data).read_exact_at(at, buf)
    }
}

#[derive(Clone, Copy)]
struct HandBuilt {
    wide_offsets: bool,
    compact_sizes: bool,
    sync: SyncTable,
    negative_cts: bool,
    edit: bool,
    external: bool,
    outside: bool,
}

#[derive(Clone, Copy)]
enum SyncTable {
    Absent,
    Empty,
    FirstOnly,
}

fn hand_built(spec: HandBuilt) -> Vec<u8> {
    let payload = b"abcWXYZ";
    let ftyp = box_of(b"ftyp", b"isom\0\0\0\0isom");
    let mdat = box_of(b"mdat", payload);
    let data_at = u64::try_from(ftyp.len()).unwrap() + 8;
    let chunk_at = if spec.outside {
        data_at + 10_000
    } else {
        data_at
    };
    let mut trak = Vec::new();
    trak.extend(tkhd());
    if spec.edit {
        let mut elst = u32_bytes(1);
        elst.extend(u32_bytes(2000));
        elst.extend(1_000i32.to_be_bytes());
        elst.extend(u32_bytes(0x0001_0000));
        trak.extend(box_of(b"edts", &box_of(b"elst", &fullbox(&elst))));
    }
    let mut stbl = Vec::new();
    stbl.extend(box_of(b"stsd", &stsd_visual()));
    let mut stts = u32_bytes(1);
    stts.extend(u32_bytes(2));
    stts.extend(u32_bytes(1000));
    stbl.extend(box_of(b"stts", &fullbox(&stts)));
    if spec.negative_cts {
        let mut ctts = vec![1, 0, 0, 0];
        ctts.extend(u32_bytes(2));
        ctts.extend(u32_bytes(1));
        ctts.extend(1_000i32.to_be_bytes());
        ctts.extend(u32_bytes(1));
        ctts.extend((-1_000i32).to_be_bytes());
        stbl.extend(box_of(b"ctts", &ctts));
    }
    let mut stsc = u32_bytes(1);
    stsc.extend(u32_bytes(1));
    stsc.extend(u32_bytes(2));
    stsc.extend(u32_bytes(1));
    stbl.extend(box_of(b"stsc", &fullbox(&stsc)));
    if spec.compact_sizes {
        let mut stz2 = vec![0, 0, 0, 0, 0, 0, 0, 8];
        stz2.extend(u32_bytes(2));
        stz2.push(3);
        stz2.push(4);
        stbl.extend(box_of(b"stz2", &stz2));
    } else {
        let mut stsz = u32_bytes(0);
        stsz.extend(u32_bytes(2));
        stsz.extend(u32_bytes(3));
        stsz.extend(u32_bytes(4));
        stbl.extend(box_of(b"stsz", &fullbox(&stsz)));
    }
    if spec.wide_offsets {
        let mut co64 = u32_bytes(1);
        co64.extend(chunk_at.to_be_bytes());
        stbl.extend(box_of(b"co64", &fullbox(&co64)));
    } else {
        let mut stco = u32_bytes(1);
        stco.extend(u32_bytes(u32::try_from(chunk_at).unwrap()));
        stbl.extend(box_of(b"stco", &fullbox(&stco)));
    }
    match spec.sync {
        SyncTable::Absent => {}
        SyncTable::Empty => stbl.extend(box_of(b"stss", &fullbox(&u32_bytes(0)))),
        SyncTable::FirstOnly => {
            let mut stss = u32_bytes(1);
            stss.extend(u32_bytes(1));
            stbl.extend(box_of(b"stss", &fullbox(&stss)));
        }
    }
    let url_flags: [u8; 4] = if spec.external {
        [0, 0, 0, 0]
    } else {
        [0, 0, 0, 1]
    };
    let dref = {
        let url = box_of(b"url ", &url_flags);
        let mut body = u32_bytes(1);
        body.extend(url);
        box_of(b"dinf", &box_of(b"dref", &fullbox(&body)))
    };
    let minf = {
        let mut body = dref;
        body.extend(box_of(b"stbl", &stbl));
        box_of(b"minf", &body)
    };
    let mdhd = {
        let mut body = vec![0u8; 12];
        body.extend(u32_bytes(1000));
        body.extend(u32_bytes(2000));
        body.extend(0u16.to_be_bytes());
        body.extend(0u16.to_be_bytes());
        box_of(b"mdhd", &body)
    };
    let hdlr = box_of(b"hdlr", &[0, 0, 0, 0, 0, 0, 0, 0, b'v', b'i', b'd', b'e']);
    let mut mdia = mdhd;
    mdia.extend(hdlr);
    mdia.extend(minf);
    trak.extend(box_of(b"mdia", &mdia));
    let moov = {
        let mut body = mvhd(1000);
        body.extend(box_of(b"trak", &trak));
        box_of(b"moov", &body)
    };
    let mut file = ftyp;
    file.extend(mdat);
    file.extend(moov);
    file
}

fn nested_wave_movie(depth: usize) -> Vec<u8> {
    let mut child = Vec::new();
    for _ in 0..depth {
        child = box_of(b"wave", &child);
    }
    let mut entry = vec![0u8; 78];
    entry.extend(child);
    let stsd = box_of(
        b"stsd",
        &fullbox(&{
            let mut body = u32_bytes(1);
            body.extend(box_of(b"avc1", &entry));
            body
        }),
    );
    let stbl = box_of(b"stbl", &stsd);
    let minf = box_of(b"minf", &stbl);
    let mdhd = box_of(
        b"mdhd",
        &[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x03, 0xe8, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
    );
    let hdlr = box_of(b"hdlr", &[0, 0, 0, 0, 0, 0, 0, 0, b'v', b'i', b'd', b'e']);
    let mut mdia = mdhd;
    mdia.extend(hdlr);
    mdia.extend(minf);
    let mut trak = tkhd();
    trak.extend(box_of(b"mdia", &mdia));
    let mut moov = mvhd(1000);
    moov.extend(box_of(b"trak", &trak));
    let mut file = box_of(b"ftyp", b"isom\0\0\0\0isom");
    file.extend(box_of(b"moov", &moov));
    file
}

fn heif_item(tile: &[u8]) -> Vec<u8> {
    let ftyp = box_of(b"ftyp", b"mif1\0\0\0\0heic");
    let hdlr = box_of(b"hdlr", &[0, 0, 0, 0, 0, 0, 0, 0, b'p', b'i', b'c', b't']);
    let pitm = box_of(b"pitm", &fullbox(&1u16.to_be_bytes()));
    let mut infe = vec![2, 0, 0, 0];
    infe.extend(1u16.to_be_bytes());
    infe.extend(0u16.to_be_bytes());
    infe.extend(b"hvc1");
    let infe = box_of(b"infe", &infe);
    let mut iinf_body = 1u16.to_be_bytes().to_vec();
    iinf_body.extend(infe);
    let iinf = box_of(b"iinf", &fullbox(&iinf_body));
    let mut meta_head = vec![0, 0, 0, 0];
    meta_head.extend(hdlr);
    meta_head.extend(pitm);
    meta_head.extend(iinf);
    let iloc = iloc_box(0, u32::try_from(tile.len()).unwrap());
    let data_at = ftyp.len() + 8 + meta_head.len() + iloc.len() + 8;
    let iloc = iloc_box(
        u32::try_from(data_at).unwrap(),
        u32::try_from(tile.len()).unwrap(),
    );
    meta_head.extend(iloc);
    let mut file = ftyp;
    file.extend(box_of(b"meta", &meta_head));
    file.extend(box_of(b"mdat", tile));
    file
}

fn iloc_box(offset: u32, length: u32) -> Vec<u8> {
    let mut body = vec![0x44, 0];
    body.extend(1u16.to_be_bytes());
    body.extend(1u16.to_be_bytes());
    body.extend(0u16.to_be_bytes());
    body.extend(1u16.to_be_bytes());
    body.extend(offset.to_be_bytes());
    body.extend(length.to_be_bytes());
    box_of(b"iloc", &fullbox(&body))
}

fn tkhd() -> Vec<u8> {
    let mut body = vec![0, 0, 0, 3];
    body.extend_from_slice(&[0; 8]);
    body.extend(u32_bytes(1));
    body.extend_from_slice(&[0; 4]);
    body.extend(u32_bytes(2000));
    body.extend_from_slice(&[0; 16]);
    body.extend_from_slice(&[0; 36]);
    body.extend(u32_bytes(2u32 << 16));
    body.extend(u32_bytes(1u32 << 16));
    box_of(b"tkhd", &body)
}

fn mvhd(timescale: u32) -> Vec<u8> {
    let mut body = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    body.extend(timescale.to_be_bytes());
    body.extend(u32_bytes(2000));
    body.extend(u32_bytes(0x0001_0000));
    body.extend(0x0100u16.to_be_bytes());
    body.extend_from_slice(&[0; 10]);
    body.extend_from_slice(&[0; 36]);
    body.extend_from_slice(&[0; 24]);
    body.extend(u32_bytes(2));
    box_of(b"mvhd", &body)
}

fn stsd_visual() -> Vec<u8> {
    let mut entry = vec![0u8; 78];
    entry[7] = 1;
    let mut body = u32_bytes(1);
    body.extend(box_of(b"avc1", &entry));
    fullbox(&body)
}

fn fullbox(payload: &[u8]) -> Vec<u8> {
    let mut body = vec![0, 0, 0, 0];
    body.extend_from_slice(payload);
    body
}

fn box_of(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = u32_bytes(u32::try_from(payload.len() + 8).unwrap());
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out
}

fn u32_bytes(value: u32) -> Vec<u8> {
    value.to_be_bytes().to_vec()
}

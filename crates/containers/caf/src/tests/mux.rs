//! Mux round trips through this crate's demuxer, and mux refusals.

use std::io::Cursor;

use cutlass_codec::{CodecId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{
    Demux, IndexPlace, Mux, MuxError, MuxStatus, NewStream, TrackDuration, TrackId, Trim,
};

use super::build::open;
use crate::{CafMux, InfoEntry, TAG_MONO, TAG_STEREO};

fn stream(codec: CodecId, rate: u32, channels: u16, extradata: Vec<u8>) -> NewStream {
    NewStream {
        params: StreamParams::try_audio(StreamCodec::Known(codec), rate, channels, extradata)
            .expect("params"),
        timescale: Timescale::new(rate),
        language: None,
    }
}

fn packet(bytes: Vec<u8>, pts: i64, duration: i64, rate: u32) -> Packet {
    let time = Timestamp::try_new(Some(pts), Some(pts), Some(duration)).expect("time");
    Packet::try_new(0, bytes, time, Timescale::new(rate), true).expect("packet")
}

fn done(rate: u32, ticks: i64) -> [TrackDuration; 1] {
    [TrackDuration {
        track: TrackId::new(0),
        timescale: Timescale::new(rate),
        ticks,
    }]
}

type Writer = CafMux<Cursor<Vec<u8>>>;

fn writer() -> Writer {
    CafMux::new(Cursor::new(Vec::new()))
}

fn bytes_of(mux: Writer) -> Vec<u8> {
    mux.into_inner().into_inner()
}

#[test]
fn pcm_round_trips_with_its_layout_and_grouping() {
    let mut mux = writer();
    mux.add_stream(stream(CodecId::Pcm, 44_100, 2, vec![1, 0, 0]))
        .expect("add");
    let mut payload = Vec::new();
    let mut pts = 0;
    for chunk in 0..7_u8 {
        let frames = 100 + usize::from(chunk) * 50;
        let bytes: Vec<u8> = (0..frames * 4).map(|i| (i as u8) ^ chunk).collect();
        payload.extend(&bytes);
        let count = i64::try_from(frames).expect("frames");
        mux.write_packet(&packet(bytes, pts, count, 44_100))
            .expect("write");
        pts += count;
    }
    let finished = mux.finish(&done(44_100, pts)).expect("finish");
    assert_eq!(finished.index(), IndexPlace::Absent);
    assert_eq!(
        finished.payload_bytes(),
        u64::try_from(payload.len()).expect("len")
    );
    let bytes = bytes_of(mux);
    let mut demux = open(&bytes).expect("open");
    let desc = demux.description();
    assert_eq!(&desc.format, b"lpcm");
    assert_eq!(desc.flags, crate::FLAG_LITTLE_ENDIAN);
    assert_eq!((desc.bytes_per_packet, desc.frames_per_packet), (4, 1));
    assert_eq!((desc.channels, desc.bits_per_channel), (2, 16));
    assert!(!demux.is_open_ended());
    assert!(demux.packet_table().is_none());
    assert_eq!(demux.channel_layout().expect("chan").tag, TAG_STEREO);
    assert_eq!(demux.streams()[0].params().extradata(), [1, 0, 0]);
    let mut joined: Vec<u8> = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        joined.extend(packet.bytes());
    }
    assert_eq!(joined, payload);
}

#[test]
fn every_pcm_coding_writes_its_own_description() {
    for (coding, endian, format, flags, bits, frame) in [
        (1_u8, 1_u8, *b"lpcm", 0, 16, 2_u32),
        (2, 0, *b"lpcm", 2, 24, 3),
        (3, 1, *b"lpcm", 0, 32, 4),
        (4, 0, *b"lpcm", 3, 32, 4),
        (5, 1, *b"lpcm", 1, 64, 8),
        (6, 1, *b"lpcm", 0, 8, 1),
        (9, 0, *b"lpcm", 2, 64, 8),
        (7, 1, *b"alaw", 0, 8, 1),
        (8, 1, *b"ulaw", 0, 8, 1),
    ] {
        let mut mux = writer();
        mux.add_stream(stream(CodecId::Pcm, 8_000, 1, vec![coding, endian, 0]))
            .expect("add");
        let bytes = vec![0x55; usize::try_from(frame * 10).expect("len")];
        mux.write_packet(&packet(bytes, 0, 10, 8_000))
            .expect("write");
        mux.finish(&done(8_000, 10)).expect("finish");
        let file = bytes_of(mux);
        let demux = open(&file).expect("open");
        let desc = demux.description();
        assert_eq!(desc.format, format, "{coding}");
        assert_eq!(desc.flags, flags, "{coding}");
        assert_eq!(desc.bits_per_channel, bits, "{coding}");
        assert_eq!(desc.bytes_per_packet, frame, "{coding}");
        assert_eq!(demux.channel_layout().expect("chan").tag, TAG_MONO);
        let params = demux.streams()[0].params();
        assert_eq!(params.codec(), &StreamCodec::Known(CodecId::Pcm));
        assert_eq!(params.extradata()[0], coding, "{coding}");
        if format == *b"lpcm" && bits > 8 {
            assert_eq!(params.extradata()[1], endian, "{coding}");
        }
        assert_eq!(demux.packet_count(), Some(10));
    }
}

fn aac_asc() -> Vec<u8> {
    vec![0x11, 0x90]
}

#[test]
fn aac_round_trips_with_priming_remainder_and_rates() {
    let mut mux = writer();
    mux.add_stream(stream(CodecId::Aac, 48_000, 2, Vec::new()))
        .expect("add");
    mux.set_config(TrackId::new(0), &aac_asc()).expect("config");
    let sizes: Vec<usize> = (0..50).map(|i| 200 + (i * 13) % 150).collect();
    for (index, size) in sizes.iter().enumerate() {
        let pts = i64::try_from(index).expect("index") * 1024;
        mux.write_packet(&packet(vec![index as u8; *size], pts, 1024, 48_000))
            .expect("write");
    }
    mux.set_trim(TrackId::new(0), Trim::new(Some(1024), Some(700)))
        .expect("trim");
    let finished = mux.finish(&done(48_000, 50 * 1024)).expect("finish");
    assert_eq!(finished.index(), IndexPlace::End);
    let bytes = bytes_of(mux);
    let mut demux = open(&bytes).expect("open");
    let header = demux.packet_table().expect("pakt");
    assert_eq!(header.packets, 50);
    assert_eq!(header.priming_frames, 1024);
    assert_eq!(header.remainder_frames, 700);
    assert_eq!(header.valid_frames, 50 * 1024 - 1024 - 700);
    assert_eq!(demux.description().frames_per_packet, 1024);
    assert_eq!(demux.description().bytes_per_packet, 0);
    let stream = demux.streams()[0].clone();
    assert_eq!(stream.params().extradata(), aac_asc());
    assert_eq!(stream.trim(), Trim::new(Some(1024), Some(700)));
    let cookie = demux.cookie().expect("kuki").to_vec();
    // ES_Descriptor header (5), ES_ID and flags (3), DecoderConfig header
    // (5), object and stream type (2): then buffer size (3), max and
    // average bit rates (4 each).
    let buffer = u32::from_be_bytes([0, cookie[15], cookie[16], cookie[17]]);
    assert_eq!(
        buffer,
        u32::try_from(*sizes.iter().max().expect("max")).expect("max")
    );
    let max = u32::from_be_bytes(cookie[18..22].try_into().expect("max"));
    let average = u32::from_be_bytes(cookie[22..26].try_into().expect("average"));
    assert!(max >= average && average > 0, "{max} {average}");
    for (index, size) in sizes.iter().enumerate() {
        let read = demux.read_packet().expect("read").expect("packet");
        assert_eq!(read.bytes().len(), *size);
        assert_eq!(read.bytes()[0], index as u8);
    }
    assert!(demux.read_packet().expect("end").is_none());
}

#[test]
fn a_short_last_packet_sets_the_remainder() {
    // ALAC: 4096 frames a packet, the last one 1000.
    let mut config = vec![0, 0, 0x10, 0, 0, 16, 40, 10, 14, 2, 0, 255];
    config.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xAC, 0x44]);
    let mut mux = writer();
    mux.add_stream(stream(CodecId::Alac, 44_100, 2, config.clone()))
        .expect("add");
    let frames = [4096_i64, 4096, 4096, 1000];
    let mut pts = 0;
    for (index, count) in frames.iter().enumerate() {
        mux.write_packet(&packet(vec![1; 300 + index], pts, *count, 44_100))
            .expect("write");
        pts += count;
    }
    mux.finish(&done(44_100, pts)).expect("finish");
    let bytes = bytes_of(mux);
    let demux = open(&bytes).expect("open");
    assert_eq!(&demux.description().format, b"alac");
    assert_eq!(demux.description().flags, 1);
    assert_eq!(demux.description().frames_per_packet, 4096);
    assert_eq!(demux.cookie(), Some(&config[..]));
    let header = demux.packet_table().expect("pakt");
    assert_eq!((header.priming_frames, header.remainder_frames), (0, 3096));
    assert_eq!(header.valid_frames, u64::try_from(pts).expect("pts"));
    let stream = &demux.streams()[0];
    assert_eq!(stream.params().extradata(), config);
    assert_eq!(stream.edits()[0].duration(), pts);
}

#[test]
fn varying_frame_counts_are_written_as_pairs() {
    let streaminfo = {
        let mut info = vec![0x10, 0, 0x10, 0, 0, 0, 0, 0, 0, 0];
        info.extend([0x0A, 0xC4, 0x42, 0xF0, 0, 0, 0, 0]);
        info.extend([0; 16]);
        info
    };
    let mut mux = writer();
    mux.add_stream(stream(CodecId::Flac, 44_100, 2, Vec::new()))
        .expect("add");
    mux.set_config(TrackId::new(0), &streaminfo)
        .expect("config");
    let frames = [4608_i64, 1152, 4608, 576];
    let mut pts = 0;
    for count in frames {
        mux.write_packet(&packet(vec![9; 40], pts, count, 44_100))
            .expect("write");
        pts += count;
    }
    let mut last = streaminfo.clone();
    last[14] = 0x99;
    mux.set_config(TrackId::new(0), &last)
        .expect("final streaminfo");
    mux.finish(&done(44_100, pts)).expect("finish");
    let bytes = bytes_of(mux);
    let mut demux = open(&bytes).expect("open");
    assert_eq!(demux.description().frames_per_packet, 0);
    assert_eq!(demux.description().flags, 1, "16-bit FLAC");
    assert_eq!(demux.streams()[0].params().extradata(), last);
    assert_eq!(demux.cookie().expect("dfLa").len(), 50);
    let mut seen = Vec::new();
    while let Some(packet) = demux.read_packet().expect("read") {
        seen.push(packet.timestamp().duration().expect("duration"));
    }
    assert_eq!(seen, frames);
}

#[test]
fn opus_writes_no_cookie_and_keeps_the_pre_skip() {
    let mut head = b"OpusHead".to_vec();
    head.extend([1, 2, 0x38, 0x01, 0x80, 0xBB, 0, 0, 0, 0, 0]);
    let mut mux = writer();
    mux.add_stream(stream(CodecId::Opus, 48_000, 2, Vec::new()))
        .expect("add");
    mux.set_config(TrackId::new(0), &head).expect("head");
    let mut pts = 0;
    for duration in [960_i64, 960, 960, 600] {
        mux.write_packet(&packet(vec![3; 80], pts, duration, 48_000))
            .expect("write");
        pts += duration;
    }
    // The encoder reports the padding both ways: as a short last packet and
    // as trailing padding.
    mux.set_trim(TrackId::new(0), Trim::new(Some(312), Some(360)))
        .expect("trim");
    mux.finish(&done(48_000, pts)).expect("finish");
    let bytes = bytes_of(mux);
    let demux = open(&bytes).expect("open");
    assert!(demux.cookie().is_none());
    let header = demux.packet_table().expect("pakt");
    assert_eq!((header.priming_frames, header.remainder_frames), (312, 360));
    assert_eq!(header.valid_frames, 4 * 960 - 312 - 360);
    let extradata = demux.streams()[0].params().extradata();
    assert_eq!(u16::from_le_bytes([extradata[10], extradata[11]]), 312);
}

#[test]
fn info_and_layout_are_written_when_asked() {
    let mut mux = writer();
    let layout = crate::ChannelLayout::tagged(TAG_MONO);
    mux.set_channel_layout(layout.clone()).expect("layout");
    mux.set_info(vec![InfoEntry {
        key: "title".to_owned(),
        value: "Take 3".to_owned(),
    }])
    .expect("info");
    mux.add_stream(stream(CodecId::Pcm, 8_000, 1, vec![1, 1, 0]))
        .expect("add");
    mux.write_packet(&packet(vec![0; 8], 0, 4, 8_000))
        .expect("write");
    assert!(mux.set_info(Vec::new()).is_err(), "header already written");
    mux.finish(&done(8_000, 4)).expect("finish");
    let bytes = bytes_of(mux);
    let demux = open(&bytes).expect("open");
    assert_eq!(demux.channel_layout(), Some(&layout));
    assert_eq!(demux.info()[0].value, "Take 3");
}

#[test]
fn an_empty_track_still_makes_a_file() {
    let mut mux = writer();
    mux.add_stream(stream(CodecId::Pcm, 48_000, 2, vec![1, 0, 0]))
        .expect("add");
    mux.finish(&done(48_000, 0)).expect("finish");
    let bytes = bytes_of(mux);
    let mut demux = open(&bytes).expect("open");
    assert_eq!(demux.packet_count(), Some(0));
    assert!(demux.read_packet().expect("read").is_none());
}

#[test]
fn bad_streams_are_refused() {
    let unknown = NewStream {
        params: StreamParams::try_audio(
            StreamCodec::Unknown(cutlass_codec::UnknownCodec::new(b"ima4".to_vec()).expect("tag")),
            44_100,
            1,
            Vec::new(),
        )
        .expect("params"),
        timescale: Timescale::new(44_100),
        language: None,
    };
    assert!(matches!(
        writer().add_stream(unknown),
        Err(MuxError::UnknownCodec(_))
    ));
    for bad in [
        stream(CodecId::Pcm, 48_000, 2, vec![1, 0]),
        stream(CodecId::Pcm, 48_000, 2, vec![11, 0, 0]),
        stream(CodecId::Pcm, 48_000, 2, vec![1, 0, 1]),
        stream(CodecId::Opus, 48_000, 3, Vec::new()),
        stream(CodecId::H264, 48_000, 2, Vec::new()),
    ] {
        let mut mux = writer();
        assert!(mux.add_stream(bad).is_err());
        assert_eq!(mux.status(), MuxStatus::Failed);
    }
    let mut mismatch = stream(CodecId::Pcm, 48_000, 2, vec![1, 0, 0]);
    mismatch.timescale = Timescale::new(1_000);
    assert!(writer().add_stream(mismatch).is_err());
    let mut mux = writer();
    mux.add_stream(stream(CodecId::Pcm, 48_000, 2, vec![1, 0, 0]))
        .expect("add");
    assert!(matches!(
        mux.add_stream(stream(CodecId::Pcm, 48_000, 2, vec![1, 0, 0])),
        Err(MuxError::TooManyTracks { .. })
    ));
}

fn pcm_mux() -> Writer {
    let mut mux = writer();
    mux.add_stream(stream(CodecId::Pcm, 48_000, 2, vec![1, 0, 0]))
        .expect("add");
    mux.write_packet(&packet(vec![0; 40], 0, 10, 48_000))
        .expect("write");
    mux
}

#[test]
fn bad_packets_fail_the_mux() {
    let cases: Vec<(Packet, &str)> = vec![
        (packet(vec![0; 40], 20, 10, 48_000), "gap"),
        (packet(vec![0; 40], 5, 10, 48_000), "overlap"),
        (packet(vec![0; 39], 10, 10, 48_000), "partial frame"),
        (packet(vec![0; 40], 10, 9, 48_000), "duration"),
        (packet(vec![0; 40], 10, 10, 44_100), "timescale"),
        (
            Packet::try_new(
                0,
                vec![0; 40],
                Timestamp::try_new(Some(10), Some(10), None).expect("time"),
                Timescale::new(48_000),
                true,
            )
            .expect("packet"),
            "no duration",
        ),
        (
            Packet::try_new(
                1,
                vec![0; 40],
                Timestamp::try_new(Some(10), Some(10), Some(10)).expect("time"),
                Timescale::new(48_000),
                true,
            )
            .expect("packet"),
            "track",
        ),
    ];
    for (bad, name) in cases {
        let mut mux = pcm_mux();
        assert!(mux.write_packet(&bad).is_err(), "{name}");
        assert_eq!(mux.status(), MuxStatus::Failed, "{name}");
        assert!(
            mux.write_packet(&packet(vec![0; 40], 10, 10, 48_000))
                .is_err()
        );
        let failed = mux.finish(&done(48_000, 10)).expect_err("failed");
        assert!(!failed.is_complete());
    }
}

#[test]
fn a_packet_over_the_limit_is_refused_before_it_is_written() {
    let mut limits = cutlass_container::Limits::standard();
    limits.max_packet_bytes = 100;
    let mut mux = CafMux::with_limits(Cursor::new(Vec::new()), limits);
    mux.add_stream(stream(CodecId::Aac, 48_000, 2, aac_asc()))
        .expect("add");
    assert!(matches!(
        mux.write_packet(&packet(vec![0; 101], 0, 1024, 48_000)),
        Err(MuxError::HostileSize { .. })
    ));
}

#[test]
fn finish_checks_the_duration_and_runs_once() {
    let mut mux = pcm_mux();
    assert!(mux.finish(&done(48_000, 11)).is_err());
    let mut mux = pcm_mux();
    assert!(mux.finish(&[]).is_err());
    let mut mux = pcm_mux();
    mux.finish(&done(48_000, 10)).expect("finish");
    assert_eq!(mux.status(), MuxStatus::Finished);
    assert_eq!(
        mux.write_packet(&packet(vec![0; 40], 10, 10, 48_000)),
        Err(MuxError::AlreadyFinished)
    );
    assert!(mux.finish(&done(48_000, 10)).is_err());
    let mut mux = pcm_mux();
    mux.cancel();
    assert_eq!(mux.status(), MuxStatus::Failed);
    assert_eq!(
        mux.finish(&done(48_000, 10))
            .expect_err("cancelled")
            .reason(),
        &MuxError::Cancelled
    );
}

#[test]
fn priming_longer_than_the_audio_is_refused() {
    let mut mux = writer();
    mux.add_stream(stream(CodecId::Aac, 48_000, 2, aac_asc()))
        .expect("add");
    mux.write_packet(&packet(vec![0; 10], 0, 1024, 48_000))
        .expect("write");
    mux.set_trim(TrackId::new(0), Trim::new(Some(2048), None))
        .expect("trim");
    assert!(mux.finish(&done(48_000, 1024)).is_err());
}

#[test]
fn a_config_of_another_length_after_the_header_is_refused() {
    let mut mux = writer();
    mux.add_stream(stream(CodecId::Aac, 48_000, 2, aac_asc()))
        .expect("add");
    mux.write_packet(&packet(vec![0; 10], 0, 1024, 48_000))
        .expect("write");
    mux.set_config(TrackId::new(0), &[0x12, 0x10])
        .expect("same length");
    assert!(
        mux.set_config(TrackId::new(0), &[0x12, 0x10, 0x56, 0xE5, 0])
            .is_err()
    );
    assert!(matches!(
        writer().set_config(TrackId::new(3), &[]),
        Err(MuxError::UnknownTrack { id: 3 })
    ));
}

/// A writer that fails after `left` bytes.
struct Failing {
    inner: Cursor<Vec<u8>>,
    left: usize,
}

impl std::io::Write for Failing {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.left {
            return Err(std::io::Error::other("disk full"));
        }
        self.left -= bytes.len();
        self.inner.write(bytes)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl std::io::Seek for Failing {
    fn seek(&mut self, to: std::io::SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(to)
    }
}

#[test]
fn a_failing_writer_fails_the_mux_without_panic() {
    for left in [0, 10, 60, 200] {
        let mut mux = CafMux::new(Failing {
            inner: Cursor::new(Vec::new()),
            left,
        });
        mux.add_stream(stream(CodecId::Pcm, 48_000, 2, vec![1, 0, 0]))
            .expect("add");
        let wrote = mux.write_packet(&packet(vec![0; 400], 0, 100, 48_000));
        let finished = mux.finish(&done(48_000, 100));
        assert!(wrote.is_err() || finished.is_err(), "{left}");
        assert_eq!(mux.status(), MuxStatus::Failed);
    }
}

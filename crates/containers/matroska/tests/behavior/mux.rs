//! What the muxer writes, read back with the demuxer.

use std::io::Cursor;

use cutlass_codec::{
    CodecId, FrameSize, Packet, StreamCodec, StreamParams, Timescale, Timestamp, UnknownCodec,
};
use cutlass_container::{
    Demux, Landing, Language, Mux, MuxError, MuxStatus, NewStream, SeekTo, TrackDuration, TrackId,
    Trim,
};
use cutlass_matroska::meta::DocType;
use cutlass_matroska::{MatroskaDemux, MatroskaMux, ids};

use crate::build;

type Out = MatroskaMux<Cursor<Vec<u8>>>;

fn video(codec: CodecId, extradata: &[u8], timescale: u32) -> NewStream {
    NewStream {
        params: StreamParams::video(
            StreamCodec::Known(codec),
            FrameSize::try_new(320, 240).expect("size"),
            extradata.to_vec(),
        ),
        timescale: Timescale::new(timescale),
        language: None,
    }
}

fn audio(codec: CodecId, extradata: &[u8], rate: u32) -> NewStream {
    NewStream {
        params: StreamParams::try_audio(StreamCodec::Known(codec), rate, 2, extradata.to_vec())
            .expect("audio"),
        timescale: Timescale::new(rate),
        language: None,
    }
}

fn packet(
    track: TrackId,
    bytes: Vec<u8>,
    pts: i64,
    duration: i64,
    scale: u32,
    key: bool,
) -> Packet {
    let time = Timestamp::try_new(Some(pts), Some(pts), Some(duration)).expect("time");
    Packet::try_new(track.get(), bytes, time, Timescale::new(scale), key).expect("packet")
}

fn read_back(bytes: Vec<u8>) -> MatroskaDemux<Cursor<Vec<u8>>> {
    let length = bytes.len() as u64;
    MatroskaDemux::open(Cursor::new(bytes), length).expect("open")
}

fn all(demux: &mut MatroskaDemux<Cursor<Vec<u8>>>) -> Vec<Packet> {
    let mut out = Vec::new();
    while let Some(packet) = demux.read_packet().expect("packet") {
        out.push(packet);
    }
    out
}

fn finish(mut mux: Out, durations: &[(TrackId, u32, i64)]) -> Vec<u8> {
    let list: Vec<TrackDuration> = durations
        .iter()
        .map(|(track, scale, ticks)| TrackDuration {
            track: *track,
            timescale: Timescale::new(*scale),
            ticks: *ticks,
        })
        .collect();
    mux.finish(&list).expect("finish");
    assert_eq!(mux.status(), MuxStatus::Finished);
    mux.into_writer().expect("writer").into_inner()
}

const AVCC: &[u8] = &[
    1, 0x64, 0, 0x1F, 0xFF, 0xE1, 0, 4, 0x67, 0x64, 0, 0x1F, 1, 0, 2, 0x68, 0xEE,
];
const ASC: &[u8] = &[0x11, 0x90];

/// Two seconds of 25 fps video at 90 kHz, a keyframe each second, and
/// AAC at 48 kHz. Configurations come with the first packets, as from
/// encoders, and audio runs ahead of video.
#[test]
fn a_movie_round_trips_with_configurations_cues_and_interleaving() {
    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    let picture = mux
        .add_stream(video(CodecId::H264, &[], 90_000))
        .expect("video");
    let sound = mux
        .add_stream(audio(CodecId::Aac, &[], 48_000))
        .expect("audio");
    assert_eq!((picture.get(), sound.get()), (1, 2));
    let mut sent = Vec::new();
    // Audio first: 1024-sample frames for the whole two seconds.
    mux.set_config(sound, ASC).expect("asc");
    mux.set_trim(sound, Trim::new(Some(1024), Some(0)))
        .expect("trim");
    let frames = 2 * 48_000 / 1024 + 1;
    for index in 0..frames {
        let bytes = vec![0xA0, u8::try_from(index % 256).expect("byte")];
        let packet = packet(sound, bytes, index * 1024, 1024, 48_000, true);
        mux.write_packet(&packet).expect("audio packet");
        sent.push(packet);
    }
    mux.set_config(picture, AVCC).expect("avcc");
    for index in 0..50_i64 {
        let bytes = vec![0x0B; 100 + usize::try_from(index).expect("i")];
        let packet = packet(picture, bytes, index * 3600, 3600, 90_000, index % 25 == 0);
        mux.write_packet(&packet).expect("video packet");
        sent.push(packet);
    }
    let bytes = finish(
        mux,
        &[(picture, 90_000, 50 * 3600), (sound, 48_000, frames * 1024)],
    );
    let mut demux = read_back(bytes);
    assert_eq!(demux.ebml_header().doc_type, DocType::Matroska);
    let streams = demux.streams().to_vec();
    assert_eq!(
        streams[0].params().codec(),
        &StreamCodec::Known(CodecId::H264)
    );
    assert_eq!(streams[0].params().extradata(), AVCC);
    assert_eq!(
        streams[1].params().codec(),
        &StreamCodec::Known(CodecId::Aac)
    );
    assert_eq!(streams[1].params().extradata(), ASC);
    assert_eq!(streams[1].params().sample_rate(), Some(48_000));
    // CodecDelay: 1024 samples, 21.333 ms, rounds to 21.
    assert_eq!(streams[1].edits()[0].media_time(), Some(21));
    assert!(streams[0].edits().is_empty());
    // Duration: the audio ends last, at 94 frames of 1024 samples.
    let audio_end = (frames * 1024 * 1000 + 24_000) / 48_000;
    assert_eq!(demux.duration_ticks(), Some(audio_end));
    let got = all(&mut demux);
    assert_eq!(got.len(), sent.len());
    // Interleaved: times never go back by more than one tick of rounding.
    for pair in got.windows(2) {
        let earlier = pair[0].timestamp().presentation().expect("pts");
        let later = pair[1].timestamp().presentation().expect("pts");
        assert!(later + 1 >= earlier, "{earlier} then {later}");
    }
    // Each track's packets in order, bytes and keyframes kept, times in ms.
    for track in [picture, sound] {
        let theirs: Vec<&Packet> = sent.iter().filter(|p| p.stream() == track.get()).collect();
        let ours: Vec<&Packet> = got.iter().filter(|p| p.stream() == track.get()).collect();
        assert_eq!(theirs.len(), ours.len());
        for (sent, read) in theirs.iter().zip(&ours) {
            assert_eq!(sent.bytes(), read.bytes());
            assert_eq!(sent.keyframe(), read.keyframe());
            let scale = i64::from(sent.timescale().ticks_per_second());
            let pts = sent.timestamp().presentation().expect("pts");
            let ms = (pts * 1000 * 2 + scale).div_euclid(2 * scale);
            assert_eq!(read.timestamp().presentation(), Some(ms));
        }
    }
    // Video frames are 40 ms each, from DefaultDuration.
    assert_eq!(
        got.iter()
            .find(|p| p.stream() == 1)
            .expect("v")
            .timestamp()
            .duration(),
        Some(40)
    );
    // Cues lead a seek to the second keyframe.
    let landing = demux
        .seek(SeekTo {
            track: 1,
            time: 1_500,
            end: None,
        })
        .expect("seek");
    assert!(landing.index_used);
    assert!(matches!(
        landing.landing,
        Landing::Packet {
            time: 1_000,
            keyframe: true,
            ..
        }
    ));
    let next = demux.read_packet().expect("read").expect("packet");
    assert_eq!(
        (next.stream(), next.timestamp().presentation()),
        (1, Some(1_000))
    );
}

fn opus_head() -> Vec<u8> {
    let mut head = b"OpusHead".to_vec();
    head.extend_from_slice(&[1, 2]);
    head.extend_from_slice(&312_u16.to_le_bytes());
    head.extend_from_slice(&48_000_u32.to_le_bytes());
    head.extend_from_slice(&[0, 0, 0]);
    head
}

#[test]
fn webm_keeps_to_its_codecs_and_writes_opus_delay() {
    let mut mux = MatroskaMux::webm(Cursor::new(Vec::new()));
    assert!(mux.add_stream(video(CodecId::H264, AVCC, 1000)).is_err());
    assert_eq!(mux.status(), MuxStatus::Failed);

    let mut mux = MatroskaMux::webm(Cursor::new(Vec::new()));
    let picture = mux
        .add_stream(video(CodecId::Vp9, &[1, 2, 3], 1000))
        .expect("vp9");
    let mut stream = audio(CodecId::Opus, &[], 48_000);
    stream.language = Some(Language::new("eng").expect("language"));
    let sound = mux.add_stream(stream).expect("opus");
    mux.set_config(sound, &opus_head()).expect("head");
    for index in 0..10 {
        mux.write_packet(&packet(
            picture,
            vec![9; 50],
            index * 40,
            40,
            1000,
            index == 0,
        ))
        .expect("vp9");
        mux.write_packet(&packet(
            sound,
            OPUS_20MS.to_vec(),
            index * 960,
            960,
            48_000,
            true,
        ))
        .expect("opus");
    }
    let bytes = finish(mux, &[(picture, 1000, 400), (sound, 48_000, 9600)]);
    assert!(bytes.windows(4).any(|window| window == b"webm"));
    let demux = read_back(bytes);
    assert_eq!(demux.ebml_header().doc_type, DocType::Webm);
    let streams = demux.streams();
    // VP9 keeps no CodecPrivate in WebM.
    assert!(streams[0].params().extradata().is_empty());
    assert_eq!(streams[1].params().extradata(), opus_head());
    assert_eq!(streams[1].language().map(Language::as_str), Some("eng"));
    // 312 samples at 48 kHz: 6.5 ms, rounded to 7.
    assert_eq!(streams[1].edits()[0].media_time(), Some(7));
    assert_eq!(streams[1].trim().encoder_delay(), Some(312));
    let entry = &demux.track_entries()[1];
    assert_eq!(entry.codec_delay, 6_500_000);
    assert_eq!(entry.seek_pre_roll, 80_000_000);
}

/// An AV1 Block is the temporal unit without its delimiter and padding,
/// every OBU with its size; the CodecPrivate is the `av1C` record.
#[test]
fn av1_blocks_are_stored_without_delimiters_and_keep_the_av1c_record() {
    let av1c = [0x81, 0x00, 0x0c, 0x00, 0x0a, 0x03, 1, 2, 3];
    let obu = |obu_type: u8, payload: &[u8]| {
        let mut out = vec![(obu_type << 3) | 2, payload.len() as u8];
        out.extend_from_slice(payload);
        out
    };
    let unit = |index: u8| {
        let mut out = obu(2, &[]);
        if index == 1 {
            out.extend(obu(15, &[0, 0]));
        }
        out.extend(obu(6, &[index; 30]));
        out
    };
    for webm in [false, true] {
        let mut mux = if webm {
            MatroskaMux::webm(Cursor::new(Vec::new()))
        } else {
            MatroskaMux::new(Cursor::new(Vec::new()))
        };
        let picture = mux.add_stream(video(CodecId::Av1, &[], 1000)).expect("av1");
        mux.set_config(picture, &av1c).expect("record");
        for index in 0..3u8 {
            mux.write_packet(&packet(
                picture,
                unit(index),
                i64::from(index) * 40,
                40,
                1000,
                index == 0,
            ))
            .expect("unit");
        }
        let bytes = finish(mux, &[(picture, 1000, 120)]);
        let mut demux = read_back(bytes);
        let stream = &demux.streams()[0];
        assert_eq!(stream.params().codec(), &StreamCodec::Known(CodecId::Av1));
        assert_eq!(stream.params().extradata(), av1c);
        let packets = all(&mut demux);
        assert_eq!(packets.len(), 3);
        for (index, packet) in packets.iter().enumerate() {
            assert_eq!(packet.bytes(), obu(6, &[index as u8; 30]).as_slice());
        }
    }
    // A packet that is not whole OBUs is refused.
    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    let picture = mux.add_stream(video(CodecId::Av1, &[], 1000)).expect("av1");
    let _ = mux.write_packet(&packet(picture, vec![0x32, 9, 1], 0, 40, 1000, true));
    assert!(mux.finish(&[]).is_err() || mux.status() == MuxStatus::Failed);
}

#[test]
fn prores_frames_lose_their_first_atom_on_disk_and_get_it_back() {
    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    let picture = mux
        .add_stream(video(CodecId::Prores, b"apch", 25))
        .expect("prores");
    let mut frame = 20_u32.to_be_bytes().to_vec();
    frame.extend_from_slice(b"icpf");
    frame.extend_from_slice(&[1; 12]);
    mux.write_packet(&packet(picture, frame.clone(), 0, 1, 25, true))
        .expect("frame");
    let bytes = finish(mux, &[(picture, 25, 1)]);
    assert!(!bytes.windows(4).any(|window| window == b"icpf"));
    let mut demux = read_back(bytes);
    assert_eq!(demux.streams()[0].params().extradata(), b"apch");
    assert_eq!(all(&mut demux)[0].bytes(), frame.as_slice());
}

#[test]
fn trailing_padding_shortens_the_last_block_and_is_written_as_discard_padding() {
    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    let sound = mux
        .add_stream(audio(CodecId::Aac, ASC, 48_000))
        .expect("aac");
    for index in 0..3 {
        if index == 2 {
            mux.set_trim(sound, Trim::new(Some(0), Some(480)))
                .expect("trim");
        }
        mux.write_packet(&packet(
            sound,
            vec![1; 10],
            index * 1024,
            1024,
            48_000,
            true,
        ))
        .expect("packet");
    }
    let bytes = finish(mux, &[(sound, 48_000, 3072)]);
    // DiscardPadding: 480 samples, 10 ms.
    let mut discard = vec![0x75, 0xA2, 0x88];
    discard.extend_from_slice(&10_000_000_u64.to_be_bytes());
    assert!(bytes.windows(discard.len()).any(|window| window == discard));
    // 1024 samples: 21.333 ms, rounded to 21; the last plays 544: 11.33.
    assert_eq!(block_durations(&bytes), vec![11]);
    // Read back, the packet keeps its whole frame and the trim states the
    // padding.
    let mut demux = read_back(bytes);
    assert_eq!(demux.streams()[0].trim().trailing_padding(), Some(480));
    let got = all(&mut demux);
    let durations: Vec<Option<i64>> = got.iter().map(|p| p.timestamp().duration()).collect();
    assert_eq!(durations, vec![Some(21), Some(21), Some(21)]);
}

/// An encoder that states its end padding on each of its last packets, as
/// the AAC encoder does when it drains, means one total: it comes off the
/// last packet first, and the rest off the one before.
#[test]
fn end_padding_stated_on_several_packets_is_one_total() {
    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    let sound = mux
        .add_stream(audio(CodecId::Aac, ASC, 48_000))
        .expect("aac");
    for index in 0..5 {
        let padding = if index >= 3 { 1500 } else { 0 };
        mux.set_trim(sound, Trim::new(Some(1024), Some(padding)))
            .expect("trim");
        mux.write_packet(&packet(
            sound,
            vec![1; 10],
            index * 1024,
            1024,
            48_000,
            true,
        ))
        .expect("packet");
    }
    let bytes = finish(mux, &[(sound, 48_000, 5 * 1024)]);
    // 476 samples off the fourth packet, all 1024 off the fifth.
    assert_eq!(discard_ns(&bytes), vec![9_916_666, 21_333_333]);
    // 1024 samples are 21 ms; 548 are 11.
    assert_eq!(block_durations(&bytes), vec![11, 0]);
    let mut demux = read_back(bytes);
    // The last block's padding is the track's.
    assert_eq!(demux.streams()[0].trim().trailing_padding(), Some(1024));
    let got = all(&mut demux);
    let durations: Vec<Option<i64>> = got.iter().map(|p| p.timestamp().duration()).collect();
    assert_eq!(durations, vec![Some(21); 5]);
}

/// One 20 ms CELT frame: TOC config 31, code 0.
const OPUS_20MS: [u8; 4] = [0xF8, 1, 2, 3];
/// One 2.5 ms CELT frame: TOC config 28, code 0.
const OPUS_2_5MS: [u8; 3] = [0xE0, 1, 2];

/// Every BlockDuration value in the file, in order. The tests' payloads
/// hold no 0x9B octet.
fn block_durations(bytes: &[u8]) -> Vec<u64> {
    let mut out = Vec::new();
    for (at, octet) in bytes.iter().enumerate() {
        let Some(&size) = bytes.get(at + 1) else {
            break;
        };
        if *octet != 0x9B || !(0x80..=0x88).contains(&size) {
            continue;
        }
        let len = usize::from(size & 0x0F);
        let Some(value) = bytes.get(at + 2..at + 2 + len) else {
            continue;
        };
        out.push(
            value
                .iter()
                .fold(0, |sum, byte| sum << 8 | u64::from(*byte)),
        );
    }
    out
}

fn discard_ns(bytes: &[u8]) -> Vec<u64> {
    bytes
        .windows(11)
        .filter(|window| window[..3] == [0x75, 0xA2, 0x88])
        .map(|window| u64::from_be_bytes(window[3..].try_into().expect("eight")))
        .collect()
}

#[test]
fn an_opus_packet_shorter_than_its_toc_has_the_rest_discarded() {
    let mut mux = MatroskaMux::webm(Cursor::new(Vec::new()));
    let sound = mux
        .add_stream(audio(CodecId::Opus, &opus_head(), 48_000))
        .expect("opus");
    for (index, duration) in [960, 960, 336].into_iter().enumerate() {
        let pts = index as i64 * 960;
        mux.write_packet(&packet(
            sound,
            OPUS_20MS.to_vec(),
            pts,
            duration,
            48_000,
            true,
        ))
        .expect("packet");
    }
    let bytes = finish(mux, &[(sound, 48_000, 2256)]);
    // The last packet holds 960 samples and plays 336: 624 are 13 ms.
    assert_eq!(discard_ns(&bytes), vec![13_000_000]);
    let got = all(&mut read_back(bytes));
    let durations: Vec<Option<i64>> = got.iter().map(|p| p.timestamp().duration()).collect();
    assert_eq!(durations, vec![Some(20), Some(20), Some(7)]);

    // Times in whole milliseconds, as from a remuxed file: 2.5 ms frames
    // land 2 or 3 ms apart, and a 2 ms gap is rounding, not padding.
    let mut mux = MatroskaMux::webm(Cursor::new(Vec::new()));
    let stream = NewStream {
        timescale: Timescale::new(1000),
        ..audio(CodecId::Opus, &opus_head(), 48_000)
    };
    let sound = mux.add_stream(stream).expect("opus");
    let mut pts = 0;
    for duration in [3, 2, 3, 2] {
        mux.write_packet(&packet(
            sound,
            OPUS_2_5MS.to_vec(),
            pts,
            duration,
            1000,
            true,
        ))
        .expect("packet");
        pts += duration;
    }
    let bytes = finish(mux, &[(sound, 1000, 10)]);
    assert!(discard_ns(&bytes).is_empty());
}

/// Trim is counted in samples whatever the track's timescale: on a
/// millisecond track, 480 samples of padding take 10 ticks off the last
/// block, and the encoder delay comes back as samples.
#[test]
fn trim_in_samples_fits_a_millisecond_track() {
    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    let stream = NewStream {
        timescale: Timescale::new(1000),
        ..audio(CodecId::Aac, ASC, 48_000)
    };
    let sound = mux.add_stream(stream).expect("aac");
    mux.set_trim(sound, Trim::new(Some(1024), None))
        .expect("delay");
    for index in 0..3 {
        if index == 2 {
            mux.set_trim(sound, Trim::new(Some(1024), Some(480)))
                .expect("trim");
        }
        mux.write_packet(&packet(sound, vec![1; 10], index * 21, 21, 1000, true))
            .expect("packet");
    }
    let bytes = finish(mux, &[(sound, 1000, 63)]);
    assert_eq!(discard_ns(&bytes), vec![10_000_000]);
    let mut demux = read_back(bytes);
    assert_eq!(demux.streams()[0].trim().encoder_delay(), Some(1024));
    assert_eq!(demux.track_entries()[0].codec_delay, 21_333_333);
    assert_eq!(demux.streams()[0].trim().trailing_padding(), Some(480));
    let got = all(&mut demux);
    let durations: Vec<Option<i64>> = got.iter().map(|p| p.timestamp().duration()).collect();
    assert_eq!(durations, vec![Some(21); 3]);
}

#[test]
fn a_flac_streaminfo_that_comes_again_is_patched_and_a_changed_size_is_refused() {
    let mut info = vec![0x10, 0, 0x10, 0, 0, 0, 0, 0, 0, 0, 0x0A, 0xC4, 0x42, 0xF0];
    info.resize(34, 0);
    let mut last = info.clone();
    last[33] = 0x77;
    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    let sound = mux
        .add_stream(audio(CodecId::Flac, &[], 44_100))
        .expect("flac");
    mux.set_config(sound, &info).expect("first");
    mux.write_packet(&packet(sound, vec![0xFF, 0xF8, 1], 0, 4096, 44_100, true))
        .expect("one");
    mux.set_config(sound, &last).expect("final");
    mux.write_packet(&packet(sound, vec![0xFF, 0xF8, 2], 4096, 100, 44_100, true))
        .expect("two");
    let bytes = finish(mux, &[(sound, 44_100, 4196)]);
    let demux = read_back(bytes);
    let private = demux.streams()[0].params().extradata();
    assert_eq!(&private[..4], b"fLaC");
    assert_eq!(&private[8..], last.as_slice());

    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    let picture = mux
        .add_stream(video(CodecId::H264, AVCC, 1000))
        .expect("h264");
    mux.write_packet(&packet(picture, vec![1], 0, 40, 1000, true))
        .expect("frame");
    assert!(mux.set_config(picture, &[1, 2, 3]).is_err());
    assert_eq!(mux.status(), MuxStatus::Failed);
}

#[test]
fn audio_only_files_cue_each_cluster_and_long_gaps_open_new_clusters() {
    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    let sound = mux
        .add_stream(audio(CodecId::Mp3, &[], 48_000))
        .expect("mp3");
    let mut ticks = 0;
    // Twelve seconds of 24 ms frames, then one frame 40 s later.
    let mut pts = 0;
    for _ in 0..500 {
        mux.write_packet(&packet(sound, vec![3; 30], pts, 1152, 48_000, true))
            .expect("frame");
        pts += 1152;
        ticks += 1152;
    }
    pts += 48_000 * 40;
    mux.write_packet(&packet(sound, vec![4; 30], pts, 1152, 48_000, true))
        .expect("late");
    ticks += 1152;
    let bytes = finish(mux, &[(sound, 48_000, ticks)]);
    let mut demux = read_back(bytes);
    let got = all(&mut demux);
    assert_eq!(got.len(), 501);
    let last = got
        .last()
        .expect("last")
        .timestamp()
        .presentation()
        .expect("pts");
    assert_eq!(last, (500 * 1152 + 48_000 * 40) * 1000 / 48_000);
    for target in [0, 3_000, 7_777, 11_000, last] {
        let landing = demux
            .seek(SeekTo {
                track: 1,
                time: target,
                end: None,
            })
            .expect("seek");
        assert!(landing.index_used);
        let Landing::Packet { time, .. } = landing.landing else {
            panic!("end at {target}");
        };
        assert!(
            time <= target && target - time <= 24,
            "{target} landed {time}"
        );
    }
}

#[test]
fn negative_times_sit_before_a_cluster_at_zero() {
    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    let sound = mux.add_stream(audio(CodecId::Mp3, &[], 1000)).expect("mp3");
    mux.write_packet(&packet(sound, vec![1], -30, 30, 1000, true))
        .expect("early");
    mux.write_packet(&packet(sound, vec![2], 0, 30, 1000, true))
        .expect("zero");
    let bytes = finish(mux, &[(sound, 1000, 60)]);
    let got = all(&mut read_back(bytes));
    let times: Vec<Option<i64>> = got.iter().map(|p| p.timestamp().presentation()).collect();
    assert_eq!(times, vec![Some(-30), Some(0)]);
}

#[test]
fn the_mux_contract_refuses_what_it_cannot_write() {
    // Durations must add up, and every track needs one.
    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    let sound = mux.add_stream(audio(CodecId::Mp3, &[], 1000)).expect("mp3");
    mux.write_packet(&packet(sound, vec![1], 0, 30, 1000, true))
        .expect("packet");
    let failed = mux
        .finish(&[TrackDuration {
            track: sound,
            timescale: Timescale::new(1000),
            ticks: 31,
        }])
        .expect_err("mismatch");
    assert_eq!(failed.reason(), &MuxError::DurationMismatch { track: 1 });
    assert!(failed.durations().is_none());
    assert_eq!(mux.status(), MuxStatus::Failed);
    assert!(mux.into_writer().is_err());

    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    mux.add_stream(audio(CodecId::Mp3, &[], 1000)).expect("mp3");
    assert_eq!(
        mux.finish(&[]).expect_err("incomplete").reason(),
        &MuxError::Incomplete
    );

    // Unknown codecs, stills, unknown tracks, timescale mismatches.
    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    let unknown = NewStream {
        params: StreamParams::opaque(
            StreamCodec::Unknown(UnknownCodec::new(b"XYZ".to_vec()).expect("tag")),
            Vec::new(),
        ),
        timescale: Timescale::new(1000),
        language: None,
    };
    assert!(matches!(
        mux.add_stream(unknown),
        Err(MuxError::UnknownCodec(_))
    ));
    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    assert!(mux.add_stream(video(CodecId::Png, &[], 1000)).is_err());
    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    let sound = mux.add_stream(audio(CodecId::Mp3, &[], 1000)).expect("mp3");
    assert_eq!(
        mux.write_packet(&packet(TrackId::new(5), vec![1], 0, 1, 1000, true)),
        Err(MuxError::UnknownTrack { id: 5 })
    );
    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    let sound2 = mux.add_stream(audio(CodecId::Mp3, &[], 1000)).expect("mp3");
    assert!(matches!(
        mux.write_packet(&packet(sound2, vec![1], 0, 1, 48_000, true)),
        Err(MuxError::TimescaleMismatch { .. })
    ));
    let _ = sound;

    // H.264 without its configuration has nothing to put in CodecPrivate.
    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    let picture = mux
        .add_stream(video(CodecId::H264, &[], 1000))
        .expect("h264");
    assert!(
        mux.write_packet(&packet(picture, vec![1], 0, 40, 1000, true))
            .is_err()
    );

    // Cancel, and nothing after finish.
    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    let sound = mux.add_stream(audio(CodecId::Mp3, &[], 1000)).expect("mp3");
    mux.cancel();
    assert_eq!(
        mux.write_packet(&packet(sound, vec![1], 0, 1, 1000, true)),
        Err(MuxError::Cancelled)
    );
    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    let sound = mux.add_stream(audio(CodecId::Mp3, &[], 1000)).expect("mp3");
    mux.write_packet(&packet(sound, vec![1], 0, 1, 1000, true))
        .expect("packet");
    mux.finish(&[TrackDuration {
        track: sound,
        timescale: Timescale::new(1000),
        ticks: 1,
    }])
    .expect("finish");
    assert_eq!(
        mux.write_packet(&packet(sound, vec![1], 1, 1, 1000, true)),
        Err(MuxError::AlreadyFinished)
    );
}

#[test]
fn a_file_with_tracks_and_no_packets_still_opens() {
    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    let sound = mux.add_stream(audio(CodecId::Mp3, &[], 1000)).expect("mp3");
    let bytes = finish(mux, &[(sound, 1000, 0)]);
    let mut demux = read_back(bytes);
    assert_eq!(demux.streams().len(), 1);
    assert!(all(&mut demux).is_empty());
}

/// The top-level elements of a written file's Segment: ID, offset from the
/// Segment's data, and the element's data.
fn segment_children(bytes: &[u8]) -> Vec<(u32, u64, Vec<u8>)> {
    use cutlass_matroska::ebml::{Size, read_id, read_size};

    let element = |at: usize| {
        let (id, id_len) = read_id(&bytes[at..]).expect("id");
        let (size, size_len) = read_size(&bytes[at + id_len..]).expect("size");
        let Size::Known(size) = size else {
            panic!("unknown size at {at}");
        };
        (
            id,
            at + id_len + size_len,
            usize::try_from(size).expect("size"),
        )
    };
    let (_, header_data, header_size) = element(0);
    let (segment, data, size) = element(header_data + header_size);
    assert_eq!(segment, ids::SEGMENT);
    let mut children = Vec::new();
    let mut at = data;
    while at < data + size {
        let (id, body, length) = element(at);
        children.push((id, (at - data) as u64, bytes[body..body + length].to_vec()));
        at = body + length;
    }
    children
}

/// A short JPEG-looking picture: the start-of-image marker, then bytes.
fn fake_jpeg() -> Vec<u8> {
    let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0];
    jpeg.extend((0..300_u32).map(|i| u8::try_from(i * 7 % 251).expect("byte")));
    jpeg.extend([0xFF, 0xD9]);
    jpeg
}

/// One second of Opus in Matroska or WebM, with `cover` set when given.
fn opus_file(mut mux: Out, cover: Option<Vec<u8>>) -> Vec<u8> {
    let sound = mux
        .add_stream(audio(CodecId::Opus, &opus_head(), 48_000))
        .expect("audio");
    if let Some(cover) = cover {
        mux.set_cover_art(cover).expect("cover");
    }
    for index in 0..50_i64 {
        let bytes = vec![0xFC, u8::try_from(index).expect("byte")];
        mux.write_packet(&packet(sound, bytes, index * 960, 960, 48_000, true))
            .expect("packet");
    }
    finish(mux, &[(sound, 48_000, 50 * 960)])
}

/// The cover is Attachments right after Tracks, holding one AttachedFile:
/// FileName `cover.jpg`, FileMediaType `image/jpeg`, FileData the picture
/// and a FileUID that is not zero (RFC 9559, 5.1.6). The SeekHead points at
/// it, and the demuxer reads the same packets as without it.
#[test]
fn a_cover_is_one_attached_file_after_tracks_and_the_packets_read_the_same() {
    let jpeg = fake_jpeg();
    let plain = opus_file(MatroskaMux::new(Cursor::new(Vec::new())), None);
    let bytes = opus_file(
        MatroskaMux::new(Cursor::new(Vec::new())),
        Some(jpeg.clone()),
    );
    let children = segment_children(&bytes);
    let order: Vec<u32> = children.iter().map(|child| child.0).collect();
    assert_eq!(
        &order[..5],
        &[
            ids::SEEK_HEAD,
            ids::VOID,
            ids::INFO,
            ids::TRACKS,
            ids::ATTACHMENTS
        ]
    );
    assert!(
        !segment_children(&plain)
            .iter()
            .any(|child| child.0 == ids::ATTACHMENTS)
    );
    let (_, at, attachments) = &children[4];

    let mut file = Vec::new();
    file.extend(build::bytes(ids::FILE_NAME, b"cover.jpg"));
    file.extend(build::bytes(ids::FILE_MEDIA_TYPE, b"image/jpeg"));
    file.extend(build::bytes(ids::FILE_DATA, &jpeg));
    // The one AttachedFile fills Attachments, and ends with its FileUID.
    let (id, id_len) = cutlass_matroska::ebml::read_id(attachments).expect("id");
    assert_eq!(id, ids::ATTACHED_FILE);
    let (_, size_len) = cutlass_matroska::ebml::read_size(&attachments[id_len..]).expect("size");
    let written = &attachments[id_len + size_len..];
    assert_eq!(&written[..file.len()], file.as_slice());
    let uid = &written[file.len()..];
    assert_eq!(&uid[..2], &[0x46, 0xAE]);
    let width = usize::from(uid[2] & 0x7F);
    assert_eq!(uid.len(), 3 + width);
    assert!(uid[3..].iter().any(|&byte| byte != 0), "FileUID is zero");

    let seeks = cutlass_matroska::meta::parse_seek_head(&children[0].2).expect("seek head");
    let entry = seeks
        .iter()
        .find(|entry| entry.id == ids::ATTACHMENTS)
        .expect("SeekHead names Attachments");
    assert_eq!(entry.position, *at);
    assert!(seeks.iter().any(|entry| entry.id == ids::CUES));

    let mut with = read_back(bytes);
    let mut without = read_back(plain);
    assert_eq!(with.streams(), without.streams());
    assert_eq!(all(&mut with), all(&mut without));
}

/// WebM has no Attachments: a cover given to a WebM file is taken and not
/// written, and the file is byte for byte the one without it.
#[test]
fn a_webm_file_takes_a_cover_and_writes_none() {
    let plain = opus_file(MatroskaMux::webm(Cursor::new(Vec::new())), None);
    let bytes = opus_file(
        MatroskaMux::webm(Cursor::new(Vec::new())),
        Some(fake_jpeg()),
    );
    assert_eq!(bytes, plain);
    assert!(
        !segment_children(&bytes)
            .iter()
            .any(|child| child.0 == ids::ATTACHMENTS)
    );
}

/// A cover that is not a JPEG, one past the limit, and one after the
/// header went out are refused, and the muxer goes on building a file
/// without one.
#[test]
fn a_late_or_wrong_cover_is_refused_and_the_file_goes_on() {
    let mut mux = MatroskaMux::new(Cursor::new(Vec::new()));
    let sound = mux
        .add_stream(audio(CodecId::Opus, &opus_head(), 48_000))
        .expect("audio");
    assert!(mux.set_cover_art(Vec::new()).is_err());
    assert!(mux.set_cover_art(b"\x89PNG\r\n\x1a\n".to_vec()).is_err());
    let mut large = vec![0xFF, 0xD8, 0xFF];
    large.resize(cutlass_matroska::limits::MAX_COVER_BYTES + 1, 0);
    assert!(mux.set_cover_art(large).is_err());
    assert_eq!(mux.status(), MuxStatus::Building);
    mux.write_packet(&packet(sound, vec![0xFC, 0], 0, 960, 48_000, true))
        .expect("packet");
    mux.write_packet(&packet(sound, vec![0xFC, 1], 960, 960, 48_000, true))
        .expect("packet");
    // The header is out once a packet is written.
    assert!(mux.set_cover_art(fake_jpeg()).is_err());
    assert_eq!(mux.status(), MuxStatus::Building);
    let bytes = finish(mux, &[(sound, 48_000, 2 * 960)]);
    assert!(
        !segment_children(&bytes)
            .iter()
            .any(|child| child.0 == ids::ATTACHMENTS)
    );
    assert_eq!(all(&mut read_back(bytes)).len(), 2);
}

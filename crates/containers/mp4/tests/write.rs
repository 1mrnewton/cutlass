//! MP4 files written by the crate and read back, with no external tool.

use std::num::NonZeroU32;

use cutlass_core::{
    AudioCodec, AudioStreamConfig, ColorTags, Packet, Timestamp, VideoCodec, VideoStreamConfig,
};
use cutlass_mp4::{Error, Fourcc, Kind, Movie, MovieFile, ProRes, Rotation};

fn scale(value: u32) -> NonZeroU32 {
    match NonZeroU32::new(value) {
        Some(scale) => scale,
        None => panic!("timescale is non-zero"),
    }
}

fn expect_err<T>(result: Result<T, Error>, expected: Error) {
    match result {
        Err(error) if error == expected => {}
        Err(error) => panic!("expected {expected}, got {error}"),
        Ok(_) => panic!("expected {expected}"),
    }
}

fn aac(rate: u32, channels: u16, config: &[u8]) -> AudioStreamConfig {
    AudioStreamConfig {
        codec: AudioCodec::Aac,
        sample_rate: rate,
        channels,
        config_record: config.to_vec(),
    }
}

fn h264(width: u32, height: u32) -> VideoStreamConfig {
    VideoStreamConfig {
        codec: VideoCodec::H264,
        width,
        height,
        config_record: vec![1, 0x42, 0x00],
    }
}

fn packet(
    ticks: NonZeroU32,
    dts: i64,
    pts: i64,
    duration: i64,
    data: &[u8],
    keyframe: bool,
) -> Packet {
    Packet {
        data: data.to_vec(),
        pts: Timestamp::new(pts, ticks),
        dts: Some(Timestamp::new(dts, ticks)),
        duration: Some(Timestamp::new(duration, ticks)),
        keyframe,
    }
}

fn write_video(
    movie_scale: u32,
    track_scale: u32,
    config: VideoStreamConfig,
    samples: Vec<Packet>,
) -> Result<Vec<u8>, Error> {
    let mut file = MovieFile::new(scale(movie_scale));
    file.add_video(scale(track_scale), config, samples)?;
    file.to_bytes()
}

#[test]
fn an_h264_track_round_trips() {
    let ticks = scale(1_000);
    let file = write_video(
        1_000,
        1_000,
        h264(320, 180),
        vec![
            packet(ticks, 0, 0, 100, &[1, 2, 3], true),
            packet(ticks, 100, 100, 100, &[4], true),
        ],
    )
    .expect("h264");
    assert_eq!(&file[..8], &[0, 0, 0, 28, b'f', b't', b'y', b'p']);
    assert_eq!(&file[8..12], b"isom");
    let mdat = file
        .windows(4)
        .position(|bytes| bytes == b"mdat")
        .expect("mdat");
    let moov = file
        .windows(4)
        .position(|bytes| bytes == b"moov")
        .expect("moov");
    assert!(mdat < moov);
    assert!(file.windows(4).all(|bytes| bytes != b"ctts"));
    assert!(file.windows(4).all(|bytes| bytes != b"stss"));
    let movie = Movie::parse(&file).expect("read");
    assert_eq!(movie.timescale(), ticks);
    assert_eq!(movie.duration(), 200);
    let track = &movie.tracks()[0];
    assert_eq!(track.id(), 1);
    assert!(track.enabled());
    assert_eq!(track.timescale(), ticks);
    assert_eq!(track.duration(), 200);
    assert_eq!(track.rotation(), Rotation::Zero);
    assert_eq!((track.display_width(), track.display_height()), (320, 180));
    assert!(track.edits().is_empty());
    assert!(track.color().is_none());
    assert!(track.mastering().is_none());
    assert!(track.content_light().is_none());
    assert_eq!(track.frame_duration(), Some(100));
    let Kind::Video(config) = track.kind() else {
        panic!("video track");
    };
    assert_eq!(config.codec, VideoCodec::H264);
    assert_eq!((config.width, config.height), (320, 180));
    assert_eq!(config.config_record, [1, 0x42, 0x00]);
    let samples = track.samples();
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].data, [1, 2, 3]);
    assert_eq!(samples[0].pts.value, 0);
    assert_eq!(samples[0].dts.map(|time| time.value), Some(0));
    assert_eq!(samples[0].duration.map(|time| time.value), Some(100));
    assert!(samples[0].keyframe);
    assert_eq!(samples[1].data, [4]);
    assert_eq!(samples[1].pts.value, 100);
    assert_eq!(samples[1].dts.map(|time| time.value), Some(100));
}

#[test]
fn a_video_track_keeps_its_colour_tags() {
    let ticks = scale(1_000);
    let tags = ColorTags {
        primaries: 9,
        transfer: 18,
        matrix: 9,
        full_range: Some(false),
    };
    let mut file = MovieFile::new(ticks);
    file.add_video_color(
        ticks,
        h264(320, 180),
        tags,
        vec![packet(ticks, 0, 0, 100, &[1], true)],
    )
    .expect("write");
    let movie = Movie::parse(&file.to_bytes().expect("bytes")).expect("read");
    assert_eq!(movie.tracks()[0].color(), Some(tags));
}

#[test]
fn a_video_track_can_leave_the_range_out() {
    let ticks = scale(1_000);
    let tags = ColorTags {
        primaries: 1,
        transfer: 1,
        matrix: 1,
        full_range: None,
    };
    let mut file = MovieFile::new(ticks);
    file.add_video_color(
        ticks,
        h264(64, 64),
        tags,
        vec![packet(ticks, 0, 0, 100, &[1], true)],
    )
    .expect("write");
    let movie = Movie::parse(&file.to_bytes().expect("bytes")).expect("read");
    assert_eq!(movie.tracks()[0].color(), Some(tags));
}

#[test]
fn an_hevc_track_round_trips() {
    let ticks = scale(1_000);
    let mut config = h264(64, 64);
    config.codec = VideoCodec::Hevc;
    config.config_record = vec![1, 2, 3, 4];
    let file = write_video(
        1_000,
        1_000,
        config,
        vec![packet(ticks, 0, 0, 40, &[9], true)],
    )
    .expect("hevc");
    let movie = Movie::parse(&file).expect("read");
    let Kind::Video(stored) = movie.tracks()[0].kind() else {
        panic!("video track");
    };
    assert_eq!(stored.codec, VideoCodec::Hevc);
    assert_eq!((stored.width, stored.height), (64, 64));
    assert_eq!(stored.config_record, [1, 2, 3, 4]);
    assert_eq!(movie.tracks()[0].samples()[0].data, [9]);
}

#[test]
fn an_aac_track_round_trips() {
    let ticks = scale(44_100);
    let mut file = MovieFile::new(ticks);
    file.add_audio(
        ticks,
        aac(44_100, 2, &[0x12, 0x10]),
        vec![packet(ticks, 0, 0, 1024, &[5, 6], true)],
    )
    .expect("aac");
    let movie = Movie::parse(&file.to_bytes().expect("bytes")).expect("read");
    let track = &movie.tracks()[0];
    assert_eq!(track.timescale(), ticks);
    assert_eq!(track.duration(), 1024);
    assert_eq!((track.display_width(), track.display_height()), (0, 0));
    let Kind::Audio(audio) = track.kind() else {
        panic!("audio track");
    };
    assert_eq!(audio.codec(), Some(AudioCodec::Aac));
    assert_eq!(audio.sample_entry(), Fourcc::new(*b"mp4a"));
    assert_eq!(audio.sample_rate(), 44_100);
    assert_eq!(audio.channels(), 2);
    assert_eq!(audio.config_record(), [0x12, 0x10]);
    assert_eq!(track.samples()[0].data, [5, 6]);
    assert_eq!(
        track.samples()[0].duration.map(|time| time.value),
        Some(1024)
    );
}

#[test]
fn a_high_sample_rate_uses_version_two() {
    let ticks = scale(96_000);
    let mut file = MovieFile::new(ticks);
    file.add_audio(ticks, aac(96_000, 1, &[0x11, 0x90]), Vec::new())
        .expect("rate");
    let movie = Movie::parse(&file.to_bytes().expect("bytes")).expect("read");
    let Kind::Audio(audio) = movie.tracks()[0].kind() else {
        panic!("audio track");
    };
    assert_eq!(audio.sample_rate(), 96_000);
    assert_eq!(audio.channels(), 1);
    assert_eq!(audio.config_record(), [0x11, 0x90]);
}

#[test]
fn a_long_decoder_config_round_trips() {
    let ticks = scale(48_000);
    let config = vec![0x11; 200];
    let mut file = MovieFile::new(ticks);
    file.add_audio(ticks, aac(48_000, 2, &config), Vec::new())
        .expect("config");
    let movie = Movie::parse(&file.to_bytes().expect("bytes")).expect("read");
    let Kind::Audio(audio) = movie.tracks()[0].kind() else {
        panic!("audio track");
    };
    assert_eq!(audio.config_record(), config);
}

#[test]
fn video_and_audio_share_the_movie() {
    let video_scale = scale(30);
    let audio_scale = scale(44_100);
    let mut file = MovieFile::new(video_scale);
    file.add_video(
        video_scale,
        h264(16, 16),
        vec![packet(video_scale, 0, 0, 30, &[1], true)],
    )
    .expect("video");
    file.add_audio(
        audio_scale,
        aac(44_100, 2, &[0x12, 0x10]),
        vec![packet(audio_scale, 0, 0, 44_100, &[2], true)],
    )
    .expect("audio");
    let movie = Movie::parse(&file.to_bytes().expect("bytes")).expect("read");
    assert_eq!(movie.duration(), 30);
    assert_eq!(movie.tracks().len(), 2);
    assert_eq!(movie.tracks()[0].duration(), 30);
    assert_eq!(movie.tracks()[0].samples()[0].data, [1]);
    assert_eq!(movie.tracks()[1].duration(), 44_100);
    assert_eq!(movie.tracks()[1].timescale(), audio_scale);
    assert_eq!(movie.tracks()[1].samples()[0].data, [2]);
    let Kind::Audio(audio) = movie.tracks()[1].kind() else {
        panic!("audio track");
    };
    assert_eq!(audio.sample_rate(), 44_100);
}

#[test]
fn a_zero_channel_count_refuses() {
    let ticks = scale(44_100);
    let mut file = MovieFile::new(ticks);
    expect_err(
        file.add_audio(ticks, aac(44_100, 0, &[]), Vec::new()),
        Error::ZeroChannels,
    );
}

#[test]
fn a_zero_sample_rate_refuses() {
    let ticks = scale(44_100);
    let mut file = MovieFile::new(ticks);
    expect_err(
        file.add_audio(ticks, aac(0, 2, &[]), Vec::new()),
        Error::BadSampleRate,
    );
}

#[test]
fn another_audio_codec_refuses() {
    let ticks = scale(44_100);
    let mut config = aac(44_100, 2, &[]);
    config.codec = AudioCodec::Mp3;
    let mut file = MovieFile::new(ticks);
    expect_err(
        file.add_audio(ticks, config, Vec::new()),
        Error::Inconsistent("the audio codec is not written"),
    );
}

#[test]
fn a_duration_change_is_kept() {
    let ticks = scale(1_000);
    let file = write_video(
        1_000,
        1_000,
        h264(16, 16),
        vec![
            packet(ticks, 0, 0, 100, &[1], true),
            packet(ticks, 100, 100, 250, &[2], true),
        ],
    )
    .expect("durations");
    let movie = Movie::parse(&file).expect("read");
    let track = &movie.tracks()[0];
    assert_eq!(track.frame_duration(), None);
    assert_eq!(track.duration(), 350);
    let samples = track.samples();
    assert_eq!(samples[0].duration.map(|time| time.value), Some(100));
    assert_eq!(samples[1].duration.map(|time| time.value), Some(250));
    assert_eq!(samples[1].dts.map(|time| time.value), Some(100));
}

#[test]
fn a_composition_offset_is_kept() {
    let ticks = scale(1_000);
    let file = write_video(
        1_000,
        1_000,
        h264(16, 16),
        vec![
            packet(ticks, 0, 10, 10, &[1], true),
            packet(ticks, 10, 10, 10, &[2], false),
        ],
    )
    .expect("offset");
    let movie = Movie::parse(&file).expect("read");
    let samples = movie.tracks()[0].samples();
    assert_eq!(samples[0].pts.value, 10);
    assert_eq!(samples[0].dts.map(|time| time.value), Some(0));
    assert_eq!(samples[1].pts.value, 10);
    assert_eq!(samples[1].dts.map(|time| time.value), Some(10));
}

#[test]
fn a_negative_composition_offset_is_kept() {
    let ticks = scale(1_000);
    let file = write_video(
        1_000,
        1_000,
        h264(16, 16),
        vec![
            packet(ticks, 0, 0, 10, &[1], true),
            packet(ticks, 10, 5, 10, &[2], false),
        ],
    )
    .expect("negative");
    let movie = Movie::parse(&file).expect("read");
    let samples = movie.tracks()[0].samples();
    assert_eq!(samples[1].pts.value, 5);
    assert_eq!(samples[1].dts.map(|time| time.value), Some(10));
}

#[test]
fn keyframes_are_marked() {
    let ticks = scale(1_000);
    let file = write_video(
        1_000,
        1_000,
        h264(16, 16),
        vec![
            packet(ticks, 0, 0, 10, &[1], true),
            packet(ticks, 10, 10, 10, &[2], false),
            packet(ticks, 20, 20, 10, &[3], true),
        ],
    )
    .expect("sync");
    let movie = Movie::parse(&file).expect("read");
    let samples = movie.tracks()[0].samples();
    assert!(samples[0].keyframe);
    assert!(!samples[1].keyframe);
    assert!(samples[2].keyframe);
}

#[test]
fn a_track_with_no_keyframes_marks_none() {
    let ticks = scale(1_000);
    let file = write_video(
        1_000,
        1_000,
        h264(16, 16),
        vec![packet(ticks, 0, 0, 10, &[1], false)],
    )
    .expect("none");
    let movie = Movie::parse(&file).expect("read");
    assert!(!movie.tracks()[0].samples()[0].keyframe);
}

#[test]
fn a_track_with_no_samples_has_no_packets() {
    let file = write_video(1_000, 1_000, h264(16, 16), Vec::new()).expect("empty");
    let movie = Movie::parse(&file).expect("read");
    let track = &movie.tracks()[0];
    assert!(track.samples().is_empty());
    assert_eq!(track.duration(), 0);
    assert_eq!(track.frame_duration(), None);
    assert_eq!(movie.duration(), 0);
}

#[test]
fn two_tracks_keep_their_order() {
    let ticks = scale(1_000);
    let mut file = MovieFile::new(ticks);
    file.add_video(
        ticks,
        h264(320, 180),
        vec![packet(ticks, 0, 0, 10, &[1], true)],
    )
    .expect("first");
    file.add_video(ticks, h264(16, 9), vec![packet(ticks, 0, 0, 5, &[2], true)])
        .expect("second");
    let movie = Movie::parse(&file.to_bytes().expect("bytes")).expect("read");
    assert_eq!(movie.tracks()[0].id(), 1);
    assert_eq!(movie.tracks()[1].id(), 2);
    assert_eq!(movie.tracks()[0].samples()[0].data, [1]);
    assert_eq!(movie.tracks()[1].samples()[0].data, [2]);
    let Kind::Video(second) = movie.tracks()[1].kind() else {
        panic!("video track");
    };
    assert_eq!((second.width, second.height), (16, 9));
    assert_eq!(movie.duration(), 10);
}

#[test]
fn the_movie_timescale_scales_the_header() {
    let track_scale = scale(100);
    let file = write_video(
        200,
        100,
        h264(16, 16),
        vec![packet(track_scale, 0, 0, 50, &[1], true)],
    )
    .expect("scaled");
    let movie = Movie::parse(&file).expect("read");
    assert_eq!(movie.timescale(), scale(200));
    assert_eq!(movie.duration(), 100);
    assert_eq!(movie.tracks()[0].timescale(), track_scale);
    assert_eq!(movie.tracks()[0].duration(), 50);
    assert_eq!(movie.tracks()[0].samples()[0].pts.timescale, track_scale);
}

#[test]
fn a_long_duration_uses_a_wide_header() {
    let ticks = scale(1_000);
    let half = i64::from(1u32 << 31);
    let file = write_video(
        1_000,
        1_000,
        h264(320, 180),
        vec![
            packet(ticks, 0, 0, half, &[1], true),
            packet(ticks, half, half, half, &[2], true),
        ],
    )
    .expect("wide");
    let movie = Movie::parse(&file).expect("read");
    assert_eq!(movie.duration(), 1 << 32);
    let track = &movie.tracks()[0];
    assert_eq!(track.duration(), 1 << 32);
    assert_eq!(track.timescale(), ticks);
    assert_eq!(track.rotation(), Rotation::Zero);
    assert_eq!((track.display_width(), track.display_height()), (320, 180));
    assert_eq!(
        track.samples()[0].duration.map(|time| time.value),
        Some(half)
    );
}

#[test]
fn a_duration_that_does_not_fit_the_movie_clock_refuses() {
    let ticks = scale(3);
    expect_err(
        write_video(2, 3, h264(16, 16), vec![packet(ticks, 0, 0, 1, &[1], true)]),
        Error::Inconsistent("the track duration does not fit the movie timescale"),
    );
}

#[test]
fn decode_time_must_start_at_zero() {
    let ticks = scale(1_000);
    expect_err(
        write_video(
            1_000,
            1_000,
            h264(16, 16),
            vec![packet(ticks, 5, 5, 10, &[1], true)],
        ),
        Error::Inconsistent("decode time does not start at zero"),
    );
}

#[test]
fn a_gap_in_decode_time_refuses() {
    let ticks = scale(1_000);
    expect_err(
        write_video(
            1_000,
            1_000,
            h264(16, 16),
            vec![
                packet(ticks, 0, 0, 10, &[1], true),
                packet(ticks, 20, 20, 10, &[2], true),
            ],
        ),
        Error::Inconsistent("decode time is not contiguous"),
    );
}

#[test]
fn a_missing_decode_time_refuses() {
    let ticks = scale(1_000);
    let sample = Packet {
        data: vec![1],
        pts: Timestamp::new(0, ticks),
        dts: None,
        duration: Some(Timestamp::new(10, ticks)),
        keyframe: true,
    };
    expect_err(
        write_video(1_000, 1_000, h264(16, 16), vec![sample]),
        Error::Inconsistent("a sample has no decode time"),
    );
}

#[test]
fn a_missing_duration_refuses() {
    let ticks = scale(1_000);
    let sample = Packet {
        data: vec![1],
        pts: Timestamp::new(0, ticks),
        dts: Some(Timestamp::new(0, ticks)),
        duration: None,
        keyframe: true,
    };
    expect_err(
        write_video(1_000, 1_000, h264(16, 16), vec![sample]),
        Error::Inconsistent("a sample has no duration"),
    );
}

#[test]
fn a_negative_duration_refuses() {
    let ticks = scale(1_000);
    expect_err(
        write_video(
            1_000,
            1_000,
            h264(16, 16),
            vec![packet(ticks, 0, 0, -1, &[1], true)],
        ),
        Error::Inconsistent("a sample duration is negative"),
    );
}

#[test]
fn a_sample_duration_that_does_not_fit_refuses() {
    let ticks = scale(1_000);
    expect_err(
        write_video(
            1_000,
            1_000,
            h264(16, 16),
            vec![packet(ticks, 0, 0, i64::from(u32::MAX) + 1, &[1], true)],
        ),
        Error::Inconsistent("a sample duration does not fit"),
    );
}

#[test]
fn a_composition_offset_that_does_not_fit_refuses() {
    let ticks = scale(1_000);
    expect_err(
        write_video(
            1_000,
            1_000,
            h264(16, 16),
            vec![packet(ticks, 0, i64::from(i32::MAX) + 1, 1, &[1], true)],
        ),
        Error::Inconsistent("the composition offset does not fit"),
    );
}

#[test]
fn a_zero_dimension_refuses() {
    expect_err(
        write_video(1_000, 1_000, h264(0, 16), Vec::new()),
        Error::ZeroDimension,
    );
}

#[test]
fn a_size_that_does_not_fit_refuses() {
    expect_err(
        write_video(1_000, 1_000, h264(65_536, 1), Vec::new()),
        Error::Inconsistent("the video size does not fit"),
    );
}

#[test]
fn another_video_codec_refuses() {
    let mut config = h264(16, 16);
    config.codec = VideoCodec::Vp9;
    expect_err(
        write_video(1_000, 1_000, config, Vec::new()),
        Error::Inconsistent("the video codec is not written"),
    );
}

#[test]
fn an_empty_movie_refuses() {
    expect_err(
        MovieFile::new(scale(1_000)).to_bytes(),
        Error::Inconsistent("the movie has no tracks"),
    );
    expect_err(
        MovieFile::new(scale(1_000)).to_bytes_index_first(),
        Error::Inconsistent("the movie has no tracks"),
    );
}

#[test]
fn the_index_can_come_first() {
    let ticks = scale(1_000);
    let mut file = MovieFile::new(ticks);
    file.add_video(
        ticks,
        h264(32, 18),
        vec![
            packet(ticks, 0, 10, 10, &[1, 2, 3], true),
            packet(ticks, 10, 10, 10, &[4], false),
        ],
    )
    .expect("video");
    file.add_audio(
        ticks,
        aac(44_100, 2, &[0x12, 0x10]),
        vec![packet(ticks, 0, 0, 20, &[5, 6, 7, 8], true)],
    )
    .expect("audio");
    let last = file.to_bytes().expect("after the samples");
    let first = file.to_bytes_index_first().expect("index first");
    assert_ne!(first, last);
    assert_eq!(&first[..8], &[0, 0, 0, 28, b'f', b't', b'y', b'p']);
    let moov = first
        .windows(4)
        .position(|bytes| bytes == b"moov")
        .expect("moov");
    let mdat = first
        .windows(4)
        .position(|bytes| bytes == b"mdat")
        .expect("mdat");
    assert!(moov < mdat);
    assert_eq!(
        Movie::parse(&first).expect("read first"),
        Movie::parse(&last).expect("read last")
    );
}

#[test]
fn a_prores_track_round_trips() {
    let ticks = scale(1_000);
    let mut file = MovieFile::new(ticks);
    file.add_prores(
        ticks,
        ProRes::Standard,
        320,
        180,
        vec![
            packet(ticks, 0, 0, 100, &[9, 9], true),
            packet(ticks, 100, 110, 100, &[8], true),
        ],
    )
    .expect("prores");
    let last = file.to_bytes().expect("after the samples");
    let first = file.to_bytes_index_first().expect("index first");
    assert_ne!(first, last);
    assert_eq!(&last[..20], b"\0\0\0\x14ftypqt  \0\0\0\0qt  ");
    assert_eq!(&first[..20], b"\0\0\0\x14ftypqt  \0\0\0\0qt  ");
    let moov = first
        .windows(4)
        .position(|bytes| bytes == b"moov")
        .expect("moov");
    let mdat = first
        .windows(4)
        .position(|bytes| bytes == b"mdat")
        .expect("mdat");
    assert!(moov < mdat);
    assert_eq!(
        Movie::parse(&first).expect("read first"),
        Movie::parse(&last).expect("read last")
    );
    let movie = Movie::parse(&last).expect("read");
    let track = &movie.tracks()[0];
    let Kind::Video(config) = track.kind() else {
        panic!("video track");
    };
    assert_eq!(config.codec, VideoCodec::ProRes);
    assert!(config.config_record.is_empty());
    assert_eq!((config.width, config.height), (320, 180));
    assert_eq!(track.samples()[1].pts.value, 110);
    assert_eq!(track.samples()[1].data, [8]);
    assert_eq!(track.samples()[1].dts.map(|time| time.value), Some(100));
}

#[test]
fn each_prores_profile_keeps_its_sample_entry() {
    let ticks = scale(1_000);
    let cases = [
        (ProRes::Hq, *b"apch", 24u16, "Apple ProRes 422 HQ"),
        (ProRes::Standard, *b"apcn", 24, "Apple ProRes 422"),
        (ProRes::Lt, *b"apcs", 24, "Apple ProRes 422 LT"),
        (ProRes::Proxy, *b"apco", 24, "Apple ProRes 422 Proxy"),
        (ProRes::Yuv4444, *b"ap4h", 32, "Apple ProRes 4444"),
        (ProRes::Xq, *b"ap4x", 32, "Apple ProRes 4444 XQ"),
    ];
    for (profile, entry, depth, name) in cases {
        let mut file = MovieFile::new(ticks);
        file.add_prores(
            ticks,
            profile,
            16,
            16,
            vec![packet(ticks, 0, 0, 10, &[9], true)],
        )
        .expect("prores");
        let bytes = file.to_bytes().expect("write");
        let at = bytes
            .windows(4)
            .position(|window| window == entry)
            .expect("entry");
        assert_eq!(&bytes[at + 16..at + 20], b"appl");
        assert_eq!(bytes[at + 46], u8::try_from(name.len()).expect("name"));
        assert_eq!(&bytes[at + 47..at + 47 + name.len()], name.as_bytes());
        assert_eq!(&bytes[at + 78..at + 80], &depth.to_be_bytes());
        let movie = Movie::parse(&bytes).expect("read");
        let Kind::Video(config) = movie.tracks()[0].kind() else {
            panic!("video track");
        };
        assert_eq!(config.codec, VideoCodec::ProRes);
        assert!(config.config_record.is_empty());
    }
}

#[test]
fn prores_on_the_h264_entry_refuses() {
    let ticks = scale(1_000);
    let mut config = h264(16, 16);
    config.codec = VideoCodec::ProRes;
    expect_err(
        write_video(
            1_000,
            1_000,
            config,
            vec![packet(ticks, 0, 0, 10, &[1], true)],
        ),
        Error::Inconsistent("the video codec is not written"),
    );
}

#[test]
fn a_prores_track_makes_the_file_quicktime() {
    let ticks = scale(1_000);
    let mut file = MovieFile::new(ticks);
    file.add_video(
        ticks,
        h264(16, 16),
        vec![packet(ticks, 0, 0, 10, &[1], true)],
    )
    .expect("h264");
    file.add_prores(
        ticks,
        ProRes::Hq,
        16,
        16,
        vec![packet(ticks, 0, 0, 10, &[2], true)],
    )
    .expect("prores");
    let bytes = file.to_bytes().expect("write");
    assert_eq!(&bytes[8..12], b"qt  ");
    let movie = Movie::parse(&bytes).expect("read");
    let Kind::Video(first) = movie.tracks()[0].kind() else {
        panic!("video track");
    };
    let Kind::Video(second) = movie.tracks()[1].kind() else {
        panic!("video track");
    };
    assert_eq!(first.codec, VideoCodec::H264);
    assert_eq!(second.codec, VideoCodec::ProRes);
    assert!(second.config_record.is_empty());
}

#[test]
fn a_zero_prores_size_refuses() {
    let ticks = scale(1_000);
    let mut file = MovieFile::new(ticks);
    expect_err(
        file.add_prores(ticks, ProRes::Lt, 0, 16, vec![]),
        Error::ZeroDimension,
    );
}

#[test]
fn a_sample_timescale_must_match_the_track() {
    let ticks = scale(1);
    expect_err(
        write_video(
            1_000,
            1_000,
            h264(16, 16),
            vec![packet(ticks, 0, 0, 10, &[1], true)],
        ),
        Error::Inconsistent("the sample timescale does not match the track"),
    );
}

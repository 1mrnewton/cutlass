//! Hard-coded demo project used while the import / persistence layers are
//! still under construction. Builds a `models::Project` so the UI can render
//! something meaningful on first boot. Delete once we can load a real file.
//!
//! IDs are fresh UUIDs per run — fine for an in-memory demo. Once we
//! persist projects, IDs become stable across runs.

use std::path::PathBuf;

use models::{
    AudioStreamInfo, Clip, ClipId, Color, MediaId, MediaKind, MediaSource, Project, ProjectId,
    Rational, RationalTime, SchemaVersion, Sequence, SequenceId, Track, TrackId, TrackKind,
    VideoStreamInfo,
};

const TIMEBASE: u32 = 90_000;

#[inline]
fn rt(num: i64) -> RationalTime {
    RationalTime::new_raw(num, TIMEBASE)
}

pub fn build_demo_project() -> Project {
    let media_intro = MediaId::new();
    let media_broll = MediaId::new();
    let media_music = MediaId::new();

    let track_v1 = TrackId::new();
    let track_a1 = TrackId::new();

    let media_bin = vec![
        MediaSource {
            id: media_intro,
            name: "intro.mp4".into(),
            path: PathBuf::from("/Users/demo/Videos/intro.mp4"),
            kind: MediaKind::Video,
            has_video: true,
            has_audio: true,
            duration: rt(900_000),
            video: Some(VideoStreamInfo {
                width: 1920,
                height: 1080,
                fps: Rational::new_raw(30_000, 1_001),
                codec: "h264".into(),
            }),
            audio: Some(AudioStreamInfo {
                sample_rate: 48_000,
                codec: "aac".into(),
            }),
            is_supported: true,
            is_loading: false,
            is_missing: false,
            error: None,
        },
        MediaSource {
            id: media_broll,
            name: "broll.mp4".into(),
            path: PathBuf::from("/Users/demo/Videos/broll.mp4"),
            kind: MediaKind::Video,
            has_video: true,
            has_audio: false,
            duration: rt(1_350_000),
            video: Some(VideoStreamInfo {
                width: 3840,
                height: 2160,
                fps: Rational::new_raw(24, 1),
                codec: "hevc".into(),
            }),
            audio: None,
            is_supported: true,
            is_loading: false,
            is_missing: false,
            error: None,
        },
        MediaSource {
            id: media_music,
            name: "background-music.wav".into(),
            path: PathBuf::from("/Users/demo/Audio/background-music.wav"),
            kind: MediaKind::Audio,
            has_video: false,
            has_audio: true,
            duration: rt(2_700_000),
            video: None,
            audio: Some(AudioStreamInfo {
                sample_rate: 48_000,
                codec: "pcm_s16le".into(),
            }),
            is_supported: true,
            is_loading: false,
            is_missing: false,
            error: None,
        },
    ];

    let tracks = vec![
        Track {
            id: track_v1,
            name: "V1".into(),
            kind: TrackKind::Video,
            height_px: 72,
            muted: false,
            solo: false,
            locked: false,
            visible: true,
            clips: vec![
                Clip {
                    id: ClipId::new(),
                    media_id: Some(media_intro),
                    track_id: track_v1,
                    name: "Intro".into(),
                    start: rt(0),
                    duration: rt(270_000),
                    source_in: rt(0),
                    source_out: rt(270_000),
                    speed: Rational::ONE,
                    opacity: 1.0,
                    volume: 1.0,
                    enabled: true,
                    color: Color::rgb(70, 130, 180),
                },
                Clip {
                    id: ClipId::new(),
                    media_id: Some(media_broll),
                    track_id: track_v1,
                    name: "B-Roll".into(),
                    start: rt(270_000),
                    duration: rt(450_000),
                    source_in: rt(90_000),
                    source_out: rt(540_000),
                    speed: Rational::ONE,
                    opacity: 1.0,
                    volume: 1.0,
                    enabled: true,
                    color: Color::rgb(100, 100, 157),
                },
            ],
        },
        Track {
            id: track_a1,
            name: "A1".into(),
            kind: TrackKind::Audio,
            height_px: 48,
            muted: false,
            solo: false,
            locked: false,
            visible: true,
            clips: vec![Clip {
                id: ClipId::new(),
                media_id: Some(media_music),
                track_id: track_a1,
                name: "Music".into(),
                start: rt(0),
                duration: rt(720_000),
                source_in: rt(0),
                source_out: rt(720_000),
                speed: Rational::ONE,
                opacity: 1.0,
                volume: 0.8,
                enabled: true,
                color: Color::rgb(60, 120, 90),
            }],
        },
    ];

    let sequence = Sequence {
        id: SequenceId::new(),
        name: "Main Sequence".into(),
        width: 1920,
        height: 1080,
        fps: Rational::new_raw(30_000, 1_001),
        sample_rate: 48_000,
        timebase: TIMEBASE,
        duration: rt(720_000),
        in_point: Some(rt(0)),
        out_point: Some(rt(720_000)),
        tracks,
    };

    Project {
        id: ProjectId::new(),
        name: "Demo Project".into(),
        file_path: Some(PathBuf::from("/Users/demo/Projects/demo.cutlass")),
        schema: SchemaVersion::CURRENT,
        sequence,
        media_bin,
        is_dirty: false,
    }
}

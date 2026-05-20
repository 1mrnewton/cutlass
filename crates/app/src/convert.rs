//! Bridge between domain types (`models::*`) and Slint DTOs (`ui::*`).
//!
//! Domain → DTO is infallible and defaults the UI-ephemeral fields
//! (selection, thumbnails, playhead/zoom/scroll) to sensible blanks.
//! DTO → domain is fallible (`TryFrom`) because IDs and time numerators
//! are stringly typed on the Slint side and must be parsed.
//!
//! Sentinel conventions (one-to-one with the doc on `project.slint`):
//!
//! | DTO field                              | Domain                  |
//! |----------------------------------------|-------------------------|
//! | `media_id == ""`                       | `None`                  |
//! | `file_path == ""`                      | `None`                  |
//! | `error == ""`                          | `None`                  |
//! | `has_in_point == false`                | `in_point = None`       |
//! | `has_out_point == false`               | `out_point = None`      |
//! | `has_video == false`                   | `video = None`          |
//! | `has_audio == false`                   | `audio = None`          |

use std::path::PathBuf;
use std::rc::Rc;
use std::str::FromStr;

use models::{
    AudioStreamInfo, Clip, Color, MediaKind, MediaSource, ModelParseError, Project, Rational,
    RationalTime, SchemaVersion, Sequence, Track, TrackKind, VideoStreamInfo,
};
use slint::{Model, ModelRc, SharedString, VecModel};

use crate::ui;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn vec_model<T: Clone + 'static>(items: Vec<T>) -> ModelRc<T> {
    ModelRc::from(Rc::new(VecModel::from(items)))
}

fn ss(s: impl Into<String>) -> SharedString {
    s.into().into()
}

fn opt_string_to_sentinel(s: &Option<String>) -> SharedString {
    s.as_deref().unwrap_or("").into()
}

fn sentinel_to_opt_string(s: &SharedString) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

fn opt_path_to_sentinel(p: &Option<PathBuf>) -> SharedString {
    match p {
        Some(p) => p.to_string_lossy().into_owned().into(),
        None => SharedString::default(),
    }
}

fn sentinel_to_opt_path(s: &SharedString) -> Option<PathBuf> {
    if s.is_empty() {
        None
    } else {
        Some(PathBuf::from(s.as_str()))
    }
}

fn parse_id<T>(field: &'static str, s: &SharedString) -> Result<T, ModelParseError>
where
    T: FromStr<Err = uuid::Error>,
{
    T::from_str(s).map_err(|source| ModelParseError::BadUuid { field, source })
}

fn parse_opt_id<T>(field: &'static str, s: &SharedString) -> Result<Option<T>, ModelParseError>
where
    T: FromStr<Err = uuid::Error>,
{
    if s.is_empty() {
        Ok(None)
    } else {
        Ok(Some(parse_id(field, s)?))
    }
}

fn parse_i64(field: &'static str, s: &SharedString) -> Result<i64, ModelParseError> {
    s.parse::<i64>().map_err(|source| ModelParseError::BadInt {
        field,
        value: s.to_string(),
        source,
    })
}

fn parse_u32(field: &'static str, s: &SharedString) -> Result<u32, ModelParseError> {
    s.parse::<u32>().map_err(|source| ModelParseError::BadInt {
        field,
        value: s.to_string(),
        source,
    })
}

// ---------------------------------------------------------------------------
// Rationals
// ---------------------------------------------------------------------------

impl From<&RationalTime> for ui::RationalTime {
    fn from(r: &RationalTime) -> Self {
        ui::RationalTime {
            num: ss(r.num.to_string()),
            den: ss(r.den.to_string()),
        }
    }
}

impl TryFrom<&ui::RationalTime> for RationalTime {
    type Error = ModelParseError;
    fn try_from(d: &ui::RationalTime) -> Result<Self, Self::Error> {
        let num = parse_i64("RationalTime.num", &d.num)?;
        let den = parse_u32("RationalTime.den", &d.den)?;
        RationalTime::new(num, den).ok_or(ModelParseError::BadDenominator {
            field: "RationalTime",
        })
    }
}

impl From<&Rational> for ui::Rational {
    fn from(r: &Rational) -> Self {
        // Slint `int` is i32. `Rational` already fits; clamp den into i32
        // for the trip across (it's a tiny number — fps denominators).
        ui::Rational {
            num: r.num,
            den: r.den.min(i32::MAX as u32) as i32,
        }
    }
}

impl TryFrom<&ui::Rational> for Rational {
    type Error = ModelParseError;
    fn try_from(d: &ui::Rational) -> Result<Self, Self::Error> {
        if d.den <= 0 {
            return Err(ModelParseError::BadDenominator { field: "Rational" });
        }
        Ok(Rational {
            num: d.num,
            den: d.den as u32,
        })
    }
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

impl From<MediaKind> for ui::MediaKind {
    fn from(k: MediaKind) -> Self {
        match k {
            MediaKind::Video => ui::MediaKind::Video,
            MediaKind::Audio => ui::MediaKind::Audio,
            MediaKind::Image => ui::MediaKind::Image,
        }
    }
}

impl From<ui::MediaKind> for MediaKind {
    fn from(k: ui::MediaKind) -> Self {
        match k {
            ui::MediaKind::Video => MediaKind::Video,
            ui::MediaKind::Audio => MediaKind::Audio,
            ui::MediaKind::Image => MediaKind::Image,
        }
    }
}

impl From<TrackKind> for ui::TrackKind {
    fn from(k: TrackKind) -> Self {
        match k {
            TrackKind::Video => ui::TrackKind::Video,
            TrackKind::Audio => ui::TrackKind::Audio,
        }
    }
}

impl From<ui::TrackKind> for TrackKind {
    fn from(k: ui::TrackKind) -> Self {
        match k {
            ui::TrackKind::Video => TrackKind::Video,
            ui::TrackKind::Audio => TrackKind::Audio,
        }
    }
}

// ---------------------------------------------------------------------------
// SchemaVersion
// ---------------------------------------------------------------------------

impl From<&SchemaVersion> for ui::SchemaVersion {
    fn from(v: &SchemaVersion) -> Self {
        ui::SchemaVersion {
            major: v.major as i32,
            minor: v.minor as i32,
            patch: v.patch as i32,
        }
    }
}

impl From<&ui::SchemaVersion> for SchemaVersion {
    fn from(v: &ui::SchemaVersion) -> Self {
        SchemaVersion {
            major: v.major.max(0) as u32,
            minor: v.minor.max(0) as u32,
            patch: v.patch.max(0) as u32,
        }
    }
}

// ---------------------------------------------------------------------------
// Color
// ---------------------------------------------------------------------------

fn color_to_slint(c: Color) -> slint::Color {
    slint::Color::from_argb_u8(c.a, c.r, c.g, c.b)
}

fn color_from_slint(c: slint::Color) -> Color {
    Color::rgba(c.red(), c.green(), c.blue(), c.alpha())
}

// ---------------------------------------------------------------------------
// MediaSource
// ---------------------------------------------------------------------------

impl From<&MediaSource> for ui::MediaSource {
    fn from(m: &MediaSource) -> Self {
        let (has_video, width, height, fps, video_codec) = match &m.video {
            Some(v) => (
                true,
                v.width as i32,
                v.height as i32,
                ui::Rational::from(&v.fps),
                ss(&v.codec),
            ),
            None => (
                false,
                0,
                0,
                ui::Rational::from(&Rational::ONE),
                SharedString::default(),
            ),
        };
        let (has_audio, sample_rate, audio_codec) = match &m.audio {
            Some(a) => (true, a.sample_rate as i32, ss(&a.codec)),
            None => (false, 0, SharedString::default()),
        };
        ui::MediaSource {
            id: ss(m.id.to_string()),
            name: ss(&m.name),
            path: ss(m.path.to_string_lossy().into_owned()),
            kind: m.kind.into(),
            has_video,
            width,
            height,
            fps,
            video_codec,
            has_audio,
            sample_rate,
            audio_codec,
            duration: (&m.duration).into(),
            is_supported: m.is_supported,
            is_loading: m.is_loading,
            is_missing: m.is_missing,
            error: opt_string_to_sentinel(&m.error),
            thumbnail: slint::Image::default(),
        }
    }
}

impl TryFrom<&ui::MediaSource> for MediaSource {
    type Error = ModelParseError;
    fn try_from(d: &ui::MediaSource) -> Result<Self, Self::Error> {
        let video = if d.has_video {
            Some(VideoStreamInfo {
                width: d.width.max(0) as u32,
                height: d.height.max(0) as u32,
                fps: (&d.fps).try_into()?,
                codec: d.video_codec.to_string(),
            })
        } else {
            None
        };
        let audio = if d.has_audio {
            Some(AudioStreamInfo {
                sample_rate: d.sample_rate.max(0) as u32,
                codec: d.audio_codec.to_string(),
            })
        } else {
            None
        };
        Ok(MediaSource {
            id: parse_id("MediaSource.id", &d.id)?,
            name: d.name.to_string(),
            path: PathBuf::from(d.path.as_str()),
            kind: d.kind.into(),
            has_video: d.has_video,
            has_audio: d.has_audio,
            duration: (&d.duration).try_into()?,
            video,
            audio,
            is_supported: d.is_supported,
            is_loading: d.is_loading,
            is_missing: d.is_missing,
            error: sentinel_to_opt_string(&d.error),
        })
    }
}

// ---------------------------------------------------------------------------
// Clip
// ---------------------------------------------------------------------------

/// Lossy-but-precise-enough seconds projection. f32 precision near 4h of
/// timeline is ~1ms, well sub-frame at 60fps. If we ever need more (long-form
/// docs, multi-day projects), promote to f64 or pre-multiply by zoom in Rust.
#[inline]
fn rt_sec_f32(r: RationalTime) -> f32 {
    r.num as f32 / r.den as f32
}

impl From<&Clip> for ui::Clip {
    fn from(c: &Clip) -> Self {
        ui::Clip {
            id: ss(c.id.to_string()),
            media_id: c
                .media_id
                .as_ref()
                .map(|m| ss(m.to_string()))
                .unwrap_or_default(),
            track_id: ss(c.track_id.to_string()),
            name: ss(&c.name),
            start: (&c.start).into(),
            duration: (&c.duration).into(),
            source_in: (&c.source_in).into(),
            source_out: (&c.source_out).into(),
            speed: (&c.speed).into(),
            opacity: c.opacity,
            volume: c.volume,
            enabled: c.enabled,
            color: color_to_slint(c.color),
            start_sec: rt_sec_f32(c.start),
            duration_sec: rt_sec_f32(c.duration),
            source_in_sec: rt_sec_f32(c.source_in),
            source_out_sec: rt_sec_f32(c.source_out),
            selected: false,
        }
    }
}

impl TryFrom<&ui::Clip> for Clip {
    type Error = ModelParseError;
    fn try_from(d: &ui::Clip) -> Result<Self, Self::Error> {
        Ok(Clip {
            id: parse_id("Clip.id", &d.id)?,
            media_id: parse_opt_id("Clip.media_id", &d.media_id)?,
            track_id: parse_id("Clip.track_id", &d.track_id)?,
            name: d.name.to_string(),
            start: (&d.start).try_into()?,
            duration: (&d.duration).try_into()?,
            source_in: (&d.source_in).try_into()?,
            source_out: (&d.source_out).try_into()?,
            speed: (&d.speed).try_into()?,
            opacity: d.opacity,
            volume: d.volume,
            enabled: d.enabled,
            color: color_from_slint(d.color),
        })
    }
}

// ---------------------------------------------------------------------------
// Track
// ---------------------------------------------------------------------------

impl From<&Track> for ui::Track {
    fn from(t: &Track) -> Self {
        let clips: Vec<ui::Clip> = t.clips.iter().map(ui::Clip::from).collect();
        ui::Track {
            id: ss(t.id.to_string()),
            name: ss(&t.name),
            kind: t.kind.into(),
            height_px: t.height_px as i32,
            muted: t.muted,
            solo: t.solo,
            locked: t.locked,
            visible: t.visible,
            clips: vec_model(clips),
        }
    }
}

impl TryFrom<&ui::Track> for Track {
    type Error = ModelParseError;
    fn try_from(d: &ui::Track) -> Result<Self, Self::Error> {
        let mut clips = Vec::with_capacity(d.clips.row_count());
        for c in d.clips.iter() {
            clips.push((&c).try_into()?);
        }
        Ok(Track {
            id: parse_id("Track.id", &d.id)?,
            name: d.name.to_string(),
            kind: d.kind.into(),
            height_px: d.height_px.max(0) as u32,
            muted: d.muted,
            solo: d.solo,
            locked: d.locked,
            visible: d.visible,
            clips,
        })
    }
}

// ---------------------------------------------------------------------------
// Sequence
// ---------------------------------------------------------------------------

impl From<&Sequence> for ui::Sequence {
    fn from(s: &Sequence) -> Self {
        let tracks: Vec<ui::Track> = s.tracks.iter().map(ui::Track::from).collect();
        let zero = RationalTime::ZERO;
        ui::Sequence {
            id: ss(s.id.to_string()),
            name: ss(&s.name),
            width: s.width as i32,
            height: s.height as i32,
            fps: (&s.fps).into(),
            sample_rate: s.sample_rate as i32,
            timebase: s.timebase.min(i32::MAX as u32) as i32,
            duration: (&s.duration).into(),

            has_in_point: s.in_point.is_some(),
            in_point: (&s.in_point.unwrap_or(zero)).into(),
            has_out_point: s.out_point.is_some(),
            out_point: (&s.out_point.unwrap_or(zero)).into(),

            tracks: vec_model(tracks),

            duration_sec: rt_sec_f32(s.duration),
            in_point_sec: s.in_point.map(rt_sec_f32).unwrap_or(0.0),
            out_point_sec: s.out_point.map(rt_sec_f32).unwrap_or(0.0),

            // Ephemeral UI state — fresh defaults; real values come from
            // TimelineState / Flickable scroll positions at runtime.
            playhead: (&zero).into(),
            zoom: 50.0,
            scroll: (&zero).into(),
        }
    }
}

impl TryFrom<&ui::Sequence> for Sequence {
    type Error = ModelParseError;
    fn try_from(d: &ui::Sequence) -> Result<Self, Self::Error> {
        let mut tracks = Vec::with_capacity(d.tracks.row_count());
        for t in d.tracks.iter() {
            tracks.push((&t).try_into()?);
        }
        Ok(Sequence {
            id: parse_id("Sequence.id", &d.id)?,
            name: d.name.to_string(),
            width: d.width.max(0) as u32,
            height: d.height.max(0) as u32,
            fps: (&d.fps).try_into()?,
            sample_rate: d.sample_rate.max(0) as u32,
            timebase: d.timebase.max(1) as u32,
            duration: (&d.duration).try_into()?,
            in_point: if d.has_in_point {
                Some((&d.in_point).try_into()?)
            } else {
                None
            },
            out_point: if d.has_out_point {
                Some((&d.out_point).try_into()?)
            } else {
                None
            },
            tracks,
        })
    }
}

// ---------------------------------------------------------------------------
// Project
// ---------------------------------------------------------------------------

impl From<&Project> for ui::Project {
    fn from(p: &Project) -> Self {
        let media_bin: Vec<ui::MediaSource> = p.media_bin.iter().map(ui::MediaSource::from).collect();
        ui::Project {
            id: ss(p.id.to_string()),
            name: ss(&p.name),
            file_path: opt_path_to_sentinel(&p.file_path),
            schema: (&p.schema).into(),
            sequence: (&p.sequence).into(),
            media_bin: vec_model(media_bin),
            is_dirty: p.is_dirty,
        }
    }
}

impl TryFrom<&ui::Project> for Project {
    type Error = ModelParseError;
    fn try_from(d: &ui::Project) -> Result<Self, Self::Error> {
        let mut media_bin = Vec::with_capacity(d.media_bin.row_count());
        for m in d.media_bin.iter() {
            media_bin.push((&m).try_into()?);
        }
        Ok(Project {
            id: parse_id("Project.id", &d.id)?,
            name: d.name.to_string(),
            file_path: sentinel_to_opt_path(&d.file_path),
            schema: (&d.schema).into(),
            sequence: (&d.sequence).try_into()?,
            media_bin,
            is_dirty: d.is_dirty,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use models::{
        AudioStreamInfo, ClipId, MediaId, ProjectId, TrackId, VideoStreamInfo,
    };
    use std::path::PathBuf;

    fn roundtrip_rt(num: i64, den: u32) {
        let r = RationalTime::new(num, den).unwrap();
        let dto: ui::RationalTime = (&r).into();
        let back: RationalTime = (&dto).try_into().unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn rational_time_roundtrip_large() {
        // Numerator that overflows i32 — proves we cleared the bottleneck.
        roundtrip_rt(i64::MAX / 2, 90_000);
        roundtrip_rt(i64::MIN / 2, 48_000);
        roundtrip_rt(0, 1);
    }

    #[test]
    fn rational_time_bad_den_rejected() {
        let bad = ui::RationalTime {
            num: "1".into(),
            den: "0".into(),
        };
        assert!(matches!(
            RationalTime::try_from(&bad),
            Err(ModelParseError::BadDenominator { .. })
        ));
    }

    #[test]
    fn rational_time_bad_num_rejected() {
        let bad = ui::RationalTime {
            num: "not-a-number".into(),
            den: "1".into(),
        };
        assert!(matches!(
            RationalTime::try_from(&bad),
            Err(ModelParseError::BadInt { .. })
        ));
    }

    #[test]
    fn id_roundtrip_and_bad_uuid() {
        let id = ClipId::new();
        let s: SharedString = id.to_string().into();
        let back: ClipId = parse_id("Clip.id", &s).unwrap();
        assert_eq!(id, back);

        let bad: SharedString = "not-a-uuid".into();
        assert!(parse_id::<ClipId>("Clip.id", &bad).is_err());
    }

    #[test]
    fn empty_media_id_maps_to_none() {
        let empty: SharedString = "".into();
        let r: Option<MediaId> = parse_opt_id("media_id", &empty).unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn sequence_in_out_points_roundtrip() {
        let seq = Sequence {
            id: models::SequenceId::new(),
            name: "S".into(),
            width: 1920,
            height: 1080,
            fps: Rational::new_raw(30, 1),
            sample_rate: 48_000,
            timebase: 90_000,
            duration: RationalTime::new_raw(720_000, 90_000),
            in_point: Some(RationalTime::new_raw(90_000, 90_000)),
            out_point: None,
            tracks: vec![],
        };
        let dto: ui::Sequence = (&seq).into();
        assert!(dto.has_in_point);
        assert!(!dto.has_out_point);

        let back: Sequence = (&dto).try_into().unwrap();
        assert_eq!(back.in_point, seq.in_point);
        assert_eq!(back.out_point, None);
    }

    /// Build a realistic project (video+audio source, audio-only source,
    /// image source, two tracks, a clip with `media_id=None`, in/out points)
    /// and verify domain → DTO → domain preserves everything important.
    /// Also asserts that ephemeral UI defaults (selected, thumbnail, zoom)
    /// don't leak back into the persisted model.
    #[test]
    fn full_project_roundtrip_preserves_persistent_state() {
        let media_clip_src = MediaId::new();
        let media_audio_only = MediaId::new();
        let media_image = MediaId::new();
        let track_v = TrackId::new();
        let track_a = TrackId::new();
        let project_id = ProjectId::new();

        let project = Project {
            id: project_id,
            name: "Round-trip Fixture".into(),
            file_path: Some(PathBuf::from("/tmp/fixture.cutlass")),
            schema: SchemaVersion::CURRENT,
            sequence: Sequence {
                id: models::SequenceId::new(),
                name: "Main".into(),
                width: 1920,
                height: 1080,
                fps: Rational::new_raw(30_000, 1_001),
                sample_rate: 48_000,
                timebase: 90_000,
                duration: RationalTime::new_raw(5_400_000, 90_000),
                in_point: Some(RationalTime::new_raw(0, 90_000)),
                out_point: Some(RationalTime::new_raw(5_400_000, 90_000)),
                tracks: vec![
                    Track {
                        id: track_v,
                        name: "V1".into(),
                        kind: TrackKind::Video,
                        height_px: 72,
                        muted: false,
                        solo: false,
                        locked: true,
                        visible: true,
                        clips: vec![
                            Clip {
                                id: ClipId::new(),
                                media_id: Some(media_clip_src),
                                track_id: track_v,
                                name: "A".into(),
                                start: RationalTime::new_raw(0, 90_000),
                                duration: RationalTime::new_raw(270_000, 90_000),
                                source_in: RationalTime::new_raw(0, 90_000),
                                source_out: RationalTime::new_raw(270_000, 90_000),
                                speed: Rational::ONE,
                                opacity: 0.75,
                                volume: 1.0,
                                enabled: true,
                                color: Color::rgba(12, 34, 56, 200),
                            },
                            // Generator / title clip — no underlying media.
                            Clip {
                                id: ClipId::new(),
                                media_id: None,
                                track_id: track_v,
                                name: "Title".into(),
                                start: RationalTime::new_raw(270_000, 90_000),
                                duration: RationalTime::new_raw(90_000, 90_000),
                                source_in: RationalTime::new_raw(0, 90_000),
                                source_out: RationalTime::new_raw(90_000, 90_000),
                                speed: Rational::new_raw(1, 2),
                                opacity: 1.0,
                                volume: 0.0,
                                enabled: false,
                                color: Color::rgb(255, 0, 128),
                            },
                        ],
                    },
                    Track {
                        id: track_a,
                        name: "A1".into(),
                        kind: TrackKind::Audio,
                        height_px: 48,
                        muted: true,
                        solo: false,
                        locked: false,
                        visible: false,
                        clips: vec![Clip {
                            id: ClipId::new(),
                            media_id: Some(media_audio_only),
                            track_id: track_a,
                            name: "Music".into(),
                            start: RationalTime::new_raw(0, 90_000),
                            duration: RationalTime::new_raw(5_400_000, 90_000),
                            source_in: RationalTime::new_raw(0, 90_000),
                            source_out: RationalTime::new_raw(5_400_000, 90_000),
                            speed: Rational::ONE,
                            opacity: 1.0,
                            volume: 0.6,
                            enabled: true,
                            color: Color::rgb(60, 120, 90),
                        }],
                    },
                ],
            },
            media_bin: vec![
                MediaSource {
                    id: media_clip_src,
                    name: "vid.mp4".into(),
                    path: PathBuf::from("/m/vid.mp4"),
                    kind: MediaKind::Video,
                    has_video: true,
                    has_audio: true,
                    duration: RationalTime::new_raw(900_000, 90_000),
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
                    id: media_audio_only,
                    name: "song.wav".into(),
                    path: PathBuf::from("/m/song.wav"),
                    kind: MediaKind::Audio,
                    has_video: false,
                    has_audio: true,
                    duration: RationalTime::new_raw(2_700_000, 90_000),
                    video: None,
                    audio: Some(AudioStreamInfo {
                        sample_rate: 44_100,
                        codec: "pcm_s16le".into(),
                    }),
                    is_supported: true,
                    is_loading: false,
                    is_missing: false,
                    error: None,
                },
                // Missing file with error — exercises the Option<String>
                // sentinel and the `is_missing` flag.
                MediaSource {
                    id: media_image,
                    name: "poster.png".into(),
                    path: PathBuf::from("/m/poster.png"),
                    kind: MediaKind::Image,
                    has_video: false,
                    has_audio: false,
                    duration: RationalTime::new_raw(1, 1),
                    video: None,
                    audio: None,
                    is_supported: false,
                    is_loading: false,
                    is_missing: true,
                    error: Some("ENOENT".into()),
                },
            ],
            is_dirty: true,
        };

        let dto: ui::Project = (&project).into();

        // Ephemeral UI defaults should be applied and NOT bleed into domain.
        assert_eq!(dto.sequence.zoom, 50.0);
        for t in dto.sequence.tracks.iter() {
            for c in t.clips.iter() {
                assert!(!c.selected, "clip selection must default to false in DTO");
            }
        }

        let back: Project = (&dto).try_into().expect("dto→domain must succeed");

        // Top-level scalars + sentinel/Option mapping.
        assert_eq!(back.id, project.id);
        assert_eq!(back.name, project.name);
        assert_eq!(back.file_path, project.file_path);
        assert_eq!(back.is_dirty, project.is_dirty);

        // Sequence
        assert_eq!(back.sequence.id, project.sequence.id);
        assert_eq!(back.sequence.fps, project.sequence.fps);
        assert_eq!(back.sequence.timebase, project.sequence.timebase);
        assert_eq!(back.sequence.in_point, project.sequence.in_point);
        assert_eq!(back.sequence.out_point, project.sequence.out_point);

        // Clip cross-references — proves ID newtypes survive the trip.
        let orig_v_clips = &project.sequence.tracks[0].clips;
        let back_v_clips = &back.sequence.tracks[0].clips;
        assert_eq!(back_v_clips.len(), orig_v_clips.len());
        for (a, b) in orig_v_clips.iter().zip(back_v_clips) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.media_id, b.media_id, "media_id must round-trip (incl. None)");
            assert_eq!(a.track_id, b.track_id);
            assert_eq!(a.speed, b.speed);
            assert_eq!(a.opacity, b.opacity);
            assert_eq!(a.color, b.color, "color must round-trip incl. alpha");
            assert_eq!(a.enabled, b.enabled);
        }

        // Audio-only source: video=None preserved.
        let audio_only_back = back
            .media_bin
            .iter()
            .find(|m| m.id == media_audio_only)
            .expect("audio-only source");
        assert!(audio_only_back.video.is_none());
        assert!(audio_only_back.audio.is_some());

        // Image source with missing file + error sentinel.
        let img_back = back
            .media_bin
            .iter()
            .find(|m| m.id == media_image)
            .expect("image source");
        assert!(img_back.video.is_none());
        assert!(img_back.audio.is_none());
        assert!(img_back.is_missing);
        assert_eq!(img_back.error.as_deref(), Some("ENOENT"));
    }
}

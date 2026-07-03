use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::ids::MediaId;
use crate::time::{Rational, RationalTime, TimeRange};

/// Tick rate for still images, which have no frame cadence of their own.
/// Millisecond ticks (like audio-only sources) keep duration math exact.
pub const STILL_TICK_RATE: Rational = Rational::new(1000, 1);

/// Default pool duration for still images: 5 seconds at
/// [`STILL_TICK_RATE`], so a library drop places a CapCut-style 5s clip.
pub const STILL_DEFAULT_DURATION_TICKS: i64 = 5_000;

/// What a [`MediaSource`] fundamentally is, derived from its fields —
/// see [`MediaSource::kind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Video,
    Audio,
    Image,
}

/// An imported source file in the project's media pool.
///
/// This is the *asset*, not a placement on the timeline; many [`Clip`](crate::Clip)s
/// can reference the same `MediaSource`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MediaSource {
    pub id: MediaId,
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    /// Native frame rate of the source.
    pub frame_rate: Rational,
    /// Total length of the source at [`frame_rate`](Self::frame_rate).
    pub duration: RationalTime,
    pub has_audio: bool,
    /// Still image (PNG/JPEG/WebP): one frame shown for the clip's whole
    /// extent. `duration` is the *default placement length* (5s), not an
    /// intrinsic property. Absent from saves while false, so old files
    /// load unchanged.
    #[serde(default, skip_serializing_if = "is_false")]
    pub is_image: bool,
}

// `&bool` is the signature `skip_serializing_if` requires.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_false(b: &bool) -> bool {
    !*b
}

impl MediaSource {
    /// Create a media source with a freshly allocated [`MediaId`].
    pub fn new(
        path: impl Into<PathBuf>,
        width: u32,
        height: u32,
        frame_rate: Rational,
        duration_ticks: i64,
        has_audio: bool,
    ) -> Self {
        let duration_ticks = duration_ticks.max(0);
        Self {
            id: MediaId::next(),
            path: path.into(),
            width,
            height,
            frame_rate,
            duration: RationalTime::new(duration_ticks, frame_rate),
            has_audio,
            is_image: false,
        }
    }

    /// Create a still-image source (PNG/JPEG/WebP) with the default 5s
    /// placement length at [`STILL_TICK_RATE`].
    pub fn image(path: impl Into<PathBuf>, width: u32, height: u32) -> Self {
        Self {
            is_image: true,
            ..Self::new(
                path,
                width,
                height,
                STILL_TICK_RATE,
                STILL_DEFAULT_DURATION_TICKS,
                false,
            )
        }
    }

    /// The full extent of the source as `[0, duration)`.
    pub fn full_range(&self) -> TimeRange {
        TimeRange::at_rate(0, self.duration.value, self.frame_rate)
    }

    /// Whether this source has no video stream (music/voiceover files).
    /// Probing reports such sources with zero dimensions.
    pub fn is_audio_only(&self) -> bool {
        self.width == 0 && self.has_audio
    }

    pub fn kind(&self) -> MediaKind {
        if self.is_image {
            MediaKind::Image
        } else if self.is_audio_only() {
            MediaKind::Audio
        } else {
            MediaKind::Video
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const R24: Rational = Rational::FPS_24;
    const R30: Rational = Rational::FPS_30;

    fn sample(path: &str, duration: i64) -> MediaSource {
        MediaSource::new(path, 1920, 1080, R24, duration, true)
    }

    // --- MediaSource::new ---------------------------------------------------

    #[test]
    fn new_wires_all_fields() {
        let media = MediaSource::new("/media/clip.mp4", 3840, 2160, R30, 1_800, false);

        assert_eq!(media.path, PathBuf::from("/media/clip.mp4"));
        assert_eq!(media.width, 3840);
        assert_eq!(media.height, 2160);
        assert_eq!(media.frame_rate, R30);
        assert_eq!(media.duration, RationalTime::new(1_800, R30));
        assert!(!media.has_audio);
        assert!(media.id.raw() >= 1);
    }

    #[test]
    fn new_accepts_path_buf_and_str() {
        let from_str = MediaSource::new("a.mp4", 1, 1, R24, 10, true);
        let from_buf = MediaSource::new(PathBuf::from("b.mp4"), 1, 1, R24, 10, true);
        assert_eq!(from_str.path(), Path::new("a.mp4"));
        assert_eq!(from_buf.path(), Path::new("b.mp4"));
    }

    #[test]
    fn new_assigns_distinct_media_ids() {
        let a = sample("a.mp4", 100);
        let b = sample("b.mp4", 100);
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn new_clamps_negative_duration_to_zero() {
        let media = sample("short.mp4", -50);
        assert_eq!(media.duration.value, 0);
        assert_eq!(media.duration.rate, R24);
    }

    #[test]
    fn new_zero_duration_is_valid() {
        let media = sample("empty.mp4", 0);
        assert_eq!(media.duration.value, 0);
        assert!(media.full_range().is_empty());
    }

    #[test]
    fn new_duration_carries_frame_rate() {
        let ntsc = MediaSource::new("ntsc.mp4", 1920, 1080, Rational::FPS_23_976, 2_400, true);
        assert_eq!(ntsc.frame_rate, Rational::FPS_23_976);
        assert_eq!(ntsc.duration.rate, Rational::FPS_23_976);
        assert_eq!(ntsc.duration.value, 2_400);
    }

    #[test]
    fn new_has_audio_flag() {
        assert!(sample("with-audio.mp4", 10).has_audio);
        assert!(!MediaSource::new("silent.mp4", 1, 1, R24, 10, false).has_audio);
    }

    // --- full_range -------------------------------------------------------

    #[test]
    fn full_range_spans_entire_source() {
        let media = sample("clip.mp4", 500);
        assert_eq!(media.full_range(), TimeRange::at_rate(0, 500, R24));
    }

    #[test]
    fn full_range_matches_duration_at_native_rate() {
        let media = MediaSource::new("clip.mp4", 1280, 720, R30, 300, true);
        let range = media.full_range();
        assert_eq!(range.start, RationalTime::new(0, R30));
        assert_eq!(range.duration, media.duration);
        assert_eq!(range.end_tick(), 300);
    }

    #[test]
    fn full_range_zero_duration_is_empty() {
        let media = sample("blank.mp4", 0);
        let range = media.full_range();
        assert!(range.is_empty());
        assert_eq!(range.start.value, 0);
        assert_eq!(range.end_tick(), 0);
    }

    // --- path -------------------------------------------------------------

    #[test]
    fn path_returns_borrowed_path() {
        let media = sample("/vault/footage/take_01.mov", 100);
        assert_eq!(media.path(), Path::new("/vault/footage/take_01.mov"));
        assert_eq!(
            media.path().file_name().and_then(|s| s.to_str()),
            Some("take_01.mov")
        );
    }

    // --- Clone / Eq / Debug -----------------------------------------------

    #[test]
    fn clone_and_eq_preserve_all_fields() {
        let original = MediaSource::new("x.mp4", 640, 480, R24, 42, true);
        let cloned = original.clone();
        assert_eq!(original, cloned);
        assert_eq!(original.id, cloned.id);
        assert_eq!(original.duration, cloned.duration);
    }

    #[test]
    fn eq_requires_matching_id_and_metadata() {
        let a = MediaSource::new("same.mp4", 1920, 1080, R24, 100, true);
        let b = MediaSource::new("same.mp4", 1920, 1080, R24, 100, true);
        // Distinct `next()` ids => not equal even with identical metadata.
        assert_ne!(a, b);

        let fixed_a = MediaSource {
            id: MediaId::from_raw(1),
            path: PathBuf::from("fixed.mp4"),
            width: 1920,
            height: 1080,
            frame_rate: R24,
            duration: RationalTime::new(100, R24),
            has_audio: true,
            is_image: false,
        };
        let fixed_b = fixed_a.clone();
        assert_eq!(fixed_a, fixed_b);
    }

    #[test]
    fn debug_includes_path_and_dimensions() {
        let media = sample("debug.mp4", 10);
        let s = format!("{media:?}");
        assert!(s.contains("debug.mp4"));
        assert!(s.contains("1920"));
        assert!(s.contains("1080"));
    }

    // --- image stills -------------------------------------------------------

    #[test]
    fn image_constructor_defaults_to_five_seconds() {
        let media = MediaSource::image("/photos/sunset.png", 4000, 3000);
        assert!(media.is_image);
        assert_eq!(media.kind(), MediaKind::Image);
        assert_eq!(media.width, 4000);
        assert_eq!(media.height, 3000);
        assert!(!media.has_audio);
        assert_eq!(media.frame_rate, STILL_TICK_RATE);
        assert_eq!(media.duration.value, STILL_DEFAULT_DURATION_TICKS);
        // 5000 ticks at 1000/1 is exactly 5 seconds.
        let seconds = media.duration.value as f64 * media.duration.rate.seconds_per_unit();
        assert_eq!(seconds, 5.0);
        assert!(!media.is_audio_only());
    }

    #[test]
    fn kind_distinguishes_video_audio_image() {
        assert_eq!(sample("v.mp4", 10).kind(), MediaKind::Video);
        let audio = MediaSource::new("a.mp3", 0, 0, Rational::new(1000, 1), 10, true);
        assert_eq!(audio.kind(), MediaKind::Audio);
        assert_eq!(MediaSource::image("i.jpg", 8, 8).kind(), MediaKind::Image);
    }

    #[test]
    fn non_image_serializes_without_image_field() {
        let media = sample("plain.mp4", 10);
        let json = serde_json::to_string(&media).unwrap();
        assert!(!json.contains("is_image"));

        let image = MediaSource::image("pic.png", 64, 64);
        let json = serde_json::to_string(&image).unwrap();
        assert!(json.contains("\"is_image\":true"));
    }

    #[test]
    fn image_flag_roundtrips_and_defaults_false() {
        let image = MediaSource::image("pic.webp", 320, 240);
        let json = serde_json::to_string(&image).unwrap();
        let loaded: MediaSource = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded, image);

        // Old saves carry no `is_image` field — it must default to false.
        let legacy = r#"{"id":1,"path":"old.mp4","width":1920,"height":1080,
            "frame_rate":{"num":24,"den":1},
            "duration":{"value":48,"rate":{"num":24,"den":1}},"has_audio":true}"#;
        let loaded: MediaSource = serde_json::from_str(legacy).unwrap();
        assert!(!loaded.is_image);
        assert_eq!(loaded.kind(), MediaKind::Video);
    }
}

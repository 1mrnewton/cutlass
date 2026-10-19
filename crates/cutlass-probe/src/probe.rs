//! Demux-only probing. Via FFmpeg on desktop; via the platform-native decoder
//! on builds without FFmpeg (iOS/Android).

use std::path::Path;
#[cfg(feature = "ffmpeg")]
use std::sync::OnceLock;

#[cfg(feature = "ffmpeg")]
use ffmpeg_next::codec::context::Context;
#[cfg(feature = "ffmpeg")]
use ffmpeg_next::format;
#[cfg(feature = "ffmpeg")]
use ffmpeg_next::format::stream::Disposition;
#[cfg(feature = "ffmpeg")]
use ffmpeg_next::media::Type;
#[cfg(feature = "ffmpeg")]
use ffmpeg_next::{Error as FfmpegError, Rational as FfRational};

use cutlass_models::Rational;

use crate::error::ProbeError;
use crate::media::MediaProbe;

#[cfg(feature = "ffmpeg")]
static FFMPEG_INIT: OnceLock<Result<(), FfmpegError>> = OnceLock::new();

#[cfg(feature = "ffmpeg")]
pub(crate) fn ensure_ffmpeg_init() -> Result<(), ProbeError> {
    match FFMPEG_INIT.get_or_init(ffmpeg_next::init) {
        Ok(()) => Ok(()),
        Err(e) => Err(ProbeError::Open(*e)),
    }
}

/// Tick rate for audio-only sources, which have no frame rate of their own.
/// Millisecond ticks keep durations sample-accurate enough for timeline math.
#[cfg(feature = "ffmpeg")]
const AUDIO_TICK_RATE: Rational = Rational::new(1000, 1);

/// Still-image extensions recognized at import (M1 scope: PNG/JPEG/WebP).
const IMAGE_EXTENSIONS: [&str; 4] = ["png", "jpg", "jpeg", "webp"];

/// Whether `path` looks like a still image we import as such (by extension;
/// [`probe`] verifies the content decodes).
pub fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| IMAGE_EXTENSIONS.iter().any(|i| ext.eq_ignore_ascii_case(i)))
}

/// Inspect `path` for container and stream metadata (ffprobe-style).
///
/// Audio-only files (no real video stream — embedded cover art doesn't count)
/// probe with `width == 0 && height == 0` and a millisecond tick rate.
///
/// Still images (recognized by extension) probe with `is_image == true` and
/// the default 5s placement duration; FFmpeg demuxes them as single-frame
/// video streams, which is where the dimensions come from.
#[cfg(feature = "ffmpeg")]
pub fn probe(path: &Path) -> Result<MediaProbe, ProbeError> {
    ensure_ffmpeg_init()?;

    let path_str = path.to_str().ok_or(ProbeError::InvalidPath)?;
    let input = format::input(path_str).map_err(ProbeError::Open)?;

    if is_image_path(path) {
        return probe_image(&input);
    }

    let has_audio = input
        .streams()
        .any(|s| s.parameters().medium() == Type::Audio);

    // mp3/m4a cover art is muxed as a video stream flagged ATTACHED_PIC;
    // treat those files as audio, not as one-frame videos.
    let video_stream = input
        .streams()
        .best(Type::Video)
        .filter(|s| !s.disposition().contains(Disposition::ATTACHED_PIC));

    let Some(stream) = video_stream else {
        if !has_audio {
            return Err(ProbeError::unsupported("no video or audio stream found"));
        }
        let micros = input.duration();
        return Ok(MediaProbe {
            width: 0,
            height: 0,
            frame_rate: AUDIO_TICK_RATE,
            duration_ticks: duration_ticks_from_micros(AUDIO_TICK_RATE, micros.max(0) as u64),
            has_audio: true,
            video_codec: "none".into(),
            is_image: false,
        });
    };

    let par = stream.parameters();
    if par.medium() != Type::Video {
        return Err(ProbeError::unsupported("best stream is not video"));
    }

    let codec_id = par.id();
    let video = Context::from_parameters(par)
        .map_err(ProbeError::Open)?
        .decoder()
        .video()
        .map_err(ProbeError::Open)?;
    let width = video.width();
    let height = video.height();
    if width == 0 || height == 0 {
        return Err(ProbeError::unsupported("zero video dimensions"));
    }

    let frame_rate = normalize_frame_rate(stream.avg_frame_rate());
    let micros = input.duration();
    let duration_ticks = duration_ticks_from_micros(frame_rate, micros.max(0) as u64);

    let video_codec = ffmpeg_next::codec::decoder::find(codec_id)
        .map(|c| c.name().to_string())
        .unwrap_or_else(|| "unknown".into());

    Ok(MediaProbe {
        width,
        height,
        frame_rate,
        duration_ticks,
        has_audio,
        video_codec,
        is_image: false,
    })
}

/// Probe a still image: FFmpeg's image demuxers (`png_pipe`, `jpeg_pipe`,
/// `webp_pipe`, …) expose one video stream whose codec parameters carry the
/// pixel dimensions. Duration is the editor's default placement length, not
/// a property of the file.
#[cfg(feature = "ffmpeg")]
fn probe_image(input: &format::context::Input) -> Result<MediaProbe, ProbeError> {
    let stream = input
        .streams()
        .best(Type::Video)
        .ok_or_else(|| ProbeError::unsupported("image file has no decodable picture"))?;

    let par = stream.parameters();
    let codec_id = par.id();
    let video = Context::from_parameters(par)
        .map_err(ProbeError::Open)?
        .decoder()
        .video()
        .map_err(ProbeError::Open)?;
    let width = video.width();
    let height = video.height();
    if width == 0 || height == 0 {
        return Err(ProbeError::unsupported("zero image dimensions"));
    }

    let video_codec = ffmpeg_next::codec::decoder::find(codec_id)
        .map(|c| c.name().to_string())
        .unwrap_or_else(|| "unknown".into());

    Ok(MediaProbe {
        width,
        height,
        frame_rate: cutlass_models::STILL_TICK_RATE,
        duration_ticks: cutlass_models::STILL_DEFAULT_DURATION_TICKS,
        has_audio: false,
        video_codec,
        is_image: true,
    })
}

#[cfg(feature = "ffmpeg")]
fn normalize_frame_rate(rate: FfRational) -> Rational {
    let frame_rate = Rational::new(rate.numerator(), rate.denominator());
    if frame_rate.is_valid() {
        frame_rate
    } else {
        Rational::FPS_24
    }
}

/// FFmpeg-free probe: read container/stream metadata from the platform-native
/// decoder ([`cutlass_decoder::open_video_decoder`]). Audio presence and the
/// exact codec name aren't exposed by the video decoder, so `has_audio` is
/// reported `false` and `video_codec` as `"native"`; the preview/render path
/// doesn't depend on either. Still images report the default placement length
/// with unknown dimensions (no still decoder in this build).
#[cfg(not(feature = "ffmpeg"))]
pub fn probe(path: &Path) -> Result<MediaProbe, ProbeError> {
    if is_image_path(path) {
        return Ok(MediaProbe {
            width: 0,
            height: 0,
            frame_rate: cutlass_models::STILL_TICK_RATE,
            duration_ticks: cutlass_models::STILL_DEFAULT_DURATION_TICKS,
            has_audio: false,
            video_codec: "image".into(),
            is_image: true,
        });
    }

    let decoder = cutlass_decoder::open_video_decoder(path, cutlass_decoder::OutputMode::Cpu)
        .map_err(|e| ProbeError::unsupported(e.to_string()))?;
    let info = decoder.info();

    let (width, height) = info.display_size;
    if width == 0 || height == 0 {
        return Err(ProbeError::unsupported("zero video dimensions"));
    }

    let frame_rate = Rational::new(info.frame_rate.num, info.frame_rate.den);
    let frame_rate = if frame_rate.is_valid() {
        frame_rate
    } else {
        Rational::FPS_24
    };

    let duration_ticks = match info.duration {
        Some(d) => (d.seconds() * frame_rate.as_f64()).round().max(0.0) as i64,
        None => 0,
    };

    Ok(MediaProbe {
        width,
        height,
        frame_rate,
        duration_ticks,
        has_audio: false,
        video_codec: "native".into(),
        is_image: false,
    })
}

/// Map container duration (microseconds) to frame ticks at `frame_rate`.
pub fn duration_ticks_from_micros(frame_rate: Rational, micros: u64) -> i64 {
    if !frame_rate.is_valid() || micros == 0 {
        return 0;
    }
    let ticks = (i128::from(micros) * i128::from(frame_rate.num))
        / (i128::from(frame_rate.den) * 1_000_000);
    ticks.clamp(0, i128::from(i64::MAX)) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_ticks_from_micros_one_second_at_24fps() {
        assert_eq!(duration_ticks_from_micros(Rational::FPS_24, 1_000_000), 24);
    }

    #[test]
    fn duration_ticks_from_micros_zero_micros_is_zero() {
        assert_eq!(duration_ticks_from_micros(Rational::FPS_24, 0), 0);
    }

    #[cfg(feature = "ffmpeg")]
    #[test]
    fn normalize_frame_rate_falls_back_to_24() {
        assert_eq!(
            normalize_frame_rate(FfRational::new(0, 1)),
            Rational::FPS_24
        );
    }

    #[test]
    fn is_image_path_recognizes_still_extensions() {
        use std::path::Path;
        for good in ["a.png", "b.jpg", "c.JPEG", "d.WebP", "/x/y/e.PNG"] {
            assert!(is_image_path(Path::new(good)), "{good}");
        }
        for bad in ["clip.mp4", "song.mp3", "noext", "tricky.png.mov", "g.gif"] {
            assert!(!is_image_path(Path::new(bad)), "{bad}");
        }
    }

    #[cfg(feature = "ffmpeg")]
    #[test]
    fn image_file_probes_with_default_still_duration() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../local-assets/assets/texture.png");
        if !path.exists() {
            return;
        }
        let probed = probe(&path).expect("probe png");
        assert!(probed.is_image);
        assert!(probed.width > 0 && probed.height > 0);
        assert!(!probed.has_audio);
        assert_eq!(probed.frame_rate, cutlass_models::STILL_TICK_RATE);
        assert_eq!(
            probed.duration_ticks,
            cutlass_models::STILL_DEFAULT_DURATION_TICKS
        );
        assert_eq!(probed.video_codec, "png");

        let media = probed.to_media_source(&path);
        assert!(media.is_image);
        assert_eq!(media.kind(), cutlass_models::MediaKind::Image);
        assert_eq!(
            media.duration.value,
            cutlass_models::STILL_DEFAULT_DURATION_TICKS
        );
    }

    #[cfg(feature = "ffmpeg")]
    #[test]
    fn audio_only_file_probes_with_zero_dimensions() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../local-assets/assets/baby.mp3");
        if !path.exists() {
            return;
        }
        let probed = probe(&path).expect("probe mp3");
        assert_eq!(probed.width, 0);
        assert_eq!(probed.height, 0);
        assert!(probed.has_audio);
        assert_eq!(probed.frame_rate, AUDIO_TICK_RATE);
        assert!(probed.duration_ticks > 0);
        assert_eq!(probed.video_codec, "none");
    }
}

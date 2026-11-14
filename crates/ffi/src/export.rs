//! Export the whole cut as an MP4 the phone can save to its photos.
//!
//! The picture is drawn on the preview's GPU, so effect clips are in the
//! file, and encoded as H.264. The sound is each clip's file mixed as in
//! playback and encoded as AAC. The Apple encoders go first on an Apple
//! phone, then the Rust ones. The shared [`cutlass_export::write_movie`]
//! writes the file, so a stopped or failed export leaves nothing behind.

use std::path::{Path, PathBuf};

use cutlass_codec::{CodecId, ContainerId};
use cutlass_encode::EncodeFactory;
use cutlass_eval::{Fonts, ReaderFactory};
use cutlass_export::{ExportError, ExportRequest, MovieSources};
use cutlass_model::{Draft, FrameRate, TimeRange};
use cutlass_render::Renderer;

use crate::error::FfiError;

/// The short sides, in pixels, the phone offers.
pub const PHONE_EXPORT_SIDES: [u32; 3] = [720, 1080, 2160];

/// The frame rates the phone offers, beside the project's own.
pub const PHONE_EXPORT_RATES: [u32; 5] = [24, 25, 30, 50, 60];

/// How many bits a second the picture gets, against what a phone's own
/// camera records at that size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PhoneQuality {
    /// Half: a smaller file.
    Lower,
    /// About what the camera records.
    #[default]
    Recommended,
    /// Double: fewer blocks in fast or busy pictures.
    Higher,
}

/// How an export is made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhoneExportOptions {
    /// One of [`PHONE_EXPORT_SIDES`].
    pub short_side: u32,
    /// One of [`PHONE_EXPORT_RATES`], or 0 for the project's own rate.
    pub frame_rate: u32,
    pub quality: PhoneQuality,
}

impl PhoneExportOptions {
    /// `short_side` at the project's own rate and the recommended quality.
    #[must_use]
    pub fn sized(short_side: u32) -> Self {
        Self {
            short_side,
            frame_rate: 0,
            quality: PhoneQuality::Recommended,
        }
    }
}

/// A finished export.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneExport {
    pub path: PathBuf,
    pub frames: u64,
    pub width: u32,
    pub height: u32,
}

/// Opus, AC-3, E-AC-3, Vorbis and Motion JPEG in Rust, the platform encoders
/// for this OS, then H.264, MPEG-4 Visual, AAC and ProRes in Rust. The
/// phone's own export still writes H.264; MPEG-4 is there when a caller asks
/// for it and the platform encoder declines.
#[must_use]
pub fn encoders() -> EncodeFactory {
    let mut encoders = EncodeFactory::new();
    // Rust audio codecs go ahead of the platform ones, so Opus, AC-3, E-AC-3
    // and Vorbis encode the same on the phone as on the desktop; Motion JPEG
    // does too.
    cutlass_opus::register_encoders(&mut encoders);
    cutlass_ac3::register_encoders(&mut encoders);
    cutlass_vorbis::register_encoders(&mut encoders);
    cutlass_mjpeg::register_encoders(&mut encoders);
    #[cfg(target_vendor = "apple")]
    cutlass_platform_apple::register_encoders(&mut encoders);
    // After the platform's, as on the desktop: a freeze frame's still is a
    // PNG, and Android and Linux have only this one.
    cutlass_png::register_encoders(&mut encoders);
    cutlass_h264::register_encoders(&mut encoders);
    cutlass_mpeg4::register_engine_encoders(&mut encoders);
    cutlass_aac::register_encoders(&mut encoders);
    cutlass_prores::register_encoders(&mut encoders);
    encoders
}

/// What one export reads and draws with.
pub(crate) struct Sources<'a> {
    pub draft: &'a Draft,
    pub root: &'a Path,
    pub readers: &'a ReaderFactory,
    pub fonts: &'a dyn Fonts,
    pub renderer: Option<&'a Renderer>,
}

/// The rate `options` asks for: the project's own when 0.
fn rate_of(draft: &Draft, frame_rate: u32) -> Result<FrameRate, FfiError> {
    if frame_rate == 0 {
        return Ok(draft.frame_rate().unwrap_or(FrameRate::FPS_30));
    }
    if !PHONE_EXPORT_RATES.contains(&frame_rate) {
        return Err(refused("that frame rate is not offered"));
    }
    Ok(FrameRate {
        numerator: frame_rate,
        denominator: 1,
    })
}

/// Bits a second for the picture: the camera's rate for the size, half
/// again above 30 frames a second, then halved or doubled for `quality`.
fn bitrate_of(short_side: u32, rate: FrameRate, quality: PhoneQuality) -> u32 {
    let base = u64::from(cutlass_export::video_bitrate(short_side));
    let fast = u64::from(rate.numerator) > 30 * u64::from(rate.denominator.max(1));
    let base = if fast { base * 3 / 2 } else { base };
    let bits = match quality {
        PhoneQuality::Lower => base / 2,
        PhoneQuality::Recommended => base,
        PhoneQuality::Higher => base * 2,
    };
    u32::try_from(bits).unwrap_or(u32::MAX)
}

/// The request that writes the whole of `draft` as `options` asks.
fn request_of(draft: &Draft, options: PhoneExportOptions) -> Result<ExportRequest, FfiError> {
    if !PHONE_EXPORT_SIDES.contains(&options.short_side) {
        return Err(refused("that size is not offered"));
    }
    let frame_rate = rate_of(draft, options.frame_rate)?;
    Ok(ExportRequest {
        range: TimeRange::new(0, draft.duration.0),
        short_side: options.short_side,
        codec: CodecId::H264,
        bitrate: bitrate_of(options.short_side, frame_rate, options.quality),
        frame_rate,
        container: ContainerId::Isobmff,
        audio: Some(cutlass_export::MOVIE_SOUND),
        sidecar: None,
        late_by: 0,
        constant_rate: false,
        prores: cutlass_export::ProresProfile::Standard,
    })
}

/// Write the whole cut to `destination` as `options` asks. `progress`
/// hears the frames done and the frames in all after each picture, and
/// stops the export by returning false.
pub(crate) fn export_movie(
    sources: &Sources<'_>,
    destination: &Path,
    options: PhoneExportOptions,
    progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<PhoneExport, FfiError> {
    let request = request_of(sources.draft, options)?;
    if sources.draft.duration.0 <= 0 {
        return Err(refused("there is nothing to export"));
    }
    if cutlass_export::part_path(destination).is_none() {
        return Err(refused("the export has no file name"));
    }
    let encoders = encoders();
    let result = cutlass_export::write_movie(
        &MovieSources {
            draft: sources.draft,
            root: Some(sources.root),
            readers: sources.readers,
            fonts: sources.fonts,
            renderer: sources.renderer,
            encoders: &encoders,
        },
        &request,
        destination,
        progress,
    )
    .map_err(exported)?;
    Ok(PhoneExport {
        path: destination.to_path_buf(),
        frames: u64::try_from(result.frames.len()).unwrap_or(u64::MAX),
        width: result.output_width,
        height: result.output_height,
    })
}

fn exported(error: ExportError) -> FfiError {
    match error {
        ExportError::Cancelled => refused("export was stopped"),
        ExportError::File(message) => refused(&message),
        error => refused(&format!("export failed: {error}")),
    }
}

fn refused(reason: &str) -> FfiError {
    FfiError::Refused {
        reason: reason.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_export_takes_the_projects_rate_or_an_offered_one_and_scales_its_bits() {
        let mut draft = Draft::empty("draft");
        draft.fps = 25.0;
        let own = request_of(&draft, PhoneExportOptions::sized(1080)).unwrap();
        assert_eq!(own.frame_rate, FrameRate::FPS_25);
        assert_eq!(own.bitrate, 12_000_000);

        let fast = request_of(
            &draft,
            PhoneExportOptions {
                short_side: 1080,
                frame_rate: 60,
                quality: PhoneQuality::Higher,
            },
        )
        .unwrap();
        assert_eq!(fast.frame_rate, FrameRate::FPS_60);
        assert_eq!(fast.bitrate, 36_000_000);

        let small = request_of(
            &draft,
            PhoneExportOptions {
                short_side: 720,
                frame_rate: 24,
                quality: PhoneQuality::Lower,
            },
        )
        .unwrap();
        assert_eq!(small.frame_rate, FrameRate::FPS_24);
        assert_eq!(small.bitrate, 3_000_000);

        for (side, rate) in [(1080, 48), (900, 30)] {
            let options = PhoneExportOptions {
                short_side: side,
                frame_rate: rate,
                quality: PhoneQuality::Recommended,
            };
            assert!(request_of(&draft, options).is_err());
        }
    }
}

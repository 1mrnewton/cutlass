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

/// A finished export.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneExport {
    pub path: PathBuf,
    pub frames: u64,
    pub width: u32,
    pub height: u32,
}

/// Opus and Vorbis in Rust, the platform encoders for this OS, then H.264,
/// AAC and ProRes in Rust.
#[must_use]
pub fn encoders() -> EncodeFactory {
    let mut encoders = EncodeFactory::new();
    // Rust audio codecs go ahead of the platform ones, so Opus and Vorbis
    // encode the same on the phone as on the desktop.
    cutlass_opus::register_encoders(&mut encoders);
    cutlass_vorbis::register_encoders(&mut encoders);
    #[cfg(target_vendor = "apple")]
    cutlass_platform_apple::register_encoders(&mut encoders);
    cutlass_h264::register_encoders(&mut encoders);
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

/// Write the whole cut to `destination` at `short_side`. `progress` hears
/// the frames done and the frames in all after each picture, and stops the
/// export by returning false.
pub(crate) fn export_movie(
    sources: &Sources<'_>,
    destination: &Path,
    short_side: u32,
    progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<PhoneExport, FfiError> {
    if !PHONE_EXPORT_SIDES.contains(&short_side) {
        return Err(refused("that size is not offered"));
    }
    if sources.draft.duration.0 <= 0 {
        return Err(refused("there is nothing to export"));
    }
    if cutlass_export::part_path(destination).is_none() {
        return Err(refused("the export has no file name"));
    }
    let request = ExportRequest {
        range: TimeRange::new(0, sources.draft.duration.0),
        short_side,
        codec: CodecId::H264,
        bitrate: cutlass_export::video_bitrate(short_side),
        frame_rate: sources.draft.frame_rate().unwrap_or(FrameRate::FPS_30),
        container: ContainerId::Isobmff,
        audio: Some(cutlass_export::MOVIE_SOUND),
        sidecar: None,
        late_by: 0,
        constant_rate: false,
        prores: cutlass_export::ProresProfile::Standard,
    };
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

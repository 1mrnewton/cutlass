//! Export the whole cut as an MP4 the phone can save to its photos.
//!
//! The picture is drawn on the preview's GPU, so effect clips are in the
//! file, and encoded as H.264. The sound is each clip's file mixed as in
//! playback and encoded as AAC. The Apple encoders go first on an Apple
//! phone, then the Rust ones. The file is written beside its destination and
//! moved there only once it is finished, so a stopped or failed export leaves
//! nothing behind.

use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use cutlass_codec::{CodecId, ContainerId};
use cutlass_encode::EncodeFactory;
use cutlass_eval::{Fonts, ReaderFactory};
use cutlass_export::{AudioExport, ExportError, ExportRequest, ExportRun};
use cutlass_isobmff::Mp4Mux;
use cutlass_model::{Draft, FrameRate, TimeRange};
use cutlass_render::Renderer;
use cutlass_renderer_software::SoftwareDevice;

use crate::error::FfiError;

/// The short sides, in pixels, the phone offers.
pub const PHONE_EXPORT_SIDES: [u32; 3] = [720, 1080, 2160];

/// The sound every phone export carries.
const SOUND: AudioExport = AudioExport {
    codec: CodecId::Aac,
    bitrate: 192_000,
    sample_rate: 48_000,
    channels: 2,
    limiter: 0,
};

/// A finished export.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneExport {
    pub path: PathBuf,
    pub frames: u64,
    pub width: u32,
    pub height: u32,
}

/// The platform encoders for this OS, then H.264 and AAC in Rust.
#[must_use]
pub fn encoders() -> EncodeFactory {
    let mut encoders = EncodeFactory::new();
    #[cfg(target_vendor = "apple")]
    cutlass_platform_apple::register_encoders(&mut encoders);
    cutlass_h264::register_encoders(&mut encoders);
    cutlass_aac::register_encoders(&mut encoders);
    encoders
}

/// Bits a second for a picture `short_side` pixels on its short side: about
/// what a phone's own camera records at that size.
#[must_use]
pub fn video_bitrate(short_side: u32) -> u32 {
    match short_side {
        0..=720 => 6_000_000,
        721..=1080 => 12_000_000,
        _ => 40_000_000,
    }
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
    let name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| refused("the export has no file name"))?;
    let part = destination.with_file_name(format!(".{name}.part"));
    let request = ExportRequest {
        range: TimeRange::new(0, sources.draft.duration.0),
        short_side,
        codec: CodecId::H264,
        bitrate: video_bitrate(short_side),
        frame_rate: sources.draft.frame_rate().unwrap_or(FrameRate::FPS_30),
        container: ContainerId::Isobmff,
        audio: Some(SOUND),
        sidecar: None,
        late_by: 0,
    };
    let written = write(sources, &request, &part, progress);
    let result = match written {
        Ok(result) => result,
        Err(error) => {
            let _ = fs::remove_file(&part);
            return Err(error);
        }
    };
    fs::rename(&part, destination).map_err(|error| {
        let _ = fs::remove_file(&part);
        refused(&error.to_string())
    })?;
    Ok(PhoneExport {
        path: destination.to_path_buf(),
        frames: u64::try_from(result.frames.len()).unwrap_or(u64::MAX),
        width: result.output_width,
        height: result.output_height,
    })
}

fn write(
    sources: &Sources<'_>,
    request: &ExportRequest,
    part: &Path,
    progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<cutlass_export::ExportResult, FfiError> {
    let file = File::create(part).map_err(|error| refused(&error.to_string()))?;
    let mut mux = Mp4Mux::new(BufWriter::new(file));
    let mut device = SoftwareDevice::new();
    let encoders = encoders();
    let result = cutlass_export::export(
        ExportRun {
            draft: sources.draft,
            device: &mut device,
            request,
            factory: &encoders,
            readers: sources.readers,
            mux: &mut mux,
            destination: None,
            root: Some(sources.root),
            fonts: sources.fonts,
            renderer: sources.renderer,
            progress: Some(progress),
        },
        cutlass_export::file_audio(sources.readers, sources.draft, Some(sources.root)),
    )
    .map_err(exported)?;
    let mut writer = mux
        .into_writer()
        .map_err(|error| refused(&error.to_string()))?;
    std::io::Write::flush(&mut writer).map_err(|error| refused(&error.to_string()))?;
    Ok(result)
}

fn exported(error: ExportError) -> FfiError {
    match error {
        ExportError::Cancelled => refused("export was stopped"),
        error => refused(&format!("export failed: {error}")),
    }
}

fn refused(reason: &str) -> FfiError {
    FfiError::Refused {
        reason: reason.to_owned(),
    }
}

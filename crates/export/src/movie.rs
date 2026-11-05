//! Write a range as a movie file on disk.
//!
//! The file is written beside its destination under a hidden part name and
//! moved there only once it is finished, so a stopped or failed export
//! leaves nothing behind. Every app calls this; each picks its own sizes,
//! bitrates and encoders.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use cutlass_codec::ContainerId;
use cutlass_encode::EncodeFactory;
use cutlass_eval::{Fonts, ReaderFactory};
use cutlass_isobmff::{FileBrand, Mp4Mux};
use cutlass_model::Draft;
use cutlass_render::Renderer;
use cutlass_renderer_software::SoftwareDevice;
use cutlass_riff::WavMux;

use cutlass_codec::CodecId;

use crate::run::export_sound;
use crate::{AudioExport, ExportError, ExportRequest, ExportResult, ExportRun, export, file_audio};

/// Stereo AAC at 48 kHz, the sound a movie carries unless asked otherwise.
pub const MOVIE_SOUND: AudioExport = AudioExport {
    codec: CodecId::Aac,
    bitrate: 192_000,
    sample_rate: 48_000,
    channels: 2,
    limiter: 0,
};

/// Bits a second for an H.264 picture `short_side` pixels on its short
/// side: about what a phone's own camera records at that size.
#[must_use]
pub fn video_bitrate(short_side: u32) -> u32 {
    match short_side {
        0..=720 => 6_000_000,
        721..=1080 => 12_000_000,
        _ => 40_000_000,
    }
}

/// What a movie export reads, draws and encodes with.
pub struct MovieSources<'a> {
    pub draft: &'a Draft,
    /// Project folder for a relative media path.
    pub root: Option<&'a Path>,
    pub readers: &'a ReaderFactory,
    pub fonts: &'a dyn Fonts,
    /// The preview's GPU, so effect clips are in the file. `None` draws on
    /// the CPU without them.
    pub renderer: Option<&'a Renderer>,
    pub encoders: &'a EncodeFactory,
}

/// The hidden file a movie is written to before it is moved into place.
#[must_use]
pub fn part_path(destination: &Path) -> Option<PathBuf> {
    let name = destination.file_name()?.to_str()?;
    if name.is_empty() {
        return None;
    }
    Some(destination.with_file_name(format!(".{name}.part")))
}

/// Write `request` to `destination` as an MP4, or a QuickTime movie when
/// the destination ends in `.mov`. `progress` hears the frames done and the
/// frames in all after each picture, and stops the export by returning
/// false. Clip sound is each clip's file, mixed as in playback.
pub fn write_movie(
    sources: &MovieSources<'_>,
    request: &ExportRequest,
    destination: &Path,
    progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<ExportResult, ExportError> {
    if request.container != ContainerId::Isobmff {
        return Err(ExportError::Container);
    }
    let part = part_path(destination).ok_or(ExportError::Path)?;
    let written = write_part(sources, request, &part, progress).and_then(|result| {
        fs::rename(&part, destination)
            .map(|()| result)
            .map_err(file)
    });
    if written.is_err() {
        let _ = fs::remove_file(&part);
    }
    written
}

fn write_part(
    sources: &MovieSources<'_>,
    request: &ExportRequest,
    part: &Path,
    progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<ExportResult, ExportError> {
    let brand = if is_mov(part) {
        FileBrand::Mov
    } else {
        FileBrand::Mp4
    };
    let mut mux = Mp4Mux::with_brand(BufWriter::new(File::create(part).map_err(file)?), brand);
    let mut device = SoftwareDevice::new();
    let result = export(
        ExportRun {
            draft: sources.draft,
            device: &mut device,
            request,
            factory: sources.encoders,
            readers: sources.readers,
            mux: &mut mux,
            destination: None,
            root: sources.root,
            fonts: sources.fonts,
            renderer: sources.renderer,
            progress: Some(progress),
        },
        file_audio(sources.readers, sources.draft, sources.root),
    )?;
    let mut writer = mux.into_writer().map_err(ExportError::Mux)?;
    writer.flush().map_err(file)?;
    writer
        .into_inner()
        .map_err(|error| file(error.into_error()))?
        .sync_all()
        .map_err(file)?;
    Ok(result)
}

/// Write only the timeline's sound over `request.range` to `destination`:
/// AAC in an M4A file, or PCM in a WAV file, as `request.audio` names.
/// The file is written beside its destination and moved there when done.
pub fn write_sound(
    sources: &MovieSources<'_>,
    request: &ExportRequest,
    destination: &Path,
    progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<(), ExportError> {
    let codec = request
        .audio
        .as_ref()
        .ok_or(ExportError::AudioLayout)?
        .codec;
    let part = part_path(destination).ok_or(ExportError::Path)?;
    let audio = file_audio(sources.readers, sources.draft, sources.root);
    let written = (|| {
        let out = BufWriter::new(File::create(&part).map_err(file)?);
        let out = match (codec, request.container) {
            (CodecId::Aac, ContainerId::Isobmff) => {
                let mut mux = Mp4Mux::with_brand(out, FileBrand::M4a);
                export_sound(
                    sources.draft,
                    request,
                    sources.encoders,
                    &mut mux,
                    audio,
                    Some(progress),
                )?;
                mux.into_writer().map_err(ExportError::Mux)?
            }
            (CodecId::Pcm, ContainerId::Riff) => {
                let mut mux = WavMux::new(out);
                export_sound(
                    sources.draft,
                    request,
                    sources.encoders,
                    &mut mux,
                    audio,
                    Some(progress),
                )?;
                mux.into_inner()
            }
            _ => return Err(ExportError::Container),
        };
        finish(out)?;
        fs::rename(&part, destination).map_err(file)
    })();
    if written.is_err() {
        let _ = fs::remove_file(&part);
    }
    written
}

fn finish(mut writer: BufWriter<File>) -> Result<(), ExportError> {
    writer.flush().map_err(file)?;
    writer
        .into_inner()
        .map_err(|error| file(error.into_error()))?
        .sync_all()
        .map_err(file)
}

/// The part file of a `.mov` destination: `.cut.mov.part`.
fn is_mov(part: &Path) -> bool {
    part.file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_suffix(".part"))
        .is_some_and(|name| name.to_ascii_lowercase().ends_with(".mov"))
}

fn file(error: std::io::Error) -> ExportError {
    ExportError::File(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use cutlass_codec::{CodecId, ContainerId};
    use cutlass_encode::EncodeFactory;
    use cutlass_eval::{NoFonts, ReaderFactory};
    use cutlass_model::{Draft, FrameRate, TimeRange};

    use super::{MovieSources, is_mov, part_path, write_sound};
    use crate::{AudioExport, ExportRequest};

    fn sound_request(codec: CodecId, container: ContainerId) -> ExportRequest {
        ExportRequest {
            range: TimeRange::new(0, 500_000),
            short_side: 0,
            codec,
            bitrate: 0,
            frame_rate: FrameRate::FPS_30,
            container,
            audio: Some(AudioExport {
                codec,
                bitrate: 128_000,
                sample_rate: 44_100,
                channels: 2,
                limiter: 0,
            }),
            sidecar: None,
            late_by: 0,
        }
    }

    fn written(codec: CodecId, container: ContainerId, name: &str) -> (PathBuf, Vec<u8>) {
        let dir = std::env::temp_dir().join(format!("cutlass-sound-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut encoders = EncodeFactory::new();
        cutlass_aac::register_encoders(&mut encoders);
        cutlass_pcm::register_encoders(&mut encoders);
        let draft = Draft::empty("D");
        let readers = ReaderFactory::new();
        let sources = MovieSources {
            draft: &draft,
            root: None,
            readers: &readers,
            fonts: &NoFonts,
            renderer: None,
            encoders: &encoders,
        };
        let path = dir.join(name);
        let mut calls = 0;
        write_sound(
            &sources,
            &sound_request(codec, container),
            &path,
            &mut |done, total| {
                calls += 1;
                done <= total
            },
        )
        .unwrap();
        assert!(calls > 0);
        assert!(!part_path(&path).unwrap().exists());
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        (path, bytes)
    }

    #[test]
    fn sound_alone_writes_an_m4a_or_a_wav() {
        let (_, m4a) = written(CodecId::Aac, ContainerId::Isobmff, "cut.m4a");
        assert_eq!(&m4a[4..12], b"ftypM4A ");
        let (_, wav) = written(CodecId::Pcm, ContainerId::Riff, "cut.wav");
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
    }

    #[test]
    fn a_mov_destination_names_a_quicktime_movie() {
        assert!(is_mov(Path::new("/films/.cut.MOV.part")));
        assert!(!is_mov(Path::new("/films/.cut.mp4.part")));
    }

    #[test]
    fn a_movie_is_written_to_a_hidden_part_beside_its_destination() {
        assert_eq!(
            part_path(Path::new("/films/cut.mp4")).as_deref(),
            Some(Path::new("/films/.cut.mp4.part"))
        );
        assert_eq!(part_path(Path::new("/")), None);
    }
}

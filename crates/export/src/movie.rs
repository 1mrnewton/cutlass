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
use cutlass_flac_container::FlacMux;
use cutlass_isobmff::{FileBrand, Mp4Mux};
use cutlass_model::Draft;
use cutlass_mp3_container::Mp3Mux;
use cutlass_ogg::OggMux;
use cutlass_render::Renderer;
use cutlass_renderer_software::SoftwareDevice;
use cutlass_riff::WavMux;
use cutlass_still::StillMux;

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
/// the destination ends in `.mov`. A caption sidecar the request asks for
/// is written in the same folder. `progress` hears the frames done and the
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
            // A caption file goes beside the movie.
            destination: request.sidecar.as_ref().and(part.parent()),
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
/// AAC or ALAC in an M4A file, PCM in a WAV file, FLAC in a FLAC file, MP3
/// in an MP3 file, Opus in an Ogg (`.opus`) file, or Vorbis in an Ogg
/// (`.ogg`) file, as `request.audio`
/// names.
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
            (CodecId::Aac | CodecId::Alac, ContainerId::Isobmff) => {
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
            (CodecId::Flac, ContainerId::Flac) => {
                let mut mux = FlacMux::new(out);
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
            (CodecId::Mp3, ContainerId::Mp3) => {
                let mut mux = Mp3Mux::new(out);
                export_sound(
                    sources.draft,
                    request,
                    sources.encoders,
                    &mut mux,
                    audio,
                    Some(progress),
                )?;
                mux.into_inner()
                    .map_err(|error| ExportError::File(error.to_string()))?
            }
            (CodecId::Opus, ContainerId::Ogg) | (CodecId::Vorbis, ContainerId::Ogg) => {
                let mut mux = OggMux::new(out);
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

/// Write `request` to `destination` as an animated GIF or WebP. `looped`
/// plays it over and over; otherwise it plays once. The picture codec must
/// be GIF or WebP, the container the still one, and there is no sound.
pub fn write_animation(
    sources: &MovieSources<'_>,
    request: &ExportRequest,
    looped: bool,
    destination: &Path,
    progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<ExportResult, ExportError> {
    if !matches!(request.codec, CodecId::Gif | CodecId::Webp)
        || request.container != ContainerId::Still
        || request.audio.is_some()
    {
        return Err(ExportError::Container);
    }
    let part = part_path(destination).ok_or(ExportError::Path)?;
    let mut mux = StillMux::new();
    if looped {
        mux.set_loop_count(0);
    }
    let mut device = SoftwareDevice::new();
    let written = export(
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
    )
    .and_then(|result| {
        let bytes = mux.bytes().ok_or(ExportError::Container)?;
        let mut out = BufWriter::new(File::create(&part).map_err(file)?);
        out.write_all(bytes).map_err(file)?;
        finish(out)?;
        fs::rename(&part, destination).map_err(file)?;
        Ok(result)
    });
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

    use super::{MovieSources, is_mov, part_path, write_animation, write_sound};
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
                // Opus is coded at 48 kHz.
                sample_rate: if codec == CodecId::Opus {
                    48_000
                } else {
                    44_100
                },
                channels: 2,
                limiter: 0,
            }),
            sidecar: None,
            late_by: 0,
            constant_rate: false,
            prores: crate::ProresProfile::Standard,
        }
    }

    fn written(codec: CodecId, container: ContainerId, name: &str) -> (PathBuf, Vec<u8>) {
        let dir = std::env::temp_dir().join(format!("cutlass-sound-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut encoders = EncodeFactory::new();
        cutlass_aac::register_encoders(&mut encoders);
        cutlass_alac::register_encoders(&mut encoders);
        cutlass_pcm::register_encoders(&mut encoders);
        cutlass_flac::register_encoders(&mut encoders);
        cutlass_mp3::register_encoders(&mut encoders);
        cutlass_opus::register_encoders(&mut encoders);
        cutlass_vorbis::register_encoders(&mut encoders);
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
    fn sound_alone_writes_an_m4a_a_wav_a_flac_or_an_mp3() {
        let (_, m4a) = written(CodecId::Aac, ContainerId::Isobmff, "cut.m4a");
        assert_eq!(&m4a[4..12], b"ftypM4A ");
        let (_, wav) = written(CodecId::Pcm, ContainerId::Riff, "cut.wav");
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        let (_, flac) = written(CodecId::Flac, ContainerId::Flac, "cut.flac");
        assert_eq!(&flac[..4], b"fLaC");
        let length = flac.len() as u64;
        let demux =
            cutlass_flac_container::FlacDemux::open(std::io::Cursor::new(flac), length).unwrap();
        let info = demux.stream_info();
        assert_eq!((info.bits, info.channels), (16, 2));
        assert!(info.total_samples > 0, "streaminfo written back at finish");
        assert_ne!(info.md5, [0; 16]);
        let (_, mp3) = written(CodecId::Mp3, ContainerId::Mp3, "cut.mp3");
        // Apple's gapless comment in an ID3v2 tag, then the first frame:
        // the Info tag (constant bitrate) with the LAME delay and padding.
        assert_eq!(&mp3[..3], b"ID3");
        let size = mp3[6..10]
            .iter()
            .fold(0usize, |size, byte| (size << 7) | usize::from(*byte));
        let frame = 10 + size;
        assert_eq!(&mp3[frame..frame + 2], &[0xFF, 0xFB], "MPEG-1 Layer III");
        let at = frame + 4 + 32;
        assert_eq!(&mp3[at..at + 4], b"Info");
        let packed = u32::from_be_bytes([0, mp3[at + 141], mp3[at + 142], mp3[at + 143]]);
        assert_eq!(packed >> 12, cutlass_mp3::ENCODER_DELAY);
        let padding = packed & 0xFFF;
        let frames = u32::from_be_bytes([mp3[at + 8], mp3[at + 9], mp3[at + 10], mp3[at + 11]]);
        // Half a second at 44.1 kHz.
        assert_eq!(frames * 1152 - cutlass_mp3::ENCODER_DELAY - padding, 22_050);
    }

    #[test]
    fn sound_alone_writes_a_lossless_m4a() {
        let (_, m4a) = written(CodecId::Alac, ContainerId::Isobmff, "lossless.m4a");
        assert_eq!(&m4a[4..12], b"ftypM4A ");
        let file = cutlass_isobmff::IsoBmffFile::open(std::io::Cursor::new(m4a)).unwrap();
        let streams = cutlass_container::Demux::streams(&file);
        let params = streams[0].params();
        assert_eq!(
            params.codec(),
            &cutlass_codec::StreamCodec::Known(CodecId::Alac)
        );
        assert_eq!(params.channels(), Some(2));
        assert_eq!(params.sample_rate(), Some(44_100));
        // The cookie: 24-bit, as the float mix is kept.
        assert_eq!(params.extradata()[5], 24);
    }

    /// Half a second of Opus in Ogg: the head, the tags, and audio pages
    /// whose last granule is the pre-skip plus exactly half a second.
    #[test]
    fn sound_alone_writes_an_opus_file() {
        let (_, opus) = written(CodecId::Opus, ContainerId::Ogg, "cut.opus");
        assert_eq!(&opus[..4], b"OggS");
        assert_eq!(&opus[28..36], b"OpusHead");
        let length = opus.len() as u64;
        let mut demux = cutlass_ogg::OggDemux::open(std::io::Cursor::new(opus), length).unwrap();
        let stream = cutlass_container::Demux::streams(&demux)[0].clone();
        let pre_skip = stream.trim().encoder_delay().unwrap();
        assert_eq!(stream.edits()[0].duration(), 24_000);
        let mut ticks = 0;
        while let Some(packet) = cutlass_container::Demux::read_packet(&mut demux).unwrap() {
            ticks += packet.timestamp().duration().unwrap();
        }
        assert_eq!(u64::try_from(ticks).unwrap(), pre_skip + 24_000);
    }

    /// Half a second of silence as Vorbis. Reopening it yields that half
    /// second, and nothing before the first sample.
    #[test]
    fn sound_alone_writes_a_vorbis_file() {
        let (_, ogg) = written(CodecId::Vorbis, ContainerId::Ogg, "cut.ogg");
        assert_eq!(&ogg[..4], b"OggS");
        let length = ogg.len() as u64;
        let mut demux = cutlass_ogg::OggDemux::open(std::io::Cursor::new(ogg), length).unwrap();
        let stream = cutlass_container::Demux::streams(&demux)[0].clone();
        assert_eq!(
            stream.params().codec(),
            &cutlass_codec::StreamCodec::Known(CodecId::Vorbis)
        );
        assert_eq!(stream.edits()[0].duration(), 22_050);
        let mut decoder =
            cutlass_vorbis::VorbisDecoder::from_config(stream.params().extradata()).unwrap();
        let mut samples = Vec::new();
        while let Some(packet) = cutlass_container::Demux::read_packet(&mut demux).unwrap() {
            let decoded = decoder.decode(packet.bytes()).unwrap();
            let keep = usize::try_from(packet.timestamp().duration().unwrap()).unwrap() * 2;
            samples.extend(decoded.into_iter().take(keep));
        }
        assert_eq!(samples.len(), 22_050 * 2);
        assert!(samples.iter().all(|sample| sample.abs() < 1e-3));
    }

    #[test]
    fn a_gif_export_writes_a_looping_animation() {
        let dir = std::env::temp_dir().join(format!("cutlass-gif-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut encoders = EncodeFactory::new();
        cutlass_gif::register_encoders(&mut encoders);
        let mut draft = Draft::empty("D");
        draft.duration = cutlass_model::Micros(1_000_000);
        let readers = ReaderFactory::new();
        let sources = MovieSources {
            draft: &draft,
            root: None,
            readers: &readers,
            fonts: &NoFonts,
            renderer: None,
            encoders: &encoders,
        };
        let request = ExportRequest {
            range: TimeRange::new(0, 400_000),
            short_side: 240,
            codec: CodecId::Gif,
            bitrate: 0,
            frame_rate: FrameRate {
                numerator: 10,
                denominator: 1,
            },
            container: ContainerId::Still,
            audio: None,
            sidecar: None,
            late_by: 0,
            constant_rate: false,
            prores: crate::ProresProfile::Standard,
        };
        let path = dir.join("cut.gif");
        let result = write_animation(&sources, &request, true, &path, &mut |_, _| true).unwrap();
        assert_eq!(result.frames.len(), 4);
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(&bytes[..6], b"GIF89a");
        assert!(bytes.windows(11).any(|w| w == b"NETSCAPE2.0"));
        assert_eq!(bytes.last(), Some(&0x3B));
        let mut sound = request.clone();
        sound.audio = Some(crate::MOVIE_SOUND);
        assert!(write_animation(&sources, &sound, false, &path, &mut |_, _| true).is_err());
    }

    #[test]
    fn a_webp_export_writes_an_animation_the_still_container_reopens() {
        use cutlass_container::{Demux, Input};
        use cutlass_decode::Decoder;

        let dir = std::env::temp_dir().join(format!("cutlass-webp-anim-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut encoders = EncodeFactory::new();
        cutlass_webp::register_encoders(&mut encoders);
        let mut draft = Draft::empty("D");
        draft.duration = cutlass_model::Micros(1_000_000);
        let readers = ReaderFactory::new();
        let sources = MovieSources {
            draft: &draft,
            root: None,
            readers: &readers,
            fonts: &NoFonts,
            renderer: None,
            encoders: &encoders,
        };
        let request = ExportRequest {
            range: TimeRange::new(0, 400_000),
            short_side: 240,
            codec: CodecId::Webp,
            bitrate: 0,
            frame_rate: FrameRate {
                numerator: 10,
                denominator: 1,
            },
            container: ContainerId::Still,
            audio: None,
            sidecar: None,
            late_by: 0,
            constant_rate: false,
            prores: crate::ProresProfile::Standard,
        };
        let path = dir.join("cut.webp");
        let result = write_animation(&sources, &request, true, &path, &mut |_, _| true).unwrap();
        assert_eq!(result.frames.len(), 4);
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(
            (&bytes[..4], &bytes[8..16]),
            (&b"RIFF"[..], &b"WEBPVP8X"[..])
        );
        let riff = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        assert_eq!(riff + 8, bytes.len());
        assert_eq!(bytes.windows(4).filter(|w| w == b"ANMF").count(), 4);

        // Reopened through the still container, every frame decodes to
        // the whole canvas.
        let length = bytes.len() as u64;
        let mut demux = cutlass_still::StillDemux::open(Input::random(
            std::io::Cursor::new(bytes),
            length,
            cutlass_still::still_limits(),
        ))
        .unwrap();
        let params = cutlass_decode::CodecParams::video(CodecId::Webp, 0, 0);
        let mut decoder = cutlass_webp::WebpDecoder::open(&params).unwrap();
        let mut frames = 0;
        while let Some(packet) = demux.read_packet().unwrap() {
            let packet = cutlass_decode::Packet::video(packet.into_bytes(), 0, 0, 0, true);
            for picture in decoder.decode_picture(&packet).unwrap() {
                assert_eq!(picture.image.height, 240);
                frames += 1;
            }
        }
        assert_eq!(frames, 4);
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

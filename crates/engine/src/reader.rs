//! Open a container the shell registered, then a decoder from the factory.
//!
//! The factory tries registrations in order. The shell registers the platform
//! decoder for this operating system first, then the Rust decoder for the
//! same id. Probe reads stream parameters and timestamps. It does not decode.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use cutlass_codec::access::{ChannelLayout, CodecParams, PixelFormat};
use cutlass_codec::{CodecId, ContainerId, FrameSize, StreamCodec, Timescale};
use cutlass_container::{Demux, OpenError, ProbeResult, ReadError, SeekTo};
use cutlass_decode::{
    AudioOutput, CodecError, DecodeFactory, Decoder, Output, Packet as AccessPacket, PictureOutput,
    PictureSamples, StillImage,
};
use cutlass_model::{Micros, TimeRange};

use crate::error::EngineError;
use crate::map::{micros_to_ticks, ticks_to_micros};

const HEADER_LIMIT: usize = 4096;
const DECODE_PACKETS: u32 = 64;

/// One stream, still in the file's own parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamReport {
    pub track: u32,
    pub codec: String,
    pub timescale: u32,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
}

/// Duration, byte size, and streams. Duration is absent when the file has no clock.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaInfo {
    pub family: ContainerId,
    pub byte_size: u64,
    pub duration: Option<Micros>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub streams: Vec<StreamReport>,
}

/// A decoded picture before color conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodedFrame {
    Platform {
        width: u32,
        height: u32,
        token: u64,
    },
    Still(Box<StillImage>),
    Coded {
        width: u32,
        height: u32,
        pixel: PixelFormat,
        bytes: Vec<u8>,
    },
}

impl DecodedFrame {
    #[must_use]
    pub fn width(&self) -> u32 {
        match self {
            Self::Platform { width, .. } | Self::Coded { width, .. } => *width,
            Self::Still(image) => image.width,
        }
    }

    #[must_use]
    pub fn height(&self) -> u32 {
        match self {
            Self::Platform { height, .. } | Self::Coded { height, .. } => *height,
            Self::Still(image) => image.height,
        }
    }
}

/// One picture and the source span it covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeldFrame {
    pub start: Micros,
    pub end: Micros,
    pub frame: DecodedFrame,
}

struct Family {
    probe: fn(&[u8]) -> ProbeResult,
    open: fn(&Path) -> Result<Box<dyn Demux>, OpenError>,
}

struct Pipe {
    track: u32,
    timescale: Timescale,
    decoder: Box<dyn Decoder>,
}

/// Registered containers and the decoder factory.
pub struct ReaderFactory {
    families: Vec<Family>,
    decoders: DecodeFactory,
}

impl ReaderFactory {
    #[must_use]
    pub fn new() -> Self {
        Self {
            families: Vec::new(),
            decoders: DecodeFactory::new(),
        }
    }

    pub fn decoders_mut(&mut self) -> &mut DecodeFactory {
        &mut self.decoders
    }

    /// Later families do not hide an earlier header match.
    pub fn register_container(
        &mut self,
        probe: fn(&[u8]) -> ProbeResult,
        open: fn(&Path) -> Result<Box<dyn Demux>, OpenError>,
    ) {
        self.families.push(Family { probe, open });
    }

    /// Duration, size, and streams. No decoder runs.
    pub fn probe(&self, path: &Path) -> Result<MediaInfo, EngineError> {
        let byte_size = file_len(path)?;
        let mut demux = self.open_demux(path)?;
        let mut info = stream_info(demux.as_ref(), byte_size);
        info.duration = duration_of(demux.as_mut())?;
        Ok(info)
    }

    /// Opens the container and a decoder for each picture or audio stream.
    pub fn open(&self, path: &Path) -> Result<Media, EngineError> {
        let info = self.probe(path)?;
        let demux = self.open_demux(path)?;
        let (picture, audio) = open_pipes(&self.decoders, demux.as_ref())?;
        Ok(Media {
            demux,
            picture,
            audio,
            info,
        })
    }

    fn open_demux(&self, path: &Path) -> Result<Box<dyn Demux>, EngineError> {
        let header = header_of(path)?;
        let family = choose(&self.families, &header, file_len(path)?)?;
        (family.open)(path).map_err(from_open)
    }
}

impl Default for ReaderFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// An open file and the decoders for its picture and audio streams.
pub struct Media {
    demux: Box<dyn Demux>,
    picture: Option<Pipe>,
    audio: Option<Pipe>,
    info: MediaInfo,
}

impl Media {
    #[must_use]
    pub fn info(&self) -> &MediaInfo {
        &self.info
    }

    /// The picture that covers `source`, seeking in the file timescale when the stream has a clock.
    pub fn decode_at(
        &mut self,
        source: Micros,
        cover: &TimeRange,
    ) -> Result<HeldFrame, EngineError> {
        let (track, timescale) = {
            let Some(pipe) = self.picture.as_mut() else {
                return Err(EngineError::NoDecoder {
                    codec: "picture".to_owned(),
                });
            };
            pipe.decoder.reset();
            (pipe.track, pipe.timescale)
        };
        if timescale.is_clock() {
            let ticks = micros_to_ticks(source, timescale)?;
            self.demux
                .seek(SeekTo {
                    track,
                    time: ticks,
                    end: None,
                })
                .map_err(from_read)?;
        } else {
            let _ = self.demux.seek(SeekTo {
                track,
                time: 0,
                end: None,
            });
        }
        let mut left = DECODE_PACKETS;
        loop {
            if left == 0 {
                return Err(EngineError::DecodeLimit);
            }
            left -= 1;
            let Some(packet) = self.demux.read_packet().map_err(from_read)? else {
                return Err(EngineError::NotReady);
            };
            if packet.stream() != track {
                continue;
            }
            let access = access_packet(&packet);
            let Some(pipe) = self.picture.as_mut() else {
                return Err(EngineError::NoDecoder {
                    codec: "picture".to_owned(),
                });
            };
            let Some(frame) = push_picture(pipe.decoder.as_mut(), &access)? else {
                continue;
            };
            let (start, end) = coverage(&packet, timescale, cover)?;
            return Ok(HeldFrame { start, end, frame });
        }
    }

    /// The next decoded audio buffer, or [`None`] when the stream ends or there is no audio.
    pub fn pull_audio(&mut self) -> Result<Option<PcmReady>, EngineError> {
        let Some(track) = self.audio.as_ref().map(|pipe| pipe.track) else {
            return Ok(None);
        };
        let mut left = DECODE_PACKETS;
        loop {
            if left == 0 {
                return Err(EngineError::DecodeLimit);
            }
            left -= 1;
            let Some(packet) = self.demux.read_packet().map_err(from_read)? else {
                return Ok(None);
            };
            if packet.stream() != track {
                continue;
            }
            let access = access_packet(&packet);
            let Some(pipe) = self.audio.as_mut() else {
                return Ok(None);
            };
            let outputs = pipe.decoder.push(&access)?;
            for output in outputs {
                if let Output::Audio(audio) = output {
                    return Ok(Some(audible(audio)?));
                }
            }
        }
    }
}

/// PCM a decoder returned, with delay samples removed.
#[derive(Clone, Debug, PartialEq)]
pub struct PcmReady {
    pub pcm: cutlass_codec::access::Pcm,
}

fn audible(output: AudioOutput) -> Result<PcmReady, EngineError> {
    if output.skip_leading == 0 && output.skip_trailing == 0 {
        return Ok(PcmReady { pcm: output.pcm });
    }
    let Some(samples) = output.pcm.as_f32() else {
        return Err(EngineError::Layout);
    };
    let channels = output.pcm.layout.len();
    let lead = usize::try_from(output.skip_leading)
        .unwrap_or(usize::MAX)
        .saturating_mul(channels);
    let tail = usize::try_from(output.skip_trailing)
        .unwrap_or(usize::MAX)
        .saturating_mul(channels);
    if lead.saturating_add(tail) >= samples.len() {
        return Ok(PcmReady {
            pcm: cutlass_codec::access::Pcm::from_f32(
                output.pcm.rate,
                output.pcm.layout.clone(),
                Vec::new(),
            )
            .map_err(|_| EngineError::Layout)?,
        });
    }
    let end = samples.len() - tail;
    let pcm = cutlass_codec::access::Pcm::from_f32(
        output.pcm.rate,
        output.pcm.layout.clone(),
        samples[lead..end].to_vec(),
    )
    .map_err(|_| EngineError::Layout)?;
    Ok(PcmReady { pcm })
}

fn coverage(
    packet: &cutlass_codec::Packet,
    timescale: Timescale,
    cover: &TimeRange,
) -> Result<(Micros, Micros), EngineError> {
    if !timescale.is_clock() {
        let end = cover.start.0.saturating_add(cover.duration.0);
        return Ok((cover.start, Micros(end)));
    }
    let Some(start_ticks) = packet.timestamp().presentation_or_decode() else {
        let end = cover.start.0.saturating_add(cover.duration.0);
        return Ok((cover.start, Micros(end)));
    };
    let start = ticks_to_micros(start_ticks, timescale)?;
    let duration = packet.timestamp().duration().unwrap_or(0);
    let end_ticks = start_ticks.saturating_add(duration.max(1));
    let end = ticks_to_micros(end_ticks, timescale)?;
    Ok((start, end))
}

fn push_picture(
    decoder: &mut dyn Decoder,
    packet: &AccessPacket,
) -> Result<Option<DecodedFrame>, EngineError> {
    match decoder.decode_picture(packet) {
        Ok(frames) => Ok(frames
            .into_iter()
            .next()
            .map(|still| DecodedFrame::Still(Box::new(still.image)))),
        Err(CodecError::Unsupported {
            what: "picture output",
        }) => {
            let outputs = decoder.push(packet)?;
            Ok(outputs.into_iter().find_map(coded_output))
        }
        Err(error) => Err(EngineError::Codec(error)),
    }
}

fn coded_output(output: Output) -> Option<DecodedFrame> {
    match output {
        Output::Picture(picture) => Some(frame_from_picture(picture)),
        Output::Audio(_) => None,
    }
}

fn frame_from_picture(picture: PictureOutput) -> DecodedFrame {
    let info = picture.info();
    match picture.samples() {
        PictureSamples::Platform { token } => DecodedFrame::Platform {
            width: info.width,
            height: info.height,
            token,
        },
        PictureSamples::Cpu(bytes) => DecodedFrame::Coded {
            width: info.width,
            height: info.height,
            pixel: info.pixel,
            bytes: bytes.to_vec(),
        },
    }
}

fn access_packet(packet: &cutlass_codec::Packet) -> AccessPacket {
    AccessPacket {
        stream: packet.stream(),
        data: packet.bytes().to_vec(),
        pts: packet.timestamp().presentation_or_decode().unwrap_or(0),
        duration: packet.timestamp().duration().unwrap_or(0),
        timescale: packet.timescale().ticks_per_second(),
        keyframe: packet.keyframe(),
        side: Vec::new(),
    }
}

fn open_pipes(
    factory: &DecodeFactory,
    demux: &dyn Demux,
) -> Result<(Option<Pipe>, Option<Pipe>), EngineError> {
    let mut picture = None;
    let mut audio = None;
    for stream in demux.streams() {
        let sized = stream.params().size();
        let rate = stream.params().sample_rate();
        if sized.is_none() && rate.is_none() {
            continue;
        }
        let codec = stream.params().codec();
        let StreamCodec::Known(id) = codec else {
            return Err(EngineError::UnknownCodec {
                tag: unknown_tag(codec),
            });
        };
        let params = params_for(
            stream.params().extradata(),
            sized,
            rate,
            stream.params().channels(),
            *id,
        )?;
        let decoder = match factory.open(&params) {
            Ok(decoder) => decoder,
            Err(CodecError::Unsupported { .. }) => {
                return Err(EngineError::NoDecoder {
                    codec: format!("{id:?}"),
                });
            }
            Err(error) => return Err(EngineError::Codec(error)),
        };
        let pipe = Pipe {
            track: stream.id().get(),
            timescale: stream.timescale(),
            decoder,
        };
        if sized.is_some() && picture.is_none() {
            picture = Some(pipe);
        } else if rate.is_some() && audio.is_none() {
            audio = Some(pipe);
        }
    }
    Ok((picture, audio))
}

fn params_for(
    extra: &[u8],
    size: Option<FrameSize>,
    rate: Option<u32>,
    channels: Option<u16>,
    id: CodecId,
) -> Result<CodecParams, EngineError> {
    if let Some(size) = size {
        let pixel = match id {
            CodecId::Jpeg | CodecId::Png | CodecId::Webp | CodecId::Gif => PixelFormat::Rgba8,
            _ => PixelFormat::Yuv420p,
        };
        let mut params = CodecParams::with_format(id, size.width(), size.height(), pixel);
        params.extra = extra.to_vec();
        return Ok(params);
    }
    let Some(rate) = rate else {
        return Err(EngineError::NotMedia);
    };
    let channels = channels.unwrap_or(0);
    let layout = match channels {
        1 => ChannelLayout::mono(),
        2 => ChannelLayout::stereo(),
        _ => return Err(EngineError::Layout),
    };
    let mut params = CodecParams::audio(id, rate, layout);
    params.extra = extra.to_vec();
    Ok(params)
}

fn unknown_tag(codec: &StreamCodec) -> String {
    match codec {
        StreamCodec::Known(id) => format!("{id:?}"),
        StreamCodec::Unknown(tag) => String::from_utf8_lossy(tag.tag()).into_owned(),
    }
}

fn stream_info(demux: &dyn Demux, byte_size: u64) -> MediaInfo {
    let streams: Vec<StreamReport> = demux
        .streams()
        .iter()
        .map(|stream| {
            let size = stream.params().size();
            StreamReport {
                track: stream.id().get(),
                codec: unknown_tag(stream.params().codec()),
                timescale: stream.timescale().ticks_per_second(),
                width: size.map(FrameSize::width),
                height: size.map(FrameSize::height),
                sample_rate: stream.params().sample_rate(),
                channels: stream.params().channels(),
            }
        })
        .collect();
    let picture = streams.iter().find(|stream| stream.width.is_some());
    MediaInfo {
        family: demux.family(),
        byte_size,
        duration: None,
        width: picture.and_then(|stream| stream.width),
        height: picture.and_then(|stream| stream.height),
        streams,
    }
}

fn duration_of(demux: &mut dyn Demux) -> Result<Option<Micros>, EngineError> {
    let mut end = None;
    while let Some(packet) = demux.read_packet().map_err(from_read)? {
        let timescale = packet.timescale();
        if !timescale.is_clock() {
            continue;
        }
        let Some(start) = packet.timestamp().presentation_or_decode() else {
            continue;
        };
        let duration = packet.timestamp().duration().unwrap_or(0);
        let ticks = start.saturating_add(duration);
        let micros = ticks_to_micros(ticks, timescale)?;
        end = Some(end.map_or(micros.0, |current: i64| current.max(micros.0)));
    }
    Ok(end.map(Micros))
}

fn choose<'a>(families: &'a [Family], header: &[u8], len: u64) -> Result<&'a Family, EngineError> {
    let mut accepted = None;
    let mut need = 0_usize;
    let mut waiting = false;
    for family in families {
        match (family.probe)(header) {
            ProbeResult::Accept { .. } => {
                if accepted.is_none() {
                    accepted = Some(family);
                }
            }
            ProbeResult::NeedBytes { need: more, .. } => {
                waiting = true;
                need = need.max(more);
            }
            ProbeResult::Refuse { .. } => {}
        }
    }
    if let Some(family) = accepted {
        return Ok(family);
    }
    if waiting {
        return Err(EngineError::Truncated {
            expected: u64::try_from(need).unwrap_or(u64::MAX),
            got: u64::try_from(header.len()).unwrap_or(u64::MAX).min(len),
        });
    }
    if families.is_empty() || header.is_empty() && len == 0 {
        return Err(if len == 0 {
            EngineError::Truncated {
                expected: 1,
                got: 0,
            }
        } else {
            EngineError::UnknownContainer
        });
    }
    Err(EngineError::UnknownContainer)
}

fn header_of(path: &Path) -> Result<Vec<u8>, EngineError> {
    let mut file = open_file(path)?;
    let len = file_len(path)?;
    let cap = usize::try_from(len.min(u64::try_from(HEADER_LIMIT).unwrap_or(u64::MAX)))
        .unwrap_or(HEADER_LIMIT);
    let mut buf = vec![0_u8; cap];
    let mut filled = 0;
    while filled < buf.len() {
        match file.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(read) => filled += read,
            Err(error) => {
                return Err(EngineError::Io {
                    message: error.to_string(),
                });
            }
        }
    }
    buf.truncate(filled);
    Ok(buf)
}

fn file_len(path: &Path) -> Result<u64, EngineError> {
    let file = open_file(path)?;
    let meta = file.metadata().map_err(|error| EngineError::Io {
        message: error.to_string(),
    })?;
    if !meta.is_file() {
        return Err(EngineError::Io {
            message: "not a file".to_owned(),
        });
    }
    Ok(meta.len())
}

fn open_file(path: &Path) -> Result<File, EngineError> {
    match File::open(path) {
        Ok(file) => Ok(file),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Err(EngineError::MissingFile {
                path: path.to_path_buf(),
            })
        }
        Err(error) => Err(EngineError::Io {
            message: error.to_string(),
        }),
    }
}

fn from_open(error: OpenError) -> EngineError {
    match error {
        OpenError::Truncated { expected, got } => EngineError::Truncated { expected, got },
        OpenError::HostileSize { declared, limit } => EngineError::Hostile { declared, limit },
        OpenError::Unrecognized { .. } => EngineError::UnknownContainer,
        other => EngineError::Read {
            message: other.to_string(),
        },
    }
}

fn from_read(error: ReadError) -> EngineError {
    match error {
        ReadError::Truncated { expected, got } => EngineError::Truncated { expected, got },
        ReadError::HostileSize { declared, limit } => EngineError::Hostile { declared, limit },
        other => EngineError::Read {
            message: other.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_file_and_an_unknown_header_are_named_errors() {
        let factory = ReaderFactory::new();
        let missing = factory
            .probe(Path::new("/no/such/cutlass-engine-media"))
            .expect_err("missing");
        assert!(matches!(missing, EngineError::MissingFile { .. }));

        let path =
            std::env::temp_dir().join(format!("cutlass-engine-unknown-{}", std::process::id()));
        std::fs::write(&path, b"not media").expect("write");
        let unknown = factory.probe(&path).expect_err("unknown");
        assert_eq!(unknown, EngineError::UnknownContainer);
        std::fs::remove_file(&path).expect("remove");
    }
}

//! Open a container the shell registered, then a decoder from the factory.
//!
//! The factory tries registrations in order. The shell registers the platform
//! decoder for this operating system first, then the Rust decoder for the
//! same id. Probe reads stream parameters and timestamps. It does not decode.
//! [`Media::decode_at`] seeks a picture. [`Media::audio_from`] seeks a PCM window.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cutlass_codec::access::{
    ChannelLayout, CodecParams, ColorPrimaries, PixelFormat, SampleFormat, StandardLayout, Transfer,
};
use cutlass_codec::{CodecId, ContainerId, FrameSize, StreamCodec, Timescale};
use cutlass_container::{Demux, DisplayQuarter, OpenError, ProbeResult, ReadError, SeekTo};
use cutlass_decode::{
    AudioOutput, CodecError, DecodeFactory, Decoder, Output, Packet as AccessPacket, PictureOutput,
    PictureSamples, SideData, StillImage, Surface,
};
use cutlass_model::{Micros, TimeRange};

use crate::error::EngineError;
use crate::map::{micros_to_ticks, micros_to_ticks_floor, ticks_to_micros};

const HEADER_LIMIT: usize = 4096;
/// Picture or audio packets one seek may decode before giving up.
const DECODE_PACKETS: u32 = 64;
/// Pictures one forward scene scan will decode. A longer file is read from its start.
const MAX_SCAN_PICTURES: u32 = 4_000;
/// How far past the last picture [`Media::decode_at`] keeps decoding forward
/// instead of seeking. A longer jump seeks to the keyframe before the time.
const FORWARD_REACH: i64 = 2_000_000;
/// How far before the asked time an audio seek lands, so the decoder is
/// primed by the time the asked sample comes out.
const AUDIO_PREROLL: i64 = 100_000;
/// Pictures one [`Media::decode_at`] may fail to decode and pass over. After
/// a seek to an open-GOP keyframe, the leading pictures reference frames
/// before the seek and cannot be decoded.
const SKIPPED_PICTURES: u32 = 16;
/// Malformed audio packets one pull may pass over. A bad syncframe must
/// not end the rest of the file. The decoder is left as it was.
const SKIPPED_AUDIO: u32 = 16;
/// Samples from another stream that may sit ahead of the wanted one.
/// A movie can store an audio chunk before the next picture.
const OTHER_STREAM_PACKETS: u32 = 4_096;
/// PCM frames one [`Media::audio_from`] call will decode.
pub(crate) const MAX_AUDIO_FRAMES: u32 = 480_000;
/// Full-scale positive [`i16`]. The mix uses the same divisor.
const I16_FULL: f32 = 32_767.0;

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
    /// Clockwise quarter turn of the picture stream. [`DisplayQuarter::Zero`] when
    /// the file has no picture, or the matrix is not a quarter turn.
    pub display: DisplayQuarter,
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
        primaries: ColorPrimaries,
        transfer: Transfer,
        /// Shared, so a cache, the read-ahead, and a GPU layer hold one
        /// picture without copying it.
        bytes: Arc<Vec<u8>>,
    },
    /// A platform GPU buffer that stands for 8-bit 4:2:0. The GPU path
    /// samples it in place. [`Self::on_cpu`] gives the bytes for every
    /// other use.
    Surface {
        width: u32,
        height: u32,
        primaries: ColorPrimaries,
        transfer: Transfer,
        surface: Surface,
    },
}

impl DecodedFrame {
    /// This frame with CPU samples: a surface becomes the [`Self::Coded`]
    /// 4:2:0 it stands for, copied out once and shared after that. Every
    /// other frame is itself.
    pub fn on_cpu(&self) -> Result<std::borrow::Cow<'_, Self>, EngineError> {
        match self {
            Self::Surface {
                width,
                height,
                primaries,
                transfer,
                surface,
            } => Ok(std::borrow::Cow::Owned(Self::Coded {
                width: *width,
                height: *height,
                pixel: PixelFormat::Yuv420p,
                primaries: *primaries,
                transfer: *transfer,
                bytes: surface.yuv420p()?,
            })),
            Self::Platform { .. } | Self::Still(_) | Self::Coded { .. } => {
                Ok(std::borrow::Cow::Borrowed(self))
            }
        }
    }

    #[must_use]
    pub fn width(&self) -> u32 {
        match self {
            Self::Platform { width, .. }
            | Self::Coded { width, .. }
            | Self::Surface { width, .. } => *width,
            Self::Still(image) => image.width,
        }
    }

    #[must_use]
    pub fn height(&self) -> u32 {
        match self {
            Self::Platform { height, .. }
            | Self::Coded { height, .. }
            | Self::Surface { height, .. } => *height,
            Self::Still(image) => image.height,
        }
    }

    /// The memory the picture holds: its bytes or samples, or for a platform
    /// buffer the 4:2:0 picture it stands for.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Coded { bytes, .. } => bytes.len(),
            Self::Platform { width, height, .. } | Self::Surface { width, height, .. } => {
                *width as usize * *height as usize * 3 / 2
            }
            Self::Still(image) => image.samples.len() * 2,
        }
    }
}

/// What a steered decode does after each picture. See
/// [`Media::decode_steered`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Steer {
    /// Keep going to the target.
    Go,
    /// Go to this time instead. A time before the picture just decoded
    /// cannot be reached going forward, so it stops the decode.
    Retarget(Micros),
    /// Stop here.
    Stop,
}

/// One picture and the source span it covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeldFrame {
    pub start: Micros,
    pub end: Micros,
    pub frame: DecodedFrame,
}

#[derive(Clone)]
struct Family {
    probe: fn(&[u8]) -> ProbeResult,
    open: fn(&Path) -> Result<Box<dyn Demux>, OpenError>,
}

struct Pipe {
    track: u32,
    timescale: Timescale,
    decoder: Box<dyn Decoder>,
    /// Pictures the decoder returned that a pull has not handed out yet.
    ready: VecDeque<Ready>,
    /// The stream ended and the decoder was drained.
    drained: bool,
    /// The last picture packet's duration, for pictures a drain releases.
    last_duration: i64,
    /// Audio buffers a packet returned after the first, not handed out yet.
    heard: VecDeque<AudioOutput>,
    /// Stream ticks before the stream's time zero, from its edit list.
    lead: i64,
    /// The container's encoder delay and padding, as decoder side data.
    gapless: Option<SideData>,
    /// No packet has been pushed since open or the last reset.
    fresh: bool,
}

impl Pipe {
    fn new(track: u32, timescale: Timescale, decoder: Box<dyn Decoder>, lead: i64) -> Self {
        Self {
            track,
            timescale,
            decoder,
            ready: VecDeque::new(),
            drained: false,
            last_duration: 0,
            heard: VecDeque::new(),
            lead,
            gapless: None,
            fresh: true,
        }
    }

    /// The access packet for `packet`. The first packet after open or a
    /// reset carries the container's trim, so a decoder that knows its own
    /// delay can mark the priming and padding as skipped.
    fn access(&mut self, packet: &cutlass_codec::Packet) -> AccessPacket {
        let mut access = access_packet(packet);
        if std::mem::take(&mut self.fresh)
            && let Some(gapless) = &self.gapless
        {
            access.side.push(gapless.clone());
        }
        access
    }

    /// `lead` in `timescale` ticks per second.
    fn lead_in(&self, timescale: u32) -> i64 {
        rescale(self.lead, self.timescale.ticks_per_second(), timescale)
    }

    /// Drop references, held pictures, and the end of stream before a seek.
    fn reset(&mut self) {
        self.decoder.reset();
        self.ready.clear();
        self.heard.clear();
        self.drained = false;
        self.fresh = true;
    }
}

/// One decoded picture and the stream ticks it starts at, when known.
struct Ready {
    frame: DecodedFrame,
    pts: Option<i64>,
    duration: i64,
}

/// Registered containers and the decoder factory.
#[derive(Clone)]
pub struct ReaderFactory {
    families: Vec<Family>,
    decoders: DecodeFactory,
    /// Files [`Self::open`] reads in place of others: a source's proxy.
    stand_ins: Arc<HashMap<PathBuf, PathBuf>>,
}

impl ReaderFactory {
    #[must_use]
    pub fn new() -> Self {
        Self {
            families: Vec::new(),
            decoders: DecodeFactory::new(),
            stand_ins: Arc::default(),
        }
    }

    /// The same readers, opening each file of `stand_ins` in place of the
    /// source it is keyed by. A preview opens proxies this way; probing
    /// still reads the source, and export keeps readers without them.
    #[must_use]
    pub fn with_stand_ins(&self, stand_ins: HashMap<PathBuf, PathBuf>) -> Self {
        Self {
            stand_ins: Arc::new(stand_ins),
            ..self.clone()
        }
    }

    /// The file [`Self::open`] reads for `path`.
    fn stand_in<'a>(&'a self, path: &'a Path) -> &'a Path {
        if self.stand_ins.is_empty() {
            return path;
        }
        let found = match self.stand_ins.get(path) {
            Some(found) => Some(found),
            None => path
                .canonicalize()
                .ok()
                .and_then(|canonical| self.stand_ins.get(&canonical)),
        };
        // A proxy since removed leaves the source to play.
        found
            .filter(|found| found.is_file())
            .map_or(path, PathBuf::as_path)
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

    /// Whether `path` has an audio stream. This reads the header and does not decode.
    pub fn carries_sound(&self, path: &Path) -> Result<bool, EngineError> {
        let info = self.probe(path)?;
        Ok(info
            .streams
            .iter()
            .any(|stream| stream.sample_rate.is_some_and(|rate| rate > 0)))
    }

    /// Opens the container and a decoder for each picture or audio stream.
    pub fn open(&self, path: &Path) -> Result<Media, EngineError> {
        let path = self.stand_in(path);
        let info = self.probe(path)?;
        let demux = self.open_demux(path)?;
        let (picture, audio) = open_pipes(&self.decoders, demux.as_ref())?;
        Ok(Media {
            demux,
            picture,
            audio,
            info,
            cursor: None,
            run: None,
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
    /// Where the last [`Media::decode_at`] left the picture stream.
    cursor: Option<Cursor>,
    /// Decoded audio past the last [`Media::audio_from`], for the next one.
    run: Option<AudioRun>,
}

/// Decoded audio the next [`Media::audio_from`] continues from.
struct AudioRun {
    /// Sample index, at `rate`, of the first held frame.
    next: i64,
    rate: u32,
    layout: ChannelLayout,
    /// Interleaved frames from `next` on.
    samples: Vec<f32>,
}

impl AudioRun {
    fn frames(&self) -> usize {
        self.samples.len() / self.layout.len().max(1)
    }

    /// Sample index just past the last held frame.
    fn end(&self) -> i64 {
        self.next
            .saturating_add(i64::try_from(self.frames()).unwrap_or(i64::MAX))
    }

    /// Hold `frames` frames of silence after the last held frame.
    fn pad(&mut self, frames: i64) {
        let frames = usize::try_from(frames.max(0)).unwrap_or(0);
        let len = self
            .samples
            .len()
            .saturating_add(frames.saturating_mul(self.layout.len()));
        self.samples.resize(len, 0.0);
    }

    /// Drop up to `frames` leading frames.
    fn skip(&mut self, frames: i64) {
        let frames = usize::try_from(frames.max(0))
            .unwrap_or(usize::MAX)
            .min(self.frames());
        self.samples.drain(..frames * self.layout.len());
        self.next = self
            .next
            .saturating_add(i64::try_from(frames).unwrap_or(i64::MAX));
    }
}

/// The picture stream after a [`Media::decode_at`]. A time inside `last`
/// returns it again. A later time continues from here without a seek.
struct Cursor {
    last: HeldFrame,
    /// A picture decoded past `last`, handed out first next time.
    ahead: Option<HeldFrame>,
}

impl Media {
    #[must_use]
    pub fn info(&self) -> &MediaInfo {
        &self.info
    }

    /// The picture that covers `source`, seeking in the file timescale when the stream has a clock.
    ///
    /// A time at or past the last picture, and within [`FORWARD_REACH`] of it,
    /// decodes forward from where the stream is. Any other time seeks to the
    /// keyframe before it. Either way pictures are decoded until one covers
    /// `source`. A time before the first picture is that first picture. A time
    /// past the last one is the last one.
    pub fn decode_at(
        &mut self,
        source: Micros,
        cover: &TimeRange,
    ) -> Result<HeldFrame, EngineError> {
        self.decode_steered(source, cover, &mut |_| Steer::Go)
    }

    /// [`Self::decode_at`], asking `steer` after each picture decoded on the
    /// way. It can move the target to a later time, which the decode then
    /// stops at, or stop the decode. A later time more than
    /// [`FORWARD_REACH`] past the picture just decoded seeks to the keyframe
    /// before it instead of decoding every picture on the way. A stopped
    /// decode is
    /// [`EngineError::Interrupted`], and the stream stays where it got to, so
    /// a later time continues from there without a seek.
    pub fn decode_steered(
        &mut self,
        mut source: Micros,
        cover: &TimeRange,
        steer: &mut dyn FnMut(&HeldFrame) -> Steer,
    ) -> Result<HeldFrame, EngineError> {
        if let Some(cursor) = &self.cursor
            && cursor.last.start.0 <= source.0
            && source.0 < cursor.last.end.0
        {
            return Ok(cursor.last.clone());
        }
        let cursor = self.cursor.take().filter(|cursor| {
            cursor.last.end.0 <= source.0 && source.0 - cursor.last.end.0 <= FORWARD_REACH
        });
        // Resuming, the last picture still shows until the next one starts.
        let (mut ahead, mut previous) = match cursor {
            Some(cursor) => (cursor.ahead, Some(cursor.last)),
            None => {
                self.restart_picture(source)?;
                (None, None)
            }
        };
        let mut skipped = 0_u32;
        for _ in 0..MAX_SCAN_PICTURES {
            let next = match ahead.take() {
                Some(held) => Some(held),
                None => match self.pull_picture(cover) {
                    Ok(next) => next,
                    Err(EngineError::Codec(CodecError::Malformed { .. }))
                        if skipped < SKIPPED_PICTURES =>
                    {
                        skipped = skipped.saturating_add(1);
                        continue;
                    }
                    Err(error) => return Err(error),
                },
            };
            let Some(held) = next else {
                let last = previous.ok_or(EngineError::NotReady)?;
                return Ok(self.keep(last, None));
            };
            match steer(&held) {
                Steer::Go => {}
                Steer::Retarget(later) if later.0.saturating_sub(held.end.0) > FORWARD_REACH => {
                    source = later;
                    self.restart_picture(source)?;
                    previous = None;
                    continue;
                }
                Steer::Retarget(later) if later.0 >= held.start.0 => source = later,
                Steer::Retarget(_) | Steer::Stop => {
                    self.keep(held, None);
                    return Err(EngineError::Interrupted);
                }
            }
            if held.start.0 > source.0 {
                return Ok(match previous {
                    Some(mut previous) => {
                        // A gap before the next picture still shows this one.
                        previous.end = Micros(previous.end.0.max(held.start.0));
                        self.keep(previous, Some(held))
                    }
                    None => self.keep(held, None),
                });
            }
            if source.0 < held.end.0 {
                return Ok(self.keep(held, None));
            }
            previous = Some(held);
        }
        Err(EngineError::DecodeLimit)
    }

    /// Drop what the picture decoder holds and seek to the keyframe before
    /// `source`.
    fn restart_picture(&mut self, source: Micros) -> Result<(), EngineError> {
        let (track, timescale) = {
            let Some(pipe) = self.picture.as_mut() else {
                return Err(EngineError::NoDecoder {
                    codec: "picture".to_owned(),
                });
            };
            pipe.reset();
            (pipe.track, pipe.timescale)
        };
        self.seek_picture(source, track, timescale)
    }

    /// Remember `last` for the next [`Media::decode_at`] and return it.
    fn keep(&mut self, last: HeldFrame, ahead: Option<HeldFrame>) -> HeldFrame {
        let returned = last.clone();
        self.cursor = Some(Cursor { last, ahead });
        returned
    }

    /// Pictures from `range`, at least `gap` microseconds apart, at most `limit`.
    ///
    /// One seek starts the read. Later pictures stay in file order. The stream
    /// ending is an empty or short list, not an error.
    pub fn pictures_across(
        &mut self,
        range: &TimeRange,
        gap: i64,
        limit: usize,
    ) -> Result<Vec<HeldFrame>, EngineError> {
        self.cursor = None;
        if limit == 0 || range.duration.0 <= 0 {
            return Err(EngineError::Window);
        }
        let (track, timescale) = {
            let Some(pipe) = self.picture.as_mut() else {
                return Err(EngineError::NoDecoder {
                    codec: "picture".to_owned(),
                });
            };
            pipe.reset();
            (pipe.track, pipe.timescale)
        };
        self.seek_picture(range.start, track, timescale)?;
        let end = range.start.0.saturating_add(range.duration.0);
        let mut kept = Vec::new();
        let mut last = None;
        let mut decoded = 0_u32;
        while kept.len() < limit {
            if decoded == MAX_SCAN_PICTURES {
                break;
            }
            decoded = decoded.saturating_add(1);
            let held = match self.pull_picture(range) {
                Ok(Some(held)) => held,
                Ok(None) => break,
                Err(_) if !kept.is_empty() => break,
                Err(error) => return Err(error),
            };
            if held.start.0 >= end {
                break;
            }
            if held.end.0 <= range.start.0 {
                continue;
            }
            if last.is_some_and(|previous: i64| held.start.0 < previous.saturating_add(gap.max(0)))
            {
                continue;
            }
            last = Some(held.start.0);
            kept.push(held);
        }
        Ok(kept)
    }

    fn seek_picture(
        &mut self,
        source: Micros,
        track: u32,
        timescale: Timescale,
    ) -> Result<(), EngineError> {
        self.run = None;
        if timescale.is_clock() {
            let lead = self.picture.as_ref().map_or(0, |pipe| pipe.lead);
            let ticks = micros_to_ticks_floor(source, timescale)?.saturating_add(lead);
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
        Ok(())
    }

    /// The next picture in display order.
    ///
    /// A push can return several pictures. The ones after the first wait in
    /// the pipe for the next pull. At the end of the stream the decoder is
    /// drained once, so a picture still held for reorder is not lost. A
    /// picture covers the span from its own timestamp, not from the packet
    /// that released it.
    fn pull_picture(&mut self, cover: &TimeRange) -> Result<Option<HeldFrame>, EngineError> {
        let mut pictures = 0_u32;
        let mut others = 0_u32;
        loop {
            let Some(pipe) = self.picture.as_mut() else {
                return Err(EngineError::NoDecoder {
                    codec: "picture".to_owned(),
                });
            };
            let (track, timescale, lead) = (pipe.track, pipe.timescale, pipe.lead);
            if let Some(ready) = pipe.ready.pop_front() {
                let pts = ready.pts.map(|pts| pts.saturating_sub(lead));
                let (start, end) = coverage(pts, ready.duration, timescale, cover)?;
                return Ok(Some(HeldFrame {
                    start,
                    end,
                    frame: ready.frame,
                }));
            }
            if pictures == DECODE_PACKETS {
                return Err(EngineError::DecodeLimit);
            }
            let Some(packet) = self.demux.read_packet().map_err(from_read)? else {
                let Some(pipe) = self.picture.as_mut() else {
                    return Ok(None);
                };
                if pipe.drained {
                    return Ok(None);
                }
                pipe.drained = true;
                let outputs = match pipe.decoder.drain() {
                    Ok(outputs) => outputs,
                    Err(CodecError::Unsupported { .. }) => Vec::new(),
                    Err(error) => return Err(EngineError::Codec(error)),
                };
                let duration = pipe.last_duration;
                for (frame, pts) in outputs
                    .into_iter()
                    .filter_map(|output| coded_output(output, timescale))
                {
                    pipe.ready.push_back(Ready {
                        frame,
                        pts,
                        duration,
                    });
                }
                continue;
            };
            if packet.stream() != track {
                if others == OTHER_STREAM_PACKETS {
                    return Err(EngineError::DecodeLimit);
                }
                others = others.saturating_add(1);
                continue;
            }
            pictures = pictures.saturating_add(1);
            let access = access_packet(&packet);
            let Some(pipe) = self.picture.as_mut() else {
                return Err(EngineError::NoDecoder {
                    codec: "picture".to_owned(),
                });
            };
            let packet_pts = packet.timestamp().presentation_or_decode();
            let duration = packet.timestamp().duration().unwrap_or(0);
            pipe.last_duration = duration;
            for (frame, pts) in push_picture(pipe.decoder.as_mut(), &access, timescale)? {
                pipe.ready.push_back(Ready {
                    frame,
                    pts: pts.or(packet_pts),
                    duration,
                });
            }
        }
    }

    /// PCM covering `source`, at the file rate, for at most `frames` sample frames.
    ///
    /// The mix resamples and stretches this buffer. Integer PCM is full-scale
    /// `f32` before it returns. A file with no audio stream is [`EngineError::NotMedia`].
    ///
    /// A call that starts where the last one ended continues the decoded
    /// stream: no seek, no decoder reset, so consecutive windows join without
    /// a seam. Any other time seeks [`AUDIO_PREROLL`] early and trims to the
    /// asked sample. Audio before the stream's first sample is silence.
    pub fn audio_from(
        &mut self,
        source: Micros,
        frames: u32,
    ) -> Result<cutlass_codec::access::Pcm, EngineError> {
        if frames == 0 || frames > MAX_AUDIO_FRAMES {
            return Err(EngineError::Window);
        }
        self.cursor = None;
        let (track, timescale, lead) = {
            let Some(pipe) = self.audio.as_ref() else {
                return Err(EngineError::NotMedia);
            };
            (pipe.track, pipe.timescale, pipe.lead)
        };
        let rate = self
            .info
            .streams
            .iter()
            .find(|stream| stream.track == track)
            .and_then(|stream| stream.sample_rate)
            .filter(|rate| *rate > 0);
        let Some(rate) = rate.filter(|_| timescale.is_clock()) else {
            return self.audio_from_start(source, frames);
        };
        let mut rate = self.run.as_ref().map_or(rate, |run| run.rate);
        let mut want = sample_at(source, rate);
        let continues = self
            .run
            .as_ref()
            .is_some_and(|run| (run.next - want).abs() <= 1);
        if continues {
            want = self.run.as_ref().map_or(want, |run| run.next);
        } else {
            self.run = None;
            if let Some(pipe) = self.audio.as_mut() {
                pipe.reset();
            }
            let early = Micros(source.0.saturating_sub(AUDIO_PREROLL).max(0));
            let ticks = micros_to_ticks_floor(early, timescale)?.saturating_add(lead);
            self.demux
                .seek(SeekTo {
                    track,
                    time: ticks,
                    end: None,
                })
                .map_err(from_read)?;
        }
        let needed = i64::from(frames);
        let mut empty = 0_u32;
        loop {
            if let Some(run) = self.run.as_mut() {
                if run.next < want {
                    run.skip(want - run.next);
                }
                let end = run.next + i64::try_from(run.frames()).unwrap_or(i64::MAX);
                if run.next <= want && end >= want + needed {
                    break;
                }
            }
            let Some(output) = self.next_audio()? else {
                break;
            };
            let lead = self
                .audio
                .as_ref()
                .map_or(0, |pipe| pipe.lead_in(output.timescale));
            let start = output
                .pts
                .saturating_sub(lead)
                .checked_mul(i64::from(output.pcm.rate))
                .and_then(|value| value.checked_div(i64::from(output.timescale.max(1))))
                .unwrap_or(0)
                .saturating_add(i64::from(output.skip_leading));
            // Sample ticks one stream tick may round away.
            let slack = i64::from(output.pcm.rate.div_ceil(output.timescale.max(1))) + 1;
            if let Some(run) = self.run.as_mut() {
                if run.frames() == 0 && run.next < want {
                    // Nothing held: the frames before the window are not kept.
                    run.next = want.min(start);
                }
                let end = run.end();
                let limit = want.saturating_add(needed).max(end);
                if start > limit.saturating_add(slack) {
                    // This output starts past the window. The window ends in
                    // silence, and the output waits for the next call.
                    run.pad(limit - end);
                    if let Some(pipe) = self.audio.as_mut() {
                        pipe.heard.push_front(output);
                    }
                    break;
                }
            }
            let pcm = audible(output)?.pcm;
            if pcm.rate == 0 {
                return Err(EngineError::NoClock);
            }
            let floats = unit_samples(&pcm)?;
            if floats.is_empty() {
                empty = empty.saturating_add(1);
                if empty == 8 {
                    break;
                }
                continue;
            }
            match self.run.as_mut() {
                Some(run) => {
                    if run.rate != pcm.rate || run.layout != pcm.layout {
                        return Err(EngineError::Layout);
                    }
                    let late = start - run.end();
                    if late.abs() <= slack {
                        run.samples.extend_from_slice(&floats);
                    } else {
                        place(run, start, &floats);
                    }
                }
                None => {
                    if pcm.rate != rate {
                        // The decoder runs at another rate than the stream
                        // names, as HE-AAC does.
                        rate = pcm.rate;
                        want = sample_at(source, rate);
                    }
                    self.run = Some(AudioRun {
                        next: start,
                        rate,
                        layout: pcm.layout.clone(),
                        samples: floats,
                    });
                }
            }
        }
        let Some(run) = self.run.as_mut() else {
            return Err(EngineError::NotReady);
        };
        let channels = run.layout.len();
        let mut out = Vec::with_capacity(usize::try_from(frames).unwrap_or(0) * channels);
        let silent = (run.next - want).clamp(0, needed);
        out.resize(usize::try_from(silent).unwrap_or(0) * channels, 0.0);
        let take = usize::try_from(needed - silent)
            .unwrap_or(0)
            .min(run.frames());
        out.extend(run.samples.drain(..take * channels));
        run.next = run.next.saturating_add(i64::try_from(take).unwrap_or(0));
        cutlass_codec::access::Pcm::from_f32(run.rate, run.layout.clone(), out)
            .map_err(|_| EngineError::Layout)
    }

    /// [`Self::audio_from`] for a stream with no clock or no stated rate:
    /// seek and read from the landed packet.
    fn audio_from_start(
        &mut self,
        source: Micros,
        frames: u32,
    ) -> Result<cutlass_codec::access::Pcm, EngineError> {
        self.run = None;
        let (track, timescale, lead) = {
            let Some(pipe) = self.audio.as_mut() else {
                return Err(EngineError::NotMedia);
            };
            pipe.reset();
            (pipe.track, pipe.timescale, pipe.lead)
        };
        if timescale.is_clock() {
            let ticks = micros_to_ticks(source, timescale)?.saturating_add(lead);
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
        let mut samples = Vec::new();
        let mut rate = 0_u32;
        let mut layout = None;
        let mut got = 0_u32;
        let mut empty = 0_u32;
        while got < frames {
            let Some(ready) = self.pull_audio()? else {
                break;
            };
            let pcm = ready.pcm;
            if pcm.rate == 0 {
                return Err(EngineError::NoClock);
            }
            if rate == 0 {
                rate = pcm.rate;
                layout = Some(pcm.layout.clone());
            } else if rate != pcm.rate || layout.as_ref() != Some(&pcm.layout) {
                return Err(EngineError::Layout);
            }
            let floats = unit_samples(&pcm)?;
            let channels = pcm.layout.len();
            if channels == 0 {
                return Err(EngineError::Layout);
            }
            let available = floats.len() / channels;
            if available == 0 {
                empty = empty.saturating_add(1);
                if empty == 8 {
                    break;
                }
                continue;
            }
            let need = usize::try_from(frames - got).unwrap_or(usize::MAX);
            let take_frames = available.min(need);
            let take = take_frames.saturating_mul(channels);
            samples.extend_from_slice(&floats[..take]);
            got = got.saturating_add(u32::try_from(take_frames).unwrap_or(0));
            if take_frames < available {
                break;
            }
        }
        let Some(layout) = layout else {
            return Err(EngineError::NotReady);
        };
        if rate == 0 {
            return Err(EngineError::NoClock);
        }
        cutlass_codec::access::Pcm::from_f32(rate, layout, samples).map_err(|_| EngineError::Layout)
    }

    /// The next picture in display order from where the read stands, or
    /// [`None`] when the stream ends.
    ///
    /// For a walk over a whole file, such as a measurement or a check:
    /// nothing is seeked and no picture is kept.
    pub fn next_picture(&mut self) -> Result<Option<HeldFrame>, EngineError> {
        self.cursor = None;
        self.run = None;
        self.pull_picture(&TimeRange::new(0, i64::MAX))
    }

    /// The next decoded audio buffer, or [`None`] when the stream ends or there is no audio.
    pub fn pull_audio(&mut self) -> Result<Option<PcmReady>, EngineError> {
        self.cursor = None;
        self.run = None;
        self.next_audio()?.map(audible).transpose()
    }

    /// The next decoded audio buffer with its timestamp. A packet that
    /// returns several keeps the rest for the next call.
    fn next_audio(&mut self) -> Result<Option<AudioOutput>, EngineError> {
        let Some(track) = self.audio.as_ref().map(|pipe| pipe.track) else {
            return Ok(None);
        };
        if let Some(output) = self.audio.as_mut().and_then(|pipe| pipe.heard.pop_front()) {
            return Ok(Some(output));
        }
        let mut taken = 0_u32;
        let mut others = 0_u32;
        let mut skipped = 0_u32;
        loop {
            if taken == DECODE_PACKETS {
                return Err(EngineError::DecodeLimit);
            }
            let Some(packet) = self.demux.read_packet().map_err(from_read)? else {
                return Ok(None);
            };
            if packet.stream() != track {
                if others == OTHER_STREAM_PACKETS {
                    return Err(EngineError::DecodeLimit);
                }
                others = others.saturating_add(1);
                continue;
            }
            taken = taken.saturating_add(1);
            let Some(pipe) = self.audio.as_mut() else {
                return Ok(None);
            };
            let access = pipe.access(&packet);
            let outputs = match pipe.decoder.push(&access) {
                Ok(outputs) => outputs,
                Err(CodecError::Malformed { .. }) if skipped < SKIPPED_AUDIO => {
                    skipped = skipped.saturating_add(1);
                    continue;
                }
                Err(error) => return Err(EngineError::Codec(error)),
            };
            pipe.heard
                .extend(outputs.into_iter().filter_map(|output| match output {
                    Output::Audio(audio) => Some(audio),
                    Output::Picture(_) => None,
                }));
            if let Some(output) = pipe.heard.pop_front() {
                return Ok(Some(output));
            }
        }
    }
}

/// Hold `floats`, decoded for sample `start`, at its own time in `run`.
///
/// A packet the decoder refused leaves a hole that becomes silence, so later
/// sound stays on its timestamps. Output that starts before the held sound
/// ends repeats time already held, and only its later frames are kept. A
/// clock more than a second back is a new timeline, and plays on in order.
fn place(run: &mut AudioRun, start: i64, floats: &[f32]) {
    let end = run.end();
    if start > end {
        run.pad(start - end);
        run.samples.extend_from_slice(floats);
        return;
    }
    let overlap = end - start;
    if overlap > i64::from(run.rate) {
        run.samples.extend_from_slice(floats);
        return;
    }
    let channels = run.layout.len().max(1);
    let held = usize::try_from(overlap)
        .unwrap_or(usize::MAX)
        .saturating_mul(channels);
    if let Some(rest) = floats.get(held..) {
        run.samples.extend_from_slice(rest);
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
    start_ticks: Option<i64>,
    duration: i64,
    timescale: Timescale,
    cover: &TimeRange,
) -> Result<(Micros, Micros), EngineError> {
    if !timescale.is_clock() {
        let end = cover.start.0.saturating_add(cover.duration.0);
        return Ok((cover.start, Micros(end)));
    }
    let Some(start_ticks) = start_ticks else {
        let end = cover.start.0.saturating_add(cover.duration.0);
        return Ok((cover.start, Micros(end)));
    };
    let start = ticks_to_micros(start_ticks, timescale)?;
    let end_ticks = start_ticks.saturating_add(duration.max(1));
    let end = ticks_to_micros(end_ticks, timescale)?;
    Ok((start, end))
}

/// Every picture one packet released, with its own start in stream ticks.
/// A still keeps only its first frame and takes the packet's time.
fn push_picture(
    decoder: &mut dyn Decoder,
    packet: &AccessPacket,
    timescale: Timescale,
) -> Result<Vec<(DecodedFrame, Option<i64>)>, EngineError> {
    match decoder.decode_picture(packet) {
        Ok(frames) => Ok(frames
            .into_iter()
            .take(1)
            .map(|still| (DecodedFrame::Still(Box::new(still.image)), None))
            .collect()),
        Err(CodecError::Unsupported {
            what: "picture output",
        }) => {
            let outputs = decoder.push(packet)?;
            Ok(outputs
                .into_iter()
                .filter_map(|output| coded_output(output, timescale))
                .collect())
        }
        Err(error) => Err(EngineError::Codec(error)),
    }
}

/// A picture and its start. The start is kept only in the stream's own ticks.
fn coded_output(output: Output, timescale: Timescale) -> Option<(DecodedFrame, Option<i64>)> {
    match output {
        Output::Picture(picture) => {
            let pts = (picture.timescale() == timescale.ticks_per_second()).then(|| picture.pts());
            Some((frame_from_picture(picture), pts))
        }
        Output::Audio(_) => None,
    }
}

fn frame_from_picture(picture: PictureOutput) -> DecodedFrame {
    let info = picture.info();
    match picture.samples() {
        PictureSamples::Platform { token } => {
            return DecodedFrame::Platform {
                width: info.width,
                height: info.height,
                token,
            };
        }
        PictureSamples::Surface(surface) => {
            return DecodedFrame::Surface {
                width: info.width,
                height: info.height,
                primaries: info.primaries,
                transfer: info.transfer,
                surface: surface.clone(),
            };
        }
        PictureSamples::Cpu(_) => {}
    }
    DecodedFrame::Coded {
        width: info.width,
        height: info.height,
        pixel: info.pixel,
        primaries: info.primaries,
        transfer: info.transfer,
        bytes: Arc::new(picture.into_cpu().unwrap_or_default()),
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

/// The first picture and the first sound stream this host can decode. A
/// stream it cannot, such as the spatial-audio track beside the AAC one in a
/// recent iPhone movie, is passed over. A file with nothing decodable is the
/// first stream's error.
fn open_pipes(
    factory: &DecodeFactory,
    demux: &dyn Demux,
) -> Result<(Option<Pipe>, Option<Pipe>), EngineError> {
    let mut picture = None;
    let mut audio = None;
    let mut first_error = None;
    for stream in demux.streams() {
        let sized = stream.params().size();
        let rate = stream.params().sample_rate();
        if sized.is_none() && rate.is_none() {
            continue;
        }
        let wanted = if sized.is_some() {
            picture.is_none()
        } else {
            audio.is_none()
        };
        if !wanted {
            continue;
        }
        match open_pipe(factory, stream) {
            Ok(pipe) if sized.is_some() => picture = Some(pipe),
            Ok(pipe) => audio = Some(pipe),
            Err(error) => {
                first_error.get_or_insert(error);
            }
        }
    }
    if picture.is_none()
        && audio.is_none()
        && let Some(error) = first_error
    {
        return Err(error);
    }
    Ok((picture, audio))
}

fn open_pipe(
    factory: &DecodeFactory,
    stream: &cutlass_container::Stream,
) -> Result<Pipe, EngineError> {
    let codec = stream.params().codec();
    let StreamCodec::Known(id) = codec else {
        return Err(EngineError::UnknownCodec {
            tag: unknown_tag(codec),
        });
    };
    let params = params_for(
        stream.params().extradata(),
        stream.params().size(),
        stream.params().sample_rate(),
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
    let mut pipe = Pipe::new(
        stream.id().get(),
        stream.timescale(),
        decoder,
        edit_lead(stream.edits()),
    );
    pipe.gapless = gapless_of(stream);
    Ok(pipe)
}

/// The stream's encoder delay and padding as [`SideData::Gapless`], when the
/// container stored both and they fit.
fn gapless_of(stream: &cutlass_container::Stream) -> Option<SideData> {
    let trim = stream.trim();
    let encoder_delay = u32::try_from(trim.encoder_delay()?).ok()?;
    let trailing_padding = u32::try_from(trim.trailing_padding()?).ok()?;
    Some(SideData::Gapless {
        encoder_delay,
        trailing_padding,
        sample_rate: stream.params().sample_rate()?,
    })
}

/// `ticks` at `from` ticks per second, counted at `to`. Rounds toward zero;
/// a value out of range is zero.
fn rescale(ticks: i64, from: u32, to: u32) -> i64 {
    if from == to {
        return ticks;
    }
    let scaled = i128::from(ticks) * i128::from(to) / i128::from(from.max(1));
    i64::try_from(scaled).unwrap_or(0)
}

/// Where the stream's time zero falls, in its own ticks: the first edit's
/// media time, less the empty edits before it that delay the stream. A
/// stream with no edits starts at tick zero. Encoders write this to hide an
/// AAC decoder's warm-up samples or the delay B-frames add to video.
fn edit_lead(edits: &[cutlass_container::Edit]) -> i64 {
    let mut empty = 0_i64;
    for edit in edits {
        match edit.media_time() {
            None => empty = empty.saturating_add(edit.duration()),
            Some(media) => return media.saturating_sub(empty),
        }
    }
    0
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
    let layout = layout_for(id, extra, channels.unwrap_or(0))?;
    let mut params = CodecParams::audio(id, rate, layout);
    let (extra, format) = codec_extra(id, extra)?;
    params.extra = extra;
    if let Some(format) = format {
        params.sample_format = Some(format);
    }
    Ok(params)
}

/// Speakers for a stream of `channels` channels, one through eight.
///
/// A wave file's `dwChannelMask` names them when it is set; a mask with a
/// different number of speakers than channels, or one that names a speaker
/// [`cutlass_codec::access::Channel`] does not have, is refused. Any single
/// speaker of a one-channel file is mono. An AAC stream uses its
/// `channelConfiguration`, in the element order the decoder writes. Every
/// other stream takes the standard layout for its count, in wave order.
fn layout_for(id: CodecId, extra: &[u8], channels: u16) -> Result<ChannelLayout, EngineError> {
    let count = usize::from(channels);
    let standard = StandardLayout::for_count(count).ok_or(EngineError::Layout)?;
    let layout = match (id, wave_mask(id, extra)) {
        (_, Some(mask)) if mask != 0 => {
            if mask.count_ones() != u32::from(channels) {
                return Err(EngineError::Layout);
            }
            if count == 1 {
                ChannelLayout::mono()
            } else {
                ChannelLayout::from_wave_mask(mask).map_err(|_| EngineError::Layout)?
            }
        }
        (CodecId::Aac, _) => aac_configuration(extra)
            .and_then(ChannelLayout::from_aac_config)
            .filter(|layout| layout.len() == count)
            .unwrap_or_else(|| ChannelLayout::standard(standard)),
        _ => ChannelLayout::standard(standard),
    };
    Ok(layout)
}

/// The `dwChannelMask` of a wave description: bytes 8 to 11 of the sixteen
/// the wave demuxer hands a PCM stream. Other streams have no mask.
fn wave_mask(id: CodecId, extra: &[u8]) -> Option<u32> {
    if id != CodecId::Pcm || extra.len() < 16 {
        return None;
    }
    let bytes: [u8; 4] = extra.get(8..12)?.try_into().ok()?;
    Some(u32::from_le_bytes(bytes))
}

/// `channelConfiguration` from an AudioSpecificConfig: an object type of
/// five bits (31 escapes to six more), a frequency index of four bits (15
/// is followed by a 24-bit rate), then four bits.
fn aac_configuration(extra: &[u8]) -> Option<u8> {
    let mut bits = 0_u64;
    for byte in extra.iter().take(8) {
        bits = (bits << 8) | u64::from(*byte);
    }
    let available = u32::try_from(extra.len().min(8)).ok()? * 8;
    let mut at = 0_u32;
    let mut read = |width: u32| -> Option<u64> {
        if at + width > available {
            return None;
        }
        at += width;
        Some((bits >> (available - at)) & ((1 << width) - 1))
    };
    let object = read(5)?;
    if object == 0 {
        return None;
    }
    if object == 31 {
        read(6)?;
    }
    if read(4)? == 15 {
        read(24)?;
    }
    u8::try_from(read(4)?).ok()
}

/// The PCM decoder reads three layout bytes and a sample format that matches
/// them. A WAV description is sixteen bytes: tag, width, a reserved field,
/// block align, channel mask, integer or float, then flags. Other codecs
/// keep the container bytes and the caller's sample format.
fn codec_extra(id: CodecId, extra: &[u8]) -> Result<(Vec<u8>, Option<SampleFormat>), EngineError> {
    if id != CodecId::Pcm {
        return Ok((extra.to_vec(), None));
    }
    if extra.is_empty() {
        return Ok((Vec::new(), Some(SampleFormat::F32)));
    }
    if extra.len() == 3 {
        return Ok((extra.to_vec(), Some(pcm_format(extra[0])?)));
    }
    if extra.len() < 16 {
        return Err(EngineError::Layout);
    }
    let tag = u16::from_le_bytes([extra[0], extra[1]]);
    let bits = u16::from_le_bytes([extra[2], extra[3]]);
    let kind = u16::from_le_bytes([extra[12], extra[13]]);
    // Coding byte: 1 s16, 2 s24, 3 s32, 4 f32, 5 f64. Then little-endian
    // interleaved. The container already refused a big-endian WAV.
    let coding = match (tag, bits, kind) {
        (0x0001, 16, 0) => 1_u8,
        (0x0001, 24, 0) => 2,
        (0x0001, 32, 0) => 3,
        (0x0003, 32, 1) => 4,
        (0x0003, 64, 1) => 5,
        _ => return Err(EngineError::Layout),
    };
    Ok((vec![coding, 0, 0], Some(pcm_format(coding)?)))
}

fn pcm_format(coding: u8) -> Result<SampleFormat, EngineError> {
    match coding {
        1 => Ok(SampleFormat::I16),
        2..=5 => Ok(SampleFormat::F32),
        _ => Err(EngineError::Layout),
    }
}

/// The sample index at `rate` nearest `time`.
fn sample_at(time: Micros, rate: u32) -> i64 {
    let product = i128::from(time.0) * i128::from(rate);
    let rounded = (product + 500_000).div_euclid(1_000_000);
    i64::try_from(rounded).unwrap_or(i64::MAX)
}

fn unit_samples(pcm: &cutlass_codec::access::Pcm) -> Result<Vec<f32>, EngineError> {
    if let Some(samples) = pcm.as_f32() {
        return Ok(samples.to_vec());
    }
    let Some(samples) = pcm.as_i16() else {
        return Err(EngineError::Layout);
    };
    Ok(samples
        .iter()
        .map(|sample| f32::from(*sample) / I16_FULL)
        .collect())
}

fn unknown_tag(codec: &StreamCodec) -> String {
    match codec {
        StreamCodec::Known(id) => format!("{id:?}"),
        StreamCodec::Unknown(tag) => String::from_utf8_lossy(tag.tag()).into_owned(),
    }
}

fn stream_info(demux: &dyn Demux, byte_size: u64) -> MediaInfo {
    let mut display = None;
    let streams: Vec<StreamReport> = demux
        .streams()
        .iter()
        .map(|stream| {
            let size = stream.params().size();
            if size.is_some() && display.is_none() {
                display = Some(stream.display());
            }
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
        display: display.unwrap_or_default(),
        streams,
    }
}

fn duration_of(demux: &mut dyn Demux) -> Result<Option<Micros>, EngineError> {
    // Each stream's time zero, as playback places it: a transport stream's
    // first picture shows at its edit's media time, not at tick zero.
    let leads: Vec<(u32, i64)> = demux
        .streams()
        .iter()
        .map(|stream| (stream.id().get(), edit_lead(stream.edits())))
        .collect();
    let mut end = None;
    while let Some(packet) = demux.read_packet().map_err(from_read)? {
        let timescale = packet.timescale();
        if !timescale.is_clock() {
            continue;
        }
        let Some(start) = packet.timestamp().presentation_or_decode() else {
            continue;
        };
        let lead = leads
            .iter()
            .find(|(id, _)| *id == packet.stream())
            .map_or(0, |(_, lead)| *lead);
        let duration = packet.timestamp().duration().unwrap_or(0);
        let ticks = start.saturating_add(duration).saturating_sub(lead);
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
    use cutlass_container::Edit;

    fn edit(media_time: Option<i64>, duration: i64) -> Edit {
        Edit::try_new(media_time, duration, 1, 1).unwrap()
    }

    #[test]
    fn the_edit_list_sets_where_a_stream_starts() {
        assert_eq!(edit_lead(&[]), 0);
        // An AAC track that hides 5058 warm-up frames.
        assert_eq!(edit_lead(&[edit(Some(5_058), 1_000_000)]), 5_058);
        // An empty edit first delays the stream.
        assert_eq!(edit_lead(&[edit(None, 600), edit(Some(1_024), 9_000)]), 424);
        assert_eq!(edit_lead(&[edit(None, 600), edit(Some(0), 9_000)]), -600);
        // Only the first media edit counts.
        assert_eq!(edit_lead(&[edit(Some(10), 5), edit(Some(99), 5)]), 10);
        // Edits with nothing to show start at zero; huge values do not wrap.
        assert_eq!(edit_lead(&[edit(None, 600)]), 0);
        assert_eq!(
            edit_lead(&[edit(None, i64::MAX), edit(Some(i64::MIN), 1)]),
            i64::MIN
        );
    }

    #[test]
    fn a_lead_moves_between_timescales() {
        assert_eq!(rescale(4_800, 48_000, 48_000), 4_800);
        assert_eq!(rescale(4_800, 48_000, 1_000), 100);
        assert_eq!(rescale(-4_800, 48_000, 96_000), -9_600);
        assert_eq!(rescale(7, 0, 1), 7);
        assert_eq!(rescale(i64::MAX, 1, u32::MAX), 0);
    }

    #[test]
    fn a_picture_counts_the_memory_it_holds() {
        let coded = DecodedFrame::Coded {
            width: 4,
            height: 2,
            pixel: PixelFormat::Yuv420p,
            primaries: ColorPrimaries::Bt709,
            transfer: Transfer::Bt709,
            bytes: vec![0; 12].into(),
        };
        assert_eq!(coded.byte_size(), 12);
        let platform = DecodedFrame::Platform {
            width: 3840,
            height: 2160,
            token: 0,
        };
        assert_eq!(platform.byte_size(), 3840 * 2160 * 3 / 2);
    }

    /// The sixteen bytes the wave demuxer hands a 16-bit integer PCM stream.
    fn wave_extra(mask: u32) -> Vec<u8> {
        let mut extra = Vec::new();
        extra.extend_from_slice(&1_u16.to_le_bytes());
        extra.extend_from_slice(&16_u16.to_le_bytes());
        extra.extend_from_slice(&16_u16.to_le_bytes());
        extra.extend_from_slice(&4_u16.to_le_bytes());
        extra.extend_from_slice(&mask.to_le_bytes());
        extra.extend_from_slice(&0_u16.to_le_bytes());
        extra.extend_from_slice(&u16::from(mask != 0).to_le_bytes());
        extra
    }

    fn sound(id: CodecId, extra: &[u8], channels: u16) -> Result<ChannelLayout, EngineError> {
        params_for(extra, None, Some(48_000), Some(channels), id)
            .map(|params| params.layout.expect("an audio stream has a layout"))
    }

    #[test]
    fn one_to_eight_channels_open_in_the_standard_layout_for_the_count() {
        use cutlass_codec::access::Channel::{
            BackCenter, BackLeft, BackRight, FrontCenter, FrontLeft, FrontRight, Lfe, SideLeft,
            SideRight,
        };
        let expected: [&[cutlass_codec::access::Channel]; 8] = [
            &[FrontCenter],
            &[FrontLeft, FrontRight],
            &[FrontLeft, FrontRight, FrontCenter],
            &[FrontLeft, FrontRight, BackLeft, BackRight],
            &[FrontLeft, FrontRight, FrontCenter, BackLeft, BackRight],
            &[FrontLeft, FrontRight, FrontCenter, Lfe, BackLeft, BackRight],
            &[
                FrontLeft,
                FrontRight,
                FrontCenter,
                Lfe,
                BackCenter,
                SideLeft,
                SideRight,
            ],
            &[
                FrontLeft,
                FrontRight,
                FrontCenter,
                Lfe,
                BackLeft,
                BackRight,
                SideLeft,
                SideRight,
            ],
        ];
        for (index, channels) in expected.into_iter().enumerate() {
            let count = u16::try_from(index + 1).unwrap();
            // A wave file with no mask, a float wave file, and the three
            // layout bytes another container writes all use the count.
            for extra in [wave_extra(0), Vec::new(), vec![1, 0, 0]] {
                let layout = sound(CodecId::Pcm, &extra, count).expect("opens");
                assert_eq!(layout.channels(), channels, "{count} channels");
            }
            let other = sound(CodecId::Flac, &[], count).expect("opens");
            assert_eq!(other.channels(), channels, "{count} flac channels");
        }
        assert_eq!(sound(CodecId::Pcm, &[], 0), Err(EngineError::Layout));
        assert_eq!(sound(CodecId::Pcm, &[], 9), Err(EngineError::Layout));
        assert_eq!(
            sound(CodecId::Pcm, &wave_extra(0), 9),
            Err(EngineError::Layout)
        );
        assert_eq!(
            sound(CodecId::Pcm, &wave_extra(0), u16::MAX),
            Err(EngineError::Layout)
        );
        assert_eq!(
            params_for(&[], None, Some(48_000), None, CodecId::Pcm),
            Err(EngineError::Layout)
        );
    }

    #[test]
    fn a_wave_channel_mask_names_the_speakers() {
        use cutlass_codec::access::Channel::{
            BackCenter, FrontCenter, FrontLeft, FrontRight, Lfe, SideLeft, SideRight,
        };
        // 5.1 with side surrounds, as ffmpeg writes it.
        let side = sound(CodecId::Pcm, &wave_extra(0x60F), 6).expect("5.1 side");
        assert_eq!(
            side.channels(),
            &[FrontLeft, FrontRight, FrontCenter, Lfe, SideLeft, SideRight]
        );
        // Two channels that are not left and right.
        let odd = sound(CodecId::Pcm, &wave_extra(0x104), 2).expect("center and back");
        assert_eq!(odd.channels(), &[FrontCenter, BackCenter]);
        assert_eq!(
            sound(CodecId::Pcm, &wave_extra(0x3), 2),
            Ok(ChannelLayout::stereo())
        );
        // Any one speaker of a one-channel file is mono.
        assert_eq!(
            sound(CodecId::Pcm, &wave_extra(0x1), 1),
            Ok(ChannelLayout::mono())
        );
        assert_eq!(
            sound(CodecId::Pcm, &wave_extra(0x4), 1),
            Ok(ChannelLayout::mono())
        );
    }

    #[test]
    fn a_wave_mask_that_disagrees_with_the_count_is_refused() {
        // Six speakers named for two channels, and two for six.
        assert_eq!(
            sound(CodecId::Pcm, &wave_extra(0x3F), 2),
            Err(EngineError::Layout)
        );
        assert_eq!(
            sound(CodecId::Pcm, &wave_extra(0x3), 6),
            Err(EngineError::Layout)
        );
        assert_eq!(
            sound(CodecId::Pcm, &wave_extra(0x3), 1),
            Err(EngineError::Layout)
        );
        // Top center has no channel to play on.
        assert_eq!(
            sound(CodecId::Pcm, &wave_extra(0x803), 3),
            Err(EngineError::Layout)
        );
        // Nine speakers.
        assert_eq!(
            sound(CodecId::Pcm, &wave_extra(0x1FF), 9),
            Err(EngineError::Layout)
        );
    }

    #[test]
    fn an_aac_stream_uses_its_channel_configuration() {
        // AAC LC, 48 kHz, configuration 6: 5.1 in element order.
        let five_one = [0x11, 0xB0];
        assert_eq!(aac_configuration(&five_one), Some(6));
        assert_eq!(
            sound(CodecId::Aac, &five_one, 6),
            Ok(ChannelLayout::five_one())
        );
        // Configuration 2 is stereo.
        assert_eq!(
            sound(CodecId::Aac, &[0x11, 0x90], 2),
            Ok(ChannelLayout::stereo())
        );
        // An escaped object type and an explicit rate still reach the field.
        // Object 31+2, index 15, rate 0x00BB80, configuration 7.
        let escaped = [0xF8, 0x5E, 0x01, 0x77, 0x00, 0xE0];
        assert_eq!(aac_configuration(&escaped), Some(7));
        // A program config element (configuration 0), a configuration that
        // disagrees with the count, and no config at all use the count.
        for extra in [&[0x11, 0x80][..], &[0x11, 0xB0], &[]] {
            assert_eq!(
                sound(CodecId::Aac, extra, 2),
                Ok(ChannelLayout::stereo()),
                "{extra:?}"
            );
        }
        assert_eq!(aac_configuration(&[0x00, 0x00]), None);
        assert_eq!(aac_configuration(&[0x11]), None);
    }

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

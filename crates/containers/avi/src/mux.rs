//! AVI muxer: OpenDML AVI 2.0 with an AVI 1.0 `idx1`.
//!
//! The file is laid out as FFmpeg and VirtualDub lay out an OpenDML AVI:
//!
//! ```text
//! RIFF 'AVI '
//!   LIST 'hdrl'  avih, one LIST 'strl' per stream (strh, strf, indx), LIST 'odml' (dmlh)
//!   LIST 'movi'  NNdc / NNwb chunks, then one ix## standard index per stream
//!   idx1         every chunk of the first RIFF
//! RIFF 'AVIX'    as many as the file needs, each at most `riff_bytes` long
//!   LIST 'movi'  chunks, then ix## per stream
//! ```
//!
//! Every stream's `indx` super index names its `ix##` chunks, so a reader
//! that knows OpenDML finds every chunk in every `RIFF`, and one that only
//! knows AVI 1.0 plays the first `RIFF` from `idx1`.
//!
//! Packets go to the file as they arrive, in that order: the caller
//! interleaves them, as for the MP4 muxer in this workspace, and [`Mux::interleave`] does not reorder. The headers are
//! written once every stream has sent its first packet (each stream's AVI
//! clock comes from that packet's duration), so packets wait in memory until
//! then, up to [`AviMuxOptions::header_buffer_bytes`]. Counts, sizes, and
//! indexes are filled in at [`Mux::finish`].
//!
//! Each stream's clock:
//!
//! - Video: `dwScale / dwRate` is the frame period over the stream
//!   timescale. One chunk is one frame, in the order packets arrive,
//!   which is decode order. A packet lasting several frames is followed by
//!   empty chunks, and a packet whose decode time is past the end of the
//!   one before is preceded by them; readers count them as dropped frames
//!   (at most [`MAX_SKIPPED_FRAMES`] in a row). Which times count as decode
//!   times is in [`decode_time`]; without one, the chunk sequence follows
//!   the packet durations, as the MP4 muxer's decode times do. The frame
//!   period comes from the first eight packets ([`frame_period`]); a time
//!   off that grid lands on the nearest frame.
//! - PCM: constant bitrate, `dwScale` the block size and `dwSampleSize` the
//!   same; the stream timescale must be the sample rate and packets must
//!   follow each other without gaps.
//! - MP3, AAC, AC-3: variable bitrate, one frame per chunk, `dwSampleSize`
//!   zero and `nBlockAlign` at least the largest frame, so FFmpeg's,
//!   GStreamer's, and DirectShow's `ceil(len / nBlockAlign)` count is one
//!   frame per chunk. Every frame but the last has the first one's duration.
//!
//! A first packet after zero is stored as `dwStart`; one before zero
//! (an encoder's negative decode time) moves that stream to start at zero.

use std::io::{BufWriter, Seek, SeekFrom, Write};

use cutlass_codec::{CodecId, ContainerId, Packet, StreamCodec, StreamParams, Timescale};
use cutlass_container::{
    FailedMux, FinishedMux, IndexPlace, Limits, Mux, MuxError, MuxMode, MuxStatus, NewStream,
    TrackDuration, TrackId,
};
use cutlass_riff::WavFormat;

use crate::header::{AVIF_HASINDEX, AVIF_ISINTERLEAVED};
use crate::index::AVIIF_KEYFRAME;
use crate::limits::{MAX_SKIPPED_FRAMES, MAX_STREAMS};

/// Settings for [`AviMux`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AviMuxOptions {
    /// Largest `RIFF` segment, index included. FFmpeg's limit, 1 GiB, is the
    /// default; anything up to 4 GiB less a margin is valid AVI.
    pub riff_bytes: u64,
    /// Rows each stream's `indx` super index has room for: one per `RIFF`.
    pub super_index_entries: u32,
    /// Packets held while waiting for every stream's first packet.
    pub header_buffer_bytes: u64,
}

impl Default for AviMuxOptions {
    fn default() -> Self {
        Self {
            riff_bytes: 1 << 30,
            super_index_entries: 256,
            header_buffer_bytes: 64 << 20,
        }
    }
}

/// The largest `RIFF` segment this muxer writes: below 4 GiB, with room
/// for a chunk header past the last limit check.
const MAX_RIFF_BYTES: u64 = 0xFFFF_0000;
const AVIH_BYTES: usize = 56;
const STRH_BYTES: usize = 56;
const DMLH_BYTES: usize = 248;
const MP3_EXTRA: usize = 12;
/// Video packets held before the headers, to find the frame period.
const VIDEO_CLOCK_PACKETS: u32 = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Kind {
    Video {
        fourcc: [u8; 4],
        width: u32,
        height: u32,
    },
    Pcm {
        format: WavFormat,
    },
    Compressed {
        tag: u16,
    },
}

/// One stream being written.
struct Track {
    id: TrackId,
    params: StreamParams,
    timescale: Timescale,
    kind: Kind,
    config: Vec<u8>,
    chunk: [u8; 4],
    /// Ticks in one AVI unit, once the first packet has set it.
    unit: i64,
    /// Ticks at unit zero.
    origin: i64,
    start_units: u32,
    /// Units written, empty chunks included.
    next: i64,
    /// A packet shorter than one frame closed a variable-bitrate stream.
    closed: bool,
    /// Packets waiting for the headers.
    queued: u32,
    /// Video: where the last packet ended, in ticks after `origin`.
    exact_end: i64,
    ticks: i64,
    bytes: u64,
    max_chunk: u32,
    first_riff_units: u64,
    strh_at: u64,
    strf_at: u64,
    strf_len: usize,
    indx_at: u64,
    segment: Vec<(u32, u32)>,
    segment_units: u64,
    supers: Vec<(u64, u32, u32)>,
}

impl Track {
    fn video(&self) -> bool {
        matches!(self.kind, Kind::Video { .. })
    }

    fn block_align(&self) -> u32 {
        match &self.kind {
            Kind::Pcm { format } => u32::from(format.block_align()),
            Kind::Video { .. } => 0,
            Kind::Compressed { tag } => {
                let conventional = match tag {
                    0x0055 => 1152,
                    0x2000 => 3840,
                    _ => 1,
                };
                self.max_chunk.max(conventional)
            }
        }
    }

    /// `dwScale` and `dwRate`, reduced.
    fn scale_rate(&self) -> (u32, u32) {
        let (scale, rate) = match &self.kind {
            Kind::Pcm { format } => {
                let align = u64::from(format.block_align());
                (align, u64::from(format.sample_rate()) * align)
            }
            _ => (
                u64::try_from(self.unit).unwrap_or(1),
                u64::from(self.timescale.ticks_per_second()),
            ),
        };
        let common = if matches!(self.kind, Kind::Pcm { .. }) {
            1
        } else {
            gcd(scale, rate).max(1)
        };
        (
            u32::try_from(scale / common).unwrap_or(u32::MAX),
            u32::try_from(rate / common).unwrap_or(u32::MAX),
        )
    }

    fn sample_size(&self) -> u32 {
        match &self.kind {
            Kind::Pcm { format } => u32::from(format.block_align()),
            _ => 0,
        }
    }
}

struct Idx1Row {
    track: u8,
    key: bool,
    offset: u32,
    size: u32,
}

/// Writes an AVI file. See the module documentation for the layout.
pub struct AviMux<W: Write + Seek> {
    writer: BufWriter<W>,
    options: AviMuxOptions,
    limits: Limits,
    tracks: Vec<Track>,
    status: MuxStatus,
    failure: Option<MuxError>,
    /// Packets waiting for the headers.
    waiting: Vec<Packet>,
    waiting_bytes: u64,
    started: bool,
    base: u64,
    position: u64,
    avih_at: u64,
    dmlh_at: u64,
    riff_at: u64,
    movi_at: u64,
    riffs: u32,
    idx1: Vec<Idx1Row>,
    segment_chunks: u64,
    segment_index_bytes: u64,
    payload: u64,
}

fn rejected(message: impl Into<String>) -> MuxError {
    MuxError::Rejected {
        message: message.into(),
    }
}

fn io(error: &std::io::Error) -> MuxError {
    rejected(format!("avi write failed: {error}"))
}

const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let rest = a % b;
        a = b;
        b = rest;
    }
    a
}

fn digits(number: usize) -> Result<[u8; 2], MuxError> {
    let number = u8::try_from(number).map_err(|_| rejected("too many streams"))?;
    if number >= 100 {
        return Err(rejected("too many streams"));
    }
    Ok([b'0' + number / 10, b'0' + number % 10])
}

fn put16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn len32(len: usize) -> Result<u32, MuxError> {
    u32::try_from(len).map_err(|_| MuxError::HostileSize {
        declared: u64::try_from(len).unwrap_or(u64::MAX),
        limit: u64::from(u32::MAX),
    })
}

impl<W: Write + Seek> AviMux<W> {
    #[must_use]
    pub fn new(writer: W) -> Self {
        Self::with_options(writer, AviMuxOptions::default())
    }

    #[must_use]
    pub fn with_options(writer: W, options: AviMuxOptions) -> Self {
        let mut limits = Limits::standard();
        limits.max_tracks = limits.max_tracks.min(MAX_STREAMS);
        Self {
            writer: BufWriter::with_capacity(256 * 1024, writer),
            options,
            limits,
            tracks: Vec::new(),
            status: MuxStatus::Building,
            failure: None,
            waiting: Vec::new(),
            waiting_bytes: 0,
            started: false,
            base: 0,
            position: 0,
            avih_at: 0,
            dmlh_at: 0,
            riff_at: 0,
            movi_at: 0,
            riffs: 0,
            idx1: Vec::new(),
            segment_chunks: 0,
            segment_index_bytes: 0,
            payload: 0,
        }
    }

    /// `RIFF` segments written so far.
    #[must_use]
    pub const fn riff_count(&self) -> u32 {
        self.riffs
    }

    /// The writer, once the file is finished.
    pub fn into_writer(self) -> Result<W, MuxError> {
        if self.status != MuxStatus::Finished {
            return Err(MuxError::NotBuilding);
        }
        self.writer.into_inner().map_err(|error| io(error.error()))
    }

    fn fail(&mut self, error: MuxError) -> MuxError {
        if self.status == MuxStatus::Building {
            self.status = MuxStatus::Failed;
            self.failure = Some(error.clone());
        }
        error
    }

    fn building(&self) -> Result<(), MuxError> {
        match self.status {
            MuxStatus::Building => Ok(()),
            MuxStatus::Failed => Err(self.failure.clone().unwrap_or(MuxError::NotBuilding)),
            MuxStatus::Finished => Err(MuxError::AlreadyFinished),
        }
    }

    fn track_index(&self, id: u32) -> Result<usize, MuxError> {
        self.tracks
            .iter()
            .position(|track| track.id.get() == id)
            .ok_or(MuxError::UnknownTrack { id })
    }

    fn describe(stream: &NewStream, number: usize) -> Result<(Kind, [u8; 4]), MuxError> {
        let [tens, ones] = digits(number)?;
        let id = match stream.params.codec() {
            StreamCodec::Unknown(unknown) => return Err(MuxError::UnknownCodec(unknown.clone())),
            StreamCodec::Known(id) => *id,
        };
        let video = |fourcc: &[u8; 4]| -> Result<Kind, MuxError> {
            let size = stream
                .params
                .size()
                .ok_or_else(|| rejected("an AVI video stream needs a frame size"))?;
            Ok(Kind::Video {
                fourcc: *fourcc,
                width: size.width(),
                height: size.height(),
            })
        };
        let kind = match id {
            CodecId::H264 => video(b"H264")?,
            CodecId::Hevc => video(b"HEVC")?,
            CodecId::Mpeg4 => video(b"FMP4")?,
            CodecId::Mpeg2 => video(b"MPG2")?,
            CodecId::Mjpeg => video(b"MJPG")?,
            CodecId::Vp8 => video(b"VP80")?,
            CodecId::Vp9 => video(b"VP90")?,
            CodecId::Av1 => video(b"AV01")?,
            CodecId::Pcm => Kind::Pcm {
                format: pcm_format(&stream.params)?,
            },
            CodecId::Mp3 => Kind::Compressed { tag: 0x0055 },
            CodecId::Aac => Kind::Compressed { tag: 0x00FF },
            CodecId::Ac3 => Kind::Compressed { tag: 0x2000 },
            other => {
                return Err(rejected(format!(
                    "{other:?} is not stored in AVI by this muxer"
                )));
            }
        };
        if !matches!(kind, Kind::Video { .. })
            && (stream.params.sample_rate().is_none() || stream.params.channels().is_none())
        {
            return Err(rejected(
                "an AVI audio stream needs a sample rate and channels",
            ));
        }
        let pair = if matches!(kind, Kind::Video { .. }) {
            *b"dc"
        } else {
            *b"wb"
        };
        Ok((kind, [tens, ones, pair[0], pair[1]]))
    }

    /// Holds a packet until every stream has sent enough to set its clock:
    /// one packet for audio, [`VIDEO_CLOCK_PACKETS`] for video. Past
    /// [`AviMuxOptions::header_buffer_bytes`] the headers go out with what
    /// has arrived, unless a stream has sent nothing.
    fn queue(&mut self, packet: &Packet, index: usize) -> Result<(), MuxError> {
        let size = u64::try_from(packet.bytes().len()).unwrap_or(u64::MAX);
        self.waiting_bytes = self.waiting_bytes.saturating_add(size);
        self.tracks[index].queued = self.tracks[index].queued.saturating_add(1);
        self.waiting.push(packet.clone());
        let full = self.waiting_bytes > self.options.header_buffer_bytes;
        if full && let Some(silent) = self.tracks.iter().find(|track| track.queued == 0) {
            return Err(rejected(format!(
                "stream {} sent no packet in the first {} bytes",
                silent.id.get(),
                self.options.header_buffer_bytes
            )));
        }
        let ready = self.tracks.iter().all(|track| {
            let want = if track.video() {
                VIDEO_CLOCK_PACKETS
            } else {
                1
            };
            track.queued >= want
        });
        if full || ready {
            self.start()?;
        }
        Ok(())
    }

    /// Sets each stream's clock from its first packet, writes the headers,
    /// and writes the packets that waited for them.
    fn start(&mut self) -> Result<(), MuxError> {
        for index in 0..self.tracks.len() {
            let id = self.tracks[index].id.get();
            let mut mine = self.waiting.iter().filter(|packet| packet.stream() == id);
            let (unit, time) = match mine.next() {
                Some(first) => {
                    let stamp = first.timestamp();
                    let mut unit = stamp.duration().ok_or(MuxError::MissingTimestamp)?;
                    if self.tracks[index].video() {
                        let track = &self.tracks[index];
                        let spans = std::iter::once(first).chain(mine).map(|packet| {
                            (
                                packet.timestamp().duration(),
                                decode_time(track, packet.timestamp()),
                            )
                        });
                        unit = frame_period(spans).unwrap_or(unit);
                    }
                    (unit, stamp.decode().or(stamp.presentation()))
                }
                None => (1, None),
            };
            set_clock(&mut self.tracks[index], unit, time)?;
        }
        self.base = self.writer.stream_position().map_err(|error| io(&error))?;
        self.position = self.base;
        let header = self.header()?;
        self.write(&header)?;
        self.started = true;
        self.riffs = 1;
        for packet in std::mem::take(&mut self.waiting) {
            self.emit(&packet)?;
        }
        self.waiting_bytes = 0;
        Ok(())
    }

    fn write(&mut self, bytes: &[u8]) -> Result<(), MuxError> {
        self.writer.write_all(bytes).map_err(|error| io(&error))?;
        self.position += u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        Ok(())
    }

    fn patch(&mut self, at: u64, bytes: &[u8]) -> Result<(), MuxError> {
        self.writer
            .seek(SeekFrom::Start(at))
            .and_then(|_| self.writer.write_all(bytes))
            .and_then(|()| self.writer.seek(SeekFrom::Start(self.position)).map(|_| ()))
            .map_err(|error| io(&error))
    }

    fn header(&mut self) -> Result<Vec<u8>, MuxError> {
        let base = self.base;
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        put32(&mut out, 0);
        out.extend_from_slice(b"AVI LIST");
        let hdrl_size = out.len();
        put32(&mut out, 0);
        out.extend_from_slice(b"hdrlavih");
        put32(&mut out, len32(AVIH_BYTES)?);
        self.avih_at = base + u64::try_from(out.len()).unwrap_or(0);
        out.extend_from_slice(&self.avih_bytes());
        for index in 0..self.tracks.len() {
            let strh = self.strh_bytes(&self.tracks[index]);
            let strf = strf_bytes(&self.tracks[index])?;
            let indx = self.indx_bytes(&self.tracks[index])?;
            out.extend_from_slice(b"LIST");
            put32(
                &mut out,
                len32(4 + 8 + STRH_BYTES + 8 + strf.len() + (strf.len() & 1) + 8 + indx.len())?,
            );
            out.extend_from_slice(b"strlstrh");
            put32(&mut out, len32(STRH_BYTES)?);
            let track = &mut self.tracks[index];
            track.strh_at = base + u64::try_from(out.len()).unwrap_or(0);
            out.extend_from_slice(&strh);
            out.extend_from_slice(b"strf");
            put32(&mut out, len32(strf.len())?);
            track.strf_at = base + u64::try_from(out.len()).unwrap_or(0);
            track.strf_len = strf.len();
            out.extend_from_slice(&strf);
            if strf.len() & 1 == 1 {
                out.push(0);
            }
            out.extend_from_slice(b"indx");
            put32(&mut out, len32(indx.len())?);
            track.indx_at = base + u64::try_from(out.len()).unwrap_or(0);
            out.extend_from_slice(&indx);
        }
        out.extend_from_slice(b"LIST");
        put32(&mut out, len32(4 + 8 + DMLH_BYTES)?);
        out.extend_from_slice(b"odmldmlh");
        put32(&mut out, len32(DMLH_BYTES)?);
        self.dmlh_at = base + u64::try_from(out.len()).unwrap_or(0);
        out.extend_from_slice(&self.dmlh_bytes());
        let hdrl = len32(out.len() - hdrl_size - 4)?;
        out[hdrl_size..hdrl_size + 4].copy_from_slice(&hdrl.to_le_bytes());
        self.riff_at = base;
        self.movi_at = base + u64::try_from(out.len()).unwrap_or(0) + 8;
        out.extend_from_slice(b"LIST");
        put32(&mut out, 0);
        out.extend_from_slice(b"movi");
        Ok(out)
    }

    fn first_video(&self) -> Option<&Track> {
        self.tracks.iter().find(|track| track.video())
    }

    fn avih_bytes(&self) -> Vec<u8> {
        let video = self.first_video();
        let per_frame = video.map_or(0, |track| {
            let micros = i128::from(track.unit) * 1_000_000
                / i128::from(track.timescale.ticks_per_second().max(1));
            u32::try_from(micros).unwrap_or(u32::MAX)
        });
        let (width, height) = match video.map(|track| &track.kind) {
            Some(Kind::Video { width, height, .. }) => (*width, *height),
            _ => (0, 0),
        };
        let seconds = self
            .tracks
            .iter()
            .map(|track| {
                u128::try_from(track.ticks).unwrap_or(0)
                    / u128::from(track.timescale.ticks_per_second().max(1))
            })
            .max()
            .unwrap_or(0)
            .max(1);
        let total: u64 = self.tracks.iter().map(|track| track.bytes).sum();
        let rate = u32::try_from(u128::from(total) / seconds).unwrap_or(u32::MAX);
        let buffer = self
            .tracks
            .iter()
            .map(|track| track.max_chunk)
            .max()
            .unwrap_or(0);
        let mut out = Vec::with_capacity(AVIH_BYTES);
        put32(&mut out, per_frame);
        put32(&mut out, rate);
        put32(&mut out, 0);
        put32(&mut out, AVIF_HASINDEX | AVIF_ISINTERLEAVED);
        put32(
            &mut out,
            u32::try_from(video.map_or(0, |track| track.first_riff_units)).unwrap_or(u32::MAX),
        );
        put32(&mut out, 0);
        put32(&mut out, u32::try_from(self.tracks.len()).unwrap_or(0));
        put32(&mut out, buffer);
        put32(&mut out, width);
        put32(&mut out, height);
        out.resize(AVIH_BYTES, 0);
        out
    }

    fn strh_bytes(&self, track: &Track) -> Vec<u8> {
        let (scale, rate) = track.scale_rate();
        let mut out = Vec::with_capacity(STRH_BYTES);
        match &track.kind {
            Kind::Video { fourcc, .. } => {
                out.extend_from_slice(b"vids");
                out.extend_from_slice(&video_fourcc(*fourcc, &track.config));
            }
            _ => {
                out.extend_from_slice(b"auds");
                put32(&mut out, 0);
            }
        }
        put32(&mut out, 0);
        put16(&mut out, 0);
        put16(&mut out, 0);
        put32(&mut out, 0);
        put32(&mut out, scale);
        put32(&mut out, rate);
        put32(&mut out, track.start_units);
        put32(
            &mut out,
            u32::try_from(track.next.max(0)).unwrap_or(u32::MAX),
        );
        put32(&mut out, track.max_chunk);
        put32(&mut out, u32::MAX);
        put32(&mut out, track.sample_size());
        let (width, height) = match &track.kind {
            Kind::Video { width, height, .. } => (*width, *height),
            _ => (0, 0),
        };
        put16(&mut out, 0);
        put16(&mut out, 0);
        put16(&mut out, u16::try_from(width).unwrap_or(u16::MAX));
        put16(&mut out, u16::try_from(height).unwrap_or(u16::MAX));
        out
    }

    fn indx_bytes(&self, track: &Track) -> Result<Vec<u8>, MuxError> {
        let room = usize::try_from(self.options.super_index_entries).unwrap_or(0);
        let mut out = Vec::with_capacity(24 + 16 * room);
        put16(&mut out, 4);
        out.push(0);
        out.push(0);
        put32(&mut out, len32(track.supers.len())?);
        out.extend_from_slice(&track.chunk);
        out.extend_from_slice(&[0; 12]);
        for &(offset, size, duration) in &track.supers {
            out.extend_from_slice(&offset.to_le_bytes());
            put32(&mut out, size);
            put32(&mut out, duration);
        }
        out.resize(24 + 16 * room, 0);
        Ok(out)
    }

    fn dmlh_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(DMLH_BYTES);
        let frames = self.first_video().map_or(0, |track| {
            u32::try_from(track.next.max(0)).unwrap_or(u32::MAX)
        });
        put32(&mut out, frames);
        out.resize(DMLH_BYTES, 0);
        out
    }

    /// Writes one packet, with the empty chunks before and after it that
    /// keep a video stream on its frame grid.
    fn emit(&mut self, packet: &Packet) -> Result<(), MuxError> {
        let index = self.track_index(packet.stream())?;
        let stamp = packet.timestamp();
        let duration = stamp.duration().ok_or(MuxError::MissingTimestamp)?;
        if duration < 0 {
            return Err(MuxError::NegativeDuration);
        }
        let time = stamp.decode().or(stamp.presentation());
        let track = &self.tracks[index];
        let (before, units) = match &track.kind {
            Kind::Video { .. } => {
                let start = decode_time(track, stamp)
                    .map_or(track.exact_end, |time| time.saturating_sub(track.origin));
                let at = nearest(start, track.unit).max(track.next);
                let end = start.saturating_add(duration);
                let last = nearest(end, track.unit).max(at + 1);
                let before = at - track.next;
                self.tracks[index].exact_end = end;
                (before, last - at)
            }
            Kind::Pcm { format } => {
                let align = u64::from(format.block_align());
                let len = u64::try_from(packet.bytes().len()).unwrap_or(u64::MAX);
                if align == 0
                    || len % align != 0
                    || u64::try_from(duration).ok() != Some(len / align)
                {
                    return Err(rejected(
                        "a PCM packet must be whole sample frames, its duration their count",
                    ));
                }
                if time.is_some_and(|time| time - track.origin != track.next) {
                    return Err(rejected("PCM packets must follow each other without gaps"));
                }
                (0, duration)
            }
            Kind::Compressed { .. } => {
                if track.closed {
                    return Err(rejected(
                        "only the last audio frame may be shorter than the first",
                    ));
                }
                if time.is_some_and(|time| {
                    Some(time - track.origin) != track.next.checked_mul(track.unit)
                }) {
                    return Err(rejected("audio frames must follow each other without gaps"));
                }
                if duration > track.unit {
                    return Err(rejected(
                        "audio frames must all have the first frame's duration",
                    ));
                }
                if packet.bytes().len() > usize::from(u16::MAX) {
                    return Err(MuxError::HostileSize {
                        declared: u64::try_from(packet.bytes().len()).unwrap_or(u64::MAX),
                        limit: u64::from(u16::MAX),
                    });
                }
                self.tracks[index].closed = duration < self.tracks[index].unit;
                (0, 1)
            }
        };
        let video = self.tracks[index].video();
        let trailing = if video { units - 1 } else { 0 };
        if before > MAX_SKIPPED_FRAMES || trailing > MAX_SKIPPED_FRAMES {
            return Err(rejected("more than 60000 frames skipped in a row"));
        }
        for _ in 0..before {
            self.chunk(index, &[], false, 1)?;
        }
        let pcm_units = if video { 1 } else { units };
        self.chunk(
            index,
            packet.bytes(),
            packet.keyframe() || !video,
            pcm_units,
        )?;
        for _ in 0..trailing {
            self.chunk(index, &[], false, 1)?;
        }
        let track = &mut self.tracks[index];
        track.ticks = track.ticks.saturating_add(duration);
        self.payload = self
            .payload
            .saturating_add(u64::try_from(packet.bytes().len()).unwrap_or(u64::MAX));
        Ok(())
    }

    /// Writes one chunk of stream `index` worth `units` AVI units, opening
    /// a new `RIFF` first when this one would grow past its limit.
    fn chunk(&mut self, index: usize, bytes: &[u8], key: bool, units: i64) -> Result<(), MuxError> {
        let size = len32(bytes.len())?;
        let padded = u64::from(size) + u64::from(size & 1);
        let idx1_bytes = if self.riffs == 1 {
            16 * (u64::try_from(self.idx1.len()).unwrap_or(u64::MAX) + 1)
        } else {
            0
        };
        let used = self.position - self.riff_at;
        let index_bytes = self.segment_index_bytes + 32 + 8 + idx1_bytes;
        let limit = self.options.riff_bytes.min(MAX_RIFF_BYTES);
        if self.segment_chunks > 0 && used + 8 + padded + index_bytes > limit {
            self.close_segment(false)?;
        }
        let header_at = self.position;
        let data = header_at + 8;
        let mut head = [0u8; 8];
        head[..4].copy_from_slice(&self.tracks[index].chunk);
        head[4..].copy_from_slice(&size.to_le_bytes());
        self.write(&head)?;
        self.write(bytes)?;
        if size & 1 == 1 {
            self.write(&[0])?;
        }
        let offset = u32::try_from(data - self.movi_at)
            .map_err(|_| rejected("a RIFF segment passed 4 GiB"))?;
        let flagged = if key { size } else { size | 0x8000_0000 };
        let budget = u64::from(self.limits.max_index_entries);
        let first_riff = self.riffs == 1;
        let track = &mut self.tracks[index];
        if u64::try_from(track.segment.len()).unwrap_or(u64::MAX) >= budget {
            return Err(MuxError::HostileSize {
                declared: budget + 1,
                limit: budget,
            });
        }
        if track.segment.is_empty() {
            self.segment_index_bytes += 32;
        }
        self.segment_index_bytes += 8;
        track.segment.push((offset, flagged));
        track.segment_units += u64::try_from(units).unwrap_or(0);
        track.next = track.next.saturating_add(units);
        track.bytes += u64::from(size);
        track.max_chunk = track.max_chunk.max(size);
        if first_riff {
            track.first_riff_units += u64::try_from(units).unwrap_or(0);
            if u64::try_from(self.idx1.len()).unwrap_or(u64::MAX) >= budget {
                return Err(MuxError::HostileSize {
                    declared: budget + 1,
                    limit: budget,
                });
            }
            self.idx1.push(Idx1Row {
                track: u8::try_from(index).unwrap_or(u8::MAX),
                key,
                offset: u32::try_from(header_at - (self.movi_at)).unwrap_or(u32::MAX),
                size,
            });
        }
        self.segment_chunks += 1;
        Ok(())
    }

    /// Writes the `ix##` chunks, closes `movi`, writes `idx1` after the
    /// first, closes the `RIFF`, and opens an `AVIX` unless this is the last.
    fn close_segment(&mut self, last: bool) -> Result<(), MuxError> {
        for index in 0..self.tracks.len() {
            if self.tracks[index].segment.is_empty() {
                continue;
            }
            let at = self.position;
            let track = &self.tracks[index];
            if track.supers.len() >= usize::try_from(self.options.super_index_entries).unwrap_or(0)
            {
                return Err(rejected(
                    "the file has more RIFF segments than the super index has rows",
                ));
            }
            let mut out = Vec::with_capacity(32 + 8 * track.segment.len());
            out.extend_from_slice(b"ix");
            out.extend_from_slice(&track.chunk[..2]);
            put32(&mut out, len32(24 + 8 * track.segment.len())?);
            put16(&mut out, 2);
            out.push(0);
            out.push(1);
            put32(&mut out, len32(track.segment.len())?);
            out.extend_from_slice(&track.chunk);
            out.extend_from_slice(&self.movi_at.to_le_bytes());
            put32(&mut out, 0);
            for &(offset, size) in &track.segment {
                put32(&mut out, offset);
                put32(&mut out, size);
            }
            let chunk_size = len32(out.len())?;
            let duration = u32::try_from(track.segment_units).unwrap_or(u32::MAX);
            self.write(&out)?;
            let track = &mut self.tracks[index];
            track.supers.push((at, chunk_size, duration));
            track.segment.clear();
            track.segment_units = 0;
        }
        let movi_size = size32(self.position - self.movi_at)?;
        self.patch(self.movi_at - 4, &movi_size.to_le_bytes())?;
        if self.riffs == 1 {
            let mut out = Vec::with_capacity(8 + 16 * self.idx1.len());
            out.extend_from_slice(b"idx1");
            put32(&mut out, len32(16 * self.idx1.len())?);
            for row in &self.idx1 {
                out.extend_from_slice(&self.tracks[usize::from(row.track)].chunk);
                put32(&mut out, if row.key { AVIIF_KEYFRAME } else { 0 });
                put32(&mut out, row.offset);
                put32(&mut out, row.size);
            }
            self.write(&out)?;
            self.idx1 = Vec::new();
        }
        let riff_size = size32(self.position - (self.riff_at + 8))?;
        self.patch(self.riff_at + 4, &riff_size.to_le_bytes())?;
        self.segment_chunks = 0;
        self.segment_index_bytes = 0;
        if !last {
            self.riffs += 1;
            self.riff_at = self.position;
            self.movi_at = self.position + 20;
            self.write(b"RIFF\0\0\0\0AVIXLIST\0\0\0\0movi")?;
        }
        Ok(())
    }

    fn finish_file(&mut self, durations: &[TrackDuration]) -> Result<(), MuxError> {
        for track in &self.tracks {
            let Some(duration) = durations.iter().find(|duration| duration.track == track.id)
            else {
                return Err(MuxError::Incomplete);
            };
            if duration.ticks < 0 {
                return Err(MuxError::NegativeDuration);
            }
            if duration.timescale != track.timescale {
                return Err(MuxError::TimescaleMismatch {
                    packet: duration.timescale.ticks_per_second(),
                    stream: track.timescale.ticks_per_second(),
                });
            }
            let waiting: i64 = self
                .waiting
                .iter()
                .filter(|packet| packet.stream() == track.id.get())
                .filter_map(|packet| packet.timestamp().duration())
                .sum();
            if duration.ticks != track.ticks.saturating_add(waiting) {
                return Err(MuxError::DurationMismatch {
                    track: track.id.get(),
                });
            }
        }
        if durations.len() != self.tracks.len() {
            return Err(MuxError::Incomplete);
        }
        if self.tracks.is_empty() {
            return Err(rejected("an AVI file needs a stream"));
        }
        if !self.started {
            self.start()?;
        }
        self.close_segment(true)?;
        let avih = self.avih_bytes();
        self.patch(self.avih_at, &avih)?;
        let dmlh = self.dmlh_bytes();
        self.patch(self.dmlh_at, &dmlh)?;
        for index in 0..self.tracks.len() {
            let track = &self.tracks[index];
            let strh = self.strh_bytes(track);
            let strf = strf_bytes(track)?;
            if strf.len() != track.strf_len {
                return Err(rejected("a stream format changed size after the header"));
            }
            let indx = self.indx_bytes(track)?;
            let (strh_at, strf_at, indx_at) = (track.strh_at, track.strf_at, track.indx_at);
            self.patch(strh_at, &strh)?;
            self.patch(strf_at, &strf)?;
            self.patch(indx_at, &indx)?;
        }
        self.writer.flush().map_err(|error| io(&error))
    }
}

fn size32(value: u64) -> Result<u32, MuxError> {
    u32::try_from(value).map_err(|_| MuxError::HostileSize {
        declared: value,
        limit: u64::from(u32::MAX),
    })
}

/// The time a video packet's chunk sits at, when the packet says it.
///
/// A decode time with no presentation time, or one that differs from the
/// presentation time, is a real decode time. A decode time equal to the
/// presentation time is taken only for video that never reorders frames:
/// an encoder that stamps H.264, HEVC, MPEG-4, or MPEG-2 packets with their
/// presentation time in both fields sends B-frames with times that go back.
/// Without a usable time the chunk follows the one before it.
fn decode_time(track: &Track, stamp: cutlass_codec::Timestamp) -> Option<i64> {
    let reorders = matches!(
        track.params.codec(),
        StreamCodec::Known(CodecId::H264 | CodecId::Hevc | CodecId::Mpeg4 | CodecId::Mpeg2)
    );
    match (stamp.decode(), stamp.presentation()) {
        (Some(decode), None) => Some(decode),
        (Some(decode), Some(presentation)) if decode != presentation || !reorders => Some(decode),
        (None, Some(presentation)) if !reorders => Some(presentation),
        _ => None,
    }
}

/// Most empty chunks a frame period may cost per frame to keep every time
/// exact; past it the period is the shortest frame and times round.
const MAX_EXACT_SPLIT: i64 = 16;

/// The video frame period from the first packets: each duration, and the
/// step between decode times that count ([`decode_time`]). Their greatest
/// common divisor keeps every time exact (a first frame that a dropped
/// frame lengthened, a 24 fps clip on a 600 Hz clock that starts 4 ticks
/// late) when that costs at most [`MAX_EXACT_SPLIT`] chunks per frame;
/// otherwise the shortest step, with later times rounded to it.
fn frame_period(spans: impl Iterator<Item = (Option<i64>, Option<i64>)>) -> Option<i64> {
    let mut steps = Vec::new();
    let mut previous: Option<i64> = None;
    for (duration, time) in spans {
        if let Some(duration) = duration.filter(|duration| *duration > 0) {
            steps.push(duration);
        }
        if let Some(time) = time {
            if let Some(before) = previous
                && time > before
            {
                steps.push(time - before);
            }
            previous = Some(time);
        }
    }
    let shortest = *steps.iter().min()?;
    let common = steps
        .iter()
        .fold(0u64, |acc, step| gcd(acc, step.unsigned_abs()));
    let common = i64::try_from(common).ok().filter(|common| *common > 0)?;
    Some(if shortest / common <= MAX_EXACT_SPLIT {
        common
    } else {
        shortest
    })
}

/// The whole number of units nearest `value` ticks.
fn nearest(value: i64, unit: i64) -> i64 {
    let unit = unit.max(1);
    let half = unit / 2;
    if value >= 0 {
        value.saturating_add(half) / unit
    } else {
        -(value.saturating_neg().saturating_add(half) / unit)
    }
}

/// `value / unit`, refusing a value off the grid.
fn on_grid(value: i64, unit: i64) -> Result<i64, MuxError> {
    if unit <= 0 || value % unit != 0 {
        return Err(rejected(
            "a time or duration is not a whole number of frames",
        ));
    }
    Ok(value / unit)
}

fn set_clock(track: &mut Track, unit: i64, time: Option<i64>) -> Result<(), MuxError> {
    if unit <= 0 {
        return Err(rejected(
            "the first packet of a stream needs a positive duration",
        ));
    }
    let unit = match &track.kind {
        Kind::Pcm { format } => {
            if track.timescale.ticks_per_second() != format.sample_rate() {
                return Err(MuxError::TimescaleMismatch {
                    packet: track.timescale.ticks_per_second(),
                    stream: format.sample_rate(),
                });
            }
            1
        }
        _ => unit,
    };
    if u32::try_from(unit).is_err() {
        return Err(rejected("a frame lasts more ticks than AVI can store"));
    }
    track.unit = unit;
    let time = time.unwrap_or(0);
    let hour = 3_600 * i64::from(track.timescale.ticks_per_second());
    if time < 0 {
        track.origin = time;
    } else if time > 0 {
        if time > hour {
            return Err(rejected("a stream starts more than an hour in"));
        }
        // Video lands on the nearest frame; audio must start on its grid.
        let units = if track.video() {
            nearest(time, unit)
        } else {
            on_grid(time, unit)?
        };
        track.origin = units * unit;
        track.start_units = u32::try_from(units).unwrap_or(u32::MAX);
    }
    Ok(())
}

/// `avc1` for H.264 with an `avcC` record and length-prefixed packets, as
/// FFmpeg tags it; the stream's FourCC otherwise.
fn video_fourcc(fourcc: [u8; 4], config: &[u8]) -> [u8; 4] {
    if &fourcc == b"H264" && config.first() == Some(&1) {
        *b"avc1"
    } else {
        fourcc
    }
}

fn strf_bytes(track: &Track) -> Result<Vec<u8>, MuxError> {
    match &track.kind {
        Kind::Video {
            fourcc,
            width,
            height,
        } => {
            let extra = video_extra(*fourcc, &track.config)?;
            let mut out = Vec::with_capacity(40 + extra.len());
            put32(&mut out, len32(40 + extra.len())?);
            put32(&mut out, *width);
            put32(&mut out, *height);
            put16(&mut out, 1);
            put16(&mut out, 24);
            out.extend_from_slice(&video_fourcc(*fourcc, &track.config));
            let image = u64::from(*width) * u64::from(*height) * 3;
            put32(&mut out, u32::try_from(image).unwrap_or(0));
            out.extend_from_slice(&[0; 16]);
            out.extend_from_slice(&extra);
            Ok(out)
        }
        Kind::Pcm { format } => {
            let avg = format
                .sample_rate()
                .checked_mul(u32::from(format.block_align()))
                .ok_or_else(|| rejected("PCM byte rate does not fit"))?;
            format.fmt_bytes(avg)
        }
        Kind::Compressed { tag } => {
            let channels = track.params.channels().unwrap_or(0);
            let rate = track.params.sample_rate().unwrap_or(0);
            let align = track.block_align();
            let seconds_ticks = track.ticks.max(1);
            let avg = u128::from(track.bytes) * u128::from(track.timescale.ticks_per_second())
                / u128::try_from(seconds_ticks).unwrap_or(1);
            let extra = match tag {
                0x0055 => {
                    let mut extra = Vec::with_capacity(MP3_EXTRA);
                    put16(&mut extra, 1);
                    put32(&mut extra, 2);
                    put16(&mut extra, u16::try_from(align).unwrap_or(u16::MAX));
                    put16(&mut extra, 1);
                    put16(&mut extra, 1393);
                    extra
                }
                0x00FF => {
                    if track.config.is_empty() {
                        return Err(rejected("AAC in AVI needs its AudioSpecificConfig"));
                    }
                    track.config.clone()
                }
                _ => Vec::new(),
            };
            let mut out = Vec::with_capacity(18 + extra.len());
            put16(&mut out, *tag);
            put16(&mut out, channels);
            put32(&mut out, rate);
            put32(&mut out, u32::try_from(avg).unwrap_or(u32::MAX));
            put16(&mut out, u16::try_from(align).unwrap_or(u16::MAX));
            put16(&mut out, if *tag == 0x0055 { 0 } else { 16 });
            put16(
                &mut out,
                u16::try_from(extra.len()).map_err(|_| MuxError::HostileSize {
                    declared: u64::try_from(extra.len()).unwrap_or(u64::MAX),
                    limit: u64::from(u16::MAX),
                })?,
            );
            out.extend_from_slice(&extra);
            Ok(out)
        }
    }
}

/// The bytes after the BITMAPINFOHEADER. A QuickTime Motion JPEG
/// description (`jpeg`, `mjpa`, with or without `fiel`) is not stored;
/// Motion JPEG B is refused, as its frames are not JPEG.
fn video_extra(fourcc: [u8; 4], config: &[u8]) -> Result<Vec<u8>, MuxError> {
    if &fourcc != b"MJPG" {
        return Ok(config.to_vec());
    }
    if config.starts_with(&[0xFF, 0xD8]) {
        return Ok(config.to_vec());
    }
    if config
        .get(..4)
        .is_some_and(|tag| tag.eq_ignore_ascii_case(b"mjpb"))
    {
        return Err(rejected("Motion JPEG B is not stored in AVI"));
    }
    Ok(Vec::new())
}

/// The wave format of a PCM stream: from its sixteen-byte description,
/// from the PCM decoder's three layout bytes, or 16-bit integer.
fn pcm_format(params: &StreamParams) -> Result<WavFormat, MuxError> {
    let extra = params.extradata();
    if extra.len() != 3 {
        return WavFormat::for_stream(params);
    }
    let (tag, bits, kind): (u16, u16, u16) = match extra[0] {
        1 => (1, 16, 0),
        2 => (1, 24, 0),
        3 => (1, 32, 0),
        4 => (3, 32, 1),
        5 => (3, 64, 1),
        _ => return Err(rejected("unknown PCM layout")),
    };
    let channels = params
        .channels()
        .ok_or_else(|| rejected("PCM needs a channel count"))?;
    let align = u16::try_from(u32::from(bits / 8) * u32::from(channels))
        .map_err(|_| rejected("PCM block does not fit"))?;
    let mut description = Vec::with_capacity(16);
    put16(&mut description, tag);
    put16(&mut description, bits);
    put16(&mut description, bits);
    put16(&mut description, align);
    put32(&mut description, 0);
    put16(&mut description, kind);
    put16(&mut description, 0);
    let rate = params
        .sample_rate()
        .ok_or_else(|| rejected("PCM needs a sample rate"))?;
    let rebuilt = StreamParams::try_audio(
        StreamCodec::Known(CodecId::Pcm),
        rate,
        channels,
        description,
    )
    .map_err(|error| rejected(error.to_string()))?;
    WavFormat::for_stream(&rebuilt)
}

impl<W: Write + Seek> Mux for AviMux<W> {
    fn family(&self) -> ContainerId {
        ContainerId::Avi
    }

    fn add_stream(&mut self, stream: NewStream) -> Result<TrackId, MuxError> {
        self.building()?;
        if self.started || !self.waiting.is_empty() {
            return Err(self.fail(rejected("streams are added before the first packet")));
        }
        let count = u32::try_from(self.tracks.len()).unwrap_or(u32::MAX);
        if count >= self.limits.max_tracks {
            return Err(self.fail(MuxError::TooManyTracks {
                count: count + 1,
                limit: self.limits.max_tracks,
            }));
        }
        let extra = u64::try_from(stream.params.extradata().len()).unwrap_or(u64::MAX);
        if extra > self.limits.max_extradata_bytes {
            return Err(self.fail(MuxError::HostileSize {
                declared: extra,
                limit: self.limits.max_extradata_bytes,
            }));
        }
        if !stream.timescale.is_clock() {
            return Err(self.fail(MuxError::Clockless));
        }
        let (kind, chunk) =
            Self::describe(&stream, self.tracks.len()).map_err(|error| self.fail(error))?;
        let id = TrackId::new(count);
        let config = if matches!(kind, Kind::Pcm { .. }) {
            Vec::new()
        } else {
            stream.params.extradata().to_vec()
        };
        self.tracks.push(Track {
            id,
            params: stream.params,
            timescale: stream.timescale,
            kind,
            config,
            chunk,
            unit: 1,
            origin: 0,
            start_units: 0,
            next: 0,
            closed: false,
            queued: 0,
            exact_end: 0,
            ticks: 0,
            bytes: 0,
            max_chunk: 0,
            first_riff_units: 0,
            strh_at: 0,
            strf_at: 0,
            strf_len: 0,
            indx_at: 0,
            segment: Vec::new(),
            segment_units: 0,
            supers: Vec::new(),
        });
        Ok(id)
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<(), MuxError> {
        self.building()?;
        let index = self
            .track_index(packet.stream())
            .map_err(|error| self.fail(error))?;
        let timescale = self.tracks[index].timescale;
        if packet.timescale() != timescale {
            return Err(self.fail(MuxError::TimescaleMismatch {
                packet: packet.timescale().ticks_per_second(),
                stream: timescale.ticks_per_second(),
            }));
        }
        let size = u64::try_from(packet.bytes().len()).unwrap_or(u64::MAX);
        if size > self.limits.max_packet_bytes {
            return Err(self.fail(MuxError::HostileSize {
                declared: size,
                limit: self.limits.max_packet_bytes,
            }));
        }
        if packet.timestamp().duration().is_none() {
            return Err(self.fail(MuxError::MissingTimestamp));
        }
        let result = if self.started {
            self.emit(packet)
        } else {
            self.queue(packet, index)
        };
        result.map_err(|error| self.fail(error))
    }

    /// Packets are written in the order they arrive; this does not reorder.
    fn interleave(&mut self) -> Result<(), MuxError> {
        self.building()
    }

    fn cancel(&mut self) {
        let _ = self.fail(MuxError::Cancelled);
    }

    fn finish(&mut self, durations: &[TrackDuration]) -> Result<FinishedMux, FailedMux> {
        if let Err(error) = self.building() {
            return Err(FailedMux::incomplete(self.payload, error));
        }
        if let Err(error) = self.finish_file(durations) {
            let error = self.fail(error);
            return Err(FailedMux::incomplete(self.payload, error));
        }
        self.status = MuxStatus::Finished;
        Ok(FinishedMux::complete(
            durations.to_vec(),
            IndexPlace::End,
            MuxMode::Progressive,
            false,
            Vec::new(),
            self.payload,
        ))
    }

    fn status(&self) -> MuxStatus {
        self.status
    }

    /// Stores `config` as the stream's extradata. Before the headers are
    /// written it replaces what the stream was added with; after, it must
    /// have the same length.
    fn set_config(&mut self, track: TrackId, config: &[u8]) -> Result<(), MuxError> {
        self.building()?;
        let index = self
            .track_index(track.get())
            .map_err(|error| self.fail(error))?;
        if matches!(self.tracks[index].kind, Kind::Pcm { .. }) {
            return Ok(());
        }
        let size = u64::try_from(config.len()).unwrap_or(u64::MAX);
        if size > self.limits.max_extradata_bytes {
            return Err(self.fail(MuxError::HostileSize {
                declared: size,
                limit: self.limits.max_extradata_bytes,
            }));
        }
        let previous = std::mem::replace(&mut self.tracks[index].config, config.to_vec());
        if !self.started {
            return Ok(());
        }
        let strf = strf_bytes(&self.tracks[index]);
        match strf {
            Ok(strf) if strf.len() == self.tracks[index].strf_len => {
                let strh = self.strh_bytes(&self.tracks[index]);
                let (strf_at, strh_at) = (self.tracks[index].strf_at, self.tracks[index].strh_at);
                self.patch(strf_at, &strf)
                    .and_then(|()| self.patch(strh_at, &strh))
                    .map_err(|error| self.fail(error))
            }
            _ => {
                self.tracks[index].config = previous;
                Err(rejected(
                    "a codec configuration of another size after the first packet",
                ))
            }
        }
    }
}

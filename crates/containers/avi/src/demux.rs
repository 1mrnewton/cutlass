//! AVI demuxer.
//!
//! Open reads chunk headers, `hdrl`, and an index (see [`crate::index`]);
//! it does not read packet payloads except the first bytes of video chunks
//! when no index is usable. Reads then go chunk by chunk from the tables.
//!
//! Packets come back in file order when the streams are interleaved, and
//! in time order when one stream runs more than [`MAX_DRIFT_SECONDS`]
//! ahead of another at the same place in the file, which is how FFmpeg
//! reads a non-interleaved AVI.
//!
//! Every packet carries its decode time and duration in the stream's own
//! `dwRate / dwScale` clock. AVI stores no presentation time. Audio, and
//! video whose frames are never reordered (Motion JPEG, raw, VP8, VP9,
//! AV1), carry the decode time as the presentation time. Video that may
//! reorder frames (H.264, HEVC, MPEG-4 Part 2, MPEG-2) carries none, as
//! FFmpeg's demuxer reports none; the decoder puts its frames in order.
//!
//! An MP3, MPEG-1 Layer I or II, AC-3 or E-AC-3 chunk is cut into its
//! frames, one packet each, as FFmpeg's demuxer does through its parsers; a
//! frame that runs on into the stream's next chunk is joined to the rest of
//! it (see [`Cutting`]). On a constant-bitrate stream a frame's time is the
//! time of its first byte. On a variable-bitrate stream the first frame that
//! starts in a chunk takes the chunk's time and each later one follows the
//! one before by its sample count. Bytes between frames that no header
//! claims are dropped, and a chunk with no frame header at all is returned
//! whole.
//!
//! A read that fails on the chunk itself (its header disagrees with the
//! index, or it is larger than [`Limits::max_packet_bytes`]) moves past
//! that chunk, so the next read returns the one after it. A read that fails
//! in the input (it ended early, or the reader failed) stays on the chunk.

use std::cmp::Ordering;
use std::io::{Read, Seek};

use cutlass_codec::{ContainerId, Packet, Timescale, Timestamp};
use cutlass_container::{
    Demux, Gap, Landing, Limits, OpenError, ReadError, SeekError, SeekLanding, SeekTo, Stream,
    TrackId, Trim, compare_ticks, seek_with_index,
};

use crate::format::{StreamKind, describe_audio, describe_other, describe_video};
use crate::frames::{Framing, HEADER_BYTES, MAX_FRAME_BYTES};
use crate::header::{MainHeader, StreamHeader, parse_hdrl};
use crate::index::{self, Fallback, IndexSource, stream_of};
use crate::layout::read_layout;
use crate::limits::MAX_DRIFT_SECONDS;
use crate::source::Source;
use crate::track::{Clock, Track};

/// An open AVI or OpenDML AVI file.
pub struct AviDemux<R: Read + Seek> {
    source: Source<R>,
    limits: Limits,
    main: MainHeader,
    odml_frames: Option<u32>,
    headers: Vec<StreamHeader>,
    streams: Vec<Stream>,
    tracks: Vec<Track>,
    cursors: Vec<usize>,
    interleaved: bool,
    index_source: IndexSource,
    fallbacks: Vec<Fallback>,
    resyncs: u32,
    truncated: bool,
    riffs: u32,
    cancelled: bool,
    cutting: Option<Cutting>,
    /// Per track: the start of a frame that the track's next chunk ends.
    carries: Vec<Option<Carry>>,
}

/// A chunk of a framed audio stream, being returned one frame at a time.
struct Cutting {
    track: usize,
    /// A carried partial frame, then the chunk's bytes.
    bytes: Vec<u8>,
    at: usize,
    /// Where the chunk's own bytes begin in `bytes`.
    chunk_at: usize,
    /// Constant bitrate: the stream offset of `bytes[0]`.
    offset: u64,
    /// Variable bitrate: the chunk's time, and the next frame's.
    chunk_time: i64,
    time: i64,
    /// A frame starting in the chunk's own bytes has been returned.
    started: bool,
    /// The chunk's own row, for a chunk returned whole.
    row: usize,
}

/// The start of a frame whose end is in the stream's next chunk.
struct Carry {
    bytes: Vec<u8>,
    offset: u64,
    time: i64,
}

enum Chunk {
    Packet(Packet),
    Cut,
    End,
}

impl<R: Read + Seek> AviDemux<R> {
    /// Opens a random-access AVI. `length` is the file size.
    pub fn open(reader: R, length: u64) -> Result<Self, OpenError> {
        Self::open_with_limits(reader, length, Limits::standard())
    }

    pub fn open_with_limits(reader: R, length: u64, limits: Limits) -> Result<Self, OpenError> {
        let mut source = Source::new(reader, length);
        let layout = read_layout(&mut source, &limits)?;
        let hdrl = parse_hdrl(&layout.hdrl, &limits)?;
        let mut streams = Vec::with_capacity(hdrl.streams.len());
        let mut tracks = Vec::with_capacity(hdrl.streams.len());
        let mut headers = Vec::with_capacity(hdrl.streams.len());
        for (number, raw) in hdrl.streams.into_iter().enumerate() {
            let number = u32::try_from(number).map_err(|_| OpenError::HostileSize {
                declared: u64::MAX,
                limit: u64::from(limits.max_tracks),
            })?;
            let kind = StreamKind::of(raw.header.kind);
            let described = match kind {
                StreamKind::Video => describe_video(&raw.format, &hdrl.main)?,
                StreamKind::Audio => describe_audio(&raw.format)?,
                StreamKind::Text | StreamKind::Other => {
                    describe_other(raw.header.kind, raw.header.handler)?
                }
            };
            let framing = if kind == StreamKind::Audio {
                Framing::of(described.params.codec())
            } else {
                None
            };
            let clock = Clock::new(
                &raw.header,
                &hdrl.main,
                kind,
                described.codec,
                described.block_align,
            );
            streams.push(Stream::new(
                TrackId::new(number),
                described.params,
                Timescale::new(clock.timescale),
                None,
                Vec::new(),
                Trim::default(),
            ));
            let mut track = Track::new(
                number,
                kind,
                described.codec,
                clock,
                described.intra,
                described.presentation,
                raw.super_index,
            );
            track.framing = framing;
            tracks.push(track);
            headers.push(raw.header);
        }
        let built = index::build(&mut source, &mut tracks, &layout, &limits)?;
        let interleaved = interleaved(&tracks);
        let cursors = vec![0; tracks.len()];
        let carries = tracks.iter().map(|_| None).collect();
        Ok(Self {
            source,
            limits,
            main: hdrl.main,
            odml_frames: hdrl.odml_frames,
            headers,
            streams,
            tracks,
            cursors,
            interleaved,
            index_source: built.source,
            fallbacks: built.fallbacks,
            resyncs: built.resyncs,
            truncated: layout.truncated,
            riffs: layout.riffs,
            cancelled: false,
            cutting: None,
            carries,
        })
    }

    /// The `avih` main header.
    #[must_use]
    pub const fn main_header(&self) -> &MainHeader {
        &self.main
    }

    /// `dwTotalFrames` from the OpenDML `dmlh` header: the frame count
    /// across every `RIFF`, where `avih` counts the first only.
    #[must_use]
    pub const fn odml_total_frames(&self) -> Option<u32> {
        self.odml_frames
    }

    /// The `strh` header of stream `track`.
    #[must_use]
    pub fn stream_header(&self, track: u32) -> Option<&StreamHeader> {
        self.headers.get(usize::try_from(track).ok()?)
    }

    /// Where the chunk tables came from.
    #[must_use]
    pub const fn index_source(&self) -> IndexSource {
        self.index_source
    }

    /// Indexes the file has that were not used, and why.
    #[must_use]
    pub fn fallbacks(&self) -> &[Fallback] {
        &self.fallbacks
    }

    /// How many times a scan skipped garbage to find the next chunk.
    #[must_use]
    pub const fn resyncs(&self) -> u32 {
        self.resyncs
    }

    /// A `RIFF` or `movi` size ran past the end of the file.
    #[must_use]
    pub const fn truncated(&self) -> bool {
        self.truncated
    }

    /// Packets come back in file order; `false` means in time order.
    #[must_use]
    pub const fn interleaved(&self) -> bool {
        self.interleaved
    }

    /// `RIFF` segments: one for AVI 1.0, more for an OpenDML file.
    #[must_use]
    pub const fn riff_count(&self) -> u32 {
        self.riffs
    }

    /// Chunks in the stream's table. A framed audio chunk (see the module
    /// documentation) may return several packets.
    #[must_use]
    pub fn packet_count(&self, track: u32) -> Option<usize> {
        Some(self.tracks.get(usize::try_from(track).ok()?)?.entries.len())
    }

    /// The end of the stream's last packet, in its timescale.
    #[must_use]
    pub fn duration_ticks(&self, track: u32) -> Option<i64> {
        self.tracks.get(usize::try_from(track).ok()?)?.end()
    }

    /// The packet the next read returns: its track and table row.
    fn next(&self) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (index, track) in self.tracks.iter().enumerate() {
            let Some(entry) = track.entries.get(self.cursors[index]) else {
                continue;
            };
            let Some(current) = best else {
                best = Some(index);
                continue;
            };
            let other = &self.tracks[current].entries[self.cursors[current]];
            let order = if self.interleaved {
                entry.byte_position.cmp(&other.byte_position)
            } else {
                compare_ticks(
                    entry.time,
                    Timescale::new(track.clock.timescale),
                    other.time,
                    Timescale::new(self.tracks[current].clock.timescale),
                )
                .unwrap_or(Ordering::Equal)
                .then(entry.byte_position.cmp(&other.byte_position))
            };
            if order == Ordering::Less {
                best = Some(index);
            }
        }
        best
    }

    fn read_next(&mut self) -> Result<Option<Packet>, ReadError> {
        loop {
            if self.cancelled {
                return Err(ReadError::Cancelled);
            }
            if let Some(packet) = self.cut_next()? {
                return Ok(Some(packet));
            }
            match self.read_chunk()? {
                Chunk::Packet(packet) => return Ok(Some(packet)),
                Chunk::Cut => {}
                Chunk::End => return self.flush_carry(),
            }
        }
    }

    fn read_chunk(&mut self) -> Result<Chunk, ReadError> {
        let Some(index) = self.next() else {
            return Ok(Chunk::End);
        };
        let row = self.cursors[index];
        let track = &self.tracks[index];
        let entry = track.entries[row];
        let span = track.spans[row];
        let skip = |demux: &mut Self, error: ReadError| {
            demux.cursors[index] = row + 1;
            Err(error)
        };
        let Some(header) = entry.byte_position.checked_sub(8) else {
            return skip(self, mismatch(entry.byte_position));
        };
        self.source.seek(header).map_err(ReadError::from_packet)?;
        let head: [u8; 8] = self.source.read_array().map_err(ReadError::from_packet)?;
        let declared = u32::from_le_bytes([head[4], head[5], head[6], head[7]]);
        let named = stream_of([head[0], head[1], head[2], head[3]]);
        if named != Some(track.number) || declared < span.size {
            return skip(self, mismatch(header));
        }
        let limit = self.limits.max_packet_bytes;
        let bytes = match self.source.read_vec(u64::from(span.size), limit) {
            Ok(bytes) => bytes,
            Err(error @ cutlass_container::SourceError::HostileSize { .. }) => {
                return skip(self, ReadError::from_packet(error));
            }
            Err(error) => return Err(ReadError::from_packet(error)),
        };
        let track = &self.tracks[index];
        if track.framing.is_some() {
            let carry = self.carries[index].take();
            let chunk_at = carry.as_ref().map_or(0, |carry| carry.bytes.len());
            let offset = track.offsets.get(row).copied().unwrap_or(0);
            let (joined, offset, time) = match carry {
                Some(mut carry) => {
                    carry.bytes.extend_from_slice(&bytes);
                    (carry.bytes, carry.offset, carry.time)
                }
                None => (bytes, offset, entry.time),
            };
            self.cutting = Some(Cutting {
                track: index,
                bytes: joined,
                at: 0,
                chunk_at,
                offset,
                chunk_time: entry.time,
                time,
                started: false,
                row,
            });
            self.cursors[index] = row + 1;
            return Ok(Chunk::Cut);
        }
        let presentation = track.presentation.then_some(entry.time);
        let timestamp =
            Timestamp::try_new(Some(entry.time), presentation, Some(i64::from(span.ticks)))
                .map_err(|error| ReadError::Source {
                    message: error.to_string(),
                })?;
        let packet = Packet::try_new(
            track.number,
            bytes,
            timestamp,
            Timescale::new(track.clock.timescale),
            entry.keyframe,
        )
        .map_err(|error| ReadError::Source {
            message: error.to_string(),
        })?;
        self.cursors[index] = row + 1;
        Ok(Chunk::Packet(packet))
    }

    /// The next frame of the chunk being cut, or `None` once it is spent;
    /// a partial frame at its end is kept for the stream's next chunk.
    fn cut_next(&mut self) -> Result<Option<Packet>, ReadError> {
        let Some(cutting) = self.cutting.as_mut() else {
            return Ok(None);
        };
        let track = &self.tracks[cutting.track];
        let Some(framing) = track.framing else {
            self.cutting = None;
            return Ok(None);
        };
        loop {
            let rest = &cutting.bytes[cutting.at..];
            if rest.is_empty() {
                self.cutting = None;
                return Ok(None);
            }
            match framing.frame(rest) {
                Some(frame) if frame.len <= rest.len() => {
                    let start = cutting.at;
                    cutting.at += frame.len;
                    let (time, ticks) = if track.clock.sample_size > 0 {
                        let from = cutting.offset.saturating_add(start as u64);
                        let time = track.byte_time(from).ok_or_else(too_long)?;
                        let end = track
                            .byte_time(from.saturating_add(frame.len as u64))
                            .ok_or_else(too_long)?;
                        (time, end.saturating_sub(time))
                    } else {
                        let time = if start >= cutting.chunk_at && !cutting.started {
                            cutting.chunk_time
                        } else {
                            cutting.time
                        };
                        let ticks = frame_ticks(frame.samples, frame.sample_rate, &track.clock);
                        cutting.time = time.saturating_add(ticks);
                        (time, ticks)
                    };
                    if start >= cutting.chunk_at {
                        cutting.started = true;
                    }
                    let bytes = cutting.bytes[start..cutting.at].to_vec();
                    return packet(track.number, bytes, time, ticks, track.clock.timescale)
                        .map(Some);
                }
                Some(frame) if frame.len <= MAX_FRAME_BYTES => break,
                None if rest.len() < HEADER_BYTES => break,
                _ => {}
            }
            // No frame starts here: a whole chunk with no frame header in it
            // is returned as it is; otherwise skip to the next possible start.
            let next = rest
                .iter()
                .skip(1)
                .position(|&byte| byte == sync_byte(framing))
                .map(|skip| skip + 1);
            match next {
                Some(skip) => cutting.at += skip,
                None if cutting.at == 0 && cutting.chunk_at == 0 => {
                    let whole = std::mem::take(&mut cutting.bytes);
                    let row = cutting.row;
                    let span = track.spans[row];
                    let time = track.entries[row].time;
                    self.cutting = None;
                    return packet(
                        track.number,
                        whole,
                        time,
                        i64::from(span.ticks),
                        track.clock.timescale,
                    )
                    .map(Some);
                }
                None => {
                    self.cutting = None;
                    return Ok(None);
                }
            }
        }
        // A frame that ends in the stream's next chunk.
        let rest = cutting.bytes[cutting.at..].to_vec();
        let offset = cutting.offset.saturating_add(cutting.at as u64);
        let time = if cutting.at >= cutting.chunk_at && !cutting.started {
            cutting.chunk_time
        } else {
            cutting.time
        };
        let index = cutting.track;
        self.cutting = None;
        self.carries[index] = Some(Carry {
            bytes: rest,
            offset,
            time,
        });
        Ok(None)
    }

    /// At the end of the tables, a carried partial frame is returned as it is.
    fn flush_carry(&mut self) -> Result<Option<Packet>, ReadError> {
        let Some(index) = self.carries.iter().position(Option::is_some) else {
            return Ok(None);
        };
        let Some(carry) = self.carries[index].take() else {
            return Ok(None);
        };
        let track = &self.tracks[index];
        let ticks = if track.clock.sample_size > 0 {
            let end = carry.offset.saturating_add(carry.bytes.len() as u64);
            track
                .byte_time(end)
                .map_or(0, |end| end.saturating_sub(carry.time_or(track)))
        } else {
            0
        };
        let time = carry.time_or(track);
        packet(
            track.number,
            carry.bytes,
            time,
            ticks,
            track.clock.timescale,
        )
        .map(Some)
    }

    fn seek_to(&mut self, to: SeekTo) -> Result<SeekLanding, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        let target = usize::try_from(to.track)
            .ok()
            .filter(|&index| index < self.tracks.len())
            .ok_or(ReadError::Seek(SeekError::UnknownTrack))?;
        if self.tracks[target].entries.is_empty() {
            return Err(ReadError::Seek(SeekError::Empty));
        }
        let landing = seek_with_index(&self.tracks[target].entries, to).map_err(ReadError::Seek)?;
        let mut cursors = vec![0; self.tracks.len()];
        match landing.landing {
            Landing::End => {
                for (cursor, track) in cursors.iter_mut().zip(&self.tracks) {
                    *cursor = track.entries.len();
                }
            }
            Landing::Packet { ordinal, time, .. } => {
                let scale = i128::from(self.tracks[target].clock.timescale);
                for (index, track) in self.tracks.iter().enumerate() {
                    if index == target {
                        cursors[index] = usize::try_from(ordinal).unwrap_or(usize::MAX);
                        continue;
                    }
                    if track.entries.is_empty() {
                        continue;
                    }
                    let ticks = i128::from(time) * i128::from(track.clock.timescale);
                    let there = i64::try_from(ticks.div_euclid(scale.max(1))).unwrap_or(i64::MAX);
                    let follow = SeekTo {
                        track: track.number,
                        time: there,
                        end: None,
                    };
                    cursors[index] = match seek_with_index(&track.entries, follow) {
                        Ok(SeekLanding {
                            landing: Landing::Packet { ordinal, .. },
                            ..
                        }) => usize::try_from(ordinal).unwrap_or(usize::MAX),
                        _ => track.entries.len(),
                    };
                }
            }
        }
        self.cursors = cursors;
        self.cutting = None;
        for carry in &mut self.carries {
            *carry = None;
        }
        Ok(landing)
    }
}

impl Carry {
    /// The carried frame's start time.
    fn time_or(&self, track: &Track) -> i64 {
        if track.clock.sample_size > 0 {
            track.byte_time(self.offset).unwrap_or(self.time)
        } else {
            self.time
        }
    }
}

const fn sync_byte(framing: Framing) -> u8 {
    match framing {
        Framing::Mpeg => 0xFF,
        Framing::Ac3 => 0x0B,
    }
}

/// `samples` at `rate` in the stream's ticks, to the nearest tick.
fn frame_ticks(samples: u32, rate: u32, clock: &Clock) -> i64 {
    let numerator = u128::from(samples) * u128::from(clock.timescale);
    let rate = u128::from(rate.max(1));
    i64::try_from((numerator + rate / 2) / rate).unwrap_or(i64::MAX)
}

fn packet(
    track: u32,
    bytes: Vec<u8>,
    time: i64,
    ticks: i64,
    timescale: u32,
) -> Result<Packet, ReadError> {
    let source = |error: &dyn std::fmt::Display| ReadError::Source {
        message: error.to_string(),
    };
    let timestamp =
        Timestamp::try_new(Some(time), Some(time), Some(ticks)).map_err(|error| source(&error))?;
    Packet::try_new(track, bytes, timestamp, Timescale::new(timescale), true)
        .map_err(|error| source(&error))
}

fn too_long() -> ReadError {
    ReadError::Source {
        message: "avi: a frame time does not fit".to_owned(),
    }
}

fn mismatch(position: u64) -> ReadError {
    ReadError::Source {
        message: format!("avi: the chunk at {position} is not the one the index names"),
    }
}

/// `false` when, walking every table in file order, a chunk's time is more
/// than [`MAX_DRIFT_SECONDS`] behind the latest time already passed.
fn interleaved(tracks: &[Track]) -> bool {
    let mut cursors = vec![0usize; tracks.len()];
    let mut lead: Option<(i128, i128)> = None;
    loop {
        let mut best: Option<usize> = None;
        for (index, track) in tracks.iter().enumerate() {
            let Some(entry) = track.entries.get(cursors[index]) else {
                continue;
            };
            if best.is_none_or(|current| {
                entry.byte_position < tracks[current].entries[cursors[current]].byte_position
            }) {
                best = Some(index);
            }
        }
        let Some(index) = best else {
            return true;
        };
        let track = &tracks[index];
        let time = i128::from(track.entries[cursors[index]].time);
        let scale = i128::from(track.clock.timescale.max(1));
        cursors[index] += 1;
        match lead {
            None => lead = Some((time, scale)),
            Some((lead_time, lead_scale)) => {
                let ahead = lead_time * scale - time * lead_scale;
                if ahead > MAX_DRIFT_SECONDS * lead_scale * scale {
                    return false;
                }
                if ahead < 0 {
                    lead = Some((time, scale));
                }
            }
        }
    }
}

impl<R: Read + Seek + Send> Demux for AviDemux<R> {
    fn family(&self) -> ContainerId {
        ContainerId::Avi
    }

    fn streams(&self) -> &[Stream] {
        &self.streams
    }

    fn gaps(&self) -> &[Gap] {
        &[]
    }

    fn read_packet(&mut self) -> Result<Option<Packet>, ReadError> {
        self.read_next()
    }

    fn seek(&mut self, to: SeekTo) -> Result<SeekLanding, ReadError> {
        self.seek_to(to)
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }

    fn consumed(&self) -> u64 {
        self.source.consumed()
    }
}

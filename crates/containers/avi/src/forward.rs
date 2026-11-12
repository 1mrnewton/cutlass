//! AVI from a reader that cannot seek: a pipe, a socket, a download.
//!
//! [`AviForwardDemux`] reads `hdrl`, then every chunk of every `movi`
//! list in file order, `RIFF AVIX` segments included, and never goes back.
//! No index is read: `idx1` sits after the packets and `ix##` beside them,
//! and neither is needed to read in order. What that costs, against
//! [`crate::AviDemux`]:
//!
//! - Packets come in file order. A file whose streams are not interleaved
//!   is not put in time order.
//! - A video keyframe is told from the start of its bitstream, as the
//!   rebuild scan does (Motion JPEG and raw video are all keyframes); the
//!   index's keyframe flags are not seen.
//! - A video packet is returned when the stream's next packet arrives, so
//!   that the empty chunks after it (dropped frames) lengthen it, as they
//!   do in [`crate::AviDemux`].
//! - Seeking is [`SeekError::ForwardOnly`].
//!
//! Times, durations, and framed audio come out the same as from
//! [`crate::AviDemux`]; the integration tests compare the two.

use std::collections::VecDeque;
use std::io::{ErrorKind, Read};

use cutlass_codec::{ContainerId, Packet, Timescale, Timestamp};
use cutlass_container::{
    Demux, Gap, Limits, OpenError, ReadError, SeekError, SeekLanding, SeekTo, SourceError, Stream,
};

use crate::cut::{self, Cutter};
use crate::demux::build_streams;
use crate::format::StreamKind;
use crate::header::{MainHeader, StreamHeader, bad, parse_hdrl};
use crate::index::stream_of;
use crate::limits::{MAX_CHUNKS, SNIFF_BYTES};
use crate::track::Track;

/// An AVI or OpenDML AVI read once, front to back.
pub struct AviForwardDemux<R: Read> {
    reader: R,
    consumed: u64,
    limits: Limits,
    main: MainHeader,
    headers: Vec<StreamHeader>,
    streams: Vec<Stream>,
    tracks: Vec<Track>,
    cutter: Cutter,
    /// Where the current `movi` list ends; `None` between lists.
    movi_end: Option<u64>,
    /// Per video track: the packet waiting for the stream's next chunk.
    held: Vec<Option<Packet>>,
    ready: VecDeque<Packet>,
    ended: bool,
    cancelled: bool,
}

impl<R: Read> AviForwardDemux<R> {
    /// Reads the headers up to the first `movi` list.
    pub fn open(reader: R) -> Result<Self, OpenError> {
        Self::open_with_limits(reader, Limits::standard())
    }

    pub fn open_with_limits(reader: R, limits: Limits) -> Result<Self, OpenError> {
        let mut demux = Self {
            reader,
            consumed: 0,
            limits,
            main: MainHeader::default(),
            headers: Vec::new(),
            streams: Vec::new(),
            tracks: Vec::new(),
            cutter: Cutter::new(0),
            movi_end: None,
            held: Vec::new(),
            ready: VecDeque::new(),
            ended: false,
            cancelled: false,
        };
        let head: [u8; 12] = demux.array().map_err(OpenError::from_header)?;
        match crate::probe::verdict(&head) {
            cutlass_container::HeaderVerdict::Match => {}
            cutlass_container::HeaderVerdict::UnsupportedBrand => {
                return Err(OpenError::UnsupportedBrand {
                    family: ContainerId::Avi,
                });
            }
            cutlass_container::HeaderVerdict::Unrecognized => {
                return Err(OpenError::Unrecognized {
                    family: ContainerId::Avi,
                });
            }
        }
        let mut hdrl = None;
        for _ in 0..MAX_CHUNKS {
            let (id, size) = demux.chunk_header().map_err(OpenError::from_header)?;
            if &id == b"LIST" && size >= 4 {
                let form: [u8; 4] = demux.array().map_err(OpenError::from_header)?;
                match &form {
                    b"hdrl" if hdrl.is_none() => {
                        let bytes = demux
                            .vec(u64::from(size) - 4, limits.max_header_bytes)
                            .map_err(OpenError::from_header)?;
                        demux.pad(size).map_err(OpenError::from_header)?;
                        hdrl = Some(parse_hdrl(&bytes, &limits)?);
                    }
                    b"movi" => {
                        let Some(hdrl) = hdrl.take() else {
                            return Err(bad("movi comes before hdrl"));
                        };
                        demux.movi_end = Some(list_end(demux.consumed, size));
                        let (streams, tracks, headers) =
                            build_streams(hdrl.streams, &hdrl.main, &limits)?;
                        demux.main = hdrl.main;
                        demux.cutter = Cutter::new(tracks.len());
                        demux.held = tracks.iter().map(|_| None).collect();
                        demux.streams = streams;
                        demux.tracks = tracks;
                        demux.headers = headers;
                        return Ok(demux);
                    }
                    _ => demux
                        .skip(u64::from(size) - 4, size)
                        .map_err(OpenError::from_header)?,
                }
            } else {
                demux
                    .skip(u64::from(size), size)
                    .map_err(OpenError::from_header)?;
            }
        }
        Err(bad("no movi list in the first chunks"))
    }

    /// The `avih` main header.
    #[must_use]
    pub const fn main_header(&self) -> &MainHeader {
        &self.main
    }

    /// The `strh` header of stream `track`.
    #[must_use]
    pub fn stream_header(&self, track: u32) -> Option<&StreamHeader> {
        self.headers.get(usize::try_from(track).ok()?)
    }

    fn read_next(&mut self) -> Result<Option<Packet>, ReadError> {
        loop {
            if self.cancelled {
                return Err(ReadError::Cancelled);
            }
            if let Some(packet) = self.ready.pop_front() {
                return Ok(Some(packet));
            }
            if let Some(packet) = self.cutter.next(&self.tracks)? {
                return Ok(Some(packet));
            }
            if self.ended {
                return self.cutter.flush(&self.tracks);
            }
            self.step()?;
        }
    }

    /// Reads one chunk header and what it holds.
    fn step(&mut self) -> Result<(), ReadError> {
        let Some(end) = self.movi_end else {
            return self.between_lists();
        };
        if self.consumed.saturating_add(8) > end {
            self.movi_end = None;
            let rest = end.saturating_sub(self.consumed);
            return self.discard(rest);
        }
        let Some((id, size)) = self.chunk_or_end()? else {
            return Ok(());
        };
        if &id == b"LIST" {
            // A `rec ` list holds ordinary chunks; read on inside it.
            return match self.array::<4>() {
                Ok(_) => Ok(()),
                Err(error) => self.end_on(error),
            };
        }
        let track = stream_of(id)
            .and_then(|number| usize::try_from(number).ok())
            .filter(|&index| index < self.tracks.len() && !is_index(id));
        let Some(index) = track else {
            return self
                .skip(u64::from(size), size)
                .or_else(|error| self.end_on(error));
        };
        if u64::from(size) > self.limits.max_packet_bytes {
            self.skip(u64::from(size), size)
                .or_else(|error| self.end_on(error))?;
            return Err(ReadError::HostileSize {
                declared: u64::from(size),
                limit: self.limits.max_packet_bytes,
            });
        }
        let data = self.consumed;
        let bytes = match self.vec(u64::from(size), self.limits.max_packet_bytes) {
            Ok(bytes) => bytes,
            Err(error) => return self.end_on(error),
        };
        if let Err(error) = self.pad(size) {
            self.end_on(error)?;
        }
        self.chunk(index, data, bytes)
    }

    /// One stream chunk of `bytes`, whose payload started at `data`.
    fn chunk(&mut self, index: usize, data: u64, bytes: Vec<u8>) -> Result<(), ReadError> {
        let track = &mut self.tracks[index];
        let video = track.kind == StreamKind::Video;
        if bytes.is_empty() {
            if video {
                // A dropped frame lengthens the packet held before it.
                let mut budget = u64::MAX;
                track
                    .push(data, 0, false, &mut budget)
                    .map_err(open_to_read)?;
            }
            return Ok(());
        }
        let key = !video
            || track.intra
            || crate::sniff::keyframe(track.codec, &bytes[..bytes.len().min(SNIFF_BYTES_USIZE)]);
        if video && let Some(held) = self.held[index].take() {
            let ticks = track.spans.last().map_or(0, |span| i64::from(span.ticks));
            self.ready.push_back(relength(held, ticks)?);
        }
        track.forget();
        let mut budget = u64::MAX;
        let size = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
        track
            .push(data, size, key, &mut budget)
            .map_err(open_to_read)?;
        let (Some(entry), Some(span)) =
            (track.entries.last().copied(), track.spans.last().copied())
        else {
            return Ok(());
        };
        if track.framing.is_some() {
            self.cutter.begin(cut::Chunk {
                track: index,
                bytes,
                offset: track.offsets.last().copied().unwrap_or(0),
                time: entry.time,
                ticks: i64::from(span.ticks),
            });
            return Ok(());
        }
        let presentation = track.presentation.then_some(entry.time);
        let timestamp =
            Timestamp::try_new(Some(entry.time), presentation, Some(i64::from(span.ticks)))
                .map_err(source)?;
        let packet = Packet::try_new(
            track.number,
            bytes,
            timestamp,
            Timescale::new(track.clock.timescale),
            entry.keyframe,
        )
        .map_err(source)?;
        if video {
            self.held[index] = Some(packet);
        } else {
            self.ready.push_back(packet);
        }
        Ok(())
    }

    /// Between `movi` lists: the next `RIFF AVIX`, its `movi`, or the end.
    fn between_lists(&mut self) -> Result<(), ReadError> {
        let Some((id, size)) = self.chunk_or_end()? else {
            return Ok(());
        };
        match &id {
            b"RIFF" => match self.array::<4>() {
                Ok(_) => Ok(()),
                Err(error) => self.end_on(error),
            },
            b"LIST" if size >= 4 => {
                let form = match self.array::<4>() {
                    Ok(form) => form,
                    Err(error) => return self.end_on(error),
                };
                if &form == b"movi" {
                    self.movi_end = Some(list_end(self.consumed, size));
                    Ok(())
                } else {
                    self.skip(u64::from(size) - 4, size)
                        .or_else(|error| self.end_on(error))
                }
            }
            _ => self
                .skip(u64::from(size), size)
                .or_else(|error| self.end_on(error)),
        }
    }

    /// The input ended: every held packet goes out, then carried frames.
    fn finish(&mut self) {
        self.ended = true;
        for index in 0..self.held.len() {
            if let Some(held) = self.held[index].take() {
                let ticks = self.tracks[index]
                    .spans
                    .last()
                    .map_or(0, |span| i64::from(span.ticks));
                if let Ok(packet) = relength(held, ticks) {
                    self.ready.push_back(packet);
                }
            }
        }
    }

    /// A read that ran out of input ends the file, as a cut recording
    /// does: the chunk it was in is not a packet. Other failures are errors.
    fn end_on(&mut self, error: SourceError) -> Result<(), ReadError> {
        match error {
            SourceError::Truncated { .. } | SourceError::ShortRead { .. } => {
                self.finish();
                Ok(())
            }
            other => Err(ReadError::from_packet(other)),
        }
    }

    fn chunk_or_end(&mut self) -> Result<Option<([u8; 4], u32)>, ReadError> {
        match self.chunk_header() {
            Ok(header) => Ok(Some(header)),
            Err(error) => self.end_on(error).map(|()| None),
        }
    }

    fn discard(&mut self, bytes: u64) -> Result<(), ReadError> {
        self.skip(bytes, 0).or_else(|error| self.end_on(error))
    }

    fn chunk_header(&mut self) -> Result<([u8; 4], u32), SourceError> {
        let head: [u8; 8] = self.array()?;
        Ok((
            [head[0], head[1], head[2], head[3]],
            u32::from_le_bytes([head[4], head[5], head[6], head[7]]),
        ))
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], SourceError> {
        let mut out = [0; N];
        self.fill(&mut out)?;
        Ok(out)
    }

    /// `len` bytes, refused above `limit` before anything is allocated.
    fn vec(&mut self, len: u64, limit: u64) -> Result<Vec<u8>, SourceError> {
        if len > limit {
            return Err(SourceError::HostileSize {
                declared: len,
                limit,
            });
        }
        let mut out = vec![
            0;
            usize::try_from(len).map_err(|_| SourceError::HostileSize {
                declared: len,
                limit,
            })?
        ];
        self.fill(&mut out)?;
        Ok(out)
    }

    /// Skips `len` bytes and the pad byte an odd `size` leaves.
    fn skip(&mut self, len: u64, size: u32) -> Result<(), SourceError> {
        let want = len.saturating_add(u64::from(size & 1));
        let moved = std::io::copy(&mut (&mut self.reader).take(want), &mut std::io::sink())
            .map_err(|error| SourceError::Failed {
                message: error.to_string(),
            })?;
        self.consumed = self.consumed.saturating_add(moved);
        if moved < want {
            return Err(SourceError::ShortRead {
                expected: want,
                got: moved,
            });
        }
        Ok(())
    }

    fn pad(&mut self, size: u32) -> Result<(), SourceError> {
        if size & 1 == 1 {
            self.array::<1>().map(|_| ())
        } else {
            Ok(())
        }
    }

    fn fill(&mut self, out: &mut [u8]) -> Result<(), SourceError> {
        let mut filled = 0;
        while filled < out.len() {
            match self.reader.read(&mut out[filled..]) {
                Ok(0) => {
                    return Err(SourceError::ShortRead {
                        expected: out.len() as u64,
                        got: filled as u64,
                    });
                }
                Ok(read) => {
                    filled += read;
                    self.consumed = self.consumed.saturating_add(read as u64);
                }
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(error) => {
                    return Err(SourceError::Failed {
                        message: error.to_string(),
                    });
                }
            }
        }
        Ok(())
    }
}

const SNIFF_BYTES_USIZE: usize = SNIFF_BYTES as usize;

/// Where a list whose size field ended at `after_form - 4` ends; a size of
/// zero, which live writers leave, runs to the end of the input.
fn list_end(after_form: u64, size: u32) -> u64 {
    if size == 0 {
        u64::MAX
    } else {
        after_form.saturating_sub(4).saturating_add(u64::from(size))
    }
}

/// `ix##`, `##ix`, and `idx1` hold indexes and `##pc` a palette change,
/// not packets.
fn is_index(id: [u8; 4]) -> bool {
    &id[..2] == b"ix" || &id[2..] == b"ix" || &id[2..] == b"pc" || &id == b"idx1"
}

fn relength(packet: Packet, ticks: i64) -> Result<Packet, ReadError> {
    let stamp = packet.timestamp();
    let timestamp =
        Timestamp::try_new(stamp.decode(), stamp.presentation(), Some(ticks)).map_err(source)?;
    let (stream, timescale, keyframe) = (packet.stream(), packet.timescale(), packet.keyframe());
    Packet::try_new(stream, packet.into_bytes(), timestamp, timescale, keyframe).map_err(source)
}

fn source(error: impl std::fmt::Display) -> ReadError {
    ReadError::Source {
        message: error.to_string(),
    }
}

fn open_to_read(error: OpenError) -> ReadError {
    source(error)
}

impl<R: Read + Send> Demux for AviForwardDemux<R> {
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

    fn seek(&mut self, _to: SeekTo) -> Result<SeekLanding, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        Err(ReadError::Seek(SeekError::ForwardOnly))
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }

    fn consumed(&self) -> u64 {
        self.consumed
    }
}

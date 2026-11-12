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
            tracks.push(Track::new(
                number,
                kind,
                described.codec,
                clock,
                described.intra,
                described.presentation,
                raw.super_index,
            ));
            headers.push(raw.header);
        }
        let built = index::build(&mut source, &mut tracks, &layout, &limits)?;
        let interleaved = interleaved(&tracks);
        let cursors = vec![0; tracks.len()];
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

    /// Packets in the stream's table.
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
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        let Some(index) = self.next() else {
            return Ok(None);
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
        let presentation = track.presentation.then_some(entry.time);
        let timestamp = Timestamp::try_new(
            Some(entry.time),
            presentation,
            Some(i64::from(span.ticks)),
        )
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
        Ok(Some(packet))
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
        Ok(landing)
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

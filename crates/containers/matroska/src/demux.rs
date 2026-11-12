//! The Matroska and WebM demuxer.
//!
//! Open reads the EBML header, finds the Segment, and walks its top-level
//! elements up to the first Cluster: SeekHead, Info, Tracks, and Cues when
//! they come first. Tracks and Info that sit after the Clusters are found
//! through the SeekHead. Cues after the Clusters are read on the first seek.
//!
//! Each read returns one frame. A laced block holds several frames; the
//! ones after the first wait in a queue for the next reads. Packets come
//! back in file order, which Matroska writers keep in decode order.
//!
//! # Timestamps
//!
//! A block's timestamp is its Cluster's Timestamp plus the block's signed
//! 16-bit offset, in Segment Ticks of TimestampScale nanoseconds (RFC 9559,
//! 11.1 and 11.2), and it is the presentation time. Packets carry that
//! value with the timescale `1e9 / TimestampScale` ticks per second, which is
//! the file's own clock: 1000 for the default millisecond scale. When
//! TimestampScale does not divide a second, the timescale is nanoseconds and
//! each tick is the Segment Tick times TimestampScale, an exact product.
//! The decode time is not stored in Matroska and stays missing.
//!
//! The duration is BlockDuration when the block has one, otherwise the
//! track's DefaultDuration (nanoseconds) rounded to the nearest tick. Frames
//! after the first in a lace start at the block time plus the durations
//! before them; with no duration known they have no presentation time.
//! CodecDelay is reported as an edit, not subtracted from packet times.

use std::collections::VecDeque;
use std::io::{Read, Seek};

use cutlass_codec::{ContainerId, Packet, Timescale, Timestamp};
use cutlass_container::{
    Demux, Edit, Gap, Input, Language, Limits, OpenError, ReadError, SeekLanding, SeekTo, Stream,
    TrackId, Trim,
};

use crate::block::{self, BlockHeader, KEYFRAME};
use crate::codec::{FrameFix, map_track};
use crate::ebml::{self, Children, Size};
use crate::error::{Error, Result};
use crate::ids;
use crate::limits::{MAX_CUES_BYTES, MAX_EBML_HEADER, MAX_TOP_LEVEL};
use crate::meta::{self, Cue, EbmlHeader, Info, SeekEntry, TrackEntry};
use crate::source::{Header, Source};

/// Nanoseconds in a second.
const SECOND_NS: u64 = 1_000_000_000;

/// The packet clock for one Segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Clock {
    /// Nanoseconds per Segment Tick.
    pub scale: u64,
    /// Ticks per second of the packets.
    pub ticks_per_second: u32,
    /// Packet ticks per Segment Tick: 1, or `scale` on the nanosecond clock.
    pub multiplier: i64,
}

impl Clock {
    /// The clock for a TimestampScale.
    pub fn new(scale: u64) -> Result<Self> {
        if scale == 0 {
            return Err(Error::malformed("TimestampScale is zero"));
        }
        if SECOND_NS.is_multiple_of(scale) {
            let ticks = u32::try_from(SECOND_NS / scale).unwrap_or(u32::MAX);
            return Ok(Self {
                scale,
                ticks_per_second: ticks,
                multiplier: 1,
            });
        }
        let multiplier = i64::try_from(scale)
            .map_err(|_| Error::unsupported(format!("TimestampScale {scale}")))?;
        Ok(Self {
            scale,
            ticks_per_second: 1_000_000_000,
            multiplier,
        })
    }

    #[must_use]
    pub const fn timescale(self) -> Timescale {
        Timescale::new(self.ticks_per_second)
    }

    /// Segment Ticks as packet ticks.
    #[must_use]
    pub fn from_segment(self, ticks: i64) -> Option<i64> {
        ticks.checked_mul(self.multiplier)
    }

    /// Packet ticks as Segment Ticks, rounded down.
    #[must_use]
    pub fn to_segment(self, ticks: i64) -> i64 {
        ticks.div_euclid(self.multiplier)
    }

    /// Nanoseconds as packet ticks, rounded to the nearest tick (RFC 9559, 11.3).
    #[must_use]
    pub fn from_ns(self, ns: u128) -> i64 {
        let ticks = (ns * u128::from(self.ticks_per_second) + u128::from(SECOND_NS / 2))
            / u128::from(SECOND_NS);
        i64::try_from(ticks).unwrap_or(i64::MAX)
    }
}

/// Per-track state the reader needs for each block.
#[derive(Clone, Debug)]
struct Track {
    number: u64,
    fix: FrameFix,
    /// Nanoseconds per frame.
    default_duration: Option<u64>,
}

/// The Cluster the reader is inside.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cluster {
    pub start: u64,
    pub end: Option<u64>,
    /// Segment Ticks, once the Timestamp element has been read.
    pub timestamp: Option<i64>,
    /// Blocks seen in this Cluster so far.
    pub blocks: u32,
}

/// Where the next read starts. A seek saves and restores all of it.
#[derive(Clone, Debug, Default)]
pub(crate) struct Cursor {
    pub cluster: Option<Cluster>,
    /// A header already read at the Segment level.
    pub peeked: Option<Header>,
    pub pending: VecDeque<Packet>,
    pub finished: bool,
}

/// Where the Cues are, or what came of loading them.
#[derive(Clone, Debug)]
/// Where a track's seek walks may start: (time in packet ticks, Cluster
/// offset), sorted by time, and whether they came from Cues.
pub(crate) struct Starts {
    pub(crate) track: u64,
    pub(crate) points: Vec<(i64, u64)>,
    pub(crate) from_cues: bool,
}

pub(crate) enum CueState {
    /// Not read yet. The SeekHead entries say where to look.
    Unread,
    Loaded(Vec<Cue>),
    /// No Cues, or Cues that did not parse. Seek walks the Clusters.
    Absent,
}

/// A Matroska or WebM file opened for reading.
pub struct MatroskaDemux<R> {
    pub(crate) source: Source<R>,
    pub(crate) header: EbmlHeader,
    pub(crate) info: Info,
    pub(crate) clock: Clock,
    tracks: Vec<Track>,
    entries: Vec<TrackEntry>,
    streams: Vec<Stream>,
    /// Offset of the Segment's first data byte (Segment Position 0).
    pub(crate) segment_data: u64,
    pub(crate) segment_end: Option<u64>,
    /// Offset of the first Cluster, when there is one.
    pub(crate) first_cluster: Option<u64>,
    pub(crate) seek_head: Vec<SeekEntry>,
    pub(crate) cues: CueState,
    /// Cluster starts and timestamps, walked on the first seek of a file
    /// without Cues.
    pub(crate) clusters: Option<Vec<(i64, u64)>>,
    /// The last seek's track and where its walks may start, kept so that
    /// the next seek on that track does not sort them again.
    pub(crate) starts: Option<Starts>,
    pub(crate) cursor: Cursor,
    pub(crate) limits: Limits,
    cancelled: bool,
}

impl<R: Read + Seek> MatroskaDemux<R> {
    /// Opens a random-access file of `length` bytes with the standard limits.
    pub fn open(reader: R, length: u64) -> std::result::Result<Self, OpenError> {
        Self::open_limited(reader, length, Limits::standard())
    }

    pub fn open_limited(
        reader: R,
        length: u64,
        limits: Limits,
    ) -> std::result::Result<Self, OpenError> {
        Self::open_input(Input::random(reader, length, limits)).map_err(Error::into_open)
    }

    /// Opens a stream read in order, such as a live recording whose Segment
    /// and Clusters have unknown sizes. Tracks must come before the first
    /// Cluster, and seek returns [`cutlass_container::SeekError::ForwardOnly`].
    pub fn open_forward(reader: R) -> std::result::Result<Self, OpenError> {
        Self::open_input(Input::forward(reader, Limits::standard())).map_err(Error::into_open)
    }

    fn open_input(input: Input<R>) -> Result<Self> {
        let limits = input.limits();
        let mut source = Source::new(input);
        let header = read_ebml_header(&mut source)?;
        let segment = find_segment(&mut source)?;
        let segment_data = segment.data;
        let segment_end = segment.end();
        let mut walk = TopLevel::default();
        walk_to_first_cluster(&mut source, segment_end, limits, &mut walk)?;
        if walk.info.is_none() || walk.tracks.is_none() {
            fetch_from_seek_head(&mut source, segment_data, limits, &mut walk)?;
            if let Some(cluster) = walk.first_cluster {
                source.seek(cluster.data)?;
            }
        }
        let info = walk.info.unwrap_or_default();
        let entries = walk
            .tracks
            .ok_or(Error::malformed("the Segment has no Tracks element"))?;
        let clock = Clock::new(info.timestamp_scale)?;
        let mut tracks = Vec::with_capacity(entries.len());
        let mut streams = Vec::with_capacity(entries.len());
        for entry in &entries {
            if (entry.track_timestamp_scale - 1.0).abs() > f64::EPSILON {
                return Err(Error::unsupported(format!(
                    "TrackTimestampScale {} on track {}",
                    entry.track_timestamp_scale, entry.number
                )));
            }
            let mapped = map_track(entry)?;
            streams.push(stream_for(
                entry,
                mapped.params,
                mapped.pre_skip,
                &info,
                clock,
            )?);
            tracks.push(Track {
                number: entry.number,
                fix: mapped.fix,
                default_duration: entry.default_duration,
            });
        }
        let cues = match walk.cues {
            Some(cues) => CueState::Loaded(cues),
            None => CueState::Unread,
        };
        let cursor = Cursor {
            cluster: None,
            peeked: walk.first_cluster,
            pending: VecDeque::new(),
            finished: walk.first_cluster.is_none(),
        };
        Ok(Self {
            source,
            header,
            info,
            clock,
            tracks,
            entries,
            streams,
            segment_data,
            segment_end,
            first_cluster: walk.first_cluster.map(|cluster| cluster.start),
            seek_head: walk.seek_head,
            cues,
            clusters: None,
            starts: None,
            cursor,
            limits,
            cancelled: false,
        })
    }

    /// The EBML header: DocType and its versions.
    #[must_use]
    pub const fn ebml_header(&self) -> EbmlHeader {
        self.header
    }

    /// The Info element: TimestampScale, Duration, and the writing apps.
    #[must_use]
    pub const fn info(&self) -> &Info {
        &self.info
    }

    /// The packet clock.
    #[must_use]
    pub const fn clock(&self) -> Clock {
        self.clock
    }

    /// The TrackEntry of each stream, in the order of [`Demux::streams`].
    #[must_use]
    pub fn track_entries(&self) -> &[TrackEntry] {
        &self.entries
    }

    /// The Segment duration in packet ticks, when Info states one.
    #[must_use]
    pub fn duration_ticks(&self) -> Option<i64> {
        duration_ticks(&self.info, self.clock)
    }

    fn track_index(&self, number: u64) -> Option<usize> {
        self.tracks.iter().position(|track| track.number == number)
    }

    /// The next frame, or `None` at the end of the Segment.
    fn next_packet(&mut self) -> Result<Option<Packet>> {
        if let Some(packet) = self.cursor.pending.pop_front() {
            return Ok(Some(packet));
        }
        loop {
            if self.cursor.finished {
                return Ok(None);
            }
            if let Some(cluster) = self.cursor.cluster
                && let Some(end) = cluster.end
                && self.source.position() >= end
            {
                self.cursor.cluster = None;
            }
            let header = match self.cursor.peeked.take() {
                Some(header) => header,
                None => {
                    if self
                        .segment_end
                        .is_some_and(|end| self.source.position() >= end)
                    {
                        self.cursor.finished = true;
                        return Ok(None);
                    }
                    match self.source.header()? {
                        Some(header) => header,
                        None => {
                            self.cursor.finished = true;
                            return Ok(None);
                        }
                    }
                }
            };
            match self.cursor.cluster {
                Some(cluster) => {
                    if let Some(packet) = self.in_cluster(cluster, header)? {
                        return Ok(Some(packet));
                    }
                }
                None => self.at_segment_level(header)?,
            }
        }
    }

    fn at_segment_level(&mut self, header: Header) -> Result<()> {
        if let (Some(end), Some(segment_end)) = (header.end(), self.segment_end)
            && end > segment_end
        {
            return Err(Error::Ebml(ebml::EbmlError::PastParent {
                id: header.id,
                size: end - header.data,
                room: segment_end.saturating_sub(header.data),
            }));
        }
        match header.id {
            ids::CLUSTER => {
                self.cursor.cluster = Some(Cluster {
                    start: header.start,
                    end: header.end(),
                    timestamp: None,
                    blocks: 0,
                });
                Ok(())
            }
            // A second EBML document or Segment: this reader stops at the
            // end of the first Segment.
            ids::EBML | ids::SEGMENT => {
                self.cursor.finished = true;
                Ok(())
            }
            _ => match header.size {
                Size::Known(size) => self.source.skip(size),
                Size::Unknown => Err(Error::Ebml(ebml::EbmlError::UnknownSize { id: header.id })),
            },
        }
    }

    fn in_cluster(&mut self, cluster: Cluster, header: Header) -> Result<Option<Packet>> {
        if cluster.end.is_none() && ids::ends_cluster(header.id) {
            self.cursor.cluster = None;
            self.cursor.peeked = Some(header);
            return Ok(None);
        }
        let Size::Known(size) = header.size else {
            return Err(Error::Ebml(ebml::EbmlError::UnknownSize { id: header.id }));
        };
        if let (Some(end), Some(cluster_end)) = (header.end(), cluster.end)
            && end > cluster_end
        {
            return Err(Error::Ebml(ebml::EbmlError::PastParent {
                id: header.id,
                size,
                room: cluster_end.saturating_sub(header.data),
            }));
        }
        match header.id {
            ids::TIMESTAMP => {
                let data = self.source.bytes(size, 8)?;
                let value = ebml::uint(header.id, &data)?;
                let value = i64::try_from(value)
                    .map_err(|_| Error::malformed("Cluster Timestamp does not fit"))?;
                if let Some(current) = self.cursor.cluster.as_mut() {
                    current.timestamp = Some(value);
                }
                Ok(None)
            }
            ids::SIMPLE_BLOCK => self.simple_block(size),
            ids::BLOCK_GROUP => self.block_group(size),
            _ => {
                self.source.skip(size)?;
                Ok(None)
            }
        }
    }

    fn bump_block(&mut self) -> Option<Cluster> {
        let cluster = self.cursor.cluster.as_mut()?;
        cluster.blocks = cluster.blocks.saturating_add(1);
        Some(*cluster)
    }

    fn simple_block(&mut self, size: u64) -> Result<Option<Packet>> {
        let cluster = self.bump_block();
        let mut head = [0_u8; block::MAX_HEADER];
        let first = {
            let mut one = [0_u8; 1];
            self.source.fill(&mut one)?;
            one[0]
        };
        let header_len = block::header_len(first).ok_or(Error::Ebml(ebml::EbmlError::BadSize))?;
        if u64::try_from(header_len).unwrap_or(u64::MAX) > size {
            return Err(Error::malformed("block is shorter than its header"));
        }
        head[0] = first;
        self.source.fill(&mut head[1..header_len])?;
        let header = block::parse_header(&head[..header_len])?;
        let body = size - u64::try_from(header_len).unwrap_or(0);
        let Some(index) = self.track_index(header.track) else {
            // A block for a track the Tracks element does not name.
            self.source.skip(body)?;
            return Ok(None);
        };
        let keyframe = header.flags & KEYFRAME != 0;
        let limit = self.block_limit();
        if header.lacing() == 0 {
            let fix = &self.tracks[index].fix;
            let bytes = match fix {
                FrameFix::None => self.source.bytes(body, limit)?,
                FrameFix::Prepend(prefix) => {
                    let mut bytes = prefix.clone();
                    self.source.append(&mut bytes, body, limit)?;
                    bytes
                }
                FrameFix::ProRes => {
                    let bytes = self.source.bytes(body, limit)?;
                    prores_frame(bytes)
                }
            };
            let times = self.frame_times(index, cluster, header, 1, None)?;
            let (presentation, duration) = times.first().copied().unwrap_or((None, None));
            return self
                .packet(index, bytes, presentation, duration, keyframe)
                .map(Some);
        }
        let data = self.source.bytes(body, limit)?;
        self.emit_laced(index, cluster, header, &data, None, keyframe)
    }

    fn block_group(&mut self, size: u64) -> Result<Option<Packet>> {
        let cluster = self.bump_block();
        let limit = self.block_limit().saturating_add(64 * 1024);
        let group = self.source.bytes(size, limit)?;
        let mut block = None;
        let mut duration = None;
        let mut referenced = false;
        for child in Children::new(&group) {
            let child = child?;
            match child.id {
                ids::BLOCK => block = Some(child.data),
                ids::BLOCK_DURATION => duration = Some(ebml::uint(child.id, child.data)?),
                ids::REFERENCE_BLOCK => referenced = true,
                _ => {}
            }
        }
        let Some(data) = block else {
            return Err(Error::malformed("BlockGroup has no Block"));
        };
        let header = block::parse_header(data)?;
        let Some(index) = self.track_index(header.track) else {
            return Ok(None);
        };
        let body = data.get(header.len..).unwrap_or(&[]);
        self.emit_laced(index, cluster, header, body, duration, !referenced)
    }

    /// The largest block body: one access unit, or a full lace of them.
    fn block_limit(&self) -> u64 {
        self.limits.max_packet_bytes
    }

    fn emit_laced(
        &mut self,
        index: usize,
        cluster: Option<Cluster>,
        header: BlockHeader,
        body: &[u8],
        block_duration: Option<u64>,
        keyframe: bool,
    ) -> Result<Option<Packet>> {
        let ranges = block::frames(body, header.lacing())?;
        let count = ranges.len();
        let times = self.frame_times(index, cluster, header, count, block_duration)?;
        let fix = self.tracks[index].fix.clone();
        let mut packets = Vec::with_capacity(count);
        for (range, (presentation, duration)) in ranges.into_iter().zip(times) {
            let frame = body.get(range).unwrap_or(&[]);
            let bytes = match &fix {
                FrameFix::None => frame.to_vec(),
                FrameFix::Prepend(prefix) => {
                    let mut bytes = Vec::with_capacity(prefix.len() + frame.len());
                    bytes.extend_from_slice(prefix);
                    bytes.extend_from_slice(frame);
                    bytes
                }
                FrameFix::ProRes => prores_frame(frame.to_vec()),
            };
            let limit = usize::try_from(self.limits.max_packet_bytes).unwrap_or(usize::MAX);
            if bytes.len() > limit {
                return Err(Error::Limit {
                    what: "frame bytes",
                    value: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
                    limit: self.limits.max_packet_bytes,
                });
            }
            packets.push(self.packet(index, bytes, presentation, duration, keyframe)?);
        }
        let mut packets = packets.into_iter();
        let first = packets.next();
        self.cursor.pending.extend(packets);
        Ok(first)
    }

    /// Presentation time and duration, in packet ticks, of each of `count`
    /// frames of one block.
    fn frame_times(
        &self,
        index: usize,
        cluster: Option<Cluster>,
        header: BlockHeader,
        count: usize,
        block_duration: Option<u64>,
    ) -> Result<Vec<(Option<i64>, Option<i64>)>> {
        let clock = self.clock;
        let start = match cluster.and_then(|cluster| cluster.timestamp) {
            Some(base) => {
                let segment = base
                    .checked_add(i64::from(header.relative))
                    .ok_or(Error::malformed("block timestamp overflows"))?;
                Some(
                    clock
                        .from_segment(segment)
                        .ok_or(Error::malformed("block timestamp overflows"))?,
                )
            }
            None => None,
        };
        let n = u128::try_from(count).unwrap_or(1).max(1);
        let mut out = Vec::with_capacity(count);
        if let Some(total) = block_duration {
            // BlockDuration covers the whole lace, split as FFmpeg splits it.
            let total = u128::from(total);
            for k in 0..n {
                let before = total * k / n;
                let after = total * (k + 1) / n;
                let offset = i64::try_from(before)
                    .ok()
                    .and_then(|value| clock.from_segment(value));
                let span = i64::try_from(after - before)
                    .ok()
                    .and_then(|value| clock.from_segment(value));
                let presentation = match (start, offset) {
                    (Some(start), Some(offset)) => start.checked_add(offset),
                    _ => None,
                };
                out.push((presentation, span));
            }
            return Ok(out);
        }
        match self.tracks[index].default_duration {
            Some(each) => {
                let each = u128::from(each);
                for k in 0..n {
                    let before = clock.from_ns(each * k);
                    let after = clock.from_ns(each * (k + 1));
                    let presentation = start.and_then(|start| start.checked_add(before));
                    out.push((presentation, Some(after - before)));
                }
            }
            None => {
                for k in 0..n {
                    out.push((if k == 0 { start } else { None }, None));
                }
            }
        }
        Ok(out)
    }

    fn packet(
        &self,
        index: usize,
        bytes: Vec<u8>,
        presentation: Option<i64>,
        duration: Option<i64>,
        keyframe: bool,
    ) -> Result<Packet> {
        let timestamp = Timestamp::try_new(None, presentation, duration.filter(|span| *span >= 0))
            .map_err(|error| Error::malformed(error.to_string()))?;
        let stream = u32::try_from(self.tracks[index].number)
            .map_err(|_| Error::malformed("track number does not fit"))?;
        Packet::try_new(stream, bytes, timestamp, self.clock.timescale(), keyframe)
            .map_err(|error| Error::malformed(error.to_string()))
    }
}

/// ProRes frames are stored without the frame size and `icpf`; put them back.
fn prores_frame(bytes: Vec<u8>) -> Vec<u8> {
    if bytes.get(4..8) == Some(b"icpf") {
        return bytes;
    }
    let total = u32::try_from(bytes.len().saturating_add(8)).unwrap_or(u32::MAX);
    let mut out = Vec::with_capacity(bytes.len() + 8);
    out.extend_from_slice(&total.to_be_bytes());
    out.extend_from_slice(b"icpf");
    out.extend_from_slice(&bytes);
    out
}

fn duration_ticks(info: &Info, clock: Clock) -> Option<i64> {
    let duration = info.duration?;
    let ticks = duration * clock.multiplier as f64;
    if !ticks.is_finite() || ticks < 0.0 || ticks >= i64::MAX as f64 {
        return None;
    }
    Some(ticks.round() as i64)
}

fn stream_for(
    entry: &TrackEntry,
    params: cutlass_codec::StreamParams,
    pre_skip: Option<u16>,
    info: &Info,
    clock: Clock,
) -> Result<Stream> {
    let id =
        u32::try_from(entry.number).map_err(|_| Error::malformed("track number does not fit"))?;
    let language = entry
        .language
        .as_deref()
        .and_then(|bytes| std::str::from_utf8(bytes).ok())
        .and_then(|text| Language::new(text).ok());
    let mut edits = Vec::new();
    if entry.codec_delay > 0 {
        let delay = clock.from_ns(u128::from(entry.codec_delay));
        let playable =
            duration_ticks(info, clock).map_or(0, |total| total.saturating_sub(delay).max(0));
        let edit = Edit::try_new(Some(delay), playable, 1, 1)
            .map_err(|error| Error::malformed(error.to_string()))?;
        edits.push(edit);
    }
    // Opus states its pre-skip; any other audio track's CodecDelay is its
    // encoder delay, counted in samples to the nearest.
    let delay = pre_skip.map(u64::from).or_else(|| {
        let rate = u128::from(params.sample_rate()?);
        let ns = u128::from(entry.codec_delay);
        (ns > 0).then(|| {
            u64::try_from((ns * rate + u128::from(SECOND_NS / 2)) / u128::from(SECOND_NS))
                .unwrap_or(u64::MAX)
        })
    });
    let trim = Trim::new(delay, None);
    Ok(Stream::new(
        TrackId::new(id),
        params,
        clock.timescale(),
        language,
        edits,
        trim,
    ))
}

fn read_ebml_header<R: Read + Seek>(source: &mut Source<R>) -> Result<EbmlHeader> {
    let Some(header) = source.header().map_err(not_matroska_on_bad_id)? else {
        return Err(Error::Source(cutlass_container::SourceError::Truncated {
            expected: 4,
            got: 0,
        }));
    };
    if header.id != ids::EBML {
        return Err(Error::NotMatroska);
    }
    let Size::Known(size) = header.size else {
        return Err(Error::Ebml(ebml::EbmlError::UnknownSize { id: ids::EBML }));
    };
    let data = source.bytes(size, MAX_EBML_HEADER)?;
    meta::parse_ebml_header(&data)
}

fn not_matroska_on_bad_id(error: Error) -> Error {
    match error {
        Error::Ebml(ebml::EbmlError::BadId { .. } | ebml::EbmlError::BadSize) => Error::NotMatroska,
        other => other,
    }
}

/// The Segment header, after any top-level Void elements.
fn find_segment<R: Read + Seek>(source: &mut Source<R>) -> Result<Header> {
    for _ in 0..MAX_TOP_LEVEL {
        let header = source
            .header()?
            .ok_or(Error::malformed("the file has no Segment"))?;
        if header.id == ids::SEGMENT {
            return Ok(header);
        }
        match header.size {
            Size::Known(size) => source.skip(size)?,
            Size::Unknown => {
                return Err(Error::Ebml(ebml::EbmlError::UnknownSize { id: header.id }));
            }
        }
    }
    Err(Error::Limit {
        what: "elements before the Segment",
        value: u64::from(MAX_TOP_LEVEL) + 1,
        limit: u64::from(MAX_TOP_LEVEL),
    })
}

#[derive(Default)]
struct TopLevel {
    info: Option<Info>,
    tracks: Option<Vec<TrackEntry>>,
    cues: Option<Vec<Cue>>,
    seek_head: Vec<SeekEntry>,
    first_cluster: Option<Header>,
}

/// Reads the Segment's children up to the first Cluster. The Cluster's
/// header is kept in `walk.first_cluster` and the input sits after it.
fn walk_to_first_cluster<R: Read + Seek>(
    source: &mut Source<R>,
    segment_end: Option<u64>,
    limits: Limits,
    walk: &mut TopLevel,
) -> Result<()> {
    for _ in 0..MAX_TOP_LEVEL {
        if segment_end.is_some_and(|end| source.position() >= end) {
            return Ok(());
        }
        let Some(header) = source.header()? else {
            return Ok(());
        };
        if let (Some(end), Some(segment_end)) = (header.end(), segment_end)
            && end > segment_end
        {
            return Err(Error::Ebml(ebml::EbmlError::PastParent {
                id: header.id,
                size: end - header.data,
                room: segment_end.saturating_sub(header.data),
            }));
        }
        if header.id == ids::CLUSTER {
            walk.first_cluster = Some(header);
            return Ok(());
        }
        if header.id == ids::EBML || header.id == ids::SEGMENT {
            return Ok(());
        }
        let Size::Known(size) = header.size else {
            return Err(Error::Ebml(ebml::EbmlError::UnknownSize { id: header.id }));
        };
        read_top_level(source, header.id, size, limits, walk)?;
    }
    Err(Error::Limit {
        what: "elements before the first Cluster",
        value: u64::from(MAX_TOP_LEVEL) + 1,
        limit: u64::from(MAX_TOP_LEVEL),
    })
}

/// Reads or skips one top-level element whose header was just read.
fn read_top_level<R: Read + Seek>(
    source: &mut Source<R>,
    id: u32,
    size: u64,
    limits: Limits,
    walk: &mut TopLevel,
) -> Result<()> {
    match id {
        ids::SEEK_HEAD if walk.seek_head.is_empty() => {
            let data = source.bytes(size, limits.max_header_bytes)?;
            walk.seek_head = meta::parse_seek_head(&data)?;
        }
        ids::INFO if walk.info.is_none() => {
            let data = source.bytes(size, limits.max_header_bytes)?;
            walk.info = Some(meta::parse_info(&data)?);
        }
        ids::TRACKS if walk.tracks.is_none() => {
            let data = source.bytes(size, limits.max_header_bytes)?;
            walk.tracks = Some(meta::parse_tracks(
                &data,
                limits.max_tracks,
                limits.max_extradata_bytes,
            )?);
        }
        ids::CUES if walk.cues.is_none() && source.random() => {
            let data = source.bytes(size, MAX_CUES_BYTES)?;
            walk.cues = Some(meta::parse_cues(&data, limits.max_index_entries)?);
        }
        _ => source.skip(size)?,
    }
    Ok(())
}

/// Reads Info and Tracks from where the SeekHead says, when they were not
/// before the first Cluster.
fn fetch_from_seek_head<R: Read + Seek>(
    source: &mut Source<R>,
    segment_data: u64,
    limits: Limits,
    walk: &mut TopLevel,
) -> Result<()> {
    if !source.random() {
        return Ok(());
    }
    let entries = walk.seek_head.clone();
    for wanted in [ids::INFO, ids::TRACKS] {
        let missing = match wanted {
            ids::INFO => walk.info.is_none(),
            _ => walk.tracks.is_none(),
        };
        if !missing {
            continue;
        }
        let Some(entry) = entries.iter().find(|entry| entry.id == wanted) else {
            continue;
        };
        let at = segment_data
            .checked_add(entry.position)
            .ok_or(Error::malformed("SeekPosition overflows"))?;
        source.seek(at)?;
        let header = source
            .header()?
            .ok_or(Error::malformed("SeekPosition is past the end"))?;
        if header.id != wanted {
            return Err(Error::malformed("SeekHead points at the wrong element"));
        }
        let Size::Known(size) = header.size else {
            return Err(Error::Ebml(ebml::EbmlError::UnknownSize { id: header.id }));
        };
        read_top_level(source, wanted, size, limits, walk)?;
    }
    Ok(())
}

impl<R: Read + Seek + Send> Demux for MatroskaDemux<R> {
    fn family(&self) -> ContainerId {
        ContainerId::Matroska
    }

    fn streams(&self) -> &[Stream] {
        &self.streams
    }

    fn gaps(&self) -> &[Gap] {
        &[]
    }

    fn read_packet(&mut self) -> std::result::Result<Option<Packet>, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        self.next_packet().map_err(Error::into_read)
    }

    fn seek(&mut self, to: SeekTo) -> std::result::Result<SeekLanding, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        self.seek_to(to).map_err(Error::into_read)
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }

    fn consumed(&self) -> u64 {
        self.source.consumed()
    }
}

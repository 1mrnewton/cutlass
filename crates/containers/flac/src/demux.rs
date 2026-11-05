//! Native `.flac` demux (RFC 9639 sections 6 to 8).
//!
//! Open reads the marker and the metadata blocks: STREAMINFO (first, once),
//! SEEKTABLE (at most once), and every other block skipped by its length
//! without being read. Frames have no length field, so a frame ends where
//! the next valid header starts and the CRC-16 of the bytes before it is
//! zero. The next header must also carry the next frame or sample number
//! and the stream's channel count, depth, and rate.

use std::io::{Read, Seek};

use cutlass_codec::{
    CodecId, ContainerId, Packet, StreamCodec, StreamParams, Timescale, Timestamp,
};
use cutlass_container::{
    Access, Demux, Gap, Input, Landing, Limits, OpenError, ReadError, SeekError, SeekLanding,
    SeekTo, SourceError, Stream, TrackId, Trim,
};

use crate::format::{
    FrameHeader, Header, MAX_HEADER_LEN, PLACEHOLDER, SEEK_POINT_LEN, STREAMINFO_LEN, SeekPoint,
    StreamInfo, crc16_byte, is_sync, parse_header,
};

/// Bytes read from the file at a time.
const CHUNK: usize = 64 * 1024;
/// Metadata blocks open walks before it gives up.
pub const MAX_METADATA_BLOCKS: u32 = 4096;
/// A bisection seek stops halving below this many bytes and walks frames.
const WALK_BYTES: u64 = 256 * 1024;

/// Where a frame ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Split {
    header: FrameHeader,
    end: u64,
    /// The CRC-16 of the frame matched.
    verified: bool,
}

/// A native FLAC file.
pub struct FlacDemux<R: Read + Seek> {
    input: Input<R>,
    streams: Vec<Stream>,
    info: StreamInfo,
    seek_points: Vec<SeekPoint>,
    first_frame: u64,
    data_end: u64,
    /// The first frame's blocking strategy, which every frame keeps.
    variable: bool,
    /// Samples per frame of a fixed-block-size stream.
    nominal: u64,
    max_packet: u64,
    buf: Vec<u8>,
    buf_at: u64,
    cursor: u64,
    cancelled: bool,
    damaged_tail: bool,
}

fn source(error: SourceError) -> ReadError {
    ReadError::from_packet(error)
}

fn malformed(message: &str) -> ReadError {
    ReadError::Source {
        message: format!("flac: {message}"),
    }
}

fn open_malformed(message: &str) -> OpenError {
    OpenError::Source {
        message: format!("flac: {message}"),
    }
}

impl<R: Read + Seek> FlacDemux<R> {
    /// Opens a file of `length` bytes with [`Limits::standard`].
    pub fn open(reader: R, length: u64) -> Result<Self, OpenError> {
        Self::open_limited(reader, length, Limits::standard())
    }

    pub fn open_limited(reader: R, length: u64, limits: Limits) -> Result<Self, OpenError> {
        let mut input = Input::random(reader, length, limits);
        let mut marker = [0u8; 4];
        input
            .read_exact(&mut marker)
            .map_err(OpenError::from_header)?;
        if &marker != b"fLaC" {
            if &marker[..3] == b"ID3" {
                return Err(OpenError::UnsupportedBrand {
                    family: ContainerId::Flac,
                });
            }
            return Err(OpenError::Unrecognized {
                family: ContainerId::Flac,
            });
        }
        let mut info = None;
        let mut seek_points = None;
        let mut blocks = 0u32;
        loop {
            blocks += 1;
            if blocks > MAX_METADATA_BLOCKS {
                return Err(open_malformed("too many metadata blocks"));
            }
            let mut head = [0u8; 4];
            input
                .read_exact(&mut head)
                .map_err(OpenError::from_header)?;
            let last = head[0] & 0x80 != 0;
            let kind = head[0] & 0x7F;
            let len = u64::from(u32::from_be_bytes([0, head[1], head[2], head[3]]));
            if kind == 127 {
                return Err(open_malformed("forbidden metadata block type"));
            }
            if info.is_none() && kind != 0 {
                return Err(open_malformed("streaminfo is not the first block"));
            }
            match kind {
                0 => {
                    if info.is_some() {
                        return Err(open_malformed("a second streaminfo"));
                    }
                    if len != STREAMINFO_LEN as u64 {
                        return Err(open_malformed("streaminfo length"));
                    }
                    let mut body = [0u8; STREAMINFO_LEN];
                    input
                        .read_exact(&mut body)
                        .map_err(OpenError::from_header)?;
                    info =
                        Some(StreamInfo::parse(&body).ok_or_else(|| open_malformed("streaminfo"))?);
                }
                3 => {
                    if seek_points.is_some() {
                        return Err(open_malformed("a second seektable"));
                    }
                    if len % SEEK_POINT_LEN as u64 != 0 {
                        return Err(open_malformed("seektable length"));
                    }
                    let limit = u64::from(limits.max_index_entries) * SEEK_POINT_LEN as u64;
                    let body = input
                        .read_limited(len, limit)
                        .map_err(OpenError::from_header)?;
                    seek_points = Some(read_seek_points(&body)?);
                }
                _ => {
                    let to = input.position().saturating_add(len);
                    if to > length {
                        return Err(OpenError::Truncated {
                            expected: to,
                            got: length,
                        });
                    }
                    input.seek(to).map_err(OpenError::from_header)?;
                }
            }
            if last {
                break;
            }
        }
        let info = info.ok_or_else(|| open_malformed("no streaminfo"))?;
        let first_frame = input.position();
        let data_end = data_end(&mut input, first_frame, length)?;
        let params = StreamParams::try_audio(
            StreamCodec::Known(CodecId::Flac),
            info.sample_rate,
            u16::from(info.channels),
            info.to_bytes().to_vec(),
        )
        .map_err(|_| open_malformed("audio parameters"))?;
        let stream = Stream::new(
            TrackId::new(0),
            params,
            Timescale::new(info.sample_rate),
            None,
            Vec::new(),
            Trim::default(),
        );
        let mut demux = Self {
            input,
            streams: vec![stream],
            info,
            seek_points: seek_points.unwrap_or_default(),
            first_frame,
            data_end,
            variable: false,
            nominal: u64::from(info.max_block),
            max_packet: limits.max_packet_bytes,
            buf: Vec::new(),
            buf_at: first_frame,
            cursor: first_frame,
            cancelled: false,
            damaged_tail: false,
        };
        if first_frame < data_end {
            let header = demux
                .header_at(first_frame)
                .map_err(|_| open_malformed("no frame after the metadata"))?
                .ok_or_else(|| open_malformed("no frame after the metadata"))?;
            demux.variable = header.variable;
            if info.min_block != info.max_block && !header.variable {
                demux.nominal = u64::from(header.block_size);
            }
        }
        Ok(demux)
    }

    /// The STREAMINFO block.
    #[must_use]
    pub fn stream_info(&self) -> &StreamInfo {
        &self.info
    }

    /// The SEEKTABLE points, placeholders removed.
    #[must_use]
    pub fn seek_points(&self) -> &[SeekPoint] {
        &self.seek_points
    }

    /// Byte offset of the first frame.
    #[must_use]
    pub fn first_frame(&self) -> u64 {
        self.first_frame
    }

    /// `true` once a read reached a last frame whose CRC-16 failed: the file
    /// was cut short, and that frame was not returned.
    #[must_use]
    pub fn damaged_tail(&self) -> bool {
        self.damaged_tail
    }

    /// Makes `[from, from + need)` buffered, or as much of it as the file
    /// holds. Returns the bytes available from `from`.
    fn fill(&mut self, from: u64, need: usize) -> Result<usize, ReadError> {
        let held_end = self.buf_at + self.buf.len() as u64;
        if from < self.buf_at || from > held_end {
            self.input.seek(from).map_err(source)?;
            self.buf.clear();
            self.buf_at = from;
        } else if from > self.buf_at {
            let drop = usize::try_from(from - self.buf_at).unwrap_or(usize::MAX);
            self.buf.drain(..drop.min(self.buf.len()));
            self.buf_at = from;
        }
        while self.buf.len() < need {
            let at = self.buf_at + self.buf.len() as u64;
            let left = self.data_end.saturating_sub(at);
            if left == 0 {
                break;
            }
            let want = (need - self.buf.len()).max(CHUNK);
            let want = usize::try_from(left).map_or(want, |left| want.min(left));
            let start = self.buf.len();
            self.buf.resize(start + want, 0);
            let got = self.input.read(&mut self.buf[start..]).map_err(source)?;
            self.buf.truncate(start + got);
            if got == 0 {
                return Err(ReadError::Truncated {
                    expected: self.data_end,
                    got: at,
                });
            }
        }
        Ok(self.buf.len().min(need))
    }

    /// The header at `at` when one is there and it fits this stream.
    fn header_at(&mut self, at: u64) -> Result<Option<FrameHeader>, ReadError> {
        let have = self.fill(at, MAX_HEADER_LEN)?;
        match parse_header(&self.buf[..have]) {
            Header::Valid(header) if self.fits(&header) => Ok(Some(header)),
            _ => Ok(None),
        }
    }

    fn fits(&self, header: &FrameHeader) -> bool {
        header.channels == self.info.channels
            && header.bits.is_none_or(|bits| bits == self.info.bits)
            && header
                .sample_rate
                .is_none_or(|rate| rate == self.info.sample_rate)
            && header.block_size <= u32::from(self.info.max_block)
    }

    /// The number the frame after `header` carries.
    fn next_number(header: &FrameHeader) -> u64 {
        if header.variable {
            header.number.saturating_add(u64::from(header.block_size))
        } else {
            header.number.saturating_add(1)
        }
    }

    /// Finds the end of the frame at `start`, reading at most the packet
    /// limit.
    fn split(&mut self, start: u64) -> Result<Option<Split>, ReadError> {
        self.split_within(start, self.max_packet)
    }

    /// Finds the end of the frame at `start`, reading at most `bound` bytes.
    fn split_within(&mut self, start: u64, bound: u64) -> Result<Option<Split>, ReadError> {
        let Some(header) = self.header_at(start)? else {
            return Ok(None);
        };
        if header.variable != self.variable {
            return Ok(None);
        }
        let next = Self::next_number(&header);
        let mut crc = 0u16;
        let mut fallback = None;
        // A frame holds its header, at least one subframe byte, and the CRC.
        let mut q = 0usize;
        let header_len = header.len;
        loop {
            let need = q + MAX_HEADER_LEN + CHUNK;
            let have = self.fill(start, need)?;
            let limit = have.saturating_sub(1);
            while q < limit {
                let byte = self.buf[q];
                if q >= header_len + 3
                    && is_sync(byte, self.buf[q + 1])
                    && (crc == 0 || fallback.is_none())
                {
                    if have - q < MAX_HEADER_LEN && (self.buf_at + have as u64) < self.data_end {
                        break;
                    }
                    if let Header::Valid(candidate) = parse_header(&self.buf[q..have])
                        && candidate.variable == header.variable
                        && candidate.number == next
                        && self.fits(&candidate)
                    {
                        if crc == 0 {
                            return Ok(Some(Split {
                                header,
                                end: start + q as u64,
                                verified: true,
                            }));
                        }
                        fallback = Some(q);
                    }
                }
                crc = crc16_byte(crc, byte);
                q += 1;
                if q as u64 > bound {
                    return match fallback {
                        Some(end) => Ok(Some(Split {
                            header,
                            end: start + end as u64,
                            verified: false,
                        })),
                        None => Err(ReadError::HostileSize {
                            declared: q as u64,
                            limit: bound,
                        }),
                    };
                }
            }
            let at_end = start + have as u64 >= self.data_end;
            if at_end && have - q <= 1 {
                // The last byte, then the end of the data.
                if have > q {
                    crc = crc16_byte(crc, self.buf[q]);
                }
                let end = start + have as u64;
                if crc == 0 && have > header_len + 2 {
                    return Ok(Some(Split {
                        header,
                        end,
                        verified: true,
                    }));
                }
                return Ok(fallback.map(|at| Split {
                    header,
                    end: start + at as u64,
                    verified: false,
                }));
            }
        }
    }

    fn pts(&self, header: &FrameHeader) -> u64 {
        if header.variable {
            header.number
        } else {
            header.number.saturating_mul(self.nominal)
        }
    }

    fn read_next(&mut self) -> Result<Option<Packet>, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        if self.cursor >= self.data_end {
            return Ok(None);
        }
        let Some(split) = self.split(self.cursor)? else {
            if self.header_at(self.cursor)?.is_some() {
                // A valid header whose frame never closes: a cut file.
                self.damaged_tail = true;
                self.cursor = self.data_end;
                return Ok(None);
            }
            return Err(malformed("no frame header where a frame starts"));
        };
        let len = usize::try_from(split.end - self.cursor).map_err(|_| ReadError::HostileSize {
            declared: split.end - self.cursor,
            limit: self.max_packet,
        })?;
        self.fill(self.cursor, len)?;
        let data = self.buf[..len].to_vec();
        let pts = i64::try_from(self.pts(&split.header))
            .map_err(|_| malformed("timestamp does not fit"))?;
        self.cursor = split.end;
        let timestamp = Timestamp::try_new(
            Some(pts),
            Some(pts),
            Some(i64::from(split.header.block_size)),
        )
        .map_err(|_| malformed("timestamp"))?;
        Packet::try_new(
            0,
            data,
            timestamp,
            Timescale::new(self.info.sample_rate),
            true,
        )
        .map(Some)
        .map_err(|_| malformed("packet"))
    }

    /// The first confirmed frame at or after `from` and before `before`.
    fn hunt(&mut self, from: u64, before: u64) -> Result<Option<(u64, Split)>, ReadError> {
        let mut at = from;
        while at < before.min(self.data_end) {
            let have = self.fill(at, CHUNK)?;
            if have < 2 {
                return Ok(None);
            }
            let window = &self.buf[..have];
            let Some(found) = window.windows(2).position(|pair| is_sync(pair[0], pair[1])) else {
                at += have as u64 - 1;
                continue;
            };
            let candidate = at + found as u64;
            if candidate >= before {
                return Ok(None);
            }
            // A false sync is bounded by the largest frame STREAMINFO names.
            let bound = if self.info.max_frame > 0 {
                u64::from(self.info.max_frame) + MAX_HEADER_LEN as u64
            } else {
                self.max_packet
            };
            match self.split_within(candidate, bound) {
                Ok(Some(split)) if split.verified => return Ok(Some((candidate, split))),
                Ok(_) | Err(ReadError::HostileSize { .. }) => {}
                Err(error) => return Err(error),
            }
            at = candidate + 1;
        }
        Ok(None)
    }

    fn seek_to(&mut self, to: SeekTo) -> Result<SeekLanding, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        if to.track != 0 {
            return Err(ReadError::Seek(SeekError::UnknownTrack));
        }
        if self.input.access() == Access::Forward {
            return Err(ReadError::Seek(SeekError::ForwardOnly));
        }
        if self.first_frame >= self.data_end {
            return Err(ReadError::Seek(SeekError::Empty));
        }
        let target = u64::try_from(to.time.max(0)).unwrap_or(0);
        let total = self.info.total_samples;
        if to.end.is_some_and(|end| to.time >= end) || (total > 0 && target >= total) {
            return Ok(self.land_end(false, 0));
        }
        let mut scans = 0u32;
        let index_used = !self.seek_points.is_empty();
        // A start at or before the target.
        let mut start = self.first_frame;
        if let Some(point) = self
            .seek_points
            .iter()
            .rev()
            .find(|point| point.sample <= target)
        {
            let at = self.first_frame.saturating_add(point.offset);
            if at < self.data_end && self.header_at(at)?.is_some() {
                start = at;
            }
        } else if self.seek_points.is_empty() {
            let (mut low, mut high) = (self.first_frame, self.data_end);
            while high - low > WALK_BYTES {
                let middle = low + (high - low) / 2;
                scans += 1;
                match self.hunt(middle, high)? {
                    Some((at, split)) if self.pts(&split.header) <= target => low = at,
                    _ => high = middle,
                }
            }
            start = low;
        }
        // Walk frames to the one holding the target.
        let mut at = start;
        loop {
            if self.cancelled {
                return Err(ReadError::Cancelled);
            }
            if at >= self.data_end {
                return Ok(self.land_end(index_used, scans));
            }
            scans = scans.saturating_add(1);
            let Some(split) = self.split(at)? else {
                return Err(malformed("no frame header on the seek path"));
            };
            let pts = self.pts(&split.header);
            if pts.saturating_add(u64::from(split.header.block_size)) > target {
                self.cursor = at;
                let ordinal = if split.header.variable {
                    pts / u64::from(self.info.max_block.max(1))
                } else {
                    split.header.number
                };
                return Ok(SeekLanding {
                    landing: Landing::Packet {
                        ordinal: u32::try_from(ordinal).unwrap_or(u32::MAX),
                        time: i64::try_from(pts).map_err(|_| malformed("timestamp"))?,
                        keyframe: true,
                        preroll: 0,
                        byte_position: at,
                    },
                    index_used,
                    payload_scans: scans,
                    entries_visited: if index_used {
                        u32::try_from(self.seek_points.len()).unwrap_or(u32::MAX)
                    } else {
                        0
                    },
                });
            }
            at = split.end;
        }
    }

    fn land_end(&mut self, index_used: bool, scans: u32) -> SeekLanding {
        self.cursor = self.data_end;
        SeekLanding {
            landing: Landing::End,
            index_used,
            payload_scans: scans,
            entries_visited: 0,
        }
    }
}

fn read_seek_points(body: &[u8]) -> Result<Vec<SeekPoint>, OpenError> {
    let (rows, _) = body.as_chunks::<SEEK_POINT_LEN>();
    let mut points = Vec::with_capacity(rows.len());
    let mut placeholders = false;
    for row in rows {
        let point = SeekPoint::parse(row);
        if point.sample == PLACEHOLDER {
            placeholders = true;
            continue;
        }
        if placeholders {
            return Err(open_malformed("seek point after a placeholder"));
        }
        if points.last().is_some_and(|last: &SeekPoint| {
            last.sample >= point.sample || last.offset > point.offset
        }) {
            return Err(open_malformed("seek points out of order"));
        }
        points.push(point);
    }
    Ok(points)
}

/// Where frames stop: the file end, or the start of an ID3v1 tag.
fn data_end<R: Read + Seek>(
    input: &mut Input<R>,
    first_frame: u64,
    length: u64,
) -> Result<u64, OpenError> {
    if length >= first_frame + 128 {
        let at = length - 128;
        input.seek(at).map_err(OpenError::from_header)?;
        let mut tag = [0u8; 3];
        input.read_exact(&mut tag).map_err(OpenError::from_header)?;
        input.seek(first_frame).map_err(OpenError::from_header)?;
        if &tag == b"TAG" {
            return Ok(at);
        }
    }
    Ok(length.max(first_frame))
}

impl<R: Read + Seek + Send> Demux for FlacDemux<R> {
    fn family(&self) -> ContainerId {
        ContainerId::Flac
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
        let saved = self.cursor;
        let result = self.seek_to(to);
        if result.is_err() {
            self.cursor = saved;
        }
        result
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }

    fn consumed(&self) -> u64 {
        self.input.consumed()
    }
}

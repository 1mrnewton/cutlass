//! Native `.flac` mux.
//!
//! Frames stream straight to the writer. The header is written before the
//! first frame: the marker, STREAMINFO, and a SEEKTABLE of placeholder
//! points, all with the sample count unknown (zero). [`Mux::finish`] seeks
//! back and fills in STREAMINFO (sample count, frame and block sizes, and
//! the MD5 signature an encoder handed over) and the seek points. A
//! cancelled or failed mux leaves the count at zero, which a reader takes
//! as "unknown", so the frames already written still play.

use std::io::{Seek, SeekFrom, Write};

use cutlass_codec::{CodecId, ContainerId, Packet, StreamCodec};
use cutlass_container::{
    FailedMux, FinishedMux, IndexPlace, Mux, MuxError, MuxMode, MuxStatus, NewStream,
    TrackDuration, TrackId,
};

use crate::format::{
    FrameHeader, Header, SEEK_POINT_LEN, STREAMINFO_LEN, SeekPoint, StreamInfo, crc16, parse_header,
};

/// Seek points reserved when none are asked for.
pub const DEFAULT_SEEK_POINTS: u32 = 128;
/// Seconds between seek points, the reference encoder's default.
pub const SEEK_SPACING_SECONDS: u64 = 10;
/// Most frames a mux records for its seek table.
const MAX_RECORDED_FRAMES: usize = 16 * 1024 * 1024;

struct Track {
    rate: u32,
    channels: u8,
    config: Option<StreamInfo>,
}

/// One frame start the seek table can point at.
#[derive(Clone, Copy)]
struct FrameStart {
    sample: u64,
    offset: u64,
    samples: u16,
}

/// Writes one FLAC stream into a native file.
pub struct FlacMux<W: Write + Seek> {
    out: W,
    status: MuxStatus,
    seek_slots: u32,
    track: Option<Track>,
    bits: Option<u8>,
    variable: Option<bool>,
    next_number: u64,
    header_end: Option<u64>,
    seektable_at: u64,
    written: u64,
    frames: Vec<FrameStart>,
    samples: u64,
    min_frame: u32,
    max_frame: u32,
    /// The first frame's block size: a fixed-block-size stream's nominal.
    first_block: u32,
    /// The smallest block before the last one.
    min_not_last: u32,
    max_block: u32,
    last_block: u32,
    frame_count: u64,
}

fn rejected(message: &str) -> MuxError {
    MuxError::Rejected {
        message: format!("flac: {message}"),
    }
}

fn io(error: &std::io::Error) -> MuxError {
    MuxError::Rejected {
        message: format!("flac write failed: {error}"),
    }
}

impl<W: Write + Seek> FlacMux<W> {
    /// A muxer reserving [`DEFAULT_SEEK_POINTS`] seek points.
    pub fn new(out: W) -> Self {
        Self::with_seek_points(out, DEFAULT_SEEK_POINTS)
    }

    /// A muxer reserving `slots` seek points; zero writes no SEEKTABLE.
    pub fn with_seek_points(out: W, slots: u32) -> Self {
        Self {
            out,
            status: MuxStatus::Building,
            seek_slots: slots.min(65_535),
            track: None,
            bits: None,
            variable: None,
            next_number: 0,
            header_end: None,
            seektable_at: 0,
            written: 0,
            frames: Vec::new(),
            samples: 0,
            min_frame: 0,
            max_frame: 0,
            first_block: 0,
            min_not_last: u32::MAX,
            max_block: 0,
            last_block: 0,
            frame_count: 0,
        }
    }

    /// The writer, after finish or cancel.
    pub fn into_inner(self) -> W {
        self.out
    }

    fn fail(&mut self, error: MuxError) -> MuxError {
        if self.status == MuxStatus::Building {
            self.status = MuxStatus::Failed;
        }
        error
    }

    fn building(&self) -> Result<(), MuxError> {
        match self.status {
            MuxStatus::Building => Ok(()),
            MuxStatus::Finished => Err(MuxError::AlreadyFinished),
            MuxStatus::Failed => Err(MuxError::NotBuilding),
        }
    }

    /// The STREAMINFO to write: the latest one an encoder handed over, or
    /// one built from the stream and the frames seen.
    fn stream_info(&self, finished: bool) -> Result<StreamInfo, MuxError> {
        let track = self.track.as_ref().ok_or_else(|| rejected("no stream"))?;
        let bits = track
            .config
            .map(|info| info.bits)
            .or(self.bits)
            .ok_or_else(|| rejected("the bit depth is in neither a config nor a frame header"))?;
        let mut info = track.config.unwrap_or(StreamInfo {
            min_block: 0,
            max_block: 0,
            min_frame: 0,
            max_frame: 0,
            sample_rate: track.rate,
            channels: track.channels,
            bits,
            total_samples: 0,
            md5: [0; 16],
        });
        // Block sizes cover what was written. The last block is not a
        // minimum, and a fixed-block-size stream names its nominal size.
        let to16 = |value: u32| u16::try_from(value).unwrap_or(u16::MAX).max(16);
        if self.frame_count > 1 && self.variable == Some(false) {
            info.min_block = to16(self.first_block);
            info.max_block = info.min_block;
        } else if self.frame_count > 0 {
            info.max_block = info.max_block.max(to16(self.max_block));
            if self.frame_count > 1 {
                info.min_block = to16(self.min_not_last).min(info.max_block);
            } else if track.config.is_none() {
                info.min_block = info.max_block;
            }
            info.min_block = info.min_block.clamp(16, info.max_block);
        } else if info.max_block == 0 {
            info.min_block = 4096;
            info.max_block = 4096;
        }
        if finished {
            if info.total_samples != 0 && info.total_samples != self.samples {
                // A signature over other samples than these is not this
                // file's signature.
                info.md5 = [0; 16];
            }
            info.total_samples = if self.samples < (1 << 36) {
                self.samples
            } else {
                0
            };
            info.min_frame = self.min_frame;
            info.max_frame = self.max_frame;
        } else {
            info.total_samples = 0;
            info.md5 = [0; 16];
            info.min_frame = 0;
            info.max_frame = 0;
        }
        Ok(info)
    }

    fn write_header(&mut self) -> Result<(), MuxError> {
        let info = self.stream_info(false)?;
        let mut head = b"fLaC".to_vec();
        let last = if self.seek_slots == 0 { 0x80 } else { 0 };
        head.extend_from_slice(&[last, 0, 0, STREAMINFO_LEN as u8]);
        head.extend_from_slice(&info.to_bytes());
        let seektable_at = head.len() as u64;
        if self.seek_slots > 0 {
            let len = self.seek_slots as usize * SEEK_POINT_LEN;
            let len_bytes = u32::try_from(len).unwrap_or(0).to_be_bytes();
            head.extend_from_slice(&[0x80 | 3, len_bytes[1], len_bytes[2], len_bytes[3]]);
            for _ in 0..self.seek_slots {
                head.extend_from_slice(&SeekPoint::placeholder().to_bytes());
            }
        }
        self.out.write_all(&head).map_err(|error| io(&error))?;
        self.seektable_at = seektable_at;
        self.header_end = Some(head.len() as u64);
        Ok(())
    }

    /// Checks `packet` is the next frame of this stream.
    fn check(&self, packet: &Packet) -> Result<FrameHeader, MuxError> {
        let track = self
            .track
            .as_ref()
            .ok_or_else(|| rejected("add a stream before a packet"))?;
        if packet.stream() != 0 {
            return Err(MuxError::UnknownTrack {
                id: packet.stream(),
            });
        }
        if packet.timescale().ticks_per_second() != track.rate {
            return Err(MuxError::TimescaleMismatch {
                packet: packet.timescale().ticks_per_second(),
                stream: track.rate,
            });
        }
        let bytes = packet.bytes();
        let Header::Valid(header) = parse_header(bytes) else {
            return Err(rejected("the packet does not start with a frame header"));
        };
        if bytes.len() < header.len + 3 || crc16(bytes) != 0 {
            return Err(rejected("the frame crc does not match"));
        }
        if header.channels != track.channels
            || header.sample_rate.is_some_and(|rate| rate != track.rate)
        {
            return Err(rejected("the frame does not match the stream"));
        }
        let bits = header
            .bits
            .or(track.config.map(|info| info.bits))
            .or(self.bits);
        if bits.is_none() || (self.bits.is_some() && bits != self.bits) {
            return Err(rejected("the frame bit depth changes or is unknown"));
        }
        if self
            .variable
            .is_some_and(|variable| variable != header.variable)
        {
            return Err(rejected("the blocking strategy changes"));
        }
        let expected = if self.variable.is_none() {
            0
        } else {
            self.next_number
        };
        if header.number != expected {
            return Err(rejected("the frame or sample number is not the next one"));
        }
        if self.variable == Some(false)
            && self.frame_count > 0
            && self.last_block != self.first_block
        {
            return Err(rejected("a short frame is followed by another frame"));
        }
        let timestamp = packet.timestamp();
        let duration = timestamp.duration().ok_or(MuxError::MissingTimestamp)?;
        if duration != i64::from(header.block_size) {
            return Err(MuxError::DurationMismatch { track: 0 });
        }
        let pts = timestamp
            .presentation_or_decode()
            .ok_or(MuxError::MissingTimestamp)?;
        if u64::try_from(pts).ok() != Some(self.samples) {
            return Err(rejected("the frame does not follow the previous one"));
        }
        Ok(header)
    }

    fn write_frame(&mut self, packet: &Packet) -> Result<(), MuxError> {
        let header = self.check(packet)?;
        if self.header_end.is_none() {
            self.bits = header
                .bits
                .or(self.track.as_ref().and_then(|t| t.config.map(|i| i.bits)));
            self.variable = Some(header.variable);
            self.write_header()?;
        }
        let bytes = packet.bytes();
        self.out.write_all(bytes).map_err(|error| io(&error))?;
        if self.frames.len() < MAX_RECORDED_FRAMES {
            self.frames.push(FrameStart {
                sample: self.samples,
                offset: self.written,
                samples: u16::try_from(header.block_size).unwrap_or(u16::MAX),
            });
        }
        let size = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
        self.min_frame = if self.frame_count == 0 {
            size
        } else {
            self.min_frame.min(size)
        };
        self.max_frame = self.max_frame.max(size);
        if self.frame_count == 0 {
            self.first_block = header.block_size;
        } else {
            self.min_not_last = self.min_not_last.min(self.last_block);
        }
        self.max_block = self.max_block.max(header.block_size);
        self.last_block = header.block_size;
        self.written += bytes.len() as u64;
        self.samples += u64::from(header.block_size);
        self.frame_count += 1;
        self.next_number = if header.variable {
            self.samples
        } else {
            self.frame_count
        };
        Ok(())
    }

    /// Seek points about every ten seconds, or spread evenly when that
    /// would need more than the reserved slots.
    fn seek_points(&self) -> Vec<SeekPoint> {
        let slots = self.seek_slots as usize;
        let Some(track) = &self.track else {
            return Vec::new();
        };
        if slots == 0 || self.frames.is_empty() {
            return Vec::new();
        }
        let spacing = u64::from(track.rate) * SEEK_SPACING_SECONDS;
        let wanted = self.samples.div_ceil(spacing.max(1)).max(1);
        let count = usize::try_from(wanted).unwrap_or(slots).min(slots);
        let step = self.samples.div_ceil(count as u64).max(1);
        let mut points = Vec::with_capacity(count);
        let mut target = 0u64;
        let mut frames = self.frames.iter().peekable();
        while points.len() < count {
            // The last frame starting at or before the target.
            let mut chosen = None;
            while let Some(frame) = frames.peek() {
                if frame.sample > target {
                    break;
                }
                chosen = Some(**frame);
                frames.next();
            }
            if let Some(frame) = chosen
                && points
                    .last()
                    .is_none_or(|last: &SeekPoint| last.sample < frame.sample)
            {
                points.push(SeekPoint {
                    sample: frame.sample,
                    offset: frame.offset,
                    samples: frame.samples,
                });
            }
            if frames.peek().is_none() {
                break;
            }
            target = target.saturating_add(step);
        }
        points
    }

    fn complete(&mut self, durations: &[TrackDuration]) -> Result<FinishedMux, MuxError> {
        let Some(track) = &self.track else {
            return Err(rejected("no stream"));
        };
        let total = i64::try_from(self.samples).map_err(|_| rejected("too many samples"))?;
        let rate = track.rate;
        if durations.len() != 1
            || durations[0].track != TrackId::new(0)
            || durations[0].timescale.ticks_per_second() != rate
        {
            return Err(MuxError::Incomplete);
        }
        if durations[0].ticks != total {
            return Err(MuxError::DurationMismatch { track: 0 });
        }
        if self.header_end.is_none() {
            // No frames: a header with nothing after it.
            if self.bits.is_none() {
                self.bits = track.config.map(|info| info.bits);
            }
            self.write_header()?;
        }
        let info = self.stream_info(true)?;
        let points = self.seek_points();
        let end = self.out.stream_position().map_err(|error| io(&error))?;
        self.out
            .seek(SeekFrom::Start(8))
            .map_err(|error| io(&error))?;
        self.out
            .write_all(&info.to_bytes())
            .map_err(|error| io(&error))?;
        if self.seek_slots > 0 {
            let mut table = Vec::with_capacity(self.seek_slots as usize * SEEK_POINT_LEN);
            for point in &points {
                table.extend_from_slice(&point.to_bytes());
            }
            for _ in points.len()..self.seek_slots as usize {
                table.extend_from_slice(&SeekPoint::placeholder().to_bytes());
            }
            self.out
                .seek(SeekFrom::Start(self.seektable_at + 4))
                .map_err(|error| io(&error))?;
            self.out.write_all(&table).map_err(|error| io(&error))?;
        }
        self.out
            .seek(SeekFrom::Start(end))
            .map_err(|error| io(&error))?;
        self.out.flush().map_err(|error| io(&error))?;
        Ok(FinishedMux::complete(
            durations.to_vec(),
            IndexPlace::Start,
            MuxMode::Progressive,
            true,
            Vec::new(),
            self.written,
        ))
    }
}

impl<W: Write + Seek + Send> Mux for FlacMux<W> {
    fn family(&self) -> ContainerId {
        ContainerId::Flac
    }

    fn add_stream(&mut self, stream: NewStream) -> Result<TrackId, MuxError> {
        self.building()?;
        if self.track.is_some() {
            return Err(self.fail(rejected("a file holds one stream")));
        }
        match stream.params.codec() {
            StreamCodec::Known(CodecId::Flac) => {}
            StreamCodec::Unknown(unknown) => {
                return Err(self.fail(MuxError::UnknownCodec(unknown.clone())));
            }
            StreamCodec::Known(_) => return Err(self.fail(rejected("the codec is not flac"))),
        }
        let (Some(rate), Some(channels)) = (stream.params.sample_rate(), stream.params.channels())
        else {
            return Err(self.fail(rejected("an audio stream needs a rate and channels")));
        };
        let channels = u8::try_from(channels)
            .ok()
            .filter(|count| (1..=8).contains(count));
        let Some(channels) = channels else {
            return Err(self.fail(rejected("one to eight channels")));
        };
        if rate == 0 || rate >= 1 << 20 {
            return Err(self.fail(rejected("sample rate")));
        }
        if stream.timescale.ticks_per_second() != rate {
            return Err(self.fail(MuxError::TimescaleMismatch {
                packet: stream.timescale.ticks_per_second(),
                stream: rate,
            }));
        }
        let extra = stream.params.extradata();
        let config = if extra.is_empty() {
            None
        } else {
            match StreamInfo::from_extra(extra) {
                Some(info) if info.sample_rate == rate && info.channels == channels => Some(info),
                _ => {
                    return Err(self.fail(rejected("the extradata is not this stream's STREAMINFO")));
                }
            }
        };
        self.track = Some(Track {
            rate,
            channels,
            config,
        });
        Ok(TrackId::new(0))
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<(), MuxError> {
        self.building()?;
        self.write_frame(packet).map_err(|error| self.fail(error))
    }

    /// One stream in decode order: there is nothing to interleave.
    fn interleave(&mut self) -> Result<(), MuxError> {
        self.building()
    }

    fn cancel(&mut self) {
        if self.status == MuxStatus::Building {
            self.status = MuxStatus::Failed;
        }
        let _ = self.out.flush();
    }

    fn finish(&mut self, durations: &[TrackDuration]) -> Result<FinishedMux, FailedMux> {
        if let Err(error) = self.building() {
            return Err(FailedMux::incomplete(self.written, error));
        }
        match self.complete(durations) {
            Ok(done) => {
                self.status = MuxStatus::Finished;
                Ok(done)
            }
            Err(error) => {
                self.status = MuxStatus::Failed;
                Err(FailedMux::incomplete(self.written, error))
            }
        }
    }

    fn status(&self) -> MuxStatus {
        self.status
    }

    /// Takes a STREAMINFO an encoder handed over. The latest one is written
    /// at finish; a different rate, channel count, or depth is refused.
    fn set_config(&mut self, track: TrackId, config: &[u8]) -> Result<(), MuxError> {
        self.building()?;
        if track != TrackId::new(0) {
            return Err(MuxError::UnknownTrack { id: track.get() });
        }
        let Some(info) = StreamInfo::from_extra(config) else {
            return Err(rejected("the config is not a STREAMINFO"));
        };
        let Some(current) = self.track.as_mut() else {
            return Err(rejected("add a stream before a config"));
        };
        if info.sample_rate != current.rate
            || info.channels != current.channels
            || current.config.is_some_and(|known| known.bits != info.bits)
            || self.bits.is_some_and(|bits| bits != info.bits)
        {
            return Err(rejected("the config does not match the stream"));
        }
        current.config = Some(info);
        Ok(())
    }
}

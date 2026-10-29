//! WAV demuxer.
//!
//! One access unit is one sample frame for PCM, A-law, and μ-law, one
//! MPEG frame for tags `0x0050` and `0x0055`, and one `nBlockAlign` block
//! otherwise. Timestamps are sample-rate ticks. Sample bytes are not decoded.

use std::io::{Read, Seek};

use cutlass_codec::{ContainerId, Packet, PacketError, Timescale, Timestamp};
use cutlass_container::{
    Access, Demux, Gap, Input, Landing, Limits, OpenError, ReadError, SeekError, SeekLanding,
    SeekTo, Stream, TrackId, Trim,
};

use crate::format::{Framing, WavFormat};
use crate::limits::MAX_MPEG_RESYNC;
use crate::mpeg::{FrameHead, find_frame, parse_header};
use crate::walk::{Brand, DroppedChunk, KeptChunk, Scanned, scan};

const DISAGREE_DATA: &str = "ds64 data size disagrees with the data chunk";
const DISAGREE_SAMPLES: &str = "ds64 sample count disagrees with the data chunk";
const DISAGREE_FACT: &str = "fact sample count disagrees with ds64";

enum Layout {
    Constant {
        frame_bytes: u64,
        frames: u64,
        per_frame: u64,
        extra: u64,
    },
    Mpeg,
}

/// An open WAV or RF64 file.
pub struct WavDemux<R: Read + Seek> {
    input: Input<R>,
    streams: Vec<Stream>,
    format: WavFormat,
    brand: Brand,
    kept: Vec<KeptChunk>,
    dropped: Vec<DroppedChunk>,
    data_start: u64,
    data_len: u64,
    layout: Layout,
    next: u64,
    cursor: u64,
    samples_at: u64,
    cancelled: bool,
}

impl<R: Read + Seek> WavDemux<R> {
    /// Opens a random-access WAV. The length is the file size, not a buffer allocation.
    pub fn open(reader: R, length: u64) -> Result<Self, OpenError> {
        Self::open_limited(reader, length, Limits::standard())
    }

    pub fn open_limited(reader: R, length: u64, limits: Limits) -> Result<Self, OpenError> {
        let input = Input::random(reader, length, limits);
        Self::from_input(input, true)
    }

    /// Opens a forward reader. Seek is refused. Chunks after `data` are not read.
    pub fn open_forward(reader: R) -> Result<Self, OpenError> {
        let input = Input::forward(reader, Limits::standard());
        Self::from_input(input, false)
    }

    #[must_use]
    pub const fn format(&self) -> &WavFormat {
        &self.format
    }

    #[must_use]
    pub const fn brand(&self) -> Brand {
        self.brand
    }

    #[must_use]
    pub fn kept(&self) -> &[KeptChunk] {
        &self.kept
    }

    #[must_use]
    pub fn dropped(&self) -> &[DroppedChunk] {
        &self.dropped
    }

    /// Sample count when every packet has a duration. MPEG stays unknown until read.
    #[must_use]
    pub const fn duration_samples(&self) -> Option<u64> {
        match self.layout {
            Layout::Constant {
                frames,
                per_frame,
                extra,
                ..
            } => {
                if per_frame == 0 {
                    None
                } else {
                    Some(frames * per_frame + extra)
                }
            }
            Layout::Mpeg => None,
        }
    }

    fn from_input(mut input: Input<R>, random: bool) -> Result<Self, OpenError> {
        let packet_limit = input.limits().max_packet_bytes;
        let scanned = scan(&mut input, random)?;
        let planned = plan(scanned, packet_limit)?;
        let rate = planned.format.sample_rate();
        let params = cutlass_codec::StreamParams::try_audio(
            planned.format.codec().clone(),
            rate,
            planned.format.channels(),
            planned.format.extradata(),
        )
        .map_err(|_| OpenError::Source {
            message: "audio parameters".to_owned(),
        })?;
        let stream = Stream::new(
            TrackId::new(0),
            params,
            Timescale::new(rate),
            None,
            Vec::new(),
            Trim::default(),
        );
        Ok(Self {
            input,
            streams: vec![stream],
            format: planned.format,
            brand: planned.brand,
            kept: planned.kept,
            dropped: planned.dropped,
            data_start: planned.data_start,
            data_len: planned.data_len,
            layout: planned.layout,
            next: 0,
            cursor: planned.data_start,
            samples_at: 0,
            cancelled: false,
        })
    }

    fn data_end(&self) -> u64 {
        self.data_start.saturating_add(self.data_len)
    }

    fn place(&mut self, at: u64) -> Result<(), ReadError> {
        if self.input.position() == at {
            return Ok(());
        }
        self.input.seek(at).map_err(ReadError::from_packet)
    }

    fn read_next(&mut self) -> Result<Option<Packet>, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        match self.layout {
            Layout::Constant { .. } => self.read_constant(),
            Layout::Mpeg => self.read_mpeg(),
        }
    }

    fn read_constant(&mut self) -> Result<Option<Packet>, ReadError> {
        let (frame_bytes, frames, per_frame, extra) = match self.layout {
            Layout::Constant {
                frame_bytes,
                frames,
                per_frame,
                extra,
            } => (frame_bytes, frames, per_frame, extra),
            Layout::Mpeg => return Ok(None),
        };
        if self.next >= frames {
            return Ok(None);
        }
        let offset = self
            .data_start
            .saturating_add(self.next.saturating_mul(frame_bytes));
        self.place(offset)?;
        let bytes = self
            .input
            .read_limited(frame_bytes, self.input.limits().max_packet_bytes)
            .map_err(ReadError::from_packet)?;
        let stamp = stamp_frame(self.next, frames, per_frame, extra)?;
        let packet = audio_packet(self.format.sample_rate(), bytes, stamp)?;
        self.next = self.next.saturating_add(1);
        self.cursor = offset.saturating_add(frame_bytes);
        Ok(Some(packet))
    }

    fn read_mpeg(&mut self) -> Result<Option<Packet>, ReadError> {
        let Some((bytes, head)) = self.pull_mpeg()? else {
            return Ok(None);
        };
        let start = ticks(self.samples_at)?;
        let duration = ticks(head.samples)?;
        self.samples_at = self.samples_at.saturating_add(head.samples);
        self.next = self.next.saturating_add(1);
        let packet = audio_packet(
            self.format.sample_rate(),
            bytes,
            (Some(start), Some(duration)),
        )?;
        Ok(Some(packet))
    }

    fn pull_mpeg(&mut self) -> Result<Option<(Vec<u8>, FrameHead)>, ReadError> {
        let end = self.data_end();
        if self.cursor >= end {
            return Ok(None);
        }
        let remaining = end - self.cursor;
        if remaining < 4 {
            return Err(ReadError::Truncated {
                expected: 4,
                got: remaining,
            });
        }
        self.place(self.cursor)?;
        let mut header = [0; 4];
        self.input
            .read_exact(&mut header)
            .map_err(ReadError::from_packet)?;
        if let Some(head) = parse_header(&header) {
            return self.take_mpeg_rest(header, head, remaining);
        }
        if self.input.access() == Access::Forward {
            return Err(ReadError::Source {
                message: "mpeg frame sync".to_owned(),
            });
        }
        self.place(self.cursor)?;
        self.resync_mpeg(remaining)
    }

    fn take_mpeg_rest(
        &mut self,
        header: [u8; 4],
        head: FrameHead,
        remaining: u64,
    ) -> Result<Option<(Vec<u8>, FrameHead)>, ReadError> {
        if head.len > remaining {
            return Err(ReadError::Truncated {
                expected: head.len,
                got: remaining,
            });
        }
        let rest_len = head.len.saturating_sub(4);
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(usize_from(head.len)?)
            .map_err(|_| ReadError::Source {
                message: "out of memory".to_owned(),
            })?;
        bytes.extend_from_slice(&header);
        if rest_len > 0 {
            let rest = self
                .input
                .read_limited(rest_len, self.input.limits().max_packet_bytes)
                .map_err(ReadError::from_packet)?;
            bytes.extend(rest);
        }
        self.cursor = self.cursor.saturating_add(head.len);
        Ok(Some((bytes, head)))
    }

    fn resync_mpeg(&mut self, remaining: u64) -> Result<Option<(Vec<u8>, FrameHead)>, ReadError> {
        let window = remaining.min(MAX_MPEG_RESYNC);
        let bytes = self
            .input
            .read_limited(window, MAX_MPEG_RESYNC)
            .map_err(ReadError::from_packet)?;
        let Some((at, head)) = find_frame(&bytes, 0) else {
            return Err(ReadError::Source {
                message: "mpeg frame sync".to_owned(),
            });
        };
        let at_u = u64::try_from(at).unwrap_or(0);
        let payload = bytes
            .get(at..at + usize_from(head.len)?)
            .ok_or(ReadError::Truncated {
                expected: head.len,
                got: remaining.saturating_sub(at_u),
            })?
            .to_vec();
        let next = self.cursor.saturating_add(at_u).saturating_add(head.len);
        self.place(next)?;
        self.cursor = next;
        Ok(Some((payload, head)))
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
        match self.layout {
            Layout::Constant { .. } => self.seek_constant(to),
            Layout::Mpeg => self.seek_mpeg(to),
        }
    }

    fn seek_constant(&mut self, to: SeekTo) -> Result<SeekLanding, ReadError> {
        let (frame_bytes, frames, per_frame, extra) = match self.layout {
            Layout::Constant {
                frame_bytes,
                frames,
                per_frame,
                extra,
            } => (frame_bytes, frames, per_frame, extra),
            Layout::Mpeg => {
                return Err(ReadError::Seek(SeekError::NoIndex));
            }
        };
        if frames == 0 {
            return Err(ReadError::Seek(SeekError::Empty));
        }
        if per_frame == 0 {
            return Err(ReadError::Seek(SeekError::NoIndex));
        }
        let total = frames * per_frame + extra;
        if past_end(to, Some(total)) {
            return self.land_end();
        }
        let time = if to.time < 0 {
            0
        } else {
            u64::try_from(to.time).unwrap_or(0)
        };
        let mut index = time / per_frame;
        if index >= frames {
            index = frames.saturating_sub(1);
        }
        let offset = self
            .data_start
            .saturating_add(index.saturating_mul(frame_bytes));
        let ordinal = u32::try_from(index).map_err(|_| ReadError::Source {
            message: "packet ordinal does not fit".to_owned(),
        })?;
        let pts = ticks(index.saturating_mul(per_frame))?;
        self.place(offset)?;
        self.next = index;
        self.cursor = offset;
        self.samples_at = index.saturating_mul(per_frame);
        Ok(packet_landing(ordinal, pts, offset))
    }

    fn land_end(&mut self) -> Result<SeekLanding, ReadError> {
        let end = self.data_end();
        self.place(end)?;
        self.next = match self.layout {
            Layout::Constant { frames, .. } => frames,
            Layout::Mpeg => self.next,
        };
        self.cursor = end;
        Ok(SeekLanding {
            landing: Landing::End,
            index_used: !matches!(self.layout, Layout::Mpeg),
            payload_scans: 0,
            entries_visited: 0,
        })
    }

    fn seek_mpeg(&mut self, to: SeekTo) -> Result<SeekLanding, ReadError> {
        if self.data_len == 0 {
            return Err(ReadError::Seek(SeekError::Empty));
        }
        if to.end.is_some_and(|end| to.time >= end) {
            return self.land_end_scans(0);
        }
        let saved = (
            self.cursor,
            self.next,
            self.samples_at,
            self.input.position(),
        );
        match self.walk_mpeg(to.time) {
            Ok(landing) => Ok(landing),
            Err(error) => {
                self.restore_mpeg(saved)?;
                Err(error)
            }
        }
    }

    fn walk_mpeg(&mut self, time: i64) -> Result<SeekLanding, ReadError> {
        let target = if time < 0 {
            0
        } else {
            u64::try_from(time).unwrap_or(u64::MAX)
        };
        let mut cursor = self.data_start;
        let mut samples = 0u64;
        let mut next = 0u64;
        let mut scans = 0u32;
        loop {
            if cursor >= self.data_end() {
                self.commit_mpeg(cursor, next, samples)?;
                return Ok(end_scans(scans));
            }
            let Some(head) = self.peek_frame(cursor)? else {
                return Err(ReadError::Source {
                    message: "mpeg frame sync".to_owned(),
                });
            };
            scans = scans.saturating_add(1);
            if samples >= target || target < samples.saturating_add(head.samples) {
                self.commit_mpeg(cursor, next, samples)?;
                let ordinal = u32::try_from(next).map_err(|_| ReadError::Source {
                    message: "packet ordinal does not fit".to_owned(),
                })?;
                let mut landing = packet_landing(ordinal, ticks(samples)?, cursor);
                landing.index_used = false;
                landing.payload_scans = scans;
                landing.entries_visited = scans;
                return Ok(landing);
            }
            let Some(advanced) = cursor.checked_add(head.len) else {
                return Err(ReadError::HostileSize {
                    declared: head.len,
                    limit: self.data_end().saturating_sub(cursor),
                });
            };
            if advanced > self.data_end() {
                return Err(ReadError::Truncated {
                    expected: head.len,
                    got: self.data_end().saturating_sub(cursor),
                });
            }
            cursor = advanced;
            samples = samples.saturating_add(head.samples);
            next = next.saturating_add(1);
        }
    }

    fn land_end_scans(&mut self, scans: u32) -> Result<SeekLanding, ReadError> {
        let end = self.data_end();
        self.place(end)?;
        self.cursor = end;
        Ok(end_scans(scans))
    }

    fn peek_frame(&mut self, at: u64) -> Result<Option<FrameHead>, ReadError> {
        if at.saturating_add(4) > self.data_end() {
            return Ok(None);
        }
        self.place(at)?;
        let mut header = [0; 4];
        self.input
            .read_exact(&mut header)
            .map_err(ReadError::from_packet)?;
        let parsed = parse_header(&header);
        self.place(at)?;
        Ok(parsed)
    }

    fn commit_mpeg(&mut self, cursor: u64, next: u64, samples: u64) -> Result<(), ReadError> {
        self.place(cursor)?;
        self.cursor = cursor;
        self.next = next;
        self.samples_at = samples;
        Ok(())
    }

    fn restore_mpeg(&mut self, saved: (u64, u64, u64, u64)) -> Result<(), ReadError> {
        self.cursor = saved.0;
        self.next = saved.1;
        self.samples_at = saved.2;
        self.place(saved.3)
    }
}

struct Planned {
    format: WavFormat,
    brand: Brand,
    kept: Vec<KeptChunk>,
    dropped: Vec<DroppedChunk>,
    data_start: u64,
    data_len: u64,
    layout: Layout,
}

fn plan(scanned: Scanned, max_packet: u64) -> Result<Planned, OpenError> {
    let Some(format) = scanned.format else {
        return Err(OpenError::Source {
            message: "missing fmt chunk".to_owned(),
        });
    };
    let Some(data) = scanned.data else {
        return Err(OpenError::Source {
            message: "missing data chunk".to_owned(),
        });
    };
    if let Some(ds_size) = scanned.ds64_data_size
        && data.declared != u32::MAX
        && u64::from(data.declared) != ds_size
    {
        return Err(OpenError::Source {
            message: DISAGREE_DATA.to_owned(),
        });
    }
    let layout = layout_for(
        &format,
        data.len,
        scanned.fact,
        scanned.ds64_samples,
        max_packet,
    )?;
    Ok(Planned {
        format,
        brand: scanned.brand,
        kept: scanned.kept,
        dropped: scanned.dropped,
        data_start: data.start,
        data_len: data.len,
        layout,
    })
}

fn layout_for(
    format: &WavFormat,
    data_len: u64,
    fact: Option<u64>,
    ds64_samples: Option<u64>,
    max_packet: u64,
) -> Result<Layout, OpenError> {
    match format.framing() {
        Framing::Samples => layout_samples(format, data_len, ds64_samples),
        Framing::Blocks => layout_blocks(format, data_len, fact, ds64_samples),
        Framing::Mpeg => Ok(Layout::Mpeg),
        Framing::Whole => layout_whole(data_len, fact, ds64_samples, max_packet),
    }
}

fn layout_samples(
    format: &WavFormat,
    data_len: u64,
    ds64_samples: Option<u64>,
) -> Result<Layout, OpenError> {
    let frame_bytes = u64::from(format.block_align());
    let frames = frames_of(data_len, frame_bytes)?;
    if let Some(samples) = ds64_samples
        && samples != frames
    {
        return Err(OpenError::Source {
            message: DISAGREE_SAMPLES.to_owned(),
        });
    }
    Ok(Layout::Constant {
        frame_bytes,
        frames,
        per_frame: 1,
        extra: 0,
    })
}

fn layout_blocks(
    format: &WavFormat,
    data_len: u64,
    fact: Option<u64>,
    ds64_samples: Option<u64>,
) -> Result<Layout, OpenError> {
    let frame_bytes = u64::from(format.block_align());
    let frames = frames_of(data_len, frame_bytes)?;
    let declared = named_samples(fact, ds64_samples)?;
    let formula = format
        .samples_per_block()
        .map(|samples| samples.saturating_mul(frames));
    let total = declared.or(formula);
    let (per_frame, extra) = split_samples(total, frames);
    Ok(Layout::Constant {
        frame_bytes,
        frames,
        per_frame,
        extra,
    })
}

fn layout_whole(
    data_len: u64,
    fact: Option<u64>,
    ds64_samples: Option<u64>,
    max_packet: u64,
) -> Result<Layout, OpenError> {
    if data_len > max_packet {
        return Err(OpenError::HostileSize {
            declared: data_len,
            limit: max_packet,
        });
    }
    let frames = u64::from(data_len > 0);
    let frame_bytes = if frames == 0 { 1 } else { data_len };
    let per_frame = named_samples(fact, ds64_samples)?.unwrap_or(0);
    Ok(Layout::Constant {
        frame_bytes,
        frames,
        per_frame,
        extra: 0,
    })
}

fn frames_of(data_len: u64, frame_bytes: u64) -> Result<u64, OpenError> {
    if frame_bytes == 0 || !data_len.is_multiple_of(frame_bytes) {
        return Err(OpenError::Source {
            message: "data size is not a multiple of the block align".to_owned(),
        });
    }
    Ok(data_len / frame_bytes)
}

fn named_samples(fact: Option<u64>, ds64_samples: Option<u64>) -> Result<Option<u64>, OpenError> {
    let fact = fact.filter(|count| *count != 0);
    let ds64 = ds64_samples.filter(|count| *count != 0);
    match (fact, ds64) {
        (Some(fact), Some(ds64)) if fact != ds64 => Err(OpenError::Source {
            message: DISAGREE_FACT.to_owned(),
        }),
        (Some(fact), _) => Ok(Some(fact)),
        (_, ds64) => Ok(ds64),
    }
}

fn split_samples(total: Option<u64>, frames: u64) -> (u64, u64) {
    let Some(total) = total else {
        return (0, 0);
    };
    let Some(per_frame) = total.checked_div(frames) else {
        return (0, 0);
    };
    (per_frame, total.checked_rem(frames).unwrap_or(0))
}

fn stamp_frame(
    index: u64,
    frames: u64,
    per_frame: u64,
    extra: u64,
) -> Result<(Option<i64>, Option<i64>), ReadError> {
    if per_frame == 0 {
        return Ok((None, None));
    }
    let start = index.saturating_mul(per_frame);
    let mut duration = per_frame;
    if index.saturating_add(1) == frames {
        duration = duration.saturating_add(extra);
    }
    Ok((Some(ticks(start)?), Some(ticks(duration)?)))
}

fn audio_packet(
    rate: u32,
    bytes: Vec<u8>,
    stamp: (Option<i64>, Option<i64>),
) -> Result<Packet, ReadError> {
    let timestamp = Timestamp::try_new(stamp.0, stamp.0, stamp.1).map_err(packet_error)?;
    Packet::try_new(0, bytes, timestamp, Timescale::new(rate), true).map_err(packet_error)
}

fn packet_error(error: PacketError) -> ReadError {
    ReadError::Source {
        message: error.to_string(),
    }
}

fn ticks(value: u64) -> Result<i64, ReadError> {
    i64::try_from(value).map_err(|_| ReadError::Source {
        message: "timestamp does not fit".to_owned(),
    })
}

fn usize_from(value: u64) -> Result<usize, ReadError> {
    usize::try_from(value).map_err(|_| ReadError::HostileSize {
        declared: value,
        limit: u64::try_from(usize::MAX).unwrap_or(u64::MAX),
    })
}

fn past_end(to: SeekTo, total: Option<u64>) -> bool {
    if to.end.is_some_and(|end| to.time >= end) {
        return true;
    }
    let Some(total) = total else {
        return false;
    };
    u64::try_from(to.time).is_ok_and(|time| time >= total)
}

fn packet_landing(ordinal: u32, time: i64, byte_position: u64) -> SeekLanding {
    SeekLanding {
        landing: Landing::Packet {
            ordinal,
            time,
            keyframe: true,
            preroll: 0,
            byte_position,
        },
        index_used: true,
        payload_scans: 0,
        entries_visited: 0,
    }
}

fn end_scans(scans: u32) -> SeekLanding {
    SeekLanding {
        landing: Landing::End,
        index_used: false,
        payload_scans: scans,
        entries_visited: scans,
    }
}

impl<R: Read + Seek> Demux for WavDemux<R> {
    fn family(&self) -> ContainerId {
        ContainerId::Riff
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
        self.input.consumed()
    }
}

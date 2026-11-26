//! AIFF demuxer.
//!
//! Open records where the sound starts and does not read it. Each read
//! returns one packet: a run of whole sample frames up to
//! [`PCM_PACKET_BYTES`] for linear, A-law and mu-law sound, and one codec
//! block otherwise. A seek uses the constant frame size and does not scan
//! payloads; on PCM it lands on the asked frame itself, and the next
//! packet starts there. Timestamps are sample-rate ticks.

use std::io::{Read, Seek};

use cutlass_codec::{ContainerId, Packet, StreamParams, Timescale, Timestamp};
use cutlass_container::{
    Access, Demux, Gap, Input, Landing, Limits, OpenError, ReadError, SeekError, SeekLanding,
    SeekTo, Stream, TrackId, Trim,
};

use crate::format::Sound;
use crate::limits::PCM_PACKET_BYTES;
use crate::walk::{self, Scanned};

/// One opened AIFF or AIFF-C file.
#[derive(Debug)]
pub struct AiffDemux<R> {
    input: Input<R>,
    streams: Vec<Stream>,
    sound: Sound,
    data_start: u64,
    /// Sample frames, or codec blocks, in the sound.
    units: u64,
    /// Units in one packet. One, except for PCM.
    group: u64,
    next: u64,
    skipped: u32,
    aifc: bool,
    cut: bool,
    /// The input ended before the header said it would.
    ended: bool,
    cancelled: bool,
}

impl<R: Read + Seek> AiffDemux<R> {
    /// Opens a random-access file. `length` is the file size, not an allocation.
    pub fn open(reader: R, length: u64) -> Result<Self, OpenError> {
        Self::open_limited(reader, length, Limits::standard())
    }

    /// Opens with caller bounds. A declared size past a bound is refused
    /// before that many bytes are allocated.
    pub fn open_limited(reader: R, length: u64, limits: Limits) -> Result<Self, OpenError> {
        Self::from_input(Input::random(reader, length, limits))
    }

    /// Opens a forward reader. Seek is refused. Chunks after the sound start
    /// are not read, and `SSND` before `COMM` is an error.
    pub fn open_forward(reader: R) -> Result<Self, OpenError> {
        Self::open_forward_limited(reader, Limits::standard())
    }

    /// [`Self::open_forward`] with caller bounds.
    pub fn open_forward_limited(reader: R, limits: Limits) -> Result<Self, OpenError> {
        Self::from_input(Input::forward(reader, limits))
    }

    fn from_input(mut input: Input<R>) -> Result<Self, OpenError> {
        let scanned = walk::scan(&mut input)?;
        Self::assemble(input, scanned)
    }

    fn assemble(input: Input<R>, scanned: Scanned) -> Result<Self, OpenError> {
        let stream = stream_of(&scanned.sound)?;
        let frame = u64::from(scanned.sound.frame_bytes()).max(1);
        let group = if scanned.sound.is_pcm() {
            (PCM_PACKET_BYTES.min(input.limits().max_packet_bytes) / frame).max(1)
        } else {
            1
        };
        let total = scanned
            .units
            .checked_mul(u64::from(scanned.sound.samples_per_packet()));
        if total.is_none_or(|total| i64::try_from(total).is_err()) {
            return Err(crate::format::open_source("sample count does not fit"));
        }
        Ok(Self {
            input,
            streams: vec![stream],
            sound: scanned.sound,
            data_start: scanned.data_start,
            units: scanned.units,
            group,
            next: 0,
            skipped: scanned.skipped,
            aifc: scanned.aifc,
            cut: scanned.cut,
            ended: false,
            cancelled: false,
        })
    }

    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.sound.rate()
    }

    #[must_use]
    pub const fn channels(&self) -> u16 {
        self.sound.channels()
    }

    /// Sample frames in the sound: the count `COMM` states, or the whole
    /// frames `SSND` holds when `COMM` states zero or more than is there.
    #[must_use]
    pub fn duration_samples(&self) -> u64 {
        self.units
            .saturating_mul(u64::from(self.sound.samples_per_packet()))
    }

    /// True when the file holds less sound than its header states: `SSND`
    /// or the form runs past the end of the file, `COMM` counts more
    /// frames than `SSND` holds, or the sound ends inside a frame. What is
    /// there plays. A forward reader learns this only when the input ends.
    #[must_use]
    pub const fn is_cut(&self) -> bool {
        self.cut
    }

    #[must_use]
    pub const fn skipped_chunks(&self) -> u32 {
        self.skipped
    }

    #[must_use]
    pub const fn is_aifc(&self) -> bool {
        self.aifc
    }

    fn read_next(&mut self) -> Result<Option<Packet>, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        if self.ended || self.next >= self.units {
            return Ok(None);
        }
        let frame = u64::from(self.sound.frame_bytes());
        let asked = self.group.min(self.units - self.next);
        let mut bytes = self.take(asked.saturating_mul(frame))?;
        let held = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        let taken = held.checked_div(frame).unwrap_or(0);
        if taken < asked {
            // The input ended first. Whole frames that arrived are sound.
            self.ended = true;
            self.cut = true;
            let keep = usize::try_from(taken.saturating_mul(frame)).unwrap_or(0);
            bytes.truncate(keep);
        }
        if taken == 0 {
            return Ok(None);
        }
        let step = u64::from(self.sound.samples_per_packet());
        let pts = ticks(self.next.saturating_mul(step))?;
        let duration = ticks(taken.saturating_mul(step))?;
        let timestamp = Timestamp::try_new(Some(pts), Some(pts), Some(duration))
            .map_err(|_| source("packet timestamp was refused"))?;
        self.next += taken;
        let packet = Packet::try_new(0, bytes, timestamp, Timescale::new(self.sound.rate()), true)
            .map_err(|_| source("packet was refused"))?;
        Ok(Some(packet))
    }

    /// Up to `want` bytes from the read position: fewer when the input
    /// ends first. The size is refused before it is allocated.
    fn take(&mut self, want: u64) -> Result<Vec<u8>, ReadError> {
        let limit = self.input.limits().max_packet_bytes;
        if want > limit {
            return Err(ReadError::HostileSize {
                declared: want,
                limit,
            });
        }
        let len = usize::try_from(want).map_err(|_| source("packet size does not fit"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .map_err(|_| source("out of memory"))?;
        bytes.resize(len, 0);
        let mut filled = 0;
        while filled < len {
            let got = self
                .input
                .read(&mut bytes[filled..])
                .map_err(ReadError::from_packet)?;
            if got == 0 {
                break;
            }
            filled += got;
        }
        bytes.truncate(filled);
        Ok(bytes)
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
        if self.units == 0 {
            return Err(ReadError::Seek(SeekError::Empty));
        }
        let total = ticks(self.duration_samples())?;
        if to.end.is_some_and(|end| to.time >= end) || to.time >= total {
            return self.land(self.units, true);
        }
        let time = u64::try_from(to.time.max(0)).unwrap_or(0);
        let step = u64::from(self.sound.samples_per_packet()).max(1);
        self.land(time / step, false)
    }

    fn land(&mut self, index: u64, end: bool) -> Result<SeekLanding, ReadError> {
        let bytes = index
            .checked_mul(u64::from(self.sound.frame_bytes()))
            .ok_or_else(|| source("seek offset overflowed"))?;
        let at = self
            .data_start
            .checked_add(bytes)
            .ok_or_else(|| source("seek offset overflowed"))?;
        self.input.seek(at).map_err(ReadError::from_packet)?;
        self.next = index;
        self.ended = false;
        if end {
            return Ok(SeekLanding {
                landing: Landing::End,
                index_used: true,
                payload_scans: 0,
                entries_visited: 0,
            });
        }
        // The landing is a frame, not a packet that was read before: the
        // next packet starts on it.
        let ordinal = u32::try_from(index).unwrap_or(u32::MAX);
        let step = u64::from(self.sound.samples_per_packet());
        let pts = ticks(index.saturating_mul(step))?;
        Ok(SeekLanding {
            landing: Landing::Packet {
                ordinal,
                time: pts,
                keyframe: true,
                preroll: 0,
                byte_position: at,
            },
            index_used: true,
            payload_scans: 0,
            entries_visited: 0,
        })
    }
}

fn source(message: &str) -> ReadError {
    ReadError::Source {
        message: message.to_owned(),
    }
}

fn ticks(value: u64) -> Result<i64, ReadError> {
    i64::try_from(value).map_err(|_| source("timestamp does not fit"))
}

fn stream_of(sound: &Sound) -> Result<Stream, OpenError> {
    let params = StreamParams::try_audio(
        sound.stream_codec()?,
        sound.rate(),
        sound.channels(),
        sound.extradata(),
    )
    .map_err(|error| crate::format::open_source(error.to_string()))?;
    Ok(Stream::new(
        TrackId::new(0),
        params,
        Timescale::new(sound.rate()),
        None,
        Vec::new(),
        Trim::default(),
    ))
}

impl<R: Read + Seek + Send> Demux for AiffDemux<R> {
    fn family(&self) -> ContainerId {
        ContainerId::Aiff
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

    fn indexed_end(&self, track: TrackId) -> Option<i64> {
        if track.get() != 0 || self.units == 0 {
            return None;
        }
        i64::try_from(self.duration_samples()).ok()
    }
}

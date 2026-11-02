//! AIFF demuxer.
//!
//! Open records where the sound starts and does not read it. Each read
//! returns one packet. A seek uses the constant frame size and does not
//! scan payloads. Timestamps are sample-rate ticks.

use std::io::{Read, Seek};

use cutlass_codec::{ContainerId, Packet, StreamParams, Timescale, Timestamp};
use cutlass_container::{
    Access, Demux, Gap, Input, Landing, Limits, OpenError, ReadError, SeekError, SeekLanding,
    SeekTo, Stream, TrackId, Trim,
};

use crate::format::Sound;
use crate::walk::{self, Scanned};

/// One opened AIFF or AIFF-C file.
#[derive(Debug)]
pub struct AiffDemux<R> {
    input: Input<R>,
    streams: Vec<Stream>,
    sound: Sound,
    data_start: u64,
    packets: u32,
    next: u64,
    skipped: u32,
    aifc: bool,
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
        let packets = scanned.sound.packet_count()?;
        let stream = stream_of(&scanned.sound)?;
        Ok(Self {
            input,
            streams: vec![stream],
            sound: scanned.sound,
            data_start: scanned.data_start,
            packets,
            next: 0,
            skipped: scanned.skipped,
            aifc: scanned.aifc,
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

    /// Sample frames from `COMM`, including zero.
    #[must_use]
    pub fn duration_samples(&self) -> u64 {
        u64::from(self.sound.sample_frames())
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
        if self.next >= u64::from(self.packets) {
            return Ok(None);
        }
        let size = usize::from(self.sound.frame_bytes());
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| ReadError::Source {
                message: "out of memory".to_owned(),
            })?;
        bytes.resize(size, 0);
        self.input
            .read_exact(&mut bytes)
            .map_err(ReadError::from_packet)?;
        let index = u32::try_from(self.next).map_err(|_| ReadError::Source {
            message: "packet index does not fit".to_owned(),
        })?;
        let step = i64::from(self.sound.samples_per_packet());
        let pts = i64::from(index)
            .checked_mul(step)
            .ok_or_else(|| ReadError::Source {
                message: "packet time overflowed".to_owned(),
            })?;
        let timestamp = Timestamp::try_new(Some(pts), Some(pts), Some(step)).map_err(|_| {
            ReadError::Source {
                message: "packet timestamp was refused".to_owned(),
            }
        })?;
        self.next += 1;
        let packet = Packet::try_new(0, bytes, timestamp, Timescale::new(self.sound.rate()), true)
            .map_err(|_| ReadError::Source {
                message: "packet was refused".to_owned(),
            })?;
        Ok(Some(packet))
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
        if self.packets == 0 {
            return Err(ReadError::Seek(SeekError::Empty));
        }
        let total = i64::from(self.sound.sample_frames());
        if to.end.is_some_and(|end| to.time >= end) || to.time >= total {
            return self.land(u64::from(self.packets), true);
        }
        let time = to.time.max(0);
        let step = i64::from(self.sound.samples_per_packet());
        let index = u64::try_from(time / step).map_err(|_| ReadError::Source {
            message: "seek index does not fit".to_owned(),
        })?;
        self.land(index, false)
    }

    fn land(&mut self, index: u64, end: bool) -> Result<SeekLanding, ReadError> {
        let bytes = index
            .checked_mul(u64::from(self.sound.frame_bytes()))
            .ok_or_else(|| ReadError::Source {
                message: "seek offset overflowed".to_owned(),
            })?;
        let at = self
            .data_start
            .checked_add(bytes)
            .ok_or_else(|| ReadError::Source {
                message: "seek offset overflowed".to_owned(),
            })?;
        self.input.seek(at).map_err(ReadError::from_packet)?;
        self.next = index;
        if end {
            return Ok(SeekLanding {
                landing: Landing::End,
                index_used: true,
                payload_scans: 0,
                entries_visited: 0,
            });
        }
        let ordinal = u32::try_from(index).map_err(|_| ReadError::Source {
            message: "packet ordinal does not fit".to_owned(),
        })?;
        let step = i64::from(self.sound.samples_per_packet());
        let pts = i64::from(ordinal)
            .checked_mul(step)
            .ok_or_else(|| ReadError::Source {
                message: "seek time overflowed".to_owned(),
            })?;
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

impl<R: Read + Seek> Demux for AiffDemux<R> {
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
}

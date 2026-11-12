//! One packet per frame out of framed audio chunks (see [`crate::frames`]).
//!
//! A chunk of an MP3, MPEG-1 Layer I or II, AC-3 or E-AC-3 stream is cut
//! into its frames, as FFmpeg's demuxer does through its parsers; a frame
//! that runs on into the stream's next chunk is joined to the rest of it.
//! On a constant-bitrate stream a frame's time is the time of its first
//! byte. On a variable-bitrate stream the first frame that starts in a
//! chunk takes the chunk's time and each later one follows the one before
//! by its sample count. Bytes between frames that no header claims are
//! dropped, and a chunk with no frame header at all is returned whole.

use cutlass_codec::{Packet, Timescale, Timestamp};
use cutlass_container::ReadError;

use crate::frames::{Framing, HEADER_BYTES, MAX_FRAME_BYTES};
use crate::track::{Clock, Track};

/// One chunk handed to the cutter.
pub(crate) struct Chunk {
    pub track: usize,
    pub bytes: Vec<u8>,
    /// Constant bitrate: the stream bytes before this chunk.
    pub offset: u64,
    /// The chunk's time and duration, as its table row gives them.
    pub time: i64,
    pub ticks: i64,
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
    /// For a chunk returned whole.
    chunk_ticks: i64,
    /// A frame starting in the chunk's own bytes has been returned.
    started: bool,
}

/// The start of a frame whose end is in the stream's next chunk.
struct Carry {
    bytes: Vec<u8>,
    offset: u64,
    time: i64,
}

pub(crate) struct Cutter {
    cutting: Option<Cutting>,
    /// Per track: the start of a frame that the track's next chunk ends.
    carries: Vec<Option<Carry>>,
}

impl Cutter {
    pub(crate) fn new(tracks: usize) -> Self {
        Self {
            cutting: None,
            carries: (0..tracks).map(|_| None).collect(),
        }
    }

    /// Forgets the chunk being cut and every carried frame, as a seek must.
    pub(crate) fn clear(&mut self) {
        self.cutting = None;
        for carry in &mut self.carries {
            *carry = None;
        }
    }

    /// Starts cutting `chunk`; [`Self::next`] then returns its frames.
    pub(crate) fn begin(&mut self, chunk: Chunk) {
        let carry = self.carries.get_mut(chunk.track).and_then(Option::take);
        let chunk_at = carry.as_ref().map_or(0, |carry| carry.bytes.len());
        let (bytes, offset, time) = match carry {
            Some(mut carry) => {
                carry.bytes.extend_from_slice(&chunk.bytes);
                (carry.bytes, carry.offset, carry.time)
            }
            None => (chunk.bytes, chunk.offset, chunk.time),
        };
        self.cutting = Some(Cutting {
            track: chunk.track,
            bytes,
            at: 0,
            chunk_at,
            offset,
            chunk_time: chunk.time,
            time,
            chunk_ticks: chunk.ticks,
            started: false,
        });
    }

    /// The next frame of the chunk being cut, or `None` once it is spent;
    /// a partial frame at its end is kept for the stream's next chunk.
    pub(crate) fn next(&mut self, tracks: &[Track]) -> Result<Option<Packet>, ReadError> {
        let Some(cutting) = self.cutting.as_mut() else {
            return Ok(None);
        };
        let Some(track) = tracks.get(cutting.track) else {
            self.cutting = None;
            return Ok(None);
        };
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
                    let (time, ticks) = (cutting.chunk_time, cutting.chunk_ticks);
                    self.cutting = None;
                    return packet(track.number, whole, time, ticks, track.clock.timescale)
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
        if let Some(slot) = self.carries.get_mut(index) {
            *slot = Some(Carry {
                bytes: rest,
                offset,
                time,
            });
        }
        Ok(None)
    }

    /// At the end of the stream, a carried partial frame is returned as it is.
    pub(crate) fn flush(&mut self, tracks: &[Track]) -> Result<Option<Packet>, ReadError> {
        let Some(index) = self.carries.iter().position(Option::is_some) else {
            return Ok(None);
        };
        let (Some(carry), Some(track)) = (self.carries[index].take(), tracks.get(index)) else {
            return Ok(None);
        };
        let (time, ticks) = if track.clock.sample_size > 0 {
            let time = track.byte_time(carry.offset).unwrap_or(carry.time);
            let end = carry.offset.saturating_add(carry.bytes.len() as u64);
            let ticks = track
                .byte_time(end)
                .map_or(0, |end| end.saturating_sub(time));
            (time, ticks)
        } else {
            (carry.time, 0)
        };
        packet(
            track.number,
            carry.bytes,
            time,
            ticks,
            track.clock.timescale,
        )
        .map(Some)
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

/// An audio packet: a keyframe whose presentation time is its decode time.
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

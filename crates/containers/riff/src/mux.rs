//! WAV muxer.
//!
//! The file is written after the duration check accepts the packets, so a
//! mismatch does not leave a complete RIFF. `fact` is written when the
//! stored format tag is not `0x0001`. Integer PCM at any sample rate stays
//! PCMWAVEFORMAT unless the stream carries a mask, more than two channels,
//! a non-matching valid-bit count, or the source was already extensible.

use std::io::{Seek, Write};

use cutlass_codec::{ContainerId, Packet};
use cutlass_container::{
    FailedMux, FinishedMux, IndexPlace, Limits, Mux, MuxDraft, MuxError, MuxMode, MuxStatus,
    NewStream, TrackDuration, TrackId,
};

use crate::format::{FormatError, Framing, WavFormat};
use crate::limits::MAX_KEPT_CHUNK;
use crate::mpeg::parse_header;
use crate::walk::KeptChunk;

/// Writes one WAV or RF64 stream.
pub struct WavMux<W: Write + Seek> {
    writer: W,
    draft: MuxDraft,
    format: Option<WavFormat>,
    kept: Vec<KeptChunk>,
    rf64: bool,
}

impl<W: Write + Seek> WavMux<W> {
    #[must_use]
    pub fn new(writer: W) -> Self {
        Self::build(writer, false)
    }

    /// Writes RF64 with `ds64` even when the file fits in a classic RIFF.
    #[must_use]
    pub fn rf64(writer: W) -> Self {
        Self::build(writer, true)
    }

    /// Keeps a chunk this crate does not interpret and writes it back before `data`.
    pub fn keep_chunk(&mut self, id: [u8; 4], payload: Vec<u8>) -> Result<(), MuxError> {
        if self.draft.status() != MuxStatus::Building {
            return Err(MuxError::NotBuilding);
        }
        if id == *b"fmt " || id == *b"data" || id == *b"fact" || id == *b"ds64" {
            return Err(self.stop(MuxError::Rejected {
                message: "chunk id is reserved".to_owned(),
            }));
        }
        let size = u64::try_from(payload.len()).unwrap_or(u64::MAX);
        if size > MAX_KEPT_CHUNK {
            return Err(self.stop(MuxError::HostileSize {
                declared: size,
                limit: MAX_KEPT_CHUNK,
            }));
        }
        self.kept.push(KeptChunk::new(id, payload));
        Ok(())
    }

    #[must_use]
    pub fn into_inner(self) -> W {
        self.writer
    }

    fn build(writer: W, rf64: bool) -> Self {
        let mut limits = Limits::standard();
        limits.max_tracks = 1;
        Self {
            writer,
            draft: MuxDraft::new(
                ContainerId::Riff,
                MuxMode::Progressive,
                IndexPlace::Absent,
                limits,
            ),
            format: None,
            kept: Vec::new(),
            rf64,
        }
    }

    fn stop(&mut self, error: MuxError) -> MuxError {
        self.draft.abort(error.clone());
        error
    }

    fn prepare_and_write(&mut self, durations: &[TrackDuration]) -> Result<(), MuxError> {
        self.check_durations(durations)?;
        let bytes = self.encode()?;
        self.writer
            .write_all(&bytes)
            .map_err(|error| MuxError::Rejected {
                message: format!("wav write failed: {error}"),
            })
    }

    fn check_durations(&self, durations: &[TrackDuration]) -> Result<(), MuxError> {
        let Some(format) = &self.format else {
            return Err(MuxError::Rejected {
                message: "wav needs one audio stream".to_owned(),
            });
        };
        if durations.len() != 1 {
            return Err(MuxError::Incomplete);
        }
        let duration = &durations[0];
        if duration.track != TrackId::new(0) || duration.ticks < 0 {
            return Err(if duration.ticks < 0 {
                MuxError::NegativeDuration
            } else {
                MuxError::Incomplete
            });
        }
        if duration.timescale.ticks_per_second() != format.sample_rate() {
            return Err(MuxError::TimescaleMismatch {
                packet: duration.timescale.ticks_per_second(),
                stream: format.sample_rate(),
            });
        }
        let packets = self.draft.packets().ok_or(MuxError::NotBuilding)?;
        if duration.ticks == 0 && packets.is_empty() {
            return Ok(());
        }
        if packets.is_empty() {
            return Err(MuxError::ShortPayload { track: 0 });
        }
        let sum = summed_duration(packets)?;
        if sum != duration.ticks {
            return Err(MuxError::DurationMismatch { track: 0 });
        }
        Ok(())
    }

    fn encode(&self) -> Result<Vec<u8>, MuxError> {
        let Some(format) = &self.format else {
            return Err(MuxError::Rejected {
                message: "wav needs one audio stream".to_owned(),
            });
        };
        let packets = self.draft.packets().ok_or(MuxError::NotBuilding)?;
        let (data, samples) = payload_of(packets)?;
        let avg = avg_bytes(
            u64::try_from(data.len()).unwrap_or(u64::MAX),
            samples,
            format.sample_rate(),
            format.block_align(),
        );
        let fmt = format.fmt_payload(avg).map_err(FormatError::into_mux)?;
        let stored_tag = u16::from_le_bytes([fmt[0], fmt[1]]);
        let fact = if stored_tag == 0x0001 {
            None
        } else {
            Some(fact_count(samples, self.rf64)?)
        };
        assemble(self.rf64, &fmt, fact, &self.kept, &data, samples)
    }

    fn read_format(stream: &NewStream) -> Result<WavFormat, MuxError> {
        let channels = stream.params.channels().ok_or_else(audio_rejected)?;
        let rate = stream.params.sample_rate().ok_or_else(audio_rejected)?;
        WavFormat::from_extradata(
            stream.params.extradata(),
            channels,
            rate,
            stream.params.codec(),
        )
        .map_err(FormatError::into_mux)
    }

    fn check_packet(&self, packet: &Packet) -> Result<(), MuxError> {
        let Some(format) = &self.format else {
            return Err(MuxError::Rejected {
                message: "add a stream before a packet".to_owned(),
            });
        };
        match format.framing() {
            Framing::Samples => check_sample_packet(format, packet),
            Framing::Blocks => check_block_packet(format, packet),
            Framing::Mpeg => check_mpeg_packet(packet),
            Framing::Whole => {
                if packet.timestamp().duration().is_none() {
                    return Err(MuxError::MissingTimestamp);
                }
                Ok(())
            }
        }
    }
}

impl<W: Write + Seek> Mux for WavMux<W> {
    fn family(&self) -> ContainerId {
        ContainerId::Riff
    }

    fn add_stream(&mut self, stream: NewStream) -> Result<TrackId, MuxError> {
        let format = match Self::read_format(&stream) {
            Ok(format) => format,
            Err(error) => return Err(self.stop(error)),
        };
        if stream.timescale.ticks_per_second() != format.sample_rate() {
            return Err(self.stop(MuxError::TimescaleMismatch {
                packet: stream.timescale.ticks_per_second(),
                stream: format.sample_rate(),
            }));
        }
        let id = self.draft.add_stream(stream)?;
        self.format = Some(format);
        Ok(id)
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<(), MuxError> {
        if let Err(error) = self.check_packet(packet) {
            return Err(self.stop(error));
        }
        self.draft.write_packet(packet)
    }

    fn interleave(&mut self) -> Result<(), MuxError> {
        self.draft.interleave()
    }

    fn cancel(&mut self) {
        self.draft.cancel();
    }

    fn finish(&mut self, durations: &[TrackDuration]) -> Result<FinishedMux, FailedMux> {
        if self.draft.status() != MuxStatus::Building {
            return self.draft.finish(durations);
        }
        if let Err(error) = self.prepare_and_write(durations) {
            self.draft.abort(error);
        }
        self.draft.finish(durations)
    }

    fn status(&self) -> MuxStatus {
        self.draft.status()
    }
}

fn audio_rejected() -> MuxError {
    MuxError::Rejected {
        message: "wav stores one audio stream".to_owned(),
    }
}

fn summed_duration(packets: &[Packet]) -> Result<i64, MuxError> {
    let mut sum = 0i64;
    for packet in packets {
        let Some(span) = packet.timestamp().duration() else {
            return Err(MuxError::MissingTimestamp);
        };
        sum = sum.checked_add(span).ok_or(MuxError::HostileSize {
            declared: u64::try_from(span).unwrap_or(u64::MAX),
            limit: u64::try_from(i64::MAX).unwrap_or(u64::MAX),
        })?;
    }
    Ok(sum)
}

fn payload_of(packets: &[Packet]) -> Result<(Vec<u8>, u64), MuxError> {
    let mut data = Vec::new();
    let mut samples = 0u64;
    for packet in packets {
        let span = packet
            .timestamp()
            .duration()
            .ok_or(MuxError::MissingTimestamp)?;
        let span_u = u64::try_from(span).map_err(|_| MuxError::NegativeDuration)?;
        samples = samples.checked_add(span_u).ok_or(MuxError::HostileSize {
            declared: span_u,
            limit: u64::MAX - samples,
        })?;
        data.try_reserve(packet.bytes().len())
            .map_err(|_| MuxError::Rejected {
                message: "out of memory".to_owned(),
            })?;
        data.extend_from_slice(packet.bytes());
    }
    Ok((data, samples))
}

fn avg_bytes(payload: u64, samples: u64, rate: u32, align: u16) -> u32 {
    let rate = u64::from(rate);
    let value = if samples == 0 {
        rate.saturating_mul(u64::from(align))
    } else {
        payload
            .saturating_mul(rate)
            .checked_div(samples)
            .unwrap_or(0)
    };
    u32::try_from(value).unwrap_or(u32::MAX)
}

fn fact_count(samples: u64, rf64: bool) -> Result<u32, MuxError> {
    match u32::try_from(samples) {
        Ok(value) => Ok(value),
        Err(_) if rf64 => Ok(u32::MAX),
        Err(_) => Err(MuxError::HostileSize {
            declared: samples,
            limit: u64::from(u32::MAX),
        }),
    }
}

fn assemble(
    rf64: bool,
    fmt: &[u8],
    fact: Option<u32>,
    kept: &[KeptChunk],
    data: &[u8],
    samples: u64,
) -> Result<Vec<u8>, MuxError> {
    let mut body = Vec::new();
    let ds64_at = if rf64 {
        let at = body.len();
        push_chunk(&mut body, b"ds64", &[0; 28], 28)?;
        Some(at)
    } else {
        None
    };
    push_chunk(&mut body, b"fmt ", fmt, chunk_size(fmt.len())?)?;
    if let Some(samples) = fact {
        push_chunk(&mut body, b"fact", &samples.to_le_bytes(), 4)?;
    }
    for chunk in kept {
        push_chunk(
            &mut body,
            &chunk.id(),
            chunk.payload(),
            chunk_size(chunk.payload().len())?,
        )?;
    }
    let data_field = if rf64 {
        u32::MAX
    } else {
        chunk_size(data.len())?
    };
    push_chunk(&mut body, b"data", data, data_field)?;
    let riff = 4u64.saturating_add(u64::try_from(body.len()).unwrap_or(u64::MAX));
    if !rf64 && riff > u64::from(u32::MAX) {
        return Err(MuxError::HostileSize {
            declared: riff,
            limit: u64::from(u32::MAX),
        });
    }
    if let Some(at) = ds64_at {
        patch_ds64(
            &mut body,
            at,
            riff,
            u64::try_from(data.len()).unwrap_or(u64::MAX),
            samples,
        )?;
    }
    let mut file = Vec::new();
    file.try_reserve(12 + body.len())
        .map_err(|_| MuxError::Rejected {
            message: "out of memory".to_owned(),
        })?;
    file.extend_from_slice(if rf64 { b"RF64" } else { b"RIFF" });
    let size_field = if rf64 {
        u32::MAX
    } else {
        u32::try_from(riff).unwrap_or(u32::MAX)
    };
    file.extend_from_slice(&size_field.to_le_bytes());
    file.extend_from_slice(b"WAVE");
    file.extend(body);
    Ok(file)
}

fn patch_ds64(
    body: &mut [u8],
    chunk_at: usize,
    riff: u64,
    data_len: u64,
    samples: u64,
) -> Result<(), MuxError> {
    let at = chunk_at.saturating_add(8);
    let Some(slot) = body.get_mut(at..at + 24) else {
        return Err(MuxError::Rejected {
            message: "ds64 slot".to_owned(),
        });
    };
    slot[0..8].copy_from_slice(&riff.to_le_bytes());
    slot[8..16].copy_from_slice(&data_len.to_le_bytes());
    slot[16..24].copy_from_slice(&samples.to_le_bytes());
    Ok(())
}

fn push_chunk(
    out: &mut Vec<u8>,
    id: &[u8; 4],
    payload: &[u8],
    size_field: u32,
) -> Result<(), MuxError> {
    out.try_reserve(8 + payload.len() + 1)
        .map_err(|_| MuxError::Rejected {
            message: "out of memory".to_owned(),
        })?;
    out.extend_from_slice(id);
    out.extend_from_slice(&size_field.to_le_bytes());
    out.extend_from_slice(payload);
    if !payload.len().is_multiple_of(2) {
        out.push(0);
    }
    Ok(())
}

fn chunk_size(len: usize) -> Result<u32, MuxError> {
    u32::try_from(len).map_err(|_| MuxError::HostileSize {
        declared: u64::try_from(len).unwrap_or(u64::MAX),
        limit: u64::from(u32::MAX),
    })
}

fn check_sample_packet(format: &WavFormat, packet: &Packet) -> Result<(), MuxError> {
    let frames = aligned_frames(format, packet)?;
    let duration = packet
        .timestamp()
        .duration()
        .ok_or(MuxError::MissingTimestamp)?;
    if u64::try_from(duration).ok() != Some(frames) {
        return Err(MuxError::Rejected {
            message: "packet duration does not match the frames".to_owned(),
        });
    }
    Ok(())
}

fn check_block_packet(format: &WavFormat, packet: &Packet) -> Result<(), MuxError> {
    let frames = aligned_frames(format, packet)?;
    let duration = packet
        .timestamp()
        .duration()
        .ok_or(MuxError::MissingTimestamp)?;
    if let Some(samples) = format.samples_per_block() {
        let expect = frames.saturating_mul(samples);
        if u64::try_from(duration).ok() != Some(expect) {
            return Err(MuxError::Rejected {
                message: "packet duration does not match the block".to_owned(),
            });
        }
    }
    Ok(())
}

fn check_mpeg_packet(packet: &Packet) -> Result<(), MuxError> {
    let bytes = packet.bytes();
    let Some(header_bytes) = bytes.get(..4) else {
        return Err(MuxError::Rejected {
            message: "mpeg frame is short".to_owned(),
        });
    };
    let mut header = [0; 4];
    header.copy_from_slice(header_bytes);
    let Some(head) = parse_header(&header) else {
        return Err(MuxError::Rejected {
            message: "mpeg frame sync".to_owned(),
        });
    };
    if u64::try_from(bytes.len()).ok() != Some(head.len) {
        return Err(MuxError::Rejected {
            message: "mpeg frame length".to_owned(),
        });
    }
    let duration = packet
        .timestamp()
        .duration()
        .ok_or(MuxError::MissingTimestamp)?;
    if u64::try_from(duration).ok() != Some(head.samples) {
        return Err(MuxError::Rejected {
            message: "mpeg frame duration".to_owned(),
        });
    }
    Ok(())
}

fn aligned_frames(format: &WavFormat, packet: &Packet) -> Result<u64, MuxError> {
    let align = u64::from(format.block_align());
    let len = u64::try_from(packet.bytes().len()).unwrap_or(u64::MAX);
    if align == 0 || !len.is_multiple_of(align) {
        return Err(MuxError::Rejected {
            message: "packet is not a whole number of frames".to_owned(),
        });
    }
    Ok(len / align)
}

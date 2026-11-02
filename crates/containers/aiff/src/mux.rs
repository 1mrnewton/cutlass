//! AIFF muxer.
//!
//! The file is written after the duration check accepts the packets, so a
//! mismatch does not leave a complete FORM. Big-endian integer PCM is
//! classic AIFF. Float and little-endian integer are AIFF-C.

use std::io::{Seek, Write};

use cutlass_codec::{ContainerId, Packet};
use cutlass_container::{
    FailedMux, FinishedMux, IndexPlace, Limits, Mux, MuxDraft, MuxError, MuxMode, MuxStatus,
    NewStream, TrackDuration, TrackId,
};

use crate::format::{self, MuxPcm};
use crate::rate;

/// AIFF-C version timestamp written in `FVER`.
const AIFC_VERSION: u32 = 0xA280_5140;

/// Writes one AIFF or AIFF-C stream.
pub struct AiffMux<W: Write + Seek> {
    writer: W,
    draft: MuxDraft,
    pcm: Option<MuxPcm>,
}

impl<W: Write + Seek> AiffMux<W> {
    #[must_use]
    pub fn new(writer: W) -> Self {
        let mut limits = Limits::standard();
        limits.max_tracks = 1;
        Self {
            writer,
            draft: MuxDraft::new(
                ContainerId::Aiff,
                MuxMode::Progressive,
                IndexPlace::Absent,
                limits,
            ),
            pcm: None,
        }
    }

    #[must_use]
    pub fn writer(&self) -> &W {
        &self.writer
    }

    #[must_use]
    pub fn into_inner(self) -> W {
        self.writer
    }

    fn stop(&mut self, error: MuxError) -> MuxError {
        self.draft.abort(error.clone());
        error
    }

    fn prepare_and_write(&mut self, durations: &[TrackDuration]) -> Result<(), MuxError> {
        self.check(durations)?;
        let bytes = self.encode()?;
        self.writer
            .write_all(&bytes)
            .map_err(|error| rejected(format!("aiff write failed: {error}")))
    }

    fn check(&self, durations: &[TrackDuration]) -> Result<(), MuxError> {
        let Some(pcm) = &self.pcm else {
            return Err(rejected("AIFF needs one audio stream"));
        };
        if durations.len() != 1 {
            return Err(MuxError::Incomplete);
        }
        let duration = &durations[0];
        if duration.ticks < 0 {
            return Err(MuxError::NegativeDuration);
        }
        if duration.track != TrackId::new(0) {
            return Err(MuxError::Incomplete);
        }
        if duration.timescale.ticks_per_second() != pcm.rate {
            return Err(MuxError::TimescaleMismatch {
                packet: duration.timescale.ticks_per_second(),
                stream: pcm.rate,
            });
        }
        let packets = self.draft.packets().ok_or(MuxError::NotBuilding)?;
        if duration.ticks == 0 && packets.is_empty() {
            return Ok(());
        }
        if packets.is_empty() {
            return Err(MuxError::ShortPayload { track: 0 });
        }
        let count =
            i64::try_from(packets.len()).map_err(|_| rejected("packet count does not fit"))?;
        if duration.ticks != count {
            return Err(MuxError::DurationMismatch { track: 0 });
        }
        Ok(())
    }

    fn encode(&self) -> Result<Vec<u8>, MuxError> {
        let Some(pcm) = self.pcm else {
            return Err(rejected("AIFF needs one audio stream"));
        };
        let packets = self.draft.packets().ok_or(MuxError::NotBuilding)?;
        let frame = pcm.frame_bytes()?;
        let sound = sound_bytes(packets, frame)?;
        let frames = u32::try_from(packets.len())
            .map_err(|_| rejected("sample count does not fit in COMM"))?;
        let comm = comm_bytes(pcm, frames)?;
        let ssnd = ssnd_payload(&sound);
        form(pcm.fourcc.is_some(), &comm, &ssnd)
    }

    fn check_packet(&self, packet: &Packet) -> Result<(), MuxError> {
        let Some(pcm) = self.pcm else {
            return Err(rejected("add a stream before a packet"));
        };
        let packets = self.draft.packets().ok_or(MuxError::NotBuilding)?;
        let expect =
            i64::try_from(packets.len()).map_err(|_| rejected("packet count does not fit"))?;
        if packet.stream() != 0 {
            return Err(MuxError::UnknownTrack {
                id: packet.stream(),
            });
        }
        let Some(pts) = packet.timestamp().presentation() else {
            return Err(MuxError::MissingTimestamp);
        };
        let Some(duration) = packet.timestamp().duration() else {
            return Err(MuxError::MissingTimestamp);
        };
        if pts != expect || duration != 1 {
            return Err(MuxError::DurationMismatch { track: 0 });
        }
        if packet.bytes().len() != pcm.frame_bytes()? {
            return Err(rejected("packet is not one sample frame"));
        }
        Ok(())
    }
}

impl<W: Write + Seek> Mux for AiffMux<W> {
    fn family(&self) -> ContainerId {
        ContainerId::Aiff
    }

    fn add_stream(&mut self, stream: NewStream) -> Result<TrackId, MuxError> {
        let pcm = match format::mux_pcm(&stream) {
            Ok(pcm) => pcm,
            Err(error) => return Err(self.stop(error)),
        };
        if stream.timescale.ticks_per_second() != pcm.rate {
            return Err(self.stop(MuxError::TimescaleMismatch {
                packet: stream.timescale.ticks_per_second(),
                stream: pcm.rate,
            }));
        }
        let id = self.draft.add_stream(stream)?;
        self.pcm = Some(pcm);
        Ok(id)
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<(), MuxError> {
        if self.draft.status() != MuxStatus::Building {
            return self.draft.write_packet(packet);
        }
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

fn sound_bytes(packets: &[Packet], frame: usize) -> Result<Vec<u8>, MuxError> {
    let mut sound = Vec::new();
    for (index, packet) in packets.iter().enumerate() {
        let expect = i64::try_from(index).map_err(|_| rejected("packet count does not fit"))?;
        if packet.stream() != 0 {
            return Err(MuxError::UnknownTrack {
                id: packet.stream(),
            });
        }
        let Some(pts) = packet.timestamp().presentation() else {
            return Err(MuxError::MissingTimestamp);
        };
        let Some(duration) = packet.timestamp().duration() else {
            return Err(MuxError::MissingTimestamp);
        };
        if pts != expect || duration != 1 {
            return Err(MuxError::DurationMismatch { track: 0 });
        }
        if packet.bytes().len() != frame {
            return Err(rejected("packet is not one sample frame"));
        }
        sound.extend(packet.bytes());
    }
    Ok(sound)
}

fn comm_bytes(pcm: MuxPcm, frames: u32) -> Result<Vec<u8>, MuxError> {
    let mut body = Vec::with_capacity(26);
    body.extend(pcm.channels.to_be_bytes());
    body.extend(frames.to_be_bytes());
    body.extend(pcm.bits.to_be_bytes());
    let rate = rate::encode_rate(pcm.rate).map_err(|error| rejected(error.to_string()))?;
    body.extend(rate);
    if let Some(tag) = pcm.fourcc {
        body.extend(tag);
        body.push(3);
        body.extend(b"pcm");
    }
    Ok(body)
}

fn ssnd_payload(sound: &[u8]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(sound.len() + 8);
    payload.extend(0u32.to_be_bytes());
    payload.extend(0u32.to_be_bytes());
    payload.extend(sound);
    payload
}

fn form(aifc: bool, comm: &[u8], ssnd: &[u8]) -> Result<Vec<u8>, MuxError> {
    let mut body = Vec::new();
    if aifc {
        body.extend_from_slice(b"AIFC");
        push_chunk(&mut body, b"FVER", &AIFC_VERSION.to_be_bytes())?;
    } else {
        body.extend_from_slice(b"AIFF");
    }
    push_chunk(&mut body, b"COMM", comm)?;
    push_chunk(&mut body, b"SSND", ssnd)?;
    let size = u32::try_from(body.len()).map_err(|_| rejected("AIFF form is larger than 4 GiB"))?;
    let mut file = Vec::with_capacity(body.len() + 8);
    file.extend_from_slice(b"FORM");
    file.extend(size.to_be_bytes());
    file.extend(body);
    Ok(file)
}

fn push_chunk(out: &mut Vec<u8>, id: &[u8; 4], payload: &[u8]) -> Result<(), MuxError> {
    let size = u32::try_from(payload.len()).map_err(|_| rejected("chunk does not fit"))?;
    out.extend(id);
    out.extend(size.to_be_bytes());
    out.extend(payload);
    if payload.len() % 2 == 1 {
        out.push(0);
    }
    Ok(())
}

fn rejected(message: impl Into<String>) -> MuxError {
    MuxError::Rejected {
        message: message.into(),
    }
}

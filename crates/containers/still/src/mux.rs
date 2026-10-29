//! Write still packets back to a file.
//!
//! A single picture is the packet bytes, unchanged. GIF and animated WebP
//! packets are concatenated in write order, which is the file when they
//! came from [`crate::StillDemux`]. A loop count set on the muxer is written
//! into the Netscape block or the `ANIM` chunk. A finish that does not
//! produce a file this demuxer can open leaves no bytes.

use crate::demux::{StillDemux, index_place};
use crate::limits::{MAX_FILE_BYTES, MAX_FRAMES, file_cap, still_limits};
use cutlass_codec::{CodecId, ContainerId, Packet, StreamCodec, UnknownCodec};
use cutlass_container::{
    FailedMux, FinishedMux, Input, Mux, MuxError, MuxMode, MuxStatus, NewStream, TrackDuration,
    TrackId,
};
use std::io::Cursor;

/// Muxer for one JPEG, PNG, WebP, or GIF stream.
#[derive(Debug)]
pub struct StillMux {
    cap: u64,
    stream: Option<NewStreamSlot>,
    packets: Vec<Packet>,
    loop_count: Option<u16>,
    output: Vec<u8>,
    status: Status,
    payload: u64,
}

#[derive(Debug)]
struct NewStreamSlot {
    id: TrackId,
    timescale: cutlass_codec::Timescale,
}

#[derive(Debug)]
enum Status {
    Building,
    Failed(MuxError),
    Finished,
}

impl StillMux {
    #[must_use]
    pub fn new() -> Self {
        Self {
            cap: file_cap(still_limits()),
            stream: None,
            packets: Vec::new(),
            loop_count: None,
            output: Vec::new(),
            status: Status::Building,
            payload: 0,
        }
    }

    /// Loop count to write. `0` means forever.
    ///
    /// Ignored once the mux has finished or failed. Packets are not repeated.
    pub fn set_loop_count(&mut self, count: u16) {
        if matches!(self.status, Status::Building) {
            self.loop_count = Some(count);
        }
    }

    /// File bytes after a successful finish.
    ///
    /// A failed or unfinished mux has none.
    #[must_use]
    pub fn bytes(&self) -> Option<&[u8]> {
        match self.status {
            Status::Finished => Some(&self.output),
            Status::Building | Status::Failed(_) => None,
        }
    }

    fn building(&self) -> Result<(), MuxError> {
        match &self.status {
            Status::Building => Ok(()),
            Status::Failed(error) => Err(error.clone()),
            Status::Finished => Err(MuxError::AlreadyFinished),
        }
    }

    fn fail(&mut self, reason: MuxError) -> MuxError {
        self.output.clear();
        self.status = Status::Failed(reason.clone());
        reason
    }

    fn failed(&mut self, reason: MuxError) -> FailedMux {
        let reason = self.fail(reason);
        FailedMux::incomplete(self.payload, reason)
    }
}

impl Default for StillMux {
    fn default() -> Self {
        Self::new()
    }
}

impl Mux for StillMux {
    fn family(&self) -> ContainerId {
        ContainerId::Still
    }

    fn add_stream(&mut self, stream: NewStream) -> Result<TrackId, MuxError> {
        self.building()?;
        if self.stream.is_some() {
            return Err(self.fail(MuxError::TooManyTracks { count: 1, limit: 1 }));
        }
        if let Err(error) = image_codec(stream.params.codec()) {
            return Err(self.fail(error));
        }
        let id = TrackId::new(0);
        self.stream = Some(NewStreamSlot {
            id,
            timescale: stream.timescale,
        });
        Ok(id)
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<(), MuxError> {
        self.building()?;
        let Some(stream) = &self.stream else {
            return Err(self.fail(MuxError::UnknownTrack {
                id: packet.stream(),
            }));
        };
        if packet.stream() != stream.id.get() {
            return Err(self.fail(MuxError::UnknownTrack {
                id: packet.stream(),
            }));
        }
        if packet.timescale() != stream.timescale {
            return Err(self.fail(MuxError::TimescaleMismatch {
                packet: packet.timescale().ticks_per_second(),
                stream: stream.timescale.ticks_per_second(),
            }));
        }
        let size = u64::try_from(packet.bytes().len()).unwrap_or(u64::MAX);
        if size > self.cap {
            return Err(self.fail(MuxError::HostileSize {
                declared: size,
                limit: self.cap,
            }));
        }
        let Some(total) = self.payload.checked_add(size) else {
            return Err(self.fail(MuxError::HostileSize {
                declared: size,
                limit: self.cap.saturating_sub(self.payload),
            }));
        };
        if total > self.cap {
            return Err(self.fail(MuxError::HostileSize {
                declared: total,
                limit: self.cap,
            }));
        }
        if self.packets.len() >= usize::try_from(MAX_FRAMES).unwrap_or(usize::MAX) {
            return Err(self.fail(MuxError::HostileSize {
                declared: u64::from(MAX_FRAMES) + 1,
                limit: u64::from(MAX_FRAMES),
            }));
        }
        self.payload = total;
        self.packets.push(packet.clone());
        Ok(())
    }

    fn interleave(&mut self) -> Result<(), MuxError> {
        self.building()?;
        let clockless = self.packets.len() == 1
            && self
                .packets
                .first()
                .is_some_and(|packet| !packet.timescale().is_clock());
        if self
            .packets
            .iter()
            .any(|packet| packet.timestamp().presentation_or_decode().is_none())
        {
            if clockless {
                return Ok(());
            }
            return Err(MuxError::MissingTimestamp);
        }
        self.packets.sort_by(|left, right| {
            let left_at = left.timestamp().presentation_or_decode().unwrap_or(0);
            let right_at = right.timestamp().presentation_or_decode().unwrap_or(0);
            left_at.cmp(&right_at)
        });
        Ok(())
    }

    fn cancel(&mut self) {
        if matches!(self.status, Status::Building) {
            self.output.clear();
            self.status = Status::Failed(MuxError::Cancelled);
        }
    }

    fn finish(&mut self, durations: &[TrackDuration]) -> Result<FinishedMux, FailedMux> {
        match &self.status {
            Status::Finished => {
                return Err(FailedMux::incomplete(
                    self.payload,
                    MuxError::AlreadyFinished,
                ));
            }
            Status::Failed(error) => {
                return Err(FailedMux::incomplete(self.payload, error.clone()));
            }
            Status::Building => {}
        }
        let Some(stream) = self.stream.as_ref() else {
            return Err(self.failed(MuxError::Incomplete));
        };
        if let Err(error) = check_durations(stream, &self.packets, durations) {
            return Err(self.failed(error));
        }
        let mut bytes = Vec::new();
        for packet in &self.packets {
            bytes.extend_from_slice(packet.bytes());
        }
        if let Some(count) = self.loop_count {
            apply_loop(&mut bytes, count);
        }
        if let Err(error) = accept(&bytes) {
            return Err(self.failed(error));
        }
        let packets = self.packets.clone();
        self.output = bytes;
        self.status = Status::Finished;
        Ok(FinishedMux::complete(
            durations.to_vec(),
            index_place(),
            MuxMode::Progressive,
            true,
            packets,
            self.payload,
        ))
    }

    fn status(&self) -> MuxStatus {
        match self.status {
            Status::Building => MuxStatus::Building,
            Status::Failed(_) => MuxStatus::Failed,
            Status::Finished => MuxStatus::Finished,
        }
    }
}

fn image_codec(codec: &StreamCodec) -> Result<CodecId, MuxError> {
    match codec {
        StreamCodec::Known(id @ (CodecId::Jpeg | CodecId::Png | CodecId::Webp | CodecId::Gif)) => {
            Ok(*id)
        }
        StreamCodec::Known(_) => match UnknownCodec::new(b"other".to_vec()) {
            Ok(tag) => Err(MuxError::UnknownCodec(tag)),
            Err(_) => Err(MuxError::Incomplete),
        },
        StreamCodec::Unknown(tag) => Err(MuxError::UnknownCodec(tag.clone())),
    }
}

fn check_durations(
    stream: &NewStreamSlot,
    packets: &[Packet],
    durations: &[TrackDuration],
) -> Result<(), MuxError> {
    if durations.len() != 1 {
        return Err(MuxError::Incomplete);
    }
    let duration = durations[0];
    if duration.track != stream.id {
        return Err(MuxError::Incomplete);
    }
    if duration.timescale != stream.timescale {
        return Err(MuxError::TimescaleMismatch {
            packet: duration.timescale.ticks_per_second(),
            stream: stream.timescale.ticks_per_second(),
        });
    }
    if packets.is_empty() {
        return Err(MuxError::ShortPayload {
            track: stream.id.get(),
        });
    }
    if !stream.timescale.is_clock() {
        if duration.ticks != 0 || packets.len() != 1 {
            return Err(MuxError::DurationMismatch {
                track: stream.id.get(),
            });
        }
        if packets
            .iter()
            .any(|packet| !packet.timestamp().is_missing())
        {
            return Err(MuxError::Incomplete);
        }
        return Ok(());
    }
    let mut sum = 0_i64;
    for packet in packets {
        let Some(span) = packet.timestamp().duration() else {
            return Err(MuxError::MissingTimestamp);
        };
        sum = sum.checked_add(span).ok_or(MuxError::HostileSize {
            declared: u64::try_from(span).unwrap_or(u64::MAX),
            limit: u64::try_from(i64::MAX).unwrap_or(u64::MAX),
        })?;
    }
    if sum != duration.ticks {
        return Err(MuxError::DurationMismatch {
            track: stream.id.get(),
        });
    }
    Ok(())
}

fn accept(bytes: &[u8]) -> Result<(), MuxError> {
    let length = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if length > MAX_FILE_BYTES {
        return Err(MuxError::HostileSize {
            declared: length,
            limit: MAX_FILE_BYTES,
        });
    }
    let input = Input::random(Cursor::new(bytes.to_vec()), length, still_limits());
    match StillDemux::open(input) {
        Ok(_) => Ok(()),
        Err(cutlass_container::OpenError::HostileSize { declared, limit }) => {
            Err(MuxError::HostileSize { declared, limit })
        }
        Err(_) => Err(MuxError::Incomplete),
    }
}

fn apply_loop(bytes: &mut Vec<u8>, count: u16) {
    if bytes.len() >= 6 && (bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a")) {
        if !patch_netscape(bytes, count)
            && let Some(at) = gif_insert_at(bytes)
        {
            let block = netscape(count);
            bytes.splice(at..at, block);
        }
        return;
    }
    if bytes.starts_with(b"RIFF") {
        let _ = patch_webp_loop(bytes, count);
    }
}

fn netscape(count: u16) -> [u8; 19] {
    let mut block = *b"!\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00";
    block[16] = u8::try_from(count & 0xFF).unwrap_or(0);
    block[17] = u8::try_from((count >> 8) & 0xFF).unwrap_or(0);
    block
}

fn patch_netscape(bytes: &mut [u8], count: u16) -> bool {
    let needle = b"NETSCAPE2.0";
    let Some(at) = find_bytes(bytes, needle) else {
        return false;
    };
    let loop_at = at + needle.len();
    if bytes.get(loop_at) == Some(&3)
        && bytes.get(loop_at + 1) == Some(&1)
        && loop_at + 4 <= bytes.len()
    {
        bytes[loop_at + 2] = u8::try_from(count & 0xFF).unwrap_or(0);
        bytes[loop_at + 3] = u8::try_from((count >> 8) & 0xFF).unwrap_or(0);
        return true;
    }
    false
}

fn gif_insert_at(data: &[u8]) -> Option<usize> {
    if data.len() < 13 {
        return None;
    }
    let packed = data[10];
    let mut at = 13usize;
    if packed & 0x80 != 0 {
        let shift = u32::from(packed & 0x07) + 1;
        let colors = 3usize.saturating_mul(1usize << shift);
        at = at.saturating_add(colors);
    }
    if at > data.len() { None } else { Some(at) }
}

fn patch_webp_loop(data: &mut [u8], count: u16) -> bool {
    if data.len() < 12 || &data[..4] != b"RIFF" || &data[8..12] != b"WEBP" {
        return false;
    }
    let mut at = 12usize;
    while at + 8 <= data.len() {
        if &data[at..at + 4] == b"ANIM" {
            let size = u32::from_le_bytes([data[at + 4], data[at + 5], data[at + 6], data[at + 7]]);
            if size >= 6 && at + 14 <= data.len() {
                data[at + 12] = u8::try_from(count & 0xFF).unwrap_or(0);
                data[at + 13] = u8::try_from((count >> 8) & 0xFF).unwrap_or(0);
                return true;
            }
        }
        let size = usize::try_from(u32::from_le_bytes([
            data[at + 4],
            data[at + 5],
            data[at + 6],
            data[at + 7],
        ]))
        .unwrap_or(usize::MAX);
        let mut next = at.saturating_add(8).saturating_add(size);
        if size % 2 == 1 {
            next = next.saturating_add(1);
        }
        if next <= at || next > data.len() {
            return false;
        }
        at = next;
    }
    false
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

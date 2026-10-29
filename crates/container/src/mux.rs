//! The mux trait, and the completion gate a muxer has to pass.
//!
//! [`MuxDraft`] stores packets the caller already split. It does not write
//! a file format. [`finish`](Mux::finish) is the only call that returns
//! durations. [`FailedMux`] has no duration list: a failed mux does not
//! look complete.

use crate::error::MuxError;
use crate::limits::Limits;
use crate::stream::{Language, Stream, TrackId, Trim};
use crate::time::compare_ticks;
use cutlass_codec::{ContainerId, Packet, StreamCodec, StreamParams, Timescale};
use std::cmp::Ordering;

/// Where the family writes the seek index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexPlace {
    /// Before the media payload.
    Start,
    /// After the media payload.
    End,
    /// Sequential read only.
    Absent,
}

/// Progressive and fragmented output are different files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MuxMode {
    Progressive,
    Fragmented,
}

/// Whether a mux is still accepting packets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MuxStatus {
    Building,
    Failed,
    Finished,
}

impl MuxStatus {
    #[must_use]
    pub const fn is_complete(self) -> bool {
        matches!(self, Self::Finished)
    }
}

/// A stream to add. The track id is assigned by the muxer.
pub struct NewStream {
    pub params: StreamParams,
    pub timescale: Timescale,
    pub language: Option<Language>,
}

/// One track's duration in that track's file timescale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrackDuration {
    pub track: TrackId,
    pub timescale: Timescale,
    pub ticks: i64,
}

/// A mux that finished the index and the durations together.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinishedMux {
    durations: Vec<TrackDuration>,
    index: IndexPlace,
    mode: MuxMode,
    interleaved: bool,
    packets: Vec<Packet>,
    payload_bytes: u64,
}

impl FinishedMux {
    /// A family muxer returns this only after the bytes, the index, and the
    /// durations are all done. [`FailedMux`] is the other outcome.
    #[must_use]
    pub fn complete(
        durations: Vec<TrackDuration>,
        index: IndexPlace,
        mode: MuxMode,
        interleaved: bool,
        packets: Vec<Packet>,
        payload_bytes: u64,
    ) -> Self {
        Self {
            durations,
            index,
            mode,
            interleaved,
            packets,
            payload_bytes,
        }
    }

    #[must_use]
    pub const fn is_complete(&self) -> bool {
        true
    }

    #[must_use]
    pub fn durations(&self) -> &[TrackDuration] {
        &self.durations
    }

    #[must_use]
    pub const fn index(&self) -> IndexPlace {
        self.index
    }

    #[must_use]
    pub const fn mode(&self) -> MuxMode {
        self.mode
    }

    #[must_use]
    pub const fn interleaved(&self) -> bool {
        self.interleaved
    }

    #[must_use]
    pub fn packets(&self) -> &[Packet] {
        &self.packets
    }

    #[must_use]
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
}

/// A mux that stopped before completion.
///
/// There is no duration list on this value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FailedMux {
    payload_bytes: u64,
    reason: MuxError,
}

impl FailedMux {
    /// A family muxer returns this when finish does not produce a file.
    /// There is still no duration list.
    #[must_use]
    pub const fn incomplete(payload_bytes: u64, reason: MuxError) -> Self {
        Self {
            payload_bytes,
            reason,
        }
    }

    #[must_use]
    pub const fn is_complete(&self) -> bool {
        false
    }

    /// A failed mux does not publish durations.
    #[must_use]
    pub const fn durations(&self) -> Option<&[TrackDuration]> {
        None
    }

    #[must_use]
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    #[must_use]
    pub const fn reason(&self) -> &MuxError {
        &self.reason
    }
}

/// Packet list, completion, and interleave for a muxer.
///
/// This is not a container format. Families keep their own writers and use
/// these rules for the success value.
#[derive(Debug)]
pub struct MuxDraft {
    family: ContainerId,
    limits: Limits,
    mode: MuxMode,
    index: IndexPlace,
    streams: Vec<Stream>,
    packets: Vec<Packet>,
    status: MuxStatus,
    payload_bytes: u64,
    cancelled: bool,
    failure: Option<MuxError>,
}

impl MuxDraft {
    #[must_use]
    pub fn new(family: ContainerId, mode: MuxMode, index: IndexPlace, limits: Limits) -> Self {
        Self {
            family,
            limits,
            mode,
            index,
            streams: Vec::new(),
            packets: Vec::new(),
            status: MuxStatus::Building,
            payload_bytes: 0,
            cancelled: false,
            failure: None,
        }
    }

    /// Marks the mux failed. A later [`Mux::finish`] cannot return durations.
    ///
    /// A draft that already finished is left finished. The [`FinishedMux`]
    /// already returned stays complete.
    pub fn abort(&mut self, reason: MuxError) {
        if self.status == MuxStatus::Finished {
            return;
        }
        self.failure = Some(reason);
        self.status = MuxStatus::Failed;
    }

    /// Packets accepted so far. `None` once the mux has failed or finished,
    /// so a failure is not a packet list to treat as a file.
    #[must_use]
    pub fn packets(&self) -> Option<&[Packet]> {
        match self.status {
            MuxStatus::Building => Some(&self.packets),
            MuxStatus::Failed | MuxStatus::Finished => None,
        }
    }

    fn fail(&mut self, reason: MuxError) -> MuxError {
        self.status = MuxStatus::Failed;
        self.failure = Some(reason.clone());
        reason
    }

    fn building(&self) -> Result<(), MuxError> {
        if self.cancelled {
            return Err(MuxError::Cancelled);
        }
        match self.status {
            MuxStatus::Building => Ok(()),
            MuxStatus::Failed => Err(self.failure.clone().unwrap_or(MuxError::NotBuilding)),
            MuxStatus::Finished => Err(MuxError::AlreadyFinished),
        }
    }

    fn failed_mux(&self, reason: MuxError) -> FailedMux {
        FailedMux {
            payload_bytes: self.payload_bytes,
            reason,
        }
    }
}

impl Mux for MuxDraft {
    fn family(&self) -> ContainerId {
        self.family
    }

    fn add_stream(&mut self, stream: NewStream) -> Result<TrackId, MuxError> {
        self.building()?;
        if let StreamCodec::Unknown(unknown) = stream.params.codec() {
            return Err(self.fail(MuxError::UnknownCodec(unknown.clone())));
        }
        let count = u32::try_from(self.streams.len()).unwrap_or(u32::MAX);
        if count >= self.limits.max_tracks {
            return Err(self.fail(MuxError::TooManyTracks {
                count,
                limit: self.limits.max_tracks,
            }));
        }
        let extra = u64::try_from(stream.params.extradata().len()).unwrap_or(u64::MAX);
        if extra > self.limits.max_extradata_bytes {
            return Err(self.fail(MuxError::HostileSize {
                declared: extra,
                limit: self.limits.max_extradata_bytes,
            }));
        }
        let id = TrackId::new(count);
        self.streams.push(Stream::new(
            id,
            stream.params,
            stream.timescale,
            stream.language,
            Vec::new(),
            Trim::default(),
        ));
        Ok(id)
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<(), MuxError> {
        self.building()?;
        let stream_scale = {
            let Some(stream) = self
                .streams
                .iter()
                .find(|stream| stream.id().get() == packet.stream())
            else {
                return Err(self.fail(MuxError::UnknownTrack {
                    id: packet.stream(),
                }));
            };
            stream.timescale()
        };
        if packet.timescale() != stream_scale {
            return Err(self.fail(MuxError::TimescaleMismatch {
                packet: packet.timescale().ticks_per_second(),
                stream: stream_scale.ticks_per_second(),
            }));
        }
        let size = u64::try_from(packet.bytes().len()).unwrap_or(u64::MAX);
        if size > self.limits.max_packet_bytes {
            return Err(self.fail(MuxError::HostileSize {
                declared: size,
                limit: self.limits.max_packet_bytes,
            }));
        }
        let Some(total) = self.payload_bytes.checked_add(size) else {
            return Err(self.fail(MuxError::HostileSize {
                declared: size,
                limit: u64::MAX - self.payload_bytes,
            }));
        };
        self.payload_bytes = total;
        self.packets.push(packet.clone());
        Ok(())
    }

    fn interleave(&mut self) -> Result<(), MuxError> {
        self.building()?;
        for packet in &self.packets {
            if !packet.timescale().is_clock()
                || packet.timestamp().presentation_or_decode().is_none()
            {
                return Err(MuxError::MissingTimestamp);
            }
        }
        self.packets.sort_by(order_packets);
        Ok(())
    }

    fn cancel(&mut self) {
        self.cancelled = true;
        if self.status == MuxStatus::Building {
            self.failure = Some(MuxError::Cancelled);
            self.status = MuxStatus::Failed;
        }
    }

    fn finish(&mut self, durations: &[TrackDuration]) -> Result<FinishedMux, FailedMux> {
        if self.cancelled && self.status != MuxStatus::Finished {
            let reason = self.fail(MuxError::Cancelled);
            return Err(self.failed_mux(reason));
        }
        match self.status {
            MuxStatus::Finished => {
                return Err(self.failed_mux(MuxError::AlreadyFinished));
            }
            MuxStatus::Failed => {
                let reason = self.failure.clone().unwrap_or(MuxError::Incomplete);
                return Err(self.failed_mux(reason));
            }
            MuxStatus::Building => {}
        }
        let durations = match self.check_durations(durations) {
            Ok(durations) => durations,
            Err(reason) => {
                let reason = self.fail(reason);
                return Err(self.failed_mux(reason));
            }
        };
        let interleaved = is_interleaved(&self.packets);
        let packets = std::mem::take(&mut self.packets);
        self.status = MuxStatus::Finished;
        Ok(FinishedMux {
            durations,
            index: self.index,
            mode: self.mode,
            interleaved,
            packets,
            payload_bytes: self.payload_bytes,
        })
    }

    fn status(&self) -> MuxStatus {
        self.status
    }
}

impl MuxDraft {
    fn check_durations(&self, durations: &[TrackDuration]) -> Result<Vec<TrackDuration>, MuxError> {
        if durations.len() != self.streams.len() {
            return Err(MuxError::Incomplete);
        }
        let mut checked = Vec::with_capacity(self.streams.len());
        for stream in &self.streams {
            let Some(duration) = durations.iter().find(|item| item.track == stream.id()) else {
                return Err(MuxError::Incomplete);
            };
            if duration.ticks < 0 {
                return Err(MuxError::NegativeDuration);
            }
            if duration.timescale != stream.timescale() {
                return Err(MuxError::TimescaleMismatch {
                    packet: duration.timescale.ticks_per_second(),
                    stream: stream.timescale().ticks_per_second(),
                });
            }
            let packets: Vec<&Packet> = self
                .packets
                .iter()
                .filter(|packet| packet.stream() == stream.id().get())
                .collect();
            if duration.ticks == 0 && packets.is_empty() {
                checked.push(*duration);
                continue;
            }
            if packets.is_empty() {
                return Err(MuxError::ShortPayload {
                    track: stream.id().get(),
                });
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
                    track: stream.id().get(),
                });
            }
            checked.push(*duration);
        }
        Ok(checked)
    }
}

fn order_packets(left: &Packet, right: &Packet) -> Ordering {
    let Some(left_time) = left.timestamp().presentation_or_decode() else {
        return Ordering::Equal;
    };
    let Some(right_time) = right.timestamp().presentation_or_decode() else {
        return Ordering::Equal;
    };
    compare_ticks(left_time, left.timescale(), right_time, right.timescale())
        .unwrap_or(Ordering::Equal)
}

fn is_interleaved(packets: &[Packet]) -> bool {
    let mut tracks = Vec::new();
    for packet in packets {
        if !tracks.contains(&packet.stream()) {
            tracks.push(packet.stream());
        }
    }
    if tracks.len() < 2 {
        return true;
    }
    for (left_at, left) in tracks.iter().enumerate() {
        for right in tracks.iter().skip(left_at + 1) {
            let Some(first_left) = packets.iter().position(|packet| packet.stream() == *left)
            else {
                return false;
            };
            let Some(first_right) = packets.iter().position(|packet| packet.stream() == *right)
            else {
                return false;
            };
            let Some(last_left) = packets.iter().rposition(|packet| packet.stream() == *left)
            else {
                return false;
            };
            let Some(last_right) = packets.iter().rposition(|packet| packet.stream() == *right)
            else {
                return false;
            };
            if first_left >= last_right || first_right >= last_left {
                return false;
            }
        }
    }
    true
}

/// Writes packets and finishes a file.
///
/// `finish` takes `&mut self` so the trait is object-safe. A second finish
/// is [`FailedMux`] and does not replace a [`FinishedMux`] already returned.
pub trait Mux {
    fn family(&self) -> ContainerId;
    fn add_stream(&mut self, stream: NewStream) -> Result<TrackId, MuxError>;
    fn write_packet(&mut self, packet: &Packet) -> Result<(), MuxError>;
    /// Orders packets by presentation time in the file timescale.
    ///
    /// A packet with no presentation time and no decode time is left where
    /// it is, and this call returns [`MuxError::MissingTimestamp`] without
    /// inventing a time. The mux stays building.
    fn interleave(&mut self) -> Result<(), MuxError>;
    fn cancel(&mut self);
    fn finish(&mut self, durations: &[TrackDuration]) -> Result<FinishedMux, FailedMux>;
    fn status(&self) -> MuxStatus;
}

#[cfg(test)]
mod tests {
    use super::{FailedMux, FinishedMux, IndexPlace, MuxError, MuxMode, TrackDuration};
    use crate::stream::TrackId;
    use cutlass_codec::{Packet, Timescale, Timestamp};

    #[test]
    fn a_family_can_finish_or_fail_without_a_draft() {
        let packet = Packet::try_new(
            0,
            b"jpeg".to_vec(),
            Timestamp::missing(),
            Timescale::new(0),
            true,
        )
        .expect("packet");
        let done = FinishedMux::complete(
            vec![TrackDuration {
                track: TrackId::new(0),
                timescale: Timescale::new(0),
                ticks: 0,
            }],
            IndexPlace::Absent,
            MuxMode::Progressive,
            true,
            vec![packet],
            4,
        );
        assert!(done.is_complete());
        assert_eq!(done.payload_bytes(), 4);
        assert_eq!(done.durations()[0].ticks, 0);
        assert!(done.durations()[0].timescale.ticks_per_second() == 0);

        let failed = FailedMux::incomplete(0, MuxError::Incomplete);
        assert!(!failed.is_complete());
        assert!(failed.durations().is_none());
    }
}

//! A packet list that implements [`crate::Demux`].
//!
//! The caller already split the access units. This type owns the input,
//! returns those units in the order given, and does not interpret the
//! bytes as a file format.

use crate::demux::Demux;
use crate::error::{OpenError, ReadError, SeekError};
use crate::input::{Access, Input};
use crate::limits::Limits;
use crate::probe::HeaderVerdict;
use crate::seek::{
    IndexEntry, Landing, ScanEntry, SeekLanding, SeekTo, ensure_index_sorted, seek_by_scan,
    seek_with_index,
};
use crate::stream::{Gap, Stream};
use cutlass_codec::{ContainerId, Packet, PacketError, Timescale, Timestamp};
use std::io::{Read, Seek};

/// Timing for one access unit whose bytes live in the owned input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PacketPlan {
    pub stream: u32,
    pub timestamp: Timestamp,
    pub timescale: Timescale,
    pub keyframe: bool,
    pub size: u64,
}

/// What [`ListDemux::open`] needs. The input bytes are the header followed
/// by each plan's payload, in order. The header bytes are consumed and not
/// parsed.
pub struct ListRequest<R> {
    pub input: Input<R>,
    pub family: ContainerId,
    pub header_bytes: usize,
    pub verdict: HeaderVerdict,
    pub streams: Vec<Stream>,
    pub packets: Vec<PacketPlan>,
    pub index: Vec<IndexEntry>,
    pub gaps: Vec<Gap>,
}

/// An open that did not publish streams. The input comes back to the caller.
#[derive(Debug)]
pub struct OpenFailure<R> {
    pub input: Input<R>,
    pub error: OpenError,
}

/// Demuxer over a caller-supplied packet list.
#[derive(Debug)]
pub struct ListDemux<R> {
    input: Input<R>,
    family: ContainerId,
    header_bytes: u64,
    streams: Vec<Stream>,
    plans: Vec<PacketPlan>,
    index: Vec<IndexEntry>,
    gaps: Vec<Gap>,
    cursor: usize,
    cancelled: bool,
    failure: Option<ReadError>,
}

impl<R: Read + Seek> ListDemux<R> {
    /// Opens when the header window is present and the verdict is a match.
    ///
    /// A short file, an unrecognized window, an unsupported brand, a hostile
    /// size, or a duplicate track returns the input and publishes no streams.
    pub fn open(mut request: ListRequest<R>) -> Result<Self, OpenFailure<R>> {
        let limits = request.input.limits();
        let header = u64::try_from(request.header_bytes).unwrap_or(u64::MAX);
        if request.header_bytes == 0 {
            return Err(fail(
                request.input,
                OpenError::Unrecognized {
                    family: request.family,
                },
            ));
        }
        if header > limits.max_header_bytes {
            return Err(fail(
                request.input,
                OpenError::HostileSize {
                    declared: header,
                    limit: limits.max_header_bytes,
                },
            ));
        }
        if let Some(available) = request.input.length()
            && available < header
        {
            return Err(fail(
                request.input,
                OpenError::Truncated {
                    expected: header,
                    got: available,
                },
            ));
        }
        if let Some(error) = verdict_error(request.family, request.verdict) {
            return Err(fail(request.input, error));
        }
        if let Some(error) = stream_error(&request.streams, &request.index, &limits) {
            return Err(fail(request.input, error));
        }
        if let Err(error) = request.input.read_limited(header, limits.max_header_bytes) {
            return Err(fail(request.input, OpenError::from_header(error)));
        }
        Ok(Self {
            input: request.input,
            family: request.family,
            header_bytes: header,
            streams: request.streams,
            plans: request.packets,
            index: request.index,
            gaps: request.gaps,
            cursor: 0,
            cancelled: false,
            failure: None,
        })
    }

    fn offset_of(&self, ordinal: usize) -> Result<u64, ReadError> {
        let mut offset = self.header_bytes;
        for plan in self.plans.iter().take(ordinal) {
            offset = offset
                .checked_add(plan.size)
                .ok_or(ReadError::HostileSize {
                    declared: plan.size,
                    limit: u64::MAX,
                })?;
        }
        Ok(offset)
    }

    fn fail(&mut self, error: ReadError) -> ReadError {
        self.failure = Some(error.clone());
        error
    }

    fn apply(&mut self, landing: SeekLanding) -> Result<SeekLanding, ReadError> {
        match landing.landing {
            Landing::End => {
                if let Some(length) = self.input.length() {
                    self.input.seek(length).map_err(ReadError::from_packet)?;
                }
                self.cursor = self.plans.len();
                Ok(landing)
            }
            Landing::Packet {
                ordinal,
                time,
                keyframe,
                preroll,
                ..
            } => {
                let ordinal_usize = usize::try_from(ordinal).map_err(|_| ReadError::Source {
                    message: "seek ordinal does not fit".to_owned(),
                })?;
                if ordinal_usize > self.plans.len() {
                    return Err(ReadError::Source {
                        message: "seek ordinal is past the packet list".to_owned(),
                    });
                }
                let offset = self.offset_of(ordinal_usize)?;
                self.input.seek(offset).map_err(ReadError::from_packet)?;
                self.cursor = ordinal_usize;
                Ok(SeekLanding {
                    landing: Landing::Packet {
                        ordinal,
                        time,
                        keyframe,
                        preroll,
                        byte_position: offset,
                    },
                    index_used: landing.index_used,
                    payload_scans: landing.payload_scans,
                    entries_visited: landing.entries_visited,
                })
            }
        }
    }
}

impl<R: Read + Seek> Demux for ListDemux<R> {
    fn family(&self) -> ContainerId {
        self.family
    }

    fn streams(&self) -> &[Stream] {
        &self.streams
    }

    fn gaps(&self) -> &[Gap] {
        &self.gaps
    }

    fn read_packet(&mut self) -> Result<Option<Packet>, ReadError> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        if self.cursor >= self.plans.len() {
            return Ok(None);
        }
        let plan = self.plans[self.cursor].clone();
        let Some(stream_scale) = self
            .streams
            .iter()
            .find(|stream| stream.id().get() == plan.stream)
            .map(|stream| stream.timescale())
        else {
            return Err(self.fail(ReadError::UnknownTrack { id: plan.stream }));
        };
        if plan.timescale != stream_scale {
            return Err(self.fail(ReadError::TimescaleMismatch {
                packet: plan.timescale.ticks_per_second(),
                stream: stream_scale.ticks_per_second(),
            }));
        }
        let bytes = match self.input.read_declared(plan.size) {
            Ok(bytes) => bytes,
            Err(error) => return Err(self.fail(ReadError::from_packet(error))),
        };
        let packet = match Packet::try_new(
            plan.stream,
            bytes,
            plan.timestamp,
            plan.timescale,
            plan.keyframe,
        ) {
            Ok(packet) => packet,
            Err(PacketError::Clockless) => {
                return Err(self.fail(ReadError::TimescaleMismatch {
                    packet: 0,
                    stream: plan.timescale.ticks_per_second(),
                }));
            }
            Err(PacketError::NegativeDuration) => {
                return Err(self.fail(ReadError::Source {
                    message: "packet duration is negative".to_owned(),
                }));
            }
        };
        self.cursor += 1;
        Ok(Some(packet))
    }

    fn seek(&mut self, to: SeekTo) -> Result<SeekLanding, ReadError> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        if !self
            .streams
            .iter()
            .any(|stream| stream.id().get() == to.track)
        {
            return Err(ReadError::Seek(SeekError::UnknownTrack));
        }
        if self.input.access() == Access::Forward {
            return Err(ReadError::Seek(SeekError::ForwardOnly));
        }
        if to.end.is_some_and(|end| to.time >= end) {
            return self.apply(SeekLanding {
                landing: Landing::End,
                index_used: !self.index.is_empty(),
                payload_scans: 0,
                entries_visited: 0,
            });
        }
        let entries: Vec<IndexEntry> = self
            .index
            .iter()
            .copied()
            .filter(|entry| entry.track == to.track)
            .collect();
        let landing = if entries.is_empty() {
            let scan = scan_entries(&self.plans);
            seek_by_scan(&scan, to).map_err(ReadError::Seek)?
        } else {
            ensure_index_sorted(&entries, to.track).map_err(ReadError::Seek)?;
            seek_with_index(&entries, to).map_err(ReadError::Seek)?
        };
        self.apply(landing)
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }

    fn consumed(&self) -> u64 {
        self.input.consumed()
    }
}

impl<R: Read + Seek> ListDemux<R> {
    /// Position in the owned input. A failed seek leaves this unchanged.
    #[must_use]
    pub fn position(&self) -> u64 {
        self.input.position()
    }
}

fn fail<R>(input: Input<R>, error: OpenError) -> OpenFailure<R> {
    OpenFailure { input, error }
}

fn verdict_error(family: ContainerId, verdict: HeaderVerdict) -> Option<OpenError> {
    match verdict {
        HeaderVerdict::Match => None,
        HeaderVerdict::Unrecognized => Some(OpenError::Unrecognized { family }),
        HeaderVerdict::UnsupportedBrand => Some(OpenError::UnsupportedBrand { family }),
    }
}

fn stream_error(streams: &[Stream], index: &[IndexEntry], limits: &Limits) -> Option<OpenError> {
    let count = u64::try_from(streams.len()).unwrap_or(u64::MAX);
    if count > u64::from(limits.max_tracks) {
        return Some(OpenError::HostileSize {
            declared: count,
            limit: u64::from(limits.max_tracks),
        });
    }
    for (at, stream) in streams.iter().enumerate() {
        if streams[..at].iter().any(|other| other.id() == stream.id()) {
            return Some(OpenError::DuplicateTrack {
                id: stream.id().get(),
            });
        }
        if !stream.within_limits(limits) {
            let extra = u64::try_from(stream.params().extradata().len()).unwrap_or(u64::MAX);
            if extra > limits.max_extradata_bytes {
                return Some(OpenError::HostileSize {
                    declared: extra,
                    limit: limits.max_extradata_bytes,
                });
            }
            let edits = u64::try_from(stream.edits().len()).unwrap_or(u64::MAX);
            return Some(OpenError::HostileSize {
                declared: edits,
                limit: u64::from(limits.max_edits),
            });
        }
    }
    let index_len = u64::try_from(index.len()).unwrap_or(u64::MAX);
    if index_len > u64::from(limits.max_index_entries) {
        return Some(OpenError::HostileSize {
            declared: index_len,
            limit: u64::from(limits.max_index_entries),
        });
    }
    None
}

fn scan_entries(plans: &[PacketPlan]) -> Vec<ScanEntry> {
    plans
        .iter()
        .enumerate()
        .map(|(ordinal, plan)| ScanEntry {
            stream: plan.stream,
            presentation: plan.timestamp.presentation_or_decode(),
            keyframe: plan.keyframe,
            ordinal: u32::try_from(ordinal).unwrap_or(u32::MAX),
            byte_position: 0,
        })
        .collect()
}

//! Demux and mux traits.
//!
//! Timestamps inside a container stay in the file timescale. A demuxer yields
//! one [`Packet`] at a time. A muxer writes those packets back. This crate
//! does not convert ticks into microseconds.

#![forbid(unsafe_code)]

use std::fmt;
use std::io::ErrorKind;

use cutlass_codec::{CodecParams, ContainerId, Packet};

/// How many leading bytes accept or refuse a family.
///
/// `NeedMore::minimum` is the prefix length, in bytes, that may be enough
/// for a later call. It is not a promise that the file is a member.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Probe {
    Matched,
    NeedMore { minimum: u64 },
    Reject,
}

/// One elementary stream.
///
/// `timescale` is the stream clock in ticks per second. `trim` is present
/// only when the file carries encoder delay and trailing padding. The
/// container does not drop those samples. `reservoir_preroll` is how many
/// earlier packets a decoder may still need before a packet is fully
/// determined. Those packets are not returned again after a seek.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stream {
    pub index: u32,
    pub params: CodecParams,
    pub timescale: u32,
    pub trim: Option<Trim>,
    pub reservoir_preroll: u32,
}

/// Encoder delay and trailing padding, in samples at the stream timescale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Trim {
    pub encoder_delay: u32,
    pub trailing_padding: u32,
}

/// Where a seek landed, in the file timescale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeekLanding {
    /// Presentation time of the next packet. Equal to the duration when the
    /// seek is past the last packet, in which case the next read is the end.
    pub pts: i64,
    /// True when a table or a constant schedule chose the byte. False when
    /// the landing is the start or the end and no index was required.
    pub indexed: bool,
    /// Frames the codec may still need from before `pts`. Those frames are
    /// not returned.
    pub preroll_frames: u32,
}

/// Failure while framing a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContainerError {
    /// The byte stream ended inside a header or a frame.
    Truncated { what: &'static str },
    /// The bytes are present and are not a legal value.
    Malformed { what: &'static str },
    /// A legal value this build does not carry.
    Unsupported { what: &'static str },
    /// A length or a search passed a documented limit.
    Limit { what: &'static str },
    /// The call does not match the demux or mux state.
    InvalidApi { what: &'static str },
    /// The reader or writer failed. The kind is the I/O class, not a path.
    Io { kind: ErrorKind },
    /// Sequential reads still work. This file has no seek index.
    SeekUnavailable,
    /// The caller cancelled the demux or the mux.
    Cancelled,
    /// The file matches the family and contains no packets.
    NoFrames,
}

impl fmt::Display for ContainerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { what } => write!(f, "truncated {what}"),
            Self::Malformed { what } => write!(f, "malformed {what}"),
            Self::Unsupported { what } => write!(f, "unsupported {what}"),
            Self::Limit { what } => write!(f, "limit exceeded: {what}"),
            Self::InvalidApi { what } => write!(f, "invalid call: {what}"),
            Self::Io { kind } => write!(f, "io error: {kind:?}"),
            Self::SeekUnavailable => write!(f, "seek index is not in this file"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::NoFrames => write!(f, "no frames"),
        }
    }
}

impl std::error::Error for ContainerError {}

impl From<std::io::Error> for ContainerError {
    fn from(err: std::io::Error) -> Self {
        if err.kind() == ErrorKind::UnexpectedEof {
            Self::Truncated { what: "read" }
        } else {
            Self::Io { kind: err.kind() }
        }
    }
}

/// Reads a file into packets in decode order.
///
/// Times are ticks of each stream's `timescale`. A seek does not return a
/// packet that ended at or before the requested time.
pub trait Demux {
    fn container_id(&self) -> ContainerId;

    fn streams(&self) -> &[Stream];

    /// Duration of the primary stream, in that stream's timescale.
    /// Absent when the file does not say, and a scan was not done.
    fn duration(&self) -> Option<i64>;

    /// True when [`duration`](Self::duration) comes from a header count
    /// rather than a bitrate estimate.
    fn duration_exact(&self) -> bool;

    fn next_packet(&mut self) -> Result<Option<Packet>, ContainerError>;

    fn seek(&mut self, time: i64) -> Result<SeekLanding, ContainerError>;

    fn cancel(&mut self);
}

/// Writes packets into a file.
///
/// `finish` completes the index and the duration. A second `finish`, or a
/// `write_packet` after `finish`, is [`ContainerError::InvalidApi`]. A
/// finish that fails does not publish a complete duration over a short
/// payload.
pub trait Mux {
    fn container_id(&self) -> ContainerId;

    fn add_stream(&mut self, params: CodecParams) -> Result<u32, ContainerError>;

    fn write_packet(&mut self, packet: &Packet) -> Result<(), ContainerError>;

    fn finish(&mut self) -> Result<(), ContainerError>;

    fn cancel(&mut self);
}

#[cfg(test)]
mod tests {
    use super::{ContainerError, Demux, Mux, SeekLanding, Stream, Trim};
    use cutlass_codec::{CodecId, CodecParams, ContainerId, Packet};

    struct Mem {
        streams: Vec<Stream>,
        packets: Vec<Packet>,
        cursor: usize,
        done: bool,
        cancelled: bool,
    }

    impl Demux for Mem {
        fn container_id(&self) -> ContainerId {
            ContainerId::Mp3
        }

        fn streams(&self) -> &[Stream] {
            &self.streams
        }

        fn duration(&self) -> Option<i64> {
            self.packets.last().map(|p| p.pts + p.duration)
        }

        fn duration_exact(&self) -> bool {
            true
        }

        fn next_packet(&mut self) -> Result<Option<Packet>, ContainerError> {
            if self.cancelled {
                return Err(ContainerError::Cancelled);
            }
            let Some(packet) = self.packets.get(self.cursor) else {
                return Ok(None);
            };
            self.cursor += 1;
            Ok(Some(packet.clone()))
        }

        fn seek(&mut self, time: i64) -> Result<SeekLanding, ContainerError> {
            if self.cancelled {
                return Err(ContainerError::Cancelled);
            }
            let time = time.max(0);
            self.cursor = self
                .packets
                .iter()
                .position(|p| p.pts + p.duration > time)
                .unwrap_or(self.packets.len());
            let pts = self.packets.get(self.cursor).map(|p| p.pts).unwrap_or(time);
            Ok(SeekLanding {
                pts,
                indexed: true,
                preroll_frames: 0,
            })
        }

        fn cancel(&mut self) {
            self.cancelled = true;
        }
    }

    impl Mux for Mem {
        fn container_id(&self) -> ContainerId {
            ContainerId::Mp3
        }

        fn add_stream(&mut self, params: CodecParams) -> Result<u32, ContainerError> {
            if self.done {
                return Err(ContainerError::InvalidApi {
                    what: "add_stream after finish",
                });
            }
            let index = u32::try_from(self.streams.len())
                .map_err(|_| ContainerError::Limit { what: "streams" })?;
            self.streams.push(Stream {
                index,
                timescale: params.sample_rate,
                trim: None,
                reservoir_preroll: 0,
                params,
            });
            Ok(index)
        }

        fn write_packet(&mut self, packet: &Packet) -> Result<(), ContainerError> {
            if self.cancelled {
                return Err(ContainerError::Cancelled);
            }
            if self.done {
                return Err(ContainerError::InvalidApi {
                    what: "write after finish",
                });
            }
            self.packets.push(packet.clone());
            Ok(())
        }

        fn finish(&mut self) -> Result<(), ContainerError> {
            if self.done {
                return Err(ContainerError::InvalidApi {
                    what: "second finish",
                });
            }
            self.done = true;
            Ok(())
        }

        fn cancel(&mut self) {
            self.cancelled = true;
        }
    }

    #[test]
    fn mux_then_demux_returns_the_same_packet_bytes_and_ticks() {
        let mut mux = Mem {
            streams: Vec::new(),
            packets: Vec::new(),
            cursor: 0,
            done: false,
            cancelled: false,
        };
        let params = CodecParams::audio(CodecId::Mp3, 44_100, 2);
        mux.add_stream(params).expect("stream");
        mux.write_packet(&Packet::audio(vec![9, 8, 7], 0, 1152, 44_100))
            .expect("write");
        mux.finish().expect("finish");
        assert!(mux.finish().is_err());

        let mut demux = Mem {
            streams: mux.streams.clone(),
            packets: mux.packets.clone(),
            cursor: 0,
            done: false,
            cancelled: false,
        };
        let packet = demux.next_packet().expect("packet").expect("some");
        assert_eq!(packet.data, vec![9, 8, 7]);
        assert_eq!(packet.pts, 0);
        assert_eq!(packet.duration, 1152);
        assert_eq!(packet.timescale, 44_100);
        assert!(demux.next_packet().expect("end").is_none());
        demux.seek(0).expect("seek");
        assert_eq!(demux.next_packet().expect("again").expect("some").pts, 0);
    }

    #[test]
    fn seek_does_not_return_a_packet_that_already_ended() {
        let mut demux = Mem {
            streams: Vec::new(),
            packets: vec![
                Packet::audio(vec![1], 0, 1152, 44_100),
                Packet::audio(vec![2], 1152, 1152, 44_100),
            ],
            cursor: 0,
            done: false,
            cancelled: false,
        };
        let landing = demux.seek(1152).expect("seek");
        assert_eq!(landing.pts, 1152);
        assert_eq!(
            demux.next_packet().expect("packet").expect("some").data,
            vec![2]
        );
    }

    #[test]
    fn cancel_refuses_the_next_read() {
        let mut demux = Mem {
            streams: Vec::new(),
            packets: vec![Packet::audio(vec![1], 0, 1152, 44_100)],
            cursor: 0,
            done: false,
            cancelled: false,
        };
        Demux::cancel(&mut demux);
        assert_eq!(demux.next_packet(), Err(ContainerError::Cancelled));
    }

    #[test]
    fn trim_is_metadata_on_the_stream() {
        let trim = Trim {
            encoder_delay: 576,
            trailing_padding: 100,
        };
        assert_eq!(trim.encoder_delay, 576);
    }
}

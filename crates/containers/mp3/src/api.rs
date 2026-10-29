//! Types this framer was written against.
//!
//! The shared [`cutlass_container::Demux`] trait is a different contract.
//! `Mp3Demux` implements both: this module keeps the frame parser stable,
//! and `shared` adapts each call onto the workspace traits.

use std::fmt;
use std::io::ErrorKind;

pub use cutlass_codec::{CodecId, ContainerId};

/// One Layer III frame, with presentation time in the file timescale.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packet {
    pub stream: u32,
    pub data: Vec<u8>,
    pub pts: i64,
    pub duration: i64,
    pub timescale: u32,
    pub keyframe: bool,
}

impl Packet {
    #[must_use]
    pub fn audio(data: Vec<u8>, pts: i64, duration: i64, timescale: u32) -> Self {
        Self {
            stream: 0,
            data,
            pts,
            duration,
            timescale,
            keyframe: true,
        }
    }
}

/// Parameters the MP3 muxer stores on its one stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodecParams {
    pub codec: CodecId,
    pub extra: Vec<u8>,
    pub sample_rate: u32,
    pub channels: u16,
    pub bitrate: u32,
    pub width: u32,
    pub height: u32,
    pub encoder_delay: u32,
    pub trailing_padding: u32,
}

impl CodecParams {
    #[must_use]
    pub fn audio(codec: CodecId, sample_rate: u32, channels: u16) -> Self {
        Self {
            codec,
            extra: Vec::new(),
            sample_rate,
            channels,
            bitrate: 0,
            width: 0,
            height: 0,
            encoder_delay: 0,
            trailing_padding: 0,
        }
    }
}

/// Encoder delay and trailing padding, in samples at the stream timescale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Trim {
    pub encoder_delay: u32,
    pub trailing_padding: u32,
}

/// One elementary stream in an MP3 file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stream {
    pub index: u32,
    pub params: CodecParams,
    pub timescale: u32,
    pub trim: Option<Trim>,
    pub reservoir_preroll: u32,
}

/// Where a seek landed, in the file timescale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeekLanding {
    pub pts: i64,
    pub indexed: bool,
    pub preroll_frames: u32,
}

/// How many leading bytes accept or refuse this family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Probe {
    Matched,
    NeedMore { minimum: u64 },
    Reject,
}

/// Failure while framing an MP3 file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContainerError {
    Truncated { what: &'static str },
    Malformed { what: &'static str },
    Unsupported { what: &'static str },
    Limit { what: &'static str },
    InvalidApi { what: &'static str },
    Io { kind: ErrorKind },
    SeekUnavailable,
    Cancelled,
    NoFrames,
}

impl fmt::Display for ContainerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { what } => write!(formatter, "truncated {what}"),
            Self::Malformed { what } => write!(formatter, "malformed {what}"),
            Self::Unsupported { what } => write!(formatter, "unsupported {what}"),
            Self::Limit { what } => write!(formatter, "limit exceeded: {what}"),
            Self::InvalidApi { what } => write!(formatter, "invalid call: {what}"),
            Self::Io { kind } => write!(formatter, "io error: {kind:?}"),
            Self::SeekUnavailable => write!(formatter, "seek index is not in this file"),
            Self::Cancelled => write!(formatter, "cancelled"),
            Self::NoFrames => write!(formatter, "no frames"),
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

/// Reads an MP3 file into frames.
pub trait Demux {
    fn container_id(&self) -> ContainerId;

    fn streams(&self) -> &[Stream];

    fn duration(&self) -> Option<i64>;

    fn duration_exact(&self) -> bool;

    fn next_packet(&mut self) -> Result<Option<Packet>, ContainerError>;

    fn seek(&mut self, time: i64) -> Result<SeekLanding, ContainerError>;

    fn cancel(&mut self);
}

/// Writes Layer III frames into an MP3 file.
pub trait Mux {
    fn container_id(&self) -> ContainerId;

    fn add_stream(&mut self, params: CodecParams) -> Result<u32, ContainerError>;

    fn write_packet(&mut self, packet: &Packet) -> Result<(), ContainerError>;

    fn finish(&mut self) -> Result<(), ContainerError>;

    fn cancel(&mut self);
}

//! Open failures and read failures are different types.
//!
//! A failed open publishes no streams. A failed read publishes no packet
//! for that call. Streams from the open that succeeded stay usable, and
//! packets already returned stay usable.

use cutlass_codec::{ContainerId, UnknownCodec};
use std::fmt;

/// A byte source could not fill a request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceError {
    /// A known-length input ended inside the requested span.
    Truncated { expected: u64, got: u64 },
    /// A forward input ended before the span was full. More bytes may arrive.
    ShortRead { expected: u64, got: u64 },
    /// `declared` is larger than `limit`. Nothing of that size was allocated.
    HostileSize { declared: u64, limit: u64 },
    /// The input is forward-only and cannot reposition.
    ForwardOnly,
    /// A seek asked for a position past the known length. The position did not change.
    OutOfRange { position: u64, length: u64 },
    /// The source returned an I/O failure.
    Failed { message: String },
}

impl fmt::Display for SourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { expected, got } => {
                write!(formatter, "truncated read: expected {expected}, got {got}")
            }
            Self::ShortRead { expected, got } => {
                write!(formatter, "short read: expected {expected}, got {got}")
            }
            Self::HostileSize { declared, limit } => {
                write!(formatter, "declared size {declared} exceeds {limit}")
            }
            Self::ForwardOnly => formatter.write_str("forward input cannot seek"),
            Self::OutOfRange { position, length } => {
                write!(formatter, "position {position} is past length {length}")
            }
            Self::Failed { message } => write!(formatter, "input failed: {message}"),
        }
    }
}

impl std::error::Error for SourceError {}

/// The file did not become a demuxer.
///
/// No stream list and no duration are published with this error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpenError {
    /// The header window was long enough and it is not this family.
    Unrecognized { family: ContainerId },
    /// The input ended inside the header.
    Truncated { expected: u64, got: u64 },
    /// A size in the header is larger than the remaining input or a limit.
    HostileSize { declared: u64, limit: u64 },
    /// The family matches and the brand is one this build will not open.
    UnsupportedBrand { family: ContainerId },
    /// Two streams claim the same track id.
    DuplicateTrack { id: u32 },
    /// The source failed while the header was being accepted.
    Source { message: String },
}

impl OpenError {
    /// Maps a header read into an open failure.
    ///
    /// A short header on a forward input is truncation: the file ended.
    /// The same short read during a packet is [`ReadError::ShortRead`].
    #[must_use]
    pub fn from_header(error: SourceError) -> Self {
        match error {
            SourceError::Truncated { expected, got } | SourceError::ShortRead { expected, got } => {
                Self::Truncated { expected, got }
            }
            SourceError::HostileSize { declared, limit } => Self::HostileSize { declared, limit },
            SourceError::ForwardOnly => Self::Source {
                message: "forward input cannot satisfy the header".to_owned(),
            },
            SourceError::OutOfRange { position, length } => Self::Source {
                message: format!("header position {position} is past length {length}"),
            },
            SourceError::Failed { message } => Self::Source { message },
        }
    }

    /// A failed open does not publish streams.
    #[must_use]
    pub const fn streams_published(&self) -> bool {
        match self {
            Self::Unrecognized { .. }
            | Self::Truncated { .. }
            | Self::HostileSize { .. }
            | Self::UnsupportedBrand { .. }
            | Self::DuplicateTrack { .. }
            | Self::Source { .. } => false,
        }
    }
}

impl fmt::Display for OpenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unrecognized { family } => write!(formatter, "unrecognized {family:?} input"),
            Self::Truncated { expected, got } => {
                write!(
                    formatter,
                    "truncated header: expected {expected}, got {got}"
                )
            }
            Self::HostileSize { declared, limit } => {
                write!(formatter, "hostile header size {declared} exceeds {limit}")
            }
            Self::UnsupportedBrand { family } => {
                write!(formatter, "unsupported brand in {family:?}")
            }
            Self::DuplicateTrack { id } => write!(formatter, "duplicate track {id}"),
            Self::Source { message } => write!(formatter, "open failed: {message}"),
        }
    }
}

impl std::error::Error for OpenError {}

/// A packet could not be read from a demuxer that is already open.
///
/// [`Self::packet_returned`] is false. [`Self::streams_remain`] is true.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadError {
    /// A known-length input ended inside an access unit.
    Truncated { expected: u64, got: u64 },
    /// A forward input does not have the rest of this access unit yet.
    ShortRead { expected: u64, got: u64 },
    /// A declared access unit is larger than the limit or the remaining input.
    HostileSize { declared: u64, limit: u64 },
    /// [`crate::Demux::cancel`] was set before this packet was returned.
    Cancelled,
    /// The packet names a track the open did not publish.
    UnknownTrack { id: u32 },
    /// The packet timescale is not the stream timescale. It was not converted.
    TimescaleMismatch { packet: u32, stream: u32 },
    /// The read position did not change.
    Seek(SeekError),
    /// The source failed.
    Source { message: String },
}

impl ReadError {
    /// Maps a packet read into a read failure.
    ///
    /// A short read stays a short read. During open, that same source error
    /// is [`OpenError::Truncated`].
    #[must_use]
    pub fn from_packet(error: SourceError) -> Self {
        match error {
            SourceError::Truncated { expected, got } => Self::Truncated { expected, got },
            SourceError::ShortRead { expected, got } => Self::ShortRead { expected, got },
            SourceError::HostileSize { declared, limit } => Self::HostileSize { declared, limit },
            SourceError::ForwardOnly => Self::Seek(SeekError::ForwardOnly),
            SourceError::OutOfRange { position, length } => Self::Source {
                message: format!("packet position {position} is past length {length}"),
            },
            SourceError::Failed { message } => Self::Source { message },
        }
    }

    /// This call did not yield a packet.
    #[must_use]
    pub const fn packet_returned(&self) -> bool {
        match self {
            Self::Truncated { .. }
            | Self::ShortRead { .. }
            | Self::HostileSize { .. }
            | Self::Cancelled
            | Self::UnknownTrack { .. }
            | Self::TimescaleMismatch { .. }
            | Self::Seek(_)
            | Self::Source { .. } => false,
        }
    }

    /// Streams from the successful open are still the stream list.
    #[must_use]
    pub const fn streams_remain(&self) -> bool {
        match self {
            Self::Truncated { .. }
            | Self::ShortRead { .. }
            | Self::HostileSize { .. }
            | Self::Cancelled
            | Self::UnknownTrack { .. }
            | Self::TimescaleMismatch { .. }
            | Self::Seek(_)
            | Self::Source { .. } => true,
        }
    }
}

impl fmt::Display for ReadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { expected, got } => {
                write!(
                    formatter,
                    "truncated packet: expected {expected}, got {got}"
                )
            }
            Self::ShortRead { expected, got } => {
                write!(formatter, "short packet: expected {expected}, got {got}")
            }
            Self::HostileSize { declared, limit } => {
                write!(formatter, "hostile packet size {declared} exceeds {limit}")
            }
            Self::Cancelled => formatter.write_str("read cancelled"),
            Self::UnknownTrack { id } => write!(formatter, "unknown track {id}"),
            Self::TimescaleMismatch { packet, stream } => {
                write!(
                    formatter,
                    "packet timescale {packet} is not stream timescale {stream}"
                )
            }
            Self::Seek(error) => write!(formatter, "seek failed: {error}"),
            Self::Source { message } => write!(formatter, "read failed: {message}"),
        }
    }
}

impl std::error::Error for ReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Seek(error) => Some(error),
            _ => None,
        }
    }
}

/// Why a seek did not move the read position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeekError {
    /// The input cannot reposition.
    ForwardOnly,
    /// `seek_with_index` was called with an empty index.
    NoIndex,
    /// Index entries for one track go backwards in time.
    UnsortedIndex,
    /// The index mixes tracks. Seek wants one track, already sorted.
    MixedTrack,
    /// No stream uses this track id.
    UnknownTrack,
    /// The track has no access units.
    Empty,
}

impl fmt::Display for SeekError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ForwardOnly => formatter.write_str("forward input cannot seek"),
            Self::NoIndex => formatter.write_str("seek index is empty"),
            Self::UnsortedIndex => formatter.write_str("seek index is not ordered"),
            Self::MixedTrack => formatter.write_str("seek index mixes tracks"),
            Self::UnknownTrack => formatter.write_str("seek track is not in the file"),
            Self::Empty => formatter.write_str("seek track has no packets"),
        }
    }
}

impl std::error::Error for SeekError {}

/// A mux call failed before a complete file exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MuxError {
    /// The stream codec is [`cutlass_codec::StreamCodec::Unknown`].
    UnknownCodec(UnknownCodec),
    /// A size is above a limit. Nothing of that size was allocated.
    HostileSize { declared: u64, limit: u64 },
    /// Track count is above [`crate::Limits::max_tracks`].
    TooManyTracks { count: u32, limit: u32 },
    /// A packet timescale does not match its stream. It was not converted.
    TimescaleMismatch { packet: u32, stream: u32 },
    /// The packet names a track that was not added.
    UnknownTrack { id: u32 },
    /// A timestamp was compared on a zero timescale.
    Clockless,
    /// A track duration is negative.
    NegativeDuration,
    /// Interleave will not invent a time for a packet that has none.
    MissingTimestamp,
    /// The declared track duration is not the sum of the packet durations.
    DurationMismatch { track: u32 },
    /// A positive duration has no payload behind it.
    ShortPayload { track: u32 },
    /// [`crate::Mux::cancel`] was set.
    Cancelled,
    /// The mux is no longer accepting packets.
    NotBuilding,
    /// [`crate::Mux::finish`] already succeeded, or it was called again.
    AlreadyFinished,
    /// A track is missing from the duration list.
    Incomplete,
    /// The family recognized the stream and will not store it.
    Rejected { message: String },
}

impl fmt::Display for MuxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownCodec(_) => formatter.write_str("mux rejected an unknown codec"),
            Self::HostileSize { declared, limit } => {
                write!(formatter, "mux size {declared} exceeds {limit}")
            }
            Self::TooManyTracks { count, limit } => {
                write!(formatter, "mux has {count} tracks and the limit is {limit}")
            }
            Self::TimescaleMismatch { packet, stream } => {
                write!(
                    formatter,
                    "mux packet timescale {packet} is not stream timescale {stream}"
                )
            }
            Self::UnknownTrack { id } => write!(formatter, "mux packet uses unknown track {id}"),
            Self::Clockless => formatter.write_str("mux timestamp has no timescale"),
            Self::NegativeDuration => formatter.write_str("mux duration is negative"),
            Self::MissingTimestamp => formatter.write_str("mux packet has no timestamp"),
            Self::DurationMismatch { track } => {
                write!(
                    formatter,
                    "mux duration for track {track} does not match the packets"
                )
            }
            Self::ShortPayload { track } => {
                write!(formatter, "mux duration for track {track} has no payload")
            }
            Self::Cancelled => formatter.write_str("mux cancelled"),
            Self::NotBuilding => formatter.write_str("mux is not accepting writes"),
            Self::AlreadyFinished => formatter.write_str("mux already finished"),
            Self::Incomplete => formatter.write_str("mux duration list does not cover the streams"),
            Self::Rejected { message } => write!(formatter, "mux rejected the stream: {message}"),
        }
    }
}

impl std::error::Error for MuxError {}

#[cfg(test)]
mod tests {
    use super::{OpenError, ReadError, SeekError, SourceError};

    #[test]
    fn a_short_header_and_a_short_packet_are_different_errors() {
        let short = SourceError::ShortRead {
            expected: 8,
            got: 3,
        };
        let open = OpenError::from_header(short.clone());
        let read = ReadError::from_packet(short);
        assert_eq!(
            open,
            OpenError::Truncated {
                expected: 8,
                got: 3
            }
        );
        assert_eq!(
            read,
            ReadError::ShortRead {
                expected: 8,
                got: 3
            }
        );
        assert!(!open.streams_published());
        assert!(!read.packet_returned());
        assert!(read.streams_remain());
    }

    #[test]
    fn a_hostile_size_stays_a_hostile_size_on_each_path() {
        let hostile = SourceError::HostileSize {
            declared: u64::MAX,
            limit: 16,
        };
        assert!(matches!(
            OpenError::from_header(hostile.clone()),
            OpenError::HostileSize {
                declared: u64::MAX,
                limit: 16
            }
        ));
        assert!(matches!(
            ReadError::from_packet(hostile),
            ReadError::HostileSize {
                declared: u64::MAX,
                limit: 16
            }
        ));
    }

    #[test]
    fn every_open_failure_publishes_no_streams() {
        let errors = [
            OpenError::Unrecognized {
                family: cutlass_codec::ContainerId::Still,
            },
            OpenError::Truncated {
                expected: 1,
                got: 0,
            },
            OpenError::HostileSize {
                declared: 2,
                limit: 1,
            },
            OpenError::UnsupportedBrand {
                family: cutlass_codec::ContainerId::Isobmff,
            },
            OpenError::DuplicateTrack { id: 1 },
            OpenError::Source {
                message: "disk".to_owned(),
            },
        ];
        for error in errors {
            assert!(!error.streams_published());
        }
    }

    #[test]
    fn every_read_failure_returns_no_packet_and_keeps_streams() {
        let errors = [
            ReadError::Truncated {
                expected: 4,
                got: 1,
            },
            ReadError::ShortRead {
                expected: 4,
                got: 1,
            },
            ReadError::HostileSize {
                declared: 9,
                limit: 8,
            },
            ReadError::Cancelled,
            ReadError::UnknownTrack { id: 3 },
            ReadError::TimescaleMismatch {
                packet: 2,
                stream: 1,
            },
            ReadError::Seek(SeekError::ForwardOnly),
            ReadError::Source {
                message: "disk".to_owned(),
            },
        ];
        for error in errors {
            assert!(!error.packet_returned());
            assert!(error.streams_remain());
        }
    }

    #[test]
    fn a_rejected_mux_names_the_reason() {
        let error = super::MuxError::Rejected {
            message: "wav cannot store this codec".to_owned(),
        };
        assert!(error.to_string().contains("wav cannot store this codec"));
    }
}

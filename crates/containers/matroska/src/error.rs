//! Failures inside this crate, and how they map onto the shared open and
//! read errors.

use std::fmt;

use cutlass_codec::ContainerId;
use cutlass_container::{OpenError, ReadError, SeekError, SourceError};

use crate::ebml::EbmlError;

/// Why a Matroska file did not open, or a packet did not come back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The first four bytes are not the EBML magic, or the DocType is not
    /// `matroska` or `webm`.
    NotMatroska,
    /// The DocType is Matroska or WebM, and a version or EBML feature this
    /// reader does not implement is required.
    UnsupportedVersion {
        what: &'static str,
        value: u64,
    },
    /// Bytes that break the EBML or Matroska structure.
    Ebml(EbmlError),
    /// A structure that is well formed EBML and not valid Matroska.
    Malformed {
        what: String,
    },
    /// A feature this crate does not read.
    Unsupported {
        what: String,
    },
    /// A size or count above a limit. Nothing of that size was allocated.
    Limit {
        what: &'static str,
        value: u64,
        limit: u64,
    },
    /// Two TrackEntry elements share a TrackNumber.
    DuplicateTrack {
        number: u64,
    },
    /// The byte source failed or ended.
    Source(SourceError),
    Cancelled,
    Seek(SeekError),
}

impl Error {
    pub(crate) fn malformed(what: impl Into<String>) -> Self {
        Self::Malformed { what: what.into() }
    }

    pub(crate) fn unsupported(what: impl Into<String>) -> Self {
        Self::Unsupported { what: what.into() }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotMatroska => formatter.write_str("not a Matroska or WebM file"),
            Self::UnsupportedVersion { what, value } => {
                write!(formatter, "unsupported {what} {value}")
            }
            Self::Ebml(error) => write!(formatter, "matroska: {error}"),
            Self::Malformed { what } => write!(formatter, "malformed matroska: {what}"),
            Self::Unsupported { what } => write!(formatter, "unsupported matroska: {what}"),
            Self::Limit { what, value, limit } => {
                write!(
                    formatter,
                    "matroska {what} {value} is over the limit {limit}"
                )
            }
            Self::DuplicateTrack { number } => {
                write!(formatter, "matroska track number {number} appears twice")
            }
            Self::Source(error) => write!(formatter, "{error}"),
            Self::Cancelled => formatter.write_str("cancelled"),
            Self::Seek(error) => write!(formatter, "seek: {error}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<EbmlError> for Error {
    fn from(error: EbmlError) -> Self {
        Self::Ebml(error)
    }
}

impl From<SourceError> for Error {
    fn from(error: SourceError) -> Self {
        Self::Source(error)
    }
}

impl Error {
    /// The shared open failure. No stream list goes with it.
    #[must_use]
    pub fn into_open(self) -> OpenError {
        match self {
            Self::NotMatroska => OpenError::Unrecognized {
                family: ContainerId::Matroska,
            },
            Self::UnsupportedVersion { .. } => OpenError::UnsupportedBrand {
                family: ContainerId::Matroska,
            },
            Self::Source(error) => OpenError::from_header(error),
            Self::DuplicateTrack { number } => OpenError::DuplicateTrack {
                id: u32::try_from(number).unwrap_or(u32::MAX),
            },
            Self::Limit { value, limit, .. } => OpenError::HostileSize {
                declared: value,
                limit,
            },
            Self::Ebml(EbmlError::PastParent { size, room, .. }) => OpenError::HostileSize {
                declared: size,
                limit: room,
            },
            other => OpenError::Source {
                message: other.to_string(),
            },
        }
    }

    /// The shared read failure. Streams from the open stay valid.
    #[must_use]
    pub fn into_read(self) -> ReadError {
        match self {
            Self::Source(error) => ReadError::from_packet(error),
            Self::Limit { value, limit, .. } => ReadError::HostileSize {
                declared: value,
                limit,
            },
            Self::Ebml(EbmlError::PastParent { size, room, .. }) => ReadError::HostileSize {
                declared: size,
                limit: room,
            },
            Self::Cancelled => ReadError::Cancelled,
            Self::Seek(error) => ReadError::Seek(error),
            other => ReadError::Source {
                message: other.to_string(),
            },
        }
    }
}

pub(crate) type Result<T> = std::result::Result<T, Error>;

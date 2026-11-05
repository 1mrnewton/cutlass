//! Typed Ogg failures.
//!
//! The shared traits carry [`cutlass_container::OpenError`] and
//! [`cutlass_container::ReadError`]. Those have no room for an Ogg reason,
//! so the demuxer maps an [`OggError`] onto the closest shared variant and
//! keeps the typed value on [`crate::OggDemux::last_error`].

use std::fmt;

use cutlass_container::{OpenError, ReadError, SourceError};

/// Why an Ogg file, page, or Opus header was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OggError {
    /// The input does not start with an Ogg page, even after the resync scan.
    NoPage,
    /// No capture pattern within [`crate::MAX_RESYNC_BYTES`] of the read position.
    NoSync { scanned: u64 },
    /// The file ended inside a page.
    TruncatedPage { expected: u64, got: u64 },
    /// The first page of the file is not a beginning-of-stream page.
    NotBos,
    /// Two beginning-of-stream pages carry the same serial number.
    DuplicateSerial { serial: u32 },
    /// More logical streams than [`cutlass_container::Limits::max_tracks`].
    TooManyStreams { limit: u32 },
    /// A logical stream's first packet does not complete on its first page.
    BadBosPage { serial: u32 },
    /// `OpusHead` is shorter than its fields, or a field is out of range.
    BadOpusHead { serial: u32, reason: &'static str },
    /// `OpusHead` names a major version this crate does not read.
    UnsupportedOpusVersion { serial: u32, version: u8 },
    /// The second packet is not `OpusTags`, or its lengths overrun it.
    BadOpusTags { serial: u32, reason: &'static str },
    /// The file ended before every Opus stream's `OpusTags` completed.
    MissingOpusTags { serial: u32 },
    /// The page where `OpusTags` completes also completes an audio packet.
    TagsPageNotFinished { serial: u32 },
    /// A header or packet is larger than its bound.
    TooLarge { declared: u64, limit: u64 },
    /// A page granule position is below an earlier one, or below the samples
    /// the page completes on a page that is not the first or the last.
    GranuleBackwards {
        serial: u32,
        granule: i64,
        floor: i64,
    },
    /// The first audio page's granule position is below the samples that
    /// complete on it and the page is not end of stream (RFC 7845 section 4.5).
    InitialGranule {
        serial: u32,
        granule: i64,
        samples: i64,
    },
    /// The stream ends before its pre-skip is used up (RFC 7845 section 4.5).
    PreSkipPastEnd {
        serial: u32,
        end: i64,
        pre_skip: i64,
    },
    /// A page completes packets and its granule position is -1.
    MissingGranule { serial: u32 },
    /// A seek needs a random-access input.
    ForwardOnly,
}

impl fmt::Display for OggError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoPage => formatter.write_str("no Ogg page"),
            Self::NoSync { scanned } => {
                write!(formatter, "no Ogg capture pattern in {scanned} bytes")
            }
            Self::TruncatedPage { expected, got } => {
                write!(
                    formatter,
                    "Ogg page truncated: expected {expected}, got {got}"
                )
            }
            Self::NotBos => formatter.write_str("first Ogg page is not beginning of stream"),
            Self::DuplicateSerial { serial } => {
                write!(formatter, "Ogg serial {serial:#010x} begins twice")
            }
            Self::TooManyStreams { limit } => {
                write!(formatter, "more than {limit} Ogg logical streams")
            }
            Self::BadBosPage { serial } => {
                write!(
                    formatter,
                    "Ogg stream {serial:#010x} has a broken first page"
                )
            }
            Self::BadOpusHead { serial, reason } => {
                write!(formatter, "OpusHead of stream {serial:#010x}: {reason}")
            }
            Self::UnsupportedOpusVersion { serial, version } => {
                write!(
                    formatter,
                    "OpusHead of stream {serial:#010x} has unsupported version {version}"
                )
            }
            Self::BadOpusTags { serial, reason } => {
                write!(formatter, "OpusTags of stream {serial:#010x}: {reason}")
            }
            Self::MissingOpusTags { serial } => {
                write!(formatter, "Ogg stream {serial:#010x} has no OpusTags")
            }
            Self::TagsPageNotFinished { serial } => {
                write!(
                    formatter,
                    "OpusTags of stream {serial:#010x} does not finish its page"
                )
            }
            Self::TooLarge { declared, limit } => {
                write!(formatter, "Ogg size {declared} exceeds {limit}")
            }
            Self::GranuleBackwards {
                serial,
                granule,
                floor,
            } => write!(
                formatter,
                "Ogg stream {serial:#010x} granule {granule} goes back below {floor}"
            ),
            Self::InitialGranule {
                serial,
                granule,
                samples,
            } => write!(
                formatter,
                "Ogg stream {serial:#010x} first granule {granule} is below its {samples} samples"
            ),
            Self::PreSkipPastEnd {
                serial,
                end,
                pre_skip,
            } => write!(
                formatter,
                "Ogg stream {serial:#010x} ends at {end}, inside its pre-skip of {pre_skip}"
            ),
            Self::MissingGranule { serial } => {
                write!(
                    formatter,
                    "Ogg stream {serial:#010x} completes packets on a page with no granule"
                )
            }
            Self::ForwardOnly => formatter.write_str("forward Ogg input cannot seek"),
        }
    }
}

impl std::error::Error for OggError {}

impl OggError {
    /// The shared open failure closest to this reason.
    #[must_use]
    pub fn to_open(&self) -> OpenError {
        match self {
            Self::NoPage | Self::NotBos => OpenError::Unrecognized {
                family: cutlass_codec::ContainerId::Ogg,
            },
            Self::TruncatedPage { expected, got } => OpenError::Truncated {
                expected: *expected,
                got: *got,
            },
            Self::TooLarge { declared, limit } => OpenError::HostileSize {
                declared: *declared,
                limit: *limit,
            },
            Self::UnsupportedOpusVersion { .. } => OpenError::UnsupportedBrand {
                family: cutlass_codec::ContainerId::Ogg,
            },
            Self::DuplicateSerial { serial } => OpenError::DuplicateTrack { id: *serial },
            _ => OpenError::Source {
                message: self.to_string(),
            },
        }
    }

    /// The shared read failure closest to this reason.
    #[must_use]
    pub fn to_read(&self) -> ReadError {
        match self {
            Self::TruncatedPage { expected, got } => ReadError::Truncated {
                expected: *expected,
                got: *got,
            },
            Self::TooLarge { declared, limit } => ReadError::HostileSize {
                declared: *declared,
                limit: *limit,
            },
            Self::ForwardOnly => ReadError::Seek(cutlass_container::SeekError::ForwardOnly),
            _ => ReadError::Source {
                message: self.to_string(),
            },
        }
    }
}

/// A failure inside the demuxer: the source, or an Ogg reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Fault {
    Source(SourceError),
    Ogg(OggError),
    Seek(cutlass_container::SeekError),
}

impl From<SourceError> for Fault {
    fn from(error: SourceError) -> Self {
        Self::Source(error)
    }
}

impl From<OggError> for Fault {
    fn from(error: OggError) -> Self {
        Self::Ogg(error)
    }
}

#[cfg(test)]
mod tests {
    use super::OggError;
    use cutlass_container::{OpenError, ReadError};

    #[test]
    fn each_reason_maps_to_a_shared_variant_and_names_itself() {
        let truncated = OggError::TruncatedPage {
            expected: 40,
            got: 12,
        };
        assert_eq!(
            truncated.to_open(),
            OpenError::Truncated {
                expected: 40,
                got: 12
            }
        );
        assert_eq!(
            truncated.to_read(),
            ReadError::Truncated {
                expected: 40,
                got: 12
            }
        );
        let backwards = OggError::GranuleBackwards {
            serial: 7,
            granule: 10,
            floor: 20,
        };
        match backwards.to_read() {
            ReadError::Source { message } => assert!(message.contains("goes back")),
            other => panic!("unexpected {other:?}"),
        }
        assert!(matches!(
            OggError::NotBos.to_open(),
            OpenError::Unrecognized { .. }
        ));
        assert!(matches!(
            OggError::UnsupportedOpusVersion {
                serial: 1,
                version: 16
            }
            .to_open(),
            OpenError::UnsupportedBrand { .. }
        ));
    }
}

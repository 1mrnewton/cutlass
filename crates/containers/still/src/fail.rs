//! Parse failures, mapped onto the shared open errors.
//!
//! A failed parse does not return frames. The caller drops the buffer.

use cutlass_codec::ContainerId;
use cutlass_container::OpenError;

pub(crate) enum Fail {
    Truncated { expected: u64, got: u64 },
    Hostile { declared: u64, limit: u64 },
    Unrecognized,
    Bad(&'static str),
}

impl Fail {
    pub(crate) fn open(self) -> OpenError {
        match self {
            Self::Truncated { expected, got } => OpenError::Truncated { expected, got },
            Self::Hostile { declared, limit } => OpenError::HostileSize { declared, limit },
            Self::Unrecognized => OpenError::Unrecognized {
                family: ContainerId::Still,
            },
            Self::Bad(message) => OpenError::Source {
                message: message.to_owned(),
            },
        }
    }
}

pub(crate) fn truncated(expected: u64, got: u64) -> Fail {
    Fail::Truncated { expected, got }
}

pub(crate) fn hostile(declared: u64, limit: u64) -> Fail {
    Fail::Hostile { declared, limit }
}

pub(crate) fn as_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

//! How a malformed CAF file is reported.
//!
//! A field that runs past the bytes present is [`OpenError::Truncated`]. A
//! declared size larger than its bound is [`OpenError::HostileSize`]. A file
//! that is not CAF is [`OpenError::Unrecognized`], a later `mFileVersion` is
//! [`OpenError::UnsupportedBrand`], and a CAF file that breaks a rule of the
//! specification is [`OpenError::Source`] with a message that names the rule.

use cutlass_codec::ContainerId;
use cutlass_container::{MuxError, OpenError};

pub(crate) fn malformed(message: impl Into<String>) -> OpenError {
    OpenError::Source {
        message: message.into(),
    }
}

pub(crate) fn hostile(declared: u64, limit: u64) -> OpenError {
    OpenError::HostileSize { declared, limit }
}

pub(crate) fn truncated(expected: u64, got: u64) -> OpenError {
    OpenError::Truncated { expected, got }
}

pub(crate) const fn unrecognized() -> OpenError {
    OpenError::Unrecognized {
        family: ContainerId::Caf,
    }
}

pub(crate) fn rejected(message: impl Into<String>) -> MuxError {
    MuxError::Rejected {
        message: message.into(),
    }
}

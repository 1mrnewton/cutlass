//! Structured decode failures.
//!
//! A preflight rejection (truncated packet, bad length) does not consume the
//! packet and does not mutate committed reference state. A failure while
//! reconstructing a frame that already accepted earlier OBUs in the same
//! packet leaves any previously completed presentations queued and marks the
//! decoder `needs_reset`.

use cutlass_core::DecodeError as CoreDecodeError;

/// Why a decode step failed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DecodeError {
    /// A syntax element violated a bitstream requirement.
    #[error("malformed AV1 syntax: {0}")]
    Malformed(String),
    /// A complete OBU or length field ended before its declared size.
    #[error("truncated AV1 input: {0}")]
    Truncated(String),
    /// `drain` found a frame header still waiting for tile groups.
    #[error("incomplete AV1 frame at end of stream")]
    Incomplete,
    /// An inter frame named a slot that is empty or incompatible.
    #[error("missing AV1 reference slot {0}")]
    MissingReference(u8),
    /// A legal AV1 feature this build does not decode.
    #[error("unsupported AV1 feature: {0}")]
    Unsupported(String),
    /// A configured cap was exceeded. The bitstream may still be legal.
    #[error("AV1 resource limit: {0}")]
    Limit(String),
    /// The host cancelled the current submit.
    #[error("AV1 decode cancelled")]
    Cancelled,
    /// The decoder is waiting for [`crate::Av1Decoder::reset`].
    #[error("AV1 decoder needs reset: {0}")]
    NeedsReset(String),
    /// The call is illegal in the current state (for example submit after drain).
    #[error("AV1 API misuse: {0}")]
    Misuse(String),
}

impl DecodeError {
    pub(crate) fn malformed(msg: impl Into<String>) -> Self {
        Self::Malformed(msg.into())
    }

    pub(crate) fn truncated(msg: impl Into<String>) -> Self {
        Self::Truncated(msg.into())
    }

    pub(crate) fn unsupported(msg: impl Into<String>) -> Self {
        Self::Unsupported(msg.into())
    }

    pub(crate) fn limit(msg: impl Into<String>) -> Self {
        Self::Limit(msg.into())
    }
}

impl From<DecodeError> for CoreDecodeError {
    fn from(err: DecodeError) -> Self {
        match err {
            DecodeError::Unsupported(msg) => CoreDecodeError::Unsupported(msg),
            DecodeError::Truncated(msg) => CoreDecodeError::Decode(msg),
            DecodeError::Incomplete => {
                CoreDecodeError::Decode("incomplete AV1 frame at end of stream".into())
            }
            other => CoreDecodeError::Decode(other.to_string()),
        }
    }
}

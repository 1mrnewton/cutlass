//! `OSStatus` codes mapped to Cutlass errors.

use cutlass_core::AudioCodec;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cutlass-audiotoolbox doesn't decode {0:?}")]
    DecodeUnsupported(AudioCodec),
    #[error("not supported by cutlass-audiotoolbox: {0}")]
    Unsupported(&'static str),
    /// AudioToolbox has no codec for this stream on this device.
    #[error("{call}: AudioToolbox has no codec for this stream here (OSStatus {status})")]
    NoCodec { call: &'static str, status: i32 },
    #[error("{call} failed with OSStatus {status}")]
    Os { call: &'static str, status: i32 },
    #[error("invalid codec config record: {0}")]
    InvalidConfig(&'static str),
    #[error("invalid input: {0}")]
    InvalidInput(&'static str),
}

impl Error {
    /// The caller should try another backend, such as a software codec.
    pub fn is_unsupported(&self) -> bool {
        match self {
            Self::DecodeUnsupported(_) | Self::Unsupported(_) | Self::NoCodec { .. } => true,
            Self::Os { .. } | Self::InvalidConfig(_) | Self::InvalidInput(_) => false,
        }
    }
}

/// Turns an `OSStatus` into a result. `no_codec` lists the codes that mean
/// this device cannot do the conversion.
pub(crate) fn check(call: &'static str, status: i32, no_codec: &[i32]) -> Result<(), Error> {
    if status == 0 {
        Ok(())
    } else if no_codec.contains(&status) {
        Err(Error::NoCodec { call, status })
    } else {
        Err(Error::Os { call, status })
    }
}

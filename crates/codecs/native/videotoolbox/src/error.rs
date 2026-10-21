//! `OSStatus` codes mapped to Cutlass errors.

use cutlass_core::VideoCodec;
use objc2_video_toolbox::{
    kVTCouldNotFindVideoDecoderErr, kVTCouldNotFindVideoEncoderErr,
    kVTVideoDecoderUnsupportedDataFormatErr,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cutlass-videotoolbox doesn't decode {0:?}")]
    DecodeUnsupported(VideoCodec),
    #[error("cutlass-videotoolbox doesn't encode {0:?}")]
    EncodeUnsupported(VideoCodec),
    #[error("not supported by cutlass-videotoolbox: {0}")]
    Unsupported(&'static str),
    /// VideoToolbox has no decoder or encoder for the stream on this device.
    #[error("{call}: VideoToolbox has no codec for this stream here (OSStatus {status})")]
    NoCodec { call: &'static str, status: i32 },
    #[error("{call} failed with OSStatus {status}")]
    Os { call: &'static str, status: i32 },
    #[error("{call} succeeded but returned nothing")]
    NullOutput { call: &'static str },
    #[error("VideoToolbox returned a frame without a valid timestamp")]
    MissingTimestamp,
    #[error("invalid codec config record: {0}")]
    InvalidConfig(&'static str),
    #[error("invalid input: {0}")]
    InvalidInput(&'static str),
}

impl Error {
    /// The caller should try another backend, such as a software codec.
    pub fn is_unsupported(&self) -> bool {
        match self {
            Self::DecodeUnsupported(_)
            | Self::EncodeUnsupported(_)
            | Self::Unsupported(_)
            | Self::NoCodec { .. } => true,
            Self::Os { .. }
            | Self::NullOutput { .. }
            | Self::MissingTimestamp
            | Self::InvalidConfig(_)
            | Self::InvalidInput(_) => false,
        }
    }
}

const NO_CODEC: [i32; 3] = [
    kVTCouldNotFindVideoDecoderErr,
    kVTCouldNotFindVideoEncoderErr,
    kVTVideoDecoderUnsupportedDataFormatErr,
];

/// Turns an `OSStatus` or `CVReturn` into a result.
pub(crate) fn check(call: &'static str, status: i32) -> Result<(), Error> {
    if status == 0 {
        Ok(())
    } else if NO_CODEC.contains(&status) {
        Err(Error::NoCodec { call, status })
    } else {
        Err(Error::Os { call, status })
    }
}

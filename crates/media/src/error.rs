//! Errors from probing a file or opening a decoder.

use cutlass_core::VideoCodec;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Container(#[from] cutlass_mp4::Error),
    /// Neither the operating system nor a Rust codec decodes this one.
    #[error("no decoder for {0:?}")]
    NoDecoder(VideoCodec),
    #[error("track index {0} is not in the file")]
    NoTrack(usize),
    #[error("track {0} is not video")]
    NotVideo(u32),
    /// A decoder that was chosen failed. The text includes the OS status when
    /// the system codec returned one.
    #[error("{0}")]
    Decode(String),
}

#[cfg(target_vendor = "apple")]
pub(crate) fn from_videotoolbox(error: cutlass_videotoolbox::Error) -> Error {
    use cutlass_videotoolbox::Error as Vt;
    match error {
        Vt::DecodeUnsupported(codec) => Error::NoDecoder(codec),
        Vt::EncodeUnsupported(_) => Error::Decode("a decoder returned an encode error".to_owned()),
        Vt::Unsupported(message) | Vt::InvalidConfig(message) | Vt::InvalidInput(message) => {
            Error::Decode(message.to_owned())
        }
        Vt::NoCodec { call, status } | Vt::Os { call, status } => {
            Error::Decode(format!("{call} failed with OSStatus {status}"))
        }
        Vt::NullOutput { call } => Error::Decode(format!("{call} succeeded but returned nothing")),
        Vt::MissingTimestamp => {
            Error::Decode("VideoToolbox returned a frame without a valid timestamp".to_owned())
        }
    }
}

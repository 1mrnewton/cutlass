//! Refusals from the device.

use std::fmt;

/// Why the device refused a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceError {
    /// An edge is past the limit.
    Extent,
    /// The picture size is not the framebuffer size.
    Size,
    /// The byte length is not four times the pixel count.
    Bytes,
    /// A platform buffer cannot be sampled on the portable path.
    PlatformBuffer,
    /// The id is not a live framebuffer.
    MissingFramebuffer,
    /// `begin` was called while a pass was already open.
    PassOpen,
    /// The pass is not the open one.
    NoPass,
    /// This pass has ended.
    Ended,
    /// `submit` was called before `end`.
    NotEnded,
    /// `read_rgba` was called before `submit`.
    NotSubmitted,
    /// Metal submission failed, or this process has no Metal device.
    Backend,
}

impl fmt::Display for DeviceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Extent => "the frame size is past the limit",
            Self::Size => "the picture size does not match the framebuffer",
            Self::Bytes => "the picture byte length does not match its size",
            Self::PlatformBuffer => "the portable path cannot sample a platform buffer",
            Self::MissingFramebuffer => "the framebuffer is not on this device",
            Self::PassOpen => "a pass is already open",
            Self::NoPass => "this pass is not open",
            Self::Ended => "this pass has ended",
            Self::NotEnded => "this pass has not ended",
            Self::NotSubmitted => "the frame has not been submitted",
            Self::Backend => "the metal device could not submit the frame",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for DeviceError {}

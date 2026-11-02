//! Why the Windows device refused a call.

use std::fmt;

/// A refusal from [`WindowsDevice`](crate::device::WindowsDevice).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowsError {
    /// An edge is past the limit.
    Extent,
    /// The picture size is not the framebuffer size.
    Size,
    /// The byte length is not four times the pixel count.
    Bytes,
    /// A platform buffer cannot be sampled here.
    PlatformBuffer,
    /// The id is not a live framebuffer.
    MissingFramebuffer,
    /// `begin` was called while a pass was already open.
    PassOpen,
    /// The pass is not the open one.
    NoPass,
    /// Draw, clear, or blend was called after `end`.
    Ended,
    /// `submit` was called before `end`.
    NotEnded,
    /// `read_rgba` was called before `submit`.
    NotSubmitted,
    /// Direct3D is not available on this host.
    GpuUnsupported,
    /// The platform token is not a Direct3D resource on this device.
    UnknownResource,
}

impl fmt::Display for WindowsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Extent => "the frame size is past the limit",
            Self::Size => "the picture size does not match the framebuffer",
            Self::Bytes => "the picture byte length does not match its size",
            Self::PlatformBuffer => "the windows device cannot sample a platform buffer",
            Self::MissingFramebuffer => "the framebuffer is not on this device",
            Self::PassOpen => "a pass is already open",
            Self::NoPass => "this pass is not open",
            Self::Ended => "this pass has ended",
            Self::NotEnded => "this pass has not ended",
            Self::NotSubmitted => "the frame has not been submitted",
            Self::GpuUnsupported => "Direct3D is not available on this host",
            Self::UnknownResource => "the platform token is not a Direct3D resource on this device",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for WindowsError {}

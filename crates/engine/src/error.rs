//! Failures from opening a file, mapping time, or mixing audio.
//!
//! A failure names the cause. It does not change the draft.

use std::fmt;
use std::path::PathBuf;

use cutlass_codec::access::CodecError;

use crate::map::MapError;

/// The engine could not open, map, decode, or mix.
#[derive(Clone, Debug, PartialEq)]
pub enum EngineError {
    /// The path is not a file this process can open.
    MissingFile { path: PathBuf },
    /// No registered container accepted the header.
    UnknownContainer,
    /// The file ended inside a header or a packet.
    Truncated { expected: u64, got: u64 },
    /// A declared size was past a limit. Nothing of that size was allocated.
    Hostile { declared: u64, limit: u64 },
    /// The codec id is known and no registered decoder accepted it.
    NoDecoder { codec: String },
    /// The stream's codec tag is not a codec this workspace names.
    UnknownCodec { tag: String },
    /// Text, a shape, or a solid color does not enter the decoder or the mix.
    NotMedia,
    /// The segment id is not on this draft.
    MissingSegment,
    /// Playback has no picture yet. The audio clock is not waiting on it.
    NotReady,
    /// `cutlass-color` cannot turn this layout into CPU pixels.
    ColorMiss { layout: String },
    /// The window size is zero or past [`crate::MAX_WINDOW`].
    Window,
    /// One seek read more packets than the decode bound.
    DecodeLimit,
    /// The timeline time is outside the segment, or the source position is not in range.
    Outside,
    /// Speed was missing, not finite, or not greater than zero.
    NonPositiveSpeed,
    /// The packet timescale is zero, so the ticks are not a clock.
    NoClock,
    /// A product does not fit in the microsecond clock.
    Overflow,
    /// The source range is absent, so file time has nothing to land on.
    NoSource,
    /// A curve with fewer than two points is unused. The constant speed stays.
    UnusedCurve,
    /// Two clips in one mix do not share a channel layout.
    Layout,
    /// The audio crate refused the buffer.
    Audio(String),
    /// A decoder refused a packet.
    Codec(CodecError),
    /// The container failed while it was already open.
    Read { message: String },
    /// The file could not be read.
    Io { message: String },
}

impl fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingFile { path } => {
                write!(formatter, "missing media file {}", path.display())
            }
            Self::UnknownContainer => formatter.write_str("unknown container"),
            Self::Truncated { expected, got } => {
                write!(formatter, "truncated media: expected {expected}, got {got}")
            }
            Self::Hostile { declared, limit } => {
                write!(formatter, "hostile media size {declared} exceeds {limit}")
            }
            Self::NoDecoder { codec } => write!(formatter, "no decoder for {codec}"),
            Self::UnknownCodec { tag } => write!(formatter, "unknown codec {tag}"),
            Self::NotMedia => formatter.write_str("text and shapes do not enter the media graph"),
            Self::MissingSegment => formatter.write_str("segment is not on the draft"),
            Self::NotReady => formatter.write_str("picture is not ready"),
            Self::ColorMiss { layout } => write!(formatter, "no color conversion for {layout}"),
            Self::Window => formatter.write_str("picture window is empty or past the limit"),
            Self::DecodeLimit => formatter.write_str("decode stopped after the packet bound"),
            Self::Outside => formatter.write_str("time is outside the mapped range"),
            Self::NonPositiveSpeed => formatter.write_str("speed must be finite and positive"),
            Self::NoClock => formatter.write_str("file timescale is not a clock"),
            Self::Overflow => formatter.write_str("media time does not fit"),
            Self::NoSource => formatter.write_str("segment has no source range"),
            Self::UnusedCurve => formatter.write_str("speed curve has fewer than two points"),
            Self::Layout => formatter.write_str("mix channel layouts differ"),
            Self::Audio(message) | Self::Read { message } | Self::Io { message } => {
                formatter.write_str(message)
            }
            Self::Codec(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<MapError> for EngineError {
    fn from(error: MapError) -> Self {
        match error {
            MapError::Outside => Self::Outside,
            MapError::NonPositiveSpeed => Self::NonPositiveSpeed,
            MapError::NoClock => Self::NoClock,
            MapError::Overflow => Self::Overflow,
            MapError::NoSource => Self::NoSource,
            MapError::UnusedCurve => Self::UnusedCurve,
        }
    }
}

impl From<cutlass_audio::Error> for EngineError {
    fn from(error: cutlass_audio::Error) -> Self {
        Self::Audio(error.to_string())
    }
}

impl From<CodecError> for EngineError {
    fn from(error: CodecError) -> Self {
        Self::Codec(error)
    }
}

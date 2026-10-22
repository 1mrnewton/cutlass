//! Failures from building or editing a project.

use crate::{AssetId, SegmentId, TrackId};

/// Why a project value was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("source string is empty")]
    EmptySource,
    #[error("width and height must be non-zero")]
    ZeroCanvas,
    #[error("drop frame is only valid at 29.97 and 59.94")]
    DropFrameRate,
    #[error("duration is negative")]
    NegativeDuration,
    #[error("time is negative")]
    NegativeTime,
    #[error("duration is zero")]
    ZeroDuration,
    #[error("range mixes timescales {start} and {duration}")]
    TimescaleMismatch { start: u32, duration: u32 },
    #[error("source range uses timescale {source_scale}, asset uses {asset}")]
    SourceTimescale { asset: u32, source_scale: u32 },
    #[error("duration overflowed")]
    DurationOverflow,
    #[error("speed is zero")]
    ZeroSpeed,
    #[error("duration rounds to zero on the sequence clock")]
    RoundsToZero,
    #[error("an image has no source range")]
    ImageSource,
    #[error("media needs a source range")]
    MediaSource,
    #[error("source range is outside the asset")]
    SourceOutsideAsset,
    #[error("channels do not match the asset")]
    Channels,
    #[error("segment does not belong on this track")]
    WrongTrack,
    #[error("the timeline already has a main track")]
    SecondMainTrack,
    #[error("a non-main track needs a segment")]
    EmptyTrack,
    #[error("segments overlap")]
    Overlap,
    #[error("the split is not inside the segment")]
    SplitOutside,
    #[error("unknown asset {0}")]
    UnknownAsset(AssetId),
    #[error("unknown track {0}")]
    UnknownTrack(TrackId),
    #[error("unknown segment {0}")]
    UnknownSegment(SegmentId),
    #[error("duplicate id {0}")]
    DuplicateSegment(SegmentId),
    #[error("duplicate id {0}")]
    DuplicateTrack(TrackId),
    #[error("keyframe is outside the segment")]
    KeyframeOutOfRange,
    #[error("two keyframes share a property and an offset")]
    DuplicateKeyframe,
    #[error("no keyframe of that property at that offset")]
    MissingKeyframe,
    #[error("opacity is outside 0..=1")]
    Opacity,
    #[error("volume is negative")]
    NegativeVolume,
    #[error("fade is longer than the segment")]
    FadeOutOfRange,
    #[error("the segment has no sound")]
    NoSound,
    #[error("the segment has no picture")]
    NoPicture,
    #[error("the crop leaves no picture")]
    Crop,
    #[error("a freeze holds one frame")]
    Freeze,
    #[error("the reverse does not match its asset")]
    Reverse,
    #[error("the track is locked")]
    Locked,
    #[error("value is not finite")]
    NonFinite,
    #[error("scale must be positive")]
    NonPositiveScale,
    #[error("frame rate {numerator}/{denominator} is not a project rate")]
    UnknownRate { numerator: u32, denominator: u32 },
    #[error("linked segment {0} would overlap")]
    LinkageOverlap(SegmentId),
    #[error("nothing to undo")]
    NothingToUndo,
    #[error("nothing to redo")]
    NothingToRedo,
    #[error("id is empty or the wrong kind")]
    BadId,
    #[error("the project has no ids left")]
    IdsExhausted,
    #[error("the project file could not be read: {0}")]
    Unreadable(String),
    /// `1` is [`crate::FILE_VERSION`].
    #[error("project file version {found} is not 1")]
    FileVersion { found: u64 },
    #[error("the id counter {next_id} does not pass every id in the file")]
    IdCounter { next_id: u64 },
    #[error("id {0} is repeated")]
    RepeatedId(u64),
    #[error("tracks are not in band order")]
    TrackOrder,
    #[error("the file has no main track")]
    MissingMain,
    #[error("timescale is zero")]
    ZeroTimescale,
    #[error("timeline duration {found} does not match {expected}")]
    TimelineDuration { found: i64, expected: i64 },
    #[error("an image segment plays at 1/1")]
    ImageSpeed,
}

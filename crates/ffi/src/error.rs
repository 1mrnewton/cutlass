//! Failures the phone shells can match.
//!
//! A session failure stays the session's error. A folder, playhead, or cache
//! failure stays that crate's error. A path that leaves the project folder is
//! its own case, so the shell does not have to read a filesystem error to
//! learn that the sandbox refused it.

use std::fmt;

use cutlass_playback::PlayheadError;
use cutlass_session::EditError;
use cutlass_storage::StorageError;
use cutlass_store::StoreError;

/// Why a phone call did not change the open project, the playhead, or the cache.
#[derive(Debug)]
pub enum FfiError {
    /// The project folder could not be read or written.
    Store(StoreError),
    /// The session refused the revision, the request, or the history call.
    Edit(EditError),
    /// The playhead refused the call. The draft was not rewritten.
    Playhead(PlayheadError),
    /// The bundle or the cache has no published file for this id.
    Storage(StorageError),
    /// `name` is not inside the open project folder.
    PathEscapes {
        /// The path the phone asked for.
        name: String,
    },
    /// The call was refused before it reached the session: a file that
    /// cannot be imported, a library item with nothing to place, or a split
    /// too close to an edge. The draft is unchanged.
    Refused {
        /// Why, for the phone to show.
        reason: String,
    },
    /// The picture at that time could not be drawn.
    Frame {
        /// Why, for the phone to show.
        reason: String,
    },
}

impl fmt::Display for FfiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(error) => write!(formatter, "{error}"),
            Self::Edit(error) => write!(formatter, "{}", edit_cause(*error)),
            Self::Playhead(error) => write!(formatter, "{error}"),
            Self::Storage(error) => write!(formatter, "{error}"),
            Self::PathEscapes { name } => {
                write!(formatter, "path {name} leaves the project folder")
            }
            Self::Refused { reason } | Self::Frame { reason } => formatter.write_str(reason),
        }
    }
}

impl std::error::Error for FfiError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Store(error) => Some(error),
            Self::Playhead(error) => Some(error),
            Self::Storage(error) => Some(error),
            Self::Edit(_)
            | Self::PathEscapes { .. }
            | Self::Refused { .. }
            | Self::Frame { .. } => None,
        }
    }
}

impl From<StoreError> for FfiError {
    fn from(error: StoreError) -> Self {
        Self::Store(error)
    }
}

impl From<EditError> for FfiError {
    fn from(error: EditError) -> Self {
        Self::Edit(error)
    }
}

impl From<PlayheadError> for FfiError {
    fn from(error: PlayheadError) -> Self {
        Self::Playhead(error)
    }
}

impl From<StorageError> for FfiError {
    fn from(error: StorageError) -> Self {
        Self::Storage(error)
    }
}

fn edit_cause(error: EditError) -> String {
    match error {
        EditError::RevisionMismatch { expected, actual } => {
            format!("revision {expected} is not the open revision {actual}")
        }
        EditError::MissingTrack => "the track is not on the draft".to_owned(),
        EditError::MissingSegment => "the segment is not on the draft".to_owned(),
        EditError::DuplicateSegment => "the segment id is empty or already on the draft".to_owned(),
        EditError::LockedTrack => "the track is locked".to_owned(),
        EditError::Overlap => "segments on one track overlap".to_owned(),
        EditError::Speed(error) => format!("the speed mapping was refused: {error:?}"),
        EditError::WordShape => "the word columns do not match".to_owned(),
        EditError::Range(error) => format!("the time range was refused: {error:?}"),
        EditError::EmptyUndo => "there is nothing to undo".to_owned(),
        EditError::EmptyRedo => "there is nothing to redo".to_owned(),
        EditError::HistoryOpen => "a history transaction is already open".to_owned(),
        EditError::HistoryClosed => "no history transaction is open".to_owned(),
        EditError::UnknownAttribute => {
            "the track attribute is not mute, hide, lock, solo, or muted-by-solo".to_owned()
        }
        EditError::MissingMaterial => "the material is not on the draft".to_owned(),
        EditError::BeforeZero => "a clip cannot start before the timeline does".to_owned(),
        EditError::LaneKind => "that lane holds another kind of clip".to_owned(),
        EditError::SourceBounds => {
            "the clip cannot reach past the start or end of its media".to_owned()
        }
        EditError::DuplicateMaterial => "the material id is already in a pool".to_owned(),
        EditError::TransitionWindow => "the split falls inside a transition window".to_owned(),
        EditError::AlreadyLinked => "the segment is already linked".to_owned(),
        EditError::TrackKind => "the track could not be added".to_owned(),
        EditError::MainTrack => "the main track cannot be removed".to_owned(),
        EditError::DuplicateMark => "a marker is already there".to_owned(),
        EditError::DuplicateBeat => "a beat is already there".to_owned(),
        EditError::NotAudio => "a beat belongs on audio".to_owned(),
        EditError::NotVideo => "an extract belongs on video".to_owned(),
        EditError::NoSound => "the clip has no sound".to_owned(),
        EditError::AlreadyExtracted => "the sound is already on its own clip".to_owned(),
        EditError::Angle => "the rotation is not a finite number of degrees".to_owned(),
        EditError::Position => "the position is not a finite number".to_owned(),
        EditError::Scale => "the scale must be greater than zero".to_owned(),
        EditError::Fade => "a fade cannot be negative".to_owned(),
        EditError::Volume => "the volume must be zero or more".to_owned(),
        EditError::Opacity => "the opacity must be from zero to one".to_owned(),
        EditError::KeyTime => "a keyframe must be on its clip".to_owned(),
        EditError::NoKey => "no keyframe is there".to_owned(),
        EditError::KeyValue => "that setting cannot be keyed to that value".to_owned(),
        EditError::Graph => "that curve is not one the app offers".to_owned(),
        EditError::Animation => "that animation is not one the app offers".to_owned(),
        EditError::NotText => "a text style belongs on text".to_owned(),
        EditError::NotShape => "a shape style belongs on a shape".to_owned(),
        EditError::ShapeStyle => {
            "that shape size, stroke, corner, or shadow is out of range".to_owned()
        }
        EditError::TextSize => "the text size must be greater than zero and at most 512".to_owned(),
        EditError::TextColor => "the text color is not a hex color".to_owned(),
        EditError::TextAlign => "text alignment is not left, center, or right".to_owned(),
        EditError::LetterSpacing => "letter spacing must be from -1 to 4".to_owned(),
        EditError::LineSpacing => "line spacing must be greater than -1 and at most 4".to_owned(),
        EditError::LineWidth => "the line width must be from zero through 8192".to_owned(),
        EditError::TextFont => "the face is not a file name".to_owned(),
        EditError::TextStyle => "that outline, shadow, or background is out of range".to_owned(),
        EditError::FreeLayer => {
            "arranging needs a picture clip and Free layer on in the project settings".to_owned()
        }
        EditError::TextEmpty => "type something to add text".to_owned(),
        EditError::UnknownEffect => "that effect is not in this version".to_owned(),
        EditError::Lut => "that file is not a 3D .cube table".to_owned(),
        EditError::Mask => "a mask number is out of its range".to_owned(),
        EditError::Track => "the track or its followers do not fit the clip".to_owned(),
        EditError::Compound => "that is not a compound clip played at its own speed".to_owned(),
        EditError::NotEffect => "a strength belongs on an effect, filter, or adjustment".to_owned(),
        EditError::EffectTarget => "an effect goes on a picture clip".to_owned(),
        EditError::EffectStrength => "the strength must be from zero to one".to_owned(),
        EditError::CanvasColor => "the canvas color is not a hex color".to_owned(),
        EditError::CanvasSize => "the canvas side must be from 1 to 7680".to_owned(),
        EditError::CanvasRatio => "the canvas ratio is not a project ratio".to_owned(),
        EditError::Letterbox => "a letterbox belongs on a picture".to_owned(),
        EditError::LetterboxBlur => "the blur is not a fraction from 0 through 1".to_owned(),
        EditError::FrameRate => "the frame rate is not a project rate".to_owned(),
        EditError::ColorSpace => "the color space is not a project color space".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    #[test]
    fn a_session_error_keeps_its_cause() {
        let error = FfiError::from(EditError::RevisionMismatch {
            expected: 0,
            actual: 2,
        });
        assert!(matches!(
            error,
            FfiError::Edit(EditError::RevisionMismatch {
                expected: 0,
                actual: 2
            })
        ));
        assert_eq!(error.to_string(), "revision 0 is not the open revision 2");
        assert!(error.source().is_none());
    }

    #[test]
    fn extract_audio_names_why_it_was_refused() {
        assert_eq!(
            FfiError::from(EditError::NotVideo).to_string(),
            "an extract belongs on video"
        );
        assert_eq!(
            FfiError::from(EditError::NoSound).to_string(),
            "the clip has no sound"
        );
        assert_eq!(
            FfiError::from(EditError::AlreadyExtracted).to_string(),
            "the sound is already on its own clip"
        );
        assert_eq!(
            FfiError::from(EditError::Position).to_string(),
            "the position is not a finite number"
        );
        assert_eq!(
            FfiError::from(EditError::Scale).to_string(),
            "the scale must be greater than zero"
        );
        assert_eq!(
            FfiError::from(EditError::Fade).to_string(),
            "a fade cannot be negative"
        );
        assert_eq!(
            FfiError::from(EditError::Opacity).to_string(),
            "the opacity must be from zero to one"
        );
        assert_eq!(
            FfiError::from(EditError::NotText).to_string(),
            "a text style belongs on text"
        );
        assert_eq!(
            FfiError::from(EditError::TextSize).to_string(),
            "the text size must be greater than zero and at most 512"
        );
        assert_eq!(
            FfiError::from(EditError::TextColor).to_string(),
            "the text color is not a hex color"
        );
        assert_eq!(
            FfiError::from(EditError::TextAlign).to_string(),
            "text alignment is not left, center, or right"
        );
        assert_eq!(
            FfiError::from(EditError::LetterSpacing).to_string(),
            "letter spacing must be from -1 to 4"
        );
        assert_eq!(
            FfiError::from(EditError::LineSpacing).to_string(),
            "line spacing must be greater than -1 and at most 4"
        );
        assert_eq!(
            FfiError::from(EditError::LineWidth).to_string(),
            "the line width must be from zero through 8192"
        );
        assert_eq!(
            FfiError::from(EditError::TextFont).to_string(),
            "the face is not a file name"
        );
        assert_eq!(
            FfiError::from(EditError::CanvasColor).to_string(),
            "the canvas color is not a hex color"
        );
        assert_eq!(
            FfiError::from(EditError::CanvasSize).to_string(),
            "the canvas side must be from 1 to 7680"
        );
        assert_eq!(
            FfiError::from(EditError::CanvasRatio).to_string(),
            "the canvas ratio is not a project ratio"
        );
        assert_eq!(
            FfiError::from(EditError::Letterbox).to_string(),
            "a letterbox belongs on a picture"
        );
        assert_eq!(
            FfiError::from(EditError::FrameRate).to_string(),
            "the frame rate is not a project rate"
        );
        assert_eq!(
            FfiError::from(EditError::ColorSpace).to_string(),
            "the color space is not a project color space"
        );
    }

    #[test]
    fn a_track_that_cannot_be_added_names_the_cause() {
        let error = FfiError::from(EditError::TrackKind);
        assert_eq!(error.to_string(), "the track could not be added");
    }

    #[test]
    fn a_path_escape_names_the_path() {
        let error = FfiError::PathEscapes {
            name: "../secret".to_owned(),
        };
        assert_eq!(
            error.to_string(),
            "path ../secret leaves the project folder"
        );
    }
}

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
        }
    }
}

impl std::error::Error for FfiError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Store(error) => Some(error),
            Self::Playhead(error) => Some(error),
            Self::Storage(error) => Some(error),
            Self::Edit(_) | Self::PathEscapes { .. } => None,
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
        EditError::NothingToAmend => "there is no recorded step to amend".to_owned(),
        EditError::UnknownAttribute => {
            "the track attribute is not mute, hide, lock, solo, or muted-by-solo".to_owned()
        }
        EditError::MissingMaterial => "the material is not on the draft".to_owned(),
        EditError::DuplicateMaterial => "the material id is already in a pool".to_owned(),
        EditError::TransitionWindow => "the split falls inside a transition window".to_owned(),
        EditError::AlreadyLinked => "the segment is already linked".to_owned(),
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

//! Writes the cut to a file.
//!
//! On Apple the file is an MP4 of the cut the preview plays: H.264 picture
//! and AAC sound. The shell asks where to put it. WebM, GIF, HEVC, ProRes,
//! a range, and the dialog's resolution are left out. Elsewhere there is no
//! encoder yet.

#![deny(unsafe_code)]

mod picture;

#[cfg(target_vendor = "apple")]
mod write;

use std::path::Path;

use cutlass_engine::cues;
use cutlass_project::Project;

/// Why a file was not written.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// This system has no H.264 and AAC encoder yet.
    #[error("this system has no encoder yet")]
    NoEncoder,
    /// The timeline has no picture and no sound.
    #[error("the cut is empty")]
    Empty,
    /// Width or height is below two pixels after the even-size step.
    #[error("the canvas is too small to encode")]
    Canvas,
    /// A picture could not be opened, decoded, or encoded.
    #[error("the picture could not be written: {0}")]
    Picture(String),
    /// A sound could not be opened, decoded, or encoded.
    #[error("the sound could not be written: {0}")]
    Sound(String),
    /// The bytes did not land on disk.
    #[error("the file could not be written: {0}")]
    File(String),
}

/// Writes `project` to `path`.
///
/// On Apple this is an MP4. The picture is the project canvas at the project
/// frame rate, and the sound is the same mix the preview plays. An odd canvas
/// loses its last row or column so H.264 can store it.
pub fn write(project: &Project, path: &Path) -> Result<(), Error> {
    let duration = project.duration().value;
    let rate = project.settings().rate().rate();
    if picture::frame_times(duration, rate).is_empty() && cues(project).is_empty() {
        return Err(Error::Empty);
    }
    #[cfg(target_vendor = "apple")]
    {
        write::file(project, path)
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = path;
        Err(Error::NoEncoder)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use cutlass_project::{FrameRate, Project, ProjectId};

    use super::picture::{encode_size, frame_times};
    use super::{Error, write};

    #[test]
    fn an_empty_cut_is_refused() {
        let project = Project::new(ProjectId::from_bytes([1; 16]), "Empty");
        let error = write(&project, Path::new("unused.mp4")).expect_err("empty");
        assert!(matches!(error, Error::Empty));
    }

    #[test]
    fn one_second_at_thirty_is_thirty_frames() {
        let frames = frame_times(1_000_000, FrameRate::Fps30);
        assert_eq!(frames.len(), 30);
        assert_eq!(frames[0], 0);
        assert_eq!(*frames.last().expect("frame"), 966_666);
    }

    #[test]
    fn an_odd_canvas_loses_its_last_row_and_column() {
        assert_eq!(encode_size(1920, 1080), Some((1920, 1080)));
        assert_eq!(encode_size(1919, 1080), Some((1918, 1080)));
        assert_eq!(encode_size(3, 1), None);
    }
}

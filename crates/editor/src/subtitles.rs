//! A subtitle file brought in as captions on a new subtitle lane.

use std::error::Error;
use std::fmt;

use cutlass_captions::{Cue, SubtitleError, parse_srt};
use cutlass_model::{MaterialId, SegmentId, TimeRange, TrackId};
use cutlass_session::{Request, SubtitleCue};

/// The subtitle files the editor reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubtitleFormat {
    Srt,
}

impl SubtitleFormat {
    /// File extensions the open dialog offers, without the dot.
    pub const EXTENSIONS: &'static [&'static str] = &["srt"];

    /// The format a file name's extension names, ignoring case.
    #[must_use]
    pub fn of_file(name: &str) -> Option<Self> {
        let (_, extension) = name.rsplit_once('.')?;
        if extension.eq_ignore_ascii_case("srt") {
            Some(Self::Srt)
        } else {
            None
        }
    }
}

/// Why a subtitle file brought in no captions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubtitleMiss {
    /// The file did not read.
    File(SubtitleError),
    /// A cue lies past the end of the timeline clock.
    TooLate,
}

impl fmt::Display for SubtitleMiss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(error) => error.fmt(f),
            Self::TooLate => write!(f, "a subtitle time is past the end of the timeline"),
        }
    }
}

impl Error for SubtitleMiss {}

/// The request that places the cues of `text` as captions on a new
/// subtitle lane, the file's zero at the timeline's zero. `fresh` makes an
/// unused id from a prefix.
pub fn subtitle_request(
    text: &str,
    format: SubtitleFormat,
    fresh: &mut dyn FnMut(&str) -> String,
) -> Result<Request, SubtitleMiss> {
    let cues = match format {
        SubtitleFormat::Srt => parse_srt(text),
    }
    .map_err(SubtitleMiss::File)?;
    let track_id = TrackId::new(fresh("subtitles"));
    let cues = cues
        .into_iter()
        .map(|cue| placed(cue, fresh))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Request::AddSubtitles { track_id, cues })
}

fn placed(cue: Cue, fresh: &mut dyn FnMut(&str) -> String) -> Result<SubtitleCue, SubtitleMiss> {
    let start = cue
        .start_ms
        .checked_mul(1_000)
        .ok_or(SubtitleMiss::TooLate)?;
    let end = cue.end_ms.checked_mul(1_000).ok_or(SubtitleMiss::TooLate)?;
    Ok(SubtitleCue {
        segment_id: SegmentId::new(fresh("caption")),
        material_id: MaterialId::new(fresh("caption")),
        target: TimeRange::new(start, end - start),
        words: cue.words,
    })
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, TrackFlag};
    use cutlass_session::Session;

    use super::*;

    fn counter() -> impl FnMut(&str) -> String {
        let mut next = 0;
        move |prefix: &str| {
            next += 1;
            format!("{prefix}-{next}")
        }
    }

    #[test]
    fn an_srt_file_becomes_captions_on_a_new_subtitle_lane() {
        let text = "1\n00:00:01,000 --> 00:00:02,500\nHello there\n\n2\n00:00:03,000 --> 00:00:04,000\nBye\n";
        let request = subtitle_request(text, SubtitleFormat::Srt, &mut counter()).unwrap();
        let mut session = Session::open(Draft::empty("D"));
        session.apply(0, &[request]).unwrap();
        let draft = session.draft();
        let lane = draft
            .tracks
            .iter()
            .find(|track| track.id.as_str() == "subtitles-1")
            .expect("lane");
        assert_eq!(lane.flag, TrackFlag::TextSubtitle);
        let targets: Vec<_> = lane.segments.iter().map(|s| s.target.clone()).collect();
        assert_eq!(
            targets,
            [
                TimeRange::new(1_000_000, 1_500_000),
                TimeRange::new(3_000_000, 1_000_000)
            ]
        );
        let texts: Vec<_> = draft
            .materials
            .texts
            .iter()
            .map(|t| t.content.as_str())
            .collect();
        assert_eq!(texts, ["Hello there", "Bye"]);
        assert_eq!(draft.materials.texts[0].words.end_time, [750, 1_500]);
    }

    #[test]
    fn a_broken_or_too_late_file_makes_no_request() {
        assert_eq!(
            subtitle_request("nothing here", SubtitleFormat::Srt, &mut counter()),
            Err(SubtitleMiss::File(SubtitleError::BadTime { line: 1 }))
        );
        let late = "9999999999:00:00,000 --> 9999999999:00:01,000\nx\n";
        assert_eq!(
            subtitle_request(late, SubtitleFormat::Srt, &mut counter()),
            Err(SubtitleMiss::TooLate)
        );
    }

    #[test]
    fn the_format_comes_from_the_extension() {
        assert_eq!(
            SubtitleFormat::of_file("talk.SRT"),
            Some(SubtitleFormat::Srt)
        );
        assert_eq!(SubtitleFormat::of_file("talk.srt.txt"), None);
        assert_eq!(SubtitleFormat::of_file("srt"), None);
    }
}

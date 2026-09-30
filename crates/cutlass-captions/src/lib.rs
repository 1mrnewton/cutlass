//! Caption logic: words in, readable cues out.
//!
//! Everything here is pure — no file IO, no speech recognition, no GPU — so the
//! desktop, the CLI, the Python bindings, and the AI agent all reach the same
//! segmentation. Recognition lives in `cutlass-transcription`, the data lives in
//! `cutlass-models`, and this crate is the arithmetic between them.
//!
//! The two entry points both produce
//! [`CaptionCueSpec`](cutlass_models::CaptionCueSpec)s ready to hand to
//! `AddCaptionGroup`:
//!
//! - [`segment`] turns [`TimedWord`]s (from Whisper, or from anything) into cues
//!   that break where speech breaks.
//! - [`place_subtitles`] turns a parsed SRT/WebVTT file into the same thing.
//!
//! ```
//! use cutlass_captions::{Placement, SegmentOptions, TimedWord, segment};
//! use cutlass_models::Rational;
//!
//! let words = [
//!     TimedWord::new(0, 300, "Captions"),
//!     TimedWord::new(300, 700, "are"),
//!     TimedWord::new(700, 1_000, "clips."),
//! ];
//! let options = SegmentOptions::new(Placement::at_rate(Rational::FPS_30));
//! let cues = segment(&words, &options).unwrap();
//! assert_eq!(cues[0].text, "Captions are clips.");
//! ```

mod error;
mod format;
mod reflow;
mod segment;
mod srt;
mod subtitle;
mod timing;
mod vtt;

pub use error::CaptionError;
pub use reflow::{estimate_word_timings, overflow_cuts, rewrap, wrap};
pub use segment::{DEFAULT_PAUSE_BREAK_MS, SegmentOptions, TimedWord, segment};
pub use srt::{parse_srt, write_srt};
pub use subtitle::{
    ImportOptions, SubtitleCue, detect_format, parse_subtitles, place_subtitles,
    subtitles_from_clips,
};
pub use timing::Placement;
pub use vtt::{parse_vtt, write_vtt};

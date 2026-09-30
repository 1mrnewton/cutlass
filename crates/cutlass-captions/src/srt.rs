// --- SubRip (.srt) --------------------------------------------------------------------

use std::fmt::Write as _;

use crate::error::CaptionError;
use crate::format::{ARROW, cues_from_blocks, format_timestamp, numbered_lines, scan_cues};
use crate::subtitle::SubtitleCue;

#[cfg(test)]
mod tests;

/// Read a SubRip file.
///
/// Forgiving about everything that does not change meaning — a byte-order mark,
/// CRLF, a missing or out-of-order cue number, stray blank lines, `<i>` markup
/// — and strict about the timing line. Errors carry the 1-based line number so
/// the UI can say where the file went wrong.
pub fn parse_srt(input: &str) -> Result<Vec<SubtitleCue>, CaptionError> {
    let lines = numbered_lines(input);
    cues_from_blocks(scan_cues(&lines, false)?)
}

/// Write a SubRip file: cues numbered from one, comma before the milliseconds.
pub fn write_srt(cues: &[SubtitleCue]) -> String {
    let mut out = String::new();
    for (index, cue) in cues.iter().enumerate() {
        let _ = writeln!(out, "{}", index + 1);
        let _ = writeln!(
            out,
            "{} {ARROW} {}",
            format_timestamp(cue.start_ms, ','),
            format_timestamp(cue.end_ms.max(cue.start_ms), ',')
        );
        out.push_str(cue.text.trim());
        out.push_str("\n\n");
    }
    out
}

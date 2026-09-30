// --- WebVTT (.vtt) --------------------------------------------------------------------

use std::fmt::Write as _;

use crate::error::CaptionError;
use crate::format::{ARROW, cues_from_blocks, format_timestamp, numbered_lines, scan_cues};
use crate::subtitle::SubtitleCue;

#[cfg(test)]
mod tests;

/// The header every WebVTT file starts with.
const HEADER: &str = "WEBVTT";

/// Read a WebVTT file.
///
/// `NOTE`, `STYLE`, and `REGION` blocks are skipped, cue settings after the end
/// timestamp (`align:start`, `line:90%`) are ignored, and inline markup —
/// including the `<00:00:01.000>` karaoke tags — is stripped to the text that
/// will render.
///
/// The `WEBVTT` header is required: it is the one thing that distinguishes this
/// from SubRip, and a file without it is either not WebVTT or is corrupt, both
/// of which the user wants told rather than half-imported.
pub fn parse_vtt(input: &str) -> Result<Vec<SubtitleCue>, CaptionError> {
    let lines = numbered_lines(input);
    let header = lines
        .iter()
        .find(|(_, line)| !line.trim().is_empty())
        .ok_or_else(|| CaptionError::parse(1, "the file is empty"))?;
    if !header.1.trim_start().starts_with(HEADER) {
        return Err(CaptionError::parse(
            header.0,
            format!("expected a '{HEADER}' header"),
        ));
    }
    let body = &lines[lines
        .iter()
        .position(|line| line.0 == header.0)
        .unwrap_or(0)
        + 1..];
    cues_from_blocks(scan_cues(body, true)?)
}

/// Write a WebVTT file: `WEBVTT` header, cue identifiers from one, dot before
/// the milliseconds.
pub fn write_vtt(cues: &[SubtitleCue]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for (index, cue) in cues.iter().enumerate() {
        let _ = writeln!(out, "{}", index + 1);
        let _ = writeln!(
            out,
            "{} {ARROW} {}",
            format_timestamp(cue.start_ms, '.'),
            format_timestamp(cue.end_ms.max(cue.start_ms), '.')
        );
        out.push_str(cue.text.trim());
        out.push_str("\n\n");
    }
    out
}

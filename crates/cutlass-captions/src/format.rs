// --- Shared subtitle-file syntax ------------------------------------------------------
//
// Timestamps and inline markup are the two things SRT and WebVTT genuinely
// share, so both parsers read them the same way and neither guesses.

use crate::error::CaptionError;
use crate::subtitle::SubtitleCue;

/// The arrow separating a cue's two timestamps, in both formats.
pub(crate) const ARROW: &str = "-->";

/// Split a timing line into its start and end halves.
///
/// The end half keeps any trailing WebVTT cue settings; callers take the first
/// whitespace-delimited token from it.
pub(crate) fn split_timing(line: &str) -> Option<(&str, &str)> {
    let (start, end) = line.split_once(ARROW)?;
    Some((start.trim(), end.trim()))
}

/// Parse `HH:MM:SS,mmm`, `HH:MM:SS.mmm`, or the hour-less `MM:SS.mmm` into
/// milliseconds.
///
/// Both separators are accepted for both formats: files in the wild mix them,
/// and rejecting a comma in a `.vtt` helps nobody. Fractions of any length are
/// read to millisecond precision.
pub(crate) fn parse_timestamp(text: &str) -> Option<u32> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let (whole, fraction) = match text.split_once([',', '.']) {
        Some((whole, fraction)) => (whole, Some(fraction)),
        None => (text, None),
    };

    let mut seconds = 0u64;
    let mut parts = 0;
    for part in whole.split(':') {
        let part = part.trim();
        if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        seconds = seconds
            .checked_mul(60)?
            .checked_add(part.parse::<u64>().ok()?)?;
        parts += 1;
        if parts > 3 {
            return None;
        }
    }

    let millis = match fraction {
        Some(fraction) => {
            let digits: &str = fraction.trim();
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            // Pad or truncate to exactly three digits.
            let mut value = 0u64;
            for index in 0..3 {
                let digit = digits
                    .as_bytes()
                    .get(index)
                    .map_or(0, |byte| u64::from(*byte - b'0'));
                value = value * 10 + digit;
            }
            value
        }
        None => 0,
    };

    let total = seconds.checked_mul(1_000)?.checked_add(millis)?;
    u32::try_from(total).ok()
}

/// Render milliseconds as `HH:MM:SS<separator>mmm`.
pub(crate) fn format_timestamp(ms: u32, separator: char) -> String {
    let (seconds, millis) = (ms / 1_000, ms % 1_000);
    let (minutes, seconds) = (seconds / 60, seconds % 60);
    let (hours, minutes) = (minutes / 60, minutes % 60);
    format!("{hours:02}:{minutes:02}:{seconds:02}{separator}{millis:03}")
}

/// Strip inline styling so the cue text is what the caption will actually
/// render.
///
/// Removes `<i>`/`<c.yellow>`/`<v Name>`-style tags (including WebVTT's
/// `<00:00:01.000>` karaoke cues) and `{\an8}` ASS override blocks, then
/// decodes the handful of entities subtitle files use. Unclosed brackets are
/// kept verbatim rather than swallowing the rest of the line.
pub(crate) fn strip_markup(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find(['<', '{']) {
        let close = if rest.as_bytes()[open] == b'<' {
            '>'
        } else {
            '}'
        };
        let Some(end) = rest[open..].find(close) else {
            break;
        };
        out.push_str(&rest[..open]);
        rest = &rest[open + end + close.len_utf8()..];
    }
    out.push_str(rest);
    decode_entities(&out)
}

fn decode_entities(text: &str) -> String {
    if !text.contains('&') {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('&') {
        out.push_str(&rest[..start]);
        let tail = &rest[start..];
        let matched = [
            ("&amp;", "&"),
            ("&lt;", "<"),
            ("&gt;", ">"),
            ("&quot;", "\""),
            ("&#39;", "'"),
            ("&apos;", "'"),
            ("&nbsp;", " "),
            ("&lrm;", ""),
            ("&rlm;", ""),
        ]
        .into_iter()
        .find(|(entity, _)| tail.starts_with(entity));
        match matched {
            Some((entity, replacement)) => {
                out.push_str(replacement);
                rest = &tail[entity.len()..];
            }
            None => {
                out.push('&');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// The file's lines with 1-based numbers, byte-order mark removed.
pub(crate) fn numbered_lines(input: &str) -> Vec<(usize, &str)> {
    input
        .trim_start_matches('\u{feff}')
        .lines()
        .enumerate()
        .map(|(index, line)| (index + 1, line))
        .collect()
}

/// One cue block as it appeared in the file, before its timing is read.
pub(crate) struct RawCue {
    /// 1-based line number of the timing line, for error reporting.
    pub(crate) number: usize,
    pub(crate) timing: String,
    /// The cue's text lines, joined with `\n`, markup stripped.
    pub(crate) text: String,
}

/// Split a subtitle file into cue blocks.
///
/// Both formats are blank-line-separated blocks of `[identifier] / timing /
/// text`, so the block walk is shared and each format only owns its header and
/// its metadata blocks. `skip_metadata` drops WebVTT's `NOTE`, `STYLE`, and
/// `REGION` blocks.
pub(crate) fn scan_cues(
    lines: &[(usize, &str)],
    skip_metadata: bool,
) -> Result<Vec<RawCue>, CaptionError> {
    let mut cues = Vec::new();
    let mut index = 0;

    while index < lines.len() {
        let (number, line) = lines[index];
        if line.trim().is_empty() {
            index += 1;
            continue;
        }
        if skip_metadata && is_metadata_block(line) {
            index += 1;
            while lines.get(index).is_some_and(|(_, l)| !l.trim().is_empty()) {
                index += 1;
            }
            continue;
        }

        // The identifier line (SRT's cue number, WebVTT's optional name) is
        // optional; the timing line is not.
        let mut timing = line;
        let mut timing_number = number;
        if !timing.contains(ARROW) {
            index += 1;
            match lines.get(index) {
                Some(&(next_number, next)) if next.contains(ARROW) => {
                    timing = next;
                    timing_number = next_number;
                }
                _ => {
                    return Err(CaptionError::parse(
                        number,
                        format!("expected a '{ARROW}' timing line after '{}'", line.trim()),
                    ));
                }
            }
        }
        index += 1;

        let mut text = String::new();
        while let Some(&(_, line)) = lines.get(index) {
            if line.trim().is_empty() {
                break;
            }
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(line.trim());
            index += 1;
        }

        cues.push(RawCue {
            number: timing_number,
            timing: timing.to_owned(),
            text: strip_markup(&text),
        });
    }
    Ok(cues)
}

/// WebVTT blocks that are not cues.
fn is_metadata_block(line: &str) -> bool {
    let line = line.trim_start();
    ["NOTE", "STYLE", "REGION"]
        .iter()
        .any(|keyword| line == *keyword || line.starts_with(&format!("{keyword} ")))
}

/// Read a timing line into a millisecond span.
///
/// The one part of a subtitle file worth being strict about: a misread
/// timestamp puts captions somewhere plausible and wrong, where a rejected file
/// sends the user back to fix it.
pub(crate) fn parse_timing(number: usize, line: &str) -> Result<(u32, u32), CaptionError> {
    let Some((start, end)) = split_timing(line) else {
        return Err(CaptionError::parse(
            number,
            format!("missing '{ARROW}' in the timing"),
        ));
    };
    // The end half may carry WebVTT cue settings; the timestamp is its first
    // token.
    let end = end.split_whitespace().next().unwrap_or_default();
    let (Some(start_ms), Some(end_ms)) = (parse_timestamp(start), parse_timestamp(end)) else {
        return Err(CaptionError::parse(
            number,
            format!("unreadable timestamp in '{}'", line.trim()),
        ));
    };
    if end_ms < start_ms {
        return Err(CaptionError::parse(number, "cue ends before it starts"));
    }
    Ok((start_ms, end_ms))
}

/// Assemble the cue blocks of an already-scanned file.
pub(crate) fn cues_from_blocks(blocks: Vec<RawCue>) -> Result<Vec<SubtitleCue>, CaptionError> {
    let mut cues = Vec::with_capacity(blocks.len());
    for block in blocks {
        let (start_ms, end_ms) = parse_timing(block.number, &block.timing)?;
        // An empty cue renders nothing and cannot be selected; files do carry
        // them, so drop rather than reject.
        if block.text.trim().is_empty() {
            continue;
        }
        cues.push(SubtitleCue {
            start_ms,
            end_ms,
            text: block.text,
        });
    }
    Ok(cues)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_parse_with_either_separator() {
        assert_eq!(parse_timestamp("00:00:01,500"), Some(1_500));
        assert_eq!(parse_timestamp("00:00:01.500"), Some(1_500));
        assert_eq!(parse_timestamp(" 01:02:03,004 "), Some(3_723_004));
    }

    #[test]
    fn timestamps_parse_without_hours_or_fractions() {
        assert_eq!(parse_timestamp("02:03.500"), Some(123_500));
        assert_eq!(parse_timestamp("00:00:07"), Some(7_000));
    }

    #[test]
    fn fractions_pad_and_truncate_to_milliseconds() {
        assert_eq!(parse_timestamp("00:00:00.5"), Some(500));
        assert_eq!(parse_timestamp("00:00:00.05"), Some(50));
        assert_eq!(parse_timestamp("00:00:00.123456"), Some(123));
    }

    #[test]
    fn malformed_timestamps_are_rejected() {
        for bad in [
            "",
            "abc",
            "00:xx:01,000",
            "00:00:01,abc",
            "1:2:3:4:5",
            "00::01,000",
            "-00:00:01,000",
        ] {
            assert_eq!(parse_timestamp(bad), None, "{bad:?} should not parse");
        }
    }

    #[test]
    fn timestamps_round_trip() {
        for ms in [0_u32, 1, 999, 1_500, 3_723_004] {
            let text = format_timestamp(ms, ',');
            assert_eq!(parse_timestamp(&text), Some(ms), "{text}");
        }
    }

    #[test]
    fn format_pads_every_field() {
        assert_eq!(format_timestamp(0, ','), "00:00:00,000");
        assert_eq!(format_timestamp(61_007, '.'), "00:01:01.007");
    }

    #[test]
    fn markup_is_stripped_and_entities_decoded() {
        assert_eq!(strip_markup("<i>hello</i>"), "hello");
        assert_eq!(strip_markup("<c.yellow>hi</c> there"), "hi there");
        assert_eq!(strip_markup("{\\an8}top"), "top");
        assert_eq!(strip_markup("<00:00:01.000>word"), "word");
        assert_eq!(strip_markup("a &amp; b &lt;c&gt;"), "a & b <c>");
    }

    #[test]
    fn an_unclosed_tag_is_kept_verbatim() {
        assert_eq!(strip_markup("2 < 3"), "2 < 3");
        assert_eq!(strip_markup("<i>a</i> 2 < 3"), "a 2 < 3");
    }

    #[test]
    fn an_unknown_entity_survives() {
        assert_eq!(strip_markup("Q&A &unknown;"), "Q&A &unknown;");
    }

    #[test]
    fn timing_lines_split_on_the_arrow() {
        assert_eq!(
            split_timing("00:00:01,000 --> 00:00:02,000"),
            Some(("00:00:01,000", "00:00:02,000"))
        );
        assert_eq!(split_timing("00:00:01,000"), None);
    }

    #[test]
    fn numbered_lines_drop_the_byte_order_mark() {
        let lines = numbered_lines("\u{feff}1\r\ntext");
        assert_eq!(lines, [(1, "1"), (2, "text")]);
    }
}

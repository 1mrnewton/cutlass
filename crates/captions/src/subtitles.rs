//! Subtitle files read as timed captions.
//!
//! A cue is a stretch of the timeline and the words said in it. Cue times
//! are milliseconds from the start of the file; word times are milliseconds
//! from the start of their cue, as a caption stores them. A cue's words
//! share its time by letters, since a subtitle file times lines, not words.

use std::error::Error;
use std::fmt;

use cutlass_model::WordColumns;

/// The most cues one file may bring in.
pub const MAX_CUES: usize = 20_000;

/// One caption a subtitle file brings in.
#[derive(Clone, Debug, PartialEq)]
pub struct Cue {
    /// Milliseconds from the start of the file.
    pub start_ms: i64,
    /// Milliseconds from the start of the file, after `start_ms`.
    pub end_ms: i64,
    /// Word times are milliseconds from `start_ms`. A line break between
    /// lines of the cue is a `"\n"` word with no length.
    pub words: WordColumns,
}

/// Why a subtitle file was refused. Lines count from 1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubtitleError {
    /// No cue with words and a length.
    Empty,
    /// A cue has no time line, or its time does not read.
    BadTime { line: usize },
    /// A cue ends before it starts.
    Backwards { line: usize },
    /// More than [`MAX_CUES`] cues.
    TooMany,
}

impl fmt::Display for SubtitleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "the subtitle file has no timed lines"),
            Self::BadTime { line } => write!(f, "line {line} is not a subtitle time"),
            Self::Backwards { line } => write!(f, "the cue on line {line} ends before it starts"),
            Self::TooMany => write!(f, "the subtitle file has more than {MAX_CUES} cues"),
        }
    }
}

impl Error for SubtitleError {}

/// Read an SRT file.
///
/// Blocks are separated by blank lines. A block is an optional number, a
/// time line `00:00:01,000 --> 00:00:02,500` (a dot also separates the
/// milliseconds, and anything after the end time is ignored), and its text.
/// Markup in `<...>` and `{...}` is dropped. A cue with no words or no
/// length is skipped. Cues come back in time order; a cue that runs into the
/// next one ends where the next starts.
pub fn parse_srt(text: &str) -> Result<Vec<Cue>, SubtitleError> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut lines = text.lines().enumerate().peekable();
    let mut cues = Vec::new();
    loop {
        while lines.next_if(|(_, line)| line.trim().is_empty()).is_some() {}
        let Some((number, mut first)) = lines.next() else {
            break;
        };
        let mut line_number = number + 1;
        if !first.contains("-->") {
            // The cue's number; the time is on the next line.
            match lines.next() {
                Some((number, line)) => {
                    first = line;
                    line_number = number + 1;
                }
                None => return Err(SubtitleError::BadTime { line: line_number }),
            }
        }
        let (start_ms, end_ms) =
            srt_times(first).ok_or(SubtitleError::BadTime { line: line_number })?;
        if end_ms < start_ms {
            return Err(SubtitleError::Backwards { line: line_number });
        }
        let mut body = Vec::new();
        while let Some((_, line)) = lines.next_if(|(_, line)| !line.trim().is_empty()) {
            body.push(line);
        }
        push_cue(&mut cues, start_ms, end_ms, &body)?;
    }
    finish(cues)
}

fn push_cue(
    cues: &mut Vec<Cue>,
    start_ms: i64,
    end_ms: i64,
    lines: &[&str],
) -> Result<(), SubtitleError> {
    let lines: Vec<Vec<String>> = lines
        .iter()
        .map(|line| {
            strip_markup(line)
                .split_whitespace()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .filter(|words| !words.is_empty())
        .collect();
    if lines.is_empty() || end_ms <= start_ms {
        return Ok(());
    }
    if cues.len() == MAX_CUES {
        return Err(SubtitleError::TooMany);
    }
    cues.push(Cue {
        start_ms,
        end_ms,
        words: WordColumns::default(),
    });
    if let Some(cue) = cues.last_mut() {
        cue.words = words_by_letters(&lines, end_ms - start_ms);
    }
    Ok(())
}

/// Sort by start and end each cue where the next begins.
fn finish(mut cues: Vec<Cue>) -> Result<Vec<Cue>, SubtitleError> {
    cues.sort_by_key(|cue| (cue.start_ms, cue.end_ms));
    let starts: Vec<i64> = cues.iter().map(|cue| cue.start_ms).skip(1).collect();
    for (cue, next) in cues.iter_mut().zip(starts) {
        if next < cue.end_ms {
            let length = next - cue.start_ms;
            let lines = lines_of(&cue.words);
            cue.end_ms = next;
            cue.words = words_by_letters(&lines, length);
        }
    }
    cues.retain(|cue| cue.end_ms > cue.start_ms);
    if cues.is_empty() {
        return Err(SubtitleError::Empty);
    }
    Ok(cues)
}

/// The words of each line, back out of `words`.
fn lines_of(words: &WordColumns) -> Vec<Vec<String>> {
    let mut lines = vec![Vec::new()];
    for word in &words.text {
        if word == "\n" {
            lines.push(Vec::new());
        } else if let Some(line) = lines.last_mut() {
            line.push(word.clone());
        }
    }
    lines
}

/// Lay `lines` over `length_ms`, each word as long as its letters.
fn words_by_letters(lines: &[Vec<String>], length_ms: i64) -> WordColumns {
    let letters: i64 = lines
        .iter()
        .flatten()
        .map(|word| i64::try_from(word.chars().count()).unwrap_or(i64::MAX))
        .fold(0, i64::saturating_add)
        .max(1);
    let at = |done: i64| {
        let scaled = i128::from(length_ms) * i128::from(done) / i128::from(letters);
        i64::try_from(scaled).unwrap_or(length_ms)
    };
    let mut words = WordColumns::default();
    let mut done = 0_i64;
    for (index, line) in lines.iter().enumerate() {
        if index > 0 {
            let here = at(done);
            words.start_time.push(here);
            words.end_time.push(here);
            words.text.push("\n".to_owned());
        }
        for word in line {
            let start = at(done);
            done = done.saturating_add(i64::try_from(word.chars().count()).unwrap_or(i64::MAX));
            words.start_time.push(start);
            words.end_time.push(at(done));
            words.text.push(word.clone());
        }
    }
    words
}

/// Drop `<i>`-style tags and `{\an8}`-style overrides.
fn strip_markup(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut closing = None;
    for ch in line.chars() {
        match closing {
            Some(end) if ch == end => closing = None,
            Some(_) => {}
            None if ch == '<' => closing = Some('>'),
            None if ch == '{' => closing = Some('}'),
            None => out.push(ch),
        }
    }
    out
}

fn srt_times(line: &str) -> Option<(i64, i64)> {
    let (start, rest) = line.split_once("-->")?;
    let end = rest.split_whitespace().next()?;
    Some((srt_stamp(start.trim())?, srt_stamp(end)?))
}

/// `HH:MM:SS,mmm`, with a comma or a dot before the milliseconds. Hours may
/// have any number of digits; minutes and seconds stay under 60.
fn srt_stamp(stamp: &str) -> Option<i64> {
    let (clock, millis) = stamp.split_once([',', '.'])?;
    let mut parts = clock.split(':');
    let hours = number(parts.next()?)?;
    let minutes = number(parts.next()?)?;
    let seconds = number(parts.next()?)?;
    if parts.next().is_some() || minutes >= 60 || seconds >= 60 {
        return None;
    }
    if millis.is_empty() || millis.len() > 3 {
        return None;
    }
    // "5" is 500 ms and "05" is 50 ms, as a decimal fraction.
    let millis = number(millis)? * 10_i64.pow(3 - u32::try_from(millis.len()).ok()?);
    hours
        .checked_mul(3_600_000)?
        .checked_add(minutes * 60_000)?
        .checked_add(seconds * 1_000)?
        .checked_add(millis)
}

fn number(digits: &str) -> Option<i64> {
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO: &str = "1\n00:00:01,000 --> 00:00:02,000\nHello <i>big</i> world\n\n2\n00:00:03,500 --> 00:00:04,500 X1:0 X2:10\n{\\an8}Second\nline here\n";

    #[test]
    fn cues_keep_their_times_and_words_share_them_by_letters() {
        let cues = parse_srt(TWO).unwrap();
        assert_eq!(cues.len(), 2);
        assert_eq!((cues[0].start_ms, cues[0].end_ms), (1_000, 2_000));
        assert_eq!(cues[0].words.text, ["Hello", "big", "world"]);
        // 13 letters over 1000 ms: 5, 3 and 5 of them.
        assert_eq!(cues[0].words.start_time, [0, 384, 615]);
        assert_eq!(cues[0].words.end_time, [384, 615, 1_000]);
        assert_eq!((cues[1].start_ms, cues[1].end_ms), (3_500, 4_500));
        assert_eq!(cues[1].words.text, ["Second", "\n", "line", "here"]);
        assert_eq!(cues[1].words.start_time[1], cues[1].words.end_time[1]);
        assert_eq!(cues[1].words.end_time.last(), Some(&1_000));
    }

    #[test]
    fn windows_lines_a_mark_dots_and_no_numbers_read_too() {
        let text = "\u{feff}00:00:00.5 --> 00:00:01.25\r\nOne\r\n\r\n\r\n10:00:00,000 --> 10:00:01,000\r\nTwo\r\n";
        let cues = parse_srt(text).unwrap();
        assert_eq!((cues[0].start_ms, cues[0].end_ms), (500, 1_250));
        assert_eq!(cues[1].start_ms, 36_000_000);
        assert_eq!(cues[1].words.text, ["Two"]);
    }

    #[test]
    fn out_of_order_and_overlapping_cues_are_sorted_and_cut() {
        let text = "00:00:05,000 --> 00:00:06,000\nlast\n\n00:00:01,000 --> 00:00:04,000\nfirst one\n\n00:00:03,000 --> 00:00:05,500\nmiddle\n\n00:00:03,000 --> 00:00:03,000\nnothing\n\n00:00:07,000 --> 00:00:08,000\n<b></b>\n";
        let cues = parse_srt(text).unwrap();
        let times: Vec<_> = cues.iter().map(|cue| (cue.start_ms, cue.end_ms)).collect();
        assert_eq!(times, [(1_000, 3_000), (3_000, 5_000), (5_000, 6_000)]);
        assert_eq!(cues[0].words.end_time, [1_250, 2_000]);
        assert_eq!(cues[1].words.text, ["middle"]);
    }

    #[test]
    fn broken_or_hostile_files_are_refused_with_the_line() {
        assert_eq!(parse_srt(""), Err(SubtitleError::Empty));
        assert_eq!(parse_srt("\n\n  \n"), Err(SubtitleError::Empty));
        assert_eq!(parse_srt("1\n"), Err(SubtitleError::BadTime { line: 1 }));
        assert_eq!(
            parse_srt("1\nhello\n"),
            Err(SubtitleError::BadTime { line: 2 })
        );
        assert_eq!(
            parse_srt("\n\n00:00:61,000 --> 00:01:00,000\nx\n"),
            Err(SubtitleError::BadTime { line: 3 })
        );
        assert_eq!(
            parse_srt("00:00:02,000 --> 00:00:01,000\nx\n"),
            Err(SubtitleError::Backwards { line: 1 })
        );
        assert_eq!(
            parse_srt("99999999999999999:00:00,000 --> 99999999999999999:00:01,000\nx\n"),
            Err(SubtitleError::BadTime { line: 1 })
        );
        assert_eq!(
            parse_srt("-1:00:00,000 --> 00:00:01,000\nx\n"),
            Err(SubtitleError::BadTime { line: 1 })
        );
        assert_eq!(
            parse_srt("00:00:00,0000 --> 00:00:01,000\nx\n"),
            Err(SubtitleError::BadTime { line: 1 })
        );
        let mut many = String::new();
        for index in 0..=MAX_CUES {
            let start = index * 10;
            many.push_str(&format!(
                "00:{:02}:{:02},{:03} --> 00:{:02}:{:02},{:03}\nx\n\n",
                start / 60_000,
                start / 1_000 % 60,
                start % 1_000,
                (start + 5) / 60_000,
                (start + 5) / 1_000 % 60,
                (start + 5) % 1_000,
            ));
        }
        assert_eq!(parse_srt(&many), Err(SubtitleError::TooMany));
    }

    #[test]
    fn a_long_file_reads_in_one_pass() {
        let mut text = String::new();
        for index in 0..MAX_CUES {
            let start = index * 10;
            text.push_str(&format!(
                "{index}\n00:{:02}:{:02},{:03} --> 00:{:02}:{:02},{:03}\nsome words here\n\n",
                start / 60_000,
                start / 1_000 % 60,
                start % 1_000,
                (start + 5) / 60_000,
                (start + 5) / 1_000 % 60,
                (start + 5) % 1_000,
            ));
        }
        let cues = parse_srt(&text).unwrap();
        assert_eq!(cues.len(), MAX_CUES);
        assert!(
            cues.windows(2)
                .all(|pair| pair[0].end_ms <= pair[1].start_ms)
        );
    }
}

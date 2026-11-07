//! Heard words split into caption lines, one caption clip each.
//!
//! The reference editor makes a caption per utterance the recognizer
//! returns; how it splits a long utterance further is native code the docs
//! do not show. Cutlass breaks a line after a word that ends a sentence, at
//! a pause between words, and before a word that would make the line too
//! long. A line is never broken inside a word.

use cutlass_model::WordColumns;

/// A gap between words this long, in milliseconds, starts a new line.
pub const LINE_PAUSE_MS: i64 = 500;
/// The most letters a line holds, spaces counted. A single longer word
/// still takes a line of its own.
pub const LINE_LETTERS: usize = 48;
/// The most words a line holds.
pub const LINE_WORDS: usize = 10;
/// The shortest a line stays up, in milliseconds, when the next line or
/// the clip's end leaves room.
pub const LINE_SHORTEST_MS: i64 = 600;

/// One caption line: where it sits on the clip's clock, and its words on
/// its own clock.
#[derive(Clone, Debug, PartialEq)]
pub struct HeardLine {
    pub start_ms: i64,
    pub end_ms: i64,
    pub words: WordColumns,
}

/// `words`, timed in milliseconds from the start of a clip `length_ms`
/// long, split into lines that sit back to back on the clip. A line runs
/// until the next one starts, unless a pause comes between them, when it
/// ends with its last word; it stays up at least [`LINE_SHORTEST_MS`] when
/// there is room. Words whose times run backwards, as a
/// reversed clip's do, stay one line over the whole clip.
#[must_use]
pub fn heard_lines(words: &WordColumns, length_ms: i64) -> Vec<HeardLine> {
    let count = words
        .text
        .len()
        .min(words.start_time.len())
        .min(words.end_time.len());
    if count == 0 || length_ms <= 0 {
        return Vec::new();
    }
    let forward = words.start_time[..count]
        .windows(2)
        .all(|pair| pair[0] <= pair[1]);
    if !forward {
        return vec![HeardLine {
            start_ms: 0,
            end_ms: length_ms,
            words: line_words(words, 0..count, 0, length_ms),
        }];
    }
    let mut groups = groups(words, count);
    // A line too short to hold time joins the one before it, or the first
    // the one after it.
    loop {
        let spans = spans(words, &groups, length_ms);
        let Some(index) = spans.iter().position(|(start, end)| end <= start) else {
            return groups
                .into_iter()
                .zip(spans)
                .map(|(range, (start_ms, end_ms))| HeardLine {
                    start_ms,
                    end_ms,
                    words: line_words(words, range, start_ms, end_ms - start_ms),
                })
                .collect();
        };
        if groups.len() == 1 {
            // Every word sits at the clip's end: the line takes the clip.
            return vec![HeardLine {
                start_ms: 0,
                end_ms: length_ms,
                words: line_words(words, 0..count, 0, length_ms),
            }];
        }
        let (keep, gone) = if index == 0 {
            (0, 1)
        } else {
            (index - 1, index)
        };
        let end = groups[gone].end;
        groups[keep].end = end;
        groups.remove(gone);
    }
}

/// The word ranges of each line.
fn groups(words: &WordColumns, count: usize) -> Vec<std::ops::Range<usize>> {
    let mut groups = Vec::new();
    let mut first = 0;
    let mut letters = 0;
    for index in 0..count {
        let word = words.text[index].trim();
        let size = word.chars().count();
        if index > first {
            let before = index - 1;
            let pause = words.start_time[index].saturating_sub(words.end_time[before]);
            if ends_sentence(&words.text[before])
                || pause >= LINE_PAUSE_MS
                || letters + 1 + size > LINE_LETTERS
                || index - first >= LINE_WORDS
            {
                groups.push(first..index);
                first = index;
                letters = 0;
            }
        }
        letters += if index > first { 1 + size } else { size };
    }
    groups.push(first..count);
    groups
}

/// Where each line sits on the clip, in milliseconds.
fn spans(
    words: &WordColumns,
    groups: &[std::ops::Range<usize>],
    length_ms: i64,
) -> Vec<(i64, i64)> {
    groups
        .iter()
        .enumerate()
        .map(|(index, range)| {
            let start = words.start_time[range.start].clamp(0, length_ms);
            let last = words.end_time[range.end - 1].clamp(start, length_ms);
            let room = match groups.get(index + 1) {
                Some(next) => words.start_time[next.start].clamp(start, length_ms),
                None => length_ms,
            };
            let end = if room - last < LINE_PAUSE_MS && room < length_ms {
                room
            } else {
                last
            };
            // A short line stays up long enough to read, if it has room.
            let end = end.max((start + LINE_SHORTEST_MS).min(room));
            (start, end)
        })
        .collect()
}

/// Words `range`, moved onto a line that starts at `start_ms` and is
/// `length_ms` long.
fn line_words(
    words: &WordColumns,
    range: std::ops::Range<usize>,
    start_ms: i64,
    length_ms: i64,
) -> WordColumns {
    let mut line = WordColumns::default();
    for index in range {
        let start = (words.start_time[index] - start_ms).clamp(0, length_ms);
        let end = (words.end_time[index] - start_ms).clamp(start, length_ms);
        line.start_time.push(start);
        line.end_time.push(end);
        line.text.push(words.text[index].clone());
    }
    line
}

/// The word ends a sentence: its last letter, past any closing quote or
/// bracket, is a full stop, a question or an exclamation mark.
fn ends_sentence(word: &str) -> bool {
    word.trim_end_matches(['"', '\'', '”', '’', ')', ']', '»'])
        .ends_with(['.', '!', '?', '…', '。', '！', '？'])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heard(words: &[(&str, i64, i64)]) -> WordColumns {
        WordColumns {
            start_time: words.iter().map(|word| word.1).collect(),
            end_time: words.iter().map(|word| word.2).collect(),
            text: words.iter().map(|word| word.0.to_owned()).collect(),
            rest: Default::default(),
        }
    }

    fn texts(lines: &[HeardLine]) -> Vec<String> {
        lines.iter().map(|line| line.words.text.join(" ")).collect()
    }

    #[test]
    fn a_long_transcript_splits_at_sentences_pauses_and_length() {
        let script = "This is good. And it's actually good so I'm gonna show you the \
                      benchmarks in a second and then we will look at what it costs";
        let mut words = Vec::new();
        let mut at = 0;
        for (index, word) in script.split_whitespace().enumerate() {
            // A long pause before "and then".
            if index == 17 {
                at += 900;
            }
            words.push((word, at, at + 250));
            at += 300;
        }
        let words = heard(&words);
        let lines = heard_lines(&words, 20_000);
        assert_eq!(
            texts(&lines),
            [
                "This is good.",
                "And it's actually good so I'm gonna show you the",
                "benchmarks in a second",
                "and then we will look at what it costs",
            ]
        );
        for line in &lines {
            assert!(line.words.text.join(" ").chars().count() <= LINE_LETTERS);
            assert!(line.words.text.len() <= LINE_WORDS);
            assert_eq!(line.words.start_time[0], 0, "a line starts with its word");
            assert!(
                line.words
                    .end_time
                    .iter()
                    .all(|end| *end <= line.end_ms - line.start_ms)
            );
        }
        // Lines sit back to back, except across the pause.
        assert_eq!(lines[0].start_ms, 0);
        assert_eq!(lines[0].end_ms, lines[1].start_ms);
        assert_eq!(lines[1].end_ms, lines[2].start_ms);
        assert_eq!(lines[2].end_ms, 16 * 300 + 250);
        assert_eq!(lines[3].start_ms, 17 * 300 + 900);
        assert_eq!(lines[3].end_ms, *words.end_time.last().unwrap());
        // Every word is kept once, in order, on the clip's clock.
        let mut back = Vec::new();
        for line in &lines {
            for (index, word) in line.words.text.iter().enumerate() {
                back.push((word.clone(), line.words.start_time[index] + line.start_ms));
            }
        }
        let original: Vec<(String, i64)> = words
            .text
            .iter()
            .cloned()
            .zip(words.start_time.iter().copied())
            .collect();
        assert_eq!(back, original);
    }

    #[test]
    fn a_word_too_long_for_a_line_keeps_a_line_of_its_own() {
        let long = "a".repeat(LINE_LETTERS + 5);
        let words = heard(&[("hi", 0, 100), (&long, 100, 200), ("there", 200, 300)]);
        assert_eq!(
            texts(&heard_lines(&words, 1_000)),
            ["hi", long.as_str(), "there"]
        );
    }

    #[test]
    fn a_closing_quote_still_ends_a_sentence() {
        assert!(ends_sentence("good.\u{201d}"));
        assert!(ends_sentence("why?)"));
        assert!(!ends_sentence("Mr"));
        assert!(!ends_sentence("one,"));
    }

    #[test]
    fn reversed_or_crowded_words_and_no_words_still_make_sense() {
        let backwards = heard(&[("b.", 500, 900), ("a.", 0, 400)]);
        let lines = heard_lines(&backwards, 1_000);
        assert_eq!(lines.len(), 1);
        assert_eq!((lines[0].start_ms, lines[0].end_ms), (0, 1_000));
        // Words at the very end of the clip cannot hold a line apart.
        let crowded = heard(&[
            ("one.", 0, 400),
            ("two.", 1_000, 1_000),
            ("three.", 1_000, 1_000),
        ]);
        let lines = heard_lines(&crowded, 1_000);
        assert!(lines.iter().all(|line| line.end_ms > line.start_ms));
        assert_eq!(texts(&lines).join(" "), "one. two. three.");
        assert!(heard_lines(&WordColumns::default(), 1_000).is_empty());
        assert!(heard_lines(&crowded, 0).is_empty());
    }

    #[test]
    fn a_short_line_stays_up_long_enough_to_read_where_there_is_room() {
        let words = heard(&[
            ("Hi.", 0, 150),
            ("Bye.", 2_000, 2_100),
            ("Now.", 2_300, 2_400),
        ]);
        let lines = heard_lines(&words, 2_500);
        let spans: Vec<(i64, i64)> = lines
            .iter()
            .map(|line| (line.start_ms, line.end_ms))
            .collect();
        // "Bye." has only until "Now." starts; "Now." only until the clip ends.
        assert_eq!(
            spans,
            [(0, LINE_SHORTEST_MS), (2_000, 2_300), (2_300, 2_500)]
        );
    }
}

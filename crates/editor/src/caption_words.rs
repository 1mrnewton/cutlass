//! Edits to a caption's words that keep their times: a word typed as two
//! is split across its time, two words are joined over both, and a line
//! break goes in before a word or comes out.
//!
//! A line break is the captions' own marker, a word that is exactly
//! `"\n"`, timed where the next word starts. Columns this model does not
//! name are dropped, as their rows would no longer line up.

use cutlass_model::WordColumns;

/// The line break marker.
pub const LINE_BREAK: &str = "\n";

fn row(words: &WordColumns, index: usize) -> Option<(i64, i64, &str)> {
    Some((
        *words.start_time.get(index)?,
        *words.end_time.get(index)?,
        words.text.get(index)?.as_str(),
    ))
}

fn rebuilt(rows: Vec<(i64, i64, String)>) -> WordColumns {
    let mut words = WordColumns::default();
    for (start, end, text) in rows {
        words.start_time.push(start);
        words.end_time.push(end);
        words.text.push(text);
    }
    words
}

fn rows(words: &WordColumns) -> Option<Vec<(i64, i64, String)>> {
    let count = words.text.len();
    if words.start_time.len() != count || words.end_time.len() != count {
        return None;
    }
    (0..count)
        .map(|index| row(words, index).map(|(start, end, text)| (start, end, text.to_owned())))
        .collect()
}

/// Word `index` reads `typed`: one word takes its place, and words
/// separated by spaces share its time by their letters. `None` when there
/// is no such word, it is a line break, or `typed` has no letters.
#[must_use]
pub fn retype_word(words: &WordColumns, index: usize, typed: &str) -> Option<WordColumns> {
    let mut all = rows(words)?;
    let (start, end, text) = all.get(index)?.clone();
    if text == LINE_BREAK {
        return None;
    }
    let parts: Vec<&str> = typed.split_whitespace().collect();
    if parts.is_empty() {
        return None;
    }
    let letters: usize = parts.iter().map(|part| part.chars().count()).sum();
    let span = (end - start).max(0);
    let mut at = start;
    let mut counted = 0;
    let mut pieces = Vec::with_capacity(parts.len());
    for (number, part) in parts.iter().enumerate() {
        counted += part.chars().count();
        let until = if number + 1 == parts.len() {
            end
        } else {
            #[allow(clippy::cast_possible_wrap)]
            let share = span * counted as i64 / letters.max(1) as i64;
            start + share
        };
        pieces.push((at, until, (*part).to_owned()));
        at = until;
    }
    all.splice(index..=index, pieces);
    Some(rebuilt(all))
}

/// Words `index` and the next as one word over both times, read with a
/// space between. `None` when either is missing or a line break.
#[must_use]
pub fn join_words(words: &WordColumns, index: usize) -> Option<WordColumns> {
    let mut all = rows(words)?;
    let (start, _, first) = all.get(index)?.clone();
    let (_, end, second) = all.get(index + 1)?.clone();
    if first == LINE_BREAK || second == LINE_BREAK {
        return None;
    }
    all.splice(
        index..=index + 1,
        [(start, end, format!("{first} {second}"))],
    );
    Some(rebuilt(all))
}

/// A line break before word `index`, or none there if one was. `None` when
/// the word is missing, is the first, or is itself a break.
#[must_use]
pub fn toggle_line_break(words: &WordColumns, index: usize) -> Option<WordColumns> {
    let mut all = rows(words)?;
    let (start, _, text) = all.get(index)?.clone();
    if index == 0 || text == LINE_BREAK {
        return None;
    }
    if all
        .get(index - 1)
        .is_some_and(|(_, _, before)| before == LINE_BREAK)
    {
        all.remove(index - 1);
    } else {
        all.insert(index, (start, start, LINE_BREAK.to_owned()));
    }
    Some(rebuilt(all))
}

/// A line break comes before word `index`.
#[must_use]
pub fn breaks_before(words: &WordColumns, index: usize) -> bool {
    index > 0
        && words
            .text
            .get(index - 1)
            .is_some_and(|before| before == LINE_BREAK)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn columns(rows: &[(&str, i64, i64)]) -> WordColumns {
        rebuilt(
            rows.iter()
                .map(|(text, start, end)| (*start, *end, (*text).to_owned()))
                .collect(),
        )
    }

    #[test]
    fn a_word_typed_as_two_shares_its_time_by_letters() {
        let words = columns(&[("one", 0, 400), ("tofar", 500, 1_000)]);
        let split = retype_word(&words, 1, " to  far ").unwrap();
        assert_eq!(split.text, ["one", "to", "far"]);
        assert_eq!(split.start_time, [0, 500, 700]);
        assert_eq!(split.end_time, [400, 700, 1_000]);
        let same = retype_word(&words, 0, "won").unwrap();
        assert_eq!((same.text[0].as_str(), same.end_time[0]), ("won", 400));
        assert!(retype_word(&words, 0, "  ").is_none());
        assert!(retype_word(&words, 2, "x").is_none());
    }

    #[test]
    fn two_words_join_over_both_times() {
        let words = columns(&[("one", 0, 400), ("two", 500, 900), ("three", 900, 1_300)]);
        let joined = join_words(&words, 0).unwrap();
        assert_eq!(joined.text, ["one two", "three"]);
        assert_eq!((joined.start_time[0], joined.end_time[0]), (0, 900));
        assert!(join_words(&words, 2).is_none());
    }

    #[test]
    fn a_line_break_goes_in_before_a_word_and_comes_out() {
        let words = columns(&[("one", 0, 400), ("two", 500, 900)]);
        let broken = toggle_line_break(&words, 1).unwrap();
        assert_eq!(broken.text, ["one", "\n", "two"]);
        assert_eq!((broken.start_time[1], broken.end_time[1]), (500, 500));
        assert!(breaks_before(&broken, 2));
        let whole = toggle_line_break(&broken, 2).unwrap();
        assert_eq!(whole, words);
        assert!(toggle_line_break(&words, 0).is_none());
        assert!(join_words(&broken, 0).is_none());
    }
}

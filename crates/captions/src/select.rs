//! Unread, reading, read, and keyword ranges, plus a pure rematch.
//!
//! Time puts every word in exactly one of unread, reading, or read. A keyword
//! is a separate split of that word and can sit beside the time range. Rematch
//! aligns an edited string onto the recognized times and returns new columns.
//! It does not write a draft.

// The crate root does not export this module yet, so a non-test build has no
// caller. Tests call every public function.
#![cfg_attr(not(test), allow(dead_code))]

use cutlass_model::WordColumns;

use crate::CaptionError;
use crate::clock::{WordSpan, words};

/// Which selector produced a range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeKind {
    Unread,
    Reading,
    Read,
    Keyword,
}

/// A keyword covers `[char_start, char_end)` of one word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeywordRange {
    pub word: usize,
    pub char_start: usize,
    pub char_end: usize,
}

/// One selected slice of a word, in milliseconds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextRange {
    pub word: usize,
    pub char_start: usize,
    pub char_end: usize,
    pub start: i64,
    pub end: i64,
}

/// One piece of a keyword split.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitPiece {
    pub text: String,
    pub char_start: usize,
    pub char_end: usize,
    pub start: i64,
    pub end: i64,
    pub keyword: bool,
}

/// Time ranges plus the keyword pieces. `original` is a clone of the columns
/// passed to [`select`].
#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    pub unread: Vec<TextRange>,
    pub reading: Vec<TextRange>,
    pub read: Vec<TextRange>,
    pub keyword: Vec<TextRange>,
    /// Recognized columns. Rematch reads these. Nothing here writes a draft.
    pub original: WordColumns,
}

// `WordColumns` is `PartialEq` but does not declare `Eq`. Its fields are
// integers, strings, and JSON values, so the derived equality is an
// equivalence and `Selection` can state that.
impl Eq for Selection {}

/// Edited text longer than this, or recognized text longer than this, is
/// [`CaptionError::Limit`]. The check runs before the alignment table exists.
pub const REMATCH_LIMIT: usize = 1024;

/// Split `text` into the prefix, the keyword, and the suffix.
///
/// Times tile `[start, end]`. The boundary at character `i` is
/// `start + (end - start) * i / len`, and the last boundary is `end`.
/// `end < start` is [`CaptionError::Inverted`]. An empty string or a char
/// range that does not sit inside the word is [`CaptionError::KeywordRange`].
pub fn split_keyword(
    text: &str,
    start: i64,
    end: i64,
    char_start: usize,
    char_end: usize,
) -> Result<Vec<SplitPiece>, CaptionError> {
    if end < start {
        return Err(CaptionError::Inverted);
    }
    let len = text.chars().count();
    if text.is_empty() || char_start >= char_end || char_end > len {
        return Err(CaptionError::KeywordRange);
    }
    let mut pieces = Vec::new();
    if char_start > 0 {
        pieces.push(cut(text, 0, char_start, start, end, len, false)?);
    }
    pieces.push(cut(text, char_start, char_end, start, end, len, true)?);
    if char_end < len {
        pieces.push(cut(text, char_end, len, start, end, len, false)?);
    }
    Ok(pieces)
}

/// Mark unread, reading, read, and keyword ranges at `millis`.
///
/// Reading is `[start, end)`. A zero-duration word is never reading. Keywords
/// use [`KeywordRange::word`] as [`WordSpan::index`]. The time lists keep each
/// word's stored interval. `original` is cloned and not modified.
pub fn select(
    words: &[WordSpan<'_>],
    millis: i64,
    keywords: &[KeywordRange],
    original: &WordColumns,
) -> Result<Selection, CaptionError> {
    validate_keywords(words, keywords)?;
    let mut selection = Selection {
        unread: Vec::new(),
        reading: Vec::new(),
        read: Vec::new(),
        keyword: Vec::new(),
        original: original.clone(),
    };
    for word in words {
        let range = TextRange {
            word: word.index,
            char_start: 0,
            char_end: word.text.chars().count(),
            start: word.start,
            end: word.end,
        };
        push_range(
            &mut selection,
            classify(word.start, word.end, millis),
            range,
        );
    }
    for keyword in keywords {
        let word = find_word(words, keyword.word)?;
        let pieces = split_keyword(
            word.text,
            word.start,
            word.end,
            keyword.char_start,
            keyword.char_end,
        )?;
        for piece in pieces {
            if !piece.keyword {
                continue;
            }
            push_range(
                &mut selection,
                RangeKind::Keyword,
                TextRange {
                    word: word.index,
                    char_start: piece.char_start,
                    char_end: piece.char_end,
                    start: piece.start,
                    end: piece.end,
                },
            );
        }
    }
    Ok(selection)
}

/// Pure function. Returns new columns. It does not accept `Draft` or `&mut MaterialText`.
pub fn rematch(original: &WordColumns, edited: &str) -> Result<WordColumns, CaptionError> {
    let recognized = words(original)?;
    let edited_chars = edited.chars().count();
    if edited_chars > REMATCH_LIMIT {
        return Err(CaptionError::Limit);
    }
    let mut recognized_chars = 0usize;
    for word in &recognized {
        let count = word.text.chars().count();
        recognized_chars = recognized_chars
            .checked_add(count)
            .ok_or(CaptionError::Limit)?;
        if recognized_chars > REMATCH_LIMIT {
            return Err(CaptionError::Limit);
        }
    }
    if edited_chars == recognized_chars {
        let concatenated: String = recognized.iter().map(|word| word.text).collect();
        if edited == concatenated {
            return Ok(original.clone());
        }
    }
    if edited.is_empty() {
        return Ok(WordColumns::default());
    }
    if recognized.is_empty() {
        return Ok(WordColumns {
            start_time: vec![0],
            end_time: vec![0],
            text: vec![edited.to_owned()],
            ..WordColumns::default()
        });
    }
    let tokens: Vec<&str> = edited.split_whitespace().collect();
    align(&recognized, &tokens)
}

fn classify(start: i64, end: i64, millis: i64) -> RangeKind {
    if millis < start {
        RangeKind::Unread
    } else if millis < end {
        RangeKind::Reading
    } else {
        RangeKind::Read
    }
}

fn push_range(selection: &mut Selection, kind: RangeKind, range: TextRange) {
    match kind {
        RangeKind::Unread => selection.unread.push(range),
        RangeKind::Reading => selection.reading.push(range),
        RangeKind::Read => selection.read.push(range),
        RangeKind::Keyword => selection.keyword.push(range),
    }
}

fn find_word<'a>(words: &'a [WordSpan<'a>], word: usize) -> Result<&'a WordSpan<'a>, CaptionError> {
    words
        .iter()
        .find(|span| span.index == word)
        .ok_or(CaptionError::KeywordRange)
}

fn validate_keywords(
    words: &[WordSpan<'_>],
    keywords: &[KeywordRange],
) -> Result<(), CaptionError> {
    for keyword in keywords {
        let word = find_word(words, keyword.word)?;
        let len = word.text.chars().count();
        if keyword.char_start >= keyword.char_end || keyword.char_end > len {
            return Err(CaptionError::KeywordRange);
        }
    }
    if keywords_overlap(keywords) {
        return Err(CaptionError::KeywordRange);
    }
    Ok(())
}

fn keywords_overlap(keywords: &[KeywordRange]) -> bool {
    let mut ordered: Vec<&KeywordRange> = keywords.iter().collect();
    ordered.sort_by(|left, right| {
        left.word
            .cmp(&right.word)
            .then(left.char_start.cmp(&right.char_start))
            .then(left.char_end.cmp(&right.char_end))
    });
    ordered.windows(2).any(|pair| {
        let [left, right] = pair else {
            return false;
        };
        left.word == right.word && right.char_start < left.char_end
    })
}

fn cut(
    text: &str,
    from: usize,
    to: usize,
    start: i64,
    end: i64,
    len: usize,
    keyword: bool,
) -> Result<SplitPiece, CaptionError> {
    Ok(SplitPiece {
        text: slice_chars(text, from, to)?,
        char_start: from,
        char_end: to,
        start: boundary(start, end, from, len)?,
        end: boundary(start, end, to, len)?,
        keyword,
    })
}

fn slice_chars(text: &str, start: usize, end: usize) -> Result<String, CaptionError> {
    let mut chars = text.chars();
    for _ in 0..start {
        if chars.next().is_none() {
            return Err(CaptionError::KeywordRange);
        }
    }
    let mut sliced = String::new();
    for _ in start..end {
        let Some(character) = chars.next() else {
            return Err(CaptionError::KeywordRange);
        };
        sliced.push(character);
    }
    Ok(sliced)
}

fn boundary(start: i64, end: i64, index: usize, len: usize) -> Result<i64, CaptionError> {
    if index == len {
        return Ok(end);
    }
    if len == 0 {
        return Err(CaptionError::KeywordRange);
    }
    let start = i128::from(start);
    let end = i128::from(end);
    let index = i128::try_from(index).map_err(|_| CaptionError::Overflow)?;
    let len = i128::try_from(len).map_err(|_| CaptionError::Overflow)?;
    let span = end.checked_sub(start).ok_or(CaptionError::Overflow)?;
    let product = span.checked_mul(index).ok_or(CaptionError::Overflow)?;
    let value = start
        .checked_add(product / len)
        .ok_or(CaptionError::Overflow)?;
    i64::try_from(value).map_err(|_| CaptionError::Overflow)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    Keep,
    Delete,
    Insert,
}

fn align(words: &[WordSpan<'_>], tokens: &[&str]) -> Result<WordColumns, CaptionError> {
    let rows = words.len();
    let cols = tokens.len();
    let width = cols.checked_add(1).ok_or(CaptionError::Limit)?;
    let height = rows.checked_add(1).ok_or(CaptionError::Limit)?;
    let cells = width.checked_mul(height).ok_or(CaptionError::Limit)?;
    let mut cost = vec![0u32; cells];
    let mut step = vec![Step::Keep; cells];

    for row in 1..=rows {
        let cell = cell_at(row, 0, width)?;
        cost[cell] = u32::try_from(row).map_err(|_| CaptionError::Overflow)?;
        step[cell] = Step::Delete;
    }
    for col in 1..=cols {
        let cell = cell_at(0, col, width)?;
        cost[cell] = u32::try_from(col).map_err(|_| CaptionError::Overflow)?;
        step[cell] = Step::Insert;
    }
    for row in 1..=rows {
        for col in 1..=cols {
            let word = words.get(row - 1).ok_or(CaptionError::Overflow)?;
            let token = tokens.get(col - 1).ok_or(CaptionError::Overflow)?;
            let keep_cost = cost[cell_at(row - 1, col - 1, width)?]
                .checked_add(u32::from(word.text != *token))
                .ok_or(CaptionError::Overflow)?;
            let delete_cost = cost[cell_at(row - 1, col, width)?]
                .checked_add(1)
                .ok_or(CaptionError::Overflow)?;
            let insert_cost = cost[cell_at(row, col - 1, width)?]
                .checked_add(1)
                .ok_or(CaptionError::Overflow)?;
            let (chosen_cost, chosen) = choose(keep_cost, delete_cost, insert_cost);
            let cell = cell_at(row, col, width)?;
            cost[cell] = chosen_cost;
            step[cell] = chosen;
        }
    }

    let mut row = rows;
    let mut col = cols;
    let mut steps = Vec::new();
    while row > 0 || col > 0 {
        let chosen = step[cell_at(row, col, width)?];
        steps.push(chosen);
        match chosen {
            Step::Keep => {
                row = row.checked_sub(1).ok_or(CaptionError::Overflow)?;
                col = col.checked_sub(1).ok_or(CaptionError::Overflow)?;
            }
            Step::Delete => {
                row = row.checked_sub(1).ok_or(CaptionError::Overflow)?;
            }
            Step::Insert => {
                col = col.checked_sub(1).ok_or(CaptionError::Overflow)?;
            }
        }
    }
    replay(words, tokens, steps)
}

fn cell_at(row: usize, col: usize, width: usize) -> Result<usize, CaptionError> {
    row.checked_mul(width)
        .and_then(|offset| offset.checked_add(col))
        .ok_or(CaptionError::Overflow)
}

fn choose(keep: u32, delete: u32, insert: u32) -> (u32, Step) {
    if keep <= delete && keep <= insert {
        (keep, Step::Keep)
    } else if delete <= insert {
        (delete, Step::Delete)
    } else {
        (insert, Step::Insert)
    }
}

fn replay(
    words: &[WordSpan<'_>],
    tokens: &[&str],
    steps: Vec<Step>,
) -> Result<WordColumns, CaptionError> {
    let mut consumed_words = 0usize;
    let mut consumed_tokens = 0usize;
    let mut text = Vec::new();
    let mut start_time = Vec::new();
    let mut end_time = Vec::new();
    for step in steps.into_iter().rev() {
        match step {
            Step::Keep => {
                let word = words.get(consumed_words).ok_or(CaptionError::Overflow)?;
                let token = tokens.get(consumed_tokens).ok_or(CaptionError::Overflow)?;
                text.push((*token).to_owned());
                start_time.push(word.start);
                end_time.push(word.end);
                consumed_words += 1;
                consumed_tokens += 1;
            }
            Step::Delete => {
                if consumed_words >= words.len() {
                    return Err(CaptionError::Overflow);
                }
                consumed_words += 1;
            }
            Step::Insert => {
                let token = tokens.get(consumed_tokens).ok_or(CaptionError::Overflow)?;
                let point = inserted_at(words, consumed_words)?;
                text.push((*token).to_owned());
                start_time.push(point);
                end_time.push(point);
                consumed_tokens += 1;
            }
        }
    }
    Ok(WordColumns {
        start_time,
        end_time,
        text,
        ..WordColumns::default()
    })
}

fn inserted_at(words: &[WordSpan<'_>], consumed: usize) -> Result<i64, CaptionError> {
    if consumed > 0 {
        return words
            .get(consumed - 1)
            .map(|word| word.end)
            .ok_or(CaptionError::Overflow);
    }
    if consumed < words.len() {
        return words
            .get(consumed)
            .map(|word| word.start)
            .ok_or(CaptionError::Overflow);
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, MaterialText, WordColumns};

    use super::{KeywordRange, REMATCH_LIMIT, Selection, SplitPiece, select, split_keyword};
    use crate::CaptionError;
    use crate::clock::WordSpan;
    use crate::select::rematch;

    fn span(index: usize, text: &str, start: i64, end: i64) -> WordSpan<'_> {
        WordSpan {
            index,
            text,
            start,
            end,
        }
    }

    fn columns(rows: &[(&str, i64, i64)]) -> WordColumns {
        let mut text = Vec::with_capacity(rows.len());
        let mut start_time = Vec::with_capacity(rows.len());
        let mut end_time = Vec::with_capacity(rows.len());
        for (word, start, end) in rows {
            text.push((*word).to_owned());
            start_time.push(*start);
            end_time.push(*end);
        }
        WordColumns {
            start_time,
            end_time,
            text,
            ..WordColumns::default()
        }
    }

    fn piece(
        text: &str,
        char_start: usize,
        char_end: usize,
        start: i64,
        end: i64,
        keyword: bool,
    ) -> SplitPiece {
        SplitPiece {
            text: text.to_owned(),
            char_start,
            char_end,
            start,
            end,
            keyword,
        }
    }

    #[test]
    fn empty_words_have_no_time_ranges() -> Result<(), CaptionError> {
        let original = columns(&[("kept", 0, 1)]);
        let selection = select(&[], 0, &[], &original)?;
        assert!(selection.unread.is_empty());
        assert!(selection.reading.is_empty());
        assert!(selection.read.is_empty());
        assert!(selection.keyword.is_empty());
        assert_eq!(selection.original, original);
        Ok(())
    }

    #[test]
    fn the_start_is_reading_and_the_end_is_read() -> Result<(), CaptionError> {
        let words = [span(0, "Hi", 100, 200)];
        let at_start = select(&words, 100, &[], &WordColumns::default())?;
        assert!(at_start.unread.is_empty());
        assert_eq!(at_start.reading.len(), 1);
        assert!(at_start.read.is_empty());
        assert_eq!(at_start.reading[0].start, 100);
        assert_eq!(at_start.reading[0].end, 200);
        assert_eq!(at_start.reading[0].char_start, 0);
        assert_eq!(at_start.reading[0].char_end, 2);

        let at_end = select(&words, 200, &[], &WordColumns::default())?;
        assert!(at_end.unread.is_empty());
        assert!(at_end.reading.is_empty());
        assert_eq!(at_end.read.len(), 1);
        assert_eq!(at_end.read[0].start, 100);
        assert_eq!(at_end.read[0].end, 200);
        Ok(())
    }

    #[test]
    fn a_gap_is_read_then_unread() -> Result<(), CaptionError> {
        let words = [span(0, "A", 0, 100), span(1, "B", 200, 300)];
        let selection = select(&words, 150, &[], &WordColumns::default())?;
        assert!(selection.reading.is_empty());
        assert_eq!(selection.read.len(), 1);
        assert_eq!(selection.read[0].word, 0);
        assert_eq!(selection.unread.len(), 1);
        assert_eq!(selection.unread[0].word, 1);
        Ok(())
    }

    #[test]
    fn a_zero_duration_word_is_never_reading() -> Result<(), CaptionError> {
        let words = [span(0, "A", 5, 5)];
        let at_start = select(&words, 5, &[], &WordColumns::default())?;
        assert!(at_start.reading.is_empty());
        assert_eq!(at_start.read.len(), 1);
        let before = select(&words, 4, &[], &WordColumns::default())?;
        assert_eq!(before.unread.len(), 1);
        assert!(before.reading.is_empty());
        Ok(())
    }

    #[test]
    fn a_partial_keyword_keeps_the_reading_interval() -> Result<(), CaptionError> {
        let words = [span(4, "ABCD", 0, 1000)];
        let keywords = [KeywordRange {
            word: 4,
            char_start: 1,
            char_end: 3,
        }];
        let original = columns(&[("ABCD", 0, 1000)]);
        let before = original.clone();
        let selection: Selection = select(&words, 500, &keywords, &original)?;
        assert_eq!(original, before);
        assert_eq!(selection.original, before);
        assert!(selection.unread.is_empty());
        assert!(selection.read.is_empty());
        assert_eq!(selection.reading.len(), 1);
        assert_eq!(selection.reading[0].word, 4);
        assert_eq!(selection.reading[0].char_start, 0);
        assert_eq!(selection.reading[0].char_end, 4);
        assert_eq!(selection.reading[0].start, 0);
        assert_eq!(selection.reading[0].end, 1000);
        assert_eq!(selection.keyword.len(), 1);
        assert_eq!(selection.keyword[0].word, 4);
        assert_eq!(selection.keyword[0].char_start, 1);
        assert_eq!(selection.keyword[0].char_end, 3);
        assert_eq!(selection.keyword[0].start, 250);
        assert_eq!(selection.keyword[0].end, 750);
        Ok(())
    }

    #[test]
    fn split_keyword_subdivides_abc() -> Result<(), CaptionError> {
        let pieces = split_keyword("ABC", 0, 10, 1, 2)?;
        assert_eq!(
            pieces,
            vec![
                piece("A", 0, 1, 0, 3, false),
                piece("B", 1, 2, 3, 6, true),
                piece("C", 2, 3, 6, 10, false),
            ]
        );
        Ok(())
    }

    #[test]
    fn a_keyword_past_the_word_is_rejected() {
        let words = [span(0, "AB", 0, 10)];
        let past_chars = [KeywordRange {
            word: 0,
            char_start: 0,
            char_end: 3,
        }];
        assert_eq!(
            select(&words, 0, &past_chars, &WordColumns::default()),
            Err(CaptionError::KeywordRange)
        );
        let missing = [KeywordRange {
            word: 3,
            char_start: 0,
            char_end: 1,
        }];
        assert_eq!(
            select(&words, 0, &missing, &WordColumns::default()),
            Err(CaptionError::KeywordRange)
        );
        assert_eq!(
            split_keyword("AB", 0, 10, 0, 0),
            Err(CaptionError::KeywordRange)
        );
        assert_eq!(
            split_keyword("", 0, 10, 0, 1),
            Err(CaptionError::KeywordRange)
        );
    }

    #[test]
    fn overlapping_keywords_are_rejected() {
        let words = [span(0, "ABCD", 0, 100)];
        let keywords = [
            KeywordRange {
                word: 0,
                char_start: 0,
                char_end: 3,
            },
            KeywordRange {
                word: 0,
                char_start: 2,
                char_end: 4,
            },
        ];
        assert_eq!(
            select(&words, 0, &keywords, &WordColumns::default()),
            Err(CaptionError::KeywordRange)
        );
    }

    #[test]
    fn touching_keywords_do_not_overlap() -> Result<(), CaptionError> {
        let words = [span(0, "ABCD", 0, 100)];
        let keywords = [
            KeywordRange {
                word: 0,
                char_start: 0,
                char_end: 2,
            },
            KeywordRange {
                word: 0,
                char_start: 2,
                char_end: 4,
            },
        ];
        let selection = select(&words, 0, &keywords, &WordColumns::default())?;
        assert_eq!(selection.keyword.len(), 2);
        assert_eq!(selection.keyword[0].char_start, 0);
        assert_eq!(selection.keyword[1].char_start, 2);
        Ok(())
    }

    #[test]
    fn an_unchanged_concatenation_keeps_the_original_times() -> Result<(), CaptionError> {
        let original = columns(&[("Hello", 0, 400), ("world", 400, 900)]);
        let current = rematch(&original, "Helloworld")?;
        assert_eq!(current, original);
        Ok(())
    }

    #[test]
    fn a_substituted_token_keeps_the_original_times() -> Result<(), CaptionError> {
        let original = columns(&[("Hello", 0, 400), ("world", 400, 900)]);
        let current = rematch(&original, "Hello there")?;
        assert_eq!(current.text, vec!["Hello".to_owned(), "there".to_owned()]);
        assert_eq!(current.start_time, vec![0, 400]);
        assert_eq!(current.end_time, vec![400, 900]);
        Ok(())
    }

    #[test]
    fn deleting_the_second_word_keeps_the_first_times() -> Result<(), CaptionError> {
        let original = columns(&[("Hello", 0, 400), ("world", 400, 900)]);
        let current = rematch(&original, "Hello")?;
        assert_eq!(current.text, vec!["Hello".to_owned()]);
        assert_eq!(current.start_time, vec![0]);
        assert_eq!(current.end_time, vec![400]);
        Ok(())
    }

    #[test]
    fn a_leading_insert_is_a_point_at_the_next_start() -> Result<(), CaptionError> {
        let original = columns(&[("Hello", 50, 400), ("world", 400, 900)]);
        let current = rematch(&original, "Oh Hello world")?;
        assert_eq!(
            current.text,
            vec!["Oh".to_owned(), "Hello".to_owned(), "world".to_owned()]
        );
        assert_eq!(current.start_time, vec![50, 50, 400]);
        assert_eq!(current.end_time, vec![50, 400, 900]);
        Ok(())
    }

    #[test]
    fn hello_does_not_match_a_different_case() -> Result<(), CaptionError> {
        let original = columns(&[("Hello", 0, 100), ("hello", 100, 200)]);
        let current = rematch(&original, "Hello")?;
        assert_eq!(current.text, vec!["Hello".to_owned()]);
        assert_eq!(current.start_time, vec![0]);
        assert_eq!(current.end_time, vec![100]);
        Ok(())
    }

    #[test]
    fn a_tie_prefers_the_diagonal() -> Result<(), CaptionError> {
        let original = columns(&[("ab", 5, 15)]);
        let current = rematch(&original, "a b")?;
        assert_eq!(current.text, vec!["a".to_owned(), "b".to_owned()]);
        assert_eq!(current.start_time, vec![5, 5]);
        assert_eq!(current.end_time, vec![5, 15]);
        Ok(())
    }

    #[test]
    fn text_past_the_rematch_limit_is_refused() {
        let edited = "a".repeat(REMATCH_LIMIT + 1);
        assert_eq!(
            rematch(&WordColumns::default(), &edited),
            Err(CaptionError::Limit)
        );
        let original = columns(&[(&"b".repeat(REMATCH_LIMIT + 1), 0, 1)]);
        assert_eq!(rematch(&original, "b"), Err(CaptionError::Limit));
    }

    #[test]
    fn an_empty_edit_returns_empty_columns() -> Result<(), CaptionError> {
        let original = columns(&[("Hello", 0, 10)]);
        let current = rematch(&original, "")?;
        assert_eq!(current, WordColumns::default());
        Ok(())
    }

    #[test]
    fn empty_original_text_still_matches_an_empty_edit() -> Result<(), CaptionError> {
        let original = columns(&[("", 0, 5)]);
        let current = rematch(&original, "")?;
        assert_eq!(current, original);
        Ok(())
    }

    #[test]
    fn no_recognized_words_become_one_untimed_word() -> Result<(), CaptionError> {
        let current = rematch(&WordColumns::default(), "Hello there")?;
        assert_eq!(current.text, vec!["Hello there".to_owned()]);
        assert_eq!(current.start_time, vec![0]);
        assert_eq!(current.end_time, vec![0]);
        Ok(())
    }

    #[test]
    fn ragged_columns_are_refused() {
        let columns = WordColumns {
            text: vec!["A".to_owned()],
            start_time: vec![0],
            ..WordColumns::default()
        };
        assert_eq!(rematch(&columns, "A"), Err(CaptionError::Ragged));
    }

    #[test]
    fn an_inverted_word_is_refused() {
        let columns = columns(&[("A", 5, 4)]);
        assert_eq!(rematch(&columns, "A"), Err(CaptionError::Inverted));
        assert_eq!(
            split_keyword("ABC", 10, 0, 1, 2),
            Err(CaptionError::Inverted)
        );
    }

    #[test]
    fn rematch_does_not_write_the_draft() -> Result<(), CaptionError> {
        let draft = Draft::empty("caption");
        let mut text = MaterialText::new("caption");
        text.content = "Hello world".to_owned();
        text.words.text = vec!["Hello".to_owned(), "world".to_owned()];
        text.words.start_time = vec![0, 400];
        text.words.end_time = vec![400, 900];
        let draft_before = draft.clone();
        let text_before = text.clone();
        let current = rematch(&text.words, "Hello there")?;
        assert_eq!(draft, draft_before);
        assert_eq!(text, text_before);
        assert_ne!(current.text, text.words.text);
        let _read_only = &draft;
        Ok(())
    }
}

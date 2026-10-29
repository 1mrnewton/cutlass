//! Speaking clock for caption words.
//!
//! Word times are milliseconds. Segment placement is microseconds. Sampling
//! reads the columns and does not write them. The same timeline always
//! returns the same clock; nothing here counts repaints.

// The crate root does not export this module yet, so a non-test build has no
// caller. Tests call every public function.
#![cfg_attr(not(test), allow(dead_code))]

use std::cmp::Reverse;

use cutlass_model::{Micros, WordColumns};

use crate::CaptionError;

/// One word borrowed from the columns.
///
/// `start` and `end` are milliseconds. Speaking uses the half-open interval
/// `[start, end)`. `end == start` is a zero-duration word and is never speaking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WordSpan<'a> {
    /// Index in the word columns.
    pub index: usize,
    /// Word text, borrowed from the columns.
    pub text: &'a str,
    /// Inclusive start, in milliseconds.
    pub start: i64,
    /// Exclusive end, in milliseconds.
    pub end: i64,
}

/// Where one millisecond sits among the words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClockPlace {
    /// The columns contain no words.
    Empty,
    /// Earlier than every start.
    Before,
    /// Inside `[start, end)`. Overlaps keep the earliest index.
    Speaking { index: usize },
    /// After `before` has ended and before `after` has started.
    Between { before: usize, after: usize },
    /// At or after every end.
    After,
}

/// One sample of the speaking clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WordClock {
    /// `(timeline - placed_at) / 1000`, truncating toward zero.
    pub millis: i64,
    /// Where [`WordClock::millis`] sits.
    pub place: ClockPlace,
}

/// `(timeline - placed_at) / 1000`, truncating toward zero.
///
/// `placed_at` is the segment target start in microseconds. The subtraction
/// is `checked_sub`; a difference that does not fit in `i64` is
/// [`CaptionError::Overflow`].
pub fn word_millis(timeline: Micros, placed_at: Micros) -> Result<i64, CaptionError> {
    let delta = timeline
        .0
        .checked_sub(placed_at.0)
        .ok_or(CaptionError::Overflow)?;
    Ok(delta / 1_000)
}

/// Borrow one span per aligned word.
///
/// Different column lengths are [`CaptionError::Ragged`]. An end strictly
/// before its start is [`CaptionError::Inverted`]. A zero-duration word is
/// kept. This does not modify the columns.
pub fn words(columns: &WordColumns) -> Result<Vec<WordSpan<'_>>, CaptionError> {
    let count = columns.start_time.len();
    if columns.end_time.len() != count || columns.text.len() != count {
        return Err(CaptionError::Ragged);
    }
    let mut spans = Vec::with_capacity(count);
    for index in 0..count {
        let start = columns
            .start_time
            .get(index)
            .copied()
            .ok_or(CaptionError::Ragged)?;
        let end = columns
            .end_time
            .get(index)
            .copied()
            .ok_or(CaptionError::Ragged)?;
        let text = columns
            .text
            .get(index)
            .map(String::as_str)
            .ok_or(CaptionError::Ragged)?;
        if end < start {
            return Err(CaptionError::Inverted);
        }
        spans.push(WordSpan {
            index,
            text,
            start,
            end,
        });
    }
    Ok(spans)
}

/// Place `timeline` on the word clock.
///
/// Milliseconds come from [`word_millis`] first. Empty columns are
/// [`ClockPlace::Empty`] and still carry that millisecond. Words that do not
/// overlap in list order are placed with a start partition. Anything else is
/// scanned in index order. The columns are not modified.
pub fn sample(
    columns: &WordColumns,
    timeline: Micros,
    placed_at: Micros,
) -> Result<WordClock, CaptionError> {
    let millis = word_millis(timeline, placed_at)?;
    let spans = words(columns)?;
    Ok(WordClock {
        millis,
        place: place(&spans, millis),
    })
}

/// Display end of the span whose [`WordSpan::index`] is `index`.
///
/// Next means the following element in `spans`, not a search by time. When
/// that start is greater than or equal to this end, the display end is the
/// next start. Otherwise, and for the last span, it is the stored end. A
/// missing index is [`CaptionError::Missing`].
pub fn held_end(spans: &[WordSpan<'_>], index: usize) -> Result<i64, CaptionError> {
    let Some((position, current)) = spans
        .iter()
        .enumerate()
        .find(|(_, span)| span.index == index)
    else {
        return Err(CaptionError::Missing);
    };
    let Some(next_position) = position.checked_add(1) else {
        return Ok(current.end);
    };
    let Some(next) = spans.get(next_position) else {
        return Ok(current.end);
    };
    if next.start >= current.end {
        Ok(next.start)
    } else {
        Ok(current.end)
    }
}

fn place(spans: &[WordSpan<'_>], millis: i64) -> ClockPlace {
    if spans.is_empty() {
        return ClockPlace::Empty;
    }
    if non_overlapping_in_list_order(spans) {
        place_ordered(spans, millis)
    } else {
        place_scan(spans, millis)
    }
}

/// Each end is at or before the next start, so the starts are sorted.
fn non_overlapping_in_list_order(spans: &[WordSpan<'_>]) -> bool {
    spans.windows(2).all(|pair| {
        let [left, right] = pair else {
            return true;
        };
        left.end <= right.start
    })
}

fn place_ordered(spans: &[WordSpan<'_>], millis: i64) -> ClockPlace {
    let next = spans.partition_point(|span| span.start <= millis);
    let Some(previous_index) = next.checked_sub(1) else {
        return ClockPlace::Before;
    };
    let Some(previous) = spans.get(previous_index) else {
        return ClockPlace::Before;
    };
    if millis < previous.end {
        return ClockPlace::Speaking {
            index: previous.index,
        };
    }
    if next == spans.len() {
        return ClockPlace::After;
    }
    let Some(following) = spans.get(next) else {
        return ClockPlace::After;
    };
    ClockPlace::Between {
        before: previous.index,
        after: following.index,
    }
}

fn place_scan(spans: &[WordSpan<'_>], millis: i64) -> ClockPlace {
    if let Some(index) = speaking_index(spans, millis) {
        return ClockPlace::Speaking { index };
    }
    let Some((min_start, max_end)) = bounds(spans) else {
        return ClockPlace::Empty;
    };
    if millis < min_start {
        return ClockPlace::Before;
    }
    if millis >= max_end {
        return ClockPlace::After;
    }
    match (
        earliest_ended(spans, millis),
        earliest_waiting(spans, millis),
    ) {
        (Some(before), Some(after)) => ClockPlace::Between { before, after },
        // `millis >= max_end` already returned After. A missing neighbor is Before.
        _ => ClockPlace::Before,
    }
}

fn speaking_index(spans: &[WordSpan<'_>], millis: i64) -> Option<usize> {
    spans
        .iter()
        .find(|span| span.start <= millis && millis < span.end)
        .map(|span| span.index)
}

fn bounds(spans: &[WordSpan<'_>]) -> Option<(i64, i64)> {
    let (first, rest) = spans.split_first()?;
    let mut min_start = first.start;
    let mut max_end = first.end;
    for span in rest {
        min_start = min_start.min(span.start);
        max_end = max_end.max(span.end);
    }
    Some((min_start, max_end))
}

/// Earliest index among the greatest end that is still at or before `millis`.
fn earliest_ended(spans: &[WordSpan<'_>], millis: i64) -> Option<usize> {
    spans
        .iter()
        .filter(|span| span.end <= millis)
        .max_by_key(|span| (span.end, Reverse(span.index)))
        .map(|span| span.index)
}

/// Earliest index among the smallest start that is still after `millis`.
fn earliest_waiting(spans: &[WordSpan<'_>], millis: i64) -> Option<usize> {
    spans
        .iter()
        .filter(|span| span.start > millis)
        .min_by_key(|span| (span.start, span.index))
        .map(|span| span.index)
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Micros, WordColumns};

    use super::{ClockPlace, WordClock, WordSpan, held_end, sample, word_millis, words};
    use crate::CaptionError;

    fn from_rows(rows: &[(i64, i64, &str)]) -> WordColumns {
        let mut start_time = Vec::with_capacity(rows.len());
        let mut end_time = Vec::with_capacity(rows.len());
        let mut text = Vec::with_capacity(rows.len());
        for (start, end, word) in rows {
            start_time.push(*start);
            end_time.push(*end);
            text.push((*word).to_owned());
        }
        WordColumns {
            start_time,
            end_time,
            text,
            ..WordColumns::default()
        }
    }

    fn clock(millis: i64, place: ClockPlace) -> WordClock {
        WordClock { millis, place }
    }

    #[test]
    fn an_empty_column_is_an_empty_clock() {
        let columns = WordColumns {
            start_time: Vec::new(),
            end_time: Vec::new(),
            text: Vec::new(),
            ..WordColumns::default()
        };
        let original = columns.clone();
        assert_eq!(words(&columns).map(|spans| spans.len()), Ok(0));
        assert_eq!(
            sample(&columns, Micros(3_500_000), Micros(1_500_000)),
            Ok(clock(2_000, ClockPlace::Empty))
        );
        assert_eq!(columns, original);
    }

    #[test]
    fn ragged_columns_are_refused_and_left_unchanged() {
        let shorter_end = WordColumns {
            start_time: vec![100, 200],
            end_time: vec![50],
            text: vec!["a".to_owned(), "b".to_owned()],
            ..WordColumns::default()
        };
        let original = shorter_end.clone();
        assert_eq!(words(&shorter_end), Err(CaptionError::Ragged));
        assert_eq!(
            sample(&shorter_end, Micros(0), Micros(0)),
            Err(CaptionError::Ragged)
        );
        assert_eq!(shorter_end, original);

        let longer_text = WordColumns {
            start_time: vec![0],
            end_time: vec![10],
            text: vec!["a".to_owned(), "b".to_owned()],
            ..WordColumns::default()
        };
        let original = longer_text.clone();
        assert_eq!(words(&longer_text), Err(CaptionError::Ragged));
        assert_eq!(longer_text, original);

        let longer_end = WordColumns {
            start_time: vec![0],
            end_time: vec![10, 20],
            text: vec!["a".to_owned()],
            ..WordColumns::default()
        };
        assert_eq!(words(&longer_end), Err(CaptionError::Ragged));
    }

    #[test]
    fn an_inverted_word_is_refused() {
        let columns = WordColumns {
            start_time: vec![0, 100],
            end_time: vec![40, 90],
            text: vec!["ok".to_owned(), "no".to_owned()],
            ..WordColumns::default()
        };
        let original = columns.clone();
        assert_eq!(words(&columns), Err(CaptionError::Inverted));
        assert_eq!(
            sample(&columns, Micros(0), Micros(0)),
            Err(CaptionError::Inverted)
        );
        assert_eq!(columns, original);

        let zero = from_rows(&[(40, 40, "dot")]);
        assert_eq!(words(&zero).map(|spans| spans.len()), Ok(1));
    }

    #[test]
    fn word_millis_truncates_toward_zero() {
        assert_eq!(word_millis(Micros(1_999), Micros(0)), Ok(1));
        assert_eq!(word_millis(Micros(999), Micros(0)), Ok(0));
        assert_eq!(word_millis(Micros(1_000), Micros(0)), Ok(1));
        assert_eq!(word_millis(Micros(-1_500), Micros(0)), Ok(-1));
        assert_eq!(word_millis(Micros(-999), Micros(0)), Ok(0));
        assert_eq!(word_millis(Micros(-1_000), Micros(0)), Ok(-1));
        assert_eq!(word_millis(Micros(4_000), Micros(4_000)), Ok(0));
        assert_eq!(word_millis(Micros(12_500), Micros(10_000)), Ok(2));
        assert_eq!(word_millis(Micros(10_000), Micros(12_500)), Ok(-2));
        assert_eq!(
            word_millis(Micros(i64::MIN), Micros(1)),
            Err(CaptionError::Overflow)
        );
        assert_eq!(
            word_millis(Micros(i64::MAX), Micros(-1)),
            Err(CaptionError::Overflow)
        );
        let ragged = WordColumns {
            start_time: vec![0],
            end_time: Vec::new(),
            text: Vec::new(),
            ..WordColumns::default()
        };
        let original = ragged.clone();
        assert_eq!(
            sample(&ragged, Micros(i64::MIN), Micros(1)),
            Err(CaptionError::Overflow)
        );
        assert_eq!(ragged, original);
    }

    #[test]
    fn a_time_before_the_first_word_is_before() {
        let columns = from_rows(&[(25, 80, "hello")]);
        let original = columns.clone();
        assert_eq!(
            sample(&columns, Micros(5_000_000), Micros(5_000_000)),
            Ok(clock(0, ClockPlace::Before))
        );
        assert_eq!(
            sample(&columns, Micros(-1_500), Micros(0)),
            Ok(clock(-1, ClockPlace::Before))
        );
        assert_eq!(columns, original);
    }

    #[test]
    fn the_start_instant_is_speaking() -> Result<(), CaptionError> {
        let columns = from_rows(&[(25, 80, "hello")]);
        let spans = words(&columns)?;
        let Some(span) = spans.first() else {
            return Err(CaptionError::Missing);
        };
        assert_eq!(span.index, 0);
        assert_eq!(span.text, "hello");
        assert_eq!(span.start, 25);
        assert_eq!(span.end, 80);
        assert_eq!(
            sample(&columns, Micros(5_025_000), Micros(5_000_000))?,
            clock(25, ClockPlace::Speaking { index: 0 })
        );
        Ok(())
    }

    #[test]
    fn the_end_of_the_only_word_is_after() {
        let columns = from_rows(&[(0, 100, "only")]);
        assert_eq!(
            sample(&columns, Micros(99_000), Micros(0)),
            Ok(clock(99, ClockPlace::Speaking { index: 0 }))
        );
        assert_eq!(
            sample(&columns, Micros(100_000), Micros(0)),
            Ok(clock(100, ClockPlace::After))
        );
    }

    #[test]
    fn the_shared_boundary_belongs_to_the_next_word() {
        let columns = from_rows(&[(0, 100, "one"), (100, 200, "two")]);
        assert_eq!(
            sample(&columns, Micros(99_000), Micros(0)),
            Ok(clock(99, ClockPlace::Speaking { index: 0 }))
        );
        assert_eq!(
            sample(&columns, Micros(100_000), Micros(0)),
            Ok(clock(100, ClockPlace::Speaking { index: 1 }))
        );
        assert_eq!(
            sample(&columns, Micros(200_000), Micros(0)),
            Ok(clock(200, ClockPlace::After))
        );
    }

    #[test]
    fn a_gap_is_between_the_two_words() {
        let columns = from_rows(&[(0, 100, "one"), (200, 300, "two")]);
        assert_eq!(
            sample(&columns, Micros(100_000), Micros(0)),
            Ok(clock(
                100,
                ClockPlace::Between {
                    before: 0,
                    after: 1,
                },
            ))
        );
        assert_eq!(
            sample(&columns, Micros(150_000), Micros(0)),
            Ok(clock(
                150,
                ClockPlace::Between {
                    before: 0,
                    after: 1,
                },
            ))
        );
        assert_eq!(
            sample(&columns, Micros(200_000), Micros(0)),
            Ok(clock(200, ClockPlace::Speaking { index: 1 }))
        );
        assert_eq!(
            sample(&columns, Micros(300_000), Micros(0)),
            Ok(clock(300, ClockPlace::After))
        );
    }

    #[test]
    fn a_zero_duration_word_is_never_speaking() {
        let only = from_rows(&[(40, 40, "dot")]);
        assert_eq!(
            sample(&only, Micros(39_000), Micros(0)),
            Ok(clock(39, ClockPlace::Before))
        );
        assert_eq!(
            sample(&only, Micros(40_000), Micros(0)),
            Ok(clock(40, ClockPlace::After))
        );
        assert_eq!(
            sample(&only, Micros(41_000), Micros(0)),
            Ok(clock(41, ClockPlace::After))
        );

        let between = from_rows(&[(0, 10, "a"), (20, 20, "dot"), (30, 40, "b")]);
        assert_eq!(
            sample(&between, Micros(20_000), Micros(0)),
            Ok(clock(
                20,
                ClockPlace::Between {
                    before: 1,
                    after: 2,
                },
            ))
        );
    }

    #[test]
    fn overlapping_words_keep_the_earliest_index() {
        // Both contain 60. A start partition would stop on index 1.
        let columns = from_rows(&[(0, 200, "wide"), (50, 150, "inner")]);
        assert_eq!(
            sample(&columns, Micros(60_000), Micros(0)),
            Ok(clock(60, ClockPlace::Speaking { index: 0 }))
        );
        assert_eq!(
            sample(&columns, Micros(160_000), Micros(0)),
            Ok(clock(160, ClockPlace::Speaking { index: 0 }))
        );

        // List order is out of time order, so the scan has to walk the indexes.
        let crossed = from_rows(&[(100, 200, "late"), (0, 40, "early")]);
        assert_eq!(
            sample(&crossed, Micros(-1_000), Micros(0)),
            Ok(clock(-1, ClockPlace::Before))
        );
        assert_eq!(
            sample(&crossed, Micros(10_000), Micros(0)),
            Ok(clock(10, ClockPlace::Speaking { index: 1 }))
        );
        assert_eq!(
            sample(&crossed, Micros(50_000), Micros(0)),
            Ok(clock(
                50,
                ClockPlace::Between {
                    before: 1,
                    after: 0,
                },
            ))
        );
        assert_eq!(
            sample(&crossed, Micros(200_000), Micros(0)),
            Ok(clock(200, ClockPlace::After))
        );
    }

    #[test]
    fn the_scan_breaks_neighbor_ties_toward_the_earlier_index() {
        let columns = from_rows(&[(0, 50, "a"), (10, 50, "b"), (80, 100, "c"), (80, 120, "d")]);
        assert_eq!(
            sample(&columns, Micros(60_000), Micros(0)),
            Ok(clock(
                60,
                ClockPlace::Between {
                    before: 0,
                    after: 2,
                },
            ))
        );
    }

    #[test]
    fn ordered_words_match_a_hand_computed_place() -> Result<(), CaptionError> {
        let columns = from_rows(&[(0, 100, "one"), (200, 300, "two"), (400, 500, "three")]);
        let cases = [
            (-1, ClockPlace::Before),
            (0, ClockPlace::Speaking { index: 0 }),
            (99, ClockPlace::Speaking { index: 0 }),
            (
                100,
                ClockPlace::Between {
                    before: 0,
                    after: 1,
                },
            ),
            (
                199,
                ClockPlace::Between {
                    before: 0,
                    after: 1,
                },
            ),
            (200, ClockPlace::Speaking { index: 1 }),
            (299, ClockPlace::Speaking { index: 1 }),
            (
                300,
                ClockPlace::Between {
                    before: 1,
                    after: 2,
                },
            ),
            (400, ClockPlace::Speaking { index: 2 }),
            (499, ClockPlace::Speaking { index: 2 }),
            (500, ClockPlace::After),
        ];
        for (millis, place) in cases {
            assert_eq!(
                sample(&columns, Micros(millis * 1_000), Micros(0)),
                Ok(clock(millis, place))
            );
        }

        // At 100 the start partition is 1: word 0 has started and ended, word 1 has not.
        let spans = words(&columns)?;
        assert_eq!(spans.partition_point(|span| span.start <= 100), 1);

        // Two zero-duration words at the same instant. The predecessor is the
        // later index, which a scan tie-break on the earliest index would not choose.
        let tied = from_rows(&[(5, 5, "a"), (5, 5, "b"), (9, 12, "c")]);
        assert_eq!(
            sample(&tied, Micros(5_000), Micros(0)),
            Ok(clock(
                5,
                ClockPlace::Between {
                    before: 1,
                    after: 2,
                },
            ))
        );
        Ok(())
    }

    #[test]
    fn held_end_extends_through_the_following_silence() -> Result<(), CaptionError> {
        let columns = WordColumns {
            start_time: vec![0, 200, 300],
            end_time: vec![100, 300, 400],
            text: vec!["alpha".to_owned(), "beta".to_owned(), "gamma".to_owned()],
            ..WordColumns::default()
        };
        let original = columns.clone();
        let spans = words(&columns)?;
        assert_eq!(held_end(&spans, 0)?, 200);
        assert_eq!(held_end(&spans, 1)?, 300);
        assert_eq!(held_end(&spans, 2)?, 400);
        assert_eq!(held_end(&spans, 3), Err(CaptionError::Missing));
        assert_eq!(columns.start_time, original.start_time);
        assert_eq!(columns.end_time, original.end_time);
        assert_eq!(columns.text, original.text);
        assert_eq!(columns, original);

        let skipped = [
            WordSpan {
                index: 4,
                text: "a",
                start: 0,
                end: 100,
            },
            WordSpan {
                index: 9,
                text: "b",
                start: 180,
                end: 240,
            },
        ];
        assert_eq!(held_end(&skipped, 4)?, 180);
        assert_eq!(held_end(&skipped, 0), Err(CaptionError::Missing));
        assert_eq!(held_end(&skipped, 9)?, 240);

        // The next element is the following list entry, not the later word at 300.
        let overlapped = [
            WordSpan {
                index: 0,
                text: "a",
                start: 0,
                end: 100,
            },
            WordSpan {
                index: 1,
                text: "b",
                start: 50,
                end: 80,
            },
            WordSpan {
                index: 2,
                text: "c",
                start: 300,
                end: 400,
            },
        ];
        assert_eq!(held_end(&overlapped, 0)?, 100);
        assert_eq!(held_end(&overlapped, 1)?, 300);
        assert_eq!(held_end(&[], 0), Err(CaptionError::Missing));
        Ok(())
    }

    #[test]
    fn the_same_time_returns_the_same_clock() {
        let columns = from_rows(&[(0, 100, "hi"), (200, 300, "there")]);
        let timeline = Micros(40_000);
        let placed_at = Micros(0);
        let first = sample(&columns, timeline, placed_at);
        let second = sample(&columns, timeline, placed_at);
        assert_eq!(first, second);
        assert_eq!(first, Ok(clock(40, ClockPlace::Speaking { index: 0 })));

        let gap = Micros(150_000);
        assert_eq!(
            sample(&columns, gap, placed_at),
            sample(&columns, gap, placed_at)
        );
    }
}

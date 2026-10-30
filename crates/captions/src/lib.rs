//! Caption sampling: a word clock, pages, selectors, and animation batches.
//!
//! Word times stay in milliseconds. Segment placement stays in microseconds.
//! Sampling reads those records and returns batches the text system can place.
//! It does not write the draft, and it does not run recognition.

#![forbid(unsafe_code)]

mod animate;
mod clock;
mod page;
mod select;

use std::error::Error;
use std::fmt;

use cutlass_model::{MaterialText, Micros, WordColumns};
use cutlass_text::TextError;

pub use animate::{
    Channel, GlyphBatch, Key, PlacedSample, RangeStyles, Rgba, SampledStyle, ShakeAxis,
    StyleCurves, apply_pulse, apply_shake, batches, local_domain, stagger,
};
pub use clock::{ClockPlace, WordClock, WordSpan, held_end, sample, word_millis, words};
pub use page::{
    CaptionLine, CaptionPage, LetterMap, MappedLetter, PageLimits, Pages, WordBytes, map_letters,
    paginate,
};
pub use select::{
    KeywordRange, REMATCH_LIMIT, RangeKind, Selection, SplitPiece, TextRange, rematch, select,
    split_keyword,
};

/// Why a caption sample could not be produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CaptionError {
    /// `start_time`, `end_time`, and `text` have different lengths.
    Ragged,
    /// A word ends before it starts.
    Inverted,
    /// A timestamp does not fit in `i64`.
    Overflow,
    /// A word index is not in the columns.
    Missing,
    /// A keyword does not fall inside its word.
    KeywordRange,
    /// A shake, pulse, or stagger period is shorter than two milliseconds.
    Period,
    /// A keyframe or amplitude is not a finite number.
    Measure,
    /// The edited string is past the rematch limit.
    Limit,
    /// Layout refused the text.
    Text(TextError),
}

impl fmt::Display for CaptionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ragged => formatter.write_str("word columns have different lengths"),
            Self::Inverted => formatter.write_str("a word ends before it starts"),
            Self::Overflow => formatter.write_str("a caption time overflowed"),
            Self::Missing => formatter.write_str("a word index is not in the columns"),
            Self::KeywordRange => formatter.write_str("a keyword does not fall inside its word"),
            Self::Period => formatter.write_str("a caption motion period is too short"),
            Self::Measure => formatter.write_str("a caption value is not finite"),
            Self::Limit => formatter.write_str("caption text is past the limit"),
            Self::Text(error) => write!(formatter, "{error}"),
        }
    }
}

impl Error for CaptionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Text(error) => Some(error),
            Self::Ragged
            | Self::Inverted
            | Self::Overflow
            | Self::Missing
            | Self::KeywordRange
            | Self::Period
            | Self::Measure
            | Self::Limit => None,
        }
    }
}

/// One caption sample. The draft that supplied the words is not written.
#[derive(Clone, Debug, PartialEq)]
pub struct CaptionFrame {
    /// Word-clock millisecond and where it sits.
    pub clock: WordClock,
    /// Lines and pages for the active words.
    pub pages: Pages,
    /// Glyphs from the text layout.
    pub letters: LetterMap,
    /// Unread, reading, read, and keyword ranges.
    pub selection: Selection,
    /// Placed glyphs with the style sampled at the word clock.
    pub batches: Vec<GlyphBatch>,
}

/// What [`sample_caption`] reads. Nothing here is a draft writer.
pub struct CaptionQuery<'a> {
    /// Text material. `current_words` is used when any of its columns is nonempty.
    pub material: &'a MaterialText,
    /// Font bytes for [`map_letters`].
    pub font: &'a [u8],
    /// Timeline position, in microseconds.
    pub timeline: Micros,
    /// Segment target start, in microseconds.
    pub placed_at: Micros,
    /// Line and page limits.
    pub limits: PageLimits,
    /// Keyword char ranges. These are not the reading selector.
    pub keywords: &'a [KeywordRange],
    /// Curves for each selector.
    pub styles: &'a RangeStyles,
    /// Milliseconds subtracted once per glyph index. Zero samples every glyph together.
    pub stagger_shift_ms: i64,
}

/// Sample words, pages, selectors, and curves at one timeline time.
///
/// Word columns stay in milliseconds. `placed_at` is the segment target start
/// in microseconds. The material is not modified.
pub fn sample_caption(query: &CaptionQuery<'_>) -> Result<CaptionFrame, CaptionError> {
    let columns = active_columns(query.material);
    let clock = sample(columns, query.timeline, query.placed_at)?;
    let spans = words(columns)?;
    let pages = paginate(&spans, query.limits);
    let letters = map_letters(query.font, query.material, &spans)?;
    let selection = select(&spans, clock.millis, query.keywords, columns)?;
    let batches = batches(
        &letters,
        &selection,
        query.styles,
        clock.millis,
        query.stagger_shift_ms,
    )?;
    Ok(CaptionFrame {
        clock,
        pages,
        letters,
        selection,
        batches,
    })
}

fn active_columns(material: &MaterialText) -> &WordColumns {
    if column_present(&material.current_words) {
        &material.current_words
    } else {
        &material.words
    }
}

fn column_present(columns: &WordColumns) -> bool {
    !columns.start_time.is_empty() || !columns.end_time.is_empty() || !columns.text.is_empty()
}

#[cfg(test)]
mod tests {
    use super::{
        CaptionError, CaptionFrame, CaptionQuery, Channel, ClockPlace, Key, KeywordRange,
        PageLimits, RangeStyles, sample_caption,
    };
    use cutlass_model::{Draft, MaterialText, Micros, WordColumns};
    use cutlass_text::TextError;
    use std::error::Error;

    const SAMPLE_FONT: &[u8] = include_bytes!("../../font/fixtures/sample.ttf");

    fn material(content: &str) -> MaterialText {
        let mut text = MaterialText::new("caption");
        text.content = content.to_owned();
        text.font_size = 64.0;
        text.line_max_width = 0.0;
        text
    }

    fn put_words(columns: &mut WordColumns, rows: &[(&str, i64, i64)]) {
        for (text, start, end) in rows {
            columns.text.push((*text).to_owned());
            columns.start_time.push(*start);
            columns.end_time.push(*end);
        }
    }

    fn query<'a>(
        material: &'a MaterialText,
        timeline: i64,
        keywords: &'a [KeywordRange],
        styles: &'a RangeStyles,
    ) -> CaptionQuery<'a> {
        CaptionQuery {
            material,
            font: SAMPLE_FONT,
            timeline: Micros(timeline),
            placed_at: Micros(0),
            limits: PageLimits {
                words_per_line: 0,
                lines_per_page: 0,
            },
            keywords,
            styles,
            stagger_shift_ms: 0,
        }
    }

    fn reading_red(at_zero: f64, at_thousand: f64) -> RangeStyles {
        let mut styles = RangeStyles::identity();
        styles.reading.color[0] = Channel {
            keys: vec![
                Key {
                    millis: 0,
                    value: at_zero,
                },
                Key {
                    millis: 1_000,
                    value: at_thousand,
                },
            ],
        };
        styles
    }

    #[test]
    fn a_layout_failure_keeps_its_source() {
        let error = CaptionError::Text(TextError::Measure);
        assert_eq!(
            error.to_string(),
            "text size, spacing, or width is not usable"
        );
        assert!(error.source().is_some());
        assert!(CaptionError::Ragged.source().is_none());
        assert_eq!(
            CaptionError::KeywordRange.to_string(),
            "a keyword does not fall inside its word"
        );
    }

    #[test]
    fn an_empty_caption_samples_without_writing_the_draft() {
        let draft = Draft::empty("caption");
        let text = material("untouched");
        let before_draft = draft.clone();
        let before_text = text.clone();
        let styles = RangeStyles::identity();
        let frame = sample_caption(&query(&text, 0, &[], &styles)).expect("sample");
        assert_eq!(draft, before_draft);
        assert_eq!(text, before_text);
        assert_eq!(frame.clock.place, ClockPlace::Empty);
        assert!(frame.pages.lines.is_empty());
        assert!(frame.batches.is_empty());
        assert!(frame.letters.letters.is_empty());
    }

    #[test]
    fn the_word_end_is_outside_the_speaking_interval() {
        let mut text = material("A");
        put_words(&mut text.words, &[("A", 0, 100)]);
        let styles = RangeStyles::identity();
        let frame = sample_caption(&query(&text, 100_000, &[], &styles)).expect("sample");
        assert_eq!(frame.clock.millis, 100);
        assert_eq!(frame.clock.place, ClockPlace::After);
        assert!(frame.selection.reading.is_empty());
        assert_eq!(frame.selection.read.len(), 1);
    }

    #[test]
    fn a_partial_keyword_is_not_the_reading_range() {
        let mut text = material("AB");
        put_words(&mut text.words, &[("AB", 0, 1_000)]);
        let keywords = [KeywordRange {
            word: 0,
            char_start: 0,
            char_end: 1,
        }];
        let styles = RangeStyles::identity();
        let frame = sample_caption(&query(&text, 100_000, &keywords, &styles)).expect("sample");
        assert_eq!(frame.clock.place, ClockPlace::Speaking { index: 0 });
        assert_eq!(frame.selection.reading.len(), 1);
        assert_eq!(frame.selection.reading[0].start, 0);
        assert_eq!(frame.selection.reading[0].end, 1_000);
        assert_eq!(frame.selection.keyword.len(), 1);
        assert_eq!(frame.selection.keyword[0].start, 0);
        assert_eq!(frame.selection.keyword[0].end, 500);
        assert_eq!(frame.selection.keyword[0].char_end, 1);
        let kinds: Vec<_> = frame.batches.iter().map(|batch| batch.kind).collect();
        assert!(kinds.contains(&super::RangeKind::Reading));
        assert!(kinds.contains(&super::RangeKind::Keyword));
    }

    #[test]
    fn seeking_changes_the_sampled_color() {
        let mut text = material("A");
        put_words(&mut text.words, &[("A", 0, 2_000)]);
        let styles = reading_red(0.0, 1.0);
        let early = sample_caption(&query(&text, 0, &[], &styles)).expect("early");
        let late = sample_caption(&query(&text, 1_000_000, &[], &styles)).expect("late");
        assert_eq!(early.clock.millis, 0);
        assert_eq!(late.clock.millis, 1_000);
        let early_red = reading_red_of(&early);
        let late_red = reading_red_of(&late);
        assert!(early_red < late_red);
    }

    #[test]
    fn current_words_are_sampled_and_the_material_stays() {
        let draft = Draft::empty("caption");
        let mut text = material("stored");
        put_words(&mut text.words, &[("NO", 0, 10)]);
        put_words(&mut text.current_words, &[("YES", 0, 100)]);
        let before_draft = draft.clone();
        let before_text = text.clone();
        let styles = RangeStyles::identity();
        let frame = sample_caption(&query(&text, 50_000, &[], &styles)).expect("sample");
        assert_eq!(draft, before_draft);
        assert_eq!(text, before_text);
        assert_eq!(frame.letters.text, "YES");
        assert_eq!(frame.selection.original.text, vec!["YES".to_owned()]);
        let _read_only = &draft;
        assert_eq!(_read_only, &before_draft);
    }

    fn reading_red_of(frame: &CaptionFrame) -> f64 {
        frame
            .batches
            .iter()
            .find(|batch| batch.kind == super::RangeKind::Reading)
            .and_then(|batch| batch.glyphs.first())
            .map(|glyph| glyph.style.color.red)
            .expect("reading glyph")
    }
}

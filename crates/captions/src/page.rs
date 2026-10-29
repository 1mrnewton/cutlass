//! Lines, pages, and letter offsets from the text layout.
//!
//! Pagination groups word spans into lines and pages. Letter mapping joins
//! the placed words, lays that string out, and records which word owns each
//! glyph. Neither step writes the caller's material.

// The crate root does not export this module yet, so a non-test build has no
// caller. Tests call every public function.
#![cfg_attr(not(test), allow(dead_code))]

use std::ops::Range;

use cutlass_model::MaterialText;
use cutlass_text::{Layout, PlacedGlyph, layout_text};

use crate::CaptionError;
use crate::clock::WordSpan;

/// How many words form a line, and how many lines form a page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageLimits {
    /// 0 does not wrap on word count.
    pub words_per_line: usize,
    /// 0 keeps every line on one page, aside from an explicit page break.
    pub lines_per_page: usize,
}

/// One line of placed word indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptionLine {
    pub words: Vec<usize>,
    pub start: i64,
    pub end: i64,
}

/// One page as a range of [`Pages::lines`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptionPage {
    pub lines: Range<usize>,
    pub start: i64,
    pub end: i64,
}

/// Lines and the pages that index them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pages {
    pub lines: Vec<CaptionLine>,
    pub pages: Vec<CaptionPage>,
}

/// Group `words` into lines and pages.
///
/// A word whose text is exactly `"\n"` ends the line. A word whose text is
/// exactly `"\n\n"` ends the line and the page. Those markers are not placed.
/// Any other word, including one that contains a newline, stays one word.
/// Empty lines and empty pages are not emitted.
pub fn paginate(words: &[WordSpan<'_>], limits: PageLimits) -> Pages {
    let mut pages = PageBuilder::default();
    for word in words {
        match word.text {
            "\n" => pages.finish_line(limits),
            "\n\n" => {
                pages.finish_line(limits);
                pages.finish_page();
            }
            _ => pages.place(word, limits),
        }
    }
    pages.finish(limits)
}

/// Byte range of one placed word inside [`LetterMap::text`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WordBytes {
    pub index: usize,
    pub byte_start: usize,
    pub byte_end: usize,
}

/// One laid-out glyph and the word that owns its source byte, when any.
#[derive(Clone, Debug, PartialEq)]
pub struct MappedLetter {
    pub source: usize,
    pub character: char,
    pub layout_line: usize,
    pub word: Option<usize>,
    pub glyph: PlacedGlyph,
}

/// Joined caption text, its layout, and a letter for every placed glyph.
#[derive(Clone, Debug, PartialEq)]
pub struct LetterMap {
    pub text: String,
    pub words: Vec<WordBytes>,
    pub layout: Layout,
    pub letters: Vec<MappedLetter>,
}

/// Join placed words, lay the clone out, and map each glyph back to a word.
///
/// `material` is cloned. Its `content` is not written. Break markers are
/// skipped. An empty join still calls layout, so a bad font is
/// [`CaptionError::Text`].
pub fn map_letters(
    font_bytes: &[u8],
    material: &MaterialText,
    words: &[WordSpan<'_>],
) -> Result<LetterMap, CaptionError> {
    let (text, placed) = join_words(words);
    let mut layout_material = material.clone();
    layout_material.content = text.clone();
    let layout = layout_text(font_bytes, &layout_material).map_err(CaptionError::Text)?;
    let mut letters = Vec::new();
    for (layout_line, line) in layout.lines.iter().enumerate() {
        for glyph in &line.glyphs {
            letters.push(MappedLetter {
                source: glyph.source,
                character: glyph.character,
                layout_line,
                word: word_at(&placed, glyph.source),
                glyph: glyph.clone(),
            });
        }
    }
    Ok(LetterMap {
        text,
        words: placed,
        layout,
        letters,
    })
}

#[derive(Default)]
struct PageBuilder {
    lines: Vec<CaptionLine>,
    pages: Vec<CaptionPage>,
    open: Option<OpenLine>,
    page_at: usize,
}

struct OpenLine {
    words: Vec<usize>,
    start: i64,
    end: i64,
}

impl PageBuilder {
    fn place(&mut self, word: &WordSpan<'_>, limits: PageLimits) {
        let full = self.open.as_ref().is_some_and(|line| {
            limits.words_per_line > 0 && line.words.len() >= limits.words_per_line
        });
        if full {
            self.finish_line(limits);
        }
        if let Some(line) = self.open.as_mut() {
            line.words.push(word.index);
            line.end = word.end;
            return;
        }
        self.open = Some(OpenLine {
            words: vec![word.index],
            start: word.start,
            end: word.end,
        });
    }

    fn finish_line(&mut self, limits: PageLimits) {
        let Some(line) = self.open.take() else {
            return;
        };
        if line.words.is_empty() {
            return;
        }
        let filled = self.lines.len().saturating_sub(self.page_at);
        if limits.lines_per_page > 0 && filled >= limits.lines_per_page {
            self.finish_page();
        }
        self.lines.push(CaptionLine {
            words: line.words,
            start: line.start,
            end: line.end,
        });
    }

    fn finish_page(&mut self) {
        let start = self.page_at;
        let end = self.lines.len();
        if start >= end {
            return;
        }
        let page_start = self.lines.get(start).map(|line| line.start);
        let page_end = end
            .checked_sub(1)
            .and_then(|index| self.lines.get(index).map(|line| line.end));
        let (Some(page_start), Some(page_end)) = (page_start, page_end) else {
            return;
        };
        self.pages.push(CaptionPage {
            lines: start..end,
            start: page_start,
            end: page_end,
        });
        self.page_at = end;
    }

    fn finish(mut self, limits: PageLimits) -> Pages {
        self.finish_line(limits);
        self.finish_page();
        Pages {
            lines: self.lines,
            pages: self.pages,
        }
    }
}

fn join_words(words: &[WordSpan<'_>]) -> (String, Vec<WordBytes>) {
    let mut text = String::new();
    let mut placed = Vec::new();
    for word in words {
        if word.text == "\n" || word.text == "\n\n" {
            continue;
        }
        if !placed.is_empty() {
            text.push(' ');
        }
        let byte_start = text.len();
        text.push_str(word.text);
        placed.push(WordBytes {
            index: word.index,
            byte_start,
            byte_end: text.len(),
        });
    }
    (text, placed)
}

fn word_at(words: &[WordBytes], source: usize) -> Option<usize> {
    let index = words.partition_point(|word| word.byte_end <= source);
    let word = words.get(index)?;
    if source >= word.byte_start && source < word.byte_end {
        Some(word.index)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::MaterialText;

    use super::{LetterMap, PageLimits, WordBytes, map_letters, paginate};
    use crate::CaptionError;
    use crate::clock::WordSpan;

    const SAMPLE_FONT: &[u8] = include_bytes!("../../font/fixtures/sample.ttf");

    fn span(index: usize, text: &str, start: i64, end: i64) -> WordSpan<'_> {
        WordSpan {
            index,
            text,
            start,
            end,
        }
    }

    fn limits(words_per_line: usize, lines_per_page: usize) -> PageLimits {
        PageLimits {
            words_per_line,
            lines_per_page,
        }
    }

    fn material(content: &str, font_size: f64, line_max_width: f64) -> MaterialText {
        let mut text = MaterialText::new("caption");
        text.content = content.to_owned();
        text.font_size = font_size;
        text.line_max_width = line_max_width;
        text.is_rich_text = false;
        text.typesetting = 0;
        text.alignment = 0;
        text
    }

    #[test]
    fn an_empty_word_list_has_no_lines_or_pages() {
        let pages = paginate(&[], limits(3, 2));
        assert!(pages.lines.is_empty());
        assert!(pages.pages.is_empty());
    }

    #[test]
    fn one_word_is_one_line_and_one_page() {
        let pages = paginate(&[span(3, "Hi", 10, 40)], limits(0, 0));
        assert_eq!(pages.lines.len(), 1);
        assert_eq!(pages.lines[0].words, vec![3]);
        assert_eq!(pages.lines[0].start, 10);
        assert_eq!(pages.lines[0].end, 40);
        assert_eq!(pages.pages.len(), 1);
        assert_eq!(pages.pages[0].lines, 0..1);
        assert_eq!(pages.pages[0].start, 10);
        assert_eq!(pages.pages[0].end, 40);
    }

    #[test]
    fn words_per_line_puts_each_word_on_its_own_line() {
        let words = [
            span(10, "A", 0, 10),
            span(11, "B", 10, 30),
            span(12, "C", 30, 40),
        ];
        let pages = paginate(&words, limits(1, 0));
        assert_eq!(pages.lines.len(), 3);
        assert_eq!(pages.pages.len(), 1);
        assert_eq!(pages.pages[0].lines, 0..3);
        assert_eq!(pages.pages[0].start, 0);
        assert_eq!(pages.pages[0].end, 40);
        assert_eq!(pages.lines[0].words, vec![10]);
        assert_eq!(pages.lines[0].start, 0);
        assert_eq!(pages.lines[0].end, 10);
        assert_eq!(pages.lines[1].words, vec![11]);
        assert_eq!(pages.lines[1].start, 10);
        assert_eq!(pages.lines[1].end, 30);
        assert_eq!(pages.lines[2].words, vec![12]);
        assert_eq!(pages.lines[2].start, 30);
        assert_eq!(pages.lines[2].end, 40);
    }

    #[test]
    fn lines_per_page_puts_each_line_on_its_own_page() {
        let words = [
            span(0, "A", 0, 10),
            span(1, "\n", 10, 10),
            span(2, "B", 10, 25),
            span(3, "\n", 25, 25),
            span(4, "C", 25, 40),
        ];
        let pages = paginate(&words, limits(0, 1));
        assert_eq!(pages.lines.len(), 3);
        assert_eq!(pages.pages.len(), 3);
        assert_eq!(pages.pages[0].lines, 0..1);
        assert_eq!(pages.pages[1].lines, 1..2);
        assert_eq!(pages.pages[2].lines, 2..3);
        assert_eq!(pages.pages[0].start, 0);
        assert_eq!(pages.pages[0].end, 10);
        assert_eq!(pages.pages[1].start, 10);
        assert_eq!(pages.pages[1].end, 25);
        assert_eq!(pages.pages[2].start, 25);
        assert_eq!(pages.pages[2].end, 40);
        assert_eq!(pages.lines[0].words, vec![0]);
        assert_eq!(pages.lines[1].words, vec![2]);
        assert_eq!(pages.lines[2].words, vec![4]);
    }

    #[test]
    fn a_newline_word_breaks_the_line_and_is_not_placed() {
        let words = [
            span(0, "A", 0, 4),
            span(1, "B", 4, 9),
            span(2, "\n", 9, 9),
            span(3, "C", 9, 15),
        ];
        let pages = paginate(&words, limits(0, 0));
        assert_eq!(pages.lines.len(), 2);
        assert_eq!(pages.pages.len(), 1);
        assert_eq!(pages.lines[0].words, vec![0, 1]);
        assert_eq!(pages.lines[0].start, 0);
        assert_eq!(pages.lines[0].end, 9);
        assert_eq!(pages.lines[1].words, vec![3]);
        assert_eq!(pages.lines[1].start, 9);
        assert_eq!(pages.lines[1].end, 15);
        assert!(pages.lines.iter().all(|line| !line.words.contains(&2)));
    }

    #[test]
    fn a_blank_line_breaks_the_page_and_is_not_placed() {
        let words = [
            span(0, "A", 0, 5),
            span(1, "B", 5, 8),
            span(2, "\n\n", 8, 8),
            span(3, "C", 8, 20),
        ];
        let pages = paginate(&words, limits(0, 0));
        assert_eq!(pages.lines.len(), 2);
        assert_eq!(pages.pages.len(), 2);
        assert_eq!(pages.lines[0].words, vec![0, 1]);
        assert_eq!(pages.lines[0].start, 0);
        assert_eq!(pages.lines[0].end, 8);
        assert_eq!(pages.lines[1].words, vec![3]);
        assert_eq!(pages.pages[0].lines, 0..1);
        assert_eq!(pages.pages[0].start, 0);
        assert_eq!(pages.pages[0].end, 8);
        assert_eq!(pages.pages[1].lines, 1..2);
        assert_eq!(pages.pages[1].start, 8);
        assert_eq!(pages.pages[1].end, 20);
        assert!(pages.lines.iter().all(|line| !line.words.contains(&2)));
    }

    #[test]
    fn a_word_containing_a_newline_stays_one_word() {
        let words = [span(7, "A\nB", 0, 10), span(8, "C", 10, 12)];
        let pages = paginate(&words, limits(0, 0));
        assert_eq!(pages.lines.len(), 1);
        assert_eq!(pages.pages.len(), 1);
        assert_eq!(pages.lines[0].words, vec![7, 8]);
        assert_eq!(pages.lines[0].start, 0);
        assert_eq!(pages.lines[0].end, 12);
    }

    #[test]
    fn break_markers_alone_do_not_emit_a_page() {
        let words = [
            span(0, "\n", 0, 1),
            span(1, "\n\n", 1, 2),
            span(2, "\n", 2, 3),
        ];
        let pages = paginate(&words, limits(1, 1));
        assert!(pages.lines.is_empty());
        assert!(pages.pages.is_empty());
    }

    #[test]
    fn map_letters_joins_words_without_writing_the_material() -> Result<(), CaptionError> {
        let source = material("do not replace", 64.0, 0.0);
        let before = source.clone();
        let words = [span(0, "A", 0, 10), span(1, "B", 10, 20)];
        let mapped = map_letters(SAMPLE_FONT, &source, &words)?;
        assert_eq!(source, before);
        assert_eq!(source.content, "do not replace");
        assert_eq!(mapped.text, "A B");
        assert_eq!(
            mapped.words,
            vec![
                WordBytes {
                    index: 0,
                    byte_start: 0,
                    byte_end: 1,
                },
                WordBytes {
                    index: 1,
                    byte_start: 2,
                    byte_end: 3,
                },
            ]
        );
        let marks: Vec<(char, Option<usize>)> = mapped
            .letters
            .iter()
            .map(|letter| (letter.character, letter.word))
            .collect();
        assert_eq!(marks, vec![('A', Some(0)), (' ', None), ('B', Some(1))]);
        assert!(
            mapped
                .letters
                .iter()
                .all(|letter| letter.source == letter.glyph.source)
        );
        assert_eq!(mapped.layout.lines.len(), 1);
        Ok(())
    }

    #[test]
    fn map_letters_skips_break_markers() -> Result<(), CaptionError> {
        let words = [
            span(0, "A", 0, 1),
            span(1, "\n", 1, 1),
            span(2, "\n\n", 1, 1),
            span(3, "B", 1, 2),
        ];
        let mapped = map_letters(SAMPLE_FONT, &material("x", 64.0, 0.0), &words)?;
        assert_eq!(mapped.text, "A B");
        assert_eq!(
            mapped
                .words
                .iter()
                .map(|word| word.index)
                .collect::<Vec<_>>(),
            vec![0, 3]
        );
        Ok(())
    }

    #[test]
    fn a_truncated_font_is_a_text_error() {
        let error = map_letters(&[], &material("A", 64.0, 0.0), &[]);
        assert!(matches!(error, Err(CaptionError::Text(_))));
    }

    #[test]
    fn a_narrow_measure_wraps_onto_more_than_one_line() -> Result<(), CaptionError> {
        let words = [span(2, "AB", 0, 10)];
        let mapped: LetterMap =
            map_letters(SAMPLE_FONT, &material("AB", 1_000.0, 1_000.0), &words)?;
        assert_eq!(mapped.text, "AB");
        assert!(mapped.layout.lines.len() > 1);
        assert!(mapped.letters.iter().any(|letter| letter.layout_line == 0));
        assert!(mapped.letters.iter().any(|letter| letter.layout_line >= 1));
        assert!(mapped.letters.iter().all(|letter| letter.word == Some(2)));
        assert!(mapped.letters.len() > 1);
        Ok(())
    }
}

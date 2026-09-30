// --- Line breaking and word retiming --------------------------------------------------

use cutlass_models::{CaptionLayout, CaptionWord};

/// Greedily wrap `text` to `max_chars_per_line`, collapsing whatever
/// whitespace it already has.
///
/// Existing line breaks are *not* respected: re-wrapping is what happens when
/// the group's layout changes, and honoring stale breaks would leave the cue
/// wrapped for the old rule. A single word longer than the limit gets its own
/// line rather than being hyphenated.
pub fn wrap(text: &str, max_chars_per_line: u16) -> String {
    let limit = usize::from(max_chars_per_line.max(1));
    let mut wrapped = String::with_capacity(text.len());
    let mut line_chars = 0usize;
    for piece in text.split_whitespace() {
        let chars = piece.chars().count();
        if line_chars == 0 {
            wrapped.push_str(piece);
            line_chars = chars;
        } else if line_chars + 1 + chars <= limit {
            wrapped.push(' ');
            wrapped.push_str(piece);
            line_chars += 1 + chars;
        } else {
            wrapped.push('\n');
            wrapped.push_str(piece);
            line_chars = chars;
        }
    }
    wrapped
}

/// Re-wrap a cue and carry its word timings across.
///
/// Wrapping only moves whitespace, so every word keeps its bytes and the ranges
/// can be remapped exactly — no timing is lost re-flowing a group. If the text
/// somehow does not survive tokenization (it always should) the timings are
/// dropped rather than left pointing at the wrong bytes.
pub fn rewrap(
    text: &str,
    words: &[CaptionWord],
    max_chars_per_line: u16,
) -> (String, Vec<CaptionWord>) {
    let wrapped = wrap(text, max_chars_per_line);
    if wrapped == text || words.is_empty() {
        return (wrapped, words.to_vec());
    }
    let Some(map) = TokenMap::new(text, &wrapped) else {
        return (wrapped, Vec::new());
    };
    let remapped = words
        .iter()
        .map(|word| CaptionWord {
            start_ms: word.start_ms,
            end_ms: word.end_ms,
            range: map.start_of(word.range.start)..map.end_of(word.range.end),
        })
        .collect();
    (wrapped, remapped)
}

/// Where a cue has to be cut to respect a layout's line budget.
///
/// `max_chars_per_line` decides where the lines fall and `max_lines` decides
/// how many of them one cue may hold, so the two rules only mean something
/// together: text that spills past the budget is not a cue that should be
/// squeezed, it is more cues. Returns the clip-relative millisecond offsets
/// where the spill starts, ascending, empty when the cue already fits.
///
/// Cuts land on word boundaries, taken from the cue's own timings. A cue
/// without any (a hand-typed line) is cut on the same length-weighted estimate
/// a subtitle file gets, spread across `duration_ms`.
pub fn overflow_cuts(
    text: &str,
    words: &[CaptionWord],
    duration_ms: u32,
    layout: &CaptionLayout,
) -> Vec<u32> {
    let max_lines = usize::from(layout.max_lines.max(1));
    let (wrapped, mut timings) = rewrap(text, words, layout.max_chars_per_line);
    if timings.is_empty() {
        timings = estimate_word_timings(&wrapped, 0, duration_ms);
    }
    if timings.is_empty() {
        return Vec::new();
    }

    let mut cuts = Vec::new();
    let mut latest = 0u32;
    let mut next_word = 0usize;
    for (index, line_start) in line_starts(&wrapped).enumerate() {
        // Line starts ascend, so the word cursor only ever moves forward.
        while timings
            .get(next_word)
            .is_some_and(|word| (word.range.start as usize) < line_start)
        {
            next_word += 1;
        }
        if index == 0 || index % max_lines != 0 {
            continue;
        }
        let Some(word) = timings.get(next_word) else {
            break;
        };
        if word.start_ms > latest {
            cuts.push(word.start_ms);
            latest = word.start_ms;
        }
    }
    cuts
}

/// Byte offset of every line's first character, the empty string included.
fn line_starts(text: &str) -> impl Iterator<Item = usize> + '_ {
    std::iter::once(0).chain(text.match_indices('\n').map(|(index, _)| index + 1))
}

/// Spread `start_ms..end_ms` across the words of `text`, weighted by how long
/// each word is.
///
/// This is the estimate for subtitle files, which carry per-cue times but no
/// per-word ones: length weighting reads far better than an even split because
/// "extraordinarily" does take longer to say than "a". Returns nothing for an
/// empty span, since a highlight that advances instantly is worse than none.
pub fn estimate_word_timings(text: &str, start_ms: u32, end_ms: u32) -> Vec<CaptionWord> {
    let span = end_ms.saturating_sub(start_ms);
    let tokens = tokenize(text);
    if span == 0 || tokens.is_empty() {
        return Vec::new();
    }
    let total: u64 = tokens.iter().map(|token| token.weight()).sum();
    if total == 0 {
        return Vec::new();
    }

    let mut words = Vec::with_capacity(tokens.len());
    let mut consumed = 0u64;
    let mut previous_end = start_ms;
    for token in &tokens {
        consumed += token.weight();
        let end = start_ms + (u64::from(span) * consumed / total) as u32;
        words.push(CaptionWord {
            start_ms: previous_end,
            end_ms: end.max(previous_end),
            range: clamp_u32(token.start)..clamp_u32(token.start + token.len),
        });
        previous_end = end;
    }
    words
}

/// One whitespace-delimited piece of cue text.
struct Token {
    start: usize,
    len: usize,
    chars: usize,
}

impl Token {
    /// Relative time weight: characters, floored at one so a lone "a" still
    /// gets a slice of the cue.
    fn weight(&self) -> u64 {
        self.chars.max(1) as u64
    }

    fn end(&self) -> usize {
        self.start + self.len
    }
}

fn tokenize(text: &str) -> Vec<Token> {
    let base = text.as_ptr() as usize;
    text.split_whitespace()
        .map(|piece| Token {
            start: piece.as_ptr() as usize - base,
            len: piece.len(),
            chars: piece.chars().count(),
        })
        .collect()
}

/// Byte-index translation between two whitespace-different versions of the
/// same text.
struct TokenMap {
    pairs: Vec<(Token, Token)>,
}

impl TokenMap {
    /// `None` when the two strings are not the same words in the same order,
    /// in which case no honest mapping exists.
    fn new(from: &str, to: &str) -> Option<Self> {
        let (old, new) = (tokenize(from), tokenize(to));
        if old.len() != new.len() {
            return None;
        }
        for (a, b) in old.iter().zip(&new) {
            if from[a.start..a.end()] != to[b.start..b.end()] {
                return None;
            }
        }
        Some(Self {
            pairs: old.into_iter().zip(new).collect(),
        })
    }

    /// Map a range start. An index in whitespace moves to the next word's
    /// start, so a range never begins on a space.
    fn start_of(&self, index: u32) -> u32 {
        let index = index as usize;
        for (old, new) in &self.pairs {
            if index <= old.start {
                return clamp_u32(new.start);
            }
            if index < old.end() {
                return clamp_u32(new.start + (index - old.start));
            }
        }
        clamp_u32(self.pairs.last().map_or(0, |(_, new)| new.end()))
    }

    /// Map a range end. An index in whitespace moves back to the previous
    /// word's end, so a range never trails a space.
    fn end_of(&self, index: u32) -> u32 {
        let index = index as usize;
        for (old, new) in self.pairs.iter().rev() {
            if index >= old.end() {
                return clamp_u32(new.end());
            }
            if index > old.start {
                return clamp_u32(new.start + (index - old.start));
            }
        }
        clamp_u32(self.pairs.first().map_or(0, |(_, new)| new.start))
    }
}

fn clamp_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_breaks_at_the_last_word_that_fits() {
        assert_eq!(wrap("the quick brown fox", 10), "the quick\nbrown fox");
    }

    #[test]
    fn wrap_collapses_existing_whitespace_and_breaks() {
        assert_eq!(wrap("  the\nquick   brown ", 20), "the quick brown");
    }

    #[test]
    fn wrap_gives_an_overlong_word_its_own_line() {
        assert_eq!(
            wrap("a supercalifragilistic b", 8),
            "a\nsupercalifragilistic\nb"
        );
    }

    #[test]
    fn wrap_counts_characters_not_bytes() {
        // Five characters, ten bytes: it fits a five-char line.
        assert_eq!(wrap("привет", 6), "привет");
    }

    #[test]
    fn rewrap_carries_word_timings_onto_the_new_line_breaks() {
        let text = "the quick brown fox";
        let words = vec![
            CaptionWord::new(0, 100, 0..3),
            CaptionWord::new(100, 200, 4..9),
            CaptionWord::new(200, 300, 10..15),
            CaptionWord::new(300, 400, 16..19),
        ];
        let (wrapped, remapped) = rewrap(text, &words, 10);
        assert_eq!(wrapped, "the quick\nbrown fox");
        for (word, expected) in remapped.iter().zip(["the", "quick", "brown", "fox"]) {
            assert_eq!(word.text(&wrapped), expected);
        }
        assert_eq!(remapped[3].start_ms, 300, "times ride along untouched");
    }

    #[test]
    fn rewrap_is_a_no_op_when_the_wrapping_already_matches() {
        let words = vec![CaptionWord::new(0, 100, 0..3)];
        let (wrapped, remapped) = rewrap("the", &words, 10);
        assert_eq!(wrapped, "the");
        assert_eq!(remapped, words);
    }

    #[test]
    fn rewrap_remaps_a_word_spanning_several_tokens() {
        // A word range covering "quick brown" keeps covering both after the
        // break moves between them.
        let words = vec![CaptionWord::new(0, 400, 4..15)];
        let (wrapped, remapped) = rewrap("the quick brown fox", &words, 10);
        assert_eq!(wrapped, "the quick\nbrown fox");
        assert_eq!(remapped[0].text(&wrapped), "quick\nbrown");
    }

    /// Four words, one second each, wrapping to one word per line at 6 chars.
    fn four_lines() -> (&'static str, Vec<CaptionWord>) {
        (
            "alpha bravo charlie delta",
            vec![
                CaptionWord::new(0, 1_000, 0..5),
                CaptionWord::new(1_000, 2_000, 6..11),
                CaptionWord::new(2_000, 3_000, 12..19),
                CaptionWord::new(3_000, 4_000, 20..25),
            ],
        )
    }

    fn layout(max_chars_per_line: u16, max_lines: u8) -> CaptionLayout {
        CaptionLayout {
            max_chars_per_line,
            max_lines,
            ..CaptionLayout::default()
        }
    }

    #[test]
    fn a_one_line_budget_cuts_at_every_line() {
        let (text, words) = four_lines();
        let cuts = overflow_cuts(text, &words, 4_000, &layout(7, 1));
        assert_eq!(cuts, vec![1_000, 2_000, 3_000]);
    }

    #[test]
    fn a_two_line_budget_cuts_at_every_other_line() {
        let (text, words) = four_lines();
        let cuts = overflow_cuts(text, &words, 4_000, &layout(7, 2));
        assert_eq!(cuts, vec![2_000], "one cut, into two two-line cues");
    }

    #[test]
    fn a_cue_that_already_fits_is_not_cut() {
        let (text, words) = four_lines();
        assert!(overflow_cuts(text, &words, 4_000, &layout(7, 4)).is_empty());
        assert!(
            overflow_cuts(text, &words, 4_000, &layout(64, 1)).is_empty(),
            "one line at 64 chars holds the whole cue"
        );
    }

    #[test]
    fn a_cue_without_timings_is_cut_on_the_length_estimate() {
        let (text, _) = four_lines();
        let cuts = overflow_cuts(text, &[], 4_000, &layout(7, 1));
        assert_eq!(cuts.len(), 3, "still three cuts, on estimated word times");
        assert!(cuts[0] > 0 && cuts.windows(2).all(|w| w[1] > w[0]));
    }

    #[test]
    fn cuts_are_measured_after_the_new_wrapping() {
        // Stale line breaks from the old rule must not decide the cuts: this
        // text arrives as two lines and has to come back as four.
        let (_, words) = four_lines();
        let cuts = overflow_cuts("alpha bravo\ncharlie delta", &words, 4_000, &layout(7, 1));
        assert_eq!(cuts, vec![1_000, 2_000, 3_000]);
    }

    #[test]
    fn estimate_weights_longer_words_with_more_time() {
        let words = estimate_word_timings("hi extraordinarily", 0, 1_000);
        assert_eq!(words.len(), 2);
        let short = words[0].end_ms - words[0].start_ms;
        let long = words[1].end_ms - words[1].start_ms;
        assert!(long > short * 4, "{long} ms vs {short} ms");
        assert_eq!(words[0].start_ms, 0);
        assert_eq!(words[1].end_ms, 1_000, "the estimate fills the whole span");
    }

    #[test]
    fn estimate_ranges_land_on_the_words() {
        let text = "alpha beta\ngamma";
        let words = estimate_word_timings(text, 100, 400);
        for (word, expected) in words.iter().zip(["alpha", "beta", "gamma"]) {
            assert_eq!(word.text(text), expected);
        }
        assert_eq!(words[0].start_ms, 100);
    }

    #[test]
    fn estimate_declines_an_empty_span_or_empty_text() {
        assert!(estimate_word_timings("hello", 500, 500).is_empty());
        assert!(estimate_word_timings("   ", 0, 1_000).is_empty());
    }

    #[test]
    fn estimated_timings_are_ascending_and_non_overlapping() {
        let text = "one two three four five six seven";
        let words = estimate_word_timings(text, 0, 37);
        let mut previous = 0;
        for word in &words {
            assert!(word.start_ms >= previous, "{word:?} went backwards");
            assert!(word.end_ms >= word.start_ms);
            previous = word.end_ms;
        }
    }
}

//! Where the words of a run of Thai, Lao, Khmer or Burmese end.
//!
//! These scripts write no space between words, so a line can only end
//! well where a list of the language's words says one does. This divides
//! a run as ICU's engines for the four scripts divide it, and is checked
//! against them: at each place it takes the word of the list that lets
//! the two words after it be words too; a stretch the list does not know
//! runs on to the next place a word of the list could start; a mark
//! stays with what it is on; and Thai's two signs that follow a word,
//! the one that cuts it short and the one that says it again, stay with
//! the word.

use super::words::{MOST_WORDS, Words};
use crate::unicode::{Props, gc, lb, props};

/// How many words are tried at once: the one being chosen and the two
/// after it.
const AHEAD: usize = 3;
/// A word of fewer characters than this is joined to a stretch the list
/// does not know that follows it.
const ROOT_JOINS_BELOW: usize = 3;
/// And that stretch must share fewer characters than this with a word of
/// the list.
const PREFIX_JOINS_BELOW: usize = 3;
/// A run of fewer characters than this is one word: two words of two.
const SHORTEST_RUN: usize = 4;

const PAIYANNOI: char = '\u{0E2F}';
const MAIYAMOK: char = '\u{0E46}';
const MAI_HAN_AKAT: char = '\u{0E31}';
const COENG: char = '\u{17D2}';

/// One of the four scripts whose lines are broken by a word list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Script {
    Thai,
    Lao,
    Khmer,
    Burmese,
}

impl Script {
    /// The script whose word list `character` is looked up in: a letter
    /// or a mark of it that the line breaking rules leave to a list.
    pub(super) fn of(character: char) -> Option<Self> {
        // The four blocks and the two Burmese ones beyond the first.
        if !matches!(
            u32::from(character),
            0x0E00..=0x0EFF | 0x1000..=0x109F | 0x1780..=0x17FF | 0xA9E0..=0xA9FF | 0xAA60..=0xAA7F
        ) {
            return None;
        }
        Self::by(props(character))
    }

    fn by(props: Props) -> Option<Self> {
        if props.line_break() != lb::COMPLEX_CONTEXT {
            return None;
        }
        match &props.script() {
            b"Thai" => Some(Self::Thai),
            b"Laoo" => Some(Self::Lao),
            b"Khmr" => Some(Self::Khmer),
            b"Mymr" => Some(Self::Burmese),
            _ => None,
        }
    }

    /// The code point the script's word list counts its letters from.
    fn first(self) -> u32 {
        match self {
            Self::Thai => 0x0E00,
            Self::Lao => 0x0E80,
            Self::Khmer => 0x1780,
            Self::Burmese => 0x1000,
        }
    }

    /// A letter a word may start with: a consonant, an independent
    /// vowel, or a vowel written ahead of its consonant.
    fn begins(self, character: char) -> bool {
        let code = u32::from(character);
        match self {
            Self::Thai => matches!(code, 0x0E01..=0x0E2E | 0x0E40..=0x0E44),
            Self::Lao => matches!(code, 0x0E81..=0x0EAE | 0x0EDC..=0x0EDD | 0x0EC0..=0x0EC4),
            Self::Khmer => matches!(code, 0x1780..=0x17B3),
            Self::Burmese => matches!(code, 0x1000..=0x102A),
        }
    }

    /// A character of the script a word may end with: any but a vowel
    /// written ahead, Thai's vowel that needs a consonant after it, and
    /// Khmer's sign that stacks the next consonant.
    fn ends(self, character: char) -> bool {
        let code = u32::from(character);
        match self {
            Self::Thai => character != MAI_HAN_AKAT && !matches!(code, 0x0E40..=0x0E44),
            Self::Lao => !matches!(code, 0x0EC0..=0x0EC4),
            Self::Khmer => character != COENG,
            Self::Burmese => true,
        }
    }
}

/// A mark, which no word ends before.
fn mark(character: char) -> bool {
    matches!(
        props(character).category(),
        gc::NONSPACING_MARK | gc::SPACING_MARK | gc::ENCLOSING_MARK
    )
}

/// The words of the list that start at one place in the run, and which
/// of them is being tried.
#[derive(Clone, Copy)]
struct Possible {
    lengths: [usize; MOST_WORDS],
    count: usize,
    /// How far the place's letters go into the list.
    prefix: usize,
    /// The place, once it has been looked up.
    offset: Option<usize>,
    /// The word to take, and the one being tried.
    marked: usize,
    current: usize,
}

impl Possible {
    const NONE: Self = Self {
        lengths: [0; MOST_WORDS],
        count: 0,
        prefix: 0,
        offset: None,
        marked: 0,
        current: 0,
    };

    /// Look up the words that start at `at` in `run`, unless this holds
    /// them already, and move `at` past the longest. How many there are.
    fn candidates(&mut self, at: &mut usize, run: &[char], words: &Words) -> usize {
        let start = *at;
        if self.offset != Some(start) {
            self.offset = Some(start);
            let found = words.starting(run.get(start..).unwrap_or(&[]));
            self.lengths = found.lengths;
            self.count = found.count;
            self.prefix = found.prefix;
        }
        if self.count > 0 {
            *at = start + self.lengths[self.count - 1];
        }
        self.current = self.count.saturating_sub(1);
        self.marked = self.current;
        self.count
    }

    /// Move `at` past the word to take; how long that word is.
    fn accept(&self, at: &mut usize) -> usize {
        let length = self.lengths[self.marked.min(MOST_WORDS - 1)];
        *at = self.offset.unwrap_or(0) + length;
        length
    }

    /// Try the next shorter word, moving `at` past it; `false` when
    /// there is none.
    fn back_up(&mut self, at: &mut usize) -> bool {
        if self.current == 0 {
            return false;
        }
        self.current -= 1;
        *at = self.offset.unwrap_or(0) + self.lengths[self.current];
        true
    }

    fn mark_current(&mut self) {
        self.marked = self.current;
    }
}

/// The places inside `run` where a word ends, in rising order, each a
/// count of characters from its start. Every character of `run` is of
/// `script`. A run too short for two words, or of a script with no list,
/// has none.
pub(super) fn word_ends(script: Script, run: &[char]) -> Vec<usize> {
    let mut ends = Vec::new();
    let end = run.len();
    // Thai asks for one character more than the others do.
    let shortest = if script == Script::Thai {
        SHORTEST_RUN + 1
    } else {
        SHORTEST_RUN
    };
    if end < shortest {
        return ends;
    }
    let Some(words) = Words::counted_from(script.first()) else {
        return ends;
    };
    let mut possible = [Possible::NONE; AHEAD];
    // How many words have been found: which of the three is next.
    let mut found = 0_usize;
    let mut at = 0_usize;
    while at < end {
        let current = at;
        let mut length = 0_usize;
        let first = found % AHEAD;
        let second = (found + 1) % AHEAD;
        let third = (found + 2) % AHEAD;
        let candidates = possible[first].candidates(&mut at, run, &words);
        if candidates == 1 {
            length = possible[first].accept(&mut at);
            found += 1;
        } else if candidates > 1 {
            // The one that lets the most words follow it: tried longest
            // first, and the first followed by two more is taken.
            'best: {
                if at >= end {
                    break 'best;
                }
                loop {
                    if possible[second].candidates(&mut at, run, &words) > 0 {
                        possible[first].mark_current();
                        if at >= end {
                            break 'best;
                        }
                        loop {
                            if possible[third].candidates(&mut at, run, &words) > 0 {
                                possible[first].mark_current();
                                break 'best;
                            }
                            if !possible[second].back_up(&mut at) {
                                break;
                            }
                        }
                    }
                    if !possible[first].back_up(&mut at) {
                        break;
                    }
                }
            }
            length = possible[first].accept(&mut at);
            found += 1;
        }
        // No word, or a short one, with no word after it: run on to the
        // next place a word could start.
        let next = found % AHEAD;
        if at < end && length < ROOT_JOINS_BELOW {
            if possible[next].candidates(&mut at, run, &words) == 0
                && (length == 0 || possible[next].prefix < PREFIX_JOINS_BELOW)
            {
                let mut left = end - (current + length);
                let mut passed = 0_usize;
                while let Some(before) = run.get(at).copied() {
                    at += 1;
                    passed += 1;
                    left = left.saturating_sub(1);
                    if left == 0 {
                        break;
                    }
                    let Some(after) = run.get(at).copied() else {
                        break;
                    };
                    if script.ends(before) && script.begins(after) {
                        let other = (found + 1) % AHEAD;
                        let starts = possible[other].candidates(&mut at, run, &words);
                        at = current + length + passed;
                        if starts > 0 {
                            break;
                        }
                    }
                }
                if length == 0 {
                    found += 1;
                }
                length += passed;
            } else {
                at = current + length;
            }
        }
        // No word ends before a mark.
        while at < end && run.get(at).copied().is_some_and(mark) {
            at += 1;
            length += 1;
        }
        // Thai's signs after a word stay with it, when no word follows.
        if script == Script::Thai && at < end && length > 0 {
            let next = found % AHEAD;
            let sign = run.get(at).copied();
            if possible[next].candidates(&mut at, run, &words) == 0
                && matches!(sign, Some(PAIYANNOI | MAIYAMOK))
            {
                let mut sign = sign;
                let before = |at: usize| at.checked_sub(1).and_then(|at| run.get(at)).copied();
                if sign == Some(PAIYANNOI) && !matches!(before(at), Some(PAIYANNOI | MAIYAMOK)) {
                    at += 1;
                    length += 1;
                    sign = run.get(at).copied();
                }
                if sign == Some(MAIYAMOK) && before(at) != Some(MAIYAMOK) {
                    at += 1;
                    length += 1;
                }
            } else {
                at = current + length;
            }
        }
        if length > 0 {
            ends.push(current + length);
        } else {
            // Nothing was taken: step on, so the run always ends.
            at = current + 1;
        }
    }
    // The end of the run is not a place inside it.
    if ends.last().is_some_and(|last| *last >= end) {
        ends.pop();
    }
    ends
}

#[cfg(test)]
mod tests {
    use super::{Script, word_ends};

    fn ends(script: Script, text: &str) -> Vec<usize> {
        let run: Vec<char> = text.chars().collect();
        word_ends(script, &run)
    }

    #[test]
    fn a_character_is_of_the_script_whose_list_it_is_looked_up_in() {
        assert_eq!(Script::of('\u{E01}'), Some(Script::Thai));
        assert_eq!(Script::of('\u{E31}'), Some(Script::Thai));
        assert_eq!(Script::of('\u{E81}'), Some(Script::Lao));
        assert_eq!(Script::of('\u{1780}'), Some(Script::Khmer));
        assert_eq!(Script::of('\u{1000}'), Some(Script::Burmese));
        assert_eq!(Script::of('\u{AA60}'), Some(Script::Burmese));
        // Digits and signs of the scripts are not words' letters, and
        // other scripts have no list.
        for other in [
            '\u{E50}', '\u{E3F}', '\u{1040}', '\u{17E0}', 'a', ' ', '\u{4E2D}',
        ] {
            assert_eq!(Script::of(other), None, "{other:?}");
        }
    }

    #[test]
    fn a_run_too_short_for_two_words_has_no_place() {
        assert!(ends(Script::Thai, "").is_empty());
        // Four Thai characters are one word; four Lao ones may be two.
        assert!(ends(Script::Thai, "\u{E20}\u{E32}\u{E29}\u{E32}").is_empty());
        assert!(ends(Script::Lao, "\u{E9E}\u{EB2}\u{EAA}").is_empty());
    }

    #[test]
    fn a_run_of_one_letter_or_of_marks_is_divided_in_time() {
        let started = std::time::Instant::now();
        for (script, letter, mark) in [
            (Script::Thai, '\u{E01}', '\u{E31}'),
            (Script::Lao, '\u{E81}', '\u{EB1}'),
            (Script::Khmer, '\u{1780}', '\u{17D2}'),
            (Script::Burmese, '\u{1000}', '\u{103A}'),
        ] {
            let found = word_ends(script, &vec![letter; 350_000]);
            assert!(found.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(found.iter().all(|end| *end > 0 && *end < 350_000));
            // Marks alone: no word ends before a mark, so none ends.
            assert!(word_ends(script, &vec![mark; 100_000]).is_empty());
            // A letter and a mark in turn, and a letter under many marks.
            let mut mixed = Vec::new();
            for _ in 0..50_000 {
                mixed.extend([letter, mark]);
            }
            let found = word_ends(script, &mixed);
            assert!(found.iter().all(|end| end % 2 == 0));
            let mut heavy = vec![letter];
            heavy.extend(vec![mark; 100_000]);
            heavy.extend(vec![letter; 8]);
            let _ = word_ends(script, &heavy);
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(20));
    }
}

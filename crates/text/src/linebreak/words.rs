//! The words of Thai, Lao, Khmer and Burmese, which are written with no
//! space between them: a line of one of them ends where a word does.
//!
//! The words are ICU's, read out of its dictionaries by
//! `tools/text-dictionaries/generate.py` into `words.bin`, whose layout
//! that script describes: for each language a trie of letters, each node
//! after everything below it, walked here in place. Nothing is built at
//! start.

/// The four lists: a header, then each language's trie.
static DATA: &[u8] = include_bytes!("words.bin");

/// The most words kept that start at one place, shortest first: what the
/// engine these lists were made for keeps.
pub(super) const MOST_WORDS: usize = 20;

/// Where the lists come from and the terms they are used under, for a
/// licenses screen.
#[derive(Clone, Copy, Debug)]
pub struct WordLists {
    pub name: &'static str,
    pub author: &'static str,
    /// The license's name.
    pub license: &'static str,
    /// Where the lists are from.
    pub link: &'static str,
    /// The notices and the license, word for word.
    pub text: &'static str,
}

/// The word lists lines of Thai, Lao, Khmer and Burmese are broken by.
pub const LINE_BREAK_WORDS: WordLists = WordLists {
    name: "ICU word lists for Thai, Lao, Khmer and Burmese",
    author: "Unicode, Inc. and others",
    license: "Unicode License V3, with the lists' own notices",
    link: "https://github.com/unicode-org/icu",
    text: include_str!("WORDS-LICENSE.txt"),
};

/// One language's words.
#[derive(Clone, Copy)]
pub(super) struct Words {
    trie: &'static [u8],
    root: usize,
    /// The code point its letters are counted from.
    first: u32,
}

/// The words that start a text, and how far into the list the text went.
#[derive(Clone, Copy)]
pub(super) struct Starting {
    /// How many characters each word is, shortest first.
    pub(super) lengths: [usize; MOST_WORDS],
    pub(super) count: usize,
    /// How many characters were looked up, the one that led nowhere
    /// counted with them.
    pub(super) prefix: usize,
}

fn word_at(data: &[u8], at: usize) -> Option<usize> {
    let bytes = data.get(at..at.checked_add(4)?)?;
    usize::try_from(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])).ok()
}

impl Words {
    /// The list whose letters are counted from `first`.
    pub(super) fn counted_from(first: u32) -> Option<Self> {
        Self::in_data(DATA, first)
    }

    fn in_data(data: &'static [u8], first: u32) -> Option<Self> {
        if data.get(..4)? != b"CLWD" || word_at(data, 4)? != 1 {
            return None;
        }
        let count = word_at(data, 8)?.min(64);
        (0..count).find_map(|language| {
            let entry = 12 + language * 20;
            if word_at(data, entry)? != usize::try_from(first).ok()? {
                return None;
            }
            let start = word_at(data, entry + 4)?;
            let length = word_at(data, entry + 8)?;
            let trie = data.get(start..start.checked_add(length)?)?;
            let root = word_at(data, entry + 12)?;
            (root < trie.len()).then_some(Self { trie, root, first })
        })
    }

    /// How many words the list holds, as its file says.
    #[cfg(test)]
    pub(super) fn listed(first: u32) -> Option<usize> {
        let count = word_at(DATA, 8)?.min(64);
        (0..count).find_map(|language| {
            let entry = 12 + language * 20;
            (word_at(DATA, entry)? == usize::try_from(first).ok()?)
                .then(|| word_at(DATA, entry + 16))?
        })
    }

    /// The byte a character is in the trie: its place after the first of
    /// its script, with the two joiners at the top. A character outside
    /// the block is read as the joiner's byte, as the engine these lists
    /// were made for reads it.
    fn byte(&self, character: char) -> u8 {
        match u32::from(character) {
            0x200D => 0xFF,
            0x200C => 0xFE,
            code => code
                .checked_sub(self.first)
                .and_then(|place| u8::try_from(place).ok())
                .filter(|place| *place <= 0xFD)
                .unwrap_or(0xFF),
        }
    }

    /// A number stored seven bits to a byte, low bits first, and where
    /// it ends.
    fn number(&self, mut at: usize) -> Option<(usize, usize)> {
        let mut value = 0_usize;
        for shift in (0..35).step_by(7) {
            let byte = *self.trie.get(at)?;
            at += 1;
            value |= usize::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                return Some((value, at));
            }
        }
        None
    }

    /// The node `back` bytes before the node at `at`.
    fn before(&self, at: usize, back: usize) -> Option<usize> {
        // A node is after everything below it: one that points at itself
        // or forward is no node, and a walk can only go toward the start.
        at.checked_sub(back).filter(|_| back > 0)
    }

    /// Whether a word ends at the node at `at`, and whether anything is
    /// below it.
    fn ends(&self, at: usize) -> Option<(bool, bool)> {
        let head = *self.trie.get(at)?;
        Some((head & 0x80 != 0, head & 0x3F != 0))
    }

    /// From `place`, the node at its first number and how many letters
    /// of that node's run are behind, take the letter `byte`: the place
    /// after it, or `None` when no word goes that way.
    fn step(&self, place: (usize, usize), byte: u8) -> Option<(usize, usize)> {
        let (at, along) = place;
        let head = *self.trie.get(at)?;
        let count = usize::from(head & 0x3F);
        if head & 0x40 != 0 {
            // A run: letters that only follow one another.
            if *self.trie.get(at + 1 + along)? != byte {
                return None;
            }
            if along + 1 < count {
                return Some((at, along + 1));
            }
            let (back, _) = self.number(at + 1 + count)?;
            return Some((self.before(at, back)?, 0));
        }
        let (count, letters) = if count == 63 {
            (usize::from(*self.trie.get(at + 1)?), at + 2)
        } else {
            (count, at + 1)
        };
        let listed = self.trie.get(letters..letters + count)?;
        let child = listed.binary_search(&byte).ok()?;
        let mut numbers = letters + count;
        for _ in 0..child {
            numbers = self.number(numbers)?.1;
        }
        let (back, _) = self.number(numbers)?;
        Some((self.before(at, back)?, 0))
    }

    /// The words of the list that start `text`.
    pub(super) fn starting(&self, text: &[char]) -> Starting {
        let mut found = Starting {
            lengths: [0; MOST_WORDS],
            count: 0,
            prefix: 0,
        };
        let mut place = (self.root, 0);
        for (index, character) in text.iter().enumerate() {
            found.prefix = index + 1;
            let Some(next) = self.step(place, self.byte(*character)) else {
                break;
            };
            place = next;
            if place.1 != 0 {
                continue;
            }
            let Some((ends, more)) = self.ends(place.0) else {
                break;
            };
            if ends && found.count < MOST_WORDS {
                found.lengths[found.count] = index + 1;
                found.count += 1;
            }
            if ends && !more {
                break;
            }
        }
        found
    }

    /// Whether `word` is in the list.
    #[cfg(test)]
    pub(super) fn has(&self, word: &str) -> bool {
        let text: Vec<char> = word.chars().collect();
        let found = self.starting(&text);
        found.lengths[..found.count].contains(&text.len())
    }
}

#[cfg(test)]
mod tests {
    use super::{DATA, LINE_BREAK_WORDS, MOST_WORDS, Words};

    const THAI: u32 = 0x0E00;
    const LAO: u32 = 0x0E80;
    const KHMER: u32 = 0x1780;
    const BURMESE: u32 = 0x1000;

    fn chars(text: &str) -> Vec<char> {
        text.chars().collect()
    }

    #[test]
    fn each_language_has_its_list_and_knows_its_words() {
        // The counts the lists were written with.
        for (first, words) in [
            (THAI, 26_383),
            (LAO, 30_550),
            (KHMER, 81_028),
            (BURMESE, 41_120),
        ] {
            assert!(Words::counted_from(first).is_some(), "{first:X}");
            assert_eq!(Words::listed(first), Some(words), "{first:X}");
        }
        assert!(Words::counted_from(0x0900).is_none());
        let thai = Words::counted_from(THAI).expect("Thai");
        // Language, Thai, hello; and no word of these letters.
        for word in [
            "\u{E20}\u{E32}\u{E29}\u{E32}",
            "\u{E44}\u{E17}\u{E22}",
            "\u{E2A}\u{E27}\u{E31}\u{E2A}\u{E14}\u{E35}",
        ] {
            assert!(thai.has(word), "{word}");
        }
        assert!(!thai.has("\u{E01}\u{E01}\u{E01}\u{E01}\u{E01}"));
        assert!(!thai.has(""));
        let lao = Words::counted_from(LAO).expect("Lao");
        assert!(lao.has("\u{E9E}\u{EB2}\u{EAA}\u{EB2}"));
        let khmer = Words::counted_from(KHMER).expect("Khmer");
        assert!(khmer.has("\u{1797}\u{17B6}\u{179F}\u{17B6}"));
        let burmese = Words::counted_from(BURMESE).expect("Burmese");
        assert!(burmese.has("\u{1019}\u{103C}\u{1014}\u{103A}\u{1019}\u{102C}"));
    }

    #[test]
    fn the_words_that_start_a_text_come_shortest_first_with_how_far_it_went() {
        let thai = Words::counted_from(THAI).expect("Thai");
        // "Language of Thailand": the first word, and longer ones that
        // start the same way.
        let text = chars("\u{E20}\u{E32}\u{E29}\u{E32}\u{E44}\u{E17}\u{E22}");
        let found = thai.starting(&text);
        assert!(found.count >= 1);
        let lengths = &found.lengths[..found.count];
        assert!(lengths.contains(&4), "{lengths:?}");
        assert!(lengths.windows(2).all(|pair| pair[0] < pair[1]));
        for length in lengths {
            let word: String = text[..*length].iter().collect();
            assert!(thai.has(&word));
        }
        // The letter that led nowhere is counted with the ones looked up.
        assert!(found.prefix > lengths[lengths.len() - 1] || found.prefix == text.len());
        // A letter of another script leads nowhere at once.
        let latin = thai.starting(&chars("abc"));
        assert_eq!((latin.count, latin.prefix), (0, 1));
        let nothing = thai.starting(&[]);
        assert_eq!((nothing.count, nothing.prefix), (0, 0));
    }

    #[test]
    fn a_long_or_strange_text_is_looked_up_in_time() {
        let started = std::time::Instant::now();
        for first in [THAI, LAO, KHMER, BURMESE] {
            let words = Words::counted_from(first).expect("list");
            // One letter a million times, every letter of the block in
            // turn, and letters from everywhere.
            let letter = char::from_u32(first + 1).expect("letter");
            let found = words.starting(&vec![letter; 1_000_000]);
            assert!(found.count <= MOST_WORDS && found.prefix < 100);
            for code in first..first + 0x100 {
                let Some(character) = char::from_u32(code) else {
                    continue;
                };
                let text = vec![character; 64];
                let found = words.starting(&text);
                assert!(found.prefix <= text.len());
                assert!(
                    found.lengths[..found.count]
                        .iter()
                        .all(|length| *length <= text.len())
                );
            }
            let _ = words.starting(&chars("\u{200D}\u{200C}\u{10FFFF}\u{0}a\u{E01}"));
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }

    #[test]
    fn a_torn_file_holds_no_list_and_a_torn_trie_no_word() {
        // The header cut at every length, and with every byte turned.
        let header = &DATA[..12 + 4 * 20];
        for cut in 0..header.len() {
            let torn: &'static [u8] = Box::leak(header[..cut].to_vec().into_boxed_slice());
            assert!(Words::in_data(torn, THAI).is_none());
        }
        let small: Vec<u8> = DATA[..4_096].to_vec();
        for at in 0..12 + 4 * 20 {
            for value in [0x00, 0x7F, 0xFF] {
                let mut torn = small.clone();
                torn[at] = value;
                let torn: &'static [u8] = Box::leak(torn.into_boxed_slice());
                for first in [THAI, LAO, KHMER, BURMESE] {
                    if let Some(words) = Words::in_data(torn, first) {
                        let _ = words.starting(&chars("\u{E20}\u{E32}\u{E29}\u{E32}"));
                    }
                }
            }
        }
        // A trie of bytes that mean nothing, walked from every place:
        // each step goes toward the start or stops.
        let noise: Vec<u8> = (0..2_048_u32)
            .map(|at| (at.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();
        let noise: &'static [u8] = Box::leak(noise.into_boxed_slice());
        let text: Vec<char> = (0..512_u32)
            .filter_map(|at| char::from_u32(THAI + at % 0x60))
            .collect();
        for root in 0..noise.len() {
            let words = Words {
                trie: noise,
                root,
                first: THAI,
            };
            let found = words.starting(&text);
            assert!(found.prefix <= text.len());
        }
    }

    #[test]
    fn the_lists_come_with_their_terms() {
        assert!(LINE_BREAK_WORDS.link.starts_with("https://"));
        assert!(LINE_BREAK_WORDS.text.contains("UNICODE LICENSE V3"));
        for named in ["Lao", "Burmese", "Khmer"] {
            assert!(LINE_BREAK_WORDS.text.contains(named), "{named}");
        }
        assert!(LINE_BREAK_WORDS.text.len() > 2_000);
    }
}

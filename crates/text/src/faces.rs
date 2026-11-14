//! The faces one title is set in.
//!
//! A title has its own face. A rich title's ranges may name others, and a
//! character none of them can draw is set in a fallback face the shell
//! offers for it: another script's letters, a symbol, an emoji. The text
//! does not open files; [`FaceSource`] hands it bytes.

use cutlass_font::Font;

use crate::linebreak::ends_line;
use crate::richtext::{Run, run_at};
use crate::shaper::has_space_for;
use crate::unicode::{decompose, props};

/// The most fallback faces one title takes. Past it, a character no face
/// so far can draw is drawn as missing.
pub const MAX_FALLBACK_FACES: usize = 8;

/// Where a title's other faces come from.
pub trait FaceSource {
    /// The bytes of the face a range's font path names.
    fn named(&self, path: &str) -> Option<&[u8]>;

    /// The bytes of a face that draws `character`, which the title's own
    /// face cannot. `emoji` asks for a face that draws it in color.
    fn fallback(&self, character: char, emoji: bool) -> Option<&[u8]>;
}

/// No other faces: every character is set in the title's own.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoFaces;

impl FaceSource for NoFaces {
    fn named(&self, _path: &str) -> Option<&[u8]> {
        None
    }

    fn fallback(&self, _character: char, _emoji: bool) -> Option<&[u8]> {
        None
    }
}

/// The faces gathered for one title, its own first, and which of them
/// each character is set in.
pub(crate) struct Faces<'f> {
    pub(crate) fonts: Vec<Font<'f>>,
    /// The byte index and face of each character not set in the title's
    /// own face, in order.
    others: Vec<(usize, usize)>,
}

impl Faces<'_> {
    /// The face the character at byte `source` is set in.
    pub(crate) fn pick(&self, source: usize) -> usize {
        self.others
            .binary_search_by_key(&source, |(byte, _)| *byte)
            .map_or(0, |found| self.others[found].1)
    }
}

fn has(font: &Font<'_>, character: char) -> bool {
    matches!(font.glyph_id(character), Ok(Some(_)))
}

/// Whether `font` can draw `character`: as its own glyph, as a letter and
/// its accent, or as its plain space at another width.
fn draws(font: &Font<'_>, character: char) -> bool {
    has(font, character)
        || has_space_for(font, character)
        || decompose(character).is_some_and(|(first, second)| {
            has(font, first) && second.is_none_or(|second| has(font, second))
        })
}

/// Gather the faces for `text`: `font` is the title's own, `runs` name
/// the faces of its ranges, and `source` supplies the bytes.
pub(crate) fn gather<'f>(
    font: Font<'f>,
    text: &str,
    runs: &[Run],
    source: &'f dyn FaceSource,
) -> Faces<'f> {
    let mut fonts = vec![font];
    // The faces ranges name, each read once.
    let mut named: Vec<(&str, usize)> = Vec::new();
    for path in runs.iter().filter_map(|run| run.font.as_deref()) {
        if named.iter().any(|(seen, _)| *seen == path) {
            continue;
        }
        let Some(other) = source
            .named(path)
            .and_then(|bytes| cutlass_font::read(bytes).ok())
        else {
            named.push((path, 0));
            continue;
        };
        named.push((path, fonts.len()));
        fonts.push(other);
    }
    let of_run = |at: usize| {
        run_at(runs, at)
            .and_then(|run| run.font.as_deref())
            .and_then(|path| named.iter().find(|(seen, _)| *seen == path))
            .map_or(0, |(_, index)| *index)
    };
    // Fallback faces, each read once: where its bytes are, and its place.
    let mut fallbacks: Vec<(*const u8, usize, usize)> = Vec::new();
    let mut others = Vec::new();
    let mut characters = text.char_indices().peekable();
    while let Some((at, character)) = characters.next() {
        let mut face = of_run(at);
        let props = props(character);
        let next = characters.peek().map(|(_, next)| *next);
        // A mark or a joiner goes in the face of the letter before it,
        // which shaping sees to; a line end draws nothing.
        let drawn = !(props.is_mark() || props.default_ignorable() || ends_line(character));
        if drawn {
            let emoji = (props.emoji_presentation() && next != Some('\u{FE0E}'))
                || (props.emoji() && next == Some('\u{FE0F}'));
            let chosen = fonts.get(face);
            let able = chosen.is_some_and(|font| draws(font, character));
            let in_color = chosen.is_some_and(Font::has_bitmap_strike);
            if !able || (emoji && !in_color) {
                let found = source.fallback(character, emoji).and_then(|bytes| {
                    let key = (bytes.as_ptr(), bytes.len());
                    if let Some((_, _, index)) = fallbacks
                        .iter()
                        .find(|(start, len, _)| (*start, *len) == key)
                    {
                        return Some(*index);
                    }
                    if fallbacks.len() == MAX_FALLBACK_FACES {
                        return None;
                    }
                    let other = cutlass_font::read(bytes).ok()?;
                    fallbacks.push((key.0, key.1, fonts.len()));
                    fonts.push(other);
                    Some(fonts.len() - 1)
                });
                // A fallback that cannot draw it either is not used.
                if let Some(index) = found
                    .filter(|index| fonts.get(*index).is_some_and(|font| draws(font, character)))
                {
                    face = index;
                }
            }
        }
        if face != 0 {
            others.push((at, face));
        }
    }
    Faces { fonts, others }
}

#[cfg(test)]
mod tests {
    use super::{FaceSource, MAX_FALLBACK_FACES, NoFaces, gather};
    use crate::richtext::Run;
    use std::cell::RefCell;

    const SAMPLE: &[u8] = include_bytes!("../../font/fixtures/sample.ttf");
    const SHAPING: &[u8] = include_bytes!("../../font/fixtures/shaping.ttf");
    const COLOR: &[u8] = include_bytes!("../../font/fixtures/color-strike.otf");

    /// Faces from two closures, and what was asked of the second.
    struct Table<N, F> {
        named: N,
        fallback: F,
        asked: RefCell<Vec<(char, bool)>>,
    }

    impl<N, F> Table<N, F>
    where
        N: Fn(&str) -> Option<&'static [u8]>,
        F: Fn(char) -> Option<&'static [u8]>,
    {
        fn new(named: N, fallback: F) -> Self {
            Self {
                named,
                fallback,
                asked: RefCell::new(Vec::new()),
            }
        }
    }

    impl<N, F> FaceSource for Table<N, F>
    where
        N: Fn(&str) -> Option<&'static [u8]>,
        F: Fn(char) -> Option<&'static [u8]>,
    {
        fn named(&self, path: &str) -> Option<&[u8]> {
            (self.named)(path)
        }

        fn fallback(&self, character: char, emoji: bool) -> Option<&[u8]> {
            self.asked.borrow_mut().push((character, emoji));
            (self.fallback)(character)
        }
    }

    fn own() -> cutlass_font::Font<'static> {
        cutlass_font::read(SAMPLE).expect("sample")
    }

    #[test]
    fn a_character_the_face_lacks_is_set_in_a_fallback_that_has_it() {
        // The sample face has A and B; the shaping face has f, e and more.
        let source = Table::new(|_| None, |character| (character != 'Q').then_some(SHAPING));
        let faces = gather(own(), "AfB e", &[], &source);
        assert_eq!(faces.fonts.len(), 2);
        let picked: Vec<usize> = [0, 1, 2, 3, 4].iter().map(|at| faces.pick(*at)).collect();
        assert_eq!(picked, [0, 1, 0, 0, 1]);
        // A character the fallback lacks too stays in the title's face,
        // and one nobody offers a face for does as well.
        let faces = gather(own(), "AZQ", &[], &source);
        assert_eq!((faces.pick(1), faces.pick(2)), (0, 0));
        // With no source, nothing changes.
        let alone = gather(own(), "Af", &[], &NoFaces);
        assert_eq!((alone.fonts.len(), alone.pick(1)), (1, 0));
    }

    #[test]
    fn what_the_face_can_make_does_not_ask_for_a_fallback() {
        let source = Table::new(|_| None, |_| None);
        let shaping = cutlass_font::read(SHAPING).expect("shaping");
        // An accent on its letter, a letter the face draws in two parts, a
        // space it lacks, a joiner and a line end: none is asked for.
        let _faces = gather(shaping, "e\u{301}\u{C1}\u{2003}\u{200D}\n", &[], &source);
        assert!(
            source.asked.borrow().is_empty(),
            "{:?}",
            source.asked.borrow()
        );
    }

    #[test]
    fn an_emoji_asks_for_a_face_in_color() {
        let source = Table::new(|_| None, |_| None);
        // A pictograph, a digit made a keycap by its selector, and a
        // plain digit the face lacks.
        let _faces = gather(own(), "\u{1F600}1\u{FE0F}2", &[], &source);
        assert_eq!(
            *source.asked.borrow(),
            [('\u{1F600}', true), ('1', true), ('2', false)]
        );
        // A face that already draws in color keeps its own emoji glyph.
        let source = Table::new(|_| None, |_| None);
        let color = cutlass_font::read(COLOR).expect("color");
        let has_a = color.glyph_id('A').expect("map").is_some();
        let _faces = gather(color, "A\u{FE0F}", &[], &source);
        assert_eq!(source.asked.borrow().is_empty(), has_a);
    }

    #[test]
    fn a_range_names_its_face_and_fallbacks_stop_at_the_limit() {
        let runs = [Run {
            start: 1,
            end: 2,
            fill: None,
            size: 1.0,
            outline: None,
            shadow: None,
            font: Some("other.ttf".to_owned()),
        }];
        let source = Table::new(|path| (path == "other.ttf").then_some(SHAPING), |_| None);
        let faces = gather(own(), "AfA", &runs, &source);
        assert_eq!((faces.pick(0), faces.pick(1), faces.pick(2)), (0, 1, 0));

        // A copy of the face for each letter is a face of its own: past
        // the limit no more are read.
        let copies: Vec<&'static [u8]> = (0..12)
            .map(|_| &*Box::leak(SHAPING.to_vec().into_boxed_slice()))
            .collect();
        let letters = [
            'f', 'i', 'e', 'x', 'A', 'V', '(', ')', '\u{628}', '\u{627}', '\u{644}',
        ];
        let source = Table::new(
            |_| None,
            |character| {
                let index = letters.iter().position(|letter| *letter == character)?;
                copies.get(index).copied()
            },
        );
        let text: String = letters.iter().filter(|letter| **letter != 'A').collect();
        let faces = gather(own(), &text, &[], &source);
        assert_eq!(faces.fonts.len(), 1 + MAX_FALLBACK_FACES);
        // The ninth letter found no room and stays in the title's face.
        let ninth = text.char_indices().nth(8).expect("ninth").0;
        assert_eq!(faces.pick(ninth), 0);
    }
}

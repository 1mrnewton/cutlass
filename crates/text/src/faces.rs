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

    /// The bold, italic, or bold italic face of the family of the face
    /// that font path `path` names. A source with no such face gives
    /// `None`, and the letters are thickened or leaned instead.
    fn styled(&self, path: &str, bold: bool, italic: bool) -> Option<StyledFace<'_>> {
        let _ = (path, bold, italic);
        None
    }
}

/// A face of a family, and which of bold and italic it is drawn as.
#[derive(Clone, Copy, Debug)]
pub struct StyledFace<'a> {
    pub bytes: &'a [u8],
    pub bold: bool,
    pub italic: bool,
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
    /// Each character not set plainly in the title's own face, in order.
    others: Vec<Other>,
}

/// One character set in another face, or in a bold or italic one.
#[derive(Clone, Copy)]
struct Other {
    /// Byte index of the character.
    at: usize,
    face: usize,
    /// Its face is a bold one, so it is not thickened.
    bold: bool,
    /// Its face is an italic one, so it is not leaned.
    italic: bool,
}

impl Faces<'_> {
    fn other(&self, source: usize) -> Option<&Other> {
        self.others
            .binary_search_by_key(&source, |other| other.at)
            .ok()
            .and_then(|found| self.others.get(found))
    }

    /// The face the character at byte `source` is set in.
    pub(crate) fn pick(&self, source: usize) -> usize {
        self.other(source).map_or(0, |other| other.face)
    }

    /// Whether the face of the character at byte `source` is itself
    /// bold, and whether it is itself italic.
    pub(crate) fn real(&self, source: usize) -> (bool, bool) {
        self.other(source)
            .map_or((false, false), |other| (other.bold, other.italic))
    }
}

/// A bold or italic face that was asked for, and what came back: its
/// place among the faces, and whether it is bold and italic itself.
struct Asked<'p> {
    path: &'p str,
    bold: bool,
    italic: bool,
    found: Option<(usize, bool, bool)>,
}

/// What a title asks of its faces.
#[derive(Clone, Copy)]
pub(crate) struct Wanted<'w> {
    /// The title's font path, which names its family.
    pub(crate) path: &'w str,
    /// The whole title is bold, or italic.
    pub(crate) bold: bool,
    pub(crate) italic: bool,
}

impl<'w> Wanted<'w> {
    pub(crate) fn of(material: &'w cutlass_model::MaterialText) -> Self {
        Self {
            path: material.font_path.as_str(),
            bold: material.bold_width.is_finite() && material.bold_width > 0.0,
            italic: material.italic_degree != 0,
        }
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
/// the faces of its ranges and say which are bold or italic, and `source`
/// supplies the bytes.
pub(crate) fn gather<'f>(
    font: Font<'f>,
    text: &str,
    runs: &[Run],
    wanted: Wanted<'_>,
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
    // The bold and italic faces asked for, each asked for once: the
    // family's path, the style, and the face found with what it is.
    let mut styled: Vec<Asked<'_>> = Vec::new();
    // Fallback faces, each read once: where its bytes are, and its place.
    let mut fallbacks: Vec<(*const u8, usize, usize)> = Vec::new();
    let mut others = Vec::new();
    // The first kana of the text, or else its first Hangul.
    let scripts = |wanted: &[&[u8; 4]]| {
        text.chars()
            .find(|character| wanted.contains(&&props(*character).script()))
    };
    let guide = scripts(&[b"Hira", b"Kana"]).or_else(|| scripts(&[b"Hang"]));
    let mut characters = text.char_indices().peekable();
    while let Some((at, character)) = characters.next() {
        let mut face = of_run(at);
        let run = run_at(runs, at);
        let bold = run.and_then(|run| run.bold).unwrap_or(wanted.bold);
        let italic = run.and_then(|run| run.italic).unwrap_or(wanted.italic);
        let (mut real_bold, mut real_italic) = (false, false);
        if bold || italic {
            // The family is that of the range's own face, or the title's.
            let path = run
                .and_then(|run| run.font.as_deref())
                .filter(|_| face != 0)
                .unwrap_or(wanted.path);
            let known = styled
                .iter()
                .find(|asked| (asked.path, asked.bold, asked.italic) == (path, bold, italic))
                .map(|asked| asked.found);
            let found = known.unwrap_or_else(|| {
                let found = source.styled(path, bold, italic).and_then(|styled| {
                    let other = cutlass_font::read(styled.bytes).ok()?;
                    fonts.push(other);
                    Some((fonts.len() - 1, styled.bold, styled.italic))
                });
                styled.push(Asked {
                    path,
                    bold,
                    italic,
                    found,
                });
                found
            });
            if let Some((index, is_bold, is_italic)) = found {
                face = index;
                real_bold = is_bold;
                real_italic = is_italic;
            }
        }
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
            let in_color =
                chosen.is_some_and(|font| font.has_color_strike() || font.has_color_layers());
            if !able || (emoji && !in_color) {
                let mut held = |bytes: &'f [u8]| {
                    let key = (bytes.as_ptr(), bytes.len());
                    let known = fallbacks
                        .iter()
                        .find(|(start, len, _)| (*start, *len) == key)
                        .map(|(_, _, index)| *index);
                    let index = match known {
                        Some(index) => index,
                        None if fallbacks.len() == MAX_FALLBACK_FACES => return None,
                        None => {
                            let other = cutlass_font::read(bytes).ok()?;
                            fallbacks.push((key.0, key.1, fonts.len()));
                            fonts.push(other);
                            fonts.len() - 1
                        }
                    };
                    // A fallback that cannot draw it either is not used.
                    fonts
                        .get(index)
                        .is_some_and(|font| draws(font, character))
                        .then_some(index)
                };
                // A Han character is written differently in Japan, in
                // Korea and in China, and one face draws one of them: it
                // takes the face of the kana or the Hangul in the same
                // text, when there are any and that face has it.
                let beside = guide
                    .filter(|_| !emoji && props.script() == *b"Hani")
                    .and_then(|guide| source.fallback(guide, false))
                    .and_then(&mut held);
                let found =
                    beside.or_else(|| source.fallback(character, emoji).and_then(&mut held));
                if let Some(index) = found {
                    face = index;
                    real_bold = false;
                    real_italic = false;
                }
            }
        }
        if face != 0 {
            others.push(Other {
                at,
                face,
                bold: real_bold,
                italic: real_italic,
            });
        }
    }
    Faces { fonts, others }
}

#[cfg(test)]
mod tests {
    use super::{FaceSource, MAX_FALLBACK_FACES, NoFaces, StyledFace, Wanted, gather};
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

    const PLAIN: Wanted<'static> = Wanted {
        path: "own.ttf",
        bold: false,
        italic: false,
    };

    fn own() -> cutlass_font::Font<'static> {
        cutlass_font::read(SAMPLE).expect("sample")
    }

    #[test]
    fn a_character_the_face_lacks_is_set_in_a_fallback_that_has_it() {
        // The sample face has A and B; the shaping face has f, e and more.
        let source = Table::new(|_| None, |character| (character != 'Q').then_some(SHAPING));
        let faces = gather(own(), "AfB e", &[], PLAIN, &source);
        assert_eq!(faces.fonts.len(), 2);
        let picked: Vec<usize> = [0, 1, 2, 3, 4].iter().map(|at| faces.pick(*at)).collect();
        assert_eq!(picked, [0, 1, 0, 0, 1]);
        // A character the fallback lacks too stays in the title's face,
        // and one nobody offers a face for does as well.
        let faces = gather(own(), "AZQ", &[], PLAIN, &source);
        assert_eq!((faces.pick(1), faces.pick(2)), (0, 0));
        // With no source, nothing changes.
        let alone = gather(own(), "Af", &[], PLAIN, &NoFaces);
        assert_eq!((alone.fonts.len(), alone.pick(1)), (1, 0));
    }

    #[test]
    fn a_han_character_takes_the_face_of_the_kana_or_hangul_beside_it() {
        const HAN: &[u8] = include_bytes!("../../font/fixtures/han.ttf");
        const KANA: &[u8] = include_bytes!("../../font/fixtures/han-kana.ttf");
        const HANGUL: &[u8] = include_bytes!("../../font/fixtures/han-hangul.ttf");
        // Water is 500 wide in the face of Han alone, 700 in the face
        // with kana and 900 in the one with Hangul. The source offers the
        // first for a Han character, as a table that ranks a Chinese face
        // first does.
        let source = Table::new(
            |_| None,
            |character| match character {
                '\u{3042}' => Some(KANA),
                '\u{AC00}' => Some(HANGUL),
                '\u{6C34}' => Some(HAN),
                _ => None,
            },
        );
        let width = |text: &str| {
            let faces = gather(own(), text, &[], PLAIN, &source);
            let at = text.find('\u{6C34}').expect("water");
            let font = &faces.fonts[faces.pick(at)];
            let glyph = font.glyph_id('\u{6C34}').expect("map").expect("water");
            font.advance(glyph).expect("advance")
        };
        assert_eq!(width("\u{6C34}"), 500);
        assert_eq!(width("A \u{6C34}"), 500);
        // Kana anywhere in the text, before or after it.
        assert_eq!(width("\u{6C34}\u{3042}"), 700);
        assert_eq!(width("\u{3042}A\u{6C34}"), 700);
        assert_eq!(width("\u{AC00}\u{6C34}"), 900);
        // Kana before Hangul when a text has both.
        assert_eq!(width("\u{AC00}\u{6C34}\u{3042}"), 700);
        // A face beside it that lacks the character is passed over.
        let lacking = Table::new(
            |_| None,
            |character| match character {
                '\u{3042}' => Some(SHAPING),
                '\u{6C34}' => Some(HAN),
                _ => None,
            },
        );
        let faces = gather(own(), "\u{3042}\u{6C34}", &[], PLAIN, &lacking);
        let font = &faces.fonts[faces.pick(3)];
        assert_eq!(font.glyph_id('\u{6C34}'), Ok(Some(1)));
        assert_eq!(font.advance(1), Ok(500));
        // The title's own face still comes first.
        let japanese = cutlass_font::read(KANA).expect("kana");
        let faces = gather(japanese, "\u{AC00}\u{6C34}", &[], PLAIN, &source);
        assert_eq!(faces.pick(3), 0);
    }

    #[test]
    fn what_the_face_can_make_does_not_ask_for_a_fallback() {
        let source = Table::new(|_| None, |_| None);
        let shaping = cutlass_font::read(SHAPING).expect("shaping");
        // An accent on its letter, a letter the face draws in two parts, a
        // space it lacks, a joiner and a line end: none is asked for.
        let _faces = gather(
            shaping,
            "e\u{301}\u{C1}\u{2003}\u{200D}\n",
            &[],
            PLAIN,
            &source,
        );
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
        let _faces = gather(own(), "\u{1F600}1\u{FE0F}2", &[], PLAIN, &source);
        assert_eq!(
            *source.asked.borrow(),
            [('\u{1F600}', true), ('1', true), ('2', false)]
        );
        // A face that already draws in color keeps its own emoji glyph.
        let source = Table::new(|_| None, |_| None);
        let color = cutlass_font::read(COLOR).expect("color");
        let has_a = color.glyph_id('A').expect("map").is_some();
        let _faces = gather(color, "A\u{FE0F}", &[], PLAIN, &source);
        assert_eq!(source.asked.borrow().is_empty(), has_a);
    }

    #[test]
    fn a_bold_or_italic_range_takes_the_family_face_that_is() {
        /// A family with a bold face and no italic one.
        struct Family;
        impl FaceSource for Family {
            fn named(&self, _path: &str) -> Option<&[u8]> {
                None
            }

            fn fallback(&self, _character: char, _emoji: bool) -> Option<&[u8]> {
                None
            }

            fn styled(&self, path: &str, bold: bool, _italic: bool) -> Option<StyledFace<'_>> {
                (path == "own.ttf" && bold).then_some(StyledFace {
                    bytes: SHAPING,
                    bold: true,
                    italic: false,
                })
            }
        }
        let style = |bold: Option<bool>, italic: Option<bool>| Run {
            start: 1,
            end: 2,
            fill: None,
            size: 1.0,
            outline: None,
            shadow: None,
            font: None,
            bold,
            italic,
            underline: None,
        };
        // The middle letter is bold: it is set in the bold face, which is
        // bold itself, so it is not thickened.
        let faces = gather(own(), "AAA", &[style(Some(true), None)], PLAIN, &Family);
        assert_eq!((faces.pick(0), faces.pick(1), faces.pick(2)), (0, 1, 0));
        assert_eq!(
            (faces.real(0), faces.real(1)),
            ((false, false), (true, false))
        );
        // Bold and italic: the bold face is taken, and the lean is left
        // to be drawn. Italic alone has no face, and keeps the title's.
        let faces = gather(
            own(),
            "AAA",
            &[style(Some(true), Some(true))],
            PLAIN,
            &Family,
        );
        assert_eq!((faces.pick(1), faces.real(1)), (1, (true, false)));
        let faces = gather(own(), "AAA", &[style(None, Some(true))], PLAIN, &Family);
        assert_eq!((faces.pick(1), faces.real(1)), (0, (false, false)));
        // A bold title is bold throughout, but for a range that is not.
        let bold = Wanted {
            bold: true,
            ..PLAIN
        };
        let faces = gather(own(), "AAA", &[style(Some(false), None)], bold, &Family);
        assert_eq!((faces.pick(0), faces.pick(1), faces.pick(2)), (1, 0, 1));
        assert_eq!(faces.fonts.len(), 2);
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
            bold: None,
            italic: None,
            underline: None,
        }];
        let source = Table::new(|path| (path == "other.ttf").then_some(SHAPING), |_| None);
        let faces = gather(own(), "AfA", &runs, PLAIN, &source);
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
        let faces = gather(own(), &text, &[], PLAIN, &source);
        assert_eq!(faces.fonts.len(), 1 + MAX_FALLBACK_FACES);
        // The ninth letter found no room and stays in the title's face.
        let ninth = text.char_indices().nth(8).expect("ninth").0;
        assert_eq!(faces.pick(ninth), 0);
    }
}

//! What Thai and Lao need before a face's own tables see them.
//!
//! The vowel sara am is one character that is drawn as two: a small
//! circle over the consonant and a vowel after it. A face's tables expect
//! the two, with the circle under any tone mark and not after it, so the
//! character is split and its circle moved back over the marks above the
//! consonant, as every shaper of these scripts does.

use super::buffer::Glyph;

/// Sara am, in Thai or in Lao, which is laid out the same 128 further on.
fn sara_am(code: u32) -> bool {
    code & !0x80 == 0x0E33
}

/// A mark set above its consonant, in either script.
fn above(code: u32) -> bool {
    matches!(code & !0x80, 0x0E31 | 0x0E34..=0x0E37 | 0x0E3B | 0x0E47..=0x0E4E)
}

/// Split each sara am of `glyphs` into its circle and its vowel.
/// `describe` makes the glyph of a character at a place in the text.
pub(super) fn split(glyphs: Vec<Glyph>, describe: impl Fn(char, usize) -> Glyph) -> Vec<Glyph> {
    if !glyphs
        .iter()
        .any(|glyph| sara_am(u32::from(glyph.character)))
    {
        return glyphs;
    }
    let mut out: Vec<Glyph> = Vec::with_capacity(glyphs.len() + 4);
    for glyph in glyphs {
        let code = u32::from(glyph.character);
        let parts = if sara_am(code) {
            // The circle is 26 after sara am, and the vowel one before it.
            char::from_u32(code + 0x1A).zip(char::from_u32(code - 1))
        } else {
            None
        };
        let Some((circle, vowel)) = parts else {
            out.push(glyph);
            continue;
        };
        // The circle goes before the marks above the consonant, and the
        // consonant, its marks, the circle and the vowel are one place in
        // the text.
        let mut start = out.len();
        while start > 0 && above(u32::from(out[start - 1].character)) {
            start -= 1;
        }
        let from = start.saturating_sub(1);
        let cluster = out[from..]
            .iter()
            .map(|earlier| earlier.cluster)
            .min()
            .map_or(glyph.cluster, |earliest| earliest.min(glyph.cluster));
        for earlier in &mut out[from..] {
            earlier.cluster = cluster;
        }
        let mut circle = describe(circle, cluster);
        circle.continues = true;
        // It keeps its place among the marks: it is not sorted with them.
        circle.combining = 0;
        out.insert(start, circle);
        let mut vowel = describe(vowel, cluster);
        vowel.continues = true;
        out.push(vowel);
    }
    out
}

/// What is done to a mark, or to the consonant under it, in a face that
/// has no tables of its own for Thai.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shift {
    None,
    /// A tone mark set lower, straight on the consonant.
    Down,
    /// A mark set to the left, clear of a consonant's tall stroke.
    Left,
    /// Both.
    DownLeft,
    /// A vowel below set lower, under a consonant's tail.
    Below,
    /// The consonant drawn without the part a vowel below would hit.
    Trim,
}

/// The glyphs a face keeps in its private use area for `code` shifted:
/// as the Thai fonts made for Windows number them, and as those made for
/// the Mac do. A face is asked for the first, then the second.
fn shifted(code: u32, shift: Shift) -> Option<[u32; 2]> {
    // The Mac's tone marks come in threes: left, down, and both.
    let tone = |first: u32| first + 3 * (code - 0x0E48);
    Some(match (shift, code) {
        (Shift::Down, 0x0E48..=0x0E4C) => [code - 0x0E48 + 0xF70A, tone(0xF88B)],
        (Shift::Below, 0x0E38..=0x0E3A) => [code - 0x0E38 + 0xF718, code - 0x0E38 + 0xF89B],
        (Shift::DownLeft, 0x0E48..=0x0E4C) => [code - 0x0E48 + 0xF705, tone(0xF88C)],
        (Shift::Left, 0x0E48..=0x0E4C) => [code - 0x0E48 + 0xF713, tone(0xF88A)],
        (Shift::Left, 0x0E34..=0x0E37) => [code - 0x0E34 + 0xF701, code - 0x0E34 + 0xF885],
        (Shift::Left, 0x0E31) => [0xF710, 0xF884],
        (Shift::Left, 0x0E4D) => [0xF711, 0xF899],
        (Shift::Left, 0x0E47) => [0xF712, 0xF889],
        (Shift::Trim, 0x0E0D) => [0xF70F, 0xF89A],
        (Shift::Trim, 0x0E10) => [0xF700, 0xF89E],
        _ => return None,
    })
}

/// A consonant by its shape: plain, with a tall stroke on the right, with
/// a part below that can be left off, or with a tail below.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    Plain,
    Tall,
    Trimmed,
    Tailed,
}

fn shape(code: u32) -> Option<Shape> {
    Some(match code {
        0x0E1B | 0x0E1D | 0x0E1F => Shape::Tall,
        0x0E0D | 0x0E10 => Shape::Trimmed,
        0x0E0E | 0x0E0F => Shape::Tailed,
        0x0E01..=0x0E2E => Shape::Plain,
        _ => return None,
    })
}

/// A mark by where it goes: a vowel above, a vowel below, or a tone mark.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mark {
    Above,
    Below,
    Tone,
}

fn mark(code: u32) -> Option<Mark> {
    Some(match code {
        0x0E31 | 0x0E34..=0x0E37 | 0x0E47 | 0x0E4D..=0x0E4E => Mark::Above,
        0x0E38..=0x0E3A => Mark::Below,
        0x0E48..=0x0E4C => Mark::Tone,
        _ => return None,
    })
}

/// For a Thai face with no tables of its own for the script: give each
/// mark the shifted glyph the face keeps for where it lands, by the
/// consonant it is on and the marks already there, as such faces were
/// made to be drawn. `has` says whether the face has a character.
pub(super) fn shift_marks(glyphs: &mut [Glyph], has: impl Fn(char) -> bool) {
    // What is above the consonant so far: nothing (0), nothing on a tall
    // consonant (1), one mark set left (2), a mark at full height (3).
    // What is below: nothing (0), nothing under a consonant whose lower
    // part can go (1), something, or a tail (2).
    // Text starts as it goes on after anything that is no consonant.
    let (mut above, mut below, mut base) = (3_u8, 0_u8, None::<usize>);
    for index in 0..glyphs.len() {
        let code = u32::from(glyphs[index].character);
        let Some(kind) = mark(code) else {
            let shape = shape(code);
            // After anything but a consonant, marks above are not moved.
            above = match shape {
                Some(Shape::Tall) => 1,
                Some(_) => 0,
                None => 3,
            };
            below = match shape {
                Some(Shape::Trimmed) => 1,
                Some(Shape::Tailed) => 2,
                _ => 0,
            };
            base = shape.map(|_| index);
            continue;
        };
        let (shift, next) = match (above, kind) {
            (0, Mark::Above) => (Shift::None, 3),
            (0, Mark::Tone) => (Shift::Down, 0),
            (1, Mark::Above) => (Shift::Left, 2),
            (1, Mark::Tone) => (Shift::DownLeft, 1),
            (2, Mark::Above) => (Shift::None, 3),
            (2, Mark::Tone) => (Shift::Left, 3),
            (state, _) => (Shift::None, state),
        };
        above = next;
        let (under, next) = match (below, kind) {
            (0, Mark::Below) => (Shift::None, 2),
            (1, Mark::Below) => (Shift::Trim, 2),
            (2, Mark::Below) => (Shift::Below, 2),
            (state, _) => (Shift::None, state),
        };
        below = next;
        // The consonant loses its lower part; the mark takes its shift.
        let (target, shift) = if under == Shift::Trim {
            (base, under)
        } else if under != Shift::None {
            (Some(index), under)
        } else {
            (Some(index), shift)
        };
        if let Some(target) = target
            && let Some(numbered) = shifted(u32::from(glyphs[target].character), shift)
            && let Some(other) = numbered
                .into_iter()
                .filter_map(char::from_u32)
                .find(|other| has(*other))
        {
            glyphs[target].character = other;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::split;
    use crate::shaper::buffer::Glyph;

    fn parts(text: &str) -> Vec<(u32, usize)> {
        let glyphs: Vec<Glyph> = text
            .char_indices()
            .map(|(at, character)| Glyph::new(character, at))
            .collect();
        split(glyphs, Glyph::new)
            .iter()
            .map(|glyph| (u32::from(glyph.character), glyph.cluster))
            .collect()
    }

    #[test]
    fn sara_am_is_its_circle_and_its_vowel() {
        // Consonant, sara am: consonant, circle, vowel, in one place.
        assert_eq!(
            parts("\u{E19}\u{E33}"),
            [(0xE19, 0), (0xE4D, 0), (0xE32, 0)]
        );
        // With a tone mark: the circle goes under the mark, before it.
        assert_eq!(
            parts("\u{E19}\u{E49}\u{E33}"),
            [(0xE19, 0), (0xE4D, 0), (0xE49, 0), (0xE32, 0)]
        );
        // Lao the same way.
        assert_eq!(
            parts("\u{E99}\u{EC9}\u{EB3}"),
            [(0xE99, 0), (0xECD, 0), (0xEC9, 0), (0xEB2, 0)]
        );
        // The letters before keep their own places.
        assert_eq!(
            parts("\u{E01}\u{E19}\u{E33}"),
            [(0xE01, 0), (0xE19, 3), (0xE4D, 3), (0xE32, 3)]
        );
    }

    /// The characters after the marks are shifted, in a face that has
    /// every shifted glyph.
    fn shifts(text: &str) -> Vec<u32> {
        let mut glyphs: Vec<Glyph> = text
            .char_indices()
            .map(|(at, character)| Glyph::new(character, at))
            .collect();
        super::shift_marks(&mut glyphs, |_| true);
        glyphs
            .iter()
            .map(|glyph| u32::from(glyph.character))
            .collect()
    }

    #[test]
    fn a_face_without_thai_tables_gets_its_shifted_marks() {
        // These are what HarfBuzz 14.2 picks in such a face.
        // A tone mark straight on a consonant is the low one.
        assert_eq!(shifts("\u{E01}\u{E48}"), [0xE01, 0xF70A]);
        // Over a vowel above, it is left at full height.
        assert_eq!(shifts("\u{E01}\u{E34}\u{E48}"), [0xE01, 0xE34, 0xE48]);
        // On a consonant with a tall stroke: the vowel goes left, and the
        // tone mark over it goes left; a tone mark alone goes low and left.
        assert_eq!(shifts("\u{E1B}\u{E34}\u{E48}"), [0xE1B, 0xF701, 0xF713]);
        assert_eq!(shifts("\u{E1B}\u{E49}"), [0xE1B, 0xF706]);
        assert_eq!(shifts("\u{E1B}\u{E31}\u{E49}"), [0xE1B, 0xF710, 0xF714]);
        // A vowel below a consonant with a tail is set lower, and one
        // below a consonant whose lower part can go takes that part off.
        assert_eq!(shifts("\u{E0E}\u{E38}"), [0xE0E, 0xF718]);
        assert_eq!(shifts("\u{E0D}\u{E39}"), [0xF70F, 0xE39]);
        assert_eq!(shifts("\u{E10}\u{E38}"), [0xF700, 0xE38]);
        // After a vowel that follows the consonant nothing is moved.
        assert_eq!(shifts("\u{E01}\u{E32}\u{E48}"), [0xE01, 0xE32, 0xE48]);
        // Nor at the start of the text, with no consonant at all.
        assert_eq!(shifts("\u{E48}"), [0xE48]);
        // A face that lacks the shifted glyph keeps the mark as typed.
        let mut glyphs = vec![Glyph::new('\u{E01}', 0), Glyph::new('\u{E48}', 3)];
        super::shift_marks(&mut glyphs, |_| false);
        assert_eq!(u32::from(glyphs[1].character), 0xE48);
    }

    #[test]
    fn a_face_numbered_as_the_mac_did_gets_its_own_shifted_marks() {
        // A face with the Mac's numbers and not Windows's. HarfBuzz 14.2
        // picks the same in the fixture face numbered this way, over
        // every consonant shape and mark.
        let mac = |text: &str| {
            let mut glyphs: Vec<Glyph> = text
                .char_indices()
                .map(|(at, character)| Glyph::new(character, at))
                .collect();
            super::shift_marks(&mut glyphs, |character| {
                !(0xF700..=0xF71A).contains(&u32::from(character))
            });
            glyphs
                .iter()
                .map(|glyph| u32::from(glyph.character))
                .collect::<Vec<u32>>()
        };
        assert_eq!(mac("\u{E01}\u{E48}"), [0xE01, 0xF88B]);
        assert_eq!(mac("\u{E01}\u{E4C}"), [0xE01, 0xF897]);
        assert_eq!(mac("\u{E1B}\u{E34}\u{E48}"), [0xE1B, 0xF885, 0xF88A]);
        assert_eq!(mac("\u{E1B}\u{E49}"), [0xE1B, 0xF88F]);
        assert_eq!(mac("\u{E1B}\u{E31}\u{E4B}"), [0xE1B, 0xF884, 0xF893]);
        assert_eq!(mac("\u{E1B}\u{E47}"), [0xE1B, 0xF889]);
        assert_eq!(mac("\u{E1B}\u{E4D}"), [0xE1B, 0xF899]);
        assert_eq!(mac("\u{E0E}\u{E3A}"), [0xE0E, 0xF89D]);
        assert_eq!(mac("\u{E0D}\u{E39}"), [0xF89A, 0xE39]);
        assert_eq!(mac("\u{E10}\u{E38}"), [0xF89E, 0xE38]);
        // Windows's numbers come first in a face that has both.
        assert_eq!(shifts("\u{E1B}\u{E49}"), [0xE1B, 0xF706]);
    }

    #[test]
    fn other_text_is_left_as_it_is() {
        assert_eq!(
            parts("\u{E01}\u{E32}a"),
            [(0xE01, 0), (0xE32, 3), (0x61, 6)]
        );
        // A sara am with nothing before it is still two.
        assert_eq!(parts("\u{E33}"), [(0xE4D, 0), (0xE32, 0)]);
        assert!(parts("").is_empty());
        let started = std::time::Instant::now();
        let long = "\u{E19}\u{E49}\u{E33}".repeat(50_000);
        assert_eq!(parts(&long).len(), 200_000);
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }
}
